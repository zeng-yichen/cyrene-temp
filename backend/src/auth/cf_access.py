"""Cloudflare Access JWT verification.

Cloudflare Access signs a short-lived RS256 JWT for every authenticated request
and attaches it in two places:
  1. ``Cf-Access-Jwt-Assertion`` header on origin requests (primary).
  2. ``CF_Authorization`` cookie on the browser (fallback — useful when the
     request reaches us via a proxy that didn't forward the header).

We verify signature + claims against the team's JWKS endpoint:
  https://<team>.cloudflareaccess.com/cdn-cgi/access/certs

Keys rotate, so we re-fetch hourly. The verifier is instantiated once at app
startup (``main.py``) and injected into the middleware.

If either ``CF_ACCESS_TEAM_DOMAIN`` or ``CF_ACCESS_AUD`` is empty, verification
is disabled and every request is treated as an anonymous local-dev user.
This is ONLY safe in local docker-compose / ``uv run uvicorn`` contexts.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import jwt
import requests
from jwt import PyJWKClient

logger = logging.getLogger("amphoreus.auth")

JWKS_TTL_SECONDS = 3600  # refresh keys hourly
JWKS_FETCH_TIMEOUT = 10.0
ALGORITHMS = ["RS256"]


@dataclass(frozen=True)
class AuthedUser:
    """An authenticated Cloudflare Access identity."""

    email: str
    sub: str  # CF Access user identifier (sub claim)

    @property
    def is_anonymous(self) -> bool:
        return not self.email


ANONYMOUS_DEV_USER = AuthedUser(email="dev@localhost", sub="local-dev")


class CfAccessVerifier:
    """Thread-safe JWT verifier for Cloudflare Access."""

    def __init__(self, team_domain: str, audience: str):
        self._team_domain = team_domain.strip().rstrip("/")
        self._audience = audience.strip()
        self._issuer = f"https://{self._team_domain}" if self._team_domain else ""
        self._certs_url = f"{self._issuer}/cdn-cgi/access/certs" if self._issuer else ""
        self._jwk_client: PyJWKClient | None = None
        self._jwk_fetched_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._team_domain and self._audience)

    def _get_jwk_client(self) -> PyJWKClient:
        """Return a PyJWKClient, refreshing it at most once per JWKS_TTL_SECONDS.

        PyJWKClient has its own internal cache via ``lifespan``, but we wrap it
        so we can force a re-fetch when keys rotate mid-session without
        restarting the process.
        """
        now = time.time()
        with self._lock:
            if self._jwk_client is None or (now - self._jwk_fetched_at) > JWKS_TTL_SECONDS:
                logger.info("Fetching Cloudflare Access JWKS from %s", self._certs_url)
                # PyJWKClient lazily fetches on first .get_signing_key_from_jwt call,
                # but we want eager failure so startup loudly breaks if the URL is wrong.
                try:
                    resp = requests.get(self._certs_url, timeout=JWKS_FETCH_TIMEOUT)
                    resp.raise_for_status()
                    _ = resp.json()  # validate JSON shape
                except Exception:
                    logger.exception("Failed to fetch JWKS from %s", self._certs_url)
                    raise
                self._jwk_client = PyJWKClient(self._certs_url, lifespan=JWKS_TTL_SECONDS)
                self._jwk_fetched_at = now
            return self._jwk_client

    def verify(self, token: str) -> AuthedUser:
        """Verify a JWT and return the authenticated user.

        Raises ``jwt.InvalidTokenError`` (or subclass) on any failure.
        """
        if not self.enabled:
            raise RuntimeError("CfAccessVerifier called with empty team_domain/aud")
        if not token:
            raise jwt.InvalidTokenError("empty token")

        client = self._get_jwk_client()
        signing_key = client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALGORITHMS,
            audience=self._audience,
            issuer=self._issuer,
            options={
                "require": ["exp", "iat", "iss", "aud", "sub"],
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )

        email = (payload.get("email") or payload.get("identity_nonce") or "").strip().lower()
        sub = str(payload.get("sub") or "")
        if not email:
            # CF Access always includes email for human identities. Service tokens don't;
            # we're only supporting human auth for Stage 2, so reject.
            raise jwt.InvalidTokenError("no email claim")

        return AuthedUser(email=email, sub=sub)
