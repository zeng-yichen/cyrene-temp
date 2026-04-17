"""ASGI middleware + FastAPI dependencies wiring CF Access auth to the app.

Flow on every incoming HTTP request:

1. ``CfAccessAuthMiddleware`` — pure ASGI middleware.
   - Exempts: ``/health``, ``/docs``, ``/openapi.json``, ``/redoc``, ``OPTIONS`` preflights.
   - Extracts JWT from (in order):
       a. ``Cf-Access-Jwt-Assertion`` header
       b. ``CF_Authorization`` cookie
       c. ``Authorization: Bearer <jwt>`` header (explicit clients)
   - Verifies signature + claims with ``CfAccessVerifier``.
   - Checks that the email is known to the ACL (admin or scoped).
   - On success: sets ``request.state.user = AuthedUser(...)``.
   - On failure: returns 401 JSON ``{"error": "unauthorized", "detail": "<why>"}``.

2. ``AuditLogMiddleware`` — pure ASGI middleware.
   - Only fires for write verbs (POST/PATCH/PUT/DELETE).
   - Appends a single JSONL line to ``/data/audit.log``:
       {ts, email, method, path, client, status}
   - Failure to log is non-fatal (never blocks the response).

3. ``require_client_from_path`` — FastAPI dependency.
   - Inspects ``request.path_params.get("company")``.
   - If present and user is not admin, checks the ACL.
   - Applied globally via ``FastAPI(dependencies=[...])`` so every route with a
     ``{company}`` path param is guarded automatically.
   - Routes that take ``company`` in the body (e.g. ghostwriter/generate) must
     call ``require_client_body(...)`` explicitly.

Local-dev bypass: if the verifier is disabled (empty ``CF_ACCESS_TEAM_DOMAIN``
or ``CF_ACCESS_AUD``), the middleware stamps every request with
``ANONYMOUS_DEV_USER`` and skips verification. All dev users are treated as
admins for ACL purposes.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Awaitable, Callable

import jwt
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from backend.src.auth.acl import Acl
from backend.src.auth.cf_access import ANONYMOUS_DEV_USER, AuthedUser, CfAccessVerifier
from backend.src.usage.context import set_request_attribution

logger = logging.getLogger("amphoreus.auth")

# Routes that never require auth. Must be evaluated as exact matches or prefix matches.
EXEMPT_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}
EXEMPT_PREFIXES = (
    "/docs/",
    "/redoc/",
)

WRITE_METHODS = frozenset({"POST", "PATCH", "PUT", "DELETE"})


def _is_exempt(path: str) -> bool:
    if path in EXEMPT_PATHS:
        return True
    return any(path.startswith(p) for p in EXEMPT_PREFIXES)


def _extract_token(request: Request) -> str:
    """Pull a JWT out of header/cookie/Authorization, in that order."""
    tok = request.headers.get("cf-access-jwt-assertion")
    if tok:
        return tok.strip()
    tok = request.cookies.get("CF_Authorization")
    if tok:
        return tok.strip()
    authz = request.headers.get("authorization", "")
    if authz.lower().startswith("bearer "):
        return authz[7:].strip()
    return ""


def _company_from_path(request: Request) -> str | None:
    """Best-effort extraction of the client slug from the request URL.

    BaseHTTPMiddleware runs BEFORE route matching, so ``request.path_params``
    is empty here — we can't rely on the ``{company}`` FastAPI path param.
    Instead we peel it off the raw path ourselves.

    Amphoreus convention: the client slug is always the 3rd path segment,
    right after ``/api/<router>/``. Examples::

        /api/ghostwriter/example-client            → "example-client"
        /api/ghostwriter/example-client/generate   → "example-client"
        /api/transcripts/example-client/upload          → "example-client"
        /api/clients                             → None
        /health                                  → None

    Returns ``None`` if the path doesn't match the pattern. This is a
    best-effort hint for the usage ContextVar — callers that need a
    guaranteed slug should read ``request.path_params`` inside the
    handler.
    """
    parts = [p for p in request.url.path.split("/") if p]
    if len(parts) >= 3 and parts[0] == "api":
        candidate = parts[2]
        # Skip obvious non-slug segments (IDs, verbs, etc.). A real slug
        # is kebab/alphanum and not a pure integer.
        if candidate and not candidate.isdigit():
            return candidate
    return None


class CfAccessAuthMiddleware(BaseHTTPMiddleware):
    """Verifies every request carries a valid CF Access JWT."""

    def __init__(self, app: ASGIApp, verifier: CfAccessVerifier, acl: Acl):
        super().__init__(app)
        self._verifier = verifier
        self._acl = acl

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.method == "OPTIONS" or _is_exempt(request.url.path):
            return await call_next(request)

        # Local-dev bypass — verifier disabled means no CF in front.
        if not self._verifier.enabled:
            request.state.user = ANONYMOUS_DEV_USER
            request.state.user_is_admin = True
            with set_request_attribution(
                ANONYMOUS_DEV_USER.email,
                _company_from_path(request),
            ):
                return await call_next(request)

        token = _extract_token(request)
        if not token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "unauthorized", "detail": "missing CF Access JWT"},
            )

        try:
            user = self._verifier.verify(token)
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "unauthorized", "detail": "token expired"},
            )
        except jwt.InvalidTokenError as e:
            logger.warning("JWT verification failed: %s", e)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "unauthorized", "detail": f"invalid token: {e}"},
            )
        except Exception:
            logger.exception("Unexpected auth failure")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "auth_error", "detail": "internal"},
            )

        # Ban check — takes priority over ACL.
        if self._acl.is_banned(user.email):
            logger.warning("Banned user %s attempted access", user.email)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "banned",
                    "detail": (
                        f"Your account ({user.email}) has been temporarily suspended. "
                        "Contact your admin to restore access."
                    ),
                },
            )

        # ACL gate — user must be known (admin or scoped) to proceed.
        if not self._acl.is_known(user.email):
            logger.warning("Authenticated user %s not in ACL file", user.email)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "forbidden",
                    "detail": (
                        f"{user.email} is allow-listed in Cloudflare Access but not in the "
                        "server ACL file. Ask an admin to add you."
                    ),
                },
            )

        request.state.user = user
        request.state.user_is_admin = self._acl.is_admin(user.email)
        with set_request_attribution(user.email, _company_from_path(request)):
            return await call_next(request)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Append JSONL audit events for write verbs to a volume-resident file."""

    def __init__(self, app: ASGIApp, log_path: str):
        super().__init__(app)
        self._path = log_path

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        if request.method not in WRITE_METHODS:
            return response
        if _is_exempt(request.url.path):
            return response
        try:
            user: AuthedUser | None = getattr(request.state, "user", None)
            email = user.email if user else ""
            entry = {
                "ts": time.time(),
                "email": email,
                "method": request.method,
                "path": request.url.path,
                "client": request.path_params.get("company"),
                "status": response.status_code,
            }
            # Best-effort append; never block the response.
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception:
            logger.exception("Audit log write failed (non-fatal)")
        return response


# ---------------------------------------------------------------- FastAPI deps


def require_client_from_path(request: Request) -> None:
    """Global dependency: if the route has a ``{company}`` path param, enforce ACL.

    Admins pass through. Non-admins get 403 if the path's ``company`` slug isn't
    in their allowlist. Routes without a ``company`` path param are no-ops here.
    """
    user: AuthedUser | None = getattr(request.state, "user", None)
    if user is None:
        # Middleware already handled unauth; this branch only hits during dev bypass
        # where request.state.user is always set. Defensive fallthrough.
        return
    if getattr(request.state, "user_is_admin", False):
        return
    company = request.path_params.get("company")
    if not company:
        return
    acl: Acl = request.app.state.acl
    decision = acl.check(user.email, company)
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to client '{company}' ({decision.reason})",
        )


def require_client_body(request: Request, client_slug: str) -> None:
    """Explicit ACL check for routes that carry ``company`` in the request body.

    Call from inside the handler after parsing the request model, e.g.::

        @router.post("/generate")
        async def generate(req: GenerateRequest, request: Request):
            require_client_body(request, req.company)
            ...
    """
    user: AuthedUser | None = getattr(request.state, "user", None)
    if user is None:
        return
    if getattr(request.state, "user_is_admin", False):
        return
    acl: Acl = request.app.state.acl
    decision = acl.check(user.email, client_slug)
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to client '{client_slug}' ({decision.reason})",
        )
