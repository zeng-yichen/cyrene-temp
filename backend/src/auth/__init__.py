"""Cloudflare Access auth + per-client ACL (Stage 2).

See:
- ``cf_access``: JWT verification against Cloudflare's JWKS.
- ``acl``: file-backed per-email client allowlist on the persistent volume.
- ``middleware``: ASGI middleware + FastAPI dependencies that wire the two together.
"""

from backend.src.auth.acl import Acl, AclDecision
from backend.src.auth.cf_access import AuthedUser, CfAccessVerifier

__all__ = ["Acl", "AclDecision", "AuthedUser", "CfAccessVerifier"]
