"""JWT authentication middleware using Supabase tokens."""

import logging
from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException, status

from backend.src.core.config import get_settings
from backend.src.models.user import AuthUser

logger = logging.getLogger(__name__)


async def get_current_user(authorization: Annotated[str | None, Header()] = None) -> AuthUser:
    """Extract and validate the JWT from the Authorization header.

    Supabase JWTs contain user_id, email, and role in the payload.
    We validate the signature against the JWT secret from settings.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )

    token = authorization.removeprefix("Bearer ").strip()
    settings = get_settings()

    if not settings.jwt_secret:
        logger.warning("JWT_SECRET not configured — accepting token without verification")
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
        except jwt.DecodeError:
            raise HTTPException(status_code=401, detail="Invalid token")
    else:
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub") or payload.get("user_id", "")
    email = payload.get("email", "")
    role = payload.get("role", "external")
    company_id = payload.get("company_id")

    if role not in ("internal", "external", "unauthorized"):
        role = "external"

    if role == "unauthorized":
        raise HTTPException(status_code=403, detail="Unauthorized user")

    return AuthUser(user_id=user_id, email=email, role=role, company_id=company_id)


CurrentUser = Annotated[AuthUser, Depends(get_current_user)]


def require_internal(user: CurrentUser) -> AuthUser:
    """Dependency that requires internal role."""
    if user.role != "internal":
        raise HTTPException(status_code=403, detail="Internal access required")
    return user


InternalUser = Annotated[AuthUser, Depends(require_internal)]
