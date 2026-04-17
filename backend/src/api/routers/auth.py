"""Auth API — permissions, profile."""

import logging

from fastapi import APIRouter

from backend.src.api.middleware.auth import CurrentUser

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/permissions")
async def get_permissions(user: CurrentUser):
    """Return current user's permissions and role."""
    return {
        "user_id": user.user_id,
        "email": user.email,
        "role": user.role,
        "company_id": user.company_id,
    }


@router.get("/profile")
async def get_profile(user: CurrentUser):
    """Return current user's profile from Supabase."""
    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()
        result = (
            sb.table("users")
            .select("id, first_name, last_name, email, company_id, role, title, linkedin_url")
            .eq("id", user.user_id)
            .limit(1)
            .execute()
        )
        if result.data:
            return {"profile": result.data[0]}
        return {"profile": {"user_id": user.user_id, "email": user.email, "role": user.role}}
    except Exception as e:
        logger.warning("Failed to fetch profile: %s", e)
        return {"profile": {"user_id": user.user_id, "email": user.email, "role": user.role}}
