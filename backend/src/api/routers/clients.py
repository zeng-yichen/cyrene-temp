"""Clients router — canonical list of Ordinal-backed client slugs."""

from fastapi import APIRouter, Request

from backend.src.db import vortex

router = APIRouter(prefix="/api/clients", tags=["clients"])


@router.get("")
async def list_clients(request: Request):
    """Return deduplicated provider_org_slug values from ordinal_auth_rows.csv.

    Filtered by the caller's ACL: admins see everything, scoped users see only
    their allowlisted slugs. The ACL object lives on ``request.app.state.acl``
    and is populated during app startup.
    """
    rows = vortex.list_ordinal_companies()
    seen: set[str] = set()
    slugs: list[str] = []
    for row in rows:
        slug = (row.get("provider_org_slug") or "").strip()
        if slug and slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    slugs.sort()

    # Filter by ACL. Middleware has already populated request.state.user and
    # request.state.user_is_admin; admins bypass the filter.
    user = getattr(request.state, "user", None)
    is_admin = bool(getattr(request.state, "user_is_admin", False))
    if user is not None and not is_admin:
        acl = request.app.state.acl
        slugs = acl.filter_clients(user.email, slugs)

    return {"clients": [{"slug": s} for s in slugs]}
