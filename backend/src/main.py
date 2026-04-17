"""Amphoreus FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.src.auth.acl import Acl
from backend.src.auth.cf_access import CfAccessVerifier
from backend.src.auth.middleware import (
    AuditLogMiddleware,
    CfAccessAuthMiddleware,
    require_client_from_path,
)
from backend.src.core.config import get_settings
from backend.src.db.local import initialize_db, mark_stale_runs_failed

logger = logging.getLogger("amphoreus")

# Single shared instances of the auth components. Instantiated at import time
# so the middleware (which is registered before lifespan runs) and the app
# state (populated during lifespan) reference the same objects.
_settings_singleton = get_settings()
CF_VERIFIER = CfAccessVerifier(
    team_domain=_settings_singleton.cf_access_team_domain,
    audience=_settings_singleton.cf_access_aud,
)
ACL = Acl(path=_settings_singleton.acl_path)


def _startup_catchup() -> None:
    """Run one-time catch-up steps on backend boot.

    Currently a no-op. The previous implementation ran
    ``feedback_distiller.distill_directives`` + ``backfill_active_directives``
    on every startup, which turned client feedback into a hand-authored
    rule list and stamped it onto every observation as ``active_directives``.
    That pattern — distill rules, tag observations with rule IDs, feed the
    tagged observations back to the writer — is a closed loop of
    prescriptive injection. Removed to comply with the Bitter Lesson
    filter: the writer reads raw feedback files directly from
    ``memory/{company}/feedback/`` and decides what matters.
    """
    return


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Amphoreus backend...")
    initialize_db()
    logger.info("SQLite initialized at %s", settings.sqlite_path)

    # Install LLM usage instrumentation — monkey-patches the Anthropic SDK
    # so every messages.create call is recorded to usage_events with the
    # authenticated user attribution pulled from the ContextVar set by the
    # CF Access middleware. Idempotent; safe to call on every startup.
    try:
        from backend.src.usage import install_instrumentation
        install_instrumentation()
    except Exception:
        logger.exception("Failed to install usage instrumentation (non-fatal).")
    stale = mark_stale_runs_failed()
    if stale:
        logger.info("Marked %d stale 'running' job(s) as failed (server restart).", stale)

    # --- Stage 2 auth: Cloudflare Access verifier + ACL ---
    # Single shared instances live at module level (CF_VERIFIER, ACL) so the
    # middleware (registered below) and the FastAPI deps (which read
    # ``request.app.state.acl``) see the same objects.
    app.state.cf_verifier = CF_VERIFIER
    app.state.acl = ACL
    if CF_VERIFIER.enabled:
        logger.info(
            "CF Access auth ENABLED (team=%s, aud=%s...)",
            settings.cf_access_team_domain,
            settings.cf_access_aud[:8],
        )
        # Eager-fetch JWKS at startup so misconfig fails loudly instead of on first request.
        try:
            CF_VERIFIER._get_jwk_client()  # noqa: SLF001
        except Exception:
            logger.exception("CF Access JWKS fetch failed at startup")
    else:
        logger.warning(
            "CF Access auth DISABLED — CF_ACCESS_TEAM_DOMAIN or CF_ACCESS_AUD is empty. "
            "Every request will be treated as an anonymous dev admin. NEVER run this way in prod."
        )

    try:
        _startup_catchup()
    except Exception:
        logger.exception("Startup catch-up failed (non-fatal).")

    # Start Ordinal sync loop (runs every hour in a background thread).
    # Feeds RuanMei with Ordinal LinkedIn analytics only (no Supabase writes).
    # ONLY on localhost — sync writes to local memory/, which the user then
    # pushes to Fly via push-to-fly.sh. This avoids paying for duplicate
    # sync loops on both local and Fly.
    if not CF_VERIFIER.enabled:
        try:
            from backend.src.services.ordinal_sync import start_sync_loop, start_fast_sync_loop, stop_sync_loop
            start_sync_loop()           # hourly: full pipeline
            start_fast_sync_loop()      # 15 min: engagement snapshots for posts <72h old
            logger.info("Ordinal sync loops started (slow=3600s, fast=900s).")
        except Exception:
            logger.exception("Failed to start Ordinal sync loop (non-fatal).")
    else:
        logger.info("Ordinal sync loops SKIPPED (Fly — run sync locally and push via push-to-fly.sh).")

    yield

    # Graceful shutdown.
    try:
        stop_sync_loop()
    except Exception:
        pass
    logger.info("Shutting down Amphoreus backend.")


settings = _settings_singleton

# Global FastAPI dependency: enforces per-client ACL on every route that has a
# ``{company}`` path param. Routes that carry ``company`` in the request body
# call ``require_client_body`` explicitly from within the handler.
app = FastAPI(
    title="Amphoreus",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_client_from_path)],
)

# Middleware ordering matters: the outermost wrapper runs last on the way in,
# first on the way out. Starlette applies them in REVERSE order of add_middleware
# calls — so the LAST add is the OUTERMOST wrapper. We want:
#   outermost: CORS (so 401 responses still get CORS headers)
#   middle:    CfAccessAuth (gates everything)
#   innermost: AuditLog (sees the final status code, knows which user)
# Therefore add order is: AuditLog → CfAccessAuth → CORS.
app.add_middleware(AuditLogMiddleware, log_path=settings.audit_log_path)
app.add_middleware(CfAccessAuthMiddleware, verifier=CF_VERIFIER, acl=ACL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register routers ---
from backend.src.api.routers import (
    auth,
    briefings,
    clients,
    cs,
    deploy,
    desktop,
    ghostwriter,
    images,
    interview,
    learning,
    posts,
    report,
    research,
    strategy,
    transcripts,
    usage,
)

app.include_router(auth.router)
app.include_router(clients.router)
app.include_router(deploy.router)
app.include_router(desktop.router)
app.include_router(ghostwriter.router)
app.include_router(briefings.router)
app.include_router(interview.router)
app.include_router(strategy.router)
app.include_router(posts.router)
app.include_router(images.router)
app.include_router(research.router)
app.include_router(cs.router)
app.include_router(learning.router)
app.include_router(report.router)
app.include_router(transcripts.router)
app.include_router(usage.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "amphoreus"}


@app.get("/api/me")
async def me(request: Request):
    """Return the current user's identity + ACL-scoped client list.

    Frontend calls this on page load to populate the auth context and decide
    which client tabs to render in the sidebar. Returns ``allowed_clients: "*"``
    for admins, or an explicit list of slugs for scoped users.
    """
    user = getattr(request.state, "user", None)
    if user is None:
        # Should never happen — middleware sets this on every non-exempt request.
        return {"email": "", "is_admin": False, "allowed_clients": []}
    is_admin = bool(getattr(request.state, "user_is_admin", False))
    # Admins always see everything, including the dev-bypass user which isn't
    # in the ACL file. Non-admins get their explicit allowlist from the ACL.
    allowed: object = "*" if is_admin else ACL.allowed_clients(user.email)
    return {
        "email": user.email,
        "is_admin": is_admin,
        "allowed_clients": allowed,  # "*" (admin) or list[str]
        "auth_enabled": CF_VERIFIER.enabled,
    }
