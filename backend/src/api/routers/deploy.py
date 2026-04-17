"""Deploy admin router — /api/deploy.

Localhost-only endpoints for pushing code and data to Fly.
All endpoints gate on auth_enabled=False (local dev mode) so they
can never be triggered from production.
"""

import logging
import subprocess
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/deploy", tags=["deploy"])

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Track running deployments so we don't overlap
_active_lock = threading.Lock()
_active: dict[str, dict] = {}  # key → {"status": ..., "log": ...}


def _require_local(request: Request) -> None:
    """Only allow deploy actions from local dev (no CF Access)."""
    cf_verifier = getattr(request.app.state, "cf_verifier", None)
    if cf_verifier and cf_verifier.enabled:
        raise HTTPException(
            status_code=403,
            detail="Deploy endpoints are disabled in production.",
        )


class DeployRequest(BaseModel):
    target: str = "both"  # "backend", "frontend", "both"


class PushDataRequest(BaseModel):
    client: str | None = None  # None = push all


@router.post("/code")
async def deploy_code(req: DeployRequest, request: Request):
    """Deploy code to Fly (backend, frontend, or both)."""
    _require_local(request)

    if req.target not in ("backend", "frontend", "both"):
        raise HTTPException(status_code=400, detail="target must be backend, frontend, or both")

    key = f"code-{req.target}"
    with _active_lock:
        if key in _active and _active[key]["status"] == "running":
            return {"status": "already_running", "message": "A deploy is already in progress."}
        _active[key] = {"status": "running", "log": ""}

    def _run():
        script = _PROJECT_ROOT / "deploy.sh"
        try:
            result = subprocess.run(
                [str(script), req.target],
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=600,
            )
            with _active_lock:
                _active[key] = {
                    "status": "completed" if result.returncode == 0 else "failed",
                    "log": (result.stdout or "") + (result.stderr or ""),
                    "returncode": result.returncode,
                }
            if result.returncode == 0:
                logger.info("[deploy] code deploy (%s) succeeded", req.target)
            else:
                logger.warning("[deploy] code deploy (%s) failed: %s", req.target, result.stderr[:300])
        except subprocess.TimeoutExpired:
            with _active_lock:
                _active[key] = {"status": "failed", "log": "Timed out after 600s"}
            logger.warning("[deploy] code deploy (%s) timed out", req.target)
        except Exception as e:
            with _active_lock:
                _active[key] = {"status": "failed", "log": str(e)}
            logger.exception("[deploy] code deploy (%s) error", req.target)

    threading.Thread(target=_run, daemon=True, name=f"deploy-{key}").start()
    return {"status": "started", "target": req.target, "key": key}


@router.post("/data")
async def push_data(req: PushDataRequest, request: Request):
    """Push memory/ data to Fly."""
    _require_local(request)

    key = f"data-{req.client or 'all'}"
    with _active_lock:
        if key in _active and _active[key]["status"] == "running":
            return {"status": "already_running", "message": "A data push is already in progress."}
        _active[key] = {"status": "running", "log": ""}

    def _run():
        script = _PROJECT_ROOT / "push-to-fly.sh"
        cmd = [str(script)]
        if req.client:
            cmd.append(req.client)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=180,
            )
            with _active_lock:
                _active[key] = {
                    "status": "completed" if result.returncode == 0 else "failed",
                    "log": (result.stdout or "") + (result.stderr or ""),
                    "returncode": result.returncode,
                }
            if result.returncode == 0:
                logger.info("[deploy] data push (%s) succeeded", req.client or "all")
            else:
                logger.warning("[deploy] data push (%s) failed", req.client or "all")
        except subprocess.TimeoutExpired:
            with _active_lock:
                _active[key] = {"status": "failed", "log": "Timed out after 180s"}
        except Exception as e:
            with _active_lock:
                _active[key] = {"status": "failed", "log": str(e)}

    threading.Thread(target=_run, daemon=True, name=f"deploy-{key}").start()
    return {"status": "started", "client": req.client, "key": key}


@router.get("/status/{key}")
async def deploy_status(key: str, request: Request):
    """Poll the status of a deploy/push operation."""
    _require_local(request)

    with _active_lock:
        job = _active.get(key)
    if not job:
        raise HTTPException(status_code=404, detail="No deploy job with that key")
    return {"key": key, **job}


# ---------------------------------------------------------------- User ban/unban


class BanRequest(BaseModel):
    email: str


@router.post("/ban")
async def ban_user(req: BanRequest, request: Request):
    """Temporarily ban a user. They'll get a 403 on every request until unbanned."""
    _require_local(request)
    from backend.src.auth.acl import Acl
    acl: Acl = request.app.state.acl
    email = req.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email must not be empty")
    if acl.is_admin(email):
        raise HTTPException(status_code=400, detail="Cannot ban an admin")
    was_new = acl.ban(email)
    # Also push the updated ACL to Fly so the ban takes effect in production
    _push_acl_to_fly(acl)
    return {"status": "banned" if was_new else "already_banned", "email": email}


@router.post("/unban")
async def unban_user(req: BanRequest, request: Request):
    """Lift a user's ban."""
    _require_local(request)
    from backend.src.auth.acl import Acl
    acl: Acl = request.app.state.acl
    email = req.email.strip().lower()
    was_banned = acl.unban(email)
    _push_acl_to_fly(acl)
    return {"status": "unbanned" if was_banned else "not_banned", "email": email}


@router.get("/banned")
async def list_banned(request: Request):
    """List currently banned users."""
    _require_local(request)
    from backend.src.auth.acl import Acl
    acl: Acl = request.app.state.acl
    return {"banned": acl.list_banned()}


@router.get("/users")
async def list_users(request: Request):
    """List all known users from the ACL (admins + scoped users)."""
    _require_local(request)
    from backend.src.auth.acl import Acl
    acl: Acl = request.app.state.acl
    return {"users": acl.list_all_users()}


def _push_acl_to_fly(acl) -> None:
    """Best-effort push of the local acl.json to Fly so bans propagate immediately."""
    import shutil
    try:
        if not shutil.which("fly"):
            return
        acl_path = acl._path
        if not acl_path.exists():
            return
        # Upload via fly sftp
        import subprocess
        result = subprocess.run(
            ["fly", "sftp", "shell", "-a", "amphoreus"],
            input=f"put {acl_path} /data/acl.json\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("[deploy] ACL pushed to Fly")
        else:
            logger.warning("[deploy] ACL push failed: %s", (result.stderr or "")[:200])
    except Exception:
        logger.exception("[deploy] ACL push to Fly failed (non-fatal)")
