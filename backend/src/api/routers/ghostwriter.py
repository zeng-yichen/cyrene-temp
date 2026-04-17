"""Ghostwriter API — generate, stream, manage workspaces."""

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ghostwriter", tags=["ghostwriter"])


class GenerateRequest(BaseModel):
    company: str
    prompt: str | None = None
    model: str = "claude-opus-4-6"


class InlineEditRequest(BaseModel):
    company: str
    post_text: str
    instruction: str


class LinkedInUsernameRequest(BaseModel):
    username: str




@router.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    # ACL check: company is in request body, not path, so the global
    # path-based middleware won't catch it.
    from backend.src.auth.middleware import require_client_body
    require_client_body(request, req.company)
    """Start a ghostwriter generation job as a DETACHED subprocess.

    Previously this handler spawned Stelle as an in-process background
    thread. That was fatal under ``uvicorn --reload``: any edit to a
    backend .py file during a 20-minute generation run would hot-reload
    the FastAPI process and kill the in-flight thread, losing the batch
    (or half of it, as happened on April 11).

    The new flow:

      1. Create a run record + job_id in SQLite (same as before)
      2. Spawn ``backend.src.agents.stelle_runner`` as a subprocess with
         ``start_new_session=True`` so it lives in its own process group
         and survives parent process death
      3. Return the job_id immediately; the frontend connects to
         ``/stream/{job_id}`` for SSE events
      4. The runner writes events directly to the ``run_events`` SQLite
         table, and ``drain_events`` polls that table in addition to
         the in-memory queue, so the SSE endpoint sees subprocess events
      5. On parent restart, the runner keeps going; a new SSE
         connection simply reconnects and resumes streaming from the
         last-seen event id
    """
    import subprocess
    import sys
    from pathlib import Path
    from backend.src.usage.context import current_user_email

    # Capture the authenticated user's email from the ContextVar before
    # spawning the subprocess (ContextVars don't cross process boundaries).
    user_email = current_user_email.get()

    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="stelle",
        prompt=req.prompt,
        creator_id=user_email,
    )

    # Build the subprocess command. Use the same python interpreter the
    # parent is running under so we inherit the same environment (venv,
    # installed packages, env vars).
    project_root = Path(__file__).resolve().parents[4]
    cmd = [
        sys.executable,
        "-m", "backend.src.agents.stelle_runner",
        "--company", req.company,
        "--job-id", job_id,
        "--model", req.model,
    ]
    if req.prompt:
        cmd.extend(["--prompt", req.prompt])
    if user_email:
        cmd.extend(["--user-email", user_email])

    # Subprocess log file (separate from Stelle's session log so we can
    # debug runner-level issues independently)
    log_dir = project_root / ".runner-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"stelle_runner_{job_id}.out"

    try:
        # start_new_session=True → os.setsid() in the child → new process
        # group that won't receive SIGTERM from uvicorn's reload handler.
        # stdin is closed because the runner is non-interactive; stdout/
        # stderr go to the log file (also tee'd to the child's own
        # logging FileHandler via _configure_logging).
        log_handle = open(log_file, "wb")
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        logger.info(
            "[ghostwriter] spawned detached stelle_runner pid=%d job_id=%s company=%s log=%s",
            proc.pid, job_id, req.company, log_file,
        )
    except Exception as exc:
        logger.exception("[ghostwriter] failed to spawn stelle_runner: %s", exc)
        job_manager.set_status(job_id, "failed", error=f"spawn failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not spawn Stelle runner: {exc}")

    return {"job_id": job_id, "status": "pending", "runner_pid": proc.pid}


@router.get("/stream/{job_id}")
async def stream_events(job_id: str, after_id: int = 0):
    """SSE endpoint — streams AgentEvents from the job's queue.

    Emits SSE keepalive comments (`: keepalive\\n\\n`) every ~15 seconds
    when no real events are available.  This prevents Cloudflare Access
    and other reverse proxies from killing the connection due to their
    idle-read timeouts (typically 100 s for CF).

    ``after_id`` lets a reconnecting client resume from the last event
    id it saw, so mid-stream drops don't cause replay or loss.
    """
    from backend.src.db.local import get_run
    if not job_manager.get_job(job_id) and not get_run(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        job_manager.sse_stream(job_id, timeout=3600, heartbeat_interval=15, after_id=after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "output": job.get("output"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


@router.post("/provision")
async def provision_workspace(req: GenerateRequest):
    """Provision a persistent workspace for a client."""
    from backend.src.services.workspace_manager import provision_workspace
    result = provision_workspace(req.company)
    return {"status": "provisioned", "workspace": result}



@router.post("/inline-edit")
async def edit_single_post(req: InlineEditRequest):
    """Inline text editing via Stelle — returns a job_id for SSE streaming."""
    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="stelle-inline-edit",
        prompt=req.instruction,
        creator_id=None,
    )

    def _run_edit(jid: str, company: str, post_text: str, instruction: str):
        from backend.src.agents.stelle_adapter import run_inline_edit
        result = run_inline_edit(company, post_text, instruction, job_id=jid)
        job_manager.emit_event(jid, done_event(result))
        return result

    job_manager.run_in_background(
        job_id,
        target=_run_edit,
        args=(job_id, req.company, req.post_text, req.instruction),
    )
    return {"job_id": job_id, "status": "pending"}


@router.get("/runs/{run_id}/events")
async def get_run_events(run_id: str):
    """Full event timeline for a specific run."""
    from backend.src.db.local import get_run_events, get_run
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    events = get_run_events(run_id)
    return {"run": run, "events": events}


@router.get("/{company}/runs")
async def get_run_history(company: str, limit: int = Query(20)):
    """Run history for a client."""
    from backend.src.db.local import list_runs
    runs = list_runs(company, limit=limit)
    return {"runs": runs}


@router.post("/{company}/rollback/{run_id}")
async def rollback_to_run(company: str, run_id: str):
    """Rollback workspace to a previous run's snapshot."""
    from backend.src.services.workspace_manager import rollback_snapshot
    rollback_snapshot(company, run_id)
    return {"status": "rolled_back", "run_id": run_id}


@router.get("/{company}/ordinal-users")
async def list_ordinal_users(company: str):
    """Workspace members from Ordinal (for approver picker). Requires API key in ordinal_auth CSV."""
    from backend.src.agents.hyacinthia import Hyacinthia
    raw = Hyacinthia().get_users(company)
    if isinstance(raw, list):
        users = raw
    elif isinstance(raw, dict):
        users = raw.get("users") or raw.get("data") or []
    else:
        users = []
    return {"users": users}


@router.get("/sandbox/{company}/files")
async def browse_workspace_files(company: str, path: str = Query("")):
    """Browse workspace files."""
    from backend.src.services.workspace_manager import list_workspace_files
    files = list_workspace_files(company, path)
    return {"files": files}


@router.get("/{company}/linkedin-username")
async def get_linkedin_username(company: str):
    """Return the stored LinkedIn username for a client, or null if not set."""
    from backend.src.db import vortex
    path = vortex.linkedin_username_path(company)
    if not path.exists():
        return {"username": None}
    return {"username": path.read_text(encoding="utf-8").strip() or None}


@router.post("/{company}/linkedin-username")
async def save_linkedin_username(company: str, req: LinkedInUsernameRequest):
    """Write the LinkedIn username file for a client."""
    from backend.src.db import vortex
    username = req.username.strip().lstrip("@").strip("/").split("/")[-1]
    if not username:
        raise HTTPException(status_code=400, detail="Username must not be empty")
    path = vortex.linkedin_username_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(username, encoding="utf-8")
    return {"status": "saved", "username": username}


# ---------------------------------------------------------------------------
# Calendar — post scheduling interface
# ---------------------------------------------------------------------------


class ScheduleRequest(BaseModel):
    scheduled_date: str | None = None  # ISO date e.g. "2026-04-14" or null to unschedule


class AutoAssignRequest(BaseModel):
    cadence: str = "3pw"  # "3pw" (Mon/Tue/Thu) or "2pw" (Tue/Thu)
    start_date: str | None = None  # ISO date; defaults to next Monday


class PushAllRequest(BaseModel):
    pass  # no body needed; pushes all unpushed posts with scheduled_dates


@router.get("/{company}/calendar")
async def get_calendar(company: str, month: str = Query(None)):
    """Return all posts for a company, optionally filtered to a month.

    Posts include scheduled_date, publication_order, status, hook preview,
    and ordinal_post_id. Suitable for rendering a calendar grid.
    """
    from backend.src.db.local import list_calendar_posts
    posts = list_calendar_posts(company, month=month)
    result = []
    for p in posts:
        content = p.get("content") or ""
        hook = content.split("\n")[0][:120] if content else ""
        result.append({
            "id": p.get("id"),
            "hook": hook,
            "content": content,
            "content_preview": content[:300],
            "status": p.get("status"),
            "scheduled_date": p.get("scheduled_date"),
            "publication_order": p.get("publication_order"),
            "ordinal_post_id": p.get("ordinal_post_id"),
            "created_at": p.get("created_at"),
            "why_post": p.get("why_post"),
        })
    return {"company": company, "month": month, "posts": result}


@router.patch("/{company}/posts/{post_id}/schedule")
async def schedule_post(company: str, post_id: str, req: ScheduleRequest):
    """Update the scheduled publication date for a post (drag-drop on calendar)."""
    from backend.src.db.local import update_post_schedule, get_local_post
    post = get_local_post(post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.get("company") != company:
        raise HTTPException(status_code=403, detail="Post belongs to a different company")
    updated = update_post_schedule(post_id, req.scheduled_date)
    return updated


@router.post("/{company}/calendar/auto-assign")
async def auto_assign_calendar(company: str, req: AutoAssignRequest):
    """Auto-assign unscheduled posts to calendar slots based on cadence.

    Distributes posts in publication_order into the cadence's day slots
    (Mon/Tue/Thu for 3pw, Tue/Thu for 2pw) starting from start_date
    (defaults to next Monday).
    """
    from datetime import date, timedelta
    from backend.src.db.local import list_calendar_posts, update_post_schedule

    posts = list_calendar_posts(company)
    unscheduled = [p for p in posts if not p.get("scheduled_date") and p.get("status") == "draft"]
    unscheduled.sort(key=lambda p: p.get("publication_order") or 999)

    if not unscheduled:
        return {"assigned": 0, "message": "No unscheduled draft posts"}

    # Determine cadence days (0=Mon, 1=Tue, ..., 4=Fri)
    if req.cadence == "2pw":
        cadence_days = {1, 3}  # Tue, Thu
    else:
        cadence_days = {0, 1, 3}  # Mon, Tue, Thu

    # Start from: explicit start_date, OR the day after the last
    # already-scheduled post, OR today — whichever is latest. Then
    # advance to the next cadence day (including today if it's one).
    if req.start_date:
        try:
            cursor = date.fromisoformat(req.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    else:
        today = date.today()
        # Find the latest already-scheduled post date
        scheduled_dates = [
            p["scheduled_date"]
            for p in posts
            if p.get("scheduled_date")
        ]
        if scheduled_dates:
            try:
                last_scheduled = max(date.fromisoformat(d) for d in scheduled_dates)
                # Start the day after the last scheduled post
                cursor = max(today, last_scheduled + timedelta(days=1))
            except Exception:
                cursor = today
        else:
            cursor = today

        # Advance cursor to the next cadence day (including today)
        for _ in range(7):
            if cursor.weekday() in cadence_days:
                break
            cursor += timedelta(days=1)

    assigned = []
    post_idx = 0
    max_search_days = 90  # don't look more than 3 months out

    for _ in range(max_search_days):
        if post_idx >= len(unscheduled):
            break
        if cursor.weekday() in cadence_days:
            post = unscheduled[post_idx]
            date_str = cursor.isoformat()
            update_post_schedule(post["id"], date_str)
            assigned.append({"id": post["id"], "scheduled_date": date_str})
            post_idx += 1
        cursor += timedelta(days=1)

    return {"assigned": len(assigned), "posts": assigned}


@router.post("/{company}/calendar/push-all")
async def push_all_scheduled(company: str):
    """Push all unpushed posts with scheduled_dates to Ordinal.

    For each post with status='draft' and a scheduled_date, calls
    Hyacinthia to push to Ordinal with that date as the publish date.
    Returns the results per post.
    """
    from backend.src.db.local import list_calendar_posts
    from backend.src.agents.hyacinthia import Hyacinthia

    posts = list_calendar_posts(company)
    pushable = [
        p for p in posts
        if p.get("status") == "draft"
        and p.get("scheduled_date")
        and not p.get("ordinal_post_id")
    ]

    if not pushable:
        return {"pushed": 0, "message": "No draft posts with scheduled dates to push"}

    hy = Hyacinthia()
    results = []
    for post in pushable:
        try:
            ordinal_result = hy.push_post(
                company,
                post["id"],
                post["content"],
                scheduled_date=post["scheduled_date"],
                why_post=post.get("why_post"),
            )
            results.append({
                "id": post["id"],
                "status": "pushed",
                "ordinal_post_id": ordinal_result.get("ordinal_post_id"),
                "scheduled_date": post["scheduled_date"],
            })
        except Exception as e:
            results.append({
                "id": post["id"],
                "status": "failed",
                "error": str(e)[:200],
            })

    return {"pushed": sum(1 for r in results if r["status"] == "pushed"), "results": results}


class PushSingleRequest(BaseModel):
    post_id: str


@router.post("/{company}/calendar/push-single")
async def push_single_post(company: str, req: PushSingleRequest):
    """Push one specific post to Ordinal."""
    from backend.src.db.local import get_local_post
    from backend.src.agents.hyacinthia import Hyacinthia

    post = get_local_post(req.post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.get("company") != company:
        raise HTTPException(status_code=403, detail="Post belongs to a different company")
    if post.get("ordinal_post_id"):
        return {"id": req.post_id, "status": "already_pushed", "ordinal_post_id": post["ordinal_post_id"]}

    try:
        hy = Hyacinthia()
        ordinal_result = hy.push_post(
            company,
            post["id"],
            post["content"],
            scheduled_date=post.get("scheduled_date"),
            why_post=post.get("why_post"),
        )
        return {
            "id": req.post_id,
            "status": "pushed",
            "ordinal_post_id": ordinal_result.get("ordinal_post_id"),
            "scheduled_date": post.get("scheduled_date"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Push failed: {str(e)[:200]}")


