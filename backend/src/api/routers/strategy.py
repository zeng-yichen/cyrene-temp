"""Strategy API — Cyrene strategic reviews + Herta content strategy generation.

Cyrene (strategic growth agent) produces a JSON brief with interview
questions, DM targets, content priorities, ABM targeting, and Stelle
scheduling. Runs on demand via POST /api/strategy/cyrene/{company}.

Herta (content strategy document generator) is the older, static
strategy path — still available at /generate for clients that prefer
a formatted strategy document over Cyrene's live brief.
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/strategy", tags=["strategy"])


class StrategyRequest(BaseModel):
    company: str
    prompt: str | None = None


# ---------------------------------------------------------------------------
# Cyrene — strategic growth agent
# ---------------------------------------------------------------------------


@router.post("/cyrene/{company}")
async def run_cyrene_review(company: str):
    """Start a Cyrene strategic review as a background job.

    Cyrene is a turn-based Opus agent that studies the client's full
    engagement history, ICP exposure trends, warm prospects, and
    transcript inventory, then produces a strategic brief with:
    interview questions, DM targets, content priorities, ABM targeting,
    Stelle scheduling, and a self-scheduled next-run trigger.

    Returns a job_id; connect to /stream/{job_id} for SSE progress.
    The final brief is also persisted to memory/{company}/cyrene_brief.json.
    """
    job_id = job_manager.create_job(
        client_slug=company,
        agent="cyrene",
        prompt=None,
        creator_id=None,
    )

    def _run(jid: str, co: str):
        from backend.src.agents.cyrene import run_strategic_review
        job_manager.emit_event(jid, status_event(f"Starting Cyrene strategic review for {co}..."))
        try:
            brief = run_strategic_review(co)
            if brief.get("_error"):
                job_manager.emit_event(jid, status_event(f"Cyrene failed: {brief['_error']}"))
                return brief
            job_manager.emit_event(jid, done_event(json.dumps(brief, default=str)))
            return brief
        except Exception as e:
            job_manager.emit_event(jid, status_event(f"Cyrene crashed: {e}"))
            raise

    job_manager.run_in_background(job_id, target=_run, args=(job_id, company))
    return {"job_id": job_id, "status": "pending"}


@router.get("/cyrene/{company}/brief")
async def get_cyrene_brief(company: str):
    """Return the most recent Cyrene brief for a client (if any)."""
    from backend.src.db import vortex
    path = vortex.memory_dir(company) / "cyrene_brief.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No Cyrene brief for {company}. Run POST /api/strategy/cyrene/{company} first.",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read brief: {e}")


# ---------------------------------------------------------------------------
# Herta content strategy generation endpoints
# ---------------------------------------------------------------------------


@router.post("/generate")
async def generate_strategy(req: StrategyRequest):
    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="herta",
        prompt=req.prompt,
        creator_id=None,
    )

    def _run(jid: str, company: str, prompt: str | None):
        from backend.src.agents.herta_adapter import run_herta
        job_manager.emit_event(jid, status_event(f"Generating strategy for {company}..."))
        result = run_herta(company, prompt, job_id=jid)
        job_manager.emit_event(jid, done_event(result))
        return result

    job_manager.run_in_background(job_id, target=_run, args=(job_id, req.company, req.prompt))
    return {"job_id": job_id, "status": "pending"}


@router.get("/stream/{job_id}")
async def stream_strategy(job_id: str, after_id: int = 0):
    from backend.src.db.local import get_run
    if not job_manager.get_job(job_id) and not get_run(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        job_manager.sse_stream(job_id, timeout=3600, heartbeat_interval=15, after_id=after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{company}/html")
async def get_strategy_html(company: str):
    """Get the most recent content strategy HTML for a client (JSON response)."""
    from backend.src.db import vortex
    strategy_dir = vortex.content_strategy_dir(company)
    if not strategy_dir.exists():
        return {"html": None}
    files = sorted(strategy_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"html": None}
    return {"html": files[0].read_text(encoding="utf-8")}


@router.get("/{company}/view", response_class=HTMLResponse)
async def view_strategy_html(company: str):
    """Serve the most recent content strategy HTML directly for browser rendering."""
    from backend.src.db import vortex
    strategy_dir = vortex.content_strategy_dir(company)
    if not strategy_dir.exists():
        raise HTTPException(status_code=404, detail="No content strategy found")
    files = sorted(strategy_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No content strategy HTML found")
    return HTMLResponse(content=files[0].read_text(encoding="utf-8"))


@router.get("/{company}")
async def get_current_strategy(company: str):
    """Get the most recent content strategy for a client."""
    from backend.src.db import vortex
    strategy_dir = vortex.content_strategy_dir(company)
    if not strategy_dir.exists():
        return {"strategy": None}
    files = sorted(strategy_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {"strategy": None}
    return {"strategy": files[0].read_text(encoding="utf-8"), "path": str(files[0])}
