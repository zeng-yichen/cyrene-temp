"""Interview API — Tribbie live interview companion."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/interview", tags=["interview"])


class StartSessionRequest(BaseModel):
    company: str
    client_name: str | None = None


class StopSessionRequest(BaseModel):
    job_id: str
    company: str


class TrashTranscriptRequest(BaseModel):
    path: str


@router.get("/devices")
async def list_devices():
    """
    List available audio input devices.
    The frontend uses this to detect whether BlackHole is installed and to show
    the setup guide if it is not.
    """
    try:
        from backend.src.agents.tribbie import list_audio_devices
        devices = list_audio_devices()
        has_blackhole = any(d["is_blackhole"] for d in devices)
        return {"devices": devices, "has_blackhole": has_blackhole}
    except Exception as e:
        logger.warning("[Interview] Could not query audio devices: %s", e)
        return {"devices": [], "has_blackhole": False, "error": str(e)}


@router.post("/start")
async def start_session(req: StartSessionRequest):
    """Start a live Tribbie capture session. Returns job_id for SSE streaming."""
    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="tribbie",
        prompt=f"Live interview session — {req.client_name or req.company}",
        creator_id=None,
    )

    def _run(jid: str, company: str) -> None:
        from backend.src.agents.tribbie_adapter import run_tribbie_session
        run_tribbie_session(company, job_id=jid)

    job_manager.run_in_background(job_id, target=_run, args=(job_id, req.company))
    return {"job_id": job_id, "status": "pending"}


@router.post("/stop")
async def stop_session(req: StopSessionRequest):
    """Signal the running Tribbie session to stop and save the transcript."""
    from backend.src.agents.tribbie import stop_session as _stop_session
    found = _stop_session(req.job_id)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found or already stopped.")
    return {"status": "stopping", "job_id": req.job_id}


@router.post("/trash-transcript")
async def trash_transcript(req: TrashTranscriptRequest):
    """Move a saved interview transcript to the macOS Trash (~/.Trash/)."""
    import shutil
    from pathlib import Path
    from backend.src.db import vortex

    file_path = Path(req.path).resolve()

    # Safety: only allow trashing files inside the memory/*/transcripts/ tree
    allowed_root = vortex.MEMORY_ROOT.resolve()
    if not str(file_path).startswith(str(allowed_root)):
        raise HTTPException(status_code=403, detail="Path is outside allowed directory.")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found.")

    trash_dir = Path.home() / ".Trash"
    dest = trash_dir / file_path.name
    # Avoid clobbering an existing file in Trash with the same name
    if dest.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        dest = trash_dir / f"{stem}_{int(file_path.stat().st_mtime)}{suffix}"

    shutil.move(str(file_path), str(dest))
    logger.info("[Interview] Trashed transcript: %s → %s", file_path, dest)
    return {"status": "trashed", "destination": str(dest)}


@router.get("/stream/{job_id}")
async def stream_session(job_id: str, after_id: int = 0):
    """SSE stream for a Tribbie session. Streams until 'done' or 'error'."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return StreamingResponse(
        # 2-hour timeout — enough for even the longest interview
        job_manager.sse_stream(job_id, timeout=7_200, heartbeat_interval=15, after_id=after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
