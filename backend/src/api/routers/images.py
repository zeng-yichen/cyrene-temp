"""Images API — generate, stream, list, and serve assembled images."""

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.src.db import vortex as P
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", tags=["images"])


class GenerateImageRequest(BaseModel):
    company: str
    post_text: str
    model: str = "claude-opus-4-6"
    feedback: str = Field(
        default="",
        description="Human revision notes for Phainon (bitter-lesson iteration).",
    )
    reference_image_id: str = Field(
        default="",
        description="Stem of a prior PNG in products/{company}/images (e.g. image_20260101_120000).",
    )
    local_post_id: str = Field(
        default="",
        description="Optional local_posts.id for feedback log correlation.",
    )


def _append_image_feedback_log(company: str, entry: dict) -> None:
    path = P.image_feedback_log_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {**entry, "ts": time.time()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@router.post("/generate")
async def generate_image(req: GenerateImageRequest):
    """Start an image assembly job (initial or revision with feedback + optional reference image)."""
    prompt_bits = req.post_text[:200]
    if req.feedback.strip():
        prompt_bits += " | feedback"

    job_id = job_manager.create_job(
        client_slug=req.company,
        agent="phainon",
        prompt=prompt_bits,
        creator_id=None,
    )

    ref_path = ""
    if (req.reference_image_id or "").strip():
        stem = req.reference_image_id.strip()
        p = P.images_dir(req.company) / f"{stem}.png"
        if p.is_file():
            ref_path = str(p)

    if req.feedback.strip() or ref_path:
        _append_image_feedback_log(
            req.company,
            {
                "job_id": job_id,
                "local_post_id": (req.local_post_id or "").strip() or None,
                "feedback": req.feedback.strip(),
                "reference_image_id": (req.reference_image_id or "").strip() or None,
                "post_text_excerpt": req.post_text[:500],
            },
        )

    def _run(
        jid: str,
        company: str,
        post_text: str,
        model: str,
        feedback: str,
        reference_path: str,
    ):
        from backend.src.agents.phainon_adapter import run_phainon

        return run_phainon(
            company,
            post_text,
            model,
            job_id=jid,
            feedback_instruction=feedback,
            reference_image_path=reference_path or None,
        )

    job_manager.run_in_background(
        job_id,
        target=_run,
        args=(
            job_id,
            req.company,
            req.post_text,
            req.model,
            req.feedback or "",
            ref_path,
        ),
    )

    return {"job_id": job_id, "status": "pending"}


@router.get("/stream/{job_id}")
async def stream_events(job_id: str, after_id: int = 0):
    """SSE endpoint — streams AgentEvents from the job's queue."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        job_manager.sse_stream(job_id, timeout=3600, heartbeat_interval=15, after_id=after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{company}")
async def list_images(company: str, limit: int = Query(50)):
    """List generated images for a company."""
    from backend.src.db import vortex
    img_dir = vortex.images_dir(company)
    if not img_dir.exists():
        return {"images": []}

    images = []
    for f in sorted(img_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        meta_path = f.with_name(f.stem + "_metadata.json")
        metadata = None
        if meta_path.exists():
            import json
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        images.append({
            "id": f.stem,
            "filename": f.name,
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "created_at": f.stat().st_mtime,
            "metadata": metadata,
        })

    return {"images": images}


@router.get("/{company}/{image_id}")
async def serve_image(company: str, image_id: str):
    """Serve an image file."""
    from backend.src.db import vortex
    img_dir = vortex.images_dir(company)

    image_path = img_dir / f"{image_id}.png"
    if not image_path.exists():
        image_path = img_dir / image_id
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=image_path.name,
    )
