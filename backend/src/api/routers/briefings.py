"""Briefings API — reads Cyrene's strategic brief for interview prep."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.src.core.events import done_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/briefings", tags=["briefings"])


def _cyrene_brief_path(company: str) -> Path:
    from backend.src.db import vortex
    return vortex.memory_dir(company) / "cyrene_brief.json"


@router.get("/check/{company}")
async def check_briefing(company: str):
    """Check whether a Cyrene brief exists for a company."""
    return {"exists": _cyrene_brief_path(company).exists()}


@router.get("/content/{company}")
async def get_briefing_content(company: str):
    """Return the Cyrene brief as structured JSON for interview prep."""
    path = _cyrene_brief_path(company)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No Cyrene brief found. Run Cyrene first.")
    brief = json.loads(path.read_text(encoding="utf-8"))
    return {"content": brief}
