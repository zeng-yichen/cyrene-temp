"""Router for launching the classic Tkinter desktop GUI."""

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/desktop", tags=["desktop"])

_gui_process: subprocess.Popen | None = None


@router.post("/launch")
async def launch_gui():
    """Spawn the classic Amphoreus Tkinter GUI as a subprocess."""
    global _gui_process

    if _gui_process is not None and _gui_process.poll() is None:
        return {"status": "already_running", "pid": _gui_process.pid}

    project_root = Path(__file__).resolve().parents[4]
    gui_script = project_root / "amphoreus.py"

    if not gui_script.exists():
        raise HTTPException(status_code=404, detail="amphoreus.py not found at project root")

    _gui_process = subprocess.Popen(
        [sys.executable, str(gui_script)],
        cwd=str(project_root),
    )

    return {"status": "launched", "pid": _gui_process.pid}


@router.get("/status")
async def gui_status():
    """Check whether the desktop GUI is running."""
    if _gui_process is None:
        return {"running": False}
    running = _gui_process.poll() is None
    return {"running": running, "pid": _gui_process.pid if running else None}
