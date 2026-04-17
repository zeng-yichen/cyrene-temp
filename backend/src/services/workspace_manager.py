"""Persistent workspace manager — replaces tempfile.mkdtemp with versioned local workspaces.

Supports content-hash selective refresh, snapshot/rollback, and optional E2B backend.
"""

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from backend.src.core.config import get_settings
from backend.src.db import local as db
from backend.src.db import vortex

logger = logging.getLogger(__name__)


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _load_sync_state(workspace: Path) -> dict[str, str]:
    state_file = workspace / ".sync_state.json"
    if state_file.exists():
        return json.loads(state_file.read_text(encoding="utf-8"))
    return {}


def _save_sync_state(workspace: Path, state: dict[str, str]) -> None:
    state_file = workspace / ".sync_state.json"
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def provision_workspace(client_slug: str) -> dict:
    """Create or resume a persistent workspace for a client."""
    settings = get_settings()

    if settings.workspace_backend == "e2b":
        return _provision_e2b(client_slug)

    ws = vortex.workspace_dir(client_slug)
    ws.mkdir(parents=True, exist_ok=True)

    for subdir in ("memory", "scratch", "output", ".sessions", "feedback/edits"):
        (ws / subdir).mkdir(parents=True, exist_ok=True)

    _sync_local_data(client_slug, ws)

    return {"path": str(ws), "backend": "local"}


def _sync_local_data(client_slug: str, workspace: Path) -> None:
    """Selectively refresh workspace data from memory/ using content hashing."""
    sync_state = _load_sync_state(workspace)
    new_state = {}

    source_dirs = {
        "source-material": vortex.transcripts_dir(client_slug),
        "published-posts": vortex.accepted_dir(client_slug),
        "feedback/edits": vortex.revisions_dir(client_slug),
        "past-posts": vortex.past_posts_dir(client_slug),
    }

    for target_name, source_dir in source_dirs.items():
        target_dir = workspace / "memory" / target_name
        target_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            continue

        for src_file in source_dir.iterdir():
            if src_file.is_file() and src_file.suffix in (".txt", ".md", ".json", ".csv"):
                content = src_file.read_text(encoding="utf-8", errors="replace")
                content_hash = _hash_content(content)
                file_key = f"{target_name}/{src_file.name}"

                if sync_state.get(file_key) == content_hash:
                    new_state[file_key] = content_hash
                    continue

                dst = target_dir / src_file.name
                dst.write_text(content, encoding="utf-8")
                new_state[file_key] = content_hash
                logger.debug("Synced %s (hash changed)", file_key)

    strategy_dir = vortex.content_strategy_dir(client_slug)
    if strategy_dir.exists():
        target = workspace / "memory" / "strategy.md"
        files = sorted(strategy_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            content = files[0].read_text(encoding="utf-8", errors="replace")
            ch = _hash_content(content)
            if sync_state.get("strategy.md") != ch:
                target.write_text(content, encoding="utf-8")
                new_state["strategy.md"] = ch

    _save_sync_state(workspace, new_state)


def create_snapshot(client_slug: str, run_id: str) -> str:
    """Snapshot the current workspace state and record it in SQLite."""
    ws = vortex.workspace_dir(client_slug)
    snap_dir = vortex.snapshots_dir(client_slug) / run_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    for item in ws.iterdir():
        if item.name in ("snapshots", ".sessions"):
            continue
        dst = snap_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    content_hashes = json.dumps(_load_sync_state(ws))
    db.cache_set(f"snapshot:{client_slug}:{run_id}", content_hashes, ttl_seconds=365 * 24 * 3600)

    _record_snapshot(client_slug, run_id, str(snap_dir), content_hashes)

    return str(snap_dir)


def _record_snapshot(client_slug: str, run_id: str, snapshot_path: str, content_hashes: str) -> None:
    from backend.src.db.local import get_connection
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO workspace_snapshots (id, client_slug, run_id, snapshot_path, content_hashes) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, client_slug, run_id, snapshot_path, content_hashes),
        )


def rollback_snapshot(client_slug: str, run_id: str) -> None:
    """Restore workspace from a snapshot."""
    ws = vortex.workspace_dir(client_slug)
    snap_dir = vortex.snapshots_dir(client_slug) / run_id

    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshot {run_id} not found for {client_slug}")

    for item in ws.iterdir():
        if item.name in ("snapshots", ".sessions"):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    for item in snap_dir.iterdir():
        dst = ws / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    logger.info("Rolled back %s to snapshot %s", client_slug, run_id)


def save_feedback(client_slug: str, original: str, revised: str) -> None:
    """Save edit feedback to the client's memory/ folder so Stelle reads it on next run."""
    feedback_dir = vortex.feedback_dir(client_slug)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    filename = f"edit_{int(time.time())}_{uuid.uuid4().hex[:6]}.md"
    filepath = feedback_dir / filename
    filepath.write_text(
        f"## Original\n\n{original}\n\n## Revised\n\n{revised}\n",
        encoding="utf-8",
    )


def list_workspace_files(client_slug: str, subpath: str = "") -> list[dict]:
    """List files in the workspace for the file tree UI."""
    ws = vortex.workspace_dir(client_slug)
    target = ws / subpath if subpath else ws

    if not target.exists():
        return []

    entries = []
    for item in sorted(target.iterdir()):
        if item.name.startswith("."):
            continue
        entries.append({
            "name": item.name,
            "path": str(item.relative_to(ws)),
            "is_dir": item.is_dir(),
            "size": item.stat().st_size if item.is_file() else None,
        })
    return entries


def _provision_e2b(client_slug: str) -> dict:
    """Provision an E2B sandbox (cloud path)."""
    try:
        from e2b_code_interpreter import Sandbox
        settings = get_settings()
        sandbox = Sandbox(api_key=settings.e2b_api_key)
        return {"sandbox_id": sandbox.id, "backend": "e2b"}
    except Exception as e:
        logger.error("E2B provisioning failed: %s", e)
        raise
