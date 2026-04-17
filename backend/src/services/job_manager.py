"""Job manager — tracks long-running agent tasks with SSE streaming.

In-memory job store pattern. Each job has an event queue
that the SSE endpoint drains.
"""

import contextlib
import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable

from backend.src.core.events import AgentEvent
from backend.src.db import local as db

logger = logging.getLogger(__name__)

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def create_job(client_slug: str, agent: str, prompt: str | None = None, creator_id: str | None = None) -> str:
    job_id = str(uuid.uuid4())
    event_queue: queue.Queue[AgentEvent] = queue.Queue(maxsize=2000)

    with _LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "client_slug": client_slug,
            "agent": agent,
            "creator_id": creator_id,
            "prompt": prompt,
            "status": "pending",
            "output": None,
            "error": None,
            "event_queue": event_queue,
            "start_time": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    db.create_run(job_id, client_slug, agent, prompt)
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _JOBS.get(job_id)


def set_status(job_id: str, status: str, output: str | None = None, error: str | None = None) -> None:
    with _LOCK:
        if job_id not in _JOBS:
            return
        _JOBS[job_id]["status"] = status
        _JOBS[job_id]["updated_at"] = time.time()
        if output is not None:
            _JOBS[job_id]["output"] = output
        if error is not None:
            _JOBS[job_id]["error"] = error

    if status in ("completed", "failed"):
        db.complete_run(job_id, output=output, error=error)


def emit_event(job_id: str, event: AgentEvent) -> None:
    # ALWAYS persist to SQLite first, regardless of whether this process
    # owns the in-memory job record. This is what lets a detached
    # subprocess (stelle_runner) emit events that the parent FastAPI
    # process can still see via the run_events table. Before this fix,
    # emit_event early-returned when the job wasn't in the local _JOBS
    # dict, so subprocess-origin events were silently dropped.
    try:
        db.record_event(job_id, event.type, event.data)
    except Exception:
        logger.debug("record_event failed for job %s", job_id, exc_info=True)

    # Also push to the in-memory queue if this process owns the job
    # (the happy fast path for in-process threaded runs).
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return
    eq = job.get("event_queue")
    if eq:
        with contextlib.suppress(queue.Full):
            eq.put_nowait(event)


def drain_events(job_id: str, timeout: float = 30.0, after_id: int = 0):
    """Generator that yields ``(AgentEvent, row_id)`` tuples.

    SQLite ``run_events`` is the single source of truth. Events poll at
    ~500ms intervals — fine given Stelle/Cyrene emit text-delta and
    tool-call events every few seconds, not per-frame.

    ``after_id`` lets a reconnecting client resume from the last event
    id it saw; the backend skips rows with ``id <= after_id`` so the UI
    neither replays nor misses events across a disconnect.
    """
    from backend.src.core.events import AgentEvent

    deadline = time.time() + timeout
    last_seen_event_id = after_id
    poll_interval = 0.5
    terminal_types = ("done", "error")

    while time.time() < deadline:
        yielded_anything_this_cycle = False

        try:
            new_rows = db.get_run_events_after(job_id, last_seen_event_id)
        except Exception:
            logger.debug("get_run_events_after failed for %s", job_id, exc_info=True)
            new_rows = []

        for row in new_rows:
            try:
                event = AgentEvent(type=row.get("event_type", "status"), data=row.get("data") or {})
            except Exception:
                continue
            row_id = row.get("id")
            yield (event, row_id if isinstance(row_id, int) else None)
            yielded_anything_this_cycle = True
            if isinstance(row_id, int) and row_id > last_seen_event_id:
                last_seen_event_id = row_id
            if event.type in terminal_types:
                return

        try:
            run = db.get_run(job_id)
        except Exception:
            run = None
        if run and run.get("status") in ("completed", "failed"):
            try:
                trailing = db.get_run_events_after(job_id, last_seen_event_id)
            except Exception:
                trailing = []
            for row in trailing:
                try:
                    event = AgentEvent(type=row.get("event_type", "status"), data=row.get("data") or {})
                    row_id = row.get("id")
                    yield (event, row_id if isinstance(row_id, int) else None)
                except Exception:
                    pass
            return

        if not yielded_anything_this_cycle:
            time.sleep(poll_interval)


def drain_events_with_heartbeat(
    job_id: str,
    timeout: float = 3600.0,
    heartbeat_interval: float = 15.0,
    after_id: int = 0,
):
    """Like ``drain_events`` but yields ``None`` as a heartbeat sentinel
    whenever ``heartbeat_interval`` seconds elapse without a real event.

    SSE endpoints translate ``None`` into a comment line (``": keepalive\\n\\n"``)
    which keeps Cloudflare / nginx / Fly proxies from killing the
    connection due to idle-read timeouts.

    Yields either ``None`` (keepalive) or ``(AgentEvent, row_id | None)``.
    """
    from backend.src.core.events import AgentEvent

    deadline = time.time() + timeout
    last_seen_event_id = after_id
    poll_interval = 0.5
    terminal_types = ("done", "error")
    last_yield_time = time.time()

    while time.time() < deadline:
        yielded_anything_this_cycle = False

        try:
            new_rows = db.get_run_events_after(job_id, last_seen_event_id)
        except Exception:
            logger.debug("get_run_events_after failed for %s", job_id, exc_info=True)
            new_rows = []

        for row in new_rows:
            try:
                event = AgentEvent(type=row.get("event_type", "status"), data=row.get("data") or {})
            except Exception:
                continue
            row_id = row.get("id")
            yield (event, row_id if isinstance(row_id, int) else None)
            last_yield_time = time.time()
            yielded_anything_this_cycle = True
            if isinstance(row_id, int) and row_id > last_seen_event_id:
                last_seen_event_id = row_id
            if event.type in terminal_types:
                return

        try:
            run = db.get_run(job_id)
        except Exception:
            run = None
        if run and run.get("status") in ("completed", "failed"):
            try:
                trailing = db.get_run_events_after(job_id, last_seen_event_id)
            except Exception:
                trailing = []
            for row in trailing:
                try:
                    event = AgentEvent(type=row.get("event_type", "status"), data=row.get("data") or {})
                    row_id = row.get("id")
                    yield (event, row_id if isinstance(row_id, int) else None)
                except Exception:
                    pass
            return

        if time.time() - last_yield_time >= heartbeat_interval:
            yield None
            last_yield_time = time.time()

        if not yielded_anything_this_cycle:
            time.sleep(poll_interval)


def sse_stream(
    job_id: str,
    timeout: float = 3600.0,
    heartbeat_interval: float = 15.0,
    after_id: int = 0,
):
    """Format ``drain_events_with_heartbeat`` output as SSE lines.

    Each AgentEvent is serialized with its SQLite row id stamped in as
    ``_event_id`` (when available) so the frontend can resume on
    reconnect via the ``after_id`` query param.
    """
    for item in drain_events_with_heartbeat(
        job_id,
        timeout=timeout,
        heartbeat_interval=heartbeat_interval,
        after_id=after_id,
    ):
        if item is None:
            yield ": keepalive\n\n"
            continue
        event, row_id = item
        payload = event.model_dump()
        if row_id is not None:
            payload["_event_id"] = row_id
        yield f"data: {json.dumps(payload)}\n\n"


def run_in_background(
    job_id: str,
    target: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
) -> threading.Thread:
    """Execute an agent function in a background thread."""

    def _wrapper():
        with _LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["status"] = "running"
                _JOBS[job_id]["start_time"] = time.time()

        try:
            result = target(*args, **(kwargs or {}))
            set_status(job_id, "completed", output=str(result) if result else None)
        except Exception as e:
            logger.exception("Job %s failed", job_id)
            set_status(job_id, "failed", error=str(e))
            from backend.src.core.events import error_event
            emit_event(job_id, error_event(str(e)))

    thread = threading.Thread(target=_wrapper, daemon=True, name=f"job-{job_id[:8]}")
    thread.start()
    return thread
