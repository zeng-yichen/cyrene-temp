"""Tribbie adapter — wraps tribbie.py for the API layer."""

import logging

from backend.src.utils.langfuse_tracing import traced

logger = logging.getLogger(__name__)


@traced(name="tribbie.start_session", kind="generation")
def run_tribbie_session(company: str, job_id: str) -> None:
    """Start a Tribbie live capture session, bridging events to job_manager."""
    from backend.src.agents.tribbie import start_session
    from backend.src.core.events import (
        done_event,
        error_event,
        status_event,
        text_delta_event,
        tool_result_event,
    )
    from backend.src.services.job_manager import emit_event

    def event_callback(event_type: str, data: dict) -> None:
        if event_type == "status":
            emit_event(job_id, status_event(data["message"]))
        elif event_type == "text_delta":
            emit_event(job_id, text_delta_event(data["text"]))
        elif event_type == "tool_result":
            emit_event(job_id, tool_result_event(data["name"], data["result"]))
        elif event_type == "error":
            emit_event(job_id, error_event(data["message"]))
        elif event_type == "done":
            emit_event(job_id, done_event(data.get("output") or data.get("message")))

    start_session(company, job_id, event_callback)
