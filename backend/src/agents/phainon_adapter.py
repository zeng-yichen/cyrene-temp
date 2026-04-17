"""Phainon adapter — wraps phainon.py for the API layer.

Bridges the agent's event_callback into the job_manager SSE queue.
"""

import logging

from backend.src.utils.langfuse_tracing import traced

logger = logging.getLogger(__name__)


def _make_event_callback(job_id: str):
    """Create a callback that forwards agent events into the job's SSE queue."""
    from backend.src.services.job_manager import emit_event
    from backend.src.core.events import (
        thinking_event, tool_call_event, tool_result_event,
        text_delta_event, error_event, compaction_event, status_event,
    )

    _builders = {
        "thinking": lambda d: thinking_event(d.get("text", "")),
        "tool_call": lambda d: tool_call_event(d.get("name", ""), {"summary": d.get("arguments", "")}),
        "tool_result": lambda d: tool_result_event(d.get("name", ""), d.get("result", ""), d.get("is_error", False)),
        "text_delta": lambda d: text_delta_event(d.get("text", "")),
        "error": lambda d: error_event(d.get("message", "")),
        "compaction": lambda _d: compaction_event(),
        "status": lambda d: status_event(d.get("message", "")),
    }

    def callback(event_type: str, data: dict):
        builder = _builders.get(event_type)
        if builder:
            emit_event(job_id, builder(data))

    return callback


@traced(name="phainon.generate_image", kind="generation")
def run_phainon(
    company: str,
    post_text: str,
    model: str = "claude-opus-4-6",
    job_id: str | None = None,
    feedback_instruction: str = "",
    reference_image_path: str | None = None,
) -> str | None:
    """Run Phainon image assembly and emit events if job_id is provided."""
    from pathlib import Path

    from backend.src.agents.phainon import generate_image
    from backend.src.core.events import done_event, status_event
    from backend.src.services.job_manager import emit_event

    event_callback = _make_event_callback(job_id) if job_id else None

    if job_id:
        emit_event(job_id, status_event(f"Starting image assembly for {company}..."))

    result_path = generate_image(
        company,
        post_text,
        model,
        event_callback=event_callback,
        feedback_instruction=feedback_instruction or "",
        reference_image_path=reference_image_path,
    )

    if job_id:
        stem = Path(result_path).stem if result_path else ""
        emit_event(job_id, done_event(result_path or "", image_id=stem or None))
    return result_path
