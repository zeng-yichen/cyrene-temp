"""Stelle adapter — wraps stelle.py for the API layer.

Bridges the agent's event_callback into the job_manager SSE queue,
and passes through prompt/model from the API layer.
"""

import logging
import os

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


@traced(name="stelle.run_writer_batch", kind="generation")
def run_stelle(
    company: str,
    prompt: str | None = None,
    model: str = "claude-opus-4-6",
    job_id: str | None = None,
) -> str | None:
    """Run Stelle ghostwriter and emit events if job_id is provided."""
    from backend.src.agents.stelle import run_writer_batch
    from backend.src.db import vortex

    vortex.ensure_dirs(company)
    output_dir = vortex.post_dir(company)
    output_filepath = str(output_dir / f"{company}_posts.md")

    event_callback = _make_event_callback(job_id) if job_id else None

    if job_id:
        from backend.src.services.job_manager import emit_event
        from backend.src.core.events import status_event
        emit_event(job_id, status_event(f"Starting Stelle generation for {company}..."))

    result_path = run_writer_batch(
        company, company, output_filepath,
        prompt=prompt, model=model, event_callback=event_callback,
    )

    if result_path and os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            return f.read()
    return result_path


@traced(name="stelle.edit_single_post", kind="generation")
def run_inline_edit(
    company: str,
    post_text: str,
    instruction: str,
    job_id: str | None = None,
) -> str:
    """Run Stelle inline edit with optional SSE streaming."""
    event_callback = _make_event_callback(job_id) if job_id else None
    try:
        from backend.src.agents.stelle import edit_single_post
        result = edit_single_post(company, post_text, instruction, event_callback=event_callback)
        return result or ""
    except (ImportError, AttributeError):
        from backend.src.agents.demiurge import Cyrene
        cyrene = Cyrene()
        result = cyrene.rewrite_single_post(post_text, instruction)
        return result.get("final_post", "")
