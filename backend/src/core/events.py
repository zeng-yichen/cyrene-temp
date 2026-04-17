"""Structured event protocol for agent communication.

All agents emit AgentEvents into a queue.Queue. The SSE endpoint drains the
queue and yields text/event-stream lines to the frontend.
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    type: Literal[
        "thinking",
        "tool_call",
        "tool_result",
        "text_delta",
        "error",
        "done",
        "compaction",
        "status",
    ]
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


def thinking_event(text: str) -> AgentEvent:
    return AgentEvent(type="thinking", data={"text": text})


def tool_call_event(name: str, args: dict[str, Any] | None = None) -> AgentEvent:
    return AgentEvent(type="tool_call", data={"name": name, "arguments": args or {}})


def tool_result_event(name: str, result: str, is_error: bool = False) -> AgentEvent:
    return AgentEvent(type="tool_result", data={"name": name, "result": result, "is_error": is_error})


def text_delta_event(text: str) -> AgentEvent:
    return AgentEvent(type="text_delta", data={"text": text})


def error_event(message: str) -> AgentEvent:
    return AgentEvent(type="error", data={"message": message})


def done_event(output: str | None = None, image_id: str | None = None) -> AgentEvent:
    data: dict = {"output": output or ""}
    if image_id:
        data["image_id"] = image_id
    return AgentEvent(type="done", data=data)


def status_event(message: str) -> AgentEvent:
    return AgentEvent(type="status", data={"message": message})


def compaction_event() -> AgentEvent:
    return AgentEvent(type="compaction", data={"message": "Context compaction in progress..."})
