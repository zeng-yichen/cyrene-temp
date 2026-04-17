"""Tests for the structured event protocol."""

from backend.src.core.events import (
    AgentEvent,
    done_event,
    error_event,
    text_delta_event,
    thinking_event,
    tool_call_event,
    tool_result_event,
)


def test_agent_event_serialization():
    event = AgentEvent(type="thinking", data={"text": "hello"})
    dumped = event.model_dump_json()
    assert "thinking" in dumped
    assert "hello" in dumped


def test_event_constructors():
    assert thinking_event("test").type == "thinking"
    assert tool_call_event("net_query", {"q": "test"}).type == "tool_call"
    assert tool_result_event("net_query", "results").type == "tool_result"
    assert text_delta_event("hello").type == "text_delta"
    assert error_event("fail").type == "error"
    assert done_event("output").type == "done"


def test_event_has_timestamp():
    event = thinking_event("test")
    assert event.timestamp > 0
