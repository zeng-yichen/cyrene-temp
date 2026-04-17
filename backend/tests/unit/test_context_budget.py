"""Tests for context budget management."""

from backend.src.utils.context_budget import count_tokens, should_compact, compact_messages


def test_count_tokens_basic():
    tokens = count_tokens("Hello, world!")
    assert tokens > 0
    assert tokens < 20


def test_should_compact_under_threshold():
    messages = [{"content": "short message"}]
    assert not should_compact(messages, "claude-opus-4-6")


def test_should_compact_over_threshold():
    long_text = "word " * 50000
    messages = [{"content": long_text}]
    assert should_compact(messages, "claude-opus-4-6")


def test_compact_preserves_system_and_recent():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First message"},
        {"role": "tool", "content": "Old tool result 1"},
        {"role": "tool", "content": "Old tool result 2"},
        {"role": "tool", "content": "Old tool result 3"},
        {"role": "user", "content": "Recent 1"},
        {"role": "assistant", "content": "Recent 2"},
        {"role": "user", "content": "Recent 3"},
        {"role": "assistant", "content": "Recent 4"},
        {"role": "user", "content": "Recent 5"},
    ]

    compacted = compact_messages(messages, "claude-opus-4-6")

    assert compacted[0]["role"] == "system"
    assert compacted[1]["content"] == "First message"
    assert compacted[-1]["content"] == "Recent 5"
