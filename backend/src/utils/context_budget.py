"""Context budget management — prevent silent quality degradation.

Tracks token usage per conversation turn and prunes old tool results
when context exceeds a configurable threshold of the model's window.
"""

import logging

import tiktoken

logger = logging.getLogger(__name__)

MODEL_CONTEXT_WINDOWS = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-5-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-haiku-3-5-20241022": 200_000,
    "gemini-3.1-pro-preview": 1_000_000,
    "gemini-2.5-flash-preview-04-17": 1_000_000,
}

DEFAULT_WINDOW = 200_000
THRESHOLD_RATIO = 0.85
PRESERVE_RECENT = 5


def count_tokens(text: str, model: str = "claude-opus-4-6") -> int:
    """Approximate token count. Uses cl100k_base as a reasonable cross-model estimate."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens across a list of message dicts."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += count_tokens(block.get("text", ""))
                else:
                    total += count_tokens(str(block))
        else:
            total += count_tokens(str(content))
    return total


def get_context_window(model: str) -> int:
    return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_WINDOW)


def should_compact(messages: list[dict], model: str) -> bool:
    window = get_context_window(model)
    current = estimate_messages_tokens(messages)
    return current > window * THRESHOLD_RATIO


def compact_messages(messages: list[dict], model: str) -> list[dict]:
    """Remove old tool results while preserving system prompt and recent messages.

    Strategy: keep the system message (index 0), the initial user message,
    and the most recent PRESERVE_RECENT messages. For everything in between,
    replace tool_result content with a summary placeholder.
    """
    if not should_compact(messages, model):
        return messages

    window = get_context_window(model)
    target = int(window * 0.70)

    if len(messages) <= PRESERVE_RECENT + 2:
        return messages

    preserved_head = messages[:2]
    preserved_tail = messages[-PRESERVE_RECENT:]
    middle = messages[2:-PRESERVE_RECENT]

    compacted_middle = []
    for msg in middle:
        role = msg.get("role", "")
        if role == "tool" or (role == "user" and isinstance(msg.get("content"), list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in msg.get("content", [])
        )):
            compacted_middle.append({
                "role": role,
                "content": "[Earlier tool result removed during context compaction]",
            })
        else:
            compacted_middle.append(msg)

    result = preserved_head + compacted_middle + preserved_tail

    current_tokens = estimate_messages_tokens(result)
    if current_tokens > target:
        result = preserved_head + [
            {"role": "user", "content": f"[{len(compacted_middle)} earlier messages compacted to save context]"}
        ] + preserved_tail
        logger.info(
            "Aggressive compaction: dropped %d middle messages, %d -> %d tokens",
            len(compacted_middle),
            current_tokens,
            estimate_messages_tokens(result),
        )

    return result
