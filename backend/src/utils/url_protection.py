"""URL protection — prevent LLM URL hallucination.

Replaces real URLs with {{URL_N}} placeholders before injecting into agent
context. Restores originals in the final user-facing output.
"""

import re
from typing import Any

_URL_PATTERN = re.compile(
    r'https?://[^\s<>\[\](){}"\'`,;|\\^]+',
    re.IGNORECASE,
)


def protect_urls(text: str, url_map: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
    """Replace URLs with {{URL_N}} placeholders.

    Returns (cleaned_text, url_map). Pass existing url_map to extend it
    across multiple tool results within the same conversation.
    """
    if url_map is None:
        url_map = {}

    reverse_map = {v: k for k, v in url_map.items()}

    def _replace(match: re.Match) -> str:
        url = match.group(0)
        if url in reverse_map:
            return reverse_map[url]
        placeholder = f"{{{{URL_{len(url_map) + 1}}}}}"
        url_map[placeholder] = url
        reverse_map[url] = placeholder
        return placeholder

    cleaned = _URL_PATTERN.sub(_replace, text)
    return cleaned, url_map


def restore_urls(text: str, url_map: dict[str, str]) -> str:
    """Restore {{URL_N}} placeholders to their original URLs."""
    result = text
    for placeholder, url in url_map.items():
        result = result.replace(placeholder, url)
    return result


def protect_tool_result(result: Any, url_map: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Convenience wrapper for tool results (may be str or other)."""
    text = str(result) if not isinstance(result, str) else result
    return protect_urls(text, url_map)
