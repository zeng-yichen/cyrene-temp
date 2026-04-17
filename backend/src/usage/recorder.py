"""Write a usage_events row.

Kept intentionally dumb: no retries, no buffering, no async. SQLite with
WAL mode is plenty for the write volume we expect (hundreds of LLM calls
per day, maybe low thousands at peak). Every call site that wants to log
usage goes through :func:`record_usage_event`.

The function NEVER raises. LLM call-site instrumentation must not
propagate recorder failures back into the user's request — if SQLite is
down, we'd rather lose accounting than break the call.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from backend.src.db.local import get_connection
from backend.src.usage.pricing import price_call

logger = logging.getLogger("amphoreus.usage.recorder")

Provider = Literal["anthropic", "anthropic_cli", "openai", "perplexity"]
CallKind = Literal["messages", "embeddings", "chat"]


def record_usage_event(
    *,
    provider: Provider,
    model: str,
    call_kind: CallKind,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    user_email: str | None = None,
    client_slug: str | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
) -> None:
    """Insert one row into ``usage_events``. Never raises.

    Cost is computed here from the token counts so the caller only has
    to pass raw usage from the SDK response.
    """
    try:
        cost_usd = price_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        )
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO usage_events (
                    provider, model, call_kind,
                    user_email, client_slug,
                    input_tokens, output_tokens,
                    cache_creation_tokens, cache_read_tokens,
                    cost_usd, duration_ms, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider,
                    model,
                    call_kind,
                    user_email,
                    client_slug,
                    int(input_tokens or 0),
                    int(output_tokens or 0),
                    int(cache_creation_tokens or 0),
                    int(cache_read_tokens or 0),
                    float(cost_usd),
                    int(duration_ms) if duration_ms is not None else None,
                    error,
                    time.time(),
                ),
            )
    except Exception:
        logger.exception(
            "usage.recorder: failed to insert usage event for model=%s user=%s",
            model,
            user_email or "<anon>",
        )
