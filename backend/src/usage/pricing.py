"""Static price table for LLM providers.

Prices are per-million-tokens in USD. **Edit these numbers when provider
pricing changes** — the cost column on existing ``usage_events`` rows is
computed at record-time and is NOT backfilled when this table is edited.
If you need to recompute historical cost, run a one-off SQL migration.

Coverage is best-effort. If a model is not in the table, ``price_call``
returns ``0.0`` and logs a warning — the usage event is still recorded
with its token counts, so you can backfill cost later.

Sources:
    - Anthropic: https://www.anthropic.com/pricing
    - OpenAI:    https://openai.com/api/pricing/
    - Perplexity: https://docs.perplexity.ai/guides/pricing

Prices below are approximations as of 2026-04-11 — UPDATE BEFORE TRUSTING
SPEND NUMBERS. The pricing table lives in one place so you can sanity-check
all of it at once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("amphoreus.usage.pricing")

_M = 1_000_000  # tokens per million


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token pricing. Missing fields default to 0."""

    input_per_m: float
    output_per_m: float
    cache_write_per_m: float = 0.0
    cache_read_per_m: float = 0.0


# Keys are matched case-insensitively via .lower() — the SDK sometimes
# returns the full versioned model name (claude-sonnet-4-5-20250929) and
# sometimes a shorthand alias (claude-sonnet-4-5). Both should resolve.
PRICES: dict[str, ModelPrice] = {
    # --- Anthropic ---
    # Claude 4.x family (approximations — verify before trusting)
    "claude-opus-4-6":              ModelPrice(15.0, 75.0, 18.75, 1.50),
    "claude-opus-4-5":              ModelPrice(15.0, 75.0, 18.75, 1.50),
    "claude-opus-4-20250514":       ModelPrice(15.0, 75.0, 18.75, 1.50),
    "claude-sonnet-4-6":            ModelPrice(3.0, 15.0, 3.75, 0.30),
    "claude-sonnet-4-5":            ModelPrice(3.0, 15.0, 3.75, 0.30),
    "claude-sonnet-4-20250514":     ModelPrice(3.0, 15.0, 3.75, 0.30),
    "claude-haiku-4-5":             ModelPrice(0.80, 4.0, 1.00, 0.08),
    # Claude 3.x legacy (still referenced in some call sites)
    "claude-3-5-sonnet-20241022":   ModelPrice(3.0, 15.0, 3.75, 0.30),
    "claude-3-5-haiku-20241022":    ModelPrice(0.80, 4.0, 1.00, 0.08),
    "claude-3-opus-20240229":       ModelPrice(15.0, 75.0, 18.75, 1.50),

    # --- OpenAI embeddings ---
    "text-embedding-3-small":       ModelPrice(0.02, 0.0),
    "text-embedding-3-large":       ModelPrice(0.13, 0.0),
    "text-embedding-ada-002":       ModelPrice(0.10, 0.0),

    # --- Perplexity ---
    # "sonar" is the default fast model; online models cost more.
    "sonar":                        ModelPrice(1.0, 1.0),
    "sonar-pro":                    ModelPrice(3.0, 15.0),
    "sonar-reasoning":              ModelPrice(1.0, 5.0),
}


def _lookup(model: str) -> ModelPrice | None:
    """Case-insensitive exact match, with a prefix-fallback for aliases."""
    if not model:
        return None
    key = model.strip().lower()
    if key in PRICES:
        return PRICES[key]
    # Fallback: find the longest prefix match. Lets "claude-sonnet-4-6-preview"
    # resolve to "claude-sonnet-4-6" without adding a new row.
    candidates = [k for k in PRICES if key.startswith(k)]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return PRICES[best]


def price_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Return cost in USD for the given token mix. Never raises.

    Unknown models return ``0.0`` and log a warning so the operator knows
    to add a row to ``PRICES``. Negative token counts are coerced to 0 —
    we'd rather undercount than crash the LLM call-site.
    """
    row = _lookup(model)
    if row is None:
        logger.warning("usage.pricing: unknown model %r — cost set to 0", model)
        return 0.0
    i = max(0, int(input_tokens or 0))
    o = max(0, int(output_tokens or 0))
    cw = max(0, int(cache_creation_tokens or 0))
    cr = max(0, int(cache_read_tokens or 0))
    return round(
        (i * row.input_per_m
         + o * row.output_per_m
         + cw * row.cache_write_per_m
         + cr * row.cache_read_per_m) / _M,
        6,
    )
