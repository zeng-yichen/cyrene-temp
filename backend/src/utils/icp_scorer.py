"""LLM-native ICP scorer for post engagers (bitter-lesson aligned).

Semantically evaluates engager headlines against raw client transcripts
using Claude Sonnet. Returns a continuous per-engager score in [0, 1]
plus the mean (`icp_match_rate`).

No segment buckets, no hand-tuned category thresholds, no curated ICP
definition JSON. Downstream consumers that want to reason about
"how ICP-ish" an engager is read the raw continuous score directly.

Retired 2026-04-11: the `icp_definition.json` path (a curated prose
summary of who the audience is) has been removed as a Bitter Lesson
violation. The scorer now always derives audience context from the
client's transcripts directly via `_load_icp_context`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def score_engagers_segmented(company: str, headlines: list[dict | str]) -> dict:
    """Score engagers with continuous 0-1 relevance scores.

    The function name is a holdover from the segmented version; it now
    returns raw scores only. No bucketed counts, no category labels, no
    hand-tuned thresholds — consumers read the per-engager scores list
    and the `icp_match_rate` mean directly and decide what matters.

    Returns:
        {
            "score": float,          # mean relevance in [0, 1]
            "scores": list[float],   # per-engager continuous scores
            "icp_match_rate": float, # continuous mean, in [0, 1]
        }
    """
    empty = {
        "score": 0.0,
        "scores": [],
        "icp_match_rate": 0.0,
    }

    if not headlines:
        return empty

    # Audience context is derived from the client's transcripts directly.
    # No curated ICP description — let the model form its own view from
    # the client's own words. Irontomb owns the transcript resolver;
    # re-using it keeps the scorer and simulator aligned on audience
    # context so a post scored high by ICP also faces the same audience
    # framing in Irontomb's panel.
    from backend.src.agents.irontomb import _load_icp_context
    audience_context = _load_icp_context(company)
    if not audience_context:
        return empty

    # Build engager block — supports both string headlines and enriched dicts
    engager_lines = []
    for i, h in enumerate(headlines[:50]):
        if isinstance(h, dict):
            parts = []
            if h.get("headline"):
                parts.append(f"headline: \"{h['headline']}\"")
            if h.get("current_company"):
                parts.append(f"company: {h['current_company']}")
            if h.get("title"):
                parts.append(f"title: {h['title']}")
            if h.get("location"):
                parts.append(f"location: {h['location']}")
            if h.get("name"):
                parts.append(f"name: {h['name']}")
            engager_lines.append(f"{i+1}. {' | '.join(parts)}")
        else:
            engager_lines.append(f"{i+1}. {h}")
    hl_block = "\n".join(engager_lines)

    prompt = (
        "You are scoring LinkedIn post engagers against the client's actual "
        "target audience. Below is the raw source material — transcripts "
        "and context about who the client is, who they talk to, and who "
        "they want to reach. Read it to form your own view of the ICP, "
        "then score each engager against that view.\n\n"
        f"{audience_context}\n\n"
        f"ENGAGER PROFILES:\n{hl_block}\n\n"
        "For each numbered engager, output a continuous relevance score from "
        "0.0 to 1.0. 1.0 means the engager closely matches the target "
        "audience you derived from the source material; 0.0 means they are "
        "clearly off-target. Use all available fields (headline, company, "
        "title, location) to make your assessment. Score continuously — do "
        "not round to a fixed set of points.\n\n"
        "Output ONLY a comma-separated list of scores. No explanation.\n"
        "Example for 4 headlines: 0.91,0.54,0.32,0.08"
    )

    try:
        from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli, cli_single_shot as _cli_ss
        if _use_cli():
            raw_out = _cli_ss(prompt, model="sonnet", max_tokens=250, timeout=120)
            if not raw_out:
                logger.warning("[icp_scorer] CLI returned empty for %s", company)
                return empty
            raw = raw_out.strip()
        else:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        scores = _parse_continuous_scores(raw)
        if not scores:
            return empty
        return _compute_result(scores)
    except Exception as e:
        logger.warning("[icp_scorer] LLM scoring failed for %s: %s", company, e)
        return empty


def score_engagers(company: str, headlines: list[str]) -> float:
    """Backward-compatible: return a single scalar ICP score in [-1, 1].

    Maps the continuous [0, 1] mean to [-1, 1] for RuanMei composite
    reward compatibility: 0.0 → -1.0, 0.5 → 0.0, 1.0 → 1.0.
    """
    result = score_engagers_segmented(company, headlines)
    raw = result["score"]
    return round(raw * 2 - 1, 4)


def _parse_continuous_scores(raw: str) -> list[float]:
    """Parse comma-separated 0.0-1.0 scores from LLM response."""
    parts = raw.replace(" ", "").split(",")
    scores: list[float] = []
    for p in parts:
        p = p.strip()
        try:
            v = float(p)
            scores.append(max(0.0, min(1.0, v)))
        except ValueError:
            # Handle legacy E/A/N/X labels for backward compat with older logs
            if p.upper() == "E" or p in ("+1", "1"):
                scores.append(1.0)
            elif p.upper() == "A":
                scores.append(0.5)
            elif p.upper() == "N" or p == "0":
                scores.append(0.3)
            elif p.upper() == "X" or p == "-1":
                scores.append(0.0)
    return scores


def _compute_result(scores: list[float]) -> dict:
    """Compute the aggregate result from per-engager continuous scores.

    Returns the mean (`icp_match_rate`) and the raw per-engager scores.
    No segment buckets — consumers that want to reason about "how ICP-ish"
    an engager is read the continuous score directly.
    """
    total = len(scores)
    if total == 0:
        return {
            "score": 0.0,
            "scores": [],
            "icp_match_rate": 0.0,
        }

    mean_score = sum(scores) / total
    return {
        "score": round(mean_score, 4),
        "scores": [round(s, 3) for s in scores],
        "icp_match_rate": round(mean_score, 4),
    }


def icp_match_rate(company: str, headlines: list[str]) -> dict:
    """Diagnostic breakdown of ICP scoring for a post's engagers.

    Returns raw per-engager scores and the mean — no bucketed counts.
    """
    result = score_engagers_segmented(company, headlines)
    return {
        "total": len(result["scores"]),
        "score": result["score"],
        "icp_match_rate": result["icp_match_rate"],
        "scores": result["scores"],
        "method": "llm-continuous",
    }
