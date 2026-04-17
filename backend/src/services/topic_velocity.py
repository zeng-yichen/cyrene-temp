"""Topic Velocity — proactive industry trend monitoring via Perplexity.

Queries Perplexity (sonar model with web search) for each client's
industry trends, producing a markdown file at
``memory/{company}/topic_velocity.md`` that Stelle reads as generation context.

Bitter-lesson aligned: we expand the *observation space* (what the model sees)
rather than prescribing strategy from trends.
"""

import json
import logging
import os
from datetime import datetime, timezone

import httpx

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)


def _gather_context(company: str) -> str:
    """Gather raw client context to inform the trend search."""
    parts: list[str] = []

    icp_path = P.MEMORY_ROOT / company / "icp_definition.json"
    if icp_path.exists():
        try:
            icp = json.loads(icp_path.read_text(encoding="utf-8"))
            desc = icp.get("description", "")
            anti = icp.get("anti_description", "")
            if desc:
                parts.append(f"ICP: {desc}")
            if anti:
                parts.append(f"Anti-ICP: {anti}")
        except Exception:
            pass

    strategy_dir = P.content_strategy_dir(company)
    if strategy_dir.exists():
        for f in sorted(strategy_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.suffix in (".md", ".txt") and f.stat().st_size < 50_000:
                try:
                    parts.append(f.read_text(encoding="utf-8", errors="replace")[:2000])
                except Exception:
                    pass
                break

    transcripts_dir = P.transcripts_dir(company)
    if transcripts_dir.exists():
        for f in sorted(transcripts_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:1]:
            if f.suffix in (".md", ".txt"):
                try:
                    parts.append(f.read_text(encoding="utf-8", errors="replace")[:1500])
                except Exception:
                    pass

    return "\n---\n".join(parts)


def refresh_topic_velocity(company: str) -> str:
    """Fetch current industry signal via Perplexity and write topic_velocity.md.

    Returns the generated markdown, or empty string on failure.
    """
    context = _gather_context(company)
    if not context.strip():
        context = company.replace("-", " ")

    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        logger.warning("[topic_velocity] PERPLEXITY_API_KEY not set — skipping")
        return ""

    prompt = (
        f"You are monitoring industry trends for a LinkedIn thought leader.\n\n"
        f"CLIENT CONTEXT:\n{context[:3000]}\n\n"
        f"Search the web for the 8-12 most important recent news stories, developments, "
        f"and trends relevant to this person's industry and audience. Focus on:\n"
        f"- Breaking news and announcements\n"
        f"- Industry shifts and emerging trends\n"
        f"- Regulatory or market changes\n"
        f"- Notable reports or research\n\n"
        f"For each item, provide:\n"
        f"- A bold title\n"
        f"- A 1-2 sentence summary\n"
        f"- The source URL if available\n\n"
        f"Output as a clean markdown list. Be specific and factual."
    )

    try:
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 2000,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        body = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        if not body:
            logger.warning("[topic_velocity] Perplexity returned empty response for %s", company)
            return ""
    except Exception as e:
        logger.warning("[topic_velocity] Perplexity search failed for %s: %s", company, e)
        return ""

    md = (
        f"# Topic Velocity — {company}\n"
        f"_Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
        f"Recent industry signal (for context, not prescription):\n\n"
        f"{body}\n"
    )

    out_path = P.MEMORY_ROOT / company / "topic_velocity.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    logger.info("[topic_velocity] Wrote trends to %s", out_path)

    return md
