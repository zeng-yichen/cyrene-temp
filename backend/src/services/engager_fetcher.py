"""Engager fetcher — pulls reactor/commenter profiles for a LinkedIn post.

Uses Apify's API Maestro actors:
  - apimaestro/linkedin-post-reactions  (reactions/likes)
  - apimaestro/linkedin-post-comments-replies-engagements-scraper-no-cookies  (comments)

Results are persisted in the local SQLite post_engagers table to avoid
re-fetching on every sync cycle.

Cost-optimised defaults:
  - Reactions only (comments actor skipped — 83% of engagers are reactors,
    comments return nothing 23% of the time and only ~10 results otherwise)
  - Capped at 30 results per post (enough for reliable LLM-based ICP scoring,
    diminishing returns past ~20 headlines)
  - Posts with <10 known reactions are skipped by the caller (noisy signal)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_APIFY_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
_APIFY_BASE = "https://api.apify.com/v2/acts"
_REACTIONS_ACTOR = "apimaestro~linkedin-post-reactions"
_COMMENTS_ACTOR = "apimaestro~linkedin-post-comments-replies-engagements-scraper-no-cookies"
_TIMEOUT = 120

# Cost control: 30 headlines is enough for reliable ICP scoring.
# At $5/1000 results, this caps each fetch at $0.15 instead of $0.25-0.50.
DEFAULT_REACTION_LIMIT = 30


def _run_actor(actor_id: str, payload: dict, engagement_type: str) -> list[dict]:
    """Run an Apify actor synchronously and return normalized engager dicts."""
    import httpx

    url = f"{_APIFY_BASE}/{actor_id}/run-sync-get-dataset-items"
    params = {"format": "json", "token": _APIFY_TOKEN}

    try:
        resp = httpx.post(url, params=params, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("[engager_fetcher] Apify actor %s failed: %s", actor_id.split("~")[-1][:30], e)
        return []

    items = data if isinstance(data, list) else data.get("data", data.get("items", []))
    if not isinstance(items, list):
        return []

    out = []
    for item in items:
        if engagement_type == "reaction":
            reactor = item.get("reactor") or {}
            urn = reactor.get("urn") or ""
            name = reactor.get("name") or ""
            headline = reactor.get("headline") or ""
            profile_url = reactor.get("profile_url") or ""
            # Extract additional fields if available from the API response
            current_company = reactor.get("currentCompany") or reactor.get("company") or ""
            title = reactor.get("title") or ""
            location = reactor.get("location") or ""
        else:
            author = item.get("author") or {}
            urn = author.get("profile_url") or ""
            name = author.get("name") or ""
            headline = author.get("headline") or ""
            profile_url = author.get("profile_url") or ""
            current_company = author.get("currentCompany") or author.get("company") or ""
            title = author.get("title") or ""
            location = author.get("location") or ""

        if not headline and not urn:
            continue

        out.append({
            "urn": str(urn or profile_url),
            "name": name,
            "headline": headline,
            "current_company": current_company,
            "title": title,
            "location": location,
            "engagement_type": engagement_type,
        })

    return out


def fetch_engagers(
    post_url: str,
    fetch_comments: bool = False,
    fetch_reactions: bool = True,
    reaction_limit: int = DEFAULT_REACTION_LIMIT,
) -> list[dict] | None:
    """Return combined list of engager dicts, or None when the API is unreachable/unauthorized.

    Defaults are cost-optimised: reactions-only, capped at 30 results.
    Pass fetch_comments=True to also fetch commenter profiles (doubles cost).
    """
    if not _APIFY_TOKEN:
        logger.warning("[engager_fetcher] APIFY_API_TOKEN not set — skipping engager fetch")
        return None

    result = []
    errors = 0
    total = 0

    if fetch_reactions:
        total += 1
        items = _run_actor(
            _REACTIONS_ACTOR,
            {"post_urls": [post_url], "reaction_type": "ALL", "limit": reaction_limit},
            "reaction",
        )
        if items:
            result.extend(items)
        else:
            errors += 1

    if fetch_comments:
        total += 1
        items = _run_actor(
            _COMMENTS_ACTOR,
            {"postIds": [post_url], "limit": 100},
            "comment",
        )
        if items:
            result.extend(items)
        else:
            errors += 1

    if errors == total and not result:
        return None

    logger.info("[engager_fetcher] Fetched %d engagers for %s", len(result), post_url)
    return result


def fetch_and_persist(
    company: str,
    ordinal_post_id: str,
    linkedin_post_url: str,
    force: bool = False,
    fetch_comments: bool = False,
    fetch_reactions: bool = True,
    reaction_limit: int = DEFAULT_REACTION_LIMIT,
) -> Optional[list[dict]]:
    """Fetch engagers, persist to SQLite, return enriched profile dicts.

    Returns None on failure. Returns cached profiles if already fetched and force=False.
    Each dict includes ``urn`` (for writing ICP scores back) plus headline/name/company/title/location
    suitable for passing directly to icp_scorer.score_engagers_segmented()
    (which returns continuous per-engager scores, no segment buckets).
    """
    from backend.src.db.local import engagers_fetched_for_post, upsert_engagers, get_engagers_for_post

    if not force and engagers_fetched_for_post(ordinal_post_id):
        rows = get_engagers_for_post(ordinal_post_id)
        return [
            {"urn": r.get("engager_urn", ""),
             "headline": r.get("headline", ""), "name": r.get("name", ""),
             "current_company": r.get("current_company", ""), "title": r.get("title", ""),
             "location": r.get("location", "")}
            for r in rows
            if r.get("headline") or r.get("current_company") or r.get("title")
        ]

    engagers = fetch_engagers(
        linkedin_post_url,
        fetch_comments=fetch_comments,
        fetch_reactions=fetch_reactions,
        reaction_limit=reaction_limit,
    )
    if engagers is None:
        return None

    if not engagers:
        return []

    upsert_engagers(company, ordinal_post_id, linkedin_post_url, engagers)
    return [
        {
            "urn": e.get("urn", ""),
            "headline": e.get("headline", ""),
            "name": e.get("name", ""),
            "current_company": e.get("current_company", ""),
            "title": e.get("title", ""),
            "location": e.get("location", ""),
        }
        for e in engagers
        if e.get("headline") or e.get("current_company") or e.get("title")
    ]
