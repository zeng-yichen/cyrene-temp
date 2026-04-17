"""Cyrene — strategic growth agent.

The outermost loop in the Amphoreus content pipeline. Every other agent
operates within a single generation run (Stelle drafts posts, Irontomb
simulates audience reactions). Cyrene operates ACROSS runs, across
interviews, across months — studying what happened, identifying what to
do next, and producing a strategic brief that shapes the entire
operation.

## Objective

Maximize the client's ICP exposure and pipeline generation on LinkedIn
over time. Three layers, in order of importance:

  1. Pipeline: engagement from ICP prospects → conversations → deals
  2. ICP exposure: each successive batch attracts more of the right people
  3. Engagement: posts perform well (the base layer that enables 1 and 2)

## Architecture

Turn-based tool-calling agent, same skeleton as Irontomb and Stelle.
Runs on demand (the operator triggers it when they want a strategy
review), produces a JSON strategic brief, self-schedules the next run.

Uses Opus for deep strategic reasoning. Runs infrequently ($5-10 per
run, maybe twice a month per client) so cost is not the bottleneck.

## Output

A JSON strategic brief with: interview questions, asset requests,
content priorities, content avoidance, ABM targets, DM-ready warm
prospects, Stelle scheduling recommendation, ICP exposure assessment,
and a self-schedule trigger for the next Cyrene run.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anthropic

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CYRENE_MODEL = "claude-opus-4-6"
_CYRENE_MAX_TOKENS = 4096
_CYRENE_MAX_TURNS = 40

# Opus 4.6 pricing per million tokens
_INPUT_COST_PER_MTOK = 15.0
_OUTPUT_COST_PER_MTOK = 75.0
_CACHE_READ_COST_PER_MTOK = 1.50
_CACHE_WRITE_COST_PER_MTOK = 18.75

_BRIEF_FILENAME = "cyrene_brief.json"
_BRIEF_HISTORY_FILENAME = "cyrene_brief_history.jsonl"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# Fields produced by observation_tagger.py for DASHBOARD DISPLAY only.
# Cyrene must not condition strategic recommendations on these because
# they're a hand-designed taxonomy — a Bitter Lesson trap. She reasons
# from post text + engagement + reactor identity, not from human buckets.
_DISPLAY_ONLY_FIELDS = ("topic_tag", "source_segment_type", "format_tag")


def _strip_display_tags(obs_list: list[dict]) -> list[dict]:
    out = []
    for o in obs_list:
        o2 = {k: v for k, v in o.items() if k not in _DISPLAY_ONLY_FIELDS}
        out.append(o2)
    return out


def _load_cyrene_observations(company: str) -> list[dict]:
    """Scored/finalized observations with display-only tags stripped."""
    from backend.src.db.local import ruan_mei_load
    state = ruan_mei_load(company) or {}
    obs = [
        o for o in state.get("observations", [])
        if o.get("status") in ("scored", "finalized")
    ]
    return _strip_display_tags(obs)


def _query_observations(company: str, args: dict) -> str:
    """Reuse the shared observation query tool from the analyst module."""
    try:
        from backend.src.agents.analyst import _tool_query_observations
        # Also forbid filtering BY the tags Cyrene can't see.
        args = {
            k: v for k, v in (args or {}).items()
            if k not in ("topic_filter", "format_filter")
        }
        return _tool_query_observations(args, _load_cyrene_observations(company))
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})


def _query_top_engagers(company: str, args: dict) -> str:
    """Reuse the shared top-engagers query."""
    try:
        from backend.src.db.local import get_top_icp_engagers
        limit = min(args.get("limit", 30), 50)
        engagers = get_top_icp_engagers(company, limit=limit)
        return json.dumps({"count": len(engagers), "engagers": engagers}, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})


def _query_transcript_inventory(company: str, args: dict) -> str:
    """List transcripts + story inventory, or read one transcript's text.

    Transcripts are the single source of truth for anything the client
    has said — story material AND offline pipeline signals (DMs,
    meetings, deals) mentioned in passing on a call. Pass
    ``read_filename`` to get the raw text of one file.
    """
    transcripts_dir = P.memory_dir(company) / "transcripts"
    read_filename = (args.get("read_filename") or "").strip()

    if read_filename:
        # Prevent path traversal — only files directly under transcripts/
        safe_name = Path(read_filename).name
        target = transcripts_dir / safe_name
        if not target.exists() or not target.is_file():
            return json.dumps({"error": f"transcript not found: {safe_name}"})
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return json.dumps({"error": f"read failed: {str(e)[:200]}"})
        return json.dumps({
            "filename": safe_name,
            "n_chars": len(text),
            "text": text[:20000],
            "truncated": len(text) > 20000,
        })

    transcript_list: list[dict] = []
    if transcripts_dir.exists():
        for f in sorted(transcripts_dir.iterdir()):
            if f.is_file() and f.suffix in (".txt", ".md", ".json", ".pdf", ".docx"):
                try:
                    size_kb = round(f.stat().st_size / 1024, 1)
                    mtime = datetime.fromtimestamp(
                        f.stat().st_mtime, tz=timezone.utc
                    ).isoformat()[:19]
                except Exception:
                    size_kb = 0
                    mtime = ""
                transcript_list.append({
                    "filename": f.name,
                    "size_kb": size_kb,
                    "modified": mtime,
                })

    story_inventory = ""
    workspace_inv = P.workspace_dir(company) / "memory" / "story-inventory.md"
    if workspace_inv.exists():
        try:
            story_inventory = workspace_inv.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

    return json.dumps({
        "transcripts": transcript_list,
        "n_transcripts": len(transcript_list),
        "story_inventory": story_inventory[:8000] if story_inventory else "(no story inventory yet)",
    }, default=str)


def _query_icp_exposure_trend(company: str, args: dict) -> str:
    """Compute icp_match_rate averaged per approximate batch over time.

    Groups posts by week (Mon-Sun), computes mean icp_match_rate and
    mean engagement per group. Returns chronological list showing whether
    ICP targeting is improving across successive batches.
    """
    try:
        from backend.src.db.local import ruan_mei_load
        state = ruan_mei_load(company) or {}
    except Exception:
        return json.dumps({"error": "could not load observations"})

    obs = [
        o for o in state.get("observations", [])
        if o.get("status") in ("scored", "finalized") and o.get("posted_at")
    ]
    if not obs:
        return json.dumps({"error": "no scored observations with posted_at"})

    # Parse posted_at into weeks
    weekly: dict[str, list[dict]] = {}
    for o in obs:
        try:
            dt = datetime.fromisoformat(
                o["posted_at"].replace("Z", "+00:00")
            )
            # ISO week key: YYYY-WNN
            week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        except Exception:
            continue
        weekly.setdefault(week_key, []).append(o)

    trend: list[dict] = []
    for week in sorted(weekly):
        posts = weekly[week]
        icp_rates = [
            o["icp_match_rate"] for o in posts
            if isinstance(o.get("icp_match_rate"), (int, float))
        ]
        raw_metrics = [
            (o.get("reward") or {}).get("raw_metrics") or {}
            for o in posts
        ]
        impressions = [m.get("impressions", 0) for m in raw_metrics]
        reactions = [m.get("reactions", 0) for m in raw_metrics]

        trend.append({
            "week": week,
            "n_posts": len(posts),
            "mean_icp_match_rate": round(
                sum(icp_rates) / len(icp_rates), 4
            ) if icp_rates else None,
            "mean_impressions": round(
                sum(impressions) / len(impressions), 1
            ) if impressions else 0,
            "mean_reactions": round(
                sum(reactions) / len(reactions), 1
            ) if reactions else 0,
        })

    return json.dumps({
        "company": company,
        "n_weeks": len(trend),
        "trend": trend,
    }, default=str)


def _query_warm_prospects(company: str, args: dict) -> str:
    """Cross-reference reactor pool against ABM targets. Return warm leads.

    Warm = engaged with min_engagements posts AND has a non-trivial
    ICP score. Cross-references against abm_profiles/ if present.
    """
    min_eng = args.get("min_engagements", 2)
    try:
        from backend.src.db.local import get_top_icp_engagers
        all_engagers = get_top_icp_engagers(company, limit=200)
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})

    warm = [
        e for e in all_engagers
        if (e.get("engagement_count") or 0) >= min_eng
    ]

    # Load ABM target names if available
    abm_names: set[str] = set()
    abm_dir = P.memory_dir(company) / "abm_profiles"
    if abm_dir.exists():
        for f in abm_dir.iterdir():
            if f.is_file():
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")[:500]
                    # Extract names from ABM profile filenames or content
                    abm_names.add(f.stem.lower().replace("-", " ").replace("_", " "))
                except Exception:
                    pass

    for w in warm:
        name_lower = (w.get("name") or "").lower()
        company_lower = (w.get("current_company") or "").lower()
        w["is_abm_target"] = any(
            abn in name_lower or abn in company_lower
            for abn in abm_names
        ) if abm_names else False

    return json.dumps({
        "min_engagements": min_eng,
        "n_warm_prospects": len(warm),
        "prospects": warm[:50],
        "n_abm_profiles_loaded": len(abm_names),
    }, default=str)


def _query_engagement_trajectories(company: str, args: dict) -> str:
    """Return posts ranked by trajectory metrics.

    Sort by velocity_first_6h, longevity_ratio, peak_velocity, etc.
    Only includes posts that have trajectory data computed.
    """
    sort_by = args.get("sort_by", "velocity_first_6h")
    limit = min(args.get("limit", 10), 20)

    try:
        from backend.src.db.local import ruan_mei_load
        state = ruan_mei_load(company) or {}
    except Exception:
        return json.dumps({"error": "could not load observations"})

    obs = [
        o for o in state.get("observations", [])
        if o.get("status") in ("scored", "finalized") and o.get("trajectory")
    ]

    if not obs:
        return json.dumps({
            "error": "no observations with trajectory data yet (need more engagement snapshots)",
            "hint": "fast sync captures trajectory every 15 min for posts <72h old",
        })

    def _sort_key(o: dict) -> float:
        traj = o.get("trajectory") or {}
        val = traj.get(sort_by)
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    ranked = sorted(obs, key=_sort_key, reverse=True)[:limit]

    results = []
    for o in ranked:
        traj = o.get("trajectory") or {}
        raw = (o.get("reward") or {}).get("raw_metrics") or {}
        body = (o.get("posted_body") or o.get("post_body") or "")
        hook = body.split("\n")[0][:120] if body else ""
        results.append({
            "ordinal_post_id": o.get("ordinal_post_id", ""),
            "posted_at": o.get("posted_at", ""),
            "hook": hook,
            "impressions": raw.get("impressions", 0),
            "reactions": raw.get("reactions", 0),
            "icp_match_rate": o.get("icp_match_rate"),
            "trajectory": {
                k: v for k, v in traj.items()
                if k != "insufficient_data"
            },
        })

    return json.dumps({
        "sorted_by": sort_by,
        "returned": len(results),
        "posts": results,
    }, default=str)


def _execute_python(company: str, args: dict) -> str:
    """Reuse the shared Python execution tool from the analyst module."""
    try:
        from backend.src.agents.analyst import _tool_execute_python
        # Pipe embeddings into the subprocess preamble so Cyrene can
        # operate on raw continuous vectors (cosine sim, PCA, clustering)
        # instead of reaching for hand-engineered feature buckets.
        try:
            from backend.src.utils.post_embeddings import get_post_embeddings
            emb = get_post_embeddings(company)
        except Exception:
            emb = None
        return _tool_execute_python(args, _load_cyrene_observations(company), embeddings=emb)
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})


def _search_linkedin_corpus(company: str, args: dict) -> str:
    """Reuse the shared LinkedIn bank search."""
    try:
        from backend.src.agents.analyst import _tool_search_linkedin_bank
        return _tool_search_linkedin_bank(args)
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})


def _fetch_url(company: str, args: dict) -> str:
    """Resolve a URL to readable plain text.

    For when a transcript mentions a link (e.g. "Sachil sent this article:
    https://...") and Cyrene needs to know what the article actually says
    rather than hallucinating from URL tokens.
    """
    try:
        from backend.src.utils.pull_page import pull_page as _fetch
        url = (args.get("url") or "").strip()
        if not url:
            return json.dumps({"error": "url is required"})
        max_chars = int(args.get("max_chars", 12000))
        result = _fetch(url, max_chars=min(max_chars, 20000))
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"pull_page failed: {str(e)[:200]}"})


def _web_search(company: str, args: dict) -> str:
    """Web search via Parallel API."""
    query = (args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"})
    try:
        import httpx
        api_key = __import__("os").environ.get("PARALLEL_API_KEY", "")
        if not api_key:
            return json.dumps({"error": "PARALLEL_API_KEY not set"})
        resp = httpx.post(
            "https://api.parallel.ai/v1/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": 5},
            timeout=30,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), default=str)[:8000]
    except Exception as e:
        return json.dumps({"error": f"web search failed: {str(e)[:200]}"})


def _query_ordinal_posts(company: str, args: dict) -> str:
    """Return the client's full Ordinal publishing history.

    Unlike ``pull_history`` (which only sees Stelle-drafted posts
    that got matched back via ordinal_sync), this returns every LinkedIn
    post Ordinal has analytics for — including ones the client wrote
    themselves and anything from before Amphoreus started tracking.

    Each post is annotated ``tracked_by_stelle`` so the caller can tell
    Stelle-drafted posts from client-authored ones.
    """
    try:
        from backend.src.services.ordinal_sync import (
            _extract_ordinal_post_id,
            fetch_ordinal_posts_for,
        )
    except Exception as e:
        return json.dumps({"error": f"import failed: {str(e)[:200]}"})

    posts = fetch_ordinal_posts_for(company)
    if posts is None:
        return json.dumps({
            "error": "Ordinal credentials or profile id not found for this client",
        })

    # Build set of ordinal_post_ids tracked as Stelle observations, to
    # tag each Ordinal post with provenance.
    tracked_ids: set[str] = set()
    try:
        from backend.src.db.local import ruan_mei_load
        state = ruan_mei_load(company) or {}
        for o in state.get("observations", []):
            if o.get("status") in ("scored", "finalized") and o.get("ordinal_post_id"):
                tracked_ids.add(str(o["ordinal_post_id"]))
    except Exception:
        pass

    limit = min(int(args.get("limit", 50)), 200)
    min_impr = int(args.get("min_impressions", 0))

    def _url(p: dict) -> str:
        direct = p.get("post_url") or p.get("postUrl") or p.get("linkedin_url") or ""
        if direct:
            return direct
        for k in ("permalink", "linkedinPermalink", "linkedin_permalink", "url"):
            v = p.get(k)
            if v:
                return v
        return ""

    flat: list[dict] = []
    for p in posts:
        text = (
            p.get("commentary") or p.get("text") or p.get("copy")
            or p.get("content") or p.get("post_text") or ""
        ).strip()
        impressions = p.get("impressionCount") or p.get("impressions") or 0
        if impressions < min_impr:
            continue
        oid = _extract_ordinal_post_id(p) or ""
        flat.append({
            "ordinal_post_id": oid,
            "tracked_by_stelle": oid in tracked_ids if oid else False,
            "posted_at": p.get("publishedAt") or p.get("postedAt") or p.get("published_at") or "",
            "linkedin_url": _url(p),
            "text": text[:3000],
            "impressions": impressions,
            "reactions": p.get("likeCount") or p.get("reactions") or p.get("total_reactions") or 0,
            "comments": p.get("commentCount") or p.get("comments") or p.get("total_comments") or 0,
            "reposts": p.get("shareCount") or p.get("repostCount") or p.get("reposts") or 0,
        })

    # Newest first
    flat.sort(key=lambda x: x.get("posted_at") or "", reverse=True)
    flat = flat[:limit]

    n_client = sum(1 for p in flat if not p["tracked_by_stelle"])
    n_stelle = sum(1 for p in flat if p["tracked_by_stelle"])
    return json.dumps({
        "company": company,
        "n_returned": len(flat),
        "n_client_authored": n_client,
        "n_stelle_drafted": n_stelle,
        "posts": flat,
    }, default=str)


def _query_brief_history(company: str, args: dict) -> str:
    """Return the trajectory of all previous Cyrene briefs for this client.

    Each entry includes the timestamp, content_priorities, content_avoid,
    icp_exposure_assessment, and cost. This lets Cyrene see how its own
    recommendations have evolved over time and correlate with outcomes.
    """
    history_path = P.memory_dir(company) / _BRIEF_HISTORY_FILENAME
    if not history_path.exists():
        return json.dumps({
            "n_briefs": 0,
            "briefs": [],
            "note": "No brief history yet. This is the first Cyrene run for this client.",
        })

    briefs: list[dict] = []
    try:
        for line in history_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                briefs.append(json.loads(line))
    except Exception as e:
        return json.dumps({"error": f"failed to read brief history: {str(e)[:200]}"})

    limit = min(int(args.get("limit", 20)), 50)
    # Return most recent first
    briefs = briefs[-limit:][::-1]

    # Slim down to the strategically relevant fields
    slim: list[dict] = []
    for b in briefs:
        slim.append({
            "computed_at": b.get("_computed_at", ""),
            "content_priorities": b.get("content_priorities", []),
            "content_avoid": b.get("content_avoid", []),
            "icp_exposure_assessment": b.get("icp_exposure_assessment", ""),
            "dm_targets_count": len(b.get("dm_targets", [])),
            "abm_targets_count": len(b.get("abm_targets", [])),
            "stelle_timing": b.get("stelle_timing", ""),
            "cost_usd": b.get("_cost_usd", 0),
        })

    return json.dumps({
        "n_briefs": len(slim),
        "briefs": slim,
    }, default=str)


def _note(company: str, args: dict) -> str:
    """Working memory — append to per-run notes list."""
    # Notes are managed by the agent loop, not stored persistently.
    # The agent loop tracks them in a list and passes them back as context.
    return json.dumps({"ok": True})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "pull_history",
        "description": (
            "Query this client's full scored post history. Each observation: "
            "stelle_draft (post_body), client-published version (posted_body), "
            "engagement metrics (impressions/reactions/comments/reposts), "
            "icp_match_rate, per-post reactor list with individual icp_scores. "
            "Filter by min_reward, max_reward. Sort by any metric. "
            "Pass summary_only=true for aggregate stats without full texts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_by": {"type": "string", "default": "posted_at"},
                "limit": {"type": "integer", "default": 10},
                "min_reward": {"type": "number"},
                "max_reward": {"type": "number"},
                "summary_only": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "pull_reactors",
        "description": (
            "Aggregated top engagers across all scored posts, ranked by "
            "ICP fit × engagement frequency. Shows who repeatedly engages "
            "with this client's content and their profile details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 30},
            },
        },
    },
    {
        "name": "query_transcript_inventory",
        "description": (
            "Without arguments: list all interview transcripts (filename, "
            "size, modified date) plus the story inventory showing which "
            "stories have been turned into posts vs are still untapped. "
            "With ``read_filename``: return the raw text of one transcript. "
            "Transcripts are the SINGLE source of truth for anything the "
            "client has said — mine them for both (a) untapped story "
            "material and (b) offline pipeline signals the client "
            "mentioned in passing (DMs from ICP, meetings booked, deals "
            "sourced). Those signals will not appear anywhere else."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "read_filename": {
                    "type": "string",
                    "description": "Name of a specific transcript to read in full.",
                },
            },
        },
    },
    {
        "name": "query_icp_exposure_trend",
        "description": (
            "ICP match rate averaged per week over the client's history. "
            "Shows whether successive batches of posts are attracting more "
            "of the right people. THE STRATEGIC KPI — if this is flat or "
            "declining, the content strategy needs adjustment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "query_warm_prospects",
        "description": (
            "Reactor pool cross-referenced against ABM targets. Returns "
            "people who have engaged with multiple posts, their ICP score, "
            "which posts they engaged with, and whether they're from a named "
            "ABM target account. These are DM-ready warm leads — the client "
            "should reach out to them directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_engagements": {
                    "type": "integer",
                    "default": 2,
                    "description": "Minimum posts engaged with to qualify as warm.",
                },
            },
        },
    },
    {
        "name": "query_engagement_trajectories",
        "description": (
            "Posts ranked by trajectory metrics: velocity_first_6h (how "
            "fast engagement grew initially), longevity_ratio (how much "
            "engagement continued past 24h), peak_velocity_imp_per_h, "
            "time_to_plateau_hours. Shows which content shapes have legs "
            "vs which peak and die. Only available for posts with enough "
            "engagement snapshots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "default": "velocity_first_6h",
                    "description": "Trajectory metric to rank by.",
                },
                "limit": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "run_py",
        "description": (
            "Run arbitrary Python code with scored observations and raw "
            "1536-dim OpenAI embeddings pre-loaded, plus numpy/scipy/"
            "sklearn/pandas. Pre-loaded globals:\n"
            "  • obs — list of scored observation dicts\n"
            "  • embeddings — {post_hash: [1536 floats]}\n"
            "  • emb_matrix — np.array shape (N, 1536), rows aligned to emb_hashes\n"
            "  • emb_hashes — list[str] of post_hash keys\n"
            "  • emb_by_obs — embeddings aligned to obs order (None for missing)\n"
            "Use the vectors directly for cosine similarity, PCA, "
            "clustering, nearest-neighbor lookups — anything continuous. "
            "Print results to stdout."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "scan_corpus",
        "description": (
            "Search the 200K+ LinkedIn post corpus by keyword or semantic "
            "similarity. Use for competitive intelligence — what's working "
            "in adjacent niches that this client could adapt?"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["keyword", "semantic"],
                    "default": "keyword",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "net_query",
        "description": (
            "Search the web for industry news, regulatory updates, "
            "competitive moves — anything timely that could become a hook "
            "for the next batch of posts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "pull_page",
        "description": (
            "Resolve a specific URL to readable plain text. Use when a "
            "transcript mentions a link the client shared (article, report, "
            "tweet) and you need to know what it actually says rather than "
            "inferring from the URL alone. Returns title + body text, "
            "nav/script/footer stripped. Fails gracefully on paywalls or "
            "4xx/5xx with a clean error message — don't fabricate content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {
                    "type": "integer",
                    "default": 12000,
                    "description": "Maximum characters to return (capped at 20000).",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "query_ordinal_posts",
        "description": (
            "Full LinkedIn publishing history for this client from Ordinal "
            "analytics — every post Ordinal has seen, including ones the "
            "client wrote themselves (not drafted by Stelle) and historical "
            "posts from before Amphoreus. Each post is tagged "
            "`tracked_by_stelle` so you can distinguish agency-drafted from "
            "self-authored. pull_history only shows Stelle-drafted "
            "posts; this tool shows the full picture."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50},
                "min_impressions": {
                    "type": "integer",
                    "default": 0,
                    "description": "Skip posts with fewer impressions.",
                },
            },
        },
    },
    {
        "name": "query_brief_history",
        "description": (
            "Return the trajectory of all previous Cyrene briefs for this "
            "client. Shows how your content_priorities, content_avoid, and "
            "ICP exposure assessments have evolved over time. Use this to "
            "see what you've recommended before and correlate with outcomes "
            "from pull_history and query_icp_exposure_trend."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max briefs to return (default 20, max 50).",
                },
            },
        },
    },
    {
        "name": "query_irontomb_predictions",
        "description": (
            "Return Irontomb's post-hoc engagement predictions for the "
            "latest batch of posts Stelle generated. These were produced "
            "AFTER Stelle finished writing (no mid-loop bias). Compare "
            "predicted engagement against real engagement from "
            "pull_history to find where Irontomb systematically "
            "over- or under-predicts for this client. That delta is "
            "gradient signal: it tells you what conventional engagement "
            "wisdom gets wrong about this client's audience."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "note",
        "description": "Record an observation to your working memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "submit_brief",
        "description": (
            "Terminal tool. Submit the strategic brief. All fields required "
            "except where noted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "interview_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "DEPRECATED — leave empty or omit. Tribbie now generates "
                        "interview questions live from content_priorities and the "
                        "conversation itself. Pre-written questions run out mid-call "
                        "and can't adapt to what the client actually says."
                    ),
                },
                "asset_requests": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific things to request from the client: "
                        "screenshots, documents, metrics, case studies."
                    ),
                },
                "content_priorities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "What Stelle should emphasize in the next batch, "
                        "grounded in engagement evidence."
                    ),
                },
                "content_avoid": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Topics/angles Stelle should NOT do: saturated, "
                        "declining, or attracted the wrong audience."
                    ),
                },
                "abm_targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "company": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                    },
                    "description": (
                        "Named prospects or companies to weave into posts."
                    ),
                },
                "dm_targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "headline": {"type": "string"},
                            "company": {"type": "string"},
                            "icp_score": {"type": "number"},
                            "posts_engaged": {"type": "integer"},
                            "suggested_angle": {"type": "string"},
                        },
                    },
                    "description": (
                        "Warm prospects the client should DM on LinkedIn "
                        "right now. Include which posts they reacted to."
                    ),
                },
                "stelle_timing": {
                    "type": "string",
                    "description": "When to run Stelle next and why.",
                },
                "icp_exposure_assessment": {
                    "type": "string",
                    "description": (
                        "Free-text assessment of ICP exposure trajectory. "
                        "Is it improving? What's driving or stalling it?"
                    ),
                },
                "next_run_trigger": {
                    "type": "object",
                    "properties": {
                        "condition": {"type": "string"},
                        "or_after_days": {"type": "integer"},
                        "rationale": {"type": "string"},
                    },
                    "description": "When Cyrene should run again.",
                },
            },
            "required": [
                "content_priorities",
                "content_avoid",
                "dm_targets",
                "stelle_timing",
                "icp_exposure_assessment",
                "next_run_trigger",
            ],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a strategic growth agent for a LinkedIn content client. Study the
client's scored post history and produce a strategic brief. When done, call
submit_brief.
"""


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _query_irontomb_predictions(company: str, args: dict) -> str:
    """Return Irontomb's post-hoc predictions for the latest batch.

    These predictions were generated AFTER Stelle finished writing,
    so they reflect unbiased evaluation of the final drafts. Compare
    against real engagement (via pull_history) to see where
    Irontomb's predictions diverge from reality — that's gradient
    signal about what conventional engagement wisdom gets wrong for
    this client.
    """
    posthoc_path = P.memory_dir(company) / "irontomb_posthoc_latest.json"
    if not posthoc_path.exists():
        return json.dumps({
            "available": False,
            "note": "No post-hoc Irontomb predictions available yet.",
        })
    try:
        predictions = json.loads(posthoc_path.read_text(encoding="utf-8"))
        return json.dumps({
            "available": True,
            "n_posts": len(predictions),
            "predictions": predictions,
        }, default=str)
    except Exception as e:
        return json.dumps({"error": f"failed to read predictions: {str(e)[:200]}"})


_TOOL_DISPATCH: dict[str, Any] = {
    "pull_history": _query_observations,
    "pull_reactors": _query_top_engagers,
    "query_transcript_inventory": _query_transcript_inventory,
    "query_icp_exposure_trend": _query_icp_exposure_trend,
    "query_warm_prospects": _query_warm_prospects,
    "query_engagement_trajectories": _query_engagement_trajectories,
    "run_py": _execute_python,
    "scan_corpus": _search_linkedin_corpus,
    "net_query": _web_search,
    "pull_page": _fetch_url,
    "query_ordinal_posts": _query_ordinal_posts,
    "query_brief_history": _query_brief_history,
    "query_irontomb_predictions": _query_irontomb_predictions,
    "note": _note,
}


def _dispatch_tool(company: str, name: str, args: dict, notes: list[str]) -> str:
    """Route a tool call to its implementation."""
    if name == "note":
        text = (args.get("text") or "").strip()
        if text:
            notes.append(text)
        return json.dumps({"ok": True, "note_count": len(notes)})

    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown tool: {name}"})

    try:
        return handler(company, args)
    except Exception as e:
        logger.exception("[Cyrene] tool %s failed", name)
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:300]}"})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_strategic_review(company: str) -> dict[str, Any]:
    """Run a full Cyrene strategic review for one client.

    Turn-based agent loop. Returns the strategic brief dict on success,
    or a dict with `_error` on failure. Persists the brief to
    memory/{company}/cyrene_brief.json.
    """
    # CLI short-circuit: route the entire run through Claude CLI when the
    # feature flag is on. No API spend. Hard-fail on CLI error — no silent
    # fallback to API.
    from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli
    if _use_cli():
        logger.info("[Cyrene] CLI mode enabled — delegating to run_cyrene_cli()")
        from backend.src.mcp_bridge.claude_cli import run_cyrene_cli
        return run_cyrene_cli(company)

    # Load audience context from transcripts
    from backend.src.agents.irontomb import _load_icp_context
    client_context = _load_icp_context(company)

    # Count scored observations
    try:
        from backend.src.db.local import ruan_mei_load
        state = ruan_mei_load(company) or {}
        n_scored = sum(
            1 for o in state.get("observations", [])
            if o.get("status") in ("scored", "finalized")
        )
    except Exception:
        n_scored = 0

    # Load previous brief so Cyrene knows what it already recommended
    # and can build on it rather than repeat itself.
    previous_brief = "No previous brief exists. This is the first Cyrene run for this client."
    try:
        prev_path = P.memory_dir(company) / _BRIEF_FILENAME
        if prev_path.exists():
            prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
            # Include the full previous brief — more data to the model,
            # no lossy summary. The model decides what's relevant.
            previous_brief = json.dumps(prev_data, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

    system_prompt = _SYSTEM_PROMPT.format(
        client_context=client_context,
        n_scored=n_scored,
        company=company,
        previous_brief=previous_brief,
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Run a strategic review for {company}. Study the data "
                f"across your tools, form your strategy from evidence, "
                f"and produce a comprehensive brief via submit_brief."
            ),
        }
    ]

    client = anthropic.Anthropic()
    notes: list[str] = []
    total_cost = 0.0
    turns_used = 0
    brief: Optional[dict] = None

    for turn in range(1, _CYRENE_MAX_TURNS + 1):
        turns_used = turn
        try:
            resp = client.messages.create(
                model=_CYRENE_MODEL,
                max_tokens=_CYRENE_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=_TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as e:
            logger.warning("[Cyrene] API call failed turn=%d: %s", turn, e)
            return {
                "_error": f"API call failed on turn {turn}: {str(e)[:200]}",
                "_turns_used": turns_used,
            }

        # Cost tracking
        try:
            usage = resp.usage
            in_tok = getattr(usage, "input_tokens", 0) or 0
            out_tok = getattr(usage, "output_tokens", 0) or 0
            cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_w = getattr(usage, "cache_creation_input_tokens", 0) or 0
            total_cost += (
                (in_tok / 1e6) * _INPUT_COST_PER_MTOK
                + (out_tok / 1e6) * _OUTPUT_COST_PER_MTOK
                + (cache_r / 1e6) * _CACHE_READ_COST_PER_MTOK
                + (cache_w / 1e6) * _CACHE_WRITE_COST_PER_MTOK
            )
        except Exception:
            pass

        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            logger.warning("[Cyrene] %s: no tool call on turn %d", company, turn)
            break

        tool_results: list[dict] = []
        for tu in tool_uses:
            if tu.name == "submit_brief":
                if isinstance(tu.input, dict):
                    brief = dict(tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": "Brief submitted. Review complete.",
                })
            else:
                result = _dispatch_tool(company, tu.name, tu.input or {}, notes)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result[:12000],
                })

        messages.append({"role": "user", "content": tool_results})

        if brief is not None:
            break

        if resp.stop_reason == "end_turn":
            break

    if brief is None:
        return {
            "_error": f"no submit_brief within {_CYRENE_MAX_TURNS} turns",
            "_turns_used": turns_used,
            "_cost_usd": round(total_cost, 2),
            "_notes": notes,
        }

    # Stamp metadata
    brief["_company"] = company
    brief["_computed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    brief["_turns_used"] = turns_used
    brief["_cost_usd"] = round(total_cost, 2)
    brief["_notes"] = notes

    # Persist current brief + append to history
    try:
        mem_dir = P.memory_dir(company)
        mem_dir.mkdir(parents=True, exist_ok=True)

        # Write current brief (overwrite)
        brief_path = mem_dir / _BRIEF_FILENAME
        tmp = brief_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(brief, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.rename(brief_path)

        # Append to history (JSONL) — one line per brief, never overwritten.
        # This is the trajectory Cyrene reads via query_brief_history to
        # see how its recommendations have evolved and correlate with
        # real engagement outcomes across cycles.
        history_path = mem_dir / _BRIEF_HISTORY_FILENAME
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(brief, ensure_ascii=False, default=str) + "\n")

        logger.info(
            "[Cyrene] %s: brief saved (%d turns, $%.2f). "
            "interview_questions=%d, dm_targets=%d, content_priorities=%d",
            company, turns_used, total_cost,
            len(brief.get("interview_questions", [])),
            len(brief.get("dm_targets", [])),
            len(brief.get("content_priorities", [])),
        )
    except Exception as e:
        logger.warning("[Cyrene] failed to persist brief for %s: %s", company, e)

    return brief


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 cyrene.py <company>")
        sys.exit(1)

    company_arg = sys.argv[1]
    result = run_strategic_review(company_arg)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
