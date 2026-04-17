"""Client Progress Report API — generates presentation-ready reports for client video calls.

Covers:
  1. Account progress in the last N weeks (posts published, engagement metrics)
  2. Content strategy applied (topics, formats, performance intelligence, learned directives)
  3. Strategy shifts for upcoming weeks (analyst findings, data-driven recommendations)

Endpoints:
  GET /api/report/{company}          — JSON report data
  GET /api/report/{company}/html     — HTML as JSON: {"html": "..."}
  GET /api/report/{company}/view     — HTML directly for browser rendering
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from backend.src.core.events import done_event, error_event, status_event
from backend.src.services import job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/report", tags=["report"])

# ---------------------------------------------------------------------------
# Ordinal API helpers
# ---------------------------------------------------------------------------

_API_BASE = "https://app.tryordinal.com/api/v1"


def _load_ordinal_keys() -> dict[str, tuple[str, str]]:
    """Returns {slug: (company_id, api_key)} from ordinal_auth_rows.csv."""
    from backend.src.db import vortex
    csv_path = vortex.MEMORY_ROOT / "ordinal_auth_rows.csv"
    if not csv_path.exists():
        # Fallback: try openclaw workspace secrets
        csv_path = Path.home() / ".openclaw" / "workspace" / "secrets" / "ordinal-keys.csv"
    keys: dict[str, tuple[str, str]] = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                slug = row.get("provider_org_slug") or row.get("slug", "")
                keys[slug] = (row.get("company_id", ""), row.get("api_key", ""))
    return keys


def _clean_ordinal_markup(text: str) -> str:
    """Strip Ordinal's LinkedIn mention markup: @[Name](urn:li:...) → Name."""
    import re
    return re.sub(r"@\[([^\]]+)\]\([^)]+\)", r"\1", text)


def _reward_label(reward: float) -> str:
    """Translate z-scored reward into client-friendly performance label."""
    if reward > 1.0:
        return "well above average"
    elif reward > 0.3:
        return "above average"
    elif reward > -0.3:
        return "near average"
    elif reward > -1.0:
        return "below average"
    else:
        return "well below average"


def _ordinal_get(endpoint: str, api_key: str) -> dict | list | None:
    url = f"{_API_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read(), strict=False)
    except Exception as e:
        logger.warning("Ordinal API error for %s: %s", endpoint, e)
        return None


def _fetch_all_posts(api_key: str) -> list[dict]:
    posts: list[dict] = []
    cursor = None
    while True:
        ep = "/posts?limit=100"
        if cursor:
            ep += f"&cursor={cursor}"
        data = _ordinal_get(ep, api_key)
        if not data or isinstance(data, list):
            break
        batch = data.get("posts", [])
        posts.extend(batch)
        if data.get("hasMore") and data.get("nextCursor"):
            cursor = data["nextCursor"]
        else:
            break
    return posts


def _extract_profile_id(posts: list[dict]) -> str | None:
    for p in posts:
        profile = (p.get("linkedIn") or {}).get("profile") or {}
        pid = profile.get("id")
        if pid:
            return pid
    return None


def _extract_profile_name(posts: list[dict]) -> str:
    for p in posts:
        profile = (p.get("linkedIn") or {}).get("profile") or {}
        name = profile.get("name")
        if name:
            return name
    return "Unknown"


def _fetch_analytics(api_key: str, profile_id: str, start: str, end: str) -> list[dict]:
    data = _ordinal_get(
        f"/analytics/linkedin/{profile_id}/posts?startDate={start}&endDate={end}",
        api_key,
    )
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("posts", [])
    return []


def _fetch_followers(api_key: str, profile_id: str, start: str, end: str) -> list[dict]:
    """Daily follower snapshots from Ordinal.

    Each element: ``{"followerCount": int, "recordedAt": ISO8601 str}``.
    """
    data = _ordinal_get(
        f"/analytics/linkedin/{profile_id}/followers?startDate={start}&endDate={end}",
        api_key,
    )
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("followers") or data.get("data") or []
    return []


# ---------------------------------------------------------------------------
# Amphoreus data helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_observations(company: str) -> list[dict]:
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        state = None
    return (state or {}).get("observations", [])


def _parse_ts(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Report data assembly
# ---------------------------------------------------------------------------

def _build_engager_section(company: str) -> dict | None:
    """Load engager distribution and pre-render SVG pie.

    Tries the CSV-based history store first.  Falls back to the automated
    SQLite ``post_engagers`` table (populated by engager_fetcher + icp_scorer)
    when the CSV store is empty, converting continuous ICP scores into the
    four-bucket classification the pie chart expects.
    """
    try:
        dist = _engager_dist_from_sqlite(company)

        if dist.get("total", 0) == 0:
            return None

        # Inline SVG pie chart (replaced old engager_report dependency)
        return {"distribution": dist, "pie_svg": None}
    except Exception as e:
        logger.warning("Engager section failed for %s: %s", company, e)
        return None


def _engager_dist_from_sqlite(company: str) -> dict:
    """Build an engager_distribution-compatible dict from the SQLite post_engagers table.

    Buckets continuous ICP scores into the four categories:
      icp_match  — score >= 0.6
      orbit      — 0.3 <= score < 0.6  (adjacent / neutral)
      non_icp    — score < 0.3
      internal   — always 0 (automated pipeline can't detect team members)
    """
    try:
        from backend.src.db.local import initialize_db
        initialize_db()

        import sqlite3
        from backend.src.core.config import get_settings
        conn = sqlite3.connect(get_settings().sqlite_path)
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            "SELECT DISTINCT engager_urn, name, icp_score "
            "FROM post_engagers WHERE company = ? AND engager_urn != ''",
            (company,),
        ).fetchall()
        conn.close()

        if not rows:
            return {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0,
                    "total": 0, "segments": {}}

        seen: dict[str, float | None] = {}
        for r in rows:
            urn = r["engager_urn"]
            score = r["icp_score"]
            if urn not in seen or (score is not None and (seen[urn] is None or score > seen[urn])):
                seen[urn] = score

        counts = {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0}
        for score in seen.values():
            if score is None or score < 0.3:
                counts["non_icp"] += 1
            elif score < 0.6:
                counts["orbit"] += 1
            else:
                counts["icp_match"] += 1

        return {**counts, "total": sum(counts.values()), "segments": {}}

    except Exception as e:
        logger.debug("SQLite engager fallback failed for %s: %s", company, e)
        return {"icp_match": 0, "non_icp": 0, "internal": 0, "orbit": 0,
                "total": 0, "segments": {}}


def _build_icp_prospects(company: str, top_n: int = 20) -> dict | None:
    """Build a ranked list of ICP prospects with cold messaging guidance.

    Queries SQLite for top ICP-scored engagers, then makes a single Claude
    batch call to generate per-prospect reasoning and messaging approach.
    """
    try:
        from backend.src.db.local import get_top_icp_engagers
        from backend.src.db import vortex

        prospects = get_top_icp_engagers(company, limit=top_n)
        if not prospects:
            return None

        icp_path = vortex.icp_definition_path(company)
        icp_desc = ""
        if icp_path.exists():
            icp = json.loads(icp_path.read_text(encoding="utf-8"))
            icp_desc = icp.get("description", "")

        if not icp_desc:
            return {"prospects": prospects}

        profile_block = []
        for i, p in enumerate(prospects):
            parts = [f"Name: {p.get('name', 'Unknown')}"]
            if p.get("headline"):
                parts.append(f"Headline: {p['headline']}")
            if p.get("current_company"):
                parts.append(f"Company: {p['current_company']}")
            if p.get("title"):
                parts.append(f"Title: {p['title']}")
            if p.get("location"):
                parts.append(f"Location: {p['location']}")
            parts.append(f"ICP Score: {p.get('mean_icp_score', 0):.2f}")
            parts.append(f"Engaged with {p.get('engagement_count', 1)} posts")
            profile_block.append(f"{i + 1}. {' | '.join(parts)}")

        prompt = (
            "You are a B2B sales strategist. Below is our Ideal Customer Profile (ICP) "
            "and a ranked list of LinkedIn users who have engaged with our client's posts.\n\n"
            f"ICP DEFINITION:\n{icp_desc}\n\n"
            f"PROSPECTS (ranked by ICP fit and engagement frequency):\n"
            f"{chr(10).join(profile_block)}\n\n"
            "For each numbered prospect, output a JSON array where each element has:\n"
            '  "reason": a 1-sentence explanation of WHY they match the ICP '
            "(reference their specific role, company, or industry)\n"
            '  "approach": a 1-2 sentence cold messaging angle our client could use '
            "(be specific to the person's background, reference the content they engaged with)\n\n"
            "Output ONLY the JSON array, no markdown fences, no extra text.\n"
            "Example: [{\"reason\": \"VP of Ops at a mid-market SaaS — exact ICP seniority and vertical\", "
            "\"approach\": \"Reference the post they reacted to about scaling ops, ask about their "
            'team\'s biggest bottleneck this quarter"}]'
        )

        from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli, cli_single_shot as _cli_ss
        if _use_cli():
            raw_txt = _cli_ss(prompt, model="sonnet", max_tokens=2000, timeout=120)
            if raw_txt is None:
                raise RuntimeError("CLI single-shot returned None for ICP prospect annotations")
            raw = raw_txt.strip()
        else:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            annotations = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[report] Failed to parse ICP prospect annotations for %s", company)
            annotations = []

        for i, p in enumerate(prospects):
            if i < len(annotations):
                p["reason"] = annotations[i].get("reason", "")
                p["approach"] = annotations[i].get("approach", "")

        return {
            "prospects": prospects,
            "icp_definition_summary": icp_desc[:200],
        }

    except Exception as e:
        logger.warning("[report] ICP prospects generation failed for %s: %s", company, e)
        return None


def _build_content_directions(company: str, cutoff: datetime | None = None) -> dict | None:
    """Ranked top posts — best overall + best in the current period.

    Client-facing, so no underperformer list. The goal is to showcase what
    worked; the bottom of the distribution is already implicit in the
    reward z-scores and doesn't need to be named.
    """
    try:
        observations = _load_observations(company)
        scored = [o for o in observations if o.get("status") in ("scored", "finalized")]
        if len(scored) < 3:
            return None

        def _post_entry(o: dict) -> dict:
            body = o.get("posted_body") or o.get("post_body") or ""
            hook = body.split("\n")[0][:120].strip()
            raw = o.get("reward", {}).get("raw_metrics") or {}
            return {
                "hook": _clean_ordinal_markup(hook),
                "impressions": raw.get("impressions", 0),
                "reactions": raw.get("reactions", 0),
                "comments": raw.get("comments", 0),
                "performance": _reward_label(o.get("reward", {}).get("immediate", 0)),
                "posted_at": (o.get("posted_at") or "")[:10],
            }

        def _top_n_by(pool: list[dict], metric: str, n: int = 3) -> list[dict]:
            def _sort_key(o: dict) -> float:
                if metric == "reward":
                    return float(o.get("reward", {}).get("immediate", 0) or 0)
                raw = o.get("reward", {}).get("raw_metrics") or {}
                return float(raw.get(metric, 0) or 0)
            return [
                _post_entry(o)
                for o in sorted(pool, key=_sort_key, reverse=True)[:n]
            ]

        metrics = ["reward", "impressions", "reactions", "comments"]

        recent: list[dict] = []
        if cutoff is not None:
            recent = [
                o for o in scored
                if (_parse_ts(o.get("posted_at", "")) or datetime.min.replace(tzinfo=timezone.utc))
                >= cutoff
            ]

        return {
            "total_scored": len(scored),
            # Default view kept for backward-compat with older prompt versions.
            "top_posts": _top_n_by(scored, "reward"),
            "top_posts_current_period": _top_n_by(recent, "reward") if recent else [],
            # Multi-sort lists for the renderer to offer as views.
            "top_overall_by": {m: _top_n_by(scored, m) for m in metrics},
            "top_current_by": (
                {m: _top_n_by(recent, m) for m in metrics} if recent else {}
            ),
        }
    except Exception as e:
        logger.debug("Content directions failed for %s: %s", company, e)
        return None


def _build_report(company: str, weeks: int = 2) -> dict:
    """Assemble the full report data structure."""
    from backend.src.db import vortex

    keys = _load_ordinal_keys()
    api_key = keys.get(company, ("", ""))[1]

    today = date.today()
    cutoff = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(weeks=weeks)
    start_date = (today - timedelta(days=90)).isoformat()
    end_date = today.isoformat()

    # Ordinal data
    posts: list[dict] = []
    analytics: list[dict] = []
    follower_series: list[dict] = []
    profile_name = company
    if api_key:
        posts = _fetch_all_posts(api_key)
        # One Ordinal account can hold multiple LinkedIn profiles (e.g.
        # example-client and example-client share credentials). Filter by
        # the slug's linkedin_username.txt so we only report on the right
        # person. profile.detail is LinkedIn's vanity handle.
        username_path = vortex.linkedin_username_path(company)
        if username_path.exists():
            want = username_path.read_text().strip().lower()
            if want:
                posts = [
                    p for p in posts
                    if ((p.get("linkedIn") or {}).get("profile") or {}).get("detail", "").lower() == want
                ]
        profile_id = _extract_profile_id(posts)
        profile_name = _extract_profile_name(posts)
        if profile_id:
            analytics = _fetch_analytics(api_key, profile_id, start_date, end_date)
            follower_series = _fetch_followers(api_key, profile_id, start_date, end_date)

    # Analytics indexed by ordinal post id
    analytics_by_id: dict[str, dict] = {}
    for a in analytics:
        oid = (a.get("ordinalPost") or {}).get("id")
        if oid:
            analytics_by_id[oid] = a

    # Amphoreus data
    observations = _load_observations(company)
    cyrene_brief = _load_json(vortex.memory_dir(company) / "cyrene_brief.json")
    directives = _load_json(vortex.memory_dir(company) / "learned_directives.json")

    # --- Posts published in window ---
    published = []
    for p in posts:
        if p.get("status") != "Posted":
            continue
        pub = p.get("publishedAt") or p.get("publishAt") or ""
        ts = _parse_ts(pub)
        if ts and ts >= cutoff:
            pid = p.get("id", "")
            a = analytics_by_id.get(pid, {})
            # Skip posts without analytics (just published, data not in yet)
            if not a.get("impressionCount"):
                continue
            copy_text = (p.get("linkedIn") or {}).get("copy") or ""
            hook = copy_text.split("\n")[0][:120] if copy_text else ""
            eng_rate = a.get("engagement")
            published.append({
                "date": ts.isoformat(),
                "date_display": ts.strftime("%a %b %d"),
                "title": _clean_ordinal_markup(p.get("title") or "(untitled)"),
                "hook": _clean_ordinal_markup(hook),
                "impressions": a.get("impressionCount", 0),
                "likes": a.get("likeCount", 0),
                "comments": a.get("commentCount", 0),
                "reposts": a.get("shareCount", 0),
                "engagement_rate_pct": f"{eng_rate * 100:.1f}%" if eng_rate else "—",
            })
    published.sort(key=lambda x: x["date"])

    total_imp = sum(p["impressions"] for p in published)
    total_likes = sum(p["likes"] for p in published)
    total_comments = sum(p["comments"] for p in published)
    n_published = len(published)

    # --- Period-over-period comparison ---
    prior_cutoff = cutoff - timedelta(weeks=weeks)
    prior_published = []
    for p in posts:
        if p.get("status") != "Posted":
            continue
        pub = p.get("publishedAt") or p.get("publishAt") or ""
        ts = _parse_ts(pub)
        if ts and prior_cutoff <= ts < cutoff:
            a = analytics_by_id.get(p.get("id", ""), {})
            if a.get("impressionCount"):
                prior_published.append({
                    "impressions": a.get("impressionCount", 0),
                    "likes": a.get("likeCount", 0),
                    "comments": a.get("commentCount", 0),
                })
    prior_imp = sum(p["impressions"] for p in prior_published)
    prior_likes = sum(p["likes"] for p in prior_published)
    prior_comments = sum(p["comments"] for p in prior_published)
    prior_count = len(prior_published)

    # --- Follower growth ---
    # Ordinal returns one snapshot per day (midnight UTC-ish). We pick the
    # closest snapshot to each cutoff and compute net + % change across
    # current and prior periods.
    followers_section: dict | None = None
    if follower_series:
        parsed: list[tuple[datetime, int]] = []
        for rec in follower_series:
            ts = _parse_ts(rec.get("recordedAt") or "")
            fc = rec.get("followerCount")
            if ts and isinstance(fc, (int, float)):
                parsed.append((ts, int(fc)))
        parsed.sort(key=lambda x: x[0])

        def _closest(target_dt: datetime) -> tuple[datetime, int] | None:
            if not parsed:
                return None
            return min(parsed, key=lambda x: abs((x[0] - target_dt).total_seconds()))

        now_dt = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
        latest = parsed[-1] if parsed else None
        at_cutoff = _closest(cutoff)
        at_prior_cutoff = _closest(cutoff - timedelta(weeks=weeks))

        if latest and at_cutoff:
            current_count = latest[1]
            start_count = at_cutoff[1]
            delta = current_count - start_count
            pct = (delta / start_count * 100) if start_count > 0 else 0.0

            prior_delta = prior_pct = None
            if at_prior_cutoff and at_prior_cutoff[0] < at_cutoff[0]:
                prior_delta = at_cutoff[1] - at_prior_cutoff[1]
                prior_pct = (
                    (prior_delta / at_prior_cutoff[1] * 100)
                    if at_prior_cutoff[1] > 0 else 0.0
                )

            # Compact daily series within the current window for sparkline/line chart
            window_series = [
                {"date": ts.strftime("%Y-%m-%d"), "count": count}
                for ts, count in parsed
                if ts >= cutoff - timedelta(days=1)
            ]

            followers_section = {
                "current": current_count,
                "at_period_start": start_count,
                "net_change": delta,
                "pct_change": round(pct, 2),
                "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
                "prior_period_net_change": prior_delta,
                "prior_period_pct_change": round(prior_pct, 2) if prior_pct is not None else None,
                "window_series": window_series,
                "period_start_date": at_cutoff[0].strftime("%Y-%m-%d"),
                "latest_recorded_at": latest[0].strftime("%Y-%m-%d"),
            }

    period_comparison = None
    if prior_count > 0 and n_published > 0:
        prior_avg_imp = prior_imp // max(prior_count, 1)
        cur_avg_imp = total_imp // max(n_published, 1)
        avg_imp_change = (
            (cur_avg_imp - prior_avg_imp) / prior_avg_imp * 100
            if prior_avg_imp > 0 else 0
        )
        period_comparison = {
            "prior_period": f"{prior_cutoff.strftime('%b %d')} – {cutoff.strftime('%b %d')}",
            "prior_posts": prior_count,
            "prior_avg_impressions": prior_avg_imp,
            "prior_avg_reactions": round(prior_likes / max(prior_count, 1), 1),
            "prior_avg_comments": round(prior_comments / max(prior_count, 1), 1),
            "avg_impressions_change_pct": f"{avg_imp_change:+.0f}%",
            "posts_change": n_published - prior_count,
        }

    # --- Upcoming posts ---
    future_cutoff = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    upcoming = []
    for p in posts:
        if p.get("status") == "Posted":
            continue
        pub = p.get("publishAt") or ""
        ts = _parse_ts(pub)
        if ts and ts >= future_cutoff:
            upcoming.append({
                "date": ts.isoformat(),
                "date_display": ts.strftime("%a %b %d"),
                "title": _clean_ordinal_markup(p.get("title") or "(untitled)"),
                "status": p.get("status", "?"),
            })
    upcoming.sort(key=lambda x: x["date"])

    # --- Recent observations (window) ---
    recent_obs = [o for o in observations if (_parse_ts(o.get("posted_at", "")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]
    scored_recent = [o for o in recent_obs if o.get("status") in ("scored", "finalized")]

    # Window performance — translated to client-friendly language
    window_performance = None
    if scored_recent:
        rewards = [o.get("reward", {}).get("immediate", 0) for o in scored_recent]
        avg_r = sum(rewards) / len(rewards)
        best = max(scored_recent, key=lambda o: o.get("reward", {}).get("immediate", 0))
        worst = min(scored_recent, key=lambda o: o.get("reward", {}).get("immediate", 0))
        window_performance = {
            "summary": (
                f"This period's posts performed {'above' if avg_r > 0.1 else 'near' if avg_r > -0.1 else 'below'} "
                f"your historical average."
            ),
            "best": {
                "performance": _reward_label(best.get("reward", {}).get("immediate", 0)),
                "impressions": (best.get("reward", {}).get("raw_metrics") or {}).get("impressions", 0),
                "hook": (best.get("post_body") or best.get("posted_body") or "")[:120].split("\n")[0],
            },
            "worst": {
                "performance": _reward_label(worst.get("reward", {}).get("immediate", 0)),
                "impressions": (worst.get("reward", {}).get("raw_metrics") or {}).get("impressions", 0),
                "hook": (worst.get("post_body") or worst.get("posted_body") or "")[:120].split("\n")[0],
            },
        }

    # --- All scored (needed by multiple sections below) ---
    all_scored = [o for o in observations if o.get("status") in ("scored", "finalized")]
    all_scored.sort(key=lambda o: o.get("posted_at", ""))

    # --- Content directions from continuous field ---
    content_directions = _build_content_directions(company, cutoff=cutoff)

    # --- Learned writing rules (translated for client) ---
    # Show what the system has learned about this client's voice — no source attribution.
    # The client doesn't need to know whether a rule came from editorial notes,
    # accepted posts, or engagement signals; just show the rule itself.
    writing_rules = []
    if directives and directives.get("directives"):
        for d in directives["directives"]:
            if d.get("priority") in ("high", "medium"):
                rule = d.get("directive", "")
                # Strip internal prefixes
                for prefix in ("DO ", "DON'T ", "DONT ", "Always ", "Never "):
                    if rule.startswith(prefix):
                        rule = rule[len(prefix):]
                        break
                writing_rules.append(
                    rule[0].upper() + rule[1:] if rule else ""
                )

    # --- Cyrene strategic brief highlights (replaces analyst findings) ---
    content_priorities = []
    content_avoid = []
    icp_exposure_assessment = ""
    stelle_timing = ""
    dm_targets_for_report: list[dict] = []
    if cyrene_brief:
        content_priorities = cyrene_brief.get("content_priorities") or []
        content_avoid = cyrene_brief.get("content_avoid") or []
        icp_exposure_assessment = cyrene_brief.get("icp_exposure_assessment") or ""
        stelle_timing = cyrene_brief.get("stelle_timing") or ""
        dm_targets_for_report = (cyrene_brief.get("dm_targets") or [])[:5]

    # --- All-time context ---
    all_time = None
    if all_scored:
        n = len(all_scored)
        rewards = [o.get("reward", {}).get("immediate", 0) for o in all_scored]
        avg_imp = sum((o.get("reward", {}).get("raw_metrics") or {}).get("impressions", 0) for o in all_scored) / n
        avg_react = sum((o.get("reward", {}).get("raw_metrics") or {}).get("reactions", 0) for o in all_scored) / n
        all_time = {
            "total_tracked": len(observations),
            "total_scored": n,
            "avg_impressions": round(avg_imp),
            "avg_reactions": round(avg_react),
            "reward_min": round(min(rewards), 2),
            "reward_max": round(max(rewards), 2),
            "reward_mean": round(sum(rewards) / n, 2),
        }

    # --- System learning status ---
    # Proves the system gets smarter over time. This is the core thesis:
    # "Virio ships posts. We ship a system that gets smarter with every post."
    learning_status = {}

    # Prediction accuracy — raw numbers only. The dashboard renders its own
    # interpretation from the numbers; there are no canned label strings
    # stored alongside the data.
    pred_acc = _load_json(vortex.memory_dir(company) / "prediction_accuracy.json")
    if pred_acc and pred_acc.get("n_predictions", 0) > 0:
        learning_status["prediction_accuracy"] = {
            "n_predictions": pred_acc.get("n_predictions", 0),
            "spearman": pred_acc.get("spearman", 0),
            "mean_abs_error": pred_acc.get("mean_abs_error", 0),
            "early_mae": pred_acc.get("early_mae"),
            "late_mae": pred_acc.get("late_mae"),
        }

    # Irontomb calibration (Phase 3) — simulator prediction accuracy
    irontomb_cal = _load_json(vortex.memory_dir(company) / "calibration_report.json")
    if irontomb_cal and irontomb_cal.get("n_pairs_joined", 0) > 0:
        _metrics = irontomb_cal.get("metrics") or {}
        learning_status["irontomb_calibration"] = {
            "n_pairs": irontomb_cal.get("n_pairs_joined", 0),
            "spearman": _metrics.get("spearman_engagement_prediction"),
            "mean_abs_error_per_1k": _metrics.get("mean_abs_error_per_1k"),
            "binary_accuracy": _metrics.get("binary_accuracy_would_react"),
        }

    # Cyrene brief metadata (if a strategic review has been run)
    if cyrene_brief:
        learning_status["cyrene_brief"] = {
            "computed_at": cyrene_brief.get("_computed_at"),
            "turns_used": cyrene_brief.get("_turns_used"),
            "cost_usd": cyrene_brief.get("_cost_usd"),
            "n_content_priorities": len(cyrene_brief.get("content_priorities") or []),
            "n_dm_targets": len(cyrene_brief.get("dm_targets") or []),
        }

    # Observation coverage + embedding model status
    if all_scored:
        posts_with_predictions = sum(1 for o in all_scored if o.get("predicted_engagement") is not None)
        learning_status["coverage"] = {
            "total_scored": len(all_scored),
            "with_predictions": posts_with_predictions,
        }

    # Embedding model status
    if all_scored:
        from backend.src.utils.post_embeddings import get_post_embeddings
        try:
            embs = get_post_embeddings(company)
            embedded_count = sum(1 for o in all_scored if o.get("post_hash") in embs)
            if embedded_count:
                learning_status["embedding_model"] = {
                    "posts_embedded": embedded_count,
                }
        except Exception:
            pass

    # --- Engager ICP distribution ---
    engager_section = _build_engager_section(company)

    # --- ICP prospects for outreach ---
    icp_prospects = _build_icp_prospects(company, top_n=5)

    return {
        "company": company,
        "profile_name": profile_name,
        "weeks": weeks,
        "period_start": (today - timedelta(weeks=weeks)).isoformat(),
        "period_end": today.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "posts_published": published,
        "period_totals": {
            "count": n_published,
            "avg_impressions": total_imp // max(n_published, 1),
            "avg_reactions": round(total_likes / max(n_published, 1), 1),
            "avg_comments": round(total_comments / max(n_published, 1), 1),
        },
        "upcoming": upcoming,
        "period_comparison": period_comparison,
        "followers": followers_section,
        "strategy_applied": {
            "window_performance": window_performance,
            "content_directions": content_directions,
            "writing_rules": writing_rules[:5],
        },
        "strategy_shifts": {
            "content_priorities": content_priorities[:5],
            "content_avoid": content_avoid[:3],
            "icp_exposure_assessment": icp_exposure_assessment,
            "stelle_timing": stelle_timing,
            "dm_targets": dm_targets_for_report,
        },
        "content_strategy": _build_strategy_report_section(company),
        "learning_status": learning_status,
        "all_time": all_time,
        "engager_icp": engager_section,
        "icp_prospects": icp_prospects,
    }


def _build_strategy_report_section(company: str) -> dict | None:
    """Build a content strategy summary for the progress report.

    Shows: strategic context, upcoming plan (next 2 weeks), content gaps,
    learning agenda, and funnel mix.
    """
    from backend.src.db import vortex as P

    plan_path = P.memory_dir(company) / "content_landscape.json"
    if not plan_path.exists():
        plan_path = P.memory_dir(company) / "content_strategy_plan.json"
    if not plan_path.exists():
        return None

    try:
        strategy = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    ctx = strategy.get("strategic_context", {})
    weekly_plan = strategy.get("weekly_plan", [])
    gaps = strategy.get("content_gaps", [])
    agenda = strategy.get("learning_agenda", {})
    funnel = strategy.get("funnel_mix", {})

    upcoming_weeks = []
    for week in weekly_plan[:2]:
        slots_summary = []
        for slot in week.get("slots", []):
            slots_summary.append({
                "slot": slot.get("slot"),
                "objective": slot.get("objective"),
                "topic": slot.get("topic"),
                "format": slot.get("format"),
                "funnel_stage": slot.get("funnel_stage"),
                "source_material_status": slot.get("source_material_status"),
            })
        upcoming_weeks.append({
            "week": week.get("week"),
            "week_of": week.get("week_of"),
            "theme": week.get("theme"),
            "learning_objective": week.get("week_learning_objective"),
            "slots": slots_summary,
        })

    return {
        "generated_at": strategy.get("generated_at"),
        "n_scored": strategy.get("n_scored"),
        "strategic_context": ctx,
        "upcoming_weeks": upcoming_weeks,
        "content_gaps": [
            {"gap": g.get("gap"), "severity": g.get("severity")}
            for g in gaps[:5]
        ],
        "learning_agenda": agenda,
        "funnel_mix": funnel,
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_REPORT_HTML_SYSTEM = """\
You write an HTML progress report for a client's LinkedIn content
performance. Use the scored post data provided to produce a report with
clear sections and visualizations.
"""

_REPORT_HTML_USER = """\
Convert the following client progress report data into the presentation HTML described in your instructions.

Client name: {profile_name}
Period: {period_start} to {period_end}
Generated: {generated_at}

---

{report_json}
"""


def _generate_report_html(report_data: dict) -> str | None:
    """Use Claude to convert the report data into presentation HTML."""
    user_msg = _REPORT_HTML_USER.format(
        profile_name=report_data["profile_name"],
        period_start=report_data["period_start"],
        period_end=report_data["period_end"],
        generated_at=report_data["generated_at"],
        report_json=json.dumps(report_data, indent=2, default=str),
    )

    from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli, cli_single_shot as _cli_ss
    if _use_cli():
        # 32k HTML output can take several minutes. Generous timeout.
        html_txt = _cli_ss(
            user_msg,
            system_prompt=_REPORT_HTML_SYSTEM,
            model="sonnet",
            max_tokens=32768,
            timeout=900,
        )
        if html_txt is None:
            raise RuntimeError("CLI single-shot returned None for report HTML generation")
        html = html_txt.strip()
        if not html:
            raise RuntimeError("CLI returned empty text for report HTML")
    else:
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic not installed, cannot generate HTML report")
            return None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=32768,
            system=_REPORT_HTML_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            response = stream.get_final_message()
        if not response.content:
            raise RuntimeError(
                f"Anthropic returned no content blocks (stop_reason={response.stop_reason})"
            )
        first = response.content[0]
        text = getattr(first, "text", None)
        if not text:
            raise RuntimeError(
                f"Anthropic returned a non-text block {type(first).__name__} "
                f"(stop_reason={response.stop_reason})"
            )
        html = text.strip()
        if not html:
            raise RuntimeError(
                f"Anthropic returned empty text (stop_reason={response.stop_reason})"
            )
    # Em dashes read as AI-generated; replace with commas unconditionally.
    html = html.replace("\u2014", ",")
    if "<!DOCTYPE" not in html[:200] and "<html" not in html[:200]:
        logger.warning("Report HTML looks malformed (first 120 chars: %r)", html[:120])
    return html


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{company}")
async def get_report(company: str, weeks: int = Query(default=2, ge=1, le=12)):
    """Get the full progress report as JSON."""
    import asyncio
    try:
        return await asyncio.to_thread(_build_report, company, weeks)
    except Exception as e:
        logger.exception("Report generation failed for %s", company)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{company}/html")
async def get_report_html(company: str, weeks: int = Query(default=2, ge=1, le=12)):
    """Get the progress report as HTML (JSON-wrapped)."""
    import asyncio

    def _run():
        report = _build_report(company, weeks)
        return _generate_report_html(report)

    html = await asyncio.to_thread(_run)
    if not html:
        raise HTTPException(status_code=500, detail="HTML generation failed")
    return {"html": html}


@router.get("/{company}/view", response_class=HTMLResponse)
async def view_report_html(company: str, weeks: int = Query(default=2, ge=1, le=12)):
    """Serve the progress report HTML directly for browser rendering.

    Synchronous — blocks up to ~4 min on one Opus call. Prefer the
    job-based flow (`POST /{company}/generate` + SSE stream) which
    streams progress and writes the HTML to a cached file.
    """
    import asyncio

    def _run():
        report = _build_report(company, weeks)
        return _generate_report_html(report)

    html = await asyncio.to_thread(_run)
    if not html:
        raise HTTPException(status_code=500, detail="HTML generation failed")
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Job-based report generation (streams progress, caches HTML to file)
# ---------------------------------------------------------------------------


def _rendered_report_path(company: str):
    from backend.src.db import vortex
    return vortex.memory_dir(company) / "progress_report.html"


def _run_report_job(job_id: str, company: str, weeks: int) -> None:
    """Background worker: builds report, renders HTML, writes to disk.

    Progress is emitted via ``job_manager.emit_event``; the final
    ``done`` event carries the URL path the frontend should load. The
    heavy HTML is NOT embedded in the event payload — 40-50 KB of
    markup per row would bloat run_events unnecessarily.
    """
    try:
        job_manager.emit_event(
            job_id, status_event(f"Aggregating {weeks} weeks of data for {company}…")
        )
        report = _build_report(company, weeks)

        job_manager.emit_event(
            job_id, status_event("Rendering presentation HTML (Opus, ~3-4 min)…")
        )
        html = _generate_report_html(report)
        if not html:
            raise RuntimeError("HTML generation returned empty")

        out_path = _rendered_report_path(company)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

        rendered_url = f"/api/report/{company}/rendered"
        job_manager.set_status(job_id, "completed", output=rendered_url)
        job_manager.emit_event(job_id, done_event(rendered_url))
    except Exception as e:
        logger.exception("[report] job %s failed for %s", job_id, company)
        job_manager.set_status(job_id, "failed", error=str(e))
        job_manager.emit_event(job_id, error_event(str(e)[:500]))


@router.post("/{company}/generate")
async def generate_report_job(company: str, weeks: int = Query(default=2, ge=1, le=12)):
    """Kick off a progress-report job. Returns ``{job_id}``; stream
    progress via ``GET /api/report/stream/{job_id}`` and fetch the
    rendered HTML via the URL in the done event."""
    job_id = job_manager.create_job(
        client_slug=company,
        agent="progress-report",
        prompt=f"weeks={weeks}",
        creator_id=None,
    )
    job_manager.run_in_background(
        job_id,
        target=_run_report_job,
        args=(job_id, company, weeks),
    )
    return {"job_id": job_id, "status": "pending"}


@router.get("/stream/{job_id}")
async def stream_report_job(job_id: str, after_id: int = 0):
    from backend.src.db.local import get_run
    if not job_manager.get_job(job_id) and not get_run(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        job_manager.sse_stream(job_id, timeout=3600, heartbeat_interval=15, after_id=after_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{company}/rendered", response_class=HTMLResponse)
async def get_rendered_report(company: str):
    """Serve the most recently generated progress-report HTML file."""
    path = _rendered_report_path(company)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No rendered report yet — POST /{company}/generate first",
        )
    return HTMLResponse(content=path.read_text(encoding="utf-8"))
