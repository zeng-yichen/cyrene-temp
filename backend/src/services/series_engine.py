"""Content Series Engine — multi-post narrative arc planning and tracking.

Moves from isolated post generation to intentional series: 4-6 posts around
a single theme, each building on the previous one, creating compound
engagement effects. LinkedIn's 360Brew rewards topic consistency over 90 days
with exponential authority gains.

Three components:
1. **Series planner** — generates a post arc from a topic + transcripts.
   Called by Herta, the API, or an operator. Not a separate agent.
2. **Series-aware Stelle context** — checks active_series.json and injects
   series context into Stelle's prompt when a series post is due.
3. **Series performance tracker** — monitors engagement within a series
   and recommends wrapping/extending. Wired into ordinal_sync.

Storage: memory/{company}/active_series.json

Usage:
    from backend.src.services.series_engine import (
        plan_series,
        get_stelle_series_context,
        check_series_health,
        mark_post_written,
        mark_post_published,
    )

    # Plan a new series:
    series = plan_series("example-client", "sample size estimation", num_posts=5)

    # Before Stelle generates:
    context = get_stelle_series_context("example-client")
    # → "You are writing post 3 of 5 in the 'sample size estimation' series..."

    # After generation:
    mark_post_written(series_id, position=3, local_post_id="uuid-...")

    # In ordinal_sync after scoring:
    check_series_health("example-client")
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic()


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass
class SeriesPost:
    """A single slot in a series."""
    position: int               # 1-indexed
    arc_role: str               # "setup" | "tension" | "insight" | "application" | "synthesis"
    outline: str                # 1-2 sentence description of what this post covers
    transcript_source: str      # file + timestamp or "from previous posts" for synthesis
    status: str = "unwritten"   # "unwritten" | "written" | "published" | "scored"
    local_post_id: str = ""     # links to SQLite local_posts.id after generation
    ordinal_post_id: str = ""   # links to Ordinal after push
    scheduled_at: str = ""      # ISO timestamp from temporal orchestrator
    engagement_score: float = 0.0  # from RuanMei after scoring
    scored_at: str = ""


@dataclass
class Series:
    """A multi-post narrative arc."""
    series_id: str
    company: str
    theme: str                  # e.g., "sample size estimation"
    topic_arm: str = ""         # content direction label, if applicable
    num_posts: int = 5
    posts: list[dict] = field(default_factory=list)  # serialized SeriesPost dicts
    status: str = "active"      # "active" | "wrapping" | "complete" | "extending"
    created_at: str = ""
    updated_at: str = ""
    engagement_trend: str = ""  # "accelerating" | "stable" | "declining" | ""
    recommendation: str = ""    # human-readable recommendation


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def _series_path(company: str) -> Path:
    return P.memory_dir(company) / "active_series.json"


def _load_all_series(company: str) -> list[dict]:
    path = _series_path(company)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_all_series(company: str, series_list: list[dict]) -> None:
    path = _series_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(series_list, indent=2, ensure_ascii=False), encoding="utf-8")


def get_active_series(company: str) -> list[dict]:
    """Return all series with status 'active' or 'extending'."""
    return [
        s for s in _load_all_series(company)
        if s.get("status") in ("active", "extending")
    ]


def get_series_by_id(company: str, series_id: str) -> dict | None:
    for s in _load_all_series(company):
        if s.get("series_id") == series_id:
            return s
    return None


# ------------------------------------------------------------------
# Series planner
# ------------------------------------------------------------------

_DEFAULT_ARC_ROLES = ["setup", "tension", "insight", "application", "synthesis"]

_PLANNER_SYSTEM_TEMPLATE = """\
You plan a series of LinkedIn posts around a theme. Given a topic and
the client's transcripts, output a structured series plan.

{arc_guidance}
"""

_DEFAULT_ARC_GUIDANCE = """\
Arc structure:
1. **Setup** — establish the problem/context. Broad TOFU. Hook the audience into the series.
2. **Tension** — deepen the problem. Show why it's harder than people think. Specific examples.
3. **Insight** — the core revelation. The thing the author knows that others don't. MOFU with ICP callout.
4. **Application** — how to act on the insight. Tactical, implementable. MOFU.
5. **Synthesis** — connect everything. Can reference specific companies (BOFU/ABM if applicable). \
Leave a thread for future series.

For a 4-post series, merge tension+insight or application+synthesis.
For a 6-post series, split the insight or application into two posts."""


def _derive_arc_from_history(company: str, num_posts: int) -> str | None:
    """Analyze the client's best-performing post sequences to derive an arc pattern.

    Finds ALL above-median consecutive windows of length 3-6, sends multiple
    successful sequences to LLM for pattern extraction. Falls back to
    cross-client arc if insufficient client data.
    """
    arc = _derive_arc_from_client(company, num_posts)
    if arc:
        return arc
    return _derive_arc_from_cross_client(num_posts)


def _derive_arc_from_client(company: str, num_posts: int) -> str | None:
    """Client-specific arc derivation from their own post sequences."""
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
    except Exception:
        return None

    scored = [
        o for o in rm._state.get("observations", [])
        if o.get("status") in ("scored", "finalized") and o.get("posted_at") and o.get("descriptor", {}).get("analysis")
    ]

    if len(scored) < 8:
        return None

    scored.sort(key=lambda o: o.get("posted_at", ""))
    rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
    median_reward = sorted(rewards)[len(rewards) // 2]

    # Find ALL consecutive windows of 3-6 posts that scored above median
    good_windows: list[list[dict]] = []
    for window_size in (3, 4, 5, 6):
        for i in range(len(scored) - window_size + 1):
            window = scored[i:i + window_size]
            window_rewards = [o.get("reward", {}).get("immediate", 0) for o in window]
            if sum(r > median_reward for r in window_rewards) >= len(window_rewards) * 0.6:
                good_windows.append(window)

    if len(good_windows) < 1:
        return None

    return _extract_arc_pattern(good_windows[:5], num_posts, "this client's")


def _derive_arc_from_cross_client(num_posts: int) -> str | None:
    """Cross-client arc derivation — middle tier of the cascade."""
    from backend.src.db import vortex as P

    all_windows: list[list[dict]] = []
    if not P.MEMORY_ROOT.exists():
        return None

    for d in P.MEMORY_ROOT.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name == "our_memory":
            continue
        try:
            from backend.src.agents.ruan_mei import RuanMei
            rm = RuanMei(d.name)
            scored = [
                o for o in rm._state.get("observations", [])
                if o.get("status") in ("scored", "finalized") and o.get("posted_at")
                and o.get("descriptor", {}).get("analysis")
            ]
            if len(scored) < 6:
                continue
            scored.sort(key=lambda o: o.get("posted_at", ""))
            rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
            median_r = sorted(rewards)[len(rewards) // 2]
            for i in range(len(scored) - 2):
                window = scored[i:i + 3]
                if all(o.get("reward", {}).get("immediate", 0) > median_r for o in window):
                    all_windows.append(window)
        except Exception:
            continue

    if len(all_windows) < 3:
        return None

    # Sort by total reward, take top 5
    all_windows.sort(key=lambda w: sum(o.get("reward", {}).get("immediate", 0) for o in w), reverse=True)
    return _extract_arc_pattern(all_windows[:5], num_posts, "top-performing cross-client")


def _extract_arc_pattern(windows: list[list[dict]], num_posts: int, source_label: str) -> str | None:
    """Send multiple successful sequences to LLM for arc pattern extraction."""
    sequences_text = ""
    for seq_idx, window in enumerate(windows, 1):
        sequences_text += f"\nSequence {seq_idx}:\n"
        for j, o in enumerate(window, 1):
            analysis = o.get("descriptor", {}).get("analysis", "")[:200]
            reward = o.get("reward", {}).get("immediate", 0)
            sequences_text += f"  Post {j} (score {reward:.2f}): {analysis}\n"

    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": (
                f"These are {source_label} top-performing consecutive post sequences:\n"
                f"{sequences_text}\n"
                f"What STRUCTURAL PATTERNS appear across multiple successful sequences? "
                f"Don't describe individual topics — describe the arc: what role does each "
                f"position play? (e.g., 'Position 1 tends to be X, Position 2 escalates by Y')\n\n"
                f"Then generalize into a {num_posts}-post arc template.\n\n"
                f"Format as a numbered list of arc roles with 1-sentence descriptions.\n"
                f"Output ONLY the numbered arc template."
            )}],
        )
        arc_text = resp.content[0].text.strip()
        if arc_text and len(arc_text) > 50:
            return (
                f"Arc structure (learned from {source_label} sequences):\n"
                f"{arc_text}\n\n"
                f"Adapt this {num_posts}-post arc to the series theme."
            )
    except Exception as e:
        logger.debug("[series_engine] Arc pattern extraction failed: %s", e)

    return None


def plan_series(
    company: str,
    theme: str,
    num_posts: int = 5,
    topic_arm: str = "",
    transcript_context: str = "",
    strategy_context: str = "",
    icp_context: str = "",
) -> dict:
    """Plan a new content series and persist it.

    Args:
        company: Client company keyword.
        theme: The series theme (e.g., "sample size estimation").
        num_posts: Number of posts in the series (4-6).
        topic_arm: Content direction label if this series comes from a learning recommendation.
        transcript_context: Relevant transcript excerpts for sourcing.
        strategy_context: Content strategy snippet for tone/audience alignment.
        icp_context: ICP definition for MOFU/BOFU targeting.

    Returns:
        The created Series as a dict.
    """
    num_posts = max(4, min(6, num_posts))

    # Gather context if not provided
    if not transcript_context:
        transcript_context = _gather_transcript_context(company)
    if not strategy_context:
        strategy_context = _gather_strategy_context(company)
    if not icp_context:
        icp_context = _gather_icp_context(company)

    # Check for existing series on same theme
    existing = get_active_series(company)
    for s in existing:
        if s.get("theme", "").lower() == theme.lower():
            logger.info("[series_engine] Series on '%s' already active for %s", theme, company)
            return s

    # Derive arc from client's historical best-performing sequences
    learned_arc = _derive_arc_from_history(company, num_posts)
    if learned_arc:
        arc_guidance = learned_arc
        logger.info("[series_engine] Using data-derived arc for %s", company)
    else:
        arc_guidance = _DEFAULT_ARC_GUIDANCE
        logger.info("[series_engine] Using default arc template for %s (insufficient history)", company)

    planner_system = _PLANNER_SYSTEM_TEMPLATE.format(arc_guidance=arc_guidance)

    user_msg = (
        f"Plan a {num_posts}-post LinkedIn series on the theme: \"{theme}\"\n\n"
        f"Client: {company}\n"
    )
    if icp_context:
        user_msg += f"\nICP:\n{icp_context}\n"
    if strategy_context:
        user_msg += f"\nContent strategy:\n{strategy_context[:2000]}\n"
    if transcript_context:
        user_msg += f"\nTranscript material:\n{transcript_context[:4000]}\n"
    user_msg += (
        f"\nPlan exactly {num_posts} posts following the arc structure. "
        "Every post must trace to a specific transcript source."
    )

    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=planner_system,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        posts_data = json.loads(raw)
        if not isinstance(posts_data, list):
            raise ValueError("Expected JSON array")

    except Exception as e:
        logger.warning("[series_engine] Planning failed for %s: %s", company, e)
        # Fallback: generate skeleton
        posts_data = _fallback_skeleton(theme, num_posts)

    # Schedule posts using temporal orchestrator
    scheduled_dates = _schedule_series(company, num_posts)

    # Build series posts
    series_posts = []
    for i, pd in enumerate(posts_data[:num_posts]):
        sched = scheduled_dates[i].isoformat() if i < len(scheduled_dates) else ""
        series_posts.append(asdict(SeriesPost(
            position=i + 1,
            arc_role=pd.get("arc_role", _ARC_ROLES[min(i, len(_ARC_ROLES) - 1)]),
            outline=pd.get("outline", f"Post {i+1} of {theme}"),
            transcript_source=pd.get("transcript_source", ""),
            scheduled_at=sched,
        )))

    series = Series(
        series_id=str(uuid.uuid4())[:8],
        company=company,
        theme=theme,
        topic_arm=topic_arm,
        num_posts=num_posts,
        posts=series_posts,
        status="active",
        created_at=_now(),
        updated_at=_now(),
    )

    # Persist
    all_series = _load_all_series(company)
    all_series.append(asdict(series))
    _save_all_series(company, all_series)

    logger.info(
        "[series_engine] Planned %d-post series '%s' for %s (id=%s)",
        num_posts, theme, company, series.series_id,
    )

    return asdict(series)


def _fallback_skeleton(theme: str, num_posts: int, arc_roles: list[str] | None = None) -> list[dict]:
    """Generate a minimal series skeleton without LLM.

    Uses provided arc_roles if available (from learned arc), otherwise defaults.
    """
    if arc_roles and len(arc_roles) >= num_posts:
        roles = arc_roles[:num_posts]
    else:
        roles = _DEFAULT_ARC_ROLES[:num_posts] if num_posts <= 5 else _DEFAULT_ARC_ROLES + ["application_2"]
    return [
        {
            "position": i + 1,
            "arc_role": roles[i] if i < len(roles) else "application",
            "outline": f"Post {i+1}: {roles[i] if i < len(roles) else 'continuation'} angle on '{theme}'",
            "transcript_source": "",
        }
        for i in range(num_posts)
    ]


def _schedule_series(company: str, num_posts: int) -> list:
    """Use temporal orchestrator to schedule series posts."""
    try:
        from backend.src.services.temporal_orchestrator import compute_publish_dates_optimized
        return compute_publish_dates_optimized(company, num_posts)
    except Exception:
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        # Fallback: Tue/Thu at 14:00 UTC
        dates = []
        cursor = now + timedelta(days=2)
        while len(dates) < num_posts:
            if cursor.weekday() in (1, 3):  # Tue, Thu
                dates.append(cursor.replace(hour=14, minute=0, second=0, microsecond=0))
            cursor += timedelta(days=1)
        return dates


# ------------------------------------------------------------------
# Context gathering helpers
# ------------------------------------------------------------------

def _gather_transcript_context(company: str) -> str:
    parts = []
    t_dir = P.transcripts_dir(company)
    if not t_dir.exists():
        return ""
    for f in sorted(t_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
        if f.suffix in (".txt", ".md") and f.stat().st_size < 50_000:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")[:2000]
                parts.append(f"[{f.name}]\n{text}")
            except Exception:
                pass
    return "\n\n---\n\n".join(parts)


def _gather_strategy_context(company: str) -> str:
    cs_dir = P.content_strategy_dir(company)
    if not cs_dir.exists():
        return ""
    for f in sorted(cs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:1]:
        if f.suffix in (".md", ".txt"):
            try:
                return f.read_text(encoding="utf-8", errors="replace")[:2000]
            except Exception:
                pass
    return ""


def _gather_icp_context(company: str) -> str:
    icp_path = P.icp_definition_path(company)
    if not icp_path.exists():
        return ""
    try:
        icp = json.loads(icp_path.read_text(encoding="utf-8"))
        return icp.get("description", "")
    except Exception:
        return ""


# ------------------------------------------------------------------
# Series-aware Stelle context
# ------------------------------------------------------------------

def get_stelle_series_context(company: str) -> str:
    """Check for an active series and return Stelle context for the next unwritten post.

    Returns empty string if no series is active or all posts are written.
    """
    active = get_active_series(company)
    if not active:
        return ""

    for series in active:
        if series.get("status") == "wrapping":
            continue

        posts = series.get("posts", [])
        # Find first unwritten post
        next_post = None
        for p in posts:
            if p.get("status") == "unwritten":
                next_post = p
                break

        if next_post is None:
            continue

        # Build context from previous posts in the series
        position = next_post["position"]
        total = series["num_posts"]
        theme = series["theme"]
        series_id = series["series_id"]

        lines = [
            f"\n\nCONTENT SERIES — POST {position} OF {total}",
            f"Series: \"{theme}\" (id: {series_id})",
            f"Arc role: {next_post.get('arc_role', 'continuation')}",
            f"This post should: {next_post.get('outline', 'continue the series')}",
        ]

        if next_post.get("transcript_source"):
            lines.append(f"Source material: {next_post['transcript_source']}")

        # Summarize what previous posts covered
        prev_posts = [p for p in posts if p["position"] < position and p["status"] != "unwritten"]
        if prev_posts:
            lines.append("\nPrevious posts in this series:")
            for pp in prev_posts:
                status_marker = "✓" if pp["status"] in ("published", "scored") else "◻"
                lines.append(f"  {status_marker} Post {pp['position']} ({pp['arc_role']}): {pp['outline'][:100]}")
            lines.append(
                "\nReference previous posts to reward return readers. "
                "Phrases like 'as I wrote about last week' or 'building on my earlier post about X' "
                "create continuity."
            )

        # Next posts preview (so Stelle knows what's coming and doesn't steal future angles)
        future_posts = [p for p in posts if p["position"] > position]
        if future_posts:
            lines.append("\nUpcoming posts (do NOT cover these angles yet):")
            for fp in future_posts[:2]:
                lines.append(f"  Post {fp['position']} ({fp['arc_role']}): {fp['outline'][:80]}")

        lines.append(
            f"\nTag this draft with series_id={series_id} position={position} "
            "in the output JSON meta field."
        )

        return "\n".join(lines)

    return ""


# ------------------------------------------------------------------
# Post lifecycle management
# ------------------------------------------------------------------

def mark_post_written(company: str, series_id: str, position: int, local_post_id: str = "") -> bool:
    """Mark a series post as written after Stelle generates it."""
    all_series = _load_all_series(company)
    for s in all_series:
        if s.get("series_id") != series_id:
            continue
        for p in s.get("posts", []):
            if p["position"] == position:
                p["status"] = "written"
                if local_post_id:
                    p["local_post_id"] = local_post_id
                s["updated_at"] = _now()
                _save_all_series(company, all_series)
                logger.info(
                    "[series_engine] Marked post %d as written in series '%s' (%s)",
                    position, s["theme"], company,
                )
                return True
    return False


def mark_post_published(company: str, series_id: str, position: int, ordinal_post_id: str = "") -> bool:
    """Mark a series post as published after Ordinal push."""
    all_series = _load_all_series(company)
    for s in all_series:
        if s.get("series_id") != series_id:
            continue
        for p in s.get("posts", []):
            if p["position"] == position:
                p["status"] = "published"
                if ordinal_post_id:
                    p["ordinal_post_id"] = ordinal_post_id
                s["updated_at"] = _now()
                _save_all_series(company, all_series)
                return True
    return False


def mark_post_scored(
    company: str,
    series_id: str,
    position: int,
    engagement_score: float,
) -> bool:
    """Mark a series post as scored with engagement data."""
    all_series = _load_all_series(company)
    for s in all_series:
        if s.get("series_id") != series_id:
            continue
        for p in s.get("posts", []):
            if p["position"] == position:
                p["status"] = "scored"
                p["engagement_score"] = round(engagement_score, 4)
                p["scored_at"] = _now()
                s["updated_at"] = _now()
                _save_all_series(company, all_series)
                return True
    return False


def _find_series_for_post(company: str, local_post_id: str) -> tuple[dict, dict] | None:
    """Find which series and position a local post belongs to."""
    for s in _load_all_series(company):
        for p in s.get("posts", []):
            if p.get("local_post_id") == local_post_id:
                return s, p
    return None


def _find_series_for_ordinal_post(company: str, ordinal_post_id: str) -> tuple[dict, dict] | None:
    """Find which series and position an Ordinal post belongs to."""
    for s in _load_all_series(company):
        for p in s.get("posts", []):
            if p.get("ordinal_post_id") == ordinal_post_id:
                return s, p
    return None


# ------------------------------------------------------------------
# Series performance tracker
# ------------------------------------------------------------------

_DEFAULT_DECLINE_THRESHOLD = 2  # cold-start default


def _compute_decline_threshold(company: str) -> int:
    """Learn the decline threshold from the client's engagement variance.

    High-variance clients need more consecutive declines to be confident
    the series is truly declining (vs normal fluctuation). Low-variance
    clients can wrap earlier.

    Returns an integer >= 2.
    """
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        scored = [o for o in rm._state.get("observations", []) if o.get("status") in ("scored", "finalized")]
        if len(scored) < 10:
            return _DEFAULT_DECLINE_THRESHOLD

        rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
        mean_r = sum(rewards) / len(rewards)
        variance = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std = variance ** 0.5

        # Scale with std: low variance (std<0.3) → 2, high variance (std>1) → 4+
        threshold = round(2 + std * 2)
        return threshold or _DEFAULT_DECLINE_THRESHOLD
    except Exception:
        return _DEFAULT_DECLINE_THRESHOLD


def check_series_health(company: str) -> list[dict]:
    """Check engagement trends for all active series.

    Called by ordinal_sync after RuanMei scoring.
    Updates series status and recommendations.

    Returns list of series that changed status.
    """
    decline_threshold = _compute_decline_threshold(company)

    all_series = _load_all_series(company)
    changed = []

    for s in all_series:
        if s.get("status") not in ("active", "extending"):
            continue

        posts = s.get("posts", [])
        scored_posts = [p for p in posts if p.get("status") in ("scored", "finalized") and p.get("engagement_score", 0) != 0]

        if len(scored_posts) < 2:
            continue  # Need at least 2 scored posts for trend detection

        # Sort by position
        scored_posts.sort(key=lambda p: p["position"])
        scores = [p["engagement_score"] for p in scored_posts]

        # Detect consecutive decline
        consecutive_declines = 0
        for i in range(1, len(scores)):
            if scores[i] < scores[i - 1]:
                consecutive_declines += 1
            else:
                consecutive_declines = 0

        # Detect acceleration
        is_accelerating = len(scores) >= 2 and all(
            scores[i] > scores[i - 1] for i in range(1, len(scores))
        )

        old_status = s["status"]
        old_trend = s.get("engagement_trend", "")

        if consecutive_declines >= decline_threshold:
            s["engagement_trend"] = "declining"
            s["recommendation"] = (
                f"Series '{s['theme']}' has {consecutive_declines} consecutive declining posts. "
                "Recommend wrapping the series early — audience interest is fading."
            )
            if s["status"] == "active":
                s["status"] = "wrapping"
                logger.info(
                    "[series_engine] Series '%s' set to wrapping for %s (declining engagement)",
                    s["theme"], company,
                )

        elif is_accelerating:
            s["engagement_trend"] = "accelerating"
            s["recommendation"] = (
                f"Series '{s['theme']}' has accelerating engagement across {len(scores)} posts. "
                "Consider extending with 1-2 additional posts."
            )
            if s["status"] == "active":
                # Check if we're near the end of the planned series
                unwritten = sum(1 for p in posts if p["status"] == "unwritten")
                if unwritten <= 1:
                    s["status"] = "extending"
                    logger.info(
                        "[series_engine] Series '%s' extending for %s (accelerating)",
                        s["theme"], company,
                    )
                    # Auto-extend: add 2 posts to the series
                    extend_series(company, s["series_id"], extra_posts=2)
        else:
            s["engagement_trend"] = "stable"
            s["recommendation"] = ""

        # Check if series is complete
        all_done = all(p.get("status") in ("scored", "published") for p in posts)
        if all_done:
            s["status"] = "complete"

        if s["status"] != old_status or s.get("engagement_trend") != old_trend:
            s["updated_at"] = _now()
            changed.append({"series_id": s["series_id"], "theme": s["theme"],
                            "old_status": old_status, "new_status": s["status"],
                            "trend": s["engagement_trend"]})

    if changed:
        _save_all_series(company, all_series)

    # B.2: Persist series state for dashboard
    _persist_series_state(company, all_series)

    return changed


def _persist_series_state(company: str, all_series: list[dict]) -> None:
    """Write series tracking state for dashboard visibility."""
    active = [s for s in all_series if s.get("status") in ("active", "extending")]
    completed = sum(1 for s in all_series if s.get("status") == "complete")

    state = {
        "company": company,
        "last_computed": _now(),
        "active_series": [
            {
                "series_id": s.get("series_id", ""),
                "theme": s.get("theme", ""),
                "total_posts": s.get("num_posts", 0),
                "completed_posts": sum(1 for p in s.get("posts", []) if p.get("status") in ("scored", "published")),
                "health": s.get("engagement_trend", ""),
                "status": s.get("status", ""),
            }
            for s in active
        ],
        "completed_series": completed,
    }
    path = P.memory_dir(company) / "series_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def update_series_from_ruan_mei(company: str) -> int:
    """Match scored RuanMei observations to series posts and update scores.

    Called by ordinal_sync after RuanMei scoring. Matches by local_post_id
    or ordinal_post_id.

    Returns number of series posts updated.
    """
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
    except Exception:
        return 0

    all_series = _load_all_series(company)
    if not all_series:
        return 0

    scored_obs = [
        o for o in rm._state.get("observations", [])
        if o.get("status") in ("scored", "finalized")
    ]

    updated = 0
    for s in all_series:
        for p in s.get("posts", []):
            if p.get("status") in ("scored", "finalized"):
                continue  # Already scored

            local_id = p.get("local_post_id", "")
            ordinal_id = p.get("ordinal_post_id", "")

            if not local_id and not ordinal_id:
                continue

            for obs in scored_obs:
                matched = False
                if ordinal_id and obs.get("ordinal_post_id") == ordinal_id:
                    matched = True
                elif local_id and obs.get("local_post_id") == local_id:
                    matched = True
                    # Backfill ordinal_post_id if available
                    if obs.get("ordinal_post_id") and not ordinal_id:
                        p["ordinal_post_id"] = obs["ordinal_post_id"]

                if matched:
                    reward = obs.get("reward", {}).get("immediate", 0)
                    p["status"] = "scored"
                    p["engagement_score"] = round(reward, 4)
                    p["scored_at"] = _now()
                    s["updated_at"] = _now()
                    updated += 1
                    break

    if updated:
        _save_all_series(company, all_series)
        logger.info("[series_engine] Updated %d series posts from RuanMei for %s", updated, company)

    return updated


# ------------------------------------------------------------------
# Extension planner
# ------------------------------------------------------------------

def extend_series(company: str, series_id: str, extra_posts: int = 2) -> dict | None:
    """Add posts to an active/extending series.

    Called when engagement is accelerating and the series is near completion.
    """
    all_series = _load_all_series(company)
    target = None
    for s in all_series:
        if s.get("series_id") == series_id:
            target = s
            break

    if target is None or target["status"] not in ("active", "extending"):
        return None

    current_count = len(target["posts"])
    new_positions = list(range(current_count + 1, current_count + extra_posts + 1))

    # Schedule the extension posts
    scheduled = _schedule_series(company, extra_posts)

    for i, pos in enumerate(new_positions):
        sched = scheduled[i].isoformat() if i < len(scheduled) else ""
        target["posts"].append(asdict(SeriesPost(
            position=pos,
            arc_role="application" if pos < current_count + extra_posts else "synthesis",
            outline=f"Extension post {pos}: continue the '{target['theme']}' series",
            transcript_source="",
            scheduled_at=sched,
        )))

    target["num_posts"] = len(target["posts"])
    target["status"] = "active"
    target["updated_at"] = _now()

    _save_all_series(company, all_series)
    logger.info(
        "[series_engine] Extended series '%s' by %d posts for %s",
        target["theme"], extra_posts, company,
    )

    return target
