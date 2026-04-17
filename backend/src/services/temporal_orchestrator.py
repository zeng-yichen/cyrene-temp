"""Temporal Orchestrator — platform-native scheduling optimizer.

Replaces Hyacinthia's fixed Mon/Wed/Thu or Tue/Thu cadence with data-driven
scheduling that considers:

1. **Client rhythm model** — per-client day-of-week and hour-of-day performance
   derived from RuanMei scored observations (no new data collection needed).
2. **Topic-time correlation** — certain topic types perform better on certain days
   (extracted from RuanMei descriptor analysis + posted_at).
3. **Velocity-aware acceleration** — when topic_velocity detects a trending topic,
   accelerate that post to the next available slot.
4. **90-minute amplification awareness** — schedule posts when the client's ICP
   audience is most active (derived from historical engagement timing).
5. **Cool-down periods** — after a high-performing post, allow 24-48h before the
   next post to avoid feed cannibalization.

Integration points:
- INPUT: RuanMei scored observations (posted_at, reward, descriptor.analysis)
- OUTPUT: Replaces `_compute_publish_dates` in Hyacinthia and the `/push-all`
  cadence logic in posts.py.

Usage:
    from backend.src.services.temporal_orchestrator import (
        compute_optimal_schedule,
        recommend_next_slot,
        build_client_rhythm,
    )

    # For batch scheduling:
    schedule = compute_optimal_schedule("example-client", num_posts=6)
    # → [datetime(...), datetime(...), ...]  each with optimal day+hour

    # For single post:
    slot = recommend_next_slot("example-client")
    # → datetime(2026, 4, 5, 14, 0) — next Thursday at 14:00 UTC

    # For Stelle context injection:
    context = build_scheduling_context("example-client")
    # → "Optimal posting: Thu/Tue 14:00-15:00 UTC. Avoid Mon/Sat."
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

# Minimum scored observations before we trust per-client patterns.
# Below this, fall back to cross-client aggregate patterns.
_MIN_CLIENT_OBS = 8

# Minimum observations per time bucket to trust the signal.
_MIN_BUCKET_OBS = 3

# Default constants (used when insufficient data for adaptive thresholds).
_DEFAULT_HOUR_UTC = 14
_DEFAULT_COOLDOWN_HOURS = 36
_DEFAULT_HIGH_PERFORMER_THRESHOLD = 0.5
_DEFAULT_MAX_POSTS_PER_DAY = 1
_DEFAULT_BEST_DAYS = [1, 3, 2]  # Tue, Thu, Wed


# ------------------------------------------------------------------
# Adaptive config
# ------------------------------------------------------------------

class TemporalAdaptiveConfig:
    """Data-driven scheduling thresholds computed from client's post history."""

    MODULE_NAME = "temporal"

    def resolve(self, company: str) -> dict:
        """Compute adaptive scheduling params from client's own data."""
        obs = _load_observations(company)
        if len(obs) < _MIN_CLIENT_OBS:
            return self.get_defaults()

        rewards = [o.get("reward", {}).get("immediate", 0) for o in obs]
        rewards.sort()

        # High performer threshold: 75th percentile of client's rewards
        high_perf = rewards[int(len(rewards) * 0.75)] if rewards else _DEFAULT_HIGH_PERFORMER_THRESHOLD

        # Cooldown: derived from gaps between above-median consecutive posts
        timestamps = []
        for o in obs:
            ts = _parse_timestamp(o.get("posted_at", ""))
            if ts:
                timestamps.append((ts, o.get("reward", {}).get("immediate", 0)))
        timestamps.sort()

        median_reward = rewards[len(rewards) // 2] if rewards else 0
        above_median_gaps = []
        for i in range(1, len(timestamps)):
            if timestamps[i - 1][1] > median_reward:
                gap_h = (timestamps[i][0] - timestamps[i - 1][0]).total_seconds() / 3600
                if 0 < gap_h < 240:  # ignore gaps > 10 days (posting breaks)
                    above_median_gaps.append(gap_h)

        cooldown = _DEFAULT_COOLDOWN_HOURS
        if len(above_median_gaps) >= 3:
            above_median_gaps.sort()
            # Use the gap that above-median posts tend to follow
            cooldown = above_median_gaps[len(above_median_gaps) // 2]
            # No hard clamp — let the data decide. soft_bound logs anomalies.
            from backend.src.utils.adaptive_config import soft_bound
            cooldown = soft_bound(cooldown, above_median_gaps, _DEFAULT_COOLDOWN_HOURS)

        return {
            "cooldown_hours": round(cooldown, 1),
            "high_performer_threshold": round(high_perf, 4),
            "_tier": "client",
        }

    def get_defaults(self) -> dict:
        return {
            "cooldown_hours": _DEFAULT_COOLDOWN_HOURS,
            "high_performer_threshold": _DEFAULT_HIGH_PERFORMER_THRESHOLD,
            "_tier": "default",
        }


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class DayHourScore:
    """Performance score for a (day_of_week, hour) slot."""
    day: int           # 0=Monday ... 6=Sunday
    hour: int          # 0-23 UTC
    avg_reward: float
    count: int
    day_name: str = ""

    def __post_init__(self):
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.day_name = days[self.day] if 0 <= self.day <= 6 else "Unknown"


@dataclass
class ClientRhythm:
    """Per-client temporal performance model built from RuanMei observations."""
    company: str
    total_observations: int
    day_scores: dict[int, float]       # day_of_week → avg reward
    hour_scores: dict[int, float]      # hour_utc → avg reward
    day_hour_scores: list[DayHourScore] # combined, sorted best-first
    best_days: list[int]               # top 3 days by reward
    best_hours: list[int]              # top 3 hours by reward
    avg_gap_hours: float               # mean hours between posts
    last_post_at: Optional[datetime]
    last_post_reward: float
    is_fallback: bool                  # True if using cross-client data


# ------------------------------------------------------------------
# Rhythm extraction from RuanMei
# ------------------------------------------------------------------

def _load_observations(company: str) -> list[dict]:
    """Load scored observations for a client from RuanMei state."""
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        return [
            o for o in rm._state.get("observations", [])
            if o.get("status") in ("scored", "finalized") and o.get("posted_at")
        ]
    except Exception as e:
        logger.warning("[temporal] Failed to load RuanMei state for %s: %s", company, e)
        return []


def _load_all_observations() -> list[dict]:
    """Load scored observations across ALL clients for cross-client fallback."""
    all_obs = []
    if not P.MEMORY_ROOT.exists():
        return all_obs
    for client_dir in P.MEMORY_ROOT.iterdir():
        if not client_dir.is_dir():
            continue
        company = client_dir.name
        all_obs.extend(_load_observations(company))
    return all_obs


def _parse_timestamp(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def build_client_rhythm(company: str) -> ClientRhythm:
    """Build the temporal performance model for a client.

    Falls back to cross-client aggregate if the client has fewer than
    _MIN_CLIENT_OBS scored observations.
    """
    obs = _load_observations(company)
    is_fallback = len(obs) < _MIN_CLIENT_OBS

    if is_fallback:
        obs = _load_all_observations()
        if len(obs) < _MIN_CLIENT_OBS:
            return _empty_rhythm(company)

    # Extract day-of-week and hour-of-day patterns
    day_rewards: dict[int, list[float]] = defaultdict(list)
    hour_rewards: dict[int, list[float]] = defaultdict(list)
    day_hour_rewards: dict[tuple[int, int], list[float]] = defaultdict(list)
    timestamps: list[datetime] = []

    for o in obs:
        dt = _parse_timestamp(o.get("posted_at", ""))
        if dt is None:
            continue
        reward = o.get("reward", {}).get("immediate", 0)
        day = dt.weekday()
        hour = dt.hour

        day_rewards[day].append(reward)
        hour_rewards[hour].append(reward)
        day_hour_rewards[(day, hour)].append(reward)
        timestamps.append(dt)

    # Compute averages
    day_scores = {
        d: sum(rs) / len(rs)
        for d, rs in day_rewards.items()
        if len(rs) >= _MIN_BUCKET_OBS
    }
    hour_scores = {
        h: sum(rs) / len(rs)
        for h, rs in hour_rewards.items()
        if len(rs) >= _MIN_BUCKET_OBS
    }

    # Combined day+hour scores
    day_hour_list = []
    for (d, h), rs in day_hour_rewards.items():
        if len(rs) >= 2:  # Lower threshold for combined since it's sparser
            day_hour_list.append(DayHourScore(
                day=d, hour=h,
                avg_reward=sum(rs) / len(rs),
                count=len(rs),
            ))
    day_hour_list.sort(key=lambda x: x.avg_reward, reverse=True)

    # Best days and hours
    best_days = sorted(day_scores, key=day_scores.get, reverse=True)[:3]
    best_hours = sorted(hour_scores, key=hour_scores.get, reverse=True)[:3]

    # If no hour data meets threshold, use the hour with the most observations
    if not best_hours and hour_rewards:
        best_hours = sorted(hour_rewards, key=lambda h: len(hour_rewards[h]), reverse=True)[:3]
    if not best_hours:
        best_hours = [_DEFAULT_HOUR_UTC]

    # Average gap between posts
    timestamps.sort()
    gaps = []
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
        if gap > 0:
            gaps.append(gap)
    avg_gap = sum(gaps) / len(gaps) if gaps else 48.0

    # Last post info (for client-specific obs only, not fallback)
    client_obs = _load_observations(company) if is_fallback else obs
    last_dt = None
    last_reward = 0.0
    if client_obs:
        latest = max(client_obs, key=lambda o: o.get("posted_at", ""))
        last_dt = _parse_timestamp(latest.get("posted_at", ""))
        last_reward = latest.get("reward", {}).get("immediate", 0)

    return ClientRhythm(
        company=company,
        total_observations=len(obs),
        day_scores=day_scores,
        hour_scores=hour_scores,
        day_hour_scores=day_hour_list,
        best_days=best_days if best_days else [1, 3, 2],  # Default: Tue, Thu, Wed
        best_hours=best_hours,
        avg_gap_hours=round(avg_gap, 1),
        last_post_at=last_dt,
        last_post_reward=last_reward,
        is_fallback=is_fallback,
    )


def _empty_rhythm(company: str) -> ClientRhythm:
    # Try cross-client aggregate before hard-coded defaults
    agg_obs = _load_all_observations()
    if len(agg_obs) >= _MIN_CLIENT_OBS:
        agg_rhythm = _build_rhythm_from_obs(agg_obs, company, is_fallback=True)
        if agg_rhythm.best_days:
            return agg_rhythm

    return ClientRhythm(
        company=company,
        total_observations=0,
        day_scores={},
        hour_scores={},
        day_hour_scores=[],
        best_days=[1, 3, 2],  # Last resort: Tue, Thu, Wed
        best_hours=[_DEFAULT_HOUR_UTC],
        avg_gap_hours=48.0,
        last_post_at=None,
        last_post_reward=0.0,
        is_fallback=True,
    )


def _build_rhythm_from_obs(obs: list[dict], company: str, is_fallback: bool = False) -> ClientRhythm:
    """Build a rhythm model from a list of observations (client-specific or aggregate)."""
    day_rewards: dict[int, list[float]] = defaultdict(list)
    hour_rewards: dict[int, list[float]] = defaultdict(list)
    timestamps: list[datetime] = []

    for o in obs:
        dt = _parse_timestamp(o.get("posted_at", ""))
        if dt is None:
            continue
        reward = o.get("reward", {}).get("immediate", 0)
        day_rewards[dt.weekday()].append(reward)
        hour_rewards[dt.hour].append(reward)
        timestamps.append(dt)

    day_scores = {d: sum(rs)/len(rs) for d, rs in day_rewards.items() if len(rs) >= _MIN_BUCKET_OBS}
    hour_scores = {h: sum(rs)/len(rs) for h, rs in hour_rewards.items() if len(rs) >= _MIN_BUCKET_OBS}
    best_days = sorted(day_scores, key=day_scores.get, reverse=True)[:3] if day_scores else [1, 3, 2]
    best_hours = sorted(hour_scores, key=hour_scores.get, reverse=True)[:3] if hour_scores else [_DEFAULT_HOUR_UTC]

    timestamps.sort()
    gaps = [(timestamps[i] - timestamps[i-1]).total_seconds()/3600 for i in range(1, len(timestamps)) if (timestamps[i]-timestamps[i-1]).total_seconds() > 0]
    avg_gap = sum(gaps)/len(gaps) if gaps else 48.0

    return ClientRhythm(
        company=company,
        total_observations=len(obs),
        day_scores=day_scores,
        hour_scores=hour_scores,
        day_hour_scores=[],
        best_days=best_days,
        best_hours=best_hours,
        avg_gap_hours=round(avg_gap, 1),
        last_post_at=None,
        last_post_reward=0.0,
        is_fallback=is_fallback,
    )


# ------------------------------------------------------------------
# Scheduling logic
# ------------------------------------------------------------------

def recommend_next_slot(
    company: str,
    after: datetime | None = None,
    rhythm: ClientRhythm | None = None,
) -> datetime:
    """Recommend the single best next publish time.

    Considers:
    - Client's best days and hours from historical data
    - Cool-down after last post (especially if it was a hit)
    - Never schedules in the past
    """
    if rhythm is None:
        rhythm = build_client_rhythm(company)

    now = datetime.now(timezone.utc)
    earliest = after or now

    # Cool-down: if last post was a hit, wait longer (adaptive thresholds)
    temporal_cfg = TemporalAdaptiveConfig().resolve(company)
    cooldown_hours = temporal_cfg.get("cooldown_hours", _DEFAULT_COOLDOWN_HOURS)
    high_perf_threshold = temporal_cfg.get("high_performer_threshold", _DEFAULT_HIGH_PERFORMER_THRESHOLD)

    if rhythm.last_post_at:
        since_last = (now - rhythm.last_post_at).total_seconds() / 3600
        if rhythm.last_post_reward >= high_perf_threshold and since_last < cooldown_hours:
            cooldown_end = rhythm.last_post_at + timedelta(hours=cooldown_hours)
            if cooldown_end > earliest:
                earliest = cooldown_end

    # Ensure we're at least 2 hours in the future
    if earliest < now + timedelta(hours=2):
        earliest = now + timedelta(hours=2)

    best_hour = rhythm.best_hours[0] if rhythm.best_hours else _DEFAULT_HOUR_UTC
    best_days_set = set(rhythm.best_days[:3]) if rhythm.best_days else {1, 2, 3}

    # Walk forward from earliest to find the next best slot
    candidate = earliest.replace(minute=0, second=0, microsecond=0)
    if candidate.hour > best_hour:
        candidate += timedelta(days=1)
    candidate = candidate.replace(hour=best_hour)

    # Find the next day that's in our best days
    for _ in range(14):  # Max 2 weeks out
        if candidate.weekday() in best_days_set and candidate > earliest:
            return candidate
        candidate += timedelta(days=1)

    # Fallback: next business day at best hour
    candidate = earliest + timedelta(days=1)
    return candidate.replace(hour=best_hour, minute=0, second=0, microsecond=0)


def compute_optimal_schedule(
    company: str,
    num_posts: int,
    start_after: datetime | None = None,
    min_gap_hours: float = 20.0,
    rhythm: ClientRhythm | None = None,
) -> list[datetime]:
    """Compute optimal publish times for a batch of posts.

    Returns a list of datetime objects, one per post, sorted chronologically.
    Uses the client's rhythm model to pick the best available slots.

    Args:
        company: Client company keyword.
        num_posts: Number of posts to schedule.
        start_after: Earliest allowed publish time (default: now + 24h).
        min_gap_hours: Minimum hours between consecutive posts (default: 20).
        rhythm: Pre-built rhythm model (optional, built if not provided).
    """
    if rhythm is None:
        rhythm = build_client_rhythm(company)

    now = datetime.now(timezone.utc)
    earliest = start_after or (now + timedelta(hours=24))

    best_hour = rhythm.best_hours[0] if rhythm.best_hours else _DEFAULT_HOUR_UTC
    best_days = rhythm.best_days[:3] if rhythm.best_days else [1, 3, 2]

    # Build a scoring function for candidate slots
    # Compute weekend effect from client data (not hardcoded)
    weekend_rewards = [v for d, v in rhythm.day_scores.items() if d >= 5]
    weekday_rewards = [v for d, v in rhythm.day_scores.items() if d < 5]
    if len(weekend_rewards) >= 1 and len(weekday_rewards) >= 1:
        weekend_effect = sum(weekend_rewards) / len(weekend_rewards) - sum(weekday_rewards) / len(weekday_rewards)
    else:
        weekend_effect = 0.0  # No data → no penalty (not -0.5)

    def _slot_score(dt: datetime) -> float:
        """Higher is better."""
        day = dt.weekday()
        hour = dt.hour
        day_score = rhythm.day_scores.get(day, 0.0)
        hour_score = rhythm.hour_scores.get(hour, 0.0)
        # Weekend effect: learned from data, not prescribed
        we = weekend_effect if day >= 5 else 0.0
        return day_score + hour_score + we

    # Generate candidate slots for the next 6 weeks.
    # Only generate candidates on the top 3-4 weekdays to avoid noise.
    candidates: list[datetime] = []
    cursor = earliest.replace(minute=0, second=0, microsecond=0)

    hours_to_try = set(rhythm.best_hours[:3])
    hours_to_try.add(best_hour)
    if not hours_to_try:
        hours_to_try = {_DEFAULT_HOUR_UTC}

    # Eligible weekdays: top 4 by day_scores, or top 3 best_days + 1 extra
    eligible_days = set(best_days[:3])
    if rhythm.day_scores:
        ranked_days = sorted(rhythm.day_scores, key=rhythm.day_scores.get, reverse=True)
        for d in ranked_days[:4]:
            eligible_days.add(d)
    # Always include at least Tue, Wed, Thu as fallback
    if len(eligible_days) < 3:
        eligible_days.update({1, 2, 3})
    # Never include weekends unless data specifically supports them
    eligible_days -= {5, 6}

    for day_offset in range(42):  # 6 weeks
        day_dt = cursor + timedelta(days=day_offset)
        if day_dt.weekday() not in eligible_days:
            continue
        for h in hours_to_try:
            slot = day_dt.replace(hour=h)
            if slot > earliest:
                candidates.append(slot)

    # Walk chronologically, rotating through the top days.
    # This guarantees diverse day-of-week distribution.
    rotation_days = list(best_days[:3])
    if len(rotation_days) < 2:
        rotation_days = [1, 3, 2]  # Tue, Thu, Wed default

    schedule: list[datetime] = []
    used_dates: set[str] = set()
    rotation_idx = 0

    # Sort candidates chronologically
    candidates.sort()

    for candidate in candidates:
        if len(schedule) >= num_posts:
            break

        date_key = candidate.strftime("%Y-%m-%d")
        if date_key in used_dates:
            continue

        # Must match current rotation target day
        target_day = rotation_days[rotation_idx % len(rotation_days)]
        if candidate.weekday() != target_day:
            continue

        # Prefer the best hour for this slot
        if candidate.hour != best_hour and any(
            c.weekday() == target_day and c.hour == best_hour
            and c.strftime("%Y-%m-%d") == date_key
            for c in candidates
        ):
            continue  # Skip non-optimal hour if optimal is available same day

        # Check min gap
        too_close = False
        for scheduled in schedule:
            gap = abs((candidate - scheduled).total_seconds()) / 3600
            if gap < min_gap_hours:
                too_close = True
                break
        if too_close:
            continue

        schedule.append(candidate)
        used_dates.add(date_key)
        rotation_idx += 1

    # Sort chronologically
    schedule.sort()

    # If we couldn't fill all slots (unlikely), pad with simple cadence
    while len(schedule) < num_posts:
        last = schedule[-1] if schedule else earliest
        next_slot = last + timedelta(hours=max(min_gap_hours, 48))
        # Avoid weekends
        while next_slot.weekday() >= 5:
            next_slot += timedelta(days=1)
        schedule.append(next_slot)

    return schedule


# ------------------------------------------------------------------
# Velocity-aware rescheduling
# ------------------------------------------------------------------

def should_accelerate_topic(company: str, topic_keywords: list[str]) -> bool:
    """Check if topic_velocity indicates a trending topic worth accelerating.

    Returns True if the topic shows acceleration signal in the velocity file.
    """
    tv_path = P.MEMORY_ROOT / company / "topic_velocity.md"
    if not tv_path.exists():
        return False

    try:
        tv_content = tv_path.read_text(encoding="utf-8").lower()
        # Simple heuristic: if multiple topic keywords appear in recent velocity data
        matches = sum(1 for kw in topic_keywords if kw.lower() in tv_content)
        return matches >= 2
    except Exception:
        return False


# ------------------------------------------------------------------
# Stelle / Hyacinthia integration
# ------------------------------------------------------------------

def build_scheduling_context(company: str) -> str:
    """Build a Stelle-ready context string with scheduling intelligence.

    Injected into the generation prompt so Stelle can consider timing
    when planning posts (e.g., "timely topics should be prioritized for
    the next available Thursday slot").
    """
    rhythm = build_client_rhythm(company)

    if rhythm.total_observations == 0:
        return ""

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    best_day_names = [days[d] for d in rhythm.best_days[:3]]
    worst_days = [
        days[d] for d in sorted(rhythm.day_scores, key=rhythm.day_scores.get)[:2]
        if rhythm.day_scores.get(d, 0) < -0.1
    ]

    lines = ["\n\nSCHEDULING INTELLIGENCE (from historical performance):"]
    source = "cross-client aggregate" if rhythm.is_fallback else f"{rhythm.total_observations} observations"
    lines.append(f"Data source: {source}")
    lines.append(f"Best posting days: {', '.join(best_day_names)}")

    if rhythm.best_hours:
        hour_strs = [f"{h:02d}:00 UTC" for h in rhythm.best_hours[:2]]
        lines.append(f"Best posting hours: {', '.join(hour_strs)}")

    if worst_days:
        lines.append(f"Avoid: {', '.join(worst_days)} (below-average engagement)")

    if rhythm.last_post_at:
        hours_since = (datetime.now(timezone.utc) - rhythm.last_post_at).total_seconds() / 3600
        if hours_since < 24:
            lines.append(f"Last post was {hours_since:.0f}h ago — consider spacing.")
        elif hours_since > 168:
            lines.append(f"No post in {hours_since/24:.0f} days — audience may need re-engagement.")

    temporal_cfg = TemporalAdaptiveConfig().resolve(company)
    high_thresh = temporal_cfg.get("high_performer_threshold", _DEFAULT_HIGH_PERFORMER_THRESHOLD)
    cooldown_h = temporal_cfg.get("cooldown_hours", _DEFAULT_COOLDOWN_HOURS)
    if rhythm.last_post_reward >= high_thresh:
        lines.append(f"Last post performed well — a {cooldown_h:.0f}h cooldown lets it accumulate engagement.")

    result = "\n".join(lines)

    # B.1: Persist temporal state for dashboard
    _persist_temporal_state(company, rhythm, temporal_cfg)

    return result


def _persist_temporal_state(company: str, rhythm: ClientRhythm, cfg: dict) -> None:
    """Write temporal state for dashboard visibility. Only writes if changed."""
    state = {
        "company": company,
        "best_days": rhythm.best_days,
        "best_hours": rhythm.best_hours,
        "cooldown_hours": cfg.get("cooldown_hours", _DEFAULT_COOLDOWN_HOURS),
        "high_performer_threshold": cfg.get("high_performer_threshold", _DEFAULT_HIGH_PERFORMER_THRESHOLD),
        "last_post_reward": round(rhythm.last_post_reward, 4),
        "avg_gap_hours": rhythm.avg_gap_hours,
        "is_fallback": rhythm.is_fallback,
        "adaptive_tier": cfg.get("_tier", "default"),
        "last_computed": datetime.now(timezone.utc).isoformat(),
    }
    path = P.memory_dir(company) / "temporal_state.json"
    # Only write if changed (compare without last_computed)
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old.pop("last_computed", None)
            check = dict(state)
            check.pop("last_computed", None)
            if old == check:
                return
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def compute_publish_dates_optimized(
    company: str,
    num_posts: int,
    start_date: datetime | None = None,
) -> list[datetime]:
    """Drop-in replacement for Hyacinthia._compute_publish_dates.

    Returns datetime objects with optimal hour set (not just 09:00 UTC).
    Falls back to Hyacinthia's original logic if no rhythm data exists.
    """
    rhythm = build_client_rhythm(company)

    if rhythm.total_observations == 0:
        # No data — fall back to standard cadence
        return _fallback_cadence(num_posts, start_date)

    return compute_optimal_schedule(
        company=company,
        num_posts=num_posts,
        start_after=start_date,
        rhythm=rhythm,
    )


def _fallback_cadence(num_posts: int, start_date: datetime | None = None) -> list[datetime]:
    """Fallback cadence using cross-client aggregate, then Mon/Wed/Thu as last resort."""
    # Try cross-client aggregate for best days/hours
    agg_rhythm = _empty_rhythm("_fallback")
    valid_weekdays = set(agg_rhythm.best_days[:3]) if agg_rhythm.best_days else {0, 2, 3}
    best_hour = agg_rhythm.best_hours[0] if agg_rhythm.best_hours else _DEFAULT_HOUR_UTC

    dates = []
    current = start_date or (datetime.now(timezone.utc) + timedelta(days=1))
    current = current.replace(hour=best_hour, minute=0, second=0, microsecond=0)

    while len(dates) < num_posts:
        if current.weekday() in valid_weekdays:
            dates.append(current)
        current += timedelta(days=1)

    return dates
