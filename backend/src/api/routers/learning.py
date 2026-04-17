"""Learning Intelligence API — serves client learning dashboard data."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import APIRouter

from backend.src.db import vortex as P

router = APIRouter(prefix="/api/learning", tags=["learning"])


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sparkline(values: list[float], width: int = 12) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    return "".join(bars[min(len(bars) - 1, int((v - mn) / rng * (len(bars) - 1)))] for v in vals)


def _collect_client(company: str) -> dict:
    """Collect all learning data for a single client."""
    data: dict = {"company": company}

    # --- RuanMei ---
    rm_state = None
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        rm_state = ruan_mei_load(company)
    except Exception:
        rm_state = None
    obs = rm_state.get("observations", []) if rm_state else []
    scored = [o for o in obs if o.get("status") in ("scored", "finalized")]
    pending = [o for o in obs if o.get("status") == "pending"]

    data["observations"] = {
        "total": len(obs), "scored": len(scored), "pending": len(pending),
        "pct_scored": round(len(scored) / max(len(obs), 1) * 100, 1),
    }

    rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
    if rewards:
        rewards_sorted = sorted(rewards)
        n = len(rewards)
        data["reward_stats"] = {
            "min": round(min(rewards), 3), "max": round(max(rewards), 3),
            "mean": round(sum(rewards) / n, 3),
            "median": round(rewards_sorted[n // 2], 3),
            "std": round(math.sqrt(sum((r - sum(rewards)/n)**2 for r in rewards) / n), 3),
        }
    else:
        data["reward_stats"] = {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}

    data["reward_sparkline"] = _sparkline(rewards)

    # Engagement averages
    n_s = max(len(scored), 1)
    data["engagement"] = {
        "avg_impressions": round(sum(o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0) for o in scored) / n_s, 1),
        "avg_reactions": round(sum(o.get("reward", {}).get("raw_metrics", {}).get("reactions", 0) for o in scored) / n_s, 1),
        "avg_comments": round(sum(o.get("reward", {}).get("raw_metrics", {}).get("comments", 0) for o in scored) / n_s, 1),
    }

    # Cadence
    timestamps = []
    for o in scored:
        ts = o.get("posted_at", "")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except Exception:
                pass
    timestamps.sort()
    gaps = [(timestamps[i] - timestamps[i-1]).total_seconds() / 86400
            for i in range(1, len(timestamps)) if (timestamps[i] - timestamps[i-1]).total_seconds() > 0]
    data["cadence"] = {
        "avg_days": round(sum(gaps) / max(len(gaps), 1), 1) if gaps else 0,
        "posts_last_7d": sum(1 for t in timestamps if (datetime.now(timezone.utc) - t).days <= 7),
    }

    # --- Observation tags (display only) ---
    from collections import Counter
    topic_dist = Counter(o.get("topic_tag") for o in scored if o.get("topic_tag"))
    format_dist = Counter(o.get("format_tag") for o in scored if o.get("format_tag"))
    data["tags"] = {
        "tagged_count": sum(1 for o in scored if o.get("topic_tag")),
        "topics": dict(topic_dist.most_common()),
        "formats": dict(format_dist.most_common()),
    }

    # --- Adaptive readiness ---
    obs_with_perm = sum(1 for o in obs if o.get("cyrene_dimensions"))
    data["readiness"] = {
        "cyrene_dims": obs_with_perm,
        "cyrene_weights_ready": obs_with_perm >= 10,
        "freeform_critic_active": len(scored) >= 10,
        "observation_tagger_active": data["tags"]["tagged_count"] > 0,
    }

    return data


@router.get("/clients")
async def list_learning_clients():
    """List all clients with learning data."""
    clients = []
    if P.MEMORY_ROOT.exists():
        for d in sorted(P.MEMORY_ROOT.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name != "our_memory":
                rm = None
                try:
                    from backend.src.db.local import initialize_db, ruan_mei_load
                    initialize_db()
                    rm = ruan_mei_load(d.name)
                except Exception:
                    rm = None
                if rm:
                    obs = rm.get("observations", [])
                    scored = sum(1 for o in obs if o.get("status") in ("scored", "finalized"))
                    clients.append({
                        "slug": d.name, "scored": scored,
                        "total": len(obs),
                    })
    return {"clients": clients}


@router.get("/clients/{company}")
async def get_learning_detail(company: str):
    """Get full learning dashboard data for a client."""
    return _collect_client(company)


@router.get("/cross-client")
async def get_cross_client():
    """Get cross-client learning summary."""
    hook_lib = _load_json(P.our_memory_dir() / "hook_library.json")
    patterns = _load_json(P.our_memory_dir() / "universal_patterns.json")

    # Collect RuanMei-based stats across all clients
    client_summaries = []
    if P.MEMORY_ROOT.exists():
        for d in P.MEMORY_ROOT.iterdir():
            if not d.is_dir() or d.name.startswith(".") or d.name == "our_memory":
                continue
            rm = None
            try:
                from backend.src.db.local import initialize_db, ruan_mei_load
                initialize_db()
                rm = ruan_mei_load(d.name)
            except Exception:
                rm = None
            if not rm:
                continue
            scored = [o for o in rm.get("observations", []) if o.get("status") in ("scored", "finalized")]
            if not scored:
                continue
            rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
            client_summaries.append({
                "company": d.name,
                "scored_posts": len(scored),
                "avg_reward": round(sum(rewards) / len(rewards), 3),
            })

    client_summaries.sort(key=lambda c: c["avg_reward"], reverse=True)

    return {
        "hook_library_size": len(hook_lib) if isinstance(hook_lib, list) else 0,
        "universal_patterns": len(patterns) if isinstance(patterns, list) else 0,
        "patterns": patterns[:5] if isinstance(patterns, list) else [],
        "client_summaries": client_summaries,
    }
