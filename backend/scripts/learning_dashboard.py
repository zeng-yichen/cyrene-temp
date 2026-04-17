#!/usr/bin/env python3
"""Amphoreus Learning Dashboard — reads all client state files and prints
a comprehensive learning progress report.

Usage:
    python3 backend/scripts/learning_dashboard.py --client example-client
    python3 backend/scripts/learning_dashboard.py --all
    python3 backend/scripts/learning_dashboard.py --all --json > /tmp/dashboard.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from backend.src.db import vortex as P


# ------------------------------------------------------------------
# Data collection helpers
# ------------------------------------------------------------------

def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sparkline(values: list[float], width: int = 8) -> str:
    if not values:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    vals = values[-width:]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1.0
    return "".join(bars[min(len(bars) - 1, int((v - mn) / rng * (len(bars) - 1)))] for v in vals)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    n = len(values)
    mn = min(values)
    mx = max(values)
    mean = sum(values) / n
    s = sorted(values)
    median = s[n // 2]
    var = sum((v - mean) ** 2 for v in values) / max(n, 1)
    std = math.sqrt(var)
    return {"min": round(mn, 3), "max": round(mx, 3), "mean": round(mean, 3),
            "median": round(median, 3), "std": round(std, 3)}


def _fmt_num(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


# ------------------------------------------------------------------
# Per-client data
# ------------------------------------------------------------------

def collect_client(company: str) -> dict:
    data: dict = {"company": company}

    # --- RuanMei ---
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        rm_state = ruan_mei_load(company)
    except Exception:
        rm_state = None
    obs = rm_state.get("observations", []) if rm_state else []
    scored = [o for o in obs if o.get("status") == "scored"]
    pending = [o for o in obs if o.get("status") == "pending"]

    data["observations"] = {
        "total": len(obs),
        "scored": len(scored),
        "pending": len(pending),
        "pct_scored": round(len(scored) / max(len(obs), 1) * 100, 1),
    }

    rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
    data["reward_stats"] = _stats(rewards)
    data["reward_sparkline"] = _sparkline(rewards)

    # Raw engagement averages
    impressions = [o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0) for o in scored]
    reactions = [o.get("reward", {}).get("raw_metrics", {}).get("reactions", 0) for o in scored]
    comments = [o.get("reward", {}).get("raw_metrics", {}).get("comments", 0) for o in scored]
    reposts = [o.get("reward", {}).get("raw_metrics", {}).get("reposts", 0) for o in scored]
    n_s = max(len(scored), 1)
    data["engagement"] = {
        "avg_impressions": round(sum(impressions) / n_s, 1),
        "avg_reactions": round(sum(reactions) / n_s, 1),
        "avg_comments": round(sum(comments) / n_s, 1),
        "avg_reposts": round(sum(reposts) / n_s, 1),
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

    # --- Content Intelligence ---
    n_scored = data["ruan_mei"]["scored"]
    if n_scored >= 10:
        ci_phase = "analyst"
        ci_label = "Phase 1: Claude-as-Analyst + KNN exploration"
    elif n_scored > 0:
        ci_phase = "cold_start"
        ci_label = f"Phase 0: cold start ({10 - n_scored} more posts needed)"
    else:
        ci_phase = "no_data"
        ci_label = "No scored posts yet"

    # Top posts for display
    top_posts = sorted(
        [o for o in obs if o.get("status") == "scored"],
        key=lambda o: o.get("reward", {}).get("immediate", 0),
        reverse=True,
    )[:3]

    data["content_intelligence"] = {
        "phase": ci_phase,
        "phase_label": ci_label,
        "scored_posts": n_scored,
        "top_posts": [{
            "reward": round(o.get("reward", {}).get("immediate", 0), 3),
            "impressions": o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0),
            "hook": (o.get("posted_body") or o.get("post_body") or "")[:80],
        } for o in top_posts],
    }

    # --- Adaptive config ---
    ac = _load_json(P.memory_dir(company) / "adaptive_config.json")
    data["adaptive"] = {}
    if ac:
        for module in ("permansor", "constitutional", "temporal", "feedback"):
            entry = ac.get(module)
            if entry:
                data["adaptive"][module] = {
                    "tier": entry.get("_tier", "unknown"),
                    "computed_at": entry.get("_computed_at", ""),
                }
                if module == "permansor":
                    data["adaptive"][module]["pass_threshold"] = entry.get("pass_threshold")
                    data["adaptive"][module]["dimension_weights"] = entry.get("dimension_weights")
                elif module == "constitutional":
                    data["adaptive"][module]["soft_principles"] = entry.get("soft_principles", [])

    # --- Adaptive readiness ---
    obs_with_perm = sum(1 for o in obs if o.get("permansor_dimensions"))
    obs_with_const = sum(1 for o in obs if o.get("constitutional_results"))
    data["readiness"] = {
        "permansor_dims_collected": obs_with_perm,
        "permansor_weights_need": max(0, 10 - obs_with_perm),
        "constitutional_collected": obs_with_const,
        "constitutional_soft_need": max(0, 15 - obs_with_const),
        "emergent_dims_need": max(0, 40 - obs_with_perm),
        "current_dimension_set": "fixed_v1",
    }
    # Check latest observation for dimension set
    for o in reversed(obs):
        ds = o.get("permansor_dimension_set")
        if ds:
            data["readiness"]["current_dimension_set"] = ds
            break

    return data


# ------------------------------------------------------------------
# Cross-client summary
# ------------------------------------------------------------------

def cross_client_summary(clients: list[dict]) -> dict:
    total_obs = sum(c.get("observations", {}).get("total", 0) for c in clients)
    total_scored = sum(c.get("observations", {}).get("scored", 0) for c in clients)

    tiers = {"default": 0, "client": 0, "aggregate": 0}
    for c in clients:
        for module, info in c.get("adaptive", {}).items():
            tier = info.get("tier", "default")
            if tier in tiers:
                tiers[tier] += 1

    # Hook library
    hook_lib = _load_json(P.our_memory_dir() / "hook_library.json")
    hook_count = len(hook_lib) if isinstance(hook_lib, list) else 0

    # Universal patterns
    patterns = _load_json(P.our_memory_dir() / "universal_patterns.json")
    pattern_count = len(patterns) if isinstance(patterns, list) else 0

    # Top/bottom arms across all clients
    # Phase distribution
    phase_dist = {"analyst": 0, "cold_start": 0, "no_data": 0}
    for c in clients:
        ci = c.get("content_intelligence", {})
        phase = ci.get("phase", "no_data")
        phase_dist[phase] = phase_dist.get(phase, 0) + 1

    return {
        "total_observations": total_obs,
        "total_scored": total_scored,
        "client_count": len(clients),
        "adaptive_tiers": tiers,
        "hook_library_size": hook_count,
        "universal_patterns": pattern_count,
        "phase_distribution": phase_dist,
    }


# ------------------------------------------------------------------
# Formatters
# ------------------------------------------------------------------

def print_client(c: dict) -> None:
    name = c["company"]
    obs = c["observations"]
    rs = c["reward_stats"]
    eng = c["engagement"]
    cad = c["cadence"]

    print(f"\n  -- {name} --")
    print(f"    Observations:  {obs['total']} total, {obs['scored']} scored ({obs['pct_scored']:.0f}%)")
    print(f"    Reward trend:  {c['reward_sparkline']}  (last 8)")
    print(f"    Reward stats:  mean={rs['mean']:.2f}  median={rs['median']:.2f}  std={rs['std']:.2f}  [{rs['min']:.2f}, {rs['max']:.2f}]")
    print(f"    Engagement:    avg {_fmt_num(eng['avg_impressions'])} impr | {eng['avg_reactions']:.0f} react | {eng['avg_comments']:.1f} comments")
    print(f"    Cadence:       every {cad['avg_days']:.1f} days | {cad['posts_last_7d']} posts last 7d")

    ci = c.get("content_intelligence", {})
    if ci:
        print(f"\n    Content Intelligence: {ci.get('phase_label', 'unknown')}")
        for tp in ci.get("top_posts", []):
            hook = tp.get("hook", "")[:60]
            print(f"      ✦ reward {tp['reward']:+.3f} | {tp['impressions']} impr | \"{hook}\"")


    adaptive = c.get("adaptive", {})
    if adaptive:
        print(f"\n    Adaptive configs:")
        for module, info in adaptive.items():
            tier = info.get("tier", "default")
            extras = ""
            if module == "permansor" and info.get("pass_threshold"):
                extras = f" threshold={info['pass_threshold']}"
            if module == "constitutional" and info.get("soft_principles"):
                extras = f" soft={info['soft_principles']}"
            print(f"      {module:<16s} {tier}{extras}")

    rd = c.get("readiness", {})
    print(f"\n    Adaptive readiness:")
    perm_need = rd.get("permansor_weights_need", 10)
    const_need = rd.get("constitutional_soft_need", 15)
    emerg_need = rd.get("emergent_dims_need", 40)
    perm_col = rd.get("permansor_dims_collected", 0)
    const_col = rd.get("constitutional_collected", 0)
    perm_status = "READY" if perm_need == 0 else f"{perm_need} more needed"
    const_status = "READY" if const_need == 0 else f"{const_need} more needed"
    emerg_status = "READY" if emerg_need == 0 else f"{emerg_need} more needed"
    print(f"      Permansor weights:  {perm_col}/10 obs ({perm_status})")
    print(f"      Constitutional:     {const_col}/15 obs ({const_status})")
    print(f"      Emergent dims:      {perm_col}/40 obs ({emerg_status})")
    print(f"      Dimension set:      {rd.get('current_dimension_set', 'fixed_v1')}")


def print_summary(summary: dict) -> None:
    print(f"\n{'═' * 62}")
    print(f"  CROSS-CLIENT SUMMARY")
    print(f"    Total observations: {summary['total_observations']} across {summary['client_count']} clients")
    print(f"    Scored: {summary['total_scored']}")
    tiers = summary["adaptive_tiers"]
    print(f"    Adaptive tiers: {tiers.get('client', 0)} client, "
          f"{tiers.get('aggregate', 0)} aggregate, {tiers.get('default', 0)} default")
    print(f"    Hook library: {summary['hook_library_size']} hooks")
    print(f"    Universal patterns: {summary['universal_patterns']}")
    phases = summary.get("phase_distribution", {})
    if phases:
        print(f"    Content intelligence: {phases.get('analyst', 0)} analyst, "
              f"{phases.get('cold_start', 0)} cold start, {phases.get('no_data', 0)} no data")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Amphoreus Learning Dashboard")
    parser.add_argument("--client", help="Show single client")
    parser.add_argument("--all", action="store_true", help="Show all clients")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.client and not args.all:
        parser.print_help()
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M %Z")

    # Collect client list
    if args.client:
        companies = [args.client]
    else:
        # Any memory/{slug} directory is a candidate client; collect_client
        # itself skips those with no SQLite state.
        companies = sorted([
            d.name for d in P.MEMORY_ROOT.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "our_memory"
        ])

    clients = [collect_client(c) for c in companies]

    if args.json:
        output = {
            "generated_at": now,
            "clients": clients,
            "summary": cross_client_summary(clients) if args.all else None,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
        return

    print(f"╔{'═' * 62}╗")
    print(f"║  AMPHOREUS LEARNING DASHBOARD  --  {now:<27s}║")
    print(f"╠{'═' * 62}╣")

    for c in clients:
        print_client(c)

    if args.all:
        print_summary(cross_client_summary(clients))

    print(f"{'═' * 64}")


if __name__ == "__main__":
    main()
