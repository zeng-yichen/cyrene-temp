"""Cross-client learning infrastructure (Breakthrough #8).

New clients inherit patterns from similar existing clients instead of
starting from zero. This is a data layer that other components can
optionally read from — it does NOT modify LOLA, Cyrene, or any
existing component.

Usage:
    from backend.src.utils.cross_client import (
        build_client_profile,
        get_similar_client,
        get_cold_start_seeds,
        update_universal_patterns,
    )

    # During ordinal_sync (after scoring):
    build_client_profile("example-client")

    # For new clients:
    similar = get_similar_client("new-client")
    seeds = get_cold_start_seeds("new-client")
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)

MIN_OBS_FOR_PROFILE = 5
MIN_OBS_FOR_NEW_CLIENT = 5  # below this, client is considered "new"
UNIVERSAL_PATTERN_MIN_CLIENTS = 3


# ------------------------------------------------------------------
# Client profile extraction
# ------------------------------------------------------------------

def build_client_profile(company: str) -> Optional[dict]:
    """Extract and save a client profile vector from scored observations.

    Returns the profile dict or None if insufficient data.
    """
    # Load RuanMei state
    state = _load_ruan_mei_state(company)
    if state is None:
        return None

    scored = [o for o in state.get("observations", []) if o.get("status") in ("scored", "finalized")]
    if len(scored) < MIN_OBS_FOR_PROFILE:
        logger.debug(
            "[cross_client] %s has %d scored obs (need %d), skipping profile",
            company, len(scored), MIN_OBS_FOR_PROFILE,
        )
        return None

    profile: dict = {
        "company": company,
        "observation_count": len(scored),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Mean Cyrene dimension scores
    cyrene_dims_agg: dict[str, list[float]] = {}
    for obs in scored:
        dims = obs.get("cyrene_dimensions", {})
        if isinstance(dims, dict):
            for k, v in dims.items():
                if isinstance(v, (int, float)):
                    cyrene_dims_agg.setdefault(k, []).append(float(v))
    if cyrene_dims_agg:
        profile["cyrene_dimensions_mean"] = {
            k: round(sum(v) / len(v), 4) for k, v in cyrene_dims_agg.items()
        }
        profile["cyrene_dimension_set"] = "emergent" if len(cyrene_dims_agg) > 5 else "fixed"
    else:
        profile["cyrene_dimensions_mean"] = {}
        profile["cyrene_dimension_set"] = "none"

    # 2. Learned depth weights
    dw_path = vortex.memory_dir(company) / "depth_weights.json"
    if dw_path.exists():
        try:
            dw = json.loads(dw_path.read_text(encoding="utf-8"))
            profile["depth_weights"] = dw.get("weights", {})
        except Exception:
            profile["depth_weights"] = {}
    else:
        profile["depth_weights"] = {}

    # 3. Posting cadence (median gap in days)
    timestamps = []
    for obs in scored:
        ts = obs.get("posted_at") or obs.get("recorded_at", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                timestamps.append(dt)
            except (ValueError, TypeError):
                pass
    timestamps.sort()
    if len(timestamps) >= 2:
        gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() / 86400
                for i in range(len(timestamps) - 1)]
        gaps.sort()
        mid = len(gaps) // 2
        profile["posting_cadence_median_days"] = round(gaps[mid], 2)
    else:
        profile["posting_cadence_median_days"] = None

    # 4. Content length distribution
    lengths = []
    for obs in scored:
        body = obs.get("posted_body") or obs.get("post_body", "")
        if body:
            lengths.append(len(body))
    if lengths:
        lengths.sort()
        n = len(lengths)
        profile["content_length"] = {
            "median": lengths[n // 2],
            "p25": lengths[max(0, n // 4)],
            "p75": lengths[min(n - 1, 3 * n // 4)],
        }
    else:
        profile["content_length"] = {"median": 0, "p25": 0, "p75": 0}

    # 5. Top 3 performing posts by reward (replaces deprecated LOLA arms)
    top_posts = []
    scored_by_reward = sorted(
        scored,
        key=lambda o: o.get("reward", {}).get("immediate", 0),
        reverse=True,
    )
    for obs in scored_by_reward[:3]:
        r = obs.get("reward", {})
        body = (obs.get("posted_body") or obs.get("post_body") or "").strip()
        top_posts.append({
            "reward": round(r.get("immediate", 0), 4),
            "impressions": r.get("raw_metrics", {}).get("impressions", 0),
            "hook": body[:100] if body else "",
        })
    profile["top_posts"] = top_posts

    # 6. Engagement distribution summary
    rewards = [
        obs.get("reward", {}).get("immediate", 0)
        for obs in scored
        if obs.get("reward", {}).get("immediate") is not None
    ]
    if rewards:
        mean_r = sum(rewards) / len(rewards)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1))
        profile["engagement_distribution"] = {
            "mean": round(mean_r, 4),
            "std": round(std_r, 4),
            "count": len(rewards),
        }
    else:
        profile["engagement_distribution"] = {"mean": 0.0, "std": 0.0, "count": 0}

    # 7. Numeric feature vector for similarity computation
    profile["_numeric_vector"] = _build_numeric_vector(profile)

    # Save
    profile_path = vortex.memory_dir(company) / "client_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = profile_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    tmp.rename(profile_path)

    logger.info(
        "[cross_client] Built profile for %s: %d obs, %d cyrene dims, %d LOLA arms",
        company, len(scored),
        len(profile.get("cyrene_dimensions_mean", {})),
        len(top_arms),
    )

    return profile


def _build_numeric_vector(profile: dict) -> list[float]:
    """Build a fixed-order numeric vector from a client profile for similarity computation."""
    vec = []

    # Engagement stats
    eng = profile.get("engagement_distribution", {})
    vec.append(eng.get("mean", 0.0))
    vec.append(eng.get("std", 0.0))

    # Content length stats
    cl = profile.get("content_length", {})
    vec.append(cl.get("median", 0) / 1000.0)  # normalize to ~1
    vec.append(cl.get("p25", 0) / 1000.0)
    vec.append(cl.get("p75", 0) / 1000.0)

    # Posting cadence
    cad = profile.get("posting_cadence_median_days")
    vec.append(cad / 7.0 if cad is not None else 0.5)  # normalize to ~1

    # Depth weights
    dw = profile.get("depth_weights", {})
    vec.append(dw.get("comments", 1.0))
    vec.append(dw.get("reposts", 1.0))
    vec.append(dw.get("reactions", 1.0))

    # Observation count (normalized)
    vec.append(profile.get("observation_count", 0) / 50.0)

    # LOLA arm performance
    arms = profile.get("top_lola_arms", [])
    for i in range(3):
        if i < len(arms):
            vec.append(arms[i].get("mean_reward", 0.0))
        else:
            vec.append(0.0)

    return vec


# ------------------------------------------------------------------
# Similarity computation
# ------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_all_profiles() -> dict[str, dict]:
    """Load all client profiles from disk."""
    profiles = {}
    memory_root = vortex.MEMORY_ROOT
    if not memory_root.exists():
        return profiles

    for company_dir in memory_root.iterdir():
        if not company_dir.is_dir():
            continue
        profile_path = company_dir / "client_profile.json"
        if not profile_path.exists():
            continue
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profiles[company_dir.name] = profile
        except Exception:
            continue

    return profiles


def get_similar_client(company: str) -> Optional[str]:
    """Find the most similar existing client for a new client.

    Returns the company slug of the most similar client, or None if
    no suitable match exists.

    A client is considered "new" if it has < MIN_OBS_FOR_NEW_CLIENT
    scored observations.
    """
    # Check if this client is actually new
    state = _load_ruan_mei_state(company)
    if state is not None:
        scored = [o for o in state.get("observations", []) if o.get("status") in ("scored", "finalized")]
        if len(scored) >= MIN_OBS_FOR_NEW_CLIENT:
            logger.debug("[cross_client] %s has %d obs, not a new client", company, len(scored))
            return None

    # Load all existing profiles
    profiles = _load_all_profiles()
    if not profiles:
        return None

    # Remove self if present
    profiles.pop(company, None)

    if not profiles:
        return None

    # Check if the new client has a profile (unlikely but possible with 3-4 obs)
    new_profile_path = vortex.memory_dir(company) / "client_profile.json"
    new_vec = None
    if new_profile_path.exists():
        try:
            new_profile = json.loads(new_profile_path.read_text(encoding="utf-8"))
            new_vec = new_profile.get("_numeric_vector")
        except Exception:
            pass

    if new_vec is None:
        # Use a default vector (zeros) — will match based on existing client
        # profile structure; essentially picks the "average" client
        # Try to build a minimal profile from whatever data exists
        new_vec = _build_numeric_vector({})

    # Find most similar
    best_company = None
    best_sim = -1.0

    for other_company, profile in profiles.items():
        other_vec = profile.get("_numeric_vector", [])
        if not other_vec or len(other_vec) != len(new_vec):
            continue
        sim = _cosine_similarity(new_vec, other_vec)
        if sim > best_sim:
            best_sim = sim
            best_company = other_company

    if best_company is not None:
        logger.info(
            "[cross_client] Most similar client for %s: %s (similarity=%.4f)",
            company, best_company, best_sim,
        )

    return best_company


def get_segment_model_seed(company: str) -> Optional[dict]:
    """Find the best cross-client segment model to transfer to ``company``.

    Unlike ``get_cold_start_seeds``, this function does NOT require the target
    client to be "new" (< 5 observations). A client with 10-14 scored
    observations might still not have its own segment model (n < 15 threshold)
    but should still benefit from transfer learning. Similarly, a client
    whose own segment model failed to train (embedding outage, etc.) should
    be able to fall back to a similar client's model.

    Returns the most similar client's ``segment_model.json`` augmented with
    a ``source_client`` field, or None if no suitable donor exists.
    """
    profiles = _load_all_profiles()
    profiles.pop(company, None)
    if not profiles:
        return None

    # Build or load the target client's profile vector for similarity lookup.
    target_profile_path = vortex.memory_dir(company) / "client_profile.json"
    target_vec: Optional[list[float]] = None
    if target_profile_path.exists():
        try:
            target_profile = json.loads(target_profile_path.read_text(encoding="utf-8"))
            target_vec = target_profile.get("_numeric_vector")
        except Exception:
            pass
    if target_vec is None:
        # Client has no profile yet — use a zero vector (picks the "most
        # average" donor client). Same fallback as get_similar_client.
        target_vec = _build_numeric_vector({})

    # Find the most similar client that actually has a segment model.
    best_company: Optional[str] = None
    best_sim = -1.0
    for other, profile in profiles.items():
        other_vec = profile.get("_numeric_vector", [])
        if not other_vec or len(other_vec) != len(target_vec):
            continue
        seg_path = vortex.memory_dir(other) / "segment_model.json"
        if not seg_path.exists():
            continue
        sim = _cosine_similarity(target_vec, other_vec)
        if sim > best_sim:
            best_sim = sim
            best_company = other

    if best_company is None:
        return None

    try:
        donor_model = json.loads(
            (vortex.memory_dir(best_company) / "segment_model.json").read_text(encoding="utf-8")
        )
    except Exception:
        return None

    logger.info(
        "[cross_client] Segment model seed for %s: donor=%s (similarity=%.4f, "
        "donor n=%d, donor LOO R²=%s)",
        company, best_company, best_sim,
        donor_model.get("observation_count", 0),
        donor_model.get("loo_r_squared", "n/a"),
    )

    return {
        "weights": donor_model.get("weights", []),
        "bias": donor_model.get("bias", 0.0),
        "embedding_dim": donor_model.get("embedding_dim", 0),
        "embedding_model": donor_model.get("embedding_model", ""),
        "ridge_alpha": donor_model.get("ridge_alpha", 0),
        "source_client": best_company,
        "source_similarity": round(best_sim, 4),
        "source_observation_count": donor_model.get("observation_count", 0),
        "source_loo_r_squared": donor_model.get("loo_r_squared", 0),
        "source": "cross_client_seed",
    }


def get_cold_start_seeds(company: str) -> Optional[dict]:
    """Get cold-start seed data from the most similar existing client.

    Returns a dict with LOLA arms, Cyrene weights, and engagement
    predictor coefficients from the similar client — all with reduced
    confidence / inflated uncertainty.

    Returns None if no similar client found.
    """
    similar = get_similar_client(company)
    if similar is None:
        return None

    seeds: dict = {
        "source_client": similar,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    sim_memory = vortex.memory_dir(similar)

    # 1. (LOLA arms removed — cold-start content intelligence now provided
    #    by RuanMei.recommend_context() using LinkedIn-wide data.)
    seeds["lola_arms"] = []  # kept for backward compat with any remaining callers

    # 2. Cyrene dimension weights
    # Check for adaptive config
    cyrene_adaptive_path = sim_memory / "cyrene_adaptive_config.json"
    profile_path = sim_memory / "client_profile.json"

    cyrene_weights = {}
    if cyrene_adaptive_path.exists():
        try:
            config = json.loads(cyrene_adaptive_path.read_text(encoding="utf-8"))
            cyrene_weights = config.get("dimension_weights", {})
        except Exception:
            pass

    if not cyrene_weights and profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            cyrene_weights = profile.get("cyrene_dimensions_mean", {})
        except Exception:
            pass

    seeds["cyrene_weights"] = cyrene_weights

    # 3. Engagement predictor coefficients with inflated residuals
    model_path = sim_memory / "engagement_model.json"
    if model_path.exists():
        try:
            model = json.loads(model_path.read_text(encoding="utf-8"))
            # Inflate residual std by 2x for cross-client uncertainty
            seeded_model = {
                "feature_names": model.get("feature_names", []),
                "coefficients": model.get("coefficients", []),
                "intercept": model.get("intercept", 0),
                "means": model.get("means", []),
                "stds": model.get("stds", []),
                "residual_std": model.get("residual_std", 0) * 2.0,
                "r_squared": model.get("r_squared", 0),
                "source": "cross_client_seed",
                "source_observation_count": model.get("observation_count", 0),
            }
            seeds["engagement_model"] = seeded_model
        except Exception:
            seeds["engagement_model"] = None
    else:
        seeds["engagement_model"] = None

    # 4. Depth weights from similar client
    dw_path = sim_memory / "depth_weights.json"
    if dw_path.exists():
        try:
            dw = json.loads(dw_path.read_text(encoding="utf-8"))
            seeds["depth_weights"] = dw.get("weights", {})
        except Exception:
            seeds["depth_weights"] = {}
    else:
        seeds["depth_weights"] = {}

    # 5. Segment model (embedding → predicted reward projection) from similar
    # client. Unlike the other seeds we don't inflate uncertainty here — the
    # weight vector is what it is; the consumer (transcript_scorer) is already
    # calibrated to expect lower accuracy from a transferred projection because
    # LOO R² is stamped on the model itself.
    seg_path = sim_memory / "segment_model.json"
    if seg_path.exists():
        try:
            seg_model = json.loads(seg_path.read_text(encoding="utf-8"))
            seeds["segment_model"] = {
                "weights": seg_model.get("weights", []),
                "bias": seg_model.get("bias", 0.0),
                "embedding_dim": seg_model.get("embedding_dim", 0),
                "embedding_model": seg_model.get("embedding_model", ""),
                "ridge_alpha": seg_model.get("ridge_alpha", 0),
                "source_client": similar,
                "source_observation_count": seg_model.get("observation_count", 0),
                "source_loo_r_squared": seg_model.get("loo_r_squared", 0),
                "source": "cross_client_seed",
            }
        except Exception:
            seeds["segment_model"] = None
    else:
        seeds["segment_model"] = None

    logger.info(
        "[cross_client] Cold-start seeds for %s from %s: %d LOLA arms, %d cyrene dims, "
        "segment_model=%s",
        company, similar,
        len(seeds.get("lola_arms", [])),
        len(seeds.get("cyrene_weights", {})),
        "yes" if seeds.get("segment_model") else "no",
    )

    return seeds


# ------------------------------------------------------------------
# Universal patterns aggregation
# ------------------------------------------------------------------

def update_universal_patterns() -> list[dict]:
    """Aggregate patterns that hold across 3+ clients. Update memory/our_memory/universal_patterns.json.

    Extracts engagement distribution and depth weight patterns to find
    structural commonalities across clients.
    """
    profiles = _load_all_profiles()
    if len(profiles) < UNIVERSAL_PATTERN_MIN_CLIENTS:
        return []

    patterns: list[dict] = []

    # Pattern 1: Depth weight patterns — which engagement type matters most
    depth_weight_leaders: dict[str, list[str]] = {}  # metric -> [companies]
    for company, profile in profiles.items():
        dw = profile.get("depth_weights", {})
        if dw:
            leader = max(dw, key=lambda k: dw[k])
            depth_weight_leaders.setdefault(leader, []).append(company)

    for metric, companies in depth_weight_leaders.items():
        if len(companies) >= UNIVERSAL_PATTERN_MIN_CLIENTS:
            patterns.append({
                "pattern": f"Engagement metric '{metric}' is the strongest engagement signal",
                "source_clients": companies,
                "confidence": round(len(companies) / len(profiles), 2),
                "pattern_type": "depth_weight",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

    # Pattern 2: Content length sweet spots
    length_ranges: list[tuple[str, int, int]] = []
    for company, profile in profiles.items():
        cl = profile.get("content_length", {})
        median = cl.get("median", 0)
        if median > 0:
            length_ranges.append((company, cl.get("p25", 0), cl.get("p75", 0)))

    if len(length_ranges) >= UNIVERSAL_PATTERN_MIN_CLIENTS:
        all_p25 = sorted(r[1] for r in length_ranges)
        all_p75 = sorted(r[2] for r in length_ranges)
        n = len(length_ranges)
        patterns.append({
            "pattern": (
                f"Effective content length across {n} clients: "
                f"P25={all_p25[n // 2]} chars, P75={all_p75[n // 2]} chars"
            ),
            "source_clients": [r[0] for r in length_ranges],
            "confidence": round(min(n / 10.0, 1.0), 2),
            "pattern_type": "content_length",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    # Pattern 3: Posting cadence patterns
    cadences: list[tuple[str, float]] = []
    for company, profile in profiles.items():
        cad = profile.get("posting_cadence_median_days")
        if cad is not None and cad > 0:
            cadences.append((company, cad))

    if len(cadences) >= UNIVERSAL_PATTERN_MIN_CLIENTS:
        cad_values = sorted(c[1] for c in cadences)
        n = len(cad_values)
        median_cad = cad_values[n // 2]
        patterns.append({
            "pattern": f"Median posting cadence across {n} clients: {median_cad:.1f} days",
            "source_clients": [c[0] for c in cadences],
            "confidence": round(min(n / 10.0, 1.0), 2),
            "pattern_type": "posting_cadence",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    # Pattern 4: High-performing LOLA arm types
    arm_type_rewards: dict[str, list[tuple[str, float]]] = {}
    for company, profile in profiles.items():
        for arm in profile.get("top_lola_arms", []):
            atype = arm.get("arm_type", "topic")
            reward = arm.get("mean_reward", 0)
            arm_type_rewards.setdefault(atype, []).append((company, reward))

    for atype, entries in arm_type_rewards.items():
        unique_companies = list(set(e[0] for e in entries))
        if len(unique_companies) >= UNIVERSAL_PATTERN_MIN_CLIENTS:
            avg_reward = sum(e[1] for e in entries) / len(entries)
            patterns.append({
                "pattern": (
                    f"LOLA arm type '{atype}' appears in top arms for "
                    f"{len(unique_companies)} clients (avg reward: {avg_reward:.3f})"
                ),
                "source_clients": unique_companies,
                "confidence": round(min(len(unique_companies) / len(profiles), 1.0), 2),
                "pattern_type": "lola_arm_type",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

    # Load existing universal_patterns.json and merge/update
    up_path = vortex.our_memory_dir() / "universal_patterns.json"
    existing_patterns = []
    if up_path.exists():
        try:
            existing_patterns = json.loads(up_path.read_text(encoding="utf-8"))
            if not isinstance(existing_patterns, list):
                existing_patterns = []
        except Exception:
            existing_patterns = []

    # Keep existing LLM-generated patterns (from cross_client_learning),
    # replace structural patterns we compute here
    our_types = {p["pattern_type"] for p in patterns}
    kept = [p for p in existing_patterns if p.get("pattern_type") not in our_types
            and p.get("category")]  # LLM-generated have 'category', ours have 'pattern_type'
    final_patterns = kept + patterns

    # Save
    up_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = up_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(final_patterns, indent=2), encoding="utf-8")
    tmp.rename(up_path)

    logger.info(
        "[cross_client] Updated universal patterns: %d structural patterns from %d clients",
        len(patterns), len(profiles),
    )

    return patterns


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_ruan_mei_state(company: str) -> Optional[dict]:
    """Load RuanMei state from SQLite."""
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        return ruan_mei_load(company)
    except Exception:
        return None
