"""Draft scorer — rank candidate posts using continuous embedding similarity.

Primary scoring path: k-NN in embedding space. The draft is embedded and
compared to all scored historical posts by cosine similarity. The
similarity-weighted average reward of the k nearest neighbors is the
predicted engagement score. No categorical features, no human-readable
labels — the 1536-dimensional embedding captures everything.

Secondary scoring path: the analyst's coefficient model (char_count,
posting_hour, quote-hook bonus). These are legitimate continuous features
that don't impose a human taxonomy. The old format_tag regression feature
has been removed — it was a categorical bottleneck.

Also computes:
- Exploration value: 1 - max_similarity to any scored post
- Trajectory alignment: cosine similarity to the reward-weighted centroid
  from the embedding trajectory model (warns about audience fatigue)

Usage:
    from backend.src.utils.draft_scorer import score_drafts

    ranked = score_drafts("example-client", [
        {"text": "We were brought in to review...", "scheduled_hour": 9},
        {"text": "If I got dropped into a Director...", "scheduled_hour": 14},
    ])
    for r in ranked:
        print(f"#{r['rank']} score={r['predicted_score']:+.3f} — {r['explanation']}")
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from backend.src.db import vortex

logger = logging.getLogger(__name__)


@dataclass
class ScoredDraft:
    """A draft post with predicted engagement score, exploration value, and explanation."""
    rank: int
    text: str
    predicted_score: float
    exploration_value: float   # 0-1: how much the system would learn from this post
    features: dict             # extracted feature values
    explanation: str           # human-readable "why this score"
    model_source: str          # "analyst_model+embedding_knn" | "embedding_knn" | "no_model"
    # Continuous provenance — no binary ready/not-ready gate. Downstream
    # code surfaces these raw numbers and decides how much to trust the
    # coefficient path based on the actual model fit, not on a hand-tuned
    # cutoff. Bitter Lesson: feed the raw metadata, let the reader judge.
    loo_r2: float = -999.0
    n_observations: int = 0


def _compute_training_stats(company: str) -> dict:
    """Compute feature normalization stats from the client's actual observations.

    Only continuous features: char_count and posting_hour. The old format_tag
    target encoding has been removed — categorical labels are no longer part
    of the scoring pipeline.
    """
    import math

    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return {}

    if state is None:
        return {}

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") in ("scored", "finalized")
        and o.get("reward", {}).get("immediate") is not None
    ]
    if not scored:
        return {}

    char_counts = [
        len(o.get("posted_body") or o.get("post_body") or "")
        for o in scored
    ]
    char_mean = sum(char_counts) / len(char_counts)
    char_std = math.sqrt(
        sum((c - char_mean) ** 2 for c in char_counts) / max(len(char_counts) - 1, 1)
    ) if len(char_counts) > 1 else 1.0

    hours = []
    for o in scored:
        ts = o.get("posted_at", "")
        if ts:
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                hours.append(float(dt.hour))
            except Exception:
                pass
    hour_mean = sum(hours) / len(hours) if hours else 12.0
    hour_std = math.sqrt(
        sum((h - hour_mean) ** 2 for h in hours) / max(len(hours) - 1, 1)
    ) if len(hours) > 1 else 3.0

    all_rewards = [o.get("reward", {}).get("immediate", 0) for o in scored]
    global_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    return {
        "char_count_mean": round(char_mean, 1),
        "char_count_std": round(char_std, 1),
        "posting_hour_mean": round(hour_mean, 1),
        "posting_hour_std": round(hour_std, 1),
        "reward_mean": round(global_mean, 4),
        "observation_count": len(scored),
    }


def _score_by_trajectory_offset_knn(
    company: str,
    text: str,
    k: int = 5,
) -> tuple[Optional[float], str]:
    """Score a draft by k-NN in trajectory-relative embedding space.

    Instead of asking "which historical posts are similar to this draft?"
    (absolute k-NN), this asks "which historical posts are in the same
    *direction* from our current position as this draft?"

    The offset query:
        query = normalize(draft_embedding - recent_centroid)

    Each historical post is also projected relative to the recent centroid:
        h_offset = normalize(post_embedding - recent_centroid)

    k-NN is run on these offsets. The result is: "given where we've been
    recently, which historical transitions does this draft resemble?"

    A contrarian post after a run of case studies looks like a different
    historical moment than the same post would look after a run of hot takes.
    The trajectory-relative query surfaces that context; absolute k-NN misses it.
    """
    import numpy as np

    traj_path = vortex.memory_dir(company) / "embedding_trajectory.json"
    if not traj_path.exists():
        return None, "No trajectory model (need 8+ scored posts)"

    try:
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return None, "Failed to load trajectory model"

    recent_centroid = traj.get("recent_centroid")
    if not recent_centroid:
        return None, "No recent centroid in trajectory model"

    from backend.src.utils.post_embeddings import get_post_embeddings, embed_text

    draft_emb = embed_text(text)
    if draft_emb is None:
        return None, "Failed to embed draft"

    centroid_np = np.array(recent_centroid, dtype=np.float64)
    centroid_norm = np.linalg.norm(centroid_np)
    if centroid_norm > 1e-10:
        centroid_np = centroid_np / centroid_norm

    draft_np = np.array(draft_emb, dtype=np.float64)
    draft_norm = np.linalg.norm(draft_np)
    if draft_norm > 1e-10:
        draft_np = draft_np / draft_norm

    # Offset: draft direction minus recent trajectory direction
    offset = draft_np - centroid_np
    offset_norm = np.linalg.norm(offset)
    if offset_norm < 1e-10:
        return None, "Draft is indistinguishable from recent trajectory (no directional signal)"
    offset = offset / offset_norm

    embeddings = get_post_embeddings(company)
    if not embeddings:
        return None, "No post embeddings"

    # Score each historical post by cosine similarity of its offset to draft's offset
    neighbors: list[tuple[str, float]] = []
    for h, emb in embeddings.items():
        emb_np = np.array(emb, dtype=np.float64)
        emb_norm = np.linalg.norm(emb_np)
        if emb_norm < 1e-10:
            continue
        emb_np = emb_np / emb_norm
        h_offset = emb_np - centroid_np
        h_norm = np.linalg.norm(h_offset)
        if h_norm < 1e-10:
            continue
        h_offset = h_offset / h_norm
        sim = float(np.dot(offset, h_offset))
        neighbors.append((h, sim))

    neighbors.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = neighbors[:k]

    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return None, "Failed to load observations"
    if state is None:
        return None, "No observation state"

    obs_by_hash = {
        o.get("post_hash"): o
        for o in state.get("observations", [])
        if o.get("status") in ("scored", "finalized")
    }

    weighted_sum = 0.0
    weight_sum = 0.0
    neighbor_details = []
    for h, sim in top_neighbors:
        obs = obs_by_hash.get(h)
        if not obs:
            continue
        reward = obs.get("reward", {}).get("immediate")
        if reward is None:
            continue
        weighted_sum += sim * reward
        weight_sum += sim
        body = (obs.get("posted_body") or obs.get("post_body") or "")
        neighbor_details.append(
            f"sim={sim:.2f}/reward={reward:+.2f} \"{body[:50].strip()}...\""
        )

    if weight_sum < 1e-6:
        return None, "No trajectory-relative neighbors with rewards"

    pred = weighted_sum / weight_sum
    explanation = (
        f"Trajectory-offset k-NN {pred:+.3f} (relative to current position): "
        + "; ".join(neighbor_details[:3])
    )
    return pred, explanation


def _score_by_embedding_knn(
    company: str,
    text: str,
    k: int = 5,
) -> tuple[Optional[float], str]:
    """Score a draft by k-NN similarity to the client's scored observations.

    Embeds the draft, finds the k most similar scored posts by cosine
    similarity, and returns a similarity-weighted average of their rewards.

    This is the embedding-based scoring path — no format_tag, no topic_tag,
    no categorical features. The embedding captures everything about the
    post in a continuous vector. Works alongside the coefficient-based path.

    Returns (predicted_score, explanation) or (None, explanation) on failure.
    """
    from backend.src.utils.post_embeddings import (
        get_post_embeddings, embed_text, find_similar, cosine_similarity,
    )

    embeddings = get_post_embeddings(company)
    if not embeddings:
        return None, "No post embeddings available"

    draft_emb = embed_text(text)
    if draft_emb is None:
        return None, "Failed to embed draft"

    similar = find_similar(draft_emb, embeddings, top_k=k)
    if not similar:
        return None, "No similar posts found"

    # Load observations to get rewards for the similar posts
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return None, "Failed to load observations"
    if state is None:
        return None, "No observation state"

    obs_by_hash = {
        o.get("post_hash"): o
        for o in state.get("observations", [])
        if o.get("status") in ("scored", "finalized")
    }

    # Similarity-weighted reward average
    weighted_sum = 0.0
    weight_sum = 0.0
    neighbor_details = []
    for h, sim in similar:
        obs = obs_by_hash.get(h)
        if not obs:
            continue
        reward = obs.get("reward", {}).get("immediate")
        if reward is None:
            continue
        weighted_sum += sim * reward
        weight_sum += sim
        body = (obs.get("posted_body") or obs.get("post_body") or "")
        neighbor_details.append(
            f"sim={sim:.2f}/reward={reward:+.2f} "
            f"\"{body[:60].strip()}...\""
        )

    if weight_sum < 1e-6:
        return None, "No similar posts with rewards"

    pred = weighted_sum / weight_sum
    explanation = (
        f"k-NN score {pred:+.3f} from {len(neighbor_details)} neighbors: "
        + "; ".join(neighbor_details[:3])
    )

    return pred, explanation


def score_drafts(
    company: str,
    drafts: list[dict],
    default_hour: int = 9,
) -> list[ScoredDraft]:
    """Score and rank a batch of draft posts using continuous features only.

    Two independent predictions per draft:

    1. **Coefficient-based** (from the analyst's regression model):
       Applies learned coefficients to char_count, posting_hour.
       No categorical features — operates on continuous dimensions.

    2. **Embedding-based** (k-NN in continuous vector space):
       Embeds the draft, finds the k most similar scored posts by cosine
       similarity, returns their similarity-weighted reward average.

    The final score is the average of both paths (when both are available).
    """
    if not drafts:
        return []

    model_spec, model_source = _load_model(company)

    # No binary readiness gate — always compute the coefficient path when a
    # model exists, and surface raw provenance (loo_r2, n_observations) so
    # downstream readers decide how much to trust it from the numbers rather
    # than from a hand-tuned cutoff.
    training_stats: dict = {}
    training_stats_full: dict = {}
    if model_source == "analyst_model":
        training_stats_full = _compute_training_stats(company)
        training_stats = training_stats_full

    loo_r2_val = float(model_spec.get("loo_r2", -999.0))
    n_obs_val = int(training_stats_full.get("observation_count", 0))

    scored: list[ScoredDraft] = []

    for i, draft in enumerate(drafts):
        text = draft.get("text", "")
        if not text.strip():
            scored.append(ScoredDraft(
                rank=0, text=text, predicted_score=0.0,
                exploration_value=0.0,
                features={}, explanation="Empty draft",
                model_source="no_model",
                loo_r2=loo_r2_val, n_observations=n_obs_val,
            ))
            continue

        hour = draft.get("scheduled_hour", default_hour)
        features = _extract_features(text, hour)

        # Path 1: coefficient-based (analyst's regression on continuous features)
        coeff_score, coeff_explanation = _apply_model(
            features, model_spec, model_source,
            training_stats=training_stats,
        )

        # Path 2: absolute embedding k-NN
        knn_score, knn_explanation = _score_by_embedding_knn(company, text)

        # Path 3: trajectory-offset k-NN
        # Asks "which historical posts are in the same direction from our
        # current position?" rather than "which posts are similar in absolute
        # content space?" Conditions the score on sequential context.
        traj_score, traj_explanation = _score_by_trajectory_offset_knn(company, text)

        # Combine: average all available paths
        path_scores: list[float] = []
        active_sources: list[str] = []
        explanation_parts: list[str] = []

        if model_source != "no_model":
            path_scores.append(coeff_score)
            active_sources.append("analyst_model")
            explanation_parts.append(f"  Coeff: {coeff_explanation}")
        if knn_score is not None:
            path_scores.append(knn_score)
            active_sources.append("embedding_knn")
            explanation_parts.append(f"  k-NN: {knn_explanation}")
        if traj_score is not None:
            path_scores.append(traj_score)
            active_sources.append("trajectory_knn")
            explanation_parts.append(f"  Trajectory: {traj_explanation}")

        if path_scores:
            combined = sum(path_scores) / len(path_scores)
            source = "+".join(active_sources)
            scores_str = " + ".join(
                f"{s:+.3f}" for s in path_scores
            )
            explanation = (
                f"Combined {combined:+.3f} = avg({scores_str})\n"
                + "\n".join(explanation_parts)
            )
        else:
            combined = 0.0
            source = "no_model"
            explanation = "No scoring model available."

        features["knn_score"] = knn_score
        features["traj_score"] = traj_score
        features["coeff_score"] = coeff_score if model_source != "no_model" else None

        # Exploration value: how much would the system learn from publishing
        # this post? Measured as 1 - max_similarity to any scored post.
        # A draft identical to a historical post (sim=0.95) teaches nothing
        # (exploration=0.05). A genuinely novel draft (sim=0.4) is highly
        # informative (exploration=0.6).
        exploration = _compute_exploration_value(company, text)

        scored.append(ScoredDraft(
            rank=0,
            text=text,
            predicted_score=round(combined, 4),
            exploration_value=round(exploration, 4),
            features=features,
            explanation=explanation,
            model_source=source,
            loo_r2=loo_r2_val,
            n_observations=n_obs_val,
        ))

    # Rank by predicted engagement (highest first).
    # Exploration value is shown but does NOT affect ranking —
    # the operator decides the engagement/exploration tradeoff.
    scored.sort(key=lambda s: s.predicted_score, reverse=True)
    for i, s in enumerate(scored):
        s.rank = i + 1

    return scored


def _compute_exploration_value(company: str, text: str) -> float:
    """How much the system would learn from publishing this draft.

    Measured as 1.0 - max_cosine_similarity to any scored observation.
    A draft that's 0.95 similar to a historical post teaches nothing new
    (exploration_value = 0.05). A draft that's only 0.40 similar to the
    nearest scored post is genuinely novel (exploration_value = 0.60).

    Returns 0.0 if embeddings are unavailable (can't assess novelty).
    """
    try:
        from backend.src.utils.post_embeddings import (
            get_post_embeddings, embed_text, find_similar,
        )
        embeddings = get_post_embeddings(company)
        if not embeddings:
            return 0.0
        draft_emb = embed_text(text)
        if draft_emb is None:
            return 0.0
        # Find the single most similar observation
        nearest = find_similar(draft_emb, embeddings, top_k=1)
        if not nearest:
            return 1.0  # no observations at all → maximally novel
        max_sim = nearest[0][1]
        return max(0.0, 1.0 - max_sim)
    except Exception:
        return 0.0


def _load_model(company: str) -> tuple[dict, str]:
    """Coefficient-model path is retired (2026-04-11).

    Previously this loaded a regression model the analyst wrote to
    analyst_findings.json. The analyst has been deleted as a Bitter
    Lesson violation. The draft scorer now relies on embedding k-NN
    + trajectory-offset k-NN paths exclusively.
    """
    return {}, "no_model"


def _extract_features(text: str, posting_hour: int) -> dict:
    """Extract continuous features from a draft. No categorical labels."""
    features: dict = {
        "char_count": len(text),
        "posting_hour": posting_hour,
        "has_quote_hook": False,
    }

    opening = text[:200]
    if '"' in opening or '\u201c' in opening or '\u201d' in opening:
        quote_matches = re.findall(r'["\u201c](.+?)["\u201d]', opening)
        if any(len(q) > 15 for q in quote_matches):
            features["has_quote_hook"] = True

    return features


def _apply_model(
    features: dict,
    model_spec: dict,
    model_source: str,
    training_stats: Optional[dict] = None,
) -> tuple[float, str]:
    """Apply the analyst's regression model to continuous features only.

    Uses char_count and posting_hour (z-normalized from training data)
    plus the analyst's optional quote-hook bonus. No categorical features.
    """
    if model_source == "no_model":
        return 0.0, "No analyst model available. Drafts are unscored."

    coefficients = model_spec.get("coefficients", {})
    intercept = model_spec.get("intercept", 0.0)
    stats = training_stats or {}

    char_count = features.get("char_count", 1500)
    posting_hour = features.get("posting_hour", 9)

    score = intercept
    explanation_parts = []

    char_coeff = coefficients.get("char_count", 0)
    if char_coeff != 0:
        char_mean = stats.get("char_count_mean", 2000)
        char_std = stats.get("char_count_std", 500)
        char_z = (char_count - char_mean) / char_std if char_std > 1e-6 else 0
        char_contribution = char_coeff * char_z
        score += char_contribution
        if abs(char_contribution) > 0.01:
            explanation_parts.append(
                f"length {char_count} chars (vs avg {char_mean:.0f}): {char_contribution:+.3f}"
            )

    hour_coeff = coefficients.get("posting_hour", 0)
    if hour_coeff != 0:
        hour_mean = stats.get("posting_hour_mean", 12)
        hour_std = stats.get("posting_hour_std", 3)
        hour_z = (posting_hour - hour_mean) / hour_std if hour_std > 1e-6 else 0
        hour_contribution = hour_coeff * hour_z
        score += hour_contribution
        if abs(hour_contribution) > 0.01:
            explanation_parts.append(
                f"posting hour {posting_hour}:00 (vs avg {hour_mean:.0f}:00): {hour_contribution:+.3f}"
            )

    heuristic = model_spec.get("heuristic_layer", {})
    hook_bonus_str = heuristic.get("scoring_guidance", {}).get("hook_bonus", "")
    if features.get("has_quote_hook") and hook_bonus_str:
        try:
            hook_bonus = float(hook_bonus_str.split()[0].replace("+", ""))
        except (ValueError, IndexError):
            hook_bonus = 0
        if hook_bonus > 0:
            score += hook_bonus
            explanation_parts.append(f"quote hook (analyst bonus): +{hook_bonus:.2f}")

    loo_r2 = model_spec.get("loo_r2", "?")
    if explanation_parts:
        explanation = f"Score {score:+.3f} = " + " + ".join(explanation_parts)
        explanation += f" (model LOO R²={loo_r2})"
    else:
        explanation = f"Score {score:+.3f} — no distinguishing features (LOO R²={loo_r2})"

    return score, explanation


# ------------------------------------------------------------------
# Batch scoring for Stelle integration
# ------------------------------------------------------------------

def score_and_explain_batch(company: str, posts: list[dict]) -> str:
    """Score a batch of posts and return a human-readable ranking.

    Designed to be called after Stelle generates drafts, producing a
    summary the operator can use to decide publishing order.

    Each post dict needs "text" and optionally "scheduled_hour" and "hook"
    (the first line, for display).

    Returns a formatted markdown string with the ranking.
    """
    if not posts:
        return "No drafts to score."

    scored = score_drafts(company, posts)

    if scored[0].model_source == "no_model":
        return (
            "No analyst model available for this client. "
            "Run the analyst agent first to build a predictive model."
        )

    lines = [
        "## Draft Ranking (by predicted engagement)\n",
        f"*Model: {scored[0].model_source}, "
        f"based on {len(scored)} drafts*\n",
    ]

    for s in scored:
        hook = s.text[:80].replace("\n", " ").strip()
        # Raw numbers only — no categorical high/moderate/low label. Readers
        # (model or human) decide what "high exploration" means from the
        # continuous score.
        lines.append(
            f"**#{s.rank}** | predicted=`{s.predicted_score:+.3f}` | "
            f"exploration=`{s.exploration_value:.2f}` | "
            f"{s.features.get('char_count', '?')} chars | "
            f"loo_r2=`{s.loo_r2:+.3f}` n_obs=`{s.n_observations}`"
        )
        lines.append(f"> {hook}...")
        lines.append(f"  {s.explanation}")
        lines.append("")

    # Summary
    best = scored[0]
    worst = scored[-1]
    spread = best.predicted_score - worst.predicted_score
    lines.append(
        f"**Spread:** {spread:.3f} (#{best.rank} vs #{worst.rank})"
    )
    most_novel = max(scored, key=lambda s: s.exploration_value)
    lines.append(
        f"**Most novel:** #{most_novel.rank} "
        f"(exploration={most_novel.exploration_value:.2f})"
    )

    return "\n".join(lines)
