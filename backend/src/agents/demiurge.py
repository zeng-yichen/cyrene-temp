"""
Demiurge (formerly Cyrene) — SELF-REFINE critique-revise loop for post quality optimization.

Implements the SELF-REFINE architecture (Madaan et al., NeurIPS 2023,
arXiv:2303.17651) as a structured generate→critique→revise loop.

The critic scores each draft along 7 platform-native dimensions:
1. 360Brew Semantic Alignment — profile-content consistency
2. Hook Scroll-Stop Score — first 140 chars effectiveness
3. Save-Worthiness — would a reader hit "Save"?
4. Comment Invitation — does the post leave an open loop?
5. Dwell Time Prediction — length, structure, density
6. Magic Moment Test — "you told my story better than I could've"
7. ICP Resonance — would the target persona find this novel/actionable?

Each dimension gets a 1-5 score with specific actionable feedback.
The reviser only addresses dimensions scoring ≤3.
After the learned iteration ceiling (or pass threshold), ships the best version
by composite score.

Usage:
    from backend.src.agents.cyrene import refine_post, refine_batch

    result = refine_post(
        company="example-client",
        draft_text="My post text...",
        transcript_excerpt="Source transcript...",
        max_iterations=3,
    )
    # result = {"final_text": "...", "iterations": [...], "best_score": 4.2}
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import anthropic

logger = logging.getLogger(__name__)

# Safety ceiling for the critique-revise loop. NOT the typical iteration count —
# most posts pass on iteration 1-2 via the learned pass_threshold. The ceiling is
# a guard against infinite loops, not a target. Set high enough that the learned
# early-exit conditions (pass_threshold, min_improvement) are what actually control
# convergence, not this number.
_DEFAULT_MAX_ITERATIONS = 8
_DEFAULT_PASS_THRESHOLD = 3.5
_DEFAULT_MIN_IMPROVEMENT = 0.15
_MIN_OBS_FOR_FREEFORM = 5     # switch from dimension-based to freeform critic early
_MIN_OBS_FOR_PROJECTION = 20  # activate linear reward projection
_DEFAULT_DIMENSION_WEIGHTS = {
    "360Brew Alignment": 1 / 7,
    "Hook Scroll-Stop": 1 / 7,
    "Save-Worthiness": 1 / 7,
    "Comment Invitation": 1 / 7,
    "Dwell Time": 1 / 7,
    "Magic Moment": 1 / 7,
    "ICP Resonance": 1 / 7,
}
# 10 is sufficient for Spearman correlation of 7 continuous dimension scores
# against continuous engagement. Each score has 5 levels of granularity,
# producing meaningful rank correlations even at small n.
_MIN_OBS_FOR_ADAPTIVE = 10

_client = anthropic.Anthropic()


# ------------------------------------------------------------------
# Adaptive config
# ------------------------------------------------------------------

class CyreneAdaptiveConfig:
    """Data-driven dimension weights and pass threshold.

    Three-tier cascade:
    1. Per-client: Spearman correlation of each dimension score vs engagement
    2. Cross-client aggregate: same analysis across all clients
    3. Defaults: equal weights (1/7), threshold 3.5
    """

    MODULE_NAME = "cyrene"

    def get_defaults(self) -> dict:
        return {
            "dimension_weights": dict(_DEFAULT_DIMENSION_WEIGHTS),
            "pass_threshold": _DEFAULT_PASS_THRESHOLD,
            "min_improvement": _DEFAULT_MIN_IMPROVEMENT,
            "max_iterations": _DEFAULT_MAX_ITERATIONS,
        }

    def sufficient_data(self, company: str) -> bool:
        obs = self._get_observations_with_dims(company)
        return len(obs) >= _MIN_OBS_FOR_ADAPTIVE

    def compute_from_client(self, company: str) -> dict:
        obs = self._get_observations_with_dims(company)
        self._current_company = company
        try:
            return self._compute(obs)
        finally:
            self._current_company = None

    def compute_from_aggregate(self) -> dict:
        from backend.src.db import vortex as P
        all_obs = []
        if P.MEMORY_ROOT.exists():
            for d in P.MEMORY_ROOT.iterdir():
                if d.is_dir() and not d.name.startswith(".") and d.name != "our_memory":
                    all_obs.extend(self._get_observations_with_dims(d.name))
        if len(all_obs) < _MIN_OBS_FOR_ADAPTIVE:
            return {}
        return self._compute(all_obs)

    def resolve(self, company: str) -> dict:
        """Three-tier cascade with cache check."""
        from backend.src.utils.adaptive_config import AdaptiveConfig
        # Inline cascade (avoids subclass boilerplate for this simple case)
        if self.sufficient_data(company):
            try:
                config = self.compute_from_client(company)
                config["_tier"] = "client"
                return config
            except Exception as e:
                logger.debug("[CyreneConfig] Client compute failed: %s", e)
        agg = self.compute_from_aggregate()
        if agg:
            agg["_tier"] = "aggregate"
            return agg
        defaults = self.get_defaults()
        defaults["_tier"] = "default"
        return defaults

    def recompute(self, company: str) -> dict:
        return self.resolve(company)

    def _get_observations_with_dims(self, company: str) -> list[dict]:
        """Load scored observations that have both cyrene_dimensions and engagement."""
        try:
            from backend.src.agents.ruan_mei import RuanMei
            rm = RuanMei(company)
            return [
                o for o in rm._state.get("observations", [])
                if o.get("status") in ("scored", "finalized")
                and o.get("cyrene_dimensions")
                and o.get("reward", {}).get("immediate") is not None
            ]
        except Exception:
            return []

    # ---- company_context carried via call sites; we stash it per-resolve ----
    _current_company: Optional[str] = None

    def _compute(self, observations: list[dict]) -> dict:
        """Compute adaptive weights and threshold from observations."""
        from backend.src.utils.correlation_analyzer import correlate_with_engagement

        # Correlate each dimension score with engagement
        correlations = correlate_with_engagement(
            observations,
            attribute_extractor=lambda obs: {
                k: float(v) for k, v in obs.get("cyrene_dimensions", {}).items()
                if isinstance(v, (int, float))
            },
            min_n=5,
        )

        if not correlations:
            return self.get_defaults()

        # Normalize correlations to weights (clamp negatives to 0, then normalize)
        raw_weights = {k: max(0.0, v) for k, v in correlations.items()}
        total = sum(raw_weights.values())
        if total == 0:
            return self.get_defaults()

        weights = {k: round(v / total, 4) for k, v in raw_weights.items()}

        # Fill in any missing dimensions with small default weight
        for dim in _DEFAULT_DIMENSION_WEIGHTS:
            if dim not in weights:
                weights[dim] = 0.05
        # Re-normalize
        total = sum(weights.values())
        weights = {k: round(v / total, 4) for k, v in weights.items()}

        # Pass threshold grounded in engagement: median composite score of
        # posts that actually performed well (above 40th percentile reward).
        rewards = [obs.get("reward", {}).get("immediate", 0) for obs in observations]
        rewards.sort()
        threshold = _DEFAULT_PASS_THRESHOLD
        if len(rewards) >= 5:
            reward_cutoff = rewards[int(len(rewards) * 0.4)]
            good_composites = []
            for obs in observations:
                r = obs.get("reward", {}).get("immediate", 0)
                if r >= reward_cutoff:
                    dims = obs.get("cyrene_dimensions", {})
                    if dims:
                        wt = sum(weights.get(d, 1 / 7) for d in weights)
                        score = sum(dims.get(d, 3) * weights.get(d, 1 / 7) for d in weights)
                        good_composites.append(score / wt if wt > 0 else 3.0)
            if good_composites:
                good_composites.sort()
                threshold = good_composites[len(good_composites) // 2]

        # min_improvement: from std of historical composite improvements.
        # If improvements are typically small, lower the threshold to avoid
        # wasting iterations. If large, raise it to keep iterating.
        min_imp = _DEFAULT_MIN_IMPROVEMENT
        if len(observations) >= 5:
            composites_for_std = []
            for obs in observations:
                dims = obs.get("cyrene_dimensions", {})
                if dims:
                    wt = sum(weights.get(d, 1 / 7) for d in weights)
                    score = sum(dims.get(d, 3) * weights.get(d, 1 / 7) for d in weights)
                    composites_for_std.append(score / wt if wt > 0 else 3.0)
            if len(composites_for_std) >= 3:
                import math as _math
                mean_c = sum(composites_for_std) / len(composites_for_std)
                var_c = sum((c - mean_c) ** 2 for c in composites_for_std) / len(composites_for_std)
                std_c = _math.sqrt(var_c)
                # Continuous: min_improvement scales linearly with std
                min_imp = round(std_c * 0.6, 3) or _DEFAULT_MIN_IMPROVEMENT

        # Emergent min obs: fast-posting clients reach eligibility faster
        timestamps = [o.get("posted_at", "") for o in observations if o.get("posted_at")]
        if len(timestamps) >= 2:
            timestamps.sort()
            try:
                from datetime import datetime as _dt, timezone as _tz
                first = _dt.fromisoformat(timestamps[0].replace("Z", "+00:00"))
                last = _dt.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
                days_active = max(1, (last - first).days)
            except Exception:
                days_active = 30
        else:
            days_active = 30
        # Emergent min obs: scale with client's posting density.
        # No hard clamp — data-sparse clients naturally get higher thresholds,
        # data-rich clients lower ones. _DEFAULT_EMERGENT_MIN_OBS is the floor.
        emergent_min = days_active * 2
        if emergent_min < _DEFAULT_EMERGENT_MIN_OBS:
            emergent_min = _DEFAULT_EMERGENT_MIN_OBS

        # Dimension cache TTL: scale with observation growth rate
        obs_at_last = len(observations)
        if days_active > 0:
            obs_per_day = obs_at_last / days_active
            # Fast growers get shorter TTL (more frequent rediscovery)
            cache_ttl = round(14 / max(obs_per_day, 0.1))
        else:
            cache_ttl = 14

        # Learned max_iterations: how many critique-revise cycles does this
        # client typically need before the critic is satisfied?
        #
        # The ceiling is a safety net, not a target. The pass_threshold and
        # min_improvement early-exit conditions are what actually drive
        # convergence — the ceiling should be generous enough that they
        # almost always fire before it binds. When data shows that extra
        # iterations don't help (or hurt), the ceiling tightens.
        learned_max = _DEFAULT_MAX_ITERATIONS
        iter_data = [
            (o.get("cyrene_iterations", 0), o.get("reward", {}).get("immediate", 0))
            for o in observations
            if o.get("cyrene_iterations") and o.get("reward", {}).get("immediate") is not None
        ]
        if len(iter_data) >= 10:
            iters = [d[0] for d in iter_data]
            iter_rewards = [d[1] for d in iter_data]
            median_iter = sorted(iters)[len(iters) // 2]

            # Check: do more iterations correlate with worse engagement?
            # If yes, the critic is overpolishing — cap at the median.
            try:
                from backend.src.utils.correlation_analyzer import _spearman_correlation
                iter_engagement_corr = _spearman_correlation(
                    [float(i) for i in iters], iter_rewards,
                )
            except Exception:
                iter_engagement_corr = 0.0

            if iter_engagement_corr < -0.15:
                # More iterations → worse posts. Cap tightly.
                learned_max = max(2, median_iter)
                logger.info(
                    "[CyreneConfig] Overpolishing detected for this client "
                    "(iter-reward corr=%.3f). Capping max_iterations at %d.",
                    iter_engagement_corr, learned_max,
                )
            else:
                # Normal: set ceiling at median + headroom, bounded by default.
                learned_max = min(
                    _DEFAULT_MAX_ITERATIONS,
                    max(2, int(median_iter * 1.5) + 1),
                )

        return {
            "dimension_weights": weights,
            "pass_threshold": round(threshold, 2),
            "min_improvement": min_imp,
            "max_iterations": learned_max,
            "observation_count": len(observations),
            "emergent_min_obs": emergent_min,
            "dimension_cache_ttl_days": cache_ttl,
        }


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class DimensionScore:
    name: str
    score: int  # 1-5
    feedback: str  # Specific, actionable feedback


@dataclass
class CritiqueResult:
    dimensions: list[DimensionScore]
    composite_score: float
    revision_instructions: str  # Specific instructions for the reviser
    pass_threshold_met: bool
    weights_applied: dict | None = None   # dim_name → weight, or None if default
    adaptive_tier: str = "default"        # "client" | "aggregate" | "default"
    dimension_set: str = "fixed_v1"       # "fixed_v1" | "emergent_v{N}" | "freeform"
    # Phase 2: free-form quality signal
    analysis_text: str = ""               # free-form quality analysis (when freeform mode)
    analysis_embedding: list | None = None  # 384-dim embedding of analysis (for projection)
    confidence: float = 0.0               # 0.0-1.0 critic confidence
    predicted_reward: float | None = None  # from linear projection (None if not available)

    @property
    def weak_dimensions(self) -> list[DimensionScore]:
        return [d for d in self.dimensions if d.score <= 3]


@dataclass
class Iteration:
    iteration_num: int
    draft_text: str
    critique: CritiqueResult
    revised_text: str
    timestamp: str


@dataclass
class RefineResult:
    final_text: str
    best_score: float
    iterations: list[dict]
    total_iterations: int
    method: str  # "refined" | "passed_first" | "no_improvement"
    adaptive_tier: str = "default"  # "client" | "aggregate" | "default"
    dimension_set: str = "fixed_v1"  # "fixed_v1" | "emergent_v{N}" | "freeform"
    # Phase 2: continuous quality signal
    analysis_text: str = ""                # final critique's free-form analysis
    analysis_embedding: list | None = None  # 384-dim quality vector
    confidence: float = 0.0
    predicted_reward: float | None = None


# ------------------------------------------------------------------
# Critic
# ------------------------------------------------------------------

# Emergent dimension thresholds — adaptive overrides computed in CyreneAdaptiveConfig
_DEFAULT_EMERGENT_MIN_OBS = 15
_ADAPTIVE_WEIGHTS_MIN_OBS = _MIN_OBS_FOR_ADAPTIVE  # 10 — for learned weights on fixed dims
_DEFAULT_DIMENSION_CACHE_TTL_DAYS = 14
_DIMENSION_DRIFT_THRESHOLD = 0.5  # if >50% of dims changed, log warning

_dimension_version_counter: dict[str, int] = {}  # company → version, in-memory only


def _discover_dimensions(company: str, n_dims: int = 7) -> list[dict] | None:
    """Discover emergent scoring dimensions from client's own post history.

    Compares top-quartile vs bottom-quartile posts to find what actually
    explains the performance difference for this specific client.

    Returns list of dimension dicts with name/description/score_5/score_3/score_1,
    or None if insufficient data.
    """
    from backend.src.db import vortex as P

    # Check cache first
    cache_path = P.memory_dir(company) / "cyrene_dimensions_cache.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_at = cached.get("_cached_at", "")
            if cached_at:
                from datetime import datetime, timezone, timedelta
                dt = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
                if age_days < _DEFAULT_DIMENSION_CACHE_TTL_DAYS:
                    return cached.get("dimensions")
        except Exception:
            pass

    # Load observations
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        scored = [
            o for o in rm._state.get("observations", [])
            if o.get("status") in ("scored", "finalized")
            and o.get("reward", {}).get("immediate") is not None
            and (o.get("posted_body") or o.get("post_body"))
        ]
    except Exception:
        return None

    if len(scored) < _DEFAULT_EMERGENT_MIN_OBS:
        return None

    # Split into top/bottom quartiles
    scored.sort(key=lambda o: o.get("reward", {}).get("immediate", 0))
    n = len(scored)
    bottom = scored[:max(1, n // 4)]
    top = scored[max(0, n - n // 4):]

    def _format_posts(posts: list[dict], label: str) -> str:
        lines = []
        for i, o in enumerate(posts[-6:], 1):  # cap at 6 per group
            body = (o.get("posted_body") or o.get("post_body") or "")[:400]
            reward = o.get("reward", {}).get("immediate", 0)
            analysis = o.get("descriptor", {}).get("analysis", "")[:150]
            lines.append(f"[{label} {i} | score={reward:.2f}] {body}")
            if analysis:
                lines.append(f"  Analysis: {analysis}")
        return "\n\n".join(lines)

    top_text = _format_posts(top, "TOP")
    bottom_text = _format_posts(bottom, "LOW")

    prompt = (
        f"You are analyzing LinkedIn post performance for a specific author.\n\n"
        f"Here are their TOP-PERFORMING posts (highest engagement):\n{top_text}\n\n"
        f"Here are their LOWEST-PERFORMING posts:\n{bottom_text}\n\n"
        f"Identify exactly {n_dims} dimensions that EXPLAIN the difference between "
        f"high and low performers.\n\n"
        "Rules:\n"
        "- Each dimension must be specific and measurable (not 'quality' or 'engagement')\n"
        "- Dimensions should be things the WRITER can control (not audience size or timing)\n"
        "- Include dimensions the default rubric would miss\n"
        "- Name each dimension in 2-4 words\n\n"
        "Return JSON:\n"
        "[\n"
        '  {\n'
        '    "name": "Concrete Number Lead",\n'
        '    "description": "Post opens with a specific number, stat, or quantified claim",\n'
        '    "score_5": "First sentence contains a precise, surprising number tied to the core insight",\n'
        '    "score_3": "Numbers appear but buried in body, or rounded/vague",\n'
        '    "score_1": "No quantification anywhere, entirely conceptual"\n'
        "  },\n"
        "  ...\n"
        "]\n"
        "Output ONLY the JSON array."
    )

    try:
        resp = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        dims = json.loads(raw)
        if not isinstance(dims, list) or len(dims) < 3:
            return None

        # Validate structure
        valid = []
        for d in dims[:n_dims]:
            if d.get("name") and d.get("score_5") and d.get("score_1"):
                valid.append({
                    "name": d["name"],
                    "description": d.get("description", d["name"]),
                    "score_5": d["score_5"],
                    "score_3": d.get("score_3", "Moderate presence"),
                    "score_1": d["score_1"],
                })
        if len(valid) < 3:
            return None

    except Exception as e:
        logger.warning("[Cyrene] Dimension discovery failed for %s: %s", company, e)
        return None

    # Drift detection: compare with previous cached dimensions
    _check_dimension_drift(company, valid, cache_path)

    # Cache
    from datetime import datetime, timezone
    cache_data = {
        "dimensions": valid,
        "_cached_at": datetime.now(timezone.utc).isoformat(),
        "_observation_count": len(scored),
        "_version": _dimension_version_counter.get(company, 0) + 1,
    }
    _dimension_version_counter[company] = cache_data["_version"]
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(cache_path)
    except Exception:
        pass

    logger.info(
        "[Cyrene] Discovered %d emergent dimensions for %s (v%d, from %d obs)",
        len(valid), company, cache_data["_version"], len(scored),
    )
    return valid


def _check_dimension_drift(company: str, new_dims: list[dict], cache_path) -> None:
    """Log warning if >50% of dimensions changed since last discovery."""
    if not cache_path.exists():
        return
    try:
        old = json.loads(cache_path.read_text(encoding="utf-8")).get("dimensions", [])
        old_names = {d["name"].lower() for d in old}
        new_names = {d["name"].lower() for d in new_dims}
        if not old_names:
            return
        overlap = len(old_names & new_names)
        total = max(len(old_names), len(new_names))
        change_rate = 1.0 - (overlap / total)
        if change_rate > _DIMENSION_DRIFT_THRESHOLD:
            logger.warning(
                "[Cyrene] Dimension drift for %s: %.0f%% changed (%d→%d). "
                "Old: %s. New: %s",
                company, change_rate * 100, len(old_names), len(new_names),
                sorted(old_names - new_names), sorted(new_names - old_names),
            )
    except Exception:
        pass


def _get_dimension_set_label(company: str, custom_dims: list[dict] | None) -> str:
    """Return a label for the dimension set used, for observation auditability."""
    if custom_dims is None:
        return "fixed_v1"
    version = _dimension_version_counter.get(company, 0)
    return f"emergent_v{version}"


_DIMENSIONS = [
    ("360Brew Alignment", "profile-content consistency",
     "5: Squarely within the author's established topic pillars\n   3: Tangentially related but not core\n   1: Completely off-topic for this author"),
    ("Hook Scroll-Stop", "first 140 characters",
     "5: Specific, curiosity-provoking, hints at the OMI, impossible to scroll past\n   3: Decent but generic or predictable\n   1: Vague, cliché, or buried lede"),
    ("Save-Worthiness", "would a reader hit 'Save'?",
     "5: Contains a framework, specific insight, or reference-worthy takeaway\n   3: Interesting but not bookmark-worthy\n   1: Generic advice anyone could write"),
    ("Comment Invitation", "does the post invite response?",
     "5: Leaves an explicit open loop, asks a genuine question, or takes a stance people will debate\n   3: Makes a point but closes the argument\n   1: Monologue that discourages engagement"),
    ("Dwell Time", "structure and density",
     "5: Optimal length (800-1500 chars), well-structured paragraphs, information-dense\n   3: Slightly too long/short, some padding\n   1: Wall of text, or too thin to justify the read"),
    ("Magic Moment", "would the client say 'you told my story better than I could've'?",
     "5: Deeply personal, grounded in specific transcript moments, authentically voiced\n   3: Plausibly the client but could be anyone in their industry\n   1: Generic thought leadership that screams AI-written"),
    ("ICP Resonance", "would the target persona find this novel/actionable?",
     "5: Directly addresses ICP pain points with specific, implementable insight\n   3: Broadly relevant but not targeted\n   1: Off-target audience entirely"),
]


# -----------------------------------------------------------------------
# Phase 2: Free-form quality signal (continuous, no dimensions)
# -----------------------------------------------------------------------

def _build_freeform_critic_system(
    high_examples: list[str],
    low_examples: list[str],
) -> str:
    """Build a freeform critic system prompt using example-based evaluation."""
    lines = [
        "You are evaluating a LinkedIn post for a B2B ghostwriting agency.\n",
        "Write a detailed, free-form quality analysis. Cover whatever aspects "
        "matter most for THIS specific post. Do not follow a predefined rubric. "
        "Instead, identify:\n",
        "1. What makes this post work (or not work)",
        "2. What specific changes would improve it most",
        "3. Your overall confidence that this post will perform well (0.0 to 1.0)\n",
    ]

    if high_examples:
        lines.append("Quality analyses of this author's BEST-PERFORMING posts:")
        for i, ex in enumerate(high_examples[:3], 1):
            lines.append(f"  [{i}] {ex[:300]}")
        lines.append("")

    if low_examples:
        lines.append("Quality analyses of this author's WORST-PERFORMING posts:")
        for i, ex in enumerate(low_examples[:3], 1):
            lines.append(f"  [{i}] {ex[:300]}")
        lines.append("")

    lines.append(
        "Your analysis should identify what makes THIS post more like the "
        "best-performers or the worst-performers above.\n"
    )

    lines.append(
        "Be specific. Reference exact phrases, structural choices, and audience impact.\n\n"
        "Respond as JSON:\n"
        "{\n"
        '  "analysis": "your detailed free-form quality analysis...",\n'
        '  "confidence": 0.73,\n'
        '  "revision_instructions": "1. ... 2. ... 3. ...",\n'
        '  "pass": true\n'
        "}"
    )

    return "\n".join(lines)


def _get_quality_examples(company: str) -> tuple[list[str], list[str]]:
    """Get quality analysis examples from top and bottom performers."""
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        scored = [
            o for o in rm._state.get("observations", [])
            if o.get("status") in ("scored", "finalized")
            and o.get("descriptor", {}).get("analysis")
            and o.get("reward", {}).get("immediate") is not None
        ]
    except Exception:
        return [], []

    if len(scored) < 6:
        return [], []

    scored.sort(key=lambda o: o.get("reward", {}).get("immediate", 0))
    n = len(scored)
    bottom = scored[:max(1, n // 4)]
    top = scored[max(0, n - n // 4):]

    high = [o.get("descriptor", {}).get("analysis", "") for o in top if o.get("descriptor", {}).get("analysis")]
    low = [o.get("descriptor", {}).get("analysis", "") for o in bottom if o.get("descriptor", {}).get("analysis")]

    return high[-3:], low[-3:]


class LinearProjection:
    """Simple linear reward predictor from quality embeddings.

    Trained on (quality_embedding, engagement_reward) pairs.
    Uses ridge regression (numpy-only, no sklearn dependency).
    """

    def __init__(self, company: str):
        self.company = company
        self._W: np.ndarray | None = None
        self._b: float = 0.0
        self._trained = False

    def train(self, embeddings: list[list[float]], rewards: list[float], ridge_alpha: float = 1.0) -> bool:
        """Fit W, b from data. Returns True if training succeeded."""
        if len(embeddings) < _MIN_OBS_FOR_PROJECTION or len(embeddings) != len(rewards):
            return False

        X = np.array(embeddings, dtype=np.float32)
        y = np.array(rewards, dtype=np.float32)

        # Ridge regression: W = (X^T X + alpha I)^-1 X^T y
        n, d = X.shape
        XtX = X.T @ X + ridge_alpha * np.eye(d, dtype=np.float32)
        Xty = X.T @ y

        try:
            self._W = np.linalg.solve(XtX, Xty)
            self._b = float(y.mean() - X.mean(axis=0) @ self._W)
            self._trained = True
            return True
        except np.linalg.LinAlgError:
            return False

    def predict(self, embedding: list[float]) -> float:
        """Predict reward from a quality embedding."""
        if not self._trained or self._W is None:
            return 0.0
        x = np.array(embedding, dtype=np.float32)
        return float(x @ self._W + self._b)

    def train_from_observations(self, company: str) -> bool:
        """Train from RuanMei observations that have quality embeddings stored.

        After the embedding model migration (sentence-transformers → OpenAI),
        old 384-dim and new 1536-dim vectors may coexist in quality_embeddings.json.
        We filter to the most recent dimension (last pair's dim) to avoid mixing
        incompatible embedding spaces. Old pairs are effectively discarded for
        training purposes — they'll be re-embedded when ordinal_sync runs again.
        """
        from backend.src.db import vortex as P
        cache_path = P.memory_dir(company) / "quality_embeddings.json"
        if not cache_path.exists():
            return False
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            pairs = data.get("pairs", [])
            if not pairs:
                return False
            # Determine target dimension from the most recent embedding
            target_dim = len(pairs[-1].get("embedding", []))
            pairs = [p for p in pairs if len(p.get("embedding", [])) == target_dim]
            if len(pairs) < _MIN_OBS_FOR_PROJECTION:
                return False
            embeddings = [p["embedding"] for p in pairs]
            rewards = [p["reward"] for p in pairs]
            return self.train(embeddings, rewards)
        except Exception:
            return False


def _persist_quality_embedding(company: str, embedding: list[float], reward: float, post_hash: str) -> None:
    """Append a quality embedding + reward pair for linear projection training."""
    from backend.src.db import vortex as P
    cache_path = P.memory_dir(company) / "quality_embeddings.json"
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"pairs": []}
    except Exception:
        data = {"pairs": []}

    # Deduplicate by post_hash
    existing_hashes = {p.get("post_hash") for p in data["pairs"]}
    if post_hash in existing_hashes:
        return

    data["pairs"].append({
        "embedding": embedding,
        "reward": reward,
        "post_hash": post_hash,
    })

    # Cap at 200 pairs
    if len(data["pairs"]) > 200:
        data["pairs"] = data["pairs"][-200:]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(cache_path)


def _get_learned_blend_weight(company: str) -> float:
    """Learn the confidence vs predicted_reward blend weight from historical data.

    Computes Spearman correlation of each signal with actual engagement.
    The signal that better predicts engagement gets proportionally more weight.
    Falls back to 0.7 (confidence-dominant) when insufficient data.
    """
    from backend.src.db import vortex as P
    cache_path = P.memory_dir(company) / "quality_embeddings.json"
    if not cache_path.exists():
        return 0.7  # default: confidence dominates

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        pairs = data.get("pairs", [])
        if len(pairs) < _MIN_OBS_FOR_PROJECTION:
            return 0.7

        # We need observations that have BOTH a confidence score and a predicted reward
        # stored alongside their actual engagement. For now, use the correlation between
        # the embedding-predicted reward and actual reward as the projection's reliability.
        # If projection is reliable (high corr), weight it more. If not, weight confidence.
        rewards = [p.get("reward", 0) for p in pairs]
        embeddings = [p.get("embedding") for p in pairs]

        if not embeddings or not all(embeddings):
            return 0.7

        # Train projection, compute prediction accuracy
        proj = LinearProjection(company)
        if not proj.train(embeddings, rewards):
            return 0.7

        # Compute correlation between predicted and actual
        predicted = [proj.predict(e) for e in embeddings]
        from backend.src.utils.correlation_analyzer import _spearman_correlation
        corr = _spearman_correlation(predicted, rewards)

        # Blend weight = continuous function of prediction correlation.
        # corr=1 → confidence_weight=0.3 (prediction reliable, lean on it)
        # corr=0 → confidence_weight=0.9 (prediction useless, lean on confidence)
        # Linear interpolation, no hard clamp.
        confidence_weight = 0.9 - 0.6 * max(corr, 0)
        return round(confidence_weight, 3)
    except Exception:
        return 0.7


def _use_freeform(company: str) -> bool:
    """Check if client has enough data for free-form critic."""
    try:
        from backend.src.agents.ruan_mei import RuanMei
        rm = RuanMei(company)
        scored_with_analysis = sum(
            1 for o in rm._state.get("observations", [])
            if o.get("status") in ("scored", "finalized") and o.get("descriptor", {}).get("analysis")
        )
        return scored_with_analysis >= _MIN_OBS_FOR_FREEFORM
    except Exception:
        return False


def _build_critic_system(
    dim_weights: dict[str, float],
    is_adaptive: bool,
    custom_dimensions: list[dict] | None = None,
) -> str:
    """Build the critic system prompt.

    Three modes:
    - custom_dimensions provided → emergent dimensions (client-specific rubric)
    - is_adaptive=True → fixed dimensions with learned weights
    - is_adaptive=False → fixed dimensions, equal weight (cold start)
    """
    use_custom = custom_dimensions and len(custom_dimensions) >= 3

    n_dims = len(custom_dimensions) if use_custom else len(_DIMENSIONS)
    lines = [
        "You are a ruthless LinkedIn post quality critic for a B2B ghostwriting agency.\n",
        f"Score the draft along these {n_dims} dimensions (1-5 each):\n",
    ]

    if use_custom:
        # Emergent dimensions: build rubric from discovered dimension data
        for i, dim in enumerate(custom_dimensions, 1):
            name = dim["name"]
            desc = dim.get("description", name)
            weight = dim_weights.get(name)
            pct = f" — {weight:.0%} importance" if is_adaptive and weight is not None else ""
            lines.append(f"{i}. **{name}** ({desc}{pct})")
            lines.append(f"   5: {dim.get('score_5', 'Excellent')}")
            lines.append(f"   3: {dim.get('score_3', 'Moderate')}")
            lines.append(f"   1: {dim.get('score_1', 'Poor')}\n")
    else:
        # Fixed dimensions (cold start or adaptive weights)
        for i, (name, subtitle, rubric) in enumerate(_DIMENSIONS, 1):
            weight = dim_weights.get(name)
            pct = f" — {weight:.0%} importance" if is_adaptive and weight is not None else ""
            lines.append(f"{i}. **{name}** ({subtitle}{pct})")
            lines.append(f"   {rubric}\n")

    if is_adaptive:
        lines.append(
            "Dimensions with higher importance weight should receive more scrutiny. "
            "A score of 2 on a high-importance dimension is much worse than a 2 on a "
            "low-importance dimension. Factor this into your revision_instructions — "
            "prioritize fixing high-weight dimensions first.\n"
        )

    lines.append(
        "For each dimension scoring ≤3, provide SPECIFIC revision instructions "
        "(not vague 'make it better').\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\n"
        '  "dimensions": [\n'
        '    {"name": "DimensionName", "score": 4, "feedback": "..."},\n'
        f"    ...all {n_dims}...\n"
        "  ],\n"
        '  "composite_score": 3.4,\n'
        '  "revision_instructions": "1. Fix X. 2. Improve Y.",\n'
        '  "pass_threshold_met": false\n'
        "}"
    )

    return "\n".join(lines)


def _build_critic_prompt(
    draft_text: str,
    company: str,
    alignment_score: dict | None = None,
    icp_definition: dict | None = None,
    transcript_excerpt: str = "",
    style_rules: list[dict] | None = None,
    iteration_num: int = 1,
) -> str:
    """Build the critic's user message with all available context."""
    parts = [f"DRAFT POST (iteration {iteration_num}):\n{draft_text}"]

    if alignment_score and alignment_score.get("score") is not None:
        parts.append(
            f"\nALIGNMENT WITH CLIENT FINGERPRINT:\n"
            f"Cosine similarity: {alignment_score.get('score')}/1.0\n"
            f"Method: {alignment_score.get('method', 'unknown')}\n"
            "This is a raw cosine similarity between the draft's embedding "
            "and the client's identity fingerprint (accepted posts, profile, "
            "ICP). No threshold, no label — decide what this number means "
            "for the draft based on the context above."
        )

    if icp_definition:
        desc = icp_definition.get("description", "")
        anti = icp_definition.get("anti_description", "")
        if desc:
            parts.append(f"\nICP DEFINITION:\n{desc}")
        if anti:
            parts.append(f"\nANTI-ICP (people we do NOT want engaging):\n{anti}")

    if transcript_excerpt:
        parts.append(f"\nSOURCE TRANSCRIPT EXCERPT:\n{transcript_excerpt[:2000]}")

    if style_rules:
        hard = [r for r in style_rules if r.get("tier") == "hard"]
        if hard:
            rules_text = "\n".join(f"- {r['description']}" for r in hard[:10])
            parts.append(f"\nCLIENT-SPECIFIC STYLE RULES (hard):\n{rules_text}")

    # Previously: built an "engagement diagnostic" by bucketing the client's
    # history into top/bottom performers and feeding the bucketed comparison
    # to the critic. That's prescriptive pattern extraction from a small
    # dataset — exactly the kind of hand-authored taxonomy the Bitter Lesson
    # filter rejects. Severed. The critic reads raw ICP, alignment, and
    # style-rule context; if it needs historical performance data it can be
    # given tool access to pull_history in a future refactor.

    return "\n\n---\n\n".join(parts)


def critique(
    draft_text: str,
    company: str,
    alignment_score: dict | None = None,
    icp_definition: dict | None = None,
    transcript_excerpt: str = "",
    style_rules: list[dict] | None = None,
    iteration_num: int = 1,
) -> CritiqueResult:
    """Run the critic on a single draft. Returns structured scores + revision instructions."""

    # Get alignment score if not provided
    if alignment_score is None:
        try:
            from backend.src.utils.alignment_scorer import score_draft_alignment
            alignment_score = score_draft_alignment(company, draft_text)
        except Exception:
            alignment_score = None

    # Load ICP if not provided
    if icp_definition is None:
        try:
            from backend.src.db import vortex as P
            icp_path = P.icp_definition_path(company)
            if icp_path.exists():
                icp_definition = json.loads(icp_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Style rules removed: feedback learning happens through RuanMei
    # observations (draft → final → engagement), not categorical rules.
    style_rules = None

    # Resolve adaptive config (weights + threshold)
    adaptive = CyreneAdaptiveConfig().resolve(company)
    dim_weights = adaptive.get("dimension_weights", _DEFAULT_DIMENSION_WEIGHTS)
    pass_threshold = adaptive.get("pass_threshold", _DEFAULT_PASS_THRESHOLD)
    is_adaptive = adaptive.get("_tier") != "default"

    # Route: freeform (continuous) vs dimension-based (categorical)
    use_freeform = _use_freeform(company)

    if use_freeform:
        return _critique_freeform(
            draft_text, company, adaptive, alignment_score,
            icp_definition, transcript_excerpt, style_rules, iteration_num,
        )

    # Dimension-based path (cold-start / insufficient data)
    # Try emergent dimensions when adaptive config has enough data OR
    # when the client has enough scored observations (even without prior
    # Cyrene dimension scores — breaks the chicken-and-egg bootstrap).
    custom_dims = None
    obs_count = adaptive.get("observation_count", 0)
    emergent_min = adaptive.get("emergent_min_obs", _DEFAULT_EMERGENT_MIN_OBS)
    if (is_adaptive and obs_count >= emergent_min) or obs_count == 0:
        # obs_count==0 means no cyrene_dimensions yet; _discover_dimensions
        # checks its own threshold against scored observations with post bodies.
        custom_dims = _discover_dimensions(company)

    user_msg = _build_critic_prompt(
        draft_text, company, alignment_score, icp_definition,
        transcript_excerpt, style_rules, iteration_num,
    )

    system = _build_critic_system(dim_weights, is_adaptive, custom_dimensions=custom_dims)

    try:
        resp = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)

        dimensions = [
            DimensionScore(
                name=d.get("name", f"dim_{i}"),
                score=max(1, min(5, int(d.get("score", 3)))),
                feedback=d.get("feedback", ""),
            )
            for i, d in enumerate(data.get("dimensions", []))
        ]

        if dimensions:
            weighted_sum = sum(d.score * dim_weights.get(d.name, 1 / 7) for d in dimensions)
            weight_total = sum(dim_weights.get(d.name, 1 / 7) for d in dimensions)
            composite = weighted_sum / weight_total if weight_total > 0 else 3.0
        else:
            composite = float(data.get("composite_score", 3.0))

        return CritiqueResult(
            dimensions=dimensions,
            composite_score=round(composite, 2),
            revision_instructions=data.get("revision_instructions", ""),
            pass_threshold_met=composite >= pass_threshold,
            weights_applied=dim_weights if is_adaptive else None,
            adaptive_tier=adaptive.get("_tier", "default"),
            dimension_set=_get_dimension_set_label(company, custom_dims),
        )

    except json.JSONDecodeError:
        logger.warning("[Cyrene] Critic returned non-JSON")
        return _default_critique(draft_text)
    except Exception as e:
        logger.warning("[Cyrene] Critique failed: %s", e)
        return _default_critique(draft_text)


def _critique_freeform(
    draft_text: str,
    company: str,
    adaptive: dict,
    alignment_score: dict | None,
    icp_definition: dict | None,
    transcript_excerpt: str,
    style_rules: list[dict] | None,
    iteration_num: int,
) -> CritiqueResult:
    """Free-form quality critique — no predefined dimensions.

    The critic writes open-ended analysis, which gets embedded into a
    continuous quality vector. A linear projection predicts engagement
    from the quality vector when enough training data exists.
    """
    pass_threshold = adaptive.get("pass_threshold", _DEFAULT_PASS_THRESHOLD)

    # Get example analyses from top/bottom performers
    high_ex, low_ex = _get_quality_examples(company)

    system = _build_freeform_critic_system(high_ex, low_ex)

    user_msg = _build_critic_prompt(
        draft_text, company, alignment_score, icp_definition,
        transcript_excerpt, style_rules, iteration_num,
    )

    try:
        resp = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)

        analysis = data.get("analysis", "")
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        revision_instructions = data.get("revision_instructions", "")
        # No hand-tuned fallback cutoff. If the critic didn't emit an
        # explicit `pass` decision, default to `True` (accept and let the
        # next iteration's revision_instructions drive behavior). The
        # previous `confidence >= 0.6` fallback was an arbitrary gate.
        critic_pass = bool(data.get("pass", True))

        # Embed the analysis for continuous quality vector
        analysis_embedding = None
        predicted_reward = None
        try:
            from backend.src.utils.post_embeddings import embed_texts
            embs = embed_texts([analysis])
            if embs:
                analysis_embedding = embs[0]

                # Try linear projection if enough training data
                proj = LinearProjection(company)
                if proj.train_from_observations(company):
                    predicted_reward = proj.predict(analysis_embedding)
        except Exception:
            pass

        # Composite score: native 0-1 space.
        # Blend weight between confidence and predicted_reward is LEARNED
        # from historical correlation with actual engagement.
        if predicted_reward is not None:
            pred_01 = max(0.0, min(1.0, 0.5 + predicted_reward * 0.25))
            blend_w = _get_learned_blend_weight(company)
            composite_01 = blend_w * confidence + (1.0 - blend_w) * pred_01
        else:
            composite_01 = confidence

        # Map to 1-5 for backward-compatible output only (NOT used for pass decision)
        composite_15 = composite_01 * 5.0

        # Pass decision in native 0-1 space.
        # Threshold: pass_threshold is in 1-5 scale from adaptive config.
        # Convert to 0-1: 3.5/5 = 0.7 (default)
        threshold_01 = pass_threshold / 5.0
        pass_met = critic_pass and composite_01 >= threshold_01

        # Create synthetic dimension for backward compat
        dim = DimensionScore(
            name="freeform_quality",
            score=max(1, min(5, round(composite_15))),
            feedback=analysis[:200] if analysis else "",
        )

        return CritiqueResult(
            dimensions=[dim],
            composite_score=round(composite_15, 2),  # 1-5 for backward compat display
            revision_instructions=revision_instructions,
            pass_threshold_met=pass_met,
            weights_applied=None,
            adaptive_tier=adaptive.get("_tier", "default"),
            dimension_set="freeform",
            analysis_text=analysis,
            analysis_embedding=analysis_embedding,
            confidence=confidence,
            predicted_reward=round(predicted_reward, 4) if predicted_reward is not None else None,
        )

    except json.JSONDecodeError:
        logger.warning("[Cyrene] Freeform critic returned non-JSON")
        return _default_critique(draft_text)
    except Exception as e:
        logger.warning("[Cyrene] Freeform critique failed: %s", e)
        return _default_critique(draft_text)


def _default_critique(draft_text: str) -> CritiqueResult:
    """Fallback critique that passes everything (don't block generation)."""
    return CritiqueResult(
        dimensions=[DimensionScore(name="fallback", score=4, feedback="Critique unavailable")],
        composite_score=4.0,
        revision_instructions="",
        pass_threshold_met=True,
    )


# ------------------------------------------------------------------
# Reviser
# ------------------------------------------------------------------

_REVISER_SYSTEM = """\
You are revising a LinkedIn post based on specific critic feedback. \
Apply ONLY the requested changes. Do not rewrite sections that weren't flagged. \
Preserve the author's voice, specific details, and transcript-sourced material.

Return ONLY the revised post text. No JSON, no markdown fences, no explanation.
"""


def revise(
    draft_text: str,
    critique_result: CritiqueResult,
    company: str = "",
) -> str:
    """Revise a draft based on critic feedback. Returns revised text."""

    if not critique_result.revision_instructions:
        return draft_text

    user_msg = (
        f"ORIGINAL DRAFT:\n{draft_text}\n\n"
        f"CRITIC SCORES:\n"
    )
    for d in critique_result.dimensions:
        user_msg += f"  {d.name}: {d.score}/5"
        if d.score <= 3:
            user_msg += f" — {d.feedback}"
        user_msg += "\n"

    user_msg += f"\nREVISION INSTRUCTIONS:\n{critique_result.revision_instructions}\n\n"
    user_msg += (
        "Apply ONLY these specific changes. Do not alter parts that scored well. "
        "Return the complete revised post text."
    )

    try:
        resp = _client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=_REVISER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        revised = resp.content[0].text.strip()

        # Basic sanity: revised should be reasonable length
        if len(revised) < 200:
            logger.warning("[Cyrene] Revision too short (%d chars), keeping original", len(revised))
            return draft_text
        if len(revised) > 3200:
            logger.warning("[Cyrene] Revision too long (%d chars), keeping original", len(revised))
            return draft_text

        return revised

    except Exception as e:
        logger.warning("[Cyrene] Revision failed: %s", e)
        return draft_text


# ------------------------------------------------------------------
# SELF-REFINE Loop
# ------------------------------------------------------------------

def refine_post(
    company: str,
    draft_text: str,
    transcript_excerpt: str = "",
    max_iterations: Optional[int] = None,
    event_callback: Callable | None = None,
) -> RefineResult:
    """Run the full SELF-REFINE loop on a single post.

    Returns the best version (by composite score), not necessarily the latest.

    The iteration count is NOT hard-coded. The loop continues until:
      1. The critic says the post passes (composite >= learned pass_threshold), OR
      2. Improvement between iterations is below the learned min_improvement, OR
      3. The safety ceiling is reached (learned per-client, default 8).

    The critic decides when to stop. The ceiling is a safety net, not a target.
    Same philosophy as Stelle's "write as many posts as the transcripts support"
    — the LLM evaluates quality, the system just caps the worst case.

    Args:
        company: Client company keyword.
        draft_text: The initial draft from Stelle.
        transcript_excerpt: Source material for Magic Moment evaluation.
        max_iterations: Override for the learned ceiling. None = use learned value.
        event_callback: Optional callback for progress events.
    """
    iterations: list[Iteration] = []
    best_text = draft_text
    best_score = 0.0
    current_text = draft_text

    # Pre-compute alignment score (reused across iterations)
    alignment_score = None
    try:
        from backend.src.utils.alignment_scorer import score_draft_alignment
        alignment_score = score_draft_alignment(company, draft_text)
    except Exception:
        pass

    # Resolve adaptive thresholds — ALL three are learned from client data:
    #   pass_threshold:   what composite score is "good enough to ship"
    #   min_improvement:  when to stop iterating (diminishing returns)
    #   max_iterations:   safety ceiling (overpolishing detection lowers it)
    adaptive = CyreneAdaptiveConfig().resolve(company)
    min_improvement = adaptive.get("min_improvement", _DEFAULT_MIN_IMPROVEMENT)
    if max_iterations is None:
        max_iterations = adaptive.get("max_iterations", _DEFAULT_MAX_ITERATIONS)

    for i in range(1, max_iterations + 1):
        if event_callback:
            event_callback("status", {"message": f"[Cyrene] Critique iteration {i}/{max_iterations}..."})

        logger.info("[Cyrene] Iteration %d/%d for %s", i, max_iterations, company)

        # Critique
        crit = critique(
            current_text,
            company,
            alignment_score=alignment_score,
            transcript_excerpt=transcript_excerpt,
            iteration_num=i,
        )

        # Track best
        if crit.composite_score > best_score:
            best_score = crit.composite_score
            best_text = current_text

        # Check if we can stop
        if crit.pass_threshold_met:
            logger.info(
                "[Cyrene] Passed threshold at iteration %d (score %.2f)",
                i, crit.composite_score,
            )
            iterations.append(Iteration(
                iteration_num=i,
                draft_text=current_text,
                critique=crit,
                revised_text=current_text,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ))
            break

        # Check for diminishing returns
        if i > 1 and iterations:
            prev_score = iterations[-1].critique.composite_score
            improvement = crit.composite_score - prev_score
            if improvement < min_improvement and crit.composite_score > prev_score:
                logger.info(
                    "[Cyrene] Diminishing returns at iteration %d (improvement %.2f)",
                    i, improvement,
                )
                iterations.append(Iteration(
                    iteration_num=i,
                    draft_text=current_text,
                    critique=crit,
                    revised_text=current_text,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ))
                break

        # Revise
        if event_callback:
            weak = [d.name for d in crit.weak_dimensions]
            event_callback("status", {
                "message": f"[Cyrene] Revising: {', '.join(weak) if weak else 'minor polish'}..."
            })

        revised = revise(current_text, crit, company)

        iterations.append(Iteration(
            iteration_num=i,
            draft_text=current_text,
            critique=crit,
            revised_text=revised,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ))

        # Update alignment score for revised text
        if revised != current_text:
            try:
                alignment_score = score_draft_alignment(company, revised)
            except Exception:
                pass

        # Track best after revision
        # We'll score the revision in the next iteration's critique
        current_text = revised

    # Final check: if the last revision was never critiqued, score it
    if current_text != best_text:
        final_crit = critique(current_text, company, alignment_score=alignment_score)
        if final_crit.composite_score > best_score:
            best_score = final_crit.composite_score
            best_text = current_text

    method = "passed_first" if len(iterations) == 1 and iterations[0].critique.pass_threshold_met else "refined"
    if best_text == draft_text and len(iterations) > 1:
        method = "no_improvement"

    if event_callback:
        event_callback("status", {
            "message": f"[Cyrene] Complete: {len(iterations)} iterations, best score {best_score:.2f}/5.0"
        })

    # Get adaptive tier and dimension set from the last critique
    adaptive_tier = "default"
    dimension_set = "fixed_v1"
    final_analysis = ""
    final_embedding = None
    final_confidence = 0.0
    final_predicted = None
    if iterations:
        last_crit = iterations[-1].critique
        adaptive_tier = last_crit.adaptive_tier
        dimension_set = last_crit.dimension_set
        final_analysis = last_crit.analysis_text
        final_embedding = last_crit.analysis_embedding
        final_confidence = last_crit.confidence
        final_predicted = last_crit.predicted_reward

    return RefineResult(
        final_text=best_text,
        best_score=round(best_score, 2),
        iterations=[
            {
                "iteration": it.iteration_num,
                "composite_score": it.critique.composite_score,
                "all_dimensions": {
                    d.name: d.score
                    for d in it.critique.dimensions
                },
                "weak_dimensions": [
                    {"name": d.name, "score": d.score, "feedback": d.feedback}
                    for d in it.critique.weak_dimensions
                ],
                "revision_instructions": it.critique.revision_instructions,
                "text_changed": it.draft_text != it.revised_text,
                "adaptive_tier": it.critique.adaptive_tier,
                "dimension_set": it.critique.dimension_set,
            }
            for it in iterations
        ],
        total_iterations=len(iterations),
        method=method,
        adaptive_tier=adaptive_tier,
        dimension_set=dimension_set,
        analysis_text=final_analysis,
        analysis_embedding=final_embedding,
        confidence=final_confidence,
        predicted_reward=final_predicted,
    )


def refine_batch(
    company: str,
    drafts: list[dict],
    max_iterations: Optional[int] = None,
    event_callback: Callable | None = None,
) -> list[RefineResult]:
    """Refine multiple posts. Each dict must have 'text' key, optionally 'transcript_excerpt'.

    Returns a list of RefineResult, one per draft.
    """
    results = []
    for i, draft in enumerate(drafts):
        if event_callback:
            event_callback("status", {"message": f"[Cyrene] Refining post {i+1}/{len(drafts)}..."})

        text = draft.get("text", "")
        excerpt = draft.get("transcript_excerpt", "")

        if not text.strip():
            results.append(RefineResult(
                final_text=text,
                best_score=0.0,
                iterations=[],
                total_iterations=0,
                method="skip",
            ))
            continue

        result = refine_post(
            company=company,
            draft_text=text,
            transcript_excerpt=excerpt,
            max_iterations=max_iterations,
            event_callback=event_callback,
        )
        results.append(result)

    return results


# ------------------------------------------------------------------
# Legacy stylistic rewrite (original Cyrene functionality)
# ------------------------------------------------------------------

class CyreneStyleRewriter:
    """Cyrene's original stylistic rewrite capability.

    Rewrites individual LinkedIn post drafts while preserving factual payload.
    Uses a 4-step XML framework: fact extraction → style approach → strategy → rewrite.
    """

    def __init__(self, model_name: str = "claude-opus-4-6"):
        self.model_name = model_name

    def rewrite_single_post(
        self,
        post_text: str,
        style_instruction: str,
        image_suggestion: str = "",
        theme: str = "",
        client_context: str = "",
    ) -> dict:
        """Rewrite a single post using Cyrene's 4-step XML framework."""
        system_prompt = (
            "You are Cyrene, a meticulous copyeditor. Your job is to completely rewrite a draft "
            "stylistically while maintaining 100% of the original factual payload and logical arguments."
        )
        if client_context:
            system_prompt = (
                "You have access to the following client context (interview transcripts, "
                "approved posts, feedback). Use this to ground your edits in what the client "
                "actually said and prefers.\n\n"
                + client_context + "\n\n" + system_prompt
            )

        if style_instruction and style_instruction.strip():
            style_directive = f"Apply the following stylistic direction strictly: '{style_instruction.strip()}'"
            analysis_directive = "Analyze the requested style direction and outline how you will apply it."
        else:
            style_directive = (
                "No specific style was provided. Introduce creative stylistic noise — "
                "randomize sentence lengths, swap vocabulary, restructure flow, and "
                "change emotional undertone slightly to make it feel organic."
            )
            analysis_directive = "Describe the random stylistic variations you are choosing to apply."

        import re as _re

        user_prompt = f"""
        <raw_draft>
        {post_text}
        </raw_draft>

        STYLE DIRECTIVE:
        {style_directive}

        INSTRUCTIONS:
        Output your response in the following exact XML format:

        <step_1_fact_extraction>
        List bullet points of every core argument, statistic, and narrative beat.
        </step_1_fact_extraction>

        <step_2_style_approach>
        {analysis_directive}
        </step_2_style_approach>

        <step_3_rewrite_strategy>
        How you will map facts onto the stylistic approach.
        </step_3_rewrite_strategy>

        <final_post>
        [Your rewritten post goes here]
        </final_post>
        """

        resp = _client.messages.create(
            model=self.model_name,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = resp.content[0].text

        def _parse_xml(text, tag):
            import re
            m = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
            if m:
                return m.group(1).strip()
            m2 = re.search(f"<{tag}>(.*)", text, re.DOTALL)
            return m2.group(1).strip() if m2 else ""

        return {
            "fact_extraction": _parse_xml(raw, "step_1_fact_extraction"),
            "style_analysis": _parse_xml(raw, "step_2_style_approach"),
            "strategy": _parse_xml(raw, "step_3_rewrite_strategy"),
            "final_post": _parse_xml(raw, "final_post"),
            "image_suggestion": image_suggestion.strip() if image_suggestion else "",
            "theme": theme.strip() if theme else "",
        }


# Backward-compatible alias
Cyrene = CyreneStyleRewriter
