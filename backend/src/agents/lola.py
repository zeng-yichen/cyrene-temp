"""LOLA — UCB1 Online Learning Agent for content strategy exploration.

██████████████████████████████████████████████████████████████████████
██  DEPRECATED: LOLA is superseded by RuanMei.recommend_context().  ██
██                                                                    ██
██  Content intelligence is now provided by a Claude-as-Analyst       ██
██  pipeline in ruan_mei.py (Phase 1) with KNN-based frontier         ██
██  detection and cross-client blind spot exploration.                 ██
██                                                                    ██
██  This file is kept for reference. No active codepath imports LOLA.  ██
██████████████████████████████████████████████████████████████████████

LOLA maintained a set of per-client content arms (topic + format combinations)
and selected which to explore or exploit next, using UCB1 bandit logic.
Reward came from RuanMei observations (ICP-weighted engagement score).

Design:
  - UCB1 for topic arms (stable enough for confidence-interval exploration)
  - Thompson Sampling for format arms (categorical, low sample size tolerant)
  - Adaptive exploration budget (decays from 20% as bandit converges)
  - Topic retirement after adaptive consecutive declining threshold
  - OpenAI text-embedding-3-small for semantic arm matching
  - Continuous kernel reward field (Phase 1) blends in after 10+ data points
  - Persisted to memory/{company}/lola_state.json
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Embedding helpers — OpenAI text-embedding-3-small (1536 dims)
# -----------------------------------------------------------------------
#
# Previously used sentence-transformers (all-MiniLM-L6-v2, 384 dims) loaded
# locally. Migrated to OpenAI because:
#   1. sentence-transformers pulls torch (~2GB, wants GPU) — overkill here
#   2. text-embedding-3-small is higher quality than most local models
#   3. One embedding provider = one failure mode to monitor
#   4. Already proven working in alignment_scorer and transcript_scorer
#
# All callers of _embed_texts (LOLA, Cyrene, ordinal_sync, cross_client_learning,
# market_intelligence) get the upgrade automatically. The model name change
# is detected by LOLA's arm_embeddings_model check, which triggers automatic
# re-embedding of all cached arm vectors on the next sync cycle.

_EMBEDDING_MODEL_NAME = "text-embedding-3-small"
_EMBEDDING_DIM = 1536
_ARM_MATCH_THRESHOLD = 0.30  # minimum cosine similarity to match an obs to an arm


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenAI. Returns list of float lists (JSON-serializable).

    Returns [] on failure — callers must handle the empty case (typically by
    falling back to substring matching or skipping the embedding-dependent path).
    """
    if not texts:
        return []
    try:
        from openai import OpenAI
        client = OpenAI()
        # token-safe truncation per text (8191 token limit for this model,
        # ~32k chars is a safe approximation)
        truncated = [t[:32000] for t in texts]
        resp = client.embeddings.create(
            input=truncated,
            model=_EMBEDDING_MODEL_NAME,
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        logger.warning("[LOLA] OpenAI embedding failed: %s", e)
        return []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

_ALPHA = 1.0          # UCB1 exploration coefficient (cold-start; learned after 5+ pulls)
_EXPLORATION_RATE = 0.20  # exploration rate (cold-start; decays with 1/sqrt(pulls))
_DEFAULT_RETIREMENT_THRESHOLD = 3  # consecutive declines before retirement (cold-start; adaptive after data)
_REST_CYCLES = 4           # rest pulls before retired arm returns (cold-start; adaptive)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------

_DEFAULT_ICP_RETIRE_THRESHOLD = 0.40
_DEFAULT_ICP_BOOST_THRESHOLD = 0.70
_DEFAULT_ICP_ROLLING_WINDOW = 3
_DEFAULT_ICP_BOOST_FACTOR = 1.5


@dataclass
class Arm:
    """A single content arm: a (topic_cluster, format_type) combination."""
    label: str                    # e.g., "protocol_error_storytelling"
    arm_type: str                 # "topic" | "format"
    description: str              # short human-readable description
    n_pulls: int = 0              # times selected
    sum_reward: float = 0.0       # cumulative reward
    last_reward: float = 0.0      # most recent reward (for decline detection)
    prev_reward: float = 0.0      # reward before last (for trend)
    positive_pulls: int = 0       # pulls where reward > 0 (for Thompson sampling)
    consecutive_declining: int = 0
    last_pulled_at: str = ""
    retired: bool = False
    rest_counter: int = 0         # increments each global pull when arm is retired
    # ICP flywheel fields
    icp_match_rates: list = field(default_factory=list)  # rolling window of per-post match rates
    icp_retired: bool = False     # retired specifically due to low ICP match rate
    icp_boosted: bool = False     # UCB bonus boosted due to high ICP match rate
    # Embedding for semantic arm matching (cached, computed once)
    embedding: list = field(default_factory=list)  # 1536-dim float list, or [] if not yet computed

    @property
    def mean_reward(self) -> float:
        return self.sum_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    def rolling_icp_match_rate_with_window(self, window: int = _DEFAULT_ICP_ROLLING_WINDOW) -> float:
        """Mean of the last N match rates, or -1 if insufficient data."""
        recent = self.icp_match_rates[-window:]
        if len(recent) < window:
            return -1.0
        return sum(recent) / len(recent)

    @property
    def rolling_icp_match_rate(self) -> float:
        """Mean of the last _DEFAULT_ICP_ROLLING_WINDOW match rates."""
        return self.rolling_icp_match_rate_with_window(_DEFAULT_ICP_ROLLING_WINDOW)

    def ucb_score(self, total_pulls: int, alpha: float = _ALPHA, boost_factor: float = _DEFAULT_ICP_BOOST_FACTOR) -> float:
        """UCB1 score with ICP boost. Unplayed arms get infinite score."""
        if self.n_pulls == 0:
            return float("inf")
        exploration_bonus = alpha * math.sqrt(math.log(max(total_pulls, 1)) / self.n_pulls)
        if self.icp_boosted:
            exploration_bonus *= boost_factor
        return self.mean_reward + exploration_bonus

    def thompson_sample(self) -> float:
        """Beta-Bernoulli Thompson sample for format arms.

        Uses per-pull success tracking (positive_pulls) for the Beta
        distribution. Alpha = positive_pulls + 1, Beta = failures + 1.
        """
        alpha = self.positive_pulls + 1
        beta = max(self.n_pulls - self.positive_pulls, 0) + 1
        return random.betavariate(alpha, beta)


@dataclass
class AdaptiveThresholds:
    """Per-client thresholds computed from the client's own data distribution."""
    icp_retire: float = _DEFAULT_ICP_RETIRE_THRESHOLD
    icp_boost: float = _DEFAULT_ICP_BOOST_THRESHOLD
    retirement_streak: int = _DEFAULT_RETIREMENT_THRESHOLD
    arm_match_sim: float = _ARM_MATCH_THRESHOLD
    exploration_rate: float = _EXPLORATION_RATE
    ucb_alpha: float = _ALPHA
    rest_cycles: int = _REST_CYCLES
    icp_rolling_window: int = _DEFAULT_ICP_ROLLING_WINDOW
    icp_boost_factor: float = _DEFAULT_ICP_BOOST_FACTOR
    icp_gap_min: float = 0.15                          # min gap between retire/boost thresholds
    computed_at: str = ""
    observation_count: int = 0


# -----------------------------------------------------------------------
# Continuous reward field (Phase 1 — embedding-native bandit)
# -----------------------------------------------------------------------

_MIN_POINTS_FOR_CONTINUOUS = 10  # fall back to arm-based below this
_DEFAULT_BANDWIDTH = 0.25        # Gaussian kernel bandwidth
_DEFAULT_TIME_DECAY = 0.02       # per-day exponential decay


@dataclass
class ContentPoint:
    """A single point in the continuous content-reward field."""
    embedding: list = field(default_factory=list)  # 1536-dim position
    reward: float = 0.0                            # z-scored engagement reward
    posted_at: str = ""
    post_hash: str = ""
    icp_match_rate: float = 0.0
    descriptor_snippet: str = ""                   # first 200 chars for label generation


@dataclass
class RewardFieldParams:
    """Learned parameters for the continuous reward field."""
    bandwidth: float = _DEFAULT_BANDWIDTH
    time_decay_lambda: float = _DEFAULT_TIME_DECAY
    icp_weight: float = 0.5        # how much ICP signal modulates reward
    exploration_alpha: float = 0.15 # exploration coefficient
    baseline_icp: float = 0.5      # mean ICP match rate for centering


@dataclass
class LOLAState:
    company: str
    arms: list[dict] = field(default_factory=list)           # legacy arm structure
    points: list[dict] = field(default_factory=list)         # continuous ContentPoint list
    field_params: dict = field(default_factory=lambda: asdict(RewardFieldParams()))
    total_pulls: int = 0
    created_at: str = ""
    last_updated: str = ""
    arm_embeddings_model: str = ""
    matched_obs_hashes: list = field(default_factory=list)
    thresholds: dict = field(default_factory=lambda: asdict(AdaptiveThresholds()))


# -----------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------

class LOLA:
    def __init__(self, company: str):
        self.company = company
        self._state = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        from backend.src.db.vortex import memory_dir
        return memory_dir(self.company) / "lola_state.json"

    def _load(self) -> LOLAState:
        p = self._state_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return LOLAState(**data)
            except Exception as e:
                logger.warning("[LOLA] Failed to load state for %s: %s", self.company, e)
        return LOLAState(company=self.company, created_at=_now())

    def _save(self) -> None:
        p = self._state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        self._state.last_updated = _now()
        p.write_text(json.dumps(asdict(self._state), indent=2), encoding="utf-8")

    def _arms(self) -> list[Arm]:
        return [Arm(**a) for a in self._state.arms]

    def _set_arms(self, arms: list[Arm]) -> None:
        self._state.arms = [asdict(a) for a in arms]

    def _get_thresholds(self) -> AdaptiveThresholds:
        """Return current adaptive thresholds from state, or sane defaults.

        Once the client has pulled enough arms to run ``recompute_thresholds``,
        the retire/boost cutoffs are learned directly from the distribution of
        arm icp_match_rates. Before that, we just use the AdaptiveThresholds
        defaults — no cold-start coupling to external threshold functions.
        """
        raw = self._state.thresholds
        if isinstance(raw, dict):
            try:
                return AdaptiveThresholds(**raw)
            except Exception:
                pass
        return AdaptiveThresholds()

    def recompute_thresholds(self) -> AdaptiveThresholds:
        """Recompute adaptive thresholds from the client's own data.

        Called each sync cycle. Thresholds are percentile-based:
        - icp_retire = 25th percentile of all arm ICP match rates
        - icp_boost = 75th percentile
        - retirement_streak = higher if reward variance is high (volatile client needs more patience)
        - arm_match_sim = 25th percentile of best-match similarities (from last matching run)

        No hard clamps — soft_bound() logs anomalies without clipping.
        """
        from backend.src.utils.adaptive_config import soft_bound

        arms = self._arms()
        all_icp_rates = []
        all_rewards = []

        for a in arms:
            all_icp_rates.extend(a.icp_match_rates)
            if a.n_pulls > 0:
                all_rewards.append(a.mean_reward)

        t = AdaptiveThresholds()

        # ICP thresholds from distribution
        if len(all_icp_rates) >= 6:
            sorted_rates = sorted(all_icp_rates)
            n = len(sorted_rates)
            t.icp_retire = sorted_rates[max(0, int(n * 0.25))]
            t.icp_boost = sorted_rates[min(n - 1, int(n * 0.75))]
            # Ensure retire < boost with minimum gap (adaptive)
            gap_min = t.icp_gap_min if hasattr(t, 'icp_gap_min') and t.icp_gap_min > 0 else 0.15
            if t.icp_boost - t.icp_retire < gap_min:
                mid = (t.icp_retire + t.icp_boost) / 2
                t.icp_retire = mid - gap_min / 2
                t.icp_boost = mid + gap_min / 2

        # Retirement streak: scale with reward variance.
        # Higher variance → need more samples to be confident arm has degraded.
        std = 0.0
        if len(all_rewards) >= 4:
            mean_r = sum(all_rewards) / len(all_rewards)
            variance = sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)
            std = math.sqrt(variance)
            t.retirement_streak = round(2 + std * 4)
            t.rest_cycles = round(4 + std * 3)

        # Exploration rate: 1/sqrt(total_pulls) (standard UCB-style decay).
        arm_count = len(arms) or 1
        total_pulls = self._state.total_pulls or 0
        if total_pulls >= 10:
            t.exploration_rate = round(1.0 / math.sqrt(max(total_pulls, 1)), 4)
        else:
            t.exploration_rate = _EXPLORATION_RATE  # cold-start default

        # UCB alpha: scale by arm density
        pull_density = total_pulls / max(arm_count, 1)
        t.ucb_alpha = round(_ALPHA * math.sqrt(arm_count / max(1, pull_density / 10)), 4)

        # ICP rolling window: longer for clients with more data
        t.icp_rolling_window = (total_pulls // 15) if total_pulls > 0 else _DEFAULT_ICP_ROLLING_WINDOW
        t.icp_rolling_window = t.icp_rolling_window or _DEFAULT_ICP_ROLLING_WINDOW

        # ICP boost factor: measured from actual performance difference
        boosted_rewards = [a.mean_reward for a in arms if a.icp_boosted and a.n_pulls > 0]
        baseline_rewards = [a.mean_reward for a in arms if not a.icp_boosted and a.n_pulls > 0]
        if boosted_rewards and baseline_rewards:
            avg_boosted = sum(boosted_rewards) / len(boosted_rewards)
            avg_baseline = sum(baseline_rewards) / len(baseline_rewards)
            if abs(avg_baseline) > 0.01:
                t.icp_boost_factor = round(avg_boosted / abs(avg_baseline), 2)
            else:
                t.icp_boost_factor = _DEFAULT_ICP_BOOST_FACTOR
        else:
            t.icp_boost_factor = _DEFAULT_ICP_BOOST_FACTOR

        # ICP gap minimum: wider for high-variance clients
        t.icp_gap_min = round(std * 0.3, 3) if std > 0 else 0.15

        # soft_bound: log anomalies without clipping any values
        _history = lambda default: [default]  # bootstrap: just the default for now
        t.retirement_streak = round(soft_bound(float(t.retirement_streak), _history(_DEFAULT_RETIREMENT_THRESHOLD), _DEFAULT_RETIREMENT_THRESHOLD))
        t.rest_cycles = round(soft_bound(float(t.rest_cycles), _history(_REST_CYCLES), _REST_CYCLES))
        t.ucb_alpha = soft_bound(t.ucb_alpha, _history(_ALPHA), _ALPHA)
        t.icp_boost_factor = soft_bound(t.icp_boost_factor, _history(_DEFAULT_ICP_BOOST_FACTOR), _DEFAULT_ICP_BOOST_FACTOR)

        t.computed_at = _now()
        t.observation_count = self._state.total_pulls

        self._state.thresholds = asdict(t)
        self._save()
        logger.info(
            "[LOLA] Recomputed thresholds for %s: icp_retire=%.2f, icp_boost=%.2f, retirement=%d",
            self.company, t.icp_retire, t.icp_boost, t.retirement_streak,
        )
        return t

    def _ensure_arm_embeddings(self) -> bool:
        """Compute and cache embeddings for any arm that doesn't have one.

        Returns True if embeddings are available (model loaded), False otherwise.
        """
        arms = self._arms()
        needs_embed = [a for a in arms if not a.embedding]

        if not needs_embed and self._state.arm_embeddings_model == _EMBEDDING_MODEL_NAME:
            return True  # All cached and model matches

        # Check if model changed (invalidate all cached embeddings)
        if self._state.arm_embeddings_model and self._state.arm_embeddings_model != _EMBEDDING_MODEL_NAME:
            logger.info("[LOLA] Embedding model changed, recomputing all arm embeddings")
            needs_embed = arms

        if not needs_embed:
            return True

        # Build texts to embed: "label: description"
        texts = [f"{a.label.replace('_', ' ')}: {a.description}" for a in needs_embed]
        embeddings = _embed_texts(texts)

        if not embeddings or len(embeddings) != len(needs_embed):
            return False  # Model not available

        for arm, emb in zip(needs_embed, embeddings):
            arm.embedding = emb

        # If embeddings were just computed (first time or model change),
        # clear the dedup set so all observations are re-matched using
        # embeddings instead of the inferior substring fallback.
        had_obs = len(self._state.matched_obs_hashes)
        if had_obs:
            logger.info(
                "[LOLA] Arm embeddings (re)computed — clearing %d matched hashes "
                "to re-match with embeddings (%s)",
                had_obs, self.company,
            )
            self._state.matched_obs_hashes = []

        self._state.arm_embeddings_model = _EMBEDDING_MODEL_NAME
        self._set_arms(arms)
        self._save()
        logger.info("[LOLA] Computed embeddings for %d arms (%s)", len(needs_embed), self.company)
        return True

    # ------------------------------------------------------------------
    # Arm management
    # ------------------------------------------------------------------

    def add_arm(self, label: str, arm_type: str, description: str) -> None:
        """Add a new arm if it doesn't already exist."""
        existing = {a["label"] for a in self._state.arms}
        if label not in existing:
            arms = self._arms()
            arms.append(Arm(label=label, arm_type=arm_type, description=description))
            self._set_arms(arms)
            self._save()

    def seed_arms(self, arms: list[dict]) -> int:
        """Seed multiple arms at once. Each dict: {label, arm_type, description}. Returns added count."""
        added = 0
        existing = {a["label"] for a in self._state.arms}
        current = self._arms()
        for spec in arms:
            if spec["label"] not in existing:
                current.append(Arm(
                    label=spec["label"],
                    arm_type=spec.get("arm_type", "topic"),
                    description=spec.get("description", ""),
                ))
                existing.add(spec["label"])
                added += 1
        if added:
            self._set_arms(current)
            self._save()
        return added

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, arm_type: str = "topic") -> Optional[Arm]:
        """Select next arm to explore/exploit.

        Follows UCB1 for topic arms, Thompson Sampling for format arms.
        Uses adaptive exploration rate (decays as bandit converges).
        """
        arms = [a for a in self._arms() if a.arm_type == arm_type and not a.retired]
        if not arms:
            return None

        total = self._state.total_pulls
        thresholds = self._get_thresholds()

        # Always try unpulled arms first.
        unpulled = [a for a in arms if a.n_pulls == 0]
        if unpulled:
            return random.choice(unpulled)

        # Exploration budget (adaptive: decays from 0.20 toward 0.10).
        if random.random() < thresholds.exploration_rate:
            low_n = sorted(arms, key=lambda a: a.n_pulls)[:max(1, len(arms) // 3)]
            return random.choice(low_n)

        # Exploitation: UCB1 (with adaptive alpha) or Thompson.
        if arm_type == "format":
            return max(arms, key=lambda a: a.thompson_sample())
        else:
            alpha = thresholds.ucb_alpha
            boost = thresholds.icp_boost_factor
            return max(arms, key=lambda a: a.ucb_score(total, alpha=alpha, boost_factor=boost))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def record_pull(self, label: str, reward: float) -> None:
        """Record an arm pull result. Called after a post's engagement is scored."""
        arms = self._arms()
        self._state.total_pulls += 1
        for arm in arms:
            if arm.label == label:
                arm.prev_reward = arm.last_reward
                arm.last_reward = reward
                arm.sum_reward += reward
                arm.n_pulls += 1
                if reward > 0:
                    arm.positive_pulls += 1
                arm.last_pulled_at = _now()
                # Track consecutive decline.
                if arm.n_pulls >= 2 and reward < arm.prev_reward:
                    arm.consecutive_declining += 1
                else:
                    arm.consecutive_declining = 0
                # Retire after threshold (adaptive per-client).
                retire_streak = self._get_thresholds().retirement_streak
                if arm.consecutive_declining >= retire_streak:
                    arm.retired = True
                    logger.info("[LOLA] Arm '%s' retired for %s (%d consecutive declines)", label, self.company, retire_streak)
                break
            # Increment rest counters for retired arms (adaptive rest cycles).
            if arm.retired:
                arm.rest_counter += 1
                rest_target = self._get_thresholds().rest_cycles
                if arm.rest_counter >= rest_target:
                    arm.retired = False
                    arm.rest_counter = 0
                    arm.consecutive_declining = 0
                    logger.info("[LOLA] Arm '%s' returned from retirement for %s", arm.label, self.company)
        self._set_arms(arms)
        self._save()

    def update_icp_signal(self, label: str, icp_match_rate: float) -> None:
        """Update an arm's ICP tracking after engager data arrives.

        Called by update_from_ruan_mei when an observation has icp_match_rate.
        Manages the rolling window, retirement, and boost signals.

        Args:
            label: The arm label.
            icp_match_rate: Mean continuous icp_score across this post's
                reactors, in [0, 1].
        """
        arms = self._arms()
        for arm in arms:
            if arm.label != label:
                continue

            # Append to rolling window
            arm.icp_match_rates.append(round(icp_match_rate, 4))

            # Check rolling rate for retirement / boost (adaptive thresholds)
            thresholds = self._get_thresholds()
            rolling = arm.rolling_icp_match_rate_with_window(thresholds.icp_rolling_window)
            if rolling < 0:
                break  # not enough data yet

            if rolling < thresholds.icp_retire and not arm.icp_retired:
                arm.icp_retired = True
                arm.retired = True
                arm.icp_boosted = False
                logger.info(
                    "[LOLA] Arm '%s' ICP-retired for %s (rolling %.1f%% < threshold %.0f%%)",
                    label, self.company, rolling * 100, thresholds.icp_retire * 100,
                )
            elif rolling >= thresholds.icp_boost:
                arm.icp_boosted = True
                arm.icp_retired = False
                retire_streak = thresholds.retirement_streak
                if arm.retired and not arm.consecutive_declining >= retire_streak:
                    arm.retired = False  # un-retire if it was ICP-retired
                logger.info(
                    "[LOLA] Arm '%s' ICP-boosted for %s (rolling %.1f%% >= threshold %.0f%%)",
                    label, self.company, rolling * 100, thresholds.icp_boost * 100,
                )
            else:
                arm.icp_boosted = False
                if arm.icp_retired:
                    arm.icp_retired = False
                    arm.retired = False  # un-retire: rate recovered above threshold
            break

        self._set_arms(arms)
        self._save()

    def update_from_ruan_mei(self) -> int:
        """Pull latest scored observations and ingest into the continuous reward field.

        Deduplicates: each observation (by post_hash) is only processed once.
        Returns number of new observations ingested.
        """
        try:
            from backend.src.agents.ruan_mei import RuanMei
        except ImportError:
            return 0

        rm = RuanMei(self.company)
        scored = [o for o in rm._state["observations"] if o.get("status") in ("scored", "finalized")]
        if not scored:
            return 0

        already_seen = set(self._state.matched_obs_hashes)
        new_obs = [
            o for o in scored
            if o.get("post_hash") and o["post_hash"] not in already_seen
        ]
        if not new_obs:
            return 0

        self._ingest_continuous_points(new_obs)
        updated = len(new_obs)

        # Record processed hashes
        for obs in new_obs:
            h = obs.get("post_hash")
            if h:
                self._state.matched_obs_hashes.append(h)
        # Cap the dedup list at 500 to prevent unbounded growth
        if len(self._state.matched_obs_hashes) > 500:
            self._state.matched_obs_hashes = self._state.matched_obs_hashes[-500:]
        self._save()

        return updated

    def _match_by_embedding(self, arms: list[Arm], observations: list[dict]) -> int:
        """Match observations to arms via cosine similarity of embeddings."""
        # Build arm embedding matrix
        arm_embeddings = []
        arm_labels = []
        for a in arms:
            if a.embedding:
                arm_embeddings.append(a.embedding)
                arm_labels.append(a.label)

        if not arm_embeddings:
            return self._match_by_substring(arms, observations)

        arm_matrix = np.array(arm_embeddings, dtype=np.float32)

        # Embed observation texts
        obs_texts = []
        for obs in observations:
            analysis = obs.get("descriptor", {}).get("analysis", "")
            body_snippet = (obs.get("posted_body") or obs.get("post_body") or "")[:300]
            obs_texts.append(f"{analysis} {body_snippet}".strip())

        obs_embeddings = _embed_texts(obs_texts)
        if not obs_embeddings:
            return self._match_by_substring(arms, observations)

        updated = 0
        for i, obs in enumerate(observations):
            if i >= len(obs_embeddings):
                break

            obs_emb = np.array(obs_embeddings[i], dtype=np.float32)

            # Compute similarity to all arms
            dots = arm_matrix @ obs_emb
            norms = np.linalg.norm(arm_matrix, axis=1) * np.linalg.norm(obs_emb)
            norms = np.where(norms == 0, 1.0, norms)
            similarities = dots / norms

            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            match_threshold = self._get_thresholds().arm_match_sim
            if best_sim < match_threshold:
                continue  # No arm is similar enough

            best_label = arm_labels[best_idx]
            reward = obs.get("reward", {}).get("immediate", 0)

            self.record_pull(best_label, reward)

            obs_icp_rate = obs.get("icp_match_rate")
            if obs_icp_rate is not None:
                self.update_icp_signal(best_label, icp_match_rate=obs_icp_rate)

            updated += 1
            logger.debug(
                "[LOLA] Embedding match: obs %s → arm '%s' (sim=%.3f)",
                obs.get("post_hash", "?")[:8], best_label, best_sim,
            )

        return updated

    def _match_by_substring(self, arms: list[Arm], observations: list[dict]) -> int:
        """Fallback: match observations to arms via keyword substring matching."""
        arm_labels = {a.label for a in arms}
        updated = 0

        for obs in observations:
            descriptor = obs.get("descriptor", {})
            analysis = (descriptor.get("analysis") or "").lower()
            posted_body = (obs.get("posted_body") or obs.get("post_body") or "").lower()
            combined = analysis + " " + posted_body

            for label in arm_labels:
                words = label.replace("_", " ").split()
                if len(words) >= 2 and all(w in combined for w in words):
                    reward = obs.get("reward", {}).get("immediate", 0)
                    self.record_pull(label, reward)
                    obs_icp_rate = obs.get("icp_match_rate")
                    if obs_icp_rate is not None:
                        self.update_icp_signal(label, icp_match_rate=obs_icp_rate)
                    updated += 1
                    break

        return updated

    # ------------------------------------------------------------------
    # Recommendation for Stelle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Continuous reward field (Phase 1)
    # ------------------------------------------------------------------

    def _ingest_continuous_points(self, observations: list[dict]) -> int:
        """Convert RuanMei observations to ContentPoints in the reward field."""
        existing_hashes = {p.get("post_hash") for p in self._state.points}
        texts_to_embed = []
        obs_for_embed = []

        for obs in observations:
            ph = obs.get("post_hash", "")
            if not ph or ph in existing_hashes:
                continue
            body = (obs.get("posted_body") or obs.get("post_body") or "")[:300]
            analysis = obs.get("descriptor", {}).get("analysis", "")
            if not body and not analysis:
                continue
            texts_to_embed.append(f"{analysis} {body}".strip())
            obs_for_embed.append(obs)

        if not texts_to_embed:
            return 0

        embeddings = _embed_texts(texts_to_embed)
        if not embeddings or len(embeddings) != len(obs_for_embed):
            return 0

        added = 0
        for obs, emb in zip(obs_for_embed, embeddings):
            point = ContentPoint(
                embedding=emb,
                reward=obs.get("reward", {}).get("immediate", 0),
                posted_at=obs.get("posted_at", ""),
                post_hash=obs.get("post_hash", ""),
                icp_match_rate=obs.get("icp_match_rate") or 0.0,
                descriptor_snippet=(obs.get("descriptor", {}).get("analysis", ""))[:200],
            )
            self._state.points.append(asdict(point))
            added += 1

        # Cap points at 500
        if len(self._state.points) > 500:
            self._state.points = self._state.points[-500:]

        if added:
            self._update_field_params()
            logger.info("[LOLA] Ingested %d content points for %s (total: %d)",
                        added, self.company, len(self._state.points))
        return added

    def _get_points(self) -> list[ContentPoint]:
        """Return continuous reward field points with valid, correctly-dimensioned embeddings.

        Filters out stale embeddings from a prior model (e.g., 384-dim vectors
        from the old sentence-transformers era). This causes the continuous field
        to rebuild from zero after an embedding model migration — the correct
        behavior, since old and new embeddings live in incompatible spaces.
        """
        return [
            ContentPoint(**p) for p in self._state.points
            if p.get("embedding") and len(p.get("embedding", [])) == _EMBEDDING_DIM
        ]

    def _use_continuous(self) -> bool:
        """True if we have enough correctly-dimensioned points for continuous selection."""
        return len(self._get_points()) >= _MIN_POINTS_FOR_CONTINUOUS

    def _continuous_blend_weight(self) -> float:
        """Blend weight for continuous vs arm-based recommendations.

        Returns 0.0 when fewer than _MIN_POINTS_FOR_CONTINUOUS points exist
        (pure arm-based), ramps linearly to 1.0 over the next 10 points
        (pure continuous). During the ramp, both systems contribute.
        """
        n = len(self._get_points())
        if n < _MIN_POINTS_FOR_CONTINUOUS:
            return 0.0
        ramp_length = 10
        return min(1.0, (n - _MIN_POINTS_FOR_CONTINUOUS) / ramp_length)

    def _update_field_params(self) -> None:
        """Recompute reward field parameters from data."""
        points = self._get_points()
        if len(points) < 5:
            return

        params = RewardFieldParams()

        # ICP weight: correlation between ICP match rate and reward
        # Negative correlation → 0 weight (ICP signal hurts, don't use it)
        icp_rates = [p.icp_match_rate for p in points if p.icp_match_rate > 0]
        rewards = [p.reward for p in points if p.icp_match_rate > 0]
        if len(icp_rates) >= 5:
            from backend.src.utils.correlation_analyzer import _spearman_correlation
            corr = _spearman_correlation(icp_rates, rewards)
            params.icp_weight = corr if corr > 0 else 0.0
            params.baseline_icp = sum(icp_rates) / len(icp_rates)

        # Exploration alpha: decay with data density
        params.exploration_alpha = 0.30 / math.sqrt(max(len(points), 1))

        # Bandwidth: from pairwise distances (median heuristic)
        if len(points) >= 10:
            sample = random.sample(points, min(50, len(points)))
            dists = []
            for i in range(len(sample)):
                for j in range(i + 1, min(i + 5, len(sample))):
                    d = np.linalg.norm(
                        np.array(sample[i].embedding) - np.array(sample[j].embedding)
                    )
                    dists.append(d)
            if dists:
                dists.sort()
                params.bandwidth = dists[len(dists) // 2]

        self._state.field_params = asdict(params)

    def _kernel_score(self, candidate: np.ndarray, points: list[ContentPoint],
                      params: RewardFieldParams) -> tuple[float, float]:
        """Compute expected reward and exploration bonus at a candidate position.

        Returns (expected_reward, exploration_bonus).
        """
        if not points:
            return 0.0, 1.0

        now = datetime.now(timezone.utc)
        bw_sq = params.bandwidth ** 2 * 2

        weighted_reward_sum = 0.0
        weight_sum = 0.0
        density = 0.0

        for p in points:
            p_emb = np.array(p.embedding, dtype=np.float32)
            dist_sq = float(np.sum((candidate - p_emb) ** 2))

            # Gaussian kernel
            kernel = math.exp(-dist_sq / max(bw_sq, 0.01))

            # Time decay
            try:
                dt = datetime.fromisoformat(p.posted_at.replace("Z", "+00:00"))
                days = max(0, (now - dt).total_seconds() / 86400)
            except Exception:
                days = 30
            time_weight = math.exp(-params.time_decay_lambda * days)

            # ICP-modulated reward
            effective_reward = p.reward * (1.0 + params.icp_weight * (p.icp_match_rate - params.baseline_icp))

            w = kernel * time_weight
            weighted_reward_sum += w * effective_reward
            weight_sum += w
            density += kernel

        expected_reward = weighted_reward_sum / max(weight_sum, 1e-8)
        # Normalize exploration bonus relative to the number of points
        # so it doesn't dominate expected reward for distant candidates
        max_density = float(len(points))
        exploration_bonus = (max_density - density) / max(max_density, 1.0)

        return expected_reward, exploration_bonus

    def select_continuous(self, n_candidates: int = 5) -> list[dict]:
        """Select top content directions using the continuous reward field.

        Returns list of dicts with 'embedding', 'expected_reward', 'label_snippets'.
        """
        points = self._get_points()
        if not points:
            return []

        params_dict = self._state.field_params
        params = RewardFieldParams(**params_dict) if params_dict else RewardFieldParams()

        # Build candidate set: recent points + reward centroid + random perturbations
        candidates = []

        # Recent posts (explore near what was just written)
        recent = sorted(points, key=lambda p: p.posted_at, reverse=True)[:3]
        for p in recent:
            candidates.append(np.array(p.embedding, dtype=np.float32))

        # Reward-weighted centroid (exploit the sweet spot)
        all_emb = np.array([p.embedding for p in points], dtype=np.float32)
        all_rewards = np.array([max(p.reward, 0) for p in points], dtype=np.float32)
        if all_rewards.sum() > 0:
            centroid = np.average(all_emb, axis=0, weights=all_rewards + 1e-8)
            candidates.append(centroid)

        # Random perturbations around centroid for exploration
        if len(candidates) > 0:
            base = candidates[-1] if all_rewards.sum() > 0 else all_emb.mean(axis=0)
            for _ in range(5):
                noise = np.random.randn(len(base)).astype(np.float32) * params.bandwidth * 0.5
                candidates.append(base + noise)

        # Score each candidate
        scored = []
        for c in candidates:
            exp_r, exp_bonus = self._kernel_score(c, points, params)
            acq = exp_r + params.exploration_alpha * exp_bonus
            scored.append((c, acq, exp_r, exp_bonus))

        # Sort by acquisition score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with nearest-neighbor labels
        results = []
        seen_directions = set()
        for emb, acq, exp_r, bonus in scored[:n_candidates * 2]:
            # Find nearest points for labeling
            dists = [
                (float(np.linalg.norm(emb - np.array(p.embedding))), p)
                for p in points
            ]
            dists.sort(key=lambda x: x[0])
            nearest = dists[:3]
            snippets = [p.descriptor_snippet[:100] for _, p in nearest if p.descriptor_snippet]

            # Deduplicate by snippet similarity
            key = snippets[0][:30] if snippets else str(len(results))
            if key in seen_directions:
                continue
            seen_directions.add(key)

            results.append({
                "embedding": emb.tolist(),
                "acquisition_score": round(acq, 4),
                "expected_reward": round(exp_r, 4),
                "exploration_bonus": round(bonus, 4),
                "label_snippets": snippets,
            })
            if len(results) >= n_candidates:
                break

        return results

    def recommend_context(self) -> str:
        """Generate a Stelle-ready context string with content direction recommendations.

        Uses the continuous reward field exclusively.  Returns a cold-start
        message when fewer than _MIN_POINTS_FOR_CONTINUOUS data points exist.
        """
        if not self._use_continuous():
            n = len(self._get_points())
            remaining = _MIN_POINTS_FOR_CONTINUOUS - n
            return (
                f"\n\nCONTENT INTELLIGENCE: collecting data ({n} posts scored, "
                f"{remaining} more needed before the system can provide data-driven "
                f"content direction recommendations)."
            )
        return self._recommend_continuous()

    def _recommend_continuous(self) -> str:
        """Generate context: actual post performance + sparsity-based exploration.

        Exploitation = actual top posts with real metrics (no interpolation).
        Exploration = density measurement in embedding space (valid in
        non-convex spaces — "where haven't we been?" is geometric, not
        a reward assumption).
        """
        points = self._get_points()
        if not points:
            return ""

        n_points = len(points)
        by_reward = sorted(points, key=lambda p: p.reward, reverse=True)

        lines = [f"\n\nCONTENT INTELLIGENCE ({n_points} scored posts):"]

        lines.append("Top performing posts (draw on what specifically worked in each):")
        for p in by_reward[:3]:
            hook = p.descriptor_snippet[:80] if p.descriptor_snippet else "(no preview)"
            lines.append(f"  • reward {p.reward:+.2f}: \"{hook}\"")

        if len(by_reward) > 5:
            lines.append("Underperformers (avoid these patterns):")
            for p in by_reward[-2:]:
                hook = p.descriptor_snippet[:80] if p.descriptor_snippet else "(no preview)"
                lines.append(f"  • reward {p.reward:+.2f}: \"{hook}\"")

        sparse = self._find_sparse_regions(points)
        if sparse:
            lines.append(
                "Exploration opportunity — underexplored territory this client "
                "hasn't tested yet (based on content space coverage, not reward "
                "interpolation):"
            )
            for label in sparse[:2]:
                lines.append(f"  • {label}")

        recent = sorted(points, key=lambda p: p.posted_at, reverse=True)[:3]
        if recent:
            avg_recent = sum(p.reward for p in recent) / len(recent)
            trend = "improving" if avg_recent > 0 else "declining"
            lines.append(f"Recent trajectory: {trend} (last 3 avg reward {avg_recent:+.2f})")

        lines.append(
            "These are actual post results, not interpolated directions. "
            "The content space is non-convex — draw on specific posts that worked, "
            "don't try to average or blend them."
        )
        return "\n".join(lines)

    def _find_sparse_regions(self, points: list[ContentPoint]) -> list[str]:
        """Identify underexplored regions by measuring local density.

        Density is a geometric property of the embedding space — it
        tells us where we *haven't* published, without assuming anything
        about the reward landscape.  Returns human-readable labels of
        the sparsest points (those furthest from their nearest neighbors).
        """
        if len(points) < 8:
            return []

        all_emb = np.array([p.embedding for p in points], dtype=np.float32)

        nn_dists = []
        for i, emb in enumerate(all_emb):
            dists = np.linalg.norm(all_emb - emb, axis=1)
            dists[i] = np.inf
            k = min(3, len(points) - 1)
            nn_dist = float(np.partition(dists, k)[:k].mean())
            nn_dists.append((nn_dist, points[i]))

        nn_dists.sort(key=lambda x: x[0], reverse=True)

        labels = []
        for dist, p in nn_dists[:3]:
            if p.descriptor_snippet:
                labels.append(
                    f"far from existing content (isolation {dist:.2f}): "
                    f"\"{p.descriptor_snippet[:60]}\""
                )
        return labels

    def _recommend_arm_based(self) -> str:
        """Legacy arm-based recommendations (fallback for < 10 points)."""
        topic_arms = [a for a in self._arms() if a.arm_type == "topic"]
        format_arms = [a for a in self._arms() if a.arm_type == "format"]

        if len(topic_arms) < 2:
            return ""

        lines = ["\n\nCONTENT INTELLIGENCE (from learning engine):"]

        # Top exploiting arms.
        ranked = sorted(
            [a for a in topic_arms if not a.retired and a.n_pulls > 0],
            key=lambda a: a.mean_reward,
            reverse=True,
        )
        if ranked:
            top = ranked[:2]
            lines.append(
                "Proven topic angles: "
                + "; ".join(
                    f'"{a.description}" (score {a.mean_reward:.2f}, {a.n_pulls} posts)'
                    for a in top
                )
            )

        # Exploration candidates (high uncertainty).
        unexplored = [a for a in topic_arms if not a.retired and a.n_pulls == 0]
        if unexplored:
            lines.append(
                "Underexplored angles (high uncertainty, worth testing): "
                + "; ".join(f'"{a.description}"' for a in unexplored[:2])
            )

        # Saturation warnings (includes ICP-retired arms).
        saturated = [a for a in topic_arms if a.retired and not a.icp_retired]
        icp_retired = [a for a in topic_arms if a.icp_retired]
        if saturated:
            lines.append(
                "Topic angles in saturation (avoid for now): "
                + ", ".join(f'"{a.description}"' for a in saturated[:3])
            )
        if icp_retired:
            lines.append(
                "Topic angles attracting wrong audience (ICP match <40%): "
                + ", ".join(
                    f'"{a.description}" ({a.rolling_icp_match_rate:.0%} ICP)'
                    for a in icp_retired[:3]
                )
            )

        # ICP-boosted arms (high-quality audience).
        icp_boosted = [a for a in topic_arms if a.icp_boosted and not a.retired]
        if icp_boosted:
            lines.append(
                "High ICP-quality angles (>70% on-profile engagers — prioritize): "
                + "; ".join(
                    f'"{a.description}" ({a.rolling_icp_match_rate:.0%} ICP, {a.n_pulls} posts)'
                    for a in icp_boosted[:3]
                )
            )

        # Format recommendation.
        if format_arms:
            best_format = self.select(arm_type="format")
            if best_format:
                if best_format.n_pulls == 0:
                    lines.append(f'Format to explore: "{best_format.description}" (untested)')
                else:
                    lines.append(
                        f'Recommended format: "{best_format.description}" '
                        f'(avg score {best_format.mean_reward:.2f})'
                    )

        if len(lines) == 1:
            return ""

        lines.append(
            "These are data-driven suggestions, not directives. "
            "Override if the source material strongly supports a different direction."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        arms = self._arms()
        topic = [a for a in arms if a.arm_type == "topic"]
        fmt = [a for a in arms if a.arm_type == "format"]
        points = self._get_points()
        return {
            "company": self.company,
            "total_pulls": self._state.total_pulls,
            "topic_arms": len(topic),
            "format_arms": len(fmt),
            "retired_arms": sum(1 for a in arms if a.retired),
            "top_topic": max(topic, key=lambda a: a.mean_reward).label if topic else None,
            "content_points": len(points),
            "using_continuous": self._use_continuous(),
            "field_params": self._state.field_params,
            "last_updated": self._state.last_updated,
        }
