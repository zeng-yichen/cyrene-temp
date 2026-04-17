"""
RuanMei — observation store and reward computer.

Scope (as of 2026-04-11): RuanMei owns the observation history and the
reward pipeline. It ingests posts from Ordinal analytics, analyses each
post into a free-text StrategyDescriptor, computes z-scored rewards
against the client's own distribution, and exposes those observations
to Stelle via the `pull_history` tool at generation time.

Retired from RuanMei's responsibilities:
  - `recommend_context()`: pre-chewed prose context for Stelle. Stelle
    now reads raw observations via `pull_history`.
  - `generate_content_strategy()`: prose content landscape brief.
    Deleted as a Bitter Lesson violation — a frozen intermediate
    representation between raw data and the writer.
  - `generate_insights()` and all expert-context helpers: same reason.

Two data sources feed the observation history:
1. Stelle-generated posts: analyzed post-hoc after generation, scored
   when Ordinal sync brings back engagement metrics.
2. Ordinal analytics: all published posts are ingested, analyzed, and
   scored directly. This is the primary data pipeline — it captures
   every post the client has ever published, not just our output.

Storage: SQLite (ruan_mei_state table, via backend.src.db.local.ruan_mei_save/load)
`ruan_mei_states` table (authoritative).

Usage:
    rm = RuanMei("runpod")

    # Ingest all published posts from Ordinal:
    await rm.ingest_from_ordinal(analytics_posts)

    # After Stelle generates a post:
    descriptor = await rm.analyze_post(post_text)
    rm.record(post_hash, descriptor, post_body=text, local_post_id=draft_uuid)

    # After push to Ordinal:
    rm.link_ordinal_post_id(draft_uuid, ordinal_workspace_post_id)

    # When Ordinal sync brings back metrics for pending posts:
    rm.update_by_ordinal_post_id(ordinal_id, metrics, posted_body=live_copy)
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import anthropic

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

# How many recent posts to include in content state tracking.
RECENT_HISTORY_LENGTH = 10

# How many recent posts to include in the content trajectory section.
TRAJECTORY_POSTS = 8

# Observation statuses that carry usable engagement data.
# "scored" = metrics collected but post may still accrue engagement.
# "finalized" = post is >= 7 days old; metrics are mature and final.
_SCORED_STATUSES = {"scored", "finalized"}

# Minimum post age before we collect engagement metrics and ICP data.
_POST_SETTLE_DAYS = 7


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class StrategyDescriptor:
    """Open-ended description of a post's notable characteristics.

    No predefined dimensions. The LLM decides what's worth noting about
    the post's construction, style, and approach. This avoids forcing
    analysis along human-designed axes (hook type, tone, etc.) that may
    not capture what actually drives engagement.
    """
    analysis: str = ""
    char_count: int = 0


@dataclass
class ContentState:
    """Content calendar context at the time of posting.

    Phase 4 sequential fields (topic_fatigue, momentum, audience_decay)
    track audience-level state across posts. These enable future MDP-based
    selection: the current system treats each post independently (bandit),
    but real audiences have memory — topic fatigue, attention decay, and
    algorithmic momentum affect the next post's ceiling.
    """
    posts_this_week: int = 0
    days_since_last_post: int = 0
    posting_streak: int = 0
    engagement_trend: float = 0.0
    avg_impressions_recent: float = 0.0
    total_observations: int = 0
    # Phase 4: Sequential state fields (embedding-based, no discrete labels)
    similarity_streak: int = 0       # consecutive posts above the client's median pairwise similarity
    recent_similarity: float = 0.0   # avg cosine similarity of last 3 posts to each other
    momentum: float = 0.0            # exponential moving average of recent rewards
    audience_decay_days: float = 0.0  # days since last high-performing post (reward > 0)


@dataclass
class Observation:
    """A single post observation: what was written, how it performed."""
    post_hash: str
    descriptor: dict  # serialized StrategyDescriptor
    content_state: dict  # serialized ContentState
    reward: Optional[dict] = None  # {immediate, raw_metrics} or None if pending
    post_body: str = ""  # draft / pre-push copy (from generation)
    posted_body: str = ""  # live LinkedIn copy from analytics when it differs from draft
    posted_at: str = ""
    recorded_at: str = ""
    scored_at: str = ""  # timestamp of most recent engagement scoring
    status: str = "pending"  # pending | scored | finalized
    local_post_id: str = ""  # SQLite local_posts.id — links push → observation
    ordinal_post_id: str = ""  # Ordinal workspace post id after push — primary sync key
    icp_reward: Optional[float] = None  # ICP quality signal in [-1, 1]; None until engagers fetched
    linkedin_post_url: str = ""  # Live LinkedIn URL (populated from ordinal analytics payload)
    edit_similarity: float = -1.0  # SequenceMatcher ratio of draft vs published (-1 = not yet computed)
    # DISPLAY-ONLY tags from observation_tagger (None until tagged).
    # Used for human dashboards/reporting. NOT consumed by any learning subsystem.
    # The learning pipeline operates on continuous post embeddings instead.
    topic_tag: Optional[str] = None
    source_segment_type: Optional[str] = None
    format_tag: Optional[str] = None
    # Source tracking — populated at generation time when the strategy brief
    # pipeline creates posts with explicit source attribution. Null for posts
    # ingested from Ordinal analytics without draft provenance.
    source_transcript: Optional[str] = None   # filename of source transcript
    abm_target: Optional[str] = None          # company name if ABM-targeted
    # active_directives, analyst_findings_version, analyst_findings_count
    # were previously stamped at generation time by the feedback_distiller
    # and analyst injection paths. Both paths have been removed as
    # prescriptive rule libraries. The fields are kept off the dataclass;
    # any legacy data on old observations is scrubbed before the writer
    # reads it (see _STELLE_STRIPPED_FIELDS in stelle.py).
    # Prediction tracking — the draft scorer's predicted engagement score
    # at generation time. After the post is scored with actual engagement,
    # prediction_error = predicted - actual. This closes the validation
    # loop: does the model actually get better over time?
    predicted_engagement: Optional[float] = None
    # Constitutional verification — populated post-hoc by ordinal_sync or
    # at generation time by Stelle. Tracks per-principle pass/fail and the
    # weighted constitutional score.
    constitutional_score: Optional[float] = None
    constitutional_results: Optional[dict] = None
    # Strategy attribution — stamped at generation time with the strategy
    # slot that this post was generated to fulfill. Enables the strategy
    # retrospective: after a cycle, we can evaluate which hypotheses were
    # tested and whether the plan's learning objectives were met.
    strategy_slot: Optional[dict] = None


# ------------------------------------------------------------------
# Core class
# ------------------------------------------------------------------

class RuanMei:
    """
    Content strategist and performance learner with closed-loop evaluation.

    Generates rolling content strategies (strong defaults for Stelle),
    observes post performance and client edits, and runs retrospective
    evaluation of each strategy cycle. The feedback loop:
    Strategy → Stelle drafts → Client edits → Live performance → Retrospective → Next Strategy.
    """

    def __init__(self, company: str):
        self.company = company
        self._state = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_post(self, post_text: str) -> StrategyDescriptor:
        """
        Open-ended post analysis. The LLM decides what's notable about
        the post's construction, not us. No predefined dimensions.
        """
        prompt = (
            "Analyze this LinkedIn post. What are the most notable "
            "characteristics of how it's written? Describe whatever stands "
            "out to you about its construction, style, approach, and content. "
            "Be specific and concrete. Reference actual phrases or structural "
            "choices from the post.\n\n"
            "Do NOT use generic marketing labels like 'thought leadership' or "
            "'engaging hook.' Describe the actual mechanics of what the author did.\n\n"
            f"Post:\n{post_text}\n\n"
            "Write 3-5 sentences of analysis. Plain text, no JSON, no bullet points."
        )

        from backend.src.mcp_bridge.claude_cli import (
            use_cli as _use_cli,
            cli_single_shot as _cli_ss,
        )
        if _use_cli():
            txt = _cli_ss(prompt, model="sonnet", max_tokens=300) or ""
            return StrategyDescriptor(
                analysis=txt.strip(),
                char_count=len(post_text),
            )

        try:
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis = resp.content[0].text.strip()
            return StrategyDescriptor(
                analysis=analysis,
                char_count=len(post_text),
            )
        except Exception as e:
            logger.warning("[RuanMei] Post analysis failed: %s", e)
            return StrategyDescriptor(char_count=len(post_text))

    def record(
        self,
        post_hash: str,
        descriptor: StrategyDescriptor,
        content_state: Optional[ContentState] = None,
        post_body: str = "",
        local_post_id: str = "",
    ) -> None:
        """Record a generated post's strategy descriptor. Awaits reward from Ordinal.

        ``local_post_id`` should be the same id as ``local_posts.id`` so a later Ordinal
        push can attach ``ordinal_post_id`` without relying on unchanged body text.
        """
        if content_state is None:
            content_state = self.build_content_state()

        obs = Observation(
            post_hash=post_hash,
            descriptor=asdict(descriptor),
            content_state=asdict(content_state),
            post_body=post_body,
            posted_body="",
            recorded_at=_now(),
            status="pending",
            local_post_id=(local_post_id or "").strip(),
            ordinal_post_id="",
        )

        # Drop prior row for same draft id or same content hash.
        self._state["observations"] = [
            o
            for o in self._state["observations"]
            if (not obs.local_post_id or o.get("local_post_id") != obs.local_post_id)
            and o.get("post_hash") != post_hash
        ]
        self._state["observations"].append(asdict(obs))
        self._save()
        logger.info(
            "[RuanMei] Recorded post for %s (hash=%s local_id=%s)",
            self.company,
            post_hash,
            obs.local_post_id or "—",
        )

    def link_ordinal_post_id(self, local_post_id: str, ordinal_post_id: str) -> bool:
        """After a push, associate the Ordinal workspace id with the observation (any status)."""
        lid = (local_post_id or "").strip()
        oid = (ordinal_post_id or "").strip()
        if not lid or not oid:
            return False
        for obs in self._state["observations"]:
            if obs.get("local_post_id") == lid:
                obs["ordinal_post_id"] = oid
                self._save()
                logger.info("[RuanMei] Linked Ordinal id %s… to local draft %s", oid[:12], lid[:8])
                return True
        return False

    @staticmethod
    def _post_age_days(obs: dict) -> float | None:
        """Days since the post was published on LinkedIn. None if unknown."""
        posted_at = obs.get("posted_at", "")
        if not posted_at:
            return None
        try:
            ts = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - ts).total_seconds() / 86400
        except Exception:
            return None

    def _apply_update(
        self, obs: dict, metrics: dict,
        posted_body: str = "", ordinal_post_id: str = "",
        linkedin_post_url: str = "", _defer_save: bool = False,
        label: str = "",
    ) -> bool:
        """Shared logic for update() and update_by_ordinal_post_id().

        Lifecycle:
        1. Already finalized -> skip entirely.
        2. Pre-settle (< 7 days) -> only stash posted_at so we can
           compute age on the next cycle. No other work.
        3. Post-settle (>= 7 days) -> collect all metadata, score with
           mature metrics, and mark finalized in one pass.
        """
        # 1. Already finalized — absolute no-op.
        if obs.get("status") == "finalized":
            return True

        # 2. Ensure we have posted_at so age gating works.
        if metrics.get("posted_at") and not obs.get("posted_at"):
            obs["posted_at"] = metrics["posted_at"]

        # 3. Always update metadata + trajectory snapshot, even before
        #    the post has settled. The trajectory (metrics_history +
        #    analyze_trajectory) is the most valuable data in the first
        #    72 hours — early velocity, growth curve, plateau detection.
        #    Blocking this behind the 7-day age gate was a bug: the fast
        #    sync captures snapshots every 15 min for fresh posts, but
        #    _apply_update threw them away until day 7.
        if ordinal_post_id:
            obs["ordinal_post_id"] = ordinal_post_id
        if linkedin_post_url:
            obs["linkedin_post_url"] = linkedin_post_url
        obs["posted_at"] = metrics.get("posted_at", obs.get("posted_at", obs.get("recorded_at", "")))
        if posted_body.strip():
            obs["posted_body"] = posted_body.strip()
            sim = _compute_edit_similarity(obs.get("post_body", ""), posted_body)
            if sim >= 0:
                obs["edit_similarity"] = round(sim, 4)

        # Always append the metrics snapshot + recompute trajectory.
        # This is what the fast sync (15-min interval) feeds into.
        _append_metrics_snapshot(obs, metrics)

        # Update the raw metrics on the observation so consumers see
        # the latest numbers, not the stale ones from first ingest.
        # The reward score is NOT recomputed until finalization (T+7d)
        # because the z-score is meaningless against a partially-
        # accumulated baseline. But the raw counts should be fresh.
        if not obs.get("reward"):
            obs["reward"] = {}
        obs["reward"]["raw_metrics"] = {
            "impressions": metrics.get("impressions", 0),
            "reactions": metrics.get("reactions", 0),
            "comments": metrics.get("comments", 0),
            "reposts": metrics.get("reposts", 0),
            "posted_at": metrics.get("posted_at", ""),
        }

        # If the post is fresh enough to have been ingested but not yet
        # scored at all, mark it as scored so it's visible to Irontomb
        # and pull_history (they filter on scored/finalized).
        if obs.get("status") == "pending":
            obs["status"] = "scored"

        # 4. Age gate: don't FINALIZE (lock in reward z-score) until
        #    the post has settled. Trajectory + raw metrics update above
        #    always runs; only the reward computation waits.
        age = self._post_age_days(obs)
        if age is not None and age < _POST_SETTLE_DAYS:
            if not _defer_save:
                self._save()
            return True

        # 5. Finalization pass — compute reward z-score with mature metrics.
        reward = self._compute_reward(metrics)
        obs["reward"] = reward
        obs["status"] = "finalized"
        obs["scored_at"] = _now()
        if not _defer_save:
            self._save()
        logger.info(
            "[RuanMei] Finalized post for %s: immediate=%.3f (%s)",
            self.company, reward.get("immediate", 0), label,
        )
        return True

    def update(
        self,
        post_hash: str,
        metrics: dict,
        posted_body: str = "",
        ordinal_post_id: str = "",
        linkedin_post_url: str = "",
        _defer_save: bool = False,
    ) -> bool:
        """Score an observation matched by text hash.

        Age-gated: posts < 7 days old get metadata backfilled only.
        Posts >= 7 days old get scored once and marked finalized.
        """
        for obs in self._state["observations"]:
            if obs.get("post_hash") == post_hash and obs.get("status") in ("pending", "scored", "finalized"):
                return self._apply_update(
                    obs, metrics,
                    posted_body=posted_body,
                    ordinal_post_id=ordinal_post_id,
                    linkedin_post_url=linkedin_post_url,
                    _defer_save=_defer_save,
                    label=f"hash={post_hash}",
                )
        return False

    def update_by_ordinal_post_id(
        self, ordinal_post_id: str, metrics: dict,
        posted_body: str = "", linkedin_post_url: str = "",
        _defer_save: bool = False,
    ) -> bool:
        """Match observation by Ordinal workspace id.

        Age-gated: posts < 7 days old get metadata backfilled only.
        Posts >= 7 days old get scored once and marked finalized.
        """
        oid = (ordinal_post_id or "").strip()
        if not oid:
            return False
        for obs in self._state["observations"]:
            if obs.get("ordinal_post_id") == oid and obs.get("status") in ("pending", "scored", "finalized"):
                return self._apply_update(
                    obs, metrics,
                    posted_body=posted_body,
                    ordinal_post_id="",
                    linkedin_post_url=linkedin_post_url,
                    _defer_save=_defer_save,
                    label=f"ordinal_id={oid[:12]}…",
                )
        return False

    def update_by_text(
        self,
        post_text: str,
        metrics: dict,
        ordinal_post_id: str = "",
        linkedin_post_url: str = "",
        _defer_save: bool = False,
    ) -> bool:
        """
        Try to match a post by text hash and update it.
        Fallback when analytics payload has no Ordinal post id or id was not linked.
        Propagates ``ordinal_post_id`` and ``linkedin_post_url`` so they get
        backfilled on observations that were created before these fields existed.
        """
        post_hash = _hash(post_text)
        return self.update(
            post_hash, metrics,
            posted_body=post_text,
            ordinal_post_id=ordinal_post_id,
            linkedin_post_url=linkedin_post_url,
            _defer_save=_defer_save,
        )

    def update_icp_reward(
        self,
        ordinal_post_id: str,
        icp_score: float,
        linkedin_post_url: str = "",
        icp_match_rate: float | None = None,
    ) -> bool:
        """Store ICP reward signal on a scored observation and recompute composite.

        Called by ordinal_sync after engager profiles are fetched and scored.
        Stores the continuous icp_score aggregate (`icp_match_rate`) directly.
        No bucketed segment counts — per-reactor raw scores live in the
        post_engagers SQLite table and can be joined in at query time.

        Returns True when a matching observation was found and updated.
        """
        oid = (ordinal_post_id or "").strip()
        if not oid:
            return False
        for obs in self._state["observations"]:
            if obs.get("ordinal_post_id") != oid:
                continue
            if obs.get("status") not in _SCORED_STATUSES:
                continue
            # Recompute reward blending in the ICP signal.
            existing_reward = obs.get("reward", {})
            raw = existing_reward.get("raw_metrics", {})
            if raw:
                obs["reward"] = self._compute_reward(raw, icp_score=icp_score)
            else:
                # No raw metrics stored (shouldn't happen, but be safe).
                obs["icp_reward"] = round(icp_score, 4)
            if icp_match_rate is not None:
                obs["icp_match_rate"] = round(icp_match_rate, 4)
            if linkedin_post_url:
                obs["linkedin_post_url"] = linkedin_post_url
            # Clean up any legacy icp_segments field that may still be present
            # from older writes — this field is no longer a learning signal.
            if "icp_segments" in obs:
                obs.pop("icp_segments", None)
            self._save()
            logger.info(
                "[RuanMei] ICP reward updated for %s: icp=%.4f match_rate=%s (ordinal_id=%s…)",
                self.company, icp_score,
                f"{icp_match_rate:.2%}" if icp_match_rate is not None else "N/A",
                oid[:12],
            )
            return True
        return False

    # -------------------------------------------------------------
    # Retired 2026-04-11: content-intelligence / landscape codepath.
    # `generate_insights`, `recommend_context`,
    # `_assemble_content_intelligence`, `_build_expert_context`,
    # `_recommend_analyst`, `_load_latest_findings`,
    # `_format_all_posts_for_analyst`, `_format_exemplars`,
    # `_find_sparse_regions`, `_find_unexplored_territory`,
    # `_query_linkedin_bank`, `_build_content_trajectory`,
    # `_build_icp_summary`, `_build_edit_delta_summary`,
    # `_build_feedback_patterns`, `_build_editorial_convergence_summary`,
    # `_build_client_historical_context`, `_build_editorial_preferences`,
    # and `generate_cross_client_insights` were pre-chewed prose
    # intermediate representations that Stelle no longer reads. Deleted
    # as Bitter Lesson violations. Stelle queries raw observations
    # directly via `pull_history` / `run_py` tools.
    # -------------------------------------------------------------

    def build_content_state(self) -> ContentState:
        """Build current content state from observation history."""
        scored = [o for o in self._state["observations"] if o.get("status") in _SCORED_STATUSES]
        all_obs = self._state["observations"]

        if not all_obs:
            return ContentState()

        # Engagement trend: slope of immediate reward over last N scored posts.
        trend = 0.0
        recent_scored = scored[-RECENT_HISTORY_LENGTH:]
        if len(recent_scored) >= 3:
            rewards = [o.get("reward", {}).get("immediate", 0) for o in recent_scored]
            n = len(rewards)
            x_mean = (n - 1) / 2
            y_mean = sum(rewards) / n
            num = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(rewards))
            den = sum((i - x_mean) ** 2 for i in range(n))
            trend = num / den if den > 0 else 0.0

        # Average impressions.
        impressions = [
            o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0)
            for o in recent_scored if o.get("reward")
        ]
        avg_imp = sum(impressions) / len(impressions) if impressions else 0.0

        # Days since last post.
        days_since = 0
        if all_obs:
            last_ts = all_obs[-1].get("recorded_at", "")
            if last_ts:
                try:
                    last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                    days_since = (datetime.now(timezone.utc) - last_dt).days
                except (ValueError, TypeError):
                    pass

        # Posts this week (last 7 days).
        now = datetime.now(timezone.utc)
        posts_this_week = 0
        for o in all_obs:
            ts = o.get("recorded_at", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if (now - dt).days <= 7:
                        posts_this_week += 1
                except (ValueError, TypeError):
                    pass

        # Sequential state (embedding-based)
        similarity_streak = 0
        recent_similarity = 0.0
        momentum = 0.0
        audience_decay_days = 0.0
        if recent_scored:
            try:
                from backend.src.utils.post_embeddings import get_post_embeddings, cosine_similarity
                embs = get_post_embeddings(self.company)
                last_hash = recent_scored[-1].get("post_hash")
                if last_hash and last_hash in embs:
                    last_emb = embs[last_hash]

                    # Derive similarity threshold from the data: median
                    # pairwise similarity across the client's post pool.
                    all_hashes = [h for h in embs if h != last_hash]
                    if all_hashes:
                        all_sims = [
                            cosine_similarity(last_emb, embs[h])
                            for h in all_hashes
                        ]
                        all_sims.sort()
                        sim_threshold = all_sims[len(all_sims) // 2]
                    else:
                        sim_threshold = 0.5

                    for o in reversed(recent_scored[:-1]):
                        h = o.get("post_hash")
                        if h and h in embs:
                            sim = cosine_similarity(last_emb, embs[h])
                            if sim > sim_threshold:
                                similarity_streak += 1
                            else:
                                break
                        else:
                            break

                    last_n_hashes = [
                        o.get("post_hash") for o in recent_scored[-3:]
                        if o.get("post_hash") in embs
                    ]
                    if len(last_n_hashes) >= 2:
                        sims = []
                        for i in range(len(last_n_hashes)):
                            for j in range(i + 1, len(last_n_hashes)):
                                sims.append(cosine_similarity(
                                    embs[last_n_hashes[i]], embs[last_n_hashes[j]]
                                ))
                        recent_similarity = sum(sims) / len(sims) if sims else 0.0
            except Exception:
                pass

            # EMA alpha derived from lag-1 autocorrelation of rewards:
            # high autocorrelation → low alpha (long memory)
            # low autocorrelation → high alpha (short memory)
            # At N < 5 the autocorrelation estimate is dominated by noise,
            # so we use Wilder's formula (window = N) as a stable fallback.
            rewards_seq = [
                o.get("reward", {}).get("immediate", 0) for o in recent_scored
            ]
            if len(rewards_seq) >= 5:
                mean_r = sum(rewards_seq) / len(rewards_seq)
                num = sum(
                    (rewards_seq[i] - mean_r) * (rewards_seq[i + 1] - mean_r)
                    for i in range(len(rewards_seq) - 1)
                )
                den = sum((r - mean_r) ** 2 for r in rewards_seq)
                autocorr = num / den if den > 0 else 0.0
                alpha = 1.0 - abs(autocorr)
            else:
                alpha = 2.0 / (len(rewards_seq) + 1)

            for o in recent_scored:
                r = o.get("reward", {}).get("immediate", 0)
                momentum = alpha * r + (1 - alpha) * momentum

            for o in reversed(recent_scored):
                r = o.get("reward", {}).get("immediate", 0)
                if r > 0:
                    ts = o.get("posted_at") or o.get("recorded_at", "")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            audience_decay_days = (datetime.now(timezone.utc) - dt).days
                        except (ValueError, TypeError):
                            pass
                    break

        return ContentState(
            posts_this_week=posts_this_week,
            days_since_last_post=days_since,
            posting_streak=self._compute_streak(all_obs),
            engagement_trend=round(trend, 4),
            avg_impressions_recent=round(avg_imp, 1),
            total_observations=len(scored),
            similarity_streak=similarity_streak,
            recent_similarity=round(recent_similarity, 4),
            momentum=round(momentum, 4),
            audience_decay_days=round(audience_decay_days, 1),
        )

    async def ingest_from_ordinal(
        self,
        posts: list[dict],
        batch_size: int = 50,
    ) -> int:
        """
        Ingest published posts from Ordinal analytics into the observation
        history. Each post gets analyzed and recorded as a scored observation
        so RuanMei learns from ALL client posts, not just Stelle-generated ones.

        Posts that already exist in the observation history (by text hash) are
        skipped to avoid duplicates.

        Args:
            posts: Raw post dicts from Ordinal analytics API.
            batch_size: Max posts to analyze per call (controls LLM cost).

        Returns number of posts successfully ingested.
        """
        existing_hashes = {
            o.get("post_hash") for o in self._state["observations"]
        }

        # Load draft map once — lets us distinguish Stelle-generated posts
        # (draft text available) from externally-authored posts (no draft).
        # When a draft exists, we populate post_body with the draft and
        # posted_body with the live analytics text, so edit_similarity
        # captures what the human changed before publishing. This is the
        # signal the feedback distiller uses to learn editorial preferences.
        draft_map = self._load_draft_map()

        ingested = 0
        for post in posts:
            if ingested >= batch_size:
                break

            text = (
                post.get("commentary")
                or post.get("text")
                or post.get("copy")
                or post.get("content")
                or post.get("post_text")
                or ""
            ).strip()
            if not text:
                continue

            post_hash = _hash(text)
            if post_hash in existing_hashes:
                continue

            impressions = post.get("impressionCount") or post.get("impressions") or 0
            reactions = post.get("likeCount") or post.get("reactions") or post.get("total_reactions") or 0
            comments = post.get("commentCount") or post.get("comments") or post.get("total_comments") or 0
            reposts = post.get("shareCount") or post.get("repostCount") or post.get("reposts") or 0

            if impressions == 0:
                continue

            posted_at = (
                post.get("publishedAt")
                or post.get("postedAt")
                or post.get("published_at")
                or ""
            )

            # Capture LinkedIn post URL for engager fetching.
            linkedin_url = (
                post.get("url")
                or post.get("linkedInUrl")
                or post.get("linkedin_url")
                or post.get("postUrl")
                or ""
            )

            # Capture Ordinal post id for linking.
            op = post.get("ordinalPost")
            ordinal_pid = ""
            if isinstance(op, dict):
                ordinal_pid = str(op.get("id") or "").strip()

            # Analyze the post (same as Stelle-generated posts).
            descriptor = await self.analyze_post(text)

            # Compute reward from metrics.
            metrics = {
                "impressions": impressions,
                "reactions": reactions,
                "comments": comments,
                "reposts": reposts,
                "posted_at": posted_at,
            }
            reward = self._compute_reward(metrics)

            # Resolve the Stelle draft that corresponds to this live post.
            # Tries exact ordinal_post_id lookup first, then fuzzy text match
            # against all draft_map entries. If neither path finds anything,
            # the post is externally authored (Sachil wrote it directly in
            # Ordinal, or it was imported from LinkedIn before Stelle existed).
            draft_text, matched_key = _find_matching_draft(draft_map, ordinal_pid, text)

            if draft_text:
                obs_post_body = draft_text
                obs_posted_body = text
                obs_edit_sim = _compute_edit_similarity(draft_text, text)
                if matched_key != ordinal_pid:
                    logger.info(
                        "[RuanMei] Paired draft via fuzzy match for %s: "
                        "ordinal_pid=%s draft_key=%s sim=%.3f",
                        self.company,
                        (ordinal_pid or "<empty>")[:12],
                        matched_key[:12],
                        obs_edit_sim,
                    )
            else:
                # No matching draft — treat as externally-authored. The live
                # text is the authoritative version; there's no "before."
                obs_post_body = text
                obs_posted_body = ""
                obs_edit_sim = -1.0

            # Record as a fully scored observation.
            obs = Observation(
                post_hash=post_hash,
                descriptor=asdict(descriptor),
                content_state=asdict(self.build_content_state()),
                reward=reward,
                post_body=obs_post_body,
                posted_body=obs_posted_body,
                edit_similarity=round(obs_edit_sim, 4) if obs_edit_sim >= 0 else -1.0,
                posted_at=posted_at,
                recorded_at=_now(),
                status="scored",
                local_post_id="",
                ordinal_post_id=ordinal_pid,
                linkedin_post_url=linkedin_url,
            )

            self._state["observations"].append(asdict(obs))
            existing_hashes.add(post_hash)
            ingested += 1

        if ingested:
            self._save()
            logger.info(
                "[RuanMei] Ingested %d posts from Ordinal for %s",
                ingested, self.company,
            )

        return ingested

    def backfill_edit_similarity_from_draft_map(self) -> int:
        """Fix observations where post_body holds live text instead of the draft.

        Walks existing scored observations. For each one, tries to resolve a
        matching Stelle draft from draft_map.json via exact ordinal_post_id
        lookup. When a draft is found, promotes the current post_body (which
        was set to live text by the pre-fix ingest_from_ordinal path) to
        posted_body and replaces post_body with the real Stelle draft.
        Recomputes edit_similarity on the corrected pair for legacy consumers
        that still read that field (the model-facing code paths no longer
        surface the scalar — they read the two raw texts directly).

        Idempotent: skips observations where post_body already equals the
        draft text, or where no draft can be matched.

        Returns the number of observations modified.
        """
        draft_map = self._load_draft_map()
        if not draft_map:
            return 0

        modified = 0
        for obs in self._state.get("observations", []):
            if obs.get("status") not in _SCORED_STATUSES:
                continue
            current_pb = (obs.get("post_body") or "").strip()
            current_posted = (obs.get("posted_body") or "").strip()
            # The live text is whichever field currently holds the analytics
            # copy. Before the fix, ingest_from_ordinal stored it in post_body.
            live = current_posted or current_pb
            if not live:
                continue
            oid = (obs.get("ordinal_post_id") or "").strip()
            draft, _matched_key = _find_matching_draft(draft_map, oid, live)
            if not draft:
                continue
            if current_pb == draft:
                continue  # already correct
            obs["post_body"] = draft
            obs["posted_body"] = live
            sim = _compute_edit_similarity(draft, live)
            obs["edit_similarity"] = round(sim, 4) if sim >= 0 else -1.0
            modified += 1

        if modified:
            self._save()
            logger.info(
                "[RuanMei] Backfilled (draft, published) pairings for "
                "%d observations (%s)",
                modified, self.company,
            )

        return modified

    def _load_draft_map(self) -> dict:
        """Load draft_map.json for this company. Returns empty dict if missing."""
        path = P.draft_map_path(self.company)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update_constitutional(self, post_hash: str, score: float, results: dict) -> bool:
        """Store constitutional verification results on an observation.

        Returns True if the observation was found and updated.
        """
        for obs in self._state.get("observations", []):
            if obs.get("post_hash") == post_hash:
                obs["constitutional_score"] = round(score, 4)
                obs["constitutional_results"] = results
                self._save()
                return True
        return False

    def observation_count(self) -> int:
        """Total observations (pending + scored)."""
        return len(self._state["observations"])

    def scored_count(self) -> int:
        """Observations with engagement data."""
        return sum(1 for o in self._state["observations"] if o.get("status") in _SCORED_STATUSES)

    # ------------------------------------------------------------------
    # -------------------------------------------------------------
    # Retired 2026-04-11: RuanMei landscape generator.
    # `generate_content_strategy`, `_assemble_strategy_inputs`,
    # `_build_strategy_retrospective`, `_STRATEGY_TOOL`,
    # `_finalize_landscape`, `_analyst_findings_hash`, and the
    # associated tool-use agent (system prompt, dispatch, schemas)
    # all produced content_landscape.json — a prose strategy document
    # that Stelle no longer reads. Deleted as a Bitter Lesson
    # violation. RuanMei now owns observation state + reward
    # computation only.
    # -------------------------------------------------------------

    def _compute_reward(self, metrics: dict, icp_score: Optional[float] = None) -> dict:
        """
        Z-scored composite reward with learned weights.

        Two decomposition paths run in parallel:
        1. Engineered: depth/reach/eng_rate (hand-designed components with learned weights)
        2. Raw: per-metric z-scores (impressions/reactions/comments/reposts z-scored directly)

        The raw z-scores are stored alongside the engineered components for
        downstream analysis. When enough data exists (>= 30 obs), the system
        can compare which decomposition better predicts future engagement and
        shift weight accordingly.
        """
        impressions = max(metrics.get("impressions", 0), 1)
        reactions = metrics.get("reactions", 0)
        comments = metrics.get("comments", 0)
        reposts = metrics.get("reposts", 0)

        # Engineered components (existing path).
        dw = self._get_depth_weights()
        depth = (comments * dw["comments"] + reposts * dw["reposts"] + reactions * dw["reactions"]) / impressions
        reach = math.log1p(impressions)
        eng_rate = (reactions + comments + reposts) / impressions

        # Z-score against client history.
        history = self._reward_history()
        raw_z = {}
        if len(history) >= 3:
            z_depth = _z_score(depth, [h["depth"] for h in history])
            z_reach = _z_score(reach, [h["reach"] for h in history])
            z_eng = _z_score(eng_rate, [h["eng_rate"] for h in history])

            z_components = {"depth": z_depth, "reach": z_reach, "eng_rate": z_eng}

            # Raw metric z-scores (no hand-designed decomposition)
            raw_z = {
                "z_impressions": _z_score(
                    math.log1p(impressions),
                    [math.log1p(h.get("impressions", 0)) for h in history if h.get("impressions") is not None],
                ),
                "z_reactions": _z_score(
                    reactions / impressions,
                    [h.get("reactions_rate", 0) for h in history if h.get("reactions_rate") is not None],
                ),
                "z_comments": _z_score(
                    comments / impressions,
                    [h.get("comments_rate", 0) for h in history if h.get("comments_rate") is not None],
                ),
                "z_reposts": _z_score(
                    reposts / impressions,
                    [h.get("reposts_rate", 0) for h in history if h.get("reposts_rate") is not None],
                ),
            }

            if icp_score is not None:
                icp_vals = [h["icp_reward"] for h in history if h.get("icp_reward") is not None]
                if len(icp_vals) >= 3:
                    z_components["icp"] = _z_score(icp_score, icp_vals)
                else:
                    z_components["icp"] = icp_score

            # Learned weights or equal weights
            weights = self._get_reward_weights()
            immediate = sum(z_components[k] * weights.get(k, 1/len(z_components))
                           for k in z_components)
        else:
            # Not enough history for z-scoring; use raw composite (equal weight).
            components = [depth, reach / max(reach, 1), eng_rate]
            if icp_score is not None:
                components.append(icp_score)
            immediate = sum(components) / len(components)

        result = {
            "immediate": round(immediate, 4),
            "depth": round(depth, 6),
            "reach": round(reach, 4),
            "eng_rate": round(eng_rate, 6),
            "icp_reward": round(icp_score, 4) if icp_score is not None else None,
            "raw_metrics": {
                "impressions": impressions,
                "reactions": reactions,
                "comments": comments,
                "reposts": reposts,
            },
        }
        if raw_z:
            result["raw_z_scores"] = {k: round(v, 4) for k, v in raw_z.items()}
        return result

    # ------------------------------------------------------------------
    # Learned depth weights
    # ------------------------------------------------------------------

    _DEFAULT_DEPTH_WEIGHTS = {"comments": 3.0, "reposts": 2.0, "reactions": 1.0}

    def _data_confidence(self) -> float:
        """Data-driven confidence in the current observation pool.

        Returns 1 - 1/sqrt(n) for n >= 3, else 0.0.  This is the ratio of
        the standard error of the mean to the standard deviation — a pure
        consequence of the Central Limit Theorem with no free parameters.

        At n=3  → 0.42    At n=10 → 0.68    At n=30 → 0.82
        At n=50 → 0.86    At n=100 → 0.90   Approaches 1.0 asymptotically
        """
        scored = [
            o for o in self._state.get("observations", [])
            if o.get("status") in _SCORED_STATUSES
        ]
        n = len(scored)
        if n < 3:
            return 0.0
        return 1.0 - 1.0 / math.sqrt(n)

    @staticmethod
    def _spearman_pvalue(rho: float, n: int) -> float:
        """Two-tailed p-value for a Spearman correlation via t-approximation.

        Uses the standard transformation t = rho * sqrt((n-2)/(1-rho^2))
        followed by a normal approximation of the Student-t CDF.
        No hand-tuned parameters — the p-value is the data's own
        answer to 'how confident should I be in this correlation?'
        """
        if n < 3:
            return 1.0
        if abs(rho) >= 1.0:
            return 0.0
        t_stat = abs(rho) * math.sqrt((n - 2) / (1.0 - rho * rho))
        df = n - 2
        z = t_stat * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + t_stat ** 2 / (2.0 * df))
        p_one_tail = 0.5 * math.erfc(z / math.sqrt(2.0))
        return min(1.0, 2.0 * p_one_tail)

    def _get_depth_weights(self) -> dict[str, float]:
        """Return p-value-blended depth component weights.

        Each component's weight is blended between its learned Spearman
        correlation weight and the prior default, proportional to the
        statistical significance of that correlation.  No hard gates.
        """
        cache_path = P.memory_dir(self.company) / "depth_weights.json"
        learned = None
        pvalues = None
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                learned = cached.get("weights")
                pvalues = cached.get("pvalues")
            except Exception:
                pass

        if learned is None:
            scored = [o for o in self._state.get("observations", []) if o.get("status") in _SCORED_STATUSES]
            if len(scored) >= 3:
                learned, pvalues = self._compute_and_cache_depth_weights(scored, cache_path)
            else:
                return dict(self._DEFAULT_DEPTH_WEIGHTS)

        if pvalues is None:
            return learned

        default = self._DEFAULT_DEPTH_WEIGHTS
        blended = {}
        for k in default:
            d = default[k]
            l_val = learned.get(k, d)
            p = pvalues.get(k, 1.0)
            confidence = max(0.0, 1.0 - p)
            blended[k] = round(confidence * l_val + (1 - confidence) * d, 4)
        return blended

    def recompute_depth_weights(self) -> dict[str, float]:
        """Recompute and cache depth weights. Call during ordinal_sync after scoring."""
        scored = [o for o in self._state.get("observations", []) if o.get("status") in _SCORED_STATUSES]
        if len(scored) < 3:
            return dict(self._DEFAULT_DEPTH_WEIGHTS)
        cache_path = P.memory_dir(self.company) / "depth_weights.json"
        weights, _ = self._compute_and_cache_depth_weights(scored, cache_path)
        return weights

    def _compute_and_cache_depth_weights(
        self, scored: list[dict], cache_path
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute depth weights via Spearman correlation with composite reward.

        Returns (weights, pvalues) — the p-value for each component's
        correlation is the data's own answer to 'how much should I trust
        this weight?'
        """
        components = {"comments": [], "reposts": [], "reactions": []}
        rewards = []

        for o in scored:
            raw = o.get("reward", {}).get("raw_metrics", {})
            imp = max(raw.get("impressions", 0), 1)
            r = o.get("reward", {}).get("immediate")
            if r is None:
                continue
            components["comments"].append(raw.get("comments", 0) / imp)
            components["reposts"].append(raw.get("reposts", 0) / imp)
            components["reactions"].append(raw.get("reactions", 0) / imp)
            rewards.append(r)

        if len(rewards) < 3:
            return dict(self._DEFAULT_DEPTH_WEIGHTS), {k: 1.0 for k in self._DEFAULT_DEPTH_WEIGHTS}

        raw_corrs = {}
        pvalues = {}
        for name, vals in components.items():
            rank_v = _rank_values(vals)
            rank_r = _rank_values(rewards)
            n = len(vals)
            mean_rv = sum(rank_v) / n
            mean_rr = sum(rank_r) / n
            num = sum((rv - mean_rv) * (rr - mean_rr) for rv, rr in zip(rank_v, rank_r))
            den = (sum((rv - mean_rv) ** 2 for rv in rank_v) * sum((rr - mean_rr) ** 2 for rr in rank_r)) ** 0.5
            rho = num / den if den > 0 else 0.0
            raw_corrs[name] = rho
            pvalues[name] = self._spearman_pvalue(rho, n)

        clamped = {k: max(0.0, v) for k, v in raw_corrs.items()}
        total = sum(clamped.values())
        if total == 0:
            weights = dict(self._DEFAULT_DEPTH_WEIGHTS)
        else:
            n_components = len(clamped)
            weights = {k: round(v / total * n_components, 4) for k, v in clamped.items()}

        try:
            cache_data = {
                "weights": weights,
                "pvalues": {k: round(v, 6) for k, v in pvalues.items()},
                "raw_correlations": {k: round(v, 4) for k, v in raw_corrs.items()},
                "observation_count": len(rewards),
                "computed_at": _now(),
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
            tmp.rename(cache_path)
            logger.info(
                "[RuanMei] Depth weights for %s: %s (from %d obs)",
                self.company, weights, len(rewards),
            )
        except Exception:
            pass

        return weights, pvalues

    def _get_reward_weights(self) -> dict[str, float]:
        """Learn per-client reward component weights from lagged engagement.

        Each component's Spearman p-value controls its blend between the
        learned weight and the equal-weight default.  No hard sample-size gate.
        """
        default_w = {"depth": 0.25, "reach": 0.25, "eng_rate": 0.25, "icp": 0.25}

        cache_path = P.memory_dir(self.company) / "reward_weights.json"
        current_obs_count = len(self._reward_history())
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("observation_count", 0) >= current_obs_count:
                    w = cached.get("blended_weights") or cached.get("weights")
                    if w:
                        return w
            except Exception:
                pass

        history = self._reward_history()
        if len(history) < 3:
            return dict(default_w)

        components = {"depth": [], "reach": [], "eng_rate": []}
        next_rewards = []
        for i in range(len(history) - 1):
            h = history[i]
            components["depth"].append(h.get("depth", 0))
            components["reach"].append(h.get("reach", 0))
            components["eng_rate"].append(h.get("eng_rate", 0))
            h_next = history[i + 1]
            next_r = h_next.get("depth", 0) + h_next.get("reach", 0) + h_next.get("eng_rate", 0)
            next_rewards.append(next_r)

        icp_vals = [h.get("icp_reward") for h in history[:-1] if h.get("icp_reward") is not None]
        if len(icp_vals) == len(next_rewards) and len(icp_vals) >= 3:
            components["icp"] = icp_vals

        raw_weights = {}
        pvalues = {}
        for name, vals in components.items():
            if len(vals) != len(next_rewards) or len(vals) < 3:
                raw_weights[name] = 0.25
                pvalues[name] = 1.0
                continue
            n = len(vals)
            rank_v = _rank_values(vals)
            rank_r = _rank_values(next_rewards)
            mean_rv = sum(rank_v) / n
            mean_rr = sum(rank_r) / n
            num = sum((rv - mean_rv) * (rr - mean_rr) for rv, rr in zip(rank_v, rank_r))
            den = (sum((rv - mean_rv) ** 2 for rv in rank_v) * sum((rr - mean_rr) ** 2 for rr in rank_r)) ** 0.5
            rho = num / den if den > 0 else 0.0
            raw_weights[name] = max(rho, 0.0)
            pvalues[name] = self._spearman_pvalue(rho, n)

        total_raw = sum(raw_weights.values())
        if total_raw > 0:
            learned = {k: v / total_raw for k, v in raw_weights.items()}
        else:
            learned = dict(default_w)

        blended = {}
        for k in default_w:
            p = pvalues.get(k, 1.0)
            conf = max(0.0, 1.0 - p)
            blended[k] = round(conf * learned.get(k, 0.25) + (1 - conf) * default_w[k], 4)
        total_blended = sum(blended.values())
        if total_blended > 0:
            blended = {k: round(v / total_blended, 4) for k, v in blended.items()}

        try:
            cache_data = {
                "blended_weights": blended,
                "raw_weights": {k: round(v, 4) for k, v in learned.items()},
                "pvalues": {k: round(v, 6) for k, v in pvalues.items()},
                "observation_count": len(history),
                "computed_at": _now(),
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
            tmp.rename(cache_path)
        except Exception:
            pass

        return blended

    def _reward_history(self) -> list[dict]:
        """Extract historical reward components for z-scoring.

        Includes both engineered (depth/reach/eng_rate) and raw per-metric
        rates so either decomposition path can compute z-scores.
        """
        history = []
        for o in self._state["observations"]:
            r = o.get("reward")
            if r and o.get("status") in _SCORED_STATUSES:
                raw = r.get("raw_metrics", {})
                imp = max(raw.get("impressions", 0), 1)
                entry = {
                    "depth": r.get("depth", 0),
                    "reach": r.get("reach", 0),
                    "eng_rate": r.get("eng_rate", 0),
                    "impressions": raw.get("impressions", 0),
                    "reactions_rate": raw.get("reactions", 0) / imp,
                    "comments_rate": raw.get("comments", 0) / imp,
                    "reposts_rate": raw.get("reposts", 0) / imp,
                }
                icp = r.get("icp_reward") if r.get("icp_reward") is not None else o.get("icp_reward")
                if icp is not None:
                    entry["icp_reward"] = icp
                history.append(entry)
        return history

    # ------------------------------------------------------------------
    # Temporal patterns
    # ------------------------------------------------------------------

    def _compute_temporal_patterns(self, scored: list[dict]) -> str:
        """Compute day-of-week performance patterns. Reports counts so
        Claude can judge meaningfulness from the data itself."""
        if len(scored) < 3:
            return ""

        by_day: dict[str, list[float]] = {}
        for o in scored:
            ts = o.get("posted_at") or o.get("recorded_at", "")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                day = dt.strftime("%A")
                reward = o.get("reward", {}).get("immediate", 0)
                by_day.setdefault(day, []).append(reward)
            except (ValueError, TypeError):
                continue

        # Only report days with >= 2 observations — a single data point
        # per day is not a pattern, it's noise.
        reportable = {d: r for d, r in by_day.items() if len(r) >= 2}
        if not reportable:
            return ""

        lines = []
        for day, rewards in sorted(reportable.items(), key=lambda x: -sum(x[1]) / len(x[1]) if x[1] else 0):
            avg = sum(rewards) / len(rewards)
            lines.append(f"{day}: avg score {avg:.3f} ({len(rewards)} posts)")

        return "\n".join(lines)

    def _compute_streak(self, observations: list[dict]) -> int:
        """Consecutive weeks with 2+ posts."""
        if not observations:
            return 0

        now = datetime.now(timezone.utc)
        weeks: dict[int, int] = {}  # week_number -> post_count

        for o in observations:
            ts = o.get("recorded_at", "")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                week = (now - dt).days // 7
                weeks[week] = weeks.get(week, 0) + 1
            except (ValueError, TypeError):
                continue

        streak = 0
        for week_ago in range(0, max(weeks.keys(), default=0) + 1):
            if weeks.get(week_ago, 0) >= 2:
                streak += 1
            else:
                break

        return streak

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(self.company)
        if state is not None:
            return state
        return {
            "company": self.company,
            "observations": [],
            "created_at": _now(),
        }

    def _save(self) -> None:
        from backend.src.db.local import initialize_db, ruan_mei_save
        initialize_db()
        self._state["last_updated"] = _now()
        ruan_mei_save(self.company, self._state)

    # ------------------------------------------------------------------
    # Observation compaction
    # ------------------------------------------------------------------

    _MAX_OBSERVATIONS = 500
    _RECENT_KEEP_DAYS = 90
    _COMPACTION_SAMPLE_RATE = 0.3

    def compact_observations(self) -> int:
        """Prune old observations to bound storage and prompt budget.

        Retention policy:
        - All observations from the last 90 days are kept unconditionally.
        - Older observations: keep top and bottom quartile (most informative
          for insight generation) plus a random 30% sample of the middle.
        - Summarize dropped observations into an aggregate stats dict stored
          in _state["compaction_summary"].

        Returns the number of observations removed. Safe to call repeatedly
        (idempotent when under the cap).
        """
        obs = self._state.get("observations", [])
        if len(obs) <= self._MAX_OBSERVATIONS:
            return 0

        import random as _rng
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (self._RECENT_KEEP_DAYS * 86400)

        recent = []
        old = []
        for o in obs:
            ts = o.get("posted_at") or o.get("recorded_at", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.timestamp() >= cutoff:
                    recent.append(o)
                else:
                    old.append(o)
            except (ValueError, TypeError):
                recent.append(o)

        if not old:
            return 0

        # Keep top/bottom quartile + random sample of middle
        scored_old = [o for o in old if o.get("status") in _SCORED_STATUSES]
        pending_old = [o for o in old if o.get("status") not in _SCORED_STATUSES]

        kept_old = list(pending_old)  # keep all pending (awaiting data)

        if scored_old:
            scored_old.sort(key=lambda o: o.get("reward", {}).get("immediate", 0))
            n = len(scored_old)
            q1 = max(1, n // 4)
            q3 = max(q1 + 1, n - n // 4)
            bottom = scored_old[:q1]
            middle = scored_old[q1:q3]
            top = scored_old[q3:]

            sample_size = max(1, int(len(middle) * self._COMPACTION_SAMPLE_RATE))
            middle_sample = _rng.sample(middle, min(sample_size, len(middle)))

            kept_old.extend(bottom)
            kept_old.extend(middle_sample)
            kept_old.extend(top)

            dropped_count = len(scored_old) - (len(bottom) + len(middle_sample) + len(top))
        else:
            dropped_count = 0

        # Build compaction summary for dropped observations
        dropped = [o for o in scored_old if o not in bottom and o not in top and o not in middle_sample] if scored_old else []
        if dropped:
            rewards = [o.get("reward", {}).get("immediate", 0) for o in dropped]
            summary = self._state.get("compaction_summary", {})
            summary["last_compacted_at"] = _now()
            summary["total_dropped"] = summary.get("total_dropped", 0) + len(dropped)
            summary["dropped_reward_mean"] = round(sum(rewards) / len(rewards), 4)
            summary["dropped_reward_range"] = [round(min(rewards), 4), round(max(rewards), 4)]
            self._state["compaction_summary"] = summary

        new_obs = recent + kept_old
        removed = len(obs) - len(new_obs)
        self._state["observations"] = new_obs
        if removed > 0:
            self._save()
            logger.info(
                "[RuanMei] Compacted %d observations for %s (kept %d recent + %d old)",
                removed, self.company, len(recent), len(kept_old),
            )
        return removed


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _sanitize_for_cross_client(analysis: str, source_company: str, all_companies: set[str]) -> str:
    """Strip client-identifying content from analysis text for cross-client aggregation."""
    import re
    result = analysis
    for company in all_companies:
        variants = {company, company.replace("-", " "), company.replace("-", ""),
                    company.title(), company.replace("-", " ").title()}
        for v in variants:
            if len(v) >= 4:
                result = re.sub(re.escape(v), '[company]', result, flags=re.IGNORECASE)
    result = re.sub(r'https?://\S+', '[url]', result)
    result = re.sub(r'\S+@\S+\.\S+', '[email]', result)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_metrics_snapshot(obs: dict, metrics: dict) -> None:
    """Append a timestamped engagement snapshot to the observation's history.

    Each sync cycle calls this before overwriting obs["reward"]. Over time,
    the history shows how engagement evolved: early velocity (first hours),
    growth curve (days 1-3), and plateau (week+). This data enables future
    analysis of which content types have fast initial velocity (favored by
    LinkedIn's algorithm) vs slow burns.

    After appending (if a new snapshot was added), recomputes the derived
    trajectory metrics via `analyze_trajectory` and stores them on
    `obs["trajectory"]`. Pure measurement — no reward formula change, no
    interpretation. Downstream consumers (analyst, Irontomb Phase 3) can
    read the trajectory dict freely.

    Capped at 50 snapshots per observation to bound storage (~5KB each).
    """
    if "metrics_history" not in obs:
        obs["metrics_history"] = []

    snapshot = {
        "t": _now(),
        "impressions": metrics.get("impressions", 0),
        "reactions": metrics.get("reactions", 0),
        "comments": metrics.get("comments", 0),
        "reposts": metrics.get("reposts", 0),
    }

    # Deduplicate: skip if the metrics are identical to the last snapshot
    # (no new engagement since last sync — don't store a redundant point)
    history = obs["metrics_history"]
    if history:
        last = history[-1]
        if (last.get("impressions") == snapshot["impressions"]
                and last.get("reactions") == snapshot["reactions"]
                and last.get("comments") == snapshot["comments"]):
            return

    history.append(snapshot)

    # Cap at 50 snapshots (50 × ~100 bytes = ~5KB per observation)
    if len(history) > 50:
        obs["metrics_history"] = history[-50:]

    # Recompute derived trajectory metrics from the updated history
    try:
        obs["trajectory"] = analyze_trajectory(obs)
    except Exception:
        # Trajectory analysis must never kill the update path
        pass


def analyze_trajectory(obs: dict) -> dict:
    """Compute derived trajectory metrics from an observation's metrics_history.

    Pure measurement function, no interpretation. Returns a dict with:

      n_snapshots              : how many snapshots are in metrics_history
      span_hours               : time span from first snapshot to last
      final_impressions        : latest impressions count
      final_reactions          : latest reactions count
      velocity_first_1h        : impressions gained in first hour (interpolated)
      velocity_first_6h        : impressions gained in first 6 hours
      velocity_first_24h       : impressions gained in first 24 hours
      velocity_first_72h       : impressions gained in first 72 hours
      peak_velocity_imp_per_h  : max segment velocity across the trajectory
      time_to_plateau_hours    : hours from posting until segment velocity
                                 drops below 10% of peak (None if never)
      post_plateau_velocity_imp_per_h : mean velocity in post-plateau segments
      longevity_ratio          : final_impressions / impressions_at_24h;
                                 ratio > 1 means the post kept growing past day 1
      insufficient_data        : True when we don't have enough snapshots

    No hand-tuned thresholds for reward — this function only measures. Any
    interpretation of what these numbers mean is the consumer's job.
    """
    history = obs.get("metrics_history") or []
    posted_at_str = (obs.get("posted_at") or "").strip()

    result: dict[str, Any] = {"n_snapshots": len(history)}

    if len(history) < 2 or not posted_at_str:
        result["insufficient_data"] = True
        return result

    try:
        posted_at = datetime.fromisoformat(posted_at_str.replace("Z", "+00:00"))
    except Exception:
        result["insufficient_data"] = True
        return result

    # Parse snapshots into (hours_from_posted, counts) tuples
    parsed: list[dict] = []
    for snap in history:
        t_str = snap.get("t") or ""
        if not t_str:
            continue
        try:
            t = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
            dh = (t - posted_at).total_seconds() / 3600.0
        except Exception:
            continue
        parsed.append({
            "h": dh,
            "imp": snap.get("impressions", 0) or 0,
            "react": snap.get("reactions", 0) or 0,
            "comment": snap.get("comments", 0) or 0,
            "repost": snap.get("reposts", 0) or 0,
        })

    if len(parsed) < 2:
        result["insufficient_data"] = True
        return result

    parsed.sort(key=lambda s: s["h"])
    first = parsed[0]
    last = parsed[-1]

    result["span_hours"] = round(last["h"] - first["h"], 2)
    result["final_impressions"] = last["imp"]
    result["final_reactions"] = last["react"]
    result["first_snapshot_hours_after_posting"] = round(first["h"], 2)

    # Interpolated impressions at a given hour-from-posted
    def _imp_at_hour(h: float) -> float:
        if h <= parsed[0]["h"]:
            return float(parsed[0]["imp"])
        if h >= parsed[-1]["h"]:
            return float(parsed[-1]["imp"])
        for i in range(len(parsed) - 1):
            a, b = parsed[i], parsed[i + 1]
            if a["h"] <= h <= b["h"]:
                if b["h"] == a["h"]:
                    return float(a["imp"])
                frac = (h - a["h"]) / (b["h"] - a["h"])
                return a["imp"] + frac * (b["imp"] - a["imp"])
        return float(last["imp"])

    baseline_imp = _imp_at_hour(0)  # impressions at posting time (usually 0)

    for window in (1, 6, 24, 72):
        key = f"velocity_first_{window}h"
        if last["h"] >= window:
            result[key] = round(_imp_at_hour(window) - baseline_imp, 1)
        else:
            result[key] = None  # trajectory hasn't reached this window yet

    # Segment velocities (impressions per hour between consecutive snapshots)
    segments: list[dict] = []
    for i in range(len(parsed) - 1):
        a, b = parsed[i], parsed[i + 1]
        dh = b["h"] - a["h"]
        if dh <= 0:
            continue
        segments.append({
            "start_h": a["h"],
            "end_h": b["h"],
            "velocity_imp_per_h": (b["imp"] - a["imp"]) / dh,
        })

    if not segments:
        result["insufficient_data"] = True
        return result

    peak_v = max((s["velocity_imp_per_h"] for s in segments), default=0.0)
    result["peak_velocity_imp_per_h"] = round(peak_v, 1)

    # Plateau detection: first segment whose velocity drops below 10% of peak
    # 10% is the conventional plateau threshold — it's scale-invariant because
    # it's derived from the data's own peak, not a hand-tuned absolute number
    plateau_hour: Optional[float] = None
    if peak_v > 0:
        threshold = peak_v * 0.1
        for s in segments:
            if s["velocity_imp_per_h"] < threshold:
                plateau_hour = s["start_h"]
                break
    result["time_to_plateau_hours"] = round(plateau_hour, 1) if plateau_hour is not None else None

    # Post-plateau velocity: mean of segment velocities after the plateau point
    if plateau_hour is not None:
        post_plateau = [s for s in segments if s["start_h"] >= plateau_hour]
        if post_plateau:
            result["post_plateau_velocity_imp_per_h"] = round(
                sum(s["velocity_imp_per_h"] for s in post_plateau) / len(post_plateau),
                1,
            )

    # Longevity ratio: how much more impressions accumulated after the 24h mark
    if last["h"] >= 24:
        imp_at_24 = _imp_at_hour(24)
        if imp_at_24 > 0:
            result["longevity_ratio"] = round(last["imp"] / imp_at_24, 3)

    return result


def _compute_edit_similarity(draft: str, live: str) -> float:
    """Return 0..1 similarity ratio between draft and published copy.

    Returns -1 if either side is missing/empty (cannot compute).
    """
    d = draft.strip()
    l = live.strip()
    if not d or not l:
        return -1.0
    return SequenceMatcher(None, d, l).ratio()


def _find_matching_draft(
    draft_map: dict,
    ordinal_pid: str,
    live_text: str,
) -> tuple[str, str]:
    """Resolve the Stelle draft that corresponds to a published post.

    Exact ordinal_post_id lookup only. Ordinal post ids are stable from
    push time through analytics/publication, so the id stored in draft_map
    at push time will equal the ordinalPost.id surfaced in the analytics
    feed after publication. If the lookup misses, the post is externally
    authored (not from Stelle) and we do not try to guess otherwise —
    guessing via fuzzy text similarity bakes in arbitrary similarity
    thresholds that violate the Bitter Lesson.

    Returns (draft_text, matched_key). Empty strings if no match.

    live_text is accepted for signature compatibility with earlier call
    sites but is unused.
    """
    pid = (ordinal_pid or "").strip()
    if pid and pid in draft_map:
        entry = draft_map[pid]
        if isinstance(entry, dict):
            draft = (entry.get("original_text") or "").strip()
            if draft:
                return draft, pid
    return "", ""


def _rank_values(values: list[float]) -> list[float]:
    """Compute ranks with average tie-breaking.

    Delegates to correlation_analyzer._rank (single implementation).
    """
    from backend.src.utils.correlation_analyzer import _rank
    return _rank(values)


def _z_score(value: float, history: list[float]) -> float:
    """Z-score a value against a list of historical values."""
    if len(history) < 2:
        return 0.0
    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / len(history)
    std = math.sqrt(variance) if variance > 0 else 1.0
    return (value - mean) / std
