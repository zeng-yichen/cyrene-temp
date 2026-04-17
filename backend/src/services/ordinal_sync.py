"""Background Ordinal sync — RuanMei engagement data via Ordinal analytics only.

Does not write to Supabase; other modules may read from Supabase where needed.
"""

import csv
import logging
import threading
import time

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_SYNC_INTERVAL = 3600           # 1 hour default (full pipeline)
_FAST_SYNC_INTERVAL = 900       # 15 min (engagement snapshots for recent posts only)
_FRESH_POST_WINDOW_HOURS = 72   # posts within this window get the fast-sync treatment

_sync_thread: threading.Thread | None = None
_fast_sync_thread: threading.Thread | None = None
_stop_event = threading.Event()
_fast_stop_event = threading.Event()

# Module-level mutex to prevent fast sync and slow sync from racing on
# the same client's ruan_mei state. Slow sync always blocks for the lock;
# fast sync skips the cycle if the lock is held (non-blocking acquire).
_SYNC_MUTEX = threading.Lock()

# Set to None to process ALL clients listed in ordinal_auth_rows.csv.
# When set to a specific set, only listed clients run through per-client
# steps. Market intelligence (step 10) always reads all client data.
#
# 2026-04-11: active set is the 6 existing clients plus the two
# some profiles (multi-user — same workspace, different. The CSV slug is authoritative:
# example-client and commenda-eb4e93-logan, not commenda-eb4e93.
_ACTIVE_CLIENT_ALLOWLIST: set[str] | None = {
    "example-client",
    "example-client",
    "example-client",
    "example-client",
    "example-client",
    "example-client",
    "example-client",
    "example-client",
    "example-client",
}


def start_sync_loop(interval: int = _SYNC_INTERVAL) -> None:
    """Start the background Ordinal sync loop (full pipeline, hourly)."""
    global _sync_thread
    if _sync_thread and _sync_thread.is_alive():
        logger.info("Ordinal sync already running")
        return

    _stop_event.clear()

    def _loop():
        while not _stop_event.is_set():
            try:
                with _SYNC_MUTEX:
                    sync_all_companies()
                _push_memory_to_fly()
            except Exception:
                logger.exception("Ordinal sync cycle failed")
            _stop_event.wait(interval)

    _sync_thread = threading.Thread(target=_loop, daemon=True, name="ordinal-sync")
    _sync_thread.start()
    logger.info("Ordinal sync started (interval=%ds)", interval)


def start_fast_sync_loop(interval: int = _FAST_SYNC_INTERVAL) -> None:
    """Start the fast sync loop (engagement snapshots for recent posts, every 15 min).

    Only fetches Ordinal analytics and updates metrics_history for posts
    whose `publishedAt` is within the last _FRESH_POST_WINDOW_HOURS (72h).
    Does NOT run the full pipeline (ICP scoring, embeddings, topic velocity,
    content strategy, series engine, etc.). Cheap and safe.

    Skips the cycle entirely if the slow sync is mid-run (non-blocking
    mutex acquire). This means a fresh post in the fast-sync window still
    gets captured by the slow sync when it runs, and the dedup in
    _append_metrics_snapshot handles any overlap.
    """
    global _fast_sync_thread
    if _fast_sync_thread and _fast_sync_thread.is_alive():
        logger.info("Fast Ordinal sync already running")
        return

    _fast_stop_event.clear()

    def _loop():
        while not _fast_stop_event.is_set():
            try:
                _sync_recent_engagement()
                _push_memory_to_fly()
            except Exception:
                logger.exception("Fast sync cycle failed")
            _fast_stop_event.wait(interval)

    _fast_sync_thread = threading.Thread(target=_loop, daemon=True, name="ordinal-fast-sync")
    _fast_sync_thread.start()
    logger.info("Fast Ordinal sync started (interval=%ds, window=%dh)", interval, _FRESH_POST_WINDOW_HOURS)


def stop_sync_loop() -> None:
    _stop_event.set()
    _fast_stop_event.set()


def _push_memory_to_fly() -> None:
    """Push local memory/ directory to Fly after a sync cycle.

    Runs push-to-fly.sh in the project root. This is a no-op on Fly
    itself (sync loops are disabled there), and fails gracefully if
    the ``fly`` CLI isn't installed or authenticated.
    """
    import pathlib
    import shutil
    import subprocess

    project_root = pathlib.Path(__file__).resolve().parents[3]
    script = project_root / "push-to-fly.sh"

    if not script.exists():
        logger.debug("[push-to-fly] script not found at %s, skipping", script)
        return

    if not shutil.which("fly"):
        logger.debug("[push-to-fly] fly CLI not found on PATH, skipping")
        return

    try:
        result = subprocess.run(
            [str(script)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("[push-to-fly] memory pushed to Fly successfully")
        else:
            logger.warning(
                "[push-to-fly] script exited %d: %s",
                result.returncode,
                (result.stderr or result.stdout or "")[:300],
            )
    except subprocess.TimeoutExpired:
        logger.warning("[push-to-fly] timed out after 120s")
    except Exception:
        logger.exception("[push-to-fly] failed (non-fatal)")


def _sync_recent_engagement() -> None:
    """Fast path: update engagement snapshots for posts published within the last 72h.

    Runs every 15 min. Only does:
      1. Fetch Ordinal analytics for each active client
      2. Filter to posts published within _FRESH_POST_WINDOW_HOURS
      3. Call _update_ruan_mei_from_posts (which appends metrics_history
         + recomputes analyze_trajectory on each updated observation)

    Does NOT run any of the full pipeline steps (ICP scoring, embeddings,
    tagging, prediction accuracy, calibration, series engine, etc.). Those
    remain on the hourly slow sync.

    Non-blocking: if the slow sync holds _SYNC_MUTEX, this cycle is skipped
    entirely (a debug log is emitted and we wait for next cycle).
    """
    acquired = _SYNC_MUTEX.acquire(blocking=False)
    if not acquired:
        logger.debug("[fast_sync] slow sync in progress — skipping this cycle")
        return
    try:
        _sync_recent_engagement_impl()
    finally:
        _SYNC_MUTEX.release()


def _sync_recent_engagement_impl() -> None:
    """Inner implementation for _sync_recent_engagement (runs under lock)."""
    from datetime import datetime, timedelta, timezone

    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(hours=_FRESH_POST_WINDOW_HOURS)

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                api_key = row.get("api_key", "").strip()
                company = row.get("provider_org_slug", "").strip() or row.get("company_id", "").strip()
                if not api_key or not company:
                    continue
                if _ACTIVE_CLIENT_ALLOWLIST is not None and company not in _ACTIVE_CLIENT_ALLOWLIST:
                    continue

                profile_id = row.get("profile_id", "").strip()
                if not profile_id:
                    profile_id = vortex.resolve_profile_id(company)
                if not profile_id:
                    continue

                analytics_posts = _fetch_ordinal_analytics(profile_id, api_key, company)
                if not analytics_posts:
                    continue

                # Filter to fresh posts (published within the window)
                fresh: list[dict] = []
                for post in analytics_posts:
                    pub = post.get("publishedAt") or post.get("postedAt") or post.get("published_at") or ""
                    if not pub:
                        continue
                    try:
                        pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    if pub_dt >= cutoff:
                        fresh.append(post)

                if not fresh:
                    continue

                try:
                    import asyncio
                    from backend.src.agents.ruan_mei import RuanMei
                    rm = RuanMei(company)
                    _update_ruan_mei_from_posts(rm, fresh)
                    logger.debug(
                        "[fast_sync] %s: updated %d fresh posts (<%.0fh old)",
                        company, len(fresh), _FRESH_POST_WINDOW_HOURS,
                    )
                except Exception:
                    logger.debug("[fast_sync] update failed for %s", company, exc_info=True)
    except Exception:
        logger.exception("[fast_sync] CSV read failed")


def sync_all_companies() -> None:
    """Iterate ordinal_auth CSV and sync each company."""
    from backend.src.usage.context import current_user_email, current_client_slug

    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        logger.debug("No ordinal_auth_rows.csv found — skipping sync")
        return

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                api_key = row.get("api_key", "").strip()
                company = row.get("provider_org_slug", "").strip() or row.get("company_id", "").strip()
                if api_key and company:
                    # Gate: skip per-client processing for inactive clients
                    if _ACTIVE_CLIENT_ALLOWLIST is not None and company not in _ACTIVE_CLIENT_ALLOWLIST:
                        logger.info("[sync] Skipping %s (not in active allowlist)", company)
                        continue

                    # Set usage attribution so LLM calls during sync are
                    # recorded with the correct client slug and "system"
                    # as the user (no human triggered this).
                    current_user_email.set("system:ordinal_sync")
                    current_client_slug.set(company)

                    # RuanMei: ingest all posts + update pending observations.
                    profile_id = row.get("profile_id", "").strip()
                    if not profile_id:
                        profile_id = vortex.resolve_profile_id(company)
                    if profile_id:
                        analytics_posts = _fetch_ordinal_analytics(profile_id, api_key, company)
                        if analytics_posts:
                            try:
                                import asyncio
                                from backend.src.agents.ruan_mei import RuanMei
                                rm = RuanMei(company)

                                # Snapshot observation count before sync for dirty-flag gating.
                                _pre_sync_scored = rm.scored_count()

                                # 1. Update any pending Stelle-generated observations.
                                _update_ruan_mei_from_posts(rm, analytics_posts)

                                # 2. Ingest all Ordinal posts as scored observations.
                                #    Posts already in the history are skipped (deduped by hash).
                                try:
                                    _loop = asyncio.get_running_loop()
                                except RuntimeError:
                                    _loop = None
                                if _loop and _loop.is_running():
                                    import concurrent.futures
                                    ingested = concurrent.futures.ThreadPoolExecutor().submit(
                                        lambda: asyncio.run(rm.ingest_from_ordinal(analytics_posts))
                                    ).result(timeout=120)
                                else:
                                    ingested = asyncio.run(rm.ingest_from_ordinal(analytics_posts))
                                if ingested:
                                    logger.info("RuanMei ingested %d new posts for %s", ingested, company)

                                # 2-pre. Observation compaction: prune old observations to
                                #        bound storage and prompt budget. Runs before all
                                #        substeps so they operate on a bounded list.
                                try:
                                    _compacted = rm.compact_observations()
                                    if _compacted:
                                        logger.info(
                                            "[ordinal_sync] Compacted %d old observations for %s",
                                            _compacted, company,
                                        )
                                except Exception:
                                    logger.warning("Observation compaction skipped for %s", company, exc_info=True)

                                # 2a. Edit similarity backfill: fix observations where
                                #     post_body was set to live text by the old
                                #     ingest_from_ordinal path. Populates the draft-vs-live
                                #     diff signal RuanMei's editorial analysis depends on.
                                #     Idempotent — skips observations already correct.
                                try:
                                    _backfilled = rm.backfill_edit_similarity_from_draft_map()
                                    if _backfilled:
                                        logger.info(
                                            "[ordinal_sync] Edit similarity backfilled on %d observations for %s",
                                            _backfilled, company,
                                        )
                                except Exception:
                                    logger.warning("Edit similarity backfill skipped for %s", company, exc_info=True)

                                # Dirty-flag: check if new scored observations arrived.
                                # If not, skip expensive recomputation substeps
                                # that would produce identical results.
                                # NOTE: evaluated BEFORE compaction so it isn't
                                # confused by removals — ingested > 0 is the
                                # authoritative signal; the post-compaction count
                                # is a secondary confirmation.
                                _post_sync_scored = rm.scored_count()
                                _has_new_data = ingested > 0 or _post_sync_scored > _pre_sync_scored
                                if not _has_new_data:
                                    logger.debug(
                                        "[ordinal_sync] No new scored data for %s (%d obs), "
                                        "skipping recomputation substeps",
                                        company, _post_sync_scored,
                                    )

                                # 2d. Depth weights: recompute learned depth component
                                #     weights from engagement correlations.
                                if _has_new_data:
                                    try:
                                        rm.recompute_depth_weights()
                                    except Exception:
                                        logger.warning("Depth weight recompute skipped for %s", company, exc_info=True)

                                # 2g. Reward component weights: recompute from lagged
                                #     engagement correlations.
                                if _has_new_data:
                                    try:
                                        rm._get_reward_weights()  # forces recompute if obs grew
                                    except Exception:
                                        logger.warning("Reward weights recompute skipped for %s", company, exc_info=True)

                                # 2j-tag. Retired 2026-04-13.
                                #   Observation tagger produced DISPLAY-ONLY fields
                                #   (topic_tag, source_segment_type, format_tag) via a
                                #   Sonnet call per untagged post every hour. Stelle
                                #   and Cyrene explicitly strip these from their views
                                #   — they encode a hand-designed taxonomy the learning
                                #   pipeline intentionally routes around. The only
                                #   consumers were the progress report and learning
                                #   dashboard; they now degrade gracefully when tags
                                #   are missing. Already-tagged observations keep
                                #   their tags; new posts simply don't get re-tagged.
                                #   observation_tagger.py is preserved for ad-hoc
                                #   backfill if ever needed.

                                # 2k2a. Prediction backfill: score historical posts that
                                #       don't have predicted_engagement yet. This bootstraps
                                #       the validation loop so we get accuracy data from
                                #       existing posts (not just newly generated ones).
                                #       Capped at 10 per cycle to bound embedding cost.
                                if _has_new_data:
                                    try:
                                        from backend.src.utils.draft_scorer import score_drafts as _score_drafts_bf
                                        _pred_backfilled = 0
                                        _pred_limit = 10
                                        for _obs in rm._state.get("observations", []):
                                            if _pred_backfilled >= _pred_limit:
                                                break
                                            if _obs.get("predicted_engagement") is not None:
                                                continue
                                            if _obs.get("status") not in ("scored", "finalized"):
                                                continue
                                            _body = (_obs.get("posted_body") or _obs.get("post_body") or "").strip()
                                            if not _body or len(_body) < 100:
                                                continue
                                            try:
                                                _scores = _score_drafts_bf(company, [{"text": _body}])
                                                if _scores and _scores[0].model_source != "no_model":
                                                    _obs["predicted_engagement"] = _scores[0].predicted_score
                                                    _pred_backfilled += 1
                                            except Exception:
                                                pass
                                        if _pred_backfilled:
                                            rm._save()
                                            logger.info(
                                                "[ordinal_sync] Backfilled predicted_engagement on %d observations for %s",
                                                _pred_backfilled, company,
                                            )
                                    except Exception:
                                        logger.warning("Prediction backfill skipped for %s", company, exc_info=True)

                                # 2k2. Prediction accuracy: compare draft scorer predictions
                                #      against actual engagement outcomes for posts that have
                                #      both predicted_engagement and scored reward.
                                try:
                                    from backend.src.utils.prediction_tracker import update_prediction_accuracy
                                    _pred_acc = update_prediction_accuracy(company)
                                    if _pred_acc and _pred_acc.get("n_predictions", 0) > 0:
                                        logger.info(
                                            "[ordinal_sync] Prediction accuracy for %s: "
                                            "Spearman=%.3f, MAE=%.3f, n=%d (%s)",
                                            company,
                                            _pred_acc.get("spearman", 0),
                                            _pred_acc.get("mean_abs_error", 0),
                                            _pred_acc.get("n_predictions", 0),
                                            _pred_acc.get("trend", "?"),
                                        )
                                except Exception:
                                    logger.warning("Prediction accuracy skipped for %s", company, exc_info=True)

                                # 2k3. Irontomb Phase 3 calibration — join logged simulator
                                #      predictions against real T+7d engagement outcomes by
                                #      draft_hash. Measures whether Irontomb's
                                #      engagement_prediction scalar tracks reality over time.
                                #      Cheap (no LLM calls, pure data join); always safe to
                                #      run. Persists memory/{company}/calibration_report.json
                                #      and logs headline metrics when pairs exist.
                                try:
                                    from backend.src.agents.irontomb import calibration_report
                                    _cal = calibration_report(company)
                                    _n_pairs = _cal.get("n_pairs_joined", 0) or 0
                                    if _n_pairs > 0:
                                        _metrics = _cal.get("metrics") or {}
                                        _spearman = _metrics.get("spearman_engagement_prediction")
                                        _mae = _metrics.get("mean_abs_error_per_1k")
                                        _acc = _metrics.get("binary_accuracy_would_react")
                                        logger.info(
                                            "[ordinal_sync] Irontomb calibration for %s: "
                                            "n_pairs=%d, spearman=%s, mae_per_1k=%s, "
                                            "would_react_accuracy=%s%s",
                                            company,
                                            _n_pairs,
                                            f"{_spearman:.3f}" if isinstance(_spearman, (int, float)) else "N/A",
                                            f"{_mae:.2f}" if isinstance(_mae, (int, float)) else "N/A",
                                            f"{_acc:.2f}" if isinstance(_acc, (int, float)) else "N/A",
                                            " (insufficient_data)" if _cal.get("insufficient_data") else "",
                                        )
                                    elif _cal.get("error"):
                                        logger.debug(
                                            "[ordinal_sync] Irontomb calibration for %s: %s",
                                            company, _cal["error"],
                                        )
                                except Exception:
                                    logger.warning("Irontomb calibration skipped for %s", company, exc_info=True)

                                # 2l-2m-3. Retired 2026-04-11.
                                #   Analyst agent, content brief generator, and ICP
                                #   definition generator all deleted as Bitter Lesson
                                #   violations. They were pre-chewed intermediate
                                #   representations (prose findings, curated topic/format
                                #   allocations, distilled ICP descriptions) sitting
                                #   between raw observations/transcripts and the generating
                                #   agents (Stelle, ICP simulator). Stelle and the ICP
                                #   simulator now query the raw data directly — the same
                                #   tool primitives the analyst used are exposed to Stelle
                                #   as live tools, and ICP resolution reads transcripts on
                                #   demand instead of a frozen JSON snapshot.

                                # 4. ICP scoring — fetch engager profiles for posts that
                                #    don't have an ICP reward yet.
                                _run_icp_scoring(rm, company)

                                # 4b. Backfill per-engager ICP scores for posts scored
                                #     before per-engager persistence was added.
                                try:
                                    _backfill_engager_icp_scores(company)
                                except Exception:
                                    logger.debug("Engager ICP score backfill skipped for %s", company)

                                # 5. Pinecone embedding — keep semantic index fresh.
                                try:
                                    from backend.src.services.linkedin_bank import embed_posts_to_pinecone
                                    embed_posts_to_pinecone(analytics_posts, company=company)
                                except Exception:
                                    logger.debug("Pinecone embedding skipped for %s", company)

                                # 6. Retired 2026-04-13.
                                #   topic_velocity refresh called Perplexity every
                                #   cycle to produce memory/{client}/topic_velocity.md.
                                #   Its only live consumer (Stelle's workspace symlink)
                                #   violated the "transcripts/ + deltas + engagement
                                #   only" input policy, and should_accelerate_topic()
                                #   in temporal_orchestrator was defined-but-never-called.
                                #   Stelle now reaches for trending-topic context
                                #   via the net_query tool on demand if she needs it.
                                #   topic_velocity.py preserved on disk for ad-hoc use.

                                # 7. Content strategy generation retired (2026-04-10).
                                #    Under the stripped architecture, RuanMei's
                                #    generate_content_strategy is no longer called — its
                                #    landscape output was prescriptive intelligence that
                                #    hurt Stelle's output at Opus 4.6 capability. The
                                #    underlying RuanMei class still manages scored
                                #    observation state, reward computation, and embeddings
                                #    but no longer synthesizes landscapes. See NEXT_STEPS.md.

                                # 8. Series Engine: update series post scores from RuanMei,
                                #    then check series health for wrap/extend signals.
                                try:
                                    from backend.src.services.series_engine import (
                                        update_series_from_ruan_mei as _series_update,
                                        check_series_health as _series_health,
                                    )
                                    _series_scored = _series_update(company)
                                    if _series_scored:
                                        logger.info(
                                            "[ordinal_sync] Series engine updated %d posts for %s",
                                            _series_scored, company,
                                        )
                                    _series_changes = _series_health(company)
                                    for sc in _series_changes:
                                        logger.info(
                                            "[ordinal_sync] Series '%s' %s → %s (trend: %s) for %s",
                                            sc["theme"], sc["old_status"], sc["new_status"],
                                            sc["trend"], company,
                                        )
                                except Exception:
                                    logger.warning("Series engine update skipped for %s", company, exc_info=True)
                            except Exception:
                                logger.exception("RuanMei sync failed for %s", company)
    except Exception:
        logger.exception("Failed to read ordinal auth CSV")

    # 10. Retired 2026-04-13.
    #   run_market_intel_cycle was a weekly burst of Claude + OpenAI calls
    #   that produced vertical creator signals with zero live consumers —
    #   its only importer was the deprecated lola.py module. Preserved
    #   on disk for future revival if we ever wire a live consumer.


def _fetch_ordinal_analytics(profile_id: str, api_key: str, company: str) -> list[dict] | None:
    """Fetch post analytics from Ordinal once. Returns parsed post list, or None on failure."""
    import httpx
    from datetime import datetime, timedelta, timezone

    base = "https://app.tryordinal.com/api/v1"
    start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        resp = httpx.get(
            f"{base}/analytics/linkedin/{profile_id}/posts",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            params={"startDate": start, "endDate": end},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Ordinal analytics fetch failed for %s: %s", company, e)
        return None

    posts = data if isinstance(data, list) else data.get("posts", data.get("data", []))
    if not posts:
        logger.info("No analytics posts returned from Ordinal for %s", company)
        return []

    logger.info("Fetched %d analytics posts from Ordinal for %s", len(posts), company)
    return posts


def fetch_ordinal_posts_for(company: str) -> list[dict] | None:
    """Public helper: fetch Ordinal analytics posts for a single company.

    Used by Cyrene's `query_ordinal_posts` tool to see the client's full
    publishing history — including posts they wrote themselves that
    never went through Stelle. Returns None if credentials or profile id
    can't be resolved, otherwise the raw analytics list.
    """
    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                slug = (
                    row.get("provider_org_slug", "").strip()
                    or row.get("company_id", "").strip()
                )
                if slug != company:
                    continue
                api_key = row.get("api_key", "").strip()
                if not api_key:
                    return None
                profile_id = row.get("profile_id", "").strip() or vortex.resolve_profile_id(company)
                if not profile_id:
                    return None
                return _fetch_ordinal_analytics(profile_id, api_key, company)
    except Exception:
        logger.exception("[ordinal_sync] fetch_ordinal_posts_for failed for %s", company)
        return None
    return None


def _extract_ordinal_post_id(post: dict) -> str:
    """Extract the Ordinal workspace post id from a LinkedIn analytics payload.

    The documented field is ``ordinalPost.id`` — present when the post was
    published through Ordinal, null otherwise.
    """
    op = post.get("ordinalPost")
    if isinstance(op, dict):
        v = op.get("id")
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _extract_linkedin_url(post: dict) -> str:
    """Extract the live LinkedIn post URL from an Ordinal analytics payload."""
    return (
        post.get("url")
        or post.get("linkedInUrl")
        or post.get("linkedin_url")
        or post.get("postUrl")
        or ""
    )


def _update_ruan_mei_from_posts(rm, analytics_posts: list[dict]) -> None:
    """Push pre-fetched Ordinal analytics into RuanMei for pending observations.

    Prefer matching by Ordinal workspace post id (set at push time) so copy can change on LinkedIn.
    Fall back to hashing live post text for older observations without a linked id.
    """
    updated = 0

    for post in analytics_posts:
        text = (
            post.get("commentary") or post.get("text") or post.get("copy")
            or post.get("content") or post.get("post_text") or ""
        ).strip()
        if not text:
            continue

        impressions = post.get("impressionCount") or post.get("impressions") or 0
        reactions = post.get("likeCount") or post.get("reactions") or post.get("total_reactions") or 0
        comments = post.get("commentCount") or post.get("comments") or post.get("total_comments") or 0
        reposts = post.get("shareCount") or post.get("repostCount") or post.get("reposts") or 0

        if impressions == 0:
            continue

        posted_at = post.get("publishedAt") or post.get("postedAt") or post.get("published_at") or ""
        linkedin_url = _extract_linkedin_url(post)

        metrics = {
            "impressions": impressions,
            "reactions": reactions,
            "comments": comments,
            "reposts": reposts,
            "posted_at": posted_at,
        }

        oid = _extract_ordinal_post_id(post)
        matched = False
        if oid:
            matched = rm.update_by_ordinal_post_id(
                oid, metrics, posted_body=text,
                linkedin_post_url=linkedin_url, _defer_save=True,
            )
        if not matched:
            matched = rm.update_by_text(
                text, metrics,
                ordinal_post_id=oid,
                linkedin_post_url=linkedin_url,
                _defer_save=True,
            )
        if matched:
            updated += 1

    if updated:
        rm._save()
        logger.info("[RuanMei] Updated %d observations for %s", updated, rm.company)


# Minimum reactions for ICP scoring to be worthwhile.
# Posts with fewer engagers produce noisy scores (sample too small).
# 47 of 262 posts (18%) have <10 reactions — skipping saves ~18% of Apify cost
# No reaction-count floor: every finalized post gets reactor identities
# captured and ICP-scored, regardless of how few reactions it received.
# Rationale: posts with 0-4 reactions carry the most learning signal about
# what flopped and who (if anyone) still engaged. Skipping them silently
# drops the bottom quartile from the audience signal, which is exactly the
# data we need to correct bad patterns. Apify cost at this volume is trivial
# compared to the learning value.
_MIN_REACTIONS_FOR_ICP = 0

# ICP scoring is now gated by _ACTIVE_CLIENT_ALLOWLIST (per-client loop gate).
# No separate allowlist needed.


def _run_icp_scoring(rm, company: str) -> None:
    """For every scored observation that has a LinkedIn URL but no ICP reward, fetch engagers and score.

    This runs after the main sync so engagement data is always current first.
    Only processes posts where engagers haven't been fetched yet (idempotent).

    Cost optimisations (total: ~87% reduction vs naive approach):
      - Skip posts with <10 reactions (noisy sample, not worth fetching)
      - Reactions only, no comments actor (reactors are 83% of engagers)
      - Capped at 30 results per post (diminishing returns past ~20)
    """
    try:
        from backend.src.services.engager_fetcher import fetch_and_persist
        from backend.src.utils.icp_scorer import score_engagers, score_engagers_segmented
    except ImportError as e:
        logger.warning("[ordinal_sync] ICP scoring imports failed: %s", e)
        return

    scored_obs = [
        o for o in rm._state.get("observations", [])
        if o.get("status") in ("scored", "finalized")
        and o.get("linkedin_post_url")
        and (o.get("ordinal_post_id") or o.get("linkedin_post_url"))
        and o.get("reward", {}).get("icp_reward") is None
    ]
    # For posts without an ordinal_post_id (ingested before ID linking),
    # use a hash of the linkedin_post_url as the key for engager storage.
    for obs in scored_obs:
        if not obs.get("ordinal_post_id"):
            import hashlib
            # Include company in the hash to prevent cross-client collisions
            # when two clients share the same LinkedIn URL (e.g., example-client
            # and example-client both have the same post in their analytics).
            url = obs["linkedin_post_url"]
            compound = f"{company}:{url}"
            obs["_icp_scoring_key"] = "url:" + hashlib.sha256(compound.encode()).hexdigest()[:16]
        else:
            obs["_icp_scoring_key"] = obs["ordinal_post_id"]

    if not scored_obs:
        return

    # Filter out low-engagement posts — not enough engagers for meaningful ICP signal.
    scoreable = []
    skipped_low = 0
    for obs in scored_obs:
        raw = obs.get("reward", {}).get("raw_metrics", {})
        reactions = raw.get("reactions", 0)
        if reactions < _MIN_REACTIONS_FOR_ICP:
            skipped_low += 1
            continue
        scoreable.append(obs)

    if skipped_low:
        logger.info(
            "[ordinal_sync] Skipped %d posts with <%d reactions for ICP scoring (%s)",
            skipped_low, _MIN_REACTIONS_FOR_ICP, company,
        )

    if not scoreable:
        return

    logger.info("[ordinal_sync] Running ICP scoring for %d posts (%s)", len(scoreable), company)

    # Auth-failure heuristic: trip only if a HIGH-REACTION post returns empty.
    # With the reaction floor dropped to 0, a post with 0-4 reactions often
    # legitimately returns an empty reactor list, so empty-on-low-reaction is
    # not a signal of auth failure. We only treat "empty reactors on a post
    # with >=10 reported reactions" as suspicious — that's when Apify should
    # definitely find someone.
    suspicious_empty = 0
    MAX_SUSPICIOUS_EMPTY = 3

    for obs in scoreable:
        # Use ordinal_post_id when available; fall back to a URL-derived key
        # for older posts ingested before the ID linking existed.
        key = obs.get("_icp_scoring_key") or obs.get("ordinal_post_id") or ""
        oid = obs.get("ordinal_post_id") or key
        url = obs["linkedin_post_url"]
        reactions_reported = obs.get("reward", {}).get("raw_metrics", {}).get("reactions", 0)
        try:
            engager_profiles = fetch_and_persist(company, key, url)
            if engager_profiles is None:
                continue
            if not engager_profiles:
                if reactions_reported >= 10:
                    suspicious_empty += 1
                    if suspicious_empty >= MAX_SUSPICIOUS_EMPTY:
                        logger.warning(
                            "[ordinal_sync] %d consecutive empty engager fetches "
                            "on high-reaction posts for %s — likely auth issue, "
                            "skipping remaining ICP scoring",
                            suspicious_empty, company,
                        )
                        break
                continue
            suspicious_empty = 0
            segmented = score_engagers_segmented(company, engager_profiles)
            if not segmented.get("scores"):
                continue

            # Persist per-engager ICP scores back to SQLite
            per_engager_scores = segmented["scores"]
            if len(per_engager_scores) == len(engager_profiles):
                from backend.src.db.local import update_engager_icp_scores
                score_pairs = [
                    (p.get("urn", ""), s)
                    for p, s in zip(engager_profiles, per_engager_scores)
                    if p.get("urn")
                ]
                if score_pairs:
                    update_engager_icp_scores(key, score_pairs)

            icp_score = segmented["score"]
            legacy_score = round(icp_score * 2 - 1, 4)
            rm.update_icp_reward(
                oid, legacy_score,
                linkedin_post_url=url,
                icp_match_rate=segmented.get("icp_match_rate"),
            )
        except Exception:
            logger.exception("[ordinal_sync] ICP scoring failed for post %s (%s)", key[:12], company)


def _backfill_engager_icp_scores(company: str) -> int:
    """Re-score engagers for posts where per-engager ICP scores are missing.

    Runs once per post until all posts have per-engager scores. Idempotent —
    skips posts that already have scores. Caps at 5 posts per sync cycle to
    limit LLM cost.
    """
    from backend.src.db.local import (
        get_unscored_engager_post_ids, get_engagers_for_post, update_engager_icp_scores,
    )
    from backend.src.utils.icp_scorer import score_engagers_segmented

    unscored_posts = get_unscored_engager_post_ids(company)
    if not unscored_posts:
        return 0

    _MAX_BACKFILL_PER_CYCLE = 5
    backfilled = 0

    for oid in unscored_posts[:_MAX_BACKFILL_PER_CYCLE]:
        rows = get_engagers_for_post(oid)
        profiles = [
            {"urn": r.get("engager_urn", ""),
             "headline": r.get("headline", ""), "name": r.get("name", ""),
             "current_company": r.get("current_company", ""),
             "title": r.get("title", ""), "location": r.get("location", "")}
            for r in rows
            if r.get("headline") or r.get("current_company") or r.get("title")
        ]
        if not profiles:
            continue

        try:
            segmented = score_engagers_segmented(company, profiles)
            scores = segmented.get("scores", [])
            if len(scores) == len(profiles):
                score_pairs = [
                    (p["urn"], s) for p, s in zip(profiles, scores) if p["urn"]
                ]
                if score_pairs:
                    update_engager_icp_scores(oid, score_pairs)
                    backfilled += 1
        except Exception:
            logger.debug("[ordinal_sync] Engager ICP backfill failed for post %s", oid[:12])

    if backfilled:
        logger.info(
            "[ordinal_sync] Backfilled per-engager ICP scores for %d/%d posts (%s)",
            backfilled, len(unscored_posts), company,
        )
    return backfilled
