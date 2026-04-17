"""Cross-Client Learning Network — transfer learning across all Amphoreus clients.

Two active components:
1. **Universal pattern extraction** — structured patterns from top performers
   across 2+ clients, stored in memory/our_memory/universal_patterns.json.
2. **Cross-client hook library** — top-quartile hooks with metadata, stored
   in memory/our_memory/hook_library.json.

(LOLA arm seeding is deprecated — cold-start content intelligence is now
handled by RuanMei.recommend_context().)

Wired into ordinal_sync as step 9 (after series health check).

Usage:
    from backend.src.services.cross_client_learning import (
        refresh_universal_patterns,
        refresh_hook_library,
        load_hook_library_for_stelle,
    )

    # In ordinal_sync (runs hourly):
    refresh_universal_patterns()
    refresh_hook_library()

    # Stelle context:
    hooks = load_hook_library_for_stelle(company="biotech-co", limit=10)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic()


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

def _our_memory() -> Path:
    return P.our_memory_dir()


def _patterns_path() -> Path:
    return _our_memory() / "universal_patterns.json"


def _hook_library_path() -> Path:
    return _our_memory() / "hook_library.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_analysis(analysis: str, source_company: str, all_companies: set[str]) -> str:
    """Remove client-identifying content from analysis text for cross-client use.

    Replaces:
    - Source company name and slug variants with "[company]" (case-insensitive)
    - Other known company names with "[company]"

    This prevents client-specific strategies, product names, and identifiers
    from leaking into the universal pattern prompt.
    """
    import re

    result = analysis

    for company in all_companies:
        slug_variants = {company, company.replace("-", " "), company.replace("-", "")}
        slug_variants.add(company.title())
        slug_variants.add(company.replace("-", " ").title())
        for variant in slug_variants:
            if len(variant) >= 4:
                result = re.sub(re.escape(variant), '[company]', result, flags=re.IGNORECASE)

    result = re.sub(r'https?://\S+', '[url]', result)
    result = re.sub(r'\S+@\S+\.\S+', '[email]', result)

    return result


# ------------------------------------------------------------------
# 1. Universal pattern extraction
# ------------------------------------------------------------------

def _load_all_scored() -> dict[str, list[dict]]:
    """Load scored observations for every client. Returns {company: [obs]}."""
    result: dict[str, list[dict]] = {}
    if not P.MEMORY_ROOT.exists():
        return result
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
    except Exception:
        return result
    for d in P.MEMORY_ROOT.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name == "our_memory":
            continue
        company = d.name
        try:
            state = ruan_mei_load(company)
        except Exception:
            state = None
        if not state:
            continue
        scored = [o for o in state.get("observations", []) if o.get("status") in ("scored", "finalized")]
        if len(scored) >= 5:
            result[company] = scored
    return result


def _top_quartile(scored: list[dict]) -> list[dict]:
    """Return top 25% by immediate reward."""
    scored.sort(key=lambda o: o.get("reward", {}).get("immediate", 0))
    cutoff = int(len(scored) * 0.75)
    return scored[cutoff:]


def refresh_universal_patterns() -> list[dict]:
    """Scan all clients, extract universal patterns, persist to JSON.

    A pattern is "universal" if it appears in top-25% performers across 2+ clients.
    Uses Claude Haiku to cluster and deduplicate patterns from post analyses.

    Returns the list of patterns written.
    """
    all_scored = _load_all_scored()
    if len(all_scored) < 2:
        return []

    # Collect top analyses with client tag (anonymized as client_1, client_2, etc).
    # Analysis text is sanitized to strip client-identifying content.
    top_entries: list[dict] = []
    client_map: dict[str, str] = {}
    company_names = set(all_scored.keys())
    for i, (company, scored) in enumerate(sorted(all_scored.items())):
        client_map[company] = f"client_{i+1}"
        top = _top_quartile(scored)
        for o in top:
            analysis = o.get("descriptor", {}).get("analysis", "")
            if not analysis:
                continue
            analysis = _sanitize_analysis(analysis, company, company_names)
            reward = o.get("reward", {}).get("immediate", 0)
            impressions = o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0)
            icp_rate = o.get("icp_match_rate")
            top_entries.append({
                "client": client_map[company],
                "analysis": analysis[:300],
                "reward": round(reward, 3),
                "impressions": impressions,
                "icp_match_rate": round(icp_rate, 2) if icp_rate is not None else None,
            })

    if len(top_entries) < 10:
        return []

    # Sample to keep prompt manageable
    top_entries.sort(key=lambda e: e["reward"], reverse=True)
    sample_size = min(60, max(20, len(top_entries)))  # scale with data, cap at 60
    sample = top_entries[:sample_size]

    parts = []
    for e in sample:
        header = f"[{e['client']} | reward={e['reward']:.3f} | impressions={e['impressions']}"
        if e.get("icp_match_rate") is not None:
            header += f" | icp_rate={e['icp_match_rate']}"
        header += "]"
        parts.append(f"{header}\n{e['analysis']}")
    entries_text = "\n\n".join(parts)

    prompt = (
        f"You are analyzing {len(top_entries)} top-performing LinkedIn posts across "
        f"{len(all_scored)} anonymous B2B clients.\n\n"
        "Extract UNIVERSAL PATTERNS — writing mechanics that appear in top performers "
        "across MULTIPLE clients (not client-specific topics).\n\n"
        f"TOP PERFORMERS:\n{entries_text}\n\n"
        "Return a JSON array of patterns. Each pattern:\n"
        "{\n"
        '  "pattern": "Posts that open with a specific number or metric consistently outperform",\n'
        '  "evidence_clients": 4,\n'
        '  "avg_reward_lift": 0.35,\n'
        '  "confidence": 0.85,\n'
        '  "category": "hook|structure|storytelling|specificity|format|engagement_driver"\n'
        "}\n\n"
        "Extract 5-10 patterns. Only include patterns supported by 2+ clients. "
        "Be specific about the writing mechanic, not vague ('good hooks' is useless; "
        "'opening with a concrete number before the thesis' is useful). "
        "Output ONLY the JSON array."
    )

    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        patterns = json.loads(raw)
        if not isinstance(patterns, list):
            patterns = []
    except Exception as e:
        logger.warning("[cross_client] Pattern extraction failed: %s", e)
        return []

    # Validate and normalize
    valid: list[dict] = []
    for p in patterns:
        if not p.get("pattern"):
            continue
        valid.append({
            "pattern": p["pattern"],
            "evidence_clients": max(2, int(p.get("evidence_clients", 2))),
            "avg_reward_lift": round(float(p.get("avg_reward_lift", 0)), 3),
            "confidence": round(max(0, min(1, float(p.get("confidence", 0.5)))), 2),
            "category": p.get("category", "general"),
            "updated_at": _now(),
            "source_observations": len(top_entries),
            "source_clients": len(all_scored),
        })

    # Persist
    path = _patterns_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(valid, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "[cross_client] Extracted %d universal patterns from %d clients (%d observations)",
        len(valid), len(all_scored), len(top_entries),
    )
    return valid


def load_universal_patterns() -> list[dict]:
    """Load persisted universal patterns."""
    path = _patterns_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


# ------------------------------------------------------------------
# 2. Cross-client hook library
# ------------------------------------------------------------------

def refresh_hook_library() -> list[dict]:
    """Extract hooks from top-quartile posts, persist with metadata.

    Hook = first 140 chars of posted_body or post_body (the "above the fold" text).
    """
    all_scored = _load_all_scored()
    if not all_scored:
        return []

    hooks: list[dict] = []
    seen_hooks: set[str] = set()

    for company, scored in all_scored.items():
        top = _top_quartile(scored)
        for o in top:
            body = (o.get("posted_body") or o.get("post_body") or "").strip()
            if not body or len(body) < 50:
                continue

            # Extract hook: first line or first 140 chars
            first_line = body.split("\n")[0].strip()
            hook = first_line[:140] if len(first_line) > 10 else body[:140]

            # Deduplicate
            hook_key = hook[:80].lower()
            if hook_key in seen_hooks:
                continue
            seen_hooks.add(hook_key)

            analysis = o.get("descriptor", {}).get("analysis", "")
            reward = o.get("reward", {}).get("immediate", 0)
            impressions = o.get("reward", {}).get("raw_metrics", {}).get("impressions", 0)
            icp_rate = o.get("icp_match_rate")

            # Infer hook style from analysis
            hook_style = _classify_hook_style(hook, analysis)

            hooks.append({
                "hook": hook,
                "hook_style": hook_style,
                "engagement_score": round(reward, 3),
                "impressions": impressions,
                "icp_match_rate": round(icp_rate, 2) if icp_rate is not None else None,
                "char_count": len(body),
                "company_anonymized": False,  # we keep company for internal use
                "company": company,
            })

    # Sort by engagement
    hooks.sort(key=lambda h: h["engagement_score"], reverse=True)

    # Scale hook library with data, cap at 400
    hook_cap = min(400, max(100, len(hooks)))
    hooks = hooks[:hook_cap]

    path = _hook_library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hooks, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("[cross_client] Hook library: %d hooks from %d clients", len(hooks), len(all_scored))
    return hooks


def _classify_hook_style(hook: str, analysis: str) -> str:
    """Classify hook style from text heuristics. No LLM call — must be fast."""
    h = hook.lower()

    # Number-led
    for char in h:
        if char.isdigit():
            return "number_led"
        if char.isalpha():
            break

    if any(h.startswith(w) for w in ("i ", "i'm ", "i've ", "my ", "when i ")):
        return "personal_story"

    if "?" in hook[:80]:
        return "question"

    if any(w in h[:60] for w in ("most ", "everyone ", "nobody ", "the biggest ", "the real ")):
        return "contrarian"

    if any(w in h[:60] for w in ("ceo", "cto", "vp ", "director", "founder", "engineer")):
        return "icp_callout"

    # Check analysis for clues
    a = analysis.lower()
    if "narrative" in a or "story" in a or "anecdot" in a:
        return "story_climax"
    if "specifi" in a and "number" in a:
        return "number_led"

    return "declarative"


def load_hook_library(limit: int = 50, hook_style: str | None = None) -> list[dict]:
    """Load hook library, optionally filtered by style."""
    path = _hook_library_path()
    if not path.exists():
        return []
    try:
        hooks = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if hook_style:
        hooks = [h for h in hooks if h.get("hook_style") == hook_style]

    return hooks[:limit]


def load_hook_library_for_stelle(company: str = "", limit: int = 15) -> str:
    """Build a Stelle-ready context string from the hook library.

    Two modes:
    - If the client has enough scored posts: retrieve hooks by embedding
      similarity to their reward-weighted content direction.
    - Fallback: return top hooks by engagement score (globally best).

    Excludes same-client hooks to avoid self-plagiarism.
    """
    hooks = load_hook_library(limit=limit * 4)
    if not hooks:
        return ""

    # Exclude same-client hooks
    if company:
        hooks = [h for h in hooks if h.get("company") != company]

    if not hooks:
        return ""

    # Try embedding-based retrieval: find hooks similar to client's content direction
    selected = _retrieve_hooks_by_embedding(company, hooks, limit)
    if not selected:
        # Fallback: top by engagement
        selected = hooks[:limit]

    lines = ["\n\nHOOK REFERENCE LIBRARY (top-performing hooks, relevance-ranked):"]
    for h in selected[:10]:
        score = h.get("engagement_score", 0)
        lines.append(f'  (score {score:.2f}) "{h["hook"]}"')

    lines.append(
        "\nStudy these hooks for scroll-stop patterns. Do NOT copy them — "
        "apply the structural patterns to this client's material."
    )
    return "\n".join(lines)


def _retrieve_hooks_by_embedding(company: str, hooks: list[dict], limit: int) -> list[dict]:
    """Retrieve hooks most relevant to the client's content direction.

    Computes a reward-weighted centroid from post embeddings and scores
    hooks by cosine similarity to that direction.
    Falls back to empty list if embeddings unavailable.
    """
    try:
        from backend.src.utils.post_embeddings import get_post_embeddings, embed_texts
        from backend.src.agents.ruan_mei import RuanMei
        import numpy as np
    except ImportError:
        return []

    embeddings = get_post_embeddings(company)
    if len(embeddings) < 5:
        return []

    rm = RuanMei(company)
    scored = [
        o for o in rm._state.get("observations", [])
        if o.get("status") in ("scored", "finalized") and o.get("post_hash") in embeddings
    ]
    if len(scored) < 5:
        return []

    emb_list = [embeddings[o["post_hash"]] for o in scored]
    rewards = [max(o.get("reward", {}).get("immediate", 0), 0) for o in scored]

    emb_matrix = np.array(emb_list, dtype=np.float32)
    reward_weights = np.array(rewards, dtype=np.float32) + 1e-8
    centroid = np.average(emb_matrix, axis=0, weights=reward_weights)

    hook_texts = [h["hook"] for h in hooks]

    # Cache hook embeddings alongside hook_library.json
    hook_embs = None
    cache_path = _our_memory() / "hook_embeddings_cache.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_hooks = cached.get("hooks", [])
            if cached_hooks == hook_texts and cached.get("embeddings"):
                hook_embs = cached["embeddings"]
        except Exception:
            pass

    if hook_embs is None:
        hook_embs = embed_texts(hook_texts)
        if hook_embs and len(hook_embs) == len(hooks):
            try:
                cache_path.write_text(json.dumps({
                    "hooks": hook_texts,
                    "embeddings": [e if isinstance(e, list) else e.tolist() for e in hook_embs],
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                }), encoding="utf-8")
            except Exception:
                pass

    if not hook_embs or len(hook_embs) != len(hooks):
        return []

    hook_matrix = np.array(hook_embs, dtype=np.float32)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
    hook_norms = hook_matrix / (np.linalg.norm(hook_matrix, axis=1, keepdims=True) + 1e-8)
    similarities = hook_norms @ centroid_norm

    eng_scores = np.array([h.get("engagement_score", 0) for h in hooks], dtype=np.float32)
    eng_norm = eng_scores / (np.max(np.abs(eng_scores)) + 1e-8)
    final_scores = 0.6 * similarities + 0.4 * eng_norm

    ranked_indices = np.argsort(final_scores)[::-1][:limit]
    return [hooks[i] for i in ranked_indices]


# ------------------------------------------------------------------
# 3. Cold-start LOLA seeding (DEPRECATED)
# ------------------------------------------------------------------
# LOLA arm seeding is no longer used. Cold-start content intelligence
# is now handled by RuanMei.recommend_context() which falls back to
# cross-client insights when a client has < 10 scored observations.

def auto_seed_lola(company: str) -> int:
    """DEPRECATED: LOLA arm seeding is no longer used.

    Kept as a no-op stub so existing callers don't crash.
    """
    logger.debug("[cross_client] auto_seed_lola called for %s — DEPRECATED, returning 0", company)
    return 0


# ------------------------------------------------------------------
# ordinal_sync integration — single entry point for step 9
# ------------------------------------------------------------------

def run_cross_client_sync() -> dict:
    """Run all cross-client learning tasks. Called by ordinal_sync as step 9.

    Returns summary dict.
    """
    result = {
        "patterns": 0,
        "hooks": 0,
        "seeded_clients": [],
    }

    # 1. Refresh universal patterns
    try:
        patterns = refresh_universal_patterns()
        result["patterns"] = len(patterns)
    except Exception as e:
        logger.warning("[cross_client] Pattern refresh failed: %s", e)

    # 2. Refresh hook library
    try:
        hooks = refresh_hook_library()
        result["hooks"] = len(hooks)
    except Exception as e:
        logger.warning("[cross_client] Hook library refresh failed: %s", e)

    # 3. (LOLA auto-seeding removed — content intelligence now handled
    #    by RuanMei.recommend_context() which uses cross-client insights
    #    as a cold-start fallback instead of discrete arms.)

    return result
