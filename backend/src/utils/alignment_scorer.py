"""360Brew alignment scorer — pre-publish semantic consistency check.

Returns the raw cosine similarity between a draft and a cached client
identity fingerprint (accepted posts, content strategy, LinkedIn profile,
ICP definition). No hand-tuned thresholds, no bucketed labels. The
consumer (currently Cyrene's critic) reads the continuous score and
decides what it means.

The previous version of this module (a) classified the similarity into
{strong, moderate, drift} buckets using hand-tuned 0.75 / 0.60 cutoffs,
(b) generated canned summary sentences per bucket, and (c) maintained an
``AlignmentAdaptiveConfig`` class that walked the score range searching
for "best" thresholds. All of that was removed: bucketing a continuous
signal into categorical labels and then feeding the labels to a model
bakes the operator's judgment into the learning signal. Raw scores only.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CACHE_TTL = 86400  # 24h fingerprint embedding cache
_EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_embedding(text: str, model: str = _EMBEDDING_MODEL) -> Optional[list[float]]:
    """Get an embedding vector from OpenAI."""
    try:
        from openai import OpenAI
        client = OpenAI()
        truncated = text[:32000]
        resp = client.embeddings.create(input=truncated, model=model)
        return resp.data[0].embedding
    except Exception as e:
        logger.warning("[alignment_scorer] Embedding failed: %s", e)
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Client fingerprint construction (authoritative texts → embedding)
# ---------------------------------------------------------------------------

def _load_fingerprint_texts(company: str) -> str:
    """Load identity-signal texts for the client's fingerprint."""
    from backend.src.db import vortex as P

    parts: list[str] = []

    accepted = P.accepted_dir(company)
    if accepted.exists():
        for f in sorted(accepted.iterdir()):
            if f.suffix in (".txt", ".md") and f.stat().st_size < 10_000:
                try:
                    parts.append(f.read_text(encoding="utf-8").strip())
                except Exception:
                    pass

    profile_path = P.memory_dir(company) / "profile.json"
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            headline = profile.get("headline", "")
            about = profile.get("about", "")
            if headline:
                parts.append(f"LinkedIn Headline: {headline}")
            if about:
                parts.append(f"LinkedIn About: {about}")
        except Exception:
            pass

    icp_path = P.icp_definition_path(company)
    if icp_path.exists():
        try:
            icp = json.loads(icp_path.read_text(encoding="utf-8"))
            desc = icp.get("description", "")
            if desc:
                parts.append(f"Target ICP: {desc}")
        except Exception:
            pass

    return "\n\n---\n\n".join(p for p in parts if p)


def _fingerprint_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _get_or_build_fingerprint(company: str) -> Optional[list[float]]:
    """Get cached fingerprint embedding or build a new one."""
    from backend.src.db.local import cache_get, cache_set

    fingerprint_text = _load_fingerprint_texts(company)
    if not fingerprint_text:
        return None

    text_hash = _fingerprint_hash(fingerprint_text)
    cache_key = f"fingerprint_embedding:{company}:{text_hash}"

    cached = cache_get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass

    embedding = _get_embedding(fingerprint_text)
    if embedding:
        cache_set(cache_key, json.dumps(embedding), ttl_seconds=_CACHE_TTL)
        logger.info(
            "[alignment_scorer] Built fingerprint embedding for %s (%d chars)",
            company, len(fingerprint_text),
        )
    return embedding


# ---------------------------------------------------------------------------
# Public scoring API — raw cosine similarity only
# ---------------------------------------------------------------------------

def score_draft_alignment(company: str, draft_text: str) -> dict:
    """Return the raw alignment score between a draft and the client's fingerprint.

    Output schema::

        {
            "score": float,    # cosine similarity in [0, 1], or None if unavailable
            "method": str,     # "embedding" | "skip"
        }

    No label, no summary, no topic decomposition. The consumer reads the
    continuous score and decides what it means. If the fingerprint cannot
    be built (missing accepted posts + profile + ICP) or the draft
    embedding fails, returns ``{"score": None, "method": "skip"}``.
    """
    if not draft_text or not draft_text.strip():
        return {"score": None, "method": "skip"}

    fingerprint = _get_or_build_fingerprint(company)
    if fingerprint is None:
        return {"score": None, "method": "skip"}

    draft_embedding = _get_embedding(draft_text)
    if draft_embedding is None:
        return {"score": None, "method": "skip"}

    similarity = _cosine_similarity(fingerprint, draft_embedding)
    return {"score": round(similarity, 4), "method": "embedding"}


def score_batch(company: str, drafts: list[str]) -> list[dict]:
    """Score multiple drafts efficiently (reuses fingerprint)."""
    fingerprint = _get_or_build_fingerprint(company)
    if fingerprint is None:
        return [{"score": None, "method": "skip"} for _ in drafts]

    results: list[dict] = []
    for draft in drafts:
        if not draft.strip():
            results.append({"score": None, "method": "skip"})
            continue
        draft_emb = _get_embedding(draft)
        if draft_emb is None:
            results.append({"score": None, "method": "skip"})
            continue
        sim = _cosine_similarity(fingerprint, draft_emb)
        results.append({"score": round(sim, 4), "method": "embedding"})
    return results
