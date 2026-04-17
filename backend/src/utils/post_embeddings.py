"""Post embedding cache — continuous vector representations of every scored post.

Every observation gets an embedding of its post body. These embeddings
are the PRIMARY feature representation for the entire learning pipeline:

  - Topic transitions → embedding trajectory model (continuous directions)
  - Causal filter → PCA components replace discrete tag categories
  - Draft scorer → k-NN similarity, no categorical features
  - Content brief → embedding clusters, not tag aggregation
  - Sequential state → embedding similarity, not tag string equality

Tags (format_tag, topic_tag) are display-only metadata for human dashboards.
No learning subsystem should depend on them.

Also provides PCA decomposition and k-means clustering utilities used by
the causal filter, content brief, and embedding trajectory modules.

Storage: memory/{company}/post_embeddings.json
Embedding model: OpenAI text-embedding-3-small (1536 dims)

Usage:
    from backend.src.utils.post_embeddings import (
        get_post_embeddings,
        embed_text,
        embed_texts,
        cosine_similarity,
        compute_pca,
        project_to_pca,
        cluster_embeddings,
    )

    embeddings = get_post_embeddings("example-client")
    pca_state = compute_pca(embeddings, n_components=10)
    projections = pca_state["projections"]  # {hash: [10 floats]}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from backend.src.db import vortex

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536


def get_post_embeddings(company: str) -> dict[str, list[float]]:
    """Load the post embedding cache, backfilling any missing scored observations.

    Lazy: only embeds posts not already in the cache. First call for a client
    with 34 unembedded observations makes one batched OpenAI call (~$0.001).
    Subsequent calls return instantly from cache.

    Returns {post_hash: embedding_vector} for all scored observations.
    """
    cache = _load_cache(company)

    # Load scored observations
    try:
        from backend.src.db.local import initialize_db, ruan_mei_load
        initialize_db()
        state = ruan_mei_load(company)
    except Exception:
        return cache
    if state is None:
        return cache

    scored = [
        o for o in state.get("observations", [])
        if o.get("status") in ("scored", "finalized")
        and o.get("post_hash")
        and (o.get("posted_body") or o.get("post_body"))
    ]

    # Find observations not yet embedded
    missing = [o for o in scored if o["post_hash"] not in cache]
    if not missing:
        return cache

    # Batch embed
    texts = [
        (o.get("posted_body") or o.get("post_body") or "")[:8000]
        for o in missing
    ]
    hashes = [o["post_hash"] for o in missing]

    embeddings = _embed_batch(texts)
    if embeddings and len(embeddings) == len(hashes):
        for h, e in zip(hashes, embeddings):
            cache[h] = e
        _save_cache(company, cache)
        logger.info(
            "[post_embeddings] Embedded %d new posts for %s (cache total: %d)",
            len(missing), company, len(cache),
        )

    return cache


def embed_text(text: str) -> Optional[list[float]]:
    """Embed a single text (e.g., a new draft) for similarity comparison.

    Returns None on failure. The caller should handle this gracefully.
    """
    if not text or len(text.strip()) < 10:
        return None
    results = _embed_batch([text[:8000]])
    return results[0] if results else None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts via OpenAI.

    Truncates each text to 32k chars (~8k tokens) for safety.
    Returns [] on failure — callers must handle the empty case.
    """
    return _embed_batch([t[:32000] for t in texts])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors (numpy-accelerated)."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if a_arr.shape != b_arr.shape or a_arr.size == 0:
        return 0.0
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar(
    target_embedding: list[float],
    embeddings: dict[str, list[float]],
    top_k: int = 5,
    exclude_hashes: Optional[set] = None,
) -> list[tuple[str, float]]:
    """Find the top_k most similar post hashes by cosine similarity.

    Returns [(post_hash, similarity), ...] sorted by similarity descending.
    """
    exclude = exclude_hashes or set()
    scored = []
    for h, emb in embeddings.items():
        if h in exclude:
            continue
        sim = cosine_similarity(target_embedding, emb)
        scored.append((h, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ------------------------------------------------------------------
# Cache I/O
# ------------------------------------------------------------------

def _cache_path(company: str):
    return vortex.memory_dir(company) / "post_embeddings.json"


def _load_cache(company: str) -> dict[str, list[float]]:
    path = _cache_path(company)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("embeddings", {})
    except Exception:
        return {}


def _save_cache(company: str, embeddings: dict[str, list[float]]) -> None:
    path = _cache_path(company)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": _EMBEDDING_MODEL,
        "dim": _EMBEDDING_DIM,
        "count": len(embeddings),
        "embeddings": embeddings,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed via OpenAI."""
    if not texts:
        return []
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(
            input=texts,
            model=_EMBEDDING_MODEL,
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        logger.warning("[post_embeddings] Embedding failed: %s", e)
        return []


# ------------------------------------------------------------------
# PCA — principal component decomposition of embedding space
# ------------------------------------------------------------------

@dataclass
class PCAState:
    """Compact PCA result for downstream consumers."""
    components: np.ndarray          # (n_components, embedding_dim) — principal axes
    mean: np.ndarray                # (embedding_dim,) — centering vector
    explained_variance: np.ndarray  # (n_components,) — variance per component
    explained_ratio: np.ndarray     # (n_components,) — fraction of total variance
    projections: dict[str, list[float]]  # {post_hash: [n_components floats]}
    hashes: list[str] = field(default_factory=list)


def compute_pca(
    embeddings: dict[str, list[float]],
    n_components: int = 10,
) -> Optional[PCAState]:
    """PCA decomposition of post embeddings.

    For n << p (typical: 30-50 posts, 1536 dims), uses the dual trick:
    eigendecompose the n×n Gram matrix instead of the p×p covariance matrix.

    Returns PCAState with components, projections per hash, and variance
    explained. Returns None if fewer than 3 embeddings are available.
    """
    if len(embeddings) < 3:
        return None

    hashes = list(embeddings.keys())
    X = np.array([embeddings[h] for h in hashes], dtype=np.float64)
    n, p = X.shape
    n_components = min(n_components, n - 1, p)

    mean = X.mean(axis=0)
    Xc = X - mean  # centered

    # Dual PCA: eigendecompose Xc @ Xc.T (n×n) instead of Xc.T @ Xc (p×p)
    gram = Xc @ Xc.T  # n×n
    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    # eigh returns ascending order; flip to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]

    # Convert Gram eigenvectors to actual principal components (p-dim)
    # V = Xc.T @ U / sqrt(eigenvalues)
    components = np.zeros((n_components, p), dtype=np.float64)
    for i in range(n_components):
        if eigenvalues[i] > 1e-10:
            components[i] = Xc.T @ eigenvectors[:, i]
            components[i] /= np.linalg.norm(components[i])

    total_var = np.sum(np.maximum(eigenvalues, 0)) + np.sum(
        np.maximum(np.linalg.eigh(gram)[0], 0)
    ) - np.sum(np.maximum(eigenvalues, 0))
    # Simpler: total variance = trace of Gram
    total_var = max(np.trace(gram), 1e-10)
    explained_ratio = np.maximum(eigenvalues, 0) / total_var

    # Project all observations onto the components
    projections_matrix = Xc @ components.T  # n × n_components
    projections = {
        h: proj.tolist()
        for h, proj in zip(hashes, projections_matrix)
    }

    return PCAState(
        components=components,
        mean=mean,
        explained_variance=np.maximum(eigenvalues, 0),
        explained_ratio=explained_ratio,
        projections=projections,
        hashes=hashes,
    )


def project_to_pca(
    embedding: list[float],
    pca: PCAState,
) -> list[float]:
    """Project a single embedding onto an existing PCA basis."""
    e = np.array(embedding, dtype=np.float64) - pca.mean
    return (e @ pca.components.T).tolist()


# ------------------------------------------------------------------
# Clustering — k-means in embedding space
# ------------------------------------------------------------------

@dataclass
class ClusterResult:
    """k-means clustering result."""
    k: int
    centroids: np.ndarray          # (k, embedding_dim)
    labels: dict[str, int]         # {post_hash: cluster_id}
    cluster_sizes: dict[int, int]  # {cluster_id: count}


def cluster_embeddings(
    embeddings: dict[str, list[float]],
    k: int = 5,
    max_iter: int = 50,
    seed: int = 42,
) -> Optional[ClusterResult]:
    """Simple k-means clustering over post embeddings.

    Uses cosine distance (L2-normalized embeddings → Euclidean k-means
    on the unit sphere ≈ spherical k-means).

    Returns None if fewer than k embeddings available.
    """
    if len(embeddings) < k:
        return None

    hashes = list(embeddings.keys())
    X = np.array([embeddings[h] for h in hashes], dtype=np.float64)

    # L2-normalize for cosine-based clustering
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    Xn = X / norms

    rng = np.random.RandomState(seed)

    # k-means++ initialization
    n = len(Xn)
    centroid_indices = [rng.randint(n)]
    for _ in range(1, k):
        dists = np.min(
            [np.sum((Xn - Xn[ci]) ** 2, axis=1) for ci in centroid_indices],
            axis=0,
        )
        probs = dists / (dists.sum() + 1e-10)
        centroid_indices.append(rng.choice(n, p=probs))

    centroids = Xn[centroid_indices].copy()

    for _ in range(max_iter):
        # Assign
        dists = np.array([
            np.sum((Xn - c) ** 2, axis=1) for c in centroids
        ]).T  # n × k
        assignments = np.argmin(dists, axis=1)

        # Update
        new_centroids = np.zeros_like(centroids)
        for ci in range(k):
            mask = assignments == ci
            if mask.sum() > 0:
                new_centroids[ci] = Xn[mask].mean(axis=0)
                norm = np.linalg.norm(new_centroids[ci])
                if norm > 1e-10:
                    new_centroids[ci] /= norm
            else:
                new_centroids[ci] = centroids[ci]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    labels = {h: int(assignments[i]) for i, h in enumerate(hashes)}
    cluster_sizes = {}
    for ci in range(k):
        cluster_sizes[ci] = int(np.sum(assignments == ci))

    return ClusterResult(
        k=k,
        centroids=centroids,
        labels=labels,
        cluster_sizes=cluster_sizes,
    )
