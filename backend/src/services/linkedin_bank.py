"""LinkedIn data ingestion — per-client fetch of posts/profiles.

Per-client LinkedIn post ingestion
rather than at-scale discovery.
"""

import logging
from typing import Any

from backend.src.core.config import get_settings

logger = logging.getLogger(__name__)


def ingest_client_posts(
    linkedin_url: str,
    max_posts: int = 50,
) -> list[dict[str, Any]]:
    """Fetch a client's recent LinkedIn posts and store in Supabase + Pinecone."""
    username = linkedin_url.rstrip("/").split("/")[-1]
    if not username:
        return []

    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()

        result = (
            sb.table("linkedin_posts")
            .select("post_text, posted_at, total_reactions, total_comments, engagement_score, hook")
            .eq("creator_username", username)
            .order("engagement_score", desc=True)
            .limit(max_posts)
            .execute()
        )

        posts = result.data or []
        logger.info("Fetched %d posts for %s from Supabase", len(posts), username)
        return posts

    except Exception as e:
        logger.warning("LinkedIn post fetch failed for %s: %s", username, e)
        return []


def ingest_client_profile(linkedin_url: str) -> dict[str, Any] | None:
    """Fetch a client's LinkedIn profile."""
    username = linkedin_url.rstrip("/").split("/")[-1]
    if not username:
        return None

    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()

        result = (
            sb.table("linkedin_profiles")
            .select("*")
            .eq("username", username)
            .limit(1)
            .execute()
        )

        return result.data[0] if result.data else None

    except Exception as e:
        logger.warning("LinkedIn profile fetch failed for %s: %s", username, e)
        return None


def embed_posts_to_pinecone(
    posts: list[dict[str, Any]],
    company: str = "",
    namespace: str = "posts",
) -> int:
    """Embed posts into Pinecone for semantic search with rich metadata.

    Uses stable IDs (``{company}-{content_hash}``) so re-runs are idempotent.
    Each vector stores the full post text (truncated to 1000 chars for metadata),
    company, engagement metrics, posted_at, and ICP reward when available.
    """
    import hashlib

    settings = get_settings()
    if not settings.pinecone_api_key:
        logger.debug("Pinecone not configured — skipping embedding")
        return 0

    try:
        from openai import OpenAI
        from pinecone import Pinecone

        oai = OpenAI(api_key=settings.openai_api_key)
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index)

        vectors = []
        for post in posts:
            text = (
                post.get("commentary") or post.get("post_text")
                or post.get("text") or post.get("copy") or ""
            ).strip()
            if not text:
                continue

            content_hash = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
            stable_id = f"{company}-{content_hash}" if company else f"post-{content_hash}"

            resp = oai.embeddings.create(model="text-embedding-3-small", input=text[:8000])

            impressions = post.get("impressionCount") or post.get("impressions") or 0
            reactions = post.get("likeCount") or post.get("reactions") or post.get("total_reactions") or 0
            comments = post.get("commentCount") or post.get("comments") or post.get("total_comments") or 0
            engagement = post.get("engagement_score") or (
                (reactions + comments * 3) / impressions if impressions > 0 else 0
            )
            posted_at = (
                post.get("publishedAt") or post.get("postedAt")
                or post.get("published_at") or post.get("posted_at") or ""
            )

            vectors.append({
                "id": stable_id,
                "values": resp.data[0].embedding,
                "metadata": {
                    "text": text[:1000],
                    "company": company,
                    "engagement": round(engagement, 4),
                    "impressions": impressions,
                    "reactions": reactions,
                    "comments": comments,
                    "posted_at": posted_at,
                    "icp_reward": post.get("icp_reward", 0),
                },
            })

        if vectors:
            BATCH = 100
            for i in range(0, len(vectors), BATCH):
                index.upsert(vectors=vectors[i:i + BATCH], namespace=namespace)
            logger.info("Embedded %d posts to Pinecone for %s", len(vectors), company or "global")

        return len(vectors)

    except Exception as e:
        logger.warning("Pinecone embedding failed: %s", e)
        return 0
