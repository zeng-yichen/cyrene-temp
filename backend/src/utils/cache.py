"""Caching layer — SQLite-backed by default, optional Redis for cloud deployments."""

import hashlib
import json
import logging
from typing import Any

from backend.src.core.config import get_settings

logger = logging.getLogger(__name__)


def _make_key(provider: str, model: str, prompt_hash: str) -> str:
    return f"{provider}:{model}:{prompt_hash}"


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]


def get(provider: str, model: str, prompt: str) -> str | None:
    """Retrieve a cached LLM response."""
    settings = get_settings()
    key = _make_key(provider, model, hash_prompt(prompt))

    if settings.cache_backend == "redis":
        return _redis_get(key)
    return _sqlite_get(key)


def set(provider: str, model: str, prompt: str, value: str, ttl_seconds: int = 3600) -> None:
    """Cache an LLM response."""
    settings = get_settings()
    key = _make_key(provider, model, hash_prompt(prompt))

    if settings.cache_backend == "redis":
        _redis_set(key, value, ttl_seconds)
    else:
        _sqlite_set(key, value, ttl_seconds)


def _sqlite_get(key: str) -> str | None:
    from backend.src.db.local import cache_get
    return cache_get(key)


def _sqlite_set(key: str, value: str, ttl_seconds: int) -> None:
    from backend.src.db.local import cache_set
    cache_set(key, value, ttl_seconds)


def _redis_get(key: str) -> str | None:
    try:
        import redis
        settings = get_settings()
        r = redis.from_url(settings.redis_url)
        val = r.get(key)
        return val.decode("utf-8") if val else None
    except Exception as e:
        logger.warning("Redis get failed: %s", e)
        return None


def _redis_set(key: str, value: str, ttl_seconds: int) -> None:
    try:
        import redis
        settings = get_settings()
        r = redis.from_url(settings.redis_url)
        r.setex(key, ttl_seconds, value)
    except Exception as e:
        logger.warning("Redis set failed: %s", e)
