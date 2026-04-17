"""Supabase client for shared/cloud state.

Connects to the Supabase database as a client.
We do NOT create or modify any Supabase tables or schema.
"""

from functools import lru_cache

from supabase import Client, create_client

from backend.src.core.config import get_settings


@lru_cache
def get_supabase() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(settings.supabase_url, settings.supabase_key)
