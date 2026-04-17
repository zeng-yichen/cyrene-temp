"""Application configuration loaded from environment variables."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("amphoreus")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


class Settings(BaseSettings):
    # --- LLM providers ---
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    parallel_api_key: str = ""

    # --- Supabase (read-only from app tables; auth/session may use Supabase client on frontend) ---
    supabase_url: str = ""
    supabase_key: str = ""

    # --- Pinecone ---
    pinecone_api_key: str = ""
    pinecone_index: str = "user-posts"

    # --- Serper (Google Search API) ---
    serper_api_key: str = ""
    serper_base_url: str = "https://google.serper.dev/search"

    # --- Ordinal ---
    ordinal_api_key: str = ""
    # Public origin where this API is reachable (no trailing slash). Required for Ordinal to fetch
    # draft images from POST /uploads (e.g. https://your-app.com or an ngrok URL).
    public_base_url: str = ""

    # --- E2B ---
    e2b_api_key: str = ""

    # --- App ---
    allowed_origins: str = "http://localhost:3000"
    jwt_secret: str = ""
    workspace_backend: str = "local"  # "local" or "e2b"
    cache_backend: str = "sqlite"  # "sqlite" or "redis"
    redis_url: str = ""
    data_dir: str = str(PROJECT_ROOT / "data")
    sqlite_path: str = str(PROJECT_ROOT / "data" / "amphoreus.db")

    # --- Cloudflare Access (Stage 2 auth) ---
    # When both are set, every request (except /health, /docs, /openapi.json) must carry a valid
    # Cloudflare Access JWT in the Cf-Access-Jwt-Assertion header or CF_Authorization cookie.
    # When either is empty, auth is bypassed (local dev mode — never do this in prod).
    cf_access_team_domain: str = ""  # e.g. "cyrene-stelle.cloudflareaccess.com"
    cf_access_aud: str = ""  # 64-char application audience tag from Zero Trust dashboard
    # Volume-resident ACL file. Maps emails → allowed client slugs. See backend/src/auth/acl.py.
    acl_path: str = str(PROJECT_ROOT / "data" / "acl.json")
    # Append-only audit log for write methods.
    audit_log_path: str = str(PROJECT_ROOT / "data" / "audit.log")

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
