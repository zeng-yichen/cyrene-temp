#!/usr/bin/env python3
"""
One-time migration: backfill ordinal_post_id and linkedin_post_url on
existing RuanMei observations by re-matching against Ordinal analytics.

For each client in ordinal_auth_rows.csv:
  1. Fetch current analytics from Ordinal
  2. For each observation missing ordinal_post_id or linkedin_post_url,
     match by text hash against analytics posts
  3. Fill in both fields and save

Usage:
    python scripts/backfill_observation_ids.py            # run
    python scripts/backfill_observation_ids.py --dry-run   # preview
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

import httpx
from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from backend.src.db import vortex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_obs_ids")

ORDINAL_BASE = "https://app.tryordinal.com/api/v1"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def fetch_analytics(profile_id: str, api_key: str) -> list[dict]:
    from datetime import datetime, timedelta, timezone

    start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    resp = httpx.get(
        f"{ORDINAL_BASE}/analytics/linkedin/{profile_id}/posts",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"startDate": start, "endDate": end},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else data.get("posts", data.get("data", []))


def build_analytics_index(posts: list[dict]) -> dict[str, dict]:
    """Map text hash -> {ordinal_post_id, linkedin_post_url, text}."""
    index = {}
    for post in posts:
        text = (
            post.get("commentary") or post.get("text") or post.get("copy")
            or post.get("content") or post.get("post_text") or ""
        ).strip()
        if not text:
            continue

        op = post.get("ordinalPost")
        oid = ""
        if isinstance(op, dict):
            oid = str(op.get("id") or "").strip()

        url = (
            post.get("url") or post.get("linkedInUrl")
            or post.get("linkedin_url") or post.get("postUrl") or ""
        )

        h = _hash(text)
        index[h] = {"ordinal_post_id": oid, "linkedin_post_url": url, "text": text}
    return index


def main():
    parser = argparse.ArgumentParser(description="Backfill observation IDs from Ordinal analytics")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        logger.error("CSV not found at %s", csv_path)
        sys.exit(1)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_fixed = 0
    total_checked = 0

    for row in rows:
        api_key = row.get("api_key", "").strip()
        company = row.get("provider_org_slug", "").strip() or row.get("company_id", "").strip()
        profile_id = row.get("profile_id", "").strip()

        if not api_key or not company or not profile_id:
            continue

        try:
            from backend.src.db.local import initialize_db, ruan_mei_load
            initialize_db()
            state = ruan_mei_load(company)
        except Exception:
            state = None
        if not state:
            continue

        observations = state.get("observations", [])
        needs_fix = [
            o for o in observations
            if not o.get("ordinal_post_id") or not o.get("linkedin_post_url")
        ]

        if not needs_fix:
            continue

        logger.info("%s: %d observations need backfill, fetching analytics...", company, len(needs_fix))

        try:
            posts = fetch_analytics(profile_id, api_key)
        except Exception as e:
            logger.warning("  Failed to fetch analytics: %s", e)
            continue

        index = build_analytics_index(posts)
        logger.info("  Built index with %d posts from Ordinal", len(index))

        fixed = 0
        for obs in needs_fix:
            total_checked += 1
            h = obs.get("post_hash", "")
            if h not in index:
                continue

            match = index[h]
            changed = False

            if not obs.get("ordinal_post_id") and match["ordinal_post_id"]:
                if not args.dry_run:
                    obs["ordinal_post_id"] = match["ordinal_post_id"]
                changed = True

            if not obs.get("linkedin_post_url") and match["linkedin_post_url"]:
                if not args.dry_run:
                    obs["linkedin_post_url"] = match["linkedin_post_url"]
                changed = True

            if changed:
                fixed += 1

        if fixed and not args.dry_run:
            from backend.src.db.local import ruan_mei_save
            ruan_mei_save(company, state)

        logger.info("  %s %d/%d observations", "Would fix" if args.dry_run else "Fixed", fixed, len(needs_fix))
        total_fixed += fixed

    logger.info("=== %s: %d observations across all clients ===",
                "DRY RUN would fix" if args.dry_run else "Total fixed", total_fixed)


if __name__ == "__main__":
    main()
