#!/usr/bin/env python3
"""
Backfill RuanMei observations from Ordinal analytics history.

One-time script that ingests all published posts for each client,
analyzes them via Haiku, and records them as scored observations.
After running, RuanMei has signal from day one instead of waiting
for the hourly sync to drip-feed 20 posts per cycle.

Usage:
    # Backfill all clients in ordinal_auth_rows.csv:
    python scripts/backfill_ruan_mei.py

    # Backfill a single client:
    python scripts/backfill_ruan_mei.py --company example-client

    # Backfill with longer lookback (default 365 days):
    python scripts/backfill_ruan_mei.py --lookback-days 730

    # Dry run (fetch + count, no analysis):
    python scripts/backfill_ruan_mei.py --dry-run
"""

import argparse
import asyncio
import csv
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on sys.path.
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

import httpx
from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from backend.src.db import vortex
from backend.src.agents.ruan_mei import RuanMei

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill")

ORDINAL_BASE = "https://app.tryordinal.com/api/v1"


def fetch_all_analytics(
    profile_id: str,
    api_key: str,
    lookback_days: int = 365,
) -> list[dict]:
    """Fetch all post analytics from Ordinal for the given lookback period."""
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        resp = httpx.get(
            f"{ORDINAL_BASE}/analytics/linkedin/{profile_id}/posts",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            params={"startDate": start, "endDate": end},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Ordinal fetch failed for profile %s: %s", profile_id, e)
        return []

    posts = data if isinstance(data, list) else data.get("posts", data.get("data", []))
    return posts


async def backfill_company(
    company: str,
    profile_id: str,
    api_key: str,
    lookback_days: int = 365,
    dry_run: bool = False,
) -> int:
    """Backfill all Ordinal posts for a single company."""
    logger.info("--- %s ---", company)

    posts = fetch_all_analytics(profile_id, api_key, lookback_days)
    if not posts:
        logger.info("  No posts found for %s", company)
        return 0

    # Filter to posts with text and impressions.
    valid = []
    for p in posts:
        text = (
            p.get("commentary") or p.get("text") or p.get("copy")
            or p.get("content") or p.get("post_text") or ""
        ).strip()
        impressions = p.get("impressionCount") or p.get("impressions") or 0
        if text and impressions > 0:
            valid.append(p)

    logger.info("  Found %d posts (%d with text + impressions)", len(posts), len(valid))

    if dry_run:
        logger.info("  [DRY RUN] Would analyze and ingest %d posts", len(valid))
        return 0

    rm = RuanMei(company)
    existing = rm.observation_count()
    logger.info("  Existing observations: %d", existing)

    # Ingest without batch_size limit for backfill.
    ingested = await rm.ingest_from_ordinal(valid, batch_size=len(valid))
    logger.info("  Ingested %d new observations (total now: %d)", ingested, rm.observation_count())

    return ingested


async def main():
    parser = argparse.ArgumentParser(description="Backfill RuanMei from Ordinal history")
    parser.add_argument("--company", help="Backfill a single company (default: all)")
    parser.add_argument("--lookback-days", type=int, default=365, help="Days of history to fetch (default: 365)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and count only, no analysis")
    args = parser.parse_args()

    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        logger.error("No ordinal_auth_rows.csv found at %s", csv_path)
        sys.exit(1)

    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            api_key = row.get("api_key", "").strip()
            company = row.get("provider_org_slug", "").strip() or row.get("company_id", "").strip()
            profile_id = row.get("profile_id", "").strip()
            if api_key and company and profile_id:
                rows.append((company, profile_id, api_key))

    if not rows:
        logger.error("No valid rows in ordinal_auth_rows.csv")
        sys.exit(1)

    if args.company:
        rows = [(c, p, k) for c, p, k in rows if c == args.company]
        if not rows:
            logger.error("Company '%s' not found in CSV", args.company)
            sys.exit(1)

    logger.info("Backfilling %d companies (lookback=%d days, dry_run=%s)",
                len(rows), args.lookback_days, args.dry_run)

    total = 0
    for company, profile_id, api_key in rows:
        ingested = await backfill_company(
            company, profile_id, api_key,
            lookback_days=args.lookback_days,
            dry_run=args.dry_run,
        )
        total += ingested

    logger.info("=== Done. Total ingested: %d ===", total)


if __name__ == "__main__":
    asyncio.run(main())
