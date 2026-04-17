#!/usr/bin/env python3
"""
Auto-fill the profile_id column in ordinal_auth_rows.csv.

For each row with an api_key but no profile_id, calls Ordinal's
GET /profiles/scheduling endpoint, filters for channel == "LinkedIn",
and writes the profile UUID back into the CSV.

Usage:
    python scripts/fill_profile_ids.py            # fill in place
    python scripts/fill_profile_ids.py --dry-run   # preview only
"""

import argparse
import csv
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
logger = logging.getLogger("fill_profile_ids")

ORDINAL_BASE = "https://app.tryordinal.com/api/v1"


def fetch_linkedin_profile_id(api_key: str) -> str | None:
    """Call GET /profiles/scheduling and return the LinkedIn profile UUID, or None."""
    try:
        resp = httpx.get(
            f"{ORDINAL_BASE}/profiles/scheduling",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        profiles = resp.json()
    except Exception as e:
        logger.warning("  API call failed: %s", e)
        return None

    linkedin = [p for p in profiles if p.get("channel") == "LinkedIn"]

    if len(linkedin) == 0:
        logger.info("  No LinkedIn profile found")
        return None
    if len(linkedin) > 1:
        logger.warning(
            "  Multiple LinkedIn profiles found (%d) — skipping (needs manual pick)",
            len(linkedin),
        )
        for p in linkedin:
            logger.warning("    id=%s  name=%s", p.get("id"), p.get("name"))
        return None

    return str(linkedin[0]["id"]).strip()


def main():
    parser = argparse.ArgumentParser(description="Auto-fill profile_id in ordinal_auth_rows.csv")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        logger.error("CSV not found at %s", csv_path)
        sys.exit(1)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "profile_id" not in fieldnames:
        logger.error("CSV missing 'profile_id' column header")
        sys.exit(1)

    filled = 0
    skipped = 0

    for row in rows:
        api_key = row.get("api_key", "").strip()
        company = row.get("provider_org_slug", "").strip() or row.get("company_id", "").strip()
        existing = row.get("profile_id", "").strip()

        if existing:
            continue
        if not api_key:
            continue

        logger.info("Resolving profile_id for %s ...", company)
        pid = fetch_linkedin_profile_id(api_key)

        if pid:
            logger.info("  -> %s", pid)
            if not args.dry_run:
                row["profile_id"] = pid
            filled += 1
        else:
            skipped += 1

    if args.dry_run:
        logger.info("=== DRY RUN: would fill %d rows, skipped %d ===", filled, skipped)
        return

    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("=== Done. Filled %d, skipped %d. CSV updated. ===", filled, skipped)


if __name__ == "__main__":
    main()
