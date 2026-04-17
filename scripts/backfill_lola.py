#!/usr/bin/env python3
"""
DEPRECATED: LOLA bandit has been superseded by RuanMei.recommend_context().

This script backfilled LOLA arm rewards from RuanMei observations.
It is no longer needed — content intelligence is now provided directly
by RuanMei's Claude-as-Analyst pipeline (Phase 1).

Kept for reference only.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from backend.src.db import vortex
from backend.src.agents.lola import LOLA
from backend.src.agents.ruan_mei import RuanMei

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_lola")


def backfill_company(company: str, seed_arms: bool = False, dry_run: bool = False) -> dict:
    """Backfill LOLA arm rewards from RuanMei observations for one company.

    Returns a summary dict with counts.
    """
    rm = RuanMei(company)
    total_obs = rm.observation_count()
    scored_obs = sum(
        1 for o in rm._state.get("observations", []) if o.get("status") == "scored"
    )

    if dry_run:
        lola = LOLA(company)
        arms = lola._arms()
        logger.info(
            "  [DRY RUN] %s: %d scored obs, %d LOLA arms (total_pulls=%d)",
            company, scored_obs, len(arms), lola._state.total_pulls,
        )
        return {"company": company, "scored_obs": scored_obs, "arms": len(arms), "updated": 0}

    lola = LOLA(company)

    # Optionally seed arms from topic_arms.json.
    arms_seeded = 0
    if seed_arms:
        arms_path = vortex.memory_dir(company) / "topic_arms.json"
        if arms_path.exists():
            try:
                seed_data = json.loads(arms_path.read_text(encoding="utf-8"))
                arms_seeded = lola.seed_arms(seed_data)
                if arms_seeded:
                    logger.info("  Seeded %d new arms from topic_arms.json for %s", arms_seeded, company)
            except Exception as e:
                logger.warning("  Failed to seed arms for %s: %s", company, e)
        else:
            logger.debug("  No topic_arms.json found for %s — skipping seed", company)

    if scored_obs == 0:
        logger.info("  %s: no scored observations yet — run backfill_ruan_mei.py first", company)
        return {"company": company, "scored_obs": 0, "arms": len(lola._arms()), "updated": 0}

    updated = lola.update_from_ruan_mei()
    summary = lola.summary()

    logger.info(
        "  %s: %d scored obs → %d arm rewards updated (top topic: %s)",
        company, scored_obs, updated, summary.get("top_topic") or "n/a",
    )
    return {
        "company": company,
        "total_obs": total_obs,
        "scored_obs": scored_obs,
        "arms_seeded": arms_seeded,
        "arms_updated": updated,
        "lola_summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill LOLA bandit rewards from RuanMei history")
    parser.add_argument("--company", help="Backfill a single company (default: all)")
    parser.add_argument("--seed-arms", action="store_true",
                        help="Seed LOLA arms from topic_arms.json in each company's memory dir before updating")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show current state without updating")
    args = parser.parse_args()

    csv_path = vortex.ordinal_auth_csv()
    if not csv_path.exists():
        logger.error("No ordinal_auth_rows.csv at %s", csv_path)
        sys.exit(1)

    companies: list[str] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            api_key = row.get("api_key", "").strip()
            company = (row.get("provider_org_slug") or row.get("company_id") or "").strip()
            if api_key and company:
                companies.append(company)

    if not companies:
        logger.error("No valid companies in ordinal_auth_rows.csv")
        sys.exit(1)

    if args.company:
        companies = [c for c in companies if c == args.company]
        if not companies:
            logger.error("Company '%s' not found in CSV", args.company)
            sys.exit(1)

    logger.info(
        "LOLA backfill: %d companies (seed_arms=%s, dry_run=%s)",
        len(companies), args.seed_arms, args.dry_run,
    )

    results = []
    for company in companies:
        result = backfill_company(company, seed_arms=args.seed_arms, dry_run=args.dry_run)
        results.append(result)

    total_updated = sum(r.get("arms_updated", 0) for r in results)
    logger.info("=== Done. Total arm reward updates: %d ===", total_updated)


if __name__ == "__main__":
    print("DEPRECATED: LOLA bandit has been superseded by RuanMei.recommend_context().")
    print("This script is no longer functional. See ruan_mei.py for the new architecture.")
    import sys
    sys.exit(0)
