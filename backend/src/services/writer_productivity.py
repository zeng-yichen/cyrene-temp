"""Writer productivity tracking — Google Docs monitoring + metrics.

Writer-productivity-service module.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def sync_productivity(company: str | None = None) -> dict[str, Any]:
    """Monitor Google Docs for writing activity and compute metrics.

    Requires Google Docs API credentials and document IDs to be configured.
    Returns productivity metrics dict.
    """
    logger.info("Productivity sync for %s", company or "all companies")

    # TODO: Implement Google Docs API integration
    # 1. Read Google Docs via Docs API
    # 2. Detect post counts, word counts, changes by comparing snapshots
    # 3. Estimate writing time via Drive Activity API session clustering
    # 4. Store metrics in Supabase writer_productivity_logs

    return {
        "status": "not_yet_implemented",
        "note": "Requires Google Docs API credentials",
    }


def generate_weekly_report(company: str) -> str:
    """Generate a weekly productivity report."""
    # TODO: Aggregate weekly data, format as email/markdown
    return f"Weekly productivity report for {company} — not yet implemented"


def get_productivity_metrics(company: str, days: int = 30) -> dict[str, Any]:
    """Get productivity metrics for a company over the last N days."""
    try:
        from backend.src.db.supabase_client import get_supabase
        sb = get_supabase()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        result = (
            sb.table("writer_productivity_logs")
            .select("*")
            .gte("created_at", cutoff)
            .execute()
        )
        return {"metrics": result.data or [], "days": days}
    except Exception as e:
        logger.warning("Productivity metrics fetch failed: %s", e)
        return {"metrics": [], "days": days}
