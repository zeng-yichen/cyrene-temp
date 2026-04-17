"""Usage / spend admin router — /api/usage.

Reads the ``usage_events`` ledger populated by
:mod:`backend.src.usage.instrumentation`. All endpoints are **admin-only**
so scoped users can't see each other's numbers (or their own — this is a
billing/ops tool, not a user-facing quota display).

Endpoints:

- ``GET /api/usage/summary``
    Aggregated totals grouped by user email, over an optional date range.
    This is what the frontend admin dashboard renders.

- ``GET /api/usage/by-user/{email}``
    Per-model drill-down for one user — shows which models they're hitting
    and how the cost decomposes. Useful for "why is a specific user's bill so high".

- ``GET /api/usage/events``
    Raw event stream (paginated). Mostly for debugging — confirms that a
    specific run actually recorded rows.

- ``GET /api/usage/by-client``
    Aggregated totals grouped by ``client_slug`` instead of user. Helps
    answer "which clients are burning the most tokens".

Date filtering: all endpoints accept ``?since=<iso>&until=<iso>``. Both
are optional; missing bounds default to (start-of-epoch, now).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from backend.src.auth.cf_access import AuthedUser
from backend.src.db.local import get_connection

logger = logging.getLogger("amphoreus.usage.router")

router = APIRouter(prefix="/api/usage", tags=["usage"])


# --- Helpers -------------------------------------------------------------


def _require_admin(request: Request) -> AuthedUser:
    """Same pattern as transcripts._require_admin — belt-and-suspenders
    admin gate on top of the path-based ACL. Usage numbers are sensitive."""
    user: AuthedUser | None = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="unauthenticated")
    if not getattr(request.state, "user_is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can view usage/spend data.",
        )
    return user


def _parse_time_bound(value: str | None, *, default: float) -> float:
    """Accept an ISO8601 timestamp or a unix float. Return a unix float."""
    if not value:
        return default
    # Unix float fast-path
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    # ISO parse — be forgiving of Z suffix
    try:
        cleaned = value.strip().rstrip("Z")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp {value!r}: {exc}",
        ) from exc


def _time_window(since: str | None, until: str | None) -> tuple[float, float]:
    lo = _parse_time_bound(since, default=0.0)
    hi = _parse_time_bound(until, default=time.time())
    if lo > hi:
        raise HTTPException(
            status_code=400, detail="`since` must be <= `until`"
        )
    return lo, hi


# --- Endpoints -----------------------------------------------------------


@router.get("/summary")
async def usage_summary(
    request: Request,
    since: str | None = Query(None, description="ISO8601 or unix float"),
    until: str | None = Query(None, description="ISO8601 or unix float"),
) -> dict[str, Any]:
    """Totals grouped by user_email. Unattributed rows bucket under ``null``."""
    _require_admin(request)
    lo, hi = _time_window(since, until)

    with get_connection() as conn:
        # Per-user totals
        per_user_rows = conn.execute(
            """
            SELECT
                user_email,
                COUNT(*)                    AS n_calls,
                SUM(input_tokens)           AS input_tokens,
                SUM(output_tokens)          AS output_tokens,
                SUM(cache_creation_tokens)  AS cache_creation_tokens,
                SUM(cache_read_tokens)      AS cache_read_tokens,
                SUM(cost_usd)               AS cost_usd
            FROM usage_events
            WHERE created_at BETWEEN ? AND ?
            GROUP BY user_email
            ORDER BY cost_usd DESC
            """,
            (lo, hi),
        ).fetchall()

        # Grand totals
        grand = conn.execute(
            """
            SELECT
                COUNT(*)                    AS n_calls,
                SUM(input_tokens)           AS input_tokens,
                SUM(output_tokens)          AS output_tokens,
                SUM(cost_usd)               AS cost_usd
            FROM usage_events
            WHERE created_at BETWEEN ? AND ?
            """,
            (lo, hi),
        ).fetchone()

    return {
        "since": lo,
        "until": hi,
        "total": {
            "n_calls": int(grand["n_calls"] or 0),
            "input_tokens": int(grand["input_tokens"] or 0),
            "output_tokens": int(grand["output_tokens"] or 0),
            "cost_usd": float(grand["cost_usd"] or 0.0),
        },
        "by_user": [
            {
                "user_email": r["user_email"],
                "n_calls": int(r["n_calls"] or 0),
                "input_tokens": int(r["input_tokens"] or 0),
                "output_tokens": int(r["output_tokens"] or 0),
                "cache_creation_tokens": int(r["cache_creation_tokens"] or 0),
                "cache_read_tokens": int(r["cache_read_tokens"] or 0),
                "cost_usd": float(r["cost_usd"] or 0.0),
            }
            for r in per_user_rows
        ],
    }


@router.get("/by-user/{email}")
async def usage_by_user(
    email: str,
    request: Request,
    since: str | None = Query(None),
    until: str | None = Query(None),
) -> dict[str, Any]:
    """Per-model drill-down for a single user."""
    _require_admin(request)
    lo, hi = _time_window(since, until)

    # Treat the literal string "null" (or empty) as "unattributed system calls".
    email_lc = (email or "").strip().lower()
    want_null = email_lc in ("", "null", "system", "unattributed")

    with get_connection() as conn:
        if want_null:
            rows = conn.execute(
                """
                SELECT model, provider,
                    COUNT(*) AS n_calls,
                    SUM(input_tokens) AS input_tokens,
                    SUM(output_tokens) AS output_tokens,
                    SUM(cost_usd) AS cost_usd
                FROM usage_events
                WHERE user_email IS NULL AND created_at BETWEEN ? AND ?
                GROUP BY model, provider
                ORDER BY cost_usd DESC
                """,
                (lo, hi),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT model, provider,
                    COUNT(*) AS n_calls,
                    SUM(input_tokens) AS input_tokens,
                    SUM(output_tokens) AS output_tokens,
                    SUM(cost_usd) AS cost_usd
                FROM usage_events
                WHERE LOWER(user_email) = ? AND created_at BETWEEN ? AND ?
                GROUP BY model, provider
                ORDER BY cost_usd DESC
                """,
                (email_lc, lo, hi),
            ).fetchall()

    return {
        "user_email": None if want_null else email_lc,
        "since": lo,
        "until": hi,
        "by_model": [
            {
                "model": r["model"],
                "provider": r["provider"],
                "n_calls": int(r["n_calls"] or 0),
                "input_tokens": int(r["input_tokens"] or 0),
                "output_tokens": int(r["output_tokens"] or 0),
                "cost_usd": float(r["cost_usd"] or 0.0),
            }
            for r in rows
        ],
    }


@router.get("/by-client")
async def usage_by_client(
    request: Request,
    since: str | None = Query(None),
    until: str | None = Query(None),
) -> dict[str, Any]:
    """Totals grouped by client_slug."""
    _require_admin(request)
    lo, hi = _time_window(since, until)

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                client_slug,
                COUNT(*)            AS n_calls,
                SUM(input_tokens)   AS input_tokens,
                SUM(output_tokens)  AS output_tokens,
                SUM(cost_usd)       AS cost_usd
            FROM usage_events
            WHERE created_at BETWEEN ? AND ?
            GROUP BY client_slug
            ORDER BY cost_usd DESC
            """,
            (lo, hi),
        ).fetchall()

    return {
        "since": lo,
        "until": hi,
        "by_client": [
            {
                "client_slug": r["client_slug"],
                "n_calls": int(r["n_calls"] or 0),
                "input_tokens": int(r["input_tokens"] or 0),
                "output_tokens": int(r["output_tokens"] or 0),
                "cost_usd": float(r["cost_usd"] or 0.0),
            }
            for r in rows
        ],
    }


@router.get("/events")
async def usage_events(
    request: Request,
    since: str | None = Query(None),
    until: str | None = Query(None),
    user_email: str | None = Query(None, description="Filter to one user"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Raw event feed. Sorted newest-first. Paginated."""
    _require_admin(request)
    lo, hi = _time_window(since, until)

    where = ["created_at BETWEEN ? AND ?"]
    params: list[Any] = [lo, hi]
    if user_email:
        where.append("LOWER(user_email) = ?")
        params.append(user_email.strip().lower())

    sql = f"""
        SELECT id, provider, model, call_kind, user_email, client_slug,
               input_tokens, output_tokens, cache_creation_tokens,
               cache_read_tokens, cost_usd, duration_ms, error, created_at
        FROM usage_events
        WHERE {' AND '.join(where)}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return {
        "since": lo,
        "until": hi,
        "limit": limit,
        "offset": offset,
        "events": [dict(r) for r in rows],
    }
