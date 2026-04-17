"""SQLite local database for operational state.

Stores runs, events, fact-check results, eval results, and cache entries.
Supabase handles shared/cloud state — we never modify its schema.
"""

import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from backend.src.core.config import get_settings

_DB_LOCK = threading.Lock()
_INITIALIZED = False


def _db_path() -> str:
    settings = get_settings()
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    return settings.sqlite_path


@contextmanager
def get_connection():
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_db() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _DB_LOCK:
        if _INITIALIZED:
            return
        with get_connection() as conn:
            conn.executescript(_SCHEMA)
            _migrate_local_posts_columns(conn)
            _migrate_post_engagers(conn)
        _INITIALIZED = True


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    client_slug TEXT NOT NULL,
    agent TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    prompt TEXT,
    output TEXT,
    error TEXT,
    config_snapshot TEXT,
    started_at REAL,
    completed_at REAL,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    event_type TEXT NOT NULL,
    data TEXT,
    timestamp REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id);

CREATE TABLE IF NOT EXISTS fact_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(id),
    post_index INTEGER,
    report TEXT,
    corrected_post TEXT,
    has_corrections INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    passed INTEGER NOT NULL DEFAULT 0,
    grade_results TEXT,
    trace TEXT,
    duration_seconds REAL,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);

CREATE TABLE IF NOT EXISTS workspace_snapshots (
    id TEXT PRIMARY KEY,
    client_slug TEXT NOT NULL,
    run_id TEXT REFERENCES runs(id),
    snapshot_path TEXT NOT NULL,
    content_hashes TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_snapshots_client ON workspace_snapshots(client_slug);

CREATE TABLE IF NOT EXISTS local_posts (
    id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    content TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    why_post TEXT,
    citation_comments TEXT,
    ordinal_post_id TEXT,
    linked_image_id TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_local_posts_company ON local_posts(company);

CREATE TABLE IF NOT EXISTS ruan_mei_state (
    company TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    last_updated REAL NOT NULL DEFAULT (unixepoch('now'))
);

CREATE TABLE IF NOT EXISTS post_engagers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    ordinal_post_id TEXT NOT NULL,
    linkedin_post_url TEXT NOT NULL,
    engager_urn TEXT NOT NULL,
    name TEXT,
    headline TEXT,
    engagement_type TEXT NOT NULL DEFAULT 'reaction',
    fetched_at REAL NOT NULL DEFAULT (unixepoch('now')),
    icp_score REAL,
    current_company TEXT,
    title TEXT,
    location TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_post_engagers_dedup
    ON post_engagers(ordinal_post_id, engager_urn, engagement_type);
CREATE INDEX IF NOT EXISTS idx_post_engagers_company ON post_engagers(company);
CREATE INDEX IF NOT EXISTS idx_post_engagers_post ON post_engagers(ordinal_post_id);

-- Per-LLM-call usage/cost ledger. One row per provider API call, attributed
-- to the authenticated user (from the CF Access middleware ContextVar) and,
-- when available, the client slug the call was made on behalf of. Cost is
-- computed at record-time from a static price table in
-- backend/src/usage/pricing.py; editing that table does not backfill
-- existing rows.
--
-- provider:  "anthropic" | "openai" | "perplexity"
-- call_kind: "messages" | "embeddings" | "chat"
-- user_email / client_slug: nullable for unattributed calls (background
--   tasks like ordinal_sync that run outside any HTTP request).
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    call_kind TEXT NOT NULL,
    user_email TEXT,
    client_slug TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    duration_ms INTEGER,
    error TEXT,
    created_at REAL NOT NULL DEFAULT (unixepoch('now'))
);
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_events(user_email, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_client ON usage_events(client_slug, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_events(created_at);
"""


def _migrate_post_engagers(conn: sqlite3.Connection) -> None:
    """Create post_engagers table and add columns for DBs that predate them."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS post_engagers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            ordinal_post_id TEXT NOT NULL,
            linkedin_post_url TEXT NOT NULL,
            engager_urn TEXT NOT NULL,
            name TEXT,
            headline TEXT,
            engagement_type TEXT NOT NULL DEFAULT 'reaction',
            fetched_at REAL NOT NULL DEFAULT (unixepoch('now')),
            icp_score REAL,
            current_company TEXT,
            title TEXT,
            location TEXT
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_post_engagers_dedup
            ON post_engagers(ordinal_post_id, engager_urn, engagement_type)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_post_engagers_company
            ON post_engagers(company)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_post_engagers_post
            ON post_engagers(ordinal_post_id)
    """)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(post_engagers)").fetchall()}
    for col, typedef in [("icp_score", "REAL"), ("current_company", "TEXT"),
                         ("title", "TEXT"), ("location", "TEXT")]:
        if col not in cols:
            conn.execute(f"ALTER TABLE post_engagers ADD COLUMN {col} {typedef}")


def _migrate_local_posts_columns(conn: sqlite3.Connection) -> None:
    """Add why_post / citation_comments for existing DBs created before these columns."""
    rows = conn.execute("PRAGMA table_info(local_posts)").fetchall()
    cols = {r[1] for r in rows}
    if "why_post" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN why_post TEXT")
    if "citation_comments" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN citation_comments TEXT")
    if "ordinal_post_id" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN ordinal_post_id TEXT")
    if "linked_image_id" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN linked_image_id TEXT")
    if "pre_revision_content" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN pre_revision_content TEXT")
    if "cyrene_score" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN cyrene_score REAL")
    if "generation_metadata" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN generation_metadata TEXT")
    if "scheduled_date" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN scheduled_date TEXT")
    if "publication_order" not in cols:
        conn.execute("ALTER TABLE local_posts ADD COLUMN publication_order INTEGER")


# --- Run helpers ---

def mark_stale_runs_failed() -> int:
    """Mark any jobs left in 'running' state as failed (interrupted by server restart)."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE runs SET status='failed', error='Interrupted by server restart', completed_at=? WHERE status='running'",
            (time.time(),),
        )
        return cursor.rowcount


def create_run(run_id: str, client_slug: str, agent: str, prompt: str | None = None, config: dict | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO runs (id, client_slug, agent, status, prompt, config_snapshot, started_at) VALUES (?, ?, ?, 'running', ?, ?, ?)",
            (run_id, client_slug, agent, prompt, json.dumps(config) if config else None, time.time()),
        )


def complete_run(run_id: str, output: str | None = None, error: str | None = None) -> None:
    status = "failed" if error else "completed"
    with get_connection() as conn:
        conn.execute(
            "UPDATE runs SET status=?, output=?, error=?, completed_at=? WHERE id=?",
            (status, output, error, time.time(), run_id),
        )


def get_run(run_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


def list_runs(client_slug: str, limit: int = 20) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM runs WHERE client_slug=? ORDER BY created_at DESC LIMIT ?",
            (client_slug, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Event helpers ---

def record_event(run_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO run_events (run_id, event_type, data) VALUES (?, ?, ?)",
            (run_id, event_type, json.dumps(data) if data else None),
        )


def get_run_events(run_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM run_events WHERE run_id=? ORDER BY timestamp",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run_events_after(run_id: str, after_id: int) -> list[dict]:
    """Return all run_events for this run_id whose id > after_id, ordered by id.

    Used by job_manager.drain_events to poll for new events emitted by a
    detached subprocess (stelle_runner). The `id` column is an
    auto-incrementing primary key, so comparing `id > after_id` gives a
    strictly monotonic cursor without timestamp ambiguity.

    Each row's `data` field is JSON-decoded before returning so the
    caller gets the same shape as an in-memory AgentEvent.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, run_id, event_type, data, timestamp FROM run_events "
            "WHERE run_id = ? AND id > ? ORDER BY id ASC",
            (run_id, after_id),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        raw = d.get("data")
        if isinstance(raw, str) and raw:
            try:
                d["data"] = json.loads(raw)
            except Exception:
                d["data"] = {"raw": raw}
        elif raw is None:
            d["data"] = {}
        out.append(d)
    return out


# --- Fact-check helpers ---

def record_fact_check(run_id: str, post_index: int, report: str, corrected_post: str | None = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO fact_checks (run_id, post_index, report, corrected_post, has_corrections) VALUES (?, ?, ?, ?, ?)",
            (run_id, post_index, report, corrected_post, 1 if corrected_post else 0),
        )


# --- Cache helpers ---

def cache_get(key: str) -> str | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM cache WHERE key=? AND expires_at > ?",
            (key, time.time()),
        ).fetchone()
        return row["value"] if row else None


def cache_set(key: str, value: str, ttl_seconds: int = 3600) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value, time.time() + ttl_seconds),
        )


def cache_cleanup() -> int:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))
        return cursor.rowcount


# --- Local post helpers ---

def create_local_post(
    post_id: str,
    company: str,
    content: str,
    title: str | None = None,
    status: str = "draft",
    why_post: str | None = None,
    citation_comments: list[str] | None = None,
    pre_revision_content: str | None = None,
    cyrene_score: float | None = None,
    generation_metadata: dict | None = None,
    publication_order: int | None = None,
    scheduled_date: str | None = None,
) -> dict:
    cc_json = json.dumps(citation_comments) if citation_comments else None
    gen_meta_json = json.dumps(generation_metadata) if generation_metadata else None
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO local_posts (id, company, content, title, status, why_post, citation_comments, ordinal_post_id, linked_image_id, pre_revision_content, cyrene_score, generation_metadata, publication_order, scheduled_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)",
            (post_id, company, content, title, status, why_post, cc_json, pre_revision_content, cyrene_score, gen_meta_json, publication_order, scheduled_date),
        )
    return get_local_post(post_id) or {
        "id": post_id,
        "company": company,
        "content": content,
        "title": title,
        "status": status,
        "why_post": why_post,
        "citation_comments": cc_json,
        "ordinal_post_id": None,
        "linked_image_id": None,
        "created_at": time.time(),
    }


def list_local_posts(company: str | None = None, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        if company:
            rows = conn.execute(
                "SELECT * FROM local_posts WHERE company=? ORDER BY created_at DESC LIMIT ?",
                (company, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM local_posts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_local_post(post_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM local_posts WHERE id=?", (post_id,)).fetchone()
        return dict(row) if row else None


def update_local_post(
    post_id: str,
    content: str | None = None,
    status: str | None = None,
    title: str | None = None,
) -> dict | None:
    fields, values = [], []
    if content is not None:
        fields.append("content=?"); values.append(content)
    if status is not None:
        fields.append("status=?"); values.append(status)
    if title is not None:
        fields.append("title=?"); values.append(title)
    if not fields:
        return get_local_post(post_id)
    values.append(post_id)
    with get_connection() as conn:
        conn.execute(f"UPDATE local_posts SET {', '.join(fields)} WHERE id=?", values)
    return get_local_post(post_id)


def update_post_schedule(post_id: str, scheduled_date: str | None) -> dict | None:
    """Update the scheduled publication date for a post (calendar drag-drop)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE local_posts SET scheduled_date=? WHERE id=?",
            (scheduled_date, post_id),
        )
    return get_local_post(post_id)


def list_calendar_posts(company: str, month: str | None = None) -> list[dict]:
    """Return all local_posts for a company, optionally filtered to a month.

    month format: '2026-04' (YYYY-MM). If None, returns all posts.
    Results ordered by scheduled_date (nulls last), then publication_order.
    """
    with get_connection() as conn:
        if month:
            rows = conn.execute(
                """SELECT * FROM local_posts
                   WHERE company = ?
                     AND (scheduled_date LIKE ? OR scheduled_date IS NULL)
                   ORDER BY
                     CASE WHEN scheduled_date IS NULL THEN 1 ELSE 0 END,
                     scheduled_date,
                     publication_order""",
                (company, f"{month}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM local_posts
                   WHERE company = ?
                   ORDER BY
                     CASE WHEN scheduled_date IS NULL THEN 1 ELSE 0 END,
                     scheduled_date,
                     publication_order""",
                (company,),
            ).fetchall()
    return [dict(r) for r in rows]


def update_local_post_fields(post_id: str, updates: dict[str, Any]) -> dict | None:
    """Partial update: only keys present in ``updates`` are written (e.g. exclude_unset from PATCH)."""
    col_map = {
        "content": "content",
        "status": "status",
        "title": "title",
        "linked_image_id": "linked_image_id",
        "scheduled_date": "scheduled_date",
        "publication_order": "publication_order",
    }
    fields, values = [], []
    for key, col in col_map.items():
        if key not in updates:
            continue
        val = updates[key]
        if key == "linked_image_id":
            val = val or None
        fields.append(f"{col}=?")
        values.append(val)
    if not fields:
        return get_local_post(post_id)
    values.append(post_id)
    with get_connection() as conn:
        conn.execute(f"UPDATE local_posts SET {', '.join(fields)} WHERE id=?", values)
    return get_local_post(post_id)


def set_local_post_ordinal_post_id(local_post_id: str, ordinal_post_id: str | None) -> dict | None:
    """Store latest Ordinal workspace post id after a successful push (re-push overwrites)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE local_posts SET ordinal_post_id=? WHERE id=?",
            (ordinal_post_id, local_post_id),
        )
    return get_local_post(local_post_id)


def delete_local_post(post_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM local_posts WHERE id=?", (post_id,))


def purge_unpushed_drafts(company: str) -> int:
    """Delete all draft local_posts rows for a company that never reached Ordinal.

    Called at the start of every Stelle run so each new batch starts with a
    clean slate. The rule is simple: if a draft has no ordinal_post_id, it
    was never committed to Ordinal, so per the user's dedup model it
    "doesn't exist" and should not persist across runs.

    Preserves:
      - Rows with a non-empty ordinal_post_id (pushed to Ordinal — even if
        the post later got unpublished, the Ordinal workspace still owns it)
      - Rows whose status is anything other than 'draft' (e.g. 'posted',
        'scheduled', 'failed' — those represent states the operator might
        still need to see)

    Returns the number of rows deleted.
    """
    company = (company or "").strip()
    if not company:
        return 0
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM local_posts "
            "WHERE company = ? "
            "  AND status = 'draft' "
            "  AND (ordinal_post_id IS NULL OR ordinal_post_id = '')",
            (company,),
        )
        return cur.rowcount or 0


# --- RuanMei state helpers ---

def ruan_mei_load(company: str) -> dict | None:
    """Return the stored state dict for a company, or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT state_json FROM ruan_mei_state WHERE company=?", (company,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def ruan_mei_save(company: str, state: dict) -> None:
    """Atomically persist RuanMei state for a company (upsert)."""
    state["last_updated"] = state.get("last_updated", "")
    blob = json.dumps(state)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO ruan_mei_state (company, state_json, last_updated) VALUES (?, ?, unixepoch('now'))"
            " ON CONFLICT(company) DO UPDATE SET state_json=excluded.state_json, last_updated=excluded.last_updated",
            (company, blob),
        )


# --- Post engager helpers ---

def upsert_engagers(company: str, ordinal_post_id: str, linkedin_post_url: str, engagers: list[dict]) -> int:
    """Insert engager records, ignoring duplicates. Returns count of newly inserted rows."""
    inserted = 0
    with get_connection() as conn:
        for e in engagers:
            urn = e.get("urn") or e.get("id") or e.get("profileId") or ""
            if not urn:
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO post_engagers "
                    "(company, ordinal_post_id, linkedin_post_url, engager_urn, name, headline, "
                    "engagement_type, current_company, title, location) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        company,
                        ordinal_post_id,
                        linkedin_post_url,
                        urn,
                        e.get("name") or e.get("firstName", "") + " " + e.get("lastName", ""),
                        e.get("headline") or e.get("occupation") or "",
                        e.get("engagement_type", "reaction"),
                        e.get("current_company") or e.get("companyName") or "",
                        e.get("title") or "",
                        e.get("location") or "",
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
            except Exception:
                pass
    return inserted


def get_engagers_for_post(ordinal_post_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM post_engagers WHERE ordinal_post_id=?",
            (ordinal_post_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def engagers_fetched_for_post(ordinal_post_id: str) -> bool:
    """Return True if we already have any engager rows for this post."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM post_engagers WHERE ordinal_post_id=? LIMIT 1",
            (ordinal_post_id,),
        ).fetchone()
        return row is not None


def update_engager_icp_scores(ordinal_post_id: str, scores: list[tuple[str, float]]) -> int:
    """Batch-update per-engager ICP scores for a post.

    Args:
        ordinal_post_id: The post these engagers belong to.
        scores: List of (engager_urn, icp_score) pairs. Order-independent.

    Returns:
        Number of rows updated.
    """
    updated = 0
    with get_connection() as conn:
        for urn, score in scores:
            conn.execute(
                "UPDATE post_engagers SET icp_score=? "
                "WHERE ordinal_post_id=? AND engager_urn=?",
                (round(score, 4), ordinal_post_id, urn),
            )
            updated += conn.execute("SELECT changes()").fetchone()[0]
    return updated


def get_top_icp_engagers(company: str, limit: int = 30) -> list[dict]:
    """Return top ICP-scored engagers for a company, aggregated across posts.

    Ranking metric: mean_icp_score * log(1 + engagement_count).
    Rewards both ICP fit and repeat engagement.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                engager_urn,
                MAX(name) AS name,
                MAX(headline) AS headline,
                MAX(current_company) AS current_company,
                MAX(title) AS title,
                MAX(location) AS location,
                AVG(icp_score) AS mean_icp_score,
                COUNT(*) AS engagement_count,
                GROUP_CONCAT(DISTINCT ordinal_post_id) AS post_ids
            FROM post_engagers
            WHERE company = ? AND icp_score IS NOT NULL AND icp_score > 0
            GROUP BY engager_urn
            ORDER BY AVG(icp_score) DESC
            """,
            (company,),
        ).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        mean_score = d["mean_icp_score"] or 0.0
        eng_count = d["engagement_count"] or 1
        d["ranking_score"] = round(mean_score * math.log(1 + eng_count), 4)
        d["posts_engaged"] = (d.pop("post_ids") or "").split(",")
        results.append(d)

    results.sort(key=lambda x: x["ranking_score"], reverse=True)
    return results[:limit]


def get_unscored_engager_post_ids(company: str) -> list[str]:
    """Return ordinal_post_ids that have engagers but no ICP scores yet."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT ordinal_post_id
            FROM post_engagers
            WHERE company = ? AND icp_score IS NULL
            """,
            (company,),
        ).fetchall()
    return [r[0] for r in rows]
