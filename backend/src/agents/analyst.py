"""Shared tool implementations exposed to Stelle at generation time.

Historical note (2026-04-11): this module used to house an "analyst" agent
that ran hypothesis-driven engagement analysis weekly and wrote prose
findings to `analyst_findings.json`. Stelle and other generators then read
those findings as pre-chewed context. That pipeline was removed as a
Bitter Lesson violation — a frozen human-readable intermediate
representation sitting between raw observations and the generating agent.
The tool primitives the analyst used are valuable though, so they remain
here for Stelle to call directly as tools during drafting. Stelle imports
them as `_tool_query_observations`, `_tool_search_linkedin_bank`, and
`_tool_execute_python` from this module.

Keep the module name for import stability. If you're looking for agent
loop code (`run_analysis`, system prompt, retry/cache helpers), it's gone.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

_PYTHON_TIMEOUT_SECONDS = 60


# ------------------------------------------------------------------
# Observation query tool
# ------------------------------------------------------------------

def _filter_observations(observations: list[dict], tool_input: dict) -> list[dict]:
    """Apply filters from tool_input to observations."""
    filtered = list(observations)
    topic = tool_input.get("topic_filter")
    if topic:
        filtered = [o for o in filtered if o.get("topic_tag") == topic]
    fmt = tool_input.get("format_filter")
    if fmt:
        filtered = [o for o in filtered if o.get("format_tag") == fmt]
    min_r = tool_input.get("min_reward")
    if min_r is not None:
        filtered = [o for o in filtered if (o.get("reward", {}).get("immediate", 0)) >= min_r]
    max_r = tool_input.get("max_reward")
    if max_r is not None:
        filtered = [o for o in filtered if (o.get("reward", {}).get("immediate", 0)) <= max_r]
    return filtered


def _tool_query_observations(tool_input: dict, observations: list[dict]) -> str:
    """Return observations as JSON, with optional filtering and summary mode."""
    filtered = _filter_observations(observations, tool_input)
    limit = tool_input.get("limit")
    if limit and isinstance(limit, int):
        filtered = filtered[:limit]

    if tool_input.get("summary_only"):
        from collections import Counter
        rewards = [o.get("reward", {}).get("immediate", 0) for o in filtered]
        topics = Counter(o.get("topic_tag", "?") for o in filtered)
        formats = Counter(o.get("format_tag", "?") for o in filtered)
        if rewards:
            import statistics as _stats
            mean_r = _stats.mean(rewards)
            std_r = _stats.stdev(rewards) if len(rewards) > 1 else 0
        else:
            mean_r = 0
            std_r = 0
        return json.dumps({
            "count": len(filtered),
            "reward_mean": round(mean_r, 4),
            "reward_std": round(std_r, 4),
            "reward_min": round(min(rewards), 4) if rewards else None,
            "reward_max": round(max(rewards), 4) if rewards else None,
            "topic_distribution": dict(topics.most_common()),
            "format_distribution": dict(formats.most_common()),
        })

    return json.dumps({"count": len(filtered), "observations": filtered}, default=str)


# ------------------------------------------------------------------
# Python execution tool
# ------------------------------------------------------------------

def _tool_execute_python(
    tool_input: dict,
    observations: list[dict],
    embeddings: dict[str, list[float]] | None = None,
) -> str:
    """Execute arbitrary Python code in a sandboxed subprocess.

    The code starts with `obs` (the full observation list) pre-loaded
    plus standard scientific Python imports. stdout is captured and
    returned. Errors and tracebacks are returned in the result so the
    model can debug.

    If ``embeddings`` is passed, it's exposed to the subprocess as:
      - ``embeddings``: ``{post_hash: [1536 floats]}`` raw dict
      - ``emb_matrix``: numpy array of shape (N, 1536), rows aligned to
        ``emb_hashes`` (list of post_hash strings)
      - ``emb_by_obs``: list of embedding vectors in the same order as
        ``obs`` (or ``None`` where the post has no embedding)
    """
    code = tool_input.get("code", "")
    if not code:
        return json.dumps({"error": "No code provided"})

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as obs_file:
        json.dump(observations, obs_file, default=str)
        obs_path = obs_file.name

    emb_path: str | None = None
    if embeddings:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as emb_file:
            json.dump(embeddings, emb_file)
            emb_path = emb_file.name

    preamble = f"""
import json as _json
import sys as _sys
import math
import statistics
from collections import Counter, defaultdict

_IMPORT_ERRORS = []
try:
    import numpy as np
except ImportError as _e:
    _IMPORT_ERRORS.append(f"numpy: {{_e}}")
    np = None
try:
    import scipy
    import scipy.stats
    from scipy import stats
except ImportError as _e:
    _IMPORT_ERRORS.append(f"scipy: {{_e}}")
    scipy = None
    stats = None
try:
    import sklearn
    from sklearn import linear_model, model_selection, preprocessing, decomposition, cluster
except ImportError as _e:
    _IMPORT_ERRORS.append(f"sklearn: {{_e}}")
    sklearn = None
try:
    import pandas as pd
except ImportError as _e:
    _IMPORT_ERRORS.append(f"pandas: {{_e}}")
    pd = None

if _IMPORT_ERRORS:
    print(f"WARNING: some scientific packages unavailable: {{_IMPORT_ERRORS}}", file=_sys.stderr)

with open({obs_path!r}, 'r', encoding='utf-8') as _f:
    obs = _json.load(_f)

embeddings = {{}}
emb_matrix = None
emb_hashes = []
emb_by_obs = []
_EMB_PATH = {emb_path!r}
if _EMB_PATH:
    try:
        with open(_EMB_PATH, 'r', encoding='utf-8') as _f:
            embeddings = _json.load(_f)
        emb_hashes = list(embeddings.keys())
        if np is not None and emb_hashes:
            emb_matrix = np.array([embeddings[h] for h in emb_hashes], dtype=float)
        emb_by_obs = [embeddings.get(o.get('post_hash')) for o in obs]
    except Exception as _e:
        print(f"WARNING: embedding load failed: {{_e}}", file=_sys.stderr)
"""
    full_code = preamble + "\n# --- user code below ---\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as code_file:
        code_file.write(full_code)
        code_path = code_file.name

    try:
        proc = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=_PYTHON_TIMEOUT_SECONDS,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "PYTHONUNBUFFERED": "1",
            },
        )
        stdout = proc.stdout
        stderr = proc.stderr
        if len(stdout) > 8000:
            stdout = stdout[:8000] + "\n... [stdout truncated at 8000 chars — print fewer details or summarize]"
        if len(stderr) > 2000:
            stderr = stderr[:2000] + "\n... [stderr truncated at 2000 chars]"
        return json.dumps({
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"Code execution timed out after {_PYTHON_TIMEOUT_SECONDS}s",
        })
    except Exception as e:
        return json.dumps({"error": f"Subprocess failed: {e}"})
    finally:
        for _p in (obs_path, code_path, emb_path):
            if _p:
                try:
                    os.unlink(_p)
                except Exception:
                    pass


# ------------------------------------------------------------------
# LinkedIn post bank search tool
# ------------------------------------------------------------------

def _tool_search_linkedin_bank(tool_input: dict) -> str:
    """Search the LinkedIn post bank — keyword or semantic mode."""
    query = (tool_input.get("query") or "").strip()
    mode = tool_input.get("mode", "keyword")
    limit = min(tool_input.get("limit", 20), 100)

    if not query:
        return json.dumps({"error": "Query is required"})

    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_KEY", "")

    if mode == "semantic":
        return _search_semantic(query, limit)
    elif mode == "keyword":
        return _search_keyword(query, limit, sb_url, sb_key)
    else:
        return json.dumps({"error": f"Unknown mode: {mode}. Use 'keyword' or 'semantic'."})


def _search_keyword(query: str, limit: int, sb_url: str, sb_key: str) -> str:
    """Keyword search via Supabase ilike on linkedin_posts.

    Strategy: pick the longest keyword in the query, ilike on post_text only
    (PostgREST `or=` with two ilike columns + ordering blows past the 30s
    statement timeout on the 200K-row table), then intersect remaining
    keywords client-side.
    """
    if not sb_url or not sb_key:
        return json.dumps({"error": "Supabase not configured"})
    keywords = [w.strip() for w in (query or "").split() if len(w.strip()) >= 3]
    if not keywords:
        return json.dumps({"error": "Query too short — use 3+ character words"})
    keywords.sort(key=len, reverse=True)
    primary = keywords[0]
    try:
        import httpx
        r = httpx.get(
            f"{sb_url}/rest/v1/linkedin_posts",
            params={
                "select": "provider_urn,creator_username,hook,post_text,total_reactions,total_comments,engagement_score,posted_at",
                "post_text": f"ilike.*{primary}*",
                "is_company_post": "eq.false",
                "engagement_score": "gt.0",
                "order": "engagement_score.desc",
                "limit": str(max(limit * 5, 80)),
            },
            headers={"apikey": sb_key, "Authorization": f"Bearer {sb_key}"},
            timeout=30.0,
        )
        r.raise_for_status()
        rows = r.json()
        results = []
        for row in rows:
            text = (row.get("post_text") or "").lower()
            if all(kw.lower() in text for kw in keywords):
                results.append({
                    "creator": row.get("creator_username", ""),
                    "hook": (row.get("hook") or "")[:200],
                    "post_text": (row.get("post_text") or "")[:1500],
                    "reactions": row.get("total_reactions", 0),
                    "comments": row.get("total_comments", 0),
                    "engagement": row.get("engagement_score", 0),
                    "posted_at": row.get("posted_at", "")[:10] if row.get("posted_at") else "",
                })
            if len(results) >= limit:
                break
        return json.dumps({"mode": "keyword", "query": query, "count": len(results), "results": results})
    except Exception as e:
        return json.dumps({"error": f"Keyword search failed: {str(e)[:200]}"})


def _search_semantic(query: str, limit: int) -> str:
    """Semantic search via Pinecone vector index."""
    pc_key = os.environ.get("PINECONE_API_KEY", "")
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    sb_url = os.environ.get("SUPABASE_URL", "")
    sb_key = os.environ.get("SUPABASE_KEY", "")

    if not pc_key or not oai_key:
        return json.dumps({"error": "Pinecone or OpenAI not configured"})

    try:
        from openai import OpenAI
        from pinecone import Pinecone
    except ImportError:
        return json.dumps({"error": "openai/pinecone packages not installed"})

    try:
        oai = OpenAI(api_key=oai_key)
        emb = oai.embeddings.create(input=query, model="text-embedding-3-small")
        query_vec = emb.data[0].embedding

        pc = Pinecone(api_key=pc_key)
        idx = pc.Index("linkedin-posts")
        results = idx.query(
            vector=query_vec,
            top_k=limit,
            namespace="v2",
            include_metadata=True,
        )
        matches = results.get("matches", [])
        if not matches:
            return json.dumps({"mode": "semantic", "query": query, "count": 0, "results": []})

        posts_by_urn: dict[str, dict] = {}
        if sb_url and sb_key:
            import httpx
            urns = [m["id"] for m in matches]
            urn_filter = ",".join(f'"{u}"' for u in urns)
            try:
                r = httpx.get(
                    f"{sb_url}/rest/v1/linkedin_posts",
                    params={
                        "select": "provider_urn,creator_username,hook,post_text,total_reactions,total_comments,engagement_score,posted_at",
                        "provider_urn": f"in.({urn_filter})",
                    },
                    headers={"apikey": sb_key, "Authorization": f"Bearer {sb_key}"},
                    timeout=30.0,
                )
                r.raise_for_status()
                for row in r.json():
                    if row.get("provider_urn"):
                        posts_by_urn[row["provider_urn"]] = row
            except Exception:
                pass

        results_out = []
        for m in matches:
            urn = m["id"]
            post = posts_by_urn.get(urn, {})
            meta = m.get("metadata", {})
            results_out.append({
                "creator": post.get("creator_username") or meta.get("creator_username", ""),
                "hook": (post.get("hook") or "")[:200],
                "post_text": (post.get("post_text") or meta.get("text", "") or "")[:1500],
                "reactions": post.get("total_reactions") or meta.get("total_reactions", 0),
                "comments": post.get("total_comments") or meta.get("total_comments", 0),
                "engagement": post.get("engagement_score", 0),
                "score": round(m.get("score", 0), 3),
            })
        return json.dumps({"mode": "semantic", "query": query, "count": len(results_out), "results": results_out})
    except Exception as e:
        return json.dumps({"error": f"Semantic search failed: {str(e)[:200]}"})
