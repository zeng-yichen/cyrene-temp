#!/usr/bin/env python3
"""MCP server exposing Irontomb's retrieval tools.

Launched as a subprocess by the Claude CLI via --mcp-config.
Reads the company keyword from the IRONTOMB_COMPANY env var
(set by the caller before spawning).

Tools exposed:
  - search_past_posts(query, limit)
  - get_recent_posts(limit)
  - get_post_detail(ordinal_post_id)
  - scan_corpus(query, mode, limit)
  - submit_reaction(...) — terminal tool, returns the prediction as-is

The submit_reaction tool doesn't do anything server-side — it just
echoes back the arguments. The Claude CLI model calls it when it's
ready to submit its prediction, and the caller parses the result
from the CLI output.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is on sys.path so imports work when launched
# as a standalone subprocess.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.src.mcp_bridge.server import MCPServer

_COMPANY = os.environ.get("IRONTOMB_COMPANY", "")

server = MCPServer("irontomb-tools")


# ---------------------------------------------------------------------------
# Tool: search_past_posts
# ---------------------------------------------------------------------------
server.register(
    name="search_past_posts",
    description=(
        "Keyword search over this client's scored post history. "
        "Returns posts ranked by relevance to the query, each with "
        "real engagement metrics (reactions, impressions, comments, "
        "reposts), the draft text, published text, and reactor list."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results (1-10, default 3)"},
        },
        "required": ["query"],
    },
    handler=lambda args: _dispatch("search_past_posts", args),
)


# ---------------------------------------------------------------------------
# Tool: get_recent_posts
# ---------------------------------------------------------------------------
server.register(
    name="get_recent_posts",
    description=(
        "Get the N most recent scored posts from this client. "
        "Returns the same fields as search_past_posts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "How many posts (1-10, default 3)"},
        },
    },
    handler=lambda args: _dispatch("get_recent_posts", args),
)


# ---------------------------------------------------------------------------
# Tool: get_post_detail
# ---------------------------------------------------------------------------
server.register(
    name="get_post_detail",
    description=(
        "Get the full untruncated text + complete reactor list for "
        "one specific post, identified by ordinal_post_id."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ordinal_post_id": {"type": "string"},
        },
        "required": ["ordinal_post_id"],
    },
    handler=lambda args: _dispatch("get_post_detail", args),
)


# ---------------------------------------------------------------------------
# Tool: scan_corpus
# ---------------------------------------------------------------------------
server.register(
    name="scan_corpus",
    description=(
        "Search 200K+ real LinkedIn posts from creators across all "
        "industries. See what performs well LinkedIn-wide for a given "
        "topic or angle. Supports keyword and semantic modes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["keyword", "semantic"],
                "description": "Search mode (default: keyword)",
            },
            "limit": {"type": "integer", "description": "Max results (1-20, default 10)"},
        },
        "required": ["query"],
    },
    handler=lambda args: _dispatch("scan_corpus", args),
)


# ---------------------------------------------------------------------------
# Tool: submit_reaction (terminal)
# ---------------------------------------------------------------------------
server.register(
    name="submit_reaction",
    description=(
        "Submit your general-audience prediction for a draft LinkedIn post. "
        "Six fields + one optional debug field. No prose critique, no "
        "fix suggestions. You are predicting how the BROAD LinkedIn "
        "audience will react, not a narrow target segment."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "engagement_prediction": {
                "type": "number",
                "description": (
                    "Predicted reactions per 1000 impressions for a post "
                    "like this one from this client."
                ),
            },
            "impression_prediction": {
                "type": "integer",
                "description": "Predicted total impressions this post would receive.",
            },
            "would_stop_scrolling": {
                "type": "boolean",
                "description": "Would a typical LinkedIn user stop and read this?",
            },
            "would_react": {"type": "boolean", "description": "Would they tap a reaction?"},
            "would_comment": {"type": "boolean", "description": "Would they comment?"},
            "would_share": {"type": "boolean", "description": "Would they share/repost?"},
            "inner_voice": {
                "type": "string",
                "description": "Optional: one-sentence gut reaction from a reader's perspective.",
            },
        },
        "required": [
            "engagement_prediction",
            "impression_prediction",
            "would_stop_scrolling",
            "would_react",
            "would_comment",
            "would_share",
        ],
    },
    # submit_reaction just echoes back — the caller parses it from the output
    handler=lambda args: json.dumps({"submitted": True, **args}),
)


# ---------------------------------------------------------------------------
# Dispatcher — delegates to Irontomb's existing _dispatch_retrieval_tool
# ---------------------------------------------------------------------------

def _dispatch(tool_name: str, args: dict) -> str:
    from backend.src.agents.irontomb import _dispatch_retrieval_tool
    return _dispatch_retrieval_tool(_COMPANY, tool_name, args)


if __name__ == "__main__":
    if not _COMPANY:
        print("Error: IRONTOMB_COMPANY env var not set", file=sys.stderr)
        sys.exit(1)
    server.run()
