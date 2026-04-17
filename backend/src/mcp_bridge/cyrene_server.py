#!/usr/bin/env python3
"""MCP server exposing Cyrene's custom tools for Claude CLI.

Launched as a subprocess by Claude CLI via --mcp-config. Reuses the tool
implementations from backend.src.agents.cyrene so there's a single source
of truth for what each tool does.

Stateful per run:
  - notes: appended via the `note` tool, returned in the final brief
  - brief: captured when submit_brief is called, serialised to
    .cyrene_cli_result.json for the runner to read

Reads config from environment variables:
  CYRENE_COMPANY — client slug (required)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.src.mcp_bridge.server import MCPServer
from backend.src.agents.cyrene import (
    _query_observations,
    _query_top_engagers,
    _query_transcript_inventory,
    _query_icp_exposure_trend,
    _query_warm_prospects,
    _query_engagement_trajectories,
    _execute_python,
    _search_linkedin_corpus,
    _web_search,
    _fetch_url,
    _query_ordinal_posts,
    _TOOL_SCHEMAS as _CYRENE_TOOL_SCHEMAS,
)

_COMPANY = os.environ.get("CYRENE_COMPANY", "")

# Per-run state
_notes: list[str] = []
_brief: dict[str, Any] | None = None


server = MCPServer("cyrene-tools")


def _register(name: str, handler_fn):
    """Pull the schema from Cyrene's _TOOL_SCHEMAS list so the server stays in sync."""
    schema = next((t for t in _CYRENE_TOOL_SCHEMAS if t["name"] == name), None)
    if schema is None:
        raise RuntimeError(f"Cyrene tool schema not found: {name}")
    server.register(
        name=name,
        description=schema["description"],
        input_schema=schema["input_schema"],
        handler=handler_fn,
    )


# ---------------------------------------------------------------------------
# Query tools — thin wrappers that inject _COMPANY
# ---------------------------------------------------------------------------

def _wrap(fn):
    def handler(args: dict) -> str:
        if not _COMPANY:
            return json.dumps({"error": "CYRENE_COMPANY not set"})
        return fn(_COMPANY, args or {})
    return handler


_register("pull_history", _wrap(_query_observations))
_register("pull_reactors", _wrap(_query_top_engagers))
_register("query_transcript_inventory", _wrap(_query_transcript_inventory))
_register("query_icp_exposure_trend", _wrap(_query_icp_exposure_trend))
_register("query_warm_prospects", _wrap(_query_warm_prospects))
_register("query_engagement_trajectories", _wrap(_query_engagement_trajectories))
_register("run_py", _wrap(_execute_python))
_register("scan_corpus", _wrap(_search_linkedin_corpus))
_register("net_query", _wrap(_web_search))
_register("pull_page", _wrap(_fetch_url))
_register("query_ordinal_posts", _wrap(_query_ordinal_posts))


# ---------------------------------------------------------------------------
# Stateful tools
# ---------------------------------------------------------------------------

def _handle_note(args: dict) -> str:
    text = (args.get("text") or "").strip()
    if text:
        _notes.append(text)
    return json.dumps({"ok": True, "note_count": len(_notes)})


_register("note", _handle_note)


def _handle_submit_brief(args: dict) -> str:
    """Terminal tool. Captures the brief and writes it to the result file."""
    global _brief
    if not isinstance(args, dict):
        return json.dumps({"_error": "submit_brief requires object input"})

    brief = dict(args)
    # Attach accumulated notes so run_cyrene_cli can include them
    brief["_notes"] = list(_notes)
    _brief = brief

    result_path = os.path.join(_PROJECT_ROOT, ".cyrene_cli_result.json")
    try:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(brief, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"_error": f"failed to write result file: {e}"})

    return json.dumps({
        "accepted": True,
        "result_path": result_path,
        "n_notes": len(_notes),
    })


_register("submit_brief", _handle_submit_brief)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _COMPANY:
        print("Error: CYRENE_COMPANY env var not set", file=sys.stderr)
        sys.exit(1)
    server.run()
