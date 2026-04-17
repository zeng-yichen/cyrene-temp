"""Minimal MCP server framework — JSON-RPC 2.0 over stdio.

No external dependencies. Implements just the two methods that Claude CLI
needs to use custom tools:

  - tools/list  → returns the tool schemas
  - tools/call  → dispatches to a Python handler and returns the result

Usage:

    from backend.src.mcp_bridge.server import MCPServer

    server = MCPServer("my-tools")
    server.tool(
        name="search_posts",
        description="Search past posts",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )(my_handler_function)

    # In __main__:
    server.run()  # blocks, reads stdin, writes stdout
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable


class MCPServer:
    """Stdio-based MCP server that exposes Python functions as tools."""

    def __init__(self, name: str = "amphoreus-tools"):
        self.name = name
        self._tools: list[dict[str, Any]] = []
        self._handlers: dict[str, Callable[..., str]] = {}

    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> Callable:
        """Decorator / registrar for a tool handler.

        The handler receives (arguments: dict) and must return a string.
        """
        schema = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
        }
        self._tools.append(schema)

        def decorator(fn: Callable[..., str]) -> Callable[..., str]:
            self._handlers[name] = fn
            return fn

        return decorator

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., str],
    ) -> None:
        """Imperative registration (no decorator)."""
        self._tools.append({
            "name": name,
            "description": description,
            "inputSchema": input_schema,
        })
        self._handlers[name] = handler

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------

    def _handle_request(self, req: dict) -> dict | None:
        method = req.get("method", "")
        req_id = req.get("id")
        params = req.get("params", {})

        if method == "initialize":
            return self._respond(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": "0.1.0"},
            })

        if method == "notifications/initialized":
            # Client ack — no response needed
            return None

        if method == "ping":
            return self._respond(req_id, {})

        if method == "tools/list":
            return self._respond(req_id, {"tools": self._tools})

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            handler = self._handlers.get(tool_name)
            if handler is None:
                return self._respond(req_id, {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                })
            try:
                result = handler(arguments)
                return self._respond(req_id, {
                    "content": [{"type": "text", "text": result}],
                    "isError": False,
                })
            except Exception as e:
                return self._respond(req_id, {
                    "content": [{"type": "text", "text": f"Error: {type(e).__name__}: {e}"}],
                    "isError": True,
                })

        # Unknown method — return error
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    @staticmethod
    def _respond(req_id: Any, result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block on stdin, dispatch JSON-RPC, write to stdout.

        Reads one JSON-RPC message per line (newline-delimited).
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue

            resp = self._handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
