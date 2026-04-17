"""MCP bridge — run Amphoreus agents through Claude CLI instead of the API.

This module provides:
  - A minimal MCP (Model Context Protocol) server framework that exposes
    Python tool handlers as JSON-RPC tools over stdio
  - Pre-built MCP servers for each agent's toolset (Irontomb, Cyrene, Stelle)
  - A drop-in adapter that replaces anthropic.Anthropic().messages.create()
    with `claude -p` CLI invocations using the MCP servers

The goal: run generation through a Claude Max subscription instead of
paying per-token API costs, while making ZERO changes to the existing
agent code. The existing API path remains the default; this is an
opt-in alternative activated by setting AMPHOREUS_USE_CLI=1.
"""
