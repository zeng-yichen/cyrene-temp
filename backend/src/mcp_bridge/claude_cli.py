"""Claude CLI adapter — run agent loops through `claude` instead of the API.

Drop-in replacements for the API-based agent entry points that invoke
the Claude CLI with MCP servers for tool access. Uses the user's
Claude Max subscription instead of per-token API billing.

ZERO changes to existing agent code. The adapter builds the same
system prompt + user message, writes a temp MCP config, and invokes:

    claude -p "user message" \
        --system-prompt-file /tmp/system.txt \
        --mcp-config /tmp/mcp.json \
        --strict-mcp-config \
        --output-format json \
        --model opus \
        --permission-mode bypassPermissions \
        --max-turns 8

Then parses the JSON output to extract the submit_reaction call.

Activate by setting AMPHOREUS_USE_CLI=1 in the environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _cli_env() -> dict:
    """Return the env dict used for every `claude` subprocess we spawn.

    Honours ``AMPHOREUS_CLAUDE_CONFIG_DIR`` so the production pipeline can
    use a separate Max account from the developer's interactive Claude
    Code sessions. The variable is forwarded to the CLI as
    ``CLAUDE_CONFIG_DIR``, which the CLI uses to locate OAuth credentials,
    settings, and session history. If unset, the subprocess inherits the
    parent's environment unchanged (i.e. uses ``~/.claude/`` like normal),
    so this is backward-compatible.

    One-time setup for the dedicated account::

        mkdir -p ~/.claude-amphoreus
        CLAUDE_CONFIG_DIR=~/.claude-amphoreus claude login
        export AMPHOREUS_CLAUDE_CONFIG_DIR=~/.claude-amphoreus  # in backend env
    """
    env = os.environ.copy()
    # Strip ANTHROPIC_API_KEY so the CLI uses OAuth (Max plan) instead of
    # falling back to API-key billing. The backend needs the key for its
    # own direct Anthropic SDK calls, but CLI subprocesses must authenticate
    # via the Max plan's OAuth token in the keychain / CLAUDE_CONFIG_DIR.
    env.pop("ANTHROPIC_API_KEY", None)
    override = os.environ.get("AMPHOREUS_CLAUDE_CONFIG_DIR", "").strip()
    if override:
        env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(override)
    return env


# ---------------------------------------------------------------------------
# Usage tracking (equivalent cost, not real spend)
# ---------------------------------------------------------------------------

# Map CLI --model shorthand to the billing-grade model id used by price_call.
# "opus" / "sonnet" / "haiku" are the CLI's aliases; they resolve to whatever
# the CLI considers the current default for that tier. We bill against our
# current canonical id for that tier.
_CLI_MODEL_TO_BILLING = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


def _parse_cli_usage(stdout: str) -> dict | None:
    """Extract token usage from CLI stream-json or json stdout.

    The CLI emits a terminal `result` event (stream-json) or a single
    result object (json) with a ``usage`` block shaped like:

        {"input_tokens": N, "output_tokens": N,
         "cache_creation_input_tokens": N, "cache_read_input_tokens": N}

    Returns None if no usage info is parseable. Never raises.
    """
    if not stdout:
        return None

    usage = None
    try:
        obj = json.loads(stdout.strip())
        if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
    except json.JSONDecodeError:
        pass

    if usage is None:
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "result" and isinstance(event.get("usage"), dict):
                usage = event["usage"]

    if not usage:
        return None

    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_creation_tokens": int(
            usage.get("cache_creation_input_tokens")
            or usage.get("cache_creation_tokens")
            or 0
        ),
        "cache_read_tokens": int(
            usage.get("cache_read_input_tokens")
            or usage.get("cache_read_tokens")
            or 0
        ),
    }


def _record_cli_usage(
    *,
    stdout: str,
    cli_model: str,
    call_kind: str,
    client_slug: str | None = None,
    duration_ms: int | None = None,
    error: str | None = None,
) -> None:
    """Parse CLI stdout for usage and insert an equivalent-cost row.

    Writes with provider='anthropic_cli' so the admin dashboard can
    distinguish zero-dollar Max-plan usage from actual API spend. Cost
    is computed as if the same tokens had gone through the API, so ops
    can see "savings from CLI = sum(anthropic_cli.cost_usd)".
    """
    try:
        usage = _parse_cli_usage(stdout)
        if usage is None:
            logger.debug("[CLI-usage] No usage block in stdout for %s", call_kind)
            return

        billing_model = _CLI_MODEL_TO_BILLING.get(cli_model, cli_model)

        from backend.src.usage.recorder import record_usage_event
        record_usage_event(
            provider="anthropic_cli",
            model=billing_model,
            call_kind="messages",
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cache_creation_tokens=usage["cache_creation_tokens"],
            cache_read_tokens=usage["cache_read_tokens"],
            client_slug=client_slug,
            duration_ms=duration_ms,
            error=error,
        )
    except Exception as e:
        logger.warning("[CLI-usage] Failed to record usage for %s: %s", call_kind, e)


# ---------------------------------------------------------------------------
# Irontomb via CLI
# ---------------------------------------------------------------------------

def simulate_flame_chase_journey_cli(
    company: str,
    draft_text: str,
) -> dict[str, Any]:
    """Drop-in replacement for irontomb.sim_audience().

    Builds the same system prompt and user message, then runs the
    simulation through `claude` CLI with the Irontomb MCP server
    instead of the Anthropic API.
    """
    from backend.src.agents.irontomb import (
        _build_system_prompt,
        _draft_hash,
        _format_calibration_block,
        _format_cross_client_block,
        _load_audience_context,
        _load_scored_observations,
        _IRONTOMB_MAX_TURNS,
    )

    draft_text = (draft_text or "").strip()
    if not draft_text:
        return {"_error": "draft_text is required"}

    t0 = time.time()

    # Build the same context that the API path uses
    audience_context = _load_audience_context(company)
    observations = _load_scored_observations(company)
    calibration_block = _format_calibration_block(observations)
    cross_client_block = _format_cross_client_block(company)
    system_prompt = _build_system_prompt(
        audience_context,
        n_scored_obs=len(observations),
        calibration_block=calibration_block,
        cross_client_block=cross_client_block,
    )

    user_message = (
        "Here is the draft LinkedIn post you are evaluating. "
        "You've already read the calibration examples from this "
        "client's real history above. Predict how this specific "
        "audience will react to THIS draft, anchored in what you "
        "saw happen to comparable past posts. If the draft is in "
        "territory the examples don't cover, retrieve more "
        "comparables first; otherwise submit_reaction directly.\n\n"
        "=== DRAFT ===\n"
        f"{draft_text}\n"
        "=== END DRAFT ==="
    )

    # Write temp files for the CLI
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="irontomb_sys_"
    ) as f:
        f.write(system_prompt)
        system_prompt_file = f.name

    mcp_config = {
        "mcpServers": {
            "irontomb-tools": {
                "command": "python3",
                "args": [
                    str(_PROJECT_ROOT / "backend" / "src" / "mcp_bridge" / "irontomb_server.py"),
                ],
                "env": {
                    "IRONTOMB_COMPANY": company,
                    # Forward relevant env vars
                    "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
                    "SUPABASE_KEY": os.environ.get("SUPABASE_KEY", ""),
                    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                    "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY", ""),
                },
            }
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="irontomb_mcp_"
    ) as f:
        json.dump(mcp_config, f)
        mcp_config_file = f.name

    try:
        cmd = [
            "claude",
            "-p", user_message,
            "--system-prompt-file", system_prompt_file,
            "--mcp-config", mcp_config_file,
            "--strict-mcp-config",
            "--output-format", "stream-json",
            "--verbose",
            "--model", "opus",
            "--permission-mode", "bypassPermissions",
            "--max-turns", str(_IRONTOMB_MAX_TURNS),
            # NOTE: do NOT use --bare here. --bare disables OAuth and
            # falls back to API key auth, bypassing the Max plan.
        ]

        logger.info("[CLI] Running Irontomb simulation for %s via claude CLI...", company)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
            cwd=str(_PROJECT_ROOT),
            env=_cli_env(),
        )

        elapsed = time.time() - t0
        dh = _draft_hash(draft_text)

        _record_cli_usage(
            stdout=proc.stdout or "",
            cli_model="opus",
            call_kind="irontomb_cli",
            client_slug=company,
            duration_ms=int(elapsed * 1000),
            error=None if proc.returncode == 0 else f"exit {proc.returncode}",
        )

        if proc.returncode != 0:
            logger.warning("[CLI] claude exited %d: %s", proc.returncode, proc.stderr[:500])
            return {
                "_error": f"claude CLI exited {proc.returncode}: {proc.stderr[:200]}",
                "_draft_hash": dh,
                "_elapsed_s": round(elapsed, 1),
                "_via": "cli",
            }

        # Parse stream-json output to find submit_reaction tool call
        reaction = _extract_reaction_from_stream(proc.stdout)

        if reaction is None:
            # Fallback: try single JSON format
            reaction = _extract_reaction_from_json(proc.stdout)

        if reaction is None:
            logger.warning("[CLI] Could not extract reaction from output")
            return {
                "_error": "No submit_reaction found in CLI output",
                "_draft_hash": dh,
                "_elapsed_s": round(elapsed, 1),
                "_via": "cli",
                "_raw_stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            }

        # Build the same return shape as the API path
        result = {
            "engagement_prediction": reaction.get("engagement_prediction"),
            "impression_prediction": reaction.get("impression_prediction"),
            "would_stop_scrolling": reaction.get("would_stop_scrolling"),
            "would_react": reaction.get("would_react"),
            "would_comment": reaction.get("would_comment"),
            "would_share": reaction.get("would_share"),
            "inner_voice": reaction.get("inner_voice", ""),
            "_draft_hash": dh,
            "_via": "cli",
            "_elapsed_s": round(elapsed, 1),
            "_cost_usd": 0.0,  # Max plan — no per-token cost
        }
        return result

    except subprocess.TimeoutExpired:
        return {
            "_error": "claude CLI timed out after 300s",
            "_draft_hash": _draft_hash(draft_text),
            "_via": "cli",
        }
    except FileNotFoundError:
        return {
            "_error": "claude CLI not found — is Claude Code installed?",
            "_draft_hash": _draft_hash(draft_text),
            "_via": "cli",
        }
    finally:
        # Clean up temp files
        for f in (system_prompt_file, mcp_config_file):
            try:
                os.unlink(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _extract_reaction_from_stream(stdout: str) -> Optional[dict]:
    """Parse stream-json output to find submit_reaction tool call.

    The Claude CLI prefixes MCP tool names with the server name:
    e.g. "mcp__irontomb-tools__submit_reaction". We match on the
    suffix "submit_reaction" to handle this.
    """
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Look for tool_use blocks in assistant messages
        if event.get("type") == "assistant":
            message = event.get("message", {})
            for block in message.get("content", []):
                if (
                    block.get("type") == "tool_use"
                    and (block.get("name", "").endswith("submit_reaction"))
                ):
                    return block.get("input", {})

        # Also check content_block_start events
        if event.get("type") == "content_block_start":
            block = event.get("content_block", {})
            if (
                block.get("type") == "tool_use"
                and (block.get("name", "").endswith("submit_reaction"))
            ):
                return block.get("input", {})

    return None


def _extract_reaction_from_json(stdout: str) -> Optional[dict]:
    """Parse CLI JSON output and extract submit_reaction arguments.

    The CLI's JSON output has a `result` field with the model's final
    text. But the submit_reaction tool call happened during the tool
    loop — its arguments were passed to the MCP server which echoed
    them back. The model's `result` text often summarizes its prediction.

    Strategy: look for engagement_prediction in the result text
    (the MCP server echoes back the submitted fields as JSON).
    """
    try:
        cli_result = json.loads(stdout.strip())
    except json.JSONDecodeError:
        return None

    result_text = cli_result.get("result", "")

    # Try to find a JSON object with engagement_prediction in the text
    # The model might include it inline or the tool result echoed it
    import re
    # Look for JSON-like blocks in the result
    for match in re.finditer(r'\{[^{}]*"engagement_prediction"[^{}]*\}', result_text):
        try:
            data = json.loads(match.group())
            if "engagement_prediction" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Try parsing the entire result as JSON
    try:
        data = json.loads(result_text)
        if "engagement_prediction" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # The submit_reaction tool echoes {"submitted": true, ...} back.
    # The model sees this and might report the numbers in prose.
    # Try to extract from prose patterns.
    eng_match = re.search(r'engagement_prediction["\s:]+(\d+\.?\d*)', result_text)
    imp_match = re.search(r'impression_prediction["\s:]+(\d+)', result_text)
    stop_match = re.search(r'would_stop_scrolling["\s:]+(\w+)', result_text)

    if eng_match:
        return {
            "engagement_prediction": float(eng_match.group(1)),
            "impression_prediction": int(imp_match.group(1)) if imp_match else 0,
            "would_stop_scrolling": stop_match.group(1).lower() == "true" if stop_match else True,
            "would_react": True,
            "would_comment": False,
            "would_share": False,
            "_parsed_from_prose": True,
        }

    return None


# ---------------------------------------------------------------------------
# Single-shot CLI call (for compaction, why-post, image suggestion, etc.)
# ---------------------------------------------------------------------------

def cli_single_shot(
    prompt: str,
    system_prompt: str | None = None,
    model: str = "sonnet",
    max_tokens: int = 4096,
    timeout: int = 60,
) -> str | None:
    """Run a single-shot Claude CLI call and return the text response.

    Drop-in replacement for:
        resp = client.messages.create(model=..., messages=[...])
        text = resp.content[0].text

    ``timeout`` is seconds to wait for the subprocess. Default 60s suits
    short helpers (why-post, image-suggestion). Pass a longer value for
    large-output calls like the progress-report HTML generator.
    """
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "json",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--tools", "",  # no tools needed for single-shot
    ]

    if system_prompt:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="cli_sys_"
        ) as f:
            f.write(system_prompt)
            sys_file = f.name
        cmd.extend(["--system-prompt-file", sys_file])
    else:
        sys_file = None

    try:
        _t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            cwd=str(_PROJECT_ROOT),
            env=_cli_env(),
        )
        _elapsed_ms = int((time.time() - _t0) * 1000)

        _record_cli_usage(
            stdout=proc.stdout or "",
            cli_model=model,
            call_kind="single_shot",
            duration_ms=_elapsed_ms,
            error=None if proc.returncode == 0 else f"exit {proc.returncode}",
        )

        if proc.returncode != 0:
            logger.warning("[CLI] single-shot failed: %s", proc.stderr[:200])
            return None

        result = json.loads(proc.stdout)
        return result.get("result", "")

    except Exception as e:
        logger.warning("[CLI] single-shot error: %s", e)
        return None
    finally:
        if sys_file:
            try:
                os.unlink(sys_file)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Stelle via CLI
# ---------------------------------------------------------------------------

def run_stelle_cli(
    client_name: str,
    company_keyword: str,
    output_filepath: str,
    num_posts: int = 12,
    prompt: str | None = None,
    event_callback: Any = None,
) -> str:
    """Run Stelle's full generation loop through Claude CLI.

    Drop-in replacement for the API-based run_writer_batch().
    Builds the same system prompt and user message, sets up the
    workspace, launches `claude -p` with the Stelle MCP server
    for custom tools (the CLI provides filesystem/web tools natively),
    then reads the result from .stelle_cli_result.json and runs
    _finalize_run() for fact-checking and output generation.
    """
    import shutil
    from pathlib import Path as _Path

    # --- Lazy imports from stelle.py (avoid circular at module level) ---
    from backend.src.agents.stelle import (
        _assemble_runtime_directives,
        _PRIMARY_INSTRUCTION_TEMPLATE,
        _collect_published_hooks,
        _finalize_run,
        _lookup_user_ids,
        _stage_env,
        _check_submission,
        AGENT_TURN_LIMIT,
        P,
    )

    # Resolve display name
    username_path = P.linkedin_username_path(company_keyword)
    if not username_path.exists():
        raise FileNotFoundError(
            f"Missing memory/{company_keyword}/linkedin_username.txt"
        )
    username = username_path.read_text().strip()
    if username:
        _, _, display_name = _lookup_user_ids(username)
        if display_name:
            client_name = display_name

    logger.info("[CLI-Stelle] Starting CLI-based generation for %s...", client_name)

    # --- Setup (same as run_writer_batch) ---
    P.ensure_dirs(company_keyword)

    try:
        from backend.src.db.local import purge_unpushed_drafts as _purge_drafts
        _purged = _purge_drafts(company_keyword)
        if _purged:
            logger.info("[CLI-Stelle] Purged %d unpushed draft(s)", _purged)
    except Exception as _e:
        logger.warning("[CLI-Stelle] Purge skipped: %s", _e)

    workspace_root = _stage_env(company_keyword)

    # --- Existing posts context (dedup) ---
    existing_posts_context = ""
    try:
        existing_posts_context = _collect_published_hooks(company_keyword)
    except Exception:
        pass

    # Series + scheduling context
    series_context = ""
    try:
        from backend.src.services.series_engine import get_stelle_series_context as _series_ctx
        series_context = _series_ctx(company_keyword)
    except Exception:
        pass

    scheduling_context = ""
    try:
        from backend.src.services.temporal_orchestrator import build_scheduling_context as _sched_ctx
        scheduling_context = _sched_ctx(company_keyword)
    except Exception:
        pass

    # --- Build user prompt ---
    base_prompt = (
        f"Write up to {num_posts} LinkedIn posts for {client_name}. "
        f"The transcripts are from content interviews — conversations designed "
        f"to surface post material. Mine them for everything worth writing about. "
        f"Only write as many posts as the transcripts can genuinely support with "
        f"distinct insights — if the material supports 7, write 7, not {num_posts}. "
        f"Quality and distinctness over quantity."
    )
    if prompt:
        user_prompt = f"{base_prompt}\n\nAdditional instructions from the user:\n{prompt}"
    else:
        user_prompt = base_prompt
    if existing_posts_context:
        user_prompt += existing_posts_context
    if scheduling_context:
        user_prompt += scheduling_context
    if series_context:
        user_prompt += series_context

    # --- Build system prompt ---
    directives = _assemble_runtime_directives(company_keyword)
    system_prompt = _PRIMARY_INSTRUCTION_TEMPLATE.format(dynamic_directives=directives)

    # --- Write temp files ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="stelle_sys_"
    ) as f:
        f.write(system_prompt)
        system_prompt_file = f.name

    # MCP config: stelle-tools server for custom tools
    # The CLI provides filesystem (Read/Write/Edit/Bash/Grep/Glob) and
    # web (WebSearch/WebFetch) tools natively.
    env_vars = {
        "STELLE_COMPANY": company_keyword,
        "STELLE_USE_CLI_IRONTOMB": "1",
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_KEY": os.environ.get("SUPABASE_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY", ""),
        "SERPER_API_KEY": os.environ.get("SERPER_API_KEY", ""),
        "ORDINAL_API_KEY": os.environ.get("ORDINAL_API_KEY", ""),
    }
    # Filter out empty values
    env_vars = {k: v for k, v in env_vars.items() if v}

    mcp_config = {
        "mcpServers": {
            "stelle-tools": {
                "command": "python3",
                "args": [
                    str(_PROJECT_ROOT / "backend" / "src" / "mcp_bridge" / "stelle_server.py"),
                ],
                "env": env_vars,
            }
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="stelle_mcp_"
    ) as f:
        json.dump(mcp_config, f)
        mcp_config_file = f.name

    # Clear any previous result file
    result_file = _PROJECT_ROOT / ".stelle_cli_result.json"
    if result_file.exists():
        result_file.unlink()

    try:
        cmd = [
            "claude",
            "-p", user_prompt,
            "--system-prompt-file", system_prompt_file,
            "--mcp-config", mcp_config_file,
            "--output-format", "stream-json",
            "--verbose",
            "--model", "opus",
            "--permission-mode", "bypassPermissions",
            "--max-turns", str(AGENT_TURN_LIMIT),
            "--add-dir", str(workspace_root),
        ]

        logger.info("[CLI-Stelle] Launching claude CLI (max %d turns)...", AGENT_TURN_LIMIT)
        t0 = time.time()

        # Streaming subprocess: read stdout line-by-line so the web UI
        # gets live progress via event_callback. Drain stderr on a
        # background thread to avoid pipe-buffer deadlock.
        import threading
        TIMEOUT_SEC = 3600  # 60 min

        # cwd = workspace root so the agent's relative paths (memory/config.md
        # etc.) resolve correctly against the staged workspace layout.
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
                cwd=str(workspace_root),
                env=_cli_env(),
            )
        except FileNotFoundError:
            raise RuntimeError("claude CLI not found — is Claude Code installed?")

        stderr_chunks: list[str] = []
        def _drain_stderr() -> None:
            try:
                for line in iter(proc.stderr.readline, ""):
                    if not line:
                        break
                    stderr_chunks.append(line)
            except Exception:
                pass
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        if event_callback:
            event_callback("status", {"message": "Stelle CLI running (Max plan)..."})

        stdout_lines: list[str] = []
        timed_out = False
        try:
            for raw_line in iter(proc.stdout.readline, ""):
                if raw_line == "":
                    break
                stdout_lines.append(raw_line)
                if time.time() - t0 > TIMEOUT_SEC:
                    timed_out = True
                    proc.kill()
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _translate_cli_event(event, event_callback)

            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            stderr_thread.join(timeout=5)

        full_stdout = "".join(stdout_lines)
        full_stderr = "".join(stderr_chunks)
        returncode = proc.returncode
        elapsed = time.time() - t0
        logger.info("[CLI-Stelle] CLI finished in %.1fs (exit %s)", elapsed, returncode)

        _record_cli_usage(
            stdout=full_stdout,
            cli_model="opus",
            call_kind="stelle_cli",
            client_slug=company_keyword,
            duration_ms=int(elapsed * 1000),
            error=None if returncode == 0 else f"exit {returncode}",
        )

        if timed_out:
            raise RuntimeError(f"claude CLI timed out after {TIMEOUT_SEC}s")

        if returncode != 0:
            logger.error("[CLI-Stelle] CLI failed: %s", full_stderr[:500])
            raise RuntimeError(
                f"claude CLI exited {returncode}: {full_stderr[:300] or full_stdout[-300:]}"
            )

        # Read the result written by finalize_output handler
        if not result_file.exists():
            logger.error("[CLI-Stelle] No result file found at %s", result_file)
            logger.info("[CLI-Stelle] stdout tail: %s", full_stdout[-1000:])
            raise RuntimeError("Stelle CLI did not produce a result file")

        result = json.loads(result_file.read_text(encoding="utf-8"))
        logger.info(
            "[CLI-Stelle] Got result with %d posts. Running post-processing...",
            len(result.get("posts", [])),
        )

        passed, val_errors, val_warnings = _check_submission(result)
        if not passed:
            logger.warning("[CLI-Stelle] Output validation issues: %s", val_errors)

        output_path = _finalize_run(result, client_name, company_keyword, output_filepath)

        session_path = output_filepath.replace(".md", "_session.jsonl")
        _Path(session_path).parent.mkdir(parents=True, exist_ok=True)
        with open(session_path, "w", encoding="utf-8") as f:
            f.write(full_stdout)
        logger.info("[CLI-Stelle] Session log saved to %s", session_path)

        return output_path

    finally:
        for f in (system_prompt_file, mcp_config_file):
            try:
                os.unlink(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Cyrene via CLI
# ---------------------------------------------------------------------------

def run_cyrene_cli(company: str) -> dict[str, Any]:
    """Run Cyrene's strategic review through Claude CLI.

    Drop-in replacement for cyrene.run_strategic_review(). Launches
    `claude -p` with the cyrene-tools MCP server, lets it run the full
    tool-use loop, then reads the brief from .cyrene_cli_result.json
    and persists it to memory/{company}/cyrene_brief.json.

    Returns the brief dict on success; returns {"_error": ...} on
    failure (same shape as the API version's error path).
    """
    import threading
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    # Lazy imports to avoid circular
    from backend.src.agents.cyrene import _SYSTEM_PROMPT, _BRIEF_FILENAME
    from backend.src.agents.irontomb import _load_icp_context
    from backend.src.db import vortex as P

    logger.info("[CLI-Cyrene] Starting CLI-based strategic review for %s...", company)

    # --- Gather context, same as the API version ---
    try:
        client_context = _load_icp_context(company)
    except Exception:
        client_context = "(no client context found)"

    try:
        from backend.src.db.local import ruan_mei_load
        state = ruan_mei_load(company) or {}
        n_scored = sum(
            1 for o in state.get("observations", [])
            if o.get("status") in ("scored", "finalized")
        )
    except Exception:
        n_scored = 0

    previous_brief = "No previous brief exists. This is the first Cyrene run for this client."
    try:
        prev_path = P.memory_dir(company) / _BRIEF_FILENAME
        if prev_path.exists():
            prev_data = json.loads(prev_path.read_text(encoding="utf-8"))
            previous_brief = json.dumps(prev_data, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass

    system_prompt = _SYSTEM_PROMPT.format(
        client_context=client_context,
        n_scored=n_scored,
        company=company,
        previous_brief=previous_brief,
    )

    user_prompt = (
        f"Run a strategic review for {company}. Study the data "
        f"across your tools, form your strategy from evidence, "
        f"and produce a comprehensive brief via submit_brief."
    )

    # --- Write temp files ---
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="cyrene_sys_"
    ) as f:
        f.write(system_prompt)
        system_prompt_file = f.name

    env_vars = {
        "CYRENE_COMPANY": company,
        "SUPABASE_URL": os.environ.get("SUPABASE_URL", ""),
        "SUPABASE_KEY": os.environ.get("SUPABASE_KEY", ""),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY", ""),
        "PARALLEL_API_KEY": os.environ.get("PARALLEL_API_KEY", ""),
        "ORDINAL_API_KEY": os.environ.get("ORDINAL_API_KEY", ""),
    }
    env_vars = {k: v for k, v in env_vars.items() if v}

    mcp_config = {
        "mcpServers": {
            "cyrene-tools": {
                "command": "python3",
                "args": [
                    str(_PROJECT_ROOT / "backend" / "src" / "mcp_bridge" / "cyrene_server.py"),
                ],
                "env": env_vars,
            }
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="cyrene_mcp_"
    ) as f:
        json.dump(mcp_config, f)
        mcp_config_file = f.name

    result_file = _PROJECT_ROOT / ".cyrene_cli_result.json"
    if result_file.exists():
        result_file.unlink()

    # Pull max turns from Cyrene constants to stay in sync.
    from backend.src.agents.cyrene import _CYRENE_MAX_TURNS

    try:
        cmd = [
            "claude",
            "-p", user_prompt,
            "--system-prompt-file", system_prompt_file,
            "--mcp-config", mcp_config_file,
            "--output-format", "stream-json",
            "--verbose",
            "--model", "opus",
            "--permission-mode", "bypassPermissions",
            "--max-turns", str(_CYRENE_MAX_TURNS),
        ]

        logger.info("[CLI-Cyrene] Launching claude CLI (max %d turns)...", _CYRENE_MAX_TURNS)
        t0 = time.time()
        TIMEOUT_SEC = 3600  # Cyrene does 15-30 turns of deep analysis; 1 hour cap.

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(_PROJECT_ROOT),
                env=_cli_env(),
            )
        except FileNotFoundError:
            raise RuntimeError("claude CLI not found — is Claude Code installed?")

        stderr_chunks: list[str] = []
        def _drain_stderr() -> None:
            try:
                for line in iter(proc.stderr.readline, ""):
                    if not line:
                        break
                    stderr_chunks.append(line)
            except Exception:
                pass
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        stdout_lines: list[str] = []
        timed_out = False
        try:
            for raw_line in iter(proc.stdout.readline, ""):
                if raw_line == "":
                    break
                stdout_lines.append(raw_line)
                if time.time() - t0 > TIMEOUT_SEC:
                    timed_out = True
                    proc.kill()
                    break
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            stderr_thread.join(timeout=5)

        full_stdout = "".join(stdout_lines)
        full_stderr = "".join(stderr_chunks)
        returncode = proc.returncode
        elapsed = time.time() - t0
        logger.info("[CLI-Cyrene] CLI finished in %.1fs (exit %s)", elapsed, returncode)

        _record_cli_usage(
            stdout=full_stdout,
            cli_model="opus",
            call_kind="cyrene_cli",
            client_slug=company,
            duration_ms=int(elapsed * 1000),
            error=None if returncode == 0 else f"exit {returncode}",
        )

        if timed_out:
            return {"_error": f"claude CLI timed out after {TIMEOUT_SEC}s"}

        if returncode != 0:
            logger.error("[CLI-Cyrene] CLI failed: %s", full_stderr[:500])
            return {
                "_error": (
                    f"claude CLI exited {returncode}: "
                    f"{full_stderr[:300] or full_stdout[-300:]}"
                ),
            }

        if not result_file.exists():
            logger.error("[CLI-Cyrene] No result file at %s", result_file)
            logger.info("[CLI-Cyrene] stdout tail: %s", full_stdout[-1000:])
            return {"_error": "Cyrene CLI did not call submit_brief"}

        brief = json.loads(result_file.read_text(encoding="utf-8"))

        # Stamp metadata (same fields as run_strategic_review)
        brief["_company"] = company
        brief["_computed_at"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        brief["_duration_sec"] = round(elapsed, 1)

        # Persist
        try:
            brief_path = P.memory_dir(company) / _BRIEF_FILENAME
            brief_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = brief_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(brief, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp.rename(brief_path)
            logger.info(
                "[CLI-Cyrene] %s: brief saved. "
                "interview_questions=%d, dm_targets=%d, content_priorities=%d",
                company,
                len(brief.get("interview_questions", [])),
                len(brief.get("dm_targets", [])),
                len(brief.get("content_priorities", [])),
            )
        except Exception as e:
            logger.warning("[CLI-Cyrene] failed to persist brief for %s: %s", company, e)

        return brief

    finally:
        for f in (system_prompt_file, mcp_config_file):
            try:
                os.unlink(f)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def use_cli() -> bool:
    """Check if CLI mode is enabled."""
    return os.environ.get("AMPHOREUS_USE_CLI", "").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Stream-json event translation
# ---------------------------------------------------------------------------

def _translate_cli_event(event: dict, event_callback: Any) -> None:
    """Map a single CLI stream-json event to the existing event_callback.

    The web UI's event_callback expects ("status"|"text_delta"|"tool_call"|
    "tool_result", payload_dict). The CLI emits coarser chunks than the
    Anthropic SDK's per-token stream — you'll see one text_delta per
    assistant message, not per token — but it's enough for live progress.
    """
    if not event_callback or not isinstance(event, dict):
        return
    etype = event.get("type")
    try:
        if etype == "assistant":
            msg = event.get("message") or {}
            for block in msg.get("content") or []:
                btype = block.get("type")
                if btype == "text":
                    txt = block.get("text") or ""
                    if txt:
                        event_callback("text_delta", {"text": txt})
                elif btype == "tool_use":
                    inp = block.get("input") or {}
                    try:
                        summary = json.dumps(inp, default=str)[:300]
                    except Exception:
                        summary = str(inp)[:300]
                    event_callback("tool_call", {
                        "name": block.get("name", "") or "",
                        "arguments": summary,
                    })
        elif etype == "user":
            msg = event.get("message") or {}
            for block in msg.get("content") or []:
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                event_callback("tool_result", {"result": str(content)[:500]})
        elif etype == "system":
            sub = event.get("subtype") or ""
            if sub:
                event_callback("status", {"message": f"CLI system: {sub}"})
    except Exception:
        # Event translation must never break the run
        pass
