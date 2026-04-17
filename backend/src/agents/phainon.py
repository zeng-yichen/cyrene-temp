"""
Phainon — Image assembly agent.

Generates composite images for LinkedIn posts by searching the web for
source images, downloading them, and assembling a final image using
arbitrary Pillow code executed via Pi's bash tool. The model decides
what to search for, what to download, and writes its own Python code
to assemble the final image — no hand-engineered composition rules.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend.src.db import vortex as P

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phainon")

_PI_AVAILABLE = shutil.which("pi") is not None

try:
    from backend.src.core.config import get_settings
    _settings = get_settings()
    SERPER_API_KEY = _settings.serper_api_key
    SERPER_BASE_URL = _settings.serper_base_url
except Exception:
    SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
    SERPER_BASE_URL = os.getenv("SERPER_BASE_URL", "https://google.serper.dev/search")


# ---------------------------------------------------------------------------
# Tool scripts — written into workspace/tools/ at runtime
# ---------------------------------------------------------------------------

_IMAGE_SEARCH_SCRIPT = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Search for images via Serper Images API.

    Usage: python3 tools/image_search.py "query" [num_results]
    Returns JSON array of image results with url, title, width, height, source.
    \"\"\"
    import json, os, sys, requests

    query = sys.argv[1] if len(sys.argv) > 1 else ""
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if not query:
        print(json.dumps({"error": "No query provided"}))
        sys.exit(1)

    base_url = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev/search")
    api_url = base_url.replace("/search", "/images")
    api_key = os.environ.get("SERPER_API_KEY", "")

    if not api_key:
        print(json.dumps({"error": "SERPER_API_KEY not set"}))
        sys.exit(1)

    try:
        resp = requests.post(
            api_url,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for img in data.get("images", []):
            results.append({
                "url": img.get("imageUrl", ""),
                "title": img.get("title", ""),
                "width": img.get("imageWidth", 0),
                "height": img.get("imageHeight", 0),
                "source": img.get("link", ""),
                "thumbnail": img.get("thumbnailUrl", ""),
            })
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
""")

_DOWNLOAD_IMAGE_SCRIPT = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Download an image from a URL to a local path.

    Usage: python3 tools/download_image.py "https://..." "scratch/bg.jpg"
    Validates the download is a valid image file.
    \"\"\"
    import json, os, sys, requests
    from PIL import Image
    from io import BytesIO

    url = sys.argv[1] if len(sys.argv) > 1 else ""
    output_path = sys.argv[2] if len(sys.argv) > 2 else "scratch/downloaded.jpg"

    if not url:
        print(json.dumps({"error": "No URL provided"}))
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        resp.raise_for_status()

        img = Image.open(BytesIO(resp.content))
        img.save(output_path)
        print(json.dumps({
            "path": output_path,
            "format": img.format,
            "size": [img.width, img.height],
            "mode": img.mode,
            "bytes": len(resp.content),
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
""")

_IMAGE_INFO_SCRIPT = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Get metadata about a local image file.

    Usage: python3 tools/image_info.py "scratch/bg.jpg"
    Returns JSON with dimensions, format, mode, file size.
    \"\"\"
    import json, os, sys
    from PIL import Image

    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path or not os.path.exists(path):
        print(json.dumps({"error": f"File not found: {path}"}))
        sys.exit(1)

    try:
        img = Image.open(path)
        print(json.dumps({
            "path": path,
            "format": img.format,
            "size": [img.width, img.height],
            "mode": img.mode,
            "file_bytes": os.path.getsize(path),
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
""")


# ---------------------------------------------------------------------------
# AGENTS.md template for Pi
# ---------------------------------------------------------------------------

_AGENTS_MD_TEMPLATE = """\
You prepare a ghostwriter's briefing for an upcoming content interview.
Surface story candidates from the client's transcripts and research.
Write the briefing to output/briefing.md.
"""


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def _stage_env(company: str) -> Path:
    """Create a clean workspace for the Phainon agent."""
    workspace = P.workspace_dir(f"{company}-phainon")
    workspace.mkdir(parents=True, exist_ok=True)

    for subdir in ("scratch", "output", "tools"):
        (workspace / subdir).mkdir(exist_ok=True)

    # Clean scratch from previous runs but keep output for history
    scratch = workspace / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
        scratch.mkdir()

    # Symlink client memory for context
    mem_link = workspace / "memory"
    if mem_link.is_symlink():
        mem_link.unlink()
    elif mem_link.is_dir():
        shutil.rmtree(mem_link)
    client_mem = P.memory_dir(company)
    if client_mem.exists():
        os.symlink(client_mem.resolve(), mem_link)
    else:
        mem_link.mkdir()

    return workspace


def _deploy_shell_tools(workspace: Path) -> None:
    """Write the image tool scripts into workspace/tools/."""
    tools_dir = workspace / "tools"
    tools_dir.mkdir(exist_ok=True)
    (tools_dir / "image_search.py").write_text(_IMAGE_SEARCH_SCRIPT, encoding="utf-8")
    (tools_dir / "download_image.py").write_text(_DOWNLOAD_IMAGE_SCRIPT, encoding="utf-8")
    (tools_dir / "image_info.py").write_text(_IMAGE_INFO_SCRIPT, encoding="utf-8")


def _write_agents_md(workspace: Path, post_text: str) -> None:
    """Write AGENTS.md to workspace root for Pi to discover."""
    content = _AGENTS_MD_TEMPLATE.format(post_text=post_text)
    (workspace / "AGENTS.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Pi-based agentic loop
# ---------------------------------------------------------------------------

def _run_pi_agent(
    workspace: Path,
    post_text: str,
    model: str = "claude-opus-4-6",
    event_callback: Any = None,
    feedback_instruction: str = "",
    reference_image_path: str | None = None,
) -> tuple[str | None, list[dict]]:
    """Run the image assembly agent via Pi CLI."""
    session_log: list[dict[str, Any]] = []

    _write_agents_md(workspace, post_text)
    _deploy_shell_tools(workspace)

    if reference_image_path:
        ref = Path(reference_image_path)
        if ref.is_file():
            scratch = workspace / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)
            dest = scratch / "reference_previous.png"
            try:
                shutil.copy2(ref, dest)
                if event_callback:
                    event_callback("status", {"message": f"Copied previous image to scratch/reference_previous.png"})
            except Exception as e:
                logger.warning("[Phainon] Could not copy reference image: %s", e)

    session_dir = workspace / ".pi-sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    has_sessions = any(session_dir.glob("*.jsonl"))

    provider = "anthropic"
    if "/" in model:
        provider, model = model.split("/", 1)

    user_prompt = (
        "Create a composite image to accompany this LinkedIn post. "
        "Search for relevant images, download them, and assemble a final "
        "image using Python/Pillow. Save the result to output/final_image.png "
        "and metadata to output/image_metadata.json."
    )
    if (feedback_instruction or "").strip():
        user_prompt += (
            "\n\n--- Human revision feedback (prioritize; iterate toward what works, not defaults) ---\n"
            + feedback_instruction.strip()
        )
    if reference_image_path and Path(reference_image_path).is_file():
        user_prompt += (
            "\n\nA prior version is at scratch/reference_previous.png — revise it to satisfy the feedback above."
        )

    pi_cmd = [
        "pi",
        "--mode", "json",
        "-p",
        "--provider", provider,
        "--model", model,
        "--thinking", "high",
        "--session-dir", str(session_dir),
        "--tools", "read,bash,edit,write,grep,find,ls",
    ]
    if has_sessions:
        pi_cmd.append("--continue")
    pi_cmd.append(user_prompt)

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
    env["SERPER_API_KEY"] = SERPER_API_KEY
    env["SERPER_BASE_URL"] = SERPER_BASE_URL
    env["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

    session_log.append({
        "type": "session_start",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner": "pi",
        "workspace": str(workspace),
    })

    pi_timeout = 600
    events_seen = 0
    compaction_count = 0
    all_lines: list[str] = []
    exit_code = -1

    def _process_event(event: dict) -> None:
        nonlocal events_seen, compaction_count
        events_seen += 1
        etype = event.get("type", "")

        if etype == "message_update":
            ae = event.get("assistantMessageEvent", {})
            ae_type = ae.get("type", "")
            msg = event.get("message", ae.get("message", {}))

            if ae_type == "text_delta":
                delta = ae.get("textDelta", "")
                if delta and event_callback:
                    event_callback("text_delta", {"text": delta})

            elif ae_type == "thinking_delta":
                delta = ae.get("thinkingDelta", "")
                if delta and event_callback:
                    event_callback("thinking", {"text": delta})

            elif ae_type.startswith("toolcall"):
                for block in msg.get("content", []):
                    if block.get("type") == "toolCall":
                        name = block.get("name", "")
                        args = block.get("arguments", {})
                        summary = ""
                        if isinstance(args, dict):
                            summary = args.get("path", args.get("command", str(args)))[:80]
                        logger.info("[Phainon/Pi] tool: %s(%s)", name, summary)
                        if event_callback:
                            event_callback("tool_call", {"name": name, "arguments": summary})

            usage = msg.get("usage", {})
            if usage:
                cost_info = usage.get("cost", {})
                cost_val = cost_info.get("total", 0) if isinstance(cost_info, dict) else 0
                if cost_val and event_callback:
                    event_callback("status", {
                        "message": f"Tokens: in={usage.get('input', 0)} out={usage.get('output', 0)} cost=${cost_val:.4f}"
                    })

        elif etype == "tool_result":
            result_text = str(event.get("result", ""))[:500]
            if event_callback:
                event_callback("tool_result", {
                    "name": event.get("tool", ""),
                    "result": result_text,
                    "is_error": False,
                })

        elif etype == "turn_end":
            msg = event.get("message", {})
            usage = msg.get("usage", {})
            if usage and event_callback:
                cost_info = usage.get("cost", {})
                cost_val = cost_info.get("total", 0) if isinstance(cost_info, dict) else 0
                event_callback("status", {
                    "message": f"Turn complete — in={usage.get('input', 0)} out={usage.get('output', 0)} cost=${cost_val:.4f}"
                })

        elif etype == "auto_compaction_start":
            compaction_count += 1
            logger.info("[Phainon/Pi] Context compaction #%d", compaction_count)
            if event_callback:
                event_callback("compaction", {"message": f"Context compaction #{compaction_count}"})

        elif etype == "error":
            err_msg = event.get("message", str(event))[:300]
            logger.error("[Phainon/Pi] Error: %s", err_msg)
            if event_callback:
                event_callback("error", {"message": err_msg})

        session_log.append({
            "type": "pi_event",
            "event_type": etype,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": str(event)[:500],
        })

    try:
        proc = subprocess.Popen(
            pi_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=str(workspace),
            env=env,
            text=True,
            bufsize=1,
        )

        stderr_chunks: list[str] = []

        def _read_stderr():
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        file_poll_stop = threading.Event()

        def _poll_files():
            known: set[str] = set()
            for p in workspace.rglob("*"):
                if p.is_file():
                    known.add(str(p))
            while not file_poll_stop.is_set():
                file_poll_stop.wait(3.0)
                if file_poll_stop.is_set():
                    break
                current: set[str] = set()
                for p in workspace.rglob("*"):
                    if p.is_file():
                        current.add(str(p))
                new_files = current - known
                if new_files and event_callback:
                    relative = [str(Path(f).relative_to(workspace)) for f in sorted(new_files)]
                    event_callback("status", {"message": f"Files changed: {', '.join(relative[:10])}"})
                known = current

        file_poll_thread = threading.Thread(target=_poll_files, daemon=True)
        file_poll_thread.start()

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            all_lines.append(line)
            try:
                event = json.loads(line)
                _process_event(event)
            except json.JSONDecodeError:
                logger.debug("[Phainon/Pi] Non-JSON line: %s", line[:200])

        exit_code = proc.wait(timeout=pi_timeout)
        file_poll_stop.set()
        stderr_thread.join(timeout=5)

        logger.info(
            "[Phainon/Pi] Finished — exit=%d events=%d compactions=%d",
            exit_code, events_seen, compaction_count,
        )

    except subprocess.TimeoutExpired:
        logger.error("[Phainon/Pi] Timed out after %ds", pi_timeout)
        proc.kill()
        if event_callback:
            event_callback("error", {"message": f"Pi agent timed out after {pi_timeout}s"})
    except Exception as e:
        logger.error("[Phainon/Pi] Exception: %s", e)
        if event_callback:
            event_callback("error", {"message": str(e)})

    # Check for output
    output_image = workspace / "output" / "final_image.png"
    if output_image.exists():
        return str(output_image), session_log
    return None, session_log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_image(
    company: str,
    post_text: str,
    model: str = "claude-opus-4-6",
    event_callback: Any = None,
    feedback_instruction: str = "",
    reference_image_path: str | None = None,
) -> str | None:
    """Generate a composite image for a post.

    ``feedback_instruction`` and ``reference_image_path`` support human-in-the-loop revision runs.

    Returns the path to the final image, or None if generation failed.
    """
    if not _PI_AVAILABLE:
        logger.error("[Phainon] Pi CLI not found — cannot run image assembly agent")
        if event_callback:
            event_callback("error", {"message": "Pi CLI not installed. Install from https://pi.dev"})
        return None

    P.ensure_dirs(company)
    workspace = _stage_env(company)

    logger.info("[Phainon] Starting image assembly for %s", company)
    if event_callback:
        event_callback("status", {"message": f"Setting up workspace for {company}..."})

    image_path, session_log = _run_pi_agent(
        workspace,
        post_text,
        model,
        event_callback,
        feedback_instruction=feedback_instruction,
        reference_image_path=reference_image_path,
    )

    if image_path:
        # Copy to products directory
        dest_dir = P.images_dir(company)
        dest_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dest_path = dest_dir / f"image_{timestamp}.png"
        shutil.copy2(image_path, dest_path)

        # Also copy metadata if it exists
        meta_src = workspace / "output" / "image_metadata.json"
        if meta_src.exists():
            meta_dest = dest_dir / f"image_{timestamp}_metadata.json"
            shutil.copy2(meta_src, meta_dest)

        logger.info("[Phainon] Image saved to %s", dest_path)
        if event_callback:
            event_callback("status", {"message": f"Image saved to {dest_path}"})
        return str(dest_path)

    logger.warning("[Phainon] No image produced for %s", company)
    if event_callback:
        event_callback("error", {"message": "Image assembly did not produce output"})
    return None
