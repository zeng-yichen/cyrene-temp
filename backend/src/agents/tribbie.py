"""
Tribbie — Live interview companion.

Captures system audio via BlackHole virtual audio device, streams it to
Deepgram Nova-3 (when DEEPGRAM_API_KEY is set) or transcribes locally with
faster-whisper as fallback, and suggests follow-up questions using Claude
Haiku with client context from memory files and Aglaea briefings.

Deepgram Nova-3 gives streaming diarization (who is speaking), domain
keyterm prompting, and much lower WER than local Whisper — at ~$0.01-0.02
per interview-hour. Falls back to faster-whisper's large-v3-turbo when no
Deepgram key is configured.

One-time BlackHole setup (Mac):
  1. brew install blackhole-2ch
  2. Reboot
  3. Open Audio MIDI Setup → Create Multi-Output Device
     (check BlackHole 2ch + your speakers, then set as system output)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from collections import Counter
from datetime import datetime
from typing import Callable

import numpy as np

from backend.src.db import vortex as P

logger = logging.getLogger("tribbie")

# Module-level stop signals keyed by job_id
_sessions: dict[str, threading.Event] = {}

# Lazy-loaded Whisper model — loaded once on first session start
_whisper_model = None
_whisper_lock = threading.Lock()

# Audio constants
SAMPLE_RATE = 16_000           # Hz — faster-whisper expects 16kHz mono
CHUNK_SECONDS = 0.1            # sounddevice callback interval (100ms for fast silence detection)
SILENCE_THRESHOLD = 0.005      # RMS below this counts as silence
SILENCE_GAP_SECONDS = 0.4      # consecutive silence → flush & transcribe buffer
MAX_BUFFER_SECONDS = 8.0       # force-flush after this many seconds regardless
MIN_SEGMENT_WORDS = 2          # min words in a segment to qualify for a Haiku suggestion
SUGGESTION_COOLDOWN_SECONDS = 15  # minimum gap between Haiku calls
MAX_CONTEXT_CHARS = 150_000    # max chars of client context sent to Opus (200k ctx window)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                # large-v3-turbo: distilled large-v3, ~8x faster at near-parity
                # WER. ~1.5 GB model — first load downloads from HF Hub, then
                # cached. int8 compute keeps inference real-time on CPU (the
                # only path on Apple Silicon since faster-whisper has no
                # Metal backend).
                logger.info("[Tribbie] Loading faster-whisper 'large-v3-turbo' model (first load may take ~60s to download)...")
                _whisper_model = WhisperModel(
                    "large-v3-turbo",
                    device="auto",
                    compute_type="int8",
                )
                logger.info("[Tribbie] Whisper model ready.")
    return _whisper_model


def _build_initial_prompt(company: str) -> str | None:
    """Seed Whisper's decoder with a sample of this client's past transcript text.

    ``initial_prompt`` is a Whisper knob that biases decoding toward the
    vocabulary, proper nouns, and speaking style present in the prompt.
    Feeding ~700 chars of actual past-transcript text from the same client
    dramatically improves recognition of domain terms (biostatistics,
    clinical-trial acronyms, product/company names, named colleagues)
    without any taxonomy extraction step — the client's own words
    carry the priors.

    Whisper caps ``initial_prompt`` around 224 tokens (~900 chars). We
    target ~700 to stay safely under that limit after tokenization.

    Returns ``None`` if no transcripts exist for this company — Whisper
    then decodes cold, same as before.
    """
    tdir = P.memory_dir(company) / "transcripts"
    if not tdir.exists():
        return None
    candidates = sorted(
        (f for f in tdir.iterdir() if f.is_file() and f.suffix.lower() in (".txt", ".md")),
        key=lambda f: f.stat().st_size,
        reverse=True,
    )
    if not candidates:
        return None
    for f in candidates:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        text = " ".join(text.split())  # collapse whitespace runs
        if len(text) < 200:
            continue
        # Slice from ~1/3 in, to skip any leading metadata headers
        # ("Transcript 1 - 2026-02-03" etc.) but still grab substantive speech.
        start = len(text) // 3
        return text[start:start + 700]
    return None


def _extract_keyterms(company: str, limit: int = 80) -> list[str]:
    """Pull likely-proper-noun domain vocabulary from past transcripts.

    Deepgram Nova-3 accepts a ``keyterm`` list that prior-boosts specific
    terms during decoding. Best candidates are multi-word capitalized
    phrases (people names, org names, product names, acronyms) that
    generic ASR priors mis-recognize. Single-word common technical terms
    are rarely mis-heard by a frontier model, so we skip those.

    Pure regex extraction over all transcript text — no taxonomy,
    no LLM call, no curation.
    """
    tdir = P.memory_dir(company) / "transcripts"
    if not tdir.exists():
        return []
    text_parts: list[str] = []
    for f in sorted(tdir.iterdir()):
        if f.is_file() and f.suffix.lower() in (".txt", ".md"):
            try:
                text_parts.append(f.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
    text = "\n".join(text_parts)
    if not text:
        return []

    # Capitalized multi-word phrases, 2-4 words each. Strips the phrase
    # boundaries at sentence starts (common English "The cat" noise)
    # via a minimum of 2 words with internal title-case.
    # Matches: "Mark Hensley", "FDA Committee", "Non Inferiority Trial", etc.
    multiword = re.findall(
        r"\b(?:[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{1,}){1,3})\b",
        text,
    )
    # Single-capitalized technical tokens of length >=4 (likely
    # acronyms/product names): FDA, JAMA, ClinDev, Abiomed.
    acronyms = re.findall(r"\b(?:[A-Z]{2,}[a-zA-Z]*)\b", text)

    counter: Counter = Counter()
    for m in multiword:
        counter[m.strip()] += 1
    for a in acronyms:
        counter[a] += 1

    # Filter out noisy boilerplate that happens to be Title-Case at
    # sentence starts.
    noise = {
        "The", "This", "That", "These", "Those", "They", "Their", "There",
        "When", "What", "Where", "Which", "While", "Why", "How",
        "Yes", "No", "Ok", "Okay", "And", "But", "Or", "So",
        "I", "Im", "Ive", "Ill", "Id", "You", "Youre", "Well",
    }

    ranked: list[tuple[str, int]] = [
        (term, count) for term, count in counter.most_common()
        if count >= 2 and term not in noise and len(term) >= 3
    ]
    return [term for term, _ in ranked[:limit]]


def _find_blackhole_device() -> int | None:
    """Return the sounddevice index of the BlackHole input device, or None."""
    import sounddevice as sd
    for i, dev in enumerate(sd.query_devices()):
        if "blackhole" in str(dev["name"]).lower() and dev["max_input_channels"] > 0:
            return i
    return None


class MissingCyreneBriefError(RuntimeError):
    """Raised when Tribbie is asked to run without a Cyrene brief on disk."""


def _load_context(company: str) -> dict[str, str]:
    """
    Load context for the follow-up model. Returns a dict of named sections the
    caller stitches into the prompt with XML tags.

    Hard requirement: `memory/{company}/cyrene_brief.json` must exist. That's
    the single source of strategic truth — `content_priorities`,
    `content_avoid`, ICP assessment, DM targets. Older artifacts
    (`strategy_brief.md`, `content_brief.json`, Aglaea's `{company}_briefing.md`,
    files under `content_strategy/`) are deprecated and no longer read.

    Also loaded (optional, nice-to-have): past interview transcripts, so Tribbie
    can dedup against what the client has already said. First-ever interview
    with zero transcripts is fine.
    """
    mem = P.memory_dir(company)
    sections: dict[str, str] = {}
    budget = MAX_CONTEXT_CHARS

    def _take(name: str, text: str) -> None:
        nonlocal budget
        if not text or budget <= 0:
            return
        chunk = text if len(text) <= budget else text[:budget]
        sections[name] = chunk
        budget -= len(chunk)

    # Required: Cyrene brief. No fallback to deprecated strategy files.
    cyrene_brief = mem / "cyrene_brief.json"
    if not cyrene_brief.exists():
        raise MissingCyreneBriefError(
            f"No cyrene_brief.json for '{company}'. Run Cyrene before starting "
            f"a Tribbie session — Tribbie needs a current strategic brief to "
            f"pick which content threads to pursue."
        )
    try:
        obj = json.loads(cyrene_brief.read_text(encoding="utf-8", errors="ignore"))
        _take("cyrene_brief", json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception as e:
        raise MissingCyreneBriefError(
            f"cyrene_brief.json for '{company}' is unreadable: {e}. Re-run Cyrene."
        ) from e

    # Optional: past interview transcripts (newest-first so oldest drop first).
    trans_dir = P.transcripts_dir(company)
    if trans_dir.exists():
        files = sorted(
            (f for f in trans_dir.iterdir() if f.suffix in (".txt", ".md")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        trans_parts: list[str] = []
        for f in files:
            try:
                trans_parts.append(f"### {f.name}\n{f.read_text(encoding='utf-8', errors='ignore')}")
            except Exception:
                continue
        if trans_parts:
            _take("past_transcripts", "\n\n".join(trans_parts))

    return sections


def _suggest_followup(
    segment: str,
    context_sections: dict[str, str],
    transcript_so_far: str,
    anthropic_client,
) -> str | None:
    """Suggest the single best follow-up question for the interviewer.

    Routes through Claude CLI when AMPHOREUS_USE_CLI=1 (Max plan, no API
    spend). Falls through to the API path only when the flag is off —
    never silently falls back to API on CLI failure.
    """
    # Stitch strategic + historical context into labelled XML blocks so the
    # model can reason about each source distinctly.
    ordered = ("cyrene_brief", "past_transcripts")
    context_blocks = "\n\n".join(
        f"<{k}>\n{context_sections[k]}\n</{k}>"
        for k in ordered
        if context_sections.get(k)
    )

    snippet = transcript_so_far[-8_000:]

    prompt = (
        "You are assisting a LinkedIn ghostwriter during a live 1-hour content "
        "interview. Your job is to suggest the single next question that opens "
        "up a rich conversational thread the client WANTS to talk about — one "
        "that produces valuable raw material for posts without making the client "
        "feel interrogated.\n\n"
        "## The quality bar\n\n"
        "The interviewer should hear your question and think: *\"Oh — that's "
        "something I actually have an opinion on.\"* That means the question "
        "found a topic the client cares about and has real experience with, "
        "but hasn't thought to talk about yet on LinkedIn. Generic prompts "
        "(\"tell me about hiring\", \"what are your thoughts on AI\") fail "
        "this bar automatically.\n\n"
        "CRITICAL: Do NOT constantly push for specific memories, specific "
        "moments, exact numbers, or named anecdotes. That pattern frustrates "
        "clients — they feel cross-examined, not interviewed. Stelle can "
        "extrapolate, composite, and fictionalize scenarios from general "
        "conversation. Your job is to open doors to TOPICS, not to extract "
        "depositions. If a specific story emerges naturally, great. If not, "
        "a rich general answer about the client's perspective, beliefs, or "
        "experience is just as valuable.\n\n"
        "Good question shapes:\n"
        "  - Opens a topic the client hasn't covered yet: *\"How do you think "
        "    about X differently than most people in your space?\"*\n"
        "  - Follows energy — builds on something the client was clearly "
        "    engaged by: *\"You seemed to light up when you mentioned Y — "
        "    what's the bigger story there?\"*\n"
        "  - Inverts the obvious: *\"Most people would frame this as a win — "
        "    what's the part nobody talks about?\"*\n"
        "  - Surfaces a tension the client lives with but hasn't articulated.\n\n"
        "## Your three judgments, in order\n\n"
        "  1. REDUNDANCY. Has this topic — or the natural follow-up to what was "
        "just said — already been covered in <past_transcripts> or earlier in "
        "<current_session>? If yes, do not ask it. Re-covering ground is the most "
        "expensive failure mode in a 1-hour call.\n\n"
        "  2. COVERAGE / PIVOT. Is the current topic yielding diminishing returns? "
        "Signs of saturation: concrete outcome + story + quotable line are all "
        "captured. If saturated, PIVOT — pick an unaddressed theme from "
        "<cyrene_brief.content_priorities>. Interviews tend to run out of "
        "questions around minute 40; your job is to prevent that by always "
        "holding the full menu of priorities in mind and knowing which are still "
        "untouched. Treat <cyrene_brief.content_avoid> as hard constraints — "
        "never pivot into those.\n\n"
        "  3. STRATEGIC FIT. Among unaddressed threads, prefer the one most "
        "load-bearing for the content strategy. <cyrene_brief.content_priorities> "
        "tells you what this client needs next — favour questions whose answer "
        "directly produces that kind of post.\n\n"
        "Note: <cyrene_brief> is your ONLY strategic input. It intentionally "
        "contains no pre-written questions — you generate them. The content "
        "priorities are your topic menu; the conversation is your steering; the "
        "quality bar above is your filter.\n\n"
        f"{context_blocks}\n\n"
        f"<current_session>\n{snippet}\n</current_session>\n\n"
        f"<just_said>\n{segment}\n</just_said>\n\n"
        "Reply in exactly two lines:\n"
        "Line 1: one short italicised note naming which judgment drove this — "
        "e.g. *Pivoting: the MTTR thread is saturated, haven't touched the "
        "50→200 hiring angle from content_priorities yet* or *Drilling: inverting "
        "the obvious win-framing because the cost hasn't surfaced* or *Skipping "
        "the obvious follow-up — that ground was covered in the Q2 transcript*.\n"
        "Line 2: the question itself — natural, conversational, specific, one "
        "sentence, and meeting the \"amazing post idea\" bar."
    )

    from backend.src.mcp_bridge.claude_cli import use_cli as _use_cli, cli_single_shot as _cli_ss
    if _use_cli():
        # 60s timeout is plenty for a 200-token opus response; live
        # interview cadence tolerates 2-5s per suggestion.
        txt = _cli_ss(prompt, model="opus", max_tokens=250, timeout=60)
        return txt.strip() if txt else None

    try:
        response = anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip() if response.content else None
    except Exception as e:
        logger.warning("[Tribbie] suggestion failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict]:
    """Return all available audio input devices. Used by the health-check endpoint."""
    import sounddevice as sd
    result = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "is_blackhole": "blackhole" in str(dev["name"]).lower(),
            })
    return result


def _open_partial_transcript(company: str) -> tuple:
    """Open a `.partial.txt` file for incremental, crash-safe writing.

    Returns ``(path, file_handle)``. The file is opened in line-buffered
    append mode so each transcript line is flushed to disk as soon as it
    arrives — if the process is killed (uvicorn reload, OOM, manual
    SIGTERM) we lose at most the in-flight segment, not the whole hour.
    Caller closes the handle and `_finalize_transcript` renames it to a
    final `.txt` on graceful stop.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = P.transcripts_dir(company)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"live_interview_{timestamp}.partial.txt"
    fh = open(partial_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    return partial_path, fh


def _append_line(fh, line: str) -> None:
    """Append one transcript line to the partial file. Best-effort — never raise."""
    try:
        fh.write(line + "\n\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass  # fsync not supported on all FS; flush is good enough
    except Exception as e:
        logger.warning("[Tribbie] partial-transcript append failed: %s", e)


def _finalize_transcript(
    company: str,
    partial_path,
    partial_fh,
    transcript_lines: list[str],
    segment_count: int,
    start_time: float,
    event_callback: Callable,
) -> None:
    """Close the partial file and rename to a final `.txt`. Emit `done` event.

    If no speech was captured, deletes the empty partial file. The partial
    file already contains every line that ``_append_line`` wrote, so this
    function does NOT need to rewrite the content from ``transcript_lines``
    — it only renames. ``transcript_lines`` is used for the segment count
    in the user-facing message.
    """
    try:
        partial_fh.close()
    except Exception:
        pass

    if not transcript_lines:
        try:
            partial_path.unlink(missing_ok=True)
        except Exception:
            pass
        event_callback("done", {
            "output": None,
            "message": "Session ended with no speech captured.",
        })
        return

    final_path = partial_path.with_name(partial_path.name.replace(".partial.txt", ".txt"))
    try:
        partial_path.rename(final_path)
        out_path = final_path
    except Exception as e:
        # Rename failed (rare); fall back to leaving the .partial.txt in place.
        logger.warning("[Tribbie] failed to finalize transcript filename: %s", e)
        out_path = partial_path

    duration_min = round((time.time() - start_time) / 60, 1)
    event_callback("done", {
        "output": str(out_path),
        "message": (
            f"Session complete — {segment_count} segments, {duration_min} min. "
            f"Transcript saved: {out_path.name}"
        ),
    })


def _run_deepgram_session(
    company: str,
    job_id: str,
    event_callback: Callable,
    device_idx: int,
    context: dict[str, str],
    anthropic_client,
    stop_event: threading.Event,
    start_time: float,
    partial_path,
    partial_fh,
) -> None:
    """Stream BlackHole audio to Deepgram Nova-3 with diarization + keyterms.

    Raises ``ImportError`` if the deepgram SDK isn't installed so
    ``start_session`` can fall back to Whisper cleanly.
    """
    # Raises ImportError if SDK missing — start_session catches and falls back.
    from deepgram import (  # type: ignore[import-not-found]
        DeepgramClient,
        LiveOptions,
        LiveTranscriptionEvents,
    )
    import sounddevice as sd

    transcript_lines: list[str] = []
    segment_count = [0]
    last_suggestion_time = [0.0]

    keyterms = _extract_keyterms(company)
    dg_client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
    conn = dg_client.listen.websocket.v("1")

    def on_transcript(_client, result, **_kwargs) -> None:
        try:
            if not result or not getattr(result, "is_final", False):
                return
            channel = getattr(result, "channel", None)
            if channel is None:
                return
            alternatives = getattr(channel, "alternatives", None) or []
            if not alternatives:
                return
            alt = alternatives[0]
            text = (getattr(alt, "transcript", "") or "").strip()
            if not text:
                return

            speaker_label: str | None = None
            words = getattr(alt, "words", None) or []
            if words:
                speakers = {
                    getattr(w, "speaker", None) for w in words
                    if getattr(w, "speaker", None) is not None
                }
                if len(speakers) == 1:
                    speaker_label = f"Speaker {next(iter(speakers))}"

            line = f"{speaker_label}: {text}" if speaker_label else text
            transcript_lines.append(line)
            _append_line(partial_fh, line)
            segment_count[0] += 1
            event_callback("text_delta", {"text": line})
            event_callback("status", {"message": "Listening…"})

            now = time.time()
            if (
                len(text.split()) >= MIN_SEGMENT_WORDS
                and now - last_suggestion_time[0] >= SUGGESTION_COOLDOWN_SECONDS
            ):
                last_suggestion_time[0] = now
                transcript_so_far = "\n".join(transcript_lines)
                suggestion = _suggest_followup(text, context, transcript_so_far, anthropic_client)
                if suggestion:
                    event_callback("tool_result", {
                        "name": "follow_up",
                        "result": suggestion,
                        "is_error": False,
                    })
        except Exception as e:
            logger.warning("[Tribbie] Deepgram on_transcript error: %s", e)

    def on_error(_client, error, **_kwargs) -> None:
        msg = getattr(error, "message", None) or str(error)
        logger.warning("[Tribbie] Deepgram error event: %s", msg)
        event_callback("status", {"message": f"Deepgram: {msg}"})

    conn.on(LiveTranscriptionEvents.Transcript, on_transcript)
    conn.on(LiveTranscriptionEvents.Error, on_error)

    options_kwargs = dict(
        model="nova-3",
        language="en-US",
        smart_format=True,
        punctuate=True,
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        channels=1,
        diarize=True,
        interim_results=False,
        vad_events=True,
        endpointing=500,  # ms silence → emit final transcript
    )
    if keyterms:
        # Nova-3 feature. Older SDKs may not know this kwarg; try it, and
        # drop it silently if LiveOptions rejects it.
        try:
            options = LiveOptions(**options_kwargs, keyterm=keyterms)
        except TypeError:
            logger.info("[Tribbie] Deepgram SDK doesn't accept keyterm kwarg — proceeding without.")
            options = LiveOptions(**options_kwargs)
    else:
        options = LiveOptions(**options_kwargs)

    if not conn.start(options):
        raise RuntimeError("Deepgram start() returned False")

    event_callback("status", {
        "message": (
            f"Deepgram Nova-3 connected. "
            f"Diarization on, {len(keyterms)} keyterms primed from past transcripts."
        ),
    })

    def _audio_cb(indata: np.ndarray, frames: int, _time_info, _status) -> None:
        if stop_event.is_set():
            return
        try:
            pcm = (np.clip(indata, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            conn.send(pcm)
        except Exception as e:
            logger.debug("[Tribbie] Deepgram send failed: %s", e)

    chunk_frames = int(SAMPLE_RATE * CHUNK_SECONDS)
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            device=device_idx,
            callback=_audio_cb,
        ):
            event_callback("status", {"message": "Recording started. Speak now…"})
            while not stop_event.is_set():
                time.sleep(0.25)
    except Exception as e:
        logger.exception("[Tribbie] Fatal error in Deepgram capture loop")
        event_callback("error", {"message": str(e)})
    finally:
        try:
            conn.finish()
        except Exception:
            pass
        _sessions.pop(job_id, None)

    _finalize_transcript(
        company, partial_path, partial_fh,
        transcript_lines, segment_count[0], start_time, event_callback,
    )


def start_session(company: str, job_id: str, event_callback: Callable) -> None:
    """
    Run the live interview capture loop.

    Called from a background thread via job_manager.run_in_background.
    Blocks until stop_session(job_id) is called or a fatal error occurs.

    event_callback(event_type: str, data: dict) — emits structured events.

    Routes to Deepgram Nova-3 streaming when ``DEEPGRAM_API_KEY`` is set,
    otherwise falls back to local faster-whisper. Deepgram path gives
    streaming diarization + keyterm prompting + ~6% WER; Whisper fallback
    is the large-v3-turbo setup with client-specific initial_prompt.
    """
    from anthropic import Anthropic

    stop_event = threading.Event()
    _sessions[job_id] = stop_event
    start_time = time.time()

    # --- Load client context (hard-fails if no cyrene_brief.json on disk) ---
    try:
        context = _load_context(company)
    except MissingCyreneBriefError as e:
        event_callback("error", {"message": str(e)})
        _sessions.pop(job_id, None)
        return
    loaded = ", ".join(context.keys())
    total_chars = sum(len(v) for v in context.values())
    event_callback("status", {
        "message": f"Context loaded: {loaded} ({total_chars:,} chars).",
    })

    # --- Verify BlackHole device ---
    device_idx = _find_blackhole_device()
    if device_idx is None:
        event_callback("error", {
            "message": (
                "BlackHole audio device not found. "
                "Install: brew install blackhole-2ch, reboot, then open Audio MIDI Setup "
                "and create a Multi-Output Device combining BlackHole 2ch + your speakers."
            ),
        })
        _sessions.pop(job_id, None)
        return

    anthropic_client = Anthropic()

    # Open a crash-safe partial-transcript file BEFORE either capture
    # path starts. Every transcript line is appended + fsync'd in real
    # time, so an unclean shutdown (uvicorn --reload, OS kill, network
    # blip) leaves a recoverable `.partial.txt` instead of nothing.
    partial_path, partial_fh = _open_partial_transcript(company)
    event_callback("status", {
        "message": f"Recording to {partial_path.name} (auto-saved per segment).",
    })

    # --- Route to Deepgram (preferred) or Whisper (fallback) ---
    if os.environ.get("DEEPGRAM_API_KEY", "").strip():
        try:
            _run_deepgram_session(
                company=company,
                job_id=job_id,
                event_callback=event_callback,
                device_idx=device_idx,
                context=context,
                anthropic_client=anthropic_client,
                stop_event=stop_event,
                start_time=start_time,
                partial_path=partial_path,
                partial_fh=partial_fh,
            )
            return
        except ImportError:
            event_callback("status", {
                "message": (
                    "deepgram-sdk not installed — falling back to local Whisper. "
                    "Install with: pip install deepgram-sdk"
                ),
            })
        except Exception as e:
            logger.exception("[Tribbie] Deepgram session failed, falling back to Whisper")
            event_callback("status", {
                "message": f"Deepgram session failed ({e}). Falling back to local Whisper.",
            })

    # --- Whisper fallback path ---
    import sounddevice as sd
    model = _get_whisper_model()
    initial_prompt = _build_initial_prompt(company)
    if initial_prompt:
        event_callback("status", {
            "message": f"Whisper primed with {len(initial_prompt)} chars of past {company} transcript context.",
        })

    # --- Audio capture state ---
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    transcribe_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    chunk_frames = int(SAMPLE_RATE * CHUNK_SECONDS)
    silence_threshold_chunks = int(SILENCE_GAP_SECONDS / CHUNK_SECONDS)
    max_work_chunks = int(MAX_BUFFER_SECONDS / CHUNK_SECONDS)

    segment_count = 0
    transcript_lines: list[str] = []
    transcribe_error: list[str] = []
    last_suggestion_time: float = 0.0

    def _audio_cb(indata: np.ndarray, frames: int, _time_info, _status) -> None:
        audio_q.put(indata.copy())

    # --- Transcription worker (separate thread so capture never stalls) ---
    def _transcribe_worker() -> None:
        while True:
            try:
                audio_array = transcribe_q.get(timeout=1.0)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            if audio_array is None:  # poison pill
                break

            nonlocal segment_count
            segment_count += 1
            event_callback("status", {"message": f"Transcribing…"})

            try:
                segs, _ = model.transcribe(
                    audio_array,
                    language="en",
                    # beam_size 5 catches more hypotheses than greedy — still
                    # within latency budget on turbo.
                    beam_size=5,
                    vad_filter=True,
                    # Seed decoder with client-specific vocabulary so
                    # technical terms, acronyms, and proper nouns decode
                    # correctly without any taxonomy extraction.
                    initial_prompt=initial_prompt,
                    # Tighter silence handling to reduce Whisper's
                    # "hallucinates text during quiet stretches" failure mode.
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    # Clamp repetition so Whisper doesn't loop on the same
                    # phrase when audio quality drops.
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text for s in segs).strip()
            except Exception as te:
                logger.warning("[Tribbie] Whisper error: %s", te)
                transcribe_error.append(str(te))
                continue

            if not text:
                event_callback("status", {"message": "Listening…"})
                continue

            transcript_lines.append(text)
            _append_line(partial_fh, text)
            event_callback("text_delta", {"text": text})
            event_callback("status", {"message": "Listening…"})

            nonlocal last_suggestion_time
            now = time.time()
            if (
                len(text.split()) >= MIN_SEGMENT_WORDS
                and now - last_suggestion_time >= SUGGESTION_COOLDOWN_SECONDS
            ):
                last_suggestion_time = now
                transcript_so_far = "\n".join(transcript_lines)
                suggestion = _suggest_followup(text, context, transcript_so_far, anthropic_client)
                if suggestion:
                    event_callback("tool_result", {
                        "name": "follow_up",
                        "result": suggestion,
                        "is_error": False,
                    })

    transcribe_thread = threading.Thread(target=_transcribe_worker, daemon=True)
    transcribe_thread.start()

    work_buf: list[np.ndarray] = []
    silence_streak = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            device=device_idx,
            callback=_audio_cb,
        ):
            event_callback("status", {"message": "Recording started. Speak now…"})

            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                work_buf.append(chunk)

                if rms < SILENCE_THRESHOLD:
                    silence_streak += 1
                else:
                    silence_streak = 0

                should_flush = (
                    silence_streak >= silence_threshold_chunks
                    or len(work_buf) >= max_work_chunks
                )

                if not should_flush:
                    continue

                audio_array = np.concatenate(work_buf, axis=0).flatten()
                work_buf.clear()
                silence_streak = 0

                if len(audio_array) < SAMPLE_RATE * 0.3:
                    continue  # skip sub-300ms fragments (clicks, breath)

                try:
                    transcribe_q.put_nowait(audio_array)
                except queue.Full:
                    logger.warning("[Tribbie] Transcription queue full — dropping segment")

    except Exception as e:
        logger.exception("[Tribbie] Fatal error in capture loop")
        event_callback("error", {"message": str(e)})
    finally:
        # Drain remaining audio, shut down transcription thread, clean up session
        if work_buf:
            audio_array = np.concatenate(work_buf, axis=0).flatten()
            if len(audio_array) >= SAMPLE_RATE * 0.3:
                try:
                    transcribe_q.put_nowait(audio_array)
                except queue.Full:
                    pass
        transcribe_q.put(None)  # poison pill
        transcribe_thread.join(timeout=30)
        _sessions.pop(job_id, None)

    # --- Save transcript ---
    _finalize_transcript(
        company, partial_path, partial_fh,
        transcript_lines, segment_count, start_time, event_callback,
    )


def stop_session(job_id: str) -> bool:
    """Signal a running session to stop gracefully. Returns True if session was found."""
    ev = _sessions.get(job_id)
    if ev:
        ev.set()
        return True
    return False
