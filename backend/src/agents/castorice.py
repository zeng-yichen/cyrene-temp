"""
Castorice — Fact-checker and source annotator.

Two tasks in one pass:
1. SOURCE ANNOTATION: independently traces every factual claim in the post
   back to source documents (transcripts, references, content strategy),
   producing an annotated version of the post with inline source comments.
2. FACT-CHECK: verifies claims against local context + Google Search,
   flags inaccuracies, and produces a corrected post if needed.

The annotated post is stored locally for editorial review (CE can see
sourcing without it touching the published text). Citation comments
are pushed as Ordinal post comments by Hyacinthia on confirmed push.
"""

import logging
import os
import re

import httpx
from google import genai
from google.genai import types

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defensive parsing helpers
# ---------------------------------------------------------------------------

# Patterns that indicate the LLM has started trailing content AFTER the
# [CORRECTED POST] body. If we hit one of these at the start of a line, we
# truncate the corrected post at that point.
#
# Note: these are operational heuristics for detecting model output drift,
# not learning-signal gates. Their purpose is to stop garbage from leaking
# into a published post if the LLM ignores the "stop after [CORRECTED POST]"
# instruction in the prompt.
_TRAILING_CONTAMINATION_LINE_PATTERNS = [
    # Bracketed section headers (the LLM recapping its own sections)
    re.compile(r"^\s*\[[A-Z][A-Z /\-]+\]\s*$"),
    # All-caps section labels with a trailing colon ("SUMMARY:", "NOTES:", etc.)
    re.compile(r"^\s*[A-Z][A-Z /\-]{3,}:\s*$"),
    # Horizontal rule separators that LinkedIn posts never contain
    re.compile(r"^\s*-{3,}\s*$"),
    re.compile(r"^\s*={3,}\s*$"),
    # Bold markdown section headers ("**SUMMARY:**", "**Corrections:**")
    re.compile(r"^\s*\*\*[A-Z][A-Za-z /\-]+:\*\*\s*$"),
    # Explicit end-of-report sentinels
    re.compile(r"^\s*\[END[A-Z /\-]*\]\s*$", re.IGNORECASE),
    # A line that looks like a restart of the Castorice report format
    re.compile(r"^\s*\[(ANNOTATED POST|CITATION COMMENTS|FACT-CHECK RESULTS|CORRECTED POST)\]\s*$"),
    # "SUMMARY:" as a standalone line
    re.compile(r"^\s*SUMMARY\s*:\s*$"),
]

# A URN is "placeholder" if its suffix is not a plain sequence of digits.
# Real LinkedIn/Ordinal organization/person URNs look like
# `urn:li:organization:11130470` — the suffix after the last colon is all
# digits. If the LLM invents `XXXXX`, `PLACEHOLDER`, `UNKNOWN`, or anything
# non-numeric, we strip the whole tag wrapper and leave the display name.
_TAG_WITH_URN_RE = re.compile(
    r"@\[([^\]]+)\]\(urn:li:(?:organization|person|member):([^\)]+)\)"
)


def _strip_trailing_contamination(text: str) -> str:
    """Truncate the corrected post at the first trailing section / recap marker.

    Walks the text line by line. Returns everything up to (but not
    including) the first line that matches a known contamination pattern.
    If no contamination is found, returns the input unchanged.
    """
    if not text:
        return text
    lines = text.split("\n")
    cut_at = None
    for i, line in enumerate(lines):
        for pat in _TRAILING_CONTAMINATION_LINE_PATTERNS:
            if pat.match(line):
                cut_at = i
                break
        if cut_at is not None:
            break
    if cut_at is None:
        return text
    # Trim trailing blank lines before the cut so we don't leave a dangling "\n\n"
    while cut_at > 0 and not lines[cut_at - 1].strip():
        cut_at -= 1
    trimmed = "\n".join(lines[:cut_at]).rstrip()
    if trimmed != text:
        logger.info(
            "[Castorice] trimmed trailing contamination at line %d "
            "(kept %d chars, discarded %d chars)",
            cut_at, len(trimmed), len(text) - len(trimmed),
        )
    return trimmed


def _strip_placeholder_urn_tags(text: str) -> str:
    """Replace any `@[Name](urn:li:*:NON_NUMERIC)` tag with just the bare name.

    Real URN suffixes are purely numeric. If the LLM slipped a placeholder
    (XXXXX, PLACEHOLDER, UNKNOWN, TBD, or any non-numeric value) into a tag
    wrapper, the LinkedIn publish will break. We defensively strip the tag
    syntax and leave the display name intact so the post is still valid,
    and log the occurrence so the operator can see the issue.
    """
    if not text or "urn:li:" not in text:
        return text

    stripped_count = 0
    def _replace(m: re.Match) -> str:
        nonlocal stripped_count
        display_name = m.group(1)
        urn_suffix = m.group(2).strip()
        # Real URN suffixes are digits only. Anything else is invented.
        if urn_suffix.isdigit():
            return m.group(0)  # keep the real tag
        stripped_count += 1
        return display_name

    new_text = _TAG_WITH_URN_RE.sub(_replace, text)
    if stripped_count:
        logger.warning(
            "[Castorice] Stripped %d placeholder-URN tag(s) from corrected post "
            "(LLM invented non-numeric URN suffixes). Flagged tags would have "
            "broken the published post.",
            stripped_count,
        )
    return new_text


class Castorice:
    """
    Castorice: The Fact-Checker and Source Annotator.
    Reviews drafted posts against the client's local context files
    AND the live internet (via Google Search) to ensure factual accuracy,
    and independently traces every claim back to its source document.

    Primary: Perplexity Sonar Pro (built-in web search, higher factuality).
    Fallback: Gemini with Google Search grounding.
    """

    def __init__(self, model_name: str | None = None):
        self._gemini_model = model_name or os.environ.get(
            "CASTORICE_GEMINI_MODEL", "gemini-2.5-flash"
        )
        self._gemini_client = genai.Client()
        self._perplexity_key = os.environ.get("PERPLEXITY_API_KEY", "")

    def _get_local_context(self, company_keyword: str) -> str:
        """
        Load client context from transcripts, references, and content strategy.
        All three are needed for complete source annotation.
        """
        dirs_to_load = [
            ("transcripts", P.transcripts_dir(company_keyword)),
            ("references", P.references_dir(company_keyword)),
            ("content_strategy", P.content_strategy_dir(company_keyword)),
        ]
        context_parts: list[str] = []

        for label, directory in dirs_to_load:
            if not directory.exists():
                continue
            for filepath in sorted(directory.rglob("*")):
                if filepath.is_file() and filepath.suffix.lower() in (".txt", ".md"):
                    try:
                        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
                        if text:
                            context_parts.append(
                                f"\n--- SOURCE [{label}]: {filepath.name} ---\n{text}\n"
                            )
                    except Exception as e:
                        print(f"[Castorice] Failed to read {filepath.name}: {e}")

        return "".join(context_parts)

    def fact_check_post(
        self,
        company_keyword: str,
        post_content: str,
    ) -> dict:
        """
        Annotate and fact-check a post.

        Args:
            company_keyword: Client identifier used to locate source files.
            post_content: The clean post text produced by Stelle.

        Returns:
            dict with keys:
                "report"             — full fact-check text (str)
                "corrected_post"     — post text after factual fixes (str)
                "annotated_post"     — post text with inline source comments (str)
                "citation_comments"  — list of formatted comment strings for
                                       Ordinal post comments (list[str])
        """
        local_context = self._get_local_context(company_keyword)

        system_instruction = """\
You fact-check LinkedIn post drafts. For each factual claim, verify or
flag it. Return a corrected post and a list of citation comments.
"""

        prompt = f"""\
<local_context>
{local_context}
</local_context>

<drafted_post>
{post_content}
</drafted_post>

Produce your annotation and fact-check in the required format.
"""

        import time as _time

        # --- Primary: Perplexity Sonar Pro (built-in web search) ---
        if self._perplexity_key:
            try:
                return self._fact_check_perplexity(system_instruction, prompt, post_content)
            except Exception as pplx_err:
                logger.warning("[Castorice] Perplexity failed: %s — falling back to Gemini", pplx_err)

        # --- Fallback: Gemini with Google Search grounding ---
        search_config = types.GenerateContentConfig(
            temperature=0.1,
            tools=[{"google_search": {}}],
        )

        last_err = None
        for attempt in range(4):
            try:
                response = self._gemini_client.models.generate_content(
                    model=self._gemini_model,
                    contents=system_instruction + "\n\n" + prompt,
                    config=search_config,
                )
                return self._parse_response(response.text)
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if any(
                    s in err_str
                    for s in (
                        "429",
                        "503",
                        "rate_limit",
                        "resource exhausted",
                        "overloaded",
                        "timeout",
                    )
                ):
                    delay = min(20.0, 2.0 * (2 ** attempt))
                    print(f"[Castorice] Gemini retryable error (attempt {attempt + 1}/4), waiting {delay}s: {e}")
                    _time.sleep(delay)
                    continue
                return {
                    "report": f"Fact-check failed due to API error: {e}",
                    "corrected_post": post_content,
                    "annotated_post": "",
                    "citation_comments": [],
                }

        return {
            "report": f"Fact-check failed after retries: {last_err}",
            "corrected_post": post_content,
            "annotated_post": "",
            "citation_comments": [],
        }

    def _fact_check_perplexity(
        self, system_instruction: str, prompt: str, post_content: str,
    ) -> dict:
        """Fallback fact-checker using Perplexity Sonar (has built-in web search)."""
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {self._perplexity_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar-pro",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 4000,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        body = (data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()
        if not body:
            return {
                "report": "Perplexity returned empty response",
                "corrected_post": post_content,
                "annotated_post": "",
                "citation_comments": [],
            }
        logger.info("[Castorice] Perplexity fallback succeeded")
        return self._parse_response(body)

    def _parse_response(self, raw: str) -> dict:
        """Parse the model output into structured fields."""
        annotated_post = ""
        corrected_post = ""
        citation_comments: list[str] = []

        # Extract [ANNOTATED POST] — everything up to the next section tag
        if "[ANNOTATED POST]" in raw:
            after = raw.split("[ANNOTATED POST]", 1)[1]
            for next_tag in ("[CITATION COMMENTS]", "[FACT-CHECK RESULTS]", "[CORRECTED POST]"):
                if next_tag in after:
                    annotated_post = after.split(next_tag, 1)[0].strip()
                    break
            else:
                annotated_post = after.strip()

        # Extract [CITATION COMMENTS] — collect blank-line-separated entries
        if "[CITATION COMMENTS]" in raw:
            after = raw.split("[CITATION COMMENTS]", 1)[1]
            for next_tag in ("[FACT-CHECK RESULTS]", "[CORRECTED POST]"):
                if next_tag in after:
                    cite_block = after.split(next_tag, 1)[0].strip()
                    break
            else:
                cite_block = after.strip()
            citation_comments = [e.strip() for e in cite_block.split("\n\n") if e.strip()]

        # Extract [CORRECTED POST] and defensively trim trailing contamination.
        #
        # The prompt tells the LLM to stop after the corrected post body, but
        # models occasionally append a SUMMARY / recap / closing metadata
        # block. Naive `split("[CORRECTED POST]", 1)[1]` would bundle all of
        # that into the post text. We trim at the first trailing marker that
        # cannot be part of a normal LinkedIn post body, and also strip any
        # placeholder-URN tags (the other known failure mode where the LLM
        # invents `urn:li:organization:XXXXX` or similar).
        if "[CORRECTED POST]" in raw:
            tail = raw.split("[CORRECTED POST]", 1)[1].strip()
            corrected_post = _strip_trailing_contamination(tail)
            corrected_post = _strip_placeholder_urn_tags(corrected_post)

        return {
            "report": raw,
            "corrected_post": corrected_post or "",
            "annotated_post": annotated_post,
            "citation_comments": citation_comments,
        }
