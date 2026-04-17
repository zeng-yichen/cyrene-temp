"""Global LLM SDK instrumentation.

Call :func:`install_instrumentation` exactly once at app startup. It
monkey-patches the Anthropic SDK's ``Messages.create`` (sync) and
``AsyncMessages.create`` so that every call site across the codebase —
all 30+ of them — transparently records a ``usage_events`` row.

Why monkey-patch instead of a wrapper client?
    There is no central LLM client in Amphoreus. 14+ modules each
    instantiate their own ``Anthropic()`` at module scope. Migrating
    them to a shared wrapper would be a ~30-file refactor and create
    merge pain against the rest of the repo. Patching the SDK classes
    once at import time reaches all of them with zero call-site edits.

Streaming calls are NOT instrumented in v1:
    - ``messages.create(stream=True)`` returns an iterator that yields
      deltas; the final ``message_stop`` event carries ``usage``. To
      capture it we'd need to wrap the iterator. TODO.
    - ``messages.stream(...)`` returns a context manager with the same
      shape. Same wrapping problem. TODO.

Non-streaming calls — which include Aglaea, Tribbie, most of Stelle's
tool-use loop, Cyrene fact-checking, and all services (market intel,
series engine, cross-client learning, etc.) — ARE captured.

The patch is idempotent: calling ``install_instrumentation`` twice is a
no-op. We stash the original method on a module-level sentinel so a
rare reload (pytest watch mode) doesn't double-wrap.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from backend.src.usage.context import current_client_slug, current_user_email
from backend.src.usage.recorder import record_usage_event

logger = logging.getLogger("amphoreus.usage.instrumentation")

# Sentinel attribute stamped on the patched callable so we can detect
# prior installation. Any value works; presence is the signal.
_PATCHED_SENTINEL = "_amphoreus_usage_patched"


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull token counts off an Anthropic Message response.

    The SDK attaches a ``usage`` attribute with:
        - input_tokens
        - output_tokens
        - cache_creation_input_tokens (optional, prompt caching)
        - cache_read_input_tokens      (optional, prompt caching)

    Any missing field defaults to 0. Never raises.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_creation_tokens": int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        ),
        "cache_read_tokens": int(
            getattr(usage, "cache_read_input_tokens", 0) or 0
        ),
    }


def _model_from_call(response: Any, kwargs: dict[str, Any]) -> str:
    """Prefer the model the server echoed back on the response, fall
    back to whatever the caller passed in kwargs."""
    return (
        getattr(response, "model", None)
        or kwargs.get("model")
        or "unknown"
    )


def _record_or_swallow(
    *,
    model: str,
    usage: dict[str, int] | None,
    duration_ms: int,
    error: str | None = None,
) -> None:
    """Bridge to the recorder — reads the ContextVars so call sites
    remain oblivious. Never raises; recorder swallows its own errors."""
    record_usage_event(
        provider="anthropic",
        model=model,
        call_kind="messages",
        input_tokens=(usage or {}).get("input_tokens", 0),
        output_tokens=(usage or {}).get("output_tokens", 0),
        cache_creation_tokens=(usage or {}).get("cache_creation_tokens", 0),
        cache_read_tokens=(usage or {}).get("cache_read_tokens", 0),
        user_email=current_user_email.get(),
        client_slug=current_client_slug.get(),
        duration_ms=duration_ms,
        error=error,
    )


def _patch_sync_messages() -> bool:
    """Install the sync ``Messages.create`` wrapper. Returns True if the
    patch was newly applied, False if it was already installed or if
    the anthropic package is missing."""
    try:
        from anthropic.resources.messages import Messages  # type: ignore[import-not-found]
    except Exception:
        logger.warning(
            "usage.instrumentation: anthropic SDK not importable — sync patch skipped"
        )
        return False

    original = Messages.create
    if getattr(original, _PATCHED_SENTINEL, False):
        return False

    @functools.wraps(original)
    def wrapped_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Streaming: the caller is responsible for draining the stream
        # themselves and we can't cleanly capture final usage from an
        # iterator here. Skip recording — will be handled in v2.
        if kwargs.get("stream"):
            return original(self, *args, **kwargs)

        start = time.perf_counter()
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - start) * 1000)
            _record_or_swallow(
                model=kwargs.get("model", "unknown"),
                usage=None,
                duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
            raise

        duration_ms = int((time.perf_counter() - start) * 1000)
        _record_or_swallow(
            model=_model_from_call(response, kwargs),
            usage=_extract_usage(response),
            duration_ms=duration_ms,
        )
        return response

    setattr(wrapped_create, _PATCHED_SENTINEL, True)
    Messages.create = wrapped_create  # type: ignore[method-assign]
    return True


def _patch_async_messages() -> bool:
    """Install the async ``AsyncMessages.create`` wrapper."""
    try:
        from anthropic.resources.messages import AsyncMessages  # type: ignore[import-not-found]
    except Exception:
        logger.warning(
            "usage.instrumentation: anthropic SDK not importable — async patch skipped"
        )
        return False

    original = AsyncMessages.create
    if getattr(original, _PATCHED_SENTINEL, False):
        return False

    @functools.wraps(original)
    async def wrapped_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return await original(self, *args, **kwargs)

        start = time.perf_counter()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - start) * 1000)
            _record_or_swallow(
                model=kwargs.get("model", "unknown"),
                usage=None,
                duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
            raise

        duration_ms = int((time.perf_counter() - start) * 1000)
        _record_or_swallow(
            model=_model_from_call(response, kwargs),
            usage=_extract_usage(response),
            duration_ms=duration_ms,
        )
        return response

    setattr(wrapped_create, _PATCHED_SENTINEL, True)
    AsyncMessages.create = wrapped_create  # type: ignore[method-assign]
    return True


def install_instrumentation() -> None:
    """Idempotent entry point. Called from :func:`main.lifespan` at startup."""
    sync_patched = _patch_sync_messages()
    async_patched = _patch_async_messages()
    if sync_patched or async_patched:
        logger.info(
            "usage.instrumentation: installed Anthropic patches (sync=%s, async=%s)",
            sync_patched,
            async_patched,
        )
    else:
        logger.debug("usage.instrumentation: patches already installed or SDK missing")
