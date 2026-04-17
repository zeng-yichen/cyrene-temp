"""Tracing stubs — no-op replacements for Langfuse.

The @traced decorator and trace_llm_call are retained as no-ops so that
adapter modules (stelle_adapter, herta_adapter, aglaea_adapter) need
zero changes.
"""

import asyncio
import functools
from typing import Any, Callable


def traced(
    name: str | None = None,
    kind: str = "generation",
    metadata: dict[str, Any] | None = None,
):
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator


def trace_llm_call(
    model: str = "",
    input_text: str = "",
    output_text: str = "",
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    duration_seconds: float | None = None,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    pass
