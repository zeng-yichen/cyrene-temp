"""Request-scoped attribution via ContextVar.

Two globals that the CF Access middleware sets on every authenticated
request, and that the LLM instrumentation layer reads on every LLM call.

ContextVars are the right primitive here because:
    - they propagate through ``asyncio.create_task`` and
      ``run_in_executor`` without manual threading of parameters;
    - they're per-request isolated (no thread-local leakage between
      concurrent requests sharing a worker thread);
    - reading is free (no locks).

Both vars default to ``None``. The instrumentation layer MUST treat
``None`` as "unattributed" — i.e. a background task like ``ordinal_sync``
or an eval harness run — and write a ``usage_events`` row with
``user_email IS NULL``. Never crash or drop the call because attribution
is missing.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

# Email of the authenticated user driving the current request.
# Set by backend.src.auth.middleware.CfAccessAuthMiddleware after JWT verify.
current_user_email: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "amphoreus_current_user_email", default=None
)

# Client slug the request is operating on, extracted from the URL path
# (e.g. /api/ghostwriter/example-client → "example-client"). Optional — many
# endpoints are not client-scoped.
current_client_slug: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "amphoreus_current_client_slug", default=None
)


@contextmanager
def set_request_attribution(
    email: str | None,
    client_slug: str | None = None,
) -> Iterator[None]:
    """Scoped setter used by middleware and test fixtures.

    Example::

        with set_request_attribution("user@example.com", "client-a"):
            await call_an_agent()

    Outside the block the previous values are restored — so nested calls
    compose cleanly.
    """
    email_token = current_user_email.set(email)
    slug_token = current_client_slug.set(client_slug)
    try:
        yield
    finally:
        current_client_slug.reset(slug_token)
        current_user_email.reset(email_token)
