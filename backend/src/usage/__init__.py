"""Per-user LLM spend tracking.

Design
------
The problem: LLM API calls happen deep inside agent code (Stelle, Aglaea,
Tribbie, Cyrene, etc.) where there's no natural access to the HTTP request
context. Plumbing ``user_email`` through 30+ call sites is untenable.

The fix, in two parts:

1. **ContextVar attribution** (:mod:`backend.src.usage.context`). The CF
   Access middleware populates ``current_user_email`` and
   ``current_client_slug`` on every authenticated request. ContextVars
   propagate through asyncio tasks automatically, so every downstream
   ``anthropic.Anthropic().messages.create(...)`` call inherits the
   attribution without knowing it exists.

2. **Monkey-patch instrumentation** (:mod:`backend.src.usage.instrumentation`).
   At app startup we patch the Anthropic SDK's ``Messages.create`` and
   ``AsyncMessages.create`` exactly once. Every call-site in the codebase is
   captured without modification. The patch reads the ContextVar, runs the
   real method, pulls ``response.usage`` off the result, prices it via
   :mod:`backend.src.usage.pricing`, and writes a ``usage_events`` row via
   :mod:`backend.src.usage.recorder`.

Out of scope for v1 (explicit TODOs):
    - Streaming calls (``messages.stream`` and ``messages.create(stream=True)``).
      Stelle uses streaming in some paths. Usage data arrives in the final
      ``message_stop`` event of the stream and needs a wrapper around the
      stream context manager.
    - OpenAI embeddings. Low cost; deferred.
    - Perplexity httpx calls. Only 2 call sites; can be touched directly.

Background tasks (e.g. ``ordinal_sync``) run outside any HTTP request and
therefore have empty ContextVars. Their usage rows are recorded with
``user_email=NULL`` and surface in the admin view under a "System" bucket.
"""

from backend.src.usage.context import (
    current_client_slug,
    current_user_email,
    set_request_attribution,
)
from backend.src.usage.instrumentation import install_instrumentation
from backend.src.usage.pricing import price_call
from backend.src.usage.recorder import record_usage_event

__all__ = [
    "current_user_email",
    "current_client_slug",
    "set_request_attribution",
    "install_instrumentation",
    "price_call",
    "record_usage_event",
]
