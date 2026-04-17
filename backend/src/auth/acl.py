"""File-backed per-email client ACL.

Stored on the persistent Fly volume at ``/data/acl.json`` (see
``settings.acl_path``). Shape:

    {
      "admins": ["admin@example.com"],
      "users": {
        "user@example.com": ["client-a", "client-b"]
      }
    }

- ``admins``: emails that see every client. No path guard fires for them.
- ``users``: email → list of allowed client slugs. Emails not in either bucket
  are rejected with 403 at the JWT middleware layer — Cloudflare Access already
  allowlisted them, so getting here means a gap between the CF policy and the
  ACL file (e.g. new engineer added to CF but not yet to ACL).

Reads are cheap: we mtime-cache the parsed JSON and reload on change. Writes
go through ``Acl.save()`` which is lock-protected and atomic (temp-file +
rename). No UI endpoint for now — seed by hand via ``flyctl ssh``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger("amphoreus.auth")

# Returned by check_access for admin users or wildcard allowlists.
Wildcard = Literal["*"]
WILDCARD: Wildcard = "*"


@dataclass(frozen=True)
class AclDecision:
    """Result of checking whether an email may access a client slug."""

    allowed: bool
    reason: str  # "admin", "allowlisted", "not_in_acl", "client_not_allowed"


class Acl:
    """Thread-safe, mtime-cached ACL reader/writer."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: dict = {"admins": [], "users": {}}
        self._mtime: float = 0.0
        self._loaded_once = False

    # ------------------------------------------------------------------ I/O

    def _reload_if_stale(self) -> None:
        """Reload the file if the mtime changed. Caller holds the lock."""
        if not self._path.exists():
            if self._loaded_once:
                return  # file was deleted mid-session; keep old data
            logger.warning(
                "ACL file %s does not exist. All non-admin users will be denied.", self._path
            )
            self._data = {"admins": [], "users": {}, "banned": []}
            self._loaded_once = True
            return
        mtime = self._path.stat().st_mtime
        if mtime == self._mtime and self._loaded_once:
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to parse ACL file %s; keeping previous state", self._path)
            return
        admins = [str(e).strip().lower() for e in raw.get("admins", []) if str(e).strip()]
        banned = [str(e).strip().lower() for e in raw.get("banned", []) if str(e).strip()]
        users_raw = raw.get("users", {}) or {}
        users: dict[str, list[str]] = {}
        for email, slugs in users_raw.items():
            if not isinstance(slugs, list):
                continue
            users[str(email).strip().lower()] = [str(s).strip() for s in slugs if str(s).strip()]
        self._data = {"admins": admins, "users": users, "banned": banned}
        self._mtime = mtime
        self._loaded_once = True
        logger.info(
            "ACL loaded from %s: %d admins, %d scoped users, %d banned",
            self._path,
            len(admins),
            len(users),
            len(banned),
        )

    def ban(self, email: str) -> bool:
        """Add email to the banned list. Returns True if newly banned."""
        email = self._normalize(email)
        with self._lock:
            self._reload_if_stale()
            banned = list(self._data.get("banned", []))
            if email in banned:
                return False
            banned.append(email)
            self._data["banned"] = banned
        self._save_current()
        logger.info("Banned user: %s", email)
        return True

    def unban(self, email: str) -> bool:
        """Remove email from the banned list. Returns True if was banned."""
        email = self._normalize(email)
        with self._lock:
            self._reload_if_stale()
            banned = list(self._data.get("banned", []))
            if email not in banned:
                return False
            banned.remove(email)
            self._data["banned"] = banned
        self._save_current()
        logger.info("Unbanned user: %s", email)
        return True

    def list_banned(self) -> list[str]:
        """Return the current banned list."""
        with self._lock:
            self._reload_if_stale()
            return list(self._data.get("banned", []))

    def _save_current(self) -> None:
        """Write the current in-memory ACL data to disk."""
        with self._lock:
            payload = {
                "admins": sorted(self._data.get("admins", [])),
                "banned": sorted(self._data.get("banned", [])),
                "users": {
                    e: sorted(slugs)
                    for e, slugs in self._data.get("users", {}).items()
                },
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".acl-", dir=str(self._path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, sort_keys=True)
                os.replace(tmp, self._path)
            except Exception:
                os.unlink(tmp)
                raise
            self._mtime = self._path.stat().st_mtime
            self._loaded_once = True

    def save(self, admins: list[str], users: dict[str, list[str]]) -> None:
        """Atomically write a new ACL. Used by admin-facing tooling (none yet)."""
        payload = {
            "admins": sorted({e.strip().lower() for e in admins if e.strip()}),
            "banned": sorted(self._data.get("banned", [])),
            "users": {
                e.strip().lower(): sorted({s.strip() for s in slugs if s.strip()})
                for e, slugs in users.items()
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd, tmp = tempfile.mkstemp(prefix=".acl-", dir=str(self._path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, sort_keys=True)
                os.replace(tmp, self._path)
            except Exception:
                os.unlink(tmp)
                raise
            self._mtime = self._path.stat().st_mtime
            self._data = payload
            self._loaded_once = True

    # ---------------------------------------------------------------- Query

    def _normalize(self, email: str) -> str:
        return (email or "").strip().lower()

    def is_admin(self, email: str) -> bool:
        email = self._normalize(email)
        with self._lock:
            self._reload_if_stale()
            return email in self._data["admins"]

    def is_banned(self, email: str) -> bool:
        """True if the email is in the banned list."""
        email = self._normalize(email)
        with self._lock:
            self._reload_if_stale()
            return email in self._data.get("banned", [])

    def is_known(self, email: str) -> bool:
        """True if the email is either an admin or in the scoped users map, AND not banned."""
        email = self._normalize(email)
        with self._lock:
            self._reload_if_stale()
            if email in self._data.get("banned", []):
                return False
            return email in self._data["admins"] or email in self._data["users"]

    def allowed_clients(self, email: str) -> list[str] | Wildcard:
        """Return ``"*"`` for admins, or the list of allowed client slugs."""
        email = self._normalize(email)
        with self._lock:
            self._reload_if_stale()
            if email in self._data["admins"]:
                return WILDCARD
            return list(self._data["users"].get(email, []))

    def check(self, email: str, client_slug: str) -> AclDecision:
        email = self._normalize(email)
        client_slug = (client_slug or "").strip()
        with self._lock:
            self._reload_if_stale()
            if email in self._data["admins"]:
                return AclDecision(allowed=True, reason="admin")
            scoped = self._data["users"].get(email)
            if scoped is None:
                return AclDecision(allowed=False, reason="not_in_acl")
            if client_slug in scoped:
                return AclDecision(allowed=True, reason="allowlisted")
            return AclDecision(allowed=False, reason="client_not_allowed")

    def list_all_users(self) -> list[dict]:
        """Return all known users (admins + scoped) with their role."""
        with self._lock:
            self._reload_if_stale()
            result: list[dict] = []
            for email in self._data["admins"]:
                result.append({"email": email, "role": "admin"})
            for email in self._data["users"]:
                if email not in self._data["admins"]:
                    result.append({"email": email, "role": "user"})
            return result

    def filter_clients(self, email: str, clients: list[str]) -> list[str]:
        """Return the subset of ``clients`` the email may see."""
        allowed = self.allowed_clients(email)
        if allowed == WILDCARD:
            return list(clients)
        allowed_set = set(allowed)
        return [c for c in clients if c in allowed_set]
