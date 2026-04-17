"""AdaptiveConfig — three-tier cascade for data-driven thresholds.

Generalized pattern: per-client data → cross-client aggregate → hard-coded default.
When insufficient data exists, resolves to exactly the current hard-coded values.
Zero behavioral change on day one.

Usage:
    class MyConfig(AdaptiveConfig):
        MODULE_NAME = "my_module"
        def get_defaults(self) -> dict: return {"threshold": 0.5}
        def sufficient_data(self, company: str) -> bool: ...
        def compute_from_client(self, company: str) -> dict: ...
        def compute_from_aggregate(self) -> dict: ...

    cfg = MyConfig()
    params = cfg.resolve("example-client")
    # → client-specific if enough data, else aggregate, else {"threshold": 0.5}
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.src.db import vortex as P

logger = logging.getLogger(__name__)

_DEFAULT_RECOMPUTE_INTERVAL = 3600  # 1 hour


class AdaptiveConfig(ABC):
    """Base class for data-driven configuration with three-tier cascade."""

    MODULE_NAME: str = "base"  # Override in subclass — used as cache key

    def __init__(self, recompute_interval: int = _DEFAULT_RECOMPUTE_INTERVAL):
        self._recompute_interval = recompute_interval

    # ------------------------------------------------------------------
    # Abstract methods — subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def get_defaults(self) -> dict:
        """Return the hard-coded default values (current behavior)."""
        ...

    @abstractmethod
    def sufficient_data(self, company: str) -> bool:
        """Return True if the client has enough data for per-client config."""
        ...

    @abstractmethod
    def compute_from_client(self, company: str) -> dict:
        """Compute config from this client's own data."""
        ...

    @abstractmethod
    def compute_from_aggregate(self) -> dict:
        """Compute config from cross-client aggregate data.

        Return empty dict if aggregate data is also insufficient.
        """
        ...

    # ------------------------------------------------------------------
    # Resolve: three-tier cascade
    # ------------------------------------------------------------------

    def resolve(self, company: str) -> dict:
        """Three-tier cascade: client → aggregate → defaults.

        Checks cache first. Recomputes if stale or missing.
        """
        cached = self._load_cached(company)
        if cached is not None:
            return cached

        if self.sufficient_data(company):
            try:
                config = self.compute_from_client(company)
                config["_tier"] = "client"
                config["_computed_at"] = _now()
                self._save_cached(company, config)
                self._log_computation(company, config)
                logger.info("[%s] Resolved from client data for %s", self.MODULE_NAME, company)
                return config
            except Exception as e:
                logger.warning("[%s] Client compute failed for %s: %s", self.MODULE_NAME, company, e)

        try:
            agg = self.compute_from_aggregate()
            if agg:
                agg["_tier"] = "aggregate"
                agg["_computed_at"] = _now()
                self._save_cached(company, agg)
                self._log_computation(company, agg)
                logger.info("[%s] Resolved from aggregate for %s", self.MODULE_NAME, company)
                return agg
        except Exception as e:
            logger.warning("[%s] Aggregate compute failed: %s", self.MODULE_NAME, e)

        defaults = self.get_defaults()
        defaults["_tier"] = "default"
        defaults["_computed_at"] = _now()
        return defaults

    def recompute(self, company: str) -> dict:
        """Force recompute, ignoring cache."""
        self._invalidate_cache(company)
        return self.resolve(company)

    # ------------------------------------------------------------------
    # Cache — persisted to memory/{company}/adaptive_config.json
    # ------------------------------------------------------------------

    def _config_path(self, company: str) -> Path:
        return P.memory_dir(company) / "adaptive_config.json"

    def _load_cached(self, company: str) -> dict | None:
        path = self._config_path(company)
        if not path.exists():
            return None
        try:
            all_configs = json.loads(path.read_text(encoding="utf-8"))
            entry = all_configs.get(self.MODULE_NAME)
            if entry is None:
                return None

            # Check staleness
            computed_at = entry.get("_computed_at", "")
            if computed_at:
                try:
                    dt = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
                    age_seconds = (datetime.now(timezone.utc) - dt).total_seconds()
                    if age_seconds > self._recompute_interval:
                        return None  # Stale
                except (ValueError, TypeError):
                    return None

            return entry
        except Exception:
            return None

    def _save_cached(self, company: str, config: dict) -> None:
        path = self._config_path(company)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            all_configs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            all_configs = {}
        all_configs[self.MODULE_NAME] = config
        # Atomic write: temp file + rename prevents concurrent clobbering
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(all_configs, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(path)

    def _invalidate_cache(self, company: str) -> None:
        path = self._config_path(company)
        if not path.exists():
            return
        try:
            all_configs = json.loads(path.read_text(encoding="utf-8"))
            all_configs.pop(self.MODULE_NAME, None)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(all_configs, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.rename(path)
        except Exception:
            pass


    # ------------------------------------------------------------------
    # History log — append-only JSONL for tracking config evolution
    # ------------------------------------------------------------------

    def _log_computation(self, company: str, config: dict) -> None:
        """Append a timestamped entry to adaptive_config_history.jsonl."""
        path = P.memory_dir(company) / "adaptive_config_history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Strip large nested dicts for the log (keep it compact)
        log_config = {k: v for k, v in config.items() if not k.startswith("_")}
        entry = {
            "module": self.MODULE_NAME,
            "tier": config.get("_tier", "unknown"),
            "computed_at": config.get("_computed_at", _now()),
            "config": log_config,
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass


def soft_bound(value: float, history: list[float], default: float, z_threshold: float = 3.0) -> float:
    """Accept any value within z_threshold standard deviations of historical mean.

    If history is empty or too short, accept any value.
    Log a warning (don't clip) if outside range.
    Always returns the learned value — never clips.
    """
    if len(history) < 5:
        return value
    mean = sum(history) / len(history)
    std = (sum((x - mean) ** 2 for x in history) / len(history)) ** 0.5
    if std == 0:
        return value
    z = abs(value - mean) / std
    if z > z_threshold:
        logger.warning(
            "Learned value %.4f is %.1f std from mean (%.4f ± %.4f)",
            value, z, mean, std,
        )
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
