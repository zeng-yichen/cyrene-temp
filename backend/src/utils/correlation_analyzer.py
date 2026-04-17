"""CorrelationAnalyzer — Spearman rank correlation for post attributes vs engagement.

Multiple modules need to correlate post attributes with engagement outcomes.
This utility provides a shared implementation with small-sample handling.

Usage:
    from backend.src.utils.correlation_analyzer import correlate_with_engagement

    correlations = correlate_with_engagement(
        observations=scored_obs,
        attribute_extractor=lambda obs: {
            "hook_score": obs.get("cyrene_dimensions", {}).get("Hook Scroll-Stop", 0),
            "save_score": obs.get("cyrene_dimensions", {}).get("Save-Worthiness", 0),
        },
        min_n=5,
    )
    # → {"hook_score": 0.72, "save_score": 0.45}
"""

from __future__ import annotations

import logging
import math
from typing import Callable

logger = logging.getLogger(__name__)


def correlate_with_engagement(
    observations: list[dict],
    attribute_extractor: Callable[[dict], dict[str, float]],
    min_n: int = 5,
) -> dict[str, float]:
    """Spearman rank correlation between extracted attributes and reward.immediate.

    Args:
        observations: RuanMei scored observations (must have reward.immediate).
        attribute_extractor: Function that takes an observation dict and returns
            {attribute_name: numeric_value}. Return empty dict for observations
            missing the attributes.
        min_n: Minimum data points per attribute. Skip attributes with fewer.

    Returns:
        {attribute_name: correlation} where correlation is in [-1, 1].
        Only includes attributes with sufficient data.
    """
    # Collect per-attribute paired data
    attr_data: dict[str, list[tuple[float, float]]] = {}

    for obs in observations:
        reward = obs.get("reward", {}).get("immediate")
        if reward is None:
            continue

        try:
            attrs = attribute_extractor(obs)
        except Exception:
            continue

        if not attrs:
            continue

        for attr_name, attr_val in attrs.items():
            if attr_val is None or (isinstance(attr_val, float) and math.isnan(attr_val)):
                continue
            attr_data.setdefault(attr_name, []).append((float(attr_val), float(reward)))

    # Compute Spearman correlation per attribute
    results: dict[str, float] = {}

    for attr_name, pairs in attr_data.items():
        if len(pairs) < min_n:
            continue

        try:
            corr = _spearman_correlation(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
            )
            results[attr_name] = round(corr, 4)
        except Exception as e:
            logger.debug("[correlation] Failed for %s: %s", attr_name, e)

    return results


def correlate_binary_with_engagement(
    observations: list[dict],
    attribute_extractor: Callable[[dict], dict[str, bool]],
    min_n: int = 5,
) -> dict[str, dict]:
    """Effect size (Cohen's d) for binary attributes vs engagement.

    Useful for constitutional verifier principles (pass/fail → engagement).

    Returns:
        {attribute_name: {"effect_size": float, "mean_pass": float,
         "mean_fail": float, "n_pass": int, "n_fail": int, "significant": bool}}
    """
    attr_groups: dict[str, dict[str, list[float]]] = {}

    for obs in observations:
        reward = obs.get("reward", {}).get("immediate")
        if reward is None:
            continue
        if isinstance(reward, float) and (math.isnan(reward) or math.isinf(reward)):
            continue

        try:
            attrs = attribute_extractor(obs)
        except Exception:
            continue

        for attr_name, passed in attrs.items():
            groups = attr_groups.setdefault(attr_name, {"pass": [], "fail": []})
            if passed:
                groups["pass"].append(float(reward))
            else:
                groups["fail"].append(float(reward))

    results: dict[str, dict] = {}

    for attr_name, groups in attr_groups.items():
        n_pass = len(groups["pass"])
        n_fail = len(groups["fail"])

        if n_pass < min_n or n_fail < max(2, min_n // 2):
            continue

        mean_pass = sum(groups["pass"]) / n_pass
        mean_fail = sum(groups["fail"]) / n_fail

        # Pooled standard deviation
        var_pass = sum((x - mean_pass) ** 2 for x in groups["pass"]) / max(n_pass - 1, 1)
        var_fail = sum((x - mean_fail) ** 2 for x in groups["fail"]) / max(n_fail - 1, 1)
        pooled_sd = math.sqrt(
            ((n_pass - 1) * var_pass + (n_fail - 1) * var_fail)
            / max(n_pass + n_fail - 2, 1)
        )

        effect_size = (mean_pass - mean_fail) / pooled_sd if pooled_sd > 0 else 0.0

        results[attr_name] = {
            "effect_size": round(effect_size, 4),
            "mean_pass": round(mean_pass, 4),
            "mean_fail": round(mean_fail, 4),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "significant": abs(effect_size) >= 0.2 and (n_pass + n_fail) >= 10,
        }

    return results


# ------------------------------------------------------------------
# Spearman rank correlation (no scipy dependency)
# ------------------------------------------------------------------

def _spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 3:
        return 0.0

    n = len(x)
    rank_x = _rank(x)
    rank_y = _rank(y)

    # Pearson correlation of ranks
    mean_rx = sum(rank_x) / n
    mean_ry = sum(rank_y) / n

    num = sum((rx - mean_rx) * (ry - mean_ry) for rx, ry in zip(rank_x, rank_y))
    den_x = math.sqrt(sum((rx - mean_rx) ** 2 for rx in rank_x))
    den_y = math.sqrt(sum((ry - mean_ry) ** 2 for ry in rank_y))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def _rank(values: list[float]) -> list[float]:
    """Compute ranks with average tie-breaking."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1

    return ranks
