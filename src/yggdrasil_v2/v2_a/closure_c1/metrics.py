"""Small, dependency-free statistics used by the C1 evaluator.

The functions in this module deliberately operate on observations supplied by
the caller.  They do not know anything about a model or its hidden state.  In
particular, paired effects are clustered at the record (or pair) level and
never at token level.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

BOOTSTRAP_SEED = 2026082503
BOOTSTRAP_SAMPLES = 10_000
WILSON_Z = 1.959963984540054


def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite metric")
    return value


def wilson_interval(successes: int | float, total: int, *, z: float = WILSON_Z) -> dict[str, float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative integer")
    if total == 0:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "successes": 0.0, "total": 0}
    s = _finite(successes)
    if s < 0 or s > total:
        raise ValueError("successes outside [0,total]")
    z = _finite(z)
    if z < 0:
        raise ValueError("z must be non-negative")
    p = s / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {"point": p, "lower": max(0.0, centre - radius), "upper": min(1.0, centre + radius), "successes": s, "total": total}


def _cluster_rows(values: Sequence[float], clusters: Sequence[Any] | None) -> list[list[float]]:
    if clusters is None:
        clusters = list(range(len(values)))
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    grouped: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters):
        v = _finite(value)
        key = json.dumps(cluster, ensure_ascii=False, sort_keys=True, default=str)
        grouped.setdefault(key, []).append(v)
    return list(grouped.values())


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * q
    low = int(math.floor(position)); high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence[Any] | None = None, *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    """Cluster bootstrap of a mean with a deterministic percentile 95% CI."""
    rows = _cluster_rows(values, clusters)
    point = sum(sum(row) for row in rows) / sum(len(row) for row in rows) if rows else 0.0
    if not rows or samples <= 0:
        return {"point": point, "lower": point, "upper": point, "clusters": len(rows), "samples": max(0, samples), "seed": seed}
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        selected = [rows[rng.randrange(len(rows))] for _ in rows]
        count = sum(len(row) for row in selected)
        boot.append(sum(sum(row) for row in selected) / count)
    boot.sort()
    return {"point": point, "lower": _percentile(boot, 0.025), "upper": _percentile(boot, 0.975), "clusters": len(rows), "samples": samples, "seed": seed}


def paired_cluster_bootstrap_drop(base: Sequence[float], intervention: Sequence[float], clusters: Sequence[Any] | None = None, *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    """Bootstrap ``mean(base - intervention)`` while resampling whole clusters."""
    if len(base) != len(intervention):
        raise ValueError("paired observations must have equal length")
    if clusters is None:
        clusters = list(range(len(base)))
    if len(clusters) != len(base):
        raise ValueError("clusters must have equal length")
    grouped: dict[str, list[tuple[float, float]]] = {}
    for b, i, cluster in zip(base, intervention, clusters):
        pair = (_finite(b), _finite(i))
        key = json.dumps(cluster, ensure_ascii=False, sort_keys=True, default=str)
        grouped.setdefault(key, []).append(pair)
    rows = list(grouped.values())
    diffs = [b - i for b, i in zip(base, intervention)]
    point = sum(diffs) / len(diffs) if diffs else 0.0
    if not rows or samples <= 0:
        return {"point": point, "lower": point, "upper": point, "clusters": len(rows), "samples": max(0, samples), "seed": seed}
    rng = random.Random(seed); boot: list[float] = []
    for _ in range(samples):
        selected = [rows[rng.randrange(len(rows))] for _ in rows]
        flat = [b - i for row in selected for b, i in row]
        boot.append(sum(flat) / len(flat))
    boot.sort()
    return {"point": point, "lower": _percentile(boot, 0.025), "upper": _percentile(boot, 0.975), "clusters": len(rows), "samples": samples, "seed": seed}


def paired_drop(base: Sequence[float], intervention: Sequence[float], clusters: Sequence[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Alias used by the evaluator and training runner."""
    return paired_cluster_bootstrap_drop(base, intervention, clusters, **kwargs)


def wilson_ci(successes: int | float, total: int, **kwargs: Any) -> dict[str, float]:
    return wilson_interval(successes, total, **kwargs)


def bootstrap_paired_drop(base: Sequence[float], intervention: Sequence[float], clusters: Sequence[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    return paired_cluster_bootstrap_drop(base, intervention, clusters, **kwargs)


def finite_json(value: Any) -> Any:
    """Validate that a report contains only JSON-native finite values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        _finite(value); return value
    if isinstance(value, Mapping):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    raise TypeError(f"non JSON-native value: {type(value).__name__}")


__all__ = ["BOOTSTRAP_SEED", "BOOTSTRAP_SAMPLES", "WILSON_Z", "wilson_interval", "wilson_ci", "cluster_bootstrap_mean", "paired_cluster_bootstrap_drop", "bootstrap_paired_drop", "paired_drop", "finite_json"]
