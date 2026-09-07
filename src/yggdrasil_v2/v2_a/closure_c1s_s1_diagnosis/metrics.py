from __future__ import annotations

"""Small, deterministic statistics for the C1S S1 failure diagnosis.

This module deliberately has no model, checkpoint, or filesystem dependency.
All estimates are record-level.  In particular, a family summary is computed
before an overall descriptive summary so that a large family cannot hide a
failed CPS or ERE cell.
"""

from collections import defaultdict
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


WILSON_Z_95 = 1.959963984540054


def _array(value: Any, *, name: str, ndim: int | None = None, dtype: Any = float) -> np.ndarray:
    """Convert an input to a finite CPU array and check its rank."""

    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid numeric array") from exc
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {result.ndim}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _vector(value: Any, *, name: str, dtype: Any = float) -> np.ndarray:
    return _array(value, name=name, ndim=1, dtype=dtype)


def _plain(value: Any) -> Any:
    """Return JSON-native values and reject non-finite values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite report value")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in list(value)]
    raise TypeError(f"report contains non JSON-native value: {type(value).__name__}")


def finite_json(value: Any) -> Any:
    """Validate and convert a report to JSON-native scalars/lists/dicts."""

    return _plain(value)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        raise ValueError("cannot take a percentile of an empty sample")
    return float(np.quantile(values, q, method="linear"))


def wilson_interval(
    successes: int | float,
    total: int,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return a two-sided Wilson interval for a binomial proportion.

    ``alpha`` is a statistical confidence parameter, not a research Gate
    threshold.  The caller supplies any study-specific decision threshold.
    """

    if isinstance(total, bool) or not isinstance(total, (int, np.integer)) or int(total) < 0:
        raise ValueError("total must be a non-negative integer")
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0,1)")
    n = int(total)
    s = float(successes)
    if not math.isfinite(s) or s < 0.0 or s > n:
        raise ValueError("successes must be finite and in [0,total]")
    if n == 0:
        return {"successes": 0, "total": 0, "point": 0.0, "lower": 0.0, "upper": 0.0, "alpha": float(alpha)}
    # scipy is intentionally not required; this is the inverse-normal
    # approximation used by the rest of the project for binomial intervals.
    z = WILSON_Z_95 if abs(float(alpha) - 0.05) < 1.0e-15 else _normal_quantile(1.0 - float(alpha) / 2.0)
    p = s / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return finite_json({
        "successes": int(round(s)) if s.is_integer() else s,
        "total": n,
        "point": p,
        "lower": max(0.0, centre - radius),
        "upper": min(1.0, centre + radius),
        "alpha": float(alpha),
    })


def _normal_quantile(probability: float) -> float:
    """Acklam-free normal quantile using the standard-library erf inverse."""

    # Python's standard library has no ndtri.  A short binary search is stable
    # enough for confidence intervals and avoids an additional dependency.
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0,1)")
    lo, hi = -9.0, 9.0
    scale = math.sqrt(2.0)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / scale))
        if cdf < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def answer_margin(logits: Any, correct: Any) -> list[float]:
    """Compute ``logit(correct) - logsumexp(logits[wrong])`` per record."""

    values = _array(logits, name="logits", ndim=2, dtype=float)
    indices = _vector(correct, name="correct", dtype=float)
    if values.shape[0] != indices.shape[0]:
        raise ValueError("logits and correct must have the same number of records")
    if values.shape[1] < 2:
        raise ValueError("logits must contain at least two classes")
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError("correct indices must be integers")
    target = indices.astype(np.int64)
    if np.any(target < 0) or np.any(target >= values.shape[1]):
        raise ValueError("correct index outside logits class range")
    correct_logits = values[np.arange(values.shape[0]), target]
    wrong = values.copy()
    wrong[np.arange(values.shape[0]), target] = -np.inf
    maximum = np.max(wrong, axis=1)
    logsumexp_wrong = maximum + np.log(np.exp(wrong - maximum[:, None]).sum(axis=1))
    return finite_json(correct_logits - logsumexp_wrong)


def paired_cluster_bootstrap(
    base: Any,
    intervention: Any,
    clusters: Sequence[Any],
    *,
    seed: int,
    samples: int = 2_000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap the paired mean ``base - intervention`` by whole clusters."""

    left = _vector(base, name="base", dtype=float)
    right = _vector(intervention, name="intervention", dtype=float)
    if left.shape != right.shape or len(clusters) != left.size:
        raise ValueError("paired values and clusters must have equal length")
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or int(samples) <= 0:
        raise ValueError("samples must be a positive integer")
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0,1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    differences = left - right
    cluster_keys = [repr(cluster) for cluster in clusters]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(cluster_keys):
        grouped[key].append(index)
    groups = list(grouped.values())
    if not groups:
        raise ValueError("paired bootstrap requires at least one record")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=float)
    for draw in range(int(samples)):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([np.asarray(groups[int(index)], dtype=np.int64) for index in chosen])
        draws[draw] = float(differences[indices].mean())
    return finite_json({
        "point": float(differences.mean()),
        "lower": _percentile(draws, float(alpha) / 2.0),
        "upper": _percentile(draws, 1.0 - float(alpha) / 2.0),
        "clusters": len(groups),
        "records": int(left.size),
        "samples": int(samples),
        "seed": int(seed),
        "alpha": float(alpha),
    })


def balanced_accuracy(predicted: Any, target: Any) -> dict[str, Any]:
    """Return binary balanced accuracy with explicit missing-class status."""

    observed = _vector(predicted, name="predicted", dtype=float)
    expected = _vector(target, name="target", dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("predicted and target must have equal shape")
    if not np.isin(observed, (0.0, 1.0)).all() or not np.isin(expected, (0.0, 1.0)).all():
        raise ValueError("balanced accuracy expects binary values 0/1")
    positive = expected == 1.0
    negative = expected == 0.0
    positives = int(positive.sum())
    negatives = int(negative.sum())
    tpr = float((observed[positive] == 1.0).mean()) if positives else None
    tnr = float((observed[negative] == 0.0).mean()) if negatives else None
    value = (tpr + tnr) / 2.0 if tpr is not None and tnr is not None else None
    return finite_json({
        "n": int(expected.size),
        "correct": int((observed == expected).sum()),
        "accuracy": wilson_interval(int((observed == expected).sum()), int(expected.size)),
        "positives": positives,
        "negatives": negatives,
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "balanced_accuracy": value,
        "both_classes": bool(positives and negatives),
    })


def family_summary(
    values: Sequence[Any],
    families: Sequence[Any],
    metric: Callable[[Sequence[Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute a metric per family, then a descriptive overall metric."""

    observations = list(values)
    labels = list(families)
    if len(observations) != len(labels):
        raise ValueError("values and families must have equal length")
    grouped: dict[str, list[Any]] = defaultdict(list)
    for value, family in zip(observations, labels):
        grouped[str(family).upper()].append(value)
    if not grouped:
        raise ValueError("family summary requires at least one record")
    per_family = {family: finite_json(metric(group)) for family, group in sorted(grouped.items())}
    return finite_json({
        "families": per_family,
        "overall": finite_json(metric(observations)),
        "family_count": len(per_family),
        "records": len(observations),
    })


def family_binary_summary(predicted: Any, target: Any, families: Sequence[Any]) -> dict[str, Any]:
    """Balanced-accuracy summary with CPS/ERE (or supplied) family cells."""

    observed = _vector(predicted, name="predicted", dtype=float)
    expected = _vector(target, name="target", dtype=float)
    if len(families) != observed.size:
        raise ValueError("families length differs from binary observations")
    return family_summary(
        list(zip(observed.tolist(), expected.tolist())),
        families,
        lambda pairs: balanced_accuracy([p[0] for p in pairs], [p[1] for p in pairs]),
    )


# Names used by adjacent diagnostic code.  They are aliases, not alternate
# implementations, so the validation and deterministic sampling stay shared.
answer_margins = answer_margin
paired_cluster_bootstrap_drop = paired_cluster_bootstrap
wilson_ci = wilson_interval
family_stratified_summary = family_summary


__all__ = [
    "WILSON_Z_95",
    "answer_margin",
    "answer_margins",
    "balanced_accuracy",
    "family_binary_summary",
    "family_summary",
    "family_stratified_summary",
    "finite_json",
    "paired_cluster_bootstrap",
    "paired_cluster_bootstrap_drop",
    "wilson_ci",
    "wilson_interval",
]
