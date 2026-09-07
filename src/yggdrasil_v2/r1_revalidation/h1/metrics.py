from __future__ import annotations

"""Deterministic metric and paired-comparison helpers for P1-H1."""

import hashlib
import json
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


FAMILIES = ("numeric", "relation")
TRACE_MAX = 4
ROUTE_VALUES = (0, 1)
PRIMARY_GAIN_THRESHOLD = 0.05
FAMILY_REGRESSION_LIMIT = 0.02


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _validate_ids(ids: Sequence[str]) -> None:
    if len(ids) == 0 or any(not isinstance(record_id, str) or not record_id for record_id in ids):
        raise ValueError("metric ids must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("metric ids must be unique")


def _integer_array(values: Sequence[int], name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integers")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of integers") from exc
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in raw):
        raise ValueError(f"{name} must contain integers only")
    return np.asarray(raw, dtype=np.int64)


def _binary_array(values: Sequence[int], name: str) -> np.ndarray:
    array = _integer_array(values, name)
    if array.ndim != 1 or bool(np.any((array < 0) | (array > 1))):
        raise ValueError(f"{name} must contain only binary values")
    return array


def _correctness_array(values: Sequence[bool], name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain binary correctness values")
    try:
        raw = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain binary correctness values") from exc
    if any(not isinstance(value, (bool, np.bool_)) and (isinstance(value, (float, np.floating)) or not isinstance(value, Integral)) for value in raw):
        raise ValueError(f"{name} must contain binary correctness values")
    array = np.asarray([int(value) for value in raw], dtype=np.int64)
    if array.ndim != 1 or bool(np.any((array < 0) | (array > 1))):
        raise ValueError(f"{name} must contain binary correctness values")
    return array


def _trace_array(values: Sequence[Sequence[int]], name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a rectangular integer array")
    try:
        rows = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a rectangular integer array") from exc
    if not rows:
        raise ValueError(f"{name} must be non-empty")
    normalized: list[list[int]] = []
    width: int | None = None
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ValueError(f"{name} must be a rectangular integer array")
        try:
            values_row = list(row)
        except TypeError as exc:
            raise ValueError(f"{name} must be a rectangular integer array") from exc
        if width is None:
            width = len(values_row)
        if len(values_row) != width or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in values_row):
            raise ValueError(f"{name} must be a rectangular integer array")
        normalized.append([int(value) for value in values_row])
    if not width:
        raise ValueError(f"{name} must have a non-empty time axis")
    return np.asarray(normalized, dtype=np.int64)


def prediction_sha256(ids: Sequence[str], predictions: Sequence[int]) -> str:
    if len(ids) != len(predictions):
        raise ValueError("prediction ids/values length mismatch")
    _validate_ids(ids)
    values = _integer_array(predictions, "predictions")
    return hashlib.sha256(
        _canonical(
            [
                {"id": record_id, "prediction": int(prediction)}
                for record_id, prediction in zip(ids, values.tolist(), strict=True)
            ]
        )
    ).hexdigest().upper()


def classification_metrics(
    ids: Sequence[str],
    predictions: Sequence[int],
    targets: Sequence[int],
    families: Sequence[str],
) -> dict[str, Any]:
    lengths = {len(ids), len(predictions), len(targets), len(families)}
    if len(lengths) != 1 or len(ids) == 0:
        raise ValueError("classification metric inputs must be equal non-zero length")
    _validate_ids(ids)
    if any(family not in FAMILIES for family in families):
        raise ValueError("unknown H1 metric family")
    predicted = _integer_array(predictions, "predictions")
    expected = _integer_array(targets, "targets")
    if bool(np.any((predicted < 0) | (predicted > 3) | (expected < 0) | (expected > 3))):
        raise ValueError("classification labels must be in the range 0..3")
    correct = predicted == expected
    by_family: dict[str, Any] = {}
    for family in FAMILIES:
        selected = np.asarray([value == family for value in families], dtype=np.bool_)
        if not bool(selected.any()):
            raise ValueError(f"metric input omits family {family}")
        by_family[family] = {
            "count": int(selected.sum()),
            "correct": int(correct[selected].sum()),
            "accuracy": float(correct[selected].mean()),
        }
    return {
        "count": len(ids),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(
            sum(by_family[family]["accuracy"] for family in FAMILIES) / len(FAMILIES)
        ),
        "by_family": by_family,
        "prediction_sha256": prediction_sha256(ids, predictions),
        "correct_vector": [bool(value) for value in correct.tolist()],
    }


def trace_metrics(
    predictions: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    families: Sequence[str],
) -> dict[str, Any]:
    if len(predictions) == 0 or len(predictions) != len(targets) or len(targets) != len(families):
        raise ValueError("trace metric inputs must be equal non-zero length")
    try:
        predicted = _trace_array(predictions, "trace predictions")
        expected = _trace_array(targets, "trace targets")
    except (TypeError, ValueError) as exc:
        raise ValueError("trace predictions/targets must be rectangular integer arrays") from exc
    if predicted.shape != expected.shape or predicted.ndim != 2 or predicted.shape[1] == 0:
        raise ValueError("trace predictions/targets must have equal [N,T] shape")
    if bool(np.any((predicted < 0) | (predicted > TRACE_MAX) | (expected < 0) | (expected > TRACE_MAX))):
        raise ValueError("trace labels must be in the range 0..4")
    if any(family not in FAMILIES for family in families):
        raise ValueError("unknown H1 metric family")
    exact = predicted == expected
    record_exact = exact.all(axis=1)
    final_exact = exact[:, -1]
    by_family: dict[str, Any] = {}
    for family in FAMILIES:
        selected = np.asarray([value == family for value in families], dtype=np.bool_)
        if not bool(selected.any()):
            raise ValueError(f"trace input omits family {family}")
        by_family[family] = {
            "count": int(selected.sum()),
            "cell_accuracy": float(exact[selected].mean()),
            "record_exact": float(record_exact[selected].mean()),
            "final_accuracy": float(final_exact[selected].mean()),
        }
    return {
        "shape": list(predicted.shape),
        "cell_accuracy": float(exact.mean()),
        "record_exact": float(record_exact.mean()),
        "final_accuracy": float(final_exact.mean()),
        "by_family": by_family,
    }


def route_metrics(
    assignments: Sequence[int],
    targets: Sequence[int],
) -> dict[str, Any]:
    if len(assignments) == 0 or len(assignments) != len(targets):
        raise ValueError("route metric inputs must be equal non-zero length")
    predicted = _binary_array(assignments, "route assignments")
    expected = _binary_array(targets, "route targets")
    return {
        "count": len(assignments),
        "accuracy": float((predicted == expected).mean()),
        "expert_load": {
            "0": float((predicted == 0).mean()),
            "1": float((predicted == 1).mean()),
        },
    }


def paired_bootstrap_gain(
    shared_correct: Sequence[bool],
    mixed_correct: Sequence[bool],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    if len(shared_correct) != len(mixed_correct) or len(shared_correct) == 0:
        raise ValueError("paired correctness vectors must have equal non-zero length")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("bootstrap seed must be an integer")
    if isinstance(resamples, bool) or not isinstance(resamples, Integral) or resamples < 1000:
        raise ValueError("H1 paired bootstrap requires at least 1000 resamples")
    shared = _correctness_array(shared_correct, "shared correctness")
    mixed = _correctness_array(mixed_correct, "mixed correctness")
    if shared.ndim != 1 or mixed.ndim != 1:
        raise ValueError("paired correctness vectors must be one-dimensional")
    shared = shared.astype(np.float64)
    mixed = mixed.astype(np.float64)
    delta = mixed - shared
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    # Chunked sampling keeps the formal metric bounded for large heldout sets.
    cursor = 0
    while cursor < resamples:
        count = min(512, resamples - cursor)
        indices = generator.integers(0, len(delta), size=(count, len(delta)))
        means[cursor : cursor + count] = delta[indices].mean(axis=1)
        cursor += count
    lower, upper = np.quantile(means, (0.025, 0.975), method="linear")
    return {
        "pairs": len(delta),
        "resamples": resamples,
        "seed": seed,
        "shared_accuracy": float(shared.mean()),
        "mixed_accuracy": float(mixed.mean()),
        "gain": float(delta.mean()),
        "ci95": {"lower": float(lower), "upper": float(upper)},
        "discordant": {
            "mixed_only_correct": int(np.sum((mixed == 1) & (shared == 0))),
            "shared_only_correct": int(np.sum((shared == 1) & (mixed == 0))),
        },
    }


def primary_comparison(
    shared: Mapping[str, Any],
    mixed: Mapping[str, Any],
    *,
    bootstrap_seed: int,
    primary_gain: float = PRIMARY_GAIN_THRESHOLD,
    family_regression_limit: float = FAMILY_REGRESSION_LIMIT,
) -> dict[str, Any]:
    if not isinstance(primary_gain, (int, float)) or not np.isfinite(primary_gain) or primary_gain < 0:
        raise ValueError("primary gain threshold must be a finite non-negative number")
    if not isinstance(family_regression_limit, (int, float)) or not np.isfinite(family_regression_limit) or family_regression_limit < 0:
        raise ValueError("family regression limit must be a finite non-negative number")
    required = {"ids", "targets", "families", "predictions"}
    if not required.issubset(shared) or not required.issubset(mixed):
        raise ValueError("primary comparison requires ids, targets, families and predictions")
    shared_ids = list(shared["ids"])
    mixed_ids = list(mixed["ids"])
    if shared_ids != mixed_ids:
        raise ValueError("H1 primary comparison requires identical ordered ids")
    if list(shared["targets"]) != list(mixed["targets"]):
        raise ValueError("H1 primary comparison requires identical targets")
    if list(shared["families"]) != list(mixed["families"]):
        raise ValueError("H1 primary comparison requires identical families")
    shared_predictions = np.asarray(shared["predictions"], dtype=np.int64)
    mixed_predictions = np.asarray(mixed["predictions"], dtype=np.int64)
    targets = np.asarray(shared["targets"], dtype=np.int64)
    families = list(shared["families"])
    _validate_ids(shared_ids)
    if any(family not in FAMILIES for family in families):
        raise ValueError("unknown H1 metric family")
    if any(families.count(family) == 0 for family in FAMILIES):
        raise ValueError("primary comparison requires both families")
    primary_targets = _integer_array(shared["targets"], "targets")
    primary_shared_predictions = _integer_array(shared["predictions"], "shared predictions")
    primary_mixed_predictions = _integer_array(mixed["predictions"], "mixed predictions")
    if primary_targets.size != len(shared_ids) or primary_shared_predictions.size != len(shared_ids) or primary_mixed_predictions.size != len(shared_ids):
        raise ValueError("primary comparison vectors must match ids length")
    if bool(np.any((primary_targets < 0) | (primary_targets > 3) | (primary_shared_predictions < 0) | (primary_shared_predictions > 3) | (primary_mixed_predictions < 0) | (primary_mixed_predictions > 3))):
        raise ValueError("primary classification labels must be in the range 0..3")
    overall = paired_bootstrap_gain(
        (shared_predictions == targets).tolist(),
        (mixed_predictions == targets).tolist(),
        seed=bootstrap_seed,
    )
    by_family: dict[str, Any] = {}
    for index, family in enumerate(FAMILIES):
        selected = np.asarray([value == family for value in families], dtype=np.bool_)
        by_family[family] = paired_bootstrap_gain(
            (shared_predictions[selected] == targets[selected]).tolist(),
            (mixed_predictions[selected] == targets[selected]).tolist(),
            seed=bootstrap_seed + index + 1,
        )
    family_regression = {
        family: by_family[family]["gain"] >= -float(family_regression_limit)
        for family in FAMILIES
    }
    macro_gain = float(sum(by_family[family]["gain"] for family in FAMILIES) / len(FAMILIES))
    gate = {
        "primary_gain_threshold": float(primary_gain),
        "family_regression_limit": float(family_regression_limit),
        "gain_pass": macro_gain >= float(primary_gain),
        "ci_lower_pass": overall["ci95"]["lower"] > 0.0,
        "family_regression_pass": family_regression,
        "passed": (
            macro_gain >= float(primary_gain)
            and overall["ci95"]["lower"] > 0.0
            and all(family_regression.values())
        ),
    }
    return {"overall": overall, "macro_gain": macro_gain, "by_family": by_family, "gate": gate}


__all__ = [
    "FAMILIES",
    "FAMILY_REGRESSION_LIMIT",
    "PRIMARY_GAIN_THRESHOLD",
    "classification_metrics",
    "paired_bootstrap_gain",
    "prediction_sha256",
    "primary_comparison",
    "route_metrics",
    "trace_metrics",
]
