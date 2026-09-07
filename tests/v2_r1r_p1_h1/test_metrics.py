from __future__ import annotations

import numpy as np
import pytest

from src.yggdrasil_v2.r1_revalidation.h1.metrics import (
    FAMILY_REGRESSION_LIMIT,
    PRIMARY_GAIN_THRESHOLD,
    classification_metrics,
    paired_bootstrap_gain,
    prediction_sha256,
    primary_comparison,
    route_metrics,
    trace_metrics,
)


def test_prediction_hash_and_classification_are_strict_and_stratified() -> None:
    ids = ["a", "b", "c", "d"]
    predictions = [0, 1, 2, 3]
    assert prediction_sha256(ids, predictions) == prediction_sha256(tuple(ids), tuple(predictions))
    result = classification_metrics(ids, predictions, predictions, ["numeric", "relation", "numeric", "relation"])
    assert result["accuracy"] == 1.0
    assert result["macro_accuracy"] == 1.0
    assert result["by_family"]["numeric"]["count"] == 2
    with pytest.raises(ValueError):
        prediction_sha256(["a", "a"], [0, 1])
    with pytest.raises(ValueError):
        classification_metrics(ids, [0, 1, 2, 9], predictions, ["numeric"] * 4)
    with pytest.raises(ValueError):
        classification_metrics(ids, predictions, predictions, ["numeric"] * 4)


def test_trace_and_route_metrics_validate_axes_and_binary_targets() -> None:
    result = trace_metrics([[0, 1, 4], [1, 1, 2]], [[0, 1, 4], [1, 0, 2]], ["numeric", "relation"])
    assert result["shape"] == [2, 3]
    assert result["record_exact"] == 0.5
    assert result["by_family"]["relation"]["final_accuracy"] == 1.0
    assert route_metrics([0, 1, 1, 0], [0, 1, 0, 0])["expert_load"] == {"0": 0.5, "1": 0.5}
    with pytest.raises(ValueError):
        trace_metrics([[0, 1], []], [[0, 1], []], ["numeric", "relation"])
    with pytest.raises(ValueError):
        trace_metrics([[0, 5]], [[0, 1]], ["numeric"])
    with pytest.raises(ValueError):
        route_metrics([0, 1], [0, 2])


def test_paired_bootstrap_is_deterministic_and_has_valid_ci() -> None:
    shared = [False] * 20
    mixed = [True] * 20
    left = paired_bootstrap_gain(shared, mixed, seed=123, resamples=10_000)
    right = paired_bootstrap_gain(shared, mixed, seed=123, resamples=10_000)
    assert left == right
    assert left["resamples"] == 10_000
    assert left["gain"] == 1.0
    assert left["ci95"] == {"lower": 1.0, "upper": 1.0}
    with pytest.raises(ValueError):
        paired_bootstrap_gain([True], [False, True], seed=1)
    with pytest.raises(ValueError):
        paired_bootstrap_gain([0.5], [True], seed=1)
    with pytest.raises(ValueError):
        paired_bootstrap_gain([True], [False], seed=1, resamples=999)
    with pytest.raises(ValueError):
        paired_bootstrap_gain([True], [False], seed=1, resamples=1000.0)


def test_primary_comparison_enforces_gain_ci_and_family_regression_boundaries() -> None:
    ids = [f"r{i}" for i in range(100)]
    families = ["numeric"] * 50 + ["relation"] * 50
    targets = [0] * 100
    shared = {"ids": ids, "targets": targets, "families": families, "predictions": [1] * 100}
    mixed_predictions = [0] * 100
    comparison = primary_comparison(
        shared,
        {**shared, "predictions": mixed_predictions},
        bootstrap_seed=7,
    )
    assert comparison["overall"]["gain"] == 1.0
    assert comparison["gate"]["passed"] is True
    assert comparison["gate"]["primary_gain_threshold"] == PRIMARY_GAIN_THRESHOLD
    assert comparison["gate"]["family_regression_limit"] == FAMILY_REGRESSION_LIMIT

    # Exact gain threshold passes gain comparison, while the one-success CI
    # correctly prevents a false PASS.
    exact_ids = [f"e{i}" for i in range(20)]
    exact_families = ["numeric"] * 10 + ["relation"] * 10
    exact_shared = {"ids": exact_ids, "targets": [0] * 20, "families": exact_families, "predictions": [1] * 20}
    exact_mixed = {**exact_shared, "predictions": [0] + [1] * 19}
    exact = primary_comparison(exact_shared, exact_mixed, bootstrap_seed=8)
    assert exact["overall"]["gain"] == PRIMARY_GAIN_THRESHOLD
    assert exact["gate"]["gain_pass"] is True
    assert exact["gate"]["ci_lower_pass"] is False
    assert exact["gate"]["passed"] is False

    with pytest.raises(ValueError):
        primary_comparison(shared, {**shared, "ids": ids[:-1]}, bootstrap_seed=1)
    with pytest.raises(ValueError):
        primary_comparison(shared, {**shared, "families": ["numeric"] * 100}, bootstrap_seed=1)
    with pytest.raises(ValueError):
        primary_comparison(shared, {**shared, "predictions": [0] * 100}, bootstrap_seed=1, primary_gain=-0.1)

