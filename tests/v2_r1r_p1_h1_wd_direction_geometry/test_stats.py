from __future__ import annotations

import json

import numpy as np
import pytest

from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.stats import (
    covariance_spectrum,
    cross_split_alignment,
    permutation_feature_fit_score,
    permutation_fit_score,
    permutation_score,
    principal_angle_summary,
    rademacher_sign_null,
    record_cluster_bootstrap,
    record_cluster_metrics,
    site_macro_record_bootstrap,
)


def _assert_jsonable(value: object) -> None:
    json.dumps(value, allow_nan=False)


def test_record_cluster_metrics_reports_micro_and_macro_metrics() -> None:
    target = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    prediction = target * 0.5
    result = record_cluster_metrics(
        target,
        prediction,
        clusters=["a", "a", "b", "b"],
    )

    assert result["records"] == 4
    assert result["clusters"] == 2
    assert result["micro_sse"] == pytest.approx(0.75)
    assert result["micro_baseline_sse"] == pytest.approx(1.0)
    assert result["micro_explained_gain"] == pytest.approx(0.25)
    assert result["cluster_macro_explained_gain"] == pytest.approx(0.25)
    _assert_jsonable(result)


def test_cluster_bootstrap_is_reproducible_and_resamples_whole_clusters() -> None:
    target = np.arange(12.0).reshape(6, 2)
    prediction = np.zeros_like(target)
    clusters = ["a", "a", "b", "b", "c", "c"]
    first = record_cluster_bootstrap(
        target,
        prediction,
        clusters=clusters,
        seed=19,
        resamples=64,
    )
    second = record_cluster_bootstrap(
        target,
        prediction,
        clusters=clusters,
        seed=19,
        resamples=64,
    )

    assert first == second
    assert first["cluster_count"] == 3
    assert set(first["ci95"]) == {"micro_explained_gain", "cluster_macro_explained_gain"}
    _assert_jsonable(first)


def test_permutation_fit_score_is_deterministic_and_stratified() -> None:
    features = np.arange(8.0).reshape(8, 1)
    targets = (2.0 * features + 1.0)
    strata = ["left"] * 4 + ["right"] * 4

    def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        design = np.column_stack([np.ones(len(x)), x[:, 0]])
        return tuple(np.linalg.lstsq(design, y[:, 0], rcond=None)[0])

    def score(model: tuple[float, float], x: np.ndarray, y: np.ndarray) -> float:
        prediction = model[0] + model[1] * x[:, 0]
        return float(1.0 - np.mean((prediction - y[:, 0]) ** 2))

    first = permutation_fit_score(
        features,
        targets,
        fit,
        score,
        strata=strata,
        seed=7,
        resamples=32,
    )
    second = permutation_fit_score(
        features,
        targets,
        fit,
        score,
        strata=strata,
        seed=7,
        resamples=32,
    )

    assert first == second
    assert first["kind"] == "fit-null"
    assert first["observed"] > first["null_mean"]
    assert len(first["scores"]) == 32
    _assert_jsonable(first)


def test_feature_permutation_fit_score_fixes_targets_and_is_deterministic() -> None:
    features = np.arange(8.0).reshape(8, 1)
    targets = 2.0 * features + 1.0
    eval_features = np.arange(8.5, 12.5).reshape(4, 1)
    eval_targets = 2.0 * eval_features + 1.0
    strata = ["left"] * 4 + ["right"] * 4

    def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        design = np.column_stack([np.ones(len(x)), x[:, 0]])
        return tuple(np.linalg.lstsq(design, y[:, 0], rcond=None)[0])

    def score(model: tuple[float, float], x: np.ndarray, y: np.ndarray) -> float:
        prediction = model[0] + model[1] * x[:, 0]
        return float(1.0 - np.mean((prediction - y[:, 0]) ** 2))

    first = permutation_feature_fit_score(
        features,
        targets,
        fit,
        score,
        eval_features=eval_features,
        eval_targets=eval_targets,
        strata=strata,
        seed=7,
        resamples=32,
    )
    second = permutation_feature_fit_score(
        features,
        targets,
        fit,
        score,
        eval_features=eval_features,
        eval_targets=eval_targets,
        strata=strata,
        seed=7,
        resamples=32,
    )

    assert first == second
    assert first["kind"] == "feature-fit-null"
    assert first["observed"] > first["null_mean"]
    assert len(first["scores"]) == 32
    _assert_jsonable(first)


def test_site_macro_bootstrap_resamples_records_not_sites() -> None:
    baseline = np.asarray([[1.0, 100.0], [1.0, 100.0], [10.0, 1.0], [10.0, 1.0]])
    residual = np.asarray([[0.0, 100.0], [0.0, 100.0], [10.0, 0.0], [10.0, 0.0]])
    clusters = ["a", "a", "b", "b"]
    first = site_macro_record_bootstrap(
        baseline,
        residual,
        clusters=clusters,
        seed=19,
        resamples=64,
    )
    second = site_macro_record_bootstrap(
        baseline,
        residual,
        clusters=clusters,
        seed=19,
        resamples=64,
    )

    assert first == second
    assert first["records"] == 4
    assert first["sites"] == 2
    assert first["cluster_count"] == 2
    assert first["unit"] == "record_cluster"
    assert first["site_as_sampling_unit"] is False
    assert first["metric_defined"] is True
    assert first["all_resamples_defined"] is True
    assert first["observed"]["micro"] == pytest.approx(1.0 - 220.0 / 224.0)
    assert first["observed"]["site_macro"] == pytest.approx(
        (1.0 - 20.0 / 22.0 + 1.0 - 200.0 / 202.0) / 2.0
    )
    assert set(first["ci95"]) == {
        "micro",
        "site_macro",
        "micro_explained_gain",
        "site_macro_explained_gain",
    }
    _assert_jsonable(first)


def test_site_macro_bootstrap_treats_zero_record_site_as_valid_energy() -> None:
    baseline = np.asarray([[0.0, 2.0], [3.0, 0.0], [1.0, 4.0]])
    residual = np.asarray([[0.0, 1.0], [2.0, 0.0], [0.5, 2.0]])
    result = site_macro_record_bootstrap(
        baseline,
        residual,
        seed=23,
        resamples=64,
    )

    assert result["metric_defined"] is True
    assert result["registered_active_sites"] == 2
    assert result["observed"]["site_macro"] == pytest.approx(
        ((1.0 - 2.5 / 4.0) + (1.0 - 3.0 / 6.0)) / 2.0
    )
    # Some small resamples can omit the only record supporting a site.  That
    # uncertainty is explicit and disqualifying, not an infrastructure crash.
    assert result["defined_resamples"] + result["undefined_resamples"] == 64
    _assert_jsonable(result)


def test_site_macro_bootstrap_all_zero_reference_is_defined_as_unmeasurable() -> None:
    result = site_macro_record_bootstrap(
        np.zeros((3, 2)),
        np.ones((3, 2)),
        seed=29,
        resamples=8,
    )

    assert result["metric_defined"] is False
    assert result["all_resamples_defined"] is False
    assert result["registered_active_sites"] == 0
    assert result["ci95"]["site_macro_explained_gain"] == {
        "lower": None,
        "upper": None,
    }
    _assert_jsonable(result)


def test_score_permutation_and_rademacher_null_are_deterministic() -> None:
    predictions = np.asarray([[1.0], [2.0], [3.0], [4.0]])
    targets = predictions.copy()

    def correlation(prediction: np.ndarray, target: np.ndarray) -> float:
        return float(np.sum(prediction * target))

    score_first = permutation_score(
        predictions,
        targets,
        correlation,
        seed=3,
        resamples=40,
    )
    score_second = permutation_score(
        predictions,
        targets,
        correlation,
        seed=3,
        resamples=40,
    )
    assert score_first == score_second
    assert score_first["observed"] > score_first["null_mean"]

    def resultant(values: np.ndarray) -> float:
        return float(np.sum(values[:, 0]))

    sign_first = rademacher_sign_null(
        targets,
        resultant,
        clusters=["a", "a", "b", "b"],
        seed=11,
        resamples=40,
    )
    sign_second = rademacher_sign_null(
        targets,
        resultant,
        clusters=["a", "a", "b", "b"],
        seed=11,
        resamples=40,
    )
    assert sign_first == sign_second
    assert sign_first["kind"] == "rademacher"
    assert len(sign_first["scores"]) == 40
    _assert_jsonable(score_first)
    _assert_jsonable(sign_first)


def test_covariance_spectrum_reports_effective_rank_and_is_jsonable() -> None:
    values = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    result = covariance_spectrum(values, rank=2, seed=5)

    assert result["rank"] == 2
    assert result["lambda1_fraction"] == pytest.approx(0.5)
    assert result["effective_rank"] == pytest.approx(2.0)
    assert result["explained_energy"][-1] == pytest.approx(1.0)
    _assert_jsonable(result)


def test_principal_angles_and_cross_split_alignment() -> None:
    basis_a = np.eye(3, 2)
    basis_b = np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    angles = principal_angle_summary(basis_a, basis_b)

    assert angles["rank"] == 2
    assert angles["cosines"][0] == pytest.approx(1.0)
    assert angles["cosines"][1] == pytest.approx(0.0)
    assert angles["max_angle_radians"] == pytest.approx(np.pi / 2.0)

    split_a = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    split_b = split_a * 3.0
    alignment = cross_split_alignment(split_a, split_b, rank=2, seed=13)
    assert alignment["mean_cosine_squared"] == pytest.approx(1.0)
    assert alignment["max_angle_radians"] == pytest.approx(0.0)
    _assert_jsonable(angles)
    _assert_jsonable(alignment)


def test_cpu_and_shape_validation() -> None:
    target = np.ones((3, 2))
    with pytest.raises(ValueError, match="equal shape"):
        record_cluster_metrics(target, np.ones((2, 2)))
    with pytest.raises(ValueError, match="positive"):
        record_cluster_bootstrap(target, target, seed=0, resamples=0)
    with pytest.raises(ValueError, match="finite"):
        covariance_spectrum(np.asarray([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="equal shape"):
        site_macro_record_bootstrap(np.ones((3, 2)), np.ones((3, 1)))
    with pytest.raises(ValueError, match="nonnegative"):
        site_macro_record_bootstrap(-np.ones((3, 2)), np.ones((3, 2)))
