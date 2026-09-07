from __future__ import annotations

"""Small, deterministic statistics for direction-geometry diagnostics.

The functions in this module deliberately operate on records rather than on
individual sites/tokens.  A record may contain a vector (or a tensor flattened
to a vector), while ``clusters`` lets callers bootstrap independent source
groups without pretending that repeated records are independent.

There is no screen or experiment orchestration here.  Inputs are copied to
CPU NumPy arrays, all random draws use an explicit local ``Generator``, and
public results contain only JSON-native values.
"""

from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


ArrayLike = Any
ScalarScore = Callable[[Any], Real]


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars/arrays and nested containers to JSON values."""

    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    return int(seed)


def _count(value: int, name: str = "resamples") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _array(value: ArrayLike, *, name: str, ndim: int | None = None) -> np.ndarray:
    """Convert a NumPy array or a CPU torch tensor without moving devices."""

    # Import lazily so this utility remains usable in a NumPy-only worker.
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present in the project
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        array = value.detach().numpy()
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an array-like value") from exc
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def _matrix(value: ArrayLike, *, name: str) -> np.ndarray:
    array = _array(value, name=name)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim >= 2:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [records, features]")
    return array


def _record_weights(weights: ArrayLike | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    array = _array(weights, name="weights", ndim=1)
    if len(array) != n or np.any(array <= 0.0):
        raise ValueError("weights must have one positive value per record")
    return array


def _labels(labels: Sequence[Any] | None, n: int, *, name: str) -> list[Any]:
    if labels is None:
        return list(range(n))
    values = list(labels)
    if len(values) != n:
        raise ValueError(f"{name} must have one label per record")
    try:
        for value in values:
            hash(value)
    except TypeError as exc:
        raise ValueError(f"{name} labels must be hashable scalars") from exc
    return values


def _groups(labels: Sequence[Any], n: int) -> list[np.ndarray]:
    order: dict[tuple[type[Any], Any], list[int]] = {}
    for index, label in enumerate(labels):
        try:
            key = (type(label), label)
            order.setdefault(key, []).append(index)
        except TypeError as exc:
            raise ValueError("group labels must be hashable") from exc
    if not order:
        raise ValueError("at least one record is required")
    return [np.asarray(indices, dtype=np.int64) for indices in order.values()]


def _ci(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
    return {"lower": float(lower), "upper": float(upper)}


def _score_value(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("score callback must return one finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("score callback must return one finite scalar")
    return result


def _metric_for_indices(
    target: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    clusters: list[Any],
    cluster_instances: Sequence[Any] | None = None,
) -> dict[str, Any]:
    selected_target = target[indices]
    selected_prediction = prediction[indices]
    selected_baseline = baseline[indices]
    selected_weights = weights[indices]
    losses = np.sum((selected_target - selected_prediction) ** 2, axis=1)
    baseline_losses = np.sum((selected_target - selected_baseline) ** 2, axis=1)
    weight_sum = float(np.sum(selected_weights))
    sse = float(np.sum(selected_weights * losses) / weight_sum)
    baseline_sse = float(np.sum(selected_weights * baseline_losses) / weight_sum)
    gain = 0.0 if baseline_sse == 0.0 else 1.0 - sse / baseline_sse

    selected_clusters = (
        list(cluster_instances)
        if cluster_instances is not None
        else [clusters[int(index)] for index in indices.tolist()]
    )
    if len(selected_clusters) != len(indices):
        raise ValueError("cluster_instances must match selected records")
    cluster_loss: list[float] = []
    cluster_baseline: list[float] = []
    for cluster in dict.fromkeys(selected_clusters):
        local = np.asarray(
            [position for position, label in enumerate(selected_clusters) if label == cluster],
            dtype=np.int64,
        )
        if local.size == 0:
            continue
        local_weights = selected_weights[local]
        local_weight_sum = float(local_weights.sum())
        cluster_loss.append(float(np.sum(local_weights * losses[local]) / local_weight_sum))
        cluster_baseline.append(
            float(np.sum(local_weights * baseline_losses[local]) / local_weight_sum)
        )
    cluster_sse = float(np.mean(cluster_loss))
    cluster_baseline_sse = float(np.mean(cluster_baseline))
    cluster_gain = 0.0 if cluster_baseline_sse == 0.0 else 1.0 - cluster_sse / cluster_baseline_sse
    return {
        "records": int(indices.size),
        "clusters": int(len(cluster_loss)),
        "micro_sse": sse,
        "micro_rmse": float(np.sqrt(sse)),
        "micro_baseline_sse": baseline_sse,
        "micro_explained_gain": float(gain),
        "cluster_macro_sse": cluster_sse,
        "cluster_macro_baseline_sse": cluster_baseline_sse,
        "cluster_macro_explained_gain": float(cluster_gain),
    }


def record_cluster_metrics(
    target: ArrayLike,
    prediction: ArrayLike,
    *,
    baseline: ArrayLike | None = None,
    clusters: Sequence[Any] | None = None,
    weights: ArrayLike | None = None,
) -> dict[str, Any]:
    """Return micro and independent-record-cluster squared-error metrics.

    ``baseline`` is the reference prediction used for explained gain.  When
    omitted, the weighted global target mean is used, matching the usual R²
    null.  Clusters affect only the macro summary; the micro summary remains
    record weighted.
    """

    target_array = _matrix(target, name="target")
    prediction_array = _matrix(prediction, name="prediction")
    if prediction_array.shape != target_array.shape:
        raise ValueError("target and prediction must have equal shape")
    n = target_array.shape[0]
    record_weight = _record_weights(weights, n)
    if baseline is None:
        mean = np.average(target_array, axis=0, weights=record_weight)
        baseline_array = np.broadcast_to(mean, target_array.shape).copy()
    else:
        baseline_array = _matrix(baseline, name="baseline")
        if baseline_array.shape != target_array.shape:
            raise ValueError("baseline must have equal shape to target")
    cluster_labels = _labels(clusters, n, name="clusters")
    return _metric_for_indices(
        target_array,
        prediction_array,
        baseline_array,
        record_weight,
        np.arange(n, dtype=np.int64),
        cluster_labels,
    )


def record_cluster_bootstrap(
    target: ArrayLike,
    prediction: ArrayLike,
    *,
    baseline: ArrayLike | None = None,
    clusters: Sequence[Any] | None = None,
    weights: ArrayLike | None = None,
    seed: int = 0,
    resamples: int = 1000,
) -> dict[str, Any]:
    """Bootstrap whole clusters, preserving all records within each cluster."""

    seed_value = _seed(seed)
    sample_count = _count(resamples)
    target_array = _matrix(target, name="target")
    prediction_array = _matrix(prediction, name="prediction")
    if target_array.shape != prediction_array.shape:
        raise ValueError("target and prediction must have equal shape")
    n = target_array.shape[0]
    record_weight = _record_weights(weights, n)
    if baseline is None:
        mean = np.average(target_array, axis=0, weights=record_weight)
        baseline_array = np.broadcast_to(mean, target_array.shape).copy()
    else:
        baseline_array = _matrix(baseline, name="baseline")
        if baseline_array.shape != target_array.shape:
            raise ValueError("baseline must have equal shape to target")
    cluster_labels = _labels(clusters, n, name="clusters")
    groups = _groups(cluster_labels, n)
    observed = _metric_for_indices(
        target_array,
        prediction_array,
        baseline_array,
        record_weight,
        np.arange(n, dtype=np.int64),
        cluster_labels,
    )
    generator = np.random.default_rng(seed_value)
    micro_gain = np.empty(sample_count, dtype=np.float64)
    macro_gain = np.empty(sample_count, dtype=np.float64)
    for sample in range(sample_count):
        chosen = generator.integers(0, len(groups), size=len(groups))
        selected_groups = [groups[int(index)] for index in chosen]
        indices = np.concatenate(selected_groups)
        # Give repeated bootstrap draws distinct identities.  Otherwise a
        # cluster selected twice would incorrectly be collapsed by dict.fromkeys
        # in the macro metric.
        instance_labels = [
            ("bootstrap", occurrence)
            for occurrence, group in enumerate(selected_groups)
            for _ in range(len(group))
        ]
        metrics = _metric_for_indices(
            target_array,
            prediction_array,
            baseline_array,
            record_weight,
            indices,
            cluster_labels,
            instance_labels,
        )
        micro_gain[sample] = metrics["micro_explained_gain"]
        macro_gain[sample] = metrics["cluster_macro_explained_gain"]
    return _jsonable(
        {
            "seed": seed_value,
            "resamples": sample_count,
            "cluster_count": len(groups),
            "observed": observed,
            "bootstrap_mean": {
                "micro_explained_gain": float(micro_gain.mean()),
                "cluster_macro_explained_gain": float(macro_gain.mean()),
            },
            "ci95": {
                "micro_explained_gain": _ci(micro_gain),
                "cluster_macro_explained_gain": _ci(macro_gain),
            },
        }
    )


def _site_energy_metrics(
    baseline: np.ndarray,
    residual: np.ndarray,
    indices: np.ndarray,
    *,
    registered_active_sites: np.ndarray,
    minimum_reference_energy: float,
) -> tuple[float, float, np.ndarray, bool, int]:
    """Compute micro and equal-site metrics for one record resample."""

    selected_baseline = baseline[indices]
    selected_residual = residual[indices]
    baseline_total = float(np.sum(selected_baseline))
    residual_total = float(np.sum(selected_residual))
    site_baseline = np.sum(selected_baseline, axis=0)
    site_residual = np.sum(selected_residual, axis=0)
    supported = site_baseline > minimum_reference_energy
    supported_active = supported & registered_active_sites
    supported_count = int(np.sum(supported_active))
    defined = bool(
        np.any(registered_active_sites)
        and baseline_total > minimum_reference_energy
        and np.all(supported[registered_active_sites])
    )
    if not defined:
        # Undefined bootstrap draws are recorded and disqualify the signal
        # Gate; they are not infrastructure failures and must not consume a
        # one-shot experiment after the scientific computation has finished.
        return 0.0, 0.0, np.zeros(site_baseline.shape, dtype=np.float64), False, supported_count
    micro = 1.0 - residual_total / baseline_total
    site_gain = np.zeros(site_baseline.shape, dtype=np.float64)
    site_gain[registered_active_sites] = (
        1.0
        - site_residual[registered_active_sites]
        / site_baseline[registered_active_sites]
    )
    return (
        micro,
        float(np.mean(site_gain[registered_active_sites])),
        site_gain,
        True,
        supported_count,
    )


def site_macro_record_bootstrap(
    baseline: ArrayLike,
    residual: ArrayLike,
    *,
    clusters: Sequence[Any] | None = None,
    seed: int = 0,
    resamples: int = 1000,
    minimum_reference_energy: float = 1.0e-30,
) -> dict[str, Any]:
    """Bootstrap site energy gains by resampling whole records or clusters.

    ``baseline`` and ``residual`` are ``[records, sites]`` energy matrices.
    Each replicate resamples records (or record clusters) with replacement;
    sites are never sampled.  Within a replicate, each site's gain is
    ``1 - sum(residual) / sum(baseline)`` and ``site_macro`` is the unweighted
    mean across sites.  Sites whose aggregate reference energy is at most
    ``minimum_reference_energy`` are registered as inactive.  A bootstrap draw
    that loses support for any registered active site is reported as undefined
    and makes ``all_resamples_defined`` false; it does not raise or silently
    change the registered site set.
    """

    seed_value = _seed(seed)
    sample_count = _count(resamples)
    baseline_array = _array(baseline, name="baseline", ndim=2)
    residual_array = _array(residual, name="residual", ndim=2)
    if baseline_array.shape != residual_array.shape:
        raise ValueError("baseline and residual must have equal shape")
    if baseline_array.shape[0] == 0 or baseline_array.shape[1] == 0:
        raise ValueError("baseline and residual must contain records and sites")
    if np.any(baseline_array < 0.0):
        raise ValueError("baseline must be nonnegative site energy")
    if np.any(residual_array < 0.0):
        raise ValueError("residual must be nonnegative site energy")
    if (
        isinstance(minimum_reference_energy, bool)
        or not isinstance(minimum_reference_energy, Real)
        or not np.isfinite(float(minimum_reference_energy))
        or float(minimum_reference_energy) < 0.0
    ):
        raise ValueError("minimum_reference_energy must be a finite nonnegative scalar")
    reference_floor = float(minimum_reference_energy)

    n_records, n_sites = baseline_array.shape
    cluster_labels = _labels(clusters, n_records, name="clusters")
    groups = _groups(cluster_labels, n_records)
    all_indices = np.arange(n_records, dtype=np.int64)
    registered_active_sites = (
        np.sum(baseline_array, axis=0) > reference_floor
    )
    (
        observed_micro,
        observed_macro,
        observed_sites,
        observed_defined,
        observed_supported_sites,
    ) = _site_energy_metrics(
        baseline_array,
        residual_array,
        all_indices,
        registered_active_sites=registered_active_sites,
        minimum_reference_energy=reference_floor,
    )
    generator = np.random.default_rng(seed_value)
    micro = np.empty(sample_count, dtype=np.float64)
    macro = np.empty(sample_count, dtype=np.float64)
    defined = np.zeros(sample_count, dtype=np.bool_)
    supported_sites = np.zeros(sample_count, dtype=np.int64)
    for sample in range(sample_count):
        chosen = generator.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[int(index)] for index in chosen])
        (
            micro[sample],
            macro[sample],
            _,
            defined[sample],
            supported_sites[sample],
        ) = _site_energy_metrics(
            baseline_array,
            residual_array,
            indices,
            registered_active_sites=registered_active_sites,
            minimum_reference_energy=reference_floor,
        )

    valid_micro = micro[defined]
    valid_macro = macro[defined]

    def optional_mean(values: np.ndarray) -> float | None:
        return float(values.mean()) if len(values) else None

    def optional_ci(values: np.ndarray) -> dict[str, float | None]:
        return _ci(values) if len(values) else {"lower": None, "upper": None}

    observed = {
        "micro": observed_micro,
        "site_macro": observed_macro,
        "micro_explained_gain": observed_micro,
        "site_macro_explained_gain": observed_macro,
        "site_explained_gain": [
            float(observed_sites[index]) if registered_active_sites[index] else None
            for index in range(n_sites)
        ],
    }
    bootstrap_mean = {
        "micro": optional_mean(valid_micro),
        "site_macro": optional_mean(valid_macro),
        "micro_explained_gain": optional_mean(valid_micro),
        "site_macro_explained_gain": optional_mean(valid_macro),
    }
    ci95 = {
        "micro": optional_ci(valid_micro),
        "site_macro": optional_ci(valid_macro),
        "micro_explained_gain": optional_ci(valid_micro),
        "site_macro_explained_gain": optional_ci(valid_macro),
    }
    return _jsonable(
        {
            "seed": seed_value,
            "resamples": sample_count,
            "records": int(n_records),
            "sites": int(n_sites),
            "registered_active_sites": int(np.sum(registered_active_sites)),
            "registered_inactive_sites": int(n_sites - np.sum(registered_active_sites)),
            "registered_active_site_mask": registered_active_sites,
            "minimum_reference_energy": reference_floor,
            "metric_defined": observed_defined,
            "observed_supported_active_sites": observed_supported_sites,
            "defined_resamples": int(np.sum(defined)),
            "undefined_resamples": int(sample_count - np.sum(defined)),
            "all_resamples_defined": bool(observed_defined and np.all(defined)),
            "minimum_supported_active_sites_per_resample": int(
                np.min(supported_sites)
            ),
            "undefined_resample_policy": (
                "report_and_disqualify_signal_without_crashing"
            ),
            "cluster_count": len(groups),
            "unit": "record_cluster",
            "site_as_sampling_unit": False,
            "observed": observed,
            "bootstrap_mean": bootstrap_mean,
            "ci95": ci95,
        }
    )


def _permutation_indices(
    n: int,
    *,
    seed: int,
    resamples: int,
    strata: Sequence[Any] | None,
) -> np.ndarray:
    labels = _labels(strata, n, name="strata") if strata is not None else [0] * n
    groups = _groups(labels, n)
    generator = np.random.default_rng(_seed(seed))
    result = np.empty((resamples, n), dtype=np.int64)
    for row in range(resamples):
        permutation = np.arange(n, dtype=np.int64)
        for group in groups:
            permutation[group] = generator.permutation(group)
        result[row] = permutation
    return result


def _null_summary(observed: float, values: np.ndarray, *, alternative: str) -> dict[str, Any]:
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    if alternative == "greater":
        extreme = values >= observed
        p_value = (1.0 + float(np.count_nonzero(extreme))) / (len(values) + 1.0)
    elif alternative == "less":
        extreme = values <= observed
        p_value = (1.0 + float(np.count_nonzero(extreme))) / (len(values) + 1.0)
    else:
        center = float(np.mean(values))
        extreme = np.abs(values - center) >= abs(observed - center)
        p_value = (1.0 + float(np.count_nonzero(extreme))) / (len(values) + 1.0)
    return {
        "observed": float(observed),
        "null_mean": float(values.mean()),
        "null_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "null_ci95": _ci(values),
        "p_value": float(p_value),
        "alternative": alternative,
        "extreme_count": int(np.count_nonzero(extreme)),
    }


def permutation_fit_score(
    features: ArrayLike,
    targets: ArrayLike,
    fit_fn: Callable[[np.ndarray, np.ndarray], Any],
    score_fn: Callable[[Any, np.ndarray, np.ndarray], Real],
    *,
    eval_features: ArrayLike | None = None,
    eval_targets: ArrayLike | None = None,
    strata: Sequence[Any] | None = None,
    seed: int = 0,
    resamples: int = 1000,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Fit and score a deterministic target-permutation null.

    The callback contract is intentionally small: ``fit_fn(X, y)`` returns a
    model and ``score_fn(model, X_eval, y_eval)`` returns one scalar, where
    larger is interpreted according to ``alternative``.  The heldout arrays
    are never permuted, making this the fit-null used to detect overfitting.
    """

    x = _matrix(features, name="features")
    y = _matrix(targets, name="targets")
    if len(x) != len(y):
        raise ValueError("features and targets must have equal record count")
    eval_x = x if eval_features is None else _matrix(eval_features, name="eval_features")
    eval_y = y if eval_targets is None else _matrix(eval_targets, name="eval_targets")
    if eval_x.shape[0] != eval_y.shape[0] or eval_x.shape[1] != x.shape[1] or eval_y.shape[1] != y.shape[1]:
        raise ValueError("evaluation feature/target shapes are incompatible")
    strata_values = None if strata is None else list(strata)
    if strata_values is not None and len(strata_values) != len(x):
        raise ValueError("strata must have one label per training record")
    observed_model = fit_fn(x.copy(), y.copy())
    observed = _score_value(score_fn(observed_model, eval_x.copy(), eval_y.copy()))
    permutation = _permutation_indices(
        len(y), seed=_seed(seed), resamples=_count(resamples), strata=strata_values
    )
    scores = np.empty(len(permutation), dtype=np.float64)
    for row, indices in enumerate(permutation):
        model = fit_fn(x.copy(), y[indices].copy())
        scores[row] = _score_value(score_fn(model, eval_x.copy(), eval_y.copy()))
    summary = _null_summary(observed, scores, alternative=alternative)
    return _jsonable({"seed": int(seed), "resamples": len(scores), "kind": "fit-null", "scores": scores, **summary})


def permutation_feature_fit_score(
    features: ArrayLike,
    targets: ArrayLike,
    fit_fn: Callable[[np.ndarray, np.ndarray], Any],
    score_fn: Callable[[Any, np.ndarray, np.ndarray], Real],
    *,
    eval_features: ArrayLike | None = None,
    eval_targets: ArrayLike | None = None,
    strata: Sequence[Any] | None = None,
    seed: int = 0,
    resamples: int = 1000,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Fit a feature-permutation null with targets and evaluation fixed.

    Training records of ``features`` are permuted within optional ``strata``
    while ``targets`` stay attached to their original records.  Every
    permutation performs a complete fit, then scores on the untouched
    evaluation ``X/y`` pair.  This is distinct from ``permutation_fit_score``:
    that function permutes the training targets instead.
    """

    x = _matrix(features, name="features")
    y = _matrix(targets, name="targets")
    if len(x) != len(y):
        raise ValueError("features and targets must have equal record count")
    eval_x = x if eval_features is None else _matrix(eval_features, name="eval_features")
    eval_y = y if eval_targets is None else _matrix(eval_targets, name="eval_targets")
    if (
        eval_x.shape[0] != eval_y.shape[0]
        or eval_x.shape[1] != x.shape[1]
        or eval_y.shape[1] != y.shape[1]
    ):
        raise ValueError("evaluation feature/target shapes are incompatible")
    strata_values = None if strata is None else list(strata)
    if strata_values is not None and len(strata_values) != len(x):
        raise ValueError("strata must have one label per training record")
    observed_model = fit_fn(x.copy(), y.copy())
    observed = _score_value(score_fn(observed_model, eval_x.copy(), eval_y.copy()))
    permutation = _permutation_indices(
        len(x), seed=_seed(seed), resamples=_count(resamples), strata=strata_values
    )
    scores = np.empty(len(permutation), dtype=np.float64)
    for row, indices in enumerate(permutation):
        model = fit_fn(x[indices].copy(), y.copy())
        scores[row] = _score_value(score_fn(model, eval_x.copy(), eval_y.copy()))
    summary = _null_summary(observed, scores, alternative=alternative)
    return _jsonable(
        {
            "seed": int(seed),
            "resamples": len(scores),
            "kind": "feature-fit-null",
            "scores": scores,
            **summary,
        }
    )


def permutation_score(
    predictions: ArrayLike,
    targets: ArrayLike,
    score_fn: Callable[[np.ndarray, np.ndarray], Real],
    *,
    strata: Sequence[Any] | None = None,
    seed: int = 0,
    resamples: int = 1000,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Score-null: keep predictions fixed and permute target pairing."""

    prediction_array = _matrix(predictions, name="predictions")
    target_array = _matrix(targets, name="targets")
    if prediction_array.shape != target_array.shape:
        raise ValueError("predictions and targets must have equal shape")
    observed = _score_value(score_fn(prediction_array.copy(), target_array.copy()))
    permutation = _permutation_indices(
        len(target_array), seed=_seed(seed), resamples=_count(resamples), strata=strata
    )
    scores = np.asarray(
        [
            _score_value(score_fn(prediction_array.copy(), target_array[index].copy()))
            for index in permutation
        ],
        dtype=np.float64,
    )
    summary = _null_summary(observed, scores, alternative=alternative)
    return _jsonable({"seed": int(seed), "resamples": len(scores), "kind": "score-null", "scores": scores, **summary})


def rademacher_sign_null(
    values: ArrayLike,
    score_fn: Callable[[np.ndarray], Real],
    *,
    clusters: Sequence[Any] | None = None,
    seed: int = 0,
    resamples: int = 1000,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Score a record-level sign-flip null while preserving each norm."""

    array = _matrix(values, name="values")
    labels = _labels(clusters, len(array), name="clusters")
    groups = _groups(labels, len(array))
    observed = _score_value(score_fn(array.copy()))
    generator = np.random.default_rng(_seed(seed))
    scores = np.empty(_count(resamples), dtype=np.float64)
    for sample in range(len(scores)):
        signs = np.ones(len(array), dtype=np.float64)
        for group in groups:
            signs[group] = generator.choice(np.asarray([-1.0, 1.0]), size=1)[0]
        scores[sample] = _score_value(score_fn(array * signs[:, None]))
    summary = _null_summary(observed, scores, alternative=alternative)
    return _jsonable({"seed": int(seed), "resamples": len(scores), "kind": "rademacher", "cluster_count": len(groups), "scores": scores, **summary})


def _spectrum_basis(
    values: np.ndarray,
    *,
    rank: int,
    seed: int,
    center: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x = values - values.mean(axis=0, keepdims=True) if center else values.copy()
    n, d = x.shape
    limit = min(n, d)
    if isinstance(rank, bool) or not isinstance(rank, Integral) or int(rank) < 1:
        raise ValueError("rank must be a positive integer")
    requested = min(int(rank), limit)
    generator = np.random.default_rng(_seed(seed))
    if requested == limit:
        _, singular, right = np.linalg.svd(x, full_matrices=False)
        return singular[:requested], right[:requested].T
    oversample = min(limit, max(requested + 4, requested * 2))
    omega = generator.normal(size=(d, oversample))
    sketch = x @ omega
    q, _ = np.linalg.qr(sketch, mode="reduced")
    _, singular, right = np.linalg.svd(q.T @ x, full_matrices=False)
    return singular[:requested], right[:requested].T


def covariance_spectrum(
    values: ArrayLike,
    *,
    rank: int | None = None,
    seed: int = 0,
    center: bool = True,
) -> dict[str, Any]:
    """Approximate covariance spectrum via deterministic randomized SVD."""

    matrix = _matrix(values, name="values")
    limit = min(matrix.shape)
    requested = limit if rank is None else int(rank)
    singular, basis = _spectrum_basis(matrix, rank=requested, seed=_seed(seed), center=center)
    eigenvalues = (singular**2) / max(matrix.shape[0] - 1, 1)
    positive = eigenvalues[eigenvalues > np.finfo(np.float64).eps]
    total = float(positive.sum())
    if total == 0.0:
        probabilities = np.zeros_like(positive)
        participation = 0.0
        stable = 0.0
        entropy_rank = 0.0
    else:
        probabilities = positive / total
        participation = float(total**2 / np.sum(positive**2))
        stable = float(total / np.max(positive))
        entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return _jsonable(
        {
            "seed": _seed(seed),
            "centered": bool(center),
            "records": matrix.shape[0],
            "features": matrix.shape[1],
            "rank": len(eigenvalues),
            "eigenvalues": eigenvalues,
            "explained_energy": np.cumsum(eigenvalues) / total if total else np.zeros_like(eigenvalues),
            "lambda1_fraction": float(eigenvalues[0] / total) if total else 0.0,
            "effective_rank": participation,
            "stable_rank": stable,
            "entropy_effective_rank": entropy_rank,
            "participation_ratio": participation,
            "basis_dimension": list(basis.shape),
        }
    )


def principal_angle_summary(
    basis_a: ArrayLike,
    basis_b: ArrayLike,
    *,
    rank: int | None = None,
) -> dict[str, Any]:
    """Summarize principal angles between two column-space bases."""

    a = _matrix(basis_a, name="basis_a")
    b = _matrix(basis_b, name="basis_b")
    if a.shape[0] != b.shape[0]:
        raise ValueError("bases must have equal ambient feature dimension")
    qa, _ = np.linalg.qr(a, mode="reduced")
    qb, _ = np.linalg.qr(b, mode="reduced")
    available = min(qa.shape[1], qb.shape[1])
    count = available if rank is None else min(available, _count(rank, "rank"))
    singular = np.linalg.svd(qa[:, :count].T @ qb[:, :count], compute_uv=False)[:count]
    singular = np.clip(singular, 0.0, 1.0)
    angles = np.arccos(singular)
    return _jsonable(
        {
            "ambient_dimension": a.shape[0],
            "rank": int(count),
            "cosines": singular,
            "angles_radians": angles,
            "angles_degrees": np.degrees(angles),
            "mean_cosine": float(singular.mean()) if count else 0.0,
            "mean_cosine_squared": float(np.mean(singular**2)) if count else 0.0,
            "max_angle_radians": float(angles.max()) if count else 0.0,
        }
    )


def cross_split_alignment(
    split_a: ArrayLike,
    split_b: ArrayLike,
    *,
    rank: int = 8,
    seed: int = 0,
    center: bool = True,
) -> dict[str, Any]:
    """Compare covariance subspaces learned independently on two splits."""

    a = _matrix(split_a, name="split_a")
    b = _matrix(split_b, name="split_b")
    if a.shape[1] != b.shape[1]:
        raise ValueError("splits must have equal feature dimension")
    requested = min(_count(rank, "rank"), a.shape[1], a.shape[0], b.shape[0])
    _, basis_a = _spectrum_basis(a, rank=requested, seed=_seed(seed), center=center)
    _, basis_b = _spectrum_basis(b, rank=requested, seed=_seed(seed) + 1, center=center)
    result = principal_angle_summary(basis_a, basis_b, rank=requested)
    return _jsonable(
        {
            "seed": _seed(seed),
            "centered": bool(center),
            "rank": requested,
            "split_a_records": a.shape[0],
            "split_b_records": b.shape[0],
            **result,
        }
    )


__all__ = [
    "record_cluster_metrics",
    "record_cluster_bootstrap",
    "site_macro_record_bootstrap",
    "permutation_fit_score",
    "permutation_feature_fit_score",
    "permutation_score",
    "rademacher_sign_null",
    "covariance_spectrum",
    "principal_angle_summary",
    "cross_split_alignment",
]
