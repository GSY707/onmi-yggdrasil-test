from __future__ import annotations

"""One-shot, read-only R0--R4 direction-geometry screen."""

from dataclasses import asdict
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from yggdrasil_v2.r1_revalidation.h1.train import load_deployment_checkpoint

from .capture import CapturedBatch, audit_full_replay_identity, iter_captures
from .contract import (
    AUTHORIZES,
    IDENTITY,
    OUTPUT_ROOT,
    PREFLIGHT_LEASE,
    SCHEMA_PREFIX,
    SCOPE,
    SOURCE_CHECKPOINT_SHA256,
    SPEC,
    contract_manifest,
    verify_fixed_inputs,
)
from .geometry import (
    predict_record_global,
    predict_record_route,
    record_global_centroid,
    record_route_centroids,
)
from .io import ReadOnlyInputs, load_and_validate_inputs, sha256_file
from .matrix_free import LocalJacobianProblem, solve_local_projection_jacobian
from .ridge import (
    GroupedRidgeFit,
    GroupedSufficientStatistics,
    fit_grouped_ridge,
    gram_spectrum,
    sufficient_metric,
)
from .stats import (
    covariance_spectrum,
    cross_split_alignment,
    permutation_feature_fit_score,
    permutation_fit_score,
    permutation_score,
    rademacher_sign_null,
    record_cluster_bootstrap,
    site_macro_record_bootstrap,
)


FEATURE_SURFACES: Mapping[str, Mapping[str, Any]] = {
    "projection_feature_trunk": {
        "attribute": "projection_feature",
        "width": 384,
        "evidence": "target_before_frozen_projection_input",
        "primary": True,
    },
    "attention_state": {
        "attribute": "attention_state",
        "width": 256,
        "evidence": "target_before_frozen_input_conditioned_state",
        "primary": False,
    },
    "common_private_hidden": {
        "attribute": "common_private_hidden",
        "width": 384,
        "evidence": "target_before_common_private_intermediate",
        "primary": False,
    },
    "common_output_proxy": {
        "attribute": "common_output_proxy",
        "width": 256,
        "evidence": "frozen_output_proxy_only_not_input_predictability",
        "primary": False,
    },
    "projection_output_proxy": {
        "attribute": "projection_output_proxy",
        "width": 256,
        "evidence": "near_target_frozen_output_proxy_only",
        "primary": False,
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any, *, durable: bool = False) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        if durable:
            handle.flush()
            os.fsync(handle.fileno())


def _create_jsonl(path: Path, value: Any) -> None:
    """Create a single-use ledger without an existence-check race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        handle.flush()


def _target_batch(
    dataset: Mapping[str, Any],
    field: str,
    start: int,
    stop: int,
    *,
    device: str,
) -> Tensor:
    return torch.stack(
        tuple(site[start:stop] for site in dataset[field]), dim=1
    ).to(device=device, non_blocking=True)


def _route_record(dataset: Mapping[str, Any]) -> Tensor:
    route = dataset["route_schedule"]
    flattened = route.reshape(route.shape[0], -1)
    if not bool(torch.all(flattened == flattened[:, :1])):
        raise ValueError("this screen requires the audited record-constant route bank")
    return flattened[:, 0].to(torch.long)


def _fold_masks(dataset: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    routes = _route_record(dataset).numpy()
    validation = np.zeros(len(routes), dtype=np.bool_)
    generator = np.random.default_rng(SPEC.random_seed ^ 0x52)
    for route in (0, 1):
        indices = np.flatnonzero(routes == route)
        indices = generator.permutation(indices)
        validation[indices[:: SPEC.ridge_validation_folds]] = True
    return ~validation, validation


def _balanced_sample_indices(
    dataset: Mapping[str, Any], count: int, *, seed: int
) -> np.ndarray:
    if count % 2:
        raise ValueError("balanced sample count must be even")
    routes = _route_record(dataset).numpy()
    generator = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for route in (0, 1):
        candidates = np.flatnonzero(routes == route)
        if len(candidates) < count // 2:
            raise ValueError("insufficient route support for balanced Jacobian sample")
        selected.append(generator.choice(candidates, size=count // 2, replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _route_prediction(
    centroid: Tensor, dataset: Mapping[str, Any], start: int, stop: int, device: str
) -> Tensor:
    routes = dataset["route_schedule"][start:stop]
    return predict_record_route(
        centroid,
        routes,
        site_metadata=dataset["site_metadata"],
    ).to(device=device, dtype=torch.float32, non_blocking=True)


def _global_prediction(centroid: Tensor, records: int, device: str) -> Tensor:
    return predict_record_global(centroid, records).to(
        device=device, dtype=torch.float32, non_blocking=True
    )


def _record_energy(value: Tensor) -> np.ndarray:
    return value.double().square().sum(dim=(1, 2, 3)).cpu().numpy()


def _record_site_energy(value: Tensor) -> np.ndarray:
    return value.double().square().sum(dim=(2, 3)).cpu().numpy()


def _energy_summary(
    target_energy: np.ndarray,
    residual_energy: np.ndarray,
    routes: np.ndarray,
    *,
    baseline_energy: np.ndarray | None = None,
) -> dict[str, Any]:
    denominator = target_energy if baseline_energy is None else baseline_energy
    total_denominator = max(float(denominator.sum()), 1.0e-30)
    total_residual = float(residual_energy.sum())
    ratios = residual_energy / np.maximum(denominator, 1.0e-30)
    result: dict[str, Any] = {
        "records": int(len(target_energy)),
        "sse": total_residual,
        "reference_energy": float(denominator.sum()),
        "increment_nmse": total_residual / total_denominator,
        "explained_energy_gain": 1.0 - total_residual / total_denominator,
        "record_macro_nmse": float(ratios.mean()),
        "record_macro_nmse_median": float(np.median(ratios)),
        "record_macro_nmse_p95": float(np.quantile(ratios, 0.95)),
        "by_route": {},
    }
    for route in (0, 1):
        selected = routes == route
        route_denominator = max(float(denominator[selected].sum()), 1.0e-30)
        route_residual = float(residual_energy[selected].sum())
        result["by_route"][str(route)] = {
            "records": int(selected.sum()),
            "increment_nmse": route_residual / route_denominator,
            "explained_energy_gain": 1.0 - route_residual / route_denominator,
        }
    return result


def _site_macro_summary(
    target_by_site: np.ndarray,
    residual_by_site: np.ndarray,
    *,
    baseline_by_site: np.ndarray | None = None,
) -> dict[str, Any]:
    denominator = target_by_site if baseline_by_site is None else baseline_by_site
    denominator_sum = denominator.sum(axis=0)
    residual_sum = residual_by_site.sum(axis=0)
    active = denominator_sum > 1.0e-30
    gains = np.full(denominator_sum.shape, np.nan, dtype=np.float64)
    gains[active] = 1.0 - residual_sum[active] / denominator_sum[active]
    return {
        "active_sites": int(active.sum()),
        "sites": int(len(active)),
        "site_explained_gain": [
            None if not np.isfinite(value) else float(value) for value in gains
        ],
        "site_macro_explained_gain": float(np.mean(gains[active]))
        if bool(active.any())
        else 0.0,
        "site_as_sampling_unit": False,
    }


def _loss_gain_bootstrap(
    residual: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    count = len(residual)
    observed = 1.0 - float(residual.sum()) / max(float(baseline.sum()), 1.0e-30)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = generator.integers(0, count, size=count)
        values[index] = 1.0 - float(residual[sample].sum()) / max(
            float(baseline[sample].sum()), 1.0e-30
        )
    return {
        "seed": seed,
        "resamples": resamples,
        "observed": observed,
        "mean": float(values.mean()),
        "ci95": {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        },
    }


def _new_feature_statistics() -> dict[str, dict[str, GroupedSufficientStatistics]]:
    return {
        name: {
            part: GroupedSufficientStatistics(
                32, int(spec["width"]), 256, device=SPEC.device
            )
            for part in ("fit", "validation", "heldout")
        }
        for name, spec in FEATURE_SURFACES.items()
    }


def _update_site_group_statistics(
    stats: GroupedSufficientStatistics,
    feature: Tensor,
    target: Tensor,
    routes: Tensor,
    site_index: int,
    record_mask: Tensor,
) -> None:
    for route in (0, 1):
        selected = record_mask & (routes == route)
        if bool(selected.any()):
            stats.update(
                site_index * 2 + route,
                feature[selected].reshape(-1, feature.shape[-1]),
                target[selected].reshape(-1, target.shape[-1]),
            )


def _predict_sites(
    fit: GroupedRidgeFit,
    features: Tensor,
    routes: Tensor,
    *,
    mode: str,
) -> Tensor:
    if features.ndim != 4 or features.shape[1] != 16:
        raise ValueError("stored feature tensor must have shape [N,16,K,H]")
    outputs: list[Tensor] = []
    for site_index in range(16):
        layer = site_index % 2
        group_base = site_index * 2 if mode == "site_route" else layer * 2
        groups = (
            routes[:, None].expand(-1, features.shape[2]).reshape(-1) + group_base
        )
        prediction = fit.predict_rows(
            features[:, site_index].reshape(-1, features.shape[-1]), groups
        ).reshape(features.shape[0], features.shape[2], fit.output_width)
        outputs.append(prediction)
    return torch.stack(outputs, dim=1)


def _sample_positions(indices: np.ndarray, count: int) -> np.ndarray:
    positions = np.full(count, -1, dtype=np.int64)
    positions[indices] = np.arange(len(indices), dtype=np.int64)
    return positions


def _capture_r2(
    model: Any,
    inputs: ReadOnlyInputs,
    centroids: Mapping[str, Tensor],
    progress: Callable[[str, Mapping[str, Any]], None],
) -> tuple[
    dict[str, dict[str, GroupedSufficientStatistics]],
    dict[str, Tensor],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    datasets = inputs.target_payload["datasets"]
    statistics = _new_feature_statistics()
    stored_primary: dict[str, Tensor] = {}
    energy: dict[str, dict[str, np.ndarray]] = {}
    replay_audit = {
        "maximum_common_absolute_error": 0.0,
        "maximum_projection_absolute_error": 0.0,
        "microbatch_size": SPEC.replay_microbatch_size,
        "analysis_batch_size": SPEC.analysis_batch_size,
    }
    sample_indices = {
        "train": _balanced_sample_indices(
            datasets["train"], SPEC.local_full_j_train_records, seed=SPEC.random_seed ^ 0x4A
        ),
        "heldout": _balanced_sample_indices(
            datasets["heldout"], SPEC.local_full_j_heldout_records, seed=SPEC.random_seed ^ 0x4B
        ),
    }
    sampled_states: dict[str, Tensor] = {}
    for split in ("train", "heldout"):
        dataset = datasets[split]
        count = len(dataset["record_ids"])
        stored_primary[split] = torch.empty(
            count, 16, 8, 384, dtype=torch.float32
        )
        target_before_store = {
            "projection_feature_trunk": stored_primary[split],
            "attention_state": torch.empty(count, 16, 8, 256, dtype=torch.float32),
            "common_private_hidden": torch.empty(
                count, 16, 8, 384, dtype=torch.float32
            ),
        }
        sample_count = len(sample_indices[split])
        sampled_states[split] = torch.empty(
            sample_count, 2, 8, 8, 256, dtype=torch.float32
        )
        energy[split] = {
            name: np.empty(count, dtype=np.float64)
            for name in ("target", "after_r0", "after_r1")
        }
        energy[split].update(
            {
                name: np.empty((count, 16), dtype=np.float64)
                for name in ("target_by_site", "after_r0_by_site", "after_r1_by_site")
            }
        )
        if split == "train":
            fit_mask_np, validation_mask_np = _fold_masks(dataset)
        else:
            fit_mask_np = validation_mask_np = np.zeros(count, dtype=np.bool_)
        completed = 0
        for batch in iter_captures(
            model,
            inputs.cache,
            dataset,
            device=SPEC.device,
            split=split,
            batch_size=SPEC.replay_microbatch_size,
        ):
            start, stop = batch.start, batch.stop
            for site_index, site in enumerate(batch.sites):
                target_before_store["projection_feature_trunk"][start:stop, site_index].copy_(
                    site.projection_feature.detach().cpu().to(torch.float32)
                )
                target_before_store["attention_state"][start:stop, site_index].copy_(
                    site.attention_state.detach().cpu().to(torch.float32)
                )
                target_before_store["common_private_hidden"][start:stop, site_index].copy_(
                    site.common_private_hidden.detach().cpu().to(torch.float32)
                )
            replay_audit["maximum_common_absolute_error"] = max(
                replay_audit["maximum_common_absolute_error"],
                batch.maximum_common_replay_error,
            )
            replay_audit["maximum_projection_absolute_error"] = max(
                replay_audit["maximum_projection_absolute_error"],
                batch.maximum_projection_replay_error,
            )
            if not batch.target_before_features_finite:
                raise RuntimeError(f"{split} target-before capture contains non-finite values")
            completed = stop
            if completed % SPEC.analysis_batch_size == 0 or completed == count:
                progress(
                    "r2_capture",
                    {"split": split, "records_completed": completed, "records": count},
                )

        selected_records = torch.as_tensor(sample_indices[split], dtype=torch.long)
        selected_attention = target_before_store["attention_state"].index_select(
            0, selected_records
        )
        for site_index, metadata in enumerate(dataset["site_metadata"]):
            sampled_states[split][
                :, int(metadata["layer_index"]), int(metadata["step_index"])
            ].copy_(selected_attention[:, site_index])
        del selected_attention

        routes_all = _route_record(dataset)
        for start in range(0, count, SPEC.analysis_batch_size):
            stop = min(start + SPEC.analysis_batch_size, count)
            batch_count = stop - start
            target = _target_batch(dataset, "transfer", start, stop, device=SPEC.device)
            global_prediction = _global_prediction(
                centroids["global"], batch_count, SPEC.device
            )
            route_prediction = _route_prediction(
                centroids["route"], dataset, start, stop, SPEC.device
            )
            residual_r0 = target - global_prediction
            residual_r1 = target - route_prediction
            energy[split]["target"][start:stop] = _record_energy(target)
            energy[split]["after_r0"][start:stop] = _record_energy(residual_r0)
            energy[split]["after_r1"][start:stop] = _record_energy(residual_r1)
            energy[split]["target_by_site"][start:stop] = _record_site_energy(target)
            energy[split]["after_r0_by_site"][start:stop] = _record_site_energy(
                residual_r0
            )
            energy[split]["after_r1_by_site"][start:stop] = _record_site_energy(
                residual_r1
            )
            routes = routes_all[start:stop].to(SPEC.device)
            fit_mask = torch.as_tensor(
                fit_mask_np[start:stop], dtype=torch.bool, device=SPEC.device
            )
            validation_mask = torch.as_tensor(
                validation_mask_np[start:stop], dtype=torch.bool, device=SPEC.device
            )
            heldout_mask = torch.ones(batch_count, dtype=torch.bool, device=SPEC.device)
            feature_batches = {
                name: value[start:stop].to(SPEC.device, non_blocking=True)
                for name, value in target_before_store.items()
            }
            feature_batches["common_output_proxy"] = _target_batch(
                dataset, "common", start, stop, device=SPEC.device
            )
            feature_batches["projection_output_proxy"] = _target_batch(
                dataset, "projection", start, stop, device=SPEC.device
            )
            for name, feature in feature_batches.items():
                for site_index in range(16):
                    site_feature = feature[:, site_index]
                    if split == "train":
                        _update_site_group_statistics(
                            statistics[name]["fit"],
                            site_feature,
                            residual_r1[:, site_index],
                            routes,
                            site_index,
                            fit_mask,
                        )
                        _update_site_group_statistics(
                            statistics[name]["validation"],
                            site_feature,
                            residual_r1[:, site_index],
                            routes,
                            site_index,
                            validation_mask,
                        )
                    else:
                        _update_site_group_statistics(
                            statistics[name]["heldout"],
                            site_feature,
                            residual_r1[:, site_index],
                            routes,
                            site_index,
                            heldout_mask,
                        )
            progress(
                "r2_statistics",
                {"split": split, "records_completed": stop, "records": count},
            )
            del (
                target,
                global_prediction,
                route_prediction,
                residual_r0,
                residual_r1,
                feature_batches,
            )
        del target_before_store["attention_state"]
        del target_before_store["common_private_hidden"]
    replay_audit["tolerance"] = SPEC.maximum_replay_absolute_error
    replay_audit["passed"] = bool(
        replay_audit["maximum_common_absolute_error"] <= replay_audit["tolerance"]
        and replay_audit["maximum_projection_absolute_error"] <= replay_audit["tolerance"]
    )
    if not replay_audit["passed"]:
        raise RuntimeError(f"target-before replay drift: {replay_audit}")
    replay_audit["sample_indices"] = {
        split: sample_indices[split].tolist() for split in sample_indices
    }
    replay_audit["sampled_states"] = sampled_states
    return statistics, stored_primary, energy, replay_audit


def _fit_r2(
    statistics: Mapping[str, Mapping[str, GroupedSufficientStatistics]],
) -> tuple[dict[str, GroupedRidgeFit], dict[str, Any]]:
    fits: dict[str, GroupedRidgeFit] = {}
    report: dict[str, Any] = {
        "primary": SPEC.primary_r2_feature,
        "target": "R1_residual_transfer",
        "target_derived_features_forbidden": True,
        "family_route_or_answer_target_used": False,
        "features": {},
    }
    keys = tuple(f"site{site}_route{route}" for site in range(16) for route in (0, 1))
    for name, parts in statistics.items():
        full = parts["fit"].merge(parts["validation"])
        fit = fit_grouped_ridge(
            parts["fit"],
            parts["validation"],
            full_stats=full,
            lambda_grid=SPEC.ridge_lambda_grid,
            group_keys=keys,
        )
        fits[name] = fit
        train_metric = sufficient_metric(full, fit.coefficients)
        heldout_metric = sufficient_metric(parts["heldout"], fit.coefficients)
        lambda_metrics: dict[str, Any] = {}
        positive = 0
        for ridge_lambda in SPEC.ridge_lambda_grid:
            candidate = fit_grouped_ridge(
                parts["fit"],
                parts["validation"],
                full_stats=full,
                lambda_grid=(ridge_lambda,),
                group_keys=keys,
            )
            candidate_heldout = sufficient_metric(
                parts["heldout"], candidate.coefficients
            )
            lambda_metrics[str(ridge_lambda)] = candidate_heldout
            positive += int(candidate_heldout["explained_energy_gain"] > 0.0)
        positive_fraction = positive / len(SPEC.ridge_lambda_grid)
        report["features"][name] = {
            "evidence_label": FEATURE_SURFACES[name]["evidence"],
            "primary": bool(FEATURE_SURFACES[name]["primary"]),
            "feature_width": int(FEATURE_SURFACES[name]["width"]),
            "predictor": "site_route_affine_ridge_train_only",
            "ridge_lambda": fit.ridge_lambda,
            "lambda_validation_nmse": {
                str(key): value for key, value in fit.lambda_validation_nmse.items()
            },
            "train": train_metric,
            "heldout": heldout_metric,
            "heldout_train_nmse_ratio": heldout_metric["increment_nmse"]
            / max(train_metric["increment_nmse"], 1.0e-20),
            "regularization_stability": {
                "heldout_by_lambda": lambda_metrics,
                "positive_gain_fraction": positive_fraction,
                "meets_registered_fraction": bool(
                    positive_fraction >= SPEC.minimum_regularization_positive_fraction
                ),
            },
        }
    return fits, report


def _new_r3_statistics() -> dict[str, GroupedSufficientStatistics]:
    return {
        part: GroupedSufficientStatistics(4, 384, 256, device=SPEC.device)
        for part in ("fit", "validation", "heldout")
    }


def _update_layer_group_statistics(
    stats: GroupedSufficientStatistics,
    features: Tensor,
    target: Tensor,
    routes: Tensor,
    site_index: int,
    record_mask: Tensor,
) -> None:
    layer = site_index % 2
    for route in (0, 1):
        selected = record_mask & (routes == route)
        if bool(selected.any()):
            stats.update(
                layer * 2 + route,
                features[selected, site_index].reshape(-1, features.shape[-1]),
                target[selected, site_index].reshape(-1, target.shape[-1]),
            )


def _accumulate_r3(
    inputs: ReadOnlyInputs,
    centroids: Mapping[str, Tensor],
    primary_features: Mapping[str, Tensor],
    r2_fit: GroupedRidgeFit,
    replay_audit: Mapping[str, Any],
    energy: dict[str, dict[str, np.ndarray]],
    progress: Callable[[str, Mapping[str, Any]], None],
) -> tuple[
    dict[str, GroupedSufficientStatistics],
    dict[str, Tensor],
]:
    datasets = inputs.target_payload["datasets"]
    statistics = _new_r3_statistics()
    sample_indices = {
        split: np.asarray(replay_audit["sample_indices"][split], dtype=np.int64)
        for split in ("train", "heldout")
    }
    sampled_targets = {
        split: torch.empty(
            len(sample_indices[split]), 2, 8, 8, 256, dtype=torch.float32
        )
        for split in ("train", "heldout")
    }
    for split in ("train", "heldout"):
        dataset = datasets[split]
        count = len(dataset["record_ids"])
        energy[split]["after_r2"] = np.empty(count, dtype=np.float64)
        energy[split]["after_r2_by_site"] = np.empty(
            (count, 16), dtype=np.float64
        )
        sample_position = _sample_positions(sample_indices[split], count)
        if split == "train":
            fit_mask_np, validation_mask_np = _fold_masks(dataset)
        else:
            fit_mask_np = validation_mask_np = np.zeros(count, dtype=np.bool_)
        routes_all = _route_record(dataset)
        for start in range(0, count, SPEC.analysis_batch_size):
            stop = min(start + SPEC.analysis_batch_size, count)
            features = primary_features[split][start:stop].to(
                device=SPEC.device, dtype=torch.float32, non_blocking=True
            )
            target = _target_batch(dataset, "transfer", start, stop, device=SPEC.device)
            route_prediction = _route_prediction(
                centroids["route"], dataset, start, stop, SPEC.device
            )
            residual_r1 = target - route_prediction
            routes = routes_all[start:stop].to(SPEC.device)
            prediction_r2 = _predict_sites(
                r2_fit, features, routes, mode="site_route"
            )
            residual_r2 = residual_r1 - prediction_r2
            energy[split]["after_r2"][start:stop] = _record_energy(residual_r2)
            energy[split]["after_r2_by_site"][start:stop] = _record_site_energy(
                residual_r2
            )
            if split == "train":
                fit_mask = torch.as_tensor(
                    fit_mask_np[start:stop], dtype=torch.bool, device=SPEC.device
                )
                validation_mask = torch.as_tensor(
                    validation_mask_np[start:stop], dtype=torch.bool, device=SPEC.device
                )
                for site_index in range(16):
                    _update_layer_group_statistics(
                        statistics["fit"],
                        features,
                        residual_r2,
                        routes,
                        site_index,
                        fit_mask,
                    )
                    _update_layer_group_statistics(
                        statistics["validation"],
                        features,
                        residual_r2,
                        routes,
                        site_index,
                        validation_mask,
                    )
            else:
                heldout_mask = torch.ones(
                    stop - start, dtype=torch.bool, device=SPEC.device
                )
                for site_index in range(16):
                    _update_layer_group_statistics(
                        statistics["heldout"],
                        features,
                        residual_r2,
                        routes,
                        site_index,
                        heldout_mask,
                    )
            local_positions = sample_position[start:stop]
            selected_local = np.flatnonzero(local_positions >= 0)
            if selected_local.size:
                destination = torch.as_tensor(
                    local_positions[selected_local], dtype=torch.long
                )
                source = torch.as_tensor(
                    selected_local, dtype=torch.long, device=SPEC.device
                )
                selected_target = residual_r2.index_select(0, source).detach().cpu()
                for site_index, metadata in enumerate(dataset["site_metadata"]):
                    sampled_targets[split][
                        destination,
                        int(metadata["layer_index"]),
                        int(metadata["step_index"]),
                    ] = selected_target[:, site_index]
            progress(
                "r3_accumulate",
                {"split": split, "records_completed": stop, "records": count},
            )
            del features, target, route_prediction, residual_r1, prediction_r2, residual_r2
    return statistics, sampled_targets


def _fit_r3_exact(
    statistics: Mapping[str, GroupedSufficientStatistics],
) -> tuple[GroupedRidgeFit, dict[str, Any]]:
    full = statistics["fit"].merge(statistics["validation"])
    fit = fit_grouped_ridge(
        statistics["fit"],
        statistics["validation"],
        full_stats=full,
        lambda_grid=SPEC.ridge_lambda_grid,
        group_keys=("layer0_route0", "layer0_route1", "layer1_route0", "layer1_route1"),
    )
    train_metric = sufficient_metric(full, fit.coefficients)
    heldout_metric = sufficient_metric(statistics["heldout"], fit.coefficients)
    lambda_metrics: dict[str, Any] = {}
    positive = 0
    for ridge_lambda in SPEC.ridge_lambda_grid:
        candidate = fit_grouped_ridge(
            statistics["fit"],
            statistics["validation"],
            full_stats=full,
            lambda_grid=(ridge_lambda,),
            group_keys=("layer0_route0", "layer0_route1", "layer1_route0", "layer1_route1"),
        )
        candidate_heldout = sufficient_metric(
            statistics["heldout"], candidate.coefficients
        )
        lambda_metrics[str(ridge_lambda)] = candidate_heldout
        positive += int(candidate_heldout["explained_energy_gain"] > 0.0)
    positive_fraction = positive / len(SPEC.ridge_lambda_grid)
    report = {
        "parameter_block": "selected_final_routed_heads_only",
        "shared_across_all_steps_and_slots": True,
        "analytic_parameter_jacobian": True,
        "dense_jacobian_materialized": False,
        "ridge_lambda": fit.ridge_lambda,
        "lambda_validation_nmse": {
            str(key): value for key, value in fit.lambda_validation_nmse.items()
        },
        "train": train_metric,
        "heldout": heldout_metric,
        "heldout_train_nmse_ratio": heldout_metric["increment_nmse"]
        / max(train_metric["increment_nmse"], 1.0e-20),
        "fisher_gram": gram_spectrum(full),
        "regularization_stability": {
            "heldout_by_lambda": lambda_metrics,
            "positive_gain_fraction": positive_fraction,
            "meets_registered_fraction": bool(
                positive_fraction >= SPEC.minimum_regularization_positive_fraction
            ),
        },
    }
    return fit, report


def _run_sampled_full_j(
    model: Any,
    inputs: ReadOnlyInputs,
    sampled_states: Mapping[str, Tensor],
    sampled_targets: Mapping[str, Tensor],
    replay_audit: Mapping[str, Any],
    progress: Callable[[str, Mapping[str, Any]], None],
) -> dict[str, Any]:
    datasets = inputs.target_payload["datasets"]
    report: dict[str, Any] = {
        "primary_parameter_block": "routed_feature_trunk_plus_selected_final_heads",
        "sampled_sensitivity": True,
        "sample_selection": "fixed_seed_balanced_by_predecessor_route_only",
        "target_derived_sample_selection": False,
        "layers": {},
    }
    for layer_index in range(2):
        train_indices = torch.as_tensor(
            replay_audit["sample_indices"]["train"], dtype=torch.long
        )
        heldout_indices = torch.as_tensor(
            replay_audit["sample_indices"]["heldout"], dtype=torch.long
        )
        train_routes = _route_record(datasets["train"]).index_select(
            0, train_indices
        ).to(SPEC.device)
        heldout_routes = _route_record(datasets["heldout"]).index_select(
            0, heldout_indices
        ).to(SPEC.device)
        train_problem = LocalJacobianProblem(
            states=sampled_states["train"][:, layer_index].to(
                SPEC.device, dtype=torch.float32
            ),
            routes=train_routes,
            targets=sampled_targets["train"][:, layer_index].to(SPEC.device),
        )
        heldout_problem = LocalJacobianProblem(
            states=sampled_states["heldout"][:, layer_index].to(
                SPEC.device, dtype=torch.float32
            ),
            routes=heldout_routes,
            targets=sampled_targets["heldout"][:, layer_index].to(SPEC.device),
        )
        progress("r3_matrix_free", {"layer": layer_index, "status": "starting"})
        layer_result = solve_local_projection_jacobian(
            model.layers[layer_index],
            train_problem,
            heldout_problem,
            damping=SPEC.local_full_j_damping,
            cg_iterations=SPEC.local_full_j_cg_iterations,
            fisher_probes=SPEC.local_full_j_fisher_probes,
            seed=SPEC.random_seed ^ (0xF0 + layer_index),
        )
        layer_result.pop("_tangent", None)
        layer_result.pop("_parameter_names", None)
        report["layers"][str(layer_index)] = layer_result
        progress("r3_matrix_free", {"layer": layer_index, "status": "complete"})
        del train_problem, heldout_problem
        torch.cuda.empty_cache()
    held_target = sum(
        float(value["heldout"]["target_energy"])
        for value in report["layers"].values()
    )
    held_sse = sum(
        float(value["heldout"]["sse"]) for value in report["layers"].values()
    )
    train_target = sum(
        float(value["train"]["target_energy"])
        for value in report["layers"].values()
    )
    train_sse = sum(
        float(value["train"]["sse"]) for value in report["layers"].values()
    )
    report["aggregate"] = {
        "train_increment_nmse": train_sse / max(train_target, 1.0e-30),
        "heldout_increment_nmse": held_sse / max(held_target, 1.0e-30),
        "heldout_train_nmse_ratio": (
            held_sse / max(held_target, 1.0e-30)
        )
        / max(train_sse / max(train_target, 1.0e-30), 1.0e-30),
        "heldout_explained_energy_gain": 1.0
        - held_sse / max(held_target, 1.0e-30),
    }
    return report


def _random_projection(
    input_width: int, output_width: int, *, seed: int
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(
        0,
        2,
        (16, input_width, output_width),
        generator=generator,
        dtype=torch.int8,
    ).float()
    return (values.mul_(2).sub_(1)) / math.sqrt(output_width)


def _sketch_sites(value: Tensor, projection: Tensor) -> np.ndarray:
    if value.ndim != 4 or value.shape[1] != 16:
        raise ValueError("site sketch input must have shape [N,16,K,D]")
    flattened = value.float().reshape(value.shape[0], 16, -1)
    result = torch.einsum(
        "nsi,sij->nsj", flattened, projection.to(value.device)
    )
    return result.reshape(value.shape[0], -1).detach().cpu().numpy().astype(np.float64)


def _feature_compact(value: Tensor, projection: Tensor) -> np.ndarray:
    pooled = value.float().mean(dim=2)
    result = torch.einsum("nsh,shj->nsj", pooled, projection.to(value.device))
    return result.reshape(value.shape[0], -1).detach().cpu().numpy().astype(np.float64)


def _functional_energy(value: Tensor, vjp_value: Tensor) -> np.ndarray:
    functional = (value.double() * vjp_value.double()).sum(dim=(2, 3))
    return functional.square().sum(dim=1).cpu().numpy()


def _route_permutation_null(
    alternative_losses: np.ndarray,
    actual_routes: np.ndarray,
    baseline_loss: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if alternative_losses.shape != (len(actual_routes), 2):
        raise ValueError("route alternative loss shape mismatch")
    rows = np.arange(len(actual_routes))
    denominator = max(float(baseline_loss.sum()), 1.0e-30)
    observed = 1.0 - float(alternative_losses[rows, actual_routes].sum()) / denominator
    generator = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        permuted = generator.permutation(actual_routes)
        values[index] = 1.0 - float(alternative_losses[rows, permuted].sum()) / denominator
    extreme = int(np.count_nonzero(values >= observed))
    return {
        "kind": "route_id_permutation_preserving_counts",
        "seed": seed,
        "resamples": resamples,
        "observed_incremental_gain_over_r0": observed,
        "null_mean": float(values.mean()),
        "null_ci95": {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        },
        "p_value": (extreme + 1.0) / (resamples + 1.0),
        "extreme_count": extreme,
        "scores": values.tolist(),
    }


def _score_gain(prediction: np.ndarray, target: np.ndarray) -> float:
    return 1.0 - float(np.sum((prediction - target) ** 2)) / max(
        float(np.sum(target**2)), 1.0e-30
    )


def _ridge_fit_compact(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    x = np.column_stack((features, np.ones(len(features), dtype=np.float64)))
    gram = x.T @ x
    scale = np.trace(gram[:-1, :-1]) / max(features.shape[1], 1)
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + 1.0e-2 * max(scale, 1.0e-20) * penalty, x.T @ targets)


def _ridge_score_compact(
    coefficients: np.ndarray, features: np.ndarray, targets: np.ndarray
) -> float:
    x = np.column_stack((features, np.ones(len(features), dtype=np.float64)))
    return _score_gain(x @ coefficients, targets)


def _negative_target_sign_symmetry(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    heldout_features: np.ndarray,
    heldout_targets: np.ndarray,
) -> dict[str, Any]:
    """Audit the exact sign symmetry of linear direction geometry."""

    positive = _ridge_fit_compact(train_features, train_targets)
    negative = _ridge_fit_compact(train_features, -train_targets)
    positive_gain = _ridge_score_compact(
        positive, heldout_features, heldout_targets
    )
    negative_gain = _ridge_score_compact(
        negative, heldout_features, -heldout_targets
    )
    coefficient_error = float(np.max(np.abs(positive + negative)))
    gain_error = abs(positive_gain - negative_gain)
    maximum_error = max(coefficient_error, gain_error)
    return {
        "kind": "explicit_negative_vjp_sign_reversal_control",
        "positive_heldout_gain": positive_gain,
        "negative_heldout_gain": negative_gain,
        "coefficient_sign_reversal_max_error": coefficient_error,
        "heldout_gain_absolute_difference": gain_error,
        "maximum_symmetry_error": maximum_error,
        "tolerance": SPEC.maximum_negative_sign_symmetry_error,
        "passed": bool(maximum_error <= SPEC.maximum_negative_sign_symmetry_error),
        "interpretation": (
            "linear reachability geometry is centrally symmetric; this control cannot "
            "establish answer-direction causal value"
        ),
    }


def _resultant(values: np.ndarray) -> float:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = values / np.maximum(norms, 1.0e-30)
    return float(np.sum(np.sum(unit, axis=0) ** 2) / max(np.sum(unit**2), 1.0e-30))


def _top_covariance_fraction(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(max(eigenvalues[-1], 0.0) / max(float(np.maximum(eigenvalues, 0.0).sum()), 1.0e-30))


def _matched_spectrum_null(
    values: np.ndarray, *, seed: int, resamples: int
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    centered = values - values.mean(axis=0, keepdims=True)
    row_norms = np.linalg.norm(centered, axis=1)
    column_scale = centered.std(axis=0, ddof=1)
    column_scale = np.maximum(column_scale, np.mean(column_scale) * 1.0e-3 + 1.0e-12)
    observed = _top_covariance_fraction(values)
    scores = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = generator.normal(size=values.shape) * column_scale[None, :]
        sample_norm = np.linalg.norm(sample, axis=1)
        sample *= (row_norms / np.maximum(sample_norm, 1.0e-30))[:, None]
        scores[index] = _top_covariance_fraction(sample)
    extreme = int(np.count_nonzero(scores >= observed))
    return {
        "kind": "matched_norm_diagonal_covariance_gaussian",
        "seed": seed,
        "resamples": resamples,
        "observed_lambda1_fraction": observed,
        "null_mean": float(scores.mean()),
        "null_ci95": {
            "lower": float(np.quantile(scores, 0.025)),
            "upper": float(np.quantile(scores, 0.975)),
        },
        "p_value": (extreme + 1.0) / (resamples + 1.0),
        "scores": scores.tolist(),
    }


def _alignment_null(
    train: np.ndarray,
    heldout: np.ndarray,
    *,
    seed: int,
    resamples: int,
    rank: int,
) -> dict[str, Any]:
    observed = cross_split_alignment(
        train, heldout, rank=rank, seed=seed
    )["mean_cosine_squared"]
    generator = np.random.default_rng(seed)
    pooled_scale = np.vstack((train, heldout)).std(axis=0, ddof=1)
    pooled_scale = np.maximum(pooled_scale, pooled_scale.mean() * 1.0e-3 + 1.0e-12)
    scores = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        left = generator.normal(size=train.shape) * pooled_scale[None, :]
        right = generator.normal(size=heldout.shape) * pooled_scale[None, :]
        scores[index] = cross_split_alignment(
            left, right, rank=rank, seed=seed + index + 1
        )["mean_cosine_squared"]
    extreme = int(np.count_nonzero(scores >= observed))
    return {
        "kind": "independent_matched_diagonal_covariance_subspaces",
        "observed_mean_cosine_squared": float(observed),
        "resamples": resamples,
        "null_mean": float(scores.mean()),
        "null_ci95": {
            "lower": float(np.quantile(scores, 0.025)),
            "upper": float(np.quantile(scores, 0.975)),
        },
        "p_value": (extreme + 1.0) / (resamples + 1.0),
        "scores": scores.tolist(),
    }


def _collect_final_geometry(
    inputs: ReadOnlyInputs,
    centroids: Mapping[str, Tensor],
    primary_features: Mapping[str, Tensor],
    r2_fit: GroupedRidgeFit,
    r3_fit: GroupedRidgeFit,
    energy: dict[str, dict[str, np.ndarray]],
    progress: Callable[[str, Mapping[str, Any]], None],
) -> dict[str, Any]:
    datasets = inputs.target_payload["datasets"]
    target_projection = _random_projection(
        8 * 256, SPEC.residual_sketch_per_site, seed=SPEC.random_seed ^ 0x53
    )
    target_projection_secondary = _random_projection(
        8 * 256, SPEC.residual_sketch_per_site, seed=SPEC.random_seed ^ 0x153
    )
    feature_projection = _random_projection(384, 4, seed=SPEC.random_seed ^ 0x54)
    feature_projection_secondary = _random_projection(
        384, 4, seed=SPEC.random_seed ^ 0x154
    )
    collected: dict[str, Any] = {}
    for split in ("train", "heldout"):
        dataset = datasets[split]
        count = len(dataset["record_ids"])
        routes_all = _route_record(dataset)
        energy[split]["after_r3"] = np.empty(count, dtype=np.float64)
        energy[split]["after_r3_by_site"] = np.empty(
            (count, 16), dtype=np.float64
        )
        functional = {
            name: np.empty(count, dtype=np.float64)
            for name in ("r2_target", "r2_residual", "r3_target", "r3_residual")
        }
        sketches = {
            name: np.empty(
                (count, 16 * SPEC.residual_sketch_per_site), dtype=np.float64
            )
            for name in ("target", "r1_residual", "r2_prediction", "r2_residual", "r3_prediction", "r3_residual")
        }
        sketches_secondary = {
            name: np.empty(
                (count, 16 * SPEC.residual_sketch_per_site), dtype=np.float64
            )
            for name in (
                "target",
                "r1_residual",
                "r2_prediction",
                "r2_residual",
                "r3_prediction",
                "r3_residual",
            )
        }
        feature_compact = np.empty((count, 16 * 4), dtype=np.float64)
        feature_compact_secondary = np.empty((count, 16 * 4), dtype=np.float64)
        route_alternative_loss = np.empty((count, 2), dtype=np.float64)
        for start in range(0, count, SPEC.analysis_batch_size):
            stop = min(start + SPEC.analysis_batch_size, count)
            features = primary_features[split][start:stop].to(
                SPEC.device, dtype=torch.float32
            )
            target = _target_batch(dataset, "transfer", start, stop, device=SPEC.device)
            route_prediction = _route_prediction(
                centroids["route"], dataset, start, stop, SPEC.device
            )
            residual_r1 = target - route_prediction
            routes = routes_all[start:stop].to(SPEC.device)
            prediction_r2 = _predict_sites(r2_fit, features, routes, mode="site_route")
            residual_r2 = residual_r1 - prediction_r2
            prediction_r3 = _predict_sites(r3_fit, features, routes, mode="layer_route")
            residual_r3 = residual_r2 - prediction_r3
            energy[split]["after_r3"][start:stop] = _record_energy(residual_r3)
            energy[split]["after_r3_by_site"][start:stop] = _record_site_energy(
                residual_r3
            )
            vjp_value = _target_batch(dataset, "vjp", start, stop, device=SPEC.device)
            functional["r2_target"][start:stop] = _functional_energy(residual_r1, vjp_value)
            functional["r2_residual"][start:stop] = _functional_energy(residual_r2, vjp_value)
            functional["r3_target"][start:stop] = _functional_energy(residual_r2, vjp_value)
            functional["r3_residual"][start:stop] = _functional_energy(residual_r3, vjp_value)
            for name, value in (
                ("target", target),
                ("r1_residual", residual_r1),
                ("r2_prediction", prediction_r2),
                ("r2_residual", residual_r2),
                ("r3_prediction", prediction_r3),
                ("r3_residual", residual_r3),
            ):
                sketches[name][start:stop] = _sketch_sites(value, target_projection)
                sketches_secondary[name][start:stop] = _sketch_sites(
                    value, target_projection_secondary
                )
            feature_compact[start:stop] = _feature_compact(features, feature_projection)
            feature_compact_secondary[start:stop] = _feature_compact(
                features, feature_projection_secondary
            )
            for candidate_route in (0, 1):
                candidate_routes = torch.full_like(routes, candidate_route)
                candidate: list[Tensor] = []
                for site_index in range(16):
                    candidate.append(
                        centroids["route"][site_index, candidate_route]
                        .to(SPEC.device)
                        .unsqueeze(0)
                        .expand(stop - start, -1, -1)
                    )
                candidate_prediction = torch.stack(candidate, dim=1)
                route_alternative_loss[start:stop, candidate_route] = _record_energy(
                    target - candidate_prediction
                )
            progress(
                "r4_collect",
                {"split": split, "records_completed": stop, "records": count},
            )
            del (
                features,
                target,
                route_prediction,
                residual_r1,
                prediction_r2,
                residual_r2,
                prediction_r3,
                residual_r3,
                vjp_value,
            )
        collected[split] = {
            "routes": routes_all.numpy(),
            "functional": functional,
            "sketches": sketches,
            "sketches_secondary": sketches_secondary,
            "feature_compact": feature_compact,
            "feature_compact_secondary": feature_compact_secondary,
            "route_alternative_loss": route_alternative_loss,
        }
    return collected


def _residual_null_decision(
    *,
    spectrum_p: float,
    resultant_p: float,
    route_resultant_p: Mapping[str, float],
    cross_alignment_p: float,
    split_half_alignment_p: float,
) -> dict[str, Any]:
    alpha = SPEC.maximum_null_p_value / SPEC.residual_null_family_tests
    spectral = bool(spectrum_p <= alpha)
    resultant = bool(resultant_p <= alpha)
    route_resultant = {
        str(route): bool(value <= alpha)
        for route, value in route_resultant_p.items()
    }
    cross_alignment = bool(cross_alignment_p <= alpha)
    split_half = bool(split_half_alignment_p <= alpha)
    random_compatible = bool(
        not spectral
        and not resultant
        and not any(route_resultant.values())
        and not cross_alignment
        and not split_half
    )
    if random_compatible:
        classification = "consistent_with_current_registered_nulls_only"
    elif (cross_alignment or split_half) and (
        spectral or resultant or any(route_resultant.values())
    ):
        classification = "R5_latent_or_unmodeled_structure"
    elif spectral or resultant or any(route_resultant.values()):
        classification = "unexplained_under_this_screen"
    else:
        classification = "unstable_signal"
    return {
        "alpha": alpha,
        "spectral": spectral,
        "resultant": resultant,
        "route_resultant": route_resultant,
        "cross_alignment": cross_alignment,
        "split_half": split_half,
        "random_compatible": random_compatible,
        "classification": classification,
    }


def _r4_report(
    collected: Mapping[str, Any],
    energy: Mapping[str, Mapping[str, np.ndarray]],
    r2_report: Mapping[str, Any],
    r3_exact: Mapping[str, Any],
    r3_full_j: Mapping[str, Any],
) -> dict[str, Any]:
    train = collected["train"]
    heldout = collected["heldout"]
    route_null = _route_permutation_null(
        heldout["route_alternative_loss"],
        heldout["routes"],
        energy["heldout"]["after_r0"],
        seed=SPEC.random_seed ^ 0x61,
        resamples=SPEC.permutation_replicates,
    )
    score_null_r2_conditional = permutation_score(
        heldout["sketches"]["r2_prediction"],
        heldout["sketches"]["r1_residual"],
        _score_gain,
        strata=heldout["routes"].tolist(),
        seed=SPEC.random_seed ^ 0x62,
        resamples=SPEC.permutation_replicates,
    )
    score_null_r2_unrestricted = permutation_score(
        heldout["sketches"]["r2_prediction"],
        heldout["sketches"]["r1_residual"],
        _score_gain,
        seed=SPEC.random_seed ^ 0x72,
        resamples=SPEC.permutation_replicates,
    )
    score_null_r3_conditional = permutation_score(
        heldout["sketches"]["r3_prediction"],
        heldout["sketches"]["r2_residual"],
        _score_gain,
        strata=heldout["routes"].tolist(),
        seed=SPEC.random_seed ^ 0x63,
        resamples=SPEC.permutation_replicates,
    )
    score_null_r3_unrestricted = permutation_score(
        heldout["sketches"]["r3_prediction"],
        heldout["sketches"]["r2_residual"],
        _score_gain,
        seed=SPEC.random_seed ^ 0x73,
        resamples=SPEC.permutation_replicates,
    )
    fit_null_conditional = permutation_fit_score(
        train["feature_compact"],
        train["sketches"]["r1_residual"],
        _ridge_fit_compact,
        _ridge_score_compact,
        eval_features=heldout["feature_compact"],
        eval_targets=heldout["sketches"]["r1_residual"],
        strata=train["routes"].tolist(),
        seed=SPEC.random_seed ^ 0x64,
        resamples=SPEC.fit_null_replicates,
    )
    fit_null_unrestricted = permutation_fit_score(
        train["feature_compact"],
        train["sketches"]["r1_residual"],
        _ridge_fit_compact,
        _ridge_score_compact,
        eval_features=heldout["feature_compact"],
        eval_targets=heldout["sketches"]["r1_residual"],
        seed=SPEC.random_seed ^ 0x74,
        resamples=SPEC.fit_null_replicates,
    )
    residual = heldout["sketches"]["r3_residual"]
    sign_null = rademacher_sign_null(
        residual,
        _resultant,
        seed=SPEC.random_seed ^ 0x65,
        resamples=SPEC.permutation_replicates,
    )
    route_sign_nulls = {
        str(route): rademacher_sign_null(
            residual[heldout["routes"] == route],
            _resultant,
            seed=SPEC.random_seed ^ (0x75 + route),
            resamples=SPEC.route_conditional_sign_replicates,
        )
        for route in (0, 1)
    }
    spectrum = covariance_spectrum(
        residual,
        rank=SPEC.residual_spectrum_rank,
        seed=SPEC.random_seed ^ 0x66,
    )
    spectrum_null = _matched_spectrum_null(
        residual,
        seed=SPEC.random_seed ^ 0x67,
        resamples=SPEC.matched_spectrum_replicates,
    )
    cross_alignment = cross_split_alignment(
        train["sketches"]["r3_residual"],
        residual,
        rank=min(16, SPEC.residual_spectrum_rank),
        seed=SPEC.random_seed ^ 0x68,
    )
    split_half_alignment = cross_split_alignment(
        residual[::2],
        residual[1::2],
        rank=min(16, SPEC.residual_spectrum_rank),
        seed=SPEC.random_seed ^ 0x69,
    )
    alignment_null = _alignment_null(
        train["sketches"]["r3_residual"],
        residual,
        seed=SPEC.random_seed ^ 0x6A,
        resamples=SPEC.alignment_null_replicates,
        rank=min(8, SPEC.residual_spectrum_rank),
    )
    split_half_alignment_null = _alignment_null(
        residual[::2],
        residual[1::2],
        seed=SPEC.random_seed ^ 0x77,
        resamples=SPEC.alignment_null_replicates,
        rank=min(8, SPEC.residual_spectrum_rank),
    )
    r2_bootstrap = _loss_gain_bootstrap(
        energy["heldout"]["after_r2"],
        energy["heldout"]["after_r1"],
        seed=SPEC.random_seed ^ 0x6B,
        resamples=SPEC.bootstrap_replicates,
    )
    r3_bootstrap = _loss_gain_bootstrap(
        energy["heldout"]["after_r3"],
        energy["heldout"]["after_r2"],
        seed=SPEC.random_seed ^ 0x6C,
        resamples=SPEC.bootstrap_replicates,
    )
    # A second record-cluster implementation on the fixed sketch catches
    # accidental site-as-independent resampling in the exact energy path.
    sketch_bootstrap_r2 = record_cluster_bootstrap(
        heldout["sketches"]["r1_residual"],
        heldout["sketches"]["r2_prediction"],
        baseline=np.zeros_like(heldout["sketches"]["r1_residual"]),
        clusters=list(range(len(heldout["routes"]))),
        seed=SPEC.random_seed ^ 0x6D,
        resamples=SPEC.bootstrap_replicates,
    )
    primary_r2 = r2_report["features"][SPEC.primary_r2_feature]
    r2_replicated = bool(
        primary_r2["heldout"]["explained_energy_gain"] > 0.0
        and r2_bootstrap["ci95"]["lower"] > 0.0
        and score_null_r2_conditional["p_value"] <= SPEC.maximum_null_p_value
        and score_null_r2_unrestricted["p_value"] <= SPEC.maximum_null_p_value
        and fit_null_conditional["p_value"] <= SPEC.maximum_null_p_value
        and fit_null_unrestricted["p_value"] <= SPEC.maximum_null_p_value
    )
    r3_replicated = bool(
        r3_exact["heldout"]["explained_energy_gain"] > 0.0
        and r3_bootstrap["ci95"]["lower"] > 0.0
        and score_null_r3_conditional["p_value"] <= SPEC.maximum_null_p_value
        and score_null_r3_unrestricted["p_value"] <= SPEC.maximum_null_p_value
    )
    full_j_aggregate = r3_full_j["aggregate"]
    r3_qualified = bool(
        r3_exact["heldout"]["increment_nmse"]
        <= SPEC.maximum_qualified_heldout_nmse
        and r3_exact["heldout_train_nmse_ratio"]
        <= SPEC.maximum_qualified_error_ratio
        and full_j_aggregate["heldout_increment_nmse"]
        <= SPEC.maximum_qualified_heldout_nmse
        and full_j_aggregate["heldout_train_nmse_ratio"]
        <= SPEC.maximum_qualified_error_ratio
        and r3_replicated
    )
    # Six pre-registered residual tests share one family-wise alpha.  This is
    # intentionally stricter than looking at independent 95% UCBs.
    residual_decision = _residual_null_decision(
        spectrum_p=float(spectrum_null["p_value"]),
        resultant_p=float(sign_null["p_value"]),
        route_resultant_p={
            route: float(value["p_value"]) for route, value in route_sign_nulls.items()
        },
        cross_alignment_p=float(alignment_null["p_value"]),
        split_half_alignment_p=float(split_half_alignment_null["p_value"]),
    )
    residual_family_alpha = residual_decision["alpha"]
    residual_spectral_signal = residual_decision["spectral"]
    residual_resultant_signal = residual_decision["resultant"]
    residual_route_resultant_signals = residual_decision["route_resultant"]
    residual_alignment_signal = residual_decision["cross_alignment"]
    residual_split_half_signal = residual_decision["split_half"]
    residual_random_compatible = residual_decision["random_compatible"]
    residual_classification = residual_decision["classification"]
    detected: list[str] = []
    if route_null["p_value"] <= SPEC.maximum_null_p_value:
        detected.append("route_conditioned_geometry")
    if r2_replicated:
        detected.append("input_predictable_geometry")
    if r3_replicated:
        detected.append("shared_final_head_parameter_reachability")
    if r3_qualified:
        detected.append("qualified_local_projection_parameter_reachability")
    return {
        "null_contract": {
            "score_null_replicates": SPEC.permutation_replicates,
            "fit_null_replicates": SPEC.fit_null_replicates,
            "matched_spectrum_replicates": SPEC.matched_spectrum_replicates,
            "alignment_null_replicates": SPEC.alignment_null_replicates,
            "route_conditional_sign_replicates": SPEC.route_conditional_sign_replicates,
            "record_cluster_bootstrap_replicates": SPEC.bootstrap_replicates,
            "residual_family_tests": SPEC.residual_null_family_tests,
            "residual_familywise_alpha": SPEC.maximum_null_p_value,
            "residual_per_test_bonferroni_alpha": residual_family_alpha,
            "rank_and_lambda_selected_from_outer_heldout": False,
        },
        "route_permutation": route_null,
        "r2_score_null_within_route": score_null_r2_conditional,
        "r2_score_null_unrestricted_record_id": score_null_r2_unrestricted,
        "r2_fit_null_compact_within_route": fit_null_conditional,
        "r2_fit_null_compact_unrestricted_record_id": fit_null_unrestricted,
        "r3_score_null_within_route": score_null_r3_conditional,
        "r3_score_null_unrestricted_record_id": score_null_r3_unrestricted,
        "residual_sign_null": sign_null,
        "residual_route_conditional_sign_nulls": route_sign_nulls,
        "residual_spectrum": spectrum,
        "residual_spectrum_null": spectrum_null,
        "residual_cross_split_alignment": cross_alignment,
        "residual_split_half_alignment": split_half_alignment,
        "residual_split_half_alignment_null": split_half_alignment_null,
        "residual_alignment_null": alignment_null,
        "r2_exact_energy_bootstrap": r2_bootstrap,
        "r3_exact_energy_bootstrap": r3_bootstrap,
        "r2_sketch_record_cluster_bootstrap": sketch_bootstrap_r2,
        "signals": {
            "r2_replicated": r2_replicated,
            "r3_replicated": r3_replicated,
            "r3_qualified": r3_qualified,
            "residual_spectral_signal": residual_spectral_signal,
            "residual_resultant_signal": residual_resultant_signal,
            "residual_route_resultant_signals": residual_route_resultant_signals,
            "residual_cross_split_alignment_signal": residual_alignment_signal,
            "residual_split_half_alignment_signal": residual_split_half_signal,
            "residual_random_compatible": residual_random_compatible,
        },
        "detected_components": detected,
        "residual_classification": residual_classification,
        "unmeasured_or_boundary": [
            "matched spectral/alignment null preserves row norm and diagonal coordinate scale, not full site/block covariance",
            "raw sketch L2 residual null is not a whitened or gauge-invariant functional null",
            "functional VJP metric is reported but has no independent R4 permutation family",
            "site-balanced null and a second independent random sketch are not measured in v1",
            "negative-VJP is an old training-arm control and is not regenerated from the frozen positive target bank",
            "sampled trunk-plus-head local Jacobian has no separate fit-null and is not the full recurrent Jacobian",
            "family/class/site-time OOD is unavailable in the frozen bank",
            "prepared heldout is an ordered finite split with 1536 unused gap records",
        ],
    }


def _sketch_family_report(
    train: Mapping[str, Any],
    heldout: Mapping[str, Any],
    *,
    sketches_key: str,
    features_key: str,
    seed_base: int,
) -> dict[str, Any]:
    train_sketches = train[sketches_key]
    heldout_sketches = heldout[sketches_key]
    train_features = train[features_key]
    heldout_features = heldout[features_key]
    train_routes = train["routes"].tolist()
    heldout_routes = heldout["routes"].tolist()

    def score_null(stage: str, conditional: bool, offset: int) -> dict[str, Any]:
        prediction = heldout_sketches[f"{stage}_prediction"]
        target = heldout_sketches[
            "r1_residual" if stage == "r2" else "r2_residual"
        ]
        return permutation_score(
            prediction,
            target,
            _score_gain,
            strata=heldout_routes if conditional else None,
            seed=SPEC.random_seed ^ (seed_base + offset),
            resamples=SPEC.permutation_replicates,
        )

    def fit_null(
        stage: str, conditional: bool, feature_permutation: bool, offset: int
    ) -> dict[str, Any]:
        train_target = train_sketches[
            "r1_residual" if stage == "r2" else "r2_residual"
        ]
        heldout_target = heldout_sketches[
            "r1_residual" if stage == "r2" else "r2_residual"
        ]
        function = (
            permutation_feature_fit_score
            if feature_permutation
            else permutation_fit_score
        )
        return function(
            train_features,
            train_target,
            _ridge_fit_compact,
            _ridge_score_compact,
            eval_features=heldout_features,
            eval_targets=heldout_target,
            strata=train_routes if conditional else None,
            seed=SPEC.random_seed ^ (seed_base + offset),
            resamples=SPEC.fit_null_replicates,
        )

    score_nulls = {
        stage: {
            "within_route": score_null(stage, True, 0x01 + 0x10 * index),
            "unrestricted_record_id": score_null(
                stage, False, 0x02 + 0x10 * index
            ),
        }
        for index, stage in enumerate(("r2", "r3"))
    }
    fit_nulls: dict[str, Any] = {}
    for index, stage in enumerate(("r2", "r3")):
        base = 0x30 + 0x10 * index
        fit_nulls[stage] = {
            "target_permutation_within_route": fit_null(stage, True, False, base + 1),
            "target_permutation_unrestricted_record_id": fit_null(
                stage, False, False, base + 2
            ),
            "feature_permutation_within_route": fit_null(stage, True, True, base + 3),
            "feature_permutation_unrestricted_record_id": fit_null(
                stage, False, True, base + 4
            ),
        }

    residual = heldout_sketches["r3_residual"]
    sign_null = rademacher_sign_null(
        residual,
        _resultant,
        seed=SPEC.random_seed ^ (seed_base + 0x60),
        resamples=SPEC.permutation_replicates,
    )
    route_sign_nulls = {
        str(route): rademacher_sign_null(
            residual[heldout["routes"] == route],
            _resultant,
            seed=SPEC.random_seed ^ (seed_base + 0x70 + route),
            resamples=SPEC.route_conditional_sign_replicates,
        )
        for route in (0, 1)
    }
    spectrum = covariance_spectrum(
        residual,
        rank=SPEC.residual_spectrum_rank,
        seed=SPEC.random_seed ^ (seed_base + 0x80),
    )
    spectrum_null = _matched_spectrum_null(
        residual,
        seed=SPEC.random_seed ^ (seed_base + 0x81),
        resamples=SPEC.matched_spectrum_replicates,
    )
    cross_alignment = cross_split_alignment(
        train_sketches["r3_residual"],
        residual,
        rank=min(16, SPEC.residual_spectrum_rank),
        seed=SPEC.random_seed ^ (seed_base + 0x82),
    )
    split_half_alignment = cross_split_alignment(
        residual[::2],
        residual[1::2],
        rank=min(16, SPEC.residual_spectrum_rank),
        seed=SPEC.random_seed ^ (seed_base + 0x83),
    )
    alignment_null = _alignment_null(
        train_sketches["r3_residual"],
        residual,
        seed=SPEC.random_seed ^ (seed_base + 0x84),
        resamples=SPEC.alignment_null_replicates,
        rank=min(8, SPEC.residual_spectrum_rank),
    )
    split_half_alignment_null = _alignment_null(
        residual[::2],
        residual[1::2],
        seed=SPEC.random_seed ^ (seed_base + 0x85),
        resamples=SPEC.alignment_null_replicates,
        rank=min(8, SPEC.residual_spectrum_rank),
    )
    residual_decision = _residual_null_decision(
        spectrum_p=float(spectrum_null["p_value"]),
        resultant_p=float(sign_null["p_value"]),
        route_resultant_p={
            route: float(value["p_value"])
            for route, value in route_sign_nulls.items()
        },
        cross_alignment_p=float(alignment_null["p_value"]),
        split_half_alignment_p=float(split_half_alignment_null["p_value"]),
    )
    sketch_bootstraps = {
        "r2": record_cluster_bootstrap(
            heldout_sketches["r1_residual"],
            heldout_sketches["r2_prediction"],
            baseline=np.zeros_like(heldout_sketches["r1_residual"]),
            clusters=list(range(len(heldout["routes"]))),
            seed=SPEC.random_seed ^ (seed_base + 0x90),
            resamples=SPEC.bootstrap_replicates,
        ),
        "r3": record_cluster_bootstrap(
            heldout_sketches["r2_residual"],
            heldout_sketches["r3_prediction"],
            baseline=np.zeros_like(heldout_sketches["r2_residual"]),
            clusters=list(range(len(heldout["routes"]))),
            seed=SPEC.random_seed ^ (seed_base + 0x91),
            resamples=SPEC.bootstrap_replicates,
        ),
    }
    sign_symmetry = {
        "r2": _negative_target_sign_symmetry(
            train_features,
            train_sketches["r1_residual"],
            heldout_features,
            heldout_sketches["r1_residual"],
        ),
        "r3": _negative_target_sign_symmetry(
            train_features,
            train_sketches["r2_residual"],
            heldout_features,
            heldout_sketches["r2_residual"],
        ),
    }
    return {
        "score_nulls": score_nulls,
        "fit_nulls": fit_nulls,
        "record_cluster_bootstraps": sketch_bootstraps,
        "negative_vjp_sign_symmetry": sign_symmetry,
        "residual": {
            "sign_null": sign_null,
            "route_conditional_sign_nulls": route_sign_nulls,
            "spectrum": spectrum,
            "spectrum_null": spectrum_null,
            "cross_split_alignment": cross_alignment,
            "split_half_alignment": split_half_alignment,
            "alignment_null": alignment_null,
            "split_half_alignment_null": split_half_alignment_null,
            "decision": residual_decision,
        },
    }


def _all_nulls_significant(value: Mapping[str, Any]) -> bool:
    return all(
        float(item["p_value"]) <= SPEC.maximum_null_p_value
        for item in value.values()
    )


def _r4_report_v2(
    collected: Mapping[str, Any],
    energy: Mapping[str, Mapping[str, np.ndarray]],
    r2_report: Mapping[str, Any],
    r3_exact: Mapping[str, Any],
    r3_full_j: Mapping[str, Any],
) -> dict[str, Any]:
    train = collected["train"]
    heldout = collected["heldout"]
    route_null = _route_permutation_null(
        heldout["route_alternative_loss"],
        heldout["routes"],
        energy["heldout"]["after_r0"],
        seed=SPEC.random_seed ^ 0x261,
        resamples=SPEC.permutation_replicates,
    )
    families = {
        "primary": _sketch_family_report(
            train,
            heldout,
            sketches_key="sketches",
            features_key="feature_compact",
            seed_base=0x300,
        ),
        "secondary": _sketch_family_report(
            train,
            heldout,
            sketches_key="sketches_secondary",
            features_key="feature_compact_secondary",
            seed_base=0x500,
        ),
    }
    r2_bootstrap = _loss_gain_bootstrap(
        energy["heldout"]["after_r2"],
        energy["heldout"]["after_r1"],
        seed=SPEC.random_seed ^ 0x26B,
        resamples=SPEC.bootstrap_replicates,
    )
    r3_bootstrap = _loss_gain_bootstrap(
        energy["heldout"]["after_r3"],
        energy["heldout"]["after_r2"],
        seed=SPEC.random_seed ^ 0x26C,
        resamples=SPEC.bootstrap_replicates,
    )
    site_macro = {
        "r2": site_macro_record_bootstrap(
            energy["heldout"]["after_r1_by_site"],
            energy["heldout"]["after_r2_by_site"],
            seed=SPEC.random_seed ^ 0x26D,
            resamples=SPEC.bootstrap_replicates,
            minimum_reference_energy=SPEC.minimum_site_reference_energy,
        ),
        "r3": site_macro_record_bootstrap(
            energy["heldout"]["after_r2_by_site"],
            energy["heldout"]["after_r3_by_site"],
            seed=SPEC.random_seed ^ 0x26E,
            resamples=SPEC.bootstrap_replicates,
            minimum_reference_energy=SPEC.minimum_site_reference_energy,
        ),
    }

    def family_stage_pass(family: Mapping[str, Any], stage: str) -> bool:
        return bool(
            _all_nulls_significant(family["score_nulls"][stage])
            and _all_nulls_significant(family["fit_nulls"][stage])
            and family["record_cluster_bootstraps"][stage]["ci95"]
            ["micro_explained_gain"]["lower"]
            > 0.0
            and family["negative_vjp_sign_symmetry"][stage]["passed"]
        )

    primary_r2 = r2_report["features"][SPEC.primary_r2_feature]
    r2_replicated = bool(
        primary_r2["heldout"]["explained_energy_gain"] > 0.0
        and primary_r2["regularization_stability"]["meets_registered_fraction"]
        and r2_bootstrap["ci95"]["lower"] > 0.0
        and site_macro["r2"]["metric_defined"]
        and site_macro["r2"]["all_resamples_defined"]
        and site_macro["r2"]["ci95"]["site_macro_explained_gain"]["lower"] > 0.0
        and all(family_stage_pass(family, "r2") for family in families.values())
    )
    r3_replicated = bool(
        r3_exact["heldout"]["explained_energy_gain"] > 0.0
        and r3_exact["regularization_stability"]["meets_registered_fraction"]
        and r3_bootstrap["ci95"]["lower"] > 0.0
        and site_macro["r3"]["metric_defined"]
        and site_macro["r3"]["all_resamples_defined"]
        and site_macro["r3"]["ci95"]["site_macro_explained_gain"]["lower"] > 0.0
        and all(family_stage_pass(family, "r3") for family in families.values())
    )
    full_j_aggregate = r3_full_j["aggregate"]
    r3_qualified = bool(
        r3_exact["heldout"]["increment_nmse"]
        <= SPEC.maximum_qualified_heldout_nmse
        and r3_exact["heldout_train_nmse_ratio"]
        <= SPEC.maximum_qualified_error_ratio
        and full_j_aggregate["heldout_increment_nmse"]
        <= SPEC.maximum_qualified_heldout_nmse
        and full_j_aggregate["heldout_train_nmse_ratio"]
        <= SPEC.maximum_qualified_error_ratio
        and r3_replicated
    )
    residual_random_compatible = all(
        bool(family["residual"]["decision"]["random_compatible"])
        for family in families.values()
    )
    residual_classes = {
        name: family["residual"]["decision"]["classification"]
        for name, family in families.items()
    }
    if residual_random_compatible:
        residual_classification = "consistent_with_both_registered_sketch_null_families_only"
    elif "R5_latent_or_unmodeled_structure" in residual_classes.values():
        residual_classification = "R5_latent_or_unmodeled_structure"
    elif "unexplained_under_this_screen" in residual_classes.values():
        residual_classification = "unexplained_under_this_screen"
    else:
        residual_classification = "unstable_or_sketch_dependent_signal"
    detected: list[str] = []
    if route_null["p_value"] <= SPEC.maximum_null_p_value:
        detected.append("route_conditioned_geometry")
    if r2_replicated:
        detected.append("input_predictable_geometry")
    if r3_replicated:
        detected.append("shared_final_head_parameter_reachability")
    if r3_qualified:
        detected.append("qualified_local_projection_parameter_reachability")
    return {
        "null_contract": {
            "score_null_replicates": SPEC.permutation_replicates,
            "fit_null_replicates": SPEC.fit_null_replicates,
            "matched_spectrum_replicates": SPEC.matched_spectrum_replicates,
            "alignment_null_replicates": SPEC.alignment_null_replicates,
            "route_conditional_sign_replicates": SPEC.route_conditional_sign_replicates,
            "record_cluster_bootstrap_replicates": SPEC.bootstrap_replicates,
            "residual_family_tests": SPEC.residual_null_family_tests,
            "residual_familywise_alpha": SPEC.maximum_null_p_value,
            "residual_per_test_bonferroni_alpha": (
                SPEC.maximum_null_p_value / SPEC.residual_null_family_tests
            ),
            "independent_sketch_families": 2,
            "target_and_feature_fit_permutations": True,
            "rank_and_lambda_selected_from_outer_heldout": False,
        },
        "route_permutation": route_null,
        "sketch_families": families,
        "r2_exact_energy_bootstrap": r2_bootstrap,
        "r3_exact_energy_bootstrap": r3_bootstrap,
        "site_macro_record_bootstrap": site_macro,
        "signals": {
            "r2_replicated": r2_replicated,
            "r3_replicated": r3_replicated,
            "r3_qualified": r3_qualified,
            "residual_by_sketch": {
                name: family["residual"]["decision"]
                for name, family in families.items()
            },
            "residual_random_compatible": residual_random_compatible,
        },
        "detected_components": detected,
        "residual_classification": residual_classification,
        "unmeasured_or_boundary": [
            "matched spectral/alignment null preserves row norm and diagonal coordinate scale, not full site/block covariance",
            "raw dual-sketch L2 residual null is not a whitened or gauge-invariant functional null",
            "functional VJP metric is reported but has no independent R4 permutation family",
            "negative-target sign reversal is algebraically symmetric and cannot establish answer-direction causal value",
            "sampled trunk-plus-head local Jacobian is not the full recurrent Jacobian",
            "family/class/site-time OOD is unavailable in the frozen bank",
            "prepared heldout is an ordered finite split with 1536 unused gap records",
        ],
    }


def _model_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def _runtime_fingerprint() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": properties.name,
        "device_total_memory": int(properties.total_memory),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "grad_enabled_during_formal": False,
        "model_eval": True,
    }


def _centroid_alignment(left: Tensor, right: Tensor) -> dict[str, float]:
    left = left.double().reshape(-1)
    right = right.double().reshape(-1)
    dot = float(torch.dot(left, right))
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    return {
        "cosine": dot / max(left_norm * right_norm, 1.0e-30),
        "left_norm": left_norm,
        "right_norm": right_norm,
    }


def _layer_report(
    energy: Mapping[str, Mapping[str, np.ndarray]],
    collected: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "heldout"):
        routes = collected[split]["routes"]
        total = energy[split]["target"]
        after_r0 = energy[split]["after_r0"]
        after_r1 = energy[split]["after_r1"]
        after_r2 = energy[split]["after_r2"]
        after_r3 = energy[split]["after_r3"]
        result[split] = {
            "R0_global": _energy_summary(total, after_r0, routes),
            "R1_route_increment": _energy_summary(
                total, after_r1, routes, baseline_energy=after_r0
            ),
            "R1_global_plus_route": _energy_summary(total, after_r1, routes),
            "R2_input_predictable_increment": _energy_summary(
                total, after_r2, routes, baseline_energy=after_r1
            ),
            "R3_exact_head_increment": _energy_summary(
                total, after_r3, routes, baseline_energy=after_r2
            ),
            "R4_residual_fraction": {
                "raw_micro": float(after_r3.sum()) / max(float(total.sum()), 1.0e-30),
                "record_macro": float(np.mean(after_r3 / np.maximum(total, 1.0e-30))),
            },
            "site_macro": {
                "R0_global": _site_macro_summary(
                    energy[split]["target_by_site"],
                    energy[split]["after_r0_by_site"],
                ),
                "R1_route_increment": _site_macro_summary(
                    energy[split]["target_by_site"],
                    energy[split]["after_r1_by_site"],
                    baseline_by_site=energy[split]["after_r0_by_site"],
                ),
                "R2_input_predictable_increment": _site_macro_summary(
                    energy[split]["target_by_site"],
                    energy[split]["after_r2_by_site"],
                    baseline_by_site=energy[split]["after_r1_by_site"],
                ),
                "R3_exact_head_increment": _site_macro_summary(
                    energy[split]["target_by_site"],
                    energy[split]["after_r3_by_site"],
                    baseline_by_site=energy[split]["after_r2_by_site"],
                ),
            },
            "functional_vjp_metric": {
                "R2": _energy_summary(
                    collected[split]["functional"]["r2_target"],
                    collected[split]["functional"]["r2_residual"],
                    routes,
                ),
                "R3": _energy_summary(
                    collected[split]["functional"]["r3_target"],
                    collected[split]["functional"]["r3_residual"],
                    routes,
                ),
            },
        }
    return result


def _without_score_arrays(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_score_arrays(item)
            for key, item in value.items()
            if key != "scores"
        }
    if isinstance(value, list):
        return [_without_score_arrays(item) for item in value]
    return value


def run_direction_geometry_screen(repo_root: Path | None = None) -> dict[str, Any]:
    """Execute the isolated screen once without optimizer or model writes."""

    root = Path(repo_root or Path.cwd()).resolve()
    output_root = root / OUTPUT_ROOT
    preflight_lease = root / PREFLIGHT_LEASE
    if output_root.exists():
        raise FileExistsError(
            f"direction-geometry output root is already consumed: {output_root}"
        )
    if preflight_lease.exists():
        raise FileExistsError(
            f"direction-geometry identity already leased/consumed: {preflight_lease}"
        )

    # The exclusive sibling ledger consumes the identity before any hard Gate.
    # This preserves root-before-replay absence while making pre-root failures
    # durable and impossible to retry under the same identity.
    preflight_started = time.perf_counter()
    _create_jsonl(
        preflight_lease,
        {
            "schema_version": f"{SCHEMA_PREFIX}.preflight-lease.v1",
            "timestamp": time.time(),
            "status": "LEASED_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2",
            "identity": IDENTITY,
            "scope": SCOPE,
            "authorizes": AUTHORIZES,
            "output_root": OUTPUT_ROOT.as_posix(),
        },
    )
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("direction-geometry screen requires the frozen CUDA path")

        # Replay every record with the historical microbatch before creating
        # the output root.  Scientific null outcomes do not fail this Gate.
        fixed_preflight = verify_fixed_inputs(
            root, expect_preflight_lease_absent=False
        )
        if fixed_preflight["passed"] is not True:
            raise RuntimeError(
                f"successor fixed-input preflight failed: {fixed_preflight}"
            )
        inputs = load_and_validate_inputs(root)
        model = load_deployment_checkpoint(inputs.checkpoint, device=SPEC.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model_digest_before = _model_digest(model)
        datasets = inputs.target_payload["datasets"]
        full_replay_preflight = audit_full_replay_identity(
            model,
            inputs.cache,
            datasets,
            device=SPEC.device,
            batch_size=SPEC.replay_microbatch_size,
            tolerance=SPEC.maximum_replay_absolute_error,
        )
        if full_replay_preflight["passed"] is not True:
            raise RuntimeError(
                "successor full replay identity preflight failed: "
                f"{full_replay_preflight}"
            )
        preflight_elapsed = time.perf_counter() - preflight_started
        _append_jsonl(
            preflight_lease,
            {
                "schema_version": f"{SCHEMA_PREFIX}.preflight-lease.v1",
                "timestamp": time.time(),
                "status": "PASS_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_PRE_ROOT",
                "identity": IDENTITY,
                "elapsed_seconds": preflight_elapsed,
                "maximum_common_absolute_error": full_replay_preflight[
                    "maximum_common_absolute_error"
                ],
                "maximum_projection_absolute_error": full_replay_preflight[
                    "maximum_projection_absolute_error"
                ],
                "authorizes": AUTHORIZES,
            },
            durable=True,
        )
        output_root.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        _append_jsonl(
            preflight_lease,
            {
                "schema_version": f"{SCHEMA_PREFIX}.preflight-lease.v1",
                "timestamp": time.time(),
                "status": "CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_PRE_ROOT",
                "identity": IDENTITY,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - preflight_started,
                "output_root_created": output_root.exists(),
                "same_identity_rerun_authorized": False,
                "authorizes": AUTHORIZES,
            },
            durable=True,
        )
        raise
    events_path = output_root / "events.jsonl"
    state_path = output_root / "run-state.json"
    started = time.perf_counter()

    def progress(stage: str, row: Mapping[str, Any]) -> None:
        event = {"timestamp": time.time(), "stage": stage, **dict(row)}
        _append_jsonl(events_path, event)
        _write_json(
            state_path,
            {
                "schema_version": f"{SCHEMA_PREFIX}.run-state.v1",
                "status": "RUNNING_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2",
                "identity": IDENTITY,
                "stage": stage,
                "latest": dict(row),
                "elapsed_seconds": time.perf_counter() - started,
                "authorizes": AUTHORIZES,
            },
        )

    def record_post_root_crash(exc: Exception) -> None:
        crash = {
            "schema_version": f"{SCHEMA_PREFIX}.crash.v1",
            "status": "CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2",
            "identity": IDENTITY,
            "scope": SCOPE,
            "authorizes": AUTHORIZES,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
            "old_root_mutation_authorized": False,
            "rerun_authorized": False,
        }
        # The durable lease is the authoritative fallback even if a filesystem
        # error prevents one of the root-local terminal files from being made.
        _append_jsonl(
            preflight_lease,
            {
                "schema_version": f"{SCHEMA_PREFIX}.preflight-lease.v1",
                "timestamp": time.time(),
                "status": "CRASH_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_ROOT_AFTER",
                "identity": IDENTITY,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": crash["elapsed_seconds"],
                "output_root_created": True,
                "same_identity_rerun_authorized": False,
                "authorizes": AUTHORIZES,
            },
            durable=True,
        )
        _write_json(output_root / "crash.json", crash)
        _write_json(state_path, crash)

    try:
        _write_json(output_root / "contract-manifest.json", contract_manifest())
        preflight = {
            "schema_version": f"{SCHEMA_PREFIX}.preflight.v1",
            "status": "PASS_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_PREFLIGHT",
            "identity": IDENTITY,
            "scope": SCOPE,
            "authorizes": AUTHORIZES,
            "checks": {
                "cuda_available": True,
                "output_root_was_absent": True,
                "immutable_inputs_valid": True,
                "optimizer_steps": SPEC.optimizer_steps,
                "model_writes": SPEC.model_writes,
                "old_root_mutation": SPEC.old_root_mutation,
            },
            "fixed_input_preflight": fixed_preflight,
            "full_replay_identity": full_replay_preflight,
            "preflight_lease": PREFLIGHT_LEASE.as_posix(),
            "runtime_fingerprint": _runtime_fingerprint(),
            "pre_root_full_replay_elapsed_seconds": preflight_elapsed,
            "input_audit": inputs.audit,
            "target_feature_contract": {
                "target_before_replay_required": True,
                "bank_common_projection_are_output_proxies_only": True,
                "margin_vjp_target_norm_positive_mask_in_X": False,
                "family_route_answer_supervision": False,
            },
        }
        _write_json(output_root / "preflight.json", preflight)
        progress(
            "preflight_complete",
            {
                "message": "full batch-4 replay identity passed before root creation",
                "common_max": full_replay_preflight["maximum_common_absolute_error"],
                "projection_max": full_replay_preflight[
                    "maximum_projection_absolute_error"
                ],
            },
        )
    except Exception as exc:
        record_post_root_crash(exc)
        raise

    try:
        train = datasets["train"]
        heldout = datasets["heldout"]

        progress("r0_r1", {"message": "fitting record-preserving centroids"})
        global_centroid = record_global_centroid(train["transfer"])
        route_centroid = record_route_centroids(
            train["transfer"],
            train["route_schedule"],
            site_metadata=train["site_metadata"],
        )
        heldout_global_oracle = record_global_centroid(heldout["transfer"])
        heldout_route_oracle = record_route_centroids(
            heldout["transfer"],
            heldout["route_schedule"],
            site_metadata=heldout["site_metadata"],
        )
        centroids = {"global": global_centroid, "route": route_centroid}
        centroid_report = {
            "R0_train_heldout_global_alignment": _centroid_alignment(
                global_centroid, heldout_global_oracle
            ),
            "R1_train_heldout_route0_alignment": _centroid_alignment(
                route_centroid[:, 0], heldout_route_oracle[:, 0]
            ),
            "R1_train_heldout_route1_alignment": _centroid_alignment(
                route_centroid[:, 1], heldout_route_oracle[:, 1]
            ),
            "coordinate_contract": "record mean preserves site_slot_output coordinates",
            "pooled_256d_centroid_used_for_energy": False,
        }

        statistics, primary_features, energy, replay_audit = _capture_r2(
            model, inputs, centroids, progress
        )
        sampled_states = replay_audit.pop("sampled_states")
        _write_json(
            output_root / "replay-audit.json",
            {"schema_version": f"{SCHEMA_PREFIX}.replay.v1", **replay_audit},
        )
        progress("r2_fit", {"message": "solving frozen-feature ridge predictors"})
        r2_fits, r2_report = _fit_r2(statistics)
        primary_r2_fit = r2_fits[SPEC.primary_r2_feature]
        del statistics

        r3_statistics, sampled_targets = _accumulate_r3(
            inputs,
            centroids,
            primary_features,
            primary_r2_fit,
            replay_audit,
            energy,
            progress,
        )
        progress("r3_exact", {"message": "solving exact shared final-head Jacobian"})
        r3_fit, r3_exact = _fit_r3_exact(r3_statistics)
        del r3_statistics
        r3_full_j = _run_sampled_full_j(
            model,
            inputs,
            sampled_states,
            sampled_targets,
            replay_audit,
            progress,
        )
        del sampled_states, sampled_targets
        torch.cuda.empty_cache()

        collected = _collect_final_geometry(
            inputs,
            centroids,
            primary_features,
            primary_r2_fit,
            r3_fit,
            energy,
            progress,
        )
        del primary_features, r2_fits
        progress("r4_nulls", {"message": "running frozen permutation and spectral nulls"})
        r4_full = _r4_report_v2(collected, energy, r2_report, r3_exact, r3_full_j)
        layers = _layer_report(energy, collected)

        # Full null score arrays are preserved separately.  result.json keeps
        # the audit summaries so it remains human-reviewable.
        _write_json(
            output_root / "null-distributions.json",
            {
                "schema_version": f"{SCHEMA_PREFIX}.nulls.v1",
                "scope": SCOPE,
                "authorizes": AUTHORIZES,
                "R4": r4_full,
            },
        )
        r4 = _without_score_arrays(r4_full)
        model_digest_after = _model_digest(model)
        source_unchanged = sha256_file(inputs.checkpoint) == SOURCE_CHECKPOINT_SHA256
        target_unchanged = (
            sha256_file(inputs.target_path) == inputs.audit["target_bank_sha256"]
        )
        parameter_unchanged = model_digest_before == model_digest_after
        post_fixed_preflight = verify_fixed_inputs(
            root, expect_preflight_lease_absent=False
        )
        post_fixed_inputs_unchanged = all(
            value
            for key, value in post_fixed_preflight["checks"].items()
            if key != "output_root_absent"
        )
        r3_qualified = bool(r4["signals"]["r3_qualified"])
        residual_random = bool(r4["signals"]["residual_random_compatible"])
        r2_replicated = bool(r4["signals"]["r2_replicated"])
        if r3_qualified and residual_random:
            scientific_status = "R3_QUALIFIED_R4_REGISTERED_NULL_COMPATIBLE"
        elif r3_qualified:
            scientific_status = "R3_QUALIFIED_WITH_RESIDUAL_STRUCTURE"
        elif r2_replicated:
            scientific_status = "R2_REPLICATED_R3_NOT_QUALIFIED"
        else:
            scientific_status = "NO_QUALIFIED_R2_R3_COMPONENT"
        status = "COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2"
        result = {
            "schema_version": f"{SCHEMA_PREFIX}.result.v1",
            "status": status,
            "scientific_status": scientific_status,
            "valid_complete_measurement": True,
            "identity": IDENTITY,
            "scope": SCOPE,
            "authorizes": AUTHORIZES,
            "claims": {
                "direction_geometry_signal": bool(r4["detected_components"]),
                "parameter_reachable_signal": r3_qualified,
                "h1_qualified": False,
                "p1_completed": False,
                "f1_authorized": False,
                "p2_authorized": False,
            },
            "input_boundary": {
                "heldout_iid_assumed": False,
                "family_group_ood_measured": False,
                "gap_records_used": False,
                "bank_output_predictors_are_proxies": True,
                "target_before_state_replayed": True,
            },
            "centroid_geometry": centroid_report,
            "layers": layers,
            "R1": {
                "centroid_geometry": centroid_report,
                "train": layers["train"]["R1_route_increment"],
                "heldout": layers["heldout"]["R1_route_increment"],
                "route_permutation": r4["route_permutation"],
            },
            "R2": r2_report,
            "R3": {
                "exact_final_head_full_bank": r3_exact,
                "matrix_free_trunk_plus_heads_sample": r3_full_j,
                "full_recurrent_jacobian_measured": False,
            },
            "R4": r4,
            "interpretation": {
                "detected_components": r4["detected_components"],
                "residual_classification": r4["residual_classification"],
                "old_raw_vjp_W_rejudged": False,
                "successor_training_authorized": False,
                "scientific_no_signal_counts_as_completed_measurement": True,
            },
            "integrity": {
                "source_checkpoint_unchanged": source_unchanged,
                "causal_target_bank_unchanged": target_unchanged,
                "in_memory_model_parameters_unchanged": parameter_unchanged,
                "optimizer_steps": 0,
                "model_or_checkpoint_files_written": False,
                "old_root_mutated": False,
                "predecessor_crash_archive_preserved": post_fixed_inputs_unchanged,
                "post_run_fixed_inputs_unchanged": post_fixed_inputs_unchanged,
                "post_run_fixed_input_checks": post_fixed_preflight["checks"],
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not all(
            (
                source_unchanged,
                target_unchanged,
                parameter_unchanged,
                post_fixed_inputs_unchanged,
            )
        ):
            raise RuntimeError("direction-geometry immutable input/parameter drift")
        _write_json(output_root / "result.json", result)
        _write_json(
            state_path,
            {
                "schema_version": f"{SCHEMA_PREFIX}.run-state.v1",
                "status": status,
                "scientific_status": scientific_status,
                "identity": IDENTITY,
                "stage": "complete",
                "elapsed_seconds": result["elapsed_seconds"],
                "authorizes": AUTHORIZES,
            },
        )
        _append_jsonl(
            events_path,
            {
                "timestamp": time.time(),
                "stage": "complete",
                "status": status,
                "scientific_status": scientific_status,
            },
        )
        return result
    except Exception as exc:
        record_post_root_crash(exc)
        raise


__all__ = ["run_direction_geometry_screen"]
