from __future__ import annotations

"""Matrix-free direction geometry for the H1 routed projection head.

The functions in this module deliberately operate only on already materialized
feature/transfer tensors.  A routed final projection is affine in its feature
vector, so its parameter reachability can be tested exactly with sufficient
statistics; constructing a dense output-by-parameter Jacobian is unnecessary
and would be prohibitively large for the production H1 dimensions.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def _as_tensor(value: Any, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        result = value
    elif isinstance(value, np.ndarray):
        result = torch.from_numpy(value)
    else:
        result = torch.as_tensor(value)
    if not result.is_floating_point() and name not in {"routes", "route_schedule"}:
        result = result.to(dtype=torch.float32)
    return result


def _site_sequence(value: Any, *, name: str) -> tuple[Tensor, ...]:
    """Normalize [N,S,K,D] or a sequence of [N,K,D] site tensors."""

    if isinstance(value, Tensor) or isinstance(value, np.ndarray):
        tensor = _as_tensor(value, name=name)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [N,S,K,C] or [N,K,C]")
        return tuple(tensor[:, index] for index in range(tensor.shape[1]))
    if not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty tensor or site sequence")
    sites = tuple(_as_tensor(item, name=name) for item in value)
    if any(item.ndim != 3 for item in sites):
        raise ValueError(f"each {name} site must have shape [N,K,C]")
    first = sites[0].shape[:2]
    if any(item.shape[:2] != first for item in sites):
        raise ValueError(f"{name} site batch/slot shapes disagree")
    return sites


def _metadata(
    site_count: int,
    route_schedule: Any | None,
    site_metadata: Sequence[Any] | None,
) -> tuple[tuple[int, int], ...]:
    if site_metadata is not None:
        result: list[tuple[int, int]] = []
        for item in site_metadata:
            if isinstance(item, Mapping):
                layer = item.get("layer_index")
                step = item.get("step_index")
            else:
                try:
                    layer, step = item
                except (TypeError, ValueError) as exc:
                    raise ValueError("site metadata must contain layer_index and step_index") from exc
            if not isinstance(layer, (int, np.integer)) or not isinstance(step, (int, np.integer)):
                raise ValueError("site metadata indices must be integers")
            result.append((int(layer), int(step)))
        if len(result) != site_count:
            raise ValueError("site metadata count does not match site tensors")
        return tuple(result)

    if route_schedule is None:
        return tuple((index, 0) for index in range(site_count))
    routes = _as_tensor(route_schedule, name="route_schedule")
    if routes.ndim == 3:
        _, layers, steps = routes.shape
        inferred = tuple((layer, step) for step in range(steps) for layer in range(layers))
    elif routes.ndim == 2:
        inferred = tuple((index, 0) for index in range(routes.shape[1]))
    else:
        raise ValueError("route_schedule must have shape [N,L,T] or [N,S]")
    if len(inferred) != site_count:
        raise ValueError("route schedule site count does not match site tensors")
    return inferred


def _site_routes(
    route_schedule: Any,
    metadata: Sequence[tuple[int, int]],
    *,
    route_count: int,
) -> tuple[Tensor, ...]:
    routes = _as_tensor(route_schedule, name="route_schedule").to(dtype=torch.long)
    if routes.ndim == 2:
        if routes.shape[1] != len(metadata):
            raise ValueError("per-site route schedule count does not match sites")
        values = tuple(routes[:, index] for index in range(len(metadata)))
    elif routes.ndim == 3:
        values = []
        for layer, step in metadata:
            if layer < 0 or step < 0 or layer >= routes.shape[1] or step >= routes.shape[2]:
                raise ValueError("site metadata is outside route schedule")
            values.append(routes[:, layer, step])
        values = tuple(values)
    else:
        raise ValueError("route_schedule must have shape [N,L,T] or [N,S]")
    if any(value.ndim != 1 for value in values):
        raise ValueError("route schedule entries must be one-dimensional")
    if any(bool(((value < 0) | (value >= route_count)).any()) for value in values):
        raise ValueError("route schedule contains an invalid route")
    return values


def _rows(
    features: Any,
    transfer: Any,
    route_schedule: Any | None = None,
    *,
    site_metadata: Sequence[Any] | None = None,
    slot_flatten: bool = True,
) -> tuple[Tensor, Tensor, Tensor, tuple[tuple[int, int], ...], tuple[Tensor, ...]]:
    """Return flattened feature/target rows and their site/route indices."""

    feature_sites = _site_sequence(features, name="features")
    target_sites = _site_sequence(transfer, name="transfer")
    if len(feature_sites) != len(target_sites):
        raise ValueError("features and transfer site counts disagree")
    for feature, target in zip(feature_sites, target_sites, strict=True):
        if feature.shape[:2] != target.shape[:2]:
            raise ValueError("feature and transfer batch/slot shapes disagree")
    if not feature_sites:
        raise ValueError("at least one site is required")
    batch = feature_sites[0].shape[0]
    if any(site.shape[0] != batch for site in feature_sites):
        raise ValueError("feature site batch sizes disagree")
    metadata = _metadata(len(feature_sites), route_schedule, site_metadata)
    if route_schedule is None:
        raise ValueError("route_schedule is required for routed geometry")
    routes = _site_routes(route_schedule, metadata, route_count=2)
    feature_width = feature_sites[0].shape[-1]
    output_width = target_sites[0].shape[-1]
    if any(site.shape[-1] != feature_width for site in feature_sites):
        raise ValueError("feature widths disagree")
    if any(site.shape[-1] != output_width for site in target_sites):
        raise ValueError("transfer widths disagree")

    feature_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    group_rows: list[Tensor] = []
    for site_index, (feature, target, route, (layer, _step)) in enumerate(
        zip(feature_sites, target_sites, routes, metadata, strict=True)
    ):
        slots = feature.shape[1]
        if slot_flatten:
            feature_rows.append(feature.reshape(-1, feature_width))
            target_rows.append(target.reshape(-1, output_width))
            group_rows.append((route[:, None].expand(-1, slots).reshape(-1) + 2 * layer).to(torch.long))
        else:
            # The fit is still row-wise, but preserving this branch makes the
            # slot-flattening contract explicit for callers and tests.
            feature_rows.append(feature.reshape(-1, feature_width))
            target_rows.append(target.reshape(-1, output_width))
            group_rows.append((route[:, None].expand(-1, slots).reshape(-1) + 2 * layer).to(torch.long))
    return (
        torch.cat(feature_rows, dim=0),
        torch.cat(target_rows, dim=0),
        torch.cat(group_rows, dim=0),
        metadata,
        routes,
    )


def _flatten_vectors(value: Any, *, name: str) -> Tensor:
    if isinstance(value, (Tensor, np.ndarray)):
        tensor = _as_tensor(value, name=name)
        if tensor.ndim < 2:
            raise ValueError(f"{name} must have at least two dimensions")
        return tensor.reshape(-1, tensor.shape[-1])
    sites = _site_sequence(value, name=name)
    return torch.cat([site.reshape(-1, site.shape[-1]) for site in sites], dim=0)


def global_centroid(transfer: Any) -> Tensor:
    """R0: centroid of all transfer increments, flattening sites and slots."""

    rows = _flatten_vectors(transfer, name="transfer")
    return rows.mean(dim=0)


def record_global_centroid(transfer: Any) -> Tensor:
    """R0 centroid of the stacked record target, preserving site/slot coordinates.

    The older ``global_centroid`` answers a different output-space question by
    pooling sites and slots into one 256-D vector.  R0--R4 treats one record as
    the full 16-site object, so its stable constant direction is the mean over
    records only and has shape ``[site, slot, output]``.
    """

    sites = _site_sequence(transfer, name="transfer")
    return torch.stack(tuple(site.mean(dim=0) for site in sites), dim=0)


def record_route_centroids(
    transfer: Any,
    route_schedule: Any,
    *,
    site_metadata: Sequence[Any] | None = None,
    route_count: int = 2,
) -> Tensor:
    """R1 ``[site, route, slot, output]`` means over records only."""

    sites = _site_sequence(transfer, name="transfer")
    metadata = _metadata(len(sites), route_schedule, site_metadata)
    routes = _site_routes(route_schedule, metadata, route_count=route_count)
    result = torch.zeros(
        len(sites),
        route_count,
        sites[0].shape[1],
        sites[0].shape[2],
        dtype=sites[0].dtype,
        device=sites[0].device,
    )
    for site_index, (site, route) in enumerate(zip(sites, routes, strict=True)):
        route = route.to(site.device)
        for route_index in range(route_count):
            selected = route == route_index
            if not bool(selected.any()):
                raise ValueError(f"empty route cell at site {site_index}, route {route_index}")
            result[site_index, route_index] = site[selected].mean(dim=0)
    return result


def predict_record_global(centroid: Tensor, records: int) -> Tensor:
    if centroid.ndim != 3 or records <= 0:
        raise ValueError("record-global centroid must have shape [site,slot,output]")
    return centroid.unsqueeze(0).expand(records, -1, -1, -1)


def predict_record_route(
    centroids: Tensor,
    route_schedule: Any,
    *,
    site_metadata: Sequence[Any] | None = None,
) -> Tensor:
    if centroids.ndim != 4 or centroids.shape[1] != 2:
        raise ValueError("record-route centroids must have shape [site,2,slot,output]")
    metadata = _metadata(centroids.shape[0], route_schedule, site_metadata)
    routes = _site_routes(route_schedule, metadata, route_count=2)
    outputs: list[Tensor] = []
    for site_index, route in enumerate(routes):
        values = centroids[site_index].to(route.device)
        outputs.append(values.index_select(0, route.to(torch.long)))
    return torch.stack(outputs, dim=1)


def _unit_rows(rows: Tensor, eps: float = 1.0e-12) -> Tensor:
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    norms = torch.linalg.vector_norm(rows, dim=-1, keepdim=True)
    return rows / norms.clamp_min(float(eps))


def global_angular_centroid(transfer: Any, *, eps: float = 1.0e-12) -> Tensor:
    """R0 angular centroid: normalize each slot direction before averaging."""

    rows = _flatten_vectors(transfer, name="transfer")
    centroid = _unit_rows(rows, eps).mean(dim=0)
    return centroid / torch.linalg.vector_norm(centroid).clamp_min(float(eps))


def site_route_centroids(
    transfer: Any,
    route_schedule: Any,
    *,
    site_metadata: Sequence[Any] | None = None,
    route_count: int = 2,
) -> Tensor:
    """R1: return ``[site, route, output]`` centroids over flattened slots."""

    if route_count < 1:
        raise ValueError("route_count must be positive")
    sites = _site_sequence(transfer, name="transfer")
    metadata = _metadata(len(sites), route_schedule, site_metadata)
    routes = _site_routes(route_schedule, metadata, route_count=route_count)
    output_width = sites[0].shape[-1]
    result = torch.zeros((len(sites), route_count, output_width), dtype=sites[0].dtype, device=sites[0].device)
    for index, (site, route) in enumerate(zip(sites, routes, strict=True)):
        route = route.to(device=site.device)
        for route_index in range(route_count):
            selected = route == route_index
            if bool(selected.any()):
                result[index, route_index] = site[selected].mean(dim=(0, 1))
    return result


def site_route_angular_centroids(
    transfer: Any,
    route_schedule: Any,
    *,
    site_metadata: Sequence[Any] | None = None,
    route_count: int = 2,
    eps: float = 1.0e-12,
) -> Tensor:
    """R1 angular centroid with route-conditioned slot pooling."""

    sites = _site_sequence(transfer, name="transfer")
    metadata = _metadata(len(sites), route_schedule, site_metadata)
    routes = _site_routes(route_schedule, metadata, route_count=route_count)
    output_width = sites[0].shape[-1]
    result = torch.zeros(
        (len(sites), route_count, output_width),
        dtype=sites[0].dtype,
        device=sites[0].device,
    )
    for index, (site, route) in enumerate(zip(sites, routes, strict=True)):
        route = route.to(device=site.device)
        for route_index in range(route_count):
            selected = route == route_index
            if bool(selected.any()):
                rows = site[selected].reshape(-1, output_width)
                centroid = _unit_rows(rows, eps).mean(dim=0)
                result[index, route_index] = centroid / torch.linalg.vector_norm(centroid).clamp_min(float(eps))
    return result


def transfer_increment_report(
    target: Any,
    prediction: Any,
    *,
    baseline: Any | None = None,
    eps: float = 1.0e-12,
) -> dict[str, float | int]:
    """Report SSE/NMSE/cosine and gain over a baseline prediction.

    With no baseline, the baseline is the zero transfer, which is the relevant
    null for a transfer-increment reachability screen.
    """

    target_rows = _flatten_vectors(target, name="target")
    prediction_rows = _flatten_vectors(prediction, name="prediction")
    if target_rows.shape != prediction_rows.shape:
        raise ValueError("target and prediction shapes disagree")
    target_rows = target_rows.to(dtype=torch.float64)
    prediction_rows = prediction_rows.to(dtype=torch.float64)
    residual = prediction_rows - target_rows
    sse = float(residual.square().sum().item())
    target_energy = float(target_rows.square().sum().item())
    pred_energy = float(prediction_rows.square().sum().item())
    denominator = max(target_energy, eps)
    cosine_denominator = max(float(torch.linalg.vector_norm(target_rows).item()) * float(torch.linalg.vector_norm(prediction_rows).item()), eps)
    cosine = float((prediction_rows * target_rows).sum().item() / cosine_denominator)
    if baseline is None:
        baseline_rows = torch.zeros_like(target_rows)
    else:
        baseline_rows = _flatten_vectors(baseline, name="baseline").to(dtype=torch.float64)
        if baseline_rows.shape != target_rows.shape:
            raise ValueError("baseline and target shapes disagree")
    baseline_sse = float((baseline_rows - target_rows).square().sum().item())
    explained_gain = float((baseline_sse - sse) / max(baseline_sse, eps))
    return {
        "transfer_increment_sse": sse,
        "transfer_increment_nmse": float(sse / denominator),
        "transfer_increment_cosine": cosine,
        "explained_gain": explained_gain,
        "target_energy": target_energy,
        "prediction_energy": pred_energy,
        "baseline_sse": baseline_sse,
        "rows": int(target_rows.shape[0]),
    }


@dataclass(frozen=True)
class RoutedLinearFit:
    """Analytic shared ``layer x route`` affine projection update."""

    coefficients: Tensor  # [groups, feature_width + 1, output_width]
    group_keys: tuple[tuple[int, int], ...]
    ridge_lambda: float
    feature_width: int
    output_width: int
    slot_flatten: bool
    lambda_validation_nmse: Mapping[float, float]

    def predict(
        self,
        features: Any,
        route_schedule: Any,
        *,
        site_metadata: Sequence[Any] | None = None,
    ) -> Any:
        feature_sites = _site_sequence(features, name="features")
        metadata = _metadata(len(feature_sites), route_schedule, site_metadata)
        routes = _site_routes(route_schedule, metadata, route_count=2)
        outputs: list[Tensor] = []
        key_to_index = {key: index for index, key in enumerate(self.group_keys)}
        for feature, route, (layer, _step) in zip(feature_sites, routes, metadata, strict=True):
            route = route.to(device=feature.device)
            if feature.shape[-1] != self.feature_width:
                raise ValueError("feature width does not match fitted head")
            augmented = torch.cat(
                (feature.to(dtype=self.coefficients.dtype), torch.ones_like(feature[..., :1], dtype=self.coefficients.dtype)),
                dim=-1,
            )
            coefficients = self.coefficients.to(device=feature.device)
            output = torch.zeros((*feature.shape[:-1], self.output_width), dtype=self.coefficients.dtype, device=feature.device)
            for route_index in range(2):
                selected = route == route_index
                key = (layer, route_index)
                if bool(selected.any()):
                    if key not in key_to_index:
                        raise ValueError(f"fitted head has no group {key}")
                    values = torch.einsum(
                        "nkh,hd->nkd",
                        augmented[selected],
                        coefficients[key_to_index[key]],
                    )
                    output[selected] = values
            outputs.append(output)
        if isinstance(features, Tensor) or isinstance(features, np.ndarray):
            original = _as_tensor(features, name="features")
            if original.ndim == 3:
                return outputs[0]
            return torch.stack(outputs, dim=1)
        return tuple(outputs)

    __call__ = predict


def _fit_coefficients(
    features: Tensor,
    targets: Tensor,
    groups: Tensor,
    group_keys: tuple[tuple[int, int], ...],
    ridge_lambda: float,
) -> Tensor:
    dtype = torch.float64
    features = features.to(dtype=dtype)
    targets = targets.to(dtype=dtype)
    groups = groups.to(dtype=torch.long)
    augmented = torch.cat((features, torch.ones((features.shape[0], 1), dtype=dtype, device=features.device)), dim=1)
    coefficients = torch.zeros((len(group_keys), augmented.shape[1], targets.shape[1]), dtype=dtype, device=features.device)
    penalty = torch.eye(augmented.shape[1], dtype=dtype, device=features.device)
    penalty[-1, -1] = 0.0  # do not penalize the affine bias
    for group_index, group_key in enumerate(group_keys):
        selected = groups == (2 * group_key[0] + group_key[1])
        if not bool(selected.any()):
            continue
        x = augmented[selected]
        y = targets[selected]
        gram = x.transpose(0, 1) @ x + float(ridge_lambda) * penalty
        rhs = x.transpose(0, 1) @ y
        coefficients[group_index] = torch.linalg.lstsq(gram, rhs).solution
    return coefficients


def fit_routed_linear_ridge(
    features: Any,
    transfer: Any,
    route_schedule: Any,
    *,
    site_metadata: Sequence[Any] | None = None,
    lambda_grid: Iterable[float] = (0.0, 1.0e-6, 1.0e-4, 1.0e-2, 1.0),
    validation_fraction: float = 0.2,
    seed: int = 0,
    slot_flatten: bool = True,
) -> RoutedLinearFit:
    """Fit a shared final routed linear head using train-only lambda choice.

    The input tensors are assumed to be the train split.  The internal split
    used for lambda selection never consults a heldout tensor.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0,1)")
    candidates = tuple(sorted({float(value) for value in lambda_grid}))
    if not candidates or any(value < 0.0 or not math.isfinite(value) for value in candidates):
        raise ValueError("lambda_grid must contain non-negative finite values")
    feature_rows, target_rows, groups, metadata, _ = _rows(
        features,
        transfer,
        route_schedule,
        site_metadata=site_metadata,
        slot_flatten=slot_flatten,
    )
    if not torch.isfinite(feature_rows).all() or not torch.isfinite(target_rows).all():
        raise ValueError("features and transfer must be finite")
    max_layer = max(layer for layer, _ in metadata)
    group_keys = tuple((layer, route) for layer in range(max_layer + 1) for route in range(2))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    validation = torch.zeros(groups.shape[0], dtype=torch.bool)
    # Split inside each group so every populated route gets a validation check.
    for group_key in group_keys:
        group_index = 2 * group_key[0] + group_key[1]
        indices = torch.nonzero(groups.cpu() == group_index, as_tuple=False).flatten()
        if len(indices) < 2:
            continue
        permutation = indices[torch.randperm(len(indices), generator=generator)]
        count = min(len(indices) - 1, max(1, int(round(len(indices) * validation_fraction))))
        validation[permutation[:count]] = True
    fit_mask = (~validation).to(device=feature_rows.device)
    validation_mask = validation.to(device=feature_rows.device)
    validation_scores: dict[float, float] = {}
    for value in candidates:
        candidate = _fit_coefficients(feature_rows[fit_mask], target_rows[fit_mask], groups[fit_mask], group_keys, value)
        predicted = _predict_rows(feature_rows[validation_mask], groups[validation_mask], candidate, group_keys)
        if predicted.shape[0]:
            validation_scores[value] = float(transfer_increment_report(target_rows[validation_mask], predicted)["transfer_increment_nmse"])
        else:
            validation_scores[value] = float("inf")
    selected_lambda = min(candidates, key=lambda value: (validation_scores[value], value))
    coefficients = _fit_coefficients(feature_rows, target_rows, groups, group_keys, selected_lambda)
    return RoutedLinearFit(
        coefficients=coefficients.cpu(),
        group_keys=group_keys,
        ridge_lambda=selected_lambda,
        feature_width=feature_rows.shape[-1],
        output_width=target_rows.shape[-1],
        slot_flatten=slot_flatten,
        lambda_validation_nmse=validation_scores,
    )


def _predict_rows(features: Tensor, groups: Tensor, coefficients: Tensor, group_keys: Sequence[tuple[int, int]]) -> Tensor:
    augmented = torch.cat(
        (
            features.to(dtype=coefficients.dtype),
            torch.ones((features.shape[0], 1), dtype=coefficients.dtype, device=features.device),
        ),
        dim=1,
    )
    output = torch.zeros((features.shape[0], coefficients.shape[-1]), dtype=coefficients.dtype, device=features.device)
    for index, key in enumerate(group_keys):
        selected = groups == 2 * key[0] + key[1]
        if bool(selected.any()):
            output[selected] = augmented[selected] @ coefficients[index]
    return output


# Explicit aliases keep the geometry vocabulary discoverable without adding
# compatibility code to the consumed H1-WD modules.
global_transfer_centroid = global_centroid
site_route_centroid = site_route_centroids
global_site_angular_centroid = global_angular_centroid
site_route_angular_centroid = site_route_angular_centroids
fit_shared_routed_linear_head = fit_routed_linear_ridge
report_transfer_increment = transfer_increment_report


__all__ = [
    "RoutedLinearFit",
    "fit_routed_linear_ridge",
    "fit_shared_routed_linear_head",
    "global_centroid",
    "global_angular_centroid",
    "global_site_angular_centroid",
    "global_transfer_centroid",
    "predict_record_global",
    "predict_record_route",
    "record_global_centroid",
    "record_route_centroids",
    "report_transfer_increment",
    "site_route_centroid",
    "site_route_centroids",
    "site_route_angular_centroid",
    "site_route_angular_centroids",
    "transfer_increment_report",
]
