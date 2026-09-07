from __future__ import annotations

"""Target-before feature capture from the immutable predecessor trajectory."""

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from yggdrasil_v2.r1_revalidation.h1.cache import TokenCache
from yggdrasil_v2.r1_revalidation.h1.model import H1LatentReasoner
from yggdrasil_v2.r1_revalidation.h1_wd_causal.core import forward_route_replay


@dataclass(frozen=True)
class CapturedSite:
    layer_index: int
    step_index: int
    route: Tensor
    attention_state: Tensor
    projection_feature: Tensor
    common_private_hidden: Tensor
    common_output_proxy: Tensor
    projection_output_proxy: Tensor


@dataclass(frozen=True)
class CapturedBatch:
    split: str
    start: int
    stop: int
    record_ids: tuple[str, ...]
    sites: tuple[CapturedSite, ...]
    maximum_common_replay_error: float
    maximum_projection_replay_error: float
    common_squared_error_sum: float
    projection_squared_error_sum: float
    common_reference_squared_sum: float
    projection_reference_squared_sum: float
    replay_elements: int
    target_before_features_finite: bool


def _common_private_hidden(layer: Any, state: Tensor) -> Tensor:
    normed = layer.shared_ffn.ffn_norm(state)
    branch = layer.shared_ffn.ffn
    return F.silu(branch.gate(normed)) * branch.up(normed)


def capture_batch(
    model: H1LatentReasoner,
    cache: TokenCache,
    dataset: Mapping[str, Any],
    indices: Sequence[int] | np.ndarray,
    *,
    device: str,
    split: str,
) -> CapturedBatch:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("capture indices must be a non-empty vector")
    record_ids = tuple(dataset["record_ids"][int(index)] for index in selected)
    cache_indices = cache.indices(record_ids)
    hidden, mask = cache.batch(cache_indices, device=device)
    route = dataset["route_schedule"].index_select(
        0, torch.as_tensor(selected, dtype=torch.long)
    ).to(device=device, non_blocking=True)
    with torch.no_grad():
        replay = forward_route_replay(
            model,
            hidden,
            mask,
            route_schedule=route,
            capture=True,
            return_trajectory=False,
        )
        raw_sites = replay["captures"]
        if raw_sites is None or len(raw_sites) != 16:
            raise RuntimeError("route replay did not return the registered 16 sites")
        captured: list[CapturedSite] = []
        common_error = 0.0
        projection_error = 0.0
        common_squared_error_sum = 0.0
        projection_squared_error_sum = 0.0
        common_reference_squared_sum = 0.0
        projection_reference_squared_sum = 0.0
        replay_elements = 0
        features_finite = True
        cpu_indices = torch.as_tensor(selected, dtype=torch.long)
        for site_index, raw in enumerate(raw_sites):
            layer = model.layers[raw.layer_index]
            feature = layer.routed_feature_trunk(raw.attention_state)
            common_hidden = _common_private_hidden(layer, raw.attention_state)
            expected_common = dataset["common"][site_index].index_select(
                0, cpu_indices
            ).to(device=device, non_blocking=True)
            expected_projection = dataset["projection"][site_index].index_select(
                0, cpu_indices
            ).to(device=device, non_blocking=True)
            common_difference = raw.common - expected_common
            projection_difference = raw.projection - expected_projection
            common_error = max(common_error, float(common_difference.abs().max().cpu()))
            projection_error = max(
                projection_error, float(projection_difference.abs().max().cpu())
            )
            common_squared_error_sum += float(
                common_difference.double().square().sum().cpu()
            )
            projection_squared_error_sum += float(
                projection_difference.double().square().sum().cpu()
            )
            common_reference_squared_sum += float(
                expected_common.double().square().sum().cpu()
            )
            projection_reference_squared_sum += float(
                expected_projection.double().square().sum().cpu()
            )
            replay_elements += int(common_difference.numel())
            features_finite = bool(
                features_finite
                and torch.isfinite(raw.attention_state).all()
                and torch.isfinite(feature).all()
                and torch.isfinite(common_hidden).all()
                and torch.isfinite(raw.common).all()
                and torch.isfinite(raw.projection).all()
            )
            captured.append(
                CapturedSite(
                    layer_index=int(raw.layer_index),
                    step_index=int(raw.step_index),
                    route=raw.route.detach(),
                    attention_state=raw.attention_state.detach(),
                    projection_feature=feature.detach(),
                    common_private_hidden=common_hidden.detach(),
                    common_output_proxy=raw.common.detach(),
                    projection_output_proxy=raw.projection.detach(),
                )
            )
    del hidden, mask, replay
    return CapturedBatch(
        split=split,
        start=int(selected.min()),
        stop=int(selected.max()) + 1,
        record_ids=record_ids,
        sites=tuple(captured),
        maximum_common_replay_error=common_error,
        maximum_projection_replay_error=projection_error,
        common_squared_error_sum=common_squared_error_sum,
        projection_squared_error_sum=projection_squared_error_sum,
        common_reference_squared_sum=common_reference_squared_sum,
        projection_reference_squared_sum=projection_reference_squared_sum,
        replay_elements=replay_elements,
        target_before_features_finite=features_finite,
    )


def iter_captures(
    model: H1LatentReasoner,
    cache: TokenCache,
    dataset: Mapping[str, Any],
    *,
    device: str,
    split: str,
    batch_size: int,
) -> Iterator[CapturedBatch]:
    if batch_size <= 0:
        raise ValueError("capture batch size must be positive")
    count = len(dataset["record_ids"])
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        yield capture_batch(
            model,
            cache,
            dataset,
            np.arange(start, stop, dtype=np.int64),
            device=device,
            split=split,
        )


def audit_full_replay_identity(
    model: H1LatentReasoner,
    cache: TokenCache,
    datasets: Mapping[str, Mapping[str, Any]],
    *,
    device: str,
    batch_size: int,
    tolerance: float,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay every registered record before a successor root is created."""

    if batch_size <= 0 or tolerance < 0.0:
        raise ValueError("invalid replay identity audit configuration")
    if set(datasets) != {"train", "heldout"}:
        raise ValueError("replay identity audit requires train and heldout")
    split_results: dict[str, Any] = {}
    global_common_max = 0.0
    global_projection_max = 0.0
    global_finite = True
    for split in ("train", "heldout"):
        dataset = datasets[split]
        record_count = len(dataset["record_ids"])
        common_batch_maxima: list[float] = []
        projection_batch_maxima: list[float] = []
        common_sse = projection_sse = 0.0
        common_reference = projection_reference = 0.0
        elements = completed = 0
        split_finite = True
        for batch in iter_captures(
            model,
            cache,
            dataset,
            device=device,
            split=split,
            batch_size=batch_size,
        ):
            common_batch_maxima.append(batch.maximum_common_replay_error)
            projection_batch_maxima.append(batch.maximum_projection_replay_error)
            common_sse += batch.common_squared_error_sum
            projection_sse += batch.projection_squared_error_sum
            common_reference += batch.common_reference_squared_sum
            projection_reference += batch.projection_reference_squared_sum
            elements += batch.replay_elements
            split_finite = bool(split_finite and batch.target_before_features_finite)
            completed += len(batch.record_ids)
            if progress is not None:
                progress(
                    "full_replay_identity_preflight",
                    {"split": split, "records_completed": completed, "records": record_count},
                )
        if completed != record_count or elements <= 0:
            raise RuntimeError(f"{split} replay coverage incomplete")
        common_values = np.asarray(common_batch_maxima, dtype=np.float64)
        projection_values = np.asarray(projection_batch_maxima, dtype=np.float64)
        split_common_max = float(common_values.max(initial=0.0))
        split_projection_max = float(projection_values.max(initial=0.0))
        split_results[split] = {
            "records": record_count,
            "batches": len(common_batch_maxima),
            "batch_size": batch_size,
            "common_maximum_absolute_error": split_common_max,
            "projection_maximum_absolute_error": split_projection_max,
            "common_rms_error": float((common_sse / elements) ** 0.5),
            "projection_rms_error": float((projection_sse / elements) ** 0.5),
            "common_relative_rms_error": float(
                (common_sse / max(common_reference, 1.0e-30)) ** 0.5
            ),
            "projection_relative_rms_error": float(
                (projection_sse / max(projection_reference, 1.0e-30)) ** 0.5
            ),
            "common_batch_maximum_percentiles": {
                "p50": float(np.quantile(common_values, 0.50)),
                "p95": float(np.quantile(common_values, 0.95)),
                "p99": float(np.quantile(common_values, 0.99)),
            },
            "projection_batch_maximum_percentiles": {
                "p50": float(np.quantile(projection_values, 0.50)),
                "p95": float(np.quantile(projection_values, 0.95)),
                "p99": float(np.quantile(projection_values, 0.99)),
            },
            "target_before_features_finite": split_finite,
            "passed": bool(
                split_finite
                and split_common_max <= tolerance
                and split_projection_max <= tolerance
            ),
        }
        global_common_max = max(global_common_max, split_common_max)
        global_projection_max = max(global_projection_max, split_projection_max)
        global_finite = bool(global_finite and split_finite)
    return {
        "microbatch_size": batch_size,
        "tolerance": tolerance,
        "splits": split_results,
        "maximum_common_absolute_error": global_common_max,
        "maximum_projection_absolute_error": global_projection_max,
        "target_before_features_finite": global_finite,
        "passed": bool(
            global_finite
            and global_common_max <= tolerance
            and global_projection_max <= tolerance
            and all(value["passed"] for value in split_results.values())
        ),
    }


__all__ = [
    "CapturedBatch",
    "CapturedSite",
    "audit_full_replay_identity",
    "capture_batch",
    "iter_captures",
]
