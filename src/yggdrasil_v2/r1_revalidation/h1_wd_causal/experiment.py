from __future__ import annotations

"""One-shot decision-causal write/delete mechanism screen.

The implementation deliberately keeps the old overlap-residual experiment out
of this module.  A frozen predecessor supplies route replay and scalar answer
margins; the only training targets are the positive common-off margin signal
and its site-wise VJP directions.  Family labels are used only for reporting.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from yggdrasil_v2.r1_revalidation.h1.cache import TokenCache, audit_token_cache
from yggdrasil_v2.r1_revalidation.h1.formal_data import (
    generate_registered_nonformal_packages,
    historical_fingerprint_registry,
    package_identity,
)
from yggdrasil_v2.r1_revalidation.h1.metrics import classification_metrics
from yggdrasil_v2.r1_revalidation.h1.model import H1LatentReasoner
from yggdrasil_v2.r1_revalidation.h1.train import (
    PreparedSplit,
    load_deployment_checkpoint,
    prepare_split,
)

from .contract import (
    CONTRACT_VERSION,
    DESIGN_DOCUMENT,
    EXECUTION_DOCUMENT,
    FORBIDDEN_WD_ROOT,
    OUTPUT_ROOT,
    PREDECESSOR_CHECKPOINT,
    SCHEMA_PREFIX,
    SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_PACKAGE_IDENTITY,
    SOURCE_ROOT,
    SOURCE_TOKEN_CACHE,
    SPEC,
    DecisionCausalSpec,
    contract_manifest,
)
from .core import (
    CausalSite,
    answer_margin,
    causal_transfer_targets,
    forward_route_replay,
)


ProgressCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class CausalTargetDataset:
    """Materialized, label-free transfer targets in prepared-split order.

    The tensors live on CPU between updates.  Training schedules therefore
    select records by row without recomputing the predecessor VJP and without
    retaining a predecessor autograd graph on the GPU.
    """

    split: str
    record_ids: tuple[str, ...]
    route_schedule: Tensor
    common: tuple[Tensor, ...]
    projection: tuple[Tensor, ...]
    transfer: tuple[Tensor, ...]
    vjp: tuple[Tensor, ...]
    requested_margin: Tensor
    positive_common_margin_drop: Tensor
    margins: Mapping[str, Tensor]
    site_metadata: tuple[Mapping[str, int], ...]
    stats: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.record_ids)

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_PREFIX}.target-dataset.v1",
            "split": self.split,
            "record_ids": self.record_ids,
            "route_schedule": self.route_schedule,
            "common": self.common,
            "projection": self.projection,
            "transfer": self.transfer,
            "vjp": self.vjp,
            "requested_margin": self.requested_margin,
            "positive_common_margin_drop": self.positive_common_margin_drop,
            "margins": dict(self.margins),
            "site_metadata": self.site_metadata,
            "stats": dict(self.stats),
            "uses_family_targets": False,
        }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _save_model(model: H1LatentReasoner, path: Path, *, stage: str) -> dict[str, Any]:
    state = {name: tensor.detach().cpu() for name, tensor in model.deployment_state().items()}
    payload = {
        "schema_version": f"{SCHEMA_PREFIX}.checkpoint.v1",
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "config": asdict(model.config),
        "state_dict": state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": path.name, "sha256": _sha256_path(path), "stage": stage}


def _lr_scale(update: int, updates: int, spec: DecisionCausalSpec) -> float:
    warmup = max(1, int(updates * spec.warmup_fraction))
    if update <= warmup:
        return update / warmup
    progress = (update - warmup) / max(1, updates - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return spec.minimum_learning_rate_scale + (1.0 - spec.minimum_learning_rate_scale) * cosine


def build_unlabeled_schedule(count: int, *, updates: int, batch_size: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if count <= 0 or updates <= 0 or batch_size <= 0:
        raise ValueError("schedule dimensions must be positive")
    generator = np.random.default_rng(seed)
    needed = updates * batch_size
    pieces: list[np.ndarray] = []
    available = 0
    while available < needed:
        pieces.append(generator.permutation(count).astype(np.int64, copy=False))
        available += count
    flat = np.concatenate(pieces)[:needed]
    schedule = flat.reshape(updates, batch_size)
    exposure = np.bincount(flat, minlength=count)
    return schedule, {
        "seed": int(seed),
        "count": int(count),
        "updates": int(updates),
        "batch_size": int(batch_size),
        "minimum_exposure": int(exposure.min()),
        "maximum_exposure": int(exposure.max()),
        "all_records_exposed": bool(np.all(exposure > 0)),
        "uses_family_targets": False,
        "sha256": hashlib.sha256(schedule.tobytes()).hexdigest().upper(),
    }


def _batch(cache: TokenCache, prepared: PreparedSplit, indices: np.ndarray, *, device: str) -> tuple[Tensor, Tensor, Tensor]:
    hidden, mask = cache.batch(prepared.cache_indices[indices], device=device)
    targets = torch.from_numpy(prepared.targets[indices].copy()).to(device)
    return hidden, mask, targets


def _optimizer(parameters: Sequence[Tensor], learning_rate: float, spec: DecisionCausalSpec) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=spec.weight_decay)


def _freeze(model: H1LatentReasoner, role: str) -> tuple[Tensor, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[Tensor] = []
    if role in {"projection", "joint"}:
        for layer in model.layers:
            for module in (layer.routed_feature_trunk, *layer.routed_projections):
                selected.extend(module.parameters())
    if role in {"common", "joint"}:
        for layer in model.layers:
            selected.extend(layer.shared_ffn.parameters())
    if role == "joint":
        selected.extend(model.answer_head.parameters())
    if role not in {"projection", "common", "joint"}:
        raise ValueError(f"unknown causal training role: {role}")
    for parameter in selected:
        parameter.requires_grad_(True)
    return tuple(selected)


def _site_projection(layer: Any, state: Tensor, route: Tensor) -> Tensor:
    features = layer.routed_feature_trunk(state)
    if len(layer.routed_projections) == 1:
        return layer.routed_projections[0](features)
    output = torch.zeros_like(state)
    for expert_index, projection in enumerate(layer.routed_projections):
        selected = torch.nonzero(route == expert_index, as_tuple=False).flatten()
        if selected.numel():
            output = output.index_copy(0, selected, projection(features.index_select(0, selected)))
    return output


def _target_for_arm(source: H1LatentReasoner, hidden: Tensor, mask: Tensor, answers: Tensor, *, sign: float, spec: DecisionCausalSpec) -> dict[str, Any]:
    # VJP needs autograd even though the predecessor itself is never updated.
    with torch.enable_grad():
        for parameter in source.parameters():
            parameter.requires_grad_(True)
        raw = causal_transfer_targets(
            source,
            hidden,
            mask,
            answers,
            request_fraction=spec.margin_transfer_fraction,
            site_transfer_cap=spec.maximum_site_transfer_fraction_of_common_norm,
        )
    captures = raw["full"]["captures"]
    transfers = raw["transfers"]
    if captures is None or transfers is None:
        raise RuntimeError("causal target did not capture predecessor sites")
    base_common = tuple(site.common.detach() for site in captures)
    base_projection = tuple(site.projection.detach() for site in captures)
    signed_transfers = tuple((float(sign) * transfer).detach() for transfer in transfers)
    return {
        "route_schedule": raw["route_schedule"].detach(),
        "common": base_common,
        "projection": base_projection,
        "transfer": signed_transfers,
        "projection_target": tuple(base + delta for base, delta in zip(base_projection, signed_transfers, strict=True)),
        "common_target": tuple(base - delta for base, delta in zip(base_common, signed_transfers, strict=True)),
        "margins": {key: value.detach() for key, value in raw["margins"].items()},
        "positive_common_margin_drop": raw["positive_common_margin_drop"].detach(),
        "requested_margin": raw["requested_margin"].detach(),
        "diagnostics": raw["diagnostics"],
        "uses_family_targets": False,
        "arm_sign": float(sign),
    }


def _fit_projection(student: H1LatentReasoner, hidden: Tensor, mask: Tensor, target: Mapping[str, Any]) -> Tensor:
    replay = forward_route_replay(student, hidden, mask, route_schedule=target["route_schedule"], capture=True, return_trajectory=False)
    captures = replay["captures"]
    if captures is None:
        raise RuntimeError("student replay did not capture sites")
    squared_error = torch.zeros((), device=hidden.device)
    transfer_energy = torch.zeros((), device=hidden.device)
    base_energy = torch.zeros((), device=hidden.device)
    for actual, expected, base, delta in zip(
        captures,
        target["projection_target"],
        target["projection"],
        target["transfer"],
        strict=True,
    ):
        squared_error = squared_error + (actual.projection - expected).square().sum()
        transfer_energy = transfer_energy + delta.detach().square().sum()
        base_energy = base_energy + base.detach().square().sum()
    denominator = torch.maximum(
        transfer_energy,
        base_energy * 1.0e-8,
    ).clamp_min(1.0e-12)
    return squared_error / denominator


def _fit_common(student: H1LatentReasoner, hidden: Tensor, mask: Tensor, target: Mapping[str, Any]) -> Tensor:
    replay = forward_route_replay(student, hidden, mask, route_schedule=target["route_schedule"], capture=True, return_trajectory=False)
    captures = replay["captures"]
    if captures is None:
        raise RuntimeError("student replay did not capture sites")
    squared_error = torch.zeros((), device=hidden.device)
    transfer_energy = torch.zeros((), device=hidden.device)
    base_energy = torch.zeros((), device=hidden.device)
    for actual, expected, base, delta in zip(
        captures,
        target["common_target"],
        target["common"],
        target["transfer"],
        strict=True,
    ):
        squared_error = squared_error + (actual.common - expected).square().sum()
        transfer_energy = transfer_energy + delta.detach().square().sum()
        base_energy = base_energy + base.detach().square().sum()
    denominator = torch.maximum(
        transfer_energy,
        base_energy * 1.0e-8,
    ).clamp_min(1.0e-12)
    return squared_error / denominator


def build_causal_target_dataset(
    source: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    spec: DecisionCausalSpec,
    *,
    batch_size: int | None = None,
    max_records: int | None = None,
    progress: ProgressCallback | None = None,
) -> CausalTargetDataset:
    """Materialize the causal target dataset with the frozen microbatch size."""

    batch_size = batch_size or spec.target_microbatch_size
    if batch_size > spec.target_microbatch_size:
        raise ValueError("causal target construction exceeds target_microbatch_size")
    total = len(prepared.ids) if max_records is None else min(len(prepared.ids), max_records)
    if total <= 0:
        raise ValueError("causal target dataset must contain at least one record")
    if spec.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    routes: list[Tensor] = []
    common_chunks: list[list[Tensor]] | None = None
    projection_chunks: list[list[Tensor]] | None = None
    transfer_chunks: list[list[Tensor]] | None = None
    vjp_chunks: list[list[Tensor]] | None = None
    requested_chunks: list[Tensor] = []
    positive_drop_chunks: list[Tensor] = []
    margin_chunks: dict[str, list[Tensor]] = {}
    site_metadata: tuple[Mapping[str, int], ...] | None = None
    finite = positive = target_values = 0
    requested_sum = realized_sum = 0.0
    finite_checks = {
        "common": True,
        "projection": True,
        "transfer": True,
        "vjp": True,
        "requested_margin": True,
        "positive_common_margin_drop": True,
        "intervention_margins": True,
    }

    for start in range(0, total, batch_size):
        indices = np.arange(start, min(start + batch_size, total), dtype=np.int64)
        hidden, mask, answers = _batch(cache, prepared, indices, device=spec.device)
        target = _target_for_arm(source, hidden, mask, answers, sign=1.0, spec=spec)
        diagnostics = target["diagnostics"]
        if common_chunks is None:
            count = len(target["common"])
            common_chunks = [[] for _ in range(count)]
            projection_chunks = [[] for _ in range(count)]
            transfer_chunks = [[] for _ in range(count)]
            vjp_chunks = [[] for _ in range(count)]
            site_metadata = tuple(
                {
                    "layer_index": int(item["layer_index"]),
                    "step_index": int(item["step_index"]),
                }
                for item in diagnostics
            )
        assert projection_chunks is not None
        assert transfer_chunks is not None
        assert vjp_chunks is not None
        if len(diagnostics) != len(common_chunks):
            raise RuntimeError("causal target site count changed across microbatches")

        routes.append(target["route_schedule"].detach().cpu())
        for site_index in range(len(common_chunks)):
            common_chunks[site_index].append(target["common"][site_index].detach().cpu())
            projection_chunks[site_index].append(target["projection"][site_index].detach().cpu())
            transfer = target["transfer"][site_index].detach()
            vjp = diagnostics[site_index]["vjp"].detach()
            transfer_chunks[site_index].append(transfer.cpu())
            vjp_chunks[site_index].append(vjp.cpu())
            finite_checks["common"] = finite_checks["common"] and bool(
                torch.isfinite(target["common"][site_index]).all().cpu()
            )
            finite_checks["projection"] = finite_checks["projection"] and bool(
                torch.isfinite(target["projection"][site_index]).all().cpu()
            )
            finite_checks["transfer"] = finite_checks["transfer"] and bool(
                torch.isfinite(transfer).all().cpu()
            )
            finite_checks["vjp"] = finite_checks["vjp"] and bool(
                torch.isfinite(vjp).all().cpu()
            )
            realized_sum += float((vjp * transfer).sum().cpu())
        requested = target["requested_margin"].detach()
        finite += int(torch.isfinite(requested).sum().cpu())
        finite_checks["requested_margin"] = finite_checks[
            "requested_margin"
        ] and bool(torch.isfinite(requested).all().cpu())
        finite_checks["positive_common_margin_drop"] = finite_checks[
            "positive_common_margin_drop"
        ] and bool(
            torch.isfinite(target["positive_common_margin_drop"]).all().cpu()
        )
        positive += int((requested > 0).sum().cpu())
        target_values += int(requested.numel())
        requested_sum += float(requested.sum().cpu())
        requested_chunks.append(requested.cpu())
        positive_drop_chunks.append(target["positive_common_margin_drop"].detach().cpu())
        for name, values in target["margins"].items():
            finite_checks["intervention_margins"] = finite_checks[
                "intervention_margins"
            ] and bool(torch.isfinite(values).all().cpu())
            margin_chunks.setdefault(name, []).append(values.detach().cpu())
        if progress is not None:
            progress(
                "target",
                {
                    "split": prepared.split,
                    "records": int(min(start + len(indices), total)),
                    "total": int(total),
                },
            )
        del hidden, mask, answers, target, diagnostics

    assert common_chunks is not None
    assert projection_chunks is not None
    assert transfer_chunks is not None
    assert vjp_chunks is not None
    assert site_metadata is not None
    peak = torch.cuda.max_memory_allocated() if spec.device.startswith("cuda") else 0
    stats = {
        "split": prepared.split,
        "records": int(total),
        "target_values": int(target_values),
        "target_finite": bool(
            finite == target_values and all(finite_checks.values())
        ),
        "finite_checks": finite_checks,
        "positive_fraction": float(positive / max(target_values, 1)),
        "requested_margin_sum": requested_sum,
        "realized_margin_sum": realized_sum,
        "realized_request_ratio": float(realized_sum / max(requested_sum, 1.0e-12)),
        "sites_per_record": len(site_metadata),
        "site_records": int(total * len(site_metadata)),
        "batch_size": int(batch_size),
        "target_microbatch_size": int(spec.target_microbatch_size),
        "gpu_smoke_gb": float(peak / (1024**3)),
        "uses_family_targets": False,
    }
    return CausalTargetDataset(
        split=prepared.split,
        record_ids=tuple(prepared.ids[:total]),
        route_schedule=torch.cat(routes, dim=0),
        common=tuple(torch.cat(chunks, dim=0) for chunks in common_chunks),
        projection=tuple(torch.cat(chunks, dim=0) for chunks in projection_chunks),
        transfer=tuple(torch.cat(chunks, dim=0) for chunks in transfer_chunks),
        vjp=tuple(torch.cat(chunks, dim=0) for chunks in vjp_chunks),
        requested_margin=torch.cat(requested_chunks, dim=0),
        positive_common_margin_drop=torch.cat(positive_drop_chunks, dim=0),
        margins={name: torch.cat(chunks, dim=0) for name, chunks in margin_chunks.items()},
        site_metadata=site_metadata,
        stats=stats,
    )


def _save_target_datasets(
    datasets: Mapping[str, CausalTargetDataset], path: Path
) -> dict[str, Any]:
    payload = {
        "schema_version": f"{SCHEMA_PREFIX}.target-datasets.v1",
        "contract_version": CONTRACT_VERSION,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "datasets": {name: dataset.as_payload() for name, dataset in datasets.items()},
        "uses_family_targets": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": _sha256_path(path),
        "bytes": int(path.stat().st_size),
        "splits": sorted(datasets),
        "uses_family_targets": False,
    }


def _target_dataset_batch(
    dataset: CausalTargetDataset,
    indices: np.ndarray,
    *,
    device: str,
    sign: float,
) -> dict[str, Any]:
    selected = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long)

    def take(tensor: Tensor) -> Tensor:
        return tensor.index_select(0, selected).to(device)

    common = tuple(take(value) for value in dataset.common)
    projection = tuple(take(value) for value in dataset.projection)
    transfer = tuple(float(sign) * take(value) for value in dataset.transfer)
    return {
        "route_schedule": take(dataset.route_schedule),
        "common": common,
        "projection": projection,
        "transfer": transfer,
        "vjp": tuple(take(value) for value in dataset.vjp),
        "projection_target": tuple(
            base + delta for base, delta in zip(projection, transfer, strict=True)
        ),
        "common_target": tuple(
            base - delta for base, delta in zip(common, transfer, strict=True)
        ),
        "requested_margin": take(dataset.requested_margin),
        "positive_common_margin_drop": take(dataset.positive_common_margin_drop),
        "uses_family_targets": False,
        "arm_sign": float(sign),
    }


def _validate_target_dataset(
    dataset: CausalTargetDataset, prepared: PreparedSplit
) -> None:
    if dataset.split != prepared.split:
        raise ValueError("causal target dataset split mismatch")
    if dataset.record_ids != tuple(prepared.ids):
        raise ValueError("causal target dataset record order mismatch")


def train_write_stage(
    student: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    targets: CausalTargetDataset,
    schedule: np.ndarray,
    spec: DecisionCausalSpec,
    *,
    sign: float = 1.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    student.to(spec.device).train()
    _validate_target_dataset(targets, prepared)
    parameters = _freeze(student, "projection")
    optimizer = _optimizer(parameters, spec.write_learning_rate, spec)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update, indices in enumerate(schedule, start=1):
        hidden, mask, answers = _batch(cache, prepared, indices, device=spec.device)
        target = _target_dataset_batch(targets, indices, device=spec.device, sign=sign)
        optimizer.zero_grad(set_to_none=True)
        loss = _fit_projection(student, hidden, mask, target)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, spec.gradient_clip).detach().cpu())
        scale = _lr_scale(update, len(schedule), spec)
        for group in optimizer.param_groups:
            group["lr"] = spec.write_learning_rate * scale
        optimizer.step()
        if update % spec.evaluation_interval == 0 or update == len(schedule):
            row = {"stage": "write", "arm_sign": sign, "update": update, "updates": len(schedule), "normalized_mse": float(loss.detach().cpu()), "gradient_norm_before_clip": grad_norm, "learning_rate": spec.write_learning_rate * scale, "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            if progress is not None:
                progress("write", row)
        del hidden, mask, answers, target, loss
    return {"updates": len(schedule), "wall_seconds": time.perf_counter() - started, "history": history, "trainable_parameters": int(sum(p.numel() for p in parameters)), "arm_sign": sign, "uses_family_targets": False}


def evaluate_write_fit(
    student: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    targets: CausalTargetDataset,
    spec: DecisionCausalSpec,
    *,
    sign: float = 1.0,
    batch_size: int | None = None,
) -> dict[str, Any]:
    student.to(spec.device).eval()
    _validate_target_dataset(targets, prepared)
    batch_size = batch_size or spec.target_microbatch_size
    error = target_energy = base_energy = transfer_energy = 0.0
    positive = finite = total = 0
    requested = realized = 0.0
    for start in range(0, len(prepared.ids), batch_size):
        indices = np.arange(start, min(start + batch_size, len(prepared.ids)), dtype=np.int64)
        hidden, mask, answers = _batch(cache, prepared, indices, device=spec.device)
        target = _target_dataset_batch(targets, indices, device=spec.device, sign=sign)
        replay = forward_route_replay(student, hidden, mask, route_schedule=target["route_schedule"], capture=True, return_trajectory=False)
        captures = replay["captures"]
        if captures is None:
            raise RuntimeError("write fit replay did not capture sites")
        for site, expected, base, delta in zip(
            captures,
            target["projection_target"],
            target["projection"],
            target["transfer"],
            strict=True,
        ):
            error += float((site.projection - expected).square().sum().detach().cpu())
            target_energy += float(expected.square().sum().detach().cpu())
            base_energy += float(base.square().sum().detach().cpu())
            transfer_energy += float(delta.square().sum().detach().cpu())
        margin = target["requested_margin"].reshape(-1)
        finite += int(torch.isfinite(margin).sum().cpu())
        positive += int((margin > 0).sum().cpu())
        total += int(margin.numel())
        requested += float(margin.sum().cpu())
        for vjp, delta in zip(target["vjp"], target["transfer"], strict=True):
            realized += float((vjp * delta * float(sign)).sum().cpu())
        del hidden, mask, answers, target, replay
    return {
        "split": prepared.split,
        "arm_sign": sign,
        "normalized_mse": error / max(target_energy, 1.0e-12),
        "transfer_normalized_mse": error / max(transfer_energy, base_energy * 1.0e-8, 1.0e-12),
        "target_energy": target_energy,
        "transfer_energy": transfer_energy,
        "positive_fraction": positive / max(total, 1),
        "target_finite": bool(finite == total),
        "requested_margin_sum": requested,
        "realized_request_ratio": realized / max(requested, 1.0e-12),
        "uses_family_targets": False,
    }


def train_delete_stage(
    student: H1LatentReasoner,
    written: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    targets: CausalTargetDataset,
    schedule: np.ndarray,
    spec: DecisionCausalSpec,
    *,
    sign: float = 1.0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    student.to(spec.device).train()
    written.to(spec.device).eval()
    _validate_target_dataset(targets, prepared)
    parameters = _freeze(student, "common")
    optimizer = _optimizer(parameters, spec.delete_learning_rate, spec)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update, indices in enumerate(schedule, start=1):
        hidden, mask, answers = _batch(cache, prepared, indices, device=spec.device)
        target = _target_dataset_batch(targets, indices, device=spec.device, sign=sign)
        # Fit the actual written projection on the frozen predecessor attention
        # states; this makes D the complement of what W really learned.
        with torch.no_grad():
            replay_written = forward_route_replay(written, hidden, mask, route_schedule=target["route_schedule"], capture=True, return_trajectory=False)
        written_sites = replay_written["captures"]
        if written_sites is None:
            raise RuntimeError("written replay did not capture sites")
        actual_transfers = tuple(
            (actual.projection - base_projection).detach()
            for actual, base_projection in zip(
                written_sites, target["projection"], strict=True
            )
        )
        common_targets = tuple(
            base - delta
            for base, delta in zip(target["common"], actual_transfers, strict=True)
        )
        target = dict(target)
        target["transfer"] = actual_transfers
        target["common_target"] = common_targets
        optimizer.zero_grad(set_to_none=True)
        loss = _fit_common(student, hidden, mask, target)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, spec.gradient_clip).detach().cpu())
        scale = _lr_scale(update, len(schedule), spec)
        for group in optimizer.param_groups:
            group["lr"] = spec.delete_learning_rate * scale
        optimizer.step()
        if update % spec.evaluation_interval == 0 or update == len(schedule):
            row = {"stage": "delete", "arm_sign": sign, "update": update, "updates": len(schedule), "normalized_mse": float(loss.detach().cpu()), "gradient_norm_before_clip": grad_norm, "learning_rate": spec.delete_learning_rate * scale, "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            if progress is not None:
                progress("delete", row)
        del hidden, mask, answers, target, replay_written, loss
    return {"updates": len(schedule), "wall_seconds": time.perf_counter() - started, "history": history, "trainable_parameters": int(sum(p.numel() for p in parameters)), "arm_sign": sign, "uses_family_targets": False}


def evaluate_delete_fit(
    student: H1LatentReasoner,
    written: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    targets: CausalTargetDataset,
    spec: DecisionCausalSpec,
    *,
    sign: float = 1.0,
    batch_size: int | None = None,
) -> dict[str, Any]:
    student.to(spec.device).eval()
    written.to(spec.device).eval()
    _validate_target_dataset(targets, prepared)
    batch_size = batch_size or spec.target_microbatch_size
    common_error = common_energy = transfer_energy = total_error = total_energy = old_common_energy = new_common_energy = 0.0
    for start in range(0, len(prepared.ids), batch_size):
        indices = np.arange(start, min(start + batch_size, len(prepared.ids)), dtype=np.int64)
        hidden, mask, answers = _batch(cache, prepared, indices, device=spec.device)
        target = _target_dataset_batch(targets, indices, device=spec.device, sign=sign)
        with torch.no_grad():
            written_replay = forward_route_replay(written, hidden, mask, route_schedule=target["route_schedule"], capture=True, return_trajectory=False)
            student_replay = forward_route_replay(student, hidden, mask, route_schedule=target["route_schedule"], capture=True, return_trajectory=False)
        written_sites, student_sites = written_replay["captures"], student_replay["captures"]
        if written_sites is None or student_sites is None:
            raise RuntimeError("delete fit replay did not capture sites")
        for base_common, old_projection, written_site, student_site in zip(
            target["common"],
            target["projection"],
            written_sites,
            student_sites,
            strict=True,
        ):
            actual_transfer = written_site.projection - old_projection
            common_target = base_common - actual_transfer
            actual = student_site.common
            old_total = base_common + old_projection
            new_total = actual + written_site.projection
            common_error += float((actual - common_target).square().sum().detach().cpu())
            common_energy += float(common_target.square().sum().detach().cpu())
            transfer_energy += float(actual_transfer.square().sum().detach().cpu())
            total_error += float((new_total - old_total).square().sum().detach().cpu())
            total_energy += float(old_total.square().sum().detach().cpu())
            old_common_energy += float(base_common.square().sum().detach().cpu())
            new_common_energy += float(actual.square().sum().detach().cpu())
        del hidden, mask, answers, target, written_replay, student_replay
    return {
        "split": prepared.split,
        "arm_sign": sign,
        "common_normalized_mse": common_error / max(common_energy, 1.0e-12),
        "common_transfer_normalized_mse": common_error / max(transfer_energy, old_common_energy * 1.0e-8, 1.0e-12),
        "total_transition_normalized_mse": total_error / max(total_energy, 1.0e-12),
        "common_residual_energy": new_common_energy / max(old_common_energy, 1.0e-12),
        "actual_transfer_energy": transfer_energy,
        "uses_family_targets": False,
    }


def _allocation_lock_loss(
    model: H1LatentReasoner,
    hidden: Tensor,
    mask: Tensor,
    target: Mapping[str, Any],
) -> Tensor:
    """Keep the W/D component transfer on fixed source routes during J.

    This is a two-path allocation constraint derived from the causal dataset,
    not a predecessor-logit target and not route supervision.
    """

    replay = forward_route_replay(
        model,
        hidden,
        mask,
        route_schedule=target["route_schedule"],
        capture=True,
        return_trajectory=False,
    )
    captures = replay["captures"]
    if captures is None:
        raise RuntimeError("allocation lock replay did not capture sites")
    squared_error = torch.zeros((), device=hidden.device)
    transfer_energy = torch.zeros((), device=hidden.device)
    base_energy = torch.zeros((), device=hidden.device)
    for site, base_common, base_projection, delta in zip(
        captures,
        target["common"],
        target["projection"],
        target["transfer"],
        strict=True,
    ):
        squared_error = squared_error + (
            site.projection - (base_projection + delta)
        ).square().sum()
        squared_error = squared_error + (
            site.common - (base_common - delta)
        ).square().sum()
        transfer_energy = transfer_energy + 2.0 * delta.detach().square().sum()
        base_energy = base_energy + base_common.detach().square().sum()
        base_energy = base_energy + base_projection.detach().square().sum()
    denominator = torch.maximum(
        transfer_energy,
        base_energy * 1.0e-8,
    ).clamp_min(1.0e-12)
    return squared_error / denominator


def train_joint_pair(
    causal: H1LatentReasoner,
    control: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    targets: CausalTargetDataset,
    schedule: np.ndarray,
    spec: DecisionCausalSpec,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    _validate_target_dataset(targets, prepared)
    arms = {"causal": causal.to(spec.device), "control": control.to(spec.device)}
    parameters = {name: _freeze(model, "joint") for name, model in arms.items()}
    optimizers = {name: _optimizer(parameters[name], spec.joint_learning_rate, spec) for name in arms}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update, indices in enumerate(schedule, start=1):
        hidden, mask, answers = _batch(cache, prepared, indices, device=spec.device)
        rows: dict[str, Any] = {}
        for name, model in arms.items():
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            output = model(hidden, mask, return_trajectory=False)
            logits = output["logits"]
            if logits is None:
                raise RuntimeError("joint forward omitted logits")
            answer_loss = F.cross_entropy(logits, answers)
            target = _target_dataset_batch(
                targets,
                indices,
                device=spec.device,
                sign=1.0 if name == "causal" else -1.0,
            )
            allocation_lock = _allocation_lock_loss(model, hidden, mask, target)
            loss = answer_loss + spec.allocation_lock_weight * allocation_lock
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters[name], spec.gradient_clip).detach().cpu())
            scale = _lr_scale(update, len(schedule), spec)
            for group in optimizer.param_groups:
                group["lr"] = spec.joint_learning_rate * scale
            optimizer.step()
            rows[name] = {"answer_loss": float(answer_loss.detach().cpu()), "allocation_lock": float(allocation_lock.detach().cpu()), "teacher_kl": 0.0, "total_loss": float(loss.detach().cpu()), "batch_accuracy": float((logits.argmax(-1) == answers).float().mean().detach().cpu()), "gradient_norm_before_clip": grad_norm}
        if update % spec.evaluation_interval == 0 or update == len(schedule):
            row = {"stage": "joint", "update": update, "updates": len(schedule), "learning_rate": spec.joint_learning_rate * _lr_scale(update, len(schedule), spec), "arms": rows, "elapsed_seconds": time.perf_counter() - started}
            history.append(row)
            if progress is not None:
                progress("joint", row)
        del hidden, mask, answers
    return {
        "updates": len(schedule),
        "wall_seconds": time.perf_counter() - started,
        "history": history,
        "uses_family_targets": False,
        "teacher_kl_used": False,
        "teacher_model_used": False,
        "route_supervision_used": False,
        "allocation_lock": "two_path_causal_target_lock",
        "objective_components": (
            "ordinary_answer_cross_entropy",
            "two_path_causal_target_lock",
        ),
        "answer_route_mode": "model_free_route",
        "allocation_lock_route_mode": "frozen_source_route_replay",
        "teacher_kl_weight": float(spec.teacher_kl_weight),
        "trainable_roles": ("shared_ffn", "routed_projection", "answer_head"),
        "frozen_roles": ("router", "attention", "upstream_boundary"),
    }


def evaluate_answer(model: H1LatentReasoner, cache: TokenCache, prepared: PreparedSplit, *, device: str, intervention: str = "none", batch_size: int = 64) -> dict[str, Any]:
    model.to(device).eval()
    predictions: list[int] = []
    routes: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(prepared.ids), batch_size):
            stop = min(start + batch_size, len(prepared.ids))
            hidden, mask = cache.batch(prepared.cache_indices[start:stop], device=device)
            output = forward_route_replay(model, hidden, mask, intervention=intervention, capture=False, return_trajectory=False)
            logits, assignments = output["logits"], output["route_assignments"]
            if logits is None or assignments is None:
                raise RuntimeError("answer evaluation omitted logits/routes")
            predictions.extend(int(value) for value in logits.argmax(-1).cpu().tolist())
            routes.extend(int(value) for value in (assignments.float().mean(dim=(1, 2)) >= 0.5).long().cpu().tolist())
    metrics = classification_metrics(prepared.ids, predictions, prepared.targets.tolist(), prepared.families)
    return {"split": prepared.split, "intervention": intervention, "ids": list(prepared.ids), "targets": prepared.targets.tolist(), "families": list(prepared.families), "predictions": predictions, "route_predictions": routes, "answer": metrics}


def evaluate_intervention_suite(model: H1LatentReasoner, cache: TokenCache, prepared: PreparedSplit, *, device: str) -> dict[str, Any]:
    names = ("none", "disable_common", "disable_projection", "disable_both")
    full = {name: evaluate_answer(model, cache, prepared, device=device, intervention=name) for name in names}
    normal = full["none"]["answer"]["accuracy"]
    compact = {}
    for name, value in full.items():
        answer = value["answer"]
        compact[name] = {"accuracy": answer["accuracy"], "macro_accuracy": answer["macro_accuracy"], "by_family": answer["by_family"], "prediction_sha256": answer["prediction_sha256"], "drop_from_normal": float(normal - answer["accuracy"])}
    return {"full": full, "compact": compact}


def compare_free_rollout(predecessor: H1LatentReasoner, successor: H1LatentReasoner, cache: TokenCache, prepared: PreparedSplit, *, device: str, batch_size: int = 64) -> dict[str, Any]:
    predecessor.to(device).eval()
    successor.to(device).eval()
    agreement = count = 0
    logit_error = logit_energy = route_agreement = route_count = 0.0
    with torch.inference_mode():
        for start in range(0, len(prepared.ids), batch_size):
            stop = min(start + batch_size, len(prepared.ids))
            hidden, mask = cache.batch(prepared.cache_indices[start:stop], device=device)
            before = forward_route_replay(predecessor, hidden, mask, capture=False, return_trajectory=False)
            after = forward_route_replay(successor, hidden, mask, capture=False, return_trajectory=False)
            if before["logits"] is None or after["logits"] is None:
                raise RuntimeError("free rollout omitted logits")
            agreement += int((before["logits"].argmax(-1) == after["logits"].argmax(-1)).sum().cpu())
            count += int(before["logits"].shape[0])
            logit_error += float((before["logits"] - after["logits"]).square().sum().cpu())
            logit_energy += float(before["logits"].square().sum().cpu())
            if before["route_assignments"] is not None and after["route_assignments"] is not None:
                route_agreement += float((before["route_assignments"] == after["route_assignments"]).float().sum().cpu())
                route_count += float(before["route_assignments"].numel())
    free_route_agreement = route_agreement / max(route_count, 1.0)
    return {"split": prepared.split, "prediction_agreement": agreement / max(count, 1), "free_route_agreement": free_route_agreement, "route_replay_agreement": free_route_agreement, "logit_relative_mse": logit_error / max(logit_energy, 1.0e-12)}


def paired_bootstrap_comparison(a: Sequence[float], b: Sequence[float], *, seed: int, bootstrap: int = 2000, families: Sequence[str] | None = None) -> dict[str, Any]:
    first, second = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or first.size == 0:
        raise ValueError("paired bootstrap inputs must be equal non-empty vectors")
    diff = second - first
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, first.size, size=(bootstrap, first.size))
    samples = diff[indices].mean(axis=1)
    result = {"gain": float(diff.mean()), "ci_lower": float(np.quantile(samples, 0.025)), "ci_upper": float(np.quantile(samples, 0.975)), "bootstrap": int(bootstrap), "count": int(first.size)}
    if families is not None:
        family_regression = {}
        for family in sorted(set(families)):
            selected = np.asarray([value == family for value in families])
            family_regression[family] = float(diff[selected].mean()) if selected.any() else 0.0
        result["family_gain"] = family_regression
    return result


def route_replay_shapley(
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    *,
    device: str,
    route_schedule: Tensor | None = None,
    batch_size: int = 64,
    bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    if route_schedule is not None and tuple(route_schedule.shape[:1]) != (
        len(prepared.ids),
    ):
        raise ValueError("Shapley route schedule record count mismatch")
    values: dict[str, list[float]] = {"none": [], "common": [], "projection": [], "both": []}
    for start in range(0, len(prepared.ids), batch_size):
        stop = min(start + batch_size, len(prepared.ids))
        hidden, mask = cache.batch(prepared.cache_indices[start:stop], device=device)
        with torch.inference_mode():
            routes = (
                route_schedule[start:stop].to(device)
                if route_schedule is not None
                else None
            )
            normal = forward_route_replay(model, hidden, mask, route_schedule=routes, capture=False, return_trajectory=False)
            routes = normal["route_schedule"]
            if routes is None:
                raise RuntimeError("route replay did not produce routes")
            outputs = {
                "none": normal,
                "common": forward_route_replay(model, hidden, mask, route_schedule=routes, disable_projection=True, capture=False, return_trajectory=False),
                "projection": forward_route_replay(model, hidden, mask, route_schedule=routes, disable_common=True, capture=False, return_trajectory=False),
                "both": forward_route_replay(model, hidden, mask, route_schedule=routes, disable_common=True, disable_projection=True, capture=False, return_trajectory=False),
            }
            for name, output in outputs.items():
                if output["logits"] is None:
                    raise RuntimeError("Shapley replay omitted logits")
                values[name].extend((output["logits"].argmax(-1) == torch.from_numpy(prepared.targets[start:stop]).to(device)).float().cpu().tolist())
    none, common, projection, both = (np.asarray(values[name], dtype=np.float64) for name in ("none", "common", "projection", "both"))
    phi_common = 0.5 * ((common - both) + (none - projection))
    phi_projection = 0.5 * ((projection - both) + (none - common))
    denominator = np.abs(phi_common) + np.abs(phi_projection) + 1.0e-12
    share_projection = np.abs(phi_projection) / denominator
    share_common = np.abs(phi_common) / denominator
    share_ci = paired_bootstrap_comparison(np.zeros_like(share_projection), share_projection, seed=seed, bootstrap=bootstrap)
    return {"count": int(none.size), "projection_share": float(share_projection.mean()), "common_share": float(share_common.mean()), "projection_share_ci_lower": share_ci["ci_lower"], "projection_share_ci_upper": share_ci["ci_upper"], "projection_share_values": share_projection.tolist(), "common_share_values": share_common.tolist(), "phi_projection": phi_projection.tolist(), "phi_common": phi_common.tolist(), "route_replay": True, "route_schedule": "frozen_source" if route_schedule is not None else "model_normal", "uses_family_targets": False}


def _compact(value: Mapping[str, Any]) -> dict[str, Any]:
    answer = value["answer"]
    return {"split": value["split"], "intervention": value["intervention"], "answer": {key: item for key, item in answer.items() if key != "correct_vector"}, "predictions": value["predictions"], "families": value["families"]}


def audit_route_replay_equivalence(
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    *,
    device: str,
    records: int = 4,
) -> dict[str, Any]:
    stop = min(records, len(prepared.ids))
    hidden, mask = cache.batch(prepared.cache_indices[:stop], device=device)
    with torch.inference_mode():
        ordinary = model(hidden, mask, return_trajectory=True)
        replay = forward_route_replay(
            model,
            hidden,
            mask,
            route_schedule=ordinary["route_assignments"],
            capture=False,
            return_trajectory=True,
        )
    logits_equal = bool(torch.equal(ordinary["logits"], replay["logits"]))
    routes_equal = bool(
        torch.equal(ordinary["route_assignments"], replay["route_assignments"])
    )
    trajectory_equal = bool(
        torch.equal(ordinary["trajectory"], replay["trajectory"])
    )
    return {
        "records": int(stop),
        "logits_bitwise_equal": logits_equal,
        "routes_bitwise_equal": routes_equal,
        "trajectory_bitwise_equal": trajectory_equal,
        "maximum_logit_absolute_error": float(
            (ordinary["logits"] - replay["logits"]).abs().max().cpu()
        ),
        "passed": bool(logits_equal and routes_equal and trajectory_equal),
    }


def _prepare_context(repo_root: Path) -> dict[str, Any]:
    source_checkpoint = repo_root / SOURCE_CHECKPOINT
    source_cache = repo_root / SOURCE_TOKEN_CACHE
    output_root = repo_root / OUTPUT_ROOT
    forbidden_wd_root = repo_root / FORBIDDEN_WD_ROOT
    source_outside_forbidden_root = not source_checkpoint.resolve().is_relative_to(
        forbidden_wd_root.resolve()
    ) and not source_cache.resolve().is_relative_to(forbidden_wd_root.resolve())
    checks = {
        "source_root_exists": (repo_root / SOURCE_ROOT).is_dir(),
        "source_checkpoint_exists": source_checkpoint.is_file(),
        "source_cache_exists": source_cache.is_dir(),
        "output_root_absent": not output_root.exists(),
        "design_exists": (repo_root / DESIGN_DOCUMENT).is_file(),
        "execution_exists": (repo_root / EXECUTION_DOCUMENT).is_file(),
        "cuda_available": torch.cuda.is_available(),
        "source_outside_forbidden_wd_root": source_outside_forbidden_root,
        "forbidden_wd_checkpoint_not_used": source_outside_forbidden_root,
    }
    if not all(checks.values()):
        raise RuntimeError(f"decision-causal preflight failed: {checks}")
    checkpoint_sha256 = _sha256_path(source_checkpoint)
    if checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("decision-causal source checkpoint hash drift")
    registry = historical_fingerprint_registry(repo_root)
    packages = generate_registered_nonformal_packages(forbidden_semantic_fingerprints=registry["semantic"], forbidden_source_fingerprints=registry["source"])
    identity = package_identity(packages["screen"])
    if identity != SOURCE_PACKAGE_IDENTITY:
        raise RuntimeError("decision-causal source package identity drift")
    cache_audit = audit_token_cache(packages["screen"].records, source_cache)
    if cache_audit.get("passed") is not True:
        raise RuntimeError(f"decision-causal cache audit failed: {cache_audit}")
    cache = TokenCache(source_cache)
    prepared = {split: prepare_split(packages["screen"].bundle, split, cache, steps=8) for split in ("train", "validation", "heldout")}
    record_order_sha256 = hashlib.sha256(_canonical([record.id for record in packages["screen"].records])).hexdigest().upper()
    return {"checks": checks, "checkpoint_sha256": checkpoint_sha256, "package_identity": identity, "record_order_sha256": record_order_sha256, "cache_audit": cache_audit, "cache": cache, "prepared": prepared, "output_root": output_root, "source_checkpoint": source_checkpoint, "forbidden_wd_root": forbidden_wd_root}


def _result_fail(
    output_root: Path,
    preflight: Mapping[str, Any],
    source_checkpoint: Path,
    status: str,
    **parts: Any,
) -> dict[str, Any]:
    result = {"schema_version": f"{SCHEMA_PREFIX}.result.v1", "status": status, "scope": "NONFORMAL_WD_DECISION_CAUSAL_SCREEN_ONLY", "authorizes": "nothing", "claims": {"decision_causal_signal": False, "h1_qualified": False, "p1_completed": False, "real_text_qualified": False, "f1_authorized": False, "p2_authorized": False}, "preflight": preflight, "source_checkpoint_unchanged": _sha256_path(source_checkpoint) == SOURCE_CHECKPOINT_SHA256, **parts}
    _write_json(output_root / "result.json", result)
    return result


def run_decision_causal_screen(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    context = _prepare_context(repo_root)
    cache, prepared = context["cache"], context["prepared"]
    source = load_deployment_checkpoint(
        context["source_checkpoint"], device=SPEC.device
    ).eval()
    route_replay_audit = audit_route_replay_equivalence(
        source,
        cache,
        prepared["heldout"],
        device=SPEC.device,
        records=SPEC.target_microbatch_size,
    )
    if route_replay_audit["passed"] is not True:
        raise RuntimeError(
            f"decision-causal route replay audit failed: {route_replay_audit}"
        )

    output_root: Path = context["output_root"]
    output_root.mkdir(parents=True, exist_ok=False)
    events_path = output_root / "training-events.jsonl"
    state_path = output_root / "run-state.json"
    started = time.perf_counter()

    def progress(stage: str, row: Mapping[str, Any]) -> None:
        event = {"timestamp": time.time(), "stage": stage, **dict(row)}
        _append_jsonl(events_path, event)
        _write_json(
            state_path,
            {
                "schema_version": f"{SCHEMA_PREFIX}.run-state.v1",
                "status": (
                    str(row.get("status"))
                    if stage == "complete" and row.get("status")
                    else "RUNNING_NONFORMAL_WD_DECISION_CAUSAL"
                ),
                "stage": stage,
                "latest": dict(row),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

    _write_json(output_root / "contract-manifest.json", contract_manifest())
    preflight = {
        "schema_version": f"{SCHEMA_PREFIX}.preflight.v1",
        "status": "PASS_NONFORMAL_WD_DECISION_CAUSAL_PREFLIGHT",
        "checks": context["checks"] | {"route_replay_equal": True},
        "route_replay_audit": route_replay_audit,
        "checkpoint_sha256": context["checkpoint_sha256"],
        "package_identity": context["package_identity"],
        "record_order_sha256": context["record_order_sha256"],
        "cache_audit": context["cache_audit"],
        "old_root_mutation_authorized": False,
        "forbidden_wd_root": str(context["forbidden_wd_root"]),
        "actual_source_checkpoint": str(context["source_checkpoint"]),
        "actual_source_token_cache": str(repo_root / SOURCE_TOKEN_CACHE),
        "input_allowlist": {
            "model_checkpoints": [str(context["source_checkpoint"])],
            "token_caches": [str(repo_root / SOURCE_TOKEN_CACHE)],
            "package_identity": SOURCE_PACKAGE_IDENTITY,
        },
        "forbidden_wd_input_paths": [str(context["forbidden_wd_root"])],
        "checkpoint_load_count": 1,
        "uses_family_targets_for_training": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    progress("baseline", {"message": "loading immutable predecessor"})
    source_suite = evaluate_intervention_suite(
        source, cache, prepared["heldout"], device=SPEC.device
    )

    target_datasets = {
        split: build_causal_target_dataset(
            source,
            cache,
            prepared[split],
            SPEC,
            batch_size=SPEC.target_microbatch_size,
            progress=progress,
        )
        for split in ("train", "heldout")
    }
    target_artifact = _save_target_datasets(
        target_datasets, output_root / "causal-target-datasets.pt"
    )
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    target_checks = {
        split: {
            "target_finite": dataset.stats["target_finite"] is True,
            "positive_fraction": dataset.stats["positive_fraction"]
            >= SPEC.minimum_positive_target_fraction,
            "realized_request_ratio": dataset.stats["realized_request_ratio"]
            >= SPEC.minimum_realized_request_ratio,
            "gpu_smoke": dataset.stats["gpu_smoke_gb"]
            < SPEC.maximum_gpu_smoke_gb,
        }
        for split, dataset in target_datasets.items()
    }
    target_gate = all(
        all(checks.values()) for checks in target_checks.values()
    )
    target_bank = {
        "schema_version": f"{SCHEMA_PREFIX}.target-bank.v1",
        "artifact": target_artifact,
        "splits": {
            split: dict(dataset.stats)
            for split, dataset in target_datasets.items()
        },
        "checks": target_checks,
        "passed": target_gate,
        "uses_family_targets": False,
    }
    _write_json(output_root / "target-bank.json", target_bank)
    if not target_gate:
        result = _result_fail(
            output_root,
            preflight,
            context["source_checkpoint"],
            "FAIL_WD_DECISION_CAUSAL_TARGET",
            target_bank=target_bank,
        )
        progress("complete", {"status": result["status"]})
        return result

    schedules: dict[str, Any] = {}
    for stage, updates, seed_tag in (
        ("write", SPEC.write_updates, 0x51),
        ("delete", SPEC.delete_updates, 0x44),
    ):
        schedule, report = build_unlabeled_schedule(
            len(prepared["train"].ids),
            updates=updates,
            batch_size=SPEC.batch_size,
            seed=SPEC.schedule_seed ^ seed_tag,
        )
        schedules[f"{stage}_causal"] = schedule
        schedules[f"{stage}_control"] = schedule.copy()
        schedules[f"{stage}_causal_report"] = report
        schedules[f"{stage}_control_report"] = dict(report)
    schedules["joint"], schedules["joint_report"] = build_unlabeled_schedule(
        len(prepared["train"].ids),
        updates=SPEC.joint_updates,
        batch_size=SPEC.batch_size,
        seed=SPEC.schedule_seed ^ 0x4A,
    )
    _write_json(
        output_root / "schedules.json",
        {key: value for key, value in schedules.items() if key.endswith("report")},
    )

    causal_written, control_written = deepcopy(source), deepcopy(source)
    write_training = {
        "causal": train_write_stage(
            causal_written,
            cache,
            prepared["train"],
            target_datasets["train"],
            schedules["write_causal"],
            SPEC,
            sign=1.0,
            progress=progress,
        ),
        "control": train_write_stage(
            control_written,
            cache,
            prepared["train"],
            target_datasets["train"],
            schedules["write_control"],
            SPEC,
            sign=-1.0,
            progress=progress,
        ),
    }
    write_fit = {
        "causal": evaluate_write_fit(
            causal_written,
            cache,
            prepared["heldout"],
            target_datasets["heldout"],
            SPEC,
            sign=1.0,
        ),
        "control": evaluate_write_fit(
            control_written,
            cache,
            prepared["heldout"],
            target_datasets["heldout"],
            SPEC,
            sign=-1.0,
        ),
    }
    write_checks = {
        "causal_transfer_fit": write_fit["causal"]["transfer_normalized_mse"]
        <= SPEC.maximum_causal_write_normalized_mse,
        "control_transfer_fit": write_fit["control"]["transfer_normalized_mse"]
        <= SPEC.maximum_control_write_normalized_mse,
        "heldout_targets_finite": all(
            value["target_finite"] for value in write_fit.values()
        ),
        "heldout_positive_fraction": all(
            value["positive_fraction"] >= SPEC.minimum_positive_target_fraction
            for value in write_fit.values()
        ),
        "heldout_realized_request_ratio": all(
            value["realized_request_ratio"] >= SPEC.minimum_realized_request_ratio
            for value in write_fit.values()
        ),
    }
    write_gate = all(write_checks.values())
    write_part = {
        "training": write_training,
        "fit": write_fit,
        "gate": {"checks": write_checks, "passed": write_gate},
    }
    if not write_gate:
        result = _result_fail(
            output_root,
            preflight,
            context["source_checkpoint"],
            "FAIL_WD_DECISION_CAUSAL_WRITE_FIT",
            target_bank=target_bank,
            write=write_part,
        )
        progress("complete", {"status": result["status"]})
        return result
    write_checkpoints = {
        "causal": _save_model(
            causal_written,
            output_root / "write-causal-checkpoint.pt",
            stage="write-causal",
        ),
        "control": _save_model(
            control_written,
            output_root / "write-control-checkpoint.pt",
            stage="write-control",
        ),
    }
    write_part["checkpoints"] = write_checkpoints

    causal_residual, control_residual = deepcopy(causal_written), deepcopy(control_written)
    delete_training = {
        "causal": train_delete_stage(
            causal_residual,
            causal_written,
            cache,
            prepared["train"],
            target_datasets["train"],
            schedules["delete_causal"],
            SPEC,
            sign=1.0,
            progress=progress,
        ),
        "control": train_delete_stage(
            control_residual,
            control_written,
            cache,
            prepared["train"],
            target_datasets["train"],
            schedules["delete_control"],
            SPEC,
            sign=-1.0,
            progress=progress,
        ),
    }
    delete_fit = {
        "causal": evaluate_delete_fit(
            causal_residual,
            causal_written,
            cache,
            prepared["heldout"],
            target_datasets["heldout"],
            SPEC,
            sign=1.0,
        ),
        "control": evaluate_delete_fit(
            control_residual,
            control_written,
            cache,
            prepared["heldout"],
            target_datasets["heldout"],
            SPEC,
            sign=-1.0,
        ),
    }
    delete_rollout = {
        "causal": compare_free_rollout(
            source,
            causal_residual,
            cache,
            prepared["heldout"],
            device=SPEC.device,
        ),
        "control": compare_free_rollout(
            source,
            control_residual,
            cache,
            prepared["heldout"],
            device=SPEC.device,
        ),
    }
    delete_checks = {
        "causal_transfer_fit": delete_fit["causal"]["common_transfer_normalized_mse"]
        <= SPEC.maximum_causal_delete_normalized_mse,
        "control_transfer_fit": delete_fit["control"]["common_transfer_normalized_mse"]
        <= SPEC.maximum_control_delete_normalized_mse,
        "causal_rollout_agreement": delete_rollout["causal"]["prediction_agreement"]
        >= SPEC.minimum_causal_rollout_agreement,
        "control_rollout_agreement": delete_rollout["control"]["prediction_agreement"]
        >= SPEC.minimum_causal_rollout_agreement,
    }
    delete_gate = all(delete_checks.values())
    delete_part = {
        "training": delete_training,
        "fit": delete_fit,
        "free_rollout": delete_rollout,
        "gate": {"checks": delete_checks, "passed": delete_gate},
    }
    if not delete_gate:
        result = _result_fail(
            output_root,
            preflight,
            context["source_checkpoint"],
            "FAIL_WD_DECISION_CAUSAL_DELETE_FIT",
            target_bank=target_bank,
            write=write_part,
            delete=delete_part,
        )
        progress("complete", {"status": result["status"]})
        return result
    residual_checkpoints = {
        "causal": _save_model(
            causal_residual,
            output_root / "residual-causal-checkpoint.pt",
            stage="delete-causal",
        ),
        "control": _save_model(
            control_residual,
            output_root / "residual-control-checkpoint.pt",
            stage="delete-control",
        ),
    }
    delete_part["checkpoints"] = residual_checkpoints
    causal_final, control_final = deepcopy(causal_residual), deepcopy(control_residual)
    joint_training = train_joint_pair(
        causal_final,
        control_final,
        cache,
        prepared["train"],
        target_datasets["train"],
        schedules["joint"],
        SPEC,
        progress=progress,
    )
    final_suites = {
        "causal": evaluate_intervention_suite(
            causal_final, cache, prepared["heldout"], device=SPEC.device
        ),
        "control": evaluate_intervention_suite(
            control_final, cache, prepared["heldout"], device=SPEC.device
        ),
        "source": source_suite,
    }
    free_rollout = {
        "causal": compare_free_rollout(
            source,
            causal_final,
            cache,
            prepared["heldout"],
            device=SPEC.device,
        ),
        "control": compare_free_rollout(
            source,
            control_final,
            cache,
            prepared["heldout"],
            device=SPEC.device,
        ),
    }
    shapley = {
        name: route_replay_shapley(
            model,
            cache,
            prepared["heldout"],
            device=SPEC.device,
            route_schedule=target_datasets["heldout"].route_schedule,
            bootstrap=2000,
            seed=SPEC.schedule_seed ^ index,
        )
        for index, (name, model) in enumerate(
            (("source", source), ("causal", causal_final), ("control", control_final))
        )
    }
    causal_pred = final_suites["causal"]["full"]["none"]["answer"]["correct_vector"]
    control_pred = final_suites["control"]["full"]["none"]["answer"]["correct_vector"]
    directional_benefit = paired_bootstrap_comparison(
        control_pred,
        causal_pred,
        seed=SPEC.schedule_seed ^ 0xB00757,
        bootstrap=2000,
        families=prepared["heldout"].families,
    )
    causal_compact = final_suites["causal"]["compact"]
    source_compact = final_suites["source"]["compact"]
    causal_projection_gain = (
        causal_compact["disable_projection"]["drop_from_normal"]
        - source_compact["disable_projection"]["drop_from_normal"]
    )
    source_unchanged = (
        _sha256_path(context["source_checkpoint"]) == SOURCE_CHECKPOINT_SHA256
    )
    mechanism_checks = {
        "target": target_gate,
        "write_fit": write_gate,
        "delete_fit": delete_gate,
        "causal_common_off_drop": causal_compact["disable_common"]["drop_from_normal"]
        >= SPEC.minimum_causal_common_off_drop,
        "projection_effect_gain": causal_projection_gain
        >= SPEC.minimum_projection_effect_gain_vs_source,
        "common_residual_energy": SPEC.minimum_common_residual_energy
        <= delete_fit["causal"]["common_residual_energy"]
        <= SPEC.maximum_common_residual_energy,
        "causal_final_free_route_agreement": free_rollout["causal"][
            "free_route_agreement"
        ]
        >= SPEC.minimum_final_free_route_agreement,
        "control_final_free_route_agreement": free_rollout["control"][
            "free_route_agreement"
        ]
        >= SPEC.minimum_final_free_route_agreement,
        "source_checkpoint_unchanged": source_unchanged,
        "no_family_training_target": True,
        "no_teacher_or_route_supervision": bool(
            tuple(joint_training["objective_components"])
            == (
                "ordinary_answer_cross_entropy",
                "two_path_causal_target_lock",
            )
            and joint_training["answer_route_mode"] == "model_free_route"
            and joint_training["allocation_lock_route_mode"]
            == "frozen_source_route_replay"
            and joint_training["teacher_kl_weight"] == 0.0
            and joint_training["teacher_model_used"] is False
            and joint_training["route_supervision_used"] is False
        ),
    }
    source_share = shapley["source"]["projection_share"]
    causal_share = shapley["causal"]["projection_share"]
    control_share = shapley["control"]["projection_share"]
    shapley_source_gain = paired_bootstrap_comparison(
        shapley["source"]["projection_share_values"],
        shapley["causal"]["projection_share_values"],
        seed=SPEC.schedule_seed ^ 0x5A,
        bootstrap=2000,
    )
    shapley_control_gain = paired_bootstrap_comparison(
        shapley["control"]["projection_share_values"],
        shapley["causal"]["projection_share_values"],
        seed=SPEC.schedule_seed ^ 0x5B,
        bootstrap=2000,
    )
    shapley_checks = {
        "projection_gain_vs_source": (
            causal_share - source_share
            >= SPEC.minimum_shapley_projection_share_gain_vs_source
            and shapley_source_gain["ci_lower"] > SPEC.minimum_shapley_ci_lower
        ),
        "projection_gain_vs_directional_control": (
            causal_share - control_share
            >= SPEC.minimum_shapley_projection_share_gain_vs_control
            and shapley_control_gain["ci_lower"] > SPEC.minimum_shapley_ci_lower
        ),
        "component_shares": (
            shapley["causal"]["projection_share"]
            >= SPEC.minimum_shapley_component_share
            and shapley["causal"]["common_share"]
            >= SPEC.minimum_shapley_component_share
        ),
    }
    directional_checks = {
        "heldout_gain": directional_benefit["gain"]
        >= SPEC.minimum_causal_control_heldout_gain,
        "ci_lower": directional_benefit["ci_lower"]
        > SPEC.minimum_causal_control_ci_lower,
        "family_regression": all(
            value >= -SPEC.maximum_family_regression
            for value in directional_benefit.get("family_gain", {}).values()
        ),
    }
    transfer_mechanism_passed = all(mechanism_checks.values()) and all(
        shapley_checks.values()
    )
    directional_control_passed = all(directional_checks.values())
    decision_causal_passed = (
        transfer_mechanism_passed and directional_control_passed
    )
    status = (
        "PASS_WD_DECISION_CAUSAL_NONFORMAL_MECHANISM"
        if decision_causal_passed
        else "FAIL_WD_DECISION_CAUSAL_NONFORMAL_MECHANISM"
    )
    final_checkpoints = {
        "causal": _save_model(
            causal_final, output_root / "causal-final.pt", stage="joint-causal"
        ),
        "control": _save_model(
            control_final, output_root / "control-final.pt", stage="joint-control"
        ),
    }
    result = {
        "schema_version": f"{SCHEMA_PREFIX}.result.v1",
        "status": status,
        "scope": "NONFORMAL_WD_DECISION_CAUSAL_SCREEN_ONLY",
        "authorizes": "nothing",
        "claims": {
            "decision_causal_signal": decision_causal_passed,
            "transfer_mechanism_signal": transfer_mechanism_passed,
            "directional_control_signal": directional_control_passed,
            "h1_qualified": False,
            "p1_completed": False,
            "real_text_qualified": False,
            "f1_authorized": False,
            "p2_authorized": False,
        },
        "preflight": preflight,
        "target_bank": target_bank,
        "schedules": {
            key: value for key, value in schedules.items() if key.endswith("report")
        },
        "source": {"interventions": source_compact},
        "source_checkpoint_unchanged": source_unchanged,
        "write": write_part,
        "delete": delete_part,
        "joint": {
            "training": joint_training,
            "interventions": {
                name: value["compact"] for name, value in final_suites.items()
            },
            "free_rollout": free_rollout,
            "shapley": shapley,
            "shapley_gain_vs_source": shapley_source_gain,
            "shapley_gain_vs_directional_control": shapley_control_gain,
            "directional_control_benefit": directional_benefit,
            "checkpoints": final_checkpoints,
        },
        "transfer_mechanism_gate": {
            "checks": mechanism_checks | shapley_checks,
            "passed": transfer_mechanism_passed,
        },
        "directional_control_gate": {
            "checks": directional_checks,
            "passed": directional_control_passed,
        },
        "decision_causal_gate": {
            "checks": {
                "transfer_mechanism": transfer_mechanism_passed,
                "directional_control": directional_control_passed,
            },
            "passed": decision_causal_passed,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_root / "result.json", result)
    progress("complete", {"status": status})
    return result


run_write_delete_decision_causal_screen = run_decision_causal_screen


__all__ = [
    "CausalTargetDataset",
    "audit_route_replay_equivalence",
    "build_causal_target_dataset",
    "build_unlabeled_schedule",
    "compare_free_rollout",
    "evaluate_answer",
    "evaluate_delete_fit",
    "evaluate_intervention_suite",
    "evaluate_write_fit",
    "paired_bootstrap_comparison",
    "route_replay_shapley",
    "run_decision_causal_screen",
    "run_write_delete_decision_causal_screen",
    "train_delete_stage",
    "train_joint_pair",
    "train_write_stage",
]
