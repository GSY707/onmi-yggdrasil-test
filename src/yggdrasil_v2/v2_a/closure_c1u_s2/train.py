from __future__ import annotations

"""Deterministic complete-group training and fixed endpoints for C1U S2."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from yggdrasil_v2.v2_a.closure_c1u.train import (
    build_optimizer,
    configure_cpu_threads,
    lr_scale,
    set_determinism,
)

from . import contract
from .model import S2Config, S2Model
from .objective import LossWeights, compute_training_loss
from .runtime import BoundedBatchProvider


@dataclass(frozen=True)
class ScheduledBatch:
    update: int
    cycle: int
    fold_id: str
    cps_group: str
    ere_group: str
    example_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrainSpec:
    batch_size: int = int(contract.TRAINING["batch_size"])
    groups_per_batch: int = int(contract.TRAINING["groups_per_batch"])
    maximum_updates: int = int(contract.TRAINING["maximum_updates"])
    boundary_lr: float = float(contract.TRAINING["boundary_lr"])
    transition_lr: float = float(contract.TRAINING["transition_lr"])
    head_lr: float = float(contract.TRAINING["head_lr"])
    weight_decay: float = float(contract.TRAINING["weight_decay"])
    gradient_clip: float = float(contract.TRAINING["gradient_clip"])
    warmup_updates: int = int(contract.TRAINING["warmup_updates"])
    log_interval: int = int(contract.TRAINING["log_interval"])
    cpu_threads: int = int(contract.TRAINING["cpu_threads"])
    pin_memory: bool = bool(contract.TRAINING["pin_memory"])


def _groups(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, tuple[str, ...]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {"CPS": {}, "ERE": {}}
    for row in records:
        family = str(row.get("family", "")).upper()
        group = str(row.get("factorial_group_id", ""))
        if family not in grouped or not group:
            raise ValueError("S2 schedule row has invalid family/group")
        grouped[family].setdefault(group, []).append(row)
    result: dict[str, dict[str, tuple[str, ...]]] = {"CPS": {}, "ERE": {}}
    for family, families in grouped.items():
        if len(families) != contract.TRAIN_GROUPS_PER_FAMILY:
            raise ValueError(f"{family} S2 train group cardinality drifted")
        for group, rows in sorted(families.items()):
            ordered = sorted(rows, key=lambda row: tuple(int(v) for v in row["factors"]))
            if tuple(tuple(int(v) for v in row["factors"]) for row in ordered) != (
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ):
                raise ValueError(f"incomplete S2 factorial group: {group}")
            result[family][group] = tuple(str(row["example_id"]) for row in ordered)
    return result


def build_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    fold_id: str,
    seed: int,
    maximum_updates: int = contract.MAXIMUM_UPDATES_PER_ENDPOINT,
) -> list[ScheduledBatch]:
    if type(maximum_updates) is not int or maximum_updates < 1:
        raise ValueError("maximum_updates must be a positive integer")
    groups = _groups(records)
    cps = sorted(groups["CPS"])
    ere = sorted(groups["ERE"])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    schedule: list[ScheduledBatch] = []
    cycle = 0
    while len(schedule) < maximum_updates:
        cps_order = torch.randperm(len(cps), generator=generator).tolist()
        ere_order = torch.randperm(len(ere), generator=generator).tolist()
        for position in range(len(cps)):
            if len(schedule) >= maximum_updates:
                break
            cps_group = cps[cps_order[position]]
            ere_group = ere[ere_order[position]]
            schedule.append(
                ScheduledBatch(
                    update=len(schedule) + 1,
                    cycle=cycle,
                    fold_id=str(fold_id),
                    cps_group=cps_group,
                    ere_group=ere_group,
                    example_ids=groups["CPS"][cps_group] + groups["ERE"][ere_group],
                )
            )
        cycle += 1
    return schedule


def schedule_report(schedule: Sequence[ScheduledBatch]) -> dict[str, Any]:
    rows = []
    counts: dict[str, int] = {}
    for scheduled in schedule:
        row = asdict(scheduled)
        row["example_ids"] = list(scheduled.example_ids)
        rows.append(row)
        counts[scheduled.cps_group] = counts.get(scheduled.cps_group, 0) + 1
        counts[scheduled.ere_group] = counts.get(scheduled.ere_group, 0) + 1
    canonical = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.schedule.v1",
        "fold_id": schedule[0].fold_id if schedule else None,
        "updates": len(schedule),
        "examples": sum(len(row.example_ids) for row in schedule),
        "cycles": 1 + max((row.cycle for row in schedule), default=-1),
        "complete_groups_per_batch": all(len(row.example_ids) == 8 for row in schedule),
        "group_counts": dict(sorted(counts.items())),
        "sha256": hashlib.sha256(canonical).hexdigest().upper(),
        "rows": rows,
    }


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, Tensor)
        else value
        for name, value in batch.items()
    }


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


ProgressCallback = Callable[[Mapping[str, Any]], None]


def train_fixed_endpoint(
    model: S2Model,
    provider: BoundedBatchProvider,
    schedule: Sequence[ScheduledBatch],
    *,
    identity: str,
    arm: str,
    fold_id: str,
    model_seed: int,
    order_seed: int,
    spec: TrainSpec,
    device: str | torch.device,
    weights: LossWeights = LossWeights(),
    on_optimizer_step: ProgressCallback | None = None,
) -> dict[str, Any]:
    if len(schedule) != spec.maximum_updates:
        raise ValueError("S2 schedule length differs from fixed endpoint")
    if [row.update for row in schedule] != list(range(1, spec.maximum_updates + 1)):
        raise ValueError("S2 schedule updates are not contiguous")
    if any(row.fold_id != fold_id for row in schedule):
        raise ValueError("S2 schedule fold drifted")
    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA device is unavailable")
    determinism = set_determinism(model_seed)
    threads = configure_cpu_threads(spec.cpu_threads)
    model.to(active_device).train()
    optimizer = build_optimizer(model, spec)
    if active_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(active_device)
        torch.cuda.synchronize(active_device)
    started = time.perf_counter()
    data_seconds = 0.0
    compute_seconds = 0.0
    durations: list[float] = []
    logs: list[dict[str, Any]] = []
    completed = 0
    last_components: dict[str, float] = {}
    for scheduled in schedule:
        data_started = time.perf_counter()
        cpu = provider.get_batch(
            scheduled.example_ids,
            pin_memory=bool(spec.pin_memory and active_device.type == "cuda"),
        )
        batch = _move_batch(cpu, active_device)
        if not bool((batch["counterfactual_indices"] >= 0).all()):
            raise ValueError("S2 training batch lacks factorial counterparts")
        data_seconds += time.perf_counter() - data_started
        step_started = time.perf_counter()
        scale = lr_scale(scheduled.update, spec)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        optimizer.zero_grad(set_to_none=True)
        with _autocast(active_device):
            loss, components, _ = compute_training_loss(model, batch, weights=weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite S2 loss at update {scheduled.update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()), spec.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                f"non-finite S2 gradient at update {scheduled.update}"
            )
        optimizer.step()
        if active_device.type == "cuda":
            torch.cuda.synchronize(active_device)
        duration = time.perf_counter() - step_started
        completed = scheduled.update
        compute_seconds += duration
        durations.append(duration)
        last_components = dict(components)
        progress = {
            "update": completed,
            "cycle": scheduled.cycle,
            "fold_id": fold_id,
            "arm": arm,
            "cps_group": scheduled.cps_group,
            "ere_group": scheduled.ere_group,
            "loss": dict(components),
            "gradient_norm": float(gradient_norm.detach().float().cpu()),
            "lr_scale": float(scale),
            "step_seconds": float(duration),
        }
        if on_optimizer_step is not None:
            on_optimizer_step(progress)
        if completed == 1 or completed == spec.maximum_updates or completed % spec.log_interval == 0:
            logs.append(progress)
    wall = time.perf_counter() - started
    if completed != spec.maximum_updates:
        raise RuntimeError("S2 training stopped before fixed endpoint")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("S2 endpoint contains non-finite parameters")
    ordered = sorted(durations)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.training.v1",
        "identity": identity,
        "arm": arm,
        "fold_id": fold_id,
        "passed": True,
        "fixed_endpoint": True,
        "fixed_endpoint_name": contract.FIXED_ENDPOINT,
        "checkpoint_selection": False,
        "intermediate_checkpoints": 0,
        "completed_updates": completed,
        "optimizer_steps": completed,
        "model_writes": 0,
        "model_seed": int(model_seed),
        "order_seed": int(order_seed),
        "spec": asdict(spec),
        "loss_weights": asdict(weights),
        "determinism": determinism,
        "threads": threads,
        "wall_seconds": wall,
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
        "data_wait_fraction": data_seconds / max(wall, 1.0e-12),
        "updates_per_second": completed / max(wall, 1.0e-12),
        "step_median_seconds": statistics.median(durations),
        "step_p95_seconds": ordered[p95_index],
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(active_device))
        if active_device.type == "cuda"
        else 0,
        "last_loss": last_components,
        "log": logs,
        "provider": provider.report(),
    }


def save_endpoint(
    path: Path,
    *,
    model: S2Model,
    identity: str,
    arm: str,
    fold_id: str,
    update: int,
    schedule_sha256: str,
) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"S2 fixed endpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.endpoint.v1",
            "identity": identity,
            "arm": arm,
            "fold_id": fold_id,
            "fixed_endpoint_name": contract.FIXED_ENDPOINT,
            "update": int(update),
            "config": asdict(model.config),
            "schedule_sha256": str(schedule_sha256),
            "model_state": {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            "optimizer_state_saved": False,
            "checkpoint_selection": False,
            "intermediate_checkpoint": False,
        },
        path,
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_endpoint(
    path: Path, *, device: str | torch.device = "cpu"
) -> tuple[S2Model, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("S2 endpoint payload must be a mapping")
    if payload.get("schema_version") != f"{contract.SCHEMA_PREFIX}.endpoint.v1":
        raise ValueError("S2 endpoint schema mismatch")
    model = S2Model(S2Config(**dict(payload["config"])))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload)


__all__ = [
    "ScheduledBatch",
    "TrainSpec",
    "build_schedule",
    "load_endpoint",
    "save_endpoint",
    "schedule_report",
    "train_fixed_endpoint",
]
