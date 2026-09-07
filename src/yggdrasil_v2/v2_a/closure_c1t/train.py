from __future__ import annotations

"""Deterministic complete-factorial training for C1T S1.

The training loop consumes only immutable public-card batches plus external
loss targets.  It never selects a checkpoint and never writes model state;
the single fixed endpoint write is owned by the stage runner.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from . import contract
from .model import C1TConfig, C1TModel
from .objective import LossWeights, compute_training_loss
from .runtime import C1TRuntime


@dataclass(frozen=True)
class ScheduledBatch:
    update: int
    cycle: int
    cps_group: str
    ere_group: str
    example_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrainSpec:
    batch_size: int = int(contract.S1_TRAINING["batch_size"])
    groups_per_batch: int = int(contract.S1_TRAINING["groups_per_batch"])
    maximum_updates: int = int(contract.S1_TRAINING["maximum_updates"])
    boundary_lr: float = float(contract.S1_TRAINING["boundary_lr"])
    transition_lr: float = float(contract.S1_TRAINING["transition_lr"])
    head_lr: float = float(contract.S1_TRAINING["head_lr"])
    weight_decay: float = float(contract.S1_TRAINING["weight_decay"])
    gradient_clip: float = float(contract.S1_TRAINING["gradient_clip"])
    warmup_updates: int = int(contract.S1_TRAINING["warmup_updates"])
    log_interval: int = int(contract.S1_TRAINING["log_interval"])
    cpu_threads: int = int(contract.S1_TRAINING["cpu_threads"])
    pin_memory: bool = bool(contract.S1_TRAINING["pin_memory"])


class RuntimeBatchProvider:
    """Memoize the sixteen possible CPS-group x ERE-group CPU batches."""

    def __init__(self, runtime: C1TRuntime) -> None:
        self.runtime = runtime
        self._batches: dict[tuple[str, ...], dict[str, Any]] = {}

    def get_batch(
        self, example_ids: Sequence[str], *, pin_memory: bool = False
    ) -> Mapping[str, Any]:
        key = tuple(str(value) for value in example_ids)
        if len(key) != contract.S1_BATCH_SIZE:
            raise ValueError("C1T S1 batches must contain exactly eight records")
        if key not in self._batches:
            batch = self.runtime.get_batch(key)
            if not bool((batch["counterfactual_indices"] >= 0).all()):
                raise ValueError("S1 batch split a registered factorial group")
            if pin_memory:
                batch = {
                    name: value.pin_memory() if isinstance(value, Tensor) else value
                    for name, value in batch.items()
                }
            self._batches[key] = dict(batch)
        return self._batches[key]

    def report(self) -> dict[str, Any]:
        return {
            "cached_batch_count": len(self._batches),
            "maximum_registered_batch_count": (
                contract.S1_GROUPS_PER_FAMILY * contract.S1_GROUPS_PER_FAMILY
            ),
        }


def _group_rows(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        family: {} for family in contract.FAMILIES
    }
    for row in records:
        family = str(row.get("family", "")).upper()
        group = str(row.get("factorial_group_id", ""))
        if family not in grouped or not group:
            raise ValueError("schedule records require a registered family and group")
        grouped[family].setdefault(group, []).append(row)
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for family, groups in grouped.items():
        if len(groups) != contract.S1_GROUPS_PER_FAMILY:
            raise ValueError(f"{family} group cardinality drifted")
        result[family] = {}
        for group, rows in sorted(groups.items()):
            ordered = sorted(rows, key=lambda row: tuple(int(v) for v in row["factors"]))
            cells = tuple(tuple(int(v) for v in row["factors"]) for row in ordered)
            if cells != contract.FACTORIAL_CELLS:
                raise ValueError(f"factorial group is incomplete: {group}")
            ids = tuple(str(row["example_id"]) for row in ordered)
            if len(set(ids)) != len(ids):
                raise ValueError(f"factorial group has duplicate example ids: {group}")
            result[family][group] = ids
    return result


def build_s1_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    maximum_updates: int = contract.S1_MAXIMUM_UPDATES,
) -> list[ScheduledBatch]:
    """Pair one complete CPS group with one complete ERE group per update."""

    if type(maximum_updates) is not int or maximum_updates < 1:
        raise ValueError("maximum_updates must be a positive integer")
    groups = _group_rows(records)
    cps_names = sorted(groups["CPS"])
    ere_names = sorted(groups["ERE"])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    schedule: list[ScheduledBatch] = []
    cycle = 0
    while len(schedule) < maximum_updates:
        cps_order = torch.randperm(len(cps_names), generator=generator).tolist()
        ere_order = torch.randperm(len(ere_names), generator=generator).tolist()
        for position in range(contract.S1_GROUPS_PER_FAMILY):
            if len(schedule) >= maximum_updates:
                break
            cps_group = cps_names[cps_order[position]]
            ere_group = ere_names[ere_order[position]]
            ids = groups["CPS"][cps_group] + groups["ERE"][ere_group]
            schedule.append(
                ScheduledBatch(
                    update=len(schedule) + 1,
                    cycle=cycle,
                    cps_group=cps_group,
                    ere_group=ere_group,
                    example_ids=ids,
                )
            )
        cycle += 1
    return schedule


def schedule_report(schedule: Sequence[ScheduledBatch]) -> dict[str, Any]:
    rows = []
    for row in schedule:
        payload = asdict(row)
        payload["example_ids"] = list(row.example_ids)
        rows.append(payload)
    canonical = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    group_counts: dict[str, int] = {}
    for row in schedule:
        group_counts[row.cps_group] = group_counts.get(row.cps_group, 0) + 1
        group_counts[row.ere_group] = group_counts.get(row.ere_group, 0) + 1
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.s1-schedule.v1",
        "updates": len(schedule),
        "examples": sum(len(row.example_ids) for row in schedule),
        "cycles": 1 + max((row.cycle for row in schedule), default=-1),
        "complete_groups_per_batch": all(
            len(row.example_ids) == contract.S1_BATCH_SIZE for row in schedule
        ),
        "group_counts": dict(sorted(group_counts.items())),
        "sha256": hashlib.sha256(canonical).hexdigest().upper(),
        "rows": rows,
    }


def set_determinism(seed: int) -> dict[str, Any]:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": int(seed),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32)
        if torch.cuda.is_available()
        else False,
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32)
        if hasattr(torch.backends, "cudnn")
        else False,
    }


def configure_cpu_threads(intraop: int) -> dict[str, int]:
    if intraop < 1:
        raise ValueError("intraop must be positive")
    torch.set_num_threads(int(intraop))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {
        "intraop": int(torch.get_num_threads()),
        "interop": int(torch.get_num_interop_threads()),
    }


def lr_scale(update: int, spec: TrainSpec) -> float:
    if not 1 <= update <= spec.maximum_updates:
        raise ValueError("update outside the frozen endpoint")
    warmup = min(spec.warmup_updates, spec.maximum_updates)
    if update <= warmup:
        return update / max(1, warmup)
    progress = (update - warmup) / max(1, spec.maximum_updates - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: C1TModel, spec: TrainSpec) -> torch.optim.Optimizer:
    groups: dict[str, list[nn.Parameter]] = {
        "boundary": [],
        "transition": [],
        "head": [],
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"duplicate trainable parameter: {name}")
        seen.add(id(parameter))
        if name.startswith("boundary."):
            groups["boundary"].append(parameter)
        elif name.startswith("transition."):
            groups["transition"].append(parameter)
        elif name.startswith("readout."):
            groups["head"].append(parameter)
        else:
            raise ValueError(f"unregistered trainable parameter group: {name}")
    if any(not values for values in groups.values()):
        raise ValueError({name: len(values) for name, values in groups.items()})
    return torch.optim.AdamW(
        [
            {
                "params": groups["boundary"],
                "lr": spec.boundary_lr,
                "initial_lr": spec.boundary_lr,
                "group_name": "boundary",
            },
            {
                "params": groups["transition"],
                "lr": spec.transition_lr,
                "initial_lr": spec.transition_lr,
                "group_name": "transition",
            },
            {
                "params": groups["head"],
                "lr": spec.head_lr,
                "initial_lr": spec.head_lr,
                "group_name": "head",
            },
        ],
        weight_decay=spec.weight_decay,
    )


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


def _parameters_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


ProgressCallback = Callable[[Mapping[str, Any]], None]


def train_fixed_endpoint(
    model: C1TModel,
    provider: RuntimeBatchProvider,
    schedule: Sequence[ScheduledBatch],
    *,
    identity: str,
    model_seed: int,
    order_seed: int,
    spec: TrainSpec,
    device: str | torch.device,
    weights: LossWeights = LossWeights(),
    on_optimizer_step: ProgressCallback | None = None,
) -> dict[str, Any]:
    if len(schedule) != spec.maximum_updates:
        raise ValueError("schedule length differs from the fixed endpoint")
    if [row.update for row in schedule] != list(range(1, spec.maximum_updates + 1)):
        raise ValueError("schedule updates must be contiguous and one-indexed")
    if spec.batch_size != contract.S1_BATCH_SIZE or spec.groups_per_batch != 2:
        raise ValueError("training spec drifted from complete two-group batches")
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
    step_durations: list[float] = []
    log_rows: list[dict[str, Any]] = []
    last_components: dict[str, float] = {}
    completed = 0
    for scheduled in schedule:
        data_started = time.perf_counter()
        cpu_batch = provider.get_batch(
            scheduled.example_ids,
            pin_memory=bool(spec.pin_memory and active_device.type == "cuda"),
        )
        batch = _move_batch(cpu_batch, active_device)
        if not bool((batch["counterfactual_indices"] >= 0).all()):
            raise ValueError("training batch lacks complete factorial counterparts")
        data_seconds += time.perf_counter() - data_started

        step_started = time.perf_counter()
        scale = lr_scale(scheduled.update, spec)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        optimizer.zero_grad(set_to_none=True)
        with _autocast(active_device):
            loss, components, _ = compute_training_loss(model, batch, weights=weights)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {scheduled.update}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()), spec.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                f"non-finite gradient at update {scheduled.update}"
            )
        optimizer.step()
        if active_device.type == "cuda":
            torch.cuda.synchronize(active_device)
        duration = time.perf_counter() - step_started
        completed = scheduled.update
        compute_seconds += duration
        step_durations.append(duration)
        last_components = dict(components)
        progress = {
            "update": completed,
            "cycle": scheduled.cycle,
            "cps_group": scheduled.cps_group,
            "ere_group": scheduled.ere_group,
            "loss": dict(components),
            "gradient_norm": float(gradient_norm.detach().float().cpu()),
            "lr_scale": float(scale),
            "step_seconds": float(duration),
        }
        if on_optimizer_step is not None:
            on_optimizer_step(progress)
        if (
            completed == 1
            or completed == spec.maximum_updates
            or completed % spec.log_interval == 0
        ):
            log_rows.append(progress)

    wall_seconds = time.perf_counter() - started
    if completed != spec.maximum_updates:
        raise RuntimeError("training stopped before the frozen endpoint")
    if not _parameters_finite(model):
        raise FloatingPointError("fixed endpoint contains non-finite parameters")
    ordered = sorted(step_durations)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.s1-training.v1",
        "identity": identity,
        "passed": True,
        "fixed_endpoint": True,
        "fixed_endpoint_name": contract.S1_FIXED_ENDPOINT,
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
        "wall_seconds": wall_seconds,
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
        "data_wait_fraction": data_seconds / max(wall_seconds, 1.0e-12),
        "updates_per_second": completed / max(wall_seconds, 1.0e-12),
        "examples_per_second": completed * spec.batch_size / max(wall_seconds, 1.0e-12),
        "step_median_seconds": statistics.median(step_durations),
        "step_p95_seconds": ordered[p95_index],
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(active_device))
        if active_device.type == "cuda"
        else 0,
        "last_loss": last_components,
        "log": log_rows,
        "provider": provider.report(),
    }


def save_endpoint(
    path: Path,
    *,
    model: C1TModel,
    identity: str,
    update: int,
    schedule_sha256: str,
) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"fixed endpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s1-endpoint.v1",
            "identity": identity,
            "fixed_endpoint_name": contract.S1_FIXED_ENDPOINT,
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
) -> tuple[C1TModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("fixed endpoint payload must be a mapping")
    if payload.get("schema_version") != f"{contract.SCHEMA_PREFIX}.s1-endpoint.v1":
        raise ValueError("fixed endpoint schema mismatch")
    model = C1TModel(C1TConfig(**dict(payload["config"])))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, dict(payload)


__all__ = [
    "RuntimeBatchProvider",
    "ScheduledBatch",
    "TrainSpec",
    "build_optimizer",
    "build_s1_schedule",
    "configure_cpu_threads",
    "load_endpoint",
    "lr_scale",
    "save_endpoint",
    "schedule_report",
    "set_determinism",
    "train_fixed_endpoint",
]
