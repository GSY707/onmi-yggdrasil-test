from __future__ import annotations

"""Deterministic fixed-endpoint training for the C1S successor chain.

The training loop deliberately knows nothing about record semantics.  It
receives an immutable, ordered batch provider and sends only public hidden
states and masks through :class:`C1SModel`; all other tensors remain external
loss targets.
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
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn

from ..closure_c1s.model import C1SConfig, C1SModel
from .objective import CausalMargins, LossWeights, SuccessorDiagnosticHeads, compute_training_loss


class BatchProvider(Protocol):
    def get_batch(
        self,
        example_ids: Sequence[str],
        *,
        slots: int,
        pin_memory: bool = False,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ScheduledBatch:
    update: int
    epoch: int
    example_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrainSpec:
    batch_size: int = 8
    family_batch_size: int = 4
    epochs: int = 6
    maximum_updates: int = 4_608
    boundary_lr: float = 1.0e-4
    transition_lr: float = 2.0e-4
    head_lr: float = 3.0e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_updates: int = 256
    log_interval: int = 100
    causal_interval: int = 4
    cpu_threads: int = 2
    pin_memory: bool = True


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def configure_cpu_threads(intraop: int = 2) -> dict[str, int]:
    """Freeze the GPU hot-path CPU budget without assuming host core count."""

    if intraop < 1:
        raise ValueError("intraop must be positive")
    torch.set_num_threads(int(intraop))
    # PyTorch allows the inter-op pool to be configured only once per process.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return {
        "intraop": int(torch.get_num_threads()),
        "interop": int(torch.get_num_interop_threads()),
    }


def _family_ids(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"ERE": [], "CPS": []}
    seen: set[str] = set()
    for record in records:
        family = str(record.get("family", "")).upper()
        example_id = record.get("example_id")
        if family not in result or not isinstance(example_id, str) or not example_id:
            raise ValueError("schedule records require ERE/CPS family and example_id")
        if example_id in seen:
            raise ValueError(f"duplicate schedule example_id: {example_id}")
        seen.add(example_id)
        result[family].append(example_id)
    for values in result.values():
        values.sort()
    if len(result["ERE"]) != len(result["CPS"]):
        raise ValueError("balanced schedule requires equal ERE/CPS populations")
    return result


def build_balanced_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    spec: TrainSpec,
) -> list[ScheduledBatch]:
    by_family = _family_ids(records)
    if spec.batch_size != 2 * spec.family_batch_size:
        raise ValueError("batch_size must equal two family_batch_size halves")
    population = len(by_family["ERE"])
    if population < spec.family_batch_size or population % spec.family_batch_size:
        raise ValueError("each family population must divide family_batch_size")

    expected = spec.epochs * population // spec.family_batch_size
    if expected != spec.maximum_updates:
        raise ValueError(
            f"frozen schedule yields {expected} updates, not {spec.maximum_updates}"
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rows: list[ScheduledBatch] = []
    update = 0
    for epoch in range(spec.epochs):
        shuffled: dict[str, list[str]] = {}
        for family, values in by_family.items():
            order = torch.randperm(len(values), generator=generator).tolist()
            shuffled[family] = [values[index] for index in order]
        for start in range(0, population, spec.family_batch_size):
            update += 1
            stop = start + spec.family_batch_size
            rows.append(
                ScheduledBatch(
                    update=update,
                    epoch=epoch,
                    example_ids=tuple(shuffled["ERE"][start:stop] + shuffled["CPS"][start:stop]),
                )
            )
    return rows


def build_overfit_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    maximum_updates: int = 4_000,
    batch_size: int = 8,
    family_batch_size: int = 4,
) -> list[ScheduledBatch]:
    """Cycle over the frozen 32 records while reshuffling every four updates."""

    if maximum_updates < 1:
        raise ValueError("maximum_updates must be positive")
    by_family = _family_ids(records)
    if len(by_family["ERE"]) != 16 or len(by_family["CPS"]) != 16:
        raise ValueError("S1 requires exactly 16 records per family")
    if batch_size != 2 * family_batch_size or 16 % family_batch_size:
        raise ValueError("invalid S1 balanced batch geometry")
    batches_per_cycle = 16 // family_batch_size
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rows: list[ScheduledBatch] = []
    update = 0
    cycle = 0
    while update < maximum_updates:
        shuffled: dict[str, list[str]] = {}
        for family, values in by_family.items():
            order = torch.randperm(len(values), generator=generator).tolist()
            shuffled[family] = [values[index] for index in order]
        for batch_index in range(batches_per_cycle):
            if update >= maximum_updates:
                break
            update += 1
            start = batch_index * family_batch_size
            stop = start + family_batch_size
            rows.append(
                ScheduledBatch(
                    update=update,
                    epoch=cycle,
                    example_ids=tuple(shuffled["ERE"][start:stop] + shuffled["CPS"][start:stop]),
                )
            )
        cycle += 1
    return rows


def schedule_report(schedule: Sequence[ScheduledBatch]) -> dict[str, Any]:
    serialised = [asdict(row) for row in schedule]
    # dataclasses.asdict preserves tuples; normalise to lists for canonical JSON.
    for row in serialised:
        row["example_ids"] = list(row["example_ids"])
    payload = json.dumps(serialised, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "updates": len(schedule),
        "examples": sum(len(row.example_ids) for row in schedule),
        "epochs_or_cycles": 1 + max((row.epoch for row in schedule), default=-1),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
        "rows": serialised,
    }


def lr_scale(update: int, spec: TrainSpec) -> float:
    if not 1 <= update <= spec.maximum_updates:
        raise ValueError("update outside frozen endpoint")
    warmup = min(spec.warmup_updates, spec.maximum_updates)
    if update <= warmup:
        return update / max(1, warmup)
    progress = (update - warmup) / max(1, spec.maximum_updates - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(
    model: C1SModel,
    diagnostic_heads: SuccessorDiagnosticHeads,
    spec: TrainSpec,
) -> torch.optim.Optimizer:
    groups: dict[str, list[nn.Parameter]] = {
        "boundary": [],
        "transition": [],
        "heads": [],
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"duplicate model parameter: {name}")
        seen.add(id(parameter))
        if name.startswith("boundary."):
            groups["boundary"].append(parameter)
        elif name.startswith("transition."):
            groups["transition"].append(parameter)
        else:
            groups["heads"].append(parameter)
    for name, parameter in diagnostic_heads.named_parameters():
        if id(parameter) in seen:
            raise ValueError(f"duplicate diagnostic parameter: {name}")
        seen.add(id(parameter))
        groups["heads"].append(parameter)
    if any(not values for values in groups.values()):
        raise ValueError({name: len(values) for name, values in groups.items()})
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["boundary"], "lr": spec.boundary_lr, "initial_lr": spec.boundary_lr, "group_name": "boundary"},
            {"params": groups["transition"], "lr": spec.transition_lr, "initial_lr": spec.transition_lr, "group_name": "transition"},
            {"params": groups["heads"], "lr": spec.head_lr, "initial_lr": spec.head_lr, "group_name": "heads"},
        ],
        weight_decay=spec.weight_decay,
    )
    return optimizer


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            result[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            result[key] = value
    return result


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _finite_parameters(module: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters())


def save_endpoint(
    path: Path,
    *,
    model: C1SModel,
    diagnostic_heads: SuccessorDiagnosticHeads,
    update: int,
    identity: str,
    schedule_sha256: str,
    metrics: Mapping[str, Any],
) -> str:
    """Write the sole fixed endpoint for an arm; optimizer state is excluded."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"fixed endpoint already exists: {path}")
    torch.save(
        {
            "schema_version": "yggdrasil.v2-a.closure-c1s-successor.endpoint.v1",
            "identity": identity,
            "update": int(update),
            "config": asdict(model.config),
            "schedule_sha256": schedule_sha256,
            "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "diagnostic_state": {
                name: tensor.detach().cpu() for name, tensor in diagnostic_heads.state_dict().items()
            },
            "metrics": dict(metrics),
            "optimizer_state_saved": False,
            "checkpoint_selection": False,
        },
        path,
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_endpoint(path: Path, *, device: str | torch.device = "cpu") -> tuple[C1SModel, SuccessorDiagnosticHeads, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = C1SConfig(**dict(payload["config"]))
    model = C1SModel(config)
    model.load_state_dict(payload["model_state"], strict=True)
    heads = SuccessorDiagnosticHeads(config.source_width, config.address_width, config.auxiliary_width)
    heads.load_state_dict(payload["diagnostic_state"], strict=True)
    model.to(device)
    heads.to(device)
    return model, heads, dict(payload)


def train_fixed_endpoint(
    model: C1SModel,
    diagnostic_heads: SuccessorDiagnosticHeads,
    provider: BatchProvider,
    schedule: Sequence[ScheduledBatch],
    *,
    identity: str,
    order_seed: int,
    model_seed: int,
    spec: TrainSpec,
    device: str | torch.device = "cuda",
    loss_weights: LossWeights = LossWeights(),
    margins: CausalMargins = CausalMargins(),
    on_optimizer_step: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run exactly one frozen schedule and return in-memory endpoint evidence."""

    if len(schedule) != spec.maximum_updates:
        raise ValueError("schedule length differs from fixed endpoint")
    if [row.update for row in schedule] != list(range(1, spec.maximum_updates + 1)):
        raise ValueError("schedule updates must be contiguous and one-indexed")
    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA device is unavailable")
    set_determinism(model_seed)
    thread_report = configure_cpu_threads(spec.cpu_threads)
    model.to(active_device).train()
    diagnostic_heads.to(active_device).train()
    optimizer = build_optimizer(model, diagnostic_heads, spec)
    if active_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(active_device)
        torch.cuda.synchronize(active_device)

    started = time.perf_counter()
    data_seconds = 0.0
    compute_seconds = 0.0
    log_rows: list[dict[str, Any]] = []
    last_components: dict[str, float] = {}
    step_durations: list[float] = []
    for row in schedule:
        data_started = time.perf_counter()
        cpu_batch = provider.get_batch(
            row.example_ids,
            slots=model.config.slots,
            pin_memory=bool(spec.pin_memory and active_device.type == "cuda"),
        )
        batch = _move_batch(cpu_batch, active_device)
        data_seconds += time.perf_counter() - data_started
        step_started = time.perf_counter()
        scale = lr_scale(row.update, spec)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        optimizer.zero_grad(set_to_none=True)
        causal = spec.causal_interval > 0 and (
            row.update == 1 or row.update == spec.maximum_updates or row.update % spec.causal_interval == 0
        )
        with _autocast(active_device):
            loss, components, _ = compute_training_loss(
                model,
                diagnostic_heads,
                batch,
                weights=loss_weights,
                margins=margins,
                causal=causal,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {row.update}")
        loss.backward()
        parameters = list(model.parameters()) + list(diagnostic_heads.parameters())
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, spec.gradient_clip)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite gradient at update {row.update}")
        optimizer.step()
        if on_optimizer_step is not None:
            # The callback runs immediately after the mutation so a later
            # exception cannot make a consumed single-use run report zero.
            on_optimizer_step(int(row.update))
        if active_device.type == "cuda":
            torch.cuda.synchronize(active_device)
        duration = time.perf_counter() - step_started
        compute_seconds += duration
        step_durations.append(duration)
        last_components = components
        if row.update == 1 or row.update == spec.maximum_updates or row.update % spec.log_interval == 0:
            log_rows.append(
                {
                    "update": row.update,
                    "epoch": row.epoch,
                    "causal": causal,
                    "loss": components,
                    "gradient_norm": float(gradient_norm.detach().float().cpu()),
                    "lr_scale": scale,
                    "step_seconds": duration,
                }
            )

    wall_seconds = time.perf_counter() - started
    if not _finite_parameters(model) or not _finite_parameters(diagnostic_heads):
        raise FloatingPointError("non-finite endpoint parameter")
    ordered = sorted(step_durations)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "schema_version": "yggdrasil.v2-a.closure-c1s-successor.training.v1",
        "identity": identity,
        "passed": True,
        "fixed_endpoint": True,
        "checkpoint_selection": False,
        "completed_updates": len(schedule),
        "optimizer_steps": len(schedule),
        "model_writes": 0,
        "model_seed": int(model_seed),
        "order_seed": int(order_seed),
        "spec": asdict(spec),
        "threads": thread_report,
        "wall_seconds": wall_seconds,
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
        "data_wait_fraction": data_seconds / max(wall_seconds, 1.0e-12),
        "updates_per_second": len(schedule) / max(wall_seconds, 1.0e-12),
        "examples_per_second": len(schedule) * spec.batch_size / max(wall_seconds, 1.0e-12),
        "step_median_seconds": statistics.median(step_durations),
        "step_p95_seconds": ordered[p95_index],
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(active_device)) if active_device.type == "cuda" else 0,
        "last_loss": last_components,
        "log": log_rows,
    }


def benchmark_steps(
    model: C1SModel,
    diagnostic_heads: SuccessorDiagnosticHeads,
    provider: BatchProvider,
    schedule: Sequence[ScheduledBatch],
    *,
    steps: int,
    seed: int,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Disposable throughput check.  It never writes a model or artifact root."""

    if not 1 <= steps <= len(schedule):
        raise ValueError("benchmark steps outside supplied disposable schedule")
    short = list(schedule[:steps])
    short = [ScheduledBatch(index + 1, row.epoch, row.example_ids) for index, row in enumerate(short)]
    spec = TrainSpec(
        epochs=1,
        maximum_updates=steps,
        warmup_updates=min(10, steps),
        log_interval=max(1, steps),
        causal_interval=max(1, steps),
    )
    return train_fixed_endpoint(
        model,
        diagnostic_heads,
        provider,
        short,
        identity="DISPOSABLE_C1S_SUCCESSOR_BENCHMARK",
        order_seed=seed,
        model_seed=seed,
        spec=spec,
        device=device,
    )


__all__ = [
    "BatchProvider",
    "ScheduledBatch",
    "TrainSpec",
    "benchmark_steps",
    "build_balanced_schedule",
    "build_optimizer",
    "build_overfit_schedule",
    "configure_cpu_threads",
    "load_endpoint",
    "lr_scale",
    "save_endpoint",
    "schedule_report",
    "set_determinism",
    "train_fixed_endpoint",
]
