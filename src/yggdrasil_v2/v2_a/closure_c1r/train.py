from __future__ import annotations

"""Staged C1 repair trainer.

This module is deliberately independent from the consumed ``closure_c1``
runner.  Stage A learns only the deployment answer path.  Stage B consumes a
frozen trajectory cache and trains only the training-time trace probe.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..closure_c1 import contract as c1_contract
from . import contract as c1r_contract
from ..closure_c1.model import C1Config, C1Model


SCHEMA = "yggdrasil.v2-a.closure-c1r.staged-trainer.v1"
TRACE_BLOCK_WIDTH = 65


@dataclass(frozen=True)
class ScheduledBatch:
    update: int
    epoch: int
    example_ids: tuple[str, ...]


@dataclass(frozen=True)
class StageASpec:
    batch_size: int = 8
    family_batch_size: int = 4
    epochs: int = 6
    maximum_updates: int = 6144
    boundary_lr: float = 1.0e-4
    core_lr: float = 2.0e-4
    head_lr: float = 3.0e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_updates: int = 256
    max_prefetch: int = 2


@dataclass(frozen=True)
class StageBSpec:
    batch_size: int = 32
    family_batch_size: int = 16
    epochs: int = 6
    maximum_updates: int = 1536
    trace_block_width: int = TRACE_BLOCK_WIDTH
    trace_lr: float = 3.0e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_updates: int = 256
    prefetch_workers: int = 0
    max_prefetch: int = 2


# Names used by callers that describe this as C1R rather than staged C1.
C1RStageASpec = StageASpec
C1RStageBSpec = StageBSpec


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _records_by_family(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    by_family: dict[str, list[str]] = {"ERE": [], "CPS": []}
    for record in records:
        family = record.get("family")
        example_id = record.get("example_id")
        if family not in by_family or not isinstance(example_id, str):
            raise ValueError("C1R schedule requires ERE/CPS example identities")
        if example_id in by_family[family]:
            raise ValueError(f"duplicate C1R example identity: {example_id}")
        by_family[family].append(example_id)
    if not by_family["ERE"] or len(by_family["ERE"]) != len(by_family["CPS"]):
        raise ValueError("C1R requires equal non-empty ERE/CPS populations")
    for rows in by_family.values():
        rows.sort()
    return by_family


def build_balanced_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = c1_contract.ORDER_SEED,
    spec: StageASpec | StageBSpec = StageASpec(),
) -> list[ScheduledBatch]:
    """Build a deterministic family-balanced schedule with exact epoch coverage."""
    by_family = _records_by_family(records)
    if spec.batch_size != 2 * spec.family_batch_size:
        raise ValueError("C1R batch must contain equal ERE/CPS halves")
    if any(len(rows) % spec.family_batch_size for rows in by_family.values()):
        raise ValueError("family size must divide family batch size")
    batches_per_epoch = len(by_family["ERE"]) // spec.family_batch_size
    expected_updates = spec.epochs * batches_per_epoch
    if expected_updates != spec.maximum_updates:
        raise ValueError(
            f"schedule generated {expected_updates} updates, expected {spec.maximum_updates}"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule: list[ScheduledBatch] = []
    update = 0
    for epoch in range(spec.epochs):
        shuffled = {
            family: [rows[i] for i in torch.randperm(len(rows), generator=generator).tolist()]
            for family, rows in by_family.items()
        }
        for batch_index in range(batches_per_epoch):
            start = batch_index * spec.family_batch_size
            stop = start + spec.family_batch_size
            update += 1
            schedule.append(
                ScheduledBatch(
                    update,
                    epoch,
                    tuple(shuffled["ERE"][start:stop] + shuffled["CPS"][start:stop]),
                )
            )
    for epoch in range(spec.epochs):
        ids = [x for row in schedule if row.epoch == epoch for x in row.example_ids]
        if len(ids) != len(set(ids)) or set(ids) != set(by_family["ERE"] + by_family["CPS"]):
            raise AssertionError("C1R schedule does not expose each record exactly once per epoch")
    return schedule


def schedule_report(schedule: Sequence[ScheduledBatch]) -> dict[str, Any]:
    rows = [asdict(row) for row in schedule]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "updates": len(rows),
        "epochs": 1 + max((row.epoch for row in schedule), default=-1),
        "examples": sum(len(row.example_ids) for row in schedule),
        "sha256": hashlib.sha256(encoded).hexdigest().upper(),
        "rows": rows,
    }


def cosine_lr_scale(update: int, maximum_updates: int, warmup_updates: int) -> float:
    if update < 1 or update > maximum_updates:
        raise ValueError("update outside C1R schedule")
    if warmup_updates > 0 and update <= warmup_updates:
        return update / warmup_updates
    progress = (update - warmup_updates) / max(1, maximum_updates - warmup_updates)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def parameter_hash(module: nn.Module, *, names: Iterable[str] | None = None) -> str:
    allowed = None if names is None else set(names)
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if allowed is not None and name not in allowed:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def _finite_gradient_norm(value: torch.Tensor | float) -> float:
    number = (
        float(value.detach().cpu())
        if isinstance(value, torch.Tensor)
        else float(value)
    )
    if not math.isfinite(number):
        raise FloatingPointError("non-finite C1R gradient norm")
    return number


def _parameters_are_finite(
    module: nn.Module, *, trainable_only: bool = False
) -> bool:
    for parameter in module.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        if not bool(torch.isfinite(parameter.detach()).all()):
            return False
    return True


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _autocast(device: torch.device):
    """Use the C1R CUDA compute dtype without affecting CPU test doubles."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _call_provider(provider: Callable[..., Any], ids: Sequence[str], epoch: int) -> Any:
    """Call providers with the small public C1R contract, tolerating positional test doubles."""
    try:
        parameters = inspect.signature(provider).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs: dict[str, Any] = {}
    if "epoch" in parameters or not parameters:
        kwargs["epoch"] = epoch
    try:
        return provider(ids, **kwargs)
    except TypeError as error:
        if kwargs:
            return provider(ids)
        raise error


class BoundedOrderedPrefetch:
    """Bounded, ordered CPU prefetch; at most ``max_prefetch`` futures exist."""

    def __init__(self, items: Iterable[Any], loader: Callable[[Any], Any], *, max_prefetch: int = 2):
        if max_prefetch < 1:
            raise ValueError("max_prefetch must be positive")
        self.items = iter(items)
        self.loader = loader
        self.max_prefetch = max_prefetch

    def __iter__(self) -> Iterator[Any]:
        # Keep only a bounded number of CPU jobs alive, while yielding in
        # input order even when later jobs finish first.
        pending: deque[Any] = deque()
        with ThreadPoolExecutor(max_workers=self.max_prefetch, thread_name_prefix="c1r-prefetch") as executor:
            for _ in range(self.max_prefetch):
                try:
                    pending.append(executor.submit(self.loader, next(self.items)))
                except StopIteration:
                    break
            while pending:
                yield pending.popleft().result()
                try:
                    pending.append(executor.submit(self.loader, next(self.items)))
                except StopIteration:
                    pass


def ordered_prefetch(items: Iterable[Any], loader: Callable[[Any], Any], *, max_prefetch: int = 2) -> Iterator[Any]:
    return iter(BoundedOrderedPrefetch(items, loader, max_prefetch=max_prefetch))


def _answer_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    required = ("source_hidden", "source_mask", "answers")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Stage A answer batch missing {missing}")
    moved = {key: _move(value, device) for key, value in batch.items()}
    for key in required:
        if not isinstance(moved[key], torch.Tensor):
            moved[key] = torch.as_tensor(moved[key], device=device)
    return {key: moved[key] for key in required}


def _build_stage_a_optimizer(model: nn.Module, spec: StageASpec) -> torch.optim.Optimizer:
    boundary: list[nn.Parameter] = []
    core: list[nn.Parameter] = []
    heads: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("boundary") or name.startswith("slot"):
            boundary.append(parameter)
        elif name.startswith("core") or name.startswith("recurrent"):
            core.append(parameter)
        else:
            heads.append(parameter)
    if not boundary or not core or not heads:
        raise ValueError("Stage A optimizer groups are incomplete")
    optimizer = torch.optim.AdamW(
        [
            {"params": boundary, "lr": spec.boundary_lr, "group_name": "boundary"},
            {"params": core, "lr": spec.core_lr, "group_name": "core"},
            {"params": heads, "lr": spec.head_lr, "group_name": "heads"},
        ],
        weight_decay=spec.weight_decay,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def _save_checkpoint(model: nn.Module, output_root: Path, *, stage: str, update: int, accounting: Mapping[str, Any]) -> Path:
    resolved = output_root.resolve()
    legacy_root = (Path.cwd() / c1_contract.C1_OUTPUT_ROOT).resolve()
    if resolved == legacy_root or legacy_root in resolved.parents:
        raise ValueError("C1R refuses to write inside the consumed closure_c1 root")
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "checkpoints" / f"{stage}-final.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA,
        "stage": stage,
        "update": update,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "state_sha256": state_hash(model),
        "accounting": dict(accounting),
    }
    torch.save(payload, path)
    return path


def train_stage_a(
    model: C1Model,
    records: Sequence[Mapping[str, Any]],
    batch_provider: Callable[..., Mapping[str, Any]],
    output_root: Path,
    *,
    device: str | torch.device = "cpu",
    spec: StageASpec = StageASpec(),
    model_seed: int = c1_contract.MODEL_SEED,
    order_seed: int = c1_contract.ORDER_SEED,
) -> dict[str, Any]:
    """Train answer-only C1R Stage A and save only the final endpoint."""
    if model.has_trace_probe:
        raise ValueError("Stage A requires C1Model(with_trace_probe=False)")
    schedule = build_balanced_schedule(records, seed=order_seed, spec=spec)
    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage A requested unavailable CUDA")
    set_determinism(model_seed)
    model.to(active_device)
    optimizer = _build_stage_a_optimizer(model, spec)
    history: list[dict[str, Any]] = []
    processed_source_tokens = 0
    parameter_finite_checks = 0
    model.train()
    prefetched = ordered_prefetch(
        schedule,
        lambda scheduled: _call_provider(batch_provider, scheduled.example_ids, scheduled.epoch),
        max_prefetch=spec.max_prefetch,
    )
    for row, cpu_batch in zip(schedule, prefetched):
        batch = _answer_batch(cpu_batch, active_device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(active_device):
            output = model(batch["source_hidden"], batch["source_mask"], return_trajectory=False)
            loss = F.cross_entropy(output["logits"], batch["answers"].long())
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Stage A answer loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
        gradient_norm_value = _finite_gradient_norm(gradient_norm)
        scale = cosine_lr_scale(row.update, spec.maximum_updates, spec.warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        optimizer.step()
        processed_source_tokens += int(batch["source_mask"].sum().detach().cpu())
        if row.update == 1 or row.update == spec.maximum_updates or row.update % max(1, spec.maximum_updates // 8) == 0:
            parameter_finite = _parameters_are_finite(model)
            parameter_finite_checks += 1
            if not parameter_finite:
                raise FloatingPointError("non-finite Stage A parameter state")
            history.append({"update": row.update, "epoch": row.epoch, "loss": float(loss.detach().cpu()), "gradient_norm": gradient_norm_value, "parameter_state_finite": parameter_finite, "lr_scale": scale})
    accounting = {
        "identity": c1r_contract.C1R_IDENTITY,
        "schema_version": SCHEMA,
        "stage": "A",
        "epochs": spec.epochs,
        "updates": len(schedule),
        "optimizer_steps": len(schedule),
        "processed_examples": sum(len(row.example_ids) for row in schedule),
        "processed_source_tokens": processed_source_tokens,
        "gradient_norms_finite": True,
        "parameter_state_finite": True,
        "parameter_finite_checks": parameter_finite_checks,
        "trace_loss": False,
        "checkpoint_selection": "fixed_final_endpoint",
        "schedule": schedule_report(schedule),
        "history": history,
        "completed_updates": len(schedule),
        "training_started": True,
        "model_writes": 1,
        "passed": True,
    }
    checkpoint = _save_checkpoint(model, output_root, stage="stage-a", update=len(schedule), accounting=accounting)
    accounting["checkpoint"] = str(checkpoint)
    accounting["state_sha256"] = state_hash(model)
    return accounting


def train_answer_overfit32(
    model: C1Model,
    example_ids: Sequence[str],
    batch_provider: Callable[..., Mapping[str, Any]],
    output_root: Path,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    updates: int = 4000,
    evaluation_interval: int = 100,
    model_seed: int = c1_contract.OVERFIT_SEED,
) -> dict[str, Any]:
    """Answer-only overfit control used before a Stage A full schedule."""
    if len(example_ids) != 32:
        raise ValueError("answer-only overfit requires exactly 32 records")
    if evaluation_interval not in (50, 100):
        raise ValueError("answer-only overfit evaluation interval must be 50 or 100")
    if model.has_trace_probe:
        raise ValueError("answer-only overfit requires a stripped C1Model")
    active_device = torch.device(device)
    set_determinism(model_seed)
    model.to(active_device)
    spec = StageASpec(batch_size=batch_size, family_batch_size=batch_size // 2, epochs=1, maximum_updates=updates, warmup_updates=min(100, updates))
    optimizer = _build_stage_a_optimizer(model, spec)
    generator = torch.Generator(device="cpu").manual_seed(model_seed)
    last_loss = math.inf
    for update in range(1, updates + 1):
        order = torch.randperm(32, generator=generator).tolist()
        start = ((update - 1) * batch_size) % 32
        ids = [example_ids[order[(start + index) % 32]] for index in range(batch_size)]
        batch = _answer_batch(_call_provider(batch_provider, ids, update - 1), active_device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(active_device):
            logits = model(batch["source_hidden"], batch["source_mask"], return_trajectory=False)["logits"]
            loss = F.cross_entropy(logits, batch["answers"].long())
        last_loss = float(loss.detach().cpu())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
        _finite_gradient_norm(gradient_norm)
        scale = cosine_lr_scale(update, updates, spec.warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        optimizer.step()
        if update % evaluation_interval == 0:
            with torch.inference_mode(), _autocast(active_device):
                eval_batch = _answer_batch(_call_provider(batch_provider, list(example_ids), update), active_device)
                eval_logits = model(eval_batch["source_hidden"], eval_batch["source_mask"], return_trajectory=False)["logits"]
                accuracy = float((eval_logits.argmax(-1) == eval_batch["answers"].long()).float().mean().cpu())
            if accuracy == 1.0:
                break
    completed_updates = update
    parameter_state_finite = _parameters_are_finite(model)
    if not parameter_state_finite:
        raise FloatingPointError("non-finite answer-overfit parameter state")
    with torch.inference_mode(), _autocast(active_device):
        batch = _answer_batch(_call_provider(batch_provider, list(example_ids), completed_updates), active_device)
        predictions = model(batch["source_hidden"], batch["source_mask"], return_trajectory=False)["logits"].argmax(-1)
        accuracy = float((predictions == batch["answers"].long()).float().mean().cpu())
    accounting = {"stage": "A-overfit32", "updates": completed_updates, "maximum_updates": updates, "evaluation_interval": evaluation_interval, "optimizer_steps": completed_updates, "answer_accuracy": accuracy, "final_loss": last_loss, "gradient_norms_finite": True, "parameter_state_finite": parameter_state_finite, "passed": accuracy == 1.0}
    accounting.update({"identity": c1r_contract.C1R_IDENTITY, "schema_version": SCHEMA, "completed_updates": completed_updates, "training_started": True, "model_writes": 1})
    checkpoint = _save_checkpoint(model, output_root, stage="stage-a-overfit32", update=completed_updates, accounting=accounting)
    accounting["checkpoint"] = str(checkpoint)
    return accounting


def _freeze_to_trace_probe(model: C1Model) -> tuple[str, ...]:
    if not model.has_trace_probe:
        raise ValueError("Stage B requires C1Model(with_trace_probe=True)")
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("trace_probe.")
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable or any(not name.startswith("trace_probe.") for name in trainable):
        raise AssertionError("Stage B freeze did not isolate trace_probe")
    return tuple(trainable)


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"trace target row missing one of {names}")


def _as_1d_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().cpu().reshape(-1).to(dtype=dtype)
    return tensor


def _normalise_target_rows(rows: Any, width: int) -> list[dict[str, torch.Tensor]]:
    if isinstance(rows, Mapping):
        # A columnar cache row is accepted in addition to a list of records.
        lengths = [len(torch.as_tensor(rows[key]).reshape(-1)) for key in ("targets", "trace_targets") if key in rows]
        if not lengths:
            raise KeyError("target rows require targets/trace_targets")
        count = lengths[0]
        rows = [{key: value[index] for key, value in rows.items()} for index in range(count)]
    result: list[dict[str, torch.Tensor]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("target rows must be mappings")
        targets = _as_1d_tensor(_row_value(row, "targets", "trace_targets", "target"), dtype=torch.long)
        mask_value = row.get("mask", row.get("trace_mask", torch.ones_like(targets, dtype=torch.bool)))
        mask = _as_1d_tensor(mask_value, dtype=torch.bool)
        step_indices = _as_1d_tensor(_row_value(row, "step_indices", "trace_step_indices"), dtype=torch.long)
        global_positions = _as_1d_tensor(_row_value(row, "global_positions", "trace_global_positions"), dtype=torch.long)
        local_positions = _as_1d_tensor(_row_value(row, "local_positions", "trace_local_positions"), dtype=torch.long)
        fields = (targets, mask, step_indices, global_positions, local_positions)
        if any(field.numel() != targets.numel() for field in fields) or targets.numel() < 1:
            raise ValueError("C1R target row fields must have equal non-empty width")
        # Callers may provide a complete target row.  The trainer owns the
        # fixed 65-token blocking rule and therefore performs the split here.
        for start in range(0, targets.numel(), width):
            stop = min(targets.numel(), start + width)
            result.append({
                "targets": targets[start:stop],
                "mask": mask[start:stop],
                "step_indices": step_indices[start:stop],
                "global_positions": global_positions[start:stop],
                "local_positions": local_positions[start:stop],
            })
    if not result:
        raise ValueError("C1R records cannot have empty target rows")
    return result


def _cache_item(cache: Mapping[str, Any] | Callable[..., Any], example_id: str, epoch: int) -> Any:
    if isinstance(cache, Mapping):
        return cache[example_id]
    return _call_provider(cache, [example_id], epoch)


def _extract_trajectory_and_rows(item: Any, record: Mapping[str, Any], width: int) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    if isinstance(item, Mapping):
        trajectory = item.get("trajectory", item.get("latent_trajectory"))
        rows = item.get("target_rows", item.get("trace_rows"))
    else:
        trajectory, rows = item, None
    if rows is None:
        rows = record.get("target_rows", record.get("trace_rows"))
    if trajectory is None or rows is None:
        raise KeyError("C1R cache item requires trajectory and target_rows")
    trajectory_tensor = trajectory if isinstance(trajectory, torch.Tensor) else torch.as_tensor(trajectory)
    if trajectory_tensor.ndim != 3:
        raise ValueError("trajectory cache rows must have shape [T, slots, width]")
    return trajectory_tensor.detach().cpu(), _normalise_target_rows(rows, width)


def _stage_b_batch(records: Sequence[Mapping[str, Any]], cache: Mapping[str, Any] | Callable[..., Any], epoch: int, width: int) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    blocks: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for record in records:
        trajectory, rows = _extract_trajectory_and_rows(_cache_item(cache, str(record["example_id"]), epoch), record, width)
        positions: list[int] = []
        for row in rows:
            positions.extend(int(value) for value in row["global_positions"].tolist())
            blocks.append((trajectory, row))
        if len(positions) != len(set(positions)):
            raise ValueError(f"duplicate target token position in {record['example_id']}")
    return blocks


def _padded_block_minibatches(
    blocks: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    *,
    max_blocks: int = 64,
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Pack blocks without changing their token-mass contribution.

    Each yielded target tensor is padded only to the longest block in that
    minibatch.  ``mask`` excludes padding, while all real blocks remain in
    the schedule batch and share its denominator.
    """
    if max_blocks < 1:
        raise ValueError("max_blocks must be positive")
    for start in range(0, len(blocks), max_blocks):
        group = blocks[start : start + max_blocks]
        max_width = max(item[1]["targets"].numel() for item in group)
        trajectory = torch.stack([item[0] for item in group])
        fields: dict[str, torch.Tensor] = {}
        for key in ("targets", "step_indices", "global_positions", "local_positions"):
            values = torch.zeros((len(group), max_width), dtype=group[0][1][key].dtype)
            if key == "step_indices":
                values.fill_(1)  # the probe validates alignment before masking
            for index, (_, row) in enumerate(group):
                value = row[key]
                values[index, : value.numel()] = value
            fields[key] = values
        mask = torch.zeros((len(group), max_width), dtype=torch.bool)
        for index, (_, row) in enumerate(group):
            value = row["mask"]
            mask[index, : value.numel()] = value
        fields["mask"] = mask
        yield trajectory, fields


def train_stage_b(
    model: C1Model,
    records: Sequence[Mapping[str, Any]],
    trajectory_cache: Mapping[str, Any] | Callable[..., Any],
    output_root: Path,
    *,
    device: str | torch.device = "cpu",
    spec: StageBSpec = StageBSpec(),
    model_seed: int = c1_contract.MODEL_SEED,
    order_seed: int = c1_contract.ORDER_SEED,
) -> dict[str, Any]:
    """Train only ``trace_probe`` from frozen trajectories, one step per record batch."""
    trainable_names = _freeze_to_trace_probe(model)
    if spec.trace_block_width != TRACE_BLOCK_WIDTH:
        raise ValueError("C1R fixes target block width at 65")
    schedule = build_balanced_schedule(records, seed=order_seed, spec=spec)
    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage B requested unavailable CUDA")
    set_determinism(model_seed)
    model.to(active_device)
    frozen_before = {name: parameter_hash(model, names=[name]) for name, parameter in model.named_parameters() if not parameter.requires_grad}
    optimizer = torch.optim.AdamW(
        [{"params": [parameter for name, parameter in model.named_parameters() if name in trainable_names], "lr": spec.trace_lr, "initial_lr": spec.trace_lr}],
        weight_decay=spec.weight_decay,
    )
    history: list[dict[str, Any]] = []
    tokens_by_epoch: dict[int, int] = {}
    blocks_by_epoch: dict[int, int] = {}
    model.train()
    record_by_id = {str(record["example_id"]): record for record in records}
    prefetched = ordered_prefetch(
        schedule,
        lambda scheduled: _stage_b_batch(
            [record_by_id[example_id] for example_id in scheduled.example_ids],
            trajectory_cache,
            scheduled.epoch,
            spec.trace_block_width,
        ),
        max_prefetch=spec.max_prefetch,
    )
    optimizer_steps = 0
    trace_forward_calls = 0
    parameter_finite_checks = 0
    for row, blocks in zip(schedule, prefetched):
        optimizer.zero_grad(set_to_none=True)
        valid_tokens = sum(int(target["mask"].sum()) for _, target in blocks)
        if not blocks or valid_tokens < 1:
            raise ValueError("Stage B batch has no valid target tokens")
        batch_numerator = torch.zeros((), dtype=torch.float32, device=active_device)
        # Every packed minibatch is scaled by the schedule batch's total
        # valid-token count.  Sequential backward calls therefore equal one
        # backward of the concatenated sum, while retaining one optimizer step.
        for trajectory, target in _padded_block_minibatches(blocks, max_blocks=64):
            trajectory = trajectory.to(active_device)
            args = {key: target[key].to(active_device) for key in ("step_indices", "global_positions", "local_positions")}
            with _autocast(active_device):
                logits = model.trace_logits(trajectory, args["step_indices"], args["global_positions"], args["local_positions"])
                mask = target["mask"].to(active_device).bool()
                numerator = F.cross_entropy(logits[mask], target["targets"].to(active_device)[mask], reduction="sum")
                loss = numerator / valid_tokens
            batch_numerator.add_(numerator.detach().float())
            trace_forward_calls += 1
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Stage B trace loss")
            loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], spec.gradient_clip)
        gradient_norm_value = _finite_gradient_norm(gradient_norm)
        scale = cosine_lr_scale(row.update, spec.maximum_updates, spec.warmup_updates)
        optimizer.param_groups[0]["lr"] = spec.trace_lr * scale
        optimizer.step()
        optimizer_steps += 1
        tokens_by_epoch[row.epoch] = tokens_by_epoch.get(row.epoch, 0) + valid_tokens
        blocks_by_epoch[row.epoch] = blocks_by_epoch.get(row.epoch, 0) + len(blocks)
        if row.update == 1 or row.update == spec.maximum_updates or row.update % max(1, spec.maximum_updates // 8) == 0:
            parameter_finite = _parameters_are_finite(model, trainable_only=True)
            parameter_finite_checks += 1
            if not parameter_finite:
                raise FloatingPointError("non-finite Stage B probe parameter state")
            history.append({"update": row.update, "epoch": row.epoch, "loss": float((batch_numerator / valid_tokens).cpu()), "valid_tokens": valid_tokens, "blocks": len(blocks), "gradient_norm": gradient_norm_value, "parameter_state_finite": parameter_finite, "lr_scale": scale})
    frozen_after = {name: parameter_hash(model, names=[name]) for name, parameter in model.named_parameters() if not parameter.requires_grad}
    if frozen_before != frozen_after:
        raise AssertionError("Stage B modified a frozen model parameter")
    expected_tokens = {epoch: tokens_by_epoch[epoch] for epoch in range(spec.epochs)}
    if any(value != expected_tokens[0] for value in expected_tokens.values()):
        raise AssertionError("Stage B did not expose every target token once per epoch")
    accounting = {
        "identity": c1r_contract.C1R_IDENTITY,
        "schema_version": SCHEMA,
        "stage": "B",
        "epochs": spec.epochs,
        "updates": len(schedule),
        "optimizer_steps": optimizer_steps,
        "batch_size": spec.batch_size,
        "trace_block_width": spec.trace_block_width,
        "trace_max_blocks_per_forward": 64,
        "trace_forward_calls": trace_forward_calls,
        "gradient_norms_finite": True,
        "parameter_state_finite": True,
        "parameter_finite_checks": parameter_finite_checks,
        "trace_blocks": sum(blocks_by_epoch.values()),
        "processed_target_tokens_by_epoch": expected_tokens,
        "processed_blocks_by_epoch": blocks_by_epoch,
        "trainable_parameters": list(trainable_names),
        "frozen_parameter_hashes_unchanged": frozen_before == frozen_after,
        "one_optimizer_step_per_batch": optimizer_steps == len(schedule),
        "schedule": schedule_report(schedule),
        "history": history,
        "completed_updates": len(schedule),
        "training_started": True,
        "model_writes": 1,
        "passed": True,
    }
    checkpoint = _save_checkpoint(model, output_root, stage="stage-b", update=len(schedule), accounting=accounting)
    accounting["checkpoint"] = str(checkpoint)
    accounting["state_sha256"] = state_hash(model)
    return accounting


__all__ = [
    "SCHEMA",
    "TRACE_BLOCK_WIDTH",
    "ScheduledBatch",
    "StageASpec",
    "StageBSpec",
    "C1RStageASpec",
    "C1RStageBSpec",
    "BoundedOrderedPrefetch",
    "ordered_prefetch",
    "state_hash",
    "parameter_hash",
    "build_balanced_schedule",
    "schedule_report",
    "cosine_lr_scale",
    "train_answer_overfit32",
    "train_stage_a",
    "train_stage_b",
    "answer_only_overfit32",
    "train_frozen_probe",
]


# Explicit aliases make the two causal stages easy to wire into a separate
# runner without importing or modifying the consumed closure_c1 runner.
answer_only_overfit32 = train_answer_overfit32
train_frozen_probe = train_stage_b
