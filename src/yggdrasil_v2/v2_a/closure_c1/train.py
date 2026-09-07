from __future__ import annotations

"""Deterministic balanced schedules and C1 answer/CT1 optimization loops."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from . import artifacts, contract
from .deployment import DeploymentArtifact, strip_trace_probe


class BatchProvider(Protocol):
    def __call__(
        self, example_ids: Sequence[str], *, epoch: int, trace_chunk_tokens: int
    ) -> Mapping[str, Any]: ...


class EvaluationCallback(Protocol):
    def __call__(self, model: nn.Module, update: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ScheduledBatch:
    update: int
    epoch: int
    example_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrainSpec:
    batch_size: int = int(contract.TRAINING_CONFIG["batch_size"])
    family_batch_size: int = int(contract.TRAINING_CONFIG["family_batch_size"])
    epochs: int = int(contract.TRAINING_CONFIG["epochs"])
    maximum_updates: int = int(contract.TRAINING_CONFIG["maximum_updates"])
    minimum_selection_update: int = int(
        contract.TRAINING_CONFIG["minimum_selection_update"]
    )
    evaluation_interval: int = int(
        contract.TRAINING_CONFIG["evaluation_interval"]
    )
    trace_chunk_tokens: int = int(
        contract.TRAINING_CONFIG["trace_chunk_tokens"]
    )
    boundary_lr: float = float(contract.TRAINING_CONFIG["boundary_lr"])
    core_lr: float = float(contract.TRAINING_CONFIG["core_lr"])
    head_lr: float = float(contract.TRAINING_CONFIG["head_lr"])
    weight_decay: float = float(contract.TRAINING_CONFIG["weight_decay"])
    gradient_clip: float = float(contract.TRAINING_CONFIG["gradient_clip"])
    warmup_updates: int = int(contract.TRAINING_CONFIG["warmup_updates"])
    loss_switch_update: int = int(
        contract.TRAINING_CONFIG["loss_switch_update"]
    )


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_balanced_schedule(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = contract.ORDER_SEED,
    spec: TrainSpec = TrainSpec(),
) -> list[ScheduledBatch]:
    by_family: dict[str, list[str]] = {"ERE": [], "CPS": []}
    for record in records:
        family = record.get("family")
        example_id = record.get("example_id")
        if family not in by_family or not isinstance(example_id, str):
            raise ValueError("training schedule requires ERE/CPS example identities")
        by_family[family].append(example_id)
    for family in by_family:
        by_family[family].sort()
        if len(by_family[family]) != 4096:
            raise ValueError(f"C1 train schedule requires 4096 {family} records")
    if spec.batch_size != 2 * spec.family_batch_size:
        raise ValueError("C1 batch must contain equal ERE/CPS halves")
    if any(len(rows) % spec.family_batch_size for rows in by_family.values()):
        raise ValueError("family size must divide family batch size")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    schedule: list[ScheduledBatch] = []
    update = 0
    for epoch in range(spec.epochs):
        shuffled: dict[str, list[str]] = {}
        for family, rows in by_family.items():
            order = torch.randperm(len(rows), generator=generator).tolist()
            shuffled[family] = [rows[index] for index in order]
        family_batches = len(shuffled["ERE"]) // spec.family_batch_size
        for batch_index in range(family_batches):
            update += 1
            start = batch_index * spec.family_batch_size
            stop = start + spec.family_batch_size
            ids = tuple(
                shuffled["ERE"][start:stop] + shuffled["CPS"][start:stop]
            )
            schedule.append(ScheduledBatch(update, epoch, ids))
    if len(schedule) != spec.maximum_updates:
        raise ValueError(
            f"schedule generated {len(schedule)} updates, expected {spec.maximum_updates}"
        )
    flattened = [example_id for row in schedule for example_id in row.example_ids]
    for epoch in range(spec.epochs):
        section = flattened[
            epoch * 8192 : (epoch + 1) * 8192
        ]
        if len(section) != len(set(section)) or len(section) != 8192:
            raise ValueError("each C1 epoch must expose every train record exactly once")
    return schedule


def schedule_report(schedule: Sequence[ScheduledBatch]) -> dict[str, Any]:
    payload = [
        {"update": row.update, "epoch": row.epoch, "example_ids": list(row.example_ids)}
        for row in schedule
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "updates": len(schedule),
        "examples": sum(len(row.example_ids) for row in schedule),
        "epochs": 1 + max((row.epoch for row in schedule), default=-1),
        "sha256": hashlib.sha256(encoded).hexdigest().upper(),
        "rows": payload,
    }


def loss_weights(update: int) -> tuple[float, float]:
    if update <= int(contract.TRAINING_CONFIG["loss_switch_update"]):
        return (
            float(contract.TRAINING_CONFIG["early_answer_weight"]),
            float(contract.TRAINING_CONFIG["early_trace_weight"]),
        )
    return (
        float(contract.TRAINING_CONFIG["late_answer_weight"]),
        float(contract.TRAINING_CONFIG["late_trace_weight"]),
    )


def lr_scale(update: int, spec: TrainSpec = TrainSpec()) -> float:
    if update <= 0 or update > spec.maximum_updates:
        raise ValueError("update is outside the frozen schedule")
    if update <= spec.warmup_updates:
        return update / spec.warmup_updates
    progress = (update - spec.warmup_updates) / max(
        1, spec.maximum_updates - spec.warmup_updates
    )
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_optimizer(model: nn.Module, spec: TrainSpec = TrainSpec()) -> torch.optim.Optimizer:
    boundary: list[nn.Parameter] = []
    core: list[nn.Parameter] = []
    heads: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"duplicate parameter in C1 model: {name}")
        seen.add(id(parameter))
        if name.startswith("boundary") or name.startswith("slot") or name.startswith(
            "generic_slot"
        ):
            boundary.append(parameter)
        elif name.startswith("core") or name.startswith("recurrent"):
            core.append(parameter)
        else:
            heads.append(parameter)
    if not boundary or not core or not heads:
        raise ValueError(
            f"C1 optimizer groups incomplete: boundary={len(boundary)}, core={len(core)}, heads={len(heads)}"
        )
    groups = [
        {"params": boundary, "lr": spec.boundary_lr, "group_name": "boundary"},
        {"params": core, "lr": spec.core_lr, "group_name": "core"},
        {"params": heads, "lr": spec.head_lr, "group_name": "heads"},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=spec.weight_decay)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    return value


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _losses(model: nn.Module, batch: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    required = (
        "source_hidden",
        "source_mask",
        "answers",
        "trace_targets",
        "trace_mask",
        "trace_step_indices",
        "trace_global_positions",
        "trace_local_positions",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"C1 training batch missing {missing}")
    output = model(
        batch["source_hidden"], batch["source_mask"], return_trajectory=True
    )
    answer_loss = F.cross_entropy(output["logits"], batch["answers"].long())
    trace_logits = model.trace_logits(
        output["trajectory"],
        batch["trace_step_indices"].long(),
        batch["trace_global_positions"].long(),
        batch["trace_local_positions"].long(),
    )
    mask = batch["trace_mask"].bool()
    if trace_logits.shape[:2] != mask.shape or batch["trace_targets"].shape != mask.shape:
        raise ValueError("trace logits/target/mask shape mismatch")
    if not bool(mask.any().item()):
        raise ValueError("C1 trace batch cannot have an empty loss mask")
    trace_loss = F.cross_entropy(
        trace_logits[mask], batch["trace_targets"].long()[mask]
    )
    return answer_loss, trace_loss, output["logits"], trace_logits


def _finite_scalar(name: str, value: torch.Tensor) -> float:
    result = float(value.detach().float().cpu())
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite C1 {name}: {result}")
    return result


def save_checkpoint(model: nn.Module, path: Path, *, update: int, metrics: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.training-checkpoint.v1",
        "identity": contract.C1_IDENTITY,
        "update": update,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "metrics": dict(metrics),
    }
    torch.save(payload, path)
    return artifacts.sha256_file(path)


def load_checkpoint(model: nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != f"{contract.SCHEMA_PREFIX}.training-checkpoint.v1":
        raise ValueError("not a C1 training checkpoint")
    model.load_state_dict(payload["model_state"], strict=True)
    return {"update": int(payload["update"]), "metrics": payload["metrics"]}


def select_trace_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_update: int = int(contract.TRAINING_CONFIG["minimum_selection_update"]),
) -> Mapping[str, Any]:
    eligible = [
        row
        for row in candidates
        if int(row["update"]) >= minimum_update
    ]
    if not eligible:
        raise ValueError("no C1 checkpoint satisfies minimum selection update")
    for row in eligible:
        by_family = row.get("trace_nll_by_family")
        if not isinstance(by_family, Mapping) or set(by_family) != {"ERE", "CPS"}:
            raise ValueError("checkpoint selection requires ERE/CPS trace NLL")
        if any(not math.isfinite(float(value)) for value in by_family.values()):
            raise ValueError("checkpoint trace NLL must be finite")
    return min(
        eligible,
        key=lambda row: (
            max(float(row["trace_nll_by_family"][family]) for family in ("ERE", "CPS")),
            int(row["update"]),
        ),
    )


def train_primary(
    model: nn.Module,
    schedule: Sequence[ScheduledBatch],
    batch_provider: BatchProvider,
    evaluation_callback: EvaluationCallback,
    output_root: Path,
    *,
    device: str = "cuda",
    spec: TrainSpec = TrainSpec(),
) -> dict[str, Any]:
    if len(schedule) != spec.maximum_updates:
        raise ValueError("primary schedule length differs from frozen maximum updates")
    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C1 formal requested unavailable CUDA")
    set_determinism(contract.MODEL_SEED)
    model.to(active_device)
    optimizer = build_optimizer(model, spec)
    if active_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(active_device)
    history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    processed_trace_tokens = 0
    processed_source_tokens = 0
    for row in schedule:
        update = row.update
        model.train()
        optimizer.zero_grad(set_to_none=True)
        cpu_batch = batch_provider(
            row.example_ids,
            epoch=row.epoch,
            trace_chunk_tokens=spec.trace_chunk_tokens,
        )
        batch = {name: _move(value, active_device) for name, value in cpu_batch.items()}
        with _autocast(active_device):
            answer_loss, trace_loss, _, _ = _losses(model, batch)
            answer_weight, trace_weight = loss_weights(update)
            loss = answer_weight * answer_loss + trace_weight * trace_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), spec.gradient_clip
        )
        gradient_value = _finite_scalar("gradient norm", gradient_norm)
        scale = lr_scale(update, spec)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        optimizer.step()
        processed_trace_tokens += int(batch["trace_mask"].sum().detach().cpu())
        processed_source_tokens += int(batch["source_mask"].sum().detach().cpu())
        if update == 1 or update % 50 == 0:
            history.append(
                {
                    "update": update,
                    "epoch": row.epoch,
                    "loss": _finite_scalar("total loss", loss),
                    "answer_loss": _finite_scalar("answer loss", answer_loss),
                    "trace_loss": _finite_scalar("trace loss", trace_loss),
                    "gradient_norm": gradient_value,
                    "lr_scale": scale,
                }
            )
        if update % spec.evaluation_interval == 0:
            if active_device.type == "cuda":
                torch.cuda.synchronize(active_device)
            model.eval()
            metrics = dict(evaluation_callback(model, update))
            by_family = metrics.get("trace_nll_by_family")
            candidate = {
                "update": update,
                "trace_nll_by_family": dict(by_family) if isinstance(by_family, Mapping) else by_family,
                "validation": metrics,
            }
            checkpoint_path = output_root / "checkpoints" / f"update-{update:05d}.pt"
            candidate["checkpoint"] = checkpoint_path.relative_to(output_root).as_posix()
            candidate["checkpoint_sha256"] = save_checkpoint(
                model, checkpoint_path, update=update, metrics=metrics
            )
            candidates.append(candidate)
    if active_device.type == "cuda":
        torch.cuda.synchronize(active_device)
    selected = dict(
        select_trace_checkpoint(candidates, minimum_update=spec.minimum_selection_update)
    )
    selected_path = output_root / str(selected["checkpoint"])
    load_checkpoint(model, selected_path)
    wall_seconds = time.perf_counter() - started
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.primary-training.v1",
        "completed_updates": len(schedule),
        "processed_examples": len(schedule) * spec.batch_size,
        "processed_source_tokens": processed_source_tokens,
        "processed_trace_tokens": processed_trace_tokens,
        "wall_seconds": wall_seconds,
        "updates_per_second": len(schedule) / wall_seconds,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(active_device))
            if active_device.type == "cuda"
            else 0
        ),
        "history": history,
        "candidates": candidates,
        "selection_rule": "minimize max(ERE trace NLL, CPS trace NLL); tie earliest",
        "selected": selected,
        "passed": len(schedule) == spec.maximum_updates,
    }


def train_overfit32(
    model: nn.Module,
    example_ids: Sequence[str],
    batch_provider: BatchProvider,
    evaluation_callback: EvaluationCallback,
    output_root: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    if len(example_ids) != int(contract.OVERFIT_CONFIG["records"]):
        raise ValueError("C1 overfit control requires exactly 32 records")
    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C1 overfit requested unavailable CUDA")
    set_determinism(contract.OVERFIT_SEED)
    model.to(active_device)
    spec = TrainSpec(maximum_updates=int(contract.OVERFIT_CONFIG["maximum_updates"]))
    optimizer = build_optimizer(model, spec)
    generator = torch.Generator(device="cpu").manual_seed(contract.OVERFIT_SEED)
    history: list[dict[str, Any]] = []
    passed = False
    strip_report: dict[str, Any] = {}
    stripped_checksum: str | None = None
    checksum: str | None = None
    final_metrics: dict[str, Any] = {}
    started = time.perf_counter()
    for update in range(1, spec.maximum_updates + 1):
        permutation = torch.randperm(len(example_ids), generator=generator).tolist()
        offset = ((update - 1) * int(contract.OVERFIT_CONFIG["batch_size"])) % len(example_ids)
        indices = [permutation[(offset + index) % len(example_ids)] for index in range(int(contract.OVERFIT_CONFIG["batch_size"]))]
        ids = [example_ids[index] for index in indices]
        cpu_batch = batch_provider(ids, epoch=update - 1, trace_chunk_tokens=512)
        batch = {name: _move(value, active_device) for name, value in cpu_batch.items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(active_device):
            answer_loss, trace_loss, _, _ = _losses(model, batch)
            loss = 0.5 * answer_loss + trace_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale = min(1.0, update / 100.0)
        for group in optimizer.param_groups:
            group["lr"] = float(group["initial_lr"]) * scale
        optimizer.step()
        if update == 1 or update % 50 == 0:
            history.append(
                {
                    "update": update,
                    "loss": _finite_scalar("overfit loss", loss),
                    "answer_loss": _finite_scalar("overfit answer loss", answer_loss),
                    "trace_loss": _finite_scalar("overfit trace loss", trace_loss),
                    "gradient_norm": _finite_scalar("overfit gradient norm", gradient_norm),
                }
            )
        if update % int(contract.OVERFIT_CONFIG["evaluation_interval"]) == 0:
            model.eval()
            final_metrics = dict(evaluation_callback(model, update))
            raw_passed = (
                float(final_metrics.get("answer_accuracy", -1.0))
                >= float(contract.OVERFIT_CONFIG["answer_accuracy"])
                and float(final_metrics.get("trace_token_accuracy", -1.0))
                >= float(contract.OVERFIT_CONFIG["trace_token_accuracy"])
                and float(final_metrics.get("trace_content_accuracy", -1.0))
                >= float(contract.OVERFIT_CONFIG["trace_content_accuracy"])
                and float(final_metrics.get("owner_nll_margin", -1.0))
                >= float(contract.OVERFIT_CONFIG["owner_nll_margin"])
            )
            if raw_passed:
                strip_report, stripped = overfit_strip_integrity(
                    model,
                    example_ids,
                    batch_provider,
                    device=active_device,
                )
                passed = (
                    strip_report["correct_before"] == len(example_ids)
                    and strip_report["correct_after"] == len(example_ids)
                    and strip_report["prediction_invariance"] == 1.0
                    and strip_report["max_abs_diff"]
                    <= float(contract.OVERFIT_CONFIG["strip_logit_max_abs_diff"])
                )
            if passed:
                checkpoint_path = output_root / "overfit32" / "checkpoint.pt"
                checksum = save_checkpoint(
                    model, checkpoint_path, update=update, metrics=final_metrics
                )
                stripped_path = output_root / "overfit32" / "stripped-state.pt"
                torch.save(
                    {
                        "schema_version": f"{contract.SCHEMA_PREFIX}.overfit32-stripped.v1",
                        "identity": contract.C1_IDENTITY,
                        "model_state": stripped.state_dict,
                        "strip_report": strip_report,
                    },
                    stripped_path,
                )
                stripped_checksum = artifacts.sha256_file(stripped_path)
                break
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.overfit32.v1",
        "passed": passed,
        "completed_updates": update,
        "maximum_updates": spec.maximum_updates,
        "metrics": final_metrics,
        "strip_integrity": strip_report,
        "history": history,
        "wall_seconds": time.perf_counter() - started,
        "checkpoint": (
            "overfit32/checkpoint.pt" if passed else None
        ),
        "checkpoint_sha256": checksum if passed else None,
        "stripped_state": "overfit32/stripped-state.pt" if passed else None,
        "stripped_state_sha256": stripped_checksum if passed else None,
    }


def overfit_strip_integrity(
    model: nn.Module,
    example_ids: Sequence[str],
    batch_provider: BatchProvider,
    *,
    device: torch.device | str,
) -> tuple[dict[str, Any], DeploymentArtifact]:
    """Prove that physical probe removal preserves all 32 answer logits."""
    active_device = torch.device(device)
    artifact = strip_trace_probe(model)
    artifact.model.to(active_device).eval()
    model.eval()
    correct_before = correct_after = same = total = 0
    maximum = 0.0
    batch_size = int(contract.OVERFIT_CONFIG["batch_size"])
    with torch.inference_mode():
        for start in range(0, len(example_ids), batch_size):
            ids = example_ids[start : start + batch_size]
            cpu_batch = batch_provider(ids, epoch=0, trace_chunk_tokens=512)
            source = _move(cpu_batch["source_hidden"], active_device)
            mask = _move(cpu_batch["source_mask"], active_device)
            answers = _move(cpu_batch["answers"], active_device).long()
            before = model(source, mask, return_trajectory=False)["logits"].float()
            after = artifact.model(source, mask, return_trajectory=False)["logits"].float()
            difference = (before - after).abs()
            maximum = max(maximum, float(difference.max().cpu()))
            before_prediction = before.argmax(dim=-1)
            after_prediction = after.argmax(dim=-1)
            correct_before += int((before_prediction == answers).sum().cpu())
            correct_after += int((after_prediction == answers).sum().cpu())
            same += int((before_prediction == after_prediction).sum().cpu())
            total += len(ids)
    report = {
        "records": total,
        "correct_before": correct_before,
        "correct_after": correct_after,
        "prediction_invariance": same / total if total else 0.0,
        "max_abs_diff": maximum,
        "tolerance": float(contract.OVERFIT_CONFIG["strip_logit_max_abs_diff"]),
        "trace_probe_parameters_after": 0,
        "passed": bool(total)
        and correct_before == total
        and correct_after == total
        and same == total
        and maximum <= float(contract.OVERFIT_CONFIG["strip_logit_max_abs_diff"]),
    }
    return report, artifact


__all__ = [
    "BatchProvider",
    "EvaluationCallback",
    "ScheduledBatch",
    "TrainSpec",
    "build_balanced_schedule",
    "build_optimizer",
    "load_checkpoint",
    "loss_weights",
    "lr_scale",
    "overfit_strip_integrity",
    "save_checkpoint",
    "schedule_report",
    "select_trace_checkpoint",
    "set_determinism",
    "train_overfit32",
    "train_primary",
]
