from __future__ import annotations

"""One-shot orchestration for the non-formal H1-WD mechanism screen."""

from copy import deepcopy
from dataclasses import asdict
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
from yggdrasil_v2.r1_revalidation.h1.metrics import (
    classification_metrics,
    primary_comparison,
)
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
    OUTPUT_ROOT,
    SCHEMA_PREFIX,
    SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_PACKAGE_IDENTITY,
    SOURCE_ROOT,
    SOURCE_TOKEN_CACHE,
    SPEC,
    WriteDeleteSpec,
    contract_manifest,
)
from .core import (
    collect_transitions,
    forward_write_delete,
    freeze_for_role,
    normalized_mse,
    overlap_write_target,
    selected_projection,
)


ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _save_model(model: H1LatentReasoner, path: Path, *, stage: str) -> dict[str, Any]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.deployment_state().items()
    }
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
    return {
        "path": path.name,
        "sha256": _sha256_path(path),
        "stage": stage,
    }


def _lr_scale(update: int, updates: int, spec: WriteDeleteSpec) -> float:
    warmup = max(1, int(updates * spec.warmup_fraction))
    if update <= warmup:
        return update / warmup
    progress = (update - warmup) / max(1, updates - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return spec.minimum_learning_rate_scale + (
        1.0 - spec.minimum_learning_rate_scale
    ) * cosine


def build_unlabeled_schedule(
    count: int, *, updates: int, batch_size: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a deterministic schedule without consulting family/task labels."""

    if count <= 0 or updates <= 0 or batch_size <= 0:
        raise ValueError("H1-WD schedule dimensions must be positive")
    generator = np.random.default_rng(seed)
    needed = updates * batch_size
    pieces: list[np.ndarray] = []
    available = 0
    while available < needed:
        permutation = generator.permutation(count).astype(np.int64, copy=False)
        pieces.append(permutation)
        available += count
    flat = np.concatenate(pieces)[:needed]
    schedule = flat.reshape(updates, batch_size)
    exposure = np.bincount(flat, minlength=count)
    report = {
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
    return schedule, report


def _batch(
    cache: TokenCache,
    prepared: PreparedSplit,
    local_indices: np.ndarray,
    *,
    device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    hidden, mask = cache.batch(prepared.cache_indices[local_indices], device=device)
    targets = torch.from_numpy(prepared.targets[local_indices].copy()).to(device)
    return hidden, mask, targets


def _stage_optimizer(
    parameters: Sequence[Tensor], *, learning_rate: float, spec: WriteDeleteSpec
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=spec.weight_decay,
    )


def train_write_stage(
    student: H1LatentReasoner,
    teacher: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    schedule: np.ndarray,
    spec: WriteDeleteSpec,
    *,
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Write only predecessor-aligned common content into the routed branch."""

    device = spec.device
    student.to(device).train()
    teacher.to(device).eval()
    parameters = freeze_for_role(student, "projection")
    optimizer = _stage_optimizer(
        parameters, learning_rate=spec.write_learning_rate, spec=spec
    )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update, local_indices in enumerate(schedule, start=1):
        hidden, mask, _ = _batch(
            cache, prepared, local_indices, device=device
        )
        with torch.no_grad():
            cells = collect_transitions(teacher, hidden, mask)
        optimizer.zero_grad(set_to_none=True)
        losses: list[Tensor] = []
        coefficients: list[Tensor] = []
        transfer_ratios: list[Tensor] = []
        for cell in cells:
            layer = student.layers[cell.layer_index]
            actual = selected_projection(
                layer, cell.attention_state, cell.route
            )
            target, transfer, coefficient = overlap_write_target(
                cell.common_update,
                cell.projection_update,
                maximum_coefficient=spec.overlap_coefficient_max,
            )
            losses.append(normalized_mse(actual, target))
            coefficients.append(coefficient)
            transfer_ratios.append(
                transfer.square().mean()
                / cell.common_update.square().mean().clamp_min(1.0e-12)
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, spec.gradient_clip)
            .detach()
            .cpu()
        )
        scale = _lr_scale(update, len(schedule), spec)
        for group in optimizer.param_groups:
            group["lr"] = spec.write_learning_rate * scale
        optimizer.step()
        if update % spec.evaluation_interval == 0 or update == len(schedule):
            row = {
                "stage": "write",
                "update": update,
                "updates": len(schedule),
                "normalized_mse": float(loss.detach().cpu()),
                "coefficient_mean": float(
                    torch.cat(coefficients).mean().detach().cpu()
                ),
                "transfer_to_common_energy": float(
                    torch.stack(transfer_ratios).mean().detach().cpu()
                ),
                "gradient_norm_before_clip": gradient_norm,
                "learning_rate": spec.write_learning_rate * scale,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            progress("write", row)
        del hidden, mask, cells, losses, coefficients, transfer_ratios, loss
    return {
        "updates": len(schedule),
        "wall_seconds": time.perf_counter() - started,
        "history": history,
        "trainable_parameters": int(sum(value.numel() for value in parameters)),
        "uses_family_targets": False,
    }


def evaluate_write_fit(
    student: H1LatentReasoner,
    teacher: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    spec: WriteDeleteSpec,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    student.to(spec.device).eval()
    teacher.to(spec.device).eval()
    squared_error = 0.0
    target_energy = 0.0
    transfer_energy = 0.0
    common_energy = 0.0
    residual_energy = 0.0
    coefficients: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(prepared.ids), batch_size):
            local = np.arange(start, min(start + batch_size, len(prepared.ids)))
            hidden, mask, _ = _batch(cache, prepared, local, device=spec.device)
            for cell in collect_transitions(teacher, hidden, mask):
                actual = selected_projection(
                    student.layers[cell.layer_index],
                    cell.attention_state,
                    cell.route,
                )
                target, transfer, coefficient = overlap_write_target(
                    cell.common_update,
                    cell.projection_update,
                    maximum_coefficient=spec.overlap_coefficient_max,
                )
                residual = cell.common_update - transfer
                squared_error += float((actual - target).square().sum().cpu())
                target_energy += float(target.square().sum().cpu())
                transfer_energy += float(transfer.square().sum().cpu())
                common_energy += float(cell.common_update.square().sum().cpu())
                residual_energy += float(residual.square().sum().cpu())
                coefficients.append(coefficient.cpu())
    coefficient = torch.cat(coefficients)
    return {
        "split": prepared.split,
        "normalized_mse": squared_error / max(target_energy, 1.0e-12),
        "coefficient": {
            "mean": float(coefficient.mean()),
            "minimum": float(coefficient.min()),
            "maximum": float(coefficient.max()),
            "positive_fraction": float((coefficient > 0).float().mean()),
        },
        "transfer_to_common_energy": transfer_energy / max(common_energy, 1.0e-12),
        "common_residual_energy": residual_energy / max(common_energy, 1.0e-12),
    }


def train_delete_stage(
    student: H1LatentReasoner,
    written: H1LatentReasoner,
    teacher: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    schedule: np.ndarray,
    spec: WriteDeleteSpec,
    *,
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Freeze the written projection and fit common to the frozen complement."""

    device = spec.device
    student.to(device).train()
    written.to(device).eval()
    teacher.to(device).eval()
    parameters = freeze_for_role(student, "common")
    optimizer = _stage_optimizer(
        parameters, learning_rate=spec.delete_learning_rate, spec=spec
    )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update, local_indices in enumerate(schedule, start=1):
        hidden, mask, _ = _batch(cache, prepared, local_indices, device=device)
        with torch.no_grad():
            cells = collect_transitions(teacher, hidden, mask)
            targets: list[Tensor] = []
            for cell in cells:
                projection_written = selected_projection(
                    written.layers[cell.layer_index],
                    cell.attention_state,
                    cell.route,
                )
                transferred = projection_written - cell.projection_update
                targets.append(cell.common_update - transferred)
        optimizer.zero_grad(set_to_none=True)
        losses = [
            normalized_mse(
                student.layers[cell.layer_index].shared_ffn(cell.attention_state),
                target,
            )
            for cell, target in zip(cells, targets, strict=True)
        ]
        loss = torch.stack(losses).mean()
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, spec.gradient_clip)
            .detach()
            .cpu()
        )
        scale = _lr_scale(update, len(schedule), spec)
        for group in optimizer.param_groups:
            group["lr"] = spec.delete_learning_rate * scale
        optimizer.step()
        if update % spec.evaluation_interval == 0 or update == len(schedule):
            row = {
                "stage": "delete",
                "update": update,
                "updates": len(schedule),
                "normalized_mse": float(loss.detach().cpu()),
                "gradient_norm_before_clip": gradient_norm,
                "learning_rate": spec.delete_learning_rate * scale,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            progress("delete", row)
        del hidden, mask, cells, targets, losses, loss
    return {
        "updates": len(schedule),
        "wall_seconds": time.perf_counter() - started,
        "history": history,
        "trainable_parameters": int(sum(value.numel() for value in parameters)),
        "uses_family_targets": False,
    }


def evaluate_delete_fit(
    student: H1LatentReasoner,
    written: H1LatentReasoner,
    teacher: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    spec: WriteDeleteSpec,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    student.to(spec.device).eval()
    written.to(spec.device).eval()
    teacher.to(spec.device).eval()
    common_error = 0.0
    common_target_energy = 0.0
    total_error = 0.0
    total_target_energy = 0.0
    common_old_energy = 0.0
    common_new_energy = 0.0
    with torch.inference_mode():
        for start in range(0, len(prepared.ids), batch_size):
            local = np.arange(start, min(start + batch_size, len(prepared.ids)))
            hidden, mask, _ = _batch(cache, prepared, local, device=spec.device)
            for cell in collect_transitions(teacher, hidden, mask):
                projection_written = selected_projection(
                    written.layers[cell.layer_index],
                    cell.attention_state,
                    cell.route,
                )
                common_target = cell.common_update - (
                    projection_written - cell.projection_update
                )
                common_actual = student.layers[cell.layer_index].shared_ffn(
                    cell.attention_state
                )
                old_total = cell.common_update + cell.projection_update
                new_total = common_actual + projection_written
                common_error += float((common_actual - common_target).square().sum().cpu())
                common_target_energy += float(common_target.square().sum().cpu())
                total_error += float((new_total - old_total).square().sum().cpu())
                total_target_energy += float(old_total.square().sum().cpu())
                common_old_energy += float(cell.common_update.square().sum().cpu())
                common_new_energy += float(common_actual.square().sum().cpu())
    return {
        "split": prepared.split,
        "common_normalized_mse": common_error / max(common_target_energy, 1.0e-12),
        "total_transition_normalized_mse": total_error
        / max(total_target_energy, 1.0e-12),
        "common_residual_energy": common_new_energy / max(common_old_energy, 1.0e-12),
    }


def train_joint_pair(
    write_delete: H1LatentReasoner,
    control: H1LatentReasoner,
    teacher: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    schedule: np.ndarray,
    spec: WriteDeleteSpec,
    *,
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Continue WD and untouched control on the same answer-only schedule."""

    device = spec.device
    teacher.to(device).eval()
    arms = {"write_delete": write_delete.to(device), "control": control.to(device)}
    parameters = {name: freeze_for_role(model, "joint") for name, model in arms.items()}
    optimizers = {
        name: _stage_optimizer(
            parameters[name], learning_rate=spec.joint_learning_rate, spec=spec
        )
        for name in arms
    }
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for update, local_indices in enumerate(schedule, start=1):
        hidden, mask, targets = _batch(
            cache, prepared, local_indices, device=device
        )
        with torch.no_grad():
            teacher_output = forward_write_delete(
                teacher, hidden, mask, return_trajectory=False
            )
            teacher_logits = teacher_output["logits"]
            assert teacher_logits is not None
            teacher_probabilities = teacher_logits.softmax(dim=-1)
        rows: dict[str, Any] = {}
        for name, model in arms.items():
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            output = forward_write_delete(
                model, hidden, mask, return_trajectory=False
            )
            logits = output["logits"]
            assert logits is not None
            answer_loss = F.cross_entropy(logits, targets)
            teacher_kl = F.kl_div(
                logits.log_softmax(dim=-1),
                teacher_probabilities,
                reduction="batchmean",
            )
            loss = answer_loss + spec.teacher_kl_weight * teacher_kl
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters[name], spec.gradient_clip
                )
                .detach()
                .cpu()
            )
            scale = _lr_scale(update, len(schedule), spec)
            for group in optimizer.param_groups:
                group["lr"] = spec.joint_learning_rate * scale
            optimizer.step()
            rows[name] = {
                "answer_loss": float(answer_loss.detach().cpu()),
                "teacher_kl": float(teacher_kl.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
                "batch_accuracy": float(
                    (logits.argmax(dim=-1) == targets).float().mean().detach().cpu()
                ),
                "gradient_norm_before_clip": gradient_norm,
            }
        if update % spec.evaluation_interval == 0 or update == len(schedule):
            row = {
                "stage": "joint",
                "update": update,
                "updates": len(schedule),
                "learning_rate": spec.joint_learning_rate
                * _lr_scale(update, len(schedule), spec),
                "arms": rows,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            progress("joint", row)
        del hidden, mask, targets, teacher_output, teacher_logits, teacher_probabilities
    return {
        "updates": len(schedule),
        "wall_seconds": time.perf_counter() - started,
        "history": history,
        "uses_family_targets": False,
        "teacher_is_task_classifier": False,
    }


def evaluate_answer(
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    *,
    device: str,
    intervention: str = "none",
    batch_size: int = 64,
) -> dict[str, Any]:
    model.to(device).eval()
    predictions: list[int] = []
    routes: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(prepared.ids), batch_size):
            stop = min(start + batch_size, len(prepared.ids))
            hidden, mask = cache.batch(
                prepared.cache_indices[start:stop], device=device
            )
            output = forward_write_delete(
                model,
                hidden,
                mask,
                intervention=intervention,
                return_trajectory=False,
            )
            logits = output["logits"]
            route_assignments = output["route_assignments"]
            assert logits is not None and route_assignments is not None
            predictions.extend(int(value) for value in logits.argmax(-1).cpu().tolist())
            majority = (
                route_assignments.to(torch.float32).mean(dim=(1, 2)) >= 0.5
            ).long()
            routes.extend(int(value) for value in majority.cpu().tolist())
    metrics = classification_metrics(
        prepared.ids,
        predictions,
        prepared.targets.tolist(),
        prepared.families,
    )
    return {
        "split": prepared.split,
        "intervention": intervention,
        "ids": list(prepared.ids),
        "targets": prepared.targets.tolist(),
        "families": list(prepared.families),
        "predictions": predictions,
        "route_predictions": routes,
        "answer": metrics,
    }


def evaluate_intervention_suite(
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    *,
    device: str,
) -> dict[str, Any]:
    names = (
        "none",
        "flip_route",
        "force_route0",
        "force_route1",
        "disable_routed_projection",
        "disable_shared_ffn",
        "disable_both_ffn_paths",
        "disable_recurrence",
        "zero_source",
        "shuffle_source",
    )
    full = {
        name: evaluate_answer(
            model, cache, prepared, device=device, intervention=name
        )
        for name in names
    }
    normal = full["none"]["answer"]["accuracy"]
    compact: dict[str, Any] = {}
    for name, value in full.items():
        answer = value["answer"]
        compact[name] = {
            "accuracy": answer["accuracy"],
            "macro_accuracy": answer["macro_accuracy"],
            "by_family": answer["by_family"],
            "prediction_sha256": answer["prediction_sha256"],
            "drop_from_normal": float(normal - answer["accuracy"]),
        }
    route_effect = max(
        compact[name]["drop_from_normal"]
        for name in ("flip_route", "force_route0", "force_route1")
    )
    return {
        "full": full,
        "compact": compact,
        "maximum_wrong_route_answer_drop": float(route_effect),
    }


def compare_free_rollout(
    predecessor: H1LatentReasoner,
    successor: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    *,
    device: str,
    batch_size: int = 64,
) -> dict[str, Any]:
    predecessor.to(device).eval()
    successor.to(device).eval()
    logits_error = 0.0
    logits_energy = 0.0
    state_error = 0.0
    state_energy = 0.0
    trajectory_error = 0.0
    trajectory_energy = 0.0
    agreement = 0
    count = 0
    kl_total = 0.0
    with torch.inference_mode():
        for start in range(0, len(prepared.ids), batch_size):
            stop = min(start + batch_size, len(prepared.ids))
            hidden, mask = cache.batch(
                prepared.cache_indices[start:stop], device=device
            )
            before = forward_write_delete(predecessor, hidden, mask)
            after = forward_write_delete(successor, hidden, mask)
            before_logits = before["logits"]
            after_logits = after["logits"]
            before_state = before["final_state"]
            after_state = after["final_state"]
            before_trajectory = before["trajectory"]
            after_trajectory = after["trajectory"]
            assert all(
                value is not None
                for value in (
                    before_logits,
                    after_logits,
                    before_state,
                    after_state,
                    before_trajectory,
                    after_trajectory,
                )
            )
            logits_error += float((after_logits - before_logits).square().sum().cpu())
            logits_energy += float(before_logits.square().sum().cpu())
            state_error += float((after_state - before_state).square().sum().cpu())
            state_energy += float(before_state.square().sum().cpu())
            trajectory_error += float(
                (after_trajectory - before_trajectory).square().sum().cpu()
            )
            trajectory_energy += float(before_trajectory.square().sum().cpu())
            agreement += int(
                (after_logits.argmax(-1) == before_logits.argmax(-1)).sum().cpu()
            )
            count += int(before_logits.shape[0])
            kl_total += float(
                F.kl_div(
                    after_logits.log_softmax(-1),
                    before_logits.softmax(-1),
                    reduction="sum",
                ).cpu()
            )
    return {
        "split": prepared.split,
        "prediction_agreement": agreement / count,
        "teacher_logit_kl_per_record": kl_total / count,
        "logit_relative_mse": logits_error / max(logits_energy, 1.0e-12),
        "final_state_relative_mse": state_error / max(state_energy, 1.0e-12),
        "trajectory_relative_mse": trajectory_error
        / max(trajectory_energy, 1.0e-12),
    }


def _compact_primary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": value["split"],
        "intervention": value["intervention"],
        "answer": {
            key: item
            for key, item in value["answer"].items()
            if key != "correct_vector"
        },
    }


def _prepare_context(repo_root: Path) -> dict[str, Any]:
    source_checkpoint = repo_root / SOURCE_CHECKPOINT
    source_cache = repo_root / SOURCE_TOKEN_CACHE
    output_root = repo_root / OUTPUT_ROOT
    checks = {
        "source_root_exists": (repo_root / SOURCE_ROOT).is_dir(),
        "source_checkpoint_exists": source_checkpoint.is_file(),
        "source_cache_exists": source_cache.is_dir(),
        "output_root_absent": not output_root.exists(),
        "design_exists": (repo_root / DESIGN_DOCUMENT).is_file(),
        "execution_exists": (repo_root / EXECUTION_DOCUMENT).is_file(),
        "cuda_available": torch.cuda.is_available(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"H1-WD path/hardware preflight failed: {checks}")
    checkpoint_sha256 = _sha256_path(source_checkpoint)
    if checkpoint_sha256 != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("H1-WD source checkpoint hash drift")

    registry = historical_fingerprint_registry(repo_root)
    packages = generate_registered_nonformal_packages(
        forbidden_semantic_fingerprints=registry["semantic"],
        forbidden_source_fingerprints=registry["source"],
    )
    package = packages["screen"]
    identity = package_identity(package)
    if identity != SOURCE_PACKAGE_IDENTITY:
        raise RuntimeError("H1-WD source package identity drift")
    cache_audit = audit_token_cache(package.records, source_cache)
    if cache_audit.get("passed") is not True:
        raise RuntimeError(f"H1-WD source cache audit failed: {cache_audit}")
    cache = TokenCache(source_cache)
    prepared = {
        split: prepare_split(package.bundle, split, cache, steps=8)
        for split in ("train", "validation", "supported", "heldout", "causal")
    }
    record_order_sha256 = hashlib.sha256(
        _canonical([record.id for record in package.records])
    ).hexdigest().upper()
    return {
        "checks": checks,
        "checkpoint_sha256": checkpoint_sha256,
        "package_identity": identity,
        "record_order_sha256": record_order_sha256,
        "cache_audit": cache_audit,
        "cache": cache,
        "prepared": prepared,
        "output_root": output_root,
        "source_checkpoint": source_checkpoint,
    }


def run_write_delete_screen(repo_root: Path) -> dict[str, Any]:
    """Run the single registered H1-WD development screen."""

    repo_root = Path(repo_root).resolve()
    context = _prepare_context(repo_root)
    output_root: Path = context["output_root"]
    output_root.mkdir(parents=True, exist_ok=False)
    events_path = output_root / "training-events.jsonl"
    state_path = output_root / "run-state.json"
    started = time.perf_counter()

    def progress(stage: str, row: Mapping[str, Any]) -> None:
        event = {
            "timestamp": time.time(),
            "stage": stage,
            **dict(row),
        }
        _append_jsonl(events_path, event)
        _write_json(
            state_path,
            {
                "schema_version": f"{SCHEMA_PREFIX}.run-state.v1",
                "status": (
                    str(row.get("status"))
                    if stage == "complete" and row.get("status")
                    else "RUNNING_H1_WD_NONFORMAL"
                ),
                "stage": stage,
                "latest": dict(row),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )

    _write_json(output_root / "contract-manifest.json", contract_manifest())
    preflight = {
        "schema_version": f"{SCHEMA_PREFIX}.preflight.v1",
        "status": "PASS_H1_WD_PREFLIGHT",
        "checks": context["checks"],
        "checkpoint_sha256": context["checkpoint_sha256"],
        "package_identity": context["package_identity"],
        "record_order_sha256": context["record_order_sha256"],
        "cache_audit": context["cache_audit"],
        "old_root_mutation_authorized": False,
        "uses_family_targets_for_training": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    progress("baseline", {"message": "loading immutable predecessor"})

    cache: TokenCache = context["cache"]
    prepared: dict[str, PreparedSplit] = context["prepared"]
    teacher = load_deployment_checkpoint(
        context["source_checkpoint"], device=SPEC.device
    )
    teacher.eval()
    source_suite = evaluate_intervention_suite(
        teacher, cache, prepared["heldout"], device=SPEC.device
    )
    source_supported = evaluate_answer(
        teacher, cache, prepared["supported"], device=SPEC.device
    )
    progress(
        "baseline",
        {
            "heldout_accuracy": source_suite["compact"]["none"]["accuracy"],
            "projection_drop": source_suite["compact"][
                "disable_routed_projection"
            ]["drop_from_normal"],
        },
    )

    write_schedule, write_schedule_report = build_unlabeled_schedule(
        len(prepared["train"].ids),
        updates=SPEC.write_updates,
        batch_size=SPEC.batch_size,
        seed=SPEC.schedule_seed ^ 0x57,
    )
    delete_schedule, delete_schedule_report = build_unlabeled_schedule(
        len(prepared["train"].ids),
        updates=SPEC.delete_updates,
        batch_size=SPEC.batch_size,
        seed=SPEC.schedule_seed ^ 0x44,
    )
    joint_schedule, joint_schedule_report = build_unlabeled_schedule(
        len(prepared["train"].ids),
        updates=SPEC.joint_updates,
        batch_size=SPEC.batch_size,
        seed=SPEC.schedule_seed ^ 0x4A,
    )
    schedules = {
        "write": write_schedule_report,
        "delete": delete_schedule_report,
        "joint": joint_schedule_report,
    }
    _write_json(output_root / "schedules.json", schedules)

    written = deepcopy(teacher)
    write_training = train_write_stage(
        written,
        teacher,
        cache,
        prepared["train"],
        write_schedule,
        SPEC,
        progress=progress,
    )
    write_fit = {
        split: evaluate_write_fit(
            written, teacher, cache, prepared[split], SPEC
        )
        for split in ("validation", "heldout")
    }
    write_checkpoint = _save_model(
        written, output_root / "write-checkpoint.pt", stage="write"
    )
    _write_json(
        output_root / "write-result.json",
        {
            "training": write_training,
            "fit": write_fit,
            "checkpoint": write_checkpoint,
        },
    )
    write_gate = (
        write_fit["heldout"]["normalized_mse"]
        <= SPEC.maximum_write_normalized_mse
    )
    if not write_gate:
        result = {
            "schema_version": f"{SCHEMA_PREFIX}.result.v1",
            "status": "FAIL_H1_WD_WRITE_FIT",
            "scope": "NONFORMAL_WD_MECHANISM_SCREEN_ONLY",
            "authorizes": "nothing",
            "preflight": preflight,
            "schedules": schedules,
            "source": {
                "heldout": source_suite["compact"],
                "supported": _compact_primary(source_supported),
            },
            "write": {"training": write_training, "fit": write_fit},
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(output_root / "result.json", result)
        progress("complete", {"status": result["status"]})
        return result

    residual = deepcopy(written)
    delete_training = train_delete_stage(
        residual,
        written,
        teacher,
        cache,
        prepared["train"],
        delete_schedule,
        SPEC,
        progress=progress,
    )
    delete_fit = {
        split: evaluate_delete_fit(
            residual, written, teacher, cache, prepared[split], SPEC
        )
        for split in ("validation", "heldout")
    }
    residual_checkpoint = _save_model(
        residual, output_root / "residual-checkpoint.pt", stage="residual"
    )
    residual_suite = evaluate_intervention_suite(
        residual, cache, prepared["heldout"], device=SPEC.device
    )
    residual_rollout = compare_free_rollout(
        teacher, residual, cache, prepared["heldout"], device=SPEC.device
    )
    _write_json(
        output_root / "delete-result.json",
        {
            "training": delete_training,
            "fit": delete_fit,
            "free_rollout": residual_rollout,
            "interventions": residual_suite["compact"],
            "checkpoint": residual_checkpoint,
        },
    )
    delete_gate = (
        delete_fit["heldout"]["common_normalized_mse"]
        <= SPEC.maximum_delete_normalized_mse
        and residual_rollout["prediction_agreement"]
        >= SPEC.minimum_prediction_agreement
        and SPEC.minimum_common_residual_energy
        <= delete_fit["heldout"]["common_residual_energy"]
        <= SPEC.maximum_common_residual_energy
    )
    if not delete_gate:
        result = {
            "schema_version": f"{SCHEMA_PREFIX}.result.v1",
            "status": "FAIL_H1_WD_DELETE_FIT",
            "scope": "NONFORMAL_WD_MECHANISM_SCREEN_ONLY",
            "authorizes": "nothing",
            "preflight": preflight,
            "schedules": schedules,
            "source": {"heldout": source_suite["compact"]},
            "write": {"training": write_training, "fit": write_fit},
            "delete": {
                "training": delete_training,
                "fit": delete_fit,
                "free_rollout": residual_rollout,
                "interventions": residual_suite["compact"],
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(output_root / "result.json", result)
        progress("complete", {"status": result["status"]})
        return result

    write_delete_final = deepcopy(residual)
    control_final = deepcopy(teacher)
    joint_training = train_joint_pair(
        write_delete_final,
        control_final,
        teacher,
        cache,
        prepared["train"],
        joint_schedule,
        SPEC,
        progress=progress,
    )
    wd_checkpoint = _save_model(
        write_delete_final,
        output_root / "write-delete-final.pt",
        stage="write-delete-joint",
    )
    control_checkpoint = _save_model(
        control_final,
        output_root / "control-final.pt",
        stage="matched-continuation-control",
    )
    final_suites = {
        "write_delete": evaluate_intervention_suite(
            write_delete_final,
            cache,
            prepared["heldout"],
            device=SPEC.device,
        ),
        "control": evaluate_intervention_suite(
            control_final,
            cache,
            prepared["heldout"],
            device=SPEC.device,
        ),
    }
    wd_heldout = final_suites["write_delete"]["full"]["none"]
    control_heldout = final_suites["control"]["full"]["none"]
    benefit = primary_comparison(
        control_heldout,
        wd_heldout,
        bootstrap_seed=SPEC.schedule_seed ^ 0xB00757,
        primary_gain=SPEC.architecture_gain_threshold,
        family_regression_limit=0.02,
    )
    final_rollout = compare_free_rollout(
        teacher,
        write_delete_final,
        cache,
        prepared["heldout"],
        device=SPEC.device,
    )
    final_allocation_fit = evaluate_delete_fit(
        write_delete_final,
        write_delete_final,
        teacher,
        cache,
        prepared["heldout"],
        SPEC,
    )
    source_projection_drop = source_suite["compact"][
        "disable_routed_projection"
    ]["drop_from_normal"]
    wd_projection_drop = final_suites["write_delete"]["compact"][
        "disable_routed_projection"
    ]["drop_from_normal"]
    wd_common_drop = final_suites["write_delete"]["compact"][
        "disable_shared_ffn"
    ]["drop_from_normal"]
    heldout_drop = (
        source_suite["compact"]["none"]["accuracy"]
        - final_suites["write_delete"]["compact"]["none"]["accuracy"]
    )
    mechanism_checks = {
        "write_fit": write_gate,
        "delete_fit": delete_gate,
        "prediction_agreement": final_rollout["prediction_agreement"]
        >= SPEC.minimum_prediction_agreement,
        "heldout_retention": heldout_drop <= SPEC.maximum_heldout_accuracy_drop,
        "common_remains_necessary": wd_common_drop >= SPEC.minimum_common_off_drop,
        "projection_effect_increased": (
            wd_projection_drop - source_projection_drop
            >= SPEC.minimum_projection_effect_gain
        ),
        "final_residual_consistency": final_allocation_fit[
            "common_normalized_mse"
        ]
        <= 0.15,
        "common_not_deleted": SPEC.minimum_common_residual_energy
        <= final_allocation_fit["common_residual_energy"]
        <= SPEC.maximum_common_residual_energy,
        "no_family_training_target": all(
            report["uses_family_targets"] is False
            for report in (write_training, delete_training, joint_training)
        ),
        "source_checkpoint_unchanged": _sha256_path(
            context["source_checkpoint"]
        )
        == SOURCE_CHECKPOINT_SHA256,
    }
    mechanism_passed = all(mechanism_checks.values())
    architecture_benefit_passed = bool(benefit["gate"]["passed"])
    status = (
        "PASS_H1_WD_NONFORMAL_MECHANISM"
        if mechanism_passed
        else "FAIL_H1_WD_NONFORMAL_MECHANISM"
    )
    result = {
        "schema_version": f"{SCHEMA_PREFIX}.result.v1",
        "status": status,
        "scope": "NONFORMAL_WD_MECHANISM_SCREEN_ONLY",
        "authorizes": "nothing",
        "claims": {
            "write_delete_mechanism_signal": mechanism_passed,
            "architecture_benefit_signal": architecture_benefit_passed,
            "h1_qualified": False,
            "p1_completed": False,
            "f1_authorized": False,
            "p2_authorized": False,
            "real_text_route_qualified": False,
        },
        "preflight": preflight,
        "schedules": schedules,
        "source": {
            "heldout": source_suite["compact"],
            "supported": _compact_primary(source_supported),
        },
        "write": {
            "training": write_training,
            "fit": write_fit,
            "checkpoint": write_checkpoint,
        },
        "delete": {
            "training": delete_training,
            "fit": delete_fit,
            "free_rollout": residual_rollout,
            "interventions": residual_suite["compact"],
            "checkpoint": residual_checkpoint,
        },
        "joint": {
            "training": joint_training,
            "write_delete": final_suites["write_delete"]["compact"],
            "control": final_suites["control"]["compact"],
            "benefit": benefit,
            "free_rollout": final_rollout,
            "final_allocation_fit": final_allocation_fit,
            "checkpoints": {
                "write_delete": wd_checkpoint,
                "control": control_checkpoint,
            },
        },
        "mechanism_gate": {
            "checks": mechanism_checks,
            "passed": mechanism_passed,
        },
        "architecture_benefit_gate": {
            "passed": architecture_benefit_passed,
            "comparison": benefit,
            "note": (
                "A causal effect created by residual reparameterization is not "
                "architecture benefit; only the matched continuation comparison "
                "can provide a non-formal benefit signal."
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_root / "result.json", result)
    progress("complete", {"status": status})
    return result


__all__ = [
    "build_unlabeled_schedule",
    "compare_free_rollout",
    "evaluate_answer",
    "evaluate_delete_fit",
    "evaluate_intervention_suite",
    "evaluate_write_fit",
    "run_write_delete_screen",
    "train_delete_stage",
    "train_joint_pair",
    "train_write_stage",
]
