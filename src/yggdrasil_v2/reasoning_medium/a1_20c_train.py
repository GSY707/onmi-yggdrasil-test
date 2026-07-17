from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_9_model import module_state_sha256
from .a1_13_train import _write_json
from .a1_19h_h2_train import load_a119h2_deployment
from .a1_20b_cache import (
    A120BCachedSplit,
    MANIFEST_SCHEMA as CACHE_MANIFEST_SCHEMA,
    collate_a120b_items,
    collate_a120b_items_cpu,
    file_sha256,
    move_a120b_inputs,
)
from .a1_20b_train import (
    A120BLabels,
    _validation_score,
    compute_a120b_loss,
    encode_a120b_labels,
    evaluate_a120b_matrix,
    move_a120b_labels,
    pin_a120b_labels,
    sample_a120b_indices,
)
from .a1_20c_model import A120CConfig, A120CFullTextBoundary
from .a1_20c_supervision import (
    A120CAnchorLabels,
    A120CSupervisionSplit,
    MANIFEST_SCHEMA as SUPERVISION_MANIFEST_SCHEMA,
)


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.20c.boundary.checkpoint.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.20c.boundary.results.v1"
ANCHOR_LOSS_WEIGHT = 1.0
PAYLOAD_REGRESSION_WEIGHT = 10.0


def _pin_a120c_anchors(
    anchors: A120CAnchorLabels,
) -> A120CAnchorLabels:
    return A120CAnchorLabels(
        **{
            name: value.pin_memory()
            for name, value in anchors.__dict__.items()
        }
    )


def _move_a120c_anchors(
    anchors: A120CAnchorLabels,
    device: str | torch.device,
    *,
    non_blocking: bool,
) -> A120CAnchorLabels:
    return A120CAnchorLabels(
        **{
            name: value.to(device, non_blocking=non_blocking)
            for name, value in anchors.__dict__.items()
        }
    )


def _prepare_a120c_cpu_batch(
    dataset: A120BCachedSplit,
    supervision: A120CSupervisionSplit,
    indices: Sequence[int],
    *,
    pin_memory: bool,
) -> tuple[dict[str, torch.Tensor], A120BLabels, A120CAnchorLabels]:
    inputs, records = collate_a120b_items_cpu(
        dataset.items(indices), pin_memory=pin_memory
    )
    labels = encode_a120b_labels(records, "cpu")
    anchors = supervision.select(indices, "cpu")
    if pin_memory:
        labels = pin_a120b_labels(labels)
        anchors = _pin_a120c_anchors(anchors)
    return inputs, labels, anchors


def _prefetched_a120c_batches(
    dataset: A120BCachedSplit,
    supervision: A120CSupervisionSplit,
    schedule: Sequence[Sequence[int]],
    device: str | torch.device,
):
    pin_memory = torch.device(device).type == "cuda"
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="a120c-prefetch"
    ) as pool:
        future = pool.submit(
            _prepare_a120c_cpu_batch,
            dataset,
            supervision,
            schedule[0],
            pin_memory=pin_memory,
        )
        for step in range(len(schedule)):
            cpu_inputs, cpu_labels, cpu_anchors = future.result()
            if step + 1 < len(schedule):
                future = pool.submit(
                    _prepare_a120c_cpu_batch,
                    dataset,
                    supervision,
                    schedule[step + 1],
                    pin_memory=pin_memory,
                )
            yield (
                move_a120b_inputs(cpu_inputs, device),
                move_a120b_labels(
                    cpu_labels, device, non_blocking=pin_memory
                ),
                _move_a120c_anchors(
                    cpu_anchors, device, non_blocking=pin_memory
                ),
            )


@dataclass(frozen=True)
class A120CTrainSpec:
    reader_seed: int
    data_seed: int
    core_model_seed: int
    compiler: str
    credit_mode: str
    steps: int = 5000
    batch_size: int = 16
    learning_rate: float = 3e-4
    state_loss_weight: float = 1.0
    state_tail_weight: float = 0.0
    state_tail_fraction: float = 0.01
    payload_regression_weight: float = PAYLOAD_REGRESSION_WEIGHT
    optimization_scope: str = "all"
    validation_interval: int = 200


def _anchor_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )


def compute_a120c_loss(
    model: A120CFullTextBoundary,
    output: dict[str, Any],
    labels: A120BLabels,
    anchors: A120CAnchorLabels,
    *,
    state_loss_weight: float = 1.0,
    state_tail_weight: float = 0.0,
    state_tail_fraction: float = 0.01,
    payload_regression_weight: float = PAYLOAD_REGRESSION_WEIGHT,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if state_loss_weight < 0:
        raise ValueError("A1.20C state loss weight must be non-negative")
    if payload_regression_weight < 0:
        raise ValueError(
            "A1.20C payload regression weight must be non-negative"
        )
    if state_tail_weight < 0:
        raise ValueError("A1.20C state tail weight must be non-negative")
    if not 0 < state_tail_fraction <= 1:
        raise ValueError(
            "A1.20C state tail fraction must be in (0, 1]"
        )
    base_loss, components = compute_a120b_loss(model, output, labels)
    zero = base_loss.new_zeros(())
    anchor_components = {
        "entity_name_anchor_ce": zero,
        "entity_value_anchor_ce": zero,
        "operation_anchor_ce": zero,
        "query_anchor_ce": zero,
    }
    if model.config.compiler != "flat":
        anchor_components = {
            "entity_name_anchor_ce": _anchor_cross_entropy(
                output["entity_name_anchor_logits"],
                anchors.entity_name_targets,
            ),
            "entity_value_anchor_ce": _anchor_cross_entropy(
                output["entity_value_anchor_logits"],
                anchors.entity_value_targets,
            ),
            "operation_anchor_ce": _anchor_cross_entropy(
                output["operation_anchor_logits"],
                anchors.operation_role_targets,
            ),
            "query_anchor_ce": _anchor_cross_entropy(
                output["query_anchor_logits"], anchors.query_targets
            ),
        }
    payload_regression = zero
    if model.config.compiler == "factorized":
        active_entities = labels.value_targets >= 0
        target_payloads = model.core.closure_targets(
            labels.value_targets.unsqueeze(1)
        ).squeeze(1).detach()
        payload_regression = nn.functional.mse_loss(
            output["continuous_entity_payloads"][active_entities],
            target_payloads[active_entities],
        )
    state_tail_ce = zero
    if state_tail_weight > 0:
        flat_targets = labels.state_targets.reshape(-1)
        per_token_state_ce = nn.functional.cross_entropy(
            output["state_logits"].reshape(
                -1, model.core.config.value_classes
            ),
            flat_targets,
            ignore_index=-100,
            reduction="none",
        )
        active_state_ce = per_token_state_ce[flat_targets >= 0]
        tail_count = max(
            1, math.ceil(active_state_ce.numel() * state_tail_fraction)
        )
        state_tail_ce = active_state_ce.topk(tail_count).values.mean()
    total = (
        base_loss
        + (state_loss_weight - 1.0) * components["state_ce"]
        + state_tail_weight * state_tail_ce
        + ANCHOR_LOSS_WEIGHT * sum(anchor_components.values(), zero)
        + payload_regression_weight * payload_regression
    )
    return total, {
        **components,
        **anchor_components,
        "payload_regression_mse": payload_regression,
        "state_tail_ce": state_tail_ce,
    }


def _exact_flags(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    active = target >= 0
    correct = (prediction == target) | ~active
    sequence = correct.reshape(correct.shape[0], -1).all(dim=-1)
    return (
        sequence,
        int(((prediction == target) & active).sum()),
        int(active.sum()),
    )


@torch.inference_mode()
def evaluate_a120c_anchors(
    model: A120CFullTextBoundary,
    dataset: A120BCachedSplit,
    supervision: A120CSupervisionSplit,
    device: str | torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    if model.config.compiler == "flat":
        return {"applicable": False}
    model.eval()
    counters = {
        "examples": 0,
        "entity_name_sequence": 0,
        "entity_name_correct": 0,
        "entity_name_total": 0,
        "entity_value_sequence": 0,
        "entity_value_correct": 0,
        "entity_value_total": 0,
        "operation_sequence": 0,
        "operation_correct": 0,
        "operation_total": 0,
        "query": 0,
        "all_anchor_sequence": 0,
    }
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
        inputs, _ = collate_a120b_items(dataset.items(indices), device)
        anchors = supervision.select(indices, device)
        output = model(**inputs)
        entity_name = output["entity_name_anchor_logits"].argmax(dim=-1)
        entity_value = output["entity_value_anchor_logits"].argmax(dim=-1)
        operation = output["operation_anchor_logits"].argmax(dim=-1)
        query = output["query_anchor_logits"].argmax(dim=-1)
        name_sequence, name_correct, name_total = _exact_flags(
            entity_name, anchors.entity_name_targets
        )
        value_sequence, value_correct, value_total = _exact_flags(
            entity_value, anchors.entity_value_targets
        )
        operation_sequence, operation_correct, operation_total = _exact_flags(
            operation, anchors.operation_role_targets
        )
        query_correct = query == anchors.query_targets
        counters["examples"] += len(indices)
        counters["entity_name_sequence"] += int(name_sequence.sum())
        counters["entity_name_correct"] += name_correct
        counters["entity_name_total"] += name_total
        counters["entity_value_sequence"] += int(value_sequence.sum())
        counters["entity_value_correct"] += value_correct
        counters["entity_value_total"] += value_total
        counters["operation_sequence"] += int(operation_sequence.sum())
        counters["operation_correct"] += operation_correct
        counters["operation_total"] += operation_total
        counters["query"] += int(query_correct.sum())
        counters["all_anchor_sequence"] += int(
            (
                name_sequence
                & value_sequence
                & operation_sequence
                & query_correct
            ).sum()
        )
    examples = counters["examples"]
    return {
        "applicable": True,
        "examples": examples,
        "entity_name_token_accuracy": counters["entity_name_correct"]
        / counters["entity_name_total"],
        "entity_name_sequence_exact": counters["entity_name_sequence"]
        / examples,
        "entity_value_token_accuracy": counters["entity_value_correct"]
        / counters["entity_value_total"],
        "entity_value_sequence_exact": counters["entity_value_sequence"]
        / examples,
        "operation_role_token_accuracy": counters["operation_correct"]
        / counters["operation_total"],
        "operation_role_sequence_exact": counters["operation_sequence"]
        / examples,
        "query_anchor_accuracy": counters["query"] / examples,
        "all_anchor_sequence_exact": counters["all_anchor_sequence"]
        / examples,
    }


def _checkpoint(
    model: A120CFullTextBoundary,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A120CTrainSpec,
    core_checkpoint: Path,
    cache_manifest_hash: str,
    supervision_manifest_hash: str,
    initialization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "config": asdict(model.config),
        "train_spec": asdict(spec),
        "core_checkpoint": str(core_checkpoint.resolve()),
        "core_checkpoint_sha256": file_sha256(core_checkpoint),
        "core_state_sha256": module_state_sha256(model.core),
        "cache_manifest_sha256": cache_manifest_hash,
        "supervision_manifest_sha256": supervision_manifest_hash,
        "initialization": initialization,
        "boundary": model.boundary_state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def _make_model(
    source_width: int,
    core_checkpoint: Path,
    spec: A120CTrainSpec,
    device: str | torch.device,
) -> A120CFullTextBoundary:
    core = load_a119h2_deployment(core_checkpoint, device)
    if core.a119h_model_seed != spec.core_model_seed:
        raise ValueError("A1.20C core model seed mismatch")
    if core.a119h_data_seed != spec.data_seed:
        raise ValueError("A1.20C core data seed mismatch")
    return A120CFullTextBoundary(
        A120CConfig(
            source_width=source_width,
            compiler=spec.compiler,
            credit_mode=spec.credit_mode,
        ),
        core,
    ).to(device)


def train_a120c_arm(
    cache_dir: Path,
    supervision_dir: Path,
    data_dir: Path,
    core_checkpoint: Path,
    output_dir: Path,
    *,
    spec: A120CTrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
    initial_checkpoint: Path | None = None,
    resume_optimizer: bool = False,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.20C requested unavailable CUDA")
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.20C checkpoints: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_dir / "manifest.json"
    supervision_manifest_path = supervision_dir / "manifest.json"
    cache_manifest = json.loads(
        cache_manifest_path.read_text(encoding="utf-8")
    )
    supervision_manifest = json.loads(
        supervision_manifest_path.read_text(encoding="utf-8")
    )
    if cache_manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ValueError("A1.20C requires A1.20B full-token cache")
    if (
        supervision_manifest.get("schema_version")
        != SUPERVISION_MANIFEST_SCHEMA
    ):
        raise ValueError("A1.20C requires compiler supervision")
    cache_is_overfit = cache_manifest["selection"].endswith(
        "overfit32-fit-only"
    )
    if cache_is_overfit != overfit_mode:
        raise ValueError("A1.20C overfit/cache selection mismatch")
    train_dataset = A120BCachedSplit(cache_dir, data_dir, "train")
    validation_dataset = A120BCachedSplit(
        cache_dir, data_dir, "validation"
    )
    train_supervision = A120CSupervisionSplit(
        supervision_dir, train_dataset, "train"
    )
    validation_supervision = A120CSupervisionSplit(
        supervision_dir, validation_dataset, "validation"
    )
    random.seed(spec.reader_seed)
    torch.manual_seed(spec.reader_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.reader_seed)
    model = _make_model(
        train_dataset.hidden_width, core_checkpoint, spec, device
    )
    core_hash = module_state_sha256(model.core)
    if spec.optimization_scope not in {"all", "identity", "payload"}:
        raise ValueError(
            f"unknown A1.20C optimization scope {spec.optimization_scope}"
        )
    if spec.optimization_scope == "payload":
        payload_prefixes = (
            "boundary.entity_payload_norm.",
            "boundary.payload_head.",
        )
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and not name.startswith(
                payload_prefixes
            ):
                parameter.requires_grad_(False)
    elif spec.optimization_scope == "identity":
        identity_prefixes = (
            "boundary.entity_identity_norm.",
            "boundary.operation_identity_norm.",
            "boundary.query_identity_norm.",
            "boundary.identity_query.",
            "boundary.identity_entity.",
        )
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and not name.startswith(
                identity_prefixes
            ):
                parameter.requires_grad_(False)
    optimized_parameter_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("A1.20C optimization scope selected no parameters")
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate)
    initialization: dict[str, Any] = {
        "fresh": initial_checkpoint is None,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "source_step": None,
        "optimizer_resumed": False,
    }
    if initial_checkpoint is not None:
        initial = torch.load(
            initial_checkpoint, map_location="cpu", weights_only=False
        )
        if initial.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("A1.20C initial checkpoint schema mismatch")
        if initial["config"] != asdict(model.config):
            raise ValueError("A1.20C initial checkpoint config mismatch")
        if initial["core_state_sha256"] != core_hash:
            raise ValueError("A1.20C initial checkpoint core mismatch")
        for seed_name in (
            "reader_seed",
            "data_seed",
            "core_model_seed",
        ):
            if int(initial["train_spec"][seed_name]) != int(
                getattr(spec, seed_name)
            ):
                raise ValueError(
                    f"A1.20C initial checkpoint {seed_name} mismatch"
                )
        model.load_boundary_state_dict(initial["boundary"])
        if resume_optimizer:
            optimizer.load_state_dict(initial["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = spec.learning_rate
        initialization = {
            "fresh": False,
            "checkpoint": str(initial_checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(initial_checkpoint),
            "source_step": int(initial["step"]),
            "optimizer_resumed": resume_optimizer,
        }
    rng = random.Random(spec.reader_seed)
    schedule = (
        None
        if overfit_mode
        else [
            sample_a120b_indices(train_dataset, spec.batch_size, rng)
            for _ in range(spec.steps)
        ]
    )
    prefetched_batches = (
        None
        if schedule is None
        else iter(
            _prefetched_a120c_batches(
                train_dataset,
                train_supervision,
                schedule,
                device,
            )
        )
    )
    overfit_indices = list(range(len(train_dataset)))
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    best_anchor_validation: dict[str, Any] | None = None
    ready_streak = completed_step = 0
    maximum_hard_state_delta = 0.0
    maximum_hard_answer_delta = 0.0
    if overfit_mode:
        preloaded_inputs, preloaded_records = collate_a120b_items(
            train_dataset.items(overfit_indices), device
        )
        preloaded_labels = encode_a120b_labels(
            preloaded_records, device
        )
        preloaded_anchors = train_supervision.select(
            overfit_indices, device
        )
    if initial_checkpoint is not None:
        model.eval()
        initial_validation = evaluate_a120b_matrix(
            model,
            validation_dataset,
            device,
            batch_size=spec.batch_size,
        )
        initial_anchor_validation = evaluate_a120c_anchors(
            model,
            validation_dataset,
            validation_supervision,
            device,
            batch_size=spec.batch_size,
        )
        best_score, _ = _validation_score(initial_validation)
        best_step = 0
        best_validation = initial_validation
        best_anchor_validation = initial_anchor_validation
        initial_saved = _checkpoint(
            model,
            optimizer,
            0,
            spec,
            core_checkpoint,
            file_sha256(cache_manifest_path),
            file_sha256(supervision_manifest_path),
            initialization,
        )
        torch.save(initial_saved, output_dir / "best.pt")
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        if overfit_mode:
            inputs = preloaded_inputs
            labels = preloaded_labels
            anchors = preloaded_anchors
        else:
            assert prefetched_batches is not None
            inputs, labels, anchors = next(prefetched_batches)
        output = model(**inputs)
        loss, components = compute_a120c_loss(
            model,
            output,
            labels,
            anchors,
            state_loss_weight=spec.state_loss_weight,
            state_tail_weight=spec.state_tail_weight,
            state_tail_fraction=spec.state_tail_fraction,
            payload_regression_weight=spec.payload_regression_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            parameter.grad is not None
            for parameter in model.core.parameters()
        ):
            raise RuntimeError("frozen A1.19H core received gradients")
        gradient_norm = nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        completed_step = step
        if step % spec.validation_interval == 0 or step == spec.steps:
            maximum_hard_state_delta = max(
                maximum_hard_state_delta,
                float(output["hard_forward_state_max_abs_delta"].detach()),
            )
            maximum_hard_answer_delta = max(
                maximum_hard_answer_delta,
                float(output["hard_forward_answer_max_abs_delta"].detach()),
            )
            model.eval()
            validation = evaluate_a120b_matrix(
                model,
                validation_dataset,
                device,
                batch_size=spec.batch_size,
            )
            anchor_validation = evaluate_a120c_anchors(
                model,
                validation_dataset,
                validation_supervision,
                device,
                batch_size=spec.batch_size,
            )
            score, score_components = _validation_score(validation)
            row = {
                "step": step,
                "loss": float(loss.detach()),
                **{
                    name: float(value.detach())
                    for name, value in components.items()
                },
                "gradient_norm": float(gradient_norm.detach()),
                "validation": validation,
                "anchor_validation": anchor_validation,
                "validation_score_components": score_components,
            }
            history.append(row)
            checkpoint = _checkpoint(
                model,
                optimizer,
                step,
                spec,
                core_checkpoint,
                file_sha256(cache_manifest_path),
                file_sha256(supervision_manifest_path),
                initialization,
            )
            torch.save(checkpoint, output_dir / "latest.pt")
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_validation = validation
                best_anchor_validation = anchor_validation
                torch.save(checkpoint, output_dir / "best.pt")
            target = 1.0 if overfit_mode else 0.995
            ready = all(value >= target for value in score_components.values())
            if (
                ready
                and model.config.compiler != "flat"
                and overfit_mode
            ):
                ready = (
                    anchor_validation["all_anchor_sequence_exact"] == 1.0
                )
            ready_streak = ready_streak + 1 if ready else 0
            _write_json(
                output_dir / "progress.json",
                {
                    "stage": "A1.20C",
                    "arm": f"{spec.compiler}__{spec.credit_mode}",
                    "step": step,
                    "steps_requested": spec.steps,
                    "best_step": best_step,
                    "best_score": best_score,
                    "elapsed_seconds": time.perf_counter() - started,
                    "overfit_mode": overfit_mode,
                },
            )
            if overfit_mode and ready_streak >= 2:
                break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    _write_json(output_dir / "history.json", history)
    checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_boundary_state_dict(checkpoint["boundary"])
    model.eval()
    validation = evaluate_a120b_matrix(
        model,
        validation_dataset,
        device,
        batch_size=spec.batch_size,
    )
    anchor_validation = evaluate_a120c_anchors(
        model,
        validation_dataset,
        validation_supervision,
        device,
        batch_size=spec.batch_size,
    )
    _, eligibility_components = _validation_score(validation)
    core_hash_after = module_state_sha256(model.core)
    overfit_gates = {
        "all_cell_mapping_exact": all(
            row["mapping_accuracy"]["minimum"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_cell_trajectory_exact": all(
            row["trajectory_full_exact"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_cell_final_state_exact": all(
            row["final_state_full_exact"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_cell_answer_exact": all(
            row["final_answer_accuracy"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "anchor_sequence_exact": model.config.compiler == "flat"
        or anchor_validation["all_anchor_sequence_exact"] == 1.0,
        "core_hash_unchanged": core_hash_after == core_hash,
        "architecture_integrity": model.integrity_report()["passed"],
        "hard_forward_state_delta_at_most_1e_6": maximum_hard_state_delta
        <= 1e-6,
        "hard_forward_answer_delta_at_most_1e_6": maximum_hard_answer_delta
        <= 1e-6,
    }
    result = {
        "schema_version": RESULTS_SCHEMA,
        "stage": "V2-A1.20C structured compiler and credit bridge",
        "arm": f"{spec.compiler}__{spec.credit_mode}",
        "selection": cache_manifest["selection"],
        "train_spec": asdict(spec),
        "config": asdict(model.config),
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": completed_step,
            "processed_examples": completed_step
            * (len(train_dataset) if overfit_mode else spec.batch_size),
            "seconds": training_seconds,
            "steps_per_second": completed_step
            / max(training_seconds, 1e-9),
            "fresh_initialization": initialization["fresh"],
            "initialization": initialization,
            "overfit_mode": overfit_mode,
        },
        "best_checkpoint_step": best_step,
        "best_validation_at_save": best_validation,
        "best_anchor_validation_at_save": best_anchor_validation,
        "validation": validation,
        "anchor_validation": anchor_validation,
        "eligibility_components": eligibility_components,
        "formal_eligibility_passed": all(
            value >= 0.995 for value in eligibility_components.values()
        ),
        "hard_forward_equivalence": {
            "maximum_state_logit_abs_delta": maximum_hard_state_delta,
            "maximum_answer_logit_abs_delta": maximum_hard_answer_delta,
            "gate": maximum_hard_state_delta <= 1e-6
            and maximum_hard_answer_delta <= 1e-6,
        },
        "integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(),
        "optimization": {
            "scope": spec.optimization_scope,
            "optimized_parameter_names": optimized_parameter_names,
            "optimized_parameters": sum(
                parameter.numel() for parameter in trainable
            ),
        },
        "core_state_sha256_before": core_hash,
        "core_state_sha256_after": core_hash_after,
        "overfit_gates": overfit_gates if overfit_mode else None,
        "overfit_passed": all(overfit_gates.values())
        if overfit_mode
        else None,
    }
    _write_json(output_dir / "results.json", result)
    return result


def load_a120c_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
) -> A120CFullTextBoundary:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("not an A1.20C checkpoint")
    core_path = Path(checkpoint["core_checkpoint"])
    if file_sha256(core_path) != checkpoint["core_checkpoint_sha256"]:
        raise ValueError("A1.20C core checkpoint hash mismatch")
    core = load_a119h2_deployment(core_path, device)
    if module_state_sha256(core) != checkpoint["core_state_sha256"]:
        raise ValueError("A1.20C core state hash mismatch")
    model = A120CFullTextBoundary(
        A120CConfig(**checkpoint["config"]), core
    ).to(device)
    model.load_boundary_state_dict(checkpoint["boundary"])
    model.a120c_reader_seed = int(
        checkpoint["train_spec"]["reader_seed"]
    )
    model.a120c_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a120c_core_model_seed = int(
        checkpoint["train_spec"]["core_model_seed"]
    )
    model.eval()
    return model
