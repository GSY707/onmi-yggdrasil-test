from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_5_data import (
    ANSWER_LABELS,
    REGISTER_NAMES,
    execute_operations,
    load_a15_records,
    render_state_labels,
)
from .a1_5_model import P0Config, StructuredRecurrentCore


FAMILY_TO_ID = {"noop": 0, "swap": 1, "copy": 2}
REGISTER_TO_ID = {name: index for index, name in enumerate(REGISTER_NAMES)}
NOOP_REGISTER_ID = len(REGISTER_NAMES)


def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch[key] for key in (
        "start_values",
        "query_register",
        "operation_family",
        "operation_source",
        "operation_target",
        "operation_mask",
    )}


def _clone_record_with_operations(record: dict[str, Any], operations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cloned = copy.deepcopy(record)
    cloned["operations"] = [dict(operation) for operation in operations]
    trajectory = execute_operations(cloned["start_state"], cloned["operations"])
    cloned["state_trajectory"] = trajectory[1:]
    cloned["answer"] = trajectory[-1][cloned["query_register"]]
    cloned["answer_index"] = ANSWER_LABELS.index(cloned["answer"])
    cloned["program_length"] = len(operations)
    cloned["operation_mask"] = [1] * len(operations)
    return cloned


def encode_records(records: Sequence[dict[str, Any]], device: str | torch.device) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot encode an empty record batch")
    max_steps = max(len(record["operations"]) for record in records)
    batch = len(records)
    start_values = torch.empty(batch, len(REGISTER_NAMES), dtype=torch.long, device=device)
    query_register = torch.empty(batch, dtype=torch.long, device=device)
    operation_family = torch.zeros(batch, max_steps, dtype=torch.long, device=device)
    operation_source = torch.full((batch, max_steps), NOOP_REGISTER_ID, dtype=torch.long, device=device)
    operation_target = torch.full((batch, max_steps), NOOP_REGISTER_ID, dtype=torch.long, device=device)
    operation_mask = torch.zeros(batch, max_steps, dtype=torch.bool, device=device)
    state_targets = torch.zeros(batch, max_steps, len(REGISTER_NAMES), dtype=torch.long, device=device)
    for row, record in enumerate(records):
        start_values[row] = torch.tensor([ANSWER_LABELS.index(record["start_state"][name]) for name in REGISTER_NAMES])
        query_register[row] = REGISTER_TO_ID[record["query_register"]]
        for step, operation in enumerate(record["operations"]):
            operation_family[row, step] = FAMILY_TO_ID[operation["family"]]
            operation_source[row, step] = REGISTER_TO_ID.get(operation.get("source"), NOOP_REGISTER_ID)
            operation_target[row, step] = REGISTER_TO_ID.get(operation.get("target"), NOOP_REGISTER_ID)
            operation_mask[row, step] = True
        labels = render_state_labels(record)
        if labels:
            state_targets[row, : len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    answer_targets = torch.tensor(
        [int(record["answer_index"]) for record in records], dtype=torch.long, device=device
    )
    return {
        "start_values": start_values,
        "query_register": query_register,
        "operation_family": operation_family,
        "operation_source": operation_source,
        "operation_target": operation_target,
        "operation_mask": operation_mask,
        "state_targets": state_targets,
        "answer_targets": answer_targets,
    }


def _batches(records: Sequence[dict[str, Any]], batch_size: int, *, rng: random.Random | None = None) -> list[list[dict[str, Any]]]:
    indexes = list(range(len(records)))
    if rng is not None:
        rng.shuffle(indexes)
    return [[records[index] for index in indexes[start : start + batch_size]] for start in range(0, len(indexes), batch_size)]


@torch.no_grad()
def evaluate_p0(
    model: StructuredRecurrentCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 64,
    disable_transition_delta: bool = False,
) -> dict[str, Any]:
    model.eval()
    answer_correct = 0
    answer_total = 0
    state_token_correct = 0
    state_token_total = 0
    state_exact = 0
    state_examples = 0
    transition_delta_norms: list[float] = []
    for batch_records in _batches(records, batch_size):
        batch = encode_records(batch_records, device)
        output = model(**_model_inputs(batch), disable_transition_delta=disable_transition_delta, return_trajectory=True)
        logits = output["logits"]
        predicted = logits.argmax(dim=-1)
        answer_targets = batch["answer_targets"]
        answer_correct += int((predicted == answer_targets).sum())
        answer_total += len(batch_records)
        trajectory = output["trajectory"]
        state_logits = model.decode_trajectory(trajectory)
        state_targets = batch["state_targets"]
        mask = batch["operation_mask"]
        if state_logits.numel():
            state_pred = state_logits.argmax(dim=-1)
            state_token_correct += int(((state_pred == state_targets) & mask.unsqueeze(-1)).sum())
            state_token_total += int(mask.sum()) * len(REGISTER_NAMES)
            per_example_exact = ((state_pred == state_targets) | ~mask.unsqueeze(-1)).all(dim=(-1, -2))
            state_exact += int(per_example_exact.sum())
            state_examples += len(batch_records)
        transition_delta_norms.append(float(model.transition.residual_scale.detach().abs()))
    return {
        "final_answer_accuracy": answer_correct / answer_total if answer_total else 0.0,
        "final_answer_correct": answer_correct,
        "final_answer_total": answer_total,
        "state_token_accuracy": state_token_correct / state_token_total if state_token_total else 0.0,
        "state_full_exact": state_exact / state_examples if state_examples else 1.0,
        "state_exact_examples": state_exact,
        "state_exact_total": state_examples,
        "transition_residual_scale_abs": sum(transition_delta_norms) / len(transition_delta_norms)
        if transition_delta_norms
        else 0.0,
        "transition_delta_disabled": disable_transition_delta,
    }


@torch.no_grad()
def _predict_records(
    model: StructuredRecurrentCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 64,
    disable_transition_delta: bool = False,
) -> list[int]:
    model.eval()
    predictions: list[int] = []
    for batch_records in _batches(records, batch_size):
        batch = encode_records(batch_records, device)
        output = model(**_model_inputs(batch), disable_transition_delta=disable_transition_delta)
        predictions.extend(output["logits"].argmax(dim=-1).tolist())
    return predictions


@torch.no_grad()
def evaluate_interventions(
    model: StructuredRecurrentCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 64,
    seed: int = 20260713,
) -> dict[str, Any]:
    baseline_predictions = _predict_records(model, records, device, batch_size=batch_size)

    def score_modified(name: str, modified: list[dict[str, Any]], *, compare_to_baseline: bool = True) -> dict[str, Any]:
        predictions = _predict_records(model, modified, device, batch_size=batch_size)
        expected = [record["answer_index"] for record in modified]
        correct = sum(prediction == target for prediction, target in zip(predictions, expected))
        changed = sum(a != b for a, b in zip(baseline_predictions, predictions))
        return {
            "final_answer_accuracy": correct / len(modified) if modified else 0.0,
            "changed_prediction_rate": changed / len(modified) if modified and compare_to_baseline else None,
            "examples": len(modified),
            "expected_counterfactual_answer": True,
        }

    replace_noop: list[dict[str, Any]] = []
    delete_step: list[dict[str, Any]] = []
    truncate: list[dict[str, Any]] = []
    shuffled: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for record in records:
        operations = [dict(operation) for operation in record["operations"]]
        for index in range(len(operations)):
            no_op = [dict(operation) for operation in operations]
            no_op[index] = {"family": "noop", "source": None, "target": None, "text": "NOOP."}
            replace_noop.append(_clone_record_with_operations(record, no_op))
            delete_step.append(_clone_record_with_operations(record, operations[:index] + operations[index + 1 :]))
        prefix_length = max(0, len(operations) // 2)
        truncate.append(_clone_record_with_operations(record, operations[:prefix_length]))
        permutation = list(range(len(operations)))
        rng.shuffle(permutation)
        shuffled.append(_clone_record_with_operations(record, [operations[index] for index in permutation]))

    return {
        "normal": evaluate_p0(model, records, device, batch_size=batch_size),
        "replace_one_step_with_noop": score_modified("replace", replace_noop),
        "delete_one_step": score_modified("delete", delete_step),
        "shuffle_operations": score_modified("shuffle", shuffled),
        "trajectory_half": score_modified("truncate", truncate),
        "t0": score_modified("t0", [_clone_record_with_operations(record, []) for record in records]),
        "t1": score_modified(
            "t1", [_clone_record_with_operations(record, record["operations"][:1]) for record in records]
        ),
        "transition_delta_disabled": evaluate_p0(
            model, records, device, batch_size=batch_size, disable_transition_delta=True
        ),
        "all_core_samples_step_necessary": True,
    }


def _losses(
    model: StructuredRecurrentCore,
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    state_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    answer_loss = nn.functional.cross_entropy(output["logits"], batch["answer_targets"])
    trajectory = output["trajectory"]
    state_logits = model.decode_trajectory(trajectory)
    state_targets = batch["state_targets"]
    mask = batch["operation_mask"]
    if state_logits.numel() and mask.any():
        token_losses = nn.functional.cross_entropy(
            state_logits.reshape(-1, len(ANSWER_LABELS)),
            state_targets.reshape(-1),
            reduction="none",
        ).view(state_logits.shape[0], state_logits.shape[1], len(REGISTER_NAMES))
        state_loss = (token_losses * mask.unsqueeze(-1)).sum() / (mask.sum() * len(REGISTER_NAMES)).clamp_min(1)
    else:
        state_loss = torch.zeros((), device=answer_loss.device)
    total = answer_loss + state_weight * state_loss
    return total, {"loss": float(total.detach()), "answer_loss": float(answer_loss.detach()), "state_loss": float(state_loss.detach())}


def _gradient_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return math.sqrt(total)


def train_p0(
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    config: P0Config | None = None,
    seed: int = 20260713,
    device: str = "cuda",
    steps: int = 3000,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    state_weight: float = 1.0,
    early_stop_overfit: bool = False,
    checkpoint_interval: int = 100,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = StructuredRecurrentCore(config or P0Config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    best_validation = -1.0
    best_state_exact = -1.0
    started = time.perf_counter()
    overfit_streak = 0
    for step in range(1, steps + 1):
        model.train()
        batch_records = [train_records[rng.randrange(len(train_records))] for _ in range(batch_size)]
        batch = encode_records(batch_records, device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        total_loss, metrics = _losses(model, output, batch, state_weight)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = _gradient_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics.update(
            {
                "step": step,
                "gradient_norm": grad_norm,
                "residual_scale": float(model.transition.residual_scale.detach()),
            }
        )
        history.append(metrics)

        if step % checkpoint_interval == 0 or step == steps:
            train_metrics = evaluate_p0(model, train_records, device, batch_size=batch_size)
            validation_metrics = evaluate_p0(model, validation_records, device, batch_size=batch_size)
            metrics["train_answer_accuracy"] = train_metrics["final_answer_accuracy"]
            metrics["train_state_full_exact"] = train_metrics["state_full_exact"]
            metrics["validation_answer_accuracy"] = validation_metrics["final_answer_accuracy"]
            metrics["validation_state_full_exact"] = validation_metrics["state_full_exact"]
            checkpoint = {
                "schema_version": "yggdrasil.v2-a1.5.p0.checkpoint.v1",
                "step": step,
                "config": model.config.__dict__,
                "seed": seed,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, output_dir / "latest.pt")
            validation_score = validation_metrics["final_answer_accuracy"]
            if validation_score > best_validation or (
                validation_score == best_validation and validation_metrics["state_full_exact"] > best_state_exact
            ):
                best_validation = validation_score
                best_state_exact = validation_metrics["state_full_exact"]
                torch.save(checkpoint, output_dir / "best.pt")
            if early_stop_overfit and train_metrics["final_answer_accuracy"] >= 0.99 and train_metrics["state_full_exact"] >= 0.99:
                overfit_streak += 1
            else:
                overfit_streak = 0
            if early_stop_overfit and overfit_streak >= 2:
                break
    elapsed = time.perf_counter() - started
    torch.save(model.state_dict(), output_dir / "final_state.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": "yggdrasil.v2-a1.5.p0.results.v1",
        "evidence_level": "surrogate-probe",
        "stage": "P0 structured-input positive control",
        "config": model.config.__dict__,
        "training": {
            "seed": seed,
            "steps_requested": steps,
            "steps_completed": history[-1]["step"] if history else 0,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "state_supervision_weight": state_weight,
            "training_seconds": elapsed,
            "early_stop_overfit": early_stop_overfit,
        },
        "data": {"train_examples": len(train_records), "validation_examples": len(validation_records)},
        "parameters": model.parameter_report(),
        "integrity": model.integrity_report(),
        "fit": evaluate_p0(model, train_records, device, batch_size=batch_size),
        "validation": evaluate_p0(model, validation_records, device, batch_size=batch_size),
        "diagnostics": {
            "residual_scale": float(model.transition.residual_scale.detach()),
            "gate": "not used by design",
            "gradient_norm_last": history[-1].get("gradient_norm") if history else None,
            "history_path": str(output_dir / "history.json"),
        },
        "recovery": {"latest_checkpoint": str(output_dir / "latest.pt"), "best_checkpoint": str(output_dir / "best.pt")},
        "boundaries": [
            "P0 uses structured symbolic inputs and is a positive control, not Qwen hidden-to-latent evidence.",
            "A1.5 requires >=99% final and state exact on the 32-example overfit before P1.",
            "Single-seed results are not architecture-formal evidence.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_p0_checkpoint(path: Path, device: str = "cuda") -> StructuredRecurrentCore:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = StructuredRecurrentCore(P0Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
