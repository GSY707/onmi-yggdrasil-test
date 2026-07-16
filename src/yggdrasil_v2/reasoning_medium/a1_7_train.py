from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_7_core import A17CoreConfig, A17RelationAddressedCore
from .a1_7_data import (
    FAMILY_TO_INDEX,
    REGISTER_NAMES,
    SUPPORTED_JOINTS,
    VALUE_LABELS,
    execute_operations,
)


CLOSURE_WEIGHT = 1.0


def encode_a17_records(records: Sequence[dict[str, Any]], device: str | torch.device) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot encode an empty record batch")
    batch_size = len(records)
    max_steps = max(int(record["program_length"]) for record in records)
    starts = torch.tensor(
        [[VALUE_LABELS.index(record["start_state"][name]) for name in REGISTER_NAMES] for record in records],
        dtype=torch.long,
        device=device,
    )
    query = torch.tensor([REGISTER_NAMES.index(record["query_register"]) for record in records], dtype=torch.long, device=device)
    family = torch.zeros((batch_size, max_steps), dtype=torch.long, device=device)
    source = torch.zeros_like(family)
    target = torch.zeros_like(family)
    mask = torch.zeros((batch_size, max_steps), dtype=torch.bool, device=device)
    state_targets = torch.zeros((batch_size, max_steps, len(REGISTER_NAMES)), dtype=torch.long, device=device)
    for row, record in enumerate(records):
        length = int(record["program_length"])
        mask[row, :length] = True
        for step, operation in enumerate(record["operations"]):
            family[row, step] = FAMILY_TO_INDEX[operation["family"]]
            source[row, step] = REGISTER_NAMES.index(operation["source"])
            target[row, step] = REGISTER_NAMES.index(operation["target"])
        for step, state in enumerate(record["state_trajectory"]):
            state_targets[row, step] = torch.tensor([VALUE_LABELS.index(state[name]) for name in REGISTER_NAMES], device=device)
    return {
        "start_values": starts,
        "query_register": query,
        "operation_family": family,
        "operation_source": source,
        "operation_target": target,
        "operation_mask": mask,
        "state_targets": state_targets,
        "answer_targets": torch.tensor([int(record["answer_index"]) for record in records], dtype=torch.long, device=device),
    }


def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for key in (
            "start_values",
            "query_register",
            "operation_family",
            "operation_source",
            "operation_target",
            "operation_mask",
        )
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def closure_targets(model: A17RelationAddressedCore, state_targets: torch.Tensor) -> torch.Tensor:
    """Stop-gradient canonical targets; only predicted trajectory receives closure gradients."""
    return model.canonical_value_prototypes[state_targets].detach()


def compute_a17_loss(
    model: A17RelationAddressedCore,
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    *,
    closure_weight: float = CLOSURE_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["operation_mask"]
    state_ce = nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, len(VALUE_LABELS)),
        batch["state_targets"].reshape(-1),
        reduction="none",
    ).reshape(mask.shape[0], mask.shape[1], len(REGISTER_NAMES))
    state_loss = _masked_mean(state_ce.mean(dim=-1), mask)
    source_ce = nn.functional.cross_entropy(
        output["source_pointer_logits"].reshape(-1, len(REGISTER_NAMES)),
        batch["operation_source"].reshape(-1),
        reduction="none",
    ).reshape(mask.shape)
    target_ce = nn.functional.cross_entropy(
        output["target_pointer_logits"].reshape(-1, len(REGISTER_NAMES)),
        batch["operation_target"].reshape(-1),
        reduction="none",
    ).reshape(mask.shape)
    source_loss = _masked_mean(source_ce, mask)
    target_loss = _masked_mean(target_ce, mask)
    answer_loss = nn.functional.cross_entropy(output["answer_logits"], batch["answer_targets"])
    target_prototypes = closure_targets(model, batch["state_targets"])
    cosine = nn.functional.cosine_similarity(output["trajectory"], target_prototypes, dim=-1)
    closure_loss = _masked_mean((1.0 - cosine).mean(dim=-1), mask)
    total = state_loss + 0.25 * source_loss + 0.25 * target_loss + 0.5 * answer_loss + closure_weight * closure_loss
    return total, {
        "loss": float(total.detach()),
        "state_loss": float(state_loss.detach()),
        "source_pointer_loss": float(source_loss.detach()),
        "target_pointer_loss": float(target_loss.detach()),
        "answer_loss": float(answer_loss.detach()),
        "closure_loss": float(closure_loss.detach()),
        "closure_weight": closure_weight,
    }


def _last_active_state_logits(
    model: A17RelationAddressedCore,
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    lengths = batch["operation_mask"].sum(dim=1)
    if output["state_logits"].shape[1] == 0:
        return model.shared_state_head(output["initial_slots"])
    last_indices = (lengths - 1).clamp_min(0)
    gather_index = last_indices.view(-1, 1, 1, 1).expand(-1, 1, len(REGISTER_NAMES), len(VALUE_LABELS))
    gathered = output["state_logits"].gather(1, gather_index).squeeze(1)
    if bool((lengths == 0).any()):
        initial_logits = model.shared_state_head(output["initial_slots"])
        gathered = torch.where((lengths == 0).view(-1, 1, 1), initial_logits, gathered)
    return gathered


def _last_state_targets(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    lengths = batch["operation_mask"].sum(dim=1)
    if batch["state_targets"].shape[1] == 0:
        return batch["start_values"]
    last_indices = (lengths - 1).clamp_min(0)
    gather_index = last_indices.view(-1, 1, 1).expand(-1, 1, len(REGISTER_NAMES))
    targets = batch["state_targets"].gather(1, gather_index).squeeze(1)
    return torch.where((lengths == 0).view(-1, 1), batch["start_values"], targets)


@torch.no_grad()
def prototype_statistics(model: A17RelationAddressedCore) -> dict[str, Any]:
    prototypes = model.canonical_value_prototypes.detach()
    normalized = nn.functional.normalize(prototypes, dim=-1)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)]
    distances = torch.cdist(prototypes, prototypes)
    off_distances = distances[~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)]
    maximum_cosine = float(off_diagonal.max())
    minimum_l2_distance = float(off_distances.min())
    gate = maximum_cosine < 0.95 and minimum_l2_distance > 0.50
    return {
        "maximum_pairwise_cosine": maximum_cosine,
        "minimum_pairwise_l2_distance": minimum_l2_distance,
        "mean_prototype_norm": float(prototypes.norm(dim=-1).mean()),
        "gate_thresholds": {"maximum_pairwise_cosine_lt": 0.95, "minimum_pairwise_l2_distance_gt": 0.50},
        "non_collapse_gate": gate,
    }


@torch.no_grad()
def evaluate_a17_core(
    model: A17RelationAddressedCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 128,
    closure_weight: float = CLOSURE_WEIGHT,
) -> dict[str, Any]:
    model.eval()
    trajectory_full = final_state_full = final_query_correct = 0
    state_tokens = state_total = source_correct = target_correct = pointer_total = 0
    identity_max_abs_diff = 0.0
    identity_predictions_equal = True
    margins: list[float] = []
    closure_values: list[float] = []
    losses: list[float] = []
    for start in range(0, len(records), batch_size):
        batch = encode_a17_records(records[start : start + batch_size], device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        loss, loss_metrics = compute_a17_loss(model, output, batch, closure_weight=closure_weight)
        losses.append(float(loss))
        closure_values.append(loss_metrics["closure_loss"])
        valid = batch["operation_mask"]
        state_pred = output["state_logits"].argmax(-1)
        state_equal = state_pred == batch["state_targets"]
        state_tokens += int((state_equal & valid.unsqueeze(-1)).sum())
        state_total += int(valid.sum()) * len(REGISTER_NAMES)
        trajectory_full += int((state_equal | ~valid.unsqueeze(-1)).all(dim=(-1, -2)).sum())
        last_logits = _last_active_state_logits(model, output, batch)
        last_targets = _last_state_targets(batch)
        final_state_full += int((last_logits.argmax(-1) == last_targets).all(dim=-1).sum())
        query_index = batch["query_register"].view(-1, 1, 1).expand(-1, 1, len(VALUE_LABELS))
        queried_last_logits = last_logits.gather(1, query_index).squeeze(1)
        difference = float((output["answer_logits"] - queried_last_logits).abs().max())
        identity_max_abs_diff = max(identity_max_abs_diff, difference)
        identity_predictions_equal = identity_predictions_equal and bool(torch.equal(output["answer_logits"].argmax(-1), queried_last_logits.argmax(-1)))
        final_query_correct += int((output["answer_logits"].argmax(-1) == batch["answer_targets"]).sum())
        source_pred = output["source_pointer_logits"].argmax(-1)
        target_pred = output["target_pointer_logits"].argmax(-1)
        source_correct += int(((source_pred == batch["operation_source"]) & valid).sum())
        target_correct += int(((target_pred == batch["operation_target"]) & valid).sum())
        pointer_total += int(valid.sum())
        predicted = nn.functional.normalize(output["trajectory"], dim=-1)
        prototypes = nn.functional.normalize(model.canonical_value_prototypes.detach(), dim=-1)
        similarities = torch.einsum("btrd,vd->btrv", predicted, prototypes)
        correct_similarity = similarities.gather(-1, batch["state_targets"].unsqueeze(-1)).squeeze(-1)
        incorrect = similarities.masked_fill(
            nn.functional.one_hot(batch["state_targets"], num_classes=len(VALUE_LABELS)).bool(),
            -float("inf"),
        ).max(dim=-1).values
        margin = correct_similarity - incorrect
        margins.extend(margin[valid.unsqueeze(-1).expand_as(margin)].detach().cpu().tolist())
    margins.sort()
    p05_index = min(len(margins) - 1, max(0, int(0.05 * len(margins)))) if margins else 0
    identity_gate = identity_max_abs_diff <= 1e-6 and identity_predictions_equal
    return {
        "examples": len(records),
        "loss": sum(losses) / max(1, len(losses)),
        "closure_loss": sum(closure_values) / max(1, len(closure_values)),
        "trajectory_full_exact": trajectory_full / max(1, len(records)),
        "final_state_full_exact": final_state_full / max(1, len(records)),
        "final_query_token_accuracy": final_query_correct / max(1, len(records)),
        "state_token_accuracy": state_tokens / max(1, state_total),
        "source_pointer_accuracy": source_correct / max(1, pointer_total),
        "target_pointer_accuracy": target_correct / max(1, pointer_total),
        "answer_state_logits_identity": {
            "max_abs_diff": identity_max_abs_diff,
            "prediction_equal": identity_predictions_equal,
            "gate": identity_gate,
            "method": "answer_logits versus shared_state_head logits gathered at each sample's last active step and queried register",
        },
        "prototype_margin": {
            "mean_correct_minus_nearest_wrong_cosine": sum(margins) / max(1, len(margins)),
            "p05_correct_minus_nearest_wrong_cosine": margins[p05_index] if margins else None,
            "samples": len(margins),
        },
        "prototype_statistics": prototype_statistics(model),
    }


def _gradient_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float(parameter.grad.detach().pow(2).sum()) for parameter in model.parameters() if parameter.grad is not None))


def _checkpoint(
    model: A17RelationAddressedCore,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: A17CoreConfig,
    seed: int,
    closure_weight: float,
) -> dict[str, Any]:
    return {
        "schema_version": "yggdrasil.v2-a1.7.c0.checkpoint.v1",
        "step": step,
        "config": asdict(config),
        "seed": seed,
        "fresh_initialization": True,
        "closure_weight": closure_weight,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def train_a17_c0(
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    config: A17CoreConfig | None = None,
    seed: int = 20260715,
    device: str = "cuda",
    steps: int = 6000,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    validation_interval: int = 100,
    early_stop_exact: float | None = None,
    early_stop_patience: int | None = None,
    overfit_mode: bool = False,
    closure_weight: float = CLOSURE_WEIGHT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to resume or mix checkpoints in existing output directory: {output_dir}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA for A1.7 C0 but CUDA is unavailable")
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    config = config or A17CoreConfig()
    model = A17RelationAddressedCore(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    patience = exact_streak = 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        batch_records = [train_records[rng.randrange(len(train_records))] for _ in range(batch_size)]
        batch = encode_a17_records(batch_records, device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        loss, metrics = compute_a17_loss(model, output, batch, closure_weight=closure_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics["gradient_norm"] = _gradient_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics["step"] = step
        if step % validation_interval == 0 or step == steps:
            train_metrics = evaluate_a17_core(model, train_records, device, batch_size=batch_size, closure_weight=closure_weight)
            validation_metrics = evaluate_a17_core(model, validation_records, device, batch_size=batch_size, closure_weight=closure_weight)
            metrics["train"] = train_metrics
            metrics["validation"] = validation_metrics
            scoring_metrics = train_metrics if overfit_mode else validation_metrics
            score = scoring_metrics["trajectory_full_exact"] + 0.25 * scoring_metrics["source_pointer_accuracy"] + 0.25 * scoring_metrics["target_pointer_accuracy"]
            checkpoint = _checkpoint(model, optimizer, step, config, seed, closure_weight)
            torch.save(checkpoint, output_dir / "latest.pt")
            if score > best_score:
                best_score = score
                best_step = step
                best_validation = validation_metrics
                torch.save(checkpoint, output_dir / "best.pt")
                patience = 0
            else:
                patience += 1
            if early_stop_exact is not None and all(
                scoring_metrics[key] >= early_stop_exact
                for key in (
                    "final_query_token_accuracy",
                    "state_token_accuracy",
                    "trajectory_full_exact",
                    "source_pointer_accuracy",
                    "target_pointer_accuracy",
                )
            ):
                exact_streak += 1
            else:
                exact_streak = 0
            history.append(metrics)
            if early_stop_exact is not None and exact_streak >= 2:
                break
            if early_stop_patience is not None and patience >= early_stop_patience:
                break
        else:
            history.append(metrics)
        (output_dir / "progress.json").write_text(
            json.dumps({"step": step, "steps_requested": steps, "best_score": best_score, "elapsed_seconds": time.perf_counter() - started}, indent=2) + "\n",
            encoding="utf-8",
        )
    if not (output_dir / "best.pt").exists():
        raise RuntimeError("A1.7 C0 training completed without a best checkpoint")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    result = {
        "schema_version": "yggdrasil.v2-a1.7.c0.results.v1",
        "stage": "A1.7 address-content-separated continuous core",
        "fresh_initialization": True,
        "config": asdict(config),
        "training": {
            "seed": seed,
            "steps_requested": steps,
            "steps_completed": history[-1]["step"],
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "closure_weight": closure_weight,
            "fresh_initialization": True,
            "resume_checkpoint": None,
            "overfit_mode": overfit_mode,
            "seconds": time.perf_counter() - started,
        },
        "best_score": best_score,
        "best_checkpoint_step": best_step,
        "fit": evaluate_a17_core(model, train_records, device, batch_size=batch_size, closure_weight=closure_weight),
        "validation": evaluate_a17_core(model, validation_records, device, batch_size=batch_size, closure_weight=closure_weight),
        "integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(),
        "best_validation_at_save": best_validation,
        "artifacts": {
            "latest": str(output_dir / "latest.pt"),
            "best": str(output_dir / "best.pt"),
            "history": str(output_dir / "history.json"),
            "progress": str(output_dir / "progress.json"),
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def load_a17_checkpoint(path: Path, device: str | torch.device = "cpu") -> A17RelationAddressedCore:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = A17RelationAddressedCore(A17CoreConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.training_closure_weight = float(checkpoint.get("closure_weight", CLOSURE_WEIGHT))
    model.eval()
    return model


def _clone_with_operations(record: dict[str, Any], operations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    clone = dict(record)
    clone["operations"] = [dict(operation) for operation in operations]
    clone["operation_mask"] = [True] * len(operations)
    trajectory = execute_operations(clone["start_state"], clone["operations"])
    clone["state_trajectory"] = trajectory[1:]
    clone["program_length"] = len(operations)
    clone["answer"] = trajectory[-1][clone["query_register"]]
    clone["answer_index"] = VALUE_LABELS.index(clone["answer"])
    return clone


@torch.no_grad()
def _score_intervention(
    model: A17RelationAddressedCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    metrics = evaluate_a17_core(
        model,
        records,
        device,
        batch_size=batch_size,
        closure_weight=float(getattr(model, "training_closure_weight", CLOSURE_WEIGHT)),
    )
    return {
        "examples": metrics["examples"],
        "trajectory_full_exact": metrics["trajectory_full_exact"],
        "final_state_full_exact": metrics["final_state_full_exact"],
        "final_query_token_accuracy": metrics["final_query_token_accuracy"],
        "answer_state_logits_identity": metrics["answer_state_logits_identity"],
    }


@torch.no_grad()
def run_a17_interventions(
    model: A17RelationAddressedCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    prefix_records: list[dict[str, Any]] = []
    for record in records:
        for length in range(1, record["program_length"] + 1):
            prefix_records.append(_clone_with_operations(record, record["operations"][:length]))
    replacement: list[dict[str, Any]] = []
    deletion: list[dict[str, Any]] = []
    shuffled: list[dict[str, Any]] = []
    rng = random.Random(20260715)
    for record in records:
        for index, operation in enumerate(record["operations"]):
            current = (operation["family"], operation["source"], operation["target"])
            candidates = [joint for joint in SUPPORTED_JOINTS if joint != current]
            alternate = _operation_dict(rng.choice(candidates))
            replacement.append(_clone_with_operations(record, [alternate if i == index else op for i, op in enumerate(record["operations"])]))
            remaining = [op for i, op in enumerate(record["operations"]) if i != index]
            if remaining:
                deletion.append(_clone_with_operations(record, remaining))
        order = list(range(record["program_length"]))
        if len(order) > 1:
            shifted = order[1:] + order[:1]
            shuffled.append(_clone_with_operations(record, [record["operations"][i] for i in shifted]))
    prefix = _score_intervention(model, prefix_records, device, batch_size=batch_size)
    replacement_score = _score_intervention(model, replacement, device, batch_size=batch_size)
    deletion_score = _score_intervention(model, deletion, device, batch_size=batch_size)
    shuffled_score = _score_intervention(model, shuffled, device, batch_size=batch_size)
    gates = {
        "prefix_fidelity": prefix["final_state_full_exact"] >= 0.99,
        "replacement_trajectory_full": replacement_score["trajectory_full_exact"] >= 0.95,
        "deletion_trajectory_full": deletion_score["trajectory_full_exact"] >= 0.95,
        "shuffled_trajectory_full": shuffled_score["trajectory_full_exact"] >= 0.95,
    }
    return {
        "prefix_fidelity": prefix,
        "replacement": replacement_score,
        "deletion": deletion_score,
        "shuffled_program": shuffled_score,
        "oracle_recomputed_after_each_intervention": True,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _operation_dict(joint: tuple[str, str, str]) -> dict[str, Any]:
    family, source, target = joint
    text = f"COPY {source} -> {target}." if family == "copy" else f"SWAP {source} <-> {target}."
    return {"family": family, "source": source, "target": target, "text": text, "template_id": 0}
