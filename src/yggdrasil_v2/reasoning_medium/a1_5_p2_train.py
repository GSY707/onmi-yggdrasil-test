from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_5_data import ANSWER_LABELS, REGISTER_NAMES
from .a1_5_p1 import P1HiddenLatentReasoner
from .a1_5_p1_train import _cache_records
from .a1_5_p2 import LearnedMultiSlotWorkspace, P2Config
from .a1_5_train import _clone_record_with_operations, encode_records


def _batch_items(
    items: Sequence[dict[str, Any]], indexes: Sequence[int], device: str
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    selected = [items[index] for index in indexes]
    records = [item["record"] for item in selected]
    max_tokens = max(item["last_hidden"].shape[0] for item in selected)
    max_steps = max(item["operation_token_mask"].shape[0] for item in selected)
    hidden_width = selected[0]["last_hidden"].shape[-1]
    hidden = torch.zeros(len(selected), max_tokens, hidden_width, dtype=torch.float16, device=device)
    source_mask = torch.zeros(len(selected), max_tokens, dtype=torch.bool, device=device)
    operation_spans = torch.zeros(len(selected), max_steps, max_tokens, dtype=torch.bool, device=device)
    operation_mask = torch.zeros(len(selected), max_steps, dtype=torch.bool, device=device)
    for row, item in enumerate(selected):
        token_count = item["last_hidden"].shape[0]
        step_count = item["operation_token_mask"].shape[0]
        hidden[row, :token_count] = item["last_hidden"].to(device)
        source_mask[row, :token_count] = item["attention_mask"]
        operation_spans[row, :step_count, : item["operation_token_mask"].shape[1]] = item["operation_token_mask"]
        operation_mask[row, :step_count] = item["operation_mask"]
    labels = encode_records(records, device)
    label_steps = labels["operation_mask"].shape[1]
    if label_steps < max_steps:
        pad = max_steps - label_steps
        labels["state_targets"] = torch.nn.functional.pad(labels["state_targets"], (0, 0, 0, pad), value=0)
        labels["operation_mask"] = torch.nn.functional.pad(labels["operation_mask"], (0, pad), value=False)
    elif label_steps > max_steps:
        raise ValueError("cache operation mask is shorter than record operations")
    operation_hidden = P1HiddenLatentReasoner.pool_operation_hidden(hidden, operation_spans)
    return {
        "operation_hidden": operation_hidden,
        "operation_mask": operation_mask,
        "source_hidden": hidden,
        "source_attention_mask": source_mask,
        "labels": labels,
    }, records


def _loss(output: dict[str, Any], labels: dict[str, torch.Tensor], state_weight: float) -> torch.Tensor:
    answer_loss = nn.functional.cross_entropy(output["logits"], labels["answer_targets"])
    state_logits = output["state_logits"].view(
        output["state_logits"].shape[0], output["state_logits"].shape[1], len(REGISTER_NAMES), len(ANSWER_LABELS)
    )
    state_loss_per_token = nn.functional.cross_entropy(
        state_logits.reshape(-1, len(ANSWER_LABELS)),
        labels["state_targets"].reshape(-1),
        reduction="none",
    ).view(state_logits.shape[0], state_logits.shape[1], len(REGISTER_NAMES))
    mask = labels["operation_mask"]
    state_loss = (state_loss_per_token * mask.unsqueeze(-1)).sum() / (
        mask.sum() * len(REGISTER_NAMES)
    ).clamp_min(1)
    return answer_loss + state_weight * state_loss


@torch.no_grad()
def evaluate_p2(
    model: LearnedMultiSlotWorkspace,
    items: Sequence[dict[str, Any]],
    device: str,
    batch_size: int = 64,
    *,
    disable_transition_delta: bool = False,
    shuffle_transition_delta: bool = False,
    truncate_steps: int | None = None,
) -> dict[str, Any]:
    model.eval()
    answer_correct = 0
    answer_total = 0
    state_correct = 0
    state_total = 0
    state_exact = 0
    for start in range(0, len(items), batch_size):
        batch, records = _batch_items(items, range(start, min(start + batch_size, len(items))), device)
        operation_mask = batch["operation_mask"].clone()
        if truncate_steps is not None:
            operation_mask[:, truncate_steps:] = False
        output = model(
            batch["operation_hidden"],
            operation_mask,
            source_hidden=batch["source_hidden"],
            source_attention_mask=batch["source_attention_mask"],
            disable_transition_delta=disable_transition_delta,
            shuffle_transition_delta=shuffle_transition_delta,
            return_trajectory=True,
        )
        predictions = output["logits"].argmax(-1)
        answer_target = batch["labels"]["answer_targets"]
        answer_correct += int((predictions == answer_target).sum())
        answer_total += len(records)
        state_logits = output["state_logits"].view(
            output["state_logits"].shape[0], output["state_logits"].shape[1], len(REGISTER_NAMES), len(ANSWER_LABELS)
        )
        state_pred = state_logits.argmax(-1)
        mask = batch["labels"]["operation_mask"]
        state_correct += int(((state_pred == batch["labels"]["state_targets"]) & mask.unsqueeze(-1)).sum())
        state_total += int(mask.sum()) * len(REGISTER_NAMES)
        state_exact += int((((state_pred == batch["labels"]["state_targets"]) | ~mask.unsqueeze(-1)).all(dim=(-1, -2))).sum())
    return {
        "final_answer_accuracy": answer_correct / answer_total if answer_total else 0.0,
        "state_token_accuracy": state_correct / state_total if state_total else 0.0,
        "state_full_exact": state_exact / answer_total if answer_total else 0.0,
        "examples": answer_total,
        "disable_transition_delta": disable_transition_delta,
        "shuffle_transition_delta": shuffle_transition_delta,
        "truncate_steps": truncate_steps,
    }


@torch.no_grad()
def evaluate_same_answer_shuffle(
    model: LearnedMultiSlotWorkspace,
    items: Sequence[dict[str, Any]],
    device: str,
    batch_size: int = 64,
    seed: int = 20260713,
) -> dict[str, Any]:
    rng = random.Random(seed)
    groups: dict[int, list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(int(item["record"]["answer_index"]), []).append(index)
    shuffled = list(range(len(items)))
    for indexes in groups.values():
        permuted = list(indexes)
        rng.shuffle(permuted)
        for target, source in zip(indexes, permuted):
            shuffled[target] = source
    correct = 0
    changed = 0
    total = 0
    for start in range(0, len(items), batch_size):
        target_indexes = list(range(start, min(start + batch_size, len(items))))
        source_indexes = [shuffled[index] for index in target_indexes]
        target_batch, target_records = _batch_items(items, target_indexes, device)
        source_batch, _ = _batch_items(items, source_indexes, device)
        target_output = model(
            target_batch["operation_hidden"], target_batch["operation_mask"],
            source_hidden=target_batch["source_hidden"],
            source_attention_mask=target_batch["source_attention_mask"],
        )
        shuffled_output = model(
            source_batch["operation_hidden"], source_batch["operation_mask"],
            source_hidden=source_batch["source_hidden"],
            source_attention_mask=source_batch["source_attention_mask"],
        )
        target_pred = target_output["logits"].argmax(-1)
        shuffled_pred = shuffled_output["logits"].argmax(-1)
        labels = target_batch["labels"]["answer_targets"]
        correct += int((shuffled_pred == labels).sum())
        changed += int((target_pred != shuffled_pred).sum())
        total += len(target_records)
    return {
        "same_answer_shuffle_accuracy": correct / total if total else 0.0,
        "changed_prediction_rate": changed / total if total else 0.0,
        "examples": total,
    }


@torch.no_grad()
def evaluate_single_operation_intervention(
    model: LearnedMultiSlotWorkspace,
    items: Sequence[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    """Replace each step's operation hidden with another real operation span."""
    model.eval()
    changed = 0
    correct = 0
    state_correct = 0
    state_total = 0
    total = 0
    for item in items:
        batch, _ = _batch_items([item], [0], device)
        normal = model(
            batch["operation_hidden"], batch["operation_mask"],
            source_hidden=batch["source_hidden"],
            source_attention_mask=batch["source_attention_mask"],
        )["logits"].argmax(-1).item()
        operations = [dict(operation) for operation in item["record"]["operations"]]
        for step in range(len(operations)):
            source_step = (step + 1) % len(operations)
            modified_hidden = batch["operation_hidden"].clone()
            modified_hidden[:, step] = batch["operation_hidden"][:, source_step]
            output = model(
                modified_hidden,
                batch["operation_mask"],
                source_hidden=batch["source_hidden"],
                source_attention_mask=batch["source_attention_mask"],
                return_trajectory=True,
            )
            modified_ops = [dict(operation) for operation in operations]
            modified_ops[step] = dict(operations[source_step])
            modified_record = _clone_record_with_operations(item["record"], modified_ops)
            target = int(modified_record["answer_index"])
            prediction = output["logits"].argmax(-1).item()
            correct += int(prediction == target)
            changed += int(prediction != normal)
            modified_labels = encode_records([modified_record], device)
            state_logits = output["state_logits"].view(1, output["state_logits"].shape[1], len(REGISTER_NAMES), len(ANSWER_LABELS))
            state_pred = state_logits.argmax(-1)
            modified_targets = modified_labels["state_targets"]
            modified_mask = modified_labels["operation_mask"]
            if modified_targets.shape[1] < state_pred.shape[1]:
                pad = state_pred.shape[1] - modified_targets.shape[1]
                modified_targets = torch.nn.functional.pad(modified_targets, (0, 0, 0, pad), value=0)
                modified_mask = torch.nn.functional.pad(modified_mask, (0, pad), value=False)
            state_correct += int(((state_pred == modified_targets) & modified_mask.unsqueeze(-1)).sum())
            state_total += int(modified_mask.sum()) * len(REGISTER_NAMES)
            total += 1
    return {
        "counterfactual_final_accuracy": correct / total if total else 0.0,
        "changed_prediction_rate": changed / total if total else 0.0,
        "counterfactual_state_token_accuracy": state_correct / state_total if state_total else 0.0,
        "examples": total,
        "intervention": "replace each operation span with the next real operation span",
    }


def train_p2_cached(
    train_items: Sequence[dict[str, Any]],
    validation_items: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    config: P2Config | None = None,
    source_width: int | None = None,
    operation_width: int | None = None,
    device: str = "cuda",
    seed: int = 20260713,
    steps: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    state_weight: float = 1.0,
    validation_interval: int = 100,
    early_stop_patience: int = 8,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    source_width = source_width or int(train_items[0]["last_hidden"].shape[-1])
    operation_width = operation_width or source_width
    model = LearnedMultiSlotWorkspace(
        operation_width,
        config=config or P2Config(),
        source_width=source_width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_step = 0
    stale = 0
    started = time.perf_counter()

    def checkpoint_payload(step: int) -> dict[str, Any]:
        return {
            "schema_version": "yggdrasil.v2-a1.5.p2.checkpoint.v1",
            "step": step,
            "config": model.config.__dict__,
            "source_width": source_width,
            "operation_width": operation_width,
            "seed": seed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

    for step in range(1, steps + 1):
        indexes = [rng.randrange(len(train_items)) for _ in range(batch_size)]
        batch, _ = _batch_items(train_items, indexes, device)
        model.train()
        output = model(
            batch["operation_hidden"],
            batch["operation_mask"],
            source_hidden=batch["source_hidden"],
            source_attention_mask=batch["source_attention_mask"],
            return_trajectory=True,
        )
        loss = _loss(output, batch["labels"], state_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach())
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": gradient_norm,
                "residual_scale": float(model.transition.residual_scale.detach()),
            }
        )
        if step != steps and step % max(1, validation_interval) != 0:
            continue
        validation = evaluate_p2(model, validation_items, device, batch_size)
        evaluation = {"step": step, "validation": validation}
        evaluations.append(evaluation)
        (output_dir / "progress.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
        print(
            f"[p2-train] step={step} val_answer={validation['final_answer_accuracy']:.4f} "
            f"val_state={validation['state_full_exact']:.4f}",
            flush=True,
        )
        score = validation["final_answer_accuracy"] + validation["state_full_exact"]
        if score > best_score:
            best_score = score
            best_step = step
            stale = 0
            torch.save(checkpoint_payload(step), output_dir / "best.pt")
        else:
            stale += 1
        torch.save(checkpoint_payload(step), output_dir / "latest.pt")
        if stale >= early_stop_patience:
            break

    if not (output_dir / "best.pt").exists():
        torch.save(checkpoint_payload(len(history)), output_dir / "best.pt")
        best_step = len(history)
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.save(model.state_dict(), output_dir / "final_state.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "evaluations.json").write_text(json.dumps(evaluations, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": "yggdrasil.v2-a1.5.p2.results.v1",
        "evidence_level": "surrogate-probe",
        "stage": "P2 learned K=8 multi-slot workspace",
        "training": {
            "steps_requested": steps,
            "steps_completed": len(history),
            "batch_size": batch_size,
            "seconds": time.perf_counter() - started,
            "best_step": best_step,
            "best_score": best_score,
            "best_checkpoint": str(output_dir / "best.pt"),
        },
        "parameters": {
            "source_width": source_width,
            "operation_width": operation_width,
            **model.parameter_report(),
        },
        "integrity": model.integrity_report(),
        "fit": evaluate_p2(model, train_items, device, batch_size),
        "validation": evaluate_p2(model, validation_items, device, batch_size),
        "boundaries": [
            "P2 keeps no explicit amber/cobalt/jade slot binding.",
            "State probe is decoded from pooled learned slots and is not a named-register architectural commitment.",
            "Final metrics are evaluated after reloading best.pt.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_p2_interventions(
    model: LearnedMultiSlotWorkspace,
    items: Sequence[dict[str, Any]],
    device: str,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    normal = evaluate_p2(model, items, device, batch_size)
    max_steps = max(len(item["record"]["operations"]) for item in items)
    half = max(1, max_steps // 2)
    return {
        "normal": normal,
        "disable_transition_delta": evaluate_p2(model, items, device, batch_size, disable_transition_delta=True),
        "shuffle_transition_delta": evaluate_p2(model, items, device, batch_size, shuffle_transition_delta=True),
        "truncate_half": evaluate_p2(model, items, device, batch_size, truncate_steps=half),
        "t0": evaluate_p2(model, items, device, batch_size, truncate_steps=0),
        "t1": evaluate_p2(model, items, device, batch_size, truncate_steps=1),
        "same_answer_batch_shuffle": evaluate_same_answer_shuffle(model, items, device, batch_size),
        "single_operation_intervention": evaluate_single_operation_intervention(model, items, device),
        "slot_permutation": _permutation_probe_from_items(model, items, device, batch_size),
    }


@torch.no_grad()
def _permutation_probe_from_items(
    model: LearnedMultiSlotWorkspace,
    items: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
) -> dict[str, float]:
    batch, _ = _batch_items(items, range(min(batch_size, len(items))), device)
    return model.permutation_probe(
        batch["operation_hidden"],
        batch["operation_mask"],
        source_hidden=batch["source_hidden"],
        source_attention_mask=batch["source_attention_mask"],
    )


def load_p2_checkpoint(checkpoint_path: Path, device: str) -> LearnedMultiSlotWorkspace:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = P2Config(**checkpoint["config"])
    model = LearnedMultiSlotWorkspace(
        checkpoint["operation_width"],
        config=config,
        source_width=checkpoint["source_width"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_p2_items(cache_dir: Path, data_dir: Path, split: str) -> list[dict[str, Any]]:
    from .a1_5_data import load_a15_records

    return _cache_records(cache_dir, split, load_a15_records(data_dir, split))
