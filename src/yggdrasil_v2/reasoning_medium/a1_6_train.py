from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_6_core import A16CoreConfig, RelationAddressedCore
from .a1_6_data import FAMILY_TO_INDEX, REGISTER_NAMES, VALUE_LABELS, execute_operations


def family_index(name: str) -> int:
    return FAMILY_TO_INDEX[name]


def encode_records(records: Sequence[dict[str, Any]], device: str | torch.device) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot encode an empty record batch")
    batch_size = len(records)
    max_steps = max(int(record["program_length"]) for record in records)
    starts = torch.tensor([[VALUE_LABELS.index(record["start_state"][name]) for name in REGISTER_NAMES] for record in records], dtype=torch.long, device=device)
    query = torch.tensor([REGISTER_NAMES.index(record["query_register"]) for record in records], dtype=torch.long, device=device)
    family = torch.zeros((batch_size, max_steps), dtype=torch.long, device=device)
    source = torch.zeros_like(family)
    target = torch.zeros_like(family)
    mask = torch.zeros((batch_size, max_steps), dtype=torch.bool, device=device)
    state_targets = torch.zeros((batch_size, max_steps, len(REGISTER_NAMES)), dtype=torch.long, device=device)
    for row, record in enumerate(records):
        length = int(record["program_length"])
        mask[row, :length] = torch.tensor(record.get("operation_mask", [True] * length), dtype=torch.bool, device=device)
        for step, operation in enumerate(record["operations"]):
            family[row, step] = family_index(operation["family"])
            source[row, step] = REGISTER_NAMES.index(operation["source"])
            target[row, step] = REGISTER_NAMES.index(operation["target"])
        for step, state in enumerate(record["state_trajectory"]):
            state_targets[row, step] = torch.tensor([VALUE_LABELS.index(state[name]) for name in REGISTER_NAMES], device=device)
    return {"start_values": starts, "query_register": query, "operation_family": family, "operation_source": source, "operation_target": target, "operation_mask": mask, "state_targets": state_targets, "answer_targets": torch.tensor([int(record["answer_index"]) for record in records], dtype=torch.long, device=device)}


def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch[key] for key in ("start_values", "query_register", "operation_family", "operation_source", "operation_target", "operation_mask")}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.to(dtype=values.dtype)).sum() / mask.to(dtype=values.dtype).sum().clamp_min(1.0)


def compute_loss(model: RelationAddressedCore, output: dict[str, Any], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["operation_mask"]
    state_ce = nn.functional.cross_entropy(output["state_logits"].reshape(-1, len(VALUE_LABELS)), batch["state_targets"].reshape(-1), reduction="none").reshape(mask.shape[0], mask.shape[1], len(REGISTER_NAMES))
    state_loss = _masked_mean(state_ce.mean(dim=-1), mask)
    source_ce = nn.functional.cross_entropy(output["source_pointer_logits"].reshape(-1, len(REGISTER_NAMES)), batch["operation_source"].reshape(-1), reduction="none").reshape(mask.shape)
    target_ce = nn.functional.cross_entropy(output["target_pointer_logits"].reshape(-1, len(REGISTER_NAMES)), batch["operation_target"].reshape(-1), reduction="none").reshape(mask.shape)
    source_loss = _masked_mean(source_ce, mask)
    target_loss = _masked_mean(target_ce, mask)
    answer_loss = nn.functional.cross_entropy(output["answer_logits"], batch["answer_targets"])
    total = state_loss + 0.25 * source_loss + 0.25 * target_loss + 0.5 * answer_loss
    return total, {"loss": float(total.detach()), "state_loss": float(state_loss.detach()), "source_pointer_loss": float(source_loss.detach()), "target_pointer_loss": float(target_loss.detach()), "answer_loss": float(answer_loss.detach())}


@torch.no_grad()
def evaluate_core(model: RelationAddressedCore, records: Sequence[dict[str, Any]], device: str | torch.device, *, batch_size: int = 128) -> dict[str, Any]:
    model.eval()
    answer_correct = state_tokens = state_total = state_full = source_correct = target_correct = pointer_total = 0
    source_entropy: list[float] = []
    target_entropy: list[float] = []
    source_gates: list[float] = []
    target_gates: list[float] = []
    losses: list[float] = []
    for start in range(0, len(records), batch_size):
        batch = encode_records(records[start : start + batch_size], device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        loss, _ = compute_loss(model, output, batch)
        losses.append(float(loss))
        answer_correct += int((output["answer_logits"].argmax(-1) == batch["answer_targets"]).sum())
        valid = batch["operation_mask"]
        state_pred = output["state_logits"].argmax(-1)
        state_equal = state_pred == batch["state_targets"]
        state_tokens += int((state_equal & valid.unsqueeze(-1)).sum())
        state_total += int(valid.sum()) * len(REGISTER_NAMES)
        state_full += int((state_equal | ~valid.unsqueeze(-1)).all(dim=(-1, -2)).sum())
        source_pred = output["source_pointer_logits"].argmax(-1)
        target_pred = output["target_pointer_logits"].argmax(-1)
        source_correct += int(((source_pred == batch["operation_source"]) & valid).sum())
        target_correct += int(((target_pred == batch["operation_target"]) & valid).sum())
        pointer_total += int(valid.sum())
        for weights, bucket in ((output["source_pointer_weights"], source_entropy), (output["target_pointer_weights"], target_entropy)):
            if weights.numel():
                entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
                bucket.extend(entropy[valid].detach().cpu().tolist())
        source_gates.extend(output["source_write_gates"].squeeze(-1)[valid].detach().cpu().tolist())
        target_gates.extend(output["target_write_gates"].squeeze(-1)[valid].detach().cpu().tolist())
    return {
        "examples": len(records),
        "loss": sum(losses) / max(1, len(losses)),
        "answer_accuracy": answer_correct / max(1, len(records)),
        "final_answer_accuracy": answer_correct / max(1, len(records)),
        "state_token_accuracy": state_tokens / max(1, state_total),
        "state_full_exact": state_full / max(1, len(records)),
        "source_pointer_accuracy": source_correct / max(1, pointer_total),
        "target_pointer_accuracy": target_correct / max(1, pointer_total),
        "pointer_entropy": {"source": sum(source_entropy) / max(1, len(source_entropy)), "target": sum(target_entropy) / max(1, len(target_entropy))},
        "operator_write_gates": {"source": sum(source_gates) / max(1, len(source_gates)), "target": sum(target_gates) / max(1, len(target_gates))},
    }


def _gradient_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float(parameter.grad.detach().pow(2).sum()) for parameter in model.parameters() if parameter.grad is not None))


def _checkpoint(model: RelationAddressedCore, optimizer: torch.optim.Optimizer, step: int, config: A16CoreConfig, seed: int) -> dict[str, Any]:
    return {"schema_version": "yggdrasil.v2-a1.6.c0.checkpoint.v1", "step": step, "config": asdict(config), "seed": seed, "fresh_initialization": True, "model": model.state_dict(), "optimizer": optimizer.state_dict()}


def train_c0(train_records: Sequence[dict[str, Any]], validation_records: Sequence[dict[str, Any]], output_dir: Path, *, config: A16CoreConfig | None = None, seed: int = 20260715, device: str = "cuda", steps: int = 6000, batch_size: int = 128, learning_rate: float = 3e-4, validation_interval: int = 100, early_stop_exact: float | None = None, early_stop_patience: int | None = None, overfit_mode: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA for A1.6 C0 but CUDA is unavailable")
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    config = config or A16CoreConfig()
    model = RelationAddressedCore(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_state: dict[str, Any] | None = None
    patience = 0
    exact_streak = 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        batch_records = [train_records[rng.randrange(len(train_records))] for _ in range(batch_size)]
        batch = encode_records(batch_records, device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        loss, metrics = compute_loss(model, output, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = _gradient_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics.update({"step": step, "gradient_norm": gradient_norm})
        if step % validation_interval == 0 or step == steps:
            train_metrics = evaluate_core(model, train_records, device, batch_size=batch_size)
            validation_metrics = evaluate_core(model, validation_records, device, batch_size=batch_size)
            metrics["train"] = train_metrics
            metrics["validation"] = validation_metrics
            score = validation_metrics["state_full_exact"] + 0.25 * validation_metrics["source_pointer_accuracy"] + 0.25 * validation_metrics["target_pointer_accuracy"]
            checkpoint = _checkpoint(model, optimizer, step, config, seed)
            torch.save(checkpoint, output_dir / "latest.pt")
            if score > best_score:
                best_score = score
                best_state = validation_metrics
                torch.save(checkpoint, output_dir / "best.pt")
                patience = 0
            else:
                patience += 1
            exact_metrics = train_metrics if overfit_mode else validation_metrics
            if early_stop_exact is not None and all(exact_metrics[key] >= early_stop_exact for key in ("answer_accuracy", "state_token_accuracy", "state_full_exact", "source_pointer_accuracy", "target_pointer_accuracy")):
                exact_streak += 1
            else:
                exact_streak = 0
            if early_stop_exact is not None and exact_streak >= 2:
                history.append(metrics)
                break
            if early_stop_patience is not None and patience >= early_stop_patience:
                history.append(metrics)
                break
        history.append(metrics)
        (output_dir / "progress.json").write_text(json.dumps({"step": step, "steps_requested": steps, "best_score": best_score, "elapsed_seconds": time.perf_counter() - started}, indent=2) + "\n", encoding="utf-8")
    if not (output_dir / "best.pt").exists():
        raise RuntimeError("C0 training completed without a best checkpoint")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"schema_version": "yggdrasil.v2-a1.6.c0.results.v1", "stage": "C0 relation-addressed latent core", "fresh_initialization": True, "config": asdict(config), "training": {"seed": seed, "steps_requested": steps, "steps_completed": history[-1]["step"], "batch_size": batch_size, "learning_rate": learning_rate, "fresh_initialization": True, "overfit_mode": overfit_mode, "seconds": time.perf_counter() - started}, "best_score": best_score, "fit": evaluate_core(model, train_records, device, batch_size=batch_size), "validation": evaluate_core(model, validation_records, device, batch_size=batch_size), "integrity": model.integrity_report(), "parameter_report": model.parameter_report(), "best_validation_at_save": best_state, "artifacts": {"latest": str(output_dir / "latest.pt"), "best": str(output_dir / "best.pt"), "history": str(output_dir / "history.json"), "progress": str(output_dir / "progress.json")}}
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def load_c0_checkpoint(path: Path, device: str | torch.device = "cpu") -> RelationAddressedCore:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RelationAddressedCore(A16CoreConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def _score_counterfactual(model: RelationAddressedCore, records: Sequence[dict[str, Any]], device: str | torch.device, *, batch_size: int = 128) -> dict[str, float]:
    state_full = answer_correct = total = 0
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch = encode_records(batch_records, device)
        out = model(**_model_inputs(batch), return_trajectory=True)
        pred = out["state_logits"].argmax(-1)
        valid = batch["operation_mask"]
        equal = pred == batch["state_targets"]
        state_full += int((equal | ~valid.unsqueeze(-1)).all(dim=(-1, -2)).sum())
        answer_correct += int((out["answer_logits"].argmax(-1) == batch["answer_targets"]).sum())
        total += len(batch_records)
    return {"state_full_exact": state_full / max(1, total), "final_answer_accuracy": answer_correct / max(1, total), "examples": total}


def _clone(record: dict[str, Any], operations: Sequence[dict[str, Any]], mask: Sequence[bool] | None = None) -> dict[str, Any]:
    clone = dict(record)
    clone["operations"] = [dict(operation) for operation in operations]
    clone["operation_mask"] = list(mask) if mask is not None else [True] * len(operations)
    trajectory = execute_operations(clone["start_state"], clone["operations"], clone["operation_mask"])
    clone["state_trajectory"] = trajectory[1:]
    clone["program_length"] = len(operations)
    clone["answer"] = trajectory[-1][clone["query_register"]]
    clone["answer_index"] = VALUE_LABELS.index(clone["answer"])
    return clone


def run_c0_interventions(model: RelationAddressedCore, records: Sequence[dict[str, Any]], device: str | torch.device, *, batch_size: int = 128) -> dict[str, Any]:
    # Prefix fidelity compares prefix-k predictions to prefix-k oracle states.
    prefix_rows = []
    for record in records:
        for k in range(record["program_length"] + 1):
            prefix = _clone(record, record["operations"][:k])
            expected = [record["start_state"]] + execute_operations(record["start_state"], record["operations"][:k])[1:]
            batch = encode_records([prefix], device)
            out = model(**_model_inputs(batch), return_trajectory=True)
            decoded = out["state_logits"].argmax(-1)[0]
            predicted = [out["initial_slots"].new_tensor([[VALUE_LABELS.index(prefix["start_state"][name]) for name in REGISTER_NAMES]])] if k == 0 else []
            if k == 0:
                # T=0 is checked through the same shared state head on initial slots.
                initial_decoded = model.shared_state_head(out["initial_slots"])[0].argmax(-1).tolist()
                exact = all(VALUE_LABELS[value] == expected[0][name] for value, name in zip(initial_decoded, REGISTER_NAMES))
            else:
                exact = all(VALUE_LABELS[value] == expected[-1][name] for value, name in zip(decoded[k - 1].tolist(), REGISTER_NAMES))
            prefix_rows.append({"k": k, "exact": exact})
    prefix_fidelity = sum(row["exact"] for row in prefix_rows) / max(1, len(prefix_rows))

    replacement: list[dict[str, Any]] = []
    deletion: list[dict[str, Any]] = []
    shuffled: list[dict[str, Any]] = []
    rng = random.Random(20260715)
    for record in records:
        for index, operation in enumerate(record["operations"]):
            alternate = dict(operation)
            alternate["family"] = "swap" if operation["family"] == "copy" else "copy"
            if alternate["family"] == "copy":
                alternate["source"], alternate["target"] = "amber", "cobalt"
            else:
                alternate["source"], alternate["target"] = "amber", "jade"
            replacement.append(_clone(record, [alternate if i == index else op for i, op in enumerate(record["operations"])]))
            deletion.append(_clone(record, record["operations"], [i != index for i in range(record["program_length"])]))
        order = list(range(record["program_length"]))
        rng.shuffle(order)
        shuffled.append(_clone(record, [record["operations"][i] for i in order]))
    return {"prefix_fidelity": prefix_fidelity, "prefix_rows": len(prefix_rows), "replacement_counterfactual": _score_counterfactual(model, replacement, device, batch_size=batch_size), "deletion_counterfactual": _score_counterfactual(model, deletion, device, batch_size=batch_size), "shuffled_program": _score_counterfactual(model, shuffled, device, batch_size=batch_size), "method": "oracle state trajectory exact, not changed-prediction rate"}
