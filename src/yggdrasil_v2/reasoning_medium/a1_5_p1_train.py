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
from .a1_5_train import encode_records


def load_p1_checkpoint(checkpoint_path: Path, device: str = "cpu") -> P1HiddenLatentReasoner:
    """Load a P1 checkpoint using the architecture contract stored in it."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = P1HiddenLatentReasoner(
        int(config["source_width"]),
        config=__import__("yggdrasil_v2.reasoning_medium.a1_5_model", fromlist=["P0Config"]).P0Config(
            latent_width=int(config["latent_width"]),
            attention_heads=int(config["attention_heads"]),
            ffn_width=int(config["ffn_width"]),
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _cache_records(cache_dir: Path, split: str, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for shard_path in manifest["splits"][split]["shards"]:
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        count = len(payload["example_ids"])
        shard_records = list(records[len(items) : len(items) + count])
        items.extend(
            {
                "record": record,
                "last_hidden": payload["last_hidden"][index],
                "attention_mask": payload["attention_mask"][index],
                "operation_token_mask": payload["operation_token_mask"][index],
                "operation_mask": payload["operation_mask"][index],
            }
            for index, record in enumerate(shard_records)
        )
    if len(items) != len(records):
        raise ValueError(f"cache/record count mismatch for {split}: {len(items)} != {len(records)}")
    return items


def _batch_items(items: Sequence[dict[str, Any]], indexes: Sequence[int], device: str) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
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
    # Cache shards are padded to their shard-local maximum operation count;
    # labels are naturally padded only to this batch's maximum. Align them so
    # warm-up/state losses never treat cache padding as a real operation.
    label_steps = labels["operation_mask"].shape[1]
    if label_steps < max_steps:
        pad = max_steps - label_steps
        labels["operation_family"] = torch.nn.functional.pad(labels["operation_family"], (0, pad), value=0)
        labels["operation_source"] = torch.nn.functional.pad(labels["operation_source"], (0, pad), value=3)
        labels["operation_target"] = torch.nn.functional.pad(labels["operation_target"], (0, pad), value=3)
        labels["operation_mask"] = torch.nn.functional.pad(labels["operation_mask"], (0, pad), value=False)
        labels["state_targets"] = torch.nn.functional.pad(labels["state_targets"], (0, 0, 0, pad), value=0)
    elif label_steps > max_steps:
        raise ValueError("cache operation mask is shorter than the record operations")
    return {
        "source_hidden": hidden,
        "source_attention_mask": source_mask,
        "operation_token_mask": operation_spans,
        "operation_mask": operation_mask,
        "labels": labels,
    }, records


def _loss(
    output: dict[str, Any],
    labels: dict[str, torch.Tensor],
    model: P1HiddenLatentReasoner,
    warmup_weight: float,
    state_weight: float,
    *,
    include_task: bool = True,
) -> torch.Tensor:
    answer_loss = nn.functional.cross_entropy(output["logits"], labels["answer_targets"])
    start_targets = labels["start_values"]
    start_loss = nn.functional.cross_entropy(output["start_state_logits"].reshape(-1, len(ANSWER_LABELS)), start_targets.reshape(-1))
    query_loss = nn.functional.cross_entropy(output["query_register_logits"], labels["query_register"])
    op_mask = labels["operation_mask"]
    family_loss = nn.functional.cross_entropy(
        output["operation_family_logits"].reshape(-1, 3), labels["operation_family"].reshape(-1), reduction="none"
    ).view_as(op_mask)
    source_loss = nn.functional.cross_entropy(
        output["operation_source_logits"].reshape(-1, len(REGISTER_NAMES)),
        labels["operation_source"].clamp_max(len(REGISTER_NAMES) - 1).reshape(-1),
        reduction="none",
    ).view_as(op_mask)
    target_loss = nn.functional.cross_entropy(
        output["operation_target_logits"].reshape(-1, len(REGISTER_NAMES)),
        labels["operation_target"].clamp_max(len(REGISTER_NAMES) - 1).reshape(-1),
        reduction="none",
    ).view_as(op_mask)
    operation_loss = ((family_loss + source_loss + target_loss) * op_mask).sum() / op_mask.sum().clamp_min(1)
    state_logits = output.get("state_logits")
    if state_logits is not None and state_logits.numel():
        state_targets = labels["state_targets"]
        token_loss = nn.functional.cross_entropy(
            state_logits.reshape(-1, len(ANSWER_LABELS)), state_targets.reshape(-1), reduction="none"
        ).view(state_logits.shape[0], state_logits.shape[1], len(REGISTER_NAMES))
        state_loss = (token_loss * op_mask.unsqueeze(-1)).sum() / (op_mask.sum() * len(REGISTER_NAMES)).clamp_min(1)
    else:
        state_loss = torch.zeros((), device=answer_loss.device)
    warmup_loss = warmup_weight * (start_loss + query_loss + operation_loss)
    if not include_task:
        return warmup_loss
    return answer_loss + warmup_loss + state_weight * state_loss


@torch.no_grad()
def evaluate_p1(model: P1HiddenLatentReasoner, items: Sequence[dict[str, Any]], device: str, batch_size: int = 16) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(items), batch_size):
        batch, records = _batch_items(items, range(start, min(start + batch_size, len(items))), device)
        output = model(
            batch["source_hidden"],
            batch["source_attention_mask"],
            batch["operation_token_mask"],
            batch["operation_mask"],
            return_trajectory=True,
        )
        correct += int((output["logits"].argmax(-1) == batch["labels"]["answer_targets"]).sum())
        total += len(records)
    return {"final_answer_accuracy": correct / total if total else 0.0, "examples": total}


@torch.no_grad()
def evaluate_p1_diagnostics(
    model: P1HiddenLatentReasoner, items: Sequence[dict[str, Any]], device: str, batch_size: int = 16
) -> dict[str, float]:
    model.eval()
    counters = {name: [0, 0] for name in ("answer", "start_state", "query_register", "operation_family", "operation_source", "operation_target", "state")}
    for start in range(0, len(items), batch_size):
        batch, _ = _batch_items(items, range(start, min(start + batch_size, len(items))), device)
        output = model(
            batch["source_hidden"],
            batch["source_attention_mask"],
            batch["operation_token_mask"],
            batch["operation_mask"],
            return_trajectory=True,
        )
        labels = batch["labels"]
        counters["answer"][0] += int((output["logits"].argmax(-1) == labels["answer_targets"]).sum())
        counters["answer"][1] += labels["answer_targets"].numel()
        counters["start_state"][0] += int((output["start_state_logits"].argmax(-1) == labels["start_values"]).sum())
        counters["start_state"][1] += labels["start_values"].numel()
        counters["query_register"][0] += int((output["query_register_logits"].argmax(-1) == labels["query_register"]).sum())
        counters["query_register"][1] += labels["query_register"].numel()
        mask = labels["operation_mask"]
        for name, key, classes in (
            ("operation_family", "operation_family_logits", 3),
            ("operation_source", "operation_source_logits", len(REGISTER_NAMES)),
            ("operation_target", "operation_target_logits", len(REGISTER_NAMES)),
        ):
            pred = output[key].argmax(-1)
            target = labels[name].clamp_max(classes - 1)
            counters[name][0] += int(((pred == target) & mask).sum())
            counters[name][1] += int(mask.sum())
        state_logits = output["state_logits"]
        state_pred = state_logits.argmax(-1)
        state_target = labels["state_targets"]
        counters["state"][0] += int(((state_pred == state_target) & mask.unsqueeze(-1)).sum())
        counters["state"][1] += int(mask.sum()) * len(REGISTER_NAMES)
    return {f"{name}_accuracy": hits / total if total else 0.0 for name, (hits, total) in counters.items()}


def train_p1_cached(
    train_items: Sequence[dict[str, Any]],
    validation_items: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    latent_width: int = 256,
    attention_heads: int = 8,
    ffn_width: int = 512,
    device: str = "cuda",
    seed: int = 20260713,
    steps: int = 500,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    warmup_weight: float = 1.0,
    state_weight: float = 1.0,
    warmup_steps: int = 500,
    validation_interval: int = 100,
    early_stop_patience: int = 8,
    warmup_gate_threshold: float = 0.95,
    stop_on_warmup_failure: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    source_width = int(train_items[0]["last_hidden"].shape[-1])
    model = P1HiddenLatentReasoner(
        source_width,
        config=__import__("yggdrasil_v2.reasoning_medium.a1_5_model", fromlist=["P0Config"]).P0Config(
            latent_width=latent_width, attention_heads=attention_heads, ffn_width=ffn_width
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = random.Random(seed)
    history: list[dict[str, float | str]] = []
    evaluations: list[dict[str, Any]] = []
    started = time.perf_counter()
    best_score = float("-inf")
    best_step = 0
    best_checkpoint_path = output_dir / "best.pt"
    stale_evaluations = 0
    early_stop_reason = "steps_exhausted"
    warmup_gate: dict[str, Any] | None = None

    def checkpoint_payload(step: int) -> dict[str, Any]:
        return {
            "schema_version": "yggdrasil.v2-a1.5.p1.checkpoint.v2",
            "step": step,
            "config": {
                "source_width": source_width,
                "latent_width": latent_width,
                "attention_heads": attention_heads,
                "ffn_width": ffn_width,
            },
            "seed": seed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

    for step in range(1, steps + 1):
        indexes = [rng.randrange(len(train_items)) for _ in range(batch_size)]
        batch, _ = _batch_items(train_items, indexes, device)
        phase = "warmup" if step <= warmup_steps else "joint"
        output = model(
            batch["source_hidden"],
            batch["source_attention_mask"],
            batch["operation_token_mask"],
            batch["operation_mask"],
            return_trajectory=True,
        )
        loss = _loss(
            output,
            batch["labels"],
            model,
            warmup_weight,
            state_weight,
            include_task=phase == "joint",
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach()
        )
        optimizer.step()
        history.append(
            {
                "step": step,
                "phase": phase,
                "loss": float(loss.detach()),
                "gradient_norm": gradient_norm,
                "residual_scale": float(model.transition.residual_scale.detach()),
            }
        )

        should_evaluate = (
            step == steps
            or step == warmup_steps
            or step % max(1, validation_interval) == 0
        )
        if not should_evaluate:
            continue

        train_diagnostics = evaluate_p1_diagnostics(model, train_items, device, batch_size)
        validation_diagnostics = evaluate_p1_diagnostics(model, validation_items, device, batch_size)
        evaluation = {
            "step": step,
            "phase": phase,
            "train": train_diagnostics,
            "validation": validation_diagnostics,
        }
        evaluations.append(evaluation)
        (output_dir / "progress.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[p1-train] step={step} phase={phase} "
            f"val_answer={validation_diagnostics['answer_accuracy']:.4f} "
            f"val_state={validation_diagnostics['state_accuracy']:.4f}",
            flush=True,
        )

        if step == warmup_steps:
            warmup_names = (
                "start_state_accuracy",
                "query_register_accuracy",
                "operation_family_accuracy",
                "operation_source_accuracy",
                "operation_target_accuracy",
            )
            warmup_gate = {
                "threshold": warmup_gate_threshold,
                "validation": validation_diagnostics,
                "required_metrics": list(warmup_names),
                "passed": all(
                    validation_diagnostics[name] >= warmup_gate_threshold
                    for name in warmup_names
                ),
            }
            evaluation["warmup_gate"] = warmup_gate

        current_score = float(
            validation_diagnostics["answer_accuracy"]
            + validation_diagnostics["state_accuracy"]
        )
        if phase == "joint" and current_score > best_score:
            best_score = current_score
            best_step = step
            stale_evaluations = 0
            torch.save(checkpoint_payload(step), best_checkpoint_path)
        elif phase == "joint":
            stale_evaluations += 1

        torch.save(checkpoint_payload(step), output_dir / "latest.pt")

        if (
            step == warmup_steps
            and warmup_gate is not None
            and not warmup_gate["passed"]
            and stop_on_warmup_failure
        ):
            early_stop_reason = "warmup_gate_failed"
            break
        if phase == "joint" and stale_evaluations >= early_stop_patience:
            early_stop_reason = "validation_early_stop"
            break

    actual_steps = len(history)
    if not best_checkpoint_path.exists():
        # A failed warm-up still gets a diagnostic checkpoint, but is never
        # mislabeled as a successful joint-training best model.
        torch.save(checkpoint_payload(actual_steps), best_checkpoint_path)
        best_step = actual_steps
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.save(model.state_dict(), output_dir / "final_state.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "evaluations.json").write_text(json.dumps(evaluations, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": "yggdrasil.v2-a1.5.p1.results.v2",
        "evidence_level": "surrogate-probe",
        "stage": "P1 frozen-Qwen-hidden to latent interface",
        "training": {
            "steps_requested": steps,
            "steps_completed": actual_steps,
            "batch_size": batch_size,
            "warmup_steps_requested": warmup_steps,
            "validation_interval": validation_interval,
            "early_stop_patience": early_stop_patience,
            "seconds": time.perf_counter() - started,
            "early_stop_reason": early_stop_reason,
            "best_step": best_step,
            "best_score": best_score if best_score != float("-inf") else None,
            "best_checkpoint": str(best_checkpoint_path),
        },
        "parameters": model.parameter_report(),
        "integrity": model.integrity_report(),
        "fit": evaluate_p1(model, train_items, device, batch_size),
        "validation": evaluate_p1(model, validation_items, device, batch_size),
        "fit_diagnostics": evaluate_p1_diagnostics(model, train_items, device, batch_size),
        "validation_diagnostics": evaluate_p1_diagnostics(model, validation_items, device, batch_size),
        "warmup": {
            "start_state_query_operation_reconstruction": True,
            "teacher_trace_forward": False,
            "gate": warmup_gate,
        },
        "boundaries": [
            "This module only becomes A1.5 evidence after P0 Gate is explicitly accepted.",
            "A cache smoke or fit score does not establish information fidelity.",
            "Final metrics are evaluated after reloading best.pt.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
