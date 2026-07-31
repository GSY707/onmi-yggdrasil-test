from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_10_model import A110AnonymousRecurrentReasoner, A110ModelConfig
from .a1_13_train import _write_json
from .a1_19h_h2_data import FORMAL_SPLITS
from .a1_20b_cache import A120BCachedSplit, collate_a120b_items
from .a1_20b_train import (
    A120BLabels,
    encode_a120b_labels,
    sample_a120b_indices,
)


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.21p.k1.checkpoint.v1"
RESULT_SCHEMA = "yggdrasil.v2-a1.21p.k1.training.v1"
EVALUATION_SCHEMA = "yggdrasil.v2-a1.21p.k1.evaluation.v1"
RECURRENT_STEPS = 32


@dataclass(frozen=True)
class A121PK1TrainSpec:
    model_seed: int
    steps: int = 4000
    batch_size: int = 16
    learning_rate: float = 3e-4
    validation_interval: int = 200
    validation_per_cell: int = 2
    early_stop_patience: int = 8


def a121p_k1_config(source_width: int) -> A110ModelConfig:
    return A110ModelConfig(
        source_width=source_width,
        latent_width=256,
        workspace_slots=1,
        attention_heads=8,
        ffn_width=1024,
        recurrent_layers=2,
        value_classes=10,
        state_fields=5,
    )


def compute_a121p_k1_loss(
    output: dict[str, Any],
    labels: A120BLabels,
) -> tuple[torch.Tensor, dict[str, float]]:
    state_ce = nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, 10),
        labels.state_targets.reshape(-1),
        ignore_index=-100,
    )
    answer_ce = nn.functional.cross_entropy(
        output["answer_logits"], labels.answer_targets
    )
    total = state_ce + answer_ce
    return total, {
        "loss": float(total.detach()),
        "state_ce": float(state_ce.detach()),
        "answer_ce": float(answer_ce.detach()),
    }


@torch.inference_mode()
def evaluate_a121p_k1_items(
    model: A110AnonymousRecurrentReasoner,
    dataset: A120BCachedSplit,
    indices: Sequence[int],
    device: str,
    *,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    trajectory = final_state = answer = 0
    state_correct = state_total = 0
    by_cell_counts: dict[str, dict[str, int]] = {}
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        inputs, records = collate_a120b_items(
            dataset.items(batch_indices), device
        )
        labels = encode_a120b_labels(records, device)
        output = model(**inputs, recurrent_steps=RECURRENT_STEPS)
        prediction = output["state_logits"].argmax(dim=-1)
        active = labels.state_targets >= 0
        equal = (prediction == labels.state_targets) | ~active
        trajectory_rows = equal.all(dim=(-1, -2))
        final_rows = equal[:, -1].all(dim=-1)
        answer_rows = (
            output["answer_logits"].argmax(dim=-1)
            == labels.answer_targets
        )
        trajectory += int(trajectory_rows.sum())
        final_state += int(final_rows.sum())
        answer += int(answer_rows.sum())
        state_correct += int(
            ((prediction == labels.state_targets) & active).sum()
        )
        state_total += int(active.sum())
        for row, record in enumerate(records):
            cell = (
                f"N{record['entity_count']}-T{record['program_length']}"
            )
            counts = by_cell_counts.setdefault(
                cell,
                {
                    "examples": 0,
                    "trajectory": 0,
                    "final_state": 0,
                    "answer": 0,
                },
            )
            counts["examples"] += 1
            counts["trajectory"] += int(trajectory_rows[row])
            counts["final_state"] += int(final_rows[row])
            counts["answer"] += int(answer_rows[row])
    by_cell = {
        cell: {
            "examples": row["examples"],
            "trajectory_full_exact": row["trajectory"] / row["examples"],
            "final_state_full_exact": row["final_state"] / row["examples"],
            "final_answer_accuracy": row["answer"] / row["examples"],
        }
        for cell, row in sorted(by_cell_counts.items())
    }
    examples = len(indices)
    return {
        "aggregate": {
            "examples": examples,
            "trajectory_full_exact": trajectory / max(1, examples),
            "final_state_full_exact": final_state / max(1, examples),
            "state_token_accuracy": state_correct / max(1, state_total),
            "final_answer_accuracy": answer / max(1, examples),
        },
        "by_cell": by_cell,
    }


def _validation_indices(
    dataset: A120BCachedSplit, per_cell: int
) -> list[int]:
    return [
        index
        for cell in sorted(dataset.indices_by_cell)
        for index in dataset.indices_by_cell[cell][:per_cell]
    ]


def _score(evaluation: dict[str, Any]) -> tuple[float, ...]:
    cells = list(evaluation["by_cell"].values())
    aggregate = evaluation["aggregate"]
    return (
        min(row["trajectory_full_exact"] for row in cells),
        min(row["final_state_full_exact"] for row in cells),
        min(row["final_answer_accuracy"] for row in cells),
        aggregate["state_token_accuracy"],
    )


def _checkpoint(
    model: A110AnonymousRecurrentReasoner,
    optimizer: torch.optim.Optimizer,
    spec: A121PK1TrainSpec,
    step: int,
    initial_checkpoint: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_config": asdict(model.config),
        "train_spec": asdict(spec),
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "qwen_frozen": True,
        "workspace_slots": 1,
        "full_text_hidden_only": True,
        "recurrent_steps": RECURRENT_STEPS,
        "initial_checkpoint": (
            str(initial_checkpoint) if initial_checkpoint else None
        ),
    }


def load_a121p_k1_checkpoint(
    checkpoint_path: Path, device: str = "cpu"
) -> A110AnonymousRecurrentReasoner:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("not an A1.21P K=1 checkpoint")
    model = A110AnonymousRecurrentReasoner(
        A110ModelConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.a121p_step = int(checkpoint["step"])
    model.a121p_train_spec = dict(checkpoint["train_spec"])
    model.eval()
    return model


def train_a121p_k1(
    *,
    cache_dir: Path,
    data_dir: Path,
    output_dir: Path,
    spec: A121PK1TrainSpec,
    device: str = "cuda",
    overfit: bool = False,
    initial_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.21P K=1 training requested unavailable CUDA")
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.21P K=1 run {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = A120BCachedSplit(cache_dir, data_dir, "train")
    validation_dataset = A120BCachedSplit(
        cache_dir, data_dir, "validation"
    )
    random.seed(spec.model_seed)
    torch.manual_seed(spec.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.model_seed)
    if initial_checkpoint is None:
        model = A110AnonymousRecurrentReasoner(
            a121p_k1_config(train_dataset.hidden_width)
        ).to(device)
    else:
        model = load_a121p_k1_checkpoint(initial_checkpoint, device)
        if model.config != a121p_k1_config(train_dataset.hidden_width):
            raise ValueError("A1.21P K=1 continuation config mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate
    )
    rng = random.Random(spec.model_seed)
    validation_indices = (
        list(range(len(validation_dataset)))
        if overfit
        else _validation_indices(
            validation_dataset, spec.validation_per_cell
        )
    )
    history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0, -1.0, -1.0)
    best_step: int | None = None
    patience = 0
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        indices = sample_a120b_indices(
            train_dataset, spec.batch_size, rng
        )
        inputs, records = collate_a120b_items(
            train_dataset.items(indices), device
        )
        labels = encode_a120b_labels(records, device)
        output = model(**inputs, recurrent_steps=RECURRENT_STEPS)
        loss, row = compute_a121p_k1_loss(output, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        row["gradient_norm_before_clip"] = float(
            torch.sqrt(
                sum(
                    parameter.grad.detach().float().square().sum()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
        )
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row["step"] = step
        if step % spec.validation_interval == 0 or step == spec.steps:
            validation = evaluate_a121p_k1_items(
                model,
                validation_dataset,
                validation_indices,
                device,
                batch_size=spec.batch_size,
            )
            observed_score = _score(validation)
            row["validation"] = validation
            row["validation_score"] = list(observed_score)
            checkpoint = _checkpoint(
                model, optimizer, spec, step, initial_checkpoint
            )
            torch.save(checkpoint, output_dir / "latest.pt")
            if observed_score > best_score:
                best_score = observed_score
                best_step = step
                patience = 0
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                patience += 1
            if overfit and all(value == 1.0 for value in observed_score):
                history.append(row)
                break
            if not overfit and patience >= spec.early_stop_patience:
                history.append(row)
                break
        history.append(row)
    if not (output_dir / "best.pt").exists():
        raise RuntimeError("A1.21P K=1 training produced no checkpoint")
    model = load_a121p_k1_checkpoint(
        output_dir / "best.pt", device
    )
    validation = evaluate_a121p_k1_items(
        model,
        validation_dataset,
        list(range(len(validation_dataset))),
        device,
        batch_size=spec.batch_size,
    )
    overfit_gates = {
        "trajectory_exact": (
            validation["aggregate"]["trajectory_full_exact"] == 1.0
        ),
        "final_state_exact": (
            validation["aggregate"]["final_state_full_exact"] == 1.0
        ),
        "answer_exact": (
            validation["aggregate"]["final_answer_accuracy"] == 1.0
        ),
        "state_token_exact": (
            validation["aggregate"]["state_token_accuracy"] == 1.0
        ),
        "workspace_slots_exactly_one": (
            model.config.workspace_slots == 1
        ),
        "architecture_integrity": model.integrity_report()["passed"],
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "stage": "A1.21P K=1 generic recurrent baseline",
        "selection": (
            "fit-only-overfit32" if overfit else "full-training"
        ),
        "model_config": asdict(model.config),
        "train_spec": asdict(spec),
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": int(history[-1]["step"]),
            "seconds": time.perf_counter() - started,
            "processed_examples": int(history[-1]["step"])
            * spec.batch_size,
            "processed_recurrent_loops": int(history[-1]["step"])
            * spec.batch_size
            * RECURRENT_STEPS,
            "fresh_initialization": initial_checkpoint is None,
            "initial_checkpoint": (
                str(initial_checkpoint) if initial_checkpoint else None
            ),
        },
        "best_step": best_step,
        "best_score": list(best_score),
        "validation": validation,
        "overfit_gates": overfit_gates if overfit else None,
        "overfit_passed": (
            all(overfit_gates.values()) if overfit else None
        ),
        "integrity": model.integrity_report(),
        "parameters": model.parameter_report(),
        "artifacts": {
            "best": str(output_dir / "best.pt"),
            "latest": str(output_dir / "latest.pt"),
        },
    }
    _write_json(output_dir / "history.json", history)
    _write_json(output_dir / "results.json", result)
    return result


@torch.inference_mode()
def evaluate_a121p_k1_formal(
    *,
    checkpoint_path: Path,
    cache_dir: Path,
    data_dir: Path,
    output_path: Path,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, Any]:
    model = load_a121p_k1_checkpoint(checkpoint_path, device)
    splits = {
        split: evaluate_a121p_k1_items(
            model,
            dataset := A120BCachedSplit(cache_dir, data_dir, split),
            list(range(len(dataset))),
            device,
            batch_size=batch_size,
        )
        for split in FORMAL_SPLITS
    }
    gates = {
        f"{split}_answer_at_least_{threshold:.2f}": (
            result["aggregate"]["final_answer_accuracy"] >= threshold
        )
        for split, result in splits.items()
        for threshold in (
            (0.95,)
            if split in {"short_regression", "causal_core"}
            else (0.90,)
        )
    }
    gates["all_state_token_at_least_0.995"] = all(
        result["aggregate"]["state_token_accuracy"] >= 0.995
        for result in splits.values()
    )
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "splits": splits,
        "gates": gates,
        "passed": all(gates.values()),
        "role": "K=1 matched-capacity baseline; not the target hybrid path",
    }
    _write_json(output_path, result)
    return result
