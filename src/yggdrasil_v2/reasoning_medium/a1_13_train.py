from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_7_train import _gradient_norm
from .a1_8_data import IN_RANGE_LENGTHS, SHORT_LENGTHS, load_a18_records
from .a1_9_cache import file_sha256
from .a1_10_cache import select_length_balanced_overfit32
from .a1_10_train import TRAIN_RECURRENT_STEPS, encode_a110_targets
from .a1_11_train import A111RecordSplit, FORMAL_SPLITS, HARD_OOD_LENGTHS, _symbolic_reasoner_inputs
from .a1_12_train import (
    _balanced_subset_indices,
    _metric_gate,
    _sample_balanced_batch,
    _validation_score,
    evaluate_a112_by_length,
)
from .a1_13_models import A113ReasonerConfig, A113TransitionClosureReasoner


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.13.transition-closure.checkpoint.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.13.transition-closure.results.v1"
FORMAL_SCHEMA = "yggdrasil.v2-a1.13.transition-closure.formal.v1"
A115_CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.15.closed-coupled-core.checkpoint.v1"
A115_RESULTS_SCHEMA = "yggdrasil.v2-a1.15.closed-coupled-core.results.v1"
A115_FORMAL_SCHEMA = "yggdrasil.v2-a1.15.closed-coupled-core.formal.v1"
A116_CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.16.no-redundant-answer-loss.checkpoint.v1"
A116_RESULTS_SCHEMA = "yggdrasil.v2-a1.16.no-redundant-answer-loss.results.v1"
A116_FORMAL_SCHEMA = "yggdrasil.v2-a1.16.no-redundant-answer-loss.formal.v1"
CLOSURE_WEIGHT = 1.0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compute_a113_loss(
    model: A113TransitionClosureReasoner,
    output: dict[str, Any],
    labels: dict[str, torch.Tensor],
    *,
    answer_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    state_ce = nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, model.a113_config.value_classes),
        labels["state_targets"].reshape(-1),
    )
    answer_ce = nn.functional.cross_entropy(output["answer_logits"], labels["answer_targets"])
    closure_loss = state_ce.new_zeros(())
    if model.a113_config.prototype_closure_objective:
        predicted = output["trajectory"][:, :, : model.a113_config.entity_slots]
        targets = model.closure_targets(labels["state_targets"]).detach()
        closure_loss = (
            1.0 - nn.functional.cosine_similarity(predicted, targets, dim=-1)
        ).mean()
    total = state_ce + answer_loss_weight * answer_ce + CLOSURE_WEIGHT * closure_loss
    return total, {
        "loss": float(total.detach()),
        "state_ce": float(state_ce.detach()),
        "answer_ce": float(answer_ce.detach()),
        "answer_loss_weight": answer_loss_weight,
        "closure_loss": float(closure_loss.detach()),
        "closure_weight": CLOSURE_WEIGHT if model.a113_config.prototype_closure_objective else 0.0,
    }


@dataclass(frozen=True)
class A113TrainSpec:
    model_seed: int
    data_seed: int
    structured_transition: bool
    prototype_closure_objective: bool
    query_coupled_answer: bool = False
    answer_loss_weight: float = 1.0
    steps: int = 4000
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_interval: int = 200
    early_stop_patience: int = 6
    train_recurrent_steps: int = TRAIN_RECURRENT_STEPS


def _checkpoint(
    model: A113TransitionClosureReasoner,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A113TrainSpec,
    data_manifest_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": (
            A116_CHECKPOINT_SCHEMA
            if spec.query_coupled_answer and spec.answer_loss_weight == 0.0
            else A115_CHECKPOINT_SCHEMA
            if spec.query_coupled_answer
            else CHECKPOINT_SCHEMA
        ),
        "step": step,
        "model_config": asdict(model.a113_config),
        "train_spec": asdict(spec),
        "data_manifest_sha256": data_manifest_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "arm": model.arm,
        "global_train_recurrent_steps": TRAIN_RECURRENT_STEPS,
    }


def train_a113(
    data_dir: Path,
    output_dir: Path,
    *,
    spec: A113TrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.13 checkpoints: {output_dir}")
    if spec.train_recurrent_steps != TRAIN_RECURRENT_STEPS:
        raise ValueError("A1.13 train recurrent steps are frozen at 16")
    if spec.answer_loss_weight not in {0.0, 1.0}:
        raise ValueError("only the preregistered answer loss weights 0 or 1 are allowed")
    if spec.answer_loss_weight == 0.0 and not spec.query_coupled_answer:
        raise ValueError("answer loss may be removed only when answer is state-coupled")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(data_manifest["data_seed"]) != spec.data_seed:
        raise ValueError("A1.13 spec does not match data seed")
    data_manifest_hash = file_sha256(manifest_path)
    if overfit_mode:
        selected = select_length_balanced_overfit32(load_a18_records(data_dir, "train"))
        train_dataset = validation_dataset = A111RecordSplit(selected)
        selection = "length-balanced-overfit32-fit-only"
    else:
        train_dataset = A111RecordSplit(load_a18_records(data_dir, "train"))
        validation_dataset = A111RecordSplit(load_a18_records(data_dir, "validation"))
        selection = "formal-a1.8-records"
    random.seed(spec.model_seed)
    torch.manual_seed(spec.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.model_seed)
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=spec.structured_transition,
            prototype_closure_objective=spec.prototype_closure_objective,
            query_coupled_answer=spec.query_coupled_answer,
        )
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate)
    rng = random.Random(spec.model_seed)
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    patience = ready_streak = processed_transitions = 0
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        records = _sample_balanced_batch(train_dataset, spec.batch_size, rng)
        inputs = _symbolic_reasoner_inputs(records, device)
        labels = encode_a110_targets(records, TRAIN_RECURRENT_STEPS, device)
        output = model(**inputs, recurrent_steps=TRAIN_RECURRENT_STEPS)
        loss, metrics = compute_a113_loss(
            model, output, labels, answer_loss_weight=spec.answer_loss_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics["gradient_norm"] = _gradient_norm(model)
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        processed_transitions += spec.batch_size * TRAIN_RECURRENT_STEPS
        metrics["step"] = step
        if step % spec.validation_interval == 0 or step == spec.steps:
            validation = evaluate_a112_by_length(
                model, validation_dataset, device, batch_size=spec.batch_size
            )
            score, components = _validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = components
            checkpoint = _checkpoint(model, optimizer, step, spec, data_manifest_hash)
            torch.save(checkpoint, output_dir / "latest.pt")
            if best_score is None or score > best_score:
                best_score, best_step, best_validation, patience = score, step, validation, 0
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                patience += 1
            target = 1.0 if overfit_mode else 0.995
            ready = all(value >= target for value in components.values())
            ready_streak = ready_streak + 1 if ready else 0
            history.append(metrics)
            _write_json(
                output_dir / "progress.json",
                {
                    "arm": model.arm,
                    "step": step,
                    "steps_requested": spec.steps,
                    "best_step": best_step,
                    "best_score": best_score,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            if ready_streak >= 2 or (not overfit_mode and patience >= spec.early_stop_patience):
                break
        else:
            history.append(metrics)
    _write_json(output_dir / "history.json", history)
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation = evaluate_a112_by_length(
        model, validation_dataset, device, batch_size=spec.batch_size
    )
    fit_diagnostic = evaluate_a112_by_length(
        model,
        train_dataset,
        device,
        batch_size=spec.batch_size,
        indices=_balanced_subset_indices(train_dataset),
    )
    overfit_gates = {
        "all_length_trajectory_exact": all(
            row["trajectory_full_exact"] == 1.0 for row in validation["by_length"].values()
        ),
        "all_length_final_state_exact": all(
            row["final_state_full_exact"] == 1.0 for row in validation["by_length"].values()
        ),
        "all_length_answer_exact": all(
            row["final_answer_accuracy"] == 1.0 for row in validation["by_length"].values()
        ),
        "architecture_integrity": model.integrity_report()["passed"],
    }
    result = {
        "schema_version": (
            A116_RESULTS_SCHEMA
            if spec.query_coupled_answer and spec.answer_loss_weight == 0.0
            else A115_RESULTS_SCHEMA
            if spec.query_coupled_answer
            else RESULTS_SCHEMA
        ),
        "stage": (
            "V2-A1.16 query-coupled core without redundant answer loss"
            if spec.query_coupled_answer and spec.answer_loss_weight == 0.0
            else "V2-A1.15 closed query-coupled core"
            if spec.query_coupled_answer
            else "V2-A1.13 transition x closure root-cause localization"
        ),
        "arm": model.arm,
        "selection": selection,
        "train_spec": asdict(spec),
        "model_config": asdict(model.a113_config),
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": history[-1]["step"],
            "processed_examples": history[-1]["step"] * spec.batch_size,
            "processed_recurrent_transitions": processed_transitions,
            "seconds": time.perf_counter() - started,
            "fresh_initialization": True,
            "overfit_mode": overfit_mode,
        },
        "best_checkpoint_step": best_step,
        "best_validation_at_save": best_validation,
        "validation": validation,
        "fit_diagnostic": fit_diagnostic,
        "integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(),
        "data_manifest_sha256": data_manifest_hash,
        "overfit_gates": overfit_gates if overfit_mode else None,
        "overfit_passed": all(overfit_gates.values()) if overfit_mode else None,
    }
    _write_json(output_dir / "results.json", result)
    return result


def load_a113_checkpoint(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> A113TransitionClosureReasoner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") not in {
        CHECKPOINT_SCHEMA,
        A115_CHECKPOINT_SCHEMA,
        A116_CHECKPOINT_SCHEMA,
    }:
        raise ValueError("not an A1.13/A1.15 checkpoint")
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.a113_model_seed = int(checkpoint["train_spec"]["model_seed"])
    model.a113_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a113_data_manifest_sha256 = checkpoint["data_manifest_sha256"]
    model.a113_answer_loss_weight = float(
        checkpoint["train_spec"].get("answer_loss_weight", 1.0)
    )
    model.eval()
    return model


@torch.no_grad()
def evaluate_a113_formal(
    checkpoint: Path,
    data_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    model = load_a113_checkpoint(checkpoint, device)
    results = {
        split: evaluate_a112_by_length(
            model,
            A111RecordSplit(load_a18_records(data_dir, split)),
            device,
            batch_size=batch_size,
        )
        for split in FORMAL_SPLITS
    }
    gates: dict[str, bool] = {}
    for prefix, split, lengths, threshold in (
        ("short", "short_regression", SHORT_LENGTHS, 0.95),
        ("supported_in_range", "supported_in_range", IN_RANGE_LENGTHS, 0.95),
        ("relation_in_range", "relation_in_range", IN_RANGE_LENGTHS, 0.95),
        ("supported_ood", "supported_ood", HARD_OOD_LENGTHS, 0.90),
        ("relation_ood", "relation_ood", HARD_OOD_LENGTHS, 0.90),
    ):
        for metric in ("trajectory_full_exact", "final_state_full_exact", "final_answer_accuracy"):
            gates[f"{prefix}_{metric}_gate"] = _metric_gate(
                results, split, lengths, metric, threshold
            )
    gates.update(
        {
            "causal_trajectory_at_least_0_95": results["causal_core"]["aggregate"]["trajectory_full_exact"] >= 0.95,
            "causal_final_state_at_least_0_95": results["causal_core"]["aggregate"]["final_state_full_exact"] >= 0.95,
            "causal_answer_at_least_0_95": results["causal_core"]["aggregate"]["final_answer_accuracy"] >= 0.95,
            "all_state_token_at_least_0_995": all(
                results[split]["aggregate"]["state_token_accuracy"] >= 0.995
                for split in FORMAL_SPLITS
            ),
            "architecture_integrity": model.integrity_report()["passed"],
            "data_manifest_hash_matches": file_sha256(data_dir / "manifest.json")
            == model.a113_data_manifest_sha256,
        }
    )
    return {
        "schema_version": (
            A116_FORMAL_SCHEMA
            if model.a113_config.query_coupled_answer
            and model.a113_answer_loss_weight == 0.0
            else A115_FORMAL_SCHEMA
            if model.a113_config.query_coupled_answer
            else FORMAL_SCHEMA
        ),
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "arm": model.arm,
        "model_seed": model.a113_model_seed,
        "data_seed": model.a113_data_seed,
        "splits": results,
        "gates": gates,
        "passed": all(gates.values()),
    }
