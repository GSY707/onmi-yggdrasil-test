from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    evaluate_a112_by_length,
)
from .a1_13_models import A113ReasonerConfig, A113TransitionClosureReasoner
from .a1_13_train import CLOSURE_WEIGHT, _write_json


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.18.training-scaffold.checkpoint.v1"
DEPLOYMENT_SCHEMA = "yggdrasil.v2-a1.18.auxiliary-stripped-deployment.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.18.training-scaffold.results.v1"
FORMAL_SCHEMA = "yggdrasil.v2-a1.18.training-scaffold.formal.v1"
AUXILIARIES = {"queried_answer", "full_state", "full_trajectory_state"}


@dataclass(frozen=True)
class A118TrainSpec:
    model_seed: int
    data_seed: int
    training_auxiliary: str
    steps: int = 4000
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_interval: int = 200
    auxiliary_weight: float = 1.0
    train_recurrent_steps: int = TRAIN_RECURRENT_STEPS


def compute_a118_loss(
    model: A113TransitionClosureReasoner,
    output: dict[str, Any],
    labels: dict[str, torch.Tensor],
    *,
    auxiliary_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    auxiliary = model.a113_config.training_auxiliary
    if auxiliary not in AUXILIARIES:
        raise ValueError("A1.18 loss requires QAUX or SAUX")
    auxiliary_logits = output.get("training_auxiliary_logits")
    if auxiliary_logits is None:
        raise RuntimeError("A1.18 auxiliary logits are available only in training mode")

    state_ce = nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, model.a113_config.value_classes),
        labels["state_targets"].reshape(-1),
    )
    predicted = output["trajectory"][:, :, : model.a113_config.entity_slots]
    closure_targets = model.closure_targets(labels["state_targets"]).detach()
    closure_loss = (
        1.0 - nn.functional.cosine_similarity(predicted, closure_targets, dim=-1)
    ).mean()
    if auxiliary == "queried_answer":
        auxiliary_targets = labels["answer_targets"]
    elif auxiliary == "full_trajectory_state":
        auxiliary_targets = labels["state_targets"]
    else:
        auxiliary_targets = labels["state_targets"][:, -1]
    auxiliary_ce = nn.functional.cross_entropy(
        auxiliary_logits.reshape(-1, model.a113_config.value_classes),
        auxiliary_targets.reshape(-1),
    )
    official_answer_ce = nn.functional.cross_entropy(
        output["answer_logits"], labels["answer_targets"]
    )
    total = state_ce + CLOSURE_WEIGHT * closure_loss + auxiliary_weight * auxiliary_ce
    auxiliary_accuracy = (
        auxiliary_logits.argmax(dim=-1) == auxiliary_targets
    ).to(dtype=torch.float32).mean()
    return total, {
        "loss": float(total.detach()),
        "state_ce": float(state_ce.detach()),
        "closure_loss": float(closure_loss.detach()),
        "closure_weight": CLOSURE_WEIGHT,
        "training_auxiliary": auxiliary,
        "auxiliary_ce": float(auxiliary_ce.detach()),
        "auxiliary_accuracy": float(auxiliary_accuracy.detach()),
        "auxiliary_weight": auxiliary_weight,
        "official_answer_ce_diagnostic_only": float(official_answer_ce.detach()),
        "official_answer_loss_weight": 0.0,
    }


def _state_validation_score(
    metrics: dict[str, Any],
) -> tuple[tuple[float, ...], dict[str, float]]:
    rows = list(metrics["by_length"].values())
    components = {
        "minimum_per_length_trajectory": min(
            row["trajectory_full_exact"] for row in rows
        ),
        "minimum_per_length_final_state": min(
            row["final_state_full_exact"] for row in rows
        ),
        "aggregate_state_token": metrics["aggregate"]["state_token_accuracy"],
    }
    return tuple(components.values()), components


def _checkpoint(
    model: A113TransitionClosureReasoner,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A118TrainSpec,
    data_manifest_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "model_config": asdict(model.a113_config),
        "train_spec": asdict(spec),
        "data_manifest_sha256": data_manifest_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "arm": model.arm,
        "global_train_recurrent_steps": TRAIN_RECURRENT_STEPS,
    }


def _export_deployment(
    training_checkpoint: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    training_config = dict(training_checkpoint["model_config"])
    trained_auxiliary = str(training_config["training_auxiliary"])
    training_config["training_auxiliary"] = "none"
    stripped_names = sorted(
        name
        for name in training_checkpoint["model"]
        if name.startswith("training_auxiliary_head.")
    )
    deployment_state = {
        name: value
        for name, value in training_checkpoint["model"].items()
        if name not in stripped_names
    }
    deployment_model = A113TransitionClosureReasoner(
        A113ReasonerConfig(**training_config)
    )
    deployment_model.load_state_dict(deployment_state, strict=True)
    integrity = deployment_model.integrity_report()
    if not stripped_names or deployment_model.training_auxiliary_head is not None:
        raise RuntimeError("A1.18 deployment export did not physically strip the auxiliary head")
    deployment = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "source_training_schema": training_checkpoint["schema_version"],
        "source_step": training_checkpoint["step"],
        "model_config": training_config,
        "train_spec": training_checkpoint["train_spec"],
        "data_manifest_sha256": training_checkpoint["data_manifest_sha256"],
        "model": deployment_state,
        "trained_auxiliary": trained_auxiliary,
        "stripped_parameter_names": stripped_names,
        "official_answer_source": "shared_state_head(last_recurrent_queried_entity_slot)",
        "integrity": integrity,
    }
    torch.save(deployment, output_path)
    return {
        "path": str(output_path),
        "sha256": file_sha256(output_path),
        "trained_auxiliary": trained_auxiliary,
        "stripped_parameter_names": stripped_names,
        "integrity": integrity,
    }


def train_a118(
    data_dir: Path,
    output_dir: Path,
    *,
    spec: A118TrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.18 checkpoints: {output_dir}")
    if spec.training_auxiliary not in AUXILIARIES:
        raise ValueError(
            "A1.18 auxiliary must be queried_answer, full_state, or full_trajectory_state"
        )
    if spec.train_recurrent_steps != TRAIN_RECURRENT_STEPS:
        raise ValueError("A1.18 recurrent steps are frozen at 16")
    if spec.auxiliary_weight != 1.0:
        raise ValueError("A1.18 auxiliary weight is preregistered at 1.0")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_dir / "manifest.json"
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(data_manifest["data_seed"]) != spec.data_seed:
        raise ValueError("A1.18 spec does not match data seed")
    data_manifest_hash = file_sha256(manifest_path)
    if overfit_mode:
        selected = select_length_balanced_overfit32(load_a18_records(data_dir, "train"))
        train_dataset = validation_dataset = A111RecordSplit(selected)
        selection = "length-balanced-overfit32-fit-only"
    else:
        train_dataset = A111RecordSplit(load_a18_records(data_dir, "train"))
        validation_dataset = A111RecordSplit(load_a18_records(data_dir, "validation"))
        selection = "formal-a1.8-records-fixed-budget"

    random.seed(spec.model_seed)
    torch.manual_seed(spec.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.model_seed)
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=True,
            prototype_closure_objective=True,
            query_coupled_answer=True,
            training_auxiliary=spec.training_auxiliary,
        )
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate)
    rng = random.Random(spec.model_seed)
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    ready_streak = processed_transitions = 0
    started = time.perf_counter()

    for step in range(1, spec.steps + 1):
        model.train()
        records = _sample_balanced_batch(train_dataset, spec.batch_size, rng)
        inputs = _symbolic_reasoner_inputs(records, device)
        labels = encode_a110_targets(records, TRAIN_RECURRENT_STEPS, device)
        output = model(**inputs, recurrent_steps=TRAIN_RECURRENT_STEPS)
        loss, metrics = compute_a118_loss(
            model, output, labels, auxiliary_weight=spec.auxiliary_weight
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
            score, components = _state_validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = components
            checkpoint = _checkpoint(model, optimizer, step, spec, data_manifest_hash)
            torch.save(checkpoint, output_dir / "latest.pt")
            if best_score is None or score > best_score:
                best_score, best_step, best_validation = score, step, validation
                torch.save(checkpoint, output_dir / "best.pt")
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
                    "fixed_budget_formal": not overfit_mode,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            if overfit_mode and ready_streak >= 2:
                break
        else:
            history.append(metrics)

    _write_json(output_dir / "history.json", history)
    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
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
    deployment = _export_deployment(best_checkpoint, output_dir / "deployable.pt")
    overfit_gates = {
        "all_length_trajectory_exact": all(
            row["trajectory_full_exact"] == 1.0
            for row in validation["by_length"].values()
        ),
        "all_length_final_state_exact": all(
            row["final_state_full_exact"] == 1.0
            for row in validation["by_length"].values()
        ),
        "all_length_answer_exact": all(
            row["final_answer_accuracy"] == 1.0
            for row in validation["by_length"].values()
        ),
        "training_architecture_integrity": model.integrity_report()["passed"],
        "deployment_auxiliary_stripped": (
            deployment["integrity"]["training_auxiliary"] == "none"
            and not deployment["integrity"]["independent_answer_head"]
        ),
    }
    result = {
        "schema_version": RESULTS_SCHEMA,
        "stage": "V2-A1.18 state-aligned training scaffold",
        "arm": model.arm,
        "selection": selection,
        "train_spec": asdict(spec),
        "model_config": asdict(model.a113_config),
        "objective_contract": {
            "state_loss": "per-step CE",
            "closure_loss": "per-step stop-gradient prototype cosine",
            "official_answer_loss_weight": 0.0,
            "training_auxiliary": spec.training_auxiliary,
            "training_auxiliary_weight": spec.auxiliary_weight,
            "checkpoint_selection": "state metrics only",
            "formal_budget_fixed": not overfit_mode,
            "deployment_auxiliary_policy": "physically stripped before formal evaluation",
        },
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": history[-1]["step"],
            "processed_examples": history[-1]["step"] * spec.batch_size,
            "processed_recurrent_transitions": processed_transitions,
            "seconds": time.perf_counter() - started,
            "fresh_initialization": True,
            "overfit_mode": overfit_mode,
            "fixed_budget_completed": overfit_mode or history[-1]["step"] == spec.steps,
        },
        "best_checkpoint_step": best_step,
        "best_validation_at_save": best_validation,
        "validation": validation,
        "fit_diagnostic": fit_diagnostic,
        "training_integrity": model.integrity_report(),
        "deployment": deployment,
        "parameter_report": model.parameter_report(),
        "data_manifest_sha256": data_manifest_hash,
        "overfit_gates": overfit_gates if overfit_mode else None,
        "overfit_passed": all(overfit_gates.values()) if overfit_mode else None,
    }
    _write_json(output_dir / "results.json", result)
    return result


def load_a118_deployment(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> A113TransitionClosureReasoner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("A1.18 formal evaluation requires an auxiliary-stripped deployment")
    if checkpoint["model_config"].get("training_auxiliary") != "none":
        raise ValueError("A1.18 deployment still declares a training auxiliary")
    if any(name.startswith("training_auxiliary_head.") for name in checkpoint["model"]):
        raise ValueError("A1.18 deployment still contains auxiliary parameters")
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.a118_model_seed = int(checkpoint["train_spec"]["model_seed"])
    model.a118_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a118_data_manifest_sha256 = checkpoint["data_manifest_sha256"]
    model.a118_trained_auxiliary = str(checkpoint["trained_auxiliary"])
    model.a118_deployment_sha256 = file_sha256(checkpoint_path)
    model.eval()
    return model


@torch.no_grad()
def evaluate_a118_formal(
    checkpoint: Path,
    data_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    model = load_a118_deployment(checkpoint, device)
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
        for metric in (
            "trajectory_full_exact",
            "final_state_full_exact",
            "final_answer_accuracy",
        ):
            gates[f"{prefix}_{metric}_gate"] = _metric_gate(
                results, split, lengths, metric, threshold
            )
    integrity = model.integrity_report()
    gates.update(
        {
            "causal_trajectory_at_least_0_95": results["causal_core"]["aggregate"]["trajectory_full_exact"] >= 0.95,
            "causal_final_state_at_least_0_95": results["causal_core"]["aggregate"]["final_state_full_exact"] >= 0.95,
            "causal_answer_at_least_0_95": results["causal_core"]["aggregate"]["final_answer_accuracy"] >= 0.95,
            "all_state_token_at_least_0_995": all(
                results[split]["aggregate"]["state_token_accuracy"] >= 0.995
                for split in FORMAL_SPLITS
            ),
            "architecture_integrity": integrity["passed"],
            "official_answer_query_coupled": (
                integrity["query_coupled_answer"]
                and not integrity["independent_answer_head"]
            ),
            "deployment_auxiliary_absent": (
                integrity["training_auxiliary"] == "none"
                and integrity["training_auxiliary_head_outputs"] == 0
            ),
            "data_manifest_hash_matches": file_sha256(data_dir / "manifest.json")
            == model.a118_data_manifest_sha256,
        }
    )
    return {
        "schema_version": FORMAL_SCHEMA,
        "checkpoint": str(checkpoint),
        "deployment_sha256": model.a118_deployment_sha256,
        "data_dir": str(data_dir),
        "arm": f"STRUCTURED-CLOSURE-COUPLED-{model.a118_trained_auxiliary.upper()}",
        "trained_auxiliary": model.a118_trained_auxiliary,
        "model_seed": model.a118_model_seed,
        "data_seed": model.a118_data_seed,
        "deployment_integrity": integrity,
        "splits": results,
        "gates": gates,
        "passed": all(gates.values()),
    }
