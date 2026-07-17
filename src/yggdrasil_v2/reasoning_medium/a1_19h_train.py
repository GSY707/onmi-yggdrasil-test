from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .a1_7_train import _gradient_norm
from .a1_8_data import IN_RANGE_LENGTHS, SHORT_LENGTHS, load_a18_records
from .a1_9_cache import file_sha256
from .a1_10_cache import select_length_balanced_overfit32
from .a1_10_train import TRAIN_RECURRENT_STEPS
from .a1_11_train import A111RecordSplit, FORMAL_SPLITS, HARD_OOD_LENGTHS
from .a1_12_train import (
    _balanced_subset_indices,
    _metric_gate,
    _sample_balanced_batch,
)
from .a1_13_train import CLOSURE_WEIGHT, _write_json
from .a1_19h_data import A119HBatch, encode_a119h_records
from .a1_19h_model import A119HConfig, A119HHybridReasoner


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.19h.h1-opaque-handle.checkpoint.v1"
DEPLOYMENT_SCHEMA = "yggdrasil.v2-a1.19h.h1-opaque-handle.deployment.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.19h.h1-opaque-handle.results.v1"
FORMAL_SCHEMA = "yggdrasil.v2-a1.19h.h1-opaque-handle.formal.v1"


@dataclass(frozen=True)
class A119HTrainSpec:
    model_seed: int
    data_seed: int
    mapping_seed: int
    steps: int = 4000
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_interval: int = 200
    auxiliary_weight: float = 1.0
    train_recurrent_steps: int = TRAIN_RECURRENT_STEPS


def compute_a119h_loss(
    model: A119HHybridReasoner,
    output: dict[str, Any],
    batch: A119HBatch,
    *,
    auxiliary_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    auxiliary_logits = output.get("training_auxiliary_logits")
    if auxiliary_logits is None:
        raise RuntimeError("A1.19H training requires the trajectory set-state auxiliary")
    state_ce = nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, model.config.value_classes),
        batch.state_targets.reshape(-1),
        ignore_index=-100,
    )
    predicted = output["trajectory"][:, :, : model.config.maximum_entities]
    closure_targets = model.closure_targets(batch.state_targets).detach()
    closure_per_slot = 1.0 - nn.functional.cosine_similarity(
        predicted, closure_targets, dim=-1
    )
    active = batch.state_targets != -100
    closure_loss = closure_per_slot[active].mean()
    auxiliary_ce = nn.functional.cross_entropy(
        auxiliary_logits.reshape(-1, model.config.value_classes),
        batch.state_targets.reshape(-1),
        ignore_index=-100,
    )
    official_answer_ce = nn.functional.cross_entropy(
        output["answer_logits"], batch.answer_targets
    )
    total = state_ce + CLOSURE_WEIGHT * closure_loss + auxiliary_weight * auxiliary_ce
    auxiliary_prediction = auxiliary_logits.argmax(dim=-1)
    auxiliary_accuracy = (auxiliary_prediction[active] == batch.state_targets[active]).float().mean()
    return total, {
        "loss": float(total.detach()),
        "state_ce": float(state_ce.detach()),
        "closure_loss": float(closure_loss.detach()),
        "closure_weight": CLOSURE_WEIGHT,
        "training_auxiliary": "trajectory_set_state",
        "auxiliary_ce": float(auxiliary_ce.detach()),
        "auxiliary_accuracy": float(auxiliary_accuracy.detach()),
        "auxiliary_weight": auxiliary_weight,
        "official_answer_ce_diagnostic_only": float(official_answer_ce.detach()),
        "official_answer_loss_weight": 0.0,
    }


@torch.no_grad()
def evaluate_a119h_items(
    model: A119HHybridReasoner,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    mapping_seed: int,
    recurrent_steps: int,
    batch_size: int = 32,
    old_records: Sequence[dict[str, Any]] | None = None,
    disable_recurrence: bool = False,
    start_value_permutation: torch.Tensor | None = None,
    handle_alias_seed: int | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("A1.19H evaluation requires records")
    if old_records is not None and len(old_records) != len(records):
        raise ValueError("old records must align with A1.19H records")
    model.eval()
    trajectory_full = final_state_full = answer_correct = consistency = 0
    state_correct = state_total = old_trajectory_full = old_examples = 0
    for start in range(0, len(records), batch_size):
        rows = list(records[start : start + batch_size])
        batch = encode_a119h_records(
            rows,
            recurrent_steps,
            device,
            mapping_seed=mapping_seed,
            maximum_entities=model.config.maximum_entities,
            handle_alias_seed=handle_alias_seed,
        )
        output = model(
            **batch.inputs,
            recurrent_steps=recurrent_steps,
            disable_recurrence=disable_recurrence,
            start_value_permutation=start_value_permutation,
        )
        prediction = output["state_logits"].argmax(dim=-1)
        active = batch.state_targets != -100
        equal = (prediction == batch.state_targets) | ~active
        trajectory_full += int(equal.all(dim=(-1, -2)).sum())
        final_state_full += int(equal[:, -1].all(dim=-1).sum())
        state_correct += int(((prediction == batch.state_targets) & active).sum())
        state_total += int(active.sum())
        answer_prediction = output["answer_logits"].argmax(dim=-1)
        answer_correct += int((answer_prediction == batch.answer_targets).sum())
        query_weights = (
            (
                batch.inputs["entity_handles"]
                == batch.inputs["query_handle"].unsqueeze(-1)
            )
            & batch.inputs["entity_mask"]
        ).long()
        state_answer = (prediction[:, -1] * query_weights).sum(dim=-1)
        consistency += int((answer_prediction == state_answer).sum())
        if old_records is not None:
            old_rows = list(old_records[start : start + len(rows)])
            old_batch = encode_a119h_records(
                old_rows,
                recurrent_steps,
                device,
                mapping_seed=mapping_seed,
                maximum_entities=model.config.maximum_entities,
                handle_alias_seed=handle_alias_seed,
            )
            old_active = old_batch.state_targets != -100
            old_equal = (prediction == old_batch.state_targets) | ~old_active
            old_trajectory_full += int(old_equal.all(dim=(-1, -2)).sum())
            old_examples += len(rows)
    result: dict[str, Any] = {
        "examples": len(records),
        "recurrent_steps": recurrent_steps,
        "trajectory_full_exact": trajectory_full / len(records),
        "final_state_full_exact": final_state_full / len(records),
        "state_token_accuracy": state_correct / max(1, state_total),
        "final_answer_accuracy": answer_correct / len(records),
        "answer_state_prediction_consistency": consistency / len(records),
        "disable_recurrence": disable_recurrence,
        "handle_alias_seed": handle_alias_seed,
    }
    if old_records is not None:
        result["old_oracle_trajectory_full_exact"] = old_trajectory_full / max(
            1, old_examples
        )
    return result


def _weighted(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["examples"] for row in rows)
    aggregate: dict[str, Any] = {"examples": total}
    for metric in (
        "trajectory_full_exact",
        "final_state_full_exact",
        "state_token_accuracy",
        "final_answer_accuracy",
        "answer_state_prediction_consistency",
        "old_oracle_trajectory_full_exact",
    ):
        available = [row for row in rows if metric in row]
        if available:
            denominator = sum(row["examples"] for row in available)
            aggregate[metric] = sum(
                row[metric] * row["examples"] for row in available
            ) / max(1, denominator)
    return aggregate


@torch.no_grad()
def evaluate_a119h_by_length(
    model: A119HHybridReasoner,
    dataset: A111RecordSplit,
    device: str | torch.device,
    *,
    mapping_seed: int,
    batch_size: int = 32,
    indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    selected = list(range(len(dataset))) if indices is None else list(indices)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index in selected:
        record = dataset[index]
        groups[int(record["program_length"])].append(record)
    by_length = {
        str(length): evaluate_a119h_items(
            model,
            rows,
            device,
            mapping_seed=mapping_seed,
            recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
            batch_size=batch_size,
        )
        for length, rows in sorted(groups.items())
    }
    return {"aggregate": _weighted(list(by_length.values())), "by_length": by_length}


def _state_validation_score(metrics: dict[str, Any]) -> tuple[tuple[float, ...], dict[str, float]]:
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
    model: A119HHybridReasoner,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A119HTrainSpec,
    manifest_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "model_config": asdict(model.config),
        "train_spec": asdict(spec),
        "data_manifest_sha256": manifest_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def _export_deployment(
    checkpoint: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    config = dict(checkpoint["model_config"])
    trained_auxiliary = str(config["training_auxiliary"])
    config["training_auxiliary"] = "none"
    stripped = sorted(
        name
        for name in checkpoint["model"]
        if name.startswith("training_auxiliary_head.")
    )
    state = {
        name: value for name, value in checkpoint["model"].items() if name not in stripped
    }
    model = A119HHybridReasoner(A119HConfig(**config))
    model.load_state_dict(state, strict=True)
    integrity = model.integrity_report()
    if not integrity["passed"]:
        raise RuntimeError("A1.19H deployment integrity failed")
    payload = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "model_config": config,
        "train_spec": checkpoint["train_spec"],
        "data_manifest_sha256": checkpoint["data_manifest_sha256"],
        "trained_auxiliary": trained_auxiliary,
        "stripped_parameter_names": stripped,
        "model": state,
        "integrity": integrity,
    }
    torch.save(payload, output_path)
    return {
        "path": str(output_path),
        "schema_version": DEPLOYMENT_SCHEMA,
        "trained_auxiliary": trained_auxiliary,
        "stripped_parameter_names": stripped,
        "integrity": integrity,
    }


def train_a119h_h1(
    data_dir: Path,
    output_dir: Path,
    *,
    spec: A119HTrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.19H requested CUDA but CUDA is unavailable")
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.19H checkpoints: {output_dir}")
    if spec.train_recurrent_steps != TRAIN_RECURRENT_STEPS:
        raise ValueError("A1.19H-H1 recurrent steps are frozen at 16")
    if spec.auxiliary_weight != 1.0:
        raise ValueError("A1.19H-H1 auxiliary weight is frozen at 1.0")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["data_seed"]) != spec.data_seed:
        raise ValueError("A1.19H spec/data seed mismatch")
    manifest_hash = file_sha256(manifest_path)
    if overfit_mode:
        selected = select_length_balanced_overfit32(load_a18_records(data_dir, "train"))
        train_dataset = validation_dataset = A111RecordSplit(selected)
        selection = "length-balanced-overfit32-fit-only"
    else:
        train_dataset = A111RecordSplit(load_a18_records(data_dir, "train"))
        validation_dataset = A111RecordSplit(load_a18_records(data_dir, "validation"))
        selection = "formal-a1.8-records-fixed-budget-with-deterministic-opaque-mapping"

    random.seed(spec.model_seed)
    torch.manual_seed(spec.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.model_seed)
    model = A119HHybridReasoner(A119HConfig()).to(device)
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
        batch = encode_a119h_records(
            records,
            TRAIN_RECURRENT_STEPS,
            device,
            mapping_seed=spec.mapping_seed,
            maximum_entities=model.config.maximum_entities,
        )
        output = model(**batch.inputs, recurrent_steps=TRAIN_RECURRENT_STEPS)
        loss, metrics = compute_a119h_loss(
            model, output, batch, auxiliary_weight=spec.auxiliary_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics["gradient_norm"] = _gradient_norm(model)
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        processed_transitions += spec.batch_size * TRAIN_RECURRENT_STEPS
        metrics["step"] = step
        if step % spec.validation_interval == 0 or step == spec.steps:
            validation = evaluate_a119h_by_length(
                model,
                validation_dataset,
                device,
                mapping_seed=spec.mapping_seed,
                batch_size=spec.batch_size,
            )
            score, components = _state_validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = components
            checkpoint = _checkpoint(model, optimizer, step, spec, manifest_hash)
            torch.save(checkpoint, output_dir / "latest.pt")
            if best_score is None or score > best_score:
                best_score, best_step, best_validation = score, step, validation
                torch.save(checkpoint, output_dir / "best.pt")
            target = 1.0 if overfit_mode else 0.995
            ready = all(value >= target for value in components.values())
            ready_streak = ready_streak + 1 if ready else 0
            _write_json(
                output_dir / "progress.json",
                {
                    "stage": "A1.19H-H1",
                    "step": step,
                    "steps_requested": spec.steps,
                    "best_step": best_step,
                    "best_score": best_score,
                    "fixed_budget_formal": not overfit_mode,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            if overfit_mode and ready_streak >= 2:
                history.append(metrics)
                break
        history.append(metrics)

    _write_json(output_dir / "history.json", history)
    best_checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    validation = evaluate_a119h_by_length(
        model,
        validation_dataset,
        device,
        mapping_seed=spec.mapping_seed,
        batch_size=spec.batch_size,
    )
    fit_diagnostic = evaluate_a119h_by_length(
        model,
        train_dataset,
        device,
        mapping_seed=spec.mapping_seed,
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
            and deployment["integrity"]["training_auxiliary_outputs_per_entity"] == 0
        ),
    }
    result = {
        "schema_version": RESULTS_SCHEMA,
        "stage": "V2-A1.19H-H1 opaque-handle hybrid core",
        "selection": selection,
        "train_spec": asdict(spec),
        "model_config": asdict(model.config),
        "objective_contract": {
            "state_loss": "per-step shared-slot CE",
            "closure_loss": "per-step shared value-prototype cosine",
            "official_answer_loss_weight": 0.0,
            "training_auxiliary": "per-step set-equivariant global-context state CE",
            "checkpoint_selection": "state metrics only",
            "deployment_auxiliary_policy": "physically stripped before formal",
        },
        "address_contract": {
            "handle_assignment": "per-record deterministic random opaque integers",
            "slot_assignment": "per-record deterministic random semantic-to-physical permutation",
            "handle_usage": "equality routing only",
            "semantic_handle_embedding": False,
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
        "data_manifest_sha256": manifest_hash,
        "overfit_gates": overfit_gates if overfit_mode else None,
        "overfit_passed": all(overfit_gates.values()) if overfit_mode else None,
    }
    _write_json(output_dir / "results.json", result)
    return result


def load_a119h_deployment(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> A119HHybridReasoner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("A1.19H formal requires an auxiliary-stripped deployment")
    if checkpoint["model_config"].get("training_auxiliary") != "none":
        raise ValueError("A1.19H deployment still declares an auxiliary")
    if any(
        name.startswith("training_auxiliary_head.") for name in checkpoint["model"]
    ):
        raise ValueError("A1.19H deployment still contains auxiliary parameters")
    model = A119HHybridReasoner(A119HConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.a119h_model_seed = int(checkpoint["train_spec"]["model_seed"])
    model.a119h_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a119h_mapping_seed = int(checkpoint["train_spec"]["mapping_seed"])
    model.a119h_data_manifest_sha256 = checkpoint["data_manifest_sha256"]
    model.a119h_trained_auxiliary = str(checkpoint["trained_auxiliary"])
    model.a119h_deployment_sha256 = file_sha256(checkpoint_path)
    model.eval()
    return model


@torch.no_grad()
def evaluate_a119h_formal(
    checkpoint: Path,
    data_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    model = load_a119h_deployment(checkpoint, device)
    results = {
        split: evaluate_a119h_by_length(
            model,
            A111RecordSplit(load_a18_records(data_dir, split)),
            device,
            mapping_seed=model.a119h_mapping_seed,
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
            "opaque_handle_has_no_semantic_embedding": not integrity["opaque_handle_semantic_embedding"],
            "fixed_register_slot_semantics_absent": not integrity["fixed_register_slot_semantics"],
            "official_answer_query_coupled": integrity["query_coupled_answer"],
            "deployment_auxiliary_absent": (
                integrity["training_auxiliary"] == "none"
                and integrity["training_auxiliary_outputs_per_entity"] == 0
            ),
            "data_manifest_hash_matches": file_sha256(data_dir / "manifest.json")
            == model.a119h_data_manifest_sha256,
        }
    )
    return {
        "schema_version": FORMAL_SCHEMA,
        "checkpoint": str(checkpoint),
        "deployment_sha256": model.a119h_deployment_sha256,
        "data_dir": str(data_dir),
        "stage": "V2-A1.19H-H1 opaque-handle hybrid core",
        "model_seed": model.a119h_model_seed,
        "data_seed": model.a119h_data_seed,
        "mapping_seed": model.a119h_mapping_seed,
        "deployment_integrity": integrity,
        "splits": results,
        "gates": gates,
        "passed": all(gates.values()),
    }
