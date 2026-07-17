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
from .a1_8_data import IN_RANGE_LENGTHS, SHORT_LENGTHS, TRAIN_LENGTHS, load_a18_records
from .a1_9_cache import file_sha256
from .a1_10_cache import select_length_balanced_overfit32
from .a1_10_train import TRAIN_RECURRENT_STEPS, compute_a110_loss, encode_a110_targets
from .a1_11_train import A111RecordSplit, FORMAL_SPLITS, HARD_OOD_LENGTHS, _symbolic_reasoner_inputs
from .a1_12_models import A112ReasonerConfig, A112RootCauseReasoner


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.12.binding-cursor-reasoner.checkpoint.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.12.binding-cursor-reasoner.results.v1"
FORMAL_SCHEMA = "yggdrasil.v2-a1.12.binding-cursor-reasoner.formal.v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def evaluate_a112_items(
    model: A112RootCauseReasoner,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    recurrent_steps: int,
    batch_size: int = 32,
    disable_recurrence: bool = False,
    start_value_permutation: torch.Tensor | None = None,
    old_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("A1.12 evaluation requires records")
    if old_records is not None and len(old_records) != len(records):
        raise ValueError("A1.12 old records must align")
    model.eval()
    trajectory = final = answer = consistency = state_correct = state_total = 0
    old_trajectory = old_examples = 0
    for start in range(0, len(records), batch_size):
        batch_records = list(records[start : start + batch_size])
        inputs = _symbolic_reasoner_inputs(batch_records, device)
        output = model(
            **inputs,
            recurrent_steps=recurrent_steps,
            disable_recurrence=disable_recurrence,
            start_value_permutation=start_value_permutation,
        )
        labels = encode_a110_targets(batch_records, recurrent_steps, device)
        prediction = output["state_logits"].argmax(dim=-1)
        equal = prediction == labels["state_targets"]
        trajectory += int(equal.all(dim=(-1, -2)).sum())
        final += int(equal[:, -1].all(dim=-1).sum())
        state_correct += int(equal.sum())
        state_total += int(equal.numel())
        answer_prediction = output["answer_logits"].argmax(dim=-1)
        answer += int((answer_prediction == labels["answer_targets"]).sum())
        state_answer = prediction[:, -1].gather(
            1, labels["query_register"].view(-1, 1)
        ).squeeze(1)
        consistency += int((answer_prediction == state_answer).sum())
        if old_records is not None:
            old_batch = old_records[start : start + len(batch_records)]
            old_labels = encode_a110_targets(old_batch, recurrent_steps, device)
            old_trajectory += int(
                (prediction == old_labels["state_targets"]).all(dim=(-1, -2)).sum()
            )
            old_examples += len(batch_records)
    result: dict[str, Any] = {
        "examples": len(records),
        "recurrent_steps": recurrent_steps,
        "trajectory_full_exact": trajectory / len(records),
        "final_state_full_exact": final / len(records),
        "state_token_accuracy": state_correct / max(1, state_total),
        "final_answer_accuracy": answer / len(records),
        "answer_state_prediction_consistency": consistency / len(records),
        "disable_recurrence": disable_recurrence,
        "start_value_permutation": (
            start_value_permutation.detach().cpu().tolist()
            if start_value_permutation is not None
            else None
        ),
    }
    if old_records is not None:
        result["old_oracle_trajectory_full_exact"] = old_trajectory / max(1, old_examples)
        result["old_oracle_examples"] = old_examples
    return result


@torch.no_grad()
def evaluate_a112_by_length(
    model: A112RootCauseReasoner,
    dataset: A111RecordSplit,
    device: str | torch.device,
    *,
    batch_size: int = 32,
    indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    selected = list(range(len(dataset))) if indices is None else list(indices)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index in selected:
        record = dataset[index]
        groups[int(record["program_length"])].append(record)
    by_length = {
        str(length): evaluate_a112_items(
            model,
            rows,
            device,
            recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
            batch_size=batch_size,
        )
        for length, rows in sorted(groups.items())
    }
    total = sum(row["examples"] for row in by_length.values())
    aggregate: dict[str, Any] = {"examples": total}
    for metric in (
        "trajectory_full_exact",
        "final_state_full_exact",
        "state_token_accuracy",
        "final_answer_accuracy",
        "answer_state_prediction_consistency",
    ):
        aggregate[metric] = sum(
            row[metric] * row["examples"] for row in by_length.values()
        ) / max(1, total)
    return {"aggregate": aggregate, "by_length": by_length}


def _balanced_subset_indices(dataset: A111RecordSplit, maximum: int = 512) -> list[int]:
    per_length = max(1, maximum // len(dataset.indices_by_length))
    return [
        index
        for length in sorted(dataset.indices_by_length)
        for index in dataset.indices_by_length[length][:per_length]
    ]


def _validation_score(metrics: dict[str, Any]) -> tuple[tuple[float, ...], dict[str, float]]:
    rows = list(metrics["by_length"].values())
    components = {
        "minimum_per_length_trajectory": min(row["trajectory_full_exact"] for row in rows),
        "minimum_per_length_final_state": min(row["final_state_full_exact"] for row in rows),
        "minimum_per_length_answer": min(row["final_answer_accuracy"] for row in rows),
        "aggregate_state_token": metrics["aggregate"]["state_token_accuracy"],
    }
    return tuple(components.values()), components


@dataclass(frozen=True)
class A112TrainSpec:
    model_seed: int
    data_seed: int
    entity_addressable: bool
    aligned_operation_cursor: bool
    steps: int = 4000
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_interval: int = 200
    early_stop_patience: int = 6
    train_recurrent_steps: int = TRAIN_RECURRENT_STEPS


def _checkpoint(
    model: A112RootCauseReasoner,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A112TrainSpec,
    data_manifest_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "model_config": asdict(model.config),
        "train_spec": asdict(spec),
        "data_manifest_sha256": data_manifest_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "arm": model.arm,
        "global_train_recurrent_steps": TRAIN_RECURRENT_STEPS,
    }


def _sample_balanced_batch(
    dataset: A111RecordSplit, batch_size: int, rng: random.Random
) -> list[dict[str, Any]]:
    lengths = sorted(dataset.indices_by_length)
    base, remainder = divmod(batch_size, len(lengths))
    selected_lengths = [length for length in lengths for _ in range(base)]
    selected_lengths.extend(rng.sample(lengths, remainder))
    indices = [rng.choice(dataset.indices_by_length[length]) for length in selected_lengths]
    rng.shuffle(indices)
    return dataset.items(indices)


def train_a112(
    data_dir: Path,
    output_dir: Path,
    *,
    spec: A112TrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.12 checkpoints: {output_dir}")
    if spec.train_recurrent_steps != TRAIN_RECURRENT_STEPS:
        raise ValueError("A1.12 train recurrent steps are frozen at 16")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(data_manifest["data_seed"]) != spec.data_seed:
        raise ValueError("A1.12 spec does not match data seed")
    data_manifest_hash = file_sha256(data_dir / "manifest.json")
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
    model = A112RootCauseReasoner(
        A112ReasonerConfig(
            entity_addressable=spec.entity_addressable,
            aligned_operation_cursor=spec.aligned_operation_cursor,
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
        loss, metrics = compute_a110_loss(output, labels)
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
            checkpoint = _checkpoint(
                model, optimizer, step, spec, data_manifest_hash
            )
            torch.save(checkpoint, output_dir / "latest.pt")
            if best_score is None or score > best_score:
                best_score, best_step, best_validation, patience = score, step, validation, 0
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                patience += 1
            target = 1.0 if overfit_mode else 0.995
            ready = all(
                components[name] >= target
                for name in (
                    "minimum_per_length_trajectory",
                    "minimum_per_length_final_state",
                    "minimum_per_length_answer",
                    "aggregate_state_token",
                )
            )
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
        "schema_version": RESULTS_SCHEMA,
        "stage": "V2-A1.12 Reasoner root-cause localization",
        "arm": model.arm,
        "selection": selection,
        "train_spec": asdict(spec),
        "model_config": asdict(model.config),
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


def load_a112_checkpoint(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> A112RootCauseReasoner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("not an A1.12 checkpoint")
    model = A112RootCauseReasoner(A112ReasonerConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.a112_model_seed = int(checkpoint["train_spec"]["model_seed"])
    model.a112_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a112_data_manifest_sha256 = checkpoint["data_manifest_sha256"]
    model.eval()
    return model


def _metric_gate(
    results: dict[str, Any], split: str, lengths: Sequence[int], metric: str, threshold: float
) -> bool:
    return all(results[split]["by_length"][str(length)][metric] >= threshold for length in lengths)


@torch.no_grad()
def evaluate_a112_formal(
    checkpoint: Path,
    data_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    model = load_a112_checkpoint(checkpoint, device)
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
            == model.a112_data_manifest_sha256,
        }
    )
    return {
        "schema_version": FORMAL_SCHEMA,
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "arm": model.arm,
        "model_seed": model.a112_model_seed,
        "data_seed": model.a112_data_seed,
        "splits": results,
        "gates": gates,
        "passed": all(gates.values()),
    }
