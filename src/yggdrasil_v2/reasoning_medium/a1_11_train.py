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

from .a1_7_data import REGISTER_NAMES, VALUE_LABELS
from .a1_7_train import (
    _gradient_norm,
    _last_active_state_logits,
    _last_state_targets,
    encode_a17_records,
    prototype_statistics,
)
from .a1_8_data import IN_RANGE_LENGTHS, OOD_LENGTHS, SHORT_LENGTHS, TRAIN_LENGTHS, load_a18_records
from .a1_8_train import load_a18_checkpoint
from .a1_9_model import module_state_sha256
from .a1_9_train import CLOSURE_WEIGHT, compute_a19_loss
from .a1_10_cache import (
    A110CachedSplit,
    CACHE_MANIFEST_SCHEMA as A110_CACHE_MANIFEST_SCHEMA,
    collate_a110_items,
    sample_length_balanced_indices,
    select_length_balanced_overfit32,
)
from .a1_10_train import TRAIN_RECURRENT_STEPS, compute_a110_loss, encode_a110_targets
from .a1_11_models import (
    A111BoundaryConfig,
    A111LearnedFullTextBoundary,
    A111OracleRoleReasoner,
    A111ReasonerConfig,
)
from .a1_9_cache import file_sha256


BOUNDARY_CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.11.learned-full-text-boundary.checkpoint.v1"
BOUNDARY_RESULTS_SCHEMA = "yggdrasil.v2-a1.11.learned-full-text-boundary.results.v1"
BOUNDARY_FORMAL_SCHEMA = "yggdrasil.v2-a1.11.learned-full-text-boundary.formal.v1"
REASONER_CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.11.exact-symbolic-role-anonymous-reasoner.checkpoint.v3"
REASONER_RESULTS_SCHEMA = "yggdrasil.v2-a1.11.exact-symbolic-role-anonymous-reasoner.results.v2"
REASONER_FORMAL_SCHEMA = "yggdrasil.v2-a1.11.exact-symbolic-role-anonymous-reasoner.formal.v2"
HARD_OOD_LENGTHS = (20, 24)
DIAGNOSTIC_OOD_LENGTH = 32
FORMAL_SPLITS = (
    "short_regression",
    "supported_in_range",
    "relation_in_range",
    "supported_ood",
    "relation_ood",
    "causal_core",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _operation_mask(records: Sequence[dict[str, Any]], device: str | torch.device) -> torch.Tensor:
    maximum = max(int(record["program_length"]) for record in records)
    mask = torch.zeros((len(records), maximum), dtype=torch.bool, device=device)
    for row, record in enumerate(records):
        mask[row, : int(record["program_length"])] = True
    return mask


def _boundary_inputs(
    items: Sequence[dict[str, Any]], device: str | torch.device
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    inputs, records = collate_a110_items(items, device)
    inputs["operation_mask"] = _operation_mask(records, device)
    return inputs, records


def _mapping_predictions(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "value": output["value_mapping_logits"].argmax(dim=-1),
        "family": output["family_mapping_logits"].argmax(dim=-1),
        "source": output["source_mapping_logits"].argmax(dim=-1),
        "target": output["target_mapping_logits"].argmax(dim=-1),
        "query": output["query_mapping_logits"].argmax(dim=-1),
    }


@torch.no_grad()
def evaluate_a111_boundary_items(
    model: A111LearnedFullTextBoundary,
    items: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 32,
    old_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not items:
        raise ValueError("A1.11 Boundary evaluation requires items")
    if old_records is not None and len(old_records) != len(items):
        raise ValueError("A1.11 Boundary old records must align")
    model.eval()
    trajectory_full = final_state_full = answer_correct = 0
    state_correct = state_total = source_correct = target_correct = pointer_total = 0
    mapping_correct: dict[str, int] = defaultdict(int)
    mapping_total: dict[str, int] = defaultdict(int)
    identity_max_abs_diff = 0.0
    identity_predictions_equal = True
    old_trajectory_full = old_examples = 0
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        inputs, records = _boundary_inputs(batch_items, device)
        labels = encode_a17_records(records, device)
        output = model(**inputs)
        valid = labels["operation_mask"]
        state_prediction = output["state_logits"].argmax(dim=-1)
        state_equal = state_prediction == labels["state_targets"]
        state_correct += int((state_equal & valid.unsqueeze(-1)).sum())
        state_total += int(valid.sum()) * len(REGISTER_NAMES)
        trajectory_full += int((state_equal | ~valid.unsqueeze(-1)).all(dim=(-1, -2)).sum())
        last_logits = _last_active_state_logits(model.core, output, labels)
        last_targets = _last_state_targets(labels)
        final_state_full += int((last_logits.argmax(dim=-1) == last_targets).all(dim=-1).sum())
        query_index = labels["query_register"].view(-1, 1, 1).expand(-1, 1, len(VALUE_LABELS))
        queried_last_logits = last_logits.gather(1, query_index).squeeze(1)
        identity_max_abs_diff = max(
            identity_max_abs_diff, float((output["answer_logits"] - queried_last_logits).abs().max())
        )
        identity_predictions_equal = identity_predictions_equal and bool(
            torch.equal(output["answer_logits"].argmax(dim=-1), queried_last_logits.argmax(dim=-1))
        )
        answer_correct += int((output["answer_logits"].argmax(dim=-1) == labels["answer_targets"]).sum())
        source_correct += int(
            ((output["source_pointer_logits"].argmax(dim=-1) == labels["operation_source"]) & valid).sum()
        )
        target_correct += int(
            ((output["target_pointer_logits"].argmax(dim=-1) == labels["operation_target"]) & valid).sum()
        )
        pointer_total += int(valid.sum())
        mapping = _mapping_predictions(output)
        mapping_correct["value"] += int((mapping["value"] == labels["start_values"]).sum())
        mapping_total["value"] += int(labels["start_values"].numel())
        mapping_correct["query"] += int((mapping["query"] == labels["query_register"]).sum())
        mapping_total["query"] += len(records)
        for name, label_name in (
            ("family", "operation_family"),
            ("source", "operation_source"),
            ("target", "operation_target"),
        ):
            mapping_correct[name] += int(((mapping[name] == labels[label_name]) & valid).sum())
            mapping_total[name] += int(valid.sum())
        if old_records is not None:
            old_batch = old_records[start : start + len(batch_items)]
            old_labels = encode_a17_records(old_batch, device)
            if old_labels["state_targets"].shape == state_prediction.shape:
                old_valid = old_labels["operation_mask"]
                old_equal = state_prediction == old_labels["state_targets"]
                old_trajectory_full += int((old_equal | ~old_valid.unsqueeze(-1)).all(dim=(-1, -2)).sum())
                old_examples += len(batch_items)
    mapping_accuracy = {
        name: mapping_correct[name] / max(1, mapping_total[name])
        for name in ("value", "family", "source", "target", "query")
    }
    result: dict[str, Any] = {
        "examples": len(items),
        "trajectory_full_exact": trajectory_full / len(items),
        "final_state_full_exact": final_state_full / len(items),
        "final_query_token_accuracy": answer_correct / len(items),
        "state_token_accuracy": state_correct / max(1, state_total),
        "source_pointer_accuracy": source_correct / max(1, pointer_total),
        "target_pointer_accuracy": target_correct / max(1, pointer_total),
        "mapping_accuracy": {**mapping_accuracy, "minimum": min(mapping_accuracy.values())},
        "answer_state_logits_identity": {
            "max_abs_diff": identity_max_abs_diff,
            "prediction_equal": identity_predictions_equal,
            "gate": identity_max_abs_diff <= 1e-6 and identity_predictions_equal,
        },
        "prototype_statistics": prototype_statistics(model.core),
    }
    if old_records is not None:
        result["old_oracle_trajectory_full_exact"] = old_trajectory_full / max(1, old_examples)
        result["old_oracle_examples"] = old_examples
    return result


@torch.no_grad()
def evaluate_a111_boundary_by_length(
    model: A111LearnedFullTextBoundary,
    dataset: A110CachedSplit,
    device: str | torch.device,
    *,
    batch_size: int = 32,
    indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    selected = list(range(len(dataset))) if indices is None else list(indices)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_items: list[dict[str, Any]] = []
    for index in selected:
        item = dataset[index]
        all_items.append(item)
        groups[int(item["record"]["program_length"])].append(item)
    by_length = {
        str(length): evaluate_a111_boundary_items(model, rows, device, batch_size=batch_size)
        for length, rows in sorted(groups.items())
    }
    aggregate = evaluate_a111_boundary_items(model, all_items, device, batch_size=batch_size)
    return {"aggregate": aggregate, "by_length": by_length}


@dataclass(frozen=True)
class A111BoundaryTrainSpec:
    reader_seed: int
    data_seed: int
    core_seed: int
    steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_interval: int = 200
    early_stop_patience: int = 10


def _boundary_validation_score(metrics: dict[str, Any]) -> tuple[tuple[float, ...], dict[str, float]]:
    rows = list(metrics["by_length"].values())
    components = {
        "minimum_per_length_trajectory": min(row["trajectory_full_exact"] for row in rows),
        "minimum_per_length_mapping": min(row["mapping_accuracy"]["minimum"] for row in rows),
        "minimum_source_pointer": min(row["source_pointer_accuracy"] for row in rows),
        "minimum_target_pointer": min(row["target_pointer_accuracy"] for row in rows),
        "aggregate_trajectory": metrics["aggregate"]["trajectory_full_exact"],
    }
    return tuple(components.values()), components


def _balanced_subset_indices(dataset: Any, maximum: int = 512) -> list[int]:
    per_length = max(1, maximum // len(dataset.indices_by_length))
    return [
        index
        for length in sorted(dataset.indices_by_length)
        for index in dataset.indices_by_length[length][:per_length]
    ]


def _boundary_checkpoint(
    model: A111LearnedFullTextBoundary,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A111BoundaryTrainSpec,
    core_checkpoint: Path,
    core_checkpoint_hash: str,
    core_state_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": BOUNDARY_CHECKPOINT_SCHEMA,
        "step": step,
        "boundary_config": asdict(model.config),
        "train_spec": asdict(spec),
        "core_checkpoint": str(core_checkpoint.resolve()),
        "core_checkpoint_sha256": core_checkpoint_hash,
        "core_state_sha256": core_state_hash,
        "reader": model.boundary_state_dict(),
        "optimizer": optimizer.state_dict(),
        "qwen_frozen": True,
        "oracle_span_input": False,
        "operation_mask_retained_for_isolation": True,
    }


def train_a111_boundary(
    cache_dir: Path,
    data_dir: Path,
    core_checkpoint: Path,
    output_dir: Path,
    *,
    spec: A111BoundaryTrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.11 Boundary checkpoints: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != A110_CACHE_MANIFEST_SCHEMA:
        raise ValueError("A1.11 Boundary requires an A1.10 full-token cache")
    selection = manifest["selection"]
    if overfit_mode != (selection == "length-balanced-overfit32-fit-only"):
        raise ValueError("A1.11 Boundary overfit mode/cache selection mismatch")
    train_dataset = A110CachedSplit(cache_dir, data_dir, "train")
    validation_dataset = A110CachedSplit(cache_dir, data_dir, "validation")
    random.seed(spec.reader_seed)
    torch.manual_seed(spec.reader_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.reader_seed)
    core = load_a18_checkpoint(core_checkpoint, device)
    if int(core.a18_data_seed) != spec.data_seed or int(core.a18_model_seed) != spec.core_seed:
        raise ValueError("A1.11 Boundary spec does not match frozen A1.8 core")
    core_checkpoint_hash = file_sha256(core_checkpoint)
    core_state_hash = module_state_sha256(core)
    model = A111LearnedFullTextBoundary(
        A111BoundaryConfig(source_width=train_dataset.hidden_width, latent_width=core.config.latent_width), core
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate)
    rng = random.Random(spec.reader_seed)
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    patience = ready_streak = 0
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        indices = sample_length_balanced_indices(train_dataset, spec.batch_size, rng)
        items = train_dataset.items(indices)
        inputs, records = _boundary_inputs(items, device)
        labels = encode_a17_records(records, device)
        output = model(**inputs)
        loss, metrics = compute_a19_loss(model, output, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in model.core.parameters()):
            raise RuntimeError("frozen A1.8 core received Boundary gradients")
        metrics["gradient_norm"] = _gradient_norm(model)
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        metrics["step"] = step
        if step % spec.validation_interval == 0 or step == spec.steps:
            validation = evaluate_a111_boundary_by_length(
                model, validation_dataset, device, batch_size=spec.batch_size
            )
            score, components = _boundary_validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = components
            checkpoint = _boundary_checkpoint(
                model, optimizer, step, spec, core_checkpoint, core_checkpoint_hash, core_state_hash
            )
            torch.save(checkpoint, output_dir / "latest.pt")
            if best_score is None or score > best_score:
                best_score, best_step, best_validation, patience = score, step, validation, 0
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                patience += 1
            target = 1.0 if overfit_mode else 0.995
            ready = (
                components["minimum_per_length_trajectory"] >= target
                and components["minimum_per_length_mapping"] >= target
                and components["minimum_source_pointer"] >= target
                and components["minimum_target_pointer"] >= target
                and validation["aggregate"]["answer_state_logits_identity"]["gate"]
            )
            ready_streak = ready_streak + 1 if ready else 0
            history.append(metrics)
            _write_json(
                output_dir / "progress.json",
                {
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
    model.load_boundary_state_dict(checkpoint["reader"])
    validation = evaluate_a111_boundary_by_length(model, validation_dataset, device, batch_size=spec.batch_size)
    fit_indices = _balanced_subset_indices(train_dataset)
    fit_diagnostic = evaluate_a111_boundary_by_length(
        model, train_dataset, device, batch_size=spec.batch_size, indices=fit_indices
    )
    core_state_after = module_state_sha256(model.core)
    overfit_gates = {
        "all_length_trajectory_exact": all(
            row["trajectory_full_exact"] == 1.0 for row in validation["by_length"].values()
        ),
        "all_mapping_exact": validation["aggregate"]["mapping_accuracy"]["minimum"] == 1.0,
        "source_pointer_exact": validation["aggregate"]["source_pointer_accuracy"] == 1.0,
        "target_pointer_exact": validation["aggregate"]["target_pointer_accuracy"] == 1.0,
        "answer_exact": validation["aggregate"]["final_query_token_accuracy"] == 1.0,
        "answer_state_identity": validation["aggregate"]["answer_state_logits_identity"]["gate"],
        "core_hash_unchanged": core_state_after == core_state_hash,
    }
    result = {
        "schema_version": BOUNDARY_RESULTS_SCHEMA,
        "arm": "A1.11-Boundary",
        "selection": selection,
        "train_spec": asdict(spec),
        "boundary_config": asdict(model.config),
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": history[-1]["step"],
            "processed_examples": history[-1]["step"] * spec.batch_size,
            "seconds": time.perf_counter() - started,
            "fresh_initialization": True,
            "overfit_mode": overfit_mode,
        },
        "best_checkpoint_step": best_step,
        "best_validation_at_save": best_validation,
        "validation": validation,
        "fit_diagnostic": fit_diagnostic,
        "integrity": model.integrity_report(),
        "core_state_sha256_before": core_state_hash,
        "core_state_sha256_after": core_state_after,
        "overfit_gates": overfit_gates if overfit_mode else None,
        "overfit_passed": all(overfit_gates.values()) if overfit_mode else None,
    }
    _write_json(output_dir / "results.json", result)
    return result


def load_a111_boundary_checkpoint(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> A111LearnedFullTextBoundary:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != BOUNDARY_CHECKPOINT_SCHEMA:
        raise ValueError("not an A1.11 Boundary checkpoint")
    core_path = Path(checkpoint["core_checkpoint"])
    if file_sha256(core_path) != checkpoint["core_checkpoint_sha256"]:
        raise ValueError("A1.11 Boundary core checkpoint hash mismatch")
    core = load_a18_checkpoint(core_path, device)
    if module_state_sha256(core) != checkpoint["core_state_sha256"]:
        raise ValueError("A1.11 Boundary core state hash mismatch")
    model = A111LearnedFullTextBoundary(A111BoundaryConfig(**checkpoint["boundary_config"]), core).to(device)
    model.load_boundary_state_dict(checkpoint["reader"])
    model.a111_reader_seed = int(checkpoint["train_spec"]["reader_seed"])
    model.a111_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a111_core_seed = int(checkpoint["train_spec"]["core_seed"])
    model.a111_core_state_sha256 = checkpoint["core_state_sha256"]
    model.eval()
    return model


def _trajectory_gate(results: dict[str, Any], split: str, lengths: Sequence[int], threshold: float) -> bool:
    return all(results[split]["by_length"][str(length)]["trajectory_full_exact"] >= threshold for length in lengths)


@torch.no_grad()
def evaluate_a111_boundary_formal(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    model = load_a111_boundary_checkpoint(checkpoint, device)
    results = {
        split: evaluate_a111_boundary_by_length(
            model, A110CachedSplit(cache_dir, data_dir, split), device, batch_size=batch_size
        )
        for split in FORMAL_SPLITS
    }
    gates = {
        "short_t1_t6_each_at_least_0_95": _trajectory_gate(results, "short_regression", SHORT_LENGTHS, 0.95),
        "supported_t8_t12_t16_each_at_least_0_95": _trajectory_gate(results, "supported_in_range", IN_RANGE_LENGTHS, 0.95),
        "relation_t8_t12_t16_each_at_least_0_95": _trajectory_gate(results, "relation_in_range", IN_RANGE_LENGTHS, 0.95),
        "supported_t20_t24_each_at_least_0_90": _trajectory_gate(results, "supported_ood", HARD_OOD_LENGTHS, 0.90),
        "relation_t20_t24_each_at_least_0_90": _trajectory_gate(results, "relation_ood", HARD_OOD_LENGTHS, 0.90),
        "causal_free_trajectory_at_least_0_95": results["causal_core"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "all_boundary_mapping_at_least_0_995": all(
            results[split]["aggregate"]["mapping_accuracy"]["minimum"] >= 0.995 for split in FORMAL_SPLITS
        ),
        "all_pointer_accuracy_at_least_0_995": all(
            results[split]["aggregate"]["source_pointer_accuracy"] >= 0.995
            and results[split]["aggregate"]["target_pointer_accuracy"] >= 0.995
            for split in FORMAL_SPLITS
        ),
        "answer_state_identity": all(
            results[split]["aggregate"]["answer_state_logits_identity"]["gate"] for split in FORMAL_SPLITS
        ),
        "prototype_non_collapse": all(
            results[split]["aggregate"]["prototype_statistics"]["non_collapse_gate"] for split in FORMAL_SPLITS
        ),
        "architecture_integrity": model.integrity_report()["passed"],
        "core_state_hash_unchanged": module_state_sha256(model.core) == model.a111_core_state_sha256,
    }
    return {
        "schema_version": BOUNDARY_FORMAL_SCHEMA,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "reader_seed": model.a111_reader_seed,
        "data_seed": model.a111_data_seed,
        "core_seed": model.a111_core_seed,
        "splits": results,
        "gates": gates,
        "diagnostic_only": {"ood_length": DIAGNOSTIC_OOD_LENGTH},
        "passed": all(gates.values()),
    }


@torch.no_grad()
def evaluate_a111_boundary_hard_reembed_diagnostic(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    *,
    split: str = "validation",
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Read-only diagnostic: snap predicted roles to frozen core prototypes after the learned reader."""

    model = load_a111_boundary_checkpoint(checkpoint, device)
    dataset = A110CachedSplit(cache_dir, data_dir, split)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        groups[int(dataset[index]["record"]["program_length"])].append(index)
    by_length: dict[str, Any] = {}
    for length, indices in sorted(groups.items()):
        trajectory = answer = source_pointer = target_pointer = operations = 0
        items = dataset.items(indices)
        for start in range(0, len(items), batch_size):
            inputs, records = _boundary_inputs(items[start : start + batch_size], device)
            labels = encode_a17_records(records, device)
            learned = model(**inputs)
            mapping = _mapping_predictions(learned)
            hard = model.core.rollout(
                model.core.canonical_value_prototypes[mapping["value"]],
                mapping["query"],
                model.core.family_embedding(mapping["family"]),
                model.core.register_embedding(mapping["source"]),
                model.core.register_embedding(mapping["target"]),
                labels["operation_mask"],
            )
            valid = labels["operation_mask"]
            state_equal = hard["state_logits"].argmax(dim=-1) == labels["state_targets"]
            trajectory += int((state_equal | ~valid.unsqueeze(-1)).all(dim=(-1, -2)).sum())
            answer += int((hard["answer_logits"].argmax(dim=-1) == labels["answer_targets"]).sum())
            source_pointer += int(
                ((hard["source_pointer_logits"].argmax(dim=-1) == labels["operation_source"]) & valid).sum()
            )
            target_pointer += int(
                ((hard["target_pointer_logits"].argmax(dim=-1) == labels["operation_target"]) & valid).sum()
            )
            operations += int(valid.sum())
        by_length[str(length)] = {
            "examples": len(items),
            "trajectory_full_exact": trajectory / len(items),
            "final_query_token_accuracy": answer / len(items),
            "source_pointer_accuracy": source_pointer / max(1, operations),
            "target_pointer_accuracy": target_pointer / max(1, operations),
        }
    gates = {
        "all_length_trajectory_exact": all(row["trajectory_full_exact"] == 1.0 for row in by_length.values()),
        "all_length_answer_exact": all(row["final_query_token_accuracy"] == 1.0 for row in by_length.values()),
        "all_length_source_pointer_exact": all(row["source_pointer_accuracy"] == 1.0 for row in by_length.values()),
        "all_length_target_pointer_exact": all(row["target_pointer_accuracy"] == 1.0 for row in by_length.values()),
    }
    return {
        "schema_version": "yggdrasil.v2-a1.11.boundary-hard-reembed-diagnostic.v1",
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "split": split,
        "method": {
            "read_only_checkpoint": True,
            "reader_mapping_argmax_hard_reembedded_to_frozen_core_prototypes": True,
            "not_training": True,
            "not_a_formal_gate": True,
            "not_a_rescue_architecture": True,
        },
        "by_length": by_length,
        "gates": gates,
        "passed": all(gates.values()),
    }


class A111RecordSplit:
    """Immutable A1.8 records grouped by length; no Qwen or boundary cache is loaded."""

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)
        self.indices_by_length: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.indices_by_length[int(record["program_length"])].append(index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    def items(self, indices: Iterable[int] | None = None) -> list[dict[str, Any]]:
        selected = range(len(self.records)) if indices is None else indices
        return [self.records[index] for index in selected]


def _symbolic_reasoner_inputs(
    records: Sequence[dict[str, Any]], device: str | torch.device
) -> dict[str, torch.Tensor]:
    encoded = encode_a17_records(records, device)
    return {
        "start_values": encoded["start_values"],
        "query_register": encoded["query_register"],
        "operation_family": encoded["operation_family"],
        "operation_source": encoded["operation_source"],
        "operation_target": encoded["operation_target"],
        "operation_mask": encoded["operation_mask"],
    }


@torch.no_grad()
def evaluate_a111_reasoner_items(
    model: A111OracleRoleReasoner,
    items: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    recurrent_steps: int,
    batch_size: int = 32,
    disable_recurrence: bool = False,
    slot_permutation: torch.Tensor | None = None,
    old_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not items:
        raise ValueError("A1.11 Reasoner evaluation requires items")
    if old_records is not None and len(old_records) != len(items):
        raise ValueError("A1.11 Reasoner old records must align")
    model.eval()
    trajectory_full = final_state_full = answer_correct = consistency = 0
    state_correct = state_total = old_trajectory_full = old_examples = 0
    for start in range(0, len(items), batch_size):
        records = list(items[start : start + batch_size])
        inputs = _symbolic_reasoner_inputs(records, device)
        output = model(
            **inputs,
            recurrent_steps=recurrent_steps,
            disable_recurrence=disable_recurrence,
            slot_permutation=slot_permutation,
        )
        labels = encode_a110_targets(records, recurrent_steps, device)
        state_prediction = output["state_logits"].argmax(dim=-1)
        state_equal = state_prediction == labels["state_targets"]
        trajectory_full += int(state_equal.all(dim=(-1, -2)).sum())
        final_state_full += int(state_equal[:, -1].all(dim=-1).sum())
        state_correct += int(state_equal.sum())
        state_total += int(state_equal.numel())
        answer_prediction = output["answer_logits"].argmax(dim=-1)
        answer_correct += int((answer_prediction == labels["answer_targets"]).sum())
        query_index = labels["query_register"].view(-1, 1)
        state_answer = state_prediction[:, -1].gather(1, query_index).squeeze(1)
        consistency += int((answer_prediction == state_answer).sum())
        if old_records is not None:
            old_batch = old_records[start : start + len(records)]
            old_labels = encode_a110_targets(old_batch, recurrent_steps, device)
            old_trajectory_full += int(
                (state_prediction == old_labels["state_targets"]).all(dim=(-1, -2)).sum()
            )
            old_examples += len(records)
    result: dict[str, Any] = {
        "examples": len(items),
        "recurrent_steps": recurrent_steps,
        "trajectory_full_exact": trajectory_full / len(items),
        "final_state_full_exact": final_state_full / len(items),
        "state_token_accuracy": state_correct / max(1, state_total),
        "final_answer_accuracy": answer_correct / len(items),
        "answer_state_prediction_consistency": consistency / len(items),
        "disable_recurrence": disable_recurrence,
    }
    if old_records is not None:
        result["old_oracle_trajectory_full_exact"] = old_trajectory_full / max(1, old_examples)
        result["old_oracle_examples"] = old_examples
    return result


@torch.no_grad()
def evaluate_a111_reasoner_by_length(
    model: A111OracleRoleReasoner,
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
        str(length): evaluate_a111_reasoner_items(
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
        aggregate[metric] = sum(row[metric] * row["examples"] for row in by_length.values()) / max(1, total)
    return {"aggregate": aggregate, "by_length": by_length}


@dataclass(frozen=True)
class A111ReasonerTrainSpec:
    reasoner_seed: int
    data_seed: int
    steps: int = 4000
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_interval: int = 200
    early_stop_patience: int = 6
    train_recurrent_steps: int = TRAIN_RECURRENT_STEPS


def _reasoner_validation_score(metrics: dict[str, Any]) -> tuple[tuple[float, ...], dict[str, float]]:
    rows = list(metrics["by_length"].values())
    components = {
        "minimum_per_length_trajectory": min(row["trajectory_full_exact"] for row in rows),
        "minimum_per_length_final_state": min(row["final_state_full_exact"] for row in rows),
        "minimum_per_length_answer": min(row["final_answer_accuracy"] for row in rows),
        "aggregate_state_token": metrics["aggregate"]["state_token_accuracy"],
    }
    return tuple(components.values()), components


def _reasoner_checkpoint(
    model: A111OracleRoleReasoner,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A111ReasonerTrainSpec,
    data_manifest_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": REASONER_CHECKPOINT_SCHEMA,
        "step": step,
        "reasoner_config": asdict(model.config),
        "train_spec": asdict(spec),
        "data_manifest_sha256": data_manifest_hash,
        "reasoner": model.reasoner_state_dict(),
        "optimizer": optimizer.state_dict(),
        "exact_symbolic_oracle_roles": True,
        "qwen_or_boundary_adapter_loaded": False,
        "structured_a1_8_core_loaded": False,
        "global_train_recurrent_steps": TRAIN_RECURRENT_STEPS,
    }


def train_a111_reasoner(
    data_dir: Path,
    output_dir: Path,
    *,
    spec: A111ReasonerTrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.11 Reasoner checkpoints: {output_dir}")
    if spec.train_recurrent_steps != TRAIN_RECURRENT_STEPS:
        raise ValueError("A1.11 Reasoner train recurrent steps are frozen at 16")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(data_manifest["data_seed"]) != spec.data_seed:
        raise ValueError("A1.11 Reasoner spec does not match the A1.8 data seed")
    data_manifest_hash = file_sha256(data_dir / "manifest.json")
    if overfit_mode:
        selected = select_length_balanced_overfit32(load_a18_records(data_dir, "train"))
        train_dataset = A111RecordSplit(selected)
        validation_dataset = A111RecordSplit(selected)
        selection = "length-balanced-overfit32-fit-only"
    else:
        train_dataset = A111RecordSplit(load_a18_records(data_dir, "train"))
        validation_dataset = A111RecordSplit(load_a18_records(data_dir, "validation"))
        selection = "formal-a1.8-records"
    random.seed(spec.reasoner_seed)
    torch.manual_seed(spec.reasoner_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.reasoner_seed)
    model = A111OracleRoleReasoner(A111ReasonerConfig()).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate)
    rng = random.Random(spec.reasoner_seed)
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    patience = ready_streak = 0
    processed_transitions = 0
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        indices = [
            rng.choice(train_dataset.indices_by_length[length])
            for length in rng.sample(
                [length for length in sorted(train_dataset.indices_by_length) for _ in range(spec.batch_size // len(TRAIN_LENGTHS))],
                spec.batch_size,
            )
        ] if spec.batch_size % len(TRAIN_LENGTHS) == 0 else [
            rng.choice(train_dataset.indices_by_length[length])
            for length in (list(TRAIN_LENGTHS) * ((spec.batch_size + len(TRAIN_LENGTHS) - 1) // len(TRAIN_LENGTHS)))[: spec.batch_size]
        ]
        rng.shuffle(indices)
        records = train_dataset.items(indices)
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
            validation = evaluate_a111_reasoner_by_length(
                model, validation_dataset, device, batch_size=spec.batch_size
            )
            score, components = _reasoner_validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = components
            checkpoint = _reasoner_checkpoint(
                model,
                optimizer,
                step,
                spec,
                data_manifest_hash,
            )
            torch.save(checkpoint, output_dir / "latest.pt")
            if best_score is None or score > best_score:
                best_score, best_step, best_validation, patience = score, step, validation, 0
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                patience += 1
            target = 1.0 if overfit_mode else 0.995
            ready = (
                components["minimum_per_length_trajectory"] >= target
                and components["minimum_per_length_final_state"] >= target
                and components["minimum_per_length_answer"] >= target
                and components["aggregate_state_token"] >= target
            )
            ready_streak = ready_streak + 1 if ready else 0
            history.append(metrics)
            _write_json(
                output_dir / "progress.json",
                {
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
    model.reasoner.load_state_dict(checkpoint["reasoner"])
    validation = evaluate_a111_reasoner_by_length(model, validation_dataset, device, batch_size=spec.batch_size)
    fit_diagnostic = evaluate_a111_reasoner_by_length(
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
        "schema_version": REASONER_RESULTS_SCHEMA,
        "arm": "A1.11-Reasoner",
        "selection": selection,
        "train_spec": asdict(spec),
        "reasoner_config": asdict(model.config),
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


def load_a111_reasoner_checkpoint(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> A111OracleRoleReasoner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != REASONER_CHECKPOINT_SCHEMA:
        raise ValueError("not an A1.11 Reasoner checkpoint")
    model = A111OracleRoleReasoner(A111ReasonerConfig(**checkpoint["reasoner_config"])).to(device)
    model.reasoner.load_state_dict(checkpoint["reasoner"])
    model.a111_reasoner_seed = int(checkpoint["train_spec"]["reasoner_seed"])
    model.a111_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a111_data_manifest_sha256 = checkpoint["data_manifest_sha256"]
    model.eval()
    return model


def _reasoner_metric_gate(
    results: dict[str, Any], split: str, lengths: Sequence[int], metric: str, threshold: float
) -> bool:
    return all(results[split]["by_length"][str(length)][metric] >= threshold for length in lengths)


@torch.no_grad()
def evaluate_a111_reasoner_formal(
    checkpoint: Path,
    data_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    model = load_a111_reasoner_checkpoint(checkpoint, device)
    results = {
        split: evaluate_a111_reasoner_by_length(
            model, A111RecordSplit(load_a18_records(data_dir, split)), device, batch_size=batch_size
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
            gates[f"{prefix}_{metric}_gate"] = _reasoner_metric_gate(
                results, split, lengths, metric, threshold
            )
    gates.update(
        {
            "causal_trajectory_at_least_0_95": results["causal_core"]["aggregate"]["trajectory_full_exact"] >= 0.95,
            "causal_final_state_at_least_0_95": results["causal_core"]["aggregate"]["final_state_full_exact"] >= 0.95,
            "causal_answer_at_least_0_95": results["causal_core"]["aggregate"]["final_answer_accuracy"] >= 0.95,
            "all_state_token_at_least_0_995": all(
                results[split]["aggregate"]["state_token_accuracy"] >= 0.995 for split in FORMAL_SPLITS
            ),
            "architecture_integrity": model.integrity_report()["passed"],
            "data_manifest_hash_matches": file_sha256(data_dir / "manifest.json") == model.a111_data_manifest_sha256,
        }
    )
    return {
        "schema_version": REASONER_FORMAL_SCHEMA,
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "reasoner_seed": model.a111_reasoner_seed,
        "data_seed": model.a111_data_seed,
        "splits": results,
        "gates": gates,
        "diagnostic_only": {"ood_length": DIAGNOSTIC_OOD_LENGTH},
        "passed": all(gates.values()),
    }
