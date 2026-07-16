from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_7_data import REGISTER_NAMES, VALUE_LABELS
from .a1_7_train import (
    CLOSURE_WEIGHT,
    _gradient_norm,
    _last_active_state_logits,
    _last_state_targets,
    _masked_mean,
    compute_a17_loss,
    encode_a17_records,
    prototype_statistics,
)
from .a1_8_data import IN_RANGE_LENGTHS, OOD_LENGTHS, SHORT_LENGTHS, TRAIN_LENGTHS
from .a1_8_train import load_a18_checkpoint
from .a1_9_cache import (
    A19CachedSplit,
    CACHE_MANIFEST_SCHEMA,
    collate_a19_items,
    file_sha256,
    sample_length_balanced_indices,
)
from .a1_9_model import A19BoundaryConfig, A19FrozenQwenBoundary, module_state_sha256


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.9.frozen-boundary.checkpoint.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.9.frozen-boundary.results.v1"
FORMAL_SCHEMA = "yggdrasil.v2-a1.9.frozen-boundary.formal-eval.v1"
MAPPING_CE_WEIGHT = 1.0
PROTOTYPE_REGRESSION_WEIGHT = 1.0
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


@dataclass(frozen=True)
class A19TrainSpec:
    adapter_seed: int
    data_seed: int
    core_seed: int
    steps: int = 3000
    batch_size: int = 128
    learning_rate: float = 3e-4
    validation_interval: int = 200
    early_stop_patience: int = 10
    closure_weight: float = CLOSURE_WEIGHT
    validation_trajectory_target: float = 0.995
    validation_mapping_target: float = 0.995


def _prototype_regression(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.smooth_l1_loss(predicted, target.detach(), reduction="none").mean(dim=-1)


def compute_a19_loss(
    model: A19FrozenQwenBoundary,
    output: dict[str, Any],
    labels: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    core_loss, core_metrics = compute_a17_loss(
        model.core,
        output,
        labels,
        closure_weight=CLOSURE_WEIGHT,
    )
    mask = labels["operation_mask"]
    value_ce = nn.functional.cross_entropy(
        output["value_mapping_logits"].reshape(-1, len(VALUE_LABELS)),
        labels["start_values"].reshape(-1),
    )
    family_ce = nn.functional.cross_entropy(
        output["family_mapping_logits"].reshape(-1, model.core.config.family_classes),
        labels["operation_family"].reshape(-1),
        reduction="none",
    ).reshape(mask.shape)
    source_ce = nn.functional.cross_entropy(
        output["source_mapping_logits"].reshape(-1, len(REGISTER_NAMES)),
        labels["operation_source"].reshape(-1),
        reduction="none",
    ).reshape(mask.shape)
    target_ce = nn.functional.cross_entropy(
        output["target_mapping_logits"].reshape(-1, len(REGISTER_NAMES)),
        labels["operation_target"].reshape(-1),
        reduction="none",
    ).reshape(mask.shape)
    query_ce = nn.functional.cross_entropy(output["query_mapping_logits"], labels["query_register"])
    mapping_ce = torch.stack(
        [
            value_ce,
            _masked_mean(family_ce, mask),
            _masked_mean(source_ce, mask),
            _masked_mean(target_ce, mask),
            query_ce,
        ]
    ).mean()
    value_target = model.core.canonical_value_prototypes[labels["start_values"]]
    family_target = model.core.family_embedding(labels["operation_family"])
    source_target = model.core.register_embedding(labels["operation_source"])
    target_target = model.core.register_embedding(labels["operation_target"])
    query_target = model.core.register_embedding(labels["query_register"])
    regression = torch.stack(
        [
            _prototype_regression(output["mapped_start_values"], value_target).mean(),
            _masked_mean(_prototype_regression(output["mapped_family"], family_target), mask),
            _masked_mean(_prototype_regression(output["mapped_source"], source_target), mask),
            _masked_mean(_prototype_regression(output["mapped_target"], target_target), mask),
            _prototype_regression(output["mapped_query"], query_target).mean(),
        ]
    ).mean()
    total = core_loss + MAPPING_CE_WEIGHT * mapping_ce + PROTOTYPE_REGRESSION_WEIGHT * regression
    return total, {
        **{f"core_{key}": value for key, value in core_metrics.items()},
        "loss": float(total.detach()),
        "mapping_ce": float(mapping_ce.detach()),
        "prototype_regression": float(regression.detach()),
        "mapping_ce_weight": MAPPING_CE_WEIGHT,
        "prototype_regression_weight": PROTOTYPE_REGRESSION_WEIGHT,
    }


def _mapping_predictions(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "value": output["value_mapping_logits"].argmax(dim=-1),
        "family": output["family_mapping_logits"].argmax(dim=-1),
        "source": output["source_mapping_logits"].argmax(dim=-1),
        "target": output["target_mapping_logits"].argmax(dim=-1),
        "query": output["query_mapping_logits"].argmax(dim=-1),
    }


@torch.no_grad()
def evaluate_a19_items(
    model: A19FrozenQwenBoundary,
    items: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 128,
    old_oracle: bool = False,
) -> dict[str, Any]:
    if not items:
        raise ValueError("A1.9 evaluation requires at least one item")
    model.eval()
    trajectory_full = final_state_full = answer_correct = 0
    state_tokens = state_total = source_pointer_correct = target_pointer_correct = pointer_total = 0
    mapping_correct = defaultdict(int)
    mapping_total = defaultdict(int)
    identity_max_abs_diff = 0.0
    identity_predictions_equal = True
    losses: list[float] = []
    closure_losses: list[float] = []
    margins: list[float] = []
    old_oracle_full = old_oracle_examples = 0
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        inputs, records = collate_a19_items(batch_items, device)
        labels = encode_a17_records(records, device)
        output = model(**inputs)
        loss, loss_metrics = compute_a19_loss(model, output, labels)
        losses.append(float(loss))
        closure_losses.append(float(loss_metrics["core_closure_loss"]))
        valid = labels["operation_mask"]
        state_prediction = output["state_logits"].argmax(dim=-1)
        state_equal = state_prediction == labels["state_targets"]
        state_tokens += int((state_equal & valid.unsqueeze(-1)).sum())
        state_total += int(valid.sum()) * len(REGISTER_NAMES)
        trajectory_full += int((state_equal | ~valid.unsqueeze(-1)).all(dim=(-1, -2)).sum())
        last_logits = _last_active_state_logits(model.core, output, labels)
        last_targets = _last_state_targets(labels)
        final_state_full += int((last_logits.argmax(dim=-1) == last_targets).all(dim=-1).sum())
        query_index = labels["query_register"].view(-1, 1, 1).expand(-1, 1, len(VALUE_LABELS))
        queried_last_logits = last_logits.gather(1, query_index).squeeze(1)
        difference = float((output["answer_logits"] - queried_last_logits).abs().max())
        identity_max_abs_diff = max(identity_max_abs_diff, difference)
        identity_predictions_equal = identity_predictions_equal and bool(
            torch.equal(output["answer_logits"].argmax(dim=-1), queried_last_logits.argmax(dim=-1))
        )
        answer_correct += int((output["answer_logits"].argmax(dim=-1) == labels["answer_targets"]).sum())
        source_pointer_correct += int(((output["source_pointer_logits"].argmax(dim=-1) == labels["operation_source"]) & valid).sum())
        target_pointer_correct += int(((output["target_pointer_logits"].argmax(dim=-1) == labels["operation_target"]) & valid).sum())
        pointer_total += int(valid.sum())
        mapping = _mapping_predictions(output)
        mapping_correct["value"] += int((mapping["value"] == labels["start_values"]).sum())
        mapping_total["value"] += int(labels["start_values"].numel())
        mapping_correct["query"] += int((mapping["query"] == labels["query_register"]).sum())
        mapping_total["query"] += len(records)
        for name, label_name in (("family", "operation_family"), ("source", "operation_source"), ("target", "operation_target")):
            mapping_correct[name] += int(((mapping[name] == labels[label_name]) & valid).sum())
            mapping_total[name] += int(valid.sum())
        predicted = nn.functional.normalize(output["trajectory"], dim=-1)
        prototypes = nn.functional.normalize(model.core.canonical_value_prototypes.detach(), dim=-1)
        similarities = torch.einsum("btrd,vd->btrv", predicted, prototypes)
        correct_similarity = similarities.gather(-1, labels["state_targets"].unsqueeze(-1)).squeeze(-1)
        incorrect = similarities.masked_fill(
            nn.functional.one_hot(labels["state_targets"], num_classes=len(VALUE_LABELS)).bool(),
            -float("inf"),
        ).max(dim=-1).values
        margin = correct_similarity - incorrect
        margins.extend(margin[valid.unsqueeze(-1).expand_as(margin)].detach().cpu().tolist())
        if old_oracle:
            eligible = [index for index, item in enumerate(batch_items) if item.get("old_record") is not None]
            if eligible:
                old_records = [batch_items[index]["old_record"] for index in eligible]
                old_labels = encode_a17_records(old_records, device)
                if old_labels["state_targets"].shape[1] == state_prediction.shape[1]:
                    old_equal = state_prediction[eligible] == old_labels["state_targets"]
                    old_valid = old_labels["operation_mask"]
                    old_oracle_full += int((old_equal | ~old_valid.unsqueeze(-1)).all(dim=(-1, -2)).sum())
                    old_oracle_examples += len(eligible)
    mapping_accuracy = {
        name: mapping_correct[name] / max(1, mapping_total[name])
        for name in ("value", "family", "source", "target", "query")
    }
    margins.sort()
    p05_index = min(len(margins) - 1, max(0, int(0.05 * len(margins)))) if margins else 0
    identity_gate = identity_max_abs_diff <= 1e-6 and identity_predictions_equal
    result = {
        "examples": len(items),
        "loss": sum(losses) / max(1, len(losses)),
        "closure_loss": sum(closure_losses) / max(1, len(closure_losses)),
        "trajectory_full_exact": trajectory_full / len(items),
        "final_state_full_exact": final_state_full / len(items),
        "final_query_token_accuracy": answer_correct / len(items),
        "state_token_accuracy": state_tokens / max(1, state_total),
        "source_pointer_accuracy": source_pointer_correct / max(1, pointer_total),
        "target_pointer_accuracy": target_pointer_correct / max(1, pointer_total),
        "mapping_accuracy": {**mapping_accuracy, "minimum": min(mapping_accuracy.values())},
        "answer_state_logits_identity": {
            "max_abs_diff": identity_max_abs_diff,
            "prediction_equal": identity_predictions_equal,
            "gate": identity_gate,
            "method": "boundary query argmax selects frozen final state logits; no independent answer head",
        },
        "prototype_margin": {
            "mean_correct_minus_nearest_wrong_cosine": sum(margins) / max(1, len(margins)),
            "p05_correct_minus_nearest_wrong_cosine": margins[p05_index] if margins else None,
            "samples": len(margins),
        },
        "prototype_statistics": prototype_statistics(model.core),
    }
    if old_oracle:
        result["old_oracle_trajectory_full_exact"] = old_oracle_full / max(1, old_oracle_examples)
        result["old_oracle_examples"] = old_oracle_examples
    return result


@torch.no_grad()
def evaluate_a19_by_length(
    model: A19FrozenQwenBoundary,
    dataset: A19CachedSplit,
    device: str | torch.device,
    *,
    batch_size: int = 128,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    selected = list(range(len(dataset))) if indices is None else list(indices)
    items = dataset.items(selected)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[int(item["record"]["program_length"])].append(item)
    return {
        "aggregate": evaluate_a19_items(model, items, device, batch_size=batch_size),
        "by_length": {
            str(length): evaluate_a19_items(model, rows, device, batch_size=batch_size)
            for length, rows in sorted(groups.items())
        },
    }


def _validation_score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    rows = list(metrics["by_length"].values())
    minimum_trajectory = min(row["trajectory_full_exact"] for row in rows)
    minimum_mapping = min(row["mapping_accuracy"]["minimum"] for row in rows)
    minimum_source_pointer = min(row["source_pointer_accuracy"] for row in rows)
    minimum_target_pointer = min(row["target_pointer_accuracy"] for row in rows)
    aggregate_trajectory = metrics["aggregate"]["trajectory_full_exact"]
    score = minimum_trajectory + 0.10 * aggregate_trajectory + 0.05 * minimum_mapping + 0.025 * minimum_source_pointer + 0.025 * minimum_target_pointer
    return score, {
        "minimum_per_length_trajectory": minimum_trajectory,
        "aggregate_trajectory": aggregate_trajectory,
        "minimum_per_length_mapping": minimum_mapping,
        "minimum_source_pointer": minimum_source_pointer,
        "minimum_target_pointer": minimum_target_pointer,
    }


def _checkpoint(
    model: A19FrozenQwenBoundary,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A19TrainSpec,
    core_checkpoint: Path,
    core_checkpoint_sha256: str,
    core_state_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "boundary_config": asdict(model.config),
        "train_spec": asdict(spec),
        "adapter_seed": spec.adapter_seed,
        "data_seed": spec.data_seed,
        "core_seed": spec.core_seed,
        "core_checkpoint": str(core_checkpoint.resolve()),
        "core_checkpoint_sha256": core_checkpoint_sha256,
        "core_state_sha256": core_state_sha256,
        "core_frozen": True,
        "qwen_frozen": True,
        "oracle_role_span_segmentation": True,
        "training_lengths": list(TRAIN_LENGTHS),
        "sampling": "uniform-length-balanced-within-every-batch",
        "loss_weights": {
            "core_closure": CLOSURE_WEIGHT,
            "mapping_ce": MAPPING_CE_WEIGHT,
            "prototype_regression": PROTOTYPE_REGRESSION_WEIGHT,
        },
        "boundary": model.boundary_state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def _balanced_subset_indices(dataset: A19CachedSplit, maximum: int = 2048) -> list[int]:
    per_length = max(1, maximum // len(dataset.indices_by_length))
    return [index for length in sorted(dataset.indices_by_length) for index in dataset.indices_by_length[length][:per_length]]


def train_a19_boundary(
    cache_dir: Path,
    data_dir: Path,
    core_checkpoint: Path,
    output_dir: Path,
    *,
    spec: A19TrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.9 training requested CUDA but CUDA is unavailable")
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to resume or mix A1.9 checkpoints in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ValueError("A1.9 training requires an A1.9 cache manifest")
    if spec.closure_weight != CLOSURE_WEIGHT:
        raise ValueError(f"A1.9 closure weight is frozen at {CLOSURE_WEIGHT}")
    selection = manifest["selection"]
    if overfit_mode != (selection == "length-balanced-overfit32-fit-only"):
        raise ValueError("A1.9 overfit mode must match the cache selection")
    train_dataset = A19CachedSplit(cache_dir, data_dir, "train")
    validation_dataset = A19CachedSplit(cache_dir, data_dir, "validation")
    if set(train_dataset.indices_by_length) != set(TRAIN_LENGTHS) or set(validation_dataset.indices_by_length) != set(TRAIN_LENGTHS):
        raise ValueError("A1.9 train/validation must cover every length T1-T16")
    random.seed(spec.adapter_seed)
    torch.manual_seed(spec.adapter_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.adapter_seed)
    core = load_a18_checkpoint(core_checkpoint, device)
    if int(core.a18_data_seed) != spec.data_seed or int(core.a18_model_seed) != spec.core_seed:
        raise ValueError("A1.9 train spec does not match the frozen A1.8 checkpoint seeds")
    core_checkpoint_hash = file_sha256(core_checkpoint)
    core_state_hash = module_state_sha256(core)
    model = A19FrozenQwenBoundary(
        A19BoundaryConfig(source_width=train_dataset.hidden_width, latent_width=core.config.latent_width),
        core,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=spec.learning_rate)
    rng = random.Random(spec.adapter_seed)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    patience = ready_streak = 0
    processed_transitions = 0
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        indices = sample_length_balanced_indices(train_dataset, spec.batch_size, rng)
        items = train_dataset.items(indices)
        inputs, records = collate_a19_items(items, device)
        labels = encode_a17_records(records, device)
        output = model(**inputs)
        loss, metrics = compute_a19_loss(model, output, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in model.core.parameters()):
            raise RuntimeError("frozen A1.8 core received gradients during A1.9")
        metrics["gradient_norm"] = _gradient_norm(model)
        nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        sampled_lengths: dict[int, int] = defaultdict(int)
        for record in records:
            sampled_lengths[int(record["program_length"])] += 1
        processed_transitions += sum(length * count for length, count in sampled_lengths.items())
        metrics["step"] = step
        metrics["sampled_length_counts"] = dict(sorted(sampled_lengths.items()))
        if step % spec.validation_interval == 0 or step == spec.steps:
            validation = evaluate_a19_by_length(model, validation_dataset, device, batch_size=spec.batch_size)
            score, components = _validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = components
            checkpoint = _checkpoint(
                model,
                optimizer,
                step,
                spec,
                core_checkpoint,
                core_checkpoint_hash,
                core_state_hash,
            )
            torch.save(checkpoint, output_dir / "latest.pt")
            if score > best_score:
                best_score = score
                best_step = step
                best_validation = validation
                torch.save(checkpoint, output_dir / "best.pt")
                patience = 0
            else:
                patience += 1
            trajectory_target = 0.999 if overfit_mode else spec.validation_trajectory_target
            mapping_target = 0.999 if overfit_mode else spec.validation_mapping_target
            ready = (
                components["minimum_per_length_trajectory"] >= trajectory_target
                and components["minimum_per_length_mapping"] >= mapping_target
                and components["minimum_source_pointer"] >= 0.999
                and components["minimum_target_pointer"] >= 0.999
                and validation["aggregate"]["answer_state_logits_identity"]["gate"]
            )
            ready_streak = ready_streak + 1 if ready else 0
            history.append(metrics)
            if ready_streak >= 2:
                break
            if not overfit_mode and patience >= spec.early_stop_patience:
                break
        else:
            history.append(metrics)
        (output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "steps_requested": spec.steps,
                    "best_score": best_score,
                    "best_step": best_step,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not (output_dir / "best.pt").exists():
        raise RuntimeError("A1.9 training completed without a best checkpoint")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_boundary_state_dict(best_checkpoint["boundary"])
    final_core_hash = module_state_sha256(model.core)
    fit_indices = _balanced_subset_indices(train_dataset)
    fit_diagnostic = evaluate_a19_by_length(model, train_dataset, device, batch_size=spec.batch_size, indices=fit_indices)
    validation = evaluate_a19_by_length(model, validation_dataset, device, batch_size=spec.batch_size)
    overfit_gates = {
        "all_length_trajectory_exact": all(row["trajectory_full_exact"] == 1.0 for row in validation["by_length"].values()),
        "all_mapping_exact": all(value == 1.0 for key, value in validation["aggregate"]["mapping_accuracy"].items() if key != "minimum"),
        "source_pointer_exact": validation["aggregate"]["source_pointer_accuracy"] == 1.0,
        "target_pointer_exact": validation["aggregate"]["target_pointer_accuracy"] == 1.0,
        "answer_exact": validation["aggregate"]["final_query_token_accuracy"] == 1.0,
        "answer_state_identity": validation["aggregate"]["answer_state_logits_identity"]["gate"],
        "core_hash_unchanged": final_core_hash == core_state_hash,
    }
    result = {
        "schema_version": RESULTS_SCHEMA,
        "stage": "A1.9 frozen Qwen hidden boundary",
        "selection": selection,
        "boundary_config": asdict(model.config),
        "train_spec": asdict(spec),
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": history[-1]["step"],
            "processed_examples": history[-1]["step"] * spec.batch_size,
            "processed_transitions": processed_transitions,
            "seconds": time.perf_counter() - started,
            "sampling": "uniform-length-balanced-within-every-batch",
            "fresh_initialization": True,
            "resume_checkpoint": None,
            "overfit_mode": overfit_mode,
        },
        "best_score": best_score,
        "best_checkpoint_step": best_step,
        "best_validation_at_save": best_validation,
        "fit_diagnostic_examples": len(fit_indices),
        "fit_diagnostic": fit_diagnostic,
        "validation": validation,
        "integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(qwen_parameters=int(manifest["qwen_parameters"])),
        "core_checkpoint_sha256": core_checkpoint_hash,
        "core_state_sha256_before": core_state_hash,
        "core_state_sha256_after": final_core_hash,
        "core_hash_unchanged": final_core_hash == core_state_hash,
        "overfit_gates": overfit_gates if overfit_mode else None,
        "overfit_passed": all(overfit_gates.values()) if overfit_mode else None,
        "artifacts": {
            "latest": str(output_dir / "latest.pt"),
            "best": str(output_dir / "best.pt"),
            "history": str(output_dir / "history.json"),
            "progress": str(output_dir / "progress.json"),
        },
    }
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def load_a19_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device = "cpu",
    *,
    core_checkpoint_override: Path | None = None,
) -> A19FrozenQwenBoundary:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not an A1.9 frozen boundary checkpoint: {checkpoint_path}")
    core_path = core_checkpoint_override or Path(checkpoint["core_checkpoint"])
    if file_sha256(core_path) != checkpoint["core_checkpoint_sha256"]:
        raise ValueError("A1.9 frozen A1.8 core checkpoint hash mismatch")
    core = load_a18_checkpoint(core_path, device)
    if module_state_sha256(core) != checkpoint["core_state_sha256"]:
        raise ValueError("A1.9 frozen A1.8 core state hash mismatch")
    model = A19FrozenQwenBoundary(A19BoundaryConfig(**checkpoint["boundary_config"]), core).to(device)
    model.load_boundary_state_dict(checkpoint["boundary"])
    model.a19_adapter_seed = int(checkpoint["adapter_seed"])
    model.a19_data_seed = int(checkpoint["data_seed"])
    model.a19_core_seed = int(checkpoint["core_seed"])
    model.a19_core_checkpoint = str(core_path)
    model.a19_core_checkpoint_sha256 = checkpoint["core_checkpoint_sha256"]
    model.a19_core_state_sha256 = checkpoint["core_state_sha256"]
    model.eval()
    return model


def _trajectory_gate(results: dict[str, Any], split: str, lengths: Sequence[int], threshold: float) -> bool:
    return all(results[split]["by_length"][str(length)]["trajectory_full_exact"] >= threshold for length in lengths)


@torch.no_grad()
def evaluate_a19_formal(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    device: str = "cuda",
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model = load_a19_checkpoint(checkpoint, device)
    results: dict[str, Any] = {}
    for split in FORMAL_SPLITS:
        dataset = A19CachedSplit(cache_dir, data_dir, split)
        results[split] = evaluate_a19_by_length(model, dataset, device, batch_size=batch_size)
        del dataset
    mapping_splits = FORMAL_SPLITS
    gates = {
        "short_t1_t6_each_at_least_0_95": _trajectory_gate(results, "short_regression", SHORT_LENGTHS, 0.95),
        "supported_t8_t12_t16_each_at_least_0_95": _trajectory_gate(results, "supported_in_range", IN_RANGE_LENGTHS, 0.95),
        "relation_t8_t12_t16_each_at_least_0_95": _trajectory_gate(results, "relation_in_range", IN_RANGE_LENGTHS, 0.95),
        "supported_t20_t24_each_at_least_0_90": _trajectory_gate(results, "supported_ood", HARD_OOD_LENGTHS, 0.90),
        "relation_t20_t24_each_at_least_0_90": _trajectory_gate(results, "relation_ood", HARD_OOD_LENGTHS, 0.90),
        "causal_free_trajectory_at_least_0_95": results["causal_core"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "all_boundary_mapping_at_least_0_995": all(results[split]["aggregate"]["mapping_accuracy"]["minimum"] >= 0.995 for split in mapping_splits),
        "all_pointer_accuracy_at_least_0_995": all(
            results[split]["aggregate"]["source_pointer_accuracy"] >= 0.995
            and results[split]["aggregate"]["target_pointer_accuracy"] >= 0.995
            for split in mapping_splits
        ),
        "answer_state_identity": all(results[split]["aggregate"]["answer_state_logits_identity"]["gate"] for split in mapping_splits),
        "prototype_non_collapse": all(results[split]["aggregate"]["prototype_statistics"]["non_collapse_gate"] for split in mapping_splits),
        "architecture_integrity": model.integrity_report()["passed"],
        "core_state_hash_unchanged": module_state_sha256(model.core) == model.a19_core_state_sha256,
    }
    return {
        "schema_version": FORMAL_SCHEMA,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "adapter_seed": model.a19_adapter_seed,
        "data_seed": model.a19_data_seed,
        "core_seed": model.a19_core_seed,
        "integrity": model.integrity_report(),
        "core_state_sha256": module_state_sha256(model.core),
        "splits": results,
        "diagnostic_only": {
            "ood_length": DIAGNOSTIC_OOD_LENGTH,
            "reason": "T32 is reported but excluded from the preregistered A1.9 trajectory Gate",
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _timed_cuda(callable_: Any, repeats: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def benchmark_a19_cached_costs(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    device: str = "cuda",
    *,
    batch_size: int = 128,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A1.9 cost benchmark requires CUDA")
    model = load_a19_checkpoint(checkpoint, device)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    parameter_report = model.parameter_report(qwen_parameters=int(manifest["qwen_parameters"]))
    inference: dict[str, Any] = {}
    length_sources = {8: "supported_in_range", 12: "supported_in_range", 16: "supported_in_range", 20: "supported_ood", 24: "supported_ood", 32: "supported_ood"}
    for length, split in length_sources.items():
        dataset = A19CachedSplit(cache_dir, data_dir, split)
        indices = dataset.indices_by_length[length][:batch_size]
        inputs, _ = collate_a19_items(dataset.items(indices), device)

        def inference_step() -> None:
            with torch.no_grad():
                model(**inputs)

        for _ in range(warmup):
            inference_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        seconds = _timed_cuda(inference_step, repeats)
        inference[str(length)] = {
            "batch_size": len(indices),
            "latent_transitions_per_example": length,
            "mean_batch_seconds": seconds,
            "mean_example_seconds": seconds / len(indices),
            "examples_per_second": len(indices) / seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        del dataset, inputs
    train_dataset = A19CachedSplit(cache_dir, data_dir, "train")
    indices = sample_length_balanced_indices(train_dataset, batch_size, random.Random(20260715))
    inputs, records = collate_a19_items(train_dataset.items(indices), device)
    labels = encode_a17_records(records, device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=3e-4)

    def training_step() -> None:
        output = model(**inputs)
        loss, _ = compute_a19_loss(model, output, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
        optimizer.step()

    for _ in range(warmup):
        training_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    training_seconds = _timed_cuda(training_step, repeats)
    transitions = sum(int(record["program_length"]) for record in records)
    return {
        "schema_version": "yggdrasil.v2-a1.9.cached-boundary-cost.v1",
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "device": {"type": device, "name": torch.cuda.get_device_name(torch.cuda.current_device()), "torch_version": torch.__version__},
        "method": {
            "warmup_repeats": warmup,
            "timed_repeats": repeats,
            "cuda_synchronized": True,
            "boundary_core_latency_uses_precomputed_role_hidden": True,
            "qwen_encoding_cost_reported_from_cache_manifest": True,
            "cache_manifest_seconds_exclude_model_and_tokenizer_load": True,
            "not_a_text_cot_pareto_measurement": True,
        },
        "parameters": parameter_report,
        "qwen_role_cache_generation": {
            "role_encoding_seconds_excluding_model_and_tokenizer_load": manifest["seconds"],
            "examples": sum(row["examples"] for row in manifest["splits"].values()),
            "peak_allocated_bytes": manifest["peak_allocated_bytes"],
            "inference_batch_size": manifest["inference_batch_size"],
        },
        "training": {
            "batch_size": batch_size,
            "latent_transitions_per_batch": transitions,
            "mean_step_seconds": training_seconds,
            "examples_per_second": batch_size / training_seconds,
            "transitions_per_second": transitions / training_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "inference": inference,
        "kv_cache": {"used": False, "reason": "Qwen cache extraction and structured core rollout both use use_cache=False"},
    }
