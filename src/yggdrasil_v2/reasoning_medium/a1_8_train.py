from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_7_core import A17CoreConfig, A17RelationAddressedCore
from .a1_7_data import REGISTER_NAMES, VALUE_LABELS
from .a1_7_train import (
    CLOSURE_WEIGHT,
    _gradient_norm,
    _model_inputs,
    compute_a17_loss,
    encode_a17_records,
    evaluate_a17_core,
    run_a17_interventions,
)
from .a1_8_data import IN_RANGE_LENGTHS, OOD_LENGTHS, SHORT_LENGTHS, TRAIN_LENGTHS, load_a18_records


FORMAL_SPLITS = (
    "short_regression",
    "supported_in_range",
    "relation_in_range",
    "supported_ood",
    "relation_ood",
    "causal_core",
)
HARD_OOD_LENGTHS = (20, 24)
DIAGNOSTIC_OOD_LENGTH = 32


@dataclass(frozen=True)
class A18TrainSpec:
    model_seed: int
    data_seed: int
    steps: int = 8000
    batch_size: int = 128
    learning_rate: float = 3e-4
    validation_interval: int = 200
    early_stop_patience: int = 15
    closure_weight: float = CLOSURE_WEIGHT
    validation_trajectory_target: float = 0.995


def _records_by_length(records: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["program_length"])].append(record)
    return dict(sorted(grouped.items()))


def sample_length_balanced_batch(
    grouped: dict[int, list[dict[str, Any]]],
    batch_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], Counter[int]]:
    if not grouped or any(not rows for rows in grouped.values()):
        raise ValueError("length-balanced sampling requires at least one record for every configured length")
    lengths = sorted(grouped)
    sampled_lengths = [lengths[index % len(lengths)] for index in range(batch_size)]
    rng.shuffle(sampled_lengths)
    batch = [rng.choice(grouped[length]) for length in sampled_lengths]
    return batch, Counter(sampled_lengths)


def _balanced_diagnostic_subset(records: Sequence[dict[str, Any]], maximum: int = 2048) -> list[dict[str, Any]]:
    grouped = _records_by_length(records)
    per_length = max(1, maximum // len(grouped))
    return [record for rows in grouped.values() for record in rows[:per_length]]


@torch.no_grad()
def evaluate_a18_by_length(
    model: A17RelationAddressedCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 128,
    closure_weight: float = CLOSURE_WEIGHT,
) -> dict[str, Any]:
    grouped = _records_by_length(records)
    return {
        "aggregate": evaluate_a17_core(
            model,
            records,
            device,
            batch_size=batch_size,
            closure_weight=closure_weight,
        ),
        "by_length": {
            str(length): evaluate_a17_core(
                model,
                rows,
                device,
                batch_size=batch_size,
                closure_weight=closure_weight,
            )
            for length, rows in grouped.items()
        },
    }


def _validation_score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    per_length = list(metrics["by_length"].values())
    minimum_trajectory = min(row["trajectory_full_exact"] for row in per_length)
    minimum_source_pointer = min(row["source_pointer_accuracy"] for row in per_length)
    minimum_target_pointer = min(row["target_pointer_accuracy"] for row in per_length)
    aggregate_trajectory = metrics["aggregate"]["trajectory_full_exact"]
    score = minimum_trajectory + 0.10 * aggregate_trajectory + 0.05 * minimum_source_pointer + 0.05 * minimum_target_pointer
    return score, {
        "minimum_per_length_trajectory": minimum_trajectory,
        "aggregate_trajectory": aggregate_trajectory,
        "minimum_source_pointer": minimum_source_pointer,
        "minimum_target_pointer": minimum_target_pointer,
    }


def _checkpoint(
    model: A17RelationAddressedCore,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: A17CoreConfig,
    spec: A18TrainSpec,
) -> dict[str, Any]:
    return {
        "schema_version": "yggdrasil.v2-a1.8.random-depth.checkpoint.v1",
        "step": step,
        "config": asdict(config),
        "train_spec": asdict(spec),
        "model_seed": spec.model_seed,
        "data_seed": spec.data_seed,
        "training_lengths": list(TRAIN_LENGTHS),
        "sampling": "uniform-length-balanced-within-every-batch",
        "fresh_initialization": True,
        "closure_weight": spec.closure_weight,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def train_a18_random_depth(
    train_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    spec: A18TrainSpec,
    config: A17CoreConfig | None = None,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if spec.closure_weight < 0:
        raise ValueError("closure weight must be non-negative")
    if spec.batch_size < len(TRAIN_LENGTHS):
        raise ValueError("batch size must be at least 16 for within-batch length balancing")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to resume or mix checkpoints in existing A1.8 output directory: {output_dir}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA for A1.8 but CUDA is unavailable")
    train_grouped = _records_by_length(train_records)
    validation_grouped = _records_by_length(validation_records)
    if set(train_grouped) != set(TRAIN_LENGTHS) or set(validation_grouped) != set(TRAIN_LENGTHS):
        raise ValueError("A1.8 train and validation must each cover every length from 1 through 16")
    random.seed(spec.model_seed)
    torch.manual_seed(spec.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.model_seed)
    config = config or A17CoreConfig()
    if config.address_content_mixing_ablation:
        raise ValueError("A1.8 formal training uses the frozen content-only A1.7 core; structural ablations are out of scope")
    model = A17RelationAddressedCore(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    rng = random.Random(spec.model_seed)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    patience = ready_streak = 0
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        batch_records, batch_lengths = sample_length_balanced_batch(train_grouped, spec.batch_size, rng)
        batch = encode_a17_records(batch_records, device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        loss, metrics = compute_a17_loss(model, output, batch, closure_weight=spec.closure_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics["gradient_norm"] = _gradient_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics["step"] = step
        metrics["sampled_length_counts"] = dict(sorted(batch_lengths.items()))
        if step % spec.validation_interval == 0 or step == spec.steps:
            validation = evaluate_a18_by_length(
                model,
                validation_records,
                device,
                batch_size=spec.batch_size,
                closure_weight=spec.closure_weight,
            )
            score, score_components = _validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = score_components
            checkpoint = _checkpoint(model, optimizer, step, config, spec)
            torch.save(checkpoint, output_dir / "latest.pt")
            if score > best_score:
                best_score = score
                best_step = step
                best_validation = validation
                torch.save(checkpoint, output_dir / "best.pt")
                patience = 0
            else:
                patience += 1
            threshold = 0.999 if overfit_mode else spec.validation_trajectory_target
            ready = (
                score_components["minimum_per_length_trajectory"] >= threshold
                and score_components["minimum_source_pointer"] >= 0.999
                and score_components["minimum_target_pointer"] >= 0.999
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
        raise RuntimeError("A1.8 training completed without a best checkpoint")
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    fit_subset = _balanced_diagnostic_subset(train_records)
    result = {
        "schema_version": "yggdrasil.v2-a1.8.random-depth.results.v1",
        "stage": "A1.8 random-depth long-horizon continuous core",
        "architecture_change_from_a1_7": False,
        "training_distribution_change_only": True,
        "config": asdict(config),
        "train_spec": asdict(spec),
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": history[-1]["step"],
            "processed_examples": history[-1]["step"] * spec.batch_size,
            "processed_transitions": sum(
                sum(int(length) * int(count) for length, count in row["sampled_length_counts"].items())
                for row in history
            ),
            "seconds": time.perf_counter() - started,
            "fresh_initialization": True,
            "resume_checkpoint": None,
            "overfit_mode": overfit_mode,
            "sampling": "uniform-length-balanced-within-every-batch",
            "training_lengths": list(TRAIN_LENGTHS),
        },
        "best_score": best_score,
        "best_checkpoint_step": best_step,
        "fit_diagnostic_examples": len(fit_subset),
        "fit_diagnostic": evaluate_a18_by_length(
            model,
            fit_subset,
            device,
            batch_size=spec.batch_size,
            closure_weight=spec.closure_weight,
        ),
        "validation": evaluate_a18_by_length(
            model,
            validation_records,
            device,
            batch_size=spec.batch_size,
            closure_weight=spec.closure_weight,
        ),
        "best_validation_at_save": best_validation,
        "integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(),
        "artifacts": {
            "latest": str(output_dir / "latest.pt"),
            "best": str(output_dir / "best.pt"),
            "history": str(output_dir / "history.json"),
            "progress": str(output_dir / "progress.json"),
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def load_a18_checkpoint(path: Path, device: str | torch.device = "cpu") -> A17RelationAddressedCore:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != "yggdrasil.v2-a1.8.random-depth.checkpoint.v1":
        raise ValueError(f"not an A1.8 random-depth checkpoint: {path}")
    model = A17RelationAddressedCore(A17CoreConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.training_closure_weight = float(checkpoint["closure_weight"])
    model.a18_model_seed = int(checkpoint["model_seed"])
    model.a18_data_seed = int(checkpoint["data_seed"])
    model.eval()
    return model


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
    return ordered[index]


@torch.no_grad()
def evaluate_a18_stability(
    model: A17RelationAddressedCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int = 128,
    perturbation_epsilon: float = 1e-3,
    perturbation_seed: int = 20260715,
) -> dict[str, Any]:
    if perturbation_epsilon <= 0:
        raise ValueError("perturbation epsilon must be positive")
    model.eval()
    step_values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    step_full_correct: Counter[int] = Counter()
    step_token_correct: Counter[int] = Counter()
    step_active: Counter[int] = Counter()
    first_errors: Counter[str] = Counter()
    generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(perturbation_seed)
    for start in range(0, len(records), batch_size):
        batch = encode_a17_records(records[start : start + batch_size], device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        initial = output["initial_slots"]
        noise = torch.randn(initial.shape, generator=generator, device=initial.device, dtype=initial.dtype)
        noise = noise / noise.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-12)
        perturbed = model.rollout(
            initial + perturbation_epsilon * noise,
            batch["query_register"],
            model.family_embedding(batch["operation_family"]),
            model.register_embedding(batch["operation_source"]),
            model.register_embedding(batch["operation_target"]),
            batch["operation_mask"],
        )
        valid = batch["operation_mask"]
        state_pred = output["state_logits"].argmax(-1)
        state_equal = state_pred == batch["state_targets"]
        state_full = state_equal.all(dim=-1)
        latent_norm = output["trajectory"].norm(dim=-1).mean(dim=-1)
        prototypes = nn.functional.normalize(model.canonical_value_prototypes.detach(), dim=-1)
        normalized = nn.functional.normalize(output["trajectory"], dim=-1)
        similarities = torch.einsum("btrd,vd->btrv", normalized, prototypes)
        correct_cosine = similarities.gather(-1, batch["state_targets"].unsqueeze(-1)).squeeze(-1)
        nearest_wrong = similarities.masked_fill(
            nn.functional.one_hot(batch["state_targets"], num_classes=len(VALUE_LABELS)).bool(),
            -float("inf"),
        ).max(dim=-1).values
        margin = correct_cosine - nearest_wrong
        amplification = (
            (perturbed["trajectory"] - output["trajectory"]).square().mean(dim=(-1, -2)).sqrt()
            / perturbation_epsilon
        )
        for row in range(valid.shape[0]):
            length = int(valid[row].sum())
            first_error: int | None = None
            for step_index in range(length):
                step = step_index + 1
                step_active[step] += 1
                step_full_correct[step] += int(state_full[row, step_index])
                step_token_correct[step] += int(state_equal[row, step_index].sum())
                step_values[step]["latent_norm"].append(float(latent_norm[row, step_index]))
                step_values[step]["correct_cosine"].append(float(correct_cosine[row, step_index].mean()))
                step_values[step]["prototype_margin"].append(float(margin[row, step_index].mean()))
                step_values[step]["perturbation_amplification"].append(float(amplification[row, step_index]))
                if first_error is None and not bool(state_full[row, step_index]):
                    first_error = step
            first_errors[str(first_error) if first_error is not None else "none"] += 1
    per_step: dict[str, Any] = {}
    for step in sorted(step_active):
        values = step_values[step]
        amplifications = values["perturbation_amplification"]
        norms = values["latent_norm"]
        margins = values["prototype_margin"]
        cosines = values["correct_cosine"]
        per_step[str(step)] = {
            "active_examples": step_active[step],
            "state_full_exact": step_full_correct[step] / step_active[step],
            "state_token_accuracy": step_token_correct[step] / (step_active[step] * len(REGISTER_NAMES)),
            "latent_norm_mean": statistics.fmean(norms),
            "latent_norm_std": statistics.pstdev(norms) if len(norms) > 1 else 0.0,
            "correct_prototype_cosine_mean": statistics.fmean(cosines),
            "prototype_margin_mean": statistics.fmean(margins),
            "prototype_margin_p05": _percentile(margins, 0.05),
            "initial_perturbation_amplification_mean": statistics.fmean(amplifications),
            "initial_perturbation_amplification_p95": _percentile(amplifications, 0.95),
            "initial_perturbation_amplification_max": max(amplifications),
        }
    return {
        "schema_version": "yggdrasil.v2-a1.8.rollout-stability.v1",
        "examples": len(records),
        "perturbation": {
            "kind": "normalized continuous perturbation added to initial slots",
            "epsilon_rms": perturbation_epsilon,
            "seed": perturbation_seed,
        },
        "per_step": per_step,
        "first_error_position": dict(sorted(first_errors.items(), key=lambda item: (item[0] == "none", int(item[0]) if item[0] != "none" else math.inf))),
    }


def _trajectory_gate(
    split_results: dict[str, Any],
    split: str,
    lengths: Sequence[int],
    threshold: float,
) -> bool:
    return all(split_results[split]["by_length"][str(length)]["trajectory_full_exact"] >= threshold for length in lengths)


@torch.no_grad()
def evaluate_a18_formal(
    checkpoint: Path,
    data_dir: Path,
    device: str,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model = load_a18_checkpoint(checkpoint, device)
    closure_weight = float(model.training_closure_weight)
    split_results: dict[str, Any] = {}
    stability: dict[str, Any] = {}
    for split in FORMAL_SPLITS:
        records = load_a18_records(data_dir, split)
        split_results[split] = evaluate_a18_by_length(
            model,
            records,
            device,
            batch_size=batch_size,
            closure_weight=closure_weight,
        )
        stability[split] = evaluate_a18_stability(
            model,
            records,
            device,
            batch_size=batch_size,
            perturbation_seed=20260715 + int(model.a18_data_seed) % 1000,
        )
    pointer_splits = ("short_regression", "supported_in_range", "relation_in_range", "supported_ood", "relation_ood", "causal_core")
    gates = {
        "short_regression_each_length_at_least_0_95": _trajectory_gate(split_results, "short_regression", SHORT_LENGTHS, 0.95),
        "supported_t8_t12_t16_each_at_least_0_95": _trajectory_gate(split_results, "supported_in_range", IN_RANGE_LENGTHS, 0.95),
        "relation_t8_t12_t16_each_at_least_0_95": _trajectory_gate(split_results, "relation_in_range", IN_RANGE_LENGTHS, 0.95),
        "supported_t20_t24_each_at_least_0_90": _trajectory_gate(split_results, "supported_ood", HARD_OOD_LENGTHS, 0.90),
        "relation_t20_t24_each_at_least_0_90": _trajectory_gate(split_results, "relation_ood", HARD_OOD_LENGTHS, 0.90),
        "causal_free_trajectory_at_least_0_95": split_results["causal_core"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "all_pointer_accuracy_at_least_0_995": all(
            split_results[split]["aggregate"]["source_pointer_accuracy"] >= 0.995
            and split_results[split]["aggregate"]["target_pointer_accuracy"] >= 0.995
            for split in pointer_splits
        ),
        "answer_state_identity": all(split_results[split]["aggregate"]["answer_state_logits_identity"]["gate"] for split in pointer_splits),
        "prototype_non_collapse": all(split_results[split]["aggregate"]["prototype_statistics"]["non_collapse_gate"] for split in pointer_splits),
    }
    return {
        "schema_version": "yggdrasil.v2-a1.8.formal-eval.v1",
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "model_seed": model.a18_model_seed,
        "data_seed": model.a18_data_seed,
        "training_closure_weight": closure_weight,
        "architecture_change_from_a1_7": False,
        "integrity": model.integrity_report(),
        "splits": split_results,
        "stability": stability,
        "diagnostic_only": {
            "ood_length": DIAGNOSTIC_OOD_LENGTH,
            "reason": "T32 is reported but excluded from the preregistered trajectory Gate",
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
def run_a18_interventions(
    checkpoint: Path,
    data_dir: Path,
    device: str,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model = load_a18_checkpoint(checkpoint, device)
    intervention = run_a17_interventions(
        model,
        load_a18_records(data_dir, "causal_core"),
        device,
        batch_size=batch_size,
    )
    return {
        "schema_version": "yggdrasil.v2-a1.8.interventions.v1",
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "model_seed": model.a18_model_seed,
        "data_seed": model.a18_data_seed,
        "results": intervention,
        "passed": intervention["passed"],
    }


def _cuda_memory_report() -> dict[str, Any]:
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _timed_cuda(callable_: Any, repeats: int) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def benchmark_a18_costs(
    checkpoint: Path,
    data_dir: Path,
    device: str = "cuda",
    *,
    batch_size: int = 128,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A1.8 cost benchmark requires CUDA peak-memory and synchronized timing support")
    model = load_a18_checkpoint(checkpoint, device)
    parameter_report = model.parameter_report()
    active_parameters = int(parameter_report["trainable_parameters"])
    length_sources = {
        8: "supported_in_range",
        12: "supported_in_range",
        16: "supported_in_range",
        20: "supported_ood",
        24: "supported_ood",
        32: "supported_ood",
    }
    inference: dict[str, Any] = {}
    model.eval()
    for length, split in length_sources.items():
        records = [record for record in load_a18_records(data_dir, split) if int(record["program_length"]) == length][:batch_size]
        batch = encode_a17_records(records, device)

        def inference_step() -> None:
            with torch.no_grad():
                model(**_model_inputs(batch), return_trajectory=True)

        for _ in range(warmup):
            inference_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        seconds = _timed_cuda(inference_step, repeats)
        inference[str(length)] = {
            "batch_size": len(records),
            "latent_transitions_per_example": length,
            "mean_batch_seconds": seconds,
            "mean_example_seconds": seconds / len(records),
            "examples_per_second": len(records) / seconds,
            "estimated_forward_flops_parameter_proxy": 2 * active_parameters * len(records) * length,
            **_cuda_memory_report(),
        }
    grouped = _records_by_length(load_a18_records(data_dir, "train"))
    training_records, length_counts = sample_length_balanced_batch(grouped, batch_size, random.Random(20260715))
    training_batch = encode_a17_records(training_records, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def training_step() -> None:
        output = model(**_model_inputs(training_batch), return_trajectory=True)
        loss, _ = compute_a17_loss(model, output, training_batch, closure_weight=float(model.training_closure_weight))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(warmup):
        training_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    training_seconds = _timed_cuda(training_step, repeats)
    transitions_per_batch = sum(length * count for length, count in length_counts.items())
    training = {
        "batch_size": batch_size,
        "length_counts": dict(sorted(length_counts.items())),
        "latent_transitions_per_batch": transitions_per_batch,
        "mean_step_seconds": training_seconds,
        "examples_per_second": batch_size / training_seconds,
        "transitions_per_second": transitions_per_batch / training_seconds,
        "estimated_forward_backward_flops_parameter_proxy": 6 * active_parameters * transitions_per_batch,
        **_cuda_memory_report(),
    }
    return {
        "schema_version": "yggdrasil.v2-a1.8.cost-benchmark.v1",
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "device": {
            "type": device,
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "torch_version": torch.__version__,
        },
        "method": {
            "warmup_repeats": warmup,
            "timed_repeats": repeats,
            "cuda_synchronized": True,
            "peak_memory_includes_model_and_batch": True,
            "flops_are_coarse_parameter_activation_proxies_not_profiler_measurements": True,
        },
        "parameters": {
            "active": active_parameters,
            "trainable": int(parameter_report["trainable_parameters"]),
            "frozen": int(parameter_report["total_parameters"]) - int(parameter_report["trainable_parameters"]),
            "total": int(parameter_report["total_parameters"]),
        },
        "kv_cache": {
            "used": False,
            "reason": "the structured recurrent core has no transformer attention KV cache",
        },
        "training": training,
        "inference": inference,
    }
