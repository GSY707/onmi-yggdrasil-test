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
from .a1_8_data import IN_RANGE_LENGTHS, OOD_LENGTHS, SHORT_LENGTHS, TRAIN_LENGTHS
from .a1_10_cache import (
    A110CachedSplit,
    CACHE_MANIFEST_SCHEMA,
    collate_a110_items,
    sample_length_balanced_indices,
)
from .a1_10_model import A110AnonymousRecurrentReasoner, A110ModelConfig


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.10.anonymous-reasoner.checkpoint.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.10.anonymous-reasoner.results.v1"
FORMAL_SCHEMA = "yggdrasil.v2-a1.10.anonymous-reasoner.formal-eval.v1"
COST_SCHEMA = "yggdrasil.v2-a1.10.cached-anonymous-reasoner-cost.v1"
TRAIN_RECURRENT_STEPS = 16
STATE_LOSS_WEIGHT = 1.0
ANSWER_LOSS_WEIGHT = 1.0
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
class A110TrainSpec:
    model_seed: int
    data_seed: int
    steps: int = 4000
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_interval: int = 200
    early_stop_patience: int = 6
    train_recurrent_steps: int = TRAIN_RECURRENT_STEPS
    state_loss_weight: float = STATE_LOSS_WEIGHT
    answer_loss_weight: float = ANSWER_LOSS_WEIGHT


def _state_tensor(records: Sequence[dict[str, Any]], recurrent_steps: int, device: str | torch.device) -> torch.Tensor:
    targets = torch.empty(
        (len(records), recurrent_steps, len(REGISTER_NAMES)), dtype=torch.long, device=device
    )
    for row, record in enumerate(records):
        trajectory = record["state_trajectory"]
        if not trajectory:
            trajectory = [record["start_state"]]
        for step in range(recurrent_steps):
            state = trajectory[min(step, len(trajectory) - 1)]
            targets[row, step] = torch.tensor(
                [VALUE_LABELS.index(state[name]) for name in REGISTER_NAMES],
                dtype=torch.long,
                device=device,
            )
    return targets


def encode_a110_targets(
    records: Sequence[dict[str, Any]], recurrent_steps: int, device: str | torch.device
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot encode empty A1.10 targets")
    return {
        "state_targets": _state_tensor(records, recurrent_steps, device),
        "answer_targets": torch.tensor([int(record["answer_index"]) for record in records], dtype=torch.long, device=device),
        "query_register": torch.tensor(
            [REGISTER_NAMES.index(record["query_register"]) for record in records], dtype=torch.long, device=device
        ),
    }


def compute_a110_loss(
    output: dict[str, Any], labels: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    state_logits = output["state_logits"]
    state_ce = nn.functional.cross_entropy(
        state_logits.reshape(-1, len(VALUE_LABELS)), labels["state_targets"].reshape(-1)
    )
    answer_ce = nn.functional.cross_entropy(output["answer_logits"], labels["answer_targets"])
    total = STATE_LOSS_WEIGHT * state_ce + ANSWER_LOSS_WEIGHT * answer_ce
    return total, {
        "loss": float(total.detach()),
        "state_ce": float(state_ce.detach()),
        "answer_ce": float(answer_ce.detach()),
        "state_loss_weight": STATE_LOSS_WEIGHT,
        "answer_loss_weight": ANSWER_LOSS_WEIGHT,
    }


def _gradient_norm(model: nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().norm().square())
    return squared**0.5


@torch.no_grad()
def evaluate_a110_items(
    model: A110AnonymousRecurrentReasoner,
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
        raise ValueError("A1.10 evaluation requires at least one item")
    if old_records is not None and len(old_records) != len(items):
        raise ValueError("old_records must align with A1.10 items")
    model.eval()
    trajectory_full = final_state_full = answer_correct = consistency = 0
    state_correct = state_total = 0
    old_trajectory_full = old_examples = 0
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        inputs, records = collate_a110_items(batch_items, device)
        output = model(
            **inputs,
            recurrent_steps=recurrent_steps,
            disable_recurrence=disable_recurrence,
            slot_permutation=slot_permutation,
        )
        labels = encode_a110_targets(records, recurrent_steps, device)
        state_pred = output["state_logits"].argmax(dim=-1)
        state_equal = state_pred == labels["state_targets"]
        trajectory_full += int(state_equal.all(dim=(-1, -2)).sum())
        final_state_full += int(state_equal[:, -1].all(dim=-1).sum())
        state_correct += int(state_equal.sum())
        state_total += int(state_equal.numel())
        answer_pred = output["answer_logits"].argmax(dim=-1)
        answer_correct += int((answer_pred == labels["answer_targets"]).sum())
        query_index = labels["query_register"].view(-1, 1)
        final_state_answer = state_pred[:, -1].gather(1, query_index).squeeze(1)
        consistency += int((answer_pred == final_state_answer).sum())
        if old_records is not None:
            old_batch = old_records[start : start + len(batch_items)]
            old_labels = encode_a110_targets(old_batch, recurrent_steps, device)
            old_trajectory_full += int((state_pred == old_labels["state_targets"]).all(dim=(-1, -2)).sum())
            old_examples += len(batch_items)
    result = {
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
    return result


@torch.no_grad()
def evaluate_a110_by_length(
    model: A110AnonymousRecurrentReasoner,
    dataset: A110CachedSplit,
    device: str | torch.device,
    *,
    batch_size: int = 32,
    indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    selected = list(range(len(dataset))) if indices is None else list(indices)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index in selected:
        item = dataset[index]
        groups[int(item["record"]["program_length"])].append(item)
    by_length = {
        str(length): evaluate_a110_items(
            model,
            rows,
            device,
            recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
            batch_size=batch_size,
        )
        for length, rows in sorted(groups.items())
    }
    total_examples = sum(row["examples"] for row in by_length.values())
    aggregate: dict[str, Any] = {"examples": total_examples}
    for metric in (
        "trajectory_full_exact",
        "final_state_full_exact",
        "state_token_accuracy",
        "final_answer_accuracy",
        "answer_state_prediction_consistency",
    ):
        aggregate[metric] = sum(row[metric] * row["examples"] for row in by_length.values()) / max(1, total_examples)
    return {"aggregate": aggregate, "by_length": by_length}


def _validation_score(metrics: dict[str, Any]) -> tuple[tuple[float, float, float, float], dict[str, float]]:
    rows = list(metrics["by_length"].values())
    components = {
        "minimum_per_length_trajectory": min(row["trajectory_full_exact"] for row in rows),
        "minimum_per_length_final_state": min(row["final_state_full_exact"] for row in rows),
        "minimum_per_length_answer": min(row["final_answer_accuracy"] for row in rows),
        "aggregate_state_token": metrics["aggregate"]["state_token_accuracy"],
    }
    score = (
        components["minimum_per_length_trajectory"],
        components["minimum_per_length_final_state"],
        components["minimum_per_length_answer"],
        components["aggregate_state_token"],
    )
    return score, components


def _checkpoint(
    model: A110AnonymousRecurrentReasoner,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A110TrainSpec,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "model_config": asdict(model.config),
        "train_spec": asdict(spec),
        "model_seed": spec.model_seed,
        "data_seed": spec.data_seed,
        "qwen_frozen": True,
        "oracle_role_span_segmentation": False,
        "explicit_register_scaffold": False,
        "training_lengths": list(TRAIN_LENGTHS),
        "train_recurrent_steps": spec.train_recurrent_steps,
        "sampling": "uniform-length-balanced-within-every-batch",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def _balanced_subset_indices(dataset: A110CachedSplit, per_length: int = 32) -> list[int]:
    return [
        index
        for length in sorted(dataset.indices_by_length)
        for index in dataset.indices_by_length[length][:per_length]
    ]


def train_a110_reasoner(
    cache_dir: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    spec: A110TrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
    model_config: A110ModelConfig | None = None,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.10 training requested CUDA but CUDA is unavailable")
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to resume or mix A1.10 checkpoints in {output_dir}")
    if spec.train_recurrent_steps != TRAIN_RECURRENT_STEPS:
        raise ValueError(f"A1.10 train recurrent steps are frozen at {TRAIN_RECURRENT_STEPS}")
    if spec.state_loss_weight != STATE_LOSS_WEIGHT or spec.answer_loss_weight != ANSWER_LOSS_WEIGHT:
        raise ValueError("A1.10 loss weights are frozen at 1.0/1.0")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ValueError("A1.10 training requires a full-token cache manifest")
    selection = manifest["selection"]
    if overfit_mode != (selection == "length-balanced-overfit32-fit-only"):
        raise ValueError("A1.10 overfit mode must match cache selection")
    train_dataset = A110CachedSplit(cache_dir, data_dir, "train")
    validation_dataset = A110CachedSplit(cache_dir, data_dir, "validation")
    if set(train_dataset.indices_by_length) != set(TRAIN_LENGTHS):
        raise ValueError("A1.10 training cache must cover T1-T16")
    random.seed(spec.model_seed)
    torch.manual_seed(spec.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.model_seed)
    config = model_config or A110ModelConfig(source_width=train_dataset.hidden_width)
    if config.source_width != train_dataset.hidden_width:
        raise ValueError("A1.10 source width does not match Qwen cache")
    model = A110AnonymousRecurrentReasoner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    rng = random.Random(spec.model_seed)
    history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0, -1.0, -1.0)
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    patience = ready_streak = 0
    started = time.perf_counter()
    validation_indices = None if overfit_mode else _balanced_subset_indices(validation_dataset)
    for step in range(1, spec.steps + 1):
        model.train()
        indices = sample_length_balanced_indices(train_dataset, spec.batch_size, rng)
        inputs, records = collate_a110_items(train_dataset.items(indices), device)
        output = model(**inputs, recurrent_steps=spec.train_recurrent_steps)
        labels = encode_a110_targets(records, spec.train_recurrent_steps, device)
        loss, metrics = compute_a110_loss(output, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics["gradient_norm"] = _gradient_norm(model)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics["step"] = step
        metrics["sampled_length_counts"] = dict(
            sorted((length, sum(int(record["program_length"]) == length for record in records)) for length in TRAIN_LENGTHS)
        )
        if step % spec.validation_interval == 0 or step == spec.steps:
            validation = evaluate_a110_by_length(
                model,
                validation_dataset,
                device,
                batch_size=spec.batch_size,
                indices=validation_indices,
            )
            score, components = _validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = components
            checkpoint = _checkpoint(model, optimizer, step, spec)
            torch.save(checkpoint, output_dir / "latest.pt")
            if score > best_score:
                best_score = score
                best_step = step
                best_validation = validation
                torch.save(checkpoint, output_dir / "best.pt")
                patience = 0
            else:
                patience += 1
            target = 1.0 if overfit_mode else 0.95
            ready = (
                components["minimum_per_length_trajectory"] >= target
                and components["minimum_per_length_final_state"] >= target
                and components["minimum_per_length_answer"] >= target
            )
            ready_streak = ready_streak + 1 if ready else 0
            history.append(metrics)
            if ready_streak >= 2:
                break
            if not overfit_mode and patience >= spec.early_stop_patience:
                break
        else:
            history.append(metrics)
        if step % 20 == 0 or step == 1:
            (output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "steps_requested": spec.steps,
                        "best_score": list(best_score),
                        "best_step": best_step,
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    if not (output_dir / "best.pt").exists():
        raise RuntimeError("A1.10 training completed without a best checkpoint")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    fit_indices = _balanced_subset_indices(train_dataset, per_length=2 if overfit_mode else 32)
    fit_diagnostic = evaluate_a110_by_length(model, train_dataset, device, batch_size=spec.batch_size, indices=fit_indices)
    validation = evaluate_a110_by_length(model, validation_dataset, device, batch_size=spec.batch_size)
    overfit_gates = {
        "all_length_trajectory_exact": all(row["trajectory_full_exact"] == 1.0 for row in validation["by_length"].values()),
        "all_length_final_state_exact": all(row["final_state_full_exact"] == 1.0 for row in validation["by_length"].values()),
        "all_length_answer_exact": all(row["final_answer_accuracy"] == 1.0 for row in validation["by_length"].values()),
        "architecture_integrity": model.integrity_report()["passed"],
    }
    result = {
        "schema_version": RESULTS_SCHEMA,
        "stage": "A1.10 full-text anonymous recurrent reasoner",
        "selection": selection,
        "model_config": asdict(model.config),
        "train_spec": asdict(spec),
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": history[-1]["step"],
            "processed_examples": history[-1]["step"] * spec.batch_size,
            "processed_recurrent_transitions": history[-1]["step"] * spec.batch_size * spec.train_recurrent_steps,
            "seconds": time.perf_counter() - started,
            "sampling": "uniform-length-balanced-within-every-batch",
            "global_train_recurrent_steps": spec.train_recurrent_steps,
            "per_example_operation_mask_forward": False,
            "fresh_initialization": True,
            "resume_checkpoint": None,
            "overfit_mode": overfit_mode,
        },
        "best_score": list(best_score),
        "best_checkpoint_step": best_step,
        "best_validation_at_save": best_validation,
        "fit_diagnostic": fit_diagnostic,
        "validation": validation,
        "integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(qwen_parameters=int(manifest["qwen_parameters"])),
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


def load_a110_checkpoint(checkpoint_path: Path, device: str | torch.device = "cpu") -> A110AnonymousRecurrentReasoner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not an A1.10 checkpoint: {checkpoint_path}")
    model = A110AnonymousRecurrentReasoner(A110ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.a110_model_seed = int(checkpoint["model_seed"])
    model.a110_data_seed = int(checkpoint["data_seed"])
    model.a110_train_recurrent_steps = int(checkpoint["train_recurrent_steps"])
    model.eval()
    return model


def _metric_gate(
    results: dict[str, Any], split: str, lengths: Sequence[int], threshold: float
) -> bool:
    metrics = ("trajectory_full_exact", "final_state_full_exact", "final_answer_accuracy")
    return all(
        results[split]["by_length"][str(length)][metric] >= threshold
        for length in lengths
        for metric in metrics
    )


@torch.no_grad()
def evaluate_a110_formal(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    device: str = "cuda",
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    model = load_a110_checkpoint(checkpoint, device)
    results: dict[str, Any] = {}
    for split in FORMAL_SPLITS:
        dataset = A110CachedSplit(cache_dir, data_dir, split)
        results[split] = evaluate_a110_by_length(model, dataset, device, batch_size=batch_size)
        del dataset
    gates = {
        "short_t1_t6_each_metric_at_least_0_95": _metric_gate(results, "short_regression", SHORT_LENGTHS, 0.95),
        "supported_t8_t12_t16_each_metric_at_least_0_95": _metric_gate(results, "supported_in_range", IN_RANGE_LENGTHS, 0.95),
        "relation_t8_t12_t16_each_metric_at_least_0_95": _metric_gate(results, "relation_in_range", IN_RANGE_LENGTHS, 0.95),
        "supported_t20_t24_each_metric_at_least_0_90": _metric_gate(results, "supported_ood", HARD_OOD_LENGTHS, 0.90),
        "relation_t20_t24_each_metric_at_least_0_90": _metric_gate(results, "relation_ood", HARD_OOD_LENGTHS, 0.90),
        "causal_each_metric_at_least_0_95": all(
            results["causal_core"]["aggregate"][metric] >= 0.95
            for metric in ("trajectory_full_exact", "final_state_full_exact", "final_answer_accuracy")
        ),
        "all_formal_state_token_at_least_0_995": all(
            results[split]["aggregate"]["state_token_accuracy"] >= 0.995 for split in FORMAL_SPLITS
        ),
        "all_formal_answer_state_consistency_at_least_0_95": all(
            results[split]["aggregate"]["answer_state_prediction_consistency"] >= 0.95 for split in FORMAL_SPLITS
        ),
        "architecture_integrity": model.integrity_report()["passed"],
    }
    return {
        "schema_version": FORMAL_SCHEMA,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "model_seed": model.a110_model_seed,
        "data_seed": model.a110_data_seed,
        "integrity": model.integrity_report(),
        "splits": results,
        "diagnostic_only": {
            "ood_length": DIAGNOSTIC_OOD_LENGTH,
            "reason": "T32 is reported but excluded from the preregistered A1.10 Gate",
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


def benchmark_a110_cached_costs(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    device: str = "cuda",
    *,
    batch_size: int = 32,
    warmup: int = 2,
    repeats: int = 5,
) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A1.10 cost benchmark requires CUDA")
    model = load_a110_checkpoint(checkpoint, device)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    inference: dict[str, Any] = {}
    length_sources = {8: "supported_in_range", 12: "supported_in_range", 16: "supported_in_range", 20: "supported_ood", 24: "supported_ood", 32: "supported_ood"}
    for length, split in length_sources.items():
        dataset = A110CachedSplit(cache_dir, data_dir, split)
        indices = dataset.indices_by_length[length][:batch_size]
        inputs, _ = collate_a110_items(dataset.items(indices), device)

        recurrent_steps = max(TRAIN_RECURRENT_STEPS, length)

        def inference_step() -> None:
            with torch.no_grad():
                model(**inputs, recurrent_steps=recurrent_steps)

        for _ in range(warmup):
            inference_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        seconds = _timed_cuda(inference_step, repeats)
        inference[str(length)] = {
            "batch_size": len(indices),
            "recurrent_steps": recurrent_steps,
            "mean_batch_seconds": seconds,
            "mean_example_seconds": seconds / len(indices),
            "examples_per_second": len(indices) / seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        del dataset, inputs
    train_dataset = A110CachedSplit(cache_dir, data_dir, "train")
    indices = sample_length_balanced_indices(train_dataset, batch_size, random.Random(20260716))
    inputs, records = collate_a110_items(train_dataset.items(indices), device)
    labels = encode_a110_targets(records, TRAIN_RECURRENT_STEPS, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def training_step() -> None:
        output = model(**inputs, recurrent_steps=TRAIN_RECURRENT_STEPS)
        loss, _ = compute_a110_loss(output, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(warmup):
        training_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_seconds = _timed_cuda(training_step, repeats)
    disk_bytes = sum(path.stat().st_size for path in cache_dir.rglob("*" ) if path.is_file())
    return {
        "schema_version": COST_SCHEMA,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "device": {"type": "cuda", "name": torch.cuda.get_device_name(), "torch_version": torch.__version__},
        "method": {
            "warmup_repeats": warmup,
            "timed_repeats": repeats,
            "cuda_synchronized": True,
            "reasoner_latency_uses_precomputed_full_token_hidden": True,
            "qwen_encoding_cost_reported_from_cache_manifest": True,
            "not_online_end_to_end": True,
            "not_a_text_cot_pareto_measurement": True,
        },
        "parameters": model.parameter_report(qwen_parameters=int(manifest["qwen_parameters"])),
        "qwen_full_token_cache_generation": {
            "encoding_seconds_excluding_model_and_tokenizer_load": manifest[
                "full_token_encoding_seconds_excluding_model_and_tokenizer_load"
            ],
            "examples": sum(split["examples"] for split in manifest["splits"].values()),
            "source_tokens": sum(split["source_tokens"] for split in manifest["splits"].values()),
            "peak_allocated_bytes": manifest["peak_allocated_bytes"],
            "inference_batch_size": manifest["inference_batch_size"],
            "disk_bytes": disk_bytes,
        },
        "training": {
            "batch_size": batch_size,
            "recurrent_steps": TRAIN_RECURRENT_STEPS,
            "mean_step_seconds": train_seconds,
            "examples_per_second": batch_size / train_seconds,
            "latent_slot_updates_per_second": batch_size * TRAIN_RECURRENT_STEPS / train_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "inference": inference,
        "kv_cache": {"used": False, "reason": "Qwen cache extraction and recurrent reasoner both use use_cache=False"},
    }
