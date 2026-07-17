from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .a1_7_data import FAMILY_TO_INDEX, VALUE_LABELS
from .a1_9_model import module_state_sha256
from .a1_13_train import _write_json
from .a1_19h_h2_data import FORMAL_SPLITS
from .a1_19h_h2_train import load_a119h2_deployment
from .a1_20b_cache import (
    A120BCachedSplit,
    MANIFEST_SCHEMA,
    collate_a120b_items,
    collate_a120b_items_cpu,
    file_sha256,
    move_a120b_inputs,
)
from .a1_20b_model import A120BBoundaryConfig, A120BLearnedFullTextBoundary


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.20b.boundary.checkpoint.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.20b.boundary.results.v1"
FORMAL_SCHEMA = "yggdrasil.v2-a1.20b.boundary.formal.v1"
MAXIMUM_ENTITIES = 5
MAXIMUM_OPERATIONS = 32
PAYLOAD_CLOSURE_WEIGHT = 0.25


@dataclass(frozen=True)
class A120BTrainSpec:
    reader_seed: int
    data_seed: int
    core_model_seed: int
    steps: int = 5000
    batch_size: int = 16
    learning_rate: float = 3e-4
    validation_interval: int = 100


@dataclass(frozen=True)
class A120BLabels:
    entity_presence: torch.Tensor
    operation_presence: torch.Tensor
    value_targets: torch.Tensor
    family_targets: torch.Tensor
    source_targets: torch.Tensor
    target_targets: torch.Tensor
    query_targets: torch.Tensor
    state_targets: torch.Tensor
    answer_targets: torch.Tensor


def encode_a120b_labels(
    records: Sequence[dict[str, Any]], device: str | torch.device
) -> A120BLabels:
    batch = len(records)
    entity_presence = torch.zeros(
        (batch, MAXIMUM_ENTITIES), dtype=torch.float32, device=device
    )
    operation_presence = torch.zeros(
        (batch, MAXIMUM_OPERATIONS), dtype=torch.float32, device=device
    )
    value_targets = torch.full(
        (batch, MAXIMUM_ENTITIES), -100, dtype=torch.long, device=device
    )
    family_targets = torch.full(
        (batch, MAXIMUM_OPERATIONS), -100, dtype=torch.long, device=device
    )
    source_targets = torch.full_like(family_targets, -100)
    target_targets = torch.full_like(family_targets, -100)
    query_targets = torch.empty((batch,), dtype=torch.long, device=device)
    state_targets = torch.full(
        (batch, MAXIMUM_OPERATIONS, MAXIMUM_ENTITIES),
        -100,
        dtype=torch.long,
        device=device,
    )
    answer_targets = torch.empty((batch,), dtype=torch.long, device=device)
    for row, record in enumerate(records):
        names = list(record["entity_names"])
        name_to_index = {name: index for index, name in enumerate(names)}
        entity_count = len(names)
        program_length = int(record["program_length"])
        entity_presence[row, :entity_count] = 1.0
        operation_presence[row, :program_length] = 1.0
        for index, name in enumerate(names):
            value_targets[row, index] = VALUE_LABELS.index(
                record["start_state"][name]
            )
        query_targets[row] = name_to_index[record["query_register"]]
        for step, operation in enumerate(record["operations"]):
            family_targets[row, step] = FAMILY_TO_INDEX[operation["family"]]
            source_targets[row, step] = name_to_index[operation["source"]]
            target_targets[row, step] = name_to_index[operation["target"]]
        trajectory = record["state_trajectory"] or [record["start_state"]]
        for step in range(MAXIMUM_OPERATIONS):
            state = trajectory[min(step, len(trajectory) - 1)]
            for index, name in enumerate(names):
                state_targets[row, step, index] = VALUE_LABELS.index(state[name])
        answer_targets[row] = int(record["answer_index"])
    return A120BLabels(
        entity_presence=entity_presence,
        operation_presence=operation_presence,
        value_targets=value_targets,
        family_targets=family_targets,
        source_targets=source_targets,
        target_targets=target_targets,
        query_targets=query_targets,
        state_targets=state_targets,
        answer_targets=answer_targets,
    )


def index_a120b_labels(
    labels: A120BLabels, indices: torch.Tensor
) -> A120BLabels:
    return A120BLabels(
        **{
            name: value.index_select(0, indices)
            for name, value in labels.__dict__.items()
        }
    )


def move_a120b_labels(
    labels: A120BLabels,
    device: str | torch.device,
    *,
    non_blocking: bool,
) -> A120BLabels:
    return A120BLabels(
        **{
            name: value.to(device, non_blocking=non_blocking)
            for name, value in labels.__dict__.items()
        }
    )


def pin_a120b_labels(labels: A120BLabels) -> A120BLabels:
    return A120BLabels(
        **{name: value.pin_memory() for name, value in labels.__dict__.items()}
    )


def _balanced_schedule(
    values: Sequence[int], size: int, rng: random.Random
) -> list[int]:
    schedule = [int(values[index % len(values)]) for index in range(size)]
    rng.shuffle(schedule)
    return schedule


def sample_a120b_indices(
    dataset: A120BCachedSplit, batch_size: int, rng: random.Random
) -> list[int]:
    counts = sorted(dataset.indices_by_entity_count)
    lengths = sorted(dataset.indices_by_length)
    count_schedule = _balanced_schedule(counts, batch_size, rng)
    length_schedule = _balanced_schedule(lengths, batch_size, rng)
    indices: list[int] = []
    for count, length in zip(count_schedule, length_schedule):
        cell = dataset.indices_by_cell.get((count, length))
        candidates = cell or dataset.indices_by_entity_count[count]
        indices.append(rng.choice(candidates))
    return indices


def _prepare_a120b_cpu_batch(
    dataset: A120BCachedSplit,
    indices: Sequence[int],
    *,
    pin_memory: bool,
) -> tuple[dict[str, torch.Tensor], A120BLabels]:
    inputs, records = collate_a120b_items_cpu(
        dataset.items(indices), pin_memory=pin_memory
    )
    labels = encode_a120b_labels(records, "cpu")
    if pin_memory:
        labels = pin_a120b_labels(labels)
    return inputs, labels


def _prefetched_a120b_batches(
    dataset: A120BCachedSplit,
    schedule: Sequence[Sequence[int]],
    device: str | torch.device,
):
    pin_memory = torch.device(device).type == "cuda"
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="a120b-prefetch") as pool:
        future = pool.submit(
            _prepare_a120b_cpu_batch,
            dataset,
            schedule[0],
            pin_memory=pin_memory,
        )
        for step, indices in enumerate(schedule):
            cpu_inputs, cpu_labels = future.result()
            if step + 1 < len(schedule):
                future = pool.submit(
                    _prepare_a120b_cpu_batch,
                    dataset,
                    schedule[step + 1],
                    pin_memory=pin_memory,
                )
            yield (
                move_a120b_inputs(cpu_inputs, device),
                move_a120b_labels(
                    cpu_labels,
                    device,
                    non_blocking=pin_memory,
                ),
            )


def compute_a120b_loss(
    model: A120BLearnedFullTextBoundary,
    output: dict[str, Any],
    labels: A120BLabels,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    entity_presence = nn.functional.binary_cross_entropy_with_logits(
        output["entity_presence_logits"], labels.entity_presence
    )
    operation_presence = nn.functional.binary_cross_entropy_with_logits(
        output["operation_presence_logits"], labels.operation_presence
    )
    value_ce = nn.functional.cross_entropy(
        output["value_mapping_logits"].reshape(-1, model.core.config.value_classes),
        labels.value_targets.reshape(-1),
        ignore_index=-100,
    )
    family_ce = nn.functional.cross_entropy(
        output["family_mapping_logits"].reshape(-1, model.core.config.family_classes),
        labels.family_targets.reshape(-1),
        ignore_index=-100,
    )
    source_ce = nn.functional.cross_entropy(
        output["source_pointer_logits"].reshape(-1, MAXIMUM_ENTITIES),
        labels.source_targets.reshape(-1),
        ignore_index=-100,
    )
    target_ce = nn.functional.cross_entropy(
        output["target_pointer_logits"].reshape(-1, MAXIMUM_ENTITIES),
        labels.target_targets.reshape(-1),
        ignore_index=-100,
    )
    query_ce = nn.functional.cross_entropy(
        output["query_pointer_logits"], labels.query_targets
    )
    state_ce = nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, model.core.config.value_classes),
        labels.state_targets.reshape(-1),
        ignore_index=-100,
    )
    active_entities = labels.value_targets != -100
    target_payload = model.core.closure_targets(
        labels.value_targets.unsqueeze(1)
    ).squeeze(1).detach()
    payload_closure = (
        1.0
        - nn.functional.cosine_similarity(
            output["continuous_entity_payloads"], target_payload, dim=-1
        )
    )[active_entities].mean()
    total = (
        entity_presence
        + operation_presence
        + value_ce
        + family_ce
        + source_ce
        + target_ce
        + query_ce
        + state_ce
        + PAYLOAD_CLOSURE_WEIGHT * payload_closure
    )
    return total, {
        "entity_presence_bce": entity_presence,
        "operation_presence_bce": operation_presence,
        "value_ce": value_ce,
        "family_ce": family_ce,
        "source_ce": source_ce,
        "target_ce": target_ce,
        "query_ce": query_ce,
        "state_ce": state_ce,
        "payload_closure": payload_closure,
    }


def _metrics_from_loss(
    loss: torch.Tensor,
    components: dict[str, torch.Tensor],
    gradient_norm: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": float(loss.detach()),
        **{name: float(value.detach()) for name, value in components.items()},
        "payload_closure_weight": PAYLOAD_CLOSURE_WEIGHT,
        "official_answer_loss_weight": 0.0,
        "gradient_norm": float(gradient_norm.detach()),
    }


@torch.inference_mode()
def evaluate_a120b_items(
    model: A120BLearnedFullTextBoundary,
    items: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    if not items:
        raise ValueError("A1.20B evaluation requires items")
    model.eval()
    counters: dict[str, int] = defaultdict(int)
    payload_cosines: list[float] = []
    answer_identity_max = 0.0
    for start in range(0, len(items), batch_size):
        inputs, records = collate_a120b_items(
            items[start : start + batch_size], device
        )
        labels = encode_a120b_labels(records, device)
        output = model(**inputs)
        state_prediction = output["state_logits"].argmax(dim=-1)
        active_state = labels.state_targets != -100
        equal = (state_prediction == labels.state_targets) | ~active_state
        counters["trajectory"] += int(equal.all(dim=(-1, -2)).sum())
        counters["final_state"] += int(equal[:, -1].all(dim=-1).sum())
        counters["state_correct"] += int(
            ((state_prediction == labels.state_targets) & active_state).sum()
        )
        counters["state_total"] += int(active_state.sum())
        counters["answer"] += int(
            (output["answer_logits"].argmax(dim=-1) == labels.answer_targets).sum()
        )
        entity_mask_target = labels.entity_presence.bool()
        operation_mask_target = labels.operation_presence.bool()
        counters["entity_presence"] += int(
            (output["predicted_entity_mask"] == entity_mask_target).all(dim=-1).sum()
        )
        counters["operation_presence"] += int(
            (output["predicted_operation_mask"] == operation_mask_target)
            .all(dim=-1)
            .sum()
        )
        value_prediction = output["value_mapping_logits"].argmax(dim=-1)
        counters["value_correct"] += int(
            ((value_prediction == labels.value_targets) & entity_mask_target).sum()
        )
        counters["value_total"] += int(entity_mask_target.sum())
        family_prediction = output["predicted_family"]
        source_prediction = output["predicted_source_pointer"]
        target_prediction = output["predicted_target_pointer"]
        counters["family_correct"] += int(
            ((family_prediction == labels.family_targets) & operation_mask_target).sum()
        )
        counters["source_correct"] += int(
            ((source_prediction == labels.source_targets) & operation_mask_target).sum()
        )
        counters["target_correct"] += int(
            ((target_prediction == labels.target_targets) & operation_mask_target).sum()
        )
        counters["operation_total"] += int(operation_mask_target.sum())
        counters["query_correct"] += int(
            (output["predicted_query_pointer"] == labels.query_targets).sum()
        )
        predicted_query = output["predicted_query_pointer"].view(-1, 1, 1)
        predicted_query = predicted_query.expand(-1, 1, model.core.config.value_classes)
        state_answer_logits = output["state_logits"][:, -1].gather(
            1, predicted_query
        ).squeeze(1)
        answer_identity_max = max(
            answer_identity_max,
            float((output["answer_logits"] - state_answer_logits).abs().max()),
        )
        target_payload = model.core.closure_targets(
            labels.value_targets.unsqueeze(1)
        ).squeeze(1)
        cosine = nn.functional.cosine_similarity(
            output["continuous_entity_payloads"], target_payload, dim=-1
        )
        payload_cosines.extend(cosine[entity_mask_target].float().cpu().tolist())
    examples = len(items)
    mapping = {
        "entity_presence": counters["entity_presence"] / examples,
        "operation_presence": counters["operation_presence"] / examples,
        "value": counters["value_correct"] / max(1, counters["value_total"]),
        "family": counters["family_correct"]
        / max(1, counters["operation_total"]),
        "source": counters["source_correct"]
        / max(1, counters["operation_total"]),
        "target": counters["target_correct"]
        / max(1, counters["operation_total"]),
        "query": counters["query_correct"] / examples,
    }
    return {
        "examples": examples,
        "trajectory_full_exact": counters["trajectory"] / examples,
        "final_state_full_exact": counters["final_state"] / examples,
        "state_token_accuracy": counters["state_correct"]
        / max(1, counters["state_total"]),
        "final_answer_accuracy": counters["answer"] / examples,
        "mapping_accuracy": {**mapping, "minimum": min(mapping.values())},
        "continuous_payload_target_cosine_mean": sum(payload_cosines)
        / max(1, len(payload_cosines)),
        "answer_state_logits_identity": {
            "max_abs_diff": answer_identity_max,
            "gate": answer_identity_max <= 1e-6,
        },
        "diagnostic_counts": dict(counters),
        "diagnostic_payload_cosine_sum": sum(payload_cosines),
        "diagnostic_payload_cosine_count": len(payload_cosines),
    }


def _aggregate_a120b_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, int] = defaultdict(int)
    examples = 0
    cosine_sum = 0.0
    cosine_count = 0
    identity_max = 0.0
    for row in rows:
        examples += int(row["examples"])
        for name, value in row["diagnostic_counts"].items():
            counters[name] += int(value)
        cosine_sum += float(row["diagnostic_payload_cosine_sum"])
        cosine_count += int(row["diagnostic_payload_cosine_count"])
        identity_max = max(
            identity_max,
            float(row["answer_state_logits_identity"]["max_abs_diff"]),
        )
    mapping = {
        "entity_presence": counters["entity_presence"] / examples,
        "operation_presence": counters["operation_presence"] / examples,
        "value": counters["value_correct"] / max(1, counters["value_total"]),
        "family": counters["family_correct"] / max(1, counters["operation_total"]),
        "source": counters["source_correct"] / max(1, counters["operation_total"]),
        "target": counters["target_correct"] / max(1, counters["operation_total"]),
        "query": counters["query_correct"] / examples,
    }
    return {
        "examples": examples,
        "trajectory_full_exact": counters["trajectory"] / examples,
        "final_state_full_exact": counters["final_state"] / examples,
        "state_token_accuracy": counters["state_correct"]
        / max(1, counters["state_total"]),
        "final_answer_accuracy": counters["answer"] / examples,
        "mapping_accuracy": {**mapping, "minimum": min(mapping.values())},
        "continuous_payload_target_cosine_mean": cosine_sum
        / max(1, cosine_count),
        "answer_state_logits_identity": {
            "max_abs_diff": identity_max,
            "gate": identity_max <= 1e-6,
        },
        "diagnostic_counts": dict(counters),
        "diagnostic_payload_cosine_sum": cosine_sum,
        "diagnostic_payload_cosine_count": cosine_count,
    }


@torch.inference_mode()
def evaluate_a120b_matrix(
    model: A120BLearnedFullTextBoundary,
    dataset: A120BCachedSplit,
    device: str | torch.device,
    *,
    batch_size: int,
    indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    selected = list(range(len(dataset))) if indices is None else list(indices)
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for index in selected:
        item = dataset[index]
        record = item["record"]
        cells[(int(record["entity_count"]), int(record["program_length"]))].append(
            item
        )
    by_cell = {
        f"N{count}-T{length}": evaluate_a120b_items(
            model, rows, device, batch_size=batch_size
        )
        for (count, length), rows in sorted(cells.items())
    }
    aggregate = _aggregate_a120b_rows(list(by_cell.values()))
    return {"aggregate": aggregate, "by_cell": by_cell}


def _validation_score(metrics: dict[str, Any]) -> tuple[tuple[float, ...], dict[str, float]]:
    rows = list(metrics["by_cell"].values())
    components = {
        "minimum_cell_mapping": min(
            row["mapping_accuracy"]["minimum"] for row in rows
        ),
        "minimum_cell_trajectory": min(row["trajectory_full_exact"] for row in rows),
        "minimum_cell_answer": min(row["final_answer_accuracy"] for row in rows),
        "aggregate_state_token": metrics["aggregate"]["state_token_accuracy"],
    }
    return tuple(components.values()), components


def _checkpoint(
    model: A120BLearnedFullTextBoundary,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A120BTrainSpec,
    core_checkpoint: Path,
    cache_manifest_hash: str,
    cache_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": step,
        "boundary_config": asdict(model.config),
        "train_spec": asdict(spec),
        "core_checkpoint": str(core_checkpoint.resolve()),
        "core_checkpoint_sha256": file_sha256(core_checkpoint),
        "core_state_sha256": module_state_sha256(model.core),
        "cache_manifest_sha256": cache_manifest_hash,
        "cache_contract": cache_contract,
        "boundary": model.boundary_state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def train_a120b_boundary(
    cache_dir: Path,
    data_dir: Path,
    core_checkpoint: Path,
    output_dir: Path,
    *,
    spec: A120BTrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.20B requested unavailable CUDA")
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.20B checkpoints: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("A1.20B training requires its full-token cache")
    if not {"train", "validation"}.issubset(manifest["splits"]):
        raise ValueError("A1.20B training cache requires train and validation splits")
    cache_is_overfit = manifest["selection"].endswith("overfit32-fit-only")
    if cache_is_overfit != overfit_mode:
        raise ValueError("A1.20B overfit mode/cache selection mismatch")
    if manifest["data_manifest_sha256"] != file_sha256(data_dir / "manifest.json"):
        raise ValueError("A1.20B cache/data manifest mismatch")
    train_dataset = A120BCachedSplit(cache_dir, data_dir, "train")
    validation_dataset = A120BCachedSplit(cache_dir, data_dir, "validation")
    random.seed(spec.reader_seed)
    torch.manual_seed(spec.reader_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.reader_seed)
    core = load_a119h2_deployment(core_checkpoint, device)
    if core.a119h_model_seed != spec.core_model_seed:
        raise ValueError("A1.20B core model seed mismatch")
    if core.a119h_data_seed != spec.data_seed:
        raise ValueError("A1.20B core/data seed mismatch")
    core_hash = module_state_sha256(core)
    model = A120BLearnedFullTextBoundary(
        A120BBoundaryConfig(source_width=train_dataset.hidden_width), core
    ).to(device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate)
    rng = random.Random(spec.reader_seed)
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    ready_streak = 0
    completed_step = 0
    cache_manifest_hash = file_sha256(manifest_path)
    cache_contract = {
        "data_manifest_sha256": manifest["data_manifest_sha256"],
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "dtype": manifest["dtype"],
        "hidden_width": train_dataset.hidden_width,
    }
    preloaded_inputs: dict[str, torch.Tensor] | None = None
    preloaded_records: list[dict[str, Any]] | None = None
    preloaded_labels: A120BLabels | None = None
    if overfit_mode:
        preloaded_inputs, preloaded_records = collate_a120b_items(
            train_dataset.items(), device
        )
        preloaded_labels = encode_a120b_labels(preloaded_records, device)
        schedule = None
    else:
        schedule = [
            sample_a120b_indices(train_dataset, spec.batch_size, rng)
            for _ in range(spec.steps)
        ]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        if overfit_mode:
            assert preloaded_inputs is not None and preloaded_labels is not None
            inputs = preloaded_inputs
            labels = preloaded_labels
        else:
            assert schedule is not None
            indices = schedule[step - 1]
            inputs, records = collate_a120b_items(
                train_dataset.items(indices), device
            )
            labels = encode_a120b_labels(records, device)
        output = model(**inputs)
        loss, components = compute_a120b_loss(model, output, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None for parameter in model.core.parameters()):
            raise RuntimeError("frozen A1.19H core received gradients")
        gradient_norm = nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        completed_step = step
        if step % spec.validation_interval == 0 or step == spec.steps:
            metrics = _metrics_from_loss(loss, components, gradient_norm)
            metrics["step"] = step
            validation = evaluate_a120b_matrix(
                model,
                validation_dataset,
                device,
                batch_size=spec.batch_size,
            )
            score, score_components = _validation_score(validation)
            metrics["validation"] = validation
            metrics["validation_score_components"] = score_components
            checkpoint = _checkpoint(
                model,
                optimizer,
                step,
                spec,
                core_checkpoint,
                cache_manifest_hash,
                cache_contract,
            )
            torch.save(checkpoint, output_dir / "latest.pt")
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_validation = validation
                torch.save(checkpoint, output_dir / "best.pt")
            target = 1.0 if overfit_mode else 0.995
            ready = all(value >= target for value in score_components.values())
            ready_streak = ready_streak + 1 if ready else 0
            history.append(metrics)
            _write_json(
                output_dir / "progress.json",
                {
                    "stage": "A1.20B",
                    "step": step,
                    "steps_requested": spec.steps,
                    "best_step": best_step,
                    "best_score": best_score,
                    "elapsed_seconds": time.perf_counter() - started,
                    "overfit_mode": overfit_mode,
                },
            )
            if overfit_mode and ready_streak >= 2:
                break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    _write_json(output_dir / "history.json", history)
    checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_boundary_state_dict(checkpoint["boundary"])
    validation = evaluate_a120b_matrix(
        model, validation_dataset, device, batch_size=spec.batch_size
    )
    core_hash_after = module_state_sha256(model.core)
    overfit_gates = {
        "all_cell_mapping_exact": all(
            row["mapping_accuracy"]["minimum"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_cell_trajectory_exact": all(
            row["trajectory_full_exact"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_cell_final_state_exact": all(
            row["final_state_full_exact"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_cell_answer_exact": all(
            row["final_answer_accuracy"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "answer_state_identity": validation["aggregate"][
            "answer_state_logits_identity"
        ]["gate"],
        "continuous_payload_cosine_at_least_0_999": validation["aggregate"][
            "continuous_payload_target_cosine_mean"
        ]
        >= 0.999,
        "core_hash_unchanged": core_hash_after == core_hash,
        "architecture_integrity": model.integrity_report()["passed"],
    }
    result = {
        "schema_version": RESULTS_SCHEMA,
        "stage": "V2-A1.20B learned full-text boundary",
        "selection": manifest["selection"],
        "train_spec": asdict(spec),
        "boundary_config": asdict(model.config),
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": completed_step,
            "processed_examples": completed_step
            * (len(train_dataset) if overfit_mode else spec.batch_size),
            "seconds": training_seconds,
            "steps_per_second": completed_step / max(training_seconds, 1e-9),
            "fresh_initialization": True,
            "overfit_mode": overfit_mode,
            "full_overfit_cache_preloaded_on_gpu": overfit_mode,
        },
        "best_checkpoint_step": best_step,
        "best_validation_at_save": best_validation,
        "validation": validation,
        "integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(),
        "core_state_sha256_before": core_hash,
        "core_state_sha256_after": core_hash_after,
        "cache_manifest_sha256": cache_manifest_hash,
        "cache_contract": cache_contract,
        "overfit_gates": overfit_gates if overfit_mode else None,
        "overfit_passed": all(overfit_gates.values()) if overfit_mode else None,
    }
    _write_json(output_dir / "results.json", result)
    return result


def benchmark_a120b_training_pipeline(
    cache_dir: Path,
    data_dir: Path,
    core_checkpoint: Path,
    output_path: Path,
    *,
    reader_seed: int,
    steps: int = 20,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("A1.20B benchmark steps must be positive")
    dataset = A120BCachedSplit(cache_dir, data_dir, "train")
    rng = random.Random(reader_seed)
    schedule = [
        sample_a120b_indices(dataset, batch_size, rng) for _ in range(steps)
    ]
    template_core = load_a119h2_deployment(core_checkpoint, device)
    torch.manual_seed(reader_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(reader_seed)
    template = A120BLearnedFullTextBoundary(
        A120BBoundaryConfig(source_width=dataset.hidden_width), template_core
    ).to(device)
    initial_boundary = deepcopy(template.boundary_state_dict())
    del template, template_core

    def new_model() -> A120BLearnedFullTextBoundary:
        core = load_a119h2_deployment(core_checkpoint, device)
        model = A120BLearnedFullTextBoundary(
            A120BBoundaryConfig(source_width=dataset.hidden_width), core
        ).to(device)
        model.load_boundary_state_dict(initial_boundary)
        return model

    def run(prefetched: bool) -> tuple[float, float, dict[str, torch.Tensor]]:
        model = new_model()
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable, lr=3e-4)
        iterator = (
            iter(_prefetched_a120b_batches(dataset, schedule, device))
            if prefetched
            else None
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        last_loss: torch.Tensor | None = None
        for indices in schedule:
            model.train()
            if iterator is None:
                inputs, records = collate_a120b_items(
                    dataset.items(indices), device
                )
                labels = encode_a120b_labels(records, device)
            else:
                inputs, labels = next(iterator)
            output = model(**inputs)
            last_loss, _ = compute_a120b_loss(model, output, labels)
            optimizer.zero_grad(set_to_none=True)
            last_loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert last_loss is not None
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.boundary_state_dict().items()
        }
        return elapsed, float(last_loss.detach()), state

    prefetch_seconds, prefetch_loss, prefetch_state = run(True)
    synchronous_seconds, synchronous_loss, synchronous_state = run(False)
    parameter_delta = max(
        float((prefetch_state[name] - synchronous_state[name]).abs().max())
        for name in prefetch_state
    )
    result = {
        "schema_version": "yggdrasil.v2-a1.20b.training-pipeline-benchmark.v1",
        "cache_dir": str(cache_dir),
        "device": device,
        "reader_seed": reader_seed,
        "steps": steps,
        "batch_size": batch_size,
        "contract": {
            "schedule_identical": True,
            "initialization_identical": True,
            "precision": "fp32-boundary/fp16-source-cache",
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
        },
        "synchronous": {
            "seconds": synchronous_seconds,
            "steps_per_second": steps / synchronous_seconds,
            "final_loss": synchronous_loss,
        },
        "prefetched": {
            "seconds": prefetch_seconds,
            "steps_per_second": steps / prefetch_seconds,
            "final_loss": prefetch_loss,
            "pinned_memory": torch.device(device).type == "cuda",
            "background_cpu_workers": 1,
        },
        "speedup": synchronous_seconds / prefetch_seconds,
        "equivalence": {
            "final_loss_abs_delta": abs(synchronous_loss - prefetch_loss),
            "final_parameter_max_abs_delta": parameter_delta,
            "passed": abs(synchronous_loss - prefetch_loss) == 0.0
            and parameter_delta == 0.0,
        },
    }
    _write_json(output_path, result)
    return result


def load_a120b_checkpoint(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> A120BLearnedFullTextBoundary:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("not an A1.20B checkpoint")
    core_path = Path(checkpoint["core_checkpoint"])
    if file_sha256(core_path) != checkpoint["core_checkpoint_sha256"]:
        raise ValueError("A1.20B core checkpoint hash mismatch")
    core = load_a119h2_deployment(core_path, device)
    if module_state_sha256(core) != checkpoint["core_state_sha256"]:
        raise ValueError("A1.20B core state hash mismatch")
    model = A120BLearnedFullTextBoundary(
        A120BBoundaryConfig(**checkpoint["boundary_config"]), core
    ).to(device)
    model.load_boundary_state_dict(checkpoint["boundary"])
    model.a120b_reader_seed = int(checkpoint["train_spec"]["reader_seed"])
    model.a120b_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a120b_core_model_seed = int(
        checkpoint["train_spec"]["core_model_seed"]
    )
    model.a120b_core_state_sha256 = checkpoint["core_state_sha256"]
    model.a120b_cache_manifest_sha256 = checkpoint["cache_manifest_sha256"]
    model.a120b_cache_contract = checkpoint["cache_contract"]
    model.eval()
    return model


@torch.inference_mode()
def evaluate_a120b_formal(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, Any]:
    model = load_a120b_checkpoint(checkpoint, device)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("A1.20B formal requires its cache schema")
    if not set(FORMAL_SPLITS).issubset(manifest["splits"]):
        raise ValueError("A1.20B formal cache is missing formal splits")
    observed_contract = {
        "data_manifest_sha256": manifest["data_manifest_sha256"],
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "dtype": manifest["dtype"],
        "hidden_width": int(
            A120BCachedSplit(cache_dir, data_dir, FORMAL_SPLITS[0]).hidden_width
        ),
    }
    if observed_contract != model.a120b_cache_contract:
        raise ValueError("A1.20B formal cache contract mismatch")
    results = {
        split: evaluate_a120b_matrix(
            model,
            A120BCachedSplit(cache_dir, data_dir, split),
            device,
            batch_size=batch_size,
        )
        for split in FORMAL_SPLITS
    }
    thresholds = {
        "short_regression": 0.95,
        "supported_in_range": 0.95,
        "relation_in_range": 0.95,
        "supported_ood": 0.90,
        "relation_ood": 0.90,
        "entity_heldout": 0.90,
        "entity_relation_heldout": 0.90,
        "causal_core": 0.95,
    }
    gates: dict[str, bool] = {}
    for split, threshold in thresholds.items():
        aggregate = results[split]["aggregate"]
        for metric in (
            "trajectory_full_exact",
            "final_state_full_exact",
            "final_answer_accuracy",
        ):
            gates[f"{split}_{metric}"] = aggregate[metric] >= threshold
        gates[f"{split}_mapping"] = (
            aggregate["mapping_accuracy"]["minimum"] >= 0.995
        )
    gates.update(
        {
            "all_state_token_at_least_0_995": all(
                results[split]["aggregate"]["state_token_accuracy"] >= 0.995
                for split in FORMAL_SPLITS
            ),
            "all_answer_state_identity": all(
                results[split]["aggregate"]["answer_state_logits_identity"]["gate"]
                for split in FORMAL_SPLITS
            ),
            "core_hash_unchanged": module_state_sha256(model.core)
            == model.a120b_core_state_sha256,
            "architecture_integrity": model.integrity_report()["passed"],
        }
    )
    return {
        "schema_version": FORMAL_SCHEMA,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "reader_seed": model.a120b_reader_seed,
        "data_seed": model.a120b_data_seed,
        "core_model_seed": model.a120b_core_model_seed,
        "splits": results,
        "gates": gates,
        "passed": all(gates.values()),
    }
