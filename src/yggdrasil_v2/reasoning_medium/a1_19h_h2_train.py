from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from .a1_8_data import IN_RANGE_LENGTHS, OOD_LENGTHS, SHORT_LENGTHS, TRAIN_LENGTHS
from .a1_9_cache import file_sha256
from .a1_10_train import TRAIN_RECURRENT_STEPS
from .a1_13_train import CLOSURE_WEIGHT, _write_json
from .a1_19h_data import (
    A119HBatch,
    a119h_batch_nbytes,
    encode_a119h_records,
    index_a119h_batch,
    move_a119h_batch,
    validate_a119h_batch_addresses,
)
from .a1_19h_h2_data import (
    FORMAL_SPLITS,
    HELDOUT_ENTITY_COUNT,
    SCHEMA_VERSION as DATA_SCHEMA,
    TRAIN_ENTITY_COUNTS,
    load_a119h2_records,
)
from .a1_19h_model import A119HConfig, A119HHybridReasoner
from .a1_19h_train import _weighted


CHECKPOINT_SCHEMA = "yggdrasil.v2-a1.19h.h2-variable-cardinality.checkpoint.v1"
DEPLOYMENT_SCHEMA = "yggdrasil.v2-a1.19h.h2-variable-cardinality.deployment.v1"
RESULTS_SCHEMA = "yggdrasil.v2-a1.19h.h2-variable-cardinality.results.v1"
FORMAL_SCHEMA = "yggdrasil.v2-a1.19h.h2-variable-cardinality.formal.v1"
BENCHMARK_SCHEMA = "yggdrasil.v2-a1.19h.h2-throughput-benchmark.v1"


@dataclass(frozen=True)
class A119H2TrainSpec:
    model_seed: int
    data_seed: int
    mapping_seed: int
    steps: int = 4000
    batch_size: int = 32
    learning_rate: float = 3e-4
    validation_interval: int = 200
    auxiliary_weight: float = 1.0
    train_recurrent_steps: int = TRAIN_RECURRENT_STEPS


class A119H2RecordSplit:
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)
        self.indices_by_length: dict[int, list[int]] = defaultdict(list)
        self.indices_by_entity_count: dict[int, list[int]] = defaultdict(list)
        self.indices_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            length = int(record["program_length"])
            count = int(record["entity_count"])
            self.indices_by_length[length].append(index)
            self.indices_by_entity_count[count].append(index)
            self.indices_by_cell[(count, length)].append(index)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _balanced_schedule(values: Sequence[int], size: int, rng: random.Random) -> list[int]:
    schedule = [int(values[index % len(values)]) for index in range(size)]
    rng.shuffle(schedule)
    return schedule


def _sample_h2_batch(
    dataset: A119H2RecordSplit, batch_size: int, rng: random.Random
) -> list[dict[str, Any]]:
    return [dataset[index] for index in _sample_h2_indices(dataset, batch_size, rng)]


def _sample_h2_indices(
    dataset: A119H2RecordSplit, batch_size: int, rng: random.Random
) -> list[int]:
    counts = sorted(dataset.indices_by_entity_count)
    lengths = sorted(dataset.indices_by_length)
    count_schedule = _balanced_schedule(counts, batch_size, rng)
    length_schedule = _balanced_schedule(lengths, batch_size, rng)
    indices: list[int] = []
    for count, length in zip(count_schedule, length_schedule):
        cell = dataset.indices_by_cell.get((count, length))
        if not cell:
            candidates = dataset.indices_by_entity_count[count]
        else:
            candidates = cell
        indices.append(rng.choice(candidates))
    return indices


def _a119h_loss_tensors(
    model: A119HHybridReasoner,
    output: dict[str, Any],
    batch: A119HBatch,
    *,
    auxiliary_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    auxiliary_logits = output.get("training_auxiliary_logits")
    if auxiliary_logits is None:
        raise RuntimeError("A1.19H-H2 training auxiliary is missing")
    state_ce = nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, model.config.value_classes),
        batch.state_targets.reshape(-1),
        ignore_index=-100,
    )
    predicted = output["trajectory"][:, :, : model.config.maximum_entities]
    closure_targets = model.closure_targets(batch.state_targets).detach()
    active = batch.state_targets != -100
    closure_loss = (
        1.0 - nn.functional.cosine_similarity(predicted, closure_targets, dim=-1)
    )[active].mean()
    auxiliary_ce = nn.functional.cross_entropy(
        auxiliary_logits.reshape(-1, model.config.value_classes),
        batch.state_targets.reshape(-1),
        ignore_index=-100,
    )
    official_answer_ce = nn.functional.cross_entropy(
        output["answer_logits"], batch.answer_targets
    )
    total = (
        state_ce
        + CLOSURE_WEIGHT * closure_loss
        + auxiliary_weight * auxiliary_ce
    )
    auxiliary_accuracy = (
        auxiliary_logits.argmax(dim=-1)[active] == batch.state_targets[active]
    ).float().mean()
    return total, {
        "state_ce": state_ce,
        "closure_loss": closure_loss,
        "auxiliary_ce": auxiliary_ce,
        "auxiliary_accuracy": auxiliary_accuracy,
        "official_answer_ce": official_answer_ce,
    }


def _loss_metrics(
    total: torch.Tensor,
    components: dict[str, torch.Tensor],
    gradient_norm: torch.Tensor,
    auxiliary_weight: float,
) -> dict[str, float]:
    return {
        "loss": float(total.detach()),
        "state_ce": float(components["state_ce"].detach()),
        "closure_loss": float(components["closure_loss"].detach()),
        "closure_weight": CLOSURE_WEIGHT,
        "training_auxiliary": "trajectory_set_state",
        "auxiliary_ce": float(components["auxiliary_ce"].detach()),
        "auxiliary_accuracy": float(components["auxiliary_accuracy"].detach()),
        "auxiliary_weight": auxiliary_weight,
        "official_answer_ce_diagnostic_only": float(
            components["official_answer_ce"].detach()
        ),
        "official_answer_loss_weight": 0.0,
        "gradient_norm": float(gradient_norm.detach()),
    }


def _select_overfit32(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset = A119H2RecordSplit(records)
    counts = sorted(dataset.indices_by_entity_count)
    lengths = sorted(dataset.indices_by_length)
    selected: list[dict[str, Any]] = []
    cursor: dict[tuple[int, int], int] = defaultdict(int)
    for index in range(32):
        cell = (counts[index % len(counts)], lengths[index % len(lengths)])
        choices = dataset.indices_by_cell[cell]
        selected.append(dataset[choices[cursor[cell] % len(choices)]])
        cursor[cell] += 1
    return selected


def _synchronize(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _slice_a119h_batch_steps(
    batch: A119HBatch, recurrent_steps: int
) -> A119HBatch:
    return A119HBatch(
        inputs=batch.inputs,
        state_targets=batch.state_targets[:, :recurrent_steps],
        answer_targets=batch.answer_targets,
        semantic_slot_orders=batch.semantic_slot_orders,
    )


def _encode_a119h2_cache(
    dataset: A119H2RecordSplit,
    device: str | torch.device,
    *,
    mapping_seed: int,
    maximum_entities: int,
    recurrent_steps: int,
) -> tuple[A119HBatch, A119HBatch, dict[str, Any]]:
    started = time.perf_counter()
    cpu_batch = encode_a119h_records(
        dataset.records,
        recurrent_steps,
        "cpu",
        mapping_seed=mapping_seed,
        maximum_entities=maximum_entities,
        include_semantic_slot_orders=False,
    )
    validate_a119h_batch_addresses(cpu_batch)
    encoded_seconds = time.perf_counter() - started
    transfer_started = time.perf_counter()
    device_batch = move_a119h_batch(cpu_batch, device)
    _synchronize(device)
    transfer_seconds = time.perf_counter() - transfer_started
    return cpu_batch, device_batch, {
        "records": len(dataset),
        "recurrent_steps": recurrent_steps,
        "bytes": a119h_batch_nbytes(cpu_batch),
        "encoding_seconds": encoded_seconds,
        "device_transfer_seconds": transfer_seconds,
        "total_seconds": encoded_seconds + transfer_seconds,
        "address_validation": "once_on_cpu_before_training",
    }


def _batches_equal(left: A119HBatch, right: A119HBatch) -> bool:
    return (
        left.inputs.keys() == right.inputs.keys()
        and all(torch.equal(left.inputs[name], right.inputs[name]) for name in left.inputs)
        and torch.equal(left.state_targets, right.state_targets)
        and torch.equal(left.answer_targets, right.answer_targets)
    )


def _verify_cache_equivalence(
    dataset: A119H2RecordSplit,
    cpu_cache: A119HBatch,
    indices: Sequence[int],
    *,
    mapping_seed: int,
    maximum_entities: int,
    recurrent_steps: int,
) -> dict[str, Any]:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    cached = index_a119h_batch(cpu_cache, index_tensor)
    direct = encode_a119h_records(
        [dataset[index] for index in indices],
        recurrent_steps,
        "cpu",
        mapping_seed=mapping_seed,
        maximum_entities=maximum_entities,
        maximum_operations=cpu_cache.inputs["operation_mask"].shape[1],
        include_semantic_slot_orders=False,
    )
    passed = _batches_equal(cached, direct)
    if not passed:
        raise RuntimeError("A1.19H-H2 encoded cache is not numerically equivalent")
    return {
        "passed": passed,
        "probe_examples": len(indices),
        "input_tensors_checked": sorted(direct.inputs),
        "state_targets_checked": True,
        "answer_targets_checked": True,
    }


def _precompute_training_schedule(
    dataset: A119H2RecordSplit,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: str | torch.device,
) -> torch.Tensor:
    rng = random.Random(seed)
    schedule = [
        _sample_h2_indices(dataset, batch_size, rng) for _ in range(steps)
    ]
    return torch.tensor(schedule, dtype=torch.long, device=device)


@torch.inference_mode()
def _evaluate_encoded_a119h2_items(
    model: A119HHybridReasoner,
    encoded: A119HBatch,
    indices: Sequence[int],
    *,
    recurrent_steps: int,
    batch_size: int,
) -> dict[str, Any]:
    if not indices:
        raise ValueError("A1.19H-H2 evaluation requires indices")
    model.eval()
    device = encoded.state_targets.device
    counters = torch.zeros(6, dtype=torch.long, device=device)
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.tensor(
            indices[start : start + batch_size], dtype=torch.long, device=device
        )
        batch = _slice_a119h_batch_steps(
            index_a119h_batch(encoded, batch_indices), recurrent_steps
        )
        output = model(**batch.inputs, recurrent_steps=recurrent_steps)
        prediction = output["state_logits"].argmax(dim=-1)
        active = batch.state_targets != -100
        equal = (prediction == batch.state_targets) | ~active
        answer_prediction = output["answer_logits"].argmax(dim=-1)
        query_weights = (
            (
                batch.inputs["entity_handles"]
                == batch.inputs["query_handle"].unsqueeze(-1)
            )
            & batch.inputs["entity_mask"]
        ).long()
        state_answer = (prediction[:, -1] * query_weights).sum(dim=-1)
        counters += torch.stack(
            (
                equal.all(dim=(-1, -2)).sum(),
                equal[:, -1].all(dim=-1).sum(),
                ((prediction == batch.state_targets) & active).sum(),
                active.sum(),
                (answer_prediction == batch.answer_targets).sum(),
                (answer_prediction == state_answer).sum(),
            )
        )
    values = counters.cpu().tolist()
    examples = len(indices)
    return {
        "examples": examples,
        "recurrent_steps": recurrent_steps,
        "trajectory_full_exact": values[0] / examples,
        "final_state_full_exact": values[1] / examples,
        "state_token_accuracy": values[2] / max(1, values[3]),
        "final_answer_accuracy": values[4] / examples,
        "answer_state_prediction_consistency": values[5] / examples,
        "disable_recurrence": False,
        "handle_alias_seed": None,
    }


@torch.inference_mode()
def evaluate_a119h2_matrix(
    model: A119HHybridReasoner,
    dataset: A119H2RecordSplit,
    device: str | torch.device,
    *,
    mapping_seed: int,
    batch_size: int = 32,
    indices: Iterable[int] | None = None,
    encoded: A119HBatch | None = None,
) -> dict[str, Any]:
    selected = list(range(len(dataset))) if indices is None else list(indices)
    if encoded is None:
        maximum_steps = max(
            TRAIN_RECURRENT_STEPS,
            max(int(dataset[index]["program_length"]) for index in selected),
        )
        _, encoded, _ = _encode_a119h2_cache(
            dataset,
            device,
            mapping_seed=mapping_seed,
            maximum_entities=model.config.maximum_entities,
            recurrent_steps=maximum_steps,
        )
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in selected:
        record = dataset[index]
        cells[(int(record["entity_count"]), int(record["program_length"]))].append(index)
    by_cell: dict[str, Any] = {}
    by_count_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_length_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (count, length), cell_indices in sorted(cells.items()):
        metrics = _evaluate_encoded_a119h2_items(
            model,
            encoded,
            cell_indices,
            recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
            batch_size=batch_size,
        )
        by_cell[f"N{count}-T{length}"] = metrics
        by_count_rows[count].append(metrics)
        by_length_rows[length].append(metrics)
    return {
        "aggregate": _weighted(list(by_cell.values())),
        "by_entity_count": {
            str(count): _weighted(rows) for count, rows in sorted(by_count_rows.items())
        },
        "by_length": {
            str(length): _weighted(rows) for length, rows in sorted(by_length_rows.items())
        },
        "by_cell": by_cell,
    }


def _validation_score(metrics: dict[str, Any]) -> tuple[tuple[float, ...], dict[str, float]]:
    rows = list(metrics["by_cell"].values())
    components = {
        "minimum_cell_trajectory": min(row["trajectory_full_exact"] for row in rows),
        "minimum_cell_final_state": min(row["final_state_full_exact"] for row in rows),
        "aggregate_state_token": metrics["aggregate"]["state_token_accuracy"],
    }
    return tuple(components.values()), components


def _checkpoint(
    model: A119HHybridReasoner,
    optimizer: torch.optim.Optimizer,
    step: int,
    spec: A119H2TrainSpec,
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


def _export_deployment(checkpoint: dict[str, Any], output_path: Path) -> dict[str, Any]:
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
        raise RuntimeError("A1.19H-H2 deployment integrity failed")
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


def train_a119h_h2(
    data_dir: Path,
    output_dir: Path,
    *,
    spec: A119H2TrainSpec,
    device: str = "cuda",
    overfit_mode: bool = False,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.19H-H2 requested CUDA but CUDA is unavailable")
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.19H-H2 checkpoints: {output_dir}")
    if spec.train_recurrent_steps != TRAIN_RECURRENT_STEPS:
        raise ValueError("A1.19H-H2 recurrent steps are frozen at 16")
    if spec.auxiliary_weight != 1.0:
        raise ValueError("A1.19H-H2 auxiliary weight is frozen at 1.0")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATA_SCHEMA:
        raise ValueError("A1.19H-H2 requires its variable-cardinality data schema")
    if int(manifest["data_seed"]) != spec.data_seed:
        raise ValueError("A1.19H-H2 spec/data seed mismatch")
    manifest_hash = file_sha256(manifest_path)
    if overfit_mode:
        selected = _select_overfit32(load_a119h2_records(data_dir, "train"))
        train_dataset = validation_dataset = A119H2RecordSplit(selected)
        selection = "N2-N4-length-cell-balanced-overfit32-fit-only"
    else:
        train_dataset = A119H2RecordSplit(load_a119h2_records(data_dir, "train"))
        validation_dataset = A119H2RecordSplit(
            load_a119h2_records(data_dir, "validation")
        )
        selection = "formal-balanced-N2-N4-T1-T16-fixed-budget"

    _write_json(
        output_dir / "progress.json",
        {
            "stage": "A1.19H-H2",
            "phase": "building_validated_tensor_cache",
            "step": 0,
            "steps_requested": spec.steps,
            "note": "GPU power may remain low during one-time CPU encoding",
        },
    )
    cache_started = time.perf_counter()
    train_cpu_cache, train_cache, train_cache_report = _encode_a119h2_cache(
        train_dataset,
        device,
        mapping_seed=spec.mapping_seed,
        maximum_entities=5,
        recurrent_steps=TRAIN_RECURRENT_STEPS,
    )
    if validation_dataset is train_dataset:
        validation_cpu_cache = train_cpu_cache
        validation_cache = train_cache
        validation_cache_report = dict(train_cache_report)
        validation_cache_report["shared_with_training_cache"] = True
    else:
        (
            validation_cpu_cache,
            validation_cache,
            validation_cache_report,
        ) = _encode_a119h2_cache(
            validation_dataset,
            device,
            mapping_seed=spec.mapping_seed,
            maximum_entities=5,
            recurrent_steps=TRAIN_RECURRENT_STEPS,
        )
        validation_cache_report["shared_with_training_cache"] = False
    probe_indices = list(range(min(spec.batch_size, len(train_dataset))))
    cache_equivalence = _verify_cache_equivalence(
        train_dataset,
        train_cpu_cache,
        probe_indices,
        mapping_seed=spec.mapping_seed,
        maximum_entities=5,
        recurrent_steps=TRAIN_RECURRENT_STEPS,
    )
    cache_total_seconds = time.perf_counter() - cache_started
    _write_json(
        output_dir / "progress.json",
        {
            "stage": "A1.19H-H2",
            "phase": "cache_ready_initializing_training",
            "step": 0,
            "steps_requested": spec.steps,
            "cache_build_seconds": cache_total_seconds,
            "cache_equivalence_passed": cache_equivalence["passed"],
        },
    )

    random.seed(spec.model_seed)
    torch.manual_seed(spec.model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec.model_seed)
    model = A119HHybridReasoner(
        A119HConfig(maximum_entities=5, workspace_slots=10)
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate)
    training_schedule = _precompute_training_schedule(
        train_dataset,
        steps=spec.steps,
        batch_size=spec.batch_size,
        seed=spec.model_seed,
        device=device,
    )
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_step: int | None = None
    best_validation: dict[str, Any] | None = None
    ready_streak = 0
    completed_step = 0
    _synchronize(device)
    started = time.perf_counter()
    for step in range(1, spec.steps + 1):
        model.train()
        batch = index_a119h_batch(
            train_cache, training_schedule[step - 1]
        )
        output = model(**batch.inputs, recurrent_steps=TRAIN_RECURRENT_STEPS)
        loss, loss_components = _a119h_loss_tensors(
            model, output, batch, auxiliary_weight=spec.auxiliary_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        completed_step = step
        if step % spec.validation_interval == 0 or step == spec.steps:
            metrics = _loss_metrics(
                loss,
                loss_components,
                gradient_norm,
                spec.auxiliary_weight,
            )
            metrics["step"] = step
            validation = evaluate_a119h2_matrix(
                model,
                validation_dataset,
                device,
                mapping_seed=spec.mapping_seed,
                batch_size=spec.batch_size,
                encoded=validation_cache,
            )
            score, components = _validation_score(validation)
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
                    "stage": "A1.19H-H2",
                    "phase": "training",
                    "step": step,
                    "steps_requested": spec.steps,
                    "best_step": best_step,
                    "best_score": best_score,
                    "fixed_budget_formal": not overfit_mode,
                    "elapsed_seconds": time.perf_counter() - started,
                    "cache_equivalence_passed": cache_equivalence["passed"],
                },
            )
            history.append(metrics)
            if overfit_mode and ready_streak >= 2:
                break
    _synchronize(device)
    training_seconds = time.perf_counter() - started
    _write_json(output_dir / "history.json", history)
    best_checkpoint = torch.load(
        output_dir / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    validation = evaluate_a119h2_matrix(
        model,
        validation_dataset,
        device,
        mapping_seed=spec.mapping_seed,
        batch_size=spec.batch_size,
        encoded=validation_cache,
    )
    fit_diagnostic = evaluate_a119h2_matrix(
        model,
        train_dataset,
        device,
        mapping_seed=spec.mapping_seed,
        batch_size=spec.batch_size,
        indices=range(min(len(train_dataset), 1536)),
        encoded=train_cache,
    )
    deployment = _export_deployment(best_checkpoint, output_dir / "deployable.pt")
    overfit_gates = {
        "all_cells_trajectory_exact": all(
            row["trajectory_full_exact"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_cells_final_state_exact": all(
            row["final_state_full_exact"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_cells_answer_exact": all(
            row["final_answer_accuracy"] == 1.0
            for row in validation["by_cell"].values()
        ),
        "all_training_entity_counts_present": set(validation["by_entity_count"])
        == {"2", "3", "4"},
        "training_architecture_integrity": model.integrity_report()["passed"],
        "deployment_auxiliary_stripped": (
            deployment["integrity"]["training_auxiliary"] == "none"
            and deployment["integrity"]["training_auxiliary_outputs_per_entity"] == 0
        ),
    }
    result = {
        "schema_version": RESULTS_SCHEMA,
        "stage": "V2-A1.19H-H2 variable-cardinality hybrid core",
        "selection": selection,
        "train_spec": asdict(spec),
        "model_config": asdict(model.config),
        "training_entity_counts": list(TRAIN_ENTITY_COUNTS),
        "heldout_entity_count": HELDOUT_ENTITY_COUNT,
        "objective_contract": {
            "state_loss": "per-step shared-slot CE with inactive-slot mask",
            "closure_loss": "per-step shared value-prototype cosine",
            "official_answer_loss_weight": 0.0,
            "training_auxiliary": "per-step set-equivariant global-context state CE",
            "checkpoint_selection": "minimum N-by-length cell state metrics only",
            "deployment_auxiliary_policy": "physically stripped before formal",
        },
        "training": {
            "steps_requested": spec.steps,
            "steps_completed": completed_step,
            "processed_examples": completed_step * spec.batch_size,
            "processed_recurrent_transitions": (
                completed_step * spec.batch_size * TRAIN_RECURRENT_STEPS
            ),
            "seconds": training_seconds,
            "steps_per_second": completed_step / max(training_seconds, 1e-9),
            "fresh_initialization": True,
            "overfit_mode": overfit_mode,
            "fixed_budget_completed": overfit_mode or completed_step == spec.steps,
            "pipeline": "preencoded-GPU-cache-index-select",
            "per_step_cuda_scalar_writes": 0,
            "per_step_python_cuda_address_syncs": 0,
            "metrics_sync_policy": "validation-interval-only",
        },
        "cache": {
            "total_build_seconds": cache_total_seconds,
            "training": train_cache_report,
            "validation": validation_cache_report,
            "equivalence": cache_equivalence,
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


def _legacy_hot_path_address_checks(
    batch: A119HBatch, recurrent_steps: int
) -> None:
    """Reproduce the removed per-step CUDA synchronization for benchmarking."""

    handles = batch.inputs["entity_handles"]
    entity_mask = batch.inputs["entity_mask"]
    operation_mask = batch.inputs["operation_mask"]
    operation_steps = operation_mask.shape[1]
    for step in range(recurrent_steps):
        index = min(step, operation_steps - 1)
        active = (
            operation_mask[:, index]
            if step < operation_steps
            else torch.zeros_like(operation_mask[:, 0])
        )
        source_weights = (
            (
                handles
                == batch.inputs["operation_source_handle"][:, index].unsqueeze(-1)
            )
            & entity_mask
        )
        target_weights = (
            (
                handles
                == batch.inputs["operation_target_handle"][:, index].unsqueeze(-1)
            )
            & entity_mask
        )
        if torch.any(active & (source_weights.sum(dim=-1) != 1)):
            raise ValueError("legacy source-address check failed")
        if torch.any(active & (target_weights.sum(dim=-1) != 1)):
            raise ValueError("legacy target-address check failed")
    query_weights = (
        handles == batch.inputs["query_handle"].unsqueeze(-1)
    ) & entity_mask
    if torch.any(query_weights.sum(dim=-1) != 1):
        raise ValueError("legacy query-address check failed")


def benchmark_a119h2_training_pipeline(
    data_dir: Path,
    output_path: Path,
    *,
    model_seed: int,
    mapping_seed: int,
    steps: int = 60,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("A1.19H-H2 benchmark steps must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.19H-H2 benchmark requested unavailable CUDA")
    dataset = A119H2RecordSplit(load_a119h2_records(data_dir, "train"))
    rng = random.Random(model_seed)
    schedule = [
        _sample_h2_indices(dataset, batch_size, rng) for _ in range(steps)
    ]
    cpu_cache, device_cache, cache_report = _encode_a119h2_cache(
        dataset,
        device,
        mapping_seed=mapping_seed,
        maximum_entities=5,
        recurrent_steps=TRAIN_RECURRENT_STEPS,
    )
    cache_equivalence = _verify_cache_equivalence(
        dataset,
        cpu_cache,
        schedule[0],
        mapping_seed=mapping_seed,
        maximum_entities=5,
        recurrent_steps=TRAIN_RECURRENT_STEPS,
    )
    device_schedule = torch.tensor(schedule, dtype=torch.long, device=device)
    torch.manual_seed(model_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(model_seed)
    template = A119HHybridReasoner(
        A119HConfig(maximum_entities=5, workspace_slots=10)
    )
    initial_state = deepcopy(template.state_dict())

    def run_pipeline(legacy: bool) -> tuple[float, float, dict[str, torch.Tensor]]:
        model = A119HHybridReasoner(
            A119HConfig(maximum_entities=5, workspace_slots=10)
        ).to(device)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        _synchronize(device)
        started = time.perf_counter()
        last_loss: torch.Tensor | None = None
        for step, indices in enumerate(schedule):
            model.train()
            if legacy:
                batch = encode_a119h_records(
                    [dataset[index] for index in indices],
                    TRAIN_RECURRENT_STEPS,
                    device,
                    mapping_seed=mapping_seed,
                    maximum_entities=5,
                    maximum_operations=TRAIN_RECURRENT_STEPS,
                    include_semantic_slot_orders=False,
                )
                _legacy_hot_path_address_checks(batch, TRAIN_RECURRENT_STEPS)
            else:
                batch = index_a119h_batch(device_cache, device_schedule[step])
            output = model(**batch.inputs, recurrent_steps=TRAIN_RECURRENT_STEPS)
            last_loss, _ = _a119h_loss_tensors(
                model, output, batch, auxiliary_weight=1.0
            )
            optimizer.zero_grad(set_to_none=True)
            last_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        assert last_loss is not None
        final_loss = float(last_loss.detach())
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        return elapsed, final_loss, state

    legacy_seconds, legacy_loss, legacy_state = run_pipeline(True)
    cached_seconds, cached_loss, cached_state = run_pipeline(False)
    parameter_max_abs_delta = max(
        float((legacy_state[name] - cached_state[name]).abs().max())
        for name in legacy_state
    )
    result = {
        "schema_version": BENCHMARK_SCHEMA,
        "data_dir": str(data_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "model_seed": model_seed,
        "mapping_seed": mapping_seed,
        "steps": steps,
        "batch_size": batch_size,
        "recurrent_steps": TRAIN_RECURRENT_STEPS,
        "contract": {
            "precision": "fp32",
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "gradient_clip": 1.0,
            "batch_sequence_identical": True,
            "objective_identical": True,
        },
        "legacy": {
            "seconds": legacy_seconds,
            "steps_per_second": steps / legacy_seconds,
            "pipeline": "per-step Python-to-CUDA encoding plus recurrent address sync",
            "final_loss": legacy_loss,
        },
        "optimized": {
            "seconds": cached_seconds,
            "steps_per_second": steps / cached_seconds,
            "pipeline": "validated preencoded GPU cache plus index-select",
            "final_loss": cached_loss,
            "cache": cache_report,
        },
        "speedup": legacy_seconds / cached_seconds,
        "equivalence": {
            "cache_encoding": cache_equivalence,
            "final_loss_abs_delta": abs(legacy_loss - cached_loss),
            "final_parameter_max_abs_delta": parameter_max_abs_delta,
            "passed": (
                cache_equivalence["passed"]
                and abs(legacy_loss - cached_loss) == 0.0
                and parameter_max_abs_delta == 0.0
            ),
        },
    }
    _write_json(output_path, result)
    return result


def load_a119h2_deployment(
    checkpoint_path: Path, device: str | torch.device = "cpu"
) -> A119HHybridReasoner:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("A1.19H-H2 formal requires its stripped deployment")
    if checkpoint["model_config"].get("training_auxiliary") != "none":
        raise ValueError("A1.19H-H2 deployment still declares an auxiliary")
    if any(
        name.startswith("training_auxiliary_head.") for name in checkpoint["model"]
    ):
        raise ValueError("A1.19H-H2 deployment still contains auxiliary parameters")
    model = A119HHybridReasoner(A119HConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.a119h_model_seed = int(checkpoint["train_spec"]["model_seed"])
    model.a119h_data_seed = int(checkpoint["train_spec"]["data_seed"])
    model.a119h_mapping_seed = int(checkpoint["train_spec"]["mapping_seed"])
    model.a119h_data_manifest_sha256 = checkpoint["data_manifest_sha256"]
    model.a119h_deployment_sha256 = file_sha256(checkpoint_path)
    model.eval()
    return model


def _all_cells(metrics: dict[str, Any], metric: str, threshold: float) -> bool:
    return all(row[metric] >= threshold for row in metrics["by_cell"].values())


@torch.no_grad()
def evaluate_a119h2_formal(
    checkpoint: Path,
    data_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, Any]:
    model = load_a119h2_deployment(checkpoint, device)
    results = {
        split: evaluate_a119h2_matrix(
            model,
            A119H2RecordSplit(load_a119h2_records(data_dir, split)),
            device,
            mapping_seed=model.a119h_mapping_seed,
            batch_size=batch_size,
        )
        for split in FORMAL_SPLITS
    }
    gates: dict[str, bool] = {}
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
    for split, threshold in thresholds.items():
        for metric in (
            "trajectory_full_exact",
            "final_state_full_exact",
            "final_answer_accuracy",
        ):
            gates[f"{split}_{metric}_gate"] = _all_cells(
                results[split], metric, threshold
            )
    integrity = model.integrity_report()
    gates.update(
        {
            "all_state_token_at_least_0_995": all(
                results[split]["aggregate"]["state_token_accuracy"] >= 0.995
                for split in FORMAL_SPLITS
            ),
            "training_counts_present_in_supported_splits": all(
                set(results[split]["by_entity_count"]) == {"2", "3", "4"}
                for split in (
                    "short_regression",
                    "supported_in_range",
                    "relation_in_range",
                    "supported_ood",
                    "relation_ood",
                    "causal_core",
                )
            ),
            "heldout_count_is_exactly_n5": all(
                set(results[split]["by_entity_count"]) == {"5"}
                for split in ("entity_heldout", "entity_relation_heldout")
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
        "stage": "V2-A1.19H-H2 variable-cardinality hybrid core",
        "model_seed": model.a119h_model_seed,
        "data_seed": model.a119h_data_seed,
        "mapping_seed": model.a119h_mapping_seed,
        "deployment_integrity": integrity,
        "splits": results,
        "gates": gates,
        "passed": all(gates.values()),
    }
