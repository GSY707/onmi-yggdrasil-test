from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .a1_7_data import VALUE_LABELS, execute_operations
from .a1_9_cache import A19CachedSplit
from .a1_9_train import evaluate_a19_items, load_a19_checkpoint


INTERVENTION_SCHEMA = "yggdrasil.v2-a1.9.hidden-interventions.v1"


def clone_recomputed_record(
    record: dict[str, Any],
    *,
    operations: Sequence[dict[str, Any]] | None = None,
    query_register: str | None = None,
) -> dict[str, Any]:
    clone = dict(record)
    clone_operations = [dict(operation) for operation in (record["operations"] if operations is None else operations)]
    if not clone_operations:
        raise ValueError("A1.9 interventions keep at least one operation")
    clone["operations"] = clone_operations
    clone["program_length"] = len(clone_operations)
    clone["operation_mask"] = [True] * len(clone_operations)
    clone["query_register"] = record["query_register"] if query_register is None else query_register
    trajectory = execute_operations(clone["start_state"], clone_operations)
    clone["state_trajectory"] = trajectory[1:]
    clone["answer"] = trajectory[-1][clone["query_register"]]
    clone["answer_index"] = VALUE_LABELS.index(clone["answer"])
    return clone


def _clone_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_hidden": item["start_hidden"].clone(),
        "query_hidden": item["query_hidden"].clone(),
        "operation_hidden": item["operation_hidden"].clone(),
        "record": item["record"],
    }


def _joint(operation: dict[str, Any]) -> tuple[str, str, str]:
    return operation["family"], operation["source"], operation["target"]


def _trajectory_changed(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["state_trajectory"] != right["state_trajectory"]


def build_a19_interventions(items: Sequence[dict[str, Any]], seed: int = 20260715) -> dict[str, list[dict[str, Any]]]:
    if not items:
        raise ValueError("A1.9 interventions require cached causal items")
    rng = random.Random(seed)
    prefix: list[dict[str, Any]] = []
    deletion: list[dict[str, Any]] = []
    shuffled_steps: list[dict[str, Any]] = []
    replacement: list[dict[str, Any]] = []
    query_swap: list[dict[str, Any]] = []
    same_answer: list[dict[str, Any]] = []

    carriers: dict[tuple[str, str, str], list[tuple[torch.Tensor, dict[str, Any]]]] = defaultdict(list)
    flat_channels = {"family": [], "source": [], "target": []}
    for item in items:
        for step, operation in enumerate(item["record"]["operations"]):
            row = item["operation_hidden"][step]
            carriers[_joint(operation)].append((row, operation))
            flat_channels["family"].append(row[0])
            flat_channels["source"].append(row[1])
            flat_channels["target"].append(row[2])
    carrier_joints = sorted(carriers)

    for item_index, item in enumerate(items):
        record = item["record"]
        length = int(record["program_length"])
        for prefix_length in range(1, length + 1):
            modified = _clone_item(item)
            modified["operation_hidden"] = item["operation_hidden"][:prefix_length].clone()
            modified["record"] = clone_recomputed_record(record, operations=record["operations"][:prefix_length])
            prefix.append(modified)
        for step in range(length):
            if length > 1:
                modified = _clone_item(item)
                keep = [index for index in range(length) if index != step]
                modified["operation_hidden"] = item["operation_hidden"][keep].clone()
                modified["record"] = clone_recomputed_record(record, operations=[record["operations"][index] for index in keep])
                deletion.append(modified)
            candidates = [joint for joint in carrier_joints if joint != _joint(record["operations"][step])]
            rng.shuffle(candidates)
            for candidate in candidates:
                donor_hidden, donor_operation = rng.choice(carriers[candidate])
                operations = [dict(operation) for operation in record["operations"]]
                operations[step] = dict(donor_operation)
                modified_record = clone_recomputed_record(record, operations=operations)
                if not _trajectory_changed(record, modified_record):
                    continue
                modified = _clone_item(item)
                modified["operation_hidden"][step] = donor_hidden
                modified["record"] = modified_record
                modified["old_record"] = record
                replacement.append(modified)
                break
        if length > 1:
            order = list(range(1, length)) + [0]
            modified_record = clone_recomputed_record(record, operations=[record["operations"][index] for index in order])
            if _trajectory_changed(record, modified_record):
                modified = _clone_item(item)
                modified["operation_hidden"] = item["operation_hidden"][order].clone()
                modified["record"] = modified_record
                modified["old_record"] = record
                shuffled_steps.append(modified)

        donor_query_index = next(
            (offset for offset in range(1, len(items)) if items[(item_index + offset) % len(items)]["record"]["query_register"] != record["query_register"]),
            None,
        )
        if donor_query_index is not None:
            donor = items[(item_index + donor_query_index) % len(items)]
            modified = _clone_item(item)
            modified["query_hidden"] = donor["query_hidden"].clone()
            modified["record"] = clone_recomputed_record(record, query_register=donor["record"]["query_register"])
            query_swap.append(modified)

        candidate_steps = list(range(length))
        rng.shuffle(candidate_steps)
        found_same_answer = False
        for step in candidate_steps:
            candidates = [joint for joint in carrier_joints if joint != _joint(record["operations"][step])]
            rng.shuffle(candidates)
            for candidate in candidates:
                donor_hidden, donor_operation = rng.choice(carriers[candidate])
                operations = [dict(operation) for operation in record["operations"]]
                operations[step] = dict(donor_operation)
                modified_record = clone_recomputed_record(record, operations=operations)
                if modified_record["answer"] != record["answer"] or not _trajectory_changed(record, modified_record):
                    continue
                modified = _clone_item(item)
                modified["operation_hidden"][step] = donor_hidden
                modified["record"] = modified_record
                modified["old_record"] = record
                same_answer.append(modified)
                found_same_answer = True
                break
            if found_same_answer:
                break

    no_hidden: list[dict[str, Any]] = []
    independent_shuffle: list[dict[str, Any]] = []
    total_operations = sum(int(item["operation_hidden"].shape[0]) for item in items)
    family_cursor = source_cursor = target_cursor = 0
    for index, item in enumerate(items):
        zeroed = _clone_item(item)
        zeroed["start_hidden"].zero_()
        zeroed["query_hidden"].zero_()
        zeroed["operation_hidden"].zero_()
        no_hidden.append(zeroed)

        shuffled = _clone_item(item)
        shuffled["start_hidden"] = items[(index + 1) % len(items)]["start_hidden"].clone()
        shuffled["query_hidden"] = items[(index + 2) % len(items)]["query_hidden"].clone()
        length = int(item["operation_hidden"].shape[0])
        for step in range(length):
            shuffled["operation_hidden"][step, 0] = flat_channels["family"][(family_cursor + 1) % total_operations]
            shuffled["operation_hidden"][step, 1] = flat_channels["source"][(source_cursor + 3) % total_operations]
            shuffled["operation_hidden"][step, 2] = flat_channels["target"][(target_cursor + 5) % total_operations]
            family_cursor += 1
            source_cursor += 1
            target_cursor += 1
        independent_shuffle.append(shuffled)

    return {
        "prefix": prefix,
        "replacement": replacement,
        "deletion": deletion,
        "operation_shuffle": shuffled_steps,
        "query_swap": query_swap,
        "same_answer_different_trajectory": same_answer,
        "no_hidden": no_hidden,
        "independent_role_shuffle": independent_shuffle,
    }


@torch.no_grad()
def run_a19_interventions(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    formal_eval_path: Path,
    device: str = "cuda",
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.9 hidden interventions are forbidden before the formal Gate passes")
    model = load_a19_checkpoint(checkpoint, device)
    dataset = A19CachedSplit(cache_dir, data_dir, "causal_core")
    items = dataset.items()
    intervention_items = build_a19_interventions(items)
    results: dict[str, Any] = {}
    for name, rows in intervention_items.items():
        results[name] = evaluate_a19_items(
            model,
            rows,
            device,
            batch_size=batch_size,
            old_oracle=name in {"replacement", "operation_shuffle", "same_answer_different_trajectory"},
        )
    gates = {
        "prefix_trajectory_at_least_0_99": results["prefix"]["trajectory_full_exact"] >= 0.99,
        "replacement_trajectory_at_least_0_95": results["replacement"]["trajectory_full_exact"] >= 0.95,
        "deletion_trajectory_at_least_0_95": results["deletion"]["trajectory_full_exact"] >= 0.95,
        "operation_shuffle_trajectory_at_least_0_95": results["operation_shuffle"]["trajectory_full_exact"] >= 0.95,
        "query_swap_trajectory_invariance_at_least_0_99": results["query_swap"]["trajectory_full_exact"] >= 0.99,
        "query_swap_answer_at_least_0_95": results["query_swap"]["final_query_token_accuracy"] >= 0.95,
        "same_answer_examples_at_least_128": results["same_answer_different_trajectory"]["examples"] >= 128,
        "same_answer_new_trajectory_at_least_0_95": results["same_answer_different_trajectory"]["trajectory_full_exact"] >= 0.95,
        "no_hidden_trajectory_at_most_0_10": results["no_hidden"]["trajectory_full_exact"] <= 0.10,
        "independent_role_shuffle_trajectory_at_most_0_10": results["independent_role_shuffle"]["trajectory_full_exact"] <= 0.10,
    }
    return {
        "schema_version": INTERVENTION_SCHEMA,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "formal_eval": str(formal_eval_path),
        "adapter_seed": model.a19_adapter_seed,
        "data_seed": model.a19_data_seed,
        "core_seed": model.a19_core_seed,
        "method": {
            "role_hidden_counterfactuals": True,
            "modified_oracles_recomputed": True,
            "full_source_hidden_available": False,
            "same_answer_scores_full_trajectory_not_only_final_answer": True,
            "necessary_changed_trajectory_filter": True,
        },
        "results": results,
        "gates": gates,
        "passed": all(gates.values()),
    }
