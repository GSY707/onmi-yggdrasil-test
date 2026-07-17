from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .a1_7_data import OPERATION_FAMILIES
from .a1_10_train import TRAIN_RECURRENT_STEPS
from .a1_19h_data import encode_a119h_records, permute_entity_axis
from .a1_19h_h2_data import load_a119h2_records, recompute_a119h2_record
from .a1_19h_h2_train import load_a119h2_deployment
from .a1_19h_train import evaluate_a119h_items, _weighted


INTERVENTION_SCHEMA = "yggdrasil.v2-a1.19h.h2-variable-cardinality.interventions.v1"


def _trajectory_key(record: dict[str, Any]) -> str:
    return json.dumps(record["state_trajectory"], sort_keys=True)


def _counterfactuals(
    records: Sequence[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    variants: dict[str, list[dict[str, Any]]] = {
        "prefix": [],
        "replacement": [],
        "deletion": [],
        "operation_shuffle": [],
        "query_swap": [],
    }
    counters: dict[str, int] = defaultdict(int)
    for old in records:
        operations = [dict(operation) for operation in old["operations"]]
        old_key = _trajectory_key(old)
        for prefix_length in range(1, len(operations) + 1):
            new = recompute_a119h2_record(
                old,
                operations=operations[:prefix_length],
                split="h2_prefix",
                index=counters["prefix"],
            )
            counters["prefix"] += 1
            variants["prefix"].append({"record": new, "old_record": old})
        names = list(old["entity_names"])
        catalog = [
            {"family": family, "source": source, "target": target}
            for family in OPERATION_FAMILIES
            for source in names
            for target in names
            if source != target
        ]
        rng.shuffle(catalog)
        replacement_pair: dict[str, Any] | None = None
        for step in rng.sample(range(len(operations)), len(operations)):
            for replacement in catalog:
                if all(
                    replacement[key] == operations[step][key]
                    for key in ("family", "source", "target")
                ):
                    continue
                changed = list(operations)
                changed[step] = dict(replacement)
                new = recompute_a119h2_record(
                    old,
                    operations=changed,
                    split="h2_replacement",
                    index=counters["replacement"],
                )
                if _trajectory_key(new) != old_key:
                    replacement_pair = {"record": new, "old_record": old}
                    break
            if replacement_pair is not None:
                break
        if replacement_pair is None:
            raise RuntimeError("A1.19H-H2 could not build replacement counterfactual")
        counters["replacement"] += 1
        variants["replacement"].append(replacement_pair)
        deletion_pair: dict[str, Any] | None = None
        for step in rng.sample(range(len(operations)), len(operations)):
            changed = operations[:step] + operations[step + 1 :]
            if not changed:
                continue
            new = recompute_a119h2_record(
                old,
                operations=changed,
                split="h2_deletion",
                index=counters["deletion"],
            )
            if _trajectory_key(new) != old_key:
                deletion_pair = {"record": new, "old_record": old}
                break
        if deletion_pair is not None:
            counters["deletion"] += 1
            variants["deletion"].append(deletion_pair)
        shuffled_pair: dict[str, Any] | None = None
        for changed in (list(reversed(operations)), operations[1:] + operations[:1]):
            new = recompute_a119h2_record(
                old,
                operations=changed,
                split="h2_shuffle",
                index=counters["operation_shuffle"],
            )
            if _trajectory_key(new) != old_key:
                shuffled_pair = {"record": new, "old_record": old}
                break
        if shuffled_pair is not None:
            counters["operation_shuffle"] += 1
            variants["operation_shuffle"].append(shuffled_pair)
        alternatives = [name for name in names if name != old["query_register"]]
        new = recompute_a119h2_record(
            old,
            query_entity=rng.choice(alternatives),
            split="h2_query",
            index=counters["query_swap"],
        )
        counters["query_swap"] += 1
        variants["query_swap"].append({"record": new, "old_record": old})
    if not all(variants.values()):
        raise RuntimeError("A1.19H-H2 counterfactual construction left an empty arm")
    return variants


def _mapping_aligned_pairs(
    pairs: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    new_records: list[dict[str, Any]] = []
    old_records: list[dict[str, Any]] = []
    for pair in pairs:
        key = str(pair["old_record"]["fingerprint"])
        new = copy.deepcopy(pair["record"])
        old = copy.deepcopy(pair["old_record"])
        new["a119h_mapping_key"] = key
        old["a119h_mapping_key"] = key
        new_records.append(new)
        old_records.append(old)
    return new_records, old_records


def _evaluate_variant(
    model: Any,
    records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    mapping_seed: int,
    old_records: Sequence[dict[str, Any]] | None = None,
    disable_recurrence: bool = False,
    handle_alias_seed: int | None = None,
) -> dict[str, Any]:
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        cells[(int(record["entity_count"]), int(record["program_length"]))].append(index)
    by_cell: dict[str, Any] = {}
    for (count, length), indices in sorted(cells.items()):
        by_cell[f"N{count}-T{length}"] = evaluate_a119h_items(
            model,
            [records[index] for index in indices],
            device,
            mapping_seed=mapping_seed,
            recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
            batch_size=batch_size,
            old_records=(
                [old_records[index] for index in indices]
                if old_records is not None
                else None
            ),
            disable_recurrence=disable_recurrence,
            handle_alias_seed=handle_alias_seed,
        )
    return {"aggregate": _weighted(list(by_cell.values())), "by_cell": by_cell}


def _same_answer_pairs(
    records: Sequence[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[
            (
                int(record["entity_count"]),
                int(record["program_length"]),
                int(record["answer_index"]),
            )
        ].append(index)
    new_records: list[dict[str, Any]] = []
    old_records: list[dict[str, Any]] = []
    for indices in groups.values():
        candidates = list(indices)
        rng.shuffle(candidates)
        for target in indices:
            source = next(
                (
                    candidate
                    for candidate in candidates
                    if _trajectory_key(records[candidate])
                    != _trajectory_key(records[target])
                ),
                None,
            )
            if source is not None:
                new_records.append(records[source])
                old_records.append(records[target])
    if not new_records:
        raise RuntimeError("A1.19H-H2 same-answer intervention has no pairs")
    return new_records, old_records


def _duplicate_value_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        names = list(record["entity_names"])
        start = dict(record["start_state"])
        start[names[1]] = start[names[0]]
        new = recompute_a119h2_record(
            record,
            start_state=start,
            split="h2_same_value",
            index=index,
        )
        new["a119h_mapping_key"] = record["fingerprint"]
        result.append(new)
    return result


def _wrong_start_pairs(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    for index, old in enumerate(records):
        names = list(old["entity_names"])
        values = [old["start_state"][name] for name in names]
        rotated = values[1:] + values[:1]
        start = dict(zip(names, rotated))
        new = recompute_a119h2_record(
            old,
            start_state=start,
            split="h2_wrong_start",
            index=index,
        )
        if _trajectory_key(new) != _trajectory_key(old):
            pairs.append({"record": new, "old_record": old})
    if not pairs:
        raise RuntimeError("A1.19H-H2 wrong-start intervention has no changed pairs")
    return _mapping_aligned_pairs(pairs)


@torch.no_grad()
def _address_invariance(
    model: Any,
    records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    mapping_seed: int,
    alias_seed: int,
) -> dict[str, Any]:
    permutation = torch.arange(
        model.config.maximum_entities - 1, -1, -1, device=device
    )
    slot_state = slot_answer = alias_state = alias_answer = 0.0
    for start in range(0, len(records), batch_size):
        rows = list(records[start : start + batch_size])
        steps = max(
            TRAIN_RECURRENT_STEPS, max(int(row["program_length"]) for row in rows)
        )
        batch = encode_a119h_records(
            rows,
            steps,
            device,
            mapping_seed=mapping_seed,
            maximum_entities=model.config.maximum_entities,
        )
        base = model(**batch.inputs, recurrent_steps=steps)
        permuted_batch = permute_entity_axis(batch, permutation)
        permuted = model(**permuted_batch.inputs, recurrent_steps=steps)
        slot_state = max(
            slot_state,
            float(
                (
                    permuted["state_logits"]
                    - base["state_logits"][:, :, permutation]
                )
                .abs()
                .max()
            ),
        )
        slot_answer = max(
            slot_answer,
            float((permuted["answer_logits"] - base["answer_logits"]).abs().max()),
        )
        aliased_batch = encode_a119h_records(
            rows,
            steps,
            device,
            mapping_seed=mapping_seed,
            maximum_entities=model.config.maximum_entities,
            handle_alias_seed=alias_seed,
        )
        aliased = model(**aliased_batch.inputs, recurrent_steps=steps)
        alias_state = max(
            alias_state,
            float((aliased["state_logits"] - base["state_logits"]).abs().max()),
        )
        alias_answer = max(
            alias_answer,
            float((aliased["answer_logits"] - base["answer_logits"]).abs().max()),
        )
    return {
        "examples": len(records),
        "slot_permutation_state_logit_max_abs": slot_state,
        "slot_permutation_answer_logit_max_abs": slot_answer,
        "handle_alias_state_logit_max_abs": alias_state,
        "handle_alias_answer_logit_max_abs": alias_answer,
    }


@torch.no_grad()
def run_a119h_h2_interventions(
    checkpoint: Path,
    data_dir: Path,
    formal_eval_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 20261980,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.19H-H2 interventions require a passed formal run")
    model = load_a119h2_deployment(checkpoint, device)
    mapping_seed = model.a119h_mapping_seed
    causal = load_a119h2_records(data_dir, "causal_core")
    variants = _counterfactuals(causal, seed)
    results: dict[str, Any] = {}
    for name, pairs in variants.items():
        records, old = _mapping_aligned_pairs(pairs)
        results[name] = _evaluate_variant(
            model,
            records,
            device,
            batch_size,
            mapping_seed=mapping_seed,
            old_records=(
                old if name in {"replacement", "deletion", "operation_shuffle"} else None
            ),
        )
    same_new, same_old = _same_answer_pairs(causal, seed)
    same_pairs = [
        {"record": record, "old_record": old}
        for record, old in zip(same_new, same_old)
    ]
    same_new, same_old = _mapping_aligned_pairs(same_pairs)
    results["same_answer_different_trajectory"] = _evaluate_variant(
        model,
        same_new,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        old_records=same_old,
    )
    same_value = _evaluate_variant(
        model,
        _duplicate_value_records(causal),
        device,
        batch_size,
        mapping_seed=mapping_seed,
    )
    address_records = causal + load_a119h2_records(data_dir, "entity_heldout")[:256]
    handle_alias = _evaluate_variant(
        model,
        address_records,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        handle_alias_seed=seed + 1,
    )
    invariance = _address_invariance(
        model,
        address_records,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        alias_seed=seed + 1,
    )
    disable_recurrence = _evaluate_variant(
        model,
        causal,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        disable_recurrence=True,
    )
    wrong_new, wrong_old = _wrong_start_pairs(causal)
    wrong_start = _evaluate_variant(
        model,
        wrong_new,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        old_records=wrong_old,
    )
    changed = (
        "replacement",
        "deletion",
        "operation_shuffle",
        "same_answer_different_trajectory",
    )
    gates = {
        "structural_counterfactual_trajectory_at_least_0_95": all(
            results[name]["aggregate"]["trajectory_full_exact"] >= 0.95
            for name in ("prefix", "replacement", "deletion", "operation_shuffle")
        ),
        "query_swap_answer_at_least_0_95": results["query_swap"]["aggregate"]["final_answer_accuracy"] >= 0.95,
        "same_answer_new_trajectory_at_least_0_95": results["same_answer_different_trajectory"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "changed_old_trajectory_at_most_0_05": all(
            results[name]["aggregate"]["old_oracle_trajectory_full_exact"] <= 0.05
            for name in changed
        ),
        "same_value_different_entity_trajectory_at_least_0.95": same_value["aggregate"]["trajectory_full_exact"] >= 0.95,
        "handle_alias_all_counts_trajectory_at_least_0.95": handle_alias["aggregate"]["trajectory_full_exact"] >= 0.95,
        "slot_permutation_equivariance_below_1e_6": max(
            invariance["slot_permutation_state_logit_max_abs"],
            invariance["slot_permutation_answer_logit_max_abs"],
        ) < 1e-6,
        "handle_alias_invariance_below_1e_6": max(
            invariance["handle_alias_state_logit_max_abs"],
            invariance["handle_alias_answer_logit_max_abs"],
        ) < 1e-6,
        "disable_recurrence_at_most_0.20": disable_recurrence["aggregate"]["trajectory_full_exact"] <= 0.20,
        "wrong_start_old_trajectory_at_most_0.20": wrong_start["aggregate"]["old_oracle_trajectory_full_exact"] <= 0.20,
        "deployment_auxiliary_absent": (
            model.integrity_report()["training_auxiliary"] == "none"
            and model.training_auxiliary_head is None
        ),
    }
    return {
        "schema_version": INTERVENTION_SCHEMA,
        "checkpoint": str(checkpoint),
        "deployment_sha256": model.a119h_deployment_sha256,
        "formal_eval": str(formal_eval_path),
        "results": results,
        "same_value_different_entity": same_value,
        "handle_alias_all_counts": handle_alias,
        "address_invariance": invariance,
        "disable_recurrence": disable_recurrence,
        "wrong_start_state": wrong_start,
        "gates": gates,
        "passed": all(gates.values()),
    }
