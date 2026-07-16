from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .a1_10_cache import A110CachedSplit
from .a1_10_interventions import build_a110_counterfactual_records, load_a110_intervention_items
from .a1_10_train import TRAIN_RECURRENT_STEPS
from .a1_8_data import load_a18_records
from .a1_11_train import (
    evaluate_a111_boundary_items,
    evaluate_a111_reasoner_items,
    load_a111_boundary_checkpoint,
    load_a111_reasoner_checkpoint,
)


BOUNDARY_INTERVENTION_SCHEMA = "yggdrasil.v2-a1.11.boundary-interventions.v1"
REASONER_INTERVENTION_SCHEMA = "yggdrasil.v2-a1.11.exact-symbolic-reasoner-interventions.v2"


def _weighted(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["examples"] for row in rows)
    result: dict[str, Any] = {"examples": total}
    for metric in (
        "trajectory_full_exact",
        "final_state_full_exact",
        "final_query_token_accuracy",
        "state_token_accuracy",
        "source_pointer_accuracy",
        "target_pointer_accuracy",
        "final_answer_accuracy",
        "answer_state_prediction_consistency",
        "old_oracle_trajectory_full_exact",
    ):
        available = [row for row in rows if metric in row]
        if available:
            denominator = sum(row["examples"] for row in available)
            result[metric] = sum(row[metric] * row["examples"] for row in available) / max(1, denominator)
    return result


def _trajectory_key(record: dict[str, Any]) -> str:
    return json.dumps(record["state_trajectory"], sort_keys=True)


def _same_answer_pairs(
    items: Sequence[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[(int(item["record"]["answer_index"]), int(item["record"]["program_length"]))].append(index)
    new_items: list[dict[str, Any]] = []
    old_records: list[dict[str, Any]] = []
    for indexes in groups.values():
        shuffled = list(indexes)
        rng.shuffle(shuffled)
        for target in indexes:
            source = next(
                (
                    candidate
                    for candidate in shuffled
                    if _trajectory_key(items[candidate]["record"]) != _trajectory_key(items[target]["record"])
                ),
                None,
            )
            if source is not None:
                new_items.append(items[source])
                old_records.append(items[target]["record"])
    if not new_items:
        raise RuntimeError("A1.11 same-answer intervention found no eligible pair")
    return new_items, old_records


def _evaluate_boundary_variant(
    model: Any,
    items: Sequence[dict[str, Any]],
    old_records: Sequence[dict[str, Any]] | None,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[int(item["record"]["program_length"])].append(index)
    by_length: dict[str, Any] = {}
    for length, indexes in sorted(groups.items()):
        by_length[str(length)] = evaluate_a111_boundary_items(
            model,
            [items[index] for index in indexes],
            device,
            batch_size=batch_size,
            old_records=[old_records[index] for index in indexes] if old_records is not None else None,
        )
    return {"aggregate": _weighted(list(by_length.values())), "by_length": by_length}


@torch.no_grad()
def run_a111_boundary_interventions(
    checkpoint: Path,
    source_cache_dir: Path,
    intervention_cache_dir: Path,
    data_dir: Path,
    formal_eval_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 20261120,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.11 Boundary interventions are forbidden before formal passes")
    model = load_a111_boundary_checkpoint(checkpoint, device)
    text_variants: dict[str, Any] = {}
    for variant in ("prefix", "replacement", "deletion", "operation_shuffle", "query_swap"):
        items, old_records = load_a110_intervention_items(intervention_cache_dir, variant)
        text_variants[variant] = _evaluate_boundary_variant(
            model, items, old_records, device, batch_size
        )
    causal_dataset = A110CachedSplit(source_cache_dir, data_dir, "causal_core")
    causal_items = causal_dataset.items()
    same_items, same_old = _same_answer_pairs(causal_items, seed)
    same_answer = _evaluate_boundary_variant(model, same_items, same_old, device, batch_size)
    no_source_items = [
        {**item, "last_hidden": torch.zeros_like(item["last_hidden"])} for item in causal_items
    ]
    no_source = _evaluate_boundary_variant(model, no_source_items, None, device, batch_size)
    permutation = list(range(len(causal_items)))
    random.Random(seed).shuffle(permutation)
    shuffled_items = [
        {
            "last_hidden": causal_items[source]["last_hidden"],
            "attention_mask": causal_items[source]["attention_mask"],
            "record": causal_items[target]["record"],
        }
        for target, source in enumerate(permutation)
    ]
    independent_shuffle = _evaluate_boundary_variant(model, shuffled_items, None, device, batch_size)
    gates = {
        "text_counterfactuals_follow_new_trajectory_at_least_0_95": all(
            text_variants[name]["aggregate"]["trajectory_full_exact"] >= 0.95
            for name in ("prefix", "replacement", "deletion", "operation_shuffle")
        ),
        "query_swap_new_answer_at_least_0_95": text_variants["query_swap"]["aggregate"]["final_query_token_accuracy"] >= 0.95,
        "changed_counterfactual_old_trajectory_at_most_0_05": all(
            text_variants[name]["aggregate"]["old_oracle_trajectory_full_exact"] <= 0.05
            for name in ("replacement", "deletion", "operation_shuffle")
        ),
        "same_answer_new_trajectory_at_least_0_95": same_answer["aggregate"]["trajectory_full_exact"] >= 0.95,
        "same_answer_old_trajectory_at_most_0_05": same_answer["aggregate"]["old_oracle_trajectory_full_exact"] <= 0.05,
        "no_source_trajectory_at_most_0_20": no_source["aggregate"]["trajectory_full_exact"] <= 0.20,
        "independent_source_shuffle_at_most_0_20": independent_shuffle["aggregate"]["trajectory_full_exact"] <= 0.20,
    }
    return {
        "schema_version": BOUNDARY_INTERVENTION_SCHEMA,
        "checkpoint": str(checkpoint),
        "formal_eval": str(formal_eval_path),
        "formal_passed_before_interventions": True,
        "text_variants": text_variants,
        "same_answer_different_trajectory": same_answer,
        "no_source": no_source,
        "independent_source_shuffle": independent_shuffle,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _evaluate_reasoner_variant(
    model: Any,
    records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    old_records: Sequence[dict[str, Any]] | None = None,
    disable_recurrence: bool = False,
) -> dict[str, Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[int(record["program_length"])].append(index)
    by_length: dict[str, Any] = {}
    for length, indexes in sorted(groups.items()):
        by_length[str(length)] = evaluate_a111_reasoner_items(
            model,
            [records[index] for index in indexes],
            device,
            recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
            batch_size=batch_size,
            disable_recurrence=disable_recurrence,
            old_records=[old_records[index] for index in indexes] if old_records is not None else None,
        )
    return {"aggregate": _weighted(list(by_length.values())), "by_length": by_length}


@torch.no_grad()
def run_a111_reasoner_interventions(
    checkpoint: Path,
    data_dir: Path,
    formal_eval_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 20261120,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.11 Reasoner interventions are forbidden before formal passes")
    model = load_a111_reasoner_checkpoint(checkpoint, device)
    causal_records = load_a18_records(data_dir, "causal_core")
    variants = build_a110_counterfactual_records(causal_records, seed=seed)
    results: dict[str, Any] = {}
    for name, pairs in variants.items():
        records = [pair["record"] for pair in pairs]
        old_records = [pair["old_record"] for pair in pairs]
        results[name] = _evaluate_reasoner_variant(
            model,
            records,
            device,
            batch_size,
            old_records=old_records if name in {"replacement", "deletion", "operation_shuffle"} else None,
        )
    same_records, same_old = _same_answer_record_pairs(causal_records, seed)
    results["same_answer_different_trajectory"] = _evaluate_reasoner_variant(
        model, same_records, device, batch_size, old_records=same_old
    )
    disable_recurrence = _evaluate_reasoner_variant(
        model, causal_records, device, batch_size, disable_recurrence=True
    )
    same_length = min(int(record["program_length"]) for record in causal_records)
    slot_records = [
        record for record in causal_records if int(record["program_length"]) == same_length
    ][:batch_size]
    permutation = torch.arange(model.config.workspace_slots - 1, -1, -1, device=device)
    maximum_delta = 0.0
    from .a1_11_train import _symbolic_reasoner_inputs

    for start in range(0, len(slot_records), batch_size):
        inputs = _symbolic_reasoner_inputs(slot_records[start : start + batch_size], device)
        standard = model(**inputs, recurrent_steps=TRAIN_RECURRENT_STEPS)
        permuted = model(
            **inputs, recurrent_steps=TRAIN_RECURRENT_STEPS, slot_permutation=permutation
        )
        maximum_delta = max(
            maximum_delta,
            float((standard["state_logits"] - permuted["state_logits"]).abs().max()),
            float((standard["answer_logits"] - permuted["answer_logits"]).abs().max()),
        )
    gates = {
        "prefix_trajectory_at_least_0_95": results["prefix"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "replacement_trajectory_at_least_0_95": results["replacement"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "deletion_trajectory_at_least_0_95": results["deletion"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "operation_shuffle_trajectory_at_least_0_95": results["operation_shuffle"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "query_swap_answer_at_least_0_95": results["query_swap"]["aggregate"]["final_answer_accuracy"] >= 0.95,
        "same_answer_new_trajectory_at_least_0_95": results["same_answer_different_trajectory"]["aggregate"]["trajectory_full_exact"] >= 0.95,
        "changed_old_trajectory_at_most_0_05": all(
            results[name]["aggregate"]["old_oracle_trajectory_full_exact"] <= 0.05
            for name in ("replacement", "operation_shuffle", "same_answer_different_trajectory")
        ),
        "disable_recurrence_at_most_0_20": disable_recurrence["aggregate"]["trajectory_full_exact"] <= 0.20,
        "slot_permutation_delta_below_1e_5": maximum_delta < 1e-5,
    }
    return {
        "schema_version": REASONER_INTERVENTION_SCHEMA,
        "checkpoint": str(checkpoint),
        "formal_eval": str(formal_eval_path),
        "formal_passed_before_interventions": True,
        "source_contract": "exact symbolic A1.8 records; no Qwen cache or boundary adapter",
        "results": results,
        "disable_recurrence": disable_recurrence,
        "slot_permutation_max_abs_logit_delta": maximum_delta,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _same_answer_record_pairs(
    records: Sequence[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wrapped = [{"record": record} for record in records]
    new_items, old_records = _same_answer_pairs(wrapped, seed)
    return [item["record"] for item in new_items], old_records
