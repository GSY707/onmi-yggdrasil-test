from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .a1_8_data import load_a18_records
from .a1_10_interventions import build_a110_counterfactual_records
from .a1_10_train import TRAIN_RECURRENT_STEPS
from .a1_12_train import evaluate_a112_items, load_a112_checkpoint


INTERVENTION_SCHEMA = "yggdrasil.v2-a1.12.symbolic-causal-interventions.v1"


def _weighted(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["examples"] for row in rows)
    result: dict[str, Any] = {"examples": total}
    for metric in (
        "trajectory_full_exact",
        "final_state_full_exact",
        "state_token_accuracy",
        "final_answer_accuracy",
        "answer_state_prediction_consistency",
        "old_oracle_trajectory_full_exact",
    ):
        available = [row for row in rows if metric in row]
        if available:
            denominator = sum(row["examples"] for row in available)
            result[metric] = sum(
                row[metric] * row["examples"] for row in available
            ) / max(1, denominator)
    return result


def _evaluate_variant(
    model: Any,
    records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    old_records: Sequence[dict[str, Any]] | None = None,
    disable_recurrence: bool = False,
    start_value_permutation: torch.Tensor | None = None,
) -> dict[str, Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[int(record["program_length"])].append(index)
    by_length: dict[str, Any] = {}
    for length, indices in sorted(groups.items()):
        by_length[str(length)] = evaluate_a112_items(
            model,
            [records[index] for index in indices],
            device,
            recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
            batch_size=batch_size,
            disable_recurrence=disable_recurrence,
            start_value_permutation=start_value_permutation,
            old_records=(
                [old_records[index] for index in indices]
                if old_records is not None
                else None
            ),
        )
    return {"aggregate": _weighted(list(by_length.values())), "by_length": by_length}


def _trajectory_key(record: dict[str, Any]) -> str:
    return json.dumps(record["state_trajectory"], sort_keys=True)


def _same_answer_pairs(
    records: Sequence[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(int(record["answer_index"]), int(record["program_length"]))].append(index)
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
                    if _trajectory_key(records[candidate]) != _trajectory_key(records[target])
                ),
                None,
            )
            if source is not None:
                new_records.append(records[source])
                old_records.append(records[target])
    if not new_records:
        raise RuntimeError("A1.12 same-answer intervention found no eligible pair")
    return new_records, old_records


@torch.no_grad()
def run_a112_interventions(
    checkpoint: Path,
    data_dir: Path,
    formal_eval_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 20261230,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.12 interventions are forbidden before formal passes")
    model = load_a112_checkpoint(checkpoint, device)
    causal_records = load_a18_records(data_dir, "causal_core")
    pairs_by_variant = build_a110_counterfactual_records(causal_records, seed=seed)
    results: dict[str, Any] = {}
    for name, pairs in pairs_by_variant.items():
        records = [pair["record"] for pair in pairs]
        old = [pair["old_record"] for pair in pairs]
        results[name] = _evaluate_variant(
            model,
            records,
            device,
            batch_size,
            old_records=old if name in {"replacement", "deletion", "operation_shuffle"} else None,
        )
    same_records, same_old = _same_answer_pairs(causal_records, seed)
    results["same_answer_different_trajectory"] = _evaluate_variant(
        model, same_records, device, batch_size, old_records=same_old
    )
    disable_recurrence = _evaluate_variant(
        model, causal_records, device, batch_size, disable_recurrence=True
    )
    start_permutation = _evaluate_variant(
        model,
        causal_records,
        device,
        batch_size,
        start_value_permutation=torch.tensor([2, 0, 1], device=device),
    )
    changed = ("replacement", "deletion", "operation_shuffle", "same_answer_different_trajectory")
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
        "disable_recurrence_at_most_0_20": disable_recurrence["aggregate"]["trajectory_full_exact"] <= 0.20,
        "wrong_start_state_at_most_0_20": start_permutation["aggregate"]["trajectory_full_exact"] <= 0.20,
    }
    return {
        "schema_version": INTERVENTION_SCHEMA,
        "checkpoint": str(checkpoint),
        "formal_eval": str(formal_eval_path),
        "formal_passed_before_interventions": True,
        "arm": model.arm,
        "results": results,
        "disable_recurrence": disable_recurrence,
        "start_state_permutation": start_permutation,
        "gates": gates,
        "passed": all(gates.values()),
    }
