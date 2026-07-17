from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .a1_7_data import REGISTER_NAMES, VALUE_LABELS, _render_record
from .a1_8_data import load_a18_records
from .a1_10_interventions import build_a110_counterfactual_records
from .a1_10_train import TRAIN_RECURRENT_STEPS
from .a1_12_interventions import _same_answer_pairs
from .a1_19h_data import encode_a119h_records, permute_entity_axis
from .a1_19h_train import evaluate_a119h_items, load_a119h_deployment


INTERVENTION_SCHEMA = "yggdrasil.v2-a1.19h.h1-opaque-handle.interventions.v1"


def _weighted(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["examples"] for row in rows)
    aggregate: dict[str, Any] = {"examples": total}
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
            aggregate[metric] = sum(
                row[metric] * row["examples"] for row in available
            ) / max(1, denominator)
    return aggregate


def _evaluate_variant(
    model: Any,
    records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
    *,
    mapping_seed: int,
    old_records: Sequence[dict[str, Any]] | None = None,
    disable_recurrence: bool = False,
    start_value_permutation: torch.Tensor | None = None,
    handle_alias_seed: int | None = None,
) -> dict[str, Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[int(record["program_length"])].append(index)
    by_length: dict[str, Any] = {}
    for length, indices in sorted(groups.items()):
        by_length[str(length)] = evaluate_a119h_items(
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
            start_value_permutation=start_value_permutation,
            handle_alias_seed=handle_alias_seed,
        )
    return {"aggregate": _weighted(list(by_length.values())), "by_length": by_length}


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


def _duplicate_value_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    duplicated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        start = dict(record["start_state"])
        first, second = REGISTER_NAMES[:2]
        start[second] = start[first]
        rendered = _render_record(
            start,
            list(record["operations"]),
            record["query_register"],
            "a1_19h_same_value",
            index,
        )
        rendered["a119h_mapping_key"] = record["fingerprint"]
        duplicated.append(rendered)
    return duplicated


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
    slot_max_abs = alias_max_abs = answer_slot_max_abs = answer_alias_max_abs = 0.0
    permutation = torch.tensor([2, 0, 1], dtype=torch.long, device=device)
    for start in range(0, len(records), batch_size):
        rows = list(records[start : start + batch_size])
        recurrent_steps = max(
            TRAIN_RECURRENT_STEPS, max(int(row["program_length"]) for row in rows)
        )
        batch = encode_a119h_records(
            rows,
            recurrent_steps,
            device,
            mapping_seed=mapping_seed,
            maximum_entities=model.config.maximum_entities,
        )
        base = model(**batch.inputs, recurrent_steps=recurrent_steps)
        permuted_batch = permute_entity_axis(batch, permutation)
        permuted = model(**permuted_batch.inputs, recurrent_steps=recurrent_steps)
        slot_max_abs = max(
            slot_max_abs,
            float(
                (
                    permuted["state_logits"]
                    - base["state_logits"][:, :, permutation]
                )
                .abs()
                .max()
            ),
        )
        answer_slot_max_abs = max(
            answer_slot_max_abs,
            float((permuted["answer_logits"] - base["answer_logits"]).abs().max()),
        )
        aliased_batch = encode_a119h_records(
            rows,
            recurrent_steps,
            device,
            mapping_seed=mapping_seed,
            maximum_entities=model.config.maximum_entities,
            handle_alias_seed=alias_seed,
        )
        aliased = model(**aliased_batch.inputs, recurrent_steps=recurrent_steps)
        alias_max_abs = max(
            alias_max_abs,
            float((aliased["state_logits"] - base["state_logits"]).abs().max()),
        )
        answer_alias_max_abs = max(
            answer_alias_max_abs,
            float((aliased["answer_logits"] - base["answer_logits"]).abs().max()),
        )
    return {
        "examples": len(records),
        "slot_permutation_state_logit_max_abs": slot_max_abs,
        "slot_permutation_answer_logit_max_abs": answer_slot_max_abs,
        "handle_alias_state_logit_max_abs": alias_max_abs,
        "handle_alias_answer_logit_max_abs": answer_alias_max_abs,
    }


@torch.no_grad()
def run_a119h_h1_interventions(
    checkpoint: Path,
    data_dir: Path,
    formal_eval_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 20261940,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.19H-H1 interventions are forbidden before formal passes")
    model = load_a119h_deployment(checkpoint, device)
    mapping_seed = model.a119h_mapping_seed
    causal_records = load_a18_records(data_dir, "causal_core")
    pairs_by_variant = build_a110_counterfactual_records(causal_records, seed=seed)
    results: dict[str, Any] = {}
    for name, pairs in pairs_by_variant.items():
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
    same_records_raw, same_old_raw = _same_answer_pairs(causal_records, seed)
    same_pairs = [
        {"record": record, "old_record": old}
        for record, old in zip(same_records_raw, same_old_raw)
    ]
    same_records, same_old = _mapping_aligned_pairs(same_pairs)
    results["same_answer_different_trajectory"] = _evaluate_variant(
        model,
        same_records,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        old_records=same_old,
    )
    same_value = _evaluate_variant(
        model,
        _duplicate_value_records(causal_records),
        device,
        batch_size,
        mapping_seed=mapping_seed,
    )
    handle_alias = _evaluate_variant(
        model,
        causal_records,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        handle_alias_seed=seed + 1,
    )
    invariance = _address_invariance(
        model,
        causal_records,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        alias_seed=seed + 1,
    )
    disable_recurrence = _evaluate_variant(
        model,
        causal_records,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        disable_recurrence=True,
    )
    wrong_start = _evaluate_variant(
        model,
        causal_records,
        device,
        batch_size,
        mapping_seed=mapping_seed,
        start_value_permutation=torch.tensor([2, 0, 1], device=device),
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
        "same_value_different_entity_trajectory_at_least_0_95": same_value["aggregate"]["trajectory_full_exact"] >= 0.95,
        "same_value_different_entity_answer_at_least_0_95": same_value["aggregate"]["final_answer_accuracy"] >= 0.95,
        "handle_alias_trajectory_at_least_0_95": handle_alias["aggregate"]["trajectory_full_exact"] >= 0.95,
        "slot_permutation_equivariance_below_1e_6": max(
            invariance["slot_permutation_state_logit_max_abs"],
            invariance["slot_permutation_answer_logit_max_abs"],
        ) < 1e-6,
        "handle_alias_invariance_below_1e_6": max(
            invariance["handle_alias_state_logit_max_abs"],
            invariance["handle_alias_answer_logit_max_abs"],
        ) < 1e-6,
        "disable_recurrence_at_most_0_20": disable_recurrence["aggregate"]["trajectory_full_exact"] <= 0.20,
        "wrong_start_state_at_most_0_20": wrong_start["aggregate"]["trajectory_full_exact"] <= 0.20,
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
        "formal_passed_before_interventions": True,
        "results": results,
        "same_value_different_entity": same_value,
        "handle_alias": handle_alias,
        "address_invariance": invariance,
        "disable_recurrence": disable_recurrence,
        "wrong_start_state": wrong_start,
        "gates": gates,
        "passed": all(gates.values()),
    }
