from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .a1_8_data import load_a18_records
from .a1_10_interventions import build_a110_counterfactual_records
from .a1_12_interventions import _evaluate_variant, _same_answer_pairs
from .a1_13_train import load_a113_checkpoint


INTERVENTION_SCHEMA = "yggdrasil.v2-a1.13.symbolic-causal-interventions.v1"


@torch.no_grad()
def run_a113_interventions(
    checkpoint: Path,
    data_dir: Path,
    formal_eval_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 20261330,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.13 interventions are forbidden before formal passes")
    model = load_a113_checkpoint(checkpoint, device)
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

