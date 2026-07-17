from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.19h.full-hybrid-core-assessment.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_a119h_h2(
    *, formal_passes: int, causal_passes: int, integrity_passed: bool
) -> dict[str, Any]:
    if not integrity_passed:
        return {
            "code": "h2_integrity_failure",
            "a119h_complete": False,
            "judgment": "variable-cardinality data, seed, budget, or deployment integrity failed",
        }
    if formal_passes != 3:
        return {
            "code": "variable_cardinality_formal_failure",
            "a119h_complete": False,
            "judgment": "the hybrid core did not close three-seed relation, horizon, and heldout-N formal gates",
        }
    if causal_passes != 3:
        return {
            "code": "variable_cardinality_causal_failure",
            "a119h_complete": False,
            "judgment": "formal variable-cardinality behavior did not survive causal/address interventions",
        }
    return {
        "code": "generalized_hybrid_core_confirmed",
        "a119h_complete": True,
        "judgment": "opaque addressing and shared continuous transitions generalized across trained N=2,3,4 and heldout N=5 with three-seed causal closure",
    }


def assess_a119h_h2(
    h1_assessment_path: Path,
    overfit_path: Path,
    data_audit_paths: Sequence[Path],
    run_dirs: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    h1 = _read(h1_assessment_path)
    overfit = _read(overfit_path)
    audits = [_read(path) for path in data_audit_paths]
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        training = _read(run_dir / "formal" / "results.json")
        formal = _read(run_dir / "formal" / "formal-eval.json")
        intervention = _read(run_dir / "interventions" / "results.json")
        rows.append(
            {
                "run_dir": str(run_dir),
                "model_seed": int(formal["model_seed"]),
                "data_seed": int(formal["data_seed"]),
                "mapping_seed": int(formal["mapping_seed"]),
                "fixed_budget_completed": bool(training["training"]["fixed_budget_completed"]),
                "deployment_auxiliary_stripped": bool(formal["gates"]["deployment_auxiliary_absent"]),
                "formal_passed": bool(formal["passed"]),
                "interventions_passed": bool(intervention["passed"]),
                "training": training["training"],
                "formal": formal,
                "interventions": intervention,
            }
        )
    expected_model = [20261961, 20261962, 20261963]
    expected_data = [20261951, 20261952, 20261953]
    expected_mapping = [20261971, 20261972, 20261973]
    integrity = {
        "h1_passed": bool(h1.get("h1_passed")),
        "overfit_passed": bool(overfit.get("overfit_passed")),
        "three_data_audits_passed": len(audits) == 3 and all(audit["passed"] for audit in audits),
        "three_formal_runs": len(rows) == 3,
        "model_seeds_exact": [row["model_seed"] for row in rows] == expected_model,
        "data_seeds_exact": [row["data_seed"] for row in rows] == expected_data,
        "mapping_seeds_exact": [row["mapping_seed"] for row in rows] == expected_mapping,
        "all_fixed_budget_completed": all(row["fixed_budget_completed"] for row in rows),
        "all_deployments_auxiliary_stripped": all(row["deployment_auxiliary_stripped"] for row in rows),
        "all_heldout_n5_gates_present_and_passed": all(
            row["formal"]["gates"]["heldout_count_is_exactly_n5"]
            and row["formal"]["gates"]["entity_heldout_trajectory_full_exact_gate"]
            and row["formal"]["gates"]["entity_relation_heldout_trajectory_full_exact_gate"]
            for row in rows
        ),
    }
    formal_passes = sum(row["formal_passed"] for row in rows)
    causal_passes = sum(row["interventions_passed"] for row in rows)
    classification = classify_a119h_h2(
        formal_passes=formal_passes,
        causal_passes=causal_passes,
        integrity_passed=all(integrity.values()),
    )
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.19H generalized hybrid core",
        "h1_assessment": {"path": str(h1_assessment_path), "passed": bool(h1.get("h1_passed"))},
        "h2_overfit": {"path": str(overfit_path), "passed": bool(overfit.get("overfit_passed"))},
        "h2_data_audits": audits,
        "h2_runs": rows,
        "h2_formal_passes": formal_passes,
        "h2_intervention_passes": causal_passes,
        "integrity": integrity,
        "classification": classification,
        "gates": {
            "h1_passed": bool(h1.get("h1_passed")),
            "h2_overfit32_passed": bool(overfit.get("overfit_passed")),
            "h2_formal_3_of_3": formal_passes == 3,
            "h2_causal_3_of_3": causal_passes == 3,
            "integrity_passed": all(integrity.values()),
        },
        "a119h_complete": bool(classification["a119h_complete"]),
        "next_stage_allowed": "A1.20B" if classification["a119h_complete"] else None,
        "training_cost": {
            "h1_formal_seconds": float(h1["training_cost"]["aggregate_training_seconds"]),
            "h2_formal_seconds": sum(float(row["training"]["seconds"]) for row in rows),
            "excludes_overfit_evaluation_and_interventions": True,
        },
        "evidence_boundaries": {
            "exact_symbolic_input_only": True,
            "copy_swap_family_only": True,
            "learned_full_text_boundary_not_tested": True,
            "matched_text_cot_pareto_not_tested": True,
            "natural_language_audit_not_tested": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
