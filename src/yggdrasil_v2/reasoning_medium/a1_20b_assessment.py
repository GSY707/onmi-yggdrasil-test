from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.20b.boundary.assessment.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assess_a120b_failure(
    overfit_path: Path,
    cache_audit_path: Path,
    training_path: Path,
    diagnostic_path: Path,
    gradient_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    overfit = _read(overfit_path)
    cache_audit = _read(cache_audit_path)
    training = _read(training_path)
    diagnostic = _read(diagnostic_path)
    gradient_audit = _read(gradient_audit_path)
    validation = training["validation"]
    cells = list(validation["by_cell"].values())
    eligibility = {
        "fixed_budget_completed": training["training"]["steps_completed"]
        == training["training"]["steps_requested"],
        "cache_audit_passed": bool(cache_audit["passed"]),
        "core_hash_unchanged": training["core_state_sha256_before"]
        == training["core_state_sha256_after"],
        "architecture_integrity": bool(training["integrity"]["passed"]),
        "minimum_cell_mapping_at_least_0_995": min(
            cell["mapping_accuracy"]["minimum"] for cell in cells
        )
        >= 0.995,
        "minimum_cell_trajectory_at_least_0_995": min(
            cell["trajectory_full_exact"] for cell in cells
        )
        >= 0.995,
        "minimum_cell_answer_at_least_0_995": min(
            cell["final_answer_accuracy"] for cell in cells
        )
        >= 0.995,
        "aggregate_state_token_at_least_0_995": validation["aggregate"][
            "state_token_accuracy"
        ]
        >= 0.995,
    }
    joint = diagnostic["joint_mapping_exactness"]["aggregate"]
    variants = diagnostic["variants"]
    root_cause = {
        "code": "full_text_entity_binding_and_program_extraction_failure",
        "frozen_core_exonerated_by_full_oracle": variants["oracle_all"][
            "trajectory_full_exact"
        ]
        == 1.0,
        "entity_presence_not_limiting": variants[
            "oracle_all_except_entity_presence"
        ]["trajectory_full_exact"]
        == 1.0,
        "operation_presence_is_secondary": variants[
            "oracle_all_except_operation_presence"
        ]["trajectory_full_exact"]
        >= 0.95,
        "initial_value_binding_is_primary": variants["oracle_all_except_values"][
            "trajectory_full_exact"
        ]
        < 0.5,
        "source_binding_is_primary": variants["oracle_all_except_source"][
            "trajectory_full_exact"
        ]
        < 0.5,
        "target_binding_is_primary": variants["oracle_all_except_target"][
            "trajectory_full_exact"
        ]
        < 0.5,
        "family_sequence_is_secondary_but_material": variants[
            "oracle_all_except_family"
        ]["trajectory_full_exact"]
        < 0.9,
        "query_is_not_the_trajectory_bottleneck": variants[
            "oracle_all_except_query"
        ]["trajectory_full_exact"]
        == 1.0,
        "joint_boundary_exactness": joint["boundary_all_exact"],
        "execution_state_credit_reaches_no_discrete_control_logits": gradient_audit[
            "gates"
        ]["state_ce_reaches_no_discrete_control_logits"],
        "judgment": "the shallow full-text Boundary learned local labels but did not compose stable anonymous entity/value/address bindings across N and T",
    }
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.20B learned full-text boundary",
        "overfit32": {
            "path": str(overfit_path),
            "passed": bool(overfit["overfit_passed"]),
        },
        "run1_cache_audit": {
            "path": str(cache_audit_path),
            "passed": bool(cache_audit["passed"]),
        },
        "run1_training": {
            "path": str(training_path),
            "reader_seed": training["train_spec"]["reader_seed"],
            "steps_completed": training["training"]["steps_completed"],
            "training_seconds": training["training"]["seconds"],
            "best_checkpoint_step": training["best_checkpoint_step"],
            "validation_aggregate": validation["aggregate"],
            "minimum_cell": {
                "mapping": min(
                    cell["mapping_accuracy"]["minimum"] for cell in cells
                ),
                "trajectory": min(cell["trajectory_full_exact"] for cell in cells),
                "answer": min(cell["final_answer_accuracy"] for cell in cells),
            },
        },
        "failure_diagnostic": {
            "path": str(diagnostic_path),
            "joint_mapping_exactness": diagnostic["joint_mapping_exactness"],
            "variants": variants,
        },
        "gradient_credit_audit": {
            "path": str(gradient_audit_path),
            "gates": gradient_audit["gates"],
            "output_gradient_norms": gradient_audit["output_gradient_norms"],
            "interpretation": gradient_audit["interpretation"],
        },
        "formal_eligibility": eligibility,
        "formal_eligibility_passed": all(eligibility.values()),
        "formal_runs_completed": 0,
        "causal_suites_completed": 0,
        "classification": root_cause,
        "gates": {
            "overfit32_passed": bool(overfit["overfit_passed"]),
            "run1_formal_eligibility_passed": all(eligibility.values()),
            "formal_3_of_3": False,
            "causal_3_of_3": False,
            "a120b_passed": False,
        },
        "a120b_complete": True,
        "a120b_passed": False,
        "next_stage_allowed": None,
        "stop_reason": "run-1 failed the heldout validation eligibility gate before formal cache/evaluation",
        "not_run_by_contract": [
            "run-1 formal split cache and formal evaluation",
            "run-1 hidden/text causal interventions",
            "run-2",
            "run-3",
            "A1.21P",
            "A1.22A",
        ],
        "evidence_boundaries": {
            "oracle_diagnostics_are_not_formal_results": True,
            "a119h_hybrid_core_evidence_remains_valid": True,
            "learned_full_text_boundary_not_validated": True,
            "matched_text_cot_pareto_not_tested": True,
            "natural_language_audit_not_tested": True,
            "v2_a_not_passed": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
