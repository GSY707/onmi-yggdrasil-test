from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.18b.trajectory-state-scaffold-assessment.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_passed(formal: dict[str, Any]) -> bool:
    return all(
        value
        for name, value in formal["gates"].items()
        if "final_answer_accuracy" not in name
        and name != "causal_answer_at_least_0_95"
    )


def _arm(run_dirs: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        formal_dir = run_dir / "formal"
        training = _read(formal_dir / "results.json")
        formal = _read(formal_dir / "formal-eval.json")
        intervention_path = run_dir / "interventions" / "results.json"
        intervention = _read(intervention_path) if intervention_path.exists() else None
        rows.append(
            {
                "run_dir": str(run_dir),
                "model_seed": int(formal["model_seed"]),
                "data_seed": int(formal["data_seed"]),
                "trained_auxiliary": formal["trained_auxiliary"],
                "fixed_budget_completed": bool(
                    training["training"]["fixed_budget_completed"]
                ),
                "deployment_auxiliary_stripped": bool(
                    formal["gates"]["deployment_auxiliary_absent"]
                ),
                "formal_passed": bool(formal["passed"]),
                "state_passed": _state_passed(formal),
                "interventions_executed": intervention is not None,
                "interventions_passed": bool(intervention and intervention["passed"]),
                "training": training["training"],
                "formal": formal,
                "interventions": intervention,
            }
        )
    return {
        "runs": rows,
        "formal_passes": sum(row["formal_passed"] for row in rows),
        "state_passes": sum(row["state_passed"] for row in rows),
        "intervention_passes": sum(row["interventions_passed"] for row in rows),
    }


def classify_a118b(
    *,
    final_saux_state: int,
    tsaux_paired_full: int,
    tsaux_paired_causal: int,
    tsaux_fresh_full: int,
    tsaux_fresh_causal: int,
    integrity_passed: bool,
) -> dict[str, Any]:
    if not integrity_passed:
        return {
            "code": "training_scaffold_integrity_failure",
            "mechanism_solved": False,
            "judgment": "seed, budget, target, or deployment-strip integrity failed",
        }
    if final_saux_state == 3:
        return {
            "code": "final_state_auxiliary_already_sufficient",
            "mechanism_solved": False,
            "judgment": "the trajectory-state rescue was not triggered by a final-SAUX instability",
        }
    counts = (
        tsaux_paired_full,
        tsaux_paired_causal,
        tsaux_fresh_full,
        tsaux_fresh_causal,
    )
    if any(value not in {0, 3} for value in counts):
        return {
            "code": "trajectory_state_auxiliary_seed_unstable",
            "mechanism_solved": False,
            "judgment": "TSAUX remains unstable in the paired or fresh three-seed matrix",
        }
    if counts != (3, 3, 3, 3):
        return {
            "code": "trajectory_state_auxiliary_insufficient",
            "mechanism_solved": False,
            "judgment": "TSAUX does not close both formal and causal gates",
        }
    return {
        "code": "per_step_global_state_credit_assignment_confirmed",
        "mechanism_solved": True,
        "judgment": (
            "a shared training-only head that predicts the complete state from the global "
            "workspace at every recurrent step yields six-seed causal closure after the "
            "head is physically stripped"
        ),
    }


def assess_a118b(
    qaux_overfit_path: Path,
    final_saux_overfit_path: Path,
    tsaux_overfit_path: Path,
    final_saux_dirs: Sequence[Path],
    tsaux_paired_dirs: Sequence[Path],
    tsaux_fresh_dirs: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    qaux_overfit = _read(qaux_overfit_path)
    final_saux_overfit = _read(final_saux_overfit_path)
    tsaux_overfit = _read(tsaux_overfit_path)
    final_saux = _arm(final_saux_dirs)
    tsaux_paired = _arm(tsaux_paired_dirs)
    tsaux_fresh = _arm(tsaux_fresh_dirs)

    expected_paired_model = [20261321, 20261322, 20261323]
    expected_fresh_model = [20261821, 20261822, 20261823]
    expected_data = [20260721, 20260722, 20260723]
    formal_rows = (
        final_saux["runs"] + tsaux_paired["runs"] + tsaux_fresh["runs"]
    )
    integrity = {
        "all_overfit_controls_passed": all(
            bool(row.get("overfit_passed"))
            for row in (qaux_overfit, final_saux_overfit, tsaux_overfit)
        ),
        "three_runs_per_formal_arm": all(
            len(rows) == 3
            for rows in (final_saux_dirs, tsaux_paired_dirs, tsaux_fresh_dirs)
        ),
        "paired_model_seeds_exact": (
            [row["model_seed"] for row in final_saux["runs"]]
            == expected_paired_model
            and [row["model_seed"] for row in tsaux_paired["runs"]]
            == expected_paired_model
        ),
        "fresh_model_seeds_exact": (
            [row["model_seed"] for row in tsaux_fresh["runs"]]
            == expected_fresh_model
        ),
        "data_seeds_exact": all(
            [row["data_seed"] for row in arm["runs"]] == expected_data
            for arm in (final_saux, tsaux_paired, tsaux_fresh)
        ),
        "all_fixed_budget_completed": all(
            row["fixed_budget_completed"] for row in formal_rows
        ),
        "all_deployments_auxiliary_stripped": all(
            row["deployment_auxiliary_stripped"] for row in formal_rows
        ),
        "final_saux_targets_exact": all(
            row["trained_auxiliary"] == "full_state"
            for row in final_saux["runs"]
        ),
        "tsaux_targets_exact": all(
            row["trained_auxiliary"] == "full_trajectory_state"
            for row in tsaux_paired["runs"] + tsaux_fresh["runs"]
        ),
    }
    classification = classify_a118b(
        final_saux_state=int(final_saux["state_passes"]),
        tsaux_paired_full=int(tsaux_paired["formal_passes"]),
        tsaux_paired_causal=int(tsaux_paired["intervention_passes"]),
        tsaux_fresh_full=int(tsaux_fresh["formal_passes"]),
        tsaux_fresh_causal=int(tsaux_fresh["intervention_passes"]),
        integrity_passed=all(integrity.values()),
    )
    training_seconds = sum(float(row["training"]["seconds"]) for row in formal_rows)
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.18/A1.18B training-objective mechanism solution",
        "overfit_controls": {
            "QAUX": {
                "path": str(qaux_overfit_path),
                "passed": bool(qaux_overfit.get("overfit_passed")),
            },
            "FINAL_SAUX": {
                "path": str(final_saux_overfit_path),
                "passed": bool(final_saux_overfit.get("overfit_passed")),
            },
            "TSAUX_FAILED_SEED": {
                "path": str(tsaux_overfit_path),
                "passed": bool(tsaux_overfit.get("overfit_passed")),
            },
        },
        "arms": {
            "FINAL_SAUX_PAIRED": final_saux,
            "TSAUX_PAIRED": tsaux_paired,
            "TSAUX_FRESH": tsaux_fresh,
        },
        "integrity": integrity,
        "classification": classification,
        "gates": {
            "final_saux_exposed_seed_instability": int(final_saux["state_passes"]) < 3,
            "tsaux_paired_formal_3_of_3": int(tsaux_paired["formal_passes"]) == 3,
            "tsaux_paired_causal_3_of_3": int(tsaux_paired["intervention_passes"]) == 3,
            "tsaux_fresh_formal_3_of_3": int(tsaux_fresh["formal_passes"]) == 3,
            "tsaux_fresh_causal_3_of_3": int(tsaux_fresh["intervention_passes"]) == 3,
            "integrity_passed": all(integrity.values()),
            "mechanism_solved": bool(classification["mechanism_solved"]),
        },
        "mechanism_solved": bool(classification["mechanism_solved"]),
        "answer_specific_auxiliary_required": False,
        "diagnostic_core_architecture_validated": bool(
            classification["mechanism_solved"]
        ),
        "complete_v2_a_validated": False,
        "training_cost": {
            "formal_run_count": len(formal_rows),
            "aggregate_training_seconds": training_seconds,
            "excludes_overfit_formal_eval_and_intervention_seconds": True,
        },
        "evidence_boundaries": {
            "exact_symbolic_input_only": True,
            "three_register_diagnostic_core_only": True,
            "oracle_state_trajectory_supervision_required_during_training": True,
            "learned_boundary_not_tested": True,
            "anonymous_workspace_not_tested": True,
            "scalable_state_target_source_not_solved": True,
            "matched_text_cot_pareto_not_tested": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
