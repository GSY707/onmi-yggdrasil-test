from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.12.reasoner-root-cause-assessment.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(run_dirs: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        formal_dir = run_dir / "formal"
        formal_path = formal_dir / "formal-eval.json"
        intervention_path = run_dir / "interventions" / "results.json"
        training = _read(formal_dir / "results.json")
        formal = _read(formal_path)
        intervention = _read(intervention_path) if intervention_path.exists() else None
        rows.append(
            {
                "run_dir": str(run_dir),
                "formal_passed": bool(formal["passed"]),
                "interventions_executed": intervention is not None,
                "interventions_passed": bool(intervention and intervention["passed"]),
                "formal_before_interventions": bool(
                    intervention is None
                    or formal_path.stat().st_mtime <= intervention_path.stat().st_mtime
                ),
                "training": training["training"],
                "fit_diagnostic": training["fit_diagnostic"],
                "formal": formal,
            }
        )
    return {
        "runs": rows,
        "formal_passes": sum(row["formal_passed"] for row in rows),
        "intervention_passes": sum(row["interventions_passed"] for row in rows),
        "formal_before_interventions": all(row["formal_before_interventions"] for row in rows),
    }


def classify_root_cause(bind: int, cursor: int, both: int) -> dict[str, Any]:
    if any(value not in {0, 3} for value in (bind, cursor, both)):
        return {
            "code": "seed_unstable_inconclusive",
            "localized": False,
            "judgment": "at least one arm is seed-unstable; no unique root-cause claim is justified",
        }
    if bind == 3 and cursor == 0:
        return {
            "code": "entity_addressability_primary",
            "localized": True,
            "judgment": "entity-addressable state is independently sufficient while cursor alignment is not",
        }
    if bind == 0 and cursor == 3:
        return {
            "code": "operation_step_alignment_primary",
            "localized": True,
            "judgment": "aligned operation visibility is independently sufficient while entity addressability is not",
        }
    if bind == 3 and cursor == 3:
        return {
            "code": "two_independent_sufficient_constraints",
            "localized": True,
            "judgment": "binding and cursor are alternative sufficient constraints; no unique single root cause exists",
        }
    if bind == 0 and cursor == 0 and both == 3:
        return {
            "code": "binding_cursor_joint_requirement",
            "localized": True,
            "judgment": "neither constraint is sufficient alone; their interaction is required",
        }
    if bind == 0 and cursor == 0 and both == 0:
        return {
            "code": "binding_and_cursor_insufficient",
            "localized": False,
            "judgment": "binding and cursor do not repair the reasoner; generic transition, closure, or objective remains",
        }
    return {
        "code": "combined_arm_interaction_anomaly",
        "localized": False,
        "judgment": "a sufficient single-factor arm is contradicted by the combined arm; inspect optimization interaction",
    }


def assess_a112(
    bind_dirs: Sequence[Path],
    cursor_dirs: Sequence[Path],
    both_dirs: Sequence[Path],
    bind_overfit: Path,
    cursor_overfit: Path,
    both_overfit: Path,
    a111_assessment: Path,
    output_path: Path,
) -> dict[str, Any]:
    arms = {
        "BIND": _arm(bind_dirs),
        "CURSOR": _arm(cursor_dirs),
        "BOTH": _arm(both_dirs),
    }
    overfit = {
        "BIND": bool(_read(bind_overfit)["overfit_passed"]),
        "CURSOR": bool(_read(cursor_overfit)["overfit_passed"]),
        "BOTH": bool(_read(both_overfit)["overfit_passed"]),
    }
    a111 = _read(a111_assessment)
    reference_passes = int(a111["reasoner_arm"]["formal_passes"])
    classification = classify_root_cause(
        arms["BIND"]["formal_passes"],
        arms["CURSOR"]["formal_passes"],
        arms["BOTH"]["formal_passes"],
    )
    passing_interventions_ok = all(
        arm["formal_passes"] == 0 or arm["intervention_passes"] == arm["formal_passes"]
        for arm in arms.values()
    )
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.12 binding x cursor root-cause localization",
        "reference": {
            "anonymous_full_program_arm": "A1.11-Reasoner",
            "formal_passes": reference_passes,
            "verified_failed_0_of_3": reference_passes == 0,
        },
        "overfit32": overfit,
        "arms": arms,
        "matrix": {
            "anonymous__full_program": f"A1.11 passed {reference_passes}/3",
            "entity__full_program": f"A1.12-BIND passed {arms['BIND']['formal_passes']}/3",
            "anonymous__cursor": f"A1.12-CURSOR passed {arms['CURSOR']['formal_passes']}/3",
            "entity__cursor": f"A1.12-BOTH passed {arms['BOTH']['formal_passes']}/3",
        },
        "classification": classification,
        "gates": {
            "all_overfit32_passed": all(overfit.values()),
            "three_runs_per_arm": all(len(arm["runs"]) == 3 for arm in arms.values()),
            "formal_before_interventions": all(
                arm["formal_before_interventions"] for arm in arms.values()
            ),
            "reference_failure_verified": reference_passes == 0,
            "passing_arm_interventions_passed": passing_interventions_ok,
            "root_cause_localized": bool(classification["localized"]),
        },
        "experiment_completed": True,
        "root_cause_localized": bool(classification["localized"] and passing_interventions_ok),
        "complete_v2_a_validated": False,
        "evidence_boundaries": {
            "exact_symbolic_input_only": True,
            "entity_contract_bundles_initialization_and_state_readout": True,
            "cursor_contract_changes_operation_visibility_schedule": True,
            "generic_transition_unchanged": True,
            "qwen_and_boundary_not_tested": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
