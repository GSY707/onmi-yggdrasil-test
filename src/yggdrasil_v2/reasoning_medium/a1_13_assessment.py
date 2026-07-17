from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.13.transition-closure-assessment.v1"


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
        state_gates = {
            name: value
            for name, value in formal["gates"].items()
            if "answer" not in name
        }
        state_passed = all(state_gates.values())
        rows.append(
            {
                "run_dir": str(run_dir),
                "formal_passed": bool(formal["passed"]),
                "state_passed": state_passed,
                "answer_only_failure": state_passed and not bool(formal["passed"]),
                "interventions_executed": intervention is not None,
                "interventions_passed": bool(intervention and intervention["passed"]),
                "training": training["training"],
                "fit_diagnostic": training["fit_diagnostic"],
                "formal": formal,
            }
        )
    return {
        "runs": rows,
        "formal_passes": sum(row["formal_passed"] for row in rows),
        "state_passes": sum(row["state_passed"] for row in rows),
        "answer_only_failures": sum(row["answer_only_failure"] for row in rows),
        "intervention_passes": sum(row["interventions_passed"] for row in rows),
    }


def _state_passes_from_assessed_arm(arm: dict[str, Any]) -> int:
    if "state_passes" in arm:
        return int(arm["state_passes"])
    return sum(
        all(
            value
            for name, value in row["formal"]["gates"].items()
            if "answer" not in name
        )
        for row in arm["runs"]
    )


def classify_a113(
    generic_ce: int,
    generic_closure: int,
    structured_ce: int,
    structured_closure: int | None,
) -> dict[str, Any]:
    observed = [generic_ce, generic_closure, structured_ce]
    if structured_closure is not None:
        observed.append(structured_closure)
    if any(value not in {0, 3} for value in observed):
        return {
            "code": "seed_unstable_inconclusive",
            "localized": False,
            "judgment": "at least one required arm is seed-unstable",
        }
    if generic_ce != 0:
        return {
            "code": "reference_contradiction",
            "localized": False,
            "judgment": "A1.12-BOTH reference did not stably fail; conditional A1.13 is invalid",
        }
    if generic_closure == 3 and structured_ce == 0:
        return {
            "code": "continuous_state_closure_primary",
            "localized": True,
            "judgment": "soft prototype closure is independently sufficient; structured transition is not",
        }
    if generic_closure == 0 and structured_ce == 3:
        return {
            "code": "generic_transition_identity_primary",
            "localized": True,
            "judgment": "relation-addressed transition identity is independently sufficient; closure is not",
        }
    if generic_closure == 3 and structured_ce == 3:
        return {
            "code": "alternative_sufficient_constraints",
            "localized": True,
            "judgment": "closure and relation-addressed transition are alternative sufficient repairs",
        }
    if structured_closure is None:
        return {
            "code": "joint_arm_required",
            "localized": False,
            "judgment": "both single-factor arms failed; STRUCTURED-CLOSURE must run",
        }
    if structured_closure == 3:
        return {
            "code": "transition_closure_joint_requirement",
            "localized": True,
            "judgment": "relation-addressed transition and closure objective are jointly required",
        }
    return {
        "code": "transition_closure_insufficient",
        "localized": False,
        "judgment": "transition identity plus closure still fail; pointer grounding or answer coupling remains",
    }


def assess_a113(
    a112_assessment: Path,
    generic_closure_dirs: Sequence[Path],
    structured_ce_dirs: Sequence[Path],
    output_path: Path,
    structured_closure_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    a112 = _read(a112_assessment)
    generic_ce = int(a112["arms"]["BOTH"]["formal_passes"])
    arms = {
        "GENERIC-CE": a112["arms"]["BOTH"],
        "GENERIC-CLOSURE": _arm(generic_closure_dirs),
        "STRUCTURED-CE": _arm(structured_ce_dirs),
    }
    structured_closure: int | None = None
    if structured_closure_dirs:
        arms["STRUCTURED-CLOSURE"] = _arm(structured_closure_dirs)
        structured_closure = int(arms["STRUCTURED-CLOSURE"]["formal_passes"])
    classification = classify_a113(
        generic_ce,
        int(arms["GENERIC-CLOSURE"]["formal_passes"]),
        int(arms["STRUCTURED-CE"]["formal_passes"]),
        structured_closure,
    )
    structured_closure_state: int | None = None
    if structured_closure_dirs:
        structured_closure_state = int(arms["STRUCTURED-CLOSURE"]["state_passes"])
    state_classification = classify_a113(
        _state_passes_from_assessed_arm(a112["arms"]["BOTH"]),
        int(arms["GENERIC-CLOSURE"]["state_passes"]),
        int(arms["STRUCTURED-CE"]["state_passes"]),
        structured_closure_state,
    )
    passing_interventions_ok = all(
        int(arm["formal_passes"]) == 0
        or int(arm.get("intervention_passes", 0)) == int(arm["formal_passes"])
        for name, arm in arms.items()
        if name != "GENERIC-CE"
    )
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.13 transition x closure root-cause localization",
        "conditional_reference_valid": generic_ce == 0,
        "arms": arms,
        "classification": classification,
        "state_classification": state_classification,
        "gates": {
            "a1_12_both_failed_0_of_3": generic_ce == 0,
            "three_runs_per_required_new_arm": len(generic_closure_dirs) == 3 and len(structured_ce_dirs) == 3,
            "joint_arm_complete_if_required": classification["code"] != "joint_arm_required",
            "passing_arm_interventions_passed": passing_interventions_ok,
            "root_cause_localized": bool(classification["localized"]),
            "state_root_cause_localized": bool(state_classification["localized"]),
        },
        "root_cause_localized": bool(classification["localized"] and passing_interventions_ok),
        "complete_v2_a_validated": False,
        "evidence_boundaries": {
            "exact_symbolic_input_only": True,
            "a1_12_entity_and_cursor_frozen": True,
            "hard_reembedding_absent": True,
            "structured_transition_is_relation_addressed_not_discrete_executor": True,
            "qwen_and_boundary_not_tested": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
