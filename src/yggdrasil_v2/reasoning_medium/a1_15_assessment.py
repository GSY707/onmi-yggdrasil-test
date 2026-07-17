from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.15.closed-coupled-core-assessment.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_passed(formal: dict[str, Any]) -> bool:
    return all(
        value for name, value in formal["gates"].items() if "answer" not in name
    )


def _arm(run_dirs: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        formal_dir = run_dir / "formal"
        formal = _read(formal_dir / "formal-eval.json")
        intervention_path = run_dir / "interventions" / "results.json"
        intervention = _read(intervention_path) if intervention_path.exists() else None
        rows.append(
            {
                "run_dir": str(run_dir),
                "formal_passed": bool(formal["passed"]),
                "state_passed": _state_passed(formal),
                "interventions_executed": intervention is not None,
                "interventions_passed": bool(intervention and intervention["passed"]),
                "training": _read(formal_dir / "results.json")["training"],
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


def classify_a115(
    *, trigger_valid: bool, formal_passes: int, intervention_passes: int
) -> dict[str, Any]:
    if not trigger_valid:
        return {
            "code": "trigger_invalid",
            "localized": False,
            "judgment": "A1.13 did not isolate an answer-only failure after stable state closure",
        }
    if formal_passes not in {0, 3}:
        return {
            "code": "seed_unstable_inconclusive",
            "localized": False,
            "judgment": "query-coupled readout is not stable across the three fresh seeds",
        }
    if formal_passes == 0:
        return {
            "code": "closed_coupled_core_insufficient",
            "localized": False,
            "judgment": "query coupling did not close the remaining formal failure",
        }
    if intervention_passes != formal_passes:
        return {
            "code": "ordinary_pass_causal_failure",
            "localized": False,
            "judgment": "ordinary formal passed but causal interventions did not pass for every seed",
        }
    return {
        "code": "stable_closed_coupled_core_confirmed",
        "localized": True,
        "judgment": (
            "relation-addressed transition, latent closure, and state-coupled answer "
            "form a seed-stable sufficient core under the exact-symbolic contract"
        ),
    }


def assess_a115(
    a113_assessment_path: Path,
    run_dirs: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    a113 = _read(a113_assessment_path)
    joint = a113["arms"]["STRUCTURED-CLOSURE"]
    trigger_valid = (
        int(joint["state_passes"]) == 3
        and int(joint["formal_passes"]) < 3
        and int(joint["answer_only_failures"])
        == 3 - int(joint["formal_passes"])
    )
    arm = _arm(run_dirs)
    classification = classify_a115(
        trigger_valid=trigger_valid,
        formal_passes=int(arm["formal_passes"]),
        intervention_passes=int(arm["intervention_passes"]),
    )
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.15 closed query-coupled core",
        "trigger": {
            "a1_13_assessment": str(a113_assessment_path),
            "structured_closure_state_passes": int(joint["state_passes"]),
            "structured_closure_formal_passes": int(joint["formal_passes"]),
            "structured_closure_answer_only_failures": int(
                joint["answer_only_failures"]
            ),
            "valid": trigger_valid,
        },
        "arm": arm,
        "classification": classification,
        "gates": {
            "trigger_valid": trigger_valid,
            "three_fresh_runs": len(run_dirs) == 3,
            "formal_passed_3_of_3": int(arm["formal_passes"]) == 3,
            "interventions_passed_3_of_3": int(arm["intervention_passes"]) == 3,
            "root_cause_localized": bool(classification["localized"]),
        },
        "root_cause_localized": bool(classification["localized"]),
        "complete_v2_a_validated": False,
        "evidence_boundaries": {
            "exact_symbolic_input_only": True,
            "three_register_core_only": True,
            "learned_boundary_not_tested": True,
            "anonymous_open_world_workspace_not_tested": True,
            "qwen_and_text_cot_not_tested": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
