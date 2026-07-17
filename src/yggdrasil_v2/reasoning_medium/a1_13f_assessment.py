from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.13f.fixed-budget-assessment.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(run_dirs: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        formal_dir = run_dir / "formal"
        training = _read(formal_dir / "results.json")
        formal = _read(formal_dir / "formal-eval.json")
        state_passed = all(
            value
            for name, value in formal["gates"].items()
            if "answer" not in name
        )
        rows.append(
            {
                "run_dir": str(run_dir),
                "steps_completed": int(training["training"]["steps_completed"]),
                "formal_passed": bool(formal["passed"]),
                "state_passed": state_passed,
                "formal": formal,
            }
        )
    return {
        "runs": rows,
        "formal_passes": sum(row["formal_passed"] for row in rows),
        "state_passes": sum(row["state_passed"] for row in rows),
        "all_ran_full_budget": all(row["steps_completed"] == 4000 for row in rows),
    }


def classify_a113f(
    *, generic_ce_state: int, generic_closure_state: int, structured_ce_state: int
) -> dict[str, Any]:
    if generic_ce_state != 0:
        return {
            "code": "generic_reference_rescued_by_budget",
            "early_stop_primary": True,
            "judgment": "fixed budget rescues the generic A1.12-BOTH reference",
        }
    if any(value not in {0, 3} for value in (generic_closure_state, structured_ce_state)):
        return {
            "code": "fixed_budget_seed_instability_persists",
            "early_stop_primary": False,
            "judgment": "full budget does not remove seed instability in the single-factor repairs",
        }
    if generic_closure_state == 3 and structured_ce_state == 3:
        return {
            "code": "two_fixed_budget_repairs_sufficient",
            "early_stop_primary": True,
            "judgment": "both single-factor repairs become stable when early stopping is removed",
        }
    if generic_closure_state == 3:
        return {
            "code": "closure_fixed_budget_sufficient",
            "early_stop_primary": True,
            "judgment": "closure alone becomes stable under the fixed budget",
        }
    if structured_ce_state == 3:
        return {
            "code": "structured_transition_fixed_budget_sufficient",
            "early_stop_primary": True,
            "judgment": "structured transition alone becomes stable under the fixed budget",
        }
    return {
        "code": "fixed_budget_does_not_rescue_single_factors",
        "early_stop_primary": False,
        "judgment": "neither single-factor repair is rescued by the full budget",
    }


def assess_a113f(
    generic_ce_dirs: Sequence[Path],
    generic_closure_dirs: Sequence[Path],
    structured_ce_dirs: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    arms = {
        "GENERIC-CE": _arm(generic_ce_dirs),
        "GENERIC-CLOSURE": _arm(generic_closure_dirs),
        "STRUCTURED-CE": _arm(structured_ce_dirs),
    }
    classification = classify_a113f(
        generic_ce_state=int(arms["GENERIC-CE"]["state_passes"]),
        generic_closure_state=int(arms["GENERIC-CLOSURE"]["state_passes"]),
        structured_ce_state=int(arms["STRUCTURED-CE"]["state_passes"]),
    )
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.13F fixed-budget early-stop method audit",
        "arms": arms,
        "classification": classification,
        "gates": {
            "three_runs_per_arm": all(len(arm["runs"]) == 3 for arm in arms.values()),
            "all_runs_full_budget": all(
                arm["all_ran_full_budget"] for arm in arms.values()
            ),
            "generic_reference_still_failed_0_of_3": int(
                arms["GENERIC-CE"]["state_passes"]
            )
            == 0,
        },
        "early_stop_primary": bool(classification["early_stop_primary"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
