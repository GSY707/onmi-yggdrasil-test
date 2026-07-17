from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from yggdrasil_v2.reasoning_medium.a1_15_assessment import _arm, _read


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.16.redundant-answer-loss-assessment.v1"


def classify_a116(
    *, trigger_valid: bool, formal_passes: int, intervention_passes: int
) -> dict[str, Any]:
    if not trigger_valid:
        return {
            "code": "trigger_invalid",
            "localized": False,
            "judgment": "A1.15 did not satisfy the preregistered failure condition",
        }
    if formal_passes not in {0, 3}:
        return {
            "code": "no_answer_ce_seed_unstable",
            "localized": False,
            "judgment": "removing the redundant answer CE is seed-unstable",
        }
    if formal_passes == 0:
        return {
            "code": "redundant_answer_ce_removal_insufficient",
            "localized": False,
            "judgment": "removing the redundant answer CE does not rescue the coupled core",
        }
    if intervention_passes != 3:
        return {
            "code": "ordinary_pass_causal_failure",
            "localized": False,
            "judgment": "ordinary formal passes but causal interventions do not pass 3/3",
        }
    return {
        "code": "redundant_answer_ce_primary",
        "localized": True,
        "judgment": "the duplicate answer CE is sufficient to explain the coupled-core failure",
    }


def assess_a116(
    a115_assessment_path: Path,
    overfit_result_path: Path,
    run_dirs: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    a115 = _read(a115_assessment_path)
    overfit = _read(overfit_result_path)
    trigger_valid = (
        bool(overfit.get("overfit_passed"))
        and int(a115["arm"]["state_passes"]) < 3
    )
    arm = _arm(run_dirs)
    classification = classify_a116(
        trigger_valid=trigger_valid,
        formal_passes=int(arm["formal_passes"]),
        intervention_passes=int(arm["intervention_passes"]),
    )
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.16 redundant answer-loss audit",
        "trigger": {
            "a1_15_assessment": str(a115_assessment_path),
            "a1_15_state_passes": int(a115["arm"]["state_passes"]),
            "overfit_result": str(overfit_result_path),
            "overfit_passed": bool(overfit.get("overfit_passed")),
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
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
