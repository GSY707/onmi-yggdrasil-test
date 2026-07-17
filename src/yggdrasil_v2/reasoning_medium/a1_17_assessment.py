from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from yggdrasil_v2.reasoning_medium.a1_15_assessment import _arm, _read
from yggdrasil_v2.reasoning_medium.a1_13_models import (
    A113ReasonerConfig,
    A113TransitionClosureReasoner,
)


ASSESSMENT_SCHEMA = (
    "yggdrasil.v2-a1.17.paired-objective-initialization-assessment.v1"
)


def classify_a117(
    *,
    trigger_valid: bool,
    reference_state: int,
    coupled_ce_state: int,
    coupled_ce_full: int,
    coupled_noce_state: int | None,
    coupled_noce_full: int | None,
) -> dict[str, Any]:
    if not trigger_valid or reference_state != 3:
        return {
            "code": "trigger_or_reference_invalid",
            "localized": False,
            "judgment": "the paired audit lacks its preregistered failed-new-seed and 3/3 reference conditions",
        }
    if coupled_ce_full == 3:
        return {
            "code": "initialization_basin_primary",
            "localized": True,
            "judgment": "coupled CE succeeds on all paired reference initializations, so new-seed failure is initialization-basin instability",
        }
    if coupled_noce_state is None or coupled_noce_full is None:
        return {
            "code": "paired_noce_required",
            "localized": False,
            "judgment": "coupled CE failed 3/3; the preregistered paired no-CE arm is required",
        }
    counts = (coupled_ce_state, coupled_noce_state)
    if any(value not in {0, 3} for value in counts):
        return {
            "code": "objective_initialization_interaction",
            "localized": False,
            "judgment": "paired outcomes remain seed-unstable, so objective topology and initialization interact",
        }
    if coupled_ce_state == 0 and coupled_noce_full == 3:
        return {
            "code": "redundant_answer_ce_direct_failure",
            "localized": True,
            "judgment": "on identical successful reference initializations, removing duplicate answer CE fully rescues query coupling",
        }
    if coupled_ce_state == 0 and coupled_noce_state == 0:
        return {
            "code": "independent_answer_auxiliary_gradient_required",
            "localized": True,
            "judgment": "both coupled objectives fail on reference initializations that pass with independent pooled-answer CE; the independent answer objective is an optimization scaffold in the current system",
        }
    if coupled_ce_state == 3 and coupled_noce_state == 3:
        return {
            "code": "initialization_basin_primary",
            "localized": True,
            "judgment": "both coupled objectives succeed on the paired reference initializations; new-seed failure is initialization-basin instability",
        }
    return {
        "code": "paired_objective_mixed_failure",
        "localized": False,
        "judgment": "the paired arms do not support a unique preregistered root-cause class",
    }


def _seed_map(arm: dict[str, Any]) -> list[int]:
    return sorted(int(row["formal"]["model_seed"]) for row in arm["runs"])


def _paired_initialization_identity(seeds: Sequence[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        independent = A113TransitionClosureReasoner(
            A113ReasonerConfig(
                structured_transition=True,
                prototype_closure_objective=True,
                query_coupled_answer=False,
            )
        )
        torch.manual_seed(seed)
        coupled = A113TransitionClosureReasoner(
            A113ReasonerConfig(
                structured_transition=True,
                prototype_closure_objective=True,
                query_coupled_answer=True,
            )
        )
        independent_state = independent.state_dict()
        coupled_state = coupled.state_dict()
        expected_names = {
            name for name in independent_state if not name.startswith("answer_head.")
        }
        names_match = set(coupled_state) == expected_names
        values_match = names_match and all(
            torch.equal(value, independent_state[name])
            for name, value in coupled_state.items()
        )
        rows.append(
            {
                "model_seed": seed,
                "shared_tensor_count": len(coupled_state),
                "shared_names_match": names_match,
                "shared_values_bitwise_equal": values_match,
            }
        )
    return {"runs": rows, "passed": all(row["shared_values_bitwise_equal"] for row in rows)}


def assess_a117(
    a113_assessment_path: Path,
    a115_assessment_path: Path,
    a116_assessment_path: Path,
    coupled_ce_dirs: Sequence[Path],
    output_path: Path,
    coupled_noce_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    a113 = _read(a113_assessment_path)
    a115 = _read(a115_assessment_path)
    a116 = _read(a116_assessment_path)
    reference = a113["arms"]["STRUCTURED-CLOSURE"]
    coupled_ce = _arm(coupled_ce_dirs)
    coupled_noce = _arm(coupled_noce_dirs) if coupled_noce_dirs else None
    trigger_valid = (
        int(a115["arm"]["state_passes"]) < 3
        and int(a116["arm"]["state_passes"]) < 3
    )
    classification = classify_a117(
        trigger_valid=trigger_valid,
        reference_state=int(reference["state_passes"]),
        coupled_ce_state=int(coupled_ce["state_passes"]),
        coupled_ce_full=int(coupled_ce["formal_passes"]),
        coupled_noce_state=(
            int(coupled_noce["state_passes"]) if coupled_noce is not None else None
        ),
        coupled_noce_full=(
            int(coupled_noce["formal_passes"]) if coupled_noce is not None else None
        ),
    )
    expected_seeds = [20261321, 20261322, 20261323]
    seed_pairing = {
        "expected_model_seeds": expected_seeds,
        "reference_model_seeds": _seed_map(reference),
        "coupled_ce_model_seeds": _seed_map(coupled_ce),
        "coupled_noce_model_seeds": (
            _seed_map(coupled_noce) if coupled_noce is not None else None
        ),
    }
    seed_pairing["passed"] = (
        seed_pairing["reference_model_seeds"] == expected_seeds
        and seed_pairing["coupled_ce_model_seeds"] == expected_seeds
        and (
            coupled_noce is None
            or seed_pairing["coupled_noce_model_seeds"] == expected_seeds
        )
    )
    initialization_identity = _paired_initialization_identity(expected_seeds)
    if bool(classification["localized"]) and not (
        bool(seed_pairing["passed"]) and bool(initialization_identity["passed"])
    ):
        classification = {
            "code": "paired_integrity_failure",
            "localized": False,
            "judgment": "outcome pattern is diagnostic but paired seed or shared-initialization identity failed",
        }
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.17 paired objective x initialization audit",
        "trigger": {
            "a1_15_state_passes": int(a115["arm"]["state_passes"]),
            "a1_16_state_passes": int(a116["arm"]["state_passes"]),
            "valid": trigger_valid,
        },
        "reference": {
            "arm": "A1.13 STRUCTURED-CLOSURE",
            "state_passes": int(reference["state_passes"]),
            "formal_passes": int(reference["formal_passes"]),
        },
        "arms": {
            "COUPLED-CE": coupled_ce,
            "COUPLED-NOCE": coupled_noce,
        },
        "seed_pairing": seed_pairing,
        "shared_initialization_identity": initialization_identity,
        "classification": classification,
        "gates": {
            "trigger_valid": trigger_valid,
            "reference_state_passed_3_of_3": int(reference["state_passes"]) == 3,
            "paired_seed_identity": bool(seed_pairing["passed"]),
            "shared_initialization_bitwise_equal": bool(
                initialization_identity["passed"]
            ),
            "root_cause_localized": bool(classification["localized"]),
        },
        "root_cause_localized": bool(classification["localized"]),
        "architecture_validated": False,
        "evidence_boundary": "exact-symbolic three-register diagnostic core only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
