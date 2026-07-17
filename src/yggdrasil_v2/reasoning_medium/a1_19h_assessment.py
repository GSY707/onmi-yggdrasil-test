from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.19h.h1-opaque-handle.assessment.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_a119h_h1(
    *, formal_passes: int, causal_passes: int, integrity_passed: bool
) -> dict[str, Any]:
    if not integrity_passed:
        return {
            "code": "h1_integrity_failure",
            "h1_passed": False,
            "judgment": "seed, budget, deployment-strip, or address contract integrity failed",
        }
    if formal_passes != 3:
        return {
            "code": "opaque_handle_core_formal_failure",
            "h1_passed": False,
            "judgment": "the exchangeable opaque-handle core did not reach three-seed formal closure",
        }
    if causal_passes != 3:
        return {
            "code": "opaque_handle_core_causal_failure",
            "h1_passed": False,
            "judgment": "formal behavior did not survive the full opaque-handle causal suite",
        }
    return {
        "code": "opaque_handle_hybrid_core_confirmed",
        "h1_passed": True,
        "judgment": "fixed register semantics were removed while three-seed formal, causal, alias, and slot-permutation gates remained closed",
    }


def assess_a119h_h1(
    overfit_path: Path, run_dirs: Sequence[Path], output_path: Path
) -> dict[str, Any]:
    overfit = _read(overfit_path)
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
                "fixed_budget_completed": bool(
                    training["training"]["fixed_budget_completed"]
                ),
                "deployment_auxiliary_stripped": bool(
                    formal["gates"]["deployment_auxiliary_absent"]
                ),
                "formal_passed": bool(formal["passed"]),
                "interventions_passed": bool(intervention["passed"]),
                "training": training["training"],
                "formal": formal,
                "interventions": intervention,
            }
        )
    expected_model = [20261921, 20261922, 20261923]
    expected_data = [20260721, 20260722, 20260723]
    expected_mapping = [20261931, 20261932, 20261933]
    integrity = {
        "overfit_passed": bool(overfit.get("overfit_passed")),
        "three_formal_runs": len(rows) == 3,
        "model_seeds_exact": [row["model_seed"] for row in rows] == expected_model,
        "data_seeds_exact": [row["data_seed"] for row in rows] == expected_data,
        "mapping_seeds_exact": [row["mapping_seed"] for row in rows]
        == expected_mapping,
        "all_fixed_budget_completed": all(
            row["fixed_budget_completed"] for row in rows
        ),
        "all_deployments_auxiliary_stripped": all(
            row["deployment_auxiliary_stripped"] for row in rows
        ),
        "all_handles_equality_only": all(
            row["formal"]["deployment_integrity"]["opaque_handle_usage"]
            == "equality routing only"
            for row in rows
        ),
        "all_fixed_register_semantics_absent": all(
            not row["formal"]["deployment_integrity"]["fixed_register_slot_semantics"]
            for row in rows
        ),
    }
    formal_passes = sum(row["formal_passed"] for row in rows)
    causal_passes = sum(row["interventions_passed"] for row in rows)
    classification = classify_a119h_h1(
        formal_passes=formal_passes,
        causal_passes=causal_passes,
        integrity_passed=all(integrity.values()),
    )
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.19H-H1 opaque-handle hybrid core",
        "overfit": {"path": str(overfit_path), "passed": bool(overfit.get("overfit_passed"))},
        "runs": rows,
        "formal_passes": formal_passes,
        "intervention_passes": causal_passes,
        "integrity": integrity,
        "classification": classification,
        "gates": {
            "overfit32_passed": bool(overfit.get("overfit_passed")),
            "formal_3_of_3": formal_passes == 3,
            "causal_3_of_3": causal_passes == 3,
            "integrity_passed": all(integrity.values()),
        },
        "h1_passed": bool(classification["h1_passed"]),
        "a119h_complete": False,
        "next_stage_allowed": "A1.19H-H2" if classification["h1_passed"] else None,
        "training_cost": {
            "formal_run_count": len(rows),
            "aggregate_training_seconds": sum(
                float(row["training"]["seconds"]) for row in rows
            ),
            "excludes_overfit_formal_eval_and_interventions": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
