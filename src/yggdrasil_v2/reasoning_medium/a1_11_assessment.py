from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.11.orthogonal-localization-assessment.v2"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(run_dirs: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        formal_path = run_dir / "formal" / "formal-eval.json"
        intervention_path = run_dir / "interventions" / "results.json"
        training = _read(run_dir / "formal" / "results.json")
        formal = _read(formal_path)
        interventions = _read(intervention_path) if intervention_path.exists() else None
        rows.append(
            {
                "run_dir": str(run_dir),
                "formal_passed": bool(formal["passed"]),
                "interventions_executed": interventions is not None,
                "interventions_passed": bool(interventions and interventions["passed"]),
                "formal_before_interventions": bool(
                    interventions is None or formal_path.stat().st_mtime <= intervention_path.stat().st_mtime
                ),
                "training": training["training"],
                "formal": formal,
            }
        )
    return {
        "runs": rows,
        "formal_passes": sum(row["formal_passed"] for row in rows),
        "intervention_passes": sum(row["interventions_passed"] for row in rows),
        "formal_before_interventions": all(row["formal_before_interventions"] for row in rows),
    }


def _classification(
    boundary_passes: int,
    reasoner_passes: int,
    *,
    boundary_overfit: bool = True,
    reasoner_overfit: bool = True,
) -> dict[str, Any]:
    if not boundary_overfit and reasoner_overfit and reasoner_passes == 3:
        return {
            "code": "learned_boundary_interface_primary_failure",
            "conclusive": True,
            "judgment": "the learned full-text boundary cannot satisfy the frozen-core interface even on overfit32, while the anonymous reasoner passes formal with the validated boundary",
        }
    if boundary_overfit and not reasoner_overfit and boundary_passes == 3:
        return {
            "code": "anonymous_reasoner_primary_failure",
            "conclusive": True,
            "judgment": "the anonymous reasoner cannot satisfy overfit32, while the learned boundary passes formal with the structured core",
        }
    if not boundary_overfit and (not reasoner_overfit or reasoner_passes == 0):
        return {
            "code": "both_isolated_arms_fail_at_independent_gates",
            "conclusive": True,
            "judgment": "the learned boundary interface fails its overfit pointer gate and the anonymous reasoner also fails independently; A1.10 is not a combination-only failure",
        }
    if not reasoner_overfit and boundary_passes == 0:
        return {
            "code": "both_isolated_arms_fail_at_independent_gates",
            "conclusive": True,
            "judgment": "the anonymous reasoner fails overfit32 and the learned boundary also fails independently; A1.10 is not a combination-only failure",
        }
    if boundary_passes == 3 and reasoner_passes == 0:
        return {
            "code": "anonymous_reasoner_primary_failure",
            "conclusive": True,
            "judgment": "full-text learned boundary can work with the structured core; anonymous binding/generic transition is the primary isolated failure",
        }
    if boundary_passes == 0 and reasoner_passes == 3:
        return {
            "code": "full_text_boundary_primary_failure",
            "conclusive": True,
            "judgment": "anonymous generic recurrence can work with the validated typed boundary; full-text role discovery is the primary isolated failure",
        }
    if boundary_passes == 3 and reasoner_passes == 3:
        return {
            "code": "joint_interface_or_optimization_interaction",
            "conclusive": True,
            "judgment": "both isolated replacements work, while A1.10 fails; the remaining fault is the integrated interface or joint optimization",
        }
    if boundary_passes == 0 and reasoner_passes == 0:
        return {
            "code": "both_isolated_replacements_fail",
            "conclusive": True,
            "judgment": "both replacements fail independently; A1.10 is not a pure combination-only failure",
        }
    return {
        "code": "seed_unstable_inconclusive",
        "conclusive": False,
        "judgment": "at least one arm passes only one or two seeds; no unique causal localization is justified",
    }


def assess_a111(
    boundary_run_dirs: Sequence[Path],
    reasoner_run_dirs: Sequence[Path],
    boundary_overfit: Path,
    reasoner_overfit: Path,
    a19_assessment: Path,
    a110_assessment: Path,
    output_path: Path,
) -> dict[str, Any]:
    boundary = _arm(boundary_run_dirs)
    reasoner = _arm(reasoner_run_dirs)
    a19 = _read(a19_assessment)
    a110 = _read(a110_assessment)
    overfit = {
        "boundary": bool(_read(boundary_overfit)["overfit_passed"]),
        "reasoner": bool(_read(reasoner_overfit)["overfit_passed"]),
    }
    classification = _classification(
        boundary["formal_passes"],
        reasoner["formal_passes"],
        boundary_overfit=overfit["boundary"],
        reasoner_overfit=overfit["reasoner"],
    )
    references = {
        "a1_9_oracle_roles_structured_core_passed": bool(a19["passed"]),
        "a1_10_full_text_anonymous_reasoner_failed": not bool(a110["passed"]),
        "a1_10_formal_passes": int(a110["target_summary"]["formal_passes"]),
    }
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.11 orthogonal Boundary x Reasoner fault localization",
        "overfit32": overfit,
        "boundary_arm": boundary,
        "reasoner_arm": reasoner,
        "reference_cells": references,
        "matrix": {
            "oracle_roles__structured_core": "A1.9 passed 3/3",
            "learned_full_text__structured_core": (
                f"A1.11-Boundary passed {boundary['formal_passes']}/3"
                if overfit["boundary"]
                else "A1.11-Boundary overfit32 failed; formal stopped"
            ),
            "oracle_roles__anonymous_reasoner": (
                f"A1.11-Reasoner passed {reasoner['formal_passes']}/3"
                if overfit["reasoner"]
                else "A1.11-Reasoner overfit32 failed; formal stopped"
            ),
            "learned_full_text__anonymous_reasoner": "A1.10 failed 0/3",
        },
        "classification": classification,
        "localization_strength": {
            "combination_only_explanation_rejected": (
                overfit["reasoner"] and reasoner["formal_passes"] == 0
            ),
            "anonymous_reasoner_is_a_sufficient_independent_failure": (
                overfit["reasoner"] and reasoner["formal_passes"] == 0
            ),
            "boundary_has_an_independent_strict_interface_defect": not overfit["boundary"],
            "boundary_task_level_formal_generalization_known": bool(boundary_run_dirs),
            "anonymous_workspace_vs_generic_transition_separated": False,
        },
        "gates": {
            "both_overfit32_passed": all(overfit.values()),
            "three_runs_for_each_overfit_passing_arm": (
                (not overfit["boundary"] or len(boundary_run_dirs) == 3)
                and (not overfit["reasoner"] or len(reasoner_run_dirs) == 3)
            ),
            "overfit_stop_rule_respected": (
                (overfit["boundary"] or len(boundary_run_dirs) == 0)
                and (overfit["reasoner"] or len(reasoner_run_dirs) == 0)
            ),
            "formal_before_any_intervention": boundary["formal_before_interventions"] and reasoner["formal_before_interventions"],
            "reference_matrix_verified": (
                references["a1_9_oracle_roles_structured_core_passed"]
                and references["a1_10_full_text_anonymous_reasoner_failed"]
                and references["a1_10_formal_passes"] == 0
            ),
            "localization_conclusive": bool(classification["conclusive"]),
        },
        "experiment_completed": True,
        "architecture_validated": (
            overfit["boundary"]
            and overfit["reasoner"]
            and boundary["formal_passes"] == 3
            and reasoner["formal_passes"] == 3
        ),
        "evidence_boundaries": {
            "boundary_arm_retains_operation_mask_and_typed_queries": True,
            "reasoner_arm_uses_exact_symbolic_roles_without_qwen_or_adapters": True,
            "neither_arm_is_the_full_target_architecture": True,
            "matched_text_cot_pareto_validated": False,
            "complete_v2_a_validated": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
