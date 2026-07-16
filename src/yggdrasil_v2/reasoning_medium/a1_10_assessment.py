from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .a1_10_cache import cached_fingerprints


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.10.assessment-summary.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _length_metric(formal: dict[str, Any], split: str, length: int, metric: str) -> float:
    return float(formal["splits"][split]["by_length"][str(length)][metric])


def assess_a110(
    run_dirs: Sequence[Path],
    overfit_results: Path,
    cost_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    overfit = _read(overfit_results)
    runs: list[dict[str, Any]] = []
    fingerprints: dict[str, set[str]] = {}
    for run_dir in run_dirs:
        audit = _read(run_dir / "cache-audit.json")
        training = _read(run_dir / "formal" / "results.json")
        formal = _read(run_dir / "formal" / "formal-eval.json")
        intervention_path = run_dir / "interventions" / "results.json"
        interventions = _read(intervention_path) if intervention_path.exists() else None
        cache_manifest = _read(run_dir / "cache" / "manifest.json")
        run_name = run_dir.name
        fingerprints[run_name] = cached_fingerprints(run_dir / "cache", "train")
        runs.append(
            {
                "run_dir": str(run_dir),
                "model_seed": training["train_spec"]["model_seed"],
                "data_seed": training["train_spec"]["data_seed"],
                "cache_audit_passed": bool(audit["passed"]),
                "formal_passed": bool(formal["passed"]),
                "interventions_executed": interventions is not None,
                "interventions_passed": bool(interventions and interventions["passed"]),
                "formal_eval_before_interventions": bool(
                    not intervention_path.exists()
                    or (run_dir / "formal" / "formal-eval.json").stat().st_mtime <= intervention_path.stat().st_mtime
                ),
                "trajectory": {
                    "supported_t16": _length_metric(formal, "supported_in_range", 16, "trajectory_full_exact"),
                    "relation_t16": _length_metric(formal, "relation_in_range", 16, "trajectory_full_exact"),
                    "supported_t24": _length_metric(formal, "supported_ood", 24, "trajectory_full_exact"),
                    "relation_t24": _length_metric(formal, "relation_ood", 24, "trajectory_full_exact"),
                    "supported_t32_diagnostic": _length_metric(formal, "supported_ood", 32, "trajectory_full_exact"),
                    "relation_t32_diagnostic": _length_metric(formal, "relation_ood", 32, "trajectory_full_exact"),
                },
                "answer": {
                    "supported_t16": _length_metric(formal, "supported_in_range", 16, "final_answer_accuracy"),
                    "relation_t16": _length_metric(formal, "relation_in_range", 16, "final_answer_accuracy"),
                    "supported_t24": _length_metric(formal, "supported_ood", 24, "final_answer_accuracy"),
                    "relation_t24": _length_metric(formal, "relation_ood", 24, "final_answer_accuracy"),
                },
                "training": training["training"],
                "parameters": training["parameter_report"],
                "qwen_full_token_cache": {
                    "examples": sum(row["examples"] for row in cache_manifest["splits"].values()),
                    "source_tokens": sum(row["source_tokens"] for row in cache_manifest["splits"].values()),
                    "encoding_seconds_excluding_model_and_tokenizer_load": cache_manifest[
                        "full_token_encoding_seconds_excluding_model_and_tokenizer_load"
                    ],
                    "peak_allocated_bytes": cache_manifest["peak_allocated_bytes"],
                    "disk_bytes": sum(path.stat().st_size for path in (run_dir / "cache").rglob("*") if path.is_file()),
                },
            }
        )
    overlap: dict[str, int] = {}
    names = sorted(fingerprints)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap[f"{left}__{right}"] = len(fingerprints[left] & fingerprints[right])
    cost = _read(cost_path) if cost_path.exists() else None
    formal_passes = sum(run["formal_passed"] for run in runs)
    intervention_passes = sum(run["interventions_passed"] for run in runs)
    gates = {
        "overfit32_passed": bool(overfit.get("overfit_passed")),
        "three_cache_audits_passed": len(runs) == 3 and all(run["cache_audit_passed"] for run in runs),
        "three_formal_gates_passed": len(runs) == 3 and formal_passes == 3,
        "three_hidden_intervention_gates_passed": len(runs) == 3 and intervention_passes == 3,
        "formal_before_interventions": all(run["formal_eval_before_interventions"] for run in runs),
        "cross_run_fingerprint_overlap_zero": all(value == 0 for value in overlap.values()),
        "cost_artifact_present": cost is not None,
    }
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.10 full-text anonymous workspace and generic recurrent reasoner",
        "overfit32": {"path": str(overfit_results), "passed": bool(overfit.get("overfit_passed"))},
        "runs": runs,
        "cross_run_fingerprint_overlap": overlap,
        "target_summary": {
            "formal_passes": formal_passes,
            "intervention_passes": intervention_passes,
            "supported_t16_trajectory_minimum": min(run["trajectory"]["supported_t16"] for run in runs),
            "relation_t16_trajectory_minimum": min(run["trajectory"]["relation_t16"] for run in runs),
            "supported_t24_trajectory_minimum": min(run["trajectory"]["supported_t24"] for run in runs),
            "relation_t24_trajectory_minimum": min(run["trajectory"]["relation_t24"] for run in runs),
            "training_seconds_total": sum(run["training"]["seconds"] for run in runs),
            "processed_examples_total": sum(run["training"]["processed_examples"] for run in runs),
            "processed_recurrent_transitions_total": sum(
                run["training"]["processed_recurrent_transitions"] for run in runs
            ),
            "qwen_encoding_seconds_total_excluding_model_and_tokenizer_load": sum(
                run["qwen_full_token_cache"]["encoding_seconds_excluding_model_and_tokenizer_load"] for run in runs
            ),
            "qwen_full_token_cache_disk_bytes_total": sum(
                run["qwen_full_token_cache"]["disk_bytes"] for run in runs
            ),
        },
        "cost": cost,
        "gates": gates,
        "evidence_boundaries": {
            "full_source_qwen_hidden_used_without_oracle_spans": True,
            "anonymous_k_slot_workspace_implemented": True,
            "generic_shared_recurrent_transformer_implemented": True,
            "joint_architecture_configuration_validated": all(gates.values()),
            "individual_boundary_workspace_reasoner_contributions_identified": False,
            "autonomous_stopping_validated": False,
            "matched_text_cot_pareto_validated": False,
            "audit_readout_validated": False,
            "complete_v2_a_validated": False,
            "t32_is_diagnostic_only": True,
        },
        "passed": all(gates.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
