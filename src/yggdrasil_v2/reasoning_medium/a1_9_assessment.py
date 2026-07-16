from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .a1_9_cache import cached_fingerprints


ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.9.assessment-summary.v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _length_metric(formal: dict[str, Any], split: str, length: int, metric: str = "trajectory_full_exact") -> float:
    return float(formal["splits"][split]["by_length"][str(length)][metric])


def assess_a19(
    run_dirs: Sequence[Path],
    overfit_results_path: Path,
    cost_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if len(run_dirs) != 3:
        raise ValueError("A1.9 assessment requires exactly three preregistered run directories")
    overfit = _read(overfit_results_path)
    cost = _read(cost_path)
    runs: list[dict[str, Any]] = []
    fingerprint_sets: list[set[str]] = []
    for run_dir in run_dirs:
        audit_path = run_dir / "cache-audit.json"
        formal_path = run_dir / "formal" / "eval.json"
        intervention_path = run_dir / "formal" / "interventions.json"
        result_path = run_dir / "formal" / "results.json"
        audit = _read(audit_path)
        formal = _read(formal_path)
        interventions = _read(intervention_path)
        result = _read(result_path)
        cache_dir = run_dir / "cache"
        cache_manifest = _read(cache_dir / "manifest.json")
        fingerprints: set[str] = set()
        for split in ("train", "validation", "short_regression", "supported_in_range", "relation_in_range", "supported_ood", "relation_ood", "causal_core"):
            fingerprints.update(cached_fingerprints(cache_dir, split))
        fingerprint_sets.append(fingerprints)
        runs.append(
            {
                "run_dir": str(run_dir),
                "adapter_seed": formal["adapter_seed"],
                "data_seed": formal["data_seed"],
                "core_seed": formal["core_seed"],
                "cache_audit_passed": audit["passed"],
                "formal_passed": formal["passed"],
                "interventions_passed": interventions["passed"],
                "core_hash_unchanged": result["core_hash_unchanged"] and formal["gates"]["core_state_hash_unchanged"],
                "formal_eval_before_interventions": formal_path.stat().st_mtime_ns <= intervention_path.stat().st_mtime_ns,
                "trajectory": {
                    "supported_t16": _length_metric(formal, "supported_in_range", 16),
                    "relation_t16": _length_metric(formal, "relation_in_range", 16),
                    "supported_t24": _length_metric(formal, "supported_ood", 24),
                    "relation_t24": _length_metric(formal, "relation_ood", 24),
                    "supported_t32_diagnostic": _length_metric(formal, "supported_ood", 32),
                    "relation_t32_diagnostic": _length_metric(formal, "relation_ood", 32),
                },
                "minimum_mapping_accuracy": min(
                    formal["splits"][split]["aggregate"]["mapping_accuracy"]["minimum"]
                    for split in formal["splits"]
                ),
                "same_answer_trajectory": interventions["results"]["same_answer_different_trajectory"]["trajectory_full_exact"],
                "query_swap_trajectory": interventions["results"]["query_swap"]["trajectory_full_exact"],
                "query_swap_answer": interventions["results"]["query_swap"]["final_query_token_accuracy"],
                "no_hidden_trajectory": interventions["results"]["no_hidden"]["trajectory_full_exact"],
                "independent_role_shuffle_trajectory": interventions["results"]["independent_role_shuffle"]["trajectory_full_exact"],
                "training": result["training"],
                "parameters": result["parameter_report"],
                "qwen_role_cache": {
                    "examples": sum(row["examples"] for row in cache_manifest["splits"].values()),
                    "encoding_seconds_excluding_model_and_tokenizer_load": cache_manifest["seconds"],
                    "peak_allocated_bytes": cache_manifest["peak_allocated_bytes"],
                    "disk_bytes": sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file()),
                },
            }
        )
    overlap = {
        "run-1__run-2": len(fingerprint_sets[0] & fingerprint_sets[1]),
        "run-1__run-3": len(fingerprint_sets[0] & fingerprint_sets[2]),
        "run-2__run-3": len(fingerprint_sets[1] & fingerprint_sets[2]),
    }
    gates = {
        "overfit32_passed": bool(overfit.get("overfit_passed")),
        "three_cache_audits_passed": all(run["cache_audit_passed"] for run in runs),
        "three_formal_gates_passed": all(run["formal_passed"] for run in runs),
        "three_hidden_intervention_gates_passed": all(run["interventions_passed"] for run in runs),
        "three_core_hashes_unchanged": all(run["core_hash_unchanged"] for run in runs),
        "formal_before_interventions": all(run["formal_eval_before_interventions"] for run in runs),
        "cross_run_fingerprint_overlap_zero": all(value == 0 for value in overlap.values()),
        "cost_artifact_present": cost.get("schema_version") == "yggdrasil.v2-a1.9.cached-boundary-cost.v1",
    }
    target_summary = {
        "formal_passes": sum(run["formal_passed"] for run in runs),
        "intervention_passes": sum(run["interventions_passed"] for run in runs),
        "supported_t16_minimum": min(run["trajectory"]["supported_t16"] for run in runs),
        "relation_t16_minimum": min(run["trajectory"]["relation_t16"] for run in runs),
        "supported_t24_minimum": min(run["trajectory"]["supported_t24"] for run in runs),
        "relation_t24_minimum": min(run["trajectory"]["relation_t24"] for run in runs),
        "supported_t32_diagnostic_minimum": min(run["trajectory"]["supported_t32_diagnostic"] for run in runs),
        "relation_t32_diagnostic_minimum": min(run["trajectory"]["relation_t32_diagnostic"] for run in runs),
        "mapping_minimum": min(run["minimum_mapping_accuracy"] for run in runs),
        "same_answer_trajectory_minimum": min(run["same_answer_trajectory"] for run in runs),
        "query_swap_trajectory_minimum": min(run["query_swap_trajectory"] for run in runs),
        "query_swap_answer_minimum": min(run["query_swap_answer"] for run in runs),
        "no_hidden_trajectory_maximum": max(run["no_hidden_trajectory"] for run in runs),
        "independent_role_shuffle_trajectory_maximum": max(run["independent_role_shuffle_trajectory"] for run in runs),
        "training_seconds_total": sum(float(run["training"]["seconds"]) for run in runs),
        "processed_examples_total": sum(int(run["training"]["processed_examples"]) for run in runs),
        "processed_transitions_total": sum(int(run["training"]["processed_transitions"]) for run in runs),
        "adapter_trainable_parameters": [int(run["parameters"]["adapter_trainable_parameters"]) for run in runs],
        "mean_formal_training_seconds": mean(float(run["training"]["seconds"]) for run in runs),
        "qwen_role_encoding_seconds_total_excluding_model_and_tokenizer_load": sum(
            float(run["qwen_role_cache"]["encoding_seconds_excluding_model_and_tokenizer_load"]) for run in runs
        ),
        "qwen_role_encoding_peak_allocated_bytes_maximum": max(
            int(run["qwen_role_cache"]["peak_allocated_bytes"]) for run in runs
        ),
        "qwen_role_cache_disk_bytes_total": sum(int(run["qwen_role_cache"]["disk_bytes"]) for run in runs),
    }
    evidence_boundaries = {
        "oracle_role_segmented_qwen_boundary_drives_frozen_structured_core": all(gates.values()),
        "contextual_hidden_role_counterfactuals_passed": gates["three_hidden_intervention_gates_passed"],
        "qwen_and_core_frozen": gates["three_core_hashes_unchanged"],
        "full_source_learned_query_reader_validated": False,
        "anonymous_k_slot_workspace_validated": False,
        "generic_recurrent_transformer_reasoner_validated": False,
        "matched_text_cot_pareto_validated": False,
        "audit_readout_validated": False,
        "complete_v2_a_validated": False,
        "t32_is_diagnostic_only": True,
        "a1_9_stage_passed": all(gates.values()),
    }
    summary = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "V2-A1.9 frozen Qwen hidden boundary",
        "overfit32": {"path": str(overfit_results_path), "passed": overfit.get("overfit_passed")},
        "runs": runs,
        "cross_run_fingerprint_overlap": overlap,
        "target_summary": target_summary,
        "cost": cost,
        "gates": gates,
        "evidence_boundaries": evidence_boundaries,
        "passed": all(gates.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
