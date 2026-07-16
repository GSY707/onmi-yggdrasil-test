from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .a1_8_data import DATA_SPLITS


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _fingerprints(data_dir: Path) -> set[str]:
    fingerprints: set[str] = set()
    for split in DATA_SPLITS:
        with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
            fingerprints.update(json.loads(line)["fingerprint"] for line in handle if line.strip())
    return fingerprints


def _trajectory(evaluation: dict[str, Any], split: str, length: int) -> float:
    return float(evaluation["splits"][split]["by_length"][str(length)]["trajectory_full_exact"])


def _stability(evaluation: dict[str, Any], split: str, step: int) -> dict[str, float] | None:
    row = evaluation["stability"][split]["per_step"].get(str(step))
    if row is None:
        return None
    return {
        "state_full_exact": float(row["state_full_exact"]),
        "latent_norm_mean": float(row["latent_norm_mean"]),
        "prototype_margin_mean": float(row["prototype_margin_mean"]),
        "perturbation_amplification_mean": float(row["initial_perturbation_amplification_mean"]),
        "perturbation_amplification_p95": float(row["initial_perturbation_amplification_p95"]),
    }


def _run_row(run_dir: Path) -> dict[str, Any]:
    manifest = _read(run_dir / "data" / "manifest.json")
    audit = _read(run_dir / "data-audit.json")
    training = _read(run_dir / "formal" / "results.json")
    evaluation = _read(run_dir / "formal" / "eval.json")
    intervention_path = run_dir / "formal" / "interventions.json"
    intervention = _read(intervention_path) if intervention_path.exists() else None
    steps_completed = int(training["training"]["steps_completed"])
    batch_size = int(training["train_spec"]["batch_size"])
    if batch_size % 16:
        raise ValueError("formal A1.8 assessment expects batch size divisible by the 16 training lengths")
    transitions_per_step = (batch_size // 16) * sum(range(1, 17))
    processed_examples = int(training["training"].get("processed_examples", steps_completed * batch_size))
    processed_transitions = int(training["training"].get("processed_transitions", steps_completed * transitions_per_step))
    active_parameters = int(training["parameter_report"]["trainable_parameters"])
    return {
        "name": run_dir.name,
        "data_seed": int(manifest["data_seed"]),
        "model_seed": int(training["train_spec"]["model_seed"]),
        "data_audit_passed": bool(audit["passed"]),
        "formal_passed": bool(evaluation["passed"]),
        "intervention_run": intervention is not None,
        "intervention_passed": bool(intervention["passed"]) if intervention is not None else None,
        "best_checkpoint_step": int(training["best_checkpoint_step"]),
        "steps_completed": steps_completed,
        "processed_examples": processed_examples,
        "processed_transitions": processed_transitions,
        "estimated_training_flops_parameter_proxy": 6 * active_parameters * processed_transitions,
        "training_seconds": float(training["training"]["seconds"]),
        "trajectory": {
            "short": {str(length): _trajectory(evaluation, "short_regression", length) for length in range(1, 7)},
            "supported": {str(length): _trajectory(evaluation, "supported_in_range", length) for length in (8, 12, 16)},
            "relation": {str(length): _trajectory(evaluation, "relation_in_range", length) for length in (8, 12, 16)},
            "supported_ood": {str(length): _trajectory(evaluation, "supported_ood", length) for length in (20, 24, 32)},
            "relation_ood": {str(length): _trajectory(evaluation, "relation_ood", length) for length in (20, 24, 32)},
        },
        "stability": {
            "supported_t16": _stability(evaluation, "supported_in_range", 16),
            "relation_t16": _stability(evaluation, "relation_in_range", 16),
            "supported_t24": _stability(evaluation, "supported_ood", 24),
            "relation_t24": _stability(evaluation, "relation_ood", 24),
            "supported_t32": _stability(evaluation, "supported_ood", 32),
            "relation_t32": _stability(evaluation, "relation_ood", 32),
        },
        "formal_gates": evaluation["gates"],
    }


def _metric_summary(rows: Sequence[dict[str, Any]], section: str, length: int) -> dict[str, float]:
    values = [float(row["trajectory"][section][str(length)]) for row in rows]
    return {"minimum": min(values), "mean": mean(values), "maximum": max(values)}


def build_a18_assessment_summary(
    artifact_root: Path,
    output: Path | None = None,
    *,
    a17_summary_path: Path | None = None,
) -> dict[str, Any]:
    run_dirs = sorted(path for path in (artifact_root / "runs").iterdir() if (path / "formal" / "eval.json").exists())
    rows = [_run_row(path) for path in run_dirs]
    if not rows:
        raise RuntimeError(f"no completed A1.8 runs found under {artifact_root / 'runs'}")
    run_fingerprints = {path.name: _fingerprints(path / "data") for path in run_dirs}
    cross_run_overlap = {
        f"{left}__{right}": len(run_fingerprints[left] & run_fingerprints[right])
        for left, right in combinations(sorted(run_fingerprints), 2)
    }
    target_summary = {
        "runs": len(rows),
        "data_audit_passes": sum(row["data_audit_passed"] for row in rows),
        "formal_passes": sum(row["formal_passed"] for row in rows),
        "intervention_runs": sum(row["intervention_run"] for row in rows),
        "intervention_passes": sum(row["intervention_passed"] is True for row in rows),
        "total_processed_examples": sum(row["processed_examples"] for row in rows),
        "total_processed_transitions": sum(row["processed_transitions"] for row in rows),
        "total_training_seconds": sum(row["training_seconds"] for row in rows),
        "total_estimated_training_flops_parameter_proxy": sum(row["estimated_training_flops_parameter_proxy"] for row in rows),
        "supported_t16": _metric_summary(rows, "supported", 16),
        "relation_t16": _metric_summary(rows, "relation", 16),
        "supported_t20": _metric_summary(rows, "supported_ood", 20),
        "relation_t20": _metric_summary(rows, "relation_ood", 20),
        "supported_t24": _metric_summary(rows, "supported_ood", 24),
        "relation_t24": _metric_summary(rows, "relation_ood", 24),
        "supported_t32_diagnostic": _metric_summary(rows, "supported_ood", 32),
        "relation_t32_diagnostic": _metric_summary(rows, "relation_ood", 32),
    }
    comparison: dict[str, Any] | None = None
    if a17_summary_path is not None and a17_summary_path.exists():
        a17 = _read(a17_summary_path)["target_summary"]
        comparison = {
            "a1_7_supported_t16_mean": float(a17["stress_supported_16"]["mean"]),
            "a1_7_relation_t16_mean": float(a17["stress_relation_16"]["mean"]),
            "a1_8_supported_t16_mean": target_summary["supported_t16"]["mean"],
            "a1_8_relation_t16_mean": target_summary["relation_t16"]["mean"],
            "supported_t16_delta": target_summary["supported_t16"]["mean"] - float(a17["stress_supported_16"]["mean"]),
            "relation_t16_delta": target_summary["relation_t16"]["mean"] - float(a17["stress_relation_16"]["mean"]),
        }
    cost_path = artifact_root / "cost-benchmark.json"
    gates = {
        "exactly_three_runs": len(rows) == 3,
        "three_unique_data_seeds": len({row["data_seed"] for row in rows}) == 3,
        "three_unique_model_seeds": len({row["model_seed"] for row in rows}) == 3,
        "cross_run_fingerprint_overlap_zero": all(value == 0 for value in cross_run_overlap.values()),
        "all_data_audits_passed": target_summary["data_audit_passes"] == 3,
        "all_formal_gates_passed": target_summary["formal_passes"] == 3,
        "all_causal_interventions_run_and_passed": target_summary["intervention_runs"] == 3 and target_summary["intervention_passes"] == 3,
        "cost_benchmark_present": cost_path.exists(),
    }
    evidence_judgment = {
        "random_depth_training_resolves_t16_drift": target_summary["supported_t16"]["minimum"] >= 0.95 and target_summary["relation_t16"]["minimum"] >= 0.95,
        "random_depth_training_generalizes_through_t24": target_summary["supported_t24"]["minimum"] >= 0.90 and target_summary["relation_t24"]["minimum"] >= 0.90,
        "t16_intrinsic_divergence_hypothesis_supported": False,
        "compute_matched_horizon_effect_isolated": False,
        "t32_is_only_diagnostic": True,
        "a1_8_stage_passed": all(gates.values()),
        "qwen_or_anonymous_workspace_validated": False,
    }
    result = {
        "schema_version": "yggdrasil.v2-a1.8.assessment.v1",
        "artifact_root": str(artifact_root),
        "runs": rows,
        "cross_run_fingerprint_overlap": cross_run_overlap,
        "target_summary": target_summary,
        "a1_7_comparison": comparison,
        "gates": gates,
        "evidence_judgment": evidence_judgment,
        "cost_benchmark": _read(cost_path) if cost_path.exists() else None,
        "passed": all(gates.values()),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
