from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _metric_summary(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {"minimum": min(values), "mean": mean(values), "maximum": max(values)}


def _formal_row(name: str, seed: int, directory: Path) -> dict[str, Any]:
    evaluation = _read(directory / "eval-all.json")
    intervention = _read(directory / "interventions.json")
    training = _read(directory / "results.json")
    return {
        "name": name,
        "seed": seed,
        "formal_passed": bool(evaluation["passed"]),
        "intervention_passed": bool(intervention["passed"]),
        "best_checkpoint_step": int(training["best_checkpoint_step"]),
        "steps_completed": int(training["training"]["steps_completed"]),
        "closure_weight": float(training["training"]["closure_weight"]),
        "target_architecture": bool(evaluation["integrity"].get("target_architecture", True)),
        "test": float(evaluation["splits"]["test"]["trajectory_full_exact"]),
        "length": float(evaluation["splits"]["length_heldout"]["trajectory_full_exact"]),
        "relation": float(evaluation["splits"]["relation_heldout"]["trajectory_full_exact"]),
        "causal_free": float(evaluation["splits"]["causal_core"]["trajectory_full_exact"]),
        "prefix": float(intervention["results"]["prefix_fidelity"]["trajectory_full_exact"]),
        "replacement": float(intervention["results"]["replacement"]["trajectory_full_exact"]),
        "deletion": float(intervention["results"]["deletion"]["trajectory_full_exact"]),
        "shuffled": float(intervention["results"]["shuffled_program"]["trajectory_full_exact"]),
    }


def _stress_row(name: str, path: Path) -> dict[str, Any]:
    result = _read(path)
    row: dict[str, Any] = {
        "name": name,
        "passed": bool(result["passed"]),
        "closure_weight": float(result["training_closure_weight"]),
        "target_architecture": bool(result["integrity"].get("target_architecture", True)),
        "supported_aggregate": float(result["splits"]["supported_length_stress"]["trajectory_full_exact"]),
        "relation_aggregate": float(result["splits"]["relation_length_stress"]["trajectory_full_exact"]),
    }
    for split_name, short in (("supported_length_stress", "supported"), ("relation_length_stress", "relation")):
        for length, metrics in result["per_length"][split_name].items():
            row[f"{short}_{length}"] = float(metrics["trajectory_full_exact"])
    return row


def build_a17_assessment_summary(artifact_root: Path, output: Path | None = None) -> dict[str, Any]:
    target_specs = (
        ("target-seed-20260715", 20260715, artifact_root / "c0-formal"),
        ("target-seed-20260716", 20260716, artifact_root / "multiseed" / "seed-20260716"),
        ("target-seed-20260717", 20260717, artifact_root / "multiseed" / "seed-20260717"),
    )
    target_runs = [_formal_row(name, seed, directory) for name, seed, directory in target_specs]
    ablation_specs = (
        ("target_content-only_with-closure", 20260715, artifact_root / "c0-formal"),
        ("content-only_no-closure", 20260715, artifact_root / "ablations" / "content-only_no-closure"),
        ("address-mixed_with-closure", 20260715, artifact_root / "ablations" / "address-mixed_with-closure"),
        ("address-mixed_no-closure", 20260715, artifact_root / "ablations" / "address-mixed_no-closure"),
    )
    ablations = [_formal_row(name, seed, directory) for name, seed, directory in ablation_specs]
    stress_names = (
        "target-seed-20260715",
        "target-seed-20260716",
        "target-seed-20260717",
        "content-only_no-closure",
        "address-mixed_with-closure",
        "address-mixed_no-closure",
    )
    stress_runs = [_stress_row(name, artifact_root / "stress" / f"{name}.json") for name in stress_names]
    target_stress = stress_runs[:3]
    target_summary = {
        "runs": len(target_runs),
        "formal_passes": sum(row["formal_passed"] for row in target_runs),
        "intervention_passes": sum(row["intervention_passed"] for row in target_runs),
        "stress_passes": sum(row["passed"] for row in target_stress),
        "test": _metric_summary(target_runs, "test"),
        "length": _metric_summary(target_runs, "length"),
        "relation": _metric_summary(target_runs, "relation"),
        "causal_free": _metric_summary(target_runs, "causal_free"),
        "stress_supported_16": _metric_summary(target_stress, "supported_16"),
        "stress_relation_16": _metric_summary(target_stress, "relation_16"),
    }
    by_name = {row["name"]: row for row in stress_runs}
    effects = {
        "closure_effect_content_only_at_supported_t16": by_name["target-seed-20260715"]["supported_16"] - by_name["content-only_no-closure"]["supported_16"],
        "closure_effect_address_mixed_at_supported_t16": by_name["address-mixed_with-closure"]["supported_16"] - by_name["address-mixed_no-closure"]["supported_16"],
        "address_separation_effect_with_closure_at_supported_t16": by_name["target-seed-20260715"]["supported_16"] - by_name["address-mixed_with-closure"]["supported_16"],
        "address_separation_effect_without_closure_at_supported_t16": by_name["content-only_no-closure"]["supported_16"] - by_name["address-mixed_no-closure"]["supported_16"],
    }
    result = {
        "schema_version": "yggdrasil.v2-a1.7.core-assessment.v1",
        "artifact_root": str(artifact_root),
        "target_runs": target_runs,
        "target_summary": target_summary,
        "ablations": ablations,
        "stress_runs": stress_runs,
        "controlled_effects": effects,
        "evidence_judgment": {
            "short_horizon_formal_reproducible": target_summary["formal_passes"] == target_summary["runs"],
            "causal_intervention_reproducible": target_summary["intervention_passes"] == target_summary["runs"],
            "strict_long_horizon_stress_reproducible": target_summary["stress_passes"] == target_summary["runs"],
            "closure_is_required_by_short_horizon_gate": False,
            "closure_materially_reduces_t16_drift": effects["closure_effect_content_only_at_supported_t16"] > 0.20 and effects["closure_effect_address_mixed_at_supported_t16"] > 0.20,
            "address_content_separation_has_independent_positive_evidence": False,
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
