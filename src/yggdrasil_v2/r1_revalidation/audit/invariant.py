from __future__ import annotations

"""Single fail-closed orchestration layer for the G02-G06 invariant audit."""

from pathlib import Path
from typing import Any

from .artifact import load_artifact
from .pairs import pair_bundle
from .provenance import provenance_bundle
from .replay import replay_bundle
from .structure import structure_bundle


METRIC_ORDER = [
    "G02_M01_input_tree_and_seal_exact",
    "G02_M02_fixed_qualification_profile_exact",
    "G02_M03_manifest_identity_counts_hashes_exact",
    "G02_M04_snapshot_profile_set_hash_digest_exact",
    "G02_M05_runtime_and_stream_profile_exact",
    "G02_M06_exact_schema_rate",
    "G02_M07_model_view_exact_rate",
    "G02_M08_recursive_forbidden_field_count",
    "G02_M09_tokenizer_identity_exact",
    "G02_M10_token_recount_match_rate",
    "G02_M11_max_source_tokens",
    "G03_M01_ere_replay_digest_rate",
    "G03_M02_cps_replay_digest_rate",
    "G03_M03_answer_teacher_budget_contract_rate",
    "G03_M04_claim_kind_profile_exact",
    "G03_M05_claim_identity_profile_exact",
    "G03_M06_claim_polarity_binding_rate",
    "G03_M07_fresh_claim_truth_rate",
    "G03_M08_claim_single_leaf_pair_rate",
    "G03_M09_claim_text_parse_rate",
    "G04_M01_fingerprint_recompute_rate",
    "G04_M02_unpaired_overlap_count",
    "G04_M03_causal_pair_profile_exact",
    "G04_M04_causal_pair_contract_rate",
    "G04_M05_mutation_family_deviation_max",
    "G04_M06_language_pair_profile_exact",
    "G04_M07_reversible_language_binding_rate",
    "G04_M08_fold_and_component_contract_exact",
    "G05_M01_core_profile_and_depth_exact",
    "G05_M02_event_declaration_exact",
    "G05_M03_fresh_event_ablation_rate",
    "G05_M04_typed_provenance_derivation_rate",
    "G05_M05_provenance_class_coverage_exact",
    "G05_M06_control_trace_binding_rate",
    "G05_M07_provenance_balance_exact",
    "G06_M01_rich_none_profile_exact",
    "G06_M02_pstar_depth_unique_minimum_exact",
    "G06_M03_valid_suboptimal_exact",
    "G06_M04_invalid_candidate_profile_exact",
    "G06_M05_failure_reason_profile_exact",
    "G06_M06_invalid_length_profile_exact",
    "G06_M07_none_single_leaf_skeleton_exact",
    "G06_M08_none_ratio_exact",
    "G06_M09_derived_composition_witnesses_exact",
    "G06_M10_canonical_role_leak_count",
]
GATE_METRICS = {
    gate: [metric for metric in METRIC_ORDER if metric.startswith(f"{gate}_")]
    for gate in ("G02", "G03", "G04", "G05", "G06")
}


def _failed(reason: str) -> dict[str, Any]:
    return {"value": False, "numerator": 0, "denominator": 1, "passed": False, "failures": [reason]}


def _merge(metrics: dict[str, dict[str, Any]], incoming: Any) -> None:
    if not isinstance(incoming, dict):
        return
    for key, value in incoming.items():
        if key in metrics and isinstance(value, dict) and set(value) == {
            "value", "numerator", "denominator", "passed", "failures"
        }:
            metrics[key] = value


def audit_invariant_bundle(root: str | Path) -> dict[str, Any]:
    """Audit one sealed qualification bundle without modifying input bytes."""

    metrics = {metric: _failed("metric was not evaluated") for metric in METRIC_ORDER}
    failures: list[str] = []
    artifact: dict[str, Any] = {
        "records": [], "causal_pairs": [], "language_pairs": [], "composition": {}, "g02": {}, "errors": []
    }
    replay: dict[str, Any] = {"outputs": {}, "g03": {}, "failures": []}
    pairs: dict[str, Any] = {"g04": {}}
    provenance: dict[str, Any] = {"g05": {}, "derived": {}}
    structure: dict[str, Any] = {"g06": {}, "derived": {}}

    try:
        artifact = load_artifact(root)
        _merge(metrics, artifact.get("g02"))
        failures.extend(str(item) for item in artifact.get("errors", []))
    except Exception as exc:
        failures.append(f"G02:{type(exc).__name__}:{exc}")

    try:
        replay = replay_bundle(artifact.get("records", []))
        _merge(metrics, replay.get("g03"))
        failures.extend(str(item) for item in replay.get("failures", []))
    except Exception as exc:
        failures.append(f"G03:{type(exc).__name__}:{exc}")

    try:
        pairs = pair_bundle(
            artifact.get("records", []),
            artifact.get("causal_pairs", []),
            artifact.get("language_pairs", []),
            artifact.get("composition", {}),
            replay,
        )
        _merge(metrics, pairs.get("g04"))
    except Exception as exc:
        failures.append(f"G04:{type(exc).__name__}:{exc}")

    try:
        provenance = provenance_bundle(artifact.get("records", []), artifact.get("composition", {}), replay)
        _merge(metrics, provenance.get("g05"))
    except Exception as exc:
        failures.append(f"G05:{type(exc).__name__}:{exc}")

    try:
        structure = structure_bundle(artifact.get("records", []), artifact.get("composition", {}), replay)
        _merge(metrics, structure.get("g06"))
    except Exception as exc:
        failures.append(f"G06:{type(exc).__name__}:{exc}")

    gates = {
        gate: all(metrics[metric].get("passed") is True for metric in metric_ids)
        for gate, metric_ids in GATE_METRICS.items()
    }
    failed_metrics = [metric for metric in METRIC_ORDER if metrics[metric].get("passed") is not True]
    failures.extend(failed_metrics)
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v9.r0b-invariant.audit-report.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": {metric: metrics[metric] for metric in METRIC_ORDER},
        "derivations": {
            "ere": provenance.get("derived", {}),
            "cps": structure.get("derived", {}),
        },
        "failures": sorted(set(failures)),
    }
