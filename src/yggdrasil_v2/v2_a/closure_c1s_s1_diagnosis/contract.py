from __future__ import annotations

"""冻结的 C1S SRW S1 失败归因合同。

本模块只描述一个只读诊断身份。它不提供训练入口，也不允许把诊断
结果解释成 S2、S3 或 V2-A qualification 的授权。
"""

from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1s.srw.s1-failure-attribution"
IDENTITY = "V2-A-CLOSURE-C1S-SRW-S1-FAILURE-ATTRIBUTION-20260831-1"

OUTPUT_ROOT = Path("artifacts/v2-a/closure-c1s-srw-s1-failure-attribution-20260831-1")
LEASE_PATH = Path(
    "artifacts/v2-a/closure-c1s-srw-s1-failure-attribution-20260831-1.diagnosis-lease.jsonl"
)
PREFLIGHT_ROOT = Path(
    "tmp/v2-a-closure-c1s-srw-s1-failure-attribution-preflight-20260831-1"
)
PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1s-srw-s1-failure-attribution-preflight-20260831-1.preflight-lease.jsonl"
)
DESIGN_DOC = Path("docs/v2-a-closure-c1s-s1-failure-attribution.md")

# These are read-only inputs.  No checkpoint, optimizer state, or old result
# may be replaced by a diagnostic output.
S1_ROOT = Path("artifacts/v2-a/closure-c1s-srw-s1-overfit32-20260829-1")
S1_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1s-srw-s1-preflight-20260829-1")
C1_CACHE_ROOT = Path("artifacts/v2-a/closure-c1-cache-20260825-1")
C0R_DATA_ROOT = Path("artifacts/v2-a/closure-c0r-data-trace-20260824-1")

S1_RESULT_SHA256 = (
    "9D0B06E14C7B2E5E0CDBDC7ED7C35D259BDD540E4BE3F65DA1E7F82D80E38892"
)
S1_EVIDENCE_SEAL_SHA256 = (
    "FAFF301AB40BEFB3D99106EEA8B0752F63DB96BC5390F7D5068A4D89F668A9D2"
)
S1_ENDPOINT_SHA256 = (
    "D6F44CFC2C6F0A6F0841AA232E0DE73DA08EB7D0873D26E930C4E28E72AAA2E9"
)
S1_SOURCE_IDENTITY = (
    "E3C5D7EC8C75518F9AB9EBE7574459A7DB27990F35D57D153FA63D869A8B77F4"
)
S1_IDENTITY = "V2-A-CLOSURE-C1S-SRW-S1-OVERFIT32-20260829-1"
S1_STATUS = "FAIL_V2_A_C1S_S1_QUALIFICATION"
S1_SCHEDULE_SHA256 = (
    "D45AD251B60B94263A0CCF4603DCF42B463A532DEF8FFCB3EE82CF7F87BED6A5"
)
S1_PREFLIGHT_IDENTITY = f"{S1_IDENTITY}-PREFLIGHT"
S1_PREFLIGHT_STATUS = "PASS_V2_A_C1S_S1_PREFLIGHT"
S1_PREFLIGHT_RESULT_SHA256 = (
    "EBEA1FAB4FC637854DA1DFAAE587F3D8C1367CEEFAFE14222EE6486D4A10AD85"
)
S1_PREFLIGHT_SEAL_SHA256 = (
    "B3F3862E0245F4C58483CAFE35CE933BF8D4DAEE08A0F930E4A21AF5384CA8F1"
)
TARGET_BANK_SHA256 = (
    "9A6257595929648BA4F0365FFBB074E8F2A8AB75012FBAA0C25D0C816C9CB4D7"
)
SPLIT_LEDGER_SHA256 = (
    "4CE56FD78FD9D8075B0B52D64221BE09252EBB4758C3A03F5FB210058041CFD8"
)
CACHE_SOURCE_IDENTITY = S1_SOURCE_IDENTITY

FAMILIES = ("CPS", "ERE")
DIAGNOSTIC_STAGES = ("D000", "D001", "D002", "D003", "D004", "D005")
BOOTSTRAP_SEED = 2026083101
BOOTSTRAP_REPLICATES = 10_000

# D001 makes the distinction that was hidden by the failed S1 objective
# explicit: a margin hinge and a categorical necessity Gate are different
# statements.
D001_OBJECTIVE_ALIGNMENT: dict[str, Any] = {
    "answer_loss": "answer_cross_entropy",
    "no_core_training_term": "relu(0.5 - (full_margin - no_core_margin))",
    "no_core_training_hinge_margin_drop": 0.5,
    "no_core_gate_metric": "answer_accuracy_after_no_core",
    "no_core_gate_upper_bound": 0.25,
    "functional_k_direct_loss": False,
    "state_loss": "global_masked_micro_bce",
    "state_gate_metric": "family_feature_balanced_accuracy",
    "state_gate_floor": 0.95,
    "answer_derived_features": ["CPS.final_winner", "ERE.query_semantic_match"],
    "answer_derived_features_hard_gate": False,
    "required_checks": [
        "static_loss_to_gate_mapping",
        "endpoint_component_loss_and_gradient_credit",
        "no_core_hinge_active_fraction",
        "macro_micro_and_rare_feature_weighting",
    ],
}

# D002: all paths use the same answer-margin endpoint.  final_delta is an
# architectural counterfactual and must not be described as a natural rollout.
D002_H0_CORE_PATHS: dict[str, Any] = {
    "state_notation": "B=(P0,A,Q,O,S); hT=R(P0,O,S,A)",
    "full": "f(R(P0,O,S,A),Q)",
    "h0": "f(P0,A,Q)",
    "core_only": "f(R(mean_present(P0),O,S,A),Q)",
    "h0_relevant_replace": "f(P0 with query-owner payload replaced by leave-one-out other-present mean,A,Q)",
    "global_mean": "f(all present payloads replaced by their row mean,A,Q)",
    "final_delta": "f(hT-P0,A,Q)",
    "final_delta_status": "architectural_counterfactual_only",
    "per_step_outputs": "h0..h10 and answer margin at every step",
    "interventions": [
        "zero_payload",
        "zero_operation_state_only; route weights and activity are retained",
        "zero_query",
        "reverse_boundary_operation_order",
        "answer_permutation_metric_null",
    ],
    "margin": "logit(correct)-logsumexp(logits(wrong))",
    "family_primary": True,
    "h0_direct_point_floor": 0.75,
    "h0_direct_wilson_lower_floor": 0.65,
    "h0_required_point_upper": 0.25,
    "margin_drop_record_bootstrap_lower_floor": 0.50,
    "core_only_point_upper_for_h0_direct": 0.35,
}

# D003 separates what the task mathematically requires from what the model
# happens to encode.  It is deliberately not a single-slot/copy heuristic.
D003_ARITY: dict[str, Any] = {
    "task_oracle": "immutable source simulator and AST only",
    "scope": "sealed_s1_overfit32_post_hoc_diagnosis_only",
    "records_per_family": 16,
    "future_s2_or_formal_rows_consumed": False,
    "task_arity_dimensions": [
        "answer_support",
        "certificate",
        "counterfactual_influence",
    ],
    "dimension_separation": (
        "registered certificate reachability never substitutes for answer support "
        "or counterfactual causal arity"
    ),
    "source_ast_policy": "only separately pinned immutable source records; never target-bank fallback",
    "subset_enumeration_max_objects": 8,
    "eligible_baseline_margin": 0.5,
    "minimum_eligible_records_per_family": 8,
    "two_contributor_definition": "each content slot replaced by leave-one-out other-content mean; drop >= 0.10 * total available margin",
    "two_contributor_denominator": "full_margin - h0_margin; nonpositive records are ineligible",
    "content_contributor_policy": "content_mask only; CPS decision slot excluded",
    "copy_all_metric": "copy_all_margin / base_margin",
    "copy_all_single_slot_point_floor": 0.80,
    "copy_all_single_slot_wilson_lower_floor": 0.65,
    "two_contributor_mult_address_wilson_lower_floor": 0.65,
    "copy_collapse_margin_drop_record_bootstrap_lower_floor": 0.50,
    "invalid_intervention": "fail_closed",
    "final_winner_in_hard_gate": False,
}

# D004 hard-gates only features whose labels are not derived from the answer.
# Readout experiments are in-memory, cross-fitted, and cannot mutate the model.
D004_TEMPORAL: dict[str, Any] = {
    "hard_features": {
        "ERE": ["touched", "changed", "operation_source", "operation_target"],
        "CPS": ["processed", "running_best"],
    },
    "reported_not_gated": {"CPS": ["final_winner"], "ERE": ["query_semantic_match"]},
    "target_adequacy": {
        "positive_per_family_feature": 32,
        "negative_per_family_feature": 32,
        "cross_step_changes": 16,
        "source_visible_replay": True,
    },
    "motion_relative_delta_floor": 0.05,
    "fixed_channel_auc_floor": 0.70,
    "frozen_decoder_balanced_accuracy_floor": 0.80,
    "decoder": {
        "kind": "cross_fitted_in_memory_ridge",
        "features": "source-only model temporal state_features; no target-derived predictor",
        "primary_metric": "record_macro_balanced_accuracy_for_inner_selection_natural_nulls_and_frozen_decoder",
        "observation_micro_metric": "diagnostic_only_never_subtracted_from_record_macro",
        "rank_grid": [8, 16, 32],
        "lambda_grid": [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0],
        "selection": "inner-fold only",
        "optimizer_steps": 0,
    },
    "nulls": ["time_shuffle", "temporal_mean", "target_shuffle"],
    "forbidden_features": [
        "answer",
        "final_winner",
        "margin",
        "vjp",
        "projection_output",
        "target_norm",
        "positive_request_mask",
        "any_target_derived_value",
    ],
}

DECISION_THRESHOLDS: dict[str, Any] = {
    "objective": {"min_active_fraction": 0.01, "min_gradient_norm": 1.0e-12},
    "axis_A": {
        "h0_direct_point_min": 0.75,
        "h0_direct_wilson_lower_min": 0.65,
        "h0_low_point_max": 0.25,
        "full_point_min": 0.90,
        "core_only_point_min": 0.75,
        "core_only_point_max": 0.35,
        "full_h0_margin_drop_min": 0.50,
        "h0_relevant_margin_drop_min": 0.50,
        "full_core_margin_drop_min": 0.50,
        "final_delta_point_max": 0.35,
        "full_final_delta_margin_drop_min": 0.50,
    },
    "axis_B": {
        "task_causal_assessed_point_min": 0.80,
        "task_multi_object_causal_point_min": 0.80,
        "task_low_order_supported_point_min": 0.80,
        "baseline_eligible_point_min": 0.80,
        "minimum_eligible_records_per_family": 8,
        "copy_retention_point_min": 0.80,
        "copy_retention_wilson_lower_min": 0.65,
        "two_contributor_point_max": 0.50,
        "two_contributor_wilson_lower_min": 0.65,
        "copy_collapse_bootstrap_lower_min": 0.50,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    },
    "axis_C": {
        "target_adequacy_point_min": 1.0,
        "latent_motion_point_min": 0.05,
        "latent_separation_auc_min": 0.70,
        "decoder_balanced_accuracy_min": 0.80,
        "readout_minus_null_min": 0.05,
        "static_null_gap_max": 0.02,
    },
}

D005_DECISIONS: dict[str, Any] = {
    "family_primary": True,
    "overall_role": "summary_only",
    "axis_A": ["H0_DIRECT", "CORE_RESIDUAL", "CORE_REQUIRED", "MIXED", "INCONCLUSIVE"],
    "axis_B": ["TASK_LOW_ORDER", "SINGLE_SLOT_COPY", "MULTI_ADDRESS_SUPPORTED", "INCONCLUSIVE"],
    "axis_C": [
        "TARGET_INSUFFICIENT",
        "READOUT_INSUFFICIENT",
        "LATENT_NOT_FORMED",
        "MIXED",
        "INCONCLUSIVE",
    ],
    "composite_pass": False,
    "authorization": "nothing",
    "v2a_passed": False,
    "completion_requires": [
        "all_required_family_cells",
        "valid_answer_permutation_control",
        "no_structurally_invalid_task_arity_record",
        "source_replayed_temporal_targets_and_all_registered_nulls",
        "zero_model_mutation",
        "terminal_external_input_replay",
    ],
}

PINS: dict[str, Any] = {
    "s1_root": S1_ROOT.as_posix(),
    "s1_result": S1_RESULT_SHA256,
    "s1_evidence_seal": S1_EVIDENCE_SEAL_SHA256,
    "s1_endpoint": S1_ENDPOINT_SHA256,
    "s1_source_identity": S1_SOURCE_IDENTITY,
    "s1_preflight_root": S1_PREFLIGHT_ROOT.as_posix(),
    "s1_preflight_result": S1_PREFLIGHT_RESULT_SHA256,
    "s1_preflight_seal": S1_PREFLIGHT_SEAL_SHA256,
    "target_bank_gzip": TARGET_BANK_SHA256,
    "split_ledger": SPLIT_LEDGER_SHA256,
    "c1_cache_root": C1_CACHE_ROOT.as_posix(),
    "c0r_data_root": C0R_DATA_ROOT.as_posix(),
    "cache_source_identity": CACHE_SOURCE_IDENTITY,
}

NEVER_AUTHORIZES = (
    "S2",
    "S3",
    "single-seed formal",
    "new training",
    "C2",
    "V2-A PASS",
    "V2-B",
    "V2-C",
)


def contract_manifest() -> dict[str, Any]:
    """返回只读、JSON-native 的冻结合同快照。"""

    return deepcopy(
        {
            "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
            "identity": IDENTITY,
            "diagnostic_only": True,
            "formal": False,
            "training_started": False,
            "authorizes": "nothing",
            "v2a_passed": False,
            "output_root": OUTPUT_ROOT.as_posix(),
            "lease_path": LEASE_PATH.as_posix(),
            "preflight_root": PREFLIGHT_ROOT.as_posix(),
            "design_doc": DESIGN_DOC.as_posix(),
            "pinned_inputs": PINS,
            "stages": list(DIAGNOSTIC_STAGES),
            "families": list(FAMILIES),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "d001_objective_alignment": D001_OBJECTIVE_ALIGNMENT,
            "d002_h0_core_paths": D002_H0_CORE_PATHS,
            "d003_task_and_model_arity": D003_ARITY,
            "d004_temporal_latent_readout": D004_TEMPORAL,
            "d005_decisions": D005_DECISIONS,
            "decision_thresholds": DECISION_THRESHOLDS,
            "optimizer_step_allowed": False,
            "parameter_update_allowed": False,
            "model_or_checkpoint_write_allowed": False,
            "old_root_write_allowed": False,
            "training_data_write_allowed": False,
            "source_only_forward_boundary": ["source_hidden", "source_mask"],
            "never_authorizes": list(NEVER_AUTHORIZES),
        }
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "C0R_DATA_ROOT",
    "C1_CACHE_ROOT",
    "D001_OBJECTIVE_ALIGNMENT",
    "D002_H0_CORE_PATHS",
    "D003_ARITY",
    "D004_TEMPORAL",
    "D005_DECISIONS",
    "DECISION_THRESHOLDS",
    "DESIGN_DOC",
    "DIAGNOSTIC_STAGES",
    "FAMILIES",
    "IDENTITY",
    "LEASE_PATH",
    "NEVER_AUTHORIZES",
    "OUTPUT_ROOT",
    "PINS",
    "PREFLIGHT_LEASE",
    "PREFLIGHT_ROOT",
    "S1_ENDPOINT_SHA256",
    "S1_EVIDENCE_SEAL_SHA256",
    "S1_IDENTITY",
    "S1_PREFLIGHT_IDENTITY",
    "S1_PREFLIGHT_ROOT",
    "S1_PREFLIGHT_RESULT_SHA256",
    "S1_PREFLIGHT_SEAL_SHA256",
    "S1_PREFLIGHT_STATUS",
    "S1_RESULT_SHA256",
    "S1_ROOT",
    "S1_SCHEDULE_SHA256",
    "S1_SOURCE_IDENTITY",
    "S1_STATUS",
    "SCHEMA_PREFIX",
    "SPLIT_LEDGER_SHA256",
    "TARGET_BANK_SHA256",
    "contract_manifest",
]
