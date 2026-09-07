from __future__ import annotations

"""Frozen successor contract for C1S semantic-routed workspace stages.

This module records eligibility and stop boundaries only.  It does not launch a
stage, claim a lease, create an artifact root, or provide a training runner.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..closure_c1 import contract as c1_contract


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1s-srw-successor"
IDENTITY = "V2-A-CLOSURE-C1S-SRW-S1-S2-S3-20260829-1"

DESIGN_DOC = Path("docs/v2-a-closure-c1s-s1-s2-s3-execution.md")

S1_IDENTITY = "V2-A-CLOSURE-C1S-SRW-S1-OVERFIT32-20260829-1"
S2_IDENTITY = "V2-A-CLOSURE-C1S-SRW-S2-DISCOVERY-20260829-1"
S3_IDENTITY = "V2-A-CLOSURE-C1S-SRW-S3-SINGLE-SEED-FORMAL-20260829-1"

S1_ROOT = Path("artifacts/v2-a/closure-c1s-srw-s1-overfit32-20260829-1")
S2_ROOT = Path("artifacts/v2-a/closure-c1s-srw-s2-discovery-20260829-1")
S3_ROOT = Path("artifacts/v2-a/closure-c1s-srw-s3-single-seed-formal-20260829-1")

S1_LEASE = Path(
    "artifacts/v2-a/closure-c1s-srw-s1-overfit32-20260829-1.preflight-lease.jsonl"
)
S2_LEASE = Path(
    "artifacts/v2-a/closure-c1s-srw-s2-discovery-20260829-1.preflight-lease.jsonl"
)
S3_LEASE = Path(
    "artifacts/v2-a/closure-c1s-srw-s3-single-seed-formal-20260829-1.preflight-lease.jsonl"
)

S1_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1s-srw-s1-preflight-20260829-1")
S2_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1s-srw-s2-preflight-20260829-1")
S3_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1s-srw-s3-preflight-20260829-1")
S1_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1s-srw-s1-preflight-20260829-1.preflight-lease.jsonl"
)
S2_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1s-srw-s2-preflight-20260829-1.preflight-lease.jsonl"
)
S3_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1s-srw-s3-preflight-20260829-1.preflight-lease.jsonl"
)

# The S0 contract remains the predecessor of S1.  The exact text is copied
# from the sealed predecessor result; successor code must not broaden or
# reinterpret its natural-language authorization.
S0_IDENTITY = "V2-A-CLOSURE-C1S-ADDRESSED-WORKSPACE-20260828-1"
S0_AUTHORIZED_SCOPE = (
    "one independent C1S S1 Overfit32 implementation and single-use run only"
)

S1_MODEL_SEED = 2026082901
S1_ORDER_SEED = 2026082902
S1_BOOTSTRAP_SEED = 2026082903
S1_PREFLIGHT_MODEL_SEED = 2026082890
S1_PREFLIGHT_ORDER_SEED = 2026082891
S2_MODEL_SEED = 2026082911
S2_ORDER_SEED = 2026082912
S2_BOOTSTRAP_SEED = 2026082913
S3_MODEL_SEED = 2026082921
S3_ORDER_SEED = 2026082922
S3_BOOTSTRAP_SEED = 2026082923

MODEL_CONFIG: dict[str, Any] = {
    "source_width": 2_048,
    "payload_width": 512,
    "address_width": 64,
    "slots": 8,
    "operations": 10,
    "reader_depth": 2,
    "attention_heads": 8,
    "ffn_width": 2_048,
    "answer_classes": 9,
    "auxiliary_width": 128,
    "route_temperature": 0.25,
    "training_auxiliary": True,
}
K1_CONFIG = {**deepcopy(MODEL_CONFIG), "slots": 1}

OPTIMIZATION_CONFIG: dict[str, Any] = {
    "optimizer": "AdamW",
    "boundary_lr": 1.0e-4,
    "transition_lr": 2.0e-4,
    "head_lr": 3.0e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "warmup_updates": 256,
    "schedule": "linear_warmup_then_cosine_to_zero",
    "autocast": "cuda_bfloat16",
    "batch_size": 8,
    "family_batch_size": 4,
    "cpu_intraop_threads": 2,
    "cpu_interop_threads": 1,
    "ordered_prefetch": 0,
    "checkpoint_selection": False,
    "optimizer_state_saved": False,
    "endpoint_model_writes_per_arm": 1,
    "loss_weights": {
        "answer": 1.0,
        "slot_identity": 0.50,
        "route": 1.0,
        "query_owner": 1.0,
        "presence": 0.20,
        "span_owner": 0.50,
        "state": 1.0,
        "operation_active": 0.20,
        "no_core_margin": 0.50,
        "wrong_start_margin": 0.50,
        "payload_binding_margin": 0.50,
        "operation_binding_margin": 0.50,
        "target_shuffle_margin": 0.50,
        "relevant_margin": 0.50,
        "irrelevant_consistency": 0.10,
    },
    "causal_margin": 0.50,
    "s1_causal_interval": 1,
    "s2_s3_causal_interval": 4,
}

MECHANISM_THRESHOLDS: dict[str, Any] = {
    "owner_accuracy": 0.80,
    "state_accuracy": 0.80,
    "no_core_accuracy_max": 0.25,
    "intervention_drop_bootstrap_lower": 0.05,
    "irrelevant_drop_abs_max": 0.10,
    "slot_permutation_max_abs": 1.0e-5,
    "strip_max_abs": 1.0e-6,
    "s2_k8_minus_k1_point": 0.05,
    "s2_k8_minus_k1_bootstrap_lower": 0.0,
    "functional_contributor_ratio": 0.10,
    "functional_min_effective_slots": 2.0,
    "duplicate_payload_logit_max_abs": 1.0e-6,
    "dynamic_feature_balanced_accuracy": 0.80,
    "public_owner_only_answer_accuracy_max": 0.35,
}

TARGET_CONTRACT: dict[str, Any] = {
    "identity": "C1S-SRW-SEMANTIC-TEMPORAL-TARGET-V1",
    "forward_visibility": "source_hidden_and_source_mask_only",
    "address_policy": {
        "CPS": "rank_of_semantic_object_in_public_present_LABEL_KEY_rows",
        "ERE": "unicode_NFKC_casefold_public_entity_name_lexical_rank_with_original_UTF8_tie_break",
        "ERE_normalization_collision": "fail_closed",
        "answer_derived_address_forbidden": True,
        "external_hash_permutation_forbidden": True,
    },
    "query_owner": {
        "CPS": "public_NONE_decision_object",
        "ERE": "public_query_entity_or_relation_target",
        "raw_answer_choice_owner_forbidden": True,
    },
    "temporal_state_shape": [10, 8, 8],
    "dynamic_state_features": {
        "ERE": ["touched", "changed", "operation_source", "operation_target"],
        "CPS": ["processed", "running_best", "final_winner"],
    },
    "answer_derived_auxiliary_label_boundary": {
        "ERE": ["query_semantic_match"],
        "CPS": ["final_winner"],
        "visibility": "detached_loss_and_evaluator_side_only",
        "standalone_mechanism_evidence": False,
        "requires_joint_non_answer_controls": True,
    },
    "operation_active_metric": "per_family_both_classes_temporal_change_and_balanced_accuracy",
    "capacity_boundary": {
        "model_slots": 8,
        "raw_answer_classes": 9,
        "nine_semantic_object_rows": "OOD_behavior_only",
        "unsupported_mechanism_rows_must_be_ood": True,
    },
    "causal_metric": "correct_minus_logsumexp_wrong_answer_margin_degradation",
    "raw_correct_logit_drop": "diagnostic_only_not_a_gate",
}

S1_SELECTION = 'sha256("C1S-SRW-S1-OVERFIT32-V1|example_id")'
S2_SELECTION = {
    "train": 'sha256("C1S-SRW-S2-DISCOVERY-TRAIN-V1|example_id")',
    "eval": 'sha256("C1S-SRW-S2-DISCOVERY-EVAL-V1|example_id")',
}

S1_CONFIG: dict[str, Any] = {
    "identity": S1_IDENTITY,
    "stage": "S1-overfit32-causal-qualification",
    "formal": False,
    "records": 32,
    "records_per_family": 16,
    "families": ["ERE", "CPS"],
    "selection": S1_SELECTION,
    "source_disjoint_from_s2_discovery": True,
    "batch_size": 8,
    "family_batch_size": 4,
    "maximum_updates": 4_000,
    "endpoint": "fixed_4000",
    "checkpoint_selection": False,
    "fresh_initialization": True,
    "model_seed": S1_MODEL_SEED,
    "order_seed": S1_ORDER_SEED,
    "bootstrap_seed": S1_BOOTSTRAP_SEED,
    "preflight_benchmark_model_seed": S1_PREFLIGHT_MODEL_SEED,
    "preflight_benchmark_order_seed": S1_PREFLIGHT_ORDER_SEED,
    "threshold_source": "S0_frozen_positive_and_k1_null_controls",
    "thresholds": {
        "answer_exact": 1.0,
        "owner_accuracy": 0.95,
        "state_accuracy": 0.95,
        "no_core_accuracy_max": 0.25,
        "causal_answer_margin_drop": 0.50,
        "irrelevant_answer_margin_drop_abs_max": 0.10,
    },
    "hard_gates": [
        "answer_exact_32_of_32_and_each_family",
        "per_step_address_ownership_and_state_target",
        "each_dynamic_state_feature_has_change_coverage_and_balanced_accuracy",
        "operation_active_has_per_family_change_coverage_and_balanced_accuracy",
        "query_owner_is_public_and_answer_independent",
        "duplicate_payload_different_owner_preserves_address_and_content",
        "slot_permutation_semantic_invariance",
        "query_relevant_pre_recurrence_mean_replace_changes_target",
        "padding_irrelevant_pre_recurrence_mean_replace_preserves_target",
        "recurrence_off_is_near_chance_and_bindings_are_causal",
        "payload_and_operation_zero_or_shuffle_break_the_target",
        "s0_single_slot_structural_control_replayed",
    ],
    "simulator_targets_are_detached_labels_only": True,
    "old_checkpoint_or_optimizer_allowed": False,
}

S2_CONFIG: dict[str, Any] = {
    "identity": S2_IDENTITY,
    "stage": "S2-discovery-versus-matched-k1",
    "formal": False,
    "train_per_family": 3_072,
    "eval_per_family": 512,
    "exclude_s1_records": True,
    "formal_validation_and_test_untouched": True,
    "batch_size": 8,
    "family_batch_size": 4,
    "epochs": 6,
    "maximum_updates_per_arm": 4_608,
    "endpoint": "fixed_4608",
    "checkpoint_selection": False,
    "gpu_hour_limit_per_arm": 4.0,
    "equal_examples": True,
    "equal_updates": True,
    "matched_k1": True,
    "model_seed": S2_MODEL_SEED,
    "order_seed": S2_ORDER_SEED,
    "bootstrap_seed": S2_BOOTSTRAP_SEED,
    "threshold_source": "S0_controls_plus_preregistered_behavior_floor",
    "behavior_point_floor": 0.75,
    "behavior_wilson_lower_floor": 0.70,
    "required_relative_gain_source": "pre_registered_practical_margin_and_paired_bootstrap",
    "required_relative_gain_point": 0.05,
    "required_relative_gain_bootstrap_lower": 0.0,
    "hard_gates": [
        "both_families_meet_absolute_behavior_floor",
        "k8_beats_k1_on_answer_in_each_family",
        "ownership_and_state_causality_pass_in_both_families",
        "functional_k8_in_positive_control_region",
        "matched_k1_is_a_valid_measured_single_state_control",
        "budget_and_accounting_match",
    ],
    "candidate_level_cps_closure_only": True,
    "ere_uses_entity_event_routes": True,
    "answer_independent_query_owner": True,
    "behavior_only_unsupported_ood_is_not_mechanism_evidence": True,
    "does_not_claim_action_level_complete_trajectory": True,
}

# These are the old C1 behavior thresholds only.  They are not evidence that
# the new SRW mechanism works and cannot be used for ownership/causal gates.
S3_BEHAVIOR_THRESHOLDS = deepcopy(c1_contract.THRESHOLDS)

S3_CONFIG: dict[str, Any] = {
    "identity": S3_IDENTITY,
    "stage": "S3-single-seed-formal",
    "formal": True,
    "fresh_initialization": True,
    "architecture": "K8_SRW_only",
    "full_train_records": 8_192,
    "full_train_per_family": 4_096,
    "maximum_updates": 6_144,
    "endpoint": "fixed_6144",
    "checkpoint_selection": False,
    "model_seed": S3_MODEL_SEED,
    "order_seed": S3_ORDER_SEED,
    "bootstrap_seed": S3_BOOTSTRAP_SEED,
    "behavior_threshold_source": "old_C1_thresholds_only",
    "behavior_thresholds": deepcopy(S3_BEHAVIOR_THRESHOLDS),
    "gate_order": [
        "G004_validation_behavior",
        "G004_OOD_behavior",
        "G004_causal_behavior",
        "G005_ownership_and_state",
        "G006_hidden_interventions",
        "G008_recurrence_interventions",
        "G009_deployment_auxiliary_strip",
        "G010_accounting_and_evidence_integrity",
    ],
    "fail_stop_after_each_gate": True,
    "unrun_stage_status": "NOT_RUN",
    "single_use": True,
    "retry_allowed": False,
    "old_checkpoint_or_optimizer_allowed": False,
    "candidate_level_cps_closure_only": True,
    "does_not_claim_action_level_complete_trajectory": True,
}

PINNED_PREDECESSOR_ROOTS: dict[str, dict[str, str]] = {
    "c1s_s0": {
        "root": "tmp/v2-a-closure-c1s-s0-preflight-20260828-1",
        "result_sha256": "022DD05065EDE6B663E1212BA980FD2D9F2AD6A89CAEC3B3DF357FFDA0FF698D",
        "seal_sha256": "2117B1EFA80C4D55C475CC41B10238D2518AA5C34951911EACEF5CA3A5F46E3D",
    },
    "c1_cache": {
        "root": "artifacts/v2-a/closure-c1-cache-20260825-1",
        "result_sha256": "A058687125FDC1315871733DD1499036A040636864665793B8D32B8C9D758D4A",
        "seal_sha256": "68CA34BB18F889405F875BF54B7CA2E55C3E9E549DE5DFE1C63C4F47E2B4DC34",
    },
    "c0r_readiness": {
        "root": "artifacts/v2-a/closure-c0r-20260824-1",
        "result_sha256": "391C846D2150FD27D2016A79A9C360F3CB4E1455EC71240720F772CF58A28D5E",
        "seal_sha256": "161FBB367DEE40D18CDD23521E7DA9EBA3257EE8BF57C1FA194886E43D140BA4",
    },
    "c0r_data_trace": {
        "root": "artifacts/v2-a/closure-c0r-data-trace-20260824-1",
        "result_sha256": "B2502F66CB5D9ECEB2CA0547CB280CC5346D1A24D2F8901F94E617E1B258FFF9",
        "seal_sha256": "5AFB37AB5950D04298FED4D2DAFA78102260A728101A5EDE4972D1EC0EEBB570",
    },
    "c1_formal": {
        "root": "artifacts/v2-a/closure-c1-single-seed-20260825-1",
        "result_sha256": "FE9F22B89E89092EDD0DCE1CC4FCBA926A3B8D4AB79123F237F4B9EDF379972C",
        "seal_sha256": "BEB46576A91ADC08B9E7A1D51249E4CA5F03BFFFC7069D1E791FBBAC8749FA19",
    },
    "c1r_formal": {
        "root": "artifacts/v2-a/closure-c1r-staged-credit-20260827-1",
        "result_sha256": "54F426D5825A00A3DE9C57F16657F9DE499C007D447986007373FCA8C5F3590D",
        "seal_sha256": "28B4FD9366FABCABCDBC0A92F86AE9008548DE34FD4B7F380E3CD12D903D6C7E",
    },
    "c1_attribution": {
        "root": "artifacts/v2-a/closure-c1-failure-attribution-20260826-1",
        "result_sha256": "0726F48FA388466239869C73A3265097B4B74CA06CFB5D96CDD8EB9EECD4AADA",
        "seal_sha256": "D632311B7607E3C51F668CBB8B20E59A7CC55C3F01BE9B610BF482451C2A2941",
    },
}

FORBIDDEN_FORWARD_FIELDS = (
    "family",
    "route",
    "ast",
    "answer",
    "label",
    "teacher",
    "trace",
    "valid_choice",
    "reasoning_budget",
    "source_pointer",
    "target_pointer",
    "query_pointer",
)

FORMAL_PROHIBITIONS = (
    "old_checkpoint",
    "old_optimizer_state",
    "old_trace_probe_or_teacher_hidden",
    "dense_c1_or_c1r_continuation",
    "post_hoc_split_or_threshold",
    "best_checkpoint_selection",
    "retry_or_tuning_after_failure",
    "validation_test_ood_leakage",
    "answer_only_mechanism_claim",
    "raw_correct_logit_drop_as_causal_evidence",
    "external_hash_slot_permutation_as_anti_shortcut_evidence",
    "automatic_c2_v2a_v2b_v2c_authorization",
)

PREDECESSOR_AUTHORIZATION: dict[str, dict[str, Any]] = {
    "S1": {
        "required_predecessor": S0_IDENTITY,
        "required_predecessor_status": "PASS_V2_A_C1S_S0_QUALIFICATION",
        "required_predecessor_authorizes": S0_AUTHORIZED_SCOPE,
        "user_authorization_required": True,
        "user_authorization_scope": "continue S1, S2, and single-seed formal",
        "does_not_override_predecessor_gate": True,
    },
    "S2": {
        "required_predecessor": S1_IDENTITY,
        "required_predecessor_status": "PASS_V2_A_C1S_S1_QUALIFICATION",
        "required_predecessor_authorizes": "S2_DISCOVERY_ONLY",
        "user_authorization_required": True,
        "does_not_override_predecessor_gate": True,
    },
    "S3": {
        "required_predecessor": S2_IDENTITY,
        "required_predecessor_status": "PASS_V2_A_C1S_S2_QUALIFICATION",
        "required_predecessor_authorizes": "S3_SINGLE_SEED_FORMAL_ELIGIBILITY",
        "user_authorization_required": True,
        "does_not_override_predecessor_gate": True,
    },
}

NEVER_AUTHORIZES = (
    "C2",
    "V2-A PASS",
    "V2-B",
    "V2-C",
    "fresh-seed-replication",
    "Pareto",
    "integrated-architecture",
)


def contract_manifest() -> dict[str, Any]:
    """Return a JSON-native frozen manifest for the successor chain."""

    return {
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "identity": IDENTITY,
        "formal": False,
        "qualification": True,
        "design_doc": DESIGN_DOC.as_posix(),
        "s0_predecessor": {
            "identity": S0_IDENTITY,
            "authorized_scope": S0_AUTHORIZED_SCOPE,
            "result_and_seal_must_be_pinned_at_preflight": True,
        },
        "model": deepcopy(MODEL_CONFIG),
        "matched_k1_model": deepcopy(K1_CONFIG),
        "optimization": deepcopy(OPTIMIZATION_CONFIG),
        "mechanism_thresholds": deepcopy(MECHANISM_THRESHOLDS),
        "target_contract": deepcopy(TARGET_CONTRACT),
        "stages": {
            "S1": {
                **deepcopy(S1_CONFIG),
                "root": S1_ROOT.as_posix(),
                "lease": S1_LEASE.as_posix(),
                "preflight_root": S1_PREFLIGHT_ROOT.as_posix(),
                "preflight_lease": S1_PREFLIGHT_LEASE.as_posix(),
            },
            "S2": {
                **deepcopy(S2_CONFIG),
                "root": S2_ROOT.as_posix(),
                "lease": S2_LEASE.as_posix(),
                "preflight_root": S2_PREFLIGHT_ROOT.as_posix(),
                "preflight_lease": S2_PREFLIGHT_LEASE.as_posix(),
            },
            "S3": {
                **deepcopy(S3_CONFIG),
                "root": S3_ROOT.as_posix(),
                "lease": S3_LEASE.as_posix(),
                "preflight_root": S3_PREFLIGHT_ROOT.as_posix(),
                "preflight_lease": S3_PREFLIGHT_LEASE.as_posix(),
            },
        },
        "pinned_predecessor_roots": deepcopy(PINNED_PREDECESSOR_ROOTS),
        "predecessor_authorization": deepcopy(PREDECESSOR_AUTHORIZATION),
        "forbidden_forward_fields": list(FORBIDDEN_FORWARD_FIELDS),
        "formal_prohibitions": list(FORMAL_PROHIBITIONS),
        "never_authorizes": list(NEVER_AUTHORIZES),
        "authorization_boundary": {
            "explicit_user_authorization_recorded": True,
            "user_authorization_does_not_waive_fail_stop": True,
            "all_stages_single_use": True,
            "s3_formal_is_one_launch_only": True,
        },
        "preflight_accounting": {
            "s1_disposable_benchmark_optimizer_steps": 100,
            "s2_s3_disposable_benchmark_optimizer_steps": 0,
            "formal_stage_optimizer_steps": 0,
            "formal_stage_training_started": False,
        },
        "integrity_revalidation": {
            "all_stage_launches_and_s2_s3_preflight": [
                "sealed_s1_target_bank_gzip_sha256",
                "sealed_s1_split_ledger_sha256",
                "c1_cache_result_and_seal_sha256",
                "c1_cache_full_seal_tree_replay",
                "c1_cache_to_c0r_source_identity",
            ],
            "failure_is_pre_training_refusal_or_sealed_preflight_failure": True,
        },
    }


def stage_manifest(stage: str, *, preflight: bool = False) -> dict[str, Any]:
    """Return the chain contract bound to one concrete stage artifact."""

    stage = str(stage).upper()
    manifest = contract_manifest()
    if stage not in manifest["stages"]:
        raise ValueError(f"unknown C1S successor stage: {stage}")
    selected = deepcopy(manifest["stages"][stage])
    manifest.update(
        {
            "manifest_scope": "stage_preflight" if preflight else "stage",
            "active_stage": stage,
            "preflight": bool(preflight),
            # A formal stage has a non-formal preflight.  The S3 stage root
            # itself must carry formal=true rather than inheriting the chain
            # contract's qualification-only top-level marker.
            "formal": bool(selected.get("formal", False) and not preflight),
            "stage_contract": selected,
        }
    )
    return manifest


__all__ = [
    "DESIGN_DOC",
    "FORMAL_PROHIBITIONS",
    "FORBIDDEN_FORWARD_FIELDS",
    "IDENTITY",
    "K1_CONFIG",
    "MODEL_CONFIG",
    "MECHANISM_THRESHOLDS",
    "NEVER_AUTHORIZES",
    "OPTIMIZATION_CONFIG",
    "PINNED_PREDECESSOR_ROOTS",
    "PREDECESSOR_AUTHORIZATION",
    "S0_AUTHORIZED_SCOPE",
    "S0_IDENTITY",
    "S1_CONFIG",
    "S1_IDENTITY",
    "S1_LEASE",
    "S1_PREFLIGHT_LEASE",
    "S1_PREFLIGHT_ROOT",
    "S1_ROOT",
    "S2_CONFIG",
    "S2_IDENTITY",
    "S2_LEASE",
    "S2_PREFLIGHT_LEASE",
    "S2_PREFLIGHT_ROOT",
    "S2_ROOT",
    "S3_BEHAVIOR_THRESHOLDS",
    "S3_CONFIG",
    "S3_IDENTITY",
    "S3_LEASE",
    "S3_PREFLIGHT_LEASE",
    "S3_PREFLIGHT_ROOT",
    "S3_ROOT",
    "SCHEMA_PREFIX",
    "TARGET_CONTRACT",
    "contract_manifest",
    "stage_manifest",
]
