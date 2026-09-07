from __future__ import annotations

"""Frozen, JSON-native contract for V2-A Closure C0."""

from copy import deepcopy
from pathlib import Path
from typing import Any


IDENTITY = "V2-A-CLOSURE-C0-20260823-1"
SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c0"
OUTPUT_ROOT = Path("artifacts/v2-a/closure-c0-20260823-1")
LEASE_PATH = Path("artifacts/v2-a/closure-c0-20260823-1.preflight-lease.jsonl")
DEFAULT_ARCHIVE_ROOT = Path(
    "D:/归档/onmi-yggdrasil-test-artifacts-2026-08-01/artifacts/v2-a"
)

P0D = {
    "root": "artifacts/v2-r1r/p0d-v17-full-production-20260810-1",
    "seal_sha256": "453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D",
    "assessment_sha256": "A8702ADB7189EE33D119C8D1964F452A0D0601BCDA9ED4B20FFDF2C6C06ECF95",
    "manifest_sha256": "E2D2DE707C456F2FF7F3382B7EF8ED50DE57DE81C36497576A894EDF7490AD77",
    "model_id": "Qwen/Qwen3.5-2B",
    "model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}

P0M = {
    "root": "artifacts/v2-r1r/p0m-v5-assessment-20260810-1",
    "seal_sha256": "17F45EAC8A3DF129B236688D3D1E2B22BD9B634D4098F903142CA42E7FEEF09E",
    "result_sha256": "3640E883FAFB8B8F92DE4DBE75FDC78ED9FCB7B5451FACAF8197AE66A2AD1F44",
    "interpretation": "training-path smoke only; no architecture or OOD conclusion",
    "components": ["cache", "cps", "direct", "ere", "joint", "text_cot", "throughput"],
}

CURRENT_SOURCE_HASHES = {
    "src/yggdrasil_v2/r1_revalidation/production/generator.py": "27C3109E53B520D087EEF8BFF05EAA4C5FD54C81F5567F22A22386BB8A64DBC8",
    "src/yggdrasil_v2/r1_revalidation/production/renderer.py": "F00F2B86E9A54E29FFEEC8B5205CFE35D36E465822640781523E7E41F801A439",
    "src/yggdrasil_v2/r1_revalidation/production/schema.py": "8910D57325A87A62E5F61D1E2753CA71279AAB64A0409639D3F2B6BA168DAE36",
}

DATASET_PROFILE = {
    "ERE": {
        "train": 4096,
        "validation": 1536,
        "composition_ood": 1536,
        "entity_ood": 1536,
        "length_ood": 1536,
        "language_ood": 1536,
        "causal_pairs": 1536,
    },
    "CPS": {
        "train": 4096,
        "validation": 1536,
        "composition_ood": 1536,
        "distractor_ood": 1536,
        "horizon_ood": 1536,
        "language_ood": 1536,
        "causal_pairs": 1536,
    },
}

AST_KEYS = {
    "ERE": ["events", "initial_state", "query", "rules"],
    "CPS": ["actions", "budget", "candidates", "final_constraints", "goal", "initial_state"],
}

VISIBLE_PATTERN_MIN_SUPPORT = 128
VISIBLE_PATTERN_MIN_CLASS_SUPPORT = 64
VISIBLE_PATTERN_MAX_ANSWER_MASS = 0.80
VISIBLE_ORACLE_MAX_EXCESS_ACCURACY = 0.10

ARM_CONTRACT: dict[str, Any] = {
    "arms": ["direct", "text_cot", "latent_k1", "latent_k8"],
    "shared_inputs": {
        "model_id": P0D["model_id"],
        "model_revision": P0D["model_revision"],
        "same_tokenizer": True,
        "same_source_text": True,
        "same_answer_label": True,
        "same_splits_and_causal_pairs": True,
        "same_teacher_source": True,
        "family_visible_to_forward": False,
        "family_used_as_training_target": False,
        "canonical_public_forward_fields": ["source_text"],
        "valid_choice_mask_visible_to_forward": False,
        "reasoning_budget_visible_to_forward": False,
        "program_ast_trace_claims_visible_to_forward": False,
    },
    "teacher_symmetry": {
        "required": True,
        "rule": (
            "Any step/state supervision consumed by a latent arm must also be supplied in "
            "consumable form to direct and text-CoT; otherwise the result is supervision-advantaged."
        ),
        "primary_lane": {
            "name": "matched_compact_trace",
            "same_record_level_compact_trace_target": True,
            "same_trace_target_tokens_and_loss_mask": True,
            "same_answer_and_trace_exposure_ledger": True,
            "same_per_record_target_bytes_sha256": True,
            "direct_training_only_trace_decoder_stripped": True,
            "latent_training_only_trace_decoder_stripped": True,
            "dense_state_or_claim_targets_allowed": False,
        },
        "secondary_lane": {
            "dense_state_or_claim_targets_allowed": True,
            "classification": "supervision-advantaged sensitivity only",
            "eligible_for_medium_superiority_claim": False,
        },
    },
    "text_baseline_match": {
        "same_lora_targets_rank_alpha_dropout": True,
        "same_chat_template_thinking_truncation": True,
    },
    "latent_capacity_match": {
        "only_registered_difference": "slot_count: 1 versus 8",
        "same_boundary_transition_readout_data_order_optimizer_evaluation": True,
        "latent_k8_core": "one shared dense recurrent core",
    },
    "selection_and_budget": {
        "same_episode_and_order_ledger": True,
        "same_validation_frequency": True,
        "same_max_optimizer_and_search_budget": True,
        "heldout_arm_specific_tuning_forbidden": True,
        "single_seed_first": True,
        "decode_and_transition_caps_selected_on_train_or_validation_only": True,
        "required_matched_slices": ["equal_examples", "equal_gpu_hours"],
    },
    "cost_ledger": [
        "training_flops",
        "teacher_generation_tokens",
        "teacher_verification_tokens",
        "training_wall_seconds",
        "online_qwen_encode",
        "output_tokens",
        "latent_transitions",
        "peak_vram",
        "activation_and_kv_bytes",
        "throughput",
        "end_to_end_latency",
    ],
    "evaluation": [
        "validation",
        "every_ood_cell",
        "causal_pair_flip",
        "worst_family",
        "worst_seed",
        "answer_quality",
        "trace_or_latent_causal_necessity",
    ],
    "forbidden_active_components": [
        "route_id",
        "projection_expert",
        "ffn_moe",
        "task_or_family_branch",
        "write_delete_checkpoint",
        "h1_checkpoint",
    ],
    "latency_rule": "online end-to-end paths are primary; cached latency is diagnostic only",
    "trace_qualification": {
        "formatter_parser_roundtrip_all_records": True,
        "semantic_replay_against_source_simulator": True,
        "malformed_trace_rejection": True,
        "directed_fault_kill": True,
        "generated_trace_metric_frozen_before_training": True,
        "decode_max_new_tokens": 512,
    },
    "cache_policy": {
        "p0m_smoke_cache_reusable_for_c1": False,
        "qwen_encode_cost_included": True,
        "cache_bytes_and_build_wall_time_included": True,
        "fresh_seed_caches_built_and_archived_sequentially": True,
    },
}

H1_EXCLUSIONS = [
    {
        "name": "factorized_routed_projection",
        "path": "artifacts/v2-r1r/p1-h1-nonformal-factorized-routed-projection-screen-20260817-1/probe-result.json",
        "sha256": "12D609D63D3321F0189A759B41E88F4A84AB74956AB298F24DFCC81DA08B1544",
        "checks": {
            "status": "NONFORMAL_P1_H1_PROBE_ONLY",
            "direction_screen_gate.authorizes": "nothing",
        },
    },
    {
        "name": "direction_geometry_v2",
        "path": "artifacts/v2-r1r/h1-wd-direction-geometry-screen-v2-20260823-1/result.json",
        "sha256": "0E482421FEA8C251F42D1014941F81891E86E0C41A35EBF3407D073CB605F3A0",
        "checks": {
            "status": "COMPLETE_NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2",
            "scientific_status": "NO_QUALIFIED_R2_R3_COMPONENT",
            "authorizes": "nothing",
        },
    },
]

HISTORICAL_EVIDENCE = [
    {
        "stage": "A1.8",
        "path": "a1_8/assessment-summary.json",
        "sha256": "1CBEEB16C05471A00AF67FF51E840959A688B8659743A6585BCB443202CB0C3F",
        "checks": {"passed": True},
        "classification": "reusable_mechanism_prior",
        "reuse": "long-horizon recurrence, three-seed formal/causal protocol",
        "limit": "structured COPY/SWAP only; no learned full-text Boundary or Pareto",
    },
    {
        "stage": "A1.9",
        "path": "a1_9/assessment-summary.json",
        "sha256": "B85B1ECD8EFA55C5B5F01FEB52E147EDFF9D76D7481F2A1FD892E7CB6BD5A824",
        "checks": {"passed": True},
        "classification": "narrow_boundary_prior",
        "reuse": "Qwen cache integrity and hidden-state causal intervention method",
        "limit": "oracle character spans and typed roles; not a learned full-text Boundary",
    },
    {
        "stage": "A1.18B",
        "path": "a1_18b/assessment-summary.json",
        "sha256": "5E91D0E4C8E6153F5AC35AA23A4F5EA82D9A00E3DBC69D8F2321B5062B93799C",
        "checks": {"mechanism_solved": True, "complete_v2_a_validated": False},
        "classification": "reusable_training_mechanism",
        "reuse": "dense per-step state credit with physical auxiliary stripping",
        "limit": "oracle symbolic state targets; does not qualify ERE/CPS target construction",
    },
    {
        "stage": "A1.19H",
        "path": "a1_19h/assessment-summary.json",
        "sha256": "96B9C20765230AA270C6255C4830C683DD7047E260EAE877D58C53BAB7DB2815",
        "checks": {
            "classification.code": "generalized_hybrid_core_confirmed",
            "a119h_complete": True,
        },
        "classification": "strongest_core_prior",
        "reuse": "shared continuous recurrence, opaque addressing, causal-closure tests",
        "limit": "exact-symbolic COPY/SWAP; weights and task results cannot transfer to ERE/CPS",
    },
    {
        "stage": "A1.20B",
        "path": "a1_20b/assessment-summary.json",
        "sha256": "60CB8F197E3A32F1BD34905FAC37891645E12DB2928BB18FE0FB5950F5B6D805",
        "checks": {"formal_eligibility_passed": False, "a120b_passed": False},
        "classification": "negative_diagnostic",
        "reuse": "entity/value binding and pointer-validity failure localization",
        "limit": "a failed Boundary, not positive architecture qualification",
    },
    {
        "stage": "A1.20C",
        "path": "a1_20c/overfit32/hierarchical__straight_through/failure-diagnostic.json",
        "sha256": "6B043BC6EE2354CFC082F32124253B41AA7B3C630BE5A2792C28EF431BCAF282",
        "checks": {
            "machine_classification": "anchor_localization_solved_but_execution_objectives_conflict",
            "stage_gate_passed": False,
        },
        "classification": "negative_diagnostic",
        "reuse": "compiler versus execution-gradient conflict checks",
        "limit": "overfit-only failed arm; no formal or causal qualification",
    },
    {
        "stage": "A1.20D/A1.21P",
        "path": "a1_21p/assessment-summary.json",
        "sha256": "EACEECE5EC9BA7BEC0D6E02AC2097DC98928AEE89E6F62E57F0647B358651AF9",
        "checks": {
            "a120d_relaxed_mechanism_passed": True,
            "a121p_passed": False,
            "a122a_authorized": False,
        },
        "classification": "mechanism_control_and_pareto_failure",
        "reuse": "bidirectional section inference as a positive control; K=1 and cost-ledger method",
        "limit": "post-stop continuation, task isomorphism, unmatched baselines, tiny samples, no fresh three seeds",
    },
]

GATE_IDS = tuple(f"C00{index}" for index in range(1, 9))


def contract_manifest() -> dict[str, Any]:
    """Return a defensive JSON-native copy of the complete frozen contract."""

    return deepcopy(
        {
            "schema_version": f"{SCHEMA_PREFIX}.contract-manifest.v1",
            "identity": IDENTITY,
            "scope": "task-suite and matched-baseline readiness only",
            "p0d": P0D,
            "p0m": P0M,
            "dataset_profile": DATASET_PROFILE,
            "ast_keys": AST_KEYS,
            "visible_pattern_shortcut_gate": {
                "pattern": "ERE source-visible query kind=relation",
                "minimum_support": VISIBLE_PATTERN_MIN_SUPPORT,
                "minimum_each_class_support": VISIBLE_PATTERN_MIN_CLASS_SUPPORT,
                "maximum_dominant_semantic_answer_mass": VISIBLE_PATTERN_MAX_ANSWER_MASS,
                "source_only_oracle_max_excess_over_chance": VISIBLE_ORACLE_MAX_EXCESS_ACCURACY,
                "scope": "every split containing the registered pattern",
            },
            "arm_contract": ARM_CONTRACT,
            "h1_exclusions": H1_EXCLUSIONS,
            "historical_evidence": HISTORICAL_EVIDENCE,
            "gates": list(GATE_IDS),
            "single_use": {
                "output_root": OUTPUT_ROOT.as_posix(),
                "lease_path": LEASE_PATH.as_posix(),
            },
            "authorization_on_pass": "C1 single-seed implementation and eligibility only",
            "never_authorizes": ["training in C0", "C2", "C3", "V2-B", "V2-C"],
        }
    )


__all__ = [
    "ARM_CONTRACT",
    "AST_KEYS",
    "CURRENT_SOURCE_HASHES",
    "DATASET_PROFILE",
    "DEFAULT_ARCHIVE_ROOT",
    "GATE_IDS",
    "H1_EXCLUSIONS",
    "HISTORICAL_EVIDENCE",
    "IDENTITY",
    "LEASE_PATH",
    "OUTPUT_ROOT",
    "P0D",
    "P0M",
    "SCHEMA_PREFIX",
    "VISIBLE_PATTERN_MAX_ANSWER_MASS",
    "VISIBLE_PATTERN_MIN_CLASS_SUPPORT",
    "VISIBLE_PATTERN_MIN_SUPPORT",
    "VISIBLE_ORACLE_MAX_EXCESS_ACCURACY",
    "contract_manifest",
]
