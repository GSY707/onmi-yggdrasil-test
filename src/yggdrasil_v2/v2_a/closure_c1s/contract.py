from __future__ import annotations

"""Frozen C1S Addressed Content Workspace qualification contract."""

from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1s"
IDENTITY = "V2-A-CLOSURE-C1S-ADDRESSED-WORKSPACE-20260828-1"

DESIGN_DOC = Path("docs/v2-a-closure-c1s-addressed-workspace-qualification.md")
S0_ROOT = Path("tmp/v2-a-closure-c1s-s0-preflight-20260828-1")
S0_LEASE = Path("tmp/v2-a-closure-c1s-s0-preflight-20260828-1.preflight-lease.jsonl")

MODEL_SEED = 2026082801
S1_MODEL_SEED = 2026082811
S1_ORDER_SEED = 2026082812
S2_MODEL_SEED = 2026082821
S2_ORDER_SEED = 2026082822

REQUIRED_CUDA_DEVICE_NAME_FRAGMENT = "RTX 4070"

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

S0_THRESHOLDS: dict[str, float] = {
    "permutation_logit_max_abs": 1.0e-5,
    "permutation_trajectory_max_abs": 1.0e-5,
    "unselected_slot_max_delta": 1.0e-6,
    "inactive_transition_max_delta": 1.0e-6,
    "selected_slot_min_delta": 1.0e-8,
    "query_swap_min_logit_l2": 1.0,
    "duplicate_query_swap_max_logit_l2": 1.0e-6,
    "positive_relevant_mean_replace_min": 1.0,
    "positive_irrelevant_mean_replace_max": 1.0e-6,
    "uniform_null_ownership_contrast_max": 1.01,
    "strip_logit_max_abs": 1.0e-6,
}

S1_CONFIG: dict[str, Any] = {
    "records": 32,
    "records_per_family": 16,
    "selection": 'sha256("C1S-S1-OVERFIT32-V1|example_id")',
    "batch_size": 8,
    "family_batch_size": 4,
    "maximum_updates": 4_000,
    "endpoint": "fixed",
    "checkpoint_selection": False,
    "model_seed": S1_MODEL_SEED,
    "order_seed": S1_ORDER_SEED,
    "answer_exact": 1.0,
    "owner_accuracy": 0.95,
    "query_owner_accuracy": 0.95,
    "state_closure_accuracy": 0.95,
    "query_swap_follow": 0.90,
    "same_value_different_owner": 0.90,
    "causal_drop": 0.50,
    "relevant_logit_drop": 0.50,
    "irrelevant_logit_drop_max": 0.10,
}

S2_CONFIG: dict[str, Any] = {
    "discovery_train_per_family": 3_072,
    "discovery_eval_per_family": 512,
    "exclude_s1_records": True,
    "batch_size": 8,
    "epochs": 6,
    "maximum_updates": 4_608,
    "endpoint": "fixed",
    "checkpoint_selection": False,
    "model_seed": S2_MODEL_SEED,
    "order_seed": S2_ORDER_SEED,
    "answer_point_floor": 0.75,
    "answer_wilson_lower_floor": 0.70,
    "k8_minus_k1_gain": 0.05,
    "paired_bootstrap_lower": 0.0,
    "ownership_floor": 0.80,
    "state_closure_floor": 0.80,
}

PINNED_ROOTS: dict[str, dict[str, str]] = {
    "c1_formal": {
        "root": "artifacts/v2-a/closure-c1-single-seed-20260825-1",
        "result_sha256": "FE9F22B89E89092EDD0DCE1CC4FCBA926A3B8D4AB79123F237F4B9EDF379972C",
        "seal_sha256": "BEB46576A91ADC08B9E7A1D51249E4CA5F03BFFFC7069D1E791FBBAC8749FA19",
    },
    "c1_attribution": {
        "root": "artifacts/v2-a/closure-c1-failure-attribution-20260826-1",
        "result_sha256": "0726F48FA388466239869C73A3265097B4B74CA06CFB5D96CDD8EB9EECD4AADA",
        "seal_sha256": "D632311B7607E3C51F668CBB8B20E59A7CC55C3F01BE9B610BF482451C2A2941",
    },
    "c1r_formal": {
        "root": "artifacts/v2-a/closure-c1r-staged-credit-20260827-1",
        "result_sha256": "54F426D5825A00A3DE9C57F16657F9DE499C007D447986007373FCA8C5F3590D",
        "seal_sha256": "28B4FD9366FABCABCDBC0A92F86AE9008548DE34FD4B7F380E3CD12D903D6C7E",
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

AUTHORIZATION_ON_S0_PASS = "one independent C1S S1 Overfit32 implementation and single-use run only"
NEVER_AUTHORIZES = (
    "automatic S1 launch",
    "S2",
    "single-seed formal",
    "C2",
    "V2-A PASS",
    "V2-B",
    "V2-C",
)


def contract_manifest() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "identity": IDENTITY,
        "stage": "S0-zero-update-structure-and-measurement-qualification",
        "formal": False,
        "qualification": True,
        "model": deepcopy(MODEL_CONFIG),
        "matched_k1_model": deepcopy(K1_CONFIG),
        "s0_thresholds": deepcopy(S0_THRESHOLDS),
        "s1_conditional_contract": deepcopy(S1_CONFIG),
        "s2_conditional_contract": deepcopy(S2_CONFIG),
        "model_seed": MODEL_SEED,
        "required_cuda_device_name_fragment": REQUIRED_CUDA_DEVICE_NAME_FRAGMENT,
        "pinned_roots": deepcopy(PINNED_ROOTS),
        "forbidden_forward_fields": list(FORBIDDEN_FORWARD_FIELDS),
        "s0_root": S0_ROOT.as_posix(),
        "s0_lease": S0_LEASE.as_posix(),
        "authorization_on_s0_pass": AUTHORIZATION_ON_S0_PASS,
        "never_authorizes": list(NEVER_AUTHORIZES),
    }


__all__ = [
    "AUTHORIZATION_ON_S0_PASS",
    "DESIGN_DOC",
    "FORBIDDEN_FORWARD_FIELDS",
    "IDENTITY",
    "K1_CONFIG",
    "MODEL_CONFIG",
    "MODEL_SEED",
    "NEVER_AUTHORIZES",
    "PINNED_ROOTS",
    "REQUIRED_CUDA_DEVICE_NAME_FRAGMENT",
    "S0_LEASE",
    "S0_ROOT",
    "S0_THRESHOLDS",
    "S1_CONFIG",
    "S2_CONFIG",
    "SCHEMA_PREFIX",
    "contract_manifest",
]
