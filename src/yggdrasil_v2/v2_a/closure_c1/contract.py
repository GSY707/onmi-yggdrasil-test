from __future__ import annotations

"""Frozen identities, hyperparameters, thresholds, and authorization for C1."""

from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1"
CACHE_IDENTITY = "V2-A-CLOSURE-C1-CACHE-20260825-1"
C1_IDENTITY = "V2-A-CLOSURE-C1-SINGLE-SEED-20260825-1"

CACHE_OUTPUT_ROOT = Path("artifacts/v2-a/closure-c1-cache-20260825-1")
CACHE_LEASE_PATH = Path(
    "artifacts/v2-a/closure-c1-cache-20260825-1.preflight-lease.jsonl"
)
CACHE_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1-cache-preflight-20260825-1")
CACHE_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1-cache-preflight-20260825-1.preflight-lease.jsonl"
)
C1_OUTPUT_ROOT = Path("artifacts/v2-a/closure-c1-single-seed-20260825-1")
C1_LEASE_PATH = Path(
    "artifacts/v2-a/closure-c1-single-seed-20260825-1.preflight-lease.jsonl"
)
C1_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1-single-seed-preflight-20260825-1")
C1_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1-single-seed-preflight-20260825-1.preflight-lease.jsonl"
)

DESIGN_DOC = Path("docs/v2-a-closure-c1-single-seed-eligibility.md")
C0R_DATA_ROOT = Path("artifacts/v2-a/closure-c0r-data-trace-20260824-1")
C0R_READINESS_ROOT = Path("artifacts/v2-a/closure-c0r-20260824-1")
C0R_DATA_RESULT_SHA256 = (
    "B2502F66CB5D9ECEB2CA0547CB280CC5346D1A24D2F8901F94E617E1B258FFF9"
)
C0R_DATA_SEAL_SHA256 = (
    "5AFB37AB5950D04298FED4D2DAFA78102260A728101A5EDE4972D1EC0EEBB570"
)
C0R_READINESS_RESULT_SHA256 = (
    "391C846D2150FD27D2016A79A9C360F3CB4E1455EC71240720F772CF58A28D5E"
)
C0R_READINESS_SEAL_SHA256 = (
    "161FBB367DEE40D18CDD23521E7DA9EBA3257EE8BF57C1FA194886E43D140BA4"
)

MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
TRANSFORMERS_VERSION = "5.13.1"
TOKENIZER_VOCAB_SIZE = 248_077
SOURCE_LEXICON_SIZE = 5_555
TRACE_VOCAB_SIZE = 5_597
CACHE_MAX_SOURCE_TOKENS = 1024
CACHE_INFERENCE_BATCH_SIZE = 8
CACHE_SHARD_SIZE = 64
TRACE_GRAMMAR_TOKEN_IDS = (
    58,
    92,
    487,
    1089,
    1123,
    1143,
    1288,
    1293,
    1666,
    1797,
    1802,
    2129,
    2456,
    2685,
    3147,
    4851,
    4891,
    5046,
    5702,
    7664,
    8631,
    8783,
    11534,
    14522,
    15050,
    15666,
    16352,
    17709,
    20691,
    22357,
    22642,
    25312,
    31928,
    32817,
    34764,
    40775,
    45404,
    46793,
    55558,
    80620,
    86451,
    93482,
)

CACHE_GATE_IDS = tuple(f"K{index:03d}" for index in range(1, 9))
C1_GATE_IDS = tuple(f"G{index:03d}" for index in range(1, 12))

# Frozen from the sealed single-use cache qualification.  The C1 launcher
# refuses if either artifact no longer matches these exact values.
CACHE_RESULT_SHA256: str | None = (
    "A058687125FDC1315871733DD1499036A040636864665793B8D32B8C9D758D4A"
)
CACHE_SEAL_SHA256: str | None = (
    "68CA34BB18F889405F875BF54B7CA2E55C3E9E549DE5DFE1C63C4F47E2B4DC34"
)

MODEL_SEED = 2026082501
ORDER_SEED = 2026082502
BOOTSTRAP_SEED = 2026082503
OVERFIT_SEED = 2026082599

MODEL_CONFIG: dict[str, Any] = {
    "source_width": 2048,
    "latent_width": 512,
    "slots": 8,
    "boundary_depth": 2,
    "core_depth": 2,
    "attention_heads": 8,
    "ffn_width": 2048,
    "recurrent_steps": 10,
    "answer_classes": 9,
    "trace_vocab_size": TRACE_VOCAB_SIZE,
    "max_global_positions": 1024,
    "max_local_positions": 512,
}

TRAINING_CONFIG: dict[str, Any] = {
    "batch_size": 8,
    "family_batch_size": 4,
    "epochs": 6,
    "maximum_updates": 6144,
    "minimum_selection_update": 2048,
    "evaluation_interval": 512,
    "trace_chunk_tokens": 64,
    "boundary_lr": 1.0e-4,
    "core_lr": 2.0e-4,
    "head_lr": 3.0e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "warmup_updates": 256,
    "early_answer_weight": 0.5,
    "early_trace_weight": 1.0,
    "late_answer_weight": 1.0,
    "late_trace_weight": 0.5,
    "loss_switch_update": 2048,
    "validation_trace_records_per_family": 256,
    "bootstrap_replicates": 10_000,
}

OVERFIT_CONFIG: dict[str, Any] = {
    "records": 32,
    "batch_size": 8,
    "maximum_updates": 4000,
    "evaluation_interval": 100,
    "answer_accuracy": 1.0,
    "trace_token_accuracy": 0.99,
    "trace_content_accuracy": 0.99,
    "owner_nll_margin": 0.50,
    "strip_logit_max_abs_diff": 1.0e-6,
}

THRESHOLDS: dict[str, Any] = {
    "validation": {"point": 0.75, "wilson_lower": 0.70},
    "ood": {"point": 0.65, "wilson_lower": 0.62},
    "causal_raw": {"point": 0.70, "wilson_lower": 0.66},
    "causal_pair": {"point": 0.60, "bootstrap_lower": 0.56},
    "trace": {
        "token_accuracy": 0.80,
        "content_accuracy": 0.60,
        "owner_margin": 0.15,
        "owner_margin_bootstrap_lower": 0.05,
    },
    "hidden": {
        "drop_bootstrap_lower": 0.20,
        "intervention_wilson_upper": 0.35,
    },
    "recurrence": {
        "h0_drop_bootstrap_lower": 0.15,
        "midpoint_shuffle_drop_bootstrap_lower": 0.05,
        "midpoint_step": 5,
    },
    "slot": {"prediction_invariance": 1.0, "logit_max_abs_diff": 1.0e-5},
    "strip": {"prediction_invariance": 1.0, "logit_max_abs_diff": 1.0e-6},
}

DATASET_COUNTS = {
    "CPS/causal_pairs": 1536,
    "CPS/composition_ood": 1536,
    "CPS/distractor_ood": 1536,
    "CPS/horizon_ood": 1536,
    "CPS/language_ood": 1536,
    "CPS/train": 4096,
    "CPS/validation": 1536,
    "ERE/causal_pairs": 1536,
    "ERE/composition_ood": 1536,
    "ERE/entity_ood": 1536,
    "ERE/language_ood": 1536,
    "ERE/length_ood": 1536,
    "ERE/train": 4096,
    "ERE/validation": 1536,
}

FORBIDDEN_FORWARD_FIELDS = (
    "family",
    "route",
    "program_ast",
    "teacher_trace",
    "training_claims",
    "reasoning_budget",
    "valid_choice_mask",
    "answer_index",
    "label_mapping",
)
FORBIDDEN_COMPONENT_TOKENS = (
    "route",
    "expert",
    "projection_expert",
    "ffn_moe",
    "family_embedding",
    "write_delete",
    "reasoning_budget",
    "choice_mask",
)


def cache_contract_manifest() -> dict[str, Any]:
    return deepcopy(
        {
            "schema_version": f"{SCHEMA_PREFIX}.cache-contract.v1",
            "identity": CACHE_IDENTITY,
            "prerequisites": {
                "c0r_data_root": C0R_DATA_ROOT.as_posix(),
                "c0r_data_result_sha256": C0R_DATA_RESULT_SHA256,
                "c0r_data_seal_sha256": C0R_DATA_SEAL_SHA256,
                "c0r_readiness_root": C0R_READINESS_ROOT.as_posix(),
                "c0r_readiness_result_sha256": C0R_READINESS_RESULT_SHA256,
                "c0r_readiness_seal_sha256": C0R_READINESS_SEAL_SHA256,
            },
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "dtype": "float16",
            "maximum_source_tokens": CACHE_MAX_SOURCE_TOKENS,
            "inference_batch_size": CACHE_INFERENCE_BATCH_SIZE,
            "shard_size": CACHE_SHARD_SIZE,
            "truncation_allowed": False,
            "trace_grammar_token_ids": list(TRACE_GRAMMAR_TOKEN_IDS),
            "expected_source_lexicon_size": SOURCE_LEXICON_SIZE,
            "expected_trace_vocab_size": TRACE_VOCAB_SIZE,
            "dataset_counts": DATASET_COUNTS,
            "gates": list(CACHE_GATE_IDS),
            "output_root": CACHE_OUTPUT_ROOT.as_posix(),
            "lease_path": CACHE_LEASE_PATH.as_posix(),
            "preflight_root": CACHE_PREFLIGHT_ROOT.as_posix(),
            "preflight_required_before_lease": True,
            "training_allowed": False,
            "authorization_on_pass": "C1 single-seed launcher only",
        }
    )


def c1_contract_manifest() -> dict[str, Any]:
    return deepcopy(
        {
            "schema_version": f"{SCHEMA_PREFIX}.single-seed-contract.v1",
            "identity": C1_IDENTITY,
            "cache_identity": CACHE_IDENTITY,
            "cache_result_sha256": CACHE_RESULT_SHA256,
            "cache_seal_sha256": CACHE_SEAL_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "seeds": {
                "model": MODEL_SEED,
                "order": ORDER_SEED,
                "bootstrap": BOOTSTRAP_SEED,
                "overfit": OVERFIT_SEED,
            },
            "model": MODEL_CONFIG,
            "training": TRAINING_CONFIG,
            "overfit": OVERFIT_CONFIG,
            "thresholds": THRESHOLDS,
            "forbidden_forward_fields": list(FORBIDDEN_FORWARD_FIELDS),
            "forbidden_component_tokens": list(FORBIDDEN_COMPONENT_TOKENS),
            "gates": list(C1_GATE_IDS),
            "output_root": C1_OUTPUT_ROOT.as_posix(),
            "lease_path": C1_LEASE_PATH.as_posix(),
            "preflight_root": C1_PREFLIGHT_ROOT.as_posix(),
            "preflight_required_before_lease": True,
            "authorization_on_pass": "two additional fresh C1 seeds only",
            "never_authorizes": ["C2", "C3", "V2-B", "V2-C", "V2-A PASS"],
        }
    )


__all__ = [
    "BOOTSTRAP_SEED",
    "C0R_DATA_ROOT",
    "C0R_READINESS_ROOT",
    "C1_GATE_IDS",
    "C1_IDENTITY",
    "C1_LEASE_PATH",
    "C1_OUTPUT_ROOT",
    "CACHE_GATE_IDS",
    "CACHE_IDENTITY",
    "CACHE_INFERENCE_BATCH_SIZE",
    "CACHE_LEASE_PATH",
    "CACHE_MAX_SOURCE_TOKENS",
    "CACHE_OUTPUT_ROOT",
    "CACHE_RESULT_SHA256",
    "CACHE_SEAL_SHA256",
    "CACHE_SHARD_SIZE",
    "DATASET_COUNTS",
    "FORBIDDEN_COMPONENT_TOKENS",
    "FORBIDDEN_FORWARD_FIELDS",
    "MODEL_CONFIG",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_SEED",
    "ORDER_SEED",
    "OVERFIT_CONFIG",
    "OVERFIT_SEED",
    "SCHEMA_PREFIX",
    "THRESHOLDS",
    "TRACE_GRAMMAR_TOKEN_IDS",
    "TRACE_VOCAB_SIZE",
    "TRANSFORMERS_VERSION",
    "TRAINING_CONFIG",
    "cache_contract_manifest",
    "c1_contract_manifest",
]
