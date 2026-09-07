from __future__ import annotations

"""Frozen identities and thresholds for the V2-A Closure C0R successor."""

from copy import deepcopy
from pathlib import Path
from typing import Any


DATA_IDENTITY = "V2-A-CLOSURE-C0R-DATA-TRACE-20260824-1"
READINESS_IDENTITY = "V2-A-CLOSURE-C0R-20260824-1"
SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c0r"

DATA_OUTPUT_ROOT = Path("artifacts/v2-a/closure-c0r-data-trace-20260824-1")
DATA_LEASE_PATH = Path("artifacts/v2-a/closure-c0r-data-trace-20260824-1.preflight-lease.jsonl")
PREFLIGHT_OUTPUT_ROOT = Path("tmp/v2-a-closure-c0r-data-trace-preflight-20260824-1")
PREFLIGHT_LEASE_PATH = Path("tmp/v2-a-closure-c0r-data-trace-preflight-20260824-1.preflight-lease.jsonl")
READINESS_OUTPUT_ROOT = Path("artifacts/v2-a/closure-c0r-20260824-1")
READINESS_LEASE_PATH = Path("artifacts/v2-a/closure-c0r-20260824-1.preflight-lease.jsonl")

OLD_C0_ROOT = Path("artifacts/v2-a/closure-c0-20260823-1")
OLD_C0_RESULT_SHA256 = "3FC871BB9EDE76421F08C386E2BB1702878B35273FF5831A1D541C7CB6CC43B1"
OLD_C0_SEAL_SHA256 = "462317E7BD97040AB901BFD79FA33B631898EA8BD9B75C4ECA8C83B191949972"
OLD_FAIRNESS_SHA256 = "545939B5E32DF9F1D415DD4A72DB4DE4A5395A933355541274C507E0B35E85BF"
P0D_V17_ROOT = Path("artifacts/v2-r1r/p0d-v17-full-production-20260810-1")
P0D_V17_SEAL_SHA256 = "453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D"

MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
FORMAL_SEED = 2026082401
DATASET_PROFILE = {
    "ERE": {
        "train": 4096,
        "validation": 1536,
        "composition_ood": 1536,
        "length_ood": 1536,
        "entity_ood": 1536,
        "language_ood": 1536,
        "causal_pairs": 1536,
    },
    "CPS": {
        "train": 4096,
        "validation": 1536,
        "composition_ood": 1536,
        "horizon_ood": 1536,
        "distractor_ood": 1536,
        "language_ood": 1536,
        "causal_pairs": 1536,
    },
}

VISIBLE_PATTERN_MIN_SUPPORT = 128
VISIBLE_PATTERN_MIN_CLASS_SUPPORT = 64
VISIBLE_PATTERN_MAX_ANSWER_MASS = 0.80
VISIBLE_ORACLE_MAX_EXCESS_ACCURACY = 0.10
TRACE_MAX_NEW_TOKENS = 512

DATA_GATE_IDS = tuple(f"D{index:03d}" for index in range(1, 13))
READINESS_GATE_IDS = tuple(f"C00{index}" for index in range(1, 9))

PINNED_SUBSTRATE_HASHES = {
    "src/yggdrasil_v2/r1_revalidation/production/generator.py": "27C3109E53B520D087EEF8BFF05EAA4C5FD54C81F5567F22A22386BB8A64DBC8",
    "src/yggdrasil_v2/r1_revalidation/production/generator_audit.py": "A87904B6AFBF4FB14A01E326011567E546B2D2A5AEE1A8310A9215A69A932AFA",
    "src/yggdrasil_v2/r1_revalidation/production/renderer.py": "F00F2B86E9A54E29FFEEC8B5205CFE35D36E465822640781523E7E41F801A439",
    "src/yggdrasil_v2/r1_revalidation/production/fingerprint.py": "E44CD89614632B862AD8E56DD132C3D8B71238A7542EFFAE7666644A18D57E89",
    "src/yggdrasil_v2/r1_revalidation/production/p0_shortcut.py": "8C81754E73D25256A20AA44036056E1C79EA258E90F061D22F7819D8B9A6F653",
    "src/yggdrasil_v2/r1_revalidation/production/g09_decision.py": "E9159E3F46EA2A3F75E2724B6BD5963F8DA80E4264E378D395C11C95264A598A",
    "src/yggdrasil_v2/r1_revalidation/common/simulator.py": "155A87092594C1D5DCCF0E9B9500CE392811359F225A1C502060C3397B595425",
}

# Filled only after the single-use data/trace qualification has sealed.  The
# readiness runner refuses while either value is unset, so the second formal
# cannot consume an unpinned first-stage artifact.
DATA_RESULT_SHA256: str | None = (
    "B2502F66CB5D9ECEB2CA0547CB280CC5346D1A24D2F8901F94E617E1B258FFF9"
)
DATA_SEAL_SHA256: str | None = (
    "5AFB37AB5950D04298FED4D2DAFA78102260A728101A5EDE4972D1EC0EEBB570"
)

FAIRNESS_REQUIREMENTS: dict[str, Any] = {
    "arms": ["direct", "text_cot", "latent_k1", "latent_k8"],
    "canonical_public_forward_fields": ["source_text"],
    "required_matched_slices": ["equal_examples", "equal_gpu_hours"],
    "decode_max_new_tokens": TRACE_MAX_NEW_TOKENS,
    "forbidden_active_components": [
        "route_id",
        "projection_expert",
        "ffn_moe",
        "task_or_family_branch",
        "write_delete_checkpoint",
        "h1_checkpoint",
    ],
}


def data_contract_manifest() -> dict[str, Any]:
    return deepcopy(
        {
            "schema_version": f"{SCHEMA_PREFIX}.data-contract.v1",
            "identity": DATA_IDENTITY,
            "formal_seed": FORMAL_SEED,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_profile": DATASET_PROFILE,
            "p0d_v17_prerequisite": {
                "root": P0D_V17_ROOT.as_posix(),
                "seal_sha256": P0D_V17_SEAL_SHA256,
                "interpretation": "immutable production substrate prerequisite only",
            },
            "pinned_substrate_hashes": PINNED_SUBSTRATE_HASHES,
            "relation_gate": {
                "minimum_support": VISIBLE_PATTERN_MIN_SUPPORT,
                "minimum_each_class_support": VISIBLE_PATTERN_MIN_CLASS_SUPPORT,
                "maximum_dominant_answer_mass": VISIBLE_PATTERN_MAX_ANSWER_MASS,
                "maximum_oracle_excess_over_chance": VISIBLE_ORACLE_MAX_EXCESS_ACCURACY,
            },
            "trace_gate": {
                "format": "CT1 <canonical-json>\\nAnswer: <A-I>",
                "maximum_new_tokens": TRACE_MAX_NEW_TOKENS,
                "full_bank_roundtrip": True,
                "fresh_simulator_replay": True,
                "malformed_rejection": True,
                "directed_fault_kill_per_record": True,
            },
            "gates": list(DATA_GATE_IDS),
            "single_use": {
                "output_root": DATA_OUTPUT_ROOT.as_posix(),
                "lease_path": DATA_LEASE_PATH.as_posix(),
            },
            "training_allowed": False,
            "authorization_on_pass": "C0R readiness audit only",
        }
    )


def readiness_contract_manifest() -> dict[str, Any]:
    return deepcopy(
        {
            "schema_version": f"{SCHEMA_PREFIX}.readiness-contract.v1",
            "identity": READINESS_IDENTITY,
            "old_c0": {
                "root": OLD_C0_ROOT.as_posix(),
                "result_sha256": OLD_C0_RESULT_SHA256,
                "seal_sha256": OLD_C0_SEAL_SHA256,
                "fairness_contract_sha256": OLD_FAIRNESS_SHA256,
            },
            "data_trace_input": {
                "root": DATA_OUTPUT_ROOT.as_posix(),
                "result_sha256": DATA_RESULT_SHA256,
                "seal_sha256": DATA_SEAL_SHA256,
            },
            "fairness_requirements": FAIRNESS_REQUIREMENTS,
            "gates": list(READINESS_GATE_IDS),
            "single_use": {
                "output_root": READINESS_OUTPUT_ROOT.as_posix(),
                "lease_path": READINESS_LEASE_PATH.as_posix(),
            },
            "authorization_on_pass": "C1 single-seed implementation and eligibility only",
            "never_authorizes": ["training in C0R", "C2", "C3", "V2-B", "V2-C"],
        }
    )


__all__ = [
    "DATA_GATE_IDS",
    "DATA_IDENTITY",
    "DATA_LEASE_PATH",
    "DATA_OUTPUT_ROOT",
    "DATA_RESULT_SHA256",
    "DATA_SEAL_SHA256",
    "DATASET_PROFILE",
    "FAIRNESS_REQUIREMENTS",
    "FORMAL_SEED",
    "MODEL_ID",
    "MODEL_REVISION",
    "OLD_C0_RESULT_SHA256",
    "OLD_C0_ROOT",
    "OLD_C0_SEAL_SHA256",
    "OLD_FAIRNESS_SHA256",
    "P0D_V17_ROOT",
    "P0D_V17_SEAL_SHA256",
    "PINNED_SUBSTRATE_HASHES",
    "PREFLIGHT_LEASE_PATH",
    "PREFLIGHT_OUTPUT_ROOT",
    "READINESS_GATE_IDS",
    "READINESS_IDENTITY",
    "READINESS_LEASE_PATH",
    "READINESS_OUTPUT_ROOT",
    "SCHEMA_PREFIX",
    "TRACE_MAX_NEW_TOKENS",
    "VISIBLE_ORACLE_MAX_EXCESS_ACCURACY",
    "VISIBLE_PATTERN_MAX_ANSWER_MASS",
    "VISIBLE_PATTERN_MIN_CLASS_SUPPORT",
    "VISIBLE_PATTERN_MIN_SUPPORT",
    "data_contract_manifest",
    "readiness_contract_manifest",
]
