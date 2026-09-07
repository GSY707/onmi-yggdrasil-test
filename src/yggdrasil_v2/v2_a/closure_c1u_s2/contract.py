from __future__ import annotations

"""Frozen contract for C1U S2 multi-bank matched K1/K8."""

from pathlib import Path

from yggdrasil_v2.v2_a.closure_c1u import contract as c1u_contract


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1u.pgf.s2-multibank-matched-k1-k8"
DESIGN_DOC = Path("docs/v2-a-closure-c1u-s2-multibank-matched-k1-k8-execution.md")
USER_AUTHORIZATION = "没问题，那只做多bank，多seed先不做。那接下来就应该继续做S2了，开始吧"

PREFLIGHT_IDENTITY = (
    "V2-A-CLOSURE-C1U-PGF-S2-MULTIBANK-MATCHED-K1-K8-PREFLIGHT-20260902-1"
)
IDENTITY = "V2-A-CLOSURE-C1U-PGF-S2-MULTIBANK-MATCHED-K1-K8-20260902-1"
PREFLIGHT_ROOT = Path(
    "tmp/v2-a-closure-c1u-pgf-s2-multibank-matched-k1-k8-preflight-20260902-1"
)
PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1u-pgf-s2-multibank-matched-k1-k8-preflight-20260902-1.preflight-lease.jsonl"
)
ROOT = Path(
    "artifacts/v2-a/closure-c1u-pgf-s2-multibank-matched-k1-k8-20260902-1"
)
LEASE = Path(
    "artifacts/v2-a/closure-c1u-pgf-s2-multibank-matched-k1-k8-20260902-1.s2-lease.jsonl"
)

S1_ROOT = c1u_contract.S1_ROOT
S1_LEASE = c1u_contract.S1_LEASE
S1_IDENTITY = c1u_contract.S1_IDENTITY
S1_STATUS = "PASS_V2_A_C1U_S1_OVERFIT32"
S1_AUTHORIZATION = c1u_contract.S1_PASS_AUTHORIZATION
S1_RESULT_SHA256 = "1FCD2DEF44F0324631FC3ECE4D1F552DC6D66181A2F6E97F6894A3FE5816090C"
S1_EVIDENCE_SEAL_SHA256 = "A5211D46340365E7B9DB4B3300C39034C89AA6C51942FDEF5A2911D152635350"
S1_ENDPOINT_SHA256 = "88F538F7B0E5AD0A0C77759EE2D72C8866D69F36F5D4E8F7AB951FFEC80F159F"
S1_SOURCE_IDENTITY = "8BD618F1960FFCCF1BBD6DEA50EA57051B48D16955AA3AF40EF194230D9194FD"

BANK_SEEDS = tuple(2_026_090_201 + index for index in range(6))
BANK_IDS = tuple(f"B{index}" for index in range(len(BANK_SEEDS)))
LABEL_ROTATIONS = tuple(range(1, len(BANK_SEEDS) + 1))
FOLDS = (
    {"fold_id": "F0", "train": ("B2", "B3", "B4", "B5"), "heldout": ("B0", "B1")},
    {"fold_id": "F1", "train": ("B0", "B1", "B4", "B5"), "heldout": ("B2", "B3")},
    {"fold_id": "F2", "train": ("B0", "B1", "B2", "B3"), "heldout": ("B4", "B5")},
)
BANK_RECORDS = c1u_contract.BANK_RECORDS
TOTAL_RECORDS = len(BANK_IDS) * BANK_RECORDS
TRAIN_RECORDS_PER_FOLD = 4 * BANK_RECORDS
HELDOUT_RECORDS_PER_FOLD = 2 * BANK_RECORDS
TRAIN_GROUPS_PER_FAMILY = 4 * c1u_contract.BANK_GROUPS_PER_FAMILY

ARMS = ("K1", "K8")
WORKSPACE_SLOTS = {"K1": 1, "K8": c1u_contract.MAX_SLOTS}
MODEL_SEED = 2_026_090_211
ORDER_SEED = 2_026_090_212
BOOTSTRAP_SEED = 2_026_090_213
BOOTSTRAP_REPLICATES = 2_000
PREFLIGHT_MODEL_SEED = 2_026_090_291
PREFLIGHT_ORDER_SEED = 2_026_090_292
PREFLIGHT_STEPS_PER_ARM = 16
MAXIMUM_UPDATES_PER_ENDPOINT = 4_000
FIXED_ENDPOINT = "fixed_4000"
ENDPOINT_COUNT = len(FOLDS) * len(ARMS)
TOTAL_FORMAL_OPTIMIZER_STEPS = ENDPOINT_COUNT * MAXIMUM_UPDATES_PER_ENDPOINT
MAX_CACHE_BATCHES = 16

TRAINING = {
    "batch_size": 8,
    "groups_per_batch": 2,
    "maximum_updates": MAXIMUM_UPDATES_PER_ENDPOINT,
    "boundary_lr": 1.0e-4,
    "transition_lr": 2.0e-4,
    "head_lr": 3.0e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "warmup_updates": 256,
    "log_interval": 100,
    "cpu_threads": 2,
    "pin_memory": True,
    "optimizer": "AdamW",
    "schedule": "linear_warmup_then_cosine",
    "dtype": "bfloat16_autocast",
    "checkpoint_selection": False,
    "intermediate_checkpoints": False,
}

GATES = {
    "answer_point_floor": 0.75,
    "answer_wilson_lower_floor": 0.70,
    "bank_family_answer_point_floor": 0.50,
    "factorial_exact_point_floor": 0.75,
    "paired_gain_point_floor": 0.05,
    "paired_gain_bootstrap_lower_floor": 0.0,
    "no_core_accuracy_ceiling": 0.55,
    "no_core_margin_drop_lower": 0.50,
    "support_margin_drop_lower": 0.50,
    "support_flip_point_floor": 0.80,
    "support_flip_wilson_lower_floor": 0.70,
    "two_contributor_point_floor": 0.80,
    "two_contributor_wilson_lower_floor": 0.70,
    "owner_deletion_margin_drop_lower": 0.50,
    "k8_owner_swap_logit_l2_floor": 1.0e-4,
    "k1_owner_swap_logit_max_abs": 1.0e-5,
    "permutation_logit_max_abs": 1.0e-5,
    "owner_only_majority_accuracy_max": 0.50,
}

PREFLIGHT_GATES = (
    "P201_s1_predecessor_authorization_and_paths",
    "P202_source_assets_device_and_registered_tests",
    "P203_multibank_disjointness_factorial_and_shortcut",
    "P204_target_free_independent_card_cache",
    "P205_k1_k8_parameter_initialization_and_structural_controls",
    "P206_disposable_bf16_benchmark_schedule_and_accounting",
    "P207_source_result_and_seal_integrity",
)
RESULT_GATES = (
    "R201_predecessor_cache_split_schedule_and_source_identity",
    "R202_fixed_endpoints_accounting_and_matched_parity",
    "R203_k8_fresh_bank_absolute_behavior",
    "R204_paired_k8_beats_k1",
    "R205_k8_heldout_causal_behavior",
    "R206_functional_k8_and_measured_k1_null",
    "R207_endpoint_source_predecessor_cache_and_seal_integrity",
)

PASS_AUTHORIZATION = "C1U_S3_CONTRACT_DESIGN_ONLY"
NEVER_AUTHORIZES = (
    "S2 retry",
    "alternate model seed",
    "multi-seed claim",
    "checkpoint selection",
    "S3 training",
    "single-seed formal",
    "C2",
    "V2-A PASS",
    "V2-B",
    "V2-C",
)


def manifest() -> dict[str, object]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "identity": IDENTITY,
        "preflight_identity": PREFLIGHT_IDENTITY,
        "design_doc": DESIGN_DOC.as_posix(),
        "user_authorization": USER_AUTHORIZATION,
        "formal": False,
        "qualification": True,
        "single_use": True,
        "retry_allowed": False,
        "multi_bank": True,
        "multi_seed": False,
        "scientific_model_seeds": [MODEL_SEED],
        "bootstrap_seed_is_not_a_training_seed": True,
        "roots": {
            "preflight": PREFLIGHT_ROOT.as_posix(),
            "preflight_lease": PREFLIGHT_LEASE.as_posix(),
            "s2": ROOT.as_posix(),
            "s2_lease": LEASE.as_posix(),
        },
        "predecessor": {
            "identity": S1_IDENTITY,
            "root": S1_ROOT.as_posix(),
            "lease": S1_LEASE.as_posix(),
            "status": S1_STATUS,
            "authorization": S1_AUTHORIZATION,
            "result_sha256": S1_RESULT_SHA256,
            "evidence_seal_sha256": S1_EVIDENCE_SEAL_SHA256,
            "endpoint_sha256": S1_ENDPOINT_SHA256,
            "source_identity": S1_SOURCE_IDENTITY,
        },
        "banks": [
            {
                "bank_id": bank_id,
                "task_seed": seed,
                "label_rotation": rotation,
                "records": BANK_RECORDS,
            }
            for bank_id, seed, rotation in zip(
                BANK_IDS, BANK_SEEDS, LABEL_ROTATIONS, strict=True
            )
        ],
        "folds": [
            {
                "fold_id": fold["fold_id"],
                "train": list(fold["train"]),
                "heldout": list(fold["heldout"]),
            }
            for fold in FOLDS
        ],
        "arms": {
            arm: {
                "workspace_slots": WORKSPACE_SLOTS[arm],
                "same_public_inputs": True,
                "same_trainable_parameterization": True,
            }
            for arm in ARMS
        },
        "model_seed": MODEL_SEED,
        "order_seed": ORDER_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "training": dict(TRAINING),
        "maximum_updates_per_endpoint": MAXIMUM_UPDATES_PER_ENDPOINT,
        "fixed_endpoint": FIXED_ENDPOINT,
        "endpoint_count": ENDPOINT_COUNT,
        "total_formal_optimizer_steps": TOTAL_FORMAL_OPTIMIZER_STEPS,
        "preflight_disposable_optimizer_steps": len(ARMS)
        * PREFLIGHT_STEPS_PER_ARM,
        "gates": dict(GATES),
        "preflight_gates": list(PREFLIGHT_GATES),
        "result_gates": list(RESULT_GATES),
        "pass_authorization": PASS_AUTHORIZATION,
        "never_authorizes": list(NEVER_AUTHORIZES),
    }


__all__ = [name for name in globals() if name.isupper()] + ["manifest"]
