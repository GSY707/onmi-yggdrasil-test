from __future__ import annotations

"""Frozen C1T CPW implementation, S0 predecessor, and S1 contract."""

from pathlib import Path


IDENTITY = "V2-A-CLOSURE-C1T-CAUSALLY-PARTITIONED-WORKSPACE-20260901-1"
SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1t.cpw"
DESIGN_DOC = Path("docs/v2-a-closure-c1t-causally-partitioned-workspace.md")
S0_EXECUTION_DOC = Path("docs/v2-a-closure-c1t-s0-execution.md")
S1_EXECUTION_DOC = Path("docs/v2-a-closure-c1t-s1-execution.md")

S0_PREFLIGHT_IDENTITY = "V2-A-CLOSURE-C1T-CPW-S0-PREFLIGHT-20260901-1"
S0_IDENTITY = "V2-A-CLOSURE-C1T-CPW-S0-20260901-1"
S1_PREFLIGHT_IDENTITY = (
    "V2-A-CLOSURE-C1T-CPW-S1-OVERFIT32-PREFLIGHT-20260901-1"
)
S1_IDENTITY = "V2-A-CLOSURE-C1T-CPW-S1-OVERFIT32-20260901-1"

S0_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1t-cpw-s0-preflight-20260901-1")
S0_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1t-cpw-s0-preflight-20260901-1.preflight-lease.jsonl"
)
S0_ROOT = Path("artifacts/v2-a/closure-c1t-cpw-s0-20260901-1")
S0_LEASE = Path("artifacts/v2-a/closure-c1t-cpw-s0-20260901-1.s0-lease.jsonl")
S1_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1t-cpw-s1-overfit32-preflight-20260901-1")
S1_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1t-cpw-s1-overfit32-preflight-20260901-1.preflight-lease.jsonl"
)
S1_ROOT = Path("artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1")
S1_LEASE = Path("artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1.s1-lease.jsonl")

S0_RESULT_SHA256 = "831BC5C2B7DE284F99CAB7E27F4A3BF3DB33FB71B4BE928B1D9A62B22A41D6A8"
S0_EVIDENCE_SEAL_SHA256 = "E4D9B3DC63D2409BD947500517B9382B42EAB8D04DB5827D0421A29A6F5FA0EE"
S0_SOURCE_IDENTITY = "709EA9B92C363BC1C90B1D336A4032BF4266D6E27D0204768C1AF83556228566"
S0_CARD_CACHE_SHA256 = "44232C7DBA87421E33DA4BD6E54F8A2792A5B720C45920FD4D573837E67A4723"
S0_CARD_LEDGER_SHA256 = "287CB720A186481DCD455C962A950370F5DAE24587065C434557B369AABFF55B"

# These are the scientific components qualified by S0 and therefore may not
# drift while the S1 execution layer is added.  S1-specific contracts,
# evaluators, runners, tests, and documentation intentionally remain outside
# this pin set because S0 authorized their implementation.
S0_SCIENTIFIC_CORE_PINS = (
    "src/yggdrasil_v2/r1_revalidation/common/simulator.py",
    "src/yggdrasil_v2/v2_a/closure_c1/cache.py",
    "src/yggdrasil_v2/v2_a/closure_c1/contract.py",
    "src/yggdrasil_v2/v2_a/closure_c1t/cache.py",
    "src/yggdrasil_v2/v2_a/closure_c1t/cards.py",
    "src/yggdrasil_v2/v2_a/closure_c1t/model.py",
    "src/yggdrasil_v2/v2_a/closure_c1t/objective.py",
    "src/yggdrasil_v2/v2_a/closure_c1t/runtime.py",
    "src/yggdrasil_v2/v2_a/closure_c1t/source.py",
    "src/yggdrasil_v2/v2_a/closure_c1t/tasks.py",
)

TASK_SEED = 2026090101
MODEL_SEED = 2026090102
BOOTSTRAP_SEED = 2026090103
BOOTSTRAP_REPLICATES = 2_000
S1_ORDER_SEED = 2026090111
S1_PREFLIGHT_MODEL_SEED = 2026090191
S1_PREFLIGHT_ORDER_SEED = 2026090192
ADDRESS_SALT = "yggdrasil-c1t-public-address-v1"

SOURCE_MODEL_ID = "Qwen/Qwen3.5-2B"
SOURCE_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SOURCE_MODEL_CLASS = "Qwen3_5TextModel"
SOURCE_TOKENIZER_CLASS = "Qwen2Tokenizer"
SOURCE_HIDDEN_WIDTH = 2_048
SOURCE_DTYPE = "float16"
SOURCE_MAX_CARD_TOKENS = 256
TRANSFORMERS_VERSION = "5.13.1"
TORCH_VERSION = "2.13.0+cu130"
PYTHON_VERSION = "3.11.9"
CUDA_VERSION = "13.0"
SOURCE_ASSETS = {
    "config.json": {
        "sha256": "ED1C1723241F23F7F4E23430759CBD7DCFB4103CBDFE052BFE7626B57C2615B4",
        "bytes": 2_908,
    },
    "tokenizer.json": {
        "sha256": "5F9E4D4901A92B997E463C1F46055088B6CCA5CA61A6522D1B9F64C4BB81CB42",
        "bytes": 12_807_982,
    },
    "tokenizer_config.json": {
        "sha256": "49E2B6E395F959F077F1E992B338919C0D4A9732FC6E613995E06557F843500C",
        "bytes": 16_709,
    },
    "model.safetensors.index.json": {
        "sha256": "ACA8AFED9DA75B0F050B408D270766FD77627F1AF401E240F61C3B47D0DB02F9",
        "bytes": 64_460,
    },
    "model.safetensors-00001-of-00001.safetensors": {
        "sha256": "AA33250C4FC64891DDFABA3A314FD9542EA371843C387178B425FBCC5ED680B1",
        "bytes": 4_548_221_488,
    },
}

DEVICE_NAME_FRAGMENT = "RTX 4070"
DEVICE_COUNT = 1
MINIMUM_COMPUTE_CAPABILITY = (8, 0)
MINIMUM_FREE_CUDA_BYTES = 5_500_000_000
S0_PREFLIGHT_GROUPS = (("CPS", 0), ("ERE", 0))
S0_PREFLIGHT_RECORDS = len(S0_PREFLIGHT_GROUPS) * 4
S0_PREFLIGHT_CARDS = S0_PREFLIGHT_RECORDS * 6
S0_RECORDS = 32
S0_CARDS = S0_RECORDS * 6
S0_AUTHORIZED_SCOPE = "C1T_S0_ZERO_TRAINING_ONLY"
S0_PASS_AUTHORIZATION = "C1T_S1_CONTRACT_DESIGN_ONLY"
S1_USER_AUTHORIZATION = "开始S1阶段"
S1_AUTHORIZED_SCOPE = "C1T_S1_OVERFIT32_SINGLE_USE_ONLY"
S1_PASS_AUTHORIZATION = "C1T_S2_CONTRACT_DESIGN_ONLY"
S0_GATES = (
    "S001_source_and_device_identity",
    "S002_task_bank_and_factorial_causality",
    "S003_independent_card_cache",
    "S004_source_only_forward_boundary",
    "S005_equivariance_isolation_and_target_only_write",
    "S006_no_core_factorial_invariance",
    "S007_bf16_forward_backward_finite",
    "S008_accounting_source_and_seal_integrity",
)

FAMILIES = ("CPS", "ERE")
FACTORIAL_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
S1_GROUPS_PER_FAMILY = 4
S1_RECORDS_PER_FAMILY = S1_GROUPS_PER_FAMILY * len(FACTORIAL_CELLS)
S1_RECORDS = S1_RECORDS_PER_FAMILY * len(FAMILIES)
S1_GROUPS_PER_BATCH = 2
S1_BATCH_SIZE = S1_GROUPS_PER_BATCH * len(FACTORIAL_CELLS)
S1_MAXIMUM_UPDATES = 4_000
S1_FIXED_ENDPOINT = "fixed_4000"
S1_PREFLIGHT_BENCHMARK_STEPS = 32
S1_PREFLIGHT_MAX_STEP_P95_SECONDS = 3.0
S1_PREFLIGHT_MAX_ESTIMATED_FORMAL_SECONDS = 4 * 60 * 60
S1_PREFLIGHT_MAX_PEAK_CUDA_BYTES = 7_000_000_000

S1_TRAINING = {
    "batch_size": S1_BATCH_SIZE,
    "groups_per_batch": S1_GROUPS_PER_BATCH,
    "maximum_updates": S1_MAXIMUM_UPDATES,
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

MAX_SLOTS = 8
MAX_OPERATIONS = 10
ANSWER_CLASSES = 9
ADDRESS_WIDTH = 64

FORWARD_FIELDS = (
    "object_hidden",
    "object_mask",
    "object_addresses",
    "object_present",
    "operation_hidden",
    "operation_mask",
    "operation_source_addresses",
    "operation_target_addresses",
    "query_hidden",
    "query_mask",
    "query_address",
)
FORBIDDEN_FORWARD_FIELDS = (
    "answer",
    "answers",
    "ast",
    "counterfactual_indices",
    "factor_cells",
    "factorial_group_ids",
    "family",
    "label_mapping",
    "semantic_answer",
    "support_slots",
    "task_causal_arity",
    "valid_choice_mask",
)

LOSS_WEIGHTS = {
    "full_answer_ce": 1.0,
    "no_core_confusion": 0.5,
    "support_margin_hinge": 0.5,
    "support_margin_floor": 0.5,
}

S1_GATES = {
    "answer_exact": 1.0,
    "factorial_group_exact": 1.0,
    "no_core_accuracy_ceiling": {"CPS": 0.55, "ERE": 0.55},
    "no_core_group_logit_tolerance": 1.0e-6,
    "no_core_margin_drop_lower": 0.50,
    "two_contributor_point_floor": 0.80,
    "two_contributor_wilson_lower_floor": 0.65,
    "support_margin_drop_floor": 0.50,
    "support_flip_answer_exact": 1.0,
    "permutation_logit_tolerance": 1.0e-5,
}

S1_PREFLIGHT_GATES = (
    "P101_s0_predecessor_and_user_authorization",
    "P102_source_tests_paths_and_device",
    "P103_task_cache_schedule_and_forward_integrity",
    "P104_disposable_bf16_training_finite_and_within_budget",
    "P105_accounting_and_source_stability",
)

S1_RESULT_GATES = (
    "R101_predecessor_source_cache_and_schedule_identity",
    "R102_fixed_endpoint_and_accounting",
    "R103_full_answer_and_factorial_exact",
    "R104_no_core_necessity",
    "R105_two_support_counterfactual_causality",
    "R106_permutation_forward_and_parameter_integrity",
    "R107_result_source_and_cache_integrity",
)

NEVER_AUTHORIZES = (
    "C1T S1 training",
    "C1T S2",
    "C1T formal",
    "C1S S2",
    "C1S S3",
    "old single-seed formal",
    "checkpoint reuse",
    "C2",
    "V2-A PASS",
    "V2-B",
    "V2-C",
)

S1_NEVER_AUTHORIZES = (
    "C1T S2 training",
    "C1T formal",
    "C1S S2",
    "C1S S3",
    "old single-seed formal",
    "checkpoint reuse",
    "C2",
    "V2-A PASS",
    "V2-B",
    "V2-C",
)


def manifest() -> dict[str, object]:
    return {
        "identity": IDENTITY,
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "architecture": "causally_partitioned_workspace",
        "direct_cutover": True,
        "implementation_only": False,
        "training_authorized": True,
        "training_authorization_is_stage_limited": True,
        "user_authorization": S1_USER_AUTHORIZATION,
        "zero_training_s0_authorized_only_after_preflight": True,
        "s0_preflight_identity": S0_PREFLIGHT_IDENTITY,
        "s0_identity": S0_IDENTITY,
        "s0_preflight_root": S0_PREFLIGHT_ROOT.as_posix(),
        "s0_root": S0_ROOT.as_posix(),
        "source": {
            "model_id": SOURCE_MODEL_ID,
            "revision": SOURCE_MODEL_REVISION,
            "model_class": SOURCE_MODEL_CLASS,
            "tokenizer_class": SOURCE_TOKENIZER_CLASS,
            "hidden_width": SOURCE_HIDDEN_WIDTH,
            "dtype": SOURCE_DTYPE,
            "maximum_card_tokens": SOURCE_MAX_CARD_TOKENS,
            "one_card_per_forward_call": True,
            "whole_record_hidden_reuse": False,
            "assets": SOURCE_ASSETS,
        },
        "runtime": {
            "python": PYTHON_VERSION,
            "torch": TORCH_VERSION,
            "cuda": CUDA_VERSION,
            "transformers": TRANSFORMERS_VERSION,
            "device_count": DEVICE_COUNT,
            "device_name_fragment": DEVICE_NAME_FRAGMENT,
            "minimum_compute_capability": list(MINIMUM_COMPUTE_CAPABILITY),
            "bf16_required": True,
            "minimum_free_cuda_bytes": MINIMUM_FREE_CUDA_BYTES,
        },
        "s0_preflight_records": S0_PREFLIGHT_RECORDS,
        "s0_preflight_cards": S0_PREFLIGHT_CARDS,
        "s0_records": S0_RECORDS,
        "s0_cards": S0_CARDS,
        "s0_gates": list(S0_GATES),
        "s0_authorized_scope": S0_AUTHORIZED_SCOPE,
        "s0_pass_authorization": S0_PASS_AUTHORIZATION,
        "optimizer_steps": 0,
        "model_writes": 0,
        "families": list(FAMILIES),
        "factorial_cells": [list(cell) for cell in FACTORIAL_CELLS],
        "s1_records": S1_RECORDS,
        "s1_preflight_identity": S1_PREFLIGHT_IDENTITY,
        "s1_identity": S1_IDENTITY,
        "s1_preflight_root": S1_PREFLIGHT_ROOT.as_posix(),
        "s1_root": S1_ROOT.as_posix(),
        "s1_authorized_scope": S1_AUTHORIZED_SCOPE,
        "s1_pass_authorization": S1_PASS_AUTHORIZATION,
        "s1_maximum_updates": S1_MAXIMUM_UPDATES,
        "s1_fixed_endpoint": S1_FIXED_ENDPOINT,
        "s1_batch_unit": "complete_factorial_group",
        "s1_groups_per_batch": S1_GROUPS_PER_BATCH,
        "s1_batch_size": S1_BATCH_SIZE,
        "forward_fields": list(FORWARD_FIELDS),
        "forbidden_forward_fields": list(FORBIDDEN_FORWARD_FIELDS),
        "loss_weights": dict(LOSS_WEIGHTS),
        "s1_training": dict(S1_TRAINING),
        "s1_gates": dict(S1_GATES),
        "s1_preflight_gates": list(S1_PREFLIGHT_GATES),
        "s1_result_gates": list(S1_RESULT_GATES),
        "s0_predecessor": {
            "result_sha256": S0_RESULT_SHA256,
            "evidence_seal_sha256": S0_EVIDENCE_SEAL_SHA256,
            "source_identity": S0_SOURCE_IDENTITY,
            "card_cache_sha256": S0_CARD_CACHE_SHA256,
            "card_ledger_sha256": S0_CARD_LEDGER_SHA256,
            "scientific_core_pins": list(S0_SCIENTIFIC_CORE_PINS),
        },
        "never_authorizes": list(NEVER_AUTHORIZES),
        "s1_never_authorizes": list(S1_NEVER_AUTHORIZES),
    }


__all__ = [
    "ADDRESS_SALT",
    "ADDRESS_WIDTH",
    "ANSWER_CLASSES",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_REPLICATES",
    "CUDA_VERSION",
    "DESIGN_DOC",
    "DEVICE_COUNT",
    "DEVICE_NAME_FRAGMENT",
    "FACTORIAL_CELLS",
    "FAMILIES",
    "FORBIDDEN_FORWARD_FIELDS",
    "FORWARD_FIELDS",
    "IDENTITY",
    "LOSS_WEIGHTS",
    "MAX_OPERATIONS",
    "MAX_SLOTS",
    "MODEL_SEED",
    "MINIMUM_COMPUTE_CAPABILITY",
    "MINIMUM_FREE_CUDA_BYTES",
    "NEVER_AUTHORIZES",
    "PYTHON_VERSION",
    "S0_LEASE",
    "S0_AUTHORIZED_SCOPE",
    "S0_CARDS",
    "S0_EXECUTION_DOC",
    "S0_GATES",
    "S0_IDENTITY",
    "S0_PASS_AUTHORIZATION",
    "S0_PREFLIGHT_CARDS",
    "S0_PREFLIGHT_GROUPS",
    "S0_PREFLIGHT_IDENTITY",
    "S0_PREFLIGHT_LEASE",
    "S0_PREFLIGHT_RECORDS",
    "S0_PREFLIGHT_ROOT",
    "S0_RECORDS",
    "S0_ROOT",
    "S0_RESULT_SHA256",
    "S0_EVIDENCE_SEAL_SHA256",
    "S0_SOURCE_IDENTITY",
    "S0_CARD_CACHE_SHA256",
    "S0_CARD_LEDGER_SHA256",
    "S0_SCIENTIFIC_CORE_PINS",
    "S1_AUTHORIZED_SCOPE",
    "S1_EXECUTION_DOC",
    "S1_FIXED_ENDPOINT",
    "S1_GATES",
    "S1_BATCH_SIZE",
    "S1_GROUPS_PER_BATCH",
    "S1_GROUPS_PER_FAMILY",
    "S1_IDENTITY",
    "S1_LEASE",
    "S1_MAXIMUM_UPDATES",
    "S1_NEVER_AUTHORIZES",
    "S1_ORDER_SEED",
    "S1_PASS_AUTHORIZATION",
    "S1_PREFLIGHT_BENCHMARK_STEPS",
    "S1_PREFLIGHT_GATES",
    "S1_PREFLIGHT_IDENTITY",
    "S1_PREFLIGHT_LEASE",
    "S1_PREFLIGHT_MAX_ESTIMATED_FORMAL_SECONDS",
    "S1_PREFLIGHT_MAX_PEAK_CUDA_BYTES",
    "S1_PREFLIGHT_MAX_STEP_P95_SECONDS",
    "S1_PREFLIGHT_MODEL_SEED",
    "S1_PREFLIGHT_ORDER_SEED",
    "S1_PREFLIGHT_ROOT",
    "S1_RECORDS",
    "S1_RECORDS_PER_FAMILY",
    "S1_ROOT",
    "S1_RESULT_GATES",
    "S1_TRAINING",
    "S1_USER_AUTHORIZATION",
    "SCHEMA_PREFIX",
    "SOURCE_ASSETS",
    "SOURCE_DTYPE",
    "SOURCE_HIDDEN_WIDTH",
    "SOURCE_MAX_CARD_TOKENS",
    "SOURCE_MODEL_CLASS",
    "SOURCE_MODEL_ID",
    "SOURCE_MODEL_REVISION",
    "SOURCE_TOKENIZER_CLASS",
    "TASK_SEED",
    "TORCH_VERSION",
    "TRANSFORMERS_VERSION",
    "manifest",
]
