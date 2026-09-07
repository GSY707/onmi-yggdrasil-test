from __future__ import annotations

"""C1U Publicly-Grounded Gate-Free Workspace staged contract."""

from pathlib import Path


IDENTITY = "V2-A-CLOSURE-C1U-PUBLICLY-GROUNDED-GATE-FREE-WORKSPACE-20260901-1"
SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1u.pgf"
DESIGN_DOC = Path("docs/v2-a-closure-c1u-publicly-grounded-gate-free-workspace.md")
S0_EXECUTION_DOC = Path("docs/v2-a-closure-c1u-s0-execution.md")
S0_RESULT_REVIEW_DOC = Path("docs/v2-a-closure-c1u-s0-result-review.md")
S1_EXECUTION_DOC = Path("docs/v2-a-closure-c1u-s1-execution.md")

S0_PREFLIGHT_IDENTITY = "V2-A-CLOSURE-C1U-PGF-S0-PREFLIGHT-20260901-1"
S0_IDENTITY = "V2-A-CLOSURE-C1U-PGF-S0-20260901-1"
S1_PREFLIGHT_IDENTITY = (
    "V2-A-CLOSURE-C1U-PGF-S1-OVERFIT32-PREFLIGHT-20260901-1"
)
S1_IDENTITY = "V2-A-CLOSURE-C1U-PGF-S1-OVERFIT32-20260901-1"

S0_PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1u-pgf-s0-preflight-20260901-1")
S0_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1u-pgf-s0-preflight-20260901-1.preflight-lease.jsonl"
)
S0_ROOT = Path("artifacts/v2-a/closure-c1u-pgf-s0-20260901-1")
S0_LEASE = Path("artifacts/v2-a/closure-c1u-pgf-s0-20260901-1.s0-lease.jsonl")
S1_PREFLIGHT_ROOT = Path(
    "tmp/v2-a-closure-c1u-pgf-s1-overfit32-preflight-20260901-1"
)
S1_PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1u-pgf-s1-overfit32-preflight-20260901-1.preflight-lease.jsonl"
)
S1_ROOT = Path("artifacts/v2-a/closure-c1u-pgf-s1-overfit32-20260901-1")
S1_LEASE = Path(
    "artifacts/v2-a/closure-c1u-pgf-s1-overfit32-20260901-1.s1-lease.jsonl"
)

C1T_S0_ROOT = Path("artifacts/v2-a/closure-c1t-cpw-s0-20260901-1")
C1T_S1_ROOT = Path("artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1")
C1T_S1_LEASE = Path(
    "artifacts/v2-a/closure-c1t-cpw-s1-overfit32-20260901-1.s1-lease.jsonl"
)
C1T_S1_IDENTITY = "V2-A-CLOSURE-C1T-CPW-S1-OVERFIT32-20260901-1"
C1T_S1_RESULT_SHA256 = "9B37620F3F1CADE4740955F7849EACEA19C83DF208ADC3EA8F76AA5B74BC1DFF"
C1T_S1_EVIDENCE_SEAL_SHA256 = "89C530396FE9DD478EFFF941776722EA391B5D13FB2CD10C0D5D5B008FA99BCF"
C1T_S1_ENDPOINT_SHA256 = "E7E66F00BF4AE29CB7229D4E012630530F49DFB0E2EDA7302E783D9ED31896F9"
C1T_S1_SOURCE_IDENTITY = "07E8707EF30A3B78D93C9FD0C1AD0572D6F947BC8945DEB5A6A477085E65900E"
C1T_S1_TERMINAL_STATUS = "FAIL_V2_A_C1T_S1_QUALIFICATION"
C1T_S1_SEAL_SCHEMA = "yggdrasil.v2-a.closure-c1t.cpw.evidence-seal.v1"

# Frozen from the only sealed C1U S0 PASS.  S1 replays these values and the
# S0 evidence tree rather than relying on the now-extended current source tree.
S0_RESULT_SHA256 = "39070BCD12AA15B5EBA6F55E84A55CE348A69992BBD5087BB6DCE526CABA714A"
S0_EVIDENCE_SEAL_SHA256 = "F224DB2101897442742FD621BAA838A78D42ACAD5B6DAE68597E7B968D8A9ADE"
S0_SOURCE_IDENTITY = "2068B1E54F18C82AC3984A1B01C483E4A3B2297C191861DD962DB925A883572E"
S0_CARD_CACHE_SHA256 = "7CC6E33B73DFFF057C77489676B0A3AFB904507254FC55328242F79B6F82CAAD"
S0_CARD_LEDGER_SHA256 = "74DFB049B56E4D013C150732E11BF11EB65988078ECF6D7DFAB728541A784929"

# These are the scientific components qualified by S0 and therefore may not
# drift while the S1 execution layer is added.  S1-specific contracts,
# evaluators, runners, tests, and documentation intentionally remain outside
# this pin set because S0 authorized their implementation.
S0_SCIENTIFIC_CORE_PINS = (
    "src/yggdrasil_v2/r1_revalidation/common/simulator.py",
    "src/yggdrasil_v2/v2_a/closure_c1/cache.py",
    "src/yggdrasil_v2/v2_a/closure_c1/contract.py",
    "src/yggdrasil_v2/v2_a/closure_c1u/cache.py",
    "src/yggdrasil_v2/v2_a/closure_c1u/cards.py",
    "src/yggdrasil_v2/v2_a/closure_c1u/model.py",
    "src/yggdrasil_v2/v2_a/closure_c1u/objective.py",
    "src/yggdrasil_v2/v2_a/closure_c1u/runtime.py",
    "src/yggdrasil_v2/v2_a/closure_c1u/source.py",
    "src/yggdrasil_v2/v2_a/closure_c1u/tasks.py",
)

TASK_SEED = 2026090121
MODEL_SEED = 2026090122
BOOTSTRAP_SEED = 2026090123
BOOTSTRAP_REPLICATES = 2_000
S1_ORDER_SEED = 2026090131
S1_PREFLIGHT_MODEL_SEED = 2026090193
S1_PREFLIGHT_ORDER_SEED = 2026090194
ADDRESS_SALT = "yggdrasil-c1u-public-address-v1"

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
S0_AUTHORIZED_SCOPE = "C1U_S0_ZERO_TRAINING_ONLY"
S0_PASS_AUTHORIZATION = "C1U_S1_CONTRACT_AND_IMPLEMENTATION_ONLY"
USER_AUTHORIZATION = "没问题，按这个顺序做"
S1_USER_AUTHORIZATION = USER_AUTHORIZATION
S1_AUTHORIZED_SCOPE = "C1U_S1_OVERFIT32_SINGLE_USE_ONLY"
S1_PASS_AUTHORIZATION = "C1U_S2_MATCHED_K1_K8_CONTRACT_DESIGN_ONLY"
S0_GATES = (
    "U001_predecessor_source_device_identity",
    "U002_task_factorial_and_public_identifiability",
    "U003_bridge_fault_registry",
    "U004_independent_card_cache",
    "U005_source_only_forward_boundary",
    "U006_gate_free_target_only_transition",
    "U007_equivariance_and_no_core_topology",
    "U008_operation2_gradient_connectivity",
    "U009_bf16_forward_backward_finite",
    "U010_accounting_source_and_seal_integrity",
)
S0_PREFLIGHT_GATES = (
    "P001_predecessor_paths_and_user_authorization",
    "P002_source_closure_assets_and_device",
    "P003_public_identifiability_and_fault_registry",
    "P004_real_independent_cards",
    "P005_gate_free_topology_and_gradient_connectivity",
    "P006_bf16_accounting_and_source_stability",
)

FAMILIES = ("CPS", "ERE")
FACTORIAL_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
BANK_GROUPS_PER_FAMILY = 4
BANK_RECORDS_PER_FAMILY = BANK_GROUPS_PER_FAMILY * len(FACTORIAL_CELLS)
BANK_RECORDS = BANK_RECORDS_PER_FAMILY * len(FAMILIES)
S1_GROUPS_PER_FAMILY = BANK_GROUPS_PER_FAMILY
S1_RECORDS_PER_FAMILY = BANK_RECORDS_PER_FAMILY
S1_RECORDS = BANK_RECORDS
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

STRUCTURAL_TOLERANCES = {
    "no_core_group_logit_tolerance": 1.0e-6,
    "permutation_logit_tolerance": 1.0e-5,
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
    "R106_permutation_gate_free_and_parameter_integrity",
    "R107_result_source_and_cache_integrity",
)

NEVER_AUTHORIZES = (
    "C1U S2 training",
    "C1U formal",
    "C1S S2",
    "C1S S3",
    "old single-seed formal",
    "checkpoint reuse",
    "C2",
    "V2-A PASS",
    "V2-B",
    "V2-C",
)

S1_NEVER_AUTHORIZES = NEVER_AUTHORIZES

def manifest() -> dict[str, object]:
    return {
        "identity": IDENTITY,
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "architecture": "publicly_grounded_gate_free_workspace",
        "direct_cutover": True,
        "implementation_only": False,
        "training_authorized": True,
        "conditional_future_training_authorization": True,
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
        "s0_preflight_gates": list(S0_PREFLIGHT_GATES),
        "s0_authorized_scope": S0_AUTHORIZED_SCOPE,
        "s0_pass_authorization": S0_PASS_AUTHORIZATION,
        "optimizer_steps": 0,
        "model_writes": 0,
        "families": list(FAMILIES),
        "public_grounding": {
            "ere_value_legend": True,
            "cps_candidate_index": True,
            "public_only_reference_replay": True,
            "bridge_fault_registry": True,
        },
        "transition": {
            "target_only": True,
            "shared_across_steps_and_families": True,
            "active_from_operation_mask_only": True,
            "learned_write_gate": False,
            "operator_output_width": "payload_width",
        },
        "factorial_cells": [list(cell) for cell in FACTORIAL_CELLS],
        "bank_records": BANK_RECORDS,
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
        "s1_training": dict(S1_TRAINING),
        "s1_gates": dict(S1_GATES),
        "s1_preflight_gates": list(S1_PREFLIGHT_GATES),
        "s1_result_gates": list(S1_RESULT_GATES),
        "forward_fields": list(FORWARD_FIELDS),
        "forbidden_forward_fields": list(FORBIDDEN_FORWARD_FIELDS),
        "loss_weights": dict(LOSS_WEIGHTS),
        "structural_tolerances": dict(STRUCTURAL_TOLERANCES),
        "s1_execution": {
            "status": "FROZEN_READY_UNCONSUMED",
            "execution_document": S1_EXECUTION_DOC.as_posix(),
            "preflight_identity": S1_PREFLIGHT_IDENTITY,
            "identity": S1_IDENTITY,
            "preflight_root": S1_PREFLIGHT_ROOT.as_posix(),
            "root": S1_ROOT.as_posix(),
            "fixed_endpoint": S1_FIXED_ENDPOINT,
            "maximum_updates": S1_MAXIMUM_UPDATES,
            "command_exposed": True,
        },
        "c1t_failed_predecessor": {
            "s0_root": C1T_S0_ROOT.as_posix(),
            "root": C1T_S1_ROOT.as_posix(),
            "lease": C1T_S1_LEASE.as_posix(),
            "identity": C1T_S1_IDENTITY,
            "terminal_status": C1T_S1_TERMINAL_STATUS,
            "result_sha256": C1T_S1_RESULT_SHA256,
            "evidence_seal_sha256": C1T_S1_EVIDENCE_SEAL_SHA256,
            "endpoint_sha256": C1T_S1_ENDPOINT_SHA256,
            "source_identity": C1T_S1_SOURCE_IDENTITY,
            "authorizes": "nothing",
        },
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
    "BANK_GROUPS_PER_FAMILY",
    "BANK_RECORDS",
    "BANK_RECORDS_PER_FAMILY",
    "C1T_S1_ENDPOINT_SHA256",
    "C1T_S1_EVIDENCE_SEAL_SHA256",
    "C1T_S1_RESULT_SHA256",
    "C1T_S1_IDENTITY",
    "C1T_S1_LEASE",
    "C1T_S1_ROOT",
    "C1T_S1_SEAL_SCHEMA",
    "C1T_S1_SOURCE_IDENTITY",
    "C1T_S1_TERMINAL_STATUS",
    "C1T_S0_ROOT",
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
    "S0_RESULT_REVIEW_DOC",
    "S0_GATES",
    "S0_IDENTITY",
    "S0_PASS_AUTHORIZATION",
    "S0_PREFLIGHT_CARDS",
    "S0_PREFLIGHT_GROUPS",
    "S0_PREFLIGHT_GATES",
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
    "S1_BATCH_SIZE",
    "S1_EXECUTION_DOC",
    "S1_FIXED_ENDPOINT",
    "S1_GATES",
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
    "S1_RESULT_GATES",
    "S1_ROOT",
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
    "STRUCTURAL_TOLERANCES",
    "USER_AUTHORIZATION",
    "manifest",
]
