from __future__ import annotations

"""Frozen C1R staged-credit identities, schedule, and authorization boundary."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..closure_c1 import contract as c1_contract
from ..closure_c1_diagnosis import contract as attribution_contract


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1r"
C1R_IDENTITY = "V2-A-CLOSURE-C1R-STAGED-CREDIT-20260827-1"

OUTPUT_ROOT = Path("artifacts/v2-a/closure-c1r-staged-credit-20260827-1")
LEASE_PATH = Path(
    "artifacts/v2-a/closure-c1r-staged-credit-20260827-1.preflight-lease.jsonl"
)
PREFLIGHT_ROOT = Path("tmp/v2-a-closure-c1r-staged-credit-preflight-20260827-1")
PREFLIGHT_LEASE_PATH = Path(
    "tmp/v2-a-closure-c1r-staged-credit-preflight-20260827-1.preflight-lease.jsonl"
)
DESIGN_DOC = Path("docs/v2-a-closure-c1r-staged-credit.md")

# These are read-only prerequisites from the consumed C1 formal.  C1R never
# mutates or continues that root; the values are included to make provenance
# explicit in a future preflight/manifest.
FORMAL_ROOT = Path("artifacts/v2-a/closure-c1-single-seed-20260825-1")
FORMAL_IDENTITY = "V2-A-CLOSURE-C1-SINGLE-SEED-20260825-1"
FORMAL_RESULT_SHA256 = (
    "FE9F22B89E89092EDD0DCE1CC4FCBA926A3B8D4AB79123F237F4B9EDF379972C"
)
FORMAL_SEAL_SHA256 = (
    "BEB46576A91ADC08B9E7A1D51249E4CA5F03BFFFC7069D1E791FBBAC8749FA19"
)
FORMAL_SELECTED_CHECKPOINT_SHA256 = (
    "3FEC2FF80CC65FAEEB5E033F77D686D35E42B113730BC555F0FF1E47B8FEF46B"
)
ATTRIBUTION_ROOT = Path("artifacts/v2-a/closure-c1-failure-attribution-20260826-1")
ATTRIBUTION_RESULT_SHA256 = "0726F48FA388466239869C73A3265097B4B74CA06CFB5D96CDD8EB9EECD4AADA"
ATTRIBUTION_SEAL_SHA256 = "D632311B7607E3C51F668CBB8B20E59A7CC55C3F01BE9B610BF482451C2A2941"
CACHE_ROOT = attribution_contract.CACHE_ROOT
CACHE_RESULT_SHA256 = c1_contract.CACHE_RESULT_SHA256
CACHE_SEAL_SHA256 = c1_contract.CACHE_SEAL_SHA256
TRACE_BANK_SHA256 = (
    "C3EFBC11E2C1DE2ED4BA29829771B23A3B9B709C9C538E89BB29416FF531EC0B"
)

MODEL_SEED = c1_contract.MODEL_SEED
ORDER_SEED = c1_contract.ORDER_SEED
BOOTSTRAP_SEED = c1_contract.BOOTSTRAP_SEED
PROBE_SEED = 2026082701
OVERFIT_SEED = c1_contract.OVERFIT_SEED
MODEL_CONFIG = deepcopy(c1_contract.MODEL_CONFIG)

STAGE_A_CONFIG: dict[str, Any] = {
    "objective": "answer_only",
    "trace_probe": False,
    "shared_initialization": True,
    "batch_size": 8,
    "family_batch_size": 4,
    "epochs": 6,
    "model_seed": MODEL_SEED,
    "order_seed": ORDER_SEED,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "maximum_updates": 6144,
    "endpoint": "fixed_6144",
    "boundary_lr": 1.0e-4,
    "core_lr": 2.0e-4,
    "head_lr": 3.0e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "warmup_updates": 256,
    "max_prefetch": 2,
    "architecture": deepcopy(MODEL_CONFIG),
}

STAGE_B_CONFIG: dict[str, Any] = {
    "objective": "trace_probe_only",
    "deployed_graph": "exact_freeze",
    "probe_initialization": "fresh",
    "probe_seed": PROBE_SEED,
    "overfit_seed": OVERFIT_SEED,
    "order_seed": ORDER_SEED,
    "epochs": 6,
    "updates_per_epoch": 256,
    "maximum_updates": 1536,
    "batch_size": 32,
    "family_batch_size": 16,
    "block_tokens": 65,
    "block_overlap": 0,
    "loss_normalization": "valid_target_token_mean",
    "exposure_per_token": 6,
    "trace_lr": 3.0e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "warmup_updates": 256,
    "prefetch_workers": 0,
    "max_prefetch": 2,
}

POSITIVE_CONTROL_CONFIG: dict[str, Any] = {
    "records": 32,
    "records_per_family": 16,
    "epochs": 100,
    "batch_size": 8,
    "family_batch_size": 4,
    "maximum_updates": 400,
    "warmup_updates": 100,
    "trace_block_width": 65,
    "trace_token_accuracy": float(c1_contract.OVERFIT_CONFIG["trace_token_accuracy"]),
    "trace_content_accuracy": float(c1_contract.OVERFIT_CONFIG["trace_content_accuracy"]),
    "owner_nll_margin": float(c1_contract.OVERFIT_CONFIG["owner_nll_margin"]),
}

REQUIRED_CUDA_DEVICE_NAME_FRAGMENT = "RTX 4070"

DATASET_COUNTS = {"ERE/train": 4096, "CPS/train": 4096}
FAMILIES = ("ERE", "CPS")
NEVER_AUTHORIZES = ("C2", "V2-A PASS", "V2-B", "V2-C")
AUTHORIZATION_ON_PASS = "two additional fresh C1R seeds only"

THRESHOLDS = deepcopy(c1_contract.THRESHOLDS)


def c1r_contract_manifest() -> dict[str, Any]:
    """Return a JSON-native manifest for the new, independent C1R contract."""

    return {
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "identity": C1R_IDENTITY,
        "formal_input": {
            "identity": FORMAL_IDENTITY,
            "root": FORMAL_ROOT.as_posix(),
            "result_sha256": FORMAL_RESULT_SHA256,
            "seal_sha256": FORMAL_SEAL_SHA256,
            "selected_checkpoint_sha256": FORMAL_SELECTED_CHECKPOINT_SHA256,
            "cache_result_sha256": CACHE_RESULT_SHA256,
            "cache_seal_sha256": CACHE_SEAL_SHA256,
            "trace_bank_sha256": TRACE_BANK_SHA256,
        },
        "attribution_input": {
            "root": ATTRIBUTION_ROOT.as_posix(),
            "result_sha256": ATTRIBUTION_RESULT_SHA256,
            "seal_sha256": ATTRIBUTION_SEAL_SHA256,
        },
        "cache_root": CACHE_ROOT.as_posix(),
        "model": deepcopy(MODEL_CONFIG),
        "seeds": {
            "model": MODEL_SEED,
            "order": ORDER_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "probe": PROBE_SEED,
            "overfit": OVERFIT_SEED,
        },
        "stage_a": deepcopy(STAGE_A_CONFIG),
        "stage_b": deepcopy(STAGE_B_CONFIG),
        "positive_control": deepcopy(POSITIVE_CONTROL_CONFIG),
        "required_cuda_device_name_fragment": REQUIRED_CUDA_DEVICE_NAME_FRAGMENT,
        "output_root": OUTPUT_ROOT.as_posix(),
        "lease_path": LEASE_PATH.as_posix(),
        "preflight_root": PREFLIGHT_ROOT.as_posix(),
        "preflight_lease_path": PREFLIGHT_LEASE_PATH.as_posix(),
        "preflight_required_before_lease": True,
        "authorization_on_pass": AUTHORIZATION_ON_PASS,
        "never_authorizes": list(NEVER_AUTHORIZES),
    }


__all__ = [
    "AUTHORIZATION_ON_PASS",
    "ATTRIBUTION_ROOT",
    "ATTRIBUTION_RESULT_SHA256",
    "ATTRIBUTION_SEAL_SHA256",
    "CACHE_ROOT",
    "C1R_IDENTITY",
    "DATASET_COUNTS",
    "DESIGN_DOC",
    "FAMILIES",
    "FORMAL_IDENTITY",
    "FORMAL_ROOT",
    "FORMAL_RESULT_SHA256",
    "FORMAL_SEAL_SHA256",
    "FORMAL_SELECTED_CHECKPOINT_SHA256",
    "LEASE_PATH",
    "MODEL_CONFIG",
    "MODEL_SEED",
    "OVERFIT_SEED",
    "NEVER_AUTHORIZES",
    "ORDER_SEED",
    "OUTPUT_ROOT",
    "PREFLIGHT_LEASE_PATH",
    "PREFLIGHT_ROOT",
    "POSITIVE_CONTROL_CONFIG",
    "PROBE_SEED",
    "REQUIRED_CUDA_DEVICE_NAME_FRAGMENT",
    "SCHEMA_PREFIX",
    "STAGE_A_CONFIG",
    "STAGE_B_CONFIG",
    "THRESHOLDS",
    "c1r_contract_manifest",
]
