from __future__ import annotations

"""Frozen identity, pins, measurements, and stop rules for C1 attribution."""

from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1.failure-attribution"
IDENTITY = "V2-A-CLOSURE-C1-FAILURE-ATTRIBUTION-20260826-1"
OUTPUT_ROOT = Path("artifacts/v2-a/closure-c1-failure-attribution-20260826-1")
LEASE_PATH = Path(
    "artifacts/v2-a/closure-c1-failure-attribution-20260826-1.preflight-lease.jsonl"
)
PREFLIGHT_ROOT = Path(
    "tmp/v2-a-closure-c1-failure-attribution-preflight-20260826-1"
)
PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1-failure-attribution-preflight-20260826-1.preflight-lease.jsonl"
)
DESIGN_DOC = Path("docs/v2-a-closure-c1-failure-attribution.md")

FORMAL_ROOT = Path("artifacts/v2-a/closure-c1-single-seed-20260825-1")
CACHE_ROOT = Path("artifacts/v2-a/closure-c1-cache-20260825-1")
C0R_DATA_ROOT = Path("artifacts/v2-a/closure-c0r-data-trace-20260824-1")

FORMAL_RESULT_SHA256 = "FE9F22B89E89092EDD0DCE1CC4FCBA926A3B8D4AB79123F237F4B9EDF379972C"
FORMAL_SEAL_SHA256 = "BEB46576A91ADC08B9E7A1D51249E4CA5F03BFFFC7069D1E791FBBAC8749FA19"
PRIMARY_TRAINING_SHA256 = "1B4711E1D2839A61E2619F1683954F6106B217B9DAF817A0686AB44886C44B3A"
TRACE_LEDGER_SHA256 = "0FDB026A005D237729CB663FC61F8C200713F4D672A41783A5AA88B656DA25CC"
TRACE_CREDIT_SHA256 = "E74EF5D27113C88352BC9E458B40E8F98507ED4A11B797A9A10DF60BB6CF6BFC"
G007_SHA256 = "4E6C6A98B27A3BD34A795BCA080CCBF9DF54120CC1DEA1D7C257A99547568FEE"
SELECTED_CHECKPOINT_SHA256 = "3FEC2FF80CC65FAEEB5E033F77D686D35E42B113730BC555F0FF1E47B8FEF46B"
FINAL_CHECKPOINT_SHA256 = "1AB9D8683D099BF64457C9B37DD1851B1BD956FB05867EBAE16A8D57404C6FC8"
CACHE_RESULT_SHA256 = "A058687125FDC1315871733DD1499036A040636864665793B8D32B8C9D758D4A"
CACHE_SEAL_SHA256 = "68CA34BB18F889405F875BF54B7CA2E55C3E9E549DE5DFE1C63C4F47E2B4DC34"
TRACE_BANK_SHA256 = "C3EFBC11E2C1DE2ED4BA29829771B23A3B9B709C9C538E89BB29416FF531EC0B"
FORMAL_SOURCE_IDENTITY = "F910F5F47EE066606BF2A7AB65F65B875B9E143EC7C3929D7D1BCD8FB992729C"
FORMAL_POSTSTOP_DOC_SHA256 = "7D104A80E0FCAB034BE1C6D3865E06181DFDF0FF56D201C2B9B1EDBE4C0A43F8"

SELECTED_UPDATE = 5_120
FINAL_UPDATE = 6_144
TRACE_RECORDS_PER_FAMILY = 256
TRAIN_EXPOSURE_RECORDS_PER_FAMILY = 256
ALIGNMENT_RECORDS_PER_FAMILY = 64
TRACE_TOKEN_CHUNK = 64
EVALUATION_BATCH_SIZE = 8
GRADIENT_BATCHES = 16
ALIGNMENT_WORKERS = 16
BOOTSTRAP_SEED = 2026082601
BOOTSTRAP_REPLICATES = 10_000

MEASUREMENT_IDS = tuple(f"A{index:03d}" for index in range(1, 7))
POSITION_BINS = (
    (0, 64, "000-063"),
    (64, 128, "064-127"),
    (128, 192, "128-191"),
    (192, 256, "192-255"),
    (256, 1024, "256-1023"),
)
ALIGNMENT_MODES = ("registered", "previous", "next", "constant_h1", "constant_h10")
THRESHOLDS: dict[str, Any] = {
    "alignment_nll_improvement": 0.05,
    "alignment_top1_improvement": 0.02,
    "selection_top1_improvement": 0.02,
    "exposure_accuracy_gap": 0.10,
    "gradient_late_trace_to_answer_ratio": 3.0,
    "gradient_negative_cosine_fraction": 0.50,
}


def manifest() -> dict[str, Any]:
    return deepcopy(
        {
            "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
            "identity": IDENTITY,
            "output_root": OUTPUT_ROOT.as_posix(),
            "lease_path": LEASE_PATH.as_posix(),
            "preflight_root": PREFLIGHT_ROOT.as_posix(),
            "formal_root": FORMAL_ROOT.as_posix(),
            "cache_root": CACHE_ROOT.as_posix(),
            "pins": {
                "formal_result": FORMAL_RESULT_SHA256,
                "formal_seal": FORMAL_SEAL_SHA256,
                "primary_training": PRIMARY_TRAINING_SHA256,
                "trace_ledger": TRACE_LEDGER_SHA256,
                "trace_credit": TRACE_CREDIT_SHA256,
                "g007": G007_SHA256,
                "selected_checkpoint": SELECTED_CHECKPOINT_SHA256,
                "final_checkpoint": FINAL_CHECKPOINT_SHA256,
                "cache_result": CACHE_RESULT_SHA256,
                "cache_seal": CACHE_SEAL_SHA256,
                "trace_bank": TRACE_BANK_SHA256,
                "formal_source_identity": FORMAL_SOURCE_IDENTITY,
                "formal_poststop_doc": FORMAL_POSTSTOP_DOC_SHA256,
            },
            "measurements": list(MEASUREMENT_IDS),
            "trace_records_per_family": TRACE_RECORDS_PER_FAMILY,
            "train_exposure_records_per_family": TRAIN_EXPOSURE_RECORDS_PER_FAMILY,
            "alignment_records_per_family": ALIGNMENT_RECORDS_PER_FAMILY,
            "alignment_workers": ALIGNMENT_WORKERS,
            "trace_token_chunk": TRACE_TOKEN_CHUNK,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "gradient_batches": GRADIENT_BATCHES,
            "alignment_modes": list(ALIGNMENT_MODES),
            "position_bins": [list(row) for row in POSITION_BINS],
            "thresholds": THRESHOLDS,
            "optimizer_step_allowed": False,
            "parameter_update_allowed": False,
            "model_or_checkpoint_write_allowed": False,
            "qwen_reencoding_allowed": False,
            "old_root_write_allowed": False,
            "authorization_on_complete": "one fresh C1 repair design only",
            "never_authorizes": ["C2", "C3", "V2-A PASS", "V2-B", "V2-C"],
        }
    )


__all__ = [
    "ALIGNMENT_MODES",
    "ALIGNMENT_WORKERS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CACHE_ROOT",
    "DESIGN_DOC",
    "EVALUATION_BATCH_SIZE",
    "FINAL_UPDATE",
    "FORMAL_ROOT",
    "GRADIENT_BATCHES",
    "IDENTITY",
    "LEASE_PATH",
    "MEASUREMENT_IDS",
    "OUTPUT_ROOT",
    "POSITION_BINS",
    "PREFLIGHT_LEASE",
    "PREFLIGHT_ROOT",
    "SCHEMA_PREFIX",
    "SELECTED_UPDATE",
    "THRESHOLDS",
    "TRACE_RECORDS_PER_FAMILY",
    "TRAIN_EXPOSURE_RECORDS_PER_FAMILY",
    "manifest",
]
