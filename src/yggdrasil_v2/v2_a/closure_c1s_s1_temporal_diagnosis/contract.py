from __future__ import annotations

"""Frozen contract for the successor temporal-only C1S S1 diagnosis.

The predecessor diagnosis is a sealed CRASH.  This identity never repairs,
extends, or reruns that root: it consumes D001--D003 only as pinned partial
evidence and performs a fresh D004R/D005R measurement under a target-only,
support-stratified fold ledger.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1s.srw.s1-temporal-attribution-repair"
IDENTITY = "V2-A-CLOSURE-C1S-SRW-S1-TEMPORAL-ATTRIBUTION-REPAIR-20260831-1"

OUTPUT_ROOT = Path(
    "artifacts/v2-a/closure-c1s-srw-s1-temporal-attribution-repair-20260831-1"
)
LEASE_PATH = Path(
    "artifacts/v2-a/closure-c1s-srw-s1-temporal-attribution-repair-20260831-1.diagnosis-lease.jsonl"
)
PREFLIGHT_ROOT = Path(
    "tmp/v2-a-closure-c1s-srw-s1-temporal-attribution-repair-preflight-20260831-1"
)
PREFLIGHT_LEASE = Path(
    "tmp/v2-a-closure-c1s-srw-s1-temporal-attribution-repair-preflight-20260831-1.preflight-lease.jsonl"
)
DESIGN_DOC = Path("docs/v2-a-closure-c1s-s1-temporal-attribution-repair.md")

PREDECESSOR_DIAGNOSIS_ROOT = Path(
    "artifacts/v2-a/closure-c1s-srw-s1-failure-attribution-20260831-1"
)
PREDECESSOR_PREFLIGHT_ROOT = Path(
    "tmp/v2-a-closure-c1s-srw-s1-failure-attribution-preflight-20260831-1"
)
PREDECESSOR_IDENTITY = "V2-A-CLOSURE-C1S-SRW-S1-FAILURE-ATTRIBUTION-20260831-1"
PREDECESSOR_PREFLIGHT_IDENTITY = f"{PREDECESSOR_IDENTITY}-PREFLIGHT"
PREDECESSOR_STAGE = "S1-FAILURE-ATTRIBUTION"
PREDECESSOR_PREFLIGHT_STAGE = f"{PREDECESSOR_STAGE}-PREFLIGHT"
PREDECESSOR_STATUS = "CRASH_V2_A_C1S_S1_FAILURE_ATTRIBUTION"
PREDECESSOR_RESULT_SHA256 = (
    "58F45CC0D1C315E3A3DBCAF33D319679C8F10ABAC08F629F1390CC8FC7FD6163"
)
PREDECESSOR_SEAL_SHA256 = (
    "AF45A857E5C674FE6B7F31E7C5931937D73D83F555102329A729138F13814242"
)
PREDECESSOR_PREFLIGHT_RESULT_SHA256 = (
    "31242B430CFB99A3675F98A2BAA12A0A4E9179C4379FA174FD11B8BB151902C0"
)
PREDECESSOR_PREFLIGHT_SEAL_SHA256 = (
    "E065BBF949A18C7950A9AF019E8BF9E962F82364E63C70FD179C82EE8A54CBD3"
)
PREDECESSOR_PARTIAL_FILES = {
    "D001-objective-gate-alignment.json": (
        "481E7F31EDA7B7D52DAF414D7ADECAD4636604E3D01748D3492AE525BC079E0B"
    ),
    "D002-h0-core-paths.json": (
        "C405B8BB46CE12B79E46DF4C86FCE79963EC69EF8F22528B18611C9C6758E74E"
    ),
    "D003-task-model-arity.json": (
        "FCA56DE67C7E2970ABC174AAB7CA987104EBCFED4AC2A6D9439E3E140B2307A2"
    ),
    "result.json": PREDECESSOR_RESULT_SHA256,
}

FAMILIES = ("CPS", "ERE")
HARD_FEATURES = {
    "CPS": ("processed", "running_best"),
    "ERE": ("touched", "changed", "operation_source", "operation_target"),
}
ANSWER_DERIVED_EXCLUDED = {
    "CPS": ("final_winner",),
    "ERE": ("query_semantic_match",),
}
STAGES = ("D000R", "D004R", "D005R")

FOLD_SEED = 2026083102
OUTER_FOLDS = 4
INNER_FOLDS = 3
MIN_OUTER_ELIGIBLE_RECORDS = 3
MIN_INNER_ELIGIBLE_RECORDS = 3
SCORE_NULL_SEED = 2026083103
SCORE_NULL_REPLICATES = 10_000

FOLD_POLICY: dict[str, Any] = {
    "basis": "target_only_before_any_latent_or_prediction_is_loaded",
    "cell": "family_x_feature",
    "eligibility": "record_has_at_least_one_positive_and_one_negative_masked_target",
    "ineligible_policy": "exclude_from_fit_inner_selection_outer_score_and_all_null_scores",
    "outer_folds": OUTER_FOLDS,
    "inner_folds": INNER_FOLDS,
    "minimum_outer_eligible_records": MIN_OUTER_ELIGIBLE_RECORDS,
    "minimum_inner_eligible_records": MIN_INNER_ELIGIBLE_RECORDS,
    "assignment": "sha256_sorted_round_robin_within_eligible_records",
    "same_ledger_for": [
        "natural",
        "time_shuffle",
        "temporal_mean",
        "within_record_target_shuffle_fit_null",
        "fixed_channel_auc",
    ],
    "record_macro_is_primary": True,
    "observation_micro_is_diagnostic_only": True,
}

TEMPORAL_READOUT: dict[str, Any] = {
    "features": {family: list(names) for family, names in HARD_FEATURES.items()},
    "answer_derived_excluded": {
        family: list(names) for family, names in ANSWER_DERIVED_EXCLUDED.items()
    },
    "target_adequacy": {
        "positive_observations_min": 32,
        "negative_observations_min": 32,
        "cross_step_changes_min": 16,
        "source_replay_required": True,
    },
    "decoder": {
        "kind": "nested_record_cross_fitted_record_and_within_record_class_balanced_truncated_ridge",
        "rank_grid": [8, 16, 32],
        "lambda_grid": [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0],
        "selection": "inner_fold_record_macro_balanced_accuracy_only",
        "outer_score": "record_macro_balanced_accuracy",
        "optimizer_steps": 0,
    },
    "fit_nulls": [
        "time_shuffle",
        "temporal_mean",
        "within_record_target_shuffle",
    ],
    "score_null": {
        "kind": "fixed_natural_crossfit_prediction_within_record_target_permutation",
        "seed": SCORE_NULL_SEED,
        "replicates": SCORE_NULL_REPLICATES,
    },
    "motion_relative_delta_floor": 0.05,
    "fixed_channel_record_macro_auc_floor": 0.70,
    "crossfit_record_macro_balanced_accuracy_floor": 0.80,
    "frozen_decoder_record_macro_balanced_accuracy_floor": 0.80,
    "readout_minus_strongest_fit_null_floor": 0.05,
    "score_null_max_p_value": 0.05,
    "static_null_gap_max": 0.02,
    "forbidden_predictors": [
        "answer",
        "final_winner",
        "margin",
        "target",
        "target_norm",
        "target_mask_as_feature",
        "any_target_derived_value",
    ],
}

ALLOWED_AXIS_C = (
    "TARGET_INSUFFICIENT",
    "READOUT_INSUFFICIENT",
    "LATENT_NOT_FORMED",
    "MIXED",
    "INCONCLUSIVE",
)

NEVER_AUTHORIZES = (
    "S2",
    "S3",
    "single-seed formal",
    "new training",
    "checkpoint selection",
    "C2",
    "V2-A PASS",
    "V2-B",
    "V2-C",
)


def contract_manifest() -> dict[str, Any]:
    """Return the frozen JSON-native contract."""

    return deepcopy(
        {
            "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
            "identity": IDENTITY,
            "diagnostic_only": True,
            "formal": False,
            "scope": "D004R_D005R_only",
            "stages": list(STAGES),
            "families": list(FAMILIES),
            "output_root": OUTPUT_ROOT.as_posix(),
            "lease_path": LEASE_PATH.as_posix(),
            "preflight_root": PREFLIGHT_ROOT.as_posix(),
            "preflight_lease": PREFLIGHT_LEASE.as_posix(),
            "design_doc": DESIGN_DOC.as_posix(),
            "predecessor": {
                "root": PREDECESSOR_DIAGNOSIS_ROOT.as_posix(),
                "preflight_root": PREDECESSOR_PREFLIGHT_ROOT.as_posix(),
                "identity": PREDECESSOR_IDENTITY,
                "status": PREDECESSOR_STATUS,
                "result_sha256": PREDECESSOR_RESULT_SHA256,
                "seal_sha256": PREDECESSOR_SEAL_SHA256,
                "preflight_result_sha256": PREDECESSOR_PREFLIGHT_RESULT_SHA256,
                "preflight_seal_sha256": PREDECESSOR_PREFLIGHT_SEAL_SHA256,
                "partial_files": PREDECESSOR_PARTIAL_FILES,
                "write_allowed": False,
                "rerun_allowed": False,
            },
            "predecessor_partial_evidence_policy": (
                "D001-D003 are hash-pinned context only and are never recomputed or "
                "presented as measurements of this identity"
            ),
            "fold_seed": FOLD_SEED,
            "fold_policy": FOLD_POLICY,
            "temporal_readout": TEMPORAL_READOUT,
            "allowed_axis_c": list(ALLOWED_AXIS_C),
            "optimizer_step_allowed": False,
            "parameter_update_allowed": False,
            "model_or_checkpoint_write_allowed": False,
            "training_data_write_allowed": False,
            "authorizes": "nothing",
            "v2a_passed": False,
            "never_authorizes": list(NEVER_AUTHORIZES),
        }
    )


__all__ = [
    "ALLOWED_AXIS_C",
    "ANSWER_DERIVED_EXCLUDED",
    "DESIGN_DOC",
    "FAMILIES",
    "FOLD_POLICY",
    "FOLD_SEED",
    "HARD_FEATURES",
    "IDENTITY",
    "INNER_FOLDS",
    "LEASE_PATH",
    "MIN_INNER_ELIGIBLE_RECORDS",
    "MIN_OUTER_ELIGIBLE_RECORDS",
    "NEVER_AUTHORIZES",
    "OUTER_FOLDS",
    "OUTPUT_ROOT",
    "PREDECESSOR_DIAGNOSIS_ROOT",
    "PREDECESSOR_IDENTITY",
    "PREDECESSOR_PARTIAL_FILES",
    "PREDECESSOR_PREFLIGHT_IDENTITY",
    "PREDECESSOR_PREFLIGHT_RESULT_SHA256",
    "PREDECESSOR_PREFLIGHT_ROOT",
    "PREDECESSOR_PREFLIGHT_SEAL_SHA256",
    "PREDECESSOR_PREFLIGHT_STAGE",
    "PREDECESSOR_RESULT_SHA256",
    "PREDECESSOR_SEAL_SHA256",
    "PREDECESSOR_STAGE",
    "PREDECESSOR_STATUS",
    "PREFLIGHT_LEASE",
    "PREFLIGHT_ROOT",
    "SCHEMA_PREFIX",
    "SCORE_NULL_REPLICATES",
    "SCORE_NULL_SEED",
    "STAGES",
    "TEMPORAL_READOUT",
    "contract_manifest",
]
