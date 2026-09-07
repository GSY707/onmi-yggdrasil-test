from __future__ import annotations

"""Frozen successor contract for the R1-R4 direction-geometry diagnosis."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "yggdrasil.v2-r1r.h1-wd.direction-geometry-v2"
CONTRACT_VERSION = "h1-wd-direction-geometry-v2-full-replay"
IDENTITY = "H1-WD-DG-V2-20260823-1"
SCOPE = "NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_ONLY"
AUTHORIZES = "nothing"

SOURCE_ROOT = Path(
    "artifacts/v2-r1r/"
    "p1-h1-nonformal-factorized-routed-projection-screen-20260817-1"
)
SOURCE_CHECKPOINT = SOURCE_ROOT / "mixed-deployment.pt"
SOURCE_TOKEN_CACHE = SOURCE_ROOT / "token-cache"
SOURCE_CACHE_MANIFEST = SOURCE_TOKEN_CACHE / "cache-manifest.json"
SOURCE_CHECKPOINT_SHA256 = (
    "112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D"
)
SOURCE_PACKAGE_IDENTITY = (
    "A52C52225921A0F834A43031A9F4E595B301D7B99AACD2419882D41198486DA4"
)
SOURCE_CACHE_MANIFEST_SHA256 = (
    "D5DA0B6875A227BDF07EC715517F432806D31A744BE06E527C0049609D633860"
)
SOURCE_TOKEN_CACHE_SHA256 = {
    "source-tokens-hidden.npy": (
        "E17BDD54F88843CA96D8A20D8C2F1B4B1B3B537A86534FA5C45125D15E8999B1"
    ),
    "source-tokens-index.json": (
        "07BAA4442F8CF2C26D6D37D976EC74F9CE71AD6929DF5B2FA368A498E62904B4"
    ),
    "source-tokens-mask.npy": (
        "FA8FE6C81D4943DF4CC60F448DD52C5E7BD21DC7E0480B594CA116245D15A15A"
    ),
}

CAUSAL_ROOT = Path(
    "artifacts/v2-r1r/"
    "p1-h1-wd-decision-causal-screen-20260821-1"
)
CAUSAL_TARGET_BANK = CAUSAL_ROOT / "causal-target-datasets.pt"
CAUSAL_TARGET_BANK_SHA256 = (
    "DFC8F08776CE56EFB8022BF35B7C3DD72F8F1C53C95DBF0DD28F593DC2BFD503"
)
CAUSAL_TARGET_BANK_MANIFEST = CAUSAL_ROOT / "target-bank.json"
CAUSAL_TARGET_BANK_MANIFEST_SHA256 = (
    "C7275930DE3D5549EA4F776E79E78E69DE7ED8FA0E11B398ED941E5853AFAAE2"
)
CAUSAL_TARGET_DATASETS = CAUSAL_TARGET_BANK
CAUSAL_TARGET_DATASETS_SHA256 = CAUSAL_TARGET_BANK_SHA256

FORBIDDEN_OLD_WD_ROOT = Path(
    "artifacts/v2-r1r/p1-h1-wd-overlap-residual-screen-20260821-1"
)
FORBIDDEN_OLD_CAUSAL_COMMAND = Path(
    "experiments/v2_r1r_h1_wd_decision_causal.py"
)
PREDECESSOR_CRASH_ROOT = Path(
    "artifacts/v2-r1r/h1-wd-direction-geometry-screen-20260823-1"
)
PREDECESSOR_CRASH_FILES_SHA256 = {
    "contract-manifest.json": (
        "59076C51DD8A0E69A995C1D7E8955603049BA688F6BEC93B5B67B93CCDFDB658"
    ),
    "preflight.json": (
        "0D629C85B7E8A34EF0C6FB9C1C732D60B95DDDAC3D21E80510A5B40F063A00CC"
    ),
    "events.jsonl": (
        "CDD06C10C887904D333BBA1C12504A4F80AD66A0163F8742C3541AFB4D713811"
    ),
    "run-state.json": (
        "EC2660A8D9B0D315FE265841B57E36A63E4E84FF3F844B8EA800012F37EA2BC9"
    ),
    "crash.json": (
        "EC2660A8D9B0D315FE265841B57E36A63E4E84FF3F844B8EA800012F37EA2BC9"
    ),
}
PREDECESSOR_CRASH_SIBLING_SHA256 = {
    "artifacts/v2-r1r/h1-wd-direction-geometry-screen-20260823-1.stdout.log": (
        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
    ),
    "artifacts/v2-r1r/h1-wd-direction-geometry-screen-20260823-1.stderr.log": (
        "A1CE63B2BAFBB78DBF5E7A07362194F4589CC0FF6BE863AAA93649A1EC54DF03"
    ),
}
OUTPUT_ROOT = Path(
    "artifacts/v2-r1r/h1-wd-direction-geometry-screen-v2-20260823-1"
)
PREFLIGHT_LEASE = Path(
    "artifacts/v2-r1r/"
    "h1-wd-direction-geometry-screen-v2-20260823-1.preflight-lease.jsonl"
)
DESIGN_DOCUMENT = Path(
    "docs/v2-r1r-h1-wd-direction-geometry-v2-design.md"
)
EXECUTION_DOCUMENT = Path(
    "docs/v2-r1r-h1-wd-direction-geometry-v2-execution-command.md"
)


@dataclass(frozen=True)
class DirectionGeometrySpec:
    device: str = "cuda"
    train_records: int = 4096
    heldout_records: int = 1024
    sites_per_record: int = 16
    latent_slots: int = 8
    latent_width: int = 256
    routed_feature_width: int = 384
    minimum_route_cell_records: int = 128
    numerical_epsilon: float = 1.0e-12
    replay_microbatch_size: int = 4
    analysis_batch_size: int = 128
    maximum_replay_absolute_error: float = 2.5e-5
    ridge_lambda_grid: tuple[float, ...] = (1.0e-6, 1.0e-4, 1.0e-2, 1.0)
    ridge_validation_folds: int = 5
    primary_r2_feature: str = "projection_feature_trunk"
    local_full_j_train_records: int = 256
    local_full_j_heldout_records: int = 128
    local_full_j_damping: float = 1.0e-4
    local_full_j_cg_iterations: int = 16
    local_full_j_fisher_probes: int = 4
    residual_sketch_per_site: int = 8
    residual_spectrum_rank: int = 32
    bootstrap_replicates: int = 2000
    permutation_replicates: int = 10000
    fit_null_replicates: int = 128
    matched_spectrum_replicates: int = 1024
    alignment_null_replicates: int = 1024
    route_conditional_sign_replicates: int = 2048
    residual_null_family_tests: int = 12
    minimum_regularization_positive_fraction: float = 0.75
    maximum_negative_sign_symmetry_error: float = 1.0e-10
    minimum_site_reference_energy: float = 1.0e-30
    maximum_qualified_heldout_nmse: float = 0.08
    maximum_qualified_error_ratio: float = 1.5
    maximum_null_p_value: float = 0.01
    random_seed: int = 2026082301
    optimizer_steps: int = 0
    model_writes: bool = False
    old_root_mutation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


SPEC = DirectionGeometrySpec()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def input_manifest() -> dict[str, Any]:
    return {
        "source_root": SOURCE_ROOT.as_posix(),
        "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_package_identity": SOURCE_PACKAGE_IDENTITY,
        "source_token_cache": SOURCE_TOKEN_CACHE.as_posix(),
        "source_cache_manifest": SOURCE_CACHE_MANIFEST.as_posix(),
        "source_cache_manifest_sha256": SOURCE_CACHE_MANIFEST_SHA256,
        "source_token_cache_sha256": dict(SOURCE_TOKEN_CACHE_SHA256),
        "causal_root": CAUSAL_ROOT.as_posix(),
        "causal_target_bank": CAUSAL_TARGET_BANK.as_posix(),
        "causal_target_bank_sha256": CAUSAL_TARGET_BANK_SHA256,
        "causal_target_bank_manifest": CAUSAL_TARGET_BANK_MANIFEST.as_posix(),
        "causal_target_bank_manifest_sha256": CAUSAL_TARGET_BANK_MANIFEST_SHA256,
        "forbidden_old_wd_root": FORBIDDEN_OLD_WD_ROOT.as_posix(),
        "forbidden_old_causal_command": FORBIDDEN_OLD_CAUSAL_COMMAND.as_posix(),
        "predecessor_crash_root": PREDECESSOR_CRASH_ROOT.as_posix(),
        "predecessor_crash_files_sha256": dict(PREDECESSOR_CRASH_FILES_SHA256),
        "predecessor_crash_sibling_sha256": dict(PREDECESSOR_CRASH_SIBLING_SHA256),
        "output_root": OUTPUT_ROOT.as_posix(),
        "preflight_lease": PREFLIGHT_LEASE.as_posix(),
        "allowed_evidence": [
            "immutable predecessor checkpoint and token cache",
            "materialized causal target bank in prepared train/heldout order",
            "target-bank/preflight/contract/result metadata",
            "frozen H1/H1-WD model and route-replay implementation",
        ],
        "forbidden_operations": [
            "optimizer step or parameter update",
            "model/checkpoint write",
            "causal target regeneration",
            "old root mutation, deletion, or rerun",
            "family/task/answer supervision",
            "calibration, formal, F1, P1, or P2 successor",
        ],
    }


def contract_manifest() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "contract_version": CONTRACT_VERSION,
        "identity": IDENTITY,
        "scope": SCOPE,
        "authorizes": AUTHORIZES,
        "source": input_manifest(),
        "spec": SPEC.as_dict(),
        "method": {
            "R0": "record_axis_centroid_preserving_site_slot_output_coordinates",
            "R1": "record_route_centroid_preserving_site_slot_output_coordinates",
            "R2": (
                "target_before_attention_projection_feature_common_hidden_ridge; "
                "bank outputs are proxy controls only"
            ),
            "R3": (
                "full-bank exact shared final-head Jacobian plus sampled matrix-free "
                "local trunk-and-head Jacobian/Fisher"
            ),
            "R4": (
                "target-and-feature fit nulls, record/site-macro bootstrap, negative-sign "
                "symmetry, dual-sketch sign/spectrum/alignment audit"
            ),
            "primary_metric": "heldout_transfer_increment_normalized_mse",
            "unit_of_resampling": "record_cluster_across_16_sites",
            "heldout_is_iid": False,
            "heldout_risk": (
                "prepared heldout is a finite historical split, not an IID population claim"
            ),
            "route_is_supervision": False,
            "family_or_task_target": False,
            "optimizer": False,
            "model_write": False,
        },
        "claims": {
            "direction_geometry_signal": False,
            "parameter_reachable_signal": False,
            "h1_qualified": False,
            "p1_completed": False,
            "f1_authorized": False,
            "p2_authorized": False,
        },
        "fail_stop": {
            "stop_on_hard_identity_or_integrity_failure": True,
            "scientific_null_or_no_signal_is_a_valid_completed_result": True,
            "attempt_is_consumed_by_exclusive_preflight_lease_before_hard_gates": True,
            "pre_root_failure_is_recorded_in_append_only_lease": True,
            "rerun_old_root": False,
            "retune_old_root": False,
            "same_identity_attempts": 1,
            "infrastructure_successors_authorized_by_user": True,
            "maximum_infrastructure_successor_identities": 3,
        },
    }


def verify_fixed_inputs(
    repo_root: Path, *, expect_preflight_lease_absent: bool = True
) -> dict[str, Any]:
    """Read-only hash/path preflight; never creates or changes a file."""

    def absolute(relative: Path) -> Path:
        return repo_root / relative

    checkpoint = absolute(SOURCE_CHECKPOINT)
    target_bank = absolute(CAUSAL_TARGET_BANK)
    target_bank_manifest = absolute(CAUSAL_TARGET_BANK_MANIFEST)
    cache_manifest = absolute(SOURCE_CACHE_MANIFEST)
    output_root = absolute(OUTPUT_ROOT)
    preflight_lease = absolute(PREFLIGHT_LEASE)
    checks = {
        "source_checkpoint_exists": checkpoint.is_file(),
        "target_bank_exists": target_bank.is_file(),
        "target_bank_manifest_exists": target_bank_manifest.is_file(),
        "cache_manifest_exists": cache_manifest.is_file(),
        "output_root_absent": not output_root.exists(),
    }
    if expect_preflight_lease_absent:
        checks["preflight_lease_absent"] = not preflight_lease.exists()
    else:
        checks["preflight_lease_present"] = preflight_lease.is_file()
    for name, expected in PREDECESSOR_CRASH_FILES_SHA256.items():
        path = absolute(PREDECESSOR_CRASH_ROOT / name)
        checks[f"predecessor_crash_{name}_exists"] = path.is_file()
        checks[f"predecessor_crash_{name}_hash"] = (
            path.is_file() and _sha256(path) == expected
        )
    for relative, expected in PREDECESSOR_CRASH_SIBLING_SHA256.items():
        path = absolute(Path(relative))
        key = Path(relative).name
        checks[f"predecessor_crash_{key}_exists"] = path.is_file()
        checks[f"predecessor_crash_{key}_hash"] = (
            path.is_file() and _sha256(path) == expected
        )
    checks["source_checkpoint_hash"] = (
        checks["source_checkpoint_exists"]
        and _sha256(checkpoint) == SOURCE_CHECKPOINT_SHA256
    )
    checks["target_bank_hash"] = (
        checks["target_bank_exists"]
        and _sha256(target_bank) == CAUSAL_TARGET_BANK_SHA256
    )
    checks["target_bank_manifest_hash"] = (
        checks["target_bank_manifest_exists"]
        and _sha256(target_bank_manifest) == CAUSAL_TARGET_BANK_MANIFEST_SHA256
    )
    checks["cache_manifest_hash"] = (
        checks["cache_manifest_exists"]
        and _sha256(cache_manifest) == SOURCE_CACHE_MANIFEST_SHA256
    )
    for name, expected in SOURCE_TOKEN_CACHE_SHA256.items():
        path = absolute(SOURCE_TOKEN_CACHE / name)
        checks[f"cache_{name}_exists"] = path.is_file()
        checks[f"cache_{name}_hash"] = (
            path.is_file() and _sha256(path) == expected
        )
    return {
        "schema_version": f"{SCHEMA_PREFIX}.preflight.v1",
        "checks": checks,
        "passed": all(checks.values()),
        "authorizes": AUTHORIZES,
    }


__all__ = [
    "AUTHORIZES", "CAUSAL_ROOT", "CAUSAL_TARGET_BANK",
    "CAUSAL_TARGET_BANK_MANIFEST", "CAUSAL_TARGET_BANK_MANIFEST_SHA256",
    "CAUSAL_TARGET_BANK_SHA256", "CAUSAL_TARGET_DATASETS",
    "CAUSAL_TARGET_DATASETS_SHA256", "CONTRACT_VERSION", "DESIGN_DOCUMENT",
    "EXECUTION_DOCUMENT", "FORBIDDEN_OLD_CAUSAL_COMMAND", "IDENTITY",
    "FORBIDDEN_OLD_WD_ROOT", "OUTPUT_ROOT", "PREFLIGHT_LEASE", "PREDECESSOR_CRASH_FILES_SHA256",
    "PREDECESSOR_CRASH_ROOT", "PREDECESSOR_CRASH_SIBLING_SHA256", "SCOPE", "SOURCE_CACHE_MANIFEST",
    "SOURCE_CACHE_MANIFEST_SHA256", "SOURCE_CHECKPOINT",
    "SOURCE_CHECKPOINT_SHA256", "SOURCE_PACKAGE_IDENTITY", "SOURCE_ROOT",
    "SOURCE_TOKEN_CACHE", "SOURCE_TOKEN_CACHE_SHA256", "SPEC",
    "DirectionGeometrySpec", "contract_manifest", "input_manifest",
    "verify_fixed_inputs",
]
