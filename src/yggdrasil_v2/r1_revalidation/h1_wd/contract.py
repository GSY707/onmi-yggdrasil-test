from __future__ import annotations

"""Frozen constants for the isolated, non-formal H1-WD screen.

This experiment consumes the failed H1 screen checkpoint as an immutable
development starting point.  It cannot alter the old root and cannot qualify
H1, P1, F1, or P2.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "yggdrasil.v2-r1r.p1-h1-wd"
CONTRACT_VERSION = "p1-h1-wd-overlap-residual-v1"

SOURCE_ROOT = Path(
    "artifacts/v2-r1r/"
    "p1-h1-nonformal-factorized-routed-projection-screen-20260817-1"
)
SOURCE_CHECKPOINT = SOURCE_ROOT / "mixed-deployment.pt"
SOURCE_TOKEN_CACHE = SOURCE_ROOT / "token-cache"
SOURCE_CHECKPOINT_SHA256 = (
    "112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D"
)
SOURCE_PACKAGE_IDENTITY = (
    "A52C52225921A0F834A43031A9F4E595B301D7B99AACD2419882D41198486DA4"
)

OUTPUT_ROOT = Path(
    "artifacts/v2-r1r/p1-h1-wd-overlap-residual-screen-20260821-1"
)
DESIGN_DOCUMENT = Path("docs/v2-r1r-p1-h1-wd-overlap-residual-design.md")
EXECUTION_DOCUMENT = Path("docs/v2-r1r-p1-h1-wd-execution-command.md")


@dataclass(frozen=True)
class WriteDeleteSpec:
    """Pre-result training and Gate constants for the one development run."""

    batch_size: int = 32
    write_updates: int = 800
    delete_updates: int = 800
    joint_updates: int = 1200
    evaluation_interval: int = 200
    write_learning_rate: float = 1.0e-4
    delete_learning_rate: float = 1.0e-4
    joint_learning_rate: float = 5.0e-5
    minimum_learning_rate_scale: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    overlap_coefficient_max: float = 0.75
    teacher_kl_weight: float = 0.25
    schedule_seed: int = 2026082117
    device: str = "cuda"

    # Development Gates.  These do not replace the frozen H1 H05/H06 Gates.
    maximum_write_normalized_mse: float = 0.08
    maximum_delete_normalized_mse: float = 0.08
    minimum_prediction_agreement: float = 0.90
    maximum_heldout_accuracy_drop: float = 0.02
    minimum_common_off_drop: float = 0.02
    minimum_projection_effect_gain: float = 0.02
    minimum_common_residual_energy: float = 0.50
    maximum_common_residual_energy: float = 0.95
    architecture_gain_threshold: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


SPEC = WriteDeleteSpec()


def contract_manifest() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "contract_version": CONTRACT_VERSION,
        "scope": "NONFORMAL_WD_MECHANISM_SCREEN_ONLY",
        "source_root": SOURCE_ROOT.as_posix(),
        "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_package_identity": SOURCE_PACKAGE_IDENTITY,
        "source_token_cache": SOURCE_TOKEN_CACHE.as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "design_document": DESIGN_DOCUMENT.as_posix(),
        "execution_document": EXECUTION_DOCUMENT.as_posix(),
        "spec": SPEC.as_dict(),
        "claims": {
            "h1_qualified": False,
            "p1_completed": False,
            "f1_authorized": False,
            "p2_authorized": False,
        },
    }


__all__ = [
    "CONTRACT_VERSION",
    "DESIGN_DOCUMENT",
    "EXECUTION_DOCUMENT",
    "OUTPUT_ROOT",
    "SCHEMA_PREFIX",
    "SOURCE_CHECKPOINT",
    "SOURCE_CHECKPOINT_SHA256",
    "SOURCE_PACKAGE_IDENTITY",
    "SOURCE_ROOT",
    "SOURCE_TOKEN_CACHE",
    "SPEC",
    "WriteDeleteSpec",
    "contract_manifest",
]
