from __future__ import annotations

"""Frozen contract for the decision-causal write/delete mechanism screen.

This is a fresh, isolated non-formal screen.  It consumes only the immutable
predecessor checkpoint used by the earlier H1 screen.  In particular, the
earlier overlap-residual WD checkpoints are not valid inputs to this contract.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "yggdrasil.v2-r1r.p1-h1-wd-decision-causal"
CONTRACT_VERSION = "p1-h1-wd-decision-causal-v1"
SCOPE = "NONFORMAL_WD_DECISION_CAUSAL_SCREEN_ONLY"

PREDECESSOR_ROOT = Path(
    "artifacts/v2-r1r/"
    "p1-h1-nonformal-factorized-routed-projection-screen-20260817-1"
)
SOURCE_ROOT = PREDECESSOR_ROOT
PREDECESSOR_CHECKPOINT = PREDECESSOR_ROOT / "mixed-deployment.pt"
SOURCE_CHECKPOINT = PREDECESSOR_CHECKPOINT
SOURCE_TOKEN_CACHE = PREDECESSOR_ROOT / "token-cache"
SOURCE_CHECKPOINT_SHA256 = (
    "112622861726391BD7D15F2937C7428302021D00DE2B0D8B93E3707EA7B0745D"
)
SOURCE_PACKAGE_IDENTITY = (
    "A52C52225921A0F834A43031A9F4E595B301D7B99AACD2419882D41198486DA4"
)

# The old WD root is intentionally named as a forbidden input.  This prevents
# a seemingly convenient checkpoint warm-start from becoming an unrecorded
# change of experiment identity.
FORBIDDEN_WD_ROOT = Path(
    "artifacts/v2-r1r/"
    "p1-h1-wd-overlap-residual-screen-20260821-1"
)
OUTPUT_ROOT = Path(
    "artifacts/v2-r1r/"
    "p1-h1-wd-decision-causal-screen-20260821-1"
)
DESIGN_DOCUMENT = Path(
    "docs/v2-r1r-p1-h1-wd-decision-causal-design.md"
)
EXECUTION_DOCUMENT = Path(
    "docs/v2-r1r-p1-h1-wd-decision-causal-execution-command.md"
)


@dataclass(frozen=True)
class DecisionCausalSpec:
    """Frozen training, target-construction, and screen Gate constants."""

    batch_size: int = 32
    target_microbatch_size: int = 4
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

    # The causal target uses half of the real common-off answer-margin drop.
    margin_transfer_fraction: float = 0.50
    maximum_site_transfer_fraction_of_common_norm: float = 0.25
    minimum_positive_target_fraction: float = 0.10
    minimum_realized_request_ratio: float = 0.20
    maximum_gpu_smoke_gb: float = 6.0

    # J preserves the W/D two-path causal allocation without a teacher model.
    allocation_lock_weight: float = 0.25
    teacher_kl_weight: float = 0.0

    # Fit and mechanism Gates.
    maximum_causal_write_normalized_mse: float = 0.08
    maximum_control_write_normalized_mse: float = 0.08
    maximum_causal_delete_normalized_mse: float = 0.08
    maximum_control_delete_normalized_mse: float = 0.08
    minimum_causal_rollout_agreement: float = 0.90
    minimum_final_free_route_agreement: float = 0.90
    minimum_causal_common_off_drop: float = 0.02
    minimum_projection_effect_gain_vs_source: float = 0.02
    minimum_common_residual_energy: float = 0.50
    maximum_common_residual_energy: float = 0.95

    # Route-replay Shapley and negative-VJP directional-control Gates.
    minimum_shapley_projection_share_gain_vs_source: float = 0.05
    minimum_shapley_projection_share_gain_vs_control: float = 0.05
    minimum_shapley_component_share: float = 0.20
    minimum_shapley_ci_lower: float = 0.0
    minimum_causal_control_heldout_gain: float = 0.05
    minimum_causal_control_ci_lower: float = 0.0
    maximum_family_regression: float = 0.02

    schedule_seed: int = 2026082121
    device: str = "cuda"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


SPEC = DecisionCausalSpec()


def contract_manifest() -> dict[str, Any]:
    """Return the serializable frozen identity consumed by the runner."""

    return {
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "contract_version": CONTRACT_VERSION,
        "scope": SCOPE,
        "source_root": SOURCE_ROOT.as_posix(),
        "predecessor_root": PREDECESSOR_ROOT.as_posix(),
        "source_checkpoint": SOURCE_CHECKPOINT.as_posix(),
        "predecessor_checkpoint": PREDECESSOR_CHECKPOINT.as_posix(),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_package_identity": SOURCE_PACKAGE_IDENTITY,
        "source_token_cache": SOURCE_TOKEN_CACHE.as_posix(),
        "forbidden_wd_root": FORBIDDEN_WD_ROOT.as_posix(),
        "output_root": OUTPUT_ROOT.as_posix(),
        "design_document": DESIGN_DOCUMENT.as_posix(),
        "execution_document": EXECUTION_DOCUMENT.as_posix(),
        "spec": SPEC.as_dict(),
        "method": {
            "answer_margin": "true_logit_minus_logsumexp_other_logits",
            "amplitude": "positive_fixed_route_common_off_margin_drop",
            "vjp_site": "selected_projection_output",
            "site_allocation": "squared_vjp_fisher_share",
            "route_schedule": "frozen_source_replay",
            "target_storage": "materialized_cpu_train_and_heldout",
            "directional_control": "equal_norm_negative_vjp",
            "write_delete_fit": "transfer_increment_normalized_mse",
            "joint_objective": "answer_ce_plus_two_path_causal_target_lock",
            "joint_answer_route_mode": "model_free_route",
            "joint_allocation_route_mode": "frozen_source_route_replay",
            "teacher_model": False,
            "route_supervision": False,
            "family_or_task_training_target": False,
        },
        "claims": {
            "h1_qualified": False,
            "p1_completed": False,
            "real_text_qualified": False,
            "f1_authorized": False,
            "p2_authorized": False,
        },
        "authorizes": "nothing",
        "fail_stop": {
            "stop_on_any_gate_failure": True,
            "rerun": False,
            "retune": False,
            "formal_successor": False,
        },
    }


__all__ = [
    "CONTRACT_VERSION",
    "DESIGN_DOCUMENT",
    "EXECUTION_DOCUMENT",
    "FORBIDDEN_WD_ROOT",
    "OUTPUT_ROOT",
    "PREDECESSOR_CHECKPOINT",
    "PREDECESSOR_ROOT",
    "SCHEMA_PREFIX",
    "SOURCE_CHECKPOINT",
    "SOURCE_CHECKPOINT_SHA256",
    "SOURCE_PACKAGE_IDENTITY",
    "SOURCE_ROOT",
    "SOURCE_TOKEN_CACHE",
    "SCOPE",
    "SPEC",
    "DecisionCausalSpec",
    "contract_manifest",
]
