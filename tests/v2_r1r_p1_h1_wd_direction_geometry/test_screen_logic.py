from __future__ import annotations

from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.screen import (
    _residual_null_decision,
)


def test_split_half_signal_blocks_random_compatible_classification() -> None:
    quiet = 0.5
    decision = _residual_null_decision(
        spectrum_p=quiet,
        resultant_p=quiet,
        route_resultant_p={"0": quiet, "1": quiet},
        cross_alignment_p=quiet,
        split_half_alignment_p=0.0001,
    )
    assert decision["split_half"] is True
    assert decision["random_compatible"] is False
    assert decision["classification"] == "unstable_signal"


def test_each_sketch_family_uses_the_twelve_test_bonferroni_alpha() -> None:
    decision = _residual_null_decision(
        spectrum_p=0.5,
        resultant_p=0.5,
        route_resultant_p={"0": 0.5, "1": 0.5},
        cross_alignment_p=0.5,
        split_half_alignment_p=0.5,
    )
    assert decision["alpha"] == 0.01 / 12
    assert decision["random_compatible"] is True
    assert decision["classification"] == "consistent_with_current_registered_nulls_only"


def test_route_conditional_residual_direction_cannot_hide_in_global_cancellation() -> None:
    decision = _residual_null_decision(
        spectrum_p=0.5,
        resultant_p=0.5,
        route_resultant_p={"0": 0.0001, "1": 0.5},
        cross_alignment_p=0.0001,
        split_half_alignment_p=0.5,
    )
    assert decision["route_resultant"]["0"] is True
    assert decision["random_compatible"] is False
    assert decision["classification"] == "R5_latent_or_unmodeled_structure"
