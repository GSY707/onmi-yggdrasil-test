from __future__ import annotations

from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis import contract
from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis.decision import (
    classify_axis_c,
)


def rows_for(
    *,
    adequate: bool = True,
    motion: float = 0.10,
    separation: float = 0.85,
    crossfit: float = 0.90,
    frozen: float = 0.70,
    gap: float = 0.10,
    p_value: float = 0.01,
) -> list[dict]:
    return [
        {
            "family": family,
            "feature": feature,
            "target_adequate": adequate,
            "fold_coverage_passed": True,
            "relative_motion": motion,
            "separation_auc": separation,
            "crossfit_decoder_balanced_accuracy": crossfit,
            "frozen_decoder_balanced_accuracy": frozen,
            "fit_null_gap": gap,
            "score_null_p_value": p_value,
        }
        for family, features in contract.HARD_FEATURES.items()
        for feature in features
    ]


def test_readout_insufficient_requires_latent_and_null_qualified_signal() -> None:
    report = classify_axis_c(rows_for())
    assert report["family_complete"] is True
    assert {cell["classification"] for cell in report["families"].values()} == {
        "READOUT_INSUFFICIENT"
    }


def test_target_insufficient_dominates() -> None:
    report = classify_axis_c(rows_for(adequate=False))
    assert {cell["classification"] for cell in report["families"].values()} == {
        "TARGET_INSUFFICIENT"
    }


def test_latent_not_formed_requires_null_like_failure() -> None:
    report = classify_axis_c(
        rows_for(motion=0.01, separation=0.55, crossfit=0.55, frozen=0.55, gap=0.0, p_value=0.5)
    )
    assert {cell["classification"] for cell in report["families"].values()} == {
        "LATENT_NOT_FORMED"
    }


def test_sufficient_frozen_decoder_does_not_become_a_composite_pass() -> None:
    report = classify_axis_c(rows_for(frozen=0.90))
    assert {cell["classification"] for cell in report["families"].values()} == {
        "INCONCLUSIVE"
    }


def test_missing_feature_fails_family_completeness() -> None:
    rows = rows_for()
    rows.pop()
    report = classify_axis_c(rows)
    assert report["family_complete"] is False
    assert "ERE" in report["missing_families"]
