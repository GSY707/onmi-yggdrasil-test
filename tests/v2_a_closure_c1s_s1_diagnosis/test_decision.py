import pytest

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis.decision import (
    classify_a_axis,
    classify_b_axis,
    classify_c_axis,
    classify_loss_gate_alignment,
)


THRESHOLDS = {
    "h0_direct_point_min": 0.75,
    "h0_direct_wilson_lower_min": 0.60,
    "h0_low_point_max": 0.25,
    "full_point_min": 0.75,
    "core_only_point_min": 0.60,
    "core_only_point_max": 0.25,
    "full_h0_margin_drop_min": 0.50,
    "h0_relevant_margin_drop_min": 0.50,
    "full_core_margin_drop_min": 0.50,
    "final_delta_point_max": 0.35,
    "full_final_delta_margin_drop_min": 0.50,
}


def _a_rows(family: str, n: int, h0: bool, core: bool) -> list[dict]:
    return [
        {
            "family": family,
            "full_correct": True,
            "h0_correct": h0,
            "core_only_correct": core,
            "final_delta_correct": False,
            "full_h0_margin_drop": 1.0 if not h0 else 0.0,
            "full_core_margin_drop": 1.0 if not core else 0.0,
            "full_h0_margin_drop_lower": 1.0 if not h0 else 0.0,
            "h0_relevant_margin_drop_lower": 1.0,
            "full_core_margin_drop_lower": 1.0 if not core else 0.0,
            "full_final_delta_margin_drop_lower": 1.0,
        }
        for _ in range(n)
    ]


def test_a_axis_keeps_ere_and_cps_family_classification_separate() -> None:
    rows = _a_rows("ERE", 16, True, False) + _a_rows("CPS", 16, False, True)
    result = classify_a_axis(rows, THRESHOLDS)
    assert result["families"]["ERE"]["classification"] == "H0_DIRECT"
    assert result["families"]["CPS"]["classification"] == "CORE_REQUIRED"
    assert result["overall"]["label"] == "MIXED_FAMILY_RESULTS"


def test_a_axis_reports_core_residual_when_core_only_does_not_suffice() -> None:
    thresholds = {**THRESHOLDS, "full_h0_margin_drop_min": 0.5, "full_core_margin_drop_min": 0.5}
    rows = _a_rows("CPS", 16, False, False)
    assert classify_a_axis(rows, thresholds)["families"]["CPS"]["classification"] == "CORE_RESIDUAL"


def test_a_axis_does_not_confuse_full_core_gain_with_h0_owner_dependence() -> None:
    rows = _a_rows("ERE", 16, True, False)
    for row in rows:
        row["h0_relevant_margin_drop_lower"] = 0.0
    assert classify_a_axis(rows, THRESHOLDS)["families"]["ERE"]["classification"] == "MIXED"


def test_b_axis_distinguishes_single_copy_from_multi_address() -> None:
    thresholds = {
        "task_causal_assessed_point_min": 0.75,
        "task_multi_object_causal_point_min": 0.75,
        "task_low_order_supported_point_min": 0.75,
        "baseline_eligible_point_min": 0.75,
        "minimum_eligible_records_per_family": 8,
        "copy_retention_point_min": 0.80,
        "copy_retention_wilson_lower_min": 0.65,
        "two_contributor_point_max": 0.50,
        "two_contributor_wilson_lower_min": 0.65,
        "copy_collapse_bootstrap_lower_min": 0.50,
        "bootstrap_seed": 17,
        "bootstrap_replicates": 100,
    }
    common = {
        "certificate_multi": True,
        "task_causal_assessed": True,
        "task_multi_object_causal": True,
        "task_low_order_supported": False,
        "task_evidence_conflict": False,
        "baseline_eligible": True,
        "contributor_denominator_valid": True,
        "full_margin": 1.0,
    }
    single = [
        {
            **common, "example_id": f"ere-{index}", "family": "ERE",
            "copy_all_correct": True, "copy_all_margin": 0.9,
            "copy_retention": 0.9, "two_contributors": False,
        }
        for index in range(8)
    ]
    multi = [
        {
            **common, "example_id": f"cps-{index}", "family": "CPS",
            "copy_all_correct": False, "copy_all_margin": 0.2,
            "copy_retention": 0.2, "two_contributors": True,
        }
        for index in range(8)
    ]
    result = classify_b_axis(single + multi, thresholds)
    assert result["families"]["ERE"]["classification"] == "SINGLE_SLOT_COPY"
    assert result["families"]["CPS"]["classification"] == "MULTI_ADDRESS_SUPPORTED"


def test_b_axis_never_promotes_certificate_only_or_small_n() -> None:
    thresholds = {
        "task_causal_assessed_point_min": 0.75,
        "task_multi_object_causal_point_min": 0.75,
        "task_low_order_supported_point_min": 0.75,
        "baseline_eligible_point_min": 0.75,
        "minimum_eligible_records_per_family": 8,
        "copy_retention_point_min": 0.80,
        "copy_retention_wilson_lower_min": 0.65,
        "two_contributor_point_max": 0.50,
        "two_contributor_wilson_lower_min": 0.65,
        "copy_collapse_bootstrap_lower_min": 0.50,
        "bootstrap_seed": 19,
        "bootstrap_replicates": 50,
    }
    certificate_only = [
        {
            "example_id": f"ere-{index}", "family": "ERE",
            "certificate_multi": True, "task_causal_assessed": False,
            "task_multi_object_causal": False, "task_low_order_supported": False,
            "task_evidence_conflict": False, "baseline_eligible": True,
            "contributor_denominator_valid": True, "full_margin": 1.0,
            "copy_all_margin": 0.9, "copy_all_correct": True,
            "copy_retention": 0.9, "two_contributors": False,
        }
        for index in range(8)
    ]
    assert classify_b_axis(certificate_only, thresholds)["families"]["ERE"]["classification"] == "INCONCLUSIVE"

    small = [
        {
            **row, "example_id": f"cps-{index}", "family": "CPS",
            "task_causal_assessed": True, "task_multi_object_causal": True,
        }
        for index, row in enumerate(certificate_only[:7])
    ]
    result = classify_b_axis(small, thresholds)["families"]["CPS"]
    assert result["classification"] == "INCONCLUSIVE"
    assert result["reason"] == "insufficient_eligible_multi_object_records"


def test_c_axis_excludes_answer_derived_final_winner() -> None:
    thresholds = {
        "target_adequacy_point_min": 0.75,
        "latent_motion_point_min": 0.05,
        "latent_separation_auc_min": 0.70,
        "decoder_balanced_accuracy_min": 0.80,
        "readout_minus_null_min": 0.05,
        "static_null_gap_max": 0.05,
    }
    rows = [{"family": "CPS", "target_adequate": True, "relative_motion": 0.01, "separation_auc": 0.51, "crossfit_decoder_balanced_accuracy": 0.51, "frozen_decoder_balanced_accuracy": 0.51, "static_null_gap": 0.0, "final_winner": True} for _ in range(8)]
    result = classify_c_axis(rows, thresholds)
    assert result["families"]["CPS"]["classification"] == "LATENT_NOT_FORMED"
    assert "final_winner" not in result["families"]["CPS"]


def test_c_axis_requires_motion_and_fixed_channel_separation_for_readout_label() -> None:
    thresholds = {
        "target_adequacy_point_min": 0.75,
        "latent_motion_point_min": 0.05,
        "latent_separation_auc_min": 0.70,
        "decoder_balanced_accuracy_min": 0.80,
        "readout_minus_null_min": 0.05,
        "static_null_gap_max": 0.05,
    }
    rows = [
        {
            "family": "CPS",
            "target_adequate": True,
            "relative_motion": 0.01,
            "separation_auc": 0.55,
            "crossfit_decoder_balanced_accuracy": 0.90,
            "frozen_decoder_balanced_accuracy": 0.50,
            "static_null_gap": 0.10,
        }
        for _ in range(8)
    ]
    result = classify_c_axis(rows, thresholds)
    assert result["families"]["CPS"]["classification"] == "MIXED"


def test_loss_gate_alignment_flags_failed_gate_without_active_loss() -> None:
    gates = [
        {"family": "ERE", "gate": "state", "passed": False},
        {"family": "CPS", "gate": "state", "passed": False},
    ]
    losses = [{"family": "ERE", "gate": "state", "present": True, "active_fraction": 0.9, "gradient_norm": 1.0}]
    result = classify_loss_gate_alignment(gates, losses, {"min_active_fraction": 0.1, "min_gradient_norm": 1.0e-6})
    assert result["families"]["ERE"]["classification"] == "OBJECTIVE_ALIGNED"
    assert result["families"]["CPS"]["classification"] == "OBJECTIVE_GATE_MISMATCH_CONFIRMED"


def test_decision_requires_registered_thresholds() -> None:
    with pytest.raises(KeyError):
        classify_a_axis(_a_rows("ERE", 1, True, False), {})
