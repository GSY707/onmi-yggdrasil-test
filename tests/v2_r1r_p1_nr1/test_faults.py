from __future__ import annotations

from yggdrasil_v2.r1_revalidation.nr1.qualification import (
    DECISION_KILL_MIN,
    run_qualification_bundle,
)


def test_all_registered_faults_are_detected_and_decision_faults_killed() -> None:
    result = run_qualification_bundle()
    assert result["gates"]["N05_adversarial_metric_decision_power"] is True
    for row in result["fault_matrix"].values():
        assert row["state_detection_rate"] == 1.0
        assert row["metric_gate_passed"] is True
        for metric in row["required_metrics"]:
            assert row["metric_kill_rates"][metric] >= DECISION_KILL_MIN
        if row["decision_gate_required"]:
            assert row["decision_kill_rate"] >= DECISION_KILL_MIN


def test_full_bundle_passes_only_measurement_gates() -> None:
    result = run_qualification_bundle()
    assert result["passed"] is True
    assert result["training_performed"] is False
    assert result["audit_statistics_fit"] is False
    assert set(result["gates"]) == {
        "N02_schema_reference_fixture_independence",
        "N03_numeric_qualification",
        "N04_relation_topology_qualification",
        "N05_adversarial_metric_decision_power",
        "N06_holdout_non_leakage",
    }
