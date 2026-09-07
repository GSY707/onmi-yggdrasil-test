from copy import deepcopy

from yggdrasil_v2.r1_revalidation.h1.contract import (
    CALIBRATION_DATA_SEED,
    CALIBRATION_MODEL_SEED,
)
from yggdrasil_v2.r1_revalidation.h1.probe import (
    calibration_gate,
    calibration_threshold_proposal,
    direction_screen_gate,
    registered_full_probe_spec,
)


_TRANSFORMS = (
    "cancelling_pair",
    "choice_permutation",
    "clause_permutation",
    "handle_rename",
    "query_permutation",
    "surface_paraphrase",
    "transitive_redundancy",
)


def test_registered_probe_specs_are_full_budget_and_scope_bound() -> None:
    screen = registered_full_probe_spec("screen")
    calibration = registered_full_probe_spec("calibration")
    assert screen.counts() == calibration.counts()
    assert screen.updates == calibration.updates == 4000
    assert screen.batch_size == calibration.batch_size == 32
    assert screen.materialized_bases_per_family == 64
    assert (screen.data_seed, screen.model_seed) != (
        calibration.data_seed,
        calibration.model_seed,
    )


def _report() -> dict:
    return {
        "primary_comparison": {"gate": {"passed": True}},
        "mixed_intervention_drops": {
            name: {"answer": 0.08, "trace": 0.06}
            for name in (
                "flip_route",
                "force_route0",
                "force_route1",
                "swap_experts",
                "disable_routed_projection",
            )
        },
        "shared_route_noop": {"passed": True},
        "architecture": {"passed": True},
        "cache_audit": {"passed": True},
        "active_flops": {"matched": True},
        "shared": {"strip": {"passed": True}},
        "mixed": {"strip": {"passed": True}},
    }


def test_direction_screen_requires_primary_and_registered_route_causality() -> None:
    report = _report()
    failed = direction_screen_gate(
        report,
        route_causal_threshold=0.10,
        conditional_write_causal_threshold=0.05,
    )
    assert failed["passed"] is False
    assert failed["checks"]["primary_h05"] is True
    assert failed["checks"]["route_causal_h06"] is False
    assert failed["authorizes"] == "nothing"

    passing = deepcopy(report)
    passing["mixed_intervention_drops"]["swap_experts"]["trace"] = 0.11
    passed = direction_screen_gate(
        passing,
        route_causal_threshold=0.10,
        conditional_write_causal_threshold=0.05,
    )
    assert passed["passed"] is True
    assert passed["maximum_registered_route_effect"] == {
        "intervention": "swap_experts",
        "metric": "trace",
        "drop": 0.11,
    }
    assert passed["checks"]["conditional_write_h06"] is True
    assert passed["h1_qualified"] is False

    bypassable = deepcopy(passing)
    bypassable["mixed_intervention_drops"]["disable_routed_projection"] = {
        "answer": 0.04,
        "trace": 0.03,
    }
    failed = direction_screen_gate(
        bypassable,
        route_causal_threshold=0.10,
        conditional_write_causal_threshold=0.05,
    )
    assert failed["passed"] is False
    assert failed["checks"]["route_causal_h06"] is True
    assert failed["checks"]["conditional_write_h06"] is False


def _evaluation(answer: float, trace: float) -> dict:
    return {
        "answer": {
            "macro_accuracy": answer,
            "by_family": {
                "numeric": {"accuracy": answer - 0.02},
                "relation": {"accuracy": answer + 0.02},
            },
        },
        "trace": {
            "cell_accuracy": trace,
            "by_family": {
                "numeric": {"cell_accuracy": trace - 0.02},
                "relation": {"cell_accuracy": trace + 0.02},
            },
        },
        "route": {"accuracy": 1.0},
    }


def _metamorphic(row: dict[str, float]) -> dict:
    return {
        "aggregate": dict(row),
        "by_transform": {name: dict(row) for name in _TRANSFORMS},
        "by_family": {
            "numeric": dict(row),
            "relation": dict(row),
        },
    }


def _calibration_report() -> dict:
    report = _report()
    report.update(
        {
            "probe_spec": {
                "data_seed": CALIBRATION_DATA_SEED,
                "model_seed": CALIBRATION_MODEL_SEED,
                "train_per_family": 2048,
                "validation_per_family": 256,
                "supported_per_family": 512,
                "heldout_per_family": 512,
                "causal_per_family": 256,
                "materialized_bases_per_family": 64,
                "updates": 4000,
                "batch_size": 32,
            },
            "data_registration": {
                "scope": "calibration",
                "registered": True,
                "seed": CALIBRATION_DATA_SEED,
                "package_identity": "FIXTURE",
            },
            "status": "NONFORMAL_P1_H1_PROBE_ONLY",
            "cache_origin": "built_for_this_probe",
            "claims": {
                "h1_qualified": False,
                "p1_f1_design_authorized": False,
                "p1_completed": False,
                "p2_eligible": False,
                "p2_started": False,
            },
        }
    )
    report["shared"].update(
        {
            "evaluations": {
                "supported": _evaluation(0.67, 0.77),
                "heldout": _evaluation(0.57, 0.72),
            }
        }
    )
    report["mixed"].update(
        {
            "evaluations": {
                "supported": _evaluation(0.72, 0.82),
                "heldout": _evaluation(0.67, 0.79),
            },
            "metamorphic_consistency": _metamorphic(
                {"relation_consistency": 0.72, "transformed_exact": 0.74}
            ),
            "metamorphic_route_consistency": _metamorphic(
                {"consistency": 1.0, "transformed_exact": 1.0}
            ),
        }
    )
    report["mixed_intervention_drops"].update(
        {
            "zero_source": {"answer": 0.30, "trace": 0.25},
            "shuffle_source": {"answer": 0.28, "trace": 0.24},
            "disable_recurrence": {"answer": 0.35, "trace": 0.31},
        }
    )
    report["direction_screen_gate"] = {"passed": True}
    return report


def test_calibration_uses_pre_registered_worst_cell_margin_rule() -> None:
    report = _calibration_report()
    proposal = calibration_threshold_proposal(report)
    assert proposal["passed"] is True
    assert proposal["thresholds"] == {
        "supported_answer_floor": 0.60,
        "supported_trace_floor": 0.70,
        "heldout_shared_answer_floor": 0.50,
        "heldout_mixed_answer_floor": 0.60,
        "heldout_shared_trace_floor": 0.65,
        "heldout_mixed_trace_floor": 0.72,
        "source_causal_drop": 0.23,
        "recurrence_causal_drop": 0.30,
        "metamorphic_consistency": 0.67,
        "route_accuracy": 0.99,
        "primary_gain": 0.05,
        "family_regression_limit": 0.02,
        "route_causal_drop": 0.10,
        "conditional_write_causal_drop": 0.05,
    }
    gate = calibration_gate(report)
    assert gate["passed"] is True
    assert gate["authorizes"] == "contract-freeze audit only"
    assert gate["h1_qualified"] is False

    report["direction_screen_gate"]["passed"] = False
    assert calibration_gate(report)["passed"] is False
