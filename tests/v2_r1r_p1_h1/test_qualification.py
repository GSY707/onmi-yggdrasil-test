from dataclasses import replace

from yggdrasil_v2.r1_revalidation.h1.contract import THRESHOLDS
from yggdrasil_v2.r1_revalidation.h1.qualification import aggregate_h1_qualification


def _thresholds():
    return replace(
        THRESHOLDS,
        supported_answer_floor=0.0,
        supported_trace_floor=0.0,
        heldout_shared_answer_floor=0.0,
        heldout_mixed_answer_floor=0.0,
        heldout_shared_trace_floor=0.0,
        heldout_mixed_trace_floor=0.0,
        primary_gain=0.0,
        family_regression_limit=0.02,
        route_causal_drop=0.0,
        conditional_write_causal_drop=0.0,
        source_causal_drop=0.0,
        recurrence_causal_drop=0.0,
        metamorphic_consistency=0.0,
        route_accuracy=0.0,
    )


def _evaluation(ids, targets, families, predictions, *, route=True):
    return {
        "ids": ids,
        "targets": targets,
        "families": families,
        "predictions": predictions,
        "answer": {
            "macro_accuracy": sum(a == b for a, b in zip(predictions, targets)) / len(ids),
            "correct_vector": [a == b for a, b in zip(predictions, targets)],
            "by_family": {
                "numeric": {"accuracy": float(predictions[0] == targets[0])},
                "relation": {"accuracy": float(predictions[1] == targets[1])},
            },
        },
        "trace": {
            "cell_accuracy": 1.0,
            "by_family": {
                "numeric": {"cell_accuracy": 1.0},
                "relation": {"cell_accuracy": 1.0},
            },
        },
        "trace_predictions": [[0] * 8 for _ in ids],
        "route": {
            "accuracy": 1.0,
            "diagnostics": {
                "all_cells_use_both_experts": route,
                "minimum_expert_load": 0.5,
                "minimum_entropy_bits": 1.0,
            },
        },
    }


def _run(data_seed, model_seed, *, mixed_good=True):
    ids = [f"{data_seed}-n", f"{data_seed}-r"]
    targets = [1, 2]
    families = ["numeric", "relation"]
    shared_eval = _evaluation(ids, targets, families, [0, 0])
    mixed_eval = _evaluation(ids, targets, families, targets if mixed_good else [0, 0])
    integrity = {
        "passed": True,
        "source_independent_initial_state": True,
        "common_attention_and_ffn_state_write": True,
        "routed_feature_trunk_shared_across_routes": True,
        "selected_routed_projection_is_only_conditional_state_write": True,
        "route_controls_final_projection_only": True,
        "route_granularity": "record",
        "conditional_transition_scope": "factorized-routed-state-write-projection",
        "routed_feature_width": 384,
        "routed_projection_input_width": 384,
        "routed_projection_output_width": 256,
        "answer_reads_pooled_final_state_only": True,
        "route_assignment_is_hard_top1": True,
    }
    arm = lambda evaluation: {
        "integrity": integrity,
        "training": {
            "updates": 4000,
            "examples_seen": 128000,
            "final_losses": {"total": 1.0},
            "history": [{"losses": {"total": 1.0}}],
        },
        "evaluations": {
            "supported": evaluation,
            "heldout": evaluation,
        },
        "strip": {
            "passed": True,
            "reload_predictions_equal": True,
        },
    }
    transforms = (
        "cancelling_pair",
        "choice_permutation",
        "clause_permutation",
        "handle_rename",
        "query_permutation",
        "surface_paraphrase",
        "transitive_redundancy",
    )
    answer_summary = {"relation_consistency": 1.0, "transformed_exact": 1.0}
    route_summary = {"consistency": 1.0, "transformed_exact": 1.0}
    return {
        "data_seed": data_seed,
        "model_seed": model_seed,
        "data": {
            "passed": True,
            "split_counts": {"train": 4096, "validation": 512, "supported": 1024, "heldout": 1024, "causal": 512},
            "semantic_overlap": [],
            "source_overlap": [],
            "forbidden_field_hits": [],
            "forbidden_source_scan": [],
        },
        "cache": {
            "passed": True,
            "checks": {"schema": True, "hashes": True, "shapes": True},
            "forbidden_field_hits": [],
        },
        "paired_inputs_passed": True,
        "architecture_passed": True,
        "runtime_probes_passed": True,
        "initialization": {"passed": True},
        "shared": arm(shared_eval),
        "mixed": arm(mixed_eval),
        "shared_noop": {"passed": True},
        "mixed_causal_drops": {
            "flip_route": {"answer": 0.2, "trace": 0.0},
            "force_route0": {"answer": 0.2, "trace": 0.0},
            "force_route1": {"answer": 0.2, "trace": 0.0},
            "swap_experts": {"answer": 0.2, "trace": 0.0},
            "disable_routed_projection": {"answer": 0.2, "trace": 0.0},
        },
        "route_metamorphic": {
            "aggregate": route_summary,
            "by_transform": {name: route_summary for name in transforms},
            "by_family": {name: route_summary for name in ("numeric", "relation")},
        },
        "source_causal_drops": {
            "zero_source.answer": 0.2,
            "zero_source.trace": 0.0,
            "shuffle_source.answer": 0.2,
            "shuffle_source.trace": 0.0,
        },
        "recurrence_causal_drop": {
            "disable_recurrence.answer": 0.2,
            "disable_recurrence.trace": 0.0,
        },
        "metamorphic": {
            "aggregate": answer_summary,
            "by_transform": {name: answer_summary for name in transforms},
            "by_family": {name: answer_summary for name in ("numeric", "relation")},
        },
        "budget": {"updates": 4000, "batch_size": 32, "examples_seen": 128000},
        "schedule": {"passed": True, "shared_sha256": "same", "mixed_sha256": "same"},
        "active_flops": {"shared": 100.0, "mixed": 100.0},
        "runner": {
            "preflight_seal_verified": True,
            "source_identity_stable": True,
            "git_identity_stable": True,
            "later_roots_absent": True,
            "process_chain_unique": True,
        },
        "runtime": {
            "updates_per_second": 1.0,
            "examples_per_second": 32.0,
            "dispatch_overhead": 0.1,
        },
    }


def _context():
    return {
        "thresholds_frozen": True,
        "contract_hashes_frozen": True,
        "prior_evidence_verified": True,
        "roots_fresh": True,
        "process_chain_unique": True,
        "no_successor_roots": True,
        "command_exact": True,
    }


def test_all_gates_pass_and_status_is_limited_to_h1_authorization():
    runs = [_run(1, 11), _run(2, 12), _run(3, 13)]
    result = aggregate_h1_qualification(
        runs,
        context=_context(),
        thresholds=_thresholds(),
        expected_pairs=((1, 11), (2, 12), (3, 13)),
    )
    assert result["passed"]
    assert all(result["gates"][name]["passed"] for name in ("H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08"))
    assert result["status"] == "PASS_P1_H1_MIXED_CORE_DEVELOPMENT"
    assert result["claims"] == {
        "p1_f1_design_authorized": True,
        "p1_completed": False,
        "p2_eligible": False,
        "p2_started": False,
    }
    assert result["gates"]["H05"]["pooled"]["gain"] == 1.0


def test_missing_seed_and_unfrozen_thresholds_fail_closed():
    result = aggregate_h1_qualification(
        [_run(1, 11)],
        context=_context(),
        thresholds=THRESHOLDS,
        expected_pairs=((1, 11), (2, 12), (3, 13)),
    )
    assert result["passed"] is False
    assert result["status"] == "FAIL_P1_H1_MIXED_CORE_DEVELOPMENT"
    assert result["claims"]["p1_f1_design_authorized"] is False
    assert result["threshold_errors"]
    assert result["gates"]["H01"]["passed"] is False


def test_pooled_primary_gain_fails_when_one_seed_reverses_direction():
    runs = [_run(1, 11), _run(2, 12), _run(3, 13, mixed_good=False)]
    result = aggregate_h1_qualification(
        runs,
        context=_context(),
        thresholds=_thresholds(),
        expected_pairs=((1, 11), (2, 12), (3, 13)),
    )
    assert result["passed"] is False
    assert result["gates"]["H05"]["passed"] is False
    assert "three_seed_positive" in result["gates"]["H05"]["failures"]


def test_h06_rejects_bypassable_conditional_write():
    runs = [_run(1, 11), _run(2, 12), _run(3, 13)]
    for run in runs:
        run["mixed_causal_drops"]["disable_routed_projection"] = {
            "answer": 0.0,
            "trace": 0.0,
        }
    thresholds = replace(
        _thresholds(),
        conditional_write_causal_drop=0.05,
    )
    result = aggregate_h1_qualification(
        runs,
        context=_context(),
        thresholds=thresholds,
        expected_pairs=((1, 11), (2, 12), (3, 13)),
    )
    assert result["passed"] is False
    assert result["gates"]["H06"]["passed"] is False
    assert result["gates"]["H06"]["failures"] == ["1/11", "2/12", "3/13"]
    assert all(
        "conditional_write_causal_drop" in row["failures"]
        for row in result["gates"]["H06"]["by_seed"].values()
    )
