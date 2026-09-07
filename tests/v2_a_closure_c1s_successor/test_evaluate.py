from __future__ import annotations

from copy import deepcopy

import torch
import pytest

from yggdrasil_v2.v2_a.closure_c1s_successor.evaluate import (
    DYNAMIC_STATE_FEATURES,
    answer_metrics,
    causal_pair_margin_report,
    dynamic_state_qualified,
    evaluate_collated_batches,
    evaluate_predictions,
    gate_g006_causal_margin,
    gate_g008_hidden,
    operation_active_qualified,
    paired_eval_gain,
    paired_intervention_margin_report,
    s1_gate,
    s2_gate,
    s3_gate,
    validate_k1_single_slot_control,
)


STEPS = 3
SLOTS = 2
FEATURES = 8


def _feature_names(family: str) -> list[str]:
    if family == "ERE":
        return [
            "present", "touched", "changed", "query_owner",
            "operation_source", "operation_target", "query_semantic_match", "stable_after",
        ]
    return [
        "processed", "legal", "valid", "budget_ok", "goal_satisfied",
        "final_constraints_satisfied", "running_best", "final_winner",
    ]


def _temporal_state() -> list[list[list[bool]]]:
    return [
        [
            [bool((step + slot + feature) % 2) for feature in range(FEATURES)]
            for slot in range(SLOTS)
        ]
        for step in range(STEPS)
    ]


def _metadata(count: int = 4):
    rows = []
    for index in range(count):
        family = "ERE" if index % 2 == 0 else "CPS"
        answer = index % 2
        # Query ownership is a public semantic address, not the answer label.
        query_owner = 1 - answer
        state = _temporal_state()
        rows.append(
            {
                "example_id": f"e{index}",
                "family": family,
                "answer_index": answer,
                "query_owner": query_owner,
                "owner_slot": query_owner,
                "source_owner": [0, 1, 0],
                "target_owner": [1, 0, 1],
                "operation_active": [True, False, True],
                "same_value_group": "v0" if index < 2 else "v1",
                "state_values": state,
                "state_feature_mask": [
                    [[True] * FEATURES for _ in range(SLOTS)]
                    for _ in range(STEPS)
                ],
                "state_step_mask": [True] * STEPS,
                "state_feature_names": _feature_names(family),
                "mechanism_supported": True,
                "query_answer_independent": True,
            }
        )
    return rows


def _prediction():
    rows = _metadata()
    logits = torch.full((len(rows), 9), -4.0)
    disrupted = torch.full_like(logits, -4.0)
    for index, row in enumerate(rows):
        logits[index, row["answer_index"]] = 4.0
        disrupted[index, (row["answer_index"] + 1) % 9] = 4.0

    query = torch.zeros((len(rows), SLOTS))
    source = torch.zeros((len(rows), STEPS, SLOTS))
    target = torch.zeros((len(rows), STEPS, SLOTS))
    for index, row in enumerate(rows):
        query[index, row["query_owner"]] = 1.0
        for step in range(STEPS):
            source[index, step, row["source_owner"][step]] = 1.0
            target[index, step, row["target_owner"][step]] = 1.0

    state_target = torch.tensor(
        [row["state_values"] for row in rows], dtype=torch.bool
    )
    state_logits = torch.where(
        state_target,
        torch.full(state_target.shape, 2.0),
        torch.full(state_target.shape, -2.0),
    )
    return {
        "logits": logits,
        "query_weights": query,
        "source_weights": source,
        "target_weights": target,
        "operation_active": torch.tensor(
            [row["operation_active"] for row in rows], dtype=torch.float32
        ),
        "state_logits": state_logits,
        "initial_slot_margin_effects": torch.full((len(rows), SLOTS), 8.0),
        "content_mask": torch.ones((len(rows), SLOTS), dtype=torch.bool),
        "interventions": {
            "no_core_logits": disrupted,
            "wrong_start_logits": disrupted,
            "payload_zero_logits": disrupted,
            "payload_shuffle_logits": disrupted,
            "operation_zero_logits": disrupted,
            "operation_shuffle_logits": disrupted,
            "relevant_replace_logits": disrupted,
            "irrelevant_replace_logits": logits.clone(),
            "target_shuffle_logits": disrupted,
        },
        "permuted_logits": logits.clone(),
        "stripped_logits": logits.clone(),
    }


def test_evaluate_predictions_is_json_native_and_reports_temporal_metrics():
    report = evaluate_predictions(_prediction(), _metadata())
    assert report["answer"]["overall"]["point"] == 1.0
    assert report["answer"]["families"]["CPS"]["n"] == 2
    assert report["ownership"]["query"]["point"] == 1.0
    assert report["ownership"]["state_masked"]["point"] == 1.0
    assert report["ownership"]["state_masked"]["dynamic_coverage_pass"] is True
    assert dynamic_state_qualified(report, floor=0.95) is True
    assert report["interventions"]["no_core"]["answer_margin_drop"]["point"] > 0.5
    assert report["invariance"]["slot_permutation"]["passed"] is True
    assert report["target_contract"] == {
        "mechanism_supported": True,
        "query_answer_independent": True,
    }
    assert "query_swap" not in report["interventions"]


def test_answer_margin_drop_is_gauge_invariant_and_uniform_shift_cannot_pass_gate():
    prediction = _prediction()
    prediction["interventions"] = dict(prediction["interventions"])
    prediction["interventions"]["no_core_logits"] = prediction["logits"] - 7.0
    report = evaluate_predictions(prediction, _metadata())
    no_core = report["interventions"]["no_core"]

    assert abs(no_core["answer_margin_drop"]["point"]) <= 1.0e-6
    assert no_core["raw_correct_logit_drop_diagnostic_only"]["point"] == 7.0
    report["duplicate_payload_control"] = {"passed": True}
    report["s0_single_slot_structural_control_replay"] = {
        "status": "PASS_STRUCTURAL_INSTRUMENT_REPLAY",
        "passed": True,
        "learned_k1_control": False,
    }
    gate = s1_gate(report)
    assert gate["checks"]["no_core_drop"] is False
    assert gate["passed"] is False


def test_successor_paired_intervention_margin_is_gauge_invariant_and_seeded():
    rows = _metadata(4)
    baseline = _prediction()["logits"]
    intervention = baseline.clone()
    for index, row in enumerate(rows):
        intervention[index, row["answer_index"]] = -4.0
    report = paired_intervention_margin_report(
        baseline,
        intervention,
        rows,
        bootstrap_seed=1234,
        bootstrap_replicates=32,
    )
    shifted = paired_intervention_margin_report(
        baseline + 17.0,
        intervention + 17.0,
        rows,
        bootstrap_seed=1234,
        bootstrap_replicates=32,
    )
    assert report == shifted
    assert report["overall"]["answer_margin_drop"]["point"] > 0.5
    assert report["overall"]["answer_margin_drop"]["seed"] == 1234
    assert set(report["families"]) == {"ERE", "CPS"}


def test_semantic_causal_pair_margin_maps_both_directions_and_rejects_bad_pairs():
    rows = []
    logits = []
    for family in ("ERE", "CPS"):
        mapping = {"A": "semantic:zero", "B": "semantic:one"}
        for role, answer, semantic in (("base", 0, "semantic:zero"), ("flip", 1, "semantic:one")):
            rows.append(
                {
                    "example_id": f"{family}-{role}",
                    "family": family,
                    "pair_id": f"{family}-p",
                    "pair_role": role,
                    "answer": answer,
                    "semantic_answer": semantic,
                    "label_mapping": mapping,
                }
            )
            value = torch.full((9,), -4.0)
            value[answer] = 4.0
            logits.append(value)
    report = causal_pair_margin_report(torch.stack(logits), rows, bootstrap_seed=77, bootstrap_replicates=32)
    assert report["overall"]["pairs"] == 2
    assert report["overall"]["base_to_flip"]["point"] > 0.5
    assert report["overall"]["flip_to_base"]["point"] > 0.5
    assert report["families"]["ERE"]["base_to_flip"]["seed"] == 77
    assert gate_g006_causal_margin(report)["passed"] is True

    with pytest.raises(ValueError, match="semantic flip"):
        causal_pair_margin_report(
            torch.stack(logits[:2]),
            [{**rows[0], "semantic_answer": "same"}, {**rows[1], "semantic_answer": "same"}],
            bootstrap_replicates=8,
        )


def test_hidden_margin_gate_requires_each_family_and_never_uses_raw_accuracy():
    rows = _metadata(4)
    baseline = _prediction()["logits"]
    # Raw predictions remain correct, but a uniform gauge shift leaves margin
    # unchanged; a margin-only Gate must reject this despite raw diagnostics.
    intervention = baseline - 7.0
    report = {"zero_hidden": paired_intervention_margin_report(baseline, intervention, rows, bootstrap_replicates=16),
              "shuffled_hidden": paired_intervention_margin_report(baseline, intervention, rows, bootstrap_replicates=16)}
    assert report["zero_hidden"]["overall"]["raw_accuracy_drop_diagnostic_only"]["point"] == 0.0
    assert gate_g008_hidden(report)["passed"] is False


def test_dynamic_gate_uses_each_feature_balanced_accuracy_not_raw_cell_accuracy():
    steps, slots = 10, 8
    rows = []
    state_logits = torch.full((2, steps, slots, FEATURES), -2.0)
    for index, family in enumerate(("ERE", "CPS")):
        names = _feature_names(family)
        state = torch.zeros((steps, slots, FEATURES), dtype=torch.bool)
        for name in DYNAMIC_STATE_FEATURES[family]:
            state[1, 0, names.index(name)] = True
        rows.append(
            {
                "example_id": f"imbalanced-{family}",
                "family": family,
                "answer_index": index,
                "query_owner": 0,
                "state_values": state.tolist(),
                "state_feature_mask": torch.ones_like(state).tolist(),
                "state_step_mask": [True] * steps,
                "state_feature_names": names,
                "mechanism_supported": True,
                "query_answer_independent": True,
            }
        )
    logits = torch.full((2, 9), -2.0)
    logits[0, 0] = 2.0
    logits[1, 1] = 2.0
    report = evaluate_predictions(
        {
            "logits": logits,
            "query_weights": torch.tensor(
                [[1.0] + [0.0] * 7, [1.0] + [0.0] * 7]
            ),
            "state_logits": state_logits,
        },
        rows,
    )

    state = report["ownership"]["state_masked"]
    assert state["point"] > 0.95
    assert state["dynamic_coverage_pass"] is True
    assert dynamic_state_qualified(report, floor=0.95) is False
    for family, names in DYNAMIC_STATE_FEATURES.items():
        for name in names:
            assert state["families"][family][name]["balanced_accuracy"] == 0.5


def test_operation_active_gate_rejects_high_pooled_accuracy_constant_prediction():
    rows = []
    for index, family in enumerate(("ERE", "CPS")):
        rows.append(
            {
                "example_id": f"operation-{family}",
                "family": family,
                "answer_index": index,
                "query_owner": 0,
                "operation_active": [True] * 9 + [False],
                "mechanism_supported": True,
                "query_answer_independent": True,
            }
        )
    logits = torch.full((2, 9), -2.0)
    logits[0, 0] = 2.0
    logits[1, 1] = 2.0
    prediction = {
        "logits": logits,
        "query_weights": torch.ones((2, 1)),
        "operation_active": torch.ones((2, 10)),
    }
    report = evaluate_predictions(prediction, rows)

    assert report["ownership"]["operation_active"]["point"] == 0.9
    assert operation_active_qualified(report, floor=0.80) is False
    prediction["operation_active"] = torch.tensor(
        [row["operation_active"] for row in rows], dtype=torch.float32
    )
    perfect = evaluate_predictions(prediction, rows)
    assert operation_active_qualified(perfect, floor=0.95) is True


def test_answer_wilson_and_paired_gain_are_finite_and_family_scoped():
    metadata = _metadata()
    logits = _prediction()["logits"]
    answer = answer_metrics(logits, metadata)
    gain = paired_eval_gain(logits, logits - 4.0, metadata, bootstrap_replicates=32)
    assert 0.0 <= answer["overall"]["wilson_lower"] <= 1.0
    assert gain["gain"]["point"] == 0.0
    assert gain["gain"]["replicates"] == 32
    assert set(gain["families"]) == {"CPS", "ERE"}


def test_collated_batch_iterable_is_concatenated_without_model_inputs():
    prediction = _prediction()
    report = evaluate_collated_batches(
        [{"logits": prediction["logits"][:2]}, {"logits": prediction["logits"][2:]}],
        [_metadata()[:2], _metadata()[2:]],
    )
    assert report["answer"]["overall"]["n"] == 4
    assert report["answer"]["overall"]["point"] == 1.0


def test_collated_batches_model_adapter_forwards_only_public_source_tensors():
    class TinyModel:
        def __call__(self, source_hidden, source_mask, **_kwargs):
            assert source_hidden.ndim == 3
            assert source_mask.dtype is torch.bool
            return {"logits": source_hidden[:, 0, :]}

    metadata = _metadata()
    source = torch.full((4, 1, 9), -1.0)
    for index, row in enumerate(metadata):
        source[index, 0, row["answer_index"]] = 1.0
    batches = [
        {"source_hidden": source[:2], "source_mask": torch.ones((2, 1), dtype=torch.bool)},
        {"source_hidden": source[2:], "source_mask": torch.ones((2, 1), dtype=torch.bool)},
    ]
    report = evaluate_collated_batches(batches, metadata, model=TinyModel())
    assert report["answer"]["overall"]["point"] == 1.0


def test_k1_endpoint_is_valid_single_slot_control_without_multi_address_failure():
    rows = [
        {**row, "query_owner": 0, "owner_slot": 0}
        for row in deepcopy(_metadata())
    ]
    base = _prediction()
    prediction = {
        "logits": base["logits"],
        "query_weights": torch.ones((4, 1)),
        "interventions": {"no_core_logits": base["interventions"]["no_core_logits"]},
    }
    report = evaluate_predictions(prediction, rows)
    control = validate_k1_single_slot_control(
        prediction, report, expected_records=len(rows)
    )

    assert control["status"] == "VALID_SINGLE_SLOT_CONTROL"
    assert control["passed"] is True
    assert control["multi_address_applicable"] is False
    assert control["multi_address_qualified"] is None
    k8 = {**report, "paired_gain": {"gain": {}, "families": {}}}
    gate = s2_gate(k8, {"single_slot_control": control})
    assert gate["checks"]["k1_control_measured_and_valid"] is True


def test_s1_s2_and_s3_gates_fail_closed_and_stop_in_order():
    report = evaluate_predictions(_prediction(), _metadata())
    report["duplicate_payload_control"] = {"passed": True}
    report["s0_single_slot_structural_control_replay"] = {
        "status": "PASS_STRUCTURAL_INSTRUMENT_REPLAY",
        "passed": True,
        "learned_k1_control": False,
    }
    s1 = s1_gate(report)
    assert s1["checks"]["s0_single_slot_structural_control_replayed"] is True
    assert s1["passed"] is False  # four records are not the frozen 32
    assert s2_gate(
        {**report, "paired_gain": {"gain": {}, "families": {}}},
        {"single_slot_control": {"status": "INVALID_SINGLE_SLOT_CONTROL"}},
        budget={"k8_updates": 4608, "k1_updates": 4608, "k8_hours": 1.0, "k1_hours": 1.0},
    )["passed"] is False
    s3 = s3_gate({"answer": {"overall": {"point": 0.0, "wilson_lower": 0.0}}, "mechanism": {}})
    assert s3["gates"]["G004_validation_behavior"]["status"] == "FAIL"
    assert s3["gates"]["G004_OOD_behavior"]["status"] == "NOT_RUN"
