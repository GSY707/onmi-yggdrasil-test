from copy import deepcopy

import pytest

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis import task_arity as module


def _record(family="CPS"):
    if family == "CPS":
        objects = [
            {"slot": 0, "semantic_slot": 0, "name": "candidate:0", "kind": "cps_candidate"},
            {"slot": 1, "semantic_slot": 1, "name": "candidate:1", "kind": "cps_candidate"},
            {"slot": 2, "semantic_slot": 2, "name": "decision", "kind": "cps_decision"},
        ]
        schedule = [
            {"step": 0, "active": True, "source_slot": 0, "target_slot": 2},
            {"step": 1, "active": True, "source_slot": 1, "target_slot": 2},
        ]
        ast = {
            "initial_state": {}, "actions": [], "candidates": [
                {"plan": ["winner"]}, {"plan": ["other"]},
            ], "budget": 1, "goal": [], "final_constraints": [],
        }
        return {"example_id": "cps-1", "family": family, "objects": objects,
                "schedule": schedule, "query_owner": 2}, ast
    objects = [{"slot": 0, "kind": "ere_entity"}, {"slot": 1, "kind": "ere_entity"}]
    schedule = [{"step": 0, "active": True, "source_slot": 0, "target_slot": 1}]
    return {"example_id": "ere-1", "family": family, "objects": objects,
            "schedule": schedule, "query_owner": 1}, None


def test_certificate_is_reverse_reachability_and_cps_dimensions_are_separate(monkeypatch):
    record, ast = _record()

    def fake_evaluate(value):
        candidates = value["candidates"]
        answer = next((i for i, candidate in enumerate(candidates)
                       if candidate["plan"] == ["winner"]), None)
        return {"answer": answer, "unique_optimum": answer is not None, "candidates": []}

    monkeypatch.setattr(module, "evaluate_cps", fake_evaluate)
    result = module.assess_task_arity(record, {"program_ast": ast})

    assert result["valid"] is True
    assert result["registered_execution"]["status"] == module.ASSESSED
    assert result["registered_execution"]["support_slots"] == [0, 1, 2]
    assert result["certificate"]["certificate"] is True
    assert result["answer_support"]["status"] == module.ASSESSED
    assert result["counterfactual_influence"]["status"] == module.ASSESSED
    # Winner identity is tied to the original plan fingerprint, not subset index.
    assert result["counterfactual_influence"]["leave_one_out"]["0"]["winner_changed"] is True
    assert result["counterfactual_influence"]["leave_one_out"]["1"]["winner_changed"] is False


def test_cps_ast_candidate_indices_map_through_permuted_model_slots(monkeypatch):
    record, ast = _record()
    record["objects"] = [
        {"slot": 0, "semantic_slot": 1, "name": "candidate:1", "kind": "cps_candidate"},
        {"slot": 1, "semantic_slot": 2, "name": "decision", "kind": "cps_decision"},
        {"slot": 2, "semantic_slot": 0, "name": "candidate:0", "kind": "cps_candidate"},
    ]
    record["query_owner"] = 1
    record["schedule"] = [
        {"step": 0, "active": True, "source_slot": 2, "target_slot": 1},
        {"step": 1, "active": True, "source_slot": 0, "target_slot": 1},
    ]

    def fake_evaluate(value):
        answer = next(
            (index for index, candidate in enumerate(value["candidates"])
             if candidate["plan"] == ["winner"]),
            None,
        )
        return {"answer": answer, "unique_optimum": answer is not None, "candidates": []}

    monkeypatch.setattr(module, "evaluate_cps", fake_evaluate)
    result = module.assess_task_arity(record, {"program_ast": ast})

    assert result["candidate_slots"] == [2, 0]
    assert result["answer_support"]["baseline_winner_slot"] == 2
    assert result["counterfactual_influence"]["leave_one_out"]["2"]["removed_baseline_winner"] is True


def test_ere_counterfactuals_are_explicitly_not_assessable():
    record, source = _record("ERE")
    result = module.assess_task_arity(record, source)

    assert result["valid"] is True
    assert result["certificate"]["certificate"] is True
    assert result["answer_support"]["status"] == module.NOT_ASSESSABLE
    assert result["counterfactual_influence"]["status"] == module.NOT_ASSESSABLE


@pytest.mark.parametrize("bad_patch", [
    {"query_owner": 99},
    {"schedule": [{"active": True, "source_slot": 0, "target_slot": 99}]},
    {"schedule": [{"active": False, "source_slot": 0, "target_slot": 1}]},
])
def test_structural_invalid_is_fail_closed(bad_patch):
    record, ast = _record()
    record.update(bad_patch)
    result = module.assess_task_arity(record, {"program_ast": ast})

    assert result["valid"] is False
    assert result["certificate"]["status"] == module.INVALID
    assert result["answer_support"]["status"] == module.INVALID
    assert result["counterfactual_influence"]["status"] == module.INVALID


def test_duplicate_plan_fingerprint_does_not_fake_winner_identity(monkeypatch):
    record, ast = _record()
    duplicate = deepcopy(ast)
    duplicate["candidates"][1]["plan"] = ["winner"]
    monkeypatch.setattr(module, "evaluate_cps", lambda value: {
        "answer": 0, "unique_optimum": True, "candidates": [],
    })

    result = module.assess_task_arity(record, {"program_ast": duplicate})

    assert result["valid"] is False
    assert result["answer_support"]["status"] == module.INVALID
    assert result["counterfactual_influence"]["status"] == module.INVALID


def test_missing_cps_ast_is_not_assessable_but_certificate_survives():
    record, _ = _record()
    result = module.assess_task_arity(record)

    assert result["valid"] is True
    assert result["certificate"]["status"] == module.ASSESSED
    assert result["answer_support"]["status"] == module.NOT_ASSESSABLE
    assert result["counterfactual_influence"]["status"] == module.NOT_ASSESSABLE


def test_none_and_tie_subsets_are_reported_and_never_counted_as_winners(monkeypatch):
    record, ast = _record()

    def fake_evaluate(value):
        candidates = value["candidates"]
        if not candidates:
            return {"answer": None, "unique_optimum": False, "candidates": []}
        return {
            "answer": None,
            "unique_optimum": False,
            "candidates": [{"valid": True} for _ in candidates],
        }

    monkeypatch.setattr(module, "evaluate_cps", fake_evaluate)
    result = module.assess_task_arity(record, {"program_ast": ast})

    assert result["answer_support"]["status"] == module.NOT_ASSESSABLE
    assert result["answer_support"]["baseline_answer_kind"] == "tie"
    assert result["answer_support"]["none_count"] == 1
    assert result["answer_support"]["tie_count"] == 3
    assert result["answer_support"]["invalid_masks"] == []
