from __future__ import annotations

from copy import deepcopy
import random

import pytest

from yggdrasil_v2.r1_revalidation.cps import (
    generate_cps_causal_pairs,
    _base_ast,
)
from yggdrasil_v2.r1_revalidation.ere import (
    _build_ast,
    _record_from_ast,
    generate_ere_causal_pairs,
)
from yggdrasil_v2.r1_revalidation.render import (
    OOD_CPS_TEMPLATES,
    OOD_ERE_TEMPLATES,
    TRAIN_CPS_TEMPLATES,
    TRAIN_ERE_TEMPLATES,
    render_ere,
    semantic_fingerprint,
)
from yggdrasil_v2.r1_revalidation.schema import assert_model_view, make_model_view
from yggdrasil_v2.r1_revalidation.simulator import (
    TypedState,
    execute_ere_rule,
    select_cps_answer,
    verify_cps_plan,
)
from yggdrasil_v2.r1_revalidation.symbols import label_permutation


def test_ere_swap_if_and_foreach_read_the_current_state() -> None:
    state = TypedState(
        attributes={"left": {"tone": "red"}, "right": {"tone": "blue"}, "neighbor": {"tone": "green"}},
        relations={"edge": set()},
    )
    execute_ere_rule(
        state,
        {"params": ["left", "right"], "primitives": [{"op": "SWAP", "left": "$arg:left", "right": "$arg:right", "attribute": "tone"}]},
        {"left": "left", "right": "right"},
    )
    assert state.attributes["left"]["tone"] == "blue"
    assert state.attributes["right"]["tone"] == "red"

    execute_ere_rule(
        state,
        {
            "params": ["source", "target"],
            "primitives": [
                {"op": "SET", "target": "$arg:source", "attribute": "tone", "value": "amber"},
                {
                    "op": "IF",
                    "predicate": {"kind": "attr_equals", "entity": "$arg:source", "attribute": "tone", "value": "amber"},
                    "then": {"op": "SET", "target": "$arg:target", "attribute": "tone", "value": "current"},
                    "else": {"op": "SET", "target": "$arg:target", "attribute": "tone", "value": "stale"},
                },
            ],
        },
        {"source": "left", "target": "right"},
    )
    assert state.attributes["right"]["tone"] == "current"

    state.attributes["left"]["tone"] = "violet"
    execute_ere_rule(
        state,
        {
            "params": ["source", "target"],
            "primitives": [
                {"op": "LINK", "relation": "edge", "source": "$arg:source", "target": "$arg:target"},
                {
                    "op": "FOREACH_LINKED",
                    "relation": "edge",
                    "source": "$arg:source",
                    "effect": {"op": "COPY", "source": "$arg:source", "target": "$neighbor", "attribute": "tone"},
                },
            ],
        },
        {"source": "left", "target": "neighbor"},
    )
    assert ("left", "neighbor") in state.relations["edge"]
    assert state.attributes["neighbor"]["tone"] == "violet"


def _small_cps_ast() -> dict:
    return {
        "kind": "cps",
        "initial_state": {"attributes": {}, "relations": {}, "facts": ["ready", "final"], "resources": {"energy": 1}},
        "actions": [
            {"name": "unlock", "preconditions": [{"kind": "fact_true", "fact": "ready"}], "effects": [{"kind": "add_fact", "fact": "unlocked"}], "cost": 1},
            {"name": "refill", "preconditions": [{"kind": "fact_true", "fact": "unlocked"}], "effects": [{"kind": "resource_delta", "resource": "energy", "delta": 2}], "cost": 3},
            {"name": "finish", "preconditions": [{"kind": "fact_true", "fact": "unlocked"}, {"kind": "resource_at_least", "resource": "energy", "amount": 2}], "effects": [{"kind": "resource_delta", "resource": "energy", "delta": -2}, {"kind": "add_fact", "fact": "goal"}], "cost": 4},
            {"name": "slow", "preconditions": [{"kind": "fact_true", "fact": "unlocked"}, {"kind": "resource_at_least", "resource": "energy", "amount": 2}], "effects": [{"kind": "resource_delta", "resource": "energy", "delta": -2}, {"kind": "add_fact", "fact": "goal"}], "cost": 6},
            {"name": "break", "preconditions": [{"kind": "fact_true", "fact": "unlocked"}, {"kind": "resource_at_least", "resource": "energy", "amount": 2}], "effects": [{"kind": "resource_delta", "resource": "energy", "delta": -2}, {"kind": "add_fact", "fact": "goal"}, {"kind": "remove_fact", "fact": "final"}], "cost": 2},
        ],
        "goal": [{"kind": "fact_true", "fact": "goal"}],
        "final_constraints": [{"kind": "fact_true", "fact": "final"}],
        "budget": 20,
        "candidates": [
            {"plan": ["unlock", "refill", "finish"]},
            {"plan": ["unlock", "refill", "slow"]},
            {"plan": ["unlock", "refill", "break"]},
            {"plan": ["refill", "unlock", "finish"]},
            {"plan": ["unlock", "finish"]},
        ],
    }


def test_cps_verifier_covers_precondition_resource_budget_final_and_none() -> None:
    ast = _small_cps_ast()
    assert verify_cps_plan(ast, ["refill", "unlock", "finish"])["failure"] == "precondition"
    assert verify_cps_plan(ast, ["unlock", "finish"])["failure"] == "precondition"
    budget_ast = deepcopy(ast)
    budget_ast["budget"] = 7
    assert verify_cps_plan(budget_ast, ["unlock", "refill", "finish"])["failure"] == "budget"
    assert verify_cps_plan(ast, ["unlock", "refill", "break"])["failure"] == "final_constraint"
    selected = select_cps_answer(ast)
    assert selected["answer"] == 0
    assert selected["unique_optimum"] is True

    none_ast = deepcopy(ast)
    none_ast["candidates"] = [{"plan": ["refill"]}, {"plan": ["unlock"]}, {"plan": ["unlock", "finish"]}]
    none = select_cps_answer(none_ast)
    assert none["answer"] is None
    assert none["unique_optimum"] is True


def _diff_paths(left, right, path=()):
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        return sum((_diff_paths(left[key], right[key], path + (key,)) for key in sorted(set(left) | set(right)) if key in left and key in right), []) + [path + (key,) for key in sorted(set(left) ^ set(right))]
    if isinstance(left, list):
        return sum((_diff_paths(left[index], right[index], path + (index,)) for index in range(min(len(left), len(right)))), []) + [path + (index,) for index in range(min(len(left), len(right)), max(len(left), len(right)))]
    return [] if left == right else [path]


def test_causal_pairs_change_one_field_and_flip_answer() -> None:
    kwargs = {"design_doc_sha256": "d" * 64, "command": "pytest", "environment": {}}
    ere_rows = generate_ere_causal_pairs(pair_count=2, seed=11, seen_semantic=set(), seen_surface=set(), **kwargs)
    cps_rows = generate_cps_causal_pairs(pair_count=2, seed=17, seen_semantic=set(), seen_surface=set(), **kwargs)
    for rows in (ere_rows, cps_rows):
        for base, flip in zip(rows[::2], rows[1::2]):
            assert base["pair_id"] == flip["pair_id"]
            assert base["pair_role"] == "base" and flip["pair_role"] == "flip"
            assert len(_diff_paths(base["program_ast"], flip["program_ast"])) == 1
            assert base["audit_answer_semantic"] != flip["audit_answer_semantic"]
            assert base["answer_index"] != flip["answer_index"]
            assert base["audit_label_mapping"] == flip["audit_label_mapping"]
            assert base["valid_choice_mask"] == flip["valid_choice_mask"]


def _rename(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_rename(item, mapping) for item in value]
    if isinstance(value, dict):
        return {_rename(key, mapping): _rename(child, mapping) for key, child in value.items()}
    return value


def test_fingerprint_ignores_alpha_names_paraphrase_and_candidate_order() -> None:
    ere_ast = _build_ast(23, "validation", 0)
    names = set(ere_ast["entities"] + [item["name"] for item in ere_ast["attributes"]] + [value for item in ere_ast["attributes"] for value in item["values"]] + ere_ast["relations"] + list(ere_ast["rules"]))
    renamed = {name: f"renamed_{index}" for index, name in enumerate(sorted(names))}
    assert semantic_fingerprint(ere_ast) == semantic_fingerprint(_rename(ere_ast, renamed))
    mapping = label_permutation(ere_ast["attributes"][0]["values"], random.Random(4))
    assert semantic_fingerprint(ere_ast) == semantic_fingerprint(ere_ast)
    assert render_ere(ere_ast, mapping, "ere_train_a") != render_ere(ere_ast, mapping, "ere_train_b")

    cps_ast = _base_ast(31, "validation", 0)
    reordered = deepcopy(cps_ast)
    reordered["candidates"] = list(reversed(reordered["candidates"]))
    assert semantic_fingerprint(cps_ast) == semantic_fingerprint(reordered)


def test_language_ood_templates_are_disjoint() -> None:
    assert TRAIN_ERE_TEMPLATES.isdisjoint(OOD_ERE_TEMPLATES)
    assert TRAIN_CPS_TEMPLATES.isdisjoint(OOD_CPS_TEMPLATES)


def test_model_view_rejects_forbidden_fields() -> None:
    record = {"reasoning_budget": 4, "valid_choice_mask": [True] * 9, "answer_index": 0, "program_ast": {}}
    view = make_model_view(record)
    assert set(view) == {"reasoning_budget", "valid_choice_mask"}
    assert_model_view(view)
    with pytest.raises(ValueError, match="outside the forward contract"):
        assert_model_view({"reasoning_budget": 4, "valid_choice_mask": [True] * 9, "answer_index": 0})
    with pytest.raises(ValueError, match="forbidden"):
        assert_model_view({"reasoning_budget": 4, "valid_choice_mask": [True] * 9, "source_hidden": {"answer": 0}})


def test_generator_rejects_overlong_record_without_truncation() -> None:
    ast = _build_ast(41, "length_ood", 0)
    distractor_rule = next(
        name
        for name, rule in ast["rules"].items()
        if rule["primitives"][0]["op"] == "SET" and rule["primitives"][0]["attribute"] == ast["attributes"][1]["name"]
    )
    target = ast["entities"][0]
    ast["events"].extend({"rule": distractor_rule, "arguments": {"target": target}} for _ in range(500))
    with pytest.raises(ValueError, match="over 1024 tokens"):
        _record_from_ast(
            ast,
            seed=41,
            split="length_ood",
            index=0,
            example_id="too-long",
            template_id="ere_train_a",
            design_doc_sha256="d" * 64,
            command="pytest",
            environment={},
        )
