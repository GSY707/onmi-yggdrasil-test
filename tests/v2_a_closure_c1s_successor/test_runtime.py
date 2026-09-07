from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1s_successor.runtime import (
    C1SSuccessorRuntime,
    forward_inputs,
    ordered_prefetch,
)


ERE_STATE_NAMES = [
    "present", "touched", "changed", "query_owner",
    "operation_source", "operation_target", "query_semantic_match", "stable_after",
]
CPS_STATE_NAMES = [
    "processed", "legal", "valid", "budget_ok", "goal_satisfied",
    "final_constraints_satisfied", "running_best", "final_winner",
]


def _temporal_targets(object_count: int) -> tuple[list, list]:
    values = []
    masks = []
    for step in range(10):
        step_values = [[0] * 8 for _ in range(8)]
        step_masks = [[0] * 8 for _ in range(8)]
        if step < 2:
            for slot in range(object_count):
                step_values[slot] = [
                    1, int(step == slot), int(step == slot), int(slot == 1),
                    int(slot == 0), int(slot == 1), 0, 1,
                ]
                step_masks[slot] = [1] * 8
        values.append(step_values)
        masks.append(step_masks)
    return values, masks


def _target_row(example_id: str, *, answer_index: int = 1) -> dict:
    objects = [
        {
            "slot": 0,
            "kind": "ere_entity",
            "name": "alice",
            "content": True,
            "raw_label_index": -1,
            "token_start": 0,
            "token_end": 1,
        },
        {
            "slot": 1,
            "kind": "ere_entity",
            "name": "bob",
            "content": True,
            "raw_label_index": -1,
            "token_start": 1,
            "token_end": 2,
        },
    ]
    active = [1, 1] + [0] * 8
    source_owner = [1, 0] + [-1] * 8
    target_owner = [0, 1] + [-1] * 8
    schedule = [
        {"active": True, "source_slot": 1, "target_slot": 0},
        {"active": True, "source_slot": 0, "target_slot": 1},
    ] + [{"active": False, "source_slot": -1, "target_slot": -1} for _ in range(8)]
    state_values, state_masks = _temporal_targets(2)
    return {
        "example_id": example_id,
        "family": "ERE",
        "split": "train",
        "pair_id": None,
        "pair_role": None,
        "objects": objects,
        "presence": [1, 1, 0, 0, 0, 0, 0, 0],
        "content_mask": [1, 1, 0, 0, 0, 0, 0, 0],
        "state_values": state_values,
        "state_feature_mask": state_masks,
        "state_step_mask": active,
        "state_feature_names": ERE_STATE_NAMES,
        "schedule": schedule,
        "operation_active": active,
        "source_owner": source_owner,
        "target_owner": target_owner,
        "query": {
            "owner_slot": 1,
            "query_owner": 1,
            "alternate_owner": 0,
            "alternate_label": None,
            "irrelevant_owner": 2,
        },
        "query_owner": 1,
        "alternate_owner": 0,
        "alternate_label": None,
        "irrelevant_owner": 2,
        "mechanism_supported": True,
        "query_answer_independent": True,
        "answer_target": {"answer_label": "B", "answer_index": answer_index},
        "answer_index": answer_index,
    }


def _unsupported_ood_row(example_id: str) -> dict:
    objects = []
    for slot in range(9):
        decision = slot == 8
        objects.append({
            "slot": slot,
            "kind": "cps_decision" if decision else "cps_candidate",
            "name": "decision" if decision else f"candidate:{slot}",
            "content": not decision,
            "raw_label_index": -1 if decision else slot,
            "token_start": slot,
            "token_end": slot + 1,
        })
    zeros = [[[0] * 8 for _ in range(8)] for _ in range(10)]
    schedule = [{"active": False, "source_slot": -1, "target_slot": -1} for _ in range(10)]
    return {
        "example_id": example_id,
        "family": "CPS",
        "split": "distractor_ood",
        "pair_id": None,
        "pair_role": None,
        "objects": objects,
        "presence": [1] * 8,
        "content_mask": [0] * 8,
        "state_values": zeros,
        "state_feature_mask": zeros,
        "state_step_mask": [0] * 10,
        "state_feature_names": CPS_STATE_NAMES,
        "schedule": schedule,
        "operation_active": [0] * 10,
        "source_owner": [-1] * 10,
        "target_owner": [-1] * 10,
        "query": {
            "owner_slot": 8,
            "query_owner": 8,
            "alternate_owner": 0,
            "alternate_label": "A",
            "irrelevant_owner": -1,
        },
        "query_owner": 8,
        "alternate_owner": 0,
        "alternate_label": "A",
        "irrelevant_owner": -1,
        "mechanism_supported": False,
        "query_answer_independent": True,
        "answer_target": {"answer_label": "I", "answer_index": 8},
        "answer_index": 8,
    }


class _FakeDataset:
    def __init__(self, target_bank: dict[str, dict]) -> None:
        self.target_bank = target_bank
        self.calls: list[list[str]] = []

    def get_batch(self, example_ids: list[str]):
        ids = list(example_ids)
        self.calls.append(ids)
        hidden = torch.arange(len(ids) * 12 * 2048, dtype=torch.float16).reshape(len(ids), 12, 2048)
        mask = torch.ones((len(ids), 12), dtype=torch.bool)
        return {"source_hidden": hidden, "source_mask": mask}, [{"example_id": item} for item in ids]


def _runtime(*ids: str) -> tuple[C1SSuccessorRuntime, _FakeDataset]:
    bank = {item: _target_row(item, answer_index=index % 2) for index, item in enumerate(ids)}
    dataset = _FakeDataset(bank)
    return C1SSuccessorRuntime(dataset, bank), dataset


def test_get_batch_preserves_order_and_returns_temporal_srw_tensors():
    runtime, dataset = _runtime("a", "b")
    batch = runtime.get_batch(["b", "a"], slots=8)
    assert dataset.calls == [["b", "a"]]
    assert batch["example_ids"] == ["b", "a"]
    assert batch["answers"].tolist() == [1, 0]
    assert batch["source_hidden"].shape == (2, 12, 2048)
    assert batch["object_labels"].tolist() == [[-1] * 8, [-1] * 8]
    assert batch["presence"].shape == (2, 8)
    assert batch["content_mask"].tolist()[0] == [True, True, False, False, False, False, False, False]
    assert batch["state_values"].shape == (2, 10, 8, 8)
    assert batch["state_feature_mask"].shape == (2, 10, 8, 8)
    assert batch["operation_active"].shape == (2, 10)
    assert batch["state_feature_names"] == [ERE_STATE_NAMES, ERE_STATE_NAMES]
    assert batch["object_kinds"][0][:2] == ["ere_entity", "ere_entity"]
    assert batch["irrelevant_owner"].tolist() == [2, 2]


def test_k1_is_query_state_compression_without_fake_answer_or_padding_owner():
    runtime, _ = _runtime("a")
    batch = runtime.get_batch(["a"], slots=1)
    assert batch["object_labels"].tolist() == [[-1]]
    assert batch["presence"].tolist() == [[True]]
    assert batch["content_mask"].tolist() == [[True]]
    assert batch["query_owner"].tolist() == [0]
    assert batch["alternate_owner"].tolist() == [-1]
    assert batch["alternate_label"].tolist() == [-1]
    assert batch["irrelevant_owner"].tolist() == [-1]
    assert batch["source_owner"].tolist() == [[0, 0, -1, -1, -1, -1, -1, -1, -1, -1]]
    assert batch["target_owner"].tolist() == [[0, 0, -1, -1, -1, -1, -1, -1, -1, -1]]
    assert batch["state_values"].shape == (1, 10, 1, 8)


def test_raw_nine_choice_ood_remains_behavior_only_in_runtime():
    row = _unsupported_ood_row("ood")
    dataset = _FakeDataset({"ood": row})
    runtime = C1SSuccessorRuntime(dataset, {"ood": row})
    batch = runtime.get_batch(["ood"], slots=8)
    assert batch["answers"].tolist() == [8]
    assert batch["mechanism_supported"].tolist() == [False]
    assert batch["query_answer_independent"].tolist() == [True]
    assert not bool(batch["operation_active"].any())
    assert not bool(batch["state_feature_mask"].any())
    assert not bool(batch["content_mask"].any())
    assert batch["query_owner"].tolist() == [0]


def test_iter_batches_and_ordered_prefetch_are_bounded_and_ordered():
    runtime, dataset = _runtime("a", "b", "c")
    batches = list(runtime.iter_batches(["a", "b", "c"], 2, slots=8))
    assert [batch["example_ids"] for batch in batches] == [["a", "b"], ["c"]]
    assert dataset.calls == [["a", "b"], ["c"]]
    events = []
    result = list(ordered_prefetch([3, 1, 2], lambda value: value * 2, max_prefetch=2, workers=2, telemetry=events.append))
    assert result == [6, 2, 4]
    assert {event.event for event in events} == {"submit", "load_done"}
    assert max(event.pending for event in events) <= 2


def test_forward_boundary_rejects_training_and_forbidden_fields():
    source = {
        "source_hidden": torch.zeros((1, 2, 2048), dtype=torch.float16),
        "source_mask": torch.ones((1, 2), dtype=torch.bool),
    }
    assert set(forward_inputs(source)) == {"source_hidden", "source_mask"}
    with pytest.raises(ValueError, match="forbidden fields"):
        forward_inputs({**source, "answer": torch.zeros(1, dtype=torch.long)})
    with pytest.raises(ValueError, match="missing"):
        forward_inputs({"source_hidden": source["source_hidden"]})


def test_source_and_target_nonfinite_shape_and_padding_errors_are_rejected():
    runtime, _ = _runtime("a")
    runtime.dataset.get_batch = lambda ids: (
        {
            "source_hidden": torch.full((1, 2, 2048), math.nan, dtype=torch.float16),
            "source_mask": torch.ones((1, 2), dtype=torch.bool),
        },
        [],
    )
    with pytest.raises(ValueError, match="finite"):
        runtime.get_batch(["a"], slots=8)

    runtime, _ = _runtime("a")
    bad = deepcopy(_target_row("a"))
    bad["presence"] = [1, 0, 1, 0, 0, 0, 0, 0]
    runtime.targets["a"] = bad
    with pytest.raises(ValueError, match="presence"):
        runtime.get_batch(["a"], slots=8)

    runtime, _ = _runtime("a")
    bad = deepcopy(_target_row("a"))
    bad["irrelevant_owner"] = 1
    bad["query"]["irrelevant_owner"] = 1
    runtime.targets["a"] = bad
    with pytest.raises(ValueError, match="padding slot"):
        runtime.get_batch(["a"], slots=8)
