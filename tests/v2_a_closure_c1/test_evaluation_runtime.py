from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1.evaluation_runtime import C1EvaluationRuntime


def _record(example_id: str, family: str, value: int) -> dict:
    mapping = {label: f"candidate:{i + 100}" for i, label in enumerate("ABCDEFGHI")}
    mapping["A"] = f"answer:{value}"
    return {"example_id": example_id, "family": family, "split": "validation", "label_mapping": mapping, "semantic_answer": f"answer:{value}"}


class FakeCache:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def get_batch(self, example_ids, *, device="cpu"):
        self.calls.append(list(example_ids))
        rows = [torch.tensor([[float(self.values[key]), 0.0, 0.0]], device=device) for key in example_ids]
        return {"source_hidden": torch.stack(rows), "source_mask": torch.ones(len(rows), 1, dtype=torch.bool, device=device)}, [{"example_id": key} for key in example_ids]


class FakeModel:
    def __init__(self):
        self.training = True
        self.forward_calls = []
        self.trace_calls = []

    def eval(self):
        self.training = False
        return self

    def train(self, value=True):
        self.training = value
        return self

    def __call__(self, source_hidden, source_mask, **kwargs):
        self.forward_calls.append((source_hidden.detach().clone(), dict(kwargs)))
        owner = source_hidden[:, 0, 0]
        logits = torch.full((source_hidden.shape[0], 9), -10.0, device=source_hidden.device)
        logits[:, 0] = owner
        trajectory = owner[:, None, None, None].expand(-1, 10, 1, 1).clone()
        return {"logits": logits, "trajectory": trajectory}

    def encode(self, source_hidden, source_mask):
        return source_hidden[:, :1, :1]

    def logits_from_state(self, state):
        logits = torch.full((state.shape[0], 9), -10.0, device=state.device)
        logits[:, 0] = state[:, 0, 0]
        return logits

    def trace_logits(self, trajectory, step_indices, global_positions, local_positions):
        # Query position is deliberately part of the result, so using the
        # next owner's own positions would be observably wrong.
        owner = trajectory[:, 0, 0, 0]
        query = global_positions.float()
        self.trace_calls.append((float(owner[0]), query.detach().cpu().tolist()))
        logits = torch.full((trajectory.shape[0], query.shape[1], 4), -5.0, device=trajectory.device)
        token = ((owner[:, None] + query) % 4).long()
        logits.scatter_(2, token[..., None], 5.0)
        return logits


def test_batches_are_single_cell_and_metadata_never_reaches_forward():
    records = [_record("e0", "ERE", 1), _record("e1", "ERE", 2), _record("c0", "CPS", 3), _record("c1", "CPS", 4)]
    model, cache = FakeModel(), FakeCache({"e0": 0, "e1": 1, "c0": 3, "c1": 4})
    report = C1EvaluationRuntime(model, cache, records, batch_size=2).evaluate_answers()
    assert set(report["cells"]) == {"ERE/validation", "CPS/validation"}
    assert all(set(kwargs) == {"return_trajectory"} for _, kwargs in model.forward_calls)
    assert cache.calls == [["e0", "e1"], ["c0", "c1"]]


def test_cyclic_hidden_and_state_shuffle_have_no_fixed_point():
    records = [_record("e0", "ERE", 0), _record("e1", "ERE", 1)]
    model, cache = FakeModel(), FakeCache({"e0": 0, "e1": 1})
    runtime = C1EvaluationRuntime(model, cache, records, batch_size=2)
    runtime.raw_answer_logits(intervention="shuffled_hidden")
    shuffled = model.forward_calls[-1][0][:, 0, 0].tolist()
    assert shuffled == [1.0, 0.0]
    runtime.raw_answer_logits(intervention="step5_state_shuffle")
    assert model.forward_calls[-1][1]["state_shuffle_indices"].tolist() == [1, 0]
    with pytest.raises(ValueError):
        C1EvaluationRuntime(model, cache, [_record("e0", "ERE", 0)], batch_size=1).raw_answer_logits(intervention="shuffled_hidden")


def test_trace_wrong_owner_uses_current_query_on_next_trajectory():
    records = [_record("e0", "ERE", 0), _record("e1", "ERE", 1), _record("c0", "CPS", 0), _record("c1", "CPS", 1)]
    model, cache = FakeModel(), FakeCache({"e0": 0, "e1": 1, "c0": 2, "c1": 3})
    bank = {key: {"token_ids": [0], "grammar_mask": [False], "step_indices": [1], "global_positions": [value], "local_positions": [0]} for key, value in (("e0", 0), ("e1", 1), ("c0", 0), ("c1", 1))}
    report = C1EvaluationRuntime(model, cache, records, batch_size=1).trace_credit(bank, per_family=2)
    assert set(report["families"]) == {"ERE", "CPS"}
    # For e0 the wrong-owner trajectory is e1, but the query remains e0's 0.
    assert (1.0, [[0]]) in model.trace_calls
    # The current target is always evaluated with its own trajectory first.
    assert (0.0, [[0]]) in model.trace_calls
