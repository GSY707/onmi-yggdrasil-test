from __future__ import annotations

import torch

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis import arity_analysis


def _logits(correct_logit: float) -> torch.Tensor:
    value = torch.zeros(9, dtype=torch.float32)
    value[0] = correct_logit
    return value


def test_model_contributors_exclude_decision_and_nonpositive_denominator(monkeypatch) -> None:
    def fake_task(row, source):
        del source
        return {
            "valid": True,
            "certificate": {
                "status": "ASSESSED",
                "support_slots": [0, 1, 2],
            },
            "answer_support": {
                "status": "ASSESSED",
                "minimal_answer_support_arity": 2,
            },
            "counterfactual_influence": {
                "status": "ASSESSED",
                "leave_one_out": {
                    "0": {"winner_changed": True},
                    "1": {"winner_changed": True},
                },
            },
        }

    monkeypatch.setattr(arity_analysis, "assess_task_arity", fake_task)
    rows = [
        {
            "example_id": example_id,
            "family": "CPS",
            "objects": [
                {"slot": 0, "kind": "cps_candidate"},
                {"slot": 1, "kind": "cps_candidate"},
                {"slot": 2, "kind": "cps_decision"},
            ],
        }
        for example_id in ("decision-only", "nonpositive-denominator")
    ]
    full = torch.stack([_logits(4.0), _logits(4.0)])
    h0 = torch.stack([_logits(2.0), _logits(5.0)])
    copy = full.clone()
    slot_replaced = torch.stack(
        [
            torch.stack([_logits(4.0), _logits(4.0), _logits(0.0)]),
            torch.stack([_logits(0.0), _logits(0.0), _logits(0.0)]),
        ]
    )
    collection = {
        "example_ids": ["decision-only", "nonpositive-denominator"],
        "answers": torch.tensor([0, 0]),
        "paths": {"full": full, "h0": h0, "copy_all_query_owner": copy},
        "initial_slot_replace_logits": slot_replaced,
        "presence": torch.ones(2, 3, dtype=torch.bool),
        "content_mask": torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool),
    }
    sources = {
        example_id: {"example_id": example_id, "family": "CPS"}
        for example_id in collection["example_ids"]
    }

    report = arity_analysis.audit_task_and_model_arity(
        rows,
        sources,
        collection,
        contributor_ratio=0.10,
        eligible_margin=0.5,
    )

    first, second = report["records"]
    assert first["model"]["contributor_slots"] == []
    assert first["model"]["two_contributors"] is False
    assert second["model"]["contributor_denominator_valid"] is False
    assert second["model"]["contributor_slots"] == []

