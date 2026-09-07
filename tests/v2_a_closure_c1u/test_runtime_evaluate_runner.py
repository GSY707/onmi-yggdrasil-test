from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1u import contract
from yggdrasil_v2.v2_a.closure_c1u.artifacts import SOURCE_FILES, source_hashes
from yggdrasil_v2.v2_a.closure_c1u.evaluate import (
    collect_prediction_rows,
    qualify_s1,
    wilson_interval,
)
from yggdrasil_v2.v2_a.closure_c1u.model import C1UConfig, C1UModel
from yggdrasil_v2.v2_a.closure_c1u.runner import inspect_successor
from yggdrasil_v2.v2_a.closure_c1u.runtime import (
    C1URuntime,
    IndependentCardCache,
    audit_independent_card_cache,
    build_independent_card_cache,
)
from yggdrasil_v2.v2_a.closure_c1u.tasks import build_s1_bank


def _encoder(text: str) -> dict[str, torch.Tensor]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tokens, width = 2 + len(text) % 4, 12
    return {
        "hidden": torch.tensor(
            [[digest[(i + j) % 32] / 255.0 for j in range(width)] for i in range(tokens)]
        ),
        "mask": torch.ones(tokens, dtype=torch.bool),
        "token_ids": torch.tensor([digest[index] for index in range(tokens)]),
    }


def _runtime() -> tuple[list[dict[str, object]], IndependentCardCache, C1URuntime]:
    bank = build_s1_bank()
    cache = build_independent_card_cache(
        bank, _encoder, source_identity={"model": "fixture", "revision": "fixed"}
    )
    return bank, cache, C1URuntime(bank, cache, address_width=16)


def test_independent_cache_has_one_call_and_ledger_row_per_card() -> None:
    bank = build_s1_bank()
    calls = []

    def counted(text: str) -> dict[str, torch.Tensor]:
        calls.append(text)
        return _encoder(text)

    cache = build_independent_card_cache(bank, counted, source_identity={"model": "fixture"})
    audit = audit_independent_card_cache(bank, cache)
    assert audit["passed"] is True
    assert len(calls) == len(cache.cards) == len(cache.ledger) == 192
    assert cache.manifest()["whole_record_hidden_reuse"] is False


def test_cache_tampering_is_detected() -> None:
    bank, cache, _ = _runtime()
    key = next(iter(cache.cards))
    tampered_card = replace(cache.cards[key], hidden=cache.cards[key].hidden + 1.0)
    cards = dict(cache.cards)
    cards[key] = tampered_card
    tampered = replace(cache, cards=cards)
    audit = audit_independent_card_cache(bank, tampered)
    assert audit["passed"] is False
    assert any("hidden_sha256" in error for error in audit["errors"])


def test_same_card_text_must_encode_deterministically_across_records() -> None:
    bank = build_s1_bank()
    calls = 0

    def drifting(text: str) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        value = _encoder(text)
        value["hidden"] = value["hidden"] + calls * 1.0e-4
        return value

    with pytest.raises(RuntimeError, match="same_text_encoding_nondeterministic"):
        build_independent_card_cache(bank, drifting, source_identity={"model": "drifting"})


def test_runtime_preserves_order_and_separates_forward_targets() -> None:
    bank, _, runtime = _runtime()
    ids = [bank[5]["example_id"], bank[0]["example_id"], bank[-1]["example_id"]]
    batch = runtime.get_batch(ids)
    assert batch["example_ids"] == ids
    assert batch["object_hidden"].shape[:2] == (3, 8)
    assert batch["operation_hidden"].shape[:2] == (3, 10)
    assert batch["query_hidden"].ndim == 3
    assert set(runtime.forward_inputs(batch)) == set(contract.FORWARD_FIELDS)
    assert set(runtime.forward_inputs(batch)).isdisjoint(contract.FORBIDDEN_FORWARD_FIELDS)
    assert batch["counterfactual_indices"].tolist() == [[-1, -1], [-1, -1], [-1, -1]]


def test_random_gate_free_model_fails_behavior_gates_but_keeps_invariance() -> None:
    bank, cache, runtime = _runtime()
    batch = runtime.get_batch([row["example_id"] for row in bank])
    model = C1UModel(
        C1UConfig(
            source_width=cache.source_width,
            payload_width=16,
            address_width=16,
            ffn_width=32,
        )
    )
    prediction = collect_prediction_rows(model, batch)
    report = qualify_s1(prediction)
    assert model.integrity_report()["gate_free_target_overwrite"] is True
    assert prediction["invariance"]["passed"] is True
    assert report["passed"] is False
    assert report["status"] == "FAIL_C1U_S1_QUALIFICATION"
    assert report["authorizes"] == "nothing"


def test_perfect_registered_rows_can_pass_all_s1_gates() -> None:
    rows = []
    for record in build_s1_bank():
        rows.append(
            {
                "example_id": record["example_id"],
                "family": record["family"],
                "factorial_group_id": record["factorial_group_id"],
                "factor_cell": record["factors"],
                "answer_index": record["answer_index"],
                "prediction_index": record["answer_index"],
                "no_core_prediction_index": -1,
                "full_correct": True,
                "no_core_correct": False,
                "no_core_logits": [0.0] * contract.ANSWER_CLASSES,
                "base_margin": 2.0,
                "no_core_margin": 0.0,
                "no_core_margin_drop": 2.0,
                "support_margin_drops": [1.0, 1.0],
                "support_prediction_indices": [record["answer_index"]] * 2,
                "support_expected_answer_indices": [record["answer_index"]] * 2,
                "support_flip_correct": [True, True],
            }
        )
    report = qualify_s1(
        {"rows": rows, "invariance": {"passed": True}},
        require_frozen_cardinality=True,
    )
    assert report["passed"] is True
    assert report["status"] == "PASS_C1U_S1"
    assert report["authorizes"] == "S2_CONTRACT_DESIGN_ONLY"


def test_wilson_interval_is_finite_and_fail_closed() -> None:
    metric = wilson_interval(16, 16)
    assert 0.65 < metric["wilson_lower"] < metric["point"] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_source_closure_and_read_only_inspection_are_state_aware() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert len(SOURCE_FILES) == len(set(SOURCE_FILES))
    assert len(source_hashes(repo_root)) == len(SOURCE_FILES)
    report = inspect_successor(repo_root)
    assert report["passed"] is True
    assert report["authorizes"] == "nothing"
    assert report["evidence_level"] == "launch-readiness-with-synthetic-structural-smoke"
    assert report["status"] in {
        "S1_CONTRACT_READY_UNCONSUMED",
        "S1_PREFLIGHT_CONSUMED",
        "S1_FORMAL_CONSUMED",
    }
    assert report["contract"]["training_authorized"] is True
    assert report["predecessor"]["passed"] is True
    assert report["inspection_mutations"] == 0
