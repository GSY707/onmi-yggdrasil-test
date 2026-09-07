from __future__ import annotations

from dataclasses import replace
import hashlib

import torch

from yggdrasil_v2.v2_a.closure_c1u.runtime import build_independent_card_cache
from yggdrasil_v2.v2_a.closure_c1u.train import set_determinism
from yggdrasil_v2.v2_a.closure_c1u_s2 import contract
from yggdrasil_v2.v2_a.closure_c1u_s2.evaluate import qualify
from yggdrasil_v2.v2_a.closure_c1u_s2.model import S2Config, S2Model
from yggdrasil_v2.v2_a.closure_c1u_s2.runtime import (
    BoundedBatchProvider,
    S2Runtime,
)
from yggdrasil_v2.v2_a.closure_c1u_s2.tasks import build_multibank, split_records
from yggdrasil_v2.v2_a.closure_c1u_s2.train import (
    TrainSpec,
    build_schedule,
    load_endpoint,
    save_endpoint,
    schedule_report,
    train_fixed_endpoint,
)


def _encoder(text: str) -> dict[str, torch.Tensor]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tokens, width = 2 + len(text) % 2, 12
    return {
        "hidden": torch.tensor(
            [
                [digest[(row + column) % 32] / 255.0 for column in range(width)]
                for row in range(tokens)
            ],
            dtype=torch.float32,
        ),
        "mask": torch.ones(tokens, dtype=torch.bool),
        "token_ids": torch.tensor([digest[index] for index in range(tokens)]),
    }


def test_multibank_schedule_is_deterministic_complete_and_balanced() -> None:
    records = build_multibank()
    train, _ = split_records(records, "F0")
    left = build_schedule(train, fold_id="F0", seed=19, maximum_updates=32)
    right = build_schedule(train, fold_id="F0", seed=19, maximum_updates=32)
    assert left == right
    report = schedule_report(left)
    assert report["updates"] == 32
    assert report["cycles"] == 2
    assert report["complete_groups_per_batch"] is True
    assert set(report["group_counts"].values()) == {2}


def test_small_cpu_training_is_bounded_and_writes_one_fixed_endpoint(tmp_path) -> None:
    records = build_multibank()
    train, _ = split_records(records, "F0")
    cache = build_independent_card_cache(
        records, _encoder, source_identity={"model": "fixture", "revision": "fixed"}
    )
    runtime = S2Runtime(records, cache, address_width=16)
    provider = BoundedBatchProvider(runtime, maximum_batches=1)
    schedule = build_schedule(train, fold_id="F0", seed=23, maximum_updates=2)
    spec = replace(
        TrainSpec(),
        maximum_updates=2,
        warmup_updates=1,
        log_interval=1,
        pin_memory=False,
    )
    set_determinism(29)
    model = S2Model(
        S2Config(
            source_width=12,
            payload_width=16,
            address_width=16,
            ffn_width=32,
            workspace_slots=8,
        )
    )
    training = train_fixed_endpoint(
        model,
        provider,
        schedule,
        identity="TEST_C1U_S2",
        arm="K8",
        fold_id="F0",
        model_seed=29,
        order_seed=23,
        spec=spec,
        device="cpu",
    )
    assert training["optimizer_steps"] == 2
    assert training["provider"]["bounded"] is True
    assert training["provider"]["cached_batch_count"] == 1
    path = tmp_path / "fixed.pt"
    digest = save_endpoint(
        path,
        model=model,
        identity="TEST_C1U_S2",
        arm="K8",
        fold_id="F0",
        update=2,
        schedule_sha256=schedule_report(schedule)["sha256"],
    )
    loaded, payload = load_endpoint(path)
    assert len(digest) == 64
    assert payload["optimizer_state_saved"] is False
    assert payload["checkpoint_selection"] is False
    assert loaded.integrity_report()["passed"] is True


def _perfect_row(record: dict, *, arm: str, correct: bool) -> dict:
    heldout_fold = {"B0": "F0", "B1": "F0", "B2": "F1", "B3": "F1", "B4": "F2", "B5": "F2"}
    answer = int(record["answer_index"])
    prediction = answer if correct else (answer + 1) % 9
    return {
        "example_id": record["example_id"],
        "fold_id": heldout_fold[record["bank_id"]],
        "arm": arm,
        "bank_id": record["bank_id"],
        "family": record["family"],
        "factorial_group_id": record["factorial_group_id"],
        "factor_cell": record["factors"],
        "answer_index": answer,
        "prediction_index": prediction,
        "full_correct": correct,
        "no_core_prediction_index": (answer + 1) % 9,
        "no_core_correct": False,
        "base_margin": 2.0,
        "no_core_margin_drop": 1.0,
        "support_slots": record["support_slots"],
        "query_owner_slot": next(
            index
            for index, address in enumerate(record["model_public"]["object_addresses"])
            if address == record["model_public"]["query_address"]
        ),
        "distinct_support_owners": True,
        "support_margin_drops": [1.0, 1.0],
        "support_prediction_indices": [answer, answer],
        "support_expected_answer_indices": [answer, answer],
        "support_flip_correct": [True, True],
        "two_contributor": True,
        "owner_deletion_margin_drops": [1.0, 1.0],
        "owner_swap_logit_l2": [1.0, 1.0] if arm == "K8" else [0.0, 0.0],
        "owner_swap_logit_max_abs": [0.5, 0.5] if arm == "K8" else [0.0, 0.0],
        "permutation_logit_max_abs": 0.0,
        "workspace_slots": 8 if arm == "K8" else 1,
    }


def test_qualification_requires_absolute_causal_functional_and_paired_gain() -> None:
    records = build_multibank()
    k8 = [_perfect_row(record, arm="K8", correct=True) for record in records]
    k1 = [_perfect_row(record, arm="K1", correct=False) for record in records]
    report = qualify(k8, k1)
    assert report["passed"] is True
    assert report["authorizes"] == contract.PASS_AUTHORIZATION
    equal = qualify(k8, [_perfect_row(record, arm="K1", correct=True) for record in records])
    assert equal["passed"] is False
    assert not all(equal["sections"]["paired_gain"].values())
