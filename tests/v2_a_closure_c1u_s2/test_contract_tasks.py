from __future__ import annotations

from yggdrasil_v2.v2_a.closure_c1u.tasks import build_s1_bank
from yggdrasil_v2.v2_a.closure_c1u_s2 import contract
from yggdrasil_v2.v2_a.closure_c1u_s2.tasks import (
    audit_multibank,
    build_multibank,
    split_records,
)


def test_contract_is_multibank_single_seed_and_single_use() -> None:
    manifest = contract.manifest()
    assert manifest["multi_bank"] is True
    assert manifest["multi_seed"] is False
    assert manifest["scientific_model_seeds"] == [contract.MODEL_SEED]
    assert manifest["bootstrap_seed_is_not_a_training_seed"] is True
    assert manifest["endpoint_count"] == 6
    assert manifest["total_formal_optimizer_steps"] == 24_000
    assert manifest["retry_allowed"] is False
    assert manifest["user_authorization"] == (
        "没问题，那只做多bank，多seed先不做。那接下来就应该继续做S2了，开始吧"
    )


def test_six_fresh_banks_are_factorial_disjoint_and_shortcut_audited() -> None:
    records = build_multibank()
    audit = audit_multibank(records, s1_records=build_s1_bank())
    assert len(records) == 192
    assert audit["passed"] is True
    assert audit["checks"]["pairwise_opaque_disjoint"] is True
    assert audit["checks"]["s1_opaque_disjoint"] is True
    assert audit["checks"]["label_permutations_unique"] is True
    assert audit["checks"]["label_permutations_nonidentity"] is True
    assert audit["owner_only"]["maximum"] == 0.5
    assert not any(audit["pairwise_opaque_overlap"].values())
    assert not any(audit["s1_opaque_overlap"].values())


def test_each_fold_has_four_train_two_heldout_and_each_bank_is_heldout_once() -> None:
    records = build_multibank()
    heldout = []
    for fold in contract.FOLDS:
        train, evaluation = split_records(records, str(fold["fold_id"]))
        assert len(train) == 128
        assert len(evaluation) == 64
        assert {row["bank_id"] for row in train} == set(fold["train"])
        assert {row["bank_id"] for row in evaluation} == set(fold["heldout"])
        assert {row["example_id"] for row in train}.isdisjoint(
            row["example_id"] for row in evaluation
        )
        heldout.extend(row["example_id"] for row in evaluation)
    assert len(heldout) == len(set(heldout)) == 192
