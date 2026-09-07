from __future__ import annotations

from yggdrasil_v2.v2_a.closure_c1s import contract


def test_contract_is_s0_only_and_never_authorizes_formal_route():
    manifest = contract.contract_manifest()

    assert manifest["identity"] == contract.IDENTITY
    assert manifest["stage"] == "S0-zero-update-structure-and-measurement-qualification"
    assert manifest["formal"] is False
    assert manifest["qualification"] is True
    assert manifest["model"]["slots"] == 8
    assert manifest["matched_k1_model"]["slots"] == 1
    assert manifest["model"]["operations"] == manifest["matched_k1_model"]["operations"] == 10
    assert manifest["authorization_on_s0_pass"] == contract.AUTHORIZATION_ON_S0_PASS
    assert "single-seed formal" in manifest["never_authorizes"]
    assert "V2-A PASS" in manifest["never_authorizes"]
    assert "V2-C" in manifest["never_authorizes"]


def test_conditional_s1_s2_contracts_are_fixed_before_s0():
    manifest = contract.contract_manifest()
    s1 = manifest["s1_conditional_contract"]
    s2 = manifest["s2_conditional_contract"]

    assert s1["records"] == 32
    assert s1["maximum_updates"] == 4_000
    assert s1["checkpoint_selection"] is False
    assert s2["discovery_train_per_family"] == 3_072
    assert s2["discovery_eval_per_family"] == 512
    assert s2["maximum_updates"] == 4_608
    assert s2["checkpoint_selection"] is False
    assert s2["answer_point_floor"] == 0.75
    assert s2["answer_wilson_lower_floor"] == 0.70

