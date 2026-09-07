from __future__ import annotations

from pathlib import Path

from yggdrasil_v2.v2_a.closure_c0r import contract
from yggdrasil_v2.v2_a.closure_c0r.readiness import decide_readiness


def test_contract_uses_new_single_use_identities_and_preserves_thresholds() -> None:
    data = contract.data_contract_manifest()
    ready = contract.readiness_contract_manifest()
    assert data["identity"] == "V2-A-CLOSURE-C0R-DATA-TRACE-20260824-1"
    assert ready["identity"] == "V2-A-CLOSURE-C0R-20260824-1"
    assert data["single_use"]["output_root"] != contract.OLD_C0_ROOT.as_posix()
    assert data["relation_gate"] == {
        "minimum_support": 128,
        "minimum_each_class_support": 64,
        "maximum_dominant_answer_mass": 0.8,
        "maximum_oracle_excess_over_chance": 0.1,
    }
    assert data["trace_gate"]["maximum_new_tokens"] == 512
    assert ready["authorization_on_pass"] == "C1 single-seed implementation and eligibility only"


def test_readiness_decision_distinguishes_pass_fail_and_incomplete() -> None:
    gates = {f"C00{index}": True for index in range(1, 9)}
    passed = decide_readiness(
        {"availability_complete": True, "passed": True, "gates": gates, "interpretation": "x"}
    )
    assert passed["status"] == "PASS_V2_A_CLOSURE_C0R_READINESS"
    assert passed["c1_single_seed_implementation_authorized"] is True
    failed = decide_readiness(
        {"availability_complete": True, "passed": False, "gates": {**gates, "C004": False}}
    )
    assert failed["status"] == "FAIL_V2_A_CLOSURE_C0R_READINESS"
    assert failed["authorizes"] == "nothing"
    incomplete = decide_readiness(
        {"availability_complete": False, "passed": False, "gates": gates}
    )
    assert incomplete["status"] == "INCOMPLETE_V2_A_CLOSURE_C0R_READINESS"


def test_fixed_roots_do_not_alias_old_c0() -> None:
    assert Path(contract.DATA_OUTPUT_ROOT) != Path(contract.OLD_C0_ROOT)
    assert Path(contract.READINESS_OUTPUT_ROOT) != Path(contract.OLD_C0_ROOT)

