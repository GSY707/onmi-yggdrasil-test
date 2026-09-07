from __future__ import annotations

from yggdrasil_v2.v2_a.closure_c0.audit import decide
from yggdrasil_v2.v2_a.closure_c0.contract import ARM_CONTRACT, GATE_IDS, contract_manifest


def test_contract_freezes_four_fair_arms_and_narrow_authorization() -> None:
    manifest = contract_manifest()
    assert ARM_CONTRACT["arms"] == ["direct", "text_cot", "latent_k1", "latent_k8"]
    assert ARM_CONTRACT["shared_inputs"]["family_visible_to_forward"] is False
    assert ARM_CONTRACT["teacher_symmetry"]["required"] is True
    assert ARM_CONTRACT["shared_inputs"]["reasoning_budget_visible_to_forward"] is False
    assert (
        ARM_CONTRACT["teacher_symmetry"]["primary_lane"]["dense_state_or_claim_targets_allowed"]
        is False
    )
    assert (
        ARM_CONTRACT["teacher_symmetry"]["secondary_lane"][
            "eligible_for_medium_superiority_claim"
        ]
        is False
    )
    assert "training_flops" in ARM_CONTRACT["cost_ledger"]
    assert "end_to_end_latency" in ARM_CONTRACT["cost_ledger"]
    assert ARM_CONTRACT["selection_and_budget"]["required_matched_slices"] == [
        "equal_examples",
        "equal_gpu_hours",
    ]
    assert ARM_CONTRACT["trace_qualification"]["decode_max_new_tokens"] == 512
    assert ARM_CONTRACT["cache_policy"]["p0m_smoke_cache_reusable_for_c1"] is False
    assert manifest["authorization_on_pass"] == "C1 single-seed implementation and eligibility only"
    assert "V2-B" in manifest["never_authorizes"]


def test_decision_distinguishes_pass_fail_and_incomplete() -> None:
    passed = decide({"availability_complete": True, "gates": dict.fromkeys(GATE_IDS, True)})
    assert passed["status"] == "PASS_V2_A_CLOSURE_C0_READINESS"
    assert passed["v2a_passed"] is False
    assert passed["four_arm_results_present"] is False
    assert passed["c1_single_seed_implementation_authorized"] is True

    failed_gates = dict.fromkeys(GATE_IDS, True)
    failed_gates["C006"] = False
    failed = decide({"availability_complete": True, "gates": failed_gates})
    assert failed["status"] == "FAIL_V2_A_CLOSURE_C0_READINESS"
    assert failed["authorizes"] == "nothing"

    incomplete = decide({"availability_complete": False, "gates": failed_gates})
    assert incomplete["status"] == "INCOMPLETE_V2_A_CLOSURE_C0_READINESS"
    assert incomplete["exit_code"] == 3
