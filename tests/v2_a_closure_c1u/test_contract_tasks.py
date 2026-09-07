from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1u import contract
from yggdrasil_v2.v2_a.closure_c1u.cards import (
    address_matrix,
    normalize_public_address,
    public_address_vector,
    validate_public_cards,
)
from yggdrasil_v2.v2_a.closure_c1u.tasks import (
    audit_bridge_fault_registry,
    audit_factorial_group,
    audit_public_identifiability,
    audit_s1_bank,
    build_factorial_group,
    build_s1_bank,
    replay_public_answer,
)


def test_contract_freezes_only_stage_limited_s1_after_sealed_s0_pass() -> None:
    manifest = contract.manifest()
    assert contract.IDENTITY == "V2-A-CLOSURE-C1U-PUBLICLY-GROUNDED-GATE-FREE-WORKSPACE-20260901-1"
    assert manifest["direct_cutover"] is True
    assert manifest["implementation_only"] is False
    assert manifest["training_authorized"] is True
    assert manifest["conditional_future_training_authorization"] is True
    assert manifest["training_authorization_is_stage_limited"] is True
    assert manifest["user_authorization"] == "没问题，按这个顺序做"
    assert manifest["zero_training_s0_authorized_only_after_preflight"] is True
    assert manifest["optimizer_steps"] == manifest["model_writes"] == 0
    assert manifest["bank_records"] == 32
    assert manifest["s1_execution"] == {
        "status": "FROZEN_READY_UNCONSUMED",
        "execution_document": "docs/v2-a-closure-c1u-s1-execution.md",
        "preflight_identity": contract.S1_PREFLIGHT_IDENTITY,
        "identity": contract.S1_IDENTITY,
        "preflight_root": contract.S1_PREFLIGHT_ROOT.as_posix(),
        "root": contract.S1_ROOT.as_posix(),
        "fixed_endpoint": "fixed_4000",
        "maximum_updates": 4_000,
        "command_exposed": True,
    }
    assert manifest["s0_predecessor"]["result_sha256"] == contract.S0_RESULT_SHA256
    assert manifest["s0_predecessor"]["evidence_seal_sha256"] == contract.S0_EVIDENCE_SEAL_SHA256
    assert manifest["architecture"] == "publicly_grounded_gate_free_workspace"
    assert manifest["transition"]["learned_write_gate"] is False
    assert manifest["public_grounding"]["public_only_reference_replay"] is True
    assert manifest["c1t_failed_predecessor"]["authorizes"] == "nothing"
    assert "source_hidden" not in manifest["forward_fields"]
    assert set(contract.FORWARD_FIELDS).isdisjoint(contract.FORBIDDEN_FORWARD_FIELDS)
    fixed = (
        contract.S0_PREFLIGHT_ROOT,
        contract.S0_PREFLIGHT_LEASE,
        contract.S0_ROOT,
        contract.S0_LEASE,
        contract.S1_PREFLIGHT_ROOT,
        contract.S1_PREFLIGHT_LEASE,
        contract.S1_ROOT,
        contract.S1_LEASE,
    )
    assert len(set(fixed)) == len(fixed)
    assert all("c1u" in path.as_posix().casefold() for path in fixed)


def test_s1_bank_has_exact_factorial_causal_arity() -> None:
    bank = build_s1_bank()
    audit = audit_s1_bank(bank)
    assert len(bank) == 32
    assert audit["passed"] is True
    assert audit["family_counts"] == {"CPS": 16, "ERE": 16}
    assert audit["groups"] == 8
    for group in audit["group_audits"].values():
        assert group["passed"] is True
        assert group["checks"]["each_single_factor_flips_answer"] is True
        assert group["checks"]["each_flip_changes_only_its_object_card"] is True


def test_public_only_replay_needs_no_family_ast_or_factor_metadata() -> None:
    for family in contract.FAMILIES:
        for record in build_factorial_group(family, 0):
            public_only = {
                "model_public": deepcopy(record["model_public"]),
                "family": "deliberately-wrong-and-unused",
            }
            replay = replay_public_answer(public_only)
            assert replay["semantic_answer"] == record["semantic_answer"]
            assert replay["raw_answer_label"] == record["raw_answer_label"]
            assert audit_public_identifiability(record)["passed"] is True


def test_bridge_fault_registry_kills_all_registered_missing_conflicting_and_swapped_faults() -> None:
    report = audit_bridge_fault_registry(build_s1_bank())
    assert report["passed"] is True
    assert report["registered"] == report["killed"] == 6
    assert set(report["faults"]) == {
        "ere_missing_value_legend",
        "ere_conflicting_value_legend",
        "ere_swapped_value_semantics",
        "cps_missing_candidate_index",
        "cps_duplicate_candidate_index",
        "cps_swapped_candidate_semantics",
    }


@pytest.mark.parametrize("family", contract.FAMILIES)
def test_factorial_group_rejects_answer_or_card_tampering(family: str) -> None:
    group = build_factorial_group(family, 0)
    answer_tampered = deepcopy(group)
    answer_tampered[0]["semantic_answer"] = "tampered"
    assert audit_factorial_group(answer_tampered)["passed"] is False
    card_tampered = deepcopy(group)
    card_tampered[1]["model_public"]["query_card"] += "\nleak"
    audit = audit_factorial_group(card_tampered)
    assert audit["passed"] is False
    assert audit["checks"]["public_invariants"] is False


def test_public_addresses_are_normalized_deterministic_unit_vectors() -> None:
    assert normalize_public_address("  Ａlice  ") == "alice"
    left = public_address_vector("Alice", width=16)
    right = public_address_vector("ＡLICE", width=16)
    assert torch.equal(left, right)
    assert torch.isclose(torch.linalg.vector_norm(left), torch.tensor(1.0), atol=1.0e-6)
    matrix = address_matrix(["left", "right", "target"], width=16)
    assert matrix.shape == (3, 16)
    with pytest.raises(ValueError, match="collide"):
        address_matrix(["Alice", "ＡLICE"], width=16)


def test_public_card_audit_rejects_unregistered_route() -> None:
    public = deepcopy(build_factorial_group("ERE", 0)[0]["model_public"])
    public["operation_source_addresses"][0] = "unknown"
    with pytest.raises(ValueError, match="absent"):
        validate_public_cards(public)


def test_c1u_source_does_not_import_c1s_runtime_or_model() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "src" / "yggdrasil_v2" / "v2_a" / "closure_c1u"
    imports = []
    for path in package.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().casefold()
            if stripped.startswith(("from ", "import ")) and "closure_c1s" in stripped:
                imports.append((path.name, line))
    assert imports == []
