from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1t import contract
from yggdrasil_v2.v2_a.closure_c1t.cards import (
    address_matrix,
    normalize_public_address,
    public_address_vector,
    validate_public_cards,
)
from yggdrasil_v2.v2_a.closure_c1t.tasks import (
    audit_factorial_group,
    audit_s1_bank,
    build_factorial_group,
    build_s1_bank,
)


def test_contract_is_a_direct_s0_to_single_use_s1_cutover() -> None:
    manifest = contract.manifest()
    assert contract.IDENTITY == "V2-A-CLOSURE-C1T-CAUSALLY-PARTITIONED-WORKSPACE-20260901-1"
    assert manifest["direct_cutover"] is True
    assert manifest["implementation_only"] is False
    assert manifest["training_authorized"] is True
    assert manifest["training_authorization_is_stage_limited"] is True
    assert manifest["user_authorization"] == "开始S1阶段"
    assert manifest["zero_training_s0_authorized_only_after_preflight"] is True
    assert manifest["optimizer_steps"] == manifest["model_writes"] == 0
    assert manifest["s1_records"] == 32
    assert manifest["s1_maximum_updates"] == 4_000
    assert manifest["s1_fixed_endpoint"] == "fixed_4000"
    assert manifest["s1_authorized_scope"] == contract.S1_AUTHORIZED_SCOPE
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
    assert all("c1s" not in path.as_posix().casefold() for path in fixed)


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


def test_c1t_source_does_not_import_c1s_runtime_or_model() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "src" / "yggdrasil_v2" / "v2_a" / "closure_c1t"
    imports = []
    for path in package.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().casefold()
            if stripped.startswith(("from ", "import ")) and "closure_c1s" in stripped:
                imports.append((path.name, line))
    assert imports == []
