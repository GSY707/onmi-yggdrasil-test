from __future__ import annotations

import hashlib

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1t import contract
from yggdrasil_v2.v2_a.closure_c1t.model import C1TConfig, C1TModel
from yggdrasil_v2.v2_a.closure_c1t.objective import (
    answer_margin,
    compute_training_loss,
    replace_support_payloads,
    paired_counterfactual_payloads,
    record_local_cross_entropy,
)
from yggdrasil_v2.v2_a.closure_c1t.runtime import (
    C1TRuntime,
    build_independent_card_cache,
)
from yggdrasil_v2.v2_a.closure_c1t.tasks import build_s1_bank


def _encoder(text: str) -> dict[str, torch.Tensor]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tokens, width = 2 + len(text) % 3, 12
    hidden = torch.tensor(
        [[digest[(row + column) % 32] / 255.0 for column in range(width)] for row in range(tokens)],
        dtype=torch.float32,
    )
    return {
        "hidden": hidden,
        "mask": torch.ones(tokens, dtype=torch.bool),
        "token_ids": torch.tensor([digest[index] for index in range(tokens)]),
    }


def _fixture(count: int = 4) -> tuple[C1TModel, dict[str, object], C1TRuntime]:
    bank = build_s1_bank()
    cache = build_independent_card_cache(bank, _encoder, source_identity={"model": "fixture"})
    runtime = C1TRuntime(bank, cache, address_width=16)
    batch = runtime.get_batch([row["example_id"] for row in bank[:count]])
    torch.manual_seed(17)
    model = C1TModel(
        C1TConfig(
            source_width=12,
            payload_width=16,
            address_width=16,
            slots=contract.MAX_SLOTS,
            operations=contract.MAX_OPERATIONS,
            ffn_width=32,
        )
    )
    return model, batch, runtime


def test_forward_contract_shapes_and_integrity() -> None:
    model, batch, runtime = _fixture()
    result = model(**runtime.forward_inputs(batch), return_trajectory=True)
    assert result["logits"].shape == (4, 9)
    assert result["trajectory"].shape == (4, 11, 8, 16)
    assert model.integrity_report()["passed"] is True
    assert model.integrity_report()["public_forward_parameters"] == list(contract.FORWARD_FIELDS)
    with pytest.raises(TypeError):
        model(**runtime.forward_inputs(batch), answers=batch["answers"])


def test_object_permutation_is_exactly_equivariant() -> None:
    model, batch, runtime = _fixture()
    model.eval()
    with torch.no_grad():
        boundary = model.boundary(**runtime.forward_inputs(batch))
        base = model.forward_from_boundary(boundary, return_trajectory=True)
        permutation = torch.tensor([2, 0, 7, 3, 5, 1, 6, 4])
        changed = model.forward_from_boundary(boundary.permuted(permutation), return_trajectory=True)
    assert torch.equal(base["logits"], changed["logits"])
    assert torch.equal(base["trajectory"][:, :, permutation], changed["trajectory"])


def test_boundary_card_locality_and_no_core_topology() -> None:
    model, batch, runtime = _fixture()
    inputs = runtime.forward_inputs(batch)
    with torch.no_grad():
        base = model.boundary(**inputs)
        changed = dict(inputs)
        changed["object_hidden"] = changed["object_hidden"].clone()
        changed["object_hidden"][0, 0, 0, 0] += 0.5
        object_changed = model.boundary(**changed)
        delta = (base.initial_payloads[0] - object_changed.initial_payloads[0]).abs().amax(-1)
        assert delta[0] > 0
        assert torch.equal(delta[1:], torch.zeros_like(delta[1:]))

        operation = dict(inputs)
        operation["operation_hidden"] = operation["operation_hidden"].clone()
        operation["operation_hidden"][0, 0, 0, 0] += 0.5
        operation_changed = model.boundary(**operation)
        no_core = model.forward_from_boundary(base, disable_recurrence=True)["logits"]
        no_core_changed = model.forward_from_boundary(
            operation_changed, disable_recurrence=True
        )["logits"]
        assert torch.equal(no_core, no_core_changed)


def test_transition_changes_only_registered_target() -> None:
    model, batch, runtime = _fixture()
    with torch.no_grad():
        boundary = model.boundary(**runtime.forward_inputs(batch))
        trajectory = model.forward_from_boundary(boundary, return_trajectory=True)["trajectory"]
    for row in range(trajectory.shape[0]):
        for step in range(model.config.operations):
            if not bool(boundary.operation_active[row, step]):
                assert torch.equal(trajectory[row, step], trajectory[row, step + 1])
                continue
            target = int(boundary.target_weights[row, step].argmax())
            keep = [index for index in range(model.config.slots) if index != target]
            assert torch.equal(trajectory[row, step, keep], trajectory[row, step + 1, keep])


def test_registered_routing_and_presence_fail_closed() -> None:
    model, batch, runtime = _fixture()
    inputs = runtime.forward_inputs(batch)
    bad_present = dict(inputs)
    bad_present["object_present"] = bad_present["object_present"].clone()
    bad_present["object_present"][0, 0] = False
    with pytest.raises(ValueError, match="exactly match"):
        model.boundary(**bad_present)
    bad_route = dict(inputs)
    bad_route["query_address"] = bad_route["query_address"].clone()
    bad_route["query_address"][0] = torch.roll(bad_route["query_address"][0], 1)
    with pytest.raises(ValueError, match="does not match"):
        model.boundary(**bad_route)


def test_causal_objective_is_finite_and_backpropagates() -> None:
    model, batch, _ = _fixture()
    total, components, outputs = compute_training_loss(model, batch)
    assert torch.isfinite(total)
    assert set(components) == {
        "full_answer_ce",
        "no_core_confusion",
        "support_margin_hinge",
        "total",
    }
    assert outputs["support_margin_drops"].shape == (4, 2)
    total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_answer_margin_is_gauge_invariant_and_penalizes_invalid_raw_logits() -> None:
    logits = torch.tensor([[1.0, 0.0, 10_000.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    valid = torch.tensor([[True, True, False, False, False, False, False, False, False]])
    answers = torch.tensor([0])
    margin = answer_margin(logits, answers, valid)
    shifted = answer_margin(logits + 123.0, answers, valid)
    assert torch.allclose(margin, shifted)
    assert margin.item() < -9_000.0
    assert record_local_cross_entropy(logits, answers, valid).item() > 9_000.0


def test_support_replacement_is_external_and_two_way() -> None:
    payloads = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    present = torch.tensor([[True, True, True, False], [True, True, True, True]])
    support = torch.tensor([[0, 1], [1, 3]])
    replaced = replace_support_payloads(payloads, present, support)
    assert replaced.shape == (2, 2, 4, 3)
    assert torch.equal(replaced[0, 0, 1:], payloads[0, 1:])
    assert torch.equal(replaced[0, 1, 0], payloads[0, 0])
    with pytest.raises(ValueError, match="distinct"):
        replace_support_payloads(payloads, present, torch.tensor([[0, 0], [1, 2]]))


def test_primary_support_intervention_uses_exact_factorial_counterpart() -> None:
    payloads = torch.arange(4 * 3 * 2, dtype=torch.float32).view(4, 3, 2)
    support = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    # row order: 00, 01, 10, 11
    counterparts = torch.tensor([[2, 1], [3, 0], [0, 3], [1, 2]])
    changed = paired_counterfactual_payloads(payloads, support, counterparts)
    assert torch.equal(changed[0, 0, 0], payloads[2, 0])
    assert torch.equal(changed[0, 0, 1:], payloads[0, 1:])
    assert torch.equal(changed[0, 1, 1], payloads[1, 1])
    with pytest.raises(ValueError, match="complete factorial"):
        paired_counterfactual_payloads(payloads, support, torch.full((4, 2), -1))
