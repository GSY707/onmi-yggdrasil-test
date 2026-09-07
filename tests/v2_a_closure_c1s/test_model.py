from __future__ import annotations

import inspect

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1s.model import (
    C1SConfig,
    C1SModel,
    strip_training_auxiliary,
)


def _config(*, slots: int = 4, auxiliary: bool = True) -> C1SConfig:
    return C1SConfig(
        source_width=32,
        payload_width=32,
        address_width=8,
        slots=slots,
        operations=10,
        reader_depth=1,
        attention_heads=4,
        ffn_width=64,
        answer_classes=9,
        auxiliary_width=8,
        route_temperature=0.25,
        training_auxiliary=auxiliary,
    )


def _source() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    hidden = torch.randn((2, 9, 32), generator=generator)
    mask = torch.ones((2, 9), dtype=torch.bool)
    mask[1, -2:] = False
    return hidden, mask


def test_public_forward_is_source_only_and_outputs_addressed_trajectory():
    model = C1SModel(_config()).eval()
    hidden, mask = _source()
    output = model(hidden, mask, return_trajectory=True, return_auxiliary=True)

    assert list(inspect.signature(model.forward).parameters) == [
        "source_hidden",
        "source_mask",
        "return_trajectory",
        "return_auxiliary",
    ]
    assert output["logits"].shape == (2, 9)
    assert output["addresses"].shape == (2, 4, 8)
    assert output["trajectory"].shape == (2, 10, 4, 32)
    assert output["query_weights"].shape == (2, 4)
    assert output["auxiliary"] is not None
    assert model.integrity_report()["passed"] is True
    with pytest.raises(TypeError):
        model(hidden, mask, family=torch.zeros(2, dtype=torch.long))


def test_address_payload_pair_permutation_is_equivariant_and_logits_invariant():
    torch.manual_seed(23)
    model = C1SModel(_config()).eval()
    hidden, mask = _source()
    with torch.no_grad():
        boundary = model.boundary(hidden, mask)
        original = model.forward_from_boundary(boundary, return_trajectory=True)
        permutation = torch.tensor([3, 1, 0, 2])
        permuted = model.forward_from_boundary(boundary.permuted(permutation), return_trajectory=True)

    assert torch.allclose(original["logits"], permuted["logits"], atol=1.0e-6, rtol=0.0)
    assert torch.allclose(
        original["trajectory"][:, :, permutation],
        permuted["trajectory"],
        atol=1.0e-6,
        rtol=0.0,
    )


def test_transition_changes_only_selected_slots_and_inactive_is_identity():
    torch.manual_seed(29)
    model = C1SModel(_config()).eval()
    payloads = torch.randn((2, 4, 32))
    operation = torch.randn((2, 32))
    source = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1)
    target = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).expand(2, -1)
    with torch.no_grad():
        updated = model.transition(payloads, operation, source, target, torch.ones(2))
        inactive = model.transition(payloads, operation, source, target, torch.zeros(2))

    assert torch.equal(updated[:, 1], payloads[:, 1])
    assert torch.equal(updated[:, 3], payloads[:, 3])
    assert not torch.equal(updated[:, 0], payloads[:, 0])
    assert not torch.equal(updated[:, 2], payloads[:, 2])
    assert torch.equal(inactive, payloads)


def test_training_auxiliary_is_physically_stripped_without_changing_logits():
    torch.manual_seed(31)
    model = C1SModel(_config()).eval()
    hidden, mask = _source()
    deployment = strip_training_auxiliary(model).eval()
    with torch.no_grad():
        before = model(hidden, mask)["logits"]
        after = deployment(hidden, mask, return_auxiliary=True)

    assert deployment.has_training_auxiliary is False
    assert after["auxiliary"] is None
    assert not any(name.startswith("training_auxiliary.") for name in deployment.state_dict())
    assert torch.equal(before, after["logits"])


def test_k8_and_k1_have_identical_trainable_parameter_count():
    torch.manual_seed(37)
    k8 = C1SModel(_config(slots=8))
    torch.manual_seed(37)
    k1 = C1SModel(_config(slots=1))
    assert k8.parameter_report()["trainable_parameters"] == k1.parameter_report()["trainable_parameters"]

