from __future__ import annotations

import inspect

import torch

from yggdrasil_v2.v2_a.closure_c1.deployment import reload_stripped, strip_trace_probe
from yggdrasil_v2.v2_a.closure_c1.interventions import (
    initial_slot_permutation,
    no_core,
    shuffle_hidden,
    step5_within_family_state_shuffle,
    zero_hidden,
)
from yggdrasil_v2.v2_a.closure_c1.model import C1Config, C1LatentReasoner


def _tiny() -> C1Config:
    return C1Config(
        source_width=16,
        latent_width=16,
        ffn_width=32,
        attention_heads=4,
        max_global_positions=32,
        max_local_positions=8,
    )


def test_c1_source_closed_shapes_probe_and_public_forward_boundary() -> None:
    config = _tiny()
    model = C1LatentReasoner(config).eval()
    source = torch.randn(3, 7, config.source_width)
    source_mask = torch.ones(3, 7, dtype=torch.bool)
    signature = inspect.signature(model.forward)
    assert "source_hidden" in signature.parameters
    assert "source_mask" in signature.parameters
    assert not {"family", "budget", "valid_choice_mask", "ast", "trace"} & set(signature.parameters)

    output = model(source, source_mask, return_trajectory=True)
    assert output["logits"].shape == (3, 9)
    assert output["h0"].shape == (3, 8, 16)
    assert output["trajectory"].shape == (3, 10, 8, 16)
    assert len(model.core_layers) == 2
    assert model.integrity_report()["source_closed_after_h0"]

    positions = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    trace = model.trace_logits(
        output["trajectory"],
        torch.ones_like(positions),
        positions,
        torch.zeros_like(positions),
    )
    assert trace.shape == (3, 3, 5597)


def test_boundary_padding_mask_is_semantically_inert() -> None:
    torch.manual_seed(17)
    model = C1LatentReasoner(_tiny()).eval()
    source = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    padded = torch.cat((source, torch.randn(2, 3, 16) * 100.0), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(2, 3, dtype=torch.bool)), dim=1)
    with torch.inference_mode():
        original = model(source, mask)["logits"]
        extended = model(padded, padded_mask)["logits"]
    assert torch.allclose(original, extended, atol=1e-6, rtol=1e-6)


def test_physical_strip_strict_reload_and_parameter_graph() -> None:
    config = _tiny()
    torch.manual_seed(5)
    model = C1LatentReasoner(config).eval()
    source = torch.randn(2, 6, config.source_width)
    source_mask = torch.ones(2, 6, dtype=torch.bool)
    before = model(source, source_mask)["logits"]
    artifact = strip_trace_probe(model)
    stripped = artifact.model.eval()
    after = stripped(source, source_mask)["logits"]
    assert stripped.trace_probe is None
    assert artifact.report["trace_probe_parameters"] == 0
    assert not any(name.startswith("trace_probe.") for name in artifact.state_dict)
    assert torch.equal(before, after)

    reloaded = reload_stripped(artifact.state_dict, config).model.eval()
    assert torch.equal(after, reloaded(source, source_mask)["logits"])


def test_hidden_recurrence_and_slot_interventions_are_explicit() -> None:
    model = C1LatentReasoner(_tiny()).eval()
    source = torch.randn(4, 6, 16)
    source_mask = torch.ones(4, 6, dtype=torch.bool)
    mapping = torch.tensor([1, 0, 3, 2])
    baseline = model(source, source_mask, return_trajectory=True)
    assert zero_hidden(model, source, source_mask)["logits"].shape == baseline["logits"].shape
    assert shuffle_hidden(model, source, source_mask, mapping)["logits"].shape == baseline["logits"].shape
    assert no_core(model, source, source_mask)["logits"].shape == baseline["logits"].shape
    shuffled = step5_within_family_state_shuffle(model, source, source_mask, mapping)
    assert shuffled["trajectory"].shape == baseline["trajectory"].shape
    # H5 is the post-intervention state owned by each output row; H6 then
    # continues from that exact shuffled state.
    assert torch.allclose(
        shuffled["trajectory"][:, 4],
        baseline["trajectory"][mapping, 4],
        atol=1e-5,
        rtol=1e-5,
    )

    permutation = torch.tensor([2, 7, 1, 5, 0, 6, 3, 4])
    slot_shuffled = initial_slot_permutation(model, source, source_mask, permutation)
    assert torch.allclose(slot_shuffled["logits"], baseline["logits"], atol=1e-5, rtol=1e-5)
