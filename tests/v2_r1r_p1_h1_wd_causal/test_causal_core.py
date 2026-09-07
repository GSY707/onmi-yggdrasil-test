from __future__ import annotations

import torch

from yggdrasil_v2.r1_revalidation.h1.model import H1Config, H1LatentReasoner
from yggdrasil_v2.r1_revalidation.h1_wd_causal.core import (
    answer_margin,
    causal_transfer_targets,
    forward_route_replay,
)


def _model() -> H1LatentReasoner:
    return H1LatentReasoner(
        H1Config(
            arm="mixed",
            source_width=8,
            latent_width=8,
            K=2,
            T=2,
            recurrent_layers=2,
            num_heads=2,
            active_ffn_multiplier=3,
            shared_ffn_inner_width=12,
            routed_feature_width=12,
        )
    ).eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    return torch.randn(3, 5, 8), torch.ones(3, 5, dtype=torch.bool)


def test_route_replay_matches_normal_forward_and_returns_fixed_schedule() -> None:
    model = _model()
    hidden, mask = _inputs()
    with torch.inference_mode():
        expected = model(hidden, mask, return_trajectory=True)
        actual = forward_route_replay(model, hidden, mask, capture=True)
    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["final_state"], expected["final_state"])
    assert torch.equal(actual["trajectory"], expected["trajectory"])
    assert actual["route_schedule"].shape == (3, 2, 2)
    assert torch.equal(actual["route_schedule"], expected["route_assignments"])
    assert len(actual["captures"]) == 4
    assert actual["captures"][0].attention_state.shape == (3, 2, 8)


def test_route_replay_matches_shared_arm_too() -> None:
    config = H1Config(
        arm="shared",
        source_width=8,
        latent_width=8,
        K=2,
        T=2,
        recurrent_layers=1,
        num_heads=2,
        active_ffn_multiplier=3,
        shared_ffn_inner_width=12,
        routed_feature_width=12,
    )
    model = H1LatentReasoner(config).eval()
    hidden, mask = _inputs()
    with torch.inference_mode():
        expected = model(hidden, mask)
        actual = forward_route_replay(model, hidden, mask)
    assert torch.equal(actual["logits"], expected["logits"])


def test_global_path_disables_use_same_route_schedule() -> None:
    model = _model()
    hidden, mask = _inputs()
    baseline = forward_route_replay(model, hidden, mask)
    schedule = baseline["route_schedule"]
    common_off = forward_route_replay(
        model, hidden, mask, route_schedule=schedule, disable_common=True
    )
    projection_off = forward_route_replay(
        model, hidden, mask, route_schedule=schedule, disable_projection=True
    )
    both_off = forward_route_replay(
        model,
        hidden,
        mask,
        route_schedule=schedule,
        intervention="disable_both",
    )
    assert torch.equal(common_off["route_schedule"], schedule)
    assert torch.equal(projection_off["route_schedule"], schedule)
    assert torch.equal(both_off["route_schedule"], schedule)
    assert not torch.equal(common_off["logits"], baseline["logits"])
    assert not torch.equal(projection_off["logits"], baseline["logits"])


def test_answer_margin_excludes_true_answer_from_logsumexp() -> None:
    logits = torch.tensor([[3.0, 1.0, 2.0], [0.0, 4.0, 0.0]])
    actual = answer_margin(logits, torch.tensor([0, 1]))
    expected = torch.stack(
        (logits[0, 0] - torch.logsumexp(logits[0, 1:], 0),
         logits[1, 1] - torch.logsumexp(logits[1, [0, 2]], 0))
    )
    assert torch.allclose(actual, expected)


def test_causal_targets_are_vjp_fisher_weighted_and_capped() -> None:
    model = _model()
    hidden, mask = _inputs()
    result = causal_transfer_targets(model, hidden, mask, torch.tensor([0, 1, 2]))
    assert result["route_schedule"].shape == (3, 2, 2)
    assert result["positive_common_margin_drop"].shape == (3,)
    assert torch.all(result["requested_margin"] >= 0)
    assert result["uses_family_targets"] is False
    assert len(result["causal_transfer"]) == 4
    for transfer, diag in zip(result["causal_transfer"], result["diagnostics"]):
        assert transfer.shape == (3, 2, 8)
        assert torch.all(diag["transfer_norm"] <= diag["transfer_cap"] + 1.0e-6)
        assert torch.all(diag["fisher_share"] >= 0)
    assert torch.allclose(
        torch.stack([row["fisher_share"] for row in result["diagnostics"]], dim=1).sum(1),
        torch.ones(3),
        atol=1.0e-5,
    )
