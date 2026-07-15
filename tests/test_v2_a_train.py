from __future__ import annotations

import torch

from yggdrasil_v2.reasoning_medium.train import verifier_policy_gradient


def test_verifier_policy_gradient_is_differentiable_and_reports_step_metrics() -> None:
    torch.manual_seed(7)
    logits = torch.randn(4, 3, 3, 10, requires_grad=True)
    targets = torch.randint(0, 10, (4, 3, 3))
    step_mask = torch.tensor(
        [[True, True, True], [True, False, False], [True, True, False], [True, True, True]]
    )

    loss, stats = verifier_policy_gradient(logits, targets, step_mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert {
        "sampled_reward",
        "greedy_reward",
        "sampled_register_accuracy",
        "greedy_register_accuracy",
        "sampled_state_exact",
        "greedy_state_exact",
        "greedy_final_state_exact",
        "policy_entropy",
    } <= set(stats)
    assert all(torch.isfinite(value) for value in stats.values())

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
