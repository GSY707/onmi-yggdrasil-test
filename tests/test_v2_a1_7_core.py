import inspect

import torch

from yggdrasil_v2.reasoning_medium.a1_7_core import (
    A17CoreConfig,
    A17RelationAddressedCore,
    A17RelationTransition,
)
from yggdrasil_v2.reasoning_medium.a1_7_train import closure_targets, compute_a17_loss


def _inputs(ops: list[tuple[int, int, int]], query: int = 2) -> dict[str, torch.Tensor]:
    return {
        "start_values": torch.tensor([[0, 1, 2]]),
        "query_register": torch.tensor([query]),
        "operation_family": torch.tensor([[op[0] for op in ops]]),
        "operation_source": torch.tensor([[op[1] for op in ops]]),
        "operation_target": torch.tensor([[op[2] for op in ops]]),
        "operation_mask": torch.ones(1, len(ops), dtype=torch.bool),
    }


def test_a17_address_content_separation_and_integrity() -> None:
    model = A17RelationAddressedCore(A17CoreConfig(latent_width=32, ffn_width=64))
    starts = torch.tensor([[0, 1, 2]])
    assert torch.equal(model.initial_slots(starts), model.value_embedding(starts))
    assert not torch.equal(model.initial_slots(starts), model.value_embedding(starts) + model.register_keys.unsqueeze(0))
    same_start_a = model(**_inputs([(0, 0, 1), (1, 1, 2)]))
    same_start_b = model(**_inputs([(1, 0, 2), (0, 2, 1)]))
    assert torch.equal(same_start_a["initial_slots"], same_start_b["initial_slots"])
    assert sum(isinstance(module, A17RelationTransition) for module in model.modules()) == 1
    assert not any("answer_head" in name for name, _ in model.named_modules())
    report = model.integrity_report()
    assert report["state_slots_content_only"]
    assert report["register_keys_addressing_only"]
    assert report["hard_reembedding_in_forward"] is False
    assert "argmax" not in inspect.getsource(A17RelationAddressedCore.forward)

    mixed = A17RelationAddressedCore(
        A17CoreConfig(latent_width=32, ffn_width=64, address_content_mixing_ablation=True)
    )
    assert torch.equal(
        mixed.initial_slots(starts),
        mixed.value_embedding(starts) + mixed.register_keys.unsqueeze(0),
    )
    mixed_report = mixed.integrity_report()
    assert mixed_report["address_content_mixing_ablation"] is True
    assert mixed_report["target_architecture"] is False


def test_a17_query_is_read_only_variable_length_identity_and_t0() -> None:
    model = A17RelationAddressedCore(A17CoreConfig(latent_width=32, ffn_width=64))
    query_a = model(**_inputs([(0, 0, 1), (1, 1, 2)], query=0))
    query_b = model(**_inputs([(0, 0, 1), (1, 1, 2)], query=2))
    assert torch.equal(query_a["trajectory"], query_b["trajectory"])
    assert torch.allclose(query_a["answer_logits"], query_a["final_state_logits"][:, 0], atol=0, rtol=0)
    assert torch.allclose(query_b["answer_logits"], query_b["final_state_logits"][:, 2], atol=0, rtol=0)

    variable = {
        "start_values": torch.tensor([[0, 1, 2], [2, 1, 0]]),
        "query_register": torch.tensor([0, 2]),
        "operation_family": torch.tensor([[0, 1], [1, 0]]),
        "operation_source": torch.tensor([[0, 1], [2, 0]]),
        "operation_target": torch.tensor([[1, 2], [1, 2]]),
        "operation_mask": torch.tensor([[True, False], [True, True]]),
    }
    out = model(**variable)
    gathered = torch.stack((out["state_logits"][0, 0, 0], out["state_logits"][1, 1, 2]))
    assert torch.allclose(out["answer_logits"], gathered, atol=1e-6, rtol=0)

    t0 = model(
        torch.tensor([[0, 1, 2]]),
        torch.tensor([2]),
        torch.empty((1, 0), dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
        torch.empty((1, 0), dtype=torch.bool),
    )
    assert torch.equal(t0["final_slots"], t0["initial_slots"])
    assert torch.allclose(t0["answer_logits"], model.shared_state_head(t0["initial_slots"][:, 2]), atol=1e-6, rtol=0)


def test_a17_closure_target_is_stop_gradient() -> None:
    model = A17RelationAddressedCore(A17CoreConfig(latent_width=32, ffn_width=64))
    targets = closure_targets(model, torch.tensor([[[0, 1, 2]]]))
    assert not targets.requires_grad
    assert targets.grad_fn is None


def test_a17_closure_weight_zero_is_a_true_loss_ablation() -> None:
    model = A17RelationAddressedCore(A17CoreConfig(latent_width=32, ffn_width=64))
    inputs = _inputs([(0, 0, 1), (1, 1, 2)])
    output = model(**inputs)
    batch = dict(inputs)
    batch["state_targets"] = torch.tensor([[[0, 0, 2], [0, 2, 0]]])
    batch["answer_targets"] = torch.tensor([0])
    with_closure, metrics = compute_a17_loss(model, output, batch, closure_weight=1.0)
    without_closure, zero_metrics = compute_a17_loss(model, output, batch, closure_weight=0.0)
    assert torch.allclose(with_closure - without_closure, with_closure.new_tensor(metrics["closure_loss"]), atol=1e-6)
    assert zero_metrics["closure_weight"] == 0.0
