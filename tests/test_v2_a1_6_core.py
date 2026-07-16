import inspect

import torch

from yggdrasil_v2.reasoning_medium.a1_6_core import A16CoreConfig, RelationAddressedCore, RelationAddressedTransition


def _inputs(ops: list[tuple[int, int, int]], query: int = 2) -> dict[str, torch.Tensor]:
    return {
        "start_values": torch.tensor([[0, 1, 2]]),
        "query_register": torch.tensor([query]),
        "operation_family": torch.tensor([[op[0] for op in ops]]),
        "operation_source": torch.tensor([[op[1] for op in ops]]),
        "operation_target": torch.tensor([[op[2] for op in ops]]),
        "operation_mask": torch.ones(1, len(ops), dtype=torch.bool),
    }


def test_a16_core_completeness_contract() -> None:
    model = RelationAddressedCore(A16CoreConfig(latent_width=32, ffn_width=64))
    same_start_a = model(**_inputs([(0, 0, 1), (1, 1, 2)]))
    same_start_b = model(**_inputs([(1, 0, 2), (0, 2, 1)]))
    assert torch.equal(same_start_a["initial_slots"], same_start_b["initial_slots"])

    future_a = model(**_inputs([(0, 0, 1), (1, 1, 2), (0, 2, 0)]))
    future_b = model(**_inputs([(0, 0, 1), (0, 2, 0), (0, 1, 2)]))
    assert torch.equal(future_a["trajectory"][:, 0], future_b["trajectory"][:, 0])
    changed = model(**_inputs([(0, 0, 1), (0, 2, 1), (0, 2, 0)]))
    assert torch.equal(future_a["trajectory"][:, 0], changed["trajectory"][:, 0])
    assert not torch.equal(future_a["trajectory"][:, 1], changed["trajectory"][:, 1])

    t0 = model(
        torch.tensor([[0, 1, 2]]), torch.tensor([2]), torch.empty((1, 0), dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long), torch.empty((1, 0), dtype=torch.long), torch.empty((1, 0), dtype=torch.bool)
    )
    assert torch.equal(t0["final_slots"], t0["initial_slots"])
    assert torch.allclose(t0["answer_logits"], model.shared_state_head(t0["initial_slots"][:, 2]))

    query_a = model(**_inputs([(0, 0, 1), (1, 1, 2)], query=0))
    query_b = model(**_inputs([(0, 0, 1), (1, 1, 2)], query=2))
    assert torch.equal(query_a["trajectory"], query_b["trajectory"])
    assert not torch.equal(query_a["answer_logits"], query_b["answer_logits"])

    assert sum(isinstance(module, RelationAddressedTransition) for module in model.modules()) == 1
    assert not any("answer_head" in name for name, _ in model.named_parameters())
    assert not any("answer_head" in name for name, _ in model.named_modules())
    assert model.integrity_report()["independent_answer_head"] is False
    assert inspect.signature(RelationAddressedTransition.forward).parameters.keys() >= {"family_latent", "source_role_latent", "target_role_latent"}
    assert torch.allclose(query_a["source_pointer_weights"].sum(-1), torch.ones(1, 2))
    assert torch.allclose(query_a["target_pointer_weights"].sum(-1), torch.ones(1, 2))


def test_a16_family_source_target_are_separate_inputs() -> None:
    model = RelationAddressedCore(A16CoreConfig(latent_width=32, ffn_width=64))
    assert list(inspect.signature(model.transition.forward).parameters)[:4] == ["family_latent", "source_role_latent", "target_role_latent", "slots"]
    assert model.integrity_report()["initial_state_operation_free"]

