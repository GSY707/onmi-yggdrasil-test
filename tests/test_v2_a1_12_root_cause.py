import torch

from yggdrasil_v2.reasoning_medium.a1_10_model import A110RecurrentBlock
from yggdrasil_v2.reasoning_medium.a1_12_models import (
    A112ReasonerConfig,
    A112RootCauseReasoner,
)
from yggdrasil_v2.reasoning_medium.a1_12_assessment import classify_root_cause


def _inputs(batch: int = 2, steps: int = 4) -> dict[str, torch.Tensor]:
    return {
        "start_values": torch.tensor([[0, 1, 2], [3, 4, 5]])[:batch],
        "query_register": torch.tensor([0, 2])[:batch],
        "operation_family": torch.randint(0, 2, (batch, steps)),
        "operation_source": torch.randint(0, 3, (batch, steps)),
        "operation_target": torch.randint(0, 3, (batch, steps)),
        "operation_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )[:batch, :steps],
    }


def test_a112_matrix_changes_only_binding_and_cursor_contracts() -> None:
    configurations = {
        "BIND": (True, False),
        "CURSOR": (False, True),
        "BOTH": (True, True),
    }
    for arm, (entity, cursor) in configurations.items():
        model = A112RootCauseReasoner(
            A112ReasonerConfig(
                entity_addressable=entity,
                aligned_operation_cursor=cursor,
                latent_width=32,
                attention_heads=4,
                ffn_width=64,
            )
        )
        report = model.integrity_report()
        assert report["passed"]
        assert report["arm"] == arm
        assert report["entity_addressable_state"] is entity
        assert report["aligned_operation_cursor"] is cursor
        assert not report["task_specific_transition"]
        assert not report["qwen_or_boundary_adapter_loaded"]
        assert not report["structured_core_loaded"]
        assert sum(isinstance(module, A110RecurrentBlock) for module in model.modules()) == 2


def test_a112_entity_contract_has_direct_state_slots_and_shared_readout() -> None:
    model = A112RootCauseReasoner(
        A112ReasonerConfig(
            entity_addressable=True,
            aligned_operation_cursor=False,
            latent_width=32,
            attention_heads=4,
            ffn_width=64,
        )
    )
    output = model(**_inputs(), recurrent_steps=16)
    assert output["trajectory"].shape == (2, 16, 8, 32)
    assert output["state_logits"].shape == (2, 16, 3, 10)
    assert output["answer_logits"].shape == (2, 10)
    assert model.entity_initialization is not None
    assert model.state_head.out_features == 10
    output["state_logits"].sum().backward()
    assert model.entity_initialization.weight.grad is not None


def test_a112_anonymous_cursor_retains_mean_state_readout_but_aligns_source() -> None:
    model = A112RootCauseReasoner(
        A112ReasonerConfig(
            entity_addressable=False,
            aligned_operation_cursor=True,
            latent_width=32,
            attention_heads=4,
            ffn_width=64,
        )
    )
    inputs = _inputs()
    typed, valid = model._typed_source(**inputs)
    source = model.read_source_norm(model.source_projection(model.source_input_norm(typed)))
    selected0, padding0 = model._step_source(source, valid, 0, inputs["operation_mask"].shape[1])
    selected5, padding5 = model._step_source(source, valid, 5, inputs["operation_mask"].shape[1])
    assert selected0.shape[1] == 4
    assert padding0.shape == (2, 4)
    assert selected5.shape[1] == 1
    assert not padding5.any()
    assert model.entity_initialization is None
    assert model.state_head.out_features == 30
    output = model(**inputs, recurrent_steps=16)
    assert output["state_logits"].shape == (2, 16, 3, 10)


def test_a112_start_state_permutation_is_a_real_input_intervention() -> None:
    inputs = _inputs()
    entity = A112RootCauseReasoner(
        A112ReasonerConfig(
            entity_addressable=True,
            aligned_operation_cursor=True,
            latent_width=32,
            attention_heads=4,
            ffn_width=64,
        )
    ).eval()
    anonymous = A112RootCauseReasoner(
        A112ReasonerConfig(
            entity_addressable=False,
            aligned_operation_cursor=True,
            latent_width=32,
            attention_heads=4,
            ffn_width=64,
        )
    ).eval()
    permutation = torch.tensor([2, 1, 0])
    with torch.no_grad():
        standard_entity = entity(**inputs, recurrent_steps=2)["state_logits"]
        permuted_entity = entity(
            **inputs, recurrent_steps=2, start_value_permutation=permutation
        )["state_logits"]
        standard_anonymous = anonymous(**inputs, recurrent_steps=2)["state_logits"]
        permuted_anonymous = anonymous(
            **inputs, recurrent_steps=2, start_value_permutation=permutation
        )["state_logits"]
    assert not torch.equal(standard_entity, permuted_entity)
    assert not torch.equal(standard_anonymous, permuted_anonymous)


def test_a112_root_cause_classification_is_strictly_three_seed() -> None:
    assert classify_root_cause(3, 0, 3)["code"] == "entity_addressability_primary"
    assert classify_root_cause(0, 3, 3)["code"] == "operation_step_alignment_primary"
    assert classify_root_cause(3, 3, 3)["code"] == "two_independent_sufficient_constraints"
    assert classify_root_cause(0, 0, 3)["code"] == "binding_cursor_joint_requirement"
    assert classify_root_cause(0, 0, 0)["code"] == "binding_and_cursor_insufficient"
    assert not classify_root_cause(2, 0, 3)["localized"]
