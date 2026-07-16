import inspect

import torch

from yggdrasil_v2.reasoning_medium.a1_7_core import A17CoreConfig, A17RelationAddressedCore
from yggdrasil_v2.reasoning_medium.a1_9_model import module_state_sha256
from yggdrasil_v2.reasoning_medium.a1_10_train import compute_a110_loss, encode_a110_targets
from yggdrasil_v2.reasoning_medium.a1_11_models import (
    A111BoundaryConfig,
    A111LearnedFullTextBoundary,
    A111OracleRoleReasoner,
    A111ReasonerConfig,
)
from yggdrasil_v2.reasoning_medium.a1_11_assessment import _classification


def test_a111_boundary_removes_only_spans_and_freezes_structured_core() -> None:
    core = A17RelationAddressedCore(A17CoreConfig(latent_width=32, ffn_width=64))
    core_hash = module_state_sha256(core)
    model = A111LearnedFullTextBoundary(
        A111BoundaryConfig(
            source_width=16,
            latent_width=32,
            attention_heads=4,
            ffn_width=64,
            reader_layers=2,
        ),
        core,
    )
    signature = inspect.signature(model.forward)
    assert "source_hidden" in signature.parameters
    assert "source_attention_mask" in signature.parameters
    assert "operation_mask" in signature.parameters
    assert "span_mask" not in signature.parameters
    assert "offset_mapping" not in signature.parameters

    output = model(
        torch.randn(2, 24, 16),
        torch.ones(2, 24, dtype=torch.bool),
        torch.tensor([[True, True, True, False], [True, True, True, True]]),
    )
    output["state_logits"].sum().backward()
    assert output["mapped_start_values"].shape == (2, 3, 32)
    assert output["mapped_family"].shape == (2, 4, 32)
    assert any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith("core."))
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.core.parameters())
    assert module_state_sha256(model.core) == core_hash
    integrity = model.integrity_report()
    assert integrity["passed"]
    assert not integrity["oracle_span_input"]
    assert integrity["operation_mask_retained_for_isolation"]
    assert integrity["structured_core_retained"]


def test_a111_reasoner_uses_exact_symbolic_roles_but_has_no_structured_core() -> None:
    model = A111OracleRoleReasoner(
        A111ReasonerConfig(
            latent_width=32,
            workspace_slots=8,
            attention_heads=4,
            ffn_width=64,
            recurrent_layers=2,
        )
    )
    batch, steps = 2, 4
    output = model(
        start_values=torch.randint(0, 10, (batch, 3)),
        query_register=torch.randint(0, 3, (batch,)),
        operation_family=torch.randint(0, 2, (batch, steps)),
        operation_source=torch.randint(0, 3, (batch, steps)),
        operation_target=torch.randint(0, 3, (batch, steps)),
        operation_mask=torch.tensor([[True, True, True, False], [True, True, True, True]]),
        recurrent_steps=16,
    )
    loss = output["state_logits"].sum() + output["answer_logits"].sum()
    loss.backward()
    assert output["trajectory"].shape == (batch, 16, 8, 32)
    assert output["state_logits"].shape == (batch, 16, 3, 10)
    assert any(parameter.grad is not None for parameter in model.reasoner.parameters())
    assert not hasattr(model, "core")
    assert not hasattr(model, "mapper")
    integrity = model.integrity_report()
    assert integrity["passed"]
    assert integrity["exact_symbolic_oracle_roles"]
    assert not integrity["qwen_or_boundary_adapter_loaded"]
    assert not integrity["structured_a1_8_core_loaded"]
    assert not integrity["explicit_register_workspace"]


def test_a111_reasoner_uses_a110_fixed_budget_targets() -> None:
    record = {
        "start_state": {"amber": "A", "cobalt": "B", "jade": "C"},
        "state_trajectory": [{"amber": "B", "cobalt": "B", "jade": "C"}],
        "query_register": "amber",
        "answer_index": 1,
    }
    labels = encode_a110_targets([record], 16, "cpu")
    output = {
        "state_logits": torch.randn(1, 16, 3, 10, requires_grad=True),
        "answer_logits": torch.randn(1, 10, requires_grad=True),
    }
    loss, metrics = compute_a110_loss(output, labels)
    loss.backward()
    assert labels["state_targets"].shape == (1, 16, 3)
    assert torch.equal(labels["state_targets"][:, 0], labels["state_targets"][:, -1])
    assert metrics["state_loss_weight"] == 1.0
    assert metrics["answer_loss_weight"] == 1.0


def test_a111_matrix_classification_does_not_confuse_interaction_and_main_effects() -> None:
    assert _classification(3, 3)["code"] == "joint_interface_or_optimization_interaction"
    assert _classification(0, 3)["code"] == "full_text_boundary_primary_failure"
    assert _classification(3, 0)["code"] == "anonymous_reasoner_primary_failure"
    assert _classification(0, 0)["code"] == "both_isolated_replacements_fail"
    assert not _classification(2, 3)["conclusive"]
    assert (
        _classification(0, 0, boundary_overfit=False, reasoner_overfit=True)["code"]
        == "both_isolated_arms_fail_at_independent_gates"
    )
