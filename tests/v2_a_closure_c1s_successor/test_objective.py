from __future__ import annotations

from dataclasses import fields

import torch

from yggdrasil_v2.v2_a.closure_c1s.model import C1SConfig, C1SModel
from yggdrasil_v2.v2_a.closure_c1s_successor.objective import (
    CausalMargins,
    STATE_FEATURE_NAMES,
    SuccessorDiagnosticHeads,
    compute_training_loss,
)


def _fixture(slots: int = 2):
    config = C1SConfig(
        source_width=16,
        payload_width=16,
        address_width=8,
        slots=slots,
        operations=10,
        reader_depth=1,
        attention_heads=4,
        ffn_width=32,
        answer_classes=9,
        auxiliary_width=8,
    )
    model = C1SModel(config)
    heads = SuccessorDiagnosticHeads(16, 8, 8)
    batch = 3
    active = torch.zeros(batch, 10, dtype=torch.bool)
    active[:, :slots] = True
    target = torch.full((batch, 10), -1, dtype=torch.long)
    source = torch.full((batch, 10), -1, dtype=torch.long)
    for step in range(slots):
        target[:, step] = step
        source[:, step] = (step - 1) % slots
    values = torch.zeros(batch, 10, slots, len(STATE_FEATURE_NAMES))
    values[:, :slots, :, 1] = 1
    state_mask = active.unsqueeze(-1).unsqueeze(-1).expand_as(values)
    return model, heads, {
        "source_hidden": torch.randn(batch, 6, 16),
        "source_mask": torch.ones(batch, 6, dtype=torch.bool),
        "answers": torch.zeros(batch, dtype=torch.long),
        # ERE entities and CPS decision objects deliberately have no answer label.
        "object_labels": torch.full((batch, slots), -1, dtype=torch.long),
        "operation_active": active,
        "source_owner": source,
        "target_owner": target,
        "query_owner": torch.zeros(batch, dtype=torch.long),
        "presence": torch.ones(batch, slots, dtype=torch.bool),
        "content_mask": torch.ones(batch, slots, dtype=torch.bool),
        "span_start": torch.zeros(batch, slots, dtype=torch.long),
        "span_end": torch.ones(batch, slots, dtype=torch.long),
        "state_values": values,
        "state_feature_mask": state_mask,
        "state_step_mask": active,
        "alternate_owner": torch.full((batch,), 1 if slots > 1 else -1, dtype=torch.long),
        "alternate_label": torch.full((batch,), -1, dtype=torch.long),
        "irrelevant_owner": torch.full((batch,), -1, dtype=torch.long),
        "mechanism_supported": torch.ones(batch, dtype=torch.bool),
        "query_answer_independent": torch.ones(batch, dtype=torch.bool),
    }


def test_external_targets_produce_finite_causal_loss_and_temporal_state_mask() -> None:
    model, heads, batch = _fixture()
    total, parts, outputs = compute_training_loss(model, heads, batch, causal=True)
    assert torch.isfinite(total)
    assert outputs["state_logits"].shape == (3, 10, 2, 8)
    assert all(value >= 0.0 for value in parts.values())
    assert parts["slot_identity"] == 0.0
    assert "base_answer_margin" in outputs["intervention"]
    assert "no_core_answer_margin" in outputs["intervention"]
    assert not any(name.endswith("correct_logit") for name in outputs["intervention"])
    total.backward()
    assert model.boundary.source_projection.weight.grad is not None


def test_causal_margin_contract_names_answer_margin_not_raw_correct_logit() -> None:
    names = {field.name for field in fields(CausalMargins)}
    assert {"no_core_answer_margin", "wrong_start_answer_margin", "relevant_answer_margin"} <= names
    assert not {"no_core_correct_logit", "wrong_start_correct_logit", "relevant_correct_logit"} & names


def test_k1_uses_no_fake_multi_owner_loss() -> None:
    model, heads, batch = _fixture(slots=1)
    total, parts, _ = compute_training_loss(model, heads, batch, causal=True)
    assert torch.isfinite(total)
    assert parts["slot_identity"] == 0.0
    assert parts["wrong_start_margin"] == 0.0
    assert parts["target_shuffle_margin"] == 0.0
    assert parts["relevant_margin"] == 0.0
    assert parts["irrelevant_consistency"] == 0.0
    assert "query_swap" not in parts


def test_behavior_only_all_ignore_batch_keeps_answer_loss_but_zeroes_mechanism_loss() -> None:
    model, heads, batch = _fixture(slots=2)
    batch["mechanism_supported"] = torch.zeros(3, dtype=torch.bool)
    batch["object_labels"].fill_(-1)
    batch["operation_active"].zero_()
    batch["source_owner"].fill_(-1)
    batch["target_owner"].fill_(-1)
    batch["content_mask"].zero_()
    batch["state_feature_mask"].zero_()
    batch["state_step_mask"].zero_()
    total, parts, _ = compute_training_loss(model, heads, batch, causal=True)
    assert torch.isfinite(total)
    assert parts["answer"] > 0.0
    mechanism_parts = set(parts) - {"answer", "total"}
    assert all(parts[name] == 0.0 for name in mechanism_parts)
