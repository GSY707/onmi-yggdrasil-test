from __future__ import annotations

import torch

from yggdrasil_v2.reasoning_medium.a1_5_model import P0Config
from yggdrasil_v2.reasoning_medium.a1_5_p1 import P1HiddenLatentReasoner, token_operation_masks
from yggdrasil_v2.reasoning_medium.a1_5_p2 import LearnedMultiSlotWorkspace, P2Config
from yggdrasil_v2.reasoning_medium.a1_5_text_baseline import parse_a15_text


def test_p1_operation_span_mask_and_warmup_interfaces() -> None:
    offsets = [[0, 5], [5, 9], [10, 15], [16, 22]]
    masks = token_operation_masks([{"char_start": 10, "char_end": 22}], offsets)
    assert masks.shape == (1, 4)
    assert masks.tolist() == [[False, False, True, True]]

    model = P1HiddenLatentReasoner(
        source_width=12,
        config=P0Config(latent_width=32, attention_heads=4, ffn_width=64),
    )
    source = torch.randn(2, 7, 12)
    source_mask = torch.ones(2, 7, dtype=torch.bool)
    op_mask = torch.zeros(2, 2, 7, dtype=torch.bool)
    op_mask[:, 0, :3] = True
    op_mask[:, 1, 3:6] = True
    step_mask = torch.ones(2, 2, dtype=torch.bool)
    output = model(source, source_mask, op_mask, step_mask, return_trajectory=True)
    assert output["logits"].shape == (2, 10)
    assert output["start_state_logits"].shape == (2, 3, 10)
    assert output["query_register_logits"].shape == (2, 3)
    assert output["operation_family_logits"].shape == (2, 2, 3)
    assert model.integrity_report()["source_mean_broadcast_main_path"] is False


def test_p2_is_unbound_learned_multi_slot_and_has_permutation_probe() -> None:
    model = LearnedMultiSlotWorkspace(
        operation_width=12,
        config=P2Config(learned_slots=8, latent_width=32, attention_heads=4, ffn_width=64),
        source_width=12,
    )
    operation_hidden = torch.randn(2, 3, 12)
    operation_mask = torch.ones(2, 3, dtype=torch.bool)
    source_hidden = torch.randn(2, 6, 12)
    source_mask = torch.ones(2, 6, dtype=torch.bool)
    output = model(
        operation_hidden,
        operation_mask,
        source_hidden=source_hidden,
        source_attention_mask=source_mask,
        return_trajectory=True,
    )
    assert output["logits"].shape == (2, 10)
    assert len(output["trajectory"]) == 3
    probe = model.permutation_probe(
        operation_hidden,
        operation_mask,
        source_hidden=source_hidden,
        source_attention_mask=source_mask,
    )
    assert "logit_mean_abs_delta" in probe
    assert model.integrity_report()["explicit_register_slot_binding"] is False


def test_a15_text_parser_scores_final_and_each_state() -> None:
    answer, states = parse_a15_text(
        "Start: amber=A, cobalt=B, jade=C.\n"
        "Step 1: amber=A, cobalt=A, jade=C.\n"
        "Step 2: amber=A, cobalt=A, jade=A.\n"
        "FINAL: A"
    )
    assert answer == "A"
    assert states[1] == ["A", "A", "C"]
    assert states[2] == ["A", "A", "A"]

    answer, states = parse_a15_text(
        "State: amber=A, cobalt=A, jade=C.\n"
        "State: amber=A, cobalt=A, jade=A."
    )
    assert answer is None
    assert states[2] == ["A", "A", "A"]
