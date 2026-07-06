from __future__ import annotations

import torch

from experiments.omni_transformer_stage_avjb_trace_verifier import (
    FAMILIES,
    StageAVJBConfig,
    StageAVJBModel,
    changed_slot_mask,
    count_value_target,
    metrics,
    rich_trace_loss,
    same_row_candidate_mask,
    total_loss,
)
from experiments.omni_transformer_stage_avj_latent_reasoning_core import build_dataset


def test_avjb_rich_trace_targets_are_nontrivial() -> None:
    config = StageAVJBConfig(train_size=18, val_size=6, test_size=6, heldout_size=6, max_objects=6)
    data = build_dataset("train", 18, config)
    candidates = same_row_candidate_mask(data)
    changed = changed_slot_mask(data)
    counts = count_value_target(data)
    same_row = data.family == FAMILIES.index("same_row_move")
    assert candidates[same_row].sum() > 0
    assert changed.sum() >= data.prompts.shape[0]
    assert int(counts.max()) >= 1


def test_avjb_model_forward_rich_loss_and_metrics() -> None:
    config = StageAVJBConfig(
        train_size=8,
        val_size=4,
        test_size=4,
        heldout_size=4,
        batch_size=4,
        d_model=32,
        layers=1,
        heads=4,
        codec_steps=1,
        trace_steps=1,
        target_steps=1,
        joint_steps=1,
        amp=False,
        gpu_resident_data=False,
    )
    batch = build_dataset("train", 8, config).subset([0, 1, 2, 3])
    model = StageAVJBModel(config)
    outputs = model(batch)
    assert torch.isfinite(rich_trace_loss(outputs, batch))
    assert torch.isfinite(total_loss(outputs, batch, config, "joint"))
    row = metrics(outputs, batch)
    for key in (
        "candidate_mask_exact",
        "count_value_accuracy",
        "copy_gate_accuracy",
        "target_color_accuracy",
        "target_record_exact",
    ):
        assert key in row
