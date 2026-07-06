from __future__ import annotations

import torch

from experiments.omni_transformer_stage_avj_latent_reasoning_core import (
    FAMILIES,
    StageAVJConfig,
    StageAVJModel,
    build_dataset,
    metrics,
    target_record_loss,
    total_loss,
)


def test_avj_dataset_contains_real_record_edits() -> None:
    config = StageAVJConfig(train_size=18, val_size=6, test_size=6, heldout_size=6, max_objects=6)
    data = build_dataset("train", 18, config)
    changed = (
        (data.source_active != data.target_active)
        | (data.source_color != data.target_color)
        | (data.source_shape != data.target_shape)
        | (data.source_row != data.target_row)
        | (data.source_col != data.target_col)
    )
    assert changed.any(dim=1).all()
    assert set(data.family.tolist()) == set(range(len(FAMILIES)))
    assert (data.edit_slot >= 0).all()
    assert (data.read_a >= 0).all()


def test_avj_model_forward_loss_and_metrics() -> None:
    config = StageAVJConfig(
        train_size=8,
        val_size=4,
        test_size=4,
        heldout_size=4,
        batch_size=4,
        d_model=32,
        layers=1,
        heads=4,
        codec_steps=1,
        operation_steps=1,
        process_steps=1,
        joint_steps=1,
        amp=False,
        gpu_resident_data=False,
    )
    batch = build_dataset("train", 8, config).subset([0, 1, 2, 3])
    model = StageAVJModel(config)
    outputs = model(batch)
    loss = total_loss(outputs, batch, config, "joint")
    assert torch.isfinite(loss)
    assert torch.isfinite(target_record_loss(outputs, batch))
    row = metrics(outputs, batch)
    for key in (
        "source_record_exact",
        "target_record_exact",
        "read_a_accuracy",
        "edit_slot_accuracy",
        "answer_sequence_exact",
    ):
        assert key in row
