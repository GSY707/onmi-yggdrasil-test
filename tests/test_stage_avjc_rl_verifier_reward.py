from __future__ import annotations

import torch

from experiments.omni_transformer_stage_avj_latent_reasoning_core import build_dataset
from experiments.omni_transformer_stage_avjc_rl_verifier_reward import (
    StageAVJCConfig,
    rl_verifier_loss,
    stage_name,
    total_loss,
)
from experiments.omni_transformer_stage_avjb_trace_verifier import StageAVJBModel


def test_avjc_stage_schedule_uses_rl_verifier_phase() -> None:
    config = StageAVJCConfig(codec_steps=2, trace_steps=3, target_steps=4, rl_steps=5, joint_steps=0)
    assert stage_name(config, 1) == "codec"
    assert stage_name(config, 3) == "trace_sft"
    assert stage_name(config, 8) == "target_sft"
    assert stage_name(config, 10) == "rl_verifier"
    assert config.total_steps == 14


def test_avjc_rl_verifier_loss_backpropagates_policy_signal() -> None:
    torch.manual_seed(7)
    config = StageAVJCConfig(
        train_size=12,
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
        rl_steps=1,
        joint_steps=0,
        amp=False,
        gpu_resident_data=False,
    )
    batch = build_dataset("train", 12, config).subset([0, 1, 2, 3])
    model = StageAVJBModel(config)
    outputs = model(batch)
    loss, stats = rl_verifier_loss(outputs, batch, config)
    assert torch.isfinite(loss)
    assert "rl_sample_reward" in stats
    assert "rl_entropy" in stats
    loss.backward()
    assert model.target_active_head.weight.grad is not None


def test_avjc_total_loss_returns_rl_stats() -> None:
    config = StageAVJCConfig(
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
        rl_steps=1,
        joint_steps=0,
        amp=False,
        gpu_resident_data=False,
    )
    batch = build_dataset("train", 8, config).subset([0, 1, 2, 3])
    outputs = StageAVJBModel(config)(batch)
    loss, stats = total_loss(outputs, batch, config, "rl_verifier")
    assert torch.isfinite(loss)
    assert stats["rl_sample_reward"] >= 0.0
