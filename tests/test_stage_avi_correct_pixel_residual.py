from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.omni_transformer_stage_avi_whole_image_latent_capacity import (
    StageAVIConfig,
    compact_prefix_loss,
    image_metrics,
    residual_transition_loss,
)


def toy_rgba() -> torch.Tensor:
    image = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
    image[:, 0, 1:3, 1:3] = 1.0
    image[:, 3, 1:3, 1:3] = 1.0
    return image


def test_correct_pixel_compact_rewards_exact_foreground() -> None:
    config = StageAVIConfig()
    target = toy_rgba()
    loss, metrics = compact_prefix_loss(target, target, config, precision_power=5.0, color_tau=0.02)
    eval_metrics = image_metrics(target, target, config)

    assert loss.item() < 0.01
    assert metrics["compact_correct_pixel_rate"].item() > 0.99
    assert metrics["compact_wrong_background_rate"].item() < 0.01
    assert metrics["compact_wrong_color_rate"].item() < 0.01
    assert metrics["compact_missed_foreground_rate"].item() < 0.01
    assert eval_metrics["hard_correct_pixel_rate"] == 1.0
    assert eval_metrics["hard_wrong_background_rate"] == 0.0
    assert eval_metrics["hard_wrong_color_rate"] == 0.0
    assert eval_metrics["hard_missed_foreground_rate"] == 0.0


def test_residual_transition_penalizes_changing_correct_pixels() -> None:
    config = StageAVIConfig()
    target = toy_rgba()
    clean_loss, clean_metrics = residual_transition_loss(target, target, target, config, precision_power=5.0, color_tau=0.02)
    damaged = target.clone()
    damaged[:, 0, 0, 0] = 1.0
    damaged[:, 3, 0, 0] = 1.0
    damaged_loss, damaged_metrics = residual_transition_loss(target, damaged, target, config, precision_power=5.0, color_tau=0.02)

    assert clean_loss.item() < damaged_loss.item()
    assert damaged_metrics["residual_preserve_loss"].item() > clean_metrics["residual_preserve_loss"].item()
    assert damaged_metrics["residual_delta_outside_loss"].item() > clean_metrics["residual_delta_outside_loss"].item()
