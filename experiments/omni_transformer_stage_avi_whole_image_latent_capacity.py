from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omni_transformer_stage_avh_text_anchored_visual_edit_dataset import (  # noqa: E402
    IMAGE_SIZE,
    render_records,
)


BACKGROUND = 0.06
ALPHA_EPS = 1e-6


@dataclass(frozen=True)
class StageAVIConfig:
    dataset_manifest: str = "artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_dataset/dataset_10m/manifest.json"
    output: str = "artifacts/omni_transformer_stage_avi_whole_image_latent_capacity/smoke_result.json"
    latent_tokens_list: str = "12,24,32"
    d_model: int = 192
    latent_dim: int = 192
    encoder_width: int = 192
    decoder_width: int = 768
    steps: int = 3000
    batch_size: int = 128
    lr: float = 1e-3
    prefix_weight: float = 0.35
    rgb_weight: float = 1.0
    alpha_weight: float = 1.0
    background_alpha_weight: float = 0.35
    monotonic_weight: float = 0.02
    compact_prefix_weight: float = 0.35
    compact_correct_weight: float = 1.0
    compact_wrong_bg_weight: float = 0.75
    compact_wrong_color_weight: float = 0.75
    compact_missed_fg_weight: float = 0.35
    compact_precision_power_start: float = 0.5
    compact_precision_power_end: float = 5.0
    compact_precision_curve_power: float = 0.5
    correct_pixel_color_tau_start: float = 0.08
    correct_pixel_color_tau_end: float = 0.02
    correct_pixel_color_tau_curve_power: float = 0.5
    correct_pixel_rgb_threshold: float = 0.08
    correct_pixel_alpha_threshold: float = 0.5
    residual_prefix_weight: float = 0.12
    residual_preserve_weight: float = 0.45
    residual_delta_weight: float = 0.25
    residual_needed_weight: float = 0.30
    residual_weight_start: float = 0.15
    residual_weight_end: float = 1.0
    residual_curve_power: float = 0.5
    train_limit: int = 0
    val_limit: int = 1300
    test_limit: int = 1300
    heldout_limit: int = 1300
    eval_batch_size: int = 128
    greedy_eval_limit: int = 512
    sample_count: int = 4
    seed: int = 20260705
    amp: bool = True
    gpu_resident_data: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class RecordImageSet:
    records: torch.Tensor
    case_ids: list[str]

    def __len__(self) -> int:
        return int(self.records.shape[0])

    def slice(self, start: int, end: int) -> "RecordImageSet":
        return RecordImageSet(self.records[start:end], self.case_ids[start:end])

    def tensor_subset(self, indices: torch.Tensor) -> "RecordImageSet":
        return RecordImageSet(self.records.index_select(0, indices), [])

    def to(self, device: torch.device) -> "RecordImageSet":
        return RecordImageSet(self.records.to(device=device, dtype=torch.long, non_blocking=True), self.case_ids)


def parse_latent_list(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError("--latent-tokens-list must contain positive integers")
    return values


def prefix_lengths(latent_tokens: int) -> list[int]:
    values = {1, latent_tokens}
    power = 2
    while power < latent_tokens:
        values.add(power)
        power *= 2
    return sorted(values)


def curriculum_segments(latent_tokens: int, total_steps: int) -> list[tuple[int, int]]:
    prefixes = prefix_lengths(latent_tokens)
    base = total_steps // len(prefixes)
    remainder = total_steps % len(prefixes)
    segments = []
    start = 1
    for index, active_tokens in enumerate(prefixes):
        steps = base + (1 if index < remainder else 0)
        end = start + steps - 1
        segments.append((active_tokens, end))
        start = end + 1
    return segments


def active_tokens_for_step(latent_tokens: int, total_steps: int, step: int) -> int:
    for active_tokens, end in curriculum_segments(latent_tokens, total_steps):
        if step <= end:
            return active_tokens
    return latent_tokens


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "AV-H":
        raise ValueError(f"expected AV-H manifest, got {manifest.get('stage')!r}")
    return manifest


def find_split(manifest: dict[str, object], split: str) -> dict[str, object]:
    for row in manifest.get("splits", []):
        if isinstance(row, dict) and row.get("split") == split:
            return row
    raise ValueError(f"missing split {split!r}")


def load_split(manifest_path: Path, manifest: dict[str, object], split: str, limit: int = 0) -> RecordImageSet:
    split_info = find_split(manifest, split)
    base = manifest_path.parent
    records = []
    case_ids: list[str] = []
    loaded = 0
    for shard in split_info.get("shards", []):
        path = Path(str(shard["path"]))
        shard_path = path if path.is_absolute() else base / path
        data = torch.load(shard_path, map_location="cpu")
        count = int(data["source_record"].shape[0])
        take = count
        if limit > 0:
            remaining = limit - loaded
            if remaining <= 0:
                break
            take = min(take, remaining)
        combined = torch.cat((data["source_record"][:take], data["target_record"][:take]), dim=0)
        records.append(combined)
        ids = list(data["case_ids"][:take])
        case_ids.extend([f"{case_id}:source" for case_id in ids])
        case_ids.extend([f"{case_id}:target" for case_id in ids])
        loaded += take
        if limit > 0 and loaded >= limit:
            break
    if not records:
        raise ValueError(f"split {split!r} has no data")
    return RecordImageSet(torch.cat(records, dim=0).long(), case_ids)


def load_datasets(config: StageAVIConfig) -> tuple[RecordImageSet, RecordImageSet, RecordImageSet, RecordImageSet, dict[str, object]]:
    manifest_path = Path(config.dataset_manifest)
    manifest = load_manifest(manifest_path)
    train = load_split(manifest_path, manifest, "train", config.train_limit)
    val = load_split(manifest_path, manifest, "val", config.val_limit)
    test = load_split(manifest_path, manifest, "test", config.test_limit)
    heldout = load_split(manifest_path, manifest, "heldout", config.heldout_limit)
    return train, val, test, heldout, {
        "source_manifest": str(manifest_path),
        "manifest_scale": manifest.get("scale", ""),
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "heldout_images": len(heldout),
        "image_size": IMAGE_SIZE,
        "records_policy": "Each AV-H source_record and target_record is rendered as an independent whole-image reconstruction sample.",
    }


def to_transparent_rgba(images: torch.Tensor) -> torch.Tensor:
    alpha = (images.amax(dim=1, keepdim=True) > 0.12).float()
    rgb = torch.where(alpha.bool(), images, torch.zeros_like(images))
    return torch.cat((rgb, alpha), dim=1)


def random_batch(data: RecordImageSet, batch_size: int, device: torch.device) -> torch.Tensor:
    index_device = data.records.device
    indices = torch.randint(0, len(data), (batch_size,), device=index_device)
    batch = data.tensor_subset(indices).to(device)
    return to_transparent_rgba(render_records(batch.records, device=device))


class WholeImagePrefixLatentAutoencoder(nn.Module):
    def __init__(self, latent_tokens: int, latent_dim: int, encoder_width: int, decoder_width: int, image_size: int = IMAGE_SIZE) -> None:
        super().__init__()
        self.latent_tokens = latent_tokens
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.decoder_width = decoder_width
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 48, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(96, 192, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(192, encoder_width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.to_latents = nn.Linear(encoder_width, latent_tokens * latent_dim)
        self.token_position = nn.Parameter(torch.randn(latent_tokens, latent_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_width),
            nn.GELU(),
            nn.Linear(decoder_width, 4 * image_size * image_size),
        )
        base = torch.empty((1, 1, 4, image_size, image_size))
        base[:, :, :3].fill_(math.log(BACKGROUND / (1.0 - BACKGROUND)))
        base[:, :, 3].fill_(-8.0)
        self.register_buffer("background_logit", base)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(images)
        latents = self.to_latents(hidden).view(images.shape[0], self.latent_tokens, self.latent_dim)
        return latents + self.token_position.view(1, self.latent_tokens, self.latent_dim)

    def token_logits(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.encode(images)
        logits = self.decoder(latents).view(images.shape[0], self.latent_tokens, 4, self.image_size, self.image_size)
        return logits

    def reconstruct_prefixes(self, token_logits: torch.Tensor, prefixes: list[int]) -> dict[int, torch.Tensor]:
        cumulative = token_logits.cumsum(dim=1)
        return {prefix: torch.sigmoid(self.background_logit[:, 0] + cumulative[:, prefix - 1]) for prefix in prefixes}

    def reconstruct_selected(self, token_logits: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        delta = (token_logits * selected.view(selected.shape[0], selected.shape[1], 1, 1, 1)).sum(dim=1)
        return torch.sigmoid(self.background_logit[:, 0] + delta)


def transparent_loss_per_sample(predicted: torch.Tensor, target: torch.Tensor, config: StageAVIConfig) -> torch.Tensor:
    pred_rgb = predicted[:, :3]
    pred_alpha = predicted[:, 3:4].float().clamp(ALPHA_EPS, 1.0 - ALPHA_EPS)
    target_rgb = target[:, :3]
    target_alpha = target[:, 3:4]
    object_area = target_alpha.sum(dim=(1, 2, 3)).clamp_min(1.0)
    rgb_loss = ((pred_rgb - target_rgb).square() * target_alpha).sum(dim=(1, 2, 3)) / (object_area * 3.0)
    alpha_bce = -(target_alpha.float() * pred_alpha.log() + (1.0 - target_alpha.float()) * (1.0 - pred_alpha).log())
    alpha_weights = target_alpha * config.alpha_weight + (1.0 - target_alpha) * config.background_alpha_weight
    alpha_loss = (alpha_bce * alpha_weights).mean(dim=(1, 2, 3))
    return config.rgb_weight * rgb_loss + alpha_loss


def transparent_loss(predicted: torch.Tensor, target: torch.Tensor, config: StageAVIConfig) -> torch.Tensor:
    return transparent_loss_per_sample(predicted, target, config).mean()


def fast_ends_slow_middle(progress: float, curve_power: float) -> float:
    t = min(1.0, max(0.0, progress))
    power = min(1.0, max(0.05, curve_power))
    if t <= 0.5:
        return 0.5 * ((2.0 * t) ** power)
    return 1.0 - 0.5 * ((2.0 * (1.0 - t)) ** power)


def segment_bounds_for_step(latent_tokens: int, total_steps: int, step: int) -> tuple[int, int, int]:
    start = 1
    for active_tokens, end in curriculum_segments(latent_tokens, total_steps):
        if step <= end:
            return active_tokens, start, end
        start = end + 1
    return latent_tokens, start, total_steps


def compact_precision_power_for_step(config: StageAVIConfig, latent_tokens: int, step: int) -> float:
    _active_tokens, start, end = segment_bounds_for_step(latent_tokens, config.steps, step)
    if end <= start:
        return float(config.compact_precision_power_end)
    segment_progress = (step - start) / float(end - start)
    shaped_progress = fast_ends_slow_middle(segment_progress, config.compact_precision_curve_power)
    return float(config.compact_precision_power_start + shaped_progress * (config.compact_precision_power_end - config.compact_precision_power_start))


def correct_pixel_color_tau_for_step(config: StageAVIConfig, latent_tokens: int, step: int) -> float:
    _active_tokens, start, end = segment_bounds_for_step(latent_tokens, config.steps, step)
    if end <= start:
        return float(config.correct_pixel_color_tau_end)
    segment_progress = (step - start) / float(end - start)
    shaped_progress = fast_ends_slow_middle(segment_progress, config.correct_pixel_color_tau_curve_power)
    delta = config.correct_pixel_color_tau_end - config.correct_pixel_color_tau_start
    return float(max(ALPHA_EPS, config.correct_pixel_color_tau_start + shaped_progress * delta))


def residual_weight_multiplier_for_step(config: StageAVIConfig, latent_tokens: int, step: int) -> float:
    _active_tokens, start, end = segment_bounds_for_step(latent_tokens, config.steps, step)
    if end <= start:
        return float(config.residual_weight_end)
    segment_progress = (step - start) / float(end - start)
    shaped_progress = fast_ends_slow_middle(segment_progress, config.residual_curve_power)
    return float(config.residual_weight_start + shaped_progress * (config.residual_weight_end - config.residual_weight_start))


def rgb_mse_map(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (predicted[:, :3] - target[:, :3]).square().mean(dim=1, keepdim=True)


def foreground_correctness_map(predicted: torch.Tensor, target: torch.Tensor, precision_power: float, color_tau: float) -> torch.Tensor:
    pred_alpha = predicted[:, 3:4].float().clamp(ALPHA_EPS, 1.0 - ALPHA_EPS)
    target_alpha = target[:, 3:4].float()
    precise_alpha = pred_alpha.pow(max(precision_power, ALPHA_EPS))
    color_score = torch.exp(-rgb_mse_map(predicted, target) / max(color_tau, ALPHA_EPS))
    return (target_alpha * precise_alpha * color_score).clamp(0.0, 1.0)


def full_correctness_map(predicted: torch.Tensor, target: torch.Tensor, precision_power: float, color_tau: float) -> torch.Tensor:
    pred_alpha = predicted[:, 3:4].float().clamp(ALPHA_EPS, 1.0 - ALPHA_EPS)
    target_alpha = target[:, 3:4].float()
    foreground = foreground_correctness_map(predicted, target, precision_power, color_tau)
    background = (1.0 - target_alpha) * (1.0 - pred_alpha)
    return (foreground + background).clamp(0.0, 1.0)


def compact_prefix_loss(predicted: torch.Tensor, target: torch.Tensor, config: StageAVIConfig, precision_power: float, color_tau: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_alpha = predicted[:, 3:4].float().clamp(ALPHA_EPS, 1.0 - ALPHA_EPS)
    target_alpha = target[:, 3:4].float()
    precise_alpha = pred_alpha.pow(max(precision_power, ALPHA_EPS))
    color_score = torch.exp(-rgb_mse_map(predicted, target) / max(color_tau, ALPHA_EPS))
    image_area = float(pred_alpha.shape[-1] * pred_alpha.shape[-2])
    target_area = target_alpha.sum(dim=(1, 2, 3)).clamp_min(1.0)
    occupied_area = pred_alpha.sum(dim=(1, 2, 3)) / image_area
    linear_covered_target = (pred_alpha * target_alpha).sum(dim=(1, 2, 3)) / target_area
    precise_covered_target = (precise_alpha * target_alpha).sum(dim=(1, 2, 3)) / target_area
    correct_pixel_rate = (target_alpha * precise_alpha * color_score).sum(dim=(1, 2, 3)) / target_area
    wrong_background_rate = ((1.0 - target_alpha) * pred_alpha).sum(dim=(1, 2, 3)) / image_area
    wrong_color_rate = (target_alpha * precise_alpha * (1.0 - color_score)).sum(dim=(1, 2, 3)) / target_area
    missed_foreground_rate = (target_alpha * (1.0 - pred_alpha)).sum(dim=(1, 2, 3)) / target_area
    total_weight = max(
        config.compact_correct_weight
        + config.compact_wrong_bg_weight
        + config.compact_wrong_color_weight
        + config.compact_missed_fg_weight,
        ALPHA_EPS,
    )
    loss_per_sample = (
        config.compact_correct_weight * (1.0 - correct_pixel_rate)
        + config.compact_wrong_bg_weight * wrong_background_rate
        + config.compact_wrong_color_weight * wrong_color_rate
        + config.compact_missed_fg_weight * missed_foreground_rate
    ) / total_weight
    return loss_per_sample.mean(), {
        "compact_area": occupied_area.mean().detach(),
        "compact_target_coverage": linear_covered_target.mean().detach(),
        "compact_target_precision_coverage": precise_covered_target.mean().detach(),
        "compact_correct_pixel_rate": correct_pixel_rate.mean().detach(),
        "compact_wrong_background_rate": wrong_background_rate.mean().detach(),
        "compact_wrong_color_rate": wrong_color_rate.mean().detach(),
        "compact_missed_foreground_rate": missed_foreground_rate.mean().detach(),
    }


def residual_transition_loss(previous: torch.Tensor, current: torch.Tensor, target: torch.Tensor, config: StageAVIConfig, precision_power: float, color_tau: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    previous_correct = full_correctness_map(previous, target, precision_power, color_tau).detach()
    current_correct = full_correctness_map(current, target, precision_power, color_tau)
    current_wrong = 1.0 - current_correct
    needed = (1.0 - previous_correct).detach()
    preserved_denominator = previous_correct.sum(dim=(1, 2, 3)).clamp_min(1.0)
    needed_denominator = needed.sum(dim=(1, 2, 3)).clamp_min(1.0)
    preserve_loss = (previous_correct * current_wrong).sum(dim=(1, 2, 3)) / preserved_denominator
    delta_map = (current - previous.detach()).abs().mean(dim=1, keepdim=True)
    delta_outside_loss = (previous_correct * delta_map).sum(dim=(1, 2, 3)) / preserved_denominator
    needed_error_loss = (needed * current_wrong).sum(dim=(1, 2, 3)) / needed_denominator
    total_weight = max(config.residual_preserve_weight + config.residual_delta_weight + config.residual_needed_weight, ALPHA_EPS)
    loss_per_sample = (
        config.residual_preserve_weight * preserve_loss
        + config.residual_delta_weight * delta_outside_loss
        + config.residual_needed_weight * needed_error_loss
    ) / total_weight
    previous_wrong = 1.0 - previous_correct
    error_reduction = ((previous_wrong - current_wrong.detach()) * needed).sum(dim=(1, 2, 3)) / needed_denominator
    return loss_per_sample.mean(), {
        "residual_preserve_loss": preserve_loss.mean().detach(),
        "residual_delta_outside_loss": delta_outside_loss.mean().detach(),
        "residual_needed_error_loss": needed_error_loss.mean().detach(),
        "residual_error_reduction": error_reduction.mean().detach(),
    }


def image_metrics(predicted: torch.Tensor, target: torch.Tensor, config: StageAVIConfig) -> dict[str, float]:
    pred_alpha = predicted[:, 3:4]
    target_alpha = target[:, 3:4]
    pred_mask = pred_alpha > 0.5
    target_mask = target_alpha > 0.5
    intersection = (pred_mask & target_mask).float().sum(dim=(1, 2, 3))
    union = (pred_mask | target_mask).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    target_area = target_mask.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    pred_area = pred_mask.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    rgb_squared = (predicted[:, :3] - target[:, :3]).square()
    object_rgb_mse = (rgb_squared * target_alpha).sum() / (target_alpha.sum().clamp_min(1.0) * 3.0)
    background_alpha_mean = (pred_alpha * (1.0 - target_alpha)).sum() / (1.0 - target_alpha).sum().clamp_min(1.0)
    object_alpha_mean = (pred_alpha * target_alpha).sum() / target_alpha.sum().clamp_min(1.0)
    soft_occupied_area = pred_alpha.sum() / float(pred_alpha.shape[0] * pred_alpha.shape[-1] * pred_alpha.shape[-2])
    soft_target_coverage = (pred_alpha * target_alpha).sum() / target_alpha.sum().clamp_min(1.0)
    precision_power = max(config.compact_precision_power_end, ALPHA_EPS)
    soft_target_precision_coverage = (pred_alpha.float().clamp(ALPHA_EPS, 1.0).pow(precision_power) * target_alpha).sum() / target_alpha.sum().clamp_min(1.0)
    color_tau = max(config.correct_pixel_color_tau_end, ALPHA_EPS)
    soft_correct_pixel_rate = foreground_correctness_map(predicted, target, precision_power, color_tau).sum() / target_alpha.sum().clamp_min(1.0)
    soft_wrong_background_rate = ((1.0 - target_alpha) * pred_alpha.float().clamp(ALPHA_EPS, 1.0 - ALPHA_EPS)).sum() / float(pred_alpha.shape[0] * pred_alpha.shape[-1] * pred_alpha.shape[-2])
    color_score = torch.exp(-rgb_mse_map(predicted, target) / color_tau)
    precise_alpha = pred_alpha.float().clamp(ALPHA_EPS, 1.0).pow(precision_power)
    soft_wrong_color_rate = (target_alpha * precise_alpha * (1.0 - color_score)).sum() / target_alpha.sum().clamp_min(1.0)
    soft_missed_foreground_rate = (target_alpha * (1.0 - pred_alpha.float().clamp(ALPHA_EPS, 1.0 - ALPHA_EPS))).sum() / target_alpha.sum().clamp_min(1.0)
    rgb_rmse = rgb_mse_map(predicted, target).sqrt()
    hard_color_correct = rgb_rmse <= config.correct_pixel_rgb_threshold
    hard_alpha = pred_alpha > config.correct_pixel_alpha_threshold
    hard_correct_foreground = (hard_alpha & target_mask & hard_color_correct).float()
    hard_wrong_background = (hard_alpha & ~target_mask).float()
    hard_wrong_color = (hard_alpha & target_mask & ~hard_color_correct).float()
    hard_missed_foreground = (~hard_alpha & target_mask).float()
    hard_correct_pixel_rate = hard_correct_foreground.sum() / target_mask.float().sum().clamp_min(1.0)
    hard_wrong_background_rate = hard_wrong_background.sum() / float(pred_alpha.shape[0] * pred_alpha.shape[-1] * pred_alpha.shape[-2])
    hard_wrong_color_rate = hard_wrong_color.sum() / target_mask.float().sum().clamp_min(1.0)
    hard_missed_foreground_rate = hard_missed_foreground.sum() / target_mask.float().sum().clamp_min(1.0)
    alpha_mse = F.mse_loss(pred_alpha, target_alpha)
    return {
        "transparent_loss": float(transparent_loss(predicted, target, config).item()),
        "object_rgb_mse": float(object_rgb_mse.item()),
        "alpha_mse": float(alpha_mse.item()),
        "alpha_iou": float((intersection / union).mean().item()),
        "alpha_coverage": float((intersection / target_area).mean().item()),
        "alpha_precision": float((intersection / pred_area).mean().item()),
        "object_alpha_mean": float(object_alpha_mean.item()),
        "background_alpha_mean": float(background_alpha_mean.item()),
        "soft_occupied_area": float(soft_occupied_area.item()),
        "soft_target_coverage": float(soft_target_coverage.item()),
        "soft_target_precision_coverage": float(soft_target_precision_coverage.item()),
        "soft_correct_pixel_rate": float(soft_correct_pixel_rate.item()),
        "soft_wrong_background_rate": float(soft_wrong_background_rate.item()),
        "soft_wrong_color_rate": float(soft_wrong_color_rate.item()),
        "soft_missed_foreground_rate": float(soft_missed_foreground_rate.item()),
        "hard_correct_pixel_rate": float(hard_correct_pixel_rate.item()),
        "hard_wrong_background_rate": float(hard_wrong_background_rate.item()),
        "hard_wrong_color_rate": float(hard_wrong_color_rate.item()),
        "hard_missed_foreground_rate": float(hard_missed_foreground_rate.item()),
    }


def loss_for_batch(model: WholeImagePrefixLatentAutoencoder, images: torch.Tensor, config: StageAVIConfig, active_tokens: int, step: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    token_logits = model.token_logits(images)
    prefixes = [prefix for prefix in prefix_lengths(active_tokens) if prefix <= active_tokens]
    recon = model.reconstruct_prefixes(token_logits, prefixes)
    final_loss = transparent_loss(recon[active_tokens], images, config)
    loss = final_loss
    precision_power = compact_precision_power_for_step(config, model.latent_tokens, step)
    color_tau = correct_pixel_color_tau_for_step(config, model.latent_tokens, step)
    residual_multiplier = residual_weight_multiplier_for_step(config, model.latent_tokens, step)
    prefix_losses = []
    compact_losses = []
    compact_areas = []
    compact_coverages = []
    compact_precision_coverages = []
    compact_correct_pixel_rates = []
    compact_wrong_background_rates = []
    compact_wrong_color_rates = []
    compact_missed_foreground_rates = []
    residual_losses = []
    residual_preserve_losses = []
    residual_delta_losses = []
    residual_needed_losses = []
    residual_error_reductions = []
    previous_prefix: int | None = None
    for prefix in prefixes:
        prefix_loss = transparent_loss(recon[prefix], images, config)
        prefix_losses.append(prefix_loss)
        if prefix < model.latent_tokens:
            compact_loss, compact_metrics = compact_prefix_loss(recon[prefix], images, config, precision_power, color_tau)
            compact_losses.append(compact_loss)
            compact_areas.append(compact_metrics["compact_area"])
            compact_coverages.append(compact_metrics["compact_target_coverage"])
            compact_precision_coverages.append(compact_metrics["compact_target_precision_coverage"])
            compact_correct_pixel_rates.append(compact_metrics["compact_correct_pixel_rate"])
            compact_wrong_background_rates.append(compact_metrics["compact_wrong_background_rate"])
            compact_wrong_color_rates.append(compact_metrics["compact_wrong_color_rate"])
            compact_missed_foreground_rates.append(compact_metrics["compact_missed_foreground_rate"])
            loss = loss + config.compact_prefix_weight * compact_loss / math.sqrt(prefix)
        if previous_prefix is not None:
            residual_loss, residual_metrics = residual_transition_loss(recon[previous_prefix], recon[prefix], images, config, precision_power, color_tau)
            residual_losses.append(residual_loss)
            residual_preserve_losses.append(residual_metrics["residual_preserve_loss"])
            residual_delta_losses.append(residual_metrics["residual_delta_outside_loss"])
            residual_needed_losses.append(residual_metrics["residual_needed_error_loss"])
            residual_error_reductions.append(residual_metrics["residual_error_reduction"])
            loss = loss + config.residual_prefix_weight * residual_multiplier * residual_loss / math.sqrt(prefix)
        if prefix != active_tokens:
            loss = loss + config.prefix_weight * prefix_loss / math.sqrt(prefix)
        previous_prefix = prefix
    active_energy = token_logits[:, :active_tokens].flatten(2).abs().mean(dim=2)
    monotonic = F.relu(active_energy[:, 1:] - active_energy[:, :-1]).mean() if active_tokens > 1 else active_energy.new_tensor(0.0)
    loss = loss + config.monotonic_weight * monotonic
    compact_loss_mean = torch.stack(compact_losses).mean().detach() if compact_losses else final_loss.new_tensor(0.0)
    compact_area_mean = torch.stack(compact_areas).mean().detach() if compact_areas else final_loss.new_tensor(0.0)
    compact_coverage_mean = torch.stack(compact_coverages).mean().detach() if compact_coverages else final_loss.new_tensor(0.0)
    compact_precision_coverage_mean = torch.stack(compact_precision_coverages).mean().detach() if compact_precision_coverages else final_loss.new_tensor(0.0)
    compact_correct_pixel_mean = torch.stack(compact_correct_pixel_rates).mean().detach() if compact_correct_pixel_rates else final_loss.new_tensor(0.0)
    compact_wrong_background_mean = torch.stack(compact_wrong_background_rates).mean().detach() if compact_wrong_background_rates else final_loss.new_tensor(0.0)
    compact_wrong_color_mean = torch.stack(compact_wrong_color_rates).mean().detach() if compact_wrong_color_rates else final_loss.new_tensor(0.0)
    compact_missed_foreground_mean = torch.stack(compact_missed_foreground_rates).mean().detach() if compact_missed_foreground_rates else final_loss.new_tensor(0.0)
    residual_loss_mean = torch.stack(residual_losses).mean().detach() if residual_losses else final_loss.new_tensor(0.0)
    residual_preserve_mean = torch.stack(residual_preserve_losses).mean().detach() if residual_preserve_losses else final_loss.new_tensor(0.0)
    residual_delta_mean = torch.stack(residual_delta_losses).mean().detach() if residual_delta_losses else final_loss.new_tensor(0.0)
    residual_needed_mean = torch.stack(residual_needed_losses).mean().detach() if residual_needed_losses else final_loss.new_tensor(0.0)
    residual_error_reduction_mean = torch.stack(residual_error_reductions).mean().detach() if residual_error_reductions else final_loss.new_tensor(0.0)
    return loss, {
        "loss_final_transparent": final_loss.detach(),
        "loss_prefix_mean": torch.stack(prefix_losses).mean().detach(),
        "loss_compact_prefix_mean": compact_loss_mean,
        "compact_prefix_area_mean": compact_area_mean,
        "compact_prefix_target_coverage_mean": compact_coverage_mean,
        "compact_prefix_target_precision_coverage_mean": compact_precision_coverage_mean,
        "compact_prefix_correct_pixel_rate_mean": compact_correct_pixel_mean,
        "compact_prefix_wrong_background_rate_mean": compact_wrong_background_mean,
        "compact_prefix_wrong_color_rate_mean": compact_wrong_color_mean,
        "compact_prefix_missed_foreground_rate_mean": compact_missed_foreground_mean,
        "compact_precision_power": final_loss.new_tensor(precision_power),
        "correct_pixel_color_tau": final_loss.new_tensor(color_tau),
        "loss_residual_prefix_mean": residual_loss_mean,
        "residual_preserve_loss_mean": residual_preserve_mean,
        "residual_delta_outside_loss_mean": residual_delta_mean,
        "residual_needed_error_loss_mean": residual_needed_mean,
        "residual_error_reduction_mean": residual_error_reduction_mean,
        "residual_weight_multiplier": final_loss.new_tensor(residual_multiplier),
        "loss_monotonic": monotonic.detach(),
        "token_energy_first": active_energy[:, 0].mean().detach(),
        "token_energy_active_last": active_energy[:, -1].mean().detach(),
    }


def greedy_prefix_reconstructions(model: WholeImagePrefixLatentAutoencoder, token_logits: torch.Tensor, target: torch.Tensor, prefixes: list[int], config: StageAVIConfig) -> dict[int, torch.Tensor]:
    max_prefix = max(prefixes)
    selected = torch.zeros((token_logits.shape[0], model.latent_tokens), dtype=torch.bool, device=token_logits.device)
    selected_logits = torch.zeros_like(token_logits[:, 0])
    batch_index = torch.arange(token_logits.shape[0], device=token_logits.device)
    outputs: dict[int, torch.Tensor] = {}
    for step in range(1, max_prefix + 1):
        scores = []
        for token_index in range(model.latent_tokens):
            candidate = torch.sigmoid(model.background_logit[:, 0] + selected_logits + token_logits[:, token_index])
            score = -transparent_loss_per_sample(candidate, target, config)
            score = torch.where(selected[:, token_index], torch.full_like(score, -1.0e9), score)
            scores.append(score)
        best = torch.stack(scores, dim=1).argmax(dim=1)
        selected[batch_index, best] = True
        selected_logits = selected_logits + token_logits[batch_index, best]
        if step in prefixes:
            outputs[step] = torch.sigmoid(model.background_logit[:, 0] + selected_logits)
    return outputs


@torch.inference_mode()
def evaluate_model(model: WholeImagePrefixLatentAutoencoder, data: RecordImageSet, config: StageAVIConfig, device: torch.device, include_greedy: bool) -> dict[str, float]:
    model.eval()
    prefixes = prefix_lengths(model.latent_tokens)
    totals: dict[str, dict[str, float]] = {}
    examples = 0
    greedy_seen = 0
    for start in range(0, len(data), config.eval_batch_size):
        batch = data.slice(start, min(start + config.eval_batch_size, len(data))).to(device)
        images = to_transparent_rgba(render_records(batch.records, device=device))
        with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
            token_logits = model.token_logits(images)
            ordered = model.reconstruct_prefixes(token_logits, prefixes)
            greedy = {}
            if include_greedy and greedy_seen < config.greedy_eval_limit:
                take = min(len(batch), config.greedy_eval_limit - greedy_seen)
                greedy = greedy_prefix_reconstructions(model, token_logits[:take], images[:take], prefixes, config)
                greedy_seen += take
        for prefix, predicted in ordered.items():
            key = f"ordered_prefix_{prefix}"
            metrics = image_metrics(predicted, images, config)
            totals.setdefault(key, {metric: 0.0 for metric in metrics})
            for metric, value in metrics.items():
                totals[key][metric] += value * len(batch)
        for prefix, predicted in greedy.items():
            key = f"greedy_prefix_{prefix}"
            metrics = image_metrics(predicted, images[: predicted.shape[0]], config)
            totals.setdefault(key, {metric: 0.0 for metric in metrics})
            for metric, value in metrics.items():
                totals[key][metric] += value * predicted.shape[0]
        examples += len(batch)
    result: dict[str, float] = {}
    for key, sums in totals.items():
        denominator = float(config.greedy_eval_limit if key.startswith("greedy") and include_greedy else examples)
        if key.startswith("greedy"):
            denominator = float(max(1, min(config.greedy_eval_limit, examples)))
        for metric, value in sums.items():
            result[f"{key}_{metric}"] = value / denominator
    result["examples"] = float(examples)
    result["greedy_examples"] = float(greedy_seen)
    return result


def write_rgba_png(path: Path, rgba_chw: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba = rgba_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (rgba * 255.0).round().astype("uint8")
    Image.fromarray(array, mode="RGBA").save(path)


def write_samples(model: WholeImagePrefixLatentAutoencoder, data: RecordImageSet, output_dir: Path, config: StageAVIConfig, device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = data.slice(0, min(config.sample_count, len(data))).to(device)
    images = to_transparent_rgba(render_records(batch.records, device=device))
    prefixes = prefix_lengths(model.latent_tokens)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
        token_logits = model.token_logits(images)
        ordered = model.reconstruct_prefixes(token_logits, prefixes)
        greedy = greedy_prefix_reconstructions(model, token_logits, images, prefixes, config)
    rows = []
    for index, case_id in enumerate(batch.case_ids):
        clean_case = case_id.replace(":", "_")
        target_path = output_dir / f"{clean_case}_target.png"
        write_rgba_png(target_path, images[index])
        row = {"case_id": case_id, "target_png": str(target_path), "ordered": {}, "greedy": {}}
        for prefix in prefixes:
            ordered_path = output_dir / f"{clean_case}_ordered_{prefix}.png"
            greedy_path = output_dir / f"{clean_case}_greedy_{prefix}.png"
            write_rgba_png(ordered_path, ordered[prefix][index])
            write_rgba_png(greedy_path, greedy[prefix][index])
            row["ordered"][str(prefix)] = str(ordered_path)
            row["greedy"][str(prefix)] = str(greedy_path)
        rows.append(row)
    (output_dir / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested cuda but CUDA is not available")
    return torch.device(name)


def train_one(latent_tokens: int, train: RecordImageSet, val: RecordImageSet, test: RecordImageSet, heldout: RecordImageSet, config: StageAVIConfig, device: torch.device, output_dir: Path, run_name: str) -> dict[str, object]:
    torch.manual_seed(config.seed + latent_tokens)
    model = WholeImagePrefixLatentAutoencoder(
        latent_tokens=latent_tokens,
        latent_dim=config.latent_dim,
        encoder_width=config.encoder_width,
        decoder_width=config.decoder_width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    history = []
    segments = curriculum_segments(latent_tokens, config.steps)
    segment_ends = {end for _active, end in segments}
    segment_starts = set()
    segment_start = 1
    for _active, segment_end in segments:
        segment_starts.add(segment_start)
        segment_start = segment_end + 1
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        model.train()
        active_tokens = active_tokens_for_step(latent_tokens, config.steps, step)
        images = random_batch(train, config.batch_size, device)
        with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
            loss, components = loss_for_batch(model, images, config, active_tokens, step)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step == config.steps or step in segment_starts or step in segment_ends:
            val_metrics = evaluate_model(model, val, config, device, include_greedy=False)
            row = {f"val_{key}": value for key, value in val_metrics.items()}
            row.update({key: float(value.item()) for key, value in components.items()})
            row["step"] = float(step)
            row["active_tokens"] = float(active_tokens)
            row["loss"] = float(loss.detach().item())
            history.append(row)
    metrics = {
        "test": evaluate_model(model, test, config, device, include_greedy=True),
        "heldout": evaluate_model(model, heldout, config, device, include_greedy=True),
    }
    sample_dir = output_dir / f"{run_name}_latent_{latent_tokens}_samples"
    write_samples(model, test, sample_dir, config, device)
    return {
        "latent_tokens": latent_tokens,
        "d_model": config.d_model,
        "latent_dim": config.latent_dim,
        "encoder_width": config.encoder_width,
        "decoder_width": config.decoder_width,
        "latent_capacity_scalars": latent_tokens * config.latent_dim,
        "parameters": sum(param.numel() for param in model.parameters()),
        "elapsed_sec": time.perf_counter() - started,
        "history": history,
        "metrics": metrics,
        "sample_dir": str(sample_dir),
        "prefixes": prefix_lengths(latent_tokens),
        "curriculum": [{"active_tokens": active, "end_step": end} for active, end in curriculum_segments(latent_tokens, config.steps)],
    }


def run(config: StageAVIConfig) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)
    train, val, test, heldout, dataset_info = load_datasets(config)
    if config.gpu_resident_data and device.type == "cuda":
        train = train.to(device)
        val = val.to(device)
    output_path = Path(config.output)
    output_dir = output_path.parent
    run_name = output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    started = time.perf_counter()
    for latent_tokens in parse_latent_list(config.latent_tokens_list):
        runs.append(train_one(latent_tokens, train, val, test, heldout, config, device, output_dir, run_name))
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    result = {
        "schema_version": 5,
        "stage": "AV-I",
        "config": asdict(config),
        "dataset": dataset_info,
        "device": str(device),
        "cost": {
            "elapsed_sec": time.perf_counter() - started,
            "peak_cuda_allocated_mb": peak_mb,
            "nominal_processed_image_tokens": config.steps * config.batch_size * sum(parse_latent_list(config.latent_tokens_list)),
            "nominal_processed_latent_scalars": config.steps * config.batch_size * sum(tokens * config.latent_dim for tokens in parse_latent_list(config.latent_tokens_list)),
        },
        "runs": runs,
        "interpretation": {
            "goal": "Test how many whole-image latent tokens are needed to reconstruct transparent-background AV-H rendered images without patch-token decoding.",
            "dimension_policy": "encoder_width and decoder_width control codec/expert capacity; latent_dim controls the actual per-token bottleneck width.",
            "transparent_target": "Black background is converted to alpha=0. RGB error is measured only on non-transparent pixels; alpha error is measured over the whole image.",
            "ordered_prefix": "Training uses an active-token curriculum. The model first learns a short target prefix, then gradually unlocks longer latent prefixes.",
            "compact_prefix": "Non-final prefixes receive correct-pixel compact guidance: maximize foreground pixels whose alpha and RGB are both correct, penalize wrong background alpha, wrong foreground color, and missed foreground. Alpha precision uses pred_alpha^n. n restarts inside every active-token curriculum segment and follows a fast-ends/slow-middle curve from compact_precision_power_start to compact_precision_power_end.",
            "residual_prefix": "Adjacent ordered prefixes receive residual routing guidance. Later prefixes are rewarded for fixing pixels that the previous prefix still missed, while preserving pixels that were already correct and limiting changes outside needed residual regions. The residual weight ramps up inside every active-token segment.",
            "greedy_prefix": "At evaluation time, each image greedily selects the next token that minimizes full transparent-image reconstruction loss, not foreground intersection.",
            "boundary": "This is an image latent capacity diagnostic, not an AV-H edit success criterion.",
        },
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "cost": result["cost"], "latent_tokens": [run["latent_tokens"] for run in runs]}, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", default=StageAVIConfig.dataset_manifest)
    parser.add_argument("--output", default=StageAVIConfig.output)
    parser.add_argument("--latent-tokens-list", default=StageAVIConfig.latent_tokens_list)
    parser.add_argument("--d-model", type=int, default=None, help="Legacy width. Used for encoder_width and latent_dim unless the explicit flags override it.")
    parser.add_argument("--latent-dim", type=int, default=None, help="Per latent-token bottleneck width.")
    parser.add_argument("--encoder-width", type=int, default=None, help="CNN encoder output width before projection to the latent bank.")
    parser.add_argument("--decoder-width", type=int, default=None, help="Output expert hidden width after each latent token.")
    parser.add_argument("--steps", type=int, default=StageAVIConfig.steps)
    parser.add_argument("--batch-size", type=int, default=StageAVIConfig.batch_size)
    parser.add_argument("--lr", type=float, default=StageAVIConfig.lr)
    parser.add_argument("--prefix-weight", type=float, default=StageAVIConfig.prefix_weight)
    parser.add_argument("--rgb-weight", type=float, default=StageAVIConfig.rgb_weight)
    parser.add_argument("--alpha-weight", type=float, default=StageAVIConfig.alpha_weight)
    parser.add_argument("--background-alpha-weight", type=float, default=StageAVIConfig.background_alpha_weight)
    parser.add_argument("--monotonic-weight", type=float, default=StageAVIConfig.monotonic_weight)
    parser.add_argument("--compact-prefix-weight", type=float, default=StageAVIConfig.compact_prefix_weight)
    parser.add_argument("--compact-correct-weight", type=float, default=StageAVIConfig.compact_correct_weight)
    parser.add_argument("--compact-wrong-bg-weight", type=float, default=StageAVIConfig.compact_wrong_bg_weight)
    parser.add_argument("--compact-wrong-color-weight", type=float, default=StageAVIConfig.compact_wrong_color_weight)
    parser.add_argument("--compact-missed-fg-weight", type=float, default=StageAVIConfig.compact_missed_fg_weight)
    parser.add_argument("--compact-precision-power-start", type=float, default=StageAVIConfig.compact_precision_power_start)
    parser.add_argument("--compact-precision-power-end", type=float, default=StageAVIConfig.compact_precision_power_end)
    parser.add_argument("--compact-precision-curve-power", type=float, default=StageAVIConfig.compact_precision_curve_power)
    parser.add_argument("--correct-pixel-color-tau-start", type=float, default=StageAVIConfig.correct_pixel_color_tau_start)
    parser.add_argument("--correct-pixel-color-tau-end", type=float, default=StageAVIConfig.correct_pixel_color_tau_end)
    parser.add_argument("--correct-pixel-color-tau-curve-power", type=float, default=StageAVIConfig.correct_pixel_color_tau_curve_power)
    parser.add_argument("--correct-pixel-rgb-threshold", type=float, default=StageAVIConfig.correct_pixel_rgb_threshold)
    parser.add_argument("--correct-pixel-alpha-threshold", type=float, default=StageAVIConfig.correct_pixel_alpha_threshold)
    parser.add_argument("--residual-prefix-weight", type=float, default=StageAVIConfig.residual_prefix_weight)
    parser.add_argument("--residual-preserve-weight", type=float, default=StageAVIConfig.residual_preserve_weight)
    parser.add_argument("--residual-delta-weight", type=float, default=StageAVIConfig.residual_delta_weight)
    parser.add_argument("--residual-needed-weight", type=float, default=StageAVIConfig.residual_needed_weight)
    parser.add_argument("--residual-weight-start", type=float, default=StageAVIConfig.residual_weight_start)
    parser.add_argument("--residual-weight-end", type=float, default=StageAVIConfig.residual_weight_end)
    parser.add_argument("--residual-curve-power", type=float, default=StageAVIConfig.residual_curve_power)
    parser.add_argument("--train-limit", type=int, default=StageAVIConfig.train_limit)
    parser.add_argument("--val-limit", type=int, default=StageAVIConfig.val_limit)
    parser.add_argument("--test-limit", type=int, default=StageAVIConfig.test_limit)
    parser.add_argument("--heldout-limit", type=int, default=StageAVIConfig.heldout_limit)
    parser.add_argument("--eval-batch-size", type=int, default=StageAVIConfig.eval_batch_size)
    parser.add_argument("--greedy-eval-limit", type=int, default=StageAVIConfig.greedy_eval_limit)
    parser.add_argument("--sample-count", type=int, default=StageAVIConfig.sample_count)
    parser.add_argument("--seed", type=int, default=StageAVIConfig.seed)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=StageAVIConfig.device)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-gpu-resident-data", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> StageAVIConfig:
    legacy_width = args.d_model if args.d_model is not None else StageAVIConfig.d_model
    latent_dim = args.latent_dim if args.latent_dim is not None else legacy_width
    encoder_width = args.encoder_width if args.encoder_width is not None else legacy_width
    decoder_width = args.decoder_width if args.decoder_width is not None else latent_dim * 4
    return StageAVIConfig(
        dataset_manifest=args.dataset_manifest,
        output=args.output,
        latent_tokens_list=args.latent_tokens_list,
        d_model=legacy_width,
        latent_dim=latent_dim,
        encoder_width=encoder_width,
        decoder_width=decoder_width,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        prefix_weight=args.prefix_weight,
        rgb_weight=args.rgb_weight,
        alpha_weight=args.alpha_weight,
        background_alpha_weight=args.background_alpha_weight,
        monotonic_weight=args.monotonic_weight,
        compact_prefix_weight=args.compact_prefix_weight,
        compact_correct_weight=args.compact_correct_weight,
        compact_wrong_bg_weight=args.compact_wrong_bg_weight,
        compact_wrong_color_weight=args.compact_wrong_color_weight,
        compact_missed_fg_weight=args.compact_missed_fg_weight,
        compact_precision_power_start=args.compact_precision_power_start,
        compact_precision_power_end=args.compact_precision_power_end,
        compact_precision_curve_power=args.compact_precision_curve_power,
        correct_pixel_color_tau_start=args.correct_pixel_color_tau_start,
        correct_pixel_color_tau_end=args.correct_pixel_color_tau_end,
        correct_pixel_color_tau_curve_power=args.correct_pixel_color_tau_curve_power,
        correct_pixel_rgb_threshold=args.correct_pixel_rgb_threshold,
        correct_pixel_alpha_threshold=args.correct_pixel_alpha_threshold,
        residual_prefix_weight=args.residual_prefix_weight,
        residual_preserve_weight=args.residual_preserve_weight,
        residual_delta_weight=args.residual_delta_weight,
        residual_needed_weight=args.residual_needed_weight,
        residual_weight_start=args.residual_weight_start,
        residual_weight_end=args.residual_weight_end,
        residual_curve_power=args.residual_curve_power,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        heldout_limit=args.heldout_limit,
        eval_batch_size=args.eval_batch_size,
        greedy_eval_limit=args.greedy_eval_limit,
        sample_count=args.sample_count,
        seed=args.seed,
        amp=not args.no_amp,
        gpu_resident_data=not args.no_gpu_resident_data,
        device=args.device,
    )


def main() -> None:
    run(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
