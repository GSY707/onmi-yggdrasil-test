from __future__ import annotations

import argparse
import binascii
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import struct
import time
import zlib

import torch
from torch import nn
from torch.nn import functional as F

from heterogeneous_latent_input import (
    CELL_COUNT,
    GRID_SIZE,
    MOVES,
    TERRAIN_TYPES,
    HeteroExample,
    TerrainCodec,
    batch_examples,
    effective_move_table,
    examples_to_tensors,
    rollout,
    train_codec,
    transition_table,
)


TILE_SIZE = 8
IMAGE_SIZE = GRID_SIZE * TILE_SIZE


STYLE_NAMES = {
    0: "train_matte",
    1: "train_bright",
    2: "ood_muted",
    3: "ood_high_contrast",
}


STYLE_PALETTES = torch.tensor(
    [
        [
            [0.18, 0.62, 0.28],
            [0.16, 0.36, 0.85],
            [0.80, 0.28, 0.22],
            [0.86, 0.72, 0.20],
        ],
        [
            [0.34, 0.82, 0.42],
            [0.34, 0.56, 0.95],
            [0.95, 0.42, 0.34],
            [0.98, 0.84, 0.30],
        ],
        [
            [0.52, 0.42, 0.72],
            [0.20, 0.68, 0.66],
            [0.76, 0.55, 0.34],
            [0.58, 0.60, 0.18],
        ],
        [
            [0.05, 0.82, 0.78],
            [0.92, 0.20, 0.62],
            [0.96, 0.92, 0.18],
            [0.16, 0.18, 0.24],
        ],
    ],
    dtype=torch.float32,
)

STYLE_PATTERN_CONTRAST = torch.tensor([0.12, 0.16, 0.14, 0.20], dtype=torch.float32)


@dataclass(frozen=True)
class VisualConfig:
    train_move_count: int = 8
    eval_move_counts: tuple[int, ...] = (8, 16)
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 1024
    batch_size: int = 128
    seed: int = 20260701
    train_styles: tuple[int, ...] = (0, 1)
    stage_a_styles: tuple[int, ...] = (0, 1)
    stage_b_styles: tuple[int, ...] = (2, 3)
    d_model: int = 96
    d_latent: int = 32
    codec_steps: int = 300
    terrain_steps: int = 700
    latent_encoder_steps: int = 700
    direct_steps: int = 1000
    eval_every: int = 500
    lr: float = 1e-3


@dataclass(frozen=True)
class VisualExample(HeteroExample):
    style_id: int


def stat(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def generate_visual_examples(
    count: int,
    *,
    seed: int,
    move_count: int,
    style_ids: tuple[int, ...],
) -> list[VisualExample]:
    rng = random.Random(seed)
    examples: list[VisualExample] = []
    for _ in range(count):
        terrain = tuple(rng.randrange(TERRAIN_TYPES) for _ in range(CELL_COUNT))
        start = rng.randrange(CELL_COUNT)
        moves = tuple(rng.randrange(MOVES) for _ in range(move_count))
        style_id = style_ids[rng.randrange(len(style_ids))]
        examples.append(
            VisualExample(
                terrain=terrain,
                start=start,
                moves=moves,
                states=rollout(terrain, start, moves),
                style_id=style_id,
            )
        )
    return examples


def split_examples(config: VisualConfig) -> tuple[list[VisualExample], list[VisualExample], dict[str, dict[int, list[VisualExample]]]]:
    train = generate_visual_examples(
        config.train_size,
        seed=config.seed,
        move_count=config.train_move_count,
        style_ids=config.train_styles,
    )
    val = generate_visual_examples(
        config.val_size,
        seed=config.seed + 1,
        move_count=config.train_move_count,
        style_ids=config.stage_a_styles,
    )
    tests: dict[str, dict[int, list[VisualExample]]] = {"stage_a_same_style": {}, "stage_b_unseen_style": {}}
    for move_count in config.eval_move_counts:
        tests["stage_a_same_style"][move_count] = generate_visual_examples(
            config.test_size,
            seed=config.seed + 1000 + move_count,
            move_count=move_count,
            style_ids=config.stage_a_styles,
        )
        tests["stage_b_unseen_style"][move_count] = generate_visual_examples(
            config.test_size,
            seed=config.seed + 2000 + move_count,
            move_count=move_count,
            style_ids=config.stage_b_styles,
        )
    return train, val, tests


def visual_examples_to_tensors(
    examples: list[VisualExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    terrain, start, moves, states = examples_to_tensors(examples, device=device)
    style_ids = torch.tensor([example.style_id for example in examples], dtype=torch.long, device=device)
    return terrain, start, moves, states, style_ids


def batch_visual_examples(
    examples: list[VisualExample],
    *,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
    return visual_examples_to_tensors(batch, device=device)


def terrain_patterns(device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(TILE_SIZE, device=device),
        torch.arange(TILE_SIZE, device=device),
        indexing="ij",
    )
    patterns = torch.zeros(TERRAIN_TYPES, TILE_SIZE, TILE_SIZE, 1, device=device)
    patterns[0, :, :, 0] = ((xx % 3) == 0).float() - 0.35
    patterns[1, :, :, 0] = ((yy % 3) == 0).float() - 0.35
    patterns[2, :, :, 0] = (((xx + yy) % 4) == 0).float() - 0.25
    patterns[3, :, :, 0] = (((xx // 2 + yy // 2) % 2) == 0).float() - 0.50
    return patterns


def render_images(terrain: torch.Tensor, style_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    palettes = STYLE_PALETTES.to(device)[style_ids]
    contrasts = STYLE_PATTERN_CONTRAST.to(device)[style_ids].view(-1, 1, 1, 1, 1, 1)
    batch_ids = torch.arange(terrain.size(0), device=device).unsqueeze(1)
    cell_colors = palettes[batch_ids, terrain].view(terrain.size(0), GRID_SIZE, GRID_SIZE, 1, 1, 3)
    patterns = terrain_patterns(device)[terrain].view(terrain.size(0), GRID_SIZE, GRID_SIZE, TILE_SIZE, TILE_SIZE, 1)
    image = (cell_colors + patterns * contrasts).clamp(0.0, 1.0)
    image = image.permute(0, 1, 3, 2, 4, 5).reshape(terrain.size(0), IMAGE_SIZE, IMAGE_SIZE, 3)
    return image.permute(0, 3, 1, 2).contiguous()


def cell_means(images: torch.Tensor) -> torch.Tensor:
    batch = images.size(0)
    cells = images.view(batch, 3, GRID_SIZE, TILE_SIZE, GRID_SIZE, TILE_SIZE)
    means = cells.mean(dim=(3, 5)).permute(0, 2, 3, 1)
    return means.reshape(batch, CELL_COUNT, 3)


def parse_with_palette(
    images: torch.Tensor,
    style_ids: torch.Tensor,
    *,
    device: torch.device,
    oracle: bool,
    train_styles: tuple[int, ...],
) -> torch.Tensor:
    means = cell_means(images)
    palettes = STYLE_PALETTES.to(device)
    if oracle:
        per_example_palette = palettes[style_ids]
        distances = ((means.unsqueeze(2) - per_example_palette.unsqueeze(1)) ** 2).sum(dim=-1)
        return distances.argmin(dim=-1)
    train_palette = palettes[torch.tensor(train_styles, dtype=torch.long, device=device)].reshape(-1, 3)
    distances = ((means.unsqueeze(2) - train_palette.view(1, 1, -1, 3)) ** 2).sum(dim=-1)
    nearest = distances.argmin(dim=-1)
    return nearest % TERRAIN_TYPES


@torch.no_grad()
def rollout_from_terrain(
    decoded_terrain: torch.Tensor,
    start: torch.Tensor,
    moves: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    effective = effective_move_table(device)
    transitions = transition_table(device)
    coord = start
    batch_ids = torch.arange(decoded_terrain.size(0), device=device)
    states: list[torch.Tensor] = []
    for index in range(moves.size(1)):
        terrain_type = decoded_terrain[batch_ids, coord]
        effective_moves = effective[terrain_type, moves[:, index]]
        coord = transitions[coord, effective_moves]
        states.append(coord)
    return torch.stack(states, dim=1)


def score_prediction(predicted: torch.Tensor, states: torch.Tensor) -> dict[str, float]:
    return {
        "final_exact": float((predicted[:, -1] == states[:, -1]).float().mean().detach().cpu()),
        "step_exact": float((predicted == states).float().mean().detach().cpu()),
    }


class CellTerrainCNN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 72, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(72, d_model, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(d_model, TERRAIN_TYPES, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(images))
        return logits.permute(0, 2, 3, 1).reshape(images.size(0), CELL_COUNT, TERRAIN_TYPES)


class ImageLatentCodecEncoder(nn.Module):
    def __init__(self, d_model: int, d_latent: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 72, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(72, d_model, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_latent, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        latent = self.features(images)
        return latent.permute(0, 2, 3, 1).reshape(images.size(0), CELL_COUNT, -1)


class VisionStepDirectModel(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 72, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(72, d_model, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.map_projection = nn.Sequential(
            nn.Linear(CELL_COUNT * d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.coord_embedding = nn.Embedding(CELL_COUNT, d_model)
        self.move_embedding = nn.Embedding(MOVES, d_model)
        self.gru = nn.GRU(d_model * 2, d_model, num_layers=2, batch_first=True)
        self.head = nn.Linear(d_model, CELL_COUNT)

    def forward(self, images: torch.Tensor, start: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        features = self.features(images).permute(0, 2, 3, 1).reshape(images.size(0), -1)
        map_context = self.map_projection(features)
        h0 = (map_context + self.coord_embedding(start)).unsqueeze(0).expand(2, -1, -1).contiguous()
        move_context = self.move_embedding(moves)
        repeated_map = map_context.unsqueeze(1).expand(-1, moves.size(1), -1)
        hidden, _ = self.gru(torch.cat([move_context, repeated_map], dim=-1), h0)
        return self.head(hidden)


def train_cell_terrain_cnn(
    model: CellTerrainCNN,
    train: list[VisualExample],
    val: list[VisualExample],
    config: VisualConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.terrain_steps + 1):
        terrain, start, moves, states, style_ids = batch_visual_examples(
            train, rng=rng, batch_size=config.batch_size, device=device
        )
        images = render_images(terrain, style_ids, device=device)
        logits = model(images)
        loss = F.cross_entropy(logits.reshape(-1, TERRAIN_TYPES), terrain.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.terrain_steps:
            metrics = evaluate_cell_terrain_cnn(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"cell_cnn step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"cell={metrics['cell_exact']:.3f} final={metrics['final_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "cost": {
            "train_seconds": round(time.perf_counter() - started, 4),
            "train_steps": config.terrain_steps,
            "trainable_parameters": count_parameters(model),
        },
    }


def train_image_latent_encoder(
    encoder: ImageLatentCodecEncoder,
    codec: TerrainCodec,
    train: list[VisualExample],
    val: list[VisualExample],
    config: VisualConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.latent_encoder_steps + 1):
        terrain, start, moves, states, style_ids = batch_visual_examples(
            train, rng=rng, batch_size=config.batch_size, device=device
        )
        images = render_images(terrain, style_ids, device=device)
        latent = encoder(images)
        logits = codec.decode(latent)
        loss = F.cross_entropy(logits.reshape(-1, TERRAIN_TYPES), terrain.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.latent_encoder_steps:
            metrics = evaluate_image_latent_encoder(encoder, codec, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"image_latent step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"cell={metrics['cell_exact']:.3f} final={metrics['final_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "cost": {
            "train_seconds": round(time.perf_counter() - started, 4),
            "train_steps": config.latent_encoder_steps + config.codec_steps,
            "trainable_parameters": count_parameters(encoder) + sum(parameter.numel() for parameter in codec.parameters()),
        },
    }


def train_vision_step_direct(
    model: VisionStepDirectModel,
    train: list[VisualExample],
    val: list[VisualExample],
    config: VisualConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.direct_steps + 1):
        terrain, start, moves, states, style_ids = batch_visual_examples(
            train, rng=rng, batch_size=config.batch_size, device=device
        )
        images = render_images(terrain, style_ids, device=device)
        logits = model(images, start, moves)
        loss = F.cross_entropy(logits.reshape(-1, CELL_COUNT), states.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.direct_steps:
            metrics = evaluate_vision_step_direct(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"vision_direct step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"final={metrics['final_exact']:.3f} step_exact={metrics['step_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "cost": {
            "train_seconds": round(time.perf_counter() - started, 4),
            "train_steps": config.direct_steps,
            "trainable_parameters": count_parameters(model),
        },
    }


@torch.no_grad()
def evaluate_parser(
    examples: list[VisualExample],
    config: VisualConfig,
    *,
    device: torch.device,
    oracle: bool,
) -> dict[str, float]:
    correct_cell = 0
    correct_final = 0
    correct_step = 0
    total_step = 0
    total_cell = 0
    for index in range(0, len(examples), config.batch_size):
        batch = examples[index : index + config.batch_size]
        terrain, start, moves, states, style_ids = visual_examples_to_tensors(batch, device=device)
        images = render_images(terrain, style_ids, device=device)
        predicted_terrain = parse_with_palette(
            images,
            style_ids,
            device=device,
            oracle=oracle,
            train_styles=config.train_styles,
        )
        predicted = rollout_from_terrain(predicted_terrain, start, moves, device=device)
        correct_cell += int((predicted_terrain == terrain).sum().detach().cpu())
        correct_final += int((predicted[:, -1] == states[:, -1]).sum().detach().cpu())
        correct_step += int((predicted == states).sum().detach().cpu())
        total_cell += terrain.numel()
        total_step += states.numel()
    return {
        "cell_exact": correct_cell / total_cell,
        "final_exact": correct_final / len(examples),
        "step_exact": correct_step / total_step,
    }


@torch.no_grad()
def evaluate_cell_terrain_cnn(
    model: CellTerrainCNN,
    examples: list[VisualExample],
    config: VisualConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct_cell = 0
    correct_final = 0
    correct_step = 0
    total_step = 0
    total_cell = 0
    for index in range(0, len(examples), config.batch_size):
        batch = examples[index : index + config.batch_size]
        terrain, start, moves, states, style_ids = visual_examples_to_tensors(batch, device=device)
        images = render_images(terrain, style_ids, device=device)
        predicted_terrain = model(images).argmax(dim=-1)
        predicted = rollout_from_terrain(predicted_terrain, start, moves, device=device)
        correct_cell += int((predicted_terrain == terrain).sum().detach().cpu())
        correct_final += int((predicted[:, -1] == states[:, -1]).sum().detach().cpu())
        correct_step += int((predicted == states).sum().detach().cpu())
        total_cell += terrain.numel()
        total_step += states.numel()
    model.train()
    return {
        "cell_exact": correct_cell / total_cell,
        "final_exact": correct_final / len(examples),
        "step_exact": correct_step / total_step,
    }


@torch.no_grad()
def evaluate_image_latent_encoder(
    encoder: ImageLatentCodecEncoder,
    codec: TerrainCodec,
    examples: list[VisualExample],
    config: VisualConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    encoder.eval()
    correct_cell = 0
    correct_final = 0
    correct_step = 0
    total_step = 0
    total_cell = 0
    for index in range(0, len(examples), config.batch_size):
        batch = examples[index : index + config.batch_size]
        terrain, start, moves, states, style_ids = visual_examples_to_tensors(batch, device=device)
        images = render_images(terrain, style_ids, device=device)
        logits = codec.decode(encoder(images))
        predicted_terrain = logits.argmax(dim=-1)
        predicted = rollout_from_terrain(predicted_terrain, start, moves, device=device)
        correct_cell += int((predicted_terrain == terrain).sum().detach().cpu())
        correct_final += int((predicted[:, -1] == states[:, -1]).sum().detach().cpu())
        correct_step += int((predicted == states).sum().detach().cpu())
        total_cell += terrain.numel()
        total_step += states.numel()
    encoder.train()
    return {
        "cell_exact": correct_cell / total_cell,
        "final_exact": correct_final / len(examples),
        "step_exact": correct_step / total_step,
    }


@torch.no_grad()
def evaluate_vision_step_direct(
    model: VisionStepDirectModel,
    examples: list[VisualExample],
    config: VisualConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct_final = 0
    correct_step = 0
    total_step = 0
    for index in range(0, len(examples), config.batch_size):
        batch = examples[index : index + config.batch_size]
        terrain, start, moves, states, style_ids = visual_examples_to_tensors(batch, device=device)
        images = render_images(terrain, style_ids, device=device)
        predicted = model(images, start, moves).argmax(dim=-1)
        correct_final += int((predicted[:, -1] == states[:, -1]).sum().detach().cpu())
        correct_step += int((predicted == states).sum().detach().cpu())
        total_step += states.numel()
    model.train()
    return {"cell_exact": 0.0, "final_exact": correct_final / len(examples), "step_exact": correct_step / total_step}


def evaluate_all_splits(
    tests: dict[str, dict[int, list[VisualExample]]],
    config: VisualConfig,
    *,
    evaluator,
) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, by_move in tests.items():
        metrics[split_name] = {}
        for move_count, examples in by_move.items():
            metrics[split_name][str(move_count)] = evaluator(examples)
    return metrics


def summarize_single(results: dict[str, object], config: VisualConfig) -> dict[str, object]:
    baseline_names = list(results.keys())
    final_by_split: dict[str, dict[str, dict[str, float]]] = {}
    cell_by_split: dict[str, dict[str, dict[str, float]]] = {}
    for split_name in ("stage_a_same_style", "stage_b_unseen_style"):
        final_by_split[split_name] = {}
        cell_by_split[split_name] = {}
        for move_count in config.eval_move_counts:
            move_key = str(move_count)
            final_by_split[split_name][move_key] = {
                name: float(results[name]["metrics"][split_name][move_key]["final_exact"])  # type: ignore[index]
                for name in baseline_names
            }
            cell_by_split[split_name][move_key] = {
                name: float(results[name]["metrics"][split_name][move_key]["cell_exact"])  # type: ignore[index]
                for name in baseline_names
            }
    return {
        "final_exact": final_by_split,
        "cell_exact": cell_by_split,
        "overall_final_exact": {
            split_name: {
                name: statistics.fmean(
                    [
                        float(results[name]["metrics"][split_name][str(move_count)]["final_exact"])  # type: ignore[index]
                        for move_count in config.eval_move_counts
                    ]
                )
                for name in baseline_names
            }
            for split_name in ("stage_a_same_style", "stage_b_unseen_style")
        },
        "cost": {name: results[name]["cost"] for name in baseline_names},  # type: ignore[index]
    }


def png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, image_chw: torch.Tensor) -> None:
    image = (image_chw.detach().cpu().clamp(0.0, 1.0) * 255).to(torch.uint8)
    height, width = int(image.shape[1]), int(image.shape[2])
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend([int(image[0, y, x]), int(image[1, y, x]), int(image[2, y, x])])
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    data += png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_sample_images(
    train: list[VisualExample],
    tests: dict[str, dict[int, list[VisualExample]]],
    config: VisualConfig,
    *,
    output_dir: Path,
    device: torch.device,
) -> None:
    samples: list[tuple[str, VisualExample]] = [("train", train[0])]
    first_move = config.eval_move_counts[0]
    samples.append(("stage_a_same_style", tests["stage_a_same_style"][first_move][0]))
    samples.append(("stage_b_unseen_style", tests["stage_b_unseen_style"][first_move][0]))
    for name, example in samples:
        terrain, start, moves, states, style_ids = visual_examples_to_tensors([example], device=device)
        image = render_images(terrain, style_ids, device=device)[0]
        write_png(output_dir / f"{name}_style{int(style_ids[0].detach().cpu())}.png", image)


def run_experiment(config: VisualConfig, output_path: Path) -> dict[str, object]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, tests = split_examples(config)
    print(f"visual device={device} seed={config.seed} train_styles={config.train_styles}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    sample_dir = output_path.parent / "samples" / f"seed{config.seed}"
    write_sample_images(train, tests, config, output_dir=sample_dir, device=device)

    codec = TerrainCodec(config.d_latent).to(device)
    codec_started = time.perf_counter()
    codec_log = train_codec(codec, config, device=device)  # type: ignore[arg-type]
    codec_cost = {
        "train_seconds": round(time.perf_counter() - codec_started, 4),
        "train_steps": config.codec_steps,
        "trainable_parameters": sum(parameter.numel() for parameter in codec.parameters()),
    }

    cell_model = CellTerrainCNN(config.d_model).to(device)
    cell_log = train_cell_terrain_cnn(cell_model, train, val, config, device=device, seed=config.seed + 10)

    latent_encoder = ImageLatentCodecEncoder(config.d_model, config.d_latent).to(device)
    latent_log = train_image_latent_encoder(
        latent_encoder,
        codec,
        train,
        val,
        config,
        device=device,
        seed=config.seed + 20,
    )

    direct_model = VisionStepDirectModel(config.d_model).to(device)
    direct_log = train_vision_step_direct(direct_model, train, val, config, device=device, seed=config.seed + 30)

    results: dict[str, object] = {
        "oracle_visual_parser": {
            "description": "Upper bound parser with access to each sample's style palette.",
            "cost": {"train_seconds": 0.0, "train_steps": 0, "trainable_parameters": 0},
            "metrics": evaluate_all_splits(
                tests,
                config,
                evaluator=lambda examples: evaluate_parser(examples, config, device=device, oracle=True),
            ),
        },
        "train_palette_parser": {
            "description": "Non-learning parser using only train-style palettes; expected to be brittle on Stage B.",
            "cost": {"train_seconds": 0.0, "train_steps": 0, "trainable_parameters": 0},
            "metrics": evaluate_all_splits(
                tests,
                config,
                evaluator=lambda examples: evaluate_parser(examples, config, device=device, oracle=False),
            ),
        },
        "cnn_to_terrain_table": {
            "description": "CNN predicts a terrain table from pixels; deterministic rollout uses the predicted table.",
            "train_log": cell_log["history"],
            "cost": cell_log["cost"],
            "metrics": evaluate_all_splits(
                tests,
                config,
                evaluator=lambda examples: evaluate_cell_terrain_cnn(cell_model, examples, config, device=device),
            ),
        },
        "cnn_to_latent_codec": {
            "description": "CNN predicts per-cell latent vectors decoded by a frozen terrain codec, then rollout.",
            "train_log": {"codec": codec_log, "image_encoder": latent_log["history"]},
            "cost": {
                "train_seconds": round(float(codec_cost["train_seconds"]) + float(latent_log["cost"]["train_seconds"]), 4),  # type: ignore[index]
                "train_steps": config.codec_steps + config.latent_encoder_steps,
                "trainable_parameters": latent_log["cost"]["trainable_parameters"],  # type: ignore[index]
            },
            "metrics": evaluate_all_splits(
                tests,
                config,
                evaluator=lambda examples: evaluate_image_latent_encoder(
                    latent_encoder, codec, examples, config, device=device
                ),
            ),
        },
        "image_to_steps_direct": {
            "description": "End-to-end image + text moves model predicts every intermediate coordinate.",
            "train_log": direct_log["history"],
            "cost": direct_log["cost"],
            "metrics": evaluate_all_splits(
                tests,
                config,
                evaluator=lambda examples: evaluate_vision_step_direct(direct_model, examples, config, device=device),
            ),
        },
    }

    output = {
        "experiment": "visual_multimodal_stage_ab",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "style_names": STYLE_NAMES,
        "config": asdict(config),
        "results": results,
        "summary": summarize_single(results, config),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def summarize_sweep(runs: list[dict[str, object]], eval_move_counts: tuple[int, ...]) -> dict[str, object]:
    baseline_names = list(runs[0]["summary"]["overall_final_exact"]["stage_a_same_style"].keys())  # type: ignore[index]
    summary: dict[str, object] = {"count": len(runs), "final_exact": {}, "cell_exact": {}, "overall_final_exact": {}, "cost": {}}
    for metric_name in ("final_exact", "cell_exact"):
        metric_summary: dict[str, object] = {}
        for split_name in ("stage_a_same_style", "stage_b_unseen_style"):
            metric_summary[split_name] = {}
            for move_count in eval_move_counts:
                move_key = str(move_count)
                metric_summary[split_name][move_key] = {
                    name: stat(
                        [
                            float(run["summary"][metric_name][split_name][move_key][name])  # type: ignore[index]
                            for run in runs
                        ]
                    )
                    for name in baseline_names
                }
        summary[metric_name] = metric_summary
    for split_name in ("stage_a_same_style", "stage_b_unseen_style"):
        summary["overall_final_exact"][split_name] = {
            name: stat(
                [
                    float(run["summary"]["overall_final_exact"][split_name][name])  # type: ignore[index]
                    for run in runs
                ]
            )
            for name in baseline_names
        }
    summary["cost"] = {
        name: {
            "train_seconds": stat([float(run["summary"]["cost"][name]["train_seconds"]) for run in runs]),  # type: ignore[index]
            "train_steps": stat([float(run["summary"]["cost"][name]["train_steps"]) for run in runs]),  # type: ignore[index]
            "trainable_parameters": stat(
                [float(run["summary"]["cost"][name]["trainable_parameters"]) for run in runs]  # type: ignore[index]
            ),
        }
        for name in baseline_names
    }
    return summary


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    eval_move_counts = tuple(int(item) for item in args.eval_move_counts.split(",") if item.strip())
    train_styles = tuple(int(item) for item in args.train_styles.split(",") if item.strip())
    stage_a_styles = tuple(int(item) for item in args.stage_a_styles.split(",") if item.strip())
    stage_b_styles = tuple(int(item) for item in args.stage_b_styles.split(",") if item.strip())
    output_dir = Path(args.output_dir)
    started = time.perf_counter()
    runs: list[dict[str, object]] = []
    for seed in seeds:
        config = VisualConfig(
            train_move_count=args.train_move_count,
            eval_move_counts=eval_move_counts,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            seed=seed,
            train_styles=train_styles,
            stage_a_styles=stage_a_styles,
            stage_b_styles=stage_b_styles,
            d_model=args.d_model,
            d_latent=args.d_latent,
            codec_steps=args.codec_steps,
            terrain_steps=args.terrain_steps,
            latent_encoder_steps=args.latent_encoder_steps,
            direct_steps=args.direct_steps,
            eval_every=args.eval_every,
            lr=args.lr,
        )
        output_path = output_dir / f"seed{seed}.json"
        print(f"=== visual sweep seed={seed} ===", flush=True)
        result = run_experiment(config, output_path)
        runs.append({"seed": seed, "output": str(output_path), "summary": result["summary"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregate = {
        "experiment": "visual_multimodal_stage_ab_sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": runs,
        "summary": summarize_sweep(runs, eval_move_counts),
        "seconds": round(time.perf_counter() - started, 2),
    }
    aggregate_path = Path(args.aggregate)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/visual_multimodal_stage_ab/result.json")
    parser.add_argument("--aggregate", default="artifacts/visual_multimodal_stage_ab/sweep_results.json")
    parser.add_argument("--output-dir", default="artifacts/visual_multimodal_stage_ab/sweep_runs")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--train-move-count", type=int, default=8)
    parser.add_argument("--eval-move-counts", default="8,16")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-styles", default="0,1")
    parser.add_argument("--stage-a-styles", default="0,1")
    parser.add_argument("--stage-b-styles", default="2,3")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--d-latent", type=int, default=32)
    parser.add_argument("--codec-steps", type=int, default=300)
    parser.add_argument("--terrain-steps", type=int, default=700)
    parser.add_argument("--latent-encoder-steps", type=int, default=700)
    parser.add_argument("--direct-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep:
        result = run_sweep(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        return
    config = VisualConfig(
        train_move_count=args.train_move_count,
        eval_move_counts=tuple(int(item) for item in args.eval_move_counts.split(",") if item.strip()),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
        train_styles=tuple(int(item) for item in args.train_styles.split(",") if item.strip()),
        stage_a_styles=tuple(int(item) for item in args.stage_a_styles.split(",") if item.strip()),
        stage_b_styles=tuple(int(item) for item in args.stage_b_styles.split(",") if item.strip()),
        d_model=args.d_model,
        d_latent=args.d_latent,
        codec_steps=args.codec_steps,
        terrain_steps=args.terrain_steps,
        latent_encoder_steps=args.latent_encoder_steps,
        direct_steps=args.direct_steps,
        eval_every=args.eval_every,
        lr=args.lr,
    )
    result = run_experiment(config, Path(args.output))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
