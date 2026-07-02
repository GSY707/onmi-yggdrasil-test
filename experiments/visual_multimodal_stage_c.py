from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import time

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
    examples_to_tensors,
    rollout,
    train_codec,
    transition_table,
)
from visual_multimodal_stage_ab import (
    IMAGE_SIZE,
    TILE_SIZE,
    ImageLatentCodecEncoder,
    cell_means,
    terrain_patterns,
    write_png,
)


SYMBOL_COLORS = torch.tensor(
    [
        [0.92, 0.18, 0.18],
        [0.18, 0.74, 0.28],
        [0.20, 0.38, 0.92],
        [0.96, 0.82, 0.18],
    ],
    dtype=torch.float32,
)
SYMBOL_CONTRAST = torch.tensor([0.15, 0.16, 0.17, 0.18], dtype=torch.float32)


@dataclass(frozen=True)
class StageCConfig:
    train_move_count: int = 8
    eval_move_counts: tuple[int, ...] = (8, 16)
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 1024
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 96
    d_latent: int = 32
    semantic_steps: int = 700
    symbol_steps: int = 700
    latent_encoder_steps: int = 700
    codec_steps: int = 300
    direct_steps: int = 900
    eval_every: int = 500
    lr: float = 1e-3
    probe_budgets: tuple[int, ...] = (0, 1, 2, 4)


@dataclass(frozen=True)
class StageCExample(HeteroExample):
    visual_symbols: tuple[int, ...]
    semantic_to_symbol: tuple[int, ...]
    symbol_to_semantic: tuple[int, ...]


class CellSymbolCNN(nn.Module):
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


class VisionProbeDirectModel(nn.Module):
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
            nn.Linear(CELL_COUNT * d_model + TERRAIN_TYPES * TERRAIN_TYPES, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.coord_embedding = nn.Embedding(CELL_COUNT, d_model)
        self.move_embedding = nn.Embedding(MOVES, d_model)
        self.gru = nn.GRU(d_model * 2, d_model, num_layers=2, batch_first=True)
        self.head = nn.Linear(d_model, CELL_COUNT)

    def forward(self, images: torch.Tensor, probe_matrix: torch.Tensor, start: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        features = self.features(images).permute(0, 2, 3, 1).reshape(images.size(0), -1)
        context_input = torch.cat([features, probe_matrix.reshape(images.size(0), -1).float()], dim=-1)
        map_context = self.map_projection(context_input)
        h0 = (map_context + self.coord_embedding(start)).unsqueeze(0).expand(2, -1, -1).contiguous()
        move_context = self.move_embedding(moves)
        repeated_map = map_context.unsqueeze(1).expand(-1, moves.size(1), -1)
        hidden, _ = self.gru(torch.cat([move_context, repeated_map], dim=-1), h0)
        return self.head(hidden)


def stat(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def inverse_permutation(values: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(values)
    for index, value in enumerate(values):
        inverse[value] = index
    return tuple(inverse)


def generate_examples(
    count: int,
    *,
    seed: int,
    move_count: int,
) -> list[StageCExample]:
    rng = random.Random(seed)
    examples: list[StageCExample] = []
    for _ in range(count):
        semantic_to_symbol = tuple(rng.sample(range(TERRAIN_TYPES), TERRAIN_TYPES))
        symbol_to_semantic = inverse_permutation(semantic_to_symbol)
        terrain = tuple(rng.randrange(TERRAIN_TYPES) for _ in range(CELL_COUNT))
        visual_symbols = tuple(semantic_to_symbol[item] for item in terrain)
        start = rng.randrange(CELL_COUNT)
        moves = tuple(rng.randrange(MOVES) for _ in range(move_count))
        examples.append(
            StageCExample(
                terrain=terrain,
                visual_symbols=visual_symbols,
                semantic_to_symbol=semantic_to_symbol,
                symbol_to_semantic=symbol_to_semantic,
                start=start,
                moves=moves,
                states=rollout(terrain, start, moves),
            )
        )
    return examples


def split_examples(config: StageCConfig) -> tuple[list[StageCExample], list[StageCExample], dict[int, list[StageCExample]]]:
    train = generate_examples(config.train_size, seed=config.seed, move_count=config.train_move_count)
    val = generate_examples(config.val_size, seed=config.seed + 1, move_count=config.train_move_count)
    tests = {
        move_count: generate_examples(config.test_size, seed=config.seed + 1000 + move_count, move_count=move_count)
        for move_count in config.eval_move_counts
    }
    return train, val, tests


def examples_to_stage_c_tensors(
    examples: list[StageCExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    terrain, start, moves, states = examples_to_tensors(examples, device=device)
    visual_symbols = torch.tensor([example.visual_symbols for example in examples], dtype=torch.long, device=device)
    semantic_to_symbol = torch.tensor([example.semantic_to_symbol for example in examples], dtype=torch.long, device=device)
    symbol_to_semantic = torch.tensor([example.symbol_to_semantic for example in examples], dtype=torch.long, device=device)
    return terrain, visual_symbols, semantic_to_symbol, symbol_to_semantic, start, moves, states


def batch_examples(
    examples: list[StageCExample],
    *,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
    return examples_to_stage_c_tensors(batch, device=device)


def render_symbol_images(visual_symbols: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    colors = SYMBOL_COLORS.to(device)[visual_symbols].view(visual_symbols.size(0), GRID_SIZE, GRID_SIZE, 1, 1, 3)
    contrasts = SYMBOL_CONTRAST.to(device)[visual_symbols].view(visual_symbols.size(0), GRID_SIZE, GRID_SIZE, 1, 1, 1)
    patterns = terrain_patterns(device)[visual_symbols].view(
        visual_symbols.size(0),
        GRID_SIZE,
        GRID_SIZE,
        TILE_SIZE,
        TILE_SIZE,
        1,
    )
    image = (colors + patterns * contrasts).clamp(0.0, 1.0)
    image = image.permute(0, 1, 3, 2, 4, 5).reshape(visual_symbols.size(0), IMAGE_SIZE, IMAGE_SIZE, 3)
    return image.permute(0, 3, 1, 2).contiguous()


def parse_symbols_by_palette(images: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    means = cell_means(images)
    distances = ((means.unsqueeze(2) - SYMBOL_COLORS.to(device).view(1, 1, TERRAIN_TYPES, 3)) ** 2).sum(dim=-1)
    return distances.argmin(dim=-1)


def build_probe_mapping(
    visual_symbols: torch.Tensor,
    symbol_to_semantic: torch.Tensor,
    *,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = torch.full((visual_symbols.size(0), TERRAIN_TYPES), -1, dtype=torch.long, device=visual_symbols.device)
    probes_used = torch.zeros(visual_symbols.size(0), dtype=torch.long, device=visual_symbols.device)
    for batch_index in range(visual_symbols.size(0)):
        present: list[int] = []
        for symbol_id in range(TERRAIN_TYPES):
            if bool((visual_symbols[batch_index] == symbol_id).any().detach().cpu()):
                present.append(symbol_id)
        for symbol_id in present[:budget]:
            mapping[batch_index, symbol_id] = symbol_to_semantic[batch_index, symbol_id]
            probes_used[batch_index] += 1
    return mapping, probes_used


def mapping_to_matrix(mapping: torch.Tensor) -> torch.Tensor:
    matrix = torch.zeros(mapping.size(0), TERRAIN_TYPES, TERRAIN_TYPES, device=mapping.device)
    known = mapping >= 0
    for semantic_id in range(TERRAIN_TYPES):
        matrix[:, :, semantic_id] = (mapping == semantic_id).float()
    matrix[:, :, 0] = torch.where(known, matrix[:, :, 0], torch.zeros_like(matrix[:, :, 0]))
    return matrix


def decode_semantics(predicted_symbols: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    fallback = predicted_symbols
    safe_mapping = mapping.clamp_min(0)
    mapped = torch.gather(safe_mapping, 1, predicted_symbols)
    known = torch.gather(mapping >= 0, 1, predicted_symbols)
    return torch.where(known, mapped, fallback)


@torch.no_grad()
def rollout_from_terrain(
    terrain: torch.Tensor,
    start: torch.Tensor,
    moves: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    transitions = transition_table(device)
    coord = start
    batch_ids = torch.arange(terrain.size(0), device=device)
    states: list[torch.Tensor] = []
    effective_table = torch.empty(TERRAIN_TYPES, MOVES, dtype=torch.long, device=device)
    effective_table[0] = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    effective_table[1] = torch.tensor([3, 2, 0, 1], dtype=torch.long, device=device)
    effective_table[2] = torch.tensor([2, 3, 1, 0], dtype=torch.long, device=device)
    effective_table[3] = torch.tensor([1, 0, 3, 2], dtype=torch.long, device=device)
    for index in range(moves.size(1)):
        current = terrain[batch_ids, coord]
        effective_moves = effective_table[current, moves[:, index]]
        coord = transitions[coord, effective_moves]
        states.append(coord)
    return torch.stack(states, dim=1)


def score(predicted: torch.Tensor, states: torch.Tensor) -> dict[str, float]:
    return {
        "final_exact": float((predicted[:, -1] == states[:, -1]).float().mean().detach().cpu()),
        "step_exact": float((predicted == states).float().mean().detach().cpu()),
    }


def metric_result(
    *,
    predicted_states: torch.Tensor,
    states: torch.Tensor,
    predicted_symbols: torch.Tensor | None,
    visual_symbols: torch.Tensor,
    predicted_terrain: torch.Tensor,
    terrain: torch.Tensor,
    probes_used: torch.Tensor,
) -> dict[str, float]:
    result = score(predicted_states, states)
    result["symbol_cell_exact"] = (
        float((predicted_symbols == visual_symbols).float().mean().detach().cpu()) if predicted_symbols is not None else 0.0
    )
    result["semantic_cell_exact"] = float((predicted_terrain == terrain).float().mean().detach().cpu())
    result["avg_probes"] = float(probes_used.float().mean().detach().cpu())
    return result


def merge_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys()
    return {key: statistics.fmean([item[key] for item in items]) for key in keys}


def train_cell_model(
    model: CellSymbolCNN,
    train: list[StageCExample],
    val: list[StageCExample],
    config: StageCConfig,
    *,
    device: torch.device,
    seed: int,
    target: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    steps = config.semantic_steps if target == "semantic" else config.symbol_steps
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        terrain, visual_symbols, semantic_to_symbol, symbol_to_semantic, start, moves, states = batch_examples(
            train, rng=rng, batch_size=config.batch_size, device=device
        )
        labels = terrain if target == "semantic" else visual_symbols
        images = render_symbol_images(visual_symbols, device=device)
        logits = model(images)
        loss = F.cross_entropy(logits.reshape(-1, TERRAIN_TYPES), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            if target == "semantic":
                metrics = evaluate_semantic_cnn(model, val, config, device=device)
            else:
                metrics = evaluate_symbol_probe(model, val, config, device=device, budget=TERRAIN_TYPES, mode="cnn_symbol")
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"{target}_cnn step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"final={metrics['final_exact']:.3f} semantic_cell={metrics['semantic_cell_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "cost": {
            "train_seconds": round(time.perf_counter() - started, 4),
            "train_steps": steps,
            "trainable_parameters": count_parameters(model),
        },
    }


def train_latent_symbol_encoder(
    encoder: ImageLatentCodecEncoder,
    codec: TerrainCodec,
    train: list[StageCExample],
    val: list[StageCExample],
    config: StageCConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.latent_encoder_steps + 1):
        terrain, visual_symbols, semantic_to_symbol, symbol_to_semantic, start, moves, states = batch_examples(
            train, rng=rng, batch_size=config.batch_size, device=device
        )
        images = render_symbol_images(visual_symbols, device=device)
        logits = codec.decode(encoder(images))
        loss = F.cross_entropy(logits.reshape(-1, TERRAIN_TYPES), visual_symbols.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.latent_encoder_steps:
            metrics = evaluate_symbol_probe(
                encoder,
                val,
                config,
                device=device,
                budget=TERRAIN_TYPES,
                mode="latent_symbol",
                codec=codec,
            )
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"latent_symbol step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"final={metrics['final_exact']:.3f} semantic_cell={metrics['semantic_cell_exact']:.3f}",
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


def train_direct_probe_model(
    model: VisionProbeDirectModel,
    train: list[StageCExample],
    val: list[StageCExample],
    config: StageCConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.direct_steps + 1):
        terrain, visual_symbols, semantic_to_symbol, symbol_to_semantic, start, moves, states = batch_examples(
            train, rng=rng, batch_size=config.batch_size, device=device
        )
        images = render_symbol_images(visual_symbols, device=device)
        mapping, probes_used = build_probe_mapping(visual_symbols, symbol_to_semantic, budget=TERRAIN_TYPES)
        logits = model(images, mapping_to_matrix(mapping), start, moves)
        loss = F.cross_entropy(logits.reshape(-1, CELL_COUNT), states.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.direct_steps:
            metrics = evaluate_direct_probe_model(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"direct_probe step={step:4d} loss={float(loss.detach().cpu()):.4f} "
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
def evaluate_semantic_cnn(
    model: CellSymbolCNN,
    examples: list[StageCExample],
    config: StageCConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    items: list[dict[str, float]] = []
    for index in range(0, len(examples), config.batch_size):
        batch = examples[index : index + config.batch_size]
        terrain, visual_symbols, semantic_to_symbol, symbol_to_semantic, start, moves, states = examples_to_stage_c_tensors(
            batch, device=device
        )
        images = render_symbol_images(visual_symbols, device=device)
        predicted_terrain = model(images).argmax(dim=-1)
        predicted_states = rollout_from_terrain(predicted_terrain, start, moves, device=device)
        items.append(
            metric_result(
                predicted_states=predicted_states,
                states=states,
                predicted_symbols=None,
                visual_symbols=visual_symbols,
                predicted_terrain=predicted_terrain,
                terrain=terrain,
                probes_used=torch.zeros(terrain.size(0), device=device),
            )
        )
    model.train()
    return merge_metrics(items)


@torch.no_grad()
def evaluate_symbol_probe(
    model: CellSymbolCNN | ImageLatentCodecEncoder | None,
    examples: list[StageCExample],
    config: StageCConfig,
    *,
    device: torch.device,
    budget: int,
    mode: str,
    codec: TerrainCodec | None = None,
) -> dict[str, float]:
    if isinstance(model, nn.Module):
        model.eval()
    items: list[dict[str, float]] = []
    for index in range(0, len(examples), config.batch_size):
        batch = examples[index : index + config.batch_size]
        terrain, visual_symbols, semantic_to_symbol, symbol_to_semantic, start, moves, states = examples_to_stage_c_tensors(
            batch, device=device
        )
        images = render_symbol_images(visual_symbols, device=device)
        if mode == "rule_symbol":
            predicted_symbols = parse_symbols_by_palette(images, device=device)
        elif mode == "cnn_symbol":
            assert isinstance(model, CellSymbolCNN)
            predicted_symbols = model(images).argmax(dim=-1)
        elif mode == "latent_symbol":
            assert isinstance(model, ImageLatentCodecEncoder)
            assert codec is not None
            predicted_symbols = codec.decode(model(images)).argmax(dim=-1)
        else:
            raise ValueError(f"unknown mode: {mode}")
        mapping, probes_used = build_probe_mapping(visual_symbols, symbol_to_semantic, budget=budget)
        predicted_terrain = decode_semantics(predicted_symbols, mapping)
        predicted_states = rollout_from_terrain(predicted_terrain, start, moves, device=device)
        items.append(
            metric_result(
                predicted_states=predicted_states,
                states=states,
                predicted_symbols=predicted_symbols,
                visual_symbols=visual_symbols,
                predicted_terrain=predicted_terrain,
                terrain=terrain,
                probes_used=probes_used,
            )
        )
    if isinstance(model, nn.Module):
        model.train()
    return merge_metrics(items)


@torch.no_grad()
def evaluate_direct_probe_model(
    model: VisionProbeDirectModel,
    examples: list[StageCExample],
    config: StageCConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    items: list[dict[str, float]] = []
    for index in range(0, len(examples), config.batch_size):
        batch = examples[index : index + config.batch_size]
        terrain, visual_symbols, semantic_to_symbol, symbol_to_semantic, start, moves, states = examples_to_stage_c_tensors(
            batch, device=device
        )
        images = render_symbol_images(visual_symbols, device=device)
        mapping, probes_used = build_probe_mapping(visual_symbols, symbol_to_semantic, budget=TERRAIN_TYPES)
        predicted_states = model(images, mapping_to_matrix(mapping), start, moves).argmax(dim=-1)
        items.append(
            {
                **score(predicted_states, states),
                "symbol_cell_exact": 0.0,
                "semantic_cell_exact": 0.0,
                "avg_probes": float(probes_used.float().mean().detach().cpu()),
            }
        )
    model.train()
    return merge_metrics(items)


def evaluate_by_move(
    tests: dict[int, list[StageCExample]],
    *,
    evaluator,
) -> dict[str, dict[str, float]]:
    return {str(move_count): evaluator(examples) for move_count, examples in tests.items()}


def summarize_single(results: dict[str, object], config: StageCConfig) -> dict[str, object]:
    baseline_names = list(results.keys())
    metrics: dict[str, dict[str, dict[str, float]]] = {key: {} for key in ("final_exact", "step_exact", "semantic_cell_exact", "avg_probes")}
    for move_count in config.eval_move_counts:
        move_key = str(move_count)
        for metric_name in metrics:
            metrics[metric_name][move_key] = {
                name: float(results[name]["metrics_by_move"][move_key][metric_name])  # type: ignore[index]
                for name in baseline_names
            }
    metrics["overall_final_exact"] = {
        "all_moves": {
            name: statistics.fmean(
                [
                    float(results[name]["metrics_by_move"][str(move_count)]["final_exact"])  # type: ignore[index]
                    for move_count in config.eval_move_counts
                ]
            )
            for name in baseline_names
        }
    }
    return {
        **metrics,
        "cost": {name: results[name]["cost"] for name in baseline_names},  # type: ignore[index]
    }


def write_sample_images(train: list[StageCExample], tests: dict[int, list[StageCExample]], config: StageCConfig, *, output_dir: Path, device: torch.device) -> None:
    first_move = config.eval_move_counts[0]
    samples = [("train", train[0]), ("test", tests[first_move][0])]
    for name, example in samples:
        _, visual_symbols, _, _, _, _, _ = examples_to_stage_c_tensors([example], device=device)
        image = render_symbol_images(visual_symbols, device=device)[0]
        write_png(output_dir / f"{name}.png", image)


def run_experiment(config: StageCConfig, output_path: Path) -> dict[str, object]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, tests = split_examples(config)
    print(f"stage_c device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    write_sample_images(train, tests, config, output_dir=output_path.parent / "samples" / f"seed{config.seed}", device=device)

    semantic_model = CellSymbolCNN(config.d_model).to(device)
    semantic_log = train_cell_model(
        semantic_model,
        train,
        val,
        config,
        device=device,
        seed=config.seed + 10,
        target="semantic",
    )

    symbol_model = CellSymbolCNN(config.d_model).to(device)
    symbol_log = train_cell_model(
        symbol_model,
        train,
        val,
        config,
        device=device,
        seed=config.seed + 20,
        target="symbol",
    )

    codec = TerrainCodec(config.d_latent).to(device)
    codec_started = time.perf_counter()
    codec_log = train_codec(codec, config, device=device)  # type: ignore[arg-type]
    codec_cost_seconds = round(time.perf_counter() - codec_started, 4)
    latent_encoder = ImageLatentCodecEncoder(config.d_model, config.d_latent).to(device)
    latent_log = train_latent_symbol_encoder(
        latent_encoder,
        codec,
        train,
        val,
        config,
        device=device,
        seed=config.seed + 30,
    )
    latent_log["cost"]["train_seconds"] = round(float(latent_log["cost"]["train_seconds"]) + codec_cost_seconds, 4)  # type: ignore[index]

    direct_model = VisionProbeDirectModel(config.d_model).to(device)
    direct_log = train_direct_probe_model(direct_model, train, val, config, device=device, seed=config.seed + 40)

    zero_cost = {"train_seconds": 0.0, "train_steps": 0, "trainable_parameters": 0}
    results: dict[str, object] = {
        "oracle_semantic": {
            "description": "Uses the ground-truth semantic terrain table; upper bound.",
            "cost": zero_cost,
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_oracle_semantic(examples, config, device=device),
            ),
        },
        "passive_symbol_identity": {
            "description": "Parses visual symbols and assumes symbol id equals terrain semantics; no probes.",
            "cost": zero_cost,
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_symbol_probe(
                    None, examples, config, device=device, budget=0, mode="rule_symbol"
                ),
            ),
        },
        "cnn_semantic_no_probe": {
            "description": "CNN tries to predict semantic terrain directly from symbols despite per-map random bindings.",
            "train_log": semantic_log["history"],
            "cost": semantic_log["cost"],
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_semantic_cnn(semantic_model, examples, config, device=device),
            ),
        },
        "rule_symbol_probe_1": {
            "description": "Rule symbol parser plus one active probe per map.",
            "cost": zero_cost,
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_symbol_probe(
                    None, examples, config, device=device, budget=1, mode="rule_symbol"
                ),
            ),
        },
        "rule_symbol_probe_2": {
            "description": "Rule symbol parser plus two active probes per map.",
            "cost": zero_cost,
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_symbol_probe(
                    None, examples, config, device=device, budget=2, mode="rule_symbol"
                ),
            ),
        },
        "rule_symbol_probe_4": {
            "description": "Rule symbol parser plus full active probing of all visual symbols.",
            "cost": zero_cost,
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_symbol_probe(
                    None, examples, config, device=device, budget=4, mode="rule_symbol"
                ),
            ),
        },
        "cnn_symbol_probe_4": {
            "description": "CNN predicts visual symbols; active probes map symbols to semantics.",
            "train_log": symbol_log["history"],
            "cost": symbol_log["cost"],
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_symbol_probe(
                    symbol_model, examples, config, device=device, budget=4, mode="cnn_symbol"
                ),
            ),
        },
        "latent_symbol_probe_4": {
            "description": "Image encoder predicts latent visual symbols; active probes map decoded symbols to semantics.",
            "train_log": {"codec": codec_log, "image_encoder": latent_log["history"]},
            "cost": latent_log["cost"],
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_symbol_probe(
                    latent_encoder,
                    examples,
                    config,
                    device=device,
                    budget=4,
                    mode="latent_symbol",
                    codec=codec,
                ),
            ),
        },
        "image_probe_direct": {
            "description": "End-to-end image model receives the full probe legend and predicts steps directly.",
            "train_log": direct_log["history"],
            "cost": direct_log["cost"],
            "metrics_by_move": evaluate_by_move(
                tests,
                evaluator=lambda examples: evaluate_direct_probe_model(direct_model, examples, config, device=device),
            ),
        },
    }
    output = {
        "experiment": "visual_multimodal_stage_c",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": asdict(config),
        "results": results,
        "summary": summarize_single(results, config),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


@torch.no_grad()
def evaluate_oracle_semantic(
    examples: list[StageCExample],
    config: StageCConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    items: list[dict[str, float]] = []
    for index in range(0, len(examples), config.batch_size):
        batch = examples[index : index + config.batch_size]
        terrain, visual_symbols, semantic_to_symbol, symbol_to_semantic, start, moves, states = examples_to_stage_c_tensors(
            batch, device=device
        )
        predicted_states = rollout_from_terrain(terrain, start, moves, device=device)
        items.append(
            metric_result(
                predicted_states=predicted_states,
                states=states,
                predicted_symbols=visual_symbols,
                visual_symbols=visual_symbols,
                predicted_terrain=terrain,
                terrain=terrain,
                probes_used=torch.zeros(terrain.size(0), device=device),
            )
        )
    return merge_metrics(items)


def summarize_sweep(runs: list[dict[str, object]], eval_move_counts: tuple[int, ...]) -> dict[str, object]:
    names = list(runs[0]["summary"]["overall_final_exact"]["all_moves"].keys())  # type: ignore[index]
    summary: dict[str, object] = {"count": len(runs), "final_exact": {}, "semantic_cell_exact": {}, "avg_probes": {}, "overall_final_exact": {}, "cost": {}}
    for metric_name in ("final_exact", "semantic_cell_exact", "avg_probes"):
        summary[metric_name] = {
            str(move_count): {
                name: stat(
                    [
                        float(run["summary"][metric_name][str(move_count)][name])  # type: ignore[index]
                        for run in runs
                    ]
                )
                for name in names
            }
            for move_count in eval_move_counts
        }
    summary["overall_final_exact"] = {
        "all_moves": {
            name: stat(
                [
                    float(run["summary"]["overall_final_exact"]["all_moves"][name])  # type: ignore[index]
                    for run in runs
                ]
            )
            for name in names
        }
    }
    summary["cost"] = {
        name: {
            "train_seconds": stat([float(run["summary"]["cost"][name]["train_seconds"]) for run in runs]),  # type: ignore[index]
            "train_steps": stat([float(run["summary"]["cost"][name]["train_steps"]) for run in runs]),  # type: ignore[index]
            "trainable_parameters": stat(
                [float(run["summary"]["cost"][name]["trainable_parameters"]) for run in runs]  # type: ignore[index]
            ),
        }
        for name in names
    }
    return summary


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    eval_move_counts = tuple(int(item) for item in args.eval_move_counts.split(",") if item.strip())
    probe_budgets = tuple(int(item) for item in args.probe_budgets.split(",") if item.strip())
    output_dir = Path(args.output_dir)
    started = time.perf_counter()
    runs: list[dict[str, object]] = []
    for seed in seeds:
        config = StageCConfig(
            train_move_count=args.train_move_count,
            eval_move_counts=eval_move_counts,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            seed=seed,
            d_model=args.d_model,
            d_latent=args.d_latent,
            semantic_steps=args.semantic_steps,
            symbol_steps=args.symbol_steps,
            latent_encoder_steps=args.latent_encoder_steps,
            codec_steps=args.codec_steps,
            direct_steps=args.direct_steps,
            eval_every=args.eval_every,
            lr=args.lr,
            probe_budgets=probe_budgets,
        )
        output_path = output_dir / f"seed{seed}.json"
        print(f"=== stage C sweep seed={seed} ===", flush=True)
        result = run_experiment(config, output_path)
        runs.append({"seed": seed, "output": str(output_path), "summary": result["summary"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregate = {
        "experiment": "visual_multimodal_stage_c_sweep",
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
    parser.add_argument("--output", default="artifacts/visual_multimodal_stage_c/result.json")
    parser.add_argument("--aggregate", default="artifacts/visual_multimodal_stage_c/sweep_results.json")
    parser.add_argument("--output-dir", default="artifacts/visual_multimodal_stage_c/sweep_runs")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--train-move-count", type=int, default=8)
    parser.add_argument("--eval-move-counts", default="8,16")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--d-latent", type=int, default=32)
    parser.add_argument("--semantic-steps", type=int, default=700)
    parser.add_argument("--symbol-steps", type=int, default=700)
    parser.add_argument("--latent-encoder-steps", type=int, default=700)
    parser.add_argument("--codec-steps", type=int, default=300)
    parser.add_argument("--direct-steps", type=int, default=900)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-budgets", default="0,1,2,4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep:
        result = run_sweep(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        return
    config = StageCConfig(
        train_move_count=args.train_move_count,
        eval_move_counts=tuple(int(item) for item in args.eval_move_counts.split(",") if item.strip()),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
        d_model=args.d_model,
        d_latent=args.d_latent,
        semantic_steps=args.semantic_steps,
        symbol_steps=args.symbol_steps,
        latent_encoder_steps=args.latent_encoder_steps,
        codec_steps=args.codec_steps,
        direct_steps=args.direct_steps,
        eval_every=args.eval_every,
        lr=args.lr,
        probe_budgets=tuple(int(item) for item in args.probe_budgets.split(",") if item.strip()),
    )
    result = run_experiment(config, Path(args.output))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
