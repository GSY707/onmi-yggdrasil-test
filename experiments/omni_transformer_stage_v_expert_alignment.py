from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import omni_transformer_stage_m_moe_multimodal_llm as stage_m
import omni_transformer_stage_q_tiny_moe_vlm_from_scratch as stage_q
import omni_transformer_stage_u_object_slots as stage_u


MODEL_MODES = ("ranking_only_object_latent", "supervised_shared_semantic_latent")
DIAGNOSTIC_STAGES = ("patch_tokens", "object_slots", "spatial_tokens", "count_tokens", "fusion_tokens", "wide_latent")


@dataclass(frozen=True)
class StageVConfig:
    train_size: int = 1024
    val_size: int = 256
    test_size: int = 384
    batch_size: int = 64
    seed: int = 20260701
    image_size: int = 64
    grid_size: int = 4
    prompt_len: int = 112
    answer_len: int = 32
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    patch_grid: int = 8
    object_slots: int = 8
    spatial_tokens: int = 8
    count_tokens: int = 8
    prompt_tokens: int = 8
    fusion_tokens: int = 8
    output_tokens: int = 8
    candidate_count: int = 8
    train_steps: int = 420
    eval_every: int = 140
    router_loss_weight: float = 0.05
    semantic_loss_weight: float = 0.35
    probe_steps: int = 70
    probe_hidden: int = 192
    modes: tuple[str, ...] = MODEL_MODES


@dataclass
class SemanticPixelSet:
    pixels: stage_q.PixelSet
    occupancy: torch.Tensor
    color: torch.Tensor
    shape: torch.Tensor
    meta: list[dict[str, object]]

    def subset(self, indices: list[int]) -> "SemanticPixelSet":
        return SemanticPixelSet(
            pixels=self.pixels.subset(indices),
            occupancy=self.occupancy[indices],
            color=self.color[indices],
            shape=self.shape[indices],
            meta=[self.meta[index] for index in indices],
        )

    def to(self, device: torch.device) -> "SemanticPixelSet":
        return SemanticPixelSet(
            pixels=self.pixels.to(device),
            occupancy=self.occupancy.to(device=device, dtype=torch.float32),
            color=self.color.to(device=device, dtype=torch.long),
            shape=self.shape.to(device=device, dtype=torch.long),
            meta=self.meta,
        )


def stage_u_config(config: StageVConfig) -> stage_u.StageUConfig:
    return stage_u.StageUConfig(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        image_size=config.image_size,
        grid_size=config.grid_size,
        prompt_len=config.prompt_len,
        answer_len=config.answer_len,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        lr=config.lr,
        patch_grid=config.patch_grid,
        object_slots=config.object_slots,
        spatial_tokens=config.spatial_tokens,
        count_tokens=config.count_tokens,
        prompt_tokens=config.prompt_tokens,
        fusion_tokens=config.fusion_tokens,
        output_tokens=config.output_tokens,
        candidate_count=config.candidate_count,
        train_steps=config.train_steps,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
        patch_context_dropout=0.0,
        probe_steps=config.probe_steps,
        probe_hidden=config.probe_hidden,
        variants=("object_slot_spatial_wide_latent",),
    )


def semantic_targets(meta: list[dict[str, object]], config: StageVConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cell_count = config.grid_size * config.grid_size
    occupancy = torch.zeros(len(meta), cell_count, dtype=torch.float32)
    color = torch.zeros(len(meta), cell_count, dtype=torch.long)
    shape = torch.zeros(len(meta), cell_count, dtype=torch.long)
    for batch_index, item in enumerate(meta):
        for obj in item["objects"]:
            cell = int(obj["row"]) * config.grid_size + int(obj["col"])
            occupancy[batch_index, cell] = 1.0
            color[batch_index, cell] = int(obj["color"]) + 1
            shape[batch_index, cell] = int(obj["shape"]) + 1
    return occupancy, color, shape


def build_semantic_data(config: StageVConfig) -> tuple[dict[str, SemanticPixelSet], dict[str, object]]:
    pixels, stats, meta = stage_u.build_data(stage_u_config(config))
    semantic = {}
    for split_name in ("train", "val", "test"):
        occupancy, color, shape = semantic_targets(meta[split_name], config)
        semantic[split_name] = SemanticPixelSet(
            pixels=pixels[split_name],
            occupancy=occupancy,
            color=color,
            shape=shape,
            meta=meta[split_name],
        )
    stats["semantic_targets"] = {
        "cells": config.grid_size * config.grid_size,
        "color_classes": len(stage_u.COLORS) + 1,
        "shape_classes": len(stage_u.SHAPES) + 1,
        "none_class": 0,
    }
    return semantic, stats


def random_semantic_batch(data: SemanticPixelSet, *, rng: random.Random, batch_size: int, device: torch.device) -> SemanticPixelSet:
    return data.subset([rng.randrange(len(data.pixels.examples)) for _ in range(batch_size)]).to(device)


class GridSemanticDecoder(nn.Module):
    def __init__(self, config: StageVConfig) -> None:
        super().__init__()
        cell_count = config.grid_size * config.grid_size
        self.query = nn.Parameter(torch.randn(cell_count, config.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_kv = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )
        self.occupancy = nn.Linear(config.d_model, 1)
        self.color = nn.Linear(config.d_model, len(stage_u.COLORS) + 1)
        self.shape = nn.Linear(config.d_model, len(stage_u.SHAPES) + 1)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.attn(self.norm_q(query), self.norm_kv(tokens), self.norm_kv(tokens), need_weights=False)
        hidden = query + attended
        hidden = hidden + self.ff(hidden)
        return {
            "occupancy": self.occupancy(hidden).squeeze(-1),
            "color": self.color(hidden),
            "shape": self.shape(hidden),
        }


def semantic_loss(output: dict[str, torch.Tensor], batch: SemanticPixelSet) -> torch.Tensor:
    occupancy_loss = F.binary_cross_entropy_with_logits(output["occupancy"], batch.occupancy)
    occupied = batch.occupancy.bool()
    if occupied.any():
        color_loss = F.cross_entropy(output["color"][occupied], batch.color[occupied])
        shape_loss = F.cross_entropy(output["shape"][occupied], batch.shape[occupied])
    else:
        color_loss = output["color"].sum() * 0.0
        shape_loss = output["shape"].sum() * 0.0
    return occupancy_loss + color_loss + shape_loss


@torch.no_grad()
def semantic_metrics(output: dict[str, torch.Tensor], batch: SemanticPixelSet) -> dict[str, float]:
    occupancy_pred = (torch.sigmoid(output["occupancy"]) >= 0.5)
    occupancy_true = batch.occupancy.bool()
    color_pred = output["color"].argmax(dim=-1)
    shape_pred = output["shape"].argmax(dim=-1)
    occupied = occupancy_true
    occupancy_acc = (occupancy_pred == occupancy_true).float().mean().item()
    if occupied.any():
        color_acc = (color_pred[occupied] == batch.color[occupied]).float().mean().item()
        shape_acc = (shape_pred[occupied] == batch.shape[occupied]).float().mean().item()
    else:
        color_acc = 1.0
        shape_acc = 1.0
    cell_correct = (~occupancy_true & ~occupancy_pred) | (
        occupancy_true
        & occupancy_pred
        & (color_pred == batch.color)
        & (shape_pred == batch.shape)
    )
    cell_info_accuracy = cell_correct.float().mean().item()
    scene_exact = cell_correct.all(dim=1).float().mean().item()
    return {
        "occupancy_accuracy": occupancy_acc,
        "occupied_color_accuracy": color_acc,
        "occupied_shape_accuracy": shape_acc,
        "cell_info_accuracy": cell_info_accuracy,
        "cell_info_loss": 1.0 - cell_info_accuracy,
        "scene_exact": scene_exact,
        "scene_info_loss": 1.0 - scene_exact,
    }


def merge_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: statistics.fmean(row[key] for row in rows) for key in keys}


@torch.no_grad()
def evaluate_semantic_decoder(
    model: stage_u.StageUWideVLM,
    decoder: GridSemanticDecoder,
    stage_name: str,
    data: SemanticPixelSet,
    config: StageVConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    decoder.eval()
    rows = []
    for start in range(0, len(data.pixels.examples), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(data.pixels.examples))))
        batch = data.subset(indices).to(device)
        tokens = model.representations(batch.pixels)[stage_name]
        rows.append(semantic_metrics(decoder(tokens), batch))
    return merge_metric_rows(rows)


def supervised_stage_loss(
    model_output: dict[str, torch.Tensor],
    decoder: GridSemanticDecoder,
    batch: SemanticPixelSet,
    stages: Iterable[str] = DIAGNOSTIC_STAGES,
) -> torch.Tensor:
    losses = [semantic_loss(decoder(model_output[stage_name]), batch) for stage_name in stages]
    return torch.stack(losses).mean()


def train_model(
    mode: str,
    train: SemanticPixelSet,
    val: SemanticPixelSet,
    config: StageVConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> tuple[stage_u.StageUWideVLM, GridSemanticDecoder | None, dict[str, object]]:
    if mode not in MODEL_MODES:
        raise ValueError(f"unknown mode: {mode}")
    rng = random.Random(seed)
    u_config = stage_u_config(config)
    model = stage_u.StageUWideVLM(u_config, variant="object_slot_spatial_wide_latent").to(device)
    shared_decoder = GridSemanticDecoder(config).to(device) if mode == "supervised_shared_semantic_latent" else None
    params = list(model.parameters()) + ([] if shared_decoder is None else list(shared_decoder.parameters()))
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    best_top1 = -1.0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_decoder_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for step in range(1, config.train_steps + 1):
        batch = random_semantic_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        rows, true_indices = stage_u.candidate_rows_for_examples(batch.pixels.examples, candidate_pool, seed=rng.randrange(1_000_000_000))
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        output = model(batch.pixels, answers)
        rank_loss = stage_u.loss_for_output(output, labels, batch.pixels, model)
        if shared_decoder is None:
            aux_loss = rank_loss.detach() * 0.0
            loss = rank_loss
        else:
            aux_loss = supervised_stage_loss(output, shared_decoder, batch)
            loss = rank_loss + config.semantic_loss_weight * aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.train_steps:
            metrics = stage_u.evaluate_ranking(
                model,
                val.pixels,
                u_config,
                device=device,
                candidate_pool=candidate_pool,
                seed=seed + step,
            )
            if float(metrics["rank_top1"]) > best_top1:
                best_top1 = float(metrics["rank_top1"])
                best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if shared_decoder is not None:
                    best_decoder_state = {key: value.detach().cpu().clone() for key, value in shared_decoder.state_dict().items()}
            history.append(
                {
                    "step": step,
                    "loss": round(float(loss.detach().cpu()), 4),
                    "rank_loss": round(float(rank_loss.detach().cpu()), 4),
                    "semantic_loss": round(float(aux_loss.detach().cpu()), 4),
                    "rank_top1": metrics["rank_top1"],
                }
            )
            print(
                f"{mode} step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"rank={float(rank_loss.detach().cpu()):.4f} sem={float(aux_loss.detach().cpu()):.4f} "
                f"top1={metrics['rank_top1']:.3f}",
                flush=True,
            )
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if shared_decoder is not None and best_decoder_state is not None:
        shared_decoder.load_state_dict(best_decoder_state)
    return (
        model,
        shared_decoder,
        {
            "history": history,
            "best_val_rank_top1": best_top1,
            "training_seconds": round(time.perf_counter() - started, 3),
            "trainable_parameter_count": sum(parameter.numel() for parameter in params if parameter.requires_grad),
        },
    )


def train_source_probe(
    model: stage_u.StageUWideVLM,
    source_stage: str,
    train: SemanticPixelSet,
    test: SemanticPixelSet,
    config: StageVConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    decoder = GridSemanticDecoder(config).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.probe_steps):
        batch = random_semantic_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            tokens = model.representations(batch.pixels)[source_stage].detach()
        loss = semantic_loss(decoder(tokens), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    transfer = {
        target_stage: evaluate_semantic_decoder(model, decoder, target_stage, test, config, device=device)
        for target_stage in DIAGNOSTIC_STAGES
    }
    return {
        "source_stage": source_stage,
        "training_seconds": round(time.perf_counter() - started, 3),
        "transfer": transfer,
    }


def run_diagnostics(
    model: stage_u.StageUWideVLM,
    shared_decoder: GridSemanticDecoder | None,
    train: SemanticPixelSet,
    test: SemanticPixelSet,
    config: StageVConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    source_probes = {
        source_stage: train_source_probe(
            model,
            source_stage,
            train,
            test,
            config,
            seed=seed + 8000 + index * 101,
            device=device,
        )
        for index, source_stage in enumerate(DIAGNOSTIC_STAGES)
    }
    diagonal = {
        stage_name: source_probes[stage_name]["transfer"][stage_name]
        for stage_name in DIAGNOSTIC_STAGES
    }
    offdiag_values = [
        float(source_probes[source]["transfer"][target]["cell_info_accuracy"])
        for source in DIAGNOSTIC_STAGES
        for target in DIAGNOSTIC_STAGES
        if source != target
    ]
    diag_values = [float(diagonal[stage]["cell_info_accuracy"]) for stage in DIAGNOSTIC_STAGES]
    diagnostics: dict[str, object] = {
        "source_probe_transfer": source_probes,
        "diagonal_information_recovery": diagonal,
        "cross_expert_summary": {
            "diagonal_cell_info_accuracy": statistics.fmean(diag_values),
            "diagonal_cell_info_loss": 1.0 - statistics.fmean(diag_values),
            "offdiag_cell_info_accuracy": statistics.fmean(offdiag_values),
            "offdiag_cell_info_loss": 1.0 - statistics.fmean(offdiag_values),
            "language_gap_cell_info_accuracy": statistics.fmean(diag_values) - statistics.fmean(offdiag_values),
        },
    }
    if shared_decoder is not None:
        diagnostics["trained_shared_decoder"] = {
            stage_name: evaluate_semantic_decoder(model, shared_decoder, stage_name, test, config, device=device)
            for stage_name in DIAGNOSTIC_STAGES
        }
    return diagnostics


def evaluate_model_bundle(
    model: stage_u.StageUWideVLM,
    test: SemanticPixelSet,
    config: StageVConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    u_config = stage_u_config(config)
    return {
        "full": stage_u.evaluate_ranking(model, test.pixels, u_config, device=device, candidate_pool=candidate_pool, seed=seed),
        "no_image_modality": stage_u.evaluate_ranking(
            model,
            test.pixels,
            u_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=seed + 1,
            zero_modalities=("image",),
        ),
        "no_patch_expert": stage_u.evaluate_ranking(
            model,
            test.pixels,
            u_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=seed + 2,
            zero_modalities=("patch_expert",),
        ),
        "no_object_experts": stage_u.evaluate_ranking(
            model,
            test.pixels,
            u_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=seed + 3,
            zero_modalities=("object_experts",),
        ),
    }


def run_experiment(config: StageVConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_v_expert_alignment device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    data, data_stats = build_semantic_data(config)
    train = data["train"].to(device)
    val = data["val"].to(device)
    test = data["test"].to(device)
    stage_u.stage_r.write_samples(data["test"].pixels, data["test"].meta, output_path.parent / "samples")
    candidate_pool = stage_u.build_candidate_pool(stage_u_config(config), device=device)

    models = {}
    shared_decoders = {}
    training = {}
    metrics = {}
    diagnostics = {}
    for index, mode in enumerate(config.modes):
        model, shared_decoder, train_stats = train_model(
            mode,
            train,
            val,
            config,
            candidate_pool=candidate_pool,
            seed=config.seed + 1000 + index * 151,
            device=device,
        )
        models[mode] = model
        shared_decoders[mode] = shared_decoder
        training[mode] = train_stats
        metrics[mode] = evaluate_model_bundle(
            model,
            test,
            config,
            candidate_pool=candidate_pool,
            device=device,
            seed=config.seed + 3000 + index * 17,
        )
        diagnostics[mode] = run_diagnostics(
            model,
            shared_decoder,
            train,
            test,
            config,
            seed=config.seed + 5000 + index * 313,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "experiment": "omni_transformer_stage_v_expert_alignment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "base": "Stage U object_slot_spatial_wide_latent with 64 patch tokens and preserved raw/weighted latent bus",
            "ranking_only_object_latent": "trains only candidate ranking and router loss; probes test whether experts independently learned a shared language",
            "supervised_shared_semantic_latent": "adds a shared grid semantic decoder over patch/object/spatial/count/fusion/wide tokens during training",
            "semantic_target": "16 grid cells with occupancy, color and shape labels from synthetic scene metadata",
            "language_test": "train a semantic decoder on one expert and evaluate it on other experts without retuning",
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "data": data_stats,
        "metrics": metrics,
        "training": training,
        "diagnostics": diagnostics,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    return output


def collect_numbers(value: object, prefix: tuple[str, ...] = ()) -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {".".join(prefix): float(value)}
    if isinstance(value, dict):
        collected: dict[str, float] = {}
        for key, child in value.items():
            collected.update(collect_numbers(child, (*prefix, str(key))))
        return collected
    return {}


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    buckets: dict[str, list[float]] = {}
    for run in runs:
        for key, value in collect_numbers(
            {
                "metrics": run["metrics"],
                "training": run["training"],
                "diagnostics": run["diagnostics"],
                "total_seconds": run["total_seconds"],
            }
        ).items():
            buckets.setdefault(key, []).append(value)
    stats = {key: stage_m.stat(values) for key, values in sorted(buckets.items())}

    def mean(path: str) -> float | None:
        item = stats.get(path)
        return None if item is None else float(item["mean"])

    summary = {}
    for mode in MODEL_MODES:
        summary[mode] = {
            "rank_top1": mean(f"metrics.{mode}.full.rank_top1"),
            "no_image_top1": mean(f"metrics.{mode}.no_image_modality.rank_top1"),
            "no_patch_top1": mean(f"metrics.{mode}.no_patch_expert.rank_top1"),
            "no_object_top1": mean(f"metrics.{mode}.no_object_experts.rank_top1"),
            "train_seconds": mean(f"training.{mode}.training_seconds"),
            "params": mean(f"training.{mode}.trainable_parameter_count"),
            "diagonal_cell_info_accuracy": mean(f"diagnostics.{mode}.cross_expert_summary.diagonal_cell_info_accuracy"),
            "offdiag_cell_info_accuracy": mean(f"diagnostics.{mode}.cross_expert_summary.offdiag_cell_info_accuracy"),
            "language_gap_cell_info_accuracy": mean(f"diagnostics.{mode}.cross_expert_summary.language_gap_cell_info_accuracy"),
            "stage_cell_info_accuracy": {
                stage_name: mean(f"diagnostics.{mode}.diagonal_information_recovery.{stage_name}.cell_info_accuracy")
                for stage_name in DIAGNOSTIC_STAGES
            },
            "stage_scene_exact": {
                stage_name: mean(f"diagnostics.{mode}.diagonal_information_recovery.{stage_name}.scene_exact")
                for stage_name in DIAGNOSTIC_STAGES
            },
        }
        if mode == "supervised_shared_semantic_latent":
            summary[mode]["trained_shared_decoder_cell_info_accuracy"] = {
                stage_name: mean(f"diagnostics.{mode}.trained_shared_decoder.{stage_name}.cell_info_accuracy")
                for stage_name in DIAGNOSTIC_STAGES
            }
    return {
        "experiment": "omni_transformer_stage_v_expert_alignment_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": stats,
        "summary": summary,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_v_expert_alignment/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_v_expert_alignment/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_v_expert_alignment/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702")
    parser.add_argument("--modes", default="ranking_only_object_latent,supervised_shared_semantic_latent")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=112)
    parser.add_argument("--answer-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patch-grid", type=int, default=8)
    parser.add_argument("--object-slots", type=int, default=8)
    parser.add_argument("--spatial-tokens", type=int, default=8)
    parser.add_argument("--count-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--fusion-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=420)
    parser.add_argument("--eval-every", type=int, default=140)
    parser.add_argument("--semantic-loss-weight", type=float, default=0.35)
    parser.add_argument("--probe-steps", type=int, default=70)
    args = parser.parse_args()

    modes = parse_csv_strings(args.modes)
    unknown = set(modes) - set(MODEL_MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    base_config = StageVConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        image_size=args.image_size,
        grid_size=args.grid_size,
        prompt_len=args.prompt_len,
        answer_len=args.answer_len,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        patch_grid=args.patch_grid,
        object_slots=args.object_slots,
        spatial_tokens=args.spatial_tokens,
        count_tokens=args.count_tokens,
        prompt_tokens=args.prompt_tokens,
        fusion_tokens=args.fusion_tokens,
        output_tokens=args.output_tokens,
        candidate_count=args.candidate_count,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        semantic_loss_weight=args.semantic_loss_weight,
        probe_steps=args.probe_steps,
        modes=modes,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageVConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
