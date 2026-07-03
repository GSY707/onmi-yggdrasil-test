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
import omni_transformer_stage_v_expert_alignment as stage_v
import omni_transformer_stage_w_alignment_mechanisms as stage_w
import omni_transformer_stage_y_parallel_input_experts as stage_y


MODES = ("baseline_parallel", "supervised_no_teacher", "supervised_teacher")
DIAGNOSTIC_STAGES = ("patch_tokens", "object_slots", "spatial_tokens", "count_tokens", "reasoning_tokens", "wide_latent")


@dataclass(frozen=True)
class StageZConfig:
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
    output_tokens: int = 8
    reasoning_layers: int = 1
    candidate_count: int = 8
    train_steps: int = 320
    eval_every: int = 160
    router_loss_weight: float = 0.05
    object_loss_weight: float = 0.35
    spatial_loss_weight: float = 0.35
    count_loss_weight: float = 0.30
    teacher_loss_weight: float = 0.30
    distill_loss_weight: float = 0.10
    semantic_probe_steps: int = 55
    count_probe_steps: int = 55
    answer_probe_steps: int = 55
    probe_hidden: int = 192
    modes: tuple[str, ...] = MODES


def stage_y_config(config: StageZConfig) -> stage_y.StageYConfig:
    return stage_y.StageYConfig(
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
        output_tokens=config.output_tokens,
        reasoning_layers=config.reasoning_layers,
        candidate_count=config.candidate_count,
        train_steps=config.train_steps,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
        probe_steps=config.answer_probe_steps,
        probe_hidden=config.probe_hidden,
        variants=("parallel_direct",),
    )


def stage_v_config(config: StageZConfig) -> stage_v.StageVConfig:
    return stage_v.StageVConfig(
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
        fusion_tokens=config.output_tokens,
        output_tokens=config.output_tokens,
        candidate_count=config.candidate_count,
        train_steps=config.train_steps,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
        semantic_loss_weight=config.spatial_loss_weight,
        probe_steps=config.semantic_probe_steps,
        probe_hidden=config.probe_hidden,
        modes=("ranking_only_object_latent",),
    )


def stage_w_config(config: StageZConfig) -> stage_w.StageWConfig:
    return stage_w.StageWConfig(
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
        fusion_tokens=config.output_tokens,
        output_tokens=config.output_tokens,
        candidate_count=config.candidate_count,
        train_steps=config.train_steps,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
        probe_steps=config.semantic_probe_steps,
        modes=("ranking_only",),
    )


def count_targets(meta: list[dict[str, object]], config: StageZConfig) -> torch.Tensor:
    pair_count = len(stage_u.COLORS) * len(stage_u.SHAPES)
    target = torch.zeros(len(meta), pair_count, dtype=torch.long)
    for batch_index, item in enumerate(meta):
        for obj in item["objects"]:
            pair_index = int(obj["color"]) * len(stage_u.SHAPES) + int(obj["shape"])
            target[batch_index, pair_index] += 1
    return target.clamp_max(config.grid_size * config.grid_size)


@dataclass
class StageZSemanticSet:
    semantic: stage_v.SemanticPixelSet
    count: torch.Tensor

    def subset(self, indices: list[int]) -> "StageZSemanticSet":
        return StageZSemanticSet(
            semantic=self.semantic.subset(indices),
            count=self.count[indices],
        )

    def to(self, device: torch.device) -> "StageZSemanticSet":
        return StageZSemanticSet(
            semantic=self.semantic.to(device),
            count=self.count.to(device=device, dtype=torch.long),
        )

    @property
    def pixels(self) -> stage_q.PixelSet:
        return self.semantic.pixels

    @property
    def meta(self) -> list[dict[str, object]]:
        return self.semantic.meta


def build_data(config: StageZConfig) -> tuple[dict[str, StageZSemanticSet], dict[str, object]]:
    semantic, stats = stage_v.build_semantic_data(stage_v_config(config))
    data = {
        split_name: StageZSemanticSet(
            semantic=semantic[split_name],
            count=count_targets(semantic[split_name].meta, config),
        )
        for split_name in ("train", "val", "test")
    }
    stats["count_targets"] = {
        "color_shape_pairs": len(stage_u.COLORS) * len(stage_u.SHAPES),
        "count_classes": config.grid_size * config.grid_size + 1,
    }
    return data, stats


def random_batch(data: StageZSemanticSet, *, rng: random.Random, batch_size: int, device: torch.device) -> StageZSemanticSet:
    return data.subset([rng.randrange(len(data.pixels.examples)) for _ in range(batch_size)]).to(device)


def tokens_for_stage(reps: dict[str, torch.Tensor], stage_name: str) -> torch.Tensor:
    if stage_name in reps:
        return reps[stage_name]
    raise KeyError(stage_name)


class CountTableDecoder(nn.Module):
    def __init__(self, config: StageZConfig) -> None:
        super().__init__()
        self.pair_count = len(stage_u.COLORS) * len(stage_u.SHAPES)
        self.query = nn.Parameter(torch.randn(self.pair_count, config.d_model) * 0.02)
        self.cross = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.out = nn.Linear(config.d_model, config.grid_size * config.grid_size + 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.cross(self.norm(query), self.norm(tokens), self.norm(tokens), need_weights=False)
        hidden = query + attended
        hidden = hidden + self.ff(hidden)
        return self.out(hidden)


def count_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view_as(targets)
    weights = torch.ones_like(loss)
    weights = weights.masked_fill(targets > 0, 5.0)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def count_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    per_pair = (pred == targets).float()
    present_true = targets > 0
    present_pred = pred > 0
    presence = (present_true == present_pred).float()
    if present_true.any():
        positive_pair_accuracy = (pred[present_true] == targets[present_true]).float().mean().item()
        positive_count_mae = (pred[present_true].float() - targets[present_true].float()).abs().mean().item()
        positive_presence_recall = present_pred[present_true].float().mean().item()
    else:
        positive_pair_accuracy = 1.0
        positive_count_mae = 0.0
        positive_presence_recall = 1.0
    return {
        "pair_accuracy": per_pair.mean().item(),
        "presence_accuracy": presence.mean().item(),
        "positive_pair_accuracy": positive_pair_accuracy,
        "positive_count_mae": positive_count_mae,
        "positive_presence_recall": positive_presence_recall,
        "table_exact": per_pair.all(dim=1).float().mean().item(),
        "pair_info_loss": 1.0 - per_pair.mean().item(),
        "positive_pair_info_loss": 1.0 - positive_pair_accuracy,
        "table_info_loss": 1.0 - per_pair.all(dim=1).float().mean().item(),
    }


class CellTokenDecoder(nn.Module):
    def __init__(self, config: StageZConfig) -> None:
        super().__init__()
        self.occupancy = nn.Linear(config.d_model, 1)
        self.color = nn.Linear(config.d_model, len(stage_u.COLORS) + 1)
        self.shape = nn.Linear(config.d_model, len(stage_u.SHAPES) + 1)

    def forward(self, cell_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "occupancy": self.occupancy(cell_tokens).squeeze(-1),
            "color": self.color(cell_tokens),
            "shape": self.shape(cell_tokens),
        }


class TeacherSemanticBus(nn.Module):
    def __init__(self, config: StageZConfig) -> None:
        super().__init__()
        self.resampler = stage_q.QueryResampler(
            config.d_model,
            config.heads,
            config.grid_size * config.grid_size,
            config.dropout,
        )
        self.type_embedding = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.resampler(patch_tokens) + self.type_embedding


class DistillProjector(nn.Module):
    def __init__(self, config: StageZConfig, source_tokens: int) -> None:
        super().__init__()
        self.resampler = stage_q.QueryResampler(config.d_model, config.heads, config.grid_size * config.grid_size, config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
        self.source_tokens = source_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.resampler(tokens))


class SupervisionHeads(nn.Module):
    def __init__(self, config: StageZConfig, *, use_teacher: bool) -> None:
        super().__init__()
        self.config = config
        self.use_teacher = use_teacher
        self.object_decoder = stage_w.SlotObjectDecoder(stage_w_config(config))
        self.spatial_decoder = stage_v.GridSemanticDecoder(stage_v_config(config))
        self.count_decoder = CountTableDecoder(config)
        if use_teacher:
            self.teacher_bus = TeacherSemanticBus(config)
            self.teacher_decoder = CellTokenDecoder(config)
            self.object_distill = DistillProjector(config, config.object_slots)
            self.spatial_distill = DistillProjector(config, config.spatial_tokens)
            self.count_distill = DistillProjector(config, config.count_tokens)
        else:
            self.teacher_bus = None
            self.teacher_decoder = None
            self.object_distill = None
            self.spatial_distill = None
            self.count_distill = None

    def explicit_loss(self, reps: dict[str, torch.Tensor], batch: StageZSemanticSet) -> tuple[torch.Tensor, dict[str, float]]:
        object_loss = stage_w.slot_supervision_loss(self.object_decoder, reps["object_slots"], batch.semantic, stage_w_config(self.config))
        spatial_loss = stage_v.semantic_loss(self.spatial_decoder(reps["spatial_tokens"]), batch.semantic)
        count_logits = self.count_decoder(reps["count_tokens"])
        count_table_loss = count_loss(count_logits, batch.count)
        loss = (
            self.config.object_loss_weight * object_loss
            + self.config.spatial_loss_weight * spatial_loss
            + self.config.count_loss_weight * count_table_loss
        )
        metrics = {
            "object_loss": float(object_loss.detach().cpu()),
            "spatial_loss": float(spatial_loss.detach().cpu()),
            "count_loss": float(count_table_loss.detach().cpu()),
        }
        if not self.use_teacher:
            return loss, metrics
        assert self.teacher_bus is not None
        assert self.teacher_decoder is not None
        assert self.object_distill is not None
        assert self.spatial_distill is not None
        assert self.count_distill is not None
        teacher_tokens = self.teacher_bus(reps["patch_tokens"])
        teacher_loss = stage_v.semantic_loss(self.teacher_decoder(teacher_tokens), batch.semantic)
        teacher_target = F.layer_norm(teacher_tokens.detach(), (teacher_tokens.shape[-1],))
        distill_losses = [
            F.mse_loss(self.object_distill(reps["object_slots"]), teacher_target),
            F.mse_loss(self.spatial_distill(reps["spatial_tokens"]), teacher_target),
            F.mse_loss(self.count_distill(reps["count_tokens"]), teacher_target),
        ]
        distill_loss = torch.stack(distill_losses).mean()
        loss = loss + self.config.teacher_loss_weight * teacher_loss + self.config.distill_loss_weight * distill_loss
        metrics.update({
            "teacher_loss": float(teacher_loss.detach().cpu()),
            "distill_loss": float(distill_loss.detach().cpu()),
        })
        return loss, metrics


def ranking_loss_for_batch(
    model: nn.Module,
    batch: StageZSemanticSet,
    config: StageZConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    rng: random.Random,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, float]]:
    rows, true_indices = stage_u.candidate_rows_for_examples(batch.pixels.examples, candidate_pool, seed=rng.randrange(1_000_000_000))
    answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
    labels = torch.tensor(true_indices, dtype=torch.long, device=device)
    output = model(batch.pixels, answers)
    rank = F.cross_entropy(output["scores"], labels)
    route_target = model.route_target(batch.pixels.required_experts)  # type: ignore[attr-defined]
    router = F.binary_cross_entropy_with_logits(output["route_logits"], route_target)
    return output, rank + config.router_loss_weight * router, {
        "rank_loss": float(rank.detach().cpu()),
        "router_loss": float(router.detach().cpu()),
    }


def train_model(
    mode: str,
    model: stage_y.ParallelInputExpertVLM,
    train: StageZSemanticSet,
    val: StageZSemanticSet,
    config: StageZConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> tuple[SupervisionHeads | None, dict[str, object]]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    use_supervision = mode in {"supervised_no_teacher", "supervised_teacher"}
    heads = SupervisionHeads(config, use_teacher=mode == "supervised_teacher").to(device) if use_supervision else None
    parameters = list(model.parameters()) + ([] if heads is None else list(heads.parameters()))
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    model.to(device)
    model.train()
    if heads is not None:
        heads.train()
    best_top1 = -1.0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_head_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.train_steps + 1):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        output, loss, loss_metrics = ranking_loss_for_batch(model, batch, config, candidate_pool=candidate_pool, rng=rng, device=device)
        if heads is not None:
            explicit_loss, explicit_metrics = heads.explicit_loss(output, batch)
            loss = loss + explicit_loss
            loss_metrics.update(explicit_metrics)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.train_steps:
            metrics = stage_y.evaluate_ranking(
                model,
                val.pixels,
                stage_y_config(config),
                device=device,
                candidate_pool=candidate_pool,
                seed=seed + step,
            )
            if float(metrics["rank_top1"]) > best_top1:
                best_top1 = float(metrics["rank_top1"])
                best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if heads is not None:
                    best_head_state = {key: value.detach().cpu().clone() for key, value in heads.state_dict().items()}
            row = {
                "step": step,
                "loss": round(float(loss.detach().cpu()), 4),
                "rank_top1": metrics["rank_top1"],
                "rank_mrr": metrics["rank_mrr"],
            }
            row.update({key: round(value, 4) for key, value in loss_metrics.items()})
            history.append(row)
            print(f"{mode} step={step:4d} loss={float(loss.detach().cpu()):.4f} top1={metrics['rank_top1']:.3f}", flush=True)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if heads is not None and best_head_state is not None:
        heads.load_state_dict(best_head_state)
    return heads, {
        "history": history,
        "best_val_rank_top1": best_top1,
        "training_seconds": round(time.perf_counter() - started, 3),
        "runtime_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "train_only_parameter_count": 0 if heads is None else sum(parameter.numel() for parameter in heads.parameters() if parameter.requires_grad),
    }


@torch.no_grad()
def evaluate_training_heads(
    model: stage_y.ParallelInputExpertVLM,
    heads: SupervisionHeads | None,
    test: StageZSemanticSet,
    config: StageZConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    if heads is None:
        return {}
    model.eval()
    heads.eval()
    object_rows = []
    spatial_rows = []
    count_rows = []
    teacher_rows = []
    for start in range(0, len(test.pixels.examples), config.batch_size):
        batch = test.subset(list(range(start, min(start + config.batch_size, len(test.pixels.examples))))).to(device)
        reps = model.representations(batch.pixels)
        object_rows.append(stage_w.slot_metrics(heads.object_decoder, model, batch.semantic, stage_w_config(config), device=device))
        spatial_rows.append(stage_v.semantic_metrics(heads.spatial_decoder(reps["spatial_tokens"]), batch.semantic))
        count_rows.append(count_metrics(heads.count_decoder(reps["count_tokens"]), batch.count))
        if heads.teacher_bus is not None and heads.teacher_decoder is not None:
            teacher_rows.append(stage_v.semantic_metrics(heads.teacher_decoder(heads.teacher_bus(reps["patch_tokens"])), batch.semantic))
    output = {
        "object_decoder": stage_v.merge_metric_rows(object_rows),
        "spatial_decoder": stage_v.merge_metric_rows(spatial_rows),
        "count_decoder": stage_v.merge_metric_rows(count_rows),
    }
    if teacher_rows:
        output["teacher_decoder"] = stage_v.merge_metric_rows(teacher_rows)
    return output


class AnswerProbe(nn.Module):
    def __init__(self, config: StageZConfig, class_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.probe_hidden),
            nn.GELU(),
            nn.Linear(config.probe_hidden, class_count),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens.mean(dim=1))


def answer_labels(data: StageZSemanticSet, label_by_answer: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([label_by_answer[example.answer] for example in data.pixels.examples], dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_answer_probe(
    model: stage_y.ParallelInputExpertVLM,
    probe: AnswerProbe,
    stage_name: str,
    test: StageZSemanticSet,
    config: StageZConfig,
    *,
    label_by_answer: dict[str, int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    probe.eval()
    hits = []
    for start in range(0, len(test.pixels.examples), config.batch_size):
        batch = test.subset(list(range(start, min(start + config.batch_size, len(test.pixels.examples))))).to(device)
        labels = answer_labels(batch, label_by_answer, device)
        pred = probe(tokens_for_stage(model.representations(batch.pixels), stage_name)).argmax(dim=1)
        hits.extend((pred == labels).float().detach().cpu().tolist())
    exact = statistics.fmean(hits)
    return {
        "answer_exact": exact,
        "answer_info_loss": 1.0 - exact,
        "random_exact": 1.0 / len(label_by_answer),
    }


def train_answer_probe(
    model: stage_y.ParallelInputExpertVLM,
    stage_name: str,
    train: StageZSemanticSet,
    test: StageZSemanticSet,
    config: StageZConfig,
    *,
    label_by_answer: dict[str, int],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = AnswerProbe(config, len(label_by_answer)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.answer_probe_steps):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        labels = answer_labels(batch, label_by_answer, device)
        with torch.no_grad():
            tokens = tokens_for_stage(model.representations(batch.pixels), stage_name).detach()
        loss = F.cross_entropy(probe(tokens), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    metrics = evaluate_answer_probe(model, probe, stage_name, test, config, label_by_answer=label_by_answer, device=device)
    metrics.update({
        "reader_parameter_count": sum(parameter.numel() for parameter in probe.parameters() if parameter.requires_grad),
        "training_seconds": round(time.perf_counter() - started, 3),
    })
    return metrics


def train_semantic_reader(
    model: stage_y.ParallelInputExpertVLM,
    source_stage: str,
    train: StageZSemanticSet,
    test: StageZSemanticSet,
    config: StageZConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    reader = stage_v.GridSemanticDecoder(stage_v_config(config)).to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.semantic_probe_steps):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            tokens = tokens_for_stage(model.representations(batch.pixels), source_stage).detach()
        loss = stage_v.semantic_loss(reader(tokens), batch.semantic)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    transfer = {}
    with torch.no_grad():
        reader.eval()
        for target_stage in DIAGNOSTIC_STAGES:
            rows = []
            for start in range(0, len(test.pixels.examples), config.batch_size):
                batch = test.subset(list(range(start, min(start + config.batch_size, len(test.pixels.examples))))).to(device)
                reps = model.representations(batch.pixels)
                rows.append(stage_v.semantic_metrics(reader(tokens_for_stage(reps, target_stage)), batch.semantic))
            transfer[target_stage] = stage_v.merge_metric_rows(rows)
    return {
        "source_stage": source_stage,
        "reader_parameter_count": sum(parameter.numel() for parameter in reader.parameters() if parameter.requires_grad),
        "training_seconds": round(time.perf_counter() - started, 3),
        "transfer": transfer,
    }


def train_count_reader(
    model: stage_y.ParallelInputExpertVLM,
    source_stage: str,
    train: StageZSemanticSet,
    test: StageZSemanticSet,
    config: StageZConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    reader = CountTableDecoder(config).to(device)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.count_probe_steps):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            tokens = tokens_for_stage(model.representations(batch.pixels), source_stage).detach()
        loss = count_loss(reader(tokens), batch.count)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    transfer = {}
    with torch.no_grad():
        reader.eval()
        for target_stage in DIAGNOSTIC_STAGES:
            rows = []
            for start in range(0, len(test.pixels.examples), config.batch_size):
                batch = test.subset(list(range(start, min(start + config.batch_size, len(test.pixels.examples))))).to(device)
                reps = model.representations(batch.pixels)
                rows.append(count_metrics(reader(tokens_for_stage(reps, target_stage)), batch.count))
            transfer[target_stage] = stage_v.merge_metric_rows(rows)
    return {
        "source_stage": source_stage,
        "reader_parameter_count": sum(parameter.numel() for parameter in reader.parameters() if parameter.requires_grad),
        "training_seconds": round(time.perf_counter() - started, 3),
        "transfer": transfer,
    }


def transfer_summary(probes: dict[str, dict[str, object]], metric_name: str) -> dict[str, object]:
    diag = [float(probes[stage]["transfer"][stage][metric_name]) for stage in DIAGNOSTIC_STAGES]
    offdiag = [
        float(probes[source]["transfer"][target][metric_name])
        for source in DIAGNOSTIC_STAGES
        for target in DIAGNOSTIC_STAGES
        if source != target
    ]
    return {
        "diagonal_mean": statistics.fmean(diag),
        "offdiag_mean": statistics.fmean(offdiag),
        "language_gap": statistics.fmean(diag) - statistics.fmean(offdiag),
        "diagonal_by_stage": {
            stage: float(probes[stage]["transfer"][stage][metric_name])
            for stage in DIAGNOSTIC_STAGES
        },
    }


def diagnostic_suite(
    model: stage_y.ParallelInputExpertVLM,
    train: StageZSemanticSet,
    test: StageZSemanticSet,
    config: StageZConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    label_by_answer = {answer: index for index, answer in enumerate(candidate_pool.answers)}
    semantic_transfer = {
        stage: train_semantic_reader(
            model,
            stage,
            train,
            test,
            config,
            seed=seed + 10_000 + index * 211,
            device=device,
        )
        for index, stage in enumerate(DIAGNOSTIC_STAGES)
    }
    count_transfer = {
        stage: train_count_reader(
            model,
            stage,
            train,
            test,
            config,
            seed=seed + 20_000 + index * 223,
            device=device,
        )
        for index, stage in enumerate(DIAGNOSTIC_STAGES)
    }
    answer_reconstruction = {
        stage: train_answer_probe(
            model,
            stage,
            train,
            test,
            config,
            label_by_answer=label_by_answer,
            seed=seed + 30_000 + index * 227,
            device=device,
        )
        for index, stage in enumerate(DIAGNOSTIC_STAGES)
    }
    return {
        "semantic_transfer": semantic_transfer,
        "semantic_transfer_summary": transfer_summary(semantic_transfer, "cell_info_accuracy"),
        "count_transfer": count_transfer,
        "count_transfer_summary": transfer_summary(count_transfer, "pair_accuracy"),
        "answer_reconstruction": answer_reconstruction,
        "answer_reconstruction_summary": {
            "best_stage": max(answer_reconstruction, key=lambda stage: float(answer_reconstruction[stage]["answer_exact"])),
            "best_answer_exact": max(float(item["answer_exact"]) for item in answer_reconstruction.values()),
            "by_stage": {
                stage: float(item["answer_exact"])
                for stage, item in answer_reconstruction.items()
            },
        },
    }


def run_experiment(config: StageZConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_z_supervised_teacher_diagnostics device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    data, data_stats = build_data(config)
    train = data["train"].to(device)
    val = data["val"].to(device)
    test = data["test"].to(device)
    stage_u.stage_r.write_samples(data["test"].pixels, data["test"].meta, output_path.parent / "samples")
    candidate_pool = stage_u.build_candidate_pool(stage_y.stage_u_config(stage_y_config(config)), device=device)

    models = {
        mode: stage_y.ParallelInputExpertVLM(stage_y_config(config), variant="parallel_direct").to(device)
        for mode in config.modes
    }
    heads_by_mode: dict[str, SupervisionHeads | None] = {}
    training = {}
    for index, (mode, model) in enumerate(models.items()):
        heads, training[mode] = train_model(
            mode,
            model,
            train,
            val,
            config,
            candidate_pool=candidate_pool,
            seed=config.seed + 1100 + index * 97,
            device=device,
        )
        heads_by_mode[mode] = heads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = {
        mode: stage_y.evaluate_ranking(
            model,
            test.pixels,
            stage_y_config(config),
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 3000 + index,
        )
        for index, (mode, model) in enumerate(models.items())
    }
    ablations = {
        f"{mode}_{ablation_name}": stage_y.evaluate_ranking(
            model,
            test.pixels,
            stage_y_config(config),
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4000 + index * 97,
            zero_modalities=zero_modalities,
            disable_latent_access=disable_latent,
        )
        for index, (mode, model) in enumerate(models.items())
        for ablation_name, zero_modalities, disable_latent in (
            ("no_image_modality", ("image",), False),
            ("no_patch_expert", ("patch_expert",), False),
            ("no_object_spatial_count_experts", ("object_experts",), False),
            ("no_latent_access", (), True),
        )
    }
    trained_head_metrics = {
        mode: evaluate_training_heads(model, heads_by_mode[mode], test, config, device=device)
        for mode, model in models.items()
    }
    diagnostics = {
        mode: diagnostic_suite(
            model,
            train,
            test,
            config,
            candidate_pool=candidate_pool,
            seed=config.seed + 5000 + index * 1000,
            device=device,
        )
        for index, (mode, model) in enumerate(models.items())
    }
    prediction_cost = {
        f"{mode}_prediction": stage_q.measure_prediction_cost(
            lambda model=model: stage_y.evaluate_ranking(
                model,
                test.pixels,
                stage_y_config(config),
                device=device,
                candidate_pool=candidate_pool,
                seed=config.seed + 8000,
            ),
            example_count=len(test.pixels.examples),
            device=device,
        )
        for mode, model in models.items()
    }
    output = {
        "experiment": "omni_transformer_stage_z_supervised_teacher_diagnostics",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "base": "Stage Y parallel_direct topology: all input experts read external image/text directly, then latent reasoner aggregates.",
            "explicit_supervision": "object slot set targets, spatial semantic grid targets, and color-shape count table targets are train-only losses.",
            "teacher": "train-only patch-derived semantic bus with grid supervision; object/spatial/count tokens are distilled toward teacher cell tokens but runtime output still uses direct latent tokens.",
            "diagnostic_stages": DIAGNOSTIC_STAGES,
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "data": data_stats,
        "metrics": metrics,
        "ablations": ablations,
        "trained_head_metrics": trained_head_metrics,
        "diagnostics": diagnostics,
        "training": training,
        "prediction_cost": prediction_cost,
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
                "ablations": run["ablations"],
                "trained_head_metrics": run["trained_head_metrics"],
                "diagnostics": run["diagnostics"],
                "training": run["training"],
                "prediction_cost": run["prediction_cost"],
                "total_seconds": run["total_seconds"],
            }
        ).items():
            buckets.setdefault(key, []).append(value)
    stats = {key: stage_m.stat(values) for key, values in sorted(buckets.items())}

    def mean(path: str) -> float | None:
        item = stats.get(path)
        return None if item is None else float(item["mean"])

    def diag_transfer_mean(mode: str, suite: str, metric_name: str) -> float | None:
        values = [
            mean(f"diagnostics.{mode}.{suite}.{stage}.transfer.{stage}.{metric_name}")
            for stage in DIAGNOSTIC_STAGES
        ]
        clean = [value for value in values if value is not None]
        return None if not clean else statistics.fmean(clean)

    def offdiag_transfer_mean(mode: str, suite: str, metric_name: str) -> float | None:
        values = [
            mean(f"diagnostics.{mode}.{suite}.{source}.transfer.{target}.{metric_name}")
            for source in DIAGNOSTIC_STAGES
            for target in DIAGNOSTIC_STAGES
            if source != target
        ]
        clean = [value for value in values if value is not None]
        return None if not clean else statistics.fmean(clean)

    modes = sorted({mode for run in runs for mode in run["metrics"].keys()})
    summary: dict[str, object] = {}
    for mode in modes:
        summary[mode] = {
            "rank_top1": mean(f"metrics.{mode}.rank_top1"),
            "no_image_top1": mean(f"ablations.{mode}_no_image_modality.rank_top1"),
            "no_patch_expert_top1": mean(f"ablations.{mode}_no_patch_expert.rank_top1"),
            "no_object_spatial_count_top1": mean(f"ablations.{mode}_no_object_spatial_count_experts.rank_top1"),
            "semantic_diag_cell_info": mean(f"diagnostics.{mode}.semantic_transfer_summary.diagonal_mean"),
            "semantic_offdiag_cell_info": mean(f"diagnostics.{mode}.semantic_transfer_summary.offdiag_mean"),
            "semantic_language_gap": mean(f"diagnostics.{mode}.semantic_transfer_summary.language_gap"),
            "semantic_diag_occupied_color": diag_transfer_mean(mode, "semantic_transfer", "occupied_color_accuracy"),
            "semantic_diag_occupied_shape": diag_transfer_mean(mode, "semantic_transfer", "occupied_shape_accuracy"),
            "semantic_diag_scene_exact": diag_transfer_mean(mode, "semantic_transfer", "scene_exact"),
            "count_diag_pair_accuracy": mean(f"diagnostics.{mode}.count_transfer_summary.diagonal_mean"),
            "count_offdiag_pair_accuracy": mean(f"diagnostics.{mode}.count_transfer_summary.offdiag_mean"),
            "count_language_gap": mean(f"diagnostics.{mode}.count_transfer_summary.language_gap"),
            "count_diag_positive_pair_accuracy": diag_transfer_mean(mode, "count_transfer", "positive_pair_accuracy"),
            "count_offdiag_positive_pair_accuracy": offdiag_transfer_mean(mode, "count_transfer", "positive_pair_accuracy"),
            "count_diag_presence_accuracy": diag_transfer_mean(mode, "count_transfer", "presence_accuracy"),
            "count_diag_positive_presence_recall": diag_transfer_mean(mode, "count_transfer", "positive_presence_recall"),
            "best_answer_reconstruction": mean(f"diagnostics.{mode}.answer_reconstruction_summary.best_answer_exact"),
            "runtime_params": mean(f"training.{mode}.runtime_parameter_count"),
            "train_only_params": mean(f"training.{mode}.train_only_parameter_count"),
            "train_seconds": mean(f"training.{mode}.training_seconds"),
            "prediction_ms_per_example": mean(f"prediction_cost.{mode}_prediction.prediction_ms_per_example"),
            "per_family": {
                family: mean(f"metrics.{mode}.per_family.{family}.rank_top1")
                for family in stage_u.FAMILIES
            },
            "semantic_diag_by_stage": {
                stage: mean(f"diagnostics.{mode}.semantic_transfer_summary.diagonal_by_stage.{stage}")
                for stage in DIAGNOSTIC_STAGES
            },
            "count_diag_by_stage": {
                stage: mean(f"diagnostics.{mode}.count_transfer_summary.diagonal_by_stage.{stage}")
                for stage in DIAGNOSTIC_STAGES
            },
            "answer_by_stage": {
                stage: mean(f"diagnostics.{mode}.answer_reconstruction_summary.by_stage.{stage}")
                for stage in DIAGNOSTIC_STAGES
            },
        }
    return {
        "experiment": "omni_transformer_stage_z_supervised_teacher_diagnostics_sweep",
        "run_count": len(runs),
        "seeds": sorted({int(run["config"]["seed"]) for run in runs}),
        "modes": modes,
        "stats": stats,
        "summary": summary,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_z_supervised_teacher_diagnostics/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702")
    parser.add_argument("--modes", default="baseline_parallel,supervised_no_teacher,supervised_teacher")
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
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--reasoning-layers", type=int, default=1)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=320)
    parser.add_argument("--eval-every", type=int, default=160)
    parser.add_argument("--semantic-probe-steps", type=int, default=55)
    parser.add_argument("--count-probe-steps", type=int, default=55)
    parser.add_argument("--answer-probe-steps", type=int, default=55)
    args = parser.parse_args()

    modes = parse_csv_strings(args.modes)
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    base_config = StageZConfig(
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
        output_tokens=args.output_tokens,
        reasoning_layers=args.reasoning_layers,
        candidate_count=args.candidate_count,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        semantic_probe_steps=args.semantic_probe_steps,
        count_probe_steps=args.count_probe_steps,
        answer_probe_steps=args.answer_probe_steps,
        modes=modes,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageZConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
