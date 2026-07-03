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


MODES = (
    "baseline_parallel",
    "supervised_no_teacher",
    "token_aligned_teacher",
    "text_latent_aligned",
    "moe_text_latent_aligned",
)
DIAGNOSTIC_STAGES = ("patch_tokens", "object_slots", "spatial_tokens", "count_tokens", "reasoning_tokens", "wide_latent")
TOKEN_ALIGNMENT_STAGES = ("patch_tokens", "object_slots", "spatial_tokens", "count_tokens", "reasoning_tokens")
TEXT_ALIGNMENT_MODES = {"text_latent_aligned", "moe_text_latent_aligned"}


@dataclass(frozen=True)
class StageABConfig:
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
    token_teacher_align_weight: float = 0.08
    token_cross_align_weight: float = 0.04
    token_alignment_temperature: float = 0.07
    text_latent_tokens: int = 8
    text_align_temperature: float = 0.07
    prompt_answer_align_weight: float = 0.08
    latent_answer_align_weight: float = 0.12
    latent_answer_rank_weight: float = 0.20
    latent_answer_class_weight: float = 0.20
    latent_answer_score_weight: float = 0.35
    moe_experts: int = 4
    moe_balance_weight: float = 0.02
    semantic_probe_steps: int = 55
    count_probe_steps: int = 55
    answer_probe_steps: int = 55
    probe_hidden: int = 192
    modes: tuple[str, ...] = MODES


def stage_y_config(config: StageABConfig) -> stage_y.StageYConfig:
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


def stage_v_config(config: StageABConfig) -> stage_v.StageVConfig:
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


def stage_w_config(config: StageABConfig) -> stage_w.StageWConfig:
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


def count_targets(meta: list[dict[str, object]], config: StageABConfig) -> torch.Tensor:
    pair_count = len(stage_u.COLORS) * len(stage_u.SHAPES)
    target = torch.zeros(len(meta), pair_count, dtype=torch.long)
    for batch_index, item in enumerate(meta):
        for obj in item["objects"]:
            pair_index = int(obj["color"]) * len(stage_u.SHAPES) + int(obj["shape"])
            target[batch_index, pair_index] += 1
    return target.clamp_max(config.grid_size * config.grid_size)


@dataclass
class StageABSemanticSet:
    semantic: stage_v.SemanticPixelSet
    count: torch.Tensor

    def subset(self, indices: list[int]) -> "StageABSemanticSet":
        return StageABSemanticSet(
            semantic=self.semantic.subset(indices),
            count=self.count[indices],
        )

    def to(self, device: torch.device) -> "StageABSemanticSet":
        return StageABSemanticSet(
            semantic=self.semantic.to(device),
            count=self.count.to(device=device, dtype=torch.long),
        )

    @property
    def pixels(self) -> stage_q.PixelSet:
        return self.semantic.pixels

    @property
    def meta(self) -> list[dict[str, object]]:
        return self.semantic.meta


def build_data(config: StageABConfig) -> tuple[dict[str, StageABSemanticSet], dict[str, object]]:
    semantic, stats = stage_v.build_semantic_data(stage_v_config(config))
    data = {
        split_name: StageABSemanticSet(
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


def random_batch(data: StageABSemanticSet, *, rng: random.Random, batch_size: int, device: torch.device) -> StageABSemanticSet:
    return data.subset([rng.randrange(len(data.pixels.examples)) for _ in range(batch_size)]).to(device)


def tokens_for_stage(reps: dict[str, torch.Tensor], stage_name: str) -> torch.Tensor:
    if stage_name in reps:
        return reps[stage_name]
    raise KeyError(stage_name)


class CountTableDecoder(nn.Module):
    def __init__(self, config: StageABConfig) -> None:
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
    def __init__(self, config: StageABConfig) -> None:
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
    def __init__(self, config: StageABConfig) -> None:
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
    def __init__(self, config: StageABConfig, source_tokens: int) -> None:
        super().__init__()
        self.resampler = stage_q.QueryResampler(config.d_model, config.heads, config.grid_size * config.grid_size, config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
        self.source_tokens = source_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.resampler(tokens))


class TokenAlignmentProjector(nn.Module):
    def __init__(self, config: StageABConfig) -> None:
        super().__init__()
        cell_count = config.grid_size * config.grid_size
        self.resamplers = nn.ModuleDict({
            stage_name: stage_q.QueryResampler(config.d_model, config.heads, cell_count, config.dropout)
            for stage_name in TOKEN_ALIGNMENT_STAGES
        })
        self.projectors = nn.ModuleDict({
            stage_name: nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
            )
            for stage_name in (*TOKEN_ALIGNMENT_STAGES, "teacher_tokens")
        })

    def cell_tokens(self, stage_name: str, tokens: torch.Tensor) -> torch.Tensor:
        if stage_name == "teacher_tokens":
            return tokens
        return self.resamplers[stage_name](tokens)

    def encode(self, stage_name: str, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projectors[stage_name](self.cell_tokens(stage_name, tokens)), dim=-1)


def same_position_contrastive_loss(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    left_flat = left.reshape(left.shape[0] * left.shape[1], left.shape[2])
    right_flat = right.reshape(right.shape[0] * right.shape[1], right.shape[2])
    labels = torch.arange(left_flat.shape[0], device=left.device)
    logits = left_flat @ right_flat.T / temperature
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def aligned_cell_mse(source_cells: torch.Tensor, teacher_cells: torch.Tensor) -> torch.Tensor:
    source_norm = F.layer_norm(source_cells, (source_cells.shape[-1],))
    teacher_norm = F.layer_norm(teacher_cells.detach(), (teacher_cells.shape[-1],))
    return F.mse_loss(source_norm, teacher_norm)


@torch.no_grad()
def token_pair_metrics(left: torch.Tensor, right: torch.Tensor, temperature: float) -> dict[str, float]:
    left_flat = left.reshape(left.shape[0] * left.shape[1], left.shape[2])
    right_flat = right.reshape(right.shape[0] * right.shape[1], right.shape[2])
    labels = torch.arange(left_flat.shape[0], device=left.device)
    logits = left_flat @ right_flat.T / temperature
    return {
        "retrieval_top1": (logits.argmax(dim=1) == labels).float().mean().item(),
        "reverse_retrieval_top1": (logits.argmax(dim=0) == labels).float().mean().item(),
        "cosine": (left_flat * right_flat).sum(dim=1).mean().item(),
    }


@torch.no_grad()
def token_alignment_metrics(
    aligner: TokenAlignmentProjector,
    reps: dict[str, torch.Tensor],
    teacher_tokens: torch.Tensor,
    config: StageABConfig,
) -> dict[str, float]:
    teacher_encoded = aligner.encode("teacher_tokens", teacher_tokens)
    encoded = {
        stage_name: aligner.encode(stage_name, reps[stage_name])
        for stage_name in TOKEN_ALIGNMENT_STAGES
    }
    metrics: dict[str, float] = {}
    teacher_top1 = []
    teacher_cosine = []
    for stage_name, tokens in encoded.items():
        pair = token_pair_metrics(tokens, teacher_encoded, config.token_alignment_temperature)
        metrics[f"{stage_name}_teacher_retrieval_top1"] = pair["retrieval_top1"]
        metrics[f"{stage_name}_teacher_cosine"] = pair["cosine"]
        teacher_top1.append(pair["retrieval_top1"])
        teacher_cosine.append(pair["cosine"])
    cross_top1 = []
    cross_cosine = []
    stage_list = list(TOKEN_ALIGNMENT_STAGES)
    for left_index, left in enumerate(stage_list):
        for right in stage_list[left_index + 1 :]:
            pair = token_pair_metrics(encoded[left], encoded[right], config.token_alignment_temperature)
            metrics[f"{left}_to_{right}_retrieval_top1"] = pair["retrieval_top1"]
            metrics[f"{left}_to_{right}_cosine"] = pair["cosine"]
            cross_top1.append(pair["retrieval_top1"])
            cross_cosine.append(pair["cosine"])
    metrics["teacher_retrieval_top1_mean"] = statistics.fmean(teacher_top1)
    metrics["teacher_cosine_mean"] = statistics.fmean(teacher_cosine)
    metrics["cross_retrieval_top1_mean"] = statistics.fmean(cross_top1)
    metrics["cross_cosine_mean"] = statistics.fmean(cross_cosine)
    return metrics


class MoELatentReasoner(nn.Module):
    def __init__(self, config: StageABConfig) -> None:
        super().__init__()
        self.output_tokens = config.output_tokens
        self.query = nn.Parameter(torch.randn(config.output_tokens, config.d_model) * 0.02)
        self.experts = nn.ModuleList()
        for _ in range(config.moe_experts):
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.experts.append(nn.TransformerEncoder(layer, num_layers=config.reasoning_layers))
        self.gate = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.moe_experts),
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.last_gate: torch.Tensor | None = None

    def forward(self, raw: torch.Tensor, weighted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source = torch.cat((raw, weighted), dim=1)
        query = self.query.unsqueeze(0).expand(source.shape[0], -1, -1)
        gate_source = torch.cat((raw.mean(dim=1), weighted.mean(dim=1)), dim=-1)
        gate = F.softmax(self.gate(gate_source), dim=-1)
        encoded_source = torch.cat((source, query), dim=1)
        expert_outputs = []
        for expert in self.experts:
            encoded = expert(encoded_source)
            expert_outputs.append(self.norm(encoded[:, -self.output_tokens :]))
        stacked = torch.stack(expert_outputs, dim=1)
        reasoning = (stacked * gate.view(gate.shape[0], gate.shape[1], 1, 1)).sum(dim=1)
        self.last_gate = gate
        return torch.cat((source, reasoning), dim=1), reasoning


class TextLatentCodec(nn.Module):
    def __init__(self, config: StageABConfig) -> None:
        super().__init__()
        q_config = stage_y.stage_q_config(stage_y_config(config))
        self.prompt_resampler = stage_q.QueryResampler(config.d_model, config.heads, config.text_latent_tokens, config.dropout)
        self.answer_encoder = stage_q.TextTokenEncoder(q_config, length=config.answer_len)
        self.answer_resampler = stage_q.QueryResampler(config.d_model, config.heads, config.text_latent_tokens, config.dropout)
        self.prompt_projector = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model))
        self.answer_projector = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model))

    def prompt_latent(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        return self.prompt_projector(self.prompt_resampler(prompt_tokens))

    def answer_latent(self, answer_tokens: torch.Tensor) -> torch.Tensor:
        leading = answer_tokens.shape[:-1]
        flat = answer_tokens.reshape(-1, answer_tokens.shape[-1])
        hidden = self.answer_encoder(flat)
        latent = self.answer_projector(self.answer_resampler(hidden))
        return latent.reshape(*leading, latent.shape[-2], latent.shape[-1])


class LatentAnswerExpert(nn.Module):
    def __init__(self, config: StageABConfig, class_count: int) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.resampler = stage_q.QueryResampler(config.d_model, config.heads, config.text_latent_tokens, config.dropout)
        self.projector = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.probe_hidden),
            nn.GELU(),
            nn.Linear(config.probe_hidden, class_count),
        )

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(self.resampler(context_tokens))

    def class_logits(self, latent_answer_tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(latent_answer_tokens.mean(dim=1))

    def candidate_scores(self, latent_answer_tokens: torch.Tensor, answer_latents: torch.Tensor) -> torch.Tensor:
        latent = F.normalize(latent_answer_tokens.mean(dim=1), dim=-1)
        answer = F.normalize(answer_latents.mean(dim=-2), dim=-1)
        return torch.einsum("bd,bcd->bc", latent, answer) * self.temperature.exp().clamp(max=100.0)


class TextAlignedMoEVLM(nn.Module):
    def __init__(self, config: StageABConfig, *, use_moe: bool, answer_class_count: int) -> None:
        super().__init__()
        self.config = config
        self.use_moe = use_moe
        self.base = stage_y.ParallelInputExpertVLM(stage_y_config(config), variant="parallel_direct")
        if use_moe:
            self.base.reasoner = MoELatentReasoner(config)
        self.text_codec = TextLatentCodec(config)
        self.latent_answer = LatentAnswerExpert(config, answer_class_count)

    def route_target(self, required_experts: torch.Tensor) -> torch.Tensor:
        return self.base.route_target(required_experts)

    def representations(self, batch: stage_q.PixelSet, *, zero_modalities: Iterable[str] = ()) -> dict[str, torch.Tensor]:
        reps = self.base.representations(batch, zero_modalities=zero_modalities)
        reps["latent_answer_tokens"] = self.latent_answer(reps["wide_latent"])
        reasoner = self.base.reasoner
        if isinstance(reasoner, MoELatentReasoner) and reasoner.last_gate is not None:
            reps["moe_reasoner_gate"] = reasoner.last_gate
        return reps

    def context(
        self,
        batch: stage_q.PixelSet,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        reps = self.representations(batch, zero_modalities=zero_modalities)
        context = torch.zeros_like(reps["wide_latent"]) if disable_latent_access else reps["wide_latent"]
        if disable_latent_access:
            reps["latent_answer_tokens"] = torch.zeros_like(reps["latent_answer_tokens"])
        return {"context": context, **reps}

    def forward(
        self,
        batch: stage_q.PixelSet,
        answer_tokens: torch.Tensor,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(batch, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        cross_scores = self.base.scorer(ctx["context"], answer_tokens)
        answer_latents = self.text_codec.answer_latent(answer_tokens)
        latent_answer_scores = self.latent_answer.candidate_scores(ctx["latent_answer_tokens"], answer_latents)
        scores = cross_scores + self.config.latent_answer_score_weight * latent_answer_scores
        return {
            "scores": scores,
            "cross_scores": cross_scores,
            "latent_answer_scores": latent_answer_scores,
            **ctx,
        }


class SupervisionHeads(nn.Module):
    def __init__(self, config: StageABConfig, *, use_teacher: bool, use_token_alignment: bool) -> None:
        super().__init__()
        self.config = config
        self.use_teacher = use_teacher
        self.use_token_alignment = use_token_alignment
        self.object_decoder = stage_w.SlotObjectDecoder(stage_w_config(config))
        self.spatial_decoder = stage_v.GridSemanticDecoder(stage_v_config(config))
        self.count_decoder = CountTableDecoder(config)
        if use_teacher:
            self.teacher_bus = TeacherSemanticBus(config)
            self.teacher_decoder = CellTokenDecoder(config)
            self.object_distill = DistillProjector(config, config.object_slots)
            self.spatial_distill = DistillProjector(config, config.spatial_tokens)
            self.count_distill = DistillProjector(config, config.count_tokens)
            self.token_aligner = TokenAlignmentProjector(config) if use_token_alignment else None
        else:
            self.teacher_bus = None
            self.teacher_decoder = None
            self.object_distill = None
            self.spatial_distill = None
            self.count_distill = None
            self.token_aligner = None

    def explicit_loss(self, reps: dict[str, torch.Tensor], batch: StageABSemanticSet) -> tuple[torch.Tensor, dict[str, float]]:
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
        if self.token_aligner is not None:
            teacher_encoded = self.token_aligner.encode("teacher_tokens", teacher_tokens.detach())
            encoded = {
                stage_name: self.token_aligner.encode(stage_name, reps[stage_name])
                for stage_name in TOKEN_ALIGNMENT_STAGES
            }
            teacher_mse = torch.stack([
                aligned_cell_mse(self.token_aligner.cell_tokens(stage_name, reps[stage_name]), teacher_tokens)
                for stage_name in TOKEN_ALIGNMENT_STAGES
            ]).mean()
            teacher_contrastive = torch.stack([
                same_position_contrastive_loss(encoded[stage_name], teacher_encoded, self.config.token_alignment_temperature)
                for stage_name in TOKEN_ALIGNMENT_STAGES
            ]).mean()
            stage_list = list(TOKEN_ALIGNMENT_STAGES)
            cross_contrastive = torch.stack([
                same_position_contrastive_loss(encoded[left], encoded[right], self.config.token_alignment_temperature)
                for left_index, left in enumerate(stage_list)
                for right in stage_list[left_index + 1 :]
            ]).mean()
            token_teacher_align = teacher_mse + teacher_contrastive
            loss = (
                loss
                + self.config.token_teacher_align_weight * token_teacher_align
                + self.config.token_cross_align_weight * cross_contrastive
            )
            metrics.update({
                "token_teacher_mse": float(teacher_mse.detach().cpu()),
                "token_teacher_contrastive": float(teacher_contrastive.detach().cpu()),
                "token_cross_contrastive": float(cross_contrastive.detach().cpu()),
            })
        return loss, metrics


def pooled_pairwise_contrastive(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    left = F.normalize(z_a.mean(dim=1), dim=-1)
    right = F.normalize(z_b.mean(dim=1), dim=-1)
    labels = torch.arange(left.shape[0], device=left.device)
    logits = left @ right.T / temperature
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def pooled_retrieval_top1(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> float:
    left = F.normalize(z_a.mean(dim=1), dim=-1)
    right = F.normalize(z_b.mean(dim=1), dim=-1)
    labels = torch.arange(left.shape[0], device=left.device)
    logits = left @ right.T / temperature
    return (logits.argmax(dim=1) == labels).float().mean().item()


def moe_balance_loss(gate: torch.Tensor | None, config: StageABConfig) -> torch.Tensor:
    if gate is None:
        return torch.zeros((), device="cpu")
    target = torch.full_like(gate.mean(dim=0), 1.0 / gate.shape[-1])
    return F.mse_loss(gate.mean(dim=0), target)


def text_alignment_loss(
    model: nn.Module,
    output: dict[str, torch.Tensor],
    config: StageABConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not isinstance(model, TextAlignedMoEVLM):
        return torch.zeros((), device=output["scores"].device), {}
    true_answer_tokens = output["true_answer_tokens"]
    true_pool_indices = output["true_pool_indices"]
    labels = output["labels"]
    prompt_latent = model.text_codec.prompt_latent(output["prompt_tokens"])
    answer_latent = model.text_codec.answer_latent(true_answer_tokens)
    latent_answer = output["latent_answer_tokens"]
    prompt_answer = pooled_pairwise_contrastive(prompt_latent, answer_latent, config.text_align_temperature)
    latent_answer_contrast = pooled_pairwise_contrastive(latent_answer, answer_latent.detach(), config.text_align_temperature)
    latent_answer_mse = F.mse_loss(
        F.layer_norm(latent_answer, (latent_answer.shape[-1],)),
        F.layer_norm(answer_latent.detach(), (answer_latent.shape[-1],)),
    )
    latent_rank = F.cross_entropy(output["latent_answer_scores"], labels)
    class_logits = model.latent_answer.class_logits(latent_answer)
    class_loss = F.cross_entropy(class_logits, true_pool_indices)
    gate = output.get("moe_reasoner_gate")
    balance = moe_balance_loss(gate, config).to(output["scores"].device)
    loss = (
        config.prompt_answer_align_weight * prompt_answer
        + config.latent_answer_align_weight * (latent_answer_contrast + latent_answer_mse)
        + config.latent_answer_rank_weight * latent_rank
        + config.latent_answer_class_weight * class_loss
        + config.moe_balance_weight * balance
    )
    with torch.no_grad():
        class_acc = (class_logits.argmax(dim=1) == true_pool_indices).float().mean().item()
        latent_top1 = (output["latent_answer_scores"].argmax(dim=1) == labels).float().mean().item()
    return loss, {
        "prompt_answer_align": float(prompt_answer.detach().cpu()),
        "latent_answer_contrast": float(latent_answer_contrast.detach().cpu()),
        "latent_answer_mse": float(latent_answer_mse.detach().cpu()),
        "latent_answer_rank": float(latent_rank.detach().cpu()),
        "latent_answer_class": float(class_loss.detach().cpu()),
        "latent_answer_class_acc": class_acc,
        "latent_answer_candidate_top1": latent_top1,
        "moe_balance_loss": float(balance.detach().cpu()),
    }


@torch.no_grad()
def evaluate_text_alignment_model(
    model: nn.Module,
    test: StageABSemanticSet,
    config: StageABConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    if not isinstance(model, TextAlignedMoEVLM):
        return {}
    model.eval()
    rng = random.Random(seed)
    rows_out = []
    for start in range(0, len(test.pixels.examples), config.batch_size):
        batch = test.subset(list(range(start, min(start + config.batch_size, len(test.pixels.examples))))).to(device)
        rows, true_indices = stage_u.candidate_rows_for_examples(batch.pixels.examples, candidate_pool, seed=rng.randrange(1_000_000_000))
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        true_pool_indices = rows[torch.arange(rows.shape[0], device=device), labels]
        true_answer_tokens = answers[torch.arange(answers.shape[0], device=device), labels]
        output = model(batch.pixels, answers)
        prompt_latent = model.text_codec.prompt_latent(output["prompt_tokens"])
        answer_latent = model.text_codec.answer_latent(true_answer_tokens)
        latent_answer = output["latent_answer_tokens"]
        class_logits = model.latent_answer.class_logits(latent_answer)
        gate = output.get("moe_reasoner_gate")
        row = {
            "combined_candidate_top1": (output["scores"].argmax(dim=1) == labels).float().mean().item(),
            "latent_answer_candidate_top1": (output["latent_answer_scores"].argmax(dim=1) == labels).float().mean().item(),
            "latent_answer_class_accuracy": (class_logits.argmax(dim=1) == true_pool_indices).float().mean().item(),
            "prompt_answer_retrieval_top1": pooled_retrieval_top1(prompt_latent, answer_latent, config.text_align_temperature),
            "latent_answer_retrieval_top1": pooled_retrieval_top1(latent_answer, answer_latent, config.text_align_temperature),
            "latent_answer_cosine": (
                F.normalize(latent_answer.mean(dim=1), dim=-1)
                * F.normalize(answer_latent.mean(dim=1), dim=-1)
            ).sum(dim=1).mean().item(),
        }
        if gate is not None:
            entropy = -(gate * gate.clamp_min(1e-8).log()).sum(dim=1).mean()
            row["moe_gate_entropy"] = entropy.item()
            row["moe_gate_max_weight"] = gate.max(dim=1).values.mean().item()
        rows_out.append(row)
    return stage_v.merge_metric_rows(rows_out)


def ranking_loss_for_batch(
    model: nn.Module,
    batch: StageABSemanticSet,
    config: StageABConfig,
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
    output["labels"] = labels
    output["candidate_answer_tokens"] = answers
    output["true_answer_tokens"] = answers[torch.arange(answers.shape[0], device=device), labels]
    output["true_pool_indices"] = rows[torch.arange(rows.shape[0], device=device), labels]
    return output, rank + config.router_loss_weight * router, {
        "rank_loss": float(rank.detach().cpu()),
        "router_loss": float(router.detach().cpu()),
    }


def train_model(
    mode: str,
    model: nn.Module,
    train: StageABSemanticSet,
    val: StageABSemanticSet,
    config: StageABConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> tuple[SupervisionHeads | None, dict[str, object]]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    use_supervision = mode in {"supervised_no_teacher", "token_aligned_teacher", *TEXT_ALIGNMENT_MODES}
    torch.manual_seed(seed + 17)
    heads = (
        SupervisionHeads(
            config,
            use_teacher=mode in {"token_aligned_teacher", *TEXT_ALIGNMENT_MODES},
            use_token_alignment=mode in {"token_aligned_teacher", *TEXT_ALIGNMENT_MODES},
        ).to(device)
        if use_supervision
        else None
    )
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
        if mode in TEXT_ALIGNMENT_MODES:
            text_loss, text_metrics = text_alignment_loss(model, output, config)
            loss = loss + text_loss
            loss_metrics.update(text_metrics)
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
    model: nn.Module,
    heads: SupervisionHeads | None,
    test: StageABSemanticSet,
    config: StageABConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    if heads is not None:
        heads.eval()
    object_rows = []
    spatial_rows = []
    count_rows = []
    teacher_rows = []
    token_alignment_rows = []
    output: dict[str, object] = {}
    if heads is not None:
        for start in range(0, len(test.pixels.examples), config.batch_size):
            batch = test.subset(list(range(start, min(start + config.batch_size, len(test.pixels.examples))))).to(device)
            reps = model.representations(batch.pixels)  # type: ignore[attr-defined]
            object_rows.append(stage_w.slot_metrics(heads.object_decoder, model, batch.semantic, stage_w_config(config), device=device))
            spatial_rows.append(stage_v.semantic_metrics(heads.spatial_decoder(reps["spatial_tokens"]), batch.semantic))
            count_rows.append(count_metrics(heads.count_decoder(reps["count_tokens"]), batch.count))
            if heads.teacher_bus is not None and heads.teacher_decoder is not None:
                teacher_tokens = heads.teacher_bus(reps["patch_tokens"])
                teacher_rows.append(stage_v.semantic_metrics(heads.teacher_decoder(teacher_tokens), batch.semantic))
                if heads.token_aligner is not None:
                    token_alignment_rows.append(token_alignment_metrics(heads.token_aligner, reps, teacher_tokens, config))
        output.update({
            "object_decoder": stage_v.merge_metric_rows(object_rows),
            "spatial_decoder": stage_v.merge_metric_rows(spatial_rows),
            "count_decoder": stage_v.merge_metric_rows(count_rows),
        })
        if teacher_rows:
            output["teacher_decoder"] = stage_v.merge_metric_rows(teacher_rows)
        if token_alignment_rows:
            output["token_alignment"] = stage_v.merge_metric_rows(token_alignment_rows)
    text_metrics = evaluate_text_alignment_model(model, test, config, candidate_pool=candidate_pool, seed=seed, device=device)
    if text_metrics:
        output["text_alignment"] = text_metrics
    return output


class AnswerProbe(nn.Module):
    def __init__(self, config: StageABConfig, class_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.probe_hidden),
            nn.GELU(),
            nn.Linear(config.probe_hidden, class_count),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens.mean(dim=1))


def answer_labels(data: StageABSemanticSet, label_by_answer: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([label_by_answer[example.answer] for example in data.pixels.examples], dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_answer_probe(
    model: stage_y.ParallelInputExpertVLM,
    probe: AnswerProbe,
    stage_name: str,
    test: StageABSemanticSet,
    config: StageABConfig,
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
    train: StageABSemanticSet,
    test: StageABSemanticSet,
    config: StageABConfig,
    *,
    label_by_answer: dict[str, int],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    torch.manual_seed(seed)
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
    train: StageABSemanticSet,
    test: StageABSemanticSet,
    config: StageABConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    torch.manual_seed(seed)
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
    train: StageABSemanticSet,
    test: StageABSemanticSet,
    config: StageABConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    torch.manual_seed(seed)
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
    train: StageABSemanticSet,
    test: StageABSemanticSet,
    config: StageABConfig,
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


def run_experiment(config: StageABConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_ab_text_moe_alignment device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    data, data_stats = build_data(config)
    train = data["train"].to(device)
    val = data["val"].to(device)
    test = data["test"].to(device)
    stage_u.stage_r.write_samples(data["test"].pixels, data["test"].meta, output_path.parent / "samples")
    candidate_pool = stage_u.build_candidate_pool(stage_y.stage_u_config(stage_y_config(config)), device=device)

    models: dict[str, nn.Module] = {}
    for mode in config.modes:
        torch.manual_seed(config.seed + 700)
        if mode in TEXT_ALIGNMENT_MODES:
            models[mode] = TextAlignedMoEVLM(
                config,
                use_moe=mode == "moe_text_latent_aligned",
                answer_class_count=len(candidate_pool.answers),
            ).to(device)
        else:
            models[mode] = stage_y.ParallelInputExpertVLM(stage_y_config(config), variant="parallel_direct").to(device)
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
        mode: evaluate_training_heads(
            model,
            heads_by_mode[mode],
            test,
            config,
            candidate_pool=candidate_pool,
            seed=config.seed + 4500 + index,
            device=device,
        )
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
        "experiment": "omni_transformer_stage_ab_text_moe_alignment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "base": "Stage Y parallel_direct topology: all input experts read external image/text directly, then latent reasoner aggregates.",
            "explicit_supervision": "object slot set targets, spatial semantic grid targets, and color-shape count table targets are train-only losses.",
            "teacher": "train-only patch-derived semantic bus with grid supervision; object/spatial/count tokens are distilled toward teacher cell tokens but runtime output still uses direct latent tokens.",
            "token_alignment": "token_aligned_teacher resamples patch/object/spatial/count/reasoning tokens to teacher cell-token positions, then applies same-position teacher and cross-expert contrastive alignment as train-only losses.",
            "text_latent_alignment": "text_latent_aligned aligns prompt latent, true-answer latent, and runtime latent-to-answer tokens; candidate scores combine cross-attention scorer with latent-answer scorer.",
            "moe_reasoner": "moe_text_latent_aligned replaces the single latent reasoner with a routed MoE Transformer reasoner while keeping input experts parallel-direct.",
            "diagnostic_stages": DIAGNOSTIC_STAGES,
            "token_alignment_stages": TOKEN_ALIGNMENT_STAGES,
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
            "token_teacher_retrieval_top1": mean(f"trained_head_metrics.{mode}.token_alignment.teacher_retrieval_top1_mean"),
            "token_teacher_cosine": mean(f"trained_head_metrics.{mode}.token_alignment.teacher_cosine_mean"),
            "token_cross_retrieval_top1": mean(f"trained_head_metrics.{mode}.token_alignment.cross_retrieval_top1_mean"),
            "token_cross_cosine": mean(f"trained_head_metrics.{mode}.token_alignment.cross_cosine_mean"),
            "prompt_answer_retrieval_top1": mean(f"trained_head_metrics.{mode}.text_alignment.prompt_answer_retrieval_top1"),
            "latent_answer_retrieval_top1": mean(f"trained_head_metrics.{mode}.text_alignment.latent_answer_retrieval_top1"),
            "latent_answer_candidate_top1": mean(f"trained_head_metrics.{mode}.text_alignment.latent_answer_candidate_top1"),
            "latent_answer_class_accuracy": mean(f"trained_head_metrics.{mode}.text_alignment.latent_answer_class_accuracy"),
            "latent_answer_cosine": mean(f"trained_head_metrics.{mode}.text_alignment.latent_answer_cosine"),
            "moe_gate_entropy": mean(f"trained_head_metrics.{mode}.text_alignment.moe_gate_entropy"),
            "moe_gate_max_weight": mean(f"trained_head_metrics.{mode}.text_alignment.moe_gate_max_weight"),
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
        "experiment": "omni_transformer_stage_ab_text_moe_alignment_sweep",
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_ab_text_moe_alignment/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_ab_text_moe_alignment/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_ab_text_moe_alignment/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702")
    parser.add_argument("--modes", default="baseline_parallel,supervised_no_teacher,token_aligned_teacher,text_latent_aligned,moe_text_latent_aligned")
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
    parser.add_argument("--token-teacher-align-weight", type=float, default=0.08)
    parser.add_argument("--token-cross-align-weight", type=float, default=0.04)
    parser.add_argument("--token-alignment-temperature", type=float, default=0.07)
    parser.add_argument("--text-latent-tokens", type=int, default=8)
    parser.add_argument("--text-align-temperature", type=float, default=0.07)
    parser.add_argument("--prompt-answer-align-weight", type=float, default=0.08)
    parser.add_argument("--latent-answer-align-weight", type=float, default=0.12)
    parser.add_argument("--latent-answer-rank-weight", type=float, default=0.20)
    parser.add_argument("--latent-answer-class-weight", type=float, default=0.20)
    parser.add_argument("--latent-answer-score-weight", type=float, default=0.35)
    parser.add_argument("--moe-experts", type=int, default=4)
    parser.add_argument("--moe-balance-weight", type=float, default=0.02)
    args = parser.parse_args()

    modes = parse_csv_strings(args.modes)
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    base_config = StageABConfig(
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
        token_teacher_align_weight=args.token_teacher_align_weight,
        token_cross_align_weight=args.token_cross_align_weight,
        token_alignment_temperature=args.token_alignment_temperature,
        text_latent_tokens=args.text_latent_tokens,
        text_align_temperature=args.text_align_temperature,
        prompt_answer_align_weight=args.prompt_answer_align_weight,
        latent_answer_align_weight=args.latent_answer_align_weight,
        latent_answer_rank_weight=args.latent_answer_rank_weight,
        latent_answer_class_weight=args.latent_answer_class_weight,
        latent_answer_score_weight=args.latent_answer_score_weight,
        moe_experts=args.moe_experts,
        moe_balance_weight=args.moe_balance_weight,
        modes=modes,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageABConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()


