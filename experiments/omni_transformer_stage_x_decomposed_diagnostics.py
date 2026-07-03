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
import omni_transformer_stage_s_scorer_reconstruction as stage_s
import omni_transformer_stage_u_object_slots as stage_u
import omni_transformer_stage_v_expert_alignment as stage_v
import omni_transformer_stage_w_alignment_mechanisms as stage_w


MODEL_MODES = ("ranking_only", "shared_grid_decoder")
TRANSFER_STAGES = stage_v.DIAGNOSTIC_STAGES
ANSWER_STAGES = (
    "patch_tokens",
    "object_slots",
    "spatial_tokens",
    "count_tokens",
    "fusion_tokens",
    "wide_latent",
    "object_combo",
    "no_patch_visual_concat",
    "raw_expert_concat",
)
OBJECT_STAGES = ("object_slots", "patch_tokens", "wide_latent", "raw_expert_concat")
SCORER_STAGES = ("wide_latent", "raw_expert_concat", "object_combo", "patch_tokens")
SEMANTIC_READERS = ("grid_mlp_small", "grid_mlp_large", "grid_transformer")
ANSWER_READERS = ("answer_mlp_small", "answer_transformer")


@dataclass(frozen=True)
class StageXConfig:
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
    train_steps: int = 320
    eval_every: int = 160
    router_loss_weight: float = 0.05
    shared_loss_weight: float = 0.30
    semantic_probe_steps: int = 70
    answer_probe_steps: int = 90
    object_probe_steps: int = 110
    scorer_probe_steps: int = 110
    probe_hidden_small: int = 192
    probe_hidden_large: int = 512
    transformer_probe_layers: int = 2
    modes: tuple[str, ...] = MODEL_MODES


def stage_w_config(config: StageXConfig) -> stage_w.StageWConfig:
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
        fusion_tokens=config.fusion_tokens,
        output_tokens=config.output_tokens,
        candidate_count=config.candidate_count,
        train_steps=config.train_steps,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
        shared_loss_weight=config.shared_loss_weight,
        probe_steps=config.semantic_probe_steps,
        modes=config.modes,
    )


def tokens_for_stage(reps: dict[str, torch.Tensor], stage_name: str) -> torch.Tensor:
    if stage_name in reps:
        return reps[stage_name]
    if stage_name == "object_combo":
        return torch.cat((reps["object_slots"], reps["spatial_tokens"], reps["count_tokens"]), dim=1)
    if stage_name == "no_patch_visual_concat":
        return torch.cat((reps["object_slots"], reps["spatial_tokens"], reps["count_tokens"], reps["fusion_tokens"]), dim=1)
    if stage_name == "visual_concat":
        return torch.cat(
            (
                reps["patch_tokens"],
                reps["object_slots"],
                reps["spatial_tokens"],
                reps["count_tokens"],
                reps["fusion_tokens"],
            ),
            dim=1,
        )
    raise KeyError(stage_name)


class MeanGridDecoder(nn.Module):
    def __init__(self, config: StageXConfig, hidden: int) -> None:
        super().__init__()
        self.cell_count = config.grid_size * config.grid_size
        self.d_model = config.d_model
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.cell_count * config.d_model),
        )
        self.occupancy = nn.Linear(config.d_model, 1)
        self.color = nn.Linear(config.d_model, len(stage_u.COLORS) + 1)
        self.shape = nn.Linear(config.d_model, len(stage_u.SHAPES) + 1)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.net(tokens.mean(dim=1)).view(tokens.shape[0], self.cell_count, self.d_model)
        return {
            "occupancy": self.occupancy(hidden).squeeze(-1),
            "color": self.color(hidden),
            "shape": self.shape(hidden),
        }


class TransformerGridDecoder(nn.Module):
    def __init__(self, config: StageXConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(config.grid_size * config.grid_size, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.transformer_probe_layers)
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_kv = nn.LayerNorm(config.d_model)
        self.cross = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
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
        encoded = self.encoder(tokens)
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.cross(self.norm_q(query), self.norm_kv(encoded), self.norm_kv(encoded), need_weights=False)
        hidden = query + attended
        hidden = hidden + self.ff(hidden)
        return {
            "occupancy": self.occupancy(hidden).squeeze(-1),
            "color": self.color(hidden),
            "shape": self.shape(hidden),
        }


def make_semantic_reader(reader_name: str, config: StageXConfig) -> nn.Module:
    if reader_name == "grid_mlp_small":
        return MeanGridDecoder(config, config.probe_hidden_small)
    if reader_name == "grid_mlp_large":
        return MeanGridDecoder(config, config.probe_hidden_large)
    if reader_name == "grid_transformer":
        return TransformerGridDecoder(config)
    raise ValueError(f"unknown semantic reader: {reader_name}")


class AnswerMeanProbe(nn.Module):
    def __init__(self, config: StageXConfig, class_count: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, class_count),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens.mean(dim=1))


class AnswerTransformerProbe(nn.Module):
    def __init__(self, config: StageXConfig, class_count: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.transformer_probe_layers)
        self.cross = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(config.d_model)
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, class_count),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(tokens)
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        hidden, _ = self.cross(self.norm(query), self.norm(encoded), self.norm(encoded), need_weights=False)
        return self.out(hidden.squeeze(1))


def make_answer_reader(reader_name: str, config: StageXConfig, class_count: int) -> nn.Module:
    if reader_name == "answer_mlp_small":
        return AnswerMeanProbe(config, class_count, config.probe_hidden_small)
    if reader_name == "answer_transformer":
        return AnswerTransformerProbe(config, class_count)
    raise ValueError(f"unknown answer reader: {reader_name}")


class ObjectTableProbe(nn.Module):
    def __init__(self, config: StageXConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(config.object_slots, config.d_model) * 0.02)
        self.cross = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.objectness = nn.Linear(config.d_model, 1)
        self.color = nn.Linear(config.d_model, len(stage_u.COLORS) + 1)
        self.shape = nn.Linear(config.d_model, len(stage_u.SHAPES) + 1)
        self.cell = nn.Linear(config.d_model, config.grid_size * config.grid_size + 1)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        attended, _ = self.cross(self.norm(query), self.norm(tokens), self.norm(tokens), need_weights=False)
        hidden = query + attended
        hidden = hidden + self.ff(hidden)
        return {
            "objectness": self.objectness(hidden).squeeze(-1),
            "color": self.color(hidden),
            "shape": self.shape(hidden),
            "cell": self.cell(hidden),
        }


def semantic_batch_loss(output: dict[str, torch.Tensor], batch: stage_v.SemanticPixelSet) -> torch.Tensor:
    return stage_v.semantic_loss(output, batch)


@torch.no_grad()
def evaluate_semantic_reader(
    model: stage_u.StageUWideVLM,
    decoder: nn.Module,
    stage_name: str,
    data: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    decoder.eval()
    rows = []
    for start in range(0, len(data.pixels.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.pixels.examples))))).to(device)
        reps = model.representations(batch.pixels)
        rows.append(stage_v.semantic_metrics(decoder(tokens_for_stage(reps, stage_name)), batch))
    return stage_v.merge_metric_rows(rows)


def train_semantic_transfer_probe(
    model: stage_u.StageUWideVLM,
    reader_name: str,
    source_stage: str,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    decoder = make_semantic_reader(reader_name, config).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.semantic_probe_steps):
        batch = stage_v.random_semantic_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            tokens = tokens_for_stage(model.representations(batch.pixels), source_stage).detach()
        loss = semantic_batch_loss(decoder(tokens), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    transfer = {
        target_stage: evaluate_semantic_reader(model, decoder, target_stage, test, config, device=device)
        for target_stage in TRANSFER_STAGES
    }
    return {
        "reader": reader_name,
        "source_stage": source_stage,
        "reader_parameter_count": sum(parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad),
        "training_seconds": round(time.perf_counter() - started, 3),
        "transfer": transfer,
    }


def semantic_transfer_suite(
    model: stage_u.StageUWideVLM,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for reader_index, reader_name in enumerate(SEMANTIC_READERS):
        probes = {
            source_stage: train_semantic_transfer_probe(
                model,
                reader_name,
                source_stage,
                train,
                test,
                config,
                seed=seed + reader_index * 10_000 + stage_index * 211,
                device=device,
            )
            for stage_index, source_stage in enumerate(TRANSFER_STAGES)
        }
        diag = [
            float(probes[stage_name]["transfer"][stage_name]["cell_info_accuracy"])
            for stage_name in TRANSFER_STAGES
        ]
        offdiag = [
            float(probes[source]["transfer"][target]["cell_info_accuracy"])
            for source in TRANSFER_STAGES
            for target in TRANSFER_STAGES
            if source != target
        ]
        output[reader_name] = {
            "source_probe_transfer": probes,
            "summary": {
                "diagonal_cell_info_accuracy": statistics.fmean(diag),
                "offdiag_cell_info_accuracy": statistics.fmean(offdiag),
                "language_gap_cell_info_accuracy": statistics.fmean(diag) - statistics.fmean(offdiag),
            },
        }
    return output


def answer_to_label_map(candidate_pool: stage_q.CandidatePool) -> dict[str, int]:
    return {answer: index for index, answer in enumerate(candidate_pool.answers)}


def answer_labels(data: stage_v.SemanticPixelSet, label_by_answer: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([label_by_answer[example.answer] for example in data.pixels.examples], dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_answer_probe(
    model: stage_u.StageUWideVLM,
    probe: nn.Module,
    stage_name: str,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
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
        reps = model.representations(batch.pixels)
        pred = probe(tokens_for_stage(reps, stage_name)).argmax(dim=1)
        hits.extend((pred == labels).float().detach().cpu().tolist())
    return {
        "answer_exact": statistics.fmean(hits),
        "random_exact": 1.0 / len(label_by_answer),
    }


def train_answer_probe(
    model: stage_u.StageUWideVLM,
    reader_name: str,
    stage_name: str,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    label_by_answer: dict[str, int],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = make_answer_reader(reader_name, config, len(label_by_answer)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.answer_probe_steps):
        batch = stage_v.random_semantic_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        labels = answer_labels(batch, label_by_answer, device)
        with torch.no_grad():
            tokens = tokens_for_stage(model.representations(batch.pixels), stage_name).detach()
        logits = probe(tokens)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    metrics = evaluate_answer_probe(model, probe, stage_name, test, config, label_by_answer=label_by_answer, device=device)
    metrics.update(
        {
            "reader_parameter_count": sum(parameter.numel() for parameter in probe.parameters() if parameter.requires_grad),
            "training_seconds": round(time.perf_counter() - started, 3),
        }
    )
    return metrics


def answer_probe_suite(
    model: stage_u.StageUWideVLM,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    label_by_answer: dict[str, int],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for reader_index, reader_name in enumerate(ANSWER_READERS):
        output[reader_name] = {
            stage_name: train_answer_probe(
                model,
                reader_name,
                stage_name,
                train,
                test,
                config,
                label_by_answer=label_by_answer,
                seed=seed + reader_index * 20_000 + stage_index * 307,
                device=device,
            )
            for stage_index, stage_name in enumerate(ANSWER_STAGES)
        }
    return output


def object_probe_loss(output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    objectness_loss = F.binary_cross_entropy_with_logits(output["objectness"], targets["objectness"])
    positive = targets["objectness"].bool()
    if positive.any():
        color_loss = F.cross_entropy(output["color"][positive], targets["color"][positive])
        shape_loss = F.cross_entropy(output["shape"][positive], targets["shape"][positive])
        cell_loss = F.cross_entropy(output["cell"][positive], targets["cell"][positive])
    else:
        color_loss = output["color"].sum() * 0.0
        shape_loss = output["shape"].sum() * 0.0
        cell_loss = output["cell"].sum() * 0.0
    return objectness_loss + color_loss + shape_loss + cell_loss


@torch.no_grad()
def object_probe_metrics(
    probe: ObjectTableProbe,
    model: stage_u.StageUWideVLM,
    stage_name: str,
    data: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    probe.eval()
    rows = []
    set_recall = []
    set_exact = []
    for start in range(0, len(data.pixels.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.pixels.examples))))).to(device)
        targets = stage_w.slot_targets(batch, stage_w_config(config), device=device)
        reps = model.representations(batch.pixels)
        output = probe(tokens_for_stage(reps, stage_name))
        objectness_pred = (torch.sigmoid(output["objectness"]) >= 0.5)
        positive = targets["objectness"].bool()
        row = {
            "objectness_accuracy": (objectness_pred == positive).float().mean().item(),
            "slot_exact": 0.0,
            "positive_color_accuracy": 0.0,
            "positive_shape_accuracy": 0.0,
            "positive_cell_accuracy": 0.0,
        }
        color_pred = output["color"].argmax(dim=-1)
        shape_pred = output["shape"].argmax(dim=-1)
        cell_pred = output["cell"].argmax(dim=-1)
        if positive.any():
            row["positive_color_accuracy"] = (color_pred[positive] == targets["color"][positive]).float().mean().item()
            row["positive_shape_accuracy"] = (shape_pred[positive] == targets["shape"][positive]).float().mean().item()
            row["positive_cell_accuracy"] = (cell_pred[positive] == targets["cell"][positive]).float().mean().item()
            exact = (
                objectness_pred
                & positive
                & (color_pred == targets["color"])
                & (shape_pred == targets["shape"])
                & (cell_pred == targets["cell"])
            ) | (~objectness_pred & ~positive)
            row["slot_exact"] = exact.float().mean().item()
        rows.append(row)
        for batch_index in range(len(batch.meta)):
            true_slots = positive[batch_index].nonzero(as_tuple=False).flatten().tolist()
            true_set = {
                (
                    int(targets["cell"][batch_index, slot].item()),
                    int(targets["color"][batch_index, slot].item()),
                    int(targets["shape"][batch_index, slot].item()),
                )
                for slot in true_slots
            }
            k = max(len(true_set), 1)
            top_slots = torch.topk(output["objectness"][batch_index], k=min(k, output["objectness"].shape[1])).indices.tolist()
            pred_set = {
                (
                    int(cell_pred[batch_index, slot].item()),
                    int(color_pred[batch_index, slot].item()),
                    int(shape_pred[batch_index, slot].item()),
                )
                for slot in top_slots
            }
            matches = len(pred_set & true_set)
            set_recall.append(matches / max(len(true_set), 1))
            set_exact.append(1.0 if pred_set == true_set else 0.0)
    merged = stage_v.merge_metric_rows(rows)
    merged["set_recall_at_true_count"] = statistics.fmean(set_recall)
    merged["set_exact_at_true_count"] = statistics.fmean(set_exact)
    return merged


def train_object_probe(
    model: stage_u.StageUWideVLM,
    stage_name: str,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = ObjectTableProbe(config).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.object_probe_steps):
        batch = stage_v.random_semantic_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        targets = stage_w.slot_targets(batch, stage_w_config(config), device=device)
        with torch.no_grad():
            tokens = tokens_for_stage(model.representations(batch.pixels), stage_name).detach()
        loss = object_probe_loss(probe(tokens), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    metrics = object_probe_metrics(probe, model, stage_name, test, config, device=device)
    metrics.update(
        {
            "reader_parameter_count": sum(parameter.numel() for parameter in probe.parameters() if parameter.requires_grad),
            "training_seconds": round(time.perf_counter() - started, 3),
        }
    )
    return metrics


def object_probe_suite(
    model: stage_u.StageUWideVLM,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    return {
        stage_name: train_object_probe(
            model,
            stage_name,
            train,
            test,
            config,
            seed=seed + index * 401,
            device=device,
        )
        for index, stage_name in enumerate(OBJECT_STAGES)
    }


@torch.no_grad()
def evaluate_scorer_probe(
    model: stage_u.StageUWideVLM,
    scorer: stage_s.CrossAttentionAnswerScorer,
    stage_name: str,
    data: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    scorer.eval()
    hits = []
    reciprocal = []
    for start in range(0, len(data.pixels.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.pixels.examples))))).to(device)
        rows, true_indices = stage_u.candidate_rows_for_examples(batch.pixels.examples, candidate_pool, seed=seed + start)
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        reps = model.representations(batch.pixels)
        scores = scorer(tokens_for_stage(reps, stage_name), answers)
        order = scores.argsort(dim=1, descending=True).detach().cpu().tolist()
        for row_order, true_index in zip(order, true_indices):
            rank = row_order.index(true_index) + 1
            hits.append(1.0 if rank == 1 else 0.0)
            reciprocal.append(1.0 / rank)
    return {
        "rank_top1": statistics.fmean(hits),
        "rank_mrr": statistics.fmean(reciprocal),
        "candidate_count": config.candidate_count,
        "random_top1": 1.0 / config.candidate_count,
    }


def train_scorer_probe(
    model: stage_u.StageUWideVLM,
    stage_name: str,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    scorer = stage_s.CrossAttentionAnswerScorer(stage_u.stage_q_config(stage_v.stage_u_config(stage_w.stage_v_config(stage_w_config(config))))).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.scorer_probe_steps):
        batch = stage_v.random_semantic_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        rows, true_indices = stage_u.candidate_rows_for_examples(batch.pixels.examples, candidate_pool, seed=rng.randrange(1_000_000_000))
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        with torch.no_grad():
            context = tokens_for_stage(model.representations(batch.pixels), stage_name).detach()
        scores = scorer(context, answers)
        loss = F.cross_entropy(scores, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    metrics = evaluate_scorer_probe(
        model,
        scorer,
        stage_name,
        test,
        config,
        candidate_pool=candidate_pool,
        seed=seed + 999,
        device=device,
    )
    metrics.update(
        {
            "reader_parameter_count": sum(parameter.numel() for parameter in scorer.parameters() if parameter.requires_grad),
            "training_seconds": round(time.perf_counter() - started, 3),
        }
    )
    return metrics


def scorer_probe_suite(
    model: stage_u.StageUWideVLM,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    return {
        stage_name: train_scorer_probe(
            model,
            stage_name,
            train,
            test,
            config,
            candidate_pool=candidate_pool,
            seed=seed + index * 503,
            device=device,
        )
        for index, stage_name in enumerate(SCORER_STAGES)
    }


def run_decomposed_diagnostics(
    mode: str,
    model: stage_u.StageUWideVLM,
    train: stage_v.SemanticPixelSet,
    test: stage_v.SemanticPixelSet,
    config: StageXConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    labels = answer_to_label_map(candidate_pool)
    return {
        "semantic_transfer": semantic_transfer_suite(model, train, test, config, seed=seed + 10_000, device=device),
        "answer_reconstruction": answer_probe_suite(
            model,
            train,
            test,
            config,
            label_by_answer=labels,
            seed=seed + 20_000,
            device=device,
        ),
        "object_table": object_probe_suite(model, train, test, config, seed=seed + 30_000, device=device),
        "frozen_scorer": scorer_probe_suite(
            model,
            train,
            test,
            config,
            candidate_pool=candidate_pool,
            seed=seed + 40_000,
            device=device,
        ),
    }


def run_experiment(config: StageXConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_x_decomposed_diagnostics device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    w_config = stage_w_config(config)
    v_config = stage_w.stage_v_config(w_config)
    data, data_stats = stage_v.build_semantic_data(v_config)
    train = data["train"].to(device)
    val = data["val"].to(device)
    test = data["test"].to(device)
    stage_u.stage_r.write_samples(data["test"].pixels, data["test"].meta, output_path.parent / "samples")
    candidate_pool = stage_u.build_candidate_pool(stage_v.stage_u_config(v_config), device=device)

    metrics: dict[str, object] = {}
    training: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    for index, mode in enumerate(config.modes):
        model, semantic_decoder, slot_decoder, projector, common_bus, train_stats = stage_w.train_mode(
            mode,
            train,
            val,
            w_config,
            candidate_pool=candidate_pool,
            seed=config.seed + 1100 + index * 173,
            device=device,
        )
        del semantic_decoder, slot_decoder, projector, common_bus
        training[mode] = train_stats
        metrics[mode] = stage_v.evaluate_model_bundle(
            model,
            test,
            v_config,
            candidate_pool=candidate_pool,
            device=device,
            seed=config.seed + 3100 + index * 19,
        )
        diagnostics[mode] = run_decomposed_diagnostics(
            mode,
            model,
            train,
            test,
            config,
            candidate_pool=candidate_pool,
            seed=config.seed + 6100 + index * 307,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "experiment": "omni_transformer_stage_x_decomposed_diagnostics",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "base": "Stage W trained expert tokens, frozen during decomposition probes",
            "semantic_transfer": "train semantic scene decoder on one source stage and evaluate all target stages, with MLP-small, MLP-large and Transformer readers",
            "answer_reconstruction": "train answer-class probes from fixed token sets; tests whether task-answer information exists independently of final scorer",
            "object_table": "train object-table probe from fixed token sets; reports sorted-slot metrics and set recall/exact ignoring slot order",
            "frozen_scorer": "train only a candidate scorer on frozen token sets; tests whether final readout/training is the bottleneck",
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

    summary: dict[str, object] = {}
    for mode in runs[0]["metrics"]:
        mode_summary: dict[str, object] = {
            "rank_top1": mean(f"metrics.{mode}.full.rank_top1"),
            "no_patch_top1": mean(f"metrics.{mode}.no_patch_expert.rank_top1"),
            "train_seconds": mean(f"training.{mode}.training_seconds"),
            "params": mean(f"training.{mode}.trainable_parameter_count"),
            "semantic_transfer": {
                reader: {
                    "diagonal_cell_info_accuracy": mean(
                        f"diagnostics.{mode}.semantic_transfer.{reader}.summary.diagonal_cell_info_accuracy"
                    ),
                    "offdiag_cell_info_accuracy": mean(
                        f"diagnostics.{mode}.semantic_transfer.{reader}.summary.offdiag_cell_info_accuracy"
                    ),
                    "language_gap_cell_info_accuracy": mean(
                        f"diagnostics.{mode}.semantic_transfer.{reader}.summary.language_gap_cell_info_accuracy"
                    ),
                    "reader_params": mean(
                        f"diagnostics.{mode}.semantic_transfer.{reader}.source_probe_transfer.patch_tokens.reader_parameter_count"
                    ),
                }
                for reader in SEMANTIC_READERS
            },
            "answer_reconstruction": {
                reader: {
                    stage_name: mean(f"diagnostics.{mode}.answer_reconstruction.{reader}.{stage_name}.answer_exact")
                    for stage_name in ANSWER_STAGES
                }
                for reader in ANSWER_READERS
            },
            "object_table": {
                stage_name: {
                    "set_recall_at_true_count": mean(
                        f"diagnostics.{mode}.object_table.{stage_name}.set_recall_at_true_count"
                    ),
                    "set_exact_at_true_count": mean(
                        f"diagnostics.{mode}.object_table.{stage_name}.set_exact_at_true_count"
                    ),
                }
                for stage_name in OBJECT_STAGES
            },
            "frozen_scorer": {
                stage_name: mean(f"diagnostics.{mode}.frozen_scorer.{stage_name}.rank_top1")
                for stage_name in SCORER_STAGES
            },
        }
        summary[mode] = mode_summary
    return {
        "experiment": "omni_transformer_stage_x_decomposed_diagnostics_sweep",
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_x_decomposed_diagnostics/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_x_decomposed_diagnostics/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_x_decomposed_diagnostics/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702")
    parser.add_argument("--modes", default="ranking_only,shared_grid_decoder")
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
    parser.add_argument("--train-steps", type=int, default=320)
    parser.add_argument("--eval-every", type=int, default=160)
    parser.add_argument("--shared-loss-weight", type=float, default=0.30)
    parser.add_argument("--semantic-probe-steps", type=int, default=70)
    parser.add_argument("--answer-probe-steps", type=int, default=90)
    parser.add_argument("--object-probe-steps", type=int, default=110)
    parser.add_argument("--scorer-probe-steps", type=int, default=110)
    parser.add_argument("--probe-hidden-small", type=int, default=192)
    parser.add_argument("--probe-hidden-large", type=int, default=512)
    parser.add_argument("--transformer-probe-layers", type=int, default=2)
    args = parser.parse_args()

    modes = parse_csv_strings(args.modes)
    unknown = set(modes) - set(stage_w.MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    base_config = StageXConfig(
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
        shared_loss_weight=args.shared_loss_weight,
        semantic_probe_steps=args.semantic_probe_steps,
        answer_probe_steps=args.answer_probe_steps,
        object_probe_steps=args.object_probe_steps,
        scorer_probe_steps=args.scorer_probe_steps,
        probe_hidden_small=args.probe_hidden_small,
        probe_hidden_large=args.probe_hidden_large,
        transformer_probe_layers=args.transformer_probe_layers,
        modes=modes,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageXConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
