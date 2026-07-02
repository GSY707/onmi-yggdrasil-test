from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import string
import sys
import time
from typing import Iterable

from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omni_transformer_stage_h import TransformerBlock
import omni_transformer_stage_m_moe_multimodal_llm as stage_m
import omni_transformer_stage_p_llava_moe as stage_p


EXPERTS = ("vision", "prompt_text", "dialogue_history", "image_text_fusion")
VISION_EXPERT = 0
PROMPT_EXPERT = 1
HISTORY_EXPERT = 2
FUSION_EXPERT = 3

CHARSET = string.ascii_lowercase + string.digits + " .,!?;:'\"-/()[]%$&+\n"
PAD = 0
BOS = 1
EOS = 2
CHAR_TO_ID = {char: index + 3 for index, char in enumerate(CHARSET)}
ID_TO_CHAR = {index: char for char, index in CHAR_TO_ID.items()}
VOCAB = len(CHAR_TO_ID) + 3


@dataclass(frozen=True)
class StageQConfig:
    train_size: int = 768
    val_size: int = 192
    test_size: int = 256
    batch_size: int = 32
    seed: int = 20260701
    image_size: int = 64
    prompt_len: int = 192
    answer_len: int = 192
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    latent_tokens: int = 12
    candidate_count: int = 8
    text_only_steps: int = 900
    direct_steps: int = 900
    mean_steps: int = 900
    moe_steps: int = 1200
    eval_every: int = 120
    router_loss_weight: float = 0.10
    max_answer_chars: int = 360
    max_prompt_chars: int = 700
    annotation_files: tuple[str, ...] = stage_p.DATASET_FILES
    data_cache_dir: str = "artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/data"


@dataclass
class PixelSet:
    examples: list[stage_p.LLaVAExample]
    image: torch.Tensor
    prompt: torch.Tensor
    required_experts: torch.Tensor

    def subset(self, indices: list[int]) -> "PixelSet":
        return PixelSet(
            examples=[self.examples[index] for index in indices],
            image=self.image[indices],
            prompt=self.prompt[indices],
            required_experts=self.required_experts[indices],
        )

    def to(self, device: torch.device) -> "PixelSet":
        return PixelSet(
            examples=self.examples,
            image=self.image.to(device=device, dtype=torch.float32),
            prompt=self.prompt.to(device=device),
            required_experts=self.required_experts.to(device=device, dtype=torch.float32),
        )


@dataclass
class CandidatePool:
    answers: list[str]
    tokens: torch.Tensor
    index_by_answer: dict[str, int]


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def encode_text(value: str, length: int, *, add_eos: bool) -> list[int]:
    normalized = value.lower()
    ids = [CHAR_TO_ID[char] for char in normalized if char in CHAR_TO_ID]
    if add_eos:
        ids = ids[: length - 1] + [EOS]
    else:
        ids = ids[:length]
    return ids + [PAD] * (length - len(ids))


def decode_tokens(tokens: Iterable[int]) -> str:
    chars: list[str] = []
    for token in tokens:
        token = int(token)
        if token in (PAD, BOS):
            continue
        if token == EOS:
            break
        chars.append(ID_TO_CHAR.get(token, ""))
    return "".join(chars).strip()


def random_batch(data: PixelSet, *, rng: random.Random, batch_size: int, device: torch.device) -> PixelSet:
    return data.subset([rng.randrange(len(data.examples)) for _ in range(batch_size)]).to(device)


def build_candidate_pool(answer_pool: list[str], config: StageQConfig, *, device: torch.device) -> CandidatePool:
    answers = list(dict.fromkeys(answer_pool))
    tokens = torch.tensor([encode_text(answer, config.answer_len, add_eos=True) for answer in answers], dtype=torch.long, device=device)
    return CandidatePool(answers=answers, tokens=tokens, index_by_answer={answer: index for index, answer in enumerate(answers)})


def image_to_tensor(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def cache_key(config: StageQConfig, split_name: str) -> str:
    payload = json.dumps(
        {
            "split": split_name,
            "seed": config.seed,
            "sizes": (config.train_size, config.val_size, config.test_size),
            "image_size": config.image_size,
            "prompt_len": config.prompt_len,
            "files": config.annotation_files,
        },
        sort_keys=True,
    )
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def save_pixel_set(path: Path, data: PixelSet, cost: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "examples": [asdict(example) for example in data.examples],
            "image": data.image,
            "prompt": data.prompt,
            "required_experts": data.required_experts,
            "cost": cost,
        },
        path,
    )


def load_pixel_set(path: Path) -> tuple[PixelSet, dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        PixelSet(
            examples=[stage_p.LLaVAExample(**item) for item in payload["examples"]],
            image=payload["image"],
            prompt=payload["prompt"],
            required_experts=payload["required_experts"],
        ),
        payload["cost"],
    )


def build_pixel_set(examples: list[stage_p.LLaVAExample], config: StageQConfig) -> tuple[PixelSet, dict[str, float]]:
    started = time.perf_counter()
    images = torch.stack([image_to_tensor(example.image_path, config.image_size) for example in examples], dim=0)
    prompts = torch.tensor([encode_text(example.prompt, config.prompt_len, add_eos=True) for example in examples], dtype=torch.long)
    required = torch.tensor([stage_p.required_experts(example) for example in examples], dtype=torch.float32)
    seconds = time.perf_counter() - started
    return (
        PixelSet(examples=examples, image=images, prompt=prompts, required_experts=required),
        {
            "pixel_cache_seconds": round(seconds, 4),
            "pixel_cache_ms_per_example": round(seconds * 1000.0 / max(len(examples), 1), 4),
        },
    )


def load_or_build_pixel_sets(
    splits: dict[str, list[stage_p.LLaVAExample]],
    config: StageQConfig,
    cache_dir: Path,
) -> tuple[dict[str, PixelSet], dict[str, dict[str, float]]]:
    pixel_dir = cache_dir / "pixel_tensors"
    sets: dict[str, PixelSet] = {}
    costs: dict[str, dict[str, float]] = {}
    for split_name, examples in splits.items():
        path = pixel_dir / f"{split_name}_{cache_key(config, split_name)}.pt"
        if path.exists():
            sets[split_name], costs[split_name] = load_pixel_set(path)
            continue
        data, cost = build_pixel_set(examples, config)
        save_pixel_set(path, data, cost)
        sets[split_name] = data
        costs[split_name] = cost
    return sets, costs


def load_data(config: StageQConfig) -> tuple[dict[str, PixelSet], dict[str, object]]:
    cache_dir = Path(config.data_cache_dir)
    annotation_paths = stage_p.download_annotations(
        stage_p.StagePConfig(
            annotation_files=config.annotation_files,
            data_cache_dir=config.data_cache_dir,
            max_answer_chars=config.max_answer_chars,
            max_prompt_chars=config.max_prompt_chars,
        ),
        cache_dir,
    )
    parsed = stage_p.parse_annotations(
        annotation_paths,
        stage_p.StagePConfig(
            annotation_files=config.annotation_files,
            max_answer_chars=config.max_answer_chars,
            max_prompt_chars=config.max_prompt_chars,
        ),
    )
    splits, data_stats = stage_p.split_examples(
        parsed,
        stage_p.StagePConfig(
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
            seed=config.seed,
            annotation_files=config.annotation_files,
            data_cache_dir=config.data_cache_dir,
        ),
        cache_dir,
    )
    pixel_sets, pixel_cost = load_or_build_pixel_sets(splits, config, cache_dir)
    data_stats["pixel_cache"] = pixel_cost
    return pixel_sets, data_stats


def noncausal_mask(length: int, *, device: torch.device) -> torch.Tensor:
    return torch.zeros(length, length, dtype=torch.bool, device=device)


class QueryResampler(nn.Module):
    def __init__(self, d_model: int, heads: int, query_count: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(query_count, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attn(self.norm_q(query), self.norm_kv(source), self.norm_kv(source), need_weights=False)
        x = query + attended
        return x + self.ff(x)


class TextTokenEncoder(nn.Module):
    def __init__(self, config: StageQConfig, *, length: int) -> None:
        super().__init__()
        self.length = length
        self.embedding = nn.Embedding(VOCAB, config.d_model)
        self.position = nn.Parameter(torch.randn(length, config.d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(max(1, config.layers // 2))])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) + self.position[: tokens.shape[1]].unsqueeze(0)
        mask = noncausal_mask(tokens.shape[1], device=tokens.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class VisionExpert(nn.Module):
    def __init__(self, config: StageQConfig) -> None:
        super().__init__()
        self.config = config
        hidden = max(32, config.d_model // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, config.d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        grid = config.image_size // 8
        self.position = nn.Parameter(torch.randn(grid * grid, config.d_model) * 0.02)
        self.resampler = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        tokens = x.flatten(2).transpose(1, 2) + self.position.unsqueeze(0)
        return self.resampler(tokens)


class ScratchExpertBank(nn.Module):
    def __init__(self, config: StageQConfig) -> None:
        super().__init__()
        self.config = config
        self.vision = VisionExpert(config)
        self.text_encoder = TextTokenEncoder(config, length=config.prompt_len)
        self.prompt_resampler = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)
        self.history_resampler = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)
        self.fusion_resampler = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)
        self.expert_type = nn.Embedding(len(EXPERTS), config.d_model)

    def forward(self, batch: PixelSet, *, zero_modalities: Iterable[str] = ()) -> list[torch.Tensor]:
        zero_set = set(zero_modalities)
        image = torch.zeros_like(batch.image) if "image" in zero_set else batch.image
        prompt = torch.full_like(batch.prompt, PAD) if "text" in zero_set else batch.prompt
        vision = self.vision(image) + self.expert_type.weight[VISION_EXPERT].view(1, 1, -1)
        text_source = self.text_encoder(prompt)
        prompt_tokens = self.prompt_resampler(text_source) + self.expert_type.weight[PROMPT_EXPERT].view(1, 1, -1)
        history_tokens = self.history_resampler(text_source) + self.expert_type.weight[HISTORY_EXPERT].view(1, 1, -1)
        fusion_tokens = self.fusion_resampler(torch.cat((vision, prompt_tokens), dim=1)) + self.expert_type.weight[FUSION_EXPERT].view(1, 1, -1)
        return [vision, prompt_tokens, history_tokens, fusion_tokens]


class AttentionPump(nn.Module):
    def __init__(self, config: StageQConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_kv = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model * 4), nn.GELU(), nn.Linear(config.d_model * 4, config.d_model))

    def forward(self, expert_tokens: list[torch.Tensor], route_weights: torch.Tensor) -> torch.Tensor:
        weighted = [tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)]
        source = torch.cat(weighted, dim=1)
        query = self.query.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attn(self.norm_q(query), self.norm_kv(source), self.norm_kv(source), need_weights=False)
        x = query + attended
        return x + self.ff(x)


class MeanPoolLatent(nn.Module):
    def __init__(self, config: StageQConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        self.proj = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model), nn.GELU())

    def forward(self, expert_tokens: list[torch.Tensor], route_weights: torch.Tensor) -> torch.Tensor:
        weighted = [tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)]
        pooled = torch.cat(weighted, dim=1).mean(dim=1, keepdim=True)
        return self.query.unsqueeze(0) + self.proj(pooled)


class ThoughtExpert(nn.Module):
    def __init__(self, config: StageQConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        mask = noncausal_mask(latent.shape[1], device=latent.device)
        for block in self.blocks:
            latent = block(latent, mask)
        return self.norm(latent)


class AnswerEncoder(nn.Module):
    def __init__(self, config: StageQConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TextTokenEncoder(config, length=config.answer_len)
        self.proj = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model), nn.GELU())

    def forward(self, answer_tokens: torch.Tensor) -> torch.Tensor:
        flat = answer_tokens.view(-1, answer_tokens.shape[-1])
        hidden = self.encoder(flat)
        mask = (flat != PAD).float().unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled).view(answer_tokens.shape[0], answer_tokens.shape[1], -1)


class TinyScratchMoEVLM(nn.Module):
    def __init__(self, config: StageQConfig, *, mode: str) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.bank = ScratchExpertBank(config)
        self.router = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(EXPERTS)),
        )
        self.pump = AttentionPump(config)
        self.mean_pool = MeanPoolLatent(config)
        self.thought = ThoughtExpert(config)
        self.context_proj = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model), nn.GELU())
        self.answer = AnswerEncoder(config)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def context(
        self,
        batch: PixelSet,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        expert_tokens = self.bank(batch, zero_modalities=zero_modalities)
        route_source = torch.cat((expert_tokens[VISION_EXPERT].mean(dim=1), expert_tokens[PROMPT_EXPERT].mean(dim=1)), dim=-1)
        route_logits = self.router(route_source)
        route_weights = torch.sigmoid(route_logits)
        if self.mode == "text_only":
            context = expert_tokens[PROMPT_EXPERT]
            latent = torch.zeros(batch.image.shape[0], self.config.latent_tokens, self.config.d_model, device=batch.image.device)
        elif self.mode == "direct":
            context = torch.cat([tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)], dim=1)
            latent = torch.zeros(batch.image.shape[0], self.config.latent_tokens, self.config.d_model, device=batch.image.device)
        elif self.mode == "mean_latent":
            latent = self.thought(self.mean_pool(expert_tokens, route_weights))
            context = torch.zeros_like(latent) if disable_latent_access else latent
        elif self.mode == "moe_latent":
            latent = self.thought(self.pump(expert_tokens, route_weights))
            context = torch.zeros_like(latent) if disable_latent_access else latent
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return {"context": context, "latent": latent, "route_logits": route_logits, "route_weights": route_weights}

    def forward(
        self,
        batch: PixelSet,
        answer_tokens: torch.Tensor,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(batch, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        context = F.normalize(self.context_proj(ctx["context"]).mean(dim=1), dim=-1)
        answers = F.normalize(self.answer(answer_tokens), dim=-1)
        scores = torch.einsum("bd,bkd->bk", context, answers) * self.temperature.exp().clamp(max=100.0)
        return {"scores": scores, **ctx}


def candidate_index_rows(
    examples: list[stage_p.LLaVAExample],
    candidate_pool: CandidatePool,
    candidate_count: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    true_indices: list[int] = []
    pool_size = len(candidate_pool.answers)
    for example in examples:
        true_pool_index = candidate_pool.index_by_answer[example.answer]
        if pool_size - 1 < candidate_count - 1:
            raise ValueError("answer pool is too small for the requested candidate count")
        sampled = rng.sample(range(pool_size - 1), candidate_count - 1)
        distractors = [item if item < true_pool_index else item + 1 for item in sampled]
        candidates = distractors + [true_pool_index]
        rng.shuffle(candidates)
        rows.append(candidates)
        true_indices.append(candidates.index(true_pool_index))
    return torch.tensor(rows, dtype=torch.long, device=candidate_pool.tokens.device), true_indices


def answer_tokens_from_indices(candidate_pool: CandidatePool, indices: torch.Tensor) -> torch.Tensor:
    return candidate_pool.tokens[indices]


def loss_for_output(output: dict[str, torch.Tensor], labels: torch.Tensor, batch: PixelSet, config: StageQConfig) -> torch.Tensor:
    rank = F.cross_entropy(output["scores"], labels)
    router = F.binary_cross_entropy_with_logits(output["route_logits"], batch.required_experts)
    return rank + config.router_loss_weight * router


@torch.no_grad()
def evaluate_ranking(
    model: TinyScratchMoEVLM,
    data: PixelSet,
    config: StageQConfig,
    *,
    device: torch.device,
    candidate_pool: CandidatePool,
    seed: int,
    zero_modalities: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    hits: list[float] = []
    reciprocal: list[float] = []
    true_scores: list[float] = []
    for start in range(0, len(data.examples), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(data.examples))))
        batch = data.subset(indices).to(device)
        rows, true_indices = candidate_index_rows(batch.examples, candidate_pool, config.candidate_count, seed=seed + start)
        answers = answer_tokens_from_indices(candidate_pool, rows)
        output = model(batch, answers, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        order = output["scores"].argsort(dim=1, descending=True).detach().cpu().tolist()
        for row_order, true_index, row_scores in zip(order, true_indices, output["scores"].detach().cpu().tolist()):
            rank = row_order.index(true_index) + 1
            hits.append(1.0 if rank == 1 else 0.0)
            reciprocal.append(1.0 / rank)
            true_scores.append(float(row_scores[true_index]))
    model.train()
    return {
        "rank_top1": statistics.fmean(hits),
        "rank_mrr": statistics.fmean(reciprocal),
        "true_answer_score": statistics.fmean(true_scores),
        "candidate_count": config.candidate_count,
    }


def train_model(
    model: TinyScratchMoEVLM,
    train: PixelSet,
    val: PixelSet,
    config: StageQConfig,
    *,
    candidate_pool: CandidatePool,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    best_top1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        rows, true_indices = candidate_index_rows(batch.examples, candidate_pool, config.candidate_count, seed=rng.randrange(1_000_000_000))
        answers = answer_tokens_from_indices(candidate_pool, rows)
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        output = model(batch, answers)
        loss = loss_for_output(output, labels, batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_ranking(model, val, config, device=device, candidate_pool=candidate_pool, seed=seed + step)
            if float(metrics["rank_top1"]) > best_top1:
                best_top1 = float(metrics["rank_top1"])
                best_state = copy.deepcopy(model.state_dict())
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "rank_top1": metrics["rank_top1"], "rank_mrr": metrics["rank_mrr"]})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} top1={metrics['rank_top1']:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_rank_top1": best_top1,
        "training_seconds": round(time.perf_counter() - started, 3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def train_latent_probe(
    model: TinyScratchMoEVLM,
    train: PixelSet,
    test: PixelSet,
    config: StageQConfig,
    *,
    device: torch.device,
    steps: int = 120,
) -> dict[str, object]:
    model.eval()
    probe = nn.Linear(config.d_model, len(EXPERTS)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr)
    rng = random.Random(config.seed + 60000)
    started = time.perf_counter()
    for _ in range(steps):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            latent = model.context(batch)["latent"].mean(dim=1)
        logits = probe(latent.detach())
        loss = F.binary_cross_entropy_with_logits(logits, batch.required_experts)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    hits = []
    with torch.no_grad():
        for start in range(0, len(test.examples), config.batch_size):
            batch = test.subset(list(range(start, min(start + config.batch_size, len(test.examples))))).to(device)
            latent = model.context(batch)["latent"].mean(dim=1)
            pred = (torch.sigmoid(probe(latent)) >= 0.5).float()
            hits.extend((pred == batch.required_experts).all(dim=1).float().detach().cpu().tolist())
    return {
        "training_seconds": round(time.perf_counter() - started, 3),
        "route_probe_exact": statistics.fmean(hits),
        "probe_steps": steps,
    }


def measure_prediction_cost(fn, *, example_count: int, device: torch.device) -> dict[str, float]:
    cuda_sync(device)
    started = time.perf_counter()
    fn()
    cuda_sync(device)
    seconds = time.perf_counter() - started
    return {"prediction_seconds": round(seconds, 4), "prediction_ms_per_example": round(seconds * 1000.0 / max(example_count, 1), 4)}


def prediction_samples(models: dict[str, TinyScratchMoEVLM], test: PixelSet, config: StageQConfig, *, device: torch.device, candidate_pool: CandidatePool) -> list[dict[str, object]]:
    selected = test.subset(list(range(min(8, len(test.examples)))))
    rows, true_indices = candidate_index_rows(selected.examples, candidate_pool, config.candidate_count, seed=config.seed + 9000)
    answers = answer_tokens_from_indices(candidate_pool, rows)
    batch = selected.to(device)
    scores = {name: model(batch, answers)["scores"].detach().cpu().tolist() for name, model in models.items()}
    output: list[dict[str, object]] = []
    for index, example in enumerate(selected.examples):
        item = {
            "id": example.id,
            "image": example.image_name,
            "prompt": example.prompt[:240],
            "expected": example.answer[:320],
            "true_candidate_index": true_indices[index],
            "candidates": [candidate_pool.answers[int(candidate_index)][:180] for candidate_index in rows[index].detach().cpu().tolist()],
        }
        for name, score_rows in scores.items():
            item[f"{name}_pred_index"] = int(max(range(len(score_rows[index])), key=lambda idx: score_rows[index][idx]))
        output.append(item)
    return output


def run_experiment(config: StageQConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_q_tiny_moe_vlm_from_scratch device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    pixel_sets, data_stats = load_data(config)
    train = pixel_sets["train"].to(device)
    val = pixel_sets["val"].to(device)
    test = pixel_sets["test"].to(device)
    stage_p.write_sample_grid(test.examples, output_path.parent / "samples" / f"seed{config.seed}")
    answer_pool = [example.answer for example in train.examples + val.examples + test.examples]
    candidate_pool = build_candidate_pool(answer_pool, config, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    models = {
        "scratch_text_only": TinyScratchMoEVLM(config, mode="text_only"),
        "scratch_direct": TinyScratchMoEVLM(config, mode="direct"),
        "scratch_mean_pool_latent": TinyScratchMoEVLM(config, mode="mean_latent"),
        "scratch_moe_attention_pump_latent": TinyScratchMoEVLM(config, mode="moe_latent"),
    }
    steps = {
        "scratch_text_only": config.text_only_steps,
        "scratch_direct": config.direct_steps,
        "scratch_mean_pool_latent": config.mean_steps,
        "scratch_moe_attention_pump_latent": config.moe_steps,
    }
    training = {
        name: train_model(
            model,
            train,
            val,
            config,
            candidate_pool=candidate_pool,
            steps=steps[name],
            seed=config.seed + 100 + index * 10,
            device=device,
            label=name,
        )
        for index, (name, model) in enumerate(models.items())
    }

    metrics = {
        name: evaluate_ranking(model, test, config, device=device, candidate_pool=candidate_pool, seed=config.seed + index * 1000)
        for index, (name, model) in enumerate(models.items(), start=1)
    }
    ablations = {
        "scratch_moe_no_latent_access": evaluate_ranking(
            models["scratch_moe_attention_pump_latent"],
            test,
            config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 8000,
            disable_latent_access=True,
        ),
        "scratch_moe_no_image_modality": evaluate_ranking(
            models["scratch_moe_attention_pump_latent"],
            test,
            config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 8001,
            zero_modalities=("image",),
        ),
        "scratch_moe_no_text_modality": evaluate_ranking(
            models["scratch_moe_attention_pump_latent"],
            test,
            config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 8002,
            zero_modalities=("text",),
        ),
    }
    latent_probe = train_latent_probe(models["scratch_moe_attention_pump_latent"], train, test, config, device=device)
    prediction_cost = {
        f"{name}_prediction": measure_prediction_cost(
            lambda model=model: evaluate_ranking(model, test, config, device=device, candidate_pool=candidate_pool, seed=config.seed + 7000),
            example_count=len(test.examples),
            device=device,
        )
        for name, model in models.items()
    }
    output = {
        "experiment": "omni_transformer_stage_q_tiny_moe_vlm_from_scratch",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "dataset": stage_p.DATASET_REPO,
            "annotation_files": config.annotation_files,
            "image_source": "COCO image URLs resolved from LLaVA image names",
            "pretrained_experts": False,
            "experts": EXPERTS,
            "vision": "tiny CNN trained from scratch on resized COCO pixels",
            "text": "character-level prompt and answer encoders trained from scratch",
            "latent_tokens": config.latent_tokens,
            "ranking_metric": "true answer versus sampled LLaVA answer distractors by trained candidate scorer",
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
        "latent_probe": latent_probe,
        "training": training,
        "prediction_cost": prediction_cost,
        "prediction_samples": prediction_samples(models, test, config, device=device, candidate_pool=candidate_pool),
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
        numbers = collect_numbers(
            {
                "metrics": run["metrics"],
                "ablations": run["ablations"],
                "latent_probe": run["latent_probe"],
                "training": run["training"],
                "prediction_cost": run["prediction_cost"],
                "data": run["data"],
            }
        )
        for key, value in numbers.items():
            buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_q_tiny_moe_vlm_from_scratch_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stage_m.stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--annotation-files", default=",".join(stage_p.DATASET_FILES))
    parser.add_argument("--data-cache-dir", default="artifacts/omni_transformer_stage_q_tiny_moe_vlm_from_scratch/data")
    parser.add_argument("--train-size", type=int, default=768)
    parser.add_argument("--val-size", type=int, default=192)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=192)
    parser.add_argument("--answer-len", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent-tokens", type=int, default=12)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--text-only-steps", type=int, default=900)
    parser.add_argument("--direct-steps", type=int, default=900)
    parser.add_argument("--mean-steps", type=int, default=900)
    parser.add_argument("--moe-steps", type=int, default=1200)
    args = parser.parse_args()

    base_config = StageQConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        image_size=args.image_size,
        prompt_len=args.prompt_len,
        answer_len=args.answer_len,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        latent_tokens=args.latent_tokens,
        candidate_count=args.candidate_count,
        text_only_steps=args.text_only_steps,
        direct_steps=args.direct_steps,
        mean_steps=args.mean_steps,
        moe_steps=args.moe_steps,
        annotation_files=parse_csv_strings(args.annotation_files),
        data_cache_dir=args.data_cache_dir,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageQConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
