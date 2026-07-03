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


MODEL_VARIANTS = ("serial_chain", "parallel_direct", "merged_direct")
INPUT_EXPERTS_Y = ("patch_vision", "object_vision", "spatial_vision", "count_vision", "prompt_text")
PATCH_EXPERT = 0
OBJECT_EXPERT = 1
SPATIAL_EXPERT = 2
COUNT_EXPERT = 3
PROMPT_EXPERT = 4
PROBE_STAGES = (
    "patch_tokens",
    "object_slots",
    "spatial_tokens",
    "count_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "raw_expert_concat",
    "wide_latent",
)


@dataclass(frozen=True)
class StageYConfig:
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
    train_steps: int = 260
    eval_every: int = 130
    router_loss_weight: float = 0.05
    patch_context_dropout: float = 0.0
    probe_steps: int = 70
    probe_hidden: int = 192
    variants: tuple[str, ...] = MODEL_VARIANTS


def stage_u_config(config: StageYConfig) -> stage_u.StageUConfig:
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
        fusion_tokens=config.output_tokens,
        output_tokens=config.output_tokens,
        candidate_count=config.candidate_count,
        train_steps=config.train_steps,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
        patch_context_dropout=config.patch_context_dropout,
        probe_steps=config.probe_steps,
        probe_hidden=config.probe_hidden,
        variants=("object_slot_spatial_wide_latent",),
    )


def stage_q_config(config: StageYConfig) -> stage_q.StageQConfig:
    return stage_q.StageQConfig(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        image_size=config.image_size,
        prompt_len=config.prompt_len,
        answer_len=config.answer_len,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        lr=config.lr,
        latent_tokens=config.output_tokens,
        candidate_count=config.candidate_count,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
    )


class DirectImageTokenExpert(nn.Module):
    def __init__(self, config: StageYConfig, token_count: int) -> None:
        super().__init__()
        self.stem = stage_u.ExpandedPatchVisionExpert(stage_u_config(config))
        self.resampler = stage_q.QueryResampler(config.d_model, config.heads, token_count, config.dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.resampler(self.stem(image))


class MergedVisualInputExpert(nn.Module):
    def __init__(self, config: StageYConfig) -> None:
        super().__init__()
        self.object_slots = config.object_slots
        self.spatial_tokens = config.spatial_tokens
        self.count_tokens = config.count_tokens
        total_tokens = config.object_slots + config.spatial_tokens + config.count_tokens
        self.stem = stage_u.ExpandedPatchVisionExpert(stage_u_config(config))
        self.resampler = stage_q.QueryResampler(config.d_model, config.heads, total_tokens, config.dropout)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.resampler(self.stem(image))
        object_end = self.object_slots
        spatial_end = object_end + self.spatial_tokens
        return tokens[:, :object_end], tokens[:, object_end:spatial_end], tokens[:, spatial_end:]


class LatentReasoner(nn.Module):
    def __init__(self, config: StageYConfig) -> None:
        super().__init__()
        self.output_tokens = config.output_tokens
        self.query = nn.Parameter(torch.randn(config.output_tokens, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.reasoning_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, raw: torch.Tensor, weighted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source = torch.cat((raw, weighted), dim=1)
        query = self.query.unsqueeze(0).expand(source.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((source, query), dim=1))
        reasoning = self.norm(encoded[:, -self.output_tokens :])
        return torch.cat((source, reasoning), dim=1), reasoning


class ParallelInputExpertVLM(nn.Module):
    def __init__(self, config: StageYConfig, *, variant: str) -> None:
        super().__init__()
        if variant not in {"parallel_direct", "merged_direct"}:
            raise ValueError(f"unknown parallel variant: {variant}")
        self.config = config
        self.variant = variant
        self.patch_vision = stage_u.ExpandedPatchVisionExpert(stage_u_config(config))
        self.text_encoder = stage_q.TextTokenEncoder(stage_q_config(config), length=config.prompt_len)
        self.prompt_resampler = stage_q.QueryResampler(config.d_model, config.heads, config.prompt_tokens, config.dropout)
        if variant == "parallel_direct":
            self.object_vision = DirectImageTokenExpert(config, config.object_slots)
            self.spatial_vision = DirectImageTokenExpert(config, config.spatial_tokens)
            self.count_vision = DirectImageTokenExpert(config, config.count_tokens)
            self.merged_vision = None
        else:
            self.object_vision = None
            self.spatial_vision = None
            self.count_vision = None
            self.merged_vision = MergedVisualInputExpert(config)
        self.expert_type = nn.Embedding(len(INPUT_EXPERTS_Y), config.d_model)
        self.router = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(INPUT_EXPERTS_Y)),
        )
        self.reasoner = LatentReasoner(config)
        self.scorer = stage_s.CrossAttentionAnswerScorer(stage_q_config(config))

    def route_target(self, required_experts: torch.Tensor) -> torch.Tensor:
        return required_experts[:, [PATCH_EXPERT, OBJECT_EXPERT, SPATIAL_EXPERT, COUNT_EXPERT, PROMPT_EXPERT]]

    def representations(self, batch: stage_q.PixelSet, *, zero_modalities: Iterable[str] = ()) -> dict[str, torch.Tensor]:
        zero_set = set(zero_modalities)
        image = torch.zeros_like(batch.image) if "image" in zero_set else batch.image
        prompt = torch.full_like(batch.prompt, stage_q.PAD) if "text" in zero_set else batch.prompt

        patch_tokens = self.patch_vision(image) + self.expert_type.weight[PATCH_EXPERT].view(1, 1, -1)
        prompt_source = self.text_encoder(prompt)
        prompt_tokens = self.prompt_resampler(prompt_source) + self.expert_type.weight[PROMPT_EXPERT].view(1, 1, -1)

        if self.variant == "parallel_direct":
            assert self.object_vision is not None
            assert self.spatial_vision is not None
            assert self.count_vision is not None
            object_slots = self.object_vision(image) + self.expert_type.weight[OBJECT_EXPERT].view(1, 1, -1)
            spatial_tokens = self.spatial_vision(image) + self.expert_type.weight[SPATIAL_EXPERT].view(1, 1, -1)
            count_tokens = self.count_vision(image) + self.expert_type.weight[COUNT_EXPERT].view(1, 1, -1)
        else:
            assert self.merged_vision is not None
            object_slots, spatial_tokens, count_tokens = self.merged_vision(image)
            object_slots = object_slots + self.expert_type.weight[OBJECT_EXPERT].view(1, 1, -1)
            spatial_tokens = spatial_tokens + self.expert_type.weight[SPATIAL_EXPERT].view(1, 1, -1)
            count_tokens = count_tokens + self.expert_type.weight[COUNT_EXPERT].view(1, 1, -1)

        if "patch_expert" in zero_set:
            patch_tokens = torch.zeros_like(patch_tokens)
        if "object_expert" in zero_set or "object_experts" in zero_set:
            object_slots = torch.zeros_like(object_slots)
        if "spatial_expert" in zero_set or "object_experts" in zero_set:
            spatial_tokens = torch.zeros_like(spatial_tokens)
        if "count_expert" in zero_set or "object_experts" in zero_set:
            count_tokens = torch.zeros_like(count_tokens)

        expert_tokens = [patch_tokens, object_slots, spatial_tokens, count_tokens, prompt_tokens]
        route_source = torch.cat((patch_tokens.mean(dim=1), prompt_tokens.mean(dim=1)), dim=-1)
        route_logits = self.router(route_source)
        route_weights = torch.sigmoid(route_logits)
        weighted = [tokens * route_weights[:, index].view(-1, 1, 1) for index, tokens in enumerate(expert_tokens)]
        raw = torch.cat(expert_tokens, dim=1)
        weighted_concat = torch.cat(weighted, dim=1)
        wide_latent, reasoning_tokens = self.reasoner(raw, weighted_concat)
        return {
            "patch_tokens": patch_tokens,
            "object_slots": object_slots,
            "spatial_tokens": spatial_tokens,
            "count_tokens": count_tokens,
            "prompt_tokens": prompt_tokens,
            "reasoning_tokens": reasoning_tokens,
            "raw_expert_concat": raw,
            "weighted_expert_concat": weighted_concat,
            "wide_latent": wide_latent,
            "route_logits": route_logits,
            "route_weights": route_weights,
        }

    def context(
        self,
        batch: stage_q.PixelSet,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        reps = self.representations(batch, zero_modalities=zero_modalities)
        context = torch.zeros_like(reps["wide_latent"]) if disable_latent_access else reps["wide_latent"]
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
        return {"scores": self.scorer(ctx["context"], answer_tokens), **ctx}


def build_model(config: StageYConfig, variant: str) -> nn.Module:
    if variant == "serial_chain":
        return stage_u.StageUWideVLM(stage_u_config(config), variant="object_slot_spatial_wide_latent")
    return ParallelInputExpertVLM(config, variant=variant)


def loss_for_output(output: dict[str, torch.Tensor], labels: torch.Tensor, batch: stage_q.PixelSet, model: nn.Module, config: StageYConfig) -> torch.Tensor:
    rank = F.cross_entropy(output["scores"], labels)
    route_target = model.route_target(batch.required_experts)  # type: ignore[attr-defined]
    router = F.binary_cross_entropy_with_logits(output["route_logits"], route_target)
    return rank + config.router_loss_weight * router


@torch.no_grad()
def evaluate_ranking(
    model: nn.Module,
    data: stage_q.PixelSet,
    config: StageYConfig,
    *,
    device: torch.device,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    zero_modalities: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    hits: list[float] = []
    reciprocal: list[float] = []
    by_family: dict[str, dict[str, list[float]]] = {family: {"hits": [], "mrr": []} for family in stage_u.FAMILIES}
    for start in range(0, len(data.examples), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(data.examples))))
        batch = data.subset(indices).to(device)
        rows, true_indices = stage_u.candidate_rows_for_examples(batch.examples, candidate_pool, seed=seed + start)
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        output = model(batch, answers, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        order = output["scores"].argsort(dim=1, descending=True).detach().cpu().tolist()
        for example, row_order, true_index in zip(batch.examples, order, true_indices):
            rank = row_order.index(true_index) + 1
            hit = 1.0 if rank == 1 else 0.0
            mrr = 1.0 / rank
            family = example.source_file.split("/")[-1]
            hits.append(hit)
            reciprocal.append(mrr)
            by_family[family]["hits"].append(hit)
            by_family[family]["mrr"].append(mrr)
    model.train()
    return {
        "rank_top1": statistics.fmean(hits),
        "rank_mrr": statistics.fmean(reciprocal),
        "candidate_count": config.candidate_count,
        "random_top1": 1.0 / config.candidate_count,
        "per_family": {
            family: {
                "rank_top1": statistics.fmean(values["hits"]),
                "rank_mrr": statistics.fmean(values["mrr"]),
                "example_count": len(values["hits"]),
            }
            for family, values in by_family.items()
            if values["hits"]
        },
    }


def train_model(
    model: nn.Module,
    train: stage_q.PixelSet,
    val: stage_q.PixelSet,
    config: StageYConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
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
        batch = stage_q.random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        rows, true_indices = stage_u.candidate_rows_for_examples(batch.examples, candidate_pool, seed=rng.randrange(1_000_000_000))
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        train_zero_modalities: tuple[str, ...] = ()
        if config.patch_context_dropout > 0.0 and rng.random() < config.patch_context_dropout:
            train_zero_modalities = ("patch_expert",)
        output = model(batch, answers, zero_modalities=train_zero_modalities)
        loss = loss_for_output(output, labels, batch, model, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_ranking(model, val, config, device=device, candidate_pool=candidate_pool, seed=seed + step)
            if float(metrics["rank_top1"]) > best_top1:
                best_top1 = float(metrics["rank_top1"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append(
                {
                    "step": step,
                    "loss": round(float(loss.detach().cpu()), 4),
                    "rank_top1": metrics["rank_top1"],
                    "rank_mrr": metrics["rank_mrr"],
                }
            )
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} top1={metrics['rank_top1']:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_rank_top1": best_top1,
        "training_seconds": round(time.perf_counter() - started, 3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def answer_labels(data: stage_q.PixelSet, answer_to_label: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([answer_to_label[example.answer] for example in data.examples], dtype=torch.long, device=device)


class TokenProbe(nn.Module):
    def __init__(self, config: StageYConfig, class_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.probe_hidden),
            nn.GELU(),
            nn.Linear(config.probe_hidden, class_count),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens.mean(dim=1))


def train_reconstruction_probe(
    model: nn.Module,
    stage_name: str,
    train: stage_q.PixelSet,
    test: stage_q.PixelSet,
    config: StageYConfig,
    *,
    answer_to_label: dict[str, int],
    seed: int,
    device: torch.device,
) -> dict[str, object] | None:
    model.eval()
    first_batch = train.subset(list(range(min(config.batch_size, len(train.examples))))).to(device)
    with torch.no_grad():
        if stage_name not in model.representations(first_batch):  # type: ignore[attr-defined]
            return None
    probe = TokenProbe(config, len(answer_to_label)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.probe_steps):
        batch = stage_q.random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        labels = answer_labels(batch, answer_to_label, device)
        with torch.no_grad():
            tokens = model.representations(batch)[stage_name].detach()  # type: ignore[attr-defined]
        logits = probe(tokens)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    hits: list[float] = []
    with torch.no_grad():
        for start in range(0, len(test.examples), config.batch_size):
            batch = test.subset(list(range(start, min(start + config.batch_size, len(test.examples))))).to(device)
            labels = answer_labels(batch, answer_to_label, device)
            tokens = model.representations(batch)[stage_name]  # type: ignore[attr-defined]
            pred = probe(tokens).argmax(dim=1)
            hits.extend((pred == labels).float().detach().cpu().tolist())
    return {
        "answer_exact": statistics.fmean(hits),
        "class_count": len(answer_to_label),
        "random_exact": 1.0 / len(answer_to_label),
        "training_seconds": round(time.perf_counter() - started, 3),
    }


def train_all_probes(
    model: nn.Module,
    train: stage_q.PixelSet,
    test: stage_q.PixelSet,
    config: StageYConfig,
    *,
    answer_to_label: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    probes: dict[str, object] = {}
    for index, stage_name in enumerate(PROBE_STAGES):
        result = train_reconstruction_probe(
            model,
            stage_name,
            train,
            test,
            config,
            answer_to_label=answer_to_label,
            seed=config.seed + 9000 + index * 37,
            device=device,
        )
        if result is not None:
            probes[stage_name] = result
    return probes


def prediction_samples(
    models: dict[str, nn.Module],
    test: stage_q.PixelSet,
    config: StageYConfig,
    *,
    device: torch.device,
    candidate_pool: stage_q.CandidatePool,
) -> list[dict[str, object]]:
    selected = test.subset(list(range(min(8, len(test.examples)))))
    rows, true_indices = stage_u.candidate_rows_for_examples(selected.examples, candidate_pool, seed=config.seed + 9000)
    answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
    batch = selected.to(device)
    scores = {name: model(batch, answers)["scores"].detach().cpu().tolist() for name, model in models.items()}
    output: list[dict[str, object]] = []
    for index, example in enumerate(selected.examples):
        item = {
            "id": example.id,
            "family": example.source_file.split("/")[-1],
            "prompt": example.prompt,
            "expected": example.answer,
            "true_candidate_index": true_indices[index],
            "candidates": [candidate_pool.answers[int(candidate_index)] for candidate_index in rows[index].detach().cpu().tolist()],
        }
        for name, score_rows in scores.items():
            pred_index = int(max(range(len(score_rows[index])), key=lambda idx: score_rows[index][idx]))
            item[f"{name}_pred"] = item["candidates"][pred_index]
        output.append(item)
    return output


def run_experiment(config: StageYConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_y_parallel_input_experts device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    pixel_sets, data_stats, meta = stage_u.build_data(stage_u_config(config))
    train = pixel_sets["train"].to(device)
    val = pixel_sets["val"].to(device)
    test = pixel_sets["test"].to(device)
    stage_u.stage_r.write_samples(pixel_sets["test"], meta["test"], output_path.parent / "samples")
    candidate_pool = stage_u.build_candidate_pool(stage_u_config(config), device=device)
    answer_to_label = {answer: index for index, answer in enumerate(candidate_pool.answers)}

    models = {variant: build_model(config, variant) for variant in config.variants}
    training = {}
    for index, (variant, model) in enumerate(models.items()):
        training[variant] = train_model(
            model,
            train,
            val,
            config,
            candidate_pool=candidate_pool,
            steps=config.train_steps,
            seed=config.seed + 1100 + index * 97,
            device=device,
            label=variant,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = {
        variant: evaluate_ranking(
            model,
            test,
            config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 3000 + index,
        )
        for index, (variant, model) in enumerate(models.items())
    }
    ablations: dict[str, object] = {}
    for index, (variant, model) in enumerate(models.items()):
        for ablation_name, zero_modalities, disable_latent in (
            ("no_image_modality", ("image",), False),
            ("no_latent_access", (), True),
            ("no_patch_expert", ("patch_expert",), False),
            ("no_object_expert", ("object_expert",), False),
            ("no_spatial_expert", ("spatial_expert",), False),
            ("no_count_expert", ("count_expert",), False),
            ("no_object_spatial_count_experts", ("object_experts",), False),
        ):
            ablations[f"{variant}_{ablation_name}"] = evaluate_ranking(
                model,
                test,
                config,
                device=device,
                candidate_pool=candidate_pool,
                seed=config.seed + 4000 + index * 97,
                zero_modalities=zero_modalities,
                disable_latent_access=disable_latent,
            )

    reconstruction = {
        variant: train_all_probes(
            model,
            train,
            test,
            config,
            answer_to_label=answer_to_label,
            device=device,
        )
        for variant, model in models.items()
    }
    prediction_cost = {
        f"{variant}_prediction": stage_q.measure_prediction_cost(
            lambda model=model: evaluate_ranking(model, test, config, device=device, candidate_pool=candidate_pool, seed=config.seed + 8000),
            example_count=len(test.examples),
            device=device,
        )
        for variant, model in models.items()
    }
    samples = prediction_samples(models, test, config, device=device, candidate_pool=candidate_pool)
    output = {
        "experiment": "omni_transformer_stage_y_parallel_input_experts",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "serial_chain": "Stage U legacy comparator: patch tokens feed object/spatial, object+spatial+prompt feed count, then fusion reads every previous token.",
            "parallel_direct": "corrected input-specialist design: patch/object/spatial/count visual experts each read the external image directly; prompt expert reads text; latent reasoner aggregates after input conversion.",
            "merged_direct": "corrected merged-input design: one wider visual input expert reads the external image directly and emits object/spatial/count token slices; prompt expert reads text; latent reasoner aggregates after input conversion.",
            "input_experts": INPUT_EXPERTS_Y,
            "runtime_categories": ("external_input_to_latent", "latent_reasoning", "latent_output"),
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
        "reconstruction": reconstruction,
        "training": training,
        "prediction_cost": prediction_cost,
        "prediction_samples": samples,
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
                "training": run["training"],
                "prediction_cost": run["prediction_cost"],
                "reconstruction": run["reconstruction"],
                "total_seconds": run["total_seconds"],
            }
        ).items():
            buckets.setdefault(key, []).append(value)
    stats = {key: stage_m.stat(values) for key, values in sorted(buckets.items())}

    def mean(path: str) -> float | None:
        item = stats.get(path)
        return None if item is None else float(item["mean"])

    summary: dict[str, object] = {}
    for variant in MODEL_VARIANTS:
        if variant not in set().union(*(set(run["metrics"].keys()) for run in runs)):
            continue
        summary[variant] = {
            "rank_top1": mean(f"metrics.{variant}.rank_top1"),
            "rank_mrr": mean(f"metrics.{variant}.rank_mrr"),
            "no_image_top1": mean(f"ablations.{variant}_no_image_modality.rank_top1"),
            "no_patch_expert_top1": mean(f"ablations.{variant}_no_patch_expert.rank_top1"),
            "no_object_spatial_count_top1": mean(f"ablations.{variant}_no_object_spatial_count_experts.rank_top1"),
            "wide_latent_probe": mean(f"reconstruction.{variant}.wide_latent.answer_exact"),
            "raw_expert_probe": mean(f"reconstruction.{variant}.raw_expert_concat.answer_exact"),
            "train_seconds": mean(f"training.{variant}.training_seconds"),
            "params": mean(f"training.{variant}.trainable_parameter_count"),
            "prediction_ms_per_example": mean(f"prediction_cost.{variant}_prediction.prediction_ms_per_example"),
            "per_family": {
                family: mean(f"metrics.{variant}.per_family.{family}.rank_top1")
                for family in stage_u.FAMILIES
            },
        }
    return {
        "experiment": "omni_transformer_stage_y_parallel_input_experts_sweep",
        "run_count": len(runs),
        "seeds": sorted({int(run["config"]["seed"]) for run in runs}),
        "variants": sorted({variant for run in runs for variant in run["metrics"].keys()}),
        "stats": stats,
        "summary": summary,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_y_parallel_input_experts/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_y_parallel_input_experts/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_y_parallel_input_experts/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702")
    parser.add_argument("--variants", default="serial_chain,parallel_direct,merged_direct")
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
    parser.add_argument("--train-steps", type=int, default=260)
    parser.add_argument("--eval-every", type=int, default=130)
    parser.add_argument("--patch-context-dropout", type=float, default=0.0)
    parser.add_argument("--probe-steps", type=int, default=70)
    args = parser.parse_args()

    variants = parse_csv_strings(args.variants)
    unknown = set(variants) - set(MODEL_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    base_config = StageYConfig(
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
        patch_context_dropout=args.patch_context_dropout,
        probe_steps=args.probe_steps,
        variants=variants,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageYConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
