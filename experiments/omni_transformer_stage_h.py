from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import time
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from multimodal_fusion_latent_flow import (
    ACTION_COUNT,
    ACTION_NAMES,
    GOALS,
    GOAL_NAMES,
    IMAGE_SIZE,
    TELEMETRY_CHANNELS,
    TELEMETRY_NAMES,
    TELEMETRY_STATES,
    TEXT_VOCAB,
    VISUAL_NAMES,
    VISUAL_STATES,
    Batch,
    DiagnosticExample,
    generate_examples,
    make_batch,
    stat,
    write_sample_panel_grid,
)


PATCH_SIZE = 8
PATCH_COUNT = (IMAGE_SIZE // PATCH_SIZE) ** 2
ANSWER_LEN = 5

ACTION_BASE = 0
VISUAL_BASE = ACTION_BASE + ACTION_COUNT
GOAL_BASE = VISUAL_BASE + VISUAL_STATES
TELEMETRY_BASE = GOAL_BASE + GOALS
EOS_TOKEN = TELEMETRY_BASE + TELEMETRY_STATES
ANSWER_VOCAB = EOS_TOKEN + 1

SPECIAL_BOS = 0
SPECIAL_IMG = 1
SPECIAL_TEXT = 2
SPECIAL_TELEMETRY = 3
SPECIAL_ANSWER = 4
SPECIAL_COUNT = 5

MODALITIES = ("image", "text", "telemetry")


@dataclass(frozen=True)
class OmniConfig:
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 2048
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 160
    layers: int = 4
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    direct_steps: int = 800
    bottleneck_steps: int = 1000
    probe_steps: int = 250
    latent_tokens: int = 8
    eval_every: int = 400


@dataclass(frozen=True)
class SequenceLayout:
    input_end: int
    latent_start: int
    latent_end: int
    answer_start: int
    seq_len: int


def answer_targets(batch: Batch) -> torch.Tensor:
    eos = torch.full_like(batch.action, EOS_TOKEN)
    return torch.stack(
        (
            ACTION_BASE + batch.action,
            VISUAL_BASE + batch.visual,
            GOAL_BASE + batch.goal,
            TELEMETRY_BASE + batch.telemetry,
            eos,
        ),
        dim=1,
    )


def answer_inputs_from_targets(targets: torch.Tensor) -> torch.Tensor:
    start = torch.full((targets.shape[0], 1), -1, dtype=torch.long, device=targets.device)
    return torch.cat((start, targets[:, :-1]), dim=1)


def token_to_text(token: int) -> str:
    if ACTION_BASE <= token < VISUAL_BASE:
        return f"ACTION_{ACTION_NAMES[token - ACTION_BASE]}"
    if VISUAL_BASE <= token < GOAL_BASE:
        return f"VISUAL_{VISUAL_NAMES[token - VISUAL_BASE]}"
    if GOAL_BASE <= token < TELEMETRY_BASE:
        return f"GOAL_{GOAL_NAMES[token - GOAL_BASE]}"
    if TELEMETRY_BASE <= token < EOS_TOKEN:
        return f"TELEMETRY_{TELEMETRY_NAMES[token - TELEMETRY_BASE]}"
    if token == EOS_TOKEN:
        return "EOS"
    return f"INVALID_{token}"


def answer_to_text(tokens: Iterable[int]) -> str:
    return " ".join(token_to_text(int(token)) for token in tokens)


def format_valid(tokens: torch.Tensor) -> torch.Tensor:
    return (
        (tokens[:, 0] >= ACTION_BASE)
        & (tokens[:, 0] < VISUAL_BASE)
        & (tokens[:, 1] >= VISUAL_BASE)
        & (tokens[:, 1] < GOAL_BASE)
        & (tokens[:, 2] >= GOAL_BASE)
        & (tokens[:, 2] < TELEMETRY_BASE)
        & (tokens[:, 3] >= TELEMETRY_BASE)
        & (tokens[:, 3] < EOS_TOKEN)
        & (tokens[:, 4] == EOS_TOKEN)
    )


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attended, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class TinyOmniTransformer(nn.Module):
    def __init__(self, config: OmniConfig) -> None:
        super().__init__()
        self.config = config
        self.image_patch = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, config.d_model)
        self.text_embedding = nn.Embedding(TEXT_VOCAB, config.d_model)
        self.telemetry_projection = nn.Linear(TELEMETRY_CHANNELS, config.d_model)
        self.answer_embedding = nn.Embedding(ANSWER_VOCAB, config.d_model)
        self.special_embedding = nn.Embedding(SPECIAL_COUNT, config.d_model)
        self.latent_embedding = nn.Parameter(torch.randn(max(config.latent_tokens, 1), config.d_model) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(96 + config.latent_tokens, config.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(config.layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, ANSWER_VOCAB)

    def layout(self, latent_tokens: int) -> SequenceLayout:
        input_end = 1 + 1 + PATCH_COUNT + 1 + 5 + 1 + 12
        latent_start = input_end
        latent_end = latent_start + latent_tokens
        answer_start = latent_end
        seq_len = answer_start + ANSWER_LEN
        return SequenceLayout(input_end, latent_start, latent_end, answer_start, seq_len)

    def patch_image(self, image: torch.Tensor) -> torch.Tensor:
        patches = image.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(image.shape[0], PATCH_COUNT, 3 * PATCH_SIZE * PATCH_SIZE)

    def build_embeddings(
        self,
        batch: Batch,
        *,
        answer_inputs: torch.Tensor,
        latent_tokens: int,
        zero_modalities: Iterable[str] = (),
        shuffle_modalities: Iterable[str] = (),
    ) -> tuple[torch.Tensor, SequenceLayout]:
        zero_set = set(zero_modalities)
        shuffle_set = set(shuffle_modalities)
        batch_size = int(batch.action.shape[0])
        layout = self.layout(latent_tokens)

        image = batch.image
        text = batch.text
        telemetry = batch.telemetry_values
        if "image" in shuffle_set and batch_size > 1:
            image = torch.roll(image, shifts=1, dims=0)
        if "text" in shuffle_set and batch_size > 1:
            text = torch.roll(text, shifts=1, dims=0)
        if "telemetry" in shuffle_set and batch_size > 1:
            telemetry = torch.roll(telemetry, shifts=1, dims=0)

        image_tokens = self.image_patch(self.patch_image(image))
        text_tokens = self.text_embedding(text)
        telemetry_tokens = self.telemetry_projection(telemetry)
        if "image" in zero_set:
            image_tokens = torch.zeros_like(image_tokens)
        if "text" in zero_set:
            text_tokens = torch.zeros_like(text_tokens)
        if "telemetry" in zero_set:
            telemetry_tokens = torch.zeros_like(telemetry_tokens)

        special = self.special_embedding.weight
        bos = special[SPECIAL_BOS].expand(batch_size, 1, -1)
        img_sep = special[SPECIAL_IMG].expand(batch_size, 1, -1)
        text_sep = special[SPECIAL_TEXT].expand(batch_size, 1, -1)
        telemetry_sep = special[SPECIAL_TELEMETRY].expand(batch_size, 1, -1)
        if "image" in zero_set:
            img_sep = torch.zeros_like(img_sep)
        if "text" in zero_set:
            text_sep = torch.zeros_like(text_sep)
        if "telemetry" in zero_set:
            telemetry_sep = torch.zeros_like(telemetry_sep)

        chunks = [bos, img_sep, image_tokens, text_sep, text_tokens, telemetry_sep, telemetry_tokens]
        if latent_tokens:
            latents = self.latent_embedding[:latent_tokens].expand(batch_size, latent_tokens, -1)
            chunks.append(latents)

        answer_start = special[SPECIAL_ANSWER].expand(batch_size, 1, -1)
        previous_answer = self.answer_embedding(answer_inputs[:, 1:].clamp_min(0))
        chunks.append(torch.cat((answer_start, previous_answer), dim=1))

        x = torch.cat(chunks, dim=1)
        x = x + self.position_embedding[: x.shape[1]].unsqueeze(0)
        return x, layout

    def attention_mask(
        self,
        layout: SequenceLayout,
        *,
        mode: str,
        disable_latent_to_answer: bool = False,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.triu(torch.ones(layout.seq_len, layout.seq_len, dtype=torch.bool, device=device), diagonal=1)
        if mode == "latent_bottleneck":
            mask[layout.answer_start :, : layout.input_end] = True
            if disable_latent_to_answer:
                mask[layout.answer_start :, layout.latent_start : layout.latent_end] = True
        elif mode != "direct":
            raise ValueError(f"unknown mode: {mode}")
        return mask

    def forward(
        self,
        batch: Batch,
        *,
        answer_inputs: torch.Tensor,
        latent_tokens: int,
        mode: str,
        zero_modalities: Iterable[str] = (),
        shuffle_modalities: Iterable[str] = (),
        disable_latent_to_answer: bool = False,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor | SequenceLayout]:
        x, layout = self.build_embeddings(
            batch,
            answer_inputs=answer_inputs,
            latent_tokens=latent_tokens,
            zero_modalities=zero_modalities,
            shuffle_modalities=shuffle_modalities,
        )
        mask = self.attention_mask(
            layout,
            mode=mode,
            disable_latent_to_answer=disable_latent_to_answer,
            device=x.device,
        )
        for block in self.blocks:
            x = block(x, mask)
        hidden = self.final_norm(x)
        answer_hidden = hidden[:, layout.answer_start : layout.answer_start + ANSWER_LEN]
        output: dict[str, torch.Tensor | SequenceLayout] = {"logits": self.output_head(answer_hidden), "layout": layout}
        if return_hidden:
            output["hidden"] = hidden
        return output


def random_batch(examples: list[DiagnosticExample], *, rng: random.Random, batch_size: int, device: torch.device) -> Batch:
    return make_batch(rng.choices(examples, k=batch_size), device=device)


def answer_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, ANSWER_VOCAB), targets.reshape(-1))


@torch.no_grad()
def greedy_generate(
    model: TinyOmniTransformer,
    batch: Batch,
    *,
    latent_tokens: int,
    mode: str,
    zero_modalities: Iterable[str] = (),
    shuffle_modalities: Iterable[str] = (),
    disable_latent_to_answer: bool = False,
) -> torch.Tensor:
    model.eval()
    generated = torch.zeros((batch.action.shape[0], ANSWER_LEN), dtype=torch.long, device=batch.action.device)
    answer_inputs = torch.full_like(generated, -1)
    for index in range(ANSWER_LEN):
        if index > 0:
            answer_inputs[:, index] = generated[:, index - 1]
        output = model(
            batch,
            answer_inputs=answer_inputs,
            latent_tokens=latent_tokens,
            mode=mode,
            zero_modalities=zero_modalities,
            shuffle_modalities=shuffle_modalities,
            disable_latent_to_answer=disable_latent_to_answer,
        )
        logits = output["logits"]
        generated[:, index] = logits[:, index].argmax(dim=-1)
    model.train()
    return generated


@torch.no_grad()
def evaluate_omni(
    model: TinyOmniTransformer,
    examples: list[DiagnosticExample],
    config: OmniConfig,
    *,
    device: torch.device,
    latent_tokens: int,
    mode: str,
    zero_modalities: Iterable[str] = (),
    shuffle_modalities: Iterable[str] = (),
    disable_latent_to_answer: bool = False,
) -> dict[str, float]:
    model.eval()
    totals = {
        "answer_exact": 0.0,
        "action_exact": 0.0,
        "visual_exact": 0.0,
        "goal_exact": 0.0,
        "telemetry_exact": 0.0,
        "format_valid": 0.0,
    }
    count = 0
    for start in range(0, len(examples), config.batch_size):
        batch = make_batch(examples[start : start + config.batch_size], device=device)
        targets = answer_targets(batch)
        predicted = greedy_generate(
            model,
            batch,
            latent_tokens=latent_tokens,
            mode=mode,
            zero_modalities=zero_modalities,
            shuffle_modalities=shuffle_modalities,
            disable_latent_to_answer=disable_latent_to_answer,
        )
        totals["answer_exact"] += float((predicted == targets).all(dim=1).sum().detach().cpu())
        totals["action_exact"] += float((predicted[:, 0] == targets[:, 0]).sum().detach().cpu())
        totals["visual_exact"] += float((predicted[:, 1] == targets[:, 1]).sum().detach().cpu())
        totals["goal_exact"] += float((predicted[:, 2] == targets[:, 2]).sum().detach().cpu())
        totals["telemetry_exact"] += float((predicted[:, 3] == targets[:, 3]).sum().detach().cpu())
        totals["format_valid"] += float(format_valid(predicted).sum().detach().cpu())
        count += len(batch.action)
    model.train()
    return {name: value / count for name, value in totals.items()}


def train_omni_model(
    model: TinyOmniTransformer,
    train: list[DiagnosticExample],
    val: list[DiagnosticExample],
    config: OmniConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    latent_tokens: int,
    mode: str,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        targets = answer_targets(batch)
        answer_inputs = answer_inputs_from_targets(targets)
        output = model(batch, answer_inputs=answer_inputs, latent_tokens=latent_tokens, mode=mode)
        loss = answer_loss(output["logits"], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_omni(model, val, config, device=device, latent_tokens=latent_tokens, mode=mode)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"answer={metrics['answer_exact']:.3f} action={metrics['action_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


class LatentProbe(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.visual = nn.Linear(d_model, VISUAL_STATES)
        self.goal = nn.Linear(d_model, GOALS)
        self.telemetry = nn.Linear(d_model, TELEMETRY_STATES)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "visual": self.visual(latent),
            "goal": self.goal(latent),
            "telemetry": self.telemetry(latent),
        }


def train_latent_probe(
    model: TinyOmniTransformer,
    train: list[DiagnosticExample],
    test: list[DiagnosticExample],
    config: OmniConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = LatentProbe(config.d_model).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.0)
    rng = random.Random(config.seed + 90000)
    started = time.perf_counter()
    for _step in range(1, config.probe_steps + 1):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        targets = answer_targets(batch)
        answer_inputs = answer_inputs_from_targets(targets)
        with torch.no_grad():
            output = model(
                batch,
                answer_inputs=answer_inputs,
                latent_tokens=config.latent_tokens,
                mode="latent_bottleneck",
                return_hidden=True,
            )
            layout = output["layout"]
            hidden = output["hidden"]
            latent = hidden[:, layout.latent_start : layout.latent_end].mean(dim=1)
        logits = probe(latent.detach())
        loss = (
            F.cross_entropy(logits["visual"], batch.visual)
            + F.cross_entropy(logits["goal"], batch.goal)
            + F.cross_entropy(logits["telemetry"], batch.telemetry)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    totals = {"visual": 0.0, "goal": 0.0, "telemetry": 0.0}
    count = 0
    with torch.no_grad():
        for start in range(0, len(test), config.batch_size):
            batch = make_batch(test[start : start + config.batch_size], device=device)
            targets = answer_targets(batch)
            answer_inputs = answer_inputs_from_targets(targets)
            output = model(
                batch,
                answer_inputs=answer_inputs,
                latent_tokens=config.latent_tokens,
                mode="latent_bottleneck",
                return_hidden=True,
            )
            layout = output["layout"]
            hidden = output["hidden"]
            latent = hidden[:, layout.latent_start : layout.latent_end].mean(dim=1)
            logits = probe(latent)
            totals["visual"] += float((logits["visual"].argmax(dim=-1) == batch.visual).sum().detach().cpu())
            totals["goal"] += float((logits["goal"].argmax(dim=-1) == batch.goal).sum().detach().cpu())
            totals["telemetry"] += float((logits["telemetry"].argmax(dim=-1) == batch.telemetry).sum().detach().cpu())
            count += len(batch.action)
    return {
        "training_seconds": round(time.perf_counter() - started, 3),
        "accuracies": {name: value / count for name, value in totals.items()},
    }


def sample_outputs(
    model: TinyOmniTransformer,
    examples: list[DiagnosticExample],
    config: OmniConfig,
    *,
    device: torch.device,
    latent_tokens: int,
    mode: str,
) -> list[dict[str, str]]:
    batch = make_batch(examples[:8], device=device)
    targets = answer_targets(batch)
    predicted = greedy_generate(model, batch, latent_tokens=latent_tokens, mode=mode)
    samples = []
    for idx in range(min(8, len(examples))):
        samples.append(
            {
                "target": answer_to_text(targets[idx].detach().cpu().tolist()),
                "prediction": answer_to_text(predicted[idx].detach().cpu().tolist()),
            }
        )
    return samples


def run_experiment(config: OmniConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_h device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train = generate_examples(config.train_size, seed=config.seed + 1)
    val = generate_examples(config.val_size, seed=config.seed + 2)
    test = generate_examples(config.test_size, seed=config.seed + 3)
    write_sample_panel_grid(test, output_dir=output_path.parent / "samples" / f"seed{config.seed}", device=device)

    direct = TinyOmniTransformer(config)
    bottleneck = TinyOmniTransformer(config)
    training = {
        "h1_direct": train_omni_model(
            direct,
            train,
            val,
            config,
            steps=config.direct_steps,
            seed=config.seed + 10,
            device=device,
            latent_tokens=0,
            mode="direct",
            label="h1_direct",
        ),
        "h2_latent_bottleneck": train_omni_model(
            bottleneck,
            train,
            val,
            config,
            steps=config.bottleneck_steps,
            seed=config.seed + 20,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
            label="h2_latent_bottleneck",
        ),
    }

    metrics = {
        "h1_direct": evaluate_omni(direct, test, config, device=device, latent_tokens=0, mode="direct"),
        "h2_latent_bottleneck": evaluate_omni(
            bottleneck,
            test,
            config,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
        ),
    }

    ablations = {
        "h1_zero_image": evaluate_omni(direct, test, config, device=device, latent_tokens=0, mode="direct", zero_modalities=("image",)),
        "h1_zero_text": evaluate_omni(direct, test, config, device=device, latent_tokens=0, mode="direct", zero_modalities=("text",)),
        "h1_zero_telemetry": evaluate_omni(
            direct, test, config, device=device, latent_tokens=0, mode="direct", zero_modalities=("telemetry",)
        ),
        "h2_no_latent_access": evaluate_omni(
            bottleneck,
            test,
            config,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
            disable_latent_to_answer=True,
        ),
        "h2_zero_image": evaluate_omni(
            bottleneck,
            test,
            config,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
            zero_modalities=("image",),
        ),
        "h2_zero_text": evaluate_omni(
            bottleneck,
            test,
            config,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
            zero_modalities=("text",),
        ),
        "h2_zero_telemetry": evaluate_omni(
            bottleneck,
            test,
            config,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
            zero_modalities=("telemetry",),
        ),
        "h2_shuffle_image": evaluate_omni(
            bottleneck,
            test,
            config,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
            shuffle_modalities=("image",),
        ),
        "h2_shuffle_text": evaluate_omni(
            bottleneck,
            test,
            config,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
            shuffle_modalities=("text",),
        ),
        "h2_shuffle_telemetry": evaluate_omni(
            bottleneck,
            test,
            config,
            device=device,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
            shuffle_modalities=("telemetry",),
        ),
    }
    latent_probe = train_latent_probe(bottleneck, train, test, config, device=device)

    output = {
        "experiment": "omni_transformer_stage_h_h1_h2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "metrics": metrics,
        "ablations": ablations,
        "latent_probe": latent_probe,
        "training": training,
        "sample_outputs": {
            "h1_direct": sample_outputs(direct, test, config, device=device, latent_tokens=0, mode="direct"),
            "h2_latent_bottleneck": sample_outputs(
                bottleneck,
                test,
                config,
                device=device,
                latent_tokens=config.latent_tokens,
                mode="latent_bottleneck",
            ),
        },
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
            {"metrics": run["metrics"], "ablations": run["ablations"], "latent_probe": run["latent_probe"]}
        ).items():
            if "training_seconds" not in key:
                buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_h_h1_h2_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_h/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_h/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_h/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--direct-steps", type=int, default=800)
    parser.add_argument("--bottleneck-steps", type=int, default=1000)
    parser.add_argument("--probe-steps", type=int, default=250)
    parser.add_argument("--latent-tokens", type=int, default=8)
    args = parser.parse_args()

    base_config = OmniConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        direct_steps=args.direct_steps,
        bottleneck_steps=args.bottleneck_steps,
        probe_steps=args.probe_steps,
        latent_tokens=args.latent_tokens,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = OmniConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
