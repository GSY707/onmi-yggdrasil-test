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

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visual_multimodal_stage_ab import write_png


COLORS = ("red", "green", "blue", "yellow", "purple", "cyan")
RGB = (
    (0.92, 0.08, 0.08),
    (0.08, 0.74, 0.18),
    (0.12, 0.30, 0.94),
    (0.96, 0.78, 0.08),
    (0.62, 0.22, 0.92),
    (0.08, 0.82, 0.90),
)
SHAPES = ("circle", "square", "triangle", "diamond")
POSITIONS = ("top left", "top right", "bottom left", "bottom right", "center")
EDIT_FAMILIES = ("color_edit", "shape_edit", "position_edit")
CHARSET = string.ascii_lowercase + string.digits + " .,!?;:'\"-/()[]%$&+\n"
PAD = 0
BOS = 1
EOS = 2
CHAR_TO_ID = {char: index + 3 for index, char in enumerate(CHARSET)}
VOCAB = len(CHAR_TO_ID) + 3


@dataclass(frozen=True)
class StageAIConfig:
    train_size: int = 1024
    val_size: int = 256
    test_size: int = 384
    batch_size: int = 64
    seed: int = 20260701
    image_size: int = 64
    prompt_len: int = 96
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    latent_tokens: int = 8
    lr: float = 8e-4
    direct_steps: int = 180
    latent_steps: int = 240
    edit_steps: int = 300
    eval_every: int = 100
    attr_loss_weight: float = 0.15
    foreground_loss_weight: float = 6.0
    max_train_seconds: int = 129_600
    variants: tuple[str, ...] = ("prompt_direct", "latent_output", "latent_image_edit")


@dataclass(frozen=True)
class SceneAttrs:
    color: int
    shape: int
    position: int

    def tensor(self) -> list[int]:
        return [self.color, self.shape, self.position]


@dataclass
class StageAISet:
    examples: list[dict[str, object]]
    generation_prompt: torch.Tensor
    edit_prompt: torch.Tensor
    source_image: torch.Tensor
    target_image: torch.Tensor
    source_attrs: torch.Tensor
    target_attrs: torch.Tensor
    edit_family: torch.Tensor

    def subset(self, indices: list[int]) -> "StageAISet":
        return StageAISet(
            examples=[self.examples[index] for index in indices],
            generation_prompt=self.generation_prompt[indices],
            edit_prompt=self.edit_prompt[indices],
            source_image=self.source_image[indices],
            target_image=self.target_image[indices],
            source_attrs=self.source_attrs[indices],
            target_attrs=self.target_attrs[indices],
            edit_family=self.edit_family[indices],
        )

    def to(self, device: torch.device) -> "StageAISet":
        return StageAISet(
            examples=self.examples,
            generation_prompt=self.generation_prompt.to(device=device, dtype=torch.long),
            edit_prompt=self.edit_prompt.to(device=device, dtype=torch.long),
            source_image=self.source_image.to(device=device, dtype=torch.float32),
            target_image=self.target_image.to(device=device, dtype=torch.float32),
            source_attrs=self.source_attrs.to(device=device, dtype=torch.long),
            target_attrs=self.target_attrs.to(device=device, dtype=torch.long),
            edit_family=self.edit_family.to(device=device, dtype=torch.long),
        )


def encode_text(value: str, length: int) -> list[int]:
    ids = [BOS]
    ids.extend(CHAR_TO_ID[char] for char in value.lower() if char in CHAR_TO_ID)
    ids = ids[: length - 1] + [EOS]
    return ids + [PAD] * (length - len(ids))


def shape_mask(shape: int, object_size: int, *, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(object_size, device=device),
        torch.arange(object_size, device=device),
        indexing="ij",
    )
    if SHAPES[shape] == "circle":
        center = (object_size - 1) / 2
        return ((yy - center) ** 2 + (xx - center) ** 2) <= (object_size * 0.43) ** 2
    if SHAPES[shape] == "square":
        return torch.ones(object_size, object_size, dtype=torch.bool, device=device)
    if SHAPES[shape] == "triangle":
        return yy >= torch.abs(xx - (object_size - 1) / 2)
    center = (object_size - 1) / 2
    return (torch.abs(xx - center) + torch.abs(yy - center)) <= object_size * 0.46


def position_xy(position: int, image_size: int, object_size: int) -> tuple[int, int]:
    margin = max(5, image_size // 7)
    center = (image_size - object_size) // 2
    coords = {
        0: (margin, margin),
        1: (image_size - object_size - margin, margin),
        2: (margin, image_size - object_size - margin),
        3: (image_size - object_size - margin, image_size - object_size - margin),
        4: (center, center),
    }
    return coords[position]


def render_scene(attrs: SceneAttrs, image_size: int, *, device: torch.device) -> torch.Tensor:
    image = torch.full((3, image_size, image_size), 0.075, dtype=torch.float32, device=device)
    image[:, 4:-4, 4:-4] = 0.13
    image[:, 4:5, 4:-4] = 0.22
    image[:, -5:-4, 4:-4] = 0.22
    image[:, 4:-4, 4:5] = 0.22
    image[:, 4:-4, -5:-4] = 0.22
    object_size = max(10, image_size // 4)
    x, y = position_xy(attrs.position, image_size, object_size)
    mask = shape_mask(attrs.shape, object_size, device=device)
    color = torch.tensor(RGB[attrs.color], dtype=torch.float32, device=device).view(3, 1)
    patch = image[:, y : y + object_size, x : x + object_size]
    patch[:, mask] = color.expand(3, int(mask.sum()))
    return image


def attrs_phrase(attrs: SceneAttrs) -> str:
    return f"{COLORS[attrs.color]} {SHAPES[attrs.shape]} at {POSITIONS[attrs.position]}"


def make_scene(rng: random.Random) -> SceneAttrs:
    return SceneAttrs(
        color=rng.randrange(len(COLORS)),
        shape=rng.randrange(len(SHAPES)),
        position=rng.randrange(len(POSITIONS)),
    )


def make_edit(source: SceneAttrs, rng: random.Random) -> tuple[SceneAttrs, int, str]:
    family = rng.randrange(len(EDIT_FAMILIES))
    if EDIT_FAMILIES[family] == "color_edit":
        choices = [index for index in range(len(COLORS)) if index != source.color]
        color = rng.choice(choices)
        target = SceneAttrs(color=color, shape=source.shape, position=source.position)
        prompt = f"change color to {COLORS[color]}"
    elif EDIT_FAMILIES[family] == "shape_edit":
        choices = [index for index in range(len(SHAPES)) if index != source.shape]
        shape = rng.choice(choices)
        target = SceneAttrs(color=source.color, shape=shape, position=source.position)
        prompt = f"change shape to {SHAPES[shape]}"
    else:
        choices = [index for index in range(len(POSITIONS)) if index != source.position]
        position = rng.choice(choices)
        target = SceneAttrs(color=source.color, shape=source.shape, position=position)
        prompt = f"move object to {POSITIONS[position]}"
    return target, family, prompt


def build_split(split: str, size: int, config: StageAIConfig) -> StageAISet:
    rng = random.Random(config.seed + {"train": 11, "val": 17, "test": 23}[split])
    device = torch.device("cpu")
    examples: list[dict[str, object]] = []
    generation_prompt: list[list[int]] = []
    edit_prompt: list[list[int]] = []
    source_images: list[torch.Tensor] = []
    target_images: list[torch.Tensor] = []
    source_attrs: list[list[int]] = []
    target_attrs: list[list[int]] = []
    edit_family: list[int] = []
    for index in range(size):
        source = make_scene(rng)
        target, family, edit = make_edit(source, rng)
        generation = f"generate a {attrs_phrase(target)}"
        examples.append(
            {
                "id": f"{split}-{index:06d}",
                "generation_prompt": generation,
                "edit_prompt": edit,
                "source": attrs_phrase(source),
                "target": attrs_phrase(target),
                "edit_family": EDIT_FAMILIES[family],
            }
        )
        generation_prompt.append(encode_text(generation, config.prompt_len))
        edit_prompt.append(encode_text(edit, config.prompt_len))
        source_images.append(render_scene(source, config.image_size, device=device))
        target_images.append(render_scene(target, config.image_size, device=device))
        source_attrs.append(source.tensor())
        target_attrs.append(target.tensor())
        edit_family.append(family)
    return StageAISet(
        examples=examples,
        generation_prompt=torch.tensor(generation_prompt, dtype=torch.long),
        edit_prompt=torch.tensor(edit_prompt, dtype=torch.long),
        source_image=torch.stack(source_images),
        target_image=torch.stack(target_images),
        source_attrs=torch.tensor(source_attrs, dtype=torch.long),
        target_attrs=torch.tensor(target_attrs, dtype=torch.long),
        edit_family=torch.tensor(edit_family, dtype=torch.long),
    )


def load_data(config: StageAIConfig) -> dict[str, StageAISet]:
    return {
        "train": build_split("train", config.train_size, config),
        "val": build_split("val", config.val_size, config),
        "test": build_split("test", config.test_size, config),
    }


class QueryResampler(nn.Module):
    def __init__(self, d_model: int, heads: int, query_count: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(query_count, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attn(self.norm_q(query), self.norm_kv(source), self.norm_kv(source), need_weights=False)
        x = query + attended
        return x + self.ff(x)


class TextEncoder(nn.Module):
    def __init__(self, config: StageAIConfig) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(config.prompt_len, config.d_model) * 0.02)
        self.embedding = nn.Embedding(VOCAB, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) + self.position[: tokens.shape[1]].unsqueeze(0)
        padding = tokens == PAD
        x = self.encoder(x, src_key_padding_mask=padding)
        return self.norm(x.masked_fill(padding.unsqueeze(-1), 0.0))


class ImageEncoder(nn.Module):
    def __init__(self, config: StageAIConfig) -> None:
        super().__init__()
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image).flatten(2).transpose(1, 2)
        return x + self.position.unsqueeze(0)


class PixelDecoder(nn.Module):
    def __init__(self, config: StageAIConfig) -> None:
        super().__init__()
        self.config = config
        self.seed_size = config.image_size // 8
        self.project = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * self.seed_size * self.seed_size),
            nn.GELU(),
        )
        hidden = config.d_model
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, max(32, hidden // 2), kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(max(32, hidden // 2), 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.dim() == 3:
            context = context.mean(dim=1)
        x = self.project(context).view(context.shape[0], self.config.d_model, self.seed_size, self.seed_size)
        return self.deconv(x)


class AttrHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.color = nn.Linear(d_model, len(COLORS))
        self.shape = nn.Linear(d_model, len(SHAPES))
        self.position = nn.Linear(d_model, len(POSITIONS))

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        if context.dim() == 3:
            context = context.mean(dim=1)
        x = self.net(context)
        return {"color": self.color(x), "shape": self.shape(x), "position": self.position(x)}


class PromptDirectGenerator(nn.Module):
    def __init__(self, config: StageAIConfig) -> None:
        super().__init__()
        self.text = TextEncoder(config)
        self.decoder = PixelDecoder(config)
        self.attr = AttrHead(config.d_model)

    def forward(self, prompt: torch.Tensor) -> dict[str, object]:
        tokens = self.text(prompt)
        context = tokens.mean(dim=1)
        return {"image": self.decoder(context), "attrs": self.attr(context), "latent_tokens": tokens}


class LatentOutputGenerator(nn.Module):
    def __init__(self, config: StageAIConfig) -> None:
        super().__init__()
        self.text = TextEncoder(config)
        self.planner = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)
        self.decoder = PixelDecoder(config)
        self.attr = AttrHead(config.d_model)

    def forward(self, prompt: torch.Tensor) -> dict[str, object]:
        text_tokens = self.text(prompt)
        latent = self.planner(text_tokens)
        return {"image": self.decoder(latent), "attrs": self.attr(latent), "latent_tokens": latent}


class LatentImageEditor(nn.Module):
    def __init__(self, config: StageAIConfig) -> None:
        super().__init__()
        self.text = TextEncoder(config)
        self.image = ImageEncoder(config)
        self.modality = nn.Embedding(2, config.d_model)
        self.planner = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)
        self.decoder = PixelDecoder(config)
        self.attr = AttrHead(config.d_model)

    def forward(self, source_image: torch.Tensor, edit_prompt: torch.Tensor, *, zero_source: bool = False) -> dict[str, object]:
        image_tokens = self.image(source_image)
        if zero_source:
            image_tokens = image_tokens * 0.0
        image_tokens = image_tokens + self.modality.weight[0].view(1, 1, -1)
        text_tokens = self.text(edit_prompt) + self.modality.weight[1].view(1, 1, -1)
        latent = self.planner(torch.cat((image_tokens, text_tokens), dim=1))
        return {"image": self.decoder(latent), "attrs": self.attr(latent), "latent_tokens": latent}


def random_batch(data: StageAISet, *, rng: random.Random, batch_size: int, device: torch.device) -> StageAISet:
    indices = [rng.randrange(len(data.examples)) for _ in range(batch_size)]
    return data.subset(indices).to(device)


def attr_loss(logits: dict[str, torch.Tensor], target_attrs: torch.Tensor) -> torch.Tensor:
    return (
        F.cross_entropy(logits["color"], target_attrs[:, 0])
        + F.cross_entropy(logits["shape"], target_attrs[:, 1])
        + F.cross_entropy(logits["position"], target_attrs[:, 2])
    ) / 3.0


def foreground_weighted_mse(predicted: torch.Tensor, target: torch.Tensor, config: StageAIConfig) -> torch.Tensor:
    background = torch.tensor((0.13, 0.13, 0.13), dtype=target.dtype, device=target.device).view(1, 3, 1, 1)
    foreground = ((target - background).abs().mean(dim=1, keepdim=True) > 0.08).float()
    weight = 1.0 + config.foreground_loss_weight * foreground
    return ((predicted - target) ** 2 * weight).mean()


def output_loss(output: dict[str, object], target_image: torch.Tensor, target_attrs: torch.Tensor, config: StageAIConfig) -> torch.Tensor:
    image = output["image"]
    attrs = output["attrs"]
    if not isinstance(image, torch.Tensor) or not isinstance(attrs, dict):
        raise TypeError("model output must contain image tensor and attrs dict")
    return foreground_weighted_mse(image, target_image, config) + config.attr_loss_weight * attr_loss(attrs, target_attrs)


def template_bank(config: StageAIConfig, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    attrs: list[list[int]] = []
    images: list[torch.Tensor] = []
    for color in range(len(COLORS)):
        for shape in range(len(SHAPES)):
            for position in range(len(POSITIONS)):
                scene = SceneAttrs(color=color, shape=shape, position=position)
                attrs.append(scene.tensor())
                images.append(render_scene(scene, config.image_size, device=device))
    return torch.stack(images).view(len(images), -1), torch.tensor(attrs, dtype=torch.long, device=device)


def nearest_attrs(images: torch.Tensor, templates: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
    flat = images.detach().view(images.shape[0], -1)
    distances = torch.cdist(flat, templates)
    nearest = distances.argmin(dim=1)
    return attrs[nearest]


def attribute_head_metrics(logits: dict[str, torch.Tensor], target_attrs: torch.Tensor) -> dict[str, float]:
    color = logits["color"].argmax(dim=1)
    shape = logits["shape"].argmax(dim=1)
    position = logits["position"].argmax(dim=1)
    exact = (color == target_attrs[:, 0]) & (shape == target_attrs[:, 1]) & (position == target_attrs[:, 2])
    return {
        "head_color_acc": float((color == target_attrs[:, 0]).float().mean().detach().cpu()),
        "head_shape_acc": float((shape == target_attrs[:, 1]).float().mean().detach().cpu()),
        "head_position_acc": float((position == target_attrs[:, 2]).float().mean().detach().cpu()),
        "head_scene_exact": float(exact.float().mean().detach().cpu()),
    }


def score_images(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_attrs: torch.Tensor,
    config: StageAIConfig,
    *,
    templates: torch.Tensor,
    template_attrs: torch.Tensor,
) -> dict[str, float]:
    parsed = nearest_attrs(predicted, templates, template_attrs)
    exact = (parsed == target_attrs).all(dim=1)
    return {
        "pixel_mse": float(F.mse_loss(predicted, target).detach().cpu()),
        "pixel_mae": float(F.l1_loss(predicted, target).detach().cpu()),
        "nearest_color_acc": float((parsed[:, 0] == target_attrs[:, 0]).float().mean().detach().cpu()),
        "nearest_shape_acc": float((parsed[:, 1] == target_attrs[:, 1]).float().mean().detach().cpu()),
        "nearest_position_acc": float((parsed[:, 2] == target_attrs[:, 2]).float().mean().detach().cpu()),
        "nearest_scene_exact": float(exact.float().mean().detach().cpu()),
        "template_count": float(len(template_attrs)),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: StageAISet,
    config: StageAIConfig,
    *,
    device: torch.device,
    task: str,
    zero_source: bool = False,
) -> dict[str, object]:
    model.eval()
    templates, template_attrs = template_bank(config, device=device)
    all_scores: list[dict[str, float]] = []
    family_scores: dict[str, list[float]] = {name: [] for name in EDIT_FAMILIES}
    for start in range(0, len(data.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.examples))))).to(device)
        if task == "generate":
            output = model(batch.generation_prompt)
        elif task == "edit":
            output = model(batch.source_image, batch.edit_prompt, zero_source=zero_source)
        else:
            raise ValueError(f"unknown task: {task}")
        image = output["image"]
        attrs = output["attrs"]
        if not isinstance(image, torch.Tensor) or not isinstance(attrs, dict):
            raise TypeError("model output must contain image tensor and attrs dict")
        parsed_metrics = score_images(
            image,
            batch.target_image,
            batch.target_attrs,
            config,
            templates=templates,
            template_attrs=template_attrs,
        )
        parsed_attrs = nearest_attrs(image, templates, template_attrs)
        scene_exact = (parsed_attrs == batch.target_attrs).all(dim=1).float().detach().cpu().tolist()
        for family_index, value in zip(batch.edit_family.detach().cpu().tolist(), scene_exact):
            family_scores[EDIT_FAMILIES[int(family_index)]].append(float(value))
        all_scores.append({**parsed_metrics, **attribute_head_metrics(attrs, batch.target_attrs)})
    model.train()
    keys = sorted(all_scores[0])
    metrics = {key: statistics.fmean(score[key] for score in all_scores) for key in keys}
    metrics["per_edit_family_scene_exact"] = {
        family: statistics.fmean(values) if values else 0.0 for family, values in family_scores.items()
    }
    return metrics


@torch.no_grad()
def evaluate_source_baseline(data: StageAISet, config: StageAIConfig, *, device: torch.device) -> dict[str, object]:
    templates, template_attrs = template_bank(config, device=device)
    all_scores: list[dict[str, float]] = []
    family_scores: dict[str, list[float]] = {name: [] for name in EDIT_FAMILIES}
    for start in range(0, len(data.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.examples))))).to(device)
        parsed_metrics = score_images(
            batch.source_image,
            batch.target_image,
            batch.target_attrs,
            config,
            templates=templates,
            template_attrs=template_attrs,
        )
        parsed_attrs = nearest_attrs(batch.source_image, templates, template_attrs)
        scene_exact = (parsed_attrs == batch.target_attrs).all(dim=1).float().detach().cpu().tolist()
        for family_index, value in zip(batch.edit_family.detach().cpu().tolist(), scene_exact):
            family_scores[EDIT_FAMILIES[int(family_index)]].append(float(value))
        all_scores.append(parsed_metrics)
    keys = sorted(all_scores[0])
    metrics: dict[str, object] = {key: statistics.fmean(score[key] for score in all_scores) for key in keys}
    metrics["per_edit_family_scene_exact"] = {
        family: statistics.fmean(values) if values else 0.0 for family, values in family_scores.items()
    }
    return metrics


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def train_model(
    model: nn.Module,
    train: StageAISet,
    val: StageAISet,
    config: StageAIConfig,
    *,
    steps: int,
    task: str,
    label: str,
    seed: int,
    device: torch.device,
    deadline: float | None,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, object]] = []
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    steps_completed = 0
    stopped_by_time_budget = False
    for step in range(1, steps + 1):
        if deadline_expired(deadline):
            stopped_by_time_budget = True
            break
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        if task == "generate":
            output = model(batch.generation_prompt)
        elif task == "edit":
            output = model(batch.source_image, batch.edit_prompt)
        else:
            raise ValueError(f"unknown task: {task}")
        loss = output_loss(output, batch.target_image, batch.target_attrs, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps_completed = step
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_model(model, val, config, device=device, task=task)
            score = float(metrics["nearest_scene_exact"])
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
            history.append(
                {
                    "step": step,
                    "loss": round(float(loss.detach().cpu()), 5),
                    "pixel_mse": metrics["pixel_mse"],
                    "nearest_scene_exact": metrics["nearest_scene_exact"],
                    "head_scene_exact": metrics["head_scene_exact"],
                }
            )
            print(
                f"{label} step={step:4d} loss={float(loss.detach().cpu()):.5f} "
                f"mse={metrics['pixel_mse']:.4f} scene={metrics['nearest_scene_exact']:.3f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "train_steps_requested": steps,
        "train_steps_completed": steps_completed,
        "stopped_by_time_budget": stopped_by_time_budget,
        "trainable_parameter_count": count_parameters(model),
    }


def write_sample_outputs(
    models: dict[str, nn.Module],
    test: StageAISet,
    config: StageAIConfig,
    *,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = test.subset(list(range(min(6, len(test.examples))))).to(device)
    sample_meta: list[dict[str, object]] = []
    with torch.no_grad():
        outputs: dict[str, torch.Tensor] = {}
        if "prompt_direct" in models:
            outputs["prompt_direct"] = models["prompt_direct"](selected.generation_prompt)["image"]  # type: ignore[index]
        if "latent_output" in models:
            outputs["latent_output"] = models["latent_output"](selected.generation_prompt)["image"]  # type: ignore[index]
        if "latent_image_edit" in models:
            editor = models["latent_image_edit"]
            outputs["latent_image_edit"] = editor(selected.source_image, selected.edit_prompt)["image"]  # type: ignore[index]
            outputs["latent_image_edit_no_source"] = editor(
                selected.source_image, selected.edit_prompt, zero_source=True
            )["image"]  # type: ignore[index]
    for index, example in enumerate(selected.examples):
        item: dict[str, object] = dict(example)
        source_path = output_dir / f"{index:02d}_source.png"
        target_path = output_dir / f"{index:02d}_target.png"
        write_png(source_path, selected.source_image[index].detach().cpu())
        write_png(target_path, selected.target_image[index].detach().cpu())
        item["source_png"] = str(source_path)
        item["target_png"] = str(target_path)
        for name, images in outputs.items():
            path = output_dir / f"{index:02d}_{name}.png"
            write_png(path, images[index].detach().cpu())
            item[f"{name}_png"] = str(path)
        sample_meta.append(item)
    (output_dir / "samples.json").write_text(json.dumps(sample_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample_meta


def run_experiment(config: StageAIConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_ai_image_generation_editing device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    started = time.perf_counter()
    deadline = started + config.max_train_seconds if config.max_train_seconds > 0 else None
    data = load_data(config)
    train = data["train"].to(device)
    val = data["val"].to(device)
    test = data["test"].to(device)

    models: dict[str, nn.Module] = {}
    training: dict[str, object] = {}
    metrics: dict[str, object] = {}
    ablations: dict[str, object] = {}

    if "prompt_direct" in config.variants:
        model = PromptDirectGenerator(config).to(device)
        training["prompt_direct"] = train_model(
            model,
            train,
            val,
            config,
            steps=config.direct_steps,
            task="generate",
            label="prompt_direct",
            seed=config.seed + 101,
            device=device,
            deadline=deadline,
        )
        metrics["prompt_direct"] = evaluate_model(model, test, config, device=device, task="generate")
        models["prompt_direct"] = model
    if "latent_output" in config.variants:
        model = LatentOutputGenerator(config).to(device)
        training["latent_output"] = train_model(
            model,
            train,
            val,
            config,
            steps=config.latent_steps,
            task="generate",
            label="latent_output",
            seed=config.seed + 202,
            device=device,
            deadline=deadline,
        )
        metrics["latent_output"] = evaluate_model(model, test, config, device=device, task="generate")
        models["latent_output"] = model
    if "latent_image_edit" in config.variants:
        model = LatentImageEditor(config).to(device)
        training["latent_image_edit"] = train_model(
            model,
            train,
            val,
            config,
            steps=config.edit_steps,
            task="edit",
            label="latent_image_edit",
            seed=config.seed + 303,
            device=device,
            deadline=deadline,
        )
        metrics["latent_image_edit"] = evaluate_model(model, test, config, device=device, task="edit")
        ablations["latent_image_edit_no_source"] = evaluate_model(
            model,
            test,
            config,
            device=device,
            task="edit",
            zero_source=True,
        )
        models["latent_image_edit"] = model
    ablations["source_image_no_edit"] = evaluate_source_baseline(test, config, device=device)
    samples = write_sample_outputs(
        models,
        test,
        config,
        device=device,
        output_dir=output_path.parent / "samples" / output_path.stem / f"seed{config.seed}",
    )
    stopped_by_time_budget = any(bool(item.get("stopped_by_time_budget")) for item in training.values() if isinstance(item, dict))
    output = {
        "experiment": "omni_transformer_stage_ai_image_generation_editing",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "task": "image generation and source-image editing from latent output experts",
            "prompt_direct": "text tokens are pooled directly into a pixel decoder baseline",
            "latent_output": "text tokens are resampled into fixed latent output tokens before the pixel decoder",
            "latent_image_edit": "source image tokens and edit text tokens are resampled into latent output tokens before the pixel decoder",
            "ablation_no_source": "edit prompt keeps only the low-entropy edit command; unchanged attributes must come from the source image",
            "time_budget_seconds": config.max_train_seconds,
            "pixel_loss": "foreground-weighted MSE so small rendered objects cannot be hidden by background reconstruction",
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "data": {
            "train_size": len(train.examples),
            "val_size": len(val.examples),
            "test_size": len(test.examples),
            "colors": COLORS,
            "shapes": SHAPES,
            "positions": POSITIONS,
            "edit_families": EDIT_FAMILIES,
        },
        "training": training,
        "metrics": metrics,
        "ablations": ablations,
        "samples": samples,
        "stopped_by_time_budget": stopped_by_time_budget,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
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


def stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    buckets: dict[str, list[float]] = {}
    for run in runs:
        for key, value in collect_numbers(
            {
                "metrics": run["metrics"],
                "ablations": run["ablations"],
                "training": run["training"],
                "total_seconds": run["total_seconds"],
            }
        ).items():
            buckets.setdefault(key, []).append(value)
    stats = {key: stat(values) for key, values in sorted(buckets.items())}

    def mean(path: str) -> float | None:
        item = stats.get(path)
        return None if item is None else float(item["mean"])

    summary = {
        "prompt_direct": {
            "pixel_mse": mean("metrics.prompt_direct.pixel_mse"),
            "nearest_scene_exact": mean("metrics.prompt_direct.nearest_scene_exact"),
            "head_scene_exact": mean("metrics.prompt_direct.head_scene_exact"),
        },
        "latent_output": {
            "pixel_mse": mean("metrics.latent_output.pixel_mse"),
            "nearest_scene_exact": mean("metrics.latent_output.nearest_scene_exact"),
            "head_scene_exact": mean("metrics.latent_output.head_scene_exact"),
        },
        "latent_image_edit": {
            "pixel_mse": mean("metrics.latent_image_edit.pixel_mse"),
            "nearest_scene_exact": mean("metrics.latent_image_edit.nearest_scene_exact"),
            "head_scene_exact": mean("metrics.latent_image_edit.head_scene_exact"),
            "no_source_scene_exact": mean("ablations.latent_image_edit_no_source.nearest_scene_exact"),
            "source_no_edit_scene_exact": mean("ablations.source_image_no_edit.nearest_scene_exact"),
        },
    }
    return {
        "experiment": "omni_transformer_stage_ai_image_generation_editing_sweep",
        "run_count": len(runs),
        "seeds": sorted({int(run["config"]["seed"]) for run in runs}),  # type: ignore[index]
        "variants": sorted({variant for run in runs for variant in run["config"]["variants"]}),  # type: ignore[index]
        "summary": summary,
        "stats": stats,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_ai_image_generation_editing/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_ai_image_generation_editing/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_ai_image_generation_editing/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--variants", default="prompt_direct,latent_output,latent_image_edit")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--direct-steps", type=int, default=180)
    parser.add_argument("--latent-steps", type=int, default=240)
    parser.add_argument("--edit-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--attr-loss-weight", type=float, default=0.15)
    parser.add_argument("--foreground-loss-weight", type=float, default=6.0)
    parser.add_argument("--max-train-seconds", type=int, default=129_600)
    args = parser.parse_args()

    variants = parse_csv_strings(args.variants)
    valid = {"prompt_direct", "latent_output", "latent_image_edit"}
    unknown = set(variants) - valid
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}; choices are {sorted(valid)}")

    base_config = StageAIConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        image_size=args.image_size,
        prompt_len=args.prompt_len,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        latent_tokens=args.latent_tokens,
        direct_steps=args.direct_steps,
        latent_steps=args.latent_steps,
        edit_steps=args.edit_steps,
        eval_every=args.eval_every,
        attr_loss_weight=args.attr_loss_weight,
        foreground_loss_weight=args.foreground_loss_weight,
        max_train_seconds=args.max_train_seconds,
        variants=variants,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageAIConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
