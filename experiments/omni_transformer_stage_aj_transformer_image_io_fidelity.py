from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import string
import sys
import time

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
class StageAJConfig:
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    seed: int = 20260701
    image_size: int = 64
    patch_size: int = 8
    prompt_len: int = 96
    d_model: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 3
    heads: int = 4
    dropout: float = 0.0
    latent_tokens: int = 16
    latent_levels: int = 4
    lr: float = 8e-4
    copy_steps: int = 1200
    edit_steps: int = 1800
    generate_steps: int = 1200
    eval_every: int = 300
    foreground_loss_weight: float = 8.0
    background_loss_weight: float = 2.0
    prefix_loss_weight: float = 0.25
    residual_token_loss_weight: float = 0.02
    object_aux_attr_loss_weight: float = 1.0
    object_aux_mask_loss_weight: float = 0.5
    object_aux_mask_positive_weight: float = 8.0
    train_eval_size: int = 128
    template_chunk_size: int = 30
    max_train_seconds: int = 129_600
    variants: tuple[str, ...] = ("memory_tree_supervised_copy",)


@dataclass(frozen=True)
class SceneAttrs:
    color: int
    shape: int
    position: int

    def tensor(self) -> list[int]:
        return [self.color, self.shape, self.position]


@dataclass(frozen=True)
class BackgroundSpec:
    style: int
    phase_x: int
    phase_y: int


@dataclass
class StageAJSet:
    examples: list[dict[str, object]]
    edit_prompt: torch.Tensor
    source_prompt: torch.Tensor
    target_prompt: torch.Tensor
    background_image: torch.Tensor
    canonical_background_image: torch.Tensor
    source_image: torch.Tensor
    target_image: torch.Tensor
    target_canonical_image: torch.Tensor
    source_mask: torch.Tensor
    target_mask: torch.Tensor
    source_attrs: torch.Tensor
    target_attrs: torch.Tensor
    edit_family: torch.Tensor

    def subset(self, indices: list[int]) -> "StageAJSet":
        return StageAJSet(
            examples=[self.examples[index] for index in indices],
            edit_prompt=self.edit_prompt[indices],
            source_prompt=self.source_prompt[indices],
            target_prompt=self.target_prompt[indices],
            background_image=self.background_image[indices],
            canonical_background_image=self.canonical_background_image[indices],
            source_image=self.source_image[indices],
            target_image=self.target_image[indices],
            target_canonical_image=self.target_canonical_image[indices],
            source_mask=self.source_mask[indices],
            target_mask=self.target_mask[indices],
            source_attrs=self.source_attrs[indices],
            target_attrs=self.target_attrs[indices],
            edit_family=self.edit_family[indices],
        )

    def to(self, device: torch.device) -> "StageAJSet":
        return StageAJSet(
            examples=self.examples,
            edit_prompt=self.edit_prompt.to(device=device, dtype=torch.long),
            source_prompt=self.source_prompt.to(device=device, dtype=torch.long),
            target_prompt=self.target_prompt.to(device=device, dtype=torch.long),
            background_image=self.background_image.to(device=device, dtype=torch.float32),
            canonical_background_image=self.canonical_background_image.to(device=device, dtype=torch.float32),
            source_image=self.source_image.to(device=device, dtype=torch.float32),
            target_image=self.target_image.to(device=device, dtype=torch.float32),
            target_canonical_image=self.target_canonical_image.to(device=device, dtype=torch.float32),
            source_mask=self.source_mask.to(device=device, dtype=torch.float32),
            target_mask=self.target_mask.to(device=device, dtype=torch.float32),
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


def render_background(spec: BackgroundSpec, image_size: int, *, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(image_size, device=device),
        torch.arange(image_size, device=device),
        indexing="ij",
    )
    x = xx + spec.phase_x
    y = yy + spec.phase_y
    if spec.style == 0:
        texture = ((x // 4 + y // 4) % 2).float()
    elif spec.style == 1:
        texture = ((x // 3) % 2).float()
    elif spec.style == 2:
        texture = ((y // 5) % 2).float()
    elif spec.style == 3:
        texture = (((x + y) // 5) % 2).float()
    elif spec.style == 4:
        texture = (((x - y).abs() // 5) % 2).float()
    elif spec.style == 5:
        texture = (((x * 3 + y * 5) % 17) / 16.0).float()
    elif spec.style == 6:
        texture = ((((x // 6) ^ (y // 6)) % 2)).float()
    else:
        center = image_size / 2.0
        texture = ((((x - center) ** 2 + (y - center) ** 2).sqrt() // 5) % 2).float()
    image = torch.stack(
        (
            0.08 + 0.050 * texture,
            0.10 + 0.035 * (1.0 - texture),
            0.12 + 0.030 * texture,
        ),
        dim=0,
    )
    image[:, 3:-3, 3:-3] = image[:, 3:-3, 3:-3].clamp(0.0, 1.0)
    image[:, :3, :] = 0.22
    image[:, -3:, :] = 0.22
    image[:, :, :3] = 0.22
    image[:, :, -3:] = 0.22
    return image.clamp(0.0, 1.0)


def render_on_background(background: torch.Tensor, attrs: SceneAttrs, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = background.clone()
    mask_full = torch.zeros((1, image_size, image_size), dtype=torch.float32, device=background.device)
    object_size = max(10, image_size // 4)
    x, y = position_xy(attrs.position, image_size, object_size)
    mask = shape_mask(attrs.shape, object_size, device=background.device)
    color = torch.tensor(RGB[attrs.color], dtype=torch.float32, device=background.device).view(3, 1)
    patch = image[:, y : y + object_size, x : x + object_size]
    patch[:, mask] = color.expand(3, int(mask.sum()))
    mask_full[:, y : y + object_size, x : x + object_size][:, mask] = 1.0
    return image.clamp(0.0, 1.0), mask_full


def attrs_phrase(attrs: SceneAttrs) -> str:
    return f"{COLORS[attrs.color]} {SHAPES[attrs.shape]} at {POSITIONS[attrs.position]}"


def make_scene(rng: random.Random) -> SceneAttrs:
    return SceneAttrs(
        color=rng.randrange(len(COLORS)),
        shape=rng.randrange(len(SHAPES)),
        position=rng.randrange(len(POSITIONS)),
    )


def make_background(rng: random.Random) -> BackgroundSpec:
    return BackgroundSpec(style=rng.randrange(8), phase_x=rng.randrange(64), phase_y=rng.randrange(64))


def make_edit(source: SceneAttrs, rng: random.Random) -> tuple[SceneAttrs, int, str]:
    family = rng.randrange(len(EDIT_FAMILIES))
    if EDIT_FAMILIES[family] == "color_edit":
        color = rng.choice([index for index in range(len(COLORS)) if index != source.color])
        target = SceneAttrs(color=color, shape=source.shape, position=source.position)
        prompt = f"change color to {COLORS[color]}"
    elif EDIT_FAMILIES[family] == "shape_edit":
        shape = rng.choice([index for index in range(len(SHAPES)) if index != source.shape])
        target = SceneAttrs(color=source.color, shape=shape, position=source.position)
        prompt = f"change shape to {SHAPES[shape]}"
    else:
        position = rng.choice([index for index in range(len(POSITIONS)) if index != source.position])
        target = SceneAttrs(color=source.color, shape=source.shape, position=position)
        prompt = f"move object to {POSITIONS[position]}"
    return target, family, prompt


def build_split(split: str, size: int, config: StageAJConfig) -> StageAJSet:
    rng = random.Random(config.seed + {"train": 31, "val": 37, "test": 41}[split])
    device = torch.device("cpu")
    examples: list[dict[str, object]] = []
    prompts: list[list[int]] = []
    source_prompts: list[list[int]] = []
    target_prompts: list[list[int]] = []
    backgrounds: list[torch.Tensor] = []
    canonical_backgrounds: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_canonical_images: list[torch.Tensor] = []
    source_masks: list[torch.Tensor] = []
    target_masks: list[torch.Tensor] = []
    source_attrs: list[list[int]] = []
    target_attrs: list[list[int]] = []
    edit_families: list[int] = []
    for index in range(size):
        bg_spec = make_background(rng)
        source = make_scene(rng)
        target, family, edit = make_edit(source, rng)
        background = render_background(bg_spec, config.image_size, device=device)
        canonical_background = render_background(BackgroundSpec(style=0, phase_x=0, phase_y=0), config.image_size, device=device)
        source_image, source_mask = render_on_background(background, source, config.image_size)
        target_image, target_mask = render_on_background(background, target, config.image_size)
        target_canonical_image, _ = render_on_background(canonical_background, target, config.image_size)
        examples.append(
            {
                "id": f"{split}-{index:06d}",
                "edit_prompt": edit,
                "source": attrs_phrase(source),
                "target": attrs_phrase(target),
                "edit_family": EDIT_FAMILIES[family],
                "background": asdict(bg_spec),
            }
        )
        prompts.append(encode_text(edit, config.prompt_len))
        source_prompts.append(encode_text(attrs_phrase(source), config.prompt_len))
        target_prompts.append(encode_text(attrs_phrase(target), config.prompt_len))
        backgrounds.append(background)
        canonical_backgrounds.append(canonical_background)
        sources.append(source_image)
        targets.append(target_image)
        target_canonical_images.append(target_canonical_image)
        source_masks.append(source_mask)
        target_masks.append(target_mask)
        source_attrs.append(source.tensor())
        target_attrs.append(target.tensor())
        edit_families.append(family)
    return StageAJSet(
        examples=examples,
        edit_prompt=torch.tensor(prompts, dtype=torch.long),
        source_prompt=torch.tensor(source_prompts, dtype=torch.long),
        target_prompt=torch.tensor(target_prompts, dtype=torch.long),
        background_image=torch.stack(backgrounds),
        canonical_background_image=torch.stack(canonical_backgrounds),
        source_image=torch.stack(sources),
        target_image=torch.stack(targets),
        target_canonical_image=torch.stack(target_canonical_images),
        source_mask=torch.stack(source_masks),
        target_mask=torch.stack(target_masks),
        source_attrs=torch.tensor(source_attrs, dtype=torch.long),
        target_attrs=torch.tensor(target_attrs, dtype=torch.long),
        edit_family=torch.tensor(edit_families, dtype=torch.long),
    )


def load_data(config: StageAJConfig) -> dict[str, StageAJSet]:
    return {
        "train": build_split("train", config.train_size, config),
        "val": build_split("val", config.val_size, config),
        "test": build_split("test", config.test_size, config),
    }


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image size must be divisible by patch size")
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(batch, -1, channels * patch_size * patch_size)


def unpatchify(patches: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
    batch = patches.shape[0]
    grid = image_size // patch_size
    patches = patches.view(batch, grid, grid, 3, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    return patches.view(batch, 3, image_size, image_size)


class TextEncoder(nn.Module):
    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.randn(config.prompt_len, config.d_model) * 0.02)
        self.embedding = nn.Embedding(VOCAB, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) + self.position[: tokens.shape[1]].unsqueeze(0)
        padding = tokens == PAD
        x = self.encoder(x, src_key_padding_mask=padding)
        return self.norm(x.masked_fill(padding.unsqueeze(-1), 0.0))


class PatchImageEncoder(nn.Module):
    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_dim = 3 * config.patch_size * config.patch_size
        self.patch_count = (config.image_size // config.patch_size) ** 2
        self.proj = nn.Linear(self.patch_dim, config.d_model)
        self.position = nn.Parameter(torch.randn(self.patch_count, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = patchify(image, self.config.patch_size)
        tokens = self.proj(patches) + self.position.unsqueeze(0)
        return self.norm(self.encoder(tokens))


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


def split_token_counts(total: int, levels: int) -> tuple[int, ...]:
    if total <= 0:
        raise ValueError("latent token count must be positive")
    if levels <= 0:
        raise ValueError("latent level count must be positive")
    base = total // levels
    remainder = total % levels
    counts = tuple(base + (1 if index < remainder else 0) for index in range(levels))
    return tuple(count for count in counts if count > 0)


class ProgressiveResidualLatent(nn.Module):
    def __init__(self, config: StageAJConfig, max_source_tokens: int) -> None:
        super().__init__()
        self.config = config
        self.max_source_tokens = max_source_tokens
        self.level_counts = split_token_counts(config.latent_tokens, config.latent_levels)
        self.level_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(count, config.d_model) * 0.02) for count in self.level_counts]
        )
        self.read_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
                for _ in self.level_counts
            ]
        )
        self.read_ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(config.d_model, config.d_model * 4),
                    nn.GELU(),
                    nn.Linear(config.d_model * 4, config.d_model),
                )
                for _ in self.level_counts
            ]
        )
        self.write_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
                for _ in self.level_counts
            ]
        )
        self.write_ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(config.d_model, config.d_model * 2),
                    nn.GELU(),
                    nn.Linear(config.d_model * 2, config.d_model),
                )
                for _ in self.level_counts
            ]
        )
        self.source_position = nn.Parameter(torch.randn(max_source_tokens, config.d_model) * 0.02)
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_residual = nn.LayerNorm(config.d_model)
        self.norm_level = nn.LayerNorm(config.d_model)

    def forward(self, source: torch.Tensor) -> dict[str, object]:
        if source.shape[1] > self.max_source_tokens:
            raise ValueError("source token count exceeds progressive latent capacity")
        residual = source
        cumulative = torch.zeros_like(source)
        levels: list[torch.Tensor] = []
        residual_energy: list[torch.Tensor] = []
        writer_query = self.source_position[: source.shape[1]].unsqueeze(0).expand(source.shape[0], -1, -1)
        for query_param, read_attn, read_ff, write_attn, write_ff in zip(
            self.level_queries,
            self.read_attn,
            self.read_ff,
            self.write_attn,
            self.write_ff,
        ):
            query = query_param.unsqueeze(0).expand(source.shape[0], -1, -1)
            attended, _ = read_attn(
                self.norm_q(query),
                self.norm_residual(residual),
                self.norm_residual(residual),
                need_weights=False,
            )
            level = query + attended
            level = level + read_ff(level)
            explained, _ = write_attn(
                writer_query,
                self.norm_level(level),
                self.norm_level(level),
                need_weights=False,
            )
            explained = explained + write_ff(explained)
            cumulative = cumulative + explained
            residual = residual - explained
            levels.append(level)
            residual_energy.append(residual.pow(2).mean(dim=(1, 2)))
        residual_token_loss = F.mse_loss(cumulative, source.detach())
        return {
            "levels": levels,
            "latent_tokens": torch.cat(levels, dim=1),
            "residual_token_loss": residual_token_loss,
            "residual_energy": torch.stack(residual_energy, dim=1),
        }


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_kv = nn.LayerNorm(config.d_model)
        self.norm_self = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

    def forward(self, patch_tokens: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross(
            self.norm_q(patch_tokens),
            self.norm_kv(latent),
            self.norm_kv(latent),
            need_weights=False,
        )
        patch_tokens = patch_tokens + attended
        self_out, _ = self.self_attn(
            self.norm_self(patch_tokens),
            self.norm_self(patch_tokens),
            self.norm_self(patch_tokens),
            need_weights=False,
        )
        patch_tokens = patch_tokens + self_out
        return patch_tokens + self.ff(patch_tokens)


class TransformerPatchDecoder(nn.Module):
    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_dim = 3 * config.patch_size * config.patch_size
        self.patch_count = (config.image_size // config.patch_size) ** 2
        self.patch_query = nn.Parameter(torch.randn(self.patch_count, config.d_model) * 0.02)
        self.position = nn.Parameter(torch.randn(self.patch_count, config.d_model) * 0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(config) for _ in range(config.decoder_layers)])
        self.out = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, self.patch_dim), nn.Sigmoid())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.patch_query.unsqueeze(0).expand(latent.shape[0], -1, -1) + self.position.unsqueeze(0)
        for block in self.blocks:
            x = block(x, latent)
        patches = self.out(x)
        return unpatchify(patches, self.config.image_size, self.config.patch_size)


class LatentObjectAuxHead(nn.Module):
    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Parameter(torch.randn(1, config.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_latent = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )
        self.color = nn.Linear(config.d_model, len(COLORS))
        self.shape = nn.Linear(config.d_model, len(SHAPES))
        self.position = nn.Linear(config.d_model, len(POSITIONS))
        self.mask = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.image_size * config.image_size),
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        query = self.query.unsqueeze(0).expand(latent.shape[0], -1, -1)
        attended, _ = self.attn(
            self.norm_q(query),
            self.norm_latent(latent),
            self.norm_latent(latent),
            need_weights=False,
        )
        hidden = query + attended
        hidden = hidden + self.ff(hidden)
        pooled = hidden[:, 0]
        return {
            "color_logits": self.color(pooled),
            "shape_logits": self.shape(pooled),
            "position_logits": self.position(pooled),
            "mask_logits": self.mask(pooled).view(latent.shape[0], 1, self.config.image_size, self.config.image_size),
        }


class TransformerImageCopyModel(nn.Module):
    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.image_encoder = PatchImageEncoder(config)
        self.latent = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)
        self.decoder = TransformerPatchDecoder(config)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image_tokens = self.image_encoder(image)
        latent = self.latent(image_tokens)
        return {"image": self.decoder(latent), "latent_tokens": latent}


class TransformerImageEditorModel(nn.Module):
    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.image_encoder = PatchImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.modality = nn.Embedding(2, config.d_model)
        self.latent = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)
        self.decoder = TransformerPatchDecoder(config)

    def forward(self, source_image: torch.Tensor, edit_prompt: torch.Tensor, *, zero_source: bool = False) -> dict[str, torch.Tensor]:
        image_tokens = self.image_encoder(source_image)
        if zero_source:
            image_tokens = image_tokens * 0.0
        image_tokens = image_tokens + self.modality.weight[0].view(1, 1, -1)
        text_tokens = self.text_encoder(edit_prompt) + self.modality.weight[1].view(1, 1, -1)
        latent = self.latent(torch.cat((image_tokens, text_tokens), dim=1))
        return {"image": self.decoder(latent), "latent_tokens": latent}


class MemoryTreeImageCopyModel(nn.Module):
    uses_hierarchical_loss = True

    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.image_encoder = PatchImageEncoder(config)
        max_tokens = (config.image_size // config.patch_size) ** 2
        self.latent = ProgressiveResidualLatent(config, max_tokens)
        self.decoder = TransformerPatchDecoder(config)

    def decode_levels(self, levels: list[torch.Tensor], *, return_prefix: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not return_prefix:
            return self.decoder(torch.cat(levels, dim=1)), []
        prefix_images: list[torch.Tensor] = []
        active: list[torch.Tensor] = []
        for level in levels:
            active.append(level)
            prefix_images.append(self.decoder(torch.cat(active, dim=1)))
        return prefix_images[-1], prefix_images

    def forward(self, image: torch.Tensor, *, return_prefix: bool = False) -> dict[str, object]:
        image_tokens = self.image_encoder(image)
        latent = self.latent(image_tokens)
        levels = latent["levels"]
        if not isinstance(levels, list):
            raise TypeError("progressive latent levels must be a list")
        image_out, prefix_images = self.decode_levels(levels, return_prefix=return_prefix)
        return {
            "image": image_out,
            "prefix_images": prefix_images,
            "latent_tokens": latent["latent_tokens"],
            "residual_token_loss": latent["residual_token_loss"],
            "residual_energy": latent["residual_energy"],
        }


class MemoryTreeSupervisedCopyModel(MemoryTreeImageCopyModel):
    uses_object_aux_loss = True

    def __init__(self, config: StageAJConfig) -> None:
        super().__init__(config)
        self.object_aux = LatentObjectAuxHead(config)

    def forward(self, image: torch.Tensor, *, return_prefix: bool = False) -> dict[str, object]:
        output = super().forward(image, return_prefix=return_prefix)
        latent_tokens = output["latent_tokens"]
        if not isinstance(latent_tokens, torch.Tensor):
            raise TypeError("latent_tokens must be a tensor")
        output["object_aux"] = self.object_aux(latent_tokens)
        return output


class MemoryTreeImageEditorModel(nn.Module):
    uses_hierarchical_loss = True

    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.image_encoder = PatchImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.modality = nn.Embedding(2, config.d_model)
        max_tokens = (config.image_size // config.patch_size) ** 2 + config.prompt_len
        self.latent = ProgressiveResidualLatent(config, max_tokens)
        self.decoder = TransformerPatchDecoder(config)

    def decode_levels(self, levels: list[torch.Tensor], *, return_prefix: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not return_prefix:
            return self.decoder(torch.cat(levels, dim=1)), []
        prefix_images: list[torch.Tensor] = []
        active: list[torch.Tensor] = []
        for level in levels:
            active.append(level)
            prefix_images.append(self.decoder(torch.cat(active, dim=1)))
        return prefix_images[-1], prefix_images

    def forward(
        self,
        source_image: torch.Tensor,
        edit_prompt: torch.Tensor,
        *,
        zero_source: bool = False,
        return_prefix: bool = False,
    ) -> dict[str, object]:
        image_tokens = self.image_encoder(source_image)
        if zero_source:
            image_tokens = image_tokens * 0.0
        image_tokens = image_tokens + self.modality.weight[0].view(1, 1, -1)
        text_tokens = self.text_encoder(edit_prompt) + self.modality.weight[1].view(1, 1, -1)
        latent = self.latent(torch.cat((image_tokens, text_tokens), dim=1))
        levels = latent["levels"]
        if not isinstance(levels, list):
            raise TypeError("progressive latent levels must be a list")
        image_out, prefix_images = self.decode_levels(levels, return_prefix=return_prefix)
        return {
            "image": image_out,
            "prefix_images": prefix_images,
            "latent_tokens": latent["latent_tokens"],
            "residual_token_loss": latent["residual_token_loss"],
            "residual_energy": latent["residual_energy"],
        }


class MemoryTreeSupervisedEditorModel(MemoryTreeImageEditorModel):
    uses_object_aux_loss = True

    def __init__(self, config: StageAJConfig) -> None:
        super().__init__(config)
        self.object_aux = LatentObjectAuxHead(config)

    def forward(
        self,
        source_image: torch.Tensor,
        edit_prompt: torch.Tensor,
        *,
        zero_source: bool = False,
        return_prefix: bool = False,
    ) -> dict[str, object]:
        output = super().forward(source_image, edit_prompt, zero_source=zero_source, return_prefix=return_prefix)
        latent_tokens = output["latent_tokens"]
        if not isinstance(latent_tokens, torch.Tensor):
            raise TypeError("latent_tokens must be a tensor")
        output["object_aux"] = self.object_aux(latent_tokens)
        return output


class TextSupervisedGenerationModel(nn.Module):
    uses_hierarchical_loss = True
    uses_object_aux_loss = True

    def __init__(self, config: StageAJConfig) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.latent = ProgressiveResidualLatent(config, config.prompt_len)
        self.decoder = TransformerPatchDecoder(config)
        self.object_aux = LatentObjectAuxHead(config)

    def decode_levels(self, levels: list[torch.Tensor], *, return_prefix: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not return_prefix:
            return self.decoder(torch.cat(levels, dim=1)), []
        prefix_images: list[torch.Tensor] = []
        active: list[torch.Tensor] = []
        for level in levels:
            active.append(level)
            prefix_images.append(self.decoder(torch.cat(active, dim=1)))
        return prefix_images[-1], prefix_images

    def forward(self, prompt: torch.Tensor, *, return_prefix: bool = False) -> dict[str, object]:
        text_tokens = self.text_encoder(prompt)
        latent = self.latent(text_tokens)
        levels = latent["levels"]
        if not isinstance(levels, list):
            raise TypeError("progressive latent levels must be a list")
        image_out, prefix_images = self.decode_levels(levels, return_prefix=return_prefix)
        latent_tokens = latent["latent_tokens"]
        if not isinstance(latent_tokens, torch.Tensor):
            raise TypeError("latent_tokens must be a tensor")
        return {
            "image": image_out,
            "prefix_images": prefix_images,
            "latent_tokens": latent_tokens,
            "residual_token_loss": latent["residual_token_loss"],
            "residual_energy": latent["residual_energy"],
            "object_aux": self.object_aux(latent_tokens),
        }


def random_batch(data: StageAJSet, *, generator: torch.Generator, batch_size: int, device: torch.device) -> StageAJSet:
    indices = torch.randint(len(data.examples), (batch_size,), device=device, generator=generator)
    return StageAJSet(
        examples=[],
        edit_prompt=data.edit_prompt.index_select(0, indices),
        source_prompt=data.source_prompt.index_select(0, indices),
        target_prompt=data.target_prompt.index_select(0, indices),
        background_image=data.background_image.index_select(0, indices),
        canonical_background_image=data.canonical_background_image.index_select(0, indices),
        source_image=data.source_image.index_select(0, indices),
        target_image=data.target_image.index_select(0, indices),
        target_canonical_image=data.target_canonical_image.index_select(0, indices),
        source_mask=data.source_mask.index_select(0, indices),
        target_mask=data.target_mask.index_select(0, indices),
        source_attrs=data.source_attrs.index_select(0, indices),
        target_attrs=data.target_attrs.index_select(0, indices),
        edit_family=data.edit_family.index_select(0, indices),
    )


def weighted_image_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    config: StageAJConfig,
) -> torch.Tensor:
    foreground_weight = 1.0 + config.foreground_loss_weight * target_mask
    foreground_loss = ((predicted - target) ** 2 * foreground_weight).mean()
    background = 1.0 - target_mask
    background_loss = ((predicted - target) ** 2 * background).sum() / background.sum().clamp_min(1.0)
    return foreground_loss + config.background_loss_weight * background_loss


def hierarchical_aux_loss(
    output: dict[str, object],
    target: torch.Tensor,
    target_mask: torch.Tensor,
    config: StageAJConfig,
) -> torch.Tensor:
    device = target.device
    loss = torch.zeros((), device=device)
    prefix_images = output.get("prefix_images", [])
    if isinstance(prefix_images, list) and len(prefix_images) > 1:
        for index, prefix_image in enumerate(prefix_images[:-1]):
            if not isinstance(prefix_image, torch.Tensor):
                continue
            weight = config.prefix_loss_weight / float(index + 1)
            loss = loss + weight * weighted_image_loss(prefix_image, target, target_mask, config)
    residual_token_loss = output.get("residual_token_loss")
    if isinstance(residual_token_loss, torch.Tensor):
        loss = loss + config.residual_token_loss_weight * residual_token_loss
    return loss


def object_aux_loss(
    output: dict[str, object],
    target_attrs: torch.Tensor,
    target_mask: torch.Tensor,
    config: StageAJConfig,
) -> torch.Tensor:
    aux = output.get("object_aux")
    loss = torch.zeros((), device=target_attrs.device)
    if not isinstance(aux, dict):
        return loss
    color_logits = aux.get("color_logits")
    shape_logits = aux.get("shape_logits")
    position_logits = aux.get("position_logits")
    if isinstance(color_logits, torch.Tensor) and isinstance(shape_logits, torch.Tensor) and isinstance(position_logits, torch.Tensor):
        attr_loss = (
            F.cross_entropy(color_logits, target_attrs[:, 0])
            + F.cross_entropy(shape_logits, target_attrs[:, 1])
            + F.cross_entropy(position_logits, target_attrs[:, 2])
        ) / 3.0
        loss = loss + config.object_aux_attr_loss_weight * attr_loss
    mask_logits = aux.get("mask_logits")
    if isinstance(mask_logits, torch.Tensor):
        mask_weight = 1.0 + config.object_aux_mask_positive_weight * target_mask
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, target_mask, weight=mask_weight)
        loss = loss + config.object_aux_mask_loss_weight * mask_loss
    return loss


@torch.no_grad()
def object_aux_metrics(
    output: dict[str, object],
    target_attrs: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, float]:
    aux = output.get("object_aux")
    if not isinstance(aux, dict):
        return {}
    metrics: dict[str, float] = {}
    color_logits = aux.get("color_logits")
    shape_logits = aux.get("shape_logits")
    position_logits = aux.get("position_logits")
    if isinstance(color_logits, torch.Tensor) and isinstance(shape_logits, torch.Tensor) and isinstance(position_logits, torch.Tensor):
        color = color_logits.argmax(dim=1)
        shape = shape_logits.argmax(dim=1)
        position = position_logits.argmax(dim=1)
        metrics["aux_color_acc"] = float((color == target_attrs[:, 0]).float().mean().detach().cpu())
        metrics["aux_shape_acc"] = float((shape == target_attrs[:, 1]).float().mean().detach().cpu())
        metrics["aux_position_acc"] = float((position == target_attrs[:, 2]).float().mean().detach().cpu())
        exact = (color == target_attrs[:, 0]) & (shape == target_attrs[:, 1]) & (position == target_attrs[:, 2])
        metrics["aux_scene_exact"] = float(exact.float().mean().detach().cpu())
    mask_logits = aux.get("mask_logits")
    if isinstance(mask_logits, torch.Tensor):
        mask_prob = mask_logits.sigmoid()
        pred = mask_prob > 0.5
        target = target_mask > 0.5
        intersection = (pred & target).float().sum(dim=(1, 2, 3))
        union = (pred | target).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
        metrics["aux_mask_iou"] = float((intersection / union).mean().detach().cpu())
        metrics["aux_mask_bce"] = float(F.binary_cross_entropy(mask_prob, target_mask).detach().cpu())
    return metrics


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


_TEMPLATE_BANK: dict[tuple[int, str, int | None], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def scene_template_bank(config: StageAJConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (config.image_size, device.type, device.index)
    cached = _TEMPLATE_BANK.get(key)
    if cached is not None:
        return cached
    masks: list[torch.Tensor] = []
    colors: list[torch.Tensor] = []
    attrs: list[list[int]] = []
    object_size = max(10, config.image_size // 4)
    for color in range(len(COLORS)):
        for shape in range(len(SHAPES)):
            for position in range(len(POSITIONS)):
                mask = torch.zeros((1, config.image_size, config.image_size), dtype=torch.float32, device=device)
                x, y = position_xy(position, config.image_size, object_size)
                local_mask = shape_mask(shape, object_size, device=device)
                mask[:, y : y + object_size, x : x + object_size][:, local_mask] = 1.0
                masks.append(mask)
                colors.append(torch.tensor(RGB[color], dtype=torch.float32, device=device).view(3, 1, 1))
                attrs.append([color, shape, position])
    bank = (
        torch.stack(masks, dim=0),
        torch.stack(colors, dim=0),
        torch.tensor(attrs, dtype=torch.long, device=device),
    )
    _TEMPLATE_BANK[key] = bank
    return bank


def nearest_attrs_with_background(
    predicted: torch.Tensor,
    backgrounds: torch.Tensor,
    config: StageAJConfig,
) -> torch.Tensor:
    masks, colors, attrs = scene_template_bank(config, predicted.device)
    best_distance = torch.full((predicted.shape[0],), math.inf, dtype=predicted.dtype, device=predicted.device)
    best_index = torch.zeros((predicted.shape[0],), dtype=torch.long, device=predicted.device)
    chunk_size = max(1, config.template_chunk_size)
    pred = predicted.unsqueeze(1)
    bg = backgrounds.unsqueeze(1)
    for start in range(0, masks.shape[0], chunk_size):
        mask = masks[start : start + chunk_size].unsqueeze(0)
        color = colors[start : start + chunk_size].unsqueeze(0)
        candidates = bg * (1.0 - mask) + color * mask
        distances = (pred - candidates).square().flatten(2).mean(dim=2)
        chunk_distance, chunk_index = distances.min(dim=1)
        better = chunk_distance < best_distance
        best_distance = torch.where(better, chunk_distance, best_distance)
        best_index = torch.where(better, chunk_index + start, best_index)
    return attrs.index_select(0, best_index)


@torch.no_grad()
def score_predictions(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    backgrounds: torch.Tensor,
    target_attrs: torch.Tensor,
    config: StageAJConfig,
) -> tuple[dict[str, float], torch.Tensor]:
    parsed = nearest_attrs_with_background(predicted, backgrounds, config)
    exact = (parsed == target_attrs).all(dim=1)
    foreground = target_mask.expand_as(target)
    background = 1.0 - foreground
    foreground_mse = ((predicted - target) ** 2 * foreground).sum() / foreground.sum().clamp_min(1.0)
    background_mse = ((predicted - target) ** 2 * background).sum() / background.sum().clamp_min(1.0)
    scores = {
        "pixel_mse": float(F.mse_loss(predicted, target).detach().cpu()),
        "pixel_mae": float(F.l1_loss(predicted, target).detach().cpu()),
        "foreground_mse": float(foreground_mse.detach().cpu()),
        "background_mse": float(background_mse.detach().cpu()),
        "nearest_color_acc": float((parsed[:, 0] == target_attrs[:, 0]).float().mean().detach().cpu()),
        "nearest_shape_acc": float((parsed[:, 1] == target_attrs[:, 1]).float().mean().detach().cpu()),
        "nearest_position_acc": float((parsed[:, 2] == target_attrs[:, 2]).float().mean().detach().cpu()),
        "nearest_scene_exact": float(exact.float().mean().detach().cpu()),
    }
    return scores, parsed


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: StageAJSet,
    config: StageAJConfig,
    *,
    task: str,
    device: torch.device,
    zero_source: bool = False,
) -> dict[str, object]:
    model.eval()
    chunks: list[dict[str, float]] = []
    family_scores: dict[str, list[float]] = {name: [] for name in EDIT_FAMILIES}
    for start in range(0, len(data.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.examples))))).to(device)
        if task == "copy":
            output = model(batch.source_image)
            predicted = output["image"]
            target = batch.source_image
            mask = batch.source_mask
            attrs = batch.source_attrs
        elif task == "edit":
            output = model(batch.source_image, batch.edit_prompt, zero_source=zero_source)
            predicted = output["image"]
            target = batch.target_image
            mask = batch.target_mask
            attrs = batch.target_attrs
        elif task == "generate":
            output = model(batch.target_prompt)
            predicted = output["image"]
            target = batch.target_canonical_image
            mask = batch.target_mask
            attrs = batch.target_attrs
        else:
            raise ValueError(f"unknown task: {task}")
        backgrounds = batch.canonical_background_image if task == "generate" else batch.background_image
        scores, parsed = score_predictions(predicted, target, mask, backgrounds, attrs, config)
        if isinstance(output, dict):
            scores.update(object_aux_metrics(output, attrs, mask))
        exact_values = (parsed == attrs).all(dim=1).float().detach().cpu().tolist()
        for family_index, value in zip(batch.edit_family.detach().cpu().tolist(), exact_values):
            family_scores[EDIT_FAMILIES[int(family_index)]].append(float(value))
        chunks.append(scores)
    model.train()
    keys = sorted(chunks[0])
    metrics: dict[str, object] = {key: statistics.fmean(chunk[key] for chunk in chunks) for key in keys}
    metrics["per_edit_family_scene_exact"] = {
        family: statistics.fmean(values) if values else 0.0 for family, values in family_scores.items()
    }
    return metrics


@torch.no_grad()
def evaluate_source_no_edit(data: StageAJSet, config: StageAJConfig, *, device: torch.device) -> dict[str, object]:
    chunks: list[dict[str, float]] = []
    family_scores: dict[str, list[float]] = {name: [] for name in EDIT_FAMILIES}
    for start in range(0, len(data.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.examples))))).to(device)
        scores, parsed = score_predictions(
            batch.source_image,
            batch.target_image,
            batch.target_mask,
            batch.background_image,
            batch.target_attrs,
            config,
        )
        exact_values = (parsed == batch.target_attrs).all(dim=1).float().detach().cpu().tolist()
        for family_index, value in zip(batch.edit_family.detach().cpu().tolist(), exact_values):
            family_scores[EDIT_FAMILIES[int(family_index)]].append(float(value))
        chunks.append(scores)
    keys = sorted(chunks[0])
    metrics: dict[str, object] = {key: statistics.fmean(chunk[key] for chunk in chunks) for key in keys}
    metrics["per_edit_family_scene_exact"] = {
        family: statistics.fmean(values) if values else 0.0 for family, values in family_scores.items()
    }
    return metrics


def deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def best_state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def checkpoint_score(metrics: dict[str, object]) -> float:
    foreground = float(metrics.get("foreground_mse", metrics.get("pixel_mse", 0.0)))
    scene = float(metrics.get("nearest_scene_exact", 0.0))
    return foreground - scene


def train_model(
    model: nn.Module,
    train: StageAJSet,
    val: StageAJSet,
    config: StageAJConfig,
    *,
    task: str,
    steps: int,
    seed: int,
    device: torch.device,
    deadline: float | None,
    label: str,
) -> dict[str, object]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.to(device)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=0.01,
            fused=(device.type == "cuda"),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    steps_completed = 0
    stopped_by_time_budget = False
    for step in range(1, steps + 1):
        if deadline_expired(deadline):
            stopped_by_time_budget = True
            break
        batch = random_batch(train, generator=generator, batch_size=config.batch_size, device=device)
        return_prefix = bool(getattr(model, "uses_hierarchical_loss", False))
        if task == "copy":
            output = model(batch.source_image, return_prefix=return_prefix) if return_prefix else model(batch.source_image)
            predicted = output["image"]
            target = batch.source_image
            mask = batch.source_mask
            attrs = batch.source_attrs
        elif task == "edit":
            output = (
                model(batch.source_image, batch.edit_prompt, return_prefix=return_prefix)
                if return_prefix
                else model(batch.source_image, batch.edit_prompt)
            )
            predicted = output["image"]
            target = batch.target_image
            mask = batch.target_mask
            attrs = batch.target_attrs
        elif task == "generate":
            output = model(batch.target_prompt, return_prefix=return_prefix) if return_prefix else model(batch.target_prompt)
            predicted = output["image"]
            target = batch.target_canonical_image
            mask = batch.target_mask
            attrs = batch.target_attrs
        else:
            raise ValueError(f"unknown task: {task}")
        if not isinstance(predicted, torch.Tensor):
            raise TypeError("model output image must be a tensor")
        loss = weighted_image_loss(predicted, target, mask, config)
        if return_prefix:
            loss = loss + hierarchical_aux_loss(output, target, mask, config)
        if bool(getattr(model, "uses_object_aux_loss", False)):
            loss = loss + object_aux_loss(output, attrs, mask, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps_completed = step
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_model(model, val, config, task=task, device=device)
            score = checkpoint_score(metrics)
            if score < best_score:
                best_score = score
                best_state = best_state_copy(model)
            item: dict[str, object] = {
                "step": step,
                "loss": round(float(loss.detach().cpu()), 6),
                "pixel_mse": metrics["pixel_mse"],
                "background_mse": metrics["background_mse"],
                "foreground_mse": metrics["foreground_mse"],
                "nearest_scene_exact": metrics["nearest_scene_exact"],
            }
            for key in ("aux_scene_exact", "aux_mask_iou", "aux_color_acc", "aux_shape_acc", "aux_position_acc"):
                if key in metrics:
                    item[key] = metrics[key]
            history.append(item)
            aux_text = ""
            if "aux_scene_exact" in metrics:
                aux_text = f" aux={metrics['aux_scene_exact']:.3f} mask_iou={metrics.get('aux_mask_iou', 0.0):.3f}"
            print(
                f"{label} step={step:4d} loss={float(loss.detach().cpu()):.6f} "
                f"mse={metrics['pixel_mse']:.5f} bg={metrics['background_mse']:.5f} "
                f"fg={metrics['foreground_mse']:.5f} scene={metrics['nearest_scene_exact']:.3f}{aux_text}",
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
    test: StageAJSet,
    config: StageAJConfig,
    *,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = test.subset(list(range(min(6, len(test.examples))))).to(device)
    outputs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, model in models.items():
            if name.endswith("_copy"):
                image = model(selected.source_image)["image"]
                if isinstance(image, torch.Tensor):
                    outputs[name] = image
            elif name.endswith("_edit"):
                image = model(selected.source_image, selected.edit_prompt)["image"]
                no_source = model(selected.source_image, selected.edit_prompt, zero_source=True)["image"]
                if isinstance(image, torch.Tensor):
                    outputs[name] = image
                if isinstance(no_source, torch.Tensor):
                    outputs[f"{name}_no_source"] = no_source
            elif name.endswith("_generate"):
                image = model(selected.target_prompt)["image"]
                if isinstance(image, torch.Tensor):
                    outputs[name] = image
    sample_meta: list[dict[str, object]] = []
    for index, example in enumerate(selected.examples):
        item: dict[str, object] = dict(example)
        for name, image in (
            ("background", selected.background_image[index]),
            ("canonical_background", selected.canonical_background_image[index]),
            ("source", selected.source_image[index]),
            ("target", selected.target_image[index]),
            ("target_canonical", selected.target_canonical_image[index]),
        ):
            path = output_dir / f"{index:02d}_{name}.png"
            write_png(path, image.detach().cpu())
            item[f"{name}_png"] = str(path)
        for name, images in outputs.items():
            path = output_dir / f"{index:02d}_{name}.png"
            write_png(path, images[index].detach().cpu())
            item[f"{name}_png"] = str(path)
        sample_meta.append(item)
    (output_dir / "samples.json").write_text(json.dumps(sample_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample_meta


def run_experiment(config: StageAJConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_aj_transformer_image_io_fidelity device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    started = time.perf_counter()
    deadline = started + config.max_train_seconds if config.max_train_seconds > 0 else None
    data = load_data(config)
    train = data["train"].to(device)
    val = data["val"].to(device)
    test = data["test"].to(device)
    train_eval_count = min(config.train_eval_size, len(val.examples)) if config.train_eval_size > 0 else len(val.examples)
    train_val = val.subset(list(range(train_eval_count))).to(device)
    models: dict[str, nn.Module] = {}
    training: dict[str, object] = {}
    metrics: dict[str, object] = {}
    ablations: dict[str, object] = {}

    if "transformer_copy" in config.variants:
        model = TransformerImageCopyModel(config).to(device)
        training["transformer_copy"] = train_model(
            model,
            train,
            train_val,
            config,
            task="copy",
            steps=config.copy_steps,
            seed=config.seed + 401,
            device=device,
            deadline=deadline,
            label="transformer_copy",
        )
        metrics["transformer_copy"] = evaluate_model(model, test, config, task="copy", device=device)
        models["transformer_copy"] = model
    if "memory_tree_copy" in config.variants:
        model = MemoryTreeImageCopyModel(config).to(device)
        training["memory_tree_copy"] = train_model(
            model,
            train,
            train_val,
            config,
            task="copy",
            steps=config.copy_steps,
            seed=config.seed + 431,
            device=device,
            deadline=deadline,
            label="memory_tree_copy",
        )
        metrics["memory_tree_copy"] = evaluate_model(model, test, config, task="copy", device=device)
        models["memory_tree_copy"] = model
    if "memory_tree_supervised_copy" in config.variants:
        model = MemoryTreeSupervisedCopyModel(config).to(device)
        training["memory_tree_supervised_copy"] = train_model(
            model,
            train,
            train_val,
            config,
            task="copy",
            steps=config.copy_steps,
            seed=config.seed + 443,
            device=device,
            deadline=deadline,
            label="memory_tree_supervised_copy",
        )
        metrics["memory_tree_supervised_copy"] = evaluate_model(model, test, config, task="copy", device=device)
        models["memory_tree_supervised_copy"] = model
    if "transformer_edit" in config.variants:
        model = TransformerImageEditorModel(config).to(device)
        training["transformer_edit"] = train_model(
            model,
            train,
            train_val,
            config,
            task="edit",
            steps=config.edit_steps,
            seed=config.seed + 503,
            device=device,
            deadline=deadline,
            label="transformer_edit",
        )
        metrics["transformer_edit"] = evaluate_model(model, test, config, task="edit", device=device)
        ablations["transformer_edit_no_source"] = evaluate_model(
            model,
            test,
            config,
            task="edit",
            device=device,
            zero_source=True,
        )
        models["transformer_edit"] = model
    if "memory_tree_edit" in config.variants:
        model = MemoryTreeImageEditorModel(config).to(device)
        training["memory_tree_edit"] = train_model(
            model,
            train,
            train_val,
            config,
            task="edit",
            steps=config.edit_steps,
            seed=config.seed + 547,
            device=device,
            deadline=deadline,
            label="memory_tree_edit",
        )
        metrics["memory_tree_edit"] = evaluate_model(model, test, config, task="edit", device=device)
        ablations["memory_tree_edit_no_source"] = evaluate_model(
            model,
            test,
            config,
            task="edit",
            device=device,
            zero_source=True,
        )
        models["memory_tree_edit"] = model
    if "memory_tree_supervised_edit" in config.variants:
        model = MemoryTreeSupervisedEditorModel(config).to(device)
        training["memory_tree_supervised_edit"] = train_model(
            model,
            train,
            train_val,
            config,
            task="edit",
            steps=config.edit_steps,
            seed=config.seed + 557,
            device=device,
            deadline=deadline,
            label="memory_tree_supervised_edit",
        )
        metrics["memory_tree_supervised_edit"] = evaluate_model(model, test, config, task="edit", device=device)
        ablations["memory_tree_supervised_edit_no_source"] = evaluate_model(
            model,
            test,
            config,
            task="edit",
            device=device,
            zero_source=True,
        )
        models["memory_tree_supervised_edit"] = model
    if "text_supervised_generate" in config.variants:
        model = TextSupervisedGenerationModel(config).to(device)
        training["text_supervised_generate"] = train_model(
            model,
            train,
            train_val,
            config,
            task="generate",
            steps=config.generate_steps,
            seed=config.seed + 601,
            device=device,
            deadline=deadline,
            label="text_supervised_generate",
        )
        metrics["text_supervised_generate"] = evaluate_model(model, test, config, task="generate", device=device)
        models["text_supervised_generate"] = model
    if any(variant.endswith("_edit") for variant in config.variants):
        ablations["source_image_no_edit"] = evaluate_source_no_edit(test, config, device=device)
    samples = write_sample_outputs(
        models,
        test,
        config,
        device=device,
        output_dir=output_path.parent / "samples" / output_path.stem / f"seed{config.seed}",
    )
    stopped_by_time_budget = any(bool(item.get("stopped_by_time_budget")) for item in training.values() if isinstance(item, dict))
    output = {
        "experiment": "omni_transformer_stage_aj_transformer_image_io_fidelity",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "task": "Transformer-only image input -> latent -> full image redraw fidelity",
            "transformer_copy": "patch-token image encoder -> fixed latent tokens -> Transformer patch decoder reconstructs the full source image",
            "transformer_edit": "source patch tokens + low-entropy edit text -> fixed latent tokens -> Transformer patch decoder redraws the full target image",
            "memory_tree_copy": "patch-token image encoder -> progressive residual memory-tree latent levels -> Transformer patch decoder reconstructs the full source image",
            "memory_tree_supervised_copy": "memory-tree copy with latent object supervision for source color, shape, position, and foreground mask",
            "memory_tree_edit": "source patch tokens + edit text -> progressive residual memory-tree latent levels -> Transformer patch decoder redraws the full target image",
            "memory_tree_supervised_edit": "memory-tree edit with target object attribute and mask supervision on latent tokens",
            "text_supervised_generate": "full target text prompt -> memory-tree latent levels -> Transformer patch decoder draws the target object on canonical background",
            "memory_tree_latent": "each level reads the current residual token field, writes back the explained part, then passes the residual to the next level",
            "copy_auxiliary_supervision": "copy-only object attribute and mask heads supervise latent content but the Transformer patch decoder still redraws the full image",
            "edit_generation_auxiliary_supervision": "edit/generation variants use the same latent object attribute and mask heads while keeping image generation in the patch decoder",
            "background_requirement": "per-example synthetic background texture is not present in the edit prompt and must be carried through the image input latent path",
            "generation_scope": "text generation uses a canonical predictable background; random per-example background is not inferable from text alone",
            "time_budget_seconds": config.max_train_seconds,
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
            "background_styles": 8,
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
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
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

    variants = sorted({variant for run in runs for variant in run["config"]["variants"]})  # type: ignore[index]
    summary: dict[str, dict[str, float | None]] = {}
    for variant in variants:
        if variant.endswith("_copy"):
            summary[variant] = {
                "pixel_mse": mean(f"metrics.{variant}.pixel_mse"),
                "background_mse": mean(f"metrics.{variant}.background_mse"),
                "foreground_mse": mean(f"metrics.{variant}.foreground_mse"),
                "nearest_scene_exact": mean(f"metrics.{variant}.nearest_scene_exact"),
            }
        elif variant.endswith("_edit"):
            summary[variant] = {
                "pixel_mse": mean(f"metrics.{variant}.pixel_mse"),
                "background_mse": mean(f"metrics.{variant}.background_mse"),
                "foreground_mse": mean(f"metrics.{variant}.foreground_mse"),
                "nearest_scene_exact": mean(f"metrics.{variant}.nearest_scene_exact"),
                "no_source_scene_exact": mean(f"ablations.{variant}_no_source.nearest_scene_exact"),
                "source_no_edit_scene_exact": mean("ablations.source_image_no_edit.nearest_scene_exact"),
            }
        elif variant.endswith("_generate"):
            summary[variant] = {
                "pixel_mse": mean(f"metrics.{variant}.pixel_mse"),
                "background_mse": mean(f"metrics.{variant}.background_mse"),
                "foreground_mse": mean(f"metrics.{variant}.foreground_mse"),
                "nearest_scene_exact": mean(f"metrics.{variant}.nearest_scene_exact"),
                "aux_scene_exact": mean(f"metrics.{variant}.aux_scene_exact"),
            }

    return {
        "experiment": "omni_transformer_stage_aj_transformer_image_io_fidelity_sweep",
        "run_count": len(runs),
        "seeds": sorted({int(run["config"]["seed"]) for run in runs}),  # type: ignore[index]
        "variants": variants,
        "summary": summary,
        "stats": stats,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_aj_transformer_image_io_fidelity/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701")
    parser.add_argument("--variants", default="memory_tree_supervised_copy")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--latent-levels", type=int, default=4)
    parser.add_argument("--copy-steps", type=int, default=1200)
    parser.add_argument("--edit-steps", type=int, default=1800)
    parser.add_argument("--generate-steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--train-eval-size", type=int, default=128)
    parser.add_argument("--template-chunk-size", type=int, default=30)
    parser.add_argument("--foreground-loss-weight", type=float, default=8.0)
    parser.add_argument("--background-loss-weight", type=float, default=2.0)
    parser.add_argument("--prefix-loss-weight", type=float, default=0.25)
    parser.add_argument("--residual-token-loss-weight", type=float, default=0.02)
    parser.add_argument("--object-aux-attr-loss-weight", type=float, default=1.0)
    parser.add_argument("--object-aux-mask-loss-weight", type=float, default=0.5)
    parser.add_argument("--object-aux-mask-positive-weight", type=float, default=8.0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--max-train-seconds", type=int, default=129_600)
    args = parser.parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    variants = parse_csv_strings(args.variants)
    valid = {
        "transformer_copy",
        "transformer_edit",
        "memory_tree_copy",
        "memory_tree_supervised_copy",
        "memory_tree_edit",
        "memory_tree_supervised_edit",
        "text_supervised_generate",
    }
    unknown = set(variants) - valid
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}; choices are {sorted(valid)}")
    base_config = StageAJConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        prompt_len=args.prompt_len,
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        latent_tokens=args.latent_tokens,
        latent_levels=args.latent_levels,
        copy_steps=args.copy_steps,
        edit_steps=args.edit_steps,
        generate_steps=args.generate_steps,
        eval_every=args.eval_every,
        train_eval_size=args.train_eval_size,
        template_chunk_size=args.template_chunk_size,
        foreground_loss_weight=args.foreground_loss_weight,
        background_loss_weight=args.background_loss_weight,
        prefix_loss_weight=args.prefix_loss_weight,
        residual_token_loss_weight=args.residual_token_loss_weight,
        object_aux_attr_loss_weight=args.object_aux_attr_loss_weight,
        object_aux_mask_loss_weight=args.object_aux_mask_loss_weight,
        object_aux_mask_positive_weight=args.object_aux_mask_positive_weight,
        max_train_seconds=args.max_train_seconds,
        variants=variants,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageAJConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
