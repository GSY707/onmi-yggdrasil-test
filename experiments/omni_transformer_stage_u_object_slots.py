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
import omni_transformer_stage_p_llava_moe as stage_p
import omni_transformer_stage_q_tiny_moe_vlm_from_scratch as stage_q
import omni_transformer_stage_r_curriculum_vlm as stage_r
import omni_transformer_stage_s_scorer_reconstruction as stage_s


COLORS = stage_r.COLORS
RGB = stage_r.RGB
SHAPES = stage_r.SHAPES
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven")
FAMILIES = ("u1_cell_attribute", "u2_spatial_relation", "u3_counting", "u4_spatial_filter")
FAMILY_DESCRIPTIONS = {
    "u1_cell_attribute": "multi-object scene, answer the color at a queried row/column cell",
    "u2_spatial_relation": "multi-object scene, answer nearest object color in a direction from a named reference",
    "u3_counting": "multi-object scene, count objects matching a queried color+shape",
    "u4_spatial_filter": "multi-object scene, answer the color of the top/bottom/left/right-most object of a queried shape",
}

PATCH_EXPERT = 0
OBJECT_EXPERT = 1
SPATIAL_EXPERT = 2
COUNT_EXPERT = 3
PROMPT_EXPERT = 4
FUSION_EXPERT = 5
EXPERTS_U = ("patch_vision", "object_slots", "spatial", "counting", "prompt_text", "fusion")
PATCH_ONLY_EXPERT_INDICES = (PATCH_EXPERT, PROMPT_EXPERT, FUSION_EXPERT)

MODEL_VARIANTS = ("patch_wide_latent", "object_slot_spatial_wide_latent")
PROBE_STAGES = ("patch_tokens", "object_slots", "spatial_tokens", "count_tokens", "fusion_tokens", "wide_latent")


@dataclass(frozen=True)
class StageUConfig:
    train_size: int = 1536
    val_size: int = 384
    test_size: int = 512
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
    output_tokens: int = 16
    candidate_count: int = 8
    train_steps: int = 520
    eval_every: int = 130
    router_loss_weight: float = 0.05
    patch_context_dropout: float = 0.25
    probe_steps: int = 100
    probe_hidden: int = 192
    variants: tuple[str, ...] = MODEL_VARIANTS


@dataclass(frozen=True)
class SceneObject:
    color: int
    shape: int
    row: int
    col: int


@dataclass(frozen=True)
class StageUItem:
    example: stage_p.LLaVAExample
    image: torch.Tensor
    meta: dict[str, object]


def stage_q_config(config: StageUConfig) -> stage_q.StageQConfig:
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


def split_seed(seed: int, family: str, split: str) -> int:
    family_offset = FAMILIES.index(family) * 100_000
    split_offset = {"train": 17, "val": 31, "test": 53}[split]
    return seed + family_offset + split_offset


def make_llava_like_example(
    *,
    family: str,
    split: str,
    index: int,
    prompt: str,
    answer: str,
) -> stage_p.LLaVAExample:
    return stage_p.LLaVAExample(
        id=f"{family}-{split}-{index:06d}",
        image_name=f"{family}_{split}_{index:06d}.png",
        image_path="synthetic://stage_u",
        prompt=prompt,
        answer=answer,
        has_history=False,
        source_file=f"stage_u/{family}",
        turn_index=0,
    )


def required_experts() -> tuple[float, ...]:
    return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def blank_image(image_size: int) -> torch.Tensor:
    image = torch.full((3, image_size, image_size), 0.050, dtype=torch.float32)
    image[:, 4:-4, 4:-4] = 0.115
    image[:, 4:5, 4:-4] = 0.23
    image[:, -5:-4, 4:-4] = 0.23
    image[:, 4:-4, 4:5] = 0.23
    image[:, 4:-4, -5:-4] = 0.23
    return image


def object_center(obj: SceneObject, config: StageUConfig) -> tuple[int, int]:
    margin = 7
    usable = config.image_size - 2 * margin
    cell = usable / config.grid_size
    cx = int(margin + (obj.col + 0.5) * cell)
    cy = int(margin + (obj.row + 0.5) * cell)
    return cx, cy


def draw_shape(image: torch.Tensor, obj: SceneObject, config: StageUConfig) -> None:
    object_size = max(7, int(config.image_size / (config.grid_size * 1.55)))
    cx, cy = object_center(obj, config)
    x0 = max(0, cx - object_size // 2)
    y0 = max(0, cy - object_size // 2)
    x1 = min(config.image_size, x0 + object_size)
    y1 = min(config.image_size, y0 + object_size)
    h = y1 - y0
    w = x1 - x0
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    if SHAPES[obj.shape] == "circle":
        center_y = (h - 1) / 2
        center_x = (w - 1) / 2
        mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= (min(h, w) * 0.43) ** 2
    elif SHAPES[obj.shape] == "square":
        mask = torch.ones(h, w, dtype=torch.bool)
    elif SHAPES[obj.shape] == "triangle":
        mask = yy >= torch.abs(xx - (w - 1) / 2)
    else:
        center_y = (h - 1) / 2
        center_x = (w - 1) / 2
        mask = (torch.abs(xx - center_x) + torch.abs(yy - center_y)) <= min(h, w) * 0.48
    color = torch.tensor(RGB[obj.color], dtype=torch.float32).view(3, 1)
    patch = image[:, y0:y1, x0:x1]
    patch[:, mask] = color.expand(3, int(mask.sum()))


def render_scene(objects: Iterable[SceneObject], config: StageUConfig) -> torch.Tensor:
    image = blank_image(config.image_size)
    for obj in objects:
        draw_shape(image, obj, config)
    return image


def object_phrase(obj: SceneObject) -> str:
    return f"{COLORS[obj.color]} {SHAPES[obj.shape]}"


def sample_unique_objects(rng: random.Random, count: int, config: StageUConfig) -> list[SceneObject]:
    cells = [(row, col) for row in range(config.grid_size) for col in range(config.grid_size)]
    rng.shuffle(cells)
    pairs = [(color, shape) for color in range(len(COLORS)) for shape in range(len(SHAPES))]
    rng.shuffle(pairs)
    objects = []
    for index in range(count):
        row, col = cells[index]
        color, shape = pairs[index]
        objects.append(SceneObject(color=color, shape=shape, row=row, col=col))
    return objects


def candidate_answers_for_family(family: str) -> tuple[str, ...]:
    return COUNT_WORDS if family == "u3_counting" else COLORS


def make_family_item(family: str, split: str, index: int, rng: random.Random, config: StageUConfig) -> StageUItem:
    if family == "u1_cell_attribute":
        objects = sample_unique_objects(rng, rng.randint(4, 7), config)
        target = rng.choice(objects)
        prompt = f"what color is the object at row {target.row + 1} column {target.col + 1}? answer one color."
        answer = COLORS[target.color]
    elif family == "u2_spatial_relation":
        objects = sample_unique_objects(rng, rng.randint(5, 8), config)
        candidates: list[tuple[SceneObject, str, SceneObject]] = []
        for ref in objects:
            directions = {
                "left of": [obj for obj in objects if obj.col < ref.col],
                "right of": [obj for obj in objects if obj.col > ref.col],
                "above": [obj for obj in objects if obj.row < ref.row],
                "below": [obj for obj in objects if obj.row > ref.row],
            }
            for direction, options in directions.items():
                if not options:
                    continue
                nearest = min(options, key=lambda obj: abs(obj.row - ref.row) + abs(obj.col - ref.col))
                candidates.append((ref, direction, nearest))
        ref, direction, target = rng.choice(candidates)
        prompt = f"what color is the nearest object {direction} the {object_phrase(ref)}? answer one color."
        answer = COLORS[target.color]
    elif family == "u3_counting":
        target_color = rng.randrange(len(COLORS))
        target_shape = rng.randrange(len(SHAPES))
        target_count = rng.randrange(0, 6)
        cells = [(row, col) for row in range(config.grid_size) for col in range(config.grid_size)]
        rng.shuffle(cells)
        objects = []
        for _ in range(target_count):
            row, col = cells.pop()
            objects.append(SceneObject(color=target_color, shape=target_shape, row=row, col=col))
        distractor_count = rng.randint(max(3, 6 - target_count), 8 - target_count)
        for _ in range(distractor_count):
            row, col = cells.pop()
            color = rng.randrange(len(COLORS))
            shape = rng.randrange(len(SHAPES))
            if color == target_color and shape == target_shape:
                shape = (shape + 1) % len(SHAPES)
            objects.append(SceneObject(color=color, shape=shape, row=row, col=col))
        rng.shuffle(objects)
        prompt = f"how many {COLORS[target_color]} {SHAPES[target_shape]} objects are in the image?"
        answer = COUNT_WORDS[target_count]
    elif family == "u4_spatial_filter":
        objects = sample_unique_objects(rng, rng.randint(5, 8), config)
        shape = rng.choice([obj.shape for obj in objects])
        shape_objects = [obj for obj in objects if obj.shape == shape]
        if len(shape_objects) < 2:
            extra_shape = shape
            used = {(obj.row, obj.col) for obj in objects}
            free = [(row, col) for row in range(config.grid_size) for col in range(config.grid_size) if (row, col) not in used]
            row, col = rng.choice(free)
            color = rng.randrange(len(COLORS))
            objects.append(SceneObject(color=color, shape=extra_shape, row=row, col=col))
            shape_objects = [obj for obj in objects if obj.shape == shape]
        relation = rng.choice(("topmost", "bottommost", "leftmost", "rightmost"))
        if relation == "topmost":
            target = min(shape_objects, key=lambda obj: (obj.row, obj.col))
        elif relation == "bottommost":
            target = max(shape_objects, key=lambda obj: (obj.row, -obj.col))
        elif relation == "leftmost":
            target = min(shape_objects, key=lambda obj: (obj.col, obj.row))
        else:
            target = max(shape_objects, key=lambda obj: (obj.col, -obj.row))
        prompt = f"what color is the {relation} {SHAPES[shape]} object? answer one color."
        answer = COLORS[target.color]
    else:
        raise ValueError(f"unknown family: {family}")

    image = render_scene(objects, config)
    example = make_llava_like_example(family=family, split=split, index=index, prompt=prompt, answer=answer)
    return StageUItem(
        example=example,
        image=image,
        meta={
            "family": family,
            "prompt": prompt,
            "answer": answer,
            "objects": [asdict(obj) for obj in objects],
        },
    )


def family_sizes(total: int) -> dict[str, int]:
    base = total // len(FAMILIES)
    sizes = {family: base for family in FAMILIES}
    for family in FAMILIES[: total - base * len(FAMILIES)]:
        sizes[family] += 1
    return sizes


def build_pixel_set(split: str, size: int, config: StageUConfig) -> tuple[stage_q.PixelSet, list[dict[str, object]]]:
    items: list[StageUItem] = []
    sizes = family_sizes(size)
    for family in FAMILIES:
        rng = random.Random(split_seed(config.seed, family, split))
        for index in range(sizes[family]):
            items.append(make_family_item(family, split, index, rng, config))
    shuffle_rng = random.Random(config.seed + {"train": 1001, "val": 1002, "test": 1003}[split])
    shuffle_rng.shuffle(items)
    prompts = torch.tensor(
        [stage_q.encode_text(item.example.prompt, config.prompt_len, add_eos=True) for item in items],
        dtype=torch.long,
    )
    required = torch.tensor([required_experts() for _ in items], dtype=torch.float32)
    return (
        stage_q.PixelSet(
            examples=[item.example for item in items],
            image=torch.stack([item.image for item in items], dim=0),
            prompt=prompts,
            required_experts=required,
        ),
        [item.meta for item in items],
    )


def build_data(config: StageUConfig) -> tuple[dict[str, stage_q.PixelSet], dict[str, object], dict[str, list[dict[str, object]]]]:
    started = time.perf_counter()
    train, train_meta = build_pixel_set("train", config.train_size, config)
    val, val_meta = build_pixel_set("val", config.val_size, config)
    test, test_meta = build_pixel_set("test", config.test_size, config)
    stats = {
        "families": FAMILY_DESCRIPTIONS,
        "split_sizes": {"train": len(train.examples), "val": len(val.examples), "test": len(test.examples)},
        "family_sizes": {
            "train": family_sizes(config.train_size),
            "val": family_sizes(config.val_size),
            "test": family_sizes(config.test_size),
        },
        "answer_spaces": {
            "color_candidate_count": len(COLORS),
            "count_candidate_count": len(COUNT_WORDS),
            "candidate_count": config.candidate_count,
            "random_top1": 1.0 / config.candidate_count,
        },
        "visual_input_tokens": config.patch_grid * config.patch_grid,
        "object_slots": config.object_slots,
        "build_seconds": round(time.perf_counter() - started, 3),
    }
    return {"train": train, "val": val, "test": test}, stats, {"train": train_meta, "val": val_meta, "test": test_meta}


def build_candidate_pool(config: StageUConfig, *, device: torch.device) -> stage_q.CandidatePool:
    answers = list(dict.fromkeys([*COLORS, *COUNT_WORDS]))
    tokens = torch.tensor([stage_q.encode_text(answer, config.answer_len, add_eos=True) for answer in answers], dtype=torch.long, device=device)
    return stage_q.CandidatePool(answers=answers, tokens=tokens, index_by_answer={answer: index for index, answer in enumerate(answers)})


def candidate_rows_for_examples(
    examples: list[stage_p.LLaVAExample],
    candidate_pool: stage_q.CandidatePool,
    *,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    true_indices: list[int] = []
    for example in examples:
        family = example.source_file.split("/")[-1]
        candidates = list(candidate_answers_for_family(family))
        rng.shuffle(candidates)
        row = [candidate_pool.index_by_answer[answer] for answer in candidates]
        true_pool_index = candidate_pool.index_by_answer[example.answer]
        rows.append(row)
        true_indices.append(row.index(true_pool_index))
    return torch.tensor(rows, dtype=torch.long, device=candidate_pool.tokens.device), true_indices


class ExpandedPatchVisionExpert(nn.Module):
    def __init__(self, config: StageUConfig) -> None:
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
        self.position = nn.Parameter(torch.randn(config.patch_grid * config.patch_grid, config.d_model) * 0.02)
        coords = []
        for row in range(config.patch_grid):
            for col in range(config.patch_grid):
                coords.append((col / max(config.patch_grid - 1, 1) * 2 - 1, row / max(config.patch_grid - 1, 1) * 2 - 1))
        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32), persistent=False)
        self.coord_proj = nn.Linear(2, config.d_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        if x.shape[-1] != self.config.patch_grid or x.shape[-2] != self.config.patch_grid:
            x = F.adaptive_avg_pool2d(x, (self.config.patch_grid, self.config.patch_grid))
        tokens = x.flatten(2).transpose(1, 2)
        return tokens + self.position.unsqueeze(0) + self.coord_proj(self.coords).unsqueeze(0)


class OutputTokenBus(nn.Module):
    def __init__(self, config: StageUConfig) -> None:
        super().__init__()
        self.output = stage_q.QueryResampler(config.d_model, config.heads, config.output_tokens, config.dropout)

    def forward(self, raw: torch.Tensor, weighted: torch.Tensor) -> torch.Tensor:
        source = torch.cat((raw, weighted), dim=1)
        return torch.cat((source, self.output(source)), dim=1)


class StageUWideVLM(nn.Module):
    def __init__(self, config: StageUConfig, *, variant: str) -> None:
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self.config = config
        self.variant = variant
        self.use_object_experts = variant == "object_slot_spatial_wide_latent"
        model_config = stage_q_config(config)
        self.patch_vision = ExpandedPatchVisionExpert(config)
        self.text_encoder = stage_q.TextTokenEncoder(model_config, length=config.prompt_len)
        self.prompt_resampler = stage_q.QueryResampler(config.d_model, config.heads, config.prompt_tokens, config.dropout)
        self.object_resampler = (
            stage_q.QueryResampler(config.d_model, config.heads, config.object_slots, config.dropout)
            if self.use_object_experts
            else None
        )
        self.spatial_resampler = (
            stage_q.QueryResampler(config.d_model, config.heads, config.spatial_tokens, config.dropout)
            if self.use_object_experts
            else None
        )
        self.count_resampler = (
            stage_q.QueryResampler(config.d_model, config.heads, config.count_tokens, config.dropout)
            if self.use_object_experts
            else None
        )
        self.fusion_resampler = stage_q.QueryResampler(config.d_model, config.heads, config.fusion_tokens, config.dropout)
        self.expert_type = nn.Embedding(len(EXPERTS_U), config.d_model)
        route_count = len(EXPERTS_U) if self.use_object_experts else len(PATCH_ONLY_EXPERT_INDICES)
        self.router = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, route_count),
        )
        self.bus = OutputTokenBus(config)
        self.scorer = stage_s.CrossAttentionAnswerScorer(model_config)

    def route_target(self, required_experts: torch.Tensor) -> torch.Tensor:
        if self.use_object_experts:
            return required_experts
        return required_experts[:, list(PATCH_ONLY_EXPERT_INDICES)]

    def representations(self, batch: stage_q.PixelSet, *, zero_modalities: Iterable[str] = ()) -> dict[str, torch.Tensor]:
        zero_set = set(zero_modalities)
        image = torch.zeros_like(batch.image) if "image" in zero_set else batch.image
        prompt = torch.full_like(batch.prompt, stage_q.PAD) if "text" in zero_set else batch.prompt
        raw_patch_tokens = self.patch_vision(image) + self.expert_type.weight[PATCH_EXPERT].view(1, 1, -1)
        patch_tokens = raw_patch_tokens
        text_source = self.text_encoder(prompt)
        prompt_tokens = self.prompt_resampler(text_source) + self.expert_type.weight[PROMPT_EXPERT].view(1, 1, -1)

        if self.use_object_experts:
            assert self.object_resampler is not None
            assert self.spatial_resampler is not None
            assert self.count_resampler is not None
            object_slots = self.object_resampler(raw_patch_tokens) + self.expert_type.weight[OBJECT_EXPERT].view(1, 1, -1)
            spatial_tokens = self.spatial_resampler(raw_patch_tokens) + self.expert_type.weight[SPATIAL_EXPERT].view(1, 1, -1)
            count_source = torch.cat((object_slots, spatial_tokens, prompt_tokens), dim=1)
            count_tokens = self.count_resampler(count_source) + self.expert_type.weight[COUNT_EXPERT].view(1, 1, -1)
            if "patch_expert" in zero_set:
                patch_tokens = torch.zeros_like(patch_tokens)
            if "object_experts" in zero_set:
                object_slots = torch.zeros_like(object_slots)
                spatial_tokens = torch.zeros_like(spatial_tokens)
                count_tokens = torch.zeros_like(count_tokens)
            if "spatial_expert" in zero_set:
                spatial_tokens = torch.zeros_like(spatial_tokens)
            if "count_expert" in zero_set:
                count_tokens = torch.zeros_like(count_tokens)
            fusion_source = torch.cat((patch_tokens, object_slots, spatial_tokens, count_tokens, prompt_tokens), dim=1)
            fusion_tokens = self.fusion_resampler(fusion_source) + self.expert_type.weight[FUSION_EXPERT].view(1, 1, -1)
            expert_tokens = [patch_tokens, object_slots, spatial_tokens, count_tokens, prompt_tokens, fusion_tokens]
        else:
            if "patch_expert" in zero_set:
                patch_tokens = torch.zeros_like(patch_tokens)
            object_slots = torch.zeros(batch.image.shape[0], self.config.object_slots, self.config.d_model, device=batch.image.device)
            spatial_tokens = torch.zeros(batch.image.shape[0], self.config.spatial_tokens, self.config.d_model, device=batch.image.device)
            count_tokens = torch.zeros(batch.image.shape[0], self.config.count_tokens, self.config.d_model, device=batch.image.device)
            fusion_tokens = self.fusion_resampler(torch.cat((patch_tokens, prompt_tokens), dim=1)) + self.expert_type.weight[FUSION_EXPERT].view(1, 1, -1)
            expert_tokens = [patch_tokens, prompt_tokens, fusion_tokens]

        route_source = torch.cat((patch_tokens.mean(dim=1), prompt_tokens.mean(dim=1)), dim=-1)
        route_logits = self.router(route_source)
        route_weights = torch.sigmoid(route_logits)
        weighted = [tokens * route_weights[:, index].view(-1, 1, 1) for index, tokens in enumerate(expert_tokens)]
        raw = torch.cat(expert_tokens, dim=1)
        weighted_concat = torch.cat(weighted, dim=1)
        wide_latent = self.bus(raw, weighted_concat)
        return {
            "patch_tokens": patch_tokens,
            "object_slots": object_slots,
            "spatial_tokens": spatial_tokens,
            "count_tokens": count_tokens,
            "prompt_tokens": prompt_tokens,
            "fusion_tokens": fusion_tokens,
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


def loss_for_output(output: dict[str, torch.Tensor], labels: torch.Tensor, batch: stage_q.PixelSet, model: StageUWideVLM) -> torch.Tensor:
    rank = F.cross_entropy(output["scores"], labels)
    router = F.binary_cross_entropy_with_logits(output["route_logits"], model.route_target(batch.required_experts))
    return rank + model.config.router_loss_weight * router


@torch.no_grad()
def evaluate_ranking(
    model: StageUWideVLM,
    data: stage_q.PixelSet,
    config: StageUConfig,
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
    true_scores: list[float] = []
    by_family: dict[str, dict[str, list[float]]] = {
        family: {"hits": [], "mrr": []}
        for family in FAMILIES
    }
    for start in range(0, len(data.examples), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(data.examples))))
        batch = data.subset(indices).to(device)
        rows, true_indices = candidate_rows_for_examples(batch.examples, candidate_pool, seed=seed + start)
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        output = model(batch, answers, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        order = output["scores"].argsort(dim=1, descending=True).detach().cpu().tolist()
        score_rows = output["scores"].detach().cpu().tolist()
        for example, row_order, true_index, row_scores in zip(batch.examples, order, true_indices, score_rows):
            rank = row_order.index(true_index) + 1
            hit = 1.0 if rank == 1 else 0.0
            mrr = 1.0 / rank
            family = example.source_file.split("/")[-1]
            hits.append(hit)
            reciprocal.append(mrr)
            true_scores.append(float(row_scores[true_index]))
            by_family[family]["hits"].append(hit)
            by_family[family]["mrr"].append(mrr)
    model.train()
    return {
        "rank_top1": statistics.fmean(hits),
        "rank_mrr": statistics.fmean(reciprocal),
        "true_answer_score": statistics.fmean(true_scores),
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
    model: StageUWideVLM,
    train: stage_q.PixelSet,
    val: stage_q.PixelSet,
    config: StageUConfig,
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
        rows, true_indices = candidate_rows_for_examples(batch.examples, candidate_pool, seed=rng.randrange(1_000_000_000))
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        train_zero_modalities: tuple[str, ...] = ()
        if model.use_object_experts and config.patch_context_dropout > 0.0 and rng.random() < config.patch_context_dropout:
            train_zero_modalities = ("patch_expert",)
        output = model(batch, answers, zero_modalities=train_zero_modalities)
        loss = loss_for_output(output, labels, batch, model)
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
    def __init__(self, config: StageUConfig, class_count: int) -> None:
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
    model: StageUWideVLM,
    stage_name: str,
    train: stage_q.PixelSet,
    test: stage_q.PixelSet,
    config: StageUConfig,
    *,
    answer_to_label: dict[str, int],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = TokenProbe(config, len(answer_to_label)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(config.probe_steps):
        batch = stage_q.random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        labels = answer_labels(batch, answer_to_label, device)
        with torch.no_grad():
            tokens = model.representations(batch)[stage_name].detach()
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
            tokens = model.representations(batch)[stage_name]
            pred = probe(tokens).argmax(dim=1)
            hits.extend((pred == labels).float().detach().cpu().tolist())
    return {
        "answer_exact": statistics.fmean(hits),
        "class_count": len(answer_to_label),
        "random_exact": 1.0 / len(answer_to_label),
        "training_seconds": round(time.perf_counter() - started, 3),
    }


def train_all_probes(
    model: StageUWideVLM,
    train: stage_q.PixelSet,
    test: stage_q.PixelSet,
    config: StageUConfig,
    *,
    answer_to_label: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    stages = [stage for stage in PROBE_STAGES if model.use_object_experts or stage not in {"object_slots", "spatial_tokens", "count_tokens"}]
    return {
        stage_name: train_reconstruction_probe(
            model,
            stage_name,
            train,
            test,
            config,
            answer_to_label=answer_to_label,
            seed=config.seed + 9000 + index * 37,
            device=device,
        )
        for index, stage_name in enumerate(stages)
    }


def prediction_samples(
    models: dict[str, StageUWideVLM],
    test: stage_q.PixelSet,
    config: StageUConfig,
    *,
    device: torch.device,
    candidate_pool: stage_q.CandidatePool,
) -> list[dict[str, object]]:
    selected = test.subset(list(range(min(8, len(test.examples)))))
    rows, true_indices = candidate_rows_for_examples(selected.examples, candidate_pool, seed=config.seed + 9000)
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


def run_experiment(config: StageUConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_u_object_slots device={device} seed={config.seed} output_tokens={config.output_tokens}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    pixel_sets, data_stats, meta = build_data(config)
    train = pixel_sets["train"].to(device)
    val = pixel_sets["val"].to(device)
    test = pixel_sets["test"].to(device)
    stage_r.write_samples(pixel_sets["test"], meta["test"], output_path.parent / "samples")
    candidate_pool = build_candidate_pool(config, device=device)
    answer_to_label = {answer: index for index, answer in enumerate(candidate_pool.answers)}

    models = {
        variant: StageUWideVLM(config, variant=variant)
        for variant in config.variants
    }
    training = {}
    for index, (variant, model) in enumerate(models.items()):
        training[variant] = train_model(
            model,
            train,
            val,
            config,
            candidate_pool=candidate_pool,
            steps=config.train_steps,
            seed=config.seed + 1100 + index * 97 + config.output_tokens,
            device=device,
            label=f"output{config.output_tokens}:{variant}",
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
        ablations[f"{variant}_no_image_modality"] = evaluate_ranking(
            model,
            test,
            config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4000 + index,
            zero_modalities=("image",),
        )
        ablations[f"{variant}_no_latent_access"] = evaluate_ranking(
            model,
            test,
            config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4100 + index,
            disable_latent_access=True,
        )
        ablations[f"{variant}_no_patch_expert"] = evaluate_ranking(
            model,
            test,
            config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4150 + index,
            zero_modalities=("patch_expert",),
        )
        if model.use_object_experts:
            ablations[f"{variant}_no_object_experts"] = evaluate_ranking(
                model,
                test,
                config,
                device=device,
                candidate_pool=candidate_pool,
                seed=config.seed + 4200 + index,
                zero_modalities=("object_experts",),
            )
            ablations[f"{variant}_no_spatial_expert"] = evaluate_ranking(
                model,
                test,
                config,
                device=device,
                candidate_pool=candidate_pool,
                seed=config.seed + 4300 + index,
                zero_modalities=("spatial_expert",),
            )
            ablations[f"{variant}_no_count_expert"] = evaluate_ranking(
                model,
                test,
                config,
                device=device,
                candidate_pool=candidate_pool,
                seed=config.seed + 4400 + index,
                zero_modalities=("count_expert",),
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
        "experiment": "omni_transformer_stage_u_object_slots",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "patch_wide_latent": "expanded visual input keeps an 8x8 patch grid as raw evidence tokens, plus prompt/fusion and output summary tokens",
            "object_slot_spatial_wide_latent": "adds object-slot, spatial and counting experts on top of the expanded patch expert",
            "latent_bus": "keeps raw expert tokens, route-weighted expert tokens and appends a swept number of output summary tokens",
            "output_token_sweep": "output_tokens controls the appended summary/output token count; raw evidence tokens are preserved",
            "patch_context_dropout": "during object-expert training, raw patch context is sometimes zeroed after object/spatial/count experts read it, testing whether expert tokens can carry the evidence",
            "experts": EXPERTS_U,
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
    output_counts = sorted({int(run["config"]["output_tokens"]) for run in runs})
    seeds = sorted({int(run["config"]["seed"]) for run in runs})
    summary: dict[str, object] = {}
    for output_tokens in output_counts:
        matching = [run for run in runs if int(run["config"]["output_tokens"]) == output_tokens]
        local_buckets: dict[str, list[float]] = {}
        for run in matching:
            for key, value in collect_numbers(
                {
                    "metrics": run["metrics"],
                    "ablations": run["ablations"],
                    "training": run["training"],
                    "prediction_cost": run["prediction_cost"],
                    "reconstruction": run["reconstruction"],
                }
            ).items():
                local_buckets.setdefault(key, []).append(value)
        local_stats = {key: stage_m.stat(values) for key, values in sorted(local_buckets.items())}

        def mean(path: str) -> float | None:
            item = local_stats.get(path)
            return None if item is None else float(item["mean"])

        variants = {}
        for variant in MODEL_VARIANTS:
            variants[variant] = {
                "rank_top1": mean(f"metrics.{variant}.rank_top1"),
                "no_image_top1": mean(f"ablations.{variant}_no_image_modality.rank_top1"),
                "no_latent_top1": mean(f"ablations.{variant}_no_latent_access.rank_top1"),
                "no_patch_expert_top1": mean(f"ablations.{variant}_no_patch_expert.rank_top1"),
                "train_seconds": mean(f"training.{variant}.training_seconds"),
                "params": mean(f"training.{variant}.trainable_parameter_count"),
                "prediction_ms_per_example": mean(f"prediction_cost.{variant}_prediction.prediction_ms_per_example"),
                "per_family": {
                    family: mean(f"metrics.{variant}.per_family.{family}.rank_top1")
                    for family in FAMILIES
                },
            }
            if variant == "object_slot_spatial_wide_latent":
                variants[variant]["no_object_experts_top1"] = mean(f"ablations.{variant}_no_object_experts.rank_top1")
                variants[variant]["no_spatial_expert_top1"] = mean(f"ablations.{variant}_no_spatial_expert.rank_top1")
                variants[variant]["no_count_expert_top1"] = mean(f"ablations.{variant}_no_count_expert.rank_top1")
                variants[variant]["wide_latent_probe"] = mean(f"reconstruction.{variant}.wide_latent.answer_exact")
        summary[str(output_tokens)] = variants
    return {
        "experiment": "omni_transformer_stage_u_object_slots_sweep",
        "run_count": len(runs),
        "seeds": seeds,
        "output_token_counts": output_counts,
        "stats": stats,
        "summary": summary,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_u_object_slots/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_u_object_slots/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_u_object_slots/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702")
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--output-token-counts", default="8,16,32")
    parser.add_argument("--variants", default="patch_wide_latent,object_slot_spatial_wide_latent")
    parser.add_argument("--train-size", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=384)
    parser.add_argument("--test-size", type=int, default=512)
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
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=520)
    parser.add_argument("--eval-every", type=int, default=130)
    parser.add_argument("--patch-context-dropout", type=float, default=0.25)
    parser.add_argument("--probe-steps", type=int, default=100)
    args = parser.parse_args()

    variants = parse_csv_strings(args.variants)
    unknown = set(variants) - set(MODEL_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    base_config = StageUConfig(
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
        patch_context_dropout=args.patch_context_dropout,
        probe_steps=args.probe_steps,
        variants=variants,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            for output_tokens in parse_csv_ints(args.output_token_counts):
                config = StageUConfig(**{**asdict(base_config), "seed": seed, "output_tokens": output_tokens})
                runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / f"out{output_tokens}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
