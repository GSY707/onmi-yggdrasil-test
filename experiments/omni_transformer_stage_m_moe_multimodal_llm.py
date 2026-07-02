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

from multimodal_fusion_latent_flow import IMAGE_SIZE, stat
from omni_transformer_stage_h import PATCH_COUNT, PATCH_SIZE, TransformerBlock
from visual_multimodal_stage_ab import write_png


COLORS = ("red", "green", "blue", "yellow")
RGB = (
    (0.92, 0.12, 0.10),
    (0.12, 0.78, 0.22),
    (0.16, 0.36, 0.95),
    (0.95, 0.78, 0.15),
)
SHAPES = ("circle", "square", "triangle")
ACTIONS = ("stop", "go", "inspect", "wait")
NUM_WORDS = ("zero", "one", "two", "three", "four")
TASKS = ("attribute", "counting", "spatial", "rule", "chart")
EXPERTS = ("vision", "text_rule", "spatial", "counting", "chart")

TASK_ATTRIBUTE = 0
TASK_COUNTING = 1
TASK_SPATIAL = 2
TASK_RULE = 3
TASK_CHART = 4

VISION_EXPERT = 0
TEXT_EXPERT = 1
SPATIAL_EXPERT = 2
COUNTING_EXPERT = 3
CHART_EXPERT = 4

CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789 .,?"
PAD = 0
BOS = 1
EOS = 2
CHAR_TO_ID = {char: index + 3 for index, char in enumerate(CHARSET)}
ID_TO_CHAR = {index: char for char, index in CHAR_TO_ID.items()}
VOCAB = len(CHAR_TO_ID) + 3

PROMPT_LEN = 96
ANSWER_LEN = 48
VISION_TOKENS = 10
TEXT_TOKENS = 8
SPATIAL_TOKENS = 5
COUNTING_TOKENS = 5
CHART_TOKENS = 5
LATENT_TOKENS = 8
SPECIAL_LATENT = 0
SPECIAL_DIRECT = 1
SPECIAL_ANSWER = 2
SPECIAL_COUNT = 3
TARGET_CLASSES = 16


@dataclass(frozen=True)
class MoEConfig:
    train_size: int = 8192
    val_size: int = 1024
    test_size: int = 1024
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    direct_steps: int = 900
    moe_steps: int = 1200
    probe_steps: int = 180
    eval_every: int = 400
    answer_loss_weight: float = 1.0
    router_loss_weight: float = 0.25
    thought_loss_weight: float = 0.20


@dataclass(frozen=True)
class Obj:
    color: int
    shape: int
    x: int
    y: int


@dataclass(frozen=True)
class Example:
    image: tuple[tuple[tuple[float, ...], ...], ...]
    prompt: str
    answer: str
    task: int
    target_class: int
    required_experts: tuple[float, ...]
    objects: tuple[Obj, ...]


@dataclass
class Batch:
    image: torch.Tensor
    prompt: torch.Tensor
    answer_input: torch.Tensor
    answer_target: torch.Tensor
    task: torch.Tensor
    target_class: torch.Tensor
    required_experts: torch.Tensor


def encode_text(value: str, length: int, *, add_eos: bool) -> list[int]:
    ids = [CHAR_TO_ID[char] for char in value.lower() if char in CHAR_TO_ID]
    if add_eos:
        ids = ids[: length - 1] + [EOS]
    else:
        ids = ids[:length]
    return ids + [PAD] * (length - len(ids))


def answer_inputs(answer_target: list[int]) -> list[int]:
    return [BOS] + answer_target[:-1]


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


def parse_semantic_target(text: str, task: int) -> int | None:
    normalized = f" {text.lower()} "
    if task == TASK_ATTRIBUTE:
        for idx, color in enumerate(COLORS):
            if f" {color} " in normalized:
                return idx
    if task == TASK_COUNTING:
        for idx, word in enumerate(NUM_WORDS):
            if f" {word} " in normalized:
                return idx
    if task == TASK_SPATIAL:
        if " yes " in normalized:
            return 1
        if " no " in normalized:
            return 0
    if task == TASK_RULE:
        for idx, action in enumerate(ACTIONS):
            if f" {action} " in normalized:
                return idx
    if task == TASK_CHART:
        for idx, name in enumerate("abcd"):
            if f" bar {name} " in normalized:
                return idx
    return None


def blank_image() -> torch.Tensor:
    image = torch.full((3, IMAGE_SIZE, IMAGE_SIZE), 0.055)
    image[:, 3:-3, 3:-3] = 0.11
    return image


def draw_square(image: torch.Tensor, obj: Obj) -> None:
    color = torch.tensor(RGB[obj.color]).view(3, 1, 1)
    image[:, obj.y : obj.y + 10, obj.x : obj.x + 10] = color


def draw_circle(image: torch.Tensor, obj: Obj) -> None:
    color = torch.tensor(RGB[obj.color]).view(3, 1)
    yy, xx = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    mask = ((yy - 4.5) ** 2 + (xx - 4.5) ** 2) <= 24.0
    patch = image[:, obj.y : obj.y + 10, obj.x : obj.x + 10]
    patch[:, mask] = color.expand(3, int(mask.sum()))


def draw_triangle(image: torch.Tensor, obj: Obj) -> None:
    color = torch.tensor(RGB[obj.color]).view(3, 1)
    yy, xx = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    mask = yy >= torch.abs(xx - 4.5)
    patch = image[:, obj.y : obj.y + 10, obj.x : obj.x + 10]
    patch[:, mask] = color.expand(3, int(mask.sum()))


def draw_object(image: torch.Tensor, obj: Obj) -> None:
    if obj.shape == 0:
        draw_circle(image, obj)
    elif obj.shape == 1:
        draw_square(image, obj)
    else:
        draw_triangle(image, obj)


def draw_chart(heights: tuple[int, int, int, int], colors: tuple[int, int, int, int]) -> torch.Tensor:
    image = blank_image()
    xs = (7, 17, 27, 37)
    for idx, (height, color_id) in enumerate(zip(heights, colors)):
        color = torch.tensor(RGB[color_id]).view(3, 1, 1)
        y1 = 40 - height
        image[:, y1:40, xs[idx] : xs[idx] + 6] = color
    return image


def render_objects(objects: tuple[Obj, ...]) -> torch.Tensor:
    image = blank_image()
    for obj in objects:
        draw_object(image, obj)
    return image


def image_tuple(image: torch.Tensor) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(tuple(tuple(float(value) for value in row) for row in channel) for channel in image)


def required(*expert_ids: int) -> tuple[float, ...]:
    values = [0.0 for _ in EXPERTS]
    for idx in expert_ids:
        values[idx] = 1.0
    return tuple(values)


def grid_positions(rng: random.Random) -> list[tuple[int, int]]:
    positions = [(7, 7), (30, 7), (7, 30), (30, 30)]
    rng.shuffle(positions)
    return positions


def make_attribute(index: int, rng: random.Random) -> Example:
    positions = grid_positions(rng)
    shape = rng.randrange(len(SHAPES))
    color = rng.randrange(len(COLORS))
    objects = (Obj(color=color, shape=shape, x=positions[0][0], y=positions[0][1]),)
    prompt = f"what color is the {SHAPES[shape]} ?"
    answer = f"the {SHAPES[shape]} is {COLORS[color]} ."
    return Example(
        image=image_tuple(render_objects(objects)),
        prompt=prompt,
        answer=answer,
        task=TASK_ATTRIBUTE,
        target_class=color,
        required_experts=required(VISION_EXPERT, TEXT_EXPERT),
        objects=objects,
    )


def make_counting(index: int, rng: random.Random) -> Example:
    positions = grid_positions(rng)
    target_color = rng.randrange(len(COLORS))
    target_shape = rng.randrange(len(SHAPES))
    count = rng.randrange(1, 4)
    objects: list[Obj] = []
    for item in range(count):
        objects.append(Obj(target_color, target_shape, positions[item][0], positions[item][1]))
    cursor = count
    while cursor < 4:
        color = rng.randrange(len(COLORS))
        shape = rng.randrange(len(SHAPES))
        if color == target_color and shape == target_shape:
            color = (color + 1) % len(COLORS)
        objects.append(Obj(color, shape, positions[cursor][0], positions[cursor][1]))
        cursor += 1
    rng.shuffle(objects)
    prompt = f"how many {COLORS[target_color]} {SHAPES[target_shape]}s are there ?"
    answer = f"there are {NUM_WORDS[count]} {COLORS[target_color]} {SHAPES[target_shape]}s ."
    return Example(
        image=image_tuple(render_objects(tuple(objects))),
        prompt=prompt,
        answer=answer,
        task=TASK_COUNTING,
        target_class=count,
        required_experts=required(VISION_EXPERT, TEXT_EXPERT, COUNTING_EXPERT),
        objects=tuple(objects),
    )


def make_spatial(index: int, rng: random.Random) -> Example:
    positions = grid_positions(rng)
    left_is_true = bool(rng.randrange(2))
    color_a = rng.randrange(len(COLORS))
    color_b = (color_a + 1 + rng.randrange(len(COLORS) - 1)) % len(COLORS)
    shape_a = rng.randrange(len(SHAPES))
    shape_b = (shape_a + 1 + rng.randrange(len(SHAPES) - 1)) % len(SHAPES)
    left_pos, right_pos = ((7, 18), (31, 18)) if left_is_true else ((31, 18), (7, 18))
    objects = (
        Obj(color_a, shape_a, left_pos[0], left_pos[1]),
        Obj(color_b, shape_b, right_pos[0], right_pos[1]),
    )
    prompt = f"is the {COLORS[color_a]} {SHAPES[shape_a]} left of the {COLORS[color_b]} {SHAPES[shape_b]} ?"
    answer = "yes ." if left_is_true else "no ."
    return Example(
        image=image_tuple(render_objects(objects)),
        prompt=prompt,
        answer=answer,
        task=TASK_SPATIAL,
        target_class=1 if left_is_true else 0,
        required_experts=required(VISION_EXPERT, TEXT_EXPERT, SPATIAL_EXPERT),
        objects=objects,
    )


def make_rule(index: int, rng: random.Random) -> Example:
    positions = grid_positions(rng)
    color = rng.randrange(len(COLORS))
    other = (color + 1 + rng.randrange(len(COLORS) - 1)) % len(COLORS)
    action = rng.randrange(len(ACTIONS))
    other_action = (action + 1 + rng.randrange(len(ACTIONS) - 1)) % len(ACTIONS)
    shape = rng.randrange(len(SHAPES))
    objects = (Obj(color, shape, positions[0][0], positions[0][1]),)
    prompt = (
        f"if {COLORS[color]} means {ACTIONS[action]} and {COLORS[other]} means {ACTIONS[other_action]} , "
        f"what should the {SHAPES[shape]} do ?"
    )
    answer = f"the {SHAPES[shape]} should {ACTIONS[action]} ."
    return Example(
        image=image_tuple(render_objects(objects)),
        prompt=prompt,
        answer=answer,
        task=TASK_RULE,
        target_class=action,
        required_experts=required(VISION_EXPERT, TEXT_EXPERT),
        objects=objects,
    )


def make_chart(index: int, rng: random.Random) -> Example:
    values = list(range(10, 30, 4))
    rng.shuffle(values)
    heights = tuple(values[:4])
    colors = tuple(rng.randrange(len(COLORS)) for _ in range(4))
    tallest = max(range(4), key=lambda idx: heights[idx])
    prompt = "which bar is tallest ?"
    answer = f"bar {'abcd'[tallest]} is tallest ."
    return Example(
        image=image_tuple(draw_chart(heights, colors)),
        prompt=prompt,
        answer=answer,
        task=TASK_CHART,
        target_class=tallest,
        required_experts=required(VISION_EXPERT, TEXT_EXPERT, CHART_EXPERT),
        objects=(),
    )


MAKERS = (make_attribute, make_counting, make_spatial, make_rule, make_chart)


def generate_examples(size: int, *, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples = [MAKERS[index % len(MAKERS)](index, rng) for index in range(size)]
    rng.shuffle(examples)
    return examples


def make_batch(examples: list[Example], *, device: torch.device) -> Batch:
    answer_target = [encode_text(example.answer, ANSWER_LEN, add_eos=True) for example in examples]
    return Batch(
        image=torch.tensor([example.image for example in examples], dtype=torch.float32, device=device),
        prompt=torch.tensor([encode_text(example.prompt, PROMPT_LEN, add_eos=False) for example in examples], dtype=torch.long, device=device),
        answer_input=torch.tensor([answer_inputs(target) for target in answer_target], dtype=torch.long, device=device),
        answer_target=torch.tensor(answer_target, dtype=torch.long, device=device),
        task=torch.tensor([example.task for example in examples], dtype=torch.long, device=device),
        target_class=torch.tensor([example.target_class for example in examples], dtype=torch.long, device=device),
        required_experts=torch.tensor([example.required_experts for example in examples], dtype=torch.float32, device=device),
    )


def random_batch(examples: list[Example], *, rng: random.Random, batch_size: int, device: torch.device) -> Batch:
    return make_batch(rng.choices(examples, k=batch_size), device=device)


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


class VisionExpert(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.patch = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, config.d_model)
        self.position = nn.Parameter(torch.randn(PATCH_COUNT, config.d_model) * 0.02)
        self.resampler = QueryResampler(config.d_model, config.heads, VISION_TOKENS, config.dropout)

    def patch_image(self, image: torch.Tensor) -> torch.Tensor:
        patches = image.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(image.shape[0], PATCH_COUNT, 3 * PATCH_SIZE * PATCH_SIZE)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(self.patch_image(image)) + self.position.unsqueeze(0)
        return self.resampler(tokens)


class TextRuleExpert(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, config.d_model)
        self.position = nn.Parameter(torch.randn(PROMPT_LEN, config.d_model) * 0.02)
        self.resampler = QueryResampler(config.d_model, config.heads, TEXT_TOKENS, config.dropout)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(prompt) + self.position.unsqueeze(0)
        return self.resampler(tokens)


class FunctionalExpert(nn.Module):
    def __init__(self, config: MoEConfig, token_count: int) -> None:
        super().__init__()
        self.resampler = QueryResampler(config.d_model, config.heads, token_count, config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(1)])

    def forward(self, vision_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        source = torch.cat((vision_tokens, text_tokens), dim=1)
        tokens = self.resampler(source)
        mask = noncausal_mask(tokens.shape[1], device=tokens.device)
        for block in self.blocks:
            tokens = block(tokens, mask)
        return tokens


class AttentionPump(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(LATENT_TOKENS, config.d_model) * 0.02)
        self.expert_type = nn.Embedding(len(EXPERTS), config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_kv = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

    def forward(self, expert_tokens: list[torch.Tensor], route_weights: torch.Tensor) -> torch.Tensor:
        weighted: list[torch.Tensor] = []
        for idx, tokens in enumerate(expert_tokens):
            typed = tokens + self.expert_type.weight[idx].view(1, 1, -1)
            weighted.append(typed * route_weights[:, idx].view(-1, 1, 1))
        source = torch.cat(weighted, dim=1)
        query = self.query.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attn(self.norm_q(query), self.norm_kv(source), self.norm_kv(source), need_weights=False)
        x = query + attended
        return x + self.ff(x)


class ThoughtExpert(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        mask = noncausal_mask(latent.shape[1], device=latent.device)
        for block in self.blocks:
            latent = block(latent, mask)
        return self.norm(latent)


class TextOutputExpert(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, config.d_model)
        self.special = nn.Embedding(SPECIAL_COUNT, config.d_model)
        max_context = VISION_TOKENS + TEXT_TOKENS + SPATIAL_TOKENS + COUNTING_TOKENS + CHART_TOKENS
        self.position = nn.Parameter(torch.randn(max_context + LATENT_TOKENS + ANSWER_LEN + 4, config.d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, VOCAB)

    def mask(self, context_len: int, answer_len: int, *, device: torch.device) -> torch.Tensor:
        seq_len = context_len + answer_len
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        mask[:context_len, context_len:] = True
        return mask

    def forward(self, context: torch.Tensor, answer_input: torch.Tensor, *, answer_context_marker: int) -> torch.Tensor:
        batch_size = int(answer_input.shape[0])
        marker = self.special.weight[answer_context_marker].view(1, 1, -1).expand(batch_size, 1, -1)
        answer_tokens = self.embedding(answer_input)
        x = torch.cat((marker, context, self.special.weight[SPECIAL_ANSWER].view(1, 1, -1).expand(batch_size, 1, -1), answer_tokens), dim=1)
        context_len = 1 + context.shape[1] + 1
        x = x + self.position[: x.shape[1]].unsqueeze(0)
        mask = self.mask(context_len, ANSWER_LEN, device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        hidden = self.norm(x)
        return self.head(hidden[:, context_len:])


class TinyMoEMultimodalLLM(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.vision = VisionExpert(config)
        self.text = TextRuleExpert(config)
        self.spatial = FunctionalExpert(config, SPATIAL_TOKENS)
        self.counting = FunctionalExpert(config, COUNTING_TOKENS)
        self.chart = FunctionalExpert(config, CHART_TOKENS)
        self.router = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(EXPERTS)),
        )
        self.pump = AttentionPump(config)
        self.thought = ThoughtExpert(config)
        self.output = TextOutputExpert(config)
        self.thought_task = nn.Linear(config.d_model, len(TASKS))
        self.thought_target = nn.Linear(config.d_model, TARGET_CLASSES)

    def expert_outputs(
        self,
        batch: Batch,
        *,
        zero_modalities: Iterable[str] = (),
        shuffle_modalities: Iterable[str] = (),
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        zero_set = set(zero_modalities)
        shuffle_set = set(shuffle_modalities)
        image = batch.image
        prompt = batch.prompt
        if "image" in shuffle_set and image.shape[0] > 1:
            image = torch.roll(image, shifts=1, dims=0)
        if "prompt" in shuffle_set and prompt.shape[0] > 1:
            prompt = torch.roll(prompt, shifts=1, dims=0)
        if "image" in zero_set:
            image = torch.zeros_like(image)
        if "prompt" in zero_set:
            prompt = torch.zeros_like(prompt)

        vision_tokens = self.vision(image)
        text_tokens = self.text(prompt)
        expert_tokens = [
            vision_tokens,
            text_tokens,
            self.spatial(vision_tokens, text_tokens),
            self.counting(vision_tokens, text_tokens),
            self.chart(vision_tokens, text_tokens),
        ]
        route_logits = self.router(torch.cat((vision_tokens.mean(dim=1), text_tokens.mean(dim=1)), dim=-1))
        return expert_tokens, route_logits

    def context(
        self,
        batch: Batch,
        *,
        mode: str,
        zero_modalities: Iterable[str] = (),
        zero_experts: Iterable[str] = (),
        shuffle_modalities: Iterable[str] = (),
        wrong_route: bool = False,
        disable_latent_access: bool = False,
        return_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        expert_tokens, route_logits = self.expert_outputs(
            batch,
            zero_modalities=zero_modalities,
            shuffle_modalities=shuffle_modalities,
        )
        route_weights = torch.sigmoid(route_logits)
        if wrong_route and route_weights.shape[0] > 1:
            route_weights = torch.roll(route_weights, shifts=1, dims=0)
        zero_expert_set = set(zero_experts)
        for idx, name in enumerate(EXPERTS):
            if name in zero_expert_set:
                route_weights[:, idx] = 0.0
                expert_tokens[idx] = torch.zeros_like(expert_tokens[idx])

        if mode == "direct":
            context = torch.cat([tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)], dim=1)
            latent = torch.zeros(batch.image.shape[0], LATENT_TOKENS, context.shape[-1], device=context.device)
        elif mode == "moe_latent":
            pumped = self.pump(expert_tokens, route_weights)
            latent = self.thought(pumped)
            context = torch.zeros_like(latent) if disable_latent_access else latent
        else:
            raise ValueError(f"unknown mode: {mode}")
        result = {"context": context, "route_logits": route_logits, "route_weights": route_weights}
        if return_latent:
            result["latent"] = latent
        return result

    def forward(
        self,
        batch: Batch,
        *,
        mode: str,
        zero_modalities: Iterable[str] = (),
        zero_experts: Iterable[str] = (),
        shuffle_modalities: Iterable[str] = (),
        wrong_route: bool = False,
        disable_latent_access: bool = False,
        return_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(
            batch,
            mode=mode,
            zero_modalities=zero_modalities,
            zero_experts=zero_experts,
            shuffle_modalities=shuffle_modalities,
            wrong_route=wrong_route,
            disable_latent_access=disable_latent_access,
            return_latent=True,
        )
        marker = SPECIAL_DIRECT if mode == "direct" else SPECIAL_LATENT
        logits = self.output(ctx["context"], batch.answer_input, answer_context_marker=marker)
        latent_summary = ctx["latent"].mean(dim=1)
        output = {
            "logits": logits,
            "route_logits": ctx["route_logits"],
            "route_weights": ctx["route_weights"],
            "thought_task": self.thought_task(latent_summary),
            "thought_target": self.thought_target(latent_summary),
        }
        if return_latent:
            output["latent"] = ctx["latent"]
        return output


def masked_token_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    mask = target != PAD
    return float(((pred == target) & mask).sum().detach().cpu()) / max(int(mask.sum().detach().cpu()), 1)


def answer_exact_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    mask = target != PAD
    exact = ((pred == target) | ~mask).all(dim=1)
    return float(exact.float().mean().detach().cpu())


@torch.no_grad()
def evaluate_teacher_forced(
    model: TinyMoEMultimodalLLM,
    examples: list[Example],
    config: MoEConfig,
    *,
    device: torch.device,
    mode: str,
    zero_modalities: Iterable[str] = (),
    zero_experts: Iterable[str] = (),
    shuffle_modalities: Iterable[str] = (),
    wrong_route: bool = False,
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    token_hits: list[float] = []
    exacts: list[float] = []
    semantic_hits: list[float] = []
    route_hits: list[float] = []
    task_values: dict[str, list[float]] = {name: [] for name in TASKS}
    for start in range(0, len(examples), config.batch_size):
        chunk = examples[start : start + config.batch_size]
        batch = make_batch(chunk, device=device)
        output = model(
            batch,
            mode=mode,
            zero_modalities=zero_modalities,
            zero_experts=zero_experts,
            shuffle_modalities=shuffle_modalities,
            wrong_route=wrong_route,
            disable_latent_access=disable_latent_access,
        )
        logits = output["logits"]
        pred = logits.argmax(dim=-1)
        mask = batch.answer_target != PAD
        exact = ((pred == batch.answer_target) | ~mask).all(dim=1).detach().cpu().tolist()
        route_pred = (torch.sigmoid(output["route_logits"]) >= 0.5).float()
        route_exact = (route_pred == batch.required_experts).all(dim=1).float().detach().cpu().tolist()
        token_hits.append(masked_token_accuracy(logits, batch.answer_target))
        exacts.extend(1.0 if item else 0.0 for item in exact)
        route_hits.extend(route_exact)
        for example, hit, predicted in zip(chunk, exact, pred.detach().cpu().tolist()):
            parsed = parse_semantic_target(decode_tokens(predicted), example.task)
            semantic_hits.append(1.0 if parsed == example.target_class else 0.0)
            task_values[TASKS[example.task]].append(1.0 if hit else 0.0)
    model.train()
    return {
        "answer_exact": statistics.fmean(exacts),
        "answer_semantic_accuracy": statistics.fmean(semantic_hits),
        "token_accuracy": statistics.fmean(token_hits),
        "router_exact": statistics.fmean(route_hits),
        "answer_exact_by_task": {key: statistics.fmean(values) if values else 0.0 for key, values in task_values.items()},
    }


@torch.no_grad()
def greedy_decode(
    model: TinyMoEMultimodalLLM,
    batch: Batch,
    *,
    mode: str,
    zero_modalities: Iterable[str] = (),
    zero_experts: Iterable[str] = (),
    shuffle_modalities: Iterable[str] = (),
    wrong_route: bool = False,
    disable_latent_access: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ctx = model.context(
        batch,
        mode=mode,
        zero_modalities=zero_modalities,
        zero_experts=zero_experts,
        shuffle_modalities=shuffle_modalities,
        wrong_route=wrong_route,
        disable_latent_access=disable_latent_access,
        return_latent=True,
    )
    marker = SPECIAL_DIRECT if mode == "direct" else SPECIAL_LATENT
    answer_input = torch.full_like(batch.answer_input, PAD)
    answer_input[:, 0] = BOS
    generated = torch.full_like(batch.answer_target, PAD)
    done = torch.zeros(batch.answer_target.shape[0], dtype=torch.bool, device=batch.answer_target.device)
    for pos in range(ANSWER_LEN):
        logits = model.output(ctx["context"], answer_input, answer_context_marker=marker)
        next_token = logits[:, pos].argmax(dim=-1)
        next_token = torch.where(done, torch.full_like(next_token, PAD), next_token)
        generated[:, pos] = next_token
        done = done | (next_token == EOS)
        if pos + 1 < ANSWER_LEN:
            answer_input[:, pos + 1] = torch.where(done, torch.full_like(next_token, PAD), next_token)
    return generated, ctx


@torch.no_grad()
def evaluate_greedy(
    model: TinyMoEMultimodalLLM,
    examples: list[Example],
    config: MoEConfig,
    *,
    device: torch.device,
    mode: str,
    zero_modalities: Iterable[str] = (),
    zero_experts: Iterable[str] = (),
    shuffle_modalities: Iterable[str] = (),
    wrong_route: bool = False,
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    token_hits: list[float] = []
    exacts: list[float] = []
    semantic_hits: list[float] = []
    route_hits: list[float] = []
    task_values: dict[str, list[float]] = {name: [] for name in TASKS}
    for start in range(0, len(examples), config.batch_size):
        chunk = examples[start : start + config.batch_size]
        batch = make_batch(chunk, device=device)
        pred, ctx = greedy_decode(
            model,
            batch,
            mode=mode,
            zero_modalities=zero_modalities,
            zero_experts=zero_experts,
            shuffle_modalities=shuffle_modalities,
            wrong_route=wrong_route,
            disable_latent_access=disable_latent_access,
        )
        mask = batch.answer_target != PAD
        exact = ((pred == batch.answer_target) | ~mask).all(dim=1).detach().cpu().tolist()
        route_pred = (torch.sigmoid(ctx["route_logits"]) >= 0.5).float()
        route_exact = (route_pred == batch.required_experts).all(dim=1).float().detach().cpu().tolist()
        token_hits.append(float(((pred == batch.answer_target) & mask).sum().detach().cpu()) / max(int(mask.sum().detach().cpu()), 1))
        exacts.extend(1.0 if item else 0.0 for item in exact)
        route_hits.extend(route_exact)
        for example, hit, predicted in zip(chunk, exact, pred.detach().cpu().tolist()):
            parsed = parse_semantic_target(decode_tokens(predicted), example.task)
            semantic_hits.append(1.0 if parsed == example.target_class else 0.0)
            task_values[TASKS[example.task]].append(1.0 if hit else 0.0)
    model.train()
    return {
        "answer_exact": statistics.fmean(exacts),
        "answer_semantic_accuracy": statistics.fmean(semantic_hits),
        "token_accuracy": statistics.fmean(token_hits),
        "router_exact": statistics.fmean(route_hits),
        "answer_exact_by_task": {key: statistics.fmean(values) if values else 0.0 for key, values in task_values.items()},
    }


def loss_for_output(output: dict[str, torch.Tensor], batch: Batch, config: MoEConfig, *, mode: str) -> torch.Tensor:
    answer_loss = F.cross_entropy(output["logits"].view(-1, VOCAB), batch.answer_target.view(-1), ignore_index=PAD)
    router_loss = F.binary_cross_entropy_with_logits(output["route_logits"], batch.required_experts)
    loss = config.answer_loss_weight * answer_loss + config.router_loss_weight * router_loss
    if mode == "moe_latent":
        thought_loss = F.cross_entropy(output["thought_task"], batch.task) + F.cross_entropy(output["thought_target"], batch.target_class)
        loss = loss + config.thought_loss_weight * thought_loss
    return loss


def train_model(
    model: TinyMoEMultimodalLLM,
    train_examples: list[Example],
    val_examples: list[Example],
    config: MoEConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    mode: str,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_batch(train_examples, rng=rng, batch_size=config.batch_size, device=device)
        output = model(batch, mode=mode)
        loss = loss_for_output(output, batch, config, mode=mode)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_teacher_forced(model, val_examples, config, device=device, mode=mode)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **{k: v for k, v in metrics.items() if isinstance(v, float)}})
            print(
                f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"exact={metrics['answer_exact']:.3f} token={metrics['token_accuracy']:.3f} route={metrics['router_exact']:.3f}",
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
        self.task = nn.Linear(d_model, len(TASKS))
        self.target = nn.Linear(d_model, TARGET_CLASSES)
        self.route = nn.Linear(d_model, len(EXPERTS))

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"task": self.task(latent), "target": self.target(latent), "route": self.route(latent)}


def train_latent_probe(
    model: TinyMoEMultimodalLLM,
    train_examples: list[Example],
    test_examples: list[Example],
    config: MoEConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = LatentProbe(config.d_model).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr)
    rng = random.Random(config.seed + 50000)
    started = time.perf_counter()
    for _ in range(config.probe_steps):
        batch = random_batch(train_examples, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            output = model(batch, mode="moe_latent", return_latent=True)
            latent = output["latent"].mean(dim=1)
        logits = probe(latent.detach())
        loss = (
            F.cross_entropy(logits["task"], batch.task)
            + F.cross_entropy(logits["target"], batch.target_class)
            + F.binary_cross_entropy_with_logits(logits["route"], batch.required_experts)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    totals = {"task": 0.0, "target": 0.0, "route_exact": 0.0}
    count = 0
    with torch.no_grad():
        for start in range(0, len(test_examples), config.batch_size):
            batch = make_batch(test_examples[start : start + config.batch_size], device=device)
            output = model(batch, mode="moe_latent", return_latent=True)
            latent = output["latent"].mean(dim=1)
            logits = probe(latent)
            totals["task"] += float((logits["task"].argmax(dim=-1) == batch.task).sum().detach().cpu())
            totals["target"] += float((logits["target"].argmax(dim=-1) == batch.target_class).sum().detach().cpu())
            route = (torch.sigmoid(logits["route"]) >= 0.5).float()
            totals["route_exact"] += float((route == batch.required_experts).all(dim=1).float().sum().detach().cpu())
            count += len(batch.task)
    return {
        "training_seconds": round(time.perf_counter() - started, 3),
        "accuracies": {key: value / count for key, value in totals.items()},
    }


def write_samples(examples: list[Example], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [torch.tensor(example.image, dtype=torch.float32) for example in examples[:8]]
    first_row = torch.cat(images[:4], dim=2)
    second_row = torch.cat(images[4:8], dim=2)
    write_png(output_dir / "moe_multimodal_grid.png", torch.cat([first_row, second_row], dim=1))
    preview = [
        {
            "prompt": example.prompt,
            "answer": example.answer,
            "task": TASKS[example.task],
            "target_class": example.target_class,
            "required_experts": [EXPERTS[idx] for idx, value in enumerate(example.required_experts) if value > 0],
        }
        for example in examples[:16]
    ]
    (output_dir / "samples.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")


@torch.no_grad()
def prediction_samples(
    model: TinyMoEMultimodalLLM,
    examples: list[Example],
    config: MoEConfig,
    *,
    device: torch.device,
    mode: str,
    limit: int = 12,
) -> list[dict[str, object]]:
    model.eval()
    selected = examples[:limit]
    batch = make_batch(selected, device=device)
    output = model(batch, mode=mode)
    pred = output["logits"].argmax(dim=-1).detach().cpu().tolist()
    greedy_pred, _ = greedy_decode(model, batch, mode=mode)
    greedy_pred = greedy_pred.detach().cpu().tolist()
    route_weights = output["route_weights"].detach().cpu().tolist()
    rows: list[dict[str, object]] = []
    for example, predicted, generated, weights in zip(selected, pred, greedy_pred, route_weights):
        teacher_text = decode_tokens(predicted)
        greedy_text = decode_tokens(generated)
        rows.append(
            {
                "task": TASKS[example.task],
                "prompt": example.prompt,
                "expected": example.answer,
                "expected_target": example.target_class,
                "predicted_teacher_forced": teacher_text,
                "predicted_teacher_forced_target": parse_semantic_target(teacher_text, example.task),
                "predicted_greedy": greedy_text,
                "predicted_greedy_target": parse_semantic_target(greedy_text, example.task),
                "required_experts": [EXPERTS[idx] for idx, value in enumerate(example.required_experts) if value > 0],
                "route_weights": {name: round(float(value), 3) for name, value in zip(EXPERTS, weights)},
            }
        )
    model.train()
    return rows


def run_experiment(config: MoEConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_m_moe_multimodal_llm device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_examples = generate_examples(config.train_size, seed=config.seed + 1)
    val_examples = generate_examples(config.val_size, seed=config.seed + 2)
    test_examples = generate_examples(config.test_size, seed=config.seed + 3)
    write_samples(test_examples, output_dir=output_path.parent / "samples" / f"seed{config.seed}")

    direct = TinyMoEMultimodalLLM(config)
    moe = TinyMoEMultimodalLLM(config)
    training = {
        "direct_routed_experts": train_model(
            direct,
            train_examples,
            val_examples,
            config,
            steps=config.direct_steps,
            seed=config.seed + 10,
            device=device,
            mode="direct",
            label="direct_routed",
        ),
        "moe_latent_bottleneck": train_model(
            moe,
            train_examples,
            val_examples,
            config,
            steps=config.moe_steps,
            seed=config.seed + 20,
            device=device,
            mode="moe_latent",
            label="moe_latent",
        ),
    }

    metrics = {
        "direct_routed_experts": evaluate_teacher_forced(direct, test_examples, config, device=device, mode="direct"),
        "moe_latent_bottleneck": evaluate_teacher_forced(moe, test_examples, config, device=device, mode="moe_latent"),
    }
    generation_metrics = {
        "direct_routed_experts": evaluate_greedy(direct, test_examples, config, device=device, mode="direct"),
        "moe_latent_bottleneck": evaluate_greedy(moe, test_examples, config, device=device, mode="moe_latent"),
    }
    ablations = {
        "no_image": evaluate_teacher_forced(moe, test_examples, config, device=device, mode="moe_latent", zero_modalities=("image",)),
        "no_prompt": evaluate_teacher_forced(moe, test_examples, config, device=device, mode="moe_latent", zero_modalities=("prompt",)),
        "shuffled_image": evaluate_teacher_forced(moe, test_examples, config, device=device, mode="moe_latent", shuffle_modalities=("image",)),
        "wrong_route": evaluate_teacher_forced(moe, test_examples, config, device=device, mode="moe_latent", wrong_route=True),
        "no_latent_access": evaluate_teacher_forced(
            moe,
            test_examples,
            config,
            device=device,
            mode="moe_latent",
            disable_latent_access=True,
        ),
        **{
            f"no_{name}_expert": evaluate_teacher_forced(
                moe,
                test_examples,
                config,
                device=device,
                mode="moe_latent",
                zero_experts=(name,),
            )
            for name in EXPERTS
        },
    }
    latent_probe = train_latent_probe(moe, train_examples, test_examples, config, device=device)
    output = {
        "experiment": "omni_transformer_stage_m_moe_multimodal_llm",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "experts": EXPERTS,
            "expert_tokens": {
                "vision": VISION_TOKENS,
                "text_rule": TEXT_TOKENS,
                "spatial": SPATIAL_TOKENS,
                "counting": COUNTING_TOKENS,
                "chart": CHART_TOKENS,
            },
            "latent_tokens": LATENT_TOKENS,
            "output": "character-level natural language answer",
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "metrics": metrics,
        "generation_metrics": generation_metrics,
        "ablations": ablations,
        "latent_probe": latent_probe,
        "prediction_samples": {
            "direct_routed_experts": prediction_samples(direct, test_examples, config, device=device, mode="direct"),
            "moe_latent_bottleneck": prediction_samples(moe, test_examples, config, device=device, mode="moe_latent"),
        },
        "training": training,
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
                "generation_metrics": run["generation_metrics"],
                "ablations": run["ablations"],
                "latent_probe": run["latent_probe"],
            }
        )
        for key, value in numbers.items():
            if "training_seconds" not in key:
                buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_m_moe_multimodal_llm_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_m_moe_multimodal_llm/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_m_moe_multimodal_llm/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--direct-steps", type=int, default=900)
    parser.add_argument("--moe-steps", type=int, default=1200)
    parser.add_argument("--probe-steps", type=int, default=180)
    args = parser.parse_args()

    base_config = MoEConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        direct_steps=args.direct_steps,
        moe_steps=args.moe_steps,
        probe_steps=args.probe_steps,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = MoEConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
