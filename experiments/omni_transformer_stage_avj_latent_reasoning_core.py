from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omni_transformer_stage_ac_latent_reasoning import (  # noqa: E402
    BOS,
    COL_WORDS,
    COLORS,
    EOS,
    PAD,
    RELATIONS,
    ROW_WORDS,
    SHAPES,
)


FAMILIES = ("conditional_recolor", "same_row_move", "count_delete_or_add")
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six")
WORDS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "if",
    "the",
    "is",
    "of",
    "then",
    "else",
    "change",
    "color",
    "to",
    "move",
    "object",
    "same",
    "row",
    "as",
    "nearest",
    "column",
    "count",
    "greater",
    "than",
    "delete",
    "add",
    "at",
    "answer",
    "yes",
    "no",
    "done",
    *COLORS,
    *SHAPES,
    *ROW_WORDS,
    *COL_WORDS,
    *RELATIONS,
    *COUNT_WORDS,
)
WORD_TO_ID = {word: index for index, word in enumerate(WORDS)}
VOCAB_SIZE = len(WORDS)


@dataclass(frozen=True)
class StageAVJConfig:
    train_size: int = 4096
    val_size: int = 512
    test_size: int = 512
    heldout_size: int = 512
    batch_size: int = 128
    eval_batch_size: int = 256
    seed: int = 20260706
    grid_size: int = 4
    max_objects: int = 6
    prompt_len: int = 40
    answer_len: int = 6
    d_model: int = 192
    layers: int = 3
    heads: int = 4
    lr: float = 8e-4
    codec_steps: int = 200
    operation_steps: int = 400
    process_steps: int = 600
    joint_steps: int = 300
    eval_every: int = 200
    save_every: int = 500
    amp: bool = True
    gpu_resident_data: bool = True
    slot_loss_weight: float = 1.0
    source_recon_weight: float = 1.0
    target_record_weight: float = 1.0
    answer_token_weight: float = 0.5
    trace_loss_weight: float = 0.5
    process_loss_weight: float = 0.5

    @property
    def total_steps(self) -> int:
        return self.codec_steps + self.operation_steps + self.process_steps + self.joint_steps


@dataclass(frozen=True)
class SceneObject:
    active: int
    color: int
    shape: int
    row: int
    col: int


@dataclass
class AVJSet:
    prompts: torch.Tensor
    source_active: torch.Tensor
    source_color: torch.Tensor
    source_shape: torch.Tensor
    source_row: torch.Tensor
    source_col: torch.Tensor
    target_active: torch.Tensor
    target_color: torch.Tensor
    target_shape: torch.Tensor
    target_row: torch.Tensor
    target_col: torch.Tensor
    family: torch.Tensor
    read_a: torch.Tensor
    read_b: torch.Tensor
    edit_slot: torch.Tensor
    condition: torch.Tensor
    edit_action: torch.Tensor
    edit_color: torch.Tensor
    edit_row: torch.Tensor
    edit_col: torch.Tensor
    answer: torch.Tensor
    examples: list[dict[str, object]]

    def subset(self, indices: list[int]) -> "AVJSet":
        return AVJSet(
            prompts=self.prompts[indices],
            source_active=self.source_active[indices],
            source_color=self.source_color[indices],
            source_shape=self.source_shape[indices],
            source_row=self.source_row[indices],
            source_col=self.source_col[indices],
            target_active=self.target_active[indices],
            target_color=self.target_color[indices],
            target_shape=self.target_shape[indices],
            target_row=self.target_row[indices],
            target_col=self.target_col[indices],
            family=self.family[indices],
            read_a=self.read_a[indices],
            read_b=self.read_b[indices],
            edit_slot=self.edit_slot[indices],
            condition=self.condition[indices],
            edit_action=self.edit_action[indices],
            edit_color=self.edit_color[indices],
            edit_row=self.edit_row[indices],
            edit_col=self.edit_col[indices],
            answer=self.answer[indices],
            examples=[self.examples[index] for index in indices],
        )

    def to(self, device: torch.device) -> "AVJSet":
        long_names = (
            "prompts",
            "source_active",
            "source_color",
            "source_shape",
            "source_row",
            "source_col",
            "target_active",
            "target_color",
            "target_shape",
            "target_row",
            "target_col",
            "family",
            "read_a",
            "read_b",
            "edit_slot",
            "condition",
            "edit_action",
            "edit_color",
            "edit_row",
            "edit_col",
            "answer",
        )
        kwargs = {name: getattr(self, name).to(device=device, dtype=torch.long) for name in long_names}
        kwargs["examples"] = self.examples
        return AVJSet(**kwargs)


def encode_words(words: list[str], length: int) -> list[int]:
    ids = [BOS] + [WORD_TO_ID[word] for word in words] + [EOS]
    ids = ids[:length]
    return ids + [PAD] * (length - len(ids))


def pair_index(color: int, shape: int) -> int:
    return color * len(SHAPES) + shape


def relation_holds(left: SceneObject, relation: int, right: SceneObject) -> bool:
    name = RELATIONS[relation]
    if name == "left":
        return left.col < right.col
    if name == "right":
        return left.col > right.col
    if name == "above":
        return left.row < right.row
    if name == "below":
        return left.row > right.row
    raise ValueError(name)


def sample_scene(rng: random.Random, config: StageAVJConfig, *, heldout: bool) -> list[SceneObject]:
    count = rng.randint(4, config.max_objects - 1)
    cells = [(row, col) for row in range(config.grid_size) for col in range(config.grid_size)]
    rng.shuffle(cells)
    objects: list[SceneObject] = []
    for index in range(count):
        color = rng.randrange(len(COLORS))
        shape = rng.randrange(len(SHAPES))
        if heldout:
            color = (color + index + 1) % len(COLORS)
            shape = (shape + index) % len(SHAPES)
        row, col = cells[index]
        objects.append(SceneObject(1, color, shape, row, col))
    while len(objects) < config.max_objects:
        objects.append(SceneObject(0, 0, 0, 0, 0))
    return objects


def record_tensors(objects: list[SceneObject]) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    return (
        [obj.active for obj in objects],
        [obj.color for obj in objects],
        [obj.shape for obj in objects],
        [obj.row for obj in objects],
        [obj.col for obj in objects],
    )


def answer_tokens(family: str, condition: bool, action: str) -> list[int]:
    if family == "conditional_recolor":
        return encode_words(["answer", "yes" if condition else "no"], 6)
    if family == "same_row_move":
        return encode_words(["answer", "done"], 6)
    return encode_words(["answer", action], 6)


def make_example(index: int, rng: random.Random, config: StageAVJConfig, split: str) -> dict[str, object]:
    heldout = split == "heldout"
    source = sample_scene(rng, config, heldout=heldout)
    active_indices = [i for i, obj in enumerate(source) if obj.active]
    family_id = index % len(FAMILIES)
    family = FAMILIES[family_id]
    target = list(source)
    condition = 1
    read_a = active_indices[index % len(active_indices)]
    read_b = active_indices[(index + 1) % len(active_indices)]
    edit_slot = read_a
    edit_action = 0
    edit_color = -100
    edit_row = -100
    edit_col = -100

    if family == "conditional_recolor":
        left = source[read_a]
        right = source[read_b]
        relation = index % len(RELATIONS)
        condition = int(relation_holds(left, relation, right))
        edit_slot = read_a if condition else read_b
        new_color = (source[edit_slot].color + 1 + (index % (len(COLORS) - 1))) % len(COLORS)
        target[edit_slot] = SceneObject(1, new_color, source[edit_slot].shape, source[edit_slot].row, source[edit_slot].col)
        edit_action = 1
        edit_color = new_color
        prompt = [
            "if",
            "the",
            COLORS[left.color],
            SHAPES[left.shape],
            "is",
            RELATIONS[relation],
            "of",
            "the",
            COLORS[right.color],
            SHAPES[right.shape],
            "then",
            "change",
            "object",
            "color",
            "to",
            COLORS[new_color],
        ]
        answer = answer_tokens(family, bool(condition), "done")
    elif family == "same_row_move":
        anchor = source[read_a]
        candidates = [idx for idx in active_indices if idx != read_a and source[idx].row == anchor.row]
        if not candidates:
            candidates = [idx for idx in active_indices if idx != read_a]
        edit_slot = min(candidates, key=lambda idx: abs(source[idx].col - anchor.col))
        used = {(obj.row, obj.col) for obj in source if obj.active and obj is not source[edit_slot]}
        free_cells = [
            (row, col)
            for row in range(config.grid_size)
            for col in range(config.grid_size)
            if (row, col) not in used and (row, col) != (source[edit_slot].row, source[edit_slot].col)
        ]
        new_row, new_col = free_cells[(index + len(candidates)) % len(free_cells)]
        old = source[edit_slot]
        target[edit_slot] = SceneObject(1, old.color, old.shape, new_row, new_col)
        edit_action = 2
        edit_row = new_row
        edit_col = new_col
        read_b = edit_slot
        prompt = [
            "move",
            "the",
            "object",
            "same",
            "row",
            "as",
            "the",
            COLORS[anchor.color],
            SHAPES[anchor.shape],
            "nearest",
            "column",
            "to",
            ROW_WORDS[new_row],
            COL_WORDS[new_col],
        ]
        answer = answer_tokens(family, True, "done")
    else:
        pair_obj = source[read_a]
        target_pair = pair_index(pair_obj.color, pair_obj.shape)
        count = sum(1 for obj in source if obj.active and pair_index(obj.color, obj.shape) == target_pair)
        condition = int(count > 1)
        if condition:
            matching = [idx for idx in active_indices if pair_index(source[idx].color, source[idx].shape) == target_pair]
            edit_slot = max(matching, key=lambda idx: source[idx].col)
            old = source[edit_slot]
            target[edit_slot] = SceneObject(0, old.color, old.shape, old.row, old.col)
            edit_action = 3
            action_word = "delete"
        else:
            inactive = [idx for idx in range(config.max_objects) if not source[idx].active]
            edit_slot = inactive[0]
            used = {(obj.row, obj.col) for obj in source if obj.active}
            free = [(row, col) for row in range(config.grid_size) for col in range(config.grid_size) if (row, col) not in used]
            row, col = free[index % len(free)]
            target[edit_slot] = SceneObject(1, pair_obj.color, pair_obj.shape, row, col)
            edit_action = 4
            edit_row = row
            edit_col = col
            edit_color = pair_obj.color
            action_word = "add"
        read_b = edit_slot
        prompt = [
            "if",
            "count",
            COLORS[pair_obj.color],
            SHAPES[pair_obj.shape],
            "greater",
            "than",
            "one",
            "then",
            "delete",
            "else",
            "add",
        ]
        answer = answer_tokens(family, bool(condition), action_word)

    return {
        "prompt": encode_words(prompt, config.prompt_len),
        "source": record_tensors(source),
        "target": record_tensors(target),
        "family": family_id,
        "read_a": read_a,
        "read_b": read_b,
        "edit_slot": edit_slot,
        "condition": condition,
        "edit_action": edit_action,
        "edit_color": edit_color,
        "edit_row": edit_row,
        "edit_col": edit_col,
        "answer": answer,
        "example": {
            "split": split,
            "index": index,
            "family": family,
            "prompt_words": prompt,
            "read_a": read_a,
            "read_b": read_b,
            "edit_slot": edit_slot,
            "condition": condition,
            "edit_action": edit_action,
        },
    }


def build_dataset(split: str, size: int, config: StageAVJConfig) -> AVJSet:
    rng = random.Random(config.seed + {"train": 0, "val": 100_000, "test": 200_000, "heldout": 300_000}[split])
    rows = [make_example(index, rng, config, split) for index in range(size)]
    source = [row["source"] for row in rows]
    target = [row["target"] for row in rows]

    def tensor(name: str) -> torch.Tensor:
        return torch.tensor([row[name] for row in rows], dtype=torch.long)

    return AVJSet(
        prompts=tensor("prompt"),
        source_active=torch.tensor([item[0] for item in source], dtype=torch.long),
        source_color=torch.tensor([item[1] for item in source], dtype=torch.long),
        source_shape=torch.tensor([item[2] for item in source], dtype=torch.long),
        source_row=torch.tensor([item[3] for item in source], dtype=torch.long),
        source_col=torch.tensor([item[4] for item in source], dtype=torch.long),
        target_active=torch.tensor([item[0] for item in target], dtype=torch.long),
        target_color=torch.tensor([item[1] for item in target], dtype=torch.long),
        target_shape=torch.tensor([item[2] for item in target], dtype=torch.long),
        target_row=torch.tensor([item[3] for item in target], dtype=torch.long),
        target_col=torch.tensor([item[4] for item in target], dtype=torch.long),
        family=tensor("family"),
        read_a=tensor("read_a"),
        read_b=tensor("read_b"),
        edit_slot=tensor("edit_slot"),
        condition=tensor("condition"),
        edit_action=tensor("edit_action"),
        edit_color=tensor("edit_color"),
        edit_row=tensor("edit_row"),
        edit_col=tensor("edit_col"),
        answer=tensor("answer"),
        examples=[row["example"] for row in rows],
    )


class StageAVJModel(nn.Module):
    def __init__(self, config: StageAVJConfig) -> None:
        super().__init__()
        self.config = config
        feature_size = 1 + len(COLORS) + len(SHAPES) + config.grid_size + config.grid_size
        self.slot_schema = nn.Parameter(torch.randn(config.max_objects, config.d_model) * 0.02)
        self.object_encoder = nn.Sequential(
            nn.Linear(feature_size, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.slot_norm = nn.LayerNorm(config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.workspace = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.token_embed = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.prompt_pos = nn.Parameter(torch.randn(config.prompt_len, config.d_model) * 0.02)
        self.prompt_encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.prompt_norm = nn.LayerNorm(config.d_model)
        self.read_a_query = nn.Linear(config.d_model, config.d_model)
        self.read_b_query = nn.Linear(config.d_model, config.d_model)
        self.edit_query = nn.Linear(config.d_model, config.d_model)
        self.process = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.condition_head = nn.Linear(config.d_model, 2)
        self.action_head = nn.Linear(config.d_model, 5)
        self.edit_color_head = nn.Linear(config.d_model, len(COLORS))
        self.edit_row_head = nn.Linear(config.d_model, config.grid_size)
        self.edit_col_head = nn.Linear(config.d_model, config.grid_size)
        self.source_active_head = nn.Linear(config.d_model, 2)
        self.source_color_head = nn.Linear(config.d_model, len(COLORS))
        self.source_shape_head = nn.Linear(config.d_model, len(SHAPES))
        self.source_row_head = nn.Linear(config.d_model, config.grid_size)
        self.source_col_head = nn.Linear(config.d_model, config.grid_size)
        self.target_decoder = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
        )
        self.target_active_head = nn.Linear(config.d_model, 2)
        self.target_color_head = nn.Linear(config.d_model, len(COLORS))
        self.target_shape_head = nn.Linear(config.d_model, len(SHAPES))
        self.target_row_head = nn.Linear(config.d_model, config.grid_size)
        self.target_col_head = nn.Linear(config.d_model, config.grid_size)
        self.answer_query = nn.Parameter(torch.randn(config.answer_len, config.d_model) * 0.02)
        self.answer_decoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.answer_out = nn.Linear(config.d_model, VOCAB_SIZE)

    def encode_record(self, batch: AVJSet, *, zero_source: bool = False) -> torch.Tensor:
        active = torch.zeros_like(batch.source_active) if zero_source else batch.source_active
        color = F.one_hot(batch.source_color, num_classes=len(COLORS)).float()
        shape = F.one_hot(batch.source_shape, num_classes=len(SHAPES)).float()
        row = F.one_hot(batch.source_row, num_classes=self.config.grid_size).float()
        col = F.one_hot(batch.source_col, num_classes=self.config.grid_size).float()
        features = torch.cat((active.unsqueeze(-1).float(), color, shape, row, col), dim=-1)
        slots = self.object_encoder(features) + self.slot_schema.unsqueeze(0)
        return self.slot_norm(self.workspace(slots))

    def encode_prompt(self, prompts: torch.Tensor, *, zero_operation: bool = False) -> torch.Tensor:
        prompt = torch.full_like(prompts, PAD) if zero_operation else prompts
        encoded = self.prompt_encoder(self.token_embed(prompt) + self.prompt_pos.unsqueeze(0)[:, : prompts.shape[1]])
        mask = (prompt != PAD).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.prompt_norm(pooled)

    def retrieval(self, query: torch.Tensor, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(query, dim=-1)
        k = F.normalize(slots, dim=-1)
        logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / 0.07
        selected = torch.bmm(F.softmax(logits, dim=-1).unsqueeze(1), slots).squeeze(1)
        return logits, selected

    def forward(
        self,
        batch: AVJSet,
        *,
        zero_source: bool = False,
        zero_operation: bool = False,
        disable_process: bool = False,
    ) -> dict[str, torch.Tensor]:
        slots = self.encode_record(batch, zero_source=zero_source)
        prompt = self.encode_prompt(batch.prompts, zero_operation=zero_operation)
        read_a_logits, read_a_slot = self.retrieval(self.read_a_query(prompt), slots)
        read_b_logits, read_b_slot = self.retrieval(self.read_b_query(prompt), slots)
        edit_logits, edit_slot = self.retrieval(self.edit_query(prompt), slots)
        process_state = self.process(torch.cat((prompt, read_a_slot, read_b_slot, edit_slot), dim=-1))
        if disable_process:
            process_state = torch.zeros_like(process_state)
        target_context = torch.cat(
            (
                slots,
                prompt.unsqueeze(1).expand(-1, slots.shape[1], -1),
                process_state.unsqueeze(1).expand(-1, slots.shape[1], -1),
            ),
            dim=-1,
        )
        target_state = self.target_decoder(target_context)
        answer_query = self.answer_query.unsqueeze(0).expand(slots.shape[0], -1, -1)
        answer_context = torch.cat((prompt.unsqueeze(1), process_state.unsqueeze(1), slots, answer_query), dim=1)
        answer = self.answer_decoder(answer_context)[:, -self.config.answer_len :]
        return {
            "slots": slots,
            "read_a": read_a_logits,
            "read_b": read_b_logits,
            "edit_slot": edit_logits,
            "condition": self.condition_head(process_state),
            "edit_action": self.action_head(process_state),
            "edit_color": self.edit_color_head(process_state),
            "edit_row": self.edit_row_head(process_state),
            "edit_col": self.edit_col_head(process_state),
            "source_active": self.source_active_head(slots),
            "source_color": self.source_color_head(slots),
            "source_shape": self.source_shape_head(slots),
            "source_row": self.source_row_head(slots),
            "source_col": self.source_col_head(slots),
            "target_active": self.target_active_head(target_state),
            "target_color": self.target_color_head(target_state),
            "target_shape": self.target_shape_head(target_state),
            "target_row": self.target_row_head(target_state),
            "target_col": self.target_col_head(target_state),
            "answer": self.answer_out(answer),
        }


def masked_slot_ce(logits: torch.Tensor, target: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
    if active is None:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    mask = active.reshape(-1) > 0
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1])[mask], target.reshape(-1)[mask])


def optional_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target >= 0
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def source_recon_loss(outputs: dict[str, torch.Tensor], batch: AVJSet) -> torch.Tensor:
    return (
        masked_slot_ce(outputs["source_active"], batch.source_active)
        + masked_slot_ce(outputs["source_color"], batch.source_color, batch.source_active)
        + masked_slot_ce(outputs["source_shape"], batch.source_shape, batch.source_active)
        + masked_slot_ce(outputs["source_row"], batch.source_row, batch.source_active)
        + masked_slot_ce(outputs["source_col"], batch.source_col, batch.source_active)
    )


def target_record_loss(outputs: dict[str, torch.Tensor], batch: AVJSet) -> torch.Tensor:
    return (
        masked_slot_ce(outputs["target_active"], batch.target_active)
        + masked_slot_ce(outputs["target_color"], batch.target_color, batch.target_active)
        + masked_slot_ce(outputs["target_shape"], batch.target_shape, batch.target_active)
        + masked_slot_ce(outputs["target_row"], batch.target_row, batch.target_active)
        + masked_slot_ce(outputs["target_col"], batch.target_col, batch.target_active)
    )


def trace_loss(outputs: dict[str, torch.Tensor], batch: AVJSet) -> torch.Tensor:
    return (
        F.cross_entropy(outputs["read_a"], batch.read_a)
        + F.cross_entropy(outputs["read_b"], batch.read_b)
        + F.cross_entropy(outputs["edit_slot"], batch.edit_slot)
    )


def process_loss(outputs: dict[str, torch.Tensor], batch: AVJSet) -> torch.Tensor:
    return (
        F.cross_entropy(outputs["condition"], batch.condition)
        + F.cross_entropy(outputs["edit_action"], batch.edit_action)
        + optional_ce(outputs["edit_color"], batch.edit_color)
        + optional_ce(outputs["edit_row"], batch.edit_row)
        + optional_ce(outputs["edit_col"], batch.edit_col)
    )


def answer_loss(outputs: dict[str, torch.Tensor], batch: AVJSet) -> torch.Tensor:
    return F.cross_entropy(outputs["answer"].reshape(-1, VOCAB_SIZE), batch.answer.reshape(-1), ignore_index=PAD)


def stage_name(config: StageAVJConfig, step: int) -> str:
    if step <= config.codec_steps:
        return "codec"
    if step <= config.codec_steps + config.operation_steps:
        return "operation"
    if step <= config.codec_steps + config.operation_steps + config.process_steps:
        return "process"
    return "joint"


def total_loss(outputs: dict[str, torch.Tensor], batch: AVJSet, config: StageAVJConfig, stage: str) -> torch.Tensor:
    loss_source = source_recon_loss(outputs, batch)
    loss_target = target_record_loss(outputs, batch)
    loss_trace = trace_loss(outputs, batch)
    loss_process = process_loss(outputs, batch)
    loss_answer = answer_loss(outputs, batch)
    if stage == "codec":
        return config.source_recon_weight * loss_source
    if stage == "operation":
        return config.source_recon_weight * loss_source + config.trace_loss_weight * loss_trace
    if stage == "process":
        return (
            0.4 * config.source_recon_weight * loss_source
            + config.trace_loss_weight * loss_trace
            + config.process_loss_weight * loss_process
            + config.target_record_weight * loss_target
        )
    return (
        0.3 * config.source_recon_weight * loss_source
        + config.trace_loss_weight * loss_trace
        + config.process_loss_weight * loss_process
        + config.target_record_weight * loss_target
        + config.answer_token_weight * loss_answer
    )


def slot_exact(outputs: dict[str, torch.Tensor], batch: AVJSet, prefix: str) -> torch.Tensor:
    active = outputs[f"{prefix}_active"].argmax(dim=-1)
    active_ok = active == getattr(batch, f"{prefix}_active")
    target_active = getattr(batch, f"{prefix}_active") > 0
    color_ok = outputs[f"{prefix}_color"].argmax(dim=-1) == getattr(batch, f"{prefix}_color")
    shape_ok = outputs[f"{prefix}_shape"].argmax(dim=-1) == getattr(batch, f"{prefix}_shape")
    row_ok = outputs[f"{prefix}_row"].argmax(dim=-1) == getattr(batch, f"{prefix}_row")
    col_ok = outputs[f"{prefix}_col"].argmax(dim=-1) == getattr(batch, f"{prefix}_col")
    slot_ok = active_ok & (~target_active | (color_ok & shape_ok & row_ok & col_ok))
    return slot_ok.all(dim=1)


def token_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    non_pad = target != PAD
    token_accuracy = (pred[non_pad] == target[non_pad]).float().mean().item() if non_pad.any() else 0.0
    sequence_exact = ((pred == target) | ~non_pad).all(dim=1).float().mean().item()
    return {"answer_token_accuracy": token_accuracy, "answer_sequence_exact": sequence_exact}


def metrics(outputs: dict[str, torch.Tensor], batch: AVJSet) -> dict[str, float]:
    result = {
        "source_record_exact": slot_exact(outputs, batch, "source").float().mean().item(),
        "target_record_exact": slot_exact(outputs, batch, "target").float().mean().item(),
        "read_a_accuracy": (outputs["read_a"].argmax(dim=-1) == batch.read_a).float().mean().item(),
        "read_b_accuracy": (outputs["read_b"].argmax(dim=-1) == batch.read_b).float().mean().item(),
        "edit_slot_accuracy": (outputs["edit_slot"].argmax(dim=-1) == batch.edit_slot).float().mean().item(),
        "condition_accuracy": (outputs["condition"].argmax(dim=-1) == batch.condition).float().mean().item(),
        "edit_action_accuracy": (outputs["edit_action"].argmax(dim=-1) == batch.edit_action).float().mean().item(),
    }
    result.update(token_metrics(outputs["answer"], batch.answer))
    for family in FAMILIES:
        mask = batch.family == FAMILIES.index(family)
        if mask.any():
            result[f"{family}_target_record_exact"] = slot_exact(
                {key: value[mask] if value.shape[0] == mask.shape[0] else value for key, value in outputs.items()},
                batch.subset(mask.nonzero(as_tuple=False).flatten().tolist()),
                "target",
            ).float().mean().item()
            result[f"{family}_answer_sequence_exact"] = token_metrics(outputs["answer"][mask], batch.answer[mask])[
                "answer_sequence_exact"
            ]
    return result


def random_batch(data: AVJSet, rng: random.Random, batch_size: int, device: torch.device) -> AVJSet:
    indices = [rng.randrange(data.prompts.shape[0]) for _ in range(batch_size)]
    batch = data.subset(indices)
    return batch if batch.prompts.device == device else batch.to(device)


@torch.no_grad()
def evaluate(model: StageAVJModel, data: AVJSet, config: StageAVJConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    result: dict[str, list[float]] = {}
    for start in range(0, data.prompts.shape[0], config.eval_batch_size):
        indices = list(range(start, min(start + config.eval_batch_size, data.prompts.shape[0])))
        batch = data.subset(indices)
        batch = batch if batch.prompts.device == device else batch.to(device)
        variants = {
            "full": model(batch),
            "no_source": model(batch, zero_source=True),
            "no_operation": model(batch, zero_operation=True),
            "no_process": model(batch, disable_process=True),
        }
        for variant, outputs in variants.items():
            for key, value in metrics(outputs, batch).items():
                result.setdefault(f"{variant}_{key}", []).append(value)
    return {key: sum(values) / len(values) for key, values in result.items()}


def save_checkpoint(path: Path, model: StageAVJModel, optimizer: torch.optim.Optimizer, step: int, config: StageAVJConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config)},
        path,
    )


def load_checkpoint(path: Path, model: StageAVJModel, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def run(config: StageAVJConfig, output: Path, checkpoint_dir: Path | None, resume: bool) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    train = build_dataset("train", config.train_size, config)
    val = build_dataset("val", config.val_size, config)
    test = build_dataset("test", config.test_size, config)
    heldout = build_dataset("heldout", config.heldout_size, config)
    if config.gpu_resident_data and device.type == "cuda":
        train, val, test, heldout = (train.to(device), val.to(device), test.to(device), heldout.to(device))
    model = StageAVJModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    start_step = 0
    if resume and checkpoint_dir is not None and (checkpoint_dir / "latest.pt").exists():
        start_step = load_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, device)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(config.seed + 909)
    history: list[dict[str, object]] = []
    for step in range(start_step + 1, config.total_steps + 1):
        model.train()
        batch = random_batch(train, rng, config.batch_size, device)
        stage = stage_name(config, step)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch)
            loss = total_loss(outputs, batch, config, stage)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % config.eval_every == 0 or step == config.total_steps:
            row: dict[str, object] = {"step": step, "stage": stage, "loss": float(loss.detach().cpu())}
            row.update({f"val_{key}": value for key, value in evaluate(model, val, config, device).items()})
            history.append(row)
        if checkpoint_dir is not None and (step % config.save_every == 0 or step == config.total_steps):
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step, config)
    final = {
        "val": evaluate(model, val, config, device),
        "test": evaluate(model, test, config, device),
        "heldout": evaluate(model, heldout, config, device),
    }
    result = {
        "schema_version": 1,
        "stage": "AV-J",
        "config": asdict(config),
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "device": str(device),
        "history": history,
        "final": final,
        "examples": train.examples[: min(8, len(train.examples))],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage AV-J latent reasoning core with teacher traces.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_avj_latent_reasoning_core/smoke_result.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--heldout-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--codec-steps", type=int, default=200)
    parser.add_argument("--operation-steps", type=int, default=400)
    parser.add_argument("--process-steps", type=int, default=600)
    parser.add_argument("--joint-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--cpu-data", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StageAVJConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        heldout_size=args.heldout_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        lr=args.lr,
        codec_steps=args.codec_steps,
        operation_steps=args.operation_steps,
        process_steps=args.process_steps,
        joint_steps=args.joint_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        amp=not args.no_amp,
        gpu_resident_data=not args.cpu_data,
    )
    result = run(config, args.output, args.checkpoint_dir, args.resume)
    print(json.dumps({"output": str(args.output), "final": result["final"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
