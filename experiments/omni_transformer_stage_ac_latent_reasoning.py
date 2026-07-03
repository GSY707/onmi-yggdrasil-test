from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F


COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("circle", "square", "triangle", "diamond")
COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")
ROW_WORDS = ("one", "two", "three", "four")
COL_WORDS = ROW_WORDS
RELATIONS = ("left", "right", "above", "below")
PAIR_COUNT = len(COLORS) * len(SHAPES)
FAMILIES = ("color_at_cell", "shape_at_cell", "count_color_shape", "relation_yes_no")

PAD = 0
BOS = 1
EOS = 2
WORDS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "what",
    "color",
    "shape",
    "is",
    "the",
    "object",
    "at",
    "row",
    "column",
    "how",
    "many",
    "objects",
    "are",
    "there",
    "left",
    "right",
    "above",
    "below",
    "of",
    "answer",
    "one",
    "word",
    "yes",
    "no",
    *COLORS,
    *SHAPES,
    *COUNT_WORDS,
)
WORD_TO_ID = {word: index for index, word in enumerate(WORDS)}
VOCAB_SIZE = len(WORDS)


@dataclass(frozen=True)
class StageACConfig:
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 64
    seed: int = 20260701
    grid_size: int = 4
    max_objects: int = 8
    prompt_len: int = 18
    answer_len: int = 6
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    latent_tokens: int = 8
    lr: float = 8e-4
    text_steps: int = 500
    evidence_steps: int = 500
    reasoner_steps: int = 900
    eval_every: int = 300
    latent_mse_weight: float = 0.35
    evidence_loss_weight: float = 0.35
    reasoner_trace_weight: float = 0.0
    reasoner_reader_weight: float = 0.0
    reasoner_variant: str = "plain"
    reasoner_training_mode: str = "mixed"
    reasoner_stage_order: tuple[str, ...] = FAMILIES
    reasoner_stage_replay_interval: int = 0
    moe_teacher_forcing: bool = False
    query_teacher_forcing: str = "none"
    modes: tuple[str, ...] = ("latent_reasoner",)
    families: tuple[str, ...] = FAMILIES
    evidence_latent_mode: str = "abstract"


@dataclass(frozen=True)
class SceneObject:
    color: int
    shape: int
    row: int
    col: int


@dataclass
class StageACSet:
    prompts: torch.Tensor
    answers: torch.Tensor
    evidence: torch.Tensor
    grid_occupied: torch.Tensor
    grid_color: torch.Tensor
    grid_shape: torch.Tensor
    count_table: torch.Tensor
    pair_occupied: torch.Tensor
    pair_single: torch.Tensor
    pair_row: torch.Tensor
    pair_col: torch.Tensor
    family: torch.Tensor
    target_cell: torch.Tensor
    target_color: torch.Tensor
    target_shape: torch.Tensor
    target_count_pair: torch.Tensor
    target_count: torch.Tensor
    target_relation: torch.Tensor
    target_left_pair: torch.Tensor
    target_right_pair: torch.Tensor
    target_relation_op: torch.Tensor
    examples: list[dict[str, object]]

    def subset(self, indices: list[int]) -> "StageACSet":
        return StageACSet(
            prompts=self.prompts[indices],
            answers=self.answers[indices],
            evidence=self.evidence[indices],
            grid_occupied=self.grid_occupied[indices],
            grid_color=self.grid_color[indices],
            grid_shape=self.grid_shape[indices],
            count_table=self.count_table[indices],
            pair_occupied=self.pair_occupied[indices],
            pair_single=self.pair_single[indices],
            pair_row=self.pair_row[indices],
            pair_col=self.pair_col[indices],
            family=self.family[indices],
            target_cell=self.target_cell[indices],
            target_color=self.target_color[indices],
            target_shape=self.target_shape[indices],
            target_count_pair=self.target_count_pair[indices],
            target_count=self.target_count[indices],
            target_relation=self.target_relation[indices],
            target_left_pair=self.target_left_pair[indices],
            target_right_pair=self.target_right_pair[indices],
            target_relation_op=self.target_relation_op[indices],
            examples=[self.examples[index] for index in indices],
        )

    def to(self, device: torch.device) -> "StageACSet":
        return StageACSet(
            prompts=self.prompts.to(device=device, dtype=torch.long),
            answers=self.answers.to(device=device, dtype=torch.long),
            evidence=self.evidence.to(device=device, dtype=torch.float32),
            grid_occupied=self.grid_occupied.to(device=device, dtype=torch.float32),
            grid_color=self.grid_color.to(device=device, dtype=torch.long),
            grid_shape=self.grid_shape.to(device=device, dtype=torch.long),
            count_table=self.count_table.to(device=device, dtype=torch.long),
            pair_occupied=self.pair_occupied.to(device=device, dtype=torch.float32),
            pair_single=self.pair_single.to(device=device, dtype=torch.float32),
            pair_row=self.pair_row.to(device=device, dtype=torch.long),
            pair_col=self.pair_col.to(device=device, dtype=torch.long),
            family=self.family.to(device=device, dtype=torch.long),
            target_cell=self.target_cell.to(device=device, dtype=torch.long),
            target_color=self.target_color.to(device=device, dtype=torch.long),
            target_shape=self.target_shape.to(device=device, dtype=torch.long),
            target_count_pair=self.target_count_pair.to(device=device, dtype=torch.long),
            target_count=self.target_count.to(device=device, dtype=torch.long),
            target_relation=self.target_relation.to(device=device, dtype=torch.long),
            target_left_pair=self.target_left_pair.to(device=device, dtype=torch.long),
            target_right_pair=self.target_right_pair.to(device=device, dtype=torch.long),
            target_relation_op=self.target_relation_op.to(device=device, dtype=torch.long),
            examples=self.examples,
        )


def encode_words(words: list[str], length: int) -> list[int]:
    ids = [BOS] + [WORD_TO_ID[word] for word in words] + [EOS]
    ids = ids[:length]
    return ids + [PAD] * (length - len(ids))


def decode_words(tokens: torch.Tensor | list[int]) -> str:
    result: list[str] = []
    for token in tokens:
        value = int(token)
        if value in (PAD, BOS):
            continue
        if value == EOS:
            break
        result.append(WORDS[value])
    return " ".join(result)


def relation_holds(left: SceneObject, relation: str, right: SceneObject) -> bool:
    if relation == "left":
        return left.col < right.col
    if relation == "right":
        return left.col > right.col
    if relation == "above":
        return left.row < right.row
    if relation == "below":
        return left.row > right.row
    raise ValueError(relation)


def sample_objects(rng: random.Random, config: StageACConfig, *, allow_duplicate_pairs: bool) -> list[SceneObject]:
    count = rng.randint(4, config.max_objects)
    cells = [(row, col) for row in range(config.grid_size) for col in range(config.grid_size)]
    rng.shuffle(cells)
    pairs = [(color, shape) for color in range(len(COLORS)) for shape in range(len(SHAPES))]
    rng.shuffle(pairs)
    objects: list[SceneObject] = []
    for index in range(count):
        row, col = cells[index]
        if allow_duplicate_pairs:
            color = rng.randrange(len(COLORS))
            shape = rng.randrange(len(SHAPES))
        else:
            color, shape = pairs[index]
        objects.append(SceneObject(color=color, shape=shape, row=row, col=col))
    return objects


def object_words(obj: SceneObject) -> list[str]:
    return [COLORS[obj.color], SHAPES[obj.shape]]


def count_table_for(objects: list[SceneObject]) -> list[int]:
    counts = [0 for _ in range(PAIR_COUNT)]
    for obj in objects:
        counts[obj.color * len(SHAPES) + obj.shape] += 1
    return [min(value, len(COUNT_WORDS) - 1) for value in counts]


def evidence_features(objects: list[SceneObject], config: StageACConfig) -> list[list[float]]:
    feature_size = 1 + len(COLORS) + len(SHAPES) + config.grid_size + config.grid_size
    features = [[0.0 for _ in range(feature_size)] for _ in range(config.max_objects)]
    for index, obj in enumerate(objects[: config.max_objects]):
        row = [0.0 for _ in range(config.grid_size)]
        col = [0.0 for _ in range(config.grid_size)]
        row[obj.row] = 1.0
        col[obj.col] = 1.0
        color = [0.0 for _ in COLORS]
        shape = [0.0 for _ in SHAPES]
        color[obj.color] = 1.0
        shape[obj.shape] = 1.0
        features[index] = [1.0, *color, *shape, *row, *col]
    return features


def grid_targets(objects: list[SceneObject], config: StageACConfig) -> tuple[list[float], list[int], list[int]]:
    occupied = [0.0 for _ in range(config.grid_size * config.grid_size)]
    colors = [0 for _ in range(config.grid_size * config.grid_size)]
    shapes = [0 for _ in range(config.grid_size * config.grid_size)]
    for obj in objects:
        cell = obj.row * config.grid_size + obj.col
        occupied[cell] = 1.0
        colors[cell] = obj.color
        shapes[cell] = obj.shape
    return occupied, colors, shapes


def pair_object_targets(objects: list[SceneObject]) -> tuple[list[float], list[float], list[int], list[int]]:
    counts = [0 for _ in range(PAIR_COUNT)]
    rows = [0 for _ in range(PAIR_COUNT)]
    cols = [0 for _ in range(PAIR_COUNT)]
    for obj in objects:
        pair = obj.color * len(SHAPES) + obj.shape
        counts[pair] += 1
        rows[pair] = obj.row
        cols[pair] = obj.col
    occupied = [1.0 if count > 0 else 0.0 for count in counts]
    single = [1.0 if count == 1 else 0.0 for count in counts]
    return occupied, single, rows, cols


def empty_trace() -> dict[str, int]:
    return {
        "target_cell": -100,
        "target_color": -100,
        "target_shape": -100,
        "target_count_pair": -100,
        "target_count": -100,
        "target_relation": -100,
        "target_left_pair": -100,
        "target_right_pair": -100,
        "target_relation_op": -100,
    }


def make_example(rng: random.Random, family: str, config: StageACConfig) -> tuple[list[str], list[str], list[SceneObject], dict[str, int]]:
    if family == "count_color_shape":
        target_color = rng.randrange(len(COLORS))
        target_shape = rng.randrange(len(SHAPES))
        target_count = rng.randrange(0, min(config.max_objects, len(COUNT_WORDS) - 1) + 1)
        cells = [(row, col) for row in range(config.grid_size) for col in range(config.grid_size)]
        rng.shuffle(cells)
        objects: list[SceneObject] = []
        for _ in range(target_count):
            row, col = cells.pop()
            objects.append(SceneObject(target_color, target_shape, row, col))
        while len(objects) < config.max_objects:
            row, col = cells.pop()
            color = rng.randrange(len(COLORS))
            shape = rng.randrange(len(SHAPES))
            if color == target_color and shape == target_shape:
                shape = (shape + 1) % len(SHAPES)
            objects.append(SceneObject(color, shape, row, col))
        prompt = ["how", "many", COLORS[target_color], SHAPES[target_shape], "objects", "are", "there"]
        answer = [COUNT_WORDS[target_count]]
        trace = empty_trace()
        trace["target_count_pair"] = target_color * len(SHAPES) + target_shape
        trace["target_count"] = target_count
        return prompt, answer, objects, trace

    objects = sample_objects(rng, config, allow_duplicate_pairs=False)
    if family == "color_at_cell":
        target = rng.choice(objects)
        prompt = [
            "what",
            "color",
            "is",
            "the",
            "object",
            "at",
            "row",
            ROW_WORDS[target.row],
            "column",
            COL_WORDS[target.col],
        ]
        trace = empty_trace()
        trace["target_cell"] = target.row * config.grid_size + target.col
        trace["target_color"] = target.color
        return prompt, [COLORS[target.color]], objects, trace
    if family == "shape_at_cell":
        target = rng.choice(objects)
        prompt = [
            "what",
            "shape",
            "is",
            "the",
            "object",
            "at",
            "row",
            ROW_WORDS[target.row],
            "column",
            COL_WORDS[target.col],
        ]
        trace = empty_trace()
        trace["target_cell"] = target.row * config.grid_size + target.col
        trace["target_shape"] = target.shape
        return prompt, [SHAPES[target.shape]], objects, trace
    if family == "relation_yes_no":
        left, right = rng.sample(objects, 2)
        relation = rng.choice(RELATIONS)
        prompt = ["is", "the", *object_words(left), relation, "of", "the", *object_words(right)]
        trace = empty_trace()
        trace["target_relation"] = 1 if relation_holds(left, relation, right) else 0
        trace["target_left_pair"] = left.color * len(SHAPES) + left.shape
        trace["target_right_pair"] = right.color * len(SHAPES) + right.shape
        trace["target_relation_op"] = RELATIONS.index(relation)
        return prompt, ["yes" if trace["target_relation"] else "no"], objects, trace
    raise ValueError(family)


def build_dataset(split: str, size: int, config: StageACConfig) -> StageACSet:
    split_offset = {"train": 11, "val": 23, "test": 37}[split]
    rng = random.Random(config.seed + split_offset)
    prompts: list[list[int]] = []
    answers: list[list[int]] = []
    evidence: list[list[list[float]]] = []
    occupied: list[list[float]] = []
    grid_color: list[list[int]] = []
    grid_shape: list[list[int]] = []
    counts: list[list[int]] = []
    pair_occupied: list[list[float]] = []
    pair_single: list[list[float]] = []
    pair_rows: list[list[int]] = []
    pair_cols: list[list[int]] = []
    families: list[int] = []
    target_cells: list[int] = []
    target_colors: list[int] = []
    target_shapes: list[int] = []
    target_count_pairs: list[int] = []
    target_counts: list[int] = []
    target_relations: list[int] = []
    target_left_pairs: list[int] = []
    target_right_pairs: list[int] = []
    target_relation_ops: list[int] = []
    examples: list[dict[str, object]] = []
    selected_families = tuple(config.families)
    for family in selected_families:
        if family not in FAMILIES:
            raise ValueError(f"unknown family: {family}")
    for index in range(size):
        family = selected_families[index % len(selected_families)]
        family_index = FAMILIES.index(family)
        prompt_words, answer_words, objects, trace = make_example(rng, family, config)
        occ, color, shape = grid_targets(objects, config)
        pair_occ, pair_one, pair_row, pair_col = pair_object_targets(objects)
        prompts.append(encode_words(prompt_words, config.prompt_len))
        answers.append(encode_words(answer_words, config.answer_len))
        evidence.append(evidence_features(objects, config))
        occupied.append(occ)
        grid_color.append(color)
        grid_shape.append(shape)
        counts.append(count_table_for(objects))
        pair_occupied.append(pair_occ)
        pair_single.append(pair_one)
        pair_rows.append(pair_row)
        pair_cols.append(pair_col)
        families.append(family_index)
        target_cells.append(trace["target_cell"])
        target_colors.append(trace["target_color"])
        target_shapes.append(trace["target_shape"])
        target_count_pairs.append(trace["target_count_pair"])
        target_counts.append(trace["target_count"])
        target_relations.append(trace["target_relation"])
        target_left_pairs.append(trace["target_left_pair"])
        target_right_pairs.append(trace["target_right_pair"])
        target_relation_ops.append(trace["target_relation_op"])
        examples.append(
            {
                "split": split,
                "index": index,
                "family": family,
                "prompt": " ".join(prompt_words),
                "answer": " ".join(answer_words),
                "objects": [asdict(obj) for obj in objects],
                "trace": trace,
            }
        )
    return StageACSet(
        prompts=torch.tensor(prompts, dtype=torch.long),
        answers=torch.tensor(answers, dtype=torch.long),
        evidence=torch.tensor(evidence, dtype=torch.float32),
        grid_occupied=torch.tensor(occupied, dtype=torch.float32),
        grid_color=torch.tensor(grid_color, dtype=torch.long),
        grid_shape=torch.tensor(grid_shape, dtype=torch.long),
        count_table=torch.tensor(counts, dtype=torch.long),
        pair_occupied=torch.tensor(pair_occupied, dtype=torch.float32),
        pair_single=torch.tensor(pair_single, dtype=torch.float32),
        pair_row=torch.tensor(pair_rows, dtype=torch.long),
        pair_col=torch.tensor(pair_cols, dtype=torch.long),
        family=torch.tensor(families, dtype=torch.long),
        target_cell=torch.tensor(target_cells, dtype=torch.long),
        target_color=torch.tensor(target_colors, dtype=torch.long),
        target_shape=torch.tensor(target_shapes, dtype=torch.long),
        target_count_pair=torch.tensor(target_count_pairs, dtype=torch.long),
        target_count=torch.tensor(target_counts, dtype=torch.long),
        target_relation=torch.tensor(target_relations, dtype=torch.long),
        target_left_pair=torch.tensor(target_left_pairs, dtype=torch.long),
        target_right_pair=torch.tensor(target_right_pairs, dtype=torch.long),
        target_relation_op=torch.tensor(target_relation_ops, dtype=torch.long),
        examples=examples,
    )


def random_batch(data: StageACSet, *, rng: random.Random, batch_size: int, device: torch.device) -> StageACSet:
    return data.subset([rng.randrange(data.prompts.shape[0]) for _ in range(batch_size)]).to(device)


def family_subset(data: StageACSet, family: str) -> StageACSet:
    family_index = FAMILIES.index(family)
    indices = torch.nonzero(data.family == family_index, as_tuple=False).flatten().tolist()
    if not indices:
        raise ValueError(f"no examples for family: {family}")
    return data.subset(indices)


class LatentTextEncoder(nn.Module):
    def __init__(self, config: StageACConfig, *, length: int) -> None:
        super().__init__()
        self.length = length
        self.embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position = nn.Parameter(torch.randn(length, config.d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) + self.position.unsqueeze(0)
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((x, query), dim=1))
        return self.norm(encoded[:, -query.shape[1] :])


class LatentTokenDecoder(nn.Module):
    def __init__(self, config: StageACConfig, *, length: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(length, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.out = nn.Linear(config.d_model, VOCAB_SIZE)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(latent.shape[0], -1, -1)
        decoded = self.decoder(torch.cat((latent, query), dim=1))
        return self.out(self.norm(decoded[:, -query.shape[1] :]))


class TokenSequenceEncoder(nn.Module):
    def __init__(self, config: StageACConfig, *, length: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position = nn.Parameter(torch.randn(length, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) + self.position.unsqueeze(0)
        return self.norm(self.encoder(x))


class TokenSequenceDecoder(nn.Module):
    def __init__(self, config: StageACConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.out = nn.Linear(config.d_model, VOCAB_SIZE)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.decoder(latent)))


class TextLatentCodec(nn.Module):
    def __init__(self, config: StageACConfig) -> None:
        super().__init__()
        self.question_encoder = TokenSequenceEncoder(config, length=config.prompt_len)
        self.answer_encoder = TokenSequenceEncoder(config, length=config.answer_len)
        self.question_decoder = TokenSequenceDecoder(config)
        self.answer_decoder = TokenSequenceDecoder(config)

    def encode_question(self, prompts: torch.Tensor) -> torch.Tensor:
        return self.question_encoder(prompts)

    def encode_answer(self, answers: torch.Tensor) -> torch.Tensor:
        return self.answer_encoder(answers)

    def decode_question(self, latent: torch.Tensor) -> torch.Tensor:
        return self.question_decoder(latent)

    def decode_answer(self, latent: torch.Tensor) -> torch.Tensor:
        return self.answer_decoder(latent)


class EvidenceCodec(nn.Module):
    def __init__(self, config: StageACConfig) -> None:
        super().__init__()
        feature_size = 1 + len(COLORS) + len(SHAPES) + config.grid_size + config.grid_size
        self.grid_size = config.grid_size
        self.evidence_latent_mode = config.evidence_latent_mode
        self.count_classes = config.max_objects + 1
        self.input = nn.Linear(feature_size, config.d_model)
        self.object_slot = nn.Parameter(torch.randn(config.max_objects, config.d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.cell_query = nn.Parameter(torch.randn(config.grid_size * config.grid_size, config.d_model) * 0.02)
        self.count_query = nn.Parameter(torch.randn(len(COLORS) * len(SHAPES), config.d_model) * 0.02)
        self.cell_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.count_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.occupancy = nn.Linear(config.d_model, 1)
        self.color = nn.Linear(config.d_model, len(COLORS))
        self.shape = nn.Linear(config.d_model, len(SHAPES))
        self.count = nn.Linear(config.d_model, self.count_classes)

    def abstract_latent(self, evidence: torch.Tensor) -> torch.Tensor:
        x = self.input(evidence) + self.object_slot.unsqueeze(0)
        query = self.query.unsqueeze(0).expand(evidence.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((x, query), dim=1))
        return self.norm(encoded[:, -query.shape[1] :])

    def cell_latent(self, abstract: torch.Tensor) -> torch.Tensor:
        cell_query = self.cell_query.unsqueeze(0).expand(abstract.shape[0], -1, -1)
        cell_ctx, _ = self.cell_attn(cell_query, abstract, abstract, need_weights=False)
        return self.norm(cell_ctx)

    def encode(self, evidence: torch.Tensor) -> torch.Tensor:
        abstract = self.abstract_latent(evidence)
        if self.evidence_latent_mode == "abstract":
            return abstract
        if self.evidence_latent_mode == "cell":
            return self.cell_latent(abstract)
        raise ValueError(f"unknown evidence_latent_mode: {self.evidence_latent_mode}")

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        count_query = self.count_query.unsqueeze(0).expand(latent.shape[0], -1, -1)
        if latent.shape[1] == self.grid_size * self.grid_size:
            cell_ctx = latent
        else:
            cell_query = self.cell_query.unsqueeze(0).expand(latent.shape[0], -1, -1)
            cell_ctx, _ = self.cell_attn(cell_query, latent, latent, need_weights=False)
        count_ctx, _ = self.count_attn(count_query, latent, latent, need_weights=False)
        return {
            "occupied": self.occupancy(cell_ctx).squeeze(-1),
            "color": self.color(cell_ctx),
            "shape": self.shape(cell_ctx),
            "count": self.count(count_ctx),
        }

    def forward(self, evidence: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latent = self.encode(evidence)
        return latent, self.decode(latent)


class LatentReasoner(nn.Module):
    def __init__(self, config: StageACConfig) -> None:
        super().__init__()
        self.variant = config.reasoner_variant
        self.grid_cell_count = config.grid_size * config.grid_size
        self.answer_len = config.answer_len
        self.query = nn.Parameter(torch.randn(config.answer_len, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.operation_head = nn.Linear(config.d_model, len(FAMILIES))
        self.occupancy_head = nn.Linear(config.d_model, 1)
        self.cell_head = nn.Linear(config.d_model, config.grid_size * config.grid_size)
        self.color_head = nn.Linear(config.d_model, len(COLORS))
        self.shape_head = nn.Linear(config.d_model, len(SHAPES))
        self.count_pair_head = nn.Linear(config.d_model, PAIR_COUNT)
        self.count_head = nn.Linear(config.d_model, config.max_objects + 1)
        self.relation_head = nn.Linear(config.d_model, 2)
        self.left_pair_head = nn.Linear(config.d_model, PAIR_COUNT)
        self.right_pair_head = nn.Linear(config.d_model, PAIR_COUNT)
        self.relation_op_head = nn.Linear(config.d_model, len(RELATIONS))
        self.relation_op_value = nn.Linear(len(RELATIONS), config.d_model, bias=False)
        self.pair_occupancy_head = nn.Linear(config.d_model, 1)
        self.pair_row_head = nn.Linear(config.d_model, config.grid_size)
        self.pair_col_head = nn.Linear(config.d_model, config.grid_size)
        self.cell_query = nn.Parameter(torch.randn(self.grid_cell_count, config.d_model) * 0.02)
        self.count_query = nn.Parameter(torch.randn(PAIR_COUNT, config.d_model) * 0.02)
        self.pair_query = nn.Parameter(torch.randn(PAIR_COUNT, config.d_model) * 0.02)
        self.cell_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.count_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.pair_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.question_norm = nn.LayerNorm(config.d_model)
        self.cell_norm = nn.LayerNorm(config.d_model)
        self.count_norm = nn.LayerNorm(config.d_model)
        self.pair_norm = nn.LayerNorm(config.d_model)
        self.answer_result = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.active_answer_result = nn.Sequential(
            nn.Linear(config.d_model * 5, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.active_relation_state = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.trace_left_state = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.trace_right_state = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.trace_relation_state = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.trace_position_compare = nn.Sequential(
            nn.Linear(config.grid_size * 4 + len(RELATIONS), config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.trace_answer_result = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.moe_gate_head = nn.Linear(config.d_model, len(FAMILIES))
        self.moe_answer_results = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model * 5, config.d_model * 2),
                    nn.GELU(),
                    nn.Linear(config.d_model * 2, config.d_model),
                )
                for _ in FAMILIES
            ]
        )
        self.moe_relation_states = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model * 4, config.d_model * 2),
                    nn.GELU(),
                    nn.Linear(config.d_model * 2, config.d_model),
                )
                for _ in FAMILIES
            ]
        )
        self.answer_token_norm = nn.LayerNorm(config.d_model)

    def plain_forward(self, question_latent: torch.Tensor, evidence_latent: torch.Tensor) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(question_latent.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((question_latent, evidence_latent, query), dim=1))
        return self.norm(encoded[:, -query.shape[1] :])

    def evidence_cell_tokens(self, evidence_latent: torch.Tensor) -> torch.Tensor:
        if evidence_latent.shape[1] == self.grid_cell_count:
            return self.cell_norm(evidence_latent)
        query = self.cell_query.unsqueeze(0).expand(evidence_latent.shape[0], -1, -1)
        cells, _ = self.cell_attn(query, evidence_latent, evidence_latent, need_weights=False)
        return self.cell_norm(cells)

    def evidence_count_tokens(self, evidence_latent: torch.Tensor) -> torch.Tensor:
        query = self.count_query.unsqueeze(0).expand(evidence_latent.shape[0], -1, -1)
        counts, _ = self.count_attn(query, evidence_latent, evidence_latent, need_weights=False)
        return self.count_norm(counts)

    def evidence_pair_tokens(self, evidence_latent: torch.Tensor) -> torch.Tensor:
        query = self.pair_query.unsqueeze(0).expand(evidence_latent.shape[0], -1, -1)
        pairs, _ = self.pair_attn(query, evidence_latent, evidence_latent, need_weights=False)
        return self.pair_norm(pairs)

    def question_only_answer_seed(self, question_latent: torch.Tensor) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(question_latent.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((question_latent, query), dim=1))
        return self.norm(encoded[:, -query.shape[1] :])

    def plain_trace_logits(self, answer_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        op_token = answer_latent[:, 0]
        selector_token = answer_latent[:, 1]
        result_token = answer_latent[:, 2]
        return {
            "operation": self.operation_head(op_token),
            "cell": self.cell_head(selector_token),
            "color": self.color_head(result_token),
            "shape": self.shape_head(result_token),
            "count_pair": self.count_pair_head(selector_token),
            "count": self.count_head(result_token),
            "relation": self.relation_head(result_token),
        }

    def readout_forward_with_trace(
        self, question_latent: torch.Tensor, evidence_latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        answer_latent = self.plain_forward(question_latent, evidence_latent)
        question_token = self.question_norm(question_latent.mean(dim=1))
        cell_tokens = self.evidence_cell_tokens(evidence_latent)
        count_tokens = self.evidence_count_tokens(evidence_latent)
        cell_logits = self.cell_head(question_token)
        count_pair_logits = self.count_pair_head(question_token)
        selected_cell = torch.bmm(F.softmax(cell_logits, dim=-1).unsqueeze(1), cell_tokens).squeeze(1)
        selected_count = torch.bmm(F.softmax(count_pair_logits, dim=-1).unsqueeze(1), count_tokens).squeeze(1)
        result_context = torch.cat((question_token, selected_cell, selected_count), dim=-1)
        result_update = self.answer_result(result_context)
        answer_latent = answer_latent.clone()
        answer_latent[:, 1] = self.answer_token_norm(answer_latent[:, 1] + result_update)
        trace_logits = {
            "operation": self.operation_head(question_token),
            "cell": cell_logits,
            "color": self.color_head(selected_cell),
            "shape": self.shape_head(selected_cell),
            "count_pair": count_pair_logits,
            "count": self.count_head(selected_count),
            "relation": self.relation_head(question_token),
            "grid_occupied": self.occupancy_head(cell_tokens).squeeze(-1),
            "grid_color": self.color_head(cell_tokens),
            "grid_shape": self.shape_head(cell_tokens),
            "count_table": self.count_head(count_tokens),
        }
        return answer_latent, trace_logits

    def active_read_forward_with_trace(
        self, question_latent: torch.Tensor, evidence_latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        answer_latent = self.question_only_answer_seed(question_latent)
        question_token = self.question_norm(question_latent.mean(dim=1))
        cell_tokens = self.evidence_cell_tokens(evidence_latent)
        count_tokens = self.evidence_count_tokens(evidence_latent)
        pair_tokens = self.evidence_pair_tokens(evidence_latent)

        cell_logits = self.cell_head(question_token)
        count_pair_logits = self.count_pair_head(question_token)
        left_pair_logits = self.left_pair_head(question_token)
        right_pair_logits = self.right_pair_head(question_token)
        relation_op_logits = self.relation_op_head(question_token)

        cell_observation = torch.bmm(F.softmax(cell_logits, dim=-1).unsqueeze(1), cell_tokens).squeeze(1)
        count_observation = torch.bmm(F.softmax(count_pair_logits, dim=-1).unsqueeze(1), count_tokens).squeeze(1)
        left_observation = torch.bmm(F.softmax(left_pair_logits, dim=-1).unsqueeze(1), pair_tokens).squeeze(1)
        right_observation = torch.bmm(F.softmax(right_pair_logits, dim=-1).unsqueeze(1), pair_tokens).squeeze(1)

        relation_op_token = self.relation_op_value(F.softmax(relation_op_logits, dim=-1))
        relation_state = self.active_relation_state(
            torch.cat((question_token, left_observation, right_observation, relation_op_token), dim=-1)
        )
        result_context = torch.cat(
            (question_token, cell_observation, count_observation, left_observation, right_observation), dim=-1
        )
        result_update = self.active_answer_result(result_context)
        answer_latent = answer_latent.clone()
        answer_latent[:, 1] = self.answer_token_norm(answer_latent[:, 1] + result_update)
        trace_logits = {
            "operation": self.operation_head(question_token),
            "cell": cell_logits,
            "color": self.color_head(cell_observation),
            "shape": self.shape_head(cell_observation),
            "count_pair": count_pair_logits,
            "count": self.count_head(count_observation),
            "relation": self.relation_head(relation_state),
            "left_pair": left_pair_logits,
            "right_pair": right_pair_logits,
            "relation_op": relation_op_logits,
            "grid_occupied": self.occupancy_head(cell_tokens).squeeze(-1),
            "grid_color": self.color_head(cell_tokens),
            "grid_shape": self.shape_head(cell_tokens),
            "count_table": self.count_head(count_tokens),
            "pair_occupied": self.pair_occupancy_head(pair_tokens).squeeze(-1),
            "pair_row": self.pair_row_head(pair_tokens),
            "pair_col": self.pair_col_head(pair_tokens),
        }
        return answer_latent, trace_logits

    def selected_observation(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        teacher_force: bool = False,
    ) -> torch.Tensor:
        weights = F.softmax(logits, dim=-1)
        if teacher_force and target is not None:
            valid = target >= 0
            if valid.any():
                forced = F.one_hot(target.clamp_min(0), num_classes=logits.shape[-1]).to(dtype=weights.dtype)
                weights = torch.where(valid.unsqueeze(-1), forced, weights)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)

    def selected_weights(
        self,
        logits: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        teacher_force: bool = False,
    ) -> torch.Tensor:
        weights = F.softmax(logits, dim=-1)
        if teacher_force and target is not None:
            valid = target >= 0
            if valid.any():
                forced = F.one_hot(target.clamp_min(0), num_classes=logits.shape[-1]).to(dtype=weights.dtype)
                weights = torch.where(valid.unsqueeze(-1), forced, weights)
        return weights

    def trace_multistep_forward_with_trace(
        self,
        question_latent: torch.Tensor,
        evidence_latent: torch.Tensor,
        batch: StageACSet | None = None,
        *,
        teacher_force_queries: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        answer_latent = self.question_only_answer_seed(question_latent)
        question_token = self.question_norm(question_latent.mean(dim=1))
        cell_tokens = self.evidence_cell_tokens(evidence_latent)
        count_tokens = self.evidence_count_tokens(evidence_latent)
        pair_tokens = self.evidence_pair_tokens(evidence_latent)

        cell_logits = self.cell_head(question_token)
        count_pair_logits = self.count_pair_head(question_token)
        left_pair_logits = self.left_pair_head(question_token)
        relation_op_logits = self.relation_op_head(question_token)

        cell_target = batch.target_cell if batch is not None else None
        count_target = batch.target_count_pair if batch is not None else None
        left_target = batch.target_left_pair if batch is not None else None
        right_target = batch.target_right_pair if batch is not None else None

        cell_weights = self.selected_weights(cell_logits, cell_target, teacher_force=teacher_force_queries)
        count_weights = self.selected_weights(count_pair_logits, count_target, teacher_force=teacher_force_queries)
        left_weights = self.selected_weights(left_pair_logits, left_target, teacher_force=teacher_force_queries)
        cell_observation = torch.bmm(cell_weights.unsqueeze(1), cell_tokens).squeeze(1)
        count_observation = torch.bmm(count_weights.unsqueeze(1), count_tokens).squeeze(1)
        left_observation = torch.bmm(left_weights.unsqueeze(1), pair_tokens).squeeze(1)
        left_state = self.trace_left_state(torch.cat((question_token, left_observation), dim=-1))
        right_pair_logits = self.right_pair_head(left_state)
        right_weights = self.selected_weights(right_pair_logits, right_target, teacher_force=teacher_force_queries)
        right_observation = torch.bmm(right_weights.unsqueeze(1), pair_tokens).squeeze(1)
        right_state = self.trace_right_state(torch.cat((question_token, left_state, right_observation), dim=-1))
        relation_op_token = self.relation_op_value(F.softmax(relation_op_logits, dim=-1))
        pair_row_logits = self.pair_row_head(pair_tokens)
        pair_col_logits = self.pair_col_head(pair_tokens)
        left_row = torch.bmm(left_weights.unsqueeze(1), pair_row_logits).squeeze(1)
        left_col = torch.bmm(left_weights.unsqueeze(1), pair_col_logits).squeeze(1)
        right_row = torch.bmm(right_weights.unsqueeze(1), pair_row_logits).squeeze(1)
        right_col = torch.bmm(right_weights.unsqueeze(1), pair_col_logits).squeeze(1)
        position_compare = self.trace_position_compare(
            torch.cat(
                (
                    F.softmax(left_row, dim=-1),
                    F.softmax(left_col, dim=-1),
                    F.softmax(right_row, dim=-1),
                    F.softmax(right_col, dim=-1),
                    F.softmax(relation_op_logits, dim=-1),
                ),
                dim=-1,
            )
        )
        relation_state = self.trace_relation_state(
            torch.cat((question_token, left_state, right_state, relation_op_token), dim=-1)
        )
        relation_state = self.answer_token_norm(relation_state + position_compare)

        result_context = torch.cat((question_token, cell_observation, count_observation, relation_state), dim=-1)
        result_update = self.trace_answer_result(result_context)
        answer_latent = answer_latent.clone()
        answer_latent[:, 1] = self.answer_token_norm(answer_latent[:, 1] + result_update)
        trace_logits = {
            "operation": self.operation_head(question_token),
            "cell": cell_logits,
            "color": self.color_head(cell_observation),
            "shape": self.shape_head(cell_observation),
            "count_pair": count_pair_logits,
            "count": self.count_head(count_observation),
            "relation": self.relation_head(relation_state),
            "left_pair": left_pair_logits,
            "right_pair": right_pair_logits,
            "relation_op": relation_op_logits,
            "grid_occupied": self.occupancy_head(cell_tokens).squeeze(-1),
            "grid_color": self.color_head(cell_tokens),
            "grid_shape": self.shape_head(cell_tokens),
            "count_table": self.count_head(count_tokens),
            "pair_occupied": self.pair_occupancy_head(pair_tokens).squeeze(-1),
            "pair_row": pair_row_logits,
            "pair_col": pair_col_logits,
        }
        if teacher_force_queries:
            trace_logits["teacher_forced_queries"] = torch.ones(
                question_token.shape[0], 1, dtype=question_token.dtype, device=question_token.device
            )
        return answer_latent, trace_logits

    def moe_active_forward_with_trace(
        self, question_latent: torch.Tensor, evidence_latent: torch.Tensor, route_target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        answer_latent = self.question_only_answer_seed(question_latent)
        question_token = self.question_norm(question_latent.mean(dim=1))
        cell_tokens = self.evidence_cell_tokens(evidence_latent)
        count_tokens = self.evidence_count_tokens(evidence_latent)
        pair_tokens = self.evidence_pair_tokens(evidence_latent)

        cell_logits = self.cell_head(question_token)
        count_pair_logits = self.count_pair_head(question_token)
        left_pair_logits = self.left_pair_head(question_token)
        right_pair_logits = self.right_pair_head(question_token)
        relation_op_logits = self.relation_op_head(question_token)
        moe_gate_logits = self.moe_gate_head(question_token)

        if route_target is not None:
            route_weights = F.one_hot(route_target, num_classes=len(FAMILIES)).to(dtype=question_token.dtype)
        else:
            route_weights = F.softmax(moe_gate_logits, dim=-1)

        cell_observation = torch.bmm(F.softmax(cell_logits, dim=-1).unsqueeze(1), cell_tokens).squeeze(1)
        count_observation = torch.bmm(F.softmax(count_pair_logits, dim=-1).unsqueeze(1), count_tokens).squeeze(1)
        left_observation = torch.bmm(F.softmax(left_pair_logits, dim=-1).unsqueeze(1), pair_tokens).squeeze(1)
        right_observation = torch.bmm(F.softmax(right_pair_logits, dim=-1).unsqueeze(1), pair_tokens).squeeze(1)
        relation_op_token = self.relation_op_value(F.softmax(relation_op_logits, dim=-1))

        result_context = torch.cat(
            (question_token, cell_observation, count_observation, left_observation, right_observation), dim=-1
        )
        relation_context = torch.cat((question_token, left_observation, right_observation, relation_op_token), dim=-1)
        expert_updates = torch.stack([expert(result_context) for expert in self.moe_answer_results], dim=1)
        expert_relation_states = torch.stack([expert(relation_context) for expert in self.moe_relation_states], dim=1)
        result_update = torch.sum(route_weights.unsqueeze(-1) * expert_updates, dim=1)
        relation_state = torch.sum(route_weights.unsqueeze(-1) * expert_relation_states, dim=1)

        answer_latent = answer_latent.clone()
        answer_latent[:, 1] = self.answer_token_norm(answer_latent[:, 1] + result_update)
        trace_logits = {
            "operation": self.operation_head(question_token),
            "moe_gate": moe_gate_logits,
            "cell": cell_logits,
            "color": self.color_head(cell_observation),
            "shape": self.shape_head(cell_observation),
            "count_pair": count_pair_logits,
            "count": self.count_head(count_observation),
            "relation": self.relation_head(relation_state),
            "left_pair": left_pair_logits,
            "right_pair": right_pair_logits,
            "relation_op": relation_op_logits,
            "grid_occupied": self.occupancy_head(cell_tokens).squeeze(-1),
            "grid_color": self.color_head(cell_tokens),
            "grid_shape": self.shape_head(cell_tokens),
            "count_table": self.count_head(count_tokens),
            "pair_occupied": self.pair_occupancy_head(pair_tokens).squeeze(-1),
            "pair_row": self.pair_row_head(pair_tokens),
            "pair_col": self.pair_col_head(pair_tokens),
        }
        return answer_latent, trace_logits

    def forward_with_trace(
        self,
        question_latent: torch.Tensor,
        evidence_latent: torch.Tensor,
        route_target: torch.Tensor | None = None,
        batch: StageACSet | None = None,
        teacher_force_queries: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.variant == "plain":
            answer_latent = self.plain_forward(question_latent, evidence_latent)
            return answer_latent, self.plain_trace_logits(answer_latent)
        if self.variant == "readout":
            return self.readout_forward_with_trace(question_latent, evidence_latent)
        if self.variant == "active_read":
            return self.active_read_forward_with_trace(question_latent, evidence_latent)
        if self.variant == "trace_multistep":
            return self.trace_multistep_forward_with_trace(
                question_latent, evidence_latent, batch, teacher_force_queries=teacher_force_queries
            )
        if self.variant == "moe_active":
            return self.moe_active_forward_with_trace(question_latent, evidence_latent, route_target)
        raise ValueError(f"unknown reasoner_variant: {self.variant}")

    def forward(self, question_latent: torch.Tensor, evidence_latent: torch.Tensor) -> torch.Tensor:
        return self.forward_with_trace(question_latent, evidence_latent)[0]


def sequence_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def token_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    non_pad = target != PAD
    token_acc = (pred[non_pad] == target[non_pad]).float().mean().item()
    exact = (pred == target).all(dim=1).float().mean().item()
    answer_word_exact = (pred[:, 1] == target[:, 1]).float().mean().item()
    return {
        "token_accuracy": token_acc,
        "sequence_exact": exact,
        "answer_word_exact": answer_word_exact,
    }


def evidence_loss(outputs: dict[str, torch.Tensor], batch: StageACSet) -> torch.Tensor:
    occupied_loss = F.binary_cross_entropy_with_logits(outputs["occupied"], batch.grid_occupied)
    occupied_mask = batch.grid_occupied > 0.5
    if occupied_mask.any():
        color_loss = F.cross_entropy(outputs["color"][occupied_mask], batch.grid_color[occupied_mask])
        shape_loss = F.cross_entropy(outputs["shape"][occupied_mask], batch.grid_shape[occupied_mask])
    else:
        color_loss = outputs["color"].sum() * 0.0
        shape_loss = outputs["shape"].sum() * 0.0
    count_loss = F.cross_entropy(outputs["count"].reshape(-1, outputs["count"].shape[-1]), batch.count_table.reshape(-1))
    return occupied_loss + color_loss + shape_loss + count_loss


def evidence_metrics(outputs: dict[str, torch.Tensor], batch: StageACSet) -> dict[str, float]:
    occupied_pred = (torch.sigmoid(outputs["occupied"]) > 0.5).float()
    occupied_exact = (occupied_pred == batch.grid_occupied).all(dim=1).float().mean().item()
    mask = batch.grid_occupied > 0.5
    color_acc = (outputs["color"].argmax(dim=-1)[mask] == batch.grid_color[mask]).float().mean().item()
    shape_acc = (outputs["shape"].argmax(dim=-1)[mask] == batch.grid_shape[mask]).float().mean().item()
    count_exact = (outputs["count"].argmax(dim=-1) == batch.count_table).all(dim=1).float().mean().item()
    return {
        "grid_occupancy_exact": occupied_exact,
        "occupied_color_accuracy": color_acc,
        "occupied_shape_accuracy": shape_acc,
        "count_table_exact": count_exact,
    }


def optional_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target >= 0
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def optional_named_ce(logits: dict[str, torch.Tensor], name: str, target: torch.Tensor) -> torch.Tensor:
    if name not in logits:
        return logits["operation"].sum() * 0.0
    return optional_ce(logits[name], target)


def trace_supervision_loss(logits: dict[str, torch.Tensor], batch: StageACSet) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {
        "operation": F.cross_entropy(logits["operation"], batch.family),
        "moe_gate": optional_named_ce(logits, "moe_gate", batch.family),
        "cell": optional_ce(logits["cell"], batch.target_cell),
        "color": optional_ce(logits["color"], batch.target_color),
        "shape": optional_ce(logits["shape"], batch.target_shape),
        "count_pair": optional_ce(logits["count_pair"], batch.target_count_pair),
        "count": optional_ce(logits["count"], batch.target_count),
        "relation": optional_ce(logits["relation"], batch.target_relation),
        "left_pair": optional_named_ce(logits, "left_pair", batch.target_left_pair),
        "right_pair": optional_named_ce(logits, "right_pair", batch.target_right_pair),
        "relation_op": optional_named_ce(logits, "relation_op", batch.target_relation_op),
    }
    total = sum(losses.values())
    metrics = {f"trace_loss_{name}": float(value.detach().cpu()) for name, value in losses.items()}
    return total, metrics


def reasoner_reader_loss(logits: dict[str, torch.Tensor], batch: StageACSet) -> tuple[torch.Tensor, dict[str, float]]:
    if "grid_occupied" not in logits:
        zero = logits["operation"].sum() * 0.0
        return zero, {}
    occupied_loss = F.binary_cross_entropy_with_logits(logits["grid_occupied"], batch.grid_occupied)
    occupied_mask = batch.grid_occupied > 0.5
    if occupied_mask.any():
        color_loss = F.cross_entropy(logits["grid_color"][occupied_mask], batch.grid_color[occupied_mask])
        shape_loss = F.cross_entropy(logits["grid_shape"][occupied_mask], batch.grid_shape[occupied_mask])
    else:
        color_loss = logits["grid_color"].sum() * 0.0
        shape_loss = logits["grid_shape"].sum() * 0.0
    count_loss = F.cross_entropy(
        logits["count_table"].reshape(-1, logits["count_table"].shape[-1]),
        batch.count_table.reshape(-1),
    )
    losses = {
        "reader_occupied": occupied_loss,
        "reader_color": color_loss,
        "reader_shape": shape_loss,
        "reader_count": count_loss,
    }
    if "pair_occupied" in logits:
        pair_occupied_loss = F.binary_cross_entropy_with_logits(logits["pair_occupied"], batch.pair_occupied)
        single_mask = batch.pair_single > 0.5
        if single_mask.any():
            pair_row_loss = F.cross_entropy(logits["pair_row"][single_mask], batch.pair_row[single_mask])
            pair_col_loss = F.cross_entropy(logits["pair_col"][single_mask], batch.pair_col[single_mask])
        else:
            pair_row_loss = logits["pair_row"].sum() * 0.0
            pair_col_loss = logits["pair_col"].sum() * 0.0
        losses.update(
            {
                "reader_pair_occupied": pair_occupied_loss,
                "reader_pair_row": pair_row_loss,
                "reader_pair_col": pair_col_loss,
            }
        )
    total = sum(losses.values())
    metrics = {f"trace_loss_{name}": float(value.detach().cpu()) for name, value in losses.items()}
    return total, metrics


def optional_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float | None:
    mask = target >= 0
    if not mask.any():
        return None
    return (logits.argmax(dim=-1)[mask] == target[mask]).float().mean().item()


def optional_named_accuracy(logits: dict[str, torch.Tensor], name: str, target: torch.Tensor) -> float | None:
    if name not in logits:
        return None
    return optional_accuracy(logits[name], target)


def trace_metrics(logits: dict[str, torch.Tensor], batch: StageACSet) -> dict[str, float]:
    raw: dict[str, float | None] = {
        "trace_operation_accuracy": (logits["operation"].argmax(dim=-1) == batch.family).float().mean().item(),
        "trace_moe_gate_accuracy": optional_named_accuracy(logits, "moe_gate", batch.family),
        "trace_cell_accuracy": optional_accuracy(logits["cell"], batch.target_cell),
        "trace_color_accuracy": optional_accuracy(logits["color"], batch.target_color),
        "trace_shape_accuracy": optional_accuracy(logits["shape"], batch.target_shape),
        "trace_count_pair_accuracy": optional_accuracy(logits["count_pair"], batch.target_count_pair),
        "trace_count_accuracy": optional_accuracy(logits["count"], batch.target_count),
        "trace_relation_accuracy": optional_accuracy(logits["relation"], batch.target_relation),
        "trace_left_pair_accuracy": optional_named_accuracy(logits, "left_pair", batch.target_left_pair),
        "trace_right_pair_accuracy": optional_named_accuracy(logits, "right_pair", batch.target_right_pair),
        "trace_relation_op_accuracy": optional_named_accuracy(logits, "relation_op", batch.target_relation_op),
    }
    result = {key: value for key, value in raw.items() if value is not None}
    if "moe_gate" in logits:
        gate_probs = F.softmax(logits["moe_gate"], dim=-1)
        entropy = -(gate_probs * torch.log(gate_probs.clamp_min(1e-8))).sum(dim=-1).mean().item()
        result["trace_moe_gate_entropy"] = entropy
    if "grid_occupied" in logits:
        reader = evidence_metrics(
            {
                "occupied": logits["grid_occupied"],
                "color": logits["grid_color"],
                "shape": logits["grid_shape"],
                "count": logits["count_table"],
            },
            batch,
        )
        result.update({f"reader_{key}": value for key, value in reader.items()})
    if "pair_occupied" in logits:
        pair_pred = (torch.sigmoid(logits["pair_occupied"]) > 0.5).float()
        result["reader_pair_occupancy_exact"] = (pair_pred == batch.pair_occupied).all(dim=1).float().mean().item()
        single_mask = batch.pair_single > 0.5
        if single_mask.any():
            result["reader_pair_row_accuracy"] = (
                logits["pair_row"].argmax(dim=-1)[single_mask] == batch.pair_row[single_mask]
            ).float().mean().item()
            result["reader_pair_col_accuracy"] = (
                logits["pair_col"].argmax(dim=-1)[single_mask] == batch.pair_col[single_mask]
            ).float().mean().item()
    return result


@torch.no_grad()
def evaluate_text_codec(model: TextLatentCodec, data: StageACSet, config: StageACConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    question_latent = model.encode_question(batch.prompts)
    answer_latent = model.encode_answer(batch.answers)
    question_logits = model.decode_question(question_latent)
    answer_logits = model.decode_answer(answer_latent)
    question = token_metrics(question_logits, batch.prompts)
    answer = token_metrics(answer_logits, batch.answers)
    return {
        "question_token_accuracy": question["token_accuracy"],
        "question_sequence_exact": question["sequence_exact"],
        "answer_token_accuracy": answer["token_accuracy"],
        "answer_sequence_exact": answer["sequence_exact"],
        "answer_word_exact": answer["answer_word_exact"],
    }


@torch.no_grad()
def evaluate_evidence_codec(model: EvidenceCodec, data: StageACSet, device: torch.device) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    _, outputs = model(batch.evidence)
    return evidence_metrics(outputs, batch)


@torch.no_grad()
def evaluate_reasoner(
    text_codec: TextLatentCodec,
    evidence_codec: EvidenceCodec,
    reasoner: LatentReasoner,
    data: StageACSet,
    config: StageACConfig,
    device: torch.device,
) -> dict[str, object]:
    text_codec.eval()
    evidence_codec.eval()
    reasoner.eval()
    batch = data.to(device)
    question_latent = text_codec.encode_question(batch.prompts)
    evidence_latent = evidence_codec.encode(batch.evidence)
    target_answer_latent = text_codec.encode_answer(batch.answers)

    def run_with(evidence: torch.Tensor, *, teacher_force_queries: bool = False) -> dict[str, float]:
        predicted_answer_latent, trace_logits = reasoner.forward_with_trace(
            question_latent,
            evidence,
            batch=batch,
            teacher_force_queries=teacher_force_queries,
        )
        logits = text_codec.decode_answer(predicted_answer_latent)
        metrics = token_metrics(logits, batch.answers)
        metrics["latent_mse"] = F.mse_loss(predicted_answer_latent, target_answer_latent).item()
        metrics["latent_cosine"] = (
            F.normalize(predicted_answer_latent.mean(dim=1), dim=-1)
            * F.normalize(target_answer_latent.mean(dim=1), dim=-1)
        ).sum(dim=-1).mean().item()
        metrics.update(trace_metrics(trace_logits, batch))
        return metrics

    shuffled = evidence_latent[torch.randperm(evidence_latent.shape[0], device=device)]
    zeros = torch.zeros_like(evidence_latent)
    full = run_with(evidence_latent)
    shuffled_metrics = run_with(shuffled)
    no_evidence = run_with(zeros)
    teacher_forced_queries = (
        run_with(evidence_latent, teacher_force_queries=True) if config.reasoner_variant == "trace_multistep" else None
    )
    family_metrics: dict[str, dict[str, float]] = {}
    for index, family in enumerate(FAMILIES):
        mask = batch.family == index
        if not mask.any():
            continue
        pred_latent = reasoner(question_latent[mask], evidence_latent[mask])
        logits = text_codec.decode_answer(pred_latent)
        family_metrics[family] = token_metrics(logits, batch.answers[mask])
    result: dict[str, object] = {
        "full": full,
        "shuffled_evidence": shuffled_metrics,
        "no_evidence": no_evidence,
        "by_family": family_metrics,
    }
    if teacher_forced_queries is not None:
        result["teacher_forced_queries"] = teacher_forced_queries
    return result


def train_text_codec(model: TextLatentCodec, train: StageACSet, val: StageACSet, config: StageACConfig, device: torch.device) -> dict[str, object]:
    rng = random.Random(config.seed + 101)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []
    for step in range(1, config.text_steps + 1):
        model.train()
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        question_latent = model.encode_question(batch.prompts)
        answer_latent = model.encode_answer(batch.answers)
        loss = sequence_loss(model.decode_question(question_latent), batch.prompts) + sequence_loss(
            model.decode_answer(answer_latent), batch.answers
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.text_steps:
            metrics = evaluate_text_codec(model, val, config, device)
            metrics["step"] = float(step)
            metrics["loss"] = float(loss.detach().cpu())
            history.append(metrics)
    return {"history": history, "final": evaluate_text_codec(model, val, config, device)}


def train_evidence_codec(model: EvidenceCodec, train: StageACSet, val: StageACSet, config: StageACConfig, device: torch.device) -> dict[str, object]:
    rng = random.Random(config.seed + 202)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []
    for step in range(1, config.evidence_steps + 1):
        model.train()
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        _, outputs = model(batch.evidence)
        loss = evidence_loss(outputs, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.evidence_steps:
            metrics = evaluate_evidence_codec(model, val, device)
            metrics["step"] = float(step)
            metrics["loss"] = float(loss.detach().cpu())
            history.append(metrics)
    return {"history": history, "final": evaluate_evidence_codec(model, val, device)}


def set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def train_reasoner(
    text_codec: TextLatentCodec,
    evidence_codec: EvidenceCodec,
    reasoner: LatentReasoner,
    train: StageACSet,
    val: StageACSet,
    config: StageACConfig,
    device: torch.device,
) -> dict[str, object]:
    rng = random.Random(config.seed + 303)
    text_codec.to(device)
    evidence_codec.to(device)
    reasoner.to(device)
    set_requires_grad(text_codec, False)
    set_requires_grad(evidence_codec, False)
    optimizer = torch.optim.AdamW(reasoner.parameters(), lr=config.lr)
    history: list[dict[str, object]] = []
    stage_families = tuple(family for family in config.reasoner_stage_order if family in config.families)
    stage_data = {family: family_subset(train, family) for family in stage_families}
    stage_span = max(1, (config.reasoner_steps + max(1, len(stage_families)) - 1) // max(1, len(stage_families)))
    for step in range(1, config.reasoner_steps + 1):
        text_codec.eval()
        evidence_codec.eval()
        reasoner.train()
        active_stage_family: str | None = None
        if config.reasoner_training_mode == "mixed":
            source = train
        elif config.reasoner_training_mode == "staged":
            if not stage_families:
                raise ValueError("reasoner_training_mode=staged requires at least one stage family")
            active_stage_family = stage_families[min((step - 1) // stage_span, len(stage_families) - 1)]
            if config.reasoner_stage_replay_interval > 0 and step % config.reasoner_stage_replay_interval == 0:
                source = train
                active_stage_family = f"{active_stage_family}+mixed_replay"
            else:
                source = stage_data[active_stage_family]
        else:
            raise ValueError(f"unknown reasoner_training_mode: {config.reasoner_training_mode}")
        batch = random_batch(source, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            question_latent = text_codec.encode_question(batch.prompts)
            evidence_latent = evidence_codec.encode(batch.evidence)
            target_answer_latent = text_codec.encode_answer(batch.answers)
        route_target = batch.family if config.moe_teacher_forcing else None
        teacher_force_queries = config.query_teacher_forcing in ("train", "always")
        predicted_answer_latent, trace_logits = reasoner.forward_with_trace(
            question_latent,
            evidence_latent,
            route_target=route_target,
            batch=batch,
            teacher_force_queries=teacher_force_queries,
        )
        logits = text_codec.decode_answer(predicted_answer_latent)
        trace_loss, trace_loss_metrics = trace_supervision_loss(trace_logits, batch)
        reader_loss, reader_loss_metrics = reasoner_reader_loss(trace_logits, batch)
        loss = sequence_loss(logits, batch.answers) + config.latent_mse_weight * F.mse_loss(
            predicted_answer_latent, target_answer_latent
        ) + config.reasoner_trace_weight * trace_loss + config.reasoner_reader_weight * reader_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.reasoner_steps:
            metrics = evaluate_reasoner(text_codec, evidence_codec, reasoner, val, config, device)
            metrics["step"] = step
            metrics["loss"] = float(loss.detach().cpu())
            if active_stage_family is not None:
                metrics["stage_family"] = active_stage_family
            metrics.update(trace_loss_metrics)
            metrics.update(reader_loss_metrics)
            history.append(metrics)
    return {"history": history, "final": evaluate_reasoner(text_codec, evidence_codec, reasoner, val, config, device)}


def run_one(config: StageACConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = build_dataset("train", config.train_size, config)
    val = build_dataset("val", config.val_size, config)
    test = build_dataset("test", config.test_size, config)

    text_codec = TextLatentCodec(config)
    evidence_codec = EvidenceCodec(config)
    reasoner = LatentReasoner(config)

    text_train = train_text_codec(text_codec, train, val, config, device)
    evidence_train = train_evidence_codec(evidence_codec, train, val, config, device)
    reasoner_train = train_reasoner(text_codec, evidence_codec, reasoner, train, val, config, device)
    test_metrics = {
        "text_codec": evaluate_text_codec(text_codec, test, config, device),
        "evidence_codec": evaluate_evidence_codec(evidence_codec, test.to(device), device),
        "latent_reasoner": evaluate_reasoner(text_codec, evidence_codec, reasoner, test, config, device),
    }
    result = {
        "experiment": "omni_transformer_stage_ac_latent_reasoning",
        "seed": config.seed,
        "device": str(device),
        "config": asdict(config),
        "data": {
            "families": FAMILIES,
            "colors": COLORS,
            "shapes": SHAPES,
            "samples": test.examples[:8],
        },
        "training": {
            "text_codec": text_train,
            "evidence_codec": evidence_train,
            "latent_reasoner": reasoner_train,
        },
        "test": test_metrics,
        "cost": {
            "seconds": round(time.perf_counter() - started, 4),
        },
        "interpretation": {
            "pipeline": "question latent + evidence latent -> latent reasoner -> answer-token latent -> frozen answer decoder",
            "success_condition": "target answer exact should be high, while no-evidence and shuffled-evidence should collapse on evidence-dependent questions.",
            "not_tested": "This stage uses structured object-table evidence, not image pixels or pretrained multimodal experts.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def collect_numbers(value: object, prefix: tuple[str, ...] = ()) -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, child in value.items():
            result.update(collect_numbers(child, (*prefix, str(key))))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {".".join(prefix): float(value)}
    return {}


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    number_rows = [collect_numbers(run) for run in runs]
    keys = sorted(set().union(*(row.keys() for row in number_rows)))
    means: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in number_rows if key in row]
        if not values:
            continue
        means[key] = statistics.mean(values)
        stdevs[key] = statistics.pstdev(values) if len(values) > 1 else 0.0

    def mean(path: str) -> float | None:
        return means.get(path)

    summary = {
        "text_answer_exact": mean("test.text_codec.answer_sequence_exact"),
        "text_question_exact": mean("test.text_codec.question_sequence_exact"),
        "evidence_occupancy_exact": mean("test.evidence_codec.grid_occupancy_exact"),
        "evidence_color_accuracy": mean("test.evidence_codec.occupied_color_accuracy"),
        "evidence_shape_accuracy": mean("test.evidence_codec.occupied_shape_accuracy"),
        "evidence_count_table_exact": mean("test.evidence_codec.count_table_exact"),
        "latent_reasoner_full_answer_exact": mean("test.latent_reasoner.full.sequence_exact"),
        "latent_reasoner_full_answer_word_exact": mean("test.latent_reasoner.full.answer_word_exact"),
        "latent_reasoner_no_evidence_answer_word_exact": mean("test.latent_reasoner.no_evidence.answer_word_exact"),
        "latent_reasoner_shuffled_evidence_answer_word_exact": mean("test.latent_reasoner.shuffled_evidence.answer_word_exact"),
        "latent_reasoner_teacher_forced_query_answer_word_exact": mean(
            "test.latent_reasoner.teacher_forced_queries.answer_word_exact"
        ),
        "latent_reasoner_teacher_forced_query_sequence_exact": mean(
            "test.latent_reasoner.teacher_forced_queries.sequence_exact"
        ),
        "latent_reasoner_latent_cosine": mean("test.latent_reasoner.full.latent_cosine"),
        "latent_reasoner_latent_mse": mean("test.latent_reasoner.full.latent_mse"),
        "latent_reasoner_trace_cell_accuracy": mean("test.latent_reasoner.full.trace_cell_accuracy"),
        "latent_reasoner_trace_moe_gate_accuracy": mean("test.latent_reasoner.full.trace_moe_gate_accuracy"),
        "latent_reasoner_trace_moe_gate_entropy": mean("test.latent_reasoner.full.trace_moe_gate_entropy"),
        "latent_reasoner_trace_color_accuracy": mean("test.latent_reasoner.full.trace_color_accuracy"),
        "latent_reasoner_trace_shape_accuracy": mean("test.latent_reasoner.full.trace_shape_accuracy"),
        "latent_reasoner_trace_count_accuracy": mean("test.latent_reasoner.full.trace_count_accuracy"),
        "latent_reasoner_trace_relation_accuracy": mean("test.latent_reasoner.full.trace_relation_accuracy"),
        "latent_reasoner_trace_left_pair_accuracy": mean("test.latent_reasoner.full.trace_left_pair_accuracy"),
        "latent_reasoner_trace_right_pair_accuracy": mean("test.latent_reasoner.full.trace_right_pair_accuracy"),
        "latent_reasoner_trace_relation_op_accuracy": mean("test.latent_reasoner.full.trace_relation_op_accuracy"),
        "latent_reasoner_reader_occupancy_exact": mean("test.latent_reasoner.full.reader_grid_occupancy_exact"),
        "latent_reasoner_reader_color_accuracy": mean("test.latent_reasoner.full.reader_occupied_color_accuracy"),
        "latent_reasoner_reader_shape_accuracy": mean("test.latent_reasoner.full.reader_occupied_shape_accuracy"),
        "latent_reasoner_reader_count_table_exact": mean("test.latent_reasoner.full.reader_count_table_exact"),
        "latent_reasoner_reader_pair_occupancy_exact": mean("test.latent_reasoner.full.reader_pair_occupancy_exact"),
        "latent_reasoner_reader_pair_row_accuracy": mean("test.latent_reasoner.full.reader_pair_row_accuracy"),
        "latent_reasoner_reader_pair_col_accuracy": mean("test.latent_reasoner.full.reader_pair_col_accuracy"),
    }
    return {
        "experiment": "omni_transformer_stage_ac_latent_reasoning_sweep",
        "seeds": [run["seed"] for run in runs],
        "summary": summary,
        "means": means,
        "stdevs": stdevs,
        "runs": runs,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_ac_latent_reasoning/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_ac_latent_reasoning/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701")
    parser.add_argument("--train-size", type=int, default=StageACConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageACConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageACConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageACConfig.batch_size)
    parser.add_argument("--d-model", type=int, default=StageACConfig.d_model)
    parser.add_argument("--layers", type=int, default=StageACConfig.layers)
    parser.add_argument("--heads", type=int, default=StageACConfig.heads)
    parser.add_argument("--latent-tokens", type=int, default=StageACConfig.latent_tokens)
    parser.add_argument("--text-steps", type=int, default=StageACConfig.text_steps)
    parser.add_argument("--evidence-steps", type=int, default=StageACConfig.evidence_steps)
    parser.add_argument("--reasoner-steps", type=int, default=StageACConfig.reasoner_steps)
    parser.add_argument("--eval-every", type=int, default=StageACConfig.eval_every)
    parser.add_argument("--lr", type=float, default=StageACConfig.lr)
    parser.add_argument("--latent-mse-weight", type=float, default=StageACConfig.latent_mse_weight)
    parser.add_argument("--reasoner-trace-weight", type=float, default=StageACConfig.reasoner_trace_weight)
    parser.add_argument("--reasoner-reader-weight", type=float, default=StageACConfig.reasoner_reader_weight)
    parser.add_argument(
        "--reasoner-variant",
        choices=("plain", "readout", "active_read", "trace_multistep", "moe_active"),
        default=StageACConfig.reasoner_variant,
    )
    parser.add_argument(
        "--reasoner-training-mode",
        choices=("mixed", "staged"),
        default=StageACConfig.reasoner_training_mode,
    )
    parser.add_argument("--reasoner-stage-order", default=",".join(StageACConfig.reasoner_stage_order))
    parser.add_argument("--reasoner-stage-replay-interval", type=int, default=StageACConfig.reasoner_stage_replay_interval)
    parser.add_argument("--moe-teacher-forcing", action="store_true")
    parser.add_argument(
        "--query-teacher-forcing",
        choices=("none", "train", "always"),
        default=StageACConfig.query_teacher_forcing,
    )
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--evidence-latent-mode", choices=("abstract", "cell"), default=StageACConfig.evidence_latent_mode)
    args = parser.parse_args()

    seeds = parse_csv_ints(args.seeds)
    runs: list[dict[str, object]] = []
    for seed in seeds:
        config = StageACConfig(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            seed=seed,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            latent_tokens=args.latent_tokens,
            lr=args.lr,
            text_steps=args.text_steps,
            evidence_steps=args.evidence_steps,
            reasoner_steps=args.reasoner_steps,
            eval_every=args.eval_every,
            latent_mse_weight=args.latent_mse_weight,
            reasoner_trace_weight=args.reasoner_trace_weight,
            reasoner_reader_weight=args.reasoner_reader_weight,
            reasoner_variant=args.reasoner_variant,
            reasoner_training_mode=args.reasoner_training_mode,
            reasoner_stage_order=parse_csv_strings(args.reasoner_stage_order),
            reasoner_stage_replay_interval=args.reasoner_stage_replay_interval,
            moe_teacher_forcing=args.moe_teacher_forcing,
            query_teacher_forcing=args.query_teacher_forcing,
            families=parse_csv_strings(args.families),
            evidence_latent_mode=args.evidence_latent_mode,
        )
        output_path = args.output_dir / f"seed{seed}.json" if args.sweep else args.aggregate
        result = run_one(config, output_path)
        runs.append(result)
        print(json.dumps({"seed": seed, "test": result["test"], "cost": result["cost"]}, ensure_ascii=False, indent=2))
    if args.sweep:
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"aggregate": aggregate["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
