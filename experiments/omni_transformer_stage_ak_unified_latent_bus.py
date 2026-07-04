from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
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
    FAMILIES,
    PAD,
    PAIR_COUNT,
    RELATIONS,
    ROW_WORDS,
    SHAPES,
    VOCAB_SIZE,
    StageACConfig,
    StageACSet,
    build_dataset,
)

COUNT_CLASS_COUNT = 9
ANSWER_CLASS_COUNT = len(COLORS) + len(SHAPES) + COUNT_CLASS_COUNT + 2


@dataclass(frozen=True)
class StageAKConfig:
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 64
    seed: int = 20260701
    grid_size: int = 4
    max_objects: int = 8
    prompt_len: int = 18
    d_model: int = 96
    heads: int = 4
    layers: int = 2
    lr: float = 8e-4
    steps: int = 1600
    eval_every: int = 400
    answer_len: int = 6
    token_loss_weight: float = 1.0
    process_loss_weight: float = 0.0
    process_state_weight: float = 1.0
    truth_table_state_weight: float = 1.0
    temperature: float = 0.07
    slot_loss_weight: float = 1.0
    query_loss_weight: float = 1.0
    compare_loss_weight: float = 1.0
    model_compare_loss_weight: float = 0.5
    answer_loss_weight: float = 1.0
    selected_count_loss_weight: float = 0.0
    amp: bool = True
    families: tuple[str, ...] = ("relation_yes_no",)


class UnifiedLatentBus(nn.Module):
    def __init__(self, config: StageAKConfig) -> None:
        super().__init__()
        self.config = config
        feature_size = 1 + len(COLORS) + len(SHAPES) + config.grid_size + config.grid_size
        self.object_proj = nn.Sequential(
            nn.Linear(feature_size, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.pair_schema = nn.Parameter(torch.randn(PAIR_COUNT, config.d_model) * 0.02)
        self.cell_schema = nn.Parameter(torch.randn(config.grid_size * config.grid_size, config.d_model) * 0.02)
        self.count_schema = nn.Parameter(torch.randn(PAIR_COUNT, config.d_model) * 0.02)
        self.pair_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.cell_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.count_input_proj = nn.Linear(config.max_objects + 1, config.d_model)
        self.slot_norm = nn.LayerNorm(config.d_model)
        self.occupied_head = nn.Linear(config.d_model, 1)
        self.row_head = nn.Linear(config.d_model, config.grid_size)
        self.col_head = nn.Linear(config.d_model, config.grid_size)
        self.color_head = nn.Linear(config.d_model, len(COLORS))
        self.shape_head = nn.Linear(config.d_model, len(SHAPES))
        self.cell_occupied_head = nn.Linear(config.d_model, 1)
        self.cell_color_head = nn.Linear(config.d_model, len(COLORS))
        self.cell_shape_head = nn.Linear(config.d_model, len(SHAPES))
        self.count_head = nn.Linear(config.d_model, config.max_objects + 1)
        self.position_state = nn.Sequential(
            nn.Linear(config.grid_size * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.token_embed = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position_embed = nn.Parameter(torch.randn(config.prompt_len, config.d_model) * 0.02)
        self.text_encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.text_norm = nn.LayerNorm(config.d_model)
        self.left_query = nn.Linear(config.d_model, config.d_model)
        self.right_query = nn.Linear(config.d_model, config.d_model)
        self.cell_query = nn.Linear(config.d_model, config.d_model)
        self.count_query = nn.Linear(config.d_model, config.d_model)
        self.op_head = nn.Linear(config.d_model, len(RELATIONS))
        self.op_embed = nn.Embedding(len(RELATIONS), config.d_model)
        self.compare = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, 2),
        )
        delta_class_count = config.grid_size * 2 - 1
        self.delta_row_head = nn.Linear(config.d_model, delta_class_count)
        self.delta_col_head = nn.Linear(config.d_model, delta_class_count)
        self.truth_table_head = nn.Linear(config.d_model, 2)
        self.delta_value = nn.Linear(delta_class_count * 2, config.d_model, bias=False)
        self.truth_table_value = nn.Linear(2, config.d_model, bias=False)
        self.relation_process_norm = nn.LayerNorm(config.d_model)
        self.answer_writer = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, ANSWER_CLASS_COUNT),
        )
        self.answer_token_query = nn.Parameter(torch.randn(config.answer_len, config.d_model) * 0.02)
        answer_token_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.answer_token_decoder = nn.TransformerEncoder(answer_token_layer, num_layers=config.layers)
        self.answer_token_norm = nn.LayerNorm(config.d_model)
        self.answer_token_out = nn.Linear(config.d_model, VOCAB_SIZE)
        self.relation_answer_state = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )

    def encode_slots(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        objects = self.object_proj(evidence)
        pair_query = self.pair_schema.unsqueeze(0).expand(evidence.shape[0], -1, -1)
        pair_attended, _ = self.pair_attn(pair_query, objects, objects, need_weights=False)
        cell_query = self.cell_schema.unsqueeze(0).expand(evidence.shape[0], -1, -1)
        cell_attended, _ = self.cell_attn(cell_query, objects, objects, need_weights=False)
        occupied = evidence[..., :1]
        color = evidence[..., 1 : 1 + len(COLORS)]
        shape = evidence[..., 1 + len(COLORS) : 1 + len(COLORS) + len(SHAPES)]
        pair_features = (color.unsqueeze(-1) * shape.unsqueeze(-2)).reshape(evidence.shape[0], evidence.shape[1], PAIR_COUNT)
        counts = (pair_features * occupied).sum(dim=1).round().long().clamp(0, self.config.max_objects)
        count_attended = self.count_input_proj(F.one_hot(counts, num_classes=self.config.max_objects + 1).float())
        count_query = self.count_schema.unsqueeze(0).expand(evidence.shape[0], -1, -1)
        return (
            self.slot_norm(pair_query + pair_attended),
            self.slot_norm(cell_query + cell_attended),
            self.slot_norm(count_query + count_attended),
        )

    def encode_question(self, prompts: torch.Tensor) -> torch.Tensor:
        positions = self.position_embed.unsqueeze(0)[:, : prompts.shape[1]]
        encoded = self.text_encoder(self.token_embed(prompts) + positions)
        mask = (prompts != PAD).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.text_norm(pooled)

    def retrieval_logits(self, query: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        query = F.normalize(query, dim=-1)
        slots = F.normalize(slots, dim=-1)
        return torch.bmm(slots, query.unsqueeze(-1)).squeeze(-1) / self.config.temperature

    def selected_slot(self, logits: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(logits, dim=-1)
        return torch.bmm(weights.unsqueeze(1), slots).squeeze(1)

    def forced_slot(self, slots: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = F.one_hot(target.clamp_min(0), num_classes=slots.shape[1]).to(dtype=slots.dtype)
        return torch.bmm(weights.unsqueeze(1), slots).squeeze(1)

    def decoded_position_state(self, slot: torch.Tensor) -> torch.Tensor:
        return self.position_state(torch.cat((self.row_head(slot), self.col_head(slot)), dim=-1))

    def forward(self, batch: StageACSet, *, zero_evidence: bool = False) -> dict[str, torch.Tensor]:
        evidence = torch.zeros_like(batch.evidence) if zero_evidence else batch.evidence
        pair_slots, cell_slots, count_slots = self.encode_slots(evidence)
        question = self.encode_question(batch.prompts)
        left_logits = self.retrieval_logits(self.left_query(question), pair_slots)
        left_model = self.selected_slot(left_logits, pair_slots)
        left_teacher = self.forced_slot(pair_slots, batch.target_left_pair)
        right_logits = self.retrieval_logits(self.right_query(question), pair_slots)
        right_model = self.selected_slot(right_logits, pair_slots)
        right_teacher = self.forced_slot(pair_slots, batch.target_right_pair)
        cell_logits = self.retrieval_logits(self.cell_query(question), cell_slots)
        cell_model = self.selected_slot(cell_logits, cell_slots)
        cell_teacher = self.forced_slot(cell_slots, batch.target_cell)
        count_logits = self.retrieval_logits(self.count_query(question), count_slots)
        count_model = self.selected_slot(count_logits, count_slots)
        count_teacher = self.forced_slot(count_slots, batch.target_count_pair)
        op_logits = self.op_head(question)
        op_teacher = self.op_embed(batch.target_relation_op.clamp_min(0))
        op_model = self.op_embed(op_logits.argmax(dim=-1))
        teacher_context = torch.cat(
            (self.decoded_position_state(left_teacher), self.decoded_position_state(right_teacher), op_teacher),
            dim=-1,
        )
        model_context = torch.cat(
            (self.decoded_position_state(left_model), self.decoded_position_state(right_model), op_model),
            dim=-1,
        )
        teacher_relation, teacher_delta_row, teacher_delta_col, teacher_truth_table, teacher_learned_truth_table = (
            self.relation_process(
                teacher_context,
                op_logits,
                forced_delta_row=batch.target_delta_row,
                forced_delta_col=batch.target_delta_col,
                forced_op=batch.target_relation_op,
            )
        )
        model_relation, model_delta_row, model_delta_col, model_truth_table, model_learned_truth_table = self.relation_process(
            model_context,
            op_logits,
        )
        teacher_answer_context = torch.cat((question, cell_teacher, count_teacher, teacher_relation), dim=-1)
        model_answer_context = torch.cat((question, cell_model, count_model, model_relation), dim=-1)
        teacher_answer_tokens = self.decode_answer_tokens(question, cell_teacher, count_teacher, teacher_relation)
        model_answer_tokens = self.decode_answer_tokens(question, cell_model, count_model, model_relation)
        return {
            "slots": pair_slots,
            "cell_slots": cell_slots,
            "count_slots": count_slots,
            "occupied": self.occupied_head(pair_slots).squeeze(-1),
            "row": self.row_head(pair_slots),
            "col": self.col_head(pair_slots),
            "color": self.color_head(pair_slots),
            "shape": self.shape_head(pair_slots),
            "cell_occupied": self.cell_occupied_head(cell_slots).squeeze(-1),
            "cell_color": self.cell_color_head(cell_slots),
            "cell_shape": self.cell_shape_head(cell_slots),
            "count": self.count_head(count_slots),
            "left_pair": left_logits,
            "right_pair": right_logits,
            "cell": cell_logits,
            "count_pair": count_logits,
            "relation_op": op_logits,
            "teacher_compare": self.compare(teacher_context),
            "model_compare": self.compare(model_context),
            "teacher_delta_row": teacher_delta_row,
            "teacher_delta_col": teacher_delta_col,
            "teacher_truth_table": teacher_truth_table,
            "teacher_learned_truth_table": teacher_learned_truth_table,
            "model_delta_row": model_delta_row,
            "model_delta_col": model_delta_col,
            "model_truth_table": model_truth_table,
            "model_learned_truth_table": model_learned_truth_table,
            "teacher_count_value": self.count_head(count_teacher),
            "model_count_value": self.count_head(count_model),
            "teacher_answer": self.answer_writer(teacher_answer_context),
            "model_answer": self.answer_writer(model_answer_context),
            "teacher_answer_tokens": teacher_answer_tokens,
            "model_answer_tokens": model_answer_tokens,
        }

    def relation_process(
        self,
        context: torch.Tensor,
        op_logits: torch.Tensor,
        *,
        forced_delta_row: torch.Tensor | None = None,
        forced_delta_col: torch.Tensor | None = None,
        forced_op: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relation = self.relation_answer_state(context)
        delta_row = self.delta_row_head(relation)
        delta_col = self.delta_col_head(relation)
        learned_truth_table = self.truth_table_head(relation)
        truth_table = self.deterministic_truth_table(delta_row, delta_col, op_logits, forced_delta_row, forced_delta_col, forced_op)
        process_state = self.delta_value(torch.cat((delta_row, delta_col), dim=-1))
        truth_state = self.truth_table_value(F.softmax(truth_table, dim=-1))
        relation = self.relation_process_norm(
            relation
            + self.config.process_state_weight * process_state
            + self.config.truth_table_state_weight * truth_state
        )
        return relation, delta_row, delta_col, truth_table, learned_truth_table

    def deterministic_truth_table(
        self,
        delta_row_logits: torch.Tensor,
        delta_col_logits: torch.Tensor,
        op_logits: torch.Tensor,
        forced_delta_row: torch.Tensor | None,
        forced_delta_col: torch.Tensor | None,
        forced_op: torch.Tensor | None,
    ) -> torch.Tensor:
        center = self.config.grid_size - 1
        delta_row = forced_delta_row.clamp_min(0) if forced_delta_row is not None else delta_row_logits.argmax(dim=-1)
        delta_col = forced_delta_col.clamp_min(0) if forced_delta_col is not None else delta_col_logits.argmax(dim=-1)
        op = forced_op.clamp_min(0) if forced_op is not None else op_logits.argmax(dim=-1)
        truth = torch.zeros_like(op)
        truth = torch.where(op == 0, delta_col < center, truth.bool())
        truth = torch.where(op == 1, delta_col > center, truth)
        truth = torch.where(op == 2, delta_row < center, truth)
        truth = torch.where(op == 3, delta_row > center, truth)
        truth_logits = F.one_hot(truth.long(), num_classes=2).to(dtype=delta_row_logits.dtype)
        return truth_logits * 20.0 - 10.0

    def decode_answer_tokens(
        self,
        question: torch.Tensor,
        cell: torch.Tensor,
        count: torch.Tensor,
        relation: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.stack((question, cell, count, relation), dim=1)
        query = self.answer_token_query.unsqueeze(0).expand(question.shape[0], -1, -1)
        decoded = self.answer_token_decoder(torch.cat((context, query), dim=1))
        token_latent = self.answer_token_norm(decoded[:, -self.config.answer_len :])
        return self.answer_token_out(token_latent)


def optional_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = target >= 0
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def slot_loss(outputs: dict[str, torch.Tensor], batch: StageACSet) -> torch.Tensor:
    occupied_loss = F.binary_cross_entropy_with_logits(outputs["occupied"], batch.pair_occupied)
    single = batch.pair_single > 0.5
    row_loss = F.cross_entropy(outputs["row"][single], batch.pair_row[single]) if single.any() else outputs["row"].sum() * 0
    col_loss = F.cross_entropy(outputs["col"][single], batch.pair_col[single]) if single.any() else outputs["col"].sum() * 0
    count_loss = F.cross_entropy(outputs["count"].reshape(-1, outputs["count"].shape[-1]), batch.count_table.reshape(-1))
    pair_colors = torch.arange(PAIR_COUNT, device=batch.prompts.device) // len(SHAPES)
    pair_shapes = torch.arange(PAIR_COUNT, device=batch.prompts.device) % len(SHAPES)
    color_loss = F.cross_entropy(outputs["color"].reshape(-1, len(COLORS)), pair_colors.repeat(batch.prompts.shape[0]))
    shape_loss = F.cross_entropy(outputs["shape"].reshape(-1, len(SHAPES)), pair_shapes.repeat(batch.prompts.shape[0]))
    cell_occupied_loss = F.binary_cross_entropy_with_logits(outputs["cell_occupied"], batch.grid_occupied)
    occupied_cells = batch.grid_occupied > 0.5
    cell_color_loss = (
        F.cross_entropy(outputs["cell_color"][occupied_cells], batch.grid_color[occupied_cells])
        if occupied_cells.any()
        else outputs["cell_color"].sum() * 0
    )
    cell_shape_loss = (
        F.cross_entropy(outputs["cell_shape"][occupied_cells], batch.grid_shape[occupied_cells])
        if occupied_cells.any()
        else outputs["cell_shape"].sum() * 0
    )
    return (
        occupied_loss
        + row_loss
        + col_loss
        + count_loss
        + cell_occupied_loss
        + cell_color_loss
        + cell_shape_loss
        + 0.2 * (color_loss + shape_loss)
    )


def query_loss(outputs: dict[str, torch.Tensor], batch: StageACSet) -> torch.Tensor:
    return (
        optional_ce(outputs["left_pair"], batch.target_left_pair)
        + optional_ce(outputs["right_pair"], batch.target_right_pair)
        + optional_ce(outputs["cell"], batch.target_cell)
        + optional_ce(outputs["count_pair"], batch.target_count_pair)
        + optional_ce(outputs["relation_op"], batch.target_relation_op)
    )


def compare_loss(outputs: dict[str, torch.Tensor], batch: StageACSet, *, model: bool) -> torch.Tensor:
    name = "model_compare" if model else "teacher_compare"
    return optional_ce(outputs[name], batch.target_relation)


def process_loss(outputs: dict[str, torch.Tensor], batch: StageACSet, *, model: bool) -> torch.Tensor:
    prefix = "model" if model else "teacher"
    return (
        optional_ce(outputs[f"{prefix}_delta_row"], batch.target_delta_row)
        + optional_ce(outputs[f"{prefix}_delta_col"], batch.target_delta_col)
        + optional_ce(outputs[f"{prefix}_learned_truth_table"], batch.target_relation)
    )


def answer_loss(outputs: dict[str, torch.Tensor], batch: StageACSet, *, model: bool) -> torch.Tensor:
    name = "model_answer" if model else "teacher_answer"
    return optional_ce(outputs[name], answer_target(batch))


def sequence_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD)


def answer_token_loss(outputs: dict[str, torch.Tensor], batch: StageACSet, *, model: bool) -> torch.Tensor:
    name = "model_answer_tokens" if model else "teacher_answer_tokens"
    return sequence_loss(outputs[name], batch.answers)


def selected_count_loss(outputs: dict[str, torch.Tensor], batch: StageACSet, *, model: bool) -> torch.Tensor:
    name = "model_count_value" if model else "teacher_count_value"
    return optional_ce(outputs[name], batch.target_count)


def answer_target(batch: StageACSet) -> torch.Tensor:
    target = torch.full_like(batch.family, -100)
    color_mask = batch.family == FAMILIES.index("color_at_cell")
    shape_mask = batch.family == FAMILIES.index("shape_at_cell")
    count_mask = batch.family == FAMILIES.index("count_color_shape")
    relation_mask = batch.family == FAMILIES.index("relation_yes_no")
    target[color_mask] = batch.target_color[color_mask]
    target[shape_mask] = len(COLORS) + batch.target_shape[shape_mask]
    target[count_mask] = len(COLORS) + len(SHAPES) + batch.target_count[count_mask]
    target[relation_mask] = len(COLORS) + len(SHAPES) + COUNT_CLASS_COUNT + batch.target_relation[relation_mask]
    return target


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    mask = target >= 0
    if not mask.any():
        return 0.0
    return (logits.argmax(dim=-1)[mask] == target[mask]).float().mean().item()


def token_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    non_pad = target != PAD
    token_accuracy = (pred[non_pad] == target[non_pad]).float().mean().item() if non_pad.any() else 0.0
    sequence_exact = ((pred == target) | ~non_pad).all(dim=1).float().mean().item()
    answer_word_exact = (pred[:, 1] == target[:, 1]).float().mean().item()
    return {
        "token_accuracy": token_accuracy,
        "sequence_exact": sequence_exact,
        "answer_word_exact": answer_word_exact,
    }


def metrics(outputs: dict[str, torch.Tensor], batch: StageACSet) -> dict[str, float]:
    pair_pred = (torch.sigmoid(outputs["occupied"]) > 0.5).float()
    cell_pred = (torch.sigmoid(outputs["cell_occupied"]) > 0.5).float()
    single = batch.pair_single > 0.5
    occupied_cells = batch.grid_occupied > 0.5
    target_answer = answer_target(batch)
    count_table_ok = (outputs["count"].argmax(dim=-1) == batch.count_table).all(dim=1)
    teacher_token = token_metrics(outputs["teacher_answer_tokens"], batch.answers)
    model_token = token_metrics(outputs["model_answer_tokens"], batch.answers)
    result = {
        "pair_occupancy_exact": (pair_pred == batch.pair_occupied).all(dim=1).float().mean().item(),
        "cell_occupancy_exact": (cell_pred == batch.grid_occupied).all(dim=1).float().mean().item(),
        "left_pair_accuracy": accuracy(outputs["left_pair"], batch.target_left_pair),
        "right_pair_accuracy": accuracy(outputs["right_pair"], batch.target_right_pair),
        "cell_accuracy": accuracy(outputs["cell"], batch.target_cell),
        "count_pair_accuracy": accuracy(outputs["count_pair"], batch.target_count_pair),
        "relation_op_accuracy": accuracy(outputs["relation_op"], batch.target_relation_op),
        "teacher_compare_accuracy": accuracy(outputs["teacher_compare"], batch.target_relation),
        "model_compare_accuracy": accuracy(outputs["model_compare"], batch.target_relation),
        "teacher_delta_row_accuracy": accuracy(outputs["teacher_delta_row"], batch.target_delta_row),
        "teacher_delta_col_accuracy": accuracy(outputs["teacher_delta_col"], batch.target_delta_col),
        "teacher_truth_table_accuracy": accuracy(outputs["teacher_truth_table"], batch.target_relation),
        "teacher_learned_truth_table_accuracy": accuracy(outputs["teacher_learned_truth_table"], batch.target_relation),
        "model_delta_row_accuracy": accuracy(outputs["model_delta_row"], batch.target_delta_row),
        "model_delta_col_accuracy": accuracy(outputs["model_delta_col"], batch.target_delta_col),
        "model_truth_table_accuracy": accuracy(outputs["model_truth_table"], batch.target_relation),
        "model_learned_truth_table_accuracy": accuracy(outputs["model_learned_truth_table"], batch.target_relation),
        "teacher_answer_accuracy": accuracy(outputs["teacher_answer"], target_answer),
        "model_answer_accuracy": accuracy(outputs["model_answer"], target_answer),
        "teacher_answer_token_accuracy": teacher_token["token_accuracy"],
        "teacher_answer_sequence_exact": teacher_token["sequence_exact"],
        "teacher_answer_word_exact": teacher_token["answer_word_exact"],
        "model_answer_token_accuracy": model_token["token_accuracy"],
        "model_answer_sequence_exact": model_token["sequence_exact"],
        "model_answer_word_exact": model_token["answer_word_exact"],
        "teacher_count_value_accuracy": accuracy(outputs["teacher_count_value"], batch.target_count),
        "model_count_value_accuracy": accuracy(outputs["model_count_value"], batch.target_count),
    }
    if single.any():
        result["pair_row_accuracy"] = (outputs["row"].argmax(dim=-1)[single] == batch.pair_row[single]).float().mean().item()
        result["pair_col_accuracy"] = (outputs["col"].argmax(dim=-1)[single] == batch.pair_col[single]).float().mean().item()
    if occupied_cells.any():
        result["cell_color_accuracy"] = (
            outputs["cell_color"].argmax(dim=-1)[occupied_cells] == batch.grid_color[occupied_cells]
        ).float().mean().item()
        result["cell_shape_accuracy"] = (
            outputs["cell_shape"].argmax(dim=-1)[occupied_cells] == batch.grid_shape[occupied_cells]
        ).float().mean().item()
    result["count_table_exact"] = count_table_ok.float().mean().item()
    for family in FAMILIES:
        mask = batch.family == FAMILIES.index(family)
        if mask.any():
            result[f"answer_{family}_accuracy"] = (
                outputs["model_answer"].argmax(dim=-1)[mask] == target_answer[mask]
            ).float().mean().item()
            family_token = token_metrics(outputs["model_answer_tokens"][mask], batch.answers[mask])
            result[f"answer_{family}_token_accuracy"] = family_token["token_accuracy"]
            result[f"answer_{family}_sequence_exact"] = family_token["sequence_exact"]
            result[f"answer_{family}_word_exact"] = family_token["answer_word_exact"]
            result[f"count_table_{family}_exact"] = count_table_ok[mask].float().mean().item()
            if family == "count_color_shape":
                result[f"teacher_count_value_{family}_accuracy"] = (
                    outputs["teacher_count_value"].argmax(dim=-1)[mask] == batch.target_count[mask]
                ).float().mean().item()
                result[f"model_count_value_{family}_accuracy"] = (
                    outputs["model_count_value"].argmax(dim=-1)[mask] == batch.target_count[mask]
                ).float().mean().item()
            if family == "relation_yes_no":
                result[f"model_delta_row_{family}_accuracy"] = (
                    outputs["model_delta_row"].argmax(dim=-1)[mask] == batch.target_delta_row[mask]
                ).float().mean().item()
                result[f"model_delta_col_{family}_accuracy"] = (
                    outputs["model_delta_col"].argmax(dim=-1)[mask] == batch.target_delta_col[mask]
                ).float().mean().item()
                result[f"model_truth_table_{family}_accuracy"] = (
                    outputs["model_truth_table"].argmax(dim=-1)[mask] == batch.target_relation[mask]
                ).float().mean().item()
                result[f"model_learned_truth_table_{family}_accuracy"] = (
                    outputs["model_learned_truth_table"].argmax(dim=-1)[mask] == batch.target_relation[mask]
                ).float().mean().item()
    return result


def random_batch(data: StageACSet, *, rng: random.Random, batch_size: int, device: torch.device) -> StageACSet:
    indices = [rng.randrange(data.prompts.shape[0]) for _ in range(batch_size)]
    if data.prompts.device == device:
        return data.subset(indices)
    return data.subset(indices).to(device)


@torch.no_grad()
def evaluate(model: UnifiedLatentBus, data: StageACSet, config: StageAKConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    full = metrics(model(batch), batch)
    no_evidence = metrics(model(batch, zero_evidence=True), batch)
    return {f"full_{key}": value for key, value in full.items()} | {
        f"no_evidence_{key}": value for key, value in no_evidence.items()
    }


def run_one(config: StageAKConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    data_config = StageACConfig(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        grid_size=config.grid_size,
        max_objects=config.max_objects,
        prompt_len=config.prompt_len,
        answer_len=config.answer_len,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        families=config.families,
    )
    train = build_dataset("train", config.train_size, data_config).to(device)
    val = build_dataset("val", config.val_size, data_config).to(device)
    test = build_dataset("test", config.test_size, data_config).to(device)
    model = UnifiedLatentBus(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(config.seed + 404)
    history: list[dict[str, float]] = []
    for step in range(1, config.steps + 1):
        model.train()
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch)
            loss_slot = slot_loss(outputs, batch)
            loss_query = query_loss(outputs, batch)
            loss_teacher = compare_loss(outputs, batch, model=False)
            loss_model = compare_loss(outputs, batch, model=True)
            loss_process_teacher = process_loss(outputs, batch, model=False)
            loss_process_model = process_loss(outputs, batch, model=True)
            loss_answer_teacher = answer_loss(outputs, batch, model=False)
            loss_answer_model = answer_loss(outputs, batch, model=True)
            loss_token_teacher = answer_token_loss(outputs, batch, model=False)
            loss_token_model = answer_token_loss(outputs, batch, model=True)
            loss_count_teacher = selected_count_loss(outputs, batch, model=False)
            loss_count_model = selected_count_loss(outputs, batch, model=True)
            loss = (
                config.slot_loss_weight * loss_slot
                + config.query_loss_weight * loss_query
                + config.compare_loss_weight * loss_teacher
                + config.model_compare_loss_weight * loss_model
                + config.process_loss_weight * (loss_process_teacher + loss_process_model)
                + config.answer_loss_weight * (loss_answer_teacher + loss_answer_model)
                + config.token_loss_weight * (loss_token_teacher + loss_token_model)
                + config.selected_count_loss_weight * (loss_count_teacher + loss_count_model)
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            row = evaluate(model, val, config, device)
            row["step"] = float(step)
            row["loss"] = float(loss.detach().cpu())
            row["loss_slot"] = float(loss_slot.detach().cpu())
            row["loss_query"] = float(loss_query.detach().cpu())
            row["loss_teacher_compare"] = float(loss_teacher.detach().cpu())
            row["loss_model_compare"] = float(loss_model.detach().cpu())
            row["loss_teacher_process"] = float(loss_process_teacher.detach().cpu())
            row["loss_model_process"] = float(loss_process_model.detach().cpu())
            row["loss_teacher_answer"] = float(loss_answer_teacher.detach().cpu())
            row["loss_model_answer"] = float(loss_answer_model.detach().cpu())
            row["loss_teacher_answer_tokens"] = float(loss_token_teacher.detach().cpu())
            row["loss_model_answer_tokens"] = float(loss_token_model.detach().cpu())
            row["loss_teacher_count"] = float(loss_count_teacher.detach().cpu())
            row["loss_model_count"] = float(loss_count_model.detach().cpu())
            history.append(row)
    test_metrics = evaluate(model, test, config, device)
    result = {
        "experiment": "omni_transformer_stage_ak_unified_latent_bus",
        "seed": config.seed,
        "device": str(device),
            "config": asdict(config),
            "training": {"history": history, "final": history[-1]},
            "test": test_metrics,
            "cost": {"seconds": round(time.perf_counter() - started, 4), "amp": use_amp},
            "interpretation": {
                "goal": "Train first-class pair/cell/count latent slots before final answer-token reasoning.",
                "success_condition": "Stage AN uses answer-token exact and no-evidence gap as the primary gate; class head metrics remain diagnostic.",
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
    rows = [collect_numbers(run) for run in runs]
    keys = sorted(set().union(*(row.keys() for row in rows)))
    means = {key: statistics.mean(row[key] for row in rows if key in row) for key in keys}
    summary = {
        "full_pair_occupancy_exact": means.get("test.full_pair_occupancy_exact"),
        "full_cell_occupancy_exact": means.get("test.full_cell_occupancy_exact"),
        "full_pair_row_accuracy": means.get("test.full_pair_row_accuracy"),
        "full_pair_col_accuracy": means.get("test.full_pair_col_accuracy"),
        "full_cell_color_accuracy": means.get("test.full_cell_color_accuracy"),
        "full_cell_shape_accuracy": means.get("test.full_cell_shape_accuracy"),
        "full_count_table_exact": means.get("test.full_count_table_exact"),
        "full_left_pair_accuracy": means.get("test.full_left_pair_accuracy"),
        "full_right_pair_accuracy": means.get("test.full_right_pair_accuracy"),
        "full_cell_accuracy": means.get("test.full_cell_accuracy"),
        "full_count_pair_accuracy": means.get("test.full_count_pair_accuracy"),
        "full_relation_op_accuracy": means.get("test.full_relation_op_accuracy"),
        "full_teacher_compare_accuracy": means.get("test.full_teacher_compare_accuracy"),
        "full_model_compare_accuracy": means.get("test.full_model_compare_accuracy"),
        "full_model_delta_row_accuracy": means.get("test.full_model_delta_row_accuracy"),
        "full_model_delta_col_accuracy": means.get("test.full_model_delta_col_accuracy"),
        "full_model_truth_table_accuracy": means.get("test.full_model_truth_table_accuracy"),
        "full_teacher_answer_accuracy": means.get("test.full_teacher_answer_accuracy"),
        "full_model_answer_accuracy": means.get("test.full_model_answer_accuracy"),
        "full_model_answer_token_accuracy": means.get("test.full_model_answer_token_accuracy"),
        "full_model_answer_sequence_exact": means.get("test.full_model_answer_sequence_exact"),
        "full_model_answer_word_exact": means.get("test.full_model_answer_word_exact"),
        "full_teacher_count_value_accuracy": means.get("test.full_teacher_count_value_accuracy"),
        "full_model_count_value_accuracy": means.get("test.full_model_count_value_accuracy"),
        "no_evidence_model_compare_accuracy": means.get("test.no_evidence_model_compare_accuracy"),
        "no_evidence_model_truth_table_accuracy": means.get("test.no_evidence_model_truth_table_accuracy"),
        "no_evidence_model_answer_accuracy": means.get("test.no_evidence_model_answer_accuracy"),
        "no_evidence_model_answer_sequence_exact": means.get("test.no_evidence_model_answer_sequence_exact"),
        "no_evidence_model_answer_word_exact": means.get("test.no_evidence_model_answer_word_exact"),
        "answer_color_at_cell_accuracy": means.get("test.full_answer_color_at_cell_accuracy"),
        "answer_shape_at_cell_accuracy": means.get("test.full_answer_shape_at_cell_accuracy"),
        "answer_count_color_shape_accuracy": means.get("test.full_answer_count_color_shape_accuracy"),
        "answer_relation_yes_no_accuracy": means.get("test.full_answer_relation_yes_no_accuracy"),
        "answer_color_at_cell_sequence_exact": means.get("test.full_answer_color_at_cell_sequence_exact"),
        "answer_shape_at_cell_sequence_exact": means.get("test.full_answer_shape_at_cell_sequence_exact"),
        "answer_count_color_shape_sequence_exact": means.get("test.full_answer_count_color_shape_sequence_exact"),
        "answer_relation_yes_no_sequence_exact": means.get("test.full_answer_relation_yes_no_sequence_exact"),
        "relation_delta_row_accuracy": means.get("test.full_model_delta_row_relation_yes_no_accuracy"),
        "relation_delta_col_accuracy": means.get("test.full_model_delta_col_relation_yes_no_accuracy"),
        "relation_truth_table_accuracy": means.get("test.full_model_truth_table_relation_yes_no_accuracy"),
        "no_evidence_relation_truth_table_accuracy": means.get("test.no_evidence_model_truth_table_relation_yes_no_accuracy"),
        "count_table_count_color_shape_exact": means.get("test.full_count_table_count_color_shape_exact"),
        "teacher_count_value_count_color_shape_accuracy": means.get(
            "test.full_teacher_count_value_count_color_shape_accuracy"
        ),
        "model_count_value_count_color_shape_accuracy": means.get("test.full_model_count_value_count_color_shape_accuracy"),
    }
    return {
        "experiment": "omni_transformer_stage_ak_unified_latent_bus_sweep",
        "seeds": [run["seed"] for run in runs],
        "summary": summary,
        "means": means,
        "runs": runs,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_ak_unified_latent_bus/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_ak_unified_latent_bus/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701")
    parser.add_argument("--train-size", type=int, default=StageAKConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageAKConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageAKConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageAKConfig.batch_size)
    parser.add_argument("--d-model", type=int, default=StageAKConfig.d_model)
    parser.add_argument("--layers", type=int, default=StageAKConfig.layers)
    parser.add_argument("--heads", type=int, default=StageAKConfig.heads)
    parser.add_argument("--steps", type=int, default=StageAKConfig.steps)
    parser.add_argument("--eval-every", type=int, default=StageAKConfig.eval_every)
    parser.add_argument("--answer-len", type=int, default=StageAKConfig.answer_len)
    parser.add_argument("--lr", type=float, default=StageAKConfig.lr)
    parser.add_argument("--token-loss-weight", type=float, default=StageAKConfig.token_loss_weight)
    parser.add_argument("--process-loss-weight", type=float, default=StageAKConfig.process_loss_weight)
    parser.add_argument("--process-state-weight", type=float, default=StageAKConfig.process_state_weight)
    parser.add_argument("--truth-table-state-weight", type=float, default=StageAKConfig.truth_table_state_weight)
    parser.add_argument("--slot-loss-weight", type=float, default=StageAKConfig.slot_loss_weight)
    parser.add_argument("--query-loss-weight", type=float, default=StageAKConfig.query_loss_weight)
    parser.add_argument("--compare-loss-weight", type=float, default=StageAKConfig.compare_loss_weight)
    parser.add_argument("--model-compare-loss-weight", type=float, default=StageAKConfig.model_compare_loss_weight)
    parser.add_argument("--answer-loss-weight", type=float, default=StageAKConfig.answer_loss_weight)
    parser.add_argument("--selected-count-loss-weight", type=float, default=StageAKConfig.selected_count_loss_weight)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=StageAKConfig.amp)
    parser.add_argument("--families", default=",".join(StageAKConfig.families))
    args = parser.parse_args()

    runs: list[dict[str, object]] = []
    for seed in parse_csv_ints(args.seeds):
        config = StageAKConfig(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            seed=seed,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            steps=args.steps,
            eval_every=args.eval_every,
            answer_len=args.answer_len,
            lr=args.lr,
            token_loss_weight=args.token_loss_weight,
            process_loss_weight=args.process_loss_weight,
            process_state_weight=args.process_state_weight,
            truth_table_state_weight=args.truth_table_state_weight,
            slot_loss_weight=args.slot_loss_weight,
            query_loss_weight=args.query_loss_weight,
            compare_loss_weight=args.compare_loss_weight,
            model_compare_loss_weight=args.model_compare_loss_weight,
            answer_loss_weight=args.answer_loss_weight,
            selected_count_loss_weight=args.selected_count_loss_weight,
            amp=args.amp,
            families=parse_csv_strings(args.families),
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
