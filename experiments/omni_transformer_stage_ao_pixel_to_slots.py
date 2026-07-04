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
    COL_WORDS,
    COLORS,
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
from omni_transformer_stage_ak_unified_latent_bus import (  # noqa: E402
    ANSWER_CLASS_COUNT,
    StageAKConfig,
    UnifiedLatentBus,
    accuracy,
    answer_loss,
    answer_target,
    answer_token_loss,
    compare_loss,
    optional_ce,
    process_loss,
    query_loss,
    selected_count_loss,
    token_metrics,
)


@dataclass(frozen=True)
class StageAOConfig:
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 256
    seed: int = 20260701
    grid_size: int = 4
    max_objects: int = 8
    prompt_len: int = 18
    image_size: int = 64
    d_model: int = 96
    heads: int = 4
    layers: int = 2
    lr: float = 8e-4
    steps: int = 2400
    eval_every: int = 600
    answer_len: int = 6
    token_loss_weight: float = 1.0
    process_loss_weight: float = 1.0
    process_state_weight: float = 1.0
    truth_table_state_weight: float = 4.0
    temperature: float = 0.07
    slot_loss_weight: float = 1.0
    pixel_loss_weight: float = 1.0
    query_loss_weight: float = 1.0
    compare_loss_weight: float = 1.0
    model_compare_loss_weight: float = 0.5
    answer_loss_weight: float = 1.0
    selected_count_loss_weight: float = 0.25
    amp: bool = True
    families: tuple[str, ...] = ("color_at_cell", "shape_at_cell", "count_color_shape", "relation_yes_no")


@dataclass(frozen=True)
class StageAOSet:
    base: StageACSet
    images: torch.Tensor

    def subset(self, indices: list[int]) -> "StageAOSet":
        return StageAOSet(base=self.base.subset(indices), images=self.images[indices])

    def to(self, device: torch.device) -> "StageAOSet":
        return StageAOSet(base=self.base.to(device), images=self.images.to(device=device, dtype=torch.float32))

    @property
    def prompts(self) -> torch.Tensor:
        return self.base.prompts

    @property
    def answers(self) -> torch.Tensor:
        return self.base.answers

    @property
    def examples(self) -> list[dict[str, object]]:
        return self.base.examples


PALETTE = torch.tensor(
    [
        [0.92, 0.16, 0.12],
        [0.12, 0.44, 0.92],
        [0.12, 0.68, 0.30],
        [0.94, 0.74, 0.10],
    ],
    dtype=torch.float32,
)
BACKGROUND = torch.tensor([0.05, 0.06, 0.07], dtype=torch.float32)
GRID_LINE = torch.tensor([0.24, 0.25, 0.27], dtype=torch.float32)


def pair_index(color: int, shape: int) -> int:
    return color * len(SHAPES) + shape


def render_symbol(image: torch.Tensor, row: int, col: int, color: int, shape: int, *, cell_size: int) -> None:
    y0 = row * cell_size
    x0 = col * cell_size
    yy = torch.arange(cell_size).view(-1, 1).expand(cell_size, cell_size)
    xx = torch.arange(cell_size).view(1, -1).expand(cell_size, cell_size)
    center = (cell_size - 1) / 2
    radius = cell_size * 0.28
    inset = max(2, cell_size // 5)
    if shape == 0:
        mask = (yy - center).pow(2) + (xx - center).pow(2) <= radius**2
    elif shape == 1:
        mask = (yy >= inset) & (yy < cell_size - inset) & (xx >= inset) & (xx < cell_size - inset)
    elif shape == 2:
        mask = (yy >= inset) & (xx >= inset) & (xx <= cell_size - inset) & (yy <= (cell_size - inset - (xx - inset).abs() * 0.6))
    else:
        mask = (yy - center).abs() + (xx - center).abs() <= radius
    color_value = PALETTE[color].view(3, 1)
    patch = image[:, y0 : y0 + cell_size, x0 : x0 + cell_size]
    patch[:, mask] = color_value


def render_example(example: dict[str, object], config: StageAOConfig) -> torch.Tensor:
    image = BACKGROUND.view(3, 1, 1).expand(3, config.image_size, config.image_size).clone()
    cell_size = config.image_size // config.grid_size
    for offset in range(0, config.image_size, cell_size):
        image[:, offset : offset + 1, :] = GRID_LINE.view(3, 1, 1)
        image[:, :, offset : offset + 1] = GRID_LINE.view(3, 1, 1)
    for obj in example["objects"]:  # type: ignore[index]
        item = obj  # type: ignore[assignment]
        render_symbol(
            image,
            int(item["row"]),  # type: ignore[index]
            int(item["col"]),  # type: ignore[index]
            int(item["color"]),  # type: ignore[index]
            int(item["shape"]),  # type: ignore[index]
            cell_size=cell_size,
        )
    return image


def build_pixel_dataset(split: str, size: int, config: StageAOConfig) -> StageAOSet:
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
    base = build_dataset(split, size, data_config)
    images = torch.stack([render_example(example, config) for example in base.examples])
    return StageAOSet(base=base, images=images)


class PixelUnifiedLatentBus(UnifiedLatentBus):
    def __init__(self, config: StageAOConfig) -> None:
        ak_config = StageAKConfig(
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
            batch_size=config.batch_size,
            seed=config.seed,
            grid_size=config.grid_size,
            max_objects=config.max_objects,
            prompt_len=config.prompt_len,
            d_model=config.d_model,
            heads=config.heads,
            layers=config.layers,
            lr=config.lr,
            steps=config.steps,
            eval_every=config.eval_every,
            answer_len=config.answer_len,
            token_loss_weight=config.token_loss_weight,
            process_loss_weight=config.process_loss_weight,
            process_state_weight=config.process_state_weight,
            truth_table_state_weight=config.truth_table_state_weight,
            temperature=config.temperature,
            slot_loss_weight=config.slot_loss_weight,
            query_loss_weight=config.query_loss_weight,
            compare_loss_weight=config.compare_loss_weight,
            model_compare_loss_weight=config.model_compare_loss_weight,
            answer_loss_weight=config.answer_loss_weight,
            selected_count_loss_weight=config.selected_count_loss_weight,
            amp=config.amp,
            families=config.families,
        )
        super().__init__(ak_config)
        self.ao_config = config
        cell_size = config.image_size // config.grid_size
        patch_dim = 3 * cell_size * cell_size
        self.cell_patch_encoder = nn.Sequential(
            nn.Linear(patch_dim, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.pixel_cell_occupied = nn.Linear(config.d_model, 1)
        self.pixel_cell_color = nn.Linear(config.d_model, len(COLORS))
        self.pixel_cell_shape = nn.Linear(config.d_model, len(SHAPES))
        self.pixel_pair_position = nn.Linear(config.grid_size * 2, config.d_model)
        self.pixel_pair_norm = nn.LayerNorm(config.d_model)
        self.pixel_count_norm = nn.LayerNorm(config.d_model)
        self.count_value_embed = nn.Linear(config.max_objects + 1, config.d_model, bias=False)

    def cell_features(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        cell_size = self.ao_config.image_size // self.ao_config.grid_size
        patches = (
            images.unfold(2, cell_size, cell_size)
            .unfold(3, cell_size, cell_size)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(batch_size, self.ao_config.grid_size * self.ao_config.grid_size, -1)
        )
        return self.cell_patch_encoder(patches)

    def encode_slots_from_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        cell_features = self.cell_features(images)
        cell_slots = self.slot_norm(self.cell_schema.unsqueeze(0) + cell_features)
        occupied_logits = self.pixel_cell_occupied(cell_features).squeeze(-1)
        color_logits = self.pixel_cell_color(cell_features)
        shape_logits = self.pixel_cell_shape(cell_features)
        occupied_prob = torch.sigmoid(occupied_logits).unsqueeze(-1)
        color_prob = F.softmax(color_logits, dim=-1)
        shape_prob = F.softmax(shape_logits, dim=-1)
        pair_prob = (color_prob.unsqueeze(-1) * shape_prob.unsqueeze(-2)).reshape(images.shape[0], cell_slots.shape[1], PAIR_COUNT)
        pair_cell_weight = occupied_prob * pair_prob
        row_ids = torch.arange(self.ao_config.grid_size, device=images.device).repeat_interleave(self.ao_config.grid_size)
        col_ids = torch.arange(self.ao_config.grid_size, device=images.device).repeat(self.ao_config.grid_size)
        row_onehot = F.one_hot(row_ids, num_classes=self.ao_config.grid_size).float()
        col_onehot = F.one_hot(col_ids, num_classes=self.ao_config.grid_size).float()
        position = torch.cat((row_onehot, col_onehot), dim=-1)
        position_latent = self.pixel_pair_position(position)
        pair_weight_sum = pair_cell_weight.sum(dim=1).unsqueeze(-1).clamp_min(1e-4)
        pair_latent = torch.bmm(pair_cell_weight.transpose(1, 2), cell_features + position_latent.unsqueeze(0)) / pair_weight_sum
        pair_slots = self.pixel_pair_norm(self.pair_schema.unsqueeze(0) + pair_latent)
        soft_counts = pair_cell_weight.sum(dim=1)
        count_classes = torch.arange(self.ao_config.max_objects + 1, device=images.device, dtype=soft_counts.dtype)
        count_logits = -((soft_counts.unsqueeze(-1) - count_classes.view(1, 1, -1)) ** 2)
        count_dist = F.softmax(count_logits * 4.0, dim=-1)
        count_slots = self.pixel_count_norm(self.count_schema.unsqueeze(0) + self.count_value_embed(count_dist))
        pixel = {
            "pixel_cell_occupied": occupied_logits,
            "pixel_cell_color": color_logits,
            "pixel_cell_shape": shape_logits,
            "pixel_count_table": count_logits,
        }
        return pair_slots, cell_slots, count_slots, pixel

    def forward(self, batch: StageAOSet, *, zero_image: bool = False) -> dict[str, torch.Tensor]:
        images = torch.zeros_like(batch.images) if zero_image else batch.images
        pair_slots, cell_slots, count_slots, pixel = self.encode_slots_from_image(images)
        question = self.encode_question(batch.prompts)
        left_logits = self.retrieval_logits(self.left_query(question), pair_slots)
        left_model = self.selected_slot(left_logits, pair_slots)
        left_teacher = self.forced_slot(pair_slots, batch.base.target_left_pair)
        right_logits = self.retrieval_logits(self.right_query(question), pair_slots)
        right_model = self.selected_slot(right_logits, pair_slots)
        right_teacher = self.forced_slot(pair_slots, batch.base.target_right_pair)
        cell_logits = self.retrieval_logits(self.cell_query(question), cell_slots)
        cell_model = self.selected_slot(cell_logits, cell_slots)
        cell_teacher = self.forced_slot(cell_slots, batch.base.target_cell)
        count_logits = self.retrieval_logits(self.count_query(question), count_slots)
        count_model = self.selected_slot(count_logits, count_slots)
        count_teacher = self.forced_slot(count_slots, batch.base.target_count_pair)
        op_logits = self.op_head(question)
        op_teacher = self.op_embed(batch.base.target_relation_op.clamp_min(0))
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
                forced_delta_row=batch.base.target_delta_row,
                forced_delta_col=batch.base.target_delta_col,
                forced_op=batch.base.target_relation_op,
            )
        )
        model_relation, model_delta_row, model_delta_col, model_truth_table, model_learned_truth_table = self.relation_process(
            model_context,
            op_logits,
        )
        teacher_answer_context = torch.cat((question, cell_teacher, count_teacher, teacher_relation), dim=-1)
        model_answer_context = torch.cat((question, cell_model, count_model, model_relation), dim=-1)
        outputs = {
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
            "teacher_answer_tokens": self.decode_answer_tokens(question, cell_teacher, count_teacher, teacher_relation),
            "model_answer_tokens": self.decode_answer_tokens(question, cell_model, count_model, model_relation),
        }
        outputs.update(pixel)
        return outputs


def pixel_loss(outputs: dict[str, torch.Tensor], batch: StageAOSet) -> torch.Tensor:
    base = batch.base
    occupied_loss = F.binary_cross_entropy_with_logits(outputs["pixel_cell_occupied"], base.grid_occupied)
    occupied_cells = base.grid_occupied > 0.5
    color_loss = (
        F.cross_entropy(outputs["pixel_cell_color"][occupied_cells], base.grid_color[occupied_cells])
        if occupied_cells.any()
        else outputs["pixel_cell_color"].sum() * 0
    )
    shape_loss = (
        F.cross_entropy(outputs["pixel_cell_shape"][occupied_cells], base.grid_shape[occupied_cells])
        if occupied_cells.any()
        else outputs["pixel_cell_shape"].sum() * 0
    )
    count_loss = F.cross_entropy(outputs["pixel_count_table"].reshape(-1, outputs["pixel_count_table"].shape[-1]), base.count_table.reshape(-1))
    return occupied_loss + color_loss + shape_loss + count_loss


def slot_loss(outputs: dict[str, torch.Tensor], batch: StageAOSet) -> torch.Tensor:
    base = batch.base
    occupied_loss = F.binary_cross_entropy_with_logits(outputs["occupied"], base.pair_occupied)
    single = base.pair_single > 0.5
    row_loss = F.cross_entropy(outputs["row"][single], base.pair_row[single]) if single.any() else outputs["row"].sum() * 0
    col_loss = F.cross_entropy(outputs["col"][single], base.pair_col[single]) if single.any() else outputs["col"].sum() * 0
    count_loss = F.cross_entropy(outputs["count"].reshape(-1, outputs["count"].shape[-1]), base.count_table.reshape(-1))
    pair_colors = torch.arange(PAIR_COUNT, device=base.prompts.device) // len(SHAPES)
    pair_shapes = torch.arange(PAIR_COUNT, device=base.prompts.device) % len(SHAPES)
    color_loss = F.cross_entropy(outputs["color"].reshape(-1, len(COLORS)), pair_colors.repeat(base.prompts.shape[0]))
    shape_loss = F.cross_entropy(outputs["shape"].reshape(-1, len(SHAPES)), pair_shapes.repeat(base.prompts.shape[0]))
    cell_occupied_loss = F.binary_cross_entropy_with_logits(outputs["cell_occupied"], base.grid_occupied)
    occupied_cells = base.grid_occupied > 0.5
    cell_color_loss = (
        F.cross_entropy(outputs["cell_color"][occupied_cells], base.grid_color[occupied_cells])
        if occupied_cells.any()
        else outputs["cell_color"].sum() * 0
    )
    cell_shape_loss = (
        F.cross_entropy(outputs["cell_shape"][occupied_cells], base.grid_shape[occupied_cells])
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


def ao_metrics(outputs: dict[str, torch.Tensor], batch: StageAOSet) -> dict[str, float]:
    base = batch.base
    pair_pred = (torch.sigmoid(outputs["occupied"]) > 0.5).float()
    cell_pred = (torch.sigmoid(outputs["cell_occupied"]) > 0.5).float()
    pixel_cell_pred = (torch.sigmoid(outputs["pixel_cell_occupied"]) > 0.5).float()
    single = base.pair_single > 0.5
    occupied_cells = base.grid_occupied > 0.5
    target_answer = answer_target(base)
    count_table_ok = (outputs["count"].argmax(dim=-1) == base.count_table).all(dim=1)
    pixel_count_table_ok = (outputs["pixel_count_table"].argmax(dim=-1) == base.count_table).all(dim=1)
    teacher_token = token_metrics(outputs["teacher_answer_tokens"], base.answers)
    model_token = token_metrics(outputs["model_answer_tokens"], base.answers)
    result = {
        "pixel_cell_occupancy_exact": (pixel_cell_pred == base.grid_occupied).all(dim=1).float().mean().item(),
        "pixel_count_table_exact": pixel_count_table_ok.float().mean().item(),
        "pair_occupancy_exact": (pair_pred == base.pair_occupied).all(dim=1).float().mean().item(),
        "cell_occupancy_exact": (cell_pred == base.grid_occupied).all(dim=1).float().mean().item(),
        "left_pair_accuracy": accuracy(outputs["left_pair"], base.target_left_pair),
        "right_pair_accuracy": accuracy(outputs["right_pair"], base.target_right_pair),
        "cell_accuracy": accuracy(outputs["cell"], base.target_cell),
        "count_pair_accuracy": accuracy(outputs["count_pair"], base.target_count_pair),
        "relation_op_accuracy": accuracy(outputs["relation_op"], base.target_relation_op),
        "teacher_compare_accuracy": accuracy(outputs["teacher_compare"], base.target_relation),
        "model_compare_accuracy": accuracy(outputs["model_compare"], base.target_relation),
        "model_truth_table_accuracy": accuracy(outputs["model_truth_table"], base.target_relation),
        "teacher_answer_accuracy": accuracy(outputs["teacher_answer"], target_answer),
        "model_answer_accuracy": accuracy(outputs["model_answer"], target_answer),
        "teacher_answer_token_accuracy": teacher_token["token_accuracy"],
        "teacher_answer_sequence_exact": teacher_token["sequence_exact"],
        "model_answer_token_accuracy": model_token["token_accuracy"],
        "model_answer_sequence_exact": model_token["sequence_exact"],
        "model_answer_word_exact": model_token["answer_word_exact"],
        "teacher_count_value_accuracy": accuracy(outputs["teacher_count_value"], base.target_count),
        "model_count_value_accuracy": accuracy(outputs["model_count_value"], base.target_count),
        "count_table_exact": count_table_ok.float().mean().item(),
    }
    if occupied_cells.any():
        result["pixel_cell_color_accuracy"] = (
            outputs["pixel_cell_color"].argmax(dim=-1)[occupied_cells] == base.grid_color[occupied_cells]
        ).float().mean().item()
        result["pixel_cell_shape_accuracy"] = (
            outputs["pixel_cell_shape"].argmax(dim=-1)[occupied_cells] == base.grid_shape[occupied_cells]
        ).float().mean().item()
        result["cell_color_accuracy"] = (
            outputs["cell_color"].argmax(dim=-1)[occupied_cells] == base.grid_color[occupied_cells]
        ).float().mean().item()
        result["cell_shape_accuracy"] = (
            outputs["cell_shape"].argmax(dim=-1)[occupied_cells] == base.grid_shape[occupied_cells]
        ).float().mean().item()
    if single.any():
        result["pair_row_accuracy"] = (outputs["row"].argmax(dim=-1)[single] == base.pair_row[single]).float().mean().item()
        result["pair_col_accuracy"] = (outputs["col"].argmax(dim=-1)[single] == base.pair_col[single]).float().mean().item()
    for family in FAMILIES:
        mask = base.family == FAMILIES.index(family)
        if mask.any():
            family_token = token_metrics(outputs["model_answer_tokens"][mask], base.answers[mask])
            result[f"answer_{family}_sequence_exact"] = family_token["sequence_exact"]
            result[f"answer_{family}_word_exact"] = family_token["answer_word_exact"]
            result[f"answer_{family}_accuracy"] = (
                outputs["model_answer"].argmax(dim=-1)[mask] == target_answer[mask]
            ).float().mean().item()
            result[f"count_table_{family}_exact"] = count_table_ok[mask].float().mean().item()
            if family == "count_color_shape":
                result[f"model_count_value_{family}_accuracy"] = (
                    outputs["model_count_value"].argmax(dim=-1)[mask] == base.target_count[mask]
                ).float().mean().item()
            if family == "relation_yes_no":
                result[f"model_truth_table_{family}_accuracy"] = (
                    outputs["model_truth_table"].argmax(dim=-1)[mask] == base.target_relation[mask]
                ).float().mean().item()
    return result


def random_batch(data: StageAOSet, *, rng: random.Random, batch_size: int, device: torch.device) -> StageAOSet:
    indices = [rng.randrange(data.prompts.shape[0]) for _ in range(batch_size)]
    if data.images.device == device:
        return data.subset(indices)
    return data.subset(indices).to(device)


@torch.no_grad()
def evaluate(model: PixelUnifiedLatentBus, data: StageAOSet, device: torch.device) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    full = ao_metrics(model(batch), batch)
    no_image = ao_metrics(model(batch, zero_image=True), batch)
    return {f"full_{key}": value for key, value in full.items()} | {f"no_image_{key}": value for key, value in no_image.items()}


def run_one(config: StageAOConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    train = build_pixel_dataset("train", config.train_size, config).to(device)
    val = build_pixel_dataset("val", config.val_size, config).to(device)
    test = build_pixel_dataset("test", config.test_size, config).to(device)
    model = PixelUnifiedLatentBus(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(config.seed + 505)
    history: list[dict[str, float]] = []
    for step in range(1, config.steps + 1):
        model.train()
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch)
            loss_pixel = pixel_loss(outputs, batch)
            loss_slot = slot_loss(outputs, batch)
            loss_query = query_loss(outputs, batch.base)
            loss_teacher = compare_loss(outputs, batch.base, model=False)
            loss_model = compare_loss(outputs, batch.base, model=True)
            loss_process_teacher = process_loss(outputs, batch.base, model=False)
            loss_process_model = process_loss(outputs, batch.base, model=True)
            loss_answer_teacher = answer_loss(outputs, batch.base, model=False)
            loss_answer_model = answer_loss(outputs, batch.base, model=True)
            loss_token_teacher = answer_token_loss(outputs, batch.base, model=False)
            loss_token_model = answer_token_loss(outputs, batch.base, model=True)
            loss_count_teacher = selected_count_loss(outputs, batch.base, model=False)
            loss_count_model = selected_count_loss(outputs, batch.base, model=True)
            loss = (
                config.pixel_loss_weight * loss_pixel
                + config.slot_loss_weight * loss_slot
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
            row = evaluate(model, val, device)
            row["step"] = float(step)
            row["loss"] = float(loss.detach().cpu())
            row["loss_pixel"] = float(loss_pixel.detach().cpu())
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
            history.append(row)
    test_metrics = evaluate(model, test, device)
    result = {
        "experiment": "omni_transformer_stage_ao_pixel_to_slots",
        "seed": config.seed,
        "device": str(device),
        "config": asdict(config),
        "training": {"history": history, "final": history[-1]},
        "test": test_metrics,
        "cost": {"seconds": round(time.perf_counter() - started, 4), "amp": use_amp},
        "interpretation": {
            "goal": "Train a pixel image expert that writes first-class cell/count/pair slots before the existing unified bus answer-token path.",
            "success_condition": "Primary gates are image-derived object/cell/count fidelity, count answer, all-task answer-token exact, and a clear no-image gap.",
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
        "full_pixel_cell_occupancy_exact": means.get("test.full_pixel_cell_occupancy_exact"),
        "full_pixel_cell_color_accuracy": means.get("test.full_pixel_cell_color_accuracy"),
        "full_pixel_cell_shape_accuracy": means.get("test.full_pixel_cell_shape_accuracy"),
        "full_pixel_count_table_exact": means.get("test.full_pixel_count_table_exact"),
        "full_cell_occupancy_exact": means.get("test.full_cell_occupancy_exact"),
        "full_cell_color_accuracy": means.get("test.full_cell_color_accuracy"),
        "full_cell_shape_accuracy": means.get("test.full_cell_shape_accuracy"),
        "full_count_table_exact": means.get("test.full_count_table_exact"),
        "full_model_answer_sequence_exact": means.get("test.full_model_answer_sequence_exact"),
        "full_model_answer_word_exact": means.get("test.full_model_answer_word_exact"),
        "full_answer_count_color_shape_sequence_exact": means.get("test.full_answer_count_color_shape_sequence_exact"),
        "full_answer_relation_yes_no_sequence_exact": means.get("test.full_answer_relation_yes_no_sequence_exact"),
        "full_model_count_value_count_color_shape_accuracy": means.get(
            "test.full_model_count_value_count_color_shape_accuracy"
        ),
        "full_model_truth_table_relation_yes_no_accuracy": means.get(
            "test.full_model_truth_table_relation_yes_no_accuracy"
        ),
        "no_image_model_answer_sequence_exact": means.get("test.no_image_model_answer_sequence_exact"),
        "no_image_pixel_count_table_exact": means.get("test.no_image_pixel_count_table_exact"),
        "no_image_count_table_exact": means.get("test.no_image_count_table_exact"),
    }
    return {
        "experiment": "omni_transformer_stage_ao_pixel_to_slots_sweep",
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_ao_pixel_to_slots/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_ao_pixel_to_slots/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701")
    parser.add_argument("--train-size", type=int, default=StageAOConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageAOConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageAOConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageAOConfig.batch_size)
    parser.add_argument("--image-size", type=int, default=StageAOConfig.image_size)
    parser.add_argument("--d-model", type=int, default=StageAOConfig.d_model)
    parser.add_argument("--layers", type=int, default=StageAOConfig.layers)
    parser.add_argument("--heads", type=int, default=StageAOConfig.heads)
    parser.add_argument("--steps", type=int, default=StageAOConfig.steps)
    parser.add_argument("--eval-every", type=int, default=StageAOConfig.eval_every)
    parser.add_argument("--answer-len", type=int, default=StageAOConfig.answer_len)
    parser.add_argument("--lr", type=float, default=StageAOConfig.lr)
    parser.add_argument("--token-loss-weight", type=float, default=StageAOConfig.token_loss_weight)
    parser.add_argument("--process-loss-weight", type=float, default=StageAOConfig.process_loss_weight)
    parser.add_argument("--process-state-weight", type=float, default=StageAOConfig.process_state_weight)
    parser.add_argument("--truth-table-state-weight", type=float, default=StageAOConfig.truth_table_state_weight)
    parser.add_argument("--slot-loss-weight", type=float, default=StageAOConfig.slot_loss_weight)
    parser.add_argument("--pixel-loss-weight", type=float, default=StageAOConfig.pixel_loss_weight)
    parser.add_argument("--query-loss-weight", type=float, default=StageAOConfig.query_loss_weight)
    parser.add_argument("--compare-loss-weight", type=float, default=StageAOConfig.compare_loss_weight)
    parser.add_argument("--model-compare-loss-weight", type=float, default=StageAOConfig.model_compare_loss_weight)
    parser.add_argument("--answer-loss-weight", type=float, default=StageAOConfig.answer_loss_weight)
    parser.add_argument("--selected-count-loss-weight", type=float, default=StageAOConfig.selected_count_loss_weight)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=StageAOConfig.amp)
    parser.add_argument("--families", default=",".join(StageAOConfig.families))
    args = parser.parse_args()

    runs: list[dict[str, object]] = []
    for seed in parse_csv_ints(args.seeds):
        config = StageAOConfig(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            seed=seed,
            image_size=args.image_size,
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
            pixel_loss_weight=args.pixel_loss_weight,
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
