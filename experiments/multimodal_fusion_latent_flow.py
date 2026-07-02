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

from visual_multimodal_stage_ab import write_png


VISUAL_STATES = 4
GOALS = 4
TELEMETRY_STATES = 4
ACTION_COUNT = 6
IMAGE_SIZE = 48
TELEMETRY_STEPS = 12
TELEMETRY_CHANNELS = 4
TEXT_LENGTH = 5
TEXT_VOCAB = 16

MODALITIES = ("image", "text", "telemetry")
FACTOR_NAMES = ("visual", "goal", "telemetry")

VISUAL_NAMES = (
    "green_lock",
    "open_interlock",
    "thermal_marker",
    "blocked_flow",
)
GOAL_NAMES = (
    "safety_first",
    "throughput_first",
    "power_saving",
    "maintenance_check",
)
TELEMETRY_NAMES = (
    "stable",
    "heat_ramp",
    "low_voltage",
    "error_spike",
)
ACTION_NAMES = (
    "hold_state",
    "isolate_system",
    "cool_down",
    "reroute_power",
    "restart_controller",
    "increase_throughput",
)


def build_policy_table() -> tuple[tuple[tuple[int, ...], ...], ...]:
    actions = [idx % ACTION_COUNT for idx in range(VISUAL_STATES * GOALS * TELEMETRY_STATES)]
    rng = random.Random(20260702)
    rng.shuffle(actions)
    table: list[list[list[int]]] = []
    cursor = 0
    for _visual in range(VISUAL_STATES):
        by_goal: list[list[int]] = []
        for _goal in range(GOALS):
            row = actions[cursor : cursor + TELEMETRY_STATES]
            by_goal.append(row)
            cursor += TELEMETRY_STATES
        table.append(by_goal)
    return tuple(tuple(tuple(row) for row in by_goal) for by_goal in table)


POLICY_TABLE = build_policy_table()


@dataclass(frozen=True)
class FusionConfig:
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 2048
    batch_size: int = 256
    seed: int = 20260701
    d_latent: int = 96
    lr: float = 8e-4
    shared_steps: int = 1300
    baseline_steps: int = 900
    probe_steps: int = 400
    eval_every: int = 450
    factor_loss_weight: float = 0.35


@dataclass(frozen=True)
class DiagnosticExample:
    visual: int
    goal: int
    telemetry: int
    action: int


@dataclass
class Batch:
    image: torch.Tensor
    text: torch.Tensor
    telemetry_values: torch.Tensor
    visual: torch.Tensor
    goal: torch.Tensor
    telemetry: torch.Tensor
    action: torch.Tensor


def stat(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def action_for(visual: int, goal: int, telemetry: int) -> int:
    return POLICY_TABLE[visual][goal][telemetry]


def generate_examples(size: int, *, seed: int) -> list[DiagnosticExample]:
    combos = [(visual, goal, telemetry) for visual in range(VISUAL_STATES) for goal in range(GOALS) for telemetry in range(TELEMETRY_STATES)]
    rng = random.Random(seed)
    examples: list[DiagnosticExample] = []
    while len(examples) < size:
        rng.shuffle(combos)
        for visual, goal, telemetry in combos:
            examples.append(DiagnosticExample(visual, goal, telemetry, action_for(visual, goal, telemetry)))
            if len(examples) == size:
                break
    rng.shuffle(examples)
    return examples


def render_panel_images(visual_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    batch = int(visual_ids.shape[0])
    image = torch.full((batch, 3, IMAGE_SIZE, IMAGE_SIZE), 0.045, device=device)
    image[:, :, 3:-3, 3:-3] = 0.09
    image[:, :, 5:43, 5:43] = 0.12

    palette = torch.tensor(
        [
            [0.10, 0.86, 0.24],
            [0.92, 0.10, 0.10],
            [0.95, 0.72, 0.10],
            [0.16, 0.42, 0.95],
        ],
        device=device,
    )
    colors = palette[visual_ids]
    image[:, :, 8:24, 8:24] = colors[:, :, None, None]

    mask = visual_ids == 0
    if bool(mask.any()):
        image[mask, :, 29:33, 10:38] = colors[mask, :, None, None]
        image[mask, :, 27:39, 20:28] = colors[mask, :, None, None] * 0.65
    mask = visual_ids == 1
    if bool(mask.any()):
        image[mask, :, 30:36, 10:38] = colors[mask, :, None, None]
        image[mask, :, 25:42, 10:16] = colors[mask, :, None, None] * 0.85
        image[mask, :, 25:31, 32:38] = colors[mask, :, None, None] * 0.85
    mask = visual_ids == 2
    if bool(mask.any()):
        image[mask, :, 31:35, 8:40] = colors[mask, :, None, None]
        image[mask, :, 27:39, 22:26] = colors[mask, :, None, None]
        image[mask, :, 27:39, 30:34] = colors[mask, :, None, None] * 0.70
    mask = visual_ids == 3
    if bool(mask.any()):
        image[mask, :, 28:40, 10:18] = colors[mask, :, None, None]
        image[mask, :, 28:40, 30:38] = colors[mask, :, None, None]
        image[mask, :, 32:36, 18:30] = 0.04

    return image.clamp(0.0, 1.0)


def render_text_tokens(goal_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    batch = int(goal_ids.shape[0])
    tokens = torch.zeros((batch, TEXT_LENGTH), dtype=torch.long, device=device)
    tokens[:, 0] = 1
    tokens[:, 1] = 3 + goal_ids
    tokens[:, 2] = 7 + goal_ids
    tokens[:, 3] = 11
    tokens[:, 4] = 2
    return tokens


def render_telemetry(telemetry_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, TELEMETRY_STEPS, device=device)
    templates = torch.stack(
        (
            torch.stack(
                (
                    torch.full_like(t, 0.32),
                    torch.full_like(t, 0.72),
                    torch.full_like(t, 0.18),
                    torch.full_like(t, 0.04),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    0.42 + 0.48 * t,
                    torch.full_like(t, 0.68),
                    0.20 + 0.05 * t,
                    torch.full_like(t, 0.06),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    torch.full_like(t, 0.36),
                    0.56 - 0.34 * t,
                    torch.full_like(t, 0.22),
                    torch.full_like(t, 0.07),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    torch.full_like(t, 0.38),
                    torch.full_like(t, 0.62),
                    0.18 + 0.18 * (t > 0.55).float(),
                    0.05 + 0.75 * (t > 0.65).float(),
                ),
                dim=-1,
            ),
        ),
        dim=0,
    )
    return templates[telemetry_ids]


def make_batch(examples: list[DiagnosticExample], *, device: torch.device) -> Batch:
    visual = torch.tensor([item.visual for item in examples], dtype=torch.long, device=device)
    goal = torch.tensor([item.goal for item in examples], dtype=torch.long, device=device)
    telemetry = torch.tensor([item.telemetry for item in examples], dtype=torch.long, device=device)
    action = torch.tensor([item.action for item in examples], dtype=torch.long, device=device)
    return Batch(
        image=render_panel_images(visual, device=device),
        text=render_text_tokens(goal, device=device),
        telemetry_values=render_telemetry(telemetry, device=device),
        visual=visual,
        goal=goal,
        telemetry=telemetry,
        action=action,
    )


def random_batch(examples: list[DiagnosticExample], *, rng: random.Random, batch_size: int, device: torch.device) -> Batch:
    return make_batch(rng.choices(examples, k=batch_size), device=device)


class ImageEncoder(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class TextEncoder(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(TEXT_VOCAB, d_latent)
        self.net = nn.Sequential(nn.Linear(d_latent, d_latent), nn.GELU(), nn.LayerNorm(d_latent))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).float().unsqueeze(-1)
        embedded = self.embedding(tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.net(pooled)


class TelemetryEncoder(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(TELEMETRY_STEPS * TELEMETRY_CHANNELS, 128),
            nn.GELU(),
            nn.Linear(128, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class SharedLatentFusionModel(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(d_latent)
        self.text_encoder = TextEncoder(d_latent)
        self.telemetry_encoder = TelemetryEncoder(d_latent)
        self.type_embedding = nn.Parameter(torch.randn(3, d_latent) * 0.02)
        self.slot_attn = nn.MultiheadAttention(d_latent, num_heads=4, batch_first=True)
        self.slot_norm = nn.LayerNorm(d_latent)
        self.slot_ff = nn.Sequential(nn.Linear(d_latent, d_latent * 2), nn.GELU(), nn.Linear(d_latent * 2, d_latent))
        self.ff_norm = nn.LayerNorm(d_latent)
        self.bus_query = nn.Parameter(torch.randn(1, 1, d_latent) * 0.02)
        self.bus_attn = nn.MultiheadAttention(d_latent, num_heads=4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d_latent, d_latent), nn.GELU(), nn.LayerNorm(d_latent))
        self.action_head = nn.Linear(d_latent, ACTION_COUNT)
        self.visual_head = nn.Linear(d_latent, VISUAL_STATES)
        self.goal_head = nn.Linear(d_latent, GOALS)
        self.telemetry_head = nn.Linear(d_latent, TELEMETRY_STATES)

    def encode_slots(self, batch: Batch) -> dict[str, torch.Tensor]:
        return {
            "image": self.image_encoder(batch.image),
            "text": self.text_encoder(batch.text),
            "telemetry": self.telemetry_encoder(batch.telemetry_values),
        }

    def forward(
        self,
        batch: Batch,
        *,
        zero_modalities: Iterable[str] = (),
        shuffle_modalities: Iterable[str] = (),
    ) -> dict[str, torch.Tensor]:
        slots = self.encode_slots(batch)
        zero_set = set(zero_modalities)
        shuffle_set = set(shuffle_modalities)
        for modality in MODALITIES:
            if modality in zero_set:
                slots[modality] = torch.zeros_like(slots[modality])
            if modality in shuffle_set and slots[modality].shape[0] > 1:
                slots[modality] = torch.roll(slots[modality], shifts=1, dims=0)

        token_stack = torch.stack([slots["image"], slots["text"], slots["telemetry"]], dim=1) + self.type_embedding[None, :, :]
        attended, _ = self.slot_attn(token_stack, token_stack, token_stack, need_weights=False)
        token_stack = self.slot_norm(token_stack + attended)
        token_stack = self.ff_norm(token_stack + self.slot_ff(token_stack))
        bus_query = self.bus_query.expand(token_stack.shape[0], -1, -1)
        bus, _ = self.bus_attn(bus_query, token_stack, token_stack, need_weights=False)
        bus = self.head(bus.squeeze(1))
        return {
            "bus": bus,
            "action": self.action_head(bus),
            "visual": self.visual_head(bus),
            "goal": self.goal_head(bus),
            "telemetry": self.telemetry_head(bus),
        }


class EarlyConcatFusionModel(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(d_latent)
        self.text_encoder = TextEncoder(d_latent)
        self.telemetry_encoder = TelemetryEncoder(d_latent)
        self.fusion = nn.Sequential(
            nn.Linear(d_latent * 3, d_latent * 2),
            nn.GELU(),
            nn.LayerNorm(d_latent * 2),
            nn.Linear(d_latent * 2, d_latent),
            nn.GELU(),
        )
        self.action_head = nn.Linear(d_latent, ACTION_COUNT)
        self.visual_head = nn.Linear(d_latent, VISUAL_STATES)
        self.goal_head = nn.Linear(d_latent, GOALS)
        self.telemetry_head = nn.Linear(d_latent, TELEMETRY_STATES)

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        fused = self.fusion(
            torch.cat(
                (
                    self.image_encoder(batch.image),
                    self.text_encoder(batch.text),
                    self.telemetry_encoder(batch.telemetry_values),
                ),
                dim=-1,
            )
        )
        return {
            "action": self.action_head(fused),
            "visual": self.visual_head(fused),
            "goal": self.goal_head(fused),
            "telemetry": self.telemetry_head(fused),
        }


class LateFusionModel(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(d_latent)
        self.text_encoder = TextEncoder(d_latent)
        self.telemetry_encoder = TelemetryEncoder(d_latent)
        self.image_head = nn.Linear(d_latent, ACTION_COUNT)
        self.text_head = nn.Linear(d_latent, ACTION_COUNT)
        self.telemetry_head = nn.Linear(d_latent, ACTION_COUNT)
        self.logit_weights = nn.Parameter(torch.zeros(3))

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        logits = torch.stack(
            (
                self.image_head(self.image_encoder(batch.image)),
                self.text_head(self.text_encoder(batch.text)),
                self.telemetry_head(self.telemetry_encoder(batch.telemetry_values)),
            ),
            dim=1,
        )
        weights = F.softmax(self.logit_weights, dim=0)
        return {"action": (logits * weights[None, :, None]).sum(dim=1)}


class SingleModalityModel(nn.Module):
    def __init__(self, d_latent: int, modality: str) -> None:
        super().__init__()
        self.modality = modality
        if modality == "image":
            self.encoder: nn.Module = ImageEncoder(d_latent)
        elif modality == "text":
            self.encoder = TextEncoder(d_latent)
        elif modality == "telemetry":
            self.encoder = TelemetryEncoder(d_latent)
        else:
            raise ValueError(f"unknown modality: {modality}")
        self.action_head = nn.Linear(d_latent, ACTION_COUNT)

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        if self.modality == "image":
            latent = self.encoder(batch.image)
        elif self.modality == "text":
            latent = self.encoder(batch.text)
        else:
            latent = self.encoder(batch.telemetry_values)
        return {"action": self.action_head(latent)}


class FrozenSlotProbes(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()
        for source in MODALITIES:
            for target, count in (("visual", VISUAL_STATES), ("goal", GOALS), ("telemetry", TELEMETRY_STATES)):
                self.heads[f"{source}_to_{target}"] = nn.Linear(d_latent, count)

    def forward(self, slots: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: head(slots[name.split("_to_")[0]].detach()) for name, head in self.heads.items()}


def supervised_loss(output: dict[str, torch.Tensor], batch: Batch, *, factor_loss_weight: float) -> torch.Tensor:
    loss = F.cross_entropy(output["action"], batch.action)
    if {"visual", "goal", "telemetry"}.issubset(output.keys()):
        loss = loss + factor_loss_weight * (
            F.cross_entropy(output["visual"], batch.visual)
            + F.cross_entropy(output["goal"], batch.goal)
            + F.cross_entropy(output["telemetry"], batch.telemetry)
        )
    return loss


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    examples: list[DiagnosticExample],
    config: FusionConfig,
    *,
    device: torch.device,
    zero_modalities: Iterable[str] = (),
    shuffle_modalities: Iterable[str] = (),
) -> dict[str, float]:
    model.eval()
    totals = {
        "action_exact": 0.0,
        "visual_exact": 0.0,
        "goal_exact": 0.0,
        "telemetry_exact": 0.0,
        "text_exact": 0.0,
    }
    count = 0
    for start in range(0, len(examples), config.batch_size):
        batch = make_batch(examples[start : start + config.batch_size], device=device)
        if isinstance(model, SharedLatentFusionModel):
            output = model(batch, zero_modalities=zero_modalities, shuffle_modalities=shuffle_modalities)
        else:
            output = model(batch)
        action_pred = output["action"].argmax(dim=-1)
        action_ok = action_pred == batch.action
        totals["action_exact"] += float(action_ok.sum().detach().cpu())
        if {"visual", "goal", "telemetry"}.issubset(output.keys()):
            visual_pred = output["visual"].argmax(dim=-1)
            goal_pred = output["goal"].argmax(dim=-1)
            telemetry_pred = output["telemetry"].argmax(dim=-1)
            visual_ok = visual_pred == batch.visual
            goal_ok = goal_pred == batch.goal
            telemetry_ok = telemetry_pred == batch.telemetry
            totals["visual_exact"] += float(visual_ok.sum().detach().cpu())
            totals["goal_exact"] += float(goal_ok.sum().detach().cpu())
            totals["telemetry_exact"] += float(telemetry_ok.sum().detach().cpu())
            totals["text_exact"] += float((action_ok & visual_ok & goal_ok & telemetry_ok).sum().detach().cpu())
        count += len(batch.action)
    metrics = {"action_exact": totals["action_exact"] / count}
    if totals["visual_exact"] > 0 or totals["goal_exact"] > 0 or totals["telemetry_exact"] > 0:
        metrics.update(
            {
                "visual_factor_exact": totals["visual_exact"] / count,
                "goal_factor_exact": totals["goal_exact"] / count,
                "telemetry_factor_exact": totals["telemetry_exact"] / count,
                "text_exact": totals["text_exact"] / count,
            }
        )
    model.train()
    return metrics


def train_supervised_model(
    model: nn.Module,
    train: list[DiagnosticExample],
    val: list[DiagnosticExample],
    config: FusionConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    model.to(device)
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        output = model(batch)
        loss = supervised_loss(output, batch, factor_loss_weight=config.factor_loss_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_model(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} action={metrics['action_exact']:.3f}", flush=True)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def train_slot_probes(
    model: SharedLatentFusionModel,
    train: list[DiagnosticExample],
    test: list[DiagnosticExample],
    config: FusionConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probes = FrozenSlotProbes(config.d_latent).to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=config.lr, weight_decay=0.0)
    rng = random.Random(config.seed + 70000)
    started = time.perf_counter()
    for step in range(1, config.probe_steps + 1):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            slots = model.encode_slots(batch)
        output = probes(slots)
        loss = torch.tensor(0.0, device=device)
        for name, logits in output.items():
            target_name = name.split("_to_")[1]
            target = getattr(batch, target_name)
            loss = loss + F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    totals = {name: 0.0 for name in probes.heads.keys()}
    count = 0
    with torch.no_grad():
        for start in range(0, len(test), config.batch_size):
            batch = make_batch(test[start : start + config.batch_size], device=device)
            slots = model.encode_slots(batch)
            output = probes(slots)
            for name, logits in output.items():
                target_name = name.split("_to_")[1]
                target = getattr(batch, target_name)
                totals[name] += float((logits.argmax(dim=-1) == target).sum().detach().cpu())
            count += len(batch.action)
    return {
        "training_seconds": round(time.perf_counter() - started, 3),
        "accuracies": {name: value / count for name, value in totals.items()},
    }


def evaluate_oracle(examples: list[DiagnosticExample]) -> dict[str, float]:
    correct = sum(1 for item in examples if action_for(item.visual, item.goal, item.telemetry) == item.action)
    return {"action_exact": correct / len(examples), "text_exact": 1.0}


def render_answer(action: int, visual: int, goal: int, telemetry: int) -> str:
    return (
        f"action={ACTION_NAMES[action]}; "
        f"visual={VISUAL_NAMES[visual]}; "
        f"goal={GOAL_NAMES[goal]}; "
        f"telemetry={TELEMETRY_NAMES[telemetry]}"
    )


@torch.no_grad()
def sample_outputs(model: SharedLatentFusionModel, examples: list[DiagnosticExample], config: FusionConfig, *, device: torch.device) -> list[dict[str, str]]:
    batch = make_batch(examples[:8], device=device)
    output = model(batch)
    action_pred = output["action"].argmax(dim=-1).detach().cpu().tolist()
    visual_pred = output["visual"].argmax(dim=-1).detach().cpu().tolist()
    goal_pred = output["goal"].argmax(dim=-1).detach().cpu().tolist()
    telemetry_pred = output["telemetry"].argmax(dim=-1).detach().cpu().tolist()
    samples = []
    for idx, item in enumerate(examples[:8]):
        samples.append(
            {
                "target": render_answer(item.action, item.visual, item.goal, item.telemetry),
                "prediction": render_answer(action_pred[idx], visual_pred[idx], goal_pred[idx], telemetry_pred[idx]),
            }
        )
    return samples


def write_sample_panel_grid(examples: list[DiagnosticExample], *, output_dir: Path, device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = make_batch(examples[:8], device=device)
    first_row = torch.cat([batch.image[idx] for idx in range(4)], dim=2)
    second_row = torch.cat([batch.image[idx] for idx in range(4, 8)], dim=2)
    grid = torch.cat([first_row, second_row], dim=1)
    write_png(output_dir / "diagnostic_panel_grid.png", grid)


def run_experiment(config: FusionConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_ef device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train = generate_examples(config.train_size, seed=config.seed + 1)
    val = generate_examples(config.val_size, seed=config.seed + 2)
    test = generate_examples(config.test_size, seed=config.seed + 3)
    write_sample_panel_grid(test, output_dir=output_path.parent / "samples" / f"seed{config.seed}", device=device)

    metrics: dict[str, dict[str, float]] = {"oracle_structured_policy": evaluate_oracle(test)}
    training: dict[str, object] = {}

    baselines: list[tuple[str, nn.Module, int, int]] = [
        ("image_only", SingleModalityModel(config.d_latent, "image"), config.baseline_steps, config.seed + 10),
        ("text_only", SingleModalityModel(config.d_latent, "text"), config.baseline_steps, config.seed + 11),
        ("telemetry_only", SingleModalityModel(config.d_latent, "telemetry"), config.baseline_steps, config.seed + 12),
        ("late_fusion_no_cross_modal_interaction", LateFusionModel(config.d_latent), config.baseline_steps, config.seed + 13),
        ("early_concat_fusion", EarlyConcatFusionModel(config.d_latent), config.baseline_steps, config.seed + 14),
    ]
    for name, model, steps, seed in baselines:
        training[name] = train_supervised_model(model, train, val, config, steps=steps, seed=seed, device=device, label=name)
        metrics[name] = evaluate_model(model, test, config, device=device)

    shared_model = SharedLatentFusionModel(config.d_latent)
    training["shared_latent_fusion"] = train_supervised_model(
        shared_model,
        train,
        val,
        config,
        steps=config.shared_steps,
        seed=config.seed + 20,
        device=device,
        label="shared_latent_fusion",
    )
    metrics["shared_latent_fusion"] = evaluate_model(shared_model, test, config, device=device)

    ablations = {
        "full": evaluate_model(shared_model, test, config, device=device),
        "zero_image_latent": evaluate_model(shared_model, test, config, device=device, zero_modalities=("image",)),
        "zero_text_latent": evaluate_model(shared_model, test, config, device=device, zero_modalities=("text",)),
        "zero_telemetry_latent": evaluate_model(shared_model, test, config, device=device, zero_modalities=("telemetry",)),
        "shuffle_image_latent": evaluate_model(shared_model, test, config, device=device, shuffle_modalities=("image",)),
        "shuffle_text_latent": evaluate_model(shared_model, test, config, device=device, shuffle_modalities=("text",)),
        "shuffle_telemetry_latent": evaluate_model(shared_model, test, config, device=device, shuffle_modalities=("telemetry",)),
    }
    probes = train_slot_probes(shared_model, train, test, config, device=device)
    samples = sample_outputs(shared_model, test, config, device=device)

    output = {
        "experiment": "multimodal_fusion_latent_flow_stage_ef",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "policy_table": POLICY_TABLE,
        "metrics": metrics,
        "latent_flow": {
            "ablation": ablations,
            "slot_probe": probes,
        },
        "training": training,
        "sample_outputs": samples,
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
        for key, value in collect_numbers({"metrics": run["metrics"], "latent_flow": run["latent_flow"]}).items():
            if "policy_table" not in key and "training_seconds" not in key:
                buckets.setdefault(key, []).append(value)
    return {
        "experiment": "multimodal_fusion_latent_flow_stage_ef_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/multimodal_fusion_latent_flow/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multimodal_fusion_latent_flow/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/multimodal_fusion_latent_flow/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-latent", type=int, default=96)
    parser.add_argument("--shared-steps", type=int, default=1300)
    parser.add_argument("--baseline-steps", type=int, default=900)
    parser.add_argument("--probe-steps", type=int, default=400)
    args = parser.parse_args()

    base_config = FusionConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_latent=args.d_latent,
        shared_steps=args.shared_steps,
        baseline_steps=args.baseline_steps,
        probe_steps=args.probe_steps,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = FusionConfig(**{**asdict(base_config), "seed": seed})
            output_path = args.output_dir / f"seed{seed}" / "result.json"
            runs.append(run_experiment(config, output_path))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
