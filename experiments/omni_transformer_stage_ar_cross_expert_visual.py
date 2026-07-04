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

from visual_multimodal_stage_ab import write_png  # noqa: E402


COLORS = (
    (0.90, 0.12, 0.10),
    (0.12, 0.72, 0.22),
    (0.12, 0.30, 0.92),
    (0.95, 0.76, 0.12),
)
RULES = ("add", "invert", "telemetry-heavy", "memory-heavy")
ACTIONS = ("hold", "scan", "move", "alert")
GRID_SIZE = 4
CELL_COUNT = GRID_SIZE * GRID_SIZE
TEXT_PAD = 0
TEXT_RULE_BASE = 1
TEXT_CELL_BASE = TEXT_RULE_BASE + len(RULES)
TEXT_VOCAB = TEXT_CELL_BASE + CELL_COUNT
TELEMETRY_STATES = 4
MEMORY_STATES = 4


@dataclass(frozen=True)
class StageARConfig:
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    seed: int = 20260701
    image_size: int = 64
    d_model: int = 96
    heads: int = 4
    layers: int = 2
    latent_tokens: int = 6
    lr: float = 8e-4
    steps: int = 800
    eval_every: int = 200
    direct_steps: int = 500


@dataclass(frozen=True)
class StageARExample:
    image: torch.Tensor
    text_rule: torch.Tensor
    telemetry: int
    memory: int
    target_cell: int
    rule: int
    visual_value: int
    answer: int
    prompt: str
    memory_note: str


@dataclass(frozen=True)
class StageARSet:
    examples: list[StageARExample]
    image: torch.Tensor
    text_rule: torch.Tensor
    telemetry: torch.Tensor
    memory: torch.Tensor
    target_cell: torch.Tensor
    rule: torch.Tensor
    visual_value: torch.Tensor
    answer: torch.Tensor

    def subset(self, indices: list[int]) -> "StageARSet":
        return StageARSet(
            examples=[self.examples[index] for index in indices],
            image=self.image[indices],
            text_rule=self.text_rule[indices],
            telemetry=self.telemetry[indices],
            memory=self.memory[indices],
            target_cell=self.target_cell[indices],
            rule=self.rule[indices],
            visual_value=self.visual_value[indices],
            answer=self.answer[indices],
        )

    def to(self, device: torch.device) -> "StageARSet":
        return StageARSet(
            examples=self.examples,
            image=self.image.to(device=device, dtype=torch.float32),
            text_rule=self.text_rule.to(device=device),
            telemetry=self.telemetry.to(device=device),
            memory=self.memory.to(device=device),
            target_cell=self.target_cell.to(device=device),
            rule=self.rule.to(device=device),
            visual_value=self.visual_value.to(device=device),
            answer=self.answer.to(device=device),
        )


def render_scene(cell_values: list[int], config: StageARConfig) -> torch.Tensor:
    image = torch.full((3, config.image_size, config.image_size), 0.06)
    cell = config.image_size // GRID_SIZE
    image[:, ::cell, :] = 0.24
    image[:, :, ::cell] = 0.24
    for index, value in enumerate(cell_values):
        row, col = divmod(index, GRID_SIZE)
        y0 = row * cell
        x0 = col * cell
        color = torch.tensor(COLORS[value]).view(3, 1, 1)
        inset = max(3, cell // 5)
        image[:, y0 + inset : y0 + cell - inset, x0 + inset : x0 + cell - inset] = color
        marker = value + 2
        image[:, y0 + marker : y0 + marker + 2, x0 + 2 : x0 + cell - 2] = 0.95
    return image


def answer_formula(visual: int, rule: int, telemetry: int, memory: int) -> int:
    if rule == 0:
        return (visual + telemetry + memory) % len(ACTIONS)
    if rule == 1:
        return (3 - visual + telemetry + memory) % len(ACTIONS)
    if rule == 2:
        return (visual + 2 * telemetry + memory) % len(ACTIONS)
    return (visual + telemetry + 2 * memory) % len(ACTIONS)


def build_split(split: str, size: int, config: StageARConfig) -> StageARSet:
    split_offsets = {"train": 0, "val": 10_000, "test": 20_000}
    rng = random.Random(config.seed + split_offsets[split])
    examples: list[StageARExample] = []
    for _ in range(size):
        cell_values = [rng.randrange(len(ACTIONS)) for _ in range(CELL_COUNT)]
        target_cell = rng.randrange(CELL_COUNT)
        rule = rng.randrange(len(RULES))
        telemetry = rng.randrange(TELEMETRY_STATES)
        memory = rng.randrange(MEMORY_STATES)
        visual = cell_values[target_cell]
        answer = answer_formula(visual, rule, telemetry, memory)
        row, col = divmod(target_cell, GRID_SIZE)
        prompt = f"read cell r{row} c{col}; apply {RULES[rule]} rule"
        memory_note = f"prior visual offset memory={memory}"
        examples.append(
            StageARExample(
                image=render_scene(cell_values, config),
                text_rule=torch.tensor([TEXT_RULE_BASE + rule, TEXT_CELL_BASE + target_cell], dtype=torch.long),
                telemetry=telemetry,
                memory=memory,
                target_cell=target_cell,
                rule=rule,
                visual_value=visual,
                answer=answer,
                prompt=prompt,
                memory_note=memory_note,
            )
        )
    return StageARSet(
        examples=examples,
        image=torch.stack([example.image for example in examples]),
        text_rule=torch.stack([example.text_rule for example in examples]),
        telemetry=torch.tensor([example.telemetry for example in examples], dtype=torch.long),
        memory=torch.tensor([example.memory for example in examples], dtype=torch.long),
        target_cell=torch.tensor([example.target_cell for example in examples], dtype=torch.long),
        rule=torch.tensor([example.rule for example in examples], dtype=torch.long),
        visual_value=torch.tensor([example.visual_value for example in examples], dtype=torch.long),
        answer=torch.tensor([example.answer for example in examples], dtype=torch.long),
    )


def random_batch(data: StageARSet, *, rng: random.Random, batch_size: int, device: torch.device) -> StageARSet:
    indices = [rng.randrange(len(data.examples)) for _ in range(batch_size)]
    return data.subset(indices).to(device)


def mask_batch(batch: StageARSet, zero_modalities: Iterable[str]) -> StageARSet:
    zero_set = set(zero_modalities)
    image = torch.zeros_like(batch.image) if "image" in zero_set else batch.image
    text_rule = torch.full_like(batch.text_rule, TEXT_PAD) if "text_rule" in zero_set else batch.text_rule
    telemetry = torch.zeros_like(batch.telemetry) if "telemetry" in zero_set else batch.telemetry
    memory = torch.zeros_like(batch.memory) if "memory" in zero_set else batch.memory
    return StageARSet(
        examples=batch.examples,
        image=image,
        text_rule=text_rule,
        telemetry=telemetry,
        memory=memory,
        target_cell=batch.target_cell,
        rule=batch.rule,
        visual_value=batch.visual_value,
        answer=batch.answer,
    )


class CellVisionExpert(nn.Module):
    def __init__(self, config: StageARConfig) -> None:
        super().__init__()
        self.config = config
        self.cell = config.image_size // GRID_SIZE
        self.proj = nn.Sequential(
            nn.Linear(3 + CELL_COUNT, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = image.unfold(2, self.cell, self.cell).unfold(3, self.cell, self.cell)
        means = patches.mean(dim=(-1, -2)).permute(0, 2, 3, 1).reshape(image.shape[0], CELL_COUNT, 3)
        pos = torch.eye(CELL_COUNT, device=image.device).unsqueeze(0).expand(image.shape[0], -1, -1)
        return self.proj(torch.cat((means, pos), dim=-1))


class StageARLatentModel(nn.Module):
    def __init__(self, config: StageARConfig) -> None:
        super().__init__()
        self.vision = CellVisionExpert(config)
        self.text = nn.Embedding(TEXT_VOCAB, config.d_model, padding_idx=TEXT_PAD)
        self.telemetry = nn.Embedding(TELEMETRY_STATES, config.d_model)
        self.memory = nn.Embedding(MEMORY_STATES, config.d_model)
        self.type_embed = nn.Embedding(4, config.d_model)
        self.latent = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
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
        self.answer = nn.Linear(config.d_model, len(ACTIONS))
        self.visual_probe = nn.Linear(config.d_model, len(ACTIONS))

    def forward(self, batch: StageARSet, *, zero_modalities: Iterable[str] = ()) -> dict[str, torch.Tensor]:
        batch = mask_batch(batch, zero_modalities)
        vision = self.vision(batch.image) + self.type_embed.weight[0].view(1, 1, -1)
        cell_index = (batch.text_rule[:, 1] - TEXT_CELL_BASE).clamp(0, CELL_COUNT - 1)
        selected = vision[torch.arange(vision.shape[0], device=vision.device), cell_index]
        text = self.text(batch.text_rule[:, :1]) + self.type_embed.weight[1].view(1, 1, -1)
        telemetry = self.telemetry(batch.telemetry).unsqueeze(1) + self.type_embed.weight[2].view(1, 1, -1)
        memory = self.memory(batch.memory).unsqueeze(1) + self.type_embed.weight[3].view(1, 1, -1)
        latent = self.latent.unsqueeze(0).expand(batch.image.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((selected.unsqueeze(1), text, telemetry, memory, latent), dim=1))
        latent_out = self.norm(encoded[:, -latent.shape[1] :])
        pooled = latent_out.mean(dim=1)
        return {"answer": self.answer(pooled), "visual": self.visual_probe(selected)}


class DirectAllInputBaseline(nn.Module):
    def __init__(self, config: StageARConfig) -> None:
        super().__init__()
        self.config = config
        self.vision = CellVisionExpert(config)
        self.text = nn.Embedding(TEXT_VOCAB, config.d_model, padding_idx=TEXT_PAD)
        self.telemetry = nn.Embedding(TELEMETRY_STATES, config.d_model)
        self.memory = nn.Embedding(MEMORY_STATES, config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model * (CELL_COUNT + 3), config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, len(ACTIONS)),
        )

    def forward(self, batch: StageARSet, *, zero_modalities: Iterable[str] = ()) -> dict[str, torch.Tensor]:
        batch = mask_batch(batch, zero_modalities)
        vision = self.vision(batch.image)
        cell_index = (batch.text_rule[:, 1] - TEXT_CELL_BASE).clamp(0, CELL_COUNT - 1)
        selected = vision[torch.arange(vision.shape[0], device=vision.device), cell_index]
        text = self.text(batch.text_rule[:, :1]).mean(dim=1)
        telemetry = self.telemetry(batch.telemetry)
        memory = self.memory(batch.memory)
        empty = torch.zeros(batch.image.shape[0], (CELL_COUNT - 1) * self.config.d_model, device=batch.image.device)
        return {"answer": self.mlp(torch.cat((selected, empty, text, telemetry, memory), dim=-1))}


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().item())


def train_model(
    model: nn.Module,
    train: StageARSet,
    val: StageARSet,
    config: StageARConfig,
    *,
    steps: int,
    device: torch.device,
    include_visual_loss: bool,
) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    rng = random.Random(config.seed + (37 if include_visual_loss else 73))
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        model.train()
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        outputs = model(batch)
        loss = F.cross_entropy(outputs["answer"], batch.answer)
        if include_visual_loss and "visual" in outputs:
            loss = loss + 0.25 * F.cross_entropy(outputs["visual"], batch.visual_value)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_model(model, val, config, device=device, prefix="val")
            metrics["step"] = float(step)
            metrics["loss"] = float(loss.detach().item())
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
    return {"history": history}


@torch.no_grad()
def evaluate_model(model: nn.Module, data: StageARSet, config: StageARConfig, *, device: torch.device, prefix: str) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    metrics: dict[str, float] = {}
    for name, zero in (
        ("full", ()),
        ("no_image", ("image",)),
        ("no_text_rule", ("text_rule",)),
        ("no_telemetry", ("telemetry",)),
        ("no_memory", ("memory",)),
    ):
        outputs = model(batch, zero_modalities=zero)
        metrics[f"{prefix}_{name}_answer_accuracy"] = accuracy(outputs["answer"], batch.answer)
    if isinstance(model, StageARLatentModel):
        outputs = model(batch)
        metrics[f"{prefix}_full_visual_probe_accuracy"] = accuracy(outputs["visual"], batch.visual_value)
    return metrics


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def write_samples(data: StageARSet, path: Path, *, limit: int = 6) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, example in enumerate(data.examples[:limit]):
        write_png(path / f"sample_{index:02d}.png", example.image)
        rows.append(
            {
                "image": f"sample_{index:02d}.png",
                "prompt": example.prompt,
                "telemetry": example.telemetry,
                "memory_note": example.memory_note,
                "target_cell": example.target_cell,
                "visual_value": example.visual_value,
                "rule": RULES[example.rule],
                "answer": ACTIONS[example.answer],
            }
        )
    (path / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(config: StageARConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = build_split("train", config.train_size, config)
    val = build_split("val", config.val_size, config)
    test = build_split("test", config.test_size, config)
    write_samples(test, output_path.parent / "samples")

    latent = StageARLatentModel(config)
    direct = DirectAllInputBaseline(config)
    latent_cost = train_model(latent, train, val, config, steps=config.steps, device=device, include_visual_loss=True)
    direct_cost = train_model(direct, train, val, config, steps=config.direct_steps, device=device, include_visual_loss=False)
    latent_test = evaluate_model(latent, test, config, device=device, prefix="latent_test")
    direct_test = evaluate_model(direct, test, config, device=device, prefix="direct_test")
    result = {
        "schema_version": 1,
        "stage": "AR",
        "config": asdict(config),
        "device": str(device),
        "metrics": {**latent_test, **direct_test},
        "cost": {
            "elapsed_sec": time.perf_counter() - started,
            "latent_parameters": parameter_count(latent),
            "direct_parameters": parameter_count(direct),
        },
        "training": {"latent": latent_cost, "direct": direct_cost},
        "interpretation": {
            "goal": "Verify a vision-centered cross-expert task where image, text rule, telemetry, and memory are all causally required.",
            "success_condition": "Latent full accuracy >= 90%, every no-modality ablation drops at least 30 percentage points, and direct all-input baseline cost is recorded.",
            "backup_route": "Agent-state and long-horizon variants are intentionally left for Stage AT or an AR hardening follow-up.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": result["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def collect_numbers(value: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            out.update(collect_numbers(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    rows = [collect_numbers(run) for run in runs]
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {key: statistics.mean(row[key] for row in rows if key in row) for key in keys}
    stdev = {
        key: statistics.pstdev([row[key] for row in rows if key in row])
        for key in keys
        if sum(1 for row in rows if key in row) > 1
    }
    full = summary.get("metrics_latent_test_full_answer_accuracy", 0.0)
    drops = {
        name: full - summary.get(f"metrics_latent_test_{name}_answer_accuracy", full)
        for name in ("no_image", "no_text_rule", "no_telemetry", "no_memory")
    }
    return {
        "schema_version": 1,
        "stage": "AR",
        "runs": runs,
        "summary": summary,
        "stdev": stdev,
        "gates": {
            "latent_full_accuracy": full,
            "ablation_drops": drops,
            "passes_full_90": full >= 0.90,
            "passes_each_ablation_drop_30pp": all(drop >= 0.30 for drop in drops.values()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_ar_cross_expert_visual/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_ar_cross_expert_visual/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_ar_cross_expert_visual/sweep_results.json"))
    parser.add_argument("--train-size", type=int, default=StageARConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageARConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageARConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageARConfig.batch_size)
    parser.add_argument("--image-size", type=int, default=StageARConfig.image_size)
    parser.add_argument("--d-model", type=int, default=StageARConfig.d_model)
    parser.add_argument("--heads", type=int, default=StageARConfig.heads)
    parser.add_argument("--layers", type=int, default=StageARConfig.layers)
    parser.add_argument("--latent-tokens", type=int, default=StageARConfig.latent_tokens)
    parser.add_argument("--steps", type=int, default=StageARConfig.steps)
    parser.add_argument("--direct-steps", type=int, default=StageARConfig.direct_steps)
    parser.add_argument("--eval-every", type=int, default=StageARConfig.eval_every)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, seed: int) -> StageARConfig:
    return StageARConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=seed,
        image_size=args.image_size,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        latent_tokens=args.latent_tokens,
        steps=args.steps,
        direct_steps=args.direct_steps,
        eval_every=args.eval_every,
    )


def main() -> None:
    args = parse_args()
    if args.sweep:
        runs = []
        for seed_text in args.seeds.split(","):
            seed = int(seed_text.strip())
            output = args.output_dir / f"seed{seed}.json"
            runs.append(run_one(config_from_args(args, seed), output))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"aggregate": aggregate["gates"]}, ensure_ascii=False, indent=2), flush=True)
    else:
        first_seed = int(args.seeds.split(",")[0].strip())
        run_one(config_from_args(args, first_seed), args.output)


if __name__ == "__main__":
    main()
