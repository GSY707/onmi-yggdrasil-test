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


TASKS = ("cross_expert_visual", "file_audit_tools", "memory_exploration")
TASK_AR = 0
TASK_AS = 1
TASK_AT = 2
COLORS = 4
ZONES = 4
VALUES = 4
TEXT_LEN = 6
TOOL_LEN = 4
MEMORY_LEN = 4
STATE_LEN = 4
ANSWER_CLASSES = 12

PAD = 0
TASK_BASE = 1
TEXT_TARGET_BASE = TASK_BASE + len(TASKS)
TEXT_RULE_BASE = TEXT_TARGET_BASE + ZONES
TEXT_TARGET_COLOR_BASE = TEXT_RULE_BASE + VALUES
TEXT_FILLER = TEXT_TARGET_COLOR_BASE + COLORS
TOOL_HTML_BASE = TEXT_FILLER + 1
TOOL_CSV_BASE = TOOL_HTML_BASE + VALUES
TOOL_PDF_BASE = TOOL_CSV_BASE + VALUES
TOOL_SCREEN_BASE = TOOL_PDF_BASE + VALUES
MEM_UNKNOWN = TOOL_SCREEN_BASE + VALUES
MEM_ZONE_BASE = MEM_UNKNOWN + 1
STATE_TELEMETRY_BASE = MEM_ZONE_BASE + ZONES * (COLORS + 1)
STATE_PHASE_BASE = STATE_TELEMETRY_BASE + VALUES
STATE_SEEN_BASE = STATE_PHASE_BASE + VALUES
STATE_LAST_BASE = STATE_SEEN_BASE + ZONES
TOKEN_VOCAB = STATE_LAST_BASE + ZONES * COLORS + 1


@dataclass(frozen=True)
class StageAVConfig:
    train_size: int = 4096
    val_size: int = 768
    test_size: int = 768
    batch_size: int = 48
    seed: int = 20260701
    d_model: int = 512
    heads: int = 8
    layers: int = 8
    latent_tokens: int = 8
    lr: float = 5e-4
    steps: int = 700
    eval_every: int = 175
    sample_count: int = 12
    amp: bool = True
    checkpoint_activations: bool = False


@dataclass(frozen=True)
class AVExample:
    task: int
    image_colors: tuple[int, ...]
    text_target_zone: int
    text_rule: int
    text_target_color: int
    tool_values: tuple[int, int, int, int]
    memory_values: tuple[int, ...]
    telemetry: int
    phase: int
    seen_zone: int
    last_zone: int
    last_color: int
    answer: int
    case_id: str


@dataclass(frozen=True)
class AVSet:
    examples: list[AVExample]
    image: torch.Tensor
    image_color: torch.Tensor
    tokens: torch.Tensor
    answer: torch.Tensor
    task: torch.Tensor

    def subset(self, indices: list[int]) -> "AVSet":
        return AVSet(
            examples=[self.examples[index] for index in indices],
            image=self.image[indices],
            image_color=self.image_color[indices],
            tokens=self.tokens[indices],
            answer=self.answer[indices],
            task=self.task[indices],
        )

    def to(self, device: torch.device) -> "AVSet":
        return AVSet(
            self.examples,
            self.image.to(device=device),
            self.image_color.to(device=device),
            self.tokens.to(device=device),
            self.answer.to(device=device),
            self.task.to(device=device),
        )


class CheckpointedEncoder(nn.Module):
    def __init__(self, layer: nn.TransformerEncoderLayer, layers: int, checkpoint_activations: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList([layer if index == 0 else self._clone_layer(layer) for index in range(layers)])
        self.checkpoint_activations = checkpoint_activations

    @staticmethod
    def _clone_layer(layer: nn.TransformerEncoderLayer) -> nn.TransformerEncoderLayer:
        clone = nn.TransformerEncoderLayer(
            d_model=layer.self_attn.embed_dim,
            nhead=layer.self_attn.num_heads,
            dim_feedforward=layer.linear1.out_features,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        return clone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_activations and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            return x
        for layer in self.layers:
            x = layer(x)
        return x


def issue_for(html: int, csv: int, pdf: int, screen: int) -> int:
    if html != csv:
        return 0
    if csv > pdf:
        return 1
    if screen != html:
        return 2
    return 3


def make_example(index: int, *, rng: random.Random) -> AVExample:
    task = index % len(TASKS)
    colors = list(range(COLORS))
    rng.shuffle(colors)
    image_colors = tuple(colors)
    text_target_zone = rng.randrange(ZONES)
    text_rule = rng.randrange(VALUES)
    text_target_color = rng.randrange(COLORS)
    telemetry = rng.randrange(VALUES)
    phase = rng.randrange(VALUES)
    seen_zone = rng.randrange(ZONES)
    last_zone = rng.randrange(ZONES)
    last_color = image_colors[last_zone]
    memory = [-1] * ZONES
    tool_values = (0, 0, 0, 0)
    if task == TASK_AR:
        memory = [rng.randrange(COLORS) for _ in range(ZONES)]
        text_target_zone = 0
        answer = (image_colors[0] + text_rule) % VALUES
    elif task == TASK_AS:
        issue_target = index % 4
        if issue_target == 0:
            html = rng.randrange(VALUES)
            csv = (html + 1 + rng.randrange(VALUES - 1)) % VALUES
            pdf = max(html, csv)
            screen = html
        elif issue_target == 1:
            html = rng.randrange(1, VALUES)
            csv = html
            pdf = rng.randrange(html)
            screen = html
        elif issue_target == 2:
            html = rng.randrange(VALUES)
            csv = html
            pdf = max(html, rng.randrange(VALUES))
            screen = (html + 1 + rng.randrange(VALUES - 1)) % VALUES
        else:
            html = rng.randrange(VALUES)
            csv = html
            pdf = max(html, rng.randrange(VALUES))
            screen = html
        tool_values = (html, csv, pdf, screen)
        answer = 4 + issue_for(html, csv, pdf, screen)
    else:
        target_color = image_colors[(phase + 1) % ZONES]
        text_target_color = target_color
        for zone, color in enumerate(image_colors):
            memory[zone] = color
        if index % 5 == 0:
            zone = image_colors.index(target_color)
            swap = (zone + 1) % ZONES
            memory[zone], memory[swap] = memory[swap], memory[zone]
            last_zone = zone
            last_color = image_colors[zone]
        answer = 8 + image_colors.index(target_color)
    return AVExample(
        task=task,
        image_colors=image_colors,
        text_target_zone=text_target_zone,
        text_rule=text_rule,
        text_target_color=text_target_color,
        tool_values=tool_values,
        memory_values=tuple(memory),
        telemetry=telemetry,
        phase=phase,
        seen_zone=seen_zone,
        last_zone=last_zone,
        last_color=last_color,
        answer=answer,
        case_id=f"case-{index:05d}",
    )


def render_image_tokens(example: AVExample) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.88, 0.10, 0.10],
            [0.10, 0.70, 0.25],
            [0.15, 0.32, 0.90],
            [0.94, 0.74, 0.12],
        ],
        dtype=torch.float32,
    )
    image = palette[list(example.image_colors)]
    return image + torch.randn_like(image) * 0.01


def token_row(example: AVExample) -> list[int]:
    text = [
        TEXT_TARGET_BASE + example.text_target_zone,
        TEXT_RULE_BASE + example.text_rule,
        TEXT_TARGET_COLOR_BASE + example.text_target_color,
        TEXT_FILLER,
        TEXT_FILLER,
        TEXT_FILLER,
    ]
    tools = [
        TOOL_HTML_BASE + example.tool_values[0],
        TOOL_CSV_BASE + example.tool_values[1],
        TOOL_PDF_BASE + example.tool_values[2],
        TOOL_SCREEN_BASE + example.tool_values[3],
    ]
    memory = [
        MEM_UNKNOWN if color < 0 else MEM_ZONE_BASE + zone * (COLORS + 1) + color + 1
        for zone, color in enumerate(example.memory_values)
    ]
    state = [
        STATE_TELEMETRY_BASE + example.telemetry,
        STATE_PHASE_BASE + example.phase,
        STATE_SEEN_BASE + example.seen_zone,
        STATE_LAST_BASE + example.last_zone * COLORS + example.last_color,
    ]
    return [TASK_BASE + example.task] + text + tools + memory + state


def build_split(split: str, size: int, config: StageAVConfig) -> AVSet:
    offsets = {"train": 0, "val": 10_000, "test": 20_000}
    rng = random.Random(config.seed + offsets[split])
    examples = [make_example(index, rng=rng) for index in range(size)]
    return AVSet(
        examples=examples,
        image=torch.stack([render_image_tokens(example) for example in examples]),
        image_color=torch.tensor([example.image_colors for example in examples], dtype=torch.long),
        tokens=torch.tensor([token_row(example) for example in examples], dtype=torch.long),
        answer=torch.tensor([example.answer for example in examples], dtype=torch.long),
        task=torch.tensor([example.task for example in examples], dtype=torch.long),
    )


def random_batch(data: AVSet, *, rng: random.Random, batch_size: int, device: torch.device) -> AVSet:
    return data.subset([rng.randrange(len(data.examples)) for _ in range(batch_size)]).to(device)


class FromScratchMicroOmni(nn.Module):
    def __init__(self, config: StageAVConfig, token_len: int) -> None:
        super().__init__()
        self.image = nn.Linear(3, config.d_model)
        self.token = nn.Embedding(TOKEN_VOCAB, config.d_model, padding_idx=PAD)
        self.type_embed = nn.Embedding(4, config.d_model)
        seq_len = ZONES + token_len + config.latent_tokens
        self.position = nn.Parameter(torch.randn(seq_len, config.d_model) * 0.02)
        self.latent = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = CheckpointedEncoder(layer, config.layers, config.checkpoint_activations)
        self.norm = nn.LayerNorm(config.d_model)
        self.answer = nn.Linear(config.d_model, ANSWER_CLASSES)
        self.task_probe = nn.Linear(config.d_model, len(TASKS))
        self.color_probe = nn.Linear(config.d_model, COLORS)

    def forward(
        self,
        batch: AVSet,
        *,
        no_image: bool = False,
        no_text: bool = False,
        no_tool_history: bool = False,
        no_memory: bool = False,
        shuffled_image: bool = False,
    ) -> dict[str, torch.Tensor]:
        image = batch.image
        if shuffled_image:
            image = image.flip(0)
        if no_image:
            image = torch.zeros_like(image)
        tokens = batch.tokens.clone()
        if no_text:
            tokens[:, 1 : 1 + TEXT_LEN] = TEXT_FILLER
        if no_tool_history:
            tokens[:, 1 + TEXT_LEN : 1 + TEXT_LEN + TOOL_LEN] = PAD
        if no_memory:
            start = 1 + TEXT_LEN + TOOL_LEN
            tokens[:, start : start + MEMORY_LEN] = MEM_UNKNOWN
        image_tokens = self.image(image) + self.type_embed.weight[0].view(1, 1, -1)
        text_tokens = self.token(tokens[:, : 1 + TEXT_LEN]) + self.type_embed.weight[1].view(1, 1, -1)
        evidence_tokens = self.token(tokens[:, 1 + TEXT_LEN :]) + self.type_embed.weight[2].view(1, 1, -1)
        latent = self.latent.unsqueeze(0).expand(tokens.shape[0], -1, -1) + self.type_embed.weight[3].view(1, 1, -1)
        x = torch.cat((image_tokens, text_tokens, evidence_tokens, latent), dim=1)
        x = x + self.position[: x.shape[1]].view(1, -1, x.shape[-1])
        encoded = self.encoder(x)
        latent_out = self.norm(encoded[:, -latent.shape[1] :])
        evidence_out = self.norm(encoded[:, : -latent.shape[1]])
        pooled = latent_out.mean(dim=1) + evidence_out.mean(dim=1)
        image_zone = self.norm(encoded[:, :ZONES])
        return {
            "answer": self.answer(pooled),
            "task": self.task_probe(pooled),
            "color": self.color_probe(image_zone),
        }


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().item())


def task_metrics(logits: torch.Tensor, target: torch.Tensor, tasks: torch.Tensor, prefix: str) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    out = {f"{prefix}_answer_accuracy": float((pred == target).float().mean().item())}
    for task_id, name in enumerate(TASKS):
        mask = tasks == task_id
        if bool(mask.any()):
            out[f"{prefix}_{name}_accuracy"] = float((pred[mask] == target[mask]).float().mean().item())
    return out


def balanced_answer_loss(logits: torch.Tensor, answer: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
    per_example = F.cross_entropy(logits, answer, reduction="none")
    losses = []
    for task_id in range(len(TASKS)):
        mask = task == task_id
        if bool(mask.any()):
            losses.append(per_example[mask].mean())
    return torch.stack(losses).mean()


def train_model(model: FromScratchMicroOmni, train: AVSet, val: AVSet, config: StageAVConfig, device: torch.device) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    rng = random.Random(config.seed + 909)
    history = []
    for step in range(1, config.steps + 1):
        model.train()
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.cuda.amp.autocast(enabled=config.amp and device.type == "cuda"):
            outputs = model(batch)
            loss = (
                balanced_answer_loss(outputs["answer"], batch.answer, batch.task)
                + 0.15 * F.cross_entropy(outputs["task"], batch.task)
                + 0.50 * F.cross_entropy(outputs["color"].reshape(-1, COLORS), batch.image_color.reshape(-1))
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate(model, val, device, prefix="val")
            metrics["step"] = float(step)
            metrics["loss"] = float(loss.detach().item())
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
    return {"history": history}


@torch.no_grad()
def evaluate(model: FromScratchMicroOmni, data: AVSet, device: torch.device, *, prefix: str) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    full = model(batch)
    metrics = task_metrics(full["answer"], batch.answer, batch.task, prefix)
    metrics[f"{prefix}_task_probe_accuracy"] = accuracy(full["task"], batch.task)
    for ablation in ("no_image", "no_text", "no_tool_history", "no_memory", "shuffled_image"):
        outputs = model(batch, **{ablation: True})
        metrics[f"{prefix}_{ablation}_answer_accuracy"] = accuracy(outputs["answer"], batch.answer)
    return metrics


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def render_case_png(example: AVExample) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.88, 0.10, 0.10],
            [0.10, 0.70, 0.25],
            [0.15, 0.32, 0.90],
            [0.94, 0.74, 0.12],
        ],
        dtype=torch.float32,
    )
    image = torch.full((3, 80, 80), 0.06)
    boxes = ((6, 6), (46, 6), (6, 46), (46, 46))
    for zone, (x, y) in enumerate(boxes):
        image[:, y : y + 28, x : x + 28] = palette[example.image_colors[zone]].view(3, 1, 1)
    return image


@torch.no_grad()
def write_samples(model: FromScratchMicroOmni, data: AVSet, path: Path, device: torch.device, limit: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    for index, example in enumerate(data.examples[:limit]):
        png = path / f"{example.case_id}.png"
        write_png(png, render_case_png(example))
        single = data.subset([index]).to(device)
        pred = int(model(single)["answer"].argmax(dim=-1).item())
        rows.append(
            {
                "case_id": example.case_id,
                "task": TASKS[example.task],
                "map_png": str(png),
                "answer": example.answer,
                "prediction": pred,
                "image_colors": list(example.image_colors),
                "memory": list(example.memory_values),
                "tool_values": list(example.tool_values),
            }
        )
    (path / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(config: StageAVConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train = build_split("train", config.train_size, config)
    val = build_split("val", config.val_size, config)
    test = build_split("test", config.test_size, config)
    model = FromScratchMicroOmni(config, train.tokens.shape[1])
    training = train_model(model, train, val, config, device)
    metrics = evaluate(model, test, device, prefix="test")
    write_samples(model, test, output_path.parent / "samples", device, config.sample_count)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    result = {
        "schema_version": 1,
        "stage": "AV",
        "config": asdict(config),
        "device": str(device),
        "metrics": metrics,
        "cost": {
            "elapsed_sec": time.perf_counter() - started,
            "parameters": parameter_count(model),
            "peak_cuda_allocated_mb": peak_mb,
        },
        "training": training,
        "interpretation": {
            "goal": "Train one larger from-scratch integrated micro-omni model over visual cross-expert, file-tool audit, and memory-exploration tasks.",
            "success_condition": "Full accuracy >= 85% with clear modality/tool/memory ablation drops and at least 3 seeds for formal claims.",
            "boundary": "This is a larger local architecture validation model, not a generally useful foundation model.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": metrics, "cost": result["cost"]}, ensure_ascii=False, indent=2), flush=True)
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
    ablation_drops = {
        name: summary["metrics_test_answer_accuracy"] - summary[f"metrics_test_{name}_answer_accuracy"]
        for name in ("no_image", "no_text", "no_tool_history", "no_memory", "shuffled_image")
    }
    return {
        "schema_version": 1,
        "stage": "AV",
        "runs": runs,
        "summary": summary,
        "stdev": stdev,
        "gates": {
            "answer_accuracy": summary["metrics_test_answer_accuracy"],
            "cross_expert_visual_accuracy": summary["metrics_test_cross_expert_visual_accuracy"],
            "file_audit_tools_accuracy": summary["metrics_test_file_audit_tools_accuracy"],
            "memory_exploration_accuracy": summary["metrics_test_memory_exploration_accuracy"],
            "ablation_drops": ablation_drops,
            "parameters": summary["cost_parameters"],
            "peak_cuda_allocated_mb": summary.get("cost_peak_cuda_allocated_mb", 0.0),
            "passes_answer_85": summary["metrics_test_answer_accuracy"] >= 0.85,
            "passes_each_task_80": all(
                summary[key] >= 0.80
                for key in (
                    "metrics_test_cross_expert_visual_accuracy",
                    "metrics_test_file_audit_tools_accuracy",
                    "metrics_test_memory_exploration_accuracy",
                )
            ),
            "passes_any_relevant_ablation_20pp": any(drop >= 0.20 for drop in ablation_drops.values()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_av_from_scratch_micro_omni/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_av_from_scratch_micro_omni/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_av_from_scratch_micro_omni/sweep_results.json"))
    parser.add_argument("--train-size", type=int, default=StageAVConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageAVConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageAVConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageAVConfig.batch_size)
    parser.add_argument("--d-model", type=int, default=StageAVConfig.d_model)
    parser.add_argument("--heads", type=int, default=StageAVConfig.heads)
    parser.add_argument("--layers", type=int, default=StageAVConfig.layers)
    parser.add_argument("--latent-tokens", type=int, default=StageAVConfig.latent_tokens)
    parser.add_argument("--steps", type=int, default=StageAVConfig.steps)
    parser.add_argument("--eval-every", type=int, default=StageAVConfig.eval_every)
    parser.add_argument("--sample-count", type=int, default=StageAVConfig.sample_count)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-activations", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, seed: int) -> StageAVConfig:
    return StageAVConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=seed,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        latent_tokens=args.latent_tokens,
        steps=args.steps,
        eval_every=args.eval_every,
        sample_count=args.sample_count,
        amp=not args.no_amp,
        checkpoint_activations=args.checkpoint_activations,
    )


def main() -> None:
    args = parse_args()
    if args.sweep:
        runs = []
        for seed_text in args.seeds.split(","):
            seed = int(seed_text.strip())
            runs.append(run_one(config_from_args(args, seed), args.output_dir / f"seed{seed}.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"aggregate": aggregate["gates"]}, ensure_ascii=False, indent=2), flush=True)
    else:
        first_seed = int(args.seeds.split(",")[0].strip())
        run_one(config_from_args(args, first_seed), args.output)


if __name__ == "__main__":
    main()
