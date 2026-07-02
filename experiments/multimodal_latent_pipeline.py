from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F


GRID_SIZE = 8
MOVE_TO_DELTA = {
    0: (0, -1),  # U
    1: (0, 1),  # D
    2: (-1, 0),  # L
    3: (1, 0),  # R
}
VOCAB_SIZE = 8 + GRID_SIZE * 2 + 4
TOKEN_Q = 0
TOKEN_M = 1
TOKEN_X_BASE = 8
TOKEN_Y_BASE = TOKEN_X_BASE + GRID_SIZE
TOKEN_MOVE_BASE = TOKEN_Y_BASE + GRID_SIZE


@dataclass(frozen=True)
class PipelineConfig:
    move_count: int = 10
    train_size: int = 4096
    val_size: int = 768
    test_size: int = 1024
    batch_size: int = 256
    seed: int = 20260701
    d_text: int = 128
    d_latent: int = 64
    text_steps: int = 700
    codec_steps: int = 300
    parallel_steps: int = 2500
    output_steps: int = 300
    eval_every: int = 300
    lr: float = 9e-4
    parallel_step_loss_weight: float = 2.0
    parallel_final_loss_weight: float = 0.5
    parallel_latent_loss_weight: float = 1.0
    parallel_codec_loss_weight: float = 5.0


@dataclass(frozen=True)
class Example:
    start: int
    moves: tuple[int, ...]
    states: tuple[int, ...]


def coord_to_xy(coord: int) -> tuple[int, int]:
    return coord // GRID_SIZE, coord % GRID_SIZE


def xy_to_coord(x: int, y: int) -> int:
    return x * GRID_SIZE + y


def apply_move(coord: int, move: int) -> int:
    x, y = coord_to_xy(coord)
    dx, dy = MOVE_TO_DELTA[move]
    return xy_to_coord((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)


def generate_examples(count: int, *, seed: int, move_count: int, exclude: set[tuple[int, tuple[int, ...]]] | None = None) -> list[Example]:
    rng = random.Random(seed)
    seen = set(exclude or set())
    examples: list[Example] = []
    while len(examples) < count:
        start = rng.randrange(GRID_SIZE * GRID_SIZE)
        moves = tuple(rng.randrange(4) for _ in range(move_count))
        key = (start, moves)
        if key in seen:
            continue
        seen.add(key)
        coord = start
        states: list[int] = []
        for move in moves:
            coord = apply_move(coord, move)
            states.append(coord)
        examples.append(Example(start=start, moves=moves, states=tuple(states)))
    return examples


def split_examples(config: PipelineConfig) -> tuple[list[Example], list[Example], list[Example]]:
    train = generate_examples(config.train_size, seed=config.seed, move_count=config.move_count)
    used = {(example.start, example.moves) for example in train}
    val = generate_examples(config.val_size, seed=config.seed + 1, move_count=config.move_count, exclude=used)
    used.update((example.start, example.moves) for example in val)
    test = generate_examples(config.test_size, seed=config.seed + 2, move_count=config.move_count, exclude=used)
    return train, val, test


def prompt_tokens(examples: list[Example], device: torch.device) -> torch.Tensor:
    tokens: list[list[int]] = []
    for example in examples:
        x, y = coord_to_xy(example.start)
        row = [TOKEN_Q, TOKEN_X_BASE + x, TOKEN_Y_BASE + y, TOKEN_M]
        row.extend(TOKEN_MOVE_BASE + move for move in example.moves)
        tokens.append(row)
    return torch.tensor(tokens, dtype=torch.long, device=device)


def batch_examples(
    examples: list[Example],
    *,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
    return (
        prompt_tokens(batch, device),
        torch.tensor([example.start for example in batch], dtype=torch.long, device=device),
        torch.tensor([example.states for example in batch], dtype=torch.long, device=device),
    )


class TextThoughtModel(nn.Module):
    """Pure-text thought model: prompt tokens -> visible coordinate chain."""

    def __init__(self, d_text: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_text)
        self.rnn = nn.GRU(d_text, d_text, batch_first=True)
        self.step_head = nn.Linear(d_text, GRID_SIZE * GRID_SIZE)
        self.final_head = nn.Linear(d_text, GRID_SIZE * GRID_SIZE)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.rnn(self.embedding(tokens))
        move_hidden = hidden[:, 4:, :]
        return self.step_head(move_hidden), self.final_head(move_hidden[:, -1, :])


class LatentCodec(nn.Module):
    """Translate visible X/Y text states into continuous latent vectors and back."""

    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.coord_embedding = nn.Embedding(GRID_SIZE * GRID_SIZE, d_latent)
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, GRID_SIZE * GRID_SIZE),
        )

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        return self.coord_embedding(coords)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class LatentReasoner(nn.Module):
    """Latent variable thinker with text-input translation and text-output decoding."""

    def __init__(self, d_text: int, d_latent: int, codec: LatentCodec) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_text)
        self.start_to_latent = nn.Sequential(
            nn.Linear(d_text * 2, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )
        self.move_to_latent = nn.Embedding(4, d_latent)
        self.cell = nn.GRUCell(d_latent, d_latent)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, GRID_SIZE * GRID_SIZE),
        )
        self.step_text_head = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, GRID_SIZE * GRID_SIZE),
        )
        self.codec = codec

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_tokens = tokens[:, 1]
        y_tokens = tokens[:, 2]
        start_text = torch.cat([self.token_embedding(x_tokens), self.token_embedding(y_tokens)], dim=-1)
        z = self.start_to_latent(start_text)
        latent_states: list[torch.Tensor] = []
        move_tokens = tokens[:, 4:] - TOKEN_MOVE_BASE
        for index in range(move_tokens.size(1)):
            z = self.cell(self.move_to_latent(move_tokens[:, index]), z)
            latent_states.append(z)
        latents = torch.stack(latent_states, dim=1)
        step_logits = self.step_text_head(latents)
        final_logits = self.output_head(latents[:, -1, :])
        return latents, step_logits, final_logits


def coord_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def final_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets[:, -1])


def train_text_model(
    model: TextThoughtModel,
    train: list[Example],
    val: list[Example],
    config: PipelineConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    for step in range(1, config.text_steps + 1):
        tokens, _, states = batch_examples(train, rng=rng, batch_size=config.batch_size, device=device)
        step_logits, final_logits = model(tokens)
        loss = coord_loss(step_logits, states) + final_loss(final_logits, states)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.text_steps:
            metrics = evaluate_text(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(f"text step={step:4d} loss={float(loss.detach().cpu()):.4f} final={metrics['final_exact']:.3f}", flush=True)
    return {"history": history}


def train_codec(codec: LatentCodec, config: PipelineConfig, *, device: torch.device) -> dict[str, object]:
    coords = torch.arange(GRID_SIZE * GRID_SIZE, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=config.lr, weight_decay=0.0)
    history: list[dict[str, float | int]] = []
    for step in range(1, config.codec_steps + 1):
        logits = codec.decode(codec.encode(coords))
        loss = F.cross_entropy(logits, coords)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.codec_steps:
            exact = float((logits.argmax(dim=-1) == coords).float().mean().detach().cpu())
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "reconstruction_exact": exact})
            print(f"codec step={step:4d} loss={float(loss.detach().cpu()):.4f} recon={exact:.3f}", flush=True)
    for parameter in codec.parameters():
        parameter.requires_grad = False
    codec.eval()
    return {"history": history}


def train_latent_reasoner(
    model: LatentReasoner,
    train: list[Example],
    val: list[Example],
    config: PipelineConfig,
    *,
    stage: str,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    rng = random.Random(seed)
    steps = config.parallel_steps if stage == "parallel" else config.output_steps
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
        weight_decay=0.01,
    )
    history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        tokens, _, states = batch_examples(train, rng=rng, batch_size=config.batch_size, device=device)
        latents, step_logits, final_logits = model(tokens)
        if stage == "parallel":
            with torch.no_grad():
                target_latents = model.codec.encode(states)
            z_loss = F.mse_loss(latents, target_latents)
            codec_step_loss = coord_loss(model.codec.decode(latents), states)
            loss = (
                config.parallel_final_loss_weight * final_loss(final_logits, states)
                + config.parallel_step_loss_weight * coord_loss(step_logits, states)
                + config.parallel_latent_loss_weight * z_loss
                + config.parallel_codec_loss_weight * codec_step_loss
            )
        else:
            loss = final_loss(final_logits, states)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_latent(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(f"{stage} step={step:4d} loss={float(loss.detach().cpu()):.4f} final={metrics['final_exact']:.3f}", flush=True)
    return {"history": history}


@torch.no_grad()
def evaluate_text(
    model: TextThoughtModel,
    examples: list[Example],
    config: PipelineConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct_step = 0
    correct_codec_step = 0
    total_step = 0
    correct_final = 0
    for start in range(0, len(examples), config.batch_size):
        batch = examples[start : start + config.batch_size]
        tokens = prompt_tokens(batch, device)
        states = torch.tensor([example.states for example in batch], dtype=torch.long, device=device)
        step_logits, final_logits = model(tokens)
        correct_step += int((step_logits.argmax(dim=-1) == states).sum().detach().cpu())
        total_step += states.numel()
        correct_final += int((final_logits.argmax(dim=-1) == states[:, -1]).sum().detach().cpu())
    model.train()
    return {
        "step_exact": correct_step / total_step,
        "final_exact": correct_final / len(examples),
    }


@torch.no_grad()
def evaluate_latent(
    model: LatentReasoner,
    examples: list[Example],
    config: PipelineConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct_step = 0
    correct_codec_step = 0
    total_step = 0
    correct_final = 0
    cosine_total = 0.0
    cosine_count = 0
    for start in range(0, len(examples), config.batch_size):
        batch = examples[start : start + config.batch_size]
        tokens = prompt_tokens(batch, device)
        states = torch.tensor([example.states for example in batch], dtype=torch.long, device=device)
        latents, step_logits, final_logits = model(tokens)
        target_latents = model.codec.encode(states)
        codec_logits = model.codec.decode(latents)
        correct_step += int((step_logits.argmax(dim=-1) == states).sum().detach().cpu())
        correct_codec_step += int((codec_logits.argmax(dim=-1) == states).sum().detach().cpu())
        total_step += states.numel()
        correct_final += int((final_logits.argmax(dim=-1) == states[:, -1]).sum().detach().cpu())
        cosine = F.cosine_similarity(latents.reshape(-1, latents.size(-1)), target_latents.reshape(-1, target_latents.size(-1)), dim=-1)
        cosine_total += float(cosine.sum().detach().cpu())
        cosine_count += cosine.numel()
    model.train()
    return {
        "step_text_exact": correct_step / total_step,
        "codec_step_exact": correct_codec_step / total_step,
        "final_exact": correct_final / len(examples),
        "latent_cosine": cosine_total / cosine_count,
    }


def run_pipeline(config: PipelineConfig, output_path: Path) -> dict[str, object]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = split_examples(config)
    print(f"device={device} move_count={config.move_count} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    text_model = TextThoughtModel(config.d_text).to(device)
    text_log = train_text_model(text_model, train, val, config, seed=config.seed + 10, device=device)
    text_test = evaluate_text(text_model, test, config, device=device)

    codec = LatentCodec(config.d_latent).to(device)
    codec_log = train_codec(codec, config, device=device)

    latent_model = LatentReasoner(config.d_text, config.d_latent, codec).to(device)
    parallel_log = train_latent_reasoner(
        latent_model,
        train,
        val,
        config,
        stage="parallel",
        seed=config.seed + 20,
        device=device,
    )
    parallel_test = evaluate_latent(latent_model, test, config, device=device)

    output_log = train_latent_reasoner(
        latent_model,
        train,
        val,
        config,
        stage="output",
        seed=config.seed + 30,
        device=device,
    )
    output_test = evaluate_latent(latent_model, test, config, device=device)

    result = {
        "experiment": "multimodal_like_latent_pipeline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": config.__dict__,
        "stages": {
            "pure_text_training": {
                "description": "Text prompt -> visible coordinate thought chain and final text coordinate.",
                "train_log": text_log,
                "test_metrics": text_test,
            },
            "latent_translation": {
                "description": "Visible X/Y state text -> continuous latent vector -> X/Y state text.",
                "train_log": codec_log,
                "test_metrics": codec_log["history"][-1],
            },
            "latent_plus_text_parallel": {
                "description": "Latent recurrent thought supervised by latent targets plus visible text-thought labels.",
                "train_log": parallel_log,
                "test_metrics": parallel_test,
            },
            "latent_thought_text_output": {
                "description": "Latent recurrent thought with only final pure-text coordinate output loss.",
                "train_log": output_log,
                "test_metrics": output_test,
            },
        },
        "interpretation": interpret(text_test, parallel_test, output_test),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def interpret(
    text_test: dict[str, float],
    parallel_test: dict[str, float],
    output_test: dict[str, float],
) -> dict[str, object]:
    return {
        "latent_pipeline_feasible": output_test["final_exact"] >= 0.75,
        "parallel_stage_close_to_text": parallel_test["final_exact"] >= text_test["final_exact"] - 0.15,
        "output_stage_retains_parallel_capability": output_test["final_exact"] >= parallel_test["final_exact"] - 0.1,
        "summary": (
            "连续潜变量管线在该 toy task 上可运行。"
            if output_test["final_exact"] >= 0.75
            else "该配置下连续潜变量管线尚未达到可接受准确率。"
        ),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    move_counts = [int(item) for item in args.move_counts.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    runs: list[dict[str, object]] = []
    started = time.perf_counter()
    for move_count in move_counts:
        for seed in seeds:
            config = PipelineConfig(
                move_count=move_count,
                seed=seed,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                batch_size=args.batch_size,
                d_text=args.d_text,
                d_latent=args.d_latent,
                text_steps=args.text_steps,
                codec_steps=args.codec_steps,
                parallel_steps=args.parallel_steps,
                output_steps=args.output_steps,
                eval_every=args.eval_every,
                lr=args.lr,
                parallel_step_loss_weight=args.parallel_step_loss_weight,
                parallel_final_loss_weight=args.parallel_final_loss_weight,
                parallel_latent_loss_weight=args.parallel_latent_loss_weight,
                parallel_codec_loss_weight=args.parallel_codec_loss_weight,
            )
            output_path = output_dir / f"move{move_count}_seed{seed}.json"
            print(f"=== multimodal sweep move={move_count} seed={seed} ===", flush=True)
            result = run_pipeline(config, output_path)
            runs.append(
                {
                    "move_count": move_count,
                    "seed": seed,
                    "output": str(output_path),
                    "metrics": {
                        stage: result["stages"][stage]["test_metrics"]
                        for stage in result["stages"]
                    },
                    "interpretation": result["interpretation"],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    aggregate = {
        "experiment": "multimodal_like_latent_pipeline_sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": runs,
        "summary": summarize_runs(runs),
        "seconds": round(time.perf_counter() - started, 2),
    }
    aggregate_path = Path(args.aggregate)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    def values(stage: str, metric: str, selected: list[dict[str, object]]) -> list[float]:
        return [float(run["metrics"][stage][metric]) for run in selected]  # type: ignore[index]

    result: dict[str, object] = {}
    for move_count in sorted({int(run["move_count"]) for run in runs}):
        selected = [run for run in runs if int(run["move_count"]) == move_count]
        result[str(move_count)] = summarize_group(selected, values)
    result["overall"] = summarize_group(runs, values)
    return result


def summarize_group(runs: list[dict[str, object]], values_fn) -> dict[str, object]:
    summary: dict[str, object] = {"count": len(runs)}
    metrics = {
        "pure_text_final": values_fn("pure_text_training", "final_exact", runs),
        "parallel_final": values_fn("latent_plus_text_parallel", "final_exact", runs),
        "parallel_step_text": values_fn("latent_plus_text_parallel", "step_text_exact", runs),
        "parallel_latent_cosine": values_fn("latent_plus_text_parallel", "latent_cosine", runs),
        "parallel_codec_step": values_fn("latent_plus_text_parallel", "codec_step_exact", runs),
        "output_final": values_fn("latent_thought_text_output", "final_exact", runs),
        "output_step_text_diagnostic": values_fn("latent_thought_text_output", "step_text_exact", runs),
        "output_codec_step_diagnostic": values_fn("latent_thought_text_output", "codec_step_exact", runs),
    }
    for name, items in metrics.items():
        summary[name] = {
            "mean": statistics.fmean(items),
            "min": min(items),
            "max": max(items),
            "stdev": statistics.pstdev(items) if len(items) > 1 else 0.0,
        }
    summary["output_minus_parallel_final_mean"] = statistics.fmean(
        output - parallel
        for output, parallel in zip(metrics["output_final"], metrics["parallel_final"])
    )
    summary["output_minus_text_final_mean"] = statistics.fmean(
        output - text
        for output, text in zip(metrics["output_final"], metrics["pure_text_final"])
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/multimodal_latent_pipeline/result.json")
    parser.add_argument("--aggregate", default="artifacts/multimodal_latent_pipeline/sweep_results.json")
    parser.add_argument("--output-dir", default="artifacts/multimodal_latent_pipeline/sweep_runs")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--move-count", type=int, default=10)
    parser.add_argument("--move-counts", default="8,10,12")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=768)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-text", type=int, default=128)
    parser.add_argument("--d-latent", type=int, default=64)
    parser.add_argument("--text-steps", type=int, default=700)
    parser.add_argument("--codec-steps", type=int, default=300)
    parser.add_argument("--parallel-steps", type=int, default=2500)
    parser.add_argument("--output-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--parallel-step-loss-weight", type=float, default=2.0)
    parser.add_argument("--parallel-final-loss-weight", type=float, default=0.5)
    parser.add_argument("--parallel-latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--parallel-codec-loss-weight", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep:
        result = run_sweep(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        return
    config = PipelineConfig(
        move_count=args.move_count,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_text=args.d_text,
        d_latent=args.d_latent,
        text_steps=args.text_steps,
        codec_steps=args.codec_steps,
        parallel_steps=args.parallel_steps,
        output_steps=args.output_steps,
        eval_every=args.eval_every,
        lr=args.lr,
        parallel_step_loss_weight=args.parallel_step_loss_weight,
        parallel_final_loss_weight=args.parallel_final_loss_weight,
        parallel_latent_loss_weight=args.parallel_latent_loss_weight,
        parallel_codec_loss_weight=args.parallel_codec_loss_weight,
    )
    result = run_pipeline(config, Path(args.output))
    print(json.dumps(result["stages"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(result["interpretation"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
