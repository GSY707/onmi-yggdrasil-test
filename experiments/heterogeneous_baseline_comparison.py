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

from heterogeneous_latent_input import (
    CELL_COUNT,
    MOVES,
    TERRAIN_TYPES,
    HeteroExample,
    TerrainCodec,
    batch_examples,
    effective_move_table,
    evaluate_latent_rollout,
    examples_to_tensors,
    generate_examples,
    latent_rollout,
    terrain_one_hot,
    train_codec,
    transition_table,
    transform_move,
)


@dataclass(frozen=True)
class BaselineConfig:
    train_move_count: int = 8
    eval_move_counts: tuple[int, ...] = (8, 16, 32)
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 2048
    batch_size: int = 256
    seed: int = 20260701
    d_latent: int = 32
    d_model: int = 256
    codec_steps: int = 300
    effective_steps: int = 120
    transition_steps: int = 180
    raw_steps: int = 2000
    eval_every: int = 500
    lr: float = 1e-3
    table_lr: float = 0.1


def stat(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_split(config: BaselineConfig) -> tuple[list[HeteroExample], list[HeteroExample]]:
    return (
        generate_examples(config.train_size, seed=config.seed, move_count=config.train_move_count),
        generate_examples(config.val_size, seed=config.seed + 1, move_count=config.train_move_count),
    )


def test_sets(config: BaselineConfig) -> dict[int, list[HeteroExample]]:
    return {
        move_count: generate_examples(config.test_size, seed=config.seed + 1000 + move_count, move_count=move_count)
        for move_count in config.eval_move_counts
    }


class EffectiveMoveTableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(TERRAIN_TYPES, MOVES, MOVES))

    def forward(self, terrain_type: torch.Tensor, move: torch.Tensor) -> torch.Tensor:
        return self.logits[terrain_type, move]

    def decoded_table(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)


class TransitionTableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(CELL_COUNT, TERRAIN_TYPES, MOVES, CELL_COUNT))

    def forward(self, coord: torch.Tensor, terrain_type: torch.Tensor, move: torch.Tensor) -> torch.Tensor:
        return self.logits[coord, terrain_type, move]

    def decoded_table(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)


class RawStepDirectModel(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.terrain_embedding = nn.Embedding(TERRAIN_TYPES, d_model // 4)
        self.coord_embedding = nn.Embedding(CELL_COUNT, d_model)
        self.move_embedding = nn.Embedding(MOVES, d_model)
        self.map_projection = nn.Sequential(
            nn.Linear(CELL_COUNT * (d_model // 4), d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.gru = nn.GRU(d_model * 2, d_model, num_layers=2, batch_first=True)
        self.head = nn.Linear(d_model, CELL_COUNT)

    def forward(self, terrain: torch.Tensor, start: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        map_flat = self.terrain_embedding(terrain).reshape(terrain.size(0), -1)
        map_context = self.map_projection(map_flat)
        h0 = (map_context + self.coord_embedding(start)).unsqueeze(0).expand(2, -1, -1).contiguous()
        move_context = self.move_embedding(moves)
        repeated_map = map_context.unsqueeze(1).expand(-1, moves.size(1), -1)
        hidden, _ = self.gru(torch.cat([move_context, repeated_map], dim=-1), h0)
        return self.head(hidden)


def all_effective_move_targets(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    terrain_values: list[int] = []
    move_values: list[int] = []
    targets: list[int] = []
    for terrain_type in range(TERRAIN_TYPES):
        for move in range(MOVES):
            terrain_values.append(terrain_type)
            move_values.append(move)
            targets.append(transform_move(move, terrain_type))
    return (
        torch.tensor(terrain_values, dtype=torch.long, device=device),
        torch.tensor(move_values, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def all_transition_targets(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords: list[int] = []
    terrain_values: list[int] = []
    moves: list[int] = []
    targets: list[int] = []
    transitions = transition_table(device)
    effective = effective_move_table(device)
    for coord in range(CELL_COUNT):
        for terrain_type in range(TERRAIN_TYPES):
            for move in range(MOVES):
                coords.append(coord)
                terrain_values.append(terrain_type)
                moves.append(move)
                targets.append(int(transitions[coord, effective[terrain_type, move]].detach().cpu()))
    return (
        torch.tensor(coords, dtype=torch.long, device=device),
        torch.tensor(terrain_values, dtype=torch.long, device=device),
        torch.tensor(moves, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def train_effective_move_model(
    model: EffectiveMoveTableModel,
    config: BaselineConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    terrain_type, move, target = all_effective_move_targets(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.table_lr)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.effective_steps + 1):
        logits = model(terrain_type, move)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.effective_steps:
            exact = float((logits.argmax(dim=-1) == target).float().mean().detach().cpu())
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 6), "effective_move_exact": exact})
            print(f"effective_table step={step:4d} loss={float(loss.detach().cpu()):.6f} exact={exact:.3f}", flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return {
        "history": history,
        "cost": {
            "train_seconds": round(time.perf_counter() - started, 4),
            "train_steps": config.effective_steps,
            "trainable_parameters": TERRAIN_TYPES * MOVES * MOVES,
        },
    }


def train_transition_table_model(
    model: TransitionTableModel,
    config: BaselineConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    coord, terrain_type, move, target = all_transition_targets(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.table_lr)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.transition_steps + 1):
        logits = model(coord, terrain_type, move)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == config.transition_steps:
            exact = float((logits.argmax(dim=-1) == target).float().mean().detach().cpu())
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 6), "transition_exact": exact})
            print(f"transition_table step={step:4d} loss={float(loss.detach().cpu()):.6f} exact={exact:.3f}", flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return {
        "history": history,
        "cost": {
            "train_seconds": round(time.perf_counter() - started, 4),
            "train_steps": config.transition_steps,
            "trainable_parameters": CELL_COUNT * TERRAIN_TYPES * MOVES * CELL_COUNT,
        },
    }


@torch.no_grad()
def rollout_effective_move_model(
    model: EffectiveMoveTableModel,
    terrain: torch.Tensor,
    start: torch.Tensor,
    moves: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    effective = model.decoded_table()
    transitions = transition_table(device)
    coord = start
    batch_ids = torch.arange(terrain.size(0), device=device)
    states: list[torch.Tensor] = []
    for index in range(moves.size(1)):
        terrain_type = terrain[batch_ids, coord]
        effective_moves = effective[terrain_type, moves[:, index]]
        coord = transitions[coord, effective_moves]
        states.append(coord)
    return torch.stack(states, dim=1)


@torch.no_grad()
def rollout_transition_table_model(
    model: TransitionTableModel,
    terrain: torch.Tensor,
    start: torch.Tensor,
    moves: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    transitions = model.decoded_table()
    coord = start
    batch_ids = torch.arange(terrain.size(0), device=device)
    states: list[torch.Tensor] = []
    for index in range(moves.size(1)):
        terrain_type = terrain[batch_ids, coord]
        coord = transitions[coord, terrain_type, moves[:, index]]
        states.append(coord)
    return torch.stack(states, dim=1)


def score_states(predicted: torch.Tensor, states: torch.Tensor) -> dict[str, float]:
    return {
        "final_exact": float((predicted[:, -1] == states[:, -1]).float().mean().detach().cpu()),
        "step_exact": float((predicted == states).float().mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate_rollout_function(
    examples: list[HeteroExample],
    config: BaselineConfig,
    *,
    device: torch.device,
    rollout_fn,
) -> dict[str, float]:
    correct_final = 0
    correct_step = 0
    total_step = 0
    for start_index in range(0, len(examples), config.batch_size):
        batch = examples[start_index : start_index + config.batch_size]
        terrain, start, moves, states = examples_to_tensors(batch, device=device)
        predicted = rollout_fn(terrain, start, moves)
        correct_final += int((predicted[:, -1] == states[:, -1]).sum().detach().cpu())
        correct_step += int((predicted == states).sum().detach().cpu())
        total_step += states.numel()
    return {"final_exact": correct_final / len(examples), "step_exact": correct_step / total_step}


@torch.no_grad()
def evaluate_raw_step_direct(
    model: RawStepDirectModel,
    examples: list[HeteroExample],
    config: BaselineConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct_final = 0
    correct_step = 0
    total_step = 0
    for start_index in range(0, len(examples), config.batch_size):
        batch = examples[start_index : start_index + config.batch_size]
        terrain, start, moves, states = examples_to_tensors(batch, device=device)
        predicted = model(terrain, start, moves).argmax(dim=-1)
        correct_final += int((predicted[:, -1] == states[:, -1]).sum().detach().cpu())
        correct_step += int((predicted == states).sum().detach().cpu())
        total_step += states.numel()
    model.train()
    return {"final_exact": correct_final / len(examples), "step_exact": correct_step / total_step}


def train_raw_step_direct(
    model: RawStepDirectModel,
    train: list[HeteroExample],
    val: list[HeteroExample],
    config: BaselineConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.raw_steps + 1):
        terrain, start, moves, states = batch_examples(train, rng=rng, batch_size=config.batch_size, device=device)
        logits = model(terrain, start, moves)
        loss = F.cross_entropy(logits.reshape(-1, CELL_COUNT), states.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.raw_steps:
            metrics = evaluate_raw_step_direct(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"raw_step_direct step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"final={metrics['final_exact']:.3f} step_exact={metrics['step_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "cost": {
            "train_seconds": round(time.perf_counter() - started, 4),
            "train_steps": config.raw_steps,
            "trainable_parameters": count_parameters(model),
        },
    }


def evaluate_baseline_by_move(
    test_by_move: dict[int, list[HeteroExample]],
    config: BaselineConfig,
    *,
    evaluator,
) -> dict[str, dict[str, float]]:
    return {str(move_count): evaluator(examples) for move_count, examples in test_by_move.items()}


def summarize_single(results: dict[str, object], eval_move_counts: tuple[int, ...]) -> dict[str, object]:
    final_by_move: dict[str, dict[str, float]] = {}
    step_by_move: dict[str, dict[str, float]] = {}
    for move_count in eval_move_counts:
        key = str(move_count)
        final_by_move[key] = {}
        step_by_move[key] = {}
        for name, result in results.items():
            metrics = result["metrics_by_move"][key]  # type: ignore[index]
            final_by_move[key][name] = float(metrics["final_exact"])
            step_by_move[key][name] = float(metrics["step_exact"])
    return {
        "final_exact_by_move": final_by_move,
        "step_exact_by_move": step_by_move,
        "overall_final_exact": {
            name: statistics.fmean(
                [float(result["metrics_by_move"][str(move_count)]["final_exact"]) for move_count in eval_move_counts]  # type: ignore[index]
            )
            for name, result in results.items()
        },
        "cost": {name: result["cost"] for name, result in results.items()},  # type: ignore[index]
    }


def run_experiment(config: BaselineConfig, output_path: Path) -> dict[str, object]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val = train_split(config)
    tests = test_sets(config)
    print(f"baseline device={device} train_move={config.train_move_count} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    codec = TerrainCodec(config.d_latent).to(device)
    codec_started = time.perf_counter()
    codec_log = train_codec(codec, config, device=device)  # type: ignore[arg-type]
    codec_cost = {
        "train_seconds": round(time.perf_counter() - codec_started, 4),
        "train_steps": config.codec_steps,
        "trainable_parameters": sum(parameter.numel() for parameter in codec.parameters()),
    }

    effective_model = EffectiveMoveTableModel().to(device)
    effective_log = train_effective_move_model(effective_model, config, device=device)

    transition_model = TransitionTableModel().to(device)
    transition_log = train_transition_table_model(transition_model, config, device=device)

    raw_step_model = RawStepDirectModel(config.d_model).to(device)
    raw_step_log = train_raw_step_direct(raw_step_model, train, val, config, seed=config.seed + 10, device=device)

    results: dict[str, object] = {
        "latent_good_codec": {
            "description": "Terrain tensor -> trained latent codec -> decoded terrain -> deterministic rollout.",
            "train_log": codec_log,
            "cost": codec_cost,
            "metrics_by_move": evaluate_baseline_by_move(
                tests,
                config,
                evaluator=lambda examples: evaluate_latent_rollout(codec, examples, config, device=device, mode="codec"),  # type: ignore[arg-type]
            ),
        },
        "direct_rule_features": {
            "description": "Raw terrain tensor is used directly by the same deterministic rollout; no latent codec.",
            "cost": {"train_seconds": 0.0, "train_steps": 0, "trainable_parameters": 0},
            "metrics_by_move": evaluate_baseline_by_move(
                tests,
                config,
                evaluator=lambda examples: evaluate_latent_rollout(None, examples, config, device=device, mode="ground_truth"),  # type: ignore[arg-type]
            ),
        },
        "learned_effective_move": {
            "description": "Learns terrain_type + move -> effective move, then uses deterministic coordinate transitions.",
            "train_log": effective_log["history"],
            "cost": effective_log["cost"],
            "metrics_by_move": evaluate_baseline_by_move(
                tests,
                config,
                evaluator=lambda examples: evaluate_rollout_function(
                    examples,
                    config,
                    device=device,
                    rollout_fn=lambda terrain, start, moves: rollout_effective_move_model(
                        effective_model, terrain, start, moves, device=device
                    ),
                ),
            ),
        },
        "learned_transition_table": {
            "description": "Learns current coord + terrain_type + move -> next coord, then rolls out.",
            "train_log": transition_log["history"],
            "cost": transition_log["cost"],
            "metrics_by_move": evaluate_baseline_by_move(
                tests,
                config,
                evaluator=lambda examples: evaluate_rollout_function(
                    examples,
                    config,
                    device=device,
                    rollout_fn=lambda terrain, start, moves: rollout_transition_table_model(
                        transition_model, terrain, start, moves, device=device
                    ),
                ),
            ),
        },
        "raw_step_direct": {
            "description": "End-to-end neural sequence model sees raw terrain and moves, trained with every intermediate state.",
            "train_log": raw_step_log["history"],
            "cost": raw_step_log["cost"],
            "metrics_by_move": evaluate_baseline_by_move(
                tests,
                config,
                evaluator=lambda examples: evaluate_raw_step_direct(raw_step_model, examples, config, device=device),
            ),
        },
    }

    output = {
        "experiment": "heterogeneous_baseline_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": asdict(config),
        "results": results,
        "summary": summarize_single(results, config.eval_move_counts),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def summarize_sweep(runs: list[dict[str, object]], eval_move_counts: tuple[int, ...]) -> dict[str, object]:
    baseline_names = list(runs[0]["summary"]["overall_final_exact"].keys())  # type: ignore[index]
    by_move: dict[str, object] = {}
    for move_count in eval_move_counts:
        move_key = str(move_count)
        by_move[move_key] = {
            name: stat(
                [float(run["summary"]["final_exact_by_move"][move_key][name]) for run in runs]  # type: ignore[index]
            )
            for name in baseline_names
        }
    return {
        "count": len(runs),
        "final_exact_by_move": by_move,
        "overall_final_exact": {
            name: stat([float(run["summary"]["overall_final_exact"][name]) for run in runs])  # type: ignore[index]
            for name in baseline_names
        },
        "cost": {
            name: {
                "train_seconds": stat([float(run["summary"]["cost"][name]["train_seconds"]) for run in runs]),  # type: ignore[index]
                "train_steps": stat([float(run["summary"]["cost"][name]["train_steps"]) for run in runs]),  # type: ignore[index]
                "trainable_parameters": stat(
                    [float(run["summary"]["cost"][name]["trainable_parameters"]) for run in runs]  # type: ignore[index]
                ),
            }
            for name in baseline_names
        },
    }


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    eval_move_counts = tuple(int(item) for item in args.eval_move_counts.split(",") if item.strip())
    output_dir = Path(args.output_dir)
    runs: list[dict[str, object]] = []
    started = time.perf_counter()
    for seed in seeds:
        config = BaselineConfig(
            train_move_count=args.train_move_count,
            eval_move_counts=eval_move_counts,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
            seed=seed,
            d_latent=args.d_latent,
            d_model=args.d_model,
            codec_steps=args.codec_steps,
            effective_steps=args.effective_steps,
            transition_steps=args.transition_steps,
            raw_steps=args.raw_steps,
            eval_every=args.eval_every,
            lr=args.lr,
            table_lr=args.table_lr,
        )
        output_path = output_dir / f"train{config.train_move_count}_seed{seed}.json"
        print(f"=== baseline sweep seed={seed} train_move={config.train_move_count} ===", flush=True)
        result = run_experiment(config, output_path)
        runs.append({"seed": seed, "output": str(output_path), "summary": result["summary"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregate = {
        "experiment": "heterogeneous_baseline_comparison_sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": runs,
        "summary": summarize_sweep(runs, eval_move_counts),
        "seconds": round(time.perf_counter() - started, 2),
    }
    aggregate_path = Path(args.aggregate)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/heterogeneous_baseline_comparison/result.json")
    parser.add_argument("--aggregate", default="artifacts/heterogeneous_baseline_comparison/sweep_results.json")
    parser.add_argument("--output-dir", default="artifacts/heterogeneous_baseline_comparison/sweep_runs")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--train-move-count", type=int, default=8)
    parser.add_argument("--eval-move-counts", default="8,16,32")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-latent", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--codec-steps", type=int, default=300)
    parser.add_argument("--effective-steps", type=int, default=120)
    parser.add_argument("--transition-steps", type=int, default=180)
    parser.add_argument("--raw-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--table-lr", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep:
        result = run_sweep(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        return
    config = BaselineConfig(
        train_move_count=args.train_move_count,
        eval_move_counts=tuple(int(item) for item in args.eval_move_counts.split(",") if item.strip()),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
        d_latent=args.d_latent,
        d_model=args.d_model,
        codec_steps=args.codec_steps,
        effective_steps=args.effective_steps,
        transition_steps=args.transition_steps,
        raw_steps=args.raw_steps,
        eval_every=args.eval_every,
        lr=args.lr,
        table_lr=args.table_lr,
    )
    result = run_experiment(config, Path(args.output))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
