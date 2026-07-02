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
CELL_COUNT = GRID_SIZE * GRID_SIZE
TERRAIN_TYPES = 4
MOVES = 4
MOVE_DELTAS = {
    0: (0, -1),  # U
    1: (0, 1),  # D
    2: (-1, 0),  # L
    3: (1, 0),  # R
}


@dataclass(frozen=True)
class HeteroConfig:
    move_count: int = 12
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 2048
    batch_size: int = 256
    seed: int = 20260701
    d_latent: int = 32
    d_model: int = 128
    codec_steps: int = 300
    raw_steps: int = 2000
    eval_every: int = 500
    lr: float = 1e-3


@dataclass(frozen=True)
class HeteroExample:
    terrain: tuple[int, ...]
    start: int
    moves: tuple[int, ...]
    states: tuple[int, ...]


def coord_to_xy(coord: int) -> tuple[int, int]:
    return coord // GRID_SIZE, coord % GRID_SIZE


def xy_to_coord(x: int, y: int) -> int:
    return x * GRID_SIZE + y


def transform_move(move: int, terrain_type: int) -> int:
    if terrain_type == 0:
        return move
    if terrain_type == 1:
        return {0: 3, 3: 1, 1: 2, 2: 0}[move]
    if terrain_type == 2:
        return {0: 2, 2: 1, 1: 3, 3: 0}[move]
    if terrain_type == 3:
        return {0: 1, 1: 0, 2: 3, 3: 2}[move]
    raise ValueError(f"unknown terrain type: {terrain_type}")


def step_coord(coord: int, move: int, terrain: tuple[int, ...]) -> int:
    effective = transform_move(move, terrain[coord])
    x, y = coord_to_xy(coord)
    dx, dy = MOVE_DELTAS[effective]
    return xy_to_coord((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)


def rollout(terrain: tuple[int, ...], start: int, moves: tuple[int, ...]) -> tuple[int, ...]:
    coord = start
    states: list[int] = []
    for move in moves:
        coord = step_coord(coord, move, terrain)
        states.append(coord)
    return tuple(states)


def generate_examples(count: int, *, seed: int, move_count: int) -> list[HeteroExample]:
    rng = random.Random(seed)
    examples: list[HeteroExample] = []
    for _ in range(count):
        terrain = tuple(rng.randrange(TERRAIN_TYPES) for _ in range(CELL_COUNT))
        start = rng.randrange(CELL_COUNT)
        moves = tuple(rng.randrange(MOVES) for _ in range(move_count))
        examples.append(HeteroExample(terrain=terrain, start=start, moves=moves, states=rollout(terrain, start, moves)))
    return examples


def split_examples(config: HeteroConfig) -> tuple[list[HeteroExample], list[HeteroExample], list[HeteroExample]]:
    return (
        generate_examples(config.train_size, seed=config.seed, move_count=config.move_count),
        generate_examples(config.val_size, seed=config.seed + 1, move_count=config.move_count),
        generate_examples(config.test_size, seed=config.seed + 2, move_count=config.move_count),
    )


def batch_examples(
    examples: list[HeteroExample],
    *,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
    terrain = torch.tensor([example.terrain for example in batch], dtype=torch.long, device=device)
    start = torch.tensor([example.start for example in batch], dtype=torch.long, device=device)
    moves = torch.tensor([example.moves for example in batch], dtype=torch.long, device=device)
    states = torch.tensor([example.states for example in batch], dtype=torch.long, device=device)
    return terrain, start, moves, states


def examples_to_tensors(
    examples: list[HeteroExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("examples must not be empty")
    terrain = torch.tensor([example.terrain for example in examples], dtype=torch.long, device=device)
    start = torch.tensor([example.start for example in examples], dtype=torch.long, device=device)
    moves = torch.tensor([example.moves for example in examples], dtype=torch.long, device=device)
    states = torch.tensor([example.states for example in examples], dtype=torch.long, device=device)
    return terrain, start, moves, states


class TerrainCodec(nn.Module):
    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(TERRAIN_TYPES, d_latent),
            nn.Tanh(),
            nn.Linear(d_latent, d_latent),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, TERRAIN_TYPES),
        )

    def encode(self, terrain_one_hot: torch.Tensor) -> torch.Tensor:
        return self.encoder(terrain_one_hot)

    def decode(self, terrain_latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(terrain_latent)


class RawHeteroDirectModel(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.terrain_embedding = nn.Embedding(TERRAIN_TYPES, d_model // 4)
        self.coord_embedding = nn.Embedding(CELL_COUNT, d_model)
        self.move_embedding = nn.Embedding(MOVES, d_model)
        self.map_projection = nn.Sequential(
            nn.Linear(CELL_COUNT * (d_model // 4), d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.gru = nn.GRU(d_model * 2, d_model, batch_first=True)
        self.head = nn.Linear(d_model, CELL_COUNT)

    def forward(self, terrain: torch.Tensor, start: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        map_flat = self.terrain_embedding(terrain).reshape(terrain.size(0), -1)
        map_context = self.map_projection(map_flat)
        h0 = (map_context + self.coord_embedding(start)).unsqueeze(0)
        move_context = self.move_embedding(moves)
        repeated_map = map_context.unsqueeze(1).expand(-1, moves.size(1), -1)
        hidden, _ = self.gru(torch.cat([move_context, repeated_map], dim=-1), h0)
        return self.head(hidden[:, -1, :])


def terrain_one_hot(terrain: torch.Tensor) -> torch.Tensor:
    return F.one_hot(terrain, num_classes=TERRAIN_TYPES).float()


def train_codec(codec: TerrainCodec, config: HeteroConfig, *, device: torch.device) -> dict[str, object]:
    optimizer = torch.optim.AdamW(codec.parameters(), lr=config.lr)
    labels = torch.arange(TERRAIN_TYPES, dtype=torch.long, device=device)
    one_hot = F.one_hot(labels, num_classes=TERRAIN_TYPES).float()
    history: list[dict[str, float | int]] = []
    for step in range(1, config.codec_steps + 1):
        logits = codec.decode(codec.encode(one_hot))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.codec_steps:
            exact = float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu())
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "terrain_reconstruction_exact": exact})
            print(f"codec step={step:4d} loss={float(loss.detach().cpu()):.4f} recon={exact:.3f}", flush=True)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad = False
    return {"history": history}


def train_raw_direct(
    model: RawHeteroDirectModel,
    train: list[HeteroExample],
    val: list[HeteroExample],
    config: HeteroConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    for step in range(1, config.raw_steps + 1):
        terrain, start, moves, states = batch_examples(train, rng=rng, batch_size=config.batch_size, device=device)
        logits = model(terrain, start, moves)
        loss = F.cross_entropy(logits, states[:, -1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.raw_steps:
            metrics = evaluate_raw_direct(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(f"raw_direct step={step:4d} loss={float(loss.detach().cpu()):.4f} final={metrics['final_exact']:.3f}", flush=True)
    return {"history": history}


def effective_move_table(device: torch.device) -> torch.Tensor:
    table = torch.empty(TERRAIN_TYPES, MOVES, dtype=torch.long, device=device)
    for terrain_type in range(TERRAIN_TYPES):
        for move in range(MOVES):
            table[terrain_type, move] = transform_move(move, terrain_type)
    return table


def transition_table(device: torch.device) -> torch.Tensor:
    table = torch.empty(CELL_COUNT, MOVES, dtype=torch.long, device=device)
    for coord in range(CELL_COUNT):
        x, y = coord_to_xy(coord)
        for move, (dx, dy) in MOVE_DELTAS.items():
            table[coord, move] = xy_to_coord((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)
    return table


@torch.no_grad()
def latent_rollout(
    codec: TerrainCodec | None,
    terrain: torch.Tensor,
    start: torch.Tensor,
    moves: torch.Tensor,
    *,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    if mode == "no_hetero":
        decoded_terrain = torch.zeros_like(terrain)
    elif mode == "ground_truth":
        decoded_terrain = terrain
    else:
        assert codec is not None
        one_hot = terrain_one_hot(terrain)
        terrain_latent = codec.encode(one_hot)
        decoded_terrain = codec.decode(terrain_latent).argmax(dim=-1)

    effective = effective_move_table(device)
    transitions = transition_table(device)
    coord = start
    states: list[torch.Tensor] = []
    batch_ids = torch.arange(terrain.size(0), device=device)
    for index in range(moves.size(1)):
        current_terrain = decoded_terrain[batch_ids, coord]
        effective_moves = effective[current_terrain, moves[:, index]]
        coord = transitions[coord, effective_moves]
        states.append(coord)
    return torch.stack(states, dim=1)


@torch.no_grad()
def evaluate_latent_rollout(
    codec: TerrainCodec | None,
    examples: list[HeteroExample],
    config: HeteroConfig,
    *,
    device: torch.device,
    mode: str,
) -> dict[str, float]:
    correct_final = 0
    correct_step = 0
    total_step = 0
    for start_index in range(0, len(examples), config.batch_size):
        batch = examples[start_index : start_index + config.batch_size]
        terrain, start, moves, states = examples_to_tensors(batch, device=device)
        predicted = latent_rollout(codec, terrain, start, moves, device=device, mode=mode)
        correct_final += int((predicted[:, -1] == states[:, -1]).sum().detach().cpu())
        correct_step += int((predicted == states).sum().detach().cpu())
        total_step += states.numel()
    return {
        "final_exact": correct_final / len(examples),
        "step_exact": correct_step / total_step,
    }


@torch.no_grad()
def evaluate_raw_direct(
    model: RawHeteroDirectModel,
    examples: list[HeteroExample],
    config: HeteroConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    for start_index in range(0, len(examples), config.batch_size):
        batch = examples[start_index : start_index + config.batch_size]
        terrain, start, moves, states = examples_to_tensors(batch, device=device)
        logits = model(terrain, start, moves)
        correct += int((logits.argmax(dim=-1) == states[:, -1]).sum().detach().cpu())
    model.train()
    return {"final_exact": correct / len(examples)}


def random_codec(config: HeteroConfig, *, device: torch.device, seed: int) -> TerrainCodec:
    torch.manual_seed(seed)
    codec = TerrainCodec(config.d_latent).to(device)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad = False
    return codec


def run_experiment(config: HeteroConfig, output_path: Path) -> dict[str, object]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = split_examples(config)
    print(f"hetero device={device} move_count={config.move_count} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    codec = TerrainCodec(config.d_latent).to(device)
    codec_log = train_codec(codec, config, device=device)
    raw_model = RawHeteroDirectModel(config.d_model).to(device)
    raw_log = train_raw_direct(raw_model, train, val, config, seed=config.seed + 10, device=device)
    bad_codec = random_codec(config, device=device, seed=config.seed + 20)

    results = {
        "text_only_no_hetero": {
            "description": "Text moves only; all terrain assumed normal.",
            "test_metrics": evaluate_latent_rollout(None, test, config, device=device, mode="no_hetero"),
        },
        "latent_good_codec": {
            "description": "Terrain tensor -> trained latent codec -> decoded terrain -> latent rollout -> text coordinate output.",
            "train_log": codec_log,
            "test_metrics": evaluate_latent_rollout(codec, test, config, device=device, mode="codec"),
        },
        "latent_bad_codec": {
            "description": "Terrain tensor -> random frozen bad codec -> decoded terrain -> rollout.",
            "test_metrics": evaluate_latent_rollout(bad_codec, test, config, device=device, mode="codec"),
        },
        "ground_truth_hetero_oracle": {
            "description": "Oracle rollout using ground-truth terrain, upper bound for the task.",
            "test_metrics": evaluate_latent_rollout(None, test, config, device=device, mode="ground_truth"),
        },
        "raw_hetero_direct": {
            "description": "End-to-end model sees raw terrain tensor and text moves, no explicit latent codec.",
            "train_log": raw_log,
            "test_metrics": evaluate_raw_direct(raw_model, test, config, device=device),
        },
    }
    output = {
        "experiment": "heterogeneous_latent_input",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": config.__dict__,
        "results": results,
        "summary": summarize_single(results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def metric(results: dict[str, object], key: str, name: str = "final_exact") -> float:
    return float(results[key]["test_metrics"][name])  # type: ignore[index]


def summarize_single(results: dict[str, object]) -> dict[str, object]:
    return {
        "final_exact": {key: metric(results, key) for key in results},
        "step_exact": {
            key: metric(results, key, "step_exact")
            for key in results
            if "step_exact" in results[key]["test_metrics"]  # type: ignore[index]
        },
        "good_codec_minus_text_only": metric(results, "latent_good_codec") - metric(results, "text_only_no_hetero"),
        "good_codec_minus_bad_codec": metric(results, "latent_good_codec") - metric(results, "latent_bad_codec"),
        "good_codec_minus_raw_direct": metric(results, "latent_good_codec") - metric(results, "raw_hetero_direct"),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    move_counts = [int(item) for item in args.move_counts.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    runs: list[dict[str, object]] = []
    started = time.perf_counter()
    for move_count in move_counts:
        for seed in seeds:
            config = HeteroConfig(
                move_count=move_count,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                batch_size=args.batch_size,
                seed=seed,
                d_latent=args.d_latent,
                d_model=args.d_model,
                codec_steps=args.codec_steps,
                raw_steps=args.raw_steps,
                eval_every=args.eval_every,
                lr=args.lr,
            )
            output_path = output_dir / f"move{move_count}_seed{seed}.json"
            print(f"=== hetero sweep move={move_count} seed={seed} ===", flush=True)
            result = run_experiment(config, output_path)
            runs.append(
                {
                    "move_count": move_count,
                    "seed": seed,
                    "output": str(output_path),
                    "summary": result["summary"],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    aggregate = {
        "experiment": "heterogeneous_latent_input_sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": runs,
        "summary": summarize_sweep(runs),
        "seconds": round(time.perf_counter() - started, 2),
    }
    aggregate_path = Path(args.aggregate)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def summarize_sweep(runs: list[dict[str, object]]) -> dict[str, object]:
    keys = [
        "text_only_no_hetero",
        "latent_good_codec",
        "latent_bad_codec",
        "ground_truth_hetero_oracle",
        "raw_hetero_direct",
    ]
    deltas = ["good_codec_minus_text_only", "good_codec_minus_bad_codec", "good_codec_minus_raw_direct"]

    def stat(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    def group(selected: list[dict[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {"count": len(selected)}
        for key in keys:
            result[key] = stat([float(run["summary"]["final_exact"][key]) for run in selected])  # type: ignore[index]
        for key in deltas:
            result[key] = stat([float(run["summary"][key]) for run in selected])  # type: ignore[index]
        result["latent_good_codec_step"] = stat(
            [float(run["summary"]["step_exact"]["latent_good_codec"]) for run in selected]  # type: ignore[index]
        )
        return result

    summary: dict[str, object] = {}
    for move_count in sorted({int(run["move_count"]) for run in runs}):
        summary[str(move_count)] = group([run for run in runs if int(run["move_count"]) == move_count])
    summary["overall"] = group(runs)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/heterogeneous_latent_input/result.json")
    parser.add_argument("--aggregate", default="artifacts/heterogeneous_latent_input/sweep_results.json")
    parser.add_argument("--output-dir", default="artifacts/heterogeneous_latent_input/sweep_runs")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--move-count", type=int, default=12)
    parser.add_argument("--move-counts", default="8,12,16")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-latent", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--codec-steps", type=int, default=300)
    parser.add_argument("--raw-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep:
        result = run_sweep(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        return
    config = HeteroConfig(
        move_count=args.move_count,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
        d_latent=args.d_latent,
        d_model=args.d_model,
        codec_steps=args.codec_steps,
        raw_steps=args.raw_steps,
        eval_every=args.eval_every,
        lr=args.lr,
    )
    result = run_experiment(config, Path(args.output))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
