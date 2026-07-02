from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import time

import torch

from heterogeneous_latent_input import GRID_SIZE, TERRAIN_TYPES
from visual_multimodal_stage_ab import TILE_SIZE, write_png
from visual_multimodal_stage_c import (
    StageCExample,
    generate_examples,
    parse_symbols_by_palette,
    render_symbol_images,
)


@dataclass(frozen=True)
class StageDConfig:
    eval_move_counts: tuple[int, ...] = (8, 16, 32)
    test_size: int = 4096
    seed: int = 20260701
    probe_budgets: tuple[int, ...] = (0, 1, 2, 3, 4)


def stat(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


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


def step_coord(coord: int, move: int, terrain_type: int) -> int:
    effective = transform_move(move, terrain_type)
    x, y = coord_to_xy(coord)
    deltas = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
    }
    dx, dy = deltas[effective]
    return xy_to_coord((x + dx) % GRID_SIZE, (y + dy) % GRID_SIZE)


def parse_symbols_for_example(example: StageCExample, *, device: torch.device) -> tuple[int, ...]:
    visual_symbols = torch.tensor([example.visual_symbols], dtype=torch.long, device=device)
    image = render_symbol_images(visual_symbols, device=device)
    parsed = parse_symbols_by_palette(image, device=device)[0].detach().cpu().tolist()
    return tuple(int(item) for item in parsed)


def run_online_policy(
    example: StageCExample,
    *,
    parsed_symbols: tuple[int, ...],
    policy: str,
    probe_budget: int = 0,
) -> dict[str, float | int | list[int]]:
    coord = example.start
    predicted_states: list[int] = []
    true_states = list(example.states)
    legend: dict[int, int] = {}
    probes = 0
    semantic_correct = 0
    frames = 0
    symbols_seen: set[int] = set()

    for move in example.moves:
        frames += 1
        symbol = parsed_symbols[coord]
        symbols_seen.add(symbol)
        true_semantic = example.symbol_to_semantic[symbol]

        if policy == "oracle_semantic":
            semantic = true_semantic
        elif policy == "passive_symbol_identity":
            semantic = symbol
        elif policy == "probe_every_step_no_memory":
            probes += 1
            semantic = true_semantic
        elif policy == "legend_memory":
            if symbol not in legend and probes < probe_budget:
                probes += 1
                legend[symbol] = true_semantic
            semantic = legend.get(symbol, symbol)
        else:
            raise ValueError(f"unknown policy: {policy}")

        if semantic == true_semantic:
            semantic_correct += 1
        coord = step_coord(coord, move, semantic)
        predicted_states.append(coord)

    final_exact = 1.0 if predicted_states[-1] == true_states[-1] else 0.0
    step_exact = sum(1 for predicted, true in zip(predicted_states, true_states) if predicted == true) / len(true_states)
    semantic_decision_exact = semantic_correct / len(example.moves)
    return {
        "final_exact": final_exact,
        "step_exact": step_exact,
        "semantic_decision_exact": semantic_decision_exact,
        "probes": probes,
        "frames": frames,
        "known_symbols_end": len(legend) if policy == "legend_memory" else (TERRAIN_TYPES if policy == "oracle_semantic" else 0),
        "symbols_seen": len(symbols_seen),
        "predicted_final": predicted_states[-1],
        "true_final": true_states[-1],
    }


def aggregate(items: list[dict[str, float | int | list[int]]]) -> dict[str, float]:
    return {
        "final_exact": statistics.fmean(float(item["final_exact"]) for item in items),
        "step_exact": statistics.fmean(float(item["step_exact"]) for item in items),
        "semantic_decision_exact": statistics.fmean(float(item["semantic_decision_exact"]) for item in items),
        "avg_probes": statistics.fmean(float(item["probes"]) for item in items),
        "avg_frames": statistics.fmean(float(item["frames"]) for item in items),
        "avg_known_symbols_end": statistics.fmean(float(item["known_symbols_end"]) for item in items),
        "avg_symbols_seen": statistics.fmean(float(item["symbols_seen"]) for item in items),
    }


def evaluate_examples(examples: list[StageCExample], *, device: torch.device, probe_budgets: tuple[int, ...]) -> dict[str, dict[str, float]]:
    parsed = [parse_symbols_for_example(example, device=device) for example in examples]
    results: dict[str, list[dict[str, float | int | list[int]]]] = {
        "oracle_semantic_full_state": [],
        "local_passive_no_probe": [],
        "local_probe_every_step_no_memory": [],
    }
    for budget in probe_budgets:
        results[f"local_legend_memory_budget_{budget}"] = []

    for example, parsed_symbols in zip(examples, parsed):
        results["oracle_semantic_full_state"].append(
            run_online_policy(example, parsed_symbols=parsed_symbols, policy="oracle_semantic")
        )
        results["local_passive_no_probe"].append(
            run_online_policy(example, parsed_symbols=parsed_symbols, policy="passive_symbol_identity")
        )
        results["local_probe_every_step_no_memory"].append(
            run_online_policy(example, parsed_symbols=parsed_symbols, policy="probe_every_step_no_memory")
        )
        for budget in probe_budgets:
            results[f"local_legend_memory_budget_{budget}"].append(
                run_online_policy(example, parsed_symbols=parsed_symbols, policy="legend_memory", probe_budget=budget)
            )
    return {name: aggregate(items) for name, items in results.items()}


def sample_frame_strip(example: StageCExample, *, device: torch.device) -> torch.Tensor:
    visual_symbols = torch.tensor([example.visual_symbols], dtype=torch.long, device=device)
    full_image = render_symbol_images(visual_symbols, device=device)[0]
    coords = [example.start, *example.states[: min(len(example.states), 15)]]
    tiles: list[torch.Tensor] = []
    for coord in coords:
        x, y = coord_to_xy(coord)
        tiles.append(full_image[:, x * TILE_SIZE : (x + 1) * TILE_SIZE, y * TILE_SIZE : (y + 1) * TILE_SIZE])
    return torch.cat(tiles, dim=2)


def write_samples(examples: list[StageCExample], *, output_dir: Path, device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    first = examples[0]
    full = render_symbol_images(torch.tensor([first.visual_symbols], dtype=torch.long, device=device), device=device)[0]
    write_png(output_dir / "debug_hidden_full_map.png", full)
    write_png(output_dir / "local_frame_strip.png", sample_frame_strip(first, device=device))


def run_experiment(config: StageDConfig, output_path: Path) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_d device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    tests = {
        move_count: generate_examples(config.test_size, seed=config.seed + 1000 + move_count, move_count=move_count)
        for move_count in config.eval_move_counts
    }
    write_samples(tests[config.eval_move_counts[0]], output_dir=output_path.parent / "samples" / f"seed{config.seed}", device=device)
    started = time.perf_counter()
    metrics_by_move = {
        str(move_count): evaluate_examples(examples, device=device, probe_budgets=config.probe_budgets)
        for move_count, examples in tests.items()
    }
    output = {
        "experiment": "visual_multimodal_stage_d",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": asdict(config),
        "metrics_by_move": metrics_by_move,
        "summary": summarize_single(metrics_by_move),
        "seconds": round(time.perf_counter() - started, 2),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def summarize_single(metrics_by_move: dict[str, dict[str, dict[str, float]]]) -> dict[str, object]:
    names = list(next(iter(metrics_by_move.values())).keys())
    return {
        "overall": {
            name: {
                metric: statistics.fmean(metrics_by_move[move][name][metric] for move in metrics_by_move)
                for metric in metrics_by_move[next(iter(metrics_by_move))][name]
            }
            for name in names
        },
        "by_move": metrics_by_move,
    }


def summarize_sweep(runs: list[dict[str, object]], eval_move_counts: tuple[int, ...]) -> dict[str, object]:
    names = list(runs[0]["summary"]["overall"].keys())  # type: ignore[index]
    metric_names = list(runs[0]["summary"]["overall"][names[0]].keys())  # type: ignore[index]
    return {
        "count": len(runs),
        "overall": {
            name: {
                metric: stat([float(run["summary"]["overall"][name][metric]) for run in runs])  # type: ignore[index]
                for metric in metric_names
            }
            for name in names
        },
        "by_move": {
            str(move_count): {
                name: {
                    metric: stat(
                        [
                            float(run["summary"]["by_move"][str(move_count)][name][metric])  # type: ignore[index]
                            for run in runs
                        ]
                    )
                    for metric in metric_names
                }
                for name in names
            }
            for move_count in eval_move_counts
        },
    }


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    eval_move_counts = tuple(int(item) for item in args.eval_move_counts.split(",") if item.strip())
    probe_budgets = tuple(int(item) for item in args.probe_budgets.split(",") if item.strip())
    output_dir = Path(args.output_dir)
    started = time.perf_counter()
    runs: list[dict[str, object]] = []
    for seed in seeds:
        config = StageDConfig(
            eval_move_counts=eval_move_counts,
            test_size=args.test_size,
            seed=seed,
            probe_budgets=probe_budgets,
        )
        output_path = output_dir / f"seed{seed}.json"
        print(f"=== stage D sweep seed={seed} ===", flush=True)
        result = run_experiment(config, output_path)
        runs.append({"seed": seed, "output": str(output_path), "summary": result["summary"]})
    aggregate = {
        "experiment": "visual_multimodal_stage_d_sweep",
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
    parser.add_argument("--output", default="artifacts/visual_multimodal_stage_d/result.json")
    parser.add_argument("--aggregate", default="artifacts/visual_multimodal_stage_d/sweep_results.json")
    parser.add_argument("--output-dir", default="artifacts/visual_multimodal_stage_d/sweep_runs")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--eval-move-counts", default="8,16,32")
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--probe-budgets", default="0,1,2,3,4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep:
        result = run_sweep(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        return
    config = StageDConfig(
        eval_move_counts=tuple(int(item) for item in args.eval_move_counts.split(",") if item.strip()),
        test_size=args.test_size,
        seed=args.seed,
        probe_budgets=tuple(int(item) for item in args.probe_budgets.split(",") if item.strip()),
    )
    result = run_experiment(config, Path(args.output))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
