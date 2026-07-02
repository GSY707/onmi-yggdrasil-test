from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from text_to_latent_thought import TrainConfig, run_experiment


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_sweep(
    *,
    move_counts: list[int],
    seeds: list[int],
    output_dir: Path,
    aggregate_path: Path,
    base_config: TrainConfig,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    started = time.perf_counter()
    for move_count in move_counts:
        for seed in seeds:
            run_name = f"move{move_count}_seed{seed}"
            output_path = output_dir / f"{run_name}.json"
            config = TrainConfig(
                move_count=move_count,
                seed=seed,
                train_size=base_config.train_size,
                val_size=base_config.val_size,
                test_size=base_config.test_size,
                batch_size=base_config.batch_size,
                text_steps=base_config.text_steps,
                direct_steps=base_config.direct_steps,
                latent_steps=base_config.latent_steps,
                latent_scratch_steps=base_config.latent_scratch_steps,
                eval_every=base_config.eval_every,
                d_model=base_config.d_model,
                n_layers=base_config.n_layers,
                n_heads=base_config.n_heads,
                dropout=base_config.dropout,
                lr=base_config.lr,
            )
            print(f"=== sweep {run_name} ===", flush=True)
            result = run_experiment(config, output_path)
            metrics = result["test_metrics"]
            runs.append(
                {
                    "run": run_name,
                    "move_count": move_count,
                    "seed": seed,
                    "output": str(output_path),
                    "test_metrics": metrics,
                    "interpretation": result["interpretation"],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregate = {
        "experiment": "text_to_latent_thought_sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "move_counts": move_counts,
        "seeds": seeds,
        "base_config": base_config.__dict__,
        "runs": runs,
        "summary": summarize(runs),
        "seconds": round(time.perf_counter() - started, 2),
    }
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    by_move: dict[int, list[dict[str, object]]] = {}
    for run in runs:
        by_move.setdefault(int(run["move_count"]), []).append(run)

    result: dict[str, object] = {}
    for move_count, move_runs in sorted(by_move.items()):
        result[str(move_count)] = summarize_group(move_runs)
    result["overall"] = summarize_group(runs)
    return result


def summarize_group(runs: list[dict[str, object]]) -> dict[str, object]:
    metric_names = ["direct_scratch", "visible_text", "latent_from_text", "latent_scratch"]
    summary: dict[str, object] = {"count": len(runs)}
    for metric_name in metric_names:
        values = [
            float(run["test_metrics"][metric_name]["exact_accuracy"])  # type: ignore[index]
            for run in runs
        ]
        summary[metric_name] = {
            "mean_exact": statistics.fmean(values),
            "min_exact": min(values),
            "max_exact": max(values),
            "stdev_exact": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    latent_values = [
        float(run["test_metrics"]["latent_from_text"]["exact_accuracy"])  # type: ignore[index]
        for run in runs
    ]
    visible_values = [
        float(run["test_metrics"]["visible_text"]["exact_accuracy"])  # type: ignore[index]
        for run in runs
    ]
    scratch_values = [
        float(run["test_metrics"]["latent_scratch"]["exact_accuracy"])  # type: ignore[index]
        for run in runs
    ]
    direct_values = [
        float(run["test_metrics"]["direct_scratch"]["exact_accuracy"])  # type: ignore[index]
        for run in runs
    ]
    summary["latent_from_text_minus_scratch_mean"] = statistics.fmean(
        latent - scratch for latent, scratch in zip(latent_values, scratch_values)
    )
    summary["visible_minus_latent_mean"] = statistics.fmean(
        visible - latent for visible, latent in zip(visible_values, latent_values)
    )
    summary["direct_minus_latent_mean"] = statistics.fmean(
        direct - latent for direct, latent in zip(direct_values, latent_values)
    )
    summary["latent_transfer_helped_all_runs"] = all(
        latent > scratch for latent, scratch in zip(latent_values, scratch_values)
    )
    summary["latent_within_20_points_of_visible_all_runs"] = all(
        latent >= visible - 0.2 for visible, latent in zip(visible_values, latent_values)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--move-counts", default="6,8")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--output-dir", default="artifacts/text_to_latent_thought/sweep_runs")
    parser.add_argument("--aggregate", default="artifacts/text_to_latent_thought/sweep_results.json")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=768)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--text-steps", type=int, default=900)
    parser.add_argument("--direct-steps", type=int, default=900)
    parser.add_argument("--latent-steps", type=int, default=700)
    parser.add_argument("--latent-scratch-steps", type=int, default=700)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = TrainConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        text_steps=args.text_steps,
        direct_steps=args.direct_steps,
        latent_steps=args.latent_steps,
        latent_scratch_steps=args.latent_scratch_steps,
        eval_every=args.eval_every,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
    )
    aggregate = run_sweep(
        move_counts=parse_int_list(args.move_counts),
        seeds=parse_int_list(args.seeds),
        output_dir=Path(args.output_dir),
        aggregate_path=Path(args.aggregate),
        base_config=base_config,
    )
    print(json.dumps(aggregate["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
