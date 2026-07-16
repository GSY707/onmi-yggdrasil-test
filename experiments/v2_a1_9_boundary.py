from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_9_assessment import assess_a19
from yggdrasil_v2.reasoning_medium.a1_9_cache import audit_a19_cache, cache_a19_role_hidden
from yggdrasil_v2.reasoning_medium.a1_9_interventions import run_a19_interventions
from yggdrasil_v2.reasoning_medium.a1_9_train import (
    A19TrainSpec,
    benchmark_a19_cached_costs,
    evaluate_a19_formal,
    train_a19_boundary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-Yggdrasil V2-A1.9 frozen Qwen role boundary")
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache")
    cache.add_argument("--data-dir", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--max-length", type=int, default=1024)
    cache.add_argument("--inference-batch-size", type=int, default=8)
    cache.add_argument("--shard-size", type=int, default=64)
    cache.add_argument("--overfit32", action="store_true")

    audit = sub.add_parser("audit-cache")
    audit.add_argument("--cache-dir", type=Path, required=True)
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    train = sub.add_parser("train")
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--core-checkpoint", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--adapter-seed", type=int, required=True)
    train.add_argument("--data-seed", type=int, required=True)
    train.add_argument("--core-seed", type=int, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--steps", type=int, default=3000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--validation-interval", type=int, default=200)
    train.add_argument("--early-stop-patience", type=int, default=10)
    train.add_argument("--overfit-mode", action="store_true")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=128)

    intervene = sub.add_parser("intervene")
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--cache-dir", type=Path, required=True)
    intervene.add_argument("--data-dir", type=Path, required=True)
    intervene.add_argument("--formal-eval", type=Path, required=True)
    intervene.add_argument("--output", type=Path, required=True)
    intervene.add_argument("--device", default="cuda")
    intervene.add_argument("--batch-size", type=int, default=128)

    benchmark = sub.add_parser("benchmark-costs")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--cache-dir", type=Path, required=True)
    benchmark.add_argument("--data-dir", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--batch-size", type=int, default=128)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--repeats", type=int, default=10)

    assess = sub.add_parser("assess")
    assess.add_argument("--run-dir", type=Path, action="append", required=True)
    assess.add_argument("--overfit-results", type=Path, required=True)
    assess.add_argument("--cost", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    return parser


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "cache":
        payload = cache_a19_role_hidden(
            args.data_dir,
            args.output_dir,
            device=args.device,
            max_length=args.max_length,
            inference_batch_size=args.inference_batch_size,
            shard_size=args.shard_size,
            overfit32=args.overfit32,
        )
    elif args.command == "audit-cache":
        payload = audit_a19_cache(args.cache_dir, args.data_dir)
        _write(args.output, payload)
    elif args.command == "train":
        payload = train_a19_boundary(
            args.cache_dir,
            args.data_dir,
            args.core_checkpoint,
            args.output_dir,
            spec=A19TrainSpec(
                adapter_seed=args.adapter_seed,
                data_seed=args.data_seed,
                core_seed=args.core_seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                validation_interval=args.validation_interval,
                early_stop_patience=args.early_stop_patience,
            ),
            device=args.device,
            overfit_mode=args.overfit_mode,
        )
    elif args.command == "evaluate":
        payload = evaluate_a19_formal(
            args.checkpoint,
            args.cache_dir,
            args.data_dir,
            args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, payload)
    elif args.command == "intervene":
        payload = run_a19_interventions(
            args.checkpoint,
            args.cache_dir,
            args.data_dir,
            args.formal_eval,
            args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, payload)
    elif args.command == "benchmark-costs":
        payload = benchmark_a19_cached_costs(
            args.checkpoint,
            args.cache_dir,
            args.data_dir,
            args.device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _write(args.output, payload)
    elif args.command == "assess":
        payload = assess_a19(args.run_dir, args.overfit_results, args.cost, args.output)
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
