from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from yggdrasil_v2.reasoning_medium.a1_10_assessment import assess_a110
from yggdrasil_v2.reasoning_medium.a1_10_cache import audit_a110_cache, cache_a110_full_hidden
from yggdrasil_v2.reasoning_medium.a1_10_interventions import (
    cache_a110_interventions,
    run_a110_interventions,
)
from yggdrasil_v2.reasoning_medium.a1_10_train import (
    A110TrainSpec,
    benchmark_a110_cached_costs,
    evaluate_a110_formal,
    train_a110_reasoner,
)


def _write(payload: dict[str, Any], output: Path | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="V2-A1.10 full-text anonymous workspace experiment")
    sub = root.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache")
    cache.add_argument("--data-dir", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--max-length", type=int, default=1024)
    cache.add_argument("--inference-batch-size", type=int, default=16)
    cache.add_argument("--shard-size", type=int, default=64)
    cache.add_argument("--overfit32", action="store_true")

    audit = sub.add_parser("audit-cache")
    audit.add_argument("--cache-dir", type=Path, required=True)
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    train = sub.add_parser("train")
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model-seed", type=int, required=True)
    train.add_argument("--data-seed", type=int, required=True)
    train.add_argument("--steps", type=int, default=4000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--validation-interval", type=int, default=200)
    train.add_argument("--early-stop-patience", type=int, default=6)
    train.add_argument("--device", default="cuda")
    train.add_argument("--overfit32", action="store_true")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=32)

    cache_interventions = sub.add_parser("cache-interventions")
    cache_interventions.add_argument("--data-dir", type=Path, required=True)
    cache_interventions.add_argument("--formal-eval", type=Path, required=True)
    cache_interventions.add_argument("--output-dir", type=Path, required=True)
    cache_interventions.add_argument("--device", default="cuda")
    cache_interventions.add_argument("--max-length", type=int, default=1024)
    cache_interventions.add_argument("--inference-batch-size", type=int, default=16)
    cache_interventions.add_argument("--shard-size", type=int, default=64)
    cache_interventions.add_argument("--seed", type=int, default=20261020)

    intervene = sub.add_parser("intervene")
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--source-cache-dir", type=Path, required=True)
    intervene.add_argument("--intervention-cache-dir", type=Path, required=True)
    intervene.add_argument("--data-dir", type=Path, required=True)
    intervene.add_argument("--formal-eval", type=Path, required=True)
    intervene.add_argument("--output", type=Path, required=True)
    intervene.add_argument("--device", default="cuda")
    intervene.add_argument("--batch-size", type=int, default=32)
    intervene.add_argument("--seed", type=int, default=20261020)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--cache-dir", type=Path, required=True)
    benchmark.add_argument("--data-dir", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--batch-size", type=int, default=32)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--repeats", type=int, default=5)

    assess = sub.add_parser("assess")
    assess.add_argument("--run-dir", type=Path, action="append", required=True)
    assess.add_argument("--overfit-results", type=Path, required=True)
    assess.add_argument("--cost", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "cache":
        payload = cache_a110_full_hidden(
            args.data_dir,
            args.output_dir,
            device=args.device,
            max_length=args.max_length,
            inference_batch_size=args.inference_batch_size,
            shard_size=args.shard_size,
            overfit32=args.overfit32,
        )
        _write(payload)
    elif args.command == "audit-cache":
        _write(audit_a110_cache(args.cache_dir, args.data_dir), args.output)
    elif args.command == "train":
        payload = train_a110_reasoner(
            args.cache_dir,
            args.data_dir,
            args.output_dir,
            spec=A110TrainSpec(
                model_seed=args.model_seed,
                data_seed=args.data_seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                validation_interval=args.validation_interval,
                early_stop_patience=args.early_stop_patience,
            ),
            device=args.device,
            overfit_mode=args.overfit32,
        )
        _write(payload)
    elif args.command == "evaluate":
        _write(
            evaluate_a110_formal(
                args.checkpoint,
                args.cache_dir,
                args.data_dir,
                args.device,
                batch_size=args.batch_size,
            ),
            args.output,
        )
    elif args.command == "cache-interventions":
        _write(
            cache_a110_interventions(
                args.data_dir,
                args.formal_eval,
                args.output_dir,
                device=args.device,
                max_length=args.max_length,
                inference_batch_size=args.inference_batch_size,
                shard_size=args.shard_size,
                seed=args.seed,
            )
        )
    elif args.command == "intervene":
        _write(
            run_a110_interventions(
                args.checkpoint,
                args.source_cache_dir,
                args.intervention_cache_dir,
                args.data_dir,
                args.formal_eval,
                args.output,
                device=args.device,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        )
    elif args.command == "benchmark":
        _write(
            benchmark_a110_cached_costs(
                args.checkpoint,
                args.cache_dir,
                args.data_dir,
                args.device,
                batch_size=args.batch_size,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            args.output,
        )
    elif args.command == "assess":
        _write(assess_a110(args.run_dir, args.overfit_results, args.cost, args.output))


if __name__ == "__main__":
    main()
