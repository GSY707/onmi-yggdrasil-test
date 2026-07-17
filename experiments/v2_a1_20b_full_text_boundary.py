from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_13_train import _write_json
from yggdrasil_v2.reasoning_medium.a1_20b_cache import (
    audit_a120b_cache,
    benchmark_a120b_cache_batches,
    cache_a120b_full_hidden,
)
from yggdrasil_v2.reasoning_medium.a1_20b_assessment import assess_a120b_failure
from yggdrasil_v2.reasoning_medium.a1_20b_diagnostics import (
    audit_a120b_gradient_credit,
    diagnose_a120b_boundary,
)
from yggdrasil_v2.reasoning_medium.a1_20b_train import (
    A120BTrainSpec,
    benchmark_a120b_training_pipeline,
    evaluate_a120b_formal,
    train_a120b_boundary,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="V2-A1.20B learned full-text Boundary")
    sub = root.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("cache")
    cache.add_argument("--data-dir", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--max-length", type=int, default=1024)
    cache.add_argument("--inference-batch-size", type=int, default=12)
    cache.add_argument("--shard-size", type=int, default=64)
    cache.add_argument("--overfit32", action="store_true")
    cache.add_argument("--split", action="append")

    cache_benchmark = sub.add_parser("cache-benchmark")
    cache_benchmark.add_argument("--data-dir", type=Path, required=True)
    cache_benchmark.add_argument("--output", type=Path, required=True)
    cache_benchmark.add_argument(
        "--batch-size", type=int, action="append", required=True
    )
    cache_benchmark.add_argument("--device", default="cuda")
    cache_benchmark.add_argument("--max-length", type=int, default=1024)

    audit = sub.add_parser("audit")
    audit.add_argument("--cache-dir", type=Path, required=True)
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    train = sub.add_parser("train")
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--core-checkpoint", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--reader-seed", type=int, required=True)
    train.add_argument("--data-seed", type=int, required=True)
    train.add_argument("--core-model-seed", type=int, required=True)
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--validation-interval", type=int, default=200)
    train.add_argument("--device", default="cuda")
    train.add_argument("--overfit", action="store_true")

    train_benchmark = sub.add_parser("train-benchmark")
    train_benchmark.add_argument("--cache-dir", type=Path, required=True)
    train_benchmark.add_argument("--data-dir", type=Path, required=True)
    train_benchmark.add_argument("--core-checkpoint", type=Path, required=True)
    train_benchmark.add_argument("--output", type=Path, required=True)
    train_benchmark.add_argument("--reader-seed", type=int, required=True)
    train_benchmark.add_argument("--steps", type=int, default=20)
    train_benchmark.add_argument("--batch-size", type=int, default=16)
    train_benchmark.add_argument("--device", default="cuda")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", default="cuda")

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--checkpoint", type=Path, required=True)
    diagnose.add_argument("--cache-dir", type=Path, required=True)
    diagnose.add_argument("--data-dir", type=Path, required=True)
    diagnose.add_argument("--split", default="validation")
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--batch-size", type=int, default=16)
    diagnose.add_argument("--device", default="cuda")

    assess = sub.add_parser("assess-failure")
    assess.add_argument("--overfit", type=Path, required=True)
    assess.add_argument("--cache-audit", type=Path, required=True)
    assess.add_argument("--training", type=Path, required=True)
    assess.add_argument("--diagnostic", type=Path, required=True)
    assess.add_argument("--gradient-audit", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)

    gradient_audit = sub.add_parser("gradient-audit")
    gradient_audit.add_argument("--checkpoint", type=Path, required=True)
    gradient_audit.add_argument("--cache-dir", type=Path, required=True)
    gradient_audit.add_argument("--data-dir", type=Path, required=True)
    gradient_audit.add_argument("--split", default="train")
    gradient_audit.add_argument("--output", type=Path, required=True)
    gradient_audit.add_argument("--batch-size", type=int, default=16)
    gradient_audit.add_argument("--device", default="cuda")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "cache":
        result = cache_a120b_full_hidden(
            args.data_dir,
            args.output_dir,
            device=args.device,
            max_length=args.max_length,
            inference_batch_size=args.inference_batch_size,
            shard_size=args.shard_size,
            overfit32=args.overfit32,
            splits=args.split,
        )
    elif args.command == "cache-benchmark":
        result = benchmark_a120b_cache_batches(
            args.data_dir,
            args.output,
            batch_sizes=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )
    elif args.command == "audit":
        result = audit_a120b_cache(args.cache_dir, args.data_dir)
        _write_json(args.output, result)
    elif args.command == "train":
        result = train_a120b_boundary(
            args.cache_dir,
            args.data_dir,
            args.core_checkpoint,
            args.output_dir,
            spec=A120BTrainSpec(
                reader_seed=args.reader_seed,
                data_seed=args.data_seed,
                core_model_seed=args.core_model_seed,
                steps=args.steps,
                batch_size=args.batch_size,
                validation_interval=args.validation_interval,
            ),
            device=args.device,
            overfit_mode=args.overfit,
        )
    elif args.command == "train-benchmark":
        result = benchmark_a120b_training_pipeline(
            args.cache_dir,
            args.data_dir,
            args.core_checkpoint,
            args.output,
            reader_seed=args.reader_seed,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "evaluate":
        result = evaluate_a120b_formal(
            args.checkpoint,
            args.cache_dir,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    elif args.command == "diagnose":
        result = diagnose_a120b_boundary(
            args.checkpoint,
            args.cache_dir,
            args.data_dir,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    elif args.command == "assess-failure":
        result = assess_a120b_failure(
            args.overfit,
            args.cache_audit,
            args.training,
            args.diagnostic,
            args.gradient_audit,
            args.output,
        )
    else:
        result = audit_a120b_gradient_credit(
            args.checkpoint,
            args.cache_dir,
            args.data_dir,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
