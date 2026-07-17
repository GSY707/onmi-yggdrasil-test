from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_13_train import _write_json
from yggdrasil_v2.reasoning_medium.a1_19h_assessment import assess_a119h_h1
from yggdrasil_v2.reasoning_medium.a1_19h_interventions import (
    run_a119h_h1_interventions,
)
from yggdrasil_v2.reasoning_medium.a1_19h_h2_assessment import assess_a119h_h2
from yggdrasil_v2.reasoning_medium.a1_19h_h2_data import (
    A119H2DatasetSpec,
    audit_a119h2_dataset,
    build_a119h2_dataset,
)
from yggdrasil_v2.reasoning_medium.a1_19h_h2_interventions import (
    run_a119h_h2_interventions,
)
from yggdrasil_v2.reasoning_medium.a1_19h_h2_train import (
    A119H2TrainSpec,
    benchmark_a119h2_training_pipeline,
    evaluate_a119h2_formal,
    train_a119h_h2,
)
from yggdrasil_v2.reasoning_medium.a1_19h_train import (
    A119HTrainSpec,
    evaluate_a119h_formal,
    train_a119h_h1,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="V2-A1.19H staged hybrid-core Gate")
    sub = root.add_subparsers(dest="command", required=True)

    train = sub.add_parser("h1-train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model-seed", type=int, required=True)
    train.add_argument("--data-seed", type=int, required=True)
    train.add_argument("--mapping-seed", type=int, required=True)
    train.add_argument("--steps", type=int, default=4000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--device", default="cuda")
    train.add_argument("--overfit", action="store_true")

    evaluate = sub.add_parser("h1-evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--device", default="cuda")

    intervene = sub.add_parser("h1-intervene")
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--data-dir", type=Path, required=True)
    intervene.add_argument("--formal-eval", type=Path, required=True)
    intervene.add_argument("--output", type=Path, required=True)
    intervene.add_argument("--batch-size", type=int, default=32)
    intervene.add_argument("--device", default="cuda")

    assess = sub.add_parser("h1-assess")
    assess.add_argument("--overfit", type=Path, required=True)
    assess.add_argument("--run-dir", type=Path, action="append", required=True)
    assess.add_argument("--output", type=Path, required=True)

    prepare_h2 = sub.add_parser("h2-prepare")
    prepare_h2.add_argument("--output-dir", type=Path, required=True)
    prepare_h2.add_argument("--data-seed", type=int, required=True)
    prepare_h2.add_argument(
        "--forbidden-data-dir", type=Path, action="append", default=[]
    )

    audit_h2 = sub.add_parser("h2-audit")
    audit_h2.add_argument("--data-dir", type=Path, required=True)
    audit_h2.add_argument("--output", type=Path, required=True)

    train_h2 = sub.add_parser("h2-train")
    train_h2.add_argument("--data-dir", type=Path, required=True)
    train_h2.add_argument("--output-dir", type=Path, required=True)
    train_h2.add_argument("--model-seed", type=int, required=True)
    train_h2.add_argument("--data-seed", type=int, required=True)
    train_h2.add_argument("--mapping-seed", type=int, required=True)
    train_h2.add_argument("--steps", type=int, default=4000)
    train_h2.add_argument("--batch-size", type=int, default=32)
    train_h2.add_argument("--device", default="cuda")
    train_h2.add_argument("--overfit", action="store_true")

    benchmark_h2 = sub.add_parser("h2-benchmark")
    benchmark_h2.add_argument("--data-dir", type=Path, required=True)
    benchmark_h2.add_argument("--output", type=Path, required=True)
    benchmark_h2.add_argument("--model-seed", type=int, required=True)
    benchmark_h2.add_argument("--mapping-seed", type=int, required=True)
    benchmark_h2.add_argument("--steps", type=int, default=60)
    benchmark_h2.add_argument("--batch-size", type=int, default=32)
    benchmark_h2.add_argument("--device", default="cuda")

    evaluate_h2 = sub.add_parser("h2-evaluate")
    evaluate_h2.add_argument("--checkpoint", type=Path, required=True)
    evaluate_h2.add_argument("--data-dir", type=Path, required=True)
    evaluate_h2.add_argument("--output", type=Path, required=True)
    evaluate_h2.add_argument("--batch-size", type=int, default=32)
    evaluate_h2.add_argument("--device", default="cuda")

    intervene_h2 = sub.add_parser("h2-intervene")
    intervene_h2.add_argument("--checkpoint", type=Path, required=True)
    intervene_h2.add_argument("--data-dir", type=Path, required=True)
    intervene_h2.add_argument("--formal-eval", type=Path, required=True)
    intervene_h2.add_argument("--output", type=Path, required=True)
    intervene_h2.add_argument("--batch-size", type=int, default=32)
    intervene_h2.add_argument("--device", default="cuda")

    assess_h2 = sub.add_parser("h2-assess")
    assess_h2.add_argument("--h1-assessment", type=Path, required=True)
    assess_h2.add_argument("--overfit", type=Path, required=True)
    assess_h2.add_argument("--data-audit", type=Path, action="append", required=True)
    assess_h2.add_argument("--run-dir", type=Path, action="append", required=True)
    assess_h2.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "h1-train":
        result = train_a119h_h1(
            args.data_dir,
            args.output_dir,
            spec=A119HTrainSpec(
                model_seed=args.model_seed,
                data_seed=args.data_seed,
                mapping_seed=args.mapping_seed,
                steps=args.steps,
                batch_size=args.batch_size,
            ),
            device=args.device,
            overfit_mode=args.overfit,
        )
    elif args.command == "h1-evaluate":
        result = evaluate_a119h_formal(
            args.checkpoint,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    elif args.command == "h1-intervene":
        result = run_a119h_h1_interventions(
            args.checkpoint,
            args.data_dir,
            args.formal_eval,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    elif args.command == "h1-assess":
        result = assess_a119h_h1(args.overfit, args.run_dir, args.output)
    elif args.command == "h2-prepare":
        result = build_a119h2_dataset(
            args.output_dir,
            A119H2DatasetSpec(data_seed=args.data_seed),
            forbidden_data_dirs=args.forbidden_data_dir,
        )
    elif args.command == "h2-audit":
        result = audit_a119h2_dataset(args.data_dir)
        _write_json(args.output, result)
    elif args.command == "h2-train":
        result = train_a119h_h2(
            args.data_dir,
            args.output_dir,
            spec=A119H2TrainSpec(
                model_seed=args.model_seed,
                data_seed=args.data_seed,
                mapping_seed=args.mapping_seed,
                steps=args.steps,
                batch_size=args.batch_size,
            ),
            device=args.device,
            overfit_mode=args.overfit,
        )
    elif args.command == "h2-benchmark":
        result = benchmark_a119h2_training_pipeline(
            args.data_dir,
            args.output,
            model_seed=args.model_seed,
            mapping_seed=args.mapping_seed,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "h2-evaluate":
        result = evaluate_a119h2_formal(
            args.checkpoint,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    elif args.command == "h2-intervene":
        result = run_a119h_h2_interventions(
            args.checkpoint,
            args.data_dir,
            args.formal_eval,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    else:
        result = assess_a119h_h2(
            args.h1_assessment,
            args.overfit,
            args.data_audit,
            args.run_dir,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
