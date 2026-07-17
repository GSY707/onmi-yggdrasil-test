from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_13_train import _write_json
from yggdrasil_v2.reasoning_medium.a1_18_interventions import run_a118_interventions
from yggdrasil_v2.reasoning_medium.a1_18_train import (
    A118TrainSpec,
    evaluate_a118_formal,
    train_a118,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="V2-A1.18 training-only QAUX, FINAL-SAUX, and TSAUX"
    )
    sub = root.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model-seed", type=int, required=True)
    train.add_argument("--data-seed", type=int, required=True)
    train.add_argument(
        "--training-auxiliary",
        choices=("queried_answer", "full_state", "full_trajectory_state"),
        required=True,
    )
    train.add_argument("--steps", type=int, default=4000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--device", default="cuda")
    train.add_argument("--overfit", action="store_true")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--device", default="cuda")

    intervene = sub.add_parser("intervene")
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--data-dir", type=Path, required=True)
    intervene.add_argument("--formal-eval", type=Path, required=True)
    intervene.add_argument("--output", type=Path, required=True)
    intervene.add_argument("--batch-size", type=int, default=32)
    intervene.add_argument("--device", default="cuda")

    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        result = train_a118(
            args.data_dir,
            args.output_dir,
            spec=A118TrainSpec(
                model_seed=args.model_seed,
                data_seed=args.data_seed,
                training_auxiliary=args.training_auxiliary,
                steps=args.steps,
                batch_size=args.batch_size,
            ),
            device=args.device,
            overfit_mode=args.overfit,
        )
    elif args.command == "evaluate":
        result = evaluate_a118_formal(
            args.checkpoint,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    elif args.command == "intervene":
        result = run_a118_interventions(
            args.checkpoint,
            args.data_dir,
            args.formal_eval,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
