from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_16_assessment import assess_a116
from yggdrasil_v2.reasoning_medium.a1_13_interventions import run_a113_interventions
from yggdrasil_v2.reasoning_medium.a1_13_train import (
    A113TrainSpec,
    _write_json,
    evaluate_a113_formal,
    train_a113,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="V2-A1.16 remove redundant CE from the state-coupled answer"
    )
    sub = root.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model-seed", type=int, required=True)
    train.add_argument("--data-seed", type=int, required=True)
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
    assess = sub.add_parser("assess")
    assess.add_argument("--a115-assessment", type=Path, required=True)
    assess.add_argument("--overfit-result", type=Path, required=True)
    assess.add_argument("--run-dir", type=Path, action="append", required=True)
    assess.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        result = train_a113(
            args.data_dir,
            args.output_dir,
            spec=A113TrainSpec(
                model_seed=args.model_seed,
                data_seed=args.data_seed,
                structured_transition=True,
                prototype_closure_objective=True,
                query_coupled_answer=True,
                answer_loss_weight=0.0,
                steps=args.steps,
                batch_size=args.batch_size,
            ),
            device=args.device,
            overfit_mode=args.overfit,
        )
    elif args.command == "evaluate":
        result = evaluate_a113_formal(
            args.checkpoint,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    elif args.command == "intervene":
        result = run_a113_interventions(
            args.checkpoint,
            args.data_dir,
            args.formal_eval,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    else:
        result = assess_a116(
            args.a115_assessment, args.overfit_result, args.run_dir, args.output
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
