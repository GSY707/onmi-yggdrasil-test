from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_13_train import (
    A113TrainSpec,
    _write_json,
    evaluate_a113_formal,
    train_a113,
)
from yggdrasil_v2.reasoning_medium.a1_13_interventions import run_a113_interventions
from yggdrasil_v2.reasoning_medium.a1_13_assessment import assess_a113
from yggdrasil_v2.reasoning_medium.a1_13f_assessment import assess_a113f


ARMS = {
    "GENERIC-CE": (False, False),
    "GENERIC-CLOSURE": (False, True),
    "STRUCTURED-CE": (True, False),
    "STRUCTURED-CLOSURE": (True, True),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2-A1.13 transition x closure localization")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--arm", choices=tuple(ARMS), required=True)
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model-seed", type=int, required=True)
    train.add_argument("--data-seed", type=int, required=True)
    train.add_argument("--steps", type=int, default=4000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--early-stop-patience", type=int, default=6)
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
    assess.add_argument("--a112-assessment", type=Path, required=True)
    assess.add_argument("--generic-closure-run-dir", type=Path, action="append", required=True)
    assess.add_argument("--structured-ce-run-dir", type=Path, action="append", required=True)
    assess.add_argument("--structured-closure-run-dir", type=Path, action="append", default=[])
    assess.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit-assess")
    audit.add_argument("--generic-ce-run-dir", type=Path, action="append", required=True)
    audit.add_argument("--generic-closure-run-dir", type=Path, action="append", required=True)
    audit.add_argument("--structured-ce-run-dir", type=Path, action="append", required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        structured, closure = ARMS[args.arm]
        result = train_a113(
            args.data_dir,
            args.output_dir,
            spec=A113TrainSpec(
                model_seed=args.model_seed,
                data_seed=args.data_seed,
                structured_transition=structured,
                prototype_closure_objective=closure,
                steps=args.steps,
                batch_size=args.batch_size,
                early_stop_patience=args.early_stop_patience,
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
    elif args.command == "assess":
        result = assess_a113(
            args.a112_assessment,
            args.generic_closure_run_dir,
            args.structured_ce_run_dir,
            args.output,
            args.structured_closure_run_dir,
        )
    else:
        result = assess_a113f(
            args.generic_ce_run_dir,
            args.generic_closure_run_dir,
            args.structured_ce_run_dir,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
