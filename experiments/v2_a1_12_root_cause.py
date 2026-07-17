from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from yggdrasil_v2.reasoning_medium.a1_12_assessment import assess_a112
from yggdrasil_v2.reasoning_medium.a1_12_interventions import run_a112_interventions
from yggdrasil_v2.reasoning_medium.a1_12_train import (
    A112TrainSpec,
    evaluate_a112_formal,
    train_a112,
)


ARM_CONFIG = {
    "BIND": (True, False),
    "CURSOR": (False, True),
    "BOTH": (True, True),
}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value[key]
                for key in ("schema_version", "arm", "overfit_passed", "passed", "best_checkpoint_step")
                if key in value
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="V2-A1.12 binding x cursor root-cause localization")
    sub = root.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--arm", choices=tuple(ARM_CONFIG), required=True)
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
    assess.add_argument("--bind-run-dir", type=Path, action="append", required=True)
    assess.add_argument("--cursor-run-dir", type=Path, action="append", required=True)
    assess.add_argument("--both-run-dir", type=Path, action="append", required=True)
    assess.add_argument("--bind-overfit", type=Path, required=True)
    assess.add_argument("--cursor-overfit", type=Path, required=True)
    assess.add_argument("--both-overfit", type=Path, required=True)
    assess.add_argument("--a1-11-assessment", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        entity, cursor = ARM_CONFIG[args.arm]
        result = train_a112(
            args.data_dir,
            args.output_dir,
            spec=A112TrainSpec(
                model_seed=args.model_seed,
                data_seed=args.data_seed,
                entity_addressable=entity,
                aligned_operation_cursor=cursor,
                steps=args.steps,
                batch_size=args.batch_size,
            ),
            device=args.device,
            overfit_mode=args.overfit,
        )
        _write(args.output_dir / "results.json", result)
    elif args.command == "evaluate":
        result = evaluate_a112_formal(
            args.checkpoint,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
    elif args.command == "intervene":
        result = run_a112_interventions(
            args.checkpoint,
            args.data_dir,
            args.formal_eval,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
    else:
        result = assess_a112(
            args.bind_run_dir,
            args.cursor_run_dir,
            args.both_run_dir,
            args.bind_overfit,
            args.cursor_overfit,
            args.both_overfit,
            args.a1_11_assessment,
            args.output,
        )
        _write(args.output, result)


if __name__ == "__main__":
    main()
