from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from yggdrasil_v2.reasoning_medium.a1_10_interventions import cache_a110_interventions
from yggdrasil_v2.reasoning_medium.a1_11_assessment import assess_a111
from yggdrasil_v2.reasoning_medium.a1_11_interventions import (
    run_a111_boundary_interventions,
    run_a111_reasoner_interventions,
)
from yggdrasil_v2.reasoning_medium.a1_11_train import (
    A111BoundaryTrainSpec,
    A111ReasonerTrainSpec,
    evaluate_a111_boundary_formal,
    evaluate_a111_boundary_hard_reembed_diagnostic,
    evaluate_a111_reasoner_formal,
    train_a111_boundary,
    train_a111_reasoner,
)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value[key]
        for key in ("arm", "schema_version", "overfit_passed", "passed", "best_checkpoint_step")
        if key in value
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="V2-A1.11 orthogonal Boundary x Reasoner localization")
    sub = root.add_subparsers(dest="command", required=True)

    boundary_train = sub.add_parser("boundary-train")
    boundary_train.add_argument("--cache-dir", type=Path, required=True)
    boundary_train.add_argument("--data-dir", type=Path, required=True)
    boundary_train.add_argument("--core-checkpoint", type=Path, required=True)
    boundary_train.add_argument("--output-dir", type=Path, required=True)
    boundary_train.add_argument("--reader-seed", type=int, required=True)
    boundary_train.add_argument("--data-seed", type=int, required=True)
    boundary_train.add_argument("--core-seed", type=int, required=True)
    boundary_train.add_argument("--steps", type=int, default=5000)
    boundary_train.add_argument("--batch-size", type=int, default=32)
    boundary_train.add_argument("--device", default="cuda")
    boundary_train.add_argument("--overfit", action="store_true")

    boundary_eval = sub.add_parser("boundary-evaluate")
    boundary_eval.add_argument("--checkpoint", type=Path, required=True)
    boundary_eval.add_argument("--cache-dir", type=Path, required=True)
    boundary_eval.add_argument("--data-dir", type=Path, required=True)
    boundary_eval.add_argument("--output", type=Path, required=True)
    boundary_eval.add_argument("--batch-size", type=int, default=32)
    boundary_eval.add_argument("--device", default="cuda")

    boundary_diagnostic = sub.add_parser("boundary-hard-reembed-diagnostic")
    boundary_diagnostic.add_argument("--checkpoint", type=Path, required=True)
    boundary_diagnostic.add_argument("--cache-dir", type=Path, required=True)
    boundary_diagnostic.add_argument("--data-dir", type=Path, required=True)
    boundary_diagnostic.add_argument("--output", type=Path, required=True)
    boundary_diagnostic.add_argument("--split", default="validation")
    boundary_diagnostic.add_argument("--batch-size", type=int, default=32)
    boundary_diagnostic.add_argument("--device", default="cuda")

    reasoner_train = sub.add_parser("reasoner-train")
    reasoner_train.add_argument("--data-dir", type=Path, required=True)
    reasoner_train.add_argument("--output-dir", type=Path, required=True)
    reasoner_train.add_argument("--reasoner-seed", type=int, required=True)
    reasoner_train.add_argument("--data-seed", type=int, required=True)
    reasoner_train.add_argument("--steps", type=int, default=4000)
    reasoner_train.add_argument("--batch-size", type=int, default=32)
    reasoner_train.add_argument("--device", default="cuda")
    reasoner_train.add_argument("--overfit", action="store_true")

    reasoner_eval = sub.add_parser("reasoner-evaluate")
    reasoner_eval.add_argument("--checkpoint", type=Path, required=True)
    reasoner_eval.add_argument("--data-dir", type=Path, required=True)
    reasoner_eval.add_argument("--output", type=Path, required=True)
    reasoner_eval.add_argument("--batch-size", type=int, default=32)
    reasoner_eval.add_argument("--device", default="cuda")

    boundary_cache_interventions = sub.add_parser("boundary-cache-interventions")
    boundary_cache_interventions.add_argument("--data-dir", type=Path, required=True)
    boundary_cache_interventions.add_argument("--formal-eval", type=Path, required=True)
    boundary_cache_interventions.add_argument("--output-dir", type=Path, required=True)
    boundary_cache_interventions.add_argument("--device", default="cuda")

    boundary_intervene = sub.add_parser("boundary-intervene")
    boundary_intervene.add_argument("--checkpoint", type=Path, required=True)
    boundary_intervene.add_argument("--source-cache-dir", type=Path, required=True)
    boundary_intervene.add_argument("--intervention-cache-dir", type=Path, required=True)
    boundary_intervene.add_argument("--data-dir", type=Path, required=True)
    boundary_intervene.add_argument("--formal-eval", type=Path, required=True)
    boundary_intervene.add_argument("--output", type=Path, required=True)
    boundary_intervene.add_argument("--batch-size", type=int, default=32)
    boundary_intervene.add_argument("--device", default="cuda")

    reasoner_intervene = sub.add_parser("reasoner-intervene")
    reasoner_intervene.add_argument("--checkpoint", type=Path, required=True)
    reasoner_intervene.add_argument("--data-dir", type=Path, required=True)
    reasoner_intervene.add_argument("--formal-eval", type=Path, required=True)
    reasoner_intervene.add_argument("--output", type=Path, required=True)
    reasoner_intervene.add_argument("--batch-size", type=int, default=32)
    reasoner_intervene.add_argument("--device", default="cuda")

    assess = sub.add_parser("assess")
    assess.add_argument("--boundary-run-dir", type=Path, action="append", default=[])
    assess.add_argument("--reasoner-run-dir", type=Path, action="append", default=[])
    assess.add_argument("--boundary-overfit", type=Path, required=True)
    assess.add_argument("--reasoner-overfit", type=Path, required=True)
    assess.add_argument("--a1-9-assessment", type=Path, required=True)
    assess.add_argument("--a1-10-assessment", type=Path, required=True)
    assess.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "boundary-train":
        result = train_a111_boundary(
            args.cache_dir,
            args.data_dir,
            args.core_checkpoint,
            args.output_dir,
            spec=A111BoundaryTrainSpec(
                reader_seed=args.reader_seed,
                data_seed=args.data_seed,
                core_seed=args.core_seed,
                steps=args.steps,
                batch_size=args.batch_size,
            ),
            device=args.device,
            overfit_mode=args.overfit,
        )
        _write(args.output_dir / "results.json", result)
    elif args.command == "boundary-evaluate":
        result = evaluate_a111_boundary_formal(
            args.checkpoint,
            args.cache_dir,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
    elif args.command == "reasoner-train":
        result = train_a111_reasoner(
            args.data_dir,
            args.output_dir,
            spec=A111ReasonerTrainSpec(
                reasoner_seed=args.reasoner_seed,
                data_seed=args.data_seed,
                steps=args.steps,
                batch_size=args.batch_size,
            ),
            device=args.device,
            overfit_mode=args.overfit,
        )
        _write(args.output_dir / "results.json", result)
    elif args.command == "reasoner-evaluate":
        result = evaluate_a111_reasoner_formal(
            args.checkpoint,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
    elif args.command == "boundary-hard-reembed-diagnostic":
        result = evaluate_a111_boundary_hard_reembed_diagnostic(
            args.checkpoint,
            args.cache_dir,
            args.data_dir,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
    elif args.command == "boundary-cache-interventions":
        result = cache_a110_interventions(
            args.data_dir,
            args.formal_eval,
            args.output_dir,
            device=args.device,
        )
        _write(args.output_dir / "manifest.json", result)
    elif args.command == "boundary-intervene":
        result = run_a111_boundary_interventions(
            args.checkpoint,
            args.source_cache_dir,
            args.intervention_cache_dir,
            args.data_dir,
            args.formal_eval,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
    elif args.command == "reasoner-intervene":
        result = run_a111_reasoner_interventions(
            args.checkpoint,
            args.data_dir,
            args.formal_eval,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
    else:
        result = assess_a111(
            args.boundary_run_dir,
            args.reasoner_run_dir,
            args.boundary_overfit,
            args.reasoner_overfit,
            args.a1_9_assessment,
            args.a1_10_assessment,
            args.output,
        )
        _write(args.output, result)


if __name__ == "__main__":
    main()
