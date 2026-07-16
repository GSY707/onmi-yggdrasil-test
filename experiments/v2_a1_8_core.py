from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from yggdrasil_v2.reasoning_medium.a1_7_core import A17CoreConfig
from yggdrasil_v2.reasoning_medium.a1_8_assessment import build_a18_assessment_summary
from yggdrasil_v2.reasoning_medium.a1_8_data import (
    A18DatasetSpec,
    TRAIN_LENGTHS,
    assert_a18_data_gate,
    audit_a18_data,
    build_a18_dataset,
    load_a18_records,
)
from yggdrasil_v2.reasoning_medium.a1_8_train import (
    A18TrainSpec,
    benchmark_a18_costs,
    evaluate_a18_formal,
    run_a18_interventions,
    train_a18_random_depth,
)


DEFAULT_ROOT = Path("artifacts/v2-a/a1_8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-Yggdrasil V2-A1.8 random-depth long-horizon core")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--data-seed", type=int, required=True)
    prepare.add_argument("--train-size", type=int, default=16384)
    prepare.add_argument("--validation-size", type=int, default=2048)
    prepare.add_argument("--short-size", type=int, default=768)
    prepare.add_argument("--in-range-size", type=int, default=768)
    prepare.add_argument("--relation-in-range-size", type=int, default=768)
    prepare.add_argument("--ood-size", type=int, default=768)
    prepare.add_argument("--relation-ood-size", type=int, default=768)
    prepare.add_argument("--causal-size", type=int, default=512)
    prepare.add_argument("--forbid-data-dir", type=Path, nargs="*", default=[])

    audit = sub.add_parser("audit-data")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--model-seed", type=int, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--steps", type=int, default=8000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--validation-interval", type=int, default=200)
    train.add_argument("--early-stop-patience", type=int, default=15)
    train.add_argument("--closure-weight", type=float, default=1.0)
    train.add_argument("--validation-trajectory-target", type=float, default=0.995)
    train.add_argument("--overfit-per-length", type=int, default=0)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=128)

    intervene = sub.add_parser("intervene")
    intervene.add_argument("--data-dir", type=Path, required=True)
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--formal-eval", type=Path, required=True)
    intervene.add_argument("--output", type=Path, required=True)
    intervene.add_argument("--device", default="cuda")
    intervene.add_argument("--batch-size", type=int, default=128)

    benchmark = sub.add_parser("benchmark-costs")
    benchmark.add_argument("--data-dir", type=Path, required=True)
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--batch-size", type=int, default=128)
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--repeats", type=int, default=10)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    summarize.add_argument("--output", type=Path, default=DEFAULT_ROOT / "assessment-summary.json")
    summarize.add_argument("--a1-7-summary", type=Path, default=Path("artifacts/v2-a/a1_7/core-assessment-summary.json"))
    return parser


def _select_per_length(records: Sequence[dict[str, Any]], per_length: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for length in TRAIN_LENGTHS:
        rows = [record for record in records if int(record["program_length"]) == length]
        if len(rows) < per_length:
            raise ValueError(f"not enough length-{length} records for overfit selection")
        selected.extend(rows[:per_length])
    return selected


def _write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        result = build_a18_dataset(
            args.output_dir,
            A18DatasetSpec(
                data_seed=args.data_seed,
                train_size=args.train_size,
                validation_size=args.validation_size,
                short_size=args.short_size,
                in_range_size=args.in_range_size,
                relation_in_range_size=args.relation_in_range_size,
                ood_size=args.ood_size,
                relation_ood_size=args.relation_ood_size,
                causal_size=args.causal_size,
            ),
            forbidden_data_dirs=args.forbid_data_dir,
        )
    elif args.command == "audit-data":
        result = audit_a18_data(args.data_dir, args.output)
        assert_a18_data_gate(result)
    elif args.command == "train":
        audit = audit_a18_data(args.data_dir)
        assert_a18_data_gate(audit)
        manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
        train_records = load_a18_records(args.data_dir, "train")
        overfit_mode = args.overfit_per_length > 0
        if overfit_mode:
            train_records = _select_per_length(train_records, args.overfit_per_length)
            validation_records = train_records
        else:
            validation_records = load_a18_records(args.data_dir, "validation")
        result = train_a18_random_depth(
            train_records,
            validation_records,
            args.output_dir,
            spec=A18TrainSpec(
                model_seed=args.model_seed,
                data_seed=int(manifest["data_seed"]),
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                validation_interval=args.validation_interval,
                early_stop_patience=args.early_stop_patience,
                closure_weight=args.closure_weight,
                validation_trajectory_target=args.validation_trajectory_target,
            ),
            config=A17CoreConfig(),
            device=args.device,
            overfit_mode=overfit_mode,
        )
        if overfit_mode:
            by_length = result["validation"]["by_length"]
            gate = {
                "all_length_trajectory_exact": all(row["trajectory_full_exact"] >= 0.999 for row in by_length.values()),
                "source_pointer_exact": result["validation"]["aggregate"]["source_pointer_accuracy"] >= 0.999,
                "target_pointer_exact": result["validation"]["aggregate"]["target_pointer_accuracy"] >= 0.999,
                "answer_state_identity": result["validation"]["aggregate"]["answer_state_logits_identity"]["gate"],
            }
            result["overfit_gate"] = gate
            result["overfit_passed"] = all(gate.values())
            _write(args.output_dir / "results.json", result)
            if not result["overfit_passed"]:
                raise RuntimeError("A1.8 length-balanced overfit Gate failed")
    elif args.command == "evaluate":
        result = evaluate_a18_formal(
            args.checkpoint,
            args.data_dir,
            args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
        if not result["passed"]:
            raise RuntimeError("A1.8 formal Gate failed; causal intervention is forbidden for this run")
    elif args.command == "intervene":
        formal = json.loads(args.formal_eval.read_text(encoding="utf-8"))
        if not formal.get("passed"):
            raise RuntimeError("A1.8 intervention requires a passing formal eval for the same run")
        result = run_a18_interventions(
            args.checkpoint,
            args.data_dir,
            args.device,
            batch_size=args.batch_size,
        )
        _write(args.output, result)
        if not result["passed"]:
            raise RuntimeError("A1.8 causal intervention Gate failed")
    elif args.command == "benchmark-costs":
        result = benchmark_a18_costs(
            args.checkpoint,
            args.data_dir,
            args.device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        _write(args.output, result)
    elif args.command == "summarize":
        result = build_a18_assessment_summary(
            args.artifact_root,
            args.output,
            a17_summary_path=args.a1_7_summary,
        )
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
