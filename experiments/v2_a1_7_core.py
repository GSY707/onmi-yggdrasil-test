from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from yggdrasil_v2.reasoning_medium.a1_6_closure_diagnostic import run_a16_closure_diagnostic
from yggdrasil_v2.reasoning_medium.a1_7_core import A17CoreConfig
from yggdrasil_v2.reasoning_medium.a1_7_data import (
    A17DatasetSpec,
    assert_a17_data_gate,
    audit_a17_data,
    build_a17_dataset,
    load_a17_records,
)
from yggdrasil_v2.reasoning_medium.a1_7_train import (
    evaluate_a17_core,
    load_a17_checkpoint,
    run_a17_interventions,
    train_a17_c0,
)
from yggdrasil_v2.reasoning_medium.a1_7_stress import (
    A17StressSpec,
    assert_a17_stress_gate,
    audit_a17_stress_data,
    build_a17_stress_dataset,
    evaluate_a17_stress,
)
from yggdrasil_v2.reasoning_medium.a1_7_assessment import build_a17_assessment_summary


DEFAULT_DATA = Path("artifacts/v2-a/a1_7/data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-Yggdrasil V2-A1.7 minimal continuous core")
    sub = parser.add_subparsers(dest="command", required=True)

    diagnose = sub.add_parser("diagnose-a1-6")
    diagnose.add_argument("--data-dir", type=Path, default=Path("artifacts/v2-a/a1_6/data"))
    diagnose.add_argument("--checkpoint", type=Path, default=Path("artifacts/v2-a/a1_6/c0-formal/best.pt"))
    diagnose.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_6/diagnostics/closure-diagnostic.json"))
    diagnose.add_argument("--device", default="cuda")
    diagnose.add_argument("--batch-size", type=int, default=128)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--seed", type=int, default=20260715)
    prepare.add_argument("--train-size", type=int, default=8192)
    prepare.add_argument("--validation-size", type=int, default=512)
    prepare.add_argument("--test-size", type=int, default=512)
    prepare.add_argument("--length-size", type=int, default=512)
    prepare.add_argument("--relation-size", type=int, default=512)
    prepare.add_argument("--causal-size", type=int, default=512)

    audit = sub.add_parser("audit-data")
    audit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    audit.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_7/data-audit.json"))

    train = sub.add_parser("c0-train")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1_7/c0-formal"))
    train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=20260715)
    train.add_argument("--train-limit", type=int, default=0)
    train.add_argument("--validation-limit", type=int, default=0)
    train.add_argument("--steps", type=int, default=6000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--validation-interval", type=int, default=100)
    train.add_argument("--early-stop-exact", type=float, default=None)
    train.add_argument("--early-stop-patience", type=int, default=None)
    train.add_argument("--closure-weight", type=float, default=1.0)
    train.add_argument("--ablation-address-content-mixing", action="store_true")

    evaluate = sub.add_parser("c0-evaluate")
    evaluate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=128)

    intervene = sub.add_parser("c0-intervene")
    intervene.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    intervene.add_argument("--split", default="causal_core", choices=("causal_core",))
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--output", type=Path, required=True)
    intervene.add_argument("--device", default="cuda")
    intervene.add_argument("--batch-size", type=int, default=128)

    stress_prepare = sub.add_parser("prepare-stress")
    stress_prepare.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1_7/stress-data"))
    stress_prepare.add_argument("--base-data-dir", type=Path, default=DEFAULT_DATA)
    stress_prepare.add_argument("--seed", type=int, default=20260718)
    stress_prepare.add_argument("--supported-size", type=int, default=768)
    stress_prepare.add_argument("--relation-size", type=int, default=768)
    stress_prepare.add_argument("--lengths", type=int, nargs="+", default=[8, 12, 16])

    stress_audit = sub.add_parser("audit-stress")
    stress_audit.add_argument("--data-dir", type=Path, default=Path("artifacts/v2-a/a1_7/stress-data"))
    stress_audit.add_argument("--base-data-dir", type=Path, default=DEFAULT_DATA)
    stress_audit.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_7/stress-data-audit.json"))

    stress_evaluate = sub.add_parser("stress-evaluate")
    stress_evaluate.add_argument("--data-dir", type=Path, default=Path("artifacts/v2-a/a1_7/stress-data"))
    stress_evaluate.add_argument("--base-data-dir", type=Path, default=DEFAULT_DATA)
    stress_evaluate.add_argument("--checkpoint", type=Path, required=True)
    stress_evaluate.add_argument("--output", type=Path, required=True)
    stress_evaluate.add_argument("--device", default="cuda")
    stress_evaluate.add_argument("--batch-size", type=int, default=128)

    assessment = sub.add_parser("summarize-assessment")
    assessment.add_argument("--artifact-root", type=Path, default=Path("artifacts/v2-a/a1_7"))
    assessment.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_7/core-assessment-summary.json"))
    return parser


def _records(data_dir: Path, split: str, limit: int = 0) -> list[dict[str, Any]]:
    records = load_a17_records(data_dir, split)
    return records[:limit] if limit else records


def _formal_gates(evaluation: dict[str, Any]) -> None:
    splits = evaluation["splits"]
    formal_splits = ("test", "length_heldout", "relation_heldout")
    gates = {
        "ordinary_trajectory_full_exact": splits["test"]["trajectory_full_exact"] >= 0.995,
        "length_trajectory_full_exact": splits["length_heldout"]["trajectory_full_exact"] >= 0.95,
        "relation_trajectory_full_exact": splits["relation_heldout"]["trajectory_full_exact"] >= 0.95,
        "source_pointer_accuracy": min(splits[name]["source_pointer_accuracy"] for name in formal_splits) >= 0.995,
        "target_pointer_accuracy": min(splits[name]["target_pointer_accuracy"] for name in formal_splits) >= 0.995,
        "answer_state_logits_identity": all(split["answer_state_logits_identity"]["gate"] for split in splits.values()),
        "prototype_non_collapse": all(split["prototype_statistics"]["non_collapse_gate"] for split in splits.values()),
    }
    evaluation["gates"] = gates
    evaluation["passed"] = all(gates.values())


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "diagnose-a1-6":
        result = run_a16_closure_diagnostic(args.checkpoint, args.data_dir, args.output, device=args.device, batch_size=args.batch_size)
    elif args.command == "prepare-data":
        result = build_a17_dataset(
            args.output_dir,
            A17DatasetSpec(
                seed=args.seed,
                train_size=args.train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                length_size=args.length_size,
                relation_size=args.relation_size,
                causal_size=args.causal_size,
            ),
        )
    elif args.command == "audit-data":
        result = audit_a17_data(args.data_dir, args.output)
        assert_a17_data_gate(result)
    elif args.command == "c0-train":
        if args.closure_weight < 0:
            raise ValueError("--closure-weight must be non-negative")
        audit = audit_a17_data(args.data_dir)
        assert_a17_data_gate(audit)
        overfit = bool(args.train_limit)
        result = train_a17_c0(
            _records(args.data_dir, "train", args.train_limit),
            _records(args.data_dir, "validation", args.validation_limit),
            args.output_dir,
            config=A17CoreConfig(address_content_mixing_ablation=args.ablation_address_content_mixing),
            seed=args.seed,
            device=args.device,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_interval=args.validation_interval,
            early_stop_exact=args.early_stop_exact if overfit else None,
            early_stop_patience=args.early_stop_patience if not overfit else None,
            overfit_mode=overfit,
            closure_weight=args.closure_weight,
        )
        if overfit:
            fit = result["fit"]
            result["overfit_gate_fit"] = {
                key: fit[key] >= 0.999
                for key in (
                    "final_query_token_accuracy",
                    "state_token_accuracy",
                    "trajectory_full_exact",
                    "source_pointer_accuracy",
                    "target_pointer_accuracy",
                )
            }
            result["few_shot_validation_diagnostic"] = result["validation"]
            result["overfit_gate"] = result["overfit_gate_fit"]
            (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if not all(result["overfit_gate"].values()):
                raise RuntimeError("A1.7 C0 overfit32 Gate failed; formal C0 is forbidden")
    elif args.command == "c0-evaluate":
        model = load_a17_checkpoint(args.checkpoint, args.device)
        result = {
            "schema_version": "yggdrasil.v2-a1.7.c0.eval.v1",
            "checkpoint": str(args.checkpoint),
            "integrity": model.integrity_report(),
            "splits": {},
        }
        for split in ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core"):
            result["splits"][split] = evaluate_a17_core(
                model,
                _records(args.data_dir, split),
                args.device,
                batch_size=args.batch_size,
                closure_weight=float(getattr(model, "training_closure_weight", 1.0)),
            )
        _formal_gates(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not result["passed"]:
            raise RuntimeError("A1.7 formal C0 Gate failed; causal intervention and C1 are forbidden")
    elif args.command == "c0-intervene":
        model = load_a17_checkpoint(args.checkpoint, args.device)
        intervention = run_a17_interventions(model, _records(args.data_dir, args.split), args.device, batch_size=args.batch_size)
        result = {
            "schema_version": "yggdrasil.v2-a1.7.c0.interventions.v1",
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "results": intervention,
            "passed": intervention["passed"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not result["passed"]:
            raise RuntimeError("A1.7 causal intervention Gate failed; C1 is forbidden")
    elif args.command == "prepare-stress":
        result = build_a17_stress_dataset(
            args.output_dir,
            args.base_data_dir,
            A17StressSpec(
                seed=args.seed,
                supported_size=args.supported_size,
                relation_size=args.relation_size,
                lengths=tuple(args.lengths),
            ),
        )
    elif args.command == "audit-stress":
        result = audit_a17_stress_data(args.data_dir, args.base_data_dir, args.output)
        assert_a17_stress_gate(result)
    elif args.command == "stress-evaluate":
        result = evaluate_a17_stress(
            args.checkpoint,
            args.data_dir,
            args.base_data_dir,
            args.device,
            batch_size=args.batch_size,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not result["passed"]:
            raise RuntimeError("A1.7 long-recurrence stress Gate failed")
    elif args.command == "summarize-assessment":
        result = build_a17_assessment_summary(args.artifact_root, args.output)
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
