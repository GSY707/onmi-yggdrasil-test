from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_13_train import _write_json
from yggdrasil_v2.reasoning_medium.a1_20c_diagnostics import (
    audit_a120d_hidden_causality,
    diagnose_a120d_heldout_tail,
    diagnose_a120c_overfit_failure,
)
from yggdrasil_v2.reasoning_medium.a1_20c_supervision import (
    audit_a120c_supervision,
    prepare_a120c_supervision,
)
from yggdrasil_v2.reasoning_medium.a1_20c_train import (
    A120CTrainSpec,
    evaluate_a120c_formal,
    evaluate_a120c_split_probe,
    train_a120c_arm,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="V2-A1.20C Boundary compiler x credit repair"
    )
    sub = root.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-supervision")
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    audit = sub.add_parser("audit-supervision")
    audit.add_argument("--cache-dir", type=Path, required=True)
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--supervision-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    train = sub.add_parser("train")
    train.add_argument("--cache-dir", type=Path, required=True)
    train.add_argument("--supervision-dir", type=Path, required=True)
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--core-checkpoint", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--reader-seed", type=int, required=True)
    train.add_argument(
        "--schedule-seed",
        type=int,
        help=(
            "Optional batch-order seed. Continuations keep reader-seed "
            "fixed for parameter identity while varying optimization path."
        ),
    )
    train.add_argument("--data-seed", type=int, required=True)
    train.add_argument("--core-model-seed", type=int, required=True)
    train.add_argument(
        "--compiler",
        choices=("flat", "hierarchical", "factorized"),
        required=True,
    )
    train.add_argument(
        "--credit-mode",
        choices=("hard_local", "straight_through"),
        required=True,
    )
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--state-loss-weight", type=float, default=1.0)
    train.add_argument("--state-tail-weight", type=float, default=0.0)
    train.add_argument("--state-tail-fraction", type=float, default=0.01)
    train.add_argument("--pointer-tail-weight", type=float, default=0.0)
    train.add_argument("--pointer-tail-fraction", type=float, default=0.01)
    train.add_argument("--control-tail-weight", type=float, default=0.0)
    train.add_argument("--control-tail-fraction", type=float, default=0.01)
    train.add_argument(
        "--payload-regression-weight", type=float, default=10.0
    )
    train.add_argument(
        "--optimization-scope",
        choices=(
            "all",
            "control",
            "identity",
            "entity",
            "entity_anchor",
            "operation",
            "operation_anchor",
            "payload",
            "structure_anchor",
        ),
        default="all",
    )
    train.add_argument(
        "--objective-scope",
        choices=("full", "anchor_only"),
        default="full",
    )
    train.add_argument("--validation-interval", type=int, default=200)
    train.add_argument("--device", default="cuda")
    train.add_argument("--overfit", action="store_true")
    train.add_argument("--initial-checkpoint", type=Path)
    train.add_argument("--resume-optimizer", action="store_true")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument(
        "--training-cache-dir", type=Path, required=True
    )
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", default="cuda")

    probe = sub.add_parser("probe-splits")
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--cache-dir", type=Path, required=True)
    probe.add_argument("--training-cache-dir", type=Path, required=True)
    probe.add_argument("--data-dir", type=Path, required=True)
    probe.add_argument("--split", action="append", required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--batch-size", type=int, default=16)
    probe.add_argument("--device", default="cuda")

    diagnose = sub.add_parser("diagnose-overfit")
    diagnose.add_argument("--checkpoint", type=Path, required=True)
    diagnose.add_argument("--results", type=Path, required=True)
    diagnose.add_argument("--history", type=Path, required=True)
    diagnose.add_argument("--cache-dir", type=Path, required=True)
    diagnose.add_argument("--supervision-dir", type=Path, required=True)
    diagnose.add_argument("--data-dir", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--device", default="cuda")

    diagnose_tail = sub.add_parser("diagnose-tail")
    diagnose_tail.add_argument("--checkpoint", type=Path, required=True)
    diagnose_tail.add_argument("--cache-dir", type=Path, required=True)
    diagnose_tail.add_argument(
        "--supervision-dir", type=Path, required=True
    )
    diagnose_tail.add_argument("--data-dir", type=Path, required=True)
    diagnose_tail.add_argument("--output", type=Path, required=True)
    diagnose_tail.add_argument("--split", default="validation")
    diagnose_tail.add_argument("--batch-size", type=int, default=16)
    diagnose_tail.add_argument("--device", default="cuda")

    causal = sub.add_parser("causal-hidden-audit")
    causal.add_argument("--checkpoint", type=Path, required=True)
    causal.add_argument("--cache-dir", type=Path, required=True)
    causal.add_argument("--data-dir", type=Path, required=True)
    causal.add_argument("--formal", type=Path, required=True)
    causal.add_argument("--output", type=Path, required=True)
    causal.add_argument("--batch-size", type=int, default=16)
    causal.add_argument("--device", default="cuda")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-supervision":
        result = prepare_a120c_supervision(
            args.cache_dir, args.data_dir, args.output_dir
        )
    elif args.command == "audit-supervision":
        result = audit_a120c_supervision(
            args.cache_dir, args.data_dir, args.supervision_dir
        )
        _write_json(args.output, result)
    elif args.command == "diagnose-overfit":
        result = diagnose_a120c_overfit_failure(
            checkpoint_path=args.checkpoint,
            results_path=args.results,
            history_path=args.history,
            cache_dir=args.cache_dir,
            supervision_dir=args.supervision_dir,
            data_dir=args.data_dir,
            output_path=args.output,
            device=args.device,
        )
    elif args.command == "diagnose-tail":
        result = diagnose_a120d_heldout_tail(
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
            supervision_dir=args.supervision_dir,
            data_dir=args.data_dir,
            output_path=args.output,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "causal-hidden-audit":
        result = audit_a120d_hidden_causality(
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            formal_path=args.formal,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "evaluate":
        result = evaluate_a120c_formal(
            args.checkpoint,
            args.cache_dir,
            args.training_cache_dir,
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    elif args.command == "probe-splits":
        result = evaluate_a120c_split_probe(
            args.checkpoint,
            args.cache_dir,
            args.training_cache_dir,
            args.data_dir,
            args.split,
            device=args.device,
            batch_size=args.batch_size,
        )
        _write_json(args.output, result)
    else:
        result = train_a120c_arm(
            args.cache_dir,
            args.supervision_dir,
            args.data_dir,
            args.core_checkpoint,
            args.output_dir,
            spec=A120CTrainSpec(
                reader_seed=args.reader_seed,
                data_seed=args.data_seed,
                core_model_seed=args.core_model_seed,
                compiler=args.compiler,
                credit_mode=args.credit_mode,
                schedule_seed=args.schedule_seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                state_loss_weight=args.state_loss_weight,
                state_tail_weight=args.state_tail_weight,
                state_tail_fraction=args.state_tail_fraction,
                pointer_tail_weight=args.pointer_tail_weight,
                pointer_tail_fraction=args.pointer_tail_fraction,
                control_tail_weight=args.control_tail_weight,
                control_tail_fraction=args.control_tail_fraction,
                payload_regression_weight=args.payload_regression_weight,
                optimization_scope=args.optimization_scope,
                objective_scope=args.objective_scope,
                validation_interval=args.validation_interval,
            ),
            device=args.device,
            overfit_mode=args.overfit,
            initial_checkpoint=args.initial_checkpoint,
            resume_optimizer=args.resume_optimizer,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
