from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_21p_pareto import (
    assess_a121p_stage,
    diagnose_a121p_entity_section_window,
    diagnose_a121p_entity_capacity_threshold,
    diagnose_a121p_operation_threshold,
    interpolate_a121p_boundary_checkpoints,
    prepare_a121p_joint_training_bundle,
    prepare_a121p_routing_data,
    run_a121p_cross_domain_probe,
    run_a121p_current_family_preflight,
)
from yggdrasil_v2.reasoning_medium.a1_21p_k1 import (
    A121PK1TrainSpec,
    evaluate_a121p_k1_formal,
    train_a121p_k1,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="V2-A1.21P matched quality-cost Pareto"
    )
    sub = root.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("current-family-preflight")
    preflight.add_argument("--checkpoint", type=Path, required=True)
    preflight.add_argument("--data-dir", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--split", action="append", required=True)
    preflight.add_argument("--examples-per-split", type=int, default=2)
    preflight.add_argument("--device", default="cuda")
    preflight.add_argument("--max-input-tokens", type=int, default=2048)
    preflight.add_argument("--max-wall-seconds", type=float, default=60.0)
    preflight.add_argument(
        "--task-variant",
        choices=("canonical", "routing"),
        default="canonical",
    )
    prepare_routing = sub.add_parser("prepare-routing-data")
    prepare_routing.add_argument(
        "--source-data-dir", type=Path, required=True
    )
    prepare_routing.add_argument("--output-dir", type=Path, required=True)
    prepare_routing.add_argument("--train-size", type=int, default=4096)
    prepare_routing.add_argument(
        "--validation-size", type=int, default=512
    )
    prepare_routing.add_argument("--extra-split", action="append")
    prepare_routing.add_argument("--extra-size", type=int, default=64)
    entity_window = sub.add_parser("diagnose-entity-window")
    entity_window.add_argument("--checkpoint", type=Path, required=True)
    entity_window.add_argument("--cache-dir", type=Path, required=True)
    entity_window.add_argument("--data-dir", type=Path, required=True)
    entity_window.add_argument(
        "--supervision-dir", type=Path, required=True
    )
    entity_window.add_argument("--output", type=Path, required=True)
    entity_window.add_argument(
        "--window", type=int, action="append", required=True
    )
    entity_window.add_argument(
        "--pair-gain-threshold",
        type=float,
        action="append",
        default=None,
    )
    entity_window.add_argument("--split", default="validation")
    entity_window.add_argument(
        "--family-filter", choices=("canonical", "routing")
    )
    entity_window.add_argument("--batch-size", type=int, default=16)
    entity_window.add_argument("--device", default="cuda")
    cross_domain = sub.add_parser("cross-domain-probe")
    cross_domain.add_argument("--checkpoint", type=Path, required=True)
    cross_domain.add_argument("--cache-dir", type=Path, required=True)
    cross_domain.add_argument("--data-dir", type=Path, required=True)
    cross_domain.add_argument("--output", type=Path, required=True)
    cross_domain.add_argument("--split", action="append", required=True)
    cross_domain.add_argument("--batch-size", type=int, default=16)
    cross_domain.add_argument("--device", default="cuda")
    joint_bundle = sub.add_parser("prepare-joint-bundle")
    joint_bundle.add_argument(
        "--canonical-cache-dir", type=Path, required=True
    )
    joint_bundle.add_argument(
        "--canonical-data-dir", type=Path, required=True
    )
    joint_bundle.add_argument(
        "--routing-cache-dir", type=Path, required=True
    )
    joint_bundle.add_argument(
        "--routing-data-dir", type=Path, required=True
    )
    joint_bundle.add_argument(
        "--output-cache-dir", type=Path, required=True
    )
    joint_bundle.add_argument(
        "--output-data-dir", type=Path, required=True
    )
    joint_bundle.add_argument(
        "--canonical-train-size", type=int, default=1024
    )
    joint_bundle.add_argument(
        "--canonical-validation-size", type=int, default=256
    )
    joint_bundle.add_argument("--shard-size", type=int, default=64)
    interpolate = sub.add_parser("interpolate-checkpoint")
    interpolate.add_argument(
        "--left-checkpoint", type=Path, required=True
    )
    interpolate.add_argument(
        "--right-checkpoint", type=Path, required=True
    )
    interpolate.add_argument("--output", type=Path, required=True)
    interpolate.add_argument(
        "--right-weight", type=float, required=True
    )
    interpolate.add_argument("--prefix", action="append", required=True)
    operation_threshold = sub.add_parser(
        "diagnose-operation-threshold"
    )
    operation_threshold.add_argument(
        "--checkpoint", type=Path, required=True
    )
    operation_threshold.add_argument(
        "--cache-dir", type=Path, required=True
    )
    operation_threshold.add_argument(
        "--data-dir", type=Path, required=True
    )
    operation_threshold.add_argument("--output", type=Path, required=True)
    operation_threshold.add_argument(
        "--split", action="append", required=True
    )
    operation_threshold.add_argument(
        "--recurrent-threshold",
        type=float,
        action="append",
    )
    operation_threshold.add_argument(
        "--drop-threshold", type=float, action="append"
    )
    operation_threshold.add_argument(
        "--crossing-threshold", type=float, default=16.1
    )
    operation_threshold.add_argument("--batch-size", type=int, default=16)
    operation_threshold.add_argument("--device", default="cuda")
    entity_capacity = sub.add_parser(
        "diagnose-entity-capacity-threshold"
    )
    entity_capacity.add_argument(
        "--checkpoint", type=Path, required=True
    )
    entity_capacity.add_argument(
        "--cache-dir", type=Path, required=True
    )
    entity_capacity.add_argument(
        "--data-dir", type=Path, required=True
    )
    entity_capacity.add_argument("--output", type=Path, required=True)
    entity_capacity.add_argument(
        "--split", action="append", required=True
    )
    entity_capacity.add_argument(
        "--capacity-threshold", type=float, action="append", required=True
    )
    entity_capacity.add_argument(
        "--family-filter", choices=("canonical", "routing")
    )
    entity_capacity.add_argument("--batch-size", type=int, default=16)
    entity_capacity.add_argument("--device", default="cuda")
    k1_train = sub.add_parser("k1-train")
    k1_train.add_argument("--cache-dir", type=Path, required=True)
    k1_train.add_argument("--data-dir", type=Path, required=True)
    k1_train.add_argument("--output-dir", type=Path, required=True)
    k1_train.add_argument("--model-seed", type=int, required=True)
    k1_train.add_argument("--steps", type=int, default=4000)
    k1_train.add_argument("--batch-size", type=int, default=16)
    k1_train.add_argument("--learning-rate", type=float, default=3e-4)
    k1_train.add_argument("--validation-interval", type=int, default=200)
    k1_train.add_argument("--validation-per-cell", type=int, default=2)
    k1_train.add_argument("--early-stop-patience", type=int, default=8)
    k1_train.add_argument("--device", default="cuda")
    k1_train.add_argument("--overfit", action="store_true")
    k1_train.add_argument("--initial-checkpoint", type=Path)
    k1_evaluate = sub.add_parser("k1-evaluate")
    k1_evaluate.add_argument("--checkpoint", type=Path, required=True)
    k1_evaluate.add_argument("--cache-dir", type=Path, required=True)
    k1_evaluate.add_argument("--data-dir", type=Path, required=True)
    k1_evaluate.add_argument("--output", type=Path, required=True)
    k1_evaluate.add_argument("--batch-size", type=int, default=16)
    k1_evaluate.add_argument("--device", default="cuda")
    assess = sub.add_parser("assess")
    assess.add_argument(
        "--canonical-matrix", type=Path, required=True
    )
    assess.add_argument("--hidden-causal", type=Path, required=True)
    assess.add_argument(
        "--routing-probe", type=Path, action="append", required=True
    )
    assess.add_argument(
        "--canonical-preflight", type=Path, required=True
    )
    assess.add_argument("--routing-preflight", type=Path, required=True)
    assess.add_argument("--routing-manifest", type=Path, required=True)
    assess.add_argument("--k1-results", type=Path, required=True)
    assess.add_argument(
        "--optimization-path-result",
        type=Path,
        action="append",
        required=True,
    )
    assess.add_argument("--training-budget-audit", type=Path)
    assess.add_argument("--cost-audit", type=Path)
    assess.add_argument(
        "--minimum-online-examples-per-family", type=int, default=64
    )
    assess.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-routing-data":
        result = prepare_a121p_routing_data(
            args.source_data_dir,
            args.output_dir,
            train_size=args.train_size,
            validation_size=args.validation_size,
            extra_splits=args.extra_split or (),
            extra_size=args.extra_size,
        )
    elif args.command == "diagnose-entity-capacity-threshold":
        result = diagnose_a121p_entity_capacity_threshold(
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            output_path=args.output,
            splits=args.split,
            capacity_thresholds=args.capacity_threshold,
            family_filter=args.family_filter,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "diagnose-operation-threshold":
        result = diagnose_a121p_operation_threshold(
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            output_path=args.output,
            splits=args.split,
            recurrent_thresholds=args.recurrent_threshold or (),
            drop_thresholds=args.drop_threshold or (),
            crossing_threshold=args.crossing_threshold,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "interpolate-checkpoint":
        result = interpolate_a121p_boundary_checkpoints(
            left_checkpoint=args.left_checkpoint,
            right_checkpoint=args.right_checkpoint,
            output_path=args.output,
            right_weight=args.right_weight,
            prefixes=args.prefix,
        )
    elif args.command == "prepare-joint-bundle":
        result = prepare_a121p_joint_training_bundle(
            canonical_cache_dir=args.canonical_cache_dir,
            canonical_data_dir=args.canonical_data_dir,
            routing_cache_dir=args.routing_cache_dir,
            routing_data_dir=args.routing_data_dir,
            output_cache_dir=args.output_cache_dir,
            output_data_dir=args.output_data_dir,
            canonical_train_size=args.canonical_train_size,
            canonical_validation_size=args.canonical_validation_size,
            shard_size=args.shard_size,
        )
    elif args.command == "cross-domain-probe":
        result = run_a121p_cross_domain_probe(
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            output_path=args.output,
            splits=args.split,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "diagnose-entity-window":
        result = diagnose_a121p_entity_section_window(
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            supervision_dir=args.supervision_dir,
            output_path=args.output,
            windows=args.window,
            pair_gain_thresholds=(
                args.pair_gain_threshold
                if args.pair_gain_threshold is not None
                else (5.0,)
            ),
            split=args.split,
            family_filter=args.family_filter,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "current-family-preflight":
        result = run_a121p_current_family_preflight(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_path=args.output,
            splits=args.split,
            examples_per_split=args.examples_per_split,
            device=args.device,
            max_input_tokens=args.max_input_tokens,
            max_wall_seconds=args.max_wall_seconds,
            task_variant=args.task_variant,
        )
    elif args.command == "k1-train":
        result = train_a121p_k1(
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            spec=A121PK1TrainSpec(
                model_seed=args.model_seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                validation_interval=args.validation_interval,
                validation_per_cell=args.validation_per_cell,
                early_stop_patience=args.early_stop_patience,
            ),
            device=args.device,
            overfit=args.overfit,
            initial_checkpoint=args.initial_checkpoint,
        )
    elif args.command == "k1-evaluate":
        result = evaluate_a121p_k1_formal(
            checkpoint_path=args.checkpoint,
            cache_dir=args.cache_dir,
            data_dir=args.data_dir,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        result = assess_a121p_stage(
            canonical_matrix_path=args.canonical_matrix,
            hidden_causal_path=args.hidden_causal,
            routing_probe_paths=args.routing_probe,
            canonical_preflight_path=args.canonical_preflight,
            routing_preflight_path=args.routing_preflight,
            routing_manifest_path=args.routing_manifest,
            k1_results_path=args.k1_results,
            optimization_path_results=args.optimization_path_result,
            training_budget_audit_path=args.training_budget_audit,
            cost_audit_path=args.cost_audit,
            minimum_online_examples_per_family=(
                args.minimum_online_examples_per_family
            ),
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
