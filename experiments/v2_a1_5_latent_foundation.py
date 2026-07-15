from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from yggdrasil_v2.reasoning_medium.a1_5_data import (  # noqa: E402
    A15DatasetSpec,
    build_a15_dataset,
    counterfactual_report,
    evaluate_weak_baselines,
    load_a15_records,
)
from yggdrasil_v2.reasoning_medium.a1_5_model import P0Config  # noqa: E402
from yggdrasil_v2.reasoning_medium.a1_5_p1 import P1HiddenLatentReasoner, cache_qwen_hidden  # noqa: E402
from yggdrasil_v2.reasoning_medium.a1_5_p1_train import (  # noqa: E402
    _cache_records,
    evaluate_p1,
    evaluate_p1_diagnostics,
    load_p1_checkpoint,
    train_p1_cached,
)
from yggdrasil_v2.reasoning_medium.a1_5_p2 import LearnedMultiSlotWorkspace, P2Config  # noqa: E402
from yggdrasil_v2.reasoning_medium.a1_5_p2_train import (  # noqa: E402
    load_p2_checkpoint,
    load_p2_items,
    run_p2_interventions,
    train_p2_cached,
)
from yggdrasil_v2.reasoning_medium.a1_5_text_baseline import run_a15_text_baseline  # noqa: E402
from yggdrasil_v2.reasoning_medium.a1_5_train import (  # noqa: E402
    evaluate_interventions,
    load_p0_checkpoint,
    train_p0,
)


DEFAULT_DATA = Path("artifacts/v2-a/a1_5/data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-Yggdrasil V2-A1.5 latent foundation experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data", help="build A1.5 necessary-step data and weak-baseline manifest")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--seed", type=int, default=20260713)
    prepare.add_argument("--train-size", type=int, default=4096)
    prepare.add_argument("--validation-size", type=int, default=256)
    prepare.add_argument("--test-size", type=int, default=256)
    prepare.add_argument("--composition-size", type=int, default=256)
    prepare.add_argument("--length-size", type=int, default=256)

    p0 = sub.add_parser("p0", help="train the structured-input P0 positive control")
    p0.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p0.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1_5/p0"))
    p0.add_argument("--device", default="cuda")
    p0.add_argument("--seed", type=int, default=20260713)
    p0.add_argument("--steps", type=int, default=3000)
    p0.add_argument("--batch-size", type=int, default=32)
    p0.add_argument("--learning-rate", type=float, default=3e-4)
    p0.add_argument("--state-weight", type=float, default=1.0)
    p0.add_argument("--latent-width", type=int, default=256)
    p0.add_argument("--attention-heads", type=int, default=8)
    p0.add_argument("--ffn-width", type=int, default=512)
    p0.add_argument("--overfit", action="store_true", help="run only the 32-example overfit gate")
    p0.add_argument("--checkpoint-interval", type=int, default=100)

    intervene = sub.add_parser("p0-intervene", help="run P0 causal interventions from best.pt")
    intervene.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    intervene.add_argument("--checkpoint", type=Path, default=Path("artifacts/v2-a/a1_5/p0/best.pt"))
    intervene.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_5/p0/interventions.json"))
    intervene.add_argument("--device", default="cuda")

    cache = sub.add_parser("p1-cache", help="cache frozen Qwen hidden states with operation span masks")
    cache.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    cache.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1_5/p1-hidden-cache"))
    cache.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    cache.add_argument("--revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--max-length", type=int, default=512)
    cache.add_argument("--shard-size", type=int, default=64)

    p1 = sub.add_parser("p1-train", help="train P1 from an existing hidden cache")
    p1.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p1.add_argument("--cache-dir", type=Path, default=Path("artifacts/v2-a/a1_5/p1-hidden-cache"))
    p1.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1_5/p1"))
    p1.add_argument("--device", default="cuda")
    p1.add_argument("--steps", type=int, default=500)
    p1.add_argument("--batch-size", type=int, default=8)
    p1.add_argument("--learning-rate", type=float, default=3e-4)
    p1.add_argument("--warmup-steps", type=int, default=500)
    p1.add_argument("--validation-interval", type=int, default=100)
    p1.add_argument("--early-stop-patience", type=int, default=8)
    p1.add_argument("--warmup-gate-threshold", type=float, default=0.95)
    p1.add_argument("--no-stop-on-warmup-failure", action="store_true")

    p1eval = sub.add_parser("p1-evaluate", help="evaluate a saved P1 best checkpoint on all A1.5 splits")
    p1eval.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p1eval.add_argument("--cache-dir", type=Path, default=Path("artifacts/v2-a/a1_5/p1-hidden-cache"))
    p1eval.add_argument("--checkpoint", type=Path, default=Path("artifacts/v2-a/a1_5/p1-final/best.pt"))
    p1eval.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_5/p1-final/eval-all.json"))
    p1eval.add_argument("--device", default="cuda")
    p1eval.add_argument("--batch-size", type=int, default=64)

    p1smoke = sub.add_parser("p1-smoke", help="run P1 interface smoke on synthetic hidden states")
    p1smoke.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_5/p1/smoke.json"))
    p1smoke.add_argument("--device", default="cuda")

    p2smoke = sub.add_parser("p2-smoke", help="run guarded P2 learned-slot interface smoke")
    p2smoke.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_5/p2/smoke.json"))
    p2smoke.add_argument("--device", default="cuda")

    p2 = sub.add_parser("p2-train", help="train P2 learned K=8 workspace from frozen hidden cache")
    p2.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p2.add_argument("--cache-dir", type=Path, default=Path("artifacts/v2-a/a1_5/p1-hidden-cache"))
    p2.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1_5/p2-final"))
    p2.add_argument("--device", default="cuda")
    p2.add_argument("--steps", type=int, default=3000)
    p2.add_argument("--batch-size", type=int, default=64)
    p2.add_argument("--learning-rate", type=float, default=3e-4)
    p2.add_argument("--validation-interval", type=int, default=100)
    p2.add_argument("--early-stop-patience", type=int, default=8)
    p2.add_argument("--latent-width", type=int, default=256)
    p2.add_argument("--attention-heads", type=int, default=8)
    p2.add_argument("--ffn-width", type=int, default=512)

    p2intervene = sub.add_parser("p2-intervene", help="run P2 learned-slot causal interventions")
    p2intervene.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p2intervene.add_argument("--cache-dir", type=Path, default=Path("artifacts/v2-a/a1_5/p1-hidden-cache"))
    p2intervene.add_argument("--checkpoint", type=Path, default=Path("artifacts/v2-a/a1_5/p2-final/best.pt"))
    p2intervene.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_5/p2-final/interventions.json"))
    p2intervene.add_argument("--device", default="cuda")
    p2intervene.add_argument("--split", default="test", choices=("validation", "test", "composition_heldout", "length_heldout"))
    p2intervene.add_argument("--batch-size", type=int, default=64)

    text = sub.add_parser("text-baseline", help="matched A1.5 visible text-CoT baseline")
    text.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    text.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_5/text-baseline.json"))
    text.add_argument("--split", default="test", choices=("validation", "test", "composition_heldout", "length_heldout"))
    text.add_argument("--max-examples", type=int, default=128)
    text.add_argument("--shots", type=int, choices=(0, 2), default=0)
    text.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    text.add_argument("--revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
    text.add_argument("--device", default="cuda")
    text.add_argument("--max-input-tokens", type=int, default=512)
    text.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="optional transparent per-example safety timeout; does not cap output tokens",
    )

    return parser


def _records(data_dir: Path, split: str) -> list[dict]:
    return load_a15_records(data_dir, split)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        result = build_a15_dataset(
            args.output_dir,
            A15DatasetSpec(
                seed=args.seed,
                train_size=args.train_size,
                validation_size=args.validation_size,
                test_size=args.test_size,
                composition_size=args.composition_size,
                length_size=args.length_size,
            ),
        )
    elif args.command == "p0":
        if args.overfit:
            train = _records(args.data_dir, "train")[:32]
            validation = _records(args.data_dir, "validation")[:32]
        else:
            train = _records(args.data_dir, "train")
            validation = _records(args.data_dir, "validation")
        config = P0Config(
            latent_width=args.latent_width,
            attention_heads=args.attention_heads,
            ffn_width=args.ffn_width,
        )
        result = train_p0(
            train,
            validation,
            args.output_dir,
            config=config,
            seed=args.seed,
            device=args.device,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            state_weight=args.state_weight,
            early_stop_overfit=args.overfit,
            checkpoint_interval=args.checkpoint_interval,
        )
        best = load_p0_checkpoint(args.output_dir / "best.pt", args.device)
        result["weak_baselines"] = {
            split: evaluate_weak_baselines(_records(args.data_dir, split), train)
            for split in ("validation", "test", "composition_heldout", "length_heldout")
        }
        result["counterfactual"] = {
            split: counterfactual_report(_records(args.data_dir, split))
            for split in ("test", "composition_heldout", "length_heldout")
        }
        if not args.overfit:
            result["causal_interventions"] = {
                split: evaluate_interventions(best, _records(args.data_dir, split), args.device)
                for split in ("test", "composition_heldout", "length_heldout")
            }
        (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "p0-intervene":
        model = load_p0_checkpoint(args.checkpoint, args.device)
        result = {
            "checkpoint": str(args.checkpoint),
            "test": evaluate_interventions(model, _records(args.data_dir, "test"), args.device),
            "composition_heldout": evaluate_interventions(
                model, _records(args.data_dir, "composition_heldout"), args.device
            ),
            "length_heldout": evaluate_interventions(model, _records(args.data_dir, "length_heldout"), args.device),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "p1-cache":
        result = cache_qwen_hidden(
            args.data_dir,
            args.output_dir,
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            max_length=args.max_length,
            shard_size=args.shard_size,
        )
    elif args.command == "p1-train":
        train_records = _records(args.data_dir, "train")
        validation_records = _records(args.data_dir, "validation")
        train_items = _cache_records(args.cache_dir, "train", train_records)
        validation_items = _cache_records(args.cache_dir, "validation", validation_records)
        result = train_p1_cached(
            train_items,
            validation_items,
            args.output_dir,
            device=args.device,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            validation_interval=args.validation_interval,
            early_stop_patience=args.early_stop_patience,
            warmup_gate_threshold=args.warmup_gate_threshold,
            stop_on_warmup_failure=not args.no_stop_on_warmup_failure,
        )
    elif args.command == "p1-evaluate":
        model = load_p1_checkpoint(args.checkpoint, args.device)
        result = {
            "schema_version": "yggdrasil.v2-a1.5.p1.eval.v1",
            "evidence_level": "surrogate-probe",
            "checkpoint": str(args.checkpoint),
            "integrity": model.integrity_report(),
            "splits": {},
        }
        for split in ("train", "validation", "test", "composition_heldout", "length_heldout"):
            records = _records(args.data_dir, split)
            items = _cache_records(args.cache_dir, split, records)
            result["splits"][split] = {
                **evaluate_p1(model, items, args.device, args.batch_size),
                "diagnostics": evaluate_p1_diagnostics(model, items, args.device, args.batch_size),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "p1-smoke":
        device = args.device
        torch = __import__("torch")
        model = P1HiddenLatentReasoner(12, P0Config(latent_width=32, attention_heads=4, ffn_width=64)).to(device)
        source = torch.randn(2, 8, 12, device=device)
        source_mask = torch.ones(2, 8, dtype=torch.bool, device=device)
        operation_mask = torch.zeros(2, 2, 8, dtype=torch.bool, device=device)
        operation_mask[:, 0, :3] = True
        operation_mask[:, 1, 3:6] = True
        step_mask = torch.ones(2, 2, dtype=torch.bool, device=device)
        output = model(source, source_mask, operation_mask, step_mask, return_trajectory=True)
        result = {
            "schema_version": "yggdrasil.v2-a1.5.p1.smoke.v1",
            "evidence_level": "mechanism-smoke",
            "quality": "not measured; synthetic hidden only",
            "logits_shape": list(output["logits"].shape),
            "integrity": model.integrity_report(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.command == "p2-smoke":
        torch = __import__("torch")
        device = args.device
        model = LearnedMultiSlotWorkspace(12, P2Config(learned_slots=8, latent_width=32, attention_heads=4, ffn_width=64)).to(device)
        operation_hidden = torch.randn(2, 3, 12, device=device)
        operation_mask = torch.ones(2, 3, dtype=torch.bool, device=device)
        result = {
            "schema_version": "yggdrasil.v2-a1.5.p2.smoke.v1",
            "evidence_level": "mechanism-smoke",
            "quality": "not measured; synthetic hidden only",
            "permutation_probe": model.permutation_probe(operation_hidden, operation_mask),
            "integrity": model.integrity_report(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.command == "p2-train":
        train_items = load_p2_items(args.cache_dir, args.data_dir, "train")
        validation_items = load_p2_items(args.cache_dir, args.data_dir, "validation")
        result = train_p2_cached(
            train_items,
            validation_items,
            args.output_dir,
            config=P2Config(
                learned_slots=8,
                latent_width=args.latent_width,
                attention_heads=args.attention_heads,
                ffn_width=args.ffn_width,
            ),
            device=args.device,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_interval=args.validation_interval,
            early_stop_patience=args.early_stop_patience,
        )
    elif args.command == "p2-intervene":
        model = load_p2_checkpoint(args.checkpoint, args.device)
        items = load_p2_items(args.cache_dir, args.data_dir, args.split)
        result = run_p2_interventions(model, items, args.device, batch_size=args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.command == "text-baseline":
        result = run_a15_text_baseline(
            data_dir=args.data_dir,
            output_path=args.output,
            split=args.split,
            max_examples=args.max_examples,
            shots=args.shots,
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            max_input_tokens=args.max_input_tokens,
            max_wall_seconds=args.max_wall_seconds,
        )
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
