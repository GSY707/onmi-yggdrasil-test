from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from yggdrasil_v2.reasoning_medium.a1_6_core import A16CoreConfig  # noqa: E402
from yggdrasil_v2.reasoning_medium.a1_6_data import (  # noqa: E402
    A16DatasetSpec,
    assert_data_gate,
    audit_a16_data,
    build_a16_dataset,
    load_a16_records,
)
from yggdrasil_v2.reasoning_medium.a1_6_qwen import QWEN_MODEL_ID, QWEN_REVISION, FrozenQwenBoundary, cache_qwen_boundary  # noqa: E402
from yggdrasil_v2.reasoning_medium.a1_6_qwen_train import load_cache_items, train_c1  # noqa: E402
from yggdrasil_v2.reasoning_medium.a1_6_train import (  # noqa: E402
    evaluate_core,
    load_c0_checkpoint,
    run_c0_interventions,
    train_c0,
)


DEFAULT_DATA = Path("artifacts/v2-a/a1_6/data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-Yggdrasil V2-A1.6 relation-addressed latent core")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--seed", type=int, default=20260715)
    for name, default in (("train", 8192), ("validation", 512), ("test", 512), ("length", 512), ("relation", 512), ("causal", 512)):
        prepare.add_argument(f"--{name}-size", type=int, default=default)

    audit = sub.add_parser("audit-data")
    audit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    audit.add_argument("--output", type=Path, default=Path("artifacts/v2-a/a1_6/data-audit.json"))

    train = sub.add_parser("c0-train")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    train.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1_6/c0-formal"))
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

    evaluate = sub.add_parser("c0-evaluate")
    evaluate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=128)

    intervene = sub.add_parser("c0-intervene")
    intervene.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    intervene.add_argument("--split", default="causal_core", choices=("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core"))
    intervene.add_argument("--checkpoint", type=Path, required=True)
    intervene.add_argument("--output", type=Path, required=True)
    intervene.add_argument("--device", default="cuda")
    intervene.add_argument("--batch-size", type=int, default=128)

    cache = sub.add_parser("c1-cache")
    cache.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    cache.add_argument("--output-dir", type=Path, default=Path("artifacts/v2-a/a1_6/c1-cache"))
    cache.add_argument("--model-id", default=QWEN_MODEL_ID)
    cache.add_argument("--revision", default=QWEN_REVISION)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--max-length", type=int, default=512)
    cache.add_argument("--max-train-examples", type=int, default=None)
    cache.add_argument("--shard-size", type=int, default=64)

    c1train = sub.add_parser("c1-train")
    c1train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    c1train.add_argument("--cache-dir", type=Path, required=True)
    c1train.add_argument("--core-checkpoint", type=Path, required=True)
    c1train.add_argument("--boundary-checkpoint", type=Path, default=None)
    c1train.add_argument("--output-dir", type=Path, required=True)
    c1train.add_argument("--device", default="cuda")
    c1train.add_argument("--seed", type=int, default=20260715)
    c1train.add_argument("--phase", choices=("boundary", "joint"), required=True)
    c1train.add_argument("--steps", type=int, default=3000)
    c1train.add_argument("--batch-size", type=int, default=64)
    c1train.add_argument("--boundary-learning-rate", type=float, default=3e-4)
    c1train.add_argument("--core-learning-rate", type=float, default=3e-5)
    c1train.add_argument("--validation-interval", type=int, default=100)
    c1train.add_argument("--early-stop-patience", type=int, default=10)

    c1eval = sub.add_parser("c1-evaluate")
    c1eval.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    c1eval.add_argument("--cache-dir", type=Path, required=True)
    c1eval.add_argument("--checkpoint", type=Path, required=True)
    c1eval.add_argument("--output", type=Path, required=True)
    c1eval.add_argument("--device", default="cuda")
    c1eval.add_argument("--batch-size", type=int, default=64)
    return parser


def _records(data_dir: Path, split: str, limit: int = 0) -> list[dict]:
    records = load_a16_records(data_dir, split)
    return records[:limit] if limit else records


def _assert_c0_gate(evaluation: dict[str, Any]) -> None:
    splits = evaluation["splits"]
    gates = {
        "ordinary_test_state_full_exact": splits["test"]["state_full_exact"] >= 0.995,
        "length_state_full_exact": splits["length_heldout"]["state_full_exact"] >= 0.95,
        "relation_state_full_exact": splits["relation_heldout"]["state_full_exact"] >= 0.95,
        "source_pointer_accuracy": min(splits[name]["source_pointer_accuracy"] for name in splits) >= 0.995,
        "target_pointer_accuracy": min(splits[name]["target_pointer_accuracy"] for name in splits) >= 0.995,
        "no_final_state_decoupling": all(splits[name]["final_answer_accuracy"] <= splits[name]["state_full_exact"] + 0.02 for name in splits),
    }
    evaluation["gates"] = gates
    evaluation["passed"] = all(gates.values())


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        result = build_a16_dataset(args.output_dir, A16DatasetSpec(seed=args.seed, train_size=args.train_size, validation_size=args.validation_size, test_size=args.test_size, length_size=args.length_size, relation_size=args.relation_size, causal_size=args.causal_size))
    elif args.command == "audit-data":
        result = audit_a16_data(args.data_dir, args.output)
        assert_data_gate(result)
    elif args.command == "c0-train":
        report = audit_a16_data(args.data_dir)
        assert_data_gate(report)
        train_records = _records(args.data_dir, "train", args.train_limit)
        validation_records = _records(args.data_dir, "validation", args.validation_limit)
        overfit = bool(args.train_limit)
        result = train_c0(train_records, validation_records, args.output_dir, config=A16CoreConfig(), seed=args.seed, device=args.device, steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate, validation_interval=args.validation_interval, early_stop_exact=args.early_stop_exact if overfit else None, early_stop_patience=args.early_stop_patience if not overfit else None, overfit_mode=overfit)
        if overfit:
            fit_metrics = result["fit"]
            validation_metrics = result["validation"]
            result["overfit_gate_fit"] = {key: fit_metrics[key] >= 0.999 for key in ("final_answer_accuracy", "state_token_accuracy", "state_full_exact", "source_pointer_accuracy", "target_pointer_accuracy")}
            result["few_shot_validation_diagnostic"] = validation_metrics
            result["overfit_gate"] = result["overfit_gate_fit"]
            (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if not all(result["overfit_gate"].values()):
                raise RuntimeError("C0 overfit32 Gate failed; stopping A1.6 before formal C0")
    elif args.command == "c0-evaluate":
        model = load_c0_checkpoint(args.checkpoint, args.device)
        result = {"schema_version": "yggdrasil.v2-a1.6.c0.eval.v1", "checkpoint": str(args.checkpoint), "integrity": model.integrity_report(), "splits": {}}
        for split in ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core"):
            result["splits"][split] = evaluate_core(model, _records(args.data_dir, split), args.device, batch_size=args.batch_size)
        _assert_c0_gate(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not result["passed"]:
            raise RuntimeError("C0 Gate failed; C1 is forbidden")
    elif args.command == "c0-intervene":
        model = load_c0_checkpoint(args.checkpoint, args.device)
        result = {"schema_version": "yggdrasil.v2-a1.6.c0.interventions.v1", "split": args.split, "checkpoint": str(args.checkpoint), "results": run_c0_interventions(model, _records(args.data_dir, args.split), args.device, batch_size=args.batch_size)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "c1-cache":
        result = cache_qwen_boundary(args.data_dir, args.output_dir, model_id=args.model_id, revision=args.revision, device=args.device, max_length=args.max_length, max_train_examples=args.max_train_examples, shard_size=args.shard_size)
    elif args.command == "c1-train":
        core = load_c0_checkpoint(args.core_checkpoint, args.device)
        train_items = load_cache_items(args.cache_dir, "train", args.data_dir)
        validation_items = load_cache_items(args.cache_dir, "validation", args.data_dir)
        result = train_c1(train_items, validation_items, core, args.output_dir, phase=args.phase, device=args.device, seed=args.seed, steps=args.steps, batch_size=args.batch_size, boundary_learning_rate=args.boundary_learning_rate, core_learning_rate=args.core_learning_rate, validation_interval=args.validation_interval, early_stop_patience=args.early_stop_patience)
    elif args.command == "c1-evaluate":
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        core = __import__("yggdrasil_v2.reasoning_medium.a1_6_core", fromlist=["RelationAddressedCore"]).RelationAddressedCore(A16CoreConfig(**checkpoint["core_config"]))
        model = FrozenQwenBoundary(checkpoint["source_width"], core).to(args.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        result = {"schema_version": "yggdrasil.v2-a1.6.c1.eval.v1", "checkpoint": str(args.checkpoint), "integrity": model.integrity_report(), "splits": {}}
        for split in ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core"):
            items = load_cache_items(args.cache_dir, split, args.data_dir)
            total = query_correct = source_correct = target_correct = pointer_total = 0
            for start in range(0, len(items), args.batch_size):
                inputs, records = __import__("yggdrasil_v2.reasoning_medium.a1_6_qwen_train", fromlist=["_batch_items", "_labels"])._batch_items(items[start : start + args.batch_size], args.device)
                labels = __import__("yggdrasil_v2.reasoning_medium.a1_6_qwen_train", fromlist=["_labels"])._labels(records, args.device)
                with torch.no_grad():
                    out = model(inputs["start"], inputs["query"], inputs["family"], inputs["source"], inputs["target"], inputs["mask"])
                query_correct += int((out["query_pointer_logits"].argmax(-1) == labels["query"]).sum())
                source_correct += int(((out["source_pointer_logits"].argmax(-1) == labels["source"]) & inputs["mask"]).sum())
                target_correct += int(((out["target_pointer_logits"].argmax(-1) == labels["target"]) & inputs["mask"]).sum())
                pointer_total += int(inputs["mask"].sum())
                total += len(records)
            result["splits"][split] = {"query_pointer_accuracy": query_correct / max(1, total), "source_pointer_accuracy": source_correct / max(1, pointer_total), "target_pointer_accuracy": target_correct / max(1, pointer_total), "examples": total}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
