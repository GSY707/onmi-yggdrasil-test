from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omni_transformer_stage_ave_bidirectional_latent_external import (  # noqa: E402
    ANSWER_CLASSES,
    OPS,
    SEQ_LEN,
    TOKEN_VOCAB,
    ZONES,
    encode_external,
    make_example,
)


SPLITS = ("train", "val", "test", "heldout")
TOKENS_PER_EXAMPLE = SEQ_LEN * 2 + 1
SPLIT_SEED_OFFSETS = {"train": 0, "val": 100_000, "test": 200_000, "heldout": 300_000}


@dataclass(frozen=True)
class AVFScale:
    train_examples: int
    val_examples: int
    test_examples: int
    heldout_examples: int
    shard_size: int


SCALES = {
    "smoke": AVFScale(train_examples=512, val_examples=128, test_examples=128, heldout_examples=128, shard_size=128),
    "10m": AVFScale(train_examples=160_000, val_examples=8_000, test_examples=8_000, heldout_examples=8_000, shard_size=10_000),
    "100m": AVFScale(train_examples=1_600_000, val_examples=80_000, test_examples=80_000, heldout_examples=80_000, shard_size=20_000),
    "1b": AVFScale(train_examples=16_000_000, val_examples=800_000, test_examples=800_000, heldout_examples=800_000, shard_size=50_000),
}


@dataclass(frozen=True)
class AVFDatasetConfig:
    scale: str = "10m"
    train_examples: int = SCALES["10m"].train_examples
    val_examples: int = SCALES["10m"].val_examples
    test_examples: int = SCALES["10m"].test_examples
    heldout_examples: int = SCALES["10m"].heldout_examples
    shard_size: int = SCALES["10m"].shard_size
    seed: int = 20260705
    tokens_per_example: int = TOKENS_PER_EXAMPLE
    distribution_note: str = (
        "Stage AV-F is the first executable 10M-level crop of the AV-D 1B-first design. "
        "It keeps the AV-E source external <-> latent <-> target external mutual-translation task, "
        "with source tokens, target tokens, and answer labels stored as tensor shards."
    )


def scale_defaults(scale: str) -> AVFScale:
    try:
        return SCALES[scale]
    except KeyError as exc:
        raise SystemExit(f"unknown scale {scale!r}; choose one of {', '.join(SCALES)}") from exc


def split_count(config: AVFDatasetConfig, split: str) -> int:
    return {
        "train": config.train_examples,
        "val": config.val_examples,
        "test": config.test_examples,
        "heldout": config.heldout_examples,
    }[split]


def write_split(split: str, count: int, config: AVFDatasetConfig, output_dir: Path) -> dict[str, object]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed + SPLIT_SEED_OFFSETS[split])
    shards = []
    op_counts = {str(op): 0 for op in range(OPS)}
    zone_counts = {str(zone): 0 for zone in range(ZONES)}
    answer_counts = {str(answer): 0 for answer in range(ANSWER_CLASSES)}
    remaining = count
    written = 0
    shard_index = 0
    while remaining > 0:
        shard_count = min(config.shard_size, remaining)
        source_rows: list[list[int]] = []
        target_rows: list[list[int]] = []
        answers: list[int] = []
        ops: list[int] = []
        target_zones: list[int] = []
        for row in range(shard_count):
            global_index = written + row
            example = make_example(global_index, rng, split)
            source_rows.append(
                encode_external(
                    example.source_image,
                    example.source_tool,
                    example.source_memory,
                    example.target_zone,
                    example.op,
                    example.op_arg,
                )
            )
            target_rows.append(
                encode_external(
                    example.target_image,
                    example.target_tool,
                    example.target_memory,
                    example.target_zone,
                    example.op,
                    example.op_arg,
                )
            )
            answers.append(example.answer)
            ops.append(example.op)
            target_zones.append(example.target_zone)
            op_counts[str(example.op)] += 1
            zone_counts[str(example.target_zone)] += 1
            answer_counts[str(example.answer)] += 1
        shard_path = split_dir / f"shard-{shard_index:05d}.pt"
        torch.save(
            {
                "source_tokens": torch.tensor(source_rows, dtype=torch.int16),
                "target_tokens": torch.tensor(target_rows, dtype=torch.int16),
                "answers": torch.tensor(answers, dtype=torch.int16),
                "ops": torch.tensor(ops, dtype=torch.int16),
                "target_zones": torch.tensor(target_zones, dtype=torch.int16),
            },
            shard_path,
        )
        shards.append(
            {
                "path": str(shard_path.relative_to(output_dir)).replace("\\", "/"),
                "examples": shard_count,
                "tokens": shard_count * config.tokens_per_example,
            }
        )
        remaining -= shard_count
        written += shard_count
        shard_index += 1
    return {
        "split": split,
        "examples": count,
        "tokens": count * config.tokens_per_example,
        "shards": shards,
        "op_counts": op_counts,
        "target_zone_counts": zone_counts,
        "answer_counts": answer_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=sorted(SCALES), default="10m")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/dataset_10m"))
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--val-examples", type=int)
    parser.add_argument("--test-examples", type=int)
    parser.add_argument("--heldout-examples", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--seed", type=int, default=AVFDatasetConfig.seed)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AVFDatasetConfig:
    defaults = scale_defaults(args.scale)
    return AVFDatasetConfig(
        scale=args.scale,
        train_examples=args.train_examples if args.train_examples is not None else defaults.train_examples,
        val_examples=args.val_examples if args.val_examples is not None else defaults.val_examples,
        test_examples=args.test_examples if args.test_examples is not None else defaults.test_examples,
        heldout_examples=args.heldout_examples if args.heldout_examples is not None else defaults.heldout_examples,
        shard_size=args.shard_size if args.shard_size is not None else defaults.shard_size,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = [write_split(split, split_count(config, split), config, args.output_dir) for split in SPLITS]
    total_examples = sum(split["examples"] for split in splits)
    total_tokens = sum(split["tokens"] for split in splits)
    manifest = {
        "schema_version": 1,
        "stage": "AV-F",
        "compatible_stage": "AV-E",
        "config": asdict(config),
        "scale": config.scale,
        "path_base": "manifest_dir",
        "vocab_size": TOKEN_VOCAB,
        "answer_classes": ANSWER_CLASSES,
        "seq_len": SEQ_LEN,
        "tokens_per_example": config.tokens_per_example,
        "total_examples": total_examples,
        "total_tokens": total_tokens,
        "unique_train_tokens": config.train_examples * config.tokens_per_example,
        "approx_examples_for_1b_tokens": int(1_000_000_000 // config.tokens_per_example),
        "splits": splits,
        "elapsed_sec": time.perf_counter() - started,
        "interpretation": {
            "route": "AV-D 1B-first design -> directed 10M crop -> AV-E bidirectional external/latent training.",
            "not_a_pass_condition": "Manifest generation only proves data scale and IO readiness; model capability still needs AV-E training metrics.",
            "task_boundary": "External states are compact token sequences, not real pixels/files/actions yet.",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "total_examples": total_examples,
                "total_tokens": total_tokens,
                "unique_train_tokens": manifest["unique_train_tokens"],
                "elapsed_sec": manifest["elapsed_sec"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
