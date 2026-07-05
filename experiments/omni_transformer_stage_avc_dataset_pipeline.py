from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import torch


TASKS = ("visual_rule", "tool_audit", "memory_route", "cross_modal")
SPLITS = ("train", "val", "test", "heldout")
COLORS = 8
ZONES = 8
VALUES = 8
TEXT_LEN = 12
TOOL_LEN = 6
MEMORY_LEN = 8
STATE_LEN = 8
SEQ_TOKENS = 1 + ZONES * 3 + TEXT_LEN + TOOL_LEN + MEMORY_LEN + STATE_LEN + 4
ANSWER_CLASSES = 32

PAD = 0
TOKEN_TASK_BASE = 1
TOKEN_ZONE_BASE = TOKEN_TASK_BASE + len(TASKS)
TOKEN_COLOR_BASE = TOKEN_ZONE_BASE + ZONES
TOKEN_VALUE_BASE = TOKEN_COLOR_BASE + COLORS
TOKEN_TEXT_BASE = TOKEN_VALUE_BASE + VALUES
TOKEN_TOOL_BASE = TOKEN_TEXT_BASE + 64
TOKEN_MEMORY_BASE = TOKEN_TOOL_BASE + 64
TOKEN_STATE_BASE = TOKEN_MEMORY_BASE + 128
TOKEN_VOCAB = TOKEN_STATE_BASE + 128


@dataclass(frozen=True)
class AVCDatasetConfig:
    train_examples: int = 100_000
    val_examples: int = 10_000
    test_examples: int = 10_000
    heldout_examples: int = 10_000
    shard_size: int = 10_000
    seed: int = 20260704
    version: int = 1
    tokens_per_example: int = SEQ_TOKENS
    target_scale_note: str = "1B tokens is about 15.9M examples at 63 tokens/example; generate progressively, do not start by materializing all shards."


def issue_for(html: int, csv: int, pdf: int, screen: int) -> int:
    if html != csv:
        return 0
    if csv > pdf:
        return 1
    if screen != html:
        return 2
    return 3


def pad(values: list[int], length: int) -> list[int]:
    return values[:length] + [PAD] * max(0, length - len(values))


def make_example(index: int, split: str, rng: random.Random) -> tuple[list[int], int, int, int]:
    task = index % len(TASKS)
    if split == "heldout":
        task = (index + 1) % len(TASKS)
    colors = list(range(COLORS))
    rng.shuffle(colors)
    if split == "heldout":
        colors = list(reversed(colors))
    target_zone = rng.randrange(ZONES)
    rule = rng.randrange(VALUES)
    target_color = rng.randrange(COLORS)
    telemetry = rng.randrange(VALUES)
    phase = rng.randrange(VALUES)
    text = [
        TOKEN_ZONE_BASE + target_zone,
        TOKEN_COLOR_BASE + target_color,
        TOKEN_VALUE_BASE + rule,
        TOKEN_VALUE_BASE + ((target_zone + rule) % VALUES),
    ]
    text.extend(TOKEN_TEXT_BASE + rng.randrange(64) for _ in range(TEXT_LEN - len(text)))

    if task == 0:
        answer = (colors[target_zone] + rule) % VALUES
        html = csv = pdf = screen = 0
    elif task == 1:
        issue = index % 4
        if issue == 0:
            html = rng.randrange(VALUES)
            csv = (html + 1 + rng.randrange(VALUES - 1)) % VALUES
            pdf = max(html, csv)
            screen = html
        elif issue == 1:
            html = rng.randrange(1, VALUES)
            csv = html
            pdf = rng.randrange(html)
            screen = html
        elif issue == 2:
            html = rng.randrange(VALUES)
            csv = html
            pdf = max(html, rng.randrange(VALUES))
            screen = (html + 1 + rng.randrange(VALUES - 1)) % VALUES
        else:
            html = rng.randrange(VALUES)
            csv = html
            pdf = max(html, rng.randrange(VALUES))
            screen = html
        answer = 8 + issue_for(html, csv, pdf, screen)
    elif task == 2:
        html = csv = pdf = screen = 0
        target_color = colors[(phase + rule) % ZONES]
        answer = 12 + colors.index(target_color)
    else:
        html = rng.randrange(VALUES)
        csv = (html + colors[target_zone] + rule) % VALUES
        pdf = max(html, csv)
        screen = colors[(target_zone + phase) % ZONES]
        answer = 20 + ((screen + csv + telemetry) % 12)

    image = []
    for zone, color in enumerate(colors):
        image.extend((TOKEN_ZONE_BASE + zone, TOKEN_COLOR_BASE + color, TOKEN_VALUE_BASE + ((zone + color) % VALUES)))
    tools = [
        TOKEN_TOOL_BASE + html,
        TOKEN_TOOL_BASE + 8 + csv,
        TOKEN_TOOL_BASE + 16 + pdf,
        TOKEN_TOOL_BASE + 24 + screen,
        TOKEN_TOOL_BASE + 32 + issue_for(html, csv, pdf, screen),
        TOKEN_TOOL_BASE + 40 + ((html + csv + screen) % VALUES),
    ]
    memory_colors = colors[:]
    if task == 2 and index % 7 == 0:
        z0 = colors.index(target_color)
        z1 = (z0 + 1) % ZONES
        memory_colors[z0], memory_colors[z1] = memory_colors[z1], memory_colors[z0]
    memory = [TOKEN_MEMORY_BASE + zone * (COLORS + 1) + color + 1 for zone, color in enumerate(memory_colors)]
    state = [
        TOKEN_STATE_BASE + telemetry,
        TOKEN_STATE_BASE + 8 + phase,
        TOKEN_STATE_BASE + 16 + target_zone,
        TOKEN_STATE_BASE + 24 + target_color,
    ]
    state.extend(TOKEN_STATE_BASE + 32 + rng.randrange(64) for _ in range(STATE_LEN - len(state)))
    tokens = [TOKEN_TASK_BASE + task] + image + pad(text, TEXT_LEN) + tools + memory + state
    tokens = pad(tokens, SEQ_TOKENS)
    return tokens, answer, task, len(tokens)


def write_split(split: str, count: int, config: AVCDatasetConfig, output_dir: Path) -> dict[str, object]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed + SPLITS.index(split) * 1_000_000)
    shards = []
    remaining = count
    shard_index = 0
    task_counts = {task: 0 for task in TASKS}
    while remaining > 0:
        shard_count = min(config.shard_size, remaining)
        tokens = torch.empty((shard_count, SEQ_TOKENS), dtype=torch.int32)
        answers = torch.empty((shard_count,), dtype=torch.int16)
        tasks = torch.empty((shard_count,), dtype=torch.int16)
        for row in range(shard_count):
            global_index = shard_index * config.shard_size + row
            token_row, answer, task, _ = make_example(global_index, split, rng)
            tokens[row] = torch.tensor(token_row, dtype=torch.int32)
            answers[row] = answer
            tasks[row] = task
            task_counts[TASKS[task]] += 1
        path = split_dir / f"shard-{shard_index:05d}.pt"
        torch.save({"tokens": tokens, "answers": answers, "tasks": tasks}, path)
        shards.append({"path": str(path), "examples": shard_count, "tokens": shard_count * SEQ_TOKENS})
        remaining -= shard_count
        shard_index += 1
    return {
        "split": split,
        "examples": count,
        "tokens": count * SEQ_TOKENS,
        "shards": shards,
        "task_counts": task_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_avc_dataset_pipeline/dataset"))
    parser.add_argument("--train-examples", type=int, default=AVCDatasetConfig.train_examples)
    parser.add_argument("--val-examples", type=int, default=AVCDatasetConfig.val_examples)
    parser.add_argument("--test-examples", type=int, default=AVCDatasetConfig.test_examples)
    parser.add_argument("--heldout-examples", type=int, default=AVCDatasetConfig.heldout_examples)
    parser.add_argument("--shard-size", type=int, default=AVCDatasetConfig.shard_size)
    parser.add_argument("--seed", type=int, default=AVCDatasetConfig.seed)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AVCDatasetConfig:
    return AVCDatasetConfig(
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        test_examples=args.test_examples,
        heldout_examples=args.heldout_examples,
        shard_size=args.shard_size,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    started = time.perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {
        "train": config.train_examples,
        "val": config.val_examples,
        "test": config.test_examples,
        "heldout": config.heldout_examples,
    }
    splits = [write_split(split, count, config, output_dir) for split, count in split_counts.items()]
    total_examples = sum(split["examples"] for split in splits)
    total_tokens = sum(split["tokens"] for split in splits)
    manifest = {
        "schema_version": 1,
        "stage": "AV-C",
        "config": asdict(config),
        "vocab_size": TOKEN_VOCAB,
        "answer_classes": ANSWER_CLASSES,
        "tasks": TASKS,
        "seq_tokens": SEQ_TOKENS,
        "total_examples": total_examples,
        "total_tokens": total_tokens,
        "approx_examples_for_1b_tokens": int(1_000_000_000 // SEQ_TOKENS),
        "splits": splits,
        "elapsed_sec": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "total_examples": total_examples, "total_tokens": total_tokens}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
