from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import string
import sys
import time

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visual_multimodal_stage_ab import write_png  # noqa: E402


COLORS = ("red", "green", "blue", "yellow", "purple", "cyan", "orange", "white")
SHAPES = ("circle", "square", "triangle", "diamond")
POSITIONS = ("top left", "top right", "middle left", "center", "middle right", "bottom left", "bottom right")
EDIT_TYPES = ("recolor", "reshape", "move")
MAX_OBJECTS = 3
FIELDS_PER_OBJECT = 4
RECORD_LEN = 1 + MAX_OBJECTS * FIELDS_PER_OBJECT
SOURCE_TEXT_LEN = 112
EDIT_TEXT_LEN = 80
TARGET_TEXT_LEN = 112
IMAGE_SIZE = 64
PATCH_SIZE = 8
PATCHES_PER_IMAGE = (IMAGE_SIZE // PATCH_SIZE) ** 2
TOKENS_PER_EXAMPLE = SOURCE_TEXT_LEN + EDIT_TEXT_LEN + TARGET_TEXT_LEN + RECORD_LEN * 2 + PATCHES_PER_IMAGE * 2

PAD = 0
BOS = 1
EOS = 2
CHARSET = string.ascii_lowercase + string.digits + " .,;:-"
CHAR_TO_ID = {char: index + 3 for index, char in enumerate(CHARSET)}
CHAR_VOCAB = len(CHAR_TO_ID) + 3

RECORD_PAD = 0
RECORD_BASE_COUNT = 1
RECORD_BASE_OBJECT = RECORD_BASE_COUNT + (MAX_OBJECTS + 1)
RECORD_BASE_COLOR = RECORD_BASE_OBJECT + MAX_OBJECTS
RECORD_BASE_SHAPE = RECORD_BASE_COLOR + len(COLORS)
RECORD_BASE_POSITION = RECORD_BASE_SHAPE + len(SHAPES)
RECORD_VOCAB = RECORD_BASE_POSITION + len(POSITIONS)

SPLITS = ("train", "val", "test", "heldout")
SPLIT_OFFSETS = {"train": 0, "val": 100_000, "test": 200_000, "heldout": 300_000}
_RENDER_CACHE: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class AVHScale:
    train_examples: int
    val_examples: int
    test_examples: int
    heldout_examples: int
    shard_size: int


SCALES = {
    "smoke": AVHScale(train_examples=512, val_examples=128, test_examples=128, heldout_examples=128, shard_size=128),
    "10m": AVHScale(train_examples=18_000, val_examples=1_300, test_examples=1_300, heldout_examples=1_300, shard_size=3_000),
    "100m": AVHScale(train_examples=420_000, val_examples=30_000, test_examples=30_000, heldout_examples=30_000, shard_size=10_000),
}


@dataclass(frozen=True)
class AVHDatasetConfig:
    scale: str = "10m"
    train_examples: int = SCALES["10m"].train_examples
    val_examples: int = SCALES["10m"].val_examples
    test_examples: int = SCALES["10m"].test_examples
    heldout_examples: int = SCALES["10m"].heldout_examples
    shard_size: int = SCALES["10m"].shard_size
    seed: int = 20260705
    image_size: int = IMAGE_SIZE
    patch_size: int = PATCH_SIZE
    tokens_per_example: int = TOKENS_PER_EXAMPLE
    distribution_note: str = (
        "Text-anchored visual editing dataset. Text descriptions and edit instructions anchor source/target object records; "
        "images are rendered from records at training time so the model still trains image edit, but the dataset does not store pixel shards."
    )


@dataclass(frozen=True)
class ObjectSpec:
    object_id: int
    color: int
    shape: int
    position: int


@dataclass(frozen=True)
class EditExample:
    source_objects: tuple[ObjectSpec, ...]
    target_objects: tuple[ObjectSpec, ...]
    edit_type: int
    edit_object: int
    edit_value: int
    source_text: str
    edit_text: str
    target_text: str
    case_id: str


def encode_text(text: str, length: int) -> list[int]:
    ids = [BOS]
    ids.extend(CHAR_TO_ID[char] for char in text.lower() if char in CHAR_TO_ID)
    ids = ids[: length - 1] + [EOS]
    return ids + [PAD] * max(0, length - len(ids))


def encode_record(objects: tuple[ObjectSpec, ...]) -> list[int]:
    tokens = [RECORD_BASE_COUNT + len(objects)]
    for slot in range(MAX_OBJECTS):
        if slot >= len(objects):
            tokens.extend([RECORD_PAD] * FIELDS_PER_OBJECT)
            continue
        obj = objects[slot]
        tokens.extend(
            [
                RECORD_BASE_OBJECT + obj.object_id,
                RECORD_BASE_COLOR + obj.color,
                RECORD_BASE_SHAPE + obj.shape,
                RECORD_BASE_POSITION + obj.position,
            ]
        )
    return tokens


def describe_objects(objects: tuple[ObjectSpec, ...]) -> str:
    chunks = []
    for obj in objects:
        chunks.append(f"object {obj.object_id} is a {COLORS[obj.color]} {SHAPES[obj.shape]} at {POSITIONS[obj.position]}")
    return "; ".join(chunks) + "."


def make_example(index: int, rng: random.Random, split: str) -> EditExample:
    object_count = 1 + rng.randrange(MAX_OBJECTS)
    if split == "heldout":
        object_count = 2 + (index % 2)
    positions = list(range(len(POSITIONS)))
    rng.shuffle(positions)
    objects = []
    for object_id in range(object_count):
        color = rng.randrange(len(COLORS))
        shape = rng.randrange(len(SHAPES))
        position = positions[object_id]
        if split == "heldout":
            color = (color + object_id + index) % len(COLORS)
            shape = (shape + 1) % len(SHAPES)
        objects.append(ObjectSpec(object_id=object_id, color=color, shape=shape, position=position))
    edit_object = rng.randrange(object_count)
    edit_type = index % len(EDIT_TYPES)
    source_objects = tuple(objects)
    target = list(objects)
    if edit_type == 0:
        value = rng.randrange(len(COLORS) - 1)
        new_color = value if value < target[edit_object].color else value + 1
        old = target[edit_object]
        target[edit_object] = ObjectSpec(old.object_id, new_color, old.shape, old.position)
        edit_text = f"change object {edit_object} color to {COLORS[new_color]}."
        edit_value = new_color
    elif edit_type == 1:
        value = rng.randrange(len(SHAPES) - 1)
        new_shape = value if value < target[edit_object].shape else value + 1
        old = target[edit_object]
        target[edit_object] = ObjectSpec(old.object_id, old.color, new_shape, old.position)
        edit_text = f"change object {edit_object} shape to {SHAPES[new_shape]}."
        edit_value = new_shape
    else:
        used_positions = {obj.position for obj in target if obj.object_id != edit_object}
        choices = [pos for pos in range(len(POSITIONS)) if pos not in used_positions and pos != target[edit_object].position]
        new_position = choices[rng.randrange(len(choices))]
        old = target[edit_object]
        target[edit_object] = ObjectSpec(old.object_id, old.color, old.shape, new_position)
        edit_text = f"move object {edit_object} to {POSITIONS[new_position]}."
        edit_value = new_position
    target_objects = tuple(target)
    return EditExample(
        source_objects=source_objects,
        target_objects=target_objects,
        edit_type=edit_type,
        edit_object=edit_object,
        edit_value=edit_value,
        source_text="source scene: " + describe_objects(source_objects),
        edit_text=edit_text,
        target_text="target scene: " + describe_objects(target_objects),
        case_id=f"{split}-{index:07d}",
    )


def split_count(config: AVHDatasetConfig, split: str) -> int:
    return {
        "train": config.train_examples,
        "val": config.val_examples,
        "test": config.test_examples,
        "heldout": config.heldout_examples,
    }[split]


def render_constants(device: torch.device, image_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (str(device), image_size)
    cached = _RENDER_CACHE.get(key)
    if cached is not None:
        return cached
    axis = torch.arange(image_size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    palette = torch.tensor(
        [
            [0.92, 0.08, 0.08],
            [0.08, 0.72, 0.18],
            [0.12, 0.30, 0.94],
            [0.96, 0.76, 0.08],
            [0.64, 0.20, 0.92],
            [0.08, 0.82, 0.90],
            [0.95, 0.48, 0.12],
            [0.92, 0.92, 0.86],
        ],
        device=device,
    )
    centers = torch.tensor(
        [
            [15, 15],
            [49, 15],
            [15, 32],
            [32, 32],
            [49, 32],
            [15, 49],
            [49, 49],
        ],
        device=device,
        dtype=torch.float32,
    )
    if image_size != IMAGE_SIZE:
        centers = centers * (float(image_size) / float(IMAGE_SIZE))
    _RENDER_CACHE[key] = (yy, xx, palette, centers)
    return _RENDER_CACHE[key]


def render_records(records: torch.Tensor, *, image_size: int = IMAGE_SIZE, device: torch.device | None = None) -> torch.Tensor:
    device = device or records.device
    records = records.to(device=device, dtype=torch.long, non_blocking=True)
    batch = records.shape[0]
    images = torch.full((batch, 3, image_size, image_size), 0.06, device=device)
    render_device = device if isinstance(device, torch.device) else torch.device(device)
    yy, xx, palette, centers = render_constants(render_device, image_size)
    object_count = (records[:, 0] - RECORD_BASE_COUNT).clamp(0, MAX_OBJECTS)
    batch_index = torch.arange(batch, device=device)
    for slot in range(MAX_OBJECTS):
        base = 1 + slot * FIELDS_PER_OBJECT
        active = object_count > slot
        color = (records[:, base + 1] - RECORD_BASE_COLOR).clamp(0, len(COLORS) - 1)
        shape = (records[:, base + 2] - RECORD_BASE_SHAPE).clamp(0, len(SHAPES) - 1)
        position = (records[:, base + 3] - RECORD_BASE_POSITION).clamp(0, len(POSITIONS) - 1)
        cx = centers[position, 0].view(batch, 1, 1)
        cy = centers[position, 1].view(batch, 1, 1)
        dx = xx.view(1, image_size, image_size) - cx
        dy = yy.view(1, image_size, image_size) - cy
        circle = dx.square() + dy.square() <= 8**2
        square = dx.abs().maximum(dy.abs()) <= 8
        triangle = (dy >= -7) & (dy <= 8) & (dx.abs() <= (dy + 9))
        diamond = dx.abs() + dy.abs() <= 10
        masks = torch.stack((circle, square, triangle, diamond), dim=1)
        mask = masks[batch_index, shape] & active.view(batch, 1, 1)
        rgb = palette[color].view(batch, 3, 1, 1)
        images = torch.where(mask.view(batch, 1, image_size, image_size), rgb, images)
    return images.clamp(0.0, 1.0)


def write_split(split: str, count: int, config: AVHDatasetConfig, output_dir: Path) -> dict[str, object]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed + SPLIT_OFFSETS[split])
    remaining = count
    written = 0
    shard_index = 0
    shards = []
    edit_counts = {name: 0 for name in EDIT_TYPES}
    object_counts = {str(index): 0 for index in range(1, MAX_OBJECTS + 1)}
    while remaining > 0:
        shard_count = min(config.shard_size, remaining)
        source_text = torch.empty((shard_count, SOURCE_TEXT_LEN), dtype=torch.int16)
        edit_text = torch.empty((shard_count, EDIT_TEXT_LEN), dtype=torch.int16)
        target_text = torch.empty((shard_count, TARGET_TEXT_LEN), dtype=torch.int16)
        source_record = torch.empty((shard_count, RECORD_LEN), dtype=torch.int16)
        target_record = torch.empty((shard_count, RECORD_LEN), dtype=torch.int16)
        edit_type = torch.empty((shard_count,), dtype=torch.int16)
        edit_object = torch.empty((shard_count,), dtype=torch.int16)
        edit_value = torch.empty((shard_count,), dtype=torch.int16)
        case_ids = []
        for row in range(shard_count):
            example = make_example(written + row, rng, split)
            source_text[row] = torch.tensor(encode_text(example.source_text, SOURCE_TEXT_LEN), dtype=torch.int16)
            edit_text[row] = torch.tensor(encode_text(example.edit_text, EDIT_TEXT_LEN), dtype=torch.int16)
            target_text[row] = torch.tensor(encode_text(example.target_text, TARGET_TEXT_LEN), dtype=torch.int16)
            source_record[row] = torch.tensor(encode_record(example.source_objects), dtype=torch.int16)
            target_record[row] = torch.tensor(encode_record(example.target_objects), dtype=torch.int16)
            edit_type[row] = example.edit_type
            edit_object[row] = example.edit_object
            edit_value[row] = example.edit_value
            case_ids.append(example.case_id)
            edit_counts[EDIT_TYPES[example.edit_type]] += 1
            object_counts[str(len(example.source_objects))] += 1
        shard_path = split_dir / f"shard-{shard_index:05d}.pt"
        torch.save(
            {
                "source_text": source_text,
                "edit_text": edit_text,
                "target_text": target_text,
                "source_record": source_record,
                "target_record": target_record,
                "edit_type": edit_type,
                "edit_object": edit_object,
                "edit_value": edit_value,
                "case_ids": case_ids,
            },
            shard_path,
        )
        shards.append({"path": str(shard_path.relative_to(output_dir)).replace("\\", "/"), "examples": shard_count, "tokens": shard_count * config.tokens_per_example})
        remaining -= shard_count
        written += shard_count
        shard_index += 1
    return {
        "split": split,
        "examples": count,
        "tokens": count * config.tokens_per_example,
        "shards": shards,
        "edit_counts": edit_counts,
        "object_counts": object_counts,
    }


def write_samples(output_dir: Path, config: AVHDatasetConfig) -> None:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed + 999)
    rows = []
    for index in range(6):
        example = make_example(index, rng, "sample")
        source_record = torch.tensor([encode_record(example.source_objects)], dtype=torch.long)
        target_record = torch.tensor([encode_record(example.target_objects)], dtype=torch.long)
        source_path = sample_dir / f"{example.case_id}_source.png"
        target_path = sample_dir / f"{example.case_id}_target.png"
        write_png(source_path, render_records(source_record)[0].detach().cpu())
        write_png(target_path, render_records(target_record)[0].detach().cpu())
        rows.append(
            {
                "case_id": example.case_id,
                "source_text": example.source_text,
                "edit_text": example.edit_text,
                "target_text": example.target_text,
                "source_png": str(source_path),
                "target_png": str(target_path),
                "source_record": encode_record(example.source_objects),
                "target_record": encode_record(example.target_objects),
            }
        )
    (sample_dir / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=sorted(SCALES), default="10m")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_dataset/dataset_10m"))
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--val-examples", type=int)
    parser.add_argument("--test-examples", type=int)
    parser.add_argument("--heldout-examples", type=int)
    parser.add_argument("--shard-size", type=int)
    parser.add_argument("--seed", type=int, default=AVHDatasetConfig.seed)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AVHDatasetConfig:
    defaults = SCALES[args.scale]
    return AVHDatasetConfig(
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
    write_samples(args.output_dir, config)
    total_examples = sum(split["examples"] for split in splits)
    total_tokens = sum(split["tokens"] for split in splits)
    manifest = {
        "schema_version": 1,
        "stage": "AV-H",
        "dataset_type": "text_anchored_visual_edit",
        "config": asdict(config),
        "scale": config.scale,
        "path_base": "manifest_dir",
        "char_vocab": CHAR_VOCAB,
        "record_vocab": RECORD_VOCAB,
        "record_len": RECORD_LEN,
        "source_text_len": SOURCE_TEXT_LEN,
        "edit_text_len": EDIT_TEXT_LEN,
        "target_text_len": TARGET_TEXT_LEN,
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "patches_per_image": PATCHES_PER_IMAGE,
        "tokens_per_example": config.tokens_per_example,
        "total_examples": total_examples,
        "total_tokens": total_tokens,
        "unique_train_tokens": config.train_examples * config.tokens_per_example,
        "splits": splits,
        "elapsed_sec": time.perf_counter() - started,
        "interpretation": {
            "route": "Replace AV-F high-entropy unanchored external translation with text-anchored visual editing.",
            "image_policy": "Images are rendered from object records at training time; samples include PNGs for visual sanity checks.",
            "gates": "Target text exact, target record exact, image scene exact, wrong instruction/source drops, and no-target-record drops.",
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
