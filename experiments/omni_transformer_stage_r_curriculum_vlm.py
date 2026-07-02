from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import omni_transformer_stage_m_moe_multimodal_llm as stage_m
import omni_transformer_stage_p_llava_moe as stage_p
import omni_transformer_stage_q_tiny_moe_vlm_from_scratch as stage_q
from visual_multimodal_stage_ab import write_png


COLORS = ("red", "green", "blue", "yellow", "purple", "cyan", "white", "orange")
RGB = (
    (0.92, 0.08, 0.08),
    (0.08, 0.74, 0.18),
    (0.12, 0.30, 0.94),
    (0.96, 0.78, 0.08),
    (0.62, 0.22, 0.92),
    (0.08, 0.82, 0.90),
    (0.94, 0.94, 0.90),
    (0.96, 0.46, 0.08),
)
SHAPES = ("circle", "square", "triangle", "diamond")
POSITIONS = ("top left", "top right", "bottom left", "bottom right", "center")
COUNT_WORDS = ("zero", "one", "two", "three", "four")
LEVELS = (
    "l1_color",
    "l2_object",
    "l3_position",
    "l4_count",
    "l5_relation",
    "l6_scene_caption",
)
LEVEL_DESCRIPTIONS = {
    "l1_color": "single centered object, answer its color among 8 colors",
    "l2_object": "single centered object, answer color+shape",
    "l3_position": "single object at one of 5 positions, answer color+shape+position",
    "l4_count": "multiple objects, count target color/shape",
    "l5_relation": "two objects, answer the color of the object left/right of a queried object",
    "l6_scene_caption": "three-object scene, choose full left-to-right caption",
}

MODEL_MODES = {
    "text_only": "text_only",
    "direct": "direct",
    "mean_latent": "mean_latent",
    "moe_latent": "moe_latent",
}
MODEL_OUTPUT_NAMES = {
    "text_only": "scratch_text_only",
    "direct": "scratch_direct",
    "mean_latent": "scratch_mean_pool_latent",
    "moe_latent": "scratch_moe_attention_pump_latent",
}


@dataclass(frozen=True)
class StageRConfig:
    train_size: int = 1536
    val_size: int = 384
    test_size: int = 512
    batch_size: int = 128
    seed: int = 20260701
    image_size: int = 64
    prompt_len: int = 96
    answer_len: int = 96
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    latent_tokens: int = 8
    candidate_count: int = 8
    text_only_steps: int = 80
    direct_steps: int = 160
    mean_steps: int = 160
    moe_steps: int = 220
    eval_every: int = 80
    router_loss_weight: float = 0.10
    probe_steps: int = 80
    pass_threshold: float = 0.90
    visual_gap_threshold: float = 0.20
    levels: tuple[str, ...] = LEVELS
    models: tuple[str, ...] = ("text_only", "direct", "moe_latent")


@dataclass(frozen=True)
class SceneObject:
    color: int
    shape: int
    position: int


@dataclass(frozen=True)
class CurriculumItem:
    example: stage_p.LLaVAExample
    image: torch.Tensor
    meta: dict[str, object]


def stage_q_config(config: StageRConfig, *, candidate_count: int) -> stage_q.StageQConfig:
    return stage_q.StageQConfig(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        image_size=config.image_size,
        prompt_len=config.prompt_len,
        answer_len=config.answer_len,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        lr=config.lr,
        latent_tokens=config.latent_tokens,
        candidate_count=candidate_count,
        text_only_steps=config.text_only_steps,
        direct_steps=config.direct_steps,
        mean_steps=config.mean_steps,
        moe_steps=config.moe_steps,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
    )


def required_experts() -> tuple[float, ...]:
    return (1.0, 1.0, 0.0, 1.0)


def blank_image(image_size: int) -> torch.Tensor:
    image = torch.full((3, image_size, image_size), 0.055, dtype=torch.float32)
    image[:, 4:-4, 4:-4] = 0.12
    image[:, 4:5, 4:-4] = 0.20
    image[:, -5:-4, 4:-4] = 0.20
    image[:, 4:-4, 4:5] = 0.20
    image[:, 4:-4, -5:-4] = 0.20
    return image


def position_xy(position: int, image_size: int, object_size: int) -> tuple[int, int]:
    margin = 10
    center = (image_size - object_size) // 2
    coords = {
        0: (margin, margin),
        1: (image_size - object_size - margin, margin),
        2: (margin, image_size - object_size - margin),
        3: (image_size - object_size - margin, image_size - object_size - margin),
        4: (center, center),
    }
    return coords[position]


def draw_shape(image: torch.Tensor, obj: SceneObject, *, image_size: int, object_size: int = 14) -> None:
    x, y = position_xy(obj.position, image_size, object_size)
    color = torch.tensor(RGB[obj.color], dtype=torch.float32).view(3, 1)
    yy, xx = torch.meshgrid(torch.arange(object_size), torch.arange(object_size), indexing="ij")
    if SHAPES[obj.shape] == "circle":
        center = (object_size - 1) / 2
        mask = ((yy - center) ** 2 + (xx - center) ** 2) <= (object_size * 0.43) ** 2
    elif SHAPES[obj.shape] == "square":
        mask = torch.ones(object_size, object_size, dtype=torch.bool)
    elif SHAPES[obj.shape] == "triangle":
        mask = yy >= torch.abs(xx - (object_size - 1) / 2)
    else:
        center = (object_size - 1) / 2
        mask = (torch.abs(xx - center) + torch.abs(yy - center)) <= object_size * 0.46
    patch = image[:, y : y + object_size, x : x + object_size]
    patch[:, mask] = color.expand(3, int(mask.sum()))


def render_scene(objects: Iterable[SceneObject], image_size: int) -> torch.Tensor:
    image = blank_image(image_size)
    for obj in objects:
        draw_shape(image, obj, image_size=image_size)
    return image


def object_phrase(obj: SceneObject) -> str:
    return f"{COLORS[obj.color]} {SHAPES[obj.shape]}"


def object_position_phrase(obj: SceneObject) -> str:
    return f"{object_phrase(obj)} at {POSITIONS[obj.position]}"


def make_llava_like_example(
    *,
    level: str,
    split: str,
    index: int,
    prompt: str,
    answer: str,
) -> stage_p.LLaVAExample:
    return stage_p.LLaVAExample(
        id=f"{level}-{split}-{index:06d}",
        image_name=f"{level}_{split}_{index:06d}.png",
        image_path="synthetic://stage_r",
        prompt=prompt,
        answer=answer,
        has_history=False,
        source_file=f"stage_r/{level}",
        turn_index=0,
    )


def make_level_item(level: str, split: str, index: int, rng: random.Random, config: StageRConfig) -> CurriculumItem:
    if level == "l1_color":
        obj = SceneObject(color=rng.randrange(len(COLORS)), shape=1, position=4)
        prompt = "what color is the object? answer one color word."
        answer = COLORS[obj.color]
        objects = (obj,)
    elif level == "l2_object":
        obj = SceneObject(color=rng.randrange(len(COLORS)), shape=rng.randrange(len(SHAPES)), position=4)
        prompt = "what object is shown? answer color and shape."
        answer = object_phrase(obj)
        objects = (obj,)
    elif level == "l3_position":
        obj = SceneObject(
            color=rng.randrange(len(COLORS)),
            shape=rng.randrange(len(SHAPES)),
            position=rng.randrange(len(POSITIONS)),
        )
        prompt = "describe the object and its position."
        answer = object_position_phrase(obj)
        objects = (obj,)
    elif level == "l4_count":
        target_color = rng.randrange(len(COLORS))
        target_shape = rng.randrange(len(SHAPES))
        match_count = rng.randrange(len(COUNT_WORDS))
        slots = list(range(len(POSITIONS)))
        rng.shuffle(slots)
        objects_list: list[SceneObject] = [
            SceneObject(color=target_color, shape=target_shape, position=slots.pop())
            for _ in range(match_count)
        ]
        distractor_count = max(1, min(len(slots), 4 - match_count + rng.randrange(2)))
        for _ in range(distractor_count):
            color = rng.randrange(len(COLORS))
            shape = rng.randrange(len(SHAPES))
            if color == target_color and shape == target_shape:
                shape = (shape + 1) % len(SHAPES)
            objects_list.append(SceneObject(color=color, shape=shape, position=slots.pop()))
        prompt = f"how many {COLORS[target_color]} {SHAPES[target_shape]} objects are in the image?"
        answer = COUNT_WORDS[match_count]
        objects = tuple(objects_list)
    elif level == "l5_relation":
        left_position, right_position = (0, 1) if rng.random() < 0.5 else (2, 3)
        left = SceneObject(color=rng.randrange(len(COLORS)), shape=rng.randrange(len(SHAPES)), position=left_position)
        right = SceneObject(color=rng.randrange(len(COLORS)), shape=rng.randrange(len(SHAPES)), position=right_position)
        if rng.random() < 0.5:
            prompt = f"what color is the object left of the {object_phrase(right)}?"
            answer = COLORS[left.color]
        else:
            prompt = f"what color is the object right of the {object_phrase(left)}?"
            answer = COLORS[right.color]
        objects = (left, right)
    elif level == "l6_scene_caption":
        positions = [0, 1, 2, 3, 4]
        rng.shuffle(positions)
        objects_list = [
            SceneObject(
                color=rng.randrange(len(COLORS)),
                shape=rng.randrange(len(SHAPES)),
                position=positions.pop(),
            )
            for _ in range(3)
        ]
        objects_list.sort(key=lambda obj: (position_xy(obj.position, config.image_size, 14)[0], position_xy(obj.position, config.image_size, 14)[1]))
        prompt = "describe all objects from left to right."
        answer = "; ".join(object_position_phrase(obj) for obj in objects_list)
        objects = tuple(objects_list)
    else:
        raise ValueError(f"unknown curriculum level: {level}")

    image = render_scene(objects, config.image_size)
    example = make_llava_like_example(level=level, split=split, index=index, prompt=prompt, answer=answer)
    return CurriculumItem(
        example=example,
        image=image,
        meta={
            "level": level,
            "prompt": prompt,
            "answer": answer,
            "objects": [asdict(obj) for obj in objects],
        },
    )


def split_seed(seed: int, level: str, split: str) -> int:
    level_offset = LEVELS.index(level) * 100_000 if level in LEVELS else 900_000
    split_offset = {"train": 11, "val": 29, "test": 47}[split]
    return seed + level_offset + split_offset


def build_pixel_set(level: str, split: str, size: int, config: StageRConfig) -> tuple[stage_q.PixelSet, list[dict[str, object]]]:
    rng = random.Random(split_seed(config.seed, level, split))
    items = [make_level_item(level, split, index, rng, config) for index in range(size)]
    prompts = torch.tensor(
        [stage_q.encode_text(item.example.prompt, config.prompt_len, add_eos=True) for item in items],
        dtype=torch.long,
    )
    required = torch.tensor([required_experts() for _ in items], dtype=torch.float32)
    return (
        stage_q.PixelSet(
            examples=[item.example for item in items],
            image=torch.stack([item.image for item in items], dim=0),
            prompt=prompts,
            required_experts=required,
        ),
        [item.meta for item in items],
    )


def build_level_data(level: str, config: StageRConfig) -> tuple[dict[str, stage_q.PixelSet], dict[str, object], dict[str, list[dict[str, object]]]]:
    started = time.perf_counter()
    train, train_meta = build_pixel_set(level, "train", config.train_size, config)
    val, val_meta = build_pixel_set(level, "val", config.val_size, config)
    test, test_meta = build_pixel_set(level, "test", config.test_size, config)
    all_examples = train.examples + val.examples + test.examples
    answer_pool = list(dict.fromkeys(example.answer for example in all_examples))
    theoretical_spaces = {
        "l1_color": len(COLORS),
        "l2_object": len(COLORS) * len(SHAPES),
        "l3_position": len(COLORS) * len(SHAPES) * len(POSITIONS),
        "l4_count": len(COUNT_WORDS),
        "l5_relation": len(COLORS),
        "l6_scene_caption": len(POSITIONS) * (len(COLORS) * len(SHAPES)) ** 3,
    }
    stats = {
        "level": level,
        "description": LEVEL_DESCRIPTIONS[level],
        "split_sizes": {"train": len(train.examples), "val": len(val.examples), "test": len(test.examples)},
        "unique_answers": len(answer_pool),
        "theoretical_answer_space": theoretical_spaces[level],
        "observed_answer_entropy_bits": round(float(torch.log2(torch.tensor(max(len(answer_pool), 1), dtype=torch.float32))), 4),
        "build_seconds": round(time.perf_counter() - started, 3),
    }
    return {"train": train, "val": val, "test": test}, stats, {"train": train_meta, "val": val_meta, "test": test_meta}


def write_samples(data: stage_q.PixelSet, meta: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(min(12, len(data.examples))):
        image_path = output_dir / f"sample_{index:02d}.png"
        write_png(image_path, data.image[index])
        rows.append({"image": image_path.name, **meta[index]})
    (output_dir / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def model_steps(name: str, config: StageRConfig) -> int:
    if name == "scratch_text_only":
        return config.text_only_steps
    if name == "scratch_direct":
        return config.direct_steps
    if name == "scratch_mean_pool_latent":
        return config.mean_steps
    if name == "scratch_moe_attention_pump_latent":
        return config.moe_steps
    raise ValueError(name)


def run_level(level: str, config: StageRConfig, *, device: torch.device, output_dir: Path) -> dict[str, object]:
    pixel_sets, data_stats, meta = build_level_data(level, config)
    answer_pool_values = [example.answer for split in pixel_sets.values() for example in split.examples]
    candidate_count = min(config.candidate_count, len(set(answer_pool_values)))
    model_config = stage_q_config(config, candidate_count=candidate_count)
    train = pixel_sets["train"].to(device)
    val = pixel_sets["val"].to(device)
    test = pixel_sets["test"].to(device)
    candidate_pool = stage_q.build_candidate_pool(answer_pool_values, model_config, device=device)
    data_stats["candidate_count"] = candidate_count
    data_stats["random_top1"] = round(1.0 / candidate_count, 6)
    write_samples(pixel_sets["test"], meta["test"], output_dir / "samples")

    models = {
        MODEL_OUTPUT_NAMES[name]: stage_q.TinyScratchMoEVLM(model_config, mode=MODEL_MODES[name])
        for name in config.models
    }
    training = {}
    for index, (name, model) in enumerate(models.items()):
        training[name] = stage_q.train_model(
            model,
            train,
            val,
            model_config,
            candidate_pool=candidate_pool,
            steps=model_steps(name, config),
            seed=config.seed + 700 + LEVELS.index(level) * 100 + index * 17,
            device=device,
            label=f"{level}:{name}",
        )

    metrics = {
        name: stage_q.evaluate_ranking(model, test, model_config, device=device, candidate_pool=candidate_pool, seed=config.seed + 3000 + index)
        for index, (name, model) in enumerate(models.items())
    }
    ablations: dict[str, object] = {}
    latent_probe: dict[str, object] = {}
    moe = models.get("scratch_moe_attention_pump_latent")
    if moe is not None:
        ablations = {
            "scratch_moe_no_latent_access": stage_q.evaluate_ranking(
                moe,
                test,
                model_config,
                device=device,
                candidate_pool=candidate_pool,
                seed=config.seed + 5000,
                disable_latent_access=True,
            ),
            "scratch_moe_no_image_modality": stage_q.evaluate_ranking(
                moe,
                test,
                model_config,
                device=device,
                candidate_pool=candidate_pool,
                seed=config.seed + 5001,
                zero_modalities=("image",),
            ),
            "scratch_moe_no_text_modality": stage_q.evaluate_ranking(
                moe,
                test,
                model_config,
                device=device,
                candidate_pool=candidate_pool,
                seed=config.seed + 5002,
                zero_modalities=("text",),
            ),
        }
        latent_probe = stage_q.train_latent_probe(moe, train, test, model_config, device=device, steps=config.probe_steps)

    prediction_cost = {
        f"{name}_prediction": stage_q.measure_prediction_cost(
            lambda model=model: stage_q.evaluate_ranking(model, test, model_config, device=device, candidate_pool=candidate_pool, seed=config.seed + 8000),
            example_count=len(test.examples),
            device=device,
        )
        for name, model in models.items()
    }
    prediction_samples = stage_q.prediction_samples(models, test, model_config, device=device, candidate_pool=candidate_pool)
    full_top1 = float(metrics.get("scratch_moe_attention_pump_latent", metrics[next(iter(metrics))])["rank_top1"])
    no_image_top1 = float(ablations.get("scratch_moe_no_image_modality", {"rank_top1": full_top1})["rank_top1"])
    solved = full_top1 >= config.pass_threshold and (full_top1 - no_image_top1) >= config.visual_gap_threshold
    return {
        "level": level,
        "description": LEVEL_DESCRIPTIONS[level],
        "config": asdict(model_config),
        "data": data_stats,
        "metrics": metrics,
        "ablations": ablations,
        "latent_probe": latent_probe,
        "training": training,
        "prediction_cost": prediction_cost,
        "prediction_samples": prediction_samples,
        "solved_by_moe_visual": solved,
        "visual_gap": full_top1 - no_image_top1,
    }


def run_experiment(config: StageRConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_r_curriculum_vlm device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    output_dir = output_path.parent
    levels = {}
    started = time.perf_counter()
    for level in config.levels:
        print(f"running curriculum level {level}: {LEVEL_DESCRIPTIONS[level]}", flush=True)
        levels[level] = run_level(level, config, device=device, output_dir=output_dir / level)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    solved_levels = [
        level
        for level in config.levels
        if bool(levels[level]["solved_by_moe_visual"])
    ]
    output = {
        "experiment": "omni_transformer_stage_r_curriculum_vlm",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "base_model": "Stage Q tiny from-scratch MoE-VLM",
            "pretrained_experts": False,
            "experts": stage_q.EXPERTS,
            "curriculum": LEVEL_DESCRIPTIONS,
            "solved_definition": f"MoE latent top1 >= {config.pass_threshold:.2f} and MoE-minus-no-image >= {config.visual_gap_threshold:.2f}",
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "levels": levels,
        "capability_ceiling": {
            "solved_levels": solved_levels,
            "highest_solved_level": solved_levels[-1] if solved_levels else None,
            "first_failed_level": next((level for level in config.levels if level not in solved_levels), None),
        },
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    return output


def collect_numbers(value: object, prefix: tuple[str, ...] = ()) -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {".".join(prefix): float(value)}
    if isinstance(value, dict):
        collected: dict[str, float] = {}
        for key, child in value.items():
            collected.update(collect_numbers(child, (*prefix, str(key))))
        return collected
    return {}


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    buckets: dict[str, list[float]] = {}
    for run in runs:
        numbers = collect_numbers(
            {
                "levels": run["levels"],
                "total_seconds": run["total_seconds"],
            }
        )
        for key, value in numbers.items():
            buckets.setdefault(key, []).append(value)
    stats = {key: stage_m.stat(values) for key, values in sorted(buckets.items())}
    level_summary: dict[str, object] = {}
    for level in runs[0]["config"]["levels"]:
        full_key = f"levels.{level}.metrics.scratch_moe_attention_pump_latent.rank_top1"
        no_image_key = f"levels.{level}.ablations.scratch_moe_no_image_modality.rank_top1"
        direct_key = f"levels.{level}.metrics.scratch_direct.rank_top1"
        random_key = f"levels.{level}.data.random_top1"
        full = stats.get(full_key, {}).get("mean")
        no_image = stats.get(no_image_key, {}).get("mean")
        direct = stats.get(direct_key, {}).get("mean")
        random_top1 = stats.get(random_key, {}).get("mean")
        visual_gap = None if full is None or no_image is None else full - no_image
        level_summary[level] = {
            "description": LEVEL_DESCRIPTIONS[level],
            "random_top1": random_top1,
            "direct_top1": direct,
            "moe_latent_top1": full,
            "moe_no_image_top1": no_image,
            "visual_gap": visual_gap,
        }
    pass_threshold = float(runs[0]["config"]["pass_threshold"])
    visual_gap_threshold = float(runs[0]["config"]["visual_gap_threshold"])
    solved_levels = [
        level
        for level, item in level_summary.items()
        if item["moe_latent_top1"] is not None
        and item["visual_gap"] is not None
        and float(item["moe_latent_top1"]) >= pass_threshold
        and float(item["visual_gap"]) >= visual_gap_threshold
    ]
    return {
        "experiment": "omni_transformer_stage_r_curriculum_vlm_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": stats,
        "level_summary": level_summary,
        "capability_ceiling": {
            "solved_definition": f"MoE latent top1 >= {pass_threshold:.2f} and MoE-minus-no-image >= {visual_gap_threshold:.2f}",
            "solved_levels": solved_levels,
            "highest_solved_level": solved_levels[-1] if solved_levels else None,
            "first_failed_level": next((level for level in runs[0]["config"]["levels"] if level not in solved_levels), None),
        },
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_r_curriculum_vlm/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_r_curriculum_vlm/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_r_curriculum_vlm/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--models", default="text_only,direct,moe_latent")
    parser.add_argument("--train-size", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=384)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=96)
    parser.add_argument("--answer-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--text-only-steps", type=int, default=80)
    parser.add_argument("--direct-steps", type=int, default=160)
    parser.add_argument("--mean-steps", type=int, default=160)
    parser.add_argument("--moe-steps", type=int, default=220)
    parser.add_argument("--eval-every", type=int, default=80)
    parser.add_argument("--probe-steps", type=int, default=80)
    parser.add_argument("--pass-threshold", type=float, default=0.90)
    parser.add_argument("--visual-gap-threshold", type=float, default=0.20)
    args = parser.parse_args()
    models = parse_csv_strings(args.models)
    invalid_models = [name for name in models if name not in MODEL_MODES]
    if invalid_models:
        raise ValueError(f"unknown model names: {invalid_models}; choices are {sorted(MODEL_MODES)}")

    base_config = StageRConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        image_size=args.image_size,
        prompt_len=args.prompt_len,
        answer_len=args.answer_len,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        latent_tokens=args.latent_tokens,
        candidate_count=args.candidate_count,
        text_only_steps=args.text_only_steps,
        direct_steps=args.direct_steps,
        mean_steps=args.mean_steps,
        moe_steps=args.moe_steps,
        eval_every=args.eval_every,
        probe_steps=args.probe_steps,
        pass_threshold=args.pass_threshold,
        visual_gap_threshold=args.visual_gap_threshold,
        levels=parse_csv_strings(args.levels),
        models=models,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageRConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
