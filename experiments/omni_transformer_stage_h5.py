from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import torch

from multimodal_fusion_latent_flow import (
    GOALS,
    IMAGE_SIZE,
    TELEMETRY_CHANNELS,
    TELEMETRY_STATES,
    VISUAL_STATES,
    Batch,
    render_telemetry,
    stat,
)
from omni_transformer_stage_h import (
    GOAL_BASE,
    OmniConfig,
    TELEMETRY_BASE,
    TinyOmniTransformer,
    VISUAL_BASE,
    answer_inputs_from_targets,
    answer_loss,
    answer_targets,
    answer_to_text,
    format_valid,
    greedy_generate,
)
from omni_transformer_stage_h3_h4 import (
    QUERY_COUNTERFACTUAL,
    QUERY_DIAGNOSE,
    QueryExample,
    generate_query_examples,
    heldout_target_triples,
    render_ood_panel_images,
    render_query_text_tokens,
)
from visual_multimodal_stage_ab import write_png


ACTION_COUNT = 6


@dataclass(frozen=True)
class H5Config:
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 2048
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    lr: float = 8e-4
    latent_tokens: int = 8
    visual_steps: int = 950
    compositional_steps: int = 950
    eval_every: int = 475


def omni_config(config: H5Config) -> OmniConfig:
    return OmniConfig(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        d_model=config.d_model,
        layers=config.layers,
        heads=config.heads,
        lr=config.lr,
        latent_tokens=config.latent_tokens,
        eval_every=config.eval_every,
    )


def compositional_action(visual: int, goal: int, telemetry: int) -> int:
    if telemetry == 3:
        return 4
    if telemetry == 1 or visual == 2:
        return 2
    if telemetry == 2:
        return 3
    if visual == 1 or goal == 0:
        return 1
    if goal == 1 and visual != 3:
        return 5
    return 0


def generate_compositional_query_examples(
    size: int,
    *,
    seed: int,
    heldout_target_triples: set[tuple[int, int, int]] | None = None,
    only_heldout: bool = False,
    counterfactual_only: bool = False,
) -> list[QueryExample]:
    heldout_target_triples = heldout_target_triples or set()
    combos = [
        (visual, observed_goal, target_goal, telemetry)
        for visual in range(VISUAL_STATES)
        for observed_goal in range(GOALS)
        for target_goal in range(GOALS)
        for telemetry in range(TELEMETRY_STATES)
    ]
    filtered: list[tuple[int, int, int, int]] = []
    for visual, observed_goal, target_goal, telemetry in combos:
        target_triple = (visual, target_goal, telemetry)
        is_heldout = target_triple in heldout_target_triples
        if only_heldout and not is_heldout:
            continue
        if not only_heldout and is_heldout:
            continue
        if counterfactual_only and target_goal == observed_goal:
            continue
        filtered.append((visual, observed_goal, target_goal, telemetry))

    rng = random.Random(seed)
    examples: list[QueryExample] = []
    while len(examples) < size:
        rng.shuffle(filtered)
        for visual, observed_goal, target_goal, telemetry in filtered:
            query = QUERY_DIAGNOSE if target_goal == observed_goal else QUERY_COUNTERFACTUAL
            examples.append(
                QueryExample(
                    visual=visual,
                    observed_goal=observed_goal,
                    target_goal=target_goal,
                    telemetry=telemetry,
                    query=query,
                    action=compositional_action(visual, target_goal, telemetry),
                )
            )
            if len(examples) == size:
                break
    rng.shuffle(examples)
    return examples


def render_augmented_panel_images(visual_ids: torch.Tensor, *, device: torch.device, variant_ids: torch.Tensor) -> torch.Tensor:
    batch = int(visual_ids.shape[0])
    image = torch.full((batch, 3, IMAGE_SIZE, IMAGE_SIZE), 0.10, device=device)
    palettes = torch.tensor(
        [
            [[0.10, 0.86, 0.24], [0.92, 0.10, 0.10], [0.95, 0.72, 0.10], [0.16, 0.42, 0.95]],
            [[0.05, 0.95, 0.92], [0.95, 0.25, 0.88], [0.92, 0.92, 0.92], [0.98, 0.52, 0.12]],
            [[0.42, 0.90, 0.18], [0.86, 0.18, 0.42], [0.98, 0.86, 0.22], [0.18, 0.72, 0.86]],
            [[0.62, 0.95, 0.62], [0.98, 0.34, 0.24], [0.88, 0.62, 0.98], [0.32, 0.42, 0.98]],
            [[0.82, 0.88, 0.18], [0.82, 0.18, 0.88], [0.18, 0.88, 0.82], [0.92, 0.44, 0.20]],
            [[0.24, 0.76, 0.36], [0.95, 0.36, 0.32], [0.82, 0.82, 0.34], [0.36, 0.48, 0.92]],
            [[0.76, 0.96, 0.44], [0.96, 0.44, 0.58], [0.96, 0.78, 0.38], [0.40, 0.86, 0.96]],
            [[0.18, 0.92, 0.70], [0.92, 0.20, 0.72], [0.82, 0.86, 0.88], [0.98, 0.58, 0.18]],
        ],
        device=device,
    )
    bg = torch.tensor([0.09, 0.14, 0.19, 0.24, 0.11, 0.17, 0.21, 0.27], device=device)[variant_ids]
    image = image * bg[:, None, None, None]
    image[:, :, 4:-4, 4:-4] += 0.12 + 0.02 * (variant_ids[:, None, None, None].float() % 2)
    colors = palettes[variant_ids, visual_ids]

    anchor_x = torch.tensor([7, 9, 6, 10, 8, 5, 11, 6], device=device)[variant_ids]
    anchor_y = torch.tensor([8, 24, 12, 20, 18, 26, 10, 22], device=device)[variant_ids]
    shift_x = torch.tensor([0, -1, 1, 2, -2, 0, 1, -1], device=device)[variant_ids]
    shift_y = torch.tensor([0, 1, -1, 2, -2, -1, 1, 2], device=device)[variant_ids]
    for variant in range(8):
        for visual in range(4):
            mask = (variant_ids == variant) & (visual_ids == visual)
            if not bool(mask.any()):
                continue
            color = colors[mask]
            x = int(anchor_x[mask][0].detach().cpu())
            y = int(anchor_y[mask][0].detach().cpu())
            sx = int(shift_x[mask][0].detach().cpu())
            sy = int(shift_y[mask][0].detach().cpu())
            image[mask, :, x : x + 13, y : y + 15] = color[:, :, None, None]
            if visual == 0:
                image[mask, :, 27 + sx : 31 + sx, 8 + sy : 40 + sy] = color[:, :, None, None]
                image[mask, :, 31 + sx : 39 + sx, 20 + sy : 25 + sy] = color[:, :, None, None] * 0.55
            elif visual == 1:
                image[mask, :, 25 + sx : 31 + sx, 8 + sy : 40 + sy] = color[:, :, None, None]
                image[mask, :, 18 + sx : 39 + sx, 8 + sy : 13 + sy] = color[:, :, None, None] * 0.9
                image[mask, :, 32 + sx : 39 + sx, 31 + sy : 40 + sy] = color[:, :, None, None] * 0.75
            elif visual == 2:
                image[mask, :, 23 + sx : 27 + sx, 8 + sy : 40 + sy] = color[:, :, None, None]
                image[mask, :, 29 + sx : 33 + sx, 8 + sy : 40 + sy] = color[:, :, None, None] * 0.70
                image[mask, :, 35 + sx : 39 + sx, 8 + sy : 40 + sy] = color[:, :, None, None] * 0.55
            else:
                image[mask, :, 22 + sx : 39 + sx, 8 + sy : 16 + sy] = color[:, :, None, None]
                image[mask, :, 22 + sx : 39 + sx, 32 + sy : 40 + sy] = color[:, :, None, None]
                image[mask, :, 28 + sx : 34 + sx, 16 + sy : 32 + sy] = 0.05
    return image.clamp(0.0, 1.0)


def make_query_batch(
    examples: list[QueryExample],
    *,
    device: torch.device,
    image_style: str,
    seed: int,
) -> Batch:
    visual = torch.tensor([item.visual for item in examples], dtype=torch.long, device=device)
    observed_goal = torch.tensor([item.observed_goal for item in examples], dtype=torch.long, device=device)
    target_goal = torch.tensor([item.target_goal for item in examples], dtype=torch.long, device=device)
    telemetry = torch.tensor([item.telemetry for item in examples], dtype=torch.long, device=device)
    query = torch.tensor([item.query for item in examples], dtype=torch.long, device=device)
    action = torch.tensor([item.action for item in examples], dtype=torch.long, device=device)
    if image_style == "augmented":
        variant_ids = torch.tensor(
            [(seed + idx * 17 + item.visual * 3 + item.target_goal) % 7 for idx, item in enumerate(examples)],
            dtype=torch.long,
            device=device,
        )
        image = render_augmented_panel_images(visual, device=device, variant_ids=variant_ids)
    elif image_style == "ood":
        variant_ids = torch.full_like(visual, 7)
        image = render_augmented_panel_images(visual, device=device, variant_ids=variant_ids)
    elif image_style == "far_ood":
        image = render_ood_panel_images(visual, device=device)
    else:
        raise ValueError(f"unknown image_style: {image_style}")
    return Batch(
        image=image,
        text=render_query_text_tokens(observed_goal, target_goal, query),
        telemetry_values=render_telemetry(telemetry, device=device),
        visual=visual,
        goal=target_goal,
        telemetry=telemetry,
        action=action,
    )


def random_batch(
    examples: list[QueryExample],
    *,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
    image_style: str,
) -> Batch:
    selected = rng.choices(examples, k=batch_size)
    return make_query_batch(selected, device=device, image_style=image_style, seed=rng.randrange(1_000_000))


@torch.no_grad()
def evaluate_model(
    model: TinyOmniTransformer,
    examples: list[QueryExample],
    config: OmniConfig,
    *,
    device: torch.device,
    image_style: str,
) -> dict[str, float]:
    model.eval()
    totals = {
        "answer_exact": 0.0,
        "action_exact": 0.0,
        "visual_exact": 0.0,
        "goal_exact": 0.0,
        "telemetry_exact": 0.0,
        "format_valid": 0.0,
    }
    count = 0
    for start in range(0, len(examples), config.batch_size):
        batch = make_query_batch(examples[start : start + config.batch_size], device=device, image_style=image_style, seed=10_000 + start)
        targets = answer_targets(batch)
        predicted = greedy_generate(model, batch, latent_tokens=config.latent_tokens, mode="latent_bottleneck")
        totals["answer_exact"] += float((predicted == targets).all(dim=1).sum().detach().cpu())
        totals["action_exact"] += float((predicted[:, 0] == targets[:, 0]).sum().detach().cpu())
        totals["visual_exact"] += float((predicted[:, 1] == targets[:, 1]).sum().detach().cpu())
        totals["goal_exact"] += float((predicted[:, 2] == targets[:, 2]).sum().detach().cpu())
        totals["telemetry_exact"] += float((predicted[:, 3] == targets[:, 3]).sum().detach().cpu())
        totals["format_valid"] += float(format_valid(predicted).sum().detach().cpu())
        count += len(batch.action)
    model.train()
    return {key: value / count for key, value in totals.items()}


def train_model(
    model: TinyOmniTransformer,
    train: list[QueryExample],
    val: list[QueryExample],
    config: OmniConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    image_style: str,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device, image_style=image_style)
        targets = answer_targets(batch)
        answer_inputs = answer_inputs_from_targets(targets)
        output = model(batch, answer_inputs=answer_inputs, latent_tokens=config.latent_tokens, mode="latent_bottleneck")
        loss = answer_loss(output["logits"], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_model(model, val, config, device=device, image_style=image_style)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"answer={metrics['answer_exact']:.3f} action={metrics['action_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def write_sample_grid(examples: list[QueryExample], *, output_dir: Path, device: torch.device, image_style: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = make_query_batch(examples[:8], device=device, image_style=image_style, seed=123)
    first_row = torch.cat([batch.image[idx] for idx in range(4)], dim=2)
    second_row = torch.cat([batch.image[idx] for idx in range(4, 8)], dim=2)
    write_png(output_dir / f"h5_panel_grid_{image_style}.png", torch.cat([first_row, second_row], dim=1))


def sample_outputs(
    model: TinyOmniTransformer,
    examples: list[QueryExample],
    config: OmniConfig,
    *,
    device: torch.device,
    image_style: str,
) -> list[dict[str, str]]:
    batch = make_query_batch(examples[:8], device=device, image_style=image_style, seed=456)
    targets = answer_targets(batch)
    predicted = greedy_generate(model, batch, latent_tokens=config.latent_tokens, mode="latent_bottleneck")
    return [
        {
            "query": "counterfactual" if example.query == QUERY_COUNTERFACTUAL else "diagnose",
            "target": answer_to_text(targets[idx].detach().cpu().tolist()),
            "prediction": answer_to_text(predicted[idx].detach().cpu().tolist()),
        }
        for idx, example in enumerate(examples[:8])
    ]


def run_visual_generalization(config: H5Config, *, model_config: OmniConfig, device: torch.device, output_dir: Path) -> dict[str, object]:
    train = generate_query_examples(config.train_size, seed=config.seed + 11)
    val = generate_query_examples(config.val_size, seed=config.seed + 12)
    test = generate_query_examples(config.test_size, seed=config.seed + 13)
    write_sample_grid(test, output_dir=output_dir / "samples" / f"seed{config.seed}", device=device, image_style="augmented")
    write_sample_grid(test, output_dir=output_dir / "samples" / f"seed{config.seed}", device=device, image_style="ood")

    model = TinyOmniTransformer(model_config)
    training = train_model(
        model,
        train,
        val,
        model_config,
        steps=config.visual_steps,
        seed=config.seed + 1000,
        device=device,
        image_style="augmented",
        label="h5_visual_augmented",
    )
    return {
        "training": training,
        "metrics": {
            "augmented_id_style": evaluate_model(model, test, model_config, device=device, image_style="augmented"),
            "unseen_visual_style": evaluate_model(model, test, model_config, device=device, image_style="ood"),
            "far_unseen_visual_style": evaluate_model(model, test, model_config, device=device, image_style="far_ood"),
        },
        "sample_outputs": {
            "unseen_visual_style": sample_outputs(model, test, model_config, device=device, image_style="ood")
        },
    }


def run_compositional_generalization(config: H5Config, *, model_config: OmniConfig, device: torch.device) -> dict[str, object]:
    heldout = heldout_target_triples()
    train = generate_compositional_query_examples(config.train_size, seed=config.seed + 21, heldout_target_triples=heldout)
    val = generate_compositional_query_examples(config.val_size, seed=config.seed + 22, heldout_target_triples=heldout)
    seen_test = generate_compositional_query_examples(config.test_size, seed=config.seed + 23, heldout_target_triples=heldout)
    heldout_test = generate_compositional_query_examples(
        config.test_size,
        seed=config.seed + 24,
        heldout_target_triples=heldout,
        only_heldout=True,
    )
    model = TinyOmniTransformer(model_config)
    training = train_model(
        model,
        train,
        val,
        model_config,
        steps=config.compositional_steps,
        seed=config.seed + 2000,
        device=device,
        image_style="augmented",
        label="h5_compositional",
    )
    return {
        "heldout_triple_count": len(heldout),
        "training": training,
        "metrics": {
            "seen_target_triples": evaluate_model(model, seen_test, model_config, device=device, image_style="augmented"),
            "heldout_target_triples": evaluate_model(model, heldout_test, model_config, device=device, image_style="augmented"),
        },
        "sample_outputs": {
            "heldout_target_triples": sample_outputs(model, heldout_test, model_config, device=device, image_style="augmented")
        },
    }


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
        for key, value in collect_numbers({"visual": run["visual"], "compositional": run["compositional"]}).items():
            if "training_seconds" not in key and "parameter_count" not in key:
                buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_h5_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def run_experiment(config: H5Config, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_h5 device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    model_config = omni_config(config)
    output = {
        "experiment": "omni_transformer_stage_h5",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "visual": run_visual_generalization(config, model_config=model_config, device=device, output_dir=output_path.parent),
        "compositional": run_compositional_generalization(config, model_config=model_config, device=device),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    return output


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_h5/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_h5/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_h5/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--visual-steps", type=int, default=950)
    parser.add_argument("--compositional-steps", type=int, default=950)
    parser.add_argument("--latent-tokens", type=int, default=8)
    args = parser.parse_args()

    base_config = H5Config(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        visual_steps=args.visual_steps,
        compositional_steps=args.compositional_steps,
        latent_tokens=args.latent_tokens,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = H5Config(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
