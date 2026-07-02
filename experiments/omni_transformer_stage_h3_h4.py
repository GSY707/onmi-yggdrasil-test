from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import time
from typing import Iterable

import torch
from torch.nn import functional as F

from multimodal_fusion_latent_flow import (
    ACTION_COUNT,
    GOALS,
    IMAGE_SIZE,
    TELEMETRY_CHANNELS,
    TELEMETRY_STATES,
    VISUAL_STATES,
    Batch,
    DiagnosticExample,
    action_for,
    generate_examples,
    render_panel_images,
    render_telemetry,
    stat,
)
from visual_multimodal_stage_ab import write_png
from omni_transformer_stage_h import (
    ANSWER_LEN,
    ANSWER_VOCAB,
    EOS_TOKEN,
    GOAL_BASE,
    OmniConfig,
    TELEMETRY_BASE,
    TinyOmniTransformer,
    VISUAL_BASE,
    answer_inputs_from_targets,
    answer_loss,
    answer_targets,
    answer_to_text,
    evaluate_omni,
    format_valid,
    greedy_generate,
    token_to_text,
    train_latent_probe,
    train_omni_model,
)


QUERY_DIAGNOSE = 0
QUERY_COUNTERFACTUAL = 1


@dataclass(frozen=True)
class H3H4Config:
    train_size: int = 4096
    val_size: int = 1024
    test_size: int = 2048
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    lr: float = 8e-4
    latent_counts: tuple[int, ...] = (0, 1, 2, 4, 8, 16)
    capacity_steps: int = 650
    h4_steps: int = 850
    heldout_steps: int = 850
    probe_steps: int = 180
    h4_latent_tokens: int = 8
    eval_every: int = 325


@dataclass(frozen=True)
class QueryExample:
    visual: int
    observed_goal: int
    target_goal: int
    telemetry: int
    query: int
    action: int


def omni_config(config: H3H4Config, *, latent_tokens: int) -> OmniConfig:
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
        bottleneck_steps=config.capacity_steps,
        probe_steps=config.probe_steps,
        latent_tokens=latent_tokens,
        eval_every=config.eval_every,
    )


def generate_query_examples(
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
                    action=action_for(visual, target_goal, telemetry),
                )
            )
            if len(examples) == size:
                break
    rng.shuffle(examples)
    return examples


def heldout_target_triples() -> set[tuple[int, int, int]]:
    return {
        (visual, goal, telemetry)
        for visual in range(VISUAL_STATES)
        for goal in range(GOALS)
        for telemetry in range(TELEMETRY_STATES)
        if (visual * 3 + goal * 5 + telemetry * 7) % 4 == 0
    }


def render_query_text_tokens(observed_goal: torch.Tensor, target_goal: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    tokens = torch.zeros((observed_goal.shape[0], 5), dtype=torch.long, device=observed_goal.device)
    tokens[:, 0] = 1
    tokens[:, 1] = 3 + observed_goal
    tokens[:, 2] = 7 + query
    tokens[:, 3] = 11 + target_goal
    tokens[:, 4] = 2
    return tokens


def render_ood_panel_images(visual_ids: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    batch = int(visual_ids.shape[0])
    image = torch.full((batch, 3, IMAGE_SIZE, IMAGE_SIZE), 0.18, device=device)
    image[:, :, 4:-4, 4:-4] = 0.24
    palette = torch.tensor(
        [
            [0.05, 0.95, 0.92],
            [0.95, 0.25, 0.88],
            [0.92, 0.92, 0.92],
            [0.98, 0.52, 0.12],
        ],
        device=device,
    )
    colors = palette[visual_ids]
    image[:, :, 6:19, 26:41] = colors[:, :, None, None]

    mask = visual_ids == 0
    if bool(mask.any()):
        image[mask, :, 25:29, 7:41] = colors[mask, :, None, None]
        image[mask, :, 29:39, 18:23] = colors[mask, :, None, None] * 0.55
    mask = visual_ids == 1
    if bool(mask.any()):
        image[mask, :, 24:30, 8:40] = colors[mask, :, None, None]
        image[mask, :, 18:38, 8:13] = colors[mask, :, None, None] * 0.9
        image[mask, :, 32:38, 31:40] = colors[mask, :, None, None] * 0.75
    mask = visual_ids == 2
    if bool(mask.any()):
        image[mask, :, 22:26, 8:40] = colors[mask, :, None, None]
        image[mask, :, 28:32, 8:40] = colors[mask, :, None, None] * 0.70
        image[mask, :, 34:38, 8:40] = colors[mask, :, None, None] * 0.55
    mask = visual_ids == 3
    if bool(mask.any()):
        image[mask, :, 22:39, 8:16] = colors[mask, :, None, None]
        image[mask, :, 22:39, 32:40] = colors[mask, :, None, None]
        image[mask, :, 27:34, 16:32] = 0.08
    return image.clamp(0.0, 1.0)


def make_query_batch(examples: list[QueryExample], *, device: torch.device, image_style: str = "train") -> Batch:
    visual = torch.tensor([item.visual for item in examples], dtype=torch.long, device=device)
    observed_goal = torch.tensor([item.observed_goal for item in examples], dtype=torch.long, device=device)
    target_goal = torch.tensor([item.target_goal for item in examples], dtype=torch.long, device=device)
    telemetry = torch.tensor([item.telemetry for item in examples], dtype=torch.long, device=device)
    query = torch.tensor([item.query for item in examples], dtype=torch.long, device=device)
    action = torch.tensor([item.action for item in examples], dtype=torch.long, device=device)
    if image_style == "train":
        image = render_panel_images(visual, device=device)
    elif image_style == "ood":
        image = render_ood_panel_images(visual, device=device)
    else:
        raise ValueError(f"unknown image style: {image_style}")
    return Batch(
        image=image,
        text=render_query_text_tokens(observed_goal, target_goal, query),
        telemetry_values=render_telemetry(telemetry, device=device),
        visual=visual,
        goal=target_goal,
        telemetry=telemetry,
        action=action,
    )


def random_query_batch(examples: list[QueryExample], *, rng: random.Random, batch_size: int, device: torch.device) -> Batch:
    return make_query_batch(rng.choices(examples, k=batch_size), device=device)


def write_query_sample_grid(examples: list[QueryExample], *, output_dir: Path, device: torch.device, image_style: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = make_query_batch(examples[:8], device=device, image_style=image_style)
    first_row = torch.cat([batch.image[idx] for idx in range(4)], dim=2)
    second_row = torch.cat([batch.image[idx] for idx in range(4, 8)], dim=2)
    write_png(output_dir / f"query_panel_grid_{image_style}.png", torch.cat([first_row, second_row], dim=1))


@torch.no_grad()
def evaluate_query_model(
    model: TinyOmniTransformer,
    examples: list[QueryExample],
    config: OmniConfig,
    *,
    device: torch.device,
    image_style: str = "train",
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
        batch = make_query_batch(examples[start : start + config.batch_size], device=device, image_style=image_style)
        targets = answer_targets(batch)
        predicted = greedy_generate(
            model,
            batch,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
        )
        totals["answer_exact"] += float((predicted == targets).all(dim=1).sum().detach().cpu())
        totals["action_exact"] += float((predicted[:, 0] == targets[:, 0]).sum().detach().cpu())
        totals["visual_exact"] += float((predicted[:, 1] == targets[:, 1]).sum().detach().cpu())
        totals["goal_exact"] += float((predicted[:, 2] == targets[:, 2]).sum().detach().cpu())
        totals["telemetry_exact"] += float((predicted[:, 3] == targets[:, 3]).sum().detach().cpu())
        totals["format_valid"] += float(format_valid(predicted).sum().detach().cpu())
        count += len(batch.action)
    model.train()
    return {name: value / count for name, value in totals.items()}


def train_query_model(
    model: TinyOmniTransformer,
    train: list[QueryExample],
    val: list[QueryExample],
    config: OmniConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_query_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        targets = answer_targets(batch)
        answer_inputs = answer_inputs_from_targets(targets)
        output = model(
            batch,
            answer_inputs=answer_inputs,
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
        )
        loss = answer_loss(output["logits"], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_query_model(model, val, config, device=device)
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


def sample_query_outputs(
    model: TinyOmniTransformer,
    examples: list[QueryExample],
    config: OmniConfig,
    *,
    device: torch.device,
    image_style: str = "train",
) -> list[dict[str, str]]:
    batch = make_query_batch(examples[:8], device=device, image_style=image_style)
    targets = answer_targets(batch)
    predicted = greedy_generate(model, batch, latent_tokens=config.latent_tokens, mode="latent_bottleneck")
    samples = []
    for idx, example in enumerate(examples[:8]):
        samples.append(
            {
                "query": "counterfactual" if example.query == QUERY_COUNTERFACTUAL else "diagnose",
                "observed_goal": str(example.observed_goal),
                "target": answer_to_text(targets[idx].detach().cpu().tolist()),
                "prediction": answer_to_text(predicted[idx].detach().cpu().tolist()),
            }
        )
    return samples


def run_h3_capacity(config: H3H4Config, *, device: torch.device, output_dir: Path) -> dict[str, object]:
    train = generate_examples(config.train_size, seed=config.seed + 1)
    val = generate_examples(config.val_size, seed=config.seed + 2)
    test = generate_examples(config.test_size, seed=config.seed + 3)
    result: dict[str, object] = {}
    for latent_count in config.latent_counts:
        k_config = omni_config(config, latent_tokens=latent_count)
        model = TinyOmniTransformer(k_config)
        print(f"h3 latent_count={latent_count}", flush=True)
        training = train_omni_model(
            model,
            train,
            val,
            k_config,
            steps=config.capacity_steps,
            seed=config.seed + 1000 + latent_count,
            device=device,
            latent_tokens=latent_count,
            mode="latent_bottleneck",
            label=f"h3_k{latent_count}",
        )
        metrics = evaluate_omni(
            model,
            test,
            k_config,
            device=device,
            latent_tokens=latent_count,
            mode="latent_bottleneck",
        )
        probe: dict[str, object] | None = None
        if latent_count > 0:
            probe = train_latent_probe(model, train, test, k_config, device=device)
        result[f"k_{latent_count}"] = {"metrics": metrics, "training": training, "latent_probe": probe}
    return result


def run_h4_ood(config: H3H4Config, *, device: torch.device, output_dir: Path) -> dict[str, object]:
    h4_config = omni_config(config, latent_tokens=config.h4_latent_tokens)
    train = generate_query_examples(config.train_size, seed=config.seed + 11)
    val = generate_query_examples(config.val_size, seed=config.seed + 12)
    test = generate_query_examples(config.test_size, seed=config.seed + 13)
    counter_test = generate_query_examples(
        config.test_size,
        seed=config.seed + 14,
        counterfactual_only=True,
    )
    write_query_sample_grid(test, output_dir=output_dir / "samples" / f"seed{config.seed}", device=device, image_style="train")
    write_query_sample_grid(test, output_dir=output_dir / "samples" / f"seed{config.seed}", device=device, image_style="ood")

    model = TinyOmniTransformer(h4_config)
    training = train_query_model(
        model,
        train,
        val,
        h4_config,
        steps=config.h4_steps,
        seed=config.seed + 2000,
        device=device,
        label="h4_counterfactual",
    )
    metrics = {
        "id_mixed_query": evaluate_query_model(model, test, h4_config, device=device),
        "counterfactual_only": evaluate_query_model(model, counter_test, h4_config, device=device),
        "unseen_visual_style": evaluate_query_model(model, test, h4_config, device=device, image_style="ood"),
    }

    heldout = heldout_target_triples()
    heldout_train = generate_query_examples(config.train_size, seed=config.seed + 21, heldout_target_triples=heldout)
    heldout_val = generate_query_examples(config.val_size, seed=config.seed + 22, heldout_target_triples=heldout)
    heldout_seen_test = generate_query_examples(config.test_size, seed=config.seed + 23, heldout_target_triples=heldout)
    heldout_test = generate_query_examples(
        config.test_size,
        seed=config.seed + 24,
        heldout_target_triples=heldout,
        only_heldout=True,
    )
    heldout_model = TinyOmniTransformer(h4_config)
    heldout_training = train_query_model(
        heldout_model,
        heldout_train,
        heldout_val,
        h4_config,
        steps=config.heldout_steps,
        seed=config.seed + 3000,
        device=device,
        label="h4_heldout",
    )
    heldout_metrics = {
        "seen_target_triples": evaluate_query_model(heldout_model, heldout_seen_test, h4_config, device=device),
        "heldout_target_triples": evaluate_query_model(heldout_model, heldout_test, h4_config, device=device),
    }

    return {
        "counterfactual_model": {
            "metrics": metrics,
            "training": training,
            "sample_outputs": sample_query_outputs(model, counter_test, h4_config, device=device),
        },
        "heldout_model": {
            "heldout_triple_count": len(heldout),
            "metrics": heldout_metrics,
            "training": heldout_training,
            "sample_outputs": sample_query_outputs(heldout_model, heldout_test, h4_config, device=device),
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
        for key, value in collect_numbers({"h3": run["h3"], "h4": run["h4"]}).items():
            if "training_seconds" not in key and "parameter_count" not in key:
                buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_h3_h4_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def run_experiment(config: H3H4Config, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_h3_h4 device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    output = {
        "experiment": "omni_transformer_stage_h3_h4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "h3": run_h3_capacity(config, device=device, output_dir=output_path.parent),
        "h4": run_h4_ood(config, device=device, output_dir=output_path.parent),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)
    return output


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_h3_h4/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_h3_h4/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_h3_h4/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--latent-counts", default="0,1,2,4,8,16")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--capacity-steps", type=int, default=650)
    parser.add_argument("--h4-steps", type=int, default=850)
    parser.add_argument("--heldout-steps", type=int, default=850)
    parser.add_argument("--probe-steps", type=int, default=180)
    parser.add_argument("--h4-latent-tokens", type=int, default=8)
    args = parser.parse_args()

    base_config = H3H4Config(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        latent_counts=parse_csv_ints(args.latent_counts),
        capacity_steps=args.capacity_steps,
        h4_steps=args.h4_steps,
        heldout_steps=args.heldout_steps,
        probe_steps=args.probe_steps,
        h4_latent_tokens=args.h4_latent_tokens,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = H3H4Config(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
