from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import torch

from multimodal_fusion_latent_flow import GOALS, TELEMETRY_STATES, VISUAL_STATES, Batch, render_telemetry, stat
from omni_transformer_stage_h import (
    ACTION_BASE,
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
    answer_to_text,
    greedy_generate,
)
from omni_transformer_stage_h3_h4 import QueryExample, heldout_target_triples, render_query_text_tokens
from omni_transformer_stage_h5 import (
    compositional_action,
    make_query_batch,
    render_augmented_panel_images,
    write_sample_grid,
)


QUERY_FULL = 0
QUERY_ACTION_ONLY = 1
QUERY_READ_VISUAL = 2
QUERY_READ_GOAL = 3
QUERY_READ_TELEMETRY = 4
QUERY_NAMES = {
    QUERY_FULL: "full",
    QUERY_ACTION_ONLY: "action_only",
    QUERY_READ_VISUAL: "read_visual",
    QUERY_READ_GOAL: "read_goal",
    QUERY_READ_TELEMETRY: "read_telemetry",
}


@dataclass(frozen=True)
class H6Config:
    train_size: int = 8192
    val_size: int = 1536
    test_size: int = 2048
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    lr: float = 8e-4
    latent_tokens: int = 8
    train_steps: int = 1200
    eval_every: int = 600


@dataclass(frozen=True)
class MultiOutputExample:
    visual: int
    observed_goal: int
    target_goal: int
    telemetry: int
    query: int
    action: int


def omni_config(config: H6Config) -> OmniConfig:
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


def generate_multi_examples(
    size: int,
    *,
    seed: int,
    heldout_target_triples: set[tuple[int, int, int]] | None = None,
    only_heldout: bool = False,
) -> list[MultiOutputExample]:
    heldout_target_triples = heldout_target_triples or set()
    combos = [
        (visual, observed_goal, target_goal, telemetry, query)
        for visual in range(VISUAL_STATES)
        for observed_goal in range(GOALS)
        for target_goal in range(GOALS)
        for telemetry in range(TELEMETRY_STATES)
        for query in QUERY_NAMES
    ]
    filtered: list[tuple[int, int, int, int, int]] = []
    for visual, observed_goal, target_goal, telemetry, query in combos:
        target_triple = (visual, target_goal, telemetry)
        is_heldout = target_triple in heldout_target_triples
        if only_heldout and not is_heldout:
            continue
        if not only_heldout and is_heldout:
            continue
        filtered.append((visual, observed_goal, target_goal, telemetry, query))

    rng = random.Random(seed)
    examples: list[MultiOutputExample] = []
    while len(examples) < size:
        rng.shuffle(filtered)
        for visual, observed_goal, target_goal, telemetry, query in filtered:
            examples.append(
                MultiOutputExample(
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


def multi_text_tokens(observed_goal: torch.Tensor, target_goal: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    tokens = torch.zeros((observed_goal.shape[0], 5), dtype=torch.long, device=observed_goal.device)
    tokens[:, 0] = 1
    tokens[:, 1] = 3 + observed_goal
    tokens[:, 2] = 7 + query
    tokens[:, 3] = 12 + target_goal
    tokens[:, 4] = 2
    return tokens


def make_batch(
    examples: list[MultiOutputExample],
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
    elif image_style == "ood":
        variant_ids = torch.full_like(visual, 7)
    else:
        raise ValueError(f"unknown image_style: {image_style}")
    return Batch(
        image=render_augmented_panel_images(visual, device=device, variant_ids=variant_ids),
        text=multi_text_tokens(observed_goal, target_goal, query),
        telemetry_values=render_telemetry(telemetry, device=device),
        visual=visual,
        goal=target_goal,
        telemetry=telemetry,
        action=action,
    )


def multi_targets(batch: Batch, examples: list[MultiOutputExample]) -> torch.Tensor:
    targets = torch.full((len(examples), ANSWER_LEN), EOS_TOKEN, dtype=torch.long, device=batch.action.device)
    query = torch.tensor([item.query for item in examples], dtype=torch.long, device=batch.action.device)
    full = query == QUERY_FULL
    action_only = query == QUERY_ACTION_ONLY
    read_visual = query == QUERY_READ_VISUAL
    read_goal = query == QUERY_READ_GOAL
    read_telemetry = query == QUERY_READ_TELEMETRY

    targets[full, 0] = ACTION_BASE + batch.action[full]
    targets[full, 1] = VISUAL_BASE + batch.visual[full]
    targets[full, 2] = GOAL_BASE + batch.goal[full]
    targets[full, 3] = TELEMETRY_BASE + batch.telemetry[full]
    targets[action_only, 0] = ACTION_BASE + batch.action[action_only]
    targets[read_visual, 0] = VISUAL_BASE + batch.visual[read_visual]
    targets[read_goal, 0] = GOAL_BASE + batch.goal[read_goal]
    targets[read_telemetry, 0] = TELEMETRY_BASE + batch.telemetry[read_telemetry]
    return targets


def random_batch(
    examples: list[MultiOutputExample],
    *,
    rng: random.Random,
    batch_size: int,
    device: torch.device,
    image_style: str,
) -> tuple[Batch, list[MultiOutputExample]]:
    selected = rng.choices(examples, k=batch_size)
    return make_batch(selected, device=device, image_style=image_style, seed=rng.randrange(1_000_000)), selected


def exact_by_query(predicted: torch.Tensor, targets: torch.Tensor, examples: list[MultiOutputExample]) -> dict[str, float]:
    output: dict[str, float] = {}
    exact = (predicted == targets).all(dim=1)
    for query, name in QUERY_NAMES.items():
        mask = torch.tensor([item.query == query for item in examples], dtype=torch.bool, device=predicted.device)
        output[f"{name}_exact"] = float(exact[mask].float().mean().detach().cpu()) if bool(mask.any()) else 0.0
    return output


@torch.no_grad()
def evaluate_model(
    model: TinyOmniTransformer,
    examples: list[MultiOutputExample],
    config: OmniConfig,
    *,
    device: torch.device,
    image_style: str,
) -> dict[str, float]:
    model.eval()
    totals = {"answer_exact": 0.0}
    per_query_sums = {f"{name}_exact": 0.0 for name in QUERY_NAMES.values()}
    per_query_counts = {f"{name}_exact": 0 for name in QUERY_NAMES.values()}
    count = 0
    for start in range(0, len(examples), config.batch_size):
        batch_examples = examples[start : start + config.batch_size]
        batch = make_batch(batch_examples, device=device, image_style=image_style, seed=10_000 + start)
        targets = multi_targets(batch, batch_examples)
        predicted = greedy_generate(model, batch, latent_tokens=config.latent_tokens, mode="latent_bottleneck")
        exact = (predicted == targets).all(dim=1)
        totals["answer_exact"] += float(exact.sum().detach().cpu())
        by_query = exact_by_query(predicted, targets, batch_examples)
        for query, name in QUERY_NAMES.items():
            key = f"{name}_exact"
            query_count = sum(1 for item in batch_examples if item.query == query)
            per_query_sums[key] += by_query[key] * query_count
            per_query_counts[key] += query_count
        count += len(batch_examples)
    metrics = {"answer_exact": totals["answer_exact"] / count}
    for key, value in per_query_sums.items():
        metrics[key] = value / per_query_counts[key]
    model.train()
    return metrics


def train_model(
    model: TinyOmniTransformer,
    train: list[MultiOutputExample],
    val: list[MultiOutputExample],
    config: OmniConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch, selected = random_batch(train, rng=rng, batch_size=config.batch_size, device=device, image_style="augmented")
        targets = multi_targets(batch, selected)
        output = model(
            batch,
            answer_inputs=answer_inputs_from_targets(targets),
            latent_tokens=config.latent_tokens,
            mode="latent_bottleneck",
        )
        loss = answer_loss(output["logits"], targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_model(model, val, config, device=device, image_style="augmented")
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(
                f"h6_multi_output step={step:4d} loss={float(loss.detach().cpu()):.4f} "
                f"answer={metrics['answer_exact']:.3f}",
                flush=True,
            )
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def sample_outputs(
    model: TinyOmniTransformer,
    examples: list[MultiOutputExample],
    config: OmniConfig,
    *,
    device: torch.device,
    image_style: str,
) -> list[dict[str, str]]:
    selected = examples[:10]
    batch = make_batch(selected, device=device, image_style=image_style, seed=456)
    targets = multi_targets(batch, selected)
    predicted = greedy_generate(model, batch, latent_tokens=config.latent_tokens, mode="latent_bottleneck")
    return [
        {
            "query": QUERY_NAMES[example.query],
            "target": answer_to_text(targets[idx].detach().cpu().tolist()),
            "prediction": answer_to_text(predicted[idx].detach().cpu().tolist()),
        }
        for idx, example in enumerate(selected)
    ]


def run_experiment(config: H6Config, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_h6 device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    model_config = omni_config(config)
    heldout = heldout_target_triples()
    train = generate_multi_examples(config.train_size, seed=config.seed + 1, heldout_target_triples=heldout)
    val = generate_multi_examples(config.val_size, seed=config.seed + 2, heldout_target_triples=heldout)
    seen_test = generate_multi_examples(config.test_size, seed=config.seed + 3, heldout_target_triples=heldout)
    heldout_test = generate_multi_examples(
        config.test_size,
        seed=config.seed + 4,
        heldout_target_triples=heldout,
        only_heldout=True,
    )

    model = TinyOmniTransformer(model_config)
    training = train_model(
        model,
        train,
        val,
        model_config,
        steps=config.train_steps,
        seed=config.seed + 1000,
        device=device,
    )
    output = {
        "experiment": "omni_transformer_stage_h6_multi_output",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "heldout_triple_count": len(heldout),
        "metrics": {
            "seen_augmented_style": evaluate_model(model, seen_test, model_config, device=device, image_style="augmented"),
            "seen_unseen_style": evaluate_model(model, seen_test, model_config, device=device, image_style="ood"),
            "heldout_augmented_style": evaluate_model(model, heldout_test, model_config, device=device, image_style="augmented"),
            "heldout_unseen_style": evaluate_model(model, heldout_test, model_config, device=device, image_style="ood"),
        },
        "training": training,
        "sample_outputs": {
            "heldout_unseen_style": sample_outputs(model, heldout_test, model_config, device=device, image_style="ood")
        },
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
        for key, value in collect_numbers({"metrics": run["metrics"]}).items():
            buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_h6_multi_output_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_h6/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_h6/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_h6/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1536)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--latent-tokens", type=int, default=8)
    args = parser.parse_args()

    base_config = H6Config(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        train_steps=args.train_steps,
        latent_tokens=args.latent_tokens,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = H6Config(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
