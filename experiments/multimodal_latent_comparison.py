from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import statistics
import time
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from multimodal_latent_pipeline import (
    GRID_SIZE,
    LatentCodec,
    LatentReasoner,
    PipelineConfig,
    TextThoughtModel,
    batch_examples,
    coord_loss,
    evaluate_latent,
    evaluate_text,
    final_loss,
    split_examples,
    train_codec,
    train_latent_reasoner,
    train_text_model,
)


@dataclass(frozen=True)
class ComparisonConfig:
    move_count: int = 10
    train_size: int = 4096
    val_size: int = 768
    test_size: int = 1024
    batch_size: int = 256
    seed: int = 20260701
    d_text: int = 128
    d_latent: int = 64
    lr: float = 9e-4
    text_steps: int = 700
    codec_steps: int = 300
    parallel_steps: int = 2500
    output_steps: int = 300
    direct_steps: int = 3500
    direct_cot_steps: int = 3500
    latent_scratch_steps: int = 2800
    eval_every: int = 700
    parallel_step_loss_weight: float = 2.0
    parallel_final_loss_weight: float = 0.5
    parallel_latent_loss_weight: float = 1.0
    parallel_codec_loss_weight: float = 5.0


class DirectAnswerModel(nn.Module):
    def __init__(self, d_text: int) -> None:
        super().__init__()
        self.text = TextThoughtModel(d_text)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, final_logits = self.text(tokens)
        return final_logits


class WideDirectAnswerModel(nn.Module):
    def __init__(self, d_text: int, width_multiplier: int = 2) -> None:
        super().__init__()
        hidden = d_text * width_multiplier
        self.text = TextThoughtModel(hidden)
        self.adapter = nn.Linear(hidden, GRID_SIZE * GRID_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, final_logits = self.text(tokens)
        return final_logits


def pipeline_config(config: ComparisonConfig) -> PipelineConfig:
    return PipelineConfig(
        move_count=config.move_count,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        d_text=config.d_text,
        d_latent=config.d_latent,
        text_steps=config.text_steps,
        codec_steps=config.codec_steps,
        parallel_steps=config.parallel_steps,
        output_steps=config.output_steps,
        eval_every=config.eval_every,
        lr=config.lr,
        parallel_step_loss_weight=config.parallel_step_loss_weight,
        parallel_final_loss_weight=config.parallel_final_loss_weight,
        parallel_latent_loss_weight=config.parallel_latent_loss_weight,
        parallel_codec_loss_weight=config.parallel_codec_loss_weight,
    )


def train_direct_answer(
    model: nn.Module,
    train,
    val,
    config: ComparisonConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        tokens, _, states = batch_examples(train, rng=rng, batch_size=config.batch_size, device=device)
        logits = model(tokens)
        loss = final_loss(logits, states)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_direct_answer(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} final={metrics['final_exact']:.3f}", flush=True)
    return {"history": history}


@torch.no_grad()
def evaluate_direct_answer(model: nn.Module, examples, config: ComparisonConfig, *, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    for start in range(0, len(examples), config.batch_size):
        batch = examples[start : start + config.batch_size]
        tokens, _, states = batch_examples(batch, rng=random.Random(0), batch_size=len(batch), device=device)
        logits = model(tokens)
        correct += int((logits.argmax(dim=-1) == states[:, -1]).sum().detach().cpu())
    model.train()
    return {"final_exact": correct / len(examples)}


def train_latent_without_codec(
    model: LatentReasoner,
    train,
    val,
    config: ComparisonConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        tokens, _, states = batch_examples(train, rng=rng, batch_size=config.batch_size, device=device)
        _, step_logits, final_logits = model(tokens)
        loss = coord_loss(step_logits, states) + final_loss(final_logits, states)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_latent(model, val, pipeline_config(config), device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} final={metrics['final_exact']:.3f}", flush=True)
    return {"history": history}


def train_latent_scratch_with_codec(
    model: LatentReasoner,
    train,
    val,
    config: ComparisonConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    pcfg = pipeline_config(config)
    for step in range(1, steps + 1):
        tokens, _, states = batch_examples(train, rng=rng, batch_size=config.batch_size, device=device)
        latents, step_logits, final_logits = model(tokens)
        with torch.no_grad():
            target_latents = model.codec.encode(states)
        loss = (
            config.parallel_final_loss_weight * final_loss(final_logits, states)
            + config.parallel_step_loss_weight * coord_loss(step_logits, states)
            + config.parallel_latent_loss_weight * F.mse_loss(latents, target_latents)
            + config.parallel_codec_loss_weight * coord_loss(model.codec.decode(latents), states)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_latent(model, val, pcfg, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), **metrics})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} final={metrics['final_exact']:.3f}", flush=True)
    return {"history": history}


def freeze_codec(codec: LatentCodec) -> None:
    for parameter in codec.parameters():
        parameter.requires_grad = False
    codec.eval()


def random_bad_codec(d_latent: int, *, device: torch.device, seed: int) -> LatentCodec:
    torch.manual_seed(seed)
    codec = LatentCodec(d_latent).to(device)
    freeze_codec(codec)
    return codec


def run_comparison(config: ComparisonConfig, output_path: Path) -> dict[str, object]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = split_examples(pipeline_config(config))
    print(f"comparison device={device} move_count={config.move_count} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    results: dict[str, object] = {}

    direct = DirectAnswerModel(config.d_text).to(device)
    direct_log = train_direct_answer(
        direct,
        train,
        val,
        config,
        steps=config.direct_steps,
        seed=config.seed + 100,
        device=device,
        label="direct_text_only",
    )
    results["direct_text_only"] = {
        "description": "Text prompt directly predicts final answer; no thought chain, no latent.",
        "train_log": direct_log,
        "test_metrics": evaluate_direct_answer(direct, test, config, device=device),
    }

    visible = TextThoughtModel(config.d_text).to(device)
    visible_log = train_text_model(visible, train, val, pipeline_config(config), seed=config.seed + 110, device=device)
    results["visible_text_cot"] = {
        "description": "Text prompt predicts visible step chain plus final answer.",
        "train_log": visible_log,
        "test_metrics": evaluate_text(visible, test, pipeline_config(config), device=device),
    }

    codec = LatentCodec(config.d_latent).to(device)
    codec_log = train_codec(codec, pipeline_config(config), device=device)
    results["latent_codec"] = {
        "description": "Good codec trained to translate text state to latent and back.",
        "train_log": codec_log,
        "test_metrics": codec_log["history"][-1],
    }

    pipeline_model = LatentReasoner(config.d_text, config.d_latent, codec).to(device)
    parallel_log = train_latent_reasoner(
        pipeline_model,
        train,
        val,
        pipeline_config(config),
        stage="parallel",
        seed=config.seed + 120,
        device=device,
    )
    output_log = train_latent_reasoner(
        pipeline_model,
        train,
        val,
        pipeline_config(config),
        stage="output",
        seed=config.seed + 121,
        device=device,
    )
    results["full_pipeline"] = {
        "description": "Pure text + good codec + parallel latent/text thought + final text output.",
        "train_log": {"parallel": parallel_log, "output": output_log},
        "test_metrics": evaluate_latent(pipeline_model, test, pipeline_config(config), device=device),
    }

    scratch_model = LatentReasoner(config.d_text, config.d_latent, codec).to(device)
    scratch_log = train_latent_scratch_with_codec(
        scratch_model,
        train,
        val,
        config,
        steps=config.latent_scratch_steps,
        seed=config.seed + 130,
        device=device,
        label="latent_from_scratch_good_codec",
    )
    results["latent_from_scratch_good_codec"] = {
        "description": "Latent reasoner from scratch with the same good codec constraints.",
        "train_log": scratch_log,
        "test_metrics": evaluate_latent(scratch_model, test, pipeline_config(config), device=device),
    }

    no_codec_model = LatentReasoner(config.d_text, config.d_latent, codec).to(device)
    no_codec_log = train_latent_without_codec(
        no_codec_model,
        train,
        val,
        config,
        steps=config.parallel_steps + config.output_steps,
        seed=config.seed + 140,
        device=device,
        label="latent_without_codec_alignment",
    )
    results["latent_without_codec_alignment"] = {
        "description": "Latent reasoner trained with step/final text losses but no codec/latent alignment loss.",
        "train_log": no_codec_log,
        "test_metrics": evaluate_latent(no_codec_model, test, pipeline_config(config), device=device),
    }

    bad_codec = random_bad_codec(config.d_latent, device=device, seed=config.seed + 150)
    bad_model = LatentReasoner(config.d_text, config.d_latent, bad_codec).to(device)
    bad_log = train_latent_scratch_with_codec(
        bad_model,
        train,
        val,
        config,
        steps=config.parallel_steps + config.output_steps,
        seed=config.seed + 151,
        device=device,
        label="latent_with_bad_codec",
    )
    results["latent_with_bad_codec"] = {
        "description": "Latent reasoner constrained by a random frozen bad codec.",
        "train_log": bad_log,
        "test_metrics": evaluate_latent(bad_model, test, pipeline_config(config), device=device),
    }

    wide_direct = WideDirectAnswerModel(config.d_text).to(device)
    wide_direct_log = train_direct_answer(
        wide_direct,
        train,
        val,
        config,
        steps=config.direct_steps,
        seed=config.seed + 160,
        device=device,
        label="wide_direct_equal_budget",
    )
    results["wide_direct_equal_budget"] = {
        "description": "Larger direct baseline with the same final-answer training budget.",
        "train_log": wide_direct_log,
        "test_metrics": evaluate_direct_answer(wide_direct, test, config, device=device),
    }

    output = {
        "experiment": "multimodal_latent_pipeline_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": config.__dict__,
        "results": results,
        "summary": summarize_single(results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def final_metric(results: dict[str, object], key: str) -> float:
    return float(results[key]["test_metrics"]["final_exact"])  # type: ignore[index]


def codec_metric(results: dict[str, object], key: str) -> float:
    metrics = results[key]["test_metrics"]  # type: ignore[index]
    return float(metrics.get("codec_step_exact", 0.0))


def summarize_single(results: dict[str, object]) -> dict[str, object]:
    return {
        "final_exact": {
            key: final_metric(results, key)
            for key in results
            if "final_exact" in results[key]["test_metrics"]  # type: ignore[index]
        },
        "codec_step_exact": {
            key: codec_metric(results, key)
            for key in results
            if "codec_step_exact" in results[key]["test_metrics"]  # type: ignore[index]
        },
        "full_pipeline_minus_direct": final_metric(results, "full_pipeline") - final_metric(results, "direct_text_only"),
        "full_pipeline_minus_visible": final_metric(results, "full_pipeline") - final_metric(results, "visible_text_cot"),
        "full_pipeline_minus_scratch": final_metric(results, "full_pipeline")
        - final_metric(results, "latent_from_scratch_good_codec"),
        "full_pipeline_minus_no_codec": final_metric(results, "full_pipeline")
        - final_metric(results, "latent_without_codec_alignment"),
        "good_codec_minus_bad_codec": final_metric(results, "full_pipeline")
        - final_metric(results, "latent_with_bad_codec"),
        "full_pipeline_minus_wide_direct": final_metric(results, "full_pipeline")
        - final_metric(results, "wide_direct_equal_budget"),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    move_counts = [int(item) for item in args.move_counts.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    runs: list[dict[str, object]] = []
    started = time.perf_counter()
    for move_count in move_counts:
        for seed in seeds:
            config = ComparisonConfig(
                move_count=move_count,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
                batch_size=args.batch_size,
                seed=seed,
                d_text=args.d_text,
                d_latent=args.d_latent,
                lr=args.lr,
                text_steps=args.text_steps,
                codec_steps=args.codec_steps,
                parallel_steps=args.parallel_steps,
                output_steps=args.output_steps,
                direct_steps=args.direct_steps,
                direct_cot_steps=args.direct_cot_steps,
                latent_scratch_steps=args.latent_scratch_steps,
                eval_every=args.eval_every,
                parallel_step_loss_weight=args.parallel_step_loss_weight,
                parallel_final_loss_weight=args.parallel_final_loss_weight,
                parallel_latent_loss_weight=args.parallel_latent_loss_weight,
                parallel_codec_loss_weight=args.parallel_codec_loss_weight,
            )
            output_path = output_dir / f"move{move_count}_seed{seed}.json"
            print(f"=== comparison sweep move={move_count} seed={seed} ===", flush=True)
            result = run_comparison(config, output_path)
            runs.append(
                {
                    "move_count": move_count,
                    "seed": seed,
                    "output": str(output_path),
                    "summary": result["summary"],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    aggregate = {
        "experiment": "multimodal_latent_pipeline_comparison_sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": runs,
        "summary": summarize_sweep(runs),
        "seconds": round(time.perf_counter() - started, 2),
    }
    aggregate_path = Path(args.aggregate)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def summarize_sweep(runs: list[dict[str, object]]) -> dict[str, object]:
    keys = [
        "direct_text_only",
        "visible_text_cot",
        "full_pipeline",
        "latent_from_scratch_good_codec",
        "latent_without_codec_alignment",
        "latent_with_bad_codec",
        "wide_direct_equal_budget",
    ]
    deltas = [
        "full_pipeline_minus_direct",
        "full_pipeline_minus_visible",
        "full_pipeline_minus_scratch",
        "full_pipeline_minus_no_codec",
        "good_codec_minus_bad_codec",
        "full_pipeline_minus_wide_direct",
    ]

    def group_summary(selected: list[dict[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {"count": len(selected)}
        for key in keys:
            values = [float(run["summary"]["final_exact"][key]) for run in selected]  # type: ignore[index]
            result[key] = stat(values)
        for key in deltas:
            values = [float(run["summary"][key]) for run in selected]  # type: ignore[index]
            result[key] = stat(values)
        codec_values = [float(run["summary"]["codec_step_exact"].get("full_pipeline", 0.0)) for run in selected]  # type: ignore[index]
        result["full_pipeline_codec_step"] = stat(codec_values)
        return result

    summary: dict[str, object] = {}
    for move_count in sorted({int(run["move_count"]) for run in runs}):
        summary[str(move_count)] = group_summary([run for run in runs if int(run["move_count"]) == move_count])
    summary["overall"] = group_summary(runs)
    return summary


def stat(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/multimodal_latent_comparison/result.json")
    parser.add_argument("--aggregate", default="artifacts/multimodal_latent_comparison/sweep_results.json")
    parser.add_argument("--output-dir", default="artifacts/multimodal_latent_comparison/sweep_runs")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--move-count", type=int, default=10)
    parser.add_argument("--move-counts", default="10,12")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=768)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-text", type=int, default=128)
    parser.add_argument("--d-latent", type=int, default=64)
    parser.add_argument("--lr", type=float, default=9e-4)
    parser.add_argument("--text-steps", type=int, default=700)
    parser.add_argument("--codec-steps", type=int, default=300)
    parser.add_argument("--parallel-steps", type=int, default=2500)
    parser.add_argument("--output-steps", type=int, default=300)
    parser.add_argument("--direct-steps", type=int, default=3500)
    parser.add_argument("--direct-cot-steps", type=int, default=3500)
    parser.add_argument("--latent-scratch-steps", type=int, default=2800)
    parser.add_argument("--eval-every", type=int, default=700)
    parser.add_argument("--parallel-step-loss-weight", type=float, default=2.0)
    parser.add_argument("--parallel-final-loss-weight", type=float, default=0.5)
    parser.add_argument("--parallel-latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--parallel-codec-loss-weight", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep:
        result = run_sweep(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)
        return
    config = ComparisonConfig(
        move_count=args.move_count,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
        d_text=args.d_text,
        d_latent=args.d_latent,
        lr=args.lr,
        text_steps=args.text_steps,
        codec_steps=args.codec_steps,
        parallel_steps=args.parallel_steps,
        output_steps=args.output_steps,
        direct_steps=args.direct_steps,
        direct_cot_steps=args.direct_cot_steps,
        latent_scratch_steps=args.latent_scratch_steps,
        eval_every=args.eval_every,
        parallel_step_loss_weight=args.parallel_step_loss_weight,
        parallel_final_loss_weight=args.parallel_final_loss_weight,
        parallel_latent_loss_weight=args.parallel_latent_loss_weight,
        parallel_codec_loss_weight=args.parallel_codec_loss_weight,
    )
    result = run_comparison(config, Path(args.output))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
