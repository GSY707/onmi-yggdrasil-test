from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import omni_transformer_stage_m_moe_multimodal_llm as stage_m
import omni_transformer_stage_q_tiny_moe_vlm_from_scratch as stage_q
import omni_transformer_stage_r_curriculum_vlm as stage_r
import omni_transformer_stage_s_scorer_reconstruction as stage_s


MODEL_MODES = ("wide_residual_latent", "slot_resampler_latent")
PROBE_STAGES = (
    "expert_concat",
    "weighted_expert_concat",
    "wide_source_concat",
    "wide_residual_latent",
    "slot_resampler_latent",
)


@dataclass(frozen=True)
class StageTConfig:
    train_size: int = 1536
    val_size: int = 384
    test_size: int = 512
    batch_size: int = 64
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
    slot_latent_tokens: int = 32
    summary_tokens: int = 8
    candidate_count: int = 8
    train_steps: int = 600
    eval_every: int = 200
    router_loss_weight: float = 0.10
    probe_steps: int = 160
    probe_hidden: int = 192
    levels: tuple[str, ...] = ("l1_color", "l2_object", "l3_position", "l4_count", "l5_relation")


def stage_q_config(config: StageTConfig, *, candidate_count: int) -> stage_q.StageQConfig:
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
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
    )


def stage_r_config(config: StageTConfig) -> stage_r.StageRConfig:
    return stage_r.StageRConfig(
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
        candidate_count=config.candidate_count,
        eval_every=config.eval_every,
        router_loss_weight=config.router_loss_weight,
        probe_steps=config.probe_steps,
        levels=config.levels,
        models=("direct", "moe_latent"),
    )


class WideResidualLatentCompressor(nn.Module):
    def __init__(self, config: stage_q.StageQConfig, *, summary_tokens: int) -> None:
        super().__init__()
        self.summary = stage_q.QueryResampler(config.d_model, config.heads, summary_tokens, config.dropout)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # Preserve the raw expert tokens exactly. Summary tokens may compress,
        # but they are appended and cannot overwrite the original evidence.
        return torch.cat((source, self.summary(source)), dim=1)


class SlotResamplerLatentCompressor(nn.Module):
    def __init__(self, config: stage_q.StageQConfig, *, slot_count: int) -> None:
        super().__init__()
        self.resampler = stage_q.QueryResampler(config.d_model, config.heads, slot_count, config.dropout)
        self.thought = stage_q.ThoughtExpert(config)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.thought(self.resampler(source))


class PreserveLatentVLM(nn.Module):
    def __init__(self, config: stage_q.StageQConfig, stage_t_config: StageTConfig, *, mode: str) -> None:
        super().__init__()
        if mode not in MODEL_MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.config = config
        self.stage_t_config = stage_t_config
        self.mode = mode
        self.bank = stage_q.ScratchExpertBank(config)
        self.router = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(stage_q.EXPERTS)),
        )
        self.wide_residual = WideResidualLatentCompressor(config, summary_tokens=stage_t_config.summary_tokens)
        self.slot_resampler = SlotResamplerLatentCompressor(config, slot_count=stage_t_config.slot_latent_tokens)
        self.scorer = stage_s.CrossAttentionAnswerScorer(config)

    def representations(self, batch: stage_q.PixelSet, *, zero_modalities: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
        expert_tokens = self.bank(batch, zero_modalities=zero_modalities)
        route_source = torch.cat(
            (
                expert_tokens[stage_q.VISION_EXPERT].mean(dim=1),
                expert_tokens[stage_q.PROMPT_EXPERT].mean(dim=1),
            ),
            dim=-1,
        )
        route_logits = self.router(route_source)
        route_weights = torch.sigmoid(route_logits)
        weighted = [tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)]
        expert_concat = torch.cat(expert_tokens, dim=1)
        weighted_expert_concat = torch.cat(weighted, dim=1)
        wide_source = torch.cat((expert_concat, weighted_expert_concat), dim=1)
        return {
            "vision": expert_tokens[stage_q.VISION_EXPERT],
            "prompt": expert_tokens[stage_q.PROMPT_EXPERT],
            "fusion": expert_tokens[stage_q.FUSION_EXPERT],
            "expert_concat": expert_concat,
            "weighted_expert_concat": weighted_expert_concat,
            "wide_source_concat": wide_source,
            "wide_residual_latent": self.wide_residual(wide_source),
            "slot_resampler_latent": self.slot_resampler(expert_concat),
            "route_logits": route_logits,
            "route_weights": route_weights,
        }

    def context(
        self,
        batch: stage_q.PixelSet,
        *,
        zero_modalities: tuple[str, ...] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        reps = self.representations(batch, zero_modalities=zero_modalities)
        context = reps[self.mode]
        if disable_latent_access:
            context = torch.zeros_like(context)
        return {"context": context, **reps}

    def forward(
        self,
        batch: stage_q.PixelSet,
        answer_tokens: torch.Tensor,
        *,
        zero_modalities: tuple[str, ...] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(batch, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        return {"scores": self.scorer(ctx["context"], answer_tokens), **ctx}


def answer_labels(data: stage_q.PixelSet, answer_to_label: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([answer_to_label[example.answer] for example in data.examples], dtype=torch.long, device=device)


def train_reconstruction_probe(
    model: PreserveLatentVLM,
    stage_name: str,
    train: stage_q.PixelSet,
    test: stage_q.PixelSet,
    config: stage_q.StageQConfig,
    stage_t_config: StageTConfig,
    *,
    answer_to_label: dict[str, int],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = stage_s.TokenReconstructionProbe(config, len(answer_to_label), stage_t_config.probe_hidden).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(stage_t_config.probe_steps):
        indices = [rng.randrange(len(train.examples)) for _ in range(config.batch_size)]
        batch = train.subset(indices).to(device)
        labels = answer_labels(batch, answer_to_label, device)
        with torch.no_grad():
            tokens = model.representations(batch)[stage_name].detach()
        logits = probe(tokens)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    hits: list[float] = []
    with torch.no_grad():
        for start in range(0, len(test.examples), config.batch_size):
            batch = test.subset(list(range(start, min(start + config.batch_size, len(test.examples))))).to(device)
            labels = answer_labels(batch, answer_to_label, device)
            tokens = model.representations(batch)[stage_name]
            pred = probe(tokens).argmax(dim=1)
            hits.extend((pred == labels).float().detach().cpu().tolist())
    return {
        "exact": sum(hits) / max(len(hits), 1),
        "class_count": len(answer_to_label),
        "random_exact": 1.0 / max(len(answer_to_label), 1),
        "training_seconds": round(time.perf_counter() - started, 3),
    }


def train_all_probes(
    model: PreserveLatentVLM,
    train: stage_q.PixelSet,
    test: stage_q.PixelSet,
    config: stage_q.StageQConfig,
    stage_t_config: StageTConfig,
    *,
    answer_to_label: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    return {
        stage_name: train_reconstruction_probe(
            model,
            stage_name,
            train,
            test,
            config,
            stage_t_config,
            answer_to_label=answer_to_label,
            seed=stage_t_config.seed + 9000 + index * 37,
            device=device,
        )
        for index, stage_name in enumerate(PROBE_STAGES)
    }


def run_level(level: str, config: StageTConfig, *, device: torch.device, output_dir: Path) -> dict[str, object]:
    pixel_sets, data_stats, meta = stage_r.build_level_data(level, stage_r_config(config))
    answer_pool_values = [example.answer for split in pixel_sets.values() for example in split.examples]
    unique_answers = list(dict.fromkeys(answer_pool_values))
    candidate_count = min(config.candidate_count, len(unique_answers))
    model_config = stage_q_config(config, candidate_count=candidate_count)
    train = pixel_sets["train"].to(device)
    val = pixel_sets["val"].to(device)
    test = pixel_sets["test"].to(device)
    stage_r.write_samples(pixel_sets["test"], meta["test"], output_dir / "samples")
    candidate_pool = stage_q.build_candidate_pool(answer_pool_values, model_config, device=device)
    answer_to_label = {answer: index for index, answer in enumerate(unique_answers)}
    data_stats["candidate_count"] = candidate_count
    data_stats["random_top1"] = 1.0 / candidate_count
    data_stats["answer_class_count"] = len(answer_to_label)

    models = {
        mode: PreserveLatentVLM(model_config, config, mode=mode)
        for mode in MODEL_MODES
    }
    training = {
        mode: stage_s.train_model(
            model,
            train,
            val,
            model_config,
            candidate_pool=candidate_pool,
            steps=config.train_steps,
            seed=config.seed + 1000 + stage_r.LEVELS.index(level) * 100 + index * 19,
            device=device,
            label=f"{level}:{mode}",
        )
        for index, (mode, model) in enumerate(models.items())
    }
    metrics = {
        mode: stage_s.evaluate_ranking(
            model,
            test,
            model_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 3000 + index,
        )
        for index, (mode, model) in enumerate(models.items())
    }
    ablations = {}
    for index, (mode, model) in enumerate(models.items()):
        ablations[f"{mode}_no_image_modality"] = stage_s.evaluate_ranking(
            model,
            test,
            model_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4000 + index,
            zero_modalities=("image",),
        )
        ablations[f"{mode}_no_latent_access"] = stage_s.evaluate_ranking(
            model,
            test,
            model_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4100 + index,
            disable_latent_access=True,
        )
    reconstruction = {
        mode: train_all_probes(
            model,
            train,
            test,
            model_config,
            config,
            answer_to_label=answer_to_label,
            device=device,
        )
        for mode, model in models.items()
    }
    return {
        "level": level,
        "description": stage_r.LEVEL_DESCRIPTIONS[level],
        "config": asdict(model_config),
        "stage_t_config": asdict(config),
        "data": data_stats,
        "metrics": metrics,
        "ablations": ablations,
        "reconstruction": reconstruction,
        "training": training,
    }


def run_experiment(config: StageTConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_t_preserve_latent device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    levels = {}
    started = time.perf_counter()
    for level in config.levels:
        print(f"running preserve-latent level {level}: {stage_r.LEVEL_DESCRIPTIONS[level]}", flush=True)
        levels[level] = run_level(level, config, device=device, output_dir=output_path.parent / level)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "experiment": "omni_transformer_stage_t_preserve_latent",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "wide_residual_latent": "keeps raw expert tokens and route-weighted expert tokens, appends summary tokens, answer cross-attends all latent tokens",
            "slot_resampler_latent": "uses 32 resampler slots over expert tokens, then thought Transformer, answer cross-attends slot latent tokens",
            "principle": "prefer retaining information with more latent tokens over early lossy compression",
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "levels": levels,
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
        for key, value in collect_numbers({"levels": run["levels"], "total_seconds": run["total_seconds"]}).items():
            buckets.setdefault(key, []).append(value)
    stats = {key: stage_m.stat(values) for key, values in sorted(buckets.items())}
    level_summary: dict[str, object] = {}
    for level in runs[0]["config"]["levels"]:
        def mean(path: str) -> float | None:
            item = stats.get(f"levels.{level}.{path}")
            return None if item is None else float(item["mean"])

        level_summary[level] = {
            "description": stage_r.LEVEL_DESCRIPTIONS[level],
            "random_top1": mean("data.random_top1"),
            "wide_residual_top1": mean("metrics.wide_residual_latent.rank_top1"),
            "wide_residual_no_image_top1": mean("ablations.wide_residual_latent_no_image_modality.rank_top1"),
            "slot_resampler_top1": mean("metrics.slot_resampler_latent.rank_top1"),
            "slot_resampler_no_image_top1": mean("ablations.slot_resampler_latent_no_image_modality.rank_top1"),
            "reconstruction": {
                f"wide.{stage_name}": mean(f"reconstruction.wide_residual_latent.{stage_name}.exact")
                for stage_name in PROBE_STAGES
            }
            | {
                f"slot.{stage_name}": mean(f"reconstruction.slot_resampler_latent.{stage_name}.exact")
                for stage_name in PROBE_STAGES
            },
        }
    return {
        "experiment": "omni_transformer_stage_t_preserve_latent_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": stats,
        "level_summary": level_summary,
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_t_preserve_latent/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_t_preserve_latent/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_t_preserve_latent/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--levels", default="l1_color,l2_object,l3_position,l4_count,l5_relation")
    parser.add_argument("--train-size", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=384)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=96)
    parser.add_argument("--answer-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--slot-latent-tokens", type=int, default=32)
    parser.add_argument("--summary-tokens", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--probe-steps", type=int, default=160)
    args = parser.parse_args()

    base_config = StageTConfig(
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
        slot_latent_tokens=args.slot_latent_tokens,
        summary_tokens=args.summary_tokens,
        candidate_count=args.candidate_count,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        probe_steps=args.probe_steps,
        levels=parse_csv_strings(args.levels),
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageTConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
