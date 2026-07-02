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
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import omni_transformer_stage_m_moe_multimodal_llm as stage_m
import omni_transformer_stage_q_tiny_moe_vlm_from_scratch as stage_q
import omni_transformer_stage_r_curriculum_vlm as stage_r


STAGES_NO_PUMP = ("vision", "prompt", "fusion", "expert_concat", "weighted_expert_concat")
STAGES_PUMP = ("pump_latent", "thought_latent")


@dataclass(frozen=True)
class StageSConfig:
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
    candidate_count: int = 8
    no_pump_steps: int = 360
    pump_steps: int = 600
    eval_every: int = 180
    router_loss_weight: float = 0.10
    probe_steps: int = 160
    probe_hidden: int = 192
    levels: tuple[str, ...] = ("l1_color", "l2_object", "l3_position", "l4_count", "l5_relation")


def stage_q_config(config: StageSConfig, *, candidate_count: int) -> stage_q.StageQConfig:
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


def noncausal_mask(length: int, *, device: torch.device) -> torch.Tensor:
    return torch.zeros(length, length, dtype=torch.bool, device=device)


class CrossAttentionAnswerScorer(nn.Module):
    def __init__(self, config: stage_q.StageQConfig) -> None:
        super().__init__()
        self.config = config
        self.answer_encoder = stage_q.TextTokenEncoder(config, length=config.answer_len)
        self.cross = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_answer = nn.LayerNorm(config.d_model)
        self.norm_context = nn.LayerNorm(config.d_model)
        self.score = nn.Sequential(
            nn.LayerNorm(config.d_model * 3),
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, context_tokens: torch.Tensor, answer_tokens: torch.Tensor) -> torch.Tensor:
        batch, candidates, answer_len = answer_tokens.shape
        flat_answers = answer_tokens.reshape(batch * candidates, answer_len)
        answer_hidden = self.answer_encoder(flat_answers)
        repeated_context = (
            context_tokens.unsqueeze(1)
            .expand(batch, candidates, context_tokens.shape[1], context_tokens.shape[2])
            .reshape(batch * candidates, context_tokens.shape[1], context_tokens.shape[2])
        )
        attended, _ = self.cross(
            self.norm_answer(answer_hidden),
            self.norm_context(repeated_context),
            self.norm_context(repeated_context),
            need_weights=False,
        )
        mask = (flat_answers != stage_q.PAD).float().unsqueeze(-1)
        answer_pooled = (answer_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        attended_pooled = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        features = torch.cat((answer_pooled, attended_pooled, answer_pooled * attended_pooled), dim=-1)
        return self.score(features).view(batch, candidates)


class CrossDiagnosticVLM(nn.Module):
    def __init__(self, config: stage_q.StageQConfig, *, mode: str) -> None:
        super().__init__()
        if mode not in {"cross_pump_latent", "cross_no_pump_expert_tokens"}:
            raise ValueError(f"unknown mode: {mode}")
        self.config = config
        self.mode = mode
        self.bank = stage_q.ScratchExpertBank(config)
        self.router = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(stage_q.EXPERTS)),
        )
        self.pump = stage_q.AttentionPump(config)
        self.thought = stage_q.ThoughtExpert(config)
        self.scorer = CrossAttentionAnswerScorer(config)

    def representations(self, batch: stage_q.PixelSet, *, zero_modalities: Iterable[str] = ()) -> dict[str, torch.Tensor]:
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
        pump_latent = self.pump(expert_tokens, route_weights)
        thought_latent = self.thought(pump_latent)
        return {
            "vision": expert_tokens[stage_q.VISION_EXPERT],
            "prompt": expert_tokens[stage_q.PROMPT_EXPERT],
            "history": expert_tokens[stage_q.HISTORY_EXPERT],
            "fusion": expert_tokens[stage_q.FUSION_EXPERT],
            "expert_concat": torch.cat(expert_tokens, dim=1),
            "weighted_expert_concat": torch.cat(weighted, dim=1),
            "pump_latent": pump_latent,
            "thought_latent": thought_latent,
            "route_logits": route_logits,
            "route_weights": route_weights,
        }

    def context(
        self,
        batch: stage_q.PixelSet,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        reps = self.representations(batch, zero_modalities=zero_modalities)
        if self.mode == "cross_pump_latent":
            context = torch.zeros_like(reps["thought_latent"]) if disable_latent_access else reps["thought_latent"]
        else:
            context = reps["weighted_expert_concat"]
        return {"context": context, **reps}

    def forward(
        self,
        batch: stage_q.PixelSet,
        answer_tokens: torch.Tensor,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(batch, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        return {"scores": self.scorer(ctx["context"], answer_tokens), **ctx}


def loss_for_output(output: dict[str, torch.Tensor], labels: torch.Tensor, batch: stage_q.PixelSet, config: stage_q.StageQConfig) -> torch.Tensor:
    rank = F.cross_entropy(output["scores"], labels)
    router = F.binary_cross_entropy_with_logits(output["route_logits"], batch.required_experts)
    return rank + config.router_loss_weight * router


@torch.no_grad()
def evaluate_ranking(
    model: CrossDiagnosticVLM,
    data: stage_q.PixelSet,
    config: stage_q.StageQConfig,
    *,
    device: torch.device,
    candidate_pool: stage_q.CandidatePool,
    seed: int,
    zero_modalities: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    hits: list[float] = []
    reciprocal: list[float] = []
    for start in range(0, len(data.examples), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(data.examples))))
        batch = data.subset(indices).to(device)
        rows, true_indices = stage_q.candidate_index_rows(batch.examples, candidate_pool, config.candidate_count, seed=seed + start)
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        output = model(batch, answers, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        order = output["scores"].argsort(dim=1, descending=True).detach().cpu().tolist()
        for row_order, true_index in zip(order, true_indices):
            rank = row_order.index(true_index) + 1
            hits.append(1.0 if rank == 1 else 0.0)
            reciprocal.append(1.0 / rank)
    model.train()
    return {
        "rank_top1": sum(hits) / max(len(hits), 1),
        "rank_mrr": sum(reciprocal) / max(len(reciprocal), 1),
        "candidate_count": config.candidate_count,
    }


def train_model(
    model: CrossDiagnosticVLM,
    train: stage_q.PixelSet,
    val: stage_q.PixelSet,
    config: stage_q.StageQConfig,
    *,
    candidate_pool: stage_q.CandidatePool,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    best_top1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = stage_q.random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        rows, true_indices = stage_q.candidate_index_rows(batch.examples, candidate_pool, config.candidate_count, seed=rng.randrange(1_000_000_000))
        answers = stage_q.answer_tokens_from_indices(candidate_pool, rows)
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        output = model(batch, answers)
        loss = loss_for_output(output, labels, batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_ranking(model, val, config, device=device, candidate_pool=candidate_pool, seed=seed + step)
            if float(metrics["rank_top1"]) > best_top1:
                best_top1 = float(metrics["rank_top1"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "rank_top1": metrics["rank_top1"], "rank_mrr": metrics["rank_mrr"]})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} top1={metrics['rank_top1']:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return {
        "history": history,
        "best_val_rank_top1": best_top1,
        "training_seconds": round(time.perf_counter() - started, 3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


class TokenReconstructionProbe(nn.Module):
    def __init__(self, config: stage_q.StageQConfig, class_count: int, hidden: int) -> None:
        super().__init__()
        self.pool = stage_q.QueryResampler(config.d_model, config.heads, 1, config.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, class_count),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(tokens).squeeze(1))


def answer_labels(data: stage_q.PixelSet, answer_to_label: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([answer_to_label[example.answer] for example in data.examples], dtype=torch.long, device=device)


def train_reconstruction_probe(
    model: CrossDiagnosticVLM,
    stage_name: str,
    train: stage_q.PixelSet,
    test: stage_q.PixelSet,
    config: stage_q.StageQConfig,
    *,
    answer_to_label: dict[str, int],
    steps: int,
    hidden: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    class_count = len(answer_to_label)
    probe = TokenReconstructionProbe(config, class_count, hidden).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=0.01)
    rng = random.Random(seed)
    started = time.perf_counter()
    for _ in range(steps):
        indices = [rng.randrange(len(train.examples)) for _ in range(config.batch_size)]
        batch = train.subset(indices).to(device)
        labels = answer_labels(batch, answer_to_label, device)
        with torch.no_grad():
            tokens = model.representations(batch)[stage_name].detach()
        logits = probe(tokens)
        loss = F.cross_entropy(logits, labels)
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
        "class_count": class_count,
        "random_exact": 1.0 / max(class_count, 1),
        "training_seconds": round(time.perf_counter() - started, 3),
    }


def train_all_probes(
    no_pump_model: CrossDiagnosticVLM,
    pump_model: CrossDiagnosticVLM,
    train: stage_q.PixelSet,
    test: stage_q.PixelSet,
    config: stage_q.StageQConfig,
    stage_s_config: StageSConfig,
    *,
    answer_to_label: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    probes: dict[str, object] = {}
    for index, stage_name in enumerate(STAGES_NO_PUMP):
        probes[f"no_pump_model.{stage_name}"] = train_reconstruction_probe(
            no_pump_model,
            stage_name,
            train,
            test,
            config,
            answer_to_label=answer_to_label,
            steps=stage_s_config.probe_steps,
            hidden=stage_s_config.probe_hidden,
            seed=stage_s_config.seed + 9000 + index * 31,
            device=device,
        )
    for index, stage_name in enumerate(STAGES_PUMP):
        probes[f"pump_model.{stage_name}"] = train_reconstruction_probe(
            pump_model,
            stage_name,
            train,
            test,
            config,
            answer_to_label=answer_to_label,
            steps=stage_s_config.probe_steps,
            hidden=stage_s_config.probe_hidden,
            seed=stage_s_config.seed + 9500 + index * 31,
            device=device,
        )
    return probes


def run_level(level: str, config: StageSConfig, *, device: torch.device, output_dir: Path) -> dict[str, object]:
    data_config = stage_r.StageRConfig(
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
    pixel_sets, data_stats, meta = stage_r.build_level_data(level, data_config)
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
        "cross_pump_latent": CrossDiagnosticVLM(model_config, mode="cross_pump_latent"),
        "cross_no_pump_expert_tokens": CrossDiagnosticVLM(model_config, mode="cross_no_pump_expert_tokens"),
    }
    training = {
        "cross_pump_latent": train_model(
            models["cross_pump_latent"],
            train,
            val,
            model_config,
            candidate_pool=candidate_pool,
            steps=config.pump_steps,
            seed=config.seed + 1000 + stage_r.LEVELS.index(level) * 100,
            device=device,
            label=f"{level}:cross_pump_latent",
        ),
        "cross_no_pump_expert_tokens": train_model(
            models["cross_no_pump_expert_tokens"],
            train,
            val,
            model_config,
            candidate_pool=candidate_pool,
            steps=config.no_pump_steps,
            seed=config.seed + 2000 + stage_r.LEVELS.index(level) * 100,
            device=device,
            label=f"{level}:cross_no_pump_expert_tokens",
        ),
    }
    metrics = {
        name: evaluate_ranking(model, test, model_config, device=device, candidate_pool=candidate_pool, seed=config.seed + 3000 + index)
        for index, (name, model) in enumerate(models.items())
    }
    ablations = {
        "cross_pump_no_latent_access": evaluate_ranking(
            models["cross_pump_latent"],
            test,
            model_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4000,
            disable_latent_access=True,
        ),
        "cross_pump_no_image_modality": evaluate_ranking(
            models["cross_pump_latent"],
            test,
            model_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4001,
            zero_modalities=("image",),
        ),
        "cross_no_pump_no_image_modality": evaluate_ranking(
            models["cross_no_pump_expert_tokens"],
            test,
            model_config,
            device=device,
            candidate_pool=candidate_pool,
            seed=config.seed + 4002,
            zero_modalities=("image",),
        ),
    }
    reconstruction = train_all_probes(
        models["cross_no_pump_expert_tokens"],
        models["cross_pump_latent"],
        train,
        test,
        model_config,
        config,
        answer_to_label=answer_to_label,
        device=device,
    )
    return {
        "level": level,
        "description": stage_r.LEVEL_DESCRIPTIONS[level],
        "config": asdict(model_config),
        "data": data_stats,
        "metrics": metrics,
        "ablations": ablations,
        "reconstruction": reconstruction,
        "training": training,
    }


def run_experiment(config: StageSConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_s_scorer_reconstruction device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    levels = {}
    started = time.perf_counter()
    for level in config.levels:
        print(f"running scorer/reconstruction level {level}: {stage_r.LEVEL_DESCRIPTIONS[level]}", flush=True)
        levels[level] = run_level(level, config, device=device, output_dir=output_path.parent / level)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "experiment": "omni_transformer_stage_s_scorer_reconstruction",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "base": "Stage Q/R tiny from-scratch expert bank",
            "cross_pump_latent": "replace final mean-dot scorer with candidate answer cross-attention over thought latent tokens",
            "cross_no_pump_expert_tokens": "remove Attention Pump from answer path; candidate answer cross-attends weighted expert tokens directly",
            "reconstruction": {
                "no_pump_model": STAGES_NO_PUMP,
                "pump_model": STAGES_PUMP,
            },
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
        numbers = collect_numbers({"levels": run["levels"], "total_seconds": run["total_seconds"]})
        for key, value in numbers.items():
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
            "stage_r_mean_dot_moe_top1": None,
            "cross_pump_latent_top1": mean("metrics.cross_pump_latent.rank_top1"),
            "cross_pump_no_image_top1": mean("ablations.cross_pump_no_image_modality.rank_top1"),
            "cross_no_pump_top1": mean("metrics.cross_no_pump_expert_tokens.rank_top1"),
            "cross_no_pump_no_image_top1": mean("ablations.cross_no_pump_no_image_modality.rank_top1"),
            "reconstruction": {
                stage: mean(f"reconstruction.{stage}.exact")
                for stage in (
                    "no_pump_model.vision",
                    "no_pump_model.prompt",
                    "no_pump_model.fusion",
                    "no_pump_model.expert_concat",
                    "no_pump_model.weighted_expert_concat",
                    "pump_model.pump_latent",
                    "pump_model.thought_latent",
                )
            },
        }
    return {
        "experiment": "omni_transformer_stage_s_scorer_reconstruction_sweep",
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_s_scorer_reconstruction/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_s_scorer_reconstruction/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_s_scorer_reconstruction/sweep_results.json"))
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
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--no-pump-steps", type=int, default=360)
    parser.add_argument("--pump-steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=180)
    parser.add_argument("--probe-steps", type=int, default=160)
    args = parser.parse_args()

    base_config = StageSConfig(
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
        no_pump_steps=args.no_pump_steps,
        pump_steps=args.pump_steps,
        eval_every=args.eval_every,
        probe_steps=args.probe_steps,
        levels=parse_csv_strings(args.levels),
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageSConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
