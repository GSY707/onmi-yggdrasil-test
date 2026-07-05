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
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omni_transformer_stage_ave_bidirectional_latent_external import (  # noqa: E402
    PAIR_TOKENS_PER_EXAMPLE,
    PAD,
    SEQ_LEN,
    TEXT_LEN,
    TOKEN_VOCAB,
    ZONES,
    AVESet,
    BidirectionalLatentExternalModel,
    StageAVEConfig,
    evaluate,
    latent_align_loss,
    load_datasets,
    parameter_count,
    random_batch,
    sequence_loss,
    write_samples,
)


TEXT_START = 1 + ZONES
TEXT_END = TEXT_START + TEXT_LEN
TEXT_POSITIONS = torch.tensor([0, *range(TEXT_START, TEXT_END)], dtype=torch.long)


@dataclass(frozen=True)
class StageSpec:
    name: str
    steps: int
    purpose: str


@dataclass(frozen=True)
class StageAVGConfig:
    dataset_manifest: str = "artifacts/omni_transformer_stage_avf_10m_bidirectional_dataset/dataset_10m/manifest.json"
    batch_size: int = 512
    seed: int = 20260705
    d_model: int = 768
    heads: int = 12
    layers: int = 10
    latent_tokens: int = 8
    lr: float = 5e-4
    text_steps: int = 400
    codec_steps: int = 1200
    translate_steps: int = 2200
    reason_steps: int = 1400
    joint_steps: int = 800
    eval_every: int = 250
    eval_batch_size: int = 128
    sample_count: int = 8
    amp: bool = True
    train_limit: int = 0
    val_limit: int = 0
    test_limit: int = 0
    heldout_limit: int = 0
    gpu_resident_data: bool = True
    checkpoint_dir: str = "artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_checkpoints"
    resume: bool = False
    save_every: int = 250
    device: str = "auto"

    @property
    def total_steps(self) -> int:
        return self.text_steps + self.codec_steps + self.translate_steps + self.reason_steps + self.joint_steps


def to_ave_config(config: StageAVGConfig) -> StageAVEConfig:
    return StageAVEConfig(
        batch_size=config.batch_size,
        seed=config.seed,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        latent_tokens=config.latent_tokens,
        lr=config.lr,
        steps=config.total_steps,
        eval_every=config.eval_every,
        eval_batch_size=config.eval_batch_size,
        sample_count=config.sample_count,
        amp=config.amp,
        dataset_manifest=config.dataset_manifest,
        train_limit=config.train_limit,
        val_limit=config.val_limit,
        test_limit=config.test_limit,
        heldout_limit=config.heldout_limit,
        gpu_resident_data=config.gpu_resident_data,
        checkpoint_dir=config.checkpoint_dir,
        resume=config.resume,
        save_every=config.save_every,
    )


def build_stage_plan(config: StageAVGConfig) -> list[StageSpec]:
    return [
        StageSpec("text_latent_base", config.text_steps, "Only text-operation positions train a seed latent/text codec."),
        StageSpec("external_codec", config.codec_steps, "Full source and target external states reconstruct through latent."),
        StageSpec("bidirectional_translate", config.translate_steps, "Source latent translates to target external and target latent translates back to source external."),
        StageSpec("latent_reason", config.reason_steps, "Target latent trains the answer head while replaying light codec/translation losses."),
        StageSpec("joint_debug", config.joint_steps, "Short final joint tuning, not a replacement for the earlier stages."),
    ]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested --device cuda, but CUDA is not available")
    return torch.device(name)


def text_only_batch(batch: AVESet, device: torch.device) -> AVESet:
    positions = TEXT_POSITIONS.to(device=batch.source_tokens.device)
    source = torch.full_like(batch.source_tokens, PAD)
    target = torch.full_like(batch.target_tokens, PAD)
    source.index_copy_(1, positions, batch.source_tokens.index_select(1, positions))
    target.index_copy_(1, positions, batch.target_tokens.index_select(1, positions))
    return AVESet([], source.to(device=device, dtype=torch.long), target.to(device=device, dtype=torch.long), batch.answer.to(device=device, dtype=torch.long))


def text_sequence_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positions = TEXT_POSITIONS.to(device=target.device)
    selected_logits = logits.index_select(1, positions)
    selected_target = target.index_select(1, positions)
    return F.cross_entropy(selected_logits.reshape(-1, TOKEN_VOCAB), selected_target.reshape(-1))


def stage_loss(model: BidirectionalLatentExternalModel, batch: AVESet, stage: str, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    if stage == "text_latent_base":
        text_batch = text_only_batch(batch, device)
        out = model(text_batch)
        source_text = text_sequence_loss(out["source_recon"], text_batch.source_tokens)
        target_text = text_sequence_loss(out["target_recon"], text_batch.target_tokens)
        loss = 0.5 * source_text + 0.5 * target_text
        return loss, {"loss_source_text": float(source_text.detach().item()), "loss_target_text": float(target_text.detach().item())}

    out = model(batch)
    source_recon = sequence_loss(out["source_recon"], batch.source_tokens)
    target_recon = sequence_loss(out["target_recon"], batch.target_tokens)
    source_to_target = sequence_loss(out["source_to_target"], batch.target_tokens)
    target_to_source = sequence_loss(out["target_to_source"], batch.source_tokens)
    answer = F.cross_entropy(out["answer"], batch.answer)
    edit_align = latent_align_loss(out["edited_latent"], out["target_latent"])
    inverse_align = latent_align_loss(out["inverse_latent"], out["source_latent"])

    if stage == "external_codec":
        loss = 0.5 * source_recon + 0.5 * target_recon
    elif stage == "bidirectional_translate":
        loss = (
            source_to_target
            + target_to_source
            + 0.25 * edit_align
            + 0.25 * inverse_align
            + 0.10 * source_recon
            + 0.10 * target_recon
        )
    elif stage == "latent_reason":
        loss = answer + 0.10 * target_recon + 0.05 * source_to_target + 0.05 * target_to_source
    elif stage == "joint_debug":
        loss = (
            0.5 * source_recon
            + 0.5 * target_recon
            + source_to_target
            + target_to_source
            + answer
            + 0.2 * edit_align
            + 0.2 * inverse_align
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")

    return loss, {
        "loss_source_recon": float(source_recon.detach().item()),
        "loss_target_recon": float(target_recon.detach().item()),
        "loss_source_to_target": float(source_to_target.detach().item()),
        "loss_target_to_source": float(target_to_source.detach().item()),
        "loss_answer": float(answer.detach().item()),
        "loss_edit_align": float(edit_align.detach().item()),
        "loss_inverse_align": float(inverse_align.detach().item()),
    }


def checkpoint_dir(config: StageAVGConfig) -> Path | None:
    if not config.checkpoint_dir:
        return None
    path = Path(config.checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    path: Path,
    model: BidirectionalLatentExternalModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: StageAVGConfig,
    global_step: int,
    history: list[dict[str, float]],
    best_score: float,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "stage": "AV-G",
            "strict_stage_schedule": True,
            "config": asdict(config),
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_score": best_score,
        },
        path,
    )


def best_metric_score(metrics: dict[str, float]) -> float:
    return (
        metrics["val_source_recon_exact"]
        + metrics["val_target_recon_exact"]
        + metrics["val_source_to_target_exact"]
        + metrics["val_target_to_source_exact"]
        + metrics["val_answer_accuracy"]
    )


def maybe_evaluate(
    model: BidirectionalLatentExternalModel,
    val: AVESet,
    config: StageAVGConfig,
    device: torch.device,
    stage: StageSpec,
    global_step: int,
    loss: torch.Tensor,
    components: dict[str, float],
    history: list[dict[str, float]],
    best_score: float,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> float:
    if global_step != 1 and global_step % config.eval_every != 0 and global_step != config.total_steps:
        return best_score
    metrics = evaluate(model, val, prefix="val", device=device, batch_size=config.eval_batch_size)
    metrics.update(components)
    metrics["global_step"] = float(global_step)
    metrics["stage"] = stage.name
    metrics["loss"] = float(loss.detach().item())
    metrics["processed_examples"] = float(global_step * config.batch_size)
    metrics["processed_pair_tokens"] = float(global_step * config.batch_size * PAIR_TOKENS_PER_EXAMPLE)
    history.append(metrics)
    score = best_metric_score(metrics)
    ckpt_dir = checkpoint_dir(config)
    if ckpt_dir is not None and score > best_score:
        best_score = score
        save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scaler, config, global_step, history, best_score)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return best_score


def train_strict_stages(
    model: BidirectionalLatentExternalModel,
    train: AVESet,
    val: AVESet,
    config: StageAVGConfig,
    device: torch.device,
) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    rng = random.Random(config.seed + 97)
    ckpt_dir = checkpoint_dir(config)
    latest_path = ckpt_dir / "latest.pt" if ckpt_dir is not None else None
    history: list[dict[str, float]] = []
    completed_global_step = 0
    best_score = -1.0
    resumed_from = ""
    if config.resume and latest_path is not None and latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = list(state.get("history", []))
        completed_global_step = int(state.get("global_step", 0))
        best_score = float(state.get("best_score", best_score))
        resumed_from = str(latest_path)

    global_step = completed_global_step
    stage_start = 1
    for stage in build_stage_plan(config):
        stage_end = stage_start + stage.steps - 1
        if stage.steps <= 0:
            stage_start = stage_end + 1
            continue
        if global_step >= stage_end:
            stage_start = stage_end + 1
            continue
        local_start = max(1, global_step - stage_start + 2)
        for _local_step in range(local_start, stage.steps + 1):
            model.train()
            batch = random_batch(train, rng, config.batch_size, device)
            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                loss, components = stage_loss(model, batch, stage.name, device)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            best_score = maybe_evaluate(
                model,
                val,
                config,
                device,
                stage,
                global_step,
                loss,
                components,
                history,
                best_score,
                optimizer,
                scaler,
            )
            if latest_path is not None and (global_step % config.save_every == 0 or global_step == config.total_steps):
                save_checkpoint(latest_path, model, optimizer, scaler, config, global_step, history, best_score)
        stage_start = stage_end + 1

    return {
        "history": history,
        "completed_steps": global_step,
        "total_steps": config.total_steps,
        "resumed_from": resumed_from,
        "checkpoint_dir": config.checkpoint_dir,
        "latest_checkpoint": str(latest_path) if latest_path is not None else "",
        "best_checkpoint": str(ckpt_dir / "best.pt") if ckpt_dir is not None else "",
        "best_score": best_score,
        "stage_plan": [asdict(stage) for stage in build_stage_plan(config)],
    }


def run_one(config: StageAVGConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ave_config = to_ave_config(config)
    train, val, test, heldout, dataset_info = load_datasets(ave_config)
    if config.gpu_resident_data and device.type == "cuda":
        train = train.to(device)
        val = val.to(device)
    model = BidirectionalLatentExternalModel(ave_config)
    training = train_strict_stages(model, train, val, config, device)
    test_metrics = evaluate(model, test, prefix="test", device=device, batch_size=config.eval_batch_size)
    heldout_metrics = evaluate(model, heldout, prefix="heldout", device=device, batch_size=config.eval_batch_size)
    write_samples(model, test, output_path.parent / "samples_strict", device, config.sample_count)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    result = {
        "schema_version": 1,
        "stage": "AV-G",
        "strict_stage_schedule": True,
        "config": asdict(config),
        "dataset": dataset_info,
        "device": str(device),
        "metrics": {**test_metrics, **heldout_metrics},
        "cost": {
            "elapsed_sec": time.perf_counter() - started,
            "parameters": parameter_count(model),
            "peak_cuda_allocated_mb": peak_mb,
            "processed_pair_tokens": config.total_steps * config.batch_size * PAIR_TOKENS_PER_EXAMPLE,
        },
        "training": training,
        "interpretation": {
            "goal": "Strict staged training for the AV-F 10M bidirectional external/latent dataset.",
            "schedule": "text latent base -> external codec -> bidirectional translation -> latent reason -> short joint debug.",
            "separation_from_running_joint_baseline": "Uses independent AV-G outputs/checkpoints and does not touch AV-E train_10m_70m checkpoints.",
            "success_condition": "External reconstruction and both translation directions must reach sequence exact before answer accuracy is treated as architectural evidence.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": result["metrics"], "cost": result["cost"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_avg_strict_staged_10m_training/train_10m_70m_strict_result.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path(StageAVGConfig.dataset_manifest))
    parser.add_argument("--batch-size", type=int, default=StageAVGConfig.batch_size)
    parser.add_argument("--seed", type=int, default=StageAVGConfig.seed)
    parser.add_argument("--d-model", type=int, default=StageAVGConfig.d_model)
    parser.add_argument("--heads", type=int, default=StageAVGConfig.heads)
    parser.add_argument("--layers", type=int, default=StageAVGConfig.layers)
    parser.add_argument("--latent-tokens", type=int, default=StageAVGConfig.latent_tokens)
    parser.add_argument("--lr", type=float, default=StageAVGConfig.lr)
    parser.add_argument("--text-steps", type=int, default=StageAVGConfig.text_steps)
    parser.add_argument("--codec-steps", type=int, default=StageAVGConfig.codec_steps)
    parser.add_argument("--translate-steps", type=int, default=StageAVGConfig.translate_steps)
    parser.add_argument("--reason-steps", type=int, default=StageAVGConfig.reason_steps)
    parser.add_argument("--joint-steps", type=int, default=StageAVGConfig.joint_steps)
    parser.add_argument("--eval-every", type=int, default=StageAVGConfig.eval_every)
    parser.add_argument("--eval-batch-size", type=int, default=StageAVGConfig.eval_batch_size)
    parser.add_argument("--sample-count", type=int, default=StageAVGConfig.sample_count)
    parser.add_argument("--train-limit", type=int, default=StageAVGConfig.train_limit)
    parser.add_argument("--val-limit", type=int, default=StageAVGConfig.val_limit)
    parser.add_argument("--test-limit", type=int, default=StageAVGConfig.test_limit)
    parser.add_argument("--heldout-limit", type=int, default=StageAVGConfig.heldout_limit)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(StageAVGConfig.checkpoint_dir))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=StageAVGConfig.save_every)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=StageAVGConfig.device)
    parser.add_argument("--no-gpu-resident-data", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> StageAVGConfig:
    return StageAVGConfig(
        dataset_manifest=str(args.dataset_manifest),
        batch_size=args.batch_size,
        seed=args.seed,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        latent_tokens=args.latent_tokens,
        lr=args.lr,
        text_steps=args.text_steps,
        codec_steps=args.codec_steps,
        translate_steps=args.translate_steps,
        reason_steps=args.reason_steps,
        joint_steps=args.joint_steps,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        sample_count=args.sample_count,
        amp=not args.no_amp,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        heldout_limit=args.heldout_limit,
        gpu_resident_data=not args.no_gpu_resident_data,
        checkpoint_dir=str(args.checkpoint_dir),
        resume=args.resume,
        save_every=args.save_every,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    run_one(config_from_args(args), args.output)


if __name__ == "__main__":
    main()
