from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omni_transformer_stage_ac_latent_reasoning import COLORS, SHAPES  # noqa: E402
from omni_transformer_stage_avj_latent_reasoning_core import (  # noqa: E402
    AVJSet,
    FAMILIES,
    build_dataset,
    random_batch,
    source_recon_loss,
    target_record_loss,
    trace_loss,
    process_loss,
)
from omni_transformer_stage_avjb_trace_verifier import (  # noqa: E402
    StageAVJBConfig,
    StageAVJBModel,
    copy_gate_loss,
    count_value_target,
    metrics,
    rich_trace_loss,
    same_row_candidate_mask,
    verifier_loss,
)


@dataclass(frozen=True)
class StageAVJCConfig(StageAVJBConfig):
    target_steps: int = 6000
    rl_steps: int = 6000
    joint_steps: int = 0
    answer_token_weight: float = 0.0
    rl_reward_weight: float = 1.0
    rl_entropy_weight: float = 0.01
    rl_supervised_anchor_weight: float = 0.15
    rl_record_exact_weight: float = 3.0
    rl_field_weight: float = 1.0
    rl_changed_slot_weight: float = 1.0
    rl_candidate_weight: float = 0.5
    rl_count_weight: float = 0.5

    @property
    def total_steps(self) -> int:
        return self.codec_steps + self.trace_steps + self.target_steps + self.rl_steps + self.joint_steps


def stage_name(config: StageAVJCConfig, step: int) -> str:
    if step <= config.codec_steps:
        return "codec"
    if step <= config.codec_steps + config.trace_steps:
        return "trace_sft"
    if step <= config.codec_steps + config.trace_steps + config.target_steps:
        return "target_sft"
    if step <= config.codec_steps + config.trace_steps + config.target_steps + config.rl_steps:
        return "rl_verifier"
    return "joint_no_answer"


def _record_reward(
    active: torch.Tensor,
    color: torch.Tensor,
    shape: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    batch: AVJSet,
    config: StageAVJCConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active_ok = active == batch.target_active
    target_active = batch.target_active > 0
    color_ok = (color == batch.target_color) | ~target_active
    shape_ok = (shape == batch.target_shape) | ~target_active
    row_ok = (row == batch.target_row) | ~target_active
    col_ok = (col == batch.target_col) | ~target_active
    slot_ok = active_ok & color_ok & shape_ok & row_ok & col_ok
    record_exact = slot_ok.all(dim=1).float()
    field_score = (
        active_ok.float().mean(dim=1)
        + color_ok.float().mean(dim=1)
        + shape_ok.float().mean(dim=1)
        + row_ok.float().mean(dim=1)
        + col_ok.float().mean(dim=1)
    ) / 5.0
    changed = (
        (batch.source_active != batch.target_active)
        | (batch.source_color != batch.target_color)
        | (batch.source_shape != batch.target_shape)
        | (batch.source_row != batch.target_row)
        | (batch.source_col != batch.target_col)
    )
    changed_present = changed.any(dim=1)
    changed_ok = ((slot_ok & changed) | ~changed).all(dim=1).float()
    changed_score = torch.where(changed_present, changed_ok, torch.ones_like(changed_ok))
    reward = (
        config.rl_record_exact_weight * record_exact
        + config.rl_field_weight * field_score
        + config.rl_changed_slot_weight * changed_score
    )
    parts = {
        "record_exact": record_exact,
        "field_score": field_score,
        "changed_score": changed_score,
    }
    return reward, parts


def _categorical_sample(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=logits.float())
    sample = dist.sample()
    log_prob = dist.log_prob(sample)
    entropy = dist.entropy()
    return sample, log_prob, entropy


def _bernoulli_sample(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Bernoulli(logits=logits.float())
    sample = dist.sample()
    log_prob = dist.log_prob(sample)
    entropy = dist.entropy()
    return sample.long(), log_prob, entropy


def rl_verifier_loss(
    outputs: dict[str, torch.Tensor],
    batch: AVJSet,
    config: StageAVJCConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    active, active_lp, active_ent = _categorical_sample(outputs["target_active"])
    color, color_lp, color_ent = _categorical_sample(outputs["target_color"])
    shape, shape_lp, shape_ent = _categorical_sample(outputs["target_shape"])
    row, row_lp, row_ent = _categorical_sample(outputs["target_row"])
    col, col_lp, col_ent = _categorical_sample(outputs["target_col"])

    record_log_prob = active_lp + color_lp + shape_lp + row_lp + col_lp
    record_entropy = active_ent + color_ent + shape_ent + row_ent + col_ent
    sample_reward, sample_parts = _record_reward(active, color, shape, row, col, batch, config)

    greedy_reward, greedy_parts = _record_reward(
        outputs["target_active"].argmax(dim=-1),
        outputs["target_color"].argmax(dim=-1),
        outputs["target_shape"].argmax(dim=-1),
        outputs["target_row"].argmax(dim=-1),
        outputs["target_col"].argmax(dim=-1),
        batch,
        config,
    )
    log_prob = record_log_prob.sum(dim=1)
    entropy = record_entropy.sum(dim=1)
    reward = sample_reward

    candidate_target = same_row_candidate_mask(batch)
    same_row = batch.family == FAMILIES.index("same_row_move")
    if same_row.any():
        candidate_sample, candidate_lp, candidate_ent = _bernoulli_sample(outputs["candidate_mask"])
        candidate_exact = (candidate_sample[same_row] == candidate_target[same_row].long()).all(dim=1).float()
        candidate_reward = torch.zeros_like(reward)
        candidate_reward[same_row] = candidate_exact
        reward = reward + config.rl_candidate_weight * candidate_reward
        greedy_candidate = (
            ((outputs["candidate_mask"].sigmoid() > 0.5).long()[same_row] == candidate_target[same_row].long())
            .all(dim=1)
            .float()
        )
        greedy_candidate_reward = torch.zeros_like(greedy_reward)
        greedy_candidate_reward[same_row] = greedy_candidate
        greedy_reward = greedy_reward + config.rl_candidate_weight * greedy_candidate_reward
        log_prob = log_prob + candidate_lp.sum(dim=1)
        entropy = entropy + candidate_ent.sum(dim=1)
    else:
        candidate_reward = torch.zeros_like(reward)

    count_sample, count_lp, count_ent = _categorical_sample(outputs["count_value"])
    count_reward = (count_sample == count_value_target(batch)).float()
    reward = reward + config.rl_count_weight * count_reward
    greedy_count_reward = (outputs["count_value"].argmax(dim=-1) == count_value_target(batch)).float()
    greedy_reward = greedy_reward + config.rl_count_weight * greedy_count_reward
    log_prob = log_prob + count_lp
    entropy = entropy + count_ent

    advantage = (reward - greedy_reward).detach()
    policy_loss = -(advantage * log_prob).mean()
    entropy_bonus = entropy.mean()
    loss = config.rl_reward_weight * policy_loss - config.rl_entropy_weight * entropy_bonus
    stats = {
        "rl_sample_reward": reward.mean().detach().item(),
        "rl_greedy_reward": greedy_reward.mean().detach().item(),
        "rl_advantage": advantage.mean().detach().item(),
        "rl_sample_record_exact": sample_parts["record_exact"].mean().detach().item(),
        "rl_greedy_record_exact": greedy_parts["record_exact"].mean().detach().item(),
        "rl_sample_field_score": sample_parts["field_score"].mean().detach().item(),
        "rl_candidate_reward": candidate_reward.mean().detach().item(),
        "rl_count_reward": count_reward.mean().detach().item(),
        "rl_entropy": entropy_bonus.detach().item(),
    }
    return loss, stats


def total_loss(
    outputs: dict[str, torch.Tensor],
    batch: AVJSet,
    config: StageAVJCConfig,
    stage: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_source = source_recon_loss(outputs, batch)
    loss_trace = trace_loss(outputs, batch)
    loss_process = process_loss(outputs, batch)
    loss_rich = rich_trace_loss(outputs, batch)
    loss_gate = copy_gate_loss(outputs, batch)
    loss_target = target_record_loss(outputs, batch)
    loss_verifier = verifier_loss(outputs, batch)
    if stage == "codec":
        return config.source_recon_weight * loss_source, {}
    if stage == "trace_sft":
        return (
            0.4 * config.source_recon_weight * loss_source
            + config.trace_loss_weight * loss_trace
            + config.process_loss_weight * loss_process
            + config.rich_trace_weight * loss_rich,
            {},
        )
    if stage == "target_sft":
        return (
            0.2 * config.source_recon_weight * loss_source
            + config.trace_loss_weight * loss_trace
            + config.process_loss_weight * loss_process
            + config.rich_trace_weight * loss_rich
            + config.copy_gate_weight * loss_gate
            + config.target_record_weight * loss_target
            + config.verifier_weight * loss_verifier,
            {},
        )
    rl_loss, stats = rl_verifier_loss(outputs, batch, config)
    anchor = (
        0.1 * config.source_recon_weight * loss_source
        + config.trace_loss_weight * loss_trace
        + config.process_loss_weight * loss_process
        + config.rich_trace_weight * loss_rich
        + config.copy_gate_weight * loss_gate
        + config.verifier_weight * loss_verifier
        + config.rl_supervised_anchor_weight * config.target_record_weight * loss_target
    )
    return anchor + rl_loss, stats


@torch.no_grad()
def evaluate(model: StageAVJBModel, data: AVJSet, config: StageAVJCConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    result: dict[str, list[float]] = {}
    for start in range(0, data.prompts.shape[0], config.eval_batch_size):
        indices = list(range(start, min(start + config.eval_batch_size, data.prompts.shape[0])))
        batch = data.subset(indices)
        batch = batch if batch.prompts.device == device else batch.to(device)
        variants = {
            "full": model(batch),
            "no_source": model(batch, zero_source=True),
            "no_operation": model(batch, zero_operation=True),
            "no_process": model(batch, disable_process=True),
        }
        for variant, outputs in variants.items():
            for key, value in metrics(outputs, batch).items():
                result.setdefault(f"{variant}_{key}", []).append(value)
    return {key: sum(values) / len(values) for key, values in result.items()}


def save_checkpoint(path: Path, model: StageAVJBModel, optimizer: torch.optim.Optimizer, step: int, config: StageAVJCConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config)}, path)


def load_checkpoint(path: Path, model: StageAVJBModel, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def load_model_only(path: Path, model: StageAVJBModel, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)


def run(
    config: StageAVJCConfig,
    output: Path,
    checkpoint_dir: Path | None,
    resume: bool,
    init_from: Path | None,
) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    train = build_dataset("train", config.train_size, config)
    val = build_dataset("val", config.val_size, config)
    test = build_dataset("test", config.test_size, config)
    heldout = build_dataset("heldout", config.heldout_size, config)
    if config.gpu_resident_data and device.type == "cuda":
        train, val, test, heldout = (train.to(device), val.to(device), test.to(device), heldout.to(device))
    model = StageAVJBModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    start_step = 0
    if resume and checkpoint_dir is not None and (checkpoint_dir / "latest.pt").exists():
        start_step = load_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, device)
    elif init_from is not None:
        load_model_only(init_from, model, device)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(config.seed + 1717)
    history: list[dict[str, object]] = []
    for step in range(start_step + 1, config.total_steps + 1):
        model.train()
        batch = random_batch(train, rng, config.batch_size, device)
        stage = stage_name(config, step)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch)
            loss, rl_stats = total_loss(outputs, batch, config, stage)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % config.eval_every == 0 or step == config.total_steps:
            row: dict[str, object] = {"step": step, "stage": stage, "loss": float(loss.detach().cpu())}
            row.update(rl_stats)
            row.update({f"val_{key}": value for key, value in evaluate(model, val, config, device).items()})
            history.append(row)
        if checkpoint_dir is not None and (step % config.save_every == 0 or step == config.total_steps):
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, step, config)
    final = {
        "val": evaluate(model, val, config, device),
        "test": evaluate(model, test, config, device),
        "heldout": evaluate(model, heldout, config, device),
    }
    result = {
        "schema_version": 1,
        "stage": "AV-J-C",
        "config": asdict(config),
        "init_from": str(init_from) if init_from is not None else None,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "device": str(device),
        "history": history,
        "final": final,
        "examples": train.examples[: min(8, len(train.examples))],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage AV-J-C RL verifier reward latent reasoning.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_avjc_rl_verifier_reward/cpu_smoke_result.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--heldout-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--codec-steps", type=int, default=200)
    parser.add_argument("--trace-steps", type=int, default=600)
    parser.add_argument("--target-sft-steps", type=int, default=1200)
    parser.add_argument("--rl-steps", type=int, default=1200)
    parser.add_argument("--joint-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rl-entropy-weight", type=float, default=0.01)
    parser.add_argument("--rl-supervised-anchor-weight", type=float, default=0.15)
    parser.add_argument("--cpu-data", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StageAVJCConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        heldout_size=args.heldout_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        lr=args.lr,
        codec_steps=args.codec_steps,
        trace_steps=args.trace_steps,
        target_steps=args.target_sft_steps,
        rl_steps=args.rl_steps,
        joint_steps=args.joint_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        rl_entropy_weight=args.rl_entropy_weight,
        rl_supervised_anchor_weight=args.rl_supervised_anchor_weight,
        amp=not args.no_amp,
        gpu_resident_data=not args.cpu_data,
    )
    result = run(config, args.output, args.checkpoint_dir, args.resume, args.init_from)
    print(json.dumps({"output": str(args.output), "final": result["final"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
