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

from omni_transformer_stage_av_from_scratch_micro_omni import (  # noqa: E402
    ANSWER_CLASSES,
    COLORS,
    MEMORY_LEN,
    MEM_UNKNOWN,
    STATE_PHASE_BASE,
    STATE_TELEMETRY_BASE,
    TASKS,
    TASK_AS,
    TASK_AT,
    TEXT_LEN,
    TEXT_RULE_BASE,
    TEXT_TARGET_BASE,
    TEXT_TARGET_COLOR_BASE,
    TOKEN_VOCAB,
    TOOL_LEN,
    ZONES,
    StageAVConfig,
    build_split,
    render_case_png,
)
from visual_multimodal_stage_ab import write_png  # noqa: E402


STAGES = ("text_base", "image_translate", "tool_memory_translate", "latent_reason", "joint")


@dataclass(frozen=True)
class StageAVBConfig:
    train_size: int = 4096
    val_size: int = 768
    test_size: int = 768
    batch_size: int = 64
    seed: int = 20260701
    d_model: int = 768
    heads: int = 12
    layers: int = 10
    latent_tokens: int = 8
    lr: float = 4e-4
    text_steps: int = 80
    image_steps: int = 120
    tool_memory_steps: int = 120
    reason_steps: int = 300
    joint_steps: int = 120
    eval_every: int = 100
    sample_count: int = 8
    amp: bool = True
    compile_model: bool = False
    checkpoint_dir: str = ""
    resume: bool = False


@dataclass(frozen=True)
class GPUData:
    image: torch.Tensor
    image_color: torch.Tensor
    tokens: torch.Tensor
    answer: torch.Tensor
    task: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.answer.shape[0])


def to_gpu_data(data, device: torch.device) -> GPUData:
    return GPUData(
        image=data.image.to(device=device, non_blocking=True),
        image_color=data.image_color.to(device=device, non_blocking=True),
        tokens=data.tokens.to(device=device, non_blocking=True),
        answer=data.answer.to(device=device, non_blocking=True),
        task=data.task.to(device=device, non_blocking=True),
    )


def gpu_batch(data: GPUData, batch_size: int) -> GPUData:
    index = torch.randint(0, data.size, (batch_size,), device=data.answer.device)
    return GPUData(
        image=data.image[index],
        image_color=data.image_color[index],
        tokens=data.tokens[index],
        answer=data.answer[index],
        task=data.task[index],
    )


class StagedLatentMicroOmni(nn.Module):
    def __init__(self, config: StageAVBConfig, token_len: int) -> None:
        super().__init__()
        self.config = config
        self.image = nn.Linear(3, config.d_model)
        self.token = nn.Embedding(TOKEN_VOCAB, config.d_model, padding_idx=0)
        self.type_embed = nn.Embedding(4, config.d_model)
        seq_len = ZONES + token_len + config.latent_tokens
        self.position = nn.Parameter(torch.randn(seq_len, config.d_model) * 0.02)
        self.latent = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)

        self.answer_head = nn.Linear(config.d_model * 2, ANSWER_CLASSES)
        self.task_head = nn.Linear(config.d_model, len(TASKS))
        self.image_color_head = nn.Linear(config.d_model, COLORS)
        self.tool_issue_head = nn.Linear(config.d_model, 4)
        self.memory_zone_head = nn.Linear(config.d_model, ZONES)
        self.text_zone_head = nn.Linear(config.d_model, ZONES)
        self.text_rule_head = nn.Linear(config.d_model, 4)
        self.text_color_head = nn.Linear(config.d_model, COLORS)
        self.telemetry_head = nn.Linear(config.d_model, 4)
        self.phase_head = nn.Linear(config.d_model, 4)

    def forward(
        self,
        batch: GPUData,
        *,
        no_image: bool = False,
        no_text: bool = False,
        no_tool_history: bool = False,
        no_memory: bool = False,
        shuffled_image: bool = False,
    ) -> dict[str, torch.Tensor]:
        image = batch.image
        if shuffled_image:
            image = image.flip(0)
        if no_image:
            image = torch.zeros_like(image)
        tokens = batch.tokens.clone()
        if no_text:
            tokens[:, 1 : 1 + TEXT_LEN] = 0
        if no_tool_history:
            tokens[:, 1 + TEXT_LEN : 1 + TEXT_LEN + TOOL_LEN] = 0
        if no_memory:
            start = 1 + TEXT_LEN + TOOL_LEN
            tokens[:, start : start + MEMORY_LEN] = MEM_UNKNOWN

        image_tokens = self.image(image) + self.type_embed.weight[0].view(1, 1, -1)
        text_tokens = self.token(tokens[:, : 1 + TEXT_LEN]) + self.type_embed.weight[1].view(1, 1, -1)
        evidence_tokens = self.token(tokens[:, 1 + TEXT_LEN :]) + self.type_embed.weight[2].view(1, 1, -1)
        latent = self.latent.unsqueeze(0).expand(tokens.shape[0], -1, -1) + self.type_embed.weight[3].view(1, 1, -1)
        x = torch.cat((image_tokens, text_tokens, evidence_tokens, latent), dim=1)
        x = x + self.position[: x.shape[1]].view(1, -1, x.shape[-1])
        encoded = self.encoder(x)

        image_out = self.norm(encoded[:, :ZONES])
        text_out = self.norm(encoded[:, ZONES : ZONES + 1 + TEXT_LEN])
        evidence_out = self.norm(encoded[:, ZONES + 1 + TEXT_LEN : -latent.shape[1]])
        latent_out = self.norm(encoded[:, -latent.shape[1] :])
        pooled_latent = latent_out.mean(dim=1)
        pooled_evidence = evidence_out.mean(dim=1)
        pooled_text = text_out.mean(dim=1)
        pooled_all = torch.cat((pooled_latent, pooled_evidence), dim=-1)
        return {
            "answer": self.answer_head(pooled_all),
            "task": self.task_head(pooled_latent),
            "image_color": self.image_color_head(image_out),
            "tool_issue": self.tool_issue_head(pooled_evidence),
            "memory_zone": self.memory_zone_head(pooled_evidence),
            "text_zone": self.text_zone_head(pooled_text),
            "text_rule": self.text_rule_head(pooled_text),
            "text_color": self.text_color_head(pooled_text),
            "telemetry": self.telemetry_head(pooled_evidence),
            "phase": self.phase_head(pooled_evidence),
        }


def balanced_answer_loss(logits: torch.Tensor, answer: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
    per_example = F.cross_entropy(logits, answer, reduction="none")
    losses = []
    for task_id in range(len(TASKS)):
        mask = task == task_id
        if bool(mask.any()):
            losses.append(per_example[mask].mean())
    return torch.stack(losses).mean()


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def targets_from_tokens(batch: GPUData) -> dict[str, torch.Tensor]:
    tokens = batch.tokens
    return {
        "text_zone": tokens[:, 1] - TEXT_TARGET_BASE,
        "text_rule": tokens[:, 2] - TEXT_RULE_BASE,
        "text_color": tokens[:, 3] - TEXT_TARGET_COLOR_BASE,
        "telemetry": tokens[:, 1 + TEXT_LEN + TOOL_LEN + MEMORY_LEN] - STATE_TELEMETRY_BASE,
        "phase": tokens[:, 2 + TEXT_LEN + TOOL_LEN + MEMORY_LEN] - STATE_PHASE_BASE,
        "tool_issue": (batch.answer - 4).clamp(0, 3),
        "memory_zone": (batch.answer - 8).clamp(0, 3),
    }


def stage_loss(model: StagedLatentMicroOmni, batch: GPUData, stage: str) -> torch.Tensor:
    out = model(batch)
    target = targets_from_tokens(batch)
    as_mask = batch.task == TASK_AS
    at_mask = batch.task == TASK_AT
    if stage == "text_base":
        return (
            F.cross_entropy(out["task"], batch.task)
            + F.cross_entropy(out["text_zone"], target["text_zone"])
            + F.cross_entropy(out["text_rule"], target["text_rule"])
            + F.cross_entropy(out["text_color"], target["text_color"])
            + 0.5 * F.cross_entropy(out["telemetry"], target["telemetry"])
            + 0.5 * F.cross_entropy(out["phase"], target["phase"])
        )
    if stage == "image_translate":
        return F.cross_entropy(out["image_color"].reshape(-1, COLORS), batch.image_color.reshape(-1))
    if stage == "tool_memory_translate":
        return masked_ce(out["tool_issue"], target["tool_issue"], as_mask) + masked_ce(out["memory_zone"], target["memory_zone"], at_mask)
    if stage == "latent_reason":
        return balanced_answer_loss(out["answer"], batch.answer, batch.task)
    if stage == "joint":
        return (
            balanced_answer_loss(out["answer"], batch.answer, batch.task)
            + 0.10 * F.cross_entropy(out["task"], batch.task)
            + 0.20 * F.cross_entropy(out["image_color"].reshape(-1, COLORS), batch.image_color.reshape(-1))
            + 0.20 * masked_ce(out["tool_issue"], target["tool_issue"], as_mask)
            + 0.20 * masked_ce(out["memory_zone"], target["memory_zone"], at_mask)
        )
    raise ValueError(f"unknown stage {stage}")


@torch.no_grad()
def evaluate(model: StagedLatentMicroOmni, data: GPUData, *, prefix: str, eval_limit: int = 0) -> dict[str, float]:
    model.eval()
    if eval_limit and data.size > eval_limit:
        index = torch.arange(eval_limit, device=data.answer.device)
        batch = GPUData(data.image[index], data.image_color[index], data.tokens[index], data.answer[index], data.task[index])
    else:
        batch = data
    out = model(batch)
    pred = out["answer"].argmax(dim=-1)
    metrics = {f"{prefix}_answer_accuracy": float((pred == batch.answer).float().mean().item())}
    for task_id, name in enumerate(TASKS):
        mask = batch.task == task_id
        if bool(mask.any()):
            metrics[f"{prefix}_{name}_accuracy"] = float((pred[mask] == batch.answer[mask]).float().mean().item())
    metrics[f"{prefix}_task_probe_accuracy"] = float((out["task"].argmax(dim=-1) == batch.task).float().mean().item())
    metrics[f"{prefix}_image_color_accuracy"] = float((out["image_color"].argmax(dim=-1) == batch.image_color).float().mean().item())
    for ablation in ("no_image", "no_text", "no_tool_history", "no_memory", "shuffled_image"):
        ablated = model(batch, **{ablation: True})
        metrics[f"{prefix}_{ablation}_answer_accuracy"] = float((ablated["answer"].argmax(dim=-1) == batch.answer).float().mean().item())
    return metrics


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def checkpoint_path(config: StageAVBConfig) -> Path | None:
    if not config.checkpoint_dir:
        return None
    return Path(config.checkpoint_dir) / "latest.pt"


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, config: StageAVBConfig, stage_index: int, step_in_stage: int) -> None:
    path = checkpoint_path(config)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "stage_index": stage_index,
            "step_in_stage": step_in_stage,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, config: StageAVBConfig) -> tuple[int, int]:
    path = checkpoint_path(config)
    if path is None or not config.resume or not path.exists():
        return 0, 0
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    return int(state.get("stage_index", 0)), int(state.get("step_in_stage", 0))


def train(model: StagedLatentMicroOmni, train_data: GPUData, val_data: GPUData, config: StageAVBConfig, device: torch.device) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    start_stage, start_step = load_checkpoint(model, optimizer, scaler, config)
    stage_steps = {
        "text_base": config.text_steps,
        "image_translate": config.image_steps,
        "tool_memory_translate": config.tool_memory_steps,
        "latent_reason": config.reason_steps,
        "joint": config.joint_steps,
    }
    history = []
    global_step = 0
    for stage_index, stage in enumerate(STAGES):
        steps = stage_steps[stage]
        if stage_index < start_stage:
            global_step += steps
            continue
        first_step = start_step + 1 if stage_index == start_stage else 1
        for step in range(first_step, steps + 1):
            model.train()
            batch = gpu_batch(train_data, config.batch_size)
            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                loss = stage_loss(model, batch, stage)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if step == 1 or step % config.eval_every == 0 or step == steps:
                metrics = evaluate(model, val_data, prefix="val", eval_limit=min(512, val_data.size))
                metrics.update({"stage": stage, "stage_step": float(step), "global_step": float(global_step), "loss": float(loss.detach().item())})
                history.append(metrics)
                print(json.dumps(metrics), flush=True)
                save_checkpoint(model, optimizer, scaler, config, stage_index, step)
    return {"history": history}


@torch.no_grad()
def write_samples(model: StagedLatentMicroOmni, source_data, gpu_data: GPUData, path: Path, limit: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    for index, example in enumerate(source_data.examples[:limit]):
        png = path / f"{example.case_id}.png"
        write_png(png, render_case_png(example))
        batch = GPUData(
            gpu_data.image[index : index + 1],
            gpu_data.image_color[index : index + 1],
            gpu_data.tokens[index : index + 1],
            gpu_data.answer[index : index + 1],
            gpu_data.task[index : index + 1],
        )
        pred = int(model(batch)["answer"].argmax(dim=-1).item())
        rows.append({"case_id": example.case_id, "task": TASKS[example.task], "answer": example.answer, "prediction": pred, "map_png": str(png)})
    (path / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(config: StageAVBConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_train = build_split("train", config.train_size, StageAVConfig(seed=config.seed))
    source_val = build_split("val", config.val_size, StageAVConfig(seed=config.seed))
    source_test = build_split("test", config.test_size, StageAVConfig(seed=config.seed))
    train_data = to_gpu_data(source_train, device)
    val_data = to_gpu_data(source_val, device)
    test_data = to_gpu_data(source_test, device)
    model = StagedLatentMicroOmni(config, train_data.tokens.shape[1]).to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    training = train(model, train_data, val_data, config, device)
    metrics = evaluate(model, test_data, prefix="test")
    write_samples(model, source_test, test_data, output_path.parent / "samples", config.sample_count)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    result = {
        "schema_version": 1,
        "stage": "AV-B",
        "config": asdict(config),
        "device": str(device),
        "metrics": metrics,
        "cost": {
            "elapsed_sec": time.perf_counter() - started,
            "parameters": parameter_count(model),
            "peak_cuda_allocated_mb": peak_mb,
        },
        "training": training,
        "interpretation": {
            "goal": "Stage latent translation before latent reasoning, then run short joint tuning.",
            "success_condition": "Each task should pass after staged training; early translator stages are intentionally not overtrained.",
            "boundary": "This is still synthetic from-scratch architecture validation, not a general model.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": metrics, "cost": result["cost"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_avb_staged_latent_training/result.json"))
    parser.add_argument("--train-size", type=int, default=StageAVBConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageAVBConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageAVBConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageAVBConfig.batch_size)
    parser.add_argument("--seed", type=int, default=StageAVBConfig.seed)
    parser.add_argument("--d-model", type=int, default=StageAVBConfig.d_model)
    parser.add_argument("--heads", type=int, default=StageAVBConfig.heads)
    parser.add_argument("--layers", type=int, default=StageAVBConfig.layers)
    parser.add_argument("--latent-tokens", type=int, default=StageAVBConfig.latent_tokens)
    parser.add_argument("--text-steps", type=int, default=StageAVBConfig.text_steps)
    parser.add_argument("--image-steps", type=int, default=StageAVBConfig.image_steps)
    parser.add_argument("--tool-memory-steps", type=int, default=StageAVBConfig.tool_memory_steps)
    parser.add_argument("--reason-steps", type=int, default=StageAVBConfig.reason_steps)
    parser.add_argument("--joint-steps", type=int, default=StageAVBConfig.joint_steps)
    parser.add_argument("--eval-every", type=int, default=StageAVBConfig.eval_every)
    parser.add_argument("--sample-count", type=int, default=StageAVBConfig.sample_count)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> StageAVBConfig:
    return StageAVBConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        latent_tokens=args.latent_tokens,
        text_steps=args.text_steps,
        image_steps=args.image_steps,
        tool_memory_steps=args.tool_memory_steps,
        reason_steps=args.reason_steps,
        joint_steps=args.joint_steps,
        eval_every=args.eval_every,
        sample_count=args.sample_count,
        amp=not args.no_amp,
        compile_model=args.compile_model,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    args = parse_args()
    run_one(config_from_args(args), args.output)
