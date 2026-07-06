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

from omni_transformer_stage_ac_latent_reasoning import COLORS, SHAPES  # noqa: E402
from omni_transformer_stage_avj_latent_reasoning_core import (  # noqa: E402
    AVJSet,
    FAMILIES,
    PAD,
    VOCAB_SIZE,
    StageAVJConfig,
    StageAVJModel,
    answer_loss,
    build_dataset,
    masked_slot_ce,
    optional_ce,
    process_loss,
    random_batch,
    source_recon_loss,
    target_record_loss,
    token_metrics,
    trace_loss,
)


@dataclass(frozen=True)
class StageAVJBConfig(StageAVJConfig):
    trace_steps: int = 3000
    target_steps: int = 8000
    joint_steps: int = 2000
    answer_token_weight: float = 0.05
    rich_trace_weight: float = 1.0
    copy_gate_weight: float = 1.0
    verifier_weight: float = 0.5

    @property
    def total_steps(self) -> int:
        return self.codec_steps + self.trace_steps + self.target_steps + self.joint_steps


def gather_slot(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return values.gather(1, index.unsqueeze(1)).squeeze(1)


def changed_slot_mask(batch: AVJSet) -> torch.Tensor:
    return (
        (batch.source_active != batch.target_active)
        | (batch.source_color != batch.target_color)
        | (batch.source_shape != batch.target_shape)
        | (batch.source_row != batch.target_row)
        | (batch.source_col != batch.target_col)
    ).long()


def same_row_candidate_mask(batch: AVJSet) -> torch.Tensor:
    anchor_row = gather_slot(batch.source_row, batch.read_a).unsqueeze(1)
    slot_ids = torch.arange(batch.source_active.shape[1], device=batch.source_active.device).unsqueeze(0)
    return (
        (batch.family == FAMILIES.index("same_row_move")).unsqueeze(1)
        & (batch.source_active > 0)
        & (batch.source_row == anchor_row)
        & (slot_ids != batch.read_a.unsqueeze(1))
    ).float()


def count_value_target(batch: AVJSet) -> torch.Tensor:
    read_color = gather_slot(batch.source_color, batch.read_a).unsqueeze(1)
    read_shape = gather_slot(batch.source_shape, batch.read_a).unsqueeze(1)
    same_pair = (batch.source_color == read_color) & (batch.source_shape == read_shape) & (batch.source_active > 0)
    return same_pair.long().sum(dim=1).clamp_max(batch.source_active.shape[1])


def source_one_hot_logits(values: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(values, num_classes=classes).float() * 16.0 - 8.0


class StageAVJBModel(StageAVJModel):
    def __init__(self, config: StageAVJBConfig) -> None:
        super().__init__(config)
        self.rich_candidate_head = nn.Linear(config.d_model, 1)
        self.copy_gate_head = nn.Linear(config.d_model, 2)
        self.count_value_head = nn.Linear(config.d_model, config.max_objects + 1)
        self.record_verifier = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
        )

    def forward(
        self,
        batch: AVJSet,
        *,
        zero_source: bool = False,
        zero_operation: bool = False,
        disable_process: bool = False,
    ) -> dict[str, torch.Tensor]:
        slots = self.encode_record(batch, zero_source=zero_source)
        prompt = self.encode_prompt(batch.prompts, zero_operation=zero_operation)
        read_a_logits, read_a_slot = self.retrieval(self.read_a_query(prompt), slots)
        read_b_logits, read_b_slot = self.retrieval(self.read_b_query(prompt), slots)
        edit_logits, edit_slot = self.retrieval(self.edit_query(prompt), slots)
        process_state = self.process(torch.cat((prompt, read_a_slot, read_b_slot, edit_slot), dim=-1))
        if disable_process:
            process_state = torch.zeros_like(process_state)
        target_context = torch.cat(
            (
                slots,
                prompt.unsqueeze(1).expand(-1, slots.shape[1], -1),
                process_state.unsqueeze(1).expand(-1, slots.shape[1], -1),
            ),
            dim=-1,
        )
        target_state = self.target_decoder(target_context)
        gate_logits = self.copy_gate_head(target_state)
        update_prob = F.softmax(gate_logits, dim=-1)[..., 1:2]

        raw_active = self.target_active_head(target_state)
        raw_color = self.target_color_head(target_state)
        raw_shape = self.target_shape_head(target_state)
        raw_row = self.target_row_head(target_state)
        raw_col = self.target_col_head(target_state)
        target_active = update_prob * raw_active + (1.0 - update_prob) * source_one_hot_logits(batch.source_active, 2)
        target_color = update_prob * raw_color + (1.0 - update_prob) * source_one_hot_logits(batch.source_color, len(COLORS))
        target_shape = update_prob * raw_shape + (1.0 - update_prob) * source_one_hot_logits(batch.source_shape, len(SHAPES))
        target_row = update_prob * raw_row + (1.0 - update_prob) * source_one_hot_logits(batch.source_row, self.config.grid_size)
        target_col = update_prob * raw_col + (1.0 - update_prob) * source_one_hot_logits(batch.source_col, self.config.grid_size)

        answer_query = self.answer_query.unsqueeze(0).expand(slots.shape[0], -1, -1)
        answer_context = torch.cat((prompt.unsqueeze(1), process_state.unsqueeze(1), slots, answer_query), dim=1)
        answer = self.answer_decoder(answer_context)[:, -self.config.answer_len :]
        verifier_context = torch.cat((process_state, edit_slot), dim=-1)
        return {
            "slots": slots,
            "read_a": read_a_logits,
            "read_b": read_b_logits,
            "edit_slot": edit_logits,
            "condition": self.condition_head(process_state),
            "edit_action": self.action_head(process_state),
            "edit_color": self.edit_color_head(process_state),
            "edit_row": self.edit_row_head(process_state),
            "edit_col": self.edit_col_head(process_state),
            "source_active": self.source_active_head(slots),
            "source_color": self.source_color_head(slots),
            "source_shape": self.source_shape_head(slots),
            "source_row": self.source_row_head(slots),
            "source_col": self.source_col_head(slots),
            "target_active": target_active,
            "target_color": target_color,
            "target_shape": target_shape,
            "target_row": target_row,
            "target_col": target_col,
            "answer": self.answer_out(answer),
            "candidate_mask": self.rich_candidate_head(slots).squeeze(-1),
            "copy_gate": gate_logits,
            "count_value": self.count_value_head(process_state),
            "verifier": self.record_verifier(verifier_context),
        }


def rich_trace_loss(outputs: dict[str, torch.Tensor], batch: AVJSet) -> torch.Tensor:
    candidate_target = same_row_candidate_mask(batch)
    candidate_loss = F.binary_cross_entropy_with_logits(outputs["candidate_mask"], candidate_target)
    count_loss = F.cross_entropy(outputs["count_value"], count_value_target(batch))
    return candidate_loss + count_loss


def copy_gate_loss(outputs: dict[str, torch.Tensor], batch: AVJSet) -> torch.Tensor:
    return F.cross_entropy(outputs["copy_gate"].reshape(-1, 2), changed_slot_mask(batch).reshape(-1))


def verifier_loss(outputs: dict[str, torch.Tensor], batch: AVJSet) -> torch.Tensor:
    exact = slot_exact(outputs, batch, "target").long()
    return F.cross_entropy(outputs["verifier"], exact)


def slot_exact(outputs: dict[str, torch.Tensor], batch: AVJSet, prefix: str) -> torch.Tensor:
    active = outputs[f"{prefix}_active"].argmax(dim=-1)
    active_ok = active == getattr(batch, f"{prefix}_active")
    target_active = getattr(batch, f"{prefix}_active") > 0
    color_ok = outputs[f"{prefix}_color"].argmax(dim=-1) == getattr(batch, f"{prefix}_color")
    shape_ok = outputs[f"{prefix}_shape"].argmax(dim=-1) == getattr(batch, f"{prefix}_shape")
    row_ok = outputs[f"{prefix}_row"].argmax(dim=-1) == getattr(batch, f"{prefix}_row")
    col_ok = outputs[f"{prefix}_col"].argmax(dim=-1) == getattr(batch, f"{prefix}_col")
    slot_ok = active_ok & (~target_active | (color_ok & shape_ok & row_ok & col_ok))
    return slot_ok.all(dim=1)


def field_metrics(outputs: dict[str, torch.Tensor], batch: AVJSet, prefix: str) -> dict[str, float]:
    active_target = getattr(batch, f"{prefix}_active")
    active_mask = active_target > 0
    result = {
        f"{prefix}_active_accuracy": (outputs[f"{prefix}_active"].argmax(dim=-1) == active_target).float().mean().item()
    }
    for name in ("color", "shape", "row", "col"):
        pred = outputs[f"{prefix}_{name}"].argmax(dim=-1)
        target = getattr(batch, f"{prefix}_{name}")
        result[f"{prefix}_{name}_accuracy"] = (pred[active_mask] == target[active_mask]).float().mean().item() if active_mask.any() else 0.0
    return result


def metrics(outputs: dict[str, torch.Tensor], batch: AVJSet) -> dict[str, float]:
    candidate_target = same_row_candidate_mask(batch)
    count_target = count_value_target(batch)
    changed_target = changed_slot_mask(batch)
    candidate_pred = (torch.sigmoid(outputs["candidate_mask"]) > 0.5).float()
    same_row = batch.family == FAMILIES.index("same_row_move")
    candidate_exact = (
        (candidate_pred[same_row] == candidate_target[same_row]).all(dim=1).float().mean().item() if same_row.any() else 0.0
    )
    result = {
        "source_record_exact": slot_exact(outputs, batch, "source").float().mean().item(),
        "target_record_exact": slot_exact(outputs, batch, "target").float().mean().item(),
        "read_a_accuracy": (outputs["read_a"].argmax(dim=-1) == batch.read_a).float().mean().item(),
        "read_b_accuracy": (outputs["read_b"].argmax(dim=-1) == batch.read_b).float().mean().item(),
        "edit_slot_accuracy": (outputs["edit_slot"].argmax(dim=-1) == batch.edit_slot).float().mean().item(),
        "condition_accuracy": (outputs["condition"].argmax(dim=-1) == batch.condition).float().mean().item(),
        "edit_action_accuracy": (outputs["edit_action"].argmax(dim=-1) == batch.edit_action).float().mean().item(),
        "candidate_mask_exact": candidate_exact,
        "count_value_accuracy": (outputs["count_value"].argmax(dim=-1) == count_target).float().mean().item(),
        "copy_gate_accuracy": (outputs["copy_gate"].argmax(dim=-1) == changed_target).float().mean().item(),
        "verifier_accuracy": (outputs["verifier"].argmax(dim=-1) == slot_exact(outputs, batch, "target").long()).float().mean().item(),
    }
    result.update(field_metrics(outputs, batch, "target"))
    result.update(token_metrics(outputs["answer"], batch.answer))
    for family in FAMILIES:
        mask = batch.family == FAMILIES.index(family)
        if mask.any():
            sub_outputs = {key: value[mask] if value.shape[0] == mask.shape[0] else value for key, value in outputs.items()}
            sub_batch = batch.subset(mask.nonzero(as_tuple=False).flatten().tolist())
            result[f"{family}_target_record_exact"] = slot_exact(sub_outputs, sub_batch, "target").float().mean().item()
            result[f"{family}_answer_sequence_exact"] = token_metrics(outputs["answer"][mask], batch.answer[mask])[
                "answer_sequence_exact"
            ]
    return result


def stage_name(config: StageAVJBConfig, step: int) -> str:
    if step <= config.codec_steps:
        return "codec"
    if step <= config.codec_steps + config.trace_steps:
        return "trace_sft"
    if step <= config.codec_steps + config.trace_steps + config.target_steps:
        return "target_verifier"
    return "joint"


def total_loss(outputs: dict[str, torch.Tensor], batch: AVJSet, config: StageAVJBConfig, stage: str) -> torch.Tensor:
    loss_source = source_recon_loss(outputs, batch)
    loss_trace = trace_loss(outputs, batch)
    loss_process = process_loss(outputs, batch)
    loss_rich = rich_trace_loss(outputs, batch)
    loss_gate = copy_gate_loss(outputs, batch)
    loss_target = target_record_loss(outputs, batch)
    loss_verifier = verifier_loss(outputs, batch)
    loss_answer = answer_loss(outputs, batch)
    if stage == "codec":
        return config.source_recon_weight * loss_source
    if stage == "trace_sft":
        return (
            0.4 * config.source_recon_weight * loss_source
            + config.trace_loss_weight * loss_trace
            + config.process_loss_weight * loss_process
            + config.rich_trace_weight * loss_rich
        )
    if stage == "target_verifier":
        return (
            0.2 * config.source_recon_weight * loss_source
            + config.trace_loss_weight * loss_trace
            + config.process_loss_weight * loss_process
            + config.rich_trace_weight * loss_rich
            + config.copy_gate_weight * loss_gate
            + config.target_record_weight * loss_target
            + config.verifier_weight * loss_verifier
        )
    return (
        0.1 * config.source_recon_weight * loss_source
        + config.trace_loss_weight * loss_trace
        + config.process_loss_weight * loss_process
        + config.rich_trace_weight * loss_rich
        + config.copy_gate_weight * loss_gate
        + config.target_record_weight * loss_target
        + config.verifier_weight * loss_verifier
        + config.answer_token_weight * loss_answer
    )


@torch.no_grad()
def evaluate(model: StageAVJBModel, data: AVJSet, config: StageAVJBConfig, device: torch.device) -> dict[str, float]:
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


def save_checkpoint(path: Path, model: StageAVJBModel, optimizer: torch.optim.Optimizer, step: int, config: StageAVJBConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config)}, path)


def load_checkpoint(path: Path, model: StageAVJBModel, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def run(config: StageAVJBConfig, output: Path, checkpoint_dir: Path | None, resume: bool) -> dict[str, object]:
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
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(config.seed + 1515)
    history: list[dict[str, object]] = []
    for step in range(start_step + 1, config.total_steps + 1):
        model.train()
        batch = random_batch(train, rng, config.batch_size, device)
        stage = stage_name(config, step)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(batch)
            loss = total_loss(outputs, batch, config, stage)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % config.eval_every == 0 or step == config.total_steps:
            row: dict[str, object] = {"step": step, "stage": stage, "loss": float(loss.detach().cpu())}
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
        "stage": "AV-J-B",
        "config": asdict(config),
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
    parser = argparse.ArgumentParser(description="Stage AV-J-B trace/verifier latent reasoning.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_avjb_trace_verifier/cpu_smoke_result.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
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
    parser.add_argument("--target-steps", type=int, default=1200)
    parser.add_argument("--joint-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--answer-token-weight", type=float, default=0.05)
    parser.add_argument("--cpu-data", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StageAVJBConfig(
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
        target_steps=args.target_steps,
        joint_steps=args.joint_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        answer_token_weight=args.answer_token_weight,
        amp=not args.no_amp,
        gpu_resident_data=not args.cpu_data,
    )
    result = run(config, args.output, args.checkpoint_dir, args.resume)
    print(json.dumps({"output": str(args.output), "final": result["final"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
