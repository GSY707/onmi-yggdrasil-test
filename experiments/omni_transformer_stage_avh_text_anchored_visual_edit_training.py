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

from omni_transformer_stage_avh_text_anchored_visual_edit_dataset import (  # noqa: E402
    CHAR_VOCAB,
    EDIT_TEXT_LEN,
    IMAGE_SIZE,
    PATCH_SIZE,
    PATCHES_PER_IMAGE,
    RECORD_LEN,
    RECORD_VOCAB,
    SOURCE_TEXT_LEN,
    TARGET_TEXT_LEN,
    TOKENS_PER_EXAMPLE,
    render_records,
)
from visual_multimodal_stage_ab import write_png  # noqa: E402


@dataclass(frozen=True)
class AVHSet:
    source_text: torch.Tensor
    edit_text: torch.Tensor
    target_text: torch.Tensor
    source_record: torch.Tensor
    target_record: torch.Tensor
    edit_type: torch.Tensor
    edit_object: torch.Tensor
    edit_value: torch.Tensor
    case_ids: list[str]

    def __len__(self) -> int:
        return int(self.source_text.shape[0])

    def slice(self, start: int, end: int) -> "AVHSet":
        return AVHSet(
            self.source_text[start:end],
            self.edit_text[start:end],
            self.target_text[start:end],
            self.source_record[start:end],
            self.target_record[start:end],
            self.edit_type[start:end],
            self.edit_object[start:end],
            self.edit_value[start:end],
            self.case_ids[start:end],
        )

    def tensor_subset(self, indices: torch.Tensor, *, include_case_ids: bool = True) -> "AVHSet":
        case_ids: list[str] = []
        if include_case_ids:
            cpu_indices = indices.detach().cpu().tolist()
            case_ids = [self.case_ids[index] for index in cpu_indices]
        return AVHSet(
            self.source_text.index_select(0, indices),
            self.edit_text.index_select(0, indices),
            self.target_text.index_select(0, indices),
            self.source_record.index_select(0, indices),
            self.target_record.index_select(0, indices),
            self.edit_type.index_select(0, indices),
            self.edit_object.index_select(0, indices),
            self.edit_value.index_select(0, indices),
            case_ids,
        )

    def to(self, device: torch.device) -> "AVHSet":
        return AVHSet(
            self.source_text.to(device=device, dtype=torch.long),
            self.edit_text.to(device=device, dtype=torch.long),
            self.target_text.to(device=device, dtype=torch.long),
            self.source_record.to(device=device, dtype=torch.long),
            self.target_record.to(device=device, dtype=torch.long),
            self.edit_type.to(device=device, dtype=torch.long),
            self.edit_object.to(device=device, dtype=torch.long),
            self.edit_value.to(device=device, dtype=torch.long),
            self.case_ids,
        )


@dataclass(frozen=True)
class StageAVHTrainConfig:
    dataset_manifest: str = "artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_dataset/dataset_10m/manifest.json"
    batch_size: int = 128
    micro_batch_size: int = 64
    seed: int = 20260705
    d_model: int = 704
    heads: int = 11
    layers: int = 10
    latent_tokens: int = 12
    lr: float = 5e-4
    optimizer: str = "adafactor"
    text_steps: int = 800
    image_ground_steps: int = 1000
    edit_reason_steps: int = 1600
    image_output_steps: int = 1800
    joint_steps: int = 800
    eval_every: int = 250
    eval_batch_size: int = 64
    sample_count: int = 6
    amp: bool = True
    train_limit: int = 0
    val_limit: int = 0
    test_limit: int = 0
    heldout_limit: int = 0
    gpu_resident_data: bool = True
    checkpoint_dir: str = "artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/train_10m_68m_checkpoints"
    resume: bool = False
    save_every: int = 500
    device: str = "auto"

    @property
    def total_steps(self) -> int:
        return self.text_steps + self.image_ground_steps + self.edit_reason_steps + self.image_output_steps + self.joint_steps


@dataclass(frozen=True)
class StageSpec:
    name: str
    steps: int
    purpose: str


def build_stage_plan(config: StageAVHTrainConfig) -> list[StageSpec]:
    return [
        StageSpec("text_latent", config.text_steps, "Text description plus edit instruction predicts target text and target object record."),
        StageSpec("image_ground", config.image_ground_steps, "Source image predicts source object record."),
        StageSpec("edit_reason", config.edit_reason_steps, "Source record/text plus edit instruction predicts target record and target text."),
        StageSpec("image_output", config.image_output_steps, "Source image plus target record trains target image patch output expert."),
        StageSpec("joint_debug", config.joint_steps, "Short final interface tune without treating answer-only shortcuts as success."),
    ]


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "AV-H":
        raise ValueError(f"expected AV-H manifest, got {manifest.get('stage')!r}")
    return manifest


def find_split(manifest: dict[str, object], split: str) -> dict[str, object]:
    for row in manifest.get("splits", []):
        if isinstance(row, dict) and row.get("split") == split:
            return row
    raise ValueError(f"missing split {split!r}")


def load_split(manifest_path: Path, manifest: dict[str, object], split: str, limit: int = 0) -> AVHSet:
    split_info = find_split(manifest, split)
    base = manifest_path.parent
    parts: dict[str, list[torch.Tensor]] = {
        "source_text": [],
        "edit_text": [],
        "target_text": [],
        "source_record": [],
        "target_record": [],
        "edit_type": [],
        "edit_object": [],
        "edit_value": [],
    }
    case_ids: list[str] = []
    loaded = 0
    for shard in split_info.get("shards", []):
        path = Path(str(shard["path"]))
        shard_path = path if path.is_absolute() else base / path
        data = torch.load(shard_path, map_location="cpu")
        count = int(data["source_text"].shape[0])
        take = count
        if limit > 0:
            remaining = limit - loaded
            if remaining <= 0:
                break
            take = min(take, remaining)
        for key in parts:
            parts[key].append(data[key][:take])
        case_ids.extend(list(data["case_ids"][:take]))
        loaded += take
        if limit > 0 and loaded >= limit:
            break
    if not parts["source_text"]:
        raise ValueError(f"split {split!r} has no data")
    return AVHSet(
        source_text=torch.cat(parts["source_text"], dim=0),
        edit_text=torch.cat(parts["edit_text"], dim=0),
        target_text=torch.cat(parts["target_text"], dim=0),
        source_record=torch.cat(parts["source_record"], dim=0),
        target_record=torch.cat(parts["target_record"], dim=0),
        edit_type=torch.cat(parts["edit_type"], dim=0),
        edit_object=torch.cat(parts["edit_object"], dim=0),
        edit_value=torch.cat(parts["edit_value"], dim=0),
        case_ids=case_ids,
    )


def load_datasets(config: StageAVHTrainConfig) -> tuple[AVHSet, AVHSet, AVHSet, AVHSet, dict[str, object]]:
    manifest_path = Path(config.dataset_manifest)
    manifest = load_manifest(manifest_path)
    train = load_split(manifest_path, manifest, "train", config.train_limit)
    val = load_split(manifest_path, manifest, "val", config.val_limit)
    test = load_split(manifest_path, manifest, "test", config.test_limit)
    heldout = load_split(manifest_path, manifest, "heldout", config.heldout_limit)
    return train, val, test, heldout, {
        "source": "manifest",
        "manifest": str(manifest_path),
        "manifest_scale": manifest.get("scale", ""),
        "manifest_total_examples": manifest.get("total_examples", 0),
        "manifest_total_tokens": manifest.get("total_tokens", 0),
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
        "heldout_examples": len(heldout),
        "unique_train_tokens": len(train) * TOKENS_PER_EXAMPLE,
        "tokens_per_example": TOKENS_PER_EXAMPLE,
    }


def random_batch(data: AVHSet, batch_size: int, device: torch.device) -> AVHSet:
    index_device = data.source_text.device
    indices = torch.randint(0, len(data), (batch_size,), device=index_device)
    return data.tensor_subset(indices, include_case_ids=False).to(device)


def micro_batch_sizes(batch_size: int, micro_batch_size: int) -> list[int]:
    micro = batch_size if micro_batch_size <= 0 else min(batch_size, micro_batch_size)
    sizes = []
    remaining = batch_size
    while remaining > 0:
        take = min(micro, remaining)
        sizes.append(take)
        remaining -= take
    return sizes


def patchify(images: torch.Tensor) -> torch.Tensor:
    patches = F.unfold(images, kernel_size=PATCH_SIZE, stride=PATCH_SIZE).transpose(1, 2)
    return patches


def unpatchify(patches: torch.Tensor) -> torch.Tensor:
    batch = patches.shape[0]
    folded = F.fold(patches.transpose(1, 2), output_size=(IMAGE_SIZE, IMAGE_SIZE), kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    return folded.view(batch, 3, IMAGE_SIZE, IMAGE_SIZE).clamp(0.0, 1.0)


class TextAnchoredVisualEditModel(nn.Module):
    def __init__(self, config: StageAVHTrainConfig) -> None:
        super().__init__()
        self.config = config
        self.char = nn.Embedding(CHAR_VOCAB, config.d_model, padding_idx=0)
        self.record = nn.Embedding(RECORD_VOCAB, config.d_model, padding_idx=0)
        self.modality = nn.Embedding(8, config.d_model)
        self.position = nn.Embedding(512, config.d_model)
        self.patch_in = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, config.d_model)
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
        self.record_query = nn.Parameter(torch.randn(RECORD_LEN, config.d_model) * 0.02)
        self.source_record_query = nn.Parameter(torch.randn(RECORD_LEN, config.d_model) * 0.02)
        self.text_query = nn.Parameter(torch.randn(TARGET_TEXT_LEN, config.d_model) * 0.02)
        self.image_query = nn.Parameter(torch.randn(PATCHES_PER_IMAGE, config.d_model) * 0.02)
        self.record_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.source_record_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.text_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.image_attn = nn.MultiheadAttention(config.d_model, config.heads, batch_first=True)
        self.record_head = nn.Linear(config.d_model, RECORD_VOCAB)
        self.source_record_head = nn.Linear(config.d_model, RECORD_VOCAB)
        self.text_head = nn.Linear(config.d_model, CHAR_VOCAB)
        self.patch_head = nn.Linear(config.d_model, 3 * PATCH_SIZE * PATCH_SIZE)

    def add_positions(self, x: torch.Tensor, modality_id: int) -> torch.Tensor:
        pos = torch.arange(x.shape[1], device=x.device)
        return x + self.position(pos).view(1, -1, x.shape[-1]) + self.modality.weight[modality_id].view(1, 1, -1)

    def encode_context(
        self,
        batch: AVHSet,
        *,
        use_source_text: bool,
        use_edit_text: bool,
        use_source_record: bool,
        use_target_record: bool,
        use_source_image: bool,
    ) -> torch.Tensor:
        chunks = []
        if use_source_text:
            chunks.append(self.add_positions(self.char(batch.source_text), 0))
        if use_edit_text:
            chunks.append(self.add_positions(self.char(batch.edit_text), 1))
        if use_source_record:
            chunks.append(self.add_positions(self.record(batch.source_record), 2))
        if use_target_record:
            chunks.append(self.add_positions(self.record(batch.target_record), 3))
        if use_source_image:
            source_images = render_records(batch.source_record, device=batch.source_record.device)
            chunks.append(self.add_positions(self.patch_in(patchify(source_images)), 4))
        latent = self.latent.unsqueeze(0).expand(len(batch), -1, -1) + self.modality.weight[5].view(1, 1, -1)
        chunks.append(latent)
        encoded = self.encoder(torch.cat(chunks, dim=1))
        return self.norm(encoded)

    def decode_from_memory(self, memory: torch.Tensor, outputs: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
        outputs = outputs or ("target_record", "source_record", "target_text", "target_patches")
        output_set = set(outputs)
        batch = memory.shape[0]
        decoded: dict[str, torch.Tensor] = {}
        if "target_record" in output_set:
            record_query = self.record_query.unsqueeze(0).expand(batch, -1, -1)
            record_hidden, _ = self.record_attn(record_query, memory, memory, need_weights=False)
            decoded["target_record"] = self.record_head(record_hidden)
        if "source_record" in output_set:
            source_record_query = self.source_record_query.unsqueeze(0).expand(batch, -1, -1)
            source_record_hidden, _ = self.source_record_attn(source_record_query, memory, memory, need_weights=False)
            decoded["source_record"] = self.source_record_head(source_record_hidden)
        if "target_text" in output_set:
            text_query = self.text_query.unsqueeze(0).expand(batch, -1, -1)
            text_hidden, _ = self.text_attn(text_query, memory, memory, need_weights=False)
            decoded["target_text"] = self.text_head(text_hidden)
        if "target_patches" in output_set:
            image_query = self.image_query.unsqueeze(0).expand(batch, -1, -1)
            image_hidden, _ = self.image_attn(image_query, memory, memory, need_weights=False)
            decoded["target_patches"] = self.patch_head(image_hidden).sigmoid()
        return decoded

    def forward(self, batch: AVHSet, mode: str, outputs: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
        if mode == "text_latent":
            memory = self.encode_context(batch, use_source_text=True, use_edit_text=True, use_source_record=False, use_target_record=False, use_source_image=False)
        elif mode == "image_ground":
            memory = self.encode_context(batch, use_source_text=False, use_edit_text=False, use_source_record=False, use_target_record=False, use_source_image=True)
        elif mode == "edit_reason":
            memory = self.encode_context(batch, use_source_text=True, use_edit_text=True, use_source_record=True, use_target_record=False, use_source_image=False)
        elif mode == "image_output":
            memory = self.encode_context(batch, use_source_text=False, use_edit_text=True, use_source_record=False, use_target_record=True, use_source_image=True)
        elif mode == "joint_debug":
            memory = self.encode_context(batch, use_source_text=True, use_edit_text=True, use_source_record=False, use_target_record=True, use_source_image=True)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return self.decode_from_memory(memory, outputs)


def sequence_exact(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).all(dim=1).float().mean().item())


def token_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).float().mean().item())


def record_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, RECORD_VOCAB), target.reshape(-1))


def text_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, CHAR_VOCAB), target.reshape(-1), ignore_index=0)


def image_loss(pred_patches: torch.Tensor, target_record: torch.Tensor) -> torch.Tensor:
    target_images = render_records(target_record, device=target_record.device)
    return F.mse_loss(pred_patches, patchify(target_images))


def stage_loss(model: TextAnchoredVisualEditModel, batch: AVHSet, stage: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if stage == "text_latent":
        out = model(batch, stage, outputs=("target_record", "target_text"))
        target_record = record_loss(out["target_record"], batch.target_record)
        target_text = text_loss(out["target_text"], batch.target_text)
        loss = target_text + target_record
        components = {"loss_target_record": target_record.detach(), "loss_target_text": target_text.detach()}
    elif stage == "image_ground":
        out = model(batch, stage, outputs=("source_record",))
        source_record = record_loss(out["source_record"], batch.source_record)
        loss = source_record
        components = {"loss_source_record": source_record.detach()}
    elif stage == "edit_reason":
        out = model(batch, stage, outputs=("target_record", "target_text"))
        target_record = record_loss(out["target_record"], batch.target_record)
        target_text = text_loss(out["target_text"], batch.target_text)
        loss = target_record + 0.5 * target_text
        components = {"loss_target_record": target_record.detach(), "loss_target_text": target_text.detach()}
    elif stage == "image_output":
        out = model(batch, stage, outputs=("target_patches",))
        image = image_loss(out["target_patches"], batch.target_record)
        loss = image
        components = {"loss_image_mse": image.detach()}
    elif stage == "joint_debug":
        out = model(batch, stage, outputs=("target_record", "source_record", "target_text", "target_patches"))
        target_record = record_loss(out["target_record"], batch.target_record)
        target_text = text_loss(out["target_text"], batch.target_text)
        source_record = record_loss(out["source_record"], batch.source_record)
        image = image_loss(out["target_patches"], batch.target_record)
        loss = target_record + 0.5 * target_text + image + 0.25 * source_record
        components = {
            "loss_target_record": target_record.detach(),
            "loss_target_text": target_text.detach(),
            "loss_source_record": source_record.detach(),
            "loss_image_mse": image.detach(),
        }
    else:
        raise ValueError(stage)
    return loss, components


def component_values(components: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.item()) for key, value in components.items()}


@torch.inference_mode()
def evaluate(model: TextAnchoredVisualEditModel, data: AVHSet, config: StageAVHTrainConfig, device: torch.device, prefix: str) -> dict[str, float]:
    model.eval()

    def new_totals() -> dict[str, float]:
        return {
            "target_record_hits": 0.0,
            "target_record_tokens": 0.0,
            "target_record_exact": 0.0,
            "source_record_hits": 0.0,
            "source_record_tokens": 0.0,
            "source_record_exact": 0.0,
            "target_text_hits": 0.0,
            "target_text_tokens": 0.0,
            "target_text_exact": 0.0,
            "image_mse_sum": 0.0,
            "examples": 0.0,
        }

    def add_outputs(totals: dict[str, float], batch: AVHSet, out: dict[str, torch.Tensor]) -> None:
        if "target_record" in out:
            target_record_pred = out["target_record"].argmax(dim=-1)
            totals["target_record_hits"] += float((target_record_pred == batch.target_record).sum().item())
            totals["target_record_tokens"] += float(batch.target_record.numel())
            totals["target_record_exact"] += float((target_record_pred == batch.target_record).all(dim=1).float().sum().item())
        if "source_record" in out:
            source_record_pred = out["source_record"].argmax(dim=-1)
            totals["source_record_hits"] += float((source_record_pred == batch.source_record).sum().item())
            totals["source_record_tokens"] += float(batch.source_record.numel())
            totals["source_record_exact"] += float((source_record_pred == batch.source_record).all(dim=1).float().sum().item())
        if "target_text" in out:
            target_text_pred = out["target_text"].argmax(dim=-1)
            text_mask = batch.target_text != 0
            totals["target_text_hits"] += float(((target_text_pred == batch.target_text) & text_mask).sum().item())
            totals["target_text_tokens"] += float(text_mask.sum().item())
            totals["target_text_exact"] += float(((target_text_pred == batch.target_text) | ~text_mask).all(dim=1).float().sum().item())
        if "target_patches" in out:
            totals["image_mse_sum"] += float(image_loss(out["target_patches"], batch.target_record).item()) * len(batch)
        totals["examples"] += float(len(batch))

    def finalize(totals: dict[str, float], mode: str) -> dict[str, float]:
        base = f"{prefix}_{mode}"
        metrics: dict[str, float] = {}
        if totals["target_record_tokens"]:
            metrics[f"{base}_target_record_token_accuracy"] = totals["target_record_hits"] / totals["target_record_tokens"]
            metrics[f"{base}_target_record_exact"] = totals["target_record_exact"] / totals["examples"]
        if totals["source_record_tokens"]:
            metrics[f"{base}_source_record_token_accuracy"] = totals["source_record_hits"] / totals["source_record_tokens"]
            metrics[f"{base}_source_record_exact"] = totals["source_record_exact"] / totals["examples"]
        if totals["target_text_tokens"]:
            metrics[f"{base}_target_text_token_accuracy"] = totals["target_text_hits"] / max(1.0, totals["target_text_tokens"])
            metrics[f"{base}_target_text_exact"] = totals["target_text_exact"] / totals["examples"]
        if totals["image_mse_sum"]:
            metrics[f"{base}_target_image_mse"] = totals["image_mse_sum"] / totals["examples"]
        return metrics

    mode_totals = {
        "text_latent": new_totals(),
        "image_ground": new_totals(),
        "edit_reason": new_totals(),
        "image_output": new_totals(),
        "joint_teacher_record": new_totals(),
        "joint_no_target_record": new_totals(),
    }
    for start in range(0, len(data), config.eval_batch_size):
        batch = data.slice(start, min(start + config.eval_batch_size, len(data))).to(device)
        with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
            add_outputs(mode_totals["text_latent"], batch, model(batch, "text_latent", outputs=("target_record", "target_text")))
            add_outputs(mode_totals["image_ground"], batch, model(batch, "image_ground", outputs=("source_record",)))
            add_outputs(mode_totals["edit_reason"], batch, model(batch, "edit_reason", outputs=("target_record", "target_text")))
            add_outputs(mode_totals["image_output"], batch, model(batch, "image_output", outputs=("target_patches",)))
            add_outputs(
                mode_totals["joint_teacher_record"],
                batch,
                model(batch, "joint_debug", outputs=("target_record", "source_record", "target_text", "target_patches")),
            )
            memory = model.encode_context(batch, use_source_text=True, use_edit_text=True, use_source_record=False, use_target_record=False, use_source_image=True)
            add_outputs(
                mode_totals["joint_no_target_record"],
                batch,
                model.decode_from_memory(memory, outputs=("target_record", "source_record", "target_text", "target_patches")),
            )
    metrics: dict[str, float] = {}
    for mode, totals in mode_totals.items():
        metrics.update(finalize(totals, mode))
    # Compatibility keys are intentionally mapped to the non-teacher-forced route.
    metrics[f"{prefix}_target_record_token_accuracy"] = metrics[f"{prefix}_joint_no_target_record_target_record_token_accuracy"]
    metrics[f"{prefix}_target_record_exact"] = metrics[f"{prefix}_joint_no_target_record_target_record_exact"]
    metrics[f"{prefix}_source_record_token_accuracy"] = metrics[f"{prefix}_joint_no_target_record_source_record_token_accuracy"]
    metrics[f"{prefix}_source_record_exact"] = metrics[f"{prefix}_joint_no_target_record_source_record_exact"]
    metrics[f"{prefix}_target_text_token_accuracy"] = metrics[f"{prefix}_joint_no_target_record_target_text_token_accuracy"]
    metrics[f"{prefix}_target_text_exact"] = metrics[f"{prefix}_joint_no_target_record_target_text_exact"]
    metrics[f"{prefix}_target_image_mse"] = metrics[f"{prefix}_joint_no_target_record_target_image_mse"]
    return metrics


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def checkpoint_dir(config: StageAVHTrainConfig) -> Path | None:
    if not config.checkpoint_dir:
        return None
    path = Path(config.checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, config: StageAVHTrainConfig, step: int, history: list[dict[str, float]], best_score: float) -> None:
    torch.save(
        {
            "schema_version": 1,
            "stage": "AV-H",
            "config": asdict(config),
            "global_step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_score": best_score,
        },
        path,
    )


def best_score(metrics: dict[str, float]) -> float:
    return metrics["val_target_record_exact"] + metrics["val_source_record_exact"] + metrics["val_target_text_exact"] - metrics["val_target_image_mse"]


def build_optimizer(model: nn.Module, config: StageAVHTrainConfig, device: torch.device) -> torch.optim.Optimizer:
    if config.optimizer == "adafactor":
        if not hasattr(torch.optim, "Adafactor"):
            raise RuntimeError("torch.optim.Adafactor is unavailable in this PyTorch build")
        return torch.optim.Adafactor(model.parameters(), lr=config.lr, weight_decay=0.01, foreach=False)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, foreach=False, fused=False)
    if config.optimizer == "adamw-fused":
        if device.type != "cuda":
            return torch.optim.AdamW(model.parameters(), lr=config.lr, foreach=False, fused=False)
        return torch.optim.AdamW(model.parameters(), lr=config.lr, fused=True)
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def train_model(model: TextAnchoredVisualEditModel, train: AVHSet, val: AVHSet, config: StageAVHTrainConfig, device: torch.device) -> dict[str, object]:
    model.to(device)
    optimizer = build_optimizer(model, config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    ckpt_dir = checkpoint_dir(config)
    latest_path = ckpt_dir / "latest.pt" if ckpt_dir is not None else None
    best_path = ckpt_dir / "best.pt" if ckpt_dir is not None else None
    history: list[dict[str, float]] = []
    global_step = 0
    score = -1e9
    resumed_from = ""
    log_step = 0
    log_started = time.perf_counter()
    if config.resume and latest_path is not None and latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = list(state.get("history", []))
        global_step = int(state.get("global_step", 0))
        score = float(state.get("best_score", score))
        resumed_from = str(latest_path)

    stage_start = 1
    for stage in build_stage_plan(config):
        stage_end = stage_start + stage.steps - 1
        if global_step >= stage_end:
            stage_start = stage_end + 1
            continue
        local_start = max(1, global_step - stage_start + 2)
        for _local in range(local_start, stage.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss: torch.Tensor | None = None
            component_sums: dict[str, torch.Tensor] = {}
            for micro_size in micro_batch_sizes(config.batch_size, config.micro_batch_size):
                batch = random_batch(train, micro_size, device)
                weight = float(micro_size) / float(config.batch_size)
                with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                    micro_loss, components = stage_loss(model, batch, stage.name)
                    weighted_loss = micro_loss * weight
                scaler.scale(weighted_loss).backward()
                total_loss = weighted_loss.detach() if total_loss is None else total_loss + weighted_loss.detach()
                for key, value in components.items():
                    weighted_value = value * weight
                    component_sums[key] = weighted_value if key not in component_sums else component_sums[key] + weighted_value
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            should_eval = (config.eval_every > 0 and global_step % config.eval_every == 0) or global_step == config.total_steps
            if should_eval:
                eval_started = time.perf_counter()
                metrics = evaluate(model, val, config, device, "val")
                train_window_sec = eval_started - log_started
                metrics.update(component_values(component_sums))
                metrics["stage"] = stage.name
                metrics["global_step"] = float(global_step)
                metrics["loss"] = float(total_loss.item()) if total_loss is not None else 0.0
                metrics["micro_batch_size"] = float(min(config.batch_size, config.micro_batch_size if config.micro_batch_size > 0 else config.batch_size))
                metrics["processed_tokens"] = float(global_step * config.batch_size * TOKENS_PER_EXAMPLE)
                metrics["steps_since_log"] = float(global_step - log_step)
                metrics["train_window_sec"] = train_window_sec
                metrics["train_steps_per_sec"] = float(global_step - log_step) / max(1e-9, train_window_sec)
                metrics["eval_sec"] = time.perf_counter() - eval_started
                history.append(metrics)
                current_score = best_score(metrics)
                if best_path is not None and current_score > score:
                    score = current_score
                    save_checkpoint(best_path, model, optimizer, scaler, config, global_step, history, score)
                print(json.dumps(metrics, ensure_ascii=False), flush=True)
                log_step = global_step
                log_started = time.perf_counter()
            if latest_path is not None and (global_step % config.save_every == 0 or global_step == config.total_steps):
                save_checkpoint(latest_path, model, optimizer, scaler, config, global_step, history, score)
        stage_start = stage_end + 1
    return {
        "history": history,
        "completed_steps": global_step,
        "total_steps": config.total_steps,
        "resumed_from": resumed_from,
        "checkpoint_dir": config.checkpoint_dir,
        "latest_checkpoint": str(latest_path) if latest_path else "",
        "best_checkpoint": str(best_path) if best_path else "",
        "best_score": score,
        "optimizer": config.optimizer,
        "stage_plan": [asdict(stage) for stage in build_stage_plan(config)],
    }


@torch.inference_mode()
def write_samples(model: TextAnchoredVisualEditModel, data: AVHSet, output_dir: Path, config: StageAVHTrainConfig, device: torch.device) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    batch = data.slice(0, min(config.sample_count, len(data))).to(device)
    with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
        memory = model.encode_context(batch, use_source_text=True, use_edit_text=True, use_source_record=False, use_target_record=False, use_source_image=True)
        out = model.decode_from_memory(memory, outputs=("target_record", "target_patches"))
    predicted_images = unpatchify(out["target_patches"]).detach().cpu()
    source_images = render_records(batch.source_record, device=device).detach().cpu()
    target_images = render_records(batch.target_record, device=device).detach().cpu()
    rows = []
    for index, case_id in enumerate(batch.case_ids):
        source_path = output_dir / f"{case_id}_source.png"
        target_path = output_dir / f"{case_id}_target.png"
        pred_path = output_dir / f"{case_id}_pred.png"
        write_png(source_path, source_images[index])
        write_png(target_path, target_images[index])
        write_png(pred_path, predicted_images[index])
        rows.append(
            {
                "case_id": case_id,
                "source_png": str(source_path),
                "target_png": str(target_path),
                "pred_png": str(pred_path),
                "target_record": batch.target_record[index].detach().cpu().tolist(),
                "pred_record": out["target_record"].argmax(dim=-1)[index].detach().cpu().tolist(),
            }
        )
    (output_dir / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested cuda but CUDA is not available")
    return torch.device(name)


def run_one(config: StageAVHTrainConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train, val, test, heldout, dataset_info = load_datasets(config)
    if config.gpu_resident_data and device.type == "cuda":
        train = train.to(device)
        val = val.to(device)
    model = TextAnchoredVisualEditModel(config)
    training = train_model(model, train, val, config, device)
    test_metrics = evaluate(model, test, config, device, "test")
    heldout_metrics = evaluate(model, heldout, config, device, "heldout")
    write_samples(model, test, output_path.parent / "samples", config, device)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    result = {
        "schema_version": 1,
        "stage": "AV-H",
        "config": asdict(config),
        "dataset": dataset_info,
        "device": str(device),
        "metrics": {**test_metrics, **heldout_metrics},
        "cost": {
            "elapsed_sec": time.perf_counter() - started,
            "parameters": parameter_count(model),
            "peak_cuda_allocated_mb": peak_mb,
            "processed_tokens": config.total_steps * config.batch_size * TOKENS_PER_EXAMPLE,
        },
        "training": training,
        "interpretation": {
            "goal": "Train text-anchored visual edit route after AV-E/AV-G high-entropy unanchored external translation failed.",
            "stages": "text latent -> image grounding -> edit reason -> image output -> short joint debug.",
            "success_condition": "Target text exact, target record exact, source image grounding, and target image output must all move together.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": result["metrics"], "cost": result["cost"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_avh_text_anchored_visual_edit_training/train_10m_68m_result.json"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path(StageAVHTrainConfig.dataset_manifest))
    parser.add_argument("--batch-size", type=int, default=StageAVHTrainConfig.batch_size)
    parser.add_argument("--micro-batch-size", type=int, default=StageAVHTrainConfig.micro_batch_size)
    parser.add_argument("--seed", type=int, default=StageAVHTrainConfig.seed)
    parser.add_argument("--d-model", type=int, default=StageAVHTrainConfig.d_model)
    parser.add_argument("--heads", type=int, default=StageAVHTrainConfig.heads)
    parser.add_argument("--layers", type=int, default=StageAVHTrainConfig.layers)
    parser.add_argument("--latent-tokens", type=int, default=StageAVHTrainConfig.latent_tokens)
    parser.add_argument("--lr", type=float, default=StageAVHTrainConfig.lr)
    parser.add_argument("--optimizer", choices=("adafactor", "adamw", "adamw-fused"), default=StageAVHTrainConfig.optimizer)
    parser.add_argument("--text-steps", type=int, default=StageAVHTrainConfig.text_steps)
    parser.add_argument("--image-ground-steps", type=int, default=StageAVHTrainConfig.image_ground_steps)
    parser.add_argument("--edit-reason-steps", type=int, default=StageAVHTrainConfig.edit_reason_steps)
    parser.add_argument("--image-output-steps", type=int, default=StageAVHTrainConfig.image_output_steps)
    parser.add_argument("--joint-steps", type=int, default=StageAVHTrainConfig.joint_steps)
    parser.add_argument("--eval-every", type=int, default=StageAVHTrainConfig.eval_every)
    parser.add_argument("--eval-batch-size", type=int, default=StageAVHTrainConfig.eval_batch_size)
    parser.add_argument("--sample-count", type=int, default=StageAVHTrainConfig.sample_count)
    parser.add_argument("--train-limit", type=int, default=StageAVHTrainConfig.train_limit)
    parser.add_argument("--val-limit", type=int, default=StageAVHTrainConfig.val_limit)
    parser.add_argument("--test-limit", type=int, default=StageAVHTrainConfig.test_limit)
    parser.add_argument("--heldout-limit", type=int, default=StageAVHTrainConfig.heldout_limit)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(StageAVHTrainConfig.checkpoint_dir))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=StageAVHTrainConfig.save_every)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=StageAVHTrainConfig.device)
    parser.add_argument("--no-gpu-resident-data", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> StageAVHTrainConfig:
    return StageAVHTrainConfig(
        dataset_manifest=str(args.dataset_manifest),
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        seed=args.seed,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        latent_tokens=args.latent_tokens,
        lr=args.lr,
        optimizer=args.optimizer,
        text_steps=args.text_steps,
        image_ground_steps=args.image_ground_steps,
        edit_reason_steps=args.edit_reason_steps,
        image_output_steps=args.image_output_steps,
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
