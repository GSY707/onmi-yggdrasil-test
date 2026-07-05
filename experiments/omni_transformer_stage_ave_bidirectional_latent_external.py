from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visual_multimodal_stage_ab import write_png  # noqa: E402


COLORS = 8
ZONES = 6
VALUES = 8
OPS = 4
TEXT_LEN = 8
TOOL_LEN = 6
MEMORY_LEN = 6
SEQ_LEN = 1 + ZONES + TEXT_LEN + TOOL_LEN + MEMORY_LEN
ANSWER_CLASSES = 16
PAIR_TOKENS_PER_EXAMPLE = SEQ_LEN * 2 + 1

PAD = 0
TASK_TOKEN = 1
ZONE_BASE = 2
COLOR_BASE = ZONE_BASE + ZONES
VALUE_BASE = COLOR_BASE + COLORS
OP_BASE = VALUE_BASE + VALUES
TEXT_BASE = OP_BASE + OPS
TOOL_BASE = TEXT_BASE + 64
MEMORY_BASE = TOOL_BASE + TOOL_LEN * VALUES
TOKEN_VOCAB = MEMORY_BASE + MEMORY_LEN * (COLORS + 1)


@dataclass(frozen=True)
class StageAVEConfig:
    train_size: int = 4096
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 64
    seed: int = 20260705
    d_model: int = 512
    heads: int = 8
    layers: int = 8
    latent_tokens: int = 8
    lr: float = 5e-4
    steps: int = 800
    eval_every: int = 200
    eval_batch_size: int = 512
    sample_count: int = 8
    amp: bool = True
    dataset_manifest: str = ""
    train_limit: int = 0
    val_limit: int = 0
    test_limit: int = 0
    heldout_limit: int = 0
    gpu_resident_data: bool = False
    checkpoint_dir: str = ""
    resume: bool = False
    save_every: int = 500


@dataclass(frozen=True)
class AVEExample:
    source_image: tuple[int, ...]
    source_tool: tuple[int, ...]
    source_memory: tuple[int, ...]
    target_image: tuple[int, ...]
    target_tool: tuple[int, ...]
    target_memory: tuple[int, ...]
    target_zone: int
    op: int
    op_arg: int
    answer: int
    case_id: str


@dataclass(frozen=True)
class AVESet:
    examples: list[AVEExample]
    source_tokens: torch.Tensor
    target_tokens: torch.Tensor
    answer: torch.Tensor

    def __len__(self) -> int:
        return int(self.source_tokens.shape[0])

    def subset(self, indices: list[int]) -> "AVESet":
        examples = [self.examples[index] for index in indices] if self.examples else []
        return AVESet(
            examples,
            self.source_tokens[indices],
            self.target_tokens[indices],
            self.answer[indices],
        )

    def tensor_subset(self, indices: torch.Tensor) -> "AVESet":
        return AVESet(
            [],
            self.source_tokens.index_select(0, indices),
            self.target_tokens.index_select(0, indices),
            self.answer.index_select(0, indices),
        )

    def slice(self, start: int, end: int) -> "AVESet":
        examples = self.examples[start:end] if self.examples else []
        return AVESet(
            examples,
            self.source_tokens[start:end],
            self.target_tokens[start:end],
            self.answer[start:end],
        )

    def to(self, device: torch.device) -> "AVESet":
        return AVESet(
            self.examples,
            self.source_tokens.to(device=device, dtype=torch.long),
            self.target_tokens.to(device=device, dtype=torch.long),
            self.answer.to(device=device, dtype=torch.long),
        )


def encode_external(image: tuple[int, ...], tool: tuple[int, ...], memory: tuple[int, ...], target_zone: int, op: int, op_arg: int) -> list[int]:
    text = [
        ZONE_BASE + target_zone,
        OP_BASE + op,
        VALUE_BASE + op_arg,
        COLOR_BASE + image[target_zone],
    ]
    text.extend(TEXT_BASE + ((target_zone * 11 + op * 7 + op_arg * 3 + i) % 64) for i in range(TEXT_LEN - len(text)))
    tokens = [TASK_TOKEN]
    tokens.extend(COLOR_BASE + color for color in image)
    tokens.extend(text)
    tokens.extend(TOOL_BASE + slot * VALUES + value for slot, value in enumerate(tool))
    tokens.extend(MEMORY_BASE + slot * (COLORS + 1) + value + 1 for slot, value in enumerate(memory))
    return tokens


def apply_operation(image: list[int], tool: list[int], memory: list[int], target_zone: int, op: int, op_arg: int) -> tuple[list[int], list[int], list[int]]:
    target_image = image[:]
    target_tool = tool[:]
    target_memory = memory[:]
    if op == 0:
        target_image[target_zone] = op_arg % COLORS
    elif op == 1:
        shift = 1 + (op_arg % (ZONES - 1))
        target_image = target_image[-shift:] + target_image[:-shift]
    elif op == 2:
        target_memory[target_zone] = target_image[target_zone]
    else:
        slot = target_zone % TOOL_LEN
        target_tool[slot] = (target_tool[slot] + target_image[target_zone] + op_arg) % VALUES
    return target_image, target_tool, target_memory


def answer_for(image: list[int], tool: list[int], memory: list[int], target_zone: int, op_arg: int) -> int:
    return (image[target_zone] + tool[target_zone % TOOL_LEN] + memory[target_zone] + op_arg) % ANSWER_CLASSES


def make_example(index: int, rng: random.Random, split: str) -> AVEExample:
    image = [rng.randrange(COLORS) for _ in range(ZONES)]
    if split == "heldout":
        image = [(value + index) % COLORS for value in reversed(image)]
    tool = [rng.randrange(VALUES) for _ in range(TOOL_LEN)]
    memory = image[:]
    if index % 5 == 0:
        z0 = rng.randrange(ZONES)
        z1 = (z0 + 1 + rng.randrange(ZONES - 1)) % ZONES
        memory[z0], memory[z1] = memory[z1], memory[z0]
    target_zone = rng.randrange(ZONES)
    op = index % OPS
    op_arg = rng.randrange(VALUES)
    target_image, target_tool, target_memory = apply_operation(image, tool, memory, target_zone, op, op_arg)
    answer = answer_for(target_image, target_tool, target_memory, target_zone, op_arg)
    return AVEExample(
        source_image=tuple(image),
        source_tool=tuple(tool),
        source_memory=tuple(memory),
        target_image=tuple(target_image),
        target_tool=tuple(target_tool),
        target_memory=tuple(target_memory),
        target_zone=target_zone,
        op=op,
        op_arg=op_arg,
        answer=answer,
        case_id=f"{split}-{index:06d}",
    )


def build_split(split: str, size: int, config: StageAVEConfig) -> AVESet:
    offsets = {"train": 0, "val": 100_000, "test": 200_000, "heldout": 300_000}
    rng = random.Random(config.seed + offsets[split])
    examples = [make_example(index, rng, split) for index in range(size)]
    return AVESet(
        examples=examples,
        source_tokens=torch.tensor(
            [encode_external(e.source_image, e.source_tool, e.source_memory, e.target_zone, e.op, e.op_arg) for e in examples],
            dtype=torch.long,
        ),
        target_tokens=torch.tensor(
            [encode_external(e.target_image, e.target_tool, e.target_memory, e.target_zone, e.op, e.op_arg) for e in examples],
            dtype=torch.long,
        ),
        answer=torch.tensor([e.answer for e in examples], dtype=torch.long),
    )


def find_manifest_split(manifest: dict[str, object], split: str) -> dict[str, object]:
    for row in manifest.get("splits", []):
        if isinstance(row, dict) and row.get("split") == split:
            return row
    raise ValueError(f"manifest is missing split {split!r}")


def resolve_manifest_path(manifest_dir: Path, shard_path: str) -> Path:
    path = Path(shard_path)
    return path if path.is_absolute() else manifest_dir / path


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("compatible_stage") not in (None, "AV-E"):
        raise ValueError(f"manifest {path} is not marked compatible with AV-E")
    if int(manifest.get("seq_len", SEQ_LEN)) != SEQ_LEN:
        raise ValueError(f"manifest seq_len does not match AV-E SEQ_LEN={SEQ_LEN}")
    if int(manifest.get("vocab_size", TOKEN_VOCAB)) > TOKEN_VOCAB:
        raise ValueError(f"manifest vocab_size exceeds AV-E TOKEN_VOCAB={TOKEN_VOCAB}")
    return manifest


def load_manifest_split(manifest_path: Path, manifest: dict[str, object], split: str, limit: int = 0) -> AVESet:
    split_info = find_manifest_split(manifest, split)
    manifest_dir = manifest_path.parent
    source_parts = []
    target_parts = []
    answer_parts = []
    loaded = 0
    for shard in split_info.get("shards", []):
        if not isinstance(shard, dict):
            continue
        shard_path = resolve_manifest_path(manifest_dir, str(shard["path"]))
        data = torch.load(shard_path, map_location="cpu")
        source = data["source_tokens"]
        target = data["target_tokens"]
        answer = data["answers"] if "answers" in data else data["answer"]
        if limit > 0:
            remaining = limit - loaded
            if remaining <= 0:
                break
            source = source[:remaining]
            target = target[:remaining]
            answer = answer[:remaining]
        source_parts.append(source)
        target_parts.append(target)
        answer_parts.append(answer)
        loaded += int(source.shape[0])
        if limit > 0 and loaded >= limit:
            break
    if not source_parts:
        raise ValueError(f"manifest split {split!r} has no loadable shards")
    return AVESet(
        examples=[],
        source_tokens=torch.cat(source_parts, dim=0),
        target_tokens=torch.cat(target_parts, dim=0),
        answer=torch.cat(answer_parts, dim=0),
    )


def random_batch(data: AVESet, rng: random.Random, batch_size: int, device: torch.device) -> AVESet:
    if len(data) == 0:
        raise ValueError("cannot sample from empty AVESet")
    if data.examples and data.source_tokens.device.type == "cpu":
        return data.subset([rng.randrange(len(data)) for _ in range(batch_size)]).to(device)
    index_device = data.source_tokens.device if data.source_tokens.device.type != "cpu" else torch.device("cpu")
    indices = torch.randint(0, len(data), (batch_size,), device=index_device)
    return data.tensor_subset(indices).to(device)


class BidirectionalLatentExternalModel(nn.Module):
    def __init__(self, config: StageAVEConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(TOKEN_VOCAB, config.d_model, padding_idx=PAD)
        self.type_embed = nn.Embedding(4, config.d_model)
        self.position = nn.Parameter(torch.randn(SEQ_LEN + config.latent_tokens, config.d_model) * 0.02)
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
        self.source_to_target = nn.Sequential(nn.Linear(config.d_model, config.d_model * 2), nn.GELU(), nn.Linear(config.d_model * 2, config.d_model))
        self.target_to_source = nn.Sequential(nn.Linear(config.d_model, config.d_model * 2), nn.GELU(), nn.Linear(config.d_model * 2, config.d_model))
        self.decode_query = nn.Parameter(torch.randn(SEQ_LEN, config.d_model) * 0.02)
        self.decode_head = nn.Linear(config.d_model, TOKEN_VOCAB)
        self.answer_head = nn.Linear(config.d_model, ANSWER_CLASSES)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        x = self.token(tokens) + self.type_embed.weight[0].view(1, 1, -1)
        latent = self.latent.unsqueeze(0).expand(batch, -1, -1) + self.type_embed.weight[1].view(1, 1, -1)
        seq = torch.cat((x, latent), dim=1)
        seq = seq + self.position[: seq.shape[1]].view(1, -1, seq.shape[-1])
        encoded = self.encoder(seq)
        return self.norm(encoded[:, -latent.shape[1] :])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        pooled = latent.mean(dim=1, keepdim=True)
        queries = self.decode_query.unsqueeze(0).expand(latent.shape[0], -1, -1)
        hidden = queries + pooled
        return self.decode_head(hidden)

    def forward(self, batch: AVESet) -> dict[str, torch.Tensor]:
        source_latent = self.encode(batch.source_tokens)
        target_latent = self.encode(batch.target_tokens)
        edited_latent = self.source_to_target(source_latent)
        inverse_latent = self.target_to_source(target_latent)
        return {
            "source_recon": self.decode(source_latent),
            "target_recon": self.decode(target_latent),
            "source_to_target": self.decode(edited_latent),
            "target_to_source": self.decode(inverse_latent),
            "answer": self.answer_head(target_latent.mean(dim=1)),
            "edited_latent": edited_latent,
            "target_latent": target_latent,
            "inverse_latent": inverse_latent,
            "source_latent": source_latent,
        }


def sequence_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, TOKEN_VOCAB), target.reshape(-1))


def latent_align_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a.flatten(1), b.detach().flatten(1), dim=-1).mean()


def checkpoint_path(config: StageAVEConfig, name: str) -> Path | None:
    if not config.checkpoint_dir:
        return None
    path = Path(config.checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def save_checkpoint(
    path: Path,
    model: BidirectionalLatentExternalModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: StageAVEConfig,
    step: int,
    history: list[dict[str, float]],
    best_score: float,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "stage": "AV-E",
            "config": asdict(config),
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_score": best_score,
        },
        path,
    )


def train_model(model: BidirectionalLatentExternalModel, train: AVESet, val: AVESet, config: StageAVEConfig, device: torch.device) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    rng = random.Random(config.seed + 77)
    history = []
    start_step = 1
    best_score = -1.0
    resumed_from = ""
    latest_path = checkpoint_path(config, "latest.pt")
    best_path = checkpoint_path(config, "best.pt")
    if config.resume and latest_path is not None and latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = list(state.get("history", []))
        best_score = float(state.get("best_score", best_score))
        start_step = int(state["step"]) + 1
        resumed_from = str(latest_path)
    for step in range(start_step, config.steps + 1):
        model.train()
        batch = random_batch(train, rng, config.batch_size, device)
        with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
            out = model(batch)
            loss = (
                0.5 * sequence_loss(out["source_recon"], batch.source_tokens)
                + 0.5 * sequence_loss(out["target_recon"], batch.target_tokens)
                + sequence_loss(out["source_to_target"], batch.target_tokens)
                + sequence_loss(out["target_to_source"], batch.source_tokens)
                + F.cross_entropy(out["answer"], batch.answer)
                + 0.2 * latent_align_loss(out["edited_latent"], out["target_latent"])
                + 0.2 * latent_align_loss(out["inverse_latent"], out["source_latent"])
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate(model, val, prefix="val", device=device, batch_size=config.eval_batch_size)
            metrics["step"] = float(step)
            metrics["loss"] = float(loss.detach().item())
            metrics["processed_examples"] = float(step * config.batch_size)
            metrics["processed_pair_tokens"] = float(step * config.batch_size * PAIR_TOKENS_PER_EXAMPLE)
            history.append(metrics)
            score = (
                metrics["val_source_recon_exact"]
                + metrics["val_target_recon_exact"]
                + metrics["val_source_to_target_exact"]
                + metrics["val_target_to_source_exact"]
                + metrics["val_answer_accuracy"]
            )
            if best_path is not None and score > best_score:
                best_score = score
                save_checkpoint(best_path, model, optimizer, scaler, config, step, history, best_score)
            print(json.dumps(metrics), flush=True)
        if latest_path is not None and (step % config.save_every == 0 or step == config.steps):
            save_checkpoint(latest_path, model, optimizer, scaler, config, step, history, best_score)
    return {
        "history": history,
        "start_step": start_step,
        "completed_steps": config.steps,
        "resumed_from": resumed_from,
        "checkpoint_dir": config.checkpoint_dir,
        "latest_checkpoint": str(latest_path) if latest_path is not None else "",
        "best_checkpoint": str(best_path) if best_path is not None else "",
        "best_score": best_score,
    }


@torch.no_grad()
def exact_sequence(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).all(dim=1).float().mean().item())


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).float().mean().item())


@torch.no_grad()
def evaluate(model: BidirectionalLatentExternalModel, data: AVESet, *, prefix: str, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    source_recon_hits = 0
    target_recon_hits = 0
    source_to_target_hits = 0
    target_to_source_hits = 0
    source_recon_exact = 0
    target_recon_exact = 0
    source_to_target_exact = 0
    target_to_source_exact = 0
    answer_hits = 0
    edit_cosine_sum = 0.0
    inverse_cosine_sum = 0.0
    token_total = 0
    example_total = 0
    for start in range(0, len(data), batch_size):
        batch = data.slice(start, min(start + batch_size, len(data))).to(device)
        out = model(batch)
        source_recon = out["source_recon"].argmax(dim=-1)
        target_recon = out["target_recon"].argmax(dim=-1)
        source_to_target = out["source_to_target"].argmax(dim=-1)
        target_to_source = out["target_to_source"].argmax(dim=-1)
        answer = out["answer"].argmax(dim=-1)
        source_recon_match = source_recon == batch.source_tokens
        target_recon_match = target_recon == batch.target_tokens
        source_to_target_match = source_to_target == batch.target_tokens
        target_to_source_match = target_to_source == batch.source_tokens
        source_recon_hits += int(source_recon_match.sum().item())
        target_recon_hits += int(target_recon_match.sum().item())
        source_to_target_hits += int(source_to_target_match.sum().item())
        target_to_source_hits += int(target_to_source_match.sum().item())
        source_recon_exact += int(source_recon_match.all(dim=1).sum().item())
        target_recon_exact += int(target_recon_match.all(dim=1).sum().item())
        source_to_target_exact += int(source_to_target_match.all(dim=1).sum().item())
        target_to_source_exact += int(target_to_source_match.all(dim=1).sum().item())
        answer_hits += int((answer == batch.answer).sum().item())
        edit_cosine_sum += float(F.cosine_similarity(out["edited_latent"].flatten(1), out["target_latent"].flatten(1), dim=-1).sum().item())
        inverse_cosine_sum += float(F.cosine_similarity(out["inverse_latent"].flatten(1), out["source_latent"].flatten(1), dim=-1).sum().item())
        token_total += int(batch.source_tokens.numel())
        example_total += len(batch)
    return {
        f"{prefix}_source_recon_token_accuracy": source_recon_hits / token_total,
        f"{prefix}_target_recon_token_accuracy": target_recon_hits / token_total,
        f"{prefix}_source_to_target_token_accuracy": source_to_target_hits / token_total,
        f"{prefix}_target_to_source_token_accuracy": target_to_source_hits / token_total,
        f"{prefix}_source_recon_exact": source_recon_exact / example_total,
        f"{prefix}_target_recon_exact": target_recon_exact / example_total,
        f"{prefix}_source_to_target_exact": source_to_target_exact / example_total,
        f"{prefix}_target_to_source_exact": target_to_source_exact / example_total,
        f"{prefix}_answer_accuracy": answer_hits / example_total,
        f"{prefix}_edit_latent_cosine": edit_cosine_sum / example_total,
        f"{prefix}_inverse_latent_cosine": inverse_cosine_sum / example_total,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def render_image(values: tuple[int, ...]) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.88, 0.10, 0.10],
            [0.10, 0.70, 0.25],
            [0.15, 0.32, 0.90],
            [0.94, 0.74, 0.12],
            [0.85, 0.20, 0.75],
            [0.10, 0.78, 0.78],
            [0.94, 0.52, 0.10],
            [0.78, 0.78, 0.78],
        ],
        dtype=torch.float32,
    )
    image = torch.full((3, 64, 96), 0.05)
    for zone, color in enumerate(values):
        x = 8 + (zone % 3) * 28
        y = 8 + (zone // 3) * 28
        image[:, y : y + 20, x : x + 20] = palette[color].view(3, 1, 1)
    return image


@torch.no_grad()
def write_samples(model: BidirectionalLatentExternalModel, data: AVESet, path: Path, device: torch.device, limit: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    if not data.examples:
        sample = data.slice(0, min(limit, len(data))).to(device)
        out = model(sample)
        source_to_target = out["source_to_target"].argmax(dim=-1)
        target_to_source = out["target_to_source"].argmax(dim=-1)
        answer = out["answer"].argmax(dim=-1)
        for index in range(len(sample)):
            rows.append(
                {
                    "case_id": f"manifest-{index:06d}",
                    "source_tokens": sample.source_tokens[index].detach().cpu().tolist(),
                    "target_tokens": sample.target_tokens[index].detach().cpu().tolist(),
                    "source_to_target_prediction": source_to_target[index].detach().cpu().tolist(),
                    "target_to_source_prediction": target_to_source[index].detach().cpu().tolist(),
                    "answer": int(sample.answer[index].item()),
                    "prediction": int(answer[index].item()),
                    "source_to_target_token_hit": float((source_to_target[index] == sample.target_tokens[index]).float().mean().item()),
                    "target_to_source_token_hit": float((target_to_source[index] == sample.source_tokens[index]).float().mean().item()),
                }
            )
        (path / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    for index, example in enumerate(data.examples[:limit]):
        source_png = path / f"{example.case_id}_source.png"
        target_png = path / f"{example.case_id}_target.png"
        write_png(source_png, render_image(example.source_image))
        write_png(target_png, render_image(example.target_image))
        single = data.subset([index]).to(device)
        out = model(single)
        rows.append(
            {
                "case_id": example.case_id,
                "source_png": str(source_png),
                "target_png": str(target_png),
                "op": example.op,
                "op_arg": example.op_arg,
                "target_zone": example.target_zone,
                "answer": example.answer,
                "prediction": int(out["answer"].argmax(dim=-1).item()),
                "source_to_target_token_hit": float((out["source_to_target"].argmax(dim=-1) == single.target_tokens).float().mean().item()),
                "target_to_source_token_hit": float((out["target_to_source"].argmax(dim=-1) == single.source_tokens).float().mean().item()),
            }
        )
    (path / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_datasets(config: StageAVEConfig) -> tuple[AVESet, AVESet, AVESet, AVESet, dict[str, object]]:
    if not config.dataset_manifest:
        train = build_split("train", config.train_size, config)
        val = build_split("val", config.val_size, config)
        test = build_split("test", config.test_size, config)
        heldout = build_split("heldout", config.test_size, config)
        dataset_info = {
            "source": "online",
            "train_examples": len(train),
            "val_examples": len(val),
            "test_examples": len(test),
            "heldout_examples": len(heldout),
            "unique_train_pair_tokens": len(train) * PAIR_TOKENS_PER_EXAMPLE,
        }
        return train, val, test, heldout, dataset_info

    manifest_path = Path(config.dataset_manifest)
    manifest = load_manifest(manifest_path)
    train = load_manifest_split(manifest_path, manifest, "train", config.train_limit)
    val = load_manifest_split(manifest_path, manifest, "val", config.val_limit)
    test = load_manifest_split(manifest_path, manifest, "test", config.test_limit)
    heldout = load_manifest_split(manifest_path, manifest, "heldout", config.heldout_limit)
    dataset_info = {
        "source": "manifest",
        "manifest": str(manifest_path),
        "manifest_stage": manifest.get("stage", ""),
        "manifest_scale": manifest.get("scale", ""),
        "manifest_total_examples": manifest.get("total_examples", 0),
        "manifest_total_tokens": manifest.get("total_tokens", 0),
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
        "heldout_examples": len(heldout),
        "unique_train_pair_tokens": len(train) * PAIR_TOKENS_PER_EXAMPLE,
        "tokens_per_example": PAIR_TOKENS_PER_EXAMPLE,
    }
    return train, val, test, heldout, dataset_info


def run_one(config: StageAVEConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train, val, test, heldout, dataset_info = load_datasets(config)
    if config.gpu_resident_data:
        train = train.to(device)
        val = val.to(device)
    model = BidirectionalLatentExternalModel(config)
    training = train_model(model, train, val, config, device)
    test_metrics = evaluate(model, test, prefix="test", device=device, batch_size=config.eval_batch_size)
    heldout_metrics = evaluate(model, heldout, prefix="heldout", device=device, batch_size=config.eval_batch_size)
    write_samples(model, test, output_path.parent / "samples", device, config.sample_count)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    result = {
        "schema_version": 1,
        "stage": "AV-E",
        "config": asdict(config),
        "dataset": dataset_info,
        "device": str(device),
        "metrics": {**test_metrics, **heldout_metrics},
        "cost": {
            "elapsed_sec": time.perf_counter() - started,
            "parameters": parameter_count(model),
            "peak_cuda_allocated_mb": peak_mb,
            "processed_pair_tokens": config.steps * config.batch_size * PAIR_TOKENS_PER_EXAMPLE,
        },
        "training": training,
        "interpretation": {
            "goal": "Train bidirectional external-latent translation plus one latent edit step.",
            "success_condition": "Source/target reconstruction and source-target/inverse translation should all improve together, not only answer classification.",
            "boundary": "This is still synthetic tokenized external state, not full real multimodal data.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": result["metrics"], "cost": result["cost"]}, ensure_ascii=False, indent=2), flush=True)
    return result


def collect_numbers(value: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            out.update(collect_numbers(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    rows = [collect_numbers(run) for run in runs]
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {key: statistics.mean(row[key] for row in rows if key in row) for key in keys}
    return {
        "schema_version": 1,
        "stage": "AV-E",
        "runs": runs,
        "summary": summary,
        "gates": {
            "source_to_target_token_accuracy": summary["metrics_test_source_to_target_token_accuracy"],
            "target_to_source_token_accuracy": summary["metrics_test_target_to_source_token_accuracy"],
            "answer_accuracy": summary["metrics_test_answer_accuracy"],
            "heldout_source_to_target_token_accuracy": summary["metrics_heldout_source_to_target_token_accuracy"],
            "passes_probe_70": summary["metrics_test_source_to_target_token_accuracy"] >= 0.70
            and summary["metrics_test_target_to_source_token_accuracy"] >= 0.70
            and summary["metrics_test_answer_accuracy"] >= 0.70,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260705")
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_ave_bidirectional_latent_external/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_ave_bidirectional_latent_external/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_ave_bidirectional_latent_external/sweep_results.json"))
    parser.add_argument("--train-size", type=int, default=StageAVEConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageAVEConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageAVEConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageAVEConfig.batch_size)
    parser.add_argument("--d-model", type=int, default=StageAVEConfig.d_model)
    parser.add_argument("--heads", type=int, default=StageAVEConfig.heads)
    parser.add_argument("--layers", type=int, default=StageAVEConfig.layers)
    parser.add_argument("--latent-tokens", type=int, default=StageAVEConfig.latent_tokens)
    parser.add_argument("--lr", type=float, default=StageAVEConfig.lr)
    parser.add_argument("--steps", type=int, default=StageAVEConfig.steps)
    parser.add_argument("--eval-every", type=int, default=StageAVEConfig.eval_every)
    parser.add_argument("--eval-batch-size", type=int, default=StageAVEConfig.eval_batch_size)
    parser.add_argument("--sample-count", type=int, default=StageAVEConfig.sample_count)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--train-limit", type=int, default=StageAVEConfig.train_limit)
    parser.add_argument("--val-limit", type=int, default=StageAVEConfig.val_limit)
    parser.add_argument("--test-limit", type=int, default=StageAVEConfig.test_limit)
    parser.add_argument("--heldout-limit", type=int, default=StageAVEConfig.heldout_limit)
    parser.add_argument("--gpu-resident-data", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=StageAVEConfig.save_every)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, seed: int) -> StageAVEConfig:
    return StageAVEConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=seed,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        latent_tokens=args.latent_tokens,
        lr=args.lr,
        steps=args.steps,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        sample_count=args.sample_count,
        amp=not args.no_amp,
        dataset_manifest=str(args.dataset_manifest) if args.dataset_manifest else "",
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        heldout_limit=args.heldout_limit,
        gpu_resident_data=args.gpu_resident_data,
        checkpoint_dir=str(args.checkpoint_dir) if args.checkpoint_dir else "",
        resume=args.resume,
        save_every=args.save_every,
    )


def main() -> None:
    args = parse_args()
    if args.sweep:
        runs = []
        for text in args.seeds.split(","):
            seed = int(text.strip())
            runs.append(run_one(config_from_args(args, seed), args.output_dir / f"seed{seed}.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"aggregate": aggregate["gates"]}, ensure_ascii=False, indent=2), flush=True)
    else:
        seed = int(args.seeds.split(",")[0].strip())
        run_one(config_from_args(args, seed), args.output)


if __name__ == "__main__":
    main()
