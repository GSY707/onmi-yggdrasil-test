from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

from multimodal_fusion_latent_flow import IMAGE_SIZE, stat
from omni_transformer_stage_h import PATCH_COUNT, PATCH_SIZE, TransformerBlock
from visual_multimodal_stage_ab import write_png


DATASET_NAME = "pixparse/docvqa-single-page-questions"
TEXT_VOCAB = 4096
QUESTION_LEN = 20
CANDIDATES = 8
CANDIDATE_LEN = 18
LATENT_TOKENS = 8
SPECIAL_BOS = 0
SPECIAL_IMG = 1
SPECIAL_QUESTION = 2
SPECIAL_OCR = 3
SPECIAL_ANSWER = 4
SPECIAL_COUNT = 5

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class DocVQAConfig:
    train_size: int = 1024
    val_size: int = 256
    test_size: int = 256
    batch_size: int = 64
    seed: int = 20260701
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    direct_steps: int = 450
    bottleneck_steps: int = 550
    probe_steps: int = 120
    eval_every: int = 275


@dataclass(frozen=True)
class Candidate:
    text: str
    bbox: tuple[float, float, float, float]
    line_index: int


@dataclass(frozen=True)
class Example:
    question_id: str
    question: str
    answers: tuple[str, ...]
    image: np.ndarray
    candidates: tuple[Candidate, ...]
    label: int
    question_type: str


@dataclass
class Batch:
    image: torch.Tensor
    question: torch.Tensor
    candidate_tokens: torch.Tensor
    candidate_features: torch.Tensor
    label: torch.Tensor
    question_type: torch.Tensor


@dataclass(frozen=True)
class SequenceLayout:
    input_end: int
    latent_start: int
    latent_end: int
    answer_pos: int
    seq_len: int


def normalize_text(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower()))


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


def token_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return 1 + (int.from_bytes(digest, "little") % (TEXT_VOCAB - 1))


def encode_tokens(value: str, length: int) -> list[int]:
    ids = [token_id(token) for token in tokenize(value)[:length]]
    return ids + [0] * (length - len(ids))


def answer_in_line(answers: Iterable[str], line_text: str) -> bool:
    line_norm = normalize_text(line_text)
    if not line_norm:
        return False
    for answer in answers:
        answer_norm = normalize_text(answer)
        if answer_norm and (answer_norm in line_norm or line_norm in answer_norm):
            return True
    return False


def line_overlap(question: str, line_text: str) -> int:
    stop = {"what", "is", "the", "in", "of", "a", "an", "to", "for", "this", "document", "page", "mentioned"}
    q = set(token for token in tokenize(question) if token not in stop)
    line = set(tokenize(line_text))
    return len(q & line)


def infer_question_type(question: str) -> str:
    tokens = tokenize(question)
    if not tokens:
        return "other"
    first = tokens[0]
    if first in {"what", "when", "who", "where", "which", "how"}:
        if first == "how" and len(tokens) > 1:
            return f"how_{tokens[1]}"
        return first
    return "other"


def extract_lines(ocr_results: dict[str, object]) -> list[Candidate]:
    raw_lines = ocr_results.get("lines", [])
    candidates: list[Candidate] = []
    for idx, line in enumerate(raw_lines if isinstance(raw_lines, list) else []):
        if not isinstance(line, dict):
            continue
        text = str(line.get("text", "")).strip()
        bbox = line.get("bounding_box", [])
        if not text or not isinstance(bbox, list) or len(bbox) < 8:
            continue
        xs = [float(bbox[pos]) for pos in range(0, min(len(bbox), 8), 2)]
        ys = [float(bbox[pos]) for pos in range(1, min(len(bbox), 8), 2)]
        width = float(ocr_results.get("width", 1) or 1)
        height = float(ocr_results.get("height", 1) or 1)
        norm_bbox = (
            max(0.0, min(xs) / width),
            max(0.0, min(ys) / height),
            min(1.0, max(xs) / width),
            min(1.0, max(ys) / height),
        )
        candidates.append(Candidate(text=text, bbox=norm_bbox, line_index=idx))
    return candidates


def pil_to_tensor(image: Image.Image) -> np.ndarray:
    resized = ImageOps.grayscale(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.stack([array, array, array], axis=0).astype(np.float32)


def make_example(raw: dict[str, object], *, rng: random.Random) -> Example | None:
    answers_raw = raw.get("answers", [])
    if not isinstance(answers_raw, list) or not answers_raw:
        return None
    answers = tuple(str(answer) for answer in answers_raw if str(answer).strip())
    if not answers:
        return None
    question = str(raw.get("question", "")).strip()
    if not question:
        return None
    image = raw.get("image")
    if not isinstance(image, Image.Image):
        return None
    ocr_results = raw.get("ocr_results", {})
    if not isinstance(ocr_results, dict):
        return None
    lines = extract_lines(ocr_results)
    answer_lines = [line for line in lines if answer_in_line(answers, line.text)]
    distractors = [line for line in lines if not answer_in_line(answers, line.text)]
    if not answer_lines or len(distractors) < CANDIDATES - 1:
        return None

    gold = rng.choice(answer_lines)
    hard = sorted(distractors, key=lambda line: line_overlap(question, line.text), reverse=True)
    top_hard = hard[: max(CANDIDATES * 2, CANDIDATES - 1)]
    selected = [gold] + rng.sample(top_hard, k=min(len(top_hard), CANDIDATES - 1))
    if len(selected) < CANDIDATES:
        remaining = [line for line in distractors if line not in selected]
        selected.extend(rng.sample(remaining, k=CANDIDATES - len(selected)))
    rng.shuffle(selected)
    label = selected.index(gold)
    question_types = raw.get("question_types", [])
    if isinstance(question_types, list) and question_types:
        question_type = str(question_types[0])
    else:
        question_type = infer_question_type(question)

    return Example(
        question_id=str(raw.get("question_id", raw.get("questionId", ""))),
        question=question,
        answers=answers,
        image=pil_to_tensor(image),
        candidates=tuple(selected),
        label=label,
        question_type=question_type,
    )


def collect_examples(split: str, size: int, *, seed: int, scan_limit: int, skip_usable: int = 0) -> list[Example]:
    rng = random.Random(seed)
    dataset = load_dataset(DATASET_NAME, split=split)
    examples: list[Example] = []
    scanned = 0
    for raw in dataset:
        scanned += 1
        example = make_example(raw, rng=rng)
        if example is not None:
            if skip_usable > 0:
                skip_usable -= 1
                continue
            examples.append(example)
            if len(examples) >= size:
                break
        if scanned >= scan_limit:
            break
    if len(examples) < size:
        raise RuntimeError(f"only collected {len(examples)} usable DocVQA examples from split={split}; requested {size}")
    return examples


def question_type_vocab(examples: list[Example]) -> dict[str, int]:
    names = sorted({example.question_type for example in examples})
    return {name: idx for idx, name in enumerate(names)}


def make_batch(examples: list[Example], *, qtype_to_id: dict[str, int], device: torch.device) -> Batch:
    image = torch.tensor(np.stack([example.image for example in examples]), dtype=torch.float32, device=device)
    question = torch.tensor([encode_tokens(example.question, QUESTION_LEN) for example in examples], dtype=torch.long, device=device)
    candidate_tokens = torch.tensor(
        [[encode_tokens(candidate.text, CANDIDATE_LEN) for candidate in example.candidates] for example in examples],
        dtype=torch.long,
        device=device,
    )
    features: list[list[list[float]]] = []
    for example in examples:
        rows = []
        denom = max(max(candidate.line_index for candidate in example.candidates), 1)
        for candidate in example.candidates:
            x1, y1, x2, y2 = candidate.bbox
            rows.append([x1, y1, x2, y2, candidate.line_index / denom])
        features.append(rows)
    candidate_features = torch.tensor(features, dtype=torch.float32, device=device)
    return Batch(
        image=image,
        question=question,
        candidate_tokens=candidate_tokens,
        candidate_features=candidate_features,
        label=torch.tensor([example.label for example in examples], dtype=torch.long, device=device),
        question_type=torch.tensor([qtype_to_id[example.question_type] for example in examples], dtype=torch.long, device=device),
    )


def random_batch(examples: list[Example], *, rng: random.Random, batch_size: int, qtype_to_id: dict[str, int], device: torch.device) -> Batch:
    return make_batch(rng.choices(examples, k=batch_size), qtype_to_id=qtype_to_id, device=device)


class DocVQATransformer(nn.Module):
    def __init__(self, config: DocVQAConfig, *, qtype_count: int) -> None:
        super().__init__()
        self.config = config
        self.qtype_count = qtype_count
        self.image_patch = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, config.d_model)
        self.text_embedding = nn.Embedding(TEXT_VOCAB, config.d_model)
        self.candidate_feature = nn.Linear(5, config.d_model)
        self.candidate_slot = nn.Embedding(CANDIDATES, config.d_model)
        self.special_embedding = nn.Embedding(SPECIAL_COUNT, config.d_model)
        self.latent_embedding = nn.Parameter(torch.randn(LATENT_TOKENS, config.d_model) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(128, config.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(config.layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.answer_head = nn.Linear(config.d_model, CANDIDATES)

    def layout(self) -> SequenceLayout:
        input_end = 1 + 1 + PATCH_COUNT + 1 + QUESTION_LEN + 1 + CANDIDATES
        latent_start = input_end
        latent_end = latent_start + LATENT_TOKENS
        answer_pos = latent_end
        return SequenceLayout(
            input_end=input_end,
            latent_start=latent_start,
            latent_end=latent_end,
            answer_pos=answer_pos,
            seq_len=answer_pos + 1,
        )

    def patch_image(self, image: torch.Tensor) -> torch.Tensor:
        patches = image.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(image.shape[0], PATCH_COUNT, 3 * PATCH_SIZE * PATCH_SIZE)

    def candidate_embeddings(self, batch: Batch) -> torch.Tensor:
        token_emb = self.text_embedding(batch.candidate_tokens).mean(dim=2)
        slot_ids = torch.arange(CANDIDATES, device=batch.label.device)
        return token_emb + self.candidate_feature(batch.candidate_features) + self.candidate_slot(slot_ids).unsqueeze(0)

    def build_embeddings(self, batch: Batch, *, zero_modalities: Iterable[str] = ()) -> tuple[torch.Tensor, SequenceLayout]:
        zero_set = set(zero_modalities)
        batch_size = int(batch.label.shape[0])
        special = self.special_embedding.weight
        image_tokens = self.image_patch(self.patch_image(batch.image))
        question_tokens = self.text_embedding(batch.question)
        candidate_tokens = self.candidate_embeddings(batch)
        if "image" in zero_set:
            image_tokens = torch.zeros_like(image_tokens)
        if "question" in zero_set:
            question_tokens = torch.zeros_like(question_tokens)
        if "ocr" in zero_set or "candidates" in zero_set:
            candidate_tokens = torch.zeros_like(candidate_tokens)

        chunks = [
            special[SPECIAL_BOS].expand(batch_size, 1, -1),
            special[SPECIAL_IMG].expand(batch_size, 1, -1),
            image_tokens,
            special[SPECIAL_QUESTION].expand(batch_size, 1, -1),
            question_tokens,
            special[SPECIAL_OCR].expand(batch_size, 1, -1),
            candidate_tokens,
            self.latent_embedding.expand(batch_size, LATENT_TOKENS, -1),
            special[SPECIAL_ANSWER].expand(batch_size, 1, -1),
        ]
        x = torch.cat(chunks, dim=1)
        x = x + self.position_embedding[: x.shape[1]].unsqueeze(0)
        return x, self.layout()

    def attention_mask(
        self,
        layout: SequenceLayout,
        *,
        mode: str,
        disable_latent_to_answer: bool,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.triu(torch.ones(layout.seq_len, layout.seq_len, dtype=torch.bool, device=device), diagonal=1)
        if mode == "latent_bottleneck":
            mask[layout.answer_pos, : layout.input_end] = True
            if disable_latent_to_answer:
                mask[layout.answer_pos, layout.latent_start : layout.latent_end] = True
        elif mode != "direct":
            raise ValueError(f"unknown mode: {mode}")
        return mask

    def forward(
        self,
        batch: Batch,
        *,
        mode: str,
        zero_modalities: Iterable[str] = (),
        disable_latent_to_answer: bool = False,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor | SequenceLayout]:
        x, layout = self.build_embeddings(batch, zero_modalities=zero_modalities)
        mask = self.attention_mask(
            layout,
            mode=mode,
            disable_latent_to_answer=disable_latent_to_answer,
            device=x.device,
        )
        for block in self.blocks:
            x = block(x, mask)
        hidden = self.norm(x)
        output: dict[str, torch.Tensor | SequenceLayout] = {
            "logits": self.answer_head(hidden[:, layout.answer_pos]),
            "layout": layout,
        }
        if return_hidden:
            output["hidden"] = hidden
        return output


@torch.no_grad()
def evaluate(
    model: DocVQATransformer,
    examples: list[Example],
    config: DocVQAConfig,
    *,
    qtype_to_id: dict[str, int],
    device: torch.device,
    mode: str,
    zero_modalities: Iterable[str] = (),
    disable_latent_to_answer: bool = False,
) -> dict[str, object]:
    model.eval()
    correct = 0
    total = 0
    by_type: dict[str, list[float]] = {}
    for start in range(0, len(examples), config.batch_size):
        chunk = examples[start : start + config.batch_size]
        batch = make_batch(chunk, qtype_to_id=qtype_to_id, device=device)
        output = model(
            batch,
            mode=mode,
            zero_modalities=zero_modalities,
            disable_latent_to_answer=disable_latent_to_answer,
        )
        pred = output["logits"].argmax(dim=-1)
        hits = (pred == batch.label).detach().cpu().tolist()
        correct += sum(1 for hit in hits if hit)
        total += len(hits)
        for example, hit in zip(chunk, hits):
            by_type.setdefault(example.question_type, []).append(1.0 if hit else 0.0)
    model.train()
    return {
        "candidate_line_accuracy": correct / total,
        "by_question_type": {key: statistics.fmean(values) for key, values in sorted(by_type.items())},
    }


def train_model(
    model: DocVQATransformer,
    train_examples: list[Example],
    val_examples: list[Example],
    config: DocVQAConfig,
    *,
    steps: int,
    seed: int,
    qtype_to_id: dict[str, int],
    device: torch.device,
    mode: str,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_batch(train_examples, rng=rng, batch_size=config.batch_size, qtype_to_id=qtype_to_id, device=device)
        output = model(batch, mode=mode)
        loss = F.cross_entropy(output["logits"], batch.label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate(model, val_examples, config, qtype_to_id=qtype_to_id, device=device, mode=mode)
            acc = float(metrics["candidate_line_accuracy"])
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "candidate_line_accuracy": acc})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} acc={acc:.3f}", flush=True)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def random_baseline(examples: list[Example]) -> dict[str, float]:
    return {"candidate_line_accuracy": 1.0 / CANDIDATES}


def lexical_overlap_baseline(examples: list[Example]) -> dict[str, float]:
    correct = 0
    for example in examples:
        scores = [line_overlap(example.question, candidate.text) for candidate in example.candidates]
        pred = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
        correct += int(pred == example.label)
    return {"candidate_line_accuracy": correct / len(examples)}


class LatentProbe(nn.Module):
    def __init__(self, d_model: int, qtype_count: int) -> None:
        super().__init__()
        self.label = nn.Linear(d_model, CANDIDATES)
        self.qtype = nn.Linear(d_model, qtype_count)

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"label": self.label(latent), "qtype": self.qtype(latent)}


def train_latent_probe(
    model: DocVQATransformer,
    train_examples: list[Example],
    test_examples: list[Example],
    config: DocVQAConfig,
    *,
    qtype_to_id: dict[str, int],
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    probe = LatentProbe(config.d_model, len(qtype_to_id)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr)
    rng = random.Random(config.seed + 90000)
    started = time.perf_counter()
    for _ in range(config.probe_steps):
        batch = random_batch(train_examples, rng=rng, batch_size=config.batch_size, qtype_to_id=qtype_to_id, device=device)
        with torch.no_grad():
            output = model(batch, mode="latent_bottleneck", return_hidden=True)
            layout = output["layout"]
            hidden = output["hidden"]
            latent = hidden[:, layout.latent_start : layout.latent_end].mean(dim=1)
        logits = probe(latent.detach())
        loss = F.cross_entropy(logits["label"], batch.label) + F.cross_entropy(logits["qtype"], batch.question_type)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    totals = {"label": 0.0, "question_type": 0.0}
    count = 0
    with torch.no_grad():
        for start in range(0, len(test_examples), config.batch_size):
            batch = make_batch(test_examples[start : start + config.batch_size], qtype_to_id=qtype_to_id, device=device)
            output = model(batch, mode="latent_bottleneck", return_hidden=True)
            layout = output["layout"]
            hidden = output["hidden"]
            latent = hidden[:, layout.latent_start : layout.latent_end].mean(dim=1)
            logits = probe(latent)
            totals["label"] += float((logits["label"].argmax(dim=-1) == batch.label).sum().detach().cpu())
            totals["question_type"] += float((logits["qtype"].argmax(dim=-1) == batch.question_type).sum().detach().cpu())
            count += len(batch.label)
    return {
        "training_seconds": round(time.perf_counter() - started, 3),
        "accuracies": {key: value / count for key, value in totals.items()},
    }


def write_samples(examples: list[Example], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = [torch.tensor(example.image, dtype=torch.float32) for example in examples[:8]]
    first_row = torch.cat(tensors[:4], dim=2)
    second_row = torch.cat(tensors[4:8], dim=2)
    write_png(output_dir / "docvqa_thumbnail_grid.png", torch.cat([first_row, second_row], dim=1))
    preview = []
    for example in examples[:8]:
        preview.append(
            {
                "question_id": example.question_id,
                "question": example.question,
                "answers": list(example.answers),
                "label": example.label,
                "gold_candidate": example.candidates[example.label].text,
                "candidates": [candidate.text for candidate in example.candidates],
            }
        )
    (output_dir / "sample_candidates.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")


def run_experiment(config: DocVQAConfig, output_path: Path, *, scan_limit: int) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_l_docvqa device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_examples = collect_examples("train", config.train_size, seed=config.seed + 1, scan_limit=scan_limit)
    val_examples = collect_examples("validation", config.val_size, seed=config.seed + 2, scan_limit=scan_limit)
    test_examples = collect_examples(
        "validation",
        config.test_size,
        seed=config.seed + 3,
        scan_limit=scan_limit + config.val_size,
        skip_usable=config.val_size,
    )
    qtype_to_id = question_type_vocab(train_examples + val_examples + test_examples)
    write_samples(test_examples, output_dir=output_path.parent / "samples" / f"seed{config.seed}")

    direct = DocVQATransformer(config, qtype_count=len(qtype_to_id))
    bottleneck = DocVQATransformer(config, qtype_count=len(qtype_to_id))
    training = {
        "direct_docvqa": train_model(
            direct,
            train_examples,
            val_examples,
            config,
            steps=config.direct_steps,
            seed=config.seed + 10,
            qtype_to_id=qtype_to_id,
            device=device,
            mode="direct",
            label="direct_docvqa",
        ),
        "latent_bottleneck_docvqa": train_model(
            bottleneck,
            train_examples,
            val_examples,
            config,
            steps=config.bottleneck_steps,
            seed=config.seed + 20,
            qtype_to_id=qtype_to_id,
            device=device,
            mode="latent_bottleneck",
            label="latent_docvqa",
        ),
    }
    metrics = {
        "random_candidate": random_baseline(test_examples),
        "lexical_overlap": lexical_overlap_baseline(test_examples),
        "direct_docvqa": evaluate(direct, test_examples, config, qtype_to_id=qtype_to_id, device=device, mode="direct"),
        "latent_bottleneck_docvqa": evaluate(
            bottleneck,
            test_examples,
            config,
            qtype_to_id=qtype_to_id,
            device=device,
            mode="latent_bottleneck",
        ),
    }
    ablations = {
        "no_image": evaluate(
            bottleneck,
            test_examples,
            config,
            qtype_to_id=qtype_to_id,
            device=device,
            mode="latent_bottleneck",
            zero_modalities=("image",),
        ),
        "no_question": evaluate(
            bottleneck,
            test_examples,
            config,
            qtype_to_id=qtype_to_id,
            device=device,
            mode="latent_bottleneck",
            zero_modalities=("question",),
        ),
        "no_ocr_candidates": evaluate(
            bottleneck,
            test_examples,
            config,
            qtype_to_id=qtype_to_id,
            device=device,
            mode="latent_bottleneck",
            zero_modalities=("ocr",),
        ),
        "no_latent_access": evaluate(
            bottleneck,
            test_examples,
            config,
            qtype_to_id=qtype_to_id,
            device=device,
            mode="latent_bottleneck",
            disable_latent_to_answer=True,
        ),
    }
    latent_probe = train_latent_probe(bottleneck, train_examples, test_examples, config, qtype_to_id=qtype_to_id, device=device)
    output = {
        "experiment": "omni_transformer_stage_l_docvqa",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "name": DATASET_NAME,
            "task": "candidate OCR line selection from real DocVQA images/questions/OCR",
            "candidate_count": CANDIDATES,
        },
        "config": asdict(config),
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "qtype_to_id": qtype_to_id,
        "metrics": metrics,
        "ablations": ablations,
        "latent_probe": latent_probe,
        "training": training,
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
        numbers = collect_numbers({"metrics": run["metrics"], "ablations": run["ablations"], "latent_probe": run["latent_probe"]})
        for key, value in numbers.items():
            if "training_seconds" not in key:
                buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_l_docvqa_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_l_docvqa/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_l_docvqa/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_l_docvqa/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--scan-limit", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--direct-steps", type=int, default=450)
    parser.add_argument("--bottleneck-steps", type=int, default=550)
    parser.add_argument("--probe-steps", type=int, default=120)
    args = parser.parse_args()

    base_config = DocVQAConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        direct_steps=args.direct_steps,
        bottleneck_steps=args.bottleneck_steps,
        probe_steps=args.probe_steps,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = DocVQAConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json", scan_limit=args.scan_limit))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output, scan_limit=args.scan_limit)


if __name__ == "__main__":
    main()
