from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import string
import sys
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omni_transformer_stage_h import TransformerBlock
import omni_transformer_stage_m_moe_multimodal_llm as stage_m


DATASET_REPO = "liuhaotian/LLaVA-Instruct-150K"
DATASET_FILES = ("detail_23k.json", "conversation_58k.json", "complex_reasoning_77k.json")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
COCO_SPLITS = ("train2017", "val2017")
IMAGE_TOKEN_DIM = 768
TEXT_TOKEN_DIM = 512
CLIP_EMBED_DIM = 512
EXPERTS = ("vision", "prompt_text", "dialogue_history", "image_text_fusion")
VISION_EXPERT = 0
PROMPT_EXPERT = 1
HISTORY_EXPERT = 2
FUSION_EXPERT = 3

CHARSET = (
    string.ascii_lowercase
    + string.digits
    + " .,!?;:'\"-/()[]%$&+\n"
)
PAD = 0
BOS = 1
EOS = 2
CHAR_TO_ID = {char: index + 3 for index, char in enumerate(CHARSET)}
ID_TO_CHAR = {index: char for char, index in CHAR_TO_ID.items()}
VOCAB = len(CHAR_TO_ID) + 3


@dataclass(frozen=True)
class StagePConfig:
    train_size: int = 512
    val_size: int = 128
    test_size: int = 256
    batch_size: int = 16
    seed: int = 20260701
    clip_model_name: str = CLIP_MODEL_NAME
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    lr: float = 7e-4
    answer_len: int = 192
    latent_tokens: int = 16
    candidate_count: int = 8
    text_only_steps: int = 120
    direct_steps: int = 160
    mean_steps: int = 160
    moe_steps: int = 220
    scorer_steps: int = 240
    eval_every: int = 80
    router_loss_weight: float = 0.10
    max_answer_chars: int = 360
    max_prompt_chars: int = 700
    annotation_files: tuple[str, ...] = DATASET_FILES
    data_cache_dir: str = "artifacts/omni_transformer_stage_p_llava_moe/data"


@dataclass(frozen=True)
class LLaVAExample:
    id: str
    image_name: str
    image_path: str
    prompt: str
    answer: str
    has_history: bool
    source_file: str
    turn_index: int


@dataclass
class FeatureSet:
    examples: list[LLaVAExample]
    image_tokens: torch.Tensor
    text_tokens: torch.Tensor
    text_mask: torch.Tensor
    image_embed: torch.Tensor
    text_embed: torch.Tensor
    answer_input: torch.Tensor
    answer_target: torch.Tensor
    required_experts: torch.Tensor

    def subset(self, indices: list[int]) -> "FeatureSet":
        return FeatureSet(
            examples=[self.examples[index] for index in indices],
            image_tokens=self.image_tokens[indices],
            text_tokens=self.text_tokens[indices],
            text_mask=self.text_mask[indices],
            image_embed=self.image_embed[indices],
            text_embed=self.text_embed[indices],
            answer_input=self.answer_input[indices],
            answer_target=self.answer_target[indices],
            required_experts=self.required_experts[indices],
        )

    def to(self, device: torch.device) -> "FeatureSet":
        return FeatureSet(
            examples=self.examples,
            image_tokens=self.image_tokens.to(device=device, dtype=torch.float32),
            text_tokens=self.text_tokens.to(device=device, dtype=torch.float32),
            text_mask=self.text_mask.to(device=device),
            image_embed=self.image_embed.to(device=device, dtype=torch.float32),
            text_embed=self.text_embed.to(device=device, dtype=torch.float32),
            answer_input=self.answer_input.to(device=device),
            answer_target=self.answer_target.to(device=device),
            required_experts=self.required_experts.to(device=device, dtype=torch.float32),
        )

    def repeat_with_answers(self, candidates: list[list[str]], answer_len: int, device: torch.device) -> tuple["FeatureSet", torch.Tensor]:
        flat_answers: list[str] = []
        indices: list[int] = []
        for row_index, row in enumerate(candidates):
            for answer in row:
                indices.append(row_index)
                flat_answers.append(answer)
        repeated = self.subset(indices).to(device)
        targets = torch.tensor([encode_text(answer, answer_len, add_eos=True) for answer in flat_answers], dtype=torch.long, device=device)
        repeated.answer_target = targets
        repeated.answer_input = torch.tensor([answer_inputs(row.tolist()) for row in targets], dtype=torch.long, device=device)
        return repeated, targets


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def encode_text(value: str, length: int, *, add_eos: bool) -> list[int]:
    normalized = value.lower()
    ids = [CHAR_TO_ID[char] for char in normalized if char in CHAR_TO_ID]
    if add_eos:
        ids = ids[: length - 1] + [EOS]
    else:
        ids = ids[:length]
    return ids + [PAD] * (length - len(ids))


def answer_inputs(answer_target: list[int]) -> list[int]:
    return [BOS] + answer_target[:-1]


def decode_tokens(tokens: Iterable[int]) -> str:
    chars: list[str] = []
    for token in tokens:
        token = int(token)
        if token in (PAD, BOS):
            continue
        if token == EOS:
            break
        chars.append(ID_TO_CHAR.get(token, ""))
    return "".join(chars).strip()


def clean_text(value: str, max_chars: int) -> str:
    value = value.replace("<image>", " ")
    value = " ".join(value.replace("\r", "\n").split())
    return value[:max_chars].strip()


def normalize_answer(value: str, max_chars: int) -> str:
    value = clean_text(value, max_chars)
    if not value:
        return ""
    return value


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def cache_key(config: StagePConfig, split_name: str) -> str:
    payload = json.dumps(
        {
            "split": split_name,
            "seed": config.seed,
            "sizes": (config.train_size, config.val_size, config.test_size),
            "clip": config.clip_model_name,
            "answer_len": config.answer_len,
            "files": config.annotation_files,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def download_annotations(config: StagePConfig, cache_dir: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    annotation_dir = cache_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename in config.annotation_files:
        source = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=filename,
            local_dir=annotation_dir,
        )
        paths.append(Path(source))
    return paths


def parse_annotations(paths: list[Path], config: StagePConfig) -> list[LLaVAExample]:
    parsed: list[LLaVAExample] = []
    for path in paths:
        records = json.loads(path.read_text(encoding="utf-8"))
        for record_index, record in enumerate(records):
            image_name = record.get("image")
            conversations = record.get("conversations") or []
            if not image_name or not conversations:
                continue
            history: list[str] = []
            turn = 0
            cursor = 0
            while cursor + 1 < len(conversations):
                human = conversations[cursor]
                assistant = conversations[cursor + 1]
                if human.get("from") != "human" or assistant.get("from") != "gpt":
                    cursor += 1
                    continue
                prompt = clean_text(human.get("value", ""), config.max_prompt_chars)
                answer = normalize_answer(assistant.get("value", ""), config.max_answer_chars)
                if len(prompt) < 4 or len(answer) < 8:
                    cursor += 2
                    continue
                if history:
                    prompt = clean_text(" ".join(history[-4:] + [f"Human: {prompt}"]), config.max_prompt_chars)
                parsed.append(
                    LLaVAExample(
                        id=f"{path.stem}:{record.get('id', record_index)}:{turn}",
                        image_name=image_name,
                        image_path="",
                        prompt=prompt,
                        answer=answer,
                        has_history=bool(history),
                        source_file=path.name,
                        turn_index=turn,
                    )
                )
                history.extend([f"Human: {prompt}", f"Assistant: {answer}"])
                turn += 1
                cursor += 2
    return parsed


def coco_candidate_urls(image_name: str) -> list[str]:
    urls = [f"http://images.cocodataset.org/{split}/{image_name}" for split in COCO_SPLITS]
    if image_name.startswith("COCO_"):
        return urls
    stem = Path(image_name).stem
    if stem.isdigit():
        urls.extend(
            [
                f"http://images.cocodataset.org/train2014/COCO_train2014_{stem}.jpg",
                f"http://images.cocodataset.org/val2014/COCO_val2014_{stem}.jpg",
            ]
        )
    return urls


def download_one_image(image_name: str, image_dir: Path, timeout: float = 8.0) -> Path | None:
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / safe_name(image_name)
    if target.exists():
        return target
    last_error: Exception | None = None
    for url in coco_candidate_urls(image_name):
        try:
            request = Request(url, headers={"User-Agent": "stage-p-llava-moe/1.0"})
            with urlopen(request, timeout=timeout) as response:
                data = response.read()
            target.write_bytes(data)
            with Image.open(target) as image:
                image.verify()
            return target
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if target.exists():
                target.unlink()
    if last_error:
        return None
    return None


def split_examples(all_examples: list[LLaVAExample], config: StagePConfig, cache_dir: Path) -> tuple[dict[str, list[LLaVAExample]], dict[str, object]]:
    rng = random.Random(config.seed)
    groups: dict[str, list[LLaVAExample]] = {}
    for example in all_examples:
        groups.setdefault(example.image_name, []).append(example)
    image_names = list(groups)
    rng.shuffle(image_names)
    image_dir = cache_dir / "coco_images"
    wanted = {"train": config.train_size, "val": config.val_size, "test": config.test_size}
    splits: dict[str, list[LLaVAExample]] = {key: [] for key in wanted}
    failures = 0
    accepted_images = 0
    for image_name in image_names:
        split_name = next((key for key, size in wanted.items() if len(splits[key]) < size), None)
        if split_name is None:
            break
        if (accepted_images + failures) % 50 == 0:
            print(
                "image collection "
                + " ".join(f"{key}={len(value)}/{wanted[key]}" for key, value in splits.items())
                + f" failures={failures}",
                flush=True,
            )
        image_path = download_one_image(image_name, image_dir)
        if image_path is None:
            failures += 1
            continue
        accepted_images += 1
        group = groups[image_name]
        rng.shuffle(group)
        remaining = wanted[split_name] - len(splits[split_name])
        for example in group[:remaining]:
            splits[split_name].append(
                LLaVAExample(
                    id=example.id,
                    image_name=example.image_name,
                    image_path=str(image_path),
                    prompt=example.prompt,
                    answer=example.answer,
                    has_history=example.has_history,
                    source_file=example.source_file,
                    turn_index=example.turn_index,
                )
            )
    for split_name, size in wanted.items():
        if len(splits[split_name]) < size:
            raise RuntimeError(f"only collected {len(splits[split_name])}/{size} examples for {split_name}")
    return splits, {
        "parsed_examples": len(all_examples),
        "unique_images": len(groups),
        "accepted_images": accepted_images,
        "download_failures": failures,
        "split_sizes": {key: len(value) for key, value in splits.items()},
    }


def required_experts(example: LLaVAExample) -> tuple[float, ...]:
    return (
        1.0,
        1.0,
        1.0 if example.has_history else 0.0,
        1.0,
    )


class FrozenCLIPTokenExperts:
    def __init__(self, model_name: str, *, device: torch.device) -> None:
        self.model_name = model_name
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    @torch.no_grad()
    def encode_examples(self, examples: list[LLaVAExample], config: StagePConfig, *, batch_size: int) -> tuple[FeatureSet, dict[str, float]]:
        image_tokens: list[torch.Tensor] = []
        text_tokens: list[torch.Tensor] = []
        text_masks: list[torch.Tensor] = []
        image_embeds: list[torch.Tensor] = []
        text_embeds: list[torch.Tensor] = []
        started = time.perf_counter()
        cuda_sync(self.device)
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            images = [Image.open(example.image_path).convert("RGB") for example in chunk]
            inputs = self.processor(
                text=[example.prompt for example in chunk],
                images=images,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            vision = self.model.vision_model(pixel_values=inputs["pixel_values"])
            text = self.model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            image_features = self.model.visual_projection(vision.pooler_output)
            text_features = self.model.text_projection(text.pooler_output)
            image_tokens.append(vision.last_hidden_state.detach().cpu().to(torch.float16))
            text_tokens.append(text.last_hidden_state.detach().cpu().to(torch.float16))
            text_masks.append(inputs["attention_mask"].detach().cpu().to(torch.bool))
            image_embeds.append(F.normalize(image_features, dim=-1).detach().cpu().to(torch.float16))
            text_embeds.append(F.normalize(text_features, dim=-1).detach().cpu().to(torch.float16))
            for image in images:
                image.close()
        cuda_sync(self.device)
        seconds = time.perf_counter() - started
        answers = [encode_text(example.answer, config.answer_len, add_eos=True) for example in examples]
        return (
            FeatureSet(
                examples=examples,
                image_tokens=torch.cat(image_tokens, dim=0),
                text_tokens=torch.cat(text_tokens, dim=0),
                text_mask=torch.cat(text_masks, dim=0),
                image_embed=torch.cat(image_embeds, dim=0),
                text_embed=torch.cat(text_embeds, dim=0),
                answer_input=torch.tensor([answer_inputs(answer) for answer in answers], dtype=torch.long),
                answer_target=torch.tensor(answers, dtype=torch.long),
                required_experts=torch.tensor([required_experts(example) for example in examples], dtype=torch.float32),
            ),
            {
                "encoding_seconds": round(seconds, 4),
                "encoding_ms_per_example": round(seconds * 1000.0 / max(len(examples), 1), 4),
            },
        )

    @torch.no_grad()
    def encode_text_features(self, texts: list[str], *, batch_size: int) -> tuple[torch.Tensor, dict[str, float]]:
        rows: list[torch.Tensor] = []
        started = time.perf_counter()
        cuda_sync(self.device)
        for start in range(0, len(texts), batch_size):
            inputs = self.processor(text=texts[start : start + batch_size], padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            features = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            rows.append(F.normalize(features, dim=-1).detach().cpu())
        cuda_sync(self.device)
        seconds = time.perf_counter() - started
        return torch.cat(rows, dim=0), {
            "encoding_seconds": round(seconds, 4),
            "encoding_ms_per_text": round(seconds * 1000.0 / max(len(texts), 1), 4),
        }


def save_feature_set(path: Path, data: FeatureSet, cost: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "examples": [asdict(example) for example in data.examples],
            "image_tokens": data.image_tokens,
            "text_tokens": data.text_tokens,
            "text_mask": data.text_mask,
            "image_embed": data.image_embed,
            "text_embed": data.text_embed,
            "answer_input": data.answer_input,
            "answer_target": data.answer_target,
            "required_experts": data.required_experts,
            "cost": cost,
        },
        path,
    )


def load_feature_set(path: Path) -> tuple[FeatureSet, dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    examples = [LLaVAExample(**item) for item in payload["examples"]]
    return (
        FeatureSet(
            examples=examples,
            image_tokens=payload["image_tokens"],
            text_tokens=payload["text_tokens"],
            text_mask=payload["text_mask"],
            image_embed=payload["image_embed"],
            text_embed=payload["text_embed"],
            answer_input=payload["answer_input"],
            answer_target=payload["answer_target"],
            required_experts=payload["required_experts"],
        ),
        payload["cost"],
    )


def load_or_encode_features(
    splits: dict[str, list[LLaVAExample]],
    config: StagePConfig,
    cache_dir: Path,
    expert: FrozenCLIPTokenExperts,
    *,
    device: torch.device,
) -> tuple[dict[str, FeatureSet], dict[str, dict[str, float]]]:
    features: dict[str, FeatureSet] = {}
    costs: dict[str, dict[str, float]] = {}
    feature_dir = cache_dir / "features"
    for split_name, examples in splits.items():
        path = feature_dir / f"{split_name}_{cache_key(config, split_name)}.pt"
        if path.exists():
            features[split_name], costs[split_name] = load_feature_set(path)
            continue
        encoded, cost = expert.encode_examples(examples, config, batch_size=config.batch_size)
        save_feature_set(path, encoded, cost)
        features[split_name] = encoded
        costs[split_name] = cost
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return features, costs


def random_batch(data: FeatureSet, *, rng: random.Random, batch_size: int, device: torch.device) -> FeatureSet:
    return data.subset([rng.randrange(len(data.examples)) for _ in range(batch_size)]).to(device)


def noncausal_mask(length: int, *, device: torch.device) -> torch.Tensor:
    return torch.zeros(length, length, dtype=torch.bool, device=device)


class QueryResampler(nn.Module):
    def __init__(self, d_model: int, heads: int, query_count: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(query_count, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(self, source: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        query = self.query.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attn(
            self.norm_q(query),
            self.norm_kv(source),
            self.norm_kv(source),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = query + attended
        return x + self.ff(x)


class LLaVAExpertBank(nn.Module):
    def __init__(self, config: StagePConfig) -> None:
        super().__init__()
        self.config = config
        self.image_proj = nn.Sequential(nn.LayerNorm(IMAGE_TOKEN_DIM), nn.Linear(IMAGE_TOKEN_DIM, config.d_model), nn.GELU())
        self.text_proj = nn.Sequential(nn.LayerNorm(TEXT_TOKEN_DIM), nn.Linear(TEXT_TOKEN_DIM, config.d_model), nn.GELU())
        self.history_proj = nn.Sequential(nn.LayerNorm(TEXT_TOKEN_DIM), nn.Linear(TEXT_TOKEN_DIM, config.d_model), nn.GELU())
        self.fusion_resampler = QueryResampler(config.d_model, config.heads, config.latent_tokens, config.dropout)
        self.expert_type = nn.Embedding(len(EXPERTS), config.d_model)

    def forward(self, batch: FeatureSet, *, zero_modalities: Iterable[str] = ()) -> list[torch.Tensor]:
        zero_set = set(zero_modalities)
        image_source = torch.zeros_like(batch.image_tokens) if "image" in zero_set else batch.image_tokens
        text_source = torch.zeros_like(batch.text_tokens) if "text" in zero_set else batch.text_tokens
        image = self.image_proj(image_source) + self.expert_type.weight[VISION_EXPERT].view(1, 1, -1)
        text = self.text_proj(text_source) + self.expert_type.weight[PROMPT_EXPERT].view(1, 1, -1)
        history = self.history_proj(text_source) + self.expert_type.weight[HISTORY_EXPERT].view(1, 1, -1)
        fused_source = torch.cat((image, text), dim=1)
        fusion = self.fusion_resampler(fused_source) + self.expert_type.weight[FUSION_EXPERT].view(1, 1, -1)
        return [image, text, history, fusion]


class AttentionPump(nn.Module):
    def __init__(self, config: StagePConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(config.d_model, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(config.d_model)
        self.norm_kv = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model * 4), nn.GELU(), nn.Linear(config.d_model * 4, config.d_model))

    def forward(self, expert_tokens: list[torch.Tensor], route_weights: torch.Tensor) -> torch.Tensor:
        weighted = [tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)]
        source = torch.cat(weighted, dim=1)
        query = self.query.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attn(self.norm_q(query), self.norm_kv(source), self.norm_kv(source), need_weights=False)
        x = query + attended
        return x + self.ff(x)


class MeanPoolLatent(nn.Module):
    def __init__(self, config: StagePConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        self.norm = nn.LayerNorm(config.d_model)
        self.proj = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model))

    def forward(self, expert_tokens: list[torch.Tensor], route_weights: torch.Tensor) -> torch.Tensor:
        weighted = [tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)]
        source = torch.cat(weighted, dim=1)
        pooled = self.norm(source.mean(dim=1, keepdim=True))
        return self.query.unsqueeze(0) + self.proj(pooled)


class ThoughtExpert(nn.Module):
    def __init__(self, config: StagePConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        mask = noncausal_mask(latent.shape[1], device=latent.device)
        for block in self.blocks:
            latent = block(latent, mask)
        return self.norm(latent)


class TextOutputExpert(nn.Module):
    def __init__(self, config: StagePConfig, max_context_tokens: int) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(VOCAB, config.d_model)
        self.special = nn.Embedding(4, config.d_model)
        self.position = nn.Parameter(torch.randn(max_context_tokens + config.answer_len + 8, config.d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(config.d_model, config.heads, config.dropout) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, VOCAB)

    def mask(self, context_len: int, answer_len: int, *, device: torch.device) -> torch.Tensor:
        seq_len = context_len + answer_len
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        mask[:context_len, context_len:] = True
        return mask

    def forward(self, context: torch.Tensor, answer_input: torch.Tensor, *, marker: int) -> torch.Tensor:
        batch_size = answer_input.shape[0]
        start = self.special.weight[marker].view(1, 1, -1).expand(batch_size, 1, -1)
        answer_prefix = self.embedding(answer_input)
        x = torch.cat((start, context, self.special.weight[3].view(1, 1, -1).expand(batch_size, 1, -1), answer_prefix), dim=1)
        context_len = 1 + context.shape[1] + 1
        if x.shape[1] > self.position.shape[0]:
            raise RuntimeError(f"sequence length {x.shape[1]} exceeds position table {self.position.shape[0]}")
        x = x + self.position[: x.shape[1]].unsqueeze(0)
        mask = self.mask(context_len, self.config.answer_len, device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x)[:, context_len:])


class StagePDecoder(nn.Module):
    def __init__(self, config: StagePConfig, *, mode: str, max_direct_context: int = 220) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.bank = LLaVAExpertBank(config)
        self.router = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM * 2),
            nn.Linear(CLIP_EMBED_DIM * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(EXPERTS)),
        )
        self.pump = AttentionPump(config)
        self.mean_pool = MeanPoolLatent(config)
        self.thought = ThoughtExpert(config)
        max_context = max_direct_context if mode in ("direct", "text_only") else config.latent_tokens
        self.output = TextOutputExpert(config, max_context_tokens=max_context)

    def context(
        self,
        batch: FeatureSet,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        expert_tokens = self.bank(batch, zero_modalities=zero_modalities)
        route_logits = self.router(torch.cat((batch.image_embed, batch.text_embed), dim=-1))
        route_weights = torch.sigmoid(route_logits)
        if self.mode == "text_only":
            context = expert_tokens[PROMPT_EXPERT]
            latent = torch.zeros(batch.text_tokens.shape[0], self.config.latent_tokens, self.config.d_model, device=batch.text_tokens.device)
        elif self.mode == "direct":
            context = torch.cat([tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)], dim=1)
            latent = torch.zeros(batch.text_tokens.shape[0], self.config.latent_tokens, self.config.d_model, device=batch.text_tokens.device)
        elif self.mode == "mean_latent":
            latent = self.thought(self.mean_pool(expert_tokens, route_weights))
            context = torch.zeros_like(latent) if disable_latent_access else latent
        elif self.mode == "moe_latent":
            latent = self.thought(self.pump(expert_tokens, route_weights))
            context = torch.zeros_like(latent) if disable_latent_access else latent
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return {"context": context, "latent": latent, "route_logits": route_logits, "route_weights": route_weights}

    def forward(
        self,
        batch: FeatureSet,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(batch, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        marker = {"text_only": 0, "direct": 1, "mean_latent": 2, "moe_latent": 2}[self.mode]
        logits = self.output(ctx["context"], batch.answer_input, marker=marker)
        return {"logits": logits, **ctx}


class StagePScorer(nn.Module):
    def __init__(self, config: StagePConfig, *, mode: str) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.bank = LLaVAExpertBank(config)
        self.router = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM * 2),
            nn.Linear(CLIP_EMBED_DIM * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(EXPERTS)),
        )
        self.pump = AttentionPump(config)
        self.mean_pool = MeanPoolLatent(config)
        self.thought = ThoughtExpert(config)
        self.context_proj = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, config.d_model), nn.GELU())
        self.answer_proj = nn.Sequential(nn.LayerNorm(CLIP_EMBED_DIM), nn.Linear(CLIP_EMBED_DIM, config.d_model), nn.GELU())
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def context(
        self,
        batch: FeatureSet,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        expert_tokens = self.bank(batch, zero_modalities=zero_modalities)
        route_logits = self.router(torch.cat((batch.image_embed, batch.text_embed), dim=-1))
        route_weights = torch.sigmoid(route_logits)
        if self.mode == "text_only":
            context = expert_tokens[PROMPT_EXPERT]
            latent = torch.zeros(batch.text_tokens.shape[0], self.config.latent_tokens, self.config.d_model, device=batch.text_tokens.device)
        elif self.mode == "direct":
            context = torch.cat([tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)], dim=1)
            latent = torch.zeros(batch.text_tokens.shape[0], self.config.latent_tokens, self.config.d_model, device=batch.text_tokens.device)
        elif self.mode == "mean_latent":
            latent = self.thought(self.mean_pool(expert_tokens, route_weights))
            context = torch.zeros_like(latent) if disable_latent_access else latent
        elif self.mode == "moe_latent":
            latent = self.thought(self.pump(expert_tokens, route_weights))
            context = torch.zeros_like(latent) if disable_latent_access else latent
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return {"context": context, "latent": latent, "route_logits": route_logits, "route_weights": route_weights}

    def forward(
        self,
        batch: FeatureSet,
        candidate_embeds: torch.Tensor,
        *,
        zero_modalities: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(batch, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        context = self.context_proj(ctx["context"]).mean(dim=1)
        answers = self.answer_proj(candidate_embeds)
        context = F.normalize(context, dim=-1)
        answers = F.normalize(answers, dim=-1)
        scores = torch.einsum("bd,bkd->bk", context, answers) * self.temperature.exp().clamp(max=100.0)
        return {"scores": scores, **ctx}


def decoder_loss(output: dict[str, torch.Tensor], batch: FeatureSet, config: StagePConfig) -> torch.Tensor:
    answer = F.cross_entropy(output["logits"].reshape(-1, VOCAB), batch.answer_target.reshape(-1), ignore_index=PAD)
    router = F.binary_cross_entropy_with_logits(output["route_logits"], batch.required_experts)
    return answer + config.router_loss_weight * router


def train_decoder(
    model: StagePDecoder,
    train: FeatureSet,
    val: FeatureSet,
    config: StagePConfig,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    label: str,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        output = model(batch)
        loss = decoder_loss(output, batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            eval_batch = val.subset(list(range(min(len(val.examples), config.batch_size)))).to(device)
            with torch.no_grad():
                val_loss = decoder_loss(model(eval_batch), eval_batch, config)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "val_loss": round(float(val_loss.detach().cpu()), 4)})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} val_loss={float(val_loss.detach().cpu()):.4f}", flush=True)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def candidate_rows(examples: list[LLaVAExample], answer_pool: list[str], candidate_count: int, *, seed: int) -> tuple[list[list[str]], list[int]]:
    rng = random.Random(seed)
    rows: list[list[str]] = []
    true_indices: list[int] = []
    pool = list(dict.fromkeys(answer_pool))
    for example in examples:
        distractors = [answer for answer in pool if answer != example.answer]
        selected = rng.sample(distractors, k=min(candidate_count - 1, len(distractors)))
        candidates = selected + [example.answer]
        rng.shuffle(candidates)
        rows.append(candidates)
        true_indices.append(candidates.index(example.answer))
    return rows, true_indices


def answer_embedding_lookup(rows: list[list[str]], table: dict[str, torch.Tensor], *, device: torch.device) -> torch.Tensor:
    return torch.stack([torch.stack([table[answer] for answer in row], dim=0) for row in rows], dim=0).to(device=device, dtype=torch.float32)


def build_answer_embedding_table(
    expert: FrozenCLIPTokenExperts,
    answer_pool: list[str],
    config: StagePConfig,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    unique_answers = list(dict.fromkeys(answer_pool))
    features, cost = expert.encode_text_features(unique_answers, batch_size=config.batch_size)
    return {answer: features[index].detach().cpu() for index, answer in enumerate(unique_answers)}, cost


def score_candidate_nll(
    model: StagePDecoder,
    batch: FeatureSet,
    candidates: list[list[str]],
    config: StagePConfig,
    *,
    device: torch.device,
    zero_modalities: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> torch.Tensor:
    repeated, targets = batch.repeat_with_answers(candidates, config.answer_len, device)
    output = model(repeated, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
    losses = F.cross_entropy(output["logits"].reshape(-1, VOCAB), targets.reshape(-1), ignore_index=PAD, reduction="none")
    losses = losses.view(targets.shape[0], targets.shape[1])
    mask = (targets != PAD).float()
    scores = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return scores.view(len(candidates), len(candidates[0]))


def scorer_loss(output: dict[str, torch.Tensor], labels: torch.Tensor, batch: FeatureSet, config: StagePConfig) -> torch.Tensor:
    rank = F.cross_entropy(output["scores"], labels)
    router = F.binary_cross_entropy_with_logits(output["route_logits"], batch.required_experts)
    return rank + config.router_loss_weight * router


def train_scorer(
    model: StagePScorer,
    train: FeatureSet,
    val: FeatureSet,
    config: StagePConfig,
    *,
    answer_pool: list[str],
    answer_table: dict[str, torch.Tensor],
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
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        rows, true_indices = candidate_rows(batch.examples, answer_pool, config.candidate_count, seed=rng.randrange(1_000_000_000))
        labels = torch.tensor(true_indices, dtype=torch.long, device=device)
        candidate_embeds = answer_embedding_lookup(rows, answer_table, device=device)
        output = model(batch, candidate_embeds)
        loss = scorer_loss(output, labels, batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_scorer_ranking(
                model,
                val,
                config,
                device=device,
                answer_pool=answer_pool,
                answer_table=answer_table,
                seed=seed + step,
            )
            if float(metrics["rank_top1"]) > best_top1:
                best_top1 = float(metrics["rank_top1"])
                best_state = copy.deepcopy(model.state_dict())
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "rank_top1": metrics["rank_top1"], "rank_mrr": metrics["rank_mrr"]})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} top1={metrics['rank_top1']:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_rank_top1": best_top1,
        "training_seconds": round(time.perf_counter() - started, 3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


@torch.no_grad()
def evaluate_scorer_ranking(
    model: StagePScorer,
    data: FeatureSet,
    config: StagePConfig,
    *,
    device: torch.device,
    answer_pool: list[str],
    answer_table: dict[str, torch.Tensor],
    seed: int,
    zero_modalities: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    hits: list[float] = []
    reciprocal: list[float] = []
    true_scores: list[float] = []
    for start in range(0, len(data.examples), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(data.examples))))
        batch = data.subset(indices).to(device)
        rows, true_indices = candidate_rows(batch.examples, answer_pool, config.candidate_count, seed=seed + start)
        candidate_embeds = answer_embedding_lookup(rows, answer_table, device=device)
        output = model(batch, candidate_embeds, zero_modalities=zero_modalities, disable_latent_access=disable_latent_access)
        order = output["scores"].argsort(dim=1, descending=True).detach().cpu().tolist()
        for row_order, true_index, row_scores in zip(order, true_indices, output["scores"].detach().cpu().tolist()):
            rank = row_order.index(true_index) + 1
            hits.append(1.0 if rank == 1 else 0.0)
            reciprocal.append(1.0 / rank)
            true_scores.append(float(row_scores[true_index]))
    model.train()
    return {
        "rank_top1": statistics.fmean(hits),
        "rank_mrr": statistics.fmean(reciprocal),
        "true_answer_score": statistics.fmean(true_scores),
        "candidate_count": config.candidate_count,
    }


@torch.no_grad()
def evaluate_ranking(
    model: StagePDecoder,
    data: FeatureSet,
    config: StagePConfig,
    *,
    device: torch.device,
    answer_pool: list[str],
    seed: int,
    zero_modalities: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    hits: list[float] = []
    reciprocal: list[float] = []
    true_scores: list[float] = []
    for start in range(0, len(data.examples), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(data.examples))))
        batch = data.subset(indices)
        rows, true_indices = candidate_rows(batch.examples, answer_pool, config.candidate_count, seed=seed + start)
        scores = score_candidate_nll(
            model,
            batch,
            rows,
            config,
            device=device,
            zero_modalities=zero_modalities,
            disable_latent_access=disable_latent_access,
        )
        order = scores.argsort(dim=1).detach().cpu().tolist()
        for row_order, true_index, row_scores in zip(order, true_indices, scores.detach().cpu().tolist()):
            rank = row_order.index(true_index) + 1
            hits.append(1.0 if rank == 1 else 0.0)
            reciprocal.append(1.0 / rank)
            true_scores.append(float(row_scores[true_index]))
    model.train()
    return {
        "rank_top1": statistics.fmean(hits),
        "rank_mrr": statistics.fmean(reciprocal),
        "true_answer_nll": statistics.fmean(true_scores),
        "candidate_count": config.candidate_count,
    }


@torch.no_grad()
def greedy_samples(
    model: StagePDecoder,
    data: FeatureSet,
    config: StagePConfig,
    *,
    device: torch.device,
    limit: int = 8,
) -> list[str]:
    model.eval()
    batch = data.subset(list(range(min(limit, len(data.examples))))).to(device)
    ctx = model.context(batch)
    marker = {"text_only": 0, "direct": 1, "mean_latent": 2, "moe_latent": 2}[model.mode]
    answer_input = torch.full_like(batch.answer_input, PAD)
    answer_input[:, 0] = BOS
    generated = torch.full_like(batch.answer_target, PAD)
    done = torch.zeros(batch.answer_target.shape[0], dtype=torch.bool, device=device)
    for pos in range(config.answer_len):
        logits = model.output(ctx["context"], answer_input, marker=marker)
        next_token = logits[:, pos].argmax(dim=-1)
        next_token = torch.where(done, torch.full_like(next_token, PAD), next_token)
        generated[:, pos] = next_token
        done = done | (next_token == EOS)
        if pos + 1 < config.answer_len:
            answer_input[:, pos + 1] = torch.where(done, torch.full_like(next_token, PAD), next_token)
    model.train()
    return [decode_tokens(row) for row in generated.detach().cpu().tolist()]


@torch.no_grad()
def evaluate_clip_zero_shot(
    expert: FrozenCLIPTokenExperts,
    data: FeatureSet,
    config: StagePConfig,
    *,
    device: torch.device,
    answer_pool: list[str],
    seed: int,
) -> tuple[dict[str, object], dict[str, float]]:
    rows, true_indices = candidate_rows(data.examples, answer_pool, config.candidate_count, seed=seed)
    flat = [answer for row in rows for answer in row]
    answer_features, cost = expert.encode_text_features(flat, batch_size=config.batch_size)
    answer_features = answer_features.to(device)
    image = data.image_embed.to(device=device, dtype=torch.float32)
    prompt = data.text_embed.to(device=device, dtype=torch.float32)
    scores = []
    for index in range(len(data.examples)):
        start = index * config.candidate_count
        end = start + config.candidate_count
        query = F.normalize(0.65 * image[index] + 0.35 * prompt[index], dim=-1)
        scores.append(answer_features[start:end] @ query)
    matrix = torch.stack(scores, dim=0)
    order = matrix.argsort(dim=1, descending=True).detach().cpu().tolist()
    hits = []
    reciprocal = []
    for row_order, true_index in zip(order, true_indices):
        rank = row_order.index(true_index) + 1
        hits.append(1.0 if rank == 1 else 0.0)
        reciprocal.append(1.0 / rank)
    return {"rank_top1": statistics.fmean(hits), "rank_mrr": statistics.fmean(reciprocal), "candidate_count": config.candidate_count}, cost


def train_latent_probe(
    model: StagePDecoder,
    train: FeatureSet,
    test: FeatureSet,
    config: StagePConfig,
    *,
    device: torch.device,
    steps: int = 100,
) -> dict[str, object]:
    model.eval()
    probe = nn.Linear(config.d_model, len(EXPERTS)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr)
    rng = random.Random(config.seed + 50000)
    started = time.perf_counter()
    for _ in range(steps):
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        with torch.no_grad():
            latent = model.context(batch)["latent"].mean(dim=1)
        logits = probe(latent.detach())
        loss = F.binary_cross_entropy_with_logits(logits, batch.required_experts)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    hits = []
    with torch.no_grad():
        for start in range(0, len(test.examples), config.batch_size):
            batch = test.subset(list(range(start, min(start + config.batch_size, len(test.examples))))).to(device)
            latent = model.context(batch)["latent"].mean(dim=1)
            pred = (torch.sigmoid(probe(latent)) >= 0.5).float()
            hits.extend((pred == batch.required_experts).all(dim=1).float().detach().cpu().tolist())
    return {
        "training_seconds": round(time.perf_counter() - started, 3),
        "route_probe_exact": statistics.fmean(hits),
        "probe_steps": steps,
    }


def measure_prediction_cost(fn, *, example_count: int, device: torch.device) -> dict[str, float]:
    cuda_sync(device)
    started = time.perf_counter()
    fn()
    cuda_sync(device)
    seconds = time.perf_counter() - started
    return {"cached_prediction_seconds": round(seconds, 4), "cached_prediction_ms_per_example": round(seconds * 1000.0 / max(example_count, 1), 4)}


def write_sample_grid(examples: list[LLaVAExample], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[Image.Image] = []
    for example in examples[:8]:
        image = Image.open(example.image_path).convert("RGB")
        image.thumbnail((160, 120))
        canvas = Image.new("RGB", (160, 120), "white")
        canvas.paste(image, ((160 - image.width) // 2, (120 - image.height) // 2))
        thumbs.append(canvas)
        image.close()
    if not thumbs:
        return
    while len(thumbs) < 8:
        thumbs.append(Image.new("RGB", (160, 120), "white"))
    grid = Image.new("RGB", (640, 240), "white")
    for idx, image in enumerate(thumbs):
        grid.paste(image, ((idx % 4) * 160, (idx // 4) * 120))
    draw = ImageDraw.Draw(grid)
    draw.rectangle((0, 0, 639, 239), outline=(40, 40, 40))
    grid.save(output_dir / "llava_coco_grid.jpg", quality=90)
    preview = [
        {
            "id": example.id,
            "image": example.image_name,
            "prompt": example.prompt[:240],
            "answer": example.answer[:240],
            "has_history": example.has_history,
            "source_file": example.source_file,
        }
        for example in examples[:16]
    ]
    (output_dir / "samples.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")


def prediction_samples(models: dict[str, StagePDecoder], test: FeatureSet, config: StagePConfig, *, device: torch.device) -> list[dict[str, object]]:
    generated = {name: greedy_samples(model, test, config, device=device) for name, model in models.items()}
    rows: list[dict[str, object]] = []
    for index, example in enumerate(test.examples[:8]):
        rows.append(
            {
                "id": example.id,
                "image": example.image_name,
                "prompt": example.prompt[:240],
                "expected": example.answer[:320],
                **{f"{name}_greedy": values[index][:320] for name, values in generated.items()},
            }
        )
    return rows


def run_experiment(config: StagePConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_p_llava_moe device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    cache_dir = Path(config.data_cache_dir)
    annotation_paths = download_annotations(config, cache_dir)
    parsed = parse_annotations(annotation_paths, config)
    splits, data_stats = split_examples(parsed, config, cache_dir)
    write_sample_grid(splits["test"], output_path.parent / "samples" / f"seed{config.seed}")

    expert = FrozenCLIPTokenExperts(config.clip_model_name, device=device)
    features, encoding_cost = load_or_encode_features(splits, config, cache_dir, expert, device=device)
    train = features["train"]
    val = features["val"]
    test = features["test"]
    answer_pool = [example.answer for example in train.examples + val.examples + test.examples]
    answer_table, answer_table_cost = build_answer_embedding_table(expert, answer_pool, config, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    models = {
        "text_only_decoder": StagePDecoder(config, mode="text_only"),
        "direct_early_concat": StagePDecoder(config, mode="direct"),
        "moe_mean_pool": StagePDecoder(config, mode="mean_latent"),
        "moe_attention_pump_latent": StagePDecoder(config, mode="moe_latent"),
    }
    training = {
        "text_only_decoder": train_decoder(
            models["text_only_decoder"], train, val, config, steps=config.text_only_steps, seed=config.seed + 10, device=device, label="text_only"
        ),
        "direct_early_concat": train_decoder(
            models["direct_early_concat"], train, val, config, steps=config.direct_steps, seed=config.seed + 20, device=device, label="direct"
        ),
        "moe_mean_pool": train_decoder(
            models["moe_mean_pool"], train, val, config, steps=config.mean_steps, seed=config.seed + 30, device=device, label="mean_pool"
        ),
        "moe_attention_pump_latent": train_decoder(
            models["moe_attention_pump_latent"], train, val, config, steps=config.moe_steps, seed=config.seed + 40, device=device, label="moe_latent"
        ),
    }
    scorer_models = {
        "scorer_text_only": StagePScorer(config, mode="text_only"),
        "scorer_direct": StagePScorer(config, mode="direct"),
        "scorer_mean_pool": StagePScorer(config, mode="mean_latent"),
        "scorer_moe_attention_pump_latent": StagePScorer(config, mode="moe_latent"),
    }
    scorer_training = {
        name: train_scorer(
            model,
            train,
            val,
            config,
            answer_pool=answer_pool,
            answer_table=answer_table,
            steps=config.scorer_steps,
            seed=config.seed + 100 + index * 10,
            device=device,
            label=name,
        )
        for index, (name, model) in enumerate(scorer_models.items())
    }
    for item in training.values():
        item["frozen_expert_parameter_count"] = expert.parameter_count
        item["total_parameter_count"] = expert.parameter_count + item["trainable_parameter_count"]
    for item in scorer_training.values():
        item["frozen_expert_parameter_count"] = expert.parameter_count
        item["total_parameter_count"] = expert.parameter_count + item["trainable_parameter_count"]

    clip_zero_shot, clip_candidate_cost = evaluate_clip_zero_shot(expert, test, config, device=device, answer_pool=answer_pool, seed=config.seed + 1000)
    metrics = {
        "clip_zero_shot_candidate_rank": clip_zero_shot,
        **{
            name: evaluate_ranking(model, test, config, device=device, answer_pool=answer_pool, seed=config.seed + idx * 100)
            for idx, (name, model) in enumerate(models.items(), start=1)
        },
        **{
            name: evaluate_scorer_ranking(
                model,
                test,
                config,
                device=device,
                answer_pool=answer_pool,
                answer_table=answer_table,
                seed=config.seed + idx * 2000,
            )
            for idx, (name, model) in enumerate(scorer_models.items(), start=1)
        },
    }
    ablations = {
        "moe_no_latent_access": evaluate_ranking(
            models["moe_attention_pump_latent"],
            test,
            config,
            device=device,
            answer_pool=answer_pool,
            seed=config.seed + 900,
            disable_latent_access=True,
        ),
        "moe_no_image_modality": evaluate_ranking(
            models["moe_attention_pump_latent"],
            test,
            config,
            device=device,
            answer_pool=answer_pool,
            seed=config.seed + 901,
            zero_modalities=("image",),
        ),
        "moe_no_text_modality": evaluate_ranking(
            models["moe_attention_pump_latent"],
            test,
            config,
            device=device,
            answer_pool=answer_pool,
            seed=config.seed + 902,
            zero_modalities=("text",),
        ),
        "scorer_moe_no_latent_access": evaluate_scorer_ranking(
            scorer_models["scorer_moe_attention_pump_latent"],
            test,
            config,
            device=device,
            answer_pool=answer_pool,
            answer_table=answer_table,
            seed=config.seed + 1900,
            disable_latent_access=True,
        ),
        "scorer_moe_no_image_modality": evaluate_scorer_ranking(
            scorer_models["scorer_moe_attention_pump_latent"],
            test,
            config,
            device=device,
            answer_pool=answer_pool,
            answer_table=answer_table,
            seed=config.seed + 1901,
            zero_modalities=("image",),
        ),
        "scorer_moe_no_text_modality": evaluate_scorer_ranking(
            scorer_models["scorer_moe_attention_pump_latent"],
            test,
            config,
            device=device,
            answer_pool=answer_pool,
            answer_table=answer_table,
            seed=config.seed + 1902,
            zero_modalities=("text",),
        ),
    }
    latent_probe = train_latent_probe(scorer_models["scorer_moe_attention_pump_latent"], train, test, config, device=device, steps=min(100, config.scorer_steps))
    prediction_cost = {
        "clip_expert_encoding": encoding_cost["test"],
        "clip_candidate_text_encoding": clip_candidate_cost,
        "clip_answer_embedding_table": answer_table_cost,
        **{
            f"{name}_cached": measure_prediction_cost(
                lambda model=model: evaluate_ranking(model, test, config, device=device, answer_pool=answer_pool, seed=config.seed + 700),
                example_count=len(test.examples),
                device=device,
            )
            for name, model in models.items()
        },
        **{
            f"{name}_cached": measure_prediction_cost(
                lambda model=model: evaluate_scorer_ranking(
                    model,
                    test,
                    config,
                    device=device,
                    answer_pool=answer_pool,
                    answer_table=answer_table,
                    seed=config.seed + 1700,
                ),
                example_count=len(test.examples),
                device=device,
            )
            for name, model in scorer_models.items()
        },
    }
    for key, cost in list(prediction_cost.items()):
        if key.endswith("_cached"):
            cost["with_clip_ms_per_example"] = round(cost["cached_prediction_ms_per_example"] + encoding_cost["test"]["encoding_ms_per_example"], 4)

    output = {
        "experiment": "omni_transformer_stage_p_llava_moe",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "dataset": DATASET_REPO,
            "annotation_files": config.annotation_files,
            "image_source": "COCO image URLs resolved from LLaVA image names",
            "frozen_expert": config.clip_model_name,
            "experts": EXPERTS,
            "latent_tokens": config.latent_tokens,
            "ranking_metric": "true answer versus sampled LLaVA answer distractors by decoder NLL",
            "note": "TextOutputExpert sees only latent tokens in the MoE attention-pump path.",
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "data": data_stats,
        "precompute_cost": encoding_cost,
        "metrics": metrics,
        "ablations": ablations,
        "latent_probe": latent_probe,
        "training": {**training, **scorer_training},
        "prediction_cost": prediction_cost,
        "prediction_samples": prediction_samples(models, test, config, device=device),
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
        numbers = collect_numbers(
            {
                "metrics": run["metrics"],
                "ablations": run["ablations"],
                "latent_probe": run["latent_probe"],
                "precompute_cost": run["precompute_cost"],
                "training": run["training"],
                "prediction_cost": run["prediction_cost"],
                "data": run["data"],
            }
        )
        for key, value in numbers.items():
            buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_p_llava_moe_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stage_m.stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_p_llava_moe/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_p_llava_moe/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_p_llava_moe/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--annotation-files", default=",".join(DATASET_FILES))
    parser.add_argument("--data-cache-dir", default="artifacts/omni_transformer_stage_p_llava_moe/data")
    parser.add_argument("--clip-model-name", default=CLIP_MODEL_NAME)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--answer-len", type=int, default=192)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--text-only-steps", type=int, default=120)
    parser.add_argument("--direct-steps", type=int, default=160)
    parser.add_argument("--mean-steps", type=int, default=160)
    parser.add_argument("--moe-steps", type=int, default=220)
    parser.add_argument("--scorer-steps", type=int, default=240)
    args = parser.parse_args()

    base_config = StagePConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        clip_model_name=args.clip_model_name,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        answer_len=args.answer_len,
        latent_tokens=args.latent_tokens,
        candidate_count=args.candidate_count,
        text_only_steps=args.text_only_steps,
        direct_steps=args.direct_steps,
        mean_steps=args.mean_steps,
        moe_steps=args.moe_steps,
        scorer_steps=args.scorer_steps,
        annotation_files=parse_csv_strings(args.annotation_files),
        data_cache_dir=args.data_cache_dir,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StagePConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
