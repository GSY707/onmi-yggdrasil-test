from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import omni_transformer_stage_m_moe_multimodal_llm as stage_m
import omni_transformer_stage_n_strong_experts as stage_n


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBED_DIM = 512
PRETRAINED_EXPERTS = ("clip_vision", "clip_text", "spatial_adapter", "counting_adapter", "chart_adapter")
TARGET_CLASSES = stage_m.TARGET_CLASSES


@dataclass(frozen=True)
class StageOConfig:
    train_size: int = 768
    val_size: int = 256
    test_size: int = 256
    batch_size: int = 64
    seed: int = 20260701
    clip_model_name: str = CLIP_MODEL_NAME
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    classifier_steps: int = 260
    direct_steps: int = 260
    moe_steps: int = 320
    eval_every: int = 200
    router_loss_weight: float = 0.25


@dataclass
class ClipEmbeddingSet:
    examples: list[stage_m.Example]
    image: torch.Tensor
    text: torch.Tensor
    answer_input: torch.Tensor
    answer_target: torch.Tensor
    task: torch.Tensor
    target_class: torch.Tensor
    required_experts: torch.Tensor

    def subset(self, indices: list[int]) -> "ClipEmbeddingSet":
        return ClipEmbeddingSet(
            examples=[self.examples[index] for index in indices],
            image=self.image[indices],
            text=self.text[indices],
            answer_input=self.answer_input[indices],
            answer_target=self.answer_target[indices],
            task=self.task[indices],
            target_class=self.target_class[indices],
            required_experts=self.required_experts[indices],
        )

    def to(self, device: torch.device) -> "ClipEmbeddingSet":
        return ClipEmbeddingSet(
            examples=self.examples,
            image=self.image.to(device),
            text=self.text.to(device),
            answer_input=self.answer_input.to(device),
            answer_target=self.answer_target.to(device),
            task=self.task.to(device),
            target_class=self.target_class.to(device),
            required_experts=self.required_experts.to(device),
        )


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def image_to_pil(example: stage_m.Example) -> Image.Image:
    array = np.array(example.image, dtype=np.float32)
    array = np.transpose(array, (1, 2, 0))
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class FrozenCLIPExperts:
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
    def encode_examples(self, examples: list[stage_m.Example], *, batch_size: int) -> tuple[ClipEmbeddingSet, dict[str, float]]:
        image_rows: list[torch.Tensor] = []
        text_rows: list[torch.Tensor] = []
        started = time.perf_counter()
        cuda_sync(self.device)
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            inputs = self.processor(
                text=[example.prompt for example in chunk],
                images=[image_to_pil(example) for example in chunk],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            image_rows.append(F.normalize(image_features, dim=-1).detach().cpu())
            text_rows.append(F.normalize(text_features, dim=-1).detach().cpu())
        cuda_sync(self.device)
        seconds = time.perf_counter() - started
        answer_target = [stage_m.encode_text(example.answer, stage_m.ANSWER_LEN, add_eos=True) for example in examples]
        embedding_set = ClipEmbeddingSet(
            examples=examples,
            image=torch.cat(image_rows, dim=0),
            text=torch.cat(text_rows, dim=0),
            answer_input=torch.tensor([stage_m.answer_inputs(target) for target in answer_target], dtype=torch.long),
            answer_target=torch.tensor(answer_target, dtype=torch.long),
            task=torch.tensor([example.task for example in examples], dtype=torch.long),
            target_class=torch.tensor([example.target_class for example in examples], dtype=torch.long),
            required_experts=torch.tensor([example.required_experts for example in examples], dtype=torch.float32),
        )
        return embedding_set, {
            "encoding_seconds": round(seconds, 4),
            "encoding_ms_per_example": round(seconds * 1000.0 / max(len(examples), 1), 4),
        }

    @torch.no_grad()
    def encode_texts(self, texts: list[str], *, batch_size: int) -> tuple[torch.Tensor, dict[str, float]]:
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


def random_embedding_batch(data: ClipEmbeddingSet, *, rng: random.Random, batch_size: int, device: torch.device) -> ClipEmbeddingSet:
    indices = [rng.randrange(len(data.examples)) for _ in range(batch_size)]
    return data.subset(indices).to(device)


class ClipExpertBank(nn.Module):
    def __init__(self, config: StageOConfig) -> None:
        super().__init__()
        self.vision = nn.Sequential(nn.LayerNorm(CLIP_EMBED_DIM), nn.Linear(CLIP_EMBED_DIM, config.d_model), nn.GELU())
        self.text = nn.Sequential(nn.LayerNorm(CLIP_EMBED_DIM), nn.Linear(CLIP_EMBED_DIM, config.d_model), nn.GELU())
        self.adapters = nn.ModuleDict(
            {
                "spatial": nn.Sequential(nn.LayerNorm(CLIP_EMBED_DIM * 2), nn.Linear(CLIP_EMBED_DIM * 2, config.d_model), nn.GELU()),
                "counting": nn.Sequential(nn.LayerNorm(CLIP_EMBED_DIM * 2), nn.Linear(CLIP_EMBED_DIM * 2, config.d_model), nn.GELU()),
                "chart": nn.Sequential(nn.LayerNorm(CLIP_EMBED_DIM * 2), nn.Linear(CLIP_EMBED_DIM * 2, config.d_model), nn.GELU()),
            }
        )
        self.expert_type = nn.Embedding(len(PRETRAINED_EXPERTS), config.d_model)

    def forward(self, batch: ClipEmbeddingSet) -> list[torch.Tensor]:
        fused = torch.cat((batch.image, batch.text), dim=-1)
        tokens = [
            self.vision(batch.image),
            self.text(batch.text),
            self.adapters["spatial"](fused),
            self.adapters["counting"](fused),
            self.adapters["chart"](fused),
        ]
        return [(token + self.expert_type.weight[idx]).unsqueeze(1) for idx, token in enumerate(tokens)]


class ClipExpertDecoder(nn.Module):
    def __init__(self, config: StageOConfig, *, mode: str) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        model_config = stage_m.MoEConfig(
            batch_size=config.batch_size,
            seed=config.seed,
            d_model=config.d_model,
            layers=config.layers,
            heads=config.heads,
            dropout=config.dropout,
            lr=config.lr,
        )
        self.bank = ClipExpertBank(config)
        self.router = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM * 2),
            nn.Linear(CLIP_EMBED_DIM * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(PRETRAINED_EXPERTS)),
        )
        self.pump = stage_m.AttentionPump(model_config)
        self.thought = stage_m.ThoughtExpert(model_config)
        self.output = stage_m.TextOutputExpert(model_config)

    def context(
        self,
        batch: ClipEmbeddingSet,
        *,
        zero_modalities: Iterable[str] = (),
        zero_experts: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        zero_modality_set = set(zero_modalities)
        image = torch.zeros_like(batch.image) if "image" in zero_modality_set else batch.image
        text = torch.zeros_like(batch.text) if "text" in zero_modality_set else batch.text
        local_batch = ClipEmbeddingSet(
            examples=batch.examples,
            image=image,
            text=text,
            answer_input=batch.answer_input,
            answer_target=batch.answer_target,
            task=batch.task,
            target_class=batch.target_class,
            required_experts=batch.required_experts,
        )
        expert_tokens = self.bank(local_batch)
        route_logits = self.router(torch.cat((image, text), dim=-1))
        route_weights = torch.sigmoid(route_logits)
        zero_expert_set = set(zero_experts)
        for idx, name in enumerate(PRETRAINED_EXPERTS):
            if name in zero_expert_set:
                route_weights[:, idx] = 0.0
                expert_tokens[idx] = torch.zeros_like(expert_tokens[idx])
        if self.mode == "direct":
            context = torch.cat([tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)], dim=1)
            latent = torch.zeros(batch.image.shape[0], stage_m.LATENT_TOKENS, context.shape[-1], device=context.device)
        elif self.mode == "moe_latent":
            pumped = self.pump(expert_tokens, route_weights)
            latent = self.thought(pumped)
            context = torch.zeros_like(latent) if disable_latent_access else latent
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return {"context": context, "latent": latent, "route_logits": route_logits, "route_weights": route_weights}

    def forward(
        self,
        batch: ClipEmbeddingSet,
        *,
        zero_modalities: Iterable[str] = (),
        zero_experts: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(
            batch,
            zero_modalities=zero_modalities,
            zero_experts=zero_experts,
            disable_latent_access=disable_latent_access,
        )
        marker = stage_m.SPECIAL_DIRECT if self.mode == "direct" else stage_m.SPECIAL_LATENT
        logits = self.output(ctx["context"], batch.answer_input, answer_context_marker=marker)
        return {"logits": logits, **ctx}


class ClipLinearClassifier(nn.Module):
    def __init__(self, config: StageOConfig) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM * 2),
            nn.Linear(CLIP_EMBED_DIM * 2, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, TARGET_CLASSES),
        )
        self.task = nn.Linear(CLIP_EMBED_DIM * 2, len(stage_m.TASKS))

    def forward(self, batch: ClipEmbeddingSet) -> dict[str, torch.Tensor]:
        hidden = torch.cat((batch.image, batch.text), dim=-1)
        return {"target": self.head(hidden), "task": self.task(hidden)}


def decoder_loss(output: dict[str, torch.Tensor], batch: ClipEmbeddingSet, config: StageOConfig) -> torch.Tensor:
    answer = F.cross_entropy(output["logits"].view(-1, stage_m.VOCAB), batch.answer_target.view(-1), ignore_index=stage_m.PAD)
    router = F.binary_cross_entropy_with_logits(output["route_logits"], batch.required_experts)
    return answer + config.router_loss_weight * router


def train_decoder(
    model: ClipExpertDecoder,
    train: ClipEmbeddingSet,
    val: ClipEmbeddingSet,
    config: StageOConfig,
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
        batch = random_embedding_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        output = model(batch)
        loss = decoder_loss(output, batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_decoder_teacher_forced(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "answer_exact": metrics["answer_exact"]})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} exact={metrics['answer_exact']:.3f}", flush=True)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def train_classifier(
    model: ClipLinearClassifier,
    train: ClipEmbeddingSet,
    val: ClipEmbeddingSet,
    config: StageOConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.classifier_steps + 1):
        batch = random_embedding_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        output = model(batch)
        loss = F.cross_entropy(output["target"], batch.target_class) + 0.2 * F.cross_entropy(output["task"], batch.task)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.classifier_steps:
            metrics = evaluate_classifier(model, val, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "answer_exact": metrics["answer_exact"]})
            print(f"clip_classifier step={step:4d} loss={float(loss.detach().cpu()):.4f} exact={metrics['answer_exact']:.3f}", flush=True)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def logits_to_text(logits: torch.Tensor) -> list[str]:
    pred = logits.argmax(dim=-1).detach().cpu().tolist()
    return [stage_m.decode_tokens(row) for row in pred]


@torch.no_grad()
def evaluate_decoder_teacher_forced(
    model: ClipExpertDecoder,
    data: ClipEmbeddingSet,
    config: StageOConfig,
    *,
    device: torch.device,
    zero_modalities: Iterable[str] = (),
    zero_experts: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    predictions: list[str] = []
    token_hits: list[float] = []
    route_hits: list[float] = []
    for start in range(0, len(data.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.examples))))).to(device)
        output = model(
            batch,
            zero_modalities=zero_modalities,
            zero_experts=zero_experts,
            disable_latent_access=disable_latent_access,
        )
        predictions.extend(logits_to_text(output["logits"]))
        token_hits.append(stage_m.masked_token_accuracy(output["logits"], batch.answer_target))
        route = (torch.sigmoid(output["route_logits"]) >= 0.5).float()
        route_hits.extend((route == batch.required_experts).all(dim=1).float().detach().cpu().tolist())
    metrics = stage_n.text_metrics(data.examples, predictions)
    metrics["token_accuracy"] = statistics.fmean(token_hits)
    metrics["router_exact"] = statistics.fmean(route_hits)
    model.train()
    return metrics


@torch.no_grad()
def greedy_decoder_predictions(
    model: ClipExpertDecoder,
    data: ClipEmbeddingSet,
    config: StageOConfig,
    *,
    device: torch.device,
    zero_modalities: Iterable[str] = (),
    zero_experts: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> list[str]:
    model.eval()
    predictions: list[str] = []
    for start in range(0, len(data.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.examples))))).to(device)
        ctx = model.context(
            batch,
            zero_modalities=zero_modalities,
            zero_experts=zero_experts,
            disable_latent_access=disable_latent_access,
        )
        marker = stage_m.SPECIAL_DIRECT if model.mode == "direct" else stage_m.SPECIAL_LATENT
        answer_input = torch.full_like(batch.answer_input, stage_m.PAD)
        answer_input[:, 0] = stage_m.BOS
        generated = torch.full_like(batch.answer_target, stage_m.PAD)
        done = torch.zeros(batch.answer_target.shape[0], dtype=torch.bool, device=device)
        for pos in range(stage_m.ANSWER_LEN):
            logits = model.output(ctx["context"], answer_input, answer_context_marker=marker)
            next_token = logits[:, pos].argmax(dim=-1)
            next_token = torch.where(done, torch.full_like(next_token, stage_m.PAD), next_token)
            generated[:, pos] = next_token
            done = done | (next_token == stage_m.EOS)
            if pos + 1 < stage_m.ANSWER_LEN:
                answer_input[:, pos + 1] = torch.where(done, torch.full_like(next_token, stage_m.PAD), next_token)
        predictions.extend(stage_m.decode_tokens(row) for row in generated.detach().cpu().tolist())
    model.train()
    return predictions


@torch.no_grad()
def evaluate_decoder_greedy(
    model: ClipExpertDecoder,
    data: ClipEmbeddingSet,
    config: StageOConfig,
    *,
    device: torch.device,
    zero_modalities: Iterable[str] = (),
    zero_experts: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> dict[str, object]:
    return stage_n.text_metrics(
        data.examples,
        greedy_decoder_predictions(
            model,
            data,
            config,
            device=device,
            zero_modalities=zero_modalities,
            zero_experts=zero_experts,
            disable_latent_access=disable_latent_access,
        ),
    )


@torch.no_grad()
def evaluate_classifier(model: ClipLinearClassifier, data: ClipEmbeddingSet, config: StageOConfig, *, device: torch.device) -> dict[str, object]:
    model.eval()
    predictions: list[str] = []
    target_hits: list[float] = []
    task_hits: list[float] = []
    for start in range(0, len(data.examples), config.batch_size):
        batch = data.subset(list(range(start, min(start + config.batch_size, len(data.examples))))).to(device)
        output = model(batch)
        targets = output["target"].argmax(dim=-1).detach().cpu().tolist()
        predictions.extend(stage_n.render_answer(example, target) for example, target in zip(batch.examples, targets))
        target_hits.extend((output["target"].argmax(dim=-1) == batch.target_class).float().detach().cpu().tolist())
        task_hits.extend((output["task"].argmax(dim=-1) == batch.task).float().detach().cpu().tolist())
    metrics = stage_n.text_metrics(data.examples, predictions)
    metrics["target_accuracy"] = statistics.fmean(target_hits)
    metrics["task_accuracy"] = statistics.fmean(task_hits)
    model.train()
    return metrics


def candidate_targets(example: stage_m.Example) -> list[int]:
    if example.task == stage_m.TASK_ATTRIBUTE:
        return list(range(len(stage_m.COLORS)))
    if example.task == stage_m.TASK_COUNTING:
        return [1, 2, 3]
    if example.task == stage_m.TASK_SPATIAL:
        return [0, 1]
    if example.task == stage_m.TASK_RULE:
        return list(range(len(stage_m.ACTIONS)))
    return list(range(4))


def evaluate_zero_shot_clip(
    expert: FrozenCLIPExperts,
    data: ClipEmbeddingSet,
    config: StageOConfig,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, float]]:
    texts: list[str] = []
    spans: list[tuple[int, int, list[int]]] = []
    cursor = 0
    for example in data.examples:
        targets = candidate_targets(example)
        candidates = [f"{example.prompt} answer: {stage_n.render_answer(example, target)}" for target in targets]
        texts.extend(candidates)
        spans.append((cursor, cursor + len(candidates), targets))
        cursor += len(candidates)
    text_features, cost = expert.encode_texts(texts, batch_size=config.batch_size)
    image = data.image.to(device)
    prompt = data.text.to(device)
    text_features = text_features.to(device)
    predictions: list[str] = []
    for example, (start, end, targets), image_feature, prompt_feature in zip(data.examples, spans, image, prompt):
        candidate_features = text_features[start:end]
        scores = 0.65 * (candidate_features @ image_feature) + 0.35 * (candidate_features @ prompt_feature)
        target = targets[int(scores.argmax().detach().cpu())]
        predictions.append(stage_n.render_answer(example, target))
    return stage_n.text_metrics(data.examples, predictions), cost


def measure_prediction_cost(fn: Callable[[], object], *, example_count: int, device: torch.device) -> dict[str, float]:
    cuda_sync(device)
    started = time.perf_counter()
    fn()
    cuda_sync(device)
    seconds = time.perf_counter() - started
    return {"cached_prediction_seconds": round(seconds, 4), "cached_prediction_ms_per_example": round(seconds * 1000.0 / example_count, 4)}


def prediction_samples(
    direct: ClipExpertDecoder,
    moe: ClipExpertDecoder,
    classifier: ClipLinearClassifier,
    data: ClipEmbeddingSet,
    config: StageOConfig,
    *,
    device: torch.device,
) -> list[dict[str, object]]:
    sample_data = data.subset(list(range(min(12, len(data.examples)))))
    direct_preds = greedy_decoder_predictions(direct, sample_data, config, device=device)
    moe_preds = greedy_decoder_predictions(moe, sample_data, config, device=device)
    classifier.eval()
    batch = sample_data.to(device)
    with torch.no_grad():
        classifier_targets = classifier(batch)["target"].argmax(dim=-1).detach().cpu().tolist()
    classifier_preds = [stage_n.render_answer(example, target) for example, target in zip(sample_data.examples, classifier_targets)]
    rows = []
    for example, direct_pred, moe_pred, classifier_pred in zip(sample_data.examples, direct_preds, moe_preds, classifier_preds):
        rows.append(
            {
                "task": stage_m.TASKS[example.task],
                "prompt": example.prompt,
                "expected": example.answer,
                "clip_direct_decoder": direct_pred,
                "clip_moe_latent_decoder": moe_pred,
                "clip_linear_classifier": classifier_pred,
                "required_experts": [PRETRAINED_EXPERTS[idx] for idx, value in enumerate(example.required_experts) if value > 0],
            }
        )
    return rows


def run_experiment(config: StageOConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_o_pretrained_experts device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_examples = stage_m.generate_examples(config.train_size, seed=config.seed + 1)
    val_examples = stage_m.generate_examples(config.val_size, seed=config.seed + 2)
    test_examples = stage_m.generate_examples(config.test_size, seed=config.seed + 3)
    stage_m.write_samples(test_examples, output_dir=output_path.parent / "samples" / f"seed{config.seed}")

    expert = FrozenCLIPExperts(config.clip_model_name, device=device)
    train, train_encoding = expert.encode_examples(train_examples, batch_size=config.batch_size)
    val, val_encoding = expert.encode_examples(val_examples, batch_size=config.batch_size)
    test, test_encoding = expert.encode_examples(test_examples, batch_size=config.batch_size)

    direct = ClipExpertDecoder(config, mode="direct")
    moe = ClipExpertDecoder(config, mode="moe_latent")
    classifier = ClipLinearClassifier(config)
    training = {
        "clip_zero_shot": {
            "training_seconds": 0.0,
            "frozen_expert_parameter_count": expert.parameter_count,
            "trainable_parameter_count": 0,
        },
        "clip_linear_classifier": train_classifier(classifier, train, val, config, seed=config.seed + 10, device=device),
        "clip_direct_decoder": train_decoder(
            direct, train, val, config, steps=config.direct_steps, seed=config.seed + 20, device=device, label="clip_direct"
        ),
        "clip_moe_latent_decoder": train_decoder(
            moe, train, val, config, steps=config.moe_steps, seed=config.seed + 30, device=device, label="clip_moe_latent"
        ),
    }
    for item in ("clip_linear_classifier", "clip_direct_decoder", "clip_moe_latent_decoder"):
        training[item]["frozen_expert_parameter_count"] = expert.parameter_count
        training[item]["total_parameter_count"] = expert.parameter_count + training[item]["trainable_parameter_count"]

    zero_shot_metrics, zero_shot_cost = evaluate_zero_shot_clip(expert, test, config, device=device)
    metrics = {
        "clip_zero_shot": zero_shot_metrics,
        "clip_linear_classifier": evaluate_classifier(classifier, test, config, device=device),
        "clip_direct_decoder": evaluate_decoder_teacher_forced(direct, test, config, device=device),
        "clip_moe_latent_decoder": evaluate_decoder_teacher_forced(moe, test, config, device=device),
    }
    generation_metrics = {
        "clip_direct_decoder": evaluate_decoder_greedy(direct, test, config, device=device),
        "clip_moe_latent_decoder": evaluate_decoder_greedy(moe, test, config, device=device),
    }
    ablations = {
        "clip_moe_no_latent_access": evaluate_decoder_teacher_forced(moe, test, config, device=device, disable_latent_access=True),
        "clip_moe_no_image_modality": evaluate_decoder_teacher_forced(
            moe, test, config, device=device, zero_modalities=("image",)
        ),
        "clip_moe_no_text_modality": evaluate_decoder_teacher_forced(
            moe, test, config, device=device, zero_modalities=("text",)
        ),
        "clip_moe_no_function_adapters": evaluate_decoder_teacher_forced(
            moe,
            test,
            config,
            device=device,
            zero_experts=("spatial_adapter", "counting_adapter", "chart_adapter"),
        ),
        **{
            f"clip_moe_no_{name}_expert": evaluate_decoder_teacher_forced(moe, test, config, device=device, zero_experts=(name,))
            for name in PRETRAINED_EXPERTS
        },
    }

    prediction_cost = {
        "clip_expert_encoding": test_encoding,
        "clip_zero_shot_candidate_text_encoding": zero_shot_cost,
        "clip_linear_classifier_cached": measure_prediction_cost(
            lambda: evaluate_classifier(classifier, test, config, device=device),
            example_count=len(test.examples),
            device=device,
        ),
        "clip_direct_decoder_cached": measure_prediction_cost(
            lambda: evaluate_decoder_greedy(direct, test, config, device=device),
            example_count=len(test.examples),
            device=device,
        ),
        "clip_moe_latent_decoder_cached": measure_prediction_cost(
            lambda: evaluate_decoder_greedy(moe, test, config, device=device),
            example_count=len(test.examples),
            device=device,
        ),
    }
    for key in ("clip_linear_classifier_cached", "clip_direct_decoder_cached", "clip_moe_latent_decoder_cached"):
        prediction_cost[key]["with_clip_ms_per_example"] = round(
            prediction_cost[key]["cached_prediction_ms_per_example"] + test_encoding["encoding_ms_per_example"], 4
        )

    output = {
        "experiment": "omni_transformer_stage_o_pretrained_experts",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "architecture": {
            "pretrained_expert": config.clip_model_name,
            "frozen_expert_parameter_count": expert.parameter_count,
            "experts": PRETRAINED_EXPERTS,
            "note": "CLIP vision/text encoders are real frozen pretrained experts; task adapters/decoders are trained locally.",
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "precompute_cost": {"train": train_encoding, "val": val_encoding, "test": test_encoding},
        "metrics": metrics,
        "generation_metrics": generation_metrics,
        "ablations": ablations,
        "training": training,
        "prediction_cost": prediction_cost,
        "prediction_samples": prediction_samples(direct, moe, classifier, test, config, device=device),
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
                "generation_metrics": run["generation_metrics"],
                "ablations": run["ablations"],
                "precompute_cost": run["precompute_cost"],
                "training": run["training"],
                "prediction_cost": run["prediction_cost"],
            }
        )
        for key, value in numbers.items():
            buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_o_pretrained_experts_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stage_m.stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_o_pretrained_experts/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_o_pretrained_experts/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_o_pretrained_experts/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--clip-model-name", default=CLIP_MODEL_NAME)
    parser.add_argument("--train-size", type=int, default=768)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--classifier-steps", type=int, default=260)
    parser.add_argument("--direct-steps", type=int, default=260)
    parser.add_argument("--moe-steps", type=int, default=320)
    args = parser.parse_args()

    base_config = StageOConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        clip_model_name=args.clip_model_name,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        classifier_steps=args.classifier_steps,
        direct_steps=args.direct_steps,
        moe_steps=args.moe_steps,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageOConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
