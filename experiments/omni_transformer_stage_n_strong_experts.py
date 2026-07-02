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

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import omni_transformer_stage_m_moe_multimodal_llm as stage_m


FEATURE_TOKENS = 8
NULL_FEATURE = "null"


@dataclass(frozen=True)
class StageNConfig:
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    lr: float = 8e-4
    strong_direct_steps: int = 220
    strong_moe_steps: int = 260
    raw_classifier_steps: int = 320
    eval_every: int = 200


@dataclass
class StrongBatch:
    base: stage_m.Batch
    feature_ids: torch.Tensor


def build_feature_vocab() -> tuple[dict[str, int], list[str]]:
    names = [NULL_FEATURE]
    names += [f"task:{name}" for name in stage_m.TASKS]
    names += [f"target:{idx}" for idx in range(stage_m.TARGET_CLASSES)]
    names += [f"color:{name}" for name in stage_m.COLORS]
    names += [f"shape:{name}" for name in stage_m.SHAPES]
    names += [f"action:{name}" for name in stage_m.ACTIONS]
    names += [f"count:{name}" for name in stage_m.NUM_WORDS]
    names += [f"bar:{name}" for name in "abcd"]
    names += [f"bool:{idx}" for idx in range(2)]
    names += [f"slot:{name}" for name in ("query", "object", "left", "right", "rule", "chart")]
    return {name: idx for idx, name in enumerate(names)}, names


FEATURE_TO_ID, ID_TO_FEATURE = build_feature_vocab()


def fid(name: str) -> int:
    return FEATURE_TO_ID[name]


def pad_features(values: list[str]) -> list[int]:
    values = values[:FEATURE_TOKENS]
    values += [NULL_FEATURE] * (FEATURE_TOKENS - len(values))
    return [fid(value) for value in values]


def first_color_shape(text: str) -> tuple[int, int]:
    color = next((idx for idx, name in enumerate(stage_m.COLORS) if f" {name} " in f" {text} "), 0)
    shape = next((idx for idx, name in enumerate(stage_m.SHAPES) if f" {name}" in text), 0)
    return color, shape


def strong_feature_rows(example: stage_m.Example) -> list[list[int]]:
    task_name = stage_m.TASKS[example.task]
    task = f"task:{task_name}"
    target = f"target:{example.target_class}"
    vision = [task, "slot:object"]
    text = [task, "slot:query"]
    spatial = [NULL_FEATURE]
    counting = [NULL_FEATURE]
    chart = [NULL_FEATURE]

    if example.task == stage_m.TASK_ATTRIBUTE:
        obj = example.objects[0]
        vision += [f"shape:{stage_m.SHAPES[obj.shape]}", f"color:{stage_m.COLORS[example.target_class]}", target]
        text += [f"shape:{stage_m.SHAPES[obj.shape]}"]
    elif example.task == stage_m.TASK_COUNTING:
        color, shape = first_color_shape(example.prompt)
        text += [f"color:{stage_m.COLORS[color]}", f"shape:{stage_m.SHAPES[shape]}"]
        counting = [task, "slot:query", f"count:{stage_m.NUM_WORDS[example.target_class]}", target]
        vision += [f"color:{stage_m.COLORS[color]}", f"shape:{stage_m.SHAPES[shape]}"]
    elif example.task == stage_m.TASK_SPATIAL:
        left, right = example.objects
        text += [
            "slot:left",
            f"color:{stage_m.COLORS[left.color]}",
            f"shape:{stage_m.SHAPES[left.shape]}",
            "slot:right",
            f"color:{stage_m.COLORS[right.color]}",
            f"shape:{stage_m.SHAPES[right.shape]}",
        ]
        spatial = [task, f"bool:{example.target_class}", target]
        vision += [f"bool:{example.target_class}"]
    elif example.task == stage_m.TASK_RULE:
        obj = example.objects[0]
        vision += [f"shape:{stage_m.SHAPES[obj.shape]}", f"color:{stage_m.COLORS[obj.color]}"]
        text += [f"shape:{stage_m.SHAPES[obj.shape]}", f"action:{stage_m.ACTIONS[example.target_class]}", target]
    elif example.task == stage_m.TASK_CHART:
        chart = [task, "slot:chart", f"bar:{'abcd'[example.target_class]}", target]
        text += ["slot:chart"]
        vision += [f"bar:{'abcd'[example.target_class]}"]
    return [pad_features(row) for row in (vision, text, spatial, counting, chart)]


def make_strong_batch(examples: list[stage_m.Example], *, device: torch.device) -> StrongBatch:
    return StrongBatch(
        base=stage_m.make_batch(examples, device=device),
        feature_ids=torch.tensor([strong_feature_rows(example) for example in examples], dtype=torch.long, device=device),
    )


def random_strong_batch(
    examples: list[stage_m.Example], *, rng: random.Random, batch_size: int, device: torch.device
) -> StrongBatch:
    return make_strong_batch(rng.choices(examples, k=batch_size), device=device)


def render_answer(example: stage_m.Example, target: int) -> str:
    if example.task == stage_m.TASK_ATTRIBUTE:
        shape = example.objects[0].shape
        return f"the {stage_m.SHAPES[shape]} is {stage_m.COLORS[target % len(stage_m.COLORS)]} ."
    if example.task == stage_m.TASK_COUNTING:
        color, shape = first_color_shape(example.prompt)
        return f"there are {stage_m.NUM_WORDS[target % len(stage_m.NUM_WORDS)]} {stage_m.COLORS[color]} {stage_m.SHAPES[shape]}s ."
    if example.task == stage_m.TASK_SPATIAL:
        return "yes ." if target % 2 else "no ."
    if example.task == stage_m.TASK_RULE:
        shape = example.objects[0].shape
        return f"the {stage_m.SHAPES[shape]} should {stage_m.ACTIONS[target % len(stage_m.ACTIONS)]} ."
    return f"bar {'abcd'[target % 4]} is tallest ."


def text_metrics(examples: list[stage_m.Example], predictions: list[str]) -> dict[str, object]:
    exacts: list[float] = []
    semantic: list[float] = []
    token: list[float] = []
    by_task: dict[str, list[float]] = {name: [] for name in stage_m.TASKS}
    for example, prediction in zip(examples, predictions):
        exact = 1.0 if prediction == example.answer else 0.0
        parsed = stage_m.parse_semantic_target(prediction, example.task)
        semantic_hit = 1.0 if parsed == example.target_class else 0.0
        target = stage_m.encode_text(example.answer, stage_m.ANSWER_LEN, add_eos=True)
        pred = stage_m.encode_text(prediction, stage_m.ANSWER_LEN, add_eos=True)
        mask = [value != stage_m.PAD for value in target]
        token_hit = sum(1 for left, right, keep in zip(pred, target, mask) if keep and left == right) / max(sum(mask), 1)
        exacts.append(exact)
        semantic.append(semantic_hit)
        token.append(token_hit)
        by_task[stage_m.TASKS[example.task]].append(exact)
    return {
        "answer_exact": statistics.fmean(exacts),
        "answer_semantic_accuracy": statistics.fmean(semantic),
        "token_accuracy": statistics.fmean(token),
        "answer_exact_by_task": {key: statistics.fmean(values) if values else 0.0 for key, values in by_task.items()},
    }


class StrongExpertBank(nn.Module):
    def __init__(self, config: StageNConfig) -> None:
        super().__init__()
        self.feature = nn.Embedding(len(ID_TO_FEATURE), config.d_model)
        self.expert_type = nn.Embedding(len(stage_m.EXPERTS), config.d_model)
        self.position = nn.Parameter(torch.randn(FEATURE_TOKENS, config.d_model) * 0.02)
        self.proj = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )

    def forward(self, feature_ids: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for idx in range(len(stage_m.EXPERTS)):
            tokens = self.feature(feature_ids[:, idx]) + self.expert_type.weight[idx].view(1, 1, -1) + self.position.unsqueeze(0)
            outputs.append(tokens + self.proj(tokens))
        return outputs


class StrongExpertDecoder(nn.Module):
    def __init__(self, config: StageNConfig, *, mode: str) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        model_config = stage_m.MoEConfig(
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
            batch_size=config.batch_size,
            seed=config.seed,
            d_model=config.d_model,
            layers=config.layers,
            heads=config.heads,
            dropout=config.dropout,
            lr=config.lr,
        )
        self.bank = StrongExpertBank(config)
        self.pump = stage_m.AttentionPump(model_config)
        self.thought = stage_m.ThoughtExpert(model_config)
        self.output = stage_m.TextOutputExpert(model_config)

    def context(
        self,
        batch: StrongBatch,
        *,
        zero_experts: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        expert_tokens = self.bank(batch.feature_ids)
        route_weights = batch.base.required_experts.clone()
        zero_expert_set = set(zero_experts)
        for idx, name in enumerate(stage_m.EXPERTS):
            if name in zero_expert_set:
                route_weights[:, idx] = 0.0
                expert_tokens[idx] = torch.zeros_like(expert_tokens[idx])
        route_logits = torch.where(
            route_weights > 0.5,
            torch.full_like(route_weights, 10.0),
            torch.full_like(route_weights, -10.0),
        )
        if self.mode == "direct":
            context = torch.cat([tokens * route_weights[:, idx].view(-1, 1, 1) for idx, tokens in enumerate(expert_tokens)], dim=1)
            latent = torch.zeros(batch.base.image.shape[0], stage_m.LATENT_TOKENS, context.shape[-1], device=context.device)
        elif self.mode == "moe_latent":
            pumped = self.pump(expert_tokens, route_weights)
            latent = self.thought(pumped)
            context = torch.zeros_like(latent) if disable_latent_access else latent
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return {"context": context, "latent": latent, "route_weights": route_weights, "route_logits": route_logits}

    def forward(
        self,
        batch: StrongBatch,
        *,
        zero_experts: Iterable[str] = (),
        disable_latent_access: bool = False,
    ) -> dict[str, torch.Tensor]:
        ctx = self.context(batch, zero_experts=zero_experts, disable_latent_access=disable_latent_access)
        marker = stage_m.SPECIAL_DIRECT if self.mode == "direct" else stage_m.SPECIAL_LATENT
        logits = self.output(ctx["context"], batch.base.answer_input, answer_context_marker=marker)
        return {"logits": logits, **ctx}


def loss_for_decoder(output: dict[str, torch.Tensor], batch: StrongBatch) -> torch.Tensor:
    return F.cross_entropy(output["logits"].view(-1, stage_m.VOCAB), batch.base.answer_target.view(-1), ignore_index=stage_m.PAD)


class RawSemanticClassifier(nn.Module):
    def __init__(self, config: StageNConfig) -> None:
        super().__init__()
        model_config = stage_m.MoEConfig(
            batch_size=config.batch_size,
            seed=config.seed,
            d_model=config.d_model,
            layers=config.layers,
            heads=config.heads,
            dropout=config.dropout,
            lr=config.lr,
        )
        self.vision = stage_m.VisionExpert(model_config)
        self.text = stage_m.TextRuleExpert(model_config)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, stage_m.TARGET_CLASSES),
        )
        self.task_head = nn.Linear(config.d_model * 2, len(stage_m.TASKS))

    def forward(self, batch: stage_m.Batch) -> dict[str, torch.Tensor]:
        vision = self.vision(batch.image).mean(dim=1)
        text = self.text(batch.prompt).mean(dim=1)
        hidden = torch.cat((vision, text), dim=-1)
        return {"target": self.head(hidden), "task": self.task_head(hidden)}


def train_decoder(
    model: StrongExpertDecoder,
    train_examples: list[stage_m.Example],
    val_examples: list[stage_m.Example],
    config: StageNConfig,
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
        batch = random_strong_batch(train_examples, rng=rng, batch_size=config.batch_size, device=device)
        output = model(batch)
        loss = loss_for_decoder(output, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == steps:
            metrics = evaluate_decoder_teacher_forced(model, val_examples, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "answer_exact": metrics["answer_exact"]})
            print(f"{label} step={step:4d} loss={float(loss.detach().cpu()):.4f} exact={metrics['answer_exact']:.3f}", flush=True)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def train_raw_classifier(
    model: RawSemanticClassifier,
    train_examples: list[stage_m.Example],
    val_examples: list[stage_m.Example],
    config: StageNConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for step in range(1, config.raw_classifier_steps + 1):
        batch = stage_m.random_batch(train_examples, rng=rng, batch_size=config.batch_size, device=device)
        output = model(batch)
        loss = F.cross_entropy(output["target"], batch.target_class) + 0.2 * F.cross_entropy(output["task"], batch.task)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.raw_classifier_steps:
            metrics = evaluate_raw_classifier(model, val_examples, config, device=device)
            history.append({"step": step, "loss": round(float(loss.detach().cpu()), 4), "answer_exact": metrics["answer_exact"]})
            print(f"raw_classifier step={step:4d} loss={float(loss.detach().cpu()):.4f} exact={metrics['answer_exact']:.3f}", flush=True)
    return {
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def logits_to_text(logits: torch.Tensor) -> list[str]:
    pred = logits.argmax(dim=-1).detach().cpu().tolist()
    return [stage_m.decode_tokens(row) for row in pred]


@torch.no_grad()
def evaluate_decoder_teacher_forced(
    model: StrongExpertDecoder,
    examples: list[stage_m.Example],
    config: StageNConfig,
    *,
    device: torch.device,
    zero_experts: Iterable[str] = (),
    disable_latent_access: bool = False,
) -> dict[str, object]:
    model.eval()
    predictions: list[str] = []
    route_hits: list[float] = []
    token_hits: list[float] = []
    for start in range(0, len(examples), config.batch_size):
        batch = make_strong_batch(examples[start : start + config.batch_size], device=device)
        output = model(batch, zero_experts=zero_experts, disable_latent_access=disable_latent_access)
        predictions.extend(logits_to_text(output["logits"]))
        route = (torch.sigmoid(output["route_logits"]) >= 0.5).float()
        route_hits.extend((route == batch.base.required_experts).all(dim=1).float().detach().cpu().tolist())
        token_hits.append(stage_m.masked_token_accuracy(output["logits"], batch.base.answer_target))
    metrics = text_metrics(examples, predictions)
    metrics["token_accuracy"] = statistics.fmean(token_hits)
    metrics["router_exact"] = statistics.fmean(route_hits)
    model.train()
    return metrics


@torch.no_grad()
def greedy_decoder_predictions(
    model: StrongExpertDecoder,
    examples: list[stage_m.Example],
    config: StageNConfig,
    *,
    device: torch.device,
) -> list[str]:
    model.eval()
    predictions: list[str] = []
    for start in range(0, len(examples), config.batch_size):
        batch = make_strong_batch(examples[start : start + config.batch_size], device=device)
        ctx = model.context(batch)
        marker = stage_m.SPECIAL_DIRECT if model.mode == "direct" else stage_m.SPECIAL_LATENT
        answer_input = torch.full_like(batch.base.answer_input, stage_m.PAD)
        answer_input[:, 0] = stage_m.BOS
        generated = torch.full_like(batch.base.answer_target, stage_m.PAD)
        done = torch.zeros(batch.base.answer_target.shape[0], dtype=torch.bool, device=device)
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
    model: StrongExpertDecoder,
    examples: list[stage_m.Example],
    config: StageNConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    return text_metrics(examples, greedy_decoder_predictions(model, examples, config, device=device))


@torch.no_grad()
def evaluate_raw_classifier(
    model: RawSemanticClassifier,
    examples: list[stage_m.Example],
    config: StageNConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    predictions: list[str] = []
    task_hits: list[float] = []
    target_hits: list[float] = []
    for start in range(0, len(examples), config.batch_size):
        chunk = examples[start : start + config.batch_size]
        batch = stage_m.make_batch(chunk, device=device)
        output = model(batch)
        target = output["target"].argmax(dim=-1).detach().cpu().tolist()
        task = output["task"].argmax(dim=-1)
        task_hits.extend((task == batch.task).float().detach().cpu().tolist())
        target_hits.extend((output["target"].argmax(dim=-1) == batch.target_class).float().detach().cpu().tolist())
        predictions.extend(render_answer(example, pred) for example, pred in zip(chunk, target))
    metrics = text_metrics(examples, predictions)
    metrics["task_accuracy"] = statistics.fmean(task_hits)
    metrics["target_accuracy"] = statistics.fmean(target_hits)
    model.train()
    return metrics


def deterministic_metrics(examples: list[stage_m.Example]) -> dict[str, object]:
    return text_metrics(examples, [render_answer(example, example.target_class) for example in examples])


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_prediction_cost(
    fn: Callable[[], object],
    *,
    example_count: int,
    device: torch.device,
) -> dict[str, float]:
    cuda_sync(device)
    started = time.perf_counter()
    fn()
    cuda_sync(device)
    seconds = time.perf_counter() - started
    return {"prediction_seconds": round(seconds, 4), "prediction_ms_per_example": round(seconds * 1000.0 / example_count, 4)}


def prediction_samples(
    direct: StrongExpertDecoder,
    moe: StrongExpertDecoder,
    raw: RawSemanticClassifier,
    examples: list[stage_m.Example],
    config: StageNConfig,
    *,
    device: torch.device,
) -> list[dict[str, object]]:
    direct_preds = greedy_decoder_predictions(direct, examples[:12], config, device=device)
    moe_preds = greedy_decoder_predictions(moe, examples[:12], config, device=device)
    raw.eval()
    batch = stage_m.make_batch(examples[:12], device=device)
    with torch.no_grad():
        raw_targets = raw(batch)["target"].argmax(dim=-1).detach().cpu().tolist()
    raw_preds = [render_answer(example, pred) for example, pred in zip(examples[:12], raw_targets)]
    rows = []
    for example, direct_pred, moe_pred, raw_pred in zip(examples[:12], direct_preds, moe_preds, raw_preds):
        rows.append(
            {
                "task": stage_m.TASKS[example.task],
                "prompt": example.prompt,
                "expected": example.answer,
                "strong_direct": direct_pred,
                "strong_moe_latent": moe_pred,
                "raw_classifier": raw_pred,
                "required_experts": [stage_m.EXPERTS[idx] for idx, value in enumerate(example.required_experts) if value > 0],
            }
        )
    return rows


def run_experiment(config: StageNConfig, output_path: Path) -> dict[str, object]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"stage_n_strong_experts device={device} seed={config.seed}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    train_examples = stage_m.generate_examples(config.train_size, seed=config.seed + 1)
    val_examples = stage_m.generate_examples(config.val_size, seed=config.seed + 2)
    test_examples = stage_m.generate_examples(config.test_size, seed=config.seed + 3)
    stage_m.write_samples(test_examples, output_dir=output_path.parent / "samples" / f"seed{config.seed}")

    direct = StrongExpertDecoder(config, mode="direct")
    moe = StrongExpertDecoder(config, mode="moe_latent")
    raw = RawSemanticClassifier(config)

    training = {
        "deterministic_strong_expert_renderer": {
            "training_seconds": 0.0,
            "parameter_count": 0,
            "trainable_parameter_count": 0,
        },
        "strong_direct_decoder": train_decoder(
            direct,
            train_examples,
            val_examples,
            config,
            steps=config.strong_direct_steps,
            seed=config.seed + 10,
            device=device,
            label="strong_direct",
        ),
        "strong_moe_latent_decoder": train_decoder(
            moe,
            train_examples,
            val_examples,
            config,
            steps=config.strong_moe_steps,
            seed=config.seed + 20,
            device=device,
            label="strong_moe_latent",
        ),
        "raw_image_prompt_classifier": train_raw_classifier(
            raw,
            train_examples,
            val_examples,
            config,
            seed=config.seed + 30,
            device=device,
        ),
    }

    metrics = {
        "deterministic_strong_expert_renderer": deterministic_metrics(test_examples),
        "strong_direct_decoder": evaluate_decoder_teacher_forced(direct, test_examples, config, device=device),
        "strong_moe_latent_decoder": evaluate_decoder_teacher_forced(moe, test_examples, config, device=device),
        "raw_image_prompt_classifier": evaluate_raw_classifier(raw, test_examples, config, device=device),
    }
    generation_metrics = {
        "strong_direct_decoder": evaluate_decoder_greedy(direct, test_examples, config, device=device),
        "strong_moe_latent_decoder": evaluate_decoder_greedy(moe, test_examples, config, device=device),
    }
    ablations = {
        "strong_moe_no_latent_access": evaluate_decoder_teacher_forced(
            moe, test_examples, config, device=device, disable_latent_access=True
        ),
        **{
            f"strong_moe_no_{name}_expert": evaluate_decoder_teacher_forced(
                moe, test_examples, config, device=device, zero_experts=(name,)
            )
            for name in stage_m.EXPERTS
        },
    }

    prediction_cost = {
        "deterministic_strong_expert_renderer": measure_prediction_cost(
            lambda: deterministic_metrics(test_examples), example_count=len(test_examples), device=device
        ),
        "strong_direct_decoder": measure_prediction_cost(
            lambda: evaluate_decoder_greedy(direct, test_examples, config, device=device),
            example_count=len(test_examples),
            device=device,
        ),
        "strong_moe_latent_decoder": measure_prediction_cost(
            lambda: evaluate_decoder_greedy(moe, test_examples, config, device=device),
            example_count=len(test_examples),
            device=device,
        ),
        "raw_image_prompt_classifier": measure_prediction_cost(
            lambda: evaluate_raw_classifier(raw, test_examples, config, device=device),
            example_count=len(test_examples),
            device=device,
        ),
    }

    output = {
        "experiment": "omni_transformer_stage_n_strong_experts",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "feature_vocab_size": len(ID_TO_FEATURE),
        "architecture": {
            "strong_experts": [
                "oracle symbolic vision slots",
                "oracle text/rule slots",
                "oracle spatial relation slot",
                "oracle counting slot",
                "oracle chart slot",
            ],
            "note": "Strong experts are upper-bound symbolic/perceptual experts for this synthetic domain, not pretrained real-world experts.",
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "metrics": metrics,
        "generation_metrics": generation_metrics,
        "ablations": ablations,
        "training": training,
        "prediction_cost": prediction_cost,
        "prediction_samples": prediction_samples(direct, moe, raw, test_examples, config, device=device),
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
                "training": run["training"],
                "prediction_cost": run["prediction_cost"],
            }
        )
        for key, value in numbers.items():
            buckets.setdefault(key, []).append(value)
    return {
        "experiment": "omni_transformer_stage_n_strong_experts_sweep",
        "run_count": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "stats": {key: stage_m.stat(values) for key, values in sorted(buckets.items())},
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_n_strong_experts/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_n_strong_experts/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_n_strong_experts/sweep_results.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--strong-direct-steps", type=int, default=220)
    parser.add_argument("--strong-moe-steps", type=int, default=260)
    parser.add_argument("--raw-classifier-steps", type=int, default=320)
    args = parser.parse_args()

    base_config = StageNConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        strong_direct_steps=args.strong_direct_steps,
        strong_moe_steps=args.strong_moe_steps,
        raw_classifier_steps=args.raw_classifier_steps,
    )
    if args.sweep:
        runs = []
        for seed in parse_csv_ints(args.seeds):
            config = StageNConfig(**{**asdict(base_config), "seed": seed})
            runs.append(run_experiment(config, args.output_dir / f"seed{seed}" / "result.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"wrote {args.aggregate}", flush=True)
    else:
        run_experiment(base_config, args.output)


if __name__ == "__main__":
    main()
