from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .artifacts import environment_record, write_json
from .data import ANSWER_LABELS, DATA_SCHEMA_VERSION, REGISTER_NAMES, load_jsonl
from .model import (
    LatentConfig,
    LatentReasoner,
    assert_no_answer_bypass,
    frozen_parameter_report,
    load_qwen35_text_only,
    text_decoder,
)
from .prompts import PROMPT_CONTRACT_VERSION, build_messages


def load_frozen_backbone(model_id: str, revision: str, device: str) -> tuple[Any, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if config.model_type == "qwen3_5":
        backbone = load_qwen35_text_only(
            model_id,
            revision,
            dtype=dtype,
            device=device,
        )
    else:
        backbone = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    if config.model_type != "qwen3_5":
        backbone.to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return tokenizer, backbone


def _cache_identity(
    records: list[dict[str, Any]],
    model_id: str,
    revision: str,
    max_length: int,
    demonstrations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    demonstrations = demonstrations or []
    return {
        "cache_schema_version": "yggdrasil.v2-a.hidden-cache.v5",
        "model_id": model_id,
        "revision": revision,
        "max_length": max_length,
        "source_representation": "last_hidden",
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "demonstration_ids": [record["example_id"] for record in demonstrations],
        "example_ids": [record["example_id"] for record in records],
    }


def encode_records(
    *,
    records: list[dict[str, Any]],
    tokenizer: Any,
    backbone: nn.Module,
    device: str,
    batch_size: int,
    max_length: int,
    cache_path: Path | None,
    model_id: str,
    revision: str,
    teacher_supervision: bool = False,
    demonstrations: list[dict[str, Any]] | None = None,
) -> dict[str, torch.Tensor]:
    identity = _cache_identity(records, model_id, revision, max_length, demonstrations)
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("identity") == identity and cached.get("teacher_supervision") == teacher_supervision:
            result = {
                "hidden": cached["hidden"],
                "mask": cached["mask"],
                "labels": cached["labels"],
                "state_labels": cached["state_labels"],
                "query_state_labels": cached["query_state_labels"],
                "state_step_mask": cached["state_step_mask"],
            }
            if teacher_supervision:
                result["teacher_hidden"] = cached["teacher_hidden"]
            return result

    hidden_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    teacher_batches: list[torch.Tensor] = []
    labels: list[int] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(record, "encoder", demonstrations),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for record in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.inference_mode():
            outputs = text_decoder(backbone)(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        hidden_batches.append(outputs.last_hidden_state.to(device="cpu", dtype=torch.float16))
        mask_batches.append(encoded["attention_mask"].to(device="cpu", dtype=torch.bool))
        if teacher_supervision:
            teacher_prompts = []
            for record in batch:
                teacher_messages = build_messages(record, "text_cot")
                teacher_messages.append(
                    {
                        "role": "assistant",
                        "content": record["trace_text"].rsplit("\nFINAL:", 1)[0],
                    }
                )
                teacher_prompts.append(
                    tokenizer.apply_chat_template(
                        teacher_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                )
            teacher_encoded = tokenizer(
                teacher_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.inference_mode():
                teacher_outputs = text_decoder(backbone)(
                    input_ids=teacher_encoded["input_ids"],
                    attention_mask=teacher_encoded["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                )
            last_positions = teacher_encoded["attention_mask"].sum(dim=-1).long() - 1
            teacher_hidden = teacher_outputs.last_hidden_state[
                torch.arange(teacher_outputs.last_hidden_state.shape[0], device=device),
                last_positions,
            ]
            teacher_batches.append(teacher_hidden.to(device="cpu", dtype=torch.float16))
        labels.extend(int(record["answer_index"]) for record in batch)

    max_state_steps = max((len(record["reasoning_steps"]) for record in records), default=1)
    state_rows: list[list[list[int]]] = []
    query_state_rows: list[list[int]] = []
    state_step_masks: list[list[bool]] = []
    for record in records:
        steps = record["reasoning_steps"]
        row = [
            [ANSWER_LABELS.index(str(step[register])) for register in REGISTER_NAMES]
            for step in steps
        ]
        if not row:
            row = [[0 for _ in REGISTER_NAMES]]
        row.extend([row[-1]] * (max_state_steps - len(row)))
        state_rows.append(row)
        state_step_masks.append(
            [True] * max(1, len(steps))
            + [False] * max(0, max_state_steps - max(1, len(steps)))
        )
        query_index = REGISTER_NAMES.index(str(record["query_register"]))
        query_state_rows.append([state[query_index] for state in row])

    payload = {
        "identity": identity,
        "hidden": torch.cat(hidden_batches),
        "mask": torch.cat(mask_batches),
        "labels": torch.tensor(labels, dtype=torch.long),
        "state_labels": torch.tensor(state_rows, dtype=torch.long),
        "query_state_labels": torch.tensor(query_state_rows, dtype=torch.long),
        "state_step_mask": torch.tensor(state_step_masks, dtype=torch.bool),
        "teacher_hidden": torch.cat(teacher_batches) if teacher_supervision else None,
        "teacher_supervision": teacher_supervision,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    result = {
        "hidden": payload["hidden"],
        "mask": payload["mask"],
        "labels": payload["labels"],
        "state_labels": payload["state_labels"],
        "query_state_labels": payload["query_state_labels"],
        "state_step_mask": payload["state_step_mask"],
    }
    if teacher_supervision:
        result["teacher_hidden"] = payload["teacher_hidden"]
    return result


def _batch_indices(size: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    if size < 1:
        raise ValueError("Cannot sample from an empty dataset")
    return torch.randint(0, size, (batch_size,), generator=generator)


def _forward(
    model: LatentReasoner,
    cache: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: str,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | list[torch.Tensor]]]:
    hidden = cache["hidden"][indices].to(device=device, non_blocking=True)
    mask = cache["mask"][indices].to(device=device, non_blocking=True)
    labels = cache["labels"][indices].to(device=device, non_blocking=True)
    output = model(hidden, mask, **kwargs)
    return output["logits"].float(), labels, output


def _trajectory_state_logits(
    model: LatentReasoner,
    trajectory: list[torch.Tensor],
) -> torch.Tensor:
    pooled = torch.stack(
        [
            (
                model.final_norm(state).mean(dim=1)
                if model.config.answer_pooling == "mean"
                else model.final_norm(state).reshape(state.shape[0], -1)
            )
            for state in trajectory
        ],
        dim=1,
    )
    return model.state_head(pooled).view(
        pooled.shape[0],
        pooled.shape[1],
        len(REGISTER_NAMES),
        len(ANSWER_LABELS),
    )


def _trajectory_state_targets(
    state_targets: torch.Tensor,
    state_step_mask: torch.Tensor,
    trajectory_steps: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state_targets.ndim != 3 or state_step_mask.ndim != 2:
        raise ValueError("state targets/mask must be [batch, steps, registers]/[batch, steps]")
    if state_targets.shape[:2] != state_step_mask.shape:
        raise ValueError("state targets and step mask have incompatible shapes")
    lengths = state_step_mask.to(device=device, dtype=torch.long).sum(dim=-1).clamp_min(1)
    step_indices = torch.arange(trajectory_steps, device=device)
    target_indices = step_indices.clamp(max=state_targets.shape[1] - 1)
    targets = state_targets.to(device=device, non_blocking=True)[:, target_indices]
    valid_mask = step_indices.unsqueeze(0) < lengths.unsqueeze(1)
    return targets, valid_mask


def verifier_policy_gradient(
    state_logits: torch.Tensor,
    state_targets: torch.Tensor,
    step_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Self-critical policy gradient for a program-verifiable state trajectory.

    The sampled policy predicts all three registers at every latent step.  The
    verifier reward combines per-register accuracy and exact full-state match,
    with an extra final-step term.  A greedy rollout is the per-example baseline;
    no answer label or trace enters the model's answer head.
    """
    if state_logits.ndim != 4 or state_targets.ndim != 3:
        raise ValueError("state logits/targets must be [batch, steps, registers, classes]/[batch, steps, registers]")
    if state_logits.shape[:3] != state_targets.shape:
        raise ValueError("state logits and targets have incompatible trajectory shapes")
    if step_mask is None:
        step_mask = torch.ones(
            state_targets.shape[:2],
            device=state_targets.device,
            dtype=torch.bool,
        )
    if step_mask.shape != state_targets.shape[:2]:
        raise ValueError("step mask must be [batch, steps]")
    step_mask = step_mask.to(device=state_logits.device, dtype=torch.bool)
    distribution = torch.distributions.Categorical(logits=state_logits)
    sampled = distribution.sample()
    greedy = state_logits.argmax(dim=-1)
    sampled_register = (sampled == state_targets).float().mean(dim=-1)
    greedy_register = (greedy == state_targets).float().mean(dim=-1)
    sampled_exact = (sampled == state_targets).all(dim=-1).float()
    greedy_exact = (greedy == state_targets).all(dim=-1).float()
    sampled_step_reward = 0.5 * sampled_register + 0.5 * sampled_exact
    greedy_step_reward = 0.5 * greedy_register + 0.5 * greedy_exact
    lengths = step_mask.sum(dim=-1).clamp_min(1)
    batch_indices = torch.arange(state_targets.shape[0], device=state_logits.device)
    last_indices = (lengths - 1).clamp(max=state_targets.shape[1] - 1)
    sampled_final = sampled_step_reward[batch_indices, last_indices]
    greedy_final = greedy_step_reward[batch_indices, last_indices]
    sampled_reward = (
        (sampled_step_reward * step_mask).sum(dim=-1) / lengths
        + 0.5 * sampled_final
    )
    greedy_reward = (
        (greedy_step_reward * step_mask).sum(dim=-1) / lengths
        + 0.5 * greedy_final
    )
    advantage = (sampled_reward - greedy_reward).detach()
    sampled_log_prob = (
        (distribution.log_prob(sampled) * step_mask.unsqueeze(-1)).sum(dim=(-1, -2))
        / (lengths * state_targets.shape[-1])
    )
    loss = -(advantage * sampled_log_prob).mean()
    stats = {
        "sampled_reward": sampled_reward.detach().mean(),
        "greedy_reward": greedy_reward.detach().mean(),
        "sampled_register_accuracy": sampled_register.detach().mean(),
        "greedy_register_accuracy": greedy_register.detach().mean(),
        "sampled_state_exact": sampled_exact.detach().mean(),
        "greedy_state_exact": greedy_exact.detach().mean(),
        "greedy_final_state_exact": greedy_exact[batch_indices, last_indices].detach().mean(),
        "policy_entropy": distribution.entropy().detach().mean(),
    }
    return loss, stats


@torch.no_grad()
def evaluate_verifier(
    model: LatentReasoner,
    cache: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    register_total = 0.0
    state_total = 0.0
    final_state_total = 0.0
    examples = 0
    for start in range(0, cache["labels"].shape[0], batch_size):
        indices = torch.arange(start, min(start + batch_size, cache["labels"].shape[0]))
        _, _, output = _forward(
            model,
            cache,
            indices,
            device,
            return_trajectory=True,
        )
        trajectory = output.get("trajectory")
        if trajectory is None:
            raise RuntimeError("verifier evaluation requires trajectory output")
        logits = _trajectory_state_logits(model, trajectory)
        targets = cache["state_labels"][indices].to(device=device, non_blocking=True)
        target_mask = cache["state_step_mask"][indices].to(
            device=device,
            non_blocking=True,
        )
        targets, target_mask = _trajectory_state_targets(
            targets,
            target_mask,
            logits.shape[1],
            device,
        )
        predictions = logits.argmax(dim=-1)
        register_accuracy = (predictions == targets).float().mean(dim=-1)
        exact = (predictions == targets).all(dim=-1).float()
        lengths = target_mask.sum(dim=-1).clamp_min(1)
        batch_indices = torch.arange(targets.shape[0], device=device)
        last_indices = (lengths - 1).clamp(max=targets.shape[1] - 1)
        register_total += float((register_accuracy * target_mask).sum(dim=-1).div(lengths).sum())
        state_total += float((exact * target_mask).sum(dim=-1).div(lengths).sum())
        final_state_total += float(exact[batch_indices, last_indices].sum())
        examples += int(targets.shape[0])
    return {
        "greedy_register_accuracy": register_total / max(1, examples),
        "greedy_state_exact": state_total / max(1, examples),
        "greedy_final_state_exact": final_state_total / max(1, examples),
        "examples": examples,
    }


@torch.no_grad()
def evaluate(
    model: LatentReasoner,
    cache: dict[str, torch.Tensor],
    device: str,
    batch_size: int,
    intervention: str = "none",
    intervention_step: int | None = None,
    truncate_steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    for start in range(0, cache["labels"].shape[0], batch_size):
        indices = torch.arange(start, min(start + batch_size, cache["labels"].shape[0]))
        logits, labels, _ = _forward(
            model,
            cache,
            indices,
            device,
            intervention=intervention,
            intervention_step=intervention_step,
            truncate_steps=truncate_steps,
        )
        loss_total += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == labels).sum())
        total += int(labels.numel())
    return {
        "exact_match": correct / max(1, total),
        "loss": loss_total / max(1, total),
        "examples": total,
    }


def train_latent_reasoner(
    *,
    data_dir: Path,
    output_dir: Path,
    model_id: str,
    revision: str,
    seed: int,
    device: str,
    latent_config: LatentConfig,
    steps: int,
    batch_size: int,
    encoder_batch_size: int,
    learning_rate: float,
    max_length: int,
    max_train_examples: int | None,
    max_eval_examples: int | None,
    resume: bool,
    teacher_hidden_weight: float = 0.0,
    state_supervision_weight: float = 0.0,
    encoder_demonstrations: int = 0,
    query_state_supervision_weight: float = 0.0,
    verifier_rl_weight: float = 0.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        state_supervision_weight > 0
        or query_state_supervision_weight > 0
        or verifier_rl_weight > 0
    ) and latent_config.recurrent_steps < 1:
        raise ValueError("trajectory supervision requires at least one recurrent step")
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    tokenizer, backbone = load_frozen_backbone(model_id, revision, device)
    if encoder_demonstrations < 0:
        raise ValueError("encoder_demonstrations must be non-negative")
    demonstrations = load_jsonl(data_dir / "train.jsonl")[:encoder_demonstrations]
    expected_heads = int(text_decoder(backbone).config.num_attention_heads)
    if latent_config.cross_attention_heads != expected_heads:
        raise ValueError(
            "cross_attention_heads must match the pinned backbone attention heads "
            f"({expected_heads}), got {latent_config.cross_attention_heads}"
        )
    reasoner = LatentReasoner(backbone, latent_config).to(device)
    bypass_check = assert_no_answer_bypass(reasoner)
    if not bypass_check["passed"]:
        raise RuntimeError(f"Answer bypass check failed: {bypass_check}")

    split_names = ("train", "validation", "test", "composition_heldout", "length_heldout")
    caches: dict[str, dict[str, torch.Tensor]] = {}
    unique_examples: dict[str, int] = {}
    encoding_started = time.perf_counter()
    for split in split_names:
        records = load_jsonl(data_dir / f"{split}.jsonl")
        limit = max_train_examples if split == "train" else max_eval_examples
        if limit is not None:
            records = records[:limit]
        unique_examples[split] = len(records)
        caches[split] = encode_records(
            records=records,
            tokenizer=tokenizer,
            backbone=backbone,
            device=device,
            batch_size=encoder_batch_size,
            max_length=max_length,
            cache_path=output_dir / "hidden-cache" / f"{split}.pt",
            model_id=model_id,
            revision=revision,
            teacher_supervision=teacher_hidden_weight > 0,
            demonstrations=demonstrations,
        )
    encoding_seconds = time.perf_counter() - encoding_started

    optimizer = torch.optim.AdamW(reasoner.parameters(), lr=learning_rate, weight_decay=0.01)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    start_step = 0
    best_validation = -math.inf
    resumed_from: str | None = None
    if resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["latent_config"] != asdict(latent_config):
            raise ValueError("Checkpoint latent config does not match the requested run")
        reasoner.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_validation = float(checkpoint["best_validation"])
        resumed_from = str(latest_path)

    generator = torch.Generator(device="cpu").manual_seed(seed + start_step)
    loss_history: list[dict[str, float]] = []
    training_started = time.perf_counter()
    reasoner.train()
    for step in range(start_step + 1, steps + 1):
        indices = _batch_indices(caches["train"]["labels"].shape[0], batch_size, generator)
        logits, labels, output = _forward(
            reasoner,
            caches["train"],
            indices,
            device,
            return_trajectory=(
                state_supervision_weight > 0
                or query_state_supervision_weight > 0
                or verifier_rl_weight > 0
            ),
        )
        loss = nn.functional.cross_entropy(logits, labels)
        state_loss_value: float | None = None
        query_state_loss_value: float | None = None
        verifier_rl_loss_value: float | None = None
        verifier_stats: dict[str, torch.Tensor] | None = None
        if state_supervision_weight > 0:
            trajectory = output["trajectory"]
            pooled_trajectory = torch.stack(
                [
                    (
                        reasoner.final_norm(state).mean(dim=1)
                        if reasoner.config.answer_pooling == "mean"
                        else reasoner.final_norm(state).reshape(state.shape[0], -1)
                    )
                    for state in trajectory
                ],
                dim=1,
            )
            state_logits = reasoner.state_head(pooled_trajectory).view(
                pooled_trajectory.shape[0],
                pooled_trajectory.shape[1],
                len(REGISTER_NAMES),
                len(ANSWER_LABELS),
            )
            state_targets = caches["train"]["state_labels"][indices].to(device=device, non_blocking=True)
            state_step_mask = caches["train"]["state_step_mask"][indices].to(
                device=device,
                non_blocking=True,
            )
            state_targets, state_step_mask = _trajectory_state_targets(
                state_targets,
                state_step_mask,
                state_logits.shape[1],
                device,
            )
            state_token_loss = nn.functional.cross_entropy(
                state_logits.reshape(-1, len(ANSWER_LABELS)),
                state_targets.reshape(-1),
                reduction="none",
            ).view(state_logits.shape[0], state_logits.shape[1], len(REGISTER_NAMES))
            state_loss = (
                state_token_loss * state_step_mask.unsqueeze(-1)
            ).sum() / (state_step_mask.sum() * len(REGISTER_NAMES)).clamp_min(1)
            state_loss_value = float(state_loss.detach())
            loss = loss + state_supervision_weight * state_loss
        if query_state_supervision_weight > 0:
            trajectory = output.get("trajectory")
            if trajectory is None:
                raise RuntimeError("query state supervision requires trajectory output")
            pooled_trajectory = torch.stack(
                [
                    (
                        reasoner.final_norm(state).mean(dim=1)
                        if reasoner.config.answer_pooling == "mean"
                        else reasoner.final_norm(state).reshape(state.shape[0], -1)
                    )
                    for state in trajectory
                ],
                dim=1,
            )
            query_logits = reasoner.query_state_head(pooled_trajectory)
            query_targets = caches["train"]["query_state_labels"][indices].to(
                device=device,
                non_blocking=True,
            )
            query_targets, query_step_mask = _trajectory_state_targets(
                query_targets.unsqueeze(-1),
                caches["train"]["state_step_mask"][indices].to(
                    device=device,
                    non_blocking=True,
                ),
                query_logits.shape[1],
                device,
            )
            query_targets = query_targets.squeeze(-1)
            query_token_loss = nn.functional.cross_entropy(
                query_logits.reshape(-1, len(ANSWER_LABELS)),
                query_targets.reshape(-1),
                reduction="none",
            ).view(query_logits.shape[0], query_logits.shape[1])
            query_state_loss = (
                query_token_loss * query_step_mask
            ).sum() / query_step_mask.sum().clamp_min(1)
            query_state_loss_value = float(query_state_loss.detach())
            loss = loss + query_state_supervision_weight * query_state_loss
        if verifier_rl_weight > 0:
            trajectory = output.get("trajectory")
            if trajectory is None:
                raise RuntimeError("verifier policy gradient requires trajectory output")
            state_logits = _trajectory_state_logits(reasoner, trajectory)
            state_targets = caches["train"]["state_labels"][indices].to(
                device=device,
                non_blocking=True,
            )
            state_step_mask = caches["train"]["state_step_mask"][indices].to(
                device=device,
                non_blocking=True,
            )
            state_targets, state_step_mask = _trajectory_state_targets(
                state_targets,
                state_step_mask,
                state_logits.shape[1],
                device,
            )
            verifier_rl_loss, verifier_stats = verifier_policy_gradient(
                state_logits,
                state_targets,
                state_step_mask,
            )
            verifier_rl_loss_value = float(verifier_rl_loss.detach())
            loss = loss + verifier_rl_weight * verifier_rl_loss
        if teacher_hidden_weight > 0:
            teacher = caches["train"]["teacher_hidden"][indices].to(device=device, dtype=torch.float32)
            predicted = reasoner.final_norm(output["final_state"]).mean(dim=1).float()
            teacher_loss = 1.0 - nn.functional.cosine_similarity(predicted, teacher, dim=-1).mean()
            loss = loss + teacher_hidden_weight * teacher_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
        optimizer.step()
        history_item: dict[str, float] = {"step": step, "loss": float(loss.detach())}
        if state_loss_value is not None:
            history_item["state_loss"] = state_loss_value
        if query_state_loss_value is not None:
            history_item["query_state_loss"] = query_state_loss_value
        if verifier_rl_loss_value is not None and verifier_stats is not None:
            history_item["verifier_rl_loss"] = verifier_rl_loss_value
            history_item.update(
                {
                    f"verifier_{name}": float(value)
                    for name, value in verifier_stats.items()
                }
            )
        loss_history.append(history_item)

        should_checkpoint = step == steps or step % max(1, min(100, steps // 4 or 1)) == 0
        if should_checkpoint:
            validation = evaluate(reasoner, caches["validation"], device, batch_size)
            checkpoint = {
                "schema_version": "yggdrasil.v2-a.checkpoint.v1",
                "step": step,
                "best_validation": max(best_validation, validation["exact_match"]),
                "latent_config": asdict(latent_config),
                "model_id": model_id,
                "revision": revision,
                "seed": seed,
                "model": reasoner.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, latest_path)
            if validation["exact_match"] >= best_validation:
                best_validation = validation["exact_match"]
                torch.save(checkpoint, best_path)
            reasoner.train()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    fit_metrics = evaluate(reasoner, caches["train"], device, batch_size)
    full_metrics = {
        split: evaluate(reasoner, cache, device, batch_size)
        for split, cache in caches.items()
        if split != "train"
    }
    middle_step = max(0, latent_config.recurrent_steps // 2 - 1)
    causal_metrics = {
        "no_latent": evaluate(reasoner, caches["test"], device, batch_size, intervention="zero"),
        "shuffled_latent_mid_step": evaluate(
            reasoner,
            caches["test"],
            device,
            batch_size,
            intervention="batch_shuffle",
            intervention_step=middle_step,
        ),
        "trajectory_half_length": evaluate(
            reasoner,
            caches["test"],
            device,
            batch_size,
            truncate_steps=max(1, latent_config.recurrent_steps // 2),
        ),
    }
    verifier_metrics = (
        evaluate_verifier(reasoner, caches["test"], device, batch_size)
        if verifier_rl_weight > 0
        else None
    )
    parameters = {**frozen_parameter_report(backbone), **reasoner.parameter_report()}
    processed_examples = max(0, steps - start_step) * batch_size
    result = {
        "schema_version": "yggdrasil.v2-a.results.v1",
        "evidence_level": "smoke" if steps <= 10 else "probe",
        "experiment": "V2-A1 continuous latent recurrence",
        "data_schema_version": DATA_SCHEMA_VERSION,
        "model": {
            "model_id": model_id,
            "revision": revision,
            "license": "apache-2.0",
            "backbone_dtype": str(next(backbone.parameters()).dtype),
            "latent_dtype": str(next(reasoner.parameters()).dtype),
            "backbone_frozen": True,
        },
        "latent_config": asdict(latent_config),
        "recurrent_layer_indices": list(reasoner.recurrent_layer_indices),
        "training": {
            "seed": seed,
            "steps_requested": steps,
            "start_step": start_step,
            "steps_completed": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "teacher_hidden_weight": teacher_hidden_weight,
            "state_supervision_weight": state_supervision_weight,
            "encoder_demonstrations": encoder_demonstrations,
            "query_state_supervision_weight": query_state_supervision_weight,
            "verifier_rl_weight": verifier_rl_weight,
            "max_length": max_length,
            "loss_history": loss_history,
        },
        "data": {
            "unique_examples": unique_examples,
            "processed_examples_this_run": processed_examples,
            "processed_token_upper_bound_this_run": processed_examples * max_length,
        },
        "parameters": parameters,
        "fit": fit_metrics,
        "quality": full_metrics,
        "causal": causal_metrics,
        "integrity": bypass_check,
        "cost": {
            "encoder_cache_seconds": encoding_seconds,
            "training_seconds": training_seconds,
            "latent_transitions_this_run": processed_examples * latent_config.recurrent_steps,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
            "flops": "not yet measured; no quality-cost claim allowed",
            "kv_cache": "frozen encoder only; recurrent latent core has fixed K and no autoregressive KV cache",
        },
        "recovery": {
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
            "resumed_from": resumed_from,
            "resume_supported": True,
        },
        "environment": environment_record(torch),
        "boundaries": [
            "Smoke/probe validates wiring and trainability only.",
            "A quality-cost advantage requires matched A0 baselines and formal multi-seed evidence.",
            "Audit readout is intentionally absent until V2-A3.",
        ],
    }
    if verifier_metrics is not None:
        result["verifier"] = verifier_metrics
    write_json(output_dir / "results.json", result)
    return result
