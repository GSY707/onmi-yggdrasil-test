from __future__ import annotations

"""Functional-K and geometry instruments for C1S qualification."""

import math
from typing import Any

import torch
from torch import Tensor


def effective_slot_count(weights: Tensor) -> Tensor:
    if weights.ndim != 2 or weights.shape[1] < 1:
        raise ValueError("weights must have shape [batch, slots]")
    probabilities = weights.float().clamp_min(1.0e-12)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return entropy.exp()


def linear_readout(
    payloads: Tensor,
    weights: Tensor,
    head_weight: Tensor,
    head_bias: Tensor | None = None,
) -> Tensor:
    if payloads.ndim != 3 or weights.shape != payloads.shape[:2]:
        raise ValueError("payloads/weights must be [batch, slots, width] and [batch, slots]")
    selected = torch.einsum("bk,bkd->bd", weights, payloads)
    logits = selected @ head_weight.transpose(0, 1)
    return logits if head_bias is None else logits + head_bias


def mean_replace_effects(
    payloads: Tensor,
    weights: Tensor,
    head_weight: Tensor,
    head_bias: Tensor | None = None,
) -> dict[str, Any]:
    baseline = linear_readout(payloads, weights, head_weight, head_bias)
    batch, slots, _ = payloads.shape
    effects = payloads.new_zeros((batch, slots), dtype=torch.float32)
    for slot in range(slots):
        replaced = payloads.clone()
        if slots == 1:
            replacement = torch.zeros_like(payloads[:, 0])
        else:
            keep = [index for index in range(slots) if index != slot]
            replacement = payloads[:, keep].mean(dim=1)
        replaced[:, slot] = replacement
        logits = linear_readout(replaced, weights, head_weight, head_bias)
        effects[:, slot] = torch.linalg.vector_norm((logits - baseline).float(), dim=-1)
    relevant_index = weights.argmax(dim=-1)
    relevant = effects.gather(1, relevant_index.unsqueeze(1)).squeeze(1)
    irrelevant_mask = torch.ones_like(effects, dtype=torch.bool)
    irrelevant_mask.scatter_(1, relevant_index.unsqueeze(1), False)
    if slots == 1:
        irrelevant_max = torch.zeros_like(relevant)
    else:
        irrelevant_max = effects.masked_fill(~irrelevant_mask, float("-inf")).max(dim=1).values
    contrast = relevant / irrelevant_max.clamp_min(1.0e-12)
    return {
        "effects": effects.detach().cpu().tolist(),
        "relevant_effect": relevant.detach().cpu().tolist(),
        "irrelevant_max_effect": irrelevant_max.detach().cpu().tolist(),
        "ownership_contrast": contrast.detach().cpu().tolist(),
        "effective_slot_count": effective_slot_count(weights).detach().cpu().tolist(),
        "single_slot_null": slots == 1,
    }


def geometry_summary(payloads: Tensor) -> dict[str, float]:
    if payloads.ndim != 3 or payloads.shape[1] < 1:
        raise ValueError("payloads must have shape [batch, slots, width]")
    value = payloads.float()
    common = value.mean(dim=1, keepdim=True)
    centered = value - common
    total_energy = value.square().sum(dim=(1, 2)).mean()
    common_energy = common.expand_as(value).square().sum(dim=(1, 2)).mean()
    centered_energy = centered.square().sum(dim=(1, 2)).mean()
    if value.shape[1] == 1:
        off_diagonal = value.new_tensor(1.0)
    else:
        normalized = torch.nn.functional.normalize(value, dim=-1, eps=1.0e-8)
        cosine = torch.einsum("bkd,bjd->bkj", normalized, normalized)
        mask = ~torch.eye(value.shape[1], dtype=torch.bool, device=value.device)
        off_diagonal = cosine[:, mask].mean()
    return {
        "total_energy": float(total_energy.item()),
        "common_energy": float(common_energy.item()),
        "slot_centered_energy": float(centered_energy.item()),
        "centered_to_total_ratio": float((centered_energy / total_energy.clamp_min(1.0e-12)).item()),
        "mean_off_diagonal_cosine": float(off_diagonal.item()),
    }


def registered_control_report() -> dict[str, Any]:
    """Evaluate the frozen addressed positive, duplicate, uniform, and K=1 controls."""

    payloads = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    head = torch.eye(4, dtype=torch.float32)
    query_zero = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    query_one = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    positive = mean_replace_effects(payloads, query_zero, head)
    swap_delta = torch.linalg.vector_norm(
        linear_readout(payloads, query_zero, head) - linear_readout(payloads, query_one, head),
        dim=-1,
    )

    duplicate = torch.ones((1, 4, 4), dtype=torch.float32)
    duplicate_swap_delta = torch.linalg.vector_norm(
        linear_readout(duplicate, query_zero, head) - linear_readout(duplicate, query_one, head),
        dim=-1,
    )

    uniform = torch.full((1, 4), 0.25, dtype=torch.float32)
    uniform_null = mean_replace_effects(payloads, uniform, head)
    k1 = mean_replace_effects(torch.ones((1, 1, 4)), torch.ones((1, 1)), head)

    return {
        "addressed_positive": positive,
        "query_swap_logit_l2": float(swap_delta.item()),
        "duplicate_query_swap_logit_l2": float(duplicate_swap_delta.item()),
        "legacy_uniform_mean_null": uniform_null,
        "k1_null": k1,
        "expected_query_swap_logit_l2": math.sqrt(2.0),
    }


__all__ = [
    "effective_slot_count",
    "geometry_summary",
    "linear_readout",
    "mean_replace_effects",
    "registered_control_report",
]
