from __future__ import annotations

"""External C1U supervision and causal interventions.

Answers, valid choices, and support-slot ledgers are consumed only here.  They
are never passed to :class:`C1UModel.forward`.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from . import contract
from .model import C1UModel, PartitionedBoundaryOutput


@dataclass(frozen=True)
class LossWeights:
    full_answer_ce: float = float(contract.LOSS_WEIGHTS["full_answer_ce"])
    no_core_confusion: float = float(contract.LOSS_WEIGHTS["no_core_confusion"])
    support_margin_hinge: float = float(contract.LOSS_WEIGHTS["support_margin_hinge"])
    support_margin_floor: float = float(contract.LOSS_WEIGHTS["support_margin_floor"])


def _validate_targets(logits: Tensor, answers: Tensor, valid_choice_mask: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != contract.ANSWER_CLASSES:
        raise ValueError("logits must have shape [batch, 9]")
    if answers.shape != (logits.shape[0],) or answers.dtype != torch.long:
        raise ValueError("answers must be int64 [batch]")
    if valid_choice_mask.shape != logits.shape or valid_choice_mask.dtype != torch.bool:
        raise ValueError("valid_choice_mask must be bool [batch, 9]")
    if bool((valid_choice_mask.sum(dim=1) < 2).any()):
        raise ValueError("every row requires at least two valid choices")
    if bool(((answers < 0) | (answers >= logits.shape[1])).any()):
        raise ValueError("answer index is out of range")
    if not bool(valid_choice_mask.gather(1, answers.unsqueeze(1)).all()):
        raise ValueError("answer must be a valid record-local choice")


def masked_logits(logits: Tensor, valid_choice_mask: Tensor) -> Tensor:
    """Diagnostic-only local-choice view; never used by primary CE/Gates."""

    if logits.shape != valid_choice_mask.shape or valid_choice_mask.dtype != torch.bool:
        raise ValueError("valid choice mask does not match logits")
    return logits.float().masked_fill(~valid_choice_mask, -torch.inf)


def answer_margin(logits: Tensor, answers: Tensor, valid_choice_mask: Tensor) -> Tensor:
    """Gauge-invariant correct-minus-logsumexp(all-eight-raw-wrong) margin."""

    _validate_targets(logits, answers, valid_choice_mask)
    raw = logits.float()
    correct = raw.gather(1, answers.unsqueeze(1)).squeeze(1)
    wrong_mask = torch.ones_like(valid_choice_mask)
    wrong_mask.scatter_(1, answers.unsqueeze(1), False)
    wrong = raw.masked_fill(~wrong_mask, -torch.inf)
    return correct - torch.logsumexp(wrong, dim=1)


def record_local_cross_entropy(
    logits: Tensor, answers: Tensor, valid_choice_mask: Tensor
) -> Tensor:
    _validate_targets(logits, answers, valid_choice_mask)
    return F.cross_entropy(logits.float(), answers)


def no_core_uniform_kl(logits: Tensor, valid_choice_mask: Tensor) -> Tensor:
    """KL(no-core || uniform) over all raw A-I labels."""

    if logits.shape != valid_choice_mask.shape or valid_choice_mask.dtype != torch.bool:
        raise ValueError("valid choice mask does not match no-core logits")
    if bool((valid_choice_mask.sum(dim=1) < 2).any()):
        raise ValueError("no-core confusion requires at least two valid choices")
    log_prob = F.log_softmax(logits.float(), dim=1)
    probability = log_prob.exp()
    log_uniform = -torch.tensor(
        float(logits.shape[1]), device=logits.device, dtype=log_prob.dtype
    ).log()
    terms = probability * (log_prob - log_uniform)
    return terms.sum(dim=1).mean()


def replace_support_payloads(
    payloads: Tensor,
    object_present: Tensor,
    support_slots: Tensor,
) -> Tensor:
    """Return [batch, two, slots, width] leave-one-support mean replacements."""

    batch, slots, width = payloads.shape
    if object_present.shape != (batch, slots) or object_present.dtype != torch.bool:
        raise ValueError("object_present has the wrong shape")
    if support_slots.shape != (batch, 2) or support_slots.dtype != torch.long:
        raise ValueError("support_slots must be int64 [batch, 2]")
    if bool(((support_slots < 0) | (support_slots >= slots)).any()):
        raise ValueError("support slot is out of range")
    if bool((support_slots[:, 0] == support_slots[:, 1]).any()):
        raise ValueError("the two support slots must be distinct")
    if not bool(object_present.gather(1, support_slots).all()):
        raise ValueError("support slot must name a present object")
    if bool((object_present.sum(dim=1) < 2).any()):
        raise ValueError("support replacement requires at least two present objects")

    present = object_present.to(dtype=payloads.dtype).unsqueeze(-1)
    total = (payloads * present).sum(dim=1)
    count = object_present.sum(dim=1).to(dtype=payloads.dtype).unsqueeze(-1)
    interventions: list[Tensor] = []
    for support_index in range(2):
        indices = support_slots[:, support_index]
        original = payloads.gather(
            1, indices.view(batch, 1, 1).expand(batch, 1, width)
        ).squeeze(1)
        replacement = (total - original) / (count - 1.0).clamp_min(1.0)
        changed = payloads.clone()
        changed = changed.scatter(
            1,
            indices.view(batch, 1, 1).expand(batch, 1, width),
            replacement.unsqueeze(1),
        )
        interventions.append(changed)
    return torch.stack(interventions, dim=1)


def paired_counterfactual_payloads(
    payloads: Tensor,
    support_slots: Tensor,
    counterfactual_indices: Tensor,
) -> Tensor:
    """Swap one support payload from its exact single-factor group counterpart."""

    batch, slots, width = payloads.shape
    if support_slots.shape != (batch, 2) or support_slots.dtype != torch.long:
        raise ValueError("support_slots must be int64 [batch, 2]")
    if counterfactual_indices.shape != (batch, 2) or counterfactual_indices.dtype != torch.long:
        raise ValueError("counterfactual_indices must be int64 [batch, 2]")
    if bool(((support_slots < 0) | (support_slots >= slots)).any()):
        raise ValueError("support slot is out of range")
    if bool(((counterfactual_indices < 0) | (counterfactual_indices >= batch)).any()):
        raise ValueError("every training/evaluation batch must contain complete factorial groups")
    if bool(
        (
            counterfactual_indices
            == torch.arange(batch, device=counterfactual_indices.device).unsqueeze(1)
        ).any()
    ):
        raise ValueError("counterfactual row cannot be the original row")
    interventions: list[Tensor] = []
    for factor in range(2):
        row_indices = counterfactual_indices[:, factor]
        slot_indices = support_slots[:, factor]
        counterpart = payloads[row_indices]
        replacement = counterpart.gather(
            1, slot_indices.view(batch, 1, 1).expand(batch, 1, width)
        ).squeeze(1)
        changed = payloads.clone().scatter(
            1,
            slot_indices.view(batch, 1, 1).expand(batch, 1, width),
            replacement.unsqueeze(1),
        )
        interventions.append(changed)
    return torch.stack(interventions, dim=1)


def causal_outputs(
    model: C1UModel,
    boundary: PartitionedBoundaryOutput,
    answers: Tensor,
    valid_choice_mask: Tensor,
    support_slots: Tensor,
    counterfactual_indices: Tensor,
) -> dict[str, Tensor]:
    base = model.forward_from_boundary(boundary, return_trajectory=True)
    no_core = model.forward_from_boundary(boundary, disable_recurrence=True)
    replacements = paired_counterfactual_payloads(
        boundary.initial_payloads, support_slots, counterfactual_indices
    )
    ablated_logits = []
    for support_index in range(2):
        result = model.forward_from_boundary(
            boundary, initial_payloads=replacements[:, support_index]
        )
        ablated_logits.append(result["logits"])
    support_logits = torch.stack(ablated_logits, dim=1)
    base_margin = answer_margin(base["logits"], answers, valid_choice_mask)
    no_core_margin = answer_margin(no_core["logits"], answers, valid_choice_mask)
    support_margins = torch.stack(
        [
            answer_margin(support_logits[:, index], answers, valid_choice_mask)
            for index in range(2)
        ],
        dim=1,
    )
    return {
        "logits": base["logits"],
        "no_core_logits": no_core["logits"],
        "support_logits": support_logits,
        "base_margin": base_margin,
        "no_core_margin": no_core_margin,
        "no_core_margin_drop": base_margin - no_core_margin,
        "support_margins": support_margins,
        "support_margin_drops": base_margin.unsqueeze(1) - support_margins,
        "trajectory": base["trajectory"],
    }


def _forward_fields(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    missing = [field for field in contract.FORWARD_FIELDS if field not in batch]
    if missing:
        raise KeyError(f"C1U batch is missing forward fields: {missing}")
    return {field: batch[field] for field in contract.FORWARD_FIELDS}


def compute_training_loss(
    model: C1UModel,
    batch: Mapping[str, Any],
    *,
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    """Compute aligned answer/no-core/support losses without target leakage."""

    answers = batch["answers"].long()
    valid_choice_mask = batch["valid_choice_mask"].bool()
    support_slots = batch["support_slots"].long()
    if "counterfactual_indices" not in batch:
        raise KeyError("C1U causal objective requires complete-group counterfactual_indices")
    counterfactual_indices = batch["counterfactual_indices"].long()
    boundary = model.boundary(**_forward_fields(batch))
    outputs = causal_outputs(
        model,
        boundary,
        answers,
        valid_choice_mask,
        support_slots,
        counterfactual_indices,
    )
    components: dict[str, Tensor] = {
        "full_answer_ce": record_local_cross_entropy(
            outputs["logits"], answers, valid_choice_mask
        ),
        "no_core_confusion": no_core_uniform_kl(
            outputs["no_core_logits"], valid_choice_mask
        ),
        "support_margin_hinge": F.relu(
            float(weights.support_margin_floor) - outputs["support_margin_drops"]
        ).mean(),
    }
    total = (
        float(weights.full_answer_ce) * components["full_answer_ce"]
        + float(weights.no_core_confusion) * components["no_core_confusion"]
        + float(weights.support_margin_hinge) * components["support_margin_hinge"]
    )
    detached = {name: float(value.detach().cpu()) for name, value in components.items()}
    detached["total"] = float(total.detach().cpu())
    return total, detached, outputs


__all__ = [
    "LossWeights",
    "answer_margin",
    "causal_outputs",
    "compute_training_loss",
    "masked_logits",
    "no_core_uniform_kl",
    "paired_counterfactual_payloads",
    "record_local_cross_entropy",
    "replace_support_payloads",
]
