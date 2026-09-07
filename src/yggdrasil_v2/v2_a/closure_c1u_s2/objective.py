from __future__ import annotations

"""Matched external objective over public-object interventions for K1/K8."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from yggdrasil_v2.v2_a.closure_c1u import contract as c1u_contract
from yggdrasil_v2.v2_a.closure_c1u.objective import (
    answer_margin,
    no_core_uniform_kl,
    paired_counterfactual_payloads,
    record_local_cross_entropy,
)

from . import contract
from .model import S2BoundaryOutput, S2Model


@dataclass(frozen=True)
class LossWeights:
    full_answer_ce: float = float(c1u_contract.LOSS_WEIGHTS["full_answer_ce"])
    no_core_confusion: float = float(c1u_contract.LOSS_WEIGHTS["no_core_confusion"])
    support_margin_hinge: float = float(
        c1u_contract.LOSS_WEIGHTS["support_margin_hinge"]
    )
    support_margin_floor: float = float(
        c1u_contract.LOSS_WEIGHTS["support_margin_floor"]
    )


def causal_outputs(
    model: S2Model,
    boundary: S2BoundaryOutput,
    answers: Tensor,
    valid_choice_mask: Tensor,
    support_slots: Tensor,
    counterfactual_indices: Tensor,
) -> dict[str, Tensor]:
    base = model.forward_from_boundary(boundary, return_trajectory=True)
    no_core = model.forward_from_boundary(boundary, disable_recurrence=True)
    replacements = paired_counterfactual_payloads(
        boundary.public_initial_payloads,
        support_slots,
        counterfactual_indices,
    )
    support_logits = []
    for support_index in range(2):
        changed = boundary.rebuild(replacements[:, support_index])
        support_logits.append(model.forward_from_boundary(changed)["logits"])
    stacked = torch.stack(support_logits, dim=1)
    base_margin = answer_margin(base["logits"], answers, valid_choice_mask)
    no_core_margin = answer_margin(no_core["logits"], answers, valid_choice_mask)
    support_margins = torch.stack(
        [
            answer_margin(stacked[:, index], answers, valid_choice_mask)
            for index in range(2)
        ],
        dim=1,
    )
    return {
        "logits": base["logits"],
        "no_core_logits": no_core["logits"],
        "support_logits": stacked,
        "base_margin": base_margin,
        "no_core_margin": no_core_margin,
        "no_core_margin_drop": base_margin - no_core_margin,
        "support_margins": support_margins,
        "support_margin_drops": base_margin.unsqueeze(1) - support_margins,
        "trajectory": base["trajectory"],
    }


def _forward_fields(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    missing = [field for field in c1u_contract.FORWARD_FIELDS if field not in batch]
    if missing:
        raise KeyError(f"C1U S2 batch lacks public forward fields: {missing}")
    return {field: batch[field] for field in c1u_contract.FORWARD_FIELDS}


def compute_training_loss(
    model: S2Model,
    batch: Mapping[str, Any],
    *,
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, float], dict[str, Tensor]]:
    answers = batch["answers"].long()
    valid = batch["valid_choice_mask"].bool()
    boundary = model.boundary(**_forward_fields(batch))
    outputs = causal_outputs(
        model,
        boundary,
        answers,
        valid,
        batch["support_slots"].long(),
        batch["counterfactual_indices"].long(),
    )
    components = {
        "full_answer_ce": record_local_cross_entropy(outputs["logits"], answers, valid),
        "no_core_confusion": no_core_uniform_kl(outputs["no_core_logits"], valid),
        "support_margin_hinge": F.relu(
            float(weights.support_margin_floor) - outputs["support_margin_drops"]
        ).mean(),
    }
    total = (
        float(weights.full_answer_ce) * components["full_answer_ce"]
        + float(weights.no_core_confusion) * components["no_core_confusion"]
        + float(weights.support_margin_hinge)
        * components["support_margin_hinge"]
    )
    detached = {name: float(value.detach().cpu()) for name, value in components.items()}
    detached["total"] = float(total.detach().cpu())
    return total, detached, outputs


__all__ = ["LossWeights", "causal_outputs", "compute_training_loss"]
