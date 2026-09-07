from __future__ import annotations

"""External supervision for the frozen C1S addressed-workspace model.

Targets are consumed here, never by :class:`C1SModel.forward`.  The deployment
graph therefore remains source-hidden/source-mask only.
"""

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from yggdrasil_v2.v2_a.closure_c1s.model import BoundaryOutput, C1SModel


STATE_FEATURE_NAMES = (
    "semantic_0",
    "semantic_1",
    "semantic_2",
    "semantic_3",
    "semantic_4",
    "semantic_5",
    "semantic_6",
    "semantic_7",
)


@dataclass(frozen=True)
class LossWeights:
    answer: float = 1.0
    slot_identity: float = 0.50
    route: float = 1.0
    query_owner: float = 1.0
    presence: float = 0.20
    span_owner: float = 0.50
    state: float = 1.0
    operation_active: float = 0.20
    no_core_margin: float = 0.50
    wrong_start_margin: float = 0.50
    payload_binding_margin: float = 0.50
    operation_binding_margin: float = 0.50
    target_shuffle_margin: float = 0.50
    relevant_margin: float = 0.50
    irrelevant_consistency: float = 0.10


@dataclass(frozen=True)
class CausalMargins:
    no_core_answer_margin: float = 0.50
    wrong_start_answer_margin: float = 0.50
    payload_binding: float = 0.50
    operation_binding: float = 0.50
    target_shuffle: float = 0.50
    relevant_answer_margin: float = 0.50


class SuccessorDiagnosticHeads(nn.Module):
    """Training-only span alignment and decision-state decoders."""

    def __init__(self, source_width: int, address_width: int, auxiliary_width: int) -> None:
        super().__init__()
        self.span_projection = nn.Linear(source_width, address_width, bias=False)
        self.state_decoder = nn.Linear(auxiliary_width, len(STATE_FEATURE_NAMES))


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _owner_mse(weights: Tensor, owners: Tensor, valid: Tensor) -> Tensor:
    slots = weights.shape[-1]
    targets = F.one_hot(owners.clamp_min(0), num_classes=slots).to(dtype=weights.dtype)
    error = (weights - targets).square().sum(dim=-1)
    return _masked_mean(error, valid)


def _span_pool(source_hidden: Tensor, span_start: Tensor, span_end: Tensor, present: Tensor) -> Tensor:
    positions = torch.arange(source_hidden.shape[1], device=source_hidden.device).view(1, 1, -1)
    mask = (
        (positions >= span_start.unsqueeze(-1))
        & (positions < span_end.unsqueeze(-1))
        & present.unsqueeze(-1)
    )
    weights = mask.to(dtype=source_hidden.dtype)
    pooled = torch.einsum("bks,bsd->bkd", weights, source_hidden)
    return pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)


def _span_alignment_loss(addresses: Tensor, span_keys: Tensor, present: Tensor) -> Tensor:
    slots = addresses.shape[1]
    if slots == 1:
        return addresses.sum() * 0.0
    left = F.normalize(addresses.float(), dim=-1)
    right = F.normalize(span_keys.float(), dim=-1)
    logits = torch.einsum("bkd,bjd->bkj", left, right) / 0.10
    pair_valid = present.unsqueeze(1) & present.unsqueeze(2)
    logits = logits.masked_fill(~pair_valid, -1.0e4)
    labels = torch.arange(slots, device=addresses.device).view(1, slots).expand(addresses.shape[0], -1)
    row_loss = F.cross_entropy(logits.reshape(-1, slots), labels.reshape(-1), reduction="none")
    col_loss = F.cross_entropy(logits.transpose(1, 2).reshape(-1, slots), labels.reshape(-1), reduction="none")
    flat_present = present.reshape(-1)
    return _masked_mean(row_loss + col_loss, flat_present)


def slot_answer_logits(model: C1SModel, payloads: Tensor) -> Tensor:
    return model.answer_head(model.final_norm(payloads))


def _selected_answer_logits(logits: Tensor, answers: Tensor) -> Tensor:
    return logits.gather(1, answers.view(-1, 1)).squeeze(1)


def _answer_margin(logits: Tensor, answers: Tensor) -> Tensor:
    correct = _selected_answer_logits(logits, answers)
    wrong = logits.float().clone()
    wrong.scatter_(1, answers.view(-1, 1), -torch.inf)
    return correct.float() - torch.logsumexp(wrong, dim=1)


def _replace_one(payloads: Tensor, index: Tensor, replacement: Tensor) -> Tensor:
    result = payloads.clone()
    scatter_index = index.view(-1, 1, 1).expand(-1, 1, payloads.shape[-1])
    return result.scatter(1, scatter_index, replacement.unsqueeze(1))


def _replacement_states(
    payloads: Tensor, present: Tensor, relevant: Tensor, irrelevant: Tensor,
) -> tuple[Tensor, Tensor]:
    weights = present.to(dtype=payloads.dtype).unsqueeze(-1)
    total = (payloads * weights).sum(dim=1)
    counts = present.sum(dim=1).clamp_min(2).to(dtype=payloads.dtype).unsqueeze(-1)
    relevant_payload = payloads.gather(
        1, relevant.view(-1, 1, 1).expand(-1, 1, payloads.shape[-1])
    ).squeeze(1)
    relevant_replacement = (total - relevant_payload) / (counts - 1.0)
    relevant_state = _replace_one(payloads, relevant, relevant_replacement)

    irrelevant_state = payloads.clone()
    eligible = irrelevant >= 0
    if bool(eligible.any().item()):
        mean_present = total / present.sum(dim=1).clamp_min(1).to(dtype=payloads.dtype).unsqueeze(-1)
        replaced = _replace_one(payloads, irrelevant.clamp_min(0), mean_present)
        irrelevant_state = torch.where(eligible.view(-1, 1, 1), replaced, irrelevant_state)
    return relevant_state, irrelevant_state


def intervention_outputs(
    model: C1SModel,
    boundary: BoundaryOutput,
    base: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Compute registered interventions while reusing the public Boundary."""

    answers = targets["answers"].long()
    payloads = base["final_payloads"]
    addresses = boundary.addresses
    query_key = boundary.query_key
    no_core = model.forward_from_boundary(boundary, disable_recurrence=True)

    payload_zero = model.forward_from_boundary(replace(boundary, payloads=torch.zeros_like(boundary.payloads)))
    operation_zero = model.forward_from_boundary(
        replace(boundary, operation_states=torch.zeros_like(boundary.operation_states))
    )

    if model.config.slots > 1:
        permutation = torch.arange(model.config.slots - 1, -1, -1, device=payloads.device)
        wrong = model.forward_from_boundary(boundary, wrong_start_permutation=permutation)
        payload_shuffle = model.forward_from_boundary(
            replace(boundary, payloads=boundary.payloads.roll(1, dims=1))
        )
        operation_shuffle = model.forward_from_boundary(
            replace(boundary, operation_states=boundary.operation_states.roll(1, dims=1))
        )
        relevant_state, irrelevant_state = _replacement_states(
            payloads,
            targets["presence"].bool(),
            targets["query_owner"].long(),
            targets.get("irrelevant_owner", torch.full_like(targets["query_owner"], -1)).long(),
        )
        relevant_logits, _ = model.logits_from_state(relevant_state, addresses, query_key)
        irrelevant_logits, _ = model.logits_from_state(irrelevant_state, addresses, query_key)
        shuffled_boundary = replace(boundary, target_weights=boundary.target_weights.roll(1, dims=-1))
        target_shuffle = model.forward_from_boundary(shuffled_boundary)
    else:
        wrong = no_core
        payload_shuffle = payload_zero
        operation_shuffle = operation_zero
        relevant_logits = no_core["logits"]
        irrelevant_logits = base["logits"]
        target_shuffle = base

    outputs = {
        "no_core_logits": no_core["logits"],
        "wrong_start_logits": wrong["logits"],
        "payload_zero_logits": payload_zero["logits"],
        "payload_shuffle_logits": payload_shuffle["logits"],
        "operation_zero_logits": operation_zero["logits"],
        "operation_shuffle_logits": operation_shuffle["logits"],
        "relevant_replace_logits": relevant_logits,
        "irrelevant_replace_logits": irrelevant_logits,
        "target_shuffle_logits": target_shuffle["logits"],
    }
    outputs["base_answer_margin"] = _answer_margin(base["logits"], answers)
    for name in (
        "no_core", "wrong_start", "payload_zero", "payload_shuffle",
        "operation_zero", "operation_shuffle", "target_shuffle",
        "relevant_replace", "irrelevant_replace",
    ):
        outputs[f"{name}_answer_margin"] = _answer_margin(outputs[f"{name}_logits"], answers)
    return outputs


def compute_training_loss(
    model: C1SModel,
    diagnostic_heads: SuccessorDiagnosticHeads,
    batch: Mapping[str, Tensor],
    *,
    weights: LossWeights = LossWeights(),
    margins: CausalMargins = CausalMargins(),
    causal: bool = True,
) -> tuple[Tensor, dict[str, float], dict[str, Any]]:
    """Return total loss, detached components, and reusable forward outputs."""

    source_hidden = batch["source_hidden"]
    source_mask = batch["source_mask"].bool()
    answers = batch["answers"].long()
    boundary = model.boundary(source_hidden, source_mask)
    base = model.forward_from_boundary(boundary, return_trajectory=True, return_auxiliary=True)
    auxiliary = base.get("auxiliary")
    if not isinstance(auxiliary, Mapping):
        raise RuntimeError("C1S successor training requires training auxiliary heads")

    components: dict[str, Tensor] = {}
    components["answer"] = F.cross_entropy(base["logits"].float(), answers)
    supported = batch.get("mechanism_supported", torch.ones_like(answers, dtype=torch.bool)).bool()

    per_slot_logits = slot_answer_logits(model, base["final_payloads"])
    object_labels = batch["object_labels"].long()
    valid_labels = (object_labels >= 0) & supported.unsqueeze(1)
    if bool(valid_labels.any().item()):
        components["slot_identity"] = F.cross_entropy(
            per_slot_logits.float()[valid_labels], object_labels[valid_labels]
        )
    else:
        components["slot_identity"] = per_slot_logits.sum() * 0.0

    active = batch["operation_active"].bool() & supported.unsqueeze(1)
    components["source_route"] = _owner_mse(
        boundary.source_weights, batch["source_owner"].long(), active
    )
    components["target_route"] = _owner_mse(
        boundary.target_weights, batch["target_owner"].long(), active
    )
    query_valid = supported
    components["query_owner"] = _owner_mse(
        base["query_weights"], batch["query_owner"].long(), query_valid
    )
    presence_loss = F.binary_cross_entropy_with_logits(
        boundary.presence_logits.float(), batch["presence"].float(), reduction="none"
    )
    components["presence"] = _masked_mean(
        presence_loss, supported.unsqueeze(1).expand_as(presence_loss)
    )
    # ``operation_active`` is already a sigmoid probability in the frozen
    # S0 model.  Manual FP32 Bernoulli NLL is safe under CUDA autocast, while
    # ``binary_cross_entropy`` intentionally rejects autocast inputs.
    active_probability = boundary.operation_active.float().clamp(1.0e-5, 1.0 - 1.0e-5)
    active_target = batch["operation_active"].float()
    active_loss = -(
        active_target * active_probability.log()
        + (1.0 - active_target) * torch.log1p(-active_probability)
    )
    components["operation_active"] = _masked_mean(
        active_loss, supported.unsqueeze(1).expand_as(active_loss)
    )

    span_pool = _span_pool(
        source_hidden,
        batch["span_start"].long(),
        batch["span_end"].long(),
        batch["presence"].bool() & supported.unsqueeze(1),
    )
    span_keys = diagnostic_heads.span_projection(span_pool)
    components["span_owner"] = _span_alignment_loss(
        boundary.addresses, span_keys, batch["presence"].bool() & supported.unsqueeze(1)
    )

    state_logits = diagnostic_heads.state_decoder(auxiliary["state_features"])
    state_values = batch["state_values"].float()
    if state_values.shape != state_logits.shape:
        raise ValueError(f"temporal state target shape {tuple(state_values.shape)} != logits {tuple(state_logits.shape)}")
    state_mask = (
        batch["state_step_mask"].bool().unsqueeze(-1).unsqueeze(-1)
        & batch["state_feature_mask"].bool()
        & supported.view(-1, 1, 1, 1)
    )
    state_loss = F.binary_cross_entropy_with_logits(
        state_logits.float(), state_values, reduction="none"
    )
    components["state"] = _masked_mean(state_loss, state_mask)

    if causal:
        intervention = intervention_outputs(model, boundary, base, batch)
        base_margin = intervention["base_answer_margin"]

        def margin_loss(name: str, threshold: float) -> Tensor:
            degradation = base_margin - intervention[f"{name}_answer_margin"]
            return _masked_mean(F.relu(float(threshold) - degradation), supported)

        components["no_core_margin"] = margin_loss("no_core", margins.no_core_answer_margin)
        components["payload_binding_margin"] = 0.5 * (
            margin_loss("payload_zero", margins.payload_binding)
            + margin_loss("payload_shuffle", margins.payload_binding)
        )
        components["operation_binding_margin"] = 0.5 * (
            margin_loss("operation_zero", margins.operation_binding)
            + margin_loss("operation_shuffle", margins.operation_binding)
        )
        if model.config.slots > 1:
            components["wrong_start_margin"] = margin_loss("wrong_start", margins.wrong_start_answer_margin)
            components["target_shuffle_margin"] = margin_loss("target_shuffle", margins.target_shuffle)
            components["relevant_margin"] = margin_loss("relevant_replace", margins.relevant_answer_margin)
            base_distribution = F.log_softmax(base["logits"].float(), dim=-1)
            irrelevant_distribution = F.log_softmax(intervention["irrelevant_replace_logits"].float(), dim=-1)
            irrelevant_error = (irrelevant_distribution - base_distribution).square().mean(dim=-1)
            irrelevant_valid = supported & (batch.get("irrelevant_owner", torch.full_like(answers, -1)) >= 0)
            components["irrelevant_consistency"] = _masked_mean(
                irrelevant_error, irrelevant_valid
            )
        else:
            zero = components["answer"] * 0.0
            components.update(
                wrong_start_margin=zero,
                target_shuffle_margin=zero,
                relevant_margin=zero,
                irrelevant_consistency=zero,
            )
    else:
        intervention = {}
        zero = components["answer"] * 0.0
        components.update(
            no_core_margin=zero,
            wrong_start_margin=zero,
            payload_binding_margin=zero,
            operation_binding_margin=zero,
            target_shuffle_margin=zero,
            relevant_margin=zero,
            irrelevant_consistency=zero,
        )

    total = (
        weights.answer * components["answer"]
        + weights.slot_identity * components["slot_identity"]
        + weights.route * (components["source_route"] + components["target_route"])
        + weights.query_owner * components["query_owner"]
        + weights.presence * components["presence"]
        + weights.span_owner * components["span_owner"]
        + weights.state * components["state"]
        + weights.operation_active * components["operation_active"]
        + weights.no_core_margin * components["no_core_margin"]
        + weights.wrong_start_margin * components["wrong_start_margin"]
        + weights.payload_binding_margin * components["payload_binding_margin"]
        + weights.operation_binding_margin * components["operation_binding_margin"]
        + weights.target_shuffle_margin * components["target_shuffle_margin"]
        + weights.relevant_margin * components["relevant_margin"]
        + weights.irrelevant_consistency * components["irrelevant_consistency"]
    )
    detached = {name: float(value.detach().float().cpu()) for name, value in components.items()}
    detached["total"] = float(total.detach().float().cpu())
    return total, detached, {
        "boundary": boundary,
        "base": base,
        "state_logits": state_logits,
        "span_keys": span_keys,
        "slot_logits": per_slot_logits,
        "intervention": intervention,
    }


__all__ = [
    "CausalMargins",
    "LossWeights",
    "STATE_FEATURE_NAMES",
    "SuccessorDiagnosticHeads",
    "compute_training_loss",
    "intervention_outputs",
    "slot_answer_logits",
]
