from __future__ import annotations

"""Functional primitives for partial projection write and common residualization."""

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor

from yggdrasil_v2.r1_revalidation.h1.model import H1LatentReasoner


@dataclass(frozen=True)
class TransitionCell:
    """One frozen layer/step transition on the predecessor trajectory."""

    layer_index: int
    step_index: int
    attention_state: Tensor
    common_update: Tensor
    projection_update: Tensor
    route: Tensor


def selected_projection(layer: Any, state: Tensor, route: Tensor) -> Tensor:
    """Run exactly one projection per record while retaining autograd."""

    features = layer.routed_feature_trunk(state)
    output = torch.zeros_like(state)
    for expert_index, projection in enumerate(layer.routed_projections):
        selected = torch.nonzero(route == expert_index, as_tuple=False).flatten()
        if selected.numel():
            update = projection(features.index_select(0, selected))
            output = output.index_copy(0, selected, update)
    return output


def _validate_inputs(model: H1LatentReasoner, hidden: Tensor, mask: Tensor) -> None:
    if hidden.ndim != 3 or hidden.shape[-1] != model.config.source_width:
        raise ValueError("hidden has an invalid H1-WD shape")
    if mask.ndim != 2 or mask.shape[:2] != hidden.shape[:2]:
        raise ValueError("mask has an invalid H1-WD shape")


def forward_write_delete(
    model: H1LatentReasoner,
    hidden: Tensor,
    mask: Tensor,
    *,
    intervention: str = "none",
    common_scale: float = 1.0,
    return_trajectory: bool = True,
) -> dict[str, Tensor | None]:
    """Run the existing model with WD-only common-path interventions.

    The original H1 model and its frozen intervention surface remain untouched.
    """

    valid = {
        "none",
        "flip_route",
        "force_route0",
        "force_route1",
        "swap_experts",
        "disable_routed_projection",
        "disable_shared_ffn",
        "disable_both_ffn_paths",
        "disable_recurrence",
        "zero_source",
        "shuffle_source",
    }
    if intervention not in valid:
        raise ValueError(f"unknown H1-WD intervention: {intervention}")
    if not 0.0 <= float(common_scale) <= 1.0:
        raise ValueError("common_scale must be in [0, 1]")
    _validate_inputs(model, hidden, mask)

    core_dtype = next(model.parameters()).dtype
    source = hidden.to(dtype=core_dtype)
    source_mask = mask.to(dtype=torch.bool)
    if intervention == "zero_source":
        source = torch.zeros_like(source)
    elif intervention == "shuffle_source":
        if source.shape[0] > 1:
            owner = torch.roll(torch.arange(source.shape[0], device=source.device), 1)
            source = source.index_select(0, owner)
            source_mask = source_mask.index_select(0, owner)
        else:
            source = torch.zeros_like(source)

    source_memory, state = model._boundary(source, source_mask)
    initial_state = state
    route_logits_steps: list[Tensor] = []
    route_steps: list[Tensor] = []
    state_steps: list[Tensor] = []
    source_kv = [layer.source_kv(source_memory) for layer in model.layers]

    for step in range(model.config.T):
        step_signal = model.control_projection(model.step_control[step]).view(1, 1, -1)
        for layer_index, layer in enumerate(model.layers):
            attention_state = layer.attention_transition(
                state, source_kv[layer_index], source_mask, step_signal
            )
            route_logits = layer.route_logits(attention_state)
            route = route_logits.argmax(dim=-1)
            route = model._route_intervention(route, intervention)
            route_logits_steps.append(route_logits)
            route_steps.append(route)

            if intervention != "disable_recurrence":
                common = layer.shared_ffn(attention_state)
                if intervention in {"disable_shared_ffn", "disable_both_ffn_paths"}:
                    common = torch.zeros_like(common)
                else:
                    common = common * float(common_scale)
                if intervention in {
                    "disable_routed_projection",
                    "disable_both_ffn_paths",
                }:
                    routed = torch.zeros_like(common)
                else:
                    routed = selected_projection(layer, attention_state, route)
                # Preserve the predecessor's exact floating-point association:
                # dispatch first forms common+routed, then the block adds it.
                state = attention_state + (common + routed)
        state_steps.append(state)

    if intervention == "disable_recurrence":
        state = initial_state
    trajectory = torch.stack(state_steps, dim=1) if return_trajectory else None
    logits = model.answer_head(model._pool(state))
    route_logits_tensor = torch.stack(route_logits_steps, dim=1).reshape(
        state.shape[0], model.config.T, model.config.recurrent_layers, 2
    ).transpose(1, 2).contiguous()
    route_assignments = torch.stack(route_steps, dim=1).reshape(
        state.shape[0], model.config.T, model.config.recurrent_layers
    ).transpose(1, 2).contiguous()
    return {
        "logits": logits,
        "final_state": state,
        "trajectory": trajectory,
        "route_logits": route_logits_tensor,
        "route_assignments": route_assignments,
        "trace_logits": None,
    }


def collect_transitions(
    model: H1LatentReasoner,
    hidden: Tensor,
    mask: Tensor,
) -> tuple[TransitionCell, ...]:
    """Collect the immutable predecessor transition dataset for one batch."""

    _validate_inputs(model, hidden, mask)
    core_dtype = next(model.parameters()).dtype
    source = hidden.to(dtype=core_dtype)
    source_mask = mask.to(dtype=torch.bool)
    source_memory, state = model._boundary(source, source_mask)
    source_kv = [layer.source_kv(source_memory) for layer in model.layers]
    cells: list[TransitionCell] = []
    for step in range(model.config.T):
        step_signal = model.control_projection(model.step_control[step]).view(1, 1, -1)
        for layer_index, layer in enumerate(model.layers):
            attention_state = layer.attention_transition(
                state, source_kv[layer_index], source_mask, step_signal
            )
            route = layer.route_logits(attention_state).argmax(dim=-1)
            common = layer.shared_ffn(attention_state)
            projection = selected_projection(layer, attention_state, route)
            cells.append(
                TransitionCell(
                    layer_index=layer_index,
                    step_index=step,
                    attention_state=attention_state.detach(),
                    common_update=common.detach(),
                    projection_update=projection.detach(),
                    route=route.detach(),
                )
            )
            state = attention_state + common + projection
    return tuple(cells)


def overlap_write_target(
    common: Tensor,
    projection: Tensor,
    *,
    maximum_coefficient: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Move only the positive common component aligned with an existing projection.

    The coefficient is computed per record over all K slots and D channels.
    It is clipped before becoming a frozen target, so the routed branch is
    never asked to absorb the complete common FFN.
    """

    if common.shape != projection.shape or common.ndim != 3:
        raise ValueError("common/projection transfer tensors must share [B,K,D]")
    if maximum_coefficient <= 0.0:
        raise ValueError("maximum_coefficient must be positive")
    common_flat = common.flatten(1)
    projection_flat = projection.flatten(1)
    denominator = projection_flat.square().sum(dim=-1).clamp_min(1.0e-12)
    coefficient = (common_flat * projection_flat).sum(dim=-1) / denominator
    coefficient = coefficient.clamp(min=0.0, max=float(maximum_coefficient))
    transfer = coefficient[:, None, None] * projection
    return projection + transfer, transfer, coefficient


def normalized_mse(actual: Tensor, target: Tensor) -> Tensor:
    denominator = target.detach().square().mean().clamp_min(1.0e-12)
    return (actual - target).square().mean() / denominator


def parameter_groups(model: H1LatentReasoner, role: str) -> tuple[Tensor, ...]:
    if role == "projection":
        modules: Iterable[Any] = (
            module
            for layer in model.layers
            for module in (layer.routed_feature_trunk, *layer.routed_projections)
        )
    elif role == "common":
        modules = (layer.shared_ffn for layer in model.layers)
    elif role == "joint":
        modules = (
            module
            for layer in model.layers
            for module in (
                layer.shared_ffn,
                layer.routed_feature_trunk,
                *layer.routed_projections,
            )
        )
        modules = (*modules, model.answer_head)
    else:
        raise ValueError(f"unknown H1-WD parameter role: {role}")
    parameters: list[Tensor] = []
    for module in modules:
        parameters.extend(module.parameters())
    return tuple(parameters)


def freeze_for_role(model: H1LatentReasoner, role: str) -> tuple[Tensor, ...]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = parameter_groups(model, role)
    for parameter in selected:
        parameter.requires_grad_(True)
    return selected


__all__ = [
    "TransitionCell",
    "collect_transitions",
    "forward_write_delete",
    "freeze_for_role",
    "normalized_mse",
    "overlap_write_target",
    "parameter_groups",
    "selected_projection",
]
