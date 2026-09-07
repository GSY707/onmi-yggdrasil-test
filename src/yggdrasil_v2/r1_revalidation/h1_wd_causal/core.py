from __future__ import annotations

"""Decision-causal probes for the single shared+routed H1 model.

This module is deliberately separate from ``h1_wd``.  It does not train a
model and it never consumes task/family route labels.  The route schedule is
either supplied by the caller or replayed from the model's ordinary forward;
all interventions then use that same fixed schedule.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from yggdrasil_v2.r1_revalidation.h1.model import H1LatentReasoner


@dataclass(frozen=True)
class CausalSite:
    """One layer/step site captured on a route-replayed trajectory."""

    layer_index: int
    step_index: int
    attention_state: Tensor
    common: Tensor
    projection: Tensor
    route: Tensor


def _validate_inputs(model: H1LatentReasoner, hidden: Tensor, mask: Tensor) -> None:
    if hidden.ndim != 3 or hidden.shape[-1] != model.config.source_width:
        raise ValueError("hidden must have shape [batch, source_tokens, source_width]")
    if mask.ndim != 2 or mask.shape[:2] != hidden.shape[:2]:
        raise ValueError("mask must have shape [batch, source_tokens]")


def _validate_schedule(
    model: H1LatentReasoner, route_schedule: Tensor, batch: int
) -> Tensor:
    if route_schedule.ndim != 3:
        raise ValueError("route_schedule must have shape [batch, layers, steps]")
    expected = (batch, model.config.recurrent_layers, model.config.T)
    if tuple(route_schedule.shape) != expected:
        raise ValueError(f"route_schedule shape {tuple(route_schedule.shape)} != {expected}")
    route_schedule = route_schedule.to(dtype=torch.long)
    if bool(((route_schedule < 0) | (route_schedule >= 2)).any()):
        raise ValueError("route_schedule contains a route outside [0, 1]")
    return route_schedule


def _selected_projection(layer: Any, state: Tensor, route: Tensor) -> Tensor:
    """Evaluate exactly the routed projection selected for each record."""

    features = layer.routed_feature_trunk(state)
    if len(layer.routed_projections) == 1:
        # The shared arm has no sparse choice; H1's dispatch uses its sole
        # projection for every route value emitted by the router.
        return layer.routed_projections[0](features)
    output = torch.zeros_like(state)
    for expert_index, projection in enumerate(layer.routed_projections):
        selected = torch.nonzero(route == expert_index, as_tuple=False).flatten()
        if selected.numel():
            update = projection(features.index_select(0, selected))
            output = output.index_copy(0, selected, update)
    return output


def _normal_route_schedule(
    model: H1LatentReasoner, hidden: Tensor, mask: Tensor
) -> Tensor:
    with torch.no_grad():
        ordinary = model(hidden, mask, return_trajectory=False)
    route = ordinary["route_assignments"]
    if route is None:
        raise RuntimeError("ordinary H1 forward did not return route_assignments")
    return route.detach().to(dtype=torch.long)


def forward_route_replay(
    model: H1LatentReasoner,
    hidden: Tensor,
    mask: Tensor,
    *,
    route_schedule: Tensor | None = None,
    disable_common: bool = False,
    disable_projection: bool = False,
    intervention: str = "none",
    capture: bool = False,
    return_trajectory: bool = True,
) -> dict[str, Any]:
    """Run one H1 model with a fixed ``[B, L, T]`` route schedule.

    ``disable_common`` and ``disable_projection`` are global path
    interventions.  When neither is set, replaying the schedule returned by
    the ordinary model forward is bitwise-equivalent on the normal path.  If
    ``capture`` is true, ``captures`` contains one :class:`CausalSite` per
    layer/step, in the same order in which the recurrent core executes them.
    """

    if intervention != "none":
        aliases = {
            "disable_shared_ffn": (True, False),
            "disable_routed_projection": (False, True),
            "disable_both_ffn_paths": (True, True),
            "disable_common": (True, False),
            "disable_projection": (False, True),
            "disable_both": (True, True),
        }
        if intervention not in aliases:
            raise ValueError(f"unknown causal intervention: {intervention}")
        alias_common, alias_projection = aliases[intervention]
        disable_common = disable_common or alias_common
        disable_projection = disable_projection or alias_projection
    _validate_inputs(model, hidden, mask)
    core_dtype = next(model.parameters()).dtype
    source = hidden.to(dtype=core_dtype)
    source_mask = mask.to(dtype=torch.bool)
    if route_schedule is None:
        route_schedule = _normal_route_schedule(model, hidden, mask)
    route_schedule = _validate_schedule(model, route_schedule, source.shape[0]).to(source.device)

    source_memory, state = model._boundary(source, source_mask)
    initial_state = state
    source_kv = [layer.source_kv(source_memory) for layer in model.layers]
    route_logits_steps: list[Tensor] = []
    route_steps: list[Tensor] = []
    state_steps: list[Tensor] = []
    sites: list[CausalSite] = []

    for step in range(model.config.T):
        step_signal = model.control_projection(model.step_control[step]).view(1, 1, -1)
        for layer_index, layer in enumerate(model.layers):
            attention_state = layer.attention_transition(
                state, source_kv[layer_index], source_mask, step_signal
            )
            route = route_schedule[:, layer_index, step]
            route_logits_steps.append(layer.route_logits(attention_state))
            route_steps.append(route)
            common = layer.shared_ffn(attention_state)
            projection = _selected_projection(layer, attention_state, route)
            if disable_common:
                common_used = torch.zeros_like(common)
            else:
                common_used = common
            if disable_projection:
                projection_used = torch.zeros_like(projection)
            else:
                projection_used = projection
            # Match H1's exact association: dispatch forms common+projection,
            # then the recurrent block adds that update to attention_state.
            state = attention_state + (common_used + projection_used)
            if capture:
                sites.append(
                    CausalSite(
                        layer_index=layer_index,
                        step_index=step,
                        attention_state=attention_state,
                        common=common,
                        projection=projection,
                        route=route,
                    )
                )
        state_steps.append(state)

    trajectory = torch.stack(state_steps, dim=1) if return_trajectory else None
    if disable_common and disable_projection:
        # Both paths are disabled globally but attention still advances in the
        # same fixed schedule, so no special state handling is required.
        pass
    logits = model.answer_head(model._pool(state))
    route_logits = torch.stack(route_logits_steps, dim=1).reshape(
        source.shape[0], model.config.T, model.config.recurrent_layers, 2
    ).transpose(1, 2).contiguous()
    route_out = torch.stack(route_steps, dim=1).reshape(
        source.shape[0], model.config.T, model.config.recurrent_layers
    ).transpose(1, 2).contiguous()
    return {
        "logits": logits,
        "final_state": state,
        "trajectory": trajectory,
        "route_schedule": route_out,
        "route_assignments": route_out,
        "route_logits": route_logits,
        "captures": tuple(sites) if capture else None,
        "site_captures": tuple(sites) if capture else None,
        "trace_logits": None,
    }


def answer_margin(logits: Tensor, answer_index: Tensor) -> Tensor:
    """Return true-answer logit minus log-sum-exp of all other logits."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    answer_index = answer_index.to(device=logits.device, dtype=torch.long).reshape(-1)
    if answer_index.shape[0] != logits.shape[0]:
        raise ValueError("answer_index batch does not match logits")
    if bool(((answer_index < 0) | (answer_index >= logits.shape[-1])).any()):
        raise ValueError("answer_index contains an invalid class")
    true_logit = logits.gather(1, answer_index[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, answer_index[:, None], float("-inf"))
    return true_logit - torch.logsumexp(other, dim=-1)


def _as_answer_index(answer_index: Tensor | Sequence[int], device: torch.device) -> Tensor:
    if not isinstance(answer_index, Tensor):
        answer_index = torch.as_tensor(answer_index, dtype=torch.long, device=device)
    return answer_index.to(device=device, dtype=torch.long).reshape(-1)


def _capture_gradients(
    margin: Tensor, sites: Sequence[CausalSite]
) -> tuple[Tensor | None, ...]:
    projections = tuple(site.projection for site in sites)
    if not projections:
        return ()
    return torch.autograd.grad(
        margin.sum(), projections, retain_graph=False, allow_unused=True
    )


def causal_transfer_targets(
    model: H1LatentReasoner,
    hidden: Tensor,
    mask: Tensor,
    answer_index: Tensor | Sequence[int],
    *,
    route_schedule: Tensor | None = None,
    request_fraction: float = 0.5,
    site_transfer_cap: float = 0.25,
) -> dict[str, Any]:
    """Construct VJP/Fisher-weighted projection transfer targets.

    The positive margin drop caused by disabling common is the causal signal;
    only ``request_fraction`` (default 0.5) is requested for transfer.  A
    scalar margin VJP with respect to each captured projection supplies the
    local direction.  Per-record Fisher energy (squared VJP norm) allocates
    the request across all sites.  The resulting minimum-norm transfer at a
    site is capped at ``site_transfer_cap * ||common||`` (default 0.25).
    """

    if request_fraction < 0.0:
        raise ValueError("request_fraction must be non-negative")
    if site_transfer_cap <= 0.0:
        raise ValueError("site_transfer_cap must be positive")
    _validate_inputs(model, hidden, mask)
    answers = _as_answer_index(answer_index, hidden.device)
    full = forward_route_replay(
        model, hidden, mask, route_schedule=route_schedule, capture=True
    )
    schedule = full["route_schedule"]
    if schedule is None:
        raise RuntimeError("route replay did not return a route schedule")
    common_off = forward_route_replay(
        model,
        hidden,
        mask,
        route_schedule=schedule,
        disable_common=True,
    )
    projection_off = forward_route_replay(
        model,
        hidden,
        mask,
        route_schedule=schedule,
        disable_projection=True,
    )
    both_off = forward_route_replay(
        model,
        hidden,
        mask,
        route_schedule=schedule,
        disable_common=True,
        disable_projection=True,
    )
    full_margin = answer_margin(full["logits"], answers)
    common_off_margin = answer_margin(common_off["logits"], answers)
    projection_off_margin = answer_margin(projection_off["logits"], answers)
    both_off_margin = answer_margin(both_off["logits"], answers)
    positive_common_drop = (full_margin - common_off_margin).clamp_min(0.0)
    requested_margin = positive_common_drop * float(request_fraction)

    sites = full["captures"]
    if sites is None:
        raise RuntimeError("full causal replay did not capture sites")
    full_vjp = _capture_gradients(full_margin, sites)
    transfers: list[Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    per_site_energy: list[Tensor] = []
    per_site_common_norm: list[Tensor] = []
    for site, grad in zip(sites, full_vjp):
        if grad is None:
            grad = torch.zeros_like(site.projection)
        energy = grad.square().sum(dim=(1, 2))
        common_norm = site.common.detach().square().sum(dim=(1, 2)).sqrt()
        per_site_energy.append(energy)
        per_site_common_norm.append(common_norm)
    energy_stack = torch.stack(per_site_energy, dim=1)
    total_energy = energy_stack.sum(dim=1).clamp_min(1.0e-12)
    for index, (site, grad) in enumerate(zip(sites, full_vjp)):
        if grad is None:
            grad = torch.zeros_like(site.projection)
        energy = per_site_energy[index]
        # Minimum-norm perturbation with the requested local margin share:
        # <vjp, transfer> = requested_margin * FisherShare.
        share = energy / total_energy
        allocated_margin = requested_margin * share
        scale_from_vjp = allocated_margin / energy.clamp_min(1.0e-12)
        transfer = grad * scale_from_vjp.view(-1, 1, 1)
        cap = float(site_transfer_cap) * per_site_common_norm[index]
        transfer_norm_before = transfer.square().sum(dim=(1, 2)).sqrt()
        scale = torch.minimum(
            torch.ones_like(transfer_norm_before),
            cap / transfer_norm_before.clamp_min(1.0e-12),
        )
        transfer = transfer * scale.view(-1, 1, 1)
        transfer_norm = transfer.square().sum(dim=(1, 2)).sqrt()
        transfers.append(transfer)
        diagnostics.append(
            {
                "layer_index": site.layer_index,
                "step_index": site.step_index,
                "fisher_energy": energy.detach(),
                "fisher_share": share.detach(),
                "vjp": grad.detach(),
                "requested_margin": allocated_margin.detach(),
                "common_norm": per_site_common_norm[index].detach(),
                "transfer_norm_before_cap": transfer_norm_before.detach(),
                "transfer_norm": transfer_norm.detach(),
                "transfer_cap": cap.detach(),
                "cap_fraction": scale.detach(),
            }
        )

    return {
        "route_schedule": schedule.detach(),
        "full": full,
        "common_off": common_off,
        "projection_off": projection_off,
        "both_off": both_off,
        "margins": {
            "full": full_margin.detach(),
            "common_off": common_off_margin.detach(),
            "projection_off": projection_off_margin.detach(),
            "both_off": both_off_margin.detach(),
        },
        "full_margin": full_margin.detach(),
        "common_off_margin": common_off_margin.detach(),
        "projection_off_margin": projection_off_margin.detach(),
        "both_off_margin": both_off_margin.detach(),
        "positive_common_margin_drop": positive_common_drop.detach(),
        "requested_margin": requested_margin.detach(),
        "causal_transfer": tuple(transfers),
        "transfers": tuple(transfers),
        "projection_targets": tuple(
            (site.projection.detach() + transfer)
            for site, transfer in zip(sites, transfers)
        ),
        "transfer_targets": tuple(transfers),
        "diagnostics": tuple(diagnostics),
        "total_fisher_energy": total_energy.detach(),
        "uses_family_targets": False,
    }


build_causal_transfer_targets = causal_transfer_targets
generate_causal_transfer_targets = causal_transfer_targets


__all__ = [
    "CausalSite",
    "answer_margin",
    "build_causal_transfer_targets",
    "causal_transfer_targets",
    "forward_route_replay",
    "generate_causal_transfer_targets",
]
