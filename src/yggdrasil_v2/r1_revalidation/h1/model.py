"""Small, auditable H1 latent reasoners.

The two arms deliberately share the same boundary, source/slot attention,
recurrence, common FFN, and readout surface.  Each recurrent block also has
one shared nonlinear conditional feature trunk followed by a sparse,
record-routed final state-write projection.  The shared arm owns one generic
projection; the mixed arm owns two equal-width projections and executes only
the selected one.  Therefore the arms differ in conditional parameters, not
in active per-record compute.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class H1Config:
    """Configuration for the fixed K=8, T=8 H1 core.

    ``latent_width`` is intentionally configurable so unit tests can use a
    compact core.  The production default is 256; source vectors are the
    2048-wide contextual vectors produced by the upstream boundary cache.
    """

    arm: str = "shared"
    source_width: int = 2048
    latent_width: int = 256
    K: int = 8
    T: int = 8
    recurrent_layers: int = 2
    num_heads: int = 8
    active_ffn_multiplier: int = 3
    shared_ffn_inner_width: int = 384
    routed_feature_width: int = 384
    trace_classes: int = 5
    answer_classes: int = 4
    route_classes: int = 2

    def __post_init__(self) -> None:
        if self.arm not in {"shared", "mixed"}:
            raise ValueError("arm must be 'shared' or 'mixed'")
        for name in ("source_width", "latent_width", "K", "T", "recurrent_layers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.latent_width % self.num_heads:
            raise ValueError("latent_width must be divisible by num_heads")
        if self.active_ffn_multiplier <= 0:
            raise ValueError("active_ffn_multiplier must be positive")
        if self.shared_ffn_inner_width <= 0 or self.routed_feature_width <= 0:
            raise ValueError("shared/routed FFN inner widths must be positive")
        if (
            self.shared_ffn_inner_width + self.routed_feature_width
            != self.latent_width * self.active_ffn_multiplier
        ):
            raise ValueError(
                "shared FFN and routed feature widths must equal the registered active budget"
            )
        if self.route_classes != 2 or self.trace_classes != 5 or self.answer_classes != 4:
            raise ValueError("H1 output class counts are fixed at route=2, trace=5, answer=4")

    @property
    def num_slots(self) -> int:
        return self.K

    @property
    def steps(self) -> int:
        return self.T


class SwiGLU(nn.Module):
    """A width-preserving SwiGLU FFN used as a residual branch."""

    def __init__(self, width: int, inner_width: int) -> None:
        super().__init__()
        if inner_width <= 0:
            raise ValueError("SwiGLU inner_width must be positive")
        self.gate = nn.Linear(width, inner_width)
        self.up = nn.Linear(width, inner_width)
        self.down = nn.Linear(inner_width, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _SourceCrossAttention(nn.Module):
    """Cross-attention whose source K/V projection is separable from query use."""

    def __init__(self, width: int, heads: int, source_width: int) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(source_width, width, bias=False)
        self.value = nn.Linear(source_width, width, bias=False)
        self.output = nn.Linear(width, width)

    def project_source(self, source: Tensor) -> Tuple[Tensor, Tensor]:
        shape = source.shape[:-1] + (self.heads, self.head_width)
        key = self.key(source).reshape(shape).transpose(1, 2)
        value = self.value(source).reshape(shape).transpose(1, 2)
        return key, value

    def forward(
        self,
        query: Tensor,
        source_k: Tensor,
        source_v: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        shape = query.shape[:-1] + (self.heads, self.head_width)
        q = self.query(query).reshape(shape).transpose(1, 2)
        scores = torch.matmul(q, source_k.transpose(-1, -2)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        result = torch.matmul(weights, source_v).transpose(1, 2).reshape_as(query)
        return self.output(result)


class _FFNBranch(nn.Module):
    """One width-preserving FFN residual branch."""

    def __init__(
        self,
        config: H1Config,
        inner_width: int,
        *,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        d = config.latent_width
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = SwiGLU(d, inner_width)
        if zero_output:
            # The generic/routed branch starts as a true residual delta.  The
            # common FFN therefore owns the initial shared computation, while
            # the second branch can learn either a global or typed correction.
            nn.init.zeros_(self.ffn.down.weight)
            nn.init.zeros_(self.ffn.down.bias)

    def forward(self, state: Tensor) -> Tensor:
        return self.ffn(self.ffn_norm(state))


class _RoutedFeatureTrunk(nn.Module):
    """Shared D-to-H nonlinear features feeding a routed H-to-D write."""

    def __init__(self, config: H1Config, feature_width: int) -> None:
        super().__init__()
        d = config.latent_width
        if feature_width <= 0:
            raise ValueError("feature_width must be positive")
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Linear(d, feature_width)
        self.up = nn.Linear(d, feature_width)

    def forward(self, state: Tensor) -> Tensor:
        normalized = self.norm(state)
        return F.silu(self.gate(normalized)) * self.up(normalized)


class _RoutedProjection(nn.Linear):
    """A standard non-zero-initialized final state-write projection."""


class _RecurrentLayer(nn.Module):
    def __init__(self, config: H1Config) -> None:
        super().__init__()
        d = config.latent_width
        self.cross_norm = nn.LayerNorm(d)
        self.cross = _SourceCrossAttention(d, config.num_heads, d)
        self.slot_norm = nn.LayerNorm(d)
        self.slot = nn.MultiheadAttention(
            d, config.num_heads, batch_first=True, dropout=0.0
        )
        self.router_norm = nn.LayerNorm(d)
        self.router = nn.Linear(d, config.route_classes)
        self.shared_ffn = _FFNBranch(config, config.shared_ffn_inner_width)
        branch_count = 1 if config.arm == "shared" else 2
        self.routed_feature_trunk = _RoutedFeatureTrunk(
            config, config.routed_feature_width
        )
        self.routed_projections = nn.ModuleList(
            [
                _RoutedProjection(config.routed_feature_width, d)
                for _ in range(branch_count)
            ]
        )

    def source_kv(self, source: Tensor) -> Tuple[Tensor, Tensor]:
        return self.cross.project_source(source)

    def source_context(self, x: Tensor, source_kv: Tuple[Tensor, Tensor], mask: Tensor) -> Tensor:
        """Use current state as query against the layer's cached source K/V."""
        query = self.cross_norm(x)
        padding = ~mask.to(dtype=torch.bool)
        return self.cross(query, source_kv[0], source_kv[1], padding)

    def attention_transition(
        self,
        state: Tensor,
        source_kv: Tuple[Tensor, Tensor],
        mask: Tensor,
        step_signal: Tensor,
    ) -> Tensor:
        """Apply the route-independent attention half of a Transformer block."""
        query_state = state + step_signal
        context = self.source_context(query_state, source_kv, mask)
        # The fixed step address conditions attention but is not itself a
        # payload residual.  Writing it directly into H_t once per layer would
        # accumulate control state and violate the S_t=(A_t,H_t) boundary.
        cross_state = state + context
        slot_input = self.slot_norm(cross_state + step_signal)
        slot_update, _ = self.slot(
            slot_input, slot_input, slot_input, need_weights=False
        )
        return cross_state + slot_update

    def route_logits(self, state: Tensor) -> Tensor:
        # One content route per record, without a duplicated K axis.
        summary = state.mean(dim=1)
        return self.router(self.router_norm(summary))

    def dispatch(
        self,
        state: Tensor,
        route: Tensor,
        arm: str,
        *,
        disable_routed_projection: bool = False,
    ) -> Tensor:
        """Run common FFN plus one sparse routed final projection."""
        common = self.shared_ffn(state)
        if disable_routed_projection:
            return common
        features = self.routed_feature_trunk(state)
        if arm == "shared":
            return common + self.routed_projections[0](features)
        output = torch.zeros_like(state)
        for expert_index, projection in enumerate(self.routed_projections):
            selected = torch.nonzero(route == expert_index, as_tuple=False).flatten()
            if selected.numel():
                update = projection(features.index_select(0, selected))
                output = output.index_copy(0, selected, update)
        return common + output


class H1LatentReasoner(nn.Module):
    """The H1 latent recurrent core.

    The public forward contract intentionally exposes only contextual hidden
    vectors, their mask, and named causal interventions.  All task semantics
    enter through ``hidden``; no task-specific control is accepted here.
    """

    VALID_INTERVENTIONS = {
        "none",
        "flip_route",
        "force_route0",
        "force_route1",
        "swap_experts",
        "disable_routed_projection",
        "disable_recurrence",
        "zero_source",
        "shuffle_source",
    }

    def __init__(self, config: Optional[H1Config] = None) -> None:
        super().__init__()
        self.config = config or H1Config()
        d = self.config.latent_width
        self.arm = self.config.arm

        # Boundary produces generic D-wide source memory only.  The K learned
        # latent slots begin source-independent, so source content can enter
        # H_t only through the recurrent source-attention path below.
        self.boundary_source_norm = nn.LayerNorm(self.config.source_width)
        self.boundary_projection = nn.Linear(self.config.source_width, d, bias=False)
        self.latent_slots = nn.Parameter(torch.empty(self.config.K, d))
        self.initial_norm = nn.LayerNorm(d)
        nn.init.normal_(self.latent_slots, mean=0.0, std=d ** -0.5)

        # Fixed, non-semantic step address.  It is a Fourier basis over the
        # bounded T-step control index, not a task/family/target embedding.
        positions = torch.arange(self.config.T, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.arange(d, dtype=torch.float32).unsqueeze(0)
        angles = positions / torch.pow(
            torch.tensor(float(max(1, d)), dtype=torch.float32),
            frequencies / float(max(1, d)),
        )
        fourier = torch.sin(angles)
        fourier[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer("step_control", fourier, persistent=True)
        self.control_projection = nn.Linear(d, d, bias=False)

        self.layers = nn.ModuleList(
            [_RecurrentLayer(self.config) for _ in range(self.config.recurrent_layers)]
        )
        self.pool_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.pool_query, mean=0.0, std=d ** -0.5)
        self.pool_norm = nn.LayerNorm(d)
        self.answer_head = nn.Linear(d, self.config.answer_classes)
        self.trace_head: Optional[nn.Linear] = nn.Linear(d, self.config.trace_classes)

    def _boundary(self, source: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        memory = self.boundary_projection(self.boundary_source_norm(source))
        memory = memory * mask.unsqueeze(-1).to(dtype=memory.dtype)
        slots = self.latent_slots.unsqueeze(0).expand(source.shape[0], -1, -1)
        return memory, self.initial_norm(slots)

    @staticmethod
    def _route_intervention(route: Tensor, intervention: str) -> Tensor:
        if intervention == "flip_route" or intervention == "swap_experts":
            return 1 - route
        if intervention == "force_route0":
            return torch.zeros_like(route)
        if intervention == "force_route1":
            return torch.ones_like(route)
        return route

    def _pool(self, state: Tensor) -> Tensor:
        scores = torch.matmul(state, self.pool_query) / math.sqrt(state.shape[-1])
        weights = scores.softmax(dim=-1)
        return self.pool_norm(torch.sum(state * weights.unsqueeze(-1), dim=1))

    def forward(
        self,
        hidden: Tensor,
        mask: Tensor,
        *,
        intervention: str = "none",
        return_trajectory: bool = False,
    ) -> Dict[str, Optional[Tensor]]:
        if intervention not in self.VALID_INTERVENTIONS:
            raise ValueError(f"unknown intervention: {intervention}")
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, source_tokens, source_width]")
        if hidden.shape[-1] != self.config.source_width:
            raise ValueError(
                f"hidden width {hidden.shape[-1]} != source_width {self.config.source_width}"
            )
        if mask.ndim != 2 or mask.shape[:2] != hidden.shape[:2]:
            raise ValueError("mask must have shape [batch, source_tokens]")

        core_dtype = next(self.parameters()).dtype
        source = hidden.to(dtype=core_dtype)
        source_mask = mask.to(dtype=torch.bool)
        if intervention == "zero_source":
            source = torch.zeros_like(source)
        elif intervention == "shuffle_source":
            # Shuffle ownership across records, not token positions.  A
            # token-axis permutation would be an exact attention symmetry.
            if source.shape[0] > 1:
                owner = torch.roll(torch.arange(source.shape[0], device=source.device), 1)
                source = source.index_select(0, owner)
                source_mask = source_mask.index_select(0, owner)
            else:
                source = torch.zeros_like(source)

        source_memory, state = self._boundary(source, source_mask)
        initial_state = state
        route_logit_steps = []
        route_assignment_steps = []
        trace_steps = []
        state_steps = []

        # Source K/V projections are deliberately materialized once per layer,
        # outside the fixed T loop.  Each step still invokes cross-attention
        # with the current state as query.
        source_kv = [layer.source_kv(source_memory) for layer in self.layers]
        for step in range(self.config.T):
            step_signal = self.control_projection(self.step_control[step]).view(1, 1, -1)
            for layer_index, layer in enumerate(self.layers):
                attention_state = layer.attention_transition(
                    state,
                    source_kv[layer_index],
                    source_mask,
                    step_signal,
                )
                route_logits = layer.route_logits(attention_state)
                route = route_logits.argmax(dim=-1)
                route = self._route_intervention(route, intervention)
                route_logit_steps.append((layer_index, route_logits))
                route_assignment_steps.append((layer_index, route))

                if intervention != "disable_recurrence":
                    expert_update = layer.dispatch(
                        attention_state,
                        route,
                        self.arm,
                        disable_routed_projection=(
                            intervention == "disable_routed_projection"
                        ),
                    )
                    # This is a standard recurrent Transformer block: dense
                    # attention updates the common residual, while route only
                    # controls the FFN residual.  There is no task-specific
                    # branch and no path from router logits to the answer.
                    state = attention_state + expert_update

            state_steps.append(state)
            trace_steps.append(self._pool(state))

        if intervention == "disable_recurrence":
            # No state update was committed in any of the T steps; make the
            # H0 contract explicit for callers and future refactors.
            state = initial_state
        trajectory = torch.stack(state_steps, dim=1) if return_trajectory else None
        pooled = self._pool(state)
        logits = self.answer_head(pooled)
        trace_logits = None
        if return_trajectory and self.trace_head is not None:
            trace_logits = self.trace_head(torch.stack(trace_steps, dim=1))

        # Reassemble record-level [B, L, T, C] and [B, L, T].
        route_logits_tensor = torch.stack(
            [item[1] for item in route_logit_steps], dim=1
        ).reshape(
            state.shape[0], self.config.T, self.config.recurrent_layers, 2
        ).transpose(1, 2).contiguous()
        route_assignments = torch.stack(
            [item[1] for item in route_assignment_steps], dim=1
        ).reshape(
            state.shape[0], self.config.T, self.config.recurrent_layers
        ).transpose(1, 2).contiguous()

        return {
            "logits": logits,
            "final_state": state,
            "trajectory": trajectory,
            "trace_logits": trace_logits,
            "route_logits": route_logits_tensor,
            "route_assignments": route_assignments,
        }

    def strip_trace_head(self) -> "H1LatentReasoner":
        """Physically remove the auxiliary trace head for deployment."""
        self.trace_head = None
        return self

    def deployment_state(self) -> Mapping[str, Tensor]:
        """Return a state dict with auxiliary trace parameters excluded."""
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("trace_head.")
        }

    def parameter_report(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        trace = 0
        if self.trace_head is not None:
            trace = sum(parameter.numel() for parameter in self.trace_head.parameters())
        inactive_sparse = 0
        if self.arm == "mixed":
            inactive_sparse = sum(
                parameter.numel()
                for layer in self.layers
                for parameter in layer.routed_projections[1].parameters()
            )
        return {
            "total": total,
            "trainable": trainable,
            "trace_head": trace,
            "deployment": total - trace,
            "inactive_sparse_parameters_per_record": inactive_sparse,
            "active_parameters_per_record": total - inactive_sparse,
            "active_deployment_parameters_per_record": total - inactive_sparse - trace,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "deployment_parameters": total - trace,
        }

    def integrity_report(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "passed": True,
            "source_width": self.config.source_width,
            "latent_width": self.config.latent_width,
            "K": self.config.K,
            "T": self.config.T,
            "recurrent_layers": self.config.recurrent_layers,
            "fixed_step_control": True,
            "step_control_type": "fixed_fourier",
            "step_control_shape": list(self.step_control.shape),
            "step_control_trainable": bool(self.step_control.requires_grad),
            "source_kv_projections_per_forward": self.config.recurrent_layers,
            "cross_attention_calls_per_forward": self.config.recurrent_layers * self.config.T,
            "source_independent_initial_state": True,
            "common_attention_and_ffn_state_write": True,
            "routed_feature_trunk_shared_across_routes": True,
            "selected_routed_projection_is_only_conditional_state_write": True,
            "route_controls_final_projection_only": True,
            "route_granularity": "record",
            "conditional_transition_scope": "factorized-routed-state-write-projection",
            "routed_projection_standard_nonzero_init": True,
            "routed_feature_width": self.config.routed_feature_width,
            "routed_projection_input_width": self.config.routed_feature_width,
            "routed_projection_output_width": self.config.latent_width,
            "routed_projection_count": len(self.layers[0].routed_projections),
            "answer_reads_pooled_final_state_only": True,
            "trace_head_present": self.trace_head is not None,
            "route_assignment_is_hard_top1": True,
            "deployment_has_trace_head": any(key.startswith("trace_head.") for key in self.deployment_state()),
            "trace_head_physically_strippable": True,
        }

    def active_flop_estimate(
        self, batch_size: int = 1, source_tokens: int = 128, *, include_router: bool = True
    ) -> float:
        """Estimate active multiply-add work, counting one mixed expert only.

        The ledger is intentionally conservative and architecture-facing: it
        counts the dense projections and attention products that dominate the
        forward, while omitting elementwise norms, activations, and indexing.
        Every counted term is arm-independent.
        """
        d = self.config.latent_width
        source_d = self.config.source_width
        k = self.config.K
        t = self.config.T
        layers = self.config.recurrent_layers
        shared_inner = self.config.shared_ffn_inner_width
        routed_inner = self.config.routed_feature_width
        # Every term is intentionally arm-independent: both arms execute the
        # common FFN plus exactly one equal-width residual FFN branch.
        boundary = batch_size * source_tokens * source_d * d
        source_projection = layers * batch_size * 2 * source_tokens * d * d
        cross = layers * t * batch_size * (
            2 * k * d * d + 2 * k * source_tokens * d
        )
        slot = layers * t * batch_size * (4 * k * d * d + 2 * k * k * d)
        common_ffn = layers * t * batch_size * k * (3 * d * shared_inner)
        routed_feature_trunk = layers * t * batch_size * k * (2 * d * routed_inner)
        routed_projection = layers * t * batch_size * k * (routed_inner * d)
        step_control = t * d * d
        router = layers * t * batch_size * d * 2 if include_router else 0
        readout = batch_size * d * (
            self.config.answer_classes + t * self.config.trace_classes
        )
        return float(
            boundary
            + source_projection
            + cross
            + slot
            + common_ffn
            + routed_feature_trunk
            + routed_projection
            + step_control
            + router
            + readout
        )

    estimate_active_flops = active_flop_estimate


def build_matched_pair(
    config: Optional[H1Config] = None, *, seed: int = 0
) -> Tuple[H1LatentReasoner, H1LatentReasoner]:
    """Build shared/mixed arms with exactly aligned common parameters.

    The common FFN and nonlinear routed feature trunk are shared exactly.  Both
    mixed final projections are copied from the shared arm's generic
    projection.  This makes the arms functionally identical at initialization
    while leaving training free to expose a typed state-write benefit.
    """
    base = config or H1Config()
    if base.arm not in {"shared", "mixed"}:
        raise ValueError("config.arm must be shared or mixed")
    shared_config = replace(base, arm="shared")
    mixed_config = replace(base, arm="mixed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        shared = H1LatentReasoner(shared_config)
        mixed = H1LatentReasoner(mixed_config)
    mixed_state = mixed.state_dict()
    shared_state = shared.state_dict()
    for key, value in shared_state.items():
        if key in mixed_state:
            mixed_state[key].copy_(value)
    for left, right in zip(shared.layers, mixed.layers):
        right.shared_ffn.load_state_dict(left.shared_ffn.state_dict())
        right.routed_feature_trunk.load_state_dict(left.routed_feature_trunk.state_dict())
        right.routed_projections[0].load_state_dict(
            left.routed_projections[0].state_dict()
        )
        right.routed_projections[1].load_state_dict(
            left.routed_projections[0].state_dict()
        )
    mixed.load_state_dict(mixed_state, strict=False)
    return shared, mixed


__all__ = ["H1Config", "H1LatentReasoner", "SwiGLU", "build_matched_pair"]
