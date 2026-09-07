from __future__ import annotations

"""C1U publicly-grounded gate-free workspace.

Each public card is encoded independently before it enters this module.  The
model never receives a whole-record hidden state.  Object payloads can interact
only through registered source/target addresses and a shared target-only
transition.
"""

from dataclasses import asdict, dataclass, replace
import inspect
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import contract


@dataclass(frozen=True)
class C1UConfig:
    source_width: int = 2_048
    payload_width: int = 512
    address_width: int = contract.ADDRESS_WIDTH
    slots: int = contract.MAX_SLOTS
    operations: int = contract.MAX_OPERATIONS
    ffn_width: int = 2_048
    answer_classes: int = contract.ANSWER_CLASSES
    route_similarity_floor: float = 0.999


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.activation = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.down(F.silu(self.activation(value)) * self.up(value))


class IndependentCardEncoder(nn.Module):
    """Shared card-local encoder with no operation across the card axis."""

    def __init__(self, config: C1UConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.source_width, elementwise_affine=False)
        self.projection = nn.Linear(config.source_width, config.payload_width)
        self.pooled_norm = nn.LayerNorm(config.payload_width)
        self.ffn_norm = nn.LayerNorm(config.payload_width)
        self.ffn = SwiGLU(config.payload_width, config.ffn_width)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        if hidden.ndim != 4 or hidden.shape[-1] != self.config.source_width:
            raise ValueError("card hidden must have shape [batch, cards, tokens, source_width]")
        if mask.shape != hidden.shape[:3] or mask.dtype != torch.bool:
            raise ValueError("card mask must be bool [batch, cards, tokens]")
        if hidden.shape[2] < 1:
            raise ValueError("card token axis must be non-empty")
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("card hidden contains non-finite values")
        if not torch.is_autocast_enabled(hidden.device.type):
            hidden = hidden.to(dtype=self.projection.weight.dtype)
        encoded = self.projection(self.input_norm(hidden))
        weights = mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        pooled = self.pooled_norm(pooled)
        pooled = pooled + self.ffn(self.ffn_norm(pooled))
        active = mask.any(dim=2).unsqueeze(-1)
        return torch.where(active, pooled, torch.zeros_like(pooled))


def _normalise(value: Tensor) -> Tensor:
    return F.normalize(value.float(), dim=-1, eps=1.0e-8)


def _validate_permutation(permutation: Tensor, slots: int) -> Tensor:
    permutation = permutation.to(dtype=torch.long)
    if permutation.ndim != 1 or permutation.numel() != slots:
        raise ValueError("object permutation has the wrong shape")
    expected = torch.arange(slots, device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("object permutation must contain every slot exactly once")
    return permutation


def _registered_route(
    keys: Tensor,
    object_addresses: Tensor,
    object_present: Tensor,
    active: Tensor,
    *,
    similarity_floor: float,
) -> tuple[Tensor, Tensor]:
    """Return exact hard routes and their matching cosine similarity."""

    similarities = torch.einsum("bta,bka->btk", _normalise(keys), _normalise(object_addresses))
    similarities = similarities.masked_fill(~object_present[:, None, :], -torch.inf)
    indices = similarities.argmax(dim=-1)
    best = similarities.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    invalid = active & (~torch.isfinite(best) | (best < float(similarity_floor)))
    if bool(invalid.any()):
        rows = invalid.nonzero(as_tuple=False)[:8].tolist()
        raise ValueError(f"registered address does not match a present object: {rows}")
    weights = F.one_hot(indices, num_classes=object_addresses.shape[1]).to(dtype=object_addresses.dtype)
    weights = weights * active.to(dtype=weights.dtype).unsqueeze(-1)
    return weights, best


@dataclass
class PartitionedBoundaryOutput:
    object_addresses: Tensor
    object_present: Tensor
    initial_payloads: Tensor
    operation_states: Tensor
    operation_active: Tensor
    source_weights: Tensor
    target_weights: Tensor
    query_state: Tensor
    query_weights: Tensor
    route_similarities: dict[str, Tensor]

    def permuted(self, permutation: Tensor) -> "PartitionedBoundaryOutput":
        permutation = _validate_permutation(
            permutation.to(device=self.object_addresses.device), self.object_addresses.shape[1]
        )
        return replace(
            self,
            object_addresses=self.object_addresses[:, permutation],
            object_present=self.object_present[:, permutation],
            initial_payloads=self.initial_payloads[:, permutation],
            source_weights=self.source_weights[:, :, permutation],
            target_weights=self.target_weights[:, :, permutation],
            query_weights=self.query_weights[:, permutation],
        )


class PartitionedBoundary(nn.Module):
    """Encode cards locally, then resolve only public registered addresses."""

    def __init__(self, config: C1UConfig) -> None:
        super().__init__()
        self.config = config
        self.card_encoder = IndependentCardEncoder(config)
        width = config.payload_width
        self.object_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.operation_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.query_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))

    def forward(
        self,
        object_hidden: Tensor,
        object_mask: Tensor,
        object_addresses: Tensor,
        object_present: Tensor,
        operation_hidden: Tensor,
        operation_mask: Tensor,
        operation_source_addresses: Tensor,
        operation_target_addresses: Tensor,
        query_hidden: Tensor,
        query_mask: Tensor,
        query_address: Tensor,
    ) -> PartitionedBoundaryOutput:
        batch, slots = object_hidden.shape[:2]
        if slots != self.config.slots:
            raise ValueError(f"C1U requires exactly {self.config.slots} padded object slots")
        if object_mask.shape != object_hidden.shape[:3] or object_mask.dtype != torch.bool:
            raise ValueError("object_mask must be bool [batch, slots, tokens]")
        if object_present.shape != (batch, slots) or object_present.dtype != torch.bool:
            raise ValueError("object_present must be bool [batch, slots]")
        if not torch.equal(object_mask.any(dim=2), object_present):
            raise ValueError("object_present must exactly match non-empty object cards")
        if bool((object_present.sum(dim=1) < 1).any()):
            raise ValueError("every record requires at least one present object")
        if object_addresses.shape != (batch, slots, self.config.address_width):
            raise ValueError("object_addresses has the wrong shape")
        if not bool(torch.isfinite(object_addresses).all()):
            raise ValueError("object_addresses contains non-finite values")
        address_norms = torch.linalg.vector_norm(object_addresses.float(), dim=-1)
        if bool((object_present & ((address_norms - 1.0).abs() > 1.0e-4)).any()):
            raise ValueError("present object addresses must be unit vectors")
        pairwise = torch.einsum(
            "bka,bja->bkj", _normalise(object_addresses), _normalise(object_addresses)
        )
        pair_mask = object_present[:, :, None] & object_present[:, None, :]
        diagonal = torch.eye(slots, dtype=torch.bool, device=object_hidden.device).unsqueeze(0)
        if bool((pair_mask & ~diagonal & (pairwise.abs() > 0.999)).any()):
            raise ValueError("present object addresses are ambiguous")

        if operation_hidden.ndim != 4 or operation_hidden.shape[:2] != (
            batch,
            self.config.operations,
        ):
            raise ValueError("operation_hidden has the wrong shape")
        if operation_mask.shape != operation_hidden.shape[:3] or operation_mask.dtype != torch.bool:
            raise ValueError("operation_mask must be bool [batch, operations, tokens]")
        if operation_source_addresses.shape != (
            batch,
            self.config.operations,
            self.config.address_width,
        ) or operation_target_addresses.shape != operation_source_addresses.shape:
            raise ValueError("operation address tensors have the wrong shape")
        if not bool(torch.isfinite(operation_source_addresses).all()) or not bool(
            torch.isfinite(operation_target_addresses).all()
        ):
            raise ValueError("operation addresses contain non-finite values")

        if query_hidden.ndim != 3 or query_hidden.shape[0] != batch:
            raise ValueError("query_hidden must have shape [batch, tokens, source_width]")
        if query_mask.shape != query_hidden.shape[:2] or query_mask.dtype != torch.bool:
            raise ValueError("query_mask must be bool [batch, tokens]")
        if bool((query_mask.sum(dim=1) < 1).any()):
            raise ValueError("every query card must contain at least one token")
        if query_address.shape != (batch, self.config.address_width):
            raise ValueError("query_address has the wrong shape")
        if not bool(torch.isfinite(query_address).all()):
            raise ValueError("query_address contains non-finite values")

        initial_payloads = self.object_head(self.card_encoder(object_hidden, object_mask))
        initial_payloads = initial_payloads * object_present.to(dtype=initial_payloads.dtype).unsqueeze(-1)
        operation_states = self.operation_head(self.card_encoder(operation_hidden, operation_mask))
        query_state = self.query_head(
            self.card_encoder(query_hidden.unsqueeze(1), query_mask.unsqueeze(1)).squeeze(1)
        )
        operation_active = operation_mask.any(dim=2)
        source_weights, source_similarity = _registered_route(
            operation_source_addresses,
            object_addresses,
            object_present,
            operation_active,
            similarity_floor=self.config.route_similarity_floor,
        )
        target_weights, target_similarity = _registered_route(
            operation_target_addresses,
            object_addresses,
            object_present,
            operation_active,
            similarity_floor=self.config.route_similarity_floor,
        )
        query_weights, query_similarity = _registered_route(
            query_address.unsqueeze(1),
            object_addresses,
            object_present,
            torch.ones(batch, 1, dtype=torch.bool, device=query_address.device),
            similarity_floor=self.config.route_similarity_floor,
        )
        return PartitionedBoundaryOutput(
            object_addresses=object_addresses,
            object_present=object_present,
            initial_payloads=initial_payloads,
            operation_states=operation_states,
            operation_active=operation_active,
            source_weights=source_weights.to(dtype=initial_payloads.dtype),
            target_weights=target_weights.to(dtype=initial_payloads.dtype),
            query_state=query_state,
            query_weights=query_weights.squeeze(1).to(dtype=initial_payloads.dtype),
            route_similarities={
                "source": source_similarity,
                "target": target_similarity,
                "query": query_similarity.squeeze(1),
            },
        )


class TargetOnlyTransition(nn.Module):
    """Shared gate-free operation which overwrites only the registered target."""

    def __init__(self, config: C1UConfig) -> None:
        super().__init__()
        width = config.payload_width
        self.operator = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, width),
        )

    def forward(
        self,
        payloads: Tensor,
        operation_state: Tensor,
        source_weights: Tensor,
        target_weights: Tensor,
        active: Tensor,
    ) -> Tensor:
        source = torch.einsum("bk,bkd->bd", source_weights, payloads)
        target = torch.einsum("bk,bkd->bd", target_weights, payloads)
        proposed_target = self.operator(torch.cat((source, target, operation_state), dim=-1))
        target_delta = proposed_target - target
        candidate = payloads + target_weights.unsqueeze(-1) * target_delta.unsqueeze(1)
        active = active.to(dtype=payloads.dtype).view(-1, 1, 1)
        return active * candidate + (1.0 - active) * payloads


class QueryReadout(nn.Module):
    def __init__(self, config: C1UConfig) -> None:
        super().__init__()
        width = config.payload_width
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, width),
        )
        self.final_norm = nn.LayerNorm(width)
        self.answer_head = nn.Linear(width, config.answer_classes)

    def forward(self, selected: Tensor, query_state: Tensor) -> Tensor:
        fused = self.fusion(torch.cat((selected, query_state, selected * query_state), dim=-1))
        return self.answer_head(self.final_norm(fused))


class C1UModel(nn.Module):
    """Causal-card workspace whose deployment graph is source-card-only."""

    def __init__(self, config: C1UConfig = C1UConfig()) -> None:
        super().__init__()
        if config.source_width < 1 or config.payload_width < 4:
            raise ValueError("source/payload widths are too small")
        if config.payload_width % 2:
            raise ValueError("payload width must be even")
        if config.address_width < 8:
            raise ValueError("address width must be at least eight")
        if config.slots < 3 or config.operations < 2:
            raise ValueError("C1U requires at least three slots and two operations")
        if config.answer_classes != contract.ANSWER_CLASSES:
            raise ValueError("C1U freezes raw A-I answer logits")
        if not 0.0 < config.route_similarity_floor <= 1.0:
            raise ValueError("route similarity floor must be in (0, 1]")
        self.config = config
        self.boundary = PartitionedBoundary(config)
        self.transition = TargetOnlyTransition(config)
        self.readout = QueryReadout(config)

    def rollout(
        self,
        boundary: PartitionedBoundaryOutput,
        *,
        disable_recurrence: bool = False,
        initial_payloads: Tensor | None = None,
    ) -> Tensor:
        payloads = boundary.initial_payloads if initial_payloads is None else initial_payloads
        if payloads.shape != boundary.initial_payloads.shape:
            raise ValueError("initial payload intervention has the wrong shape")
        trajectory = [payloads]
        for step in range(self.config.operations):
            if not disable_recurrence:
                payloads = self.transition(
                    payloads,
                    boundary.operation_states[:, step],
                    boundary.source_weights[:, step],
                    boundary.target_weights[:, step],
                    boundary.operation_active[:, step],
                )
            trajectory.append(payloads)
        return torch.stack(trajectory, dim=1)

    def logits_from_payloads(
        self, payloads: Tensor, boundary: PartitionedBoundaryOutput
    ) -> tuple[Tensor, Tensor]:
        if payloads.shape != boundary.initial_payloads.shape:
            raise ValueError("payload state has the wrong shape")
        selected = torch.einsum("bk,bkd->bd", boundary.query_weights, payloads)
        return self.readout(selected, boundary.query_state), selected

    def forward_from_boundary(
        self,
        boundary: PartitionedBoundaryOutput,
        *,
        disable_recurrence: bool = False,
        initial_payloads: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:
        trajectory = self.rollout(
            boundary,
            disable_recurrence=disable_recurrence,
            initial_payloads=initial_payloads,
        )
        logits, selected = self.logits_from_payloads(trajectory[:, -1], boundary)
        result: dict[str, Any] = {
            "logits": logits,
            "initial_payloads": trajectory[:, 0],
            "final_payloads": trajectory[:, -1],
            "selected_payload": selected,
            "source_weights": boundary.source_weights,
            "target_weights": boundary.target_weights,
            "query_weights": boundary.query_weights,
            "operation_active": boundary.operation_active,
        }
        if return_trajectory:
            result["trajectory"] = trajectory
        return result

    def forward(
        self,
        object_hidden: Tensor,
        object_mask: Tensor,
        object_addresses: Tensor,
        object_present: Tensor,
        operation_hidden: Tensor,
        operation_mask: Tensor,
        operation_source_addresses: Tensor,
        operation_target_addresses: Tensor,
        query_hidden: Tensor,
        query_mask: Tensor,
        query_address: Tensor,
        *,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:
        boundary = self.boundary(
            object_hidden,
            object_mask,
            object_addresses,
            object_present,
            operation_hidden,
            operation_mask,
            operation_source_addresses,
            operation_target_addresses,
            query_hidden,
            query_mask,
            query_address,
        )
        return self.forward_from_boundary(boundary, return_trajectory=return_trajectory)

    def integrity_report(self) -> dict[str, Any]:
        # ``answer_head`` is the legitimate raw A-I deployment head.  Only
        # target/ledger concepts are forbidden from parameter names.
        forbidden_fragments = (
            "family",
            "ast",
            "counterfactual",
            "factor_cell",
            "factorial_group",
            "label_mapping",
            "semantic_answer",
            "support_slot",
            "task_causal_arity",
            "valid_choice",
            "teacher",
            "trace_target",
        )
        parameter_names = [name for name, _ in self.named_parameters()]
        forbidden_parameters = sorted(
            name for name in parameter_names if any(fragment in name.casefold() for fragment in forbidden_fragments)
        )
        signature = inspect.signature(self.forward)
        public_parameters = [
            name for name in signature.parameters if name != "return_trajectory"
        ]
        forward_forbidden_fragments = tuple(
            fragment.casefold() for fragment in contract.FORBIDDEN_FORWARD_FIELDS
        )
        forbidden_forward = sorted(
            name
            for name in public_parameters
            if any(fragment in name.casefold() for fragment in forward_forbidden_fragments)
        )
        dense_card_mixers = sorted(
            name
            for name, module in self.named_modules()
            if isinstance(module, nn.MultiheadAttention)
            or "selfattention" in type(module).__name__.casefold()
        )
        transition_parameter_names = [
            name for name in parameter_names if name.startswith("transition.")
        ]
        learned_write_gate_parameters = sorted(
            name for name in transition_parameter_names if "gate" in name.casefold()
        )
        final_projection = self.transition.operator[-1]
        gate_free_transition = (
            isinstance(final_projection, nn.Linear)
            and final_projection.out_features == self.config.payload_width
            and not learned_write_gate_parameters
        )
        exact_forward_contract = public_parameters == list(contract.FORWARD_FIELDS)
        return {
            "passed": not forbidden_parameters
            and not forbidden_forward
            and not dense_card_mixers
            and exact_forward_contract
            and gate_free_transition,
            "architecture": "publicly-grounded-gate-free-workspace",
            "public_forward_parameters": public_parameters,
            "exact_forward_contract": exact_forward_contract,
            "whole_record_hidden_accepted": False,
            "card_encoder_shared": True,
            "card_axis_mixing_before_routing": False,
            "public_address_routing": "hard cosine match with fail-closed floor",
            "source_slot_updates": False,
            "target_only_transition": True,
            "gate_free_target_overwrite": gate_free_transition,
            "learned_write_gate": False,
            "learned_write_gate_parameters": learned_write_gate_parameters,
            "transition_operator_output_width": int(final_projection.out_features),
            "transition_shared_across_steps": True,
            "query_reads_final_registered_owner_only": True,
            "forbidden_parameter_names": forbidden_parameters,
            "forbidden_forward_parameters": forbidden_forward,
            "dense_card_mixers": dense_card_mixers,
            "config": asdict(self.config),
        }

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "deployment_parameters": total,
        }


__all__ = [
    "C1UConfig",
    "C1UModel",
    "IndependentCardEncoder",
    "PartitionedBoundary",
    "PartitionedBoundaryOutput",
    "QueryReadout",
    "TargetOnlyTransition",
]
