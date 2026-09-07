from __future__ import annotations

"""C1S Addressed Content Workspace.

The deployment path accepts only frozen public-source hidden states and their
padding mask.  Addresses, operation routes, and the final query are learned
from that source.  No dataset family, AST, answer, trace, or oracle pointer is
accepted by the model.
"""

from dataclasses import asdict, dataclass, replace
import inspect
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _fixed_fourier(count: int, width: int) -> Tensor:
    positions = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    half = width // 2
    if half:
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(10_000.0), half, dtype=torch.float32) * -1.0
        )
        encoded = torch.cat(
            ((positions * frequencies.unsqueeze(0)).sin(), (positions * frequencies.unsqueeze(0)).cos()),
            dim=1,
        )
    else:
        encoded = positions.new_zeros(count, 0)
    return F.pad(encoded, (0, width - encoded.shape[1]))


@dataclass(frozen=True)
class C1SConfig:
    source_width: int = 2_048
    payload_width: int = 512
    address_width: int = 64
    slots: int = 8
    operations: int = 10
    reader_depth: int = 2
    attention_heads: int = 8
    ffn_width: int = 2_048
    answer_classes: int = 9
    auxiliary_width: int = 128
    route_temperature: float = 0.25
    training_auxiliary: bool = True


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(value)) * self.up(value))


class CrossAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("payload width must be divisible by attention heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def _heads(self, value: Tensor) -> Tensor:
        return value.view(value.shape[0], value.shape[1], self.heads, self.head_width).transpose(1, 2)

    def forward(self, latent: Tensor, source: Tensor, source_mask: Tensor) -> Tensor:
        query = self._heads(self.query(latent))
        key = self._heads(self.key(source))
        value = self._heads(self.value(source))
        attention_mask = torch.zeros(
            (source.shape[0], 1, 1, source.shape[1]),
            dtype=query.dtype,
            device=query.device,
        ).masked_fill(~source_mask[:, None, None, :], torch.finfo(query.dtype).min)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view_as(latent)
        return self.out(attended)


class SourceReaderLayer(nn.Module):
    """Independent query-to-source reading; it never mixes latent slots."""

    def __init__(self, config: C1SConfig) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(config.payload_width)
        self.ffn_norm = nn.LayerNorm(config.payload_width)
        self.cross = CrossAttention(config.payload_width, config.attention_heads)
        self.ffn = SwiGLU(config.payload_width, config.ffn_width)

    def forward(self, latent: Tensor, source: Tensor, source_mask: Tensor) -> Tensor:
        latent = latent + self.cross(self.cross_norm(latent), source, source_mask)
        return latent + self.ffn(self.ffn_norm(latent))


def _normalise_address(value: Tensor) -> Tensor:
    return F.normalize(value.float(), dim=-1, eps=1.0e-8).to(dtype=value.dtype)


def _route_logits(query: Tensor, addresses: Tensor, temperature: float) -> Tensor:
    query = _normalise_address(query)
    addresses = _normalise_address(addresses)
    return torch.einsum("btd,bkd->btk", query, addresses) / float(temperature)


def _hard_straight_through(logits: Tensor, *, training: bool) -> Tensor:
    soft = torch.softmax(logits, dim=-1)
    hard = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).to(dtype=soft.dtype)
    return hard + soft - soft.detach() if training else hard


@dataclass
class BoundaryOutput:
    addresses: Tensor
    payloads: Tensor
    operation_states: Tensor
    source_weights: Tensor
    target_weights: Tensor
    operation_active: Tensor
    query_key: Tensor
    presence_logits: Tensor

    def permuted(self, permutation: Tensor) -> BoundaryOutput:
        permutation = permutation.to(device=self.addresses.device, dtype=torch.long)
        if permutation.ndim != 1 or permutation.numel() != self.addresses.shape[1]:
            raise ValueError("slot permutation has the wrong shape")
        expected = torch.arange(permutation.numel(), device=permutation.device)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("slot permutation must contain every slot exactly once")
        return BoundaryOutput(
            addresses=self.addresses[:, permutation],
            payloads=self.payloads[:, permutation],
            operation_states=self.operation_states,
            source_weights=self.source_weights[:, :, permutation],
            target_weights=self.target_weights[:, :, permutation],
            operation_active=self.operation_active,
            query_key=self.query_key,
            presence_logits=self.presence_logits[:, permutation],
        )


class AddressedBoundary(nn.Module):
    """Public-source reader with routed operation/entity refinement."""

    def __init__(self, config: C1SConfig) -> None:
        super().__init__()
        self.config = config
        width = config.payload_width
        self.input_norm = nn.LayerNorm(config.source_width, elementwise_affine=False)
        self.source_projection = nn.Linear(config.source_width, width)
        self.source_norm = nn.LayerNorm(width)

        self.slot_seed = nn.Parameter(torch.randn(width) * 0.02)
        self.operation_seed = nn.Parameter(torch.randn(width) * 0.02)
        self.query_seed = nn.Parameter(torch.randn(width) * 0.02)
        self.register_buffer("slot_fourier", _fixed_fourier(config.slots, width), persistent=True)
        self.register_buffer("operation_fourier", _fixed_fourier(config.operations, width), persistent=True)

        self.entity_readers = nn.ModuleList(SourceReaderLayer(config) for _ in range(config.reader_depth))
        self.operation_readers = nn.ModuleList(SourceReaderLayer(config) for _ in range(config.reader_depth))
        self.query_readers = nn.ModuleList(SourceReaderLayer(config) for _ in range(config.reader_depth))

        self.address_head = nn.Linear(width, config.address_width, bias=False)
        self.payload_head = nn.Linear(width, width)
        self.source_route_head = nn.Linear(width, config.address_width, bias=False)
        self.target_route_head = nn.Linear(width, config.address_width, bias=False)
        self.query_key_head = nn.Linear(width, config.address_width, bias=False)
        self.operation_refine = nn.Linear(3 * width, width)
        self.entity_refine = nn.Linear(2 * width, width)
        self.query_refine = nn.Linear(2 * width, width)
        self.operation_context = nn.Linear(width, width)
        self.operation_active = nn.Linear(width, 1)
        self.presence_head = nn.Linear(width, 1)
        self.entity_norm = nn.LayerNorm(width)
        self.operation_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)

    def _read(self, latent: Tensor, readers: nn.ModuleList, source: Tensor, source_mask: Tensor) -> Tensor:
        for reader in readers:
            latent = reader(latent, source, source_mask)
        return latent

    def forward(self, source_hidden: Tensor, source_mask: Tensor) -> BoundaryOutput:
        if not torch.is_autocast_enabled(source_hidden.device.type):
            source_hidden = source_hidden.to(dtype=self.source_projection.weight.dtype)
        source = self.source_norm(self.source_projection(self.input_norm(source_hidden)))
        batch = source.shape[0]
        entities = self.slot_seed.view(1, 1, -1) + self.slot_fourier.to(source.dtype).unsqueeze(0)
        operations = self.operation_seed.view(1, 1, -1) + self.operation_fourier.to(source.dtype).unsqueeze(0)
        query = self.query_seed.view(1, 1, -1)
        entities = self._read(entities.expand(batch, -1, -1), self.entity_readers, source, source_mask)
        operations = self._read(operations.expand(batch, -1, -1), self.operation_readers, source, source_mask)
        query = self._read(query.expand(batch, -1, -1), self.query_readers, source, source_mask)

        coarse_addresses = _normalise_address(self.address_head(entities))
        coarse_payloads = self.payload_head(entities)
        source_weights = _hard_straight_through(
            _route_logits(self.source_route_head(operations), coarse_addresses, self.config.route_temperature),
            training=self.training,
        )
        target_weights = _hard_straight_through(
            _route_logits(self.target_route_head(operations), coarse_addresses, self.config.route_temperature),
            training=self.training,
        )
        source_read = torch.einsum("btk,bkd->btd", source_weights, coarse_payloads)
        target_read = torch.einsum("btk,bkd->btd", target_weights, coarse_payloads)
        operations = self.operation_norm(
            operations + self.operation_refine(torch.cat((operations, source_read, target_read), dim=-1))
        )

        feedback_weights = source_weights + target_weights
        feedback = torch.einsum("btk,btd->bkd", feedback_weights, operations)
        counts = feedback_weights.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        feedback = feedback / counts
        entities = self.entity_norm(
            entities + self.entity_refine(torch.cat((entities, feedback), dim=-1))
        )

        addresses = _normalise_address(self.address_head(entities))
        payloads = self.payload_head(entities)
        source_weights = _hard_straight_through(
            _route_logits(self.source_route_head(operations), addresses, self.config.route_temperature),
            training=self.training,
        )
        target_weights = _hard_straight_through(
            _route_logits(self.target_route_head(operations), addresses, self.config.route_temperature),
            training=self.training,
        )

        coarse_query_key = self.query_key_head(query)
        coarse_query_weights = _hard_straight_through(
            _route_logits(coarse_query_key, addresses, self.config.route_temperature),
            training=self.training,
        )
        query_read = torch.einsum("bqk,bkd->bqd", coarse_query_weights, payloads)
        query = self.query_norm(query + self.query_refine(torch.cat((query, query_read), dim=-1)))

        return BoundaryOutput(
            addresses=addresses,
            payloads=payloads,
            operation_states=self.operation_context(operations),
            source_weights=source_weights,
            target_weights=target_weights,
            operation_active=torch.sigmoid(self.operation_active(operations).squeeze(-1)),
            query_key=self.query_key_head(query).squeeze(1),
            presence_logits=self.presence_head(entities).squeeze(-1),
        )


class AddressRoutedTransition(nn.Module):
    """One shared source/target-gated payload update with no all-slot mixing."""

    def __init__(self, config: C1SConfig) -> None:
        super().__init__()
        width = config.payload_width
        self.operator = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, 2 * width + 2),
        )

    def forward(
        self,
        payloads: Tensor,
        operation_state: Tensor,
        source_weights: Tensor,
        target_weights: Tensor,
        active: Tensor,
    ) -> Tensor:
        source_read = torch.einsum("bk,bkd->bd", source_weights, payloads)
        target_read = torch.einsum("bk,bkd->bd", target_weights, payloads)
        update = self.operator(torch.cat((source_read, target_read, operation_state), dim=-1))
        width = payloads.shape[-1]
        proposed_source, proposed_target, gate_logits = torch.split(update, (width, width, 2), dim=-1)
        source_delta = (proposed_source - source_read).unsqueeze(1)
        target_delta = (proposed_target - target_read).unsqueeze(1)
        candidate = payloads + source_weights.unsqueeze(-1) * torch.sigmoid(gate_logits[:, :1]).unsqueeze(1) * source_delta
        candidate = candidate + target_weights.unsqueeze(-1) * torch.sigmoid(gate_logits[:, 1:]).unsqueeze(1) * target_delta
        active = active.to(dtype=payloads.dtype).view(-1, 1, 1)
        return active * candidate + (1.0 - active) * payloads


class TrainingAuxiliary(nn.Module):
    """Training-only feature heads; their outputs never feed deployment logits."""

    def __init__(self, config: C1SConfig) -> None:
        super().__init__()
        self.state = nn.Linear(config.payload_width, config.auxiliary_width)
        self.address = nn.Linear(config.address_width, config.auxiliary_width)
        self.section = nn.Linear(config.payload_width, config.auxiliary_width)

    def forward(self, trajectory: Tensor, boundary: BoundaryOutput) -> dict[str, Tensor]:
        return {
            "state_features": self.state(trajectory),
            "address_features": self.address(boundary.addresses),
            "section_features": self.section(boundary.operation_states),
        }


class C1SModel(nn.Module):
    """Source-closed addressed workspace with an input-conditioned query."""

    def __init__(self, config: C1SConfig = C1SConfig()) -> None:
        super().__init__()
        if config.slots < 1:
            raise ValueError("C1S requires at least one slot")
        if config.operations != 10:
            raise ValueError("C1S freezes ten recurrent operations")
        if config.answer_classes != 9:
            raise ValueError("C1S freezes raw A-I answer logits")
        if config.route_temperature <= 0:
            raise ValueError("route temperature must be positive")
        self.config = config
        self.boundary = AddressedBoundary(config)
        self.transition = AddressRoutedTransition(config)
        self.final_norm = nn.LayerNorm(config.payload_width)
        self.answer_head = nn.Linear(config.payload_width, config.answer_classes)
        self.training_auxiliary = TrainingAuxiliary(config) if config.training_auxiliary else None

    @property
    def has_training_auxiliary(self) -> bool:
        return self.training_auxiliary is not None

    def _validate_source(self, source_hidden: Tensor, source_mask: Tensor) -> None:
        if source_hidden.ndim != 3 or source_hidden.shape[-1] != self.config.source_width:
            raise ValueError("source_hidden must have shape [batch, full_tokens, source_width]")
        if source_mask.shape != source_hidden.shape[:2] or source_mask.dtype != torch.bool:
            raise ValueError("source_mask must be bool [batch, full_tokens]")
        if source_hidden.shape[1] < 1 or bool((source_mask.sum(dim=1) < 1).any()):
            raise ValueError("every row must contain at least one public token")

    def query_weights(self, addresses: Tensor, query_key: Tensor) -> Tensor:
        logits = _route_logits(query_key.unsqueeze(1), addresses, self.config.route_temperature).squeeze(1)
        return _hard_straight_through(logits, training=self.training)

    def rollout(
        self,
        boundary: BoundaryOutput,
        *,
        disable_recurrence: bool = False,
        wrong_start_permutation: Tensor | None = None,
    ) -> Tensor:
        payloads = boundary.payloads
        if wrong_start_permutation is not None:
            permutation = wrong_start_permutation.to(device=payloads.device, dtype=torch.long)
            if permutation.ndim != 1 or permutation.numel() != self.config.slots:
                raise ValueError("wrong-start permutation has the wrong shape")
            payloads = payloads[:, permutation]
        trajectory: list[Tensor] = []
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

    def logits_from_state(self, payloads: Tensor, addresses: Tensor, query_key: Tensor) -> tuple[Tensor, Tensor]:
        weights = self.query_weights(addresses, query_key)
        selected = torch.einsum("bk,bkd->bd", weights, payloads)
        return self.answer_head(self.final_norm(selected)), weights

    def forward_from_boundary(
        self,
        boundary: BoundaryOutput,
        *,
        return_trajectory: bool = False,
        return_auxiliary: bool = False,
        disable_recurrence: bool = False,
        wrong_start_permutation: Tensor | None = None,
    ) -> dict[str, Any]:
        trajectory = self.rollout(
            boundary,
            disable_recurrence=disable_recurrence,
            wrong_start_permutation=wrong_start_permutation,
        )
        logits, query_weights = self.logits_from_state(
            trajectory[:, -1], boundary.addresses, boundary.query_key
        )
        result: dict[str, Any] = {
            "logits": logits,
            "addresses": boundary.addresses,
            "initial_payloads": boundary.payloads,
            "final_payloads": trajectory[:, -1],
            "query_weights": query_weights,
        }
        if return_trajectory:
            result["trajectory"] = trajectory
        if return_auxiliary:
            if self.training_auxiliary is None:
                result["auxiliary"] = None
            else:
                result["auxiliary"] = self.training_auxiliary(trajectory, boundary)
        return result

    def forward(
        self,
        source_hidden: Tensor,
        source_mask: Tensor,
        *,
        return_trajectory: bool = False,
        return_auxiliary: bool = False,
    ) -> dict[str, Any]:
        self._validate_source(source_hidden, source_mask)
        boundary = self.boundary(source_hidden, source_mask)
        return self.forward_from_boundary(
            boundary,
            return_trajectory=return_trajectory,
            return_auxiliary=return_auxiliary,
        )

    def integrity_report(self) -> dict[str, Any]:
        forbidden_fragments = (
            "family",
            "route_embedding",
            "ast",
            "answer_label",
            "teacher",
            "trace",
            "valid_choice",
            "reasoning_budget",
        )
        forbidden_parameters = sorted(
            name
            for name, _ in self.named_parameters()
            if any(fragment in name.casefold() for fragment in forbidden_fragments)
        )
        forward_parameters = list(inspect.signature(self.forward).parameters)
        forbidden_forward = sorted(
            name
            for name in forward_parameters
            if any(fragment in name.casefold() for fragment in forbidden_fragments)
        )
        dense_slot_modules = sorted(
            name
            for name, module in self.named_modules()
            if "selfattention" in type(module).__name__.casefold()
            or "densecore" in type(module).__name__.casefold()
        )
        return {
            "passed": not forbidden_parameters and not forbidden_forward and not dense_slot_modules,
            "architecture": "addressed-content-workspace",
            "state_form": "S_t=({address_k,payload_k},input_conditioned_query)",
            "public_forward_parameters": forward_parameters,
            "public_source_hidden_only": forward_parameters[:2] == ["source_hidden", "source_mask"],
            "source_closed_after_boundary": True,
            "opaque_address_semantic_embedding": False,
            "address_usage": "hard-value soft-gradient routing and paired permutation only",
            "dense_all_slot_attention": False,
            "global_answer_query": False,
            "input_conditioned_query": True,
            "transition_shared_across_steps": True,
            "transition_instances": 1,
            "recurrent_steps": self.config.operations,
            "answer_source": "query-selected final payload only",
            "training_auxiliary_present": self.training_auxiliary is not None,
            "forbidden_parameter_names": forbidden_parameters,
            "forbidden_forward_parameters": forbidden_forward,
            "dense_slot_modules": dense_slot_modules,
            "config": asdict(self.config),
        }

    def parameter_report(self) -> dict[str, int]:
        auxiliary = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("training_auxiliary.")
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "training_auxiliary_parameters": auxiliary,
            "deployment_parameters": total - auxiliary,
        }


def strip_training_auxiliary(model: C1SModel) -> C1SModel:
    """Create a deployment-only C1S model with strict state loading."""

    deployment = C1SModel(replace(model.config, training_auxiliary=False))
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("training_auxiliary.")
    }
    deployment.load_state_dict(state, strict=True)
    deployment.train(model.training)
    return deployment


__all__ = [
    "AddressRoutedTransition",
    "AddressedBoundary",
    "BoundaryOutput",
    "C1SConfig",
    "C1SModel",
    "TrainingAuxiliary",
    "strip_training_auxiliary",
]
