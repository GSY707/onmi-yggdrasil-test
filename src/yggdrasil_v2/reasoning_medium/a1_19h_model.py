from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class A119HConfig:
    maximum_entities: int = 3
    workspace_slots: int = 8
    latent_width: int = 256
    ffn_width: int = 1024
    value_classes: int = 10
    family_classes: int = 2
    training_auxiliary: str = "trajectory_set_state"


def _masked_workspace_mean(
    slots: torch.Tensor, entity_mask: torch.Tensor, maximum_entities: int
) -> torch.Tensor:
    scratch = slots.shape[-2] - maximum_entities
    scratch_mask = torch.ones(
        (*entity_mask.shape[:-1], scratch),
        dtype=torch.bool,
        device=entity_mask.device,
    )
    mask = torch.cat((entity_mask, scratch_mask), dim=-1).to(dtype=slots.dtype)
    while mask.ndim < slots.ndim - 1:
        mask = mask.unsqueeze(1)
    return (slots * mask.unsqueeze(-1)).sum(dim=-2) / mask.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)


class A119HHandleAddressedTransition(nn.Module):
    """Shared semantic update; opaque handles are used only for equality routing."""

    def __init__(self, config: A119HConfig) -> None:
        super().__init__()
        d = config.latent_width
        self.maximum_entities = config.maximum_entities
        self.family_embedding = nn.Embedding(config.family_classes, d)
        self.operator = nn.Sequential(
            nn.Linear(3 * d, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, 2 * d + 2),
        )

    def forward(
        self,
        slots: torch.Tensor,
        family_embedding: torch.Tensor,
        source_weights: torch.Tensor,
        target_weights: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        entity = slots[:, : self.maximum_entities]
        source_read = torch.einsum("be,bed->bd", source_weights, entity)
        target_read = torch.einsum("be,bed->bd", target_weights, entity)
        update = self.operator(
            torch.cat((source_read, target_read, family_embedding), dim=-1)
        )
        proposed_source, proposed_target, gate_logits = torch.split(
            update, [slots.shape[-1], slots.shape[-1], 2], dim=-1
        )
        source_gate = torch.sigmoid(gate_logits[:, :1])
        target_gate = torch.sigmoid(gate_logits[:, 1:])
        next_entity = entity + source_weights.unsqueeze(-1) * source_gate.unsqueeze(
            1
        ) * (proposed_source.unsqueeze(1) - entity)
        next_entity = next_entity + target_weights.unsqueeze(-1) * target_gate.unsqueeze(
            1
        ) * (proposed_target.unsqueeze(1) - next_entity)
        candidate = torch.cat((next_entity, slots[:, self.maximum_entities :]), dim=1)
        active_float = active.to(dtype=slots.dtype).view(-1, 1, 1)
        return active_float * candidate + (1.0 - active_float) * slots


class A119HHybridReasoner(nn.Module):
    """Exchangeable entity payloads plus an opaque equality-address sidecar."""

    def __init__(self, config: A119HConfig) -> None:
        super().__init__()
        if config.maximum_entities < 1:
            raise ValueError("maximum_entities must be positive")
        if config.workspace_slots < config.maximum_entities:
            raise ValueError("workspace must contain all entity slots")
        if config.training_auxiliary not in {"none", "trajectory_set_state"}:
            raise ValueError("unknown A1.19H training auxiliary")
        self.config = config
        d = config.latent_width
        self.value_initializer = nn.Linear(config.value_classes, d, bias=False)
        self.shared_entity_seed = nn.Parameter(torch.randn(d) * 0.02)
        scratch = config.workspace_slots - config.maximum_entities
        self.scratch_slots = nn.Parameter(torch.randn(scratch, d) * 0.02)
        self.transition = A119HHandleAddressedTransition(config)
        self.output_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, config.value_classes)
        self.training_auxiliary_head: nn.Linear | None = None
        if config.training_auxiliary == "trajectory_set_state":
            self.training_auxiliary_head = nn.Linear(d, config.value_classes)

    def _initial_slots(
        self,
        entity_values: torch.Tensor,
        entity_mask: torch.Tensor,
        entity_payloads: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if entity_payloads is None:
            values = nn.functional.one_hot(
                entity_values, num_classes=self.config.value_classes
            ).to(dtype=self.value_initializer.weight.dtype)
            entity = self.value_initializer(values) + self.shared_entity_seed
        else:
            expected = (
                entity_values.shape[0],
                self.config.maximum_entities,
                self.config.latent_width,
            )
            if entity_payloads.shape != expected:
                raise ValueError(
                    f"A1.19H continuous entity payload shape must be {expected}"
                )
            entity = entity_payloads.to(dtype=self.value_initializer.weight.dtype)
        entity = entity * entity_mask.to(dtype=entity.dtype).unsqueeze(-1)
        scratch = self.scratch_slots.unsqueeze(0).expand(entity.shape[0], -1, -1)
        return torch.cat((entity, scratch), dim=1)

    def closure_targets(self, state_targets: torch.Tensor) -> torch.Tensor:
        safe = state_targets.clamp_min(0)
        one_hot = nn.functional.one_hot(
            safe, num_classes=self.config.value_classes
        ).to(dtype=self.value_initializer.weight.dtype)
        return self.value_initializer(one_hot) + self.shared_entity_seed

    def forward(
        self,
        entity_values: torch.Tensor,
        entity_handles: torch.Tensor,
        entity_mask: torch.Tensor,
        query_handle: torch.Tensor,
        operation_family: torch.Tensor,
        operation_source_handle: torch.Tensor,
        operation_target_handle: torch.Tensor,
        operation_mask: torch.Tensor,
        recurrent_steps: int,
        *,
        entity_payloads: torch.Tensor | None = None,
        disable_recurrence: bool = False,
        start_value_permutation: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if recurrent_steps < 1:
            raise ValueError("A1.19H recurrent_steps must be positive")
        if start_value_permutation is not None:
            entity_values = entity_values[:, start_value_permutation]
        slots = self._initial_slots(entity_values, entity_mask, entity_payloads)
        initial_slots = slots
        trajectory: list[torch.Tensor] = []
        operation_steps = operation_mask.shape[1]
        entity_mask_by_step = entity_mask.unsqueeze(1)
        source_weights = (
            (
                entity_handles.unsqueeze(1)
                == operation_source_handle.unsqueeze(-1)
            )
            & entity_mask_by_step
        ).to(dtype=slots.dtype)
        target_weights = (
            (
                entity_handles.unsqueeze(1)
                == operation_target_handle.unsqueeze(-1)
            )
            & entity_mask_by_step
        ).to(dtype=slots.dtype)
        family_embeddings = self.transition.family_embedding(operation_family)
        for step in range(recurrent_steps):
            if not disable_recurrence:
                index = min(step, operation_steps - 1)
                active = (
                    operation_mask[:, index]
                    if step < operation_steps
                    else torch.zeros_like(operation_mask[:, 0])
                )
                slots = self.transition(
                    slots,
                    family_embeddings[:, index],
                    source_weights[:, index],
                    target_weights[:, index],
                    active,
                )
            trajectory.append(slots)
        trajectory_tensor = torch.stack(trajectory, dim=1)
        normalized = self.output_norm(trajectory_tensor)
        entity_normalized = normalized[:, :, : self.config.maximum_entities]
        state_logits = self.state_head(entity_normalized)
        query_weights = (
            (entity_handles == query_handle.unsqueeze(-1)) & entity_mask
        ).to(dtype=state_logits.dtype)
        answer_logits = torch.einsum(
            "be,bev->bv", query_weights, state_logits[:, -1]
        )
        auxiliary_logits: torch.Tensor | None = None
        if self.training and self.training_auxiliary_head is not None:
            global_context = _masked_workspace_mean(
                normalized, entity_mask, self.config.maximum_entities
            )
            auxiliary_source = entity_normalized + global_context.unsqueeze(2)
            auxiliary_logits = self.training_auxiliary_head(auxiliary_source)
        return {
            "initial_slots": initial_slots,
            "trajectory": trajectory_tensor,
            "state_logits": state_logits,
            "answer_logits": answer_logits,
            "training_auxiliary_logits": auxiliary_logits,
        }

    def integrity_report(self) -> dict[str, Any]:
        forbidden = [
            name
            for name, _ in self.named_parameters()
            if any(
                fragment in name.lower()
                for fragment in ("amber", "cobalt", "jade", "copy", "swap")
            )
        ]
        auxiliary_outputs = (
            self.training_auxiliary_head.out_features
            if self.training_auxiliary_head is not None
            else 0
        )
        expected_auxiliary = (
            self.config.value_classes
            if self.config.training_auxiliary == "trajectory_set_state"
            else 0
        )
        return {
            "passed": not forbidden and auxiliary_outputs == expected_auxiliary,
            "architecture": "opaque-handle-exchangeable-hybrid-workspace",
            "state_form": "S_t=(opaque equality address sidecar, continuous payload)",
            "opaque_handle_semantic_embedding": False,
            "opaque_handle_usage": "equality routing only",
            "fixed_register_slot_semantics": False,
            "entity_payload_initialization_shared": True,
            "entity_state_head_shared": True,
            "transition_shared_across_entities_and_steps": True,
            "task_family_specific_transition": False,
            "copy_swap_executor_branches": False,
            "query_coupled_answer": True,
            "answer_source": "shared_state_head(final handle-addressed entity payload)",
            "training_auxiliary": self.config.training_auxiliary,
            "training_auxiliary_outputs_per_entity": auxiliary_outputs,
            "training_auxiliary_active_in_eval": False,
            "hard_reembedding_in_forward": False,
            "teacher_state_in_forward": False,
            "forbidden_parameter_names": forbidden,
            "config": asdict(self.config),
        }

    def parameter_report(self) -> dict[str, int]:
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "trainable_parameters": trainable,
            "total_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }
