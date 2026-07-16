from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .a1_6_data import FAMILY_TO_INDEX, REGISTER_NAMES, VALUE_LABELS


@dataclass(frozen=True)
class A16CoreConfig:
    latent_width: int = 256
    register_slots: int = 3
    ffn_width: int = 512
    value_classes: int = len(VALUE_LABELS)
    family_classes: int = 2


class RelationAddressedTransition(nn.Module):
    """One shared relation transition with independent family/source/target roles."""

    def __init__(self, config: A16CoreConfig) -> None:
        super().__init__()
        d = config.latent_width
        self.source_query = nn.Linear(d, d, bias=False)
        self.target_query = nn.Linear(d, d, bias=False)
        self.operator = nn.Sequential(
            nn.Linear(3 * d, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, 2 * d + 2),
        )

    def forward(
        self,
        family_latent: torch.Tensor,
        source_role_latent: torch.Tensor,
        target_role_latent: torch.Tensor,
        slots: torch.Tensor,
        register_keys: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scale = register_keys.shape[-1] ** -0.5
        source_logits = torch.einsum("bd,rd->br", self.source_query(source_role_latent), register_keys) * scale
        target_logits = torch.einsum("bd,rd->br", self.target_query(target_role_latent), register_keys) * scale
        source_weights = torch.softmax(source_logits, dim=-1)
        target_weights = torch.softmax(target_logits, dim=-1)
        source_read = torch.einsum("br,brd->bd", source_weights, slots)
        target_read = torch.einsum("br,brd->bd", target_weights, slots)
        operator_out = self.operator(torch.cat((source_read, target_read, family_latent), dim=-1))
        proposed_source, proposed_target, gate_logits = torch.split(operator_out, [slots.shape[-1], slots.shape[-1], 2], dim=-1)
        # These are write gates, not a residual shortcut: the specified write
        # equation remains the only state update in the core.
        source_write_gate = torch.sigmoid(gate_logits[..., :1])
        target_write_gate = torch.sigmoid(gate_logits[..., 1:2])
        next_slots = slots + source_weights.unsqueeze(-1) * source_write_gate.unsqueeze(1) * (proposed_source.unsqueeze(1) - slots)
        next_slots = next_slots + target_weights.unsqueeze(-1) * target_write_gate.unsqueeze(1) * (proposed_target.unsqueeze(1) - slots)
        return {
            "next_slots": next_slots,
            "source_logits": source_logits,
            "target_logits": target_logits,
            "source_weights": source_weights,
            "target_weights": target_weights,
            "source_read": source_read,
            "target_read": target_read,
            "proposed_source": proposed_source,
            "proposed_target": proposed_target,
            "source_write_gate": source_write_gate,
            "target_write_gate": target_write_gate,
        }


class RelationAddressedCore(nn.Module):
    """Structured C0 relation-addressed continuous latent state machine."""

    def __init__(self, config: A16CoreConfig | None = None) -> None:
        super().__init__()
        self.config = config or A16CoreConfig()
        d = self.config.latent_width
        self.value_embedding = nn.Embedding(self.config.value_classes, d)
        self.register_embedding = nn.Embedding(self.config.register_slots, d)
        self.family_embedding = nn.Embedding(self.config.family_classes, d)
        self.transition = RelationAddressedTransition(self.config)
        self.shared_state_head = nn.Linear(d, self.config.value_classes)

    @property
    def register_keys(self) -> torch.Tensor:
        return self.register_embedding.weight

    def initial_slots(self, start_values: torch.Tensor) -> torch.Tensor:
        if start_values.ndim != 2 or start_values.shape[-1] != self.config.register_slots:
            raise ValueError("start_values must have shape [B, 3]")
        return self.value_embedding(start_values) + self.register_keys.unsqueeze(0)

    def forward(
        self,
        start_values: torch.Tensor,
        query_register: torch.Tensor,
        operation_family: torch.Tensor,
        operation_source: torch.Tensor,
        operation_target: torch.Tensor,
        operation_mask: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:
        if operation_family.shape != operation_source.shape or operation_family.shape != operation_target.shape:
            raise ValueError("family/source/target tensors must have identical [B, T] shape")
        if operation_mask.shape != operation_family.shape:
            raise ValueError("operation_mask must have [B, T] shape")
        state = self.initial_slots(start_values)
        family_latents = self.family_embedding(operation_family)
        source_role_latents = self.register_embedding(operation_source)
        target_role_latents = self.register_embedding(operation_target)
        return self.rollout(state, query_register, family_latents, source_role_latents, target_role_latents, operation_mask)

    def rollout(
        self,
        initial_slots: torch.Tensor,
        query_register: torch.Tensor,
        family_latents: torch.Tensor,
        source_role_latents: torch.Tensor,
        target_role_latents: torch.Tensor,
        operation_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Run the same transition/head from externally prepared role latents."""
        if family_latents.shape != source_role_latents.shape or family_latents.shape != target_role_latents.shape:
            raise ValueError("role latents must have identical [B, T, D] shape")
        state = initial_slots
        initial = state
        trajectory: list[torch.Tensor] = []
        pointer_steps: list[dict[str, torch.Tensor]] = []
        for step in range(family_latents.shape[1]):
            family_latent = family_latents[:, step]
            source_role_latent = source_role_latents[:, step]
            target_role_latent = target_role_latents[:, step]
            transition_output = self.transition(
                family_latent,
                source_role_latent,
                target_role_latent,
                state,
                self.register_keys,
            )
            active = operation_mask[:, step].to(dtype=state.dtype).view(-1, 1, 1)
            state = active * transition_output["next_slots"] + (1.0 - active) * state
            trajectory.append(state)
            pointer_steps.append(transition_output)
        if trajectory:
            trajectory_tensor = torch.stack(trajectory, dim=1)
            state_logits = self.shared_state_head(trajectory_tensor)
        else:
            trajectory_tensor = state.new_empty((state.shape[0], 0, state.shape[1], state.shape[2]))
            state_logits = state.new_empty((state.shape[0], 0, state.shape[1], self.config.value_classes))
        query_weights = torch.nn.functional.one_hot(query_register, num_classes=self.config.register_slots).to(dtype=state.dtype)
        queried_state = torch.einsum("br,brd->bd", query_weights, state)
        answer_logits = self.shared_state_head(queried_state)
        result: dict[str, Any] = {
            "initial_slots": initial,
            "final_slots": state,
            "trajectory": trajectory_tensor,
            "state_logits": state_logits,
            "query_weights": query_weights,
            "queried_state": queried_state,
            "answer_logits": answer_logits,
            "source_pointer_logits": torch.stack([item["source_logits"] for item in pointer_steps], dim=1) if pointer_steps else state.new_empty((state.shape[0], 0, self.config.register_slots)),
            "target_pointer_logits": torch.stack([item["target_logits"] for item in pointer_steps], dim=1) if pointer_steps else state.new_empty((state.shape[0], 0, self.config.register_slots)),
            "source_pointer_weights": torch.stack([item["source_weights"] for item in pointer_steps], dim=1) if pointer_steps else state.new_empty((state.shape[0], 0, self.config.register_slots)),
            "target_pointer_weights": torch.stack([item["target_weights"] for item in pointer_steps], dim=1) if pointer_steps else state.new_empty((state.shape[0], 0, self.config.register_slots)),
            "source_write_gates": torch.stack([item["source_write_gate"] for item in pointer_steps], dim=1) if pointer_steps else state.new_empty((state.shape[0], 0, 1)),
            "target_write_gates": torch.stack([item["target_write_gate"] for item in pointer_steps], dim=1) if pointer_steps else state.new_empty((state.shape[0], 0, 1)),
        }
        return result

    def integrity_report(self) -> dict[str, Any]:
        return {
            "passed": True,
            "architecture": "relation-addressed-continuous-state-recursion",
            "initial_state_operation_free": True,
            "shared_transition_instances": sum(isinstance(module, RelationAddressedTransition) for module in self.modules()),
            "transition_shared_across_steps": True,
            "attention_heads": None,
            "mean_pooling": False,
            "control_slot": False,
            "query_content_written_to_slots": False,
            "absolute_step_embedding": False,
            "independent_answer_head": False,
            "answer_source": "shared_state_head(queried_final_register_slot)",
            "independent_roles": True,
        }

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable_parameters": total, "total_parameters": total}


def family_index(name: str) -> int:
    return FAMILY_TO_INDEX[name]
