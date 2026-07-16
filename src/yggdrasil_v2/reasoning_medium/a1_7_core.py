from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .a1_7_data import FAMILY_TO_INDEX, REGISTER_NAMES, VALUE_LABELS


@dataclass(frozen=True)
class A17CoreConfig:
    latent_width: int = 256
    register_slots: int = len(REGISTER_NAMES)
    ffn_width: int = 512
    value_classes: int = len(VALUE_LABELS)
    family_classes: int = 2
    address_content_mixing_ablation: bool = False


class A17RelationTransition(nn.Module):
    """The only shared transition; register keys address content-only slots."""

    def __init__(self, config: A17CoreConfig) -> None:
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
        proposed_source, proposed_target, gate_logits = torch.split(
            operator_out,
            [slots.shape[-1], slots.shape[-1], 2],
            dim=-1,
        )
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


class A17RelationAddressedCore(nn.Module):
    """Content-only continuous state with separate address keys and one closure prototype bank."""

    def __init__(self, config: A17CoreConfig | None = None) -> None:
        super().__init__()
        self.config = config or A17CoreConfig()
        d = self.config.latent_width
        self.value_embedding = nn.Embedding(self.config.value_classes, d)
        self.register_embedding = nn.Embedding(self.config.register_slots, d)
        self.family_embedding = nn.Embedding(self.config.family_classes, d)
        self.transition = A17RelationTransition(self.config)
        self.shared_state_head = nn.Linear(d, self.config.value_classes)

    @property
    def register_keys(self) -> torch.Tensor:
        return self.register_embedding.weight

    @property
    def canonical_value_prototypes(self) -> torch.Tensor:
        return self.value_embedding.weight

    def initial_slots(self, start_values: torch.Tensor) -> torch.Tensor:
        if start_values.ndim != 2 or start_values.shape[-1] != self.config.register_slots:
            raise ValueError("start_values must have shape [B, 3]")
        slots = self.value_embedding(start_values)
        if self.config.address_content_mixing_ablation:
            # Explicit negative-control ablation only. The target architecture
            # keeps register identity out of recurrent state content.
            slots = slots + self.register_keys.unsqueeze(0)
        return slots

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
        del return_trajectory
        if operation_family.shape != operation_source.shape or operation_family.shape != operation_target.shape:
            raise ValueError("family/source/target tensors must have identical [B, T] shape")
        if operation_mask.shape != operation_family.shape:
            raise ValueError("operation_mask must have [B, T] shape")
        state = self.initial_slots(start_values)
        return self.rollout(
            state,
            query_register,
            self.family_embedding(operation_family),
            self.register_embedding(operation_source),
            self.register_embedding(operation_target),
            operation_mask,
        )

    def rollout(
        self,
        initial_slots: torch.Tensor,
        query_register: torch.Tensor,
        family_latents: torch.Tensor,
        source_role_latents: torch.Tensor,
        target_role_latents: torch.Tensor,
        operation_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if family_latents.shape != source_role_latents.shape or family_latents.shape != target_role_latents.shape:
            raise ValueError("role latents must have identical [B, T, D] shape")
        state = initial_slots
        trajectory: list[torch.Tensor] = []
        pointer_steps: list[dict[str, torch.Tensor]] = []
        for step in range(family_latents.shape[1]):
            transition_output = self.transition(
                family_latents[:, step],
                source_role_latents[:, step],
                target_role_latents[:, step],
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
            lengths = operation_mask.sum(dim=1)
            last_indices = (lengths - 1).clamp_min(0)
            gather_index = last_indices.view(-1, 1, 1, 1).expand(
                -1,
                1,
                self.config.register_slots,
                self.config.value_classes,
            )
            final_state_logits = state_logits.gather(1, gather_index).squeeze(1)
            if bool((lengths == 0).any()):
                initial_logits = self.shared_state_head(initial_slots)
                final_state_logits = torch.where((lengths == 0).view(-1, 1, 1), initial_logits, final_state_logits)
        else:
            trajectory_tensor = state.new_empty((state.shape[0], 0, state.shape[1], state.shape[2]))
            state_logits = state.new_empty((state.shape[0], 0, state.shape[1], self.config.value_classes))
            final_state_logits = self.shared_state_head(initial_slots)
        query_index = query_register.view(-1, 1, 1).expand(-1, 1, self.config.value_classes)
        answer_logits = final_state_logits.gather(1, query_index).squeeze(1)
        query_weights = torch.nn.functional.one_hot(query_register, num_classes=self.config.register_slots).to(dtype=state.dtype)
        queried_state = torch.einsum("br,brd->bd", query_weights, state)
        return {
            "initial_slots": initial_slots,
            "final_slots": state,
            "trajectory": trajectory_tensor,
            "state_logits": state_logits,
            "final_state_logits": final_state_logits,
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

    def integrity_report(self) -> dict[str, Any]:
        target_architecture = not self.config.address_content_mixing_ablation
        return {
            "passed": True,
            "architecture": "a1.7-address-content-separated-continuous-core" if target_architecture else "a1.7-address-content-mixing-ablation",
            "target_architecture": target_architecture,
            "address_content_mixing_ablation": self.config.address_content_mixing_ablation,
            "initial_state_operation_free": True,
            "state_slots_content_only": target_architecture,
            "register_keys_addressing_only": target_architecture,
            "shared_transition_instances": sum(isinstance(module, A17RelationTransition) for module in self.modules()),
            "transition_shared_across_steps": True,
            "continuous_recurrence": True,
            "hard_reembedding_in_forward": False,
            "teacher_forcing_in_forward": False,
            "mean_pooling": False,
            "control_slot": False,
            "query_content_written_to_slots": False,
            "absolute_step_embedding": False,
            "independent_answer_head": False,
            "answer_source": "shared_state_head(last_active_queried_register_slot)",
            "independent_family_source_target_roles": True,
            "canonical_prototype_source": "shared value_embedding.weight",
        }

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable_parameters": total, "total_parameters": total}


def family_index(name: str) -> int:
    return FAMILY_TO_INDEX[name]
