from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class P0Config:
    latent_width: int = 256
    attention_heads: int = 8
    ffn_width: int = 512
    value_classes: int = 10
    register_count: int = 3
    operation_families: int = 3  # noop, swap, copy
    register_targets: int = 4  # three registers plus noop sentinel


class SharedStructuredTransition(nn.Module):
    """One shared transition: slot self-attention, op cross-attention, FFN.

    The residual scale is a direct, non-saturating parameter initialized to
    0.1. There is deliberately no absolute step embedding and no sigmoid gate.
    """

    def __init__(self, config: P0Config) -> None:
        super().__init__()
        d = config.latent_width
        self.slot_norm = nn.LayerNorm(d)
        self.slot_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True)
        self.operation_norm = nn.LayerNorm(d)
        self.operation_cross_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, d),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, slots: torch.Tensor, operation: torch.Tensor) -> torch.Tensor:
        normalized = self.slot_norm(slots)
        attended, _ = self.slot_attention(normalized, normalized, normalized, need_weights=False)
        x = slots + attended
        query = self.operation_norm(x)
        op_token = operation.unsqueeze(1)
        conditioned, _ = self.operation_cross_attention(query, op_token, op_token, need_weights=False)
        x = x + conditioned
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StructuredRecurrentCore(nn.Module):
    """P0 explicit-register recurrent latent state machine."""

    def __init__(self, config: P0Config | None = None) -> None:
        super().__init__()
        self.config = config or P0Config()
        d = self.config.latent_width
        self.value_embedding = nn.Embedding(self.config.value_classes, d)
        self.register_embedding = nn.Embedding(self.config.register_count + 1, d)
        self.query_embedding = nn.Embedding(self.config.register_count, d)
        self.control_embedding = nn.Parameter(torch.empty(d))
        self.operation_family_embedding = nn.Embedding(self.config.operation_families, d)
        self.operation_source_embedding = nn.Embedding(self.config.register_targets, d)
        self.operation_target_embedding = nn.Embedding(self.config.register_targets, d)
        self.transition = SharedStructuredTransition(self.config)
        self.final_norm = nn.LayerNorm(d)
        self.state_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, self.config.value_classes)
        # The answer head reads only the final query/control slot.
        self.answer_head = nn.Linear(d, self.config.value_classes)
        nn.init.normal_(self.control_embedding, std=0.02)

    def _operation_embedding(
        self,
        family: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.operation_family_embedding(family)
            + self.operation_source_embedding(source)
            + self.operation_target_embedding(target)
        )

    def _initial_slots(self, start_values: torch.Tensor, query_register: torch.Tensor) -> torch.Tensor:
        batch = start_values.shape[0]
        register_slots = self.value_embedding(start_values)
        register_ids = torch.arange(self.config.register_count, device=start_values.device)
        register_slots = register_slots + self.register_embedding(register_ids).view(1, -1, self.config.latent_width)
        query_slot = self.query_embedding(query_register) + self.control_embedding
        query_slot = query_slot.unsqueeze(1)
        return torch.cat([register_slots, query_slot], dim=1)

    def forward(
        self,
        start_values: torch.Tensor,
        query_register: torch.Tensor,
        operation_family: torch.Tensor,
        operation_source: torch.Tensor,
        operation_target: torch.Tensor,
        operation_mask: torch.Tensor,
        *,
        disable_transition_delta: bool = False,
        return_trajectory: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if operation_family.ndim != 2:
            raise ValueError("operation tensors must have shape [batch, steps]")
        slots = self._initial_slots(start_values, query_register)
        trajectory: list[torch.Tensor] = []
        max_steps = operation_family.shape[1]
        for step in range(max_steps):
            operation = self._operation_embedding(
                operation_family[:, step],
                operation_source[:, step],
                operation_target[:, step],
            )
            transformed = self.transition(slots, operation)
            if not disable_transition_delta:
                delta = transformed - slots
                active = operation_mask[:, step].to(dtype=slots.dtype).view(-1, 1, 1)
                slots = slots + active * self.transition.residual_scale * delta
            if return_trajectory:
                trajectory.append(slots)
        normalized = self.final_norm(slots)
        logits = self.answer_head(normalized[:, self.config.register_count])
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "logits": logits,
            "final_state": slots,
            "state_logits": self.state_head(self.state_norm(slots[:, : self.config.register_count])),
        }
        if return_trajectory:
            result["trajectory"] = trajectory
        return result

    def decode_trajectory(self, trajectory: list[torch.Tensor]) -> torch.Tensor:
        if not trajectory:
            return torch.empty(0, device=self.control_embedding.device)
        states = torch.stack([self.state_norm(state[:, : self.config.register_count]) for state in trajectory], dim=1)
        return self.state_head(states)

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable_parameters": total, "active_parameters": total, "total_parameters": total}

    def integrity_report(self) -> dict[str, Any]:
        names = [name for name, _ in self.named_parameters()]
        forbidden = [name for name in names if any(token in name for token in ("input_ids", "labels", "trace", "answer_bypass"))]
        return {
            "passed": not forbidden,
            "forbidden_parameter_names": forbidden,
            "answer_head_input": "final latent query/control slot only",
            "absolute_step_embedding": False,
            "transition_weights_shared": True,
            "transition_delta_gate": "none; direct residual_scale initialized at 0.1",
        }

