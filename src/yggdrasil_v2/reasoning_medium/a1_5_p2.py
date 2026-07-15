from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .a1_5_model import P0Config, SharedStructuredTransition


@dataclass(frozen=True)
class P2Config:
    learned_slots: int = 8
    latent_width: int = 256
    attention_heads: int = 8
    ffn_width: int = 512
    value_classes: int = 10


class LearnedMultiSlotWorkspace(nn.Module):
    """P2 learned K-slot workspace; it has no amber/cobalt/jade slot binding."""

    def __init__(
        self,
        operation_width: int,
        config: P2Config | None = None,
        *,
        source_width: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or P2Config()
        d = self.config.latent_width
        transition_config = P0Config(
            latent_width=d,
            attention_heads=self.config.attention_heads,
            ffn_width=self.config.ffn_width,
            value_classes=self.config.value_classes,
        )
        self.learned_slots = nn.Parameter(torch.randn(self.config.learned_slots, d) * 0.02)
        self.operation_projection = nn.Linear(operation_width, d)
        self.source_projection = nn.Linear(source_width, d) if source_width is not None else None
        self.source_norm = nn.LayerNorm(d) if source_width is not None else None
        self.slot_type = nn.Embedding(self.config.learned_slots, d) if source_width is not None else None
        self.source_cross_attention = (
            nn.MultiheadAttention(d, self.config.attention_heads, batch_first=True)
            if source_width is not None
            else None
        )
        self.transition = SharedStructuredTransition(transition_config)
        self.final_norm = nn.LayerNorm(d)
        self.answer_head = nn.Linear(d, self.config.value_classes)
        self.state_head = nn.Linear(d, 3 * self.config.value_classes)

    @staticmethod
    def pool_operation_hidden(
        source_hidden: torch.Tensor,
        operation_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = operation_token_mask.to(dtype=source_hidden.dtype)
        pooled = torch.einsum("btl,blh->bth", weights, source_hidden)
        return pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        operation_hidden: torch.Tensor,
        operation_mask: torch.Tensor,
        *,
        source_hidden: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
        slot_permutation: torch.Tensor | None = None,
        disable_transition_delta: bool = False,
        shuffle_transition_delta: bool = False,
        delta_permutation: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:
        batch = operation_hidden.shape[0]
        slots = self.learned_slots.unsqueeze(0).expand(batch, -1, -1)
        if slot_permutation is not None:
            slots = slots[:, slot_permutation]
        if source_hidden is not None:
            if self.source_projection is None or self.source_cross_attention is None or self.slot_type is None or self.source_norm is None:
                raise ValueError("source_hidden was provided but P2 has no source interface")
            source = self.source_norm(self.source_projection(source_hidden.to(self.source_projection.weight.dtype)))
            slot_ids = torch.arange(self.config.learned_slots, device=slots.device)
            if slot_permutation is not None:
                slot_ids = slot_ids[slot_permutation]
            queries = slots + self.slot_type(slot_ids).view(1, -1, self.config.latent_width)
            encoded, _ = self.source_cross_attention(
                queries,
                source,
                source,
                key_padding_mask=(~source_attention_mask.bool()) if source_attention_mask is not None else None,
                need_weights=False,
            )
            slots = slots + encoded
        initial_slots = slots
        op = self.operation_projection(operation_hidden.to(self.operation_projection.weight.dtype))
        deltas: list[torch.Tensor] = []
        normal_slots = slots
        for step in range(op.shape[1]):
            transformed = self.transition(normal_slots, op[:, step])
            deltas.append(transformed - normal_slots)
            active = operation_mask[:, step].to(dtype=normal_slots.dtype).view(-1, 1, 1)
            if not disable_transition_delta:
                normal_slots = normal_slots + active * self.transition.residual_scale * deltas[-1]
        if shuffle_transition_delta and deltas:
            permutation = delta_permutation
            if permutation is None:
                permutation = torch.arange(len(deltas) - 1, -1, -1, device=op.device)
            slots = initial_slots
            trajectory = []
            for step in range(op.shape[1]):
                active = operation_mask[:, step].to(dtype=slots.dtype).view(-1, 1, 1)
                delta = deltas[int(permutation[step])]
                if not disable_transition_delta:
                    slots = slots + active * self.transition.residual_scale * delta
                if return_trajectory:
                    trajectory.append(slots)
        else:
            slots = normal_slots
            trajectory = []
            if return_trajectory:
                replay_slots = initial_slots
                for step in range(op.shape[1]):
                    transformed = self.transition(replay_slots, op[:, step])
                    active = operation_mask[:, step].to(dtype=replay_slots.dtype).view(-1, 1, 1)
                    if not disable_transition_delta:
                        replay_slots = replay_slots + active * self.transition.residual_scale * (transformed - replay_slots)
                    trajectory.append(replay_slots)
        normalized = self.final_norm(slots)
        pooled = normalized.mean(dim=1)
        result: dict[str, Any] = {
            "logits": self.answer_head(pooled),
            "final_state": slots,
        }
        if return_trajectory:
            result["trajectory"] = trajectory
            result["state_logits"] = self.state_head(
                torch.stack([self.final_norm(item).mean(dim=1) for item in trajectory], dim=1)
            ) if trajectory else torch.empty(0, device=slots.device)
        return result

    def permutation_probe(
        self,
        operation_hidden: torch.Tensor,
        operation_mask: torch.Tensor,
        *,
        source_hidden: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        identity = torch.arange(self.config.learned_slots, device=operation_hidden.device)
        permutation = identity.flip(0)
        normal = self(
            operation_hidden,
            operation_mask,
            source_hidden=source_hidden,
            source_attention_mask=source_attention_mask,
        )["logits"]
        permuted = self(
            operation_hidden,
            operation_mask,
            source_hidden=source_hidden,
            source_attention_mask=source_attention_mask,
            slot_permutation=permutation,
        )["logits"]
        return {
            "logit_mean_abs_delta": float((normal - permuted).abs().mean().detach()),
            "permutation": permutation.tolist(),
        }

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable_parameters": total, "active_parameters": total, "total_parameters": total}

    def integrity_report(self) -> dict[str, Any]:
        names = [name for name, _ in self.named_parameters()]
        forbidden_slot_bindings = [name for name in names if any(register in name for register in ("amber", "cobalt", "jade"))]
        return {
            "passed": not forbidden_slot_bindings,
            "learned_slot_count": self.config.learned_slots,
            "explicit_register_slot_binding": False,
            "shared_transition": True,
            "source_mean_broadcast_main_path": False,
            "answer_bypass": False,
        }
