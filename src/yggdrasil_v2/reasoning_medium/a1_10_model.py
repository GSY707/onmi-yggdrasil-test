from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class A110ModelConfig:
    source_width: int
    latent_width: int = 256
    workspace_slots: int = 8
    attention_heads: int = 8
    ffn_width: int = 1024
    recurrent_layers: int = 2
    value_classes: int = 10
    state_fields: int = 3


class A110RecurrentBlock(nn.Module):
    """Task-agnostic pre-norm latent self/cross-attention plus dense FFN."""

    def __init__(self, config: A110ModelConfig) -> None:
        super().__init__()
        d = config.latent_width
        self.self_norm = nn.LayerNorm(d)
        self.self_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True, dropout=0.0)
        self.source_norm = nn.LayerNorm(d)
        self.source_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True, dropout=0.0)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, d),
        )

    def forward(self, slots: torch.Tensor, source: torch.Tensor, source_padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(slots)
        update, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        slots = slots + update
        normalized = self.source_norm(slots)
        update, _ = self.source_attention(
            normalized,
            source,
            source,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        slots = slots + update
        slots = slots + self.ffn(self.ffn_norm(slots))
        return slots


class A110AnonymousRecurrentReasoner(nn.Module):
    """Full-text reader with anonymous K-slot workspace and shared recurrent Transformer."""

    def __init__(self, config: A110ModelConfig) -> None:
        super().__init__()
        if config.workspace_slots < 1:
            raise ValueError("workspace_slots must be positive")
        if config.latent_width % config.attention_heads:
            raise ValueError("latent_width must be divisible by attention_heads")
        self.config = config
        d = config.latent_width
        self.source_input_norm = nn.LayerNorm(config.source_width, elementwise_affine=False)
        self.source_projection = nn.Linear(config.source_width, d)
        self.learned_queries = nn.Parameter(torch.randn(config.workspace_slots, d) * 0.02)
        self.read_query_norm = nn.LayerNorm(d)
        self.read_source_norm = nn.LayerNorm(d)
        self.read_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True, dropout=0.0)
        self.read_ffn_norm = nn.LayerNorm(d)
        self.read_ffn = nn.Sequential(nn.Linear(d, config.ffn_width), nn.GELU(), nn.Linear(config.ffn_width, d))
        self.recurrent_reasoner = nn.ModuleList(A110RecurrentBlock(config) for _ in range(config.recurrent_layers))
        self.output_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, config.state_fields * config.value_classes)
        self.answer_head = nn.Linear(d, config.value_classes)

    def _read(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
        slot_permutation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if source_hidden.ndim != 3 or source_attention_mask.shape != source_hidden.shape[:2]:
            raise ValueError("source hidden/mask must have [B, L, H] and [B, L] shapes")
        source_hidden = source_hidden.to(dtype=self.source_projection.weight.dtype)
        source = self.source_projection(self.source_input_norm(source_hidden))
        source = self.read_source_norm(source)
        queries = self.learned_queries
        if slot_permutation is not None:
            queries = queries[slot_permutation]
        slots = queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        update, _ = self.read_attention(
            self.read_query_norm(slots),
            source,
            source,
            key_padding_mask=~source_attention_mask.bool(),
            need_weights=False,
        )
        slots = slots + update
        slots = slots + self.read_ffn(self.read_ffn_norm(slots))
        return slots, source, ~source_attention_mask.bool()

    def _pooled(self, slots: torch.Tensor) -> torch.Tensor:
        return self.output_norm(slots).mean(dim=1)

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
        recurrent_steps: int,
        *,
        slot_permutation: torch.Tensor | None = None,
        disable_recurrence: bool = False,
    ) -> dict[str, Any]:
        if recurrent_steps < 1:
            raise ValueError("A1.10 recurrent_steps must be positive")
        slots, source, source_padding_mask = self._read(source_hidden, source_attention_mask, slot_permutation)
        initial_slots = slots
        trajectory: list[torch.Tensor] = []
        for _ in range(recurrent_steps):
            if not disable_recurrence:
                for block in self.recurrent_reasoner:
                    slots = block(slots, source, source_padding_mask)
            trajectory.append(slots)
        trajectory_tensor = torch.stack(trajectory, dim=1)
        pooled_trajectory = self.output_norm(trajectory_tensor).mean(dim=2)
        state_logits = self.state_head(pooled_trajectory).view(
            source.shape[0], recurrent_steps, self.config.state_fields, self.config.value_classes
        )
        final_pooled = self._pooled(slots)
        return {
            "initial_slots": initial_slots,
            "final_slots": slots,
            "trajectory": trajectory_tensor,
            "pooled_trajectory": pooled_trajectory,
            "state_logits": state_logits,
            "answer_logits": self.answer_head(final_pooled),
        }

    def integrity_report(self) -> dict[str, Any]:
        names = [name.lower() for name, _ in self.named_parameters()]
        forbidden_fragments = ("amber", "cobalt", "jade", "register", "family", "source_role", "target_role", "operation_embedding")
        forbidden = [name for name in names if any(fragment in name for fragment in forbidden_fragments)]
        block_count = sum(isinstance(module, A110RecurrentBlock) for module in self.modules())
        return {
            "passed": not forbidden and block_count == self.config.recurrent_layers,
            "architecture": "full-token-anonymous-k-slot-shared-recurrent-transformer",
            "full_source_cross_attention": True,
            "oracle_span_mask_input": False,
            "operation_mask_input": False,
            "program_length_input": False,
            "typed_role_tensor_input": False,
            "explicit_register_slots": False,
            "register_address_keys": False,
            "task_specific_transition": False,
            "workspace_slots": self.config.workspace_slots,
            "workspace_slots_anonymous": True,
            "slot_readout_permutation_invariant": True,
            "recurrent_block_instances": block_count,
            "recurrent_weights_shared_across_steps": True,
            "absolute_step_embedding": False,
            "hard_reembedding": False,
            "teacher_state_input": False,
            "answer_reads_source_directly": False,
            "answer_source": "final_anonymous_workspace_mean_only",
            "state_labels_are_readout_only": True,
            "forbidden_parameter_names": forbidden,
            "config": asdict(self.config),
        }

    def parameter_report(self, *, qwen_parameters: int = 0) -> dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "qwen_frozen_parameters": int(qwen_parameters),
            "reasoner_trainable_parameters": trainable,
            "reasoner_total_parameters": total,
            "total_parameters": int(qwen_parameters) + total,
            "trainable_parameters": trainable,
            "frozen_parameters": int(qwen_parameters) + total - trainable,
        }
