from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .a1_10_model import A110ModelConfig, A110RecurrentBlock
from .a1_11_models import _sinusoidal_positions


@dataclass(frozen=True)
class A112ReasonerConfig:
    entity_addressable: bool
    aligned_operation_cursor: bool
    latent_width: int = 256
    workspace_slots: int = 8
    entity_slots: int = 3
    attention_heads: int = 8
    ffn_width: int = 1024
    recurrent_layers: int = 2
    value_classes: int = 10
    state_fields: int = 3
    maximum_operation_steps: int = 32
    semantic_features: int = 15
    type_features: int = 7
    step_feature_width: int = 32


class A112RootCauseReasoner(nn.Module):
    """Exact-symbolic generic reasoner with only binding and cursor as experimental axes."""

    def __init__(self, config: A112ReasonerConfig) -> None:
        super().__init__()
        if config.workspace_slots != 8 or config.entity_slots != 3:
            raise ValueError("A1.12 freezes the workspace at 3 entity candidates + 5 scratch slots")
        if config.semantic_features != 15 or config.type_features != 7:
            raise ValueError("A1.12 exact-symbolic feature widths are frozen")
        if config.latent_width % config.attention_heads:
            raise ValueError("A1.12 latent width must divide attention heads")
        self.config = config
        source_width = config.semantic_features + config.type_features + config.step_feature_width
        generic = A110ModelConfig(
            source_width=source_width,
            latent_width=config.latent_width,
            workspace_slots=config.workspace_slots,
            attention_heads=config.attention_heads,
            ffn_width=config.ffn_width,
            recurrent_layers=config.recurrent_layers,
            value_classes=config.value_classes,
            state_fields=config.state_fields,
        )
        d = config.latent_width
        self.source_input_norm = nn.LayerNorm(source_width, elementwise_affine=False)
        self.source_projection = nn.Linear(source_width, d)
        self.learned_queries = nn.Parameter(torch.randn(config.workspace_slots, d) * 0.02)
        self.read_query_norm = nn.LayerNorm(d)
        self.read_source_norm = nn.LayerNorm(d)
        self.read_attention = nn.MultiheadAttention(
            d, config.attention_heads, batch_first=True, dropout=0.0
        )
        self.read_ffn_norm = nn.LayerNorm(d)
        self.read_ffn = nn.Sequential(
            nn.Linear(d, config.ffn_width), nn.GELU(), nn.Linear(config.ffn_width, d)
        )
        self.recurrent_reasoner = nn.ModuleList(
            A110RecurrentBlock(generic) for _ in range(config.recurrent_layers)
        )
        self.output_norm = nn.LayerNorm(d)
        if config.entity_addressable:
            self.entity_initialization = nn.Linear(
                config.value_classes + config.entity_slots, d, bias=False
            )
            self.state_head = nn.Linear(d, config.value_classes)
        else:
            self.entity_initialization = None
            self.state_head = nn.Linear(d, config.state_fields * config.value_classes)
        self.answer_head = nn.Linear(d, config.value_classes)

    @property
    def arm(self) -> str:
        if self.config.entity_addressable and self.config.aligned_operation_cursor:
            return "BOTH"
        if self.config.entity_addressable:
            return "BIND"
        if self.config.aligned_operation_cursor:
            return "CURSOR"
        return "REFERENCE"

    def _typed_source(
        self,
        start_values: torch.Tensor,
        query_register: torch.Tensor,
        operation_family: torch.Tensor,
        operation_source: torch.Tensor,
        operation_target: torch.Tensor,
        operation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = operation_mask.shape
        if steps > self.config.maximum_operation_steps:
            raise ValueError("A1.12 operation count exceeds the frozen maximum")
        semantic = torch.zeros(
            (batch, 4 + steps * 3, self.config.semantic_features),
            dtype=torch.float32,
            device=operation_mask.device,
        )
        semantic[:, :3, :10] = nn.functional.one_hot(start_values, num_classes=10).to(
            dtype=semantic.dtype
        )
        semantic[:, 3, 12:15] = nn.functional.one_hot(query_register, num_classes=3).to(
            dtype=semantic.dtype
        )
        operations = semantic[:, 4:].view(batch, steps, 3, self.config.semantic_features)
        operations[:, :, 0, 10:12] = nn.functional.one_hot(
            operation_family, num_classes=2
        ).to(dtype=semantic.dtype)
        operations[:, :, 1, 12:15] = nn.functional.one_hot(
            operation_source, num_classes=3
        ).to(dtype=semantic.dtype)
        operations[:, :, 2, 12:15] = nn.functional.one_hot(
            operation_target, num_classes=3
        ).to(dtype=semantic.dtype)
        type_ids = torch.cat(
            (
                torch.arange(4, device=semantic.device).unsqueeze(0).expand(batch, -1),
                torch.tensor([4, 5, 6], device=semantic.device)
                .view(1, 1, 3)
                .expand(batch, steps, -1)
                .reshape(batch, steps * 3),
            ),
            dim=1,
        )
        types = nn.functional.one_hot(type_ids, num_classes=self.config.type_features).to(
            dtype=semantic.dtype
        )
        positions = torch.zeros(
            (batch, 4 + steps * 3, self.config.step_feature_width),
            dtype=semantic.dtype,
            device=semantic.device,
        )
        sinusoidal = _sinusoidal_positions(
            steps, self.config.step_feature_width, semantic
        )
        positions[:, 4:] = (
            sinusoidal.view(1, steps, 1, -1)
            .expand(batch, -1, 3, -1)
            .reshape(batch, steps * 3, -1)
        )
        valid = torch.cat(
            (
                torch.ones((batch, 4), dtype=torch.bool, device=semantic.device),
                operation_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(batch, steps * 3),
            ),
            dim=1,
        )
        return torch.cat((semantic, types, positions), dim=-1), valid

    def _read(
        self,
        source: torch.Tensor,
        valid: torch.Tensor,
        start_values: torch.Tensor,
    ) -> torch.Tensor:
        batch = source.shape[0]
        initial_source = source[:, :4] if self.config.aligned_operation_cursor else source
        initial_valid = valid[:, :4] if self.config.aligned_operation_cursor else valid
        slots = self.learned_queries.unsqueeze(0).expand(batch, -1, -1)
        update, _ = self.read_attention(
            self.read_query_norm(slots),
            initial_source,
            initial_source,
            key_padding_mask=~initial_valid,
            need_weights=False,
        )
        slots = slots + update
        slots = slots + self.read_ffn(self.read_ffn_norm(slots))
        if self.entity_initialization is not None:
            register_identity = torch.eye(
                self.config.entity_slots,
                dtype=source.dtype,
                device=source.device,
            ).unsqueeze(0).expand(batch, -1, -1)
            value_identity = nn.functional.one_hot(
                start_values, num_classes=self.config.value_classes
            ).to(dtype=source.dtype)
            entity_input = torch.cat((value_identity, register_identity), dim=-1)
            entity_slots = slots[:, : self.config.entity_slots] + self.entity_initialization(
                entity_input
            )
            slots = torch.cat((entity_slots, slots[:, self.config.entity_slots :]), dim=1)
        return slots

    def _step_source(
        self,
        source: torch.Tensor,
        valid: torch.Tensor,
        step: int,
        operation_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.aligned_operation_cursor:
            return source, ~valid
        query = source[:, 3:4]
        query_valid = valid[:, 3:4]
        if step >= operation_steps:
            return query, ~query_valid
        begin = 4 + step * 3
        current = source[:, begin : begin + 3]
        current_valid = valid[:, begin : begin + 3]
        selected = torch.cat((query, current), dim=1)
        selected_valid = torch.cat((query_valid, current_valid), dim=1)
        return selected, ~selected_valid

    def forward(
        self,
        start_values: torch.Tensor,
        query_register: torch.Tensor,
        operation_family: torch.Tensor,
        operation_source: torch.Tensor,
        operation_target: torch.Tensor,
        operation_mask: torch.Tensor,
        recurrent_steps: int,
        *,
        disable_recurrence: bool = False,
        start_value_permutation: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if recurrent_steps < 1:
            raise ValueError("A1.12 recurrent_steps must be positive")
        input_start_values = start_values
        if start_value_permutation is not None:
            input_start_values = start_values[:, start_value_permutation]
        typed, valid = self._typed_source(
            input_start_values,
            query_register,
            operation_family,
            operation_source,
            operation_target,
            operation_mask,
        )
        typed = typed.to(dtype=self.source_projection.weight.dtype)
        source = self.read_source_norm(
            self.source_projection(self.source_input_norm(typed))
        )
        slots = self._read(source, valid, input_start_values)
        initial_slots = slots
        trajectory: list[torch.Tensor] = []
        operation_steps = operation_mask.shape[1]
        for step in range(recurrent_steps):
            if not disable_recurrence:
                step_source, padding_mask = self._step_source(
                    source, valid, step, operation_steps
                )
                for block in self.recurrent_reasoner:
                    slots = block(slots, step_source, padding_mask)
            trajectory.append(slots)
        trajectory_tensor = torch.stack(trajectory, dim=1)
        normalized = self.output_norm(trajectory_tensor)
        if self.config.entity_addressable:
            state_logits = self.state_head(
                normalized[:, :, : self.config.entity_slots]
            )
        else:
            pooled = normalized.mean(dim=2)
            state_logits = self.state_head(pooled).view(
                source.shape[0], recurrent_steps, self.config.state_fields, self.config.value_classes
            )
        final_pooled = self.output_norm(slots).mean(dim=1)
        return {
            "initial_slots": initial_slots,
            "final_slots": slots,
            "trajectory": trajectory_tensor,
            "state_logits": state_logits,
            "answer_logits": self.answer_head(final_pooled),
        }

    def integrity_report(self) -> dict[str, Any]:
        block_count = sum(isinstance(module, A110RecurrentBlock) for module in self.modules())
        forbidden = [
            name
            for name, _ in self.named_parameters()
            if any(fragment in name.lower() for fragment in ("copy", "swap", "family_transition"))
        ]
        return {
            "passed": not forbidden and block_count == self.config.recurrent_layers,
            "arm": self.arm,
            "exact_symbolic_oracle_roles": True,
            "qwen_or_boundary_adapter_loaded": False,
            "structured_core_loaded": False,
            "entity_addressable_state": self.config.entity_addressable,
            "entity_slots": list(range(self.config.entity_slots)) if self.config.entity_addressable else [],
            "scratch_slots": self.config.workspace_slots - (
                self.config.entity_slots if self.config.entity_addressable else 0
            ),
            "shared_entity_state_head": self.config.entity_addressable,
            "direct_exact_start_state_initialization": self.config.entity_addressable,
            "aligned_operation_cursor": self.config.aligned_operation_cursor,
            "cursor_executes_operations": False,
            "full_program_visible_each_recurrent_step": not self.config.aligned_operation_cursor,
            "task_specific_transition": False,
            "recurrent_weights_shared_across_steps": True,
            "recurrent_block_instances": block_count,
            "answer_source": "final_workspace_mean_only",
            "forbidden_parameter_names": forbidden,
            "config": asdict(self.config),
        }

    def parameter_report(self) -> dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "qwen_parameters": 0,
            "structured_core_parameters": 0,
            "trainable_parameters": trainable,
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
        }
