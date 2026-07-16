from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .a1_7_core import A17RelationAddressedCore
from .a1_9_model import _cosine_logits, module_state_sha256
from .a1_10_model import A110AnonymousRecurrentReasoner, A110ModelConfig


@dataclass(frozen=True)
class A111BoundaryConfig:
    source_width: int
    latent_width: int = 256
    attention_heads: int = 8
    ffn_width: int = 1024
    reader_layers: int = 2
    maximum_operation_steps: int = 32
    similarity_scale: float = 20.0


class A111TypedReaderBlock(nn.Module):
    """Generic typed-query self/cross-attention reader; no span or token selector input."""

    def __init__(self, config: A111BoundaryConfig) -> None:
        super().__init__()
        d = config.latent_width
        self.query_norm = nn.LayerNorm(d)
        self.query_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True, dropout=0.0)
        self.source_norm = nn.LayerNorm(d)
        self.source_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True, dropout=0.0)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, config.ffn_width), nn.GELU(), nn.Linear(config.ffn_width, d))

    def forward(
        self,
        queries: torch.Tensor,
        source: torch.Tensor,
        *,
        query_valid: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.query_norm(queries)
        update, _ = self.query_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~query_valid,
            need_weights=False,
        )
        queries = (queries + update) * query_valid.unsqueeze(-1)
        update, _ = self.source_attention(
            self.source_norm(queries),
            source,
            source,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        queries = (queries + update) * query_valid.unsqueeze(-1)
        queries = (queries + self.ffn(self.ffn_norm(queries))) * query_valid.unsqueeze(-1)
        return queries


def _sinusoidal_positions(length: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(length, device=reference.device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=reference.device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros((length, width), device=reference.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequency)
    encoding[:, 1::2] = torch.cos(positions * frequency[: encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=reference.dtype)


class A111LearnedFullTextBoundary(nn.Module):
    """Full-token learned typed reader around one frozen, already-passed A1.8 core."""

    def __init__(self, config: A111BoundaryConfig, core: A17RelationAddressedCore) -> None:
        super().__init__()
        if config.latent_width != core.config.latent_width:
            raise ValueError("A1.11 Boundary width must match the frozen A1.8 core")
        if config.latent_width % config.attention_heads:
            raise ValueError("A1.11 Boundary latent width must divide attention heads")
        self.config = config
        self.core = core
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.core.eval()
        d = config.latent_width
        self.source_input_norm = nn.LayerNorm(config.source_width, elementwise_affine=False)
        self.source_projection = nn.Linear(config.source_width, d)
        self.source_output_norm = nn.LayerNorm(d)
        self.start_register_queries = nn.Parameter(torch.randn(core.config.register_slots, d) * 0.02)
        self.query_register_query = nn.Parameter(torch.randn(1, d) * 0.02)
        self.operation_role_queries = nn.Parameter(torch.randn(3, d) * 0.02)
        self.reader = nn.ModuleList(A111TypedReaderBlock(config) for _ in range(config.reader_layers))

    def train(self, mode: bool = True) -> A111LearnedFullTextBoundary:
        super().train(mode)
        self.core.eval()
        return self

    def _queries(self, operation_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = operation_mask.shape
        if steps > self.config.maximum_operation_steps:
            raise ValueError("A1.11 Boundary operation count exceeds the frozen maximum")
        static = torch.cat((self.start_register_queries, self.query_register_query), dim=0)
        static = static.unsqueeze(0).expand(batch, -1, -1)
        step_positions = _sinusoidal_positions(steps, self.config.latent_width, self.operation_role_queries)
        operations = self.operation_role_queries.view(1, 1, 3, -1) + step_positions.view(1, steps, 1, -1)
        operations = operations.expand(batch, -1, -1, -1).reshape(batch, steps * 3, -1)
        queries = torch.cat((static, operations), dim=1)
        valid = torch.cat(
            (
                torch.ones((batch, static.shape[1]), dtype=torch.bool, device=operation_mask.device),
                operation_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(batch, steps * 3),
            ),
            dim=1,
        )
        return queries, valid

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
        operation_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if source_hidden.ndim != 3 or source_attention_mask.shape != source_hidden.shape[:2]:
            raise ValueError("A1.11 Boundary source hidden/mask shapes are invalid")
        if operation_mask.ndim != 2 or operation_mask.shape[0] != source_hidden.shape[0]:
            raise ValueError("A1.11 Boundary operation mask must have [B, T] shape")
        source_hidden = source_hidden.to(dtype=self.source_projection.weight.dtype)
        source = self.source_output_norm(self.source_projection(self.source_input_norm(source_hidden)))
        queries, query_valid = self._queries(operation_mask.bool())
        for block in self.reader:
            queries = block(
                queries,
                source,
                query_valid=query_valid,
                source_padding_mask=~source_attention_mask.bool(),
            )
        initial_slots = queries[:, : self.core.config.register_slots]
        query_latent = queries[:, self.core.config.register_slots]
        operations = queries[:, self.core.config.register_slots + 1 :].reshape(
            source.shape[0], operation_mask.shape[1], 3, self.config.latent_width
        )
        family_latents = operations[:, :, 0]
        source_latents = operations[:, :, 1]
        target_latents = operations[:, :, 2]
        query_logits = _cosine_logits(query_latent, self.core.register_keys, self.config.similarity_scale)
        query_register = query_logits.argmax(dim=-1)
        output = self.core.rollout(
            initial_slots,
            query_register,
            family_latents,
            source_latents,
            target_latents,
            operation_mask,
        )
        output.update(
            {
                "mapped_start_values": initial_slots,
                "mapped_family": family_latents,
                "mapped_source": source_latents,
                "mapped_target": target_latents,
                "mapped_query": query_latent,
                "value_mapping_logits": _cosine_logits(
                    initial_slots, self.core.canonical_value_prototypes, self.config.similarity_scale
                ),
                "family_mapping_logits": _cosine_logits(
                    family_latents, self.core.family_embedding.weight, self.config.similarity_scale
                ),
                "source_mapping_logits": _cosine_logits(
                    source_latents, self.core.register_keys, self.config.similarity_scale
                ),
                "target_mapping_logits": _cosine_logits(
                    target_latents, self.core.register_keys, self.config.similarity_scale
                ),
                "query_mapping_logits": query_logits,
                "query_register": query_register,
            }
        )
        return output

    def boundary_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self.state_dict().items() if not name.startswith("core.")}

    def load_boundary_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        invalid_missing = [name for name in missing if not name.startswith("core.")]
        if invalid_missing or unexpected:
            raise ValueError(f"invalid A1.11 Boundary state: missing={invalid_missing}, unexpected={unexpected}")

    def integrity_report(self) -> dict[str, Any]:
        core_frozen = all(not parameter.requires_grad for parameter in self.core.parameters())
        return {
            "passed": self.core.integrity_report()["passed"] and core_frozen,
            "arm": "A1.11-Boundary",
            "full_source_hidden": True,
            "oracle_span_input": False,
            "offset_mapping_input": False,
            "typed_role_queries": True,
            "operation_mask_retained_for_isolation": True,
            "fixed_sinusoidal_operation_positions": True,
            "untrained_step_specific_query_parameters": False,
            "structured_core_retained": True,
            "core_frozen": core_frozen,
            "core_state_sha256": module_state_sha256(self.core),
            "reader_weights_shared_across_operation_positions": True,
            "unconstrained_latent_output_for_frozen_core_geometry": True,
            "source_direct_answer_head": False,
            "config": asdict(self.config),
        }


@dataclass(frozen=True)
class A111ReasonerConfig:
    latent_width: int = 256
    workspace_slots: int = 8
    attention_heads: int = 8
    ffn_width: int = 1024
    recurrent_layers: int = 2
    value_classes: int = 10
    state_fields: int = 3
    maximum_operation_steps: int = 32
    semantic_features: int = 15
    type_features: int = 7
    step_feature_width: int = 32


class A111OracleRoleReasoner(nn.Module):
    """Exact symbolic typed roles feeding the anonymous generic reasoner family."""

    def __init__(self, config: A111ReasonerConfig) -> None:
        super().__init__()
        if config.semantic_features != 15:
            raise ValueError("A1.11 Reasoner semantic features are frozen at 10 values + 2 families + 3 registers")
        self.config = config
        source_width = config.semantic_features + config.type_features + config.step_feature_width
        self.reasoner = A110AnonymousRecurrentReasoner(
            A110ModelConfig(
                source_width=source_width,
                latent_width=config.latent_width,
                workspace_slots=config.workspace_slots,
                attention_heads=config.attention_heads,
                ffn_width=config.ffn_width,
                recurrent_layers=config.recurrent_layers,
                value_classes=config.value_classes,
                state_fields=config.state_fields,
            )
        )

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
            raise ValueError("A1.11 Reasoner operation count exceeds the frozen maximum")
        semantic = torch.zeros(
            (batch, 4 + steps * 3, self.config.semantic_features),
            dtype=torch.float32,
            device=operation_mask.device,
        )
        semantic[:, :3, :10] = nn.functional.one_hot(start_values, num_classes=10).to(dtype=semantic.dtype)
        semantic[:, 3, 12:15] = nn.functional.one_hot(query_register, num_classes=3).to(dtype=semantic.dtype)
        operations = semantic[:, 4:].view(batch, steps, 3, self.config.semantic_features)
        operations[:, :, 0, 10:12] = nn.functional.one_hot(operation_family, num_classes=2).to(
            dtype=semantic.dtype
        )
        operations[:, :, 1, 12:15] = nn.functional.one_hot(operation_source, num_classes=3).to(
            dtype=semantic.dtype
        )
        operations[:, :, 2, 12:15] = nn.functional.one_hot(operation_target, num_classes=3).to(
            dtype=semantic.dtype
        )
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
        types = nn.functional.one_hot(type_ids, num_classes=self.config.type_features).to(dtype=semantic.dtype)
        step_features = torch.zeros(
            (batch, 4 + steps * 3, self.config.step_feature_width),
            dtype=semantic.dtype,
            device=semantic.device,
        )
        sinusoidal = _sinusoidal_positions(steps, self.config.step_feature_width, semantic)
        step_features[:, 4:] = (
            sinusoidal.view(1, steps, 1, -1).expand(batch, -1, 3, -1).reshape(batch, steps * 3, -1)
        )
        valid = torch.cat(
            (
                torch.ones((batch, 4), dtype=torch.bool, device=semantic.device),
                operation_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(batch, steps * 3),
            ),
            dim=1,
        )
        return torch.cat((semantic, types, step_features), dim=-1), valid

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
        slot_permutation: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        typed_source, typed_mask = self._typed_source(
            start_values,
            query_register,
            operation_family,
            operation_source,
            operation_target,
            operation_mask,
        )
        return self.reasoner(
            typed_source,
            typed_mask,
            recurrent_steps,
            disable_recurrence=disable_recurrence,
            slot_permutation=slot_permutation,
        )

    def reasoner_state_dict(self) -> dict[str, torch.Tensor]:
        return self.reasoner.state_dict()

    def integrity_report(self) -> dict[str, Any]:
        reasoner_report = self.reasoner.integrity_report()
        return {
            "passed": reasoner_report["passed"],
            "arm": "A1.11-Reasoner",
            "exact_symbolic_oracle_roles": True,
            "typed_role_features_retained": True,
            "fixed_sinusoidal_operation_positions": True,
            "untrained_ood_step_columns": False,
            "qwen_or_boundary_adapter_loaded": False,
            "structured_a1_8_core_loaded": False,
            "explicit_register_workspace": False,
            "task_specific_transition": False,
            "anonymous_reasoner": reasoner_report,
            "config": asdict(self.config),
        }

    def parameter_report(self) -> dict[str, int]:
        reasoner = sum(parameter.numel() for parameter in self.reasoner.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "qwen_parameters": 0,
            "boundary_adapter_parameters": 0,
            "reasoner_parameters": reasoner,
            "trainable_parameters": trainable,
            "frozen_parameters": 0,
            "total_parameters": reasoner,
        }
