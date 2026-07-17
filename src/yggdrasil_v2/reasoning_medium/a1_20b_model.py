from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .a1_9_model import module_state_sha256
from .a1_19h_model import A119HHybridReasoner


@dataclass(frozen=True)
class A120BBoundaryConfig:
    source_width: int
    latent_width: int = 256
    attention_heads: int = 8
    ffn_width: int = 1024
    reader_layers: int = 2
    maximum_entities: int = 5
    maximum_operations: int = 32
    pointer_scale: float = 20.0


def _sinusoidal_positions(
    length: int, width: int, reference: torch.Tensor
) -> torch.Tensor:
    positions = torch.arange(
        length, device=reference.device, dtype=torch.float32
    ).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=reference.device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros(
        (length, width), device=reference.device, dtype=torch.float32
    )
    encoding[:, 0::2] = torch.sin(positions * frequency)
    encoding[:, 1::2] = torch.cos(
        positions * frequency[: encoding[:, 1::2].shape[1]]
    )
    return encoding.to(dtype=reference.dtype)


class A120BReaderBlock(nn.Module):
    def __init__(self, config: A120BBoundaryConfig) -> None:
        super().__init__()
        d = config.latent_width
        self.query_norm = nn.LayerNorm(d)
        self.query_attention = nn.MultiheadAttention(
            d, config.attention_heads, batch_first=True, dropout=0.0
        )
        self.source_norm = nn.LayerNorm(d)
        self.source_attention = nn.MultiheadAttention(
            d, config.attention_heads, batch_first=True, dropout=0.0
        )
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, d),
        )

    def forward(
        self,
        queries: torch.Tensor,
        source: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.query_norm(queries)
        update, _ = self.query_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + update
        update, _ = self.source_attention(
            self.source_norm(queries),
            source,
            source,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        queries = queries + update
        return queries + self.ffn(self.ffn_norm(queries))


def _cosine_pointer_logits(
    queries: torch.Tensor,
    entities: torch.Tensor,
    query_projection: nn.Linear,
    entity_projection: nn.Linear,
    scale: float,
) -> torch.Tensor:
    query = nn.functional.normalize(query_projection(queries), dim=-1)
    keys = nn.functional.normalize(entity_projection(entities), dim=-1)
    return torch.einsum("b...d,bed->b...e", query, keys) * scale


class A120BLearnedFullTextBoundary(nn.Module):
    """Full-text Boundary that predicts every mask, role and address it consumes."""

    def __init__(
        self, config: A120BBoundaryConfig, core: A119HHybridReasoner
    ) -> None:
        super().__init__()
        if config.maximum_entities != core.config.maximum_entities:
            raise ValueError("A1.20B entity capacity must match the frozen core")
        if config.latent_width != core.config.latent_width:
            raise ValueError("A1.20B latent width must match the frozen core")
        if config.latent_width % config.attention_heads:
            raise ValueError("A1.20B latent width must divide attention heads")
        if core.config.training_auxiliary != "none":
            raise ValueError("A1.20B requires an auxiliary-stripped A1.19H core")
        self.config = config
        self.core = core
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.core.eval()
        d = config.latent_width
        self.source_input_norm = nn.LayerNorm(
            config.source_width, elementwise_affine=False
        )
        self.source_projection = nn.Linear(config.source_width, d)
        self.source_output_norm = nn.LayerNorm(d)
        self.entity_role_query = nn.Parameter(torch.randn(d) * 0.02)
        self.query_role_query = nn.Parameter(torch.randn(d) * 0.02)
        self.operation_role_queries = nn.Parameter(torch.randn(3, d) * 0.02)
        self.reader = nn.ModuleList(
            A120BReaderBlock(config) for _ in range(config.reader_layers)
        )
        self.entity_presence_head = nn.Linear(d, 1)
        self.operation_presence_head = nn.Linear(d, 1)
        self.value_head = nn.Linear(d, core.config.value_classes)
        self.family_head = nn.Linear(d, core.config.family_classes)
        self.pointer_query = nn.Linear(d, d, bias=False)
        self.pointer_entity = nn.Linear(d, d, bias=False)

    def train(self, mode: bool = True) -> A120BLearnedFullTextBoundary:
        super().train(mode)
        self.core.eval()
        return self

    def _queries(self, batch: int, reference: torch.Tensor) -> torch.Tensor:
        d = self.config.latent_width
        entity_positions = _sinusoidal_positions(
            self.config.maximum_entities, d, reference
        )
        entities = self.entity_role_query.unsqueeze(0) + entity_positions
        entities = entities.unsqueeze(0).expand(batch, -1, -1)
        query = self.query_role_query.view(1, 1, d).expand(batch, -1, -1)
        operation_positions = _sinusoidal_positions(
            self.config.maximum_operations, d, reference
        )
        operations = (
            self.operation_role_queries.view(1, 3, d)
            + operation_positions.view(self.config.maximum_operations, 1, d)
        )
        operations = operations.reshape(1, -1, d).expand(batch, -1, -1)
        return torch.cat((entities, query, operations), dim=1)

    @staticmethod
    def _prefix_mask(logits: torch.Tensor, minimum: int) -> torch.Tensor:
        count = (logits > 0).sum(dim=-1).clamp(
            min=minimum, max=logits.shape[-1]
        )
        positions = torch.arange(logits.shape[-1], device=logits.device)
        return positions.unsqueeze(0) < count.unsqueeze(-1)

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if source_hidden.ndim != 3:
            raise ValueError("A1.20B source hidden must be [B,S,D]")
        if source_attention_mask.shape != source_hidden.shape[:2]:
            raise ValueError("A1.20B source mask shape mismatch")
        source_hidden = source_hidden.to(dtype=self.source_projection.weight.dtype)
        source = self.source_output_norm(
            self.source_projection(self.source_input_norm(source_hidden))
        )
        queries = self._queries(source.shape[0], source)
        for block in self.reader:
            queries = block(queries, source, ~source_attention_mask.bool())
        entity_end = self.config.maximum_entities
        entity_latents = queries[:, :entity_end]
        query_latent = queries[:, entity_end]
        operations = queries[:, entity_end + 1 :].reshape(
            source.shape[0], self.config.maximum_operations, 3, -1
        )
        family_latents = operations[:, :, 0]
        source_latents = operations[:, :, 1]
        target_latents = operations[:, :, 2]
        entity_presence_logits = self.entity_presence_head(entity_latents).squeeze(-1)
        operation_presence_logits = self.operation_presence_head(
            family_latents
        ).squeeze(-1)
        value_logits = self.value_head(entity_latents)
        family_logits = self.family_head(family_latents)
        query_pointer_logits = _cosine_pointer_logits(
            query_latent,
            entity_latents,
            self.pointer_query,
            self.pointer_entity,
            self.config.pointer_scale,
        )
        source_pointer_logits = _cosine_pointer_logits(
            source_latents,
            entity_latents,
            self.pointer_query,
            self.pointer_entity,
            self.config.pointer_scale,
        )
        target_pointer_logits = _cosine_pointer_logits(
            target_latents,
            entity_latents,
            self.pointer_query,
            self.pointer_entity,
            self.config.pointer_scale,
        )
        entity_mask = self._prefix_mask(entity_presence_logits, minimum=1)
        operation_mask = self._prefix_mask(operation_presence_logits, minimum=1)
        minimum = torch.finfo(query_pointer_logits.dtype).min
        masked_query = query_pointer_logits.masked_fill(~entity_mask, minimum)
        masked_source = source_pointer_logits.masked_fill(
            ~entity_mask.unsqueeze(1), minimum
        )
        masked_target = target_pointer_logits.masked_fill(
            ~entity_mask.unsqueeze(1), minimum
        )
        query_pointer = masked_query.argmax(dim=-1)
        source_pointer = masked_source.argmax(dim=-1)
        target_pointer = masked_target.argmax(dim=-1)
        family = family_logits.argmax(dim=-1)
        handles = (
            torch.arange(
                self.config.maximum_entities,
                device=source.device,
                dtype=torch.long,
            )
            .mul(104729)
            .add(10007)
            .unsqueeze(0)
            .expand(source.shape[0], -1)
        )
        query_handle = handles.gather(1, query_pointer.unsqueeze(-1)).squeeze(-1)
        source_handle = handles.gather(1, source_pointer)
        target_handle = handles.gather(1, target_pointer)
        value_probabilities = value_logits.softmax(dim=-1)
        entity_payloads = (
            self.core.value_initializer(value_probabilities)
            + self.core.shared_entity_seed
        )
        core_output = self.core(
            entity_values=torch.zeros_like(handles),
            entity_handles=handles,
            entity_mask=entity_mask,
            query_handle=query_handle,
            operation_family=family,
            operation_source_handle=source_handle,
            operation_target_handle=target_handle,
            operation_mask=operation_mask,
            recurrent_steps=self.config.maximum_operations,
            entity_payloads=entity_payloads,
        )
        core_output.update(
            {
                "entity_presence_logits": entity_presence_logits,
                "operation_presence_logits": operation_presence_logits,
                "value_mapping_logits": value_logits,
                "family_mapping_logits": family_logits,
                "query_pointer_logits": query_pointer_logits,
                "source_pointer_logits": source_pointer_logits,
                "target_pointer_logits": target_pointer_logits,
                "predicted_entity_mask": entity_mask,
                "predicted_operation_mask": operation_mask,
                "predicted_query_pointer": query_pointer,
                "predicted_source_pointer": source_pointer,
                "predicted_target_pointer": target_pointer,
                "predicted_family": family,
                "continuous_entity_payloads": entity_payloads,
            }
        )
        return core_output

    def boundary_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("core.")
        }

    def load_boundary_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        invalid_missing = [name for name in missing if not name.startswith("core.")]
        if invalid_missing or unexpected:
            raise ValueError(
                f"invalid A1.20B boundary state: missing={invalid_missing}, "
                f"unexpected={unexpected}"
            )

    def integrity_report(self) -> dict[str, Any]:
        core_frozen = all(
            not parameter.requires_grad for parameter in self.core.parameters()
        )
        return {
            "passed": self.core.integrity_report()["passed"] and core_frozen,
            "stage": "A1.20B",
            "full_source_hidden_only": True,
            "oracle_span_input": False,
            "oracle_role_tensor_input": False,
            "oracle_entity_mask_input": False,
            "oracle_operation_mask_input": False,
            "entity_presence_predicted": True,
            "operation_presence_predicted": True,
            "continuous_payload_to_core": True,
            "discrete_pointer_and_control_predicted": True,
            "entity_queries_share_parameters_across_positions": True,
            "operation_role_queries_share_parameters_across_steps": True,
            "source_direct_answer_head": False,
            "task_specific_executor": False,
            "core_frozen": core_frozen,
            "core_state_sha256": module_state_sha256(self.core),
            "config": asdict(self.config),
        }

    def parameter_report(self) -> dict[str, int]:
        boundary_trainable = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("core.") and parameter.requires_grad
        )
        return {
            "boundary_trainable_parameters": boundary_trainable,
            "frozen_core_parameters": sum(
                parameter.numel() for parameter in self.core.parameters()
            ),
        }
