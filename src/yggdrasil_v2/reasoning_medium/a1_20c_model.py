from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .a1_9_model import module_state_sha256
from .a1_19h_model import A119HHybridReasoner
from .a1_20b_model import (
    A120BBoundaryConfig,
    A120BLearnedFullTextBoundary,
    A120BReaderBlock,
    _cosine_pointer_logits,
    _sinusoidal_positions,
)


COMPILERS = ("flat", "hierarchical", "factorized")
CREDIT_MODES = ("hard_local", "straight_through")


@dataclass(frozen=True)
class A120CConfig:
    source_width: int
    compiler: str
    credit_mode: str
    latent_width: int = 256
    attention_heads: int = 8
    ffn_width: int = 1024
    reader_layers: int = 2
    maximum_entities: int = 5
    maximum_operations: int = 32
    pointer_scale: float = 20.0
    anchor_scale: float = 12.0
    pointer_temperature: float = 2.0
    family_temperature: float = 1.0


def _masked_anchor(
    queries: torch.Tensor,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    query_projection: nn.Linear,
    source_projection: nn.Linear,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = nn.functional.normalize(query_projection(queries), dim=-1)
    key = nn.functional.normalize(source_projection(source), dim=-1)
    logits = torch.einsum("bqd,bsd->bqs", query, key) * scale
    logits = logits.masked_fill(
        ~source_mask.bool().unsqueeze(1),
        torch.finfo(logits.dtype).min,
    )
    context = torch.einsum("bqs,bsd->bqd", logits.softmax(dim=-1), source)
    return logits, context


def _masked_anchor_values(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    source_mask: torch.Tensor,
    query_projection: nn.Linear,
    source_projection: nn.Linear,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = nn.functional.normalize(query_projection(queries), dim=-1)
    key = nn.functional.normalize(source_projection(keys), dim=-1)
    logits = torch.einsum("bqd,bsd->bqs", query, key) * scale
    logits = logits.masked_fill(
        ~source_mask.bool().unsqueeze(1),
        torch.finfo(logits.dtype).min,
    )
    context = torch.einsum(
        "bqs,bsd->bqd", logits.softmax(dim=-1), values
    )
    return logits, context


def _prefix_mask(logits: torch.Tensor, minimum: int) -> torch.Tensor:
    count = (logits > 0).sum(dim=-1).clamp(
        min=minimum, max=logits.shape[-1]
    )
    positions = torch.arange(logits.shape[-1], device=logits.device)
    return positions.unsqueeze(0) < count.unsqueeze(-1)


def _categorical_prefix(
    count_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = count_logits.softmax(dim=-1)
    capacity = count_logits.shape[-1]
    slot = torch.arange(capacity, device=count_logits.device)
    hard_count = count_logits.argmax(dim=-1) + 1
    hard_mask = slot.unsqueeze(0) < hard_count.unsqueeze(-1)
    survival = torch.stack(
        [probabilities[:, index:].sum(dim=-1) for index in range(capacity)],
        dim=-1,
    )
    epsilon = torch.finfo(survival.dtype).eps
    survival = survival.clamp(min=epsilon, max=1.0 - epsilon)
    presence_logits = torch.log(survival) - torch.log1p(-survival)
    return presence_logits, hard_mask


def _hard_core_forward(
    core: A119HHybridReasoner,
    *,
    entity_mask: torch.Tensor,
    operation_mask: torch.Tensor,
    family: torch.Tensor,
    source_pointer: torch.Tensor,
    target_pointer: torch.Tensor,
    query_pointer: torch.Tensor,
    entity_payloads: torch.Tensor,
    recurrent_steps: int,
) -> dict[str, Any]:
    handles = (
        torch.arange(
            core.config.maximum_entities,
            device=entity_mask.device,
            dtype=torch.long,
        )
        .mul(104729)
        .add(10007)
        .unsqueeze(0)
        .expand(entity_mask.shape[0], -1)
    )
    return core(
        entity_values=torch.zeros_like(handles),
        entity_handles=handles,
        entity_mask=entity_mask,
        query_handle=handles.gather(
            1, query_pointer.unsqueeze(-1)
        ).squeeze(-1),
        operation_family=family,
        operation_source_handle=handles.gather(1, source_pointer),
        operation_target_handle=handles.gather(1, target_pointer),
        operation_mask=operation_mask,
        recurrent_steps=recurrent_steps,
        entity_payloads=entity_payloads,
    )


def _straight_through(
    hard: torch.Tensor, soft: torch.Tensor
) -> torch.Tensor:
    return hard.to(dtype=soft.dtype) - soft.detach() + soft


def _straight_through_one_hot(
    logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    soft = (logits / temperature).softmax(dim=-1)
    hard = nn.functional.one_hot(
        logits.argmax(dim=-1), num_classes=logits.shape[-1]
    )
    return _straight_through(hard, soft)


def _weighted_core_forward(
    core: A119HHybridReasoner,
    *,
    entity_payloads: torch.Tensor,
    entity_mask: torch.Tensor,
    operation_mask: torch.Tensor,
    family_weights: torch.Tensor,
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    query_weights: torch.Tensor,
    recurrent_steps: int,
) -> dict[str, Any]:
    batch = entity_payloads.shape[0]
    dummy_values = torch.zeros(
        (batch, core.config.maximum_entities),
        dtype=torch.long,
        device=entity_payloads.device,
    )
    slots = core._initial_slots(dummy_values, entity_mask, entity_payloads)
    initial_slots = slots
    family_embeddings = torch.einsum(
        "btf,fd->btd",
        family_weights,
        core.transition.family_embedding.weight,
    )
    trajectory: list[torch.Tensor] = []
    for step in range(recurrent_steps):
        slots = core.transition(
            slots,
            family_embeddings[:, step],
            source_weights[:, step],
            target_weights[:, step],
            operation_mask[:, step],
        )
        trajectory.append(slots)
    trajectory_tensor = torch.stack(trajectory, dim=1)
    normalized = core.output_norm(trajectory_tensor)
    entity_normalized = normalized[:, :, : core.config.maximum_entities]
    state_logits = core.state_head(entity_normalized)
    effective_query = query_weights * entity_mask
    answer_logits = torch.einsum(
        "be,bev->bv", effective_query, state_logits[:, -1]
    )
    return {
        "initial_slots": initial_slots,
        "trajectory": trajectory_tensor,
        "state_logits": state_logits,
        "answer_logits": answer_logits,
        "training_auxiliary_logits": None,
    }


class A120CHierarchicalCompiler(nn.Module):
    """Entity-table-first compiler with learned token anchors as outputs."""

    def __init__(
        self, config: A120CConfig, core: A119HHybridReasoner
    ) -> None:
        super().__init__()
        self.config = config
        self.core = core
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.core.eval()
        d = config.latent_width
        reader_config = A120BBoundaryConfig(
            source_width=config.source_width,
            latent_width=d,
            attention_heads=config.attention_heads,
            ffn_width=config.ffn_width,
            reader_layers=config.reader_layers,
            maximum_entities=config.maximum_entities,
            maximum_operations=config.maximum_operations,
            pointer_scale=config.pointer_scale,
        )
        self.source_input_norm = nn.LayerNorm(
            config.source_width, elementwise_affine=False
        )
        self.source_projection = nn.Linear(config.source_width, d)
        self.source_output_norm = nn.LayerNorm(d)
        self.entity_role_query = nn.Parameter(torch.randn(d) * 0.02)
        self.operation_role_queries = nn.Parameter(torch.randn(3, d) * 0.02)
        self.query_role_query = nn.Parameter(torch.randn(d) * 0.02)
        self.entity_reader = nn.ModuleList(
            A120BReaderBlock(reader_config)
            for _ in range(config.reader_layers)
        )
        self.operation_reader = nn.ModuleList(
            A120BReaderBlock(reader_config)
            for _ in range(config.reader_layers)
        )
        self.query_reader = nn.ModuleList(
            A120BReaderBlock(reader_config)
            for _ in range(config.reader_layers)
        )
        self.anchor_source = nn.Linear(d, d, bias=False)
        self.entity_name_anchor = nn.Linear(d, d, bias=False)
        self.entity_value_anchor = nn.Linear(d, d, bias=False)
        self.operation_anchors = nn.ModuleList(
            nn.Linear(d, d, bias=False) for _ in range(3)
        )
        self.query_anchor = nn.Linear(d, d, bias=False)
        self.entity_output_norm = nn.LayerNorm(d)
        self.operation_output_norm = nn.LayerNorm(d)
        self.query_output_norm = nn.LayerNorm(d)
        self.entity_presence_head = nn.Linear(d, 1)
        self.operation_presence_head = nn.Linear(d, 1)
        self.value_head = nn.Linear(d, core.config.value_classes)
        self.family_head = nn.Linear(d, core.config.family_classes)
        self.pointer_query = nn.Linear(d, d, bias=False)
        self.pointer_entity = nn.Linear(d, d, bias=False)

    def train(self, mode: bool = True) -> A120CHierarchicalCompiler:
        super().train(mode)
        self.core.eval()
        return self

    def _source(
        self, source_hidden: torch.Tensor, source_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = source_hidden.to(dtype=self.source_projection.weight.dtype)
        source = self.source_output_norm(
            self.source_projection(self.source_input_norm(hidden))
        )
        positions = _sinusoidal_positions(
            source.shape[1], source.shape[-1], source
        )
        return source + positions.unsqueeze(0) * source_attention_mask.unsqueeze(
            -1
        ).to(dtype=source.dtype)

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if source_hidden.ndim != 3:
            raise ValueError("A1.20C source hidden must be [B,S,D]")
        source = self._source(source_hidden, source_attention_mask)
        batch, _, d = source.shape

        entity_positions = _sinusoidal_positions(
            self.config.maximum_entities, d, source
        )
        entities = (
            self.entity_role_query.unsqueeze(0) + entity_positions
        ).unsqueeze(0).expand(batch, -1, -1)
        for block in self.entity_reader:
            entities = block(
                entities, source, ~source_attention_mask.bool()
            )
        entity_name_anchor_logits, entity_name_context = _masked_anchor(
            entities,
            source,
            source_attention_mask,
            self.entity_name_anchor,
            self.anchor_source,
            self.config.anchor_scale,
        )
        entity_value_anchor_logits, entity_value_context = _masked_anchor(
            entities,
            source,
            source_attention_mask,
            self.entity_value_anchor,
            self.anchor_source,
            self.config.anchor_scale,
        )
        entity_latents = self.entity_output_norm(
            entities + entity_name_context + entity_value_context
        )

        operation_positions = _sinusoidal_positions(
            self.config.maximum_operations, d, source
        )
        operations = (
            self.operation_role_queries.view(1, 3, d)
            + operation_positions.view(
                self.config.maximum_operations, 1, d
            )
        )
        operations = operations.reshape(1, -1, d).expand(batch, -1, -1)
        for block in self.operation_reader:
            operations = block(
                operations, source, ~source_attention_mask.bool()
            )
        operation_roles = operations.reshape(
            batch, self.config.maximum_operations, 3, d
        )
        role_logits: list[torch.Tensor] = []
        role_contexts: list[torch.Tensor] = []
        for role in range(3):
            logits, context = _masked_anchor(
                operation_roles[:, :, role],
                source,
                source_attention_mask,
                self.operation_anchors[role],
                self.anchor_source,
                self.config.anchor_scale,
            )
            role_logits.append(logits)
            role_contexts.append(context)
        operation_anchor_logits = torch.stack(role_logits, dim=2)
        operation_context = torch.stack(role_contexts, dim=2)
        operation_roles = self.operation_output_norm(
            operation_roles + operation_context
        )
        family_latents = operation_roles[:, :, 0]
        source_latents = operation_roles[:, :, 1]
        target_latents = operation_roles[:, :, 2]

        query = self.query_role_query.view(1, 1, d).expand(batch, -1, -1)
        for block in self.query_reader:
            query = block(query, source, ~source_attention_mask.bool())
        query_anchor_logits, query_context = _masked_anchor(
            query,
            source,
            source_attention_mask,
            self.query_anchor,
            self.anchor_source,
            self.config.anchor_scale,
        )
        query_latent = self.query_output_norm(
            query + query_context
        ).squeeze(1)

        entity_presence_logits = self.entity_presence_head(
            entity_latents
        ).squeeze(-1)
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
        entity_mask = _prefix_mask(entity_presence_logits, minimum=1)
        operation_mask = _prefix_mask(
            operation_presence_logits, minimum=1
        )
        minimum = torch.finfo(query_pointer_logits.dtype).min
        masked_query = query_pointer_logits.masked_fill(
            ~entity_mask, minimum
        )
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
        value_probabilities = value_logits.softmax(dim=-1)
        entity_payloads = (
            self.core.value_initializer(value_probabilities)
            + self.core.shared_entity_seed
        )
        core_output = _hard_core_forward(
            self.core,
            entity_mask=entity_mask,
            operation_mask=operation_mask,
            family=family,
            source_pointer=source_pointer,
            target_pointer=target_pointer,
            query_pointer=query_pointer,
            entity_payloads=entity_payloads,
            recurrent_steps=self.config.maximum_operations,
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
                "entity_name_anchor_logits": entity_name_anchor_logits,
                "entity_value_anchor_logits": entity_value_anchor_logits,
                "operation_anchor_logits": operation_anchor_logits,
                "query_anchor_logits": query_anchor_logits.squeeze(1),
            }
        )
        return core_output


class A120DFactorizedCompiler(nn.Module):
    """Address/content-factorized compiler for the hybrid workspace."""

    def __init__(
        self, config: A120CConfig, core: A119HHybridReasoner
    ) -> None:
        super().__init__()
        self.config = config
        self.core = core
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.core.eval()
        d = config.latent_width
        reader_config = A120BBoundaryConfig(
            source_width=config.source_width,
            latent_width=d,
            attention_heads=config.attention_heads,
            ffn_width=config.ffn_width,
            reader_layers=config.reader_layers,
            maximum_entities=config.maximum_entities,
            maximum_operations=config.maximum_operations,
            pointer_scale=config.pointer_scale,
        )
        self.source_input_norm = nn.LayerNorm(
            config.source_width, elementwise_affine=False
        )
        self.source_projection = nn.Linear(config.source_width, d)
        self.source_output_norm = nn.LayerNorm(d)
        self.entity_role_query = nn.Parameter(torch.randn(d) * 0.02)
        self.operation_role_queries = nn.Parameter(torch.randn(3, d) * 0.02)
        self.query_role_query = nn.Parameter(torch.randn(d) * 0.02)
        self.entity_reader = nn.ModuleList(
            A120BReaderBlock(reader_config)
            for _ in range(config.reader_layers)
        )
        self.operation_reader = nn.ModuleList(
            A120BReaderBlock(reader_config)
            for _ in range(config.reader_layers)
        )
        self.query_reader = nn.ModuleList(
            A120BReaderBlock(reader_config)
            for _ in range(config.reader_layers)
        )
        self.anchor_source = nn.Linear(d, d, bias=False)
        self.entity_name_anchor = nn.Linear(d, d, bias=False)
        self.entity_value_anchor = nn.Linear(d, d, bias=False)
        self.operation_anchors = nn.ModuleList(
            nn.Linear(d, d, bias=False) for _ in range(3)
        )
        self.query_anchor = nn.Linear(d, d, bias=False)
        self.entity_identity_norm = nn.LayerNorm(d)
        self.entity_payload_norm = nn.LayerNorm(d)
        self.family_norm = nn.LayerNorm(d)
        self.operation_identity_norm = nn.LayerNorm(d)
        self.query_identity_norm = nn.LayerNorm(d)
        self.entity_count_head = nn.Linear(d, config.maximum_entities)
        self.operation_presence_head = nn.Linear(d, 1)
        self.value_head = nn.Linear(d, core.config.value_classes)
        self.payload_head = nn.Linear(d, d)
        nn.init.zeros_(self.payload_head.weight)
        nn.init.zeros_(self.payload_head.bias)
        self.family_head = nn.Linear(d, core.config.family_classes)
        self.identity_query = nn.Linear(d, d, bias=False)
        self.identity_entity = nn.Linear(d, d, bias=False)

    def train(self, mode: bool = True) -> A120DFactorizedCompiler:
        super().train(mode)
        self.core.eval()
        return self

    def _source(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = source_hidden.to(dtype=self.source_projection.weight.dtype)
        semantic = self.source_output_norm(
            self.source_projection(self.source_input_norm(hidden))
        )
        positions = _sinusoidal_positions(
            semantic.shape[1], semantic.shape[-1], semantic
        )
        reader = semantic + positions.unsqueeze(
            0
        ) * source_attention_mask.unsqueeze(-1).to(dtype=semantic.dtype)
        return semantic, reader

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if source_hidden.ndim != 3:
            raise ValueError("A1.20D source hidden must be [B,S,D]")
        semantic_source, reader_source = self._source(
            source_hidden, source_attention_mask
        )
        batch, _, d = reader_source.shape
        mask = source_attention_mask.bool()

        entity_positions = _sinusoidal_positions(
            self.config.maximum_entities, d, reader_source
        )
        entity_queries = (
            self.entity_role_query.unsqueeze(0) + entity_positions
        ).unsqueeze(0).expand(batch, -1, -1)
        for block in self.entity_reader:
            entity_queries = block(
                entity_queries, reader_source, ~mask
            )
        entity_name_logits, entity_name_context = _masked_anchor_values(
            entity_queries,
            reader_source,
            semantic_source,
            mask,
            self.entity_name_anchor,
            self.anchor_source,
            self.config.anchor_scale,
        )
        entity_value_logits, entity_value_context = _masked_anchor_values(
            entity_queries,
            reader_source,
            semantic_source,
            mask,
            self.entity_value_anchor,
            self.anchor_source,
            self.config.anchor_scale,
        )
        entity_identities = self.entity_identity_norm(
            entity_name_context
        )
        entity_payload_latents = self.entity_payload_norm(
            entity_queries + entity_value_context
        )

        operation_positions = _sinusoidal_positions(
            self.config.maximum_operations, d, reader_source
        )
        operation_queries = (
            self.operation_role_queries.view(1, 3, d)
            + operation_positions.view(
                self.config.maximum_operations, 1, d
            )
        )
        operation_queries = operation_queries.reshape(
            1, -1, d
        ).expand(batch, -1, -1)
        for block in self.operation_reader:
            operation_queries = block(
                operation_queries, reader_source, ~mask
            )
        operation_queries = operation_queries.reshape(
            batch, self.config.maximum_operations, 3, d
        )
        role_logits: list[torch.Tensor] = []
        role_contexts: list[torch.Tensor] = []
        for role in range(3):
            logits, context = _masked_anchor_values(
                operation_queries[:, :, role],
                reader_source,
                semantic_source,
                mask,
                self.operation_anchors[role],
                self.anchor_source,
                self.config.anchor_scale,
            )
            role_logits.append(logits)
            role_contexts.append(context)
        operation_anchor_logits = torch.stack(role_logits, dim=2)
        operation_contexts = torch.stack(role_contexts, dim=2)
        family_latents = self.family_norm(
            operation_queries[:, :, 0] + operation_contexts[:, :, 0]
        )
        source_identities = self.operation_identity_norm(
            operation_contexts[:, :, 1]
        )
        target_identities = self.operation_identity_norm(
            operation_contexts[:, :, 2]
        )

        query = self.query_role_query.view(1, 1, d).expand(
            batch, -1, -1
        )
        for block in self.query_reader:
            query = block(query, reader_source, ~mask)
        query_anchor_logits, query_context = _masked_anchor_values(
            query,
            reader_source,
            semantic_source,
            mask,
            self.query_anchor,
            self.anchor_source,
            self.config.anchor_scale,
        )
        query_identity = self.query_identity_norm(
            query_context
        ).squeeze(1)

        source_float_mask = source_attention_mask.unsqueeze(-1).to(
            dtype=semantic_source.dtype
        )
        global_source = (
            semantic_source * source_float_mask
        ).sum(dim=1) / source_float_mask.sum(dim=1).clamp_min(1.0)
        entity_count_logits = self.entity_count_head(global_source)
        entity_presence_logits, entity_mask = _categorical_prefix(
            entity_count_logits
        )
        operation_presence_logits = self.operation_presence_head(
            family_latents
        ).squeeze(-1)
        operation_mask = _prefix_mask(
            operation_presence_logits, minimum=1
        )

        value_logits = self.value_head(entity_payload_latents)
        family_logits = self.family_head(family_latents)
        query_pointer_logits = _cosine_pointer_logits(
            query_identity,
            entity_identities,
            self.identity_query,
            self.identity_entity,
            self.config.pointer_scale,
        )
        source_pointer_logits = _cosine_pointer_logits(
            source_identities,
            entity_identities,
            self.identity_query,
            self.identity_entity,
            self.config.pointer_scale,
        )
        target_pointer_logits = _cosine_pointer_logits(
            target_identities,
            entity_identities,
            self.identity_query,
            self.identity_entity,
            self.config.pointer_scale,
        )
        minimum = torch.finfo(query_pointer_logits.dtype).min
        masked_query = query_pointer_logits.masked_fill(
            ~entity_mask, minimum
        )
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
        entity_payloads = (
            self.payload_head(entity_payload_latents)
            + self.core.shared_entity_seed
        )
        core_output = _hard_core_forward(
            self.core,
            entity_mask=entity_mask,
            operation_mask=operation_mask,
            family=family,
            source_pointer=source_pointer,
            target_pointer=target_pointer,
            query_pointer=query_pointer,
            entity_payloads=entity_payloads,
            recurrent_steps=self.config.maximum_operations,
        )
        core_output.update(
            {
                "entity_presence_logits": entity_presence_logits,
                "operation_presence_logits": operation_presence_logits,
                "entity_count_logits": entity_count_logits,
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
                "entity_identity_latents": entity_identities,
                "entity_payload_latents": entity_payload_latents,
                "source_identity_latents": source_identities,
                "target_identity_latents": target_identities,
                "query_identity_latent": query_identity,
                "entity_name_anchor_logits": entity_name_logits,
                "entity_value_anchor_logits": entity_value_logits,
                "operation_anchor_logits": operation_anchor_logits,
                "query_anchor_logits": query_anchor_logits.squeeze(1),
            }
        )
        return core_output


class A120CFullTextBoundary(nn.Module):
    """A1.20C 2x2 wrapper: compiler structure x execution credit."""

    def __init__(
        self, config: A120CConfig, core: A119HHybridReasoner
    ) -> None:
        super().__init__()
        if config.compiler not in COMPILERS:
            raise ValueError(f"unknown A1.20C compiler {config.compiler}")
        if config.credit_mode not in CREDIT_MODES:
            raise ValueError(
                f"unknown A1.20C credit mode {config.credit_mode}"
            )
        if config.maximum_entities != core.config.maximum_entities:
            raise ValueError("A1.20C/core entity capacity mismatch")
        if config.latent_width != core.config.latent_width:
            raise ValueError("A1.20C/core latent width mismatch")
        self.config = config
        if config.compiler == "flat":
            self.boundary = A120BLearnedFullTextBoundary(
                A120BBoundaryConfig(
                    source_width=config.source_width,
                    latent_width=config.latent_width,
                    attention_heads=config.attention_heads,
                    ffn_width=config.ffn_width,
                    reader_layers=config.reader_layers,
                    maximum_entities=config.maximum_entities,
                    maximum_operations=config.maximum_operations,
                    pointer_scale=config.pointer_scale,
                ),
                core,
            )
        elif config.compiler == "hierarchical":
            self.boundary = A120CHierarchicalCompiler(config, core)
        else:
            self.boundary = A120DFactorizedCompiler(config, core)

    @property
    def core(self) -> A119HHybridReasoner:
        return self.boundary.core

    def train(self, mode: bool = True) -> A120CFullTextBoundary:
        super().train(mode)
        self.core.eval()
        return self

    def _straight_through_core(
        self, output: dict[str, Any]
    ) -> dict[str, Any]:
        entity_soft = output["entity_presence_logits"].sigmoid()
        operation_soft = output["operation_presence_logits"].sigmoid()
        entity_mask = _straight_through(
            output["predicted_entity_mask"], entity_soft
        )
        operation_mask = _straight_through(
            output["predicted_operation_mask"], operation_soft
        )
        minimum = torch.finfo(
            output["source_pointer_logits"].dtype
        ).min
        valid_entities = output["predicted_entity_mask"]
        source_logits = output["source_pointer_logits"].masked_fill(
            ~valid_entities.unsqueeze(1), minimum
        )
        target_logits = output["target_pointer_logits"].masked_fill(
            ~valid_entities.unsqueeze(1), minimum
        )
        query_logits = output["query_pointer_logits"].masked_fill(
            ~valid_entities, minimum
        )
        source_weights = _straight_through_one_hot(
            source_logits, temperature=self.config.pointer_temperature
        )
        target_weights = _straight_through_one_hot(
            target_logits, temperature=self.config.pointer_temperature
        )
        query_weights = _straight_through_one_hot(
            query_logits, temperature=self.config.pointer_temperature
        )
        family_weights = _straight_through_one_hot(
            output["family_mapping_logits"],
            temperature=self.config.family_temperature,
        )
        return _weighted_core_forward(
            self.core,
            entity_payloads=output["continuous_entity_payloads"],
            entity_mask=entity_mask,
            operation_mask=operation_mask,
            family_weights=family_weights,
            source_weights=source_weights,
            target_weights=target_weights,
            query_weights=query_weights,
            recurrent_steps=self.config.maximum_operations,
        )

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        output = self.boundary(source_hidden, source_attention_mask)
        if self.training and self.config.credit_mode == "straight_through":
            hard_state = output["state_logits"]
            hard_answer = output["answer_logits"]
            weighted = self._straight_through_core(output)
            output.update(weighted)
            output["hard_forward_state_max_abs_delta"] = (
                weighted["state_logits"] - hard_state.detach()
            ).abs().max()
            output["hard_forward_answer_max_abs_delta"] = (
                weighted["answer_logits"] - hard_answer.detach()
            ).abs().max()
        else:
            zero = output["state_logits"].new_zeros(())
            output["hard_forward_state_max_abs_delta"] = zero
            output["hard_forward_answer_max_abs_delta"] = zero
        return output

    def boundary_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if ".core." not in name and not name.startswith("core.")
        }

    def load_boundary_state_dict(
        self, state: dict[str, torch.Tensor]
    ) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        invalid_missing = [
            name
            for name in missing
            if ".core." not in name and not name.startswith("core.")
        ]
        if invalid_missing or unexpected:
            raise ValueError(
                "invalid A1.20C boundary state: "
                f"missing={invalid_missing}, unexpected={unexpected}"
            )

    def integrity_report(self) -> dict[str, Any]:
        core_frozen = all(
            not parameter.requires_grad for parameter in self.core.parameters()
        )
        return {
            "passed": self.core.integrity_report()["passed"] and core_frozen,
            "stage": "A1.20C",
            "compiler": self.config.compiler,
            "credit_mode": self.config.credit_mode,
            "full_source_hidden_only": True,
            "training_anchor_targets_are_forward_inputs": False,
            "hard_discrete_deployment": True,
            "straight_through_training_only": self.config.credit_mode
            == "straight_through",
            "source_direct_answer_head": False,
            "core_frozen": core_frozen,
            "core_state_sha256": module_state_sha256(self.core),
            "config": asdict(self.config),
        }

    def parameter_report(self) -> dict[str, int]:
        return {
            "boundary_trainable_parameters": sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if ".core." not in name
                and not name.startswith("core.")
                and parameter.requires_grad
            ),
            "frozen_core_parameters": sum(
                parameter.numel() for parameter in self.core.parameters()
            ),
        }
