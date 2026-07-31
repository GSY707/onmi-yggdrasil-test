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
IDENTITY_WINDOW_START = 1
IDENTITY_WINDOW_TOKENS = 11
ENTITY_PAIR_GAIN_THRESHOLD = 12.0
OPERATION_SUPPORTED_GAIN_THRESHOLD = 19.0
OPERATION_RECURRENT_GAIN_THRESHOLD = 10.5
OPERATION_CROSSING_GAIN_THRESHOLD = (
    OPERATION_SUPPORTED_GAIN_THRESHOLD
    + OPERATION_RECURRENT_GAIN_THRESHOLD
) / 2.0
OPERATION_POTENTIAL_VOTERS = 16


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


def _anchor_window_context(
    logits: torch.Tensor,
    values: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    start_offset: int = 0,
    width: int,
) -> torch.Tensor:
    return _anchor_window_tokens(
        logits,
        values,
        source_mask,
        start_offset=start_offset,
        width=width,
    ).mean(dim=-2)


def _anchor_window_tokens(
    logits: torch.Tensor,
    values: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    start_offset: int = 0,
    width: int,
) -> torch.Tensor:
    if start_offset < 0 or width < 1:
        raise ValueError("identity window offsets must be non-negative")
    return _anchor_window_tokens_from_probabilities(
        logits.softmax(dim=-1),
        values,
        source_mask,
        start_offset=start_offset,
        width=width,
    )


def _anchor_window_tokens_from_probabilities(
    probabilities: torch.Tensor,
    values: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    start_offset: int = 0,
    width: int,
) -> torch.Tensor:
    if start_offset < 0 or width < 1:
        raise ValueError("identity window offsets must be non-negative")
    sequence = values.shape[1]
    stop = min(start_offset + width, sequence)
    contexts: list[torch.Tensor] = []
    for offset in range(start_offset, stop):
        shifted_values = values.new_zeros(values.shape)
        shifted_values[:, : sequence - offset] = values[:, offset:]
        shifted_valid = source_mask.new_zeros(source_mask.shape)
        shifted_valid[:, : sequence - offset] = source_mask[:, offset:]
        weights = probabilities * shifted_valid.unsqueeze(1).to(
            dtype=probabilities.dtype
        )
        context = torch.einsum(
            "bqs,bsd->bqd", weights, shifted_values
        )
        context = context / weights.sum(dim=-1).clamp_min(
            torch.finfo(context.dtype).eps
        ).unsqueeze(-1)
        contexts.append(context)
    return torch.stack(contexts, dim=-2)


def _aligned_window_pointer_logits(
    queries: torch.Tensor,
    entities: torch.Tensor,
    query_projection: nn.Linear,
    entity_projection: nn.Linear,
    scale: float,
) -> torch.Tensor:
    """Compare identity tokens at equal offsets without pooling away order."""
    if queries.shape[-2:] != entities.shape[-2:]:
        raise ValueError("query/entity identity windows must have equal shape")
    query = nn.functional.normalize(
        query_projection(queries), dim=-1
    )
    entity = nn.functional.normalize(
        entity_projection(entities), dim=-1
    )
    aligned = torch.einsum("bqwd,bewd->bqew", query, entity)
    return aligned.mean(dim=-1) * scale


def _operation_count_from_gains(
    triple_gains: torch.Tensor,
    *,
    supported_gain_threshold: float,
    recurrent_gain_threshold: float,
    potential_voters: int,
) -> torch.Tensor:
    """Apply a high-confidence crossing gate, then recurrent continuation."""
    if triple_gains.ndim != 2:
        raise ValueError("operation triple gains must be [B,O-1]")
    increment_index = torch.arange(
        triple_gains.shape[1], device=triple_gains.device
    )
    supported_end = max(0, potential_voters - 1)
    crossing_threshold = (
        supported_gain_threshold + recurrent_gain_threshold
    ) / 2.0
    gain_thresholds = torch.where(
        increment_index < supported_end,
        triple_gains.new_full((), supported_gain_threshold),
        triple_gains.new_full((), recurrent_gain_threshold),
    )
    gain_thresholds = torch.where(
        increment_index == supported_end,
        triple_gains.new_full((), crossing_threshold),
        gain_thresholds,
    )
    supported_increment = (
        triple_gains > gain_thresholds.unsqueeze(0)
    ).long().cumprod(dim=-1)
    return 1 + supported_increment.sum(dim=-1)


def _shared_operation_viterbi_decode(
    logits: torch.Tensor,
    query_position: torch.Tensor,
    *,
    minimum_position: torch.Tensor | None = None,
    triple_gain_threshold: float = OPERATION_SUPPORTED_GAIN_THRESHOLD,
    recurrent_gain_threshold: float = OPERATION_RECURRENT_GAIN_THRESHOLD,
    potential_voters: int = OPERATION_POTENTIAL_VOTERS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode repeated family/source/target records from shared potentials.

    Operation slots are recurrent addresses, not distinct semantic roles.
    Their learned anchor logits therefore vote for one generic three-role
    grammar.  The query anchor closes the operation segment, while incremental
    Viterbi path score determines how many complete triples are present.
    Straight-through hard probabilities keep training gradients without
    changing the discrete deployment path.
    """
    if logits.ndim != 4 or logits.shape[2] != 3:
        raise ValueError("operation anchor logits must be [B,O,3,S]")
    batch, operations, roles, sequence = logits.shape
    if query_position.shape != (batch,):
        raise ValueError("query position batch mismatch")
    if minimum_position is not None and minimum_position.shape != (batch,):
        raise ValueError("operation minimum position batch mismatch")
    if sequence < roles:
        raise ValueError("operation decode requires at least three tokens")
    if potential_voters < 1:
        raise ValueError("operation potential voters must be positive")

    # Only positively supervised readers may vote for the generic role
    # potential.  Queries 16--31 are recurrent capacity for OOD execution, not
    # separately trained semantic roles; including them in a max creates
    # arbitrary long-context peaks.
    supported_voters = min(operations, potential_voters)
    shared_logits = logits[:, :supported_voters].max(dim=1).values
    source_position = torch.arange(sequence, device=logits.device)
    query_boundary = query_position.clamp(min=roles, max=sequence)
    before_query = (
        source_position.unsqueeze(0) < query_boundary.unsqueeze(-1)
    )
    in_operation_section = before_query
    if minimum_position is not None:
        latest_valid_start = (query_boundary - roles).clamp_min(0)
        lower_boundary = torch.minimum(
            minimum_position.clamp(min=0, max=sequence - 1),
            latest_valid_start,
        )
        in_operation_section = in_operation_section & (
            source_position.unsqueeze(0) >= lower_boundary.unsqueeze(-1)
        )
    shared_logits = shared_logits.masked_fill(
        ~in_operation_section.unsqueeze(1), -torch.inf
    )
    repeated_logits = shared_logits.unsqueeze(1).expand(
        -1, operations, -1, -1
    ).reshape(batch, operations * roles, sequence)

    dynamic = repeated_logits[:, 0]
    dynamic_rows = [dynamic]
    backpointers: list[torch.Tensor] = []
    triple_scores: list[torch.Tensor] = []
    for step in range(1, operations * roles):
        prefix_score, prefix_index = torch.cummax(dynamic, dim=-1)
        previous_score = torch.cat(
            (
                dynamic.new_full((batch, 1), -torch.inf),
                prefix_score[:, :-1],
            ),
            dim=-1,
        )
        previous_index = torch.cat(
            (
                prefix_index.new_full((batch, 1), -1),
                prefix_index[:, :-1],
            ),
            dim=-1,
        )
        dynamic = repeated_logits[:, step] + previous_score
        dynamic_rows.append(dynamic)
        backpointers.append(previous_index)
        if step % roles == roles - 1:
            triple_scores.append(dynamic.max(dim=-1).values)

    first_triple_score = dynamic_rows[roles - 1].max(dim=-1).values
    stacked_triple_scores = torch.stack(
        (first_triple_score, *triple_scores[1:]), dim=1
    )
    triple_gains = (
        stacked_triple_scores[:, 1:] - stacked_triple_scores[:, :-1]
    )
    # Inside the directly supervised horizon use its high
    # validation-calibrated threshold.  Crossing the horizon uses the midpoint
    # between supported and recurrent regimes; only after that proof do later
    # records use the validation background ceiling.  Shared max-potential
    # magnitude attenuates outside supervised positions, while true/background
    # ordering remains intact.
    operation_count = _operation_count_from_gains(
        triple_gains,
        supported_gain_threshold=triple_gain_threshold,
        recurrent_gain_threshold=recurrent_gain_threshold,
        potential_voters=potential_voters,
    )
    path_lengths = operation_count * roles
    stacked_dynamic = torch.stack(dynamic_rows, dim=1)
    terminal_scores = stacked_dynamic[
        torch.arange(batch, device=logits.device),
        path_lengths - 1,
    ]
    current = terminal_scores.argmax(dim=-1)
    selected_positions = logits.argmax(dim=-1).reshape(
        batch, operations * roles
    )
    for step in range(operations * roles - 1, -1, -1):
        active = step < path_lengths
        selected_positions[:, step] = torch.where(
            active, current, selected_positions[:, step]
        )
        if step:
            previous = backpointers[step - 1].gather(
                1, current.unsqueeze(-1)
            ).squeeze(-1)
            current = torch.where(active, previous, current)

    selected_positions = selected_positions.reshape(
        batch, operations, roles
    )
    raw_probabilities = logits.softmax(dim=-1)
    shared_probabilities = shared_logits.softmax(dim=-1)
    repeated_probabilities = shared_probabilities.unsqueeze(1).expand(
        -1, operations, -1, -1
    )
    hard_probabilities = nn.functional.one_hot(
        selected_positions, num_classes=sequence
    ).to(dtype=repeated_probabilities.dtype)
    straight_through = (
        hard_probabilities
        + repeated_probabilities
        - repeated_probabilities.detach()
    )
    operation_mask = (
        torch.arange(operations, device=logits.device)
        .unsqueeze(0)
        .lt(operation_count.unsqueeze(-1))
    )
    decoded_probabilities = torch.where(
        operation_mask.unsqueeze(-1).unsqueeze(-1),
        straight_through,
        raw_probabilities,
    )
    return (
        decoded_probabilities,
        selected_positions,
        operation_mask,
        triple_gains,
    )


def _shared_entity_viterbi_decode(
    entity_name_logits: torch.Tensor,
    entity_value_logits: torch.Tensor,
    first_operation_position: torch.Tensor,
    *,
    pair_gain_threshold: float = ENTITY_PAIR_GAIN_THRESHOLD,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Decode an ordered entity table from shared learned potentials.

    The capacity-final query is excluded from the potential pool because the
    N2--N4 training contract never gives it a positive anchor target.  The
    supported queries vote for generic name/value positions; a single
    alternating Viterbi path then extrapolates to the capacity-final entity.
    """
    if entity_name_logits.shape != entity_value_logits.shape:
        raise ValueError("entity name/value logits must have equal shape")
    if entity_name_logits.ndim != 3:
        raise ValueError("entity anchor logits must be [B,E,S]")
    batch, entities, sequence = entity_name_logits.shape
    if entities < 2:
        raise ValueError("shared entity decode requires capacity >= 2")
    if first_operation_position.shape != (batch,):
        raise ValueError("first operation position batch mismatch")
    if sequence < 2:
        raise ValueError("entity decode requires at least two tokens")

    supported = entities - 1
    shared_name = entity_name_logits[:, :supported].max(dim=1).values
    shared_value = entity_value_logits[:, :supported].max(dim=1).values
    source_position = torch.arange(
        sequence, device=entity_name_logits.device
    )
    operation_boundary = first_operation_position.clamp(
        min=2, max=sequence
    )
    before_operations = (
        source_position.unsqueeze(0)
        < operation_boundary.unsqueeze(-1)
    )
    shared_name = shared_name.masked_fill(
        ~before_operations, -torch.inf
    )
    shared_value = shared_value.masked_fill(
        ~before_operations, -torch.inf
    )
    alternating = torch.stack(
        (shared_name, shared_value), dim=1
    ).repeat(1, entities, 1)

    dynamic = alternating[:, 0]
    dynamic_rows = [dynamic]
    backpointers: list[torch.Tensor] = []
    pair_scores: list[torch.Tensor] = []
    for step in range(1, entities * 2):
        prefix_score, prefix_index = torch.cummax(dynamic, dim=-1)
        previous_score = torch.cat(
            (
                dynamic.new_full((batch, 1), -torch.inf),
                prefix_score[:, :-1],
            ),
            dim=-1,
        )
        previous_index = torch.cat(
            (
                prefix_index.new_full((batch, 1), -1),
                prefix_index[:, :-1],
            ),
            dim=-1,
        )
        dynamic = alternating[:, step] + previous_score
        dynamic_rows.append(dynamic)
        backpointers.append(previous_index)
        if step % 2:
            pair_scores.append(dynamic.max(dim=-1).values)

    stacked_pair_scores = torch.stack(pair_scores, dim=1)
    pair_gains = (
        stacked_pair_scores[:, 1:] - stacked_pair_scores[:, :-1]
    )
    supported_increment = (
        pair_gains > pair_gain_threshold
    ).long().cumprod(dim=-1)
    entity_count = 1 + supported_increment.sum(dim=-1)
    path_lengths = entity_count * 2

    stacked_dynamic = torch.stack(dynamic_rows, dim=1)
    terminal_scores = stacked_dynamic[
        torch.arange(batch, device=entity_name_logits.device),
        path_lengths - 1,
    ]
    current = terminal_scores.argmax(dim=-1)
    raw_positions = torch.stack(
        (
            entity_name_logits.argmax(dim=-1),
            entity_value_logits.argmax(dim=-1),
        ),
        dim=2,
    ).reshape(batch, entities * 2)
    selected_positions = raw_positions.clone()
    for step in range(entities * 2 - 1, -1, -1):
        active = step < path_lengths
        selected_positions[:, step] = torch.where(
            active, current, selected_positions[:, step]
        )
        if step:
            previous = backpointers[step - 1].gather(
                1, current.unsqueeze(-1)
            ).squeeze(-1)
            current = torch.where(active, previous, current)

    selected_positions = selected_positions.reshape(
        batch, entities, 2
    )
    raw_probabilities = torch.stack(
        (
            entity_name_logits.softmax(dim=-1),
            entity_value_logits.softmax(dim=-1),
        ),
        dim=2,
    )
    shared_probabilities = torch.stack(
        (
            shared_name.softmax(dim=-1),
            shared_value.softmax(dim=-1),
        ),
        dim=1,
    ).unsqueeze(1).expand(-1, entities, -1, -1)
    hard_probabilities = nn.functional.one_hot(
        selected_positions, num_classes=sequence
    ).to(dtype=shared_probabilities.dtype)
    straight_through = (
        hard_probabilities
        + shared_probabilities
        - shared_probabilities.detach()
    )
    entity_mask = (
        torch.arange(entities, device=entity_name_logits.device)
        .unsqueeze(0)
        .lt(entity_count.unsqueeze(-1))
    )
    decoded_probabilities = torch.where(
        entity_mask.unsqueeze(-1).unsqueeze(-1),
        straight_through,
        raw_probabilities,
    )
    return (
        decoded_probabilities[:, :, 0],
        decoded_probabilities[:, :, 1],
        selected_positions[:, :, 0],
        selected_positions[:, :, 1],
        entity_mask,
        pair_gains,
    )


def _prefix_mask(logits: torch.Tensor, minimum: int) -> torch.Tensor:
    count = (logits > 0).sum(dim=-1).clamp(
        min=minimum, max=logits.shape[-1]
    )
    positions = torch.arange(logits.shape[-1], device=logits.device)
    return positions.unsqueeze(0) < count.unsqueeze(-1)


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
        entity_name_logits, _ = _masked_anchor_values(
            entity_queries,
            reader_source,
            semantic_source,
            mask,
            self.entity_name_anchor,
            self.anchor_source,
            self.config.anchor_scale,
        )
        entity_value_logits, _ = _masked_anchor_values(
            entity_queries,
            reader_source,
            semantic_source,
            mask,
            self.entity_value_anchor,
            self.anchor_source,
            self.config.anchor_scale,
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
        for role in range(3):
            logits, _ = _masked_anchor_values(
                operation_queries[:, :, role],
                reader_source,
                semantic_source,
                mask,
                self.operation_anchors[role],
                self.anchor_source,
                self.config.anchor_scale,
            )
            role_logits.append(logits)
        operation_anchor_logits = torch.stack(role_logits, dim=2)

        query = self.query_role_query.view(1, 1, d).expand(
            batch, -1, -1
        )
        for block in self.query_reader:
            query = block(query, reader_source, ~mask)
        query_anchor_logits, _ = _masked_anchor_values(
            query,
            reader_source,
            semantic_source,
            mask,
            self.query_anchor,
            self.anchor_source,
            self.config.anchor_scale,
        )
        raw_query_position = query_anchor_logits.squeeze(1).argmax(dim=-1)
        (
            _,
            preliminary_operation_positions,
            _,
            _,
        ) = _shared_operation_viterbi_decode(
            operation_anchor_logits,
            raw_query_position,
        )
        # A raw slot-0 reader and the shared recurrent path fail in opposite
        # directions under layout transfer: the raw reader can lock onto an
        # instruction word, while the recurrent path can insert a prefix
        # triple.  Their later start is a conservative, task-agnostic section
        # boundary.  It closes the entity table and then becomes a hard lower
        # bound for a second operation decode.
        raw_first_operation_position = operation_anchor_logits[
            :, 0, 0
        ].argmax(dim=-1)
        first_operation_position = torch.maximum(
            preliminary_operation_positions[:, 0, 0],
            raw_first_operation_position,
        )
        (
            entity_name_probabilities,
            entity_value_probabilities,
            entity_name_positions,
            entity_value_positions,
            entity_mask,
            entity_pair_gains,
        ) = _shared_entity_viterbi_decode(
            entity_name_logits,
            entity_value_logits,
            first_operation_position,
        )
        (
            operation_anchor_probabilities,
            operation_anchor_positions,
            operation_mask,
            operation_triple_gains,
        ) = _shared_operation_viterbi_decode(
            operation_anchor_logits,
            raw_query_position,
            minimum_position=first_operation_position,
        )
        # Complete the bidirectional section refinement.  The conservative
        # preliminary operation boundary is intentionally allowed to be early
        # so that operation decoding cannot consume the entity table.  Once
        # the ordered operation path has found its actual first family token,
        # that sharper boundary must be fed back into the entity decoder.
        # Without this final pass, an early preliminary boundary can truncate
        # the fifth entity even though the refined operation path is exact.
        refined_first_operation_position = operation_anchor_positions[
            :, 0, 0
        ]
        (
            entity_name_probabilities,
            entity_value_probabilities,
            entity_name_positions,
            entity_value_positions,
            entity_mask,
            entity_pair_gains,
        ) = _shared_entity_viterbi_decode(
            entity_name_logits,
            entity_value_logits,
            refined_first_operation_position,
        )
        operation_contexts = torch.einsum(
            "bors,bsd->bord",
            operation_anchor_probabilities,
            semantic_source,
        )
        source_identity_tokens = (
            _anchor_window_tokens_from_probabilities(
                operation_anchor_probabilities[:, :, 1],
                semantic_source,
                mask,
                start_offset=IDENTITY_WINDOW_START,
                width=IDENTITY_WINDOW_TOKENS,
            )
        )
        target_identity_tokens = (
            _anchor_window_tokens_from_probabilities(
                operation_anchor_probabilities[:, :, 2],
                semantic_source,
                mask,
                start_offset=IDENTITY_WINDOW_START,
                width=IDENTITY_WINDOW_TOKENS,
            )
        )
        entity_identity_tokens = (
            _anchor_window_tokens_from_probabilities(
                entity_name_probabilities,
                semantic_source,
                mask,
                start_offset=IDENTITY_WINDOW_START,
                width=IDENTITY_WINDOW_TOKENS,
            )
        )
        entity_value_context = torch.einsum(
            "bes,bsd->bed",
            entity_value_probabilities,
            semantic_source,
        )
        entity_identity_tokens = self.entity_identity_norm(
            entity_identity_tokens
        )
        entity_identities = entity_identity_tokens.mean(dim=-2)
        entity_payload_latents = self.entity_payload_norm(
            entity_queries + entity_value_context
        )
        # Family semantics are recurrent and slot-anonymous.  Capacity slots
        # beyond the supervised horizon must not inject their untrained
        # positional query into the classifier.  A schema-conditioned prior
        # pooled only from supervised readers is shared across every step;
        # the decoded family-token context supplies step-local evidence.
        shared_family_query = operation_queries[
            :, : min(
                self.config.maximum_operations,
                OPERATION_POTENTIAL_VOTERS,
            ),
            0,
        ].mean(dim=1, keepdim=True)
        family_latents = self.family_norm(
            shared_family_query.expand(-1, self.config.maximum_operations, -1)
            + operation_contexts[:, :, 0]
        )
        source_identity_tokens = self.operation_identity_norm(
            source_identity_tokens
        )
        target_identity_tokens = self.operation_identity_norm(
            target_identity_tokens
        )
        source_identities = source_identity_tokens.mean(dim=-2)
        target_identities = target_identity_tokens.mean(dim=-2)

        query_identity_tokens = _anchor_window_tokens(
            query_anchor_logits,
            semantic_source,
            mask,
            start_offset=IDENTITY_WINDOW_START,
            width=IDENTITY_WINDOW_TOKENS,
        )
        query_identity_tokens = self.query_identity_norm(
            query_identity_tokens
        )
        query_identity = query_identity_tokens.mean(dim=-2).squeeze(1)

        entity_presence_logits = torch.where(
            entity_mask,
            entity_queries.new_full((), 20.0),
            entity_queries.new_full((), -20.0),
        )
        operation_presence_logits = torch.where(
            operation_mask,
            operation_queries.new_full((), 20.0),
            operation_queries.new_full((), -20.0),
        )

        value_logits = self.value_head(entity_payload_latents)
        family_logits = self.family_head(family_latents)
        query_pointer_logits = _aligned_window_pointer_logits(
            query_identity_tokens,
            entity_identity_tokens,
            self.identity_query,
            self.identity_entity,
            self.config.pointer_scale,
        ).squeeze(1)
        source_pointer_logits = _aligned_window_pointer_logits(
            source_identity_tokens,
            entity_identity_tokens,
            self.identity_query,
            self.identity_entity,
            self.config.pointer_scale,
        )
        target_pointer_logits = _aligned_window_pointer_logits(
            target_identity_tokens,
            entity_identity_tokens,
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
                "predicted_entity_name_anchor_positions": (
                    entity_name_positions
                ),
                "predicted_entity_value_anchor_positions": (
                    entity_value_positions
                ),
                "entity_pair_score_gains": entity_pair_gains,
                "operation_anchor_logits": operation_anchor_logits,
                "predicted_operation_anchor_positions": (
                    operation_anchor_positions
                ),
                "operation_triple_score_gains": operation_triple_gains,
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
            "structural_entity_cardinality": (
                self.config.compiler == "factorized"
            ),
            "shared_entity_viterbi_decode": (
                self.config.compiler == "factorized"
            ),
            "shared_operation_viterbi_decode": (
                self.config.compiler == "factorized"
            ),
            "operation_potential_voters": (
                OPERATION_POTENTIAL_VOTERS
                if self.config.compiler == "factorized"
                else None
            ),
            "operation_supported_gain_threshold": (
                OPERATION_SUPPORTED_GAIN_THRESHOLD
                if self.config.compiler == "factorized"
                else None
            ),
            "operation_recurrent_gain_threshold": (
                OPERATION_RECURRENT_GAIN_THRESHOLD
                if self.config.compiler == "factorized"
                else None
            ),
            "operation_crossing_gain_threshold": (
                OPERATION_CROSSING_GAIN_THRESHOLD
                if self.config.compiler == "factorized"
                else None
            ),
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
