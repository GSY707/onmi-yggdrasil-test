from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .a1_7_data import FAMILY_TO_INDEX, REGISTER_NAMES, VALUE_LABELS


@dataclass(frozen=True)
class A119HBatch:
    inputs: dict[str, torch.Tensor]
    state_targets: torch.Tensor
    answer_targets: torch.Tensor
    semantic_slot_orders: tuple[tuple[str, ...], ...]


def move_a119h_batch(
    batch: A119HBatch, device: str | torch.device
) -> A119HBatch:
    return A119HBatch(
        inputs={name: value.to(device) for name, value in batch.inputs.items()},
        state_targets=batch.state_targets.to(device),
        answer_targets=batch.answer_targets.to(device),
        semantic_slot_orders=batch.semantic_slot_orders,
    )


def index_a119h_batch(
    batch: A119HBatch, indices: torch.Tensor
) -> A119HBatch:
    return A119HBatch(
        inputs={
            name: value.index_select(0, indices) for name, value in batch.inputs.items()
        },
        state_targets=batch.state_targets.index_select(0, indices),
        answer_targets=batch.answer_targets.index_select(0, indices),
        semantic_slot_orders=(),
    )


def a119h_batch_nbytes(batch: A119HBatch) -> int:
    tensors = [*batch.inputs.values(), batch.state_targets, batch.answer_targets]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def validate_a119h_batch_addresses(batch: A119HBatch) -> None:
    """Validate opaque-address routing once, outside the recurrent hot path."""

    handles = batch.inputs["entity_handles"]
    entity_mask = batch.inputs["entity_mask"]
    operation_mask = batch.inputs["operation_mask"]
    source_matches = (
        handles.unsqueeze(1)
        == batch.inputs["operation_source_handle"].unsqueeze(-1)
    ) & entity_mask.unsqueeze(1)
    target_matches = (
        handles.unsqueeze(1)
        == batch.inputs["operation_target_handle"].unsqueeze(-1)
    ) & entity_mask.unsqueeze(1)
    if not torch.all((source_matches.sum(dim=-1) == 1) | ~operation_mask):
        raise ValueError("active source handle must match exactly one entity")
    if not torch.all((target_matches.sum(dim=-1) == 1) | ~operation_mask):
        raise ValueError("active target handle must match exactly one entity")
    query_matches = (
        handles == batch.inputs["query_handle"].unsqueeze(-1)
    ) & entity_mask
    if not torch.all(query_matches.sum(dim=-1) == 1):
        raise ValueError("query handle must match exactly one active entity")


def _record_seed(record: dict[str, Any], mapping_seed: int) -> int:
    key = str(record.get("a119h_mapping_key", record["fingerprint"]))
    digest = hashlib.sha256(f"{mapping_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def entity_names(record: dict[str, Any]) -> tuple[str, ...]:
    names = tuple(record.get("entity_names", REGISTER_NAMES))
    if not names or len(set(names)) != len(names):
        raise ValueError("A1.19H entity names must be non-empty and unique")
    return names


def _mapping(
    record: dict[str, Any], mapping_seed: int
) -> tuple[tuple[str, ...], dict[str, int]]:
    rng = random.Random(_record_seed(record, mapping_seed))
    names = entity_names(record)
    slot_order = list(names)
    rng.shuffle(slot_order)
    handles = rng.sample(range(10_000, 2_000_000_000), len(slot_order))
    return tuple(slot_order), dict(zip(names, handles))


def encode_a119h_records(
    records: Sequence[dict[str, Any]],
    recurrent_steps: int,
    device: str | torch.device,
    *,
    mapping_seed: int,
    maximum_entities: int = 3,
    maximum_operations: int | None = None,
    handle_alias_seed: int | None = None,
    include_semantic_slot_orders: bool = True,
) -> A119HBatch:
    if not records:
        raise ValueError("A1.19H cannot encode an empty batch")
    if maximum_entities < max(len(entity_names(record)) for record in records):
        raise ValueError("A1.19H maximum entity capacity is too small for this batch")
    batch = len(records)
    required_operations = max(int(record["program_length"]) for record in records)
    if maximum_operations is None:
        maximum_operations = required_operations
    elif maximum_operations < required_operations:
        raise ValueError("A1.19H operation capacity is too small for this batch")
    entity_values = torch.zeros(
        (batch, maximum_entities), dtype=torch.long, device=device
    )
    entity_handles = torch.full(
        (batch, maximum_entities), -1, dtype=torch.long, device=device
    )
    entity_mask = torch.zeros(
        (batch, maximum_entities), dtype=torch.bool, device=device
    )
    query_handle = torch.empty((batch,), dtype=torch.long, device=device)
    operation_family = torch.zeros(
        (batch, maximum_operations), dtype=torch.long, device=device
    )
    operation_source_handle = torch.full(
        (batch, maximum_operations), -2, dtype=torch.long, device=device
    )
    operation_target_handle = torch.full(
        (batch, maximum_operations), -3, dtype=torch.long, device=device
    )
    operation_mask = torch.zeros(
        (batch, maximum_operations), dtype=torch.bool, device=device
    )
    state_targets = torch.full(
        (batch, recurrent_steps, maximum_entities),
        -100,
        dtype=torch.long,
        device=device,
    )
    answer_targets = torch.empty((batch,), dtype=torch.long, device=device)
    slot_orders: list[tuple[str, ...]] = []

    for row, record in enumerate(records):
        slot_order, handles = _mapping(record, mapping_seed)
        if handle_alias_seed is not None:
            alias_rng = random.Random(
                _record_seed(record, mapping_seed) ^ int(handle_alias_seed)
            )
            names = entity_names(record)
            aliases = alias_rng.sample(
                range(2_000_000_001, 4_000_000_000), len(names)
            )
            handles = dict(zip(names, aliases))
        if include_semantic_slot_orders:
            slot_orders.append(
                slot_order
                + tuple(
                    f"__inactive_{slot}"
                    for slot in range(len(slot_order), maximum_entities)
                )
            )
        entity_mask[row, : len(slot_order)] = True
        for slot, name in enumerate(slot_order):
            entity_values[row, slot] = VALUE_LABELS.index(record["start_state"][name])
            entity_handles[row, slot] = handles[name]
        query_handle[row] = handles[record["query_register"]]
        for step, operation in enumerate(record["operations"]):
            operation_mask[row, step] = True
            operation_family[row, step] = FAMILY_TO_INDEX[operation["family"]]
            operation_source_handle[row, step] = handles[operation["source"]]
            operation_target_handle[row, step] = handles[operation["target"]]
        trajectory = record["state_trajectory"] or [record["start_state"]]
        for step in range(recurrent_steps):
            state = trajectory[min(step, len(trajectory) - 1)]
            for slot, name in enumerate(slot_order):
                state_targets[row, step, slot] = VALUE_LABELS.index(state[name])
        answer_targets[row] = int(record["answer_index"])

    return A119HBatch(
        inputs={
            "entity_values": entity_values,
            "entity_handles": entity_handles,
            "entity_mask": entity_mask,
            "query_handle": query_handle,
            "operation_family": operation_family,
            "operation_source_handle": operation_source_handle,
            "operation_target_handle": operation_target_handle,
            "operation_mask": operation_mask,
        },
        state_targets=state_targets,
        answer_targets=answer_targets,
        semantic_slot_orders=tuple(slot_orders),
    )


def permute_entity_axis(
    batch: A119HBatch, permutation: torch.Tensor
) -> A119HBatch:
    inputs = dict(batch.inputs)
    for name in ("entity_values", "entity_handles", "entity_mask"):
        inputs[name] = inputs[name][:, permutation]
    return A119HBatch(
        inputs=inputs,
        state_targets=batch.state_targets[:, :, permutation],
        answer_targets=batch.answer_targets,
        semantic_slot_orders=tuple(
            tuple(order[int(index)] for index in permutation.cpu())
            for order in batch.semantic_slot_orders
        ),
    )
