from __future__ import annotations

"""C1S successor data boundary.

The cache is immutable and model-facing data is deliberately narrow.  This
module joins a caller-ordered cache batch with an offline successor target
bank, but keeps the two interfaces explicit: ``get_batch`` returns an
objective batch, while ``forward_inputs`` accepts *only* source hidden states
and their padding mask.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from ..closure_c1.cache import CachedShardDataset
from . import contract
from .targets import LABELS, MAX_CHOICES, MAX_SEMANTIC_OBJECTS, MODEL_STEPS, TARGET_BANK_SCHEMA


RUNTIME_SCHEMA = "yggdrasil.v2-a.closure-c1s-successor.runtime.v1"
SOURCE_KEYS = frozenset({"source_hidden", "source_mask"})
FORWARD_FORBIDDEN_FIELDS = frozenset(contract.FORBIDDEN_FORWARD_FIELDS)
SOURCE_WIDTH = int(contract.MODEL_CONFIG["source_width"])
STATE_FEATURES = 8


def _require_slots(slots: int) -> int:
    if type(slots) is not int or slots not in {1, MAX_CHOICES}:
        raise ValueError("successor runtime supports matched slots=1 or slots=8")
    return slots


def _int(value: Any, name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _vector(value: Any, name: str, length: int) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must be a sequence of length {length}")
    return list(value)


def _lookup_targets(target_bank: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(target_bank, Mapping):
        raise TypeError("target_bank must be a mapping")
    if target_bank.get("schema_version") == TARGET_BANK_SCHEMA:
        records = target_bank.get("records")
    elif isinstance(target_bank.get("records"), Mapping):
        records = target_bank["records"]
    else:
        records = target_bank
    if not isinstance(records, Mapping):
        raise ValueError("target_bank.records must be a mapping")
    return records  # type: ignore[return-value]


def _validate_source_inputs(source: Mapping[str, Any], ids: Sequence[str]) -> tuple[Tensor, Tensor]:
    if not isinstance(source, Mapping):
        raise TypeError("cache provider must return a source mapping")
    keys = set(source)
    if keys != SOURCE_KEYS:
        forbidden = sorted(keys & FORWARD_FORBIDDEN_FIELDS)
        detail = f" forbidden={forbidden}" if forbidden else ""
        raise ValueError(f"cache source mapping must contain exactly {sorted(SOURCE_KEYS)}; got {sorted(keys)}{detail}")
    hidden = source["source_hidden"]
    mask = source["source_mask"]
    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise ValueError("source_hidden must be a rank-3 tensor")
    if hidden.shape[0] != len(ids) or hidden.shape[-1] != SOURCE_WIDTH:
        raise ValueError(f"source_hidden must have shape [batch,{SOURCE_WIDTH}] at the trailing dimensions")
    if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all().item()):
        raise ValueError("source_hidden must be finite floating-point data")
    if not isinstance(mask, Tensor) or mask.dtype is not torch.bool or mask.shape != hidden.shape[:2]:
        raise ValueError("source_mask must be bool [batch,source_tokens]")
    if bool((mask.sum(dim=1) < 1).any().item()):
        raise ValueError("every source row must contain at least one token")
    return hidden, mask


def _label_index(value: Any, name: str, *, allow_missing: bool = False) -> int:
    if allow_missing and (value is None or value == -1):
        return -1
    if type(value) is int:
        return _int(value, name, minimum=0, maximum=len(LABELS) - 1)
    if isinstance(value, str) and value.upper() in LABELS:
        return LABELS.index(value.upper())
    raise ValueError(f"{name} must be an A-I label or index")


def _validate_target_row(row: Mapping[str, Any], example_id: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"target row is not a mapping: {example_id}")
    if row.get("example_id") != example_id:
        raise ValueError(f"target example_id mismatch: {example_id}")
    if row.get("family") not in {"ERE", "CPS"}:
        raise ValueError(f"target family missing or invalid: {example_id}")

    objects = row.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not 1 <= len(objects) <= MAX_SEMANTIC_OBJECTS:
        raise ValueError(f"target objects must contain 1..{MAX_SEMANTIC_OBJECTS} rows: {example_id}")
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping) or obj.get("slot") != index:
            raise ValueError(f"target object slot order is invalid: {example_id}")
        label_index = _label_index(obj.get("raw_label_index"), f"objects[{index}].raw_label_index", allow_missing=True)
        if type(obj.get("kind")) is not str or type(obj.get("content")) is not bool:
            raise ValueError(f"target object kind/content metadata is invalid: {example_id}")
        if (row.get("family") == "ERE" or obj.get("kind") == "cps_decision") and label_index != -1:
            raise ValueError(f"ERE/decision objects must not carry answer labels: {example_id}:{index}")
        start = _int(obj.get("token_start"), f"objects[{index}].token_start", minimum=-1)
        end = _int(obj.get("token_end"), f"objects[{index}].token_end", minimum=-1)
        if (start, end) != (-1, -1) and not start < end:
            raise ValueError(f"target object token bounds are invalid: {example_id}")

    presence = _vector(row.get("presence"), "presence", MAX_CHOICES)
    expected_presence = [1 if index < min(len(objects), MAX_CHOICES) else 0 for index in range(MAX_CHOICES)]
    if presence != expected_presence:
        raise ValueError(f"target presence does not match objects: {example_id}")
    content_mask = _vector(row.get("content_mask"), "content_mask", MAX_CHOICES)
    if any(type(value) is not int or value not in {0, 1} for value in presence + content_mask):
        raise ValueError(f"presence/content masks must be binary integers: {example_id}")
    expected_content = [int(bool(obj.get("content"))) for obj in objects[:MAX_CHOICES]]
    expected_content += [0] * (MAX_CHOICES - len(expected_content))
    state_values = _vector(row.get("state_values"), "state_values", MODEL_STEPS)
    state_feature_mask = _vector(row.get("state_feature_mask"), "state_feature_mask", MODEL_STEPS)
    for step, (values, masks) in enumerate(zip(state_values, state_feature_mask)):
        value_rows = _vector(values, f"state_values[{step}]", MAX_CHOICES)
        mask_rows = _vector(masks, f"state_feature_mask[{step}]", MAX_CHOICES)
        for slot, (value, mask) in enumerate(zip(value_rows, mask_rows)):
            value = _vector(value, f"state_values[{step}][{slot}]", STATE_FEATURES)
            mask = _vector(mask, f"state_feature_mask[{step}][{slot}]", STATE_FEATURES)
            if any(type(item) is not int or item not in {0, 1} for item in value + mask):
                raise ValueError(f"temporal target features must be binary integers: {example_id}")

    for key in ("operation_active", "source_owner", "target_owner"):
        _vector(row.get(key), key, MODEL_STEPS)
    active = [_int(value, f"operation_active[{index}]", minimum=0, maximum=1) for index, value in enumerate(row["operation_active"])]
    source_owner = [_int(value, f"source_owner[{index}]") for index, value in enumerate(row["source_owner"])]
    target_owner = [_int(value, f"target_owner[{index}]") for index, value in enumerate(row["target_owner"])]
    for index, (is_active, source, target) in enumerate(zip(active, source_owner, target_owner)):
        if is_active:
            if not 0 <= source < len(objects) or not 0 <= target < len(objects):
                raise ValueError(f"active operation owner is out of range at {example_id}:{index}")
        elif source != -1 or target != -1:
            raise ValueError(f"inactive operation must have -1 owners at {example_id}:{index}")
    schedule = _vector(row.get("schedule"), "schedule", MODEL_STEPS)
    for index, item in enumerate(schedule):
        if not isinstance(item, Mapping):
            raise ValueError(f"schedule row is not a mapping: {example_id}:{index}")
        if bool(item.get("active")) != bool(active[index]) or _int(item.get("source_slot"), "schedule.source_slot") != source_owner[index] or _int(item.get("target_slot"), "schedule.target_slot") != target_owner[index]:
            raise ValueError(f"schedule disagrees with owner tensors: {example_id}:{index}")

    query = row.get("query")
    if not isinstance(query, Mapping):
        raise ValueError(f"target query is missing: {example_id}")
    query_owner = _int(row.get("query_owner"), "query_owner", minimum=0, maximum=len(objects) - 1)
    if _int(query.get("owner_slot"), "query.owner_slot", minimum=0, maximum=len(objects) - 1) != query_owner or _int(query.get("query_owner"), "query.query_owner", minimum=0, maximum=len(objects) - 1) != query_owner:
        raise ValueError(f"query owner fields disagree: {example_id}")
    alternate_owner = _int(row.get("alternate_owner"), "alternate_owner", minimum=-1, maximum=len(objects) - 1)
    if _int(query.get("alternate_owner"), "query.alternate_owner", minimum=-1, maximum=len(objects) - 1) != alternate_owner:
        raise ValueError(f"alternate owner fields disagree: {example_id}")
    alternate_label = _label_index(row.get("alternate_label"), "alternate_label", allow_missing=True)
    if _label_index(query.get("alternate_label"), "query.alternate_label", allow_missing=True) != alternate_label:
        raise ValueError(f"alternate label fields disagree: {example_id}")
    irrelevant_owner = _int(row.get("irrelevant_owner"), "irrelevant_owner", minimum=-1, maximum=MAX_CHOICES - 1)
    if _int(query.get("irrelevant_owner"), "query.irrelevant_owner", minimum=-1, maximum=MAX_CHOICES - 1) != irrelevant_owner:
        raise ValueError(f"irrelevant owner fields disagree: {example_id}")
    if irrelevant_owner >= 0 and irrelevant_owner < len(objects):
        raise ValueError(f"irrelevant owner must name a padding slot, not a semantic object: {example_id}")

    mechanism_supported = row.get("mechanism_supported")
    query_answer_independent = row.get("query_answer_independent")
    if type(mechanism_supported) is not bool or type(query_answer_independent) is not bool:
        raise ValueError(f"target mechanism/query contract flags are invalid: {example_id}")
    if mechanism_supported and (len(objects) > MAX_CHOICES or not query_answer_independent):
        raise ValueError(f"supported mechanism exceeds capacity or has answer-dependent query: {example_id}")
    if not mechanism_supported and not str(row.get("split", "")).endswith("ood"):
        raise ValueError(f"behavior-only mechanism rows must be OOD: {example_id}")
    if content_mask != (expected_content if mechanism_supported else [0] * MAX_CHOICES):
        raise ValueError(f"target content_mask disagrees with semantic capacity boundary: {example_id}")
    state_step_mask = _vector(row.get("state_step_mask"), "state_step_mask", MODEL_STEPS)
    if any(type(value) is not int or value not in {0, 1} for value in state_step_mask):
        raise ValueError(f"state_step_mask must be binary: {example_id}")
    if not mechanism_supported and (any(active) or any(state_step_mask)):
        raise ValueError(f"behavior-only rows must disable mechanism supervision: {example_id}")
    state_feature_names = _vector(row.get("state_feature_names"), "state_feature_names", STATE_FEATURES)
    if any(type(value) is not str or not value for value in state_feature_names):
        raise ValueError(f"state_feature_names must contain {STATE_FEATURES} nonempty names: {example_id}")
    if mechanism_supported and len(set(state_feature_names)) != STATE_FEATURES:
        raise ValueError(f"supported state_feature_names must be unique: {example_id}")

    answer_target = row.get("answer_target")
    if not isinstance(answer_target, Mapping):
        raise ValueError(f"answer target is missing: {example_id}")
    answer_index = _label_index(answer_target.get("answer_index"), "answer_target.answer_index")
    if _label_index(row.get("answer_index"), "answer_index") != answer_index:
        raise ValueError(f"answer index fields disagree: {example_id}")
    return {
        "row": row,
        "objects": list(objects),
        "active": active,
        "source_owner": source_owner,
        "target_owner": target_owner,
        "query_owner": query_owner,
        "alternate_owner": alternate_owner,
        "alternate_label": alternate_label,
        "irrelevant_owner": irrelevant_owner,
        "answer_index": answer_index,
        "mechanism_supported": mechanism_supported,
        "query_answer_independent": query_answer_independent,
        "state_feature_names": state_feature_names,
    }


def _collated_row(parsed: Mapping[str, Any], slots: int, source_tokens: int) -> dict[str, list[Any]]:
    row = parsed["row"]
    objects = parsed["objects"]
    supported = bool(parsed["mechanism_supported"])
    if slots == 1:
        keep = [parsed["query_owner"]]
    elif supported:
        keep = list(range(len(objects)))
    else:
        keep = list(range(min(len(objects), MAX_CHOICES)))
    if any(index >= len(objects) for index in keep):
        raise ValueError("query owner is not present in target objects")
    index_map = {old: new for new, old in enumerate(keep)}
    object_labels = [-1] * slots
    presence = [0] * slots
    content_mask = [0] * slots
    span_start = [0] * slots
    span_end = [0] * slots
    object_kinds = ["padding"] * slots
    for new, old in enumerate(keep):
        obj = objects[old]
        object_labels[new] = _label_index(obj["raw_label_index"], "object.raw_label_index", allow_missing=True)
        presence[new] = 1
        content_mask[new] = int(bool(obj["content"])) if supported else 0
        object_kinds[new] = str(obj["kind"])
        start = _int(obj.get("token_start"), "object.token_start", minimum=-1)
        end = _int(obj.get("token_end"), "object.token_end", minimum=-1)
        if start < 0 or end < 0:
            raise ValueError("present target object lacks token bounds")
        if end > source_tokens:
            raise ValueError("target object token bound exceeds cached source length")
        span_start[new], span_end[new] = start, end

    source_owner: list[int] = []
    target_owner: list[int] = []
    for is_active, source, target in zip(parsed["active"], parsed["source_owner"], parsed["target_owner"]):
        if not is_active or not supported:
            source_owner.append(-1)
            target_owner.append(-1)
        elif slots == 1:
            source_owner.append(0)
            target_owner.append(0)
        else:
            if source not in index_map or target not in index_map:
                raise ValueError("active K8 operation owner is absent from collated slots")
            source_owner.append(index_map[source])
            target_owner.append(index_map[target])

    if slots == 1 or not supported:
        query_owner, alternate_owner, alternate_label, irrelevant_owner = 0, -1, -1, -1
    else:
        query_owner = index_map[parsed["query_owner"]]
        alternate_owner = index_map.get(parsed["alternate_owner"], -1)
        alternate_label = parsed["alternate_label"] if alternate_owner >= 0 else -1
        irrelevant_owner = int(parsed["irrelevant_owner"])

    state_values: list[list[list[int]]] = []
    state_masks: list[list[list[int]]] = []
    for step in range(MODEL_STEPS):
        value_row = [[0] * STATE_FEATURES for _ in range(slots)]
        mask_row = [[0] * STATE_FEATURES for _ in range(slots)]
        if supported:
            for new, old in enumerate(keep):
                value_row[new] = [int(value) for value in row["state_values"][step][old]]
                mask_row[new] = [int(value) for value in row["state_feature_mask"][step][old]]
        state_values.append(value_row)
        state_masks.append(mask_row)

    return {
        "answers": [parsed["answer_index"]],
        "object_labels": object_labels,
        "presence": presence,
        "content_mask": content_mask,
        "source_owner": source_owner,
        "target_owner": target_owner,
        "operation_active": list(parsed["active"]) if supported else [0] * MODEL_STEPS,
        "query_owner": [query_owner],
        "alternate_owner": [alternate_owner],
        "alternate_label": [alternate_label],
        "irrelevant_owner": [irrelevant_owner],
        "span_start": span_start,
        "span_end": span_end,
        "state_values": state_values,
        "state_feature_mask": state_masks,
        "state_step_mask": list(row["state_step_mask"]) if supported else [0] * MODEL_STEPS,
        "mechanism_supported": [supported],
        "query_answer_independent": [bool(parsed["query_answer_independent"])],
        "object_kinds": [object_kinds],
        "state_feature_names": [list(parsed["state_feature_names"])],
        "semantic_object_counts": [len(objects)],
        "families": [row["family"]],
        "splits": [row.get("split")],
        "pair_ids": [row.get("pair_id")],
        "pair_roles": [row.get("pair_role")],
    }


def _pin(value: Tensor) -> Tensor:
    return value.pin_memory() if not value.is_pinned() else value


def collate_target_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    slots: int,
    source_lengths: Sequence[int],
    pin_memory: bool = False,
) -> dict[str, Any]:
    """Validate and collate target rows without exposing them to ``forward``."""
    slots = _require_slots(slots)
    if len(rows) != len(source_lengths) or not rows:
        raise ValueError("rows and source_lengths must be non-empty and have equal length")
    parsed = [_validate_target_row(row, str(row.get("example_id", ""))) for row in rows]
    collated = [_collated_row(item, slots, int(source_lengths[index])) for index, item in enumerate(parsed)]
    tensor_keys = (
        "answers", "object_labels", "presence", "content_mask", "source_owner", "target_owner",
        "operation_active", "query_owner", "alternate_owner", "alternate_label", "irrelevant_owner",
        "span_start", "span_end", "state_values", "state_feature_mask", "state_step_mask",
        "mechanism_supported", "query_answer_independent",
    )
    result: dict[str, Any] = {}
    for key in tensor_keys:
        values = [item[key] for item in collated]
        if key in {"answers", "query_owner", "alternate_owner", "alternate_label", "irrelevant_owner", "mechanism_supported", "query_answer_independent"}:
            values = [value[0] for value in values]
        result[key] = torch.tensor(values, dtype=torch.bool if key in {"presence", "content_mask", "operation_active", "state_feature_mask", "state_step_mask", "mechanism_supported", "query_answer_independent"} else torch.long)
        if pin_memory:
            result[key] = _pin(result[key])
    for key in (
        "families", "splits", "pair_ids", "pair_roles", "object_kinds",
        "state_feature_names", "semantic_object_counts",
    ):
        result[key] = [item[key][0] for item in collated]
    for key, value in result.items():
        if isinstance(value, Tensor) and value.ndim == 0:
            raise ValueError(f"collated target tensor unexpectedly scalar: {key}")
    return result


def forward_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Return the only mapping that may be passed to the deployment model."""
    if not isinstance(batch, Mapping):
        raise TypeError("forward batch must be a mapping")
    keys = set(batch)
    forbidden = sorted(keys & FORWARD_FORBIDDEN_FIELDS)
    if forbidden:
        raise ValueError(f"forbidden fields cannot enter model forward: {forbidden}")
    if keys != SOURCE_KEYS:
        missing = sorted(SOURCE_KEYS - keys)
        extras = sorted(keys - SOURCE_KEYS)
        raise ValueError(f"model forward requires exactly source_hidden/source_mask; missing={missing}, extras={extras}")
    _validate_source_inputs(batch, [str(index) for index in range(batch["source_hidden"].shape[0])])
    return {"source_hidden": batch["source_hidden"], "source_mask": batch["source_mask"]}


@dataclass(frozen=True)
class RuntimeTelemetry:
    event: str
    index: int
    pending: int


def ordered_prefetch(
    items: Iterable[Any],
    loader: Callable[[Any], Any],
    *,
    max_prefetch: int = 0,
    workers: int = 1,
    telemetry: Callable[[RuntimeTelemetry], None] | None = None,
) -> Iterator[Any]:
    """Yield loader results in input order with bounded optional threads.

    The default ``max_prefetch=0`` is a deterministic synchronous path.  The
    worker count is local to this iterator; no process-wide torch/OpenMP
    setting is changed, which keeps Windows execution predictable.
    """
    if max_prefetch < 0 or workers < 1:
        raise ValueError("max_prefetch must be >=0 and workers must be >=1")
    iterator = iter(items)
    if max_prefetch == 0:
        for index, item in enumerate(iterator):
            if telemetry:
                telemetry(RuntimeTelemetry("load_start", index, 0))
            result = loader(item)
            if telemetry:
                telemetry(RuntimeTelemetry("load_done", index, 0))
            yield result
        return
    limit = max_prefetch
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="c1s-successor-prefetch") as pool:
        pending: deque[tuple[int, Any]] = deque()
        next_index = 0
        while len(pending) < limit:
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append((next_index, pool.submit(loader, item)))
            if telemetry:
                telemetry(RuntimeTelemetry("submit", next_index, len(pending)))
            next_index += 1
        while pending:
            index, future = pending.popleft()
            result = future.result()
            if telemetry:
                telemetry(RuntimeTelemetry("load_done", index, len(pending)))
            yield result
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append((next_index, pool.submit(loader, item)))
            if telemetry:
                telemetry(RuntimeTelemetry("submit", next_index, len(pending)))
            next_index += 1


class C1SSuccessorRuntime:
    """Caller-ordered source cache + offline target-bank provider."""

    def __init__(self, dataset: CachedShardDataset, target_bank: Mapping[str, Any]) -> None:
        if not hasattr(dataset, "get_batch"):
            raise TypeError("dataset must expose the C1 immutable get_batch API")
        self.dataset = dataset
        self.targets = _lookup_targets(target_bank)

    def get_batch(
        self,
        example_ids: Sequence[str],
        *,
        slots: int,
        pin_memory: bool = False,
    ) -> dict[str, Any]:
        """Return objective tensors and audit metadata in caller order."""
        slots = _require_slots(slots)
        ids = [str(value) for value in example_ids]
        if not ids:
            raise ValueError("successor batch cannot be empty")
        if pin_memory and not torch.cuda.is_available():
            raise RuntimeError("pin_memory=True requires an available CUDA allocator")
        source, _ = self.dataset.get_batch(ids)
        hidden, mask = _validate_source_inputs(source, ids)
        rows: list[Mapping[str, Any]] = []
        for example_id in ids:
            row = self.targets.get(example_id)
            if row is None:
                raise KeyError(f"successor target not found: {example_id}")
            rows.append(row)
        result = collate_target_rows(rows, slots=slots, source_lengths=[int(value) for value in mask.sum(dim=1).tolist()], pin_memory=pin_memory)
        result["source_hidden"] = _pin(hidden) if pin_memory else hidden
        result["source_mask"] = _pin(mask) if pin_memory else mask
        result["example_ids"] = ids
        return result

    __call__ = get_batch

    def iter_batches(
        self,
        example_ids: Sequence[str],
        batch_size: int,
        slots: int,
        pin_memory: bool = False,
        *,
        max_prefetch: int = 0,
        workers: int = 1,
        telemetry: Callable[[RuntimeTelemetry], None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        ids = [str(value) for value in example_ids]
        chunks = [ids[start : start + batch_size] for start in range(0, len(ids), batch_size)]
        yield from ordered_prefetch(
            chunks,
            lambda chunk: self.get_batch(chunk, slots=slots, pin_memory=pin_memory),
            max_prefetch=max_prefetch,
            workers=workers,
            telemetry=telemetry,
        )


SuccessorRuntime = C1SSuccessorRuntime
C1SSuccessorBatchProvider = C1SSuccessorRuntime


__all__ = [
    "C1SSuccessorBatchProvider",
    "C1SSuccessorRuntime",
    "FORWARD_FORBIDDEN_FIELDS",
    "RUNTIME_SCHEMA",
    "RuntimeTelemetry",
    "SOURCE_KEYS",
    "SuccessorRuntime",
    "collate_target_rows",
    "forward_inputs",
    "ordered_prefetch",
]
