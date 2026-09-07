from __future__ import annotations

"""Read-only runtime assembly for the frozen C1S S1 endpoint.

The model-facing mapping is constructed explicitly from ``source_hidden`` and
``source_mask``.  Target-bank fields remain evaluator-side and are never
forwarded through :class:`C1SModel.forward`.
"""

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ..closure_c1.cache import CachedShardDataset
from ..closure_c1s_successor.runtime import C1SSuccessorRuntime, forward_inputs


def _json_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return value


def load_gzip_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return _json_object(json.load(handle), name=str(path))


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    return _json_object(json.loads(path.read_text(encoding="utf-8")), name=str(path))


def build_runtime(
    cache_root: Path,
    target_bank: Mapping[str, Any],
) -> C1SSuccessorRuntime:
    """Open the immutable mmap cache and bind it to an already-audited bank."""

    dataset = CachedShardDataset(Path(cache_root))
    return C1SSuccessorRuntime(dataset, target_bank)


def s1_ids(split_ledger: Mapping[str, Any]) -> list[str]:
    cell = split_ledger.get("s1")
    if not isinstance(cell, Mapping):
        raise ValueError("split ledger has no S1 cell")
    raw = cell.get("ids")
    if not isinstance(raw, list) or len(raw) != 32 or any(type(item) is not str for item in raw):
        raise ValueError("S1 ledger must contain exactly 32 string IDs")
    ids = list(raw)
    if len(set(ids)) != len(ids):
        raise ValueError("S1 ledger contains duplicate IDs")
    return ids


def bank_rows(target_bank: Mapping[str, Any], example_ids: Sequence[str]) -> list[dict[str, Any]]:
    records = target_bank.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("target bank records must be an object")
    rows: list[dict[str, Any]] = []
    for example_id in example_ids:
        row = records.get(str(example_id))
        if not isinstance(row, Mapping):
            raise KeyError(f"target-bank row absent: {example_id}")
        rows.append(dict(row))
    return rows


def chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    if type(size) is not int or size < 1:
        raise ValueError("batch size must be a positive integer")
    rows = [str(value) for value in values]
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            result[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            result[key] = value
    return result


def model_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Return the exact two-key deployment boundary and re-run its validator."""

    return forward_inputs(
        {
            "source_hidden": batch["source_hidden"],
            "source_mask": batch["source_mask"],
        }
    )


__all__ = [
    "bank_rows",
    "build_runtime",
    "chunks",
    "load_gzip_json",
    "load_json",
    "model_inputs",
    "move_batch",
    "s1_ids",
]
