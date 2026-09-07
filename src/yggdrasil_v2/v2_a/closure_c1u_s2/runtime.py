from __future__ import annotations

"""Multi-bank runtime and bounded caller-order batch cache for C1U S2."""

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from yggdrasil_v2.v2_a.closure_c1u.runtime import (
    C1URuntime,
    IndependentCardCache,
)

from . import contract


class S2Runtime(C1URuntime):
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        cache: IndependentCardCache,
        *,
        address_width: int = 64,
    ) -> None:
        super().__init__(records, cache, address_width=address_width)
        self.s2_records = {str(row["example_id"]): dict(row) for row in records}

    def get_batch(self, example_ids: Sequence[str]) -> dict[str, Any]:
        result = super().get_batch(example_ids)
        rows = [self.s2_records[str(example_id)] for example_id in example_ids]
        result.update(
            {
                "bank_ids": [str(row["bank_id"]) for row in rows],
                "bank_seeds": [int(row["bank_seed"]) for row in rows],
                "label_rotations": [int(row["label_rotation"]) for row in rows],
            }
        )
        return result


class BoundedBatchProvider:
    """LRU cache which cannot retain all 16x16 multi-bank group pairs."""

    def __init__(
        self, runtime: S2Runtime, *, maximum_batches: int = contract.MAX_CACHE_BATCHES
    ) -> None:
        if type(maximum_batches) is not int or maximum_batches < 1:
            raise ValueError("maximum_batches must be a positive integer")
        self.runtime = runtime
        self.maximum_batches = maximum_batches
        self._batches: OrderedDict[tuple[str, ...], dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_batch(
        self, example_ids: Sequence[str], *, pin_memory: bool = False
    ) -> Mapping[str, Any]:
        key = tuple(str(value) for value in example_ids)
        if len(key) != int(contract.TRAINING["batch_size"]):
            raise ValueError("C1U S2 batches must contain exactly eight records")
        if key in self._batches:
            self.hits += 1
            self._batches.move_to_end(key)
            return self._batches[key]
        self.misses += 1
        batch = self.runtime.get_batch(key)
        if not bool((batch["counterfactual_indices"] >= 0).all()):
            raise ValueError("S2 batch split a registered factorial group")
        groups = set(str(value) for value in batch["factorial_group_ids"])
        families = set(str(value) for value in batch["families"])
        if len(groups) != 2 or families != {"CPS", "ERE"}:
            raise ValueError("S2 batch must contain one complete group per family")
        if pin_memory:
            batch = {
                name: value.pin_memory() if isinstance(value, Tensor) else value
                for name, value in batch.items()
            }
        self._batches[key] = dict(batch)
        self._batches.move_to_end(key)
        if len(self._batches) > self.maximum_batches:
            self._batches.popitem(last=False)
            self.evictions += 1
        return self._batches[key]

    def report(self) -> dict[str, Any]:
        return {
            "cached_batch_count": len(self._batches),
            "maximum_cached_batch_count": self.maximum_batches,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "bounded": len(self._batches) <= self.maximum_batches,
        }


__all__ = ["BoundedBatchProvider", "S2Runtime"]
