from __future__ import annotations

"""Deterministic complete-target block coverage for the C1R staged trainer.

This module is deliberately independent of model execution.  A schedule row
contains one record per optimizer step; its complete list of 65-token blocks
is consumed and token-normalized as one trace loss.  That keeps block coverage
from changing the answer example balance or the optimizer-step budget.
"""

from collections import Counter
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from . import contract


COVERAGE_SCHEMA = f"{contract.SCHEMA_PREFIX}.coverage.v1"
BLOCK_TOKENS = int(contract.STAGE_B_CONFIG["block_tokens"])


def _family(example_id: str, value: Any) -> str:
    family = value
    if family is None:
        family = {"ere": "ERE", "cps": "CPS"}.get(example_id.lower().split("-", 1)[0])
    if family not in contract.FAMILIES:
        raise ValueError(f"cannot determine C1R family for {example_id!r}")
    return str(family)


def _target_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("targets"), (Mapping, list)):
        value = value["targets"]
    if isinstance(value, Mapping):
        raw = []
        for key, row in value.items():
            if not isinstance(row, Mapping):
                raise TypeError(f"target {key!r} must be an object")
            item = dict(row)
            item.setdefault("example_id", str(key))
            raw.append(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        raw = list(value)
    else:
        raise TypeError("targets must be a mapping, target-bank object, or sequence")

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping) or type(row.get("example_id")) is not str:
            raise ValueError(f"target[{index}] requires string example_id")
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"duplicate target example_id: {example_id}")
        seen.add(example_id)
        ids = row.get("token_ids", row.get("global_token_ids"))
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
            raise ValueError(f"target {example_id!r} requires token_ids or global_token_ids")
        token_count = len(ids)
        if token_count < 1:
            raise ValueError(f"target {example_id!r} is empty")
        arrays = {
            "step_indices": row.get("step_indices", row.get("step_ids")),
            "global_positions": row.get("global_positions"),
            "local_positions": row.get("local_positions"),
            "grammar_mask": row.get("grammar_mask"),
        }
        for field, array in arrays.items():
            if array is not None and len(array) != token_count:
                raise ValueError(f"target {example_id!r} {field} length mismatch")
        result.append(
            {
                "example_id": example_id,
                "family": _family(example_id, row.get("family")),
                "target_tokens": token_count,
            }
        )
    if not result:
        raise ValueError("targets must not be empty")
    return result


def partition_target(
    example_id: str, target_tokens: int, *, block_tokens: int = BLOCK_TOKENS
) -> list[dict[str, int]]:
    """Return deterministic, non-overlapping blocks covering one target."""

    if type(example_id) is not str or not example_id:
        raise ValueError("example_id must be a non-empty string")
    if type(target_tokens) is not int or target_tokens < 1:
        raise ValueError("target_tokens must be a positive integer")
    if type(block_tokens) is not int or block_tokens < 1:
        raise ValueError("block_tokens must be a positive integer")
    blocks: list[dict[str, int]] = []
    for block_index, start in enumerate(range(0, target_tokens, block_tokens)):
        stop = min(target_tokens, start + block_tokens)
        blocks.append(
            {
                "block_index": block_index,
                "start": start,
                "stop": stop,
                "valid_tokens": stop - start,
            }
        )
    if blocks[-1]["stop"] != target_tokens:
        raise AssertionError("block partition does not cover target")
    return blocks


def _schedule_hash(schedule: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(list(schedule), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def build_coverage_plan(
    targets: Any,
    *,
    epochs: int = int(contract.STAGE_B_CONFIG["epochs"]),
    batch_size: int = int(contract.STAGE_B_CONFIG["batch_size"]),
    family_batch_size: int = int(contract.STAGE_B_CONFIG["family_batch_size"]),
    block_tokens: int = BLOCK_TOKENS,
    order_seed: int = contract.ORDER_SEED,
) -> dict[str, Any]:
    """Build a balanced schedule and complete block plan without model work."""

    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(batch_size) is not int or type(family_batch_size) is not int:
        raise ValueError("batch sizes must be integers")
    if batch_size != 2 * family_batch_size:
        raise ValueError("C1R batch must contain equal ERE/CPS halves")
    rows = _target_rows(targets)
    by_family: dict[str, list[str]] = {family: [] for family in contract.FAMILIES}
    targets_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        targets_by_id[row["example_id"]] = row
        by_family[row["family"]].append(row["example_id"])
    for family in contract.FAMILIES:
        by_family[family].sort()
        if len(by_family[family]) % family_batch_size:
            raise ValueError(f"{family} record count must divide family_batch_size")

    schedule: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(int(order_seed))
    update = 0
    for epoch in range(epochs):
        shuffled: dict[str, list[str]] = {}
        for family in contract.FAMILIES:
            rows_for_family = by_family[family]
            shuffled[family] = [
                rows_for_family[index]
                for index in torch.randperm(len(rows_for_family), generator=generator).tolist()
            ]
        updates = len(shuffled[contract.FAMILIES[0]]) // family_batch_size
        for batch_index in range(updates):
            update += 1
            start = batch_index * family_batch_size
            stop = start + family_batch_size
            ids = shuffled["ERE"][start:stop] + shuffled["CPS"][start:stop]
            schedule.append(
                {
                    "update": update,
                    "epoch": epoch,
                    "example_ids": ids,
                }
            )

    blocks = {
        row["example_id"]: partition_target(
            row["example_id"], row["target_tokens"], block_tokens=block_tokens
        )
        for row in rows
    }
    return {
        "schema_version": COVERAGE_SCHEMA,
        "block_tokens": block_tokens,
        "block_overlap": 0,
        "epochs": epochs,
        "batch_size": batch_size,
        "family_batch_size": family_batch_size,
        "updates_per_epoch": len(schedule) // epochs,
        "updates": len(schedule),
        "blocks_aggregated_per_record": True,
        "targets": targets_by_id,
        "blocks": blocks,
        "schedule": schedule,
        "schedule_sha256": _schedule_hash(schedule),
    }


def coverage_report(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Audit exact per-token exposure and family-balanced schedule invariants."""

    if not isinstance(plan, Mapping):
        raise TypeError("coverage plan must be a mapping")
    targets = plan.get("targets")
    blocks = plan.get("blocks")
    schedule = plan.get("schedule")
    if not isinstance(targets, Mapping) or not isinstance(blocks, Mapping) or not isinstance(schedule, list):
        raise ValueError("coverage plan is missing targets, blocks, or schedule")
    epochs = int(plan.get("epochs", 0))
    block_tokens = int(plan.get("block_tokens", 0))
    if plan.get("schema_version") != COVERAGE_SCHEMA:
        raise ValueError("coverage plan has the wrong C1R schema")
    if epochs < 1 or block_tokens < 1:
        raise ValueError("coverage plan has invalid epochs or block_tokens")
    expected_ids = set(str(key) for key in targets)
    if set(str(key) for key in blocks) != expected_ids:
        raise ValueError("coverage blocks must have exactly one entry per target")
    for example_id, target in targets.items():
        if not isinstance(target, Mapping) or type(target.get("target_tokens")) is not int:
            raise ValueError(f"target {example_id!r} has invalid target_tokens")
        target_tokens = int(target["target_tokens"])
        if target_tokens < 1 or not isinstance(blocks[example_id], list):
            raise ValueError(f"target {example_id!r} has invalid blocks")
        cursor = 0
        for index, block in enumerate(blocks[example_id]):
            if not isinstance(block, Mapping):
                raise ValueError(f"target {example_id!r} block is not an object")
            if any(type(block.get(field)) is not int for field in ("block_index", "start", "stop", "valid_tokens")):
                raise ValueError(f"target {example_id!r} block has non-integer bounds")
            start, stop = int(block["start"]), int(block["stop"])
            expected_stop = min(target_tokens, cursor + block_tokens)
            if block["block_index"] != index or start != cursor or stop != expected_stop:
                raise ValueError(f"target {example_id!r} blocks are overlapping or incomplete")
            if block["valid_tokens"] != stop - start or stop <= start:
                raise ValueError(f"target {example_id!r} block has invalid valid_tokens")
            cursor = stop
        if cursor != target_tokens:
            raise ValueError(f"target {example_id!r} blocks do not cover full target")
    exposure: Counter[tuple[str, int]] = Counter()
    schedule_checks = {
        "schema": True,
        "rows": True,
        "blocks_aggregated_per_record": plan.get("blocks_aggregated_per_record") is True,
        "one_record_per_epoch": True,
        "balanced_batch": True,
        "schedule_hash": _schedule_hash(schedule) == plan.get("schedule_sha256"),
    }
    epoch_seen: dict[int, list[str]] = {}
    expected_family_batch = int(plan["family_batch_size"])
    expected_batch = int(plan["batch_size"])
    for expected_update, row in enumerate(schedule, start=1):
        if not isinstance(row, Mapping) or type(row.get("epoch")) is not int or not isinstance(row.get("example_ids"), list):
            raise ValueError("schedule rows require integer epoch and example_ids list")
        epoch = int(row["epoch"])
        ids = [str(value) for value in row["example_ids"]]
        schedule_checks["rows"] &= row.get("update") == expected_update
        schedule_checks["rows"] &= 0 <= epoch < epochs
        schedule_checks["balanced_batch"] &= len(ids) == expected_batch
        schedule_checks["balanced_batch"] &= sum(_family(example_id, targets[example_id].get("family")) == "ERE" for example_id in ids if example_id in targets) == expected_family_batch
        if any(example_id not in expected_ids for example_id in ids):
            raise ValueError("schedule references unknown target")
        epoch_seen.setdefault(epoch, []).extend(ids)
        for example_id in ids:
            for block in blocks[example_id]:
                start, stop = int(block["start"]), int(block["stop"])
                for position in range(start, stop):
                    exposure[(example_id, position)] += 1
    for epoch in range(epochs):
        values = epoch_seen.get(epoch, [])
        schedule_checks["one_record_per_epoch"] &= len(values) == len(expected_ids) and len(set(values)) == len(expected_ids)
    expected_exposure = epochs
    zero_exposure = [key for key in exposure if exposure[key] == 0]
    all_positions = [
        (example_id, position)
        for example_id, target in targets.items()
        for position in range(int(target["target_tokens"]))
    ]
    zero_exposure = [key for key in all_positions if exposure[key] == 0]
    counts = [exposure[key] for key in all_positions]
    by_family: dict[str, dict[str, Any]] = {}
    for family in contract.FAMILIES:
        family_positions = [key for key in all_positions if _family(key[0], targets[key[0]].get("family")) == family]
        family_counts = [exposure[key] for key in family_positions]
        family_blocks = sum(len(blocks[example_id]) for example_id in expected_ids if _family(example_id, targets[example_id].get("family")) == family)
        by_family[family] = {
            "records": sum(1 for example_id in expected_ids if _family(example_id, targets[example_id].get("family")) == family),
            "target_tokens": len(family_positions),
            "blocks": family_blocks,
            "min_exposure": min(family_counts) if family_counts else 0,
            "max_exposure": max(family_counts) if family_counts else 0,
            "zero_exposure_tokens": sum(value == 0 for value in family_counts),
        }
    complete = bool(all_positions) and not zero_exposure and all(value == expected_exposure for value in counts)
    return {
        "schema_version": COVERAGE_SCHEMA,
        "block_tokens": block_tokens,
        "epochs": epochs,
        "batch_size": int(plan["batch_size"]),
        "family_batch_size": int(plan["family_batch_size"]),
        "updates": len(schedule),
        "records": len(expected_ids),
        "raw_target_tokens": len(all_positions),
        "blocks": sum(len(value) for value in blocks.values()),
        "schedule_checks": schedule_checks,
        "exact_exposure_per_token": expected_exposure,
        "min_exposure": min(counts) if counts else 0,
        "max_exposure": max(counts) if counts else 0,
        "zero_exposure_tokens": len(zero_exposure),
        "by_family": by_family,
        "complete_cycle": complete,
        "passed": complete and all(schedule_checks.values()),
    }


__all__ = [
    "BLOCK_TOKENS",
    "COVERAGE_SCHEMA",
    "build_coverage_plan",
    "coverage_report",
    "partition_target",
]
