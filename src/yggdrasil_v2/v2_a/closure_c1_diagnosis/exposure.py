from __future__ import annotations

"""Read-only reconstruction of the C1 trace exposure schedule.

The formal ledger records windows, rather than the set of target positions
which those windows touched.  This module expands those windows in memory and
reports coverage by family, grammar/content, step, and global position.  It
does not read a dataset and never writes an artifact.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import contract


EXPOSURE_SCHEMA = f"{contract.SCHEMA_PREFIX}.exposure-coverage.v1"
_FAMILIES = ("ERE", "CPS")


def _input(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} must be a JSON object or readable JSON path") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or JSON path")
    return value


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _family(row: Mapping[str, Any], example_id: str) -> str:
    value = row.get("family")
    if value is None:
        prefix = example_id.lower().split("-", 1)[0]
        value = {"ere": "ERE", "cps": "CPS"}.get(prefix)
    if value not in _FAMILIES:
        raise ValueError(f"cannot determine ERE/CPS family for {example_id!r}")
    return str(value)


def _target_rows(bank: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = bank.get("targets")
    if isinstance(raw, Mapping):
        rows = {str(key): value for key, value in raw.items()}
    elif isinstance(raw, list):
        rows = {}
        for row in raw:
            if not isinstance(row, Mapping) or type(row.get("example_id")) is not str:
                raise ValueError("target bank list rows require string example_id")
            key = str(row["example_id"])
            if key in rows:
                raise ValueError(f"duplicate target example_id: {key}")
            rows[key] = row
    else:
        raise ValueError("target bank requires a targets mapping or list")
    if not rows:
        raise ValueError("target bank has no targets")
    result: dict[str, Mapping[str, Any]] = {}
    for key, row in rows.items():
        if not isinstance(row, Mapping) or row.get("example_id", key) != key:
            raise ValueError(f"target bank identity mismatch for {key!r}")
        ids = row.get("global_token_ids", row.get("token_ids"))
        steps = row.get("step_indices", row.get("step_ids"))
        grammar = row.get("grammar_mask")
        positions = row.get("global_positions")
        locals_ = row.get("local_positions")
        if not all(isinstance(seq, list) for seq in (ids, steps, grammar, positions, locals_)):
            raise ValueError(f"target {key!r} has incomplete token arrays")
        length = len(ids)
        if length == 0 or any(len(seq) != length for seq in (steps, grammar, positions, locals_)):
            raise ValueError(f"target {key!r} token arrays have inconsistent lengths")
        for index, (token_id, step, is_grammar, position, local) in enumerate(
            zip(ids, steps, grammar, positions, locals_)
        ):
            _integer(token_id, f"{key}.token_ids[{index}]", minimum=0)
            _integer(step, f"{key}.step_indices[{index}]", minimum=1)
            if type(is_grammar) is not bool:
                raise ValueError(f"{key}.grammar_mask[{index}] must be bool")
            _integer(position, f"{key}.global_positions[{index}]", minimum=0)
            _integer(local, f"{key}.local_positions[{index}]", minimum=0)
        if positions != list(range(length)):
            raise ValueError(f"{key}.global_positions must be contiguous from zero")
        if any(step > 10 for step in steps):
            raise ValueError(f"{key}.step_indices exceed H10")
        result[key] = row
    return result


def _ledger_rows(ledger: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = ledger.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("exposure ledger requires a non-empty rows list")
    result: list[Mapping[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or type(row.get("example_id")) is not str:
            raise ValueError(f"ledger row {index} requires string example_id")
        example_id = str(row["example_id"])
        epoch = _integer(row.get("epoch"), f"ledger[{index}].epoch", minimum=0)
        key = (example_id, epoch)
        if key in seen:
            raise ValueError(f"duplicate ledger exposure for {example_id!r}, epoch {epoch}")
        seen.add(key)
        start = _integer(row.get("start"), f"ledger[{index}].start", minimum=0)
        stop = _integer(row.get("stop"), f"ledger[{index}].stop", minimum=0)
        if stop <= start:
            raise ValueError(f"ledger[{index}] stop must be greater than start")
        if type(row.get("exposed_tokens")) is not int or row["exposed_tokens"] != stop - start:
            raise ValueError(f"ledger[{index}] exposed_tokens disagrees with start/stop")
        if type(row.get("target_tokens")) is not int or row["target_tokens"] <= 0:
            raise ValueError(f"ledger[{index}] target_tokens must be positive integer")
        result.append(row)
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _coverage(total: int, exposed: int) -> dict[str, Any]:
    return {"tokens": int(total), "unique_exposed_tokens": int(exposed), "coverage": _ratio(exposed, total)}


def _dimension(targets: list[tuple[str, Mapping[str, Any], list[int]]], dimension: str) -> dict[str, Any]:
    # ``dimension`` is either step_indices or global_positions.  A position
    # report uses the frozen five bins from the diagnostic contract.
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    if dimension == "global_positions":
        for lo, hi, label in contract.POSITION_BINS:
            groups[label]  # preserve empty bins deterministically
        for _, target, counts in targets:
            for position, exposed_count in zip(target["global_positions"], counts):
                label = next((label for lo, hi, label in contract.POSITION_BINS if lo <= position < hi), None)
                if label is None:
                    raise ValueError(f"global position {position} is outside frozen bins")
                groups[label][0] += 1
                groups[label][1] += int(exposed_count > 0)
    else:
        for _, target, counts in targets:
            for key, exposed_count in zip(target[dimension], counts):
                groups[str(key)][0] += 1
                groups[str(key)][1] += int(exposed_count > 0)
    return {key: _coverage(*groups[key]) for key in sorted(groups)}


def _family_report(family: str, target_data: list[tuple[str, Mapping[str, Any], list[int]]]) -> dict[str, Any]:
    records = len(target_data)
    all_counts = [count for _, _, counts in target_data for count in counts]
    exposed = sum(count > 0 for count in all_counts)
    grammar_total = grammar_exposed = content_total = content_exposed = 0
    full: list[str] = []
    zero: list[dict[str, Any]] = []
    record_coverages: list[float] = []
    for example_id, target, counts in target_data:
        masks = target["grammar_mask"]
        record_exposed = sum(count > 0 for count in counts)
        record_coverages.append(_ratio(record_exposed, len(counts)))
        if record_exposed == len(counts):
            full.append(example_id)
        missing = [index for index, count in enumerate(counts) if count == 0]
        if missing:
            zero.append({"example_id": example_id, "positions": missing, "count": len(missing)})
        for is_grammar, count in zip(masks, counts):
            if is_grammar:
                grammar_total += 1
                grammar_exposed += count > 0
            else:
                content_total += 1
                content_exposed += count > 0
    histogram = Counter(all_counts)
    grammar = _coverage(grammar_total, grammar_exposed)
    content = _coverage(content_total, content_exposed)
    report = {
        "records": records,
        "tokens": len(all_counts),
        "unique_exposed_tokens": exposed,
        "micro_unique_coverage": _ratio(exposed, len(all_counts)),
        "macro_unique_coverage": _ratio(sum(record_coverages), records),
        "grammar": grammar,
        "content": content,
        "step": _dimension(target_data, "step_indices"),
        "global_position_bins": _dimension(target_data, "global_positions"),
        "exposure_count_histogram": {str(key): int(histogram[key]) for key in sorted(histogram)},
        "full_coverage_records": sorted(full),
        "full_coverage_record_count": len(full),
        "zero_exposure_tokens": zero,
        "zero_exposure_token_count": sum(item["count"] for item in zero),
    }
    # Short aliases make the machine report convenient without changing the
    # unambiguous canonical names above.
    report["micro"] = report["micro_unique_coverage"]
    report["macro"] = report["macro_unique_coverage"]
    return report


def exposure_coverage_report(target_bank: Any, ledger: Any) -> dict[str, Any]:
    """Reconstruct per-token exposure counts from a sealed target bank/ledger.

    Inputs may be already-loaded JSON mappings or paths to JSON files.  Only
    train records present in the ledger are included; validation tokens are
    never inferred or merged into the training exposure statistics.
    """
    bank = _input(target_bank, "target_bank")
    ledger_value = _input(ledger, "ledger")
    targets = _target_rows(bank)
    rows = _ledger_rows(ledger_value)
    counts: dict[str, list[int]] = {}
    family_by_id: dict[str, str] = {}
    for row in rows:
        example_id = str(row["example_id"])
        target = targets.get(example_id)
        if target is None:
            raise ValueError(f"ledger references target not in bank: {example_id}")
        family = _family(target, example_id)
        if row.get("family") is not None and row.get("family") != family:
            raise ValueError(f"ledger/target family mismatch for {example_id}")
        family_by_id[example_id] = family
        token_count = len(target["global_token_ids"])
        start, stop = int(row["start"]), int(row["stop"])
        if int(row["target_tokens"]) != token_count or stop > token_count:
            raise ValueError(f"ledger window is outside target {example_id}")
        vector = counts.setdefault(example_id, [0] * token_count)
        for position in range(start, stop):
            vector[position] += 1
    if not counts:
        raise ValueError("ledger contains no usable target records")
    by_family: dict[str, dict[str, Any]] = {}
    for family in _FAMILIES:
        data = [
            (example_id, targets[example_id], counts[example_id])
            for example_id in sorted(counts)
            if family_by_id[example_id] == family
        ]
        by_family[family] = _family_report(family, data)
    combined = [(example_id, targets[example_id], counts[example_id]) for example_id in sorted(counts)]
    overall = _family_report("ALL", combined)
    complete_cycle = all(count > 0 for _, _, vector in combined for count in vector)
    cycle_status = "PASS" if complete_cycle else "EXPOSURE_CYCLE_BREACH"
    report: dict[str, Any] = {
        "schema_version": EXPOSURE_SCHEMA,
        "source": {"target_records": len(targets), "ledger_rows": len(rows), "train_records": len(counts)},
        "complete_cycle": complete_cycle,
        "exposure_cycle_status": cycle_status,
        "decision": cycle_status,
        "EXPOSURE_CYCLE_BREACH": not complete_cycle,
        "by_family": by_family,
        "families": by_family,
        "overall": overall,
        "micro_unique_coverage": overall["micro_unique_coverage"],
        "macro_unique_coverage": overall["macro_unique_coverage"],
        "grammar": overall["grammar"],
        "content": overall["content"],
        "step": overall["step"],
        "global_position_bins": overall["global_position_bins"],
        "exposure_count_histogram": overall["exposure_count_histogram"],
        "full_coverage_records": overall["full_coverage_records"],
        "zero_exposure_tokens": overall["zero_exposure_tokens"],
        "zero_exposure_token_count": overall["zero_exposure_token_count"],
    }
    # JSON round-trip is an intentional final guard against bool/int/float
    # scalar leakage from callers supplying JSON-compatible subclasses.
    try:
        json.dumps(report, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("exposure report is not JSON-native") from exc
    return report


__all__ = ["EXPOSURE_SCHEMA", "exposure_coverage_report"]
