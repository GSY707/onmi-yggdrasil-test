from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .a1_7_data import (
    HELDOUT_JOINT,
    REGISTER_NAMES,
    SUPPORTED_JOINTS,
    VALUE_LABELS,
    _all_steps_necessary,
    _balanced_values,
    _causal_split,
    _joint,
    _joint_name,
    _operation,
    _render_record,
    _split_stats,
    _stratified_operations,
    _write_split,
)


SCHEMA_VERSION = "yggdrasil.v2-a1.8.random-depth-relation-state-machine.v1"
GENERATOR_VERSION = "a1.8-random-depth-relation-state-machine-2026-07-15"
TRAIN_LENGTHS = tuple(range(1, 17))
SHORT_LENGTHS = tuple(range(1, 7))
IN_RANGE_LENGTHS = (8, 12, 16)
OOD_LENGTHS = (20, 24, 32)
DATA_SPLITS = (
    "train",
    "validation",
    "short_regression",
    "supported_in_range",
    "relation_in_range",
    "supported_ood",
    "relation_ood",
    "causal_core",
)
RELATION_SPLITS = ("relation_in_range", "relation_ood")
SUPPORTED_SPLITS = (
    "train",
    "validation",
    "short_regression",
    "supported_in_range",
    "supported_ood",
    "causal_core",
)


@dataclass(frozen=True)
class A18DatasetSpec:
    data_seed: int
    train_size: int = 16384
    validation_size: int = 2048
    short_size: int = 768
    in_range_size: int = 768
    relation_in_range_size: int = 768
    ood_size: int = 768
    relation_ood_size: int = 768
    causal_size: int = 512


def _external_fingerprints(data_dirs: Sequence[Path]) -> set[str]:
    fingerprints: set[str] = set()
    for data_dir in data_dirs:
        if not data_dir.exists():
            raise FileNotFoundError(f"external forbidden data directory does not exist: {data_dir}")
        for path in data_dir.glob("*.jsonl"):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        fingerprints.add(json.loads(line)["fingerprint"])
    return fingerprints


def _decorate(record: dict[str, Any]) -> dict[str, Any]:
    record["schema_version"] = SCHEMA_VERSION
    record["generator_version"] = GENERATOR_VERSION
    record["task_family"] = "a1.8_random_depth_relation_addressed_three_register_copy_swap"
    return record


def _ordinary_split(
    split: str,
    count: int,
    lengths: Sequence[int],
    rng: random.Random,
    forbidden: set[str],
) -> list[dict[str, Any]]:
    rows = _stratified_operations(lengths, rng)
    records: list[dict[str, Any]] = []
    for index, operations in enumerate(rows):
        for _ in range(5000):
            start_state = dict(zip(REGISTER_NAMES, rng.sample(VALUE_LABELS, len(REGISTER_NAMES))))
            record = _decorate(_render_record(start_state, operations, rng.choice(REGISTER_NAMES), split, index))
            if record["fingerprint"] in forbidden:
                continue
            records.append(record)
            forbidden.add(record["fingerprint"])
            break
        else:
            raise RuntimeError(f"could not create a unique A1.8 {split} record at index {index}")
    return records


def _relation_assignments(count: int, allowed_lengths: Sequence[int], rng: random.Random) -> list[tuple[int, int]]:
    lengths = _balanced_values(tuple(allowed_lengths), count, rng)
    length_counts = Counter(lengths)
    positions = {
        length: _balanced_values(tuple(range(length)), length_count, rng)
        for length, length_count in length_counts.items()
    }
    assignments = [(length, positions[length].pop()) for length in lengths]
    rng.shuffle(assignments)
    return assignments


def _relation_split(
    split: str,
    count: int,
    allowed_lengths: Sequence[int],
    rng: random.Random,
    forbidden: set[str],
) -> list[dict[str, Any]]:
    assignments = _relation_assignments(count, allowed_lengths, rng)
    rows = _stratified_operations([length for length, _ in assignments], rng)
    records: list[dict[str, Any]] = []
    for index, ((_, heldout_position), operations) in enumerate(zip(assignments, rows)):
        operations[heldout_position] = _operation(HELDOUT_JOINT)
        for _ in range(5000):
            start_state = dict(zip(REGISTER_NAMES, rng.sample(VALUE_LABELS, len(REGISTER_NAMES))))
            record = _decorate(_render_record(start_state, operations, rng.choice(REGISTER_NAMES), split, index))
            if record["fingerprint"] in forbidden:
                continue
            records.append(record)
            forbidden.add(record["fingerprint"])
            break
        else:
            raise RuntimeError(f"could not create a unique A1.8 {split} record at index {index}")
    return records


def _decorate_causal(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_decorate(record) for record in records]


def build_a18_dataset(
    output_dir: Path,
    spec: A18DatasetSpec,
    *,
    forbidden_data_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    if any(getattr(spec, field) <= 0 for field in asdict(spec) if field != "data_seed"):
        raise ValueError("all A1.8 split sizes must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.jsonl")):
        raise RuntimeError(f"refusing to overwrite an existing A1.8 dataset: {output_dir}")
    forbidden = _external_fingerprints(forbidden_data_dirs)
    external_forbidden_count = len(forbidden)
    split_records: dict[str, list[dict[str, Any]]] = {}
    ordinary_specs = (
        ("train", spec.train_size, TRAIN_LENGTHS),
        ("validation", spec.validation_size, TRAIN_LENGTHS),
        ("short_regression", spec.short_size, SHORT_LENGTHS),
        ("supported_in_range", spec.in_range_size, IN_RANGE_LENGTHS),
        ("supported_ood", spec.ood_size, OOD_LENGTHS),
    )
    for split_index, (split, count, allowed_lengths) in enumerate(ordinary_specs):
        rng = random.Random(spec.data_seed + split_index * 1000003)
        lengths = _balanced_values(allowed_lengths, count, rng)
        split_records[split] = _ordinary_split(split, count, lengths, rng, forbidden)
    relation_specs = (
        ("relation_in_range", spec.relation_in_range_size, IN_RANGE_LENGTHS, 5000015),
        ("relation_ood", spec.relation_ood_size, OOD_LENGTHS, 6000018),
    )
    for split, count, allowed_lengths, offset in relation_specs:
        split_records[split] = _relation_split(
            split,
            count,
            allowed_lengths,
            random.Random(spec.data_seed + offset),
            forbidden,
        )
    causal = _causal_split(
        "causal_core",
        spec.causal_size,
        random.Random(spec.data_seed + 7000021),
        forbidden,
    )
    split_records["causal_core"] = _decorate_causal(causal)
    for split in DATA_SPLITS:
        _write_split(output_dir, split, split_records[split])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "data_seed": spec.data_seed,
        "spec": asdict(spec),
        "training_lengths": list(TRAIN_LENGTHS),
        "short_regression_lengths": list(SHORT_LENGTHS),
        "in_range_lengths": list(IN_RANGE_LENGTHS),
        "ood_lengths": list(OOD_LENGTHS),
        "heldout_joint": _joint_name(HELDOUT_JOINT),
        "training_support": [_joint_name(joint) for joint in SUPPORTED_JOINTS],
        "external_forbidden_sources": [str(path) for path in forbidden_data_dirs],
        "external_forbidden_fingerprints": external_forbidden_count,
        "splits": {split: _split_stats(split_records[split]) for split in DATA_SPLITS},
        "unique_examples": sum(len(records) for records in split_records.values()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_a18_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    if split not in DATA_SPLITS:
        raise ValueError(f"unknown A1.8 split: {split}")
    with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _lengths_balanced(records: Sequence[dict[str, Any]]) -> bool:
    counts = Counter(int(record["program_length"]) for record in records)
    return bool(counts) and max(counts.values()) - min(counts.values()) <= 1


def _position_coverage(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    maximum = max(int(record["program_length"]) for record in records)
    for position in range(1, maximum + 1):
        counts = Counter(
            _joint(record["operations"][position - 1])
            for record in records
            if int(record["program_length"]) >= position
        )
        missing = [_joint_name(joint) for joint in SUPPORTED_JOINTS if counts[joint] == 0]
        coverage[str(position)] = {
            "active_examples": sum(counts.values()),
            "supported_present": len(SUPPORTED_JOINTS) - len(missing),
            "supported_total": len(SUPPORTED_JOINTS),
            "missing": missing,
        }
    return coverage


def _relation_position_balance(records: Sequence[dict[str, Any]]) -> tuple[bool, dict[str, int]]:
    counts = Counter(
        (int(record["program_length"]), position)
        for record in records
        for position, operation in enumerate(record["operations"], start=1)
        if _joint(operation) == HELDOUT_JOINT
    )
    lengths = sorted({int(record["program_length"]) for record in records})
    balanced = all(
        set(position for row_length, position in counts if row_length == length) == set(range(1, length + 1))
        and max(counts[(length, position)] for position in range(1, length + 1))
        - min(counts[(length, position)] for position in range(1, length + 1))
        <= 1
        for length in lengths
    )
    return balanced, {f"{length}:{position}": count for (length, position), count in sorted(counts.items())}


def audit_a18_data(data_dir: Path, output: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    records = {split: load_a18_records(data_dir, split) for split in DATA_SPLITS}
    stats = {split: _split_stats(rows) for split, rows in records.items()}
    expected_lengths = {
        "train": set(TRAIN_LENGTHS),
        "validation": set(TRAIN_LENGTHS),
        "short_regression": set(SHORT_LENGTHS),
        "supported_in_range": set(IN_RANGE_LENGTHS),
        "relation_in_range": set(IN_RANGE_LENGTHS),
        "supported_ood": set(OOD_LENGTHS),
        "relation_ood": set(OOD_LENGTHS),
        "causal_core": {2, 3, 4},
    }
    actual_lengths = {
        split: {int(record["program_length"]) for record in rows}
        for split, rows in records.items()
    }
    relation_counts = {
        split: [sum(_joint(operation) == HELDOUT_JOINT for operation in record["operations"]) for record in records[split]]
        for split in RELATION_SPLITS
    }
    relation_position_reports = {
        split: _relation_position_balance(records[split])
        for split in RELATION_SPLITS
    }
    supported_violations = [
        {"split": split, "example_id": record["example_id"], "joint": _joint_name(_joint(operation))}
        for split in SUPPORTED_SPLITS
        for record in records[split]
        for operation in record["operations"]
        if _joint(operation) not in SUPPORTED_JOINTS
    ]
    relation_background_violations = [
        {"split": split, "example_id": record["example_id"], "joint": _joint_name(_joint(operation))}
        for split in RELATION_SPLITS
        for record in records[split]
        for operation in record["operations"]
        if _joint(operation) not in SUPPORTED_JOINTS and _joint(operation) != HELDOUT_JOINT
    ]
    coverage = {
        split: _position_coverage(records[split])
        for split in ("train", "validation", "short_regression", "supported_in_range", "supported_ood")
    }
    fingerprints = {split: {record["fingerprint"] for record in rows} for split, rows in records.items()}
    overlaps = {
        f"{left}__{right}": len(fingerprints[left] & fingerprints[right])
        for left_index, left in enumerate(DATA_SPLITS)
        for right in DATA_SPLITS[left_index + 1 :]
    }
    external = _external_fingerprints(tuple(Path(path) for path in manifest["external_forbidden_sources"]))
    current = set().union(*fingerprints.values())
    causal_necessary = [_all_steps_necessary(record) for record in records["causal_core"]]
    multi_step_diversity = {
        split: len(
            {
                tuple(_joint(operation) for operation in record["operations"])
                for record in rows
                if int(record["program_length"]) >= 2
            }
        )
        / max(1, sum(int(record["program_length"]) >= 2 for record in rows))
        for split, rows in records.items()
    }
    gates = {
        "exact_preregistered_lengths": all(actual_lengths[split] == expected for split, expected in expected_lengths.items()),
        "length_counts_balanced": all(_lengths_balanced(rows) for rows in records.values()),
        "supported_splits_use_training_support_only": not supported_violations,
        "supported_position_coverage": all(not position["missing"] for split in coverage.values() for position in split.values()),
        "relation_exactly_one_heldout": all(all(count == 1 for count in relation_counts[split]) for split in RELATION_SPLITS),
        "relation_background_supported": not relation_background_violations,
        "relation_positions_balanced_within_length": all(report[0] for report in relation_position_reports.values()),
        "cross_split_fingerprint_overlap_zero": all(value == 0 for value in overlaps.values()),
        "external_fingerprint_overlap_zero": not (current & external),
        "causal_all_steps_deletion_necessary": bool(causal_necessary) and all(causal_necessary),
        # Short horizons have a small finite operation-sequence universe;
        # long evaluation splits must remain effectively unique. causal_core
        # is filtered for every-step deletion necessity and is gated separately.
        "multi_step_sequence_diversity": (
            all(multi_step_diversity[split] >= 0.45 for split in ("train", "validation", "short_regression"))
            and all(
                multi_step_diversity[split] >= 0.95
                for split in ("supported_in_range", "relation_in_range", "supported_ood", "relation_ood")
            )
        ),
        "schema_is_a1_8": all(record["schema_version"] == SCHEMA_VERSION for rows in records.values() for record in rows),
    }
    report = {
        "schema_version": "yggdrasil.v2-a1.8.data-audit.v1",
        "data_dir": str(data_dir),
        "data_seed": manifest["data_seed"],
        "expected_lengths": {split: sorted(lengths) for split, lengths in expected_lengths.items()},
        "splits": stats,
        "multi_step_sequence_diversity": multi_step_diversity,
        "position_coverage": coverage,
        "relation": {
            split: {
                "heldout_counts": dict(sorted(Counter(relation_counts[split]).items())),
                "heldout_positions": relation_position_reports[split][1],
            }
            for split in RELATION_SPLITS
        },
        "supported_violations": supported_violations,
        "relation_background_violations": relation_background_violations,
        "cross_split_overlap": overlaps,
        "external_forbidden_fingerprints": len(external),
        "external_overlap": len(current & external),
        "causal_core": {
            "examples": len(causal_necessary),
            "all_steps_deletion_necessary_rate": sum(causal_necessary) / max(1, len(causal_necessary)),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def assert_a18_data_gate(report: dict[str, Any]) -> None:
    failed = [name for name, passed in report["gates"].items() if not passed]
    if failed:
        raise RuntimeError("A1.8 data audit failed: " + ", ".join(failed))
