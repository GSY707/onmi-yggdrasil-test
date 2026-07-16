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
    _balanced_values,
    _joint,
    _joint_name,
    _operation,
    _render_record,
    _stratified_operations,
    _write_split,
    load_a17_records,
)
from .a1_7_train import CLOSURE_WEIGHT, evaluate_a17_core, load_a17_checkpoint


STRESS_SCHEMA_VERSION = "yggdrasil.v2-a1.7.long-recurrence-stress.v1"
STRESS_SPLITS = ("supported_length_stress", "relation_length_stress")
BASE_SPLITS = ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core")


@dataclass(frozen=True)
class A17StressSpec:
    seed: int = 20260718
    supported_size: int = 768
    relation_size: int = 768
    lengths: tuple[int, ...] = (8, 12, 16)


def _base_fingerprints(data_dir: Path) -> set[str]:
    return {
        record["fingerprint"]
        for split in BASE_SPLITS
        for record in load_a17_records(data_dir, split)
    }


def _render_unique_stress_records(
    split: str,
    lengths: Sequence[int],
    rng: random.Random,
    forbidden: set[str],
    *,
    relation_holdout: bool,
) -> list[dict[str, Any]]:
    rows = _stratified_operations(lengths, rng)
    heldout_position_use: dict[int, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for index, (length, operations) in enumerate(zip(lengths, rows)):
        if relation_holdout:
            position = heldout_position_use[length] % length
            heldout_position_use[length] += 1
            operations[position] = _operation(HELDOUT_JOINT)
        for _ in range(5000):
            start_state = dict(zip(REGISTER_NAMES, rng.sample(VALUE_LABELS, len(REGISTER_NAMES))))
            record = _render_record(start_state, operations, rng.choice(REGISTER_NAMES), split, index)
            if record["fingerprint"] in forbidden:
                continue
            record["schema_version"] = STRESS_SCHEMA_VERSION
            record["task_family"] = "a1.7_long_recurrence_stress"
            records.append(record)
            forbidden.add(record["fingerprint"])
            break
        else:
            raise RuntimeError(f"could not create a unique {split} record at index {index}")
    return records


def _stress_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    joint_counts = Counter(_joint(operation) for record in records for operation in record["operations"])
    heldout_by_length_position = Counter(
        (record["program_length"], position)
        for record in records
        for position, operation in enumerate(record["operations"], start=1)
        if _joint(operation) == HELDOUT_JOINT
    )
    sequences = Counter(tuple(_joint(operation) for operation in record["operations"]) for record in records)
    return {
        "examples": len(records),
        "length_distribution": dict(sorted(Counter(record["program_length"] for record in records).items())),
        "joint_counts": {_joint_name(joint): joint_counts[joint] for joint in ((*SUPPORTED_JOINTS, HELDOUT_JOINT))},
        "heldout_by_length_position": {
            f"{length}:{position}": count
            for (length, position), count in sorted(heldout_by_length_position.items())
        },
        "unique_sequence_patterns": len(sequences),
        "sequence_pattern_diversity": len(sequences) / max(1, len(records)),
    }


def build_a17_stress_dataset(output_dir: Path, base_data_dir: Path, spec: A17StressSpec) -> dict[str, Any]:
    if not spec.lengths or min(spec.lengths) <= 6:
        raise ValueError("stress lengths must be non-empty and strictly greater than the formal length-heldout maximum 6")
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden = _base_fingerprints(base_data_dir)
    base_count = len(forbidden)
    split_records: dict[str, list[dict[str, Any]]] = {}
    configs = (
        ("supported_length_stress", spec.supported_size, False, spec.seed),
        ("relation_length_stress", spec.relation_size, True, spec.seed + 1000003),
    )
    for split, count, relation_holdout, seed in configs:
        rng = random.Random(seed)
        lengths = _balanced_values(spec.lengths, count, rng)
        split_records[split] = _render_unique_stress_records(
            split,
            lengths,
            rng,
            forbidden,
            relation_holdout=relation_holdout,
        )
        _write_split(output_dir, split, split_records[split])
    manifest = {
        "schema_version": STRESS_SCHEMA_VERSION,
        "seed": spec.seed,
        "spec": asdict(spec),
        "base_data_dir": str(base_data_dir),
        "base_fingerprints": base_count,
        "splits": {split: _stress_stats(records) for split, records in split_records.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_a17_stress_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    if split not in STRESS_SPLITS:
        raise ValueError(f"unknown A1.7 stress split: {split}")
    with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_a17_stress_data(data_dir: Path, base_data_dir: Path, output: Path | None = None) -> dict[str, Any]:
    records = {split: load_a17_stress_records(data_dir, split) for split in STRESS_SPLITS}
    base = _base_fingerprints(base_data_dir)
    fingerprints = {split: {record["fingerprint"] for record in rows} for split, rows in records.items()}
    supported = records["supported_length_stress"]
    relation = records["relation_length_stress"]
    lengths = sorted({record["program_length"] for rows in records.values() for record in rows})
    relation_heldout_counts = [
        sum(_joint(operation) == HELDOUT_JOINT for operation in record["operations"])
        for record in relation
    ]
    relation_background_valid = all(
        _joint(operation) in SUPPORTED_JOINTS or _joint(operation) == HELDOUT_JOINT
        for record in relation
        for operation in record["operations"]
    )
    heldout_positions = Counter(
        (record["program_length"], position)
        for record in relation
        for position, operation in enumerate(record["operations"], start=1)
        if _joint(operation) == HELDOUT_JOINT
    )
    position_balance = all(
        set(position for (row_length, position) in heldout_positions if row_length == length) == set(range(1, length + 1))
        and max(heldout_positions[(length, position)] for position in range(1, length + 1))
        - min(heldout_positions[(length, position)] for position in range(1, length + 1))
        <= 1
        for length in lengths
    )
    all_stress_fingerprints = fingerprints[STRESS_SPLITS[0]] | fingerprints[STRESS_SPLITS[1]]
    gates = {
        "stress_lengths_strictly_above_six": bool(lengths) and min(lengths) > 6,
        "supported_uses_training_support_only": all(_joint(operation) in SUPPORTED_JOINTS for record in supported for operation in record["operations"]),
        "supported_covers_all_training_joints": all(any(_joint(operation) == joint for record in supported for operation in record["operations"]) for joint in SUPPORTED_JOINTS),
        "relation_exactly_one_heldout": all(count == 1 for count in relation_heldout_counts),
        "relation_background_supported": relation_background_valid,
        "relation_heldout_positions_balanced_within_length": position_balance,
        "no_base_fingerprint_overlap": not (all_stress_fingerprints & base),
        "no_stress_split_overlap": not (fingerprints[STRESS_SPLITS[0]] & fingerprints[STRESS_SPLITS[1]]),
        "stress_schema": all(record["schema_version"] == STRESS_SCHEMA_VERSION for rows in records.values() for record in rows),
    }
    report = {
        "schema_version": "yggdrasil.v2-a1.7.long-recurrence-stress-audit.v1",
        "data_dir": str(data_dir),
        "base_data_dir": str(base_data_dir),
        "lengths": lengths,
        "splits": {split: _stress_stats(rows) for split, rows in records.items()},
        "gates": gates,
        "passed": all(gates.values()),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def assert_a17_stress_gate(report: dict[str, Any]) -> None:
    failed = [name for name, passed in report["gates"].items() if not passed]
    if failed:
        raise RuntimeError("A1.7 stress data audit failed: " + ", ".join(failed))


def evaluate_a17_stress(
    checkpoint: Path,
    data_dir: Path,
    base_data_dir: Path,
    device: str,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    audit = audit_a17_stress_data(data_dir, base_data_dir)
    assert_a17_stress_gate(audit)
    model = load_a17_checkpoint(checkpoint, device)
    closure_weight = float(getattr(model, "training_closure_weight", CLOSURE_WEIGHT))
    results: dict[str, Any] = {}
    per_length: dict[str, Any] = {}
    for split in STRESS_SPLITS:
        rows = load_a17_stress_records(data_dir, split)
        results[split] = evaluate_a17_core(model, rows, device, batch_size=batch_size, closure_weight=closure_weight)
        per_length[split] = {
            str(length): evaluate_a17_core(
                model,
                [record for record in rows if record["program_length"] == length],
                device,
                batch_size=batch_size,
                closure_weight=closure_weight,
            )
            for length in sorted({record["program_length"] for record in rows})
        }
    gates = {
        "supported_aggregate_trajectory_at_least_0_95": results["supported_length_stress"]["trajectory_full_exact"] >= 0.95,
        "relation_aggregate_trajectory_at_least_0_95": results["relation_length_stress"]["trajectory_full_exact"] >= 0.95,
        "all_per_length_trajectory_at_least_0_95": all(
            metrics["trajectory_full_exact"] >= 0.95
            for split_metrics in per_length.values()
            for metrics in split_metrics.values()
        ),
        "all_pointer_accuracy_at_least_0_995": all(
            metrics["source_pointer_accuracy"] >= 0.995 and metrics["target_pointer_accuracy"] >= 0.995
            for metrics in results.values()
        ),
        "answer_state_identity": all(metrics["answer_state_logits_identity"]["gate"] for metrics in results.values()),
    }
    return {
        "schema_version": "yggdrasil.v2-a1.7.long-recurrence-stress-eval.v1",
        "checkpoint": str(checkpoint),
        "integrity": model.integrity_report(),
        "training_closure_weight": closure_weight,
        "splits": results,
        "per_length": per_length,
        "gates": gates,
        "passed": all(gates.values()),
    }
