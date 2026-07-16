from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "yggdrasil.v2-a1.7.relation-state-machine.v1"
GENERATOR_VERSION = "a1.7-controlled-relation-state-machine-2026-07-15"
REGISTER_NAMES = ("amber", "cobalt", "jade")
VALUE_LABELS = tuple("ABCDEFGHIJ")
OPERATION_FAMILIES = ("copy", "swap")
FAMILY_TO_INDEX = {name: index for index, name in enumerate(OPERATION_FAMILIES)}
DIRECTED_PAIRS = tuple((source, target) for source in REGISTER_NAMES for target in REGISTER_NAMES if source != target)
HELDOUT_JOINT = ("copy", "amber", "jade")
SUPPORTED_JOINTS = tuple(
    (family, source, target)
    for family in OPERATION_FAMILIES
    for source, target in DIRECTED_PAIRS
    if (family, source, target) != HELDOUT_JOINT
)


@dataclass(frozen=True)
class A17DatasetSpec:
    seed: int = 20260715
    train_size: int = 8192
    validation_size: int = 512
    test_size: int = 512
    length_size: int = 512
    relation_size: int = 512
    causal_size: int = 512


def execute_operations(
    start_state: dict[str, str],
    operations: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    state = dict(start_state)
    trajectory = [dict(state)]
    for operation in operations:
        family = operation["family"]
        source, target = operation["source"], operation["target"]
        if family == "copy":
            state[target] = state[source]
        elif family == "swap":
            state[source], state[target] = state[target], state[source]
        else:
            raise ValueError(f"unknown operation family: {family}")
        trajectory.append(dict(state))
    return trajectory


def _operation(joint: tuple[str, str, str]) -> dict[str, Any]:
    family, source, target = joint
    if source == target:
        raise ValueError("source and target must differ")
    text = f"COPY {source} -> {target}." if family == "copy" else f"SWAP {source} <-> {target}."
    return {"family": family, "source": source, "target": target, "text": text, "template_id": 0}


def _render_record(
    start_state: dict[str, str],
    operations: Sequence[dict[str, Any]],
    query_register: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    prefix = "Three registers each contain one symbol. Execute the operations in order.\nInitially "
    pieces = [prefix]
    cursor = len(prefix)
    initial_spans: dict[str, dict[str, int]] = {}
    for position, register in enumerate(REGISTER_NAMES):
        text = f"{register}={start_state[register]}"
        if position:
            pieces.append(", ")
            cursor += 2
        start = cursor
        pieces.append(text)
        cursor += len(text)
        initial_spans[register] = {"char_start": start, "char_end": cursor}
    marker = ".\nOperations:\n"
    pieces.append(marker)
    cursor += len(marker)
    operation_spans: list[dict[str, Any]] = []
    trajectory = execute_operations(start_state, operations)[1:]
    for step, (operation, state_after) in enumerate(zip(operations, trajectory), start=1):
        line = f"{step}. {operation['text']}\n"
        line_start = cursor
        pieces.append(line)
        cursor += len(line)
        family_start = line_start + len(f"{step}. ")
        source_start = line.find(operation["source"], len(f"{step}. ")) + line_start
        target_start = line.find(operation["target"], source_start - line_start + 1) + line_start
        operation_spans.append(
            {
                "step": step,
                "char_start": line_start,
                "char_end": cursor - 1,
                "family_char_start": family_start,
                "family_char_end": family_start + len(operation["family"]),
                "source_char_start": source_start,
                "source_char_end": source_start + len(operation["source"]),
                "target_char_start": target_start,
                "target_char_end": target_start + len(operation["target"]),
                "state_after": state_after,
            }
        )
    query_prefix = "Which symbol is in the "
    pieces.append(query_prefix)
    cursor += len(query_prefix)
    query_start = cursor
    pieces.append(query_register)
    cursor += len(query_register)
    pieces.append(" register at the end?")
    question = "".join(pieces)
    fingerprint = hashlib.sha256(question.encode("utf-8")).hexdigest()
    answer = trajectory[-1][query_register] if trajectory else start_state[query_register]
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "example_id": f"{split}-{index:06d}-{fingerprint[:10]}",
        "fingerprint": fingerprint,
        "split": split,
        "task_family": "controlled_relation_addressed_three_register_copy_swap",
        "start_state": start_state,
        "start_value_spans": initial_spans,
        "operations": [dict(operation) for operation in operations],
        "operation_spans": operation_spans,
        "operation_mask": [True] * len(operations),
        "query_register": query_register,
        "query_register_span": {"char_start": query_start, "char_end": cursor},
        "question": question,
        "state_trajectory": trajectory,
        "answer": answer,
        "answer_index": VALUE_LABELS.index(answer),
        "program_length": len(operations),
    }


def _balanced_values(items: Sequence[Any], count: int, rng: random.Random) -> list[Any]:
    values = [items[index % len(items)] for index in range(count)]
    rng.shuffle(values)
    return values


def _balanced_supported_joints(count: int, rng: random.Random) -> list[tuple[str, str, str]]:
    # Balance family first, then directed pairs inside each family. This avoids
    # turning the missing COPY amber->jade cell into a family-frequency cue.
    families = _balanced_values(OPERATION_FAMILIES, count, rng)
    copy_pairs = [pair for pair in DIRECTED_PAIRS if ("copy", *pair) != HELDOUT_JOINT]
    swap_pairs = list(DIRECTED_PAIRS)
    copy_values = _balanced_values(copy_pairs, families.count("copy"), rng)
    swap_values = _balanced_values(swap_pairs, families.count("swap"), rng)
    copy_index = swap_index = 0
    joints: list[tuple[str, str, str]] = []
    for family in families:
        if family == "copy":
            source, target = copy_values[copy_index]
            copy_index += 1
        else:
            source, target = swap_values[swap_index]
            swap_index += 1
        joints.append((family, source, target))
    return joints


def _lengths(count: int, allowed: Sequence[int], rng: random.Random) -> list[int]:
    return _balanced_values(allowed, count, rng)


def _stratified_operations(lengths: Sequence[int], rng: random.Random) -> list[list[dict[str, Any]]]:
    operations: list[list[dict[str, Any] | None]] = [[None] * length for length in lengths]
    for position in range(1, max(lengths) + 1):
        active = [index for index, length in enumerate(lengths) if length >= position]
        joints = _balanced_supported_joints(len(active), rng)
        for index, joint in zip(active, joints):
            operations[index][position - 1] = _operation(joint)
    return [[operation for operation in row if operation is not None] for row in operations]


def _ordinary_split(
    split: str,
    count: int,
    lengths: Sequence[int],
    rng: random.Random,
    forbidden: set[str],
) -> list[dict[str, Any]]:
    operation_rows = _stratified_operations(lengths, rng)
    records: list[dict[str, Any]] = []
    for index, operations in enumerate(operation_rows):
        for _ in range(1000):
            start_state = dict(zip(REGISTER_NAMES, rng.sample(VALUE_LABELS, 3)))
            record = _render_record(start_state, operations, rng.choice(REGISTER_NAMES), split, index)
            if record["fingerprint"] not in forbidden:
                records.append(record)
                forbidden.add(record["fingerprint"])
                break
        else:
            raise RuntimeError(f"could not create a unique {split} record at index {index}")
    return records


def _relation_assignments(count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count % 4:
        raise ValueError("relation_size must be divisible by four for exact heldout-position stratification")
    per_position = count // 4
    # Greedy balancing keeps length 2/3/4 counts within one while enforcing
    # position <= length. Candidate choice is randomized among equally needed lengths.
    length_targets = {length: count // 3 + (1 if offset < count % 3 else 0) for offset, length in enumerate((2, 3, 4))}
    length_used = Counter()
    assignments: list[tuple[int, int]] = []
    positions = [position for position in range(1, 5) for _ in range(per_position)]
    rng.shuffle(positions)
    for position in sorted(positions, reverse=True):
        eligible = [length for length in (2, 3, 4) if length >= position and length_used[length] < length_targets[length]]
        if not eligible:
            eligible = [length for length in (2, 3, 4) if length >= position]
        remaining = max(length_targets[length] - length_used[length] for length in eligible)
        choices = [length for length in eligible if length_targets[length] - length_used[length] == remaining]
        length = rng.choice(choices)
        length_used[length] += 1
        assignments.append((position, length))
    rng.shuffle(assignments)
    return assignments


def _relation_split(
    split: str,
    count: int,
    rng: random.Random,
    forbidden: set[str],
) -> list[dict[str, Any]]:
    assignments = _relation_assignments(count, rng)
    lengths = [length for _, length in assignments]
    rows = _stratified_operations(lengths, rng)
    records: list[dict[str, Any]] = []
    for index, ((heldout_position, _), operations) in enumerate(zip(assignments, rows)):
        operations[heldout_position - 1] = _operation(HELDOUT_JOINT)
        for _ in range(1000):
            start_state = dict(zip(REGISTER_NAMES, rng.sample(VALUE_LABELS, 3)))
            record = _render_record(start_state, operations, rng.choice(REGISTER_NAMES), split, index)
            if record["fingerprint"] not in forbidden:
                records.append(record)
                forbidden.add(record["fingerprint"])
                break
        else:
            raise RuntimeError(f"could not create a unique relation record at index {index}")
    return records


def _all_steps_necessary(record: dict[str, Any]) -> bool:
    expected = record["answer"]
    for step in range(record["program_length"]):
        operations = [operation for index, operation in enumerate(record["operations"]) if index != step]
        final = execute_operations(record["start_state"], operations)[-1][record["query_register"]]
        if final == expected:
            return False
    return True


def _causal_split(
    split: str,
    count: int,
    rng: random.Random,
    forbidden: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    target_lengths = _lengths(count, (2, 3, 4), rng)
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 20000:
            raise RuntimeError("could not generate enough all-step-necessary causal records")
        length = target_lengths[len(records)]
        joints = rng.sample(SUPPORTED_JOINTS, k=length) if length <= len(SUPPORTED_JOINTS) else [rng.choice(SUPPORTED_JOINTS) for _ in range(length)]
        operations = [_operation(joint) for joint in joints]
        start_state = dict(zip(REGISTER_NAMES, rng.sample(VALUE_LABELS, 3)))
        record = _render_record(start_state, operations, rng.choice(REGISTER_NAMES), split, len(records))
        if record["fingerprint"] in forbidden or not _all_steps_necessary(record):
            continue
        records.append(record)
        forbidden.add(record["fingerprint"])
    return records


def _write_split(output_dir: Path, split: str, records: Sequence[dict[str, Any]]) -> None:
    with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_a17_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _joint(operation: dict[str, Any]) -> tuple[str, str, str]:
    return operation["family"], operation["source"], operation["target"]


def _joint_name(joint: tuple[str, str, str]) -> str:
    return f"{joint[0]}:{joint[1]}->{joint[2]}"


def _split_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    joint_counts = Counter(_joint(operation) for record in records for operation in record["operations"])
    position_counts: dict[int, Counter[tuple[str, str, str]]] = defaultdict(Counter)
    for record in records:
        for position, operation in enumerate(record["operations"], start=1):
            position_counts[position][_joint(operation)] += 1
    sequence_counts = Counter(tuple(_joint(operation) for operation in record["operations"]) for record in records)
    multi_step_records = [record for record in records if record["program_length"] >= 2]
    multi_step_sequences = Counter(tuple(_joint(operation) for operation in record["operations"]) for record in multi_step_records)
    family_counts = Counter(operation["family"] for record in records for operation in record["operations"])
    source_counts = Counter(operation["source"] for record in records for operation in record["operations"])
    target_counts = Counter(operation["target"] for record in records for operation in record["operations"])
    return {
        "examples": len(records),
        "length_distribution": dict(sorted(Counter(record["program_length"] for record in records).items())),
        "joint_counts": {_joint_name(joint): joint_counts[joint] for joint in ((*SUPPORTED_JOINTS, HELDOUT_JOINT))},
        "family_pair_position_counts": {
            str(position): {_joint_name(joint): counts[joint] for joint in ((*SUPPORTED_JOINTS, HELDOUT_JOINT))}
            for position, counts in sorted(position_counts.items())
        },
        "sequence_pattern_diversity": len(sequence_counts) / max(1, len(records)),
        "unique_sequence_patterns": len(sequence_counts),
        "multi_step_sequence_pattern_diversity": len(multi_step_sequences) / max(1, len(multi_step_records)),
        "unique_multi_step_sequence_patterns": len(multi_step_sequences),
        "marginals": {
            "family": dict(sorted(family_counts.items())),
            "source": dict(sorted(source_counts.items())),
            "target": dict(sorted(target_counts.items())),
        },
    }


def build_a17_dataset(output_dir: Path, spec: A17DatasetSpec) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden: set[str] = set()
    split_records: dict[str, list[dict[str, Any]]] = {}
    ordinary_configs = {
        "train": (spec.train_size, (1, 2, 3, 4)),
        "validation": (spec.validation_size, (1, 2, 3, 4)),
        "test": (spec.test_size, (1, 2, 3, 4)),
        "length_heldout": (spec.length_size, (5, 6)),
    }
    for split_index, (split, (count, allowed_lengths)) in enumerate(ordinary_configs.items()):
        rng = random.Random(spec.seed + split_index * 1000003)
        lengths = _lengths(count, allowed_lengths, rng)
        split_records[split] = _ordinary_split(split, count, lengths, rng, forbidden)
    relation_rng = random.Random(spec.seed + 4000012)
    split_records["relation_heldout"] = _relation_split("relation_heldout", spec.relation_size, relation_rng, forbidden)
    causal_rng = random.Random(spec.seed + 5000015)
    split_records["causal_core"] = _causal_split("causal_core", spec.causal_size, causal_rng, forbidden)
    for split, records in split_records.items():
        _write_split(output_dir, split, records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": spec.seed,
        "spec": asdict(spec),
        "heldout_joint": _joint_name(HELDOUT_JOINT),
        "training_support": [_joint_name(joint) for joint in SUPPORTED_JOINTS],
        "splits": {split: _split_stats(records) for split, records in split_records.items()},
        "unique_examples": sum(len(records) for records in split_records.values()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _approximately_balanced(counts: Counter[str], tolerance: float = 0.20) -> bool:
    if not counts:
        return False
    mean = sum(counts.values()) / len(counts)
    return max(abs(value - mean) for value in counts.values()) <= tolerance * mean


def audit_a17_data(data_dir: Path, output: Path | None = None) -> dict[str, Any]:
    split_names = ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core")
    records = {split: load_a17_records(data_dir, split) for split in split_names}
    stats = {split: _split_stats(rows) for split, rows in records.items()}
    ordinary = ("train", "validation", "test", "length_heldout")
    joint_counts = {
        split: Counter(_joint(operation) for record in rows for operation in record["operations"])
        for split, rows in records.items()
    }
    heldout_counts_per_relation_example = [
        sum(_joint(operation) == HELDOUT_JOINT for operation in record["operations"])
        for record in records["relation_heldout"]
    ]
    heldout_positions = Counter(
        position
        for record in records["relation_heldout"]
        for position, operation in enumerate(record["operations"], start=1)
        if _joint(operation) == HELDOUT_JOINT
    )
    background_support_violations = [
        {"split": split, "example_id": record["example_id"], "joint": _joint_name(_joint(operation))}
        for split, rows in records.items()
        for record in rows
        for operation in record["operations"]
        if _joint(operation) not in SUPPORTED_JOINTS and not (split == "relation_heldout" and _joint(operation) == HELDOUT_JOINT)
    ]
    coverage: dict[str, dict[str, Any]] = {}
    for split, rows in records.items():
        by_position: dict[str, Any] = {}
        for position in range(1, max(record["program_length"] for record in rows) + 1):
            counts = Counter(_joint(record["operations"][position - 1]) for record in rows if record["program_length"] >= position)
            missing = [_joint_name(joint) for joint in SUPPORTED_JOINTS if counts[joint] == 0]
            by_position[str(position)] = {"supported_present": len(SUPPORTED_JOINTS) - len(missing), "supported_total": len(SUPPORTED_JOINTS), "missing": missing}
        coverage[split] = by_position
    fingerprints = {split: {record["fingerprint"] for record in rows} for split, rows in records.items()}
    overlaps = {
        f"{left}__{right}": len(fingerprints[left] & fingerprints[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    }
    necessary = [
        _all_steps_necessary(record)
        for record in records["causal_core"]
    ]
    marginal_balance = {}
    for split, rows in records.items():
        marginal_balance[split] = {
            "family": _approximately_balanced(Counter(operation["family"] for record in rows for operation in record["operations"])),
            "source": _approximately_balanced(Counter(operation["source"] for record in rows for operation in record["operations"])),
            "target": _approximately_balanced(Counter(operation["target"] for record in rows for operation in record["operations"])),
        }
    gates = {
        "ordinary_support_exactly_11": all(joint_counts[split][HELDOUT_JOINT] == 0 and all(joint_counts[split][joint] > 0 for joint in SUPPORTED_JOINTS) for split in ordinary),
        "relation_exactly_one_heldout_per_example": all(count == 1 for count in heldout_counts_per_relation_example),
        "relation_background_within_training_support": not background_support_violations,
        "relation_heldout_positions_balanced": set(heldout_positions) == {1, 2, 3, 4} and max(heldout_positions.values()) - min(heldout_positions.values()) <= 1,
        "ordinary_family_pair_position_coverage": all(not coverage[split][str(position)]["missing"] for split in ordinary for position in range(1, max(int(key) for key in coverage[split]) + 1)),
        # Length-1 rows have only eleven possible sequences by construction;
        # diversity is therefore gated on compositional (T>=2) programs.
        "ordinary_sequence_pattern_diversity": all(stats[split]["multi_step_sequence_pattern_diversity"] >= 0.45 for split in ordinary),
        "ordinary_marginals_balanced": all(all(marginal_balance[split].values()) for split in ordinary),
        "cross_split_fingerprint_overlap_zero": all(value == 0 for value in overlaps.values()),
        "causal_all_steps_deletion_necessary": bool(necessary) and all(necessary),
        "schema_is_a1_7": all(record["schema_version"] == SCHEMA_VERSION for rows in records.values() for record in rows),
    }
    report = {
        "schema_version": "yggdrasil.v2-a1.7.data-audit.v1",
        "data_dir": str(data_dir),
        "heldout_joint": _joint_name(HELDOUT_JOINT),
        "training_support": [_joint_name(joint) for joint in SUPPORTED_JOINTS],
        "splits": stats,
        "heldout": {
            "counts_per_relation_example": dict(sorted(Counter(heldout_counts_per_relation_example).items())),
            "position_distribution": dict(sorted(heldout_positions.items())),
            "background_support_violations": background_support_violations,
        },
        "family_pair_position_coverage": coverage,
        "marginal_balance": marginal_balance,
        "cross_split_overlap": overlaps,
        "causal_core": {"examples": len(necessary), "necessary_rate": sum(necessary) / max(1, len(necessary))},
        "gates": gates,
        "passed": all(gates.values()),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def assert_a17_data_gate(report: dict[str, Any]) -> None:
    failed = [name for name, passed in report["gates"].items() if not passed]
    if failed:
        raise RuntimeError(f"A1.7 data audit failed: {', '.join(failed)}")
