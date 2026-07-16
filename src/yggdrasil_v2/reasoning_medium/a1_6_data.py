from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "yggdrasil.v2-a1.6.relation-state-machine.v1"
GENERATOR_VERSION = "a1.6-relation-addressed-state-machine-2026-07-15"
REGISTER_NAMES = ("amber", "cobalt", "jade")
VALUE_LABELS = tuple("ABCDEFGHIJ")
OPERATION_FAMILIES = ("copy", "swap")
FAMILY_TO_INDEX = {name: index for index, name in enumerate(OPERATION_FAMILIES)}


@dataclass(frozen=True)
class A16DatasetSpec:
    seed: int = 20260715
    train_size: int = 8192
    validation_size: int = 512
    test_size: int = 512
    length_size: int = 512
    relation_size: int = 512
    causal_size: int = 512
    train_min_length: int = 1
    train_max_length: int = 4
    eval_min_length: int = 2
    eval_max_length: int = 4
    length_min_length: int = 5
    length_max_length: int = 6
    causal_min_length: int = 2
    causal_max_length: int = 6


def execute_operations(start_state: dict[str, str], operations: Sequence[dict[str, Any]], operation_mask: Sequence[bool] | None = None) -> list[dict[str, str]]:
    state = dict(start_state)
    trajectory = [dict(state)]
    mask = list(operation_mask) if operation_mask is not None else [True] * len(operations)
    for active, operation in zip(mask, operations):
        if active:
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


def _operation(family: str, source: str, target: str) -> dict[str, Any]:
    if source == target:
        raise ValueError("source and target must differ")
    if family == "copy":
        text = f"COPY {source} -> {target}."
    elif family == "swap":
        text = f"SWAP {source} <-> {target}."
    else:
        raise ValueError(f"unknown family {family}")
    return {"family": family, "source": source, "target": target, "text": text, "template_id": 0}


def _sample_pair(rng: random.Random, *, forbidden: tuple[str, str] | None = None) -> tuple[str, str]:
    pairs = [(source, target) for source in REGISTER_NAMES for target in REGISTER_NAMES if source != target]
    if forbidden is not None:
        pairs = [pair for pair in pairs if pair != forbidden]
    return rng.choice(pairs)


def _family_sequence(index: int, length: int, rng: random.Random) -> list[str]:
    # Four phase shifts of CCSS provide all four bigrams while keeping every
    # valid absolute position close to 50% COPY.
    # The split generator stratifies lengths by index modulo four; use the
    # higher-order index for the family phase so length and family do not
    # accidentally correlate (especially at position four).
    phase = (index // 4) % 4
    base = ("copy", "copy", "swap", "swap")
    families = [base[(phase + position) % 4] for position in range(length)]
    if length > 4 and rng.random() < 0.5:
        families = [base[(phase + position + 1) % 4] for position in range(length)]
    return families


def _make_operations(
    rng: random.Random,
    index: int,
    length: int,
    *,
    relation_holdout: bool = False,
    causal: bool = False,
) -> list[dict[str, Any]]:
    families = _family_sequence(index, length, rng)
    operations: list[dict[str, Any]] = []
    balanced_cycle = (("amber", "cobalt"), ("cobalt", "jade"), ("jade", "amber"))
    for position, family in enumerate(families):
        if relation_holdout:
            source, target = _sample_pair(rng)
        else:
            # The three-cycle keeps both source and target registers balanced
            # while never emitting the held-out COPY amber -> jade relation.
            source, target = balanced_cycle[(index + position) % len(balanced_cycle)]
        operations.append(_operation(family, source, target))
    if relation_holdout:
        copy_positions = [i for i, op in enumerate(operations) if op["family"] == "copy"]
        if not copy_positions:
            operations[0] = _operation("copy", "amber", "jade")
            copy_positions = [0]
        position = copy_positions[index % len(copy_positions)]
        operations[position] = _operation("copy", "amber", "jade")
    if not relation_holdout and index < 3 and length:
        # Make the three required train-side relation witnesses explicit.
        special = (_operation("copy", "amber", "cobalt"), _operation("copy", "cobalt", "jade"), _operation("swap", "amber", "jade"))[index]
        eligible = next((i for i, family in enumerate(families) if family == special["family"]), None)
        if eligible is not None:
            operations[eligible] = special
    return operations


def _render_record(start_state: dict[str, str], operations: Sequence[dict[str, Any]], query_register: str, split: str, index: int) -> dict[str, Any]:
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
    pieces.append(".\nOperations:\n")
    cursor += len(".\nOperations:\n")
    operation_spans: list[dict[str, Any]] = []
    trajectory = execute_operations(start_state, operations)[1:]
    for step, (operation, state_after) in enumerate(zip(operations, trajectory), start=1):
        line = f"{step}. {operation['text']}\n"
        line_start = cursor
        pieces.append(line)
        cursor += len(line)
        family_start = line_start + len(f"{step}. ")
        family_end = family_start + len(operation["family"].upper())
        source_start = line.find(operation["source"], len(f"{step}. ")) + line_start
        target_start = line.find(operation["target"], source_start - line_start + 1) + line_start
        operation_spans.append({
            "step": step,
            "char_start": line_start,
            "char_end": cursor - 1,
            "family_char_start": family_start,
            "family_char_end": family_end,
            "source_char_start": source_start,
            "source_char_end": source_start + len(operation["source"]),
            "target_char_start": target_start,
            "target_char_end": target_start + len(operation["target"]),
            "state_after": state_after,
        })
    query_prefix = "Which symbol is in the "
    pieces.append(query_prefix)
    cursor += len(query_prefix)
    query_start = cursor
    pieces.append(query_register)
    cursor += len(query_register)
    pieces.append(" register at the end?")
    question = "".join(pieces)
    fingerprint = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "example_id": f"{split}-{index:06d}-{fingerprint[:10]}",
        "fingerprint": fingerprint,
        "split": split,
        "task_family": "relation_addressed_three_register_copy_swap",
        "start_state": start_state,
        "start_value_spans": initial_spans,
        "operations": list(operations),
        "operation_spans": operation_spans,
        "operation_mask": [True] * len(operations),
        "query_register": query_register,
        "query_register_span": {"char_start": query_start, "char_end": query_start + len(query_register)},
        "question": question,
        "state_trajectory": trajectory,
        "answer": trajectory[-1][query_register] if trajectory else start_state[query_register],
        "answer_index": VALUE_LABELS.index(trajectory[-1][query_register] if trajectory else start_state[query_register]),
        "program_length": len(operations),
    }


def _causal_candidate(rng: random.Random, split: str, index: int, length: int, relation_holdout: bool = False) -> dict[str, Any] | None:
    start_values = rng.sample(VALUE_LABELS, 3)
    start_state = dict(zip(REGISTER_NAMES, start_values))
    if length == 5:
        operations = [
            _operation("swap", "jade", "amber"),
            _operation("swap", "amber", "jade"),
            _operation("swap", "jade", "amber"),
            _operation("swap", "amber", "jade"),
            _operation("swap", "amber", "cobalt"),
        ]
        query = "cobalt"
    elif length == 6:
        operations = [
            _operation("swap", "cobalt", "amber"),
            _operation("swap", "amber", "cobalt"),
            _operation("swap", "jade", "amber"),
            _operation("copy", "jade", "amber"),
            _operation("swap", "amber", "cobalt"),
            _operation("copy", "cobalt", "amber"),
        ]
        query = "amber"
    else:
        operations = _make_operations(rng, index, length, relation_holdout=relation_holdout, causal=True)
        query = rng.choice(REGISTER_NAMES)
    trajectory = execute_operations(start_state, operations)
    final = trajectory[-1][query]
    for step in range(length):
        mask = [True] * length
        mask[step] = False
        if execute_operations(start_state, operations, mask)[-1][query] == final:
            return None
    return _render_record(start_state, operations, query, split, index)


def _ordinary_candidate(rng: random.Random, split: str, index: int, length: int, *, relation_holdout: bool = False) -> dict[str, Any]:
    start_values = rng.sample(VALUE_LABELS, 3)
    start_state = dict(zip(REGISTER_NAMES, start_values))
    operations = _make_operations(rng, index, length, relation_holdout=relation_holdout)
    return _render_record(start_state, operations, rng.choice(REGISTER_NAMES), split, index)


def _generate_split(split: str, count: int, rng: random.Random, *, min_length: int, max_length: int, relation_holdout: bool = False, causal: bool = False, forbidden: set[str] | None = None) -> list[dict[str, Any]]:
    forbidden = set(forbidden or ())
    records: list[dict[str, Any]] = []
    for index in range(count):
        length = min_length + (index % (max_length - min_length + 1))
        for attempt in range(10000):
            record = _causal_candidate(rng, split, index, length, relation_holdout) if causal else _ordinary_candidate(rng, split, index, length, relation_holdout=relation_holdout)
            if record is not None and record["fingerprint"] not in forbidden:
                records.append(record)
                forbidden.add(record["fingerprint"])
                break
            if causal:
                length = min_length + ((index + attempt + 1) % (max_length - min_length + 1))
        else:
            raise RuntimeError(f"unable to generate unique record for {split} index={index}")
    return records


def _write_split(output_dir: Path, split: str, records: Sequence[dict[str, Any]]) -> None:
    with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_a16_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _split_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    position_families: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        for position, operation in enumerate(record["operations"], start=1):
            position_families[str(position)][operation["family"]] += 1
    return {
        "examples": len(records),
        "length_distribution": dict(sorted(Counter(r["program_length"] for r in records).items())),
        "family_distribution": dict(sorted(Counter(o["family"] for r in records for o in r["operations"]).items())),
        "position_family_distribution": {key: dict(sorted(value.items())) for key, value in sorted(position_families.items(), key=lambda item: int(item[0]))},
        "bigram_distribution": {"->".join(key): value for key, value in sorted(Counter(tuple(o["family"] for o in r["operations"][i : i + 2]) for r in records for i in range(max(0, len(r["operations"]) - 1))).items(), key=str)},
        "copy_count_distribution": dict(sorted(Counter(sum(o["family"] == "copy" for o in r["operations"]) for r in records).items())),
        "copy_position_distribution": dict(sorted(Counter(i + 1 for r in records for i, o in enumerate(r["operations"]) if o["family"] == "copy").items())),
        "source_distribution": dict(sorted(Counter(o["source"] for r in records for o in r["operations"]).items())),
        "target_distribution": dict(sorted(Counter(o["target"] for r in records for o in r["operations"]).items())),
    }


def build_a16_dataset(output_dir: Path, spec: A16DatasetSpec) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "train": (spec.train_size, spec.train_min_length, spec.train_max_length, False, False),
        "validation": (spec.validation_size, spec.eval_min_length, spec.eval_max_length, False, False),
        "test": (spec.test_size, spec.eval_min_length, spec.eval_max_length, False, False),
        "length_heldout": (spec.length_size, spec.length_min_length, spec.length_max_length, False, False),
        "relation_heldout": (spec.relation_size, spec.eval_min_length, spec.eval_max_length, True, False),
        "causal_core": (spec.causal_size, spec.causal_min_length, spec.causal_max_length, False, True),
    }
    seen: set[str] = set()
    stats: dict[str, Any] = {}
    for split_index, (split, (count, low, high, relation, causal)) in enumerate(configs.items()):
        rng = random.Random(spec.seed + split_index * 1000003)
        records = _generate_split(split, count, rng, min_length=low, max_length=high, relation_holdout=relation, causal=causal, forbidden=seen)
        seen.update(r["fingerprint"] for r in records)
        _write_split(output_dir, split, records)
        stats[split] = _split_stats(records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": spec.seed,
        "spec": asdict(spec),
        "splits": stats,
        "unique_examples": sum(item["examples"] for item in stats.values()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _family_pair(record: dict[str, Any]) -> list[tuple[str, str]]:
    families = [o["family"] for o in record["operations"]]
    return list(zip(families, families[1:]))


def audit_a16_data(data_dir: Path, output: Path | None = None, *, write: bool = True) -> dict[str, Any]:
    splits = ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core")
    records = {split: load_a16_records(data_dir, split) for split in splits}
    train = records["train"]
    all_fingerprints = [r["fingerprint"] for items in records.values() for r in items]
    pair = ("amber", "jade")
    train_relation_count = sum(o["family"] == "copy" and (o["source"], o["target"]) == pair for r in train for o in r["operations"])
    relation_coverage = sum(any(o["family"] == "copy" and (o["source"], o["target"]) == pair for o in r["operations"]) for r in records["relation_heldout"]) / max(1, len(records["relation_heldout"]))
    causal_checks = []
    for record in records["causal_core"]:
        final = record["answer"]
        necessary = []
        for step in range(record["program_length"]):
            mask = [True] * record["program_length"]
            mask[step] = False
            necessary.append(execute_operations(record["start_state"], record["operations"], mask)[-1][record["query_register"]] != final)
        causal_checks.extend(necessary)
    bigrams = Counter(_family_pair(r)[0] if False else pair_value for r in train for pair_value in _family_pair(r))
    position_rates: dict[str, float] = {}
    for position in range(1, 5):
        values = [r["operations"][position - 1]["family"] == "copy" for r in train if len(r["operations"]) >= position]
        position_rates[str(position)] = sum(values) / max(1, len(values))
    overlaps = {}
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len({r["fingerprint"] for r in records[left]} & {r["fingerprint"] for r in records[right]})
    bigram_total = max(1, sum(bigrams.values()))
    source_counts = Counter(o["source"] for r in train for o in r["operations"])
    target_counts = Counter(o["target"] for r in train for o in r["operations"])
    source_mean = sum(source_counts.values()) / len(REGISTER_NAMES)
    target_mean = sum(target_counts.values()) / len(REGISTER_NAMES)
    gates = {
        "train_four_bigrams_each_at_least_10_percent": all(bigrams[pair] / bigram_total >= 0.10 for pair in (("copy", "copy"), ("copy", "swap"), ("swap", "copy"), ("swap", "swap"))),
        "train_copy_at_positions_1_to_4": all(any(len(r["operations"]) >= p and r["operations"][p - 1]["family"] == "copy" for r in train) for p in range(1, 5)),
        "train_copy_rate_per_position_0_35_to_0_65": all(0.35 <= rate <= 0.65 for rate in position_rates.values()),
        "train_source_target_approximately_balanced": max(source_counts.values()) - min(source_counts.values()) <= source_mean * 0.25 and max(target_counts.values()) - min(target_counts.values()) <= target_mean * 0.25,
        "relation_pair_absent_from_non_holdout": train_relation_count == 0 and all(not any(o["family"] == "copy" and (o["source"], o["target"]) == pair for r in records[split] for o in r["operations"]) for split in ("validation", "test", "length_heldout")),
        "relation_pair_coverage_1": relation_coverage == 1.0,
        "causal_core_necessary_rate_1": bool(causal_checks) and sum(causal_checks) / len(causal_checks) == 1.0,
        "cross_split_overlap_0": all(value == 0 for value in overlaps.values()),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "data_dir": str(data_dir),
        "splits": {split: _split_stats(items) for split, items in records.items()},
        "relation_holdout": {"pair": ["copy", "amber", "jade"], "train_count": train_relation_count, "heldout_coverage": relation_coverage},
        "fingerprint_overlap": overlaps,
        "causal_core": {"necessary_rate": sum(causal_checks) / max(1, len(causal_checks)), "checks": len(causal_checks)},
        "train_bigrams": {str(key): value for key, value in sorted(bigrams.items(), key=str)},
        "train_copy_rate_by_position": position_rates,
        "train_source_distribution": dict(sorted(source_counts.items())),
        "train_target_distribution": dict(sorted(target_counts.items())),
        "gates": gates,
        "passed": all(gates.values()),
    }
    if write and output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def assert_data_gate(report: dict[str, Any]) -> None:
    failed = [name for name, passed in report.get("gates", {}).items() if not passed]
    if failed:
        raise RuntimeError("A1.6 data Gate failed: " + ", ".join(failed))
