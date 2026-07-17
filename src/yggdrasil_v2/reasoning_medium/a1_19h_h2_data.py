from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .a1_7_data import OPERATION_FAMILIES, VALUE_LABELS, execute_operations
from .a1_8_data import IN_RANGE_LENGTHS, OOD_LENGTHS, SHORT_LENGTHS, TRAIN_LENGTHS


SCHEMA_VERSION = "yggdrasil.v2-a1.19h.h2-variable-cardinality.v1"
GENERATOR_VERSION = "a1.19h-h2-variable-cardinality-2026-07-17"
TRAIN_ENTITY_COUNTS = (2, 3, 4)
HELDOUT_ENTITY_COUNT = 5
DATA_SPLITS = (
    "train",
    "validation",
    "short_regression",
    "supported_in_range",
    "relation_in_range",
    "supported_ood",
    "relation_ood",
    "entity_heldout",
    "entity_relation_heldout",
    "causal_core",
)
FORMAL_SPLITS = DATA_SPLITS[2:]


@dataclass(frozen=True)
class A119H2DatasetSpec:
    data_seed: int
    train_size: int = 16384
    validation_size: int = 2048
    short_size: int = 768
    in_range_size: int = 768
    relation_in_range_size: int = 768
    ood_size: int = 768
    relation_ood_size: int = 768
    entity_heldout_size: int = 768
    entity_relation_heldout_size: int = 768
    causal_size: int = 512


def _names(count: int, rng: random.Random) -> tuple[str, ...]:
    names: list[str] = []
    while len(names) < count:
        candidate = f"object_{rng.randrange(1 << 48):012x}"
        if candidate not in names:
            names.append(candidate)
    return tuple(names)


def _heldout_joint(names: Sequence[str]) -> tuple[str, str, str]:
    return "copy", names[0], names[-1]


def _operation(family: str, source: str, target: str) -> dict[str, str]:
    if source == target:
        raise ValueError("operation source and target must differ")
    return {"family": family, "source": source, "target": target}


def _supported_operation(names: Sequence[str], rng: random.Random) -> dict[str, str]:
    heldout = _heldout_joint(names)
    while True:
        family = rng.choice(OPERATION_FAMILIES)
        source, target = rng.sample(names, 2)
        if (family, source, target) != heldout:
            return _operation(family, source, target)


def _render_record(
    *,
    count: int,
    entity_names: Sequence[str],
    start_state: dict[str, str],
    operations: Sequence[dict[str, str]],
    query_entity: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    names = tuple(entity_names)
    if len(names) != count:
        raise ValueError("entity_names must match entity count")
    trajectory = execute_operations(start_state, operations)[1:]
    answer_state = trajectory[-1] if trajectory else start_state
    question = (
        "Objects: "
        + ", ".join(f"{name}={start_state[name]}" for name in names)
        + ". Operations: "
        + "; ".join(
            f"{op['family'].upper()} {op['source']} -> {op['target']}"
            for op in operations
        )
        + f". Query: {query_entity}."
    )
    semantic = json.dumps(
        {
            "start": start_state,
            "operations": operations,
            "query": query_entity,
        },
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(semantic.encode("utf-8")).hexdigest()
    answer = answer_state[query_entity]
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "example_id": f"{split}-{index:06d}-{fingerprint[:10]}",
        "fingerprint": fingerprint,
        "split": split,
        "task_family": "variable_cardinality_handle_addressed_copy_swap",
        "entity_count": count,
        "entity_names": list(names),
        "start_state": dict(start_state),
        "operations": [dict(operation) for operation in operations],
        "query_register": query_entity,
        "question": question,
        "state_trajectory": trajectory,
        "answer": answer,
        "answer_index": VALUE_LABELS.index(answer),
        "program_length": len(operations),
    }


def recompute_a119h2_record(
    record: dict[str, Any],
    *,
    start_state: dict[str, str] | None = None,
    operations: Sequence[dict[str, str]] | None = None,
    query_entity: str | None = None,
    split: str,
    index: int,
) -> dict[str, Any]:
    return _render_record(
        count=int(record["entity_count"]),
        entity_names=list(record["entity_names"]),
        start_state=dict(start_state or record["start_state"]),
        operations=list(operations if operations is not None else record["operations"]),
        query_entity=query_entity or str(record["query_register"]),
        split=split,
        index=index,
    )


def _balanced(items: Sequence[int], count: int, rng: random.Random) -> list[int]:
    result = [int(items[index % len(items)]) for index in range(count)]
    rng.shuffle(result)
    return result


def _records(
    *,
    split: str,
    size: int,
    counts: Sequence[int],
    lengths: Sequence[int],
    rng: random.Random,
    forbidden: set[str],
    relation_heldout: bool = False,
    causal: bool = False,
) -> list[dict[str, Any]]:
    count_schedule = _balanced(counts, size, rng)
    length_schedule = _balanced(lengths, size, rng)
    rows: list[dict[str, Any]] = []
    for index, (count, length) in enumerate(zip(count_schedule, length_schedule)):
        for _ in range(10000):
            names = _names(count, rng)
            operations = [_supported_operation(names, rng) for _ in range(length)]
            if relation_heldout:
                position = rng.randrange(length)
                operations[position] = _operation(*_heldout_joint(names))
            start_state = dict(zip(names, rng.sample(VALUE_LABELS, count)))
            query = rng.choice(names)
            record = _render_record(
                count=count,
                entity_names=names,
                start_state=start_state,
                operations=operations,
                query_entity=query,
                split=split,
                index=index,
            )
            if record["fingerprint"] in forbidden:
                continue
            if causal:
                full = [start_state] + record["state_trajectory"]
                if any(before == after for before, after in zip(full, full[1:])):
                    continue
            forbidden.add(record["fingerprint"])
            rows.append(record)
            break
        else:
            raise RuntimeError(f"could not generate unique A1.19H-H2 {split} record")
    return rows


def _external_fingerprints(data_dirs: Sequence[Path]) -> set[str]:
    fingerprints: set[str] = set()
    for data_dir in data_dirs:
        for path in data_dir.glob("*.jsonl"):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        fingerprints.add(json.loads(line)["fingerprint"])
    return fingerprints


def _write_split(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "entity_counts": dict(sorted(Counter(int(row["entity_count"]) for row in rows).items())),
        "lengths": dict(sorted(Counter(int(row["program_length"]) for row in rows).items())),
        "families": dict(
            sorted(Counter(op["family"] for row in rows for op in row["operations"]).items())
        ),
    }


def build_a119h2_dataset(
    output_dir: Path,
    spec: A119H2DatasetSpec,
    *,
    forbidden_data_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("*.jsonl")):
        raise RuntimeError(f"refusing to overwrite A1.19H-H2 data: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    forbidden = _external_fingerprints(forbidden_data_dirs)
    external_count = len(forbidden)
    configs = (
        ("train", spec.train_size, TRAIN_ENTITY_COUNTS, TRAIN_LENGTHS, False, False, 1),
        ("validation", spec.validation_size, TRAIN_ENTITY_COUNTS, TRAIN_LENGTHS, False, False, 2),
        ("short_regression", spec.short_size, TRAIN_ENTITY_COUNTS, SHORT_LENGTHS, False, False, 3),
        ("supported_in_range", spec.in_range_size, TRAIN_ENTITY_COUNTS, IN_RANGE_LENGTHS, False, False, 4),
        ("relation_in_range", spec.relation_in_range_size, TRAIN_ENTITY_COUNTS, IN_RANGE_LENGTHS, True, False, 5),
        ("supported_ood", spec.ood_size, TRAIN_ENTITY_COUNTS, OOD_LENGTHS, False, False, 6),
        ("relation_ood", spec.relation_ood_size, TRAIN_ENTITY_COUNTS, OOD_LENGTHS, True, False, 7),
        ("entity_heldout", spec.entity_heldout_size, (HELDOUT_ENTITY_COUNT,), IN_RANGE_LENGTHS, False, False, 8),
        ("entity_relation_heldout", spec.entity_relation_heldout_size, (HELDOUT_ENTITY_COUNT,), IN_RANGE_LENGTHS, True, False, 9),
        ("causal_core", spec.causal_size, TRAIN_ENTITY_COUNTS, (2, 3, 4), False, True, 10),
    )
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, size, counts, lengths, relation, causal, offset in configs:
        split_rows[split] = _records(
            split=split,
            size=size,
            counts=counts,
            lengths=lengths,
            rng=random.Random(spec.data_seed + offset * 1_000_003),
            forbidden=forbidden,
            relation_heldout=relation,
            causal=causal,
        )
        _write_split(output_dir / f"{split}.jsonl", split_rows[split])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "data_seed": spec.data_seed,
        "spec": asdict(spec),
        "training_entity_counts": list(TRAIN_ENTITY_COUNTS),
        "heldout_entity_count": HELDOUT_ENTITY_COUNT,
        "training_lengths": list(TRAIN_LENGTHS),
        "heldout_relation_rule": "COPY first-listed opaque object -> last-listed opaque object",
        "external_forbidden_sources": [str(path) for path in forbidden_data_dirs],
        "external_forbidden_fingerprints": external_count,
        "splits": {split: _stats(split_rows[split]) for split in DATA_SPLITS},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_a119h2_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    if split not in DATA_SPLITS:
        raise ValueError(f"unknown A1.19H-H2 split: {split}")
    return [
        json.loads(line)
        for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def audit_a119h2_dataset(data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = {split: load_a119h2_records(data_dir, split) for split in DATA_SPLITS}
    fingerprints = [row["fingerprint"] for rows in splits.values() for row in rows]
    train_like = ("train", "validation", "short_regression", "supported_in_range", "supported_ood", "causal_core")
    relation_like = ("relation_in_range", "relation_ood", "entity_relation_heldout")
    gates = {
        "schema_matches": manifest.get("schema_version") == SCHEMA_VERSION,
        "all_fingerprints_unique": len(fingerprints) == len(set(fingerprints)),
        "train_counts_exact": set(int(row["entity_count"]) for row in splits["train"]) == set(TRAIN_ENTITY_COUNTS),
        "heldout_count_absent_from_training": all(
            int(row["entity_count"]) != HELDOUT_ENTITY_COUNT
            for split in ("train", "validation")
            for row in splits[split]
        ),
        "heldout_splits_are_n5": all(
            int(row["entity_count"]) == HELDOUT_ENTITY_COUNT
            for split in ("entity_heldout", "entity_relation_heldout")
            for row in splits[split]
        ),
        "heldout_relation_absent_from_supported": all(
            tuple((op["family"], op["source"], op["target"])) != _heldout_joint(row["entity_names"])
            for split in train_like
            for row in splits[split]
            for op in row["operations"]
        ),
        "heldout_relation_present_in_relation_splits": all(
            any(
                tuple((op["family"], op["source"], op["target"])) == _heldout_joint(row["entity_names"])
                for op in row["operations"]
            )
            for split in relation_like
            for row in splits[split]
        ),
        "no_fixed_register_names": all(
            not ({"amber", "cobalt", "jade"} & set(row["entity_names"]))
            for rows in splits.values()
            for row in rows
        ),
        "all_operations_well_formed": all(
            op["source"] != op["target"]
            and op["source"] in row["entity_names"]
            and op["target"] in row["entity_names"]
            for rows in splits.values()
            for row in rows
            for op in row["operations"]
        ),
        "all_trajectories_recompute": all(
            execute_operations(row["start_state"], row["operations"])[1:]
            == row["state_trajectory"]
            for rows in splits.values()
            for row in rows
        ),
        "all_answers_recompute": all(
            (row["state_trajectory"][-1] if row["state_trajectory"] else row["start_state"])[row["query_register"]]
            == row["answer"]
            for rows in splits.values()
            for row in rows
        ),
    }
    return {
        "schema_version": "yggdrasil.v2-a1.19h.h2-data-audit.v1",
        "data_dir": str(data_dir),
        "data_seed": int(manifest["data_seed"]),
        "gates": gates,
        "passed": all(gates.values()),
    }
