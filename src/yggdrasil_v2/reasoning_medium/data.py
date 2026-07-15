from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

DATA_SCHEMA_VERSION = "yggdrasil.v2-a.symbolic-state-machine.v4"
GENERATOR_VERSION = "symbolic-state-machine-canonical-2026-07-12"
REGISTER_NAMES = ("amber", "cobalt", "jade")
ANSWER_LABELS = tuple("ABCDEFGHIJ")
HELDOUT_BIGRAMS = {
    ("swap", "copy"),
}


@dataclass(frozen=True)
class DatasetSpec:
    seed: int = 20260712
    train_size: int = 512
    validation_size: int = 128
    test_size: int = 128
    composition_size: int = 128
    length_size: int = 128

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": self.train_size,
            "validation": self.validation_size,
            "test": self.test_size,
            "composition_heldout": self.composition_size,
            "length_heldout": self.length_size,
        }


def _operation_family(operation: dict[str, Any]) -> str:
    return str(operation["family"])


def _has_heldout_bigram(operations: Iterable[dict[str, Any]]) -> bool:
    families = [_operation_family(operation) for operation in operations]
    return any((a, b) in HELDOUT_BIGRAMS for a, b in zip(families, families[1:]))


def _sample_operation(rng: random.Random) -> dict[str, Any]:
    family = rng.choices(
        ("swap", "copy"),
        weights=(1, 1),
        k=1,
    )[0]
    if family == "swap":
        source, target = rng.sample(REGISTER_NAMES, 2)
        if REGISTER_NAMES.index(source) > REGISTER_NAMES.index(target):
            source, target = target, source
        template_id = 0
        text = f"SWAP {source} <-> {target}."
    elif family == "copy":
        source, target = rng.sample(REGISTER_NAMES, 2)
        template_id = 0
        text = f"COPY {source} -> {target}."
    return {
        "family": family,
        "source": source,
        "target": target,
        "template_id": template_id,
        "text": text,
    }


def _sample_operations(rng: random.Random, length: int, require_heldout: bool) -> list[dict[str, Any]]:
    for _ in range(10_000):
        operations = [_sample_operation(rng) for _ in range(length)]
        if _has_heldout_bigram(operations) == require_heldout:
            return operations
    raise RuntimeError("Could not sample an operation sequence for the requested split")


def _apply(
    operation: dict[str, Any],
    state: dict[str, tuple[str, frozenset[int]]],
    step: int,
) -> dict[str, tuple[str, frozenset[int]]]:
    updated = dict(state)
    family = operation["family"]
    if family == "swap":
        source, target = operation["source"], operation["target"]
        source_value, target_value = state[source], state[target]
        updated[source] = (target_value[0], target_value[1] | {step})
        updated[target] = (source_value[0], source_value[1] | {step})
    elif family == "copy":
        source, target = operation["source"], operation["target"]
        source_value = state[source]
        updated[target] = (source_value[0], source_value[1] | {step})
    else:
        raise ValueError(f"Unknown operation family: {family}")
    return updated


def _length_for_split(split: str, rng: random.Random) -> int:
    if split in {"train", "validation", "test"}:
        return rng.randint(2, 3)
    if split == "composition_heldout":
        return rng.randint(3, 4)
    if split == "length_heldout":
        return rng.randint(5, 6)
    raise ValueError(f"Unknown split: {split}")


def _render_example(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    length = _length_for_split(split, rng)
    for _ in range(10_000):
        initial_symbols = rng.sample(ANSWER_LABELS, len(REGISTER_NAMES))
        state = {
            register: (symbol, frozenset())
            for register, symbol in zip(REGISTER_NAMES, initial_symbols)
        }
        operations = _sample_operations(rng, length, require_heldout=split == "composition_heldout")
        reasoning_steps: list[dict[str, Any]] = []
        for step, operation in enumerate(operations, start=1):
            state = _apply(operation, state, step)
            reasoning_steps.append(
                {
                    "step": step,
                    "operation": operation["family"],
                    "amber": state["amber"][0],
                    "cobalt": state["cobalt"][0],
                    "jade": state["jade"][0],
                }
            )
        eligible = [
            register
            for register in REGISTER_NAMES
            if len(state[register][1]) >= min(3, length)
            and max(state[register][1], default=0) >= length - 1
        ]
        if eligible:
            break
    else:
        raise RuntimeError("Could not generate an example with a multi-step target dependency")

    query_register = rng.choice(eligible)
    answer = state[query_register][0]
    operation_text = "\n".join(f"{i}. {op['text']}" for i, op in enumerate(operations, start=1))
    question = (
        "Three registers each contain one symbol. Copying never erases the source. "
        f"Initially amber={initial_symbols[0]}, cobalt={initial_symbols[1]}, and jade={initial_symbols[2]}.\n"
        "Execute the instructions in order:\n"
        f"{operation_text}\n"
        f"Which symbol is in the {query_register} register at the end?"
    )
    trace_lines = [
        f"Start: amber={initial_symbols[0]}, cobalt={initial_symbols[1]}, jade={initial_symbols[2]}.",
        *[
            f"Step {step['step']}: amber={step['amber']}, cobalt={step['cobalt']}, jade={step['jade']}."
            for step in reasoning_steps
        ],
        f"FINAL: {answer}",
    ]
    fingerprint = hashlib.sha256(question.encode("utf-8")).hexdigest()
    families = [_operation_family(operation) for operation in operations]
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "example_id": f"{split}-{index:06d}-{fingerprint[:10]}",
        "fingerprint": fingerprint,
        "split": split,
        "task_family": "three_register_swap_copy_state_tracking",
        "start_state": dict(zip(REGISTER_NAMES, initial_symbols)),
        "operations": operations,
        "question": question,
        "query_register": query_register,
        "answer": answer,
        "answer_index": ANSWER_LABELS.index(answer),
        "answer_labels": list(ANSWER_LABELS),
        "trace_text": "\n".join(trace_lines),
        "reasoning_steps": reasoning_steps,
        "program_length": length,
        "target_dependency_steps": sorted(state[query_register][1]),
        "composition_tags": [
            f"{a}->{b}" for a, b in zip(families, families[1:]) if (a, b) in HELDOUT_BIGRAMS
        ],
    }


def build_dataset(output_dir: Path, spec: DatasetSpec) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    split_stats: dict[str, dict[str, Any]] = {}
    total_examples = 0
    total_tokens_proxy = 0

    for split_index, (split, size) in enumerate(spec.split_sizes().items()):
        rng = random.Random(spec.seed + split_index * 1_000_003)
        records: list[dict[str, Any]] = []
        attempts = 0
        while len(records) < size:
            attempts += 1
            if attempts > size * 100:
                raise RuntimeError(f"Could not generate {size} unique examples for {split}")
            record = _render_example(split, len(records), rng)
            if record["fingerprint"] in seen:
                continue
            seen.add(record["fingerprint"])
            records.append(record)

        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        token_proxy = sum(len(record["question"].split()) + len(record["trace_text"].split()) for record in records)
        split_stats[split] = {
            "examples": len(records),
            "program_length_min": min(record["program_length"] for record in records),
            "program_length_max": max(record["program_length"] for record in records),
            "answer_histogram": {
                label: sum(record["answer"] == label for record in records)
                for label in ANSWER_LABELS
            },
            "dependency_steps_min": min(len(record["target_dependency_steps"]) for record in records),
            "unique_fingerprints": len({record["fingerprint"] for record in records}),
            "whitespace_token_proxy": token_proxy,
        }
        total_examples += len(records)
        total_tokens_proxy += token_proxy

    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": spec.seed,
        "spec": asdict(spec),
        "answer_labels": list(ANSWER_LABELS),
        "splits": split_stats,
        "unique_examples": total_examples,
        "unique_whitespace_token_proxy": total_tokens_proxy,
        "heldout_bigrams": [f"{a}->{b}" for a, b in sorted(HELDOUT_BIGRAMS)],
        "leakage_checks": {
            "cross_split_fingerprint_overlap": 0,
            "trace_present_in_question": False,
            "answer_field_present_in_model_input": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
