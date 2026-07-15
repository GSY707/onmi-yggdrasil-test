from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

A15_SCHEMA_VERSION = "yggdrasil.v2-a1.5.symbolic-state-machine.v1"
A15_GENERATOR_VERSION = "a1.5-necessary-step-machine-2026-07-13"
REGISTER_NAMES = ("amber", "cobalt", "jade")
ANSWER_LABELS = tuple("ABCDEFGHIJ")
OPERATION_FAMILIES = ("swap", "copy")
HELDOUT_BIGRAMS = {("swap", "copy")}


@dataclass(frozen=True)
class A15DatasetSpec:
    seed: int = 20260713
    train_size: int = 4096
    validation_size: int = 256
    test_size: int = 256
    composition_size: int = 256
    length_size: int = 256
    train_min_length: int = 1
    train_max_length: int = 4
    # A composition bigram needs at least two operations; ordinary test uses
    # the same 2--4 range so composition is not confounded with length.
    eval_min_length: int = 2
    eval_max_length: int = 4
    length_min_length: int = 5
    length_max_length: int = 6


def _has_heldout_bigram(operations: Sequence[dict[str, Any]]) -> bool:
    families = [str(operation["family"]) for operation in operations]
    return any(pair in HELDOUT_BIGRAMS for pair in zip(families, families[1:]))


def _sample_operation(
    rng: random.Random,
    *,
    copy_bias: float = 0.5,
    family: str | None = None,
) -> dict[str, Any]:
    family = family or ("copy" if rng.random() < copy_bias else "swap")
    source, target = rng.sample(REGISTER_NAMES, 2)
    if family == "swap" and REGISTER_NAMES.index(source) > REGISTER_NAMES.index(target):
        source, target = target, source
    if family == "swap":
        text = f"SWAP {source} <-> {target}."
    else:
        text = f"COPY {source} -> {target}."
    return {
        "family": family,
        "source": source,
        "target": target,
        "text": text,
        "template_id": 0,
    }


def _sample_operations(
    rng: random.Random,
    length: int,
    *,
    require_heldout: bool | None,
    copy_bias: float = 0.5,
    desired_copy_count: int | None = None,
) -> list[dict[str, Any]]:
    for _ in range(20_000):
        if desired_copy_count is None:
            operations = [_sample_operation(rng, copy_bias=copy_bias) for _ in range(length)]
        else:
            if desired_copy_count < 0 or desired_copy_count > length:
                raise ValueError("desired_copy_count must be within the operation length")
            if desired_copy_count == 1 and length >= 5 and require_heldout is False:
                # A no-swap->copy length control is possible but rare under
                # unrestricted placement. Put the sole COPY first, then let
                # SWAP carry the remaining necessary dependency chain.
                copy_positions = {0}
            else:
                copy_positions = set(rng.sample(range(length), desired_copy_count))
            operations = [
                _sample_operation(rng, family="copy" if index in copy_positions else "swap")
                for index in range(length)
            ]
        if require_heldout is None or _has_heldout_bigram(operations) == require_heldout:
            return operations
    raise RuntimeError(f"could not sample operation sequence length={length} heldout={require_heldout}")


def execute_operations(
    start_state: dict[str, str],
    operations: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return the state after every operation; step zero is the input state."""
    state = dict(start_state)
    trajectory = [dict(state)]
    for operation in operations:
        family = operation["family"]
        if family == "noop":
            pass
        elif family == "swap":
            source, target = operation["source"], operation["target"]
            state[source], state[target] = state[target], state[source]
        elif family == "copy":
            state[operation["target"]] = state[operation["source"]]
        else:
            raise ValueError(f"unknown operation family: {family}")
        trajectory.append(dict(state))
    return trajectory


def _render_question(
    start_state: dict[str, str],
    operations: Sequence[dict[str, Any]],
    query_register: str,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    prefix = (
        "Three registers each contain one symbol. Execute the operations in order. "
        f"Initially amber={start_state['amber']}, cobalt={start_state['cobalt']}, jade={start_state['jade']}.\n"
        "Operations:\n"
    )
    pieces = [prefix]
    spans: list[dict[str, Any]] = []
    tokens = ["<START>", *[f"{name}={start_state[name]}" for name in REGISTER_NAMES], "<OPS>"]
    cursor = len(prefix)
    for index, operation in enumerate(operations, start=1):
        line = f"{index}. {operation['text']}\n"
        start = cursor
        pieces.append(line)
        cursor += len(line)
        spans.append(
            {
                "step": index,
                "char_start": start,
                "char_end": cursor,
                "text": line.rstrip("\n"),
                "family": operation["family"],
            }
        )
        tokens.extend([f"<OP_{index}>", operation["family"], operation["source"], operation["target"]])
    suffix = f"Which symbol is in the {query_register} register at the end?"
    pieces.append(suffix)
    tokens.extend(["<QUERY>", query_register])
    return "".join(pieces), spans, tokens


def _candidate(
    split: str,
    index: int,
    rng: random.Random,
    *,
    min_length: int,
    max_length: int,
    require_heldout: bool | None,
    forced_length: int | None = None,
    desired_copy_count: int | None = None,
) -> dict[str, Any] | None:
    length = forced_length if forced_length is not None else rng.randint(min_length, max_length)
    start_values = rng.sample(ANSWER_LABELS, len(REGISTER_NAMES))
    start_state = dict(zip(REGISTER_NAMES, start_values))
    operations = _sample_operations(
        rng,
        length,
        require_heldout=require_heldout,
        # Necessary-step filtering naturally favors SWAP; bias proposals to
        # COPY so the accepted corpus does not silently become swap-only.
        copy_bias=(0.56 if length >= 5 else 0.65),
        desired_copy_count=desired_copy_count,
    )
    trajectory = execute_operations(start_state, operations)
    eligible_queries: list[str] = []
    query_counterfactuals: dict[str, tuple[str, list[str], list[int]]] = {}
    for candidate_query in REGISTER_NAMES:
        candidate_answer = trajectory[-1][candidate_query]
        candidate_answers: list[str] = []
        candidate_necessary: list[int] = []
        for step_index in range(length):
            counterfactual_ops = [dict(operation) for operation in operations]
            counterfactual_ops[step_index] = {
                "family": "noop",
                "source": None,
                "target": None,
                "text": "NOOP.",
                "template_id": -1,
            }
            counterfactual_answer = execute_operations(start_state, counterfactual_ops)[-1][candidate_query]
            candidate_answers.append(counterfactual_answer)
            if counterfactual_answer != candidate_answer:
                candidate_necessary.append(step_index + 1)
        if len(candidate_necessary) == length:
            eligible_queries.append(candidate_query)
            query_counterfactuals[candidate_query] = (candidate_answer, candidate_answers, candidate_necessary)
    if not eligible_queries:
        return None
    query_register = rng.choice(eligible_queries)
    answer, counterfactual_answers, necessary_steps = query_counterfactuals[query_register]
    question, operation_spans, structured_tokens = _render_question(start_state, operations, query_register)
    fingerprint = hashlib.sha256(question.encode("utf-8")).hexdigest()
    families = [operation["family"] for operation in operations]
    return {
        "schema_version": A15_SCHEMA_VERSION,
        "generator_version": A15_GENERATOR_VERSION,
        "example_id": f"{split}-{index:06d}-{fingerprint[:10]}",
        "fingerprint": fingerprint,
        "split": split,
        "task_family": "three_register_swap_copy_necessary_state_tracking",
        "start_state": start_state,
        "operations": operations,
        "operation_spans": operation_spans,
        "operation_mask": [1] * length,
        "structured_tokens": structured_tokens,
        "question": question,
        "query_register": query_register,
        "answer": answer,
        "answer_index": ANSWER_LABELS.index(answer),
        "answer_labels": list(ANSWER_LABELS),
        "state_trajectory": trajectory[1:],
        "counterfactual_answers": counterfactual_answers,
        "necessary_steps": necessary_steps,
        "all_steps_necessary": True,
        "program_length": length,
        "composition_tags": [
            f"{a}->{b}"
            for a, b in zip(families, families[1:])
            if (a, b) in HELDOUT_BIGRAMS
        ],
    }


def _balanced_pick(
    candidates: Iterable[dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    answer_counts: Counter[str] = Counter()
    query_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for candidate in candidates:
        if len(selected) >= count or candidate["fingerprint"] in seen:
            continue
        answer = str(candidate["answer"])
        query = str(candidate["query_register"])
        length = int(candidate["program_length"])
        family = "".join(operation["family"][0] for operation in candidate["operations"])
        candidate_operation_counts = Counter(operation["family"] for operation in candidate["operations"])
        proposed_imbalance = abs(
            (operation_counts["copy"] + candidate_operation_counts["copy"])
            - (operation_counts["swap"] + candidate_operation_counts["swap"])
        )
        # Greedy stratification keeps small requested splits balanced without a
        # fragile finite Cartesian enumeration.
        score = (
            answer_counts[answer] * 4
            + query_counts[query] * 2
            + length_counts[length] * 2
            + family_counts[family]
            + max(operation_counts.values(), default=0)
            - operation_counts["copy"]
            - operation_counts["swap"]
            + proposed_imbalance * 2
        )
        minimum = min(
            answer_counts[label] * 4 + query_counts[query] * 2 + length_counts[length] * 2 + family_counts[family]
            for label in ANSWER_LABELS
        )
        if selected and score > minimum + 8 and rng.random() < 0.85:
            continue
        selected.append(candidate)
        seen.add(candidate["fingerprint"])
        answer_counts[answer] += 1
        query_counts[query] += 1
        length_counts[length] += 1
        family_counts[family] += 1
        operation_counts.update(operation["family"] for operation in candidate["operations"])
    if len(selected) < count:
        # The stratification guard is advisory, not a reason to make dataset
        # generation fail after a valid candidate reserve was built.
        for candidate in candidates:
            if len(selected) >= count:
                break
            if candidate["fingerprint"] in seen:
                continue
            selected.append(candidate)
            seen.add(candidate["fingerprint"])
    if len(selected) != count:
        raise RuntimeError(f"could not select {count} balanced candidates; selected={len(selected)}")
    return selected


def _generate_split(
    split: str,
    count: int,
    rng: random.Random,
    *,
    min_length: int,
    max_length: int,
    require_heldout: bool | None,
    forbidden_fingerprints: set[str] | None = None,
) -> list[dict[str, Any]]:
    forbidden_fingerprints = set(forbidden_fingerprints or set())
    candidates: list[dict[str, Any]] = []
    # Only a small candidate reserve is needed because selection is stratified
    # online. The previous oversized reserve made a 4k train split needlessly
    # expensive while adding no contract strength.
    target_pool = max(count * 5, 256)
    attempts = 0
    # Force an approximately equal quota for every contract length. This is
    # stronger than merely checking min/max: the P0 positive control must not
    # learn a length-1 shortcut and then face mostly unseen lengths at eval.
    required_lengths = list(range(min_length, max_length + 1))
    forced: list[dict[str, Any]] = []
    base_quota, remainder = divmod(count, len(required_lengths))
    for length_index, forced_length in enumerate(required_lengths):
        quota = base_quota + int(length_index < remainder)
        for _ in range(quota):
            while True:
                attempts += 1
                candidate = _candidate(
                    split,
                    len(candidates),
                    rng,
                    min_length=min_length,
                    max_length=max_length,
                    require_heldout=require_heldout,
                    forced_length=forced_length,
                    desired_copy_count=min(1, forced_length),
                )
                if candidate is not None and candidate["fingerprint"] not in forbidden_fingerprints:
                    candidates.append(candidate)
                    forced.append(candidate)
                    forbidden_fingerprints.add(candidate["fingerprint"])
                    break
                if attempts > target_pool * 200:
                    raise RuntimeError(f"unable to force length {forced_length} for {split}")
    while len(candidates) < target_pool:
        attempts += 1
        if attempts > target_pool * 200:
            raise RuntimeError(f"unable to generate enough necessary-step examples for {split}")
        candidate = _candidate(
            split,
            len(candidates),
            rng,
            min_length=min_length,
            max_length=max_length,
            require_heldout=require_heldout,
            desired_copy_count=1 if min_length >= 5 else None,
        )
        if candidate is not None and candidate["fingerprint"] not in forbidden_fingerprints:
            candidates.append(candidate)
            forbidden_fingerprints.add(candidate["fingerprint"])
    forced_fingerprints = {candidate["fingerprint"] for candidate in forced}
    remaining = [candidate for candidate in candidates if candidate["fingerprint"] not in forced_fingerprints]
    if len(forced) > count:
        raise RuntimeError(f"split {split} is smaller than its required length strata")
    return forced + _balanced_pick(remaining, count - len(forced), rng=rng)


def _split_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(records),
        "program_length_min": min(record["program_length"] for record in records),
        "program_length_max": max(record["program_length"] for record in records),
        "answer_histogram": dict(sorted(Counter(record["answer"] for record in records).items())),
        "query_histogram": dict(sorted(Counter(record["query_register"] for record in records).items())),
        "operation_family_histogram": dict(
            sorted(Counter(op["family"] for record in records for op in record["operations"]).items())
        ),
        "length_histogram": dict(sorted(Counter(record["program_length"] for record in records).items())),
        "necessary_step_rate": sum(record["all_steps_necessary"] for record in records) / len(records),
        "composition_example_rate": sum(bool(record["composition_tags"]) for record in records) / len(records),
    }


def build_a15_dataset(output_dir: Path, spec: A15DatasetSpec) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_config = {
        "train": (spec.train_size, spec.train_min_length, spec.train_max_length, False),
        "validation": (spec.validation_size, spec.eval_min_length, spec.eval_max_length, False),
        "test": (spec.test_size, spec.eval_min_length, spec.eval_max_length, False),
        "composition_heldout": (spec.composition_size, spec.eval_min_length, spec.eval_max_length, True),
        # Length-heldout is strictly 5--6 steps and keeps the heldout family
        # pair out; composition and length are therefore separable.
        "length_heldout": (spec.length_size, spec.length_min_length, spec.length_max_length, False),
    }
    seen: set[str] = set()
    split_stats: dict[str, Any] = {}
    for split_index, (split, (count, min_length, max_length, require_heldout)) in enumerate(split_config.items()):
        rng = random.Random(spec.seed + split_index * 1_000_003)
        records = _generate_split(
            split,
            count,
            rng,
            min_length=min_length,
            max_length=max_length,
            require_heldout=require_heldout,
            forbidden_fingerprints=seen,
        )
        for record in records:
            if record["fingerprint"] in seen:
                raise AssertionError("cross-split fingerprint collision")
            seen.add(record["fingerprint"])
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        split_stats[split] = _split_stats(records)

    manifest = {
        "schema_version": A15_SCHEMA_VERSION,
        "generator_version": A15_GENERATOR_VERSION,
        "seed": spec.seed,
        "spec": asdict(spec),
        "answer_labels": list(ANSWER_LABELS),
        "splits": split_stats,
        "unique_examples": sum(item["examples"] for item in split_stats.values()),
        "contract": {
            "ordinary_and_composition_same_length_range": True,
            "ordinary_and_composition_length_range": [spec.eval_min_length, spec.eval_max_length],
            "length_heldout_range": [spec.length_min_length, spec.length_max_length],
            "train_covers_lengths_1_to_4": spec.train_min_length <= 1 and spec.train_max_length >= 4,
            "all_core_eval_steps_necessary": True,
            "operation_spans_saved": True,
            "operation_masks_saved": True,
            "answer_field_present_in_model_input": False,
        },
        "leakage_checks": {
            "cross_split_fingerprint_overlap": 0,
            "composition_length_mixing": False,
            "length_mixing_in_composition": False,
            "counterfactual_necessary_step_failures": 0,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_a15_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records


def counterfactual_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    total_steps = 0
    necessary_steps = 0
    for record in records:
        operations = record["operations"]
        total_steps += len(operations)
        for index, operation in enumerate(operations):
            no_op = [dict(item) for item in operations]
            no_op[index] = {"family": "noop", "source": None, "target": None, "text": "NOOP."}
            answer = execute_operations(record["start_state"], no_op)[-1][record["query_register"]]
            required = answer != record["answer"]
            necessary_steps += int(required)
            if not required:
                failures.append({"example_id": record["example_id"], "step": index + 1, "answer": answer})
    return {
        "examples": len(records),
        "total_steps": total_steps,
        "necessary_steps": necessary_steps,
        "necessary_rate": necessary_steps / total_steps if total_steps else 0.0,
        "failures": failures,
    }


def _answer_after_prefix(record: dict[str, Any], steps: int) -> str:
    trajectory = execute_operations(record["start_state"], record["operations"][:steps])
    return trajectory[-1][record["query_register"]]


def evaluate_weak_baselines(records: Sequence[dict[str, Any]], train_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    majority = Counter(record["answer"] for record in train_records).most_common(1)[0][0]
    methods = {
        "zero_step": lambda record: _answer_after_prefix(record, 0),
        "prefix_1": lambda record: _answer_after_prefix(record, 1),
        "prefix_2": lambda record: _answer_after_prefix(record, 2),
        "majority": lambda record: majority,
        "initial_query": lambda record: record["start_state"][record["query_register"]],
        "oracle": lambda record: record["answer"],
    }
    result: dict[str, Any] = {}
    for name, predictor in methods.items():
        correct = sum(predictor(record) == record["answer"] for record in records)
        result[name] = {"correct": correct, "total": len(records), "accuracy": correct / len(records)}
    return result


def render_state_labels(record: dict[str, Any]) -> list[list[int]]:
    return [
        [ANSWER_LABELS.index(state[register]) for register in REGISTER_NAMES]
        for state in record["state_trajectory"]
    ]
