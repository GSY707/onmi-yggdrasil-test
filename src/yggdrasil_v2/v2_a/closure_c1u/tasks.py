from __future__ import annotations

"""Causal-factorial ERE/CPS task construction for C1U."""

import hashlib
import random
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence

from yggdrasil_v2.r1_revalidation.common.simulator import evaluate_cps, simulate_ere

from . import contract
from .cards import validate_public_cards


LABELS = tuple("ABCDEFGHI")
_VALUE_LEGEND_RE = re.compile(
    r"^Value legend: (?P<zero>\S+) means bit 0; (?P<one>\S+) means bit 1\.$"
)
_CHOICE_RE = re.compile(r"(?P<label>[A-I]) means (?P<meaning>[^|]+)")


def _symbol(seed: int, role: str) -> str:
    digest = hashlib.sha256(f"{seed}\0{role}".encode("utf-8")).hexdigest()[:10]
    return f"n{digest}"


def _mapping(group_index: int, meanings: Sequence[str]) -> dict[str, str]:
    if len(set(meanings)) != len(meanings):
        raise ValueError("label meanings must be unique")
    start = (group_index * 2) % len(LABELS)
    labels = [LABELS[(start + offset * 3) % len(LABELS)] for offset in range(len(meanings))]
    if len(set(labels)) != len(labels):
        raise RuntimeError("deterministic label schedule collided")
    return {label: meaning for label, meaning in zip(labels, meanings, strict=True)}


def _answer_fields(mapping: Mapping[str, str], semantic: str) -> tuple[int, list[bool], str]:
    labels = [label for label, meaning in mapping.items() if meaning == semantic]
    if len(labels) != 1:
        raise ValueError("semantic answer must occur exactly once in label mapping")
    label = labels[0]
    valid = [candidate in mapping for candidate in LABELS]
    return LABELS.index(label), valid, label


def _permutation(seed: int, names: Sequence[str]) -> list[int]:
    order = list(range(len(names)))
    random.Random(seed).shuffle(order)
    return order


def _ere_ast(seed: int, left: int, right: int, target_initial: int) -> tuple[dict[str, Any], dict[str, str]]:
    names = {
        "left": _symbol(seed, "ere-left"),
        "right": _symbol(seed, "ere-right"),
        "target": _symbol(seed, "ere-target"),
        "attribute": _symbol(seed, "ere-bit"),
        "zero": _symbol(seed, "ere-zero"),
        "one": _symbol(seed, "ere-one"),
        "copy_rule": _symbol(seed, "ere-copy-rule"),
        "xor_rule": _symbol(seed, "ere-xor-rule"),
    }
    bit = names["attribute"]
    zero, one = names["zero"], names["one"]
    value = lambda item: one if item else zero
    state = {
        "attributes": {
            names["left"]: {bit: value(left)},
            names["right"]: {bit: value(right)},
            names["target"]: {bit: value(target_initial)},
        },
        "relations": {},
        "facts": [],
        "resources": {},
    }
    copy_rule = {
        "params": ["source", "target"],
        "primitives": [
            {
                "op": "COPY",
                "source": "$arg:source",
                "target": "$arg:target",
                "attribute": bit,
            }
        ],
    }
    xor_primitive = {
        "op": "IF",
        "predicate": {
            "kind": "attribute_equals",
            "entity": "$arg:target",
            "attribute": bit,
            "value": one,
        },
        "then": {
            "op": "IF",
            "predicate": {
                "kind": "attribute_equals",
                "entity": "$arg:source",
                "attribute": bit,
                "value": one,
            },
            "then": {"op": "SET", "target": "$arg:target", "attribute": bit, "value": zero},
            "else": {"op": "SET", "target": "$arg:target", "attribute": bit, "value": one},
        },
        "else": {
            "op": "COPY",
            "source": "$arg:source",
            "target": "$arg:target",
            "attribute": bit,
        },
    }
    ast = {
        "initial_state": state,
        "rules": {
            names["copy_rule"]: copy_rule,
            names["xor_rule"]: {"params": ["source", "target"], "primitives": [xor_primitive]},
        },
        "events": [
            {
                "rule": names["copy_rule"],
                "arguments": {"source": names["left"], "target": names["target"]},
            },
            {
                "rule": names["xor_rule"],
                "arguments": {"source": names["right"], "target": names["target"]},
            },
        ],
        "query": {"kind": "attribute", "entity": names["target"], "attribute": bit},
    }
    return ast, names


def _ere_record(group_index: int, left: int, right: int, *, seed: int) -> dict[str, Any]:
    group_seed = seed + group_index * 1009
    target_initial = group_index % 2
    ast, names = _ere_ast(group_seed, left, right, target_initial)
    output = simulate_ere(ast)
    expected = names["one"] if left ^ right else names["zero"]
    if output.get("answer") != expected:
        raise RuntimeError("ERE XOR simulator postcondition failed")
    mapping = _mapping(group_index, (names["zero"], names["one"]))
    answer_index, valid_mask, raw_label = _answer_fields(mapping, expected)
    semantic_order = [names["left"], names["right"], names["target"]]
    order = _permutation(group_seed + 17, semantic_order)
    object_addresses = [semantic_order[index] for index in order]
    values = {
        names["left"]: names["one"] if left else names["zero"],
        names["right"]: names["one"] if right else names["zero"],
        names["target"]: names["one"] if target_initial else names["zero"],
    }
    object_cards = [
        "\n".join(
            (
                "PUBLIC OBJECT CARD",
                f"Address: {address}",
                f"Local attribute {names['attribute']}: {values[address]}",
            )
        )
        for address in object_addresses
    ]
    operation_cards = [
        "\n".join(
            (
                "PUBLIC OPERATION CARD",
                "Step: 1",
                f"Source address: {names['left']}",
                f"Target address: {names['target']}",
                f"Copy attribute {names['attribute']} from source to target.",
            )
        ),
        "\n".join(
            (
                "PUBLIC OPERATION CARD",
                "Step: 2",
                f"Source address: {names['right']}",
                f"Target address: {names['target']}",
                f"Value legend: {names['zero']} means bit 0; {names['one']} means bit 1.",
                f"Replace target attribute {names['attribute']} with XOR(source,target).",
            )
        ),
    ]
    legend = " | ".join(f"{label} means {meaning}" for label, meaning in sorted(mapping.items()))
    query_card = "\n".join(
        (
            "PUBLIC QUERY CARD",
            f"Query address: {names['target']}",
            f"Return final attribute {names['attribute']}.",
            f"Value legend: {names['zero']} means bit 0; {names['one']} means bit 1.",
            f"Choices: {legend}",
        )
    )
    public = {
        "object_cards": object_cards,
        "object_addresses": object_addresses,
        "operation_cards": operation_cards,
        "operation_source_addresses": [names["left"], names["right"]],
        "operation_target_addresses": [names["target"], names["target"]],
        "query_card": query_card,
        "query_address": names["target"],
    }
    validate_public_cards(public)
    slots = {address: index for index, address in enumerate(object_addresses)}
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.task-record.v1",
        "example_id": f"c1u-ere-g{group_index:02d}-{left}{right}",
        "factorial_group_id": f"c1u-ere-g{group_index:02d}",
        "family": "ERE",
        "factors": [left, right],
        "factor_names": [names["left"], names["right"]],
        "model_public": public,
        "program_ast": ast,
        "simulator_output": output,
        "semantic_answer": expected,
        "raw_answer_label": raw_label,
        "answer_index": answer_index,
        "valid_choice_mask": valid_mask,
        "label_mapping": mapping,
        "support_slots": [slots[names["left"]], slots[names["right"]]],
        "task_causal_arity": 2,
        "target_initial_factor_independent": True,
    }


def _cps_ast(seed: int, left: int, right: int) -> tuple[dict[str, Any], dict[str, str]]:
    names = {
        "left": _symbol(seed, "cps-left"),
        "right": _symbol(seed, "cps-right"),
        "decision": _symbol(seed, "cps-decision"),
        "left_action": _symbol(seed, "cps-left-action"),
        "right_action": _symbol(seed, "cps-right-action"),
        "left_fact": _symbol(seed, "cps-left-fact"),
        "right_fact": _symbol(seed, "cps-right-fact"),
        "goal": _symbol(seed, "cps-goal"),
    }
    facts = []
    if left:
        facts.append(names["left_fact"])
    if right:
        facts.append(names["right_fact"])
    ast = {
        "initial_state": {"attributes": {}, "relations": {}, "facts": facts, "resources": {}},
        "actions": [
            {
                "name": names["left_action"],
                "cost": 1,
                "preconditions": [{"kind": "fact_true", "fact": names["left_fact"]}],
                "effects": [{"kind": "add_fact", "fact": names["goal"]}],
            },
            {
                "name": names["right_action"],
                "cost": 1,
                "preconditions": [{"kind": "fact_true", "fact": names["right_fact"]}],
                "effects": [{"kind": "add_fact", "fact": names["goal"]}],
            },
        ],
        "candidates": [
            {"plan": [names["left_action"]]},
            {"plan": [names["right_action"]]},
        ],
        "budget": 1,
        "goal": [{"kind": "fact_true", "fact": names["goal"]}],
        "final_constraints": [],
    }
    return ast, names


def _cps_record(group_index: int, left: int, right: int, *, seed: int) -> dict[str, Any]:
    group_seed = seed + 500_000 + group_index * 1009
    ast, names = _cps_ast(group_seed, left, right)
    output = evaluate_cps(ast)
    answer = output.get("answer")
    semantic = "none" if answer is None else f"candidate:{int(answer)}"
    expected = "none" if left == right else ("candidate:0" if left else "candidate:1")
    if semantic != expected:
        raise RuntimeError("CPS parity simulator postcondition failed")
    mapping = _mapping(group_index + 4, ("candidate:0", "candidate:1", "none"))
    answer_index, valid_mask, raw_label = _answer_fields(mapping, semantic)
    semantic_order = [names["left"], names["right"], names["decision"]]
    order = _permutation(group_seed + 17, semantic_order)
    object_addresses = [semantic_order[index] for index in order]
    available = {names["left"]: bool(left), names["right"]: bool(right)}
    action = {names["left"]: names["left_action"], names["right"]: names["right_action"]}
    prerequisite = {names["left"]: names["left_fact"], names["right"]: names["right_fact"]}
    object_cards: list[str] = []
    for address in object_addresses:
        if address == names["decision"]:
            object_cards.append(
                "\n".join(
                    (
                        "PUBLIC DECISION OBJECT CARD",
                        f"Address: {address}",
                        "State: no candidate has been evaluated.",
                        f"Goal fact: {names['goal']}",
                        "Budget: 1",
                    )
                )
            )
        else:
            object_cards.append(
                "\n".join(
                    (
                        "PUBLIC CANDIDATE OBJECT CARD",
                        f"Address: {address}",
                        f"Local prerequisite {prerequisite[address]} present: {'yes' if available[address] else 'no'}",
                        f"Plan: {action[address]}",
                        "Cost: 1",
                        f"Effect when legal: add goal fact {names['goal']}",
                    )
                )
            )
    operation_cards = [
        "\n".join(
            (
                "PUBLIC OPERATION CARD",
                f"Step: {index + 1}",
                f"Candidate index: {index}",
                f"Source address: {source}",
                f"Target address: {names['decision']}",
                "Evaluate this candidate and merge validity/cost into the decision state.",
            )
        )
        for index, source in enumerate((names["left"], names["right"]))
    ]
    legend = " | ".join(f"{label} means {meaning}" for label, meaning in sorted(mapping.items()))
    query_card = "\n".join(
        (
            "PUBLIC QUERY CARD",
            f"Query address: {names['decision']}",
            "Choose the unique cheapest valid candidate, or NONE.",
            f"Choices: {legend}",
        )
    )
    public = {
        "object_cards": object_cards,
        "object_addresses": object_addresses,
        "operation_cards": operation_cards,
        "operation_source_addresses": [names["left"], names["right"]],
        "operation_target_addresses": [names["decision"], names["decision"]],
        "query_card": query_card,
        "query_address": names["decision"],
    }
    validate_public_cards(public)
    slots = {address: index for index, address in enumerate(object_addresses)}
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.task-record.v1",
        "example_id": f"c1u-cps-g{group_index:02d}-{left}{right}",
        "factorial_group_id": f"c1u-cps-g{group_index:02d}",
        "family": "CPS",
        "factors": [left, right],
        "factor_names": [names["left"], names["right"]],
        "model_public": public,
        "program_ast": ast,
        "simulator_output": output,
        "semantic_answer": semantic,
        "raw_answer_label": raw_label,
        "answer_index": answer_index,
        "valid_choice_mask": valid_mask,
        "label_mapping": mapping,
        "support_slots": [slots[names["left"]], slots[names["right"]]],
        "task_causal_arity": 2,
    }


def build_factorial_group(family: str, group_index: int, *, seed: int = contract.TASK_SEED) -> list[dict[str, Any]]:
    family = str(family).upper()
    if family not in contract.FAMILIES:
        raise ValueError(f"unsupported C1U family: {family}")
    if type(group_index) is not int or group_index < 0:
        raise ValueError("group_index must be a nonnegative integer")
    builder = _ere_record if family == "ERE" else _cps_record
    rows = [builder(group_index, left, right, seed=seed) for left, right in contract.FACTORIAL_CELLS]
    audit = audit_factorial_group(rows)
    if not audit["passed"]:
        raise RuntimeError(f"constructed factorial group failed its own audit: {audit}")
    return rows


def _public_invariants(record: Mapping[str, Any]) -> dict[str, Any]:
    public = record["model_public"]
    return {
        "object_addresses": list(public["object_addresses"]),
        "operation_cards": list(public["operation_cards"]),
        "operation_source_addresses": list(public["operation_source_addresses"]),
        "operation_target_addresses": list(public["operation_target_addresses"]),
        "query_card": public["query_card"],
        "query_address": public["query_address"],
        "label_mapping": dict(record["label_mapping"]),
        "valid_choice_mask": list(record["valid_choice_mask"]),
    }


def _simulator_replay_answer(record: Mapping[str, Any]) -> str:
    family = str(record["family"]).upper()
    if family == "ERE":
        output = simulate_ere(record["program_ast"])
        return str(output["answer"])
    output = evaluate_cps(record["program_ast"])
    return "none" if output.get("answer") is None else f"candidate:{int(output['answer'])}"


def _line_value(card: str, prefix: str) -> str:
    rows = [line[len(prefix) :].strip() for line in card.splitlines() if line.startswith(prefix)]
    if len(rows) != 1 or not rows[0]:
        raise ValueError(f"public card requires exactly one {prefix!r} line")
    return rows[0]


def _parse_choices(query_card: str) -> dict[str, str]:
    line = _line_value(query_card, "Choices:")
    rows = {
        match.group("label"): match.group("meaning").strip()
        for match in _CHOICE_RE.finditer(line)
    }
    if len(rows) < 2 or len(set(rows.values())) != len(rows):
        raise ValueError("public query choices are incomplete or ambiguous")
    return rows


def _parse_value_legend(card: str) -> tuple[str, str]:
    rows = [line.strip() for line in card.splitlines() if line.startswith("Value legend:")]
    if len(rows) != 1:
        raise ValueError("ERE public card requires exactly one value legend")
    match = _VALUE_LEGEND_RE.fullmatch(rows[0])
    if match is None or match.group("zero") == match.group("one"):
        raise ValueError("ERE public value legend is malformed or ambiguous")
    return match.group("zero"), match.group("one")


def _parse_object_values(public: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for card in public["object_cards"]:
        address = _line_value(str(card), "Address:")
        value_lines = [
            line.split(":", 1)[1].strip()
            for line in str(card).splitlines()
            if line.startswith("Local attribute ") and ":" in line
        ]
        if len(value_lines) != 1 or address in values:
            raise ValueError("ERE object card lacks a unique local value")
        values[address] = value_lines[0]
    return values


def replay_public_answer(record: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the answer from public card text only, never AST/factors/targets."""

    public = record["model_public"]
    operation_cards = [str(card) for card in public["operation_cards"]]
    if len(operation_cards) != 2:
        raise ValueError("public replay requires exactly two operations")
    choices = _parse_choices(str(public["query_card"]))
    object_cards = [str(card) for card in public["object_cards"]]
    candidate_cards = [
        card for card in object_cards if card.startswith("PUBLIC CANDIDATE OBJECT CARD")
    ]
    ere_cards = [card for card in object_cards if card.startswith("PUBLIC OBJECT CARD")]
    if len(candidate_cards) == 2 and not ere_cards:
        family = "CPS"
    elif len(ere_cards) == 3 and not candidate_cards:
        family = "ERE"
    else:
        raise ValueError("public cards do not identify exactly one registered task grammar")
    if family == "ERE":
        operation_legend = _parse_value_legend(operation_cards[1])
        query_legend = _parse_value_legend(str(public["query_card"]))
        if operation_legend != query_legend:
            raise ValueError("ERE operation/query value legends conflict")
        zero, one = operation_legend
        values = _parse_object_values(public)
        first_source = _line_value(operation_cards[0], "Source address:")
        second_source = _line_value(operation_cards[1], "Source address:")
        target = _line_value(operation_cards[0], "Target address:")
        if _line_value(operation_cards[1], "Target address:") != target:
            raise ValueError("ERE operations do not share one public target")
        if first_source not in values or second_source not in values or target not in values:
            raise ValueError("ERE operation address is absent from public objects")
        to_bit = {zero: 0, one: 1}
        if values[first_source] not in to_bit or values[second_source] not in to_bit:
            raise ValueError("ERE object value is absent from the public value legend")
        semantic = (zero, one)[to_bit[values[first_source]] ^ to_bit[values[second_source]]]
    elif family == "CPS":
        availability: dict[str, bool] = {}
        for text in object_cards:
            if not text.startswith("PUBLIC CANDIDATE OBJECT CARD"):
                continue
            address = _line_value(text, "Address:")
            present = [
                line.rsplit(":", 1)[1].strip()
                for line in text.splitlines()
                if line.startswith("Local prerequisite ") and " present:" in line
            ]
            if len(present) != 1 or present[0] not in {"yes", "no"} or address in availability:
                raise ValueError("CPS candidate card lacks unique public availability")
            availability[address] = present[0] == "yes"
        indexed: dict[int, str] = {}
        for card in operation_cards:
            index_text = _line_value(card, "Candidate index:")
            if index_text not in {"0", "1"}:
                raise ValueError("CPS candidate index must be 0 or 1")
            index = int(index_text)
            source = _line_value(card, "Source address:")
            if index in indexed or source not in availability:
                raise ValueError("CPS candidate index is duplicated or ungrounded")
            indexed[index] = source
        if set(indexed) != {0, 1}:
            raise ValueError("CPS public cards must ground candidate indices 0 and 1")
        valid = [index for index, address in sorted(indexed.items()) if availability[address]]
        semantic = f"candidate:{valid[0]}" if len(valid) == 1 else "none"
    else:
        raise ValueError(f"unsupported public replay family: {family}")
    labels = [label for label, meaning in choices.items() if meaning == semantic]
    if len(labels) != 1:
        raise ValueError("public semantic answer has no unique raw label")
    return {"semantic_answer": semantic, "raw_answer_label": labels[0]}


def audit_public_identifiability(record: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        replay = replay_public_answer(record)
        checks["semantic_answer"] = replay["semantic_answer"] == record.get("semantic_answer")
        checks["raw_answer_label"] = replay["raw_answer_label"] == record.get("raw_answer_label")
        checks["answer_index"] = (
            LABELS.index(str(replay["raw_answer_label"])) == int(record.get("answer_index", -1))
        )
    except Exception as exc:
        checks["semantic_answer"] = False
        checks["raw_answer_label"] = False
        checks["answer_index"] = False
        errors.append(f"{type(exc).__name__}:{exc}")
    return {"passed": all(checks.values()), "checks": checks, "errors": errors}


def _drop_prefixed_line(card: str, prefix: str) -> str:
    return "\n".join(line for line in card.splitlines() if not line.startswith(prefix))


def _replace_prefixed_line(card: str, prefix: str, replacement: str) -> str:
    rows = [replacement if line.startswith(prefix) else line for line in card.splitlines()]
    return "\n".join(rows)


def audit_bridge_fault_registry(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove that missing, swapped, or conflicting public bridges fail closed."""

    ere = deepcopy(next(row for row in records if row["family"] == "ERE" and row["factors"] == [0, 1]))
    cps = deepcopy(next(row for row in records if row["family"] == "CPS" and row["factors"] == [1, 0]))
    faults: dict[str, Mapping[str, Any]] = {}

    missing_ere = deepcopy(ere)
    missing_ere["model_public"]["operation_cards"][1] = _drop_prefixed_line(
        missing_ere["model_public"]["operation_cards"][1], "Value legend:"
    )
    faults["ere_missing_value_legend"] = missing_ere

    conflict_ere = deepcopy(ere)
    zero, one = _parse_value_legend(conflict_ere["model_public"]["query_card"])
    conflict_ere["model_public"]["query_card"] = _replace_prefixed_line(
        conflict_ere["model_public"]["query_card"],
        "Value legend:",
        f"Value legend: {one} means bit 0; {zero} means bit 1.",
    )
    faults["ere_conflicting_value_legend"] = conflict_ere

    swapped_ere = deepcopy(ere)
    swapped_line = f"Value legend: {one} means bit 0; {zero} means bit 1."
    swapped_ere["model_public"]["operation_cards"][1] = _replace_prefixed_line(
        swapped_ere["model_public"]["operation_cards"][1], "Value legend:", swapped_line
    )
    swapped_ere["model_public"]["query_card"] = _replace_prefixed_line(
        swapped_ere["model_public"]["query_card"], "Value legend:", swapped_line
    )
    faults["ere_swapped_value_semantics"] = swapped_ere

    missing_cps = deepcopy(cps)
    missing_cps["model_public"]["operation_cards"][0] = _drop_prefixed_line(
        missing_cps["model_public"]["operation_cards"][0], "Candidate index:"
    )
    faults["cps_missing_candidate_index"] = missing_cps

    duplicate_cps = deepcopy(cps)
    duplicate_cps["model_public"]["operation_cards"][1] = _replace_prefixed_line(
        duplicate_cps["model_public"]["operation_cards"][1],
        "Candidate index:",
        "Candidate index: 0",
    )
    faults["cps_duplicate_candidate_index"] = duplicate_cps

    swapped_cps = deepcopy(cps)
    swapped_cps["model_public"]["operation_cards"][0] = _replace_prefixed_line(
        swapped_cps["model_public"]["operation_cards"][0],
        "Candidate index:",
        "Candidate index: 1",
    )
    swapped_cps["model_public"]["operation_cards"][1] = _replace_prefixed_line(
        swapped_cps["model_public"]["operation_cards"][1],
        "Candidate index:",
        "Candidate index: 0",
    )
    faults["cps_swapped_candidate_semantics"] = swapped_cps

    killed = {name: not audit_public_identifiability(row)["passed"] for name, row in faults.items()}
    return {
        "passed": bool(killed) and all(killed.values()),
        "registered": len(killed),
        "killed": sum(killed.values()),
        "faults": killed,
    }


def audit_factorial_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    if len(rows) != 4:
        return {"passed": False, "checks": {"four_cells": False}, "errors": ["factorial group requires four rows"]}
    families = {str(row.get("family", "")).upper() for row in rows}
    groups = {str(row.get("factorial_group_id", "")) for row in rows}
    cells = {tuple(row.get("factors", ())) for row in rows}
    checks["one_family"] = len(families) == 1 and next(iter(families), None) in contract.FAMILIES
    checks["one_group"] = len(groups) == 1 and next(iter(groups), "") != ""
    checks["four_cells"] = cells == set(contract.FACTORIAL_CELLS)
    checks["arity_declared_two"] = all(row.get("task_causal_arity") == 2 for row in rows)
    simulator_errors: list[str] = []
    public_errors: list[str] = []
    for row in rows:
        try:
            validate_public_cards(row["model_public"])
        except Exception as exc:
            public_errors.append(
                f"public validation:{row.get('example_id')}:{type(exc).__name__}:{exc}"
            )
        try:
            if _simulator_replay_answer(row) != row.get("semantic_answer"):
                simulator_errors.append(f"simulator replay mismatch: {row.get('example_id')}")
        except Exception as exc:
            simulator_errors.append(
                f"simulator replay:{row.get('example_id')}:{type(exc).__name__}:{exc}"
            )
        public_audit = audit_public_identifiability(row)
        if not public_audit["passed"]:
            public_errors.append(
                f"public replay mismatch: {row.get('example_id')}:{public_audit['errors']}"
            )
    checks["simulator_replay"] = not simulator_errors
    checks["public_only_replay"] = not public_errors
    errors.extend(simulator_errors)
    errors.extend(public_errors)
    base_invariant = _public_invariants(rows[0])
    checks["public_invariants"] = all(_public_invariants(row) == base_invariant for row in rows[1:])
    by_cell = {tuple(row["factors"]): row for row in rows}
    isolated = True
    all_flips = True
    for cell, row in by_cell.items():
        for factor in (0, 1):
            other_cell = list(cell)
            other_cell[factor] = 1 - other_cell[factor]
            other = by_cell[tuple(other_cell)]
            if row["semantic_answer"] == other["semantic_answer"]:
                all_flips = False
            left_cards = row["model_public"]["object_cards"]
            right_cards = other["model_public"]["object_cards"]
            changed = [index for index, (left, right) in enumerate(zip(left_cards, right_cards, strict=True)) if left != right]
            expected = int(row["support_slots"][factor])
            if changed != [expected]:
                isolated = False
    checks["each_single_factor_flips_answer"] = all_flips
    checks["each_flip_changes_only_its_object_card"] = isolated
    checks["support_slots_stable"] = all(row["support_slots"] == rows[0]["support_slots"] for row in rows)
    checks["answer_fields_consistent"] = all(
        LABELS[int(row["answer_index"])] == row["raw_answer_label"]
        and row["label_mapping"][row["raw_answer_label"]] == row["semantic_answer"]
        for row in rows
    )
    passed = all(checks.values()) and not errors
    return {
        "passed": passed,
        "family": next(iter(families), None),
        "factorial_group_id": next(iter(groups), None),
        "checks": checks,
        "errors": errors,
        "cells": [list(cell) for cell in sorted(cells)],
        "semantic_answers": {"".join(map(str, row["factors"])): row["semantic_answer"] for row in rows},
        "support_slots": list(rows[0]["support_slots"]),
    }


def build_s1_bank(*, seed: int = contract.TASK_SEED) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in contract.FAMILIES:
        for group_index in range(contract.BANK_GROUPS_PER_FAMILY):
            records.extend(build_factorial_group(family, group_index, seed=seed))
    if len(records) != contract.BANK_RECORDS:
        raise RuntimeError("registered bank cardinality drifted")
    return records


def audit_s1_bank(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row.get("factorial_group_id", "")), []).append(row)
    audits = {group: audit_factorial_group(rows) for group, rows in sorted(grouped.items())}
    family_counts = {
        family: sum(str(row.get("family", "")).upper() == family for row in records)
        for family in contract.FAMILIES
    }
    answer_histogram = {label: 0 for label in LABELS}
    for row in records:
        answer_histogram[LABELS[int(row["answer_index"])]] += 1
    public_audits = {
        str(row.get("example_id")): audit_public_identifiability(row) for row in records
    }
    fault_registry = audit_bridge_fault_registry(records) if records else {"passed": False}
    checks = {
        "record_count": len(records) == contract.BANK_RECORDS,
        "group_count": len(grouped)
        == contract.BANK_GROUPS_PER_FAMILY * len(contract.FAMILIES),
        "family_counts": all(
            value == contract.BANK_RECORDS_PER_FAMILY for value in family_counts.values()
        ),
        "all_groups": bool(audits) and all(value["passed"] for value in audits.values()),
        "unique_example_ids": len({str(row.get("example_id")) for row in records}) == len(records),
        "public_only_replay": bool(public_audits)
        and all(value["passed"] for value in public_audits.values()),
        "bridge_fault_registry": bool(fault_registry.get("passed")),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "records": len(records),
        "groups": len(grouped),
        "family_counts": family_counts,
        "answer_histogram": answer_histogram,
        "group_audits": audits,
        "public_identifiability": public_audits,
        "bridge_fault_registry": fault_registry,
    }


__all__ = [
    "LABELS",
    "audit_bridge_fault_registry",
    "audit_factorial_group",
    "audit_public_identifiability",
    "audit_s1_bank",
    "build_factorial_group",
    "build_s1_bank",
    "replay_public_answer",
]
