from __future__ import annotations

"""Offline C1S successor target materialisation and split bookkeeping.

This module is deliberately outside the model package.  It consumes complete
C0R rows only while compiling an immutable target artifact; the model-facing
provider must pass only the public hidden state and padding mask.  In
particular, ``family``, AST, simulator output, labels and all targets in this
file are loss/evaluator-side data, never forward arguments.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from ..closure_c0r.compact_trace import format_compact_trace, verify_compact_trace
from ..closure_c1.data import source_byte_sha256
from ...r1_revalidation.production.renderer import parse_source
from ...r1_revalidation.common.simulator import evaluate_cps, simulate_ere


TARGET_BANK_SCHEMA = "yggdrasil.v2-a.closure-c1s.srw-target-bank.v2"
SPLIT_LEDGER_SCHEMA = "yggdrasil.v2-a.closure-c1s.split-ledger.v1"
S1_SELECTION_SALT = "C1S-SRW-S1-OVERFIT32-V1"
S2_TRAIN_SELECTION_SALT = "C1S-SRW-S2-DISCOVERY-TRAIN-V1"
S2_EVAL_SELECTION_SALT = "C1S-SRW-S2-DISCOVERY-EVAL-V1"
ADDRESS_POLICY_NAME = "C1S-SRW-PUBLIC-ADDRESS-V1"
LABELS = tuple("ABCDEFGHI")
MAX_CHOICES = 8
MAX_SEMANTIC_OBJECTS = 9
MODEL_STEPS = 10

_CHOICE_HEADINGS = {
    "CHOICES", "LABEL KEY", "RESPONSE KEY",
}
_CHOICE_LINE = re.compile(
    r"^(?:\s*Label\s+)?(?P<label>[A-I])\s+(?P<verb>means|denotes)\s+(?P<meaning>.+?)\s*$",
    re.IGNORECASE,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    if isinstance(value, bytes):
        data = value
    else:
        data = _canonical(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest().upper()


def _target_bank_digest_preimage(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the target-bank payload covered by its top-level digest.

    The digest is deliberately excluded from the preimage so verification is
    not self-referential. Keeping this projection shared by materialisation
    and audit is important: every top-level bank field (including metadata
    such as ``record_count``) is covered, while the generated ``sha256`` field
    is the sole exception.
    """
    return {key: value for key, value in payload.items() if key != "sha256"}


def _target_bank_digest(payload: Mapping[str, Any]) -> str:
    return _sha256(_target_bank_digest_preimage(payload))


def _hash_rank(example_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{example_id}".encode("utf-8")).hexdigest()


def _as_int(value: Any, default: int | None = None) -> int | None:
    if type(value) is int:
        return value
    return default


def _normalise_meaning(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"candidate:{value}"
    text = str(value).strip().strip("'").strip('"')
    upper = text.upper()
    if upper in {"TRUE", "FALSE", "NONE"}:
        return upper.lower()
    candidate = re.fullmatch(r"candidate\s+([1-9][0-9]*)", text, re.IGNORECASE)
    if candidate:
        return f"candidate:{int(candidate.group(1)) - 1}"
    value_match = re.fullmatch(r"value\s+['\"]?([^'\"]+)['\"]?", text, re.IGNORECASE)
    if value_match:
        return f"value:{value_match.group(1)}"
    return text.casefold()


def _normalise_public_entity_name(value: Any) -> str:
    if type(value) is not str or not value:
        raise ValueError("public entity name must be a nonempty string")
    return unicodedata.normalize("NFKC", value).casefold()


def _choice_section_offsets(source_text: str) -> tuple[int, int]:
    lines = source_text.splitlines(keepends=True)
    cursor = 0
    starts: list[tuple[str, int]] = []
    for line in lines:
        starts.append((line.rstrip("\r\n"), cursor))
        cursor += len(line)
    heading = next(((line, start) for line, start in starts if line.strip().upper() in _CHOICE_HEADINGS), None)
    if heading is None:
        return 0, len(source_text)
    _, start = heading
    end = len(source_text)
    return start + len(starts[0][0]) if starts else start, end


def extract_choice_slots(source_text: str, *, max_choices: int = len(LABELS)) -> dict[str, Any]:
    """Extract ordered raw A-I choice assumptions from the public source.

    The parser only treats lines in the CHOICES/LABEL KEY/RESPONSE KEY
    section as candidate declarations.  Slot order is textual order; it is
    not derived from family, AST ordering or semantic answer values.
    """
    if type(source_text) is not str or not source_text:
        raise ValueError("source_text must be a nonempty string")
    lines = source_text.splitlines(keepends=True)
    cursor = 0
    in_choice_section = False
    seen_heading = False
    slots: list[dict[str, Any]] = []
    for line in lines:
        bare = line.rstrip("\r\n")
        upper = bare.strip().upper()
        if upper in _CHOICE_HEADINGS:
            in_choice_section = True
            seen_heading = True
            cursor += len(line)
            continue
        if in_choice_section and upper and upper.isupper() and upper in {
            "TEMPORARY RULES", "RULE DEFINITIONS", "INITIAL WORLD", "WORLD AT THE START",
            "ORDERED EVENTS", "EVENT SEQUENCE", "QUERY", "QUESTION TO ANSWER",
            "CANDIDATE PLANS", "PROPOSED PLANS", "ACTION DEFINITIONS", "TEMPORARY ACTIONS",
            "REQUIRED OUTCOME", "CHOICES", "LABEL KEY", "RESPONSE KEY",
        } and upper not in _CHOICE_HEADINGS:
            in_choice_section = False
        if in_choice_section:
            match = _CHOICE_LINE.match(bare)
            if match:
                label = match.group("label").upper()
                if any(item["label"] == label for item in slots):
                    raise ValueError(f"duplicate raw choice label: {label}")
                if len(slots) >= max_choices:
                    raise ValueError(f"choice count exceeds {max_choices}")
                leading = len(bare) - len(bare.lstrip())
                start = cursor + leading
                slots.append({
                    "label": label,
                    "meaning": match.group("meaning").strip(),
                    "start": start,
                    "end": cursor + len(bare),
                    "text": bare.strip(),
                    "slot": len(slots),
                })
        cursor += len(line)
    if not seen_heading:
        raise ValueError("source has no CHOICES/LABEL KEY/RESPONSE KEY section")
    if not slots:
        raise ValueError("choice section has no raw A-I declarations")
    return {
        "slots": slots,
        "label_to_slot": {item["label"]: item["slot"] for item in slots},
        "slot_to_label": [item["label"] for item in slots],
    }


def _tokenize_offsets(tokenizer: Any, text: str) -> dict[str, Any]:
    if tokenizer is None:
        return {"token_offsets": [], "token_owner": [], "token_count": 0}
    try:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer must return a mapping with input_ids and offset_mapping")
    ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    if not isinstance(ids, Sequence) or not isinstance(offsets, Sequence) or len(ids) != len(offsets):
        raise ValueError("tokenizer returned invalid input_ids/offset_mapping")
    return {
        "token_offsets": [[int(item[0]), int(item[1])] for item in offsets],
        "token_owner": [-1 for _ in ids],
        "token_count": len(ids),
    }


def _resolve_answer_label(record: Mapping[str, Any], mapping: Mapping[str, Any] | None, slots: Sequence[Mapping[str, Any]]) -> str:
    for key in ("answer_label", "target_label"):
        value = record.get(key)
        if isinstance(value, str) and value.upper() in LABELS:
            return value.upper()
    value = record.get("answer")
    if isinstance(value, str) and value.upper() in LABELS:
        return value.upper()
    index = record.get("answer_index")
    if type(index) is int and 0 <= index < len(LABELS):
        return LABELS[index]
    semantic = record.get("semantic_answer")
    if mapping is not None and semantic is not None:
        wanted = _normalise_meaning(semantic)
        matches = [str(label).upper() for label, target in mapping.items() if _normalise_meaning(target) == wanted]
        if len(matches) == 1:
            return matches[0]
    if mapping is not None and record.get("simulator_output"):
        output = record["simulator_output"]
        if isinstance(output, Mapping) and "answer" in output:
            wanted = _normalise_meaning(output["answer"])
            matches = [str(label).upper() for label, target in mapping.items() if _normalise_meaning(target) == wanted]
            if len(matches) == 1:
                return matches[0]
    raise ValueError(f"cannot resolve answer label for {record.get('example_id')}")


def _state_digest(state: Any) -> str:
    return _sha256(state)


def _simulator_and_trace(record: Mapping[str, Any]) -> tuple[Any | None, str | None, dict[str, Any]]:
    family = record.get("family")
    ast = record.get("program_ast")
    mapping = record.get("label_mapping")
    if family not in {"ERE", "CPS"} or not isinstance(ast, Mapping):
        return None, None, {}
    if family == "ERE":
        output = simulate_ere(ast)
    else:
        output = evaluate_cps(ast)
    formatted = format_compact_trace({"family": family, "program_ast": ast, "label_mapping": mapping or {}}) if isinstance(mapping, Mapping) else None
    if formatted is not None:
        replay = verify_compact_trace(trace_text=formatted, program_ast=ast, label_mapping=mapping)
        if not replay.passed:
            raise ValueError(f"compact trace replay failed for {record.get('example_id')}: {replay.errors}")
    return output, formatted, {"simulator_output": output}


def _cps_closure(output: Mapping[str, Any], slots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_candidate = {
        int(item.get("candidate_index", index)): item
        for index, item in enumerate(output.get("candidates", []))
        if isinstance(item, Mapping)
    }
    decisions: list[dict[str, Any]] = []
    for slot in slots:
        meaning = _normalise_meaning(slot["meaning"])
        candidate_index: int | None = None
        if meaning.startswith("candidate:"):
            candidate_index = int(meaning.split(":", 1)[1])
        item = by_candidate.get(candidate_index) if candidate_index is not None else None
        row: dict[str, Any] = {"slot": slot["slot"], "label": slot["label"], "meaning": meaning}
        if item is not None:
            row.update({key: item.get(key) for key in ("candidate_index", "legal", "valid", "total_cost", "budget_ok", "goal_satisfied", "final_constraints_satisfied")})
            trace = item.get("trace", [])
            row["action_trace_audit"] = {
                "steps": len(trace) if isinstance(trace, list) else 0,
                "sha256": _state_digest(trace),
            }
            row["final_state_sha256"] = _state_digest(item.get("final_state"))
        else:
            row["candidate_index"] = candidate_index
        decisions.append(row)
    return {"kind": "cps_candidate_decision", "decisions": decisions}


def _ere_closure(output: Mapping[str, Any], slots: Sequence[Mapping[str, Any]], answer_label: str) -> dict[str, Any]:
    final_state = output.get("final_state")
    return {
        "kind": "ere_candidate_decision",
        "decisions": [
            {
                "slot": slot["slot"],
                "label": slot["label"],
                "meaning": _normalise_meaning(slot["meaning"]),
                "selected": slot["label"] == answer_label,
                "final_state_sha256": _state_digest(final_state),
            }
            for slot in slots
        ],
    }


def _token_bounds_for_chars(token_offsets: Sequence[Sequence[int]], start: int, end: int) -> tuple[int, int]:
    positions = [
        index for index, offset in enumerate(token_offsets)
        if int(offset[1]) > int(start) and int(offset[0]) < int(end)
    ]
    return (-1, -1) if not positions else (min(positions), max(positions) + 1)


def _line_span(source_text: str, pattern: str) -> tuple[int, int]:
    match = re.search(pattern, source_text, flags=re.IGNORECASE | re.MULTILINE)
    return (-1, -1) if match is None else (match.start(), match.end())


def _name_span(source_text: str, name: str) -> tuple[int, int]:
    quoted = re.search(rf"['\"]{re.escape(name)}['\"]", source_text)
    if quoted is not None:
        return quoted.start(), quoted.end()
    match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", source_text)
    return (-1, -1) if match is None else (match.start(), match.end())


def _section_span(source_text: str, headings: Sequence[str], end_headings: Sequence[str]) -> tuple[int, int]:
    lines = list(re.finditer(r"(?m)^.*$", source_text))
    start = next((item.start() for item in lines if item.group(0).strip().upper() in headings), -1)
    if start < 0:
        return -1, -1
    end = next(
        (item.start() for item in lines if item.start() > start and item.group(0).strip().upper() in end_headings),
        len(source_text),
    )
    return start, end


def _label_for_meaning(mapping: Mapping[str, Any] | None, meaning: str) -> str | None:
    if mapping is None:
        return None
    matches = [str(label).upper() for label, value in mapping.items() if _normalise_meaning(value) == meaning]
    return matches[0] if len(matches) == 1 and matches[0] in LABELS else None


def _object_row(
    *, slot: int, kind: str, name: str, raw_label: str | None,
    char_span: tuple[int, int], token_offsets: Sequence[Sequence[int]], content: bool,
) -> dict[str, Any]:
    token_start, token_end = _token_bounds_for_chars(token_offsets, *char_span) if char_span != (-1, -1) else (-1, -1)
    return {
        "slot": slot,
        "kind": kind,
        "name": name,
        "raw_label": raw_label,
        "raw_label_index": LABELS.index(raw_label) if raw_label in LABELS else -1,
        "char_start": int(char_span[0]),
        "char_end": int(char_span[1]),
        "token_start": token_start,
        "token_end": token_end,
        "content": bool(content),
    }


def _state_entity_signature(state: Mapping[str, Any], entity: str) -> tuple[Any, Any]:
    attributes = state.get("attributes", {}) if isinstance(state, Mapping) else {}
    relations = state.get("relations", {}) if isinstance(state, Mapping) else {}
    relation_membership: list[tuple[str, str, str]] = []
    if isinstance(relations, Mapping):
        for relation, pairs in relations.items():
            if not isinstance(pairs, Sequence):
                continue
            for pair in pairs:
                if isinstance(pair, Sequence) and len(pair) == 2 and entity in {str(pair[0]), str(pair[1])}:
                    relation_membership.append((str(relation), str(pair[0]), str(pair[1])))
    entity_attributes = attributes.get(entity, {}) if isinstance(attributes, Mapping) else {}
    return _canonical(entity_attributes), tuple(sorted(relation_membership))


def _resolve_event_operand(value: Any, arguments: Mapping[str, Any]) -> str | None:
    if not isinstance(value, str):
        return None
    return str(arguments.get(value[5:])) if value.startswith("$arg:") and value[5:] in arguments else value


def _primitive_routes(primitive: Mapping[str, Any], arguments: Mapping[str, Any]) -> list[tuple[str, str]]:
    op = primitive.get("op")
    if op == "SET":
        target = _resolve_event_operand(primitive.get("target"), arguments)
        return [(target, target)] if target is not None else []
    if op == "COPY":
        source = _resolve_event_operand(primitive.get("source"), arguments)
        target = _resolve_event_operand(primitive.get("target"), arguments)
        return [(source, target)] if source is not None and target is not None else []
    if op == "SWAP":
        left = _resolve_event_operand(primitive.get("left"), arguments)
        right = _resolve_event_operand(primitive.get("right"), arguments)
        return [(left, right), (right, left)] if left is not None and right is not None else []
    if op in {"LINK", "UNLINK"}:
        source = _resolve_event_operand(primitive.get("source"), arguments)
        target = _resolve_event_operand(primitive.get("target"), arguments)
        return [(source, target)] if source is not None and target is not None else []
    if op == "IF":
        routes: list[tuple[str, str]] = []
        for branch in (primitive.get("then"), primitive.get("else")):
            if isinstance(branch, Mapping):
                routes.extend(_primitive_routes(branch, arguments))
        return routes
    if op == "FOREACH_LINKED":
        source = _resolve_event_operand(primitive.get("source"), arguments)
        return [(source, source)] if source is not None else []
    return []


def _ere_entities(ast: Mapping[str, Any], source_text: str) -> list[str]:
    names: set[str] = set()
    state = ast.get("initial_state", {})
    attributes = state.get("attributes", {}) if isinstance(state, Mapping) else {}
    if isinstance(attributes, Mapping):
        names.update(str(value) for value in attributes)
    relations = state.get("relations", {}) if isinstance(state, Mapping) else {}
    if isinstance(relations, Mapping):
        for pairs in relations.values():
            if isinstance(pairs, Sequence):
                for pair in pairs:
                    if isinstance(pair, Sequence) and len(pair) == 2:
                        names.update((str(pair[0]), str(pair[1])))
    query = ast.get("query", {})
    if isinstance(query, Mapping):
        for key in ("entity", "source", "target"):
            if isinstance(query.get(key), str):
                names.add(str(query[key]))
    events = ast.get("events", [])
    if isinstance(events, Sequence):
        for event in events:
            if isinstance(event, Mapping) and isinstance(event.get("arguments"), Mapping):
                for key, value in event["arguments"].items():
                    if key not in {"v", "value"} and isinstance(value, str):
                        names.add(value)
    def order(name: str) -> tuple[int, str]:
        start, _ = _name_span(source_text, name)
        return (start if start >= 0 else len(source_text), name)
    return sorted(names, key=order)


def _ere_semantics(
    ast: Mapping[str, Any], output: Mapping[str, Any], source_text: str,
    token_offsets: Sequence[Sequence[int]], mapping: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del mapping
    entities = _ere_entities(ast, source_text)
    if not entities or len(entities) > MAX_SEMANTIC_OBJECTS:
        raise ValueError(f"ERE semantic object count is outside 1..{MAX_SEMANTIC_OBJECTS}: {len(entities)}")
    objects = [
        _object_row(
            slot=index, kind="ere_entity", name=entity, raw_label=None,
            char_span=_name_span(source_text, entity), token_offsets=token_offsets, content=True,
        )
        for index, entity in enumerate(entities)
    ]
    slots = {name: index for index, name in enumerate(entities)}
    query = ast.get("query", {})
    query_entity = query.get("entity") if query.get("kind") == "attribute" else query.get("target")
    if query_entity not in slots:
        raise ValueError("ERE public query owner is absent from semantic entities")
    query_owner = slots[str(query_entity)]
    traces = output.get("trace", [])
    events = ast.get("events", [])
    rules = ast.get("rules", {})
    schedule: list[dict[str, Any]] = []
    values: list[list[list[int]]] = []
    masks: list[list[list[int]]] = []
    step_mask: list[int] = []
    answer = output.get("answer")
    for step in range(MODEL_STEPS):
        if step >= len(events) or step >= len(traces):
            schedule.append({"step": step, "step_index": step + 1, "active": False, "source_slot": -1, "target_slot": -1})
            values.append([[0] * 8 for _ in range(MAX_CHOICES)])
            masks.append([[0] * 8 for _ in range(MAX_CHOICES)])
            step_mask.append(0)
            continue
        event = events[step]
        trace = traces[step]
        arguments = event.get("arguments", {}) if isinstance(event, Mapping) else {}
        before = trace.get("state_before", {}) if isinstance(trace, Mapping) else {}
        after = trace.get("state_after", {}) if isinstance(trace, Mapping) else {}
        touched = [str(value) for value in arguments.values() if str(value) in slots]
        changed = [name for name in entities if _state_entity_signature(before, name) != _state_entity_signature(after, name)]
        route_pairs: list[tuple[str, str]] = []
        rule = rules.get(event.get("rule"), {}) if isinstance(rules, Mapping) and isinstance(event, Mapping) else {}
        for primitive in rule.get("primitives", []) if isinstance(rule, Mapping) else []:
            if isinstance(primitive, Mapping):
                route_pairs.extend(_primitive_routes(primitive, arguments))
        route = next((pair for pair in route_pairs if pair[0] in slots and pair[1] in slots and pair[1] in changed), None)
        route = route or next((pair for pair in route_pairs if pair[0] in slots and pair[1] in slots), None)
        target_name = route[1] if route is not None else (changed[0] if changed else touched[-1])
        source_name = route[0] if route is not None else next((name for name in touched if name != target_name), target_name)
        source_slot, target_slot = slots[source_name], slots[target_name]
        schedule.append({
            "step": step, "step_index": step + 1, "active": True,
            "rule": event.get("rule"), "source_slot": source_slot, "target_slot": target_slot,
            "touched_slots": [slots[name] for name in touched], "changed_slots": [slots[name] for name in changed],
            "state_before_sha256": _state_digest(before), "state_after_sha256": _state_digest(after),
        })
        step_values: list[list[int]] = []
        step_masks: list[list[int]] = []
        relations = after.get("relations", {}) if isinstance(after, Mapping) else {}
        attributes = after.get("attributes", {}) if isinstance(after, Mapping) else {}
        for name in entities[:MAX_CHOICES]:
            if query.get("kind") == "attribute":
                semantic = int(isinstance(attributes, Mapping) and attributes.get(name, {}).get(query.get("attribute")) == answer)
            else:
                pairs = relations.get(query.get("relation"), []) if isinstance(relations, Mapping) else []
                exists = [str(query.get("source")), name] in pairs
                semantic = int(exists == bool(answer))
            is_changed = name in changed
            step_values.append([
                1, int(name in touched), int(is_changed), int(slots[name] == query_owner),
                int(slots[name] == source_slot), int(slots[name] == target_slot), semantic, int(not is_changed),
            ])
            step_masks.append([1] * 8)
        while len(step_values) < MAX_CHOICES:
            step_values.append([0] * 8)
            step_masks.append([0] * 8)
        values.append(step_values)
        masks.append(step_masks)
        step_mask.append(1)
    alternate = next((index for index in range(len(objects)) if index != query_owner), -1)
    irrelevant = len(objects) if len(objects) < MAX_CHOICES else -1
    return {
        "objects": objects, "schedule": schedule, "state_values": values,
        "state_feature_mask": masks, "state_step_mask": step_mask,
        "query_owner": query_owner, "alternate_owner": alternate,
        "alternate_label": None, "irrelevant_owner": irrelevant,
        "content_mask": [True] * len(objects),
        "object_kinds": ["ere_entity"] * len(objects),
        "mechanism_supported": len(objects) <= MAX_CHOICES,
        "query_answer_independent": True,
        "state_feature_names": [
            "present", "touched", "changed", "query_owner",
            "operation_source", "operation_target", "query_semantic_match", "stable_after",
        ],
    }


def _cps_semantics(
    ast: Mapping[str, Any], output: Mapping[str, Any], source_text: str,
    token_offsets: Sequence[Sequence[int]], mapping: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidates = list(ast.get("candidates", []))
    object_count = len(candidates) + 1
    if not candidates or object_count > MAX_SEMANTIC_OBJECTS:
        raise ValueError(f"CPS semantic object count is outside 2..{MAX_SEMANTIC_OBJECTS}: {object_count}")
    objects: list[dict[str, Any]] = []
    for index in range(len(candidates)):
        span = _line_span(source_text, rf"^\s*Candidate\s+{index + 1}\s*:.*$")
        label = _label_for_meaning(mapping, f"candidate:{index}")
        objects.append(_object_row(
            slot=index, kind="cps_candidate", name=f"candidate:{index}", raw_label=label,
            char_span=span, token_offsets=token_offsets, content=True,
        ))
    decision_slot = len(candidates)
    decision_span = _section_span(
        source_text,
        ("GOAL, CONSTRAINTS, AND BUDGET", "REQUIRED OUTCOME", "OBJECTIVE"),
        tuple(_CHOICE_HEADINGS),
    )
    objects.append(_object_row(
        slot=decision_slot, kind="cps_decision", name="decision", raw_label=None,
        char_span=decision_span, token_offsets=token_offsets, content=False,
    ))
    evaluated = {
        int(item.get("candidate_index", index)): item
        for index, item in enumerate(output.get("candidates", [])) if isinstance(item, Mapping)
    }
    final_winner = output.get("answer")
    schedule: list[dict[str, Any]] = []
    values: list[list[list[int]]] = []
    masks: list[list[list[int]]] = []
    step_mask: list[int] = []
    for step in range(MODEL_STEPS):
        if step >= len(candidates):
            schedule.append({"step": step, "step_index": step + 1, "active": False, "source_slot": -1, "target_slot": -1})
            values.append([[0] * 8 for _ in range(MAX_CHOICES)])
            masks.append([[0] * 8 for _ in range(MAX_CHOICES)])
            step_mask.append(0)
            continue
        item = evaluated.get(step, {})
        schedule.append({
            "step": step, "step_index": step + 1, "active": True,
            "candidate_index": step, "source_slot": step, "target_slot": decision_slot,
            "state_after_sha256": _state_digest(item.get("final_state")),
            "action_trace_audit": {
                "steps": len(item.get("trace", [])) if isinstance(item.get("trace"), list) else 0,
                "sha256": _state_digest(item.get("trace", [])),
            },
        })
        processed = [evaluated.get(index, {}) for index in range(step + 1)]
        valid_costs = [
            (int(value.get("total_cost", 0)), index)
            for index, value in enumerate(processed) if bool(value.get("valid", False))
        ]
        best = min(valid_costs)[1] if valid_costs else None
        final_step = step == len(candidates) - 1
        step_values: list[list[int]] = []
        step_masks: list[list[int]] = []
        for index in range(min(len(candidates), MAX_CHOICES)):
            known = index <= step
            value = evaluated.get(index, {})
            step_values.append([
                int(known), int(bool(value.get("legal", False))) if known else 0,
                int(bool(value.get("valid", False))) if known else 0,
                int(bool(value.get("budget_ok", False))) if known else 0,
                int(bool(value.get("goal_satisfied", False))) if known else 0,
                int(bool(value.get("final_constraints_satisfied", False))) if known else 0,
                int(known and best == index), int(final_step and final_winner == index),
            ])
            step_masks.append([1] + ([1] * 7 if known else [0] * 7))
        if decision_slot < MAX_CHOICES:
            step_values.append([
                1, int(any(bool(value.get("legal", False)) for value in processed)),
                int(any(bool(value.get("valid", False)) for value in processed)),
                int(any(bool(value.get("budget_ok", False)) for value in processed)),
                int(any(bool(value.get("goal_satisfied", False)) for value in processed)),
                int(any(bool(value.get("final_constraints_satisfied", False)) for value in processed)),
                int(best is not None), int(final_step),
            ])
            step_masks.append([1] * 8)
        while len(step_values) < MAX_CHOICES:
            step_values.append([0] * 8)
            step_masks.append([0] * 8)
        values.append(step_values[:MAX_CHOICES])
        masks.append(step_masks[:MAX_CHOICES])
        step_mask.append(1)
    alternate = 0 if candidates else -1
    irrelevant = len(objects) if len(objects) < MAX_CHOICES else -1
    alternate_label = objects[alternate]["raw_label"] if alternate >= 0 else None
    return {
        "objects": objects, "schedule": schedule, "state_values": values,
        "state_feature_mask": masks, "state_step_mask": step_mask,
        "query_owner": decision_slot, "alternate_owner": alternate,
        "alternate_label": alternate_label, "irrelevant_owner": irrelevant,
        "content_mask": [True] * len(candidates) + [False],
        "object_kinds": ["cps_candidate"] * len(candidates) + ["cps_decision"],
        "mechanism_supported": object_count <= MAX_CHOICES,
        "query_answer_independent": True,
        "state_feature_names": [
            "processed", "legal", "valid", "budget_ok", "goal_satisfied",
            "final_constraints_satisfied", "running_best", "final_winner",
        ],
    }


def _fallback_semantics(
    choices: Sequence[Mapping[str, Any]], token_offsets: Sequence[Sequence[int]],
) -> dict[str, Any]:
    objects = [
        _object_row(
            slot=index, kind="unqualified_choice", name=str(slot["label"]), raw_label=str(slot["label"]),
            char_span=(int(slot["start"]), int(slot["end"])), token_offsets=token_offsets, content=False,
        )
        for index, slot in enumerate(choices[:MAX_CHOICES])
    ]
    inactive = [{"step": step, "step_index": step + 1, "active": False, "source_slot": -1, "target_slot": -1} for step in range(MODEL_STEPS)]
    return {
        "objects": objects, "schedule": inactive,
        "state_values": [[[0] * 8 for _ in range(MAX_CHOICES)] for _ in range(MODEL_STEPS)],
        "state_feature_mask": [[[0] * 8 for _ in range(MAX_CHOICES)] for _ in range(MODEL_STEPS)],
        "state_step_mask": [0] * MODEL_STEPS, "query_owner": 0, "alternate_owner": -1,
        "alternate_label": None, "irrelevant_owner": -1, "content_mask": [False] * len(objects),
        "object_kinds": ["unqualified_choice"] * len(objects),
        "mechanism_supported": False, "query_answer_independent": False,
        "state_feature_names": ["unqualified"] * 8,
    }


def _public_address_semantics(
    semantics: Mapping[str, Any], family: str, choices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Map semantics to a public, answer-independent, record-local address order."""
    objects = list(semantics["objects"])
    if family == "CPS":
        by_name = {str(obj["name"]): index for index, obj in enumerate(objects)}
        # Physical addresses are the ranks of the *present* public A-I labels,
        # not source-line order and not candidate/answer order.  In particular,
        # the decision object occupies the rank of the public NONE label, so it
        # moves from record to record instead of becoming a fixed last slot.
        ranked_choices = sorted(choices, key=lambda choice: LABELS.index(str(choice["label"]).upper()))
        public_names = [
            "decision" if _normalise_meaning(choice["meaning"]) == "none" else _normalise_meaning(choice["meaning"])
            for choice in ranked_choices
        ]
        if len(public_names) != len(objects) or set(public_names) != set(by_name):
            raise ValueError("CPS public LABEL KEY does not cover every semantic object exactly once")
        order = [by_name[name] for name in public_names]
        derivation = "present_raw_label_rank"
    else:
        # First-mention order made the ERE query slot almost a deterministic
        # function of entity count in the full bank.  Canonical lexical order
        # is source-visible and learnable on unseen entity strings; unlike a
        # hash rank it does not turn routing into a vocabulary lookup.  A
        # normalization collision is ambiguous and therefore fails closed.
        normalized_names = [_normalise_public_entity_name(obj["name"]) for obj in objects]
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("ERE public entity names collide after NFKC+casefold normalization")
        order = sorted(
            range(len(objects)),
            key=lambda index: (
                normalized_names[index],
                str(objects[index]["name"]).encode("utf-8"),
            ),
        )
        derivation = "public_entity_name_nfkc_casefold_lexical"
    old_to_new = {old: new for new, old in enumerate(order)}
    result = dict(semantics)
    permuted_objects: list[dict[str, Any]] = []
    for new, old in enumerate(order):
        obj = dict(objects[old])
        obj["slot"] = new
        obj["semantic_slot"] = old
        permuted_objects.append(obj)
    result["objects"] = permuted_objects
    result["object_kinds"] = [str(objects[old]["kind"]) for old in order]
    result["content_mask"] = [bool(objects[old]["content"]) for old in order]
    result["query_owner"] = old_to_new[int(semantics["query_owner"])]
    value = int(semantics["alternate_owner"])
    result["alternate_owner"] = old_to_new[value] if value >= 0 else -1
    irrelevant = int(semantics["irrelevant_owner"])
    result["irrelevant_owner"] = old_to_new[irrelevant] if 0 <= irrelevant < len(objects) else irrelevant
    remapped_schedule: list[dict[str, Any]] = []
    for item in semantics["schedule"]:
        row = dict(item)
        if row.get("active"):
            row["source_slot"] = old_to_new[int(row["source_slot"])]
            row["target_slot"] = old_to_new[int(row["target_slot"])]
            for key in ("touched_slots", "changed_slots"):
                if key in row:
                    row[key] = [old_to_new[int(value)] for value in row[key]]
        remapped_schedule.append(row)
    result["schedule"] = remapped_schedule
    if bool(semantics["mechanism_supported"]):
        permuted_values: list[list[list[int]]] = []
        permuted_masks: list[list[list[int]]] = []
        for step_values, step_masks in zip(semantics["state_values"], semantics["state_feature_mask"]):
            values = [list(step_values[old]) for old in order]
            masks = [list(step_masks[old]) for old in order]
            while len(values) < MAX_CHOICES:
                values.append([0] * 8)
                masks.append([0] * 8)
            permuted_values.append(values[:MAX_CHOICES])
            permuted_masks.append(masks[:MAX_CHOICES])
        result["state_values"] = permuted_values
        result["state_feature_mask"] = permuted_masks
    else:
        result["state_values"] = [[[0] * 8 for _ in range(MAX_CHOICES)] for _ in range(MODEL_STEPS)]
        result["state_feature_mask"] = [[[0] * 8 for _ in range(MAX_CHOICES)] for _ in range(MODEL_STEPS)]
    result["address_policy"] = {
        "name": ADDRESS_POLICY_NAME,
        "derivation": derivation,
        "semantic_to_physical": [old_to_new[index] for index in range(len(objects))],
        "physical_to_semantic": order,
        "answer_independent": True,
        "publicly_observable": True,
        "normalization": "NFKC_casefold" if family == "ERE" else None,
        "normalization_collision": "fail_closed" if family == "ERE" else None,
        "tie_break": "original_utf8" if family == "ERE" else None,
    }
    return result


def materialize_target(
    record: Mapping[str, Any],
    tokenizer: Any | None = None,
    *,
    cache_metadata: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Compile one full C0R row into an external S1 target row."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    example_id = record.get("example_id")
    source_text = record.get("source_text")
    family = record.get("family")
    if type(example_id) is not str or not example_id or type(source_text) is not str or not source_text:
        raise ValueError("record requires example_id and source_text")
    if family not in {"ERE", "CPS"}:
        raise ValueError(f"unknown family for {example_id}: {family}")
    choices = extract_choice_slots(source_text)
    mapping = record.get("label_mapping")
    mapping = mapping if isinstance(mapping, Mapping) else None
    answer_label = _resolve_answer_label(record, mapping, choices["slots"])
    if answer_label not in choices["label_to_slot"]:
        raise ValueError(f"answer label {answer_label} is absent from public choices for {example_id}")
    token_data = _tokenize_offsets(tokenizer, source_text)
    for index, (start, end) in enumerate(token_data["token_offsets"]):
        midpoint = start if end <= start else (start + end) // 2
        owners = [slot["slot"] for slot in choices["slots"] if int(slot["start"]) <= midpoint < int(slot["end"])]
        if owners:
            token_data["token_owner"][index] = owners[0]
    source_hash = source_byte_sha256(source_text)
    cache_source_hash = None
    if cache_metadata is not None:
        cache_source_hash = cache_metadata.get("source_hash", cache_metadata.get("source_sha256"))
        if cache_source_hash is not None and str(cache_source_hash) != source_hash:
            raise ValueError(f"source/cache hash mismatch for {example_id}")
        if cache_metadata.get("silent_truncation") not in (None, False):
            raise ValueError(f"cache metadata permits silent truncation for {example_id}")
        cached_tokens = cache_metadata.get("token_count")
        if cached_tokens is not None and int(cached_tokens) != int(token_data["token_count"]):
            raise ValueError(
                f"tokenizer/cache token-count mismatch for {example_id}: "
                f"{token_data['token_count']} != {cached_tokens}"
            )
    parsed_ast = None
    if strict:
        parsed_ast = parse_source(source_text)
        if isinstance(record.get("program_ast"), Mapping) and _canonical(parsed_ast["program_ast"]) != _canonical(record["program_ast"]):
            raise ValueError(f"public source AST mismatch for {example_id}")
    elif "program_ast" in record:
        # Small unit fixtures may intentionally provide only a choice section;
        # production/full-bank mode remains strict and source-parses here.
        try:
            parsed_ast = parse_source(source_text)
        except (TypeError, ValueError):
            parsed_ast = None
    output, compact_trace, _ = _simulator_and_trace(record)
    if strict and output is None:
        raise ValueError(f"strict target materialisation requires AST for {example_id}")
    if strict and not isinstance(mapping, Mapping):
        raise ValueError(f"strict target materialisation requires label_mapping for {example_id}")
    ast = parsed_ast.get("program_ast") if isinstance(parsed_ast, Mapping) else record.get("program_ast")
    if isinstance(ast, Mapping) and isinstance(output, Mapping):
        semantics = (
            _ere_semantics(ast, output, source_text, token_data["token_offsets"], mapping)
            if family == "ERE"
            else _cps_semantics(ast, output, source_text, token_data["token_offsets"], mapping)
        )
    else:
        semantics = _fallback_semantics(choices["slots"], token_data["token_offsets"])
    semantics = _public_address_semantics(semantics, family, choices["slots"])
    objects = semantics["objects"]
    schedule = semantics["schedule"]
    state_values = semantics["state_values"]
    state_feature_mask = semantics["state_feature_mask"]
    state_step_mask = semantics["state_step_mask"]
    query_slot = int(semantics["query_owner"])
    alternate_slot = int(semantics["alternate_owner"])
    alternate_label = semantics["alternate_label"]
    irrelevant_slot = int(semantics["irrelevant_owner"])
    model_objects = objects[:MAX_CHOICES]
    model_presence = [1 if index < len(model_objects) else 0 for index in range(MAX_CHOICES)]
    mechanism_supported = bool(semantics["mechanism_supported"])
    if not mechanism_supported and not str(record.get("split", "")).endswith("ood"):
        raise ValueError(
            f"behavior-only semantic target is allowed only on an OOD split: {example_id}"
        )
    safe_schedule = schedule if mechanism_supported else [
        {"step": step, "step_index": step + 1, "active": False, "source_slot": -1, "target_slot": -1}
        for step in range(MODEL_STEPS)
    ]
    safe_state_mask = state_feature_mask if mechanism_supported else [
        [[0] * 8 for _ in range(MAX_CHOICES)] for _ in range(MODEL_STEPS)
    ]
    safe_state_values = state_values if mechanism_supported else [
        [[0] * 8 for _ in range(MAX_CHOICES)] for _ in range(MODEL_STEPS)
    ]
    safe_step_mask = state_step_mask if mechanism_supported else [0] * MODEL_STEPS
    safe_content_mask = (
        [int(obj["content"]) for obj in model_objects] + [0] * (MAX_CHOICES - len(model_objects))
        if mechanism_supported else [0] * MAX_CHOICES
    )
    row: dict[str, Any] = {
        "schema_version": TARGET_BANK_SCHEMA,
        "example_id": example_id,
        "family": family,
        "split": record.get("split"),
        "pair_id": record.get("pair_id"),
        "pair_role": record.get("pair_role"),
        "ownership": {
            "semantic_object_count": len(objects),
            "object_kinds": semantics["object_kinds"],
            "query_owner": query_slot,
            "query_answer_independent": bool(semantics["query_answer_independent"]),
        },
        "public_choices": choices,
        "address_policy": semantics["address_policy"],
        "tensor_targets": {
            "object_labels": [obj["raw_label_index"] for obj in model_objects] + [-1] * (MAX_CHOICES - len(model_objects)),
            "presence": model_presence,
            "content_mask": safe_content_mask,
            "operation_active": [int(item["active"]) for item in safe_schedule],
            "source_owner": [int(item["source_slot"]) for item in safe_schedule],
            "target_owner": [int(item["target_slot"]) for item in safe_schedule],
            "query_owner": query_slot if mechanism_supported else 0,
            "alternate_owner": alternate_slot if mechanism_supported else -1,
            "irrelevant_owner": irrelevant_slot if mechanism_supported else -1,
            "alternate_label": LABELS.index(alternate_label) if alternate_label is not None else -1,
            "state_values": safe_state_values,
            "state_feature_mask": safe_state_mask,
            "state_step_mask": safe_step_mask,
            "token_start": [obj["token_start"] for obj in model_objects] + [-1] * (MAX_CHOICES - len(model_objects)),
            "token_end": [obj["token_end"] for obj in model_objects] + [-1] * (MAX_CHOICES - len(model_objects)),
        },
        "objects": objects,
        "presence": model_presence,
        "content_mask": safe_content_mask,
        "object_kinds": semantics["object_kinds"],
        "state_values": safe_state_values,
        "state_feature_mask": safe_state_mask,
        "state_step_mask": safe_step_mask,
        "state_feature_names": semantics["state_feature_names"],
        "token_start": [obj["token_start"] for obj in model_objects] + [-1] * (MAX_CHOICES - len(model_objects)),
        "token_end": [obj["token_end"] for obj in model_objects] + [-1] * (MAX_CHOICES - len(model_objects)),
        "schedule": safe_schedule,
        "operation_active": [int(item["active"]) for item in safe_schedule],
        "source_owner": [int(item["source_slot"]) for item in safe_schedule],
        "target_owner": [int(item["target_slot"]) for item in safe_schedule],
        "query": {
            "owner_slot": query_slot,
            "query_owner": query_slot,
            "alternate_owner": alternate_slot,
            "alternate_label": alternate_label,
            "irrelevant_owner": irrelevant_slot,
            "answer_independent": bool(semantics["query_answer_independent"]),
        },
        "query_owner": query_slot,
        "alternate_owner": alternate_slot,
        "irrelevant_owner": irrelevant_slot,
        "alternate_label": alternate_label,
        "alternate_label_index": LABELS.index(alternate_label) if alternate_label is not None else -1,
        "same_value_pair": False,
        "mechanism_supported": mechanism_supported,
        "query_answer_independent": bool(semantics["query_answer_independent"]),
        "capacity_boundary": {
            "model_slots": MAX_CHOICES,
            "semantic_objects": len(objects),
            "behavior_only": not mechanism_supported,
            "reason": None if mechanism_supported else "semantic_object_count_exceeds_model_slots_or_unqualified_source",
        },
        "answer_target": {"answer_label": answer_label, "answer_index": LABELS.index(answer_label)},
        "answer_index": LABELS.index(answer_label),
        "source": {
            "source_text_sha256": source_hash,
            "source_length": len(source_text),
            "token_count": token_data["token_count"],
            # Full offset mappings are not duplicated for all 26,624 rows;
            # choice object token_start/token_end below are the usable spans.
            "token_offsets_sha256": _sha256(token_data["token_offsets"]),
            "offset_mapping_verified": tokenizer is not None,
            "cache_source_hash": cache_source_hash,
            "cache_metadata": {
                key: value for key, value in (cache_metadata or {}).items()
                if key in {"identity", "schema_version", "manifest_sha256", "source_hash", "source_sha256", "token_count", "silent_truncation"}
            },
        },
        "provenance": {
            "ast_sha256": _sha256(record.get("program_ast")) if isinstance(record.get("program_ast"), Mapping) else None,
            "parsed_ast_sha256": _sha256(parsed_ast["program_ast"]) if isinstance(parsed_ast, Mapping) else None,
            "simulator_output_sha256": _sha256(output) if output is not None else None,
            "compact_trace_sha256": _sha256(compact_trace.encode("utf-8")) if compact_trace is not None else None,
            "c0r_schema_version": record.get("schema_version"),
            "generator_version": record.get("generator_version"),
            "record_seed": record.get("record_seed"),
            "source_identity": record.get("provenance", {}).get("source_identity") if isinstance(record.get("provenance"), Mapping) else None,
        },
    }
    if family == "ERE" and isinstance(output, Mapping):
        row["closure"] = _ere_closure(output, choices["slots"], answer_label)
    elif family == "CPS" and isinstance(output, Mapping):
        row["closure"] = _cps_closure(output, choices["slots"])
    else:
        row["closure"] = {"kind": "unmaterialized", "decisions": []}
    return row


def materialize_target_bank(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any | None = None,
    *,
    cache_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        example_id = str(record.get("example_id", ""))
        if not example_id or example_id in rows:
            raise ValueError(f"duplicate or empty target example_id: {example_id}")
        metadata = cache_metadata.get(example_id) if cache_metadata is not None else None
        rows[example_id] = materialize_target(record, tokenizer, cache_metadata=metadata, strict=strict)
    payload = {
        "schema_version": TARGET_BANK_SCHEMA,
        "records": {key: rows[key] for key in sorted(rows)},
        "record_count": len(rows),
    }
    payload["sha256"] = _target_bank_digest(payload)
    return payload


def _selection(records: Sequence[Mapping[str, Any]], *, family: str, count: int, salt: str, excluded: Iterable[str] = ()) -> list[str]:
    excluded_set = {str(value) for value in excluded}
    rows = [row for row in records if row.get("family") == family and row.get("split") == "train" and str(row.get("example_id")) not in excluded_set]
    rows.sort(key=lambda row: (_hash_rank(str(row["example_id"]), salt), str(row["example_id"])))
    if len(rows) < count:
        raise ValueError(f"insufficient {family}/train rows for {salt}: {len(rows)} < {count}")
    return [str(row["example_id"]) for row in rows[:count]]


def select_s1_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The frozen S1 selector: train population, hash-only, 16 per family."""
    selected: list[str] = []
    for family in ("ERE", "CPS"):
        selected.extend(_selection(records, family=family, count=16, salt=S1_SELECTION_SALT))
    return selected


def select_s2_ids(records: Sequence[Mapping[str, Any]], *, s1_ids: Sequence[str] | None = None) -> dict[str, list[str]]:
    excluded = set(s1_ids or select_s1_ids(records))
    train_ids: list[str] = []
    for family in ("ERE", "CPS"):
        train_ids.extend(_selection(records, family=family, count=3072, salt=S2_TRAIN_SELECTION_SALT, excluded=excluded))
    excluded |= set(train_ids)
    eval_ids: list[str] = []
    for family in ("ERE", "CPS"):
        eval_ids.extend(_selection(records, family=family, count=512, salt=S2_EVAL_SELECTION_SALT, excluded=excluded))
    return {"train": train_ids, "eval": eval_ids}


def _cell_key(row: Mapping[str, Any]) -> str:
    return f"{str(row.get('family', '')).upper()}/{row.get('split', '')}"


def build_split_ledger(records: Sequence[Mapping[str, Any]], *, s1_ids: Sequence[str] | None = None, s2: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    s1 = set(s1_ids if s1_ids is not None else select_s1_ids(records))
    s2_data = dict(s2 if s2 is not None else select_s2_ids(records, s1_ids=sorted(s1)))
    s2_train = set(s2_data.get("train", ()))
    s2_eval = set(s2_data.get("eval", ()))
    if s1 & (s2_train | s2_eval) or s2_train & s2_eval:
        raise ValueError("S1/S2 split overlap")
    cells: Counter[str] = Counter()
    assignments: dict[str, str] = {}
    for row in records:
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in assignments:
            raise ValueError(f"duplicate or empty ledger example_id: {example_id}")
        cell = _cell_key(row)
        cells[cell] += 1
        if example_id in s1:
            assignment = "s1_overfit32"
        elif example_id in s2_train:
            assignment = "s2_discovery_train"
        elif example_id in s2_eval:
            assignment = "s2_discovery_eval"
        elif row.get("split") == "train":
            assignment = "formal_train_reserved"
        elif row.get("split") == "validation":
            assignment = "formal_validation_reserved"
        elif row.get("split") == "causal_pairs":
            assignment = "formal_causal_reserved"
        elif str(row.get("split", "")).endswith("ood"):
            assignment = "formal_ood_reserved"
        else:
            assignment = "reserved"
        assignments[example_id] = assignment
    assignment_counts = Counter(assignments.values())
    return {
        "schema_version": SPLIT_LEDGER_SCHEMA,
        "population": len(records),
        "cells": dict(sorted(cells.items())),
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "s1": {"salt": S1_SELECTION_SALT, "ids": sorted(s1), "count": len(s1)},
        "s2": {
            "train_salt": S2_TRAIN_SELECTION_SALT,
            "eval_salt": S2_EVAL_SELECTION_SALT,
            "train_ids": sorted(s2_train),
            "eval_ids": sorted(s2_eval),
            "train_count": len(s2_train),
            "eval_count": len(s2_eval),
        },
        "formal": {
            "train": sum(1 for row in records if row.get("split") == "train"),
            "validation": sum(1 for row in records if row.get("split") == "validation"),
            "ood": sum(1 for row in records if str(row.get("split", "")).endswith("ood")),
            "causal": sum(1 for row in records if row.get("split") == "causal_pairs"),
        },
        "assignments": dict(sorted(assignments.items())),
        "selection_preimage_sha256": _sha256({"s1": sorted(s1), "s2_train": sorted(s2_train), "s2_eval": sorted(s2_eval)}),
        "coverage_boundary": {
            "s1_population": "ERE/train + CPS/train only",
            "s2_population": "remaining ERE/train + CPS/train only",
            "validation_ood_causal_never_used_for_discovery": True,
        },
    }


def audit_target_bank(bank: Mapping[str, Any], *, expected_count: int | None = None) -> dict[str, Any]:
    failures: list[str] = []
    supported = 0
    unsupported = 0
    query_slots: Counter[str] = Counter()
    conditional_query_slots: Counter[str] = Counter()
    supported_by_family: Counter[str] = Counter()
    source_slots: Counter[str] = Counter()
    target_slots: Counter[str] = Counter()
    owner_answer_cells: defaultdict[str, defaultdict[int, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    object_count_query_cells: defaultdict[str, defaultdict[int, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    construction_checked = 0
    query_independence_checked = 0
    if "sha256" not in bank:
        failures.append("sha256_missing")
    else:
        digest = bank.get("sha256")
    if "sha256" in bank and (not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None):
        failures.append("sha256_format")
    elif "sha256" in bank:
        try:
            expected_digest = _target_bank_digest(bank)
        except (TypeError, ValueError, OverflowError):
            failures.append("sha256_preimage")
        else:
            if digest.upper() != expected_digest:
                failures.append("sha256_mismatch")
    if bank.get("schema_version") != TARGET_BANK_SCHEMA:
        failures.append("schema_version")
    records = bank.get("records")
    if not isinstance(records, Mapping):
        failures.append("records")
        records = {}
    if type(bank.get("record_count")) is not int or bank.get("record_count") != len(records):
        failures.append("record_count")
    if expected_count is not None and len(records) != int(expected_count) and "record_count" not in failures:
        failures.append("record_count")
    for example_id, row in records.items():
        if row.get("example_id") != example_id:
            failures.append(f"example_id:{example_id}")
        ownership = row.get("ownership", {})
        objects = row.get("objects", [])
        if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_SEMANTIC_OBJECTS:
            failures.append(f"slot_count:{example_id}")
            objects = []
        mechanism_supported = row.get("mechanism_supported") is True
        supported += int(mechanism_supported)
        unsupported += int(not mechanism_supported)
        if mechanism_supported and len(objects) > MAX_CHOICES:
            failures.append(f"supported_capacity:{example_id}")
        if not mechanism_supported and not str(row.get("split", "")).endswith("ood"):
            failures.append(f"behavior_only_not_ood:{example_id}")
        if mechanism_supported and row.get("query_answer_independent") is not True:
            failures.append(f"answer_dependent_query:{example_id}")
        policy = row.get("address_policy", {})
        if policy.get("name") != ADDRESS_POLICY_NAME or policy.get("publicly_observable") is not True:
            failures.append(f"address_policy:{example_id}")
        family = str(row.get("family"))
        if family == "ERE" and objects:
            construction_checked += 1
            try:
                normalized = [_normalise_public_entity_name(obj.get("name")) for obj in objects]
            except ValueError:
                failures.append(f"ere_address_name:{example_id}")
                normalized = []
            expected = sorted(
                range(len(objects)),
                key=lambda index: (normalized[index], str(objects[index].get("name")).encode("utf-8")),
            ) if len(normalized) == len(objects) and len(set(normalized)) == len(normalized) else []
            if (
                expected != list(range(len(objects)))
                or policy.get("derivation") != "public_entity_name_nfkc_casefold_lexical"
                or policy.get("normalization") != "NFKC_casefold"
                or policy.get("normalization_collision") != "fail_closed"
                or policy.get("tie_break") != "original_utf8"
            ):
                failures.append(f"ere_address_construction:{example_id}")
        elif family == "CPS" and policy.get("derivation") != "present_raw_label_rank":
            failures.append(f"cps_address_construction:{example_id}")
        if mechanism_supported:
            query_independence_checked += 1
            if policy.get("answer_independent") is not True or row.get("query_answer_independent") is not True:
                failures.append(f"query_independence:{example_id}")
        schedule = row.get("schedule", [])
        if len(schedule) != MODEL_STEPS or not schedule[-1].get("active") is False:
            failures.append(f"schedule:{example_id}")
        tensor_targets = row.get("tensor_targets", {})
        for field, length in (("presence", MAX_CHOICES), ("content_mask", MAX_CHOICES), ("object_labels", MAX_CHOICES), ("state_values", MODEL_STEPS), ("state_feature_mask", MODEL_STEPS), ("operation_active", MODEL_STEPS), ("source_owner", MODEL_STEPS), ("target_owner", MODEL_STEPS), ("state_step_mask", MODEL_STEPS)):
            value = tensor_targets.get(field)
            if not isinstance(value, list) or len(value) != length:
                failures.append(f"tensor_shape:{example_id}:{field}")
        if tensor_targets.get("state_values") and any(
            not isinstance(step, list) or len(step) != MAX_CHOICES or any(len(item) != 8 for item in step)
            for step in tensor_targets["state_values"]
        ):
            failures.append(f"state_values_width:{example_id}")
        if tensor_targets.get("state_feature_mask") and any(
            not isinstance(step, list) or len(step) != MAX_CHOICES or any(len(item) != 8 for item in step)
            for step in tensor_targets["state_feature_mask"]
        ):
            failures.append(f"state_feature_mask_width:{example_id}")
        query = row.get("query", {})
        if query.get("owner_slot") != row.get("query_owner") or not isinstance(row.get("query_owner"), int) or not 0 <= int(row.get("query_owner", -1)) < len(objects):
            failures.append(f"query_owner:{example_id}")
        if ownership.get("query_owner") != row.get("query_owner"):
            failures.append(f"ownership_query_owner:{example_id}")
        if mechanism_supported:
            family = str(row.get("family"))
            supported_by_family[family] += 1
            query_slots[f"{family}:{row.get('query_owner')}"] += 1
            conditional_query_slots[f"{family}:answer{row.get('answer_index')}:slot{row.get('query_owner')}"] += 1
            for item in schedule:
                if item.get("active"):
                    source_slots[f"{family}:{item.get('source_slot')}"] += 1
                    target_slots[f"{family}:{item.get('target_slot')}"] += 1
            split = str(row.get("split", ""))
            if split in {"train", "validation"} and type(row.get("answer_index")) is int:
                owner_answer_cells[f"{family}/{split}"][int(row["query_owner"])][int(row["answer_index"])] += 1
                object_count_query_cells[f"{family}/{split}"][len(objects)][int(row["query_owner"])] += 1
        source = row.get("source", {})
        if source.get("source_text_sha256") and len(source["source_text_sha256"]) != 64:
            failures.append(f"source_hash:{example_id}")
    nondegenerate = {}
    for family, count in supported_by_family.items():
        distinct = len({key.rsplit(":", 1)[-1] for key in query_slots if key.startswith(f"{family}:")})
        passed = count < 8 or distinct >= 2
        nondegenerate[family] = {"records": count, "distinct_query_slots": distinct, "passed": passed}
        if not passed:
            failures.append(f"degenerate_query_slot:{family}")
    owner_only_majority_accuracy: dict[str, dict[str, Any]] = {}
    owner_answer_contingency: dict[str, dict[str, dict[str, int]]] = {}
    minimum_proxy_records = 128
    maximum_train_accuracy = 0.35
    for cell, by_owner in sorted(owner_answer_cells.items()):
        total = sum(sum(answers.values()) for answers in by_owner.values())
        correct = sum(max(answers.values()) for answers in by_owner.values() if answers)
        accuracy = float(correct / total) if total else 0.0
        sufficient = total >= minimum_proxy_records
        split = cell.rsplit("/", 1)[-1]
        gated_split = split in {"train", "validation"}
        passed = not (
            sufficient and gated_split and accuracy > maximum_train_accuracy
        )
        owner_only_majority_accuracy[cell] = {
            "records": total,
            "correct": correct,
            "accuracy": accuracy,
            "minimum_records": minimum_proxy_records,
            "sufficient": sufficient,
            "gated_split": gated_split,
            "maximum_accuracy": maximum_train_accuracy if gated_split else None,
            "passed": passed,
        }
        owner_answer_contingency[cell] = {
            str(owner): {str(answer): count for answer, count in sorted(answers.items())}
            for owner, answers in sorted(by_owner.items())
        }
        if not passed:
            failures.append(f"owner_answer_proxy:{cell}:{accuracy:.6f}")
    object_count_conditioned_query: dict[str, dict[str, Any]] = {}
    object_count_query_contingency: dict[str, dict[str, dict[str, int]]] = {}
    object_count_query_cell_metrics: dict[str, dict[str, Any]] = {}
    for cell, by_count in sorted(object_count_query_cells.items()):
        total = sum(sum(owners.values()) for owners in by_count.values())
        correct = sum(max(owners.values()) for owners in by_count.values() if owners)
        accuracy = float(correct / total) if total else 0.0
        chance_floor = (
            sum(sum(owners.values()) / max(1, object_count) for object_count, owners in by_count.items()) / total
            if total else 0.0
        )
        maximum_conditioned_accuracy = max(0.35, chance_floor + 0.10)
        sufficient = total >= minimum_proxy_records
        passed = not (sufficient and accuracy > maximum_conditioned_accuracy)
        object_count_conditioned_query[cell] = {
            "records": total,
            "majority_correct": correct,
            "majority_accuracy": accuracy,
            "minimum_records": minimum_proxy_records,
            "sufficient": sufficient,
            "uniform_within_count_floor": chance_floor,
            "maximum_accuracy": maximum_conditioned_accuracy,
            "passed": passed,
        }
        object_count_query_contingency[cell] = {
            str(object_count): {str(owner): count for owner, count in sorted(owners.items())}
            for object_count, owners in sorted(by_count.items())
        }
        if not passed:
            failures.append(f"object_count_query_proxy:{cell}:{accuracy:.6f}")
        for object_count, owners in sorted(by_count.items()):
            cell_total = sum(owners.values())
            majority = max(owners.values(), default=0)
            cell_accuracy = float(majority / cell_total) if cell_total else 0.0
            uniform_floor = 1.0 / max(1, object_count)
            maximum_accuracy = max(0.35, uniform_floor + 0.10)
            cell_sufficient = cell_total >= minimum_proxy_records
            cell_passed = not (
                cell_sufficient and cell_accuracy > maximum_accuracy
            )
            probabilities = [count / cell_total for count in owners.values()] if cell_total else []
            entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
            normalized_entropy = (
                entropy / math.log(object_count) if object_count > 1 and cell_total else 0.0
            )
            cell_name = f"{cell}/K{object_count}"
            object_count_query_cell_metrics[cell_name] = {
                "records": cell_total,
                "distinct_query_slots": len(owners),
                "majority_correct": majority,
                "majority_accuracy": cell_accuracy,
                "normalized_entropy": normalized_entropy,
                "uniform_floor": uniform_floor,
                "maximum_accuracy": maximum_accuracy,
                "minimum_records": minimum_proxy_records,
                "sufficient": cell_sufficient,
                "passed": cell_passed,
                "query_slot_counts": {
                    str(owner): count for owner, count in sorted(owners.items())
                },
            }
            if not cell_passed:
                failures.append(
                    f"object_count_query_proxy_cell:{cell_name}:{cell_accuracy:.6f}"
                )
    return {
        "schema_version": TARGET_BANK_SCHEMA,
        "examples": len(records),
        "supported": supported,
        "unsupported": unsupported,
        "unsupported_only_ood": not any("behavior_only_not_ood:" in value for value in failures),
        "slot_distributions": {
            "query": dict(sorted(query_slots.items())),
            "query_by_answer": dict(sorted(conditional_query_slots.items())),
            "source": dict(sorted(source_slots.items())),
            "target": dict(sorted(target_slots.items())),
        },
        "address_policy": ADDRESS_POLICY_NAME,
        "construction_invariance": {
            "checked": construction_checked,
            "rule": "ERE NFKC+casefold lexical rank; CPS present raw-label rank",
            "passed": not any(
                value.startswith(("ere_address_", "cps_address_")) for value in failures
            ),
        },
        "query_independence": {
            "checked": query_independence_checked,
            "passed": not any(value.startswith("query_independence:") for value in failures),
        },
        "query_slot_nondegenerate": nondegenerate,
        "owner_only_majority_accuracy": owner_only_majority_accuracy,
        "owner_answer_contingency": owner_answer_contingency,
        "owner_only_proxy_passed": not any(value.startswith("owner_answer_proxy:") for value in failures),
        "object_count_conditioned_query": object_count_conditioned_query,
        "object_count_query_cells": object_count_query_cell_metrics,
        "object_count_query_contingency": object_count_query_contingency,
        "object_count_query_proxy_passed": not any(
            value.startswith(("object_count_query_proxy:", "object_count_query_proxy_cell:"))
            for value in failures
        ),
        "failures": failures,
        "passed": not failures,
    }


# Explicit short aliases keep the provider/runner surface readable while the
# long names remain the canonical API used by audit code.
materialize_bank = materialize_target_bank
select_s1 = select_s1_ids
select_s2 = select_s2_ids
split_ledger = build_split_ledger


__all__ = [
    "ADDRESS_POLICY_NAME", "LABELS", "MAX_CHOICES", "MAX_SEMANTIC_OBJECTS", "MODEL_STEPS", "S1_SELECTION_SALT", "S2_EVAL_SELECTION_SALT",
    "S2_TRAIN_SELECTION_SALT", "SPLIT_LEDGER_SCHEMA", "TARGET_BANK_SCHEMA", "audit_target_bank",
    "build_split_ledger", "extract_choice_slots", "materialize_target", "materialize_target_bank",
    "materialize_bank", "select_s1", "select_s1_ids", "select_s2", "select_s2_ids", "split_ledger",
]
