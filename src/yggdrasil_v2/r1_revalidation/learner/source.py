from __future__ import annotations

"""Source-only qualification grammar and deterministic heuristic parsers."""

from dataclasses import dataclass
import re
from typing import Any, Sequence

from .canonical import ContractError, LABELS, choose_label, normalize_source, require_mask


_CHOICE_RE = re.compile(r"^(L[0-2])=([^|]+)$")
_RULE_RE = re.compile(r"^Rule ([^:]+): (.+)$")
_EVENT_RE = re.compile(r"^Event ([^:]+): (.+)$")
_ACTION_RE = re.compile(r"^Action ([^:]+): (.+)$")
_CANDIDATE_RE = re.compile(r"^Candidate (L[0-2]): ([^;]+); cost=([0-9]+)$")


@dataclass(frozen=True)
class ParsedSource:
    normalized: str
    family: str
    question: str
    choices: dict[str, str]
    value_to_label: dict[str, str]
    lines: tuple[str, ...]
    initial: str | None
    rules: tuple[str, ...]
    events: tuple[str, ...]
    actions: tuple[str, ...]
    candidates: tuple[tuple[str, tuple[str, ...], int], ...]


def _single_prefixed(lines: Sequence[str], prefix: str) -> str:
    values = [line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise ContractError("malformed_source", f"expected exactly one {prefix!r} line")
    return values[0]


def parse_source(source: str, family: str) -> ParsedSource:
    if family not in {"ERE", "CPS"}:
        raise ContractError("unknown_family", f"unsupported family {family!r}")
    normalized = normalize_source(source)
    lines = tuple(normalized.split("\n")) if normalized else ()
    if not lines:
        raise ContractError("malformed_source", "source is empty")
    question = _single_prefixed(lines, "Question:")
    raw_choices = _single_prefixed(lines, "Choices:")
    choices: dict[str, str] = {}
    for part in raw_choices.split("|"):
        match = _CHOICE_RE.fullmatch(part.strip())
        if match is None:
            raise ContractError("malformed_choices", f"invalid choice entry {part!r}")
        label, value = match.group(1), match.group(2).strip()
        if label in choices or not value:
            raise ContractError("malformed_choices", "duplicate label or empty choice value")
        choices[label] = value
    if tuple(sorted(choices)) != LABELS or len(set(choices.values())) != len(LABELS):
        raise ContractError("malformed_choices", "choices must define unique L0/L1/L2 values")
    value_to_label = {value: label for label, value in choices.items()}

    initial_values = [line[len("Initial:"):].strip() for line in lines if line.startswith("Initial:")]
    family_header = "World:" if family == "ERE" else "Planning:"
    if lines.count(family_header) != 1:
        raise ContractError("malformed_source", f"expected exactly one {family_header!r} line")
    rules: list[str] = []
    events: list[str] = []
    actions: list[str] = []
    candidates: list[tuple[str, tuple[str, ...], int]] = []
    for line in lines:
        if (
            line == family_header
            or line.startswith("Question:")
            or line.startswith("Choices:")
            or line.startswith("Initial:")
        ):
            continue
        if line.startswith("Rule "):
            match = _RULE_RE.fullmatch(line)
            if match is None:
                raise ContractError("malformed_source", f"invalid rule line {line!r}")
            rules.append(match.group(2).strip())
        elif line.startswith("Event "):
            match = _EVENT_RE.fullmatch(line)
            if match is None:
                raise ContractError("malformed_source", f"invalid event line {line!r}")
            events.append(match.group(2).strip())
        elif line.startswith("Action "):
            match = _ACTION_RE.fullmatch(line)
            if match is None:
                raise ContractError("malformed_source", f"invalid action line {line!r}")
            actions.append(match.group(1).strip())
        elif line.startswith("Candidate "):
            match = _CANDIDATE_RE.fullmatch(line)
            if match is None:
                raise ContractError("malformed_candidate", f"invalid candidate line {line!r}")
            plan = tuple(item.strip() for item in match.group(2).split(",") if item.strip())
            if not plan:
                raise ContractError("malformed_candidate", "candidate plan cannot be empty")
            candidates.append((match.group(1), plan, int(match.group(3))))
        else:
            raise ContractError("malformed_source", f"unknown source line {line!r}")

    if family == "ERE":
        if len(initial_values) != 1 or not rules or not events:
            raise ContractError("malformed_source", "ERE requires one initial line and non-empty rules/events")
        visible_values = [initial_values[0], *rules, *events]
        if any(value not in value_to_label for value in visible_values):
            raise ContractError("unknown_choice_value", "ERE literal does not map to a visible choice")
        if actions or candidates:
            raise ContractError("malformed_source", "ERE cannot contain CPS action/candidate lines")
    else:
        if initial_values or rules or events:
            raise ContractError("malformed_source", "CPS cannot contain ERE initial/rule/event lines")
        if not actions or len(candidates) != len(LABELS):
            raise ContractError("malformed_source", "CPS requires actions and exactly three candidates")
        if len(set(actions)) != len(actions):
            raise ContractError("malformed_source", "CPS action names must be unique")
        if tuple(sorted(label for label, _, _ in candidates)) != LABELS:
            raise ContractError("malformed_candidate", "CPS candidates must define L0/L1/L2")
        action_set = set(actions)
        if any(action not in action_set for _, plan, _ in candidates for action in plan):
            raise ContractError("unknown_action", "candidate references an undefined action")

    return ParsedSource(
        normalized=normalized,
        family=family,
        question=question,
        choices=choices,
        value_to_label=value_to_label,
        lines=lines,
        initial=initial_values[0] if initial_values else None,
        rules=tuple(rules),
        events=tuple(events),
        actions=tuple(actions),
        candidates=tuple(candidates),
    )


def surface_counts(parsed: ParsedSource) -> dict[str, int]:
    return {
        "line_count": len(parsed.lines),
        "word_count": len(re.findall(r"(?u)\b\w+\b", parsed.normalized.lower())),
        "character_count": len(parsed.normalized),
        "rule_count": len(parsed.rules),
        "event_count": len(parsed.events),
        "action_count": len(parsed.actions),
        "candidate_count": len(parsed.candidates),
        "choice_count": len(parsed.choices),
    }


def _masked(raw_label: str, mask: Sequence[bool]) -> str:
    scores = {label: int(label == raw_label) for label in LABELS}
    return choose_label(scores, mask, LABELS)


def heuristic_prediction(
    source: str,
    family: str,
    heuristic: str,
    valid_choice_mask: Sequence[bool],
) -> dict[str, Any]:
    mask = require_mask(list(valid_choice_mask), LABELS)
    parsed = parse_source(source, family)

    if family == "ERE":
        values: dict[str, str] = {
            "initial_value": parsed.initial or "",
            "first_rule_literal": parsed.rules[0],
            "last_rule_literal": parsed.rules[-1],
            "first_event_literal": parsed.events[0],
            "last_event_literal": parsed.events[-1],
            "last_mention": (*parsed.rules, *parsed.events)[-1],
        }
        if heuristic not in values:
            raise ContractError("unknown_heuristic", f"unknown ERE heuristic {heuristic!r}")
        raw_label = parsed.value_to_label[values[heuristic]]
    else:
        candidates = list(parsed.candidates)
        by_label = {label: (plan, cost) for label, plan, cost in candidates}
        if heuristic == "first_candidate":
            raw_label = candidates[0][0]
        elif heuristic == "last_candidate":
            raw_label = candidates[-1][0]
        elif heuristic == "shortest_candidate":
            raw_label = min(LABELS, key=lambda label: (len(by_label[label][0]), LABELS.index(label)))
        elif heuristic == "longest_candidate":
            raw_label = max(LABELS, key=lambda label: (len(by_label[label][0]), -LABELS.index(label)))
        elif heuristic == "lowest_raw_cost":
            raw_label = min(LABELS, key=lambda label: (by_label[label][1], LABELS.index(label)))
        elif heuristic == "highest_raw_cost":
            raw_label = max(LABELS, key=lambda label: (by_label[label][1], -LABELS.index(label)))
        elif heuristic == "first_definition":
            first_action = parsed.actions[0]
            matches = [label for label in LABELS if by_label[label][0][0] == first_action]
            raw_label = matches[0] if matches else LABELS[0]
        else:
            raise ContractError("unknown_heuristic", f"unknown CPS heuristic {heuristic!r}")

    return {
        "question": parsed.question,
        "counts": surface_counts(parsed),
        "raw_prediction": raw_label,
        "prediction": _masked(raw_label, mask),
    }
