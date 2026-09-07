from __future__ import annotations

"""Reversible controlled-natural-language renderer for production ERE/CPS ASTs."""

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from ..common import evaluate_cps, simulate_ere


LABELS = tuple("ABCDEFGHI")
TEMPLATES = ("plain_v1", "reordered_v1", "indirect_v1")
TRAIN_TEMPLATES = ("plain_v1", "reordered_v1")
OOD_TEMPLATES = ("indirect_v1",)

_ATOM = r"[A-Za-z][A-Za-z0-9_-]*"
_ATOM_FULL = re.compile(rf"{_ATOM}\Z")
_OPERAND = rf"(?:'{_ATOM}'|<{_ATOM}>)"


class RendererError(ValueError):
    """Fail-closed renderer/parser contract error."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"RENDERER_CONTRACT|{code}|{message}")


_OPENINGS = {
    ("ERE", "plain_v1"): "Apply the temporary rules to the events in order, then choose one listed label.",
    ("ERE", "reordered_v1"): "Determine the requested world value after processing every occurrence in sequence.",
    ("ERE", "indirect_v1"): "A response is to be selected once the stated occurrences have been resolved under the temporary procedures.",
    ("CPS", "plain_v1"): "Evaluate every candidate plan in order and choose the unique cheapest valid option, or NONE.",
    ("CPS", "reordered_v1"): "Select the least-cost candidate that remains legal, reaches the goal, and satisfies the final requirements.",
    ("CPS", "indirect_v1"): "A response is to be selected after each proposed course has been checked for legality, outcome, budget, and final constraints.",
}

_LAYOUTS = {
    ("ERE", "plain_v1"): (
        ("rules", "TEMPORARY RULES"),
        ("state", "INITIAL WORLD"),
        ("events", "ORDERED EVENTS"),
        ("query", "QUERY"),
        ("choices", "CHOICES"),
    ),
    ("ERE", "reordered_v1"): (
        ("state", "WORLD AT THE START"),
        ("rules", "RULE DEFINITIONS"),
        ("query", "QUESTION TO ANSWER"),
        ("events", "EVENT SEQUENCE"),
        ("choices", "LABEL KEY"),
    ),
    ("ERE", "indirect_v1"): (
        ("state", "STARTING SITUATION"),
        ("rules", "PROCEDURES IN FORCE"),
        ("query", "DECISION REQUEST"),
        ("events", "OCCURRENCES TO RESOLVE"),
        ("choices", "RESPONSE KEY"),
    ),
    ("CPS", "plain_v1"): (
        ("actions", "TEMPORARY ACTIONS"),
        ("state", "INITIAL WORLD"),
        ("candidates", "CANDIDATE PLANS"),
        ("objective", "GOAL, CONSTRAINTS, AND BUDGET"),
        ("choices", "CHOICES"),
    ),
    ("CPS", "reordered_v1"): (
        ("state", "WORLD AT THE START"),
        ("objective", "REQUIRED OUTCOME"),
        ("actions", "ACTION DEFINITIONS"),
        ("candidates", "PROPOSED PLANS"),
        ("choices", "LABEL KEY"),
    ),
    ("CPS", "indirect_v1"): (
        ("state", "STARTING SITUATION"),
        ("actions", "OPERATIONS AVAILABLE"),
        ("objective", "ACCEPTANCE CONDITIONS"),
        ("candidates", "COURSES TO EXAMINE"),
        ("choices", "RESPONSE KEY"),
    ),
}


def _fail(code: str, message: str) -> None:
    raise RendererError(code, message)


def _atom(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _ATOM_FULL.fullmatch(value) is None:
        _fail("invalid_identifier", f"{field} must be an ASCII atom")
    return value


def _validate_string_domain(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _atom(key, field=f"{path}.key")
            _validate_string_domain(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_string_domain(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if value == "$neighbor":
            return
        if value.startswith("$arg:"):
            name = _atom(value[5:], field=path)
            if name == "neighbor":
                _fail("reserved_identifier", f"{path} uses reserved parameter neighbor")
            return
        _atom(value, field=path)


def _render_operand(value: Any) -> str:
    if value == "$neighbor":
        return "<neighbor>"
    if isinstance(value, str) and value.startswith("$arg:"):
        return f"<{_atom(value[5:], field='operand')}>"
    return f"'{_atom(value, field='operand')}'"


def _parse_operand(text: str) -> str:
    if re.fullmatch(rf"'{_ATOM}'", text):
        return text[1:-1]
    if re.fullmatch(rf"<{_ATOM}>", text):
        name = text[1:-1]
        return "$neighbor" if name == "neighbor" else f"$arg:{name}"
    _fail("invalid_operand", text)
    raise AssertionError("unreachable")


def _render_ere_predicate(predicate: Mapping[str, Any], *, indirect: bool) -> str:
    kind = predicate["kind"]
    if kind == "attribute_equals":
        entity = _render_operand(predicate["entity"])
        attribute = _render_operand(predicate["attribute"])
        value = _render_operand(predicate["value"])
        if indirect:
            return f"attribute {attribute} of {entity} has value {value}"
        return f"attribute {attribute} of {entity} equals {value}"
    relation = _render_operand(predicate["relation"])
    source = _render_operand(predicate["source"])
    target = _render_operand(predicate["target"])
    if indirect:
        return f"there is a {relation} link from {source} to {target}"
    return f"relation {relation} holds from {source} to {target}"


def _parse_ere_predicate(text: str, *, indirect: bool) -> dict[str, Any]:
    if indirect:
        attribute = re.fullmatch(rf"attribute (?P<a>{_OPERAND}) of (?P<e>{_OPERAND}) has value (?P<v>{_OPERAND})", text)
        relation = re.fullmatch(rf"there is a (?P<r>{_OPERAND}) link from (?P<s>{_OPERAND}) to (?P<t>{_OPERAND})", text)
    else:
        attribute = re.fullmatch(rf"attribute (?P<a>{_OPERAND}) of (?P<e>{_OPERAND}) equals (?P<v>{_OPERAND})", text)
        relation = re.fullmatch(rf"relation (?P<r>{_OPERAND}) holds from (?P<s>{_OPERAND}) to (?P<t>{_OPERAND})", text)
    if attribute:
        return {
            "kind": "attribute_equals",
            "entity": _parse_operand(attribute["e"]),
            "attribute": _parse_operand(attribute["a"]),
            "value": _parse_operand(attribute["v"]),
        }
    if relation:
        return {
            "kind": "relation_exists",
            "relation": _parse_operand(relation["r"]),
            "source": _parse_operand(relation["s"]),
            "target": _parse_operand(relation["t"]),
        }
    _fail("invalid_ere_predicate", text)
    raise AssertionError("unreachable")


def _take_group(text: str, prefix: str, opening: str, closing: str) -> tuple[str, str]:
    marker = prefix + opening
    if not text.startswith(marker):
        _fail("invalid_group", f"expected {marker!r}")
    start = len(marker)
    depth = 1
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start:index], text[index + 1 :]
    _fail("unclosed_group", text)
    raise AssertionError("unreachable")


def _render_ere_primitive(primitive: Mapping[str, Any], *, indirect: bool) -> str:
    op = primitive["op"]
    if op == "SET":
        target = _render_operand(primitive["target"])
        attribute = _render_operand(primitive["attribute"])
        value = _render_operand(primitive["value"])
        return f"attribute {attribute} of {target} becomes {value}" if indirect else f"set attribute {attribute} of {target} to {value}"
    if op == "COPY":
        source = _render_operand(primitive["source"])
        target = _render_operand(primitive["target"])
        attribute = _render_operand(primitive["attribute"])
        return f"attribute {attribute} of {target} takes the current value from {source}" if indirect else f"copy attribute {attribute} from {source} to {target}"
    if op == "SWAP":
        left = _render_operand(primitive["left"])
        right = _render_operand(primitive["right"])
        attribute = _render_operand(primitive["attribute"])
        return f"the {attribute} values of {left} and {right} exchange" if indirect else f"swap attribute {attribute} between {left} and {right}"
    if op in {"LINK", "UNLINK"}:
        relation = _render_operand(primitive["relation"])
        source = _render_operand(primitive["source"])
        target = _render_operand(primitive["target"])
        if indirect:
            verb = "is created" if op == "LINK" else "is removed"
            return f"a {relation} link from {source} to {target} {verb}"
        verb = "link" if op == "LINK" else "unlink"
        return f"{verb} {source} to {target} through relation {relation}"
    if op == "IF":
        predicate = _render_ere_predicate(primitive["predicate"], indirect=indirect)
        yes = _render_ere_primitive(primitive["then"], indirect=indirect)
        no = _render_ere_primitive(primitive["else"], indirect=indirect)
        if indirect:
            return f"provided [{predicate}], apply {{{yes}}}; otherwise apply {{{no}}}"
        return f"if [{predicate}] then {{{yes}}} otherwise {{{no}}}"
    relation = _render_operand(primitive["relation"])
    source = _render_operand(primitive["source"])
    effect = _render_ere_primitive(primitive["effect"], indirect=indirect)
    if indirect:
        return f"for every {relation} successor of {source}, apply {{{effect}}}"
    return f"for each entity linked from {source} through relation {relation}, do {{{effect}}}"


def _parse_ere_primitive(text: str, *, indirect: bool) -> dict[str, Any]:
    patterns: list[tuple[str, str, tuple[str, ...]]]
    if indirect:
        patterns = [
            ("SET", rf"attribute (?P<a>{_OPERAND}) of (?P<t>{_OPERAND}) becomes (?P<v>{_OPERAND})", ("t", "a", "v")),
            ("COPY", rf"attribute (?P<a>{_OPERAND}) of (?P<t>{_OPERAND}) takes the current value from (?P<s>{_OPERAND})", ("s", "t", "a")),
            ("SWAP", rf"the (?P<a>{_OPERAND}) values of (?P<l>{_OPERAND}) and (?P<r>{_OPERAND}) exchange", ("l", "r", "a")),
            ("LINK", rf"a (?P<rel>{_OPERAND}) link from (?P<s>{_OPERAND}) to (?P<t>{_OPERAND}) is created", ("rel", "s", "t")),
            ("UNLINK", rf"a (?P<rel>{_OPERAND}) link from (?P<s>{_OPERAND}) to (?P<t>{_OPERAND}) is removed", ("rel", "s", "t")),
        ]
    else:
        patterns = [
            ("SET", rf"set attribute (?P<a>{_OPERAND}) of (?P<t>{_OPERAND}) to (?P<v>{_OPERAND})", ("t", "a", "v")),
            ("COPY", rf"copy attribute (?P<a>{_OPERAND}) from (?P<s>{_OPERAND}) to (?P<t>{_OPERAND})", ("s", "t", "a")),
            ("SWAP", rf"swap attribute (?P<a>{_OPERAND}) between (?P<l>{_OPERAND}) and (?P<r>{_OPERAND})", ("l", "r", "a")),
            ("LINK", rf"link (?P<s>{_OPERAND}) to (?P<t>{_OPERAND}) through relation (?P<rel>{_OPERAND})", ("rel", "s", "t")),
            ("UNLINK", rf"unlink (?P<s>{_OPERAND}) to (?P<t>{_OPERAND}) through relation (?P<rel>{_OPERAND})", ("rel", "s", "t")),
        ]
    for op, pattern, _ in patterns:
        match = re.fullmatch(pattern, text)
        if not match:
            continue
        if op == "SET":
            return {"op": op, "target": _parse_operand(match["t"]), "attribute": _parse_operand(match["a"]), "value": _parse_operand(match["v"])}
        if op == "COPY":
            return {"op": op, "source": _parse_operand(match["s"]), "target": _parse_operand(match["t"]), "attribute": _parse_operand(match["a"])}
        if op == "SWAP":
            return {"op": op, "left": _parse_operand(match["l"]), "right": _parse_operand(match["r"]), "attribute": _parse_operand(match["a"])}
        return {"op": op, "relation": _parse_operand(match["rel"]), "source": _parse_operand(match["s"]), "target": _parse_operand(match["t"])}

    if indirect and text.startswith("provided ["):
        predicate, rest = _take_group(text, "provided ", "[", "]")
        yes, rest = _take_group(rest, ", apply ", "{", "}")
        no, rest = _take_group(rest, "; otherwise apply ", "{", "}")
        if rest:
            _fail("trailing_primitive_text", rest)
        return {
            "op": "IF",
            "predicate": _parse_ere_predicate(predicate, indirect=True),
            "then": _parse_ere_primitive(yes, indirect=True),
            "else": _parse_ere_primitive(no, indirect=True),
        }
    if not indirect and text.startswith("if ["):
        predicate, rest = _take_group(text, "if ", "[", "]")
        yes, rest = _take_group(rest, " then ", "{", "}")
        no, rest = _take_group(rest, " otherwise ", "{", "}")
        if rest:
            _fail("trailing_primitive_text", rest)
        return {
            "op": "IF",
            "predicate": _parse_ere_predicate(predicate, indirect=False),
            "then": _parse_ere_primitive(yes, indirect=False),
            "else": _parse_ere_primitive(no, indirect=False),
        }

    if indirect:
        prefix = re.match(rf"for every (?P<r>{_OPERAND}) successor of (?P<s>{_OPERAND}), apply ", text)
    else:
        prefix = re.match(rf"for each entity linked from (?P<s>{_OPERAND}) through relation (?P<r>{_OPERAND}), do ", text)
    if prefix:
        effect, rest = _take_group(text[prefix.end() :], "", "{", "}")
        if rest:
            _fail("trailing_primitive_text", rest)
        return {
            "op": "FOREACH_LINKED",
            "relation": _parse_operand(prefix["r"]),
            "source": _parse_operand(prefix["s"]),
            "effect": _parse_ere_primitive(effect, indirect=indirect),
        }
    _fail("invalid_ere_primitive", text)
    raise AssertionError("unreachable")


def _render_cps_condition(condition: Mapping[str, Any], *, indirect: bool) -> str:
    kind = condition["kind"]
    if kind in {"fact_true", "fact_false"}:
        fact = f"'{_atom(condition['fact'], field='fact')}'"
        truth = "holds" if kind == "fact_true" else "does not hold"
        return f"fact {fact} {truth}" if not indirect else f"{fact} is present" if kind == "fact_true" else f"{fact} is absent"
    if kind == "resource_at_least":
        resource = f"'{_atom(condition['resource'], field='resource')}'"
        amount = int(condition["amount"])
        return f"resource {resource} is at least {amount}" if not indirect else f"no fewer than {amount} units of {resource} remain"
    if kind == "attribute_equals":
        entity = f"'{_atom(condition['entity'], field='entity')}'"
        attribute = f"'{_atom(condition['attribute'], field='attribute')}'"
        value = f"'{_atom(condition['value'], field='value')}'"
        return f"attribute {attribute} of {entity} equals {value}" if not indirect else f"{entity} has {attribute} value {value}"
    relation = f"'{_atom(condition['relation'], field='relation')}'"
    source = f"'{_atom(condition['source'], field='source')}'"
    target = f"'{_atom(condition['target'], field='target')}'"
    return f"relation {relation} holds from {source} to {target}" if not indirect else f"a {relation} link exists from {source} to {target}"


def _parse_cps_condition(text: str, *, indirect: bool) -> dict[str, Any]:
    if indirect:
        patterns = (
            ("fact_true", rf"'(?P<f>{_ATOM})' is present"),
            ("fact_false", rf"'(?P<f>{_ATOM})' is absent"),
            ("resource_at_least", rf"no fewer than (?P<n>[0-9]+) units of '(?P<r>{_ATOM})' remain"),
            ("attribute_equals", rf"'(?P<e>{_ATOM})' has '(?P<a>{_ATOM})' value '(?P<v>{_ATOM})'"),
            ("relation_exists", rf"a '(?P<r>{_ATOM})' link exists from '(?P<s>{_ATOM})' to '(?P<t>{_ATOM})'"),
        )
    else:
        patterns = (
            ("fact_true", rf"fact '(?P<f>{_ATOM})' holds"),
            ("fact_false", rf"fact '(?P<f>{_ATOM})' does not hold"),
            ("resource_at_least", rf"resource '(?P<r>{_ATOM})' is at least (?P<n>[0-9]+)"),
            ("attribute_equals", rf"attribute '(?P<a>{_ATOM})' of '(?P<e>{_ATOM})' equals '(?P<v>{_ATOM})'"),
            ("relation_exists", rf"relation '(?P<r>{_ATOM})' holds from '(?P<s>{_ATOM})' to '(?P<t>{_ATOM})'"),
        )
    for kind, pattern in patterns:
        match = re.fullmatch(pattern, text)
        if not match:
            continue
        if kind in {"fact_true", "fact_false"}:
            return {"kind": kind, "fact": match["f"]}
        if kind == "resource_at_least":
            return {"kind": kind, "resource": match["r"], "amount": int(match["n"])}
        if kind == "attribute_equals":
            return {"kind": kind, "entity": match["e"], "attribute": match["a"], "value": match["v"]}
        return {"kind": kind, "relation": match["r"], "source": match["s"], "target": match["t"]}
    _fail("invalid_cps_condition", text)
    raise AssertionError("unreachable")


def _render_cps_effect(effect: Mapping[str, Any], *, indirect: bool) -> str:
    kind = effect["kind"]
    if kind in {"add_fact", "remove_fact"}:
        fact = f"'{_atom(effect['fact'], field='fact')}'"
        if indirect:
            return f"fact {fact} becomes present" if kind == "add_fact" else f"fact {fact} becomes absent"
        return f"add fact {fact}" if kind == "add_fact" else f"remove fact {fact}"
    if kind == "resource_delta":
        resource = f"'{_atom(effect['resource'], field='resource')}'"
        delta = int(effect["delta"])
        return f"resource {resource} changes by {delta}" if not indirect else f"{delta} is added to resource {resource}"
    if kind == "set_attribute":
        entity = f"'{_atom(effect['entity'], field='entity')}'"
        attribute = f"'{_atom(effect['attribute'], field='attribute')}'"
        value = f"'{_atom(effect['value'], field='value')}'"
        return f"set attribute {attribute} of {entity} to {value}" if not indirect else f"attribute {attribute} of {entity} becomes {value}"
    relation = f"'{_atom(effect['relation'], field='relation')}'"
    source = f"'{_atom(effect['source'], field='source')}'"
    target = f"'{_atom(effect['target'], field='target')}'"
    if indirect:
        verb = "is created" if kind == "link" else "is removed"
        return f"a {relation} link from {source} to {target} {verb}"
    verb = "link" if kind == "link" else "unlink"
    return f"{verb} {source} to {target} through relation {relation}"


def _parse_cps_effect(text: str, *, indirect: bool) -> dict[str, Any]:
    if indirect:
        patterns = (
            ("add_fact", rf"fact '(?P<f>{_ATOM})' becomes present"),
            ("remove_fact", rf"fact '(?P<f>{_ATOM})' becomes absent"),
            ("resource_delta", rf"(?P<n>-?[0-9]+) is added to resource '(?P<r>{_ATOM})'"),
            ("set_attribute", rf"attribute '(?P<a>{_ATOM})' of '(?P<e>{_ATOM})' becomes '(?P<v>{_ATOM})'"),
            ("link", rf"a '(?P<r>{_ATOM})' link from '(?P<s>{_ATOM})' to '(?P<t>{_ATOM})' is created"),
            ("unlink", rf"a '(?P<r>{_ATOM})' link from '(?P<s>{_ATOM})' to '(?P<t>{_ATOM})' is removed"),
        )
    else:
        patterns = (
            ("add_fact", rf"add fact '(?P<f>{_ATOM})'"),
            ("remove_fact", rf"remove fact '(?P<f>{_ATOM})'"),
            ("resource_delta", rf"resource '(?P<r>{_ATOM})' changes by (?P<n>-?[0-9]+)"),
            ("set_attribute", rf"set attribute '(?P<a>{_ATOM})' of '(?P<e>{_ATOM})' to '(?P<v>{_ATOM})'"),
            ("link", rf"link '(?P<s>{_ATOM})' to '(?P<t>{_ATOM})' through relation '(?P<r>{_ATOM})'"),
            ("unlink", rf"unlink '(?P<s>{_ATOM})' to '(?P<t>{_ATOM})' through relation '(?P<r>{_ATOM})'"),
        )
    for kind, pattern in patterns:
        match = re.fullmatch(pattern, text)
        if not match:
            continue
        if kind in {"add_fact", "remove_fact"}:
            return {"kind": kind, "fact": match["f"]}
        if kind == "resource_delta":
            return {"kind": kind, "resource": match["r"], "delta": int(match["n"])}
        if kind == "set_attribute":
            return {"kind": kind, "entity": match["e"], "attribute": match["a"], "value": match["v"]}
        return {"kind": kind, "relation": match["r"], "source": match["s"], "target": match["t"]}
    _fail("invalid_cps_effect", text)
    raise AssertionError("unreachable")


def _render_state(state: Mapping[str, Any], *, indirect: bool) -> list[str]:
    attributes = [
        item
        for entity, values in state["attributes"].items()
        for item in (
            [f"'{entity}'/'{attribute}'='{value}'" for attribute, value in values.items()]
            if values
            else [f"'{entity}'/none"]
        )
    ]
    relations = [
        item
        for relation, pairs in state["relations"].items()
        for item in (
            [f"'{relation}'('{source}','{target}')" for source, target in pairs]
            if pairs
            else [f"'{relation}'()"]
        )
    ]
    facts = [f"'{fact}'" for fact in state["facts"]]
    resources = [f"'{resource}'={amount}" for resource, amount in state["resources"].items()]
    labels = ("Attribute ledger", "Relation ledger", "True facts", "Resource ledger") if indirect else ("Attributes", "Relations", "Facts", "Resources")
    values = (attributes, relations, facts, resources)
    return [f"{label}: {' | '.join(items) if items else 'none'}" for label, items in zip(labels, values, strict=True)]


def _parse_state(lines: Sequence[str], *, indirect: bool) -> dict[str, Any]:
    labels = ("Attribute ledger", "Relation ledger", "True facts", "Resource ledger") if indirect else ("Attributes", "Relations", "Facts", "Resources")
    if len(lines) != 4 or any(not line.startswith(label + ": ") for line, label in zip(lines, labels, strict=False)):
        _fail("invalid_state_section", "state must contain four exact ledger lines")
    payloads = [line[len(label) + 2 :] for line, label in zip(lines, labels, strict=True)]

    def items(payload: str) -> list[str]:
        if payload == "none":
            return []
        parts = payload.split(" | ")
        if not parts or any(not part for part in parts):
            _fail("invalid_state_item", payload)
        return parts

    attributes: dict[str, dict[str, str]] = {}
    for item in items(payloads[0]):
        empty = re.fullmatch(rf"'(?P<e>{_ATOM})'/none", item)
        if empty:
            if empty["e"] in attributes:
                _fail("invalid_attribute_fact", item)
            attributes[empty["e"]] = {}
            continue
        match = re.fullmatch(rf"'(?P<e>{_ATOM})'/'(?P<a>{_ATOM})'='(?P<v>{_ATOM})'", item)
        if not match or (match["e"] in attributes and not attributes[match["e"]]) or match["a"] in attributes.get(match["e"], {}):
            _fail("invalid_attribute_fact", item)
        attributes.setdefault(match["e"], {})[match["a"]] = match["v"]
    relations: dict[str, list[list[str]]] = {}
    for item in items(payloads[1]):
        empty = re.fullmatch(rf"'(?P<r>{_ATOM})'\(\)", item)
        if empty:
            if empty["r"] in relations:
                _fail("invalid_relation_fact", item)
            relations[empty["r"]] = []
            continue
        match = re.fullmatch(rf"'(?P<r>{_ATOM})'\('(?P<s>{_ATOM})','(?P<t>{_ATOM})'\)", item)
        if not match or (match["r"] in relations and not relations[match["r"]]):
            _fail("invalid_relation_fact", item)
        pair = [match["s"], match["t"]]
        if pair in relations.setdefault(match["r"], []):
            _fail("duplicate_relation_fact", item)
        relations[match["r"]].append(pair)
    facts: list[str] = []
    for item in items(payloads[2]):
        match = re.fullmatch(rf"'(?P<f>{_ATOM})'", item)
        if not match or match["f"] in facts:
            _fail("invalid_fact", item)
        facts.append(match["f"])
    resources: dict[str, int] = {}
    for item in items(payloads[3]):
        match = re.fullmatch(rf"'(?P<r>{_ATOM})'=(?P<n>[0-9]+)", item)
        if not match or match["r"] in resources:
            _fail("invalid_resource", item)
        resources[match["r"]] = int(match["n"])
    return {"attributes": attributes, "relations": relations, "facts": facts, "resources": resources}


def _render_rules(rules: Mapping[str, Any], *, indirect: bool) -> list[str]:
    lines: list[str] = []
    for name, rule in rules.items():
        params = ", ".join(f"<{param}>" for param in rule["params"]) or "none"
        if indirect:
            lines.append(f"Procedure '{name}' arguments: {params}")
            prefix = f"Procedure '{name}' clause"
        else:
            lines.append(f"Rule '{name}' parameters: {params}")
            prefix = f"Rule '{name}' step"
        for index, primitive in enumerate(rule["primitives"], start=1):
            lines.append(f"{prefix} {index}: {_render_ere_primitive(primitive, indirect=indirect)}")
    return lines


def _parse_params(text: str) -> list[str]:
    if text == "none":
        return []
    parts = text.split(", ")
    params: list[str] = []
    for part in parts:
        match = re.fullmatch(rf"<(?P<p>{_ATOM})>", part)
        if not match or match["p"] == "neighbor" or match["p"] in params:
            _fail("invalid_parameters", text)
        params.append(match["p"])
    return params


def _parse_rules(lines: Sequence[str], *, indirect: bool) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    index = 0
    header_pattern = rf"Procedure '(?P<n>{_ATOM})' arguments: (?P<p>.+)" if indirect else rf"Rule '(?P<n>{_ATOM})' parameters: (?P<p>.+)"
    step_word = "clause" if indirect else "step"
    while index < len(lines):
        header = re.fullmatch(header_pattern, lines[index])
        if not header or header["n"] in rules:
            _fail("invalid_rule_header", lines[index])
        name = header["n"]
        params = _parse_params(header["p"])
        index += 1
        primitives: list[dict[str, Any]] = []
        while index < len(lines):
            step = re.fullmatch(rf"(?:Procedure|Rule) '{re.escape(name)}' {step_word} (?P<i>[0-9]+): (?P<x>.+)", lines[index])
            if not step:
                break
            if int(step["i"]) != len(primitives) + 1:
                _fail("invalid_rule_step", lines[index])
            primitives.append(_parse_ere_primitive(step["x"], indirect=indirect))
            index += 1
        if not 1 <= len(primitives) <= 2:
            _fail("invalid_rule_arity", name)
        rules[name] = {"params": params, "primitives": primitives}
    if not rules:
        _fail("empty_rules", "at least one rule is required")
    return rules


def _render_events(events: Sequence[Mapping[str, Any]], *, indirect: bool) -> list[str]:
    lines: list[str] = []
    for index, event in enumerate(events, start=1):
        separator = "; " if indirect else ", "
        arguments = separator.join(f"<{name}>='{value}'" for name, value in event["arguments"].items()) or "none"
        if indirect:
            lines.append(f"Occurrence {index} applies '{event['rule']}' using {arguments}")
        else:
            lines.append(f"Event {index}: invoke '{event['rule']}' with {arguments}")
    return lines


def _parse_events(lines: Sequence[str], *, indirect: bool) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    pattern = rf"Occurrence (?P<i>[0-9]+) applies '(?P<r>{_ATOM})' using (?P<a>.+)" if indirect else rf"Event (?P<i>[0-9]+): invoke '(?P<r>{_ATOM})' with (?P<a>.+)"
    separator = "; " if indirect else ", "
    for line in lines:
        match = re.fullmatch(pattern, line)
        if not match or int(match["i"]) != len(events) + 1:
            _fail("invalid_event", line)
        arguments: dict[str, str] = {}
        if match["a"] != "none":
            for item in match["a"].split(separator):
                argument = re.fullmatch(rf"<(?P<n>{_ATOM})>='(?P<v>{_ATOM})'", item)
                if not argument or argument["n"] in arguments:
                    _fail("invalid_event_arguments", item)
                arguments[argument["n"]] = argument["v"]
        events.append({"rule": match["r"], "arguments": arguments})
    if not events:
        _fail("empty_events", "at least one event is required")
    return events


def _render_ere_query(query: Mapping[str, Any], *, indirect: bool) -> list[str]:
    if query["kind"] == "attribute":
        if indirect:
            return [f"Select the label denoting the final value of '{query['attribute']}' for '{query['entity']}'."]
        return [f"What is the final value of attribute '{query['attribute']}' for entity '{query['entity']}'?"]
    if indirect:
        return [f"Select the label stating whether a '{query['relation']}' link finally runs from '{query['source']}' to '{query['target']}'."]
    return [f"Does relation '{query['relation']}' finally hold from '{query['source']}' to '{query['target']}'?"]


def _parse_ere_query(lines: Sequence[str], *, indirect: bool) -> dict[str, Any]:
    if len(lines) != 1:
        _fail("invalid_query", "query section must have one line")
    text = lines[0]
    if indirect:
        attribute = re.fullmatch(rf"Select the label denoting the final value of '(?P<a>{_ATOM})' for '(?P<e>{_ATOM})'\.", text)
        relation = re.fullmatch(rf"Select the label stating whether a '(?P<r>{_ATOM})' link finally runs from '(?P<s>{_ATOM})' to '(?P<t>{_ATOM})'\.", text)
    else:
        attribute = re.fullmatch(rf"What is the final value of attribute '(?P<a>{_ATOM})' for entity '(?P<e>{_ATOM})'\?", text)
        relation = re.fullmatch(rf"Does relation '(?P<r>{_ATOM})' finally hold from '(?P<s>{_ATOM})' to '(?P<t>{_ATOM})'\?", text)
    if attribute:
        return {"kind": "attribute", "entity": attribute["e"], "attribute": attribute["a"]}
    if relation:
        return {"kind": "relation", "relation": relation["r"], "source": relation["s"], "target": relation["t"]}
    _fail("invalid_query", text)
    raise AssertionError("unreachable")


def _render_actions(actions: Sequence[Mapping[str, Any]], *, indirect: bool) -> list[str]:
    lines: list[str] = []
    for action in actions:
        name = action["name"]
        if indirect:
            lines.append(f"Operation '{name}' carries cost {action['cost']}")
            lines.append(f"Operation '{name}' is permitted when: {' && '.join(_render_cps_condition(item, indirect=True) for item in action['preconditions']) or 'nothing'}")
            lines.append(f"Operation '{name}' then causes: {' >> '.join(_render_cps_effect(item, indirect=True) for item in action['effects']) or 'nothing'}")
        else:
            lines.append(f"Action '{name}' cost: {action['cost']}")
            lines.append(f"Action '{name}' requires: {' && '.join(_render_cps_condition(item, indirect=False) for item in action['preconditions']) or 'nothing'}")
            lines.append(f"Action '{name}' effects: {' >> '.join(_render_cps_effect(item, indirect=False) for item in action['effects']) or 'nothing'}")
    return lines


def _parse_joined(text: str, parser: Any, *, indirect: bool, separator: str) -> list[dict[str, Any]]:
    if text == "nothing":
        return []
    parts = text.split(separator)
    if not parts or any(not part for part in parts):
        _fail("invalid_joined_expression", text)
    return [parser(part, indirect=indirect) for part in parts]


def _parse_actions(lines: Sequence[str], *, indirect: bool) -> list[dict[str, Any]]:
    if len(lines) % 3 != 0 or not lines:
        _fail("invalid_actions", "actions require exact three-line records")
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for offset in range(0, len(lines), 3):
        if indirect:
            head = re.fullmatch(rf"Operation '(?P<n>{_ATOM})' carries cost (?P<c>[0-9]+)", lines[offset])
            pre = re.fullmatch(rf"Operation '(?P<n>{_ATOM})' is permitted when: (?P<x>.+)", lines[offset + 1])
            eff = re.fullmatch(rf"Operation '(?P<n>{_ATOM})' then causes: (?P<x>.+)", lines[offset + 2])
        else:
            head = re.fullmatch(rf"Action '(?P<n>{_ATOM})' cost: (?P<c>[0-9]+)", lines[offset])
            pre = re.fullmatch(rf"Action '(?P<n>{_ATOM})' requires: (?P<x>.+)", lines[offset + 1])
            eff = re.fullmatch(rf"Action '(?P<n>{_ATOM})' effects: (?P<x>.+)", lines[offset + 2])
        if not head or not pre or not eff or head["n"] != pre["n"] or head["n"] != eff["n"] or head["n"] in seen:
            _fail("invalid_action_record", lines[offset])
        seen.add(head["n"])
        actions.append(
            {
                "name": head["n"],
                "cost": int(head["c"]),
                "preconditions": _parse_joined(pre["x"], _parse_cps_condition, indirect=indirect, separator=" && "),
                "effects": _parse_joined(eff["x"], _parse_cps_effect, indirect=indirect, separator=" >> "),
            }
        )
    return actions


def _render_candidates(candidates: Sequence[Mapping[str, Any]], *, indirect: bool) -> list[str]:
    lines: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        plan = " -> ".join(f"'{name}'" for name in candidate["plan"]) or "do nothing"
        lines.append(f"Course {index} consists of: {plan}" if indirect else f"Candidate {index}: {plan}")
    return lines


def _parse_candidates(lines: Sequence[str], *, indirect: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    pattern = r"Course (?P<i>[0-9]+) consists of: (?P<x>.+)" if indirect else r"Candidate (?P<i>[0-9]+): (?P<x>.+)"
    for line in lines:
        match = re.fullmatch(pattern, line)
        if not match or int(match["i"]) != len(candidates) + 1:
            _fail("invalid_candidate", line)
        plan: list[str] = []
        if match["x"] != "do nothing":
            for item in match["x"].split(" -> "):
                action = re.fullmatch(rf"'(?P<n>{_ATOM})'", item)
                if not action:
                    _fail("invalid_candidate_plan", item)
                plan.append(action["n"])
        candidates.append({"plan": plan})
    if not candidates:
        _fail("empty_candidates", "at least one candidate is required")
    return candidates


def _render_objective(ast: Mapping[str, Any], *, indirect: bool) -> list[str]:
    goal = " && ".join(_render_cps_condition(item, indirect=indirect) for item in ast["goal"]) or "nothing"
    constraints = " && ".join(_render_cps_condition(item, indirect=indirect) for item in ast["final_constraints"]) or "nothing"
    if indirect:
        return [f"Required goal: {goal}", f"Required final state: {constraints}", f"Total cost may not exceed {ast['budget']}"]
    return [f"Goal: {goal}", f"Final constraints: {constraints}", f"Budget: {ast['budget']}"]


def _parse_objective(lines: Sequence[str], *, indirect: bool) -> dict[str, Any]:
    if len(lines) != 3:
        _fail("invalid_objective", "objective requires three lines")
    if indirect:
        prefixes = ("Required goal: ", "Required final state: ", "Total cost may not exceed ")
    else:
        prefixes = ("Goal: ", "Final constraints: ", "Budget: ")
    if any(not line.startswith(prefix) for line, prefix in zip(lines, prefixes, strict=True)):
        _fail("invalid_objective", "objective prefixes differ")
    goal = _parse_joined(lines[0][len(prefixes[0]) :], _parse_cps_condition, indirect=indirect, separator=" && ")
    constraints = _parse_joined(lines[1][len(prefixes[1]) :], _parse_cps_condition, indirect=indirect, separator=" && ")
    budget_text = lines[2][len(prefixes[2]) :]
    if re.fullmatch(r"[0-9]+", budget_text) is None:
        _fail("invalid_budget", budget_text)
    return {"goal": goal, "final_constraints": constraints, "budget": int(budget_text)}


def _choice_semantics(family: str, ast: Mapping[str, Any]) -> set[str]:
    if family == "ERE":
        return set()
    return {*(f"candidate:{index}" for index in range(len(ast["candidates"]))), "none"}


def _validate_choices(family: str, ast: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(mapping, Mapping) or not 2 <= len(mapping) <= len(LABELS):
        _fail("invalid_choice_mapping", "choice mapping must contain 2-9 entries")
    output: dict[str, str] = {}
    seen: set[str] = set()
    for raw_label, raw_semantic in mapping.items():
        if raw_label not in LABELS or raw_label in output or not isinstance(raw_semantic, str):
            _fail("invalid_choice_mapping", "invalid or duplicate local label")
        semantic = raw_semantic
        if semantic in seen:
            _fail("invalid_choice_mapping", "choice semantics must be unique")
        if family == "ERE":
            if semantic not in {"true", "false"}:
                if not semantic.startswith("value:"):
                    _fail("invalid_choice_semantic", semantic)
                _atom(semantic[6:], field="choice value")
            if ast["query"]["kind"] == "relation" and semantic not in {"true", "false"}:
                _fail("invalid_choice_semantic", "relation query choices must be true/false")
        elif not (semantic == "none" or re.fullmatch(r"candidate:[0-9]+", semantic)):
            _fail("invalid_choice_semantic", semantic)
        output[raw_label] = semantic
        seen.add(semantic)
    if family == "ERE" and ast["query"]["kind"] == "relation" and seen != {"true", "false"}:
        _fail("invalid_choice_mapping", "relation query requires exactly true and false")
    if family == "CPS" and seen != _choice_semantics(family, ast):
        _fail("invalid_choice_mapping", "CPS choices must cover every candidate and NONE")
    return output


def _render_choices(family: str, mapping: Mapping[str, str], *, indirect: bool) -> list[str]:
    lines: list[str] = []
    for label, semantic in mapping.items():
        if semantic.startswith("value:"):
            meaning = f"value '{semantic[6:]}'"
        elif semantic in {"true", "false"}:
            meaning = semantic.upper()
        elif semantic == "none":
            meaning = "NONE"
        else:
            meaning = f"candidate {int(semantic.split(':')[1]) + 1}"
        lines.append(f"Label {label} denotes {meaning}" if indirect else f"{label} means {meaning}")
    return lines


def _parse_choices(lines: Sequence[str], family: str, ast: Mapping[str, Any], *, indirect: bool) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pattern = r"Label (?P<l>[A-I]) denotes (?P<x>.+)" if indirect else r"(?P<l>[A-I]) means (?P<x>.+)"
    for line in lines:
        match = re.fullmatch(pattern, line)
        if not match or match["l"] in mapping:
            _fail("invalid_choice_line", line)
        meaning = match["x"]
        value = re.fullmatch(rf"value '(?P<v>{_ATOM})'", meaning)
        candidate = re.fullmatch(r"candidate (?P<i>[1-9][0-9]*)", meaning)
        if value:
            semantic = f"value:{value['v']}"
        elif meaning in {"TRUE", "FALSE"}:
            semantic = meaning.lower()
        elif meaning == "NONE":
            semantic = "none"
        elif candidate:
            semantic = f"candidate:{int(candidate['i']) - 1}"
        else:
            _fail("invalid_choice_meaning", meaning)
        mapping[match["l"]] = semantic
    return _validate_choices(family, ast, mapping)


def _section_payloads(source_text: str) -> tuple[str, str, dict[str, list[str]]]:
    if not isinstance(source_text, str) or not source_text or "\r" in source_text or source_text.endswith("\n"):
        _fail("invalid_source_text", "source must be nonempty canonical LF text without trailing newline")
    lines = source_text.split("\n")
    if any(not line or line != line.strip() for line in lines):
        _fail("invalid_source_text", "blank or padded line")
    matches = [(family, template) for (family, template), opening in _OPENINGS.items() if lines[0] == opening]
    if len(matches) != 1:
        _fail("unknown_opening", lines[0])
    family, template = matches[0]
    layout = _LAYOUTS[(family, template)]
    headings = [heading for _, heading in layout]
    positions: list[int] = []
    for heading in headings:
        found = [index for index, line in enumerate(lines[1:], start=1) if line == heading]
        if len(found) != 1:
            _fail("missing_or_duplicate_section", heading)
        positions.append(found[0])
    if positions != sorted(positions) or positions[0] != 1:
        _fail("section_order", template)
    payloads: dict[str, list[str]] = {}
    for section_index, (semantic, _) in enumerate(layout):
        start = positions[section_index] + 1
        end = positions[section_index + 1] if section_index + 1 < len(positions) else len(lines)
        body = lines[start:end]
        if not body:
            _fail("empty_section", semantic)
        payloads[semantic] = body
    return family, template, payloads


def render_source(family: str, program_ast: Mapping[str, Any], label_mapping: Mapping[str, str], template_id: str) -> str:
    """Render a complete AST and local choice key without receiving the answer."""

    if family not in {"ERE", "CPS"} or template_id not in TEMPLATES:
        _fail("unsupported_profile", f"{family}/{template_id}")
    if not isinstance(program_ast, Mapping):
        _fail("invalid_ast", "program_ast must be an object")
    ast = deepcopy(dict(program_ast))
    _validate_string_domain(ast)
    if family == "ERE":
        simulate_ere(ast)
    else:
        evaluate_cps(ast)
    choices = _validate_choices(family, ast, label_mapping)
    indirect = template_id == "indirect_v1"
    if family == "ERE":
        bodies = {
            "rules": _render_rules(ast["rules"], indirect=indirect),
            "state": _render_state(ast["initial_state"], indirect=indirect),
            "events": _render_events(ast["events"], indirect=indirect),
            "query": _render_ere_query(ast["query"], indirect=indirect),
            "choices": _render_choices(family, choices, indirect=indirect),
        }
    else:
        bodies = {
            "actions": _render_actions(ast["actions"], indirect=indirect),
            "state": _render_state(ast["initial_state"], indirect=indirect),
            "candidates": _render_candidates(ast["candidates"], indirect=indirect),
            "objective": _render_objective(ast, indirect=indirect),
            "choices": _render_choices(family, choices, indirect=indirect),
        }
    lines = [_OPENINGS[(family, template_id)]]
    for semantic, heading in _LAYOUTS[(family, template_id)]:
        lines.append(heading)
        lines.extend(bodies[semantic])
    return "\n".join(lines)


def parse_source(source_text: str) -> dict[str, Any]:
    """Parse only visible source text into its complete semantic document."""

    family, template, payloads = _section_payloads(source_text)
    indirect = template == "indirect_v1"
    if family == "ERE":
        ast = {
            "initial_state": _parse_state(payloads["state"], indirect=indirect),
            "rules": _parse_rules(payloads["rules"], indirect=indirect),
            "events": _parse_events(payloads["events"], indirect=indirect),
            "query": _parse_ere_query(payloads["query"], indirect=indirect),
        }
        simulate_ere(ast)
    else:
        objective = _parse_objective(payloads["objective"], indirect=indirect)
        ast = {
            "initial_state": _parse_state(payloads["state"], indirect=indirect),
            "actions": _parse_actions(payloads["actions"], indirect=indirect),
            "candidates": _parse_candidates(payloads["candidates"], indirect=indirect),
            "budget": objective["budget"],
            "goal": objective["goal"],
            "final_constraints": objective["final_constraints"],
        }
        evaluate_cps(ast)
    choices = _parse_choices(payloads["choices"], family, ast, indirect=indirect)
    return {"family": family, "template_id": template, "program_ast": ast, "label_mapping": choices}
