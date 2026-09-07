from __future__ import annotations

"""Reversible finite grammars used by the R0B language-binding audit.

The parser deliberately accepts only four fully anchored render languages.  It
does not attempt to understand unrestricted prose.  Both parsing and AST
projection return the same small semantic object, so equality is meaningful
without trusting spans or annotations supplied by the bundle.
"""

import re
from typing import Any


_ATOM = r"[A-Za-z][A-Za-z0-9_-]*"

_ERE_ACTIVE = re.compile(
    rf"GRAMMAR ERE-ACTIVE-V1\n"
    rf"INITIAL (?P<i1e>{_ATOM})\.(?P<i1a>{_ATOM})=(?P<i1v>{_ATOM}); "
    rf"(?P<i2e>{_ATOM})\.(?P<i2a>{_ATOM})=(?P<i2v>{_ATOM})\n"
    rf"RULE (?P<rule>{_ATOM}) IF (?P<pe>{_ATOM})\.(?P<pa>{_ATOM})==(?P<pv>{_ATOM}) "
    rf"THEN SET (?P<te>{_ATOM})\.(?P<ta>{_ATOM})=(?P<tv>{_ATOM}) "
    rf"ELSE SET (?P<ee>{_ATOM})\.(?P<ea>{_ATOM})=(?P<ev>{_ATOM})\n"
    rf"EVENT (?P<event>{_ATOM})\n"
    rf"QUERY (?P<qe>{_ATOM})\.(?P<qa>{_ATOM})"
)

_ERE_PASSIVE = re.compile(
    rf"GRAMMAR ERE-PASSIVE-V1\n"
    rf"INITIAL (?P<i1v>{_ATOM})=(?P<i1e>{_ATOM})\.(?P<i1a>{_ATOM}); "
    rf"(?P<i2v>{_ATOM})=(?P<i2e>{_ATOM})\.(?P<i2a>{_ATOM})\n"
    rf"RULE (?P<rule>{_ATOM}) SET (?P<tv>{_ATOM})=>(?P<te>{_ATOM})\.(?P<ta>{_ATOM}) "
    rf"IF (?P<pv>{_ATOM})==(?P<pe>{_ATOM})\.(?P<pa>{_ATOM}) "
    rf"ELSE (?P<ev>{_ATOM})=>(?P<ee>{_ATOM})\.(?P<ea>{_ATOM})\n"
    rf"EVENT (?P<event>{_ATOM})\n"
    rf"QUERY (?P<qa>{_ATOM})@(?P<qe>{_ATOM})"
)

_CPS_FORWARD = re.compile(
    rf"GRAMMAR CPS-FORWARD-V1\n"
    rf"INITIAL FACTS (?P<initial>{_ATOM})\n"
    rf"ACTION (?P<action>{_ATOM}) COST (?P<cost>\d+) REQUIRES FACT (?P<required>{_ATOM}) "
    rf"EFFECT ADD_FACT (?P<effect>{_ATOM})\n"
    rf"BUDGET (?P<budget>\d+)\n"
    rf"GOAL FACT (?P<goal>{_ATOM})\n"
    rf"CANDIDATE (?P<candidate>{_ATOM})\n"
    rf"FINAL NONE"
)

_CPS_REORDERED = re.compile(
    rf"GRAMMAR CPS-REORDERED-V1\n"
    rf"CANDIDATE (?P<candidate>{_ATOM})\n"
    rf"GOAL FACT (?P<goal>{_ATOM}); BUDGET (?P<budget>\d+)\n"
    rf"ACTION (?P<action>{_ATOM}) EFFECT ADD_FACT (?P<effect>{_ATOM}) "
    rf"IF FACT (?P<required>{_ATOM}) COST (?P<cost>\d+)\n"
    rf"INITIAL FACTS (?P<initial>{_ATOM})\n"
    rf"FINAL NONE"
)


def _ere_projection(groups: dict[str, str]) -> dict[str, Any]:
    initial = sorted(
        [
            [groups["i1e"], groups["i1a"], groups["i1v"]],
            [groups["i2e"], groups["i2a"], groups["i2v"]],
        ]
    )
    return {
        "family": "ere",
        "initial_attributes": initial,
        "rule": {
            "name": groups["rule"],
            "predicate": [groups["pe"], groups["pa"], groups["pv"]],
            "then_set": [groups["te"], groups["ta"], groups["tv"]],
            "else_set": [groups["ee"], groups["ea"], groups["ev"]],
        },
        "events": [groups["event"]],
        "query": [groups["qe"], groups["qa"]],
    }


def _cps_projection(groups: dict[str, str]) -> dict[str, Any]:
    return {
        "family": "cps",
        "initial_facts": [groups["initial"]],
        "action": {
            "name": groups["action"],
            "cost": int(groups["cost"]),
            "requires_fact": groups["required"],
            "adds_fact": groups["effect"],
        },
        "budget": int(groups["budget"]),
        "goal_fact": groups["goal"],
        "candidate": [groups["candidate"]],
        "final_constraints": [],
    }


_GRAMMARS = (
    ("ERE-ACTIVE-V1", "ere", _ERE_ACTIVE, _ere_projection),
    ("ERE-PASSIVE-V1", "ere", _ERE_PASSIVE, _ere_projection),
    ("CPS-FORWARD-V1", "cps", _CPS_FORWARD, _cps_projection),
    ("CPS-REORDERED-V1", "cps", _CPS_REORDERED, _cps_projection),
)


def parse_language(text: Any) -> dict[str, Any] | None:
    """Return one grammar variant and projection, or ``None`` on ambiguity/failure."""

    if not isinstance(text, str):
        return None
    matches: list[dict[str, Any]] = []
    for variant, family, pattern, builder in _GRAMMARS:
        match = pattern.fullmatch(text)
        if match is not None:
            matches.append(
                {
                    "variant": variant,
                    "family": family,
                    "projection": builder(match.groupdict()),
                }
            )
    return matches[0] if len(matches) == 1 else None


def project_language_ast(family: Any, ast: Any) -> dict[str, Any] | None:
    """Project an AST only when it belongs to the exact reversible subset."""

    if not isinstance(ast, dict):
        return None
    if family == "ere":
        return _project_ere_ast(ast)
    if family == "cps":
        return _project_cps_ast(ast)
    return None


def _project_ere_ast(ast: dict[str, Any]) -> dict[str, Any] | None:
    try:
        state = ast["initial_state"]
        if set(ast) != {"initial_state", "rules", "events", "query"}:
            return None
        if state.get("relations") != {} or state.get("facts") != [] or state.get("resources") != {}:
            return None
        initial = sorted(
            [entity, attribute, value]
            for entity, attributes in state["attributes"].items()
            for attribute, value in attributes.items()
        )
        if len(initial) != 2 or len(ast["rules"]) != 1 or len(ast["events"]) != 1:
            return None
        rule_name, rule = next(iter(ast["rules"].items()))
        if rule.get("params") != [] or len(rule.get("primitives", [])) != 1:
            return None
        primitive = rule["primitives"][0]
        if set(primitive) != {"op", "predicate", "then", "else"} or primitive["op"] != "IF":
            return None
        predicate = primitive["predicate"]
        then = primitive["then"]
        otherwise = primitive["else"]
        if set(predicate) != {"kind", "entity", "attribute", "value"} or predicate["kind"] != "attribute_equals":
            return None
        if set(then) != {"op", "target", "attribute", "value"} or then["op"] != "SET":
            return None
        if set(otherwise) != {"op", "target", "attribute", "value"} or otherwise["op"] != "SET":
            return None
        event = ast["events"][0]
        query = ast["query"]
        if event != {"rule": rule_name, "arguments": {}}:
            return None
        if set(query) != {"kind", "entity", "attribute"} or query["kind"] != "attribute":
            return None
        return {
            "family": "ere",
            "initial_attributes": initial,
            "rule": {
                "name": rule_name,
                "predicate": [predicate["entity"], predicate["attribute"], predicate["value"]],
                "then_set": [then["target"], then["attribute"], then["value"]],
                "else_set": [otherwise["target"], otherwise["attribute"], otherwise["value"]],
            },
            "events": [rule_name],
            "query": [query["entity"], query["attribute"]],
        }
    except (KeyError, TypeError, ValueError):
        return None


def _project_cps_ast(ast: dict[str, Any]) -> dict[str, Any] | None:
    try:
        state = ast["initial_state"]
        if set(ast) != {"initial_state", "actions", "candidates", "budget", "goal", "final_constraints"}:
            return None
        if state.get("attributes") != {} or state.get("relations") != {} or state.get("resources") != {}:
            return None
        if len(state.get("facts", [])) != 1 or len(ast["actions"]) != 1 or len(ast["candidates"]) != 1:
            return None
        action = ast["actions"][0]
        if set(action) != {"name", "cost", "preconditions", "effects"}:
            return None
        if len(action["preconditions"]) != 1 or len(action["effects"]) != 1:
            return None
        precondition = action["preconditions"][0]
        effect = action["effects"][0]
        if precondition.get("kind") != "fact_true" or set(precondition) != {"kind", "fact"}:
            return None
        if effect.get("kind") != "add_fact" or set(effect) != {"kind", "fact"}:
            return None
        candidate = ast["candidates"][0]
        if candidate != {"plan": [action["name"]]}:
            return None
        if len(ast["goal"]) != 1 or ast["goal"][0].get("kind") != "fact_true":
            return None
        if set(ast["goal"][0]) != {"kind", "fact"} or ast["final_constraints"] != []:
            return None
        return {
            "family": "cps",
            "initial_facts": list(state["facts"]),
            "action": {
                "name": action["name"],
                "cost": action["cost"],
                "requires_fact": precondition["fact"],
                "adds_fact": effect["fact"],
            },
            "budget": ast["budget"],
            "goal_fact": ast["goal"][0]["fact"],
            "candidate": list(candidate["plan"]),
            "final_constraints": [],
        }
    except (KeyError, TypeError, ValueError):
        return None
