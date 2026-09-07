from __future__ import annotations

"""Deterministic full-scale production ERE/CPS dataset generator."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..common import evaluate_cps, replay_cps_prefix, simulate_ere
from .fingerprint import semantic_fingerprint, surface_fingerprint
from .renderer import LABELS, TRAIN_TEMPLATES, render_source


SCHEMA_VERSION = "yggdrasil.v2-r1r.p0d-v17.record.v1"
GENERATOR_VERSION = "r1r-p0d-v17-visible-domain-alpha-repair"
FORMAL_SEED = 2026081702
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SPLITS = {
    "ERE": ("train", "validation", "composition_ood", "length_ood", "entity_ood", "language_ood", "causal_pairs"),
    "CPS": ("train", "validation", "composition_ood", "horizon_ood", "distractor_ood", "language_ood", "causal_pairs"),
}


@dataclass(frozen=True)
class DatasetSpec:
    profile: str
    root_seed: int
    train_count: int
    heldout_count: int

    def count(self, split: str) -> int:
        return self.train_count if split == "train" else self.heldout_count


FORMAL_SPEC = DatasetSpec("formal", FORMAL_SEED, 4096, 1536)

_FAMILY_SEED_STRIDE = 10_000_019
_SPLIT_SEED_STRIDE = 1_000_003
_UNIT_SEED_STRIDE = 104_729
_ATTEMPT_SEED_STRIDE = 7_919
SEED_DERIVATION_VERSION = "root-family-split-unit-attempt-v1"

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z")
_VOWELS = ("a", "e", "i", "o", "u")
_CODAS = ("", "n", "r", "s", "t", "m", "l")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _leaf_differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path or "/", "before": deepcopy(left), "after": deepcopy(right)}]
    if isinstance(left, Mapping):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                differences.append(
                    {
                        "path": child,
                        "before": deepcopy(left.get(key)),
                        "after": deepcopy(right.get(key)),
                    }
                )
            else:
                differences.extend(_leaf_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        differences = []
        for index in range(max(len(left), len(right))):
            child = f"{path}/{index}"
            if index >= len(left) or index >= len(right):
                differences.append(
                    {
                        "path": child,
                        "before": deepcopy(left[index]) if index < len(left) else None,
                        "after": deepcopy(right[index]) if index < len(right) else None,
                    }
                )
            else:
                differences.extend(_leaf_differences(left[index], right[index], child))
        return differences
    return [] if left == right else [{"path": path or "/", "before": deepcopy(left), "after": deepcopy(right)}]


class _NonceFactory:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set[str] = set()

    def name(self) -> str:
        for _ in range(10_000):
            syllables = self.rng.choice((2, 2, 3))
            value = "".join(self.rng.choice(_ONSETS) + self.rng.choice(_VOWELS) + self.rng.choice(_CODAS) for _ in range(syllables))
            if value not in self.used and value not in {"none", "true", "false", "neighbor"}:
                self.used.add(value)
                return value
        raise RuntimeError("nonce namespace exhausted")

    def names(self, count: int) -> list[str]:
        return [self.name() for _ in range(count)]


def _state(entities: Sequence[str], attributes: Sequence[str], values: Sequence[str], relation: str, rng: random.Random) -> dict[str, Any]:
    state = {
        "attributes": {
            entity: {attribute: rng.choice(values) for attribute in attributes}
            for entity in entities
        },
        "relations": {relation: []},
        "facts": [],
        "resources": {},
    }
    # The untouched secondary attribute is a guaranteed two-value audit
    # witness at every prefix.  It lets claim pairs balance both values across
    # positive and negative labels without relying on accidental random state.
    if len(entities) >= 2 and len(attributes) >= 2 and len(values) >= 4:
        witnesses = [values[2], values[3]]
        rng.shuffle(witnesses)
        state["attributes"][entities[0]][attributes[1]] = witnesses[0]
        state["attributes"][entities[1]][attributes[1]] = witnesses[1]
    if len(entities) >= 4 and len(attributes) >= 3 and len(values) >= 6:
        # The same four visible distractor values are independently permuted
        # across event-role entities.  This adds real alpha-invariant state
        # structure instead of hidden nonce salt while retaining all six
        # declared values and the accepted three-attribute task domain.
        distractors = [values[4], values[5], values[0], values[1]]
        rng.shuffle(distractors)
        for entity, value in zip(entities[:4], distractors, strict=True):
            state["attributes"][entity][attributes[2]] = value
    return state


def _ere_symbols(rng: random.Random, entity_count: int) -> dict[str, Any]:
    factory = _NonceFactory(rng)
    return {
        "entities": factory.names(entity_count),
        # Three fully rendered attributes and six values create a legitimate
        # state-space large enough for the 4,096/1,536 production contract.  The
        # first four values remain the local answer domain; the other two are
        # visible state distractors, never hidden fingerprint salt.
        "attributes": factory.names(3),
        "values": factory.names(6),
        "relation": factory.name(),
        "rules": factory.names(10),
    }


def _set_rule(attribute: str) -> dict[str, Any]:
    return {"params": ["x", "v"], "primitives": [{"op": "SET", "target": "$arg:x", "attribute": attribute, "value": "$arg:v"}]}


def _copy_rule(attribute: str) -> dict[str, Any]:
    return {"params": ["x", "y"], "primitives": [{"op": "COPY", "source": "$arg:x", "target": "$arg:y", "attribute": attribute}]}


def _ere_chain(rng: random.Random, entity_count: int, depth: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    symbols = _ere_symbols(rng, entity_count)
    entities = symbols["entities"]
    attribute = symbols["attributes"][0]
    target, alternate = symbols["values"][:2]
    state = _state(entities, symbols["attributes"], symbols["values"], symbols["relation"], rng)
    state["attributes"][entities[0]][attribute] = alternate
    set_name, copy_name = symbols["rules"][:2]
    rules = {set_name: _set_rule(attribute), copy_name: _copy_rule(attribute)}
    path = entities[:depth]
    events = [{"rule": set_name, "arguments": {"x": path[0], "v": target}}]
    events.extend({"rule": copy_name, "arguments": {"x": path[index - 1], "y": path[index]}} for index in range(1, depth))
    ast = {
        "initial_state": state,
        "rules": rules,
        "events": events,
        "query": {"kind": "attribute", "entity": path[-1], "attribute": attribute},
    }
    alternate_ast = deepcopy(ast)
    alternate_ast["events"][0]["arguments"]["v"] = alternate
    metadata = {
        "pattern": "copy_chain",
        "declared_dependency_depth": depth,
        "entity_count": entity_count,
        "choice_values": list(symbols["values"]),
        "mutation_path": "/events/0/arguments/v",
    }
    return ast, alternate_ast, metadata


def _ere_if_chain(rng: random.Random, entity_count: int = 4) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    symbols = _ere_symbols(rng, entity_count)
    e = symbols["entities"]
    attribute = symbols["attributes"][0]
    target, alternate, fallback = symbols["values"][:3]
    control_value = symbols["values"][2]
    control_attribute = symbols["attributes"][1]
    state = _state(e, symbols["attributes"], symbols["values"], symbols["relation"], rng)
    state["attributes"][e[0]][attribute] = alternate
    state["attributes"][e[0]][control_attribute] = control_value
    # Pattern-specific control setup must not erase the visible witness for the
    # other control value.  This location is semantically inert for the branch.
    state["attributes"][e[1]][control_attribute] = symbols["values"][3]
    set_name, if_name, copy_name = symbols["rules"][:3]
    rules = {
        set_name: _set_rule(attribute),
        if_name: {
            "params": ["x", "y", "v"],
            "primitives": [
                {
                    "op": "IF",
                    "predicate": {"kind": "attribute_equals", "entity": "$arg:x", "attribute": control_attribute, "value": control_value},
                    "then": {"op": "COPY", "source": "$arg:x", "target": "$arg:y", "attribute": attribute},
                    "else": {"op": "SET", "target": "$arg:y", "attribute": attribute, "value": fallback},
                }
            ],
        },
        copy_name: _copy_rule(attribute),
    }
    ast = {
        "initial_state": state,
        "rules": rules,
        "events": [
            {"rule": set_name, "arguments": {"x": e[0], "v": target}},
            {"rule": if_name, "arguments": {"x": e[0], "y": e[1], "v": target}},
            {"rule": copy_name, "arguments": {"x": e[1], "y": e[2]}},
        ],
        "query": {"kind": "attribute", "entity": e[2], "attribute": attribute},
    }
    alternate_ast = deepcopy(ast)
    alternate_ast["events"][0]["arguments"]["v"] = alternate
    return ast, alternate_ast, {
        "pattern": "if_copy",
        "declared_dependency_depth": 3,
        "entity_count": entity_count,
        "choice_values": list(symbols["values"]),
        "mutation_path": "/events/0/arguments/v",
    }


def _ere_swap_chain(rng: random.Random, entity_count: int = 4, with_if: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    symbols = _ere_symbols(rng, entity_count)
    e = symbols["entities"]
    attribute = symbols["attributes"][0]
    target, alternate, fallback = symbols["values"][:3]
    state = _state(e, symbols["attributes"], symbols["values"], symbols["relation"], rng)
    state["attributes"][e[0]][attribute] = alternate
    state["attributes"][e[1]][attribute] = fallback
    set_name, swap_name, branch_name, copy_name = symbols["rules"][:4]
    rules = {
        set_name: _set_rule(attribute),
        swap_name: {"params": ["x", "y"], "primitives": [{"op": "SWAP", "left": "$arg:x", "right": "$arg:y", "attribute": attribute}]},
        copy_name: _copy_rule(attribute),
    }
    events = [
        {"rule": set_name, "arguments": {"x": e[0], "v": target}},
        {"rule": swap_name, "arguments": {"x": e[0], "y": e[1]}},
    ]
    if with_if:
        rules[branch_name] = {
            "params": ["x", "y", "v"],
            "primitives": [
                {
                    "op": "IF",
                    "predicate": {"kind": "attribute_equals", "entity": "$arg:x", "attribute": attribute, "value": "$arg:v"},
                    "then": {"op": "COPY", "source": "$arg:x", "target": "$arg:y", "attribute": attribute},
                    "else": {"op": "SET", "target": "$arg:y", "attribute": attribute, "value": fallback},
                }
            ],
        }
        events.append({"rule": branch_name, "arguments": {"x": e[1], "y": e[2], "v": target}})
        events.append({"rule": copy_name, "arguments": {"x": e[2], "y": e[3]}})
        query_entity = e[3]
        pattern = "swap_if_copy"
        depth = 4
    else:
        events.append({"rule": copy_name, "arguments": {"x": e[1], "y": e[2]}})
        query_entity = e[2]
        pattern = "swap_copy"
        depth = 3
    ast = {"initial_state": state, "rules": rules, "events": events, "query": {"kind": "attribute", "entity": query_entity, "attribute": attribute}}
    alternate_ast = deepcopy(ast)
    alternate_ast["events"][0]["arguments"]["v"] = alternate
    return ast, alternate_ast, {
        "pattern": pattern,
        "declared_dependency_depth": depth,
        "entity_count": entity_count,
        "choice_values": list(symbols["values"]),
        "mutation_path": "/events/0/arguments/v",
    }


def _ere_foreach(rng: random.Random, entity_count: int = 4, mutate_relation: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    symbols = _ere_symbols(rng, entity_count)
    e = symbols["entities"]
    attribute = symbols["attributes"][0]
    target, alternate = symbols["values"][:2]
    relation = symbols["relation"]
    state = _state(e, symbols["attributes"], symbols["values"], relation, rng)
    state["attributes"][e[0]][attribute] = alternate
    if not mutate_relation:
        state["relations"][relation] = [[e[0], e[1]]]
    set_name, link_name, spread_name, copy_name = symbols["rules"][:4]
    rules = {
        set_name: _set_rule(attribute),
        spread_name: {
            "params": ["x"],
            "primitives": [
                {
                    "op": "FOREACH_LINKED",
                    "relation": relation,
                    "source": "$arg:x",
                    "effect": {"op": "COPY", "source": "$arg:x", "target": "$neighbor", "attribute": attribute},
                }
            ],
        },
        copy_name: _copy_rule(attribute),
    }
    events = [{"rule": set_name, "arguments": {"x": e[0], "v": target}}]
    if mutate_relation:
        rules[link_name] = {"params": ["x", "y"], "primitives": [{"op": "LINK", "relation": relation, "source": "$arg:x", "target": "$arg:y"}]}
        events.append({"rule": link_name, "arguments": {"x": e[0], "y": e[1]}})
    events.extend(
        [
            {"rule": spread_name, "arguments": {"x": e[0]}},
            {"rule": copy_name, "arguments": {"x": e[1], "y": e[2]}},
        ]
    )
    ast = {"initial_state": state, "rules": rules, "events": events, "query": {"kind": "attribute", "entity": e[2], "attribute": attribute}}
    alternate_ast = deepcopy(ast)
    alternate_ast["events"][0]["arguments"]["v"] = alternate
    depth = 4 if mutate_relation else 3
    pattern = "link_foreach_copy" if mutate_relation else "existing_foreach_copy"
    return ast, alternate_ast, {
        "pattern": pattern,
        "declared_dependency_depth": depth,
        "entity_count": entity_count,
        "choice_values": list(symbols["values"]),
        "mutation_path": "/events/0/arguments/v",
    }


def _ere_relation_query(rng: random.Random, entity_count: int = 4) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    symbols = _ere_symbols(rng, entity_count)
    e = symbols["entities"]
    relation = symbols["relation"]
    state = _state(e, symbols["attributes"], symbols["values"], relation, rng)
    link_name, unlink_name = symbols["rules"][:2]
    rules = {
        link_name: {"params": ["x", "y"], "primitives": [{"op": "LINK", "relation": relation, "source": "$arg:x", "target": "$arg:y"}]},
        unlink_name: {"params": ["x", "y"], "primitives": [{"op": "UNLINK", "relation": relation, "source": "$arg:x", "target": "$arg:y"}]},
    }
    ast = {
        "initial_state": state,
        "rules": rules,
        "events": [
            {"rule": link_name, "arguments": {"x": e[0], "y": e[1]}},
            {"rule": link_name, "arguments": {"x": e[1], "y": e[2]}},
            {"rule": unlink_name, "arguments": {"x": e[0], "y": e[1]}},
        ],
        "query": {"kind": "relation", "relation": relation, "source": e[1], "target": e[2]},
    }
    alternate_ast = deepcopy(ast)
    alternate_ast["events"][1]["arguments"]["y"] = e[3]
    return ast, alternate_ast, {
        "pattern": "relation_query",
        "declared_dependency_depth": 1,
        "entity_count": entity_count,
        "choice_values": list(symbols["values"]),
        "mutation_path": "/events/1/arguments/y",
    }


def _validate_ere_visible_domain(
    pair: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ast, alternate_ast, metadata = pair
    declared = metadata.get("choice_values")
    if not isinstance(declared, list) or len(declared) != 6 or len(set(declared)) != 6:
        raise ValueError("ERE builder must declare six distinct visible values")
    for role, candidate in (("base", ast), ("alternate", alternate_ast)):
        visible = {
            value
            for attributes in candidate["initial_state"]["attributes"].values()
            for value in attributes.values()
        }
        if visible != set(declared):
            raise ValueError(f"ERE {role} visible-domain postcondition failed: {len(visible)}/6")
    return pair


def build_ere_pair(seed: int, split: str, index: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    if split == "composition_ood":
        pair = _ere_foreach(rng, mutate_relation=True) if index % 2 == 0 else _ere_swap_chain(rng, with_if=True)
    elif split == "length_ood":
        pair = _ere_chain(rng, 8, 8)
    elif split == "entity_ood":
        entity_count = 6 if index % 2 == 0 else 8
        pattern = index % 4
        if pattern == 0:
            pair = _ere_chain(rng, entity_count, 4)
        elif pattern == 1:
            pair = _ere_if_chain(rng, entity_count)
        elif pattern == 2:
            pair = _ere_swap_chain(rng, entity_count)
        else:
            pair = _ere_foreach(rng, entity_count)
    else:
        # Keep shallow relation-query diagnostics below ten percent so the
        # independently derived >=3-event dependency quota remains satisfiable.
        pattern = index % 12
        if pattern == 0:
            pair = _ere_relation_query(rng)
        elif pattern in {1, 2, 3, 4}:
            pair = _ere_chain(rng, 4 + index % 2, 3 + index % 2)
        elif pattern in {5, 6}:
            pair = _ere_if_chain(rng)
        elif pattern in {7, 8}:
            pair = _ere_swap_chain(rng)
        else:
            pair = _ere_foreach(rng)
    return _validate_ere_visible_domain(pair)


def _cps_symbols(rng: random.Random, depth: int) -> dict[str, Any]:
    factory = _NonceFactory(rng)
    return {
        "stages": factory.names(depth + 1),
        "safe": factory.name(),
        "irrelevant": factory.name(),
        "resource": factory.name(),
        "auxiliary_resource": factory.name(),
        "actions": factory.names(depth + 5),
    }


def build_cps_pair(seed: int, split: str, index: int, *, force_none: bool | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    if split == "horizon_ood":
        depth = 6 + index % 2
    else:
        depth = 3 + (1 if split == "composition_ood" and index % 2 else 0)
    symbols = _cps_symbols(rng, depth)
    stages = symbols["stages"]
    safe = symbols["safe"]
    irrelevant = symbols["irrelevant"]
    resource = symbols["resource"]
    auxiliary_resource = symbols["auxiliary_resource"]
    names = symbols["actions"]
    none_case = (index % 6 == 0) if force_none is None else force_none
    initial_resource = 0 if none_case else 1
    initial = {
        "attributes": {},
        "relations": {},
        "facts": [stages[0], safe],
        "resources": {resource: initial_resource, auxiliary_resource: 17 + rng.randrange(4)},
    }
    actions: list[dict[str, Any]] = []
    for step in range(depth):
        preconditions = [{"kind": "fact_true", "fact": stages[step]}]
        effects: list[dict[str, Any]] = [
            {"kind": "remove_fact", "fact": stages[step]},
            {"kind": "add_fact", "fact": stages[step + 1]},
        ]
        if step == 0:
            preconditions.append({"kind": "resource_at_least", "resource": resource, "amount": 1})
            effects.append({"kind": "resource_delta", "resource": resource, "delta": 2})
        if step == depth - 1:
            preconditions.append({"kind": "resource_at_least", "resource": resource, "amount": 3})
            effects.insert(0, {"kind": "resource_delta", "resource": resource, "delta": -3})
        actions.append({"name": names[step], "cost": 1 + rng.randrange(7), "preconditions": preconditions, "effects": effects})

    final = actions[-1]
    alternate_name = names[depth]
    bad_name = names[depth + 1]
    decoy_name = names[depth + 2]
    alternate = deepcopy(final)
    alternate["name"] = alternate_name
    alternate["cost"] = final["cost"] + 2 + rng.randrange(4)
    bad = deepcopy(final)
    bad["name"] = bad_name
    bad["cost"] = final["cost"] + 1 + rng.randrange(2)
    bad["effects"] = [*bad["effects"], {"kind": "remove_fact", "fact": safe}]
    decoy = {
        "name": decoy_name,
        "cost": 4 + rng.randrange(7),
        "preconditions": [{"kind": "fact_true", "fact": safe}],
        "effects": [{"kind": "add_fact", "fact": irrelevant}],
    }
    actions.extend([alternate, bad, decoy])
    chain = [action["name"] for action in actions[:depth]]
    candidates = [
        {"plan": chain},
        {"plan": [*chain[:-1], alternate_name]},
        {"plan": list(reversed(chain))},
        {"plan": chain[1:]},
        {"plan": [*chain[:-1], bad_name]},
    ]
    if split == "composition_ood":
        # This course reaches the same safe goal but exceeds the budget solely
        # because the semantically irrelevant final action costs eleven units.
        decoy["cost"] = 11
        candidates.append({"plan": [*chain, decoy_name]})
    if split == "distractor_ood":
        extra_names = names[depth + 3 : depth + 5]
        for offset, extra in enumerate(extra_names):
            actions.append(
                {
                    "name": extra,
                    "cost": 6 + offset,
                    "preconditions": [{"kind": "fact_true", "fact": irrelevant}],
                    "effects": [{"kind": "add_fact", "fact": stages[min(depth, offset + 1)]}],
                }
            )
        candidates.extend(
            [
                {"plan": [decoy_name]},
                {"plan": [*chain, decoy_name]},
                {"plan": [chain[0], decoy_name, *chain[1:]]},
            ]
        )

    pstar_cost = sum(action["cost"] for action in actions[:depth])
    ast = {
        "initial_state": initial,
        "actions": actions,
        "candidates": candidates,
        # The original P* remains valid after the largest permitted cost
        # intervention; the alternate can therefore win against a still-valid
        # suboptimal plan instead of winning by disqualification.
        "budget": pstar_cost + 10,
        "goal": [{"kind": "fact_true", "fact": stages[-1]}],
        "final_constraints": [{"kind": "fact_true", "fact": safe}],
    }
    rng.shuffle(ast["actions"])
    rng.shuffle(ast["candidates"])
    alternate_ast = deepcopy(ast)
    if none_case:
        alternate_ast["initial_state"]["resources"][resource] = 1
        mutation_path = f"/initial_state/resources/{resource}"
        mutation_kind = "resource"
    else:
        alternate_cost = next(action["cost"] for action in ast["actions"] if action["name"] == alternate_name)
        for action in alternate_ast["actions"]:
            if action["name"] == final["name"]:
                action["cost"] = alternate_cost + 3
                break
        mutation_path = f"/actions/name={final['name']}/cost"
        mutation_kind = "cost"
    metadata = {
        "pattern": "resource_final_budget",
        "plan_depth": depth,
        "action_count": len(ast["actions"]),
        "candidate_count": len(ast["candidates"]),
        "none_case": none_case,
        "mutation_path": mutation_path,
        "mutation_kind": mutation_kind,
    }
    return ast, alternate_ast, metadata


def _semantic_answer(family: str, output: Mapping[str, Any]) -> str:
    if family == "ERE":
        answer = output["answer"]
        if isinstance(answer, bool):
            return "true" if answer else "false"
        if not isinstance(answer, str):
            raise ValueError("ERE answer must be a value atom or boolean")
        return f"value:{answer}"
    answer = output["answer"]
    if answer is None:
        if any(candidate["valid"] for candidate in output["candidates"]):
            raise ValueError("CPS NONE cannot represent an optimum tie")
        return "none"
    if output["unique_optimum"] is not True:
        raise ValueError("CPS non-NONE answer must be unique")
    return f"candidate:{answer}"


def _choice_options(family: str, ast: Mapping[str, Any], answer: str, alternate_answer: str, values: Sequence[str] | None = None) -> list[str]:
    if family == "CPS":
        return [*(f"candidate:{index}" for index in range(len(ast["candidates"]))), "none"]
    if ast["query"]["kind"] == "relation":
        return ["true", "false"]
    candidates = [f"value:{value}" for value in (values or [])]
    for required in (answer, alternate_answer):
        if required not in candidates:
            candidates.append(required)
    if len(candidates) < 4:
        raise ValueError("attribute query requires four value choices")
    return candidates[:4]


def _mapping(
    options: Sequence[str],
    targets: Mapping[str, str],
    rng: random.Random,
    *,
    target_position: int | None = None,
) -> dict[str, str]:
    if len(options) > len(LABELS) or len(set(options)) != len(options) or len(set(targets.values())) != len(targets):
        raise ValueError("invalid choice target contract")
    if any(option not in options or label not in LABELS for option, label in targets.items()):
        raise ValueError("choice target outside domain")
    used_labels = set(targets.values())
    remaining_labels = [label for label in LABELS if label not in used_labels]
    rng.shuffle(remaining_labels)
    semantic_to_label = dict(targets)
    for option in options:
        if option not in semantic_to_label:
            semantic_to_label[option] = remaining_labels.pop()
    items = [(semantic_to_label[option], option) for option in options]
    rng.shuffle(items)
    if target_position is not None:
        if len(targets) != 1:
            raise ValueError("one stratified target is required")
        target_label = next(iter(targets.values()))
        target_item = next(item for item in items if item[0] == target_label)
        items.remove(target_item)
        items.insert(target_position % (len(items) + 1), target_item)
    return dict(items)


def _balanced_label_schedule(count: int, seed: int, *, paired: bool) -> list[str]:
    if count <= 0 or (paired and count % 2):
        raise ValueError("label schedule count must be positive and paired counts must be even")
    rng = random.Random(seed ^ 0x5CA1AB1E)
    quotient, remainder = divmod(count, len(LABELS))
    extras = list(LABELS)
    rng.shuffle(extras)
    schedule = [label for label in LABELS for _ in range(quotient)] + extras[:remainder]
    if not paired:
        rng.shuffle(schedule)
        return schedule

    remaining = Counter(schedule)
    pairs: list[list[str]] = []
    for _ in range(count // 2):
        tie_order = list(LABELS)
        rng.shuffle(tie_order)
        rank = {label: index for index, label in enumerate(tie_order)}
        active = [label for label in LABELS if remaining[label] > 0]
        active.sort(key=lambda label: (-remaining[label], rank[label]))
        if len(active) < 2:
            raise RuntimeError("could not construct distinct paired label schedule")
        left, right = active[:2]
        remaining[left] -= 1
        remaining[right] -= 1
        pairs.append([left, right])
    rng.shuffle(pairs)
    for pair in pairs:
        if rng.randrange(2):
            pair.reverse()
    return [label for pair in pairs for label in pair]


def _label_for(mapping: Mapping[str, str], semantic: str) -> str:
    labels = [label for label, meaning in mapping.items() if meaning == semantic]
    if len(labels) != 1:
        raise ValueError("semantic choice does not have one label")
    return labels[0]


def _ere_claims(ast: Mapping[str, Any], values: Sequence[str], record_id: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for pair_index, prefix in enumerate(sorted({0, len(ast["events"]) // 2, len(ast["events"])})):
        state = simulate_ere(ast, prefix)["final_state"]
        slots = [
            (entity, attribute, value)
            for entity, attributes in state["attributes"].items()
            for attribute, value in attributes.items()
        ]
        left = slots[0]
        right = next(slot for slot in slots[1:] if slot[2] != left[2])
        for slot_index, ((entity, attribute, actual), alternate) in enumerate(((left, right[2]), (right, left[2]))):
            pair_id = f"{record_id}-claim-{pair_index}-{slot_index}"
            for role, value, label in (("positive", actual, True), ("negative", alternate, False)):
                claims.append(
                    {
                        "claim_id": f"{pair_id}-{role}",
                        "pair_id": pair_id,
                        "pair_role": role,
                        "kind": "ere_attribute_equals",
                        "label": label,
                        "prefix": prefix,
                        "predicate": {"entity": entity, "attribute": attribute, "value": value},
                        "text": f"After prefix {prefix}, entity {entity} has attribute {attribute} equal to {value}.",
                    }
                )
    return claims


def _cps_claims(ast: Mapping[str, Any], record_id: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    output = evaluate_cps(ast)
    candidate_index = output["answer"]
    if candidate_index is not None:
        before = replay_cps_prefix(ast, candidate_index, 0)["state"]
        after = replay_cps_prefix(ast, candidate_index, 1)["state"]
        removed = sorted(set(before["facts"]) - set(after["facts"]))
        added = sorted(set(after["facts"]) - set(before["facts"]))
        if not removed or not added:
            raise ValueError("winning CPS plan lacks a fact-transition claim witness")
        fact_contexts = ((0, removed[0], added[0]), (1, added[0], removed[0]))
        for context_index, (prefix, actual, alternate) in enumerate(fact_contexts):
            pair_id = f"{record_id}-fact-{context_index}"
            for role, fact, label in (("positive", actual, True), ("negative", alternate, False)):
                claims.append(
                    {
                        "claim_id": f"{pair_id}-{role}",
                        "pair_id": pair_id,
                        "pair_role": role,
                        "kind": "cps_fact_holds",
                        "label": label,
                        "candidate_index": candidate_index,
                        "prefix": prefix,
                        "predicate": {"fact": fact},
                        "text": f"At prefix {prefix} of candidate {candidate_index + 1}, fact {fact} is present.",
                    }
                )

    resource_candidate = candidate_index if candidate_index is not None else 0
    plan_length = len(ast["candidates"][resource_candidate]["plan"])
    for pair_index, prefix in enumerate(sorted({0, plan_length})):
        replay = replay_cps_prefix(ast, resource_candidate, prefix)
        state = replay["state"]
        resource_rows = list(state["resources"].items())
        left = resource_rows[0]
        right = next(item for item in resource_rows[1:] if item[1] != left[1])
        for resource_index, ((resource, amount), alternate) in enumerate(((left, right[1]), (right, left[1]))):
            pair_id = f"{record_id}-resource-{pair_index}-{resource_index}"
            for role, value, label in (("positive", amount, True), ("negative", alternate, False)):
                claims.append(
                    {
                        "claim_id": f"{pair_id}-{role}",
                        "pair_id": pair_id,
                        "pair_role": role,
                        "kind": "cps_resource_equals",
                        "label": label,
                        "candidate_index": resource_candidate,
                        "prefix": prefix,
                        "predicate": {"resource": resource, "amount": value},
                        "text": f"At prefix {prefix} of candidate {resource_candidate + 1}, resource {resource} equals {value}.",
                    }
                )
    return claims


def verify_claim(ast: Mapping[str, Any], family: str, claim: Mapping[str, Any]) -> bool:
    kind = claim["kind"]
    predicate = claim["predicate"]
    if family == "ERE" and kind == "ere_attribute_equals":
        state = simulate_ere(ast, int(claim["prefix"]))["final_state"]
        return state["attributes"].get(predicate["entity"], {}).get(predicate["attribute"]) == predicate["value"]
    if family == "CPS":
        state = replay_cps_prefix(ast, int(claim["candidate_index"]), int(claim["prefix"]))["state"]
        if kind == "cps_fact_holds":
            return predicate["fact"] in state["facts"]
        if kind == "cps_resource_equals":
            return state["resources"].get(predicate["resource"], 0) == predicate["amount"]
    raise ValueError(f"unknown claim kind {kind!r} for {family}")


def _record(
    *,
    family: str,
    split: str,
    index: int,
    record_seed: int,
    ast: Mapping[str, Any],
    alternate_ast: Mapping[str, Any],
    generation_metadata: Mapping[str, Any],
    mapping: Mapping[str, str],
    template_id: str,
    tokenizer: Any,
    example_id: str,
    pair_id: str | None,
    pair_role: str | None,
    design_sha256: str,
    root_seed: int,
    family_index: int,
    split_index: int,
    split_seed: int,
    unit_index: int,
    generation_attempt: int,
) -> dict[str, Any]:
    differences = _leaf_differences(ast, alternate_ast)
    if len(differences) != 1:
        raise ValueError(f"counterfactual must change exactly one leaf, got {len(differences)}")
    mutation = differences[0]
    output = simulate_ere(ast) if family == "ERE" else evaluate_cps(ast)
    alternate_output = simulate_ere(alternate_ast) if family == "ERE" else evaluate_cps(alternate_ast)
    answer = _semantic_answer(family, output)
    alternate_answer = _semantic_answer(family, alternate_output)
    if answer == alternate_answer:
        raise ValueError("designated counterfactual did not change the answer")
    source_text = render_source(family, ast, mapping, template_id)
    token_count = len(tokenizer.encode(source_text, add_special_tokens=False))
    if token_count > 1024:
        raise OverflowError(f"source exceeds 1024 tokens: {token_count}")
    label = _label_for(mapping, answer)
    mask = [candidate in mapping for candidate in LABELS]
    if family == "ERE":
        values = sorted({value for attributes in ast["initial_state"]["attributes"].values() for value in attributes.values()} | {item[6:] for item in mapping.values() if item.startswith("value:")})
        claims = _ere_claims(ast, values, example_id)
        teacher_trace = output["trace"]
    else:
        claims = _cps_claims(ast, example_id)
        teacher_trace = output["candidates"]
    if any(verify_claim(ast, family, claim) is not claim["label"] for claim in claims):
        raise ValueError("claim construction disagrees with fresh replay")
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "example_id": example_id,
        "family": family,
        "split": split,
        "record_seed": record_seed,
        "template_id": template_id,
        "source_text": source_text,
        "token_count": token_count,
        "reasoning_budget": min(24, (len(ast["events"]) if family == "ERE" else max(len(candidate["plan"]) for candidate in ast["candidates"])) + 2),
        "valid_choice_mask": mask,
        "answer_index": LABELS.index(label),
        "semantic_answer": answer,
        "semantic_fingerprint": semantic_fingerprint(deepcopy(dict(ast))),
        "surface_fingerprint": surface_fingerprint(source_text),
        "pair_id": pair_id,
        "pair_role": pair_role,
        "program_ast": deepcopy(dict(ast)),
        "label_mapping": dict(mapping),
        "teacher_trace": teacher_trace,
        "training_claims": claims,
        "causal_certificate": {
            "mutation_path": mutation["path"],
            "declared_mutation_path": generation_metadata["mutation_path"],
            "before_value": mutation["before"],
            "after_value": mutation["after"],
            "before_answer": answer,
            "after_answer": alternate_answer,
            "before_output_sha256": hashlib.sha256(_canonical(output)).hexdigest().upper(),
            "after_output_sha256": hashlib.sha256(_canonical(alternate_output)).hexdigest().upper(),
            "pair_id": pair_id,
        },
        "simulator_output": output,
        "generation_metadata": dict(generation_metadata),
        "provenance": {
            "generator_version": GENERATOR_VERSION,
            "root_seed": root_seed,
            "family_index": family_index,
            "split_index": split_index,
            "split_seed": split_seed,
            "unit_index": unit_index,
            "generation_attempt": generation_attempt,
            "record_seed": record_seed,
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "design_sha256": design_sha256,
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        },
    }


def _template(split: str, index: int) -> str:
    return "indirect_v1" if split == "language_ood" else TRAIN_TEMPLATES[index % len(TRAIN_TEMPLATES)]


def _build_family_split(
    *,
    family: str,
    split: str,
    count: int,
    seed: int,
    root_seed: int,
    family_index: int,
    split_index: int,
    tokenizer: Any,
    design_sha256: str,
    seen_semantic: set[str],
    seen_surface: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    label_schedule = _balanced_label_schedule(count, seed, paired=split == "causal_pairs")
    attempts_total = 0
    attempts_max = 0
    overlap_rejections = 0
    overflow_rejections = 0
    if split == "causal_pairs":
        if count % 2:
            raise ValueError("causal pair record count must be even")
        for pair_index in range(count // 2):
            for attempt in range(10_000):
                record_seed = seed + pair_index * _UNIT_SEED_STRIDE + attempt * _ATTEMPT_SEED_STRIDE
                if family == "ERE":
                    ast, alternate_ast, metadata = build_ere_pair(record_seed, "train", pair_index + 1)
                    base_output = simulate_ere(ast)
                    alternate_output = simulate_ere(alternate_ast)
                    values = metadata["choice_values"]
                    options = _choice_options(family, ast, _semantic_answer(family, base_output), _semantic_answer(family, alternate_output), values)
                else:
                    ast, alternate_ast, metadata = build_cps_pair(record_seed, "train", pair_index + 1, force_none=False)
                    base_output = evaluate_cps(ast)
                    alternate_output = evaluate_cps(alternate_ast)
                    options = _choice_options(family, ast, _semantic_answer(family, base_output), _semantic_answer(family, alternate_output))
                base_answer = _semantic_answer(family, base_output)
                alternate_answer = _semantic_answer(family, alternate_output)
                target_base = label_schedule[pair_index * 2]
                target_alternate = label_schedule[pair_index * 2 + 1]
                mapping = _mapping(options, {base_answer: target_base, alternate_answer: target_alternate}, random.Random(record_seed ^ 0xC0A5A1))
                template_id = TRAIN_TEMPLATES[pair_index % len(TRAIN_TEMPLATES)]
                pair_id = f"{family.lower()}-causal-{pair_index:04d}"
                base = _record(
                    family=family,
                    split=split,
                    index=pair_index * 2,
                    record_seed=record_seed,
                    ast=ast,
                    alternate_ast=alternate_ast,
                    generation_metadata=metadata,
                    mapping=mapping,
                    template_id=template_id,
                    tokenizer=tokenizer,
                    example_id=f"{pair_id}-base",
                    pair_id=pair_id,
                    pair_role="base",
                    design_sha256=design_sha256,
                    root_seed=root_seed,
                    family_index=family_index,
                    split_index=split_index,
                    split_seed=seed,
                    unit_index=pair_index,
                    generation_attempt=attempt,
                )
                reverse_metadata = dict(metadata)
                reverse_metadata["mutation_path"] = metadata["mutation_path"]
                flip = _record(
                    family=family,
                    split=split,
                    index=pair_index * 2 + 1,
                    record_seed=record_seed,
                    ast=alternate_ast,
                    alternate_ast=ast,
                    generation_metadata=reverse_metadata,
                    mapping=mapping,
                    template_id=template_id,
                    tokenizer=tokenizer,
                    example_id=f"{pair_id}-flip",
                    pair_id=pair_id,
                    pair_role="flip",
                    design_sha256=design_sha256,
                    root_seed=root_seed,
                    family_index=family_index,
                    split_index=split_index,
                    split_seed=seed,
                    unit_index=pair_index,
                    generation_attempt=attempt,
                )
                if any(row["semantic_fingerprint"] in seen_semantic or row["surface_fingerprint"] in seen_surface for row in (base, flip)):
                    overlap_rejections += 1
                    continue
                for row in (base, flip):
                    seen_semantic.add(row["semantic_fingerprint"])
                    seen_surface.add(row["surface_fingerprint"])
                records.extend((base, flip))
                attempts_total += attempt + 1
                attempts_max = max(attempts_max, attempt + 1)
                break
            else:
                raise RuntimeError(f"could not generate unique pair {family}/{pair_index}")
        return records, {
            "unit_count": count // 2,
            "attempts_total": attempts_total,
            "attempts_max": attempts_max,
            "overlap_rejections": overlap_rejections,
            "overflow_rejections": overflow_rejections,
        }

    for index in range(count):
        for attempt in range(10_000):
            record_seed = seed + index * _UNIT_SEED_STRIDE + attempt * _ATTEMPT_SEED_STRIDE
            if family == "ERE":
                ast, alternate_ast, metadata = build_ere_pair(record_seed, split, index)
                output = simulate_ere(ast)
                alternate_output = simulate_ere(alternate_ast)
                values = metadata["choice_values"]
                options = _choice_options(family, ast, _semantic_answer(family, output), _semantic_answer(family, alternate_output), values)
            else:
                ast, alternate_ast, metadata = build_cps_pair(record_seed, split, index)
                output = evaluate_cps(ast)
                alternate_output = evaluate_cps(alternate_ast)
                options = _choice_options(family, ast, _semantic_answer(family, output), _semantic_answer(family, alternate_output))
            answer = _semantic_answer(family, output)
            target = label_schedule[index]
            mapping = _mapping(
                options,
                {answer: target},
                random.Random(record_seed ^ 0x1ABE1),
                target_position=index % len(options),
            )
            try:
                record = _record(
                    family=family,
                    split=split,
                    index=index,
                    record_seed=record_seed,
                    ast=ast,
                    alternate_ast=alternate_ast,
                    generation_metadata=metadata,
                    mapping=mapping,
                    template_id=_template(split, index),
                    tokenizer=tokenizer,
                    example_id=f"{family.lower()}-{split}-{index:04d}",
                    pair_id=None,
                    pair_role=None,
                    design_sha256=design_sha256,
                    root_seed=root_seed,
                    family_index=family_index,
                    split_index=split_index,
                    split_seed=seed,
                    unit_index=index,
                    generation_attempt=attempt,
                )
            except OverflowError:
                overflow_rejections += 1
                continue
            if record["semantic_fingerprint"] in seen_semantic or record["surface_fingerprint"] in seen_surface:
                overlap_rejections += 1
                continue
            seen_semantic.add(record["semantic_fingerprint"])
            seen_surface.add(record["surface_fingerprint"])
            records.append(record)
            attempts_total += attempt + 1
            attempts_max = max(attempts_max, attempt + 1)
            break
        else:
            raise RuntimeError(f"could not generate unique record {family}/{split}/{index}")
    return records, {
        "unit_count": count,
        "attempts_total": attempts_total,
        "attempts_max": attempts_max,
        "overlap_rejections": overlap_rejections,
        "overflow_rejections": overflow_rejections,
    }


def generate_production_dataset(
    root: Path,
    *,
    spec: DatasetSpec,
    design_doc: Path,
    tokenizer: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate one immutable preflight or formal production dataset."""

    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)
    if not isinstance(spec, DatasetSpec) or spec.train_count <= 0 or spec.heldout_count <= 0 or spec.heldout_count % 2:
        raise ValueError("invalid dataset spec")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    emit = progress or (lambda _: None)
    design_sha256 = _sha_file(design_doc)
    seen_semantic: set[str] = set()
    seen_surface: set[str] = set()
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    generation_stats: dict[str, dict[str, int]] = {}
    for family_index, family in enumerate(("ERE", "CPS")):
        for split_index, split in enumerate(SPLITS[family]):
            emit(f"generate:{family}:{split}:start")
            count = spec.count(split)
            derived_seed = spec.root_seed + family_index * _FAMILY_SEED_STRIDE + split_index * _SPLIT_SEED_STRIDE
            records, stats = _build_family_split(
                family=family,
                split=split,
                count=count,
                seed=derived_seed,
                root_seed=spec.root_seed,
                family_index=family_index,
                split_index=split_index,
                tokenizer=tokenizer,
                design_sha256=design_sha256,
                seen_semantic=seen_semantic,
                seen_surface=seen_surface,
            )
            relative = f"{family.lower()}-{split}.jsonl"
            path = root / relative
            path.write_bytes(b"".join(_canonical(record) for record in records))
            files[relative] = _sha_file(path)
            counts[f"{family}/{split}"] = len(records)
            generation_stats[f"{family}/{split}"] = stats
            emit(f"generate:{family}:{split}:done")
    manifest = {
        "schema_version": "yggdrasil.v2-r1r.p0d-v17.dataset-manifest.v1",
        "generator_version": GENERATOR_VERSION,
        "profile": spec.profile,
        "root_seed": spec.root_seed,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "add_special_tokens": False,
        "design_sha256": design_sha256,
        "counts": counts,
        "total_records": sum(counts.values()),
        "generation_stats": generation_stats,
        "files": files,
    }
    (root / "manifest.json").write_bytes(_canonical(manifest))
    return manifest


def load_production_dataset(root: Path) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dataset: dict[str, list[dict[str, Any]]] = {}
    for relative, expected in manifest["files"].items():
        path = root / relative
        if _sha_file(path) != expected:
            raise ValueError(f"dataset file hash mismatch: {relative}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        dataset[relative.removesuffix(".jsonl")] = rows
    return dataset


__all__ = [
    "DatasetSpec",
    "FORMAL_SPEC",
    "GENERATOR_VERSION",
    "MODEL_ID",
    "MODEL_REVISION",
    "SCHEMA_VERSION",
    "SEED_DERIVATION_VERSION",
    "SPLITS",
    "build_cps_pair",
    "build_ere_pair",
    "generate_production_dataset",
    "load_production_dataset",
    "verify_claim",
]
