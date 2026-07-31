from __future__ import annotations

"""Causal-spine ERE generation and deterministic replay metadata."""

from copy import deepcopy
import random
from typing import Any, Mapping, Sequence

from .render import (
    OOD_ERE_TEMPLATES,
    TRAIN_ERE_TEMPLATES,
    contract_token_count,
    render_ere,
    semantic_fingerprint,
    surface_fingerprint,
)
from .schema import GENERATOR_VERSION, LABELS, SCHEMA_VERSION
from .simulator import simulate_ere
from .symbols import NonceFactory, balanced_label_index, balanced_label_mapping, balanced_pair_mapping, label_index, label_permutation, valid_choice_mask


ERE_SPLITS = (
    "train",
    "validation",
    "composition_ood",
    "length_ood",
    "entity_ood",
    "language_ood",
    "causal_pairs",
)


def _split_pattern(split: str, index: int) -> str:
    if split == "composition_ood":
        return ("relation", "swap_if", "if")[index % 3]
    if split == "length_ood":
        return "chain"
    if split == "language_ood":
        return ("relation", "swap_if", "if")[index % 3]
    return ("chain", "if", "swap")[index % 3]


def _make_rule_set(
    *,
    pattern: str,
    seed_rule: str,
    copy_rule: str,
    swap_rule: str,
    link_rule: str,
    spread_rule: str,
    gate_rule: str,
    distractor_rule: str,
    attribute: str,
    other_attribute: str,
    relation: str,
    target_value: str,
    alternate_value: str,
    distractor_value: str,
) -> dict[str, Any]:
    rules: dict[str, Any] = {
        seed_rule: {
            "params": ["target"],
            "primitives": [{"op": "SET", "target": "$arg:target", "attribute": attribute, "value": target_value}],
        },
        distractor_rule: {
            "params": ["target"],
            "primitives": [{"op": "SET", "target": "$arg:target", "attribute": other_attribute, "value": distractor_value}],
        },
    }
    if pattern in {"chain", "if", "swap", "swap_if"}:
        rules[copy_rule] = {
            "params": ["source", "target"],
            "primitives": [{"op": "COPY", "source": "$arg:source", "target": "$arg:target", "attribute": attribute}],
        }
    if pattern in {"swap", "swap_if"}:
        rules[swap_rule] = {
            "params": ["left", "right"],
            "primitives": [{"op": "SWAP", "left": "$arg:left", "right": "$arg:right", "attribute": attribute}],
        }
    if pattern == "relation":
        rules[link_rule] = {
            "params": ["source", "target"],
            "primitives": [{"op": "LINK", "relation": relation, "source": "$arg:source", "target": "$arg:target"}],
        }
        rules[spread_rule] = {
            "params": ["source"],
            "primitives": [{
                "op": "FOREACH_LINKED",
                "relation": relation,
                "source": "$arg:source",
                "effect": {"op": "COPY", "source": "$arg:source", "target": "$neighbor", "attribute": attribute},
            }],
        }
        rules[copy_rule] = {
            "params": ["source", "target"],
            "primitives": [{"op": "COPY", "source": "$arg:source", "target": "$arg:target", "attribute": attribute}],
        }
    if pattern in {"if", "swap_if"}:
        rules[gate_rule] = {
            "params": ["source", "target"],
            "primitives": [{
                "op": "IF",
                "predicate": {"kind": "attr_equals", "entity": "$arg:source", "attribute": attribute, "value": target_value},
                "then": {"op": "COPY", "source": "$arg:source", "target": "$arg:target", "attribute": attribute},
                "else": {"op": "SET", "target": "$arg:target", "attribute": attribute, "value": alternate_value},
            }],
        }
    return rules


def _build_ast(seed: int, split: str, index: int, *, mutated: bool = False) -> dict[str, Any]:
    rng = random.Random(seed)
    factory = NonceFactory(rng, set())
    entity_count = rng.choice((6, 8)) if split == "entity_ood" else rng.randint(3, 5)
    entities = factory.names(entity_count)
    attributes = [factory.name(), factory.name()]
    values = [factory.name() for _ in range(8)]
    value_sets = [values[:4], values[4:]]
    relation = factory.name()
    rule_names = factory.names(7)
    seed_rule, copy_rule, swap_rule, link_rule, spread_rule, gate_rule, distractor_rule = rule_names
    pattern = _split_pattern(split, index)
    if split == "length_ood":
        chain_length = 8 + (index % 13)
    elif split == "composition_ood":
        chain_length = 4 + (index % 3)
    else:
        chain_length = 3 + (index % 3)
    target_value = rng.choice(value_sets[0])
    alternate_value = rng.choice([value for value in value_sets[0] if value != target_value])
    distractor_value = rng.choice(value_sets[1])
    if mutated:
        target_value, alternate_value = alternate_value, target_value

    initial_attributes: dict[str, dict[str, str]] = {}
    for entity in entities:
        initial_attributes[entity] = {
            attributes[0]: rng.choice(value_sets[0]),
            attributes[1]: rng.choice(value_sets[1]),
        }
    actors = rng.sample(entities, 3)
    query_entity = actors[-1]
    events: list[dict[str, Any]] = []
    spine_indices: list[int] = []
    if pattern == "chain":
        events.append({"rule": seed_rule, "arguments": {"target": entities[0]}})
        spine_indices.append(0)
        current = entities[0]
        for step in range(1, chain_length):
            target = rng.choice(entities)
            events.append({"rule": copy_rule, "arguments": {"source": current, "target": target}})
            spine_indices.append(len(events) - 1)
            current = target
        query_entity = current
    elif pattern == "if":
        events.extend(
            [
                {"rule": seed_rule, "arguments": {"target": actors[0]}},
                {"rule": gate_rule, "arguments": {"source": actors[0], "target": actors[1]}},
                {"rule": copy_rule, "arguments": {"source": actors[1], "target": actors[2]}},
            ]
        )
        spine_indices.extend([0, 1, 2])
        query_entity = actors[2]
    elif pattern == "swap":
        events.extend(
            [
                {"rule": seed_rule, "arguments": {"target": actors[0]}},
                {"rule": swap_rule, "arguments": {"left": actors[0], "right": actors[1]}},
                {"rule": copy_rule, "arguments": {"source": actors[1], "target": actors[2]}},
            ]
        )
        spine_indices.extend([0, 1, 2])
        query_entity = actors[2]
    elif pattern == "swap_if":
        events.extend(
            [
                {"rule": seed_rule, "arguments": {"target": actors[0]}},
                {"rule": swap_rule, "arguments": {"left": actors[0], "right": actors[1]}},
                {"rule": gate_rule, "arguments": {"source": actors[1], "target": actors[2]}},
            ]
        )
        spine_indices.extend([0, 1, 2])
        query_entity = actors[2]
    elif pattern == "relation":
        events.extend(
            [
                {"rule": seed_rule, "arguments": {"target": actors[0]}},
                {"rule": link_rule, "arguments": {"source": actors[0], "target": actors[1]}},
                {"rule": spread_rule, "arguments": {"source": actors[0]}},
                {"rule": copy_rule, "arguments": {"source": actors[1], "target": actors[2]}},
            ]
        )
        spine_indices.extend([0, 1, 2, 3])
        query_entity = actors[2]
    else:
        raise ValueError(f"unknown ERE pattern: {pattern}")

    while len(events) < chain_length:
        target = rng.choice(entities)
        events.append({"rule": distractor_rule, "arguments": {"target": target}})

    rules = _make_rule_set(
        pattern=pattern,
        seed_rule=seed_rule,
        copy_rule=copy_rule,
        swap_rule=swap_rule,
        link_rule=link_rule,
        spread_rule=spread_rule,
        gate_rule=gate_rule,
        distractor_rule=distractor_rule,
        attribute=attributes[0],
        other_attribute=attributes[1],
        relation=relation,
        target_value=target_value,
        alternate_value=alternate_value,
        distractor_value=distractor_value,
    )
    ast = {
        "kind": "ere",
        "entities": entities,
        "attributes": [
            {"name": attributes[0], "values": value_sets[0]},
            {"name": attributes[1], "values": value_sets[1]},
        ],
        "relations": [relation],
        "initial_state": {
            "attributes": initial_attributes,
            "relations": {relation: []},
            "facts": [],
            "resources": {},
        },
        "rules": rules,
        "events": events,
        "query": {"kind": "attribute", "entity": query_entity, "attribute": attributes[0]},
        "dependency": {
            "depth": len(spine_indices),
            "spine_event_indices": spine_indices,
            "necessary_event_indices": [0],
        },
        "composition_features": (
            ["relation_mutation_foreach", "relation_and_attribute_dependency"]
            if pattern == "relation"
            else ["swap_before_condition"]
            if pattern == "swap_if"
            else ["condition_copy"]
            if pattern == "if"
            else ["attribute_chain"]
        ),
        "causal_target": {
            "event_index": 0,
            "primitive_index": 0,
            "field": "value",
            "original": target_value,
            "alternate": alternate_value,
        },
    }
    return ast


def mutate_ere_ast(ast: Mapping[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(dict(ast))
    target = mutated["causal_target"]
    event = mutated["events"][int(target["event_index"])]
    rule = mutated["rules"][event["rule"]]
    primitive = rule["primitives"][int(target["primitive_index"])]
    current = primitive.get(target["field"])
    primitive[target["field"]] = target["alternate"] if current == target["original"] else target["original"]
    return mutated


def _claims(ast: Mapping[str, Any], simulation: Mapping[str, Any]) -> list[dict[str, Any]]:
    query = ast["query"]
    values = ast["attributes"][0]["values"]
    alternate = next(value for value in values if value != simulation["answer"])
    claims: list[dict[str, Any]] = []
    initial_state = ast["initial_state"]
    states = [initial_state] + [item["after"] for item in simulation["trace"]]
    for prefix, state in enumerate(states):
        value = state["attributes"][query["entity"]][query["attribute"]]
        claims.extend(
            [
                {"prefix": prefix, "kind": "attribute_value", "text": f"{query['entity']} has {query['attribute']} equal to {value}", "label": True},
                {"prefix": prefix, "kind": "attribute_value", "text": f"{query['entity']} has {query['attribute']} equal to {alternate if value == simulation['answer'] else simulation['answer']}", "label": False},
            ]
        )
    return claims


def _record_from_ast(
    ast: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    index: int,
    example_id: str,
    template_id: str,
    design_doc_sha256: str,
    command: str,
    environment: Mapping[str, Any],
    pair_id: str | None = None,
    pair_role: str | None = None,
    label_targets: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    simulation = simulate_ere(ast)
    values = list(ast["attributes"][0]["values"])
    mapping = label_permutation(values, random.Random(seed + index * 9973), target_labels=label_targets or {simulation["answer"]: "A"})
    source_text = render_ere(ast, mapping, template_id)
    token_count = contract_token_count(source_text)
    if token_count > 1024:
        raise ValueError(f"ERE generator rejected a record over 1024 tokens: {token_count}")
    claims = _claims(ast, simulation)
    alternate_ast = mutate_ere_ast(ast)
    alternate_simulation = simulate_ere(alternate_ast)
    if alternate_simulation["answer"] == simulation["answer"]:
        raise ValueError("ERE causal spine mutation did not flip the semantic answer")
    record = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "example_id": example_id,
        "family": "ere",
        "split": split,
        "source_text": source_text,
        "token_count": token_count,
        "reasoning_budget": min(24, len(ast["events"]) + 2),
        "valid_choice_mask": valid_choice_mask(mapping),
        "answer_index": label_index(mapping[simulation["answer"]]),
        "semantic_fingerprint": semantic_fingerprint(ast),
        "surface_fingerprint": surface_fingerprint(source_text),
        "pair_id": pair_id,
        "pair_role": pair_role,
        "template_id": template_id,
        "audit_label_mapping": mapping,
        "audit_answer_semantic": simulation["answer"],
        "program_ast": deepcopy(dict(ast)),
        "teacher_trace": simulation["trace"],
        "training_claims": claims,
        "causal_certificate": {
            "kind": "event",
            "changed_path": ["events", int(ast["causal_target"]["event_index"]), "rule", "primitives", int(ast["causal_target"]["primitive_index"]), ast["causal_target"]["field"]],
            "before_answer": simulation["answer"],
            "after_answer": alternate_simulation["answer"],
            "specified_change": {"field": ast["causal_target"]["field"], "from": ast["causal_target"]["original"], "to": ast["causal_target"]["alternate"]},
            "dependency_depth": ast["dependency"]["depth"],
            "pair_id": pair_id,
        },
        "provenance": {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "design_doc_sha256": design_doc_sha256,
            "command": command,
            "environment": dict(environment),
        },
    }
    return record


def generate_ere_split(
    *,
    split: str,
    count: int,
    seed: int,
    design_doc_sha256: str,
    command: str,
    environment: Mapping[str, Any],
    seen_semantic: set[str] | None = None,
    seen_surface: set[str] | None = None,
) -> list[dict[str, Any]]:
    if split not in ERE_SPLITS or split == "causal_pairs":
        raise ValueError(f"generate_ere_split handles non-pair splits only: {split}")
    seen_semantic = seen_semantic if seen_semantic is not None else set()
    seen_surface = seen_surface if seen_surface is not None else set()
    records: list[dict[str, Any]] = []
    for index in range(count):
        if index % 256 == 0:
            print(f"starting ere/{split}: {index + 1}-{min(count, index + 256)}", flush=True)
        for attempt in range(10_000):
            record_seed = seed + index * 104729 + attempt * 7919
            ast = _build_ast(record_seed, split, index)
            template_pool = sorted(OOD_ERE_TEMPLATES) if split == "language_ood" else sorted(TRAIN_ERE_TEMPLATES)
            template_id = template_pool[index % len(template_pool)]
            answer_semantic = simulate_ere(ast)["answer"]
            record = _record_from_ast(
                ast,
                seed=record_seed,
                split=split,
                index=index,
                example_id=f"ere-{split}-{index:05d}",
                template_id=template_id,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
                label_targets=balanced_label_mapping(
                    list(ast["attributes"][0]["values"]),
                    answer_semantic,
                    balanced_label_index(index, seed, len(ast["attributes"][0]["values"])),
                    record_index=index,
                    salt=seed,
                ),
            )
            if record["semantic_fingerprint"] not in seen_semantic and record["surface_fingerprint"] not in seen_surface:
                seen_semantic.add(record["semantic_fingerprint"])
                seen_surface.add(record["surface_fingerprint"])
                records.append(record)
                if (index + 1) % 256 == 0:
                    print(f"generated ere/{split}: {index + 1}", flush=True)
                break
            if attempt and attempt % 1000 == 0:
                print(f"ere/{split} index {index}: uniqueness attempts {attempt}", flush=True)
        else:
            raise RuntimeError(f"unable to generate unique ERE {split} record {index}")
    return records


def generate_ere_causal_pairs(
    *,
    pair_count: int,
    seed: int,
    design_doc_sha256: str,
    command: str,
    environment: Mapping[str, Any],
    seen_semantic: set[str],
    seen_surface: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair_index in range(pair_count):
        pair_id = f"ere-causal-{pair_index:05d}"
        pair_seed = seed + pair_index * 104729
        for attempt in range(10_000):
            ast = _build_ast(pair_seed + attempt * 7919, "causal_pairs", pair_index)
            mutated = mutate_ere_ast(ast)
            base_sim = simulate_ere(ast)
            flip_sim = simulate_ere(mutated)
            if base_sim["answer"] == flip_sim["answer"]:
                continue
            values = list(ast["attributes"][0]["values"])
            mapping = balanced_pair_mapping(values, base_sim["answer"], flip_sim["answer"], pair_index, salt=seed)
            template_id = sorted(TRAIN_ERE_TEMPLATES)[pair_index % len(TRAIN_ERE_TEMPLATES)]
            base = _record_from_ast(
                ast,
                seed=pair_seed,
                split="causal_pairs",
                index=pair_index * 2,
                example_id=f"ere-causal-{pair_index:05d}-base",
                template_id=template_id,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
                pair_id=pair_id,
                pair_role="base",
                label_targets=mapping,
            )
            flip = _record_from_ast(
                mutated,
                seed=pair_seed,
                split="causal_pairs",
                index=pair_index * 2 + 1,
                example_id=f"ere-causal-{pair_index:05d}-flip",
                template_id=template_id,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
                pair_id=pair_id,
                pair_role="flip",
                label_targets=mapping,
            )
            if any(item["semantic_fingerprint"] in seen_semantic or item["surface_fingerprint"] in seen_surface for item in (base, flip)):
                continue
            for item in (base, flip):
                seen_semantic.add(item["semantic_fingerprint"])
                seen_surface.add(item["surface_fingerprint"])
            records.extend((base, flip))
            break
        else:
            raise RuntimeError(f"unable to generate ERE causal pair {pair_id}")
    return records
