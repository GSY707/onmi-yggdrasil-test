from __future__ import annotations

"""Surface rendering, tokenizer accounting, and semantic fingerprints."""

import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .schema import LABELS


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SPECIAL_TOKEN_CONFIG = {"add_special_tokens": False}
TRAIN_ERE_TEMPLATES = frozenset({"ere_train_a", "ere_train_b", "ere_train_c"})
OOD_ERE_TEMPLATES = frozenset({"ere_language_ood_a", "ere_language_ood_b", "ere_language_ood_c"})
TRAIN_CPS_TEMPLATES = frozenset({"cps_train_a", "cps_train_b", "cps_train_c"})
OOD_CPS_TEMPLATES = frozenset({"cps_language_ood_a", "cps_language_ood_b", "cps_language_ood_c"})

_TEMPLATE_FEATURES = {
    "ere_train_a": ["active_baseline"],
    "ere_train_b": ["active_baseline"],
    "ere_train_c": ["active_baseline"],
    "ere_language_ood_a": ["active_passive"],
    "ere_language_ood_b": ["condition_postposed"],
    "ere_language_ood_c": ["clause_order_reversed"],
    "cps_train_a": ["active_baseline"],
    "cps_train_b": ["active_baseline"],
    "cps_train_c": ["active_baseline"],
    "cps_language_ood_a": ["active_passive"],
    "cps_language_ood_b": ["condition_postposed"],
    "cps_language_ood_c": ["clause_order_reversed"],
}

_TOKENIZER = None


def contract_token_count(text: str) -> int:
    """Cheap prefilter only; it is never used as the formal token count."""

    return len(re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text, flags=re.UNICODE))


def get_qwen_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            use_fast=True,
        )
    return _TOKENIZER


def qwen_token_count(text: str) -> int:
    tokenizer = get_qwen_tokenizer()
    encoded = tokenizer(text, **SPECIAL_TOKEN_CONFIG, truncation=False, return_attention_mask=False)
    return len(encoded["input_ids"])


def tokenizer_metadata() -> dict[str, Any]:
    tokenizer = get_qwen_tokenizer()
    import transformers

    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
        "special_token_config": dict(SPECIAL_TOKEN_CONFIG),
    }


def renderer_features(template_id: str) -> list[str]:
    if template_id not in _TEMPLATE_FEATURES:
        raise ValueError(f"unknown template id: {template_id}")
    return list(_TEMPLATE_FEATURES[template_id])


def _surface_order(ast: Mapping[str, Any], key: str, default: Sequence[Any]) -> list[Any]:
    value = ast.get("surface_order", {}).get(key)
    return list(value) if value is not None else list(default)


def _name(value: Any) -> str:
    if isinstance(value, str) and value.startswith("$arg:"):
        return value[5:]
    return str(value)


def _predicate_text(predicate: Mapping[str, Any]) -> str:
    kind = predicate["kind"]
    if kind in {"attr_equals", "attribute_equals"}:
        return f"{_name(predicate['entity'])}.{predicate['attribute']} equals {predicate['value']}"
    if kind == "relation_exists":
        return f"{predicate['source']} points to {predicate['target']} by {predicate['relation']}"
    if kind == "fact_true":
        return f"fact {predicate['fact']} is true"
    if kind == "fact_false":
        return f"fact {predicate['fact']} is false"
    if kind == "resource_at_least":
        return f"resource {predicate['resource']} is at least {predicate['amount']}"
    raise ValueError(f"unknown predicate kind: {kind}")


def _ere_primitive_text(primitive: Mapping[str, Any], *, feature: str = "active_baseline") -> str:
    op = primitive["op"]
    if op == "NOOP":
        return "do nothing"
    if op == "SET":
        return f"set {_name(primitive['target'])}.{primitive['attribute']} to {primitive['value']}"
    if op == "COPY":
        if feature == "active_passive":
            return f"{_name(primitive['target'])}.{primitive['attribute']} receives the value from {_name(primitive['source'])}"
        return f"copy {_name(primitive['source'])}.{primitive['attribute']} to {_name(primitive['target'])}"
    if op == "SWAP":
        return f"swap {_name(primitive['left'])}.{primitive['attribute']} with {_name(primitive['right'])}.{primitive['attribute']}"
    if op in {"LINK", "UNLINK"}:
        verb = "link" if op == "LINK" else "unlink"
        return f"{verb} {_name(primitive['source'])} to {_name(primitive['target'])} by {primitive['relation']}"
    if op == "FOREACH_LINKED":
        effect = _ere_primitive_text(primitive["effect"], feature=feature)
        return f"for each neighbor of {_name(primitive['source'])} under {primitive['relation']}, {effect}"
    if op == "IF":
        condition = _predicate_text(primitive["predicate"])
        then_text = _ere_primitive_text(primitive["then"], feature=feature)
        else_text = _ere_primitive_text(primitive["else"], feature=feature)
        if feature == "condition_postposed":
            return f"{then_text}; otherwise {else_text}; provided that {condition}"
        if feature == "clause_order_reversed":
            return f"condition: {condition}; effect if true: {then_text}; effect otherwise: {else_text}"
        return f"if {condition}, {then_text}; otherwise {else_text}"
    raise ValueError(f"unknown ERE primitive: {op}")


def _ere_rule_text(name: str, rule: Mapping[str, Any], feature: str) -> str:
    params = ", ".join(str(item) for item in rule.get("params", [])) or "none"
    effects = "; then ".join(_ere_primitive_text(item, feature=feature) for item in rule.get("primitives", [])) or "do nothing"
    return f'Rule "{name}" takes {params}: {effects}.'


def _render_ere(ast: Mapping[str, Any], label_mapping: Mapping[str, str], template_id: str) -> tuple[str, dict[str, Any]]:
    features = renderer_features(template_id)
    feature = features[0]
    opening = {
        "ere_train_a": "Solve the temporary episode using the definitions and ordered events.",
        "ere_train_b": "Determine the requested final property in this local world.",
        "ere_train_c": "Execute the named rules in time order and answer the query.",
        "ere_language_ood_a": "The temporary episode is described below; infer its final property.",
        "ere_language_ood_b": "This one-off world has a result determined by its local rules.",
        "ere_language_ood_c": "The following isolated episode specifies a final value to determine.",
    }[template_id]
    attr_items = {item["name"]: item for item in ast["attributes"]}
    attrs = _surface_order(ast, "attributes", [item["name"] for item in ast["attributes"]])
    entities = _surface_order(ast, "entities", ast["entities"])
    rules = _surface_order(ast, "rules", list(ast["rules"]))
    values_by_attr = ast.get("surface_order", {}).get("values_by_attribute", {})
    initial = ast["initial_state"]["attributes"]
    lines = [opening, f"Syntax feature: {feature}."]
    domains = []
    for attr in attrs:
        values = list(values_by_attr.get(attr, attr_items[attr]["values"]))
        domains.append(f"{attr} domain {{{', '.join(values)}}}")
    lines.append("Attributes: " + "; ".join(domains) + ".")
    lines.append("Entities: " + ", ".join(entities) + ".")
    initial_lines = []
    for entity in entities:
        initial_lines.append(entity + ": " + ", ".join(f"{attr}={initial[entity][attr]}" for attr in attrs))
    lines.append("Initial state: " + "; ".join(initial_lines) + ".")
    relation = ast["relations"][0]
    edges = ast["initial_state"].get("relations", {}).get(relation, [])
    edge_text = ", ".join(f"{source}->{target}" for source, target in edges) or "none"
    lines.append(f"Relation: {relation} is directed. Initial relation edges: {edge_text}.")
    for name in rules:
        lines.append(_ere_rule_text(name, ast["rules"][name], feature))
    for index, event in enumerate(ast["events"], start=1):
        args = ", ".join(f"{key}={value}" for key, value in event["arguments"].items())
        lines.append(f"Event {index}: invoke rule \"{event['rule']}\" with {args}.")
    query = ast["query"]
    if query["kind"] == "attribute":
        lines.append(f"Query: read final {query['attribute']} from {query['entity']}.")
    else:
        lines.append(f"Query: determine whether {query['source']} points to {query['target']} by {query['relation']}.")
    legend_values = list(values_by_attr.get(query.get("attribute"), attr_items.get(query.get("attribute"), {}).get("values", [])))
    legend_values = legend_values or list(attr_items[attrs[0]]["values"])
    legend_values = _surface_order(ast, "label_legend", legend_values)
    lines.append("Label legend: " + "; ".join(f"value {value} -> label {label_mapping[value]}" for value in legend_values) + ".")
    lines.append("Available choices: " + ", ".join(label_mapping[value] for value in legend_values) + ". Return one label.")
    text = "\n".join(lines)
    return text, {"features": features, "body": "\n".join(lines[1:]), "opening": lines[0]}


def render_ere(ast: Mapping[str, Any], label_mapping: Mapping[str, str], template_id: str) -> str:
    return _render_ere(ast, label_mapping, template_id)[0]


def _condition_text(condition: Mapping[str, Any]) -> str:
    kind = condition["kind"]
    if kind == "fact_true":
        return f"fact {condition['fact']} is true"
    if kind == "fact_false":
        return f"fact {condition['fact']} is false"
    if kind == "resource_at_least":
        return f"resource {condition['resource']} is at least {condition['amount']}"
    if kind in {"attribute_equals", "attr_equals"}:
        return f"{condition['entity']}.{condition['attribute']} equals {condition['value']}"
    if kind == "relation_exists":
        return f"{condition['source']} points to {condition['target']} by {condition['relation']}"
    raise ValueError(f"unknown CPS condition kind: {kind}")


def _effect_text(effect: Mapping[str, Any], *, feature: str) -> str:
    kind = effect["kind"]
    if kind == "add_fact":
        return f"make fact {effect['fact']} true"
    if kind == "remove_fact":
        return f"make fact {effect['fact']} false"
    if kind == "resource_delta":
        return f"change resource {effect['resource']} by {effect['delta']:+d}"
    if kind == "set_attribute":
        return f"set {effect['entity']}.{effect['attribute']} to {effect['value']}"
    if kind in {"link", "unlink"}:
        verb = "add" if kind == "link" else "remove"
        return f"{verb} relation {effect['relation']} from {effect['source']} to {effect['target']}"
    raise ValueError(f"unknown CPS effect kind: {kind}")


def _action_text(action: Mapping[str, Any], *, feature: str) -> str:
    pre = " and ".join(_condition_text(condition) for condition in action.get("preconditions", [])) or "none"
    effects = "; then ".join(_effect_text(effect, feature=feature) for effect in action.get("effects", [])) or "none"
    if feature == "condition_postposed":
        return f'Action "{action["name"]}": effect {effects}; precondition {pre}; cost {action["cost"]}.'
    if feature == "clause_order_reversed":
        return f'Action "{action["name"]}": effect={effects}; condition={pre}; cost={action["cost"]}.'
    if feature == "active_passive":
        return f'Action "{action["name"]}": {effects} when {pre}; cost {action["cost"]}.'
    return f'Action "{action["name"]}": if {pre}, {effects}; cost {action["cost"]}.'


def _render_cps(ast: Mapping[str, Any], label_mapping: Mapping[str, str], template_id: str) -> tuple[str, dict[str, Any]]:
    features = renderer_features(template_id)
    feature = features[0]
    opening = {
        "cps_train_a": "Choose the unique lowest-cost candidate that satisfies the planning problem.",
        "cps_train_b": "Evaluate each candidate against the local action definitions and constraints.",
        "cps_train_c": "Find the candidate plan that remains legal and reaches the required goal.",
        "cps_language_ood_a": "The isolated planning episode below has one surviving local choice.",
        "cps_language_ood_b": "Use the one-off action meanings to determine which option works.",
        "cps_language_ood_c": "The following temporary world defines a constrained plan selection task.",
    }[template_id]
    initial = ast["initial_state"]
    lines = [opening, f"Syntax feature: {feature}."]
    lines.append("Facts initially true: " + (", ".join(_surface_order(ast, "facts", sorted(initial.get("facts", [])))) or "none") + ".")
    resources = _surface_order(ast, "resources", list(initial.get("resources", {})))
    lines.append("Resources initially: " + (", ".join(f"{name}={initial['resources'][name]}" for name in resources) or "none") + ".")
    relation_names = list(initial.get("relations", {}))
    for relation in relation_names:
        edges = ", ".join(f"{source}->{target}" for source, target in initial["relations"][relation]) or "none"
        lines.append(f"Relation {relation} initially: {edges}.")
    for name in _surface_order(ast, "actions", [action["name"] for action in ast["actions"]]):
        action = next(item for item in ast["actions"] if item["name"] == name)
        lines.append(_action_text(action, feature=feature))
    lines.append("Goal: " + (" and ".join(_condition_text(item) for item in ast["goal"]) or "none") + ".")
    lines.append("Final restriction: " + (" and ".join(_condition_text(item) for item in ast.get("final_constraints", [])) or "none") + ".")
    lines.append(f"Total cost budget: {ast['budget']}.")
    for index in _surface_order(ast, "candidates", list(range(len(ast["candidates"])))):
        candidate = ast["candidates"][int(index)]
        plan = " -> ".join(candidate["plan"]) or "empty"
        lines.append(f"Candidate {int(index) + 1}: label {label_mapping[f'candidate_{int(index)}']}; plan {plan}.")
    lines.append(f"If every candidate fails, use label {label_mapping['NONE']} for NONE.")
    choice_order = _surface_order(ast, "available_choices", [f"candidate_{index}" for index in range(len(ast["candidates"]))] + ["NONE"])
    lines.append("Available choices: " + ", ".join(label_mapping[item] for item in choice_order) + ". Return one label.")
    text = "\n".join(lines)
    return text, {"features": features, "body": "\n".join(lines[1:]), "opening": lines[0]}


def render_cps(ast: Mapping[str, Any], label_mapping: Mapping[str, str], template_id: str) -> str:
    return _render_cps(ast, label_mapping, template_id)[0]


def body_without_opening(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[1:]).strip() if lines else ""


def detected_language_features(text: str) -> set[str]:
    features: set[str] = set()
    lowered = text.lower()
    if "receives the value from" in lowered or "when fact" in lowered or " when " in lowered and "action" in lowered:
        features.add("active_passive")
    if "provided that" in lowered or "; precondition " in lowered:
        features.add("condition_postposed")
    if "condition:" in lowered or "condition=" in lowered:
        features.add("clause_order_reversed")
    return features


def _replace_names(value: Any, maps: Mapping[str, Mapping[str, str]], *, key: str | None = None) -> Any:
    if isinstance(value, list):
        return [_replace_names(item, maps, key=key) for item in value]
    if isinstance(value, dict):
        return {name: _replace_names(child, maps, key=name) for name, child in value.items()}
    if not isinstance(value, str):
        return value
    if value.startswith("$arg:"):
        return "$arg:" + maps.get("param", {}).get(value[5:], value[5:])
    role_map = {
        "entity": "entity", "source": "entity", "target": "entity", "left": "entity", "right": "entity",
        "attribute": "attribute", "value": "value", "relation": "relation", "rule": "rule", "action": "action",
        "fact": "fact", "resource": "resource",
    }
    return maps.get(role_map.get(key or "", ""), {}).get(value, value)


def canonical_ere_payload(ast: Mapping[str, Any]) -> dict[str, Any]:
    entities = {name: f"entity_{index}" for index, name in enumerate(ast.get("entities", []))}
    attrs = {item["name"]: f"attribute_{index}" for index, item in enumerate(ast.get("attributes", []))}
    values: dict[str, str] = {}
    for attr_index, item in enumerate(ast.get("attributes", [])):
        for value_index, value in enumerate(item.get("values", [])):
            values[value] = f"value_{attr_index}_{value_index}"
    relations = {name: f"relation_{index}" for index, name in enumerate(ast.get("relations", []))}
    rules = {name: f"rule_{index}" for index, name in enumerate(ast.get("rules", {}))}
    params: dict[str, str] = {}
    for rule in ast.get("rules", {}).values():
        for index, param in enumerate(rule.get("params", [])):
            params.setdefault(param, f"param_{index}")
    maps = {"entity": entities, "attribute": attrs, "value": values, "relation": relations, "rule": rules, "param": params}
    return {
        "kind": "ere", "recipe": ast.get("recipe"), "entities": list(entities.values()),
        "attributes": _replace_names(ast.get("attributes", []), maps),
        "initial_state": _replace_names(ast.get("initial_state", {}), maps),
        "rules": _replace_names(ast.get("rules", {}), maps),
        "events": _replace_names(ast.get("events", []), maps),
        "query": _replace_names(ast.get("query", {}), maps),
    }


def canonical_cps_payload(ast: Mapping[str, Any]) -> dict[str, Any]:
    entities = sorted(ast.get("initial_state", {}).get("attributes", {}))
    entity_map = {name: f"entity_{index}" for index, name in enumerate(entities)}
    action_names = [action["name"] for action in ast.get("actions", [])]
    action_map = {name: f"action_{index}" for index, name in enumerate(action_names)}
    resources = sorted(ast.get("initial_state", {}).get("resources", {}))
    resource_map = {name: f"resource_{index}" for index, name in enumerate(resources)}
    all_facts = set(ast.get("initial_state", {}).get("facts", []))
    for condition in ast.get("goal", []) + ast.get("final_constraints", []):
        if "fact" in condition:
            all_facts.add(condition["fact"])
    for action in ast.get("actions", []):
        for condition in action.get("preconditions", []):
            if "fact" in condition:
                all_facts.add(condition["fact"])
        for effect in action.get("effects", []):
            if "fact" in effect:
                all_facts.add(effect["fact"])
    fact_map = {name: f"fact_{index}" for index, name in enumerate(sorted(all_facts))}
    relations = sorted({condition.get("relation") for action in ast.get("actions", []) for condition in action.get("preconditions", []) if condition.get("relation")} | {effect.get("relation") for action in ast.get("actions", []) for effect in action.get("effects", []) if effect.get("relation")})
    relation_map = {name: f"relation_{index}" for index, name in enumerate(relations)}
    maps = {"entity": entity_map, "action": action_map, "rule": action_map, "resource": resource_map, "fact": fact_map, "relation": relation_map}
    candidates = [_replace_names(candidate, maps) for candidate in ast.get("candidates", [])]
    candidates.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return {
        "kind": "cps", "recipe": ast.get("recipe"), "initial_state": _replace_names(ast.get("initial_state", {}), maps),
        "actions": _replace_names(ast.get("actions", []), maps), "goal": _replace_names(ast.get("goal", []), maps),
        "final_constraints": _replace_names(ast.get("final_constraints", []), maps), "budget": int(ast.get("budget", 0)),
        "candidates": candidates,
    }


def semantic_fingerprint(ast: Mapping[str, Any]) -> str:
    payload = canonical_ere_payload(ast) if ast.get("kind") == "ere" else canonical_cps_payload(ast)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def surface_fingerprint(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()
