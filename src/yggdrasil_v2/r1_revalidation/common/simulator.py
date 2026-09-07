from __future__ import annotations

"""Strict schema validators and deterministic ERE/CPS interpreters."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


_STATE_FIELDS = ("attributes", "relations", "facts", "resources")
_ERE_ROOT_FIELDS = ("initial_state", "rules", "events", "query")
_CPS_ROOT_FIELDS = ("initial_state", "actions", "candidates", "budget", "goal", "final_constraints")
_ERE_PREDICATES = {"attribute_equals", "relation_exists"}
_ERE_PRIMITIVES = {"SET", "COPY", "SWAP", "LINK", "UNLINK", "IF", "FOREACH_LINKED"}
_CPS_CONDITIONS = {"fact_true", "fact_false", "resource_at_least", "attribute_equals", "relation_exists"}
_CPS_EFFECTS = {"add_fact", "remove_fact", "resource_delta", "set_attribute", "link", "unlink"}


def _path(base: str, part: Any) -> str:
    encoded = str(part).replace("~", "~0").replace("/", "~1")
    return f"{base}/{encoded}" if base else f"/{encoded}"


def _invalid(code: str, path: str) -> None:
    if not path:
        path = "/"
    raise ValueError(f"AST_VALIDATION|{code}|{path}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _invalid("wrong_type", path)
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _invalid("wrong_type", path)
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        _invalid("wrong_type", path)
    if not value.strip():
        _invalid("invalid_value", path)
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid("wrong_type", path)
    return int(value)


def _nonnegative_integer(value: Any, path: str) -> int:
    number = _integer(value, path)
    if number < 0:
        _invalid("invalid_value", path)
    return number


def _exact(value: Any, fields: tuple[str, ...], path: str) -> Mapping[str, Any]:
    source = _object(value, path)
    for key in source:
        if not isinstance(key, str):
            _invalid("wrong_type", _path(path, key))
        if key not in fields:
            _invalid("unexpected_field", _path(path, key))
    for field in fields:
        if field not in source:
            _invalid("missing_field", _path(path, field))
    return source


def _map_key(value: Any, path: str) -> str:
    return _string(value, path)


def _operand(value: Any, path: str, params: set[str], neighbor_allowed: bool) -> str:
    text = _string(value, path)
    if not text.startswith("$"):
        return text
    if text == "$neighbor":
        if not neighbor_allowed:
            _invalid("invalid_placeholder", path)
        return text
    if text.startswith("$arg:"):
        name = text[5:]
        if not name or name not in params:
            _invalid("invalid_placeholder", path)
        return text
    _invalid("invalid_placeholder", path)
    return text


def _validate_state(value: Any, path: str) -> None:
    source = _exact(value, _STATE_FIELDS, path)

    attributes_path = _path(path, "attributes")
    attributes = _object(source["attributes"], attributes_path)
    for entity, values in attributes.items():
        entity_path = _path(attributes_path, entity)
        _map_key(entity, entity_path)
        values_map = _object(values, entity_path)
        for attribute, attribute_value in values_map.items():
            attribute_path = _path(entity_path, attribute)
            _map_key(attribute, attribute_path)
            _string(attribute_value, attribute_path)

    relations_path = _path(path, "relations")
    relations = _object(source["relations"], relations_path)
    for relation, pairs in relations.items():
        relation_path = _path(relations_path, relation)
        _map_key(relation, relation_path)
        pair_values = _list(pairs, relation_path)
        seen: set[tuple[str, str]] = set()
        for index, pair in enumerate(pair_values):
            pair_path = _path(relation_path, index)
            if not isinstance(pair, list) or len(pair) != 2:
                _invalid("invalid_shape", pair_path)
            left = _string(pair[0], _path(pair_path, 0))
            right = _string(pair[1], _path(pair_path, 1))
            item = (left, right)
            if item in seen:
                _invalid("duplicate_value", pair_path)
            seen.add(item)

    facts_path = _path(path, "facts")
    facts = _list(source["facts"], facts_path)
    seen_facts: set[str] = set()
    for index, fact in enumerate(facts):
        fact_path = _path(facts_path, index)
        fact_name = _string(fact, fact_path)
        if fact_name in seen_facts:
            _invalid("duplicate_value", fact_path)
        seen_facts.add(fact_name)

    resources_path = _path(path, "resources")
    resources = _object(source["resources"], resources_path)
    for resource, amount in resources.items():
        resource_path = _path(resources_path, resource)
        _map_key(resource, resource_path)
        _nonnegative_integer(amount, resource_path)


def _validate_ere_predicate(value: Any, path: str, params: set[str], neighbor_allowed: bool) -> None:
    source = _object(value, path)
    kind_path = _path(path, "kind")
    if "kind" not in source:
        _invalid("missing_field", kind_path)
    kind = _string(source["kind"], kind_path)
    if kind not in _ERE_PREDICATES:
        _invalid("unknown_kind", kind_path)
    if kind == "attribute_equals":
        exact = _exact(source, ("kind", "entity", "attribute", "value"), path)
        _operand(exact["entity"], _path(path, "entity"), params, neighbor_allowed)
        _operand(exact["attribute"], _path(path, "attribute"), params, neighbor_allowed)
        _operand(exact["value"], _path(path, "value"), params, neighbor_allowed)
    else:
        exact = _exact(source, ("kind", "relation", "source", "target"), path)
        _operand(exact["relation"], _path(path, "relation"), params, neighbor_allowed)
        _operand(exact["source"], _path(path, "source"), params, neighbor_allowed)
        _operand(exact["target"], _path(path, "target"), params, neighbor_allowed)


def _validate_ere_primitive(value: Any, path: str, params: set[str], neighbor_allowed: bool) -> None:
    source = _object(value, path)
    op_path = _path(path, "op")
    if "op" not in source:
        _invalid("missing_field", op_path)
    op = _string(source["op"], op_path)
    if op not in _ERE_PRIMITIVES:
        _invalid("unknown_kind", op_path)

    if op == "SET":
        exact = _exact(source, ("op", "target", "attribute", "value"), path)
        _operand(exact["target"], _path(path, "target"), params, neighbor_allowed)
        _operand(exact["attribute"], _path(path, "attribute"), params, neighbor_allowed)
        _operand(exact["value"], _path(path, "value"), params, neighbor_allowed)
    elif op == "COPY":
        exact = _exact(source, ("op", "source", "target", "attribute"), path)
        _operand(exact["source"], _path(path, "source"), params, neighbor_allowed)
        _operand(exact["target"], _path(path, "target"), params, neighbor_allowed)
        _operand(exact["attribute"], _path(path, "attribute"), params, neighbor_allowed)
    elif op == "SWAP":
        exact = _exact(source, ("op", "left", "right", "attribute"), path)
        _operand(exact["left"], _path(path, "left"), params, neighbor_allowed)
        _operand(exact["right"], _path(path, "right"), params, neighbor_allowed)
        _operand(exact["attribute"], _path(path, "attribute"), params, neighbor_allowed)
    elif op in {"LINK", "UNLINK"}:
        exact = _exact(source, ("op", "relation", "source", "target"), path)
        _operand(exact["relation"], _path(path, "relation"), params, neighbor_allowed)
        _operand(exact["source"], _path(path, "source"), params, neighbor_allowed)
        _operand(exact["target"], _path(path, "target"), params, neighbor_allowed)
    elif op == "IF":
        exact = _exact(source, ("op", "predicate", "then", "else"), path)
        _validate_ere_predicate(exact["predicate"], _path(path, "predicate"), params, neighbor_allowed)
        _validate_ere_primitive(exact["then"], _path(path, "then"), params, neighbor_allowed)
        _validate_ere_primitive(exact["else"], _path(path, "else"), params, neighbor_allowed)
    else:
        exact = _exact(source, ("op", "relation", "source", "effect"), path)
        _operand(exact["relation"], _path(path, "relation"), params, neighbor_allowed)
        _operand(exact["source"], _path(path, "source"), params, neighbor_allowed)
        _validate_ere_primitive(exact["effect"], _path(path, "effect"), params, True)


def _validate_ere_query(value: Any, path: str) -> None:
    source = _object(value, path)
    kind_path = _path(path, "kind")
    if "kind" not in source:
        _invalid("missing_field", kind_path)
    kind = _string(source["kind"], kind_path)
    if kind == "attribute":
        exact = _exact(source, ("kind", "entity", "attribute"), path)
        _operand(exact["entity"], _path(path, "entity"), params=set(), neighbor_allowed=False)
        _operand(exact["attribute"], _path(path, "attribute"), params=set(), neighbor_allowed=False)
    elif kind == "relation":
        exact = _exact(source, ("kind", "relation", "source", "target"), path)
        _operand(exact["relation"], _path(path, "relation"), params=set(), neighbor_allowed=False)
        _operand(exact["source"], _path(path, "source"), params=set(), neighbor_allowed=False)
        _operand(exact["target"], _path(path, "target"), params=set(), neighbor_allowed=False)
    else:
        _invalid("unknown_kind", kind_path)


def _validate_ere(value: Any) -> Mapping[str, Any]:
    source = _exact(value, _ERE_ROOT_FIELDS, "")
    _validate_state(source["initial_state"], "/initial_state")

    rules_path = "/rules"
    rules = _object(source["rules"], rules_path)
    rule_params: dict[str, set[str]] = {}
    for name, rule in rules.items():
        rule_path = _path(rules_path, name)
        rule_name = _map_key(name, rule_path)
        rule_source = _exact(rule, ("params", "primitives"), rule_path)
        params_path = _path(rule_path, "params")
        params = _list(rule_source["params"], params_path)
        params_set: set[str] = set()
        for index, param in enumerate(params):
            param_path = _path(params_path, index)
            param_name = _string(param, param_path)
            if param_name in params_set:
                _invalid("duplicate_value", param_path)
            params_set.add(param_name)
        primitive_path = _path(rule_path, "primitives")
        primitives = _list(rule_source["primitives"], primitive_path)
        for index, primitive in enumerate(primitives):
            _validate_ere_primitive(primitive, _path(primitive_path, index), params_set, False)
        rule_params[rule_name] = params_set

    events_path = "/events"
    events = _list(source["events"], events_path)
    for index, event in enumerate(events):
        event_path = _path(events_path, index)
        event_source = _exact(event, ("rule", "arguments"), event_path)
        rule_path = _path(event_path, "rule")
        rule_name = _string(event_source["rule"], rule_path)
        arguments_path = _path(event_path, "arguments")
        arguments = _object(event_source["arguments"], arguments_path)
        if rule_name not in rules:
            _invalid("unknown_reference", rule_path)
        params_set = rule_params[rule_name]
        for argument, argument_value in arguments.items():
            argument_path = _path(arguments_path, argument)
            _map_key(argument, argument_path)
            _string(argument_value, argument_path)
        if set(arguments) != params_set:
            _invalid("argument_mismatch", arguments_path)

    _validate_ere_query(source["query"], "/query")
    return source


def _validate_cps_condition(value: Any, path: str) -> None:
    source = _object(value, path)
    kind_path = _path(path, "kind")
    if "kind" not in source:
        _invalid("missing_field", kind_path)
    kind = _string(source["kind"], kind_path)
    if kind not in _CPS_CONDITIONS:
        _invalid("unknown_kind", kind_path)
    if kind in {"fact_true", "fact_false"}:
        exact = _exact(source, ("kind", "fact"), path)
        _string(exact["fact"], _path(path, "fact"))
    elif kind == "resource_at_least":
        exact = _exact(source, ("kind", "resource", "amount"), path)
        _string(exact["resource"], _path(path, "resource"))
        _nonnegative_integer(exact["amount"], _path(path, "amount"))
    elif kind == "attribute_equals":
        exact = _exact(source, ("kind", "entity", "attribute", "value"), path)
        _string(exact["entity"], _path(path, "entity"))
        _string(exact["attribute"], _path(path, "attribute"))
        _string(exact["value"], _path(path, "value"))
    else:
        exact = _exact(source, ("kind", "relation", "source", "target"), path)
        _string(exact["relation"], _path(path, "relation"))
        _string(exact["source"], _path(path, "source"))
        _string(exact["target"], _path(path, "target"))


def _validate_cps_effect(value: Any, path: str) -> None:
    source = _object(value, path)
    kind_path = _path(path, "kind")
    if "kind" not in source:
        _invalid("missing_field", kind_path)
    kind = _string(source["kind"], kind_path)
    if kind not in _CPS_EFFECTS:
        _invalid("unknown_kind", kind_path)
    if kind in {"add_fact", "remove_fact"}:
        exact = _exact(source, ("kind", "fact"), path)
        _string(exact["fact"], _path(path, "fact"))
    elif kind == "resource_delta":
        exact = _exact(source, ("kind", "resource", "delta"), path)
        _string(exact["resource"], _path(path, "resource"))
        _integer(exact["delta"], _path(path, "delta"))
    elif kind == "set_attribute":
        exact = _exact(source, ("kind", "entity", "attribute", "value"), path)
        _string(exact["entity"], _path(path, "entity"))
        _string(exact["attribute"], _path(path, "attribute"))
        _string(exact["value"], _path(path, "value"))
    else:
        exact = _exact(source, ("kind", "relation", "source", "target"), path)
        _string(exact["relation"], _path(path, "relation"))
        _string(exact["source"], _path(path, "source"))
        _string(exact["target"], _path(path, "target"))


def _validate_cps(value: Any) -> Mapping[str, Any]:
    source = _exact(value, _CPS_ROOT_FIELDS, "")
    _validate_state(source["initial_state"], "/initial_state")

    actions_path = "/actions"
    actions = _list(source["actions"], actions_path)
    action_names: set[str] = set()
    for index, action in enumerate(actions):
        action_path = _path(actions_path, index)
        action_source = _exact(action, ("name", "cost", "preconditions", "effects"), action_path)
        name_path = _path(action_path, "name")
        name = _string(action_source["name"], name_path)
        if name in action_names:
            _invalid("duplicate_value", name_path)
        action_names.add(name)
        _nonnegative_integer(action_source["cost"], _path(action_path, "cost"))
        preconditions_path = _path(action_path, "preconditions")
        preconditions = _list(action_source["preconditions"], preconditions_path)
        for condition_index, condition in enumerate(preconditions):
            _validate_cps_condition(condition, _path(preconditions_path, condition_index))
        effects_path = _path(action_path, "effects")
        effects = _list(action_source["effects"], effects_path)
        for effect_index, effect in enumerate(effects):
            _validate_cps_effect(effect, _path(effects_path, effect_index))

    candidates_path = "/candidates"
    candidates = _list(source["candidates"], candidates_path)
    for index, candidate in enumerate(candidates):
        candidate_path = _path(candidates_path, index)
        candidate_source = _exact(candidate, ("plan",), candidate_path)
        plan_path = _path(candidate_path, "plan")
        plan = _list(candidate_source["plan"], plan_path)
        for action_index, action_name in enumerate(plan):
            _string(action_name, _path(plan_path, action_index))

    _nonnegative_integer(source["budget"], "/budget")
    goal_path = "/goal"
    goal = _list(source["goal"], goal_path)
    for index, condition in enumerate(goal):
        _validate_cps_condition(condition, _path(goal_path, index))
    constraints_path = "/final_constraints"
    constraints = _list(source["final_constraints"], constraints_path)
    for index, condition in enumerate(constraints):
        _validate_cps_condition(condition, _path(constraints_path, index))
    return source


def _copy_state(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attributes": {
            entity: {attribute: deepcopy(attribute_value) for attribute, attribute_value in values.items()}
            for entity, values in value["attributes"].items()
        },
        "relations": {
            relation: {tuple(pair) for pair in pairs}
            for relation, pairs in value["relations"].items()
        },
        "facts": set(value["facts"]),
        "resources": {resource: int(amount) for resource, amount in value["resources"].items()},
    }


def _state_json(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attributes": {
            entity: {attribute: deepcopy(values[attribute]) for attribute in sorted(values)}
            for entity, values in sorted(state["attributes"].items())
        },
        "relations": {
            relation: [list(pair) for pair in sorted(pairs)]
            for relation, pairs in sorted(state["relations"].items())
        },
        "facts": sorted(state["facts"]),
        "resources": {resource: int(amount) for resource, amount in sorted(state["resources"].items())},
    }


def _resolve(value: str, bindings: Mapping[str, Any], neighbor: str | None = None) -> str:
    if value == "$neighbor":
        return str(neighbor)
    if value.startswith("$arg:"):
        return str(bindings[value[5:]])
    return value


def _relation_exists(state: Mapping[str, Any], relation: str, source: str, target: str) -> bool:
    return (source, target) in state["relations"].get(relation, set())


def _ere_predicate_holds(state: Mapping[str, Any], predicate: Mapping[str, Any], bindings: Mapping[str, Any], neighbor: str | None) -> bool:
    kind = predicate["kind"]
    if kind == "attribute_equals":
        entity = _resolve(predicate["entity"], bindings, neighbor)
        attribute = _resolve(predicate["attribute"], bindings, neighbor)
        value = _resolve(predicate["value"], bindings, neighbor)
        return state["attributes"].get(entity, {}).get(attribute) == value
    relation = _resolve(predicate["relation"], bindings, neighbor)
    source = _resolve(predicate["source"], bindings, neighbor)
    target = _resolve(predicate["target"], bindings, neighbor)
    return _relation_exists(state, relation, source, target)


def _apply_ere_primitive(
    state: Mapping[str, Any], primitive: Mapping[str, Any], bindings: Mapping[str, Any], neighbor: str | None
) -> tuple[list[str], int]:
    op = primitive["op"]
    if op == "IF":
        branch_true = _ere_predicate_holds(state, primitive["predicate"], bindings, neighbor)
        branch = primitive["then"] if branch_true else primitive["else"]
        operations, delta = _apply_ere_primitive(state, branch, bindings, neighbor)
        return [f"IF:{'true' if branch_true else 'false'}", *operations], delta
    if op == "FOREACH_LINKED":
        relation = _resolve(primitive["relation"], bindings, neighbor)
        source = _resolve(primitive["source"], bindings, neighbor)
        neighbors = sorted(target for left, target in state["relations"].get(relation, set()) if left == source)
        operations = [f"FOREACH_LINKED:{len(neighbors)}"]
        delta = 0
        for target in neighbors:
            child_operations, child_delta = _apply_ere_primitive(state, primitive["effect"], bindings, target)
            operations.extend(child_operations)
            delta += child_delta
        return operations, delta

    resolved = {key: _resolve(value, bindings, neighbor) for key, value in primitive.items() if key != "op"}
    if op == "SET":
        target = resolved["target"]
        state["attributes"].setdefault(target, {})[resolved["attribute"]] = deepcopy(resolved["value"])
        return ["SET"], 1
    if op == "COPY":
        source = resolved["source"]
        target = resolved["target"]
        attribute = resolved["attribute"]
        if attribute not in state["attributes"].get(source, {}):
            raise ValueError(f"COPY source attribute is absent: {source}.{attribute}")
        state["attributes"].setdefault(target, {})[attribute] = deepcopy(state["attributes"][source][attribute])
        return ["COPY"], 1
    if op == "SWAP":
        left = resolved["left"]
        right = resolved["right"]
        attribute = resolved["attribute"]
        if attribute not in state["attributes"].get(left, {}) or attribute not in state["attributes"].get(right, {}):
            raise ValueError("SWAP attribute is absent")
        state["attributes"][left][attribute], state["attributes"][right][attribute] = (
            state["attributes"][right][attribute],
            state["attributes"][left][attribute],
        )
        return ["SWAP"], 1
    if op == "LINK":
        state["relations"].setdefault(resolved["relation"], set()).add((resolved["source"], resolved["target"]))
        return ["LINK"], 1
    state["relations"].setdefault(resolved["relation"], set()).discard((resolved["source"], resolved["target"]))
    return ["UNLINK"], 1


def _validate_ere_prefix(events: list[Any], prefix: int | None) -> int:
    if prefix is None:
        return len(events)
    number = _integer(prefix, "/prefix")
    if number < 0 or number > len(events):
        _invalid("invalid_prefix", "/prefix")
    return number


def simulate_ere(ast: Mapping[str, Any], prefix: int | None = None) -> dict[str, Any]:
    """Validate and execute an episodic rule execution AST."""

    source = _validate_ere(ast)
    event_count = _validate_ere_prefix(source["events"], prefix)
    state = _copy_state(source["initial_state"])
    rules = source["rules"]
    trace: list[dict[str, Any]] = []
    for event_index, event in enumerate(source["events"][:event_count]):
        rule_name = event["rule"]
        arguments = dict(event["arguments"])
        rule = rules[rule_name]
        before = _state_json(state)
        operations: list[str] = []
        for primitive in rule["primitives"]:
            child_operations, _ = _apply_ere_primitive(state, primitive, arguments, None)
            operations.extend(child_operations)
        after = _state_json(state)
        delta_budget = sum(
            1
            for item in operations
            if item not in {"IF:true", "IF:false"} and not item.startswith("FOREACH_LINKED:")
        )
        trace.append(
            {
                "event_index": event_index,
                "rule": rule_name,
                "arguments": deepcopy(arguments),
                "executed_ops": operations,
                "delta_budget": delta_budget,
                "state_before": before,
                "state_after": after,
            }
        )

    query = source["query"]
    if query["kind"] == "attribute":
        answer: Any = state["attributes"].get(query["entity"], {}).get(query["attribute"])
    else:
        answer = _relation_exists(state, query["relation"], query["source"], query["target"])
    return {"answer": answer, "final_state": _state_json(state), "trace": trace}


def _condition_holds(state: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    kind = condition["kind"]
    if kind == "fact_true":
        return condition["fact"] in state["facts"]
    if kind == "fact_false":
        return condition["fact"] not in state["facts"]
    if kind == "resource_at_least":
        return state["resources"].get(condition["resource"], 0) >= condition["amount"]
    if kind == "attribute_equals":
        return state["attributes"].get(condition["entity"], {}).get(condition["attribute"]) == condition["value"]
    return _relation_exists(state, condition["relation"], condition["source"], condition["target"])


def _apply_cps_effect(state: Mapping[str, Any], effect: Mapping[str, Any]) -> None:
    kind = effect["kind"]
    if kind == "add_fact":
        state["facts"].add(effect["fact"])
    elif kind == "remove_fact":
        state["facts"].discard(effect["fact"])
    elif kind == "resource_delta":
        resource = effect["resource"]
        state["resources"][resource] = state["resources"].get(resource, 0) + effect["delta"]
    elif kind == "set_attribute":
        state["attributes"].setdefault(effect["entity"], {})[effect["attribute"]] = deepcopy(effect["value"])
    elif kind == "link":
        state["relations"].setdefault(effect["relation"], set()).add((effect["source"], effect["target"]))
    else:
        state["relations"].setdefault(effect["relation"], set()).discard((effect["source"], effect["target"]))


def _run_cps_plan(
    initial: Mapping[str, Any], actions: Mapping[str, Mapping[str, Any]], plan: list[str], limit: int | None = None
) -> tuple[dict[str, Any], int, bool, list[dict[str, Any]]]:
    state = deepcopy(initial)
    trace: list[dict[str, Any]] = []
    total_cost = 0
    legal = True
    steps = plan if limit is None else plan[:limit]
    for position, action_name in enumerate(steps):
        action = actions.get(action_name)
        if action is None:
            legal = False
            trace.append(
                {
                    "position": position,
                    "action": action_name,
                    "legal": False,
                    "failure": "unknown_action",
                    "cost": 0,
                    "state_after": _state_json(state),
                }
            )
            break
        if not all(_condition_holds(state, condition) for condition in action["preconditions"]):
            legal = False
            trace.append(
                {
                    "position": position,
                    "action": action_name,
                    "legal": False,
                    "failure": "precondition",
                    "cost": 0,
                    "state_after": _state_json(state),
                }
            )
            break
        for effect in action["effects"]:
            _apply_cps_effect(state, effect)
        action_cost = action["cost"]
        total_cost += action_cost
        trace.append(
            {
                "position": position,
                "action": action_name,
                "legal": True,
                "failure": None,
                "cost": action_cost,
                "state_after": _state_json(state),
            }
        )
    return state, total_cost, legal, trace


def _candidate_report(
    initial: Mapping[str, Any],
    actions: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any],
    candidate_index: int,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    plan = list(candidate["plan"])
    state, total_cost, legal, trace = _run_cps_plan(initial, actions, plan)
    budget_ok = total_cost <= source["budget"]
    goal_satisfied = all(_condition_holds(state, condition) for condition in source["goal"])
    final_constraints_satisfied = all(_condition_holds(state, condition) for condition in source["final_constraints"])
    if not legal:
        failure_reasons = [str(trace[-1]["failure"])] if trace else []
    else:
        failure_reasons = []
        if not budget_ok:
            failure_reasons.append("budget")
        if not goal_satisfied:
            failure_reasons.append("goal")
        if not final_constraints_satisfied:
            failure_reasons.append("final_constraint")
    return {
        "candidate_index": candidate_index,
        "plan": plan,
        "legal": legal,
        "valid": legal and budget_ok and goal_satisfied and final_constraints_satisfied,
        "total_cost": total_cost,
        "budget_ok": budget_ok,
        "goal_satisfied": goal_satisfied,
        "final_constraints_satisfied": final_constraints_satisfied,
        "failure_reasons": failure_reasons,
        "trace": trace,
        "final_state": _state_json(state),
    }


def evaluate_cps(ast: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every CPS definition, then select a unique minimum-cost plan."""

    source = _validate_cps(ast)
    initial = _copy_state(source["initial_state"])
    actions = {action["name"]: action for action in source["actions"]}
    reports = [
        _candidate_report(initial, actions, source, index, candidate)
        for index, candidate in enumerate(source["candidates"])
    ]
    valid = [row for row in reports if row["valid"]]
    if not valid:
        answer = None
        unique_optimum = False
    else:
        minimum = min(row["total_cost"] for row in valid)
        winners = [row for row in valid if row["total_cost"] == minimum]
        unique_optimum = len(winners) == 1
        answer = winners[0]["candidate_index"] if unique_optimum else None
    return {"answer": answer, "unique_optimum": unique_optimum, "candidates": reports}


def _validate_call_integer(value: Any, path: str) -> int:
    return _integer(value, path)


def replay_cps_prefix(ast: Mapping[str, Any], candidate_index: int, prefix: int) -> dict[str, Any]:
    """Validate the complete CPS AST before replaying one candidate prefix."""

    source = _validate_cps(ast)
    index = _validate_call_integer(candidate_index, "/candidate_index")
    if index < 0 or index >= len(source["candidates"]):
        _invalid("invalid_index", "/candidate_index")
    prefix_value = _validate_call_integer(prefix, "/prefix")
    plan = list(source["candidates"][index]["plan"])
    if prefix_value < 0 or prefix_value > len(plan):
        _invalid("invalid_prefix", "/prefix")
    initial = _copy_state(source["initial_state"])
    actions = {action["name"]: action for action in source["actions"]}
    state, cost, legal, trace = _run_cps_plan(initial, actions, plan, prefix_value)
    return {
        "candidate_index": index,
        "prefix": prefix_value,
        "legal_so_far": legal,
        "cost": cost,
        "state": _state_json(state),
        "trace": trace,
    }
