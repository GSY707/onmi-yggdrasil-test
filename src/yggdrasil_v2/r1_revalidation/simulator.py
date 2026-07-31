from __future__ import annotations

"""Independent ERE/CPS simulators and fresh prefix replay helpers."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class TypedState:
    attributes: dict[str, dict[str, str]] = field(default_factory=dict)
    relations: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    facts: set[str] = field(default_factory=set)
    resources: dict[str, int] = field(default_factory=dict)

    def clone(self) -> "TypedState":
        return deepcopy(self)

    def as_json(self) -> dict[str, Any]:
        return {
            "attributes": deepcopy(self.attributes),
            "relations": {name: [list(pair) for pair in sorted(pairs)] for name, pairs in sorted(self.relations.items())},
            "facts": sorted(self.facts),
            "resources": dict(sorted(self.resources.items())),
        }


def state_from_json(payload: Mapping[str, Any]) -> TypedState:
    return TypedState(
        attributes={str(entity): {str(attr): str(value) for attr, value in attrs.items()} for entity, attrs in payload.get("attributes", {}).items()},
        relations={str(name): {tuple(pair) for pair in pairs} for name, pairs in payload.get("relations", {}).items()},
        facts={str(item) for item in payload.get("facts", [])},
        resources={str(name): int(value) for name, value in payload.get("resources", {}).items()},
    )


def _resolve(value: Any, bindings: Mapping[str, str], *, neighbor: str | None = None) -> Any:
    if isinstance(value, str):
        if value == "$neighbor":
            if neighbor is None:
                raise ValueError("$neighbor used outside FOREACH_LINKED")
            return neighbor
        if value.startswith("$arg:"):
            name = value[5:]
            if name not in bindings:
                raise ValueError(f"missing rule argument: {name}")
            return bindings[name]
        return value
    if isinstance(value, list):
        return [_resolve(item, bindings, neighbor=neighbor) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(child, bindings, neighbor=neighbor) for key, child in value.items()}
    return value


def predicate_holds(state: TypedState, predicate: Mapping[str, Any], bindings: Mapping[str, str] | None = None) -> bool:
    predicate = _resolve(predicate, bindings or {})
    kind = predicate.get("kind")
    if kind in {"attr_equals", "attribute_equals"}:
        return state.attributes.get(predicate["entity"], {}).get(predicate["attribute"]) == predicate["value"]
    if kind == "relation_exists":
        return (predicate["source"], predicate["target"]) in state.relations.get(predicate["relation"], set())
    if kind == "fact_true":
        return predicate["fact"] in state.facts
    if kind == "fact_false":
        return predicate["fact"] not in state.facts
    if kind == "resource_at_least":
        return state.resources.get(predicate["resource"], 0) >= int(predicate["amount"])
    raise ValueError(f"unknown ERE predicate kind: {kind}")


def _apply_ere_effect(state: TypedState, effect: Mapping[str, Any], bindings: Mapping[str, str], *, neighbor: str | None = None) -> dict[str, Any]:
    resolved = _resolve(effect, bindings, neighbor=neighbor)
    op = resolved.get("op")
    before = state.as_json()
    if op == "NOOP":
        pass
    elif op == "SET":
        state.attributes.setdefault(resolved["target"], {})[resolved["attribute"]] = resolved["value"]
    elif op == "COPY":
        value = state.attributes.get(resolved["source"], {}).get(resolved["attribute"])
        if value is None:
            raise ValueError("COPY source attribute is absent")
        state.attributes.setdefault(resolved["target"], {})[resolved["attribute"]] = value
    elif op == "SWAP":
        left = state.attributes.setdefault(resolved["left"], {})
        right = state.attributes.setdefault(resolved["right"], {})
        left_old = left.get(resolved["attribute"])
        right_old = right.get(resolved["attribute"])
        left[resolved["attribute"]], right[resolved["attribute"]] = right_old, left_old
    elif op == "IF":
        branch = resolved["then"] if predicate_holds(state, resolved["predicate"]) else resolved["else"]
        _apply_ere_effect(state, branch, {}, neighbor=neighbor)
    elif op == "LINK":
        state.relations.setdefault(resolved["relation"], set()).add((resolved["source"], resolved["target"]))
    elif op == "UNLINK":
        state.relations.setdefault(resolved["relation"], set()).discard((resolved["source"], resolved["target"]))
    elif op == "FOREACH_LINKED":
        neighbors = sorted(target for source, target in state.relations.get(resolved["relation"], set()) if source == resolved["source"])
        for item in neighbors:
            _apply_ere_effect(state, resolved["effect"], {}, neighbor=item)
    else:
        raise ValueError(f"unknown ERE effect op: {op}")
    return {"op": op, "before": before, "after": state.as_json()}


def execute_ere_rule(state: TypedState, rule: Mapping[str, Any], arguments: Mapping[str, str]) -> list[dict[str, Any]]:
    if set(arguments) != set(rule.get("params", [])):
        raise ValueError("rule arguments do not match rule parameters")
    return [_apply_ere_effect(state, primitive, arguments) for primitive in rule.get("primitives", [])]


def simulate_ere(ast: Mapping[str, Any]) -> dict[str, Any]:
    state = state_from_json(ast["initial_state"])
    trace: list[dict[str, Any]] = []
    for index, event in enumerate(ast["events"]):
        before = state.as_json()
        deltas = execute_ere_rule(state, ast["rules"][event["rule"]], event["arguments"])
        trace.append({
            "event_index": index,
            "rule": event["rule"],
            "arguments": deepcopy(event["arguments"]),
            "before": before,
            "deltas": deltas,
            "after": state.as_json(),
        })
    query = ast["query"]
    if query["kind"] == "attribute":
        answer = state.attributes[query["entity"]][query["attribute"]]
    elif query["kind"] == "relation":
        answer = (query["source"], query["target"]) in state.relations.get(query["relation"], set())
    else:
        raise ValueError(f"unknown ERE query kind: {query['kind']}")
    return {"final_state": state, "answer": answer, "trace": trace}


def ablate_ere_event(ast: Mapping[str, Any], event_index: int) -> dict[str, Any]:
    """Freshly replay an event as an explicit no-op while preserving its call shape."""

    mutated = deepcopy(dict(ast))
    event = mutated["events"][int(event_index)]
    rule_name = event["rule"]
    original = mutated["rules"][rule_name]
    noop_name = f"__ablate_{rule_name}_{event_index}"
    mutated["rules"][noop_name] = {"params": list(original.get("params", [])), "primitives": [{"op": "NOOP"}]}
    event["rule"] = noop_name
    return simulate_ere(mutated)


def _cps_precondition_holds(state: TypedState, condition: Mapping[str, Any]) -> bool:
    kind = condition["kind"]
    if kind == "fact_true":
        return condition["fact"] in state.facts
    if kind == "fact_false":
        return condition["fact"] not in state.facts
    if kind == "resource_at_least":
        return state.resources.get(condition["resource"], 0) >= int(condition["amount"])
    if kind in {"attribute_equals", "attr_equals"}:
        return state.attributes.get(condition["entity"], {}).get(condition["attribute"]) == condition["value"]
    if kind == "relation_exists":
        return (condition["source"], condition["target"]) in state.relations.get(condition["relation"], set())
    raise ValueError(f"unknown CPS precondition kind: {kind}")


def _apply_cps_effect(state: TypedState, effect: Mapping[str, Any]) -> dict[str, Any]:
    before = state.as_json()
    kind = effect["kind"]
    if kind == "add_fact":
        state.facts.add(effect["fact"])
    elif kind == "remove_fact":
        state.facts.discard(effect["fact"])
    elif kind == "resource_delta":
        state.resources[effect["resource"]] = state.resources.get(effect["resource"], 0) + int(effect["delta"])
    elif kind == "set_attribute":
        state.attributes.setdefault(effect["entity"], {})[effect["attribute"]] = effect["value"]
    elif kind == "link":
        state.relations.setdefault(effect["relation"], set()).add((effect["source"], effect["target"]))
    elif kind == "unlink":
        state.relations.setdefault(effect["relation"], set()).discard((effect["source"], effect["target"]))
    else:
        raise ValueError(f"unknown CPS effect kind: {kind}")
    return {"effect": deepcopy(effect), "before": before, "after": state.as_json()}


def _goal_holds(state: TypedState, goal: Sequence[Mapping[str, Any]]) -> bool:
    return all(_cps_precondition_holds(state, condition) for condition in goal)


def _constraints_hold(state: TypedState, constraints: Sequence[Mapping[str, Any]]) -> bool:
    return all(_cps_precondition_holds(state, condition) for condition in constraints)


def replay_cps_prefix(ast: Mapping[str, Any], candidate_index: int, prefix: int) -> dict[str, Any]:
    """Replay exactly the first ``prefix`` actions; legal_so_far is prefix-only."""

    plan = list(ast["candidates"][int(candidate_index)]["plan"])
    actions = {action["name"]: action for action in ast["actions"]}
    state = state_from_json(ast["initial_state"])
    trace: list[dict[str, Any]] = []
    legal_so_far = True
    cost = 0
    for position, action_name in enumerate(plan[: int(prefix)]):
        action = actions.get(action_name)
        if action is None:
            legal_so_far = False
            trace.append({"position": position, "action": action_name, "legal": False, "failure": "unknown_action", "state": state.as_json(), "cost": cost})
            break
        missing = [condition for condition in action["preconditions"] if not _cps_precondition_holds(state, condition)]
        if missing:
            legal_so_far = False
            trace.append({"position": position, "action": action_name, "legal": False, "failure": "precondition", "missing": missing, "state": state.as_json(), "cost": cost})
            break
        cost += int(action["cost"])
        deltas = [_apply_cps_effect(state, effect) for effect in action["effects"]]
        trace.append({"position": position, "action": action_name, "legal": True, "cost": int(action["cost"]), "deltas": deltas, "state": state.as_json(), "total_cost": cost})
    return {"candidate_index": int(candidate_index), "prefix": int(prefix), "legal_so_far": legal_so_far, "state": state, "state_json": state.as_json(), "cost": cost, "trace": trace}


def verify_cps_plan(ast: Mapping[str, Any], plan: Sequence[str]) -> dict[str, Any]:
    state = state_from_json(ast["initial_state"])
    actions = {action["name"]: action for action in ast["actions"]}
    trace: list[dict[str, Any]] = []
    legal = True
    failure: str | None = None
    failure_reasons: list[str] = []
    total_cost = 0
    for position, action_name in enumerate(plan):
        action = actions.get(action_name)
        if action is None:
            legal = False
            failure = "unknown_action"
            failure_reasons.append(failure)
            break
        missing = [condition for condition in action["preconditions"] if not _cps_precondition_holds(state, condition)]
        if missing:
            legal = False
            failure = "precondition"
            failure_reasons.append(failure)
            trace.append({"position": position, "action": action_name, "legal": False, "missing": missing, "state": state.as_json(), "total_cost": total_cost})
            break
        total_cost += int(action["cost"])
        deltas = [_apply_cps_effect(state, effect) for effect in action["effects"]]
        trace.append({"position": position, "action": action_name, "legal": True, "cost": int(action["cost"]), "deltas": deltas, "state": state.as_json(), "total_cost": total_cost})
    budget_ok = total_cost <= int(ast["budget"])
    goal_ok = legal and _goal_holds(state, ast["goal"])
    final_constraints_ok = legal and _constraints_hold(state, ast.get("final_constraints", []))
    if legal and not budget_ok:
        failure = "budget"
        failure_reasons.append(failure)
    if legal and not goal_ok:
        failure = "goal"
        failure_reasons.append(failure)
    if legal and goal_ok and not final_constraints_ok:
        failure = "final_constraint"
        failure_reasons.append(failure)
    valid = legal and budget_ok and goal_ok and final_constraints_ok
    return {
        "plan": list(plan), "legal": legal, "valid": valid, "total_cost": total_cost,
        "budget_ok": budget_ok, "goal_ok": goal_ok, "final_constraints_ok": final_constraints_ok,
        "failure": failure, "failure_reasons": sorted(set(failure_reasons)), "trace": trace, "final_state": state,
    }


def select_cps_answer(ast: Mapping[str, Any]) -> dict[str, Any]:
    results = [verify_cps_plan(ast, candidate["plan"]) for candidate in ast["candidates"]]
    valid = [(index, result) for index, result in enumerate(results) if result["valid"]]
    if not valid:
        return {"answer": None, "results": results, "unique_optimum": True}
    valid.sort(key=lambda item: (item[1]["total_cost"], item[0]))
    unique = len(valid) == 1 or valid[0][1]["total_cost"] < valid[1][1]["total_cost"]
    return {"answer": valid[0][0] if unique else None, "results": results, "unique_optimum": unique}
