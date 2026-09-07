from __future__ import annotations

"""Typed ERE dataflow derivation for G05.

The bundle declares only a class label and event-ablation outcomes.  This module
re-executes the AST with symbolic origins and derives the actual witness graph;
no path or neighbor supplied by the input is trusted.
"""

from copy import deepcopy
import hashlib
import json
from typing import Any

from yggdrasil_v2.r1_revalidation.common import simulate_ere


ERE_CLASSES = {
    "initial-copy",
    "condition-true",
    "condition-false",
    "swap-source",
    "relation-neighbor",
}
COMPOSITION_CLASSES = {"condition-true", "swap-source", "relation-neighbor"}


def _metric(value: Any, numerator: int, denominator: int, passed: bool, failures: list[str]) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "passed": bool(passed),
        "failures": sorted(set(str(item) for item in failures)),
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest().upper()


def _resolve(value: str, bindings: dict[str, str], neighbor: str | None) -> str:
    if value == "$neighbor":
        if neighbor is None:
            raise ValueError("neighbor placeholder outside FOREACH_LINKED")
        return neighbor
    if value.startswith("$arg:"):
        return bindings[value[5:]]
    return value


def _predicate(state: dict[str, Any], predicate: dict[str, Any], bindings: dict[str, str], neighbor: str | None) -> bool:
    if predicate["kind"] == "attribute_equals":
        entity = _resolve(predicate["entity"], bindings, neighbor)
        attribute = _resolve(predicate["attribute"], bindings, neighbor)
        value = _resolve(predicate["value"], bindings, neighbor)
        return state["attributes"].get(entity, {}).get(attribute) == value
    relation = _resolve(predicate["relation"], bindings, neighbor)
    source = _resolve(predicate["source"], bindings, neighbor)
    target = _resolve(predicate["target"], bindings, neighbor)
    return (source, target) in state["relations"].get(relation, set())


def _origin(kind: str, event: int, primitive: int, parent: Any = None, controls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": kind, "event": event, "primitive": primitive}
    if parent is not None:
        result["parent"] = deepcopy(parent)
    if controls:
        result["controls"] = deepcopy(controls)
    return result


def _apply(
    state: dict[str, Any],
    attribute_origins: dict[tuple[str, str], dict[str, Any]],
    relation_origins: dict[tuple[str, str, str], dict[str, Any]],
    primitive: dict[str, Any],
    bindings: dict[str, str],
    neighbor: str | None,
    event_index: int,
    primitive_index: int,
    controls: list[dict[str, Any]],
) -> list[str]:
    op = primitive["op"]
    if op == "IF":
        result = _predicate(state, primitive["predicate"], bindings, neighbor)
        control = {
            "kind": "if",
            "event": event_index,
            "primitive": primitive_index,
            "result": result,
            "branch": "then" if result else "else",
        }
        branch = primitive["then"] if result else primitive["else"]
        return [f"IF:{str(result).lower()}", *_apply(
            state,
            attribute_origins,
            relation_origins,
            branch,
            bindings,
            neighbor,
            event_index,
            primitive_index,
            [*controls, control],
        )]
    if op == "FOREACH_LINKED":
        relation = _resolve(primitive["relation"], bindings, neighbor)
        source = _resolve(primitive["source"], bindings, neighbor)
        neighbors = sorted(target for left, target in state["relations"].get(relation, set()) if left == source)
        operations = [f"FOREACH_LINKED:{len(neighbors)}"]
        for target in neighbors:
            edge_origin = relation_origins.get((relation, source, target), {"kind": "initial-relation"})
            control = {
                "kind": "foreach",
                "event": event_index,
                "primitive": primitive_index,
                "edge_origin": deepcopy(edge_origin),
            }
            operations.extend(
                _apply(
                    state,
                    attribute_origins,
                    relation_origins,
                    primitive["effect"],
                    bindings,
                    target,
                    event_index,
                    primitive_index,
                    [*controls, control],
                )
            )
        return operations

    resolved = {
        key: _resolve(value, bindings, neighbor)
        for key, value in primitive.items()
        if key != "op"
    }
    if op == "SET":
        key = (resolved["target"], resolved["attribute"])
        state["attributes"].setdefault(key[0], {})[key[1]] = resolved["value"]
        attribute_origins[key] = _origin("set", event_index, primitive_index, controls=controls)
        return ["SET"]
    if op == "COPY":
        source_key = (resolved["source"], resolved["attribute"])
        target_key = (resolved["target"], resolved["attribute"])
        if source_key not in attribute_origins:
            raise ValueError("COPY source has no typed origin")
        state["attributes"].setdefault(target_key[0], {})[target_key[1]] = deepcopy(
            state["attributes"][source_key[0]][source_key[1]]
        )
        attribute_origins[target_key] = _origin(
            "copy", event_index, primitive_index, parent=attribute_origins[source_key], controls=controls
        )
        return ["COPY"]
    if op == "SWAP":
        left = (resolved["left"], resolved["attribute"])
        right = (resolved["right"], resolved["attribute"])
        if left not in attribute_origins or right not in attribute_origins:
            raise ValueError("SWAP operand has no typed origin")
        left_value = deepcopy(state["attributes"][left[0]][left[1]])
        right_value = deepcopy(state["attributes"][right[0]][right[1]])
        left_origin = deepcopy(attribute_origins[left])
        right_origin = deepcopy(attribute_origins[right])
        state["attributes"][left[0]][left[1]] = right_value
        state["attributes"][right[0]][right[1]] = left_value
        attribute_origins[left] = _origin("swap", event_index, primitive_index, parent=right_origin, controls=controls)
        attribute_origins[right] = _origin("swap", event_index, primitive_index, parent=left_origin, controls=controls)
        return ["SWAP"]
    relation_key = (resolved["relation"], resolved["source"], resolved["target"])
    edge = (relation_key[1], relation_key[2])
    if op == "LINK":
        state["relations"].setdefault(relation_key[0], set()).add(edge)
        relation_origins[relation_key] = _origin("link", event_index, primitive_index, controls=controls)
        return ["LINK"]
    state["relations"].setdefault(relation_key[0], set()).discard(edge)
    relation_origins.pop(relation_key, None)
    return ["UNLINK"]


def _shape(value: Any) -> Any:
    """Discard symbol names and payload values while retaining causal topology."""

    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    result: dict[str, Any] = {"kind": kind}
    if "event" in value:
        result["event"] = value["event"]
    if "primitive" in value:
        result["primitive"] = value["primitive"]
    if kind == "if":
        result["result"] = value.get("result")
        result["branch"] = value.get("branch")
    if "parent" in value:
        result["parent"] = _shape(value["parent"])
    if "controls" in value:
        result["controls"] = [_shape(item) for item in value["controls"]]
    if "edge_origin" in value:
        result["edge_origin"] = _shape(value["edge_origin"])
    return result


def _collect_kinds(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("kind"), str):
            result.append(value["kind"])
        for child in value.values():
            result.extend(_collect_kinds(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_collect_kinds(child))
    return result


def derive_ere_provenance(ast: dict[str, Any], fresh_output: dict[str, Any]) -> dict[str, Any]:
    initial = ast["initial_state"]
    state = {
        "attributes": deepcopy(initial["attributes"]),
        "relations": {name: {tuple(edge) for edge in edges} for name, edges in initial["relations"].items()},
    }
    attribute_origins = {
        (entity, attribute): {"kind": "initial-attribute"}
        for entity, attributes in initial["attributes"].items()
        for attribute in attributes
    }
    relation_origins = {
        (relation, source, target): {"kind": "initial-relation"}
        for relation, edges in initial["relations"].items()
        for source, target in edges
    }
    actual_operations: list[list[str]] = []
    for event_index, event in enumerate(ast["events"]):
        rule = ast["rules"][event["rule"]]
        operations: list[str] = []
        for primitive_index, primitive in enumerate(rule["primitives"]):
            operations.extend(
                _apply(
                    state,
                    attribute_origins,
                    relation_origins,
                    primitive,
                    dict(event["arguments"]),
                    None,
                    event_index,
                    primitive_index,
                    [],
                )
            )
        actual_operations.append(operations)

    query = ast["query"]
    if query.get("kind") != "attribute":
        raise ValueError("G05 qualification requires an attribute query")
    query_key = (query["entity"], query["attribute"])
    witness = attribute_origins.get(query_key)
    if witness is None:
        raise ValueError("query payload has no typed origin")
    trace = fresh_output.get("trace")
    trace_binding = isinstance(trace, list) and len(trace) == len(actual_operations) and all(
        isinstance(trace[index], dict) and trace[index].get("executed_ops") == operations
        for index, operations in enumerate(actual_operations)
    )
    kinds = _collect_kinds(witness)
    if "foreach" in kinds and "link" in kinds and "copy" in kinds:
        derived_class = "relation-neighbor"
    elif "swap" in kinds and "copy" in kinds:
        derived_class = "swap-source"
    else:
        controls = [item for item in _walk_dicts(witness) if item.get("kind") == "if"]
        if controls:
            outcomes = {item.get("result") for item in controls}
            if outcomes == {True}:
                derived_class = "condition-true"
            elif outcomes == {False}:
                derived_class = "condition-false"
            else:
                derived_class = "ambiguous-condition"
        elif kinds.count("copy") >= 3 and "initial-attribute" in kinds:
            derived_class = "initial-copy"
        else:
            derived_class = "unclassified"
    signature = {
        "initial-copy": "copy-chain-3",
        "condition-true": "set-if-true-copy",
        "condition-false": "set-if-false-copy",
        "swap-source": "swap-copy-chain",
        "relation-neighbor": "link-foreach-copy",
    }.get(derived_class)
    shape = _shape(witness)
    return {
        "derived_class": derived_class,
        "composition_signature": signature,
        "trace_binding": trace_binding,
        "witness_sha256": _digest(shape),
    }


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        result.append(value)
        for child in value.values():
            result.extend(_walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_walk_dicts(child))
    return result


def _ablation_answers(ast: dict[str, Any]) -> list[Any]:
    answers: list[Any] = []
    for event_index, event in enumerate(ast["events"]):
        ablated = deepcopy(ast)
        source_rule = ablated["rules"][event["rule"]]
        noop_name = f"audit_noop_{event_index}"
        ablated["rules"][noop_name] = {"params": deepcopy(source_rule["params"]), "primitives": []}
        ablated["events"][event_index]["rule"] = noop_name
        answers.append(simulate_ere(ablated)["answer"])
    return answers


def provenance_bundle(records: list[dict[str, Any]], composition: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    outputs = replay.get("outputs", {})
    core = [row for row in records if row.get("structure_certificate", {}).get("role") == "ere_core"]
    derived: dict[str, dict[str, Any]] = {}
    depth_good = declaration_good = ablation_good = derivation_good = control_good = 0
    for row in core:
        identifier = str(row.get("example_id"))
        ast = row.get("program_ast")
        certificate = row.get("structure_certificate", {})
        if isinstance(ast, dict) and len(ast.get("events", [])) >= 3:
            depth_good += 1
        indices = list(range(len(ast.get("events", [])))) if isinstance(ast, dict) else []
        if certificate.get("necessary_event_indices") == indices:
            declaration_good += 1
        try:
            answers = _ablation_answers(ast)
            answer = outputs[identifier]["answer"]
            if answers == certificate.get("ablation_answers") and all(item != answer for item in answers):
                ablation_good += 1
            item = derive_ere_provenance(ast, outputs[identifier])
            derived[identifier] = item
            if (
                item["derived_class"] == certificate.get("provenance")
                and item["composition_signature"] == certificate.get("composition_signature")
                and item["derived_class"] in ERE_CLASSES
            ):
                derivation_good += 1
            if item["trace_binding"] is True:
                control_good += 1
        except Exception as exc:
            derived[identifier] = {
                "derived_class": "error",
                "composition_signature": None,
                "trace_binding": False,
                "witness_sha256": _digest({"error": type(exc).__name__}),
            }

    classes = [item.get("derived_class") for item in derived.values()]
    coverage_ok = len(core) == 5 and set(classes) == ERE_CLASSES and all(classes.count(name) == 1 for name in ERE_CLASSES)
    by_id = {str(row.get("example_id")): row for row in core}
    ordinary_ids = composition.get("ere_ordinary_provenance_ids")
    composition_ids = composition.get("ere_composition_provenance_ids")
    balance_ok = False
    if isinstance(ordinary_ids, list) and isinstance(composition_ids, list):
        ordinary_classes = [derived.get(str(identifier), {}).get("derived_class") for identifier in ordinary_ids]
        composition_classes = [derived.get(str(identifier), {}).get("derived_class") for identifier in composition_ids]
        shares = [composition_classes.count(name) for name in set(composition_classes)]
        balance_ok = bool(
            len(ordinary_ids) == 5
            and len(set(ordinary_ids)) == 5
            and all(str(identifier) in by_id for identifier in ordinary_ids)
            and set(ordinary_classes) == ERE_CLASSES
            and len(composition_ids) == 3
            and len(set(composition_ids)) == 3
            and set(composition_classes) == COMPOSITION_CLASSES
            and shares
            and all(count * 3 <= len(composition_classes) for count in shares)
        )
    denominator = len(core)
    return {
        "derived": {identifier: derived[identifier] for identifier in sorted(derived)},
        "g05": {
            "G05_M01_core_profile_and_depth_exact": _metric(depth_good, depth_good, denominator, denominator == 5 and depth_good == 5, ["ERE core profile/depth mismatch"] if denominator != 5 or depth_good != 5 else []),
            "G05_M02_event_declaration_exact": _metric(declaration_good / denominator if denominator else 0.0, declaration_good, denominator, denominator == 5 and declaration_good == 5, ["necessary event declaration mismatch"] if declaration_good != denominator else []),
            "G05_M03_fresh_event_ablation_rate": _metric(ablation_good / denominator if denominator else 0.0, ablation_good, denominator, denominator == 5 and ablation_good == 5, ["fresh ablation mismatch"] if ablation_good != denominator else []),
            "G05_M04_typed_provenance_derivation_rate": _metric(derivation_good / denominator if denominator else 0.0, derivation_good, denominator, denominator == 5 and derivation_good == 5, ["typed provenance mismatch"] if derivation_good != denominator else []),
            "G05_M05_provenance_class_coverage_exact": _metric(coverage_ok, int(coverage_ok), 1, coverage_ok, ["derived provenance class profile mismatch"] if not coverage_ok else []),
            "G05_M06_control_trace_binding_rate": _metric(control_good / denominator if denominator else 0.0, control_good, denominator, denominator == 5 and control_good == 5, ["control/trace binding mismatch"] if control_good != denominator else []),
            "G05_M07_provenance_balance_exact": _metric(balance_ok, int(balance_ok), 1, balance_ok, ["ordinary/composition provenance balance mismatch"] if not balance_ok else []),
        },
    }
