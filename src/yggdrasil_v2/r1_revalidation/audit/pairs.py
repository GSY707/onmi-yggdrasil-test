from __future__ import annotations

"""Independent finite-schema fingerprints and pair contracts for G04."""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from .language import parse_language, project_language_ast


SCHEMA_KEYS = {
    "actions", "amount", "arguments", "attribute", "attributes", "budget", "candidates", "cost", "delta",
    "effect", "effects", "else", "entity", "events", "fact", "facts", "final_constraints", "goal", "initial_state",
    "kind", "left", "name", "op", "params", "plan", "preconditions", "predicate", "primitives", "query", "relation",
    "relations", "resource", "resources", "right", "rule", "rules", "source", "target", "then", "value",
}
RESERVED_VALUES = {
    "SET", "COPY", "SWAP", "LINK", "UNLINK", "IF", "FOREACH_LINKED", "attribute", "relation", "attribute_equals",
    "relation_exists", "fact_true", "fact_false", "resource_at_least", "add_fact", "remove_fact", "resource_delta",
    "set_attribute", "link", "unlink", "$neighbor",
}


class _Graph:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self.outgoing: dict[int, list[tuple[str, int]]] = defaultdict(list)
        self.incoming: dict[int, list[tuple[str, int]]] = defaultdict(list)
        self.symbols: dict[str, int] = {}

    def node(self, label: str) -> int:
        index = len(self.labels)
        self.labels.append(label)
        return index

    def edge(self, source: int, label: str, target: int) -> None:
        self.outgoing[source].append((label, target))
        self.incoming[target].append((label, source))

    def symbol(self, value: str) -> int:
        if value not in self.symbols:
            self.symbols[value] = self.node("symbol")
        return self.symbols[value]


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _primitive_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        return f"float:{value!r}"
    raise TypeError(type(value))


def _dynamic_key_map(path: tuple[str, ...]) -> bool:
    if path in {("rules",), ("initial_state", "attributes"), ("initial_state", "relations"), ("initial_state", "resources")}:
        return True
    if len(path) == 3 and path[:2] == ("initial_state", "attributes"):
        return True
    return bool(path and path[-1] == "arguments")


def _build_graph(graph: _Graph, value: Any, path: tuple[str, ...] = ()) -> int:
    if isinstance(value, dict):
        root = graph.node("dict")
        dynamic = _dynamic_key_map(path)
        for key, child in value.items():
            if not dynamic and key in SCHEMA_KEYS:
                child_node = _build_graph(graph, child, (*path, key))
                graph.edge(root, f"field:{key}", child_node)
            else:
                entry = graph.node("dynamic-entry")
                graph.edge(root, "entry", entry)
                graph.edge(entry, "key", graph.symbol(str(key)))
                graph.edge(entry, "value", _build_graph(graph, child, (*path, "<dynamic>")))
        return root
    if isinstance(value, list):
        root = graph.node("list")
        for index, child in enumerate(value):
            graph.edge(root, f"index:{index}", _build_graph(graph, child, (*path, "<item>")))
        return root
    if isinstance(value, str):
        if value in RESERVED_VALUES:
            return graph.node(f"reserved:{value}")
        if value.startswith("$arg:"):
            root = graph.node("arg-reference")
            graph.edge(root, "parameter", graph.symbol(value[5:]))
            return root
        return graph.symbol(value)
    return graph.node(_primitive_label(value))


def semantic_fingerprint(ast: dict[str, Any]) -> str:
    graph = _Graph()
    root = _build_graph(graph, ast)
    colors = [_sha(label.encode("utf-8")) for label in graph.labels]
    for _ in range(len(graph.labels) + 1):
        updated: list[str] = []
        for index, label in enumerate(graph.labels):
            neighborhood = [f"out:{edge}:{colors[target]}" for edge, target in graph.outgoing[index]]
            neighborhood.extend(f"in:{edge}:{colors[source]}" for edge, source in graph.incoming[index])
            updated.append(_sha("\n".join([label, *sorted(neighborhood)]).encode("utf-8")))
        if updated == colors:
            break
        colors = updated
    rows = []
    for index, label in enumerate(graph.labels):
        rows.append({
            "label": label,
            "color": colors[index],
            "out": sorted((edge, colors[target]) for edge, target in graph.outgoing[index]),
            "in": sorted((edge, colors[source]) for edge, source in graph.incoming[index]),
        })
    return _sha(_canonical_bytes({"root": colors[root], "nodes": sorted(rows, key=lambda row: _canonical_bytes(row))}))


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def surface_fingerprint(text: str) -> str:
    return _sha(normalize_surface(text).encode("utf-8"))


def _metric(value: Any, numerator: int, denominator: int, passed: bool, failures: list[str]) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "passed": bool(passed),
        "failures": sorted(set(str(item) for item in failures)),
    }


def _diff_paths(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_diff_paths(left[key], right[key], child))
        return result
    if isinstance(left, list):
        result = []
        for index in range(max(len(left), len(right))):
            child = f"{path}/{index}"
            if index >= len(left) or index >= len(right):
                result.append(child)
            else:
                result.extend(_diff_paths(left[index], right[index], child))
        return result
    return [] if left == right else [path or "/"]


def _at_path(value: Any, path: str) -> Any:
    current = value
    if not path.startswith("/"):
        raise ValueError(path)
    for part in path.split("/")[1:]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _ops(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("op"), str):
            found.add(value["op"])
        for child in value.values():
            found |= _ops(child)
    elif isinstance(value, list):
        for child in value:
            found |= _ops(child)
    return found


def _record_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("example_id")): row for row in records if isinstance(row.get("example_id"), str)}


def _causal_ok(pair: dict[str, Any], rows: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]]) -> bool:
    base = rows.get(str(pair.get("base_id")))
    flip = rows.get(str(pair.get("flip_id")))
    if (
        base is None
        or flip is None
        or base.get("family") != flip.get("family")
        or pair.get("family") != base.get("family")
        or base.get("structure_certificate", {}).get("role") != "causal_member"
        or flip.get("structure_certificate", {}).get("role") != "causal_member"
    ):
        return False
    try:
        differences = _diff_paths(base["program_ast"], flip["program_ast"])
        actual_from = _at_path(base["program_ast"], str(pair["changed_path"]))
        actual_to = _at_path(flip["program_ast"], str(pair["changed_path"]))
    except Exception:
        return False
    changed_path = pair.get("changed_path")
    derived_mutation = None
    if pair.get("family") == "ere" and isinstance(changed_path, str) and changed_path.startswith("/initial_state/attributes/"):
        derived_mutation = "initial-attribute"
    if pair.get("family") == "cps" and isinstance(changed_path, str) and re.fullmatch(r"/actions/\d+/cost", changed_path):
        derived_mutation = "action-cost"
    return bool(
        differences == [pair.get("changed_path")]
        and actual_from == pair.get("from")
        and actual_to == pair.get("to")
        and pair.get("expected_answer_flip") == [outputs.get(str(pair.get("base_id")), {}).get("answer"), outputs.get(str(pair.get("flip_id")), {}).get("answer")]
        and outputs.get(str(pair.get("base_id")), {}).get("answer") != outputs.get(str(pair.get("flip_id")), {}).get("answer")
        and base.get("label_mapping") == flip.get("label_mapping")
        and base.get("valid_choice_mask") == flip.get("valid_choice_mask")
        and pair.get("mutation_family") == derived_mutation
    )


def _language_ok(pair: dict[str, Any], rows: dict[str, dict[str, Any]]) -> bool:
    row = rows.get(str(pair.get("record_id")))
    if (
        row is None
        or pair.get("family") != row.get("family")
        or row.get("structure_certificate", {}).get("role") != "causal_member"
    ):
        return False
    train = pair.get("train_text")
    ood = pair.get("ood_text")
    if not isinstance(train, str) or not isinstance(ood, str):
        return False
    parsed_train = parse_language(train)
    parsed_ood = parse_language(ood)
    projection = project_language_ast(row.get("family"), row.get("program_ast"))
    return bool(
        parsed_train is not None
        and parsed_ood is not None
        and projection is not None
        and parsed_train["family"] == row.get("family")
        and parsed_ood["family"] == row.get("family")
        and parsed_train["variant"] == pair.get("train_variant")
        and parsed_ood["variant"] == pair.get("ood_variant")
        and parsed_train["variant"] != parsed_ood["variant"]
        and parsed_train["projection"] == projection
        and parsed_ood["projection"] == projection
        and train.splitlines()[1:] != ood.splitlines()[1:]
        and pair.get("semantic_fingerprint") == row.get("semantic_fingerprint")
        and pair.get("train_surface_fingerprint") == surface_fingerprint(train)
        and pair.get("ood_surface_fingerprint") == surface_fingerprint(ood)
        and pair.get("train_surface_fingerprint") != pair.get("ood_surface_fingerprint")
    )


def pair_bundle(
    records: list[dict[str, Any]],
    causal_pairs: list[dict[str, Any]],
    language_pairs: list[dict[str, Any]],
    composition: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    rows = _record_map(records)
    outputs = replay.get("outputs", {}) if isinstance(replay, dict) else {}
    fingerprint_good = 0
    fingerprint_denominator = 0
    for row in records:
        for key, function in (("semantic_fingerprint", lambda value: semantic_fingerprint(value)), ("surface_fingerprint", surface_fingerprint)):
            fingerprint_denominator += 1
            try:
                if row.get(key) == function(row["program_ast"] if key.startswith("semantic") else row["source_text"]):
                    fingerprint_good += 1
            except Exception:
                pass
    for pair in language_pairs:
        row = rows.get(str(pair.get("record_id")), {})
        checks = (
            pair.get("semantic_fingerprint") == row.get("semantic_fingerprint"),
            pair.get("train_surface_fingerprint") == surface_fingerprint(pair.get("train_text", "")),
            pair.get("ood_surface_fingerprint") == surface_fingerprint(pair.get("ood_text", "")),
        )
        fingerprint_denominator += len(checks)
        fingerprint_good += sum(checks)

    semantic_counts = Counter(str(row.get("semantic_fingerprint")) for row in records)
    surface_counts = Counter(str(row.get("surface_fingerprint")) for row in records)
    semantic_overlap = sum(count - 1 for count in semantic_counts.values() if count > 1)
    surface_overlap = sum(count - 1 for count in surface_counts.values() if count > 1)
    overlap_count = semantic_overlap + surface_overlap

    causal_profile_ok = bool(
        len(causal_pairs) == 2
        and {pair.get("family") for pair in causal_pairs} == {"ere", "cps"}
        and len({pair.get("pair_id") for pair in causal_pairs}) == 2
        and all(isinstance(pair.get("pair_id"), str) and pair["pair_id"].strip() for pair in causal_pairs)
    )
    causal_good = sum(_causal_ok(pair, rows, outputs) for pair in causal_pairs)
    causal_denominator = len(causal_pairs)
    families = Counter(str(pair.get("mutation_family")) for pair in causal_pairs)
    family_deviation = max(families.values()) - min(families.values()) if families else 0

    language_profile_ok = bool(
        len(language_pairs) == 2
        and {pair.get("family") for pair in language_pairs} == {"ere", "cps"}
        and len({pair.get("pair_id") for pair in language_pairs}) == 2
        and all(isinstance(pair.get("pair_id"), str) and pair["pair_id"].strip() for pair in language_pairs)
    )
    language_good = sum(_language_ok(pair, rows) for pair in language_pairs)
    language_denominator = len(language_pairs)
    fold_good = 0
    fold_denominator = 0
    for pair in causal_pairs:
        base = rows.get(str(pair.get("base_id")))
        flip = rows.get(str(pair.get("flip_id")))
        fold_denominator += 1
        if base is not None and flip is not None and base.get("fold_group") == flip.get("fold_group") == pair.get("pair_id"):
            fold_good += 1
    for pair in language_pairs:
        fold_denominator += 1
        if pair.get("fold_group") == pair.get("pair_id"):
            fold_good += 1

    required_ops = composition.get("ere_required_component_ops")
    ordinary_ids = composition.get("ere_ordinary_provenance_ids")
    ordinary_rows = [rows.get(str(identifier)) for identifier in ordinary_ids] if isinstance(ordinary_ids, list) else []
    actual_ops = set().union(*(_ops(row.get("program_ast")) for row in ordinary_rows if isinstance(row, dict))) if ordinary_rows else set()
    component_ok = isinstance(required_ops, list) and len(required_ops) == len(set(required_ops)) and set(required_ops) == actual_ops
    cps_id = composition.get("cps_record_id")
    cps_row = rows.get(str(cps_id))
    required_witnesses = composition.get("cps_required_witnesses")
    witness_ok = isinstance(required_witnesses, list) and set(required_witnesses) == {
        "unlock_resource_goal", "same_goal_different_final", "resource_cost_budget", "early_fact_destruction"
    }
    composition_ok = component_ok and witness_ok

    return {
        "record_map": rows,
        "fingerprint_good": fingerprint_good,
        "fingerprint_denominator": fingerprint_denominator,
        "causal_good": causal_good,
        "causal_denominator": causal_denominator,
        "language_good": language_good,
        "language_denominator": language_denominator,
        "fold_good": fold_good,
        "fold_denominator": fold_denominator,
        "overlap_count": overlap_count,
        "family_deviation": family_deviation,
        "component_ok": component_ok,
        "g04": {
            "G04_M01_fingerprint_recompute_rate": _metric(fingerprint_good / fingerprint_denominator if fingerprint_denominator else 0.0, fingerprint_good, fingerprint_denominator, fingerprint_denominator > 0 and fingerprint_good == fingerprint_denominator, ["fingerprint recomputation mismatch"] if fingerprint_good != fingerprint_denominator else []),
            "G04_M02_unpaired_overlap_count": _metric(overlap_count, overlap_count, 1, overlap_count == 0, ["unpaired semantic/surface overlap"] if overlap_count else []),
            "G04_M03_causal_pair_profile_exact": _metric(causal_profile_ok, int(causal_profile_ok), 1, causal_profile_ok, ["causal pair profile mismatch"] if not causal_profile_ok else []),
            "G04_M04_causal_pair_contract_rate": _metric(causal_good / causal_denominator if causal_denominator else 0.0, causal_good, causal_denominator, causal_denominator == 2 and causal_good == 2, ["causal pair contract mismatch"] if causal_good != causal_denominator else []),
            "G04_M05_mutation_family_deviation_max": _metric(family_deviation, family_deviation, 1, causal_profile_ok and family_deviation <= 1, ["mutation family imbalance"] if family_deviation > 1 or not causal_profile_ok else []),
            "G04_M06_language_pair_profile_exact": _metric(language_profile_ok, int(language_profile_ok), 1, language_profile_ok, ["language pair profile mismatch"] if not language_profile_ok else []),
            "G04_M07_reversible_language_binding_rate": _metric(language_good / language_denominator if language_denominator else 0.0, language_good, language_denominator, language_denominator == 2 and language_good == 2, ["reversible language binding mismatch"] if language_good != language_denominator else []),
            "G04_M08_fold_and_component_contract_exact": _metric(1.0 if composition_ok and fold_good == fold_denominator and fold_denominator == 4 else 0.0, int(composition_ok and fold_good == fold_denominator and fold_denominator == 4), 1, composition_ok and fold_good == fold_denominator and fold_denominator == 4, ["fold/component contract mismatch"] if not composition_ok or fold_good != fold_denominator or fold_denominator != 4 else []),
        },
    }
