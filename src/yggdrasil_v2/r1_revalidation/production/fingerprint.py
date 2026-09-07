from __future__ import annotations

"""Alpha-invariant semantic fingerprints with explicit ordered/unordered paths."""

from collections import defaultdict
import hashlib
import json
from typing import Any


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
UNORDERED_LIST_PATHS = {
    ("actions",),
    ("candidates",),
    ("goal",),
    ("final_constraints",),
    ("initial_state", "facts"),
    ("initial_state", "relations", "<dynamic>"),
    ("actions", "<item>", "preconditions"),
    ("rules", "<dynamic>", "params"),
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


def _reserved_literal(path: tuple[str, ...], value: str) -> bool:
    """Distinguish grammar literals from domain symbols by AST position.

    A domain symbol is allowed to have the same spelling as a schema key or an
    operator literal.  Only ``op``/``kind`` payloads and ``$neighbor`` are
    grammar literals; the same bytes elsewhere participate in alpha-renaming.
    """

    return value == "$neighbor" or bool(path and path[-1] in {"op", "kind"} and value in RESERVED_VALUES)


def alpha_rename(value: Any) -> Any:
    """Return a deterministic alpha-renamed AST using fingerprint path rules."""

    mapping: dict[str, str] = {}

    def symbol(item: str) -> str:
        if item not in mapping:
            mapping[item] = f"z{len(mapping)}"
        return mapping[item]

    def walk(item: Any, path: tuple[str, ...] = ()) -> Any:
        if isinstance(item, dict):
            dynamic = _dynamic_key_map(path)
            renamed: dict[str, Any] = {}
            for key, child in item.items():
                if not dynamic and key in SCHEMA_KEYS:
                    renamed[key] = walk(child, (*path, key))
                else:
                    renamed[symbol(str(key))] = walk(child, (*path, "<dynamic>"))
            return renamed
        if isinstance(item, list):
            return [walk(child, (*path, "<item>")) for child in item]
        if isinstance(item, str):
            if _reserved_literal(path, item):
                return item
            if item.startswith("$arg:"):
                return "$arg:" + symbol(item[5:])
            return symbol(item)
        return item

    return walk(value)


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
        unordered = path in UNORDERED_LIST_PATHS
        for index, child in enumerate(value):
            edge = "member" if unordered else f"index:{index}"
            graph.edge(root, edge, _build_graph(graph, child, (*path, "<item>")))
        return root
    if isinstance(value, str):
        if _reserved_literal(path, value):
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
    initial_palette = {label: index for index, label in enumerate(sorted(set(graph.labels)))}
    colors = [initial_palette[label] for label in graph.labels]
    for _ in range(len(graph.labels) + 1):
        signatures: list[tuple[Any, ...]] = []
        for index, label in enumerate(graph.labels):
            neighborhood = [f"out:{edge}:{colors[target]}" for edge, target in graph.outgoing[index]]
            neighborhood.extend(f"in:{edge}:{colors[source]}" for edge, source in graph.incoming[index])
            # Retaining the previous color makes refinement monotone: classes
            # may split but can never merge again.
            signatures.append((label, f"self:{colors[index]}", *sorted(neighborhood)))
        palette = {signature: index for index, signature in enumerate(sorted(set(signatures)))}
        updated = [palette[signature] for signature in signatures]
        stable = len(set(updated)) == len(set(colors))
        colors = updated
        if stable:
            break
    rows = [
        {
            "label": label,
            "color": colors[index],
            "out": sorted((edge, colors[target]) for edge, target in graph.outgoing[index]),
            "in": sorted((edge, colors[source]) for edge, source in graph.incoming[index]),
        }
        for index, label in enumerate(graph.labels)
    ]
    return _sha(_canonical_bytes({"root": colors[root], "nodes": sorted(rows, key=_canonical_bytes)}))


def surface_fingerprint(source_text: str) -> str:
    if not isinstance(source_text, str):
        raise TypeError("source_text must be a string")
    return _sha(source_text.encode("utf-8"))


__all__ = ["SCHEMA_KEYS", "RESERVED_VALUES", "alpha_rename", "semantic_fingerprint", "surface_fingerprint"]
