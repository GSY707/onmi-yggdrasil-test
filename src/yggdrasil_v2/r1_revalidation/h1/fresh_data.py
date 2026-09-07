from __future__ import annotations

"""Fresh, task-agnostic source/target components for P1-H1.

The model view is deliberately small: :class:`SourceRecord` has only ``id``,
``source_text`` and ``source_atoms``.  Structured semantics, labels and traces
live in a separate target store and are never needed to render a source.
This module has no imports from the historical P0 data or simulator modules.
"""

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..nr1.reference import numeric_oracle as nr1_numeric_oracle
from ..nr1.reference import relation_oracle as nr1_relation_oracle
from ..nr1.schema import (
    NumericCandidate as NR1NumericCandidate,
    NumericCase as NR1NumericCase,
    RelationCase as NR1RelationCase,
    RelationQuery as NR1RelationQuery,
)


SCHEMA_VERSION = "yggdrasil.v2-r1r.p1-h1.fresh-data.v2"
SPLITS = ("train", "validation", "supported", "heldout", "causal")
LABELS = ("A", "B", "C", "D")
UNDECIDED = 4
_HANDLE = re.compile(r"^[a-z][a-z0-9]{5,11}$")
_FORBIDDEN = (
    "task_id", "task", "family", "winner", "final", "prefix", "closure", "decision",
    "candidate_index", "oracle", "span", "role", "answer", "simulator",
    "state", "numeric", "relation", "cache",
)
_TEMPLATES = (
    "Review these entries carefully. {instruction} {body} Options: {options}.",
    "Study the listed entries and select carefully. {instruction} {body} Choices: {options}.",
    "Read the entries in order, then select carefully. {instruction} Select from {options}. {body}",
    "Inspect these entries from another angle. {instruction} {body} Options: {options}.",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest().upper()


def _check_handle(handle: str) -> None:
    if not isinstance(handle, str) or _HANDLE.fullmatch(handle) is None:
        raise ValueError(f"invalid opaque handle: {handle!r}")
    if any(word in handle.casefold() for word in _FORBIDDEN):
        raise ValueError(f"forbidden handle token: {handle!r}")


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split!r}")


def _unique(items: Iterable[Any], message: str) -> None:
    values = list(items)
    if len(values) != len(set(values)):
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class NumericComponent:
    """Four opaque handles with equal-length, non-zero signed changes."""

    handles: tuple[str, ...]
    deltas: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.handles) != 4 or len(self.deltas) != 4:
            raise ValueError("a numeric component requires exactly four handles")
        _unique(self.handles, "numeric handles must be unique")
        for handle in self.handles:
            _check_handle(handle)
        lengths = set()
        for values in self.deltas:
            if not 4 <= len(values) <= 12:
                raise ValueError("numeric horizon must be four to twelve")
            lengths.add(len(values))
            if any(type(value) is not int or value == 0 for value in values):
                raise ValueError("changes must be non-zero signed integers")
        if len(lengths) != 1:
            raise ValueError("numeric handles must share a horizon")

    def renamed(self, mapping: Mapping[str, str]) -> "NumericComponent":
        _validate_rename(self.handles, mapping)
        return NumericComponent(tuple(mapping[h] for h in self.handles), self.deltas)


@dataclass(frozen=True, slots=True)
class RelationQuery:
    source: str
    target: str

    def __post_init__(self) -> None:
        _check_handle(self.source)
        _check_handle(self.target)
        if self.source == self.target:
            raise ValueError("a query needs distinct handles")


@dataclass(frozen=True, slots=True)
class RelationComponent:
    """Opaque handles and a non-chain directed acyclic graph."""

    handles: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    queries: tuple[RelationQuery, ...]

    def __post_init__(self) -> None:
        if not 6 <= len(self.handles) <= 10:
            raise ValueError("a relation component needs six to ten handles")
        _unique(self.handles, "relation handles must be unique")
        for handle in self.handles:
            _check_handle(handle)
        known = set(self.handles)
        if not self.edges:
            raise ValueError("relation graph cannot be empty")
        _unique(self.edges, "relation edges must be unique")
        for source, target in self.edges:
            if source not in known or target not in known or source == target:
                raise ValueError("edge uses unknown or self handle")
        if _has_cycle(self.handles, self.edges):
            raise ValueError("relation graph must be acyclic")
        degree_out = {h: 0 for h in self.handles}
        degree_in = {h: 0 for h in self.handles}
        for source, target in self.edges:
            degree_out[source] += 1
            degree_in[target] += 1
        if max(degree_in.values()) < 2 and max(degree_out.values()) < 2:
            raise ValueError("relation graph must branch or merge; a chain is not allowed")
        if len(self.queries) != 4:
            raise ValueError("relation components require four query pairs")
        _unique(self.queries, "relation queries must be unique")
        if any(q.source not in known or q.target not in known for q in self.queries):
            raise ValueError("query uses unknown handle")

    def renamed(self, mapping: Mapping[str, str]) -> "RelationComponent":
        _validate_rename(self.handles, mapping)
        rename = lambda h: mapping[h]
        return RelationComponent(
            tuple(rename(h) for h in self.handles),
            tuple((rename(a), rename(b)) for a, b in self.edges),
            tuple(RelationQuery(rename(q.source), rename(q.target)) for q in self.queries),
        )


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """The complete model-visible record, and nothing more."""

    id: str
    source_text: str
    source_atoms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("source record id is required")
        if not isinstance(self.source_text, str) or not self.source_text.strip():
            raise ValueError("source text is required")
        if "\r" in self.source_text or "\n" not in self.source_text:
            raise ValueError("source_text must use newline-separated clause atoms")
        if "\n" not in self.source_text or "" in self.source_atoms:
            raise ValueError("source_atoms must be non-empty newline clauses")
        if "\n".join(self.source_atoms) != self.source_text:
            raise ValueError("source_atoms must reconstruct source_text exactly")
        violations = _scan_text(self.source_text)
        if violations:
            raise ValueError(f"forbidden model-view fields: {sorted(violations)}")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "source_text": self.source_text, "source_atoms": self.source_atoms}

@dataclass(frozen=True, slots=True)
class NumericOracleResult:
    prefix_winners: tuple[str | int, ...]
    final_winner: str


@dataclass(frozen=True, slots=True)
class RelationOracleResult:
    reachable: tuple[bool, ...]
    query_index: int


@dataclass(frozen=True, slots=True)
class NumericTarget:
    prefix_labels: tuple[str | int, ...]
    final_label: str
    choice_to_handle: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RelationTarget:
    label: str
    query_index: int
    trace: tuple[int, ...]
    choice_to_query: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class FreshBundle:
    records: tuple[SourceRecord, ...]
    targets: Mapping[str, NumericTarget | RelationTarget]
    components: Mapping[str, NumericComponent | RelationComponent]
    splits: Mapping[str, tuple[str, ...]]
    semantic_fingerprints: Mapping[str, str]
    source_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        ids = tuple(record.id for record in self.records)
        _unique(ids, "record ids must be unique")
        if set(ids) != set(self.targets) or set(ids) != set(self.components):
            raise ValueError("records, targets and components must have identical ids")
        if set(self.semantic_fingerprints) != set(ids) or set(self.source_fingerprints) != set(ids):
            raise ValueError("fingerprints must cover every record")
        listed = [item for rows in self.splits.values() for item in rows]
        if set(self.splits) != set(SPLITS) or sorted(listed) != sorted(ids):
            raise ValueError("split membership must cover every record exactly once")
        _unique(listed, "a record cannot occur in multiple splits")


def _scan_text(text: str) -> set[str]:
    lowered = text.casefold()
    return {word for word in _FORBIDDEN if re.search(rf"(?<![a-z0-9_]){re.escape(word)}(?![a-z0-9_])", lowered)}


def forbidden_source_scan(value: FreshBundle | SourceRecord | Iterable[SourceRecord]) -> dict[str, tuple[str, ...]]:
    records = [value] if isinstance(value, SourceRecord) else list(value.records if isinstance(value, FreshBundle) else value)
    return {record.id: tuple(sorted(_scan_text(record.source_text))) for record in records if _scan_text(record.source_text)}


def _has_cycle(handles: Sequence[str], edges: Sequence[tuple[str, str]]) -> bool:
    adjacency = {h: [] for h in handles}
    indegree = {h: 0 for h in handles}
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    todo = [h for h in handles if indegree[h] == 0]
    seen = 0
    while todo:
        source = todo.pop()
        seen += 1
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                todo.append(target)
    return seen != len(handles)


def _validate_rename(handles: Sequence[str], mapping: Mapping[str, str]) -> None:
    if set(mapping) != set(handles) or len(set(mapping.values())) != len(handles):
        raise ValueError("handle rename must be total and collision-free")
    for value in mapping.values():
        _check_handle(value)


def numeric_oracle(component: NumericComponent) -> NumericOracleResult:
    """Use the qualified NR1 accumulate/minimum oracle only on the target side."""
    if not isinstance(component, NumericComponent):
        raise TypeError("numeric_oracle requires NumericComponent")
    qualified = nr1_numeric_oracle(
        NR1NumericCase(
            case_id="h1-offline-numeric",
            split="qualification",
            candidates=tuple(
                NR1NumericCandidate(handle=handle, deltas=values)
                for handle, values in zip(
                    component.handles, component.deltas, strict=True
                )
            ),
        )
    )
    prefixes: list[str | int] = []
    for step in range(len(component.deltas[0])):
        totals = [
            int(qualified["prefixes"][handle][step])
            for handle in component.handles
        ]
        top = min(totals)
        winners = [i for i, total in enumerate(totals) if total == top]
        prefixes.append(component.handles[winners[0]] if len(winners) == 1 else UNDECIDED)
    return NumericOracleResult(tuple(prefixes), str(qualified["winner"]))


def relation_oracle(component: RelationComponent) -> RelationOracleResult:
    """Use the qualified NR1 one-BFS-per-query oracle only on the target side."""
    if not isinstance(component, RelationComponent):
        raise TypeError("relation_oracle requires RelationComponent")
    qualified = nr1_relation_oracle(
        NR1RelationCase(
            case_id="h1-offline-relation",
            split="qualification",
            handles=component.handles,
            edges=component.edges,
            queries=tuple(
                NR1RelationQuery(source=query.source, target=query.target)
                for query in component.queries
            ),
        )
    )
    reachable = [bool(value) for value in qualified["decisions"]]
    if sum(reachable) != 1:
        raise ValueError("relation component must have exactly one reachable query")
    return RelationOracleResult(tuple(reachable), reachable.index(True))


def relation_fixed_point_trace(component: RelationComponent) -> tuple[int, ...]:
    """Independent iterative closure trace: query index per round, or ``4``."""
    if not isinstance(component, RelationComponent):
        raise TypeError("relation_fixed_point_trace requires RelationComponent")
    known = {(a, b) for a, b in component.edges}
    trace: list[int] = []
    query_edges = [(q.source, q.target) for q in component.queries]
    while True:
        hits = [i for i, edge in enumerate(query_edges) if edge in known]
        trace.append(hits[0] if hits else UNDECIDED)
        additions = {
            (a, d)
            for a, b in known
            for c, d in known
            if b == c and a != d and (a, d) not in known
        }
        if not additions:
            return tuple(trace)
        known.update(additions)


def _semantic_value(component: NumericComponent | RelationComponent) -> Any:
    if isinstance(component, NumericComponent):
        # Handle names and presentation order are not semantic identity.
        return [sorted(tuple(values) for values in component.deltas)]
    # Canonical graph and query shape, with handles renamed in first-seen order.
    names: dict[str, str] = {}
    def canon(handle: str) -> str:
        if handle not in names:
            names[handle] = f"h{len(names)}"
        return names[handle]
    edges = sorted((canon(a), canon(b)) for a, b in component.edges)
    queries = sorted((canon(q.source), canon(q.target)) for q in component.queries)
    return {
        "handle_count": len(component.handles),
        "edges": edges,
        "queries": queries,
    }


def _relation_example_semantic_value(
    component: RelationComponent,
    target: RelationTarget,
) -> dict[str, Any]:
    """Return a rename/query-order invariant supervised graph identity.

    Component-only graph shape is insufficient for leakage detection: option
    labels, unused distractor handles, and the registered trace are observable
    parts of an example.  A small directed Weisfeiler--Lehman refinement gives
    opaque handles stable structural colors without importing a graph library.
    The final payload keeps only color multisets and label-bound query pairs.
    """
    predecessors = {handle: [] for handle in component.handles}
    successors = {handle: [] for handle in component.handles}
    for source, destination in component.edges:
        successors[source].append(destination)
        predecessors[destination].append(source)
    roles = {handle: [] for handle in component.handles}
    for label, query_index in target.choice_to_query:
        query = component.queries[query_index]
        roles[query.source].append(f"{label}:source")
        roles[query.target].append(f"{label}:target")

    def ranks(signatures: Mapping[str, Any]) -> dict[str, int]:
        unique = {
            value: index
            for index, value in enumerate(sorted(set(signatures.values())))
        }
        return {handle: unique[value] for handle, value in signatures.items()}

    colors = ranks(
        {
            handle: (
                len(predecessors[handle]),
                len(successors[handle]),
                tuple(sorted(roles[handle])),
            )
            for handle in component.handles
        }
    )
    for _ in range(len(component.handles)):
        refined = ranks(
            {
                handle: (
                    colors[handle],
                    tuple(sorted(colors[value] for value in predecessors[handle])),
                    tuple(sorted(colors[value] for value in successors[handle])),
                )
                for handle in component.handles
            }
        )
        same_partition = all(
            (colors[left] == colors[right]) == (refined[left] == refined[right])
            for left in component.handles
            for right in component.handles
        )
        colors = refined
        if same_partition:
            break

    query_to_label = {
        int(query_index): label for label, query_index in target.choice_to_query
    }
    trace = [
        UNDECIDED if value == UNDECIDED else query_to_label[int(value)]
        for value in target.trace
    ]
    choices = []
    for label, query_index in sorted(target.choice_to_query):
        query = component.queries[query_index]
        choices.append((label, colors[query.source], colors[query.target]))
    return {
        "family": "relation",
        "handle_colors": sorted(colors.values()),
        "edges": sorted((colors[source], colors[target]) for source, target in component.edges),
        "choices": choices,
        "answer": target.label,
        "trace": trace,
    }


def _numeric_example_semantic_value(
    component: NumericComponent,
    target: NumericTarget,
) -> dict[str, Any]:
    by_handle = {
        handle: tuple(values)
        for handle, values in zip(component.handles, component.deltas, strict=True)
    }
    return {
        "family": "numeric",
        "choices": sorted(
            (label, by_handle[handle])
            for label, handle in target.choice_to_handle
        ),
        "prefix_labels": list(target.prefix_labels),
        "answer": target.final_label,
    }


def semantic_fingerprint(
    component: NumericComponent | RelationComponent,
    target: NumericTarget | RelationTarget | None = None,
) -> str:
    if target is None:
        return _sha(_semantic_value(component))
    if isinstance(component, NumericComponent) and isinstance(target, NumericTarget):
        return _sha(_numeric_example_semantic_value(component, target))
    if isinstance(component, RelationComponent) and isinstance(target, RelationTarget):
        return _sha(_relation_example_semantic_value(component, target))
    raise TypeError("H1 semantic fingerprint component/target family mismatch")


def source_fingerprint(record: SourceRecord | str) -> str:
    return _sha(record.source_text if isinstance(record, SourceRecord) else str(record))


def cross_split_overlap_audit(bundle: FreshBundle) -> dict[str, Any]:
    semantic: dict[str, list[str]] = {}
    surface: dict[str, list[str]] = {}
    split_by_id = {record_id: split for split, ids in bundle.splits.items() for record_id in ids}
    for record_id, fingerprint in bundle.semantic_fingerprints.items():
        semantic.setdefault(fingerprint, []).append(record_id)
    for record_id, fingerprint in bundle.source_fingerprints.items():
        surface.setdefault(fingerprint, []).append(record_id)
    def cross(mapping: Mapping[str, list[str]]) -> list[tuple[str, tuple[str, ...]]]:
        return [(fingerprint, tuple(ids)) for fingerprint, ids in mapping.items() if len({split_by_id[i] for i in ids}) > 1]
    return {"semantic_overlap": cross(semantic), "source_overlap": cross(surface), "passed": not cross(semantic) and not cross(surface)}


def _handle(rng: random.Random) -> str:
    alphabet = "bcdfghjklmnpqrstvwxyz"
    return "h" + "".join(rng.choice(alphabet) for _ in range(7))


def _ranges(split: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if split == "heldout":
        return (7, 8), (6, 12), (9, 10)
    if split == "supported":
        return (4, 7), (1, 7), (6, 9)
    return (4, 6), (1, 5), (6, 8)


def _numeric_component(rng: random.Random, split: str) -> NumericComponent:
    (lo_h, hi_h), (lo_m, hi_m), _ = _ranges(split)
    horizon = rng.randint(lo_h, hi_h)
    while True:
        handles = tuple(_handle(rng) for _ in range(4))
        deltas = tuple(tuple(rng.choice(tuple(range(-hi_m, 0)) + tuple(range(1, hi_m + 1))) for _ in range(horizon)) for _ in range(4))
        component = NumericComponent(handles, deltas)
        try:
            numeric_oracle(component)
            return component
        except ValueError:
            continue


def _relation_component(rng: random.Random, split: str) -> RelationComponent:
    _, _, (lo_n, hi_n) = _ranges(split)
    for _ in range(5000):
        count = rng.randint(lo_n, hi_n)
        handles = tuple(_handle(rng) for _ in range(count))
        # A guaranteed branch/merge backbone; extras add alternate routes.
        edges: set[tuple[str, str]] = {
            (handles[0], handles[1]), (handles[0], handles[2]),
            (handles[1], handles[3]), (handles[2], handles[3]),
        }
        if split == "heldout":
            for i in range(3, min(count - 1, 6)):
                edges.add((handles[i], handles[i + 1]))
        else:
            for i in range(4, count - 1):
                if rng.random() < 0.45:
                    edges.add((handles[i], handles[i + 1]))
        for i in range(count):
            for j in range(i + 1, count):
                # Keep one guaranteed length-two path without its direct edge
                # so transitive-redundancy materialization is total.
                if i == 0 and j == 3:
                    continue
                if rng.random() < 0.10:
                    edges.add((handles[i], handles[j]))
        edges_tuple = tuple(sorted(edges))
        if _has_cycle(handles, edges_tuple):
            continue
        reach = _reachable_pairs(handles, edges_tuple)
        if not reach:
            continue
        reachable_pair = rng.choice(sorted(reach))
        unreachable = [(a, b) for a in handles for b in handles if a != b and (a, b) not in reach]
        if len(unreachable) < 3:
            continue
        queries = [RelationQuery(*reachable_pair)] + [RelationQuery(*pair) for pair in rng.sample(unreachable, 3)]
        rng.shuffle(queries)
        component = RelationComponent(handles, edges_tuple, tuple(queries))
        if split == "heldout" and _longest_path(component) < 4:
            continue
        if split != "heldout" and _longest_path(component) > 3:
            continue
        return component
    raise RuntimeError(f"could not generate relation component for {split}")


def _reachable_pairs(handles: Sequence[str], edges: Sequence[tuple[str, str]]) -> set[tuple[str, str]]:
    adjacency = {h: [] for h in handles}
    for source, target in edges:
        adjacency[source].append(target)
    output: set[tuple[str, str]] = set()
    for source in handles:
        todo = list(adjacency[source])
        seen = set(todo)
        while todo:
            current = todo.pop()
            output.add((source, current))
            for target in adjacency[current]:
                if target not in seen:
                    seen.add(target)
                    todo.append(target)
    return output


def _longest_path(component: RelationComponent) -> int:
    adjacency = {h: [] for h in component.handles}
    for source, target in component.edges:
        adjacency[source].append(target)
    def depth(h: str, seen: set[str]) -> int:
        return 1 + max((depth(t, seen | {t}) for t in adjacency[h] if t not in seen), default=0)
    return max(depth(h, {h}) for h in component.handles)


def _choice_map(rng: random.Random, handles: Sequence[str]) -> tuple[tuple[str, str], ...]:
    shuffled = list(handles)
    rng.shuffle(shuffled)
    return tuple(zip(LABELS, shuffled))


def _render_numeric(record_id: str, component: NumericComponent, choices: tuple[tuple[str, str], ...], template: int) -> SourceRecord:
    deltas = "; ".join(f"{h}: " + ", ".join(f"{v:+d}" for v in values) for h, values in zip(component.handles, component.deltas))
    lines = [
        _TEMPLATES[template % 4].split(" {instruction}")[0],
        "At each point choose the option with the smallest running total.",
        "Changes follow.",
        *(f"{h}: " + ", ".join(f"{v:+d}" for v in values) for h, values in zip(component.handles, component.deltas)),
        "Options follow.",
        *(f"{label}: {handle}" for label, handle in choices),
    ]
    return SourceRecord(record_id, "\n".join(lines), tuple(lines))


def _render_relation(record_id: str, component: RelationComponent, choices: tuple[tuple[str, int], ...], template: int) -> SourceRecord:
    lines = [
        _TEMPLATES[template % 4].split(" {instruction}")[0],
        "For each pair choose the option when a path exists from the first handle to the second.",
        "Links follow.",
        *(f"{a}->{b}" for a, b in component.edges),
        "Pairs follow.",
        *(f"{label}: ({component.queries[i].source}, {component.queries[i].target})" for label, i in choices),
    ]
    return SourceRecord(record_id, "\n".join(lines), tuple(lines))


def _make_numeric_target(component: NumericComponent, rng: random.Random) -> NumericTarget:
    choices = _choice_map(rng, component.handles)
    by_handle = {handle: label for label, handle in choices}
    oracle = numeric_oracle(component)
    labels = tuple(UNDECIDED if winner == UNDECIDED else by_handle[winner] for winner in oracle.prefix_winners)
    return NumericTarget(labels, by_handle[oracle.final_winner], choices)


def _make_relation_target(component: RelationComponent, rng: random.Random) -> RelationTarget:
    order = list(range(4))
    rng.shuffle(order)
    choices = tuple((LABELS[i], query_index) for i, query_index in enumerate(order))
    by_query = {query_index: label for label, query_index in choices}
    oracle = relation_oracle(component)
    return RelationTarget(by_query[oracle.query_index], oracle.query_index, relation_fixed_point_trace(component), choices)


def _count(counts: Any, split: str, kind: str) -> int:
    if counts is None:
        return 8
    if isinstance(counts, int):
        return counts
    if kind in counts:
        value = counts[kind]
        if isinstance(value, Mapping):
            return int(value.get(split, 0))
        if isinstance(value, int):
            return value
    if split in counts:
        return int(counts[split])
    return 0


def generate_bundle(
    seed: int,
    counts: Mapping[str, Any] | int | None = None,
    *,
    forbidden_semantic_fingerprints: Iterable[str] = (),
    forbidden_source_fingerprints: Iterable[str] = (),
) -> FreshBundle:
    """Generate deterministic fresh components for all five H1 splits."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    records: list[SourceRecord] = []
    targets: dict[str, NumericTarget | RelationTarget] = {}
    components: dict[str, NumericComponent | RelationComponent] = {}
    split_ids: dict[str, list[str]] = {split: [] for split in SPLITS}
    semantic_seen = {str(value) for value in forbidden_semantic_fingerprints}
    source_seen = {str(value) for value in forbidden_source_fingerprints}
    serial = 0
    for split_index, split in enumerate(SPLITS):
        for kind in ("numeric", "relation"):
            requested = _count(counts, split, kind)
            if requested < 0:
                raise ValueError("counts cannot be negative")
            for item in range(requested):
                for attempt in range(10000):
                    rng = random.Random(seed ^ (split_index * 0x9E3779B1) ^ (item * 0x85EBCA77) ^ (attempt * 0xC2B2AE3D) ^ (0x11 if kind == "numeric" else 0x22))
                    component = _numeric_component(rng, split) if kind == "numeric" else _relation_component(rng, split)
                    seed_tag = hashlib.sha256(str(seed).encode("ascii")).hexdigest()[:8]
                    record_id = f"h1{seed_tag}{serial:06d}"
                    if kind == "numeric":
                        target = _make_numeric_target(component, rng)
                        record = _render_numeric(record_id, component, target.choice_to_handle, item % (3 if split != "heldout" else 1) if split != "heldout" else 3)
                    else:
                        target = _make_relation_target(component, rng)
                        record = _render_relation(record_id, component, target.choice_to_query, item % (3 if split != "heldout" else 1) if split != "heldout" else 3)
                    semantic = semantic_fingerprint(component, target)
                    surface = source_fingerprint(record)
                    if semantic in semantic_seen or surface in source_seen:
                        continue
                    break
                else:
                    raise RuntimeError(f"could not make unique {split}/{kind} record")
                serial += 1
                semantic_seen.add(semantic)
                source_seen.add(surface)
                records.append(record)
                targets[record_id] = target
                components[record_id] = component
                split_ids[split].append(record_id)
    semantics = {
        record.id: semantic_fingerprint(
            components[record.id],
            targets[record.id],
        )
        for record in records
    }
    sources = {record.id: source_fingerprint(record) for record in records}
    bundle = FreshBundle(tuple(records), targets, components, {k: tuple(v) for k, v in split_ids.items()}, semantics, sources)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: FreshBundle) -> None:
    if not isinstance(bundle, FreshBundle):
        raise TypeError("expected FreshBundle")
    if forbidden_source_scan(bundle):
        raise ValueError("forbidden source field detected")
    if not cross_split_overlap_audit(bundle)["passed"]:
        raise ValueError("cross-split overlap detected")
    for record in bundle.records:
        component = bundle.components[record.id]
        target = bundle.targets[record.id]
        if isinstance(component, NumericComponent):
            if not isinstance(target, NumericTarget):
                raise ValueError("numeric record has wrong target type")
            oracle = numeric_oracle(component)
            mapping = dict(target.choice_to_handle)
            expected = next(label for label, handle in mapping.items() if handle == oracle.final_winner)
            if target.final_label != expected:
                raise ValueError("numeric target final label mismatch")
        else:
            if not isinstance(target, RelationTarget):
                raise ValueError("relation record has wrong target type")
            oracle = relation_oracle(component)
            if target.query_index != oracle.query_index or target.label not in LABELS:
                raise ValueError("relation target mismatch")


def rename_handles(component: NumericComponent | RelationComponent, mapping: Mapping[str, str]) -> NumericComponent | RelationComponent:
    if not isinstance(component, (NumericComponent, RelationComponent)):
        raise TypeError("expected a fresh component")
    return component.renamed(mapping)


def permute_clauses(component: NumericComponent, order: Sequence[int]) -> NumericComponent:
    if not isinstance(component, NumericComponent):
        raise TypeError("permute_clauses requires NumericComponent")
    if tuple(sorted(order)) != tuple(range(4)):
        raise ValueError("order must be a permutation of four clauses")
    return NumericComponent(tuple(component.handles[i] for i in order), tuple(component.deltas[i] for i in order))


def add_numeric_cancelling_pair(component: NumericComponent, amount: int = 1) -> NumericComponent:
    if not isinstance(component, NumericComponent) or type(amount) is not int or amount == 0:
        raise ValueError("amount must be a non-zero integer")
    if len(component.deltas[0]) >= 8:
        raise ValueError("cannot extend an eight-step component")
    return NumericComponent(component.handles, tuple(values + (amount, -amount) for values in component.deltas))


def add_relation_transitive_redundancy(component: RelationComponent) -> RelationComponent:
    if not isinstance(component, RelationComponent):
        raise TypeError("expected RelationComponent")
    reach = _reachable_pairs(component.handles, component.edges)
    direct = set(component.edges)
    for edge in sorted(reach - direct):
        return RelationComponent(component.handles, tuple(sorted((*component.edges, edge))), component.queries)
    raise ValueError("graph has no transitive edge available")


def paraphrase_source(record: SourceRecord, variant: int = 1) -> SourceRecord:
    if not isinstance(record, SourceRecord):
        raise TypeError("expected SourceRecord")
    text = record.source_text
    if variant % 2 == 0:
        for old in ("Review these entries", "Study the listed entries", "Read the entries", "Inspect these entries"):
            if old in text:
                text = text.replace(old, "Consider these entries", 1)
                break
    else:
        for old in ("At each point choose", "For each pair choose"):
            if old in text:
                text = text.replace(old, "At every point choose" if old.startswith("At") else "For every pair choose", 1)
                break
    if text == record.source_text:
        text = text.replace("Options:", "Choices:", 1)
    return SourceRecord(record.id, text, tuple(text.splitlines()))


def metamorphic_transforms(component: NumericComponent | RelationComponent, record: SourceRecord | None = None) -> dict[str, Any]:
    mapping = {handle: f"h{chr(97 + i)}xxxxxx" for i, handle in enumerate(component.handles)}
    result: dict[str, Any] = {"handle_rename": rename_handles(component, mapping)}
    if isinstance(component, NumericComponent):
        result["clause_permutation"] = permute_clauses(component, tuple(reversed(range(4))))
        result["cancelling_pair"] = add_numeric_cancelling_pair(component)
    else:
        result["transitive_redundancy"] = add_relation_transitive_redundancy(component)
    if record is not None:
        result["surface_paraphrase"] = paraphrase_source(record)
    return result


def materialize_metamorphic(
    component: NumericComponent | RelationComponent,
    target: NumericTarget | RelationTarget,
    record: SourceRecord | None = None,
    transform: str = "handle_rename",
) -> dict[str, Any]:
    """Build an evaluable transformed example, including its independent target.

    This is intentionally a data-side operation.  It never changes a source
    record in place and never places target fields into the returned record.
    """
    if isinstance(component, NumericComponent) and not isinstance(target, NumericTarget):
        raise TypeError("numeric component requires NumericTarget")
    if isinstance(component, RelationComponent) and not isinstance(target, RelationTarget):
        raise TypeError("relation component requires RelationTarget")
    new_component: NumericComponent | RelationComponent = component
    new_target: NumericTarget | RelationTarget = target
    if transform == "handle_rename":
        mapping = {handle: f"h{chr(97 + i)}xxxxxx" for i, handle in enumerate(component.handles)}
        new_component = rename_handles(component, mapping)
        if isinstance(target, NumericTarget):
            new_target = NumericTarget(target.prefix_labels, target.final_label, tuple((label, mapping[handle]) for label, handle in target.choice_to_handle))
        else:
            new_target = target
    elif transform == "clause_permutation":
        if not isinstance(component, NumericComponent):
            raise TypeError("clause permutation requires NumericComponent")
        new_component = permute_clauses(component, tuple(reversed(range(4))))
    elif transform == "choice_permutation":
        if isinstance(component, NumericComponent):
            old = dict(target.choice_to_handle)  # type: ignore[union-attr]
            order = (1, 0, 3, 2)
            choices = tuple((LABELS[i], old[LABELS[order[i]]]) for i in range(4))
            by_handle = {handle: label for label, handle in choices}
            oracle = numeric_oracle(component)
            new_target = NumericTarget(tuple(UNDECIDED if item == UNDECIDED else by_handle[item] for item in oracle.prefix_winners), by_handle[oracle.final_winner], choices)
        else:
            old = dict(target.choice_to_query)  # type: ignore[union-attr]
            order = (1, 0, 3, 2)
            choices = tuple((LABELS[i], old[LABELS[order[i]]]) for i in range(4))
            by_query = {query: label for label, query in choices}
            oracle = relation_oracle(component)
            new_target = RelationTarget(by_query[oracle.query_index], oracle.query_index, relation_fixed_point_trace(component), choices)
    elif transform == "query_permutation":
        if not isinstance(component, RelationComponent):
            raise TypeError("query permutation requires RelationComponent")
        order = (1, 0, 3, 2)
        new_component = RelationComponent(component.handles, component.edges, tuple(component.queries[i] for i in order))
        old_to_new = {old: new for new, old in enumerate(order)}
        choices = tuple((label, old_to_new[query]) for label, query in target.choice_to_query)  # type: ignore[union-attr]
        new_target = RelationTarget(target.label, old_to_new[target.query_index], tuple(old_to_new[i] if i != UNDECIDED else UNDECIDED for i in relation_fixed_point_trace(component)), choices)  # type: ignore[union-attr]
    elif transform == "cancelling_pair":
        if not isinstance(component, NumericComponent):
            raise TypeError("cancelling pair requires NumericComponent")
        new_component = add_numeric_cancelling_pair(component)
        choices = target.choice_to_handle  # type: ignore[union-attr]
        by_handle = {handle: label for label, handle in choices}
        oracle = numeric_oracle(new_component)
        new_target = NumericTarget(tuple(UNDECIDED if item == UNDECIDED else by_handle[item] for item in oracle.prefix_winners), by_handle[oracle.final_winner], choices)
    elif transform == "transitive_redundancy":
        if not isinstance(component, RelationComponent):
            raise TypeError("transitive redundancy requires RelationComponent")
        new_component = add_relation_transitive_redundancy(component)
        new_target = RelationTarget(target.label, target.query_index, relation_fixed_point_trace(new_component), target.choice_to_query)  # type: ignore[union-attr]
    elif transform == "surface_paraphrase":
        if record is None:
            raise ValueError("surface paraphrase needs a source record")
    else:
        raise ValueError(f"unknown metamorphic transform: {transform}")
    if record is not None:
        if isinstance(new_component, NumericComponent):
            choices = new_target.choice_to_handle  # type: ignore[union-attr]
            new_record = _render_numeric(record.id, new_component, choices, 0)
        else:
            choices = new_target.choice_to_query  # type: ignore[union-attr]
            new_record = _render_relation(record.id, new_component, choices, 0)
        if transform == "surface_paraphrase":
            new_record = paraphrase_source(record)
        opaque_suffix = hashlib.sha256(transform.encode("utf-8")).hexdigest()[:8]
        new_record = SourceRecord(
            f"{record.id}m{opaque_suffix}",
            new_record.source_text,
            new_record.source_atoms,
        )
    else:
        new_record = None
    return {"component": new_component, "target": new_target, "record": new_record, "transform": transform}


# Short aliases used by audit/runner code; the descriptive names above remain
# the canonical public API.
relation_trace = relation_fixed_point_trace
fixed_point_trace = relation_fixed_point_trace
overlap_audit = cross_split_overlap_audit
source_forbidden_scan = forbidden_source_scan
FreshDataBundle = FreshBundle


__all__ = [
    "SCHEMA_VERSION", "SPLITS", "LABELS", "UNDECIDED", "NumericComponent", "RelationQuery",
    "RelationComponent", "SourceRecord", "NumericOracleResult", "RelationOracleResult",
    "NumericTarget", "RelationTarget", "FreshBundle", "generate_bundle", "numeric_oracle",
    "relation_oracle", "relation_fixed_point_trace", "semantic_fingerprint", "source_fingerprint",
    "cross_split_overlap_audit", "forbidden_source_scan", "validate_bundle", "rename_handles",
    "permute_clauses", "add_numeric_cancelling_pair", "add_relation_transitive_redundancy",
    "paraphrase_source", "metamorphic_transforms", "materialize_metamorphic", "relation_trace",
    "fixed_point_trace", "overlap_audit", "source_forbidden_scan", "FreshDataBundle",
]
