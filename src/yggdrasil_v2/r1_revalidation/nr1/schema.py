from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "yggdrasil.v2-r1r.p1-nr1.case-state.v2"
_HANDLE = re.compile(r"^[a-z][a-z0-9-]{5,31}$")


def _validate_handle(value: str) -> None:
    if not isinstance(value, str) or _HANDLE.fullmatch(value) is None:
        raise ValueError(f"invalid opaque handle: {value!r}")


def _validate_dag(
    handles: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> None:
    adjacency = {handle: set() for handle in handles}
    indegree = {handle: 0 for handle in handles}
    for source, target in edges:
        adjacency[source].add(target)
        indegree[target] += 1
    ready = sorted(handle for handle, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        current = ready.pop(0)
        visited += 1
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if visited != len(handles):
        raise ValueError("relation edges must form a directed acyclic graph")


@dataclass(frozen=True, slots=True)
class NumericCandidate:
    handle: str
    deltas: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_handle(self.handle)
        if not 2 <= len(self.deltas) <= 12:
            raise ValueError("numeric trajectory must contain 2..12 deltas")
        if any(type(value) is not int or value == 0 for value in self.deltas):
            raise ValueError("numeric deltas must be non-zero integers")


@dataclass(frozen=True, slots=True)
class NumericCase:
    case_id: str
    split: str
    candidates: tuple[NumericCandidate, ...]

    def __post_init__(self) -> None:
        if not self.case_id or self.split not in {"qualification", "heldout", "fixture"}:
            raise ValueError("invalid numeric case identity or split")
        if not 2 <= len(self.candidates) <= 9:
            raise ValueError("numeric case must contain 2..9 candidates")
        handles = [candidate.handle for candidate in self.candidates]
        if len(set(handles)) != len(handles):
            raise ValueError("numeric candidate handles must be unique")
        horizons = {len(candidate.deltas) for candidate in self.candidates}
        if len(horizons) != 1:
            raise ValueError("all numeric candidates must share one horizon")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RelationQuery:
    source: str
    target: str

    def __post_init__(self) -> None:
        _validate_handle(self.source)
        _validate_handle(self.target)
        if self.source == self.target:
            raise ValueError("relation query must use distinct handles")


@dataclass(frozen=True, slots=True)
class RelationCase:
    case_id: str
    split: str
    handles: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    queries: tuple[RelationQuery, ...]

    def __post_init__(self) -> None:
        if not self.case_id or self.split not in {"qualification", "heldout", "fixture"}:
            raise ValueError("invalid relation case identity or split")
        if not 4 <= len(self.handles) <= 16:
            raise ValueError("relation case must contain 4..16 handles")
        for handle in self.handles:
            _validate_handle(handle)
        known = set(self.handles)
        if len(known) != len(self.handles):
            raise ValueError("relation handles must be unique")
        if not self.edges or len(set(self.edges)) != len(self.edges):
            raise ValueError("relation edges must be non-empty and unique")
        for source, target in self.edges:
            if source not in known or target not in known or source == target:
                raise ValueError("relation edge uses unknown or self handle")
        _validate_dag(self.handles, self.edges)
        if not self.queries or len(set(self.queries)) != len(self.queries):
            raise ValueError("relation queries must be non-empty and unique")
        for query in self.queries:
            if query.source not in known or query.target not in known:
                raise ValueError("relation query uses unknown handle")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NumericCandidateState:
    handle: str
    prefixes: tuple[int, ...]
    final: int

    def __post_init__(self) -> None:
        _validate_handle(self.handle)
        if not self.prefixes or any(type(value) is not int for value in self.prefixes):
            raise ValueError("numeric state prefixes must be non-empty integers")
        if type(self.final) is not int or self.final != self.prefixes[-1]:
            raise ValueError("numeric final must equal the final prefix")


@dataclass(frozen=True, slots=True)
class NumericState:
    case_id: str
    candidates: tuple[NumericCandidateState, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.candidates:
            raise ValueError("invalid numeric state")
        handles = [candidate.handle for candidate in self.candidates]
        if len(handles) != len(set(handles)):
            raise ValueError("numeric state handles must be unique")


@dataclass(frozen=True, slots=True)
class RelationState:
    case_id: str
    handles: tuple[str, ...]
    closure: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.handles:
            raise ValueError("invalid relation state identity or handles")
        for handle in self.handles:
            _validate_handle(handle)
        known = set(self.handles)
        if len(known) != len(self.handles):
            raise ValueError("relation state handles must be unique")
        if len(self.closure) != len(set(self.closure)):
            raise ValueError("relation state closure must be unique")
        for source, target in self.closure:
            _validate_handle(source)
            _validate_handle(target)
            if source not in known or target not in known:
                raise ValueError("relation closure uses unknown handle")
            if source == target:
                raise ValueError("relation closure must be irreflexive")
