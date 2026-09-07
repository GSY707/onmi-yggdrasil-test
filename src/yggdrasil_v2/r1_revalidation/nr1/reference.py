from __future__ import annotations

from collections import deque
from itertools import accumulate
from typing import Any

from .schema import NumericCase, RelationCase


def numeric_oracle(case: NumericCase) -> dict[str, Any]:
    """Independent reference: itertools.accumulate plus exact minimum."""
    prefixes = {
        candidate.handle: tuple(accumulate(candidate.deltas))
        for candidate in case.candidates
    }
    totals = {handle: values[-1] for handle, values in prefixes.items()}
    ordered = sorted(totals.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) < 2 or ordered[0][1] == ordered[1][1]:
        raise ValueError("numeric oracle requires a unique optimum")
    return {
        "prefixes": prefixes,
        "totals": totals,
        "winner": ordered[0][0],
        "ordered_handles": tuple(handle for handle, _ in ordered),
    }


def _reachable(adjacency: dict[str, set[str]], source: str, target: str) -> bool:
    queue: deque[str] = deque([source])
    visited = {source}
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def relation_oracle(case: RelationCase) -> dict[str, Any]:
    """Independent reference: one BFS per source/query, not fixed-point closure."""
    adjacency = {handle: set() for handle in case.handles}
    for source, target in case.edges:
        adjacency[source].add(target)
    closure = {
        (source, target)
        for source in case.handles
        for target in case.handles
        if source != target and _reachable(adjacency, source, target)
    }
    decisions = tuple(
        _reachable(adjacency, query.source, query.target) for query in case.queries
    )
    return {"closure": closure, "decisions": decisions}
