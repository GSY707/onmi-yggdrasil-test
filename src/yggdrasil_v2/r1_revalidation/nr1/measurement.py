from __future__ import annotations

from typing import Any

from .schema import (
    NumericCandidateState,
    NumericCase,
    NumericState,
    RelationCase,
    RelationQuery,
    RelationState,
)


def build_numeric_state(case: NumericCase) -> NumericState:
    """Build typed scalar state by iterative updates; consume no golden data."""
    states: list[NumericCandidateState] = []
    for candidate in case.candidates:
        running = 0
        prefixes: list[int] = []
        for delta in candidate.deltas:
            running += delta
            prefixes.append(running)
        states.append(
            NumericCandidateState(
                handle=candidate.handle,
                prefixes=tuple(prefixes),
                final=running,
            )
        )
    return NumericState(case_id=case.case_id, candidates=tuple(states))


def compare_scalars(left: int, right: int) -> int:
    return (left > right) - (left < right)


def numeric_decision(state: NumericState) -> str:
    ordered = sorted(
        ((candidate.final, candidate.handle) for candidate in state.candidates),
        key=lambda item: (item[0], item[1]),
    )
    if len(ordered) < 2 or ordered[0][0] == ordered[1][0]:
        raise ValueError("numeric decision requires a unique optimum")
    return ordered[0][1]


def audit_numeric_state(case: NumericCase, state: NumericState) -> dict[str, Any]:
    by_handle = {candidate.handle: candidate for candidate in state.candidates}
    identity_exact = state.case_id == case.case_id
    handle_exact = set(by_handle) == {candidate.handle for candidate in case.candidates}
    step_total = 0
    step_good = 0
    final_good = 0
    for candidate in case.candidates:
        observed = by_handle.get(candidate.handle)
        if observed is None:
            step_total += len(candidate.deltas)
            continue
        running = 0
        for index, delta in enumerate(candidate.deltas):
            running += delta
            step_total += 1
            if index < len(observed.prefixes) and observed.prefixes[index] == running:
                step_good += 1
        final_good += int(observed.final == running)
    pair_total = 0
    antisymmetric = 0
    finals = [candidate.final for candidate in state.candidates]
    for left_index, left in enumerate(finals):
        for right in finals[left_index + 1 :]:
            pair_total += 1
            antisymmetric += int(
                compare_scalars(left, right) == -compare_scalars(right, left)
            )
    return {
        "identity_exact": identity_exact,
        "handle_exact": handle_exact,
        "step_exact": step_good,
        "step_total": step_total,
        "final_exact": final_good,
        "final_total": len(case.candidates),
        "comparison_antisymmetric": antisymmetric,
        "comparison_total": pair_total,
        "passed": (
            identity_exact
            and handle_exact
            and step_good == step_total
            and final_good == len(case.candidates)
            and antisymmetric == pair_total
        ),
    }


def build_relation_state(case: RelationCase) -> RelationState:
    """Build reachability by fixed-point composition, not the BFS reference path."""
    closure = set(case.edges)
    changed = True
    while changed:
        changed = False
        additions = {
            (left_source, right_target)
            for left_source, left_target in closure
            for right_source, right_target in closure
            if left_target == right_source and left_source != right_target
        }
        additions.difference_update(closure)
        if additions:
            closure.update(additions)
            changed = True
    return RelationState(
        case_id=case.case_id,
        handles=case.handles,
        closure=tuple(sorted(closure)),
    )


def relation_decisions(
    state: RelationState, queries: tuple[RelationQuery, ...]
) -> tuple[bool, ...]:
    known = set(state.handles)
    for query in queries:
        if query.source not in known or query.target not in known:
            raise ValueError("relation decision query uses unknown state handle")
    closure = set(state.closure)
    return tuple((query.source, query.target) in closure for query in queries)


def audit_relation_state(case: RelationCase, state: RelationState) -> dict[str, Any]:
    closure = set(state.closure)
    identity_exact = state.case_id == case.case_id
    handle_exact = tuple(state.handles) == tuple(case.handles)
    direct_good = sum(edge in closure for edge in case.edges)
    composition_total = 0
    composition_good = 0
    for left_source, left_target in closure:
        for right_source, right_target in closure:
            if left_target == right_source and left_source != right_target:
                composition_total += 1
                composition_good += int((left_source, right_target) in closure)
    known = set(case.handles)
    known_good = sum(source in known and target in known for source, target in closure)
    irreflexive = sum(source != target for source, target in closure)
    antisymmetric = sum((target, source) not in closure for source, target in closure)
    return {
        "identity_exact": identity_exact,
        "handle_exact": handle_exact,
        "direct_exact": direct_good,
        "direct_total": len(case.edges),
        "composition_exact": composition_good,
        "composition_total": composition_total,
        "known_handle_exact": known_good,
        "closure_total": len(closure),
        "irreflexive_exact": irreflexive,
        "antisymmetric_exact": antisymmetric,
        "passed": (
            identity_exact
            and handle_exact
            and direct_good == len(case.edges)
            and composition_good == composition_total
            and known_good == len(closure)
            and irreflexive == len(closure)
            and antisymmetric == len(closure)
        ),
    }
