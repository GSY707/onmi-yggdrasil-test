from __future__ import annotations

from dataclasses import dataclass, replace

from .schema import NumericCase, NumericState, RelationCase, RelationState


@dataclass(frozen=True, slots=True)
class NumericFaultTarget:
    primary_handle: str
    step_index: int | None = None
    secondary_handle: str | None = None


@dataclass(frozen=True, slots=True)
class RelationFaultTarget:
    edge: tuple[str, str]
    alias_from: str | None = None
    alias_to: str | None = None
    assignments: tuple[tuple[tuple[str, str], bool], ...] = ()


def _numeric_source(case: NumericCase, handle: str):
    return next(candidate for candidate in case.candidates if candidate.handle == handle)


def _numeric_observed(state: NumericState, handle: str):
    return next(candidate for candidate in state.candidates if candidate.handle == handle)


def _replace_numeric(state: NumericState, handle: str, replacement) -> NumericState:
    return replace(
        state,
        candidates=tuple(
            replacement if candidate.handle == handle else candidate
            for candidate in state.candidates
        ),
    )


def numeric_sign_flip(
    case: NumericCase, state: NumericState, target: NumericFaultTarget
) -> NumericState:
    source = _numeric_source(case, target.primary_handle)
    observed = _numeric_observed(state, target.primary_handle)
    index = target.step_index
    if index is None or not 0 <= index < len(source.deltas):
        raise ValueError("numeric sign-flip target has invalid step")
    adjustment = -2 * source.deltas[index]
    prefixes = tuple(
        value if prefix_index < index else value + adjustment
        for prefix_index, value in enumerate(observed.prefixes)
    )
    return _replace_numeric(
        state,
        target.primary_handle,
        replace(observed, prefixes=prefixes, final=prefixes[-1]),
    )


def numeric_drop_last(
    case: NumericCase, state: NumericState, target: NumericFaultTarget
) -> NumericState:
    source = _numeric_source(case, target.primary_handle)
    observed = _numeric_observed(state, target.primary_handle)
    if target.step_index != len(source.deltas) - 1:
        raise ValueError("numeric drop-last target must identify the final update")
    dropped = observed.prefixes[-2]
    prefixes = (*observed.prefixes[:-1], dropped)
    return _replace_numeric(
        state,
        target.primary_handle,
        replace(observed, prefixes=prefixes, final=dropped),
    )


def numeric_final_off_by_one(
    _case: NumericCase, state: NumericState, target: NumericFaultTarget
) -> NumericState:
    observed = _numeric_observed(state, target.primary_handle)
    prefixes = (*observed.prefixes[:-1], observed.prefixes[-1] + 1)
    return _replace_numeric(
        state,
        target.primary_handle,
        replace(observed, prefixes=prefixes, final=prefixes[-1]),
    )


def numeric_constant_collapse(
    _case: NumericCase, state: NumericState, _target: NumericFaultTarget
) -> NumericState:
    candidates = tuple(
        replace(
            candidate,
            prefixes=tuple(0 for _ in candidate.prefixes),
            final=0,
        )
        for candidate in state.candidates
    )
    return replace(state, candidates=candidates)


def numeric_winner_swap(
    _case: NumericCase, state: NumericState, target: NumericFaultTarget
) -> NumericState:
    if target.secondary_handle is None:
        raise ValueError("numeric winner-swap target requires a runner")
    winner = _numeric_observed(state, target.primary_handle)
    runner = _numeric_observed(state, target.secondary_handle)
    candidates = []
    for candidate in state.candidates:
        if candidate.handle == target.primary_handle:
            candidates.append(
                replace(candidate, prefixes=runner.prefixes, final=runner.final)
            )
        elif candidate.handle == target.secondary_handle:
            candidates.append(
                replace(candidate, prefixes=winner.prefixes, final=winner.final)
            )
        else:
            candidates.append(candidate)
    return replace(state, candidates=tuple(candidates))


def numeric_position_shortcut(
    _case: NumericCase, state: NumericState, _target: NumericFaultTarget
) -> NumericState:
    candidates = []
    for index, candidate in enumerate(state.candidates):
        candidates.append(
            replace(
                candidate,
                prefixes=tuple(index for _ in candidate.prefixes),
                final=index,
            )
        )
    return replace(state, candidates=tuple(candidates))


def relation_drop_bridge(
    _case: RelationCase, state: RelationState, target: RelationFaultTarget
) -> RelationState:
    return replace(
        state,
        closure=tuple(edge for edge in state.closure if edge != target.edge),
    )


def relation_reverse_pollution(
    _case: RelationCase, state: RelationState, target: RelationFaultTarget
) -> RelationState:
    closure = set(state.closure)
    closure.add(target.edge)
    return replace(state, closure=tuple(sorted(closure)))


def relation_spurious(
    _case: RelationCase, state: RelationState, target: RelationFaultTarget
) -> RelationState:
    closure = set(state.closure)
    closure.add(target.edge)
    return replace(state, closure=tuple(sorted(closure)))


def relation_handle_alias(
    _case: RelationCase, state: RelationState, target: RelationFaultTarget
) -> RelationState:
    if target.alias_from is None or target.alias_to is None:
        raise ValueError("relation alias target is incomplete")
    closure = tuple(
        (
            target.alias_to if source == target.alias_from else source,
            target.alias_to if destination == target.alias_from else destination,
        )
        for source, destination in state.closure
    )
    # RelationState owns the known-handle set, so an unknown alias must fail closed.
    return replace(state, closure=closure)


def relation_query_shortcut(
    _case: RelationCase, state: RelationState, target: RelationFaultTarget
) -> RelationState:
    closure = set(state.closure)
    for edge, predicted in target.assignments:
        if predicted:
            closure.add(edge)
        else:
            closure.discard(edge)
    return replace(state, closure=tuple(sorted(closure)))
