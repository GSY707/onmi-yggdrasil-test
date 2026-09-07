from __future__ import annotations

from yggdrasil_v2.r1_revalidation.nr1.measurement import (
    audit_numeric_state,
    build_numeric_state,
    numeric_decision,
)
from yggdrasil_v2.r1_revalidation.nr1.qualification import (
    FORMAL_SEED,
    build_numeric_cases,
    evaluate_numeric,
    evaluate_numeric_metamorphic,
)
from yggdrasil_v2.r1_revalidation.nr1.reference import numeric_oracle


def test_numeric_measurement_matches_independent_oracle() -> None:
    cases = build_numeric_cases(FORMAL_SEED, "qualification", 32)
    for case in cases:
        state = build_numeric_state(case)
        expected = numeric_oracle(case)
        assert numeric_decision(state) == expected["winner"]
        assert audit_numeric_state(case, state)["passed"] is True
    metrics = evaluate_numeric(cases)
    assert all(value == 1.0 for key, value in metrics.items() if key != "case_count")


def test_numeric_holdout_and_metamorphic_are_exact() -> None:
    qualification = build_numeric_cases(FORMAL_SEED, "qualification", 16)
    heldout = build_numeric_cases(FORMAL_SEED, "heldout", 16)
    assert max(
        abs(delta)
        for case in qualification
        for candidate in case.candidates
        for delta in candidate.deltas
    ) <= 32
    assert min(
        abs(delta)
        for case in heldout
        for candidate in case.candidates
        for delta in candidate.deltas
    ) >= 64
    assert all(value == 1.0 for value in evaluate_numeric_metamorphic(qualification + heldout).values())
    assert all(
        any(delta > 0 for delta in candidate.deltas)
        and any(delta < 0 for delta in candidate.deltas)
        for case in qualification + heldout
        for candidate in case.candidates
    )
    for case in qualification + heldout:
        expected = numeric_oracle(case)
        winner = next(
            candidate for candidate in case.candidates
            if candidate.handle == expected["winner"]
        )
        assert winner.deltas[-1] == -max(
            abs(candidate.deltas[-1]) for candidate in case.candidates
        )
