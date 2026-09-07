from __future__ import annotations

from yggdrasil_v2.r1_revalidation.nr1.measurement import (
    audit_relation_state,
    build_relation_state,
    relation_decisions,
)
from yggdrasil_v2.r1_revalidation.nr1.qualification import (
    FORMAL_SEED,
    build_relation_cases,
    evaluate_relation,
    evaluate_relation_metamorphic,
    run_qualification_bundle,
)
from yggdrasil_v2.r1_revalidation.nr1.reference import relation_oracle


def test_relation_fixed_point_matches_bfs_oracle() -> None:
    cases = build_relation_cases(FORMAL_SEED, "qualification", 32)
    for case in cases:
        state = build_relation_state(case)
        expected = relation_oracle(case)
        assert set(state.closure) == expected["closure"]
        assert relation_decisions(state, case.queries) == expected["decisions"]
        assert audit_relation_state(case, state)["passed"] is True
    metrics = evaluate_relation(cases)
    assert all(value == 1.0 for key, value in metrics.items() if key != "case_count")


def test_relation_holdout_and_metamorphic_are_exact() -> None:
    qualification = build_relation_cases(FORMAL_SEED, "qualification", 16)
    heldout = build_relation_cases(FORMAL_SEED, "heldout", 16)
    assert max(len(case.handles) for case in qualification) <= 8
    assert min(len(case.handles) for case in heldout) >= 9
    assert all(value == 1.0 for value in evaluate_relation_metamorphic(qualification + heldout).values())


def test_relation_generator_has_non_chain_topology_and_varied_truth_layouts() -> None:
    manifest = run_qualification_bundle()["case_manifest"]["relation"]["topology"]
    for split in ("qualification", "heldout"):
        row = manifest[split]
        assert row["branch_case_count"] > 0
        assert row["merge_case_count"] > 0
        assert row["multi_component_case_count"] > 0
        assert row["redundant_direct_case_count"] > 0
        assert len(row["query_count_histogram"]) == 5
        assert row["query_truth_pattern_count"] >= 16
        assert 0.35 <= row["positive_query_rate"] <= 0.65
