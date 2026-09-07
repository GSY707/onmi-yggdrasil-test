from __future__ import annotations

from yggdrasil_v2.r1_revalidation.nr1.qualification import (
    HAND_FIXTURE_SHA256,
    evaluate_hand_fixtures,
)


def test_hand_authored_expected_bundle_is_frozen_and_independent() -> None:
    result = evaluate_hand_fixtures()
    assert HAND_FIXTURE_SHA256 != "TO_BE_FROZEN"
    assert result["hash_matches"] is True
    assert result["provenance_valid"] is True
    assert result["numeric_case_count"] >= 6
    assert result["numeric_exact_rate"] == 1.0
    assert result["relation_case_count"] >= 6
    assert result["relation_exact_rate"] == 1.0
    assert result["fault_registry"]["registry_complete"] is True
    assert result["fault_registry"]["passed"] is True
    assert result["passed"] is True
