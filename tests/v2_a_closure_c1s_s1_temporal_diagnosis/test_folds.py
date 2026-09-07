from __future__ import annotations

from copy import deepcopy

import pytest

from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis import contract
from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis.folds import (
    audit_fold_ledger,
    build_fold_ledger,
    record_feature_support,
)


CPS_NAMES = [
    "processed",
    "legal",
    "valid",
    "budget_ok",
    "goal_satisfied",
    "final_constraints_satisfied",
    "running_best",
    "final_winner",
]
ERE_NAMES = [
    "present",
    "touched",
    "changed",
    "query_owner",
    "operation_source",
    "operation_target",
    "query_semantic_match",
    "stable_after",
]


def make_target_rows() -> list[dict]:
    rows: list[dict] = []
    for family, names in (("CPS", CPS_NAMES), ("ERE", ERE_NAMES)):
        for record in range(16):
            values = []
            for step in range(4):
                step_values = []
                for slot in range(2):
                    feature_values = []
                    for feature, name in enumerate(names):
                        value = bool((record + step + slot + feature) % 2)
                        if family == "CPS" and name == "running_best" and record in {2, 5, 8, 11}:
                            value = False
                        feature_values.append(value)
                    step_values.append(feature_values)
                values.append(step_values)
            rows.append(
                {
                    "example_id": f"{family.lower()}-{record:02d}",
                    "family": family,
                    "state_feature_names": names,
                    "state_values": values,
                    "state_feature_mask": [
                        [[True] * len(names) for _slot in range(2)] for _step in range(4)
                    ],
                    "state_step_mask": [True] * 4,
                }
            )
    return rows


def test_target_only_ledger_stratifies_every_nested_cell() -> None:
    ledger = build_fold_ledger(make_target_rows())
    assert ledger["passed"] is True
    assert ledger["latent_or_prediction_loaded"] is False
    for family, features in contract.HARD_FEATURES.items():
        for feature in features:
            cell = ledger["cells"][family][feature]
            expected = 12 if (family, feature) == ("CPS", "running_best") else 16
            per_fold = 3 if expected == 12 else 4
            assert cell["eligible_records"] == expected
            assert [row["test_eligible_records"] for row in cell["outer_cells"]] == [
                per_fold
            ] * 4
            assert all(
                [inner["test_eligible_records"] for inner in outer["inner_cells"]]
                == [per_fold] * 3
                for outer in cell["outer_cells"]
            )


def test_ledger_is_deterministic_and_exactly_regenerated() -> None:
    rows = make_target_rows()
    left = build_fold_ledger(rows)
    right = build_fold_ledger(list(reversed(rows)))
    assert left == right
    assert audit_fold_ledger(rows, left)["passed"] is True
    damaged = deepcopy(left)
    damaged["cells"]["CPS"]["running_best"]["outer_assignment"][
        damaged["cells"]["CPS"]["running_best"]["eligible_ids"][0]
    ] = 3
    assert audit_fold_ledger(rows, damaged)["passed"] is False


def test_running_best_ineligible_records_are_explicit_not_micro_pooled() -> None:
    rows = make_target_rows()
    ledger = build_fold_ledger(rows)
    cell = ledger["cells"]["CPS"]["running_best"]
    assert cell["ineligible_records"] == 4
    assert cell["ineligible_ids"] == ["cps-02", "cps-05", "cps-08", "cps-11"]
    assert all(
        row["positive_observations"] == 0 and row["negative_observations"] == 8
        for row in cell["records"]
        if row["example_id"] in cell["ineligible_ids"]
    )


def test_support_parser_accepts_binary_ints_but_rejects_other_truthy_values() -> None:
    row = make_target_rows()[0]
    row["state_values"][0][0][0] = 1
    assert record_feature_support(row, "processed")["valid_observations"] == 8
    row["state_values"][0][0][0] = 2
    with pytest.raises(TypeError, match="integer 0/1"):
        record_feature_support(row, "processed")
