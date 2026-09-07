from __future__ import annotations

"""Target-only support audit and deterministic nested-fold ledger."""

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from . import contract


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_order(values: Sequence[str], *, salt: str) -> list[str]:
    return sorted(
        (str(value) for value in values),
        key=lambda value: (
            hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )


def _round_robin(values: Sequence[str], folds: int, *, salt: str) -> dict[str, int]:
    if type(folds) is not int or folds < 2:
        raise ValueError("fold count must be an integer >= 2")
    ordered = _hash_order(values, salt=salt)
    if len(ordered) < folds:
        raise ValueError("fewer eligible records than folds")
    return {value: index % folds for index, value in enumerate(ordered)}


def _strict_binary_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.ndim != ndim or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}D array")
    if any(
        not (
            type(item) is bool
            or (type(item) is int and item in {0, 1})
        )
        for item in raw.reshape(-1).tolist()
    ):
        raise TypeError(f"{name} must contain only JSON booleans or integer 0/1")
    return raw.astype(bool)


def record_feature_support(row: Mapping[str, Any], feature_name: str) -> dict[str, Any]:
    """Return record-local target support without reading model state."""

    example_id = row.get("example_id")
    family = row.get("family")
    names = row.get("state_feature_names")
    if type(example_id) is not str or not example_id:
        raise ValueError("target row has no example_id")
    if type(family) is not str or family.upper() not in contract.FAMILIES:
        raise ValueError(f"unsupported family for {example_id}")
    if not isinstance(names, list) or any(type(name) is not str for name in names):
        raise TypeError(f"state_feature_names malformed for {example_id}")
    if names.count(feature_name) != 1:
        raise ValueError(f"feature {feature_name} is not unique for {example_id}")

    values = _strict_binary_array(row.get("state_values"), name="state_values", ndim=3)
    masks = _strict_binary_array(
        row.get("state_feature_mask"), name="state_feature_mask", ndim=3
    )
    steps = _strict_binary_array(row.get("state_step_mask"), name="state_step_mask", ndim=1)
    if values.shape != masks.shape:
        raise ValueError(f"state values/mask shape mismatch for {example_id}")
    if values.shape[0] != steps.shape[0] or values.shape[2] != len(names):
        raise ValueError(f"state target axes mismatch for {example_id}")

    feature = names.index(feature_name)
    keep = masks[:, :, feature] & steps[:, None]
    target = values[:, :, feature][keep]
    if target.size == 0:
        raise ValueError(f"feature {feature_name} has no valid target for {example_id}")
    positives = int(target.sum())
    negatives = int((~target).sum())

    changes = 0
    previous: dict[int, bool] = {}
    for step in range(values.shape[0]):
        if not steps[step]:
            continue
        for slot in range(values.shape[1]):
            if not masks[step, slot, feature]:
                continue
            current = bool(values[step, slot, feature])
            if slot in previous and previous[slot] != current:
                changes += 1
            previous[slot] = current
    return {
        "example_id": example_id,
        "family": family.upper(),
        "feature": feature_name,
        "positive_observations": positives,
        "negative_observations": negatives,
        "valid_observations": positives + negatives,
        "cross_step_changes": int(changes),
        "eligible": positives > 0 and negatives > 0,
    }


def _cell_ledger(
    support_rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    feature: str,
    identity: str,
    seed: int,
    outer_folds: int,
    inner_folds: int,
    min_outer: int,
    min_inner: int,
) -> dict[str, Any]:
    if len({str(row["example_id"]) for row in support_rows}) != len(support_rows):
        raise ValueError(f"duplicate record in {family}/{feature}")
    eligible = sorted(
        str(row["example_id"]) for row in support_rows if row.get("eligible") is True
    )
    ineligible = sorted(
        str(row["example_id"]) for row in support_rows if row.get("eligible") is not True
    )
    salt = f"{identity}|{seed}|{family}|{feature}"
    outer = _round_robin(eligible, outer_folds, salt=f"{salt}|outer")
    outer_cells: list[dict[str, Any]] = []
    passed = True
    for heldout in range(outer_folds):
        outer_test = sorted(record for record in eligible if outer[record] == heldout)
        outer_train = sorted(record for record in eligible if outer[record] != heldout)
        inner = _round_robin(
            outer_train,
            inner_folds,
            salt=f"{salt}|outer={heldout}|inner",
        )
        inner_cells: list[dict[str, Any]] = []
        for inner_heldout in range(inner_folds):
            inner_test = sorted(
                record for record in outer_train if inner[record] == inner_heldout
            )
            inner_train = sorted(
                record for record in outer_train if inner[record] != inner_heldout
            )
            cell_passed = len(inner_test) >= min_inner and len(inner_train) >= min_inner
            passed = passed and cell_passed
            inner_cells.append(
                {
                    "fold": inner_heldout,
                    "test_ids": inner_test,
                    "train_ids": inner_train,
                    "test_eligible_records": len(inner_test),
                    "train_eligible_records": len(inner_train),
                    "passed": cell_passed,
                }
            )
        outer_passed = len(outer_test) >= min_outer and len(outer_train) >= min_outer
        passed = passed and outer_passed
        outer_cells.append(
            {
                "fold": heldout,
                "test_ids": outer_test,
                "train_ids": outer_train,
                "test_eligible_records": len(outer_test),
                "train_eligible_records": len(outer_train),
                "inner_assignment": inner,
                "inner_cells": inner_cells,
                "passed": outer_passed and all(cell["passed"] for cell in inner_cells),
            }
        )
    return {
        "family": family,
        "feature": feature,
        "records": [dict(row) for row in sorted(support_rows, key=lambda row: str(row["example_id"]))],
        "eligible_ids": eligible,
        "ineligible_ids": ineligible,
        "eligible_records": len(eligible),
        "ineligible_records": len(ineligible),
        "positive_observations": sum(int(row["positive_observations"]) for row in support_rows),
        "negative_observations": sum(int(row["negative_observations"]) for row in support_rows),
        "cross_step_changes": sum(int(row["cross_step_changes"]) for row in support_rows),
        "outer_assignment": outer,
        "outer_cells": outer_cells,
        "passed": passed,
    }


def build_fold_ledger(
    target_rows: Sequence[Mapping[str, Any]],
    *,
    identity: str = contract.IDENTITY,
    seed: int = contract.FOLD_SEED,
    outer_folds: int = contract.OUTER_FOLDS,
    inner_folds: int = contract.INNER_FOLDS,
    min_outer: int = contract.MIN_OUTER_ELIGIBLE_RECORDS,
    min_inner: int = contract.MIN_INNER_ELIGIBLE_RECORDS,
) -> dict[str, Any]:
    """Build the complete fold ledger before any endpoint output exists."""

    rows = [dict(row) for row in target_rows]
    ids = [row.get("example_id") for row in rows]
    if len(rows) != 32 or any(type(value) is not str or not value for value in ids):
        raise ValueError("temporal diagnosis requires exactly 32 identified target rows")
    if len(set(ids)) != len(ids):
        raise ValueError("temporal diagnosis target rows contain duplicate IDs")
    family_rows: dict[str, list[dict[str, Any]]] = {}
    for family in contract.FAMILIES:
        selected = [row for row in rows if str(row.get("family", "")).upper() == family]
        if len(selected) != 16:
            raise ValueError(f"{family} must contain exactly 16 records")
        family_rows[family] = selected

    cells: dict[str, dict[str, Any]] = {}
    for family, features in contract.HARD_FEATURES.items():
        cells[family] = {}
        for feature in features:
            support = [record_feature_support(row, feature) for row in family_rows[family]]
            cells[family][feature] = _cell_ledger(
                support,
                family=family,
                feature=feature,
                identity=identity,
                seed=seed,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                min_outer=min_outer,
                min_inner=min_inner,
            )
    passed = all(cell["passed"] for family in cells.values() for cell in family.values())
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.fold-ledger.v1",
        "identity": identity,
        "basis": contract.FOLD_POLICY["basis"],
        "latent_or_prediction_loaded": False,
        "seed": int(seed),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "minimum_outer_eligible_records": int(min_outer),
        "minimum_inner_eligible_records": int(min_inner),
        "eligibility": contract.FOLD_POLICY["eligibility"],
        "ineligible_policy": contract.FOLD_POLICY["ineligible_policy"],
        "cells": cells,
        "passed": passed,
    }


def audit_fold_ledger(
    target_rows: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild the target-only ledger and require byte-equivalent JSON data."""

    rebuilt = build_fold_ledger(target_rows)
    checks = {
        "identity": ledger.get("identity") == contract.IDENTITY,
        "schema": ledger.get("schema_version")
        == f"{contract.SCHEMA_PREFIX}.fold-ledger.v1",
        "target_only": ledger.get("latent_or_prediction_loaded") is False,
        "coverage": ledger.get("passed") is True,
        "exact_regeneration": _canonical(dict(ledger)) == _canonical(rebuilt),
    }
    return {"checks": checks, "passed": all(checks.values()), "rebuilt": rebuilt}


__all__ = ["audit_fold_ledger", "build_fold_ledger", "record_feature_support"]
