from __future__ import annotations

"""Deterministic grouped folds, cross-validation, and heldout evaluation."""

from collections import Counter
import hashlib
from typing import Any, Mapping, Sequence

from .canonical import (
    ContractError,
    LABELS,
    exact_keys,
    require_label,
    require_mask,
    require_string,
    sha256_json,
    validate_unique_ids,
)
from .models import fit_multinomial_nb, predict_multinomial_nb


def _validate_fold_rows(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> None:
    if not isinstance(rows, list) or not rows:
        raise ContractError("empty_rows", "fold rows cannot be empty")
    validate_unique_ids(rows)
    for row in rows:
        require_string(row.get("group_id"), code="invalid_group", field="group_id")
        require_label(row.get("label"), labels)


def assign_grouped_folds(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str] = LABELS,
    fold_count: int = 5,
) -> dict[str, Any]:
    if fold_count != 5:
        raise ContractError("invalid_fold_count", "R0C qualification requires exactly five folds")
    _validate_fold_rows(rows, labels)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    if len(groups) < fold_count:
        raise ContractError("insufficient_groups", "grouped CV requires at least five groups")

    group_counts = {
        group_id: Counter(row["label"] for row in group_rows)
        for group_id, group_rows in groups.items()
    }
    ordered_groups = sorted(
        groups,
        key=lambda group_id: hashlib.sha256(f"r1r-v10-fold|{group_id}".encode("utf-8")).hexdigest(),
    )
    fold_label_counts = [{label: 0 for label in labels} for _ in range(fold_count)]
    fold_totals = [0] * fold_count
    group_to_fold: dict[str, int] = {}

    def candidate_key(group_id: str, fold_index: int) -> tuple[int, int, int]:
        simulated_counts = [dict(item) for item in fold_label_counts]
        simulated_totals = list(fold_totals)
        for label, count in group_counts[group_id].items():
            simulated_counts[fold_index][label] += count
        simulated_totals[fold_index] += len(groups[group_id])
        label_deviation = max(
            max(row[label] for row in simulated_counts) - min(row[label] for row in simulated_counts)
            for label in labels
        )
        row_deviation = max(simulated_totals) - min(simulated_totals)
        return label_deviation, row_deviation, fold_index

    for group_id in ordered_groups:
        target = min(range(fold_count), key=lambda index: candidate_key(group_id, index))
        group_to_fold[group_id] = target
        for label, count in group_counts[group_id].items():
            fold_label_counts[target][label] += count
        fold_totals[target] += len(groups[group_id])

    if any(total == 0 for total in fold_totals):
        raise ContractError("empty_fold", "grouped assignment produced an empty fold")
    for heldout_fold in range(fold_count):
        fit_labels = {
            row["label"]
            for row in rows
            if group_to_fold[row["group_id"]] != heldout_fold
        }
        if len(fit_labels) < 2:
            raise ContractError("single_class_fold", f"fit side for fold {heldout_fold} is single-class")

    balance = {
        "label_fold_counts": {
            label: [fold_label_counts[index][label] for index in range(fold_count)]
            for label in labels
        },
        "row_totals": fold_totals,
        "max_label_deviation": max(
            max(row[label] for row in fold_label_counts) - min(row[label] for row in fold_label_counts)
            for label in labels
        ),
        "max_row_deviation": max(fold_totals) - min(fold_totals),
    }
    return {
        "group_to_fold": {group_id: group_to_fold[group_id] for group_id in sorted(group_to_fold)},
        "balance": balance,
    }


def confusion_matrix(
    rows: Sequence[Mapping[str, str]],
    labels: Sequence[str] = LABELS,
) -> dict[str, dict[str, int]]:
    matrix = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    for row in rows:
        truth = require_label(row["truth"], labels)
        prediction = require_label(row["prediction"], labels)
        matrix[truth][prediction] += 1
    return matrix


def _fit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"row_id": row["row_id"], "text": row["text"], "label": row["label"]}
        for row in rows
    ]


def _validate_evaluation_rows(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], *, code: str) -> None:
    if not isinstance(rows, list) or not rows:
        raise ContractError("empty_rows", f"{code} rows cannot be empty")
    for row in rows:
        exact_keys(
            row,
            ("row_id", "group_id", "text", "label", "valid_choice_mask"),
            code=f"{code}_row_schema",
        )
        require_string(row["row_id"], code="invalid_id", field="row_id")
        require_string(row["group_id"], code="invalid_group", field="group_id")
        require_string(row["text"], code="invalid_text", field="text")
        require_label(row["label"], labels)
        require_mask(row["valid_choice_mask"], labels)
    validate_unique_ids(rows)


def evaluate_grouped_cv(case: Mapping[str, Any], labels: Sequence[str] = LABELS) -> dict[str, Any]:
    rows = case["rows"]
    _validate_evaluation_rows(rows, labels, code="cv")
    assignment = assign_grouped_folds(rows, labels, case["fold_count"])
    group_to_fold = assignment["group_to_fold"]
    prediction_rows: list[dict[str, Any]] = []
    fold_states: dict[str, dict[str, str]] = {}
    for fold_index in range(case["fold_count"]):
        fit = [row for row in rows if group_to_fold[row["group_id"]] != fold_index]
        heldout = [row for row in rows if group_to_fold[row["group_id"]] == fold_index]
        model = fit_multinomial_nb(_fit_rows(fit), case["analyzer"], labels)
        fold_states[str(fold_index)] = {
            "vocabulary_sha256": model["vocabulary_sha256"],
            "fit_row_fingerprint": model["fit_row_fingerprint"],
            "model_state_sha256": sha256_json(model),
        }
        for row in heldout:
            result = predict_multinomial_nb(model, row["text"], row["valid_choice_mask"])
            prediction_rows.append(
                {
                    "row_id": row["row_id"],
                    "fold": fold_index,
                    "truth": row["label"],
                    "prediction": result["prediction"],
                }
            )
    prediction_rows.sort(key=lambda item: item["row_id"])
    return {
        "case_id": case["case_id"],
        "group_to_fold": group_to_fold,
        "balance": assignment["balance"],
        "predictions": prediction_rows,
        "confusion": confusion_matrix(prediction_rows, labels),
        "fold_states": fold_states,
    }


def evaluate_train_heldout(case: Mapping[str, Any], labels: Sequence[str] = LABELS) -> dict[str, Any]:
    train_rows = case["train_rows"]
    heldout_rows = case["heldout_rows"]
    _validate_evaluation_rows(train_rows, labels, code="train")
    _validate_evaluation_rows(heldout_rows, labels, code="heldout")
    train_row_ids = {row["row_id"] for row in train_rows}
    heldout_row_ids = {row["row_id"] for row in heldout_rows}
    if train_row_ids & heldout_row_ids:
        raise ContractError("fit_eval_row_overlap", "train and heldout row ids overlap")
    train_groups = {row["group_id"] for row in train_rows}
    heldout_groups = {row["group_id"] for row in heldout_rows}
    if train_groups & heldout_groups:
        raise ContractError("fit_eval_group_overlap", "train and heldout group ids overlap")
    model = fit_multinomial_nb(_fit_rows(train_rows), case["analyzer"], labels)
    predictions: list[dict[str, str]] = []
    for row in heldout_rows:
        result = predict_multinomial_nb(model, row["text"], row["valid_choice_mask"])
        predictions.append(
            {"row_id": row["row_id"], "truth": row["label"], "prediction": result["prediction"]}
        )
    predictions.sort(key=lambda item: item["row_id"])
    return {
        "case_id": case["case_id"],
        "predictions": predictions,
        "confusion": confusion_matrix(predictions, labels),
        "train_state": {
            "vocabulary": model["vocabulary"],
            "vocabulary_sha256": model["vocabulary_sha256"],
            "fit_row_fingerprint": model["fit_row_fingerprint"],
            "model_state_sha256": sha256_json(model),
        },
    }

