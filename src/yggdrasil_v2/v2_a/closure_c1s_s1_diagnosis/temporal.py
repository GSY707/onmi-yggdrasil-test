from __future__ import annotations

"""Cross-fitted readout audit for non-answer-derived temporal state targets."""

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import balanced_accuracy, finite_json


HARD_FEATURES = {
    "ERE": ("touched", "changed", "operation_source", "operation_target"),
    "CPS": ("processed", "running_best"),
}
ANSWER_DERIVED = {"CPS": ("final_winner",), "ERE": ("query_semantic_match",)}


def _fold_map(record_ids: Sequence[str], folds: int, *, salt: str) -> dict[str, int]:
    unique = sorted(set(str(value) for value in record_ids))
    if len(unique) < folds:
        raise ValueError("fewer records than registered folds")
    mapped = {
        value: int.from_bytes(
            hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest()[:8], "big"
        )
        % folds
        for value in unique
    }
    if set(mapped.values()) != set(range(folds)):
        ordered = sorted(
            unique,
            key=lambda value: (
                hashlib.sha256(f"{salt}|fallback|{value}".encode("utf-8")).hexdigest(),
                value,
            ),
        )
        mapped = {value: index % folds for index, value in enumerate(ordered)}
    return mapped


def _binary_ba(prediction: np.ndarray, target: np.ndarray) -> float:
    report = balanced_accuracy(prediction.astype(int), target.astype(int))
    value = report["balanced_accuracy"]
    if value is None:
        raise ValueError("both target classes are required")
    return float(value)


def _record_macro_ba(
    prediction: np.ndarray,
    target: np.ndarray,
    records: np.ndarray,
) -> tuple[float, int, int]:
    """Average binary BA over records, never over flattened slot/step rows.

    A record with no positive or no negative observation cannot define BA and
    is excluded from the macro estimate.  The caller receives the valid count
    and total count so a diagnosis cannot hide this support loss.
    """

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    records = np.asarray(records)
    if prediction.ndim != 1 or target.shape != prediction.shape or records.shape != prediction.shape:
        raise ValueError("record macro inputs must be equal-length vectors")
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records.tolist()):
        grouped.setdefault(str(record), []).append(index)
    values: list[float] = []
    for indices in grouped.values():
        local_target = target[indices]
        if not local_target.any() or local_target.all():
            continue
        values.append(_binary_ba(prediction[indices], local_target))
    if not values:
        return 0.0, 0, len(grouped)
    return float(np.mean(values)), len(values), len(grouped)


def _auc(score: np.ndarray, target: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=bool)
    positive = np.flatnonzero(target)
    negative = np.flatnonzero(~target)
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires both classes")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[positive].sum() - len(positive) * (len(positive) + 1) / 2.0)
        / (len(positive) * len(negative))
    )


def _class_weights(target: np.ndarray) -> np.ndarray:
    target = target.astype(bool)
    positive = int(target.sum())
    negative = int((~target).sum())
    if not positive or not negative:
        raise ValueError("ridge training requires both classes")
    weights = np.where(target, 0.5 / positive, 0.5 / negative)
    return weights * len(target)


def _ridge_fit(
    features: np.ndarray,
    target: np.ndarray,
    *,
    rank: int,
    regularization: float,
) -> dict[str, np.ndarray | float]:
    weights = _class_weights(target)
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    x = (features - mean) / scale
    y_mean = float(np.average(target.astype(float), weights=weights))
    square_root = np.sqrt(weights)
    xw = x * square_root[:, None]
    yw = (target.astype(float) - y_mean) * square_root
    u, singular, vt = np.linalg.svd(xw, full_matrices=False)
    used = min(int(rank), len(singular))
    coefficient = vt[:used].T @ (
        (singular[:used] / (singular[:used] ** 2 + float(regularization)))
        * (u[:, :used].T @ yw)
    )
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": y_mean,
    }


def _ridge_predict(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    return (
        ((features - model["mean"]) / model["scale"]) @ model["coefficient"]
        + float(model["intercept"])
    )


def _select_ridge(
    features: np.ndarray,
    target: np.ndarray,
    records: np.ndarray,
    *,
    ranks: Sequence[int],
    regularizations: Sequence[float],
    salt: str,
) -> tuple[int, float, list[dict[str, Any]]]:
    fold = _fold_map(records.tolist(), 3, salt=salt)
    candidates: list[dict[str, Any]] = []
    for rank in ranks:
        for regularization in regularizations:
            scores: list[float] = []
            valid_counts: list[int] = []
            total_counts: list[int] = []
            for heldout in range(3):
                test = np.asarray([fold[str(value)] == heldout for value in records])
                train = ~test
                if not train.any() or not test.any():
                    raise ValueError("empty inner readout fold")
                model = _ridge_fit(
                    features[train],
                    target[train],
                    rank=int(rank),
                    regularization=float(regularization),
                )
                prediction = _ridge_predict(model, features[test]) >= 0.5
                score, valid, total = _record_macro_ba(
                    prediction, target[test], records[test]
                )
                if valid == 0:
                    raise ValueError("inner readout fold has no record with both target classes")
                scores.append(score)
                valid_counts.append(valid)
                total_counts.append(total)
            candidates.append(
                {
                    "rank": int(rank),
                    "regularization": float(regularization),
                    "mean_inner_record_macro_balanced_accuracy": float(np.mean(scores)),
                    "inner_record_macro_valid_records": int(sum(valid_counts)),
                    "inner_record_macro_total_records": int(sum(total_counts)),
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row["mean_inner_record_macro_balanced_accuracy"]),
            -int(row["inner_record_macro_valid_records"]),
            int(row["rank"]),
            float(row["regularization"]),
        )
    )
    selected = candidates[0]
    return int(selected["rank"]), float(selected["regularization"]), candidates


def _cross_fitted_ridge(
    features: np.ndarray,
    target: np.ndarray,
    records: np.ndarray,
    *,
    ranks: Sequence[int],
    regularizations: Sequence[float],
    salt: str,
) -> dict[str, Any]:
    folds = _fold_map(records.tolist(), 4, salt=f"{salt}|outer")
    score = np.empty(len(target), dtype=float)
    selections: list[dict[str, Any]] = []
    for heldout in range(4):
        test = np.asarray([folds[str(value)] == heldout for value in records])
        train = ~test
        rank, regularization, candidates = _select_ridge(
            features[train],
            target[train],
            records[train],
            ranks=ranks,
            regularizations=regularizations,
            salt=f"{salt}|outer={heldout}|inner",
        )
        model = _ridge_fit(
            features[train], target[train], rank=rank, regularization=regularization
        )
        score[test] = _ridge_predict(model, features[test])
        selections.append(
            {
                "outer_fold": heldout,
                "rank": rank,
                "regularization": regularization,
                "inner_best_record_macro_balanced_accuracy": candidates[0]["mean_inner_record_macro_balanced_accuracy"],
                "inner_record_macro_valid_records": candidates[0]["inner_record_macro_valid_records"],
                "inner_record_macro_total_records": candidates[0]["inner_record_macro_total_records"],
            }
        )
    prediction = score >= 0.5
    record_macro, valid_records, total_records = _record_macro_ba(
        prediction, target, records
    )
    return {
        "balanced_accuracy": _binary_ba(prediction, target),
        "observation_balanced_accuracy": _binary_ba(prediction, target),
        "record_macro_balanced_accuracy": record_macro,
        "record_macro_valid_records": valid_records,
        "record_macro_total_records": total_records,
        "auc": _auc(score, target),
        "predictions": prediction,
        "scores": score,
        "selections": selections,
    }


def _cross_fitted_channel_auc(
    features: np.ndarray,
    target: np.ndarray,
    records: np.ndarray,
    *,
    salt: str,
) -> float:
    folds = _fold_map(records.tolist(), 4, salt=salt)
    score = np.empty(len(target), dtype=float)
    for heldout in range(4):
        test = np.asarray([folds[str(value)] == heldout for value in records])
        train = ~test
        choices: list[tuple[float, int, float]] = []
        for channel in range(features.shape[1]):
            auc = _auc(features[train, channel], target[train])
            orientation = 1.0 if auc >= 0.5 else -1.0
            choices.append((max(auc, 1.0 - auc), channel, orientation))
        _quality, channel, orientation = max(choices, key=lambda row: (row[0], -row[1]))
        score[test] = orientation * features[test, channel]
    auc = _auc(score, target)
    # Orientation was already selected from each outer-training partition.
    # Flipping again from the combined heldout targets would be leakage.
    return float(auc)


def _record_macro_null_gap(
    natural: Mapping[str, Any], nulls: Mapping[str, Mapping[str, Any]]
) -> tuple[float, float]:
    """Compare only record-macro BA values; never mix observation units."""

    natural_value = float(natural["record_macro_balanced_accuracy"])
    if int(natural["record_macro_valid_records"]) <= 0:
        raise ValueError("natural readout has no record-macro support")
    values: list[float] = []
    for name, report in nulls.items():
        if int(report["record_macro_valid_records"]) <= 0:
            raise ValueError(f"{name} null has no record-macro support")
        values.append(float(report["record_macro_balanced_accuracy"]))
    if not values:
        raise ValueError("registered temporal nulls are absent")
    strongest = max(values)
    return strongest, natural_value - strongest


def _feature_rows(
    collection: Mapping[str, Any],
    *,
    family: str,
    feature_name: str,
    transform: str = "natural",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    state = collection["state_features"].numpy().astype(np.float64)
    values = collection["state_values"].numpy().astype(bool)
    masks = collection["state_feature_mask"].numpy().astype(bool)
    step_masks = collection["state_step_mask"].numpy().astype(bool)
    families = [str(value).upper() for value in collection["families"]]
    names = collection["feature_names"]
    ids = list(collection["example_ids"])
    selected = [index for index, value in enumerate(families) if value == family]
    if transform == "time_shuffle":
        altered = state.copy()
        for index in selected:
            offset = 1 + int.from_bytes(
                hashlib.sha256(f"{seed}|time|{ids[index]}".encode()).digest()[:4], "big"
            ) % (state.shape[1] - 1)
            altered[index] = np.roll(altered[index], offset, axis=0)
        state = altered
    elif transform == "temporal_mean":
        state = np.repeat(state.mean(axis=1, keepdims=True), state.shape[1], axis=1)
    elif transform == "target_shuffle":
        if len(selected) < 2:
            raise ValueError("target shuffle needs two records")
        rng = np.random.default_rng(int(seed))
        donors = list(selected)
        while True:
            rng.shuffle(donors)
            if all(left != right for left, right in zip(selected, donors)):
                break
        copied_values = values.copy()
        copied_masks = masks.copy()
        for receiver, donor in zip(selected, donors):
            copied_values[receiver] = values[donor]
            copied_masks[receiver] = masks[donor]
        values, masks = copied_values, copied_masks
    elif transform != "natural":
        raise ValueError(f"unknown temporal transform: {transform}")

    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    record_rows: list[np.ndarray] = []
    temporal_changes = 0
    for index in selected:
        if feature_name not in names[index]:
            raise ValueError(f"feature absent from row: {family}/{feature_name}/{ids[index]}")
        feature = names[index].index(feature_name)
        keep = masks[index, ..., feature] & step_masks[index, :, None]
        feature_rows.append(state[index][keep])
        target_rows.append(values[index, ..., feature][keep])
        record_rows.append(np.repeat(ids[index], int(keep.sum())))
        previous: dict[int, bool] = {}
        for step in range(values.shape[1]):
            if not step_masks[index, step]:
                continue
            for slot in range(values.shape[2]):
                if not masks[index, step, slot, feature]:
                    continue
                value = bool(values[index, step, slot, feature])
                if slot in previous and previous[slot] != value:
                    temporal_changes += 1
                previous[slot] = value
    return (
        np.concatenate(feature_rows, axis=0),
        np.concatenate(target_rows, axis=0).astype(bool),
        np.concatenate(record_rows, axis=0),
        temporal_changes,
    )


def audit_temporal_readout(
    collection: Mapping[str, Any],
    *,
    ranks: Sequence[int],
    regularizations: Sequence[float],
    seed: int,
    target_positive_min: int,
    target_negative_min: int,
    target_changes_min: int,
    source_visible_replay: bool,
) -> dict[str, Any]:
    """Compare the frozen decoder with a record-cross-fitted linear ceiling."""

    state_logits = collection["state_logits"].numpy().astype(np.float64)
    values = collection["state_values"].numpy().astype(bool)
    masks = collection["state_feature_mask"].numpy().astype(bool)
    step_masks = collection["state_step_mask"].numpy().astype(bool)
    families = [str(value).upper() for value in collection["families"]]
    names = collection["feature_names"]
    ids = list(collection["example_ids"])
    trajectory = collection["trajectory"].numpy().astype(np.float64)
    initial = collection["initial_payloads"].numpy().astype(np.float64)
    previous = np.concatenate((initial[:, None], trajectory[:, :-1]), axis=1)
    relative = np.linalg.norm(trajectory - previous, axis=-1) / np.median(
        np.linalg.norm(initial, axis=-1), axis=1
    ).clip(min=1.0e-8)[:, None, None]
    reports: dict[str, Any] = {}
    decision_rows: list[dict[str, Any]] = []
    for family, hard_names in HARD_FEATURES.items():
        selected = [index for index, value in enumerate(families) if value == family]
        if not selected:
            continue
        active = collection["operation_active_target"].numpy().astype(bool)[selected]
        motion = float(relative[selected][active].mean()) if active.any() else 0.0
        family_features: dict[str, Any] = {}
        for feature_name in hard_names:
            x, y, records, changes = _feature_rows(
                collection, family=family, feature_name=feature_name
            )
            natural = _cross_fitted_ridge(
                x,
                y,
                records,
                ranks=ranks,
                regularizations=regularizations,
                salt=f"C1S-S1-D004|{seed}|{family}|{feature_name}|natural",
            )
            nulls: dict[str, Any] = {}
            for offset, transform in enumerate(("time_shuffle", "temporal_mean", "target_shuffle"), start=1):
                nx, ny, nrecords, _ = _feature_rows(
                    collection,
                    family=family,
                    feature_name=feature_name,
                    transform=transform,
                    seed=seed + offset,
                )
                null = _cross_fitted_ridge(
                    nx,
                    ny,
                    nrecords,
                    ranks=ranks,
                    regularizations=regularizations,
                    salt=f"C1S-S1-D004|{seed}|{family}|{feature_name}|{transform}",
                )
                nulls[transform] = {
                    "balanced_accuracy": null["balanced_accuracy"],
                    "observation_balanced_accuracy": null["observation_balanced_accuracy"],
                    "record_macro_balanced_accuracy": null["record_macro_balanced_accuracy"],
                    "record_macro_valid_records": null["record_macro_valid_records"],
                    "record_macro_total_records": null["record_macro_total_records"],
                    "auc": null["auc"],
                }
            frozen_predictions: list[bool] = []
            frozen_targets: list[bool] = []
            frozen_records: list[str] = []
            for index in selected:
                feature = names[index].index(feature_name)
                keep = masks[index, ..., feature] & step_masks[index, :, None]
                frozen_predictions.extend((state_logits[index, ..., feature][keep] >= 0.0).tolist())
                frozen_targets.extend(values[index, ..., feature][keep].tolist())
                frozen_records.extend([ids[index]] * int(keep.sum()))
            frozen = balanced_accuracy(frozen_predictions, frozen_targets)
            frozen_record_macro, frozen_valid, frozen_total = _record_macro_ba(
                np.asarray(frozen_predictions, dtype=bool),
                np.asarray(frozen_targets, dtype=bool),
                np.asarray(frozen_records, dtype=object),
            )
            fixed_channel_auc = _cross_fitted_channel_auc(
                x,
                y,
                records,
                salt=f"C1S-S1-D004|{seed}|{family}|{feature_name}|channel",
            )
            positives, negatives = int(y.sum()), int((~y).sum())
            adequate = bool(
                source_visible_replay
                and positives >= int(target_positive_min)
                and negatives >= int(target_negative_min)
                and changes >= int(target_changes_min)
            )
            strongest_null, null_gap = _record_macro_null_gap(natural, nulls)
            report = {
                "target_adequate": adequate,
                "positives": positives,
                "negatives": negatives,
                "temporal_changes": int(changes),
                "source_visible_replay": bool(source_visible_replay),
                "source_visible_replay_basis": (
                    "source_ast_materialize_target_regeneration_same_implementation"
                    if source_visible_replay
                    else "not_verified; target_bank_integrity_is_not_replay_evidence"
                ),
                "source_visible_replay_boundary": (
                    "checks sealed source-to-bank consistency but cannot detect a bug shared with materialize_target"
                    if source_visible_replay
                    else "source-derived regeneration was not verified"
                ),
                "frozen_decoder_balanced_accuracy": frozen_record_macro,
                "frozen_decoder_observation_balanced_accuracy": frozen["balanced_accuracy"],
                "frozen_decoder_record_macro_valid_records": frozen_valid,
                "frozen_decoder_record_macro_total_records": frozen_total,
                "cross_fitted_ridge_balanced_accuracy": natural["record_macro_balanced_accuracy"],
                "cross_fitted_ridge_observation_balanced_accuracy": natural["observation_balanced_accuracy"],
                "cross_fitted_ridge_record_macro_valid_records": natural["record_macro_valid_records"],
                "cross_fitted_ridge_record_macro_total_records": natural["record_macro_total_records"],
                "cross_fitted_ridge_auc": natural["auc"],
                "cross_fitted_fixed_channel_auc": fixed_channel_auc,
                "ridge_selections": natural["selections"],
                "nulls": nulls,
                "ridge_minus_strongest_null": null_gap,
            }
            family_features[feature_name] = report
            decision_rows.append(
                {
                    "family": family,
                    "feature": feature_name,
                    "target_adequate": adequate,
                    "relative_motion": motion,
                    "separation_auc": fixed_channel_auc,
                    "crossfit_decoder_balanced_accuracy": natural["record_macro_balanced_accuracy"],
                    "crossfit_observation_balanced_accuracy": natural["observation_balanced_accuracy"],
                    "frozen_decoder_balanced_accuracy": frozen_record_macro,
                    "frozen_decoder_observation_balanced_accuracy": frozen["balanced_accuracy"],
                    "static_null_gap": null_gap,
                }
            )
        reports[family] = {
            "active_step_relative_motion": motion,
            "features": family_features,
            "answer_derived_features_excluded": list(ANSWER_DERIVED.get(family, ())),
        }
    return finite_json(
        {
            "families": reports,
            "decision_rows": decision_rows,
            "readout": {
                "kind": "record_cross_fitted_class_balanced_truncated_ridge",
                "ranks": [int(value) for value in ranks],
                "regularizations": [float(value) for value in regularizations],
                "outer_folds": 4,
                "inner_folds": 3,
                "optimizer_steps": 0,
                "model_writes": 0,
            },
        }
    )


__all__ = ["ANSWER_DERIVED", "HARD_FEATURES", "audit_temporal_readout"]
