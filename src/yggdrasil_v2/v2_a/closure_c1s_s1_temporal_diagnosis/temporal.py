from __future__ import annotations

"""Support-stratified temporal readout for the D004R successor diagnosis."""

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from ..closure_c1s_s1_diagnosis.metrics import balanced_accuracy, finite_json
from . import contract


def _binary_ba(prediction: np.ndarray, target: np.ndarray) -> float:
    report = balanced_accuracy(prediction.astype(int), target.astype(int))
    value = report["balanced_accuracy"]
    if value is None:
        raise ValueError("both target classes are required")
    return float(value)


def _auc(score: np.ndarray, target: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=bool)
    positive = np.flatnonzero(target)
    negative = np.flatnonzero(~target)
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires both target classes")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return float(
        (ranks[positive].sum() - len(positive) * (len(positive) + 1) / 2.0)
        / (len(positive) * len(negative))
    )


def _group_indices(records: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(np.asarray(records, dtype=object).tolist()):
        grouped.setdefault(str(record), []).append(index)
    return {
        record: np.asarray(indices, dtype=np.int64)
        for record, indices in grouped.items()
    }


def _record_macro_ba(
    prediction: np.ndarray, target: np.ndarray, records: np.ndarray
) -> tuple[float, int, int]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    records = np.asarray(records, dtype=object)
    if prediction.ndim != 1 or target.shape != prediction.shape or records.shape != target.shape:
        raise ValueError("record macro inputs must be equal-length vectors")
    values: list[float] = []
    grouped = _group_indices(records)
    for indices in grouped.values():
        local_target = target[indices]
        if not local_target.any() or local_target.all():
            continue
        values.append(_binary_ba(prediction[indices], local_target))
    if not values:
        return 0.0, 0, len(grouped)
    return float(np.mean(values)), len(values), len(grouped)


def _record_macro_auc(
    score: np.ndarray, target: np.ndarray, records: np.ndarray
) -> tuple[float, int, int]:
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=bool)
    records = np.asarray(records, dtype=object)
    if score.ndim != 1 or target.shape != score.shape or records.shape != target.shape:
        raise ValueError("record macro AUC inputs must be equal-length vectors")
    values: list[float] = []
    grouped = _group_indices(records)
    for indices in grouped.values():
        local_target = target[indices]
        if not local_target.any() or local_target.all():
            continue
        values.append(_auc(score[indices], local_target))
    if not values:
        return 0.0, 0, len(grouped)
    return float(np.mean(values)), len(values), len(grouped)


def _record_evidence(
    prediction: np.ndarray,
    target: np.ndarray,
    records: np.ndarray,
    *,
    score: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    records = np.asarray(records, dtype=object)
    if prediction.shape != target.shape or records.shape != target.shape:
        raise ValueError("record evidence inputs have incompatible shapes")
    if score is not None and np.asarray(score).shape != target.shape:
        raise ValueError("record evidence score shape mismatch")
    rows: list[dict[str, Any]] = []
    for record, indices in sorted(_group_indices(records).items()):
        local_target = target[indices]
        local_prediction = prediction[indices]
        if not local_target.any() or local_target.all():
            raise ValueError("record evidence contains an ineligible record")
        row: dict[str, Any] = {
            "example_id": record,
            "observations": int(len(indices)),
            "positives": int(local_target.sum()),
            "negatives": int((~local_target).sum()),
            "target_bits": "".join("1" if value else "0" for value in local_target),
            "prediction_bits": "".join(
                "1" if value else "0" for value in local_prediction
            ),
            "balanced_accuracy": _binary_ba(local_prediction, local_target),
        }
        if score is not None:
            local_score = np.asarray(score, dtype=float)[indices]
            row["scores"] = [float(value) for value in local_score]
            row["auc"] = _auc(local_score, local_target)
        rows.append(row)
    return rows


def _record_score_evidence(
    score: np.ndarray, target: np.ndarray, records: np.ndarray
) -> list[dict[str, Any]]:
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=bool)
    records = np.asarray(records, dtype=object)
    if score.shape != target.shape or records.shape != target.shape:
        raise ValueError("record score evidence inputs have incompatible shapes")
    rows: list[dict[str, Any]] = []
    for record, indices in sorted(_group_indices(records).items()):
        local_target = target[indices]
        local_score = score[indices]
        if not local_target.any() or local_target.all():
            raise ValueError("record score evidence contains an ineligible record")
        rows.append(
            {
                "example_id": record,
                "observations": int(len(indices)),
                "positives": int(local_target.sum()),
                "negatives": int((~local_target).sum()),
                "target_bits": "".join(
                    "1" if value else "0" for value in local_target
                ),
                "scores": [float(value) for value in local_score],
                "auc": _auc(local_score, local_target),
            }
        )
    return rows


def _record_class_weights(target: np.ndarray, records: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=bool)
    records = np.asarray(records, dtype=object)
    if records.shape != target.shape:
        raise ValueError("ridge record weights have incompatible shapes")
    grouped = _group_indices(records)
    if not grouped:
        raise ValueError("ridge training has no records")
    weights = np.zeros(len(target), dtype=float)
    for indices in grouped.values():
        local = target[indices]
        positives = int(local.sum())
        negatives = int((~local).sum())
        if positives == 0 or negatives == 0:
            raise ValueError("ridge training received an ineligible record")
        record_weight = 1.0 / len(grouped)
        weights[indices[local]] = 0.5 * record_weight / positives
        weights[indices[~local]] = 0.5 * record_weight / negatives
    return weights * len(target)


def _ridge_prepare(
    features: np.ndarray, target: np.ndarray, records: np.ndarray
) -> dict[str, Any]:
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=bool)
    if features.ndim != 2 or target.shape != (features.shape[0],):
        raise ValueError("ridge inputs have incompatible shapes")
    weights = _record_class_weights(target, records)
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    x = (features - mean) / scale
    intercept = float(np.average(target.astype(float), weights=weights))
    root = np.sqrt(weights)
    xw = x * root[:, None]
    yw = (target.astype(float) - intercept) * root
    u, singular, vt = np.linalg.svd(xw, full_matrices=False)
    return {
        "mean": mean,
        "scale": scale,
        "intercept": intercept,
        "singular": singular,
        "vt": vt,
        "uy": u.T @ yw,
    }


def _ridge_model(
    prepared: Mapping[str, Any], *, rank: int, regularization: float
) -> dict[str, Any]:
    singular = np.asarray(prepared["singular"], dtype=float)
    vt = np.asarray(prepared["vt"], dtype=float)
    uy = np.asarray(prepared["uy"], dtype=float)
    used = min(int(rank), len(singular))
    coefficient = vt[:used].T @ (
        (singular[:used] / (singular[:used] ** 2 + float(regularization)))
        * uy[:used]
    )
    return {
        "mean": prepared["mean"],
        "scale": prepared["scale"],
        "intercept": prepared["intercept"],
        "coefficient": coefficient,
    }


def _ridge_predict(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    return (
        (np.asarray(features, dtype=float) - model["mean"])
        / model["scale"]
        @ model["coefficient"]
        + float(model["intercept"])
    )


def _select_ridge(
    features: np.ndarray,
    target: np.ndarray,
    records: np.ndarray,
    *,
    inner_assignment: Mapping[str, int],
    ranks: Sequence[int],
    regularizations: Sequence[float],
) -> tuple[int, float, list[dict[str, Any]]]:
    folds = sorted(set(int(value) for value in inner_assignment.values()))
    if folds != list(range(contract.INNER_FOLDS)):
        raise ValueError("inner fold ledger is incomplete")
    candidate_scores: dict[tuple[int, float], list[float]] = {
        (int(rank), float(regularization)): []
        for rank in ranks
        for regularization in regularizations
    }
    candidate_valid: dict[tuple[int, float], int] = {
        key: 0 for key in candidate_scores
    }
    candidate_total: dict[tuple[int, float], int] = {
        key: 0 for key in candidate_scores
    }
    for heldout in folds:
        test = np.asarray(
            [int(inner_assignment[str(record)]) == heldout for record in records],
            dtype=bool,
        )
        train = ~test
        if not train.any() or not test.any():
            raise ValueError("empty registered inner readout fold")
        prepared = _ridge_prepare(features[train], target[train], records[train])
        for key in candidate_scores:
            rank, regularization = key
            model = _ridge_model(prepared, rank=rank, regularization=regularization)
            prediction = _ridge_predict(model, features[test]) >= 0.5
            score, valid, total = _record_macro_ba(
                prediction, target[test], records[test]
            )
            if valid < contract.MIN_INNER_ELIGIBLE_RECORDS or valid != total:
                raise ValueError("registered inner fold lost eligible record support")
            candidate_scores[key].append(score)
            candidate_valid[key] += valid
            candidate_total[key] += total
    candidates = [
        {
            "rank": rank,
            "regularization": regularization,
            "mean_inner_record_macro_balanced_accuracy": float(np.mean(scores)),
            "inner_record_macro_valid_records": int(candidate_valid[(rank, regularization)]),
            "inner_record_macro_total_records": int(candidate_total[(rank, regularization)]),
        }
        for (rank, regularization), scores in candidate_scores.items()
    ]
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
    cell_ledger: Mapping[str, Any],
    ranks: Sequence[int],
    regularizations: Sequence[float],
) -> dict[str, Any]:
    outer_assignment = {
        str(key): int(value)
        for key, value in dict(cell_ledger["outer_assignment"]).items()
    }
    expected = set(str(value) for value in cell_ledger["eligible_ids"])
    if set(str(value) for value in np.unique(records)) != expected:
        raise ValueError("readout records differ from frozen eligible ledger")
    score = np.full(len(target), np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    outer_cells = {
        int(row["fold"]): row for row in cell_ledger["outer_cells"]
    }
    for heldout in range(contract.OUTER_FOLDS):
        test = np.asarray(
            [outer_assignment[str(record)] == heldout for record in records], dtype=bool
        )
        train = ~test
        cell = outer_cells[heldout]
        inner_assignment = {
            str(key): int(value)
            for key, value in dict(cell["inner_assignment"]).items()
        }
        rank, regularization, candidates = _select_ridge(
            features[train],
            target[train],
            records[train],
            inner_assignment=inner_assignment,
            ranks=ranks,
            regularizations=regularizations,
        )
        prepared = _ridge_prepare(features[train], target[train], records[train])
        model = _ridge_model(prepared, rank=rank, regularization=regularization)
        score[test] = _ridge_predict(model, features[test])
        test_records = set(str(value) for value in np.unique(records[test]))
        if test_records != set(str(value) for value in cell["test_ids"]):
            raise ValueError("outer score records differ from frozen fold ledger")
        selections.append(
            {
                "outer_fold": heldout,
                "rank": rank,
                "regularization": regularization,
                "outer_test_ids": sorted(test_records),
                "inner_best_record_macro_balanced_accuracy": candidates[0][
                    "mean_inner_record_macro_balanced_accuracy"
                ],
                "inner_record_macro_valid_records": candidates[0][
                    "inner_record_macro_valid_records"
                ],
                "inner_record_macro_total_records": candidates[0][
                    "inner_record_macro_total_records"
                ],
            }
        )
    if not bool(np.isfinite(score).all()):
        raise FloatingPointError("outer cross-fit did not score every observation")
    prediction = score >= 0.5
    record_ba, valid, total = _record_macro_ba(prediction, target, records)
    record_auc, auc_valid, auc_total = _record_macro_auc(score, target, records)
    if valid != len(expected) or total != len(expected) or auc_valid != auc_total or auc_total != len(expected):
        raise ValueError("outer record-macro score lost eligible records")
    return {
        "record_macro_balanced_accuracy": record_ba,
        "record_macro_auc": record_auc,
        "record_macro_valid_records": valid,
        "record_macro_total_records": total,
        "observation_balanced_accuracy": _binary_ba(prediction, target),
        "observation_auc": _auc(score, target),
        "predictions": prediction,
        "scores": score,
        "selections": selections,
        "record_evidence": _record_evidence(
            prediction, target, records, score=score
        ),
    }


def _shuffle_target_within_records(
    target: np.ndarray, records: np.ndarray, *, seed: int, salt: str
) -> np.ndarray:
    result = np.asarray(target, dtype=bool).copy()
    for record, indices in _group_indices(records).items():
        local = result[indices].copy()
        if not local.any() or local.all():
            raise ValueError("target shuffle received an ineligible record")
        local_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{salt}|{record}".encode("utf-8")).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(local_seed)
        candidate = local[rng.permutation(len(local))]
        if np.array_equal(candidate, local):
            positive = int(np.flatnonzero(local)[0])
            negative = int(np.flatnonzero(~local)[0])
            candidate = local.copy()
            candidate[positive], candidate[negative] = (
                candidate[negative],
                candidate[positive],
            )
        result[indices] = candidate
    return result


def _feature_rows(
    collection: Mapping[str, Any],
    *,
    family: str,
    feature_name: str,
    eligible_ids: Sequence[str],
    transform: str = "natural",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = collection["state_features"].numpy().astype(np.float64)
    values = collection["state_values"].numpy().astype(bool)
    masks = collection["state_feature_mask"].numpy().astype(bool)
    step_masks = collection["state_step_mask"].numpy().astype(bool)
    families = [str(value).upper() for value in collection["families"]]
    names = collection["feature_names"]
    ids = [str(value) for value in collection["example_ids"]]
    by_id = {example_id: index for index, example_id in enumerate(ids)}
    if len(by_id) != len(ids):
        raise ValueError("collection contains duplicate record IDs")
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    record_rows: list[np.ndarray] = []
    for example_id in sorted(str(value) for value in eligible_ids):
        if example_id not in by_id:
            raise KeyError(f"eligible record absent from collection: {example_id}")
        index = by_id[example_id]
        if families[index] != family:
            raise ValueError(f"fold ledger family mismatch for {example_id}")
        if feature_name not in names[index]:
            raise ValueError(f"feature absent from collection row: {feature_name}/{example_id}")
        feature = names[index].index(feature_name)
        local_state = state[index]
        if transform == "time_shuffle":
            if local_state.shape[0] < 2:
                raise ValueError("time shuffle requires at least two steps")
            offset = 1 + int.from_bytes(
                hashlib.sha256(
                    f"{seed}|time|{family}|{feature_name}|{example_id}".encode("utf-8")
                ).digest()[:8],
                "big",
            ) % (local_state.shape[0] - 1)
            local_state = np.roll(local_state, offset, axis=0)
        elif transform == "temporal_mean":
            local_state = np.repeat(
                local_state.mean(axis=0, keepdims=True), local_state.shape[0], axis=0
            )
        elif transform not in {"natural", "within_record_target_shuffle"}:
            raise ValueError(f"unknown temporal transform: {transform}")
        keep = masks[index, :, :, feature] & step_masks[index, :, None]
        feature_rows.append(local_state[keep])
        target_rows.append(values[index, :, :, feature][keep])
        record_rows.append(np.repeat(example_id, int(keep.sum())))
    x = np.concatenate(feature_rows, axis=0)
    y = np.concatenate(target_rows, axis=0).astype(bool)
    records = np.concatenate(record_rows, axis=0).astype(object)
    if transform == "within_record_target_shuffle":
        y = _shuffle_target_within_records(
            y,
            records,
            seed=seed,
            salt=f"fit-null|{family}|{feature_name}",
        )
    return x, y, records


def _fixed_channel_record_macro_auc(
    features: np.ndarray,
    target: np.ndarray,
    records: np.ndarray,
    *,
    cell_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    outer_assignment = {
        str(key): int(value)
        for key, value in dict(cell_ledger["outer_assignment"]).items()
    }
    score = np.full(len(target), np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    for heldout in range(contract.OUTER_FOLDS):
        test = np.asarray(
            [outer_assignment[str(record)] == heldout for record in records], dtype=bool
        )
        train = ~test
        candidates: list[tuple[float, int, int]] = []
        for channel in range(features.shape[1]):
            direct, valid, total = _record_macro_auc(
                features[train, channel], target[train], records[train]
            )
            inverse, inverse_valid, inverse_total = _record_macro_auc(
                -features[train, channel], target[train], records[train]
            )
            if valid != total or inverse_valid != inverse_total or valid == 0:
                raise ValueError("fixed-channel training fold lost eligible support")
            candidates.append((direct, channel, 1))
            candidates.append((inverse, channel, -1))
        candidates.sort(key=lambda row: (-row[0], row[1], -row[2]))
        selected_auc, channel, orientation = candidates[0]
        score[test] = features[test, channel] * orientation
        selections.append(
            {
                "outer_fold": heldout,
                "channel": int(channel),
                "orientation": int(orientation),
                "outer_train_record_macro_auc": float(selected_auc),
            }
        )
    if not bool(np.isfinite(score).all()):
        raise FloatingPointError("fixed-channel outer score is incomplete")
    value, valid, total = _record_macro_auc(score, target, records)
    return {
        "record_macro_auc": value,
        "record_macro_valid_records": valid,
        "record_macro_total_records": total,
        "observation_auc": _auc(score, target),
        "selections": selections,
        "record_evidence": _record_score_evidence(score, target, records),
    }


def _fixed_prediction_score_null(
    prediction: np.ndarray,
    target: np.ndarray,
    records: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    observed, valid, total = _record_macro_ba(prediction, target, records)
    if valid != total or valid == 0:
        raise ValueError("score null requires eligible record-macro inputs")
    rng = np.random.default_rng(int(seed))
    grouped = _group_indices(records)
    draws = np.empty(int(replicates), dtype=float)
    for draw in range(int(replicates)):
        shuffled = target.copy()
        for indices in grouped.values():
            shuffled[indices] = shuffled[indices][rng.permutation(len(indices))]
        value, draw_valid, draw_total = _record_macro_ba(
            prediction, shuffled, records
        )
        if draw_valid != valid or draw_total != total:
            raise ValueError("score-null permutation changed record eligibility")
        draws[draw] = value
    return {
        "kind": "fixed_natural_crossfit_prediction_within_record_target_permutation",
        "seed": int(seed),
        "replicates": int(replicates),
        "observed_record_macro_balanced_accuracy": observed,
        "null_mean": float(draws.mean()),
        "null_upper_95": float(np.quantile(draws, 0.95)),
        "right_tail_p_value": float(
            (1 + np.count_nonzero(draws >= observed)) / (int(replicates) + 1)
        ),
    }


def _frozen_decoder_metrics(
    collection: Mapping[str, Any],
    *,
    family: str,
    feature_name: str,
    eligible_ids: Sequence[str],
) -> dict[str, Any]:
    logits = collection["state_logits"].numpy().astype(np.float64)
    values = collection["state_values"].numpy().astype(bool)
    masks = collection["state_feature_mask"].numpy().astype(bool)
    step_masks = collection["state_step_mask"].numpy().astype(bool)
    ids = [str(value) for value in collection["example_ids"]]
    families = [str(value).upper() for value in collection["families"]]
    names = collection["feature_names"]
    by_id = {example_id: index for index, example_id in enumerate(ids)}
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    records: list[np.ndarray] = []
    for example_id in sorted(str(value) for value in eligible_ids):
        index = by_id[example_id]
        if families[index] != family or feature_name not in names[index]:
            raise ValueError("frozen decoder row differs from fold ledger")
        feature = names[index].index(feature_name)
        keep = masks[index, :, :, feature] & step_masks[index, :, None]
        predictions.append(logits[index, :, :, feature][keep] >= 0.0)
        targets.append(values[index, :, :, feature][keep])
        records.append(np.repeat(example_id, int(keep.sum())))
    prediction = np.concatenate(predictions).astype(bool)
    target = np.concatenate(targets).astype(bool)
    record = np.concatenate(records).astype(object)
    record_ba, valid, total = _record_macro_ba(prediction, target, record)
    return {
        "record_macro_balanced_accuracy": record_ba,
        "record_macro_valid_records": valid,
        "record_macro_total_records": total,
        "observation_balanced_accuracy": _binary_ba(prediction, target),
        "record_evidence": _record_evidence(prediction, target, record),
    }


def _family_motion(collection: Mapping[str, Any], family: str) -> dict[str, float]:
    trajectory = collection["trajectory"].numpy().astype(np.float64)
    initial = collection["initial_payloads"].numpy().astype(np.float64)
    families = np.asarray([str(value).upper() for value in collection["families"]])
    selected = np.flatnonzero(families == family)
    previous = np.concatenate((initial[:, None], trajectory[:, :-1]), axis=1)
    scale = np.median(np.linalg.norm(initial, axis=-1), axis=1).clip(min=1.0e-8)
    relative = np.linalg.norm(trajectory - previous, axis=-1) / scale[:, None, None]
    active = collection["operation_active_target"].numpy().astype(bool)
    local = relative[selected]
    local_active = active[selected]
    return {
        "all_relative_delta_mean": float(local.mean()),
        "active_step_relative_delta_mean": float(local[local_active].mean())
        if local_active.any()
        else 0.0,
        "inactive_step_relative_delta_mean": float(local[~local_active].mean())
        if (~local_active).any()
        else 0.0,
    }


def audit_temporal_readout(
    collection: Mapping[str, Any],
    fold_ledger: Mapping[str, Any],
    *,
    ranks: Sequence[int],
    regularizations: Sequence[float],
    fit_null_seed: int,
    score_null_seed: int,
    score_null_replicates: int,
    source_visible_replay: bool,
) -> dict[str, Any]:
    """Measure D004R using the preflight-sealed target-only ledger."""

    if fold_ledger.get("passed") is not True or fold_ledger.get("identity") != contract.IDENTITY:
        raise ValueError("temporal fold ledger is not the registered sealed ledger")
    reports: dict[str, Any] = {}
    decision_rows: list[dict[str, Any]] = []
    adequacy = contract.TEMPORAL_READOUT["target_adequacy"]
    for family, feature_names in contract.HARD_FEATURES.items():
        motion = _family_motion(collection, family)
        feature_reports: dict[str, Any] = {}
        for offset, feature_name in enumerate(feature_names):
            cell = fold_ledger["cells"][family][feature_name]
            eligible_ids = cell["eligible_ids"]
            natural_x, natural_y, records = _feature_rows(
                collection,
                family=family,
                feature_name=feature_name,
                eligible_ids=eligible_ids,
            )
            natural = _cross_fitted_ridge(
                natural_x,
                natural_y,
                records,
                cell_ledger=cell,
                ranks=ranks,
                regularizations=regularizations,
            )
            nulls: dict[str, Any] = {}
            for transform_index, transform in enumerate(
                ("time_shuffle", "temporal_mean", "within_record_target_shuffle"),
                start=1,
            ):
                nx, ny, nrecords = _feature_rows(
                    collection,
                    family=family,
                    feature_name=feature_name,
                    eligible_ids=eligible_ids,
                    transform=transform,
                    seed=int(fit_null_seed) + offset * 10 + transform_index,
                )
                fitted = _cross_fitted_ridge(
                    nx,
                    ny,
                    nrecords,
                    cell_ledger=cell,
                    ranks=ranks,
                    regularizations=regularizations,
                )
                nulls[transform] = {
                    "record_macro_balanced_accuracy": fitted[
                        "record_macro_balanced_accuracy"
                    ],
                    "record_macro_auc": fitted["record_macro_auc"],
                    "record_macro_valid_records": fitted[
                        "record_macro_valid_records"
                    ],
                    "record_macro_total_records": fitted[
                        "record_macro_total_records"
                    ],
                    "observation_balanced_accuracy": fitted[
                        "observation_balanced_accuracy"
                    ],
                    "selections": fitted["selections"],
                    "record_evidence": fitted["record_evidence"],
                }
            strongest_null = max(
                float(report["record_macro_balanced_accuracy"])
                for report in nulls.values()
            )
            null_gap = float(natural["record_macro_balanced_accuracy"]) - strongest_null
            fixed_channel = _fixed_channel_record_macro_auc(
                natural_x,
                natural_y,
                records,
                cell_ledger=cell,
            )
            frozen = _frozen_decoder_metrics(
                collection,
                family=family,
                feature_name=feature_name,
                eligible_ids=eligible_ids,
            )
            score_null = _fixed_prediction_score_null(
                natural["predictions"],
                natural_y,
                records,
                seed=int(score_null_seed) + offset,
                replicates=score_null_replicates,
            )
            target_adequate = bool(
                source_visible_replay
                and cell["passed"] is True
                and int(cell["positive_observations"])
                >= int(adequacy["positive_observations_min"])
                and int(cell["negative_observations"])
                >= int(adequacy["negative_observations_min"])
                and int(cell["cross_step_changes"])
                >= int(adequacy["cross_step_changes_min"])
            )
            report = {
                "target_adequate": target_adequate,
                "positive_observations": int(cell["positive_observations"]),
                "negative_observations": int(cell["negative_observations"]),
                "cross_step_changes": int(cell["cross_step_changes"]),
                "eligible_records": int(cell["eligible_records"]),
                "ineligible_records": int(cell["ineligible_records"]),
                "ineligible_ids": list(cell["ineligible_ids"]),
                "fold_coverage_passed": cell["passed"] is True,
                "natural": {
                    "record_macro_balanced_accuracy": natural[
                        "record_macro_balanced_accuracy"
                    ],
                    "record_macro_auc": natural["record_macro_auc"],
                    "record_macro_valid_records": natural[
                        "record_macro_valid_records"
                    ],
                    "record_macro_total_records": natural[
                        "record_macro_total_records"
                    ],
                    "observation_balanced_accuracy": natural[
                        "observation_balanced_accuracy"
                    ],
                    "selections": natural["selections"],
                    "record_evidence": natural["record_evidence"],
                },
                "fixed_channel": fixed_channel,
                "frozen_decoder": frozen,
                "fit_nulls": nulls,
                "natural_minus_strongest_fit_null": null_gap,
                "score_null": score_null,
            }
            feature_reports[feature_name] = report
            decision_rows.append(
                {
                    "family": family,
                    "feature": feature_name,
                    "target_adequate": target_adequate,
                    "relative_motion": motion["active_step_relative_delta_mean"],
                    "separation_auc": fixed_channel["record_macro_auc"],
                    "crossfit_decoder_balanced_accuracy": natural[
                        "record_macro_balanced_accuracy"
                    ],
                    "frozen_decoder_balanced_accuracy": frozen[
                        "record_macro_balanced_accuracy"
                    ],
                    "fit_null_gap": null_gap,
                    "score_null_p_value": score_null["right_tail_p_value"],
                    "eligible_records": int(cell["eligible_records"]),
                    "fold_coverage_passed": cell["passed"] is True,
                }
            )
        reports[family] = {
            "motion": motion,
            "features": feature_reports,
            "answer_derived_features_excluded": list(
                contract.ANSWER_DERIVED_EXCLUDED[family]
            ),
        }
    return finite_json(
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.D004R.v1",
            "identity": contract.IDENTITY,
            "families": reports,
            "decision_rows": decision_rows,
            "source_visible_replay": {
                "passed": bool(source_visible_replay),
                "basis": "source_ast_materialize_target_regeneration_same_implementation",
                "boundary": (
                    "same implementation consistency only; not an independent semantic oracle"
                ),
            },
            "readout": {
                "kind": contract.TEMPORAL_READOUT["decoder"]["kind"],
                "ranks": [int(value) for value in ranks],
                "regularizations": [float(value) for value in regularizations],
                "outer_folds": contract.OUTER_FOLDS,
                "inner_folds": contract.INNER_FOLDS,
                "record_macro_primary": True,
                "observation_micro_diagnostic_only": True,
                "optimizer_steps": 0,
                "model_writes": 0,
            },
        }
    )


__all__ = ["audit_temporal_readout"]
