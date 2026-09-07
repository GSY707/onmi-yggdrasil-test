from __future__ import annotations

"""Memory-bounded NB with a qualified log fast path and exact fallback."""

from collections import Counter
from functools import cmp_to_key
import hashlib
import math
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

from ..learner.canonical import ContractError, exact_keys, require_label, require_mask, require_string, validate_unique_ids
from ..learner.models import analyze
from .scalable import _add_factor, _compare_expressions, _log_with_bound


PreparedRow = dict[str, Any]


def prepare_nb_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    analyzer: str,
    row_id: Callable[[Mapping[str, Any]], str],
    text: Callable[[Mapping[str, Any]], str],
    label: Callable[[Mapping[str, Any]], str],
) -> list[PreparedRow]:
    """Analyze once without retaining a second copy of the source text."""

    prepared: list[PreparedRow] = []
    seen: set[str] = set()
    for row in rows:
        identifier = require_string(row_id(row), code="invalid_id", field="row_id")
        if identifier in seen:
            raise ContractError("duplicate_id", f"duplicate row_id {identifier!r}")
        seen.add(identifier)
        source = require_string(text(row), code="invalid_text", field="text")
        prepared.append(
            {
                "row_id": identifier,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest().upper(),
                "label": label(row),
                "features": analyze(source, analyzer),
            }
        )
    if not prepared:
        raise ContractError("empty_fit", "prepared rows cannot be empty")
    return prepared


def fit_prepared_multinomial_nb(rows: Sequence[PreparedRow], labels: Sequence[str]) -> dict[str, Any]:
    """Fit accepted Laplace NB semantics as sparse integer counters."""

    if not isinstance(rows, list) or not rows:
        raise ContractError("empty_fit", "NB fit rows cannot be empty")
    label_order = tuple(labels)
    if not label_order or len(set(label_order)) != len(label_order) or any(not isinstance(item, str) for item in label_order):
        raise ContractError("invalid_labels", "NB labels must be unique strings")
    class_counts = {candidate: 0 for candidate in label_order}
    feature_counts: dict[str, Counter[str]] = {candidate: Counter() for candidate in label_order}
    vocabulary: set[str] = set()
    observed: set[str] = set()
    seen: set[str] = set()
    for row in rows:
        identifier = require_string(row["row_id"], code="invalid_id", field="row_id")
        if identifier in seen:
            raise ContractError("duplicate_id", f"duplicate row_id {identifier!r}")
        seen.add(identifier)
        candidate = require_label(row["label"], label_order)
        features = row["features"]
        if not isinstance(features, Counter):
            raise ContractError("invalid_features", "prepared features must be a Counter")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in features.values()):
            raise ContractError("invalid_features", "feature multiplicities must be positive integers")
        class_counts[candidate] += 1
        feature_counts[candidate].update(features)
        vocabulary.update(features)
        observed.add(candidate)
    if len(observed) < 2:
        raise ContractError("single_class_fit", "NB fit requires at least two observed labels")
    if not vocabulary:
        raise ContractError("empty_vocabulary", "NB fit vocabulary cannot be empty")
    vocabulary_order = tuple(sorted(vocabulary))
    model = {
        "labels": label_order,
        "class_counts": class_counts,
        "feature_totals": {candidate: sum(feature_counts[candidate].values()) for candidate in label_order},
        "feature_counts": feature_counts,
        "vocabulary": vocabulary_order,
        "vocabulary_lookup": frozenset(vocabulary_order),
    }
    _install_fast_projection(model)
    return model


def _install_fast_projection(model: MutableMapping[str, Any]) -> None:
    labels = tuple(model["labels"])
    vocabulary = tuple(model["vocabulary"])
    index = {feature: offset for offset, feature in enumerate(vocabulary)}
    matrix = np.zeros((len(vocabulary), len(labels)), dtype=np.float64)
    for label_index, label in enumerate(labels):
        for feature, count in model["feature_counts"][label].items():
            matrix[index[feature], label_index] = math.log(int(count) + 1)
    matrix.flags.writeable = False
    class_numerator = np.asarray([math.log(int(model["class_counts"][label]) + 1) for label in labels], dtype=np.float64)
    denominator = np.asarray(
        [math.log(int(model["feature_totals"][label]) + len(vocabulary)) for label in labels], dtype=np.float64
    )
    class_numerator.flags.writeable = False
    denominator.flags.writeable = False
    model["vocabulary_index"] = index
    model["log_feature_numerator_matrix"] = matrix
    model["log_class_numerator"] = class_numerator
    model["log_feature_denominator"] = denominator


def _fast_projection(model: Mapping[str, Any]) -> tuple[Mapping[str, int], np.ndarray, np.ndarray, np.ndarray]:
    if "log_feature_numerator_matrix" not in model:
        if not isinstance(model, MutableMapping):
            raise ContractError("invalid_model", "model lacks fast projection")
        _install_fast_projection(model)
    return (
        model["vocabulary_index"],
        model["log_feature_numerator_matrix"],
        model["log_class_numerator"],
        model["log_feature_denominator"],
    )


def _score_expression(model: Mapping[str, Any], candidate: str, used: Mapping[str, int]) -> dict[int, int]:
    labels = tuple(model["labels"])
    row_count = sum(int(model["class_counts"][label]) for label in labels)
    vocabulary_size = len(model["vocabulary"])
    exponents: Counter[int] = Counter()
    _add_factor(exponents, int(model["class_counts"][candidate]) + 1, 1)
    _add_factor(exponents, row_count + len(labels), -1)
    denominator = int(model["feature_totals"][candidate]) + vocabulary_size
    multiplicity_total = 0
    for feature, multiplicity in used.items():
        multiplicity = int(multiplicity)
        _add_factor(exponents, int(model["feature_counts"][candidate].get(feature, 0)) + 1, multiplicity)
        multiplicity_total += multiplicity
    _add_factor(exponents, denominator, -multiplicity_total)
    return dict(exponents)


def _validated_used(
    model: Mapping[str, Any], features: Mapping[str, int], valid_choice_mask: Sequence[bool]
) -> tuple[tuple[str, ...], list[str], Counter[str]]:
    labels = tuple(model.get("labels", ()))
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(label, str) for label in labels):
        raise ContractError("invalid_model", "model labels must be unique strings")
    mask = require_mask(list(valid_choice_mask), labels)
    if not isinstance(features, Mapping):
        raise ContractError("invalid_features", "features must be a mapping")
    vocabulary = model["vocabulary_lookup"]
    used: Counter[str] = Counter()
    for feature, count in features.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ContractError("invalid_features", "feature multiplicities must be positive integers")
        if feature in vocabulary:
            used[str(feature)] = count
    active = [label for label, enabled in zip(labels, mask, strict=True) if enabled]
    return labels, active, used


def _predict_exact_used(model: Mapping[str, Any], labels: tuple[str, ...], active: Sequence[str], used: Mapping[str, int]) -> str:
    expressions = {label: _score_expression(model, label, used) for label in active}
    logs = {label: _log_with_bound(expressions[label]) for label in active}

    def compare(left: str, right: str) -> int:
        relation, _ = _compare_expressions(expressions[left], expressions[right], logs[left], logs[right])
        if relation:
            return -relation
        return labels.index(left) - labels.index(right)

    return sorted(active, key=cmp_to_key(compare))[0]


def predict_prepared_multinomial_nb_exact(
    model: Mapping[str, Any], features: Mapping[str, int], valid_choice_mask: Sequence[bool]
) -> str:
    """Legacy exact path retained only as the qualification oracle/fallback."""

    labels, active, used = _validated_used(model, features, valid_choice_mask)
    return _predict_exact_used(model, labels, active, used)


def _bounded_log_scores(model: Mapping[str, Any], labels: tuple[str, ...], used: Mapping[str, int]) -> dict[str, tuple[float, float]]:
    vocabulary_index, matrix, class_numerator, denominator = _fast_projection(model)
    indexed = [(vocabulary_index[feature], int(multiplicity)) for feature, multiplicity in used.items()]
    if indexed:
        indexes = np.fromiter((row[0] for row in indexed), dtype=np.int64, count=len(indexed))
        multiplicities = np.fromiter((row[1] for row in indexed), dtype=np.float64, count=len(indexed))
        numerator = multiplicities @ matrix[indexes]
        multiplicity_total = float(multiplicities.sum())
    else:
        numerator = np.zeros(len(labels), dtype=np.float64)
        multiplicity_total = 0.0
    values = class_numerator + numerator - multiplicity_total * denominator
    scales = class_numerator + numerator + multiplicity_total * denominator
    term_count = len(indexed) + 2
    return {
        label: (
            float(values[index]),
            max(1e-12, (float(scales[index]) + 1.0) * (term_count + 1) * 3.2e-14),
        )
        for index, label in enumerate(labels)
    }


def predict_prepared_multinomial_nb_diagnostic(
    model: Mapping[str, Any], features: Mapping[str, int], valid_choice_mask: Sequence[bool]
) -> dict[str, Any]:
    labels, active, used = _validated_used(model, features, valid_choice_mask)
    all_scores = _bounded_log_scores(model, labels, used)
    scores = {label: all_scores[label] for label in active}
    approximate = max(active, key=lambda label: (scores[label][0], -labels.index(label)))
    lower = scores[approximate][0] - scores[approximate][1]
    ambiguous = [
        label
        for label in active
        if label != approximate and lower <= scores[label][0] + scores[label][1]
    ]
    if ambiguous:
        prediction = _predict_exact_used(model, labels, active, used)
        exact_fallback = True
    else:
        prediction = approximate
        exact_fallback = False
    return {
        "prediction": prediction,
        "exact_fallback": exact_fallback,
        "ambiguous_candidate_count": len(ambiguous),
        "active_label_count": len(active),
        "used_feature_count": len(used),
        "used_feature_multiplicity": sum(used.values()),
    }


def predict_prepared_multinomial_nb_batch(
    model: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, int]],
    valid_choice_masks: Sequence[Sequence[bool]],
    *,
    diagnostics: MutableMapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Score a batch with one sparse gather and C-level reductions."""

    if len(feature_rows) != len(valid_choice_masks) or not feature_rows:
        raise ContractError("invalid_batch", "feature rows and masks must be non-empty and aligned")
    labels = tuple(model.get("labels", ()))
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(label, str) for label in labels):
        raise ContractError("invalid_model", "model labels must be unique strings")
    vocabulary_index, matrix, class_numerator, denominator = _fast_projection(model)
    masks = [require_mask(list(mask), labels) for mask in valid_choice_masks]
    row_indexes: list[int] = []
    feature_indexes: list[int] = []
    multiplicities: list[int] = []
    for row_index, features in enumerate(feature_rows):
        if not isinstance(features, Mapping):
            raise ContractError("invalid_features", "features must be a mapping")
        for feature, count in features.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ContractError("invalid_features", "feature multiplicities must be positive integers")
            index = vocabulary_index.get(feature)
            if index is not None:
                row_indexes.append(row_index)
                feature_indexes.append(index)
                multiplicities.append(count)
    batch_size = len(feature_rows)
    if row_indexes:
        row_array = np.asarray(row_indexes, dtype=np.int64)
        feature_array = np.asarray(feature_indexes, dtype=np.int64)
        multiplicity_array = np.asarray(multiplicities, dtype=np.float64)
        numerator = np.empty((batch_size, len(labels)), dtype=np.float64)
        gathered = matrix[feature_array]
        for label_index in range(len(labels)):
            numerator[:, label_index] = np.bincount(
                row_array,
                weights=multiplicity_array * gathered[:, label_index],
                minlength=batch_size,
            )
        multiplicity_totals = np.bincount(row_array, weights=multiplicity_array, minlength=batch_size)
        feature_counts = np.bincount(row_array, minlength=batch_size)
    else:
        numerator = np.zeros((batch_size, len(labels)), dtype=np.float64)
        multiplicity_totals = np.zeros(batch_size, dtype=np.float64)
        feature_counts = np.zeros(batch_size, dtype=np.int64)
    values = class_numerator[None, :] + numerator - multiplicity_totals[:, None] * denominator[None, :]
    scales = class_numerator[None, :] + numerator + multiplicity_totals[:, None] * denominator[None, :]
    results: list[dict[str, Any]] = []
    fallback_count = 0
    for row_index, mask in enumerate(masks):
        active_indexes = [index for index, enabled in enumerate(mask) if enabled]
        approximate_index = max(active_indexes, key=lambda index: (float(values[row_index, index]), -index))
        term_count = int(feature_counts[row_index]) + 2
        bounds = {
            index: max(1e-12, (float(scales[row_index, index]) + 1.0) * (term_count + 1) * 3.2e-14)
            for index in active_indexes
        }
        lower = float(values[row_index, approximate_index]) - bounds[approximate_index]
        ambiguous = [
            index
            for index in active_indexes
            if index != approximate_index and lower <= float(values[row_index, index]) + bounds[index]
        ]
        if ambiguous:
            used = Counter(
                {
                    str(feature): int(count)
                    for feature, count in feature_rows[row_index].items()
                    if feature in vocabulary_index
                }
            )
            prediction = _predict_exact_used(model, labels, [labels[index] for index in active_indexes], used)
            exact_fallback = True
            fallback_count += 1
        else:
            prediction = labels[approximate_index]
            exact_fallback = False
        results.append(
            {
                "prediction": prediction,
                "exact_fallback": exact_fallback,
                "ambiguous_candidate_count": len(ambiguous),
                "active_label_count": len(active_indexes),
                "used_feature_count": int(feature_counts[row_index]),
                "used_feature_multiplicity": int(multiplicity_totals[row_index]),
            }
        )
    if diagnostics is not None:
        diagnostics["prediction_count"] = diagnostics.get("prediction_count", 0) + batch_size
        diagnostics["exact_fallback_count"] = diagnostics.get("exact_fallback_count", 0) + fallback_count
        diagnostics["batch_count"] = diagnostics.get("batch_count", 0) + 1
    return results


def predict_prepared_multinomial_nb(
    model: Mapping[str, Any],
    features: Mapping[str, int],
    valid_choice_mask: Sequence[bool],
    *,
    diagnostics: MutableMapping[str, int] | None = None,
) -> str:
    result = predict_prepared_multinomial_nb_diagnostic(model, features, valid_choice_mask)
    if diagnostics is not None:
        diagnostics["prediction_count"] = diagnostics.get("prediction_count", 0) + 1
        if result["exact_fallback"]:
            diagnostics["exact_fallback_count"] = diagnostics.get("exact_fallback_count", 0) + 1
    return str(result["prediction"])


def fit_sparse_multinomial_nb(rows: Sequence[Mapping[str, Any]], analyzer: str, labels: Sequence[str]) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise ContractError("empty_fit", "NB fit rows cannot be empty")
    for row in rows:
        exact_keys(row, ("row_id", "text", "label"), code="nb_fit_row_schema")
    validate_unique_ids(rows)
    prepared = prepare_nb_rows(
        rows,
        analyzer=analyzer,
        row_id=lambda row: row["row_id"],
        text=lambda row: row["text"],
        label=lambda row: row["label"],
    )
    return fit_prepared_multinomial_nb(prepared, labels)


def predict_sparse_multinomial_nb(
    model: Mapping[str, Any], text: str, analyzer: str, valid_choice_mask: Sequence[bool]
) -> str:
    return predict_prepared_multinomial_nb(model, analyze(text, analyzer), valid_choice_mask)


__all__ = [
    "fit_prepared_multinomial_nb",
    "fit_sparse_multinomial_nb",
    "predict_prepared_multinomial_nb",
    "predict_prepared_multinomial_nb_batch",
    "predict_prepared_multinomial_nb_diagnostic",
    "predict_prepared_multinomial_nb_exact",
    "predict_sparse_multinomial_nb",
    "prepare_nb_rows",
]
