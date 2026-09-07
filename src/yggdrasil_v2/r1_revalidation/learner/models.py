from __future__ import annotations

"""Deterministic categorical and multinomial Naive Bayes learners."""

from collections import Counter
from fractions import Fraction
import re
from typing import Any, Mapping, Sequence

from .canonical import (
    ContractError,
    LABELS,
    choose_label,
    exact_keys,
    normalize_char_text,
    normalize_source,
    require_label,
    require_mask,
    require_string,
    sha256_json,
    validate_unique_ids,
)


_WORD_RE = re.compile(r"(?u)\b\w+\b")


def word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_source(text).lower())


def character_ngrams(text: str, minimum: int = 3, maximum: int = 5) -> list[str]:
    normalized = normalize_char_text(text)
    grams: list[str] = []
    for width in range(minimum, maximum + 1):
        grams.extend(normalized[index:index + width] for index in range(max(0, len(normalized) - width + 1)))
    return grams


def analyze(text: str, analyzer: str) -> Counter[str]:
    if analyzer == "word":
        return Counter(word_tokens(text))
    if analyzer == "char_3_5":
        return Counter(character_ngrams(text))
    raise ContractError("unknown_analyzer", f"unsupported analyzer {analyzer!r}")


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def fit_conditional_majority(rows: Sequence[Mapping[str, Any]], labels: Sequence[str] = LABELS) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise ContractError("empty_fit", "categorical fit rows cannot be empty")
    for row in rows:
        exact_keys(row, ("row_id", "category", "label"), code="categorical_row_schema")
    validate_unique_ids(rows)
    observed: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        category = require_string(row["category"], code="invalid_category", field="category")
        label = require_label(row["label"], labels)
        observed.add(label)
        counts.setdefault(category, {candidate: 0 for candidate in labels})[label] += 1
    if len(observed) < 2:
        raise ContractError("single_class_fit", "categorical fit requires at least two observed labels")
    normalized_rows = [
        {"row_id": row["row_id"], "category": row["category"], "label": row["label"]}
        for row in sorted(rows, key=lambda item: item["row_id"])
    ]
    return {
        "labels": list(labels),
        "counts": {category: counts[category] for category in sorted(counts)},
        "fit_row_fingerprint": sha256_json(normalized_rows),
    }


def predict_conditional_majority(
    model: Mapping[str, Any],
    category: str,
    valid_choice_mask: Sequence[bool],
) -> dict[str, Any]:
    labels = tuple(model["labels"])
    mask = require_mask(list(valid_choice_mask), labels)
    raw_counts = model["counts"].get(category, {label: 0 for label in labels})
    scores = {label: int(raw_counts[label]) + 1 for label in labels}
    return {
        "category": category,
        "scores": scores,
        "prediction": choose_label(scores, mask, labels),
    }


def fit_multinomial_nb(
    rows: Sequence[Mapping[str, Any]],
    analyzer: str,
    labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise ContractError("empty_fit", "NB fit rows cannot be empty")
    for row in rows:
        exact_keys(row, ("row_id", "text", "label"), code="nb_fit_row_schema")
    validate_unique_ids(rows)
    class_counts = {label: 0 for label in labels}
    feature_counts: dict[str, Counter[str]] = {label: Counter() for label in labels}
    observed: set[str] = set()
    normalized_rows: list[dict[str, str]] = []
    vocabulary_set: set[str] = set()
    for row in rows:
        row_id = require_string(row["row_id"], code="invalid_id", field="row_id")
        text = require_string(row["text"], code="invalid_text", field="text")
        label = require_label(row["label"], labels)
        features = analyze(text, analyzer)
        class_counts[label] += 1
        feature_counts[label].update(features)
        vocabulary_set.update(features)
        observed.add(label)
        normalized_rows.append({"row_id": row_id, "text": normalize_source(text), "label": label})
    if len(observed) < 2:
        raise ContractError("single_class_fit", "NB fit requires at least two observed labels")
    vocabulary = sorted(vocabulary_set)
    if not vocabulary:
        raise ContractError("empty_vocabulary", "NB fit vocabulary cannot be empty")
    matrix = {
        label: {feature: int(feature_counts[label].get(feature, 0)) for feature in vocabulary}
        for label in labels
    }
    feature_totals = {label: sum(matrix[label].values()) for label in labels}
    normalized_rows.sort(key=lambda item: item["row_id"])
    return {
        "analyzer": analyzer,
        "labels": list(labels),
        "class_counts": class_counts,
        "feature_totals": feature_totals,
        "feature_counts": matrix,
        "feature_counts_sha256": sha256_json(matrix),
        "vocabulary": vocabulary,
        "vocabulary_sha256": sha256_json(vocabulary),
        "fit_row_fingerprint": sha256_json(normalized_rows),
    }


def predict_multinomial_nb(
    model: Mapping[str, Any],
    text: str,
    valid_choice_mask: Sequence[bool],
) -> dict[str, Any]:
    labels = tuple(model["labels"])
    mask = require_mask(list(valid_choice_mask), labels)
    features = analyze(text, model["analyzer"])
    vocabulary = set(model["vocabulary"])
    used = Counter({feature: count for feature, count in features.items() if feature in vocabulary})
    row_count = sum(int(model["class_counts"][label]) for label in labels)
    class_count = len(labels)
    vocabulary_size = len(vocabulary)
    scores: dict[str, Fraction] = {}
    for label in labels:
        score = Fraction(int(model["class_counts"][label]) + 1, row_count + class_count)
        denominator = int(model["feature_totals"][label]) + vocabulary_size
        for feature, multiplicity in used.items():
            numerator = int(model["feature_counts"][label][feature]) + 1
            score *= Fraction(numerator, denominator) ** multiplicity
        scores[label] = score
    prediction = choose_label(scores, mask, labels)
    return {
        "prediction": prediction,
        "scores": {label: _fraction_text(scores[label]) for label in labels},
        "used_features": {feature: used[feature] for feature in sorted(used)},
    }


def evaluate_categorical_case(case: Mapping[str, Any], labels: Sequence[str] = LABELS) -> dict[str, Any]:
    model = fit_conditional_majority(case["fit_rows"], labels)
    predictions = []
    for row in case["eval_rows"]:
        exact_keys(row, ("row_id", "category", "valid_choice_mask"), code="categorical_eval_row_schema")
        row_id = require_string(row["row_id"], code="invalid_id", field="row_id")
        category = require_string(row["category"], code="invalid_category", field="category")
        result = predict_conditional_majority(model, category, row["valid_choice_mask"])
        predictions.append({"row_id": row_id, **result})
    predictions.sort(key=lambda item: item["row_id"])
    return {"case_id": case["case_id"], "model": model, "predictions": predictions}


def evaluate_nb_case(case: Mapping[str, Any], labels: Sequence[str] = LABELS) -> dict[str, Any]:
    model = fit_multinomial_nb(case["fit_rows"], case["analyzer"], labels)
    predictions = []
    seen: set[str] = set()
    for row in case["eval_rows"]:
        exact_keys(row, ("row_id", "text", "valid_choice_mask"), code="nb_eval_row_schema")
        row_id = require_string(row["row_id"], code="invalid_id", field="row_id")
        if row_id in seen:
            raise ContractError("duplicate_id", f"duplicate eval row_id {row_id!r}")
        seen.add(row_id)
        result = predict_multinomial_nb(model, row["text"], row["valid_choice_mask"])
        predictions.append({"row_id": row_id, **result})
    predictions.sort(key=lambda item: item["row_id"])
    return {"case_id": case["case_id"], "model": model, "predictions": predictions}

