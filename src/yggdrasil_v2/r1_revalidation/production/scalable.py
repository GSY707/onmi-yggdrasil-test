from __future__ import annotations

"""Compact exact-rational ordering for long-source multinomial NB audits."""

from collections import Counter
from functools import cmp_to_key, lru_cache
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from ..learner.canonical import ContractError, require_mask, require_string, sha256_json
from ..learner.models import analyze, fit_multinomial_nb


@lru_cache(maxsize=None)
def _factorize(number: int) -> tuple[tuple[int, int], ...]:
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ContractError("invalid_factor", f"factor base must be positive, got {number!r}")
    factors: list[tuple[int, int]] = []
    remainder = number
    divisor = 2
    while divisor * divisor <= remainder:
        exponent = 0
        while remainder % divisor == 0:
            remainder //= divisor
            exponent += 1
        if exponent:
            factors.append((divisor, exponent))
        divisor = 3 if divisor == 2 else divisor + 2
    if remainder > 1:
        factors.append((remainder, 1))
    return tuple(factors)


def _add_factor(exponents: Counter[int], base: int, exponent: int) -> None:
    if exponent == 0 or base == 1:
        return
    for prime, multiplicity in _factorize(base):
        exponents[prime] += multiplicity * exponent
        if exponents[prime] == 0:
            del exponents[prime]


def _score_expression(model: Mapping[str, Any], label: str, used: Mapping[str, int]) -> dict[int, int]:
    labels = tuple(model["labels"])
    row_count = sum(int(model["class_counts"][candidate]) for candidate in labels)
    vocabulary_size = len(model["vocabulary"])
    exponents: Counter[int] = Counter()
    _add_factor(exponents, int(model["class_counts"][label]) + 1, 1)
    _add_factor(exponents, row_count + len(labels), -1)
    denominator = int(model["feature_totals"][label]) + vocabulary_size
    for feature, multiplicity in used.items():
        _add_factor(exponents, int(model["feature_counts"][label][feature]) + 1, int(multiplicity))
        _add_factor(exponents, denominator, -int(multiplicity))
    return dict(sorted(exponents.items()))


def _expression_digest(expression: Mapping[int, int]) -> str:
    payload = json.dumps([[prime, expression[prime]] for prime in sorted(expression)], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def _log_with_bound(expression: Mapping[int, int]) -> tuple[float, float]:
    terms = [exponent * math.log(prime) for prime, exponent in expression.items()]
    value = math.fsum(terms)
    scale = math.fsum(abs(term) for term in terms)
    bound = max(1e-12, (scale + 1.0) * max(1, len(terms)) * 4e-15)
    return value, bound


def _exact_compare(left: Mapping[int, int], right: Mapping[int, int]) -> int:
    delta: Counter[int] = Counter(left)
    delta.subtract(right)
    delta = Counter({prime: exponent for prime, exponent in delta.items() if exponent})
    if not delta:
        return 0
    numerator = 1
    denominator = 1
    for prime, exponent in delta.items():
        if exponent > 0:
            numerator *= pow(prime, exponent)
        else:
            denominator *= pow(prime, -exponent)
    return (numerator > denominator) - (numerator < denominator)


def _compare_expressions(
    left: Mapping[int, int],
    right: Mapping[int, int],
    left_log: tuple[float, float],
    right_log: tuple[float, float],
) -> tuple[int, bool]:
    if left == right:
        return 0, False
    difference = left_log[0] - right_log[0]
    uncertainty = left_log[1] + right_log[1]
    if abs(difference) > uncertainty:
        return ((difference > 0) - (difference < 0)), False
    return _exact_compare(left, right), True


def compact_predict_multinomial_nb(
    model: Mapping[str, Any],
    text: str,
    valid_choice_mask: Sequence[bool],
) -> dict[str, Any]:
    """Predict without materializing or decimal-stringifying enormous Fractions."""

    labels = tuple(model.get("labels", ()))
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(label, str) for label in labels):
        raise ContractError("invalid_model", "model labels must be unique strings")
    mask = require_mask(list(valid_choice_mask), labels)
    source = require_string(text, code="invalid_text", field="text")
    features = analyze(source, model["analyzer"])
    vocabulary = set(model["vocabulary"])
    used = Counter({feature: count for feature, count in features.items() if feature in vocabulary})
    expressions = {label: _score_expression(model, label, used) for label in labels}
    logs = {label: _log_with_bound(expressions[label]) for label in labels}
    exact_fallback_pairs: set[tuple[str, str]] = set()

    def compare(left: str, right: str) -> int:
        relation, exact = _compare_expressions(expressions[left], expressions[right], logs[left], logs[right])
        if exact:
            exact_fallback_pairs.add(tuple(sorted((left, right))))
        if relation:
            return -relation
        return labels.index(left) - labels.index(right)

    active = [label for label, enabled in zip(labels, mask, strict=True) if enabled]
    order = sorted(active, key=cmp_to_key(compare))
    used_projection = {feature: used[feature] for feature in sorted(used)}
    return {
        "prediction": order[0],
        "score_order": order,
        "score_expression_sha256": {label: _expression_digest(expressions[label]) for label in labels},
        "used_feature_count": len(used_projection),
        "used_feature_multiplicity": sum(used_projection.values()),
        "used_features_sha256": sha256_json(used_projection),
        "exact_fallback_pair_count": len(exact_fallback_pairs),
    }


def fit_scalable_multinomial_nb(
    rows: Sequence[Mapping[str, Any]],
    analyzer: str,
    labels: Sequence[str],
) -> dict[str, Any]:
    """Reuse the accepted v10 fit semantics without modifying its runtime."""

    return fit_multinomial_nb(rows, analyzer, labels)
