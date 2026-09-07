from __future__ import annotations

"""Qualified G09 decision topology shared by synthetic power and production."""

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Mapping


BASELINES = (
    "majority",
    "first_valid_label",
    "last_choice_line",
    "token_length_bucket",
    "question_objective_word_nb",
    "full_text_word_nb",
    "full_text_char_3_5_nb",
)
FAMILY_SPLITS = {
    "ERE": ("train", "validation", "composition_ood", "length_ood", "entity_ood", "language_ood", "causal_pairs"),
    "CPS": ("train", "validation", "composition_ood", "horizon_ood", "distractor_ood", "language_ood", "causal_pairs"),
}
SOURCE_CELL_COUNT = 98
SOURCE_FAMILY_CELL_COUNT = 14

# These are task/mask profiles, not v14 observed shortcut accuracies.  ERE's
# mixed three/four-choice splits use the design expectation 13/48.
SYNTHETIC_RANDOM_PROFILE = {
    "ERE": {
        "train": 13 / 48,
        "validation": 13 / 48,
        "composition_ood": 1 / 4,
        "length_ood": 1 / 4,
        "entity_ood": 1 / 4,
        "language_ood": 13 / 48,
        "causal_pairs": 13 / 48,
    },
    "CPS": {
        "train": 1 / 6,
        "validation": 1 / 6,
        "composition_ood": 1 / 7,
        "horizon_ood": 1 / 6,
        "distractor_ood": 1 / 9,
        "language_ood": 1 / 6,
        "causal_pairs": 1 / 6,
    },
}


@dataclass(frozen=True)
class CellSpec:
    key: str
    family: str
    split: str
    baseline: str
    total: int
    mean_random: float


@dataclass(frozen=True)
class DecisionConfig:
    alpha: float = 0.05
    source_comparisons: int = SOURCE_CELL_COUNT
    source_excess_ceiling: float = 0.10
    aggregate_comparisons: int = SOURCE_FAMILY_CELL_COUNT
    aggregate_excess_ceiling: float = 0.05


DEFAULT_CONFIG = DecisionConfig()


def source_cell_key(family: str, split: str, baseline: str) -> str:
    return f"{family}/{split}/{baseline}"


def source_cell_specs(*, train_count: int, heldout_count: int) -> tuple[CellSpec, ...]:
    if train_count <= 0 or heldout_count <= 0:
        raise ValueError("cell counts must be positive")
    result: list[CellSpec] = []
    for family in ("ERE", "CPS"):
        for split in FAMILY_SPLITS[family]:
            total = train_count if split == "train" else heldout_count
            mean_random = SYNTHETIC_RANDOM_PROFILE[family][split]
            for baseline in BASELINES:
                result.append(CellSpec(source_cell_key(family, split, baseline), family, split, baseline, total, mean_random))
    if len(result) != SOURCE_CELL_COUNT or len({row.key for row in result}) != SOURCE_CELL_COUNT:
        raise AssertionError("invalid source-cell topology")
    return tuple(result)


def wilson_upper(successes: int, total: int, comparisons: int, *, alpha: float = 0.05) -> dict[str, float | int]:
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("total must be an integer")
    if not 0 <= successes <= total or total <= 0 or comparisons <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid Wilson inputs")
    adjusted_alpha = alpha / comparisons
    z = NormalDist().inv_cdf(1.0 - adjusted_alpha)
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
    return {
        "successes": successes,
        "total": total,
        "accuracy": proportion,
        "alpha": alpha,
        "comparisons": comparisons,
        "adjusted_alpha": adjusted_alpha,
        "z": z,
        "upper": min(1.0, (center + radius) / denominator),
    }


def wilson_lower(successes: int, total: int, *, alpha: float = 0.05) -> float:
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise ValueError("successes must be an integer")
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("total must be an integer")
    if not 0 <= successes <= total or total <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid Wilson inputs")
    z = NormalDist().inv_cdf(1.0 - alpha)
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = proportion + z2 / (2.0 * total)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
    return max(0.0, (center - radius) / denominator)


def _validate_exact_keys(name: str, observed: Mapping[str, Any], expected: set[str]) -> None:
    actual = set(observed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} topology mismatch: missing={missing[:3]} extra={extra[:3]}")


def evaluate_g09_counts(
    *,
    successes: Mapping[str, int],
    totals: Mapping[str, int],
    random_mass: Mapping[str, float],
    complete: Mapping[str, bool],
    config: DecisionConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Apply the exact 98-cell/14-aggregate production decision."""

    if config != DEFAULT_CONFIG:
        raise ValueError("non-canonical G09 decision configuration")
    expected = {
        source_cell_key(family, split, baseline)
        for family in FAMILY_SPLITS
        for split in FAMILY_SPLITS[family]
        for baseline in BASELINES
    }
    _validate_exact_keys("successes", successes, expected)
    _validate_exact_keys("totals", totals, expected)
    _validate_exact_keys("random_mass", random_mass, expected)
    _validate_exact_keys("complete", complete, expected)
    failures: list[str] = []
    cells: dict[str, Any] = {}
    for key in sorted(expected):
        success = successes[key]
        total = totals[key]
        mass = random_mass[key]
        if isinstance(success, bool) or not isinstance(success, int):
            raise ValueError(f"invalid success count for {key}")
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError(f"invalid total for {key}")
        if not isinstance(mass, (int, float)) or isinstance(mass, bool) or not math.isfinite(float(mass)):
            raise ValueError(f"invalid random mass for {key}")
        if not 0 <= success <= total or not 0.0 < float(mass) < total:
            raise ValueError(f"out-of-range cell values for {key}")
        mean_random = float(mass) / total
        if not isinstance(complete[key], bool):
            raise ValueError(f"invalid completeness flag for {key}")
        threshold = mean_random + config.source_excess_ceiling
        bound = wilson_upper(success, total, config.source_comparisons, alpha=config.alpha)
        passed = complete[key] and bound["accuracy"] <= threshold + 1e-12 and bound["upper"] <= threshold + 1e-12
        cells[key] = {
            **bound,
            "mean_random": mean_random,
            "random_mass": float(mass),
            "point_excess": bound["accuracy"] - mean_random,
            "gate_threshold": threshold,
            "complete": complete[key],
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"shortcut:{key}:accuracy={bound['accuracy']:.6f}:upper={bound['upper']:.6f}:limit={threshold:.6f}"
            )

    aggregates: dict[str, Any] = {}
    for family in ("ERE", "CPS"):
        for baseline in BASELINES:
            keys = [source_cell_key(family, split, baseline) for split in FAMILY_SPLITS[family] if split != "train"]
            success = sum(successes[key] for key in keys)
            total = sum(totals[key] for key in keys)
            mass = math.fsum(float(random_mass[key]) for key in keys)
            mean_random = mass / total
            threshold = mean_random + config.aggregate_excess_ceiling
            bound = wilson_upper(success, total, config.aggregate_comparisons, alpha=config.alpha)
            passed = bound["accuracy"] <= threshold + 1e-12 and bound["upper"] <= threshold + 1e-12
            key = f"{family}/{baseline}"
            aggregates[key] = {
                **bound,
                "mean_random": mean_random,
                "random_mass": mass,
                "point_excess": bound["accuracy"] - mean_random,
                "gate_threshold": threshold,
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"shortcut_aggregate:{key}:accuracy={bound['accuracy']:.6f}:upper={bound['upper']:.6f}:limit={threshold:.6f}"
                )

    if len(cells) != SOURCE_CELL_COUNT or len(aggregates) != SOURCE_FAMILY_CELL_COUNT:
        raise AssertionError("G09 result topology changed")
    return {
        "passed": not failures,
        "cells": cells,
        "family_aggregates": aggregates,
        "failures": failures,
        "topology": {
            "source_cell_count": SOURCE_CELL_COUNT,
            "family_aggregate_count": SOURCE_FAMILY_CELL_COUNT,
            "baseline_count": len(BASELINES),
        },
    }


__all__ = [
    "BASELINES",
    "FAMILY_SPLITS",
    "SOURCE_CELL_COUNT",
    "SOURCE_FAMILY_CELL_COUNT",
    "SYNTHETIC_RANDOM_PROFILE",
    "CellSpec",
    "DecisionConfig",
    "DEFAULT_CONFIG",
    "evaluate_g09_counts",
    "source_cell_key",
    "source_cell_specs",
    "wilson_lower",
    "wilson_upper",
]
