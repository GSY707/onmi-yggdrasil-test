from __future__ import annotations

"""Family-primary D005R classification for the temporal successor."""

from typing import Any, Mapping, Sequence

from . import contract


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def classify_axis_c(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify every family without pooling its feature failures away."""

    thresholds = contract.TEMPORAL_READOUT
    result: dict[str, Any] = {}
    missing: list[str] = []
    for family, expected_features in contract.HARD_FEATURES.items():
        rows = [
            row
            for row in records
            if str(row.get("family", "")).upper() == family
        ]
        observed = [str(row.get("feature", "")) for row in rows]
        if sorted(observed) != sorted(expected_features) or len(set(observed)) != len(observed):
            missing.append(family)
            continue
        target_adequate = all(
            row.get("target_adequate") is True
            and row.get("fold_coverage_passed") is True
            for row in rows
        )
        motion = min(_number(row, "relative_motion") for row in rows)
        separation = min(_number(row, "separation_auc") for row in rows)
        crossfit = min(
            _number(row, "crossfit_decoder_balanced_accuracy") for row in rows
        )
        frozen = min(
            _number(row, "frozen_decoder_balanced_accuracy") for row in rows
        )
        fit_null_gap = min(_number(row, "fit_null_gap") for row in rows)
        score_null_p = max(_number(row, "score_null_p_value") for row in rows)
        motion_signal = bool(
            motion >= float(thresholds["motion_relative_delta_floor"])
            and separation
            >= float(thresholds["fixed_channel_record_macro_auc_floor"])
        )
        readout_signal = bool(
            crossfit
            >= float(thresholds["crossfit_record_macro_balanced_accuracy_floor"])
            and fit_null_gap
            >= float(thresholds["readout_minus_strongest_fit_null_floor"])
            and score_null_p <= float(thresholds["score_null_max_p_value"])
        )
        frozen_sufficient = bool(
            frozen
            >= float(
                thresholds["frozen_decoder_record_macro_balanced_accuracy_floor"]
            )
        )
        static_null_like = bool(
            max(_number(row, "fit_null_gap") for row in rows)
            <= float(thresholds["static_null_gap_max"])
        )
        if not target_adequate:
            label = "TARGET_INSUFFICIENT"
        elif not motion_signal and not readout_signal and static_null_like:
            label = "LATENT_NOT_FORMED"
        elif motion_signal and readout_signal and not frozen_sufficient:
            label = "READOUT_INSUFFICIENT"
        elif motion_signal and readout_signal and frozen_sufficient:
            label = "INCONCLUSIVE"
        else:
            label = "MIXED"
        if label not in contract.ALLOWED_AXIS_C:
            raise RuntimeError("axis-C classifier emitted an unregistered label")
        result[family] = {
            "classification": label,
            "n": len(rows),
            "features": sorted(observed),
            "target_adequate": target_adequate,
            "minimum_active_relative_motion": motion,
            "minimum_fixed_channel_record_macro_auc": separation,
            "minimum_crossfit_record_macro_balanced_accuracy": crossfit,
            "minimum_frozen_decoder_record_macro_balanced_accuracy": frozen,
            "minimum_natural_minus_strongest_fit_null": fit_null_gap,
            "maximum_score_null_p_value": score_null_p,
            "motion_signal": motion_signal,
            "readout_signal": readout_signal,
            "frozen_decoder_sufficient": frozen_sufficient,
            "static_null_like": static_null_like,
        }
    family_complete = not missing and set(result) == set(contract.FAMILIES)
    counts = {
        label: sum(cell["classification"] == label for cell in result.values())
        for label in contract.ALLOWED_AXIS_C
        if any(cell["classification"] == label for cell in result.values())
    }
    overall_label = (
        next(iter(counts)) if family_complete and len(counts) == 1 else "MIXED_FAMILY_SUMMARY"
    )
    return {
        "families": result,
        "required_families": list(contract.FAMILIES),
        "missing_families": missing,
        "family_complete": family_complete,
        "gate_scope": "family_only",
        "overall": {
            "role": "summary_only",
            "label": overall_label,
            "family_counts": counts,
        },
    }


__all__ = ["classify_axis_c"]
