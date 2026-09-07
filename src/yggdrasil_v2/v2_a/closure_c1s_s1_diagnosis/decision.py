from __future__ import annotations

"""Explicit, family-first classifications for the C1S S1 diagnosis.

These functions do not choose thresholds.  Every scientific threshold is a
required value in the caller's mapping, making the frozen contract the only
place where a conclusion is registered.  A returned ``overall`` value is a
descriptive roll-up only; it is never a Gate.
"""

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

from .metrics import finite_json, paired_cluster_bootstrap, wilson_interval


REQUIRED_FAMILIES = ("CPS", "ERE")


def _threshold(thresholds: Mapping[str, Any], key: str) -> float:
    if key not in thresholds:
        raise KeyError(f"missing registered threshold: {key}")
    value = float(thresholds[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite threshold: {key}")
    return value


def _rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = list(records)
    if not result or any(not isinstance(row, Mapping) for row in result):
        raise ValueError("records must be a non-empty sequence of mappings")
    return result


def _families(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        family = row.get("family")
        if family is None:
            raise ValueError("each record requires family")
        grouped[str(family).upper()].append(row)
    return dict(sorted(grouped.items()))


def _bool(row: Mapping[str, Any], *names: str) -> bool:
    for name in names:
        if name in row:
            value = row[name]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) in (0.0, 1.0):
                return bool(value)
            raise ValueError(f"{names[0]} must be boolean")
    raise ValueError(f"record is missing {names[0]}")


def _number(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in row:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"{name} is non-finite")
            return value
    raise ValueError(f"record is missing {names[0]}")


def _accuracy(values: Sequence[bool]) -> dict[str, Any]:
    return wilson_interval(sum(values), len(values))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty family")
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise ValueError("non-finite mean")
    return float(result)


def _overall_labels(labels: Mapping[str, str]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for label in labels.values():
        counts[label] += 1
    unique = sorted(counts)
    # Deliberately not a Gate: this only says whether family conclusions agree.
    overall = unique[0] if len(unique) == 1 else "MIXED_FAMILY_RESULTS"
    return {"label": overall, "family_counts": dict(sorted(counts.items())), "families": len(labels)}


def _family_report(records: Sequence[Mapping[str, Any]], classify: Any) -> dict[str, Any]:
    grouped = _families(_rows(records))
    labels: dict[str, str] = {}
    reports: dict[str, Any] = {}
    for family, family_rows in grouped.items():
        report = classify(family_rows)
        labels[family] = str(report["classification"])
        reports[family] = finite_json(report)
    missing = [family for family in REQUIRED_FAMILIES if family not in reports]
    for family in missing:
        reports[family] = {
            "classification": "INCONCLUSIVE",
            "n": 0,
            "reason": "required_family_cell_missing",
        }
        labels[family] = "INCONCLUSIVE"
    return finite_json({
        "families": dict(sorted(reports.items())),
        "overall": _overall_labels(labels),
        "gate_scope": "family_only",
        "required_families": list(REQUIRED_FAMILIES),
        "missing_families": missing,
        "family_complete": not missing,
    })


def classify_loss_gate_alignment(
    gate_rows: Sequence[Mapping[str, Any]],
    loss_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Check whether failed family Gates have an active, registered loss.

    A loss absent from the objective, or present but below the registered
    activity/gradient floors, is an objective--Gate mismatch.  This is a
    diagnosis, never permission to change the training objective in-place.
    """

    gates = _rows(gate_rows)
    losses = list(loss_rows)
    if any(not isinstance(row, Mapping) for row in losses):
        raise ValueError("loss_rows must contain mappings")
    minimum_activity = _threshold(thresholds, "min_active_fraction")
    minimum_gradient = _threshold(thresholds, "min_gradient_norm")
    grouped_gates = _families(gates)
    grouped_losses: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in losses:
        family = str(row.get("family", "")).upper()
        gate = str(row.get("gate", row.get("name", "")))
        if not family or not gate:
            raise ValueError("each loss row requires family and gate/name")
        grouped_losses[(family, gate)] = row

    def one_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        family = str(rows[0]["family"]).upper()
        failed = [row for row in rows if row.get("passed") is False]
        cells: list[dict[str, Any]] = []
        mismatch = False
        for gate in failed:
            name = str(gate.get("gate", gate.get("name", "")))
            loss = grouped_losses.get((family, name))
            active_fraction = float(loss.get("active_fraction", 0.0)) if loss is not None else 0.0
            gradient = float(loss.get("gradient_norm", 0.0)) if loss is not None else 0.0
            present = bool(loss is not None and loss.get("present", True))
            metric_equivalent = bool(
                loss is not None and loss.get("metric_equivalent", True)
            )
            if not math.isfinite(active_fraction) or not math.isfinite(gradient):
                raise ValueError("loss activity/gradient must be finite")
            effective = present and active_fraction >= minimum_activity and gradient >= minimum_gradient
            metric_aligned = bool(metric_equivalent)
            aligned = (
                present
                and effective
                and metric_aligned
            )
            # Non-equivalence is a warning, not proof of mismatch.  Confirm a
            # mismatch only when the failed Gate has no effective pressure:
            # the loss is absent, inactive/saturated, or has no gradient.
            confirmed = not present or not effective
            mismatch |= confirmed
            cells.append({"gate": name, "loss_present": present, "metric_equivalent": metric_equivalent, "active_fraction": active_fraction, "gradient_norm": gradient, "effective_loss": effective, "aligned": aligned, "confirmed_mismatch": confirmed})
        if not rows:
            classification = "INCONCLUSIVE"
        elif mismatch:
            classification = "OBJECTIVE_GATE_MISMATCH_CONFIRMED"
        elif any(not bool(cell["metric_equivalent"]) for cell in cells):
            classification = "OBJECTIVE_NOT_EQUIVALENT"
        else:
            classification = "OBJECTIVE_ALIGNED"
        return {"classification": classification, "n": len(rows), "failed_gates": len(failed), "checks": cells}

    return _family_report(gates, one_family)


def classify_a_axis(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Classify H0 direct, core-required, residual, or mixed per family."""

    h0_direct_min = _threshold(thresholds, "h0_direct_point_min")
    h0_direct_lower = _threshold(thresholds, "h0_direct_wilson_lower_min")
    h0_low_max = _threshold(thresholds, "h0_low_point_max")
    full_min = _threshold(thresholds, "full_point_min")
    core_high_min = _threshold(thresholds, "core_only_point_min")
    core_low_max = _threshold(thresholds, "core_only_point_max")
    full_h0_drop_min = _threshold(thresholds, "full_h0_margin_drop_min")
    h0_relevant_drop_min = _threshold(thresholds, "h0_relevant_margin_drop_min")
    residual_drop_min = _threshold(thresholds, "full_core_margin_drop_min")
    final_delta_max = _threshold(thresholds, "final_delta_point_max")
    final_delta_drop_min = _threshold(thresholds, "full_final_delta_margin_drop_min")

    def one(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        h0 = [_bool(row, "h0_correct") for row in rows]
        full = [_bool(row, "full_correct") for row in rows]
        core = [_bool(row, "core_only_correct") for row in rows]
        final_delta = [_bool(row, "final_delta_correct") for row in rows]
        h0_metric = _accuracy(h0)
        full_metric = _accuracy(full)
        core_metric = _accuracy(core)
        final_delta_metric = _accuracy(final_delta)
        h0_drop = [_number(row, "full_h0_margin_drop_lower", "h0_margin_drop_lower") for row in rows]
        h0_relevant_drop = [_number(row, "h0_relevant_margin_drop_lower") for row in rows]
        core_drop = [_number(row, "full_core_margin_drop_lower", "full_core_margin_drop") for row in rows]
        final_delta_drop = [_number(row, "full_final_delta_margin_drop_lower", "final_delta_margin_drop_lower") for row in rows]
        h0_low = h0_metric["point"] <= h0_low_max
        full_high = full_metric["point"] >= full_min
        core_high = core_metric["point"] >= core_high_min
        core_low = core_metric["point"] <= core_low_max
        direct = (
            h0_metric["point"] >= h0_direct_min
            and h0_metric["lower"] >= h0_direct_lower
            and core_low
            and _mean(h0_relevant_drop) >= h0_relevant_drop_min
        )
        required = h0_low and full_high and core_high and _mean(h0_drop) >= full_h0_drop_min
        residual = (
            h0_low
            and full_high
            and core_low
            and final_delta_metric["point"] <= final_delta_max
            and _mean(core_drop) >= residual_drop_min
            and _mean(final_delta_drop) >= final_delta_drop_min
        )
        if direct:
            classification = "H0_DIRECT"
        elif required:
            classification = "CORE_REQUIRED"
        elif residual:
            classification = "CORE_RESIDUAL"
        elif full_high and (h0_metric["point"] >= h0_direct_min or core_high):
            classification = "MIXED"
        else:
            classification = "INCONCLUSIVE"
        return {"classification": classification, "n": len(rows), "full": full_metric, "h0": h0_metric, "core_only": core_metric, "final_delta": final_delta_metric, "full_h0_margin_drop_lower": _mean(h0_drop), "h0_relevant_margin_drop_lower": _mean(h0_relevant_drop), "full_core_margin_drop_lower": _mean(core_drop), "full_final_delta_margin_drop_lower": _mean(final_delta_drop)}

    return _family_report(records, one)


def classify_b_axis(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Classify B without substituting a reachability certificate for task arity."""

    assessed_min = _threshold(thresholds, "task_causal_assessed_point_min")
    task_multi_min = _threshold(thresholds, "task_multi_object_causal_point_min")
    low_order_min = _threshold(thresholds, "task_low_order_supported_point_min")
    eligible_min = _threshold(thresholds, "baseline_eligible_point_min")
    copy_min = _threshold(thresholds, "copy_retention_point_min")
    copy_lower_min = _threshold(thresholds, "copy_retention_wilson_lower_min")
    two_max = _threshold(thresholds, "two_contributor_point_max")
    two_lower_min = _threshold(thresholds, "two_contributor_wilson_lower_min")
    copy_drop_lower_min = _threshold(thresholds, "copy_collapse_bootstrap_lower_min")
    minimum_eligible_raw = _threshold(thresholds, "minimum_eligible_records_per_family")
    bootstrap_seed_raw = _threshold(thresholds, "bootstrap_seed")
    bootstrap_samples_raw = _threshold(thresholds, "bootstrap_replicates")
    if any(value != int(value) for value in (minimum_eligible_raw, bootstrap_seed_raw, bootstrap_samples_raw)):
        raise ValueError("B-axis count/seed thresholds must be integers")
    minimum_eligible = int(minimum_eligible_raw)
    bootstrap_seed = int(bootstrap_seed_raw)
    bootstrap_samples = int(bootstrap_samples_raw)
    if minimum_eligible <= 0 or bootstrap_samples <= 0:
        raise ValueError("B-axis minimum eligible and bootstrap replicates must be positive")

    def one(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        assessed = [_bool(row, "task_causal_assessed") for row in rows]
        task_multi = [_bool(row, "task_multi_object_causal") for row in rows]
        low_order = [_bool(row, "task_low_order_supported") for row in rows]
        conflict = [_bool(row, "task_evidence_conflict") for row in rows]
        certificate_multi = [_bool(row, "certificate_multi") for row in rows]
        assessed_metric = _accuracy(assessed)
        task_multi_metric = _accuracy(task_multi)
        low_order_metric = _accuracy(low_order)
        conflict_metric = _accuracy(conflict)
        certificate_metric = _accuracy(certificate_multi)

        multi_rows = [row for row in rows if _bool(row, "task_multi_object_causal")]
        eligible_rows = [
            row for row in multi_rows
            if _bool(row, "baseline_eligible")
            and _bool(row, "contributor_denominator_valid")
        ]
        eligible_metric = wilson_interval(len(eligible_rows), len(multi_rows))
        copy_successes = sum(
            _bool(row, "copy_all_correct") and _number(row, "copy_retention") >= copy_min
            for row in eligible_rows
        )
        copy_metric = wilson_interval(copy_successes, len(eligible_rows))
        contributor_metric = wilson_interval(
            sum(_bool(row, "two_contributors") for row in eligible_rows),
            len(eligible_rows),
        )
        collapse_bootstrap = None
        if eligible_rows:
            collapse_bootstrap = paired_cluster_bootstrap(
                [_number(row, "full_margin") for row in eligible_rows],
                [_number(row, "copy_all_margin") for row in eligible_rows],
                [str(row.get("example_id", index)) for index, row in enumerate(eligible_rows)],
                seed=bootstrap_seed,
                samples=bootstrap_samples,
            )

        reason = "registered_evidence_does_not_meet_any_classification"
        if assessed_metric["point"] < assessed_min:
            label = "INCONCLUSIVE"
            reason = "causal_task_arity_not_established"
        elif task_multi_metric["point"] < task_multi_min:
            if low_order_metric["point"] >= low_order_min and conflict_metric["successes"] == 0:
                label = "TASK_LOW_ORDER"
                reason = "causal_low_order_and_registered_certificate_agree"
            else:
                label = "INCONCLUSIVE"
                reason = "causal_and_certificate_arity_do_not_support_low_order_claim"
        elif len(eligible_rows) < minimum_eligible or eligible_metric["point"] < eligible_min:
            label = "INCONCLUSIVE"
            reason = "insufficient_eligible_multi_object_records"
        elif (
            copy_metric["point"] >= copy_min
            and copy_metric["lower"] >= copy_lower_min
            and contributor_metric["point"] <= two_max
        ):
            label = "SINGLE_SLOT_COPY"
            reason = "copy_retained_with_fewer_than_two_functional_contributors"
        elif (
            contributor_metric["lower"] >= two_lower_min
            and collapse_bootstrap is not None
            and collapse_bootstrap["lower"] >= copy_drop_lower_min
        ):
            label = "MULTI_ADDRESS_SUPPORTED"
            reason = "two_contributors_and_copy_collapse_replicate"
        else:
            label = "INCONCLUSIVE"
        return {
            "classification": label,
            "reason": reason,
            "n": len(rows),
            "task_causal_assessed": assessed_metric,
            "task_multi_object_causal": task_multi_metric,
            "task_low_order_supported": low_order_metric,
            "task_evidence_conflict": conflict_metric,
            "certificate_multi": certificate_metric,
            "eligible_multi_object_records": eligible_metric,
            "eligible_n": len(eligible_rows),
            "minimum_eligible_n": minimum_eligible,
            "copy_retained": copy_metric,
            "two_contributors": contributor_metric,
            "copy_collapse_margin_drop": collapse_bootstrap,
            "certificate_is_not_causal_arity": True,
        }

    return _family_report(records, one)


def classify_c_axis(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Classify target insufficiency, readout insufficiency, or latent motion."""

    target_min = _threshold(thresholds, "target_adequacy_point_min")
    motion_min = _threshold(thresholds, "latent_motion_point_min")
    auc_min = _threshold(thresholds, "latent_separation_auc_min")
    decoder_min = _threshold(thresholds, "decoder_balanced_accuracy_min")
    null_gap_min = _threshold(thresholds, "readout_minus_null_min")
    null_gap_max = _threshold(thresholds, "static_null_gap_max")

    def one(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        adequate = [_bool(row, "target_adequate") for row in rows]
        motion = [_number(row, "relative_motion", "active_relative_delta") for row in rows]
        auc = [_number(row, "separation_auc", "fixed_channel_auc") for row in rows]
        crossfit = [_number(row, "crossfit_decoder_balanced_accuracy", "decoder_balanced_accuracy") for row in rows]
        frozen = [_number(row, "frozen_decoder_balanced_accuracy", "frozen_decoder_ba") for row in rows]
        null_gap = [_number(row, "static_null_gap", "shuffle_gap") for row in rows]
        target_metric = _accuracy(adequate)
        motion_point = _mean(motion)
        auc_point = min(auc)
        crossfit_point = min(crossfit)
        frozen_point = min(frozen)
        null_point = min(null_gap)
        target_bad = target_metric["point"] < target_min
        latent_decodable = crossfit_point >= decoder_min and null_point >= null_gap_min
        latent_motion = motion_point >= motion_min and auc_point >= auc_min
        static_like = max(null_gap) <= null_gap_max
        if target_bad:
            label = "TARGET_INSUFFICIENT"
        elif not latent_decodable and not latent_motion and static_like:
            label = "LATENT_NOT_FORMED"
        elif latent_decodable and latent_motion and frozen_point < decoder_min:
            label = "READOUT_INSUFFICIENT"
        elif not latent_decodable or not latent_motion or frozen_point < decoder_min:
            label = "MIXED"
        else:
            label = "INCONCLUSIVE"
        return {"classification": label, "n": len(rows), "target_adequacy": target_metric, "relative_motion_point": motion_point, "minimum_separation_auc": auc_point, "minimum_crossfit_decoder_balanced_accuracy": crossfit_point, "minimum_frozen_decoder_balanced_accuracy": frozen_point, "minimum_readout_minus_null": null_point, "static_null_like": static_like}

    return _family_report(records, one)


__all__ = ["classify_a_axis", "classify_b_axis", "classify_c_axis", "classify_loss_gate_alignment"]
