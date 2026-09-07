from __future__ import annotations

"""Join task-side dependency arity with frozen-model functional effects."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import answer_margin, finite_json, wilson_interval
from .task_arity import ASSESSED, assess_task_arity


def load_source_records(data_root: Path, example_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    wanted = {str(value) for value in example_ids}
    found: dict[str, dict[str, Any]] = {}
    dataset = Path(data_root) / "dataset"
    for path in sorted(dataset.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not any(example_id in line for example_id in wanted - set(found)):
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"JSON object required: {path}:{line_number}")
                example_id = str(value.get("example_id", ""))
                if example_id in wanted:
                    if example_id in found:
                        raise ValueError(f"duplicate source record: {example_id}")
                    found[example_id] = value
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - set(found))
    if missing:
        raise KeyError(f"source records absent: {missing[:8]}")
    return found


def _content_certificate_arity(row: Mapping[str, Any], task: Mapping[str, Any]) -> int | None:
    certificate = task.get("certificate")
    if not isinstance(certificate, Mapping) or certificate.get("status") != ASSESSED:
        return None
    slots = certificate.get("support_slots")
    if not isinstance(slots, list):
        return None
    objects = {
        int(obj["slot"]): obj
        for obj in row.get("objects", [])
        if isinstance(obj, Mapping) and type(obj.get("slot")) is int
    }
    family = str(row.get("family", "")).upper()
    count = 0
    for slot in slots:
        obj = objects.get(int(slot))
        if obj is None:
            return None
        if family == "CPS" and str(obj.get("kind", "")).lower() == "cps_decision":
            continue
        count += 1
    return count


def _family_distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "counts": {}, "mean": None}
    counts: dict[str, int] = {}
    for value in values:
        counts[str(int(value))] = counts.get(str(int(value)), 0) + 1
    return {
        "n": len(values),
        "counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "mean": float(np.mean(values)),
    }


def audit_task_and_model_arity(
    target_rows: Sequence[Mapping[str, Any]],
    source_records: Mapping[str, Mapping[str, Any]],
    path_collection: Mapping[str, Any],
    *,
    contributor_ratio: float,
    eligible_margin: float,
) -> dict[str, Any]:
    ids = list(path_collection["example_ids"])
    if [str(row.get("example_id")) for row in target_rows] != ids:
        raise ValueError("target/path record order mismatch")
    answers = path_collection["answers"].numpy().astype(np.int64)
    full_logits = path_collection["paths"]["full"].numpy().astype(np.float64)
    h0_logits = path_collection["paths"]["h0"].numpy().astype(np.float64)
    copy_logits = path_collection["paths"]["copy_all_query_owner"].numpy().astype(np.float64)
    slot_logits = path_collection["initial_slot_replace_logits"].numpy().astype(np.float64)
    presence = path_collection["presence"].numpy().astype(bool)
    content_mask = path_collection["content_mask"].numpy().astype(bool)
    if content_mask.shape != presence.shape:
        raise ValueError("content/presence masks differ")
    if np.any(content_mask & ~presence):
        raise ValueError("content_mask selects an absent slot")
    full_margin = np.asarray(answer_margin(full_logits, answers))
    h0_margin = np.asarray(answer_margin(h0_logits, answers))
    copy_margin = np.asarray(answer_margin(copy_logits, answers))
    flat_answers = np.repeat(answers, slot_logits.shape[1])
    replaced_margin = np.asarray(
        answer_margin(slot_logits.reshape(-1, slot_logits.shape[-1]), flat_answers)
    ).reshape(slot_logits.shape[:2])
    available = full_margin - h0_margin
    contributor_denominator_valid = available > 0.0
    effects = np.maximum(full_margin[:, None] - replaced_margin, 0.0)
    contributors = (
        (effects >= float(contributor_ratio) * available[:, None])
        & content_mask
        & contributor_denominator_valid[:, None]
    )

    record_reports: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for index, row in enumerate(target_rows):
        example_id = ids[index]
        source = source_records.get(example_id)
        if not isinstance(source, Mapping):
            raise KeyError(f"source record absent: {example_id}")
        if str(source.get("family", "")).upper() != str(row.get("family", "")).upper():
            raise ValueError(f"source/target family mismatch: {example_id}")
        task = assess_task_arity(row, source)
        certificate_arity = _content_certificate_arity(row, task)
        answer_support = task.get("answer_support", {})
        influence = task.get("counterfactual_influence", {})
        answer_support_assessed = (
            isinstance(answer_support, Mapping)
            and answer_support.get("status") == ASSESSED
            and answer_support.get("minimal_answer_support_arity") is not None
        )
        influence_assessed = (
            isinstance(influence, Mapping)
            and influence.get("status") == ASSESSED
            and isinstance(influence.get("leave_one_out"), Mapping)
        )
        answer_support_arity = (
            int(answer_support["minimal_answer_support_arity"])
            if answer_support_assessed else None
        )
        influence_count = (
            sum(
                bool(value.get("winner_changed"))
                for value in influence["leave_one_out"].values()
                if isinstance(value, Mapping)
            )
            if influence_assessed else None
        )
        causal_assessed = bool(answer_support_assessed and influence_assessed)
        causal_multi = bool(
            causal_assessed
            and (
                int(answer_support_arity or 0) >= 2
                or int(influence_count or 0) >= 2
            )
        )
        certificate_multi = bool(certificate_arity is not None and certificate_arity >= 2)
        low_order_supported = bool(
            causal_assessed
            and not causal_multi
            and certificate_arity is not None
            and certificate_arity < 2
        )
        evidence_conflict = bool(causal_assessed and not causal_multi and certificate_multi)
        two = int(contributors[index].sum()) >= 2
        full_correct = bool(full_logits[index].argmax() == answers[index])
        copy_correct = bool(copy_logits[index].argmax() == answers[index])
        baseline_eligible = bool(full_correct and full_margin[index] > float(eligible_margin))
        retention = float(copy_margin[index] / max(full_margin[index], 1.0e-8))
        report = {
            "example_id": example_id,
            "family": str(row.get("family", "")).upper(),
            "task": task,
            "content_certificate_arity": certificate_arity,
            "model": {
                "baseline_correct": full_correct,
                "baseline_margin": float(full_margin[index]),
                "baseline_eligible": baseline_eligible,
                "full_minus_h0_margin": float(full_margin[index] - h0_margin[index]),
                "contributor_denominator_valid": bool(contributor_denominator_valid[index]),
                "copy_all_correct": copy_correct,
                "copy_all_margin": float(copy_margin[index]),
                "copy_retention": retention,
                "copy_collapse_drop": float(full_margin[index] - copy_margin[index]),
                "contributor_ratio_threshold": float(contributor_ratio),
                "contributor_slots": np.flatnonzero(contributors[index]).tolist(),
                "contributor_count": int(contributors[index].sum()),
                "two_contributors": two,
                "slot_margin_effects": effects[index].tolist(),
                "content_slots": np.flatnonzero(content_mask[index]).tolist(),
            },
            "task_evidence": {
                "certificate_multi": certificate_multi,
                "causal_dimensions_assessed": causal_assessed,
                "answer_support_arity": answer_support_arity,
                "independently_influential_content_objects": influence_count,
                "multi_object_causal": causal_multi,
                "low_order_supported": low_order_supported,
                "certificate_causal_conflict": evidence_conflict,
            },
        }
        record_reports.append(report)
        decision_rows.append(
            {
                "example_id": example_id,
                "family": report["family"],
                "certificate_multi": certificate_multi,
                "task_causal_assessed": causal_assessed,
                "task_multi_object_causal": causal_multi,
                "task_low_order_supported": low_order_supported,
                "task_evidence_conflict": evidence_conflict,
                "baseline_eligible": baseline_eligible,
                "contributor_denominator_valid": bool(contributor_denominator_valid[index]),
                "full_margin": float(full_margin[index]),
                "copy_all_margin": float(copy_margin[index]),
                "copy_all_correct": copy_correct,
                "copy_retention": retention,
                "two_contributors": two,
                "copy_collapse_drop": float(full_margin[index] - copy_margin[index]),
            }
        )

    family_reports: dict[str, Any] = {}
    for family in sorted({row["family"] for row in record_reports}):
        selected = [row for row in record_reports if row["family"] == family]
        certificates = [
            int(row["content_certificate_arity"])
            for row in selected
            if row["content_certificate_arity"] is not None
        ]
        answer_support = [
            int(row["task"]["answer_support"]["minimal_answer_support_arity"])
            for row in selected
            if row["task"].get("answer_support", {}).get("status") == ASSESSED
            and row["task"]["answer_support"].get("minimal_answer_support_arity") is not None
        ]
        causal_assessed_count = sum(
            bool(row["task_evidence"]["causal_dimensions_assessed"])
            for row in selected
        )
        causal_multi_count = sum(
            bool(row["task_evidence"]["multi_object_causal"])
            for row in selected
        )
        low_order_count = sum(
            bool(row["task_evidence"]["low_order_supported"])
            for row in selected
        )
        conflict_count = sum(
            bool(row["task_evidence"]["certificate_causal_conflict"])
            for row in selected
        )
        eligible = [row for row in selected if row["model"]["baseline_eligible"]]
        two_count = sum(bool(row["model"]["two_contributors"]) for row in eligible)
        copy_count = sum(bool(row["model"]["copy_all_correct"]) for row in eligible)
        family_reports[family] = {
            "n": len(selected),
            "task": {
                "content_certificate_arity": _family_distribution(certificates),
                "answer_support_arity": _family_distribution(answer_support),
                "answer_support_not_assessable": len(selected) - len(answer_support),
                "causal_dimensions_assessed": wilson_interval(causal_assessed_count, len(selected)),
                "multi_object_causal": wilson_interval(causal_multi_count, len(selected)),
                "low_order_supported": wilson_interval(low_order_count, len(selected)),
                "certificate_causal_conflicts": conflict_count,
                "invalid_records": sum(not bool(row["task"].get("valid")) for row in selected),
            },
            "model": {
                "eligible_n": len(eligible),
                "two_contributors": wilson_interval(two_count, len(eligible)),
                "copy_all_answer_retained": wilson_interval(copy_count, len(eligible)),
                "mean_copy_retention": float(np.mean([row["model"]["copy_retention"] for row in eligible])) if eligible else None,
                "mean_contributor_count": float(np.mean([row["model"]["contributor_count"] for row in eligible])) if eligible else None,
            },
        }
    return finite_json(
        {
            "families": family_reports,
            "records": record_reports,
            "decision_rows": decision_rows,
            "definitions": {
                "task_answer_support": "source_simulator_subset_answer_preservation",
                "task_certificate": "registered_query_reachable_schedule_support",
                "model_contributor": "pre_recurrence_single_slot_mean_replace_margin_effect",
                "model_contributor_denominator": "full_margin_minus_h0_margin; nonpositive is invalid",
                "cps_decision_slot_counted_as_content": False,
                "ere_counterfactual_answer_support": "NOT_ASSESSABLE",
                "certificate_never_substitutes_for_answer_or_counterfactual_arity": True,
            },
        }
    )


__all__ = ["audit_task_and_model_arity", "load_source_records"]
