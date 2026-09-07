"""Read-only task-side arity and certificate attribution for C1S/S1.

This module deliberately does not inspect model activations or answer-derived
features.  ``certificate`` is a static reachability result obtained by
walking the registered target schedule backwards from ``query_owner``.
``answer_support`` and ``counterfactual_influence`` are separate dimensions:
for CPS they are obtained by exact source-simulator subset enumeration, while
ERE has no legal object-removal counterfactual in this contract and is marked
``NOT_ASSESSABLE``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from yggdrasil_v2.r1_revalidation.common.simulator import evaluate_cps


ASSESSED = "ASSESSED"
NOT_ASSESSABLE = "NOT_ASSESSABLE"
INVALID = "INVALID"
SCHEMA_VERSION = "C1S-S1-D003-TASK-ARITY-V2"
MAX_ENUMERATED_CANDIDATES = 8


class TaskArityError(ValueError):
    """Raised internally for a malformed target/source pair."""


def _status(status: str, **values: Any) -> dict[str, Any]:
    return {"status": status, **values}


def _integer(value: Any) -> bool:
    return type(value) is int and not isinstance(value, bool)


def _family(record: Mapping[str, Any]) -> str | None:
    value = record.get("family")
    if value is None and isinstance(record.get("target"), Mapping):
        value = record["target"].get("family")
    return str(value).upper() if value is not None else None


def _query_owner(record: Mapping[str, Any]) -> int:
    value = record.get("query_owner")
    query = record.get("query")
    ownership = record.get("ownership")
    if value is None and isinstance(query, Mapping):
        value = query.get("query_owner", query.get("owner_slot"))
    if value is None and isinstance(ownership, Mapping):
        value = ownership.get("query_owner")
    if not _integer(value):
        raise TaskArityError("query_owner must be an integer")
    return int(value)


def _objects(record: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    objects = record.get("objects")
    if not isinstance(objects, list) or not objects:
        raise TaskArityError("objects must be a non-empty list")
    by_slot: dict[int, Mapping[str, Any]] = {}
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            raise TaskArityError(f"objects[{index}] must be a mapping")
        slot = obj.get("slot", index)
        if not _integer(slot) or int(slot) < 0 or int(slot) in by_slot:
            raise TaskArityError(f"objects[{index}] has an invalid or duplicate slot")
        by_slot[int(slot)] = obj
    return objects, by_slot


def _schedule(record: Mapping[str, Any], object_slots: set[int]) -> list[dict[str, Any]]:
    schedule = record.get("schedule")
    if not isinstance(schedule, list):
        raise TaskArityError("schedule must be a list; owner arrays are not a substitute")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(schedule):
        if not isinstance(item, Mapping):
            raise TaskArityError(f"schedule[{index}] must be a mapping")
        active = item.get("active")
        source = item.get("source_slot")
        target = item.get("target_slot")
        if type(active) not in (bool, int) or int(active) not in (0, 1):
            raise TaskArityError(f"schedule[{index}].active must be boolean")
        if not _integer(source) or not _integer(target):
            raise TaskArityError(f"schedule[{index}] requires integer source_slot/target_slot")
        source, target = int(source), int(target)
        if int(active):
            if source not in object_slots or target not in object_slots:
                raise TaskArityError(f"schedule[{index}] points outside objects")
        elif (source, target) != (-1, -1):
            raise TaskArityError(f"inactive schedule[{index}] must use -1 owners")
        normalized.append({
            "step": int(item.get("step", index)) if _integer(item.get("step", index)) else index,
            "active": bool(active),
            "source_slot": source,
            "target_slot": target,
        })
    return normalized


def _registered_certificate(query_owner: int, schedule: Sequence[Mapping[str, Any]], object_slots: set[int]) -> dict[str, Any]:
    """Reverse-link target -> source, retaining only query-reachable edges."""

    if query_owner not in object_slots:
        raise TaskArityError("query_owner is not a registered object slot")
    required: set[int] = {query_owner}
    support_steps: list[int] = []
    support_edges: list[dict[str, int]] = []
    changed = True
    while changed:
        changed = False
        for index in range(len(schedule) - 1, -1, -1):
            item = schedule[index]
            if not item["active"] or item["target_slot"] not in required:
                continue
            source, target = item["source_slot"], item["target_slot"]
            edge = {"schedule_index": index, "source_slot": source, "target_slot": target}
            if index not in support_steps:
                support_steps.append(index)
                support_edges.append(edge)
            if source not in required:
                required.add(source)
                changed = True
    support_steps.sort()
    support_edges.sort(key=lambda item: item["schedule_index"])
    certificate = bool(support_edges and required - {query_owner})
    return {
        "registered_execution": bool(support_edges),
        "certificate": certificate,
        "support_slots": sorted(required),
        "support_steps": support_steps,
        "support_edges": support_edges,
        "query_owner": query_owner,
    }


def _resolve_ast(record: Mapping[str, Any], source: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    # Target-bank rows are supervision artifacts and are never an independent
    # source-side counterfactual oracle.  Only the separately pinned immutable
    # source record may supply the AST used for task arity.
    del record
    if not isinstance(source, Mapping):
        return None
    for key in ("program_ast", "ast"):
        value = source.get(key)
        if isinstance(value, Mapping):
            return value
    if all(key in source for key in ("initial_state", "actions", "candidates", "budget", "goal", "final_constraints")):
        return source
    return None


def _cps_candidate_slots(by_slot: Mapping[int, Mapping[str, Any]]) -> list[int]:
    """Return physical model slots in original AST candidate-index order."""

    indexed: dict[int, int] = {}
    for slot, obj in by_slot.items():
        if str(obj.get("kind", "")).lower() not in {"cps_candidate", "candidate"}:
            continue
        semantic = obj.get("semantic_slot")
        name_match = re.fullmatch(r"candidate:(\d+)", str(obj.get("name", "")))
        named = int(name_match.group(1)) if name_match else None
        if not _integer(semantic) and named is None:
            raise TaskArityError(
                f"CPS candidate at model slot {slot} lacks semantic_slot/candidate:n identity"
            )
        candidate = int(semantic) if _integer(semantic) else int(named)
        if candidate < 0 or (named is not None and named != candidate):
            raise TaskArityError(
                f"CPS candidate at model slot {slot} has inconsistent semantic identity"
            )
        if candidate in indexed:
            raise TaskArityError("duplicate CPS semantic candidate index")
        indexed[candidate] = int(slot)
    if not indexed or set(indexed) != set(range(len(indexed))):
        raise TaskArityError("CPS semantic candidate indices must be the complete range 0..N-1")
    return [indexed[index] for index in range(len(indexed))]


def _plan_fingerprints(ast: Mapping[str, Any]) -> list[str]:
    candidates = ast.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TaskArityError("CPS source AST requires a non-empty candidates list")
    fingerprints: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("plan"), list):
            raise TaskArityError(f"CPS candidate {index} has no plan list")
        # The production semantic fingerprint intentionally alpha-renames
        # symbols and treats candidate lists as unordered.  That is useful
        # for AST equivalence, but would collapse two distinct answer plans.
        # For winner identity we therefore hash the original, ordered plan
        # payload and retain its action literals.
        payload = json.dumps(
            {"plan": deepcopy(candidate["plan"])},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprints.append(hashlib.sha256(payload).hexdigest().upper())
    if len(set(fingerprints)) != len(fingerprints):
        raise TaskArityError("duplicate CPS plan fingerprints make winner identity ambiguous")
    return fingerprints


def _subset_result(ast: Mapping[str, Any], selected: Sequence[int], fingerprints: Sequence[str]) -> dict[str, Any]:
    transformed = deepcopy(dict(ast))
    original = list(ast["candidates"])
    transformed["candidates"] = [deepcopy(original[index]) for index in selected]
    try:
        output = evaluate_cps(transformed)
    except Exception as exc:  # simulator validation is a fail-closed boundary
        return {"valid": False, "error": f"simulator_invalid:{type(exc).__name__}:{exc}"}
    if not isinstance(output, Mapping) or not isinstance(output.get("candidates"), list):
        return {"valid": False, "error": "simulator_returned_malformed_output"}
    answer = output.get("answer")
    unique_optimum = bool(output.get("unique_optimum", False))
    if answer is not None and (not _integer(answer) or answer < 0 or answer >= len(selected)):
        return {"valid": False, "error": "simulator_returned_invalid_answer"}
    if (answer is None) != (not unique_optimum):
        return {"valid": False, "error": "simulator_answer_unique_optimum_disagree"}
    valid_candidates = sum(
        bool(candidate.get("valid", False))
        for candidate in output["candidates"]
        if isinstance(candidate, Mapping)
    )
    if answer is not None:
        answer_kind = "winner"
    elif valid_candidates == 0:
        answer_kind = "none"
    else:
        answer_kind = "tie"
    winner_fp = None if answer is None else fingerprints[selected[int(answer)]]
    return {
        "valid": True,
        "selected_indices": list(selected),
        "winner_index": None if answer is None else selected[int(answer)],
        "winner_plan_fingerprint": winner_fp,
        "unique_optimum": unique_optimum,
        "answer_kind": answer_kind,
        "valid_candidate_count": valid_candidates,
    }


def _cps_counterfactuals(ast: Mapping[str, Any], candidate_slots: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    count = len(candidate_slots)
    if count > MAX_ENUMERATED_CANDIDATES:
        unavailable = _status(NOT_ASSESSABLE, reason="candidate_count_exceeds_exact_enumeration_cap", candidate_count=count)
        return unavailable, unavailable.copy()
    fingerprints = _plan_fingerprints(ast)
    if len(fingerprints) != count:
        raise TaskArityError("source candidate count does not match target candidate count")
    all_subsets: list[dict[str, Any]] = []
    invalid_masks: list[int] = []
    for mask in range(1 << count):
        selected = [index for index in range(count) if mask & (1 << index)]
        result = _subset_result(ast, selected, fingerprints)
        row = {"mask": mask, "selected_slots": [candidate_slots[index] for index in selected], **result}
        all_subsets.append(row)
        if not result.get("valid", False):
            invalid_masks.append(mask)
    if invalid_masks:
        reason = "invalid_source_counterfactual_masks"
        failed = _status(INVALID, reason=reason, invalid_masks=invalid_masks, subsets=all_subsets)
        return failed, failed.copy()
    none_masks = [int(row["mask"]) for row in all_subsets if row.get("answer_kind") == "none"]
    tie_masks = [int(row["mask"]) for row in all_subsets if row.get("answer_kind") == "tie"]
    baseline = all_subsets[-1]
    baseline_fp = baseline.get("winner_plan_fingerprint")
    if baseline.get("answer_kind") != "winner" or baseline_fp is None:
        unavailable = _status(
            NOT_ASSESSABLE,
            reason=f"source_baseline_{baseline.get('answer_kind', 'unknown')}_not_causal_arity",
            baseline_answer_kind=baseline.get("answer_kind"),
            none_count=len(none_masks),
            tie_count=len(tie_masks),
            none_masks=none_masks,
            tie_masks=tie_masks,
            invalid_masks=[],
            subsets=all_subsets,
        )
        return unavailable, unavailable.copy()
    preserving = [
        row for row in all_subsets
        if row.get("answer_kind") == "winner"
        and row["winner_plan_fingerprint"] == baseline_fp
    ]
    minimum = min((len(row["selected_slots"]) for row in preserving), default=None)
    answer_support = _status(
        ASSESSED,
        baseline_winner_plan_fingerprint=baseline_fp,
        baseline_winner_slot=candidate_slots[int(baseline["winner_index"])],
        exact_subset_count=len(all_subsets),
        none_count=len(none_masks),
        tie_count=len(tie_masks),
        none_masks=none_masks,
        tie_masks=tie_masks,
        invalid_masks=[],
        answer_preserving_subset_count=len(preserving),
        minimal_answer_support_arity=minimum,
        minimal_answer_support_subsets=[row["selected_slots"] for row in preserving if len(row["selected_slots"]) == minimum],
        candidate_support={
            str(candidate_slots[index]): any(
                candidate_slots[index] in row["selected_slots"]
                for row in preserving if len(row["selected_slots"]) == minimum
            )
            for index in range(count)
        },
        subsets=all_subsets,
    )
    effects: dict[str, Any] = {}
    full_mask = (1 << count) - 1
    for index, slot in enumerate(candidate_slots):
        mask = full_mask ^ (1 << index)
        leave_one_out = all_subsets[mask]
        effects[str(slot)] = {
            "valid": leave_one_out["valid"],
            "removed_slot": slot,
            "winner_plan_fingerprint": leave_one_out.get("winner_plan_fingerprint"),
            "winner_changed": leave_one_out.get("winner_plan_fingerprint") != baseline_fp,
            "removed_baseline_winner": index == baseline.get("winner_index"),
        }
    influence = _status(
        ASSESSED,
        baseline_winner_plan_fingerprint=baseline_fp,
        exact_subset_count=len(all_subsets),
        none_count=len(none_masks),
        tie_count=len(tie_masks),
        none_masks=none_masks,
        tie_masks=tie_masks,
        invalid_masks=[],
        leave_one_out=effects,
        subsets=all_subsets,
    )
    return answer_support, influence


def _invalid_report(reason: str) -> dict[str, Any]:
    dimension = _status(INVALID, reason=reason)
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": False,
        "reason": reason,
        "registered_execution": dimension.copy(),
        "certificate": dimension.copy(),
        "answer_support": dimension.copy(),
        "counterfactual_influence": dimension.copy(),
    }


def assess_task_arity(record: Mapping[str, Any], source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assess one target row without mutating it or invoking model code.

    Missing ASTs and unsupported ERE counterfactuals are represented as
    ``NOT_ASSESSABLE``.  Structural corruption is represented as ``INVALID``
    for every dimension (fail-closed), never as a zero-arity finding.
    """

    try:
        if not isinstance(record, Mapping):
            raise TaskArityError("record must be a mapping")
        family = _family(record)
        if family not in {"CPS", "ERE"}:
            raise TaskArityError("family must be CPS or ERE")
        objects, by_slot = _objects(record)
        query_owner = _query_owner(record)
        schedule = _schedule(record, set(by_slot))
        registered = _registered_certificate(query_owner, schedule, set(by_slot))
        candidate_slots = _cps_candidate_slots(by_slot) if family == "CPS" else []
        if family == "CPS":
            ast = _resolve_ast(record, source)
            if ast is None:
                answer_support = _status(NOT_ASSESSABLE, reason="CPS_program_ast_unavailable")
                influence = _status(NOT_ASSESSABLE, reason="CPS_program_ast_unavailable")
            else:
                answer_support, influence = _cps_counterfactuals(ast, candidate_slots)
        else:
            answer_support = _status(NOT_ASSESSABLE, reason="ERE_no_legal_object_counterfactual")
            influence = _status(NOT_ASSESSABLE, reason="ERE_no_legal_object_counterfactual")
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": True,
            "example_id": record.get("example_id"),
            "family": family,
            "object_count": len(objects),
            "candidate_slots": candidate_slots,
            "query_owner": query_owner,
            "registered_execution": _status(ASSESSED, **registered),
            "certificate": _status(ASSESSED, **registered),
            "answer_support": answer_support,
            "counterfactual_influence": influence,
        }
    except (TaskArityError, KeyError, TypeError, ValueError) as exc:
        return _invalid_report(str(exc))


# Explicit aliases make the contract convenient for runners without adding a
# second implementation or a mutable registry.
task_arity = assess_task_arity
analyze_task_arity = assess_task_arity


__all__ = [
    "ASSESSED", "NOT_ASSESSABLE", "INVALID", "SCHEMA_VERSION",
    "MAX_ENUMERATED_CANDIDATES", "TaskArityError", "assess_task_arity",
    "analyze_task_arity", "task_arity",
]
