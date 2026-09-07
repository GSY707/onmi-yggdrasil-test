from __future__ import annotations

"""CPS outcome and composition derivation for G06."""

from copy import deepcopy
import hashlib
import json
from typing import Any


FAILURE_REASONS = {"precondition", "unknown_action", "budget", "final_constraint", "goal"}
COMPOSITION_WITNESSES = {
    "unlock_resource_goal",
    "same_goal_different_final",
    "resource_cost_budget",
    "early_fact_destruction",
}


def _metric(value: Any, numerator: int, denominator: int, passed: bool, failures: list[str]) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "passed": bool(passed),
        "failures": sorted(set(str(item) for item in failures)),
    }


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _diff_paths(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_diff_paths(left[key], right[key], child))
        return result
    if isinstance(left, list):
        result = []
        for index in range(max(len(left), len(right))):
            child = f"{path}/{index}"
            if index >= len(left) or index >= len(right):
                result.append(child)
            else:
                result.extend(_diff_paths(left[index], right[index], child))
        return result
    return [] if left == right else [path or "/"]


def _remove_path(value: Any, path: str) -> Any:
    result = deepcopy(value)
    parts = path.split("/")[1:]
    current = result
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    if isinstance(current, list):
        current.pop(int(parts[-1]))
    else:
        current.pop(parts[-1])
    return result


def _candidate(output: dict[str, Any], index: Any) -> dict[str, Any] | None:
    candidates = output.get("candidates")
    if not isinstance(index, int) or not isinstance(candidates, list) or not 0 <= index < len(candidates):
        return None
    row = candidates[index]
    return row if isinstance(row, dict) else None


def _fresh_pstar(output: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [row for row in output.get("candidates", []) if isinstance(row, dict) and row.get("valid") is True]
    if not candidates:
        return None
    minimum = min(row.get("total_cost") for row in candidates)
    winners = [row for row in candidates if row.get("total_cost") == minimum]
    return winners[0] if len(winners) == 1 else None


def _action_features(ast: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, action in enumerate(ast.get("actions", [])):
        result.append(
            {
                "index": index,
                "name": action.get("name"),
                "cost": action.get("cost"),
                "requires_facts": {
                    item.get("fact")
                    for item in action.get("preconditions", [])
                    if item.get("kind") == "fact_true"
                },
                "requires_resources": {
                    (item.get("resource"), item.get("amount"))
                    for item in action.get("preconditions", [])
                    if item.get("kind") == "resource_at_least"
                },
                "adds_facts": {
                    item.get("fact")
                    for item in action.get("effects", [])
                    if item.get("kind") == "add_fact"
                },
                "removes_facts": {
                    item.get("fact")
                    for item in action.get("effects", [])
                    if item.get("kind") == "remove_fact"
                },
                "resource_deltas": {
                    (item.get("resource"), item.get("delta"))
                    for item in action.get("effects", [])
                    if item.get("kind") == "resource_delta"
                },
            }
        )
    return result


def derive_cps_composition(ast: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    """Derive four composition relations from AST topology and fresh outcomes."""

    actions = _action_features(ast)
    by_name = {str(row["name"]): row for row in actions}
    goal_facts = {item.get("fact") for item in ast.get("goal", []) if item.get("kind") == "fact_true"}
    final_facts = {item.get("fact") for item in ast.get("final_constraints", []) if item.get("kind") == "fact_true"}
    pstar = _fresh_pstar(output)
    pstar_plan = pstar.get("plan", []) if isinstance(pstar, dict) else []
    witnesses: dict[str, Any] = {}

    for producer in actions:
        for unlocked in producer["adds_facts"]:
            for charger in actions:
                if unlocked not in charger["requires_facts"]:
                    continue
                for resource, delta in charger["resource_deltas"]:
                    if not isinstance(delta, int) or delta <= 0:
                        continue
                    for finisher in actions:
                        if not (finisher["adds_facts"] & goal_facts):
                            continue
                        if not any(name == resource and isinstance(amount, int) and amount <= delta for name, amount in finisher["requires_resources"]):
                            continue
                        names = [producer["name"], charger["name"], finisher["name"]]
                        try:
                            positions = [pstar_plan.index(name) for name in names]
                        except ValueError:
                            continue
                        if positions == sorted(positions) and len(set(positions)) == 3:
                            witnesses["unlock_resource_goal"] = {
                                "action_indices": [producer["index"], charger["index"], finisher["index"]],
                                "plan_positions": positions,
                            }
                            break
                    if "unlock_resource_goal" in witnesses:
                        break
                if "unlock_resource_goal" in witnesses:
                    break
            if "unlock_resource_goal" in witnesses:
                break
        if "unlock_resource_goal" in witnesses:
            break

    safe_finishers = [row for row in actions if row["adds_facts"] & goal_facts and not (row["removes_facts"] & final_facts)]
    unsafe_finishers = [row for row in actions if row["adds_facts"] & goal_facts and row["removes_facts"] & final_facts]
    for safe in safe_finishers:
        for unsafe in unsafe_finishers:
            safe_candidates = [row for row in output.get("candidates", []) if safe["name"] in row.get("plan", []) and row.get("valid") is True]
            unsafe_candidates = [
                row
                for row in output.get("candidates", [])
                if unsafe["name"] in row.get("plan", []) and "final_constraint" in row.get("failure_reasons", [])
            ]
            if safe_candidates and unsafe_candidates:
                witnesses["same_goal_different_final"] = {
                    "action_indices": [safe["index"], unsafe["index"]],
                    "candidate_indices": [safe_candidates[0]["candidate_index"], unsafe_candidates[0]["candidate_index"]],
                }
                witnesses["early_fact_destruction"] = {
                    "action_index": unsafe["index"],
                    "candidate_index": unsafe_candidates[0]["candidate_index"],
                }
                break
        if "same_goal_different_final" in witnesses:
            break

    budget_failures = [
        row for row in output.get("candidates", []) if isinstance(row, dict) and "budget" in row.get("failure_reasons", [])
    ]
    resource_actions = [row for row in actions if any(isinstance(delta, int) and delta > 0 for _, delta in row["resource_deltas"])]
    if budget_failures and resource_actions and isinstance(ast.get("budget"), int):
        candidate = budget_failures[0]
        cost = sum(by_name.get(str(name), {}).get("cost", 0) for name in candidate.get("plan", []) if name in by_name)
        if isinstance(cost, int) and cost > ast["budget"]:
            witnesses["resource_cost_budget"] = {
                "resource_action_index": resource_actions[0]["index"],
                "candidate_index": candidate["candidate_index"],
                "cost_over_budget": cost - ast["budget"],
            }

    return {
        name: _digest({"name": name, "shape": witnesses[name]})
        for name in sorted(witnesses)
    }


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for child in value.values():
            result.extend(_strings(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_strings(child))
        return result
    return []


def structure_bundle(records: list[dict[str, Any]], composition: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    outputs = replay.get("outputs", {})
    rich_rows = [row for row in records if row.get("structure_certificate", {}).get("role") == "cps_rich"]
    none_rows = [row for row in records if row.get("structure_certificate", {}).get("role") == "cps_none"]
    profile_ok = len(rich_rows) == 1 and len(none_rows) == 1
    rich = rich_rows[0] if len(rich_rows) == 1 else None
    none = none_rows[0] if len(none_rows) == 1 else None
    rich_output = outputs.get(str(rich.get("example_id"))) if isinstance(rich, dict) else None
    none_output = outputs.get(str(none.get("example_id"))) if isinstance(none, dict) else None
    certificate = rich.get("structure_certificate", {}) if isinstance(rich, dict) else {}

    pstar = _fresh_pstar(rich_output) if isinstance(rich_output, dict) else None
    pstar_ok = bool(
        isinstance(pstar, dict)
        and pstar.get("candidate_index") == certificate.get("pstar_index")
        and len(pstar.get("plan", [])) >= 3
        and rich_output.get("unique_optimum") is True
    )
    actual_suboptimal = []
    if isinstance(pstar, dict) and isinstance(rich_output, dict):
        actual_suboptimal = [
            row.get("candidate_index")
            for row in rich_output.get("candidates", [])
            if row.get("valid") is True and row.get("total_cost", 0) > pstar.get("total_cost", 0)
        ]
    suboptimal_ok = bool(actual_suboptimal and actual_suboptimal == certificate.get("valid_suboptimal_indices"))
    actual_invalid = [
        row.get("candidate_index")
        for row in rich_output.get("candidates", [])
        if isinstance(row, dict) and row.get("valid") is False
    ] if isinstance(rich_output, dict) else []
    invalid_ok = len(actual_invalid) >= 5 and actual_invalid == certificate.get("invalid_indices")

    expected_reasons = certificate.get("expected_failure_reasons")
    reasons_ok = isinstance(expected_reasons, dict)
    seen_reasons: set[str] = set()
    if reasons_ok and isinstance(rich_output, dict):
        for key, value in expected_reasons.items():
            candidate = _candidate(rich_output, int(key))
            if candidate is None or candidate.get("failure_reasons") != value:
                reasons_ok = False
                break
            seen_reasons.update(str(item) for item in value)
        reasons_ok = reasons_ok and seen_reasons == FAILURE_REASONS
    else:
        reasons_ok = False

    length_relations = certificate.get("invalid_length_relations")
    lengths_ok = isinstance(length_relations, dict) and isinstance(pstar, dict) and isinstance(rich_output, dict)
    actual_length_classes: set[str] = set()
    if lengths_ok:
        base_length = len(pstar.get("plan", []))
        for key, expected in length_relations.items():
            candidate = _candidate(rich_output, int(key))
            if candidate is None:
                lengths_ok = False
                break
            length = len(candidate.get("plan", []))
            actual = "lt" if length < base_length else "gt" if length > base_length else "eq"
            actual_length_classes.add(actual)
            if actual != expected:
                lengths_ok = False
                break
        lengths_ok = lengths_ok and actual_length_classes == {"lt", "eq", "gt"}

    skeleton_ok = False
    if isinstance(rich, dict) and isinstance(none, dict) and isinstance(none_output, dict):
        rich_ast = rich.get("program_ast")
        none_ast = none.get("program_ast")
        none_certificate = none.get("structure_certificate", {})
        differences = _diff_paths(rich_ast, none_ast)
        try:
            skeleton_ok = bool(
                differences == ["/budget"]
                and none_certificate.get("skeleton_partner") == rich.get("example_id")
                and none_certificate.get("changed_path") == "/budget"
                and none_certificate.get("from") == rich_ast["budget"]
                and none_certificate.get("to") == none_ast["budget"]
                and none_certificate.get("expected_valid_candidate_count") == 0
                and not any(row.get("valid") is True for row in none_output.get("candidates", []))
                and _remove_path(rich_ast, "/budget") == _remove_path(none_ast, "/budget")
            )
        except (KeyError, TypeError, ValueError):
            skeleton_ok = False

    ratio = composition.get("none_ratio")
    none_pair = composition.get("none_pair")
    ratio_ok = bool(
        profile_ok
        and ratio == {"numerator": 1, "denominator": 2}
        and none_pair == [rich.get("example_id"), none.get("example_id")]
    ) if isinstance(rich, dict) and isinstance(none, dict) else False

    derived_witnesses: dict[str, str] = {}
    if isinstance(rich, dict) and isinstance(rich_output, dict):
        try:
            derived_witnesses = derive_cps_composition(rich["program_ast"], rich_output)
        except Exception:
            derived_witnesses = {}
    required = composition.get("cps_required_witnesses")
    composition_ok = bool(
        isinstance(rich, dict)
        and composition.get("cps_record_id") == rich.get("example_id")
        and isinstance(required, list)
        and len(required) == len(set(required)) == 4
        and set(required) == COMPOSITION_WITNESSES
        and set(derived_witnesses) == COMPOSITION_WITNESSES
    )

    leak_count = 0
    for row in records:
        for text in _strings({"source_text": row.get("source_text"), "model_view": row.get("model_view")}):
            normalized = text.upper().replace("-", "_").replace(" ", "_")
            leak_count += normalized.count("PSTAR") + normalized.count("CANONICAL_CANDIDATE_ROLE")

    return {
        "derived": derived_witnesses,
        "g06": {
            "G06_M01_rich_none_profile_exact": _metric(profile_ok, int(profile_ok), 1, profile_ok, ["rich/NONE role profile mismatch"] if not profile_ok else []),
            "G06_M02_pstar_depth_unique_minimum_exact": _metric(pstar_ok, int(pstar_ok), 1, pstar_ok, ["P-star unique minimum/depth mismatch"] if not pstar_ok else []),
            "G06_M03_valid_suboptimal_exact": _metric(suboptimal_ok, int(suboptimal_ok), 1, suboptimal_ok, ["valid-suboptimal mismatch"] if not suboptimal_ok else []),
            "G06_M04_invalid_candidate_profile_exact": _metric(invalid_ok, int(invalid_ok), 1, invalid_ok, ["invalid candidate profile mismatch"] if not invalid_ok else []),
            "G06_M05_failure_reason_profile_exact": _metric(reasons_ok, int(reasons_ok), 1, reasons_ok, ["failure-reason profile mismatch"] if not reasons_ok else []),
            "G06_M06_invalid_length_profile_exact": _metric(lengths_ok, int(lengths_ok), 1, lengths_ok, ["invalid length profile mismatch"] if not lengths_ok else []),
            "G06_M07_none_single_leaf_skeleton_exact": _metric(skeleton_ok, int(skeleton_ok), 1, skeleton_ok, ["NONE single-leaf skeleton mismatch"] if not skeleton_ok else []),
            "G06_M08_none_ratio_exact": _metric(ratio_ok, int(ratio_ok), 1, ratio_ok, ["NONE ratio/pair mismatch"] if not ratio_ok else []),
            "G06_M09_derived_composition_witnesses_exact": _metric(composition_ok, int(composition_ok), 1, composition_ok, ["derived CPS composition mismatch"] if not composition_ok else []),
            "G06_M10_canonical_role_leak_count": _metric(leak_count, leak_count, 1, leak_count == 0, ["canonical role leak"] if leak_count else []),
        },
    }
