from __future__ import annotations

"""Independent semantic replay and exact claim algebra for G03."""

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from yggdrasil_v2.r1_revalidation.common import evaluate_cps, replay_cps_prefix, simulate_ere


REQUIRED_CLAIM_KINDS = {
    "ere": {"attribute_value", "relation_exists", "condition_truth"},
    "cps": {
        "prefix_legality",
        "resource_value",
        "cost_value",
        "fact_truth",
        "goal_status",
        "final_constraint_status",
    },
}


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _metric(value: Any, numerator: int, denominator: int, passed: bool, failures: list[str]) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "passed": bool(passed),
        "failures": sorted(set(str(item) for item in failures)),
    }


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


def _state_value(state: Any, path: list[str]) -> Any:
    value = state
    for part in path:
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list):
            value = value[int(part)]
        else:
            raise KeyError(part)
    return value


def _candidate_key(family: str, answer: Any) -> str:
    if family == "cps":
        return "NONE" if answer is None else f"candidate:{answer}"
    return str(answer)


def _answer_contract(row: dict[str, Any], output: dict[str, Any]) -> bool:
    family = row.get("family")
    if not isinstance(output, dict) or output.get("answer") != row.get("answer_semantic"):
        return False
    mapping = row.get("label_mapping")
    if not isinstance(mapping, dict) or mapping.get(_candidate_key(str(family), output.get("answer"))) != row.get("answer_index"):
        return False
    mask = row.get("valid_choice_mask")
    if not isinstance(mask, list) or len(mask) != len(mapping):
        return False
    if family == "ere":
        trace = output.get("trace")
        if not isinstance(trace, list) or len(trace) != row.get("reasoning_budget"):
            return False
        return all(isinstance(item, dict) and item.get("delta_budget") == 1 for item in trace)
    candidates = output.get("candidates")
    if not isinstance(candidates, list) or output.get("unique_optimum") not in (True, False):
        return False
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or candidate.get("candidate_index") != index:
            return False
        plan = candidate.get("plan")
        trace = candidate.get("trace")
        if not isinstance(plan, list) or not isinstance(trace, list) or len(trace) != len(plan):
            return False
        for position, step in enumerate(trace):
            if not isinstance(step, dict) or step.get("position") != position or step.get("action") != plan[position]:
                return False
    return True


def _truth_ere(ast: dict[str, Any], predicate: dict[str, Any]) -> bool:
    prefix = predicate.get("prefix")
    if not isinstance(prefix, int) or prefix < 0:
        raise ValueError("invalid ERE prefix")
    output = simulate_ere(deepcopy(ast), prefix=prefix)
    state = output["final_state"]
    kind = predicate.get("kind")
    if kind == "attribute_value" or kind == "condition_truth":
        return _state_value(state, ["attributes", str(predicate["entity"]), str(predicate["attribute"])]) == predicate["value"]
    if kind == "relation_exists":
        edges = _state_value(state, ["relations", str(predicate["relation"])])
        return [predicate["source"], predicate["target"]] in edges
    raise ValueError("unknown ERE claim kind")


def _goal_truth(ast: dict[str, Any], state: dict[str, Any], key: str) -> bool:
    conditions = ast.get(key)
    if not isinstance(conditions, list):
        return False
    for condition in conditions:
        if condition.get("kind") != "fact_true" or condition.get("fact") not in state.get("facts", []):
            return False
    return True


def _truth_cps(ast: dict[str, Any], predicate: dict[str, Any]) -> bool:
    candidate_index = predicate.get("candidate_index")
    prefix = predicate.get("prefix")
    if not isinstance(candidate_index, int) or not isinstance(prefix, int):
        raise ValueError("invalid CPS reference")
    output = replay_cps_prefix(deepcopy(ast), candidate_index, prefix)
    kind = predicate.get("kind")
    if kind == "prefix_legality":
        return output.get("legal_so_far") is predicate.get("expected")
    if kind == "resource_value":
        return output.get("state", {}).get("resources", {}).get(predicate.get("resource")) == predicate.get("value")
    if kind == "cost_value":
        return output.get("cost") == predicate.get("value")
    if kind == "fact_truth":
        present = predicate.get("fact") in output.get("state", {}).get("facts", [])
        return present is predicate.get("expected")
    if kind == "goal_status":
        return _goal_truth(ast, output["state"], "goal") is predicate.get("expected")
    if kind == "final_constraint_status":
        return _goal_truth(ast, output["state"], "final_constraints") is predicate.get("expected")
    raise ValueError("unknown CPS claim kind")


_PARSERS = {
    "attribute_value": re.compile(r"At ERE prefix (\d+), entity ([^ ]+) has attribute ([^ ]+) equal to ([^.]+)\."),
    "relation_exists": re.compile(r"At ERE prefix (\d+), relation ([^ ]+) contains edge ([^ ]+) to ([^.]+)\."),
    "condition_truth": re.compile(r"At ERE prefix (\d+), condition ([^.]+)\.([^ ]+) equals ([^.]+)\."),
    "prefix_legality": re.compile(r"For CPS candidate (\d+) at prefix (\d+), legality is (true|false)\."),
    "resource_value": re.compile(r"For CPS candidate (\d+) at prefix (\d+), resource ([^ ]+) equals (-?\d+)\."),
    "cost_value": re.compile(r"For CPS candidate (\d+) at prefix (\d+), accumulated cost equals (-?\d+)\."),
    "fact_truth": re.compile(r"For CPS candidate (\d+) at prefix (\d+), fact ([^ ]+) is (true|false)\."),
    "goal_status": re.compile(r"For CPS candidate (\d+) at prefix (\d+), the goal status is (true|false)\."),
    "final_constraint_status": re.compile(r"For CPS candidate (\d+) at prefix (\d+), final-constraint status is (true|false)\."),
}


def _parse_text(kind: str, text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or kind not in _PARSERS:
        return None
    match = _PARSERS[kind].fullmatch(text)
    if match is None:
        return None
    groups = match.groups()
    if kind == "attribute_value":
        return {"attribute": groups[2], "entity": groups[1], "prefix": int(groups[0]), "value": groups[3]}
    if kind == "relation_exists":
        return {"prefix": int(groups[0]), "relation": groups[1], "source": groups[2], "target": groups[3]}
    if kind == "condition_truth":
        return {"attribute": groups[2], "entity": groups[1], "prefix": int(groups[0]), "value": groups[3]}
    candidate = int(groups[0])
    prefix = int(groups[1])
    if kind == "prefix_legality":
        return {"candidate_index": candidate, "expected": groups[2] == "true", "prefix": prefix}
    if kind == "resource_value":
        return {"candidate_index": candidate, "prefix": prefix, "resource": groups[2], "value": int(groups[3])}
    if kind == "cost_value":
        return {"candidate_index": candidate, "prefix": prefix, "value": int(groups[2])}
    if kind == "fact_truth":
        return {"candidate_index": candidate, "expected": groups[3] == "true", "fact": groups[2], "prefix": prefix}
    return {"candidate_index": candidate, "expected": groups[2] == "true", "prefix": prefix}


def _claim_truth(row: dict[str, Any], claim: dict[str, Any]) -> bool:
    family = str(row.get("family"))
    predicate = claim.get("predicate")
    if not isinstance(predicate, dict):
        raise ValueError("claim predicate is not an object")
    predicate = deepcopy(predicate)
    predicate["kind"] = claim.get("kind")
    return _truth_ere(row["program_ast"], predicate) if family == "ere" else _truth_cps(row["program_ast"], predicate)


def replay_bundle(records: list[dict[str, Any]]) -> dict[str, Any]:
    ere_rows = [row for row in records if row.get("family") == "ere"]
    cps_rows = [row for row in records if row.get("family") == "cps"]
    outputs: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    ere_digest_good = 0
    cps_digest_good = 0
    contract_good = 0
    claim_rows: list[dict[str, Any]] = []
    for row in records:
        identifier = str(row.get("example_id", ""))
        try:
            ast = deepcopy(row["program_ast"])
            output = simulate_ere(ast) if row.get("family") == "ere" else evaluate_cps(ast)
            outputs[identifier] = output
            digest_ok = _digest(output) == row.get("teacher_output_sha256")
            if row.get("family") == "ere" and digest_ok:
                ere_digest_good += 1
            if row.get("family") == "cps" and digest_ok:
                cps_digest_good += 1
            contract_ok = digest_ok and _answer_contract(row, output)
            if contract_ok:
                contract_good += 1
            if not digest_ok:
                failures.append(f"digest:{identifier}")
            if not contract_ok:
                failures.append(f"contract:{identifier}")
        except Exception as exc:
            failures.append(f"replay:{identifier}:{type(exc).__name__}")
    for row in records:
        claims = row.get("claims", [])
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            item = {"row": row, "claim": claim, "truth": None, "parsed": None, "error": None}
            try:
                item["truth"] = _claim_truth(row, claim)
            except Exception as exc:
                item["error"] = f"truth:{type(exc).__name__}"
            item["parsed"] = _parse_text(str(claim.get("kind")), claim.get("text"))
            claim_rows.append(item)

    required = set().union(*REQUIRED_CLAIM_KINDS.values())
    kind_counts: dict[str, int] = {kind: 0 for kind in required}
    family_kind_good = True
    for item in claim_rows:
        kind = str(item["claim"].get("kind"))
        if kind in kind_counts:
            kind_counts[kind] += 1
        family = str(item["row"].get("family"))
        if kind not in REQUIRED_CLAIM_KINDS.get(family, set()):
            family_kind_good = False
    profile_ok = len(claim_rows) == 18 and family_kind_good and all(count == 2 for count in kind_counts.values())
    claim_pairs: dict[str, list[dict[str, Any]]] = {}
    for item in claim_rows:
        pair_id = item["claim"].get("pair_id")
        claim_pairs.setdefault(str(pair_id), []).append(item)
    claim_ids = [str(item["claim"].get("claim_id")) for item in claim_rows]
    identity_ok = bool(
        len(claim_rows) == 18
        and len(set(claim_ids)) == len(claim_ids)
        and all(identifier.strip() for identifier in claim_ids)
        and len(claim_pairs) == 9
        and all(pair_id.strip() and len(items) == 2 for pair_id, items in claim_pairs.items())
    )
    truth_good = 0
    text_good = 0
    pair_good = 0
    polarity_good = 0
    for item in claim_rows:
        claim = item["claim"]
        if item["truth"] is not None and item["truth"] is bool(claim.get("label")):
            truth_good += 1
        if item["parsed"] == claim.get("predicate"):
            text_good += 1
    for pair_id, items in claim_pairs.items():
        if len(items) != 2:
            continue
        left, right = items
        left_claim, right_claim = left["claim"], right["claim"]
        same_record = left["row"].get("example_id") == right["row"].get("example_id")
        roles = {left_claim.get("pair_role"): left_claim, right_claim.get("pair_role"): right_claim}
        if (
            set(roles) == {"positive", "negative"}
            and roles["positive"].get("label") is True
            and roles["negative"].get("label") is False
        ):
            polarity_good += 2
        pair_good += int(
            same_record
            and left_claim.get("kind") == right_claim.get("kind")
            and {left_claim.get("pair_role"), right_claim.get("pair_role")} == {"positive", "negative"}
            and _diff_paths(left_claim.get("predicate"), right_claim.get("predicate"))
            and len(_diff_paths(left_claim.get("predicate"), right_claim.get("predicate"))) == 1
            and left_claim.get("label") is not right_claim.get("label")
        )
    ere_denominator = len(ere_rows)
    cps_denominator = len(cps_rows)
    claim_denominator = len(claim_rows)
    pair_denominator = len(claim_pairs)
    return {
        "outputs": outputs,
        "claims": claim_rows,
        "pair_claims": claim_pairs,
        "failures": sorted(set(failures)),
        "g03": {
            "G03_M01_ere_replay_digest_rate": _metric(ere_digest_good / ere_denominator if ere_denominator else 0.0, ere_digest_good, ere_denominator, ere_denominator > 0 and ere_digest_good == ere_denominator, ["ERE replay digest mismatch"] if ere_digest_good != ere_denominator else []),
            "G03_M02_cps_replay_digest_rate": _metric(cps_digest_good / cps_denominator if cps_denominator else 0.0, cps_digest_good, cps_denominator, cps_denominator > 0 and cps_digest_good == cps_denominator, ["CPS replay digest mismatch"] if cps_digest_good != cps_denominator else []),
            "G03_M03_answer_teacher_budget_contract_rate": _metric(contract_good / len(records) if records else 0.0, contract_good, len(records), bool(records) and contract_good == len(records), ["answer/teacher/budget contract mismatch"] if contract_good != len(records) else []),
            "G03_M04_claim_kind_profile_exact": _metric(profile_ok, int(profile_ok), 1, profile_ok, ["claim kind/family/cardinality profile mismatch"] if not profile_ok else []),
            "G03_M05_claim_identity_profile_exact": _metric(identity_ok, int(identity_ok), 1, identity_ok, ["claim or pair identity profile mismatch"] if not identity_ok else []),
            "G03_M06_claim_polarity_binding_rate": _metric(polarity_good / claim_denominator if claim_denominator else 0.0, polarity_good, claim_denominator, claim_denominator == 18 and polarity_good == 18, ["positive/negative role is decoupled from label"] if polarity_good != claim_denominator else []),
            "G03_M07_fresh_claim_truth_rate": _metric(truth_good / claim_denominator if claim_denominator else 0.0, truth_good, claim_denominator, claim_denominator == 18 and truth_good == 18, ["fresh claim truth/label mismatch"] if truth_good != claim_denominator else []),
            "G03_M08_claim_single_leaf_pair_rate": _metric(pair_good / pair_denominator if pair_denominator else 0.0, pair_good, pair_denominator, pair_denominator == 9 and pair_good == 9, ["claim pair is not a single-leaf contrast"] if pair_good != pair_denominator else []),
            "G03_M09_claim_text_parse_rate": _metric(text_good / claim_denominator if claim_denominator else 0.0, text_good, claim_denominator, claim_denominator == 18 and text_good == 18, ["claim text parse mismatch"] if text_good != claim_denominator else []),
        },
    }
