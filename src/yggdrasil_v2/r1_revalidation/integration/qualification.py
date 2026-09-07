from __future__ import annotations

"""Finite R0D audit that composes, but does not reimplement, R0A--R0C."""

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping

from yggdrasil_v2.r1_revalidation.audit import audit_invariant_bundle
from yggdrasil_v2.r1_revalidation.audit.provenance import derive_ere_provenance
from yggdrasil_v2.r1_revalidation.audit.structure import derive_cps_composition
from yggdrasil_v2.r1_revalidation.common import evaluate_cps, simulate_ere
from yggdrasil_v2.r1_revalidation.learner import audit_statistical_bundle
from yggdrasil_v2.r1_revalidation.learner.canonical import (
    LABELS,
    canonical_bytes,
    normalize_source,
    sha256_file,
    sha256_json,
    tree_hashes,
)
from yggdrasil_v2.r1_revalidation.learner.folds import (
    evaluate_grouped_cv,
    evaluate_train_heldout,
)
from yggdrasil_v2.r1_revalidation.learner.source import heuristic_prediction, parse_source


CONTRACT_VERSION = "r1r-p0d-v11-r0d-integrated"
CASES_SCHEMA = "yggdrasil.v2-r1r.p0d-v11.r0d-integrated.cases.v1"
EXPECTED_SCHEMA = "yggdrasil.v2-r1r.p0d-v11.r0d-integrated.expected.v1"
MATRIX_SCHEMA = "yggdrasil.v2-r1r.p0d-v11.r0d-integrated.matrix-spec.v1"
PRIOR_TREE_DIGESTS = {
    "prior-r0b": "9506BD1581B71D11B42D57EC2CB201F4F38385CC22D0C6CAB2A5B3B686543124",
    "prior-r0c": "BC0359404794C74D8F161E3EDE0C0C3F78FA53C4ECF78B8FBD8A8340D525711C",
}
CASE_IDS = tuple(f"I{index:02d}_" for index in range(1, 13))
HEURISTICS = {
    "ERE": (
        "initial_value",
        "first_rule_literal",
        "last_rule_literal",
        "first_event_literal",
        "last_event_literal",
        "last_mention",
    ),
    "CPS": (
        "first_candidate",
        "last_candidate",
        "shortest_candidate",
        "longest_candidate",
        "lowest_raw_cost",
        "highest_raw_cost",
        "first_definition",
    ),
}
METRIC_GATES = {
    "G01_M01_matrix_spec_exact": "G01",
    "G01_M02_completed_matrix_fixture_exact": "G01",
    "G02_M01_input_tree_and_seal_exact": "G02",
    "G02_M02_prior_bundle_identity_and_audit": "G02",
    "G02_M03_model_view_exact_boundary": "G02",
    "G03_M01_semantic_replay_digest_rate": "G03",
    "G03_M02_answer_label_binding_rate": "G03",
    "G04_M01_split_pair_group_contract_exact": "G04",
    "G04_M02_surface_parser_and_projection_exact": "G04",
    "G05_M01_ere_typed_provenance_rate": "G05",
    "G05_M02_ere_single_leaf_counterfactual_rate": "G05",
    "G06_M01_cps_composition_witness_rate": "G06",
    "G06_M02_cps_candidate_permutation_rate": "G06",
    "G07_M01_label_and_position_balance_exact": "G07",
    "G07_M02_heuristic_results_exact": "G07",
    "G07_M03_heuristic_accuracy_ceiling": "G07",
    "G08_M01_grouped_cv_results_exact": "G08",
    "G08_M02_train_heldout_results_exact": "G08",
    "G08_M03_statistical_ceiling_and_source_boundary": "G08",
}
GATE_METRICS = {
    gate: tuple(metric for metric, owner in METRIC_GATES.items() if owner == gate)
    for gate in (f"G{index:02d}" for index in range(1, 9))
}
FAULT_TARGETS = {
    "F401": "G01",
    "F402": "G02",
    "F403": "G02",
    "F404": "G02",
    "F405": "G03",
    "F406": "G03",
    "F407": "G04",
    "F408": "G04",
    "F409": "G04",
    "F410": "G05",
    "F411": "G05",
    "F412": "G06",
    "F413": "G06",
    "F414": "G06",
    "F415": "G07",
    "F416": "G07",
    "F417": "G07",
    "F418": "G07",
    "F419": "G08",
    "F420": "G08",
}
_PRIOR_AUDIT_CACHE: dict[tuple[str, str], tuple[bool, bool]] = {}


def _metric(value: Any, passed: bool, numerator: int, denominator: int, failure: str) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "passed": bool(passed),
        "failures": [] if passed else [failure],
    }


def _failed(reason: str) -> dict[str, Any]:
    return _metric(False, False, 0, 1, reason)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_digest(root: Path) -> str:
    return sha256_json(tree_hashes(root))


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


def _records(root: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in (root / "prior-r0b/records.jsonl").read_text(encoding="utf-8").splitlines()]
    return {str(row["example_id"]): row for row in rows}


def _materialize_ast(
    case: Mapping[str, Any],
    semantic_sources: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source = semantic_sources[case["semantic_source_id"]]
    ast = deepcopy(records[source["record_id"]]["program_ast"])
    selection = source["candidate_selection"]
    order = case["candidate_order"]
    if case["family"] == "ERE":
        if selection != [] or order != []:
            raise ValueError("ERE projection must not select or reorder candidates")
        return ast
    if selection != [0, 4, 5] or sorted(order) != [0, 1, 2]:
        raise ValueError("CPS projection is outside the frozen three-way lattice")
    selected = [deepcopy(ast["candidates"][index]) for index in selection]
    ast["candidates"] = [selected[index] for index in order]
    return ast


def _output_digest(case: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    if case["family"] == "ERE":
        return str(source["expected_output_sha256"])
    key = "-".join(str(index) for index in case["candidate_order"])
    return str(source["expected_output_sha256_by_order"][key])


def _answer_label(case: Mapping[str, Any], parsed: Any, output: Mapping[str, Any]) -> str | None:
    if case["family"] == "ERE":
        return parsed.value_to_label.get(output.get("answer"))
    answer = output.get("answer")
    return f"L{answer}" if isinstance(answer, int) and 0 <= answer < 3 else None


def _accuracy(predictions: list[Mapping[str, Any]]) -> Fraction:
    if not predictions:
        return Fraction(0, 1)
    correct = sum(row.get("truth") == row.get("prediction") for row in predictions)
    return Fraction(correct, len(predictions))


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _expected_matrix() -> dict[str, Any]:
    return {
        "schema_version": MATRIX_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "metric_registry": [
            {"metric_id": metric, "gate": METRIC_GATES[metric]}
            for metric in METRIC_GATES
        ],
        "fault_groups": [
            {"fault_id": fault, "target_gate": FAULT_TARGETS[fault]}
            for fault in FAULT_TARGETS
        ],
        "completed_matrix_fixture": {
            "positive_gates": {f"G{index:02d}": True for index in range(2, 9)},
            "fault_groups": list(FAULT_TARGETS),
            "metamorphic_ids": ["M01_relocation", "M02_case_order", "M03_json_format", "M04_line_endings"],
            "import_boundary": True,
            "artifact_internal_replay": True,
        },
    }


def _load_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    cases = _read_json(root / "integration-cases.json")
    expected = _read_json(root / "expected-results.json")
    matrix = _read_json(root / "matrix-spec.json")
    seal = _read_json(root / "input-seal.json")
    manifest = _read_json(root / "manifest.json")
    actual_seal = tree_hashes(root, exclude=("input-seal.json",))
    seal_ok = (
        seal.get("schema_version") == "yggdrasil.v2-r1r.p0d-v11.r0d-integrated.input-seal.v1"
        and seal.get("files") == actual_seal
    )
    manifest_files = tree_hashes(root, exclude=("manifest.json", "input-seal.json"))
    manifest_ok = (
        manifest.get("schema_version") == "yggdrasil.v2-r1r.p0d-v11.r0d-integrated.manifest.v1"
        and manifest.get("contract_version") == CONTRACT_VERSION
        and manifest.get("case_count") == 12
        and manifest.get("files") == manifest_files
        and manifest.get("prior_tree_digests") == PRIOR_TREE_DIGESTS
    )
    return cases, expected, matrix, bool(seal_ok and manifest_ok)


def audit_integrated_bundle(root: str | Path) -> dict[str, Any]:
    """Audit one sealed v11 integration bundle without modifying its bytes."""

    bundle = Path(root)
    metrics = {metric: _failed("metric was not evaluated") for metric in METRIC_GATES}
    failures: list[str] = []
    cases: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    matrix: dict[str, Any] = {}
    integrity_ok = False
    try:
        cases, expected, matrix, integrity_ok = _load_inputs(bundle)
    except Exception as exc:
        failures.append(f"load:{type(exc).__name__}:{exc}")

    # G01: non-recursive hand-authored matrix fixture, independent of the real runner ledger.
    try:
        wanted = _expected_matrix()
        registry_ok = (
            matrix.get("schema_version") == wanted["schema_version"]
            and matrix.get("contract_version") == wanted["contract_version"]
            and matrix.get("metric_registry") == wanted["metric_registry"]
            and matrix.get("fault_groups") == wanted["fault_groups"]
        )
        fixture_ok = matrix.get("completed_matrix_fixture") == wanted["completed_matrix_fixture"]
        metrics["G01_M01_matrix_spec_exact"] = _metric(registry_ok, registry_ok, int(registry_ok), 1, "matrix registry/fault binding mismatch")
        metrics["G01_M02_completed_matrix_fixture_exact"] = _metric(fixture_ok, fixture_ok, int(fixture_ok), 1, "completed matrix fixture mismatch")
    except Exception as exc:
        failures.append(f"G01:{type(exc).__name__}:{exc}")

    # G02: sealed input, exact accepted prior bundles, and a strict source-only model view.
    records: dict[str, dict[str, Any]] = {}
    try:
        metrics["G02_M01_input_tree_and_seal_exact"] = _metric(integrity_ok, integrity_ok, int(integrity_ok), 1, "input tree, manifest, or seal mismatch")
        prior_digests = {name: _tree_digest(bundle / name) for name in PRIOR_TREE_DIGESTS}
        cache_key = (prior_digests["prior-r0b"], prior_digests["prior-r0c"])
        if cache_key not in _PRIOR_AUDIT_CACHE:
            r0b = audit_invariant_bundle(bundle / "prior-r0b")
            r0c = audit_statistical_bundle(bundle / "prior-r0c")
            _PRIOR_AUDIT_CACHE[cache_key] = (r0b.get("passed") is True, r0c.get("passed") is True)
        r0b_ok, r0c_ok = _PRIOR_AUDIT_CACHE[cache_key]
        prior_ok = prior_digests == PRIOR_TREE_DIGESTS and r0b_ok and r0c_ok
        metrics["G02_M02_prior_bundle_identity_and_audit"] = _metric(prior_ok, prior_ok, int(prior_ok), 1, "accepted prior bundle identity or audit mismatch")
        records = _records(bundle)
        rows = cases.get("cases", [])
        profile_ok = (
            cases.get("schema_version") == CASES_SCHEMA
            and cases.get("contract_version") == CONTRACT_VERSION
            and cases.get("labels") == list(LABELS)
            and len(rows) == 12
            and sorted(str(row.get("case_id", "")) for row in rows)
            == sorted(
                "I01_ere_open_map0 I02_ere_closed_map0 I03_ere_open_map1 I04_ere_closed_map1 "
                "I05_ere_open_map2 I06_ere_closed_map2 I07_cps_order012 I08_cps_order102 "
                "I09_cps_order021 I10_cps_order120 I11_cps_order201 I12_cps_order210".split()
            )
        )
        boundary_ok = profile_ok and all(
            set(row) == {
                "case_id", "family", "semantic_source_id", "candidate_order", "group_id",
                "partition", "pair_id", "pair_role", "expected_label", "model_view",
            }
            and set(row["model_view"]) == {"family", "source_text", "valid_choice_mask"}
            and row["model_view"]["family"] == row["family"]
            and row["model_view"]["valid_choice_mask"] == [True, True, True]
            and isinstance(row["model_view"]["source_text"], str)
            and len(normalize_source(row["model_view"]["source_text"])) <= 400
            for row in rows
        )
        metrics["G02_M03_model_view_exact_boundary"] = _metric(boundary_ok, boundary_ok, int(boundary_ok), 1, "model view schema, mask, or forbidden-field boundary mismatch")
    except Exception as exc:
        failures.append(f"G02:{type(exc).__name__}:{exc}")

    outputs: dict[str, dict[str, Any]] = {}
    asts: dict[str, dict[str, Any]] = {}
    parsed_sources: dict[str, Any] = {}
    # G03: the accepted simulator is the sole semantic oracle; labels bind outside model_view.
    try:
        semantic_sources = cases["semantic_sources"]
        digest_good = 0
        binding_good = 0
        for case in cases["cases"]:
            ast = _materialize_ast(case, semantic_sources, records)
            output = simulate_ere(ast) if case["family"] == "ERE" else evaluate_cps(ast)
            parsed = parse_source(case["model_view"]["source_text"], case["family"])
            source = semantic_sources[case["semantic_source_id"]]
            asts[case["case_id"]] = ast
            outputs[case["case_id"]] = output
            parsed_sources[case["case_id"]] = parsed
            digest_good += sha256_json(output) == _output_digest(case, source)
            semantic_answer_ok = (
                output.get("answer") == source["expected_answer"]
                if case["family"] == "ERE"
                else output.get("answer") is not None
            )
            binding_good += semantic_answer_ok and _answer_label(case, parsed, output) == case["expected_label"]
        denominator = len(cases["cases"])
        metrics["G03_M01_semantic_replay_digest_rate"] = _metric(
            digest_good / denominator, digest_good == denominator, digest_good, denominator, "fresh simulator output digest mismatch"
        )
        metrics["G03_M02_answer_label_binding_rate"] = _metric(
            binding_good / denominator, binding_good == denominator, binding_good, denominator, "semantic answer does not bind to the public label"
        )
    except Exception as exc:
        failures.append(f"G03:{type(exc).__name__}:{exc}")

    # G04: pair/group/split identity plus a parser-to-projection interface with no label adapter.
    try:
        rows = cases["cases"]
        train_groups = {row["group_id"] for row in rows if row["partition"] == "train"}
        heldout_groups = {row["group_id"] for row in rows if row["partition"] == "heldout"}
        pair_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            pair_groups.setdefault(row["pair_id"], []).append(row)
        split_ok = (
            {row["partition"] for row in rows} == {"train", "heldout"}
            and not (train_groups & heldout_groups)
            and len(train_groups) == 3
            and len(heldout_groups) == 3
            and len(pair_groups) == 6
            and all(len(group) == 2 and len({row["group_id"] for row in group}) == 1 for group in pair_groups.values())
            and all(
                {row["pair_role"] for row in group}
                == ({"open", "closed"} if group[0]["family"] == "ERE" else {"a", "b"})
                for group in pair_groups.values()
            )
        )
        parser_good = 0
        for row in rows:
            parsed = parsed_sources[row["case_id"]]
            ast = asts[row["case_id"]]
            source_ok = parsed.normalized == normalize_source(row["model_view"]["source_text"])
            if row["family"] == "ERE":
                mode = ast["initial_state"]["attributes"]["flag"]["mode"]
                source_ok = source_ok and f"Event mode_{mode}:" in parsed.normalized
            else:
                action_costs = {action["name"]: action["cost"] for action in ast["actions"]}
                visible = [(label, list(plan), cost) for label, plan, cost in parsed.candidates]
                projected = [
                    (f"L{index}", candidate["plan"], sum(action_costs[name] for name in candidate["plan"]))
                    for index, candidate in enumerate(ast["candidates"])
                ]
                source_ok = source_ok and list(parsed.actions) == [action["name"] for action in ast["actions"]] and visible == projected
            parser_good += source_ok
        denominator = len(rows)
        metrics["G04_M01_split_pair_group_contract_exact"] = _metric(split_ok, split_ok, int(split_ok), 1, "split, pair, or group contract mismatch")
        metrics["G04_M02_surface_parser_and_projection_exact"] = _metric(
            parser_good / denominator, parser_good == denominator, parser_good, denominator, "toy surface does not exactly project the sealed case"
        )
    except Exception as exc:
        failures.append(f"G04:{type(exc).__name__}:{exc}")

    # G05: accepted typed provenance plus fresh single-leaf open/closed counterfactuals.
    try:
        ere_rows = [row for row in cases["cases"] if row["family"] == "ERE"]
        provenance_good = 0
        for row in ere_rows:
            derived = derive_ere_provenance(asts[row["case_id"]], outputs[row["case_id"]])
            expected_class = cases["semantic_sources"][row["semantic_source_id"]]["expected_provenance"]
            provenance_good += derived["derived_class"] == expected_class and derived["trace_binding"] is True
        counterfactual_good = 0
        for pair_id in sorted({row["pair_id"] for row in ere_rows}):
            pair = [row for row in ere_rows if row["pair_id"] == pair_id]
            by_role = {row["pair_role"]: row for row in pair}
            if set(by_role) != {"open", "closed"}:
                continue
            left, right = by_role["open"], by_role["closed"]
            diff = _diff_paths(asts[left["case_id"]], asts[right["case_id"]])
            counterfactual_good += (
                diff == ["/initial_state/attributes/flag/mode"]
                and outputs[left["case_id"]]["answer"] == "allow"
                and outputs[right["case_id"]]["answer"] == "deny"
                and left["expected_label"] != right["expected_label"]
            )
        metrics["G05_M01_ere_typed_provenance_rate"] = _metric(
            provenance_good / len(ere_rows), provenance_good == len(ere_rows), provenance_good, len(ere_rows), "typed ERE provenance mismatch"
        )
        metrics["G05_M02_ere_single_leaf_counterfactual_rate"] = _metric(
            counterfactual_good / 3, counterfactual_good == 3, counterfactual_good, 3, "ERE counterfactual is not a single-leaf answer flip"
        )
    except Exception as exc:
        failures.append(f"G05:{type(exc).__name__}:{exc}")

    # G06: accepted CPS composition derivation and candidate-order equivariance.
    try:
        cps_rows = [row for row in cases["cases"] if row["family"] == "CPS"]
        required = cases["semantic_sources"]["cps_three_way"]["expected_composition"]
        winner_plan = cases["semantic_sources"]["cps_three_way"]["winner_plan"]
        composition_good = 0
        permutation_good = 0
        for row in cps_rows:
            output = outputs[row["case_id"]]
            derived = derive_cps_composition(asts[row["case_id"]], output)
            composition_good += sorted(derived) == required
            winner = output["candidates"][output["answer"]] if output.get("answer") is not None else {}
            invalid = [candidate for candidate in output["candidates"] if candidate["valid"] is False]
            permutation_good += (
                output.get("unique_optimum") is True
                and winner.get("plan") == winner_plan
                and f"L{output['answer']}" == row["expected_label"]
                and sorted(reason for candidate in invalid for reason in candidate["failure_reasons"]) == ["budget", "final_constraint"]
            )
        metrics["G06_M01_cps_composition_witness_rate"] = _metric(
            composition_good / len(cps_rows), composition_good == len(cps_rows), composition_good, len(cps_rows), "derived CPS composition witnesses mismatch"
        )
        metrics["G06_M02_cps_candidate_permutation_rate"] = _metric(
            permutation_good / len(cps_rows), permutation_good == len(cps_rows), permutation_good, len(cps_rows), "CPS winning plan is not equivariant to candidate order"
        )
    except Exception as exc:
        failures.append(f"G06:{type(exc).__name__}:{exc}")

    heuristic_results: dict[str, Any] = {}
    # G07: deterministic parsers touch the exact same model_view bytes.
    try:
        rows = cases["cases"]
        labels = Counter(row["expected_label"] for row in rows)
        ere_labels = Counter(row["expected_label"] for row in rows if row["family"] == "ERE")
        cps_labels = Counter(row["expected_label"] for row in rows if row["family"] == "CPS")
        balance_ok = labels == Counter({"L0": 4, "L1": 4, "L2": 4}) and ere_labels == cps_labels == Counter({"L0": 2, "L1": 2, "L2": 2})
        prediction_rows: list[dict[str, str]] = []
        for row in rows:
            for heuristic in HEURISTICS[row["family"]]:
                result = heuristic_prediction(
                    row["model_view"]["source_text"],
                    row["family"],
                    heuristic,
                    row["model_view"]["valid_choice_mask"],
                )
                prediction_rows.append(
                    {
                        "case_id": row["case_id"],
                        "heuristic": heuristic,
                        "truth": row["expected_label"],
                        "prediction": result["prediction"],
                    }
                )
        prediction_rows.sort(key=lambda item: (item["case_id"], item["heuristic"]))
        accuracies: dict[str, str] = {}
        for family in HEURISTICS:
            for heuristic in HEURISTICS[family]:
                subset = [
                    row for row in prediction_rows
                    if row["heuristic"] == heuristic and next(case for case in rows if case["case_id"] == row["case_id"])["family"] == family
                ]
                accuracies[f"{family}:{heuristic}"] = _fraction(
                    Fraction(sum(row["truth"] == row["prediction"] for row in subset), len(subset))
                )
        heuristic_results = {"predictions": prediction_rows, "accuracies": accuracies}
        heuristic_projection = {"sha256": sha256_json(heuristic_results), "accuracies": accuracies}
        exact = expected.get("heuristics") == heuristic_projection
        ceiling_ok = all(Fraction(value) <= Fraction(1, 2) for value in accuracies.values())
        metrics["G07_M01_label_and_position_balance_exact"] = _metric(balance_ok, balance_ok, int(balance_ok), 1, "label or correct-position balance mismatch")
        metrics["G07_M02_heuristic_results_exact"] = _metric(exact, exact, int(exact), 1, "deterministic heuristic golden mismatch")
        metrics["G07_M03_heuristic_accuracy_ceiling"] = _metric(
            max(accuracies.values(), key=lambda value: Fraction(value)), ceiling_ok, int(ceiling_ok), 1, "a deterministic surface heuristic exceeds one half"
        )
    except Exception as exc:
        failures.append(f"G07:{type(exc).__name__}:{exc}")

    statistical_results: dict[str, Any] = {}
    # G08: accepted NB/grouped evaluators consume only row_id/group/text/label/mask.
    try:
        rows = cases["cases"]
        learner_rows = [
            {
                "row_id": row["case_id"],
                "group_id": row["group_id"],
                "text": row["model_view"]["source_text"],
                "label": row["expected_label"],
                "valid_choice_mask": row["model_view"]["valid_choice_mask"],
            }
            for row in rows
        ]
        cv: dict[str, Any] = {}
        heldout: dict[str, Any] = {}
        for analyzer in ("word", "char_3_5"):
            cv[analyzer] = evaluate_grouped_cv(
                {"case_id": f"R0D_CV_{analyzer}", "analyzer": analyzer, "fold_count": 5, "rows": learner_rows}
            )
            heldout[analyzer] = evaluate_train_heldout(
                {
                    "case_id": f"R0D_HELDOUT_{analyzer}",
                    "analyzer": analyzer,
                    "train_rows": [row for row, case in zip(learner_rows, rows, strict=True) if case["partition"] == "train"],
                    "heldout_rows": [row for row, case in zip(learner_rows, rows, strict=True) if case["partition"] == "heldout"],
                }
            )
        statistical_results = {"cv": cv, "heldout": heldout}
        cv_projection = {
            analyzer: {
                "sha256": sha256_json(result),
                "confusion": result["confusion"],
                "accuracy": _fraction(_accuracy(result["predictions"])),
            }
            for analyzer, result in cv.items()
        }
        heldout_projection = {
            analyzer: {
                "sha256": sha256_json(result),
                "confusion": result["confusion"],
                "accuracy": _fraction(_accuracy(result["predictions"])),
            }
            for analyzer, result in heldout.items()
        }
        cv_exact = expected.get("cv") == cv_projection
        heldout_exact = expected.get("heldout") == heldout_projection
        accuracies = {
            f"cv:{analyzer}": _accuracy(result["predictions"])
            for analyzer, result in cv.items()
        } | {
            f"heldout:{analyzer}": _accuracy(result["predictions"])
            for analyzer, result in heldout.items()
        }
        source_boundary = all(
            set(row) == {"row_id", "group_id", "text", "label", "valid_choice_mask"}
            and row["text"] == case["model_view"]["source_text"]
            for row, case in zip(learner_rows, rows, strict=True)
        )
        ceiling_ok = source_boundary and all(value <= Fraction(1, 2) for value in accuracies.values())
        metrics["G08_M01_grouped_cv_results_exact"] = _metric(cv_exact, cv_exact, int(cv_exact), 1, "grouped CV result differs from the independent golden")
        metrics["G08_M02_train_heldout_results_exact"] = _metric(heldout_exact, heldout_exact, int(heldout_exact), 1, "train-heldout result differs from the independent golden")
        metrics["G08_M03_statistical_ceiling_and_source_boundary"] = _metric(
            {key: _fraction(value) for key, value in accuracies.items()}, ceiling_ok, int(ceiling_ok), 1, "learned shortcut exceeds one half or consumes a forbidden field"
        )
    except Exception as exc:
        failures.append(f"G08:{type(exc).__name__}:{exc}")

    gates = {
        gate: all(metrics[metric]["passed"] is True for metric in metric_ids)
        for gate, metric_ids in GATE_METRICS.items()
    }
    failed_metrics = [metric for metric in METRIC_GATES if metrics[metric]["passed"] is not True]
    failures.extend(failed_metrics)
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v11.r0d-integrated.audit-report.v1",
        "contract_version": CONTRACT_VERSION,
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "results": {
            "heuristics_sha256": sha256_json(heuristic_results),
            "statistical_sha256": sha256_json(statistical_results),
        },
        "failures": sorted(set(failures)),
    }


__all__ = [
    "CONTRACT_VERSION",
    "FAULT_TARGETS",
    "GATE_METRICS",
    "METRIC_GATES",
    "audit_integrated_bundle",
]
