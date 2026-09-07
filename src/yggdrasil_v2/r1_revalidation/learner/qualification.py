from __future__ import annotations

"""Finite R0C qualification audit for source-only and statistical learners."""

from collections import Counter
from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import (
    ContractError,
    LABELS,
    exact_keys,
    require_labels,
    sha256_json,
    tree_hashes,
    normalize_source,
)
from .folds import assign_grouped_folds, evaluate_grouped_cv, evaluate_train_heldout
from .models import (
    character_ngrams,
    evaluate_categorical_case,
    evaluate_nb_case,
    fit_conditional_majority,
    fit_multinomial_nb,
    word_tokens,
)
from .source import heuristic_prediction


CONTRACT_VERSION = "r1r-p0d-v10-r0c-statistical"
INPUT_SCHEMA = "yggdrasil.v2-r1r.p0d-v10.r0c-statistical.inputs.v1"
EXPECTED_SCHEMA = "yggdrasil.v2-r1r.p0d-v10.r0c-statistical.expected.v1"
INPUT_SEMANTIC_SHA256 = "D3D168DC1C96B34A74E90B1DF12F54B1260B1327724D9484E514762CE8598615"
EXPECTED_SEMANTIC_SHA256 = "2F4B36C9D78BEAC5D2832C66E3DC723B0AE5116CB8691886CE31F248C78779F9"

PROFILE = {
    "parser_case_ids": [
        "P01_initial", "P02_first_rule", "P03_last_rule", "P04_first_event",
        "P05_last_event", "P06_last_mention", "P07_first_candidate",
        "P08_last_candidate", "P09_shortest", "P10_longest", "P11_lowest_cost",
        "P12_highest_cost", "P13_first_definition", "P14_mask_fallback",
    ],
    "categorical_case_ids": ["C01"],
    "analyzer_case_ids": ["A01_word", "A02_char", "A03_unicode_word"],
    "nb_case_ids": ["NB01_word", "NB02_char"],
    "fold_case_ids": ["F01"],
    "cv_case_ids": ["CV01"],
    "heldout_case_ids": ["H01"],
    "negative_case_ids": [f"N{index:02d}" for index in range(1, 11)],
}


METRIC_GATES = {
    "G07_M01_bundle_schema_exact": "G07",
    "G07_M02_manifest_input_seal_exact": "G07",
    "G07_M03_profile_ids_counts_exact": "G07",
    "G07_M04_parser_case_schema_exact": "G07",
    "G07_M05_question_view_exact": "G07",
    "G07_M06_surface_counts_exact": "G07",
    "G07_M07_heuristic_predictions_exact": "G07",
    "G07_M08_parser_surface_only_boundary": "G07",
    "G07_M09_categorical_case_schema_exact": "G07",
    "G07_M10_categorical_counts_exact": "G07",
    "G07_M11_categorical_predictions_exact": "G07",
    "G07_M12_categorical_tie_mask_unseen_exact": "G07",
    "G07_M13_parser_negative_controls_exact": "G07",
    "G08_M01_analyzer_case_schema_exact": "G08",
    "G08_M02_word_tokens_exact": "G08",
    "G08_M03_character_ngrams_exact": "G08",
    "G08_M04_nb_case_schema_exact": "G08",
    "G08_M05_vocabulary_order_exact": "G08",
    "G08_M06_vocabulary_hash_exact": "G08",
    "G08_M07_nb_class_feature_counts_exact": "G08",
    "G08_M08_nb_exact_scores_predictions": "G08",
    "G08_M09_nb_oov_mask_tie_exact": "G08",
    "G08_M10_fold_case_schema_exact": "G08",
    "G08_M11_grouped_fold_map_exact": "G08",
    "G08_M13_fold_balance_exact": "G08",
    "G08_M14_cv_case_schema_exact": "G08",
    "G08_M15_cv_predictions_exact": "G08",
    "G08_M16_cv_confusion_exact": "G08",
    "G08_M17_cv_fold_state_exact": "G08",
    "G08_M18_heldout_case_schema_exact": "G08",
    "G08_M19_heldout_predictions_exact": "G08",
    "G08_M20_heldout_confusion_exact": "G08",
    "G08_M21_heldout_train_only_state_exact": "G08",
    "G08_M22_negative_controls_exact": "G08",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_ids(inputs: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        "parser_case_ids": sorted(row["case_id"] for row in inputs["parser_cases"]),
        "categorical_case_ids": sorted(row["case_id"] for row in inputs["categorical_cases"]),
        "analyzer_case_ids": sorted(row["case_id"] for row in inputs["analyzer_cases"]),
        "nb_case_ids": sorted(row["case_id"] for row in inputs["nb_cases"]),
        "fold_case_ids": sorted(row["case_id"] for row in inputs["fold_cases"]),
        "cv_case_ids": sorted(row["case_id"] for row in inputs["cv_cases"]),
        "heldout_case_ids": sorted(row["case_id"] for row in inputs["heldout_cases"]),
        "negative_case_ids": sorted(row["case_id"] for row in inputs["negative_cases"]),
    }


def _semantic_profile_digests(inputs: Mapping[str, Any], expected: Mapping[str, Any]) -> tuple[str, str]:
    normalized_inputs = deepcopy(inputs)
    for section in (
        "parser_cases", "categorical_cases", "analyzer_cases", "nb_cases",
        "fold_cases", "cv_cases", "heldout_cases", "negative_cases",
    ):
        normalized_inputs[section] = sorted(normalized_inputs[section], key=lambda row: row["case_id"])
    for case in normalized_inputs["parser_cases"]:
        case["source"] = normalize_source(case["source"])
    for case in normalized_inputs["analyzer_cases"]:
        case["text"] = normalize_source(case["text"])
    for section in ("categorical_cases", "nb_cases"):
        for case in normalized_inputs[section]:
            for key in ("fit_rows", "eval_rows"):
                case[key] = sorted(case[key], key=lambda row: row["row_id"])
                for row in case[key]:
                    if "text" in row:
                        row["text"] = normalize_source(row["text"])
    for case in normalized_inputs["fold_cases"]:
        case["rows"] = sorted(case["rows"], key=lambda row: row["row_id"])
    for case in normalized_inputs["cv_cases"]:
        case["rows"] = sorted(case["rows"], key=lambda row: row["row_id"])
        for row in case["rows"]:
            row["text"] = normalize_source(row["text"])
    for case in normalized_inputs["heldout_cases"]:
        for key in ("train_rows", "heldout_rows"):
            case[key] = sorted(case[key], key=lambda row: row["row_id"])
            for row in case[key]:
                row["text"] = normalize_source(row["text"])

    normalized_expected = deepcopy(expected)
    for section in (
        "parser_results", "categorical_results", "analyzer_results", "nb_results",
        "fold_results", "cv_results", "heldout_results", "negative_results",
    ):
        normalized_expected[section] = sorted(normalized_expected[section], key=lambda row: row["case_id"])
    for section in ("categorical_results", "nb_results", "cv_results", "heldout_results"):
        for case in normalized_expected[section]:
            if "predictions" in case:
                case["predictions"] = sorted(case["predictions"], key=lambda row: row["row_id"])
    return sha256_json(normalized_inputs), sha256_json(normalized_expected)


def _validate_input_schema(inputs: Mapping[str, Any]) -> None:
    exact_keys(
        inputs,
        (
            "schema_version", "contract_version", "labels", "parser_cases",
            "categorical_cases", "analyzer_cases", "nb_cases", "fold_cases",
            "cv_cases", "heldout_cases", "negative_cases",
        ),
        code="input_schema",
    )
    if inputs["schema_version"] != INPUT_SCHEMA or inputs["contract_version"] != CONTRACT_VERSION:
        raise ContractError("input_schema", "schema or contract version mismatch")
    require_labels(inputs["labels"])
    for key in (
        "parser_cases", "categorical_cases", "analyzer_cases", "nb_cases",
        "fold_cases", "cv_cases", "heldout_cases", "negative_cases",
    ):
        if not isinstance(inputs[key], list):
            raise ContractError("input_schema", f"{key} must be an array")
    if _case_ids(inputs) != PROFILE:
        raise ContractError("profile_mismatch", "qualification case ids do not match the fixed profile")


def _validate_case_schemas(inputs: Mapping[str, Any]) -> dict[str, bool]:
    flags = {
        "parser": True,
        "categorical": True,
        "analyzer": True,
        "nb": True,
        "fold": True,
        "cv": True,
        "heldout": True,
        "negative": True,
    }
    schemas = {
        "parser": (inputs["parser_cases"], ("case_id", "family", "heuristic", "source", "valid_choice_mask")),
        "categorical": (inputs["categorical_cases"], ("case_id", "fit_rows", "eval_rows")),
        "analyzer": (inputs["analyzer_cases"], ("case_id", "analyzer", "text")),
        "nb": (inputs["nb_cases"], ("case_id", "analyzer", "fit_rows", "eval_rows")),
        "fold": (inputs["fold_cases"], ("case_id", "fold_count", "rows")),
        "cv": (inputs["cv_cases"], ("case_id", "analyzer", "fold_count", "rows")),
        "heldout": (inputs["heldout_cases"], ("case_id", "analyzer", "train_rows", "heldout_rows")),
        "negative": (inputs["negative_cases"], ("case_id", "kind")),
    }
    for family, (rows, keys) in schemas.items():
        try:
            for row in rows:
                exact_keys(row, keys, code=f"{family}_case_schema")
        except ContractError:
            flags[family] = False
    return flags


def _model_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    model = result["model"]
    return {
        "case_id": result["case_id"],
        "model": {
            "analyzer": model["analyzer"],
            "labels": model["labels"],
            "class_counts": model["class_counts"],
            "feature_totals": model["feature_totals"],
            "feature_counts_sha256": model["feature_counts_sha256"],
            "vocabulary": model["vocabulary"],
            "vocabulary_sha256": model["vocabulary_sha256"],
            "fit_row_fingerprint": model["fit_row_fingerprint"],
        },
        "predictions": result["predictions"],
    }


def _run_negative(kind: str, inputs: Mapping[str, Any]) -> str:
    parser = deepcopy(inputs["parser_cases"][0])
    categorical = deepcopy(inputs["categorical_cases"][0])
    nb_case = deepcopy(inputs["nb_cases"][0])
    fold_case = deepcopy(inputs["fold_cases"][0])
    heldout = deepcopy(inputs["heldout_cases"][0])
    try:
        if kind == "parser_unknown_heuristic":
            heuristic_prediction(parser["source"], parser["family"], "not-a-heuristic", parser["valid_choice_mask"])
        elif kind == "parser_all_false_mask":
            heuristic_prediction(parser["source"], parser["family"], parser["heuristic"], [False, False, False])
        elif kind == "categorical_single_class":
            fit_conditional_majority([
                {"row_id": "one", "category": "x", "label": "L0"},
                {"row_id": "two", "category": "y", "label": "L0"},
            ])
        elif kind == "nb_empty_vocabulary":
            fit_multinomial_nb([
                {"row_id": "one", "text": "!!!", "label": "L0"},
                {"row_id": "two", "text": "???", "label": "L1"},
            ], "word")
        elif kind == "nb_single_class":
            fit_multinomial_nb([
                {"row_id": "one", "text": "alpha", "label": "L0"},
                {"row_id": "two", "text": "beta", "label": "L0"},
            ], "word")
        elif kind == "fold_insufficient_groups":
            assign_grouped_folds(fold_case["rows"][:4], fold_count=5)
        elif kind == "fold_duplicate_row_id":
            fold_case["rows"][1]["row_id"] = fold_case["rows"][0]["row_id"]
            assign_grouped_folds(fold_case["rows"], fold_count=5)
        elif kind == "heldout_row_overlap":
            heldout["heldout_rows"][0]["row_id"] = heldout["train_rows"][0]["row_id"]
            evaluate_train_heldout(heldout)
        elif kind == "heldout_group_overlap":
            heldout["heldout_rows"][0]["group_id"] = heldout["train_rows"][0]["group_id"]
            evaluate_train_heldout(heldout)
        elif kind == "nb_unknown_analyzer":
            fit_multinomial_nb(nb_case["fit_rows"], "not-an-analyzer")
        else:
            raise ContractError("unknown_negative_kind", f"unknown negative kind {kind!r}")
    except ContractError as exc:
        return exc.code
    raise ContractError("negative_not_rejected", f"negative case {kind!r} did not raise")


def compute_qualification_results(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _validate_input_schema(inputs)
    schemas = _validate_case_schemas(inputs)
    if not all(schemas.values()):
        raise ContractError("case_schema", f"invalid case schemas: {schemas}")
    labels = tuple(inputs["labels"])

    parser_results = []
    for case in inputs["parser_cases"]:
        result = heuristic_prediction(
            case["source"], case["family"], case["heuristic"], case["valid_choice_mask"],
        )
        parser_results.append({"case_id": case["case_id"], **result})
    parser_results.sort(key=lambda item: item["case_id"])

    categorical_results = [evaluate_categorical_case(case, labels) for case in inputs["categorical_cases"]]
    categorical_results.sort(key=lambda item: item["case_id"])

    analyzer_results = []
    for case in inputs["analyzer_cases"]:
        sequence = word_tokens(case["text"]) if case["analyzer"] == "word" else character_ngrams(case["text"])
        analyzer_results.append(
            {
                "case_id": case["case_id"],
                "analyzer": case["analyzer"],
                "sequence": sequence,
                "counts": {feature: count for feature, count in sorted(Counter(sequence).items())},
            }
        )
    analyzer_results.sort(key=lambda item: item["case_id"])

    nb_results = [_model_projection(evaluate_nb_case(case, labels)) for case in inputs["nb_cases"]]
    nb_results.sort(key=lambda item: item["case_id"])

    fold_results = []
    for case in inputs["fold_cases"]:
        result = assign_grouped_folds(case["rows"], labels, case["fold_count"])
        fold_results.append({"case_id": case["case_id"], **result})
    fold_results.sort(key=lambda item: item["case_id"])

    cv_results = [evaluate_grouped_cv(case, labels) for case in inputs["cv_cases"]]
    cv_results.sort(key=lambda item: item["case_id"])
    heldout_results = [evaluate_train_heldout(case, labels) for case in inputs["heldout_cases"]]
    heldout_results.sort(key=lambda item: item["case_id"])
    negative_results = [
        {"case_id": case["case_id"], "error_code": _run_negative(case["kind"], inputs)}
        for case in inputs["negative_cases"]
    ]
    negative_results.sort(key=lambda item: item["case_id"])

    return {
        "profile": PROFILE,
        "parser_results": parser_results,
        "categorical_results": categorical_results,
        "analyzer_results": analyzer_results,
        "nb_results": nb_results,
        "fold_results": fold_results,
        "cv_results": cv_results,
        "heldout_results": heldout_results,
        "negative_results": negative_results,
    }


def _metric(passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "observed": observed, "expected": expected}


def _profile_equal(inputs: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    try:
        return _case_ids(inputs) == PROFILE and expected["profile"] == PROFILE
    except Exception:
        return False


def _parser_surface_only_boundary(inputs: Mapping[str, Any]) -> dict[str, Any]:
    parameters = list(inspect.signature(heuristic_prediction).parameters)
    source = inputs["parser_cases"][0]["source"] + "\nAnswer: L0"
    error_code = None
    try:
        heuristic_prediction(source, "ERE", "initial_value", [True, True, True])
    except ContractError as exc:
        error_code = exc.code
    return {"parameters": parameters, "unknown_line_error": error_code}


def _manifest_and_seal(bundle: Path) -> tuple[bool, dict[str, Any]]:
    try:
        manifest = _read_json(bundle / "manifest.json")
        seal = _read_json(bundle / "input-seal.json")
        exact_keys(
            manifest,
            ("schema_version", "contract_version", "counts", "qualified_prior", "files"),
            code="manifest_schema",
        )
        exact_keys(seal, ("schema_version", "files"), code="input_seal_schema")
        if manifest["schema_version"] != "yggdrasil.v2-r1r.p0d-v10.r0c-statistical.manifest.v1":
            raise ContractError("manifest_schema", "manifest schema mismatch")
        if manifest["contract_version"] != CONTRACT_VERSION:
            raise ContractError("manifest_schema", "manifest contract mismatch")
        if seal["schema_version"] != "yggdrasil.v2-r1r.p0d-v10.r0c-statistical.input-seal.v1":
            raise ContractError("input_seal_schema", "input seal schema mismatch")
        payload = tree_hashes(bundle, exclude=("manifest.json", "input-seal.json"))
        sealed = tree_hashes(bundle, exclude=("input-seal.json",))
        if manifest["files"] != payload or seal["files"] != sealed:
            raise ContractError("input_seal_mismatch", "manifest/input seal does not match bundle bytes")
        prior = manifest["qualified_prior"]
        expected_prior = {
            "v7_r0a_evidence_seal_sha256": "794CB9518F094EE6CF99800AF0B5848D22A7746DAA5E76A84489F3F8201987F0",
            "v9_r0b_evidence_seal_sha256": "94B80FBD2BB38BC46E95F88E0747D9DB1D8CB279E689F68BB4A43AE7B76D8209",
        }
        if prior != expected_prior:
            raise ContractError("prior_evidence_mismatch", "qualified prior evidence hashes mismatch")
        return True, {"sealed_file_count": len(sealed), "qualified_prior": prior}
    except Exception as exc:
        code = exc.code if isinstance(exc, ContractError) else type(exc).__name__
        return False, {"error": code}


def _section(results: Mapping[str, Any], key: str) -> Any:
    return results.get(key, "unavailable")


def audit_statistical_bundle(bundle: str | Path) -> dict[str, Any]:
    bundle = Path(bundle)
    before = tree_hashes(bundle) if bundle.is_dir() else {}
    metrics = {metric_id: _metric(False, "unavailable", True) for metric_id in METRIC_GATES}
    failures: list[str] = []
    profile_digest_observed = {"inputs": None, "expected": None}
    profile_digest_ok = False

    expected_top = {
        "manifest.json", "qualification-inputs.json", "expected-results.json",
        "input-seal.json", "source_snapshot",
    }
    top_exact = bundle.is_dir() and {path.name for path in bundle.iterdir()} == expected_top
    metrics["G07_M01_bundle_schema_exact"] = _metric(top_exact, top_exact, True)
    seal_ok, seal_observed = _manifest_and_seal(bundle) if bundle.is_dir() else (False, {"error": "missing_bundle"})
    metrics["G07_M02_manifest_input_seal_exact"] = _metric(seal_ok, seal_observed, True)

    try:
        inputs = _read_json(bundle / "qualification-inputs.json")
        expected = _read_json(bundle / "expected-results.json")
        _validate_input_schema(inputs)
        exact_keys(
            expected,
            (
                "schema_version", "contract_version", "profile", "parser_results",
                "categorical_results", "analyzer_results", "nb_results", "fold_results",
                "cv_results", "heldout_results", "negative_results",
            ),
            code="expected_schema",
        )
        if expected["schema_version"] != EXPECTED_SCHEMA or expected["contract_version"] != CONTRACT_VERSION:
            raise ContractError("expected_schema", "expected schema or contract mismatch")
        input_digest, expected_digest = _semantic_profile_digests(inputs, expected)
        profile_digest_observed = {"inputs": input_digest, "expected": expected_digest}
        profile_digest_ok = input_digest == INPUT_SEMANTIC_SHA256 and expected_digest == EXPECTED_SEMANTIC_SHA256
        for section_name in (
            "parser_results", "categorical_results", "analyzer_results", "nb_results",
            "fold_results", "cv_results", "heldout_results", "negative_results",
        ):
            expected[section_name] = sorted(expected[section_name], key=lambda row: row["case_id"])
        schema_flags = _validate_case_schemas(inputs)
        observed = compute_qualification_results(inputs)
        profile_ok = _profile_equal(inputs, expected)
        metrics["G07_M03_profile_ids_counts_exact"] = _metric(profile_ok, observed["profile"], expected["profile"])
        metrics["G07_M04_parser_case_schema_exact"] = _metric(schema_flags["parser"], schema_flags["parser"], True)
        parser_boundary = _parser_surface_only_boundary(inputs)
        expected_parser_boundary = {
            "parameters": ["source", "family", "heuristic", "valid_choice_mask"],
            "unknown_line_error": "malformed_source",
        }
        metrics["G07_M08_parser_surface_only_boundary"] = _metric(
            parser_boundary == expected_parser_boundary,
            parser_boundary,
            expected_parser_boundary,
        )
        metrics["G07_M09_categorical_case_schema_exact"] = _metric(schema_flags["categorical"], schema_flags["categorical"], True)
        metrics["G08_M01_analyzer_case_schema_exact"] = _metric(schema_flags["analyzer"], schema_flags["analyzer"], True)
        metrics["G08_M04_nb_case_schema_exact"] = _metric(schema_flags["nb"], schema_flags["nb"], True)
        metrics["G08_M10_fold_case_schema_exact"] = _metric(schema_flags["fold"], schema_flags["fold"], True)
        metrics["G08_M14_cv_case_schema_exact"] = _metric(schema_flags["cv"], schema_flags["cv"], True)
        metrics["G08_M18_heldout_case_schema_exact"] = _metric(schema_flags["heldout"], schema_flags["heldout"], True)

        parser_observed = observed["parser_results"]
        parser_expected = expected["parser_results"]
        parser_by_id = {row["case_id"]: row for row in parser_observed}
        expected_parser_by_id = {row["case_id"]: row for row in parser_expected}
        questions = {key: row["question"] for key, row in parser_by_id.items()}
        expected_questions = {key: row["question"] for key, row in expected_parser_by_id.items()}
        counts = {key: row["counts"] for key, row in parser_by_id.items()}
        expected_counts = {key: row["counts"] for key, row in expected_parser_by_id.items()}
        predictions = {
            key: {"raw_prediction": row["raw_prediction"], "prediction": row["prediction"]}
            for key, row in parser_by_id.items()
        }
        expected_predictions = {
            key: {"raw_prediction": row["raw_prediction"], "prediction": row["prediction"]}
            for key, row in expected_parser_by_id.items()
        }
        metrics["G07_M05_question_view_exact"] = _metric(questions == expected_questions, questions, expected_questions)
        metrics["G07_M06_surface_counts_exact"] = _metric(counts == expected_counts, counts, expected_counts)
        metrics["G07_M07_heuristic_predictions_exact"] = _metric(
            predictions == expected_predictions, predictions, expected_predictions,
        )

        categorical_observed = observed["categorical_results"]
        categorical_expected = expected["categorical_results"]
        observed_counts = [{"case_id": row["case_id"], "model": row["model"]} for row in categorical_observed]
        expected_counts_rows = [{"case_id": row["case_id"], "model": row["model"]} for row in categorical_expected]
        observed_predictions = [row["predictions"] for row in categorical_observed]
        expected_categorical_predictions = [row["predictions"] for row in categorical_expected]
        metrics["G07_M10_categorical_counts_exact"] = _metric(
            observed_counts == expected_counts_rows, observed_counts, expected_counts_rows,
        )
        metrics["G07_M11_categorical_predictions_exact"] = _metric(
            observed_predictions == expected_categorical_predictions,
            observed_predictions,
            expected_categorical_predictions,
        )
        edge_ids = {"ce2_tie", "ce3_unseen", "ce4_mask"}
        observed_edges = [row for result in categorical_observed for row in result["predictions"] if row["row_id"] in edge_ids]
        expected_edges = [row for result in categorical_expected for row in result["predictions"] if row["row_id"] in edge_ids]
        metrics["G07_M12_categorical_tie_mask_unseen_exact"] = _metric(
            observed_edges == expected_edges, observed_edges, expected_edges,
        )

        negative_observed = observed["negative_results"]
        negative_expected = expected["negative_results"]
        metrics["G07_M13_parser_negative_controls_exact"] = _metric(
            negative_observed[:2] == negative_expected[:2], negative_observed[:2], negative_expected[:2],
        )

        analyzer_observed = observed["analyzer_results"]
        analyzer_expected = expected["analyzer_results"]
        observed_word = [row for row in analyzer_observed if row["analyzer"] == "word"]
        expected_word = [row for row in analyzer_expected if row["analyzer"] == "word"]
        observed_char = [row for row in analyzer_observed if row["analyzer"] == "char_3_5"]
        expected_char = [row for row in analyzer_expected if row["analyzer"] == "char_3_5"]
        metrics["G08_M02_word_tokens_exact"] = _metric(observed_word == expected_word, observed_word, expected_word)
        metrics["G08_M03_character_ngrams_exact"] = _metric(observed_char == expected_char, observed_char, expected_char)

        nb_observed = observed["nb_results"]
        nb_expected = expected["nb_results"]
        vocab_observed = [row["model"]["vocabulary"] for row in nb_observed]
        vocab_expected = [row["model"]["vocabulary"] for row in nb_expected]
        vocab_hash_observed = [row["model"]["vocabulary_sha256"] for row in nb_observed]
        vocab_hash_expected = [row["model"]["vocabulary_sha256"] for row in nb_expected]
        count_fields = ("class_counts", "feature_totals", "feature_counts_sha256", "fit_row_fingerprint")
        counts_observed = [{field: row["model"][field] for field in count_fields} for row in nb_observed]
        counts_expected = [{field: row["model"][field] for field in count_fields} for row in nb_expected]
        predictions_observed = [row["predictions"] for row in nb_observed]
        predictions_expected = [row["predictions"] for row in nb_expected]
        metrics["G08_M05_vocabulary_order_exact"] = _metric(vocab_observed == vocab_expected, vocab_observed, vocab_expected)
        metrics["G08_M06_vocabulary_hash_exact"] = _metric(
            vocab_hash_observed == vocab_hash_expected, vocab_hash_observed, vocab_hash_expected,
        )
        metrics["G08_M07_nb_class_feature_counts_exact"] = _metric(
            counts_observed == counts_expected, counts_observed, counts_expected,
        )
        metrics["G08_M08_nb_exact_scores_predictions"] = _metric(
            predictions_observed == predictions_expected, predictions_observed, predictions_expected,
        )
        nb_edge_ids = {"we3_oov", "we4_mask", "che4_oov"}
        nb_edges_observed = [row for result in nb_observed for row in result["predictions"] if row["row_id"] in nb_edge_ids]
        nb_edges_expected = [row for result in nb_expected for row in result["predictions"] if row["row_id"] in nb_edge_ids]
        metrics["G08_M09_nb_oov_mask_tie_exact"] = _metric(
            nb_edges_observed == nb_edges_expected, nb_edges_observed, nb_edges_expected,
        )

        fold_observed = observed["fold_results"]
        fold_expected = expected["fold_results"]
        observed_maps = [row["group_to_fold"] for row in fold_observed]
        expected_maps = [row["group_to_fold"] for row in fold_expected]
        observed_balance = [row["balance"] for row in fold_observed]
        expected_balance = [row["balance"] for row in fold_expected]
        metrics["G08_M11_grouped_fold_map_exact"] = _metric(observed_maps == expected_maps, observed_maps, expected_maps)
        metrics["G08_M13_fold_balance_exact"] = _metric(
            observed_balance == expected_balance, observed_balance, expected_balance,
        )

        cv_observed = observed["cv_results"]
        cv_expected = expected["cv_results"]
        cv_predictions_observed = [row["predictions"] for row in cv_observed]
        cv_predictions_expected = [row["predictions"] for row in cv_expected]
        cv_confusion_observed = [row["confusion"] for row in cv_observed]
        cv_confusion_expected = [row["confusion"] for row in cv_expected]
        cv_states_observed = [
            {"group_to_fold": row["group_to_fold"], "balance": row["balance"], "fold_states": row["fold_states"]}
            for row in cv_observed
        ]
        cv_states_expected = [
            {"group_to_fold": row["group_to_fold"], "balance": row["balance"], "fold_states": row["fold_states"]}
            for row in cv_expected
        ]
        metrics["G08_M15_cv_predictions_exact"] = _metric(
            cv_predictions_observed == cv_predictions_expected, cv_predictions_observed, cv_predictions_expected,
        )
        metrics["G08_M16_cv_confusion_exact"] = _metric(
            cv_confusion_observed == cv_confusion_expected, cv_confusion_observed, cv_confusion_expected,
        )
        metrics["G08_M17_cv_fold_state_exact"] = _metric(
            cv_states_observed == cv_states_expected, cv_states_observed, cv_states_expected,
        )

        heldout_observed = observed["heldout_results"]
        heldout_expected = expected["heldout_results"]
        heldout_predictions_observed = [row["predictions"] for row in heldout_observed]
        heldout_predictions_expected = [row["predictions"] for row in heldout_expected]
        heldout_confusion_observed = [row["confusion"] for row in heldout_observed]
        heldout_confusion_expected = [row["confusion"] for row in heldout_expected]
        heldout_state_observed = [row["train_state"] for row in heldout_observed]
        heldout_state_expected = [row["train_state"] for row in heldout_expected]
        metrics["G08_M19_heldout_predictions_exact"] = _metric(
            heldout_predictions_observed == heldout_predictions_expected,
            heldout_predictions_observed,
            heldout_predictions_expected,
        )
        metrics["G08_M20_heldout_confusion_exact"] = _metric(
            heldout_confusion_observed == heldout_confusion_expected,
            heldout_confusion_observed,
            heldout_confusion_expected,
        )
        metrics["G08_M21_heldout_train_only_state_exact"] = _metric(
            heldout_state_observed == heldout_state_expected, heldout_state_observed, heldout_state_expected,
        )
        metrics["G08_M22_negative_controls_exact"] = _metric(
            negative_observed == negative_expected, negative_observed, negative_expected,
        )
    except Exception as exc:
        code = exc.code if isinstance(exc, ContractError) else type(exc).__name__
        failures.append(code)

    after = tree_hashes(bundle) if bundle.is_dir() else {}
    read_only = before == after
    if not read_only:
        failures.append("input_modified")
    if not profile_digest_ok:
        failures.append("semantic_profile_digest_mismatch")
    gates = {
        gate: all(row["passed"] for metric_id, row in metrics.items() if METRIC_GATES[metric_id] == gate)
        for gate in ("G07", "G08")
    }
    failures.extend(metric_id for metric_id, row in metrics.items() if row["passed"] is not True)
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v10.r0c-statistical.report.v1",
        "gates": gates,
        "metrics": metrics,
        "semantic_profile": {
            "observed": profile_digest_observed,
            "expected": {"inputs": INPUT_SEMANTIC_SHA256, "expected": EXPECTED_SEMANTIC_SHA256},
            "passed": profile_digest_ok,
        },
        "failures": sorted(set(failures)),
        "passed": all(gates.values()) and read_only and profile_digest_ok,
    }
