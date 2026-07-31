from __future__ import annotations

"""P0-D replay, leakage, overlap, and interface audits."""

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from .cps import CPS_SPLITS, mutate_cps_ast
from .ere import ERE_SPLITS, mutate_ere_ast
from .render import OOD_CPS_TEMPLATES, OOD_ERE_TEMPLATES, TRAIN_CPS_TEMPLATES, TRAIN_ERE_TEMPLATES, contract_token_count, semantic_fingerprint, surface_fingerprint
from .schema import MAX_SOURCE_TOKENS, SCHEMA_VERSION, assert_model_view, make_model_view, validate_record_shape
from .simulator import select_cps_answer, simulate_ere


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_dataset(root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    data_root = root / "data" if (root / "data").is_dir() else root
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family, splits in (("ere", ERE_SPLITS), ("cps", CPS_SPLITS)):
        result[family] = {}
        for split in splits:
            path = data_root / family / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            result[family][split] = load_jsonl(path)
    return result


def _answer_option(record: Mapping[str, Any]) -> str:
    return str(record["audit_answer_semantic"])


def _replay_record(record: Mapping[str, Any]) -> dict[str, Any]:
    family = record["family"]
    ast = record["program_ast"]
    if family == "ere":
        simulation = simulate_ere(ast)
        option = simulation["answer"]
        trace_ok = simulation["trace"] == record["teacher_trace"]
        selected_ok = option == _answer_option(record)
        unique_ok = True
    elif family == "cps":
        selected = select_cps_answer(ast)
        option = "NONE" if selected["answer"] is None else f"candidate_{selected['answer']}"
        trace_ok = [result["trace"] for result in selected["results"]] == record["teacher_trace"]
        selected_ok = option == _answer_option(record)
        unique_ok = bool(selected["unique_optimum"])
    else:
        raise ValueError(f"unknown family: {family}")
    mapping = record["audit_label_mapping"]
    label_ok = mapping.get(option) is not None and record["answer_index"] == ord(mapping[option]) - ord("A")
    return {
        "simulator_answer": option,
        "record_answer": _answer_option(record),
        "answer_ok": selected_ok and label_ok,
        "teacher_trace_ok": trace_ok,
        "unique_optimum_ok": unique_ok,
        "replay_ok": selected_ok and label_ok and trace_ok and unique_ok,
    }


def _deep_diff_paths(left: Any, right: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        paths: list[tuple[Any, ...]] = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                paths.append(path + (key,))
            else:
                paths.extend(_deep_diff_paths(left[key], right[key], path + (key,)))
        return paths
    if isinstance(left, list):
        paths = []
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                paths.append(path + (index,))
            else:
                paths.extend(_deep_diff_paths(left[index], right[index], path + (index,)))
        return paths
    return [] if left == right else [path]


def _pair_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("pair_id"):
            groups[str(record["pair_id"])].append(record)
    pair_reports: list[dict[str, Any]] = []
    all_good = True
    for pair_id, pair in sorted(groups.items()):
        roles = {item.get("pair_role") for item in pair}
        same_mapping = len({json.dumps(item["audit_label_mapping"], sort_keys=True) for item in pair}) == 1
        same_choices = len({json.dumps(item["valid_choice_mask"]) for item in pair}) == 1
        diff_paths = _deep_diff_paths(pair[0]["program_ast"], pair[1]["program_ast"]) if len(pair) == 2 else []
        answer_flip = len({item["audit_answer_semantic"] for item in pair}) == 2 and len({item["answer_index"] for item in pair}) == 2
        certificate_paths = [tuple(item["causal_certificate"]["changed_path"]) for item in pair]
        one_change = len(diff_paths) == 1
        good = len(pair) == 2 and roles == {"base", "flip"} and same_mapping and same_choices and one_change and answer_flip and len(set(certificate_paths)) == 1
        all_good = all_good and good
        pair_reports.append({"pair_id": pair_id, "count": len(pair), "roles": sorted(str(role) for role in roles), "diff_paths": [list(path) for path in diff_paths], "answer_flip": answer_flip, "same_mapping": same_mapping, "same_choice_mask": same_choices, "passed": good})
    return {"pair_count": len(groups), "pairs": pair_reports, "passed": all_good and bool(groups)}


def _overlap_gate(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    semantic: dict[str, dict[str, set[str]]] = {}
    surface: dict[str, dict[str, set[str]]] = {}
    undeclared_semantic: list[dict[str, Any]] = []
    undeclared_surface: list[dict[str, Any]] = []
    for family, split_rows in dataset.items():
        semantic[family] = {split: {str(row["semantic_fingerprint"]) for row in rows} for split, rows in split_rows.items()}
        surface[family] = {split: {str(row["surface_fingerprint"]) for row in rows} for split, rows in split_rows.items()}
        splits = sorted(split_rows)
        for index, left in enumerate(splits):
            for right in splits[index + 1 :]:
                semantic_overlap = semantic[family][left] & semantic[family][right]
                surface_overlap = surface[family][left] & surface[family][right]
                for fingerprint in semantic_overlap:
                    owners = [row for row in split_rows[left] + split_rows[right] if row["semantic_fingerprint"] == fingerprint]
                    if not all(row.get("pair_id") and row["split"] == "causal_pairs" for row in owners):
                        undeclared_semantic.append({"family": family, "left": left, "right": right, "fingerprint": fingerprint})
                for fingerprint in surface_overlap:
                    owners = [row for row in split_rows[left] + split_rows[right] if row["surface_fingerprint"] == fingerprint]
                    if not all(row.get("pair_id") and row["split"] == "causal_pairs" for row in owners):
                        undeclared_surface.append({"family": family, "left": left, "right": right, "fingerprint": fingerprint})
    return {
        "semantic_overlap": {family: {split: len(values) for split, values in splits.items()} for family, splits in semantic.items()},
        "surface_overlap": {family: {split: len(values) for split, values in splits.items()} for family, splits in surface.items()},
        "undeclared_semantic_overlap": undeclared_semantic,
        "undeclared_surface_overlap": undeclared_surface,
        "passed": not undeclared_semantic and not undeclared_surface,
    }


def _balance_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"examples": 0, "answer_counts": {}, "answer_max_deviation": 1.0, "choice_counts": {}, "choice_max_deviation": 1.0}
    answer_counts = Counter(int(record["answer_index"]) for record in records)
    choice_counts = Counter(index for record in records for index, active in enumerate(record["valid_choice_mask"]) if active)
    expected_answer = len(records) / 9.0
    expected_choice = sum(sum(record["valid_choice_mask"]) for record in records) / 9.0
    return {
        "examples": len(records),
        "answer_counts": {chr(index + 65): answer_counts[index] for index in range(9)},
        "answer_max_deviation": max(abs(answer_counts[index] - expected_answer) / max(1.0, expected_answer) for index in range(9)),
        "choice_counts": {chr(index + 65): choice_counts[index] for index in range(9)},
        "choice_max_deviation": max(abs(choice_counts[index] - expected_choice) / max(1.0, expected_choice) for index in range(9)),
    }


def _group_halves(records: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record.get("pair_id") or record.get("example_id") or index)].append(record)
    train: list[Mapping[str, Any]] = []
    test: list[Mapping[str, Any]] = []
    for index, (_group, rows) in enumerate(sorted(groups.items())):
        (train if index % 2 == 0 else test).extend(rows)
    if not train or not test:
        midpoint = max(1, len(records) // 2)
        return list(records[:midpoint]), list(records[midpoint:] or records[:midpoint])
    return train, test


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def unigram_naive_bayes_accuracy(
    records: Sequence[Mapping[str, Any]],
    *,
    text_getter: Callable[[Mapping[str, Any]], str] | None = None,
) -> float:
    """Custom Laplace-smoothed unigram NB used by the P0 heuristic gate."""

    if not records:
        return 0.0
    text_getter = text_getter or (lambda record: str(record["source_text"]))
    train, test = _group_halves(records)
    labels = sorted({int(record["answer_index"]) for record in train})
    if not labels:
        return 0.0
    vocabulary = sorted({token for record in train for token in _tokens(text_getter(record))})
    token_counts: dict[int, Counter[str]] = {label: Counter() for label in labels}
    totals: Counter[int] = Counter()
    priors = Counter(int(record["answer_index"]) for record in train)
    for record in train:
        label = int(record["answer_index"])
        counts = Counter(_tokens(text_getter(record)))
        token_counts[label].update(counts)
        totals[label] += sum(counts.values())
    correct = 0
    for record in test:
        tokens = Counter(_tokens(text_getter(record)))
        scores: dict[int, float] = {}
        for label in labels:
            score = math.log((priors[label] + 1) / (len(train) + len(labels)))
            denominator = totals[label] + len(vocabulary) + 1
            for token, count in tokens.items():
                score += count * math.log((token_counts[label][token] + 1) / denominator)
            scores[label] = score
        prediction = max(scores, key=scores.get)
        correct += int(prediction == int(record["answer_index"]))
    return correct / max(1, len(test))


def _majority_accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        return 0.0
    train, test = _group_halves(records)
    majority = Counter(int(row["answer_index"]) for row in train).most_common(1)[0][0]
    return sum(int(int(row["answer_index"]) == majority) for row in test) / max(1, len(test))


def _length_bucket_accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    train, test = _group_halves(records)
    buckets: dict[int, Counter[int]] = defaultdict(Counter)
    for row in train:
        buckets[int(row.get("token_count", contract_token_count(str(row["source_text"]))) // 16)][int(row["answer_index"])] += 1
    correct = 0
    for row in test:
        counts = buckets.get(int(row.get("token_count", contract_token_count(str(row["source_text"]))) // 16))
        prediction = counts.most_common(1)[0][0] if counts else Counter(int(item["answer_index"]) for item in train).most_common(1)[0][0]
        correct += int(prediction == int(row["answer_index"]))
    return correct / max(1, len(test))


def _label_position_accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    predictions: list[int] = []
    for row in records:
        active = [index for index, value in enumerate(row["valid_choice_mask"]) if value]
        predictions.append(active[0] if active else 0)
    return sum(prediction == int(row["answer_index"]) for prediction, row in zip(predictions, records)) / max(1, len(records))


def _question_text(source: str) -> str:
    for marker in ("Question:", "The point to determine is", "The requested result"):
        if marker in source:
            return source.split(marker, 1)[1]
    return source


def _claim_text(row: Mapping[str, Any]) -> str:
    return " ".join(str(claim.get("text", "")) for claim in row.get("training_claims", []))


def evaluate_heuristics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"examples": 0, "chance": 0.0, "metrics": {}}
    chance = sum(1.0 / max(1, sum(row["valid_choice_mask"])) for row in records) / len(records)
    metrics = {
        "majority": _majority_accuracy(records),
        "label_position": _label_position_accuracy(records),
        "length_bucket": _length_bucket_accuracy(records),
        "last_mention_unigram_nb": unigram_naive_bayes_accuracy(records, text_getter=lambda row: " ".join(_tokens(str(row["source_text"]))[-4:])),
        "question_only_unigram_nb": unigram_naive_bayes_accuracy(records, text_getter=lambda row: _question_text(str(row["source_text"]))),
        "full_text_unigram_nb": unigram_naive_bayes_accuracy(records),
        "claim_only_unigram_nb": unigram_naive_bayes_accuracy(records, text_getter=_claim_text),
        "claim_length": _length_bucket_accuracy([dict(row, token_count=len(row.get("training_claims", []))) for row in records]),
    }
    return {"examples": len(records), "chance": chance, "metrics": metrics, "max_allowed": chance + 0.10}


def _necessity_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    necessary = 0
    certificates_valid = 0
    depth_values: list[int] = []
    for record in records:
        certificate = record["causal_certificate"]
        depth_values.append(int(certificate.get("dependency_depth", 0)))
        before = str(certificate["before_answer"])
        after = str(certificate["after_answer"])
        if before != after:
            necessary += 1
        if record["family"] == "ere":
            mutated = mutate_ere_ast(record["program_ast"])
            replay = simulate_ere(mutated)["answer"]
            valid = replay == after
        else:
            mutated = mutate_cps_ast(record["program_ast"])
            selected = select_cps_answer(mutated)
            replay = "NONE" if selected["answer"] is None else f"candidate_{selected['answer']}"
            valid = replay == after
        certificates_valid += int(valid and before == _answer_option(record))
    return {
        "examples": len(records),
        "necessary_rate": necessary / max(1, len(records)),
        "certificate_rate": certificates_valid / max(1, len(records)),
        "min_dependency_depth": min(depth_values) if depth_values else 0,
        "mean_dependency_depth": sum(depth_values) / max(1, len(depth_values)),
    }


def _claim_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(bool(claim.get("label")) for record in records for claim in record.get("training_claims", []))
    total = sum(counts.values())
    return {"positive": counts[True], "negative": counts[False], "total": total, "balanced": counts[True] == counts[False] and total > 0}


def _template_report(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    train_ids = set()
    language_ids = set()
    for family, split_rows in dataset.items():
        for split, rows in split_rows.items():
            target = language_ids if split == "language_ood" else train_ids
            target.update(str(row.get("template_id")) for row in rows)
    return {"train_templates": sorted(train_ids), "language_ood_templates": sorted(language_ids), "intersection": sorted(train_ids & language_ids), "passed": not train_ids.intersection(language_ids)}


def _task_specific_report(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    forbidden = []
    for family, split_rows in dataset.items():
        for split, rows in split_rows.items():
            for row in rows:
                ast = row["program_ast"]
                if family == "ere" and any(key in ast for key in ("slot_assignment", "task_transition", "oracle_rule_index")):
                    forbidden.append(row["example_id"])
                if family == "cps" and any(key in ast for key in ("candidate_validity", "correct_plan_index", "oracle_candidate")):
                    forbidden.append(row["example_id"])
    return {"forbidden_ast_metadata": forbidden, "passed": not forbidden}


def _split_structure_report(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    passed = True
    for family, split_rows in dataset.items():
        family_report: dict[str, Any] = {}
        for split, rows in split_rows.items():
            if family == "ere":
                depths = [int(row["program_ast"]["dependency"]["depth"]) for row in rows]
                entities = [len(row["program_ast"]["entities"]) for row in rows]
                requirement = min(depths, default=0) >= (8 if split == "length_ood" else 3)
                if split == "entity_ood":
                    requirement = requirement and all(entity_count in {6, 8} for entity_count in entities)
                if split == "composition_ood":
                    features = {
                        feature
                        for row in rows
                        for feature in row["program_ast"].get("composition_features", [])
                    }
                    required_features = {
                        "relation_mutation_foreach",
                        "swap_before_condition",
                        "relation_and_attribute_dependency",
                    }
                    requirement = requirement and required_features <= features
                family_report[split] = {
                    "min_dependency_depth": min(depths, default=0),
                    "entity_counts": sorted(set(entities)),
                    "composition_features": sorted({feature for row in rows for feature in row["program_ast"].get("composition_features", [])}),
                    "passed": requirement,
                }
            else:
                candidate_counts = [len(row["program_ast"]["candidates"]) for row in rows]
                valid_suboptimal = [
                    bool(row["causal_certificate"].get("valid_suboptimal_present"))
                    for row in rows
                    if row["audit_answer_semantic"] != "NONE" and row["program_ast"].get("causal_target", {}).get("kind") != "resource"
                ]
                requirement = min(candidate_counts, default=0) >= (8 if split == "distractor_ood" else 5)
                if split == "horizon_ood":
                    requirement = requirement and min((max(len(candidate["plan"]) for candidate in row["program_ast"]["candidates"]) for row in rows), default=0) >= 6
                if split == "composition_ood":
                    requirement = requirement and all(set((row["program_ast"].get("composition_features") or [])) >= {"unlock_resource", "final_constraint", "cost_budget"} for row in rows)
                requirement = requirement and (sum(valid_suboptimal) / max(1, len(valid_suboptimal)) >= 0.95)
                family_report[split] = {"candidate_counts": sorted(set(candidate_counts)), "valid_suboptimal_rate": sum(valid_suboptimal) / max(1, len(valid_suboptimal)), "passed": requirement}
            passed = passed and requirement
        reports[family] = family_report
    return {"families": reports, "passed": passed}


def _scale_expectations(mode: str) -> dict[str, int]:
    if mode == "formal":
        return {"train": 4096, "validation": 512, "composition_ood": 512, "length_ood": 512, "entity_ood": 512, "language_ood": 512, "horizon_ood": 512, "distractor_ood": 512, "causal_pairs": 512}
    return {"train": 64, "validation": 32, "composition_ood": 32, "length_ood": 32, "entity_ood": 32, "language_ood": 32, "horizon_ood": 32, "distractor_ood": 32, "causal_pairs": 32}


def _scale_report(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], mode: str) -> dict[str, Any]:
    expected = _scale_expectations(mode)
    observed: dict[str, dict[str, int]] = {}
    passed = True
    for family, split_rows in dataset.items():
        observed[family] = {}
        for split, rows in split_rows.items():
            wanted = expected[split]
            observed[family][split] = len(rows)
            passed = passed and len(rows) == wanted
    return {"mode": mode, "expected_per_split": expected, "observed": observed, "passed": passed}


def _model_view_report(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    checked = 0
    failures: list[str] = []
    for split_rows in dataset.values():
        for rows in split_rows.values():
            for record in rows:
                checked += 1
                try:
                    view = make_model_view(record)
                    assert_model_view(view)
                    if set(view) != {"reasoning_budget", "valid_choice_mask"}:
                        raise ValueError("P0 model view without cache must contain only public budget and mask")
                except Exception as exc:  # pragma: no cover - report must preserve the concrete row
                    failures.append(f"{record.get('example_id')}: {exc}")
    return {"checked": checked, "failures": failures, "passed": not failures}


def _record_provenance_report(
    dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    manifest: Mapping[str, Any],
    design_doc_sha256: str,
) -> dict[str, Any]:
    """Check the per-record provenance required by the P0 artifact contract."""

    failures: list[str] = []
    expected_generator = str(manifest.get("generator_version", ""))
    for split_rows in dataset.values():
        for rows in split_rows.values():
            for record in rows:
                provenance = record.get("provenance")
                if not isinstance(provenance, Mapping):
                    failures.append(f"{record.get('example_id')}: provenance is not a mapping")
                    continue
                required = {
                    "generator_version",
                    "seed",
                    "input_files_sha256",
                    "design_doc_sha256",
                    "environment",
                    "command",
                    "started_at",
                    "finished_at",
                }
                missing = required - set(provenance)
                if missing:
                    failures.append(f"{record.get('example_id')}: missing provenance {sorted(missing)}")
                    continue
                input_hashes = provenance["input_files_sha256"]
                good = (
                    record.get("generator_version") == expected_generator == provenance["generator_version"]
                    and isinstance(provenance["seed"], int)
                    and provenance["design_doc_sha256"] == design_doc_sha256
                    and isinstance(input_hashes, Mapping)
                    and input_hashes.get("design_doc") == design_doc_sha256
                    and isinstance(provenance["environment"], Mapping)
                    and bool(str(provenance["command"]).strip())
                    and bool(str(provenance["started_at"]).strip())
                    and bool(str(provenance["finished_at"]).strip())
                )
                if not good:
                    failures.append(f"{record.get('example_id')}: provenance values are inconsistent")
    return {"checked": sum(len(rows) for split_rows in dataset.values() for rows in split_rows.values()), "failures": failures, "passed": not failures}


def _replay_report(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family, split_rows in dataset.items():
        for split, records in split_rows.items():
            for record in records:
                result = _replay_record(record)
                rows.append({"family": family, "split": split, "example_id": record["example_id"], **result})
    return {"examples": len(rows), "failed": [row for row in rows if not row["replay_ok"]], "passed": all(row["replay_ok"] for row in rows)}


def _necessity_all_report(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    family_reports = {family: {split: _necessity_report(rows) for split, rows in split_rows.items()} for family, split_rows in dataset.items()}
    rates = [report["necessary_rate"] for family in family_reports.values() for report in family.values()]
    certs = [report["certificate_rate"] for family in family_reports.values() for report in family.values()]
    return {"families": family_reports, "min_necessary_rate": min(rates, default=0.0), "min_certificate_rate": min(certs, default=0.0), "passed": min(rates, default=0.0) >= 0.95 and min(certs, default=0.0) == 1.0}


def _label_gate(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], mode: str) -> dict[str, Any]:
    # Smoke uses a deliberately non-formal diagnostic margin because 32/64
    # observations cannot resolve the formal 3%/7% frequency contract.  The
    # formal branch below is the only result eligible for P0-D acceptance.
    threshold = 0.03 if mode == "formal" else 0.25
    split_reports: dict[str, Any] = {}
    passed = True
    for family, split_rows in dataset.items():
        split_reports[family] = {}
        for split, rows in split_rows.items():
            report = _balance_report(rows)
            limit = threshold if split == "train" else (0.07 if mode == "formal" else 0.50)
            report["max_allowed_deviation"] = limit
            report["passed"] = report["answer_max_deviation"] <= limit and report["choice_max_deviation"] <= limit
            passed = passed and report["passed"]
            split_reports[family][split] = report
    return {"families": split_reports, "passed": passed}


def _heuristic_gate(dataset: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], mode: str) -> tuple[dict[str, Any], bool]:
    reports: dict[str, Any] = {}
    passed = True
    for family, split_rows in dataset.items():
        reports[family] = {}
        for split, rows in split_rows.items():
            report = evaluate_heuristics(rows)
            if mode != "formal":
                report["formal_max_allowed"] = report["max_allowed"]
                report["max_allowed"] = report["chance"] + 0.40
            report["passed"] = all(value <= report["max_allowed"] for value in report["metrics"].values())
            passed = passed and report["passed"]
            reports[family][split] = report
    return reports, passed


def audit_p0(root: Path, *, design_doc: Path, command: str | None = None) -> dict[str, Any]:
    started = utc_now()
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = load_dataset(root)
    for split_rows in dataset.values():
        for rows in split_rows.values():
            for record in rows:
                validate_record_shape(record)
    current_design_hash = sha256_file(design_doc)
    manifest_design_hash = manifest.get("design_doc_sha256")
    replay = _replay_report(dataset)
    overlap = _overlap_gate(dataset)
    pair = _pair_gate([row for rows in dataset["ere"].values() for row in rows] + [row for rows in dataset["cps"].values() for row in rows])
    necessity = _necessity_all_report(dataset)
    structure = _split_structure_report(dataset)
    templates = _template_report(dataset)
    model_view = _model_view_report(dataset)
    provenance = _record_provenance_report(dataset, manifest=manifest, design_doc_sha256=current_design_hash)
    mode = str(manifest.get("mode", "formal"))
    heuristics, heuristics_passed = _heuristic_gate(dataset, mode)
    labels = _label_gate(dataset, mode)
    claims = {family: {split: _claim_report(rows) for split, rows in split_rows.items()} for family, split_rows in dataset.items()}
    claims_passed = all(report["balanced"] for family in claims.values() for report in family.values())
    token_passed = all(
        int(record.get("token_count", -1)) == contract_token_count(record["source_text"]) and int(record.get("token_count", 0)) <= MAX_SOURCE_TOKENS
        for split_rows in dataset.values()
        for rows in split_rows.values()
        for record in rows
    )
    scale = _scale_report(dataset, str(manifest.get("mode", "formal")))
    source_hashes_match = True
    file_hash_report: dict[str, str] = {}
    for relative_path, expected_hash in manifest.get("input_files", {}).items():
        path = root / relative_path
        observed = sha256_file(path) if path.exists() else "missing"
        file_hash_report[relative_path] = observed
        source_hashes_match = source_hashes_match and observed == expected_hash
    gate_results = {
        "simulator_teacher_replay": replay["passed"],
        "undeclared_overlap_zero": overlap["passed"],
        "label_choice_balance": labels["passed"],
        "causal_necessity_ge_95": necessity["passed"],
        "causal_pair_single_change_and_flip": pair["passed"],
        "task_structure_coverage": structure["passed"],
        "heuristics_at_chance": heuristics_passed,
        "claim_balance_and_limit": claims_passed and all(
            report["metrics"].get("claim_only_unigram_nb", 1.0) <= 0.60
            and report["metrics"].get("claim_length", 1.0) <= 0.60
            for family in heuristics.values()
            for report in family.values()
        ),
        "token_hash_seed_version_integrity": token_passed and source_hashes_match and manifest_design_hash == current_design_hash and provenance["passed"],
        "model_view_forbidden_fields": model_view["passed"],
    }
    finished = utc_now()
    report = {
        "schema_version": "yggdrasil.v2-r1r.p0.audit.v1",
        "evidence_level": "formal" if manifest.get("mode") == "formal" else "smoke",
        "started_at": started,
        "finished_at": finished,
        "command": command or " ".join(sys.argv),
        "environment": environment_record(),
        "design_doc_sha256": current_design_hash,
        "manifest_design_doc_sha256": manifest_design_hash,
        "manifest_sha256": sha256_file(manifest_path),
        "input_file_sha256_observed": file_hash_report,
        "gate_results": gate_results,
        "passed": all(gate_results.values()) and scale["passed"],
        "scale": scale,
        "replay": replay,
        "overlap": overlap,
        "pairs": pair,
        "necessity": necessity,
        "structure": structure,
        "templates": templates,
        "labels": labels,
        "heuristics": heuristics,
        "claims": claims,
        "model_view": model_view,
        "provenance": provenance,
        "token_limit_passed": token_passed,
    }
    return report


def make_assessment(audit_report: Mapping[str, Any]) -> dict[str, Any]:
    gates = dict(audit_report["gate_results"])
    gates["scale_complete"] = bool(audit_report["scale"]["passed"])
    return {
        "schema_version": "yggdrasil.v2-r1r.p0.assessment.v1",
        "evidence_level": audit_report["evidence_level"],
        "passed": all(gates.values()),
        "gate_results": gates,
        "stop_rule": "continue only after formal P0-D conjunction" if all(gates.values()) else "stop at P0-D and diagnose generator/simulator/task-validity/heuristic failure",
        "audit_sha256": hashlib.sha256(json.dumps(audit_report, sort_keys=True).encode("utf-8")).hexdigest(),
    }
