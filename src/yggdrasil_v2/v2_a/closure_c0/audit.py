from __future__ import annotations

"""Read-only, fail-closed audits for V2-A Closure C0."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any, Iterable

from .contract import (
    ARM_CONTRACT,
    AST_KEYS,
    CURRENT_SOURCE_HASHES,
    DATASET_PROFILE,
    GATE_IDS,
    H1_EXCLUSIONS,
    HISTORICAL_EVIDENCE,
    IDENTITY,
    P0D,
    P0M,
    SCHEMA_PREFIX,
    VISIBLE_PATTERN_MAX_ANSWER_MASS,
    VISIBLE_PATTERN_MIN_CLASS_SUPPORT,
    VISIBLE_PATTERN_MIN_SUPPORT,
    VISIBLE_ORACLE_MAX_EXCESS_ACCURACY,
)


_VISIBLE_BOOLEAN_LABEL = re.compile(
    r"(?:\bLabel\s+)?([A-I])\s+(?:means|denotes)\s+(TRUE|FALSE)\b",
    re.IGNORECASE,
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_file(root: Path, relative: str) -> Path | None:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return None
    try:
        base = root.resolve()
        candidate = (root / relative).resolve()
    except OSError:
        return None
    return candidate if candidate != base and base in candidate.parents else None


def _tree_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "evidence-seal.json"
    )


def verify_evidence_seal(
    root: Path,
    *,
    expected_seal_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the seal hash, every declared file, and exact sealed tree."""

    seal_path = root / "evidence-seal.json"
    missing = [] if root.is_dir() else [root.as_posix()]
    if not seal_path.is_file():
        missing.append(seal_path.as_posix())
    if missing:
        return {"available": False, "passed": False, "missing": missing, "failures": []}
    failures: list[str] = []
    seal_hash = sha256_file(seal_path)
    if expected_seal_sha256 is not None and seal_hash != expected_seal_sha256:
        failures.append("seal_sha256_mismatch")
    try:
        payload = read_json(seal_path)
    except (OSError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "passed": False,
            "missing": [],
            "failures": [f"invalid_seal:{type(exc).__name__}:{exc}"],
            "seal_sha256": seal_hash,
        }
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict) or not files:
        return {
            "available": False,
            "passed": False,
            "missing": [],
            "failures": ["invalid_seal_file_map"],
            "seal_sha256": seal_hash,
        }
    declared = sorted(files)
    if declared != _tree_files(root):
        failures.append("sealed_tree_mismatch")
    checked = 0
    for relative, expected in sorted(files.items()):
        path = _safe_file(root, relative)
        if path is None:
            failures.append(f"unsafe_seal_path:{relative}")
            continue
        if not path.is_file():
            failures.append(f"missing_sealed_file:{relative}")
            continue
        if not isinstance(expected, str) or sha256_file(path) != expected.upper():
            failures.append(f"sealed_file_hash_mismatch:{relative}")
        checked += 1
    return {
        "available": True,
        "passed": not failures,
        "missing": [],
        "failures": failures,
        "seal_sha256": seal_hash,
        "declared_file_count": len(files),
        "checked_file_count": checked,
    }


def _cell_filename(family: str, split: str) -> str:
    return f"{family.lower()}-{split}.jsonl"


def _read_json_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"empty JSONL row at {path}:{line_number}")
            value = json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            )
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            yield line_number, value


def scan_dataset(dataset_root: Path) -> dict[str, Any]:
    """Stream all 26,624 records and audit task identity and split isolation."""

    missing = [
        (dataset_root / _cell_filename(family, split)).as_posix()
        for family, splits in DATASET_PROFILE.items()
        for split in splits
        if not (dataset_root / _cell_filename(family, split)).is_file()
    ]
    if missing:
        return {"available": False, "passed": False, "missing": missing, "failures": []}

    failures: list[str] = []
    failure_limit = 100
    counts: dict[str, int] = {}
    claim_balance: dict[str, Counter[bool]] = {}
    ids: set[str] = set()
    semantic_owner: dict[str, str] = {}
    surface_owner: dict[str, str] = {}
    semantic_overlaps: list[dict[str, str]] = []
    surface_overlaps: list[dict[str, str]] = []
    pair_members: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)
    ast_keysets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    visible_support: dict[str, Counter[str]] = defaultdict(Counter)
    relation_legend_correct: Counter[str] = Counter()
    relation_legend_total: Counter[str] = Counter()
    relation_legend_chance: dict[str, float] = defaultdict(float)

    def fail(reason: str) -> None:
        if len(failures) < failure_limit:
            failures.append(reason)

    try:
        for family, splits in DATASET_PROFILE.items():
            for split, expected_count in splits.items():
                cell = f"{family}/{split}"
                path = dataset_root / _cell_filename(family, split)
                cell_count = 0
                balance: Counter[bool] = Counter()
                for line_number, row in _read_json_lines(path):
                    cell_count += 1
                    prefix = f"{cell}:{line_number}"
                    if row.get("family") != family or row.get("split") != split:
                        fail(f"family_or_split_mismatch:{prefix}")
                    identifier = row.get("example_id")
                    if not isinstance(identifier, str) or not identifier:
                        fail(f"invalid_example_id:{prefix}")
                    elif identifier in ids:
                        fail(f"duplicate_example_id:{identifier}")
                    else:
                        ids.add(identifier)
                    if not isinstance(row.get("source_text"), str) or not row["source_text"].strip():
                        fail(f"empty_source_text:{prefix}")
                    ast = row.get("program_ast")
                    if not isinstance(ast, dict):
                        fail(f"invalid_program_ast:{prefix}")
                    else:
                        keys = tuple(sorted(ast))
                        ast_keysets[family].add(keys)
                        if keys != tuple(sorted(AST_KEYS[family])):
                            fail(f"unexpected_ast_keys:{prefix}")
                        if family == "ERE":
                            query = ast.get("query")
                            query_kind = query.get("kind") if isinstance(query, dict) else None
                            if isinstance(query_kind, str):
                                pattern = f"{cell}/query_kind={query_kind}"
                                visible_support[pattern][str(row.get("semantic_answer"))] += 1
                        visible_boolean_mapping = {
                            label.upper(): semantic.lower()
                            for label, semantic in _VISIBLE_BOOLEAN_LABEL.findall(row["source_text"])
                        }
                        if set(visible_boolean_mapping.values()) == {"true", "false"}:
                            relation_legend_total[cell] += 1
                            relation_legend_chance[cell] += 1.0 / sum(row["valid_choice_mask"])
                            true_labels = [
                                label
                                for label, semantic in visible_boolean_mapping.items()
                                if semantic == "true"
                            ]
                            if len(true_labels) == 1:
                                predicted = ord(true_labels[0]) - ord("A")
                                relation_legend_correct[cell] += predicted == row.get("answer_index")
                    answer = row.get("answer_index")
                    mask = row.get("valid_choice_mask")
                    if (
                        isinstance(answer, bool)
                        or not isinstance(answer, int)
                        or not isinstance(mask, list)
                        or not all(type(item) is bool for item in mask)
                        or not 0 <= answer < len(mask)
                        or not mask[answer]
                    ):
                        fail(f"invalid_answer_or_mask:{prefix}")
                    trace = row.get("teacher_trace")
                    if not isinstance(trace, list) or not trace:
                        fail(f"missing_teacher_trace:{prefix}")
                    claims = row.get("training_claims")
                    if not isinstance(claims, list) or not claims:
                        fail(f"missing_training_claims:{prefix}")
                    else:
                        for claim in claims:
                            if not isinstance(claim, dict) or type(claim.get("label")) is not bool:
                                fail(f"invalid_training_claim:{prefix}")
                                continue
                            balance[claim["label"]] += 1
                    for field, owners, overlaps in (
                        ("semantic_fingerprint", semantic_owner, semantic_overlaps),
                        ("surface_fingerprint", surface_owner, surface_overlaps),
                    ):
                        fingerprint = row.get(field)
                        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                            fail(f"invalid_{field}:{prefix}")
                        elif fingerprint in owners:
                            overlaps.append(
                                {"fingerprint": fingerprint, "first": owners[fingerprint], "second": cell}
                            )
                        else:
                            owners[fingerprint] = cell
                    if split == "causal_pairs":
                        pair_id = row.get("pair_id")
                        role = row.get("pair_role")
                        semantic_answer = row.get("semantic_answer")
                        if not isinstance(pair_id, str) or role not in {"base", "flip"}:
                            fail(f"invalid_causal_pair_identity:{prefix}")
                        elif role in pair_members[pair_id]:
                            fail(f"duplicate_causal_pair_role:{pair_id}:{role}")
                        else:
                            pair_members[pair_id][role] = (str(semantic_answer), answer)
                counts[cell] = cell_count
                claim_balance[cell] = balance
                if cell_count != expected_count:
                    fail(f"count_mismatch:{cell}:{cell_count}!={expected_count}")
                if balance[True] != balance[False] or balance[True] == 0:
                    fail(f"claim_imbalance:{cell}:{balance[True]}!={balance[False]}")
    except (OSError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "passed": False,
            "missing": [],
            "failures": [f"dataset_parse_error:{type(exc).__name__}:{exc}"],
        }

    for pair_id, members in pair_members.items():
        if set(members) != {"base", "flip"}:
            fail(f"incomplete_causal_pair:{pair_id}")
        elif members["base"] == members["flip"] or members["base"][0] == members["flip"][0]:
            fail(f"causal_pair_does_not_flip_answer:{pair_id}")
    if semantic_overlaps:
        fail(f"semantic_fingerprint_overlap:{len(semantic_overlaps)}")
    if surface_overlaps:
        fail(f"surface_fingerprint_overlap:{len(surface_overlaps)}")
    ast_distinct = (
        ast_keysets.get("ERE") == {tuple(sorted(AST_KEYS["ERE"]))}
        and ast_keysets.get("CPS") == {tuple(sorted(AST_KEYS["CPS"]))}
        and ast_keysets["ERE"] != ast_keysets["CPS"]
    )
    if not ast_distinct:
        fail("task_ast_families_not_distinct")
    record_integrity_passed = not failures
    registered_visible_support: list[dict[str, Any]] = []
    semantic_support_passed = True
    for pattern, answers in sorted(visible_support.items()):
        if not pattern.endswith("query_kind=relation"):
            continue
        support = sum(answers.values())
        dominant_answer, dominant_count = answers.most_common(1)[0]
        dominant_mass = dominant_count / support
        minority_count = min(answers.values()) if len(answers) >= 2 else 0
        row_passed = support < VISIBLE_PATTERN_MIN_SUPPORT or (
            dominant_mass <= VISIBLE_PATTERN_MAX_ANSWER_MASS
            and minority_count >= VISIBLE_PATTERN_MIN_CLASS_SUPPORT
        )
        semantic_support_passed = semantic_support_passed and row_passed
        registered_visible_support.append(
            {
                "pattern": pattern,
                "support": support,
                "answer_counts": dict(sorted(answers.items())),
                "dominant_answer": dominant_answer,
                "dominant_answer_mass": dominant_mass,
                "minimum_support": VISIBLE_PATTERN_MIN_SUPPORT,
                "maximum_allowed_mass": VISIBLE_PATTERN_MAX_ANSWER_MASS,
                "minority_count": minority_count,
                "minimum_each_class_support": VISIBLE_PATTERN_MIN_CLASS_SUPPORT,
                "passed": row_passed,
            }
        )
    tested_oracle_cells = [
        cell for cell, total in relation_legend_total.items() if total >= VISIBLE_PATTERN_MIN_SUPPORT
    ]
    adjusted_alpha = 0.05 / max(1, len(tested_oracle_cells))

    def wilson_upper(successes: int, total: int) -> float:
        if total <= 0:
            return 1.0
        z = NormalDist().inv_cdf(1.0 - adjusted_alpha)
        proportion = successes / total
        z2 = z * z
        return (
            proportion
            + z2 / (2 * total)
            + z * math.sqrt((proportion * (1 - proportion) + z2 / (4 * total)) / total)
        ) / (1 + z2 / total)

    relation_oracle_rows: dict[str, dict[str, Any]] = {}
    oracle_passed = True
    for cell, total in sorted(relation_legend_total.items()):
        correct = relation_legend_correct[cell]
        chance = relation_legend_chance[cell] / total
        upper = wilson_upper(correct, total)
        tested = total >= VISIBLE_PATTERN_MIN_SUPPORT
        row_passed = not tested or upper <= chance + VISIBLE_ORACLE_MAX_EXCESS_ACCURACY
        oracle_passed = oracle_passed and row_passed
        relation_oracle_rows[cell] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "mean_random_chance": chance,
            "one_sided_wilson_upper": upper,
            "adjusted_alpha": adjusted_alpha,
            "maximum_excess_accuracy": VISIBLE_ORACLE_MAX_EXCESS_ACCURACY,
            "tested": tested,
            "passed": row_passed,
            "materiality_accuracy_points": (total / counts[cell]) * (correct / total - chance),
            "input_fields": ["source_text"],
        }
    visible_pattern_passed = semantic_support_passed and oracle_passed
    if not semantic_support_passed:
        fail("registered_visible_pattern_semantic_support_failed")
    if not oracle_passed:
        fail("registered_source_only_visible_legend_oracle_failed")
    return {
        "available": True,
        "passed": not failures,
        "missing": [],
        "failures": failures,
        "failure_count_capped_at": failure_limit,
        "counts": counts,
        "total_records": sum(counts.values()),
        "unique_example_ids": len(ids),
        "semantic_fingerprint_overlap_count": len(semantic_overlaps),
        "surface_fingerprint_overlap_count": len(surface_overlaps),
        "causal_pair_count": len(pair_members),
        "claim_balance": {
            cell: {"positive": balance[True], "negative": balance[False]}
            for cell, balance in sorted(claim_balance.items())
        },
        "ast_keysets": {
            family: [list(keys) for keys in sorted(keysets)]
            for family, keysets in sorted(ast_keysets.items())
        },
        "ast_families_distinct": ast_distinct,
        "record_integrity_passed": record_integrity_passed,
        "registered_visible_pattern_support": registered_visible_support,
        "registered_visible_pattern_support_passed": visible_pattern_passed,
        "relation_visible_legend_oracle": relation_oracle_rows,
    }


def _get_path(value: Any, dotted: str) -> Any:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def _report_passed(report: Any) -> bool:
    return isinstance(report, dict) and report.get("passed") is True and report.get("failures") == []


def _claim_report_balanced(report: Any) -> bool:
    try:
        cells = report["value"]["balance"]
        return bool(cells) and all(
            counts.get("True") == counts.get("False") and counts.get("True", 0) > 0
            for cell in cells.values()
            for counts in cell["by_kind"].values()
        )
    except (KeyError, TypeError):
        return False


def _arm_contract_valid() -> bool:
    return bool(
        ARM_CONTRACT.get("arms") == ["direct", "text_cot", "latent_k1", "latent_k8"]
        and ARM_CONTRACT["shared_inputs"].get("family_visible_to_forward") is False
        and ARM_CONTRACT["shared_inputs"].get("family_used_as_training_target") is False
        and ARM_CONTRACT["shared_inputs"].get("canonical_public_forward_fields")
        == ["source_text"]
        and ARM_CONTRACT["shared_inputs"].get("valid_choice_mask_visible_to_forward") is False
        and ARM_CONTRACT["shared_inputs"].get("reasoning_budget_visible_to_forward") is False
        and ARM_CONTRACT["teacher_symmetry"].get("required") is True
        and ARM_CONTRACT["teacher_symmetry"]["primary_lane"].get(
            "dense_state_or_claim_targets_allowed"
        )
        is False
        and ARM_CONTRACT["teacher_symmetry"]["secondary_lane"].get(
            "eligible_for_medium_superiority_claim"
        )
        is False
        and ARM_CONTRACT["latent_capacity_match"].get("only_registered_difference")
        == "slot_count: 1 versus 8"
        and ARM_CONTRACT["selection_and_budget"].get("heldout_arm_specific_tuning_forbidden")
        is True
        and len(ARM_CONTRACT.get("cost_ledger", [])) >= 10
        and len(ARM_CONTRACT.get("evaluation", [])) >= 7
        and ARM_CONTRACT["selection_and_budget"].get("required_matched_slices")
        == ["equal_examples", "equal_gpu_hours"]
        and ARM_CONTRACT["trace_qualification"].get("decode_max_new_tokens") == 512
        and ARM_CONTRACT["cache_policy"].get("p0m_smoke_cache_reusable_for_c1") is False
        and "route_id" in ARM_CONTRACT.get("forbidden_active_components", [])
        and ARM_CONTRACT.get("latency_rule")
        == "online end-to-end paths are primary; cached latency is diagnostic only"
    )


def _audit_compact_trace_qualification(repo_root: Path) -> dict[str, Any]:
    """Distinguish the historical formatter/answer check from a trace verifier."""

    source_root = repo_root / P0M["root"] / "source_snapshot"
    formatter_path = source_root / "src/yggdrasil_v2/r1_revalidation/p0m/data.py"
    evaluator_path = source_root / "src/yggdrasil_v2/r1_revalidation/p0m/baseline.py"
    available = formatter_path.is_file() and evaluator_path.is_file()
    formatter_text = formatter_path.read_text(encoding="utf-8") if formatter_path.is_file() else ""
    evaluator_text = evaluator_path.read_text(encoding="utf-8") if evaluator_path.is_file() else ""
    formatter_present = "def compact_teacher_trace" in formatter_text
    answer_parser_present = "def _extract_answer" in evaluator_text
    trace_parser_present = any(
        marker in formatter_text + evaluator_text
        for marker in ("def parse_compact_teacher_trace", "def _extract_trace", "def verify_compact_teacher_trace")
    )
    semantic_verifier_present = any(
        marker in formatter_text + evaluator_text
        for marker in ("trace_semantic", "semantic_trace", "simulator_replay")
    )
    # The pinned P0-M sources contain no full-bank trace roundtrip or directed trace-fault suite.
    full_bank_roundtrip_present = "formatter_parser_roundtrip_all_records" in formatter_text + evaluator_text
    directed_fault_kill_present = "directed_trace_fault_kill" in formatter_text + evaluator_text
    passed = bool(
        available
        and formatter_present
        and trace_parser_present
        and semantic_verifier_present
        and full_bank_roundtrip_present
        and directed_fault_kill_present
    )
    return {
        "available": available,
        "passed": passed,
        "formatter_present": formatter_present,
        "answer_parser_present": answer_parser_present,
        "compact_trace_parser_present": trace_parser_present,
        "semantic_trace_verifier_present": semantic_verifier_present,
        "full_bank_roundtrip_present": full_bank_roundtrip_present,
        "directed_fault_kill_present": directed_fault_kill_present,
        "interpretation": (
            "P0-M proves compact trace formatting and final-answer parsing only; it does not "
            "qualify generated trace semantics for the matched-supervision primary lane."
        ),
    }


def audit_historical_evidence(archive_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    failures: list[str] = []
    for entry in HISTORICAL_EVIDENCE:
        path = archive_root / entry["path"]
        row = {
            key: entry[key]
            for key in ("stage", "path", "classification", "reuse", "limit")
        }
        if not path.is_file():
            missing.append(path.as_posix())
            row.update({"available": False, "verified": False})
            rows.append(row)
            continue
        actual_hash = sha256_file(path)
        row.update({"available": True, "sha256": actual_hash})
        ok = actual_hash == entry["sha256"]
        try:
            value = read_json(path)
            semantic_checks = {
                dotted: _get_path(value, dotted) == expected
                for dotted, expected in entry["checks"].items()
            }
        except (OSError, TypeError, ValueError, KeyError) as exc:
            semantic_checks = {"parse": False}
            failures.append(f"historical_parse_error:{entry['stage']}:{type(exc).__name__}")
        ok = ok and all(semantic_checks.values())
        if not ok:
            failures.append(f"historical_identity_or_semantics_mismatch:{entry['stage']}")
        row.update({"semantic_checks": semantic_checks, "verified": ok})
        rows.append(row)
    return {
        "schema_version": f"{SCHEMA_PREFIX}.reuse-matrix.v1",
        "archive_root": archive_root.as_posix(),
        "available": not missing,
        "passed": not missing and not failures and all(row["verified"] for row in rows),
        "missing": missing,
        "failures": failures,
        "entries": rows,
        "global_boundary": (
            "Historical evidence may supply mechanisms, tests, and negative constraints only; "
            "it may not initialize C1, replace fresh ERE/CPS results, or close matched Pareto."
        ),
    }


def audit_closure_c0(repo_root: Path, archive_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the complete C0 audit and return input audit plus reuse matrix."""

    p0d_root = repo_root / P0D["root"]
    p0m_root = repo_root / P0M["root"]
    p0d_seal = verify_evidence_seal(p0d_root, expected_seal_sha256=P0D["seal_sha256"])
    p0m_seal = verify_evidence_seal(p0m_root, expected_seal_sha256=P0M["seal_sha256"])
    dataset = scan_dataset(p0d_root / "dataset")
    missing: list[str] = []
    failures: list[str] = []

    def load(relative: str) -> Any | None:
        path = repo_root / relative
        if not path.is_file():
            missing.append(path.as_posix())
            return None
        try:
            return read_json(path)
        except (OSError, TypeError, ValueError) as exc:
            failures.append(f"invalid_json:{relative}:{type(exc).__name__}:{exc}")
            return None

    assessment = load(f"{P0D['root']}/assessment.json")
    manifest = load(f"{P0D['root']}/dataset/manifest.json")
    shortcut = load(f"{P0D['root']}/shortcut-report.json")
    claim = load(f"{P0D['root']}/claim-report.json")
    p0m_result = load(f"{P0M['root']}/result.json")

    pinned_hashes: dict[str, dict[str, Any]] = {}
    for relative, expected in (
        (f"{P0D['root']}/assessment.json", P0D["assessment_sha256"]),
        (f"{P0D['root']}/dataset/manifest.json", P0D["manifest_sha256"]),
        (f"{P0M['root']}/result.json", P0M["result_sha256"]),
    ):
        path = repo_root / relative
        actual = sha256_file(path) if path.is_file() else None
        pinned_hashes[relative] = {"expected": expected, "actual": actual, "passed": actual == expected}

    source_identity: dict[str, dict[str, Any]] = {}
    for relative, expected in CURRENT_SOURCE_HASHES.items():
        path = repo_root / relative
        if not path.is_file():
            missing.append(path.as_posix())
            actual = None
        else:
            actual = sha256_file(path)
        source_identity[relative] = {"expected": expected, "actual": actual, "passed": actual == expected}

    manifest_profile_ok = False
    if isinstance(manifest, dict):
        expected_counts = {
            f"{family}/{split}": count
            for family, splits in DATASET_PROFILE.items()
            for split, count in splits.items()
        }
        manifest_profile_ok = bool(
            manifest.get("counts") == expected_counts
            and manifest.get("total_records") == 26624
            and manifest.get("model_id") == P0D["model_id"]
            and manifest.get("model_revision") == P0D["model_revision"]
        )

    p0m_components: dict[str, Any] = {}
    if isinstance(p0m_result, dict) and isinstance(p0m_result.get("roots"), dict):
        for component in P0M["components"]:
            relative = p0m_result["roots"].get(component)
            if not isinstance(relative, str):
                missing.append(f"p0m_component_root:{component}")
                continue
            component_root = repo_root / relative
            seal_audit = verify_evidence_seal(component_root)
            result_path = component_root / "result.json"
            result_passed = False
            if result_path.is_file():
                try:
                    result_passed = read_json(result_path).get("passed") is True
                except (OSError, TypeError, ValueError):
                    result_passed = False
            else:
                missing.append(result_path.as_posix())
            p0m_components[component] = {
                "root": relative,
                "seal": seal_audit,
                "result_passed": result_passed,
                "passed": seal_audit["passed"] and result_passed,
            }

    historical = audit_historical_evidence(archive_root)
    trace_qualification = _audit_compact_trace_qualification(repo_root)

    h1_results: list[dict[str, Any]] = []
    for entry in H1_EXCLUSIONS:
        path = repo_root / entry["path"]
        if not path.is_file():
            missing.append(path.as_posix())
            h1_results.append({"name": entry["name"], "available": False, "passed": False})
            continue
        actual_hash = sha256_file(path)
        try:
            payload = read_json(path)
            checks = {
                dotted: _get_path(payload, dotted) == expected
                for dotted, expected in entry["checks"].items()
            }
        except (OSError, TypeError, ValueError, KeyError):
            checks = {"parse": False}
        h1_results.append(
            {
                "name": entry["name"],
                "path": entry["path"],
                "available": True,
                "sha256": actual_hash,
                "hash_passed": actual_hash == entry["sha256"],
                "checks": checks,
                "passed": actual_hash == entry["sha256"] and all(checks.values()),
            }
        )

    for section in (p0d_seal, p0m_seal, dataset, historical):
        missing.extend(section.get("missing", []))
        failures.extend(section.get("failures", []))

    c001 = bool(
        p0d_seal["passed"]
        and isinstance(assessment, dict)
        and assessment.get("passed") is True
        and assessment.get("status") == "PASS_P0D_PRODUCTION"
        and set(assessment.get("gates", {}).values()) == {True}
        and all(item["passed"] for item in pinned_hashes.values())
        and all(item["passed"] for item in source_identity.values())
    )
    c002 = bool(dataset.get("ast_families_distinct"))
    c003 = bool(dataset.get("record_integrity_passed") and manifest_profile_ok)
    c004 = bool(
        _report_passed(shortcut)
        and _report_passed(claim)
        and _claim_report_balanced(claim)
        and dataset.get("registered_visible_pattern_support_passed") is True
    )
    p0m_gate_values = p0m_result.get("gates", {}) if isinstance(p0m_result, dict) else {}
    c005 = bool(
        p0m_seal["passed"]
        and isinstance(p0m_result, dict)
        and p0m_result.get("passed") is True
        and p0m_result.get("status") == "PASS_P0M"
        and p0m_result.get("interpretation") == P0M["interpretation"]
        and set(p0m_gate_values) == {f"M0{index}" for index in range(1, 9)}
        and set(p0m_gate_values.values()) == {True}
        and set(p0m_components) == set(P0M["components"])
        and all(item["passed"] for item in p0m_components.values())
    )
    c006 = _arm_contract_valid() and trace_qualification["passed"]
    c007 = historical["passed"]
    c008 = len(h1_results) == len(H1_EXCLUSIONS) and all(item["passed"] for item in h1_results)

    gates = dict(zip(GATE_IDS, (c001, c002, c003, c004, c005, c006, c007, c008), strict=True))
    if not trace_qualification["passed"]:
        failures.append("matched_compact_trace_verifier_not_qualified")
    availability_complete = not missing and all(
        section.get("available") is True for section in (p0d_seal, p0m_seal, dataset, historical)
    )
    audit = {
        "schema_version": f"{SCHEMA_PREFIX}.input-audit.v1",
        "identity": IDENTITY,
        "availability_complete": availability_complete,
        "missing": sorted(set(missing)),
        "audit_failures": sorted(set(failures)),
        "gates": gates,
        "p0d": {
            "seal": p0d_seal,
            "pinned_hashes": pinned_hashes,
            "source_identity": source_identity,
            "assessment_passed": isinstance(assessment, dict) and assessment.get("passed") is True,
            "manifest_profile_passed": manifest_profile_ok,
            "shortcut_report_passed": _report_passed(shortcut),
            "claim_report_passed": _report_passed(claim),
            "claim_report_balanced": _claim_report_balanced(claim),
            "dataset": dataset,
        },
        "p0m": {
            "seal": p0m_seal,
            "assessment_interpretation": p0m_result.get("interpretation") if isinstance(p0m_result, dict) else None,
            "components": p0m_components,
        },
        "arm_contract": {
            "passed": c006,
            "contract_frozen": _arm_contract_valid(),
            "trace_qualification": trace_qualification,
            "contract": ARM_CONTRACT,
        },
        "h1_exclusions": {
            "passed": c008,
            "results": h1_results,
            "active_c1_excludes_routed_projection_and_old_checkpoints": True,
        },
    }
    return audit, historical


def decide(audit: dict[str, Any]) -> dict[str, Any]:
    gates = audit.get("gates", {})
    complete = audit.get("availability_complete") is True
    all_passed = set(gates) == set(GATE_IDS) and all(gates.values())
    if not complete:
        status = "INCOMPLETE_V2_A_CLOSURE_C0_READINESS"
        exit_code = 3
    elif not all_passed:
        status = "FAIL_V2_A_CLOSURE_C0_READINESS"
        exit_code = 1
    else:
        status = "PASS_V2_A_CLOSURE_C0_READINESS"
        exit_code = 0
    passed = status.startswith("PASS_")
    return {
        "schema_version": f"{SCHEMA_PREFIX}.result.v1",
        "identity": IDENTITY,
        "status": status,
        "passed": passed,
        "exit_code": exit_code,
        "gates": gates,
        "task_suite_qualified": passed,
        "baseline_contract_qualified": passed,
        "four_arm_results_present": False,
        "training_started": False,
        "optimizer_steps": 0,
        "model_writes": 0,
        "c1_single_seed_implementation_authorized": passed,
        "v2a_passed": False,
        "v2b_authorized": False,
        "v2c_authorized": False,
        "authorizes": "C1 single-seed implementation and eligibility only" if passed else "nothing",
        "interpretation": (
            "C0 qualifies the task suite and comparison contract only. It does not provide "
            "four-arm results, architecture evidence, matched Pareto closure, or V2-A completion."
        ),
    }


__all__ = [
    "audit_closure_c0",
    "audit_historical_evidence",
    "decide",
    "read_json",
    "scan_dataset",
    "sha256_file",
    "verify_evidence_seal",
]
