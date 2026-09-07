from __future__ import annotations

"""Independent audits for the v17 repaired production P0-D dataset."""

from collections import Counter, defaultdict
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from ..common import evaluate_cps, replay_cps_prefix, simulate_ere
from .fingerprint import alpha_rename, semantic_fingerprint, surface_fingerprint
from .generator import (
    GENERATOR_VERSION,
    DatasetSpec,
    MODEL_ID,
    MODEL_REVISION,
    SCHEMA_VERSION,
    SEED_DERIVATION_VERSION,
    SPLITS,
    load_production_dataset,
    verify_claim,
)
from .renderer import LABELS, TRAIN_TEMPLATES, parse_source, render_source
from .g09_decision import (
    BASELINES,
    SOURCE_CELL_COUNT,
    SOURCE_FAMILY_CELL_COUNT,
    evaluate_g09_counts,
    wilson_upper as _wilson_upper,
)
from .p0_shortcut import (
    fit_prepared_multinomial_nb,
    predict_prepared_multinomial_nb_batch,
    prepare_nb_rows,
)
from .scalable import compact_predict_multinomial_nb, fit_scalable_multinomial_nb
from .schema import MODEL_VIEW_FIELDS, model_view


PRIOR_SEALS = {
    "artifacts/v2-r1r/p0d-v7-r0a-lattice-20260801-1/evidence-seal.json": "794CB9518F094EE6CF99800AF0B5848D22A7746DAA5E76A84489F3F8201987F0",
    "artifacts/v2-r1r/p0d-v9-r0b-invariant-20260802-1/evidence-seal.json": "94B80FBD2BB38BC46E95F88E0747D9DB1D8CB279E689F68BB4A43AE7B76D8209",
    "artifacts/v2-r1r/p0d-v10-r0c-statistical-20260802-1/evidence-seal.json": "3F003C8BED40F54F010F4C7A96207D703ABA59F1088F189C1265D0F9E1C79BA1",
    "artifacts/v2-r1r/p0d-v11-r0d-integrated-20260802-1/evidence-seal.json": "1C4AD436CECA3F96F4286A2E5A62D272782E7EF02DF5A52BBD5C6493A43408CF",
    "artifacts/v2-r1r/r1e-v12-entry-qualification-20260809-1/evidence-seal.json": "86B49779D4E5A7EF496EE6FC0C0C722071983E5B3C6FFB9F8D90D92FBB9E78CA",
    "artifacts/v2-r1r/r1g-v13-generator-smoke-20260809-1/evidence-seal.json": "A9CF968F15634E7F1F9471CDE5383736C6D0E17DCC8208E871B6C779062AE455",
}
SHARED_PREFIXES = (
    "src/yggdrasil_v2/r1_revalidation/common/",
    "src/yggdrasil_v2/r1_revalidation/audit/",
    "src/yggdrasil_v2/r1_revalidation/learner/",
    "src/yggdrasil_v2/r1_revalidation/integration/",
)
ENTRY_FILES = (
    "src/yggdrasil_v2/r1_revalidation/production/renderer.py",
    "src/yggdrasil_v2/r1_revalidation/production/scalable.py",
    "src/yggdrasil_v2/r1_revalidation/production/schema.py",
)
RUNTIME_QUALIFICATION_ROOT = Path("artifacts/v2-r1r/p0d-v16-runtime-qualification-20260810-1")
QUALIFICATION_ROOT = Path("artifacts/v2-r1r/p0d-v17-repair-qualification-20260810-1")
RECORD_FIELDS = {
    "schema_version", "generator_version", "example_id", "family", "split", "record_seed",
    "template_id", "source_text", "token_count", "reasoning_budget", "valid_choice_mask",
    "answer_index", "semantic_answer", "semantic_fingerprint", "surface_fingerprint", "pair_id",
    "pair_role", "program_ast", "label_mapping", "teacher_trace", "training_claims",
    "causal_certificate", "simulator_output", "generation_metadata", "provenance",
}
FORBIDDEN_MARKERS = (
    "composition_ood", "length_ood", "entity_ood", "horizon_ood", "distractor_ood",
    "language_ood", "causal_pairs", "plain_v1", "reordered_v1", "indirect_v1",
    "20260814", "20260815", "20260816", "record_seed", "root_seed", "answer_index", "semantic_answer", "generator_version",
)

_FAMILY_SEED_STRIDE = 10_000_019
_SPLIT_SEED_STRIDE = 1_000_003
_UNIT_SEED_STRIDE = 104_729
_ATTEMPT_SEED_STRIDE = 7_919
CLAIM_CELL_COUNT = 24


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest().upper()


def _metric(passed: bool, value: Any, failures: Sequence[str] = ()) -> dict[str, Any]:
    return {"passed": bool(passed), "value": value, "failures": list(failures)[:50]}


def _verify_qualification(repo_root: Path) -> dict[str, Any]:
    root = repo_root / QUALIFICATION_ROOT
    assessment_path = root / "assessment.json"
    seal_path = root / "evidence-seal.json"
    if not assessment_path.is_file() or not seal_path.is_file():
        return {"root": QUALIFICATION_ROOT.as_posix(), "passed": False, "failure": "missing"}
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    actual = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "evidence-seal.json"
    }
    passed = (
        assessment.get("status") == "PASS_REPAIR_QUALIFICATION"
        and assessment.get("passed") is True
        and isinstance(seal, dict)
        and actual == seal.get("files")
    )
    return {
        "root": QUALIFICATION_ROOT.as_posix(),
        "assessment_sha256": _sha(assessment_path),
        "evidence_seal_sha256": _sha(seal_path),
        "sealed_file_count": len(actual),
        "passed": passed,
        "failure": None if passed else "assessment_or_seal",
    }


def verify_prerequisites(repo_root: Path, *, require_qualification: bool = False) -> dict[str, Any]:
    failures: list[str] = []
    seals: list[dict[str, Any]] = []
    for relative, expected in PRIOR_SEALS.items():
        path = repo_root / relative
        actual = _sha(path) if path.is_file() else None
        passed = actual == expected
        seals.append({"path": relative, "expected": expected, "actual": actual, "passed": passed})
        if not passed:
            failures.append(f"seal:{relative}")

    v11_manifest_path = repo_root / "artifacts/v2-r1r/p0d-v11-r0d-integrated-20260802-1/integration-bundle/manifest.json"
    v11_manifest = json.loads(v11_manifest_path.read_text(encoding="utf-8")) if v11_manifest_path.is_file() else {}
    v11_files = v11_manifest.get("files", {}) if isinstance(v11_manifest, dict) else {}
    runtime_rows: list[dict[str, Any]] = []
    actual_shared: list[str] = []
    for path in sorted((repo_root / "src/yggdrasil_v2/r1_revalidation").rglob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        if not relative.startswith(SHARED_PREFIXES):
            continue
        actual_shared.append(relative)
        expected = v11_files.get(f"source_snapshot/{relative}")
        actual = _sha(path)
        passed = isinstance(expected, str) and actual == expected
        runtime_rows.append({"path": relative, "expected": expected, "actual": actual, "passed": passed})
        if not passed:
            failures.append(f"shared_runtime:{relative}")
    expected_shared = sorted(
        key.removeprefix("source_snapshot/")
        for key in v11_files
        if key.startswith("source_snapshot/") and key.removeprefix("source_snapshot/").startswith(SHARED_PREFIXES)
    )
    if actual_shared != expected_shared:
        failures.append("shared_runtime:file_set")

    v12_seal_path = repo_root / "artifacts/v2-r1r/r1e-v12-entry-qualification-20260809-1/evidence-seal.json"
    v12_seal = json.loads(v12_seal_path.read_text(encoding="utf-8")) if v12_seal_path.is_file() else {}
    v12_files = v12_seal.get("files", {}) if isinstance(v12_seal, dict) else {}
    entry_rows: list[dict[str, Any]] = []
    for relative in ENTRY_FILES:
        expected = v12_files.get(f"source_snapshot/{relative}")
        path = repo_root / relative
        actual = _sha(path) if path.is_file() else None
        passed = isinstance(expected, str) and actual == expected
        entry_rows.append({"path": relative, "expected": expected, "actual": actual, "passed": passed})
        if not passed:
            failures.append(f"entry_runtime:{relative}")
    qualification = _verify_qualification(repo_root) if require_qualification else {
        "root": QUALIFICATION_ROOT.as_posix(), "required": False, "passed": True
    }
    if not qualification["passed"]:
        failures.append("v17_repair_qualification")
    return {
        "prior_seals": seals,
        "shared_runtime": runtime_rows,
        "entry_runtime": entry_rows,
        "qualification": qualification,
        "passed": not failures and len(seals) == len(PRIOR_SEALS) and actual_shared == expected_shared,
        "failures": failures,
    }


def _semantic_answer(family: str, output: Mapping[str, Any]) -> str:
    if family == "ERE":
        answer = output["answer"]
        if isinstance(answer, bool):
            return "true" if answer else "false"
        return f"value:{answer}"
    answer = output["answer"]
    return "none" if answer is None else f"candidate:{answer}"


def _answer_label(record: Mapping[str, Any], semantic: str) -> str:
    labels = [label for label, meaning in record["label_mapping"].items() if meaning == semantic]
    if len(labels) != 1:
        raise ValueError("semantic answer does not bind to exactly one label")
    return labels[0]


def _get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("/")[1:]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _patched(value: Any, path: str, replacement: Any) -> Any:
    output = deepcopy(value)
    parts = path.split("/")[1:]
    current = output
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    if isinstance(current, list):
        current[int(parts[-1])] = deepcopy(replacement)
    else:
        current[parts[-1]] = deepcopy(replacement)
    return output


def _diff_paths(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, Mapping):
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


def _flatten(dataset: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for key in sorted(dataset) for row in dataset[key]]


def _audit_schema(
    repo_root: Path,
    dataset_root: Path,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
) -> dict[str, Any]:
    failures: list[str] = []
    expected_counts = {
        f"{family}/{split}": spec.count(split)
        for family in ("ERE", "CPS")
        for split in SPLITS[family]
    }
    expected_files = {f"{family.lower()}-{split}.jsonl" for family in ("ERE", "CPS") for split in SPLITS[family]}
    if manifest.get("counts") != expected_counts:
        failures.append("manifest_counts")
    if set(manifest.get("files", {})) != expected_files:
        failures.append("manifest_files")
    expected_total = sum(expected_counts.values())
    if manifest.get("total_records") != expected_total or len(records) != expected_total:
        failures.append("total_records")
    expected_manifest = {
        "schema_version": "yggdrasil.v2-r1r.p0d-v17.dataset-manifest.v1",
        "generator_version": GENERATOR_VERSION,
        "profile": spec.profile,
        "root_seed": spec.root_seed,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "add_special_tokens": False,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            failures.append(f"manifest:{key}")
    design = repo_root / "docs/v2-r1r-p0d-v17-repaired-production-design.md"
    if manifest.get("design_sha256") != _sha(design):
        failures.append("manifest:design_sha256")
    ids: set[str] = set()
    seeds: set[int] = set()
    expected_stats: dict[str, dict[str, int]] = {}
    for row in records:
        identifier = str(row.get("example_id"))
        if set(row) != RECORD_FIELDS:
            failures.append(f"record_schema:{identifier}")
        if row.get("schema_version") != SCHEMA_VERSION or row.get("generator_version") != GENERATOR_VERSION:
            failures.append(f"record_version:{identifier}")
        family = row.get("family")
        split = row.get("split")
        if family not in SPLITS or split not in SPLITS.get(family, ()):
            failures.append(f"partition:{identifier}")
        if identifier in ids:
            failures.append(f"duplicate_id:{identifier}")
        ids.add(identifier)
        seed = row.get("record_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            failures.append(f"record_seed:{identifier}")
        else:
            # A causal pair intentionally shares one seed; all other records do not.
            if row.get("pair_id") is None and seed in seeds:
                failures.append(f"duplicate_seed:{identifier}")
            seeds.add(seed)
        provenance = row.get("provenance", {})
        family_index = ("ERE", "CPS").index(family) if family in {"ERE", "CPS"} else -1
        split_index = SPLITS.get(family, ()).index(split) if split in SPLITS.get(family, ()) else -1
        split_seed = spec.root_seed + family_index * _FAMILY_SEED_STRIDE + split_index * _SPLIT_SEED_STRIDE
        unit_index = provenance.get("unit_index")
        attempt = provenance.get("generation_attempt")
        expected_record_seed = (
            split_seed + unit_index * _UNIT_SEED_STRIDE + attempt * _ATTEMPT_SEED_STRIDE
            if isinstance(unit_index, int) and not isinstance(unit_index, bool) and isinstance(attempt, int) and not isinstance(attempt, bool)
            else None
        )
        if provenance != {
            "generator_version": GENERATOR_VERSION,
            "root_seed": spec.root_seed,
            "family_index": family_index,
            "split_index": split_index,
            "split_seed": split_seed,
            "unit_index": unit_index,
            "generation_attempt": attempt,
            "record_seed": seed,
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "design_sha256": manifest.get("design_sha256"),
            "python_major_minor": provenance.get("python_major_minor"),
        } or seed != expected_record_seed or re.fullmatch(r"[0-9]+\.[0-9]+", str(provenance.get("python_major_minor"))) is None:
            failures.append(f"provenance:{identifier}")
        try:
            view = model_view(row)
            if tuple(view) != MODEL_VIEW_FIELDS:
                failures.append(f"model_view:{identifier}")
        except Exception:
            failures.append(f"model_view:{identifier}")
    for family in ("ERE", "CPS"):
        for split in SPLITS[family]:
            selected = [row for row in records if row["family"] == family and row["split"] == split]
            units: dict[int, int] = {}
            for row in selected:
                provenance = row["provenance"]
                units[int(provenance["unit_index"])] = int(provenance["generation_attempt"])
            expected_stats[f"{family}/{split}"] = {
                "unit_count": len(units),
                "attempts_total": sum(value + 1 for value in units.values()),
                "attempts_max": max((value + 1 for value in units.values()), default=0),
                "overlap_rejections": sum(units.values()),
                "overflow_rejections": 0,
            }
    if manifest.get("generation_stats") != expected_stats:
        failures.append("generation_stats")
    return _metric(
        not failures,
        {"record_count": len(records), "split_count": len(expected_files), "generation_stats": expected_stats},
        failures,
    )


def _audit_replay(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    claim_count = 0
    for row in records:
        identifier = str(row["example_id"])
        try:
            parsed = parse_source(row["source_text"])
            if parsed != {
                "family": row["family"],
                "template_id": row["template_id"],
                "program_ast": row["program_ast"],
                "label_mapping": row["label_mapping"],
            }:
                raise ValueError("source-only roundtrip differs")
            if render_source(row["family"], parsed["program_ast"], parsed["label_mapping"], parsed["template_id"]) != row["source_text"]:
                raise ValueError("canonical rerender differs")
            output = simulate_ere(parsed["program_ast"]) if row["family"] == "ERE" else evaluate_cps(parsed["program_ast"])
            if output != row["simulator_output"]:
                raise ValueError("fresh simulator output differs")
            trace = output["trace"] if row["family"] == "ERE" else output["candidates"]
            if trace != row["teacher_trace"]:
                raise ValueError("teacher trace differs")
            semantic = _semantic_answer(row["family"], output)
            label = _answer_label(row, semantic)
            if semantic != row["semantic_answer"] or LABELS[row["answer_index"]] != label:
                raise ValueError("answer/label binding differs")
            if semantic_fingerprint(parsed["program_ast"]) != row["semantic_fingerprint"]:
                raise ValueError("semantic fingerprint differs")
            if surface_fingerprint(row["source_text"]) != row["surface_fingerprint"]:
                raise ValueError("surface fingerprint differs")
            certificate = row["causal_certificate"]
            path = certificate["mutation_path"]
            if _get_path(row["program_ast"], path) != certificate["before_value"]:
                raise ValueError("counterfactual before value differs")
            alternate_ast = _patched(row["program_ast"], path, certificate["after_value"])
            alternate_output = simulate_ere(alternate_ast) if row["family"] == "ERE" else evaluate_cps(alternate_ast)
            alternate_answer = _semantic_answer(row["family"], alternate_output)
            if alternate_answer == semantic or alternate_answer != certificate["after_answer"]:
                raise ValueError("counterfactual did not replay to an answer flip")
            if certificate["before_answer"] != semantic:
                raise ValueError("counterfactual before answer differs")
            if certificate["before_output_sha256"] != _sha_value(output) or certificate["after_output_sha256"] != _sha_value(alternate_output):
                raise ValueError("counterfactual output hash differs")
            for claim in row["training_claims"]:
                claim_count += 1
                if verify_claim(row["program_ast"], row["family"], claim) is not claim["label"]:
                    raise ValueError(f"claim replay differs: {claim.get('claim_id')}")
        except Exception as exc:
            failures.append(f"{identifier}:{type(exc).__name__}:{exc}")
    return _metric(not failures, {"record_count": len(records), "claim_count": claim_count}, failures)


def _audit_pairs(records: Sequence[Mapping[str, Any]], spec: DatasetSpec) -> dict[str, Any]:
    failures: list[str] = []
    semantic = [row["semantic_fingerprint"] for row in records]
    surface = [row["surface_fingerprint"] for row in records]
    if len(set(semantic)) != len(semantic):
        failures.append("semantic_overlap")
    if len(set(surface)) != len(surface):
        failures.append("surface_overlap")
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row["pair_id"] is not None:
            pairs[str(row["pair_id"])].append(row)
        elif row["pair_role"] is not None:
            failures.append(f"orphan_pair_role:{row['example_id']}")
    family_counts = Counter(str(rows[0]["family"]) for rows in pairs.values() if rows)
    expected_pairs = spec.heldout_count // 2
    if family_counts != {"ERE": expected_pairs, "CPS": expected_pairs}:
        failures.append(f"pair_counts:{dict(family_counts)}")
    for pair_id, pair_rows in pairs.items():
        try:
            if len(pair_rows) != 2 or {row["pair_role"] for row in pair_rows} != {"base", "flip"}:
                raise ValueError("pair membership")
            base = next(row for row in pair_rows if row["pair_role"] == "base")
            flip = next(row for row in pair_rows if row["pair_role"] == "flip")
            paths = _diff_paths(base["program_ast"], flip["program_ast"])
            if paths != [base["causal_certificate"]["mutation_path"]]:
                raise ValueError(f"not one certified leaf: {paths}")
            if base["semantic_answer"] == flip["semantic_answer"] or base["answer_index"] == flip["answer_index"]:
                raise ValueError("answer did not flip")
            if base["template_id"] != flip["template_id"] or base["label_mapping"] != flip["label_mapping"] or base["record_seed"] != flip["record_seed"]:
                raise ValueError("surface factors differ")
            if base["causal_certificate"]["after_value"] != flip["causal_certificate"]["before_value"]:
                raise ValueError("reverse certificate differs")
        except Exception as exc:
            failures.append(f"{pair_id}:{type(exc).__name__}:{exc}")
    return _metric(not failures, {"semantic_unique": len(set(semantic)), "surface_unique": len(set(surface)), "pair_count": len(pairs)}, failures)


def _resolve(value: str, bindings: Mapping[str, str], neighbor: str | None) -> str:
    if value == "$neighbor":
        if neighbor is None:
            raise ValueError("neighbor outside FOREACH")
        return neighbor
    if value.startswith("$arg:"):
        return bindings[value[5:]]
    return value


def _ere_dependency_events(ast: Mapping[str, Any]) -> set[int]:
    initial = ast["initial_state"]
    attributes = deepcopy(initial["attributes"])
    attribute_origins = {(entity, attribute): set() for entity, attrs in attributes.items() for attribute in attrs}
    relations = {name: {tuple(pair) for pair in pairs} for name, pairs in initial["relations"].items()}
    relation_origins = {(name, source, target): set() for name, pairs in initial["relations"].items() for source, target in pairs}

    def predicate(item: Mapping[str, Any], bindings: Mapping[str, str], neighbor: str | None) -> tuple[bool, set[int]]:
        if item["kind"] == "attribute_equals":
            entity = _resolve(item["entity"], bindings, neighbor)
            attribute = _resolve(item["attribute"], bindings, neighbor)
            value = _resolve(item["value"], bindings, neighbor)
            return attributes.get(entity, {}).get(attribute) == value, set(attribute_origins.get((entity, attribute), set()))
        relation = _resolve(item["relation"], bindings, neighbor)
        source = _resolve(item["source"], bindings, neighbor)
        target = _resolve(item["target"], bindings, neighbor)
        return (source, target) in relations.get(relation, set()), set(relation_origins.get((relation, source, target), set()))

    def apply(item: Mapping[str, Any], bindings: Mapping[str, str], neighbor: str | None, event: int, controls: set[int]) -> None:
        op = item["op"]
        if op == "IF":
            result, deps = predicate(item["predicate"], bindings, neighbor)
            apply(item["then"] if result else item["else"], bindings, neighbor, event, controls | deps | {event})
            return
        if op == "FOREACH_LINKED":
            relation = _resolve(item["relation"], bindings, neighbor)
            source = _resolve(item["source"], bindings, neighbor)
            for target in sorted(right for left, right in relations.get(relation, set()) if left == source):
                edge = relation_origins.get((relation, source, target), set())
                apply(item["effect"], bindings, target, event, controls | set(edge) | {event})
            return
        resolved = {key: _resolve(value, bindings, neighbor) for key, value in item.items() if key != "op"}
        event_deps = controls | {event}
        if op == "SET":
            key = (resolved["target"], resolved["attribute"])
            attributes.setdefault(key[0], {})[key[1]] = resolved["value"]
            attribute_origins[key] = set(event_deps)
        elif op == "COPY":
            source = (resolved["source"], resolved["attribute"])
            target = (resolved["target"], resolved["attribute"])
            attributes.setdefault(target[0], {})[target[1]] = attributes[source[0]][source[1]]
            attribute_origins[target] = set(attribute_origins.get(source, set())) | event_deps
        elif op == "SWAP":
            left = (resolved["left"], resolved["attribute"])
            right = (resolved["right"], resolved["attribute"])
            left_value, right_value = attributes[left[0]][left[1]], attributes[right[0]][right[1]]
            left_origin, right_origin = set(attribute_origins.get(left, set())), set(attribute_origins.get(right, set()))
            attributes[left[0]][left[1]], attributes[right[0]][right[1]] = right_value, left_value
            attribute_origins[left], attribute_origins[right] = right_origin | event_deps, left_origin | event_deps
        else:
            key = (resolved["relation"], resolved["source"], resolved["target"])
            edge = (key[1], key[2])
            if op == "LINK":
                relations.setdefault(key[0], set()).add(edge)
                relation_origins[key] = set(event_deps)
            elif op == "UNLINK":
                relations.setdefault(key[0], set()).discard(edge)
                relation_origins.pop(key, None)
            else:
                raise ValueError(op)

    for event_index, event in enumerate(ast["events"]):
        rule = ast["rules"][event["rule"]]
        for primitive in rule["primitives"]:
            apply(primitive, event["arguments"], None, event_index, set())
    query = ast["query"]
    if query["kind"] == "attribute":
        return set(attribute_origins.get((query["entity"], query["attribute"]), set()))
    return set(relation_origins.get((query["relation"], query["source"], query["target"]), set()))


def _primitive_ops(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        if isinstance(value.get("op"), str):
            result.add(value["op"])
        for child in value.values():
            result.update(_primitive_ops(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_primitive_ops(child))
    return result


def _audit_ere(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    rows = [row for row in records if row["family"] == "ERE"]
    depth_by_id: dict[str, int] = {}
    all_ops: set[str] = set()
    for row in rows:
        try:
            state = row["program_ast"]["initial_state"]
            attributes = {attribute for entity in state["attributes"].values() for attribute in entity}
            declared_values = row["generation_metadata"].get("choice_values", [])
            state_values = {value for entity in state["attributes"].values() for value in entity.values()}
            if len(attributes) != 3 or len(declared_values) != 6 or set(declared_values) != state_values:
                raise ValueError(f"state domain differs: attributes={len(attributes)} values={len(state_values)}")
            deps = _ere_dependency_events(row["program_ast"])
            depth_by_id[row["example_id"]] = len(deps)
            all_ops.update(_primitive_ops(row["program_ast"]["rules"]))
        except Exception as exc:
            failures.append(f"dependency:{row['example_id']}:{type(exc).__name__}")
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        good = sum(depth_by_id.get(row["example_id"], 0) >= 3 for row in selected)
        if good / len(selected) < 0.90:
            failures.append(f"depth_quota:{split}:{good}/{len(selected)}")
        query_kinds = {row["program_ast"]["query"]["kind"] for row in selected}
        if query_kinds != {"attribute", "relation"}:
            failures.append(f"query_coverage:{split}")
    expected_ops = {"SET", "COPY", "SWAP", "LINK", "UNLINK", "IF", "FOREACH_LINKED"}
    if not expected_ops <= all_ops:
        failures.append(f"primitive_coverage:{sorted(expected_ops - all_ops)}")
    composition = [row for row in rows if row["split"] == "composition_ood"]
    topology_counts = Counter()
    for row in composition:
        ops = _primitive_ops(row["program_ast"]["rules"])
        if {"LINK", "FOREACH_LINKED", "COPY"} <= ops:
            topology_counts["link_foreach_copy"] += 1
        elif {"SWAP", "IF", "COPY"} <= ops:
            topology_counts["swap_if_copy"] += 1
        else:
            failures.append(f"composition_topology:{row['example_id']}")
    if set(topology_counts) != {"link_foreach_copy", "swap_if_copy"}:
        failures.append("composition_coverage")
    length_rows = [row for row in rows if row["split"] == "length_ood"]
    if not all(depth_by_id.get(row["example_id"], 0) >= 8 for row in length_rows):
        failures.append("length_depth")
    entity_rows = [row for row in rows if row["split"] == "entity_ood"]
    entity_counts = Counter(len(row["program_ast"]["initial_state"]["attributes"]) for row in entity_rows)
    if set(entity_counts) != {6, 8} or min(entity_counts.values(), default=0) < 30:
        failures.append(f"entity_quota:{dict(entity_counts)}")
    necessity = sum(row["causal_certificate"]["before_answer"] != row["causal_certificate"]["after_answer"] for row in rows)
    if necessity != len(rows):
        failures.append(f"necessity:{necessity}/{len(rows)}")
    return _metric(
        not failures,
        {
            "record_count": len(rows),
            "dependency_depth_histogram": dict(sorted(Counter(depth_by_id.values()).items())),
            "primitive_ops": sorted(all_ops),
            "composition_topologies": dict(topology_counts),
            "entity_counts": dict(entity_counts),
            "necessary_counterfactuals": necessity,
        },
        failures,
    )


def _winner_plan(output: Mapping[str, Any]) -> list[str] | None:
    answer = output["answer"]
    return None if answer is None else list(output["candidates"][answer]["plan"])


def _audit_cps(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    rows = [row for row in records if row["family"] == "CPS"]
    none_count = Counter()
    reason_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        output = evaluate_cps(row["program_ast"])
        split = row["split"]
        valid = [candidate for candidate in output["candidates"] if candidate["valid"]]
        for candidate in output["candidates"]:
            reason_by_split[split].update(candidate["failure_reasons"])
        if output["answer"] is None:
            none_count[split] += 1
            if valid or output["unique_optimum"] is not False:
                failures.append(f"none_semantics:{row['example_id']}")
        else:
            if output["unique_optimum"] is not True or len(valid) < 2:
                failures.append(f"optimum_or_suboptimal:{row['example_id']}")
            elif not any(candidate["total_cost"] > output["candidates"][output["answer"]]["total_cost"] for candidate in valid):
                failures.append(f"valid_suboptimal:{row['example_id']}")
    for split in SPLITS["CPS"]:
        selected = [row for row in rows if row["split"] == split]
        expected_none = (len(selected) + 5) // 6 if split != "causal_pairs" else 0
        if none_count[split] != expected_none:
            failures.append(f"none_ratio:{split}:{none_count[split]}/{len(selected)}")
        if len(reason_by_split[split]) < 2:
            failures.append(f"hard_negative_reasons:{split}:{sorted(reason_by_split[split])}")
    for row in (row for row in rows if row["split"] == "composition_ood"):
        ast = row["program_ast"]
        added = {effect["fact"] for action in ast["actions"] for effect in action["effects"] if effect["kind"] == "add_fact"}
        required = {pre["fact"] for action in ast["actions"] for pre in action["preconditions"] if pre["kind"] == "fact_true"}
        positive_resources = {(effect["resource"], effect["delta"]) for action in ast["actions"] for effect in action["effects"] if effect["kind"] == "resource_delta" and effect["delta"] > 0}
        required_resources = {(pre["resource"], pre["amount"]) for action in ast["actions"] for pre in action["preconditions"] if pre["kind"] == "resource_at_least"}
        final_facts = {condition["fact"] for condition in ast["final_constraints"] if condition["kind"] == "fact_true"}
        final_destroyers = {
            action["name"]
            for action in ast["actions"]
            if any(effect["kind"] == "remove_fact" and effect["fact"] in final_facts for effect in action["effects"])
        }
        action_costs = {action["name"]: action["cost"] for action in ast["actions"]}
        over_budget = any(sum(action_costs[name] for name in candidate["plan"]) > ast["budget"] for candidate in ast["candidates"])
        if not (added & required and positive_resources and required_resources and final_destroyers and over_budget):
            failures.append(f"composition_dependencies:{row['example_id']}")
    for row in (row for row in rows if row["split"] == "horizon_ood"):
        plan_length = max(len(candidate["plan"]) for candidate in row["program_ast"]["candidates"])
        if not 6 <= plan_length <= 8:
            failures.append(f"horizon:{row['example_id']}")
    for row in (row for row in rows if row["split"] == "distractor_ood"):
        ast = row["program_ast"]
        plan = set(_winner_plan(row["simulator_output"]) or [])
        if len(ast["candidates"]) != 8 or len(ast["actions"]) < 8 or not ({action["name"] for action in ast["actions"]} - plan):
            failures.append(f"distractor:{row['example_id']}")
    return _metric(
        not failures,
        {
            "record_count": len(rows),
            "none_counts": dict(none_count),
            "failure_reasons": {split: sorted(values) for split, values in sorted(reason_by_split.items())},
        },
        failures,
    )


def _audit_balance_permutations(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    balance: dict[str, Any] = {}
    for family in ("ERE", "CPS"):
        for split in SPLITS[family]:
            selected = [row for row in records if row["family"] == family and row["split"] == split]
            counts = Counter(LABELS[row["answer_index"]] for row in selected)
            deviation = max(abs(counts[label] / len(selected) - 1 / 9) for label in LABELS)
            limit = 1 / len(selected) + 1e-12
            balance[f"{family}/{split}"] = {"counts": dict(counts), "maximum_deviation": deviation, "limit": limit}
            if deviation > limit + 1e-12:
                failures.append(f"label_balance:{family}/{split}:{deviation}")
    permutation_count = 0
    for row in records:
        ast = row["program_ast"]
        original_fingerprint = semantic_fingerprint(ast)
        renamed = alpha_rename(ast)
        if semantic_fingerprint(renamed) != original_fingerprint:
            failures.append(f"alpha_invariance:{row['example_id']}")
        transformed = deepcopy(ast)
        if row["family"] == "CPS":
            transformed["actions"].reverse()
            transformed["candidates"].reverse()
            for action in transformed["actions"]:
                action["preconditions"].reverse()
            if semantic_fingerprint(transformed) != original_fingerprint:
                failures.append(f"cps_permutation_fingerprint:{row['example_id']}")
            if _winner_plan(evaluate_cps(transformed)) != _winner_plan(evaluate_cps(ast)):
                failures.append(f"cps_permutation_outcome:{row['example_id']}")
            sensitive = deepcopy(ast)
            sensitive["candidates"][0]["plan"].reverse()
        else:
            transformed["rules"] = dict(reversed(list(transformed["rules"].items())))
            transformed["initial_state"]["attributes"] = dict(reversed(list(transformed["initial_state"]["attributes"].items())))
            if semantic_fingerprint(transformed) != original_fingerprint or simulate_ere(transformed)["answer"] != simulate_ere(ast)["answer"]:
                failures.append(f"ere_permutation:{row['example_id']}")
            sensitive = deepcopy(ast)
            sensitive["events"].reverse()
        if sensitive != ast and semantic_fingerprint(sensitive) == original_fingerprint:
            failures.append(f"ordered_path_insensitive:{row['example_id']}")
        reversed_mapping = dict(reversed(list(row["label_mapping"].items())))
        rendered = render_source(row["family"], ast, reversed_mapping, row["template_id"])
        parsed = parse_source(rendered)
        if parsed["program_ast"] != ast or parsed["label_mapping"] != reversed_mapping:
            failures.append(f"choice_permutation:{row['example_id']}")
        permutation_count += 1
    return _metric(not failures, {"balance": balance, "permutation_count": permutation_count}, failures)


def _p99(values: Sequence[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)]


def _audit_language_length(records: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    failures: list[str] = []
    counts: dict[str, int] = {}
    for row in records:
        identifier = row["example_id"]
        expected_template = "indirect_v1" if row["split"] == "language_ood" else None
        if expected_template is not None and row["template_id"] != expected_template:
            failures.append(f"language_template:{identifier}")
        if expected_template is None and row["template_id"] not in TRAIN_TEMPLATES:
            failures.append(f"ordinary_template:{identifier}")
        lowered = row["source_text"].lower()
        if any(marker in lowered for marker in FORBIDDEN_MARKERS):
            failures.append(f"forbidden_marker:{identifier}")
        token_count = len(tokenizer.encode(row["source_text"], add_special_tokens=False))
        counts[identifier] = token_count
        if token_count != row["token_count"] or token_count > 1024:
            failures.append(f"token_count:{identifier}:{token_count}")
    p99_rows: dict[str, int] = {}
    for family in ("ERE", "CPS"):
        for split in ("train", "validation"):
            values = [counts[row["example_id"]] for row in records if row["family"] == family and row["split"] == split]
            percentile = _p99(values)
            p99_rows[f"{family}/{split}"] = percentile
            if percentile > 900:
                failures.append(f"p99:{family}/{split}:{percentile}")
    return _metric(not failures, {"maximum": max(counts.values()), "p99": p99_rows, "tokenizer_class": type(tokenizer).__name__}, failures)


def _label(row: Mapping[str, Any]) -> str:
    return LABELS[int(row["answer_index"])]


def _choice_count(row: Mapping[str, Any]) -> int:
    return sum(bool(item) for item in row["valid_choice_mask"])


def _choose(counts: Mapping[str, int], mask: Sequence[bool], labels: Sequence[str] = LABELS) -> str:
    active = [label for label, enabled in zip(labels, mask, strict=True) if enabled]
    return max(active, key=lambda label: (int(counts.get(label, 0)), -labels.index(label)))


def _last_choice(source: str) -> str:
    match = re.fullmatch(r"(?:(?P<plain>[A-I]) means|Label (?P<indirect>[A-I]) denotes) .+", source.splitlines()[-1])
    if not match:
        raise ValueError("last source line is not a choice")
    return match["plain"] or match["indirect"]


def _question_objective(source: str) -> str:
    selected: list[str] = []
    prefixes = (
        "What is ", "Does relation ", "Select the label denoting ", "Select the label stating ",
        "Goal: ", "Final constraints: ", "Budget: ", "Required goal: ", "Required final state: ",
        "Total cost may not exceed ",
    )
    for line in source.splitlines():
        if line.startswith(prefixes):
            selected.append(line)
    if not selected:
        raise ValueError("no visible question/objective text")
    return "\n".join(selected)


def _fold(group_id: str, fold_count: int = 5) -> int:
    return int(hashlib.sha256(f"p0d-v14-fold|{group_id}".encode("utf-8")).hexdigest(), 16) % fold_count


def _categorical_predictions(fit: Sequence[Mapping[str, Any]], evaluation: Sequence[Mapping[str, Any]], category: Callable[[Mapping[str, Any]], str]) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in fit:
        counts[category(row)][_label(row)] += 1
    return {row["example_id"]: _choose(counts[category(row)], row["valid_choice_mask"]) for row in evaluation}


_NB_SPECS: tuple[tuple[str, str, Callable[[Mapping[str, Any]], str]], ...] = (
    ("question_objective_word_nb", "word", lambda row: _question_objective(row["source_text"])),
    ("full_text_word_nb", "word", lambda row: row["source_text"]),
    ("full_text_char_3_5_nb", "char_3_5", lambda row: row["source_text"]),
)


def _sparse_equivalence(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    analyzer: str,
    text: Callable[[Mapping[str, Any]], str],
    bank: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = list(rows[:90])
    evaluation = list(rows[90:120])
    if len(evaluation) < 30:
        raise ValueError("sparse equivalence requires at least 120 rows")
    failures: list[str] = []
    fit_rows = [{"row_id": row["example_id"], "text": text(row), "label": _label(row)} for row in selected]
    dense = fit_scalable_multinomial_nb(fit_rows, analyzer, LABELS)
    sparse = fit_prepared_multinomial_nb([bank[row["example_id"]] for row in selected], LABELS)
    expected = [compact_predict_multinomial_nb(dense, text(row), row["valid_choice_mask"])["prediction"] for row in evaluation]
    actual_rows = predict_prepared_multinomial_nb_batch(
        sparse,
        [bank[row["example_id"]]["features"] for row in evaluation],
        [row["valid_choice_mask"] for row in evaluation],
    )
    for row, expected_label, actual in zip(evaluation, expected, actual_rows, strict=True):
        if actual["prediction"] != expected_label:
            failures.append(f"{name}:{row['example_id']}:{actual['prediction']}!={expected_label}")
    return {"passed": not failures, "prediction_count": len(evaluation), "failures": failures[:20]}


def _batch_prediction_map(
    model: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Mapping[str, Any]],
    diagnostics: dict[str, int],
) -> dict[str, str]:
    if not rows:
        return {}
    results = predict_prepared_multinomial_nb_batch(
        model,
        [bank[row["example_id"]]["features"] for row in rows],
        [row["valid_choice_mask"] for row in rows],
        diagnostics=diagnostics,
    )
    return {row["example_id"]: str(result["prediction"]) for row, result in zip(rows, results, strict=True)}


def _source_prediction_matrix(
    family_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, Any]]:
    emit = progress or (lambda _: None)
    train = [row for row in family_rows if row["split"] == "train"]
    raw_by_id = {row["example_id"]: row for row in family_rows}
    family = str(family_rows[0]["family"])
    predictions: dict[str, dict[str, dict[str, str]]] = {
        split: {name: {} for name in BASELINES}
        for split in SPLITS[family]
    }
    token_category = lambda row: str(len(tokenizer.encode(row["source_text"], add_special_tokens=False)) // 64)

    for heldout_fold in range(5):
        fit = [row for row in train if _fold(row["pair_id"] or row["example_id"]) != heldout_fold]
        evaluation = [row for row in train if _fold(row["pair_id"] or row["example_id"]) == heldout_fold]
        if not fit or not evaluation:
            raise ValueError(f"empty fold {heldout_fold}")
        predictions["train"]["majority"].update(_categorical_predictions(fit, evaluation, lambda row: "all"))
        predictions["train"]["token_length_bucket"].update(_categorical_predictions(fit, evaluation, token_category))

    for split in SPLITS[family]:
        evaluation = train if split == "train" else [row for row in family_rows if row["split"] == split]
        predictions[split]["first_valid_label"] = {
            row["example_id"]: next(label for label, enabled in zip(LABELS, row["valid_choice_mask"], strict=True) if enabled)
            for row in evaluation
        }
        predictions[split]["last_choice_line"] = {row["example_id"]: _last_choice(row["source_text"]) for row in evaluation}
        if split == "train":
            continue
        predictions[split]["majority"] = _categorical_predictions(train, evaluation, lambda row: "all")
        predictions[split]["token_length_bucket"] = _categorical_predictions(train, evaluation, token_category)

    equivalence_rows: dict[str, Any] = {}
    scorer_diagnostics: dict[str, int] = {"prediction_count": 0, "exact_fallback_count": 0, "batch_count": 0}
    progress_events: list[str] = []
    peak_feature_bank_count = 0
    for name, analyzer, text in _NB_SPECS:
        event = f"G09:{family}:{name}:prepare"
        emit(event)
        progress_events.append(event)
        prepared = prepare_nb_rows(
            family_rows,
            analyzer=analyzer,
            row_id=lambda row: str(row["example_id"]),
            text=text,
            label=_label,
        )
        bank = {row["row_id"]: row for row in prepared}
        del prepared
        peak_feature_bank_count = max(peak_feature_bank_count, 1)
        equivalence_rows[name] = _sparse_equivalence(train, name=name, analyzer=analyzer, text=text, bank=bank)
        for heldout_fold in range(5):
            fit = [row for row in train if _fold(row["pair_id"] or row["example_id"]) != heldout_fold]
            evaluation = [row for row in train if _fold(row["pair_id"] or row["example_id"]) == heldout_fold]
            model = fit_prepared_multinomial_nb([bank[row["example_id"]] for row in fit], LABELS)
            predictions["train"][name].update(_batch_prediction_map(model, evaluation, bank, scorer_diagnostics))
            del model
        model = fit_prepared_multinomial_nb([bank[row["example_id"]] for row in train], LABELS)
        for split in SPLITS[family]:
            if split == "train":
                continue
            evaluation = [row for row in family_rows if row["split"] == split]
            predictions[split][name] = _batch_prediction_map(model, evaluation, bank, scorer_diagnostics)
        del model, bank
        gc.collect()
        event = f"G09:{family}:{name}:done"
        emit(event)
        progress_events.append(event)
    equivalence = {
        "passed": all(row["passed"] for row in equivalence_rows.values()),
        "prediction_count": sum(row["prediction_count"] for row in equivalence_rows.values()),
        "analyzers": equivalence_rows,
        "failures": [failure for row in equivalence_rows.values() for failure in row["failures"]][:20],
    }
    return predictions, {
        "sparse_dense_equivalence": equivalence,
        "row_count": len(raw_by_id),
        "peak_feature_bank_count": peak_feature_bank_count,
        "progress_events": progress_events,
        "scorer_diagnostics": scorer_diagnostics,
    }


def _audit_shortcuts(
    records: Sequence[Mapping[str, Any]], tokenizer: Any, progress: Callable[[str], None] | None = None
) -> dict[str, Any]:
    failures: list[str] = []
    equivalence: dict[str, Any] = {}
    successes: dict[str, int] = {}
    totals: dict[str, int] = {}
    random_mass: dict[str, float] = {}
    complete: dict[str, bool] = {}
    prediction_hashes: dict[str, str] = {}
    aggregate_projections: dict[str, dict[str, str]] = defaultdict(dict)
    for family in ("ERE", "CPS"):
        family_rows = [row for row in records if row["family"] == family]
        split_predictions, qualification = _source_prediction_matrix(family_rows, tokenizer, progress)
        equivalence[family] = qualification
        if (
            not qualification["sparse_dense_equivalence"]["passed"]
            or qualification["peak_feature_bank_count"] != 1
            or len(qualification["progress_events"]) != 6
        ):
            failures.append(f"sparse_dense_equivalence:{family}")
        for split in SPLITS[family]:
            evaluation = [row for row in family_rows if row["split"] == split]
            for name, prediction in sorted(split_predictions[split].items()):
                key = f"{family}/{split}/{name}"
                successes[key] = sum(prediction.get(row["example_id"]) == _label(row) for row in evaluation)
                totals[key] = len(evaluation)
                random_mass[key] = math.fsum(1 / _choice_count(row) for row in evaluation)
                complete[key] = len(prediction) == len(evaluation) and set(prediction) == {row["example_id"] for row in evaluation}
                prediction_hashes[key] = _sha_value(dict(sorted(prediction.items())))
                if split != "train":
                    aggregate_projections[f"{family}/{name}"].update(prediction)
    decision = evaluate_g09_counts(successes=successes, totals=totals, random_mass=random_mass, complete=complete)
    failures.extend(decision["failures"])
    cells = decision["cells"]
    aggregates = decision["family_aggregates"]
    for key, value in cells.items():
        value["prediction_sha256"] = prediction_hashes[key]
    for key, value in aggregates.items():
        value["prediction_sha256"] = _sha_value(dict(sorted(aggregate_projections[key].items())))
    return _metric(
        not failures,
        {"cells": cells, "family_aggregates": aggregates, "equivalence": equivalence, "decision_topology": decision["topology"]},
        failures,
    )


def _audit_claims(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    balance: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    for family in ("ERE", "CPS"):
        family_rows = [row for row in records if row["family"] == family]
        claims_by_split: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS[family]:
            claims = [dict(claim) for row in family_rows if row["split"] == split for claim in row["training_claims"]]
            claims_by_split[split] = claims
            by_kind: dict[str, Counter[bool]] = defaultdict(Counter)
            by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for claim in claims:
                by_kind[claim["kind"]][bool(claim["label"])] += 1
                by_pair[claim["pair_id"]].append(claim)
            bad_pairs = [pair_id for pair_id, pair in by_pair.items() if len(pair) != 2 or {bool(item["label"]) for item in pair} != {True, False}]
            balanced = all(counts[True] == counts[False] and counts[True] > 0 for counts in by_kind.values())
            balance[f"{family}/{split}"] = {
                "claim_count": len(claims),
                "by_kind": {kind: {str(key): value for key, value in counts.items()} for kind, counts in by_kind.items()},
            }
            if not balanced or bad_pairs:
                failures.append(f"claim_balance:{family}/{split}:{bad_pairs[:3]}")
        for analyzer in ("word", "char_3_5"):
            prepared_train = prepare_nb_rows(
                claims_by_split["train"],
                analyzer=analyzer,
                row_id=lambda claim: claim["claim_id"],
                text=lambda claim: claim["text"],
                label=lambda claim: "P" if claim["label"] else "N",
            )
            model = fit_prepared_multinomial_nb(prepared_train, ("P", "N"))
            del prepared_train
            for split in SPLITS[family]:
                if split == "train":
                    continue
                evaluation = claims_by_split[split]
                prepared_evaluation = prepare_nb_rows(
                    evaluation,
                    analyzer=analyzer,
                    row_id=lambda claim: claim["claim_id"],
                    text=lambda claim: claim["text"],
                    label=lambda claim: "P" if claim["label"] else "N",
                )
                results = predict_prepared_multinomial_nb_batch(
                    model,
                    [row["features"] for row in prepared_evaluation],
                    [[True, True] for _ in evaluation],
                )
                del prepared_evaluation
                predictions = {
                    claim["claim_id"]: str(result["prediction"])
                    for claim, result in zip(evaluation, results, strict=True)
                }
                successes = sum(predictions[claim["claim_id"]] == ("P" if claim["label"] else "N") for claim in evaluation)
                bound = _wilson_upper(successes, len(evaluation), CLAIM_CELL_COUNT)
                passed = bound["accuracy"] <= 0.60 + 1e-12 and bound["upper"] <= 0.60 + 1e-12
                key = f"{family}/{split}/{analyzer}"
                cells[key] = {**bound, "gate_threshold": 0.60, "prediction_sha256": _sha_value(dict(sorted(predictions.items()))), "passed": passed}
                if not passed:
                    failures.append(f"claim_nb:{key}:accuracy={bound['accuracy']:.6f}:upper={bound['upper']:.6f}")
            del model
            gc.collect()
    if len(cells) != CLAIM_CELL_COUNT:
        failures.append(f"claim_cell_count:{len(cells)}")
    return _metric(not failures, {"balance": balance, "cells": cells}, failures)


def audit_production_dataset(
    repo_root: str | Path,
    dataset_root: str | Path,
    *,
    spec: DatasetSpec,
    tokenizer: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = Path(dataset_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dataset = load_production_dataset(root)
    records = _flatten(dataset)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    emit = progress or (lambda _: None)
    operations: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("G01", lambda: verify_prerequisites(repo, require_qualification=spec.profile == "formal")),
        ("G02", lambda: _audit_schema(repo, root, manifest, records, spec)),
        ("G03", lambda: _audit_replay(records)),
        ("G04", lambda: _audit_pairs(records, spec)),
        ("G05", lambda: _audit_ere(records)),
        ("G06", lambda: _audit_cps(records)),
        ("G07", lambda: _audit_balance_permutations(records)),
        ("G08", lambda: _audit_language_length(records, tokenizer)),
        ("G09", lambda: _audit_shortcuts(records, tokenizer, progress)),
        ("G10", lambda: _audit_claims(records)),
    )
    metrics: dict[str, Any] = {}
    for gate, operation in operations:
        emit(f"audit:{gate}:start")
        metrics[gate] = operation()
        emit(f"audit:{gate}:done")
    gates = {gate: result["passed"] for gate, result in metrics.items()}
    passed = all(gates.values())
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v17.generator-report.v1",
        "contract_version": GENERATOR_VERSION,
        "dataset_manifest_sha256": _sha(root / "manifest.json"),
        "gates": gates,
        "metrics": metrics,
        "passed": passed,
        "status": "PASS_P0D_PRODUCTION" if passed else "FAIL_P0D_PRODUCTION",
        "authorization_created": False,
    }


__all__ = ["audit_production_dataset", "verify_prerequisites"]
