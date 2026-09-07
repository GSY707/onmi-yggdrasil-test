from __future__ import annotations

"""Fail-closed D001--D011 audit for the C0 successor data package.

The old v17 production audit is deliberately treated as an immutable substrate.
It is evidence about the shared production machinery, not evidence that the new
generator identity or its relation cells are qualified.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any, Mapping, Sequence


MIN_SUPPORT = 128
MIN_CLASS_SUPPORT = 64
MAX_DOMINANT_MASS = 0.80
MAX_ORACLE_EXCESS = 0.10
_LEGEND = re.compile(r"(?:\bLabel\s+)?([A-I])\s+(?:means|denotes)\s+(TRUE|FALSE)\b", re.I)


def _sha_value(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _metric(passed: bool, value: Any, failures: Sequence[str] = ()) -> dict[str, Any]:
    return {"passed": bool(passed), "value": value, "failures": list(failures)[:100]}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_and_manifest(dataset_root: Path, generator: Any, records: Sequence[Mapping[str, Any]] | None) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None]:
    manifest_path = dataset_root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else None
    if records is None:
        loader = getattr(generator, "load", None)
        if loader is None:
            raise AttributeError("data_generator.load")
        loaded = loader(dataset_root)
        if isinstance(loaded, Mapping):
            # The successor loader returns {"ERE/train": [rows], ...}; cell
            # order is not evidence, so flatten in sorted-key order.
            records = [row for cell in sorted(loaded) for row in loaded[cell]]
        elif isinstance(loaded, tuple) and len(loaded) == 2:
            records, loaded_manifest = loaded
            if manifest is None and isinstance(loaded_manifest, Mapping):
                manifest = loaded_manifest
        else:
            records = loaded
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("successor dataset must be a sequence of records")
    return list(records), manifest


def _d001(repo_root: Path, generator: Any) -> dict[str, Any]:
    failures: list[str] = []
    substrate: dict[str, Any] = {}
    try:
        from yggdrasil_v2.v2_a.closure_c0.contract import P0D
        from yggdrasil_v2.v2_a.closure_c0.audit import verify_evidence_seal
        old_root = repo_root / P0D["root"]
        seal = verify_evidence_seal(old_root, expected_seal_sha256=P0D["seal_sha256"])
        substrate["old_v17_seal"] = seal
        if seal.get("passed") is not True:
            failures.append("old_v17_seal")
        from yggdrasil_v2.r1_revalidation.production.generator_audit import verify_prerequisites
        substrate = verify_prerequisites(repo_root, require_qualification=True)
        substrate["old_v17_seal"] = seal
        if substrate.get("passed") is not True:
            failures.extend(substrate.get("failures", []))
    except Exception as exc:  # fail closed, including absent old substrate
        failures.append(f"exception:{type(exc).__name__}:{exc}")
    return _metric(not failures, {"immutable_substrate": True, "audit": substrate}, failures)


def _d002(
    repo_root: Path,
    dataset_root: Path,
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any] | None,
    generator: Any,
) -> dict[str, Any]:
    from yggdrasil_v2.v2_a.closure_c0r import contract

    failures: list[str] = []
    expected_schema = getattr(generator, "SCHEMA_VERSION", None)
    expected_manifest_schema = getattr(generator, "MANIFEST_SCHEMA", None)
    expected_generator = getattr(generator, "GENERATOR_VERSION", None)
    if not isinstance(manifest, Mapping):
        failures.append("manifest_missing")
        manifest = {}
    for key, expected in (
        ("schema_version", expected_manifest_schema),
        ("generator_version", expected_generator),
        ("model_id", contract.MODEL_ID),
        ("model_revision", contract.MODEL_REVISION),
        ("add_special_tokens", False),
    ):
        if expected is not None and (
            key not in manifest
            or type(manifest.get(key)) is not type(expected)
            or manifest.get(key) != expected
        ):
            failures.append(f"manifest:{key}")
    spec = getattr(generator, "FORMAL_SPEC", None)
    if (
        spec is None
        or manifest.get("profile") != spec.profile
        or manifest.get("root_seed") != spec.root_seed
        or manifest.get("seed_derivation_version") != getattr(generator, "SEED_DERIVATION_VERSION", None)
    ):
        failures.append("manifest:identity")
    design_path = repo_root / "docs/v2-a-closure-c0r-data-trace-qualification.md"
    if not design_path.is_file() or manifest.get("design_sha256") != _sha_file(design_path):
        failures.append("manifest:design_sha256")
    splits = getattr(generator, "SPLITS", {})
    expected_cells = {f"{family}/{split}" for family, values in splits.items() for split in values}
    actual_counts: Counter[str] = Counter(f"{row.get('family')}/{row.get('split')}" for row in records)
    if set(actual_counts) != expected_cells:
        failures.append("cells")
    manifest_counts = manifest.get("counts")
    if not isinstance(manifest_counts, Mapping) or {
        str(k): int(v) for k, v in manifest_counts.items()
    } != dict(actual_counts):
        failures.append("manifest:counts")
    if spec is not None:
        expected_counts = {
            f"{family}/{split}": spec.count(split)
            for family, values in splits.items()
            for split in values
        }
        if dict(actual_counts) != expected_counts:
            failures.append("profile:counts")
    expected_files = {f"{family.lower()}-{split}.jsonl" for family, split in (cell.split("/", 1) for cell in expected_cells)}
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, Mapping) or set(manifest_files) != expected_files or any(
        not isinstance(v, str) or len(v) != 64 for v in manifest_files.values()
    ):
        failures.append("manifest:files")
    elif any(
        not (dataset_root / relative).is_file()
        or _sha_file(dataset_root / relative) != str(expected).upper()
        for relative, expected in manifest_files.items()
    ):
        failures.append("manifest:file_hash")
    ids = [row.get("example_id") for row in records]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        failures.append("unique_example_ids")
    view = getattr(generator, "public_forward_view", None)
    view_hashes: dict[str, str] = {}
    from yggdrasil_v2.r1_revalidation.production import generator as substrate
    from yggdrasil_v2.r1_revalidation.production import generator_audit as legacy_audit

    expected_fields = set(legacy_audit.RECORD_FIELDS)
    generation_units: dict[str, dict[int, int]] = defaultdict(dict)
    for row in records:
        if set(row) != expected_fields:
            failures.append(f"row_fields:{row.get('example_id')}")
        if row.get("schema_version") != expected_schema:
            failures.append(f"row_schema:{row.get('example_id')}")
        if row.get("generator_version") != expected_generator:
            failures.append(f"row_generator:{row.get('example_id')}")
        if f"{row.get('family')}/{row.get('split')}" not in expected_cells:
            failures.append(f"row_cell:{row.get('example_id')}")
        provenance = row.get("provenance")
        family = row.get("family")
        split = row.get("split")
        family_index = ("ERE", "CPS").index(family) if family in {"ERE", "CPS"} else -1
        split_values = splits.get(family, ())
        split_index = split_values.index(split) if split in split_values else -1
        unit = provenance.get("unit_index") if isinstance(provenance, Mapping) else None
        attempt = provenance.get("generation_attempt") if isinstance(provenance, Mapping) else None
        split_seed = (
            spec.root_seed
            + family_index * substrate._FAMILY_SEED_STRIDE
            + split_index * substrate._SPLIT_SEED_STRIDE
            if spec is not None and family_index >= 0 and split_index >= 0
            else None
        )
        expected_record_seed = (
            split_seed
            + unit * substrate._UNIT_SEED_STRIDE
            + attempt * substrate._ATTEMPT_SEED_STRIDE
            if isinstance(split_seed, int)
            and type(unit) is int
            and type(attempt) is int
            else None
        )
        expected_provenance = {
            "generator_version": expected_generator,
            "root_seed": manifest.get("root_seed"),
            "family_index": family_index,
            "split_index": split_index,
            "split_seed": split_seed,
            "unit_index": unit,
            "generation_attempt": attempt,
            "record_seed": row.get("record_seed"),
            "seed_derivation_version": manifest.get("seed_derivation_version"),
            "design_sha256": manifest.get("design_sha256"),
            "python_major_minor": provenance.get("python_major_minor")
            if isinstance(provenance, Mapping)
            else None,
        }
        if (
            not isinstance(provenance, Mapping)
            or dict(provenance) != expected_provenance
            or row.get("record_seed") != expected_record_seed
            or type(unit) is not int
            or type(attempt) is not int
            or unit < 0
            or attempt < 0
            or not re.fullmatch(r"[0-9]+\.[0-9]+", str(expected_provenance["python_major_minor"]))
        ):
            failures.append(f"row_provenance:{row.get('example_id')}")
        elif family_index >= 0 and split_index >= 0:
            generation_units[f"{family}/{split}"][unit] = attempt
        try:
            public = view(row) if view else {"source_text": row.get("source_text")}
            if public != {"source_text": row.get("source_text")}:
                failures.append(f"public_view:{row.get('example_id')}")
            view_hashes[str(row.get("example_id"))] = _sha_value(public)
        except Exception as exc:
            failures.append(f"public_view_exception:{row.get('example_id')}:{type(exc).__name__}")
    if manifest.get("total_records") != len(records) or manifest.get("total_records") != sum(actual_counts.values()):
        failures.append("manifest:total_records")
    expected_stats = {
        cell: {
            "unit_count": len(units),
            "attempts_total": sum(value + 1 for value in units.values()),
            "attempts_max": max((value + 1 for value in units.values()), default=0),
            "overlap_rejections": sum(units.values()),
            "overflow_rejections": 0,
        }
        for cell, units in sorted(generation_units.items())
    }
    if manifest.get("generation_stats") != expected_stats:
        failures.append("manifest:generation_stats")
    return _metric(
        not failures,
        {
            "record_count": len(records),
            "unique_ids": len(set(ids)),
            "cell_counts": dict(sorted(actual_counts.items())),
            "generation_stats": expected_stats,
            "public_view_prediction_sha256": _sha_value(view_hashes),
            "immutable_substrate": False,
        },
        failures,
    )


def _call_legacy(name: str, records: Sequence[Mapping[str, Any]], *, spec: Any = None, tokenizer: Any = None) -> dict[str, Any]:
    try:
        from yggdrasil_v2.r1_revalidation.production import generator_audit as legacy
        fn = getattr(legacy, name)
        if name == "_audit_pairs":
            result = fn(records, spec)
        elif name in {"_audit_language_length", "_audit_shortcuts"}:
            result = fn(records, tokenizer)
        else:
            result = fn(records)
        if not isinstance(result, Mapping):
            raise TypeError("legacy audit returned non-object")
        return {"passed": result.get("passed") is True, "legacy_substrate": True, "audit": result, "failures": list(result.get("failures", []))}
    except Exception as exc:
        return {"passed": False, "legacy_substrate": True, "audit": {}, "failures": [f"exception:{name}:{type(exc).__name__}:{exc}"]}


def _wilson_upper(correct: int, total: int, alpha: float) -> float:
    if total <= 0:
        return 1.0
    z = NormalDist().inv_cdf(1.0 - alpha)
    p = correct / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (centre + spread) / denominator


def relation_visible_pattern_gate(records: Sequence[Mapping[str, Any]], *, min_support: int = MIN_SUPPORT, min_class_support: int = MIN_CLASS_SUPPORT, max_mass: float = MAX_DOMINANT_MASS, max_excess: float = MAX_ORACLE_EXCESS, expected_causal_pairs: int = 64) -> dict[str, Any]:
    """Audit only ERE relation cells; causal pairs additionally need one TRUE/one FALSE per pair."""
    failures: list[str] = []
    cells: dict[str, Any] = {}
    relation_rows = [row for row in records if row.get("family") == "ERE" and isinstance(row.get("program_ast"), Mapping) and row["program_ast"].get("query", {}).get("kind") == "relation"]
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        by_cell[f"ERE/{row.get('split')}"] .append(row)
    for cell, rows in sorted(by_cell.items()):
        answers = [str(row.get("semantic_answer")).upper() for row in rows]
        counts = Counter(answers)
        dominant = max(counts.values(), default=0) / len(rows) if rows else 0.0
        legend_predictions: dict[str, str] = {}
        correct = 0
        for row in rows:
            labels = {label: meaning.upper() for label, meaning in _LEGEND.findall(str(row.get("source_text", "")))}
            true_labels = [label for label, meaning in labels.items() if meaning == "TRUE"]
            prediction = true_labels[0] if len(true_labels) == 1 else None
            legend_predictions[str(row.get("example_id"))] = prediction or "<invalid>"
            actual = chr(ord("A") + row["answer_index"]) if isinstance(row.get("answer_index"), int) else None
            correct += int(prediction is not None and prediction == actual)
        total = len(rows)
        tested = total >= min_support
        # The predictor is intentionally a function of source_text only.  The
        # answer index is the independent target; label_mapping is never used.
        visible_choice_counts = [len({label for label, _ in _LEGEND.findall(str(row.get("source_text", "")))}) for row in rows]
        chance = sum(1.0 / count if count else 0.5 for count in visible_choice_counts) / total if total else 0.5
        upper = 1.0
        passed = not tested or (dominant <= max_mass and min(counts.values(), default=0) >= min_class_support)
        if not passed:
            failures.append(f"relation_visible_pattern:{cell}")
        cells[cell] = {"support": total, "answer_counts": dict(counts), "dominant_answer_mass": dominant, "minimum_each_class_support": min_class_support, "maximum_allowed_mass": max_mass, "oracle_correct": correct, "oracle_accuracy": correct / total if total else None, "chance": chance, "input_fields": ["source_text"], "prediction_sha256": _sha_value(legend_predictions), "tested": tested, "passed": passed}
    pair_failures: list[str] = []
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in relation_rows:
        if row.get("split") == "causal_pairs" and row.get("pair_id") is not None:
            pairs[str(row["pair_id"])].append(row)
    for pair_id, rows in pairs.items():
        values = {str(row.get("semantic_answer")).upper() for row in rows}
        if len(rows) != 2 or values != {"TRUE", "FALSE"}:
            pair_failures.append(pair_id)
    tested_cells = sum(bool(cell["tested"]) for cell in cells.values())
    alpha = 0.05 / tested_cells if tested_cells else 0.05
    for cell, report in cells.items():
        if report["tested"]:
            report["one_sided_wilson_upper"] = _wilson_upper(report["oracle_correct"], report["support"], alpha)
            report["bonferroni_alpha"] = alpha
            report["passed"] = report["passed"] and report["one_sided_wilson_upper"] <= report["chance"] + max_excess
            if not report["passed"] and f"relation_visible_pattern:{cell}" not in failures:
                failures.append(f"relation_visible_pattern:{cell}")
    if len(pairs) != expected_causal_pairs:
        failures.append(f"causal_relation_pair_count:{len(pairs)}:{expected_causal_pairs}")
    if pair_failures:
        failures.append("causal_relation_pairs:" + ",".join(pair_failures[:20]))
    return _metric(not failures, {"cells": cells, "causal_pair_count": len(pairs), "expected_causal_pairs": expected_causal_pairs, "causal_pair_failures": pair_failures, "tested_cell_count": tested_cells, "bonferroni_alpha": alpha, "thresholds": {"min_support": min_support, "min_class_support": min_class_support, "max_mass": max_mass, "max_excess": max_excess}}, failures)


def audit_successor_dataset(repo_root: Path, dataset_root: Path, generator: Any, *, records: Sequence[Mapping[str, Any]] | None = None, tokenizer: Any = None) -> dict[str, Any]:
    """Return JSON-native D001--D011 audit; every unavailable dependency fails closed."""
    try:
        loaded, manifest = _records_and_manifest(dataset_root, generator, records)
    except Exception as exc:
        failure = f"load:{type(exc).__name__}:{exc}"
        return {"schema_version": "yggdrasil.v2-a.closure-c0r.data-audit.v1", "availability": False, "available": False, "passed": False, "gates": {f"D{i:03d}": False for i in range(1, 12)}, "failures": [failure]}
    d001 = _d001(repo_root, generator)
    d002 = _d002(repo_root, dataset_root, loaded, manifest, generator)
    try:
        spec = getattr(generator, "FORMAL_SPEC")
        legacy = {"D%03d" % (index + 3): _call_legacy(name, loaded, spec=spec, tokenizer=tokenizer) for index, name in enumerate(("_audit_replay", "_audit_pairs", "_audit_ere", "_audit_cps", "_audit_balance_permutations", "_audit_language_length", "_audit_shortcuts", "_audit_claims"))}
    except Exception as exc:
        legacy = {f"D{index:03d}": {"passed": False, "legacy_substrate": True, "failures": [f"setup:{type(exc).__name__}:{exc}"]} for index in range(3, 11)}
    d011 = relation_visible_pattern_gate(loaded)
    gates = {"D001": d001["passed"], "D002": d002["passed"], **{key: value["passed"] for key, value in legacy.items()}, "D011": d011["passed"]}
    audits = {"D001": d001, "D002": d002, **legacy, "D011": d011}
    failures = [f"{key}:{failure}" for key, value in audits.items() for failure in value.get("failures", [])]
    return {"schema_version": "yggdrasil.v2-a.closure-c0r.data-audit.v1", "identity": getattr(generator, "GENERATOR_VERSION", None), "availability": True, "available": True, "gates": gates, "passed": all(gates.values()), "failures": failures, "immutable_substrate_boundary": "D001 and D003-D010 legacy results are substrate evidence; they do not qualify the new generator identity by themselves.", "audits": audits}


__all__ = ["audit_successor_dataset", "relation_visible_pattern_gate"]
