from __future__ import annotations

"""Deterministic v17 qualification for the two v16 production defects."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping

from ..common import simulate_ere
from .fingerprint import RESERVED_VALUES, SCHEMA_KEYS, alpha_rename, semantic_fingerprint
from .generator import (
    SPLITS,
    _ATTEMPT_SEED_STRIDE,
    _SPLIT_SEED_STRIDE,
    _UNIT_SEED_STRIDE,
    _ere_if_chain,
    build_ere_pair,
)


QUALIFICATION_SEED = 2026081701
CAPACITY_SEED = 2026081791
IF_CHAIN_SWEEP_COUNT = 50_000
EXPECTED_V16_VISIBLE_FAILURES = 307
V16_RUNTIME_ROOT = Path("artifacts/v2-r1r/p0d-v16-runtime-qualification-20260810-1")
V16_FORMAL_ROOT = Path("artifacts/v2-r1r/p0d-v16-full-production-20260810-1")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "evidence-seal.json"
    }


def verify_v16_runtime_evidence(repo_root: Path) -> dict[str, Any]:
    root = repo_root / V16_RUNTIME_ROOT
    assessment_path = root / "assessment.json"
    seal_path = root / "evidence-seal.json"
    if not assessment_path.is_file() or not seal_path.is_file():
        return {"root": V16_RUNTIME_ROOT.as_posix(), "passed": False, "failure": "missing"}
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    actual = _tree_hashes(root)
    passed = (
        assessment.get("status") == "PASS_RUNTIME_QUALIFICATION"
        and assessment.get("passed") is True
        and all(assessment.get("gates", {}).values())
        and actual == seal.get("files")
    )
    return {
        "root": V16_RUNTIME_ROOT.as_posix(),
        "assessment_sha256": _sha(assessment_path),
        "evidence_seal_sha256": _sha(seal_path),
        "sealed_file_count": len(actual),
        "gates": assessment.get("gates"),
        "passed": passed,
        "failure": None if passed else "assessment_or_seal",
    }


def _named_ast_hashes(path: Path, names: set[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, str] = {}
    for node in tree.body:
        name: str | None = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            identifiers = [target.id for target in targets if isinstance(target, ast.Name)]
            name = identifiers[0] if len(identifiers) == 1 else None
        if name in names:
            found[name] = _sha_bytes(ast.dump(node, include_attributes=False).encode("utf-8"))
    missing = names - found.keys()
    if missing:
        raise ValueError(f"protected symbols missing from {path}: {sorted(missing)}")
    return dict(sorted(found.items()))


def verify_runtime_surface_unchanged(repo_root: Path) -> dict[str, Any]:
    snapshot = repo_root / V16_RUNTIME_ROOT / "source_snapshot"
    exact_files = (
        "src/yggdrasil_v2/r1_revalidation/production/runtime_qualification.py",
        "src/yggdrasil_v2/r1_revalidation/production/p0_shortcut.py",
        "src/yggdrasil_v2/r1_revalidation/production/g09_decision.py",
        "src/yggdrasil_v2/r1_revalidation/production/scalable.py",
    )
    exact: dict[str, Any] = {}
    for relative in exact_files:
        prior = snapshot / relative
        current = repo_root / relative
        prior_sha = _sha(prior) if prior.is_file() else None
        current_sha = _sha(current) if current.is_file() else None
        exact[relative] = {"prior": prior_sha, "current": current_sha, "passed": prior_sha == current_sha and prior_sha is not None}

    symbols = {
        "src/yggdrasil_v2/r1_revalidation/production/generator_audit.py": {
            "_NB_SPECS", "_flatten", "_audit_shortcuts", "_audit_claims"
        },
        "src/yggdrasil_v2/r1_revalidation/production/generator.py": {
            "MODEL_ID", "MODEL_REVISION", "load_production_dataset"
        },
    }
    protected: dict[str, Any] = {}
    for relative, names in symbols.items():
        prior_hashes = _named_ast_hashes(snapshot / relative, names)
        current_hashes = _named_ast_hashes(repo_root / relative, names)
        protected[relative] = {
            "prior": prior_hashes,
            "current": current_hashes,
            "passed": prior_hashes == current_hashes,
        }
    passed = all(row["passed"] for row in exact.values()) and all(row["passed"] for row in protected.values())
    return {"exact_files": exact, "protected_symbols": protected, "passed": passed}


def _visible_domain(pair: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> bool:
    ast, alternate, metadata = pair
    declared = metadata.get("choice_values", [])
    if len(declared) != 6 or len(set(declared)) != 6:
        return False
    for candidate in (ast, alternate):
        visible = {
            value
            for attributes in candidate["initial_state"]["attributes"].values()
            for value in attributes.values()
        }
        if visible != set(declared):
            return False
    return True


def run_visible_domain_sweep() -> dict[str, Any]:
    pattern_counts: dict[str, int] = {}
    for index in range(IF_CHAIN_SWEEP_COUNT):
        pair = _ere_if_chain(random.Random(QUALIFICATION_SEED + index * _ATTEMPT_SEED_STRIDE), (4, 6, 8)[index % 3])
        if not _visible_domain(pair):
            return {"passed": False, "failure": f"if_chain:{index}"}
        if simulate_ere(pair[0])["answer"] == simulate_ere(pair[1])["answer"]:
            return {"passed": False, "failure": f"counterfactual:{index}"}
        pattern = pair[2]["pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    seen: set[str] = set()
    split_counts: dict[str, int] = {}
    for split_index, split in enumerate(SPLITS["ERE"]):
        count = 4096 if split == "train" else 1536
        units = count // 2 if split == "causal_pairs" else count
        split_seed = CAPACITY_SEED + split_index * _SPLIT_SEED_STRIDE
        for index in range(units):
            for attempt in range(100):
                record_seed = split_seed + index * _UNIT_SEED_STRIDE + attempt * _ATTEMPT_SEED_STRIDE
                pair = build_ere_pair(record_seed, "train" if split == "causal_pairs" else split, index)
                fingerprints = [semantic_fingerprint(pair[0]), semantic_fingerprint(pair[1])] if split == "causal_pairs" else [semantic_fingerprint(pair[0])]
                if len(set(fingerprints)) == len(fingerprints) and not any(value in seen for value in fingerprints):
                    seen.update(fingerprints)
                    break
            else:
                return {"passed": False, "failure": f"capacity:{split}:{index}"}
        split_counts[split] = count
    expected = 4096 + 6 * 1536
    return {
        "if_chain_sweep_count": IF_CHAIN_SWEEP_COUNT,
        "pattern_counts": pattern_counts,
        "capacity_seed": CAPACITY_SEED,
        "split_counts": split_counts,
        "unique_semantic_fingerprints": len(seen),
        "expected_semantic_fingerprints": expected,
        "passed": len(seen) == expected,
    }


def _collision_ast(symbol: str) -> dict[str, Any]:
    return {
        "initial_state": {
            "attributes": {symbol: {symbol: symbol}},
            "relations": {symbol: [[symbol, symbol]]},
            "facts": [symbol],
            "resources": {symbol: 2},
        },
        "rules": {
            symbol: {
                "params": [symbol],
                "primitives": [{"op": "SET", "target": "$arg:" + symbol, "attribute": symbol, "value": symbol}],
            }
        },
        "events": [{"rule": symbol, "arguments": {symbol: symbol}}],
        "query": {"kind": "attribute", "entity": symbol, "attribute": symbol},
        "actions": [
            {
                "name": symbol,
                "cost": 1,
                "preconditions": [{"kind": "fact_true", "fact": symbol}],
                "effects": [{"kind": "resource_delta", "resource": symbol, "delta": -1}],
            }
        ],
        "candidates": [{"name": symbol, "plan": [symbol]}],
        "goal": [symbol],
        "final_constraints": [symbol],
        "budget": 2,
    }


def _legacy_alpha_rename(value: Any) -> Any:
    mapping: dict[str, str] = {}

    def symbol(item: str) -> str:
        if item not in mapping:
            mapping[item] = f"z{len(mapping)}"
        return mapping[item]

    def walk(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {(key if key in SCHEMA_KEYS else symbol(str(key))): walk(child) for key, child in item.items()}
        if isinstance(item, list):
            return [walk(child) for child in item]
        if isinstance(item, str):
            if item in RESERVED_VALUES:
                return item
            if item.startswith("$arg:"):
                return "$arg:" + symbol(item[5:])
            return symbol(item)
        return item

    return walk(value)


def run_alpha_collision_matrix() -> dict[str, Any]:
    literals = sorted((SCHEMA_KEYS | RESERVED_VALUES) - {"$neighbor"})
    failures: list[str] = []
    legacy_kills: list[str] = []
    fingerprints: dict[str, str] = {}
    for literal in literals:
        candidate = _collision_ast(literal)
        original = semantic_fingerprint(candidate)
        renamed = semantic_fingerprint(alpha_rename(candidate))
        fingerprints[literal] = original
        if renamed != original:
            failures.append(literal)
        if semantic_fingerprint(_legacy_alpha_rename(candidate)) != original:
            legacy_kills.append(literal)

    order_probe = _collision_ast("probe")
    order_probe["candidates"][0]["plan"] = ["probe", "second"]
    reordered = deepcopy(order_probe)
    reordered["candidates"][0]["plan"].reverse()
    order_sensitive = semantic_fingerprint(order_probe) != semantic_fingerprint(reordered)
    return {
        "literal_count": len(literals),
        "literals_sha256": _sha_bytes(_canonical(literals)),
        "fingerprints_sha256": _sha_bytes(_canonical(fingerprints)),
        "failures": failures,
        "legacy_fault_kill_count": len(legacy_kills),
        "legacy_fault_examples": legacy_kills[:12],
        "order_sensitive_negative": order_sensitive,
        "passed": not failures and len(legacy_kills) > 0 and order_sensitive,
    }


def run_v16_regressions(repo_root: Path) -> dict[str, Any]:
    dataset = repo_root / V16_FORMAL_ROOT / "dataset"
    failures: list[dict[str, Any]] = []
    repaired = 0
    for path in sorted(dataset.glob("ere-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            state_values = {
                value
                for attributes in row["program_ast"]["initial_state"]["attributes"].values()
                for value in attributes.values()
            }
            if state_values == set(row["generation_metadata"]["choice_values"]):
                continue
            failures.append({"example_id": row["example_id"], "record_seed": row["record_seed"]})
            split = "train" if row["split"] == "causal_pairs" else row["split"]
            rebuilt = build_ere_pair(row["record_seed"], split, row["provenance"]["unit_index"])
            repaired += int(_visible_domain(rebuilt))

    alpha_row = None
    for line in (dataset / "cps-validation.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["example_id"] == "cps-validation-0777":
            alpha_row = row
            break
    if alpha_row is None:
        return {"passed": False, "failure": "sealed_alpha_regression_row_missing"}
    alpha_original = semantic_fingerprint(alpha_row["program_ast"])
    alpha_repaired = semantic_fingerprint(alpha_rename(alpha_row["program_ast"])) == alpha_original
    legacy_reproduced = semantic_fingerprint(_legacy_alpha_rename(alpha_row["program_ast"])) != alpha_original
    return {
        "v16_visible_failure_count": len(failures),
        "v16_visible_failure_ids_sha256": _sha_bytes(_canonical([row["example_id"] for row in failures])),
        "repaired_count": repaired,
        "alpha_example_id": alpha_row["example_id"],
        "alpha_original_fingerprint": alpha_original,
        "alpha_repaired": alpha_repaired,
        "legacy_failure_reproduced": legacy_reproduced,
        "passed": len(failures) == EXPECTED_V16_VISIBLE_FAILURES and repaired == len(failures) and alpha_repaired and legacy_reproduced,
    }


def run_fault_matrix() -> dict[str, Any]:
    pair = build_ere_pair(QUALIFICATION_SEED, "train", 5)
    corrupted = deepcopy(pair)
    ast, alternate, metadata = corrupted
    victim = metadata["choice_values"][-1]
    replacement = metadata["choice_values"][0]
    for candidate in (ast, alternate):
        for attributes in candidate["initial_state"]["attributes"].values():
            for attribute, value in list(attributes.items()):
                if value == victim:
                    attributes[attribute] = replacement
    visible_fault_killed = not _visible_domain(corrupted)
    collision = _collision_ast("rules")
    alpha_fault_killed = semantic_fingerprint(_legacy_alpha_rename(collision)) != semantic_fingerprint(collision)
    return {
        "visible_domain_postcondition_fault_killed": visible_fault_killed,
        "path_insensitive_alpha_fault_killed": alpha_fault_killed,
        "passed": visible_fault_killed and alpha_fault_killed,
    }


def assess_repair_qualification(repo_root: Path, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    emit = progress or (lambda _: None)
    operations: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("R01", lambda: verify_v16_runtime_evidence(repo_root)),
        ("R02", lambda: verify_runtime_surface_unchanged(repo_root)),
        ("R03", run_visible_domain_sweep),
        ("R04", run_alpha_collision_matrix),
        ("R05", lambda: run_v16_regressions(repo_root)),
        ("R06", run_fault_matrix),
    )
    metrics: dict[str, Any] = {}
    for gate, operation in operations:
        emit(f"repair:{gate}:start")
        metrics[gate] = operation()
        emit(f"repair:{gate}:done")
    gates = {gate: result.get("passed") is True for gate, result in metrics.items()}
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v17.repair-report.v1",
        "qualification_seed": QUALIFICATION_SEED,
        "gates": gates,
        "metrics": metrics,
        "passed": all(gates.values()),
        "status": "PASS_REPAIR_QUALIFICATION" if all(gates.values()) else "FAIL_REPAIR_QUALIFICATION",
    }


__all__ = [
    "CAPACITY_SEED",
    "IF_CHAIN_SWEEP_COUNT",
    "QUALIFICATION_SEED",
    "V16_FORMAL_ROOT",
    "V16_RUNTIME_ROOT",
    "assess_repair_qualification",
    "run_alpha_collision_matrix",
    "run_fault_matrix",
    "run_v16_regressions",
    "run_visible_domain_sweep",
    "verify_runtime_surface_unchanged",
    "verify_v16_runtime_evidence",
]
