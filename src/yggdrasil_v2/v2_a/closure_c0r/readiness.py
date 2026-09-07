from __future__ import annotations

"""Read-only C001--C008 audit for the C0R successor."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import contract


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
    base = root.resolve()
    candidate = (root / relative).resolve()
    return candidate if candidate != base and base in candidate.parents else None


def verify_seal(root: Path, expected_sha256: str | None) -> dict[str, Any]:
    seal_path = root / "evidence-seal.json"
    if expected_sha256 is None:
        return {
            "available": root.is_dir() and seal_path.is_file(),
            "passed": False,
            "failures": ["expected_seal_sha256_not_frozen"],
        }
    if not root.is_dir() or not seal_path.is_file():
        return {
            "available": False,
            "passed": False,
            "failures": ["root_or_seal_missing"],
        }
    failures: list[str] = []
    actual_seal = sha256_file(seal_path)
    if actual_seal != expected_sha256:
        failures.append("seal_sha256_mismatch")
    try:
        payload = read_json(seal_path)
        files = payload["files"]
        if not isinstance(files, dict) or not files:
            raise TypeError("files must be a nonempty object")
        actual_tree = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "evidence-seal.json"
        )
        if sorted(files) != actual_tree:
            failures.append("sealed_tree_mismatch")
        checked = 0
        for relative, expected in sorted(files.items()):
            path = _safe_file(root, relative)
            if path is None or not path.is_file():
                failures.append(f"missing_or_unsafe:{relative}")
                continue
            if not isinstance(expected, str) or sha256_file(path) != expected.upper():
                failures.append(f"file_hash_mismatch:{relative}")
            checked += 1
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "available": True,
            "passed": False,
            "seal_sha256": actual_seal,
            "failures": [*failures, f"invalid_seal:{type(exc).__name__}:{exc}"],
        }
    return {
        "available": True,
        "passed": not failures,
        "seal_sha256": actual_seal,
        "declared_file_count": len(files),
        "checked_file_count": checked,
        "failures": failures,
    }


def _load_pinned_json(path: Path, expected_sha256: str | None) -> tuple[Any | None, list[str]]:
    if expected_sha256 is None:
        return None, [f"unfrozen_hash:{path.as_posix()}"]
    if not path.is_file():
        return None, [f"missing:{path.as_posix()}"]
    if sha256_file(path) != expected_sha256:
        return None, [f"hash_mismatch:{path.as_posix()}"]
    try:
        return read_json(path), []
    except (OSError, TypeError, ValueError) as exc:
        return None, [f"invalid_json:{path.as_posix()}:{type(exc).__name__}:{exc}"]


def _fairness_valid(value: Any) -> bool:
    try:
        requirements = contract.FAIRNESS_REQUIREMENTS
        return bool(
            value["arms"] == requirements["arms"]
            and value["shared_inputs"]["canonical_public_forward_fields"]
            == requirements["canonical_public_forward_fields"]
            and value["shared_inputs"]["family_visible_to_forward"] is False
            and value["shared_inputs"]["family_used_as_training_target"] is False
            and value["shared_inputs"]["reasoning_budget_visible_to_forward"] is False
            and value["shared_inputs"]["valid_choice_mask_visible_to_forward"] is False
            and value["shared_inputs"]["program_ast_trace_claims_visible_to_forward"] is False
            and value["teacher_symmetry"]["required"] is True
            and value["teacher_symmetry"]["primary_lane"]["dense_state_or_claim_targets_allowed"] is False
            and value["teacher_symmetry"]["secondary_lane"]["eligible_for_medium_superiority_claim"] is False
            and value["selection_and_budget"]["required_matched_slices"]
            == requirements["required_matched_slices"]
            and value["trace_qualification"]["decode_max_new_tokens"]
            == requirements["decode_max_new_tokens"]
            and set(requirements["forbidden_active_components"])
            <= set(value["forbidden_active_components"])
            and value["cache_policy"]["p0m_smoke_cache_reusable_for_c1"] is False
            and value["latency_rule"]
            == "online end-to-end paths are primary; cached latency is diagnostic only"
        )
    except (KeyError, TypeError):
        return False


def audit_readiness(repo_root: Path) -> dict[str, Any]:
    old_root = repo_root / contract.OLD_C0_ROOT
    data_root = repo_root / contract.DATA_OUTPUT_ROOT
    old_seal = verify_seal(old_root, contract.OLD_C0_SEAL_SHA256)
    data_seal = verify_seal(data_root, contract.DATA_SEAL_SHA256)
    old_result, old_result_failures = _load_pinned_json(
        old_root / "result.json", contract.OLD_C0_RESULT_SHA256
    )
    fairness, fairness_failures = _load_pinned_json(
        old_root / "fairness-contract.json", contract.OLD_FAIRNESS_SHA256
    )
    data_result, data_result_failures = _load_pinned_json(
        data_root / "result.json", contract.DATA_RESULT_SHA256
    )
    failures = [*old_result_failures, *fairness_failures, *data_result_failures]

    old_gates = old_result.get("gates", {}) if isinstance(old_result, Mapping) else {}
    data_gates = data_result.get("gates", {}) if isinstance(data_result, Mapping) else {}
    data_complete = bool(
        isinstance(data_result, Mapping)
        and data_result.get("status") == "PASS_V2_A_C0R_DATA_TRACE_QUALIFICATION"
        and data_result.get("passed") is True
        and data_result.get("training_started") is False
        and data_result.get("optimizer_steps") == 0
        and data_result.get("model_writes") == 0
        and data_result.get("source_stable") is True
        and set(data_gates) == set(contract.DATA_GATE_IDS)
    )
    old_complete = bool(
        isinstance(old_result, Mapping)
        and old_result.get("status") == "FAIL_V2_A_CLOSURE_C0_READINESS"
        and old_result.get("authorizes") == "nothing"
        and old_result.get("training_started") is False
        and old_result.get("optimizer_steps") == 0
        and old_result.get("model_writes") == 0
    )
    fairness_passed = _fairness_valid(fairness)
    gates = {
        "C001": bool(
            old_seal["passed"]
            and data_seal["passed"]
            and data_complete
            and data_gates.get("D001") is True
            and data_gates.get("D002") is True
        ),
        "C002": bool(data_gates.get("D005") is True and data_gates.get("D006") is True),
        "C003": all(data_gates.get(f"D{index:03d}") is True for index in range(2, 9)),
        "C004": all(data_gates.get(f"D{index:03d}") is True for index in range(9, 12)),
        "C005": bool(old_complete and old_gates.get("C005") is True),
        "C006": bool(fairness_passed and data_gates.get("D012") is True),
        "C007": bool(old_complete and old_gates.get("C007") is True),
        "C008": bool(old_complete and old_gates.get("C008") is True),
    }
    availability_complete = bool(
        old_seal["available"]
        and data_seal["available"]
        and not failures
        and old_result is not None
        and fairness is not None
        and data_result is not None
        and contract.DATA_RESULT_SHA256 is not None
        and contract.DATA_SEAL_SHA256 is not None
    )
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.readiness-audit.v1",
        "identity": contract.READINESS_IDENTITY,
        "availability_complete": availability_complete,
        "passed": availability_complete and all(gates.values()),
        "gates": gates,
        "failures": failures,
        "old_c0": {
            "seal": old_seal,
            "result_verified": not old_result_failures,
            "historical_failures_preserved": ["C004", "C006"],
        },
        "data_trace": {
            "seal": data_seal,
            "result_verified": not data_result_failures,
            "gates": dict(data_gates) if isinstance(data_gates, Mapping) else {},
        },
        "fairness_contract": {
            "hash_verified": not fairness_failures,
            "semantic_requirements_passed": fairness_passed,
        },
        "interpretation": (
            "C0R only qualifies the repaired task bank, trace metric, and matched four-arm "
            "contract. It does not train or validate an architecture."
        ),
    }


def decide_readiness(audit: Mapping[str, Any]) -> dict[str, Any]:
    complete = audit.get("availability_complete") is True
    passed = complete and audit.get("passed") is True
    status = (
        "PASS_V2_A_CLOSURE_C0R_READINESS"
        if passed
        else "FAIL_V2_A_CLOSURE_C0R_READINESS"
        if complete
        else "INCOMPLETE_V2_A_CLOSURE_C0R_READINESS"
    )
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.readiness-result.v1",
        "identity": contract.READINESS_IDENTITY,
        "status": status,
        "passed": passed,
        "exit_code": 0 if passed else 1,
        "gates": dict(audit.get("gates", {})),
        "availability_complete": complete,
        "training_started": False,
        "optimizer_steps": 0,
        "model_writes": 0,
        "four_arm_results_present": False,
        "v2a_passed": False,
        "c1_single_seed_implementation_authorized": passed,
        "authorizes": "C1 single-seed implementation and eligibility only" if passed else "nothing",
        "interpretation": audit.get("interpretation"),
    }


__all__ = [
    "audit_readiness",
    "decide_readiness",
    "read_json",
    "sha256_file",
    "verify_seal",
]
