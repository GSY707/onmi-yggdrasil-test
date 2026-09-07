from __future__ import annotations

"""C1R-only provenance, single-use, and evidence-seal primitives.

The helpers intentionally have a new schema namespace and source manifest.
They may reuse the mechanics of the consumed C1 implementation, but a C1R
artifact can never be mistaken for a C1 result or a continuation of its root.
"""

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from . import contract


SOURCE_FILES = (
    contract.DESIGN_DOC,
    Path("experiments/v2_a_closure_c1r.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1r/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1r/artifacts.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1r/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1r/coverage.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1r/runner.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1r/train.py"),
    Path("tests/v2_a_closure_c1r/test_contract_coverage.py"),
    Path("tests/v2_a_closure_c1r/test_runner.py"),
    Path("tests/v2_a_closure_c1r/test_train.py"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def source_hashes(repo_root: Path) -> dict[str, str]:
    missing = [path.as_posix() for path in SOURCE_FILES if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"C1R source identity is incomplete: {missing}")
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_FILES}


def source_identity(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def snapshot_sources(repo_root: Path, output_root: Path) -> None:
    for relative in SOURCE_FILES:
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)


def refusal(
    *, identity: str = contract.C1R_IDENTITY,
    output_root: Path = contract.OUTPUT_ROOT,
    lease_path: Path = contract.LEASE_PATH,
    status: str = "REFUSED_SINGLE_USE_COLLISION",
) -> dict[str, Any]:
    """Return a non-authorizing refusal without touching the fixed root."""

    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.refusal.v1",
        "identity": identity,
        "status": status,
        "passed": False,
        "exit_code": 2,
        "output_root": output_root.as_posix(),
        "lease_path": lease_path.as_posix(),
        "reason": "fixed output root or sibling lease already exists",
        "authorizes": "nothing",
    }


def claim_single_use(
    *,
    output_root: Path = contract.OUTPUT_ROOT,
    lease_path: Path = contract.LEASE_PATH,
    formal: bool,
    identity: str = contract.C1R_IDENTITY,
) -> None:
    """Atomically claim a fresh root and sibling lease exactly once."""

    if output_root.exists() or lease_path.exists():
        raise FileExistsError("fixed output root or sibling lease already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    payload = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.single-use-lease.v1",
        "identity": identity,
        "output_root": output_root.as_posix(),
        "formal": formal,
        "created_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
    }
    with lease_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def tree_hashes(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def write_evidence_seal(root: Path) -> str:
    seal_name = "evidence-seal.json"
    write_json(
        root / seal_name,
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.evidence-seal.v1",
            "identity": contract.C1R_IDENTITY,
            "files": tree_hashes(root, exclude=(seal_name,)),
        },
    )
    return sha256_file(root / seal_name)


def audit_evidence_seal(
    root: Path,
    *,
    expected_schema_version: str | None = None,
    expected_identity: str | None = contract.C1R_IDENTITY,
) -> dict[str, Any]:
    """Replay a seal, optionally against a foreign namespace's schema/identity."""

    path = root / "evidence-seal.json"
    if not path.is_file():
        return {"passed": False, "reason": "missing evidence-seal.json"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_schema_version = expected_schema_version or f"{contract.SCHEMA_PREFIX}.evidence-seal.v1"
        if payload.get("schema_version") != expected_schema_version:
            return {"passed": False, "reason": "wrong C1R evidence-seal schema"}
        if expected_identity is not None and payload.get("identity") != expected_identity:
            return {"passed": False, "reason": "wrong C1R evidence-seal identity"}
        expected = payload["files"]
        if not isinstance(expected, dict):
            raise TypeError("seal files must be an object")
        actual = tree_hashes(root, exclude=("evidence-seal.json",))
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        return {
            "passed": not missing and not unexpected and not mismatched,
            "entries": len(expected),
            "matched": len(expected) - len(missing) - len(mismatched),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
            "seal_sha256": sha256_file(path),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def audit_pinned_root(
    root: Path,
    *,
    result_sha256: str,
    seal_sha256: str,
    expected_schema_version: str | None = None,
    expected_identity: str | None = contract.C1R_IDENTITY,
) -> dict[str, Any]:
    """Audit a pinned result root under C1R or another explicit namespace."""

    result_path = root / "result.json"
    seal_path = root / "evidence-seal.json"
    actual_result = sha256_file(result_path) if result_path.is_file() else None
    actual_seal = sha256_file(seal_path) if seal_path.is_file() else None
    replay = audit_evidence_seal(
        root,
        expected_schema_version=expected_schema_version,
        expected_identity=expected_identity,
    )
    return {
        "root": root.as_posix(),
        "expected_result_sha256": result_sha256,
        "actual_result_sha256": actual_result,
        "expected_seal_sha256": seal_sha256,
        "actual_seal_sha256": actual_seal,
        "seal_replay": replay,
        "passed": actual_result == result_sha256 and actual_seal == seal_sha256 and replay.get("passed") is True,
    }


def audit_preflight_root(
    root: Path,
    *,
    identity: str = contract.C1R_IDENTITY,
    status: str = "PASS_V2_A_C1R_PREFLIGHT",
    source_identity_value: str,
) -> dict[str, Any]:
    """Replay a sealed, non-formal, no-training preflight result."""

    result_path = root / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"root": root.as_posix(), "passed": False, "reason": f"{type(exc).__name__}: {exc}"}
    replay = audit_evidence_seal(root)
    device = result.get("device")
    checks = {
        "schema": result.get("schema_version") == f"{contract.SCHEMA_PREFIX}.preflight.v1",
        "identity": result.get("identity") == identity,
        "status": result.get("status") == status,
        "passed": result.get("passed") is True,
        "non_formal": result.get("formal") is False,
        "source_identity": result.get("source_identity") == source_identity_value,
        "source_stable": result.get("source_stable") is True,
        "training_not_started": result.get("training_started") is False,
        "zero_optimizer_steps": type(result.get("optimizer_steps")) is int and result.get("optimizer_steps") == 0,
        "zero_model_writes": type(result.get("model_writes")) is int and result.get("model_writes") == 0,
        "zero_old_root_writes": type(result.get("old_root_writes")) is int and result.get("old_root_writes") == 0,
        "old_roots_stable": result.get("old_roots_stable") is True,
        "registered_cuda_device": isinstance(device, Mapping)
        and device.get("passed") is True
        and contract.REQUIRED_CUDA_DEVICE_NAME_FRAGMENT.casefold()
        in str(device.get("selected_device_name", "")).casefold(),
        "seal_replay": replay.get("passed") is True,
    }
    return {
        "root": root.as_posix(),
        "result": result,
        "result_sha256": sha256_file(result_path),
        "seal_sha256": sha256_file(root / "evidence-seal.json") if (root / "evidence-seal.json").is_file() else None,
        "checks": checks,
        "seal_replay": replay,
        "passed": all(checks.values()),
    }


__all__ = [
    "SOURCE_FILES",
    "audit_evidence_seal",
    "audit_pinned_root",
    "audit_preflight_root",
    "canonical_json_bytes",
    "claim_single_use",
    "refusal",
    "sha256_file",
    "snapshot_sources",
    "source_hashes",
    "source_identity",
    "tree_hashes",
    "write_evidence_seal",
    "write_json",
]
