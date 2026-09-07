from __future__ import annotations

"""C1S provenance, source identity, single-use, and seal primitives."""

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
    Path("experiments/v2_a_closure_c1s.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/artifacts.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/metrics.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/model.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/runner.py"),
    Path("tests/v2_a_closure_c1s/test_contract.py"),
    Path("tests/v2_a_closure_c1s/test_metrics.py"),
    Path("tests/v2_a_closure_c1s/test_model.py"),
    Path("tests/v2_a_closure_c1s/test_runner.py"),
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
        raise FileNotFoundError(f"C1S source identity is incomplete: {missing}")
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_FILES}


def source_identity(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def snapshot_sources(repo_root: Path, output_root: Path) -> None:
    for relative in SOURCE_FILES:
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)


def claim_single_use(output_root: Path, lease_path: Path) -> None:
    if output_root.exists() or lease_path.exists():
        raise FileExistsError("fixed C1S root or sibling lease already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    payload = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.single-use-lease.v1",
        "identity": contract.IDENTITY,
        "stage": "S0",
        "formal": False,
        "output_root": output_root.as_posix(),
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
    name = "evidence-seal.json"
    write_json(
        root / name,
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.evidence-seal.v1",
            "identity": contract.IDENTITY,
            "stage": "S0",
            "files": tree_hashes(root, exclude=(name,)),
        },
    )
    return sha256_file(root / name)


def audit_evidence_seal(root: Path) -> dict[str, Any]:
    path = root / "evidence-seal.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = payload["files"]
        actual = tree_hashes(root, exclude=("evidence-seal.json",))
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        identity_ok = payload.get("identity") == contract.IDENTITY
        schema_ok = payload.get("schema_version") == f"{contract.SCHEMA_PREFIX}.evidence-seal.v1"
        return {
            "passed": identity_ok and schema_ok and not missing and not unexpected and not mismatched,
            "identity": identity_ok,
            "schema": schema_ok,
            "entries": len(expected),
            "matched": len(expected) - len(missing) - len(mismatched),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
            "seal_sha256": sha256_file(path),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def audit_external_pin(repo_root: Path, pin: Mapping[str, str], *, replay: bool = True) -> dict[str, Any]:
    root = repo_root / pin["root"]
    result_path = root / "result.json"
    seal_path = root / "evidence-seal.json"
    result_hash = sha256_file(result_path) if result_path.is_file() else None
    seal_hash = sha256_file(seal_path) if seal_path.is_file() else None
    replay_report: dict[str, Any] = {"passed": False, "reason": "not run"}
    if replay and seal_path.is_file():
        try:
            payload = json.loads(seal_path.read_text(encoding="utf-8"))
            expected = payload["files"]
            actual = tree_hashes(root, exclude=("evidence-seal.json",))
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            mismatched = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
            replay_report = {
                "passed": not missing and not unexpected and not mismatched,
                "entries": len(expected),
                "matched": len(expected) - len(missing) - len(mismatched),
                "missing": missing,
                "unexpected": unexpected,
                "mismatched": mismatched,
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            replay_report = {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}
    checks = {
        "result_hash": result_hash == pin["result_sha256"],
        "seal_hash": seal_hash == pin["seal_sha256"],
        "seal_replay": replay_report.get("passed") is True,
    }
    return {
        "root": pin["root"],
        "expected_result_sha256": pin["result_sha256"],
        "actual_result_sha256": result_hash,
        "expected_seal_sha256": pin["seal_sha256"],
        "actual_seal_sha256": seal_hash,
        "seal_replay": replay_report,
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "SOURCE_FILES",
    "audit_evidence_seal",
    "audit_external_pin",
    "canonical_json_bytes",
    "claim_single_use",
    "sha256_file",
    "snapshot_sources",
    "source_hashes",
    "source_identity",
    "tree_hashes",
    "write_evidence_seal",
    "write_json",
]
