from __future__ import annotations

"""Canonical source identity, single-use claims, and small diagnostic seals."""

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
    Path("experiments/v2_a_closure_c1_diagnosis.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/artifacts.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/evaluator.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/exposure.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/gradients.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/runner.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/selection.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1_diagnosis/trace_metrics.py"),
    Path("tests/v2_a_closure_c1_diagnosis/test_evaluator.py"),
    Path("tests/v2_a_closure_c1_diagnosis/test_exposure_selection.py"),
    Path("tests/v2_a_closure_c1_diagnosis/test_trace_gradients_runner.py"),
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
        raise FileNotFoundError(f"C1 diagnosis source identity is incomplete: {missing}")
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_FILES}


def source_identity(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def snapshot_sources(repo_root: Path, output_root: Path) -> None:
    for relative in SOURCE_FILES:
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)


def claim_single_use(
    *,
    output_root: Path,
    lease_path: Path,
    formal: bool,
    identity: str = contract.IDENTITY,
) -> None:
    if output_root.exists() or lease_path.exists():
        raise FileExistsError("fixed diagnostic root or sibling lease already exists")
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
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
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
    write_json(
        root / "evidence-seal.json",
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.evidence-seal.v1",
            "files": tree_hashes(root, exclude=("evidence-seal.json",)),
        },
    )
    return sha256_file(root / "evidence-seal.json")


def audit_evidence_seal(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads((root / "evidence-seal.json").read_text(encoding="utf-8"))
        expected = payload["files"]
        if not isinstance(expected, dict):
            raise TypeError("seal files must be an object")
        actual = tree_hashes(root, exclude=("evidence-seal.json",))
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            key for key in set(expected) & set(actual) if expected[key] != actual[key]
        )
        return {
            "passed": not missing and not unexpected and not mismatched,
            "entries": len(expected),
            "matched": len(expected) - len(missing) - len(mismatched),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
            "seal_sha256": sha256_file(root / "evidence-seal.json"),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "SOURCE_FILES",
    "audit_evidence_seal",
    "canonical_json_bytes",
    "claim_single_use",
    "sha256_file",
    "snapshot_sources",
    "source_hashes",
    "source_identity",
    "write_evidence_seal",
    "write_json",
]
