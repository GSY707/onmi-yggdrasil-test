from __future__ import annotations

"""Fail-closed artifact helpers for the C1S S1 read-only diagnosis.

This module deliberately has no dependency on the successor contract or on a
training module.  The diagnosis runner may use these primitives to pin old
sealed evidence and to create its own disposable output root, but this module
never opens an old root for writing and never creates model artifacts itself.
"""

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_PREFIX = "yggdrasil.v2-a.closure-c1s-s1-diagnosis"
SEAL_NAME = "evidence-seal.json"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"regular file required: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative_files(root: Path, *, exclude: Iterable[str] = ()) -> list[Path]:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise NotADirectoryError(f"regular directory required: {root}")
    excluded = {str(item).replace("\\", "/") for item in exclude}
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"symlinks are not permitted in sealed trees: {relative}")
        if path.is_file() and relative not in excluded:
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def tree_hashes(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, str]:
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _relative_files(root, exclude=exclude)
    }


def snapshot_tree(root: Path) -> dict[str, str]:
    """Capture a content snapshot used to prove an old root was not changed."""

    return tree_hashes(Path(root))


def audit_tree_snapshot(root: Path, snapshot: Mapping[str, str]) -> dict[str, Any]:
    """Compare a root with a prior snapshot without mutating either side."""

    try:
        expected = {str(key): str(value).upper() for key, value in snapshot.items()}
        actual = tree_hashes(Path(root))
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        return {
            "passed": not missing and not unexpected and not mismatched,
            "entries": len(expected),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
        }
    except (OSError, ValueError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


# More explicit aliases are useful to callers documenting old-root safety.
capture_old_root_snapshot = snapshot_tree
audit_old_root_unchanged = audit_tree_snapshot


def claim_single_use(
    *,
    identity: str,
    stage: str,
    output_root: Path,
    lease_path: Path,
    formal: bool = False,
) -> dict[str, Any]:
    """Atomically claim a previously unused disposable root and lease.

    The root and sibling lease are both exclusive.  If either already exists,
    this function refuses before touching it.  The caller remains responsible
    for writing and sealing its diagnosis result.
    """

    output_root = Path(output_root)
    lease_path = Path(lease_path)
    if output_root.exists() or lease_path.exists():
        raise FileExistsError(f"root or lease already exists: {output_root} / {lease_path}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    payload = {
        "schema_version": f"{SCHEMA_PREFIX}.single-use-lease.v1",
        "identity": str(identity),
        "stage": str(stage),
        "formal": bool(formal),
        "diagnostic_only": not bool(formal),
        "output_root": output_root.as_posix(),
        "created_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
    }
    try:
        with lease_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # The newly-created directory is empty and belongs to this failed
        # claim.  Remove only that exact directory so a partial claim cannot
        # masquerade as a usable root.
        try:
            output_root.rmdir()
        except OSError:
            pass
        raise
    return payload


def write_evidence_seal(
    root: Path,
    *,
    identity: str,
    stage: str,
    schema_prefix: str = SCHEMA_PREFIX,
) -> str:
    root = Path(root)
    if (root / SEAL_NAME).exists():
        raise FileExistsError(f"evidence seal already exists: {root / SEAL_NAME}")
    write_json(
        root / SEAL_NAME,
        {
            "schema_version": f"{schema_prefix}.evidence-seal.v1",
            "identity": str(identity),
            "stage": str(stage),
            "files": tree_hashes(root, exclude=(SEAL_NAME,)),
        },
    )
    return sha256_file(root / SEAL_NAME)


def audit_evidence_seal(
    root: Path,
    *,
    expected_identity: str | None = None,
    expected_stage: str | None = None,
    expected_schema_prefix: str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    path = root / SEAL_NAME
    try:
        payload = read_json(path)
        expected = payload.get("files")
        if not isinstance(expected, Mapping):
            raise TypeError("evidence seal files must be an object")
        expected = {str(key): str(value).upper() for key, value in expected.items()}
        actual = tree_hashes(root, exclude=(SEAL_NAME,))
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        schema = payload.get("schema_version")
        schema_ok = isinstance(schema, str) and schema.endswith(".evidence-seal.v1")
        if expected_schema_prefix is not None:
            schema_ok = schema == f"{expected_schema_prefix}.evidence-seal.v1"
        identity_ok = expected_identity is None or payload.get("identity") == expected_identity
        stage_ok = expected_stage is None or payload.get("stage") == expected_stage
        return {
            "passed": bool(schema_ok and identity_ok and stage_ok and not missing and not unexpected and not mismatched),
            "identity": identity_ok,
            "stage": stage_ok,
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


def audit_sealed_file_pins(
    root: Path,
    *,
    expected_identity: str,
    expected_stage: str,
    expected_files: Mapping[str, str],
    expected_schema_prefix: str | None = None,
) -> dict[str, Any]:
    replay = audit_evidence_seal(
        root,
        expected_identity=expected_identity,
        expected_stage=expected_stage,
        expected_schema_prefix=expected_schema_prefix,
    )
    rows: dict[str, Any] = {}
    try:
        seal = read_json(Path(root) / SEAL_NAME)
        sealed_files = seal.get("files")
        if not isinstance(sealed_files, Mapping):
            raise TypeError("evidence seal files must be an object")
        for relative, expected_sha_raw in expected_files.items():
            relative = str(relative).replace("\\", "/")
            expected_sha = str(expected_sha_raw).upper()
            path = Path(root) / relative
            actual_sha = sha256_file(path) if path.is_file() and not path.is_symlink() else None
            sealed_sha = str(sealed_files[relative]).upper() if relative in sealed_files else None
            checks = {
                "present": actual_sha is not None,
                "listed_in_seal": sealed_sha is not None,
                "expected_matches_seal": sealed_sha == expected_sha,
                "actual_matches_expected": actual_sha == expected_sha,
            }
            rows[relative] = {
                "expected_sha256": expected_sha,
                "sealed_sha256": sealed_sha,
                "actual_sha256": actual_sha,
                "checks": checks,
                "passed": all(checks.values()),
            }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"root": Path(root).as_posix(), "seal_replay": replay, "files": rows, "passed": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "root": Path(root).as_posix(),
        "seal_replay": replay,
        "files": rows,
        "passed": replay.get("passed") is True and all(row["passed"] for row in rows.values()),
    }


def audit_external_pin(repo_root: Path, pin: Mapping[str, Any], *, replay: bool = True) -> dict[str, Any]:
    """Revalidate an external sealed root without writing to it."""

    root = Path(repo_root) / str(pin["root"])
    result_path = root / "result.json"
    seal_path = root / SEAL_NAME
    try:
        result_hash = sha256_file(result_path)
        seal_hash = sha256_file(seal_path)
    except (OSError, ValueError):
        result_hash = seal_hash = None
    replay_report: dict[str, Any] = {"passed": True, "skipped": True}
    if replay:
        replay_report = audit_evidence_seal(root)
    checks = {
        "result_hash": result_hash == pin.get("result_sha256"),
        "seal_hash": seal_hash == pin.get("seal_sha256"),
        "seal_replay": replay_report.get("passed") is True,
    }
    return {
        "root": str(pin["root"]),
        "actual_result_sha256": result_hash,
        "actual_seal_sha256": seal_hash,
        "seal_replay": replay_report,
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_stage_predecessor(
    root: Path,
    *,
    expected_identity: str,
    expected_stage: str,
    expected_status: str,
    expected_authorizes: Any = None,
) -> dict[str, Any]:
    """Check an exact predecessor result and replay its immutable seal.

    ``expected_authorizes`` is intentionally compared as an object (rather
    than coerced to text), so a natural-language authorization cannot silently
    satisfy a structured successor scope.
    """

    root = Path(root)
    try:
        result = read_json(root / "result.json")
        replay = audit_evidence_seal(
            root,
            expected_identity=expected_identity,
            expected_stage=expected_stage,
        )
        checks = {
            "identity": result.get("identity") == expected_identity,
            "stage": result.get("stage") == expected_stage,
            "status": result.get("status") == expected_status,
            "passed": result.get("passed") is True,
            "authorizes": result.get("authorizes") == expected_authorizes,
            "seal": replay.get("passed") is True,
        }
        return {
            "root": root.as_posix(),
            "result_sha256": sha256_file(root / "result.json"),
            "seal_sha256": sha256_file(root / SEAL_NAME),
            "checks": checks,
            "seal_replay": replay,
            "passed": all(checks.values()),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"root": root.as_posix(), "passed": False, "reason": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "SCHEMA_PREFIX",
    "audit_evidence_seal",
    "audit_external_pin",
    "audit_old_root_unchanged",
    "audit_sealed_file_pins",
    "audit_stage_predecessor",
    "audit_tree_snapshot",
    "canonical_json_bytes",
    "capture_old_root_snapshot",
    "claim_single_use",
    "read_json",
    "sha256_file",
    "snapshot_tree",
    "tree_hashes",
    "write_evidence_seal",
    "write_json",
]
