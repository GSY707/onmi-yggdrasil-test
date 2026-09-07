from __future__ import annotations

"""Provenance, single-use leases, dynamic predecessor pins, and evidence seals."""

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
    Path("experiments/v2_a_closure_c1s_successor.py"),
    Path("src/yggdrasil_v2/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/artifacts.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/evaluate.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/objective.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/runner.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/runtime.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/targets.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/train.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/model.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/cache.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/data.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/evaluate.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/evaluation_runtime.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/metrics.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/qualification.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/runtime.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/trace_targets.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/compact_trace.py"),
    Path("src/yggdrasil_v2/r1_revalidation/__init__.py"),
    Path("src/yggdrasil_v2/r1_revalidation/common/__init__.py"),
    Path("src/yggdrasil_v2/r1_revalidation/common/simulator.py"),
    Path("src/yggdrasil_v2/r1_revalidation/learner/__init__.py"),
    Path("src/yggdrasil_v2/r1_revalidation/learner/canonical.py"),
    Path("src/yggdrasil_v2/r1_revalidation/learner/folds.py"),
    Path("src/yggdrasil_v2/r1_revalidation/learner/models.py"),
    Path("src/yggdrasil_v2/r1_revalidation/learner/qualification.py"),
    Path("src/yggdrasil_v2/r1_revalidation/learner/source.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/__init__.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/fingerprint.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/g09_decision.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/generator.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/generator_audit.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/p0_shortcut.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/renderer.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/scalable.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/schema.py"),
    Path("tests/v2_a_closure_c1s_successor/test_artifacts.py"),
    Path("tests/v2_a_closure_c1s_successor/test_contract.py"),
    Path("tests/v2_a_closure_c1s_successor/test_evaluate.py"),
    Path("tests/v2_a_closure_c1s_successor/test_objective.py"),
    Path("tests/v2_a_closure_c1s_successor/test_runner.py"),
    Path("tests/v2_a_closure_c1s_successor/test_runtime.py"),
    Path("tests/v2_a_closure_c1s_successor/test_targets.py"),
    Path("tests/v2_a_closure_c1s_successor/test_train.py"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def source_hashes(repo_root: Path) -> dict[str, str]:
    missing = [path.as_posix() for path in SOURCE_FILES if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"C1S successor source identity is incomplete: {missing}")
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_FILES}


def source_identity(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def snapshot_sources(repo_root: Path, output_root: Path) -> None:
    for relative in SOURCE_FILES:
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)


def claim_single_use(
    *,
    identity: str,
    stage: str,
    output_root: Path,
    lease_path: Path,
    formal: bool,
) -> None:
    if output_root.exists() or lease_path.exists():
        raise FileExistsError(f"fixed {stage} root or sibling lease already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    payload = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.single-use-lease.v1",
        "identity": identity,
        "stage": stage,
        "formal": bool(formal),
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
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def write_evidence_seal(root: Path, *, identity: str, stage: str) -> str:
    name = "evidence-seal.json"
    write_json(
        root / name,
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.evidence-seal.v1",
            "identity": identity,
            "stage": stage,
            "files": tree_hashes(root, exclude=(name,)),
        },
    )
    return sha256_file(root / name)


def audit_evidence_seal(
    root: Path,
    *,
    expected_identity: str | None = None,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    path = Path(root) / "evidence-seal.json"
    try:
        payload = read_json(path)
        expected = dict(payload["files"])
        actual = tree_hashes(Path(root), exclude=("evidence-seal.json",))
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        identity_ok = expected_identity is None or payload.get("identity") == expected_identity
        stage_ok = expected_stage is None or payload.get("stage") == expected_stage
        schema_ok = payload.get("schema_version") == f"{contract.SCHEMA_PREFIX}.evidence-seal.v1"
        return {
            "passed": bool(identity_ok and stage_ok and schema_ok and not missing and not unexpected and not mismatched),
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
) -> dict[str, Any]:
    """Replay a seal and independently re-hash selected contract-critical files.

    A successful tree replay already covers every file in the sealed root.  The
    explicit file rows are intentional redundancy for large reused inputs: a
    later preflight/launcher can demonstrate that the exact target bank and
    split ledger it consumed match the SHA values recorded by the producing
    preflight, rather than merely reporting that an opaque parent seal passed.
    """

    root = Path(root)
    replay = audit_evidence_seal(
        root,
        expected_identity=expected_identity,
        expected_stage=expected_stage,
    )
    rows: dict[str, Any] = {}
    try:
        seal = read_json(root / "evidence-seal.json")
        sealed_files = seal.get("files")
        if not isinstance(sealed_files, Mapping):
            raise TypeError("evidence seal files must be an object")
        for relative, expected_sha in expected_files.items():
            relative = str(relative)
            expected_sha = str(expected_sha).upper()
            path = root / relative
            actual_sha = sha256_file(path) if path.is_file() else None
            sealed_sha_raw = sealed_files.get(relative)
            sealed_sha = str(sealed_sha_raw).upper() if sealed_sha_raw is not None else None
            checks = {
                "present": path.is_file(),
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
        return {
            "root": root.as_posix(),
            "seal_replay": replay,
            "files": rows,
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "root": root.as_posix(),
        "seal_replay": replay,
        "files": rows,
        "passed": replay.get("passed") is True and all(row["passed"] for row in rows.values()),
    }


def audit_external_pin(
    repo_root: Path,
    pin: Mapping[str, str],
    *,
    replay: bool = True,
) -> dict[str, Any]:
    root = repo_root / str(pin["root"])
    result_path = root / "result.json"
    seal_path = root / "evidence-seal.json"
    result_hash = sha256_file(result_path) if result_path.is_file() else None
    seal_hash = sha256_file(seal_path) if seal_path.is_file() else None
    replay_report: dict[str, Any] = {"passed": True, "skipped": True}
    if replay:
        # External roots may use a different schema/identity, but their sealed
        # file map has the same immutable semantics.
        try:
            payload = read_json(seal_path)
            expected = dict(payload["files"])
            actual = tree_hashes(root, exclude=("evidence-seal.json",))
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            mismatched = sorted(
                name for name in set(expected) & set(actual) if expected[name] != actual[name]
            )
            replay_report = {
                "passed": not missing and not unexpected and not mismatched,
                "entries": len(expected),
                "missing": missing,
                "unexpected": unexpected,
                "mismatched": mismatched,
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            replay_report = {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}
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
    expected_authorizes: str,
) -> dict[str, Any]:
    root = Path(root)
    result_path = root / "result.json"
    try:
        result = read_json(result_path)
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
            "result_sha256": sha256_file(result_path),
            "seal_sha256": sha256_file(root / "evidence-seal.json"),
            "checks": checks,
            "seal_replay": replay,
            "passed": all(checks.values()),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"root": root.as_posix(), "passed": False, "reason": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "SOURCE_FILES",
    "audit_evidence_seal",
    "audit_external_pin",
    "audit_sealed_file_pins",
    "audit_stage_predecessor",
    "canonical_json_bytes",
    "claim_single_use",
    "read_json",
    "sha256_file",
    "snapshot_sources",
    "source_hashes",
    "source_identity",
    "tree_hashes",
    "write_evidence_seal",
    "write_json",
]
