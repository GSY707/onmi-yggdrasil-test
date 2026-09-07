from __future__ import annotations

"""S2 source closure, single-use leases, snapshots, and evidence seals."""

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from yggdrasil_v2.v2_a.closure_c1u.artifacts import (
    canonical_json_bytes,
    read_json,
    sha256_file,
    tree_hashes,
    write_json,
)

from . import contract


SOURCE_FILES = (
    contract.DESIGN_DOC,
    Path("experiments/v2_a_closure_c1u_s2.py"),
    Path("src/yggdrasil_v2/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/__init__.py"),
    Path("src/yggdrasil_v2/r1_revalidation/common/simulator.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/cache.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/data.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/artifacts.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/cache.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/cards.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/evaluate.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/model.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/objective.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/runtime.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/s1.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/source.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/tasks.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u/train.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/artifacts.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/cache.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/evaluate.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/model.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/objective.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/runner.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/runtime.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/tasks.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1u_s2/train.py"),
    Path("tests/v2_a_closure_c1u/test_contract_tasks.py"),
    Path("tests/v2_a_closure_c1u/test_model_objective.py"),
    Path("tests/v2_a_closure_c1u/test_runtime_evaluate_runner.py"),
    Path("tests/v2_a_closure_c1u/test_s0_runner.py"),
    Path("tests/v2_a_closure_c1u/test_s1_training.py"),
    Path("tests/v2_a_closure_c1u/test_source_cache.py"),
    Path("tests/v2_a_closure_c1u_s2/test_cache.py"),
    Path("tests/v2_a_closure_c1u_s2/test_contract_tasks.py"),
    Path("tests/v2_a_closure_c1u_s2/test_model_objective.py"),
    Path("tests/v2_a_closure_c1u_s2/test_runner.py"),
    Path("tests/v2_a_closure_c1u_s2/test_train_evaluate.py"),
)


def source_hashes(repo_root: Path) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    missing = [path.as_posix() for path in SOURCE_FILES if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"C1U S2 source closure is incomplete: {missing}")
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_FILES}


def source_identity(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest().upper()


def snapshot_sources(repo_root: Path, output_root: Path) -> None:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root)
    for relative in SOURCE_FILES:
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)


def claim_single_use(
    *, identity: str, stage: str, output_root: Path, lease_path: Path
) -> None:
    output_root = Path(output_root)
    lease_path = Path(lease_path)
    if output_root.exists() or lease_path.exists():
        raise FileExistsError(f"fixed {stage} root or sibling lease already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    payload = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.single-use-lease.v1",
        "identity": identity,
        "stage": stage,
        "output_root": output_root.as_posix(),
        "created_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
    }
    with lease_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def write_evidence_seal(root: Path, *, identity: str, stage: str) -> str:
    root = Path(root)
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
    root: Path, *, expected_identity: str | None = None, expected_stage: str | None = None
) -> dict[str, Any]:
    root = Path(root)
    try:
        payload = read_json(root / "evidence-seal.json")
        expected = dict(payload["files"])
        actual = tree_hashes(root, exclude=("evidence-seal.json",))
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        checks = {
            "schema": payload.get("schema_version")
            == f"{contract.SCHEMA_PREFIX}.evidence-seal.v1",
            "identity": expected_identity is None
            or payload.get("identity") == expected_identity,
            "stage": expected_stage is None or payload.get("stage") == expected_stage,
            "tree": not missing and not unexpected and not mismatched,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "entries": len(expected),
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
    "read_json",
    "sha256_file",
    "snapshot_sources",
    "source_hashes",
    "source_identity",
    "tree_hashes",
    "write_evidence_seal",
    "write_json",
]
