from __future__ import annotations

"""Single-use artifact runner for V2-A Closure C0."""

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from .audit import audit_closure_c0, decide, sha256_file
from .contract import (
    DEFAULT_ARCHIVE_ROOT,
    IDENTITY,
    LEASE_PATH,
    OUTPUT_ROOT,
    SCHEMA_PREFIX,
    contract_manifest,
)


SOURCE_FILES = (
    Path("docs/v2-a-closure-c0-task-baseline-qualification.md"),
    Path("experiments/v2_a_closure_c0.py"),
    Path("src/yggdrasil_v2/v2_a/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0/audit.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0/runner.py"),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "evidence-seal.json"
    }


def _write_seal(root: Path) -> str:
    write_json(
        root / "evidence-seal.json",
        {
            "schema_version": f"{SCHEMA_PREFIX}.evidence-seal.v1",
            "files": _tree_hashes(root),
        },
    )
    return sha256_file(root / "evidence-seal.json")


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_FILES}


def _source_identity(hashes: dict[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _write_lease(path: Path, output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": f"{SCHEMA_PREFIX}.preflight-lease.v1",
        "identity": IDENTITY,
        "output_root": output_root.as_posix(),
        "created_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _refusal(output_root: Path, lease_path: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}.refusal.v1",
        "identity": IDENTITY,
        "status": "REFUSE_V2_A_CLOSURE_C0_SINGLE_USE",
        "passed": False,
        "exit_code": 2,
        "output_root": output_root.as_posix(),
        "lease_path": lease_path.as_posix(),
        "reason": "fixed output root or sibling lease already exists",
        "authorizes": "nothing",
    }


def run_closure_c0(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
) -> dict[str, Any]:
    """Run C0 once. Overrides exist only for isolated unit tests."""

    root = output_root if output_root is not None else repo_root / OUTPUT_ROOT
    lease = lease_path if lease_path is not None else repo_root / LEASE_PATH
    if root.exists() or lease.exists():
        return _refusal(root, lease)

    started_at = datetime.now(UTC).isoformat()
    wall_started = time.perf_counter()
    _write_lease(lease, root)
    root.mkdir(parents=True, exist_ok=False)
    source_before = _source_hashes(repo_root)
    source_identity = _source_identity(source_before)
    for relative in SOURCE_FILES:
        destination = root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)

    write_json(root / "contract-manifest.json", contract_manifest())
    write_json(root / "fairness-contract.json", contract_manifest()["arm_contract"])
    try:
        audit, reuse = audit_closure_c0(repo_root, archive_root)
        result = decide(audit)
        write_json(root / "input-audit.json", audit)
        write_json(root / "reuse-matrix.json", reuse)
    except Exception as exc:  # noqa: BLE001 - crash evidence must be sealed
        result = {
            "schema_version": f"{SCHEMA_PREFIX}.crash.v1",
            "identity": IDENTITY,
            "status": "CRASH_V2_A_CLOSURE_C0_READINESS",
            "passed": False,
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "authorizes": "nothing",
        }
        write_json(root / "crash.json", result)

    source_after = _source_hashes(repo_root)
    snapshot_hashes = {
        path.as_posix(): sha256_file(root / "source_snapshot" / path) for path in SOURCE_FILES
    }
    integrity = {
        "source_identity": source_identity,
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "source_snapshot_hashes": snapshot_hashes,
        "source_stable": source_before == source_after == snapshot_hashes,
    }
    if result.get("passed") is True and not integrity["source_stable"]:
        result["pre_integrity_status"] = result["status"]
        result["status"] = "FAIL_V2_A_CLOSURE_C0_SOURCE_INTEGRITY"
        result["passed"] = False
        result["exit_code"] = 1
        result["c1_single_seed_implementation_authorized"] = False
        result["authorizes"] = "nothing"
    result["integrity"] = integrity
    write_json(root / "result.json", result)
    write_json(
        root / "run-state.json",
        {
            "schema_version": f"{SCHEMA_PREFIX}.run-state.v1",
            "identity": IDENTITY,
            "status": result["status"],
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_seconds": time.perf_counter() - wall_started,
            "pid": os.getpid(),
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "exit_code": result["exit_code"],
        },
    )
    seal_hash = _write_seal(root)
    result["output_root"] = root.as_posix()
    result["lease_path"] = lease.as_posix()
    result["evidence_seal_sha256"] = seal_hash
    return result


__all__ = ["SOURCE_FILES", "run_closure_c0", "write_json"]

