from __future__ import annotations

"""Single-use runners for C0R data/trace qualification and readiness."""

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

from . import compact_trace, contract, data_audit, data_generator, readiness


SOURCE_FILES = (
    Path("docs/v2-a-closure-c0r-data-trace-qualification.md"),
    Path("experiments/v2_a_closure_c0r.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/compact_trace.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/data_audit.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/data_generator.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/readiness.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/runner.py"),
    Path("tests/v2_a_closure_c0r/test_compact_trace.py"),
    Path("tests/v2_a_closure_c0r/test_contract_and_readiness.py"),
    Path("tests/v2_a_closure_c0r/test_data_audit.py"),
    Path("tests/v2_a_closure_c0r/test_data_generator.py"),
    Path("tests/v2_a_closure_c0r/test_runner.py"),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): readiness.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "evidence-seal.json"
    }


def _write_seal(root: Path) -> str:
    write_json(
        root / "evidence-seal.json",
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.evidence-seal.v1",
            "files": _tree_hashes(root),
        },
    )
    return readiness.sha256_file(root / "evidence-seal.json")


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): readiness.sha256_file(repo_root / path) for path in SOURCE_FILES}


def _source_identity(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _snapshot_sources(repo_root: Path, root: Path) -> None:
    for relative in SOURCE_FILES:
        destination = root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)
    for relative in sorted(contract.PINNED_SUBSTRATE_HASHES):
        source = repo_root / relative
        destination = root / "substrate_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _pinned_substrate(repo_root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for relative, expected in sorted(contract.PINNED_SUBSTRATE_HASHES.items()):
        path = repo_root / relative
        actual = readiness.sha256_file(path) if path.is_file() else None
        rows[relative] = {"expected": expected, "actual": actual, "passed": actual == expected}
    return {"passed": all(row["passed"] for row in rows.values()), "files": rows}


def _write_lease(path: Path, *, identity: str, output_root: Path, formal: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-lease.v1",
        "identity": identity,
        "output_root": output_root.as_posix(),
        "formal": formal,
        "created_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _refusal(identity: str, root: Path, lease: Path, status: str) -> dict[str, Any]:
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.refusal.v1",
        "identity": identity,
        "status": status,
        "passed": False,
        "exit_code": 2,
        "output_root": root.as_posix(),
        "lease_path": lease.as_posix(),
        "reason": "fixed output root or sibling lease already exists",
        "authorizes": "nothing",
    }


def _claim(root: Path, lease: Path, *, identity: str, formal: bool) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=False, exist_ok=False)
    _write_lease(lease, identity=identity, output_root=root, formal=formal)


def _flatten(dataset: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for cell in sorted(dataset) for row in dataset[cell]]


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        contract.MODEL_ID,
        revision=contract.MODEL_REVISION,
        local_files_only=True,
        use_fast=True,
    )


def run_data_trace_qualification(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    formal: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Generate, qualify, and seal the new bank exactly once."""

    root = output_root or repo_root / contract.DATA_OUTPUT_ROOT
    lease = lease_path or repo_root / contract.DATA_LEASE_PATH
    identity = contract.DATA_IDENTITY
    refusal_status = "REFUSE_V2_A_C0R_DATA_TRACE_SINGLE_USE"
    if root.exists() or lease.exists():
        return _refusal(identity, root, lease, refusal_status)
    started_at = datetime.now(UTC).isoformat()
    wall_started = time.perf_counter()
    try:
        _claim(root, lease, identity=identity, formal=formal)
        source_before = _source_hashes(repo_root)
        source_identity = _source_identity(source_before)
        _snapshot_sources(repo_root, root)
        write_json(root / "contract-manifest.json", contract.data_contract_manifest())
        substrate = _pinned_substrate(repo_root)
        active_tokenizer = tokenizer or _load_tokenizer()
        data_generator.generate(
            root / "dataset",
            spec=data_generator.FORMAL_SPEC,
            design_doc=repo_root / "docs/v2-a-closure-c0r-data-trace-qualification.md",
            tokenizer=active_tokenizer,
        )
        records = _flatten(data_generator.load(root / "dataset"))
        audit = data_audit.audit_successor_dataset(
            repo_root,
            root / "dataset",
            data_generator,
            records=records,
            tokenizer=active_tokenizer,
        )
        trace = compact_trace.qualify_full_bank(records, active_tokenizer)
        gates = dict(audit.get("gates", {}))
        gates["D001"] = bool(gates.get("D001") is True and substrate["passed"])
        gates["D012"] = trace.get("passed") is True
        source_after = _source_hashes(repo_root)
        source_stable = source_before == source_after
        if not source_stable:
            gates["D001"] = False
        expected_gate_set = set(contract.DATA_GATE_IDS)
        passed = set(gates) == expected_gate_set and all(gates.values())
        status = (
            "PASS_V2_A_C0R_DATA_TRACE_QUALIFICATION"
            if passed
            else "FAIL_V2_A_C0R_DATA_TRACE_QUALIFICATION"
        )
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.data-result.v1",
            "identity": identity,
            "formal": formal,
            "status": status,
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "gates": gates,
            "source_identity": source_identity,
            "source_stable": source_stable,
            "dataset_manifest_sha256": readiness.sha256_file(root / "dataset/manifest.json"),
            "record_count": len(records),
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "authorizes": "C0R readiness audit only" if passed else "nothing",
        }
        write_json(root / "substrate-identity.json", substrate)
        write_json(root / "generator-report.json", audit)
        write_json(root / "trace-qualification.json", trace)
        write_json(root / "result.json", result)
        write_json(
            root / "run-state.json",
            {
                "schema_version": f"{contract.SCHEMA_PREFIX}.run-state.v1",
                "identity": identity,
                "formal": formal,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "wall_seconds": time.perf_counter() - wall_started,
                "source_identity": source_identity,
                "source_stable": source_stable,
                "training_started": False,
                "optimizer_steps": 0,
                "model_writes": 0,
            },
        )
    except Exception as exc:  # noqa: BLE001 - crash evidence must be sealed
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.data-crash.v1",
            "identity": identity,
            "formal": formal,
            "status": "CRASH_V2_A_C0R_DATA_TRACE_QUALIFICATION",
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
        write_json(root / "result.json", result)
    seal_sha256 = _write_seal(root)
    return {**result, "evidence_seal_sha256": seal_sha256, "output_root": root.as_posix()}


def run_readiness(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
) -> dict[str, Any]:
    """Run the read-only C0R readiness decision once."""

    root = output_root or repo_root / contract.READINESS_OUTPUT_ROOT
    lease = lease_path or repo_root / contract.READINESS_LEASE_PATH
    if root.exists() or lease.exists():
        return _refusal(
            contract.READINESS_IDENTITY,
            root,
            lease,
            "REFUSE_V2_A_CLOSURE_C0R_SINGLE_USE",
        )
    started_at = datetime.now(UTC).isoformat()
    wall_started = time.perf_counter()
    try:
        _claim(root, lease, identity=contract.READINESS_IDENTITY, formal=True)
        source_before = _source_hashes(repo_root)
        source_identity = _source_identity(source_before)
        _snapshot_sources(repo_root, root)
        write_json(root / "contract-manifest.json", contract.readiness_contract_manifest())
        audit = readiness.audit_readiness(repo_root)
        result = readiness.decide_readiness(audit)
        source_after = _source_hashes(repo_root)
        source_stable = source_before == source_after
        if not source_stable:
            result.update(
                {
                    "status": "FAIL_V2_A_CLOSURE_C0R_READINESS",
                    "passed": False,
                    "exit_code": 1,
                    "authorizes": "nothing",
                    "c1_single_seed_implementation_authorized": False,
                }
            )
        result["source_identity"] = source_identity
        result["source_stable"] = source_stable
        write_json(root / "input-audit.json", audit)
        write_json(root / "result.json", result)
        write_json(
            root / "run-state.json",
            {
                "schema_version": f"{contract.SCHEMA_PREFIX}.run-state.v1",
                "identity": contract.READINESS_IDENTITY,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "wall_seconds": time.perf_counter() - wall_started,
                "source_identity": source_identity,
                "source_stable": source_stable,
                "training_started": False,
                "optimizer_steps": 0,
                "model_writes": 0,
            },
        )
    except Exception as exc:  # noqa: BLE001 - crash evidence must be sealed
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.readiness-crash.v1",
            "identity": contract.READINESS_IDENTITY,
            "status": "CRASH_V2_A_CLOSURE_C0R_READINESS",
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
        write_json(root / "result.json", result)
    seal_sha256 = _write_seal(root)
    return {**result, "evidence_seal_sha256": seal_sha256, "output_root": root.as_posix()}


__all__ = [
    "SOURCE_FILES",
    "run_data_trace_qualification",
    "run_readiness",
    "write_json",
]
