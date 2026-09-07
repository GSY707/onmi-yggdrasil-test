from __future__ import annotations

"""Single-use H1 control plane.

This module owns launcher binding, readiness, stage creation, and fail-stop
sequencing.  Expensive work is implemented in :mod:`.workflow` and imported
only after all before-mutation entry checks have passed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import (
    git_identity,
    run_stage,
    sha256,
    source_identity,
    verify_seal,
    write_json,
)
from .contract import (
    ATTEMPT_ID,
    DESIGN,
    EXECUTION,
    FROZEN_CONTRACT_HASHES,
    NR1_ROOTS,
    P0D_HISTORY,
    P0M_ROOTS,
    ROOTS,
    THRESHOLDS,
    TRANSPORT_ROOT,
    V8L_QUALIFICATION,
)


def source_files(repo_root: Path) -> tuple[Path, ...]:
    """Return the complete H1 implementation/test snapshot, deterministically."""
    explicit = {
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("README.md"),
        DESIGN,
        EXECUTION,
        Path("docs/DIRECTORY_REFERENCE.md"),
        Path("docs/next-stage-test-plan.md"),
        Path("docs/v2-r1-revalidation-task-design.md"),
        Path("docs/v2-r1r-p0-result.md"),
        Path("experiments/v2_r1_revalidation.py"),
        Path("src/yggdrasil_v2/r1_revalidation/__init__.py"),
        Path("src/yggdrasil_v2/reasoning_medium/model.py"),
    }
    for directory in (
        Path("src/yggdrasil_v2/r1_revalidation/h1"),
        Path("src/yggdrasil_v2/r1_revalidation/nr1"),
        Path("tests/v2_r1r_p1_h1"),
    ):
        base = repo_root / directory
        if not base.is_dir():
            raise FileNotFoundError(f"missing active H1 source directory: {directory}")
        explicit.update(
            path.relative_to(repo_root)
            for path in base.rglob("*")
            if path.is_file() and path.suffix in {".py", ".json"}
        )
    files = tuple(sorted(explicit, key=lambda path: path.as_posix()))
    missing = [path.as_posix() for path in files if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing H1 source inputs: {missing}")
    return files


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def fixed_root_freshness(repo_root: Path) -> dict[str, Any]:
    # The launcher creates and binds TRANSPORT_ROOT before entering run_all;
    # only the two stage roots must still be fresh here.
    roots = tuple(ROOTS.values())
    existing = [path.as_posix() for path in roots if (repo_root / path).exists()]
    return {"roots": [path.as_posix() for path in roots], "existing_roots": existing, "passed": not existing}


def _runtime_exclusions() -> tuple[Path, ...]:
    return (*ROOTS.values(), TRANSPORT_ROOT)


def contract_hash_audit(repo_root: Path) -> dict[str, Any]:
    actual: dict[str, str | None] = {}
    checks: dict[str, bool] = {}
    for relative, expected in FROZEN_CONTRACT_HASHES.items():
        path = repo_root / relative
        actual[relative] = sha256(path) if path.is_file() else None
        checks[relative] = expected != "TO_BE_FROZEN" and actual[relative] == expected
    return {"expected": FROZEN_CONTRACT_HASHES, "actual": actual, "checks": checks, "passed": bool(checks) and all(checks.values())}


def _history_row(repo_root: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    relative = Path(specification["root"])
    root = repo_root / relative
    seal_path = root / "evidence-seal.json"
    result_path = root / str(specification.get("status_file", "result.json"))
    try:
        result = _load(result_path) if result_path.is_file() else {}
    except (OSError, TypeError, ValueError):
        result = {}
    actual_seal = sha256(seal_path) if seal_path.is_file() else None
    tree_valid = verify_seal(root)
    expected_status = specification.get("status")
    status_matches = (
        result.get("status") == expected_status
        if expected_status is not None
        else result.get("passed") is True
    )
    return {
        "root": relative.as_posix(),
        "status": result.get("status"),
        "expected_status": expected_status,
        "result_passed": result.get("passed"),
        "seal_sha256": actual_seal,
        "expected_seal_sha256": specification.get("seal_sha256"),
        "tree_seal_valid": tree_valid,
        "passed": (
            status_matches
            and actual_seal == specification.get("seal_sha256")
            and tree_valid
        ),
    }


def history_replay(repo_root: Path) -> dict[str, Any]:
    """Replay P0-D/P0-M/NR1 PASS and V8L FAIL without old runners."""
    p0d = _history_row(repo_root, P0D_HISTORY)
    p0m = {name: _history_row(repo_root, spec) for name, spec in P0M_ROOTS.items()}
    nr1 = {name: _history_row(repo_root, spec) for name, spec in NR1_ROOTS.items()}
    nr1_qualification_path = repo_root / NR1_ROOTS["qualification"]["root"] / "result.json"
    nr1_qualification = _load(nr1_qualification_path) if nr1["qualification"]["tree_seal_valid"] else {}
    nr1_boundary = {
        "p1_h1_design_authorized": nr1_qualification.get("p1_h1_design_authorized"),
        "p1_completed": nr1_qualification.get("p1_completed"),
        "p2_eligible": nr1_qualification.get("p2_eligible"),
        "p2_started": nr1_qualification.get("p2_started"),
    }
    nr1_boundary_passed = nr1_boundary == {
        "p1_h1_design_authorized": True,
        "p1_completed": False,
        "p2_eligible": False,
        "p2_started": False,
    }
    v8l = _history_row(repo_root, V8L_QUALIFICATION)
    result = _load(repo_root / V8L_QUALIFICATION["root"] / "result.json") if v8l["tree_seal_valid"] else {}
    boundary = {
        key: result.get(key)
        for key in ("fresh_p1_v9_authorized", "p1_completed", "p2_eligible", "p2_started")
    }
    boundary_expected = {
        "fresh_p1_v9_authorized": False,
        "p1_completed": False,
        "p2_eligible": False,
        "p2_started": False,
    }
    v8l["boundary"] = boundary
    v8l["boundary_passed"] = boundary == boundary_expected
    return {
        "p0d": p0d,
        "p0m": p0m,
        "nr1": nr1,
        "nr1_boundary": nr1_boundary,
        "nr1_boundary_passed": nr1_boundary_passed,
        "v8l": v8l,
        "passed": (
            p0d["passed"]
            and all(row["passed"] for row in p0m.values())
            and all(row["passed"] for row in nr1.values())
            and nr1_boundary_passed
            and v8l["passed"]
            and v8l["boundary_passed"]
        ),
    }


def successor_audit(repo_root: Path, allowed_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Reject formal H1/F1/v9/P2 roots, while allowing nonformal H1 artifacts."""
    allowed = {(repo_root / path).resolve() for path in allowed_paths}
    forbidden: list[str] = []
    artifact_bases = (repo_root / "artifacts/v2-r1r", repo_root / "tmp")
    for base in artifact_bases:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_dir() or path.resolve() in allowed:
                continue
            name = path.name.lower()
            formal_h1 = (
                name.startswith("p1-h1-preflight-")
                or name.startswith("p1-h1-development-")
            ) and "nonformal" not in name
            formal_successor = (
                name.startswith("p1-f1-")
                or name.startswith("p1-v9-")
                or name.startswith("p2-")
            )
            if formal_h1 or formal_successor:
                forbidden.append(path.relative_to(repo_root).as_posix())
    return {"forbidden_roots": sorted(set(forbidden)), "passed": not forbidden}


def _process_snapshot() -> dict[str, Any]:
    if os.name != "nt":
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,comm=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
        rows = []
        for line in completed.stdout.splitlines():
            parts = line.strip().split(None, 3)
            if len(parts) == 4:
                rows.append({"pid": int(parts[0]), "parent_pid": int(parts[1]), "name": parts[2], "command_line": parts[3]})
        return {"returncode": completed.returncode, "rows": rows}
    script = "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        raw = json.loads(completed.stdout or "[]") if completed.returncode == 0 else []
    except (TypeError, ValueError):
        raw = []
        completed.returncode = completed.returncode or 1
    if isinstance(raw, dict):
        raw = [raw]
    rows = [
        {"pid": int(row.get("ProcessId") or 0), "parent_pid": int(row.get("ParentProcessId") or 0), "name": row.get("Name") or "", "command_line": row.get("CommandLine") or ""}
        for row in raw
    ]
    return {"returncode": completed.returncode, "stderr": completed.stderr, "rows": rows}


def audit_process_rows(rows: list[dict[str, Any]], snapshot_returncode: int, current_pid: int) -> dict[str, Any]:
    by_pid = {int(row["pid"]): row for row in rows}
    allowed_chain = {current_pid}
    cursor = current_pid
    while cursor in by_pid:
        parent = int(by_pid[cursor].get("parent_pid") or 0)
        if not parent or parent in allowed_chain:
            break
        allowed_chain.add(parent)
        cursor = parent
    matches = []
    successor_matches = []
    for row in rows:
        normalized = str(row.get("command_line", "")).lower().replace("\\", "/")
        if "v2_r1_revalidation.py" in normalized and "run-p1-h1" in normalized:
            matches.append(row)
        if any(token in normalized for token in ("run-p1-f1", "run-p1-v9", "run-p2")):
            successor_matches.append(row)
    external = [row for row in matches if int(row["pid"]) not in allowed_chain]
    current_present = any(int(row["pid"]) == current_pid for row in matches)
    return {
        "snapshot_returncode": snapshot_returncode,
        "current_pid": current_pid,
        "allowed_ancestor_chain": sorted(allowed_chain),
        "formal_matches": matches,
        "external_formal_matches": external,
        "successor_matches": successor_matches,
        "passed": snapshot_returncode == 0 and current_present and not external and not successor_matches,
    }


def formal_process_audit() -> dict[str, Any]:
    snapshot = _process_snapshot()
    return audit_process_rows(snapshot["rows"], snapshot["returncode"], os.getpid())


def transport_audit(repo_root: Path) -> dict[str, Any]:
    root = (repo_root / TRANSPORT_ROOT).resolve()
    launch_path = root / "launch.json"
    try:
        launch = _load(launch_path) if launch_path.is_file() else {}
    except (OSError, TypeError, ValueError):
        launch = {}
    stdout_path = (root / "stdout.log").resolve()
    stderr_path = (root / "stderr.log").resolve()
    stdout_name = Path(str(getattr(sys.stdout, "name", ""))).resolve()
    stderr_name = Path(str(getattr(sys.stderr, "name", ""))).resolve()
    checks = {
        "attempt_id": os.environ.get("YGGDRASIL_H1_ATTEMPT_ID") == ATTEMPT_ID,
        "transport_environment": Path(os.environ.get("YGGDRASIL_H1_TRANSPORT_ROOT", ".")).resolve() == root,
        "launch_present": launch_path.is_file(),
        "launch_attempt_id": launch.get("attempt_id") == ATTEMPT_ID,
        "launch_pid": launch.get("pid") == os.getpid(),
        "launch_parent_pid": launch.get("parent_pid") == os.getppid(),
        "launch_cwd": Path(launch.get("cwd", ".")).resolve() == repo_root.resolve(),
        "stdout_redirected": stdout_name == stdout_path and not sys.stdout.isatty(),
        "stderr_redirected": stderr_name == stderr_path and not sys.stderr.isatty(),
        "launch_stdout_bound": Path(launch.get("stdout", ".")).resolve() == stdout_path,
        "launch_stderr_bound": Path(launch.get("stderr", ".")).resolve() == stderr_path,
        "formal_argv": launch.get("argv") == sys.argv,
    }
    return {"transport_root": TRANSPORT_ROOT.as_posix(), "launch_sha256": sha256(launch_path) if launch_path.is_file() else None, "checks": checks, "passed": all(checks.values())}


def formal_runtime_audit(repo_root: Path, allowed_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    process = formal_process_audit()
    transport = transport_audit(repo_root)
    successors = successor_audit(repo_root, allowed_paths)
    return {"process": process, "transport": transport, "successors": successors, "passed": process["passed"] and transport["passed"] and successors["passed"]}


def formal_entry_audit(repo_root: Path) -> dict[str, Any]:
    freshness = fixed_root_freshness(repo_root)
    runtime = formal_runtime_audit(repo_root)
    readiness = contract_readiness_audit(repo_root)
    return {
        "fixed_roots": freshness,
        "runtime": runtime,
        "readiness": readiness,
        "passed": freshness["passed"] and runtime["passed"] and readiness["passed"],
    }


def _cli_audit(repo_root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run([sys.executable, "experiments/v2_r1_revalidation.py", "--help"], cwd=repo_root, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"returncode": None, "error": str(exc), "required": {"run-p1-h1": False}, "forbidden": {}, "passed": False}
    text = completed.stdout + completed.stderr
    required = {"run-p1-h1"}
    forbidden = {"run-p1-nr1", "preflight-p1-h1", "run-p1-f1", "run-p1-v9", "run-p2"}
    return {"returncode": completed.returncode, "required": {value: value in text for value in sorted(required)}, "forbidden": {value: value in text for value in sorted(forbidden)}, "passed": completed.returncode == 0 and all(value in text for value in required) and all(value not in text for value in forbidden)}


def contract_readiness_audit(repo_root: Path) -> dict[str, Any]:
    """Check all immutable prerequisites before a fixed root may be created."""
    contract = contract_hash_audit(repo_root)
    history = history_replay(repo_root)
    cli = _cli_audit(repo_root)
    try:
        files = source_files(repo_root)
        source_surface = bool(files)
        source_error = None
    except (OSError, RuntimeError, ValueError) as exc:
        files = ()
        source_surface = False
        source_error = str(exc)
    checks = {
        "thresholds_frozen": THRESHOLDS.frozen(),
        "contract_hashes_frozen": contract["passed"] is True,
        "history_replay": history["passed"] is True,
        "cli_direct_switch": cli["passed"] is True,
        "source_surface_complete": source_surface,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "contract_hash_audit": contract,
        "history_replay": history,
        "cli_audit": cli,
        "source_file_count": len(files),
        "source_error": source_error,
    }


def launch_readiness_audit(repo_root: Path) -> dict[str, Any]:
    """Pre-transport audit used by the CLI before any filesystem mutation."""
    fixed = fixed_root_freshness(repo_root)
    transport_fresh = not (repo_root / TRANSPORT_ROOT).exists()
    successors = successor_audit(repo_root)
    process = formal_process_audit()
    contract = contract_readiness_audit(repo_root)
    checks = {
        "fixed_roots_fresh": fixed["passed"] is True,
        "transport_fresh": transport_fresh,
        "no_successor_roots": successors["passed"] is True,
        "formal_process_unique": process["passed"] is True,
        "contract_ready": contract["passed"] is True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "fixed_roots": fixed,
        "transport_fresh": transport_fresh,
        "successors": successors,
        "process": process,
        "contract": contract,
    }


def preflight_valid(repo_root: Path, identity: str) -> dict[str, Any]:
    root = repo_root / ROOTS["preflight"]
    path = root / "result.json"
    if not path.is_file():
        return {"passed": False, "reason": "H1 preflight result missing"}
    try:
        result = _load(path)
    except (OSError, TypeError, ValueError):
        return {"passed": False, "reason": "H1 preflight result malformed"}
    checks = {
        "status": result.get("status") == "PASS_P1_H1_PREFLIGHT",
        "passed": result.get("passed") is True,
        "source_identity": result.get("source_identity") == identity,
        "runner_integrity": result.get("runner_integrity", {}).get("passed") is True,
        "seal": verify_seal(root),
    }
    return {"checks": checks, "passed": all(checks.values())}


ControlAction = Callable[[Path, str, dict[str, Any]], dict[str, Any]]


def _not_implemented(stage: str) -> ControlAction:
    def action(_root: Path, _identity: str, _environment: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"H1 {stage} action is intentionally not implemented in the control-plane skeleton")
    return action


def run_all(repo_root: Path, *, preflight_action: ControlAction | None = None, development_action: ControlAction | None = None) -> int:
    """Run the only two-stage H1 path; injected actions remain test-only."""
    repo_root = Path(repo_root).resolve()
    files = source_files(repo_root)
    entry = formal_entry_audit(repo_root)
    if not entry["passed"]:
        print(json.dumps({"status": "REFUSE_P1_H1_BEFORE_MUTATION", "passed": False, "reason": "entry gate failed", "details": entry}, ensure_ascii=False), flush=True)
        return 2
    excluded = _runtime_exclusions()
    if preflight_action is None:
        from .workflow import preflight_action as execute_preflight

        preflight = lambda root, identity, environment: execute_preflight(
            repo_root,
            entry,
            root,
            identity,
            environment,
        )
    else:
        preflight = preflight_action
    preflight_exit = run_stage(
        repo_root,
        ROOTS["preflight"],
        "p1-h1-preflight",
        files,
        preflight,
        excluded,
        lambda: formal_runtime_audit(repo_root, (ROOTS["preflight"], TRANSPORT_ROOT)),
    )
    if preflight_exit != 0:
        return 1
    identity = source_identity(repo_root, files)
    valid = preflight_valid(repo_root, identity)
    if not valid["passed"]:
        print(json.dumps({"status": "REFUSE_P1_H1_PREFLIGHT_NOT_SEALED_PASS", "passed": False, "details": valid}, ensure_ascii=False), flush=True)
        return 2
    if (repo_root / ROOTS["development"]).exists():
        print(json.dumps({"status": "REFUSE_P1_H1_DEVELOPMENT_ROOT_APPEARED", "passed": False}, ensure_ascii=False), flush=True)
        return 2
    if development_action is None:
        from .workflow import development_action as execute_development

        development = lambda root, stage_identity, environment: execute_development(
            repo_root,
            root,
            stage_identity,
            environment,
            preflight_validation=valid,
        )
    else:
        development = development_action
    return run_stage(
        repo_root,
        ROOTS["development"],
        "p1-h1-development",
        files,
        development,
        excluded,
        lambda: formal_runtime_audit(repo_root, (ROOTS["preflight"], ROOTS["development"], TRANSPORT_ROOT)),
    )


# Keep the audit names used by the earlier single-use runner recognizable to
# callers, while retaining H1-owned implementations and no NR1 import.
_audit_process_rows = audit_process_rows
_contract_hash_audit = contract_hash_audit
_preflight_valid = preflight_valid


__all__ = [
    "audit_process_rows", "contract_hash_audit", "contract_readiness_audit", "fixed_root_freshness", "formal_entry_audit",
    "formal_process_audit", "formal_runtime_audit", "history_replay", "preflight_valid",
    "launch_readiness_audit", "run_all", "source_files", "successor_audit", "transport_audit",
]
