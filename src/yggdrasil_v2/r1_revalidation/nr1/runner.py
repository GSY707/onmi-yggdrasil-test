from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import (
    git_identity,
    run_stage,
    sha256,
    source_identity,
    verify_seal,
    write_json,
)
from .contract import (
    CONTRACT_VERSION,
    DESIGN,
    EXECUTION,
    FROZEN_CONTRACT_HASHES,
    ROOTS,
    TRANSPORT_ROOT,
    V8L_ROOTS,
    contract_manifest,
)
from .qualification import run_qualification_bundle


def source_files(repo_root: Path) -> tuple[Path, ...]:
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
    }
    for directory in (
        Path("src/yggdrasil_v2/r1_revalidation/nr1"),
        Path("tests/v2_r1r_p1_nr1"),
    ):
        base = repo_root / directory
        if not base.is_dir():
            raise FileNotFoundError(f"missing active NR1 source directory: {directory}")
        for pattern in ("*.py", "*.json"):
            explicit.update(
                path.relative_to(repo_root) for path in base.rglob(pattern)
            )
    files = tuple(sorted(explicit, key=lambda path: path.as_posix()))
    missing = [path.as_posix() for path in files if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing NR1 source inputs: {missing}")
    return files


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_exclusions() -> tuple[Path, ...]:
    return (*ROOTS.values(), TRANSPORT_ROOT)


def fixed_root_freshness(repo_root: Path) -> dict[str, Any]:
    existing = [
        root.as_posix() for root in ROOTS.values() if (repo_root / root).exists()
    ]
    return {"existing_roots": existing, "passed": not existing}


def _contract_hash_audit(repo_root: Path) -> dict[str, Any]:
    actual = {
        relative: sha256(repo_root / relative)
        for relative in FROZEN_CONTRACT_HASHES
    }
    checks = {
        relative: expected != "TO_BE_FROZEN" and actual[relative] == expected
        for relative, expected in FROZEN_CONTRACT_HASHES.items()
    }
    return {
        "expected": FROZEN_CONTRACT_HASHES,
        "actual": actual,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _v8l_audit(repo_root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, specification in V8L_ROOTS.items():
        root = repo_root / specification["root"]
        result_path = root / "result.json"
        seal_path = root / "evidence-seal.json"
        result = _load(result_path) if result_path.is_file() else {}
        actual_seal = sha256(seal_path) if seal_path.is_file() else None
        rows[name] = {
            "root": specification["root"].as_posix(),
            "status": result.get("status"),
            "expected_status": specification["status"],
            "seal_sha256": actual_seal,
            "expected_seal_sha256": specification["seal_sha256"],
            "tree_seal_valid": verify_seal(root),
            "passed": (
                result.get("status") == specification["status"]
                and actual_seal == specification["seal_sha256"]
                and verify_seal(root)
            ),
        }
    result = _load(repo_root / V8L_ROOTS["qualification"]["root"] / "result.json")
    boundary = {
        "fresh_p1_v9_authorized": result.get("fresh_p1_v9_authorized"),
        "p1_completed": result.get("p1_completed"),
        "p2_eligible": result.get("p2_eligible"),
        "p2_started": result.get("p2_started"),
    }
    return {
        "roots": rows,
        "boundary": boundary,
        "passed": (
            all(row["passed"] for row in rows.values())
            and rows["qualification"]["status"] == "FAIL_P1_V8L_BOOTSTRAP"
            and boundary
            == {
                "fresh_p1_v9_authorized": False,
                "p1_completed": False,
                "p2_eligible": False,
                "p2_started": False,
            }
        ),
    }


def successor_audit(repo_root: Path) -> dict[str, Any]:
    forbidden: list[str] = []
    prefixes = ("p1-h1", "p1-f1", "p1-v9", "p2")
    for relative_base in (Path("artifacts/v2-r1r"), Path("tmp")):
        base = repo_root / relative_base
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_dir() and path.name.lower().startswith(prefixes):
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
                rows.append(
                    {
                        "pid": int(parts[0]),
                        "parent_pid": int(parts[1]),
                        "name": parts[2],
                        "command_line": parts[3],
                    }
                )
        return {"returncode": completed.returncode, "rows": rows}
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    raw = json.loads(completed.stdout or "[]") if completed.returncode == 0 else []
    if isinstance(raw, dict):
        raw = [raw]
    rows = [
        {
            "pid": int(row.get("ProcessId") or 0),
            "parent_pid": int(row.get("ParentProcessId") or 0),
            "name": row.get("Name") or "",
            "command_line": row.get("CommandLine") or "",
        }
        for row in raw
    ]
    return {
        "returncode": completed.returncode,
        "stderr": completed.stderr,
        "rows": rows,
    }


def _audit_process_rows(
    rows: list[dict[str, Any]], snapshot_returncode: int, current_pid: int
) -> dict[str, Any]:
    by_pid = {row["pid"]: row for row in rows}
    allowed_chain = {current_pid}
    cursor = current_pid
    while cursor in by_pid:
        parent = by_pid[cursor]["parent_pid"]
        if not parent or parent in allowed_chain:
            break
        allowed_chain.add(parent)
        cursor = parent
    matches = []
    successor_matches = []
    for row in rows:
        normalized = row["command_line"].lower().replace("\\", "/")
        if "v2_r1_revalidation.py" in normalized and "run-p1-nr1" in normalized:
            matches.append(row)
        if any(token in normalized for token in ("run-p1-h1", "run-p1-f1", "run-p1-v9", "run-p2")):
            successor_matches.append(row)
    external = [row for row in matches if row["pid"] not in allowed_chain]
    current_present = any(row["pid"] == current_pid for row in matches)
    return {
        "snapshot_returncode": snapshot_returncode,
        "current_pid": current_pid,
        "allowed_ancestor_chain": sorted(allowed_chain),
        "formal_matches": matches,
        "external_formal_matches": external,
        "successor_matches": successor_matches,
        "passed": (
            snapshot_returncode == 0
            and current_present
            and not external
            and not successor_matches
        ),
    }


def formal_process_audit() -> dict[str, Any]:
    snapshot = _process_snapshot()
    return _audit_process_rows(snapshot["rows"], snapshot["returncode"], os.getpid())


def transport_audit(repo_root: Path) -> dict[str, Any]:
    root = (repo_root / TRANSPORT_ROOT).resolve()
    launch_path = root / "launch.json"
    launch = _load(launch_path) if launch_path.is_file() else {}
    stdout_path = (root / "stdout.log").resolve()
    stderr_path = (root / "stderr.log").resolve()
    stdout_name = Path(str(getattr(sys.stdout, "name", ""))).resolve()
    stderr_name = Path(str(getattr(sys.stderr, "name", ""))).resolve()
    checks = {
        "attempt_id": os.environ.get("YGGDRASIL_NR1_ATTEMPT_ID") == "p1-nr1-20260817-1",
        "transport_environment": Path(
            os.environ.get("YGGDRASIL_NR1_TRANSPORT_ROOT", ".")
        ).resolve()
        == root,
        "launch_present": launch_path.is_file(),
        "launch_attempt_id": launch.get("attempt_id") == "p1-nr1-20260817-1",
        "launch_pid": launch.get("pid") == os.getpid(),
        "launch_parent_pid": launch.get("parent_pid") == os.getppid(),
        "launch_cwd": Path(launch.get("cwd", ".")).resolve() == repo_root.resolve(),
        "stdout_redirected": stdout_name == stdout_path and not sys.stdout.isatty(),
        "stderr_redirected": stderr_name == stderr_path and not sys.stderr.isatty(),
        "launch_stdout_bound": Path(launch.get("stdout", ".")).resolve() == stdout_path,
        "launch_stderr_bound": Path(launch.get("stderr", ".")).resolve() == stderr_path,
        "formal_argv": launch.get("argv") == sys.argv,
    }
    return {
        "transport_root": TRANSPORT_ROOT.as_posix(),
        "launch_sha256": sha256(launch_path) if launch_path.is_file() else None,
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def formal_runtime_audit(repo_root: Path) -> dict[str, Any]:
    process = formal_process_audit()
    transport = transport_audit(repo_root)
    successors = successor_audit(repo_root)
    return {
        "process": process,
        "transport": transport,
        "successors": successors,
        "passed": process["passed"] and transport["passed"] and successors["passed"],
    }


def formal_entry_audit(repo_root: Path) -> dict[str, Any]:
    roots = fixed_root_freshness(repo_root)
    runtime = formal_runtime_audit(repo_root)
    return {
        "fixed_roots": roots,
        "runtime": runtime,
        "passed": roots["passed"] and runtime["passed"],
    }


def _prediction_tests(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/v2_r1r_p1_nr1",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": (
            f"{sys.executable} -m pytest -q -p no:cacheprovider "
            "tests/v2_r1r_p1_nr1"
        ),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def _cli_audit(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "experiments/v2_r1_revalidation.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    text = completed.stdout + completed.stderr
    required = {"run-p1-nr1"}
    forbidden = {
        "preflight-p1-nr1",
        "qualify-p1-nr1",
        "run-p1-causal-state-ladder",
        "run-p1-decision-witness",
        "run-p2",
    }
    return {
        "returncode": completed.returncode,
        "required": {value: value in text for value in sorted(required)},
        "forbidden": {value: value in text for value in sorted(forbidden)},
        "passed": (
            completed.returncode == 0
            and all(value in text for value in required)
            and all(value not in text for value in forbidden)
        ),
    }


def _preflight_action(
    repo_root: Path,
    root: Path,
    identity: str,
    environment: dict[str, Any],
    initial_entry: dict[str, Any],
) -> dict[str, Any]:
    files = source_files(repo_root)
    process_before = formal_process_audit()
    transport = transport_audit(repo_root)
    contract = _contract_hash_audit(repo_root)
    v8l = _v8l_audit(repo_root)
    tests = _prediction_tests(repo_root)
    cli = _cli_audit(repo_root)
    process_after = formal_process_audit()
    successors = successor_audit(repo_root)
    qualification_fresh = not (repo_root / ROOTS["qualification"]).exists()
    checks = {
        "initial_entry_gate": initial_entry["passed"],
        "contract_hashes": contract["passed"],
        "v8l_history_sealed": v8l["passed"],
        "prediction_tests": tests["passed"],
        "cli_single_entry_direct_switch": cli["passed"],
        "formal_process_chain_before": process_before["passed"],
        "formal_process_chain_after": process_after["passed"],
        "transport_redirect": transport["passed"],
        "qualification_root_fresh": qualification_fresh,
        "successor_absence": successors["passed"],
        "git_identity_captured": environment["git_identity_start"].get("passed") is True,
        "source_identity_stable": identity == source_identity(repo_root, files),
    }
    write_json(root / "contract-manifest.json", contract_manifest())
    write_json(root / "contract-hash-audit.json", contract)
    write_json(root / "v8l-source-audit.json", v8l)
    write_json(root / "prediction-tests.json", tests)
    write_json(root / "cli-audit.json", cli)
    write_json(root / "process-audit-before.json", process_before)
    write_json(root / "process-audit-after.json", process_after)
    write_json(root / "transport-audit.json", transport)
    write_json(root / "successor-audit.json", successors)
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-nr1.preflight-result.v2",
        "contract_version": CONTRACT_VERSION,
        "status": (
            "PASS_P1_NR1_PREFLIGHT" if all(checks.values()) else "FAIL_P1_NR1_PREFLIGHT"
        ),
        "passed": all(checks.values()),
        "source_identity": identity,
        "checks": checks,
        "formal_entry_audit": initial_entry,
        "process_audit_before": process_before,
        "process_audit_after": process_after,
        "transport_audit": transport,
        "successor_audit": successors,
    }


def _preflight_valid(repo_root: Path, identity: str) -> dict[str, Any]:
    root = repo_root / ROOTS["preflight"]
    result_path = root / "result.json"
    if not result_path.is_file():
        return {"passed": False, "reason": "preflight result missing"}
    result = _load(result_path)
    checks = {
        "status": result.get("status") == "PASS_P1_NR1_PREFLIGHT",
        "passed": result.get("passed") is True,
        "source_identity": result.get("source_identity") == identity,
        "runner_integrity": result.get("runner_integrity", {}).get("passed") is True,
        "seal": verify_seal(root),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _qualification_action(
    repo_root: Path,
    root: Path,
    identity: str,
    _environment: dict[str, Any],
) -> dict[str, Any]:
    preflight = _preflight_valid(repo_root, identity)
    runtime_before = formal_runtime_audit(repo_root)
    contract = _contract_hash_audit(repo_root)
    bundle = run_qualification_bundle()
    runtime_after = formal_runtime_audit(repo_root)
    gates = {
        "N01_contract_source_history_integrity": (
            preflight["passed"] and contract["passed"]
        ),
        **bundle["gates"],
        "N07_stop_process_artifact_integrity": (
            runtime_before["passed"] and runtime_after["passed"]
        ),
    }
    passed = all(gates.values())
    write_json(root / "case-manifest.json", bundle["case_manifest"])
    write_json(root / "hand-fixture-probe.json", bundle["hand_fixture_probe"])
    write_json(root / "independence-audit.json", bundle["independence"])
    write_json(
        root / "metric-ledger.json",
        {
            "numeric": bundle["numeric_metrics"],
            "relation": bundle["relation_metrics"],
        },
    )
    write_json(root / "metamorphic-ledger.json", bundle["metamorphic"])
    write_json(root / "fault-matrix.json", bundle["fault_matrix"])
    write_json(root / "runtime-audit-before.json", runtime_before)
    write_json(root / "runtime-audit-after.json", runtime_after)
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-nr1.qualification-result.v2",
        "contract_version": CONTRACT_VERSION,
        "status": (
            "PASS_P1_NR1_MEASUREMENT_QUALIFICATION"
            if passed
            else "FAIL_P1_NR1_MEASUREMENT_QUALIFICATION"
        ),
        "passed": passed,
        "source_identity": identity,
        "formal_seed": bundle["seed"],
        "training_performed": bundle["training_performed"],
        "audit_statistics_fit": bundle["audit_statistics_fit"],
        "preflight_validation": preflight,
        "gates": gates,
        "hand_fixture_probe": bundle["hand_fixture_probe"],
        "numeric_metrics": bundle["numeric_metrics"],
        "relation_metrics": bundle["relation_metrics"],
        "metamorphic": bundle["metamorphic"],
        "fault_matrix": bundle["fault_matrix"],
        "case_manifest_summary": {
            "numeric_counts": bundle["case_manifest"]["numeric"]["counts"],
            "relation_counts": bundle["case_manifest"]["relation"]["counts"],
            "relation_topology": bundle["case_manifest"]["relation"]["topology"],
            "overlap": bundle["case_manifest"]["overlap"],
        },
        "runtime_audit_before": runtime_before,
        "runtime_audit_after": runtime_after,
        "p1_h1_design_authorized": passed,
        "p1_completed": False,
        "p2_eligible": False,
        "p2_started": False,
    }


def _refuse_before_mutation(reason: str, details: dict[str, Any]) -> int:
    print(
        json.dumps(
            {
                "status": "REFUSE_P1_NR1_BEFORE_MUTATION",
                "passed": False,
                "reason": reason,
                "details": details,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 2


def run_all(repo_root: Path) -> int:
    files = source_files(repo_root)
    entry = formal_entry_audit(repo_root)
    if not entry["passed"]:
        return _refuse_before_mutation(
            "fixed roots are not fresh or formal transport/process binding is invalid",
            entry,
        )
    excluded = _runtime_exclusions()
    preflight_exit = run_stage(
        repo_root,
        ROOTS["preflight"],
        "p1-nr1-preflight",
        files,
        lambda root, identity, environment: _preflight_action(
            repo_root, root, identity, environment, entry
        ),
        excluded,
        lambda: formal_runtime_audit(repo_root),
    )
    if preflight_exit != 0:
        return 1
    identity = source_identity(repo_root, files)
    preflight = _preflight_valid(repo_root, identity)
    if not preflight["passed"]:
        return _refuse_before_mutation(
            "preflight is not sealed PASS for the current source identity", preflight
        )
    if (repo_root / ROOTS["qualification"]).exists():
        return _refuse_before_mutation(
            "qualification root appeared before qualification launch",
            {"root": ROOTS["qualification"].as_posix()},
        )
    return run_stage(
        repo_root,
        ROOTS["qualification"],
        "p1-nr1-qualification",
        files,
        lambda root, stage_identity, environment: _qualification_action(
            repo_root, root, stage_identity, environment
        ),
        excluded,
        lambda: formal_runtime_audit(repo_root),
    )
