from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

from yggdrasil_v2.r1_revalidation.nr1.artifacts import (
    run_stage,
    verify_seal,
    write_json,
)
from yggdrasil_v2.r1_revalidation.nr1 import runner
from yggdrasil_v2.r1_revalidation.nr1.runner import (
    _audit_process_rows,
    transport_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_existing_stage_root_is_refused_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "already-exists"
    root.mkdir()
    marker = root / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    invoked = []
    exit_code = run_stage(
        REPO_ROOT,
        root,
        "unit-refusal",
        (Path("pyproject.toml"),),
        lambda *_args: invoked.append(True) or {"passed": True},
        (root,),
    )
    assert exit_code == 2
    assert invoked == []
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in root.iterdir()) == ["marker.txt"]


def test_stage_replays_snapshot_and_seals_in_system_temp(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    exit_code = run_stage(
        REPO_ROOT,
        root,
        "unit-stage",
        (Path("pyproject.toml"),),
        lambda *_args: {"status": "PASS_UNIT_STAGE", "passed": True},
        (root,),
    )
    assert exit_code == 0
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    assert result["runner_integrity"]["passed"] is True
    assert result["runner_integrity"]["source_identity_stable"] is True
    assert result["runner_integrity"]["git_identity_stable"] is True
    assert verify_seal(root) is True


def test_process_audit_accepts_one_ancestor_chain_and_rejects_external() -> None:
    command = "python experiments/v2_r1_revalidation.py run-p1-nr1"
    base = [
        {"pid": 10, "parent_pid": 0, "name": "powershell", "command_line": command},
        {"pid": 20, "parent_pid": 10, "name": "python", "command_line": command},
    ]
    assert _audit_process_rows(base, 0, 20)["passed"] is True
    external = {
        "pid": 30,
        "parent_pid": 0,
        "name": "python",
        "command_line": command,
    }
    result = _audit_process_rows([*base, external], 0, 20)
    assert result["passed"] is False
    assert [row["pid"] for row in result["external_formal_matches"]] == [30]


def test_transport_audit_requires_bound_launch_and_real_redirection(
    tmp_path: Path, monkeypatch,
) -> None:
    transport = tmp_path / "transport"
    transport.mkdir()
    monkeypatch.setattr(runner, "TRANSPORT_ROOT", transport)
    monkeypatch.setenv("YGGDRASIL_NR1_ATTEMPT_ID", "p1-nr1-20260817-1")
    monkeypatch.setenv("YGGDRASIL_NR1_TRANSPORT_ROOT", str(transport.resolve()))
    write_json(
        transport / "launch.json",
        {
            "attempt_id": "p1-nr1-20260817-1",
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "cwd": REPO_ROOT.resolve().as_posix(),
            "argv": sys.argv,
            "stdout": (transport / "stdout.log").resolve().as_posix(),
            "stderr": (transport / "stderr.log").resolve().as_posix(),
        },
    )
    with (transport / "stdout.log").open("x", encoding="utf-8") as stdout_handle:
        with (transport / "stderr.log").open("x", encoding="utf-8") as stderr_handle:
            with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(
                stderr_handle
            ):
                result = transport_audit(REPO_ROOT)
    assert result["passed"] is True

    write_json(
        transport / "launch.json",
        {
            "attempt_id": "wrong-attempt",
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "cwd": REPO_ROOT.resolve().as_posix(),
            "argv": sys.argv,
            "stdout": (transport / "wrong-stdout.log").resolve().as_posix(),
            "stderr": (transport / "wrong-stderr.log").resolve().as_posix(),
        },
    )
    with (transport / "stdout.log").open("a", encoding="utf-8") as stdout_handle:
        with (transport / "stderr.log").open("a", encoding="utf-8") as stderr_handle:
            with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(
                stderr_handle
            ):
                tampered = transport_audit(REPO_ROOT)
    assert tampered["passed"] is False
    assert tampered["checks"]["launch_attempt_id"] is False
    assert tampered["checks"]["launch_stdout_bound"] is False
    assert tampered["checks"]["launch_stderr_bound"] is False
