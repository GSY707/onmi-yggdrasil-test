from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from yggdrasil_v2.r1_revalidation.h1 import runner


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_fixed_roots_and_successor_policy_allow_nonformal_but_reject_formal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "ROOTS", {"preflight": Path("artifacts/p1-h1-preflight"), "development": Path("artifacts/p1-h1-development")})
    monkeypatch.setattr(runner, "TRANSPORT_ROOT", Path("tmp/p1-h1-transport"))
    assert runner.fixed_root_freshness(tmp_path)["passed"] is True
    nonformal = tmp_path / "artifacts/v2-r1r/p1-h1-nonformal-probe-1"
    formal = tmp_path / "artifacts/v2-r1r/p1-f1-execution-1"
    current = tmp_path / "artifacts/v2-r1r/p1-h1-preflight-1"
    nonformal.mkdir(parents=True)
    formal.mkdir(parents=True)
    current.mkdir(parents=True)
    assert runner.successor_audit(tmp_path)["passed"] is False
    audit = runner.successor_audit(tmp_path, (Path("artifacts/v2-r1r/p1-h1-preflight-1"),))
    assert audit["passed"] is False
    assert "artifacts/v2-r1r/p1-f1-execution-1" in audit["forbidden_roots"]


def test_process_audit_accepts_one_chain_and_rejects_external_or_successor() -> None:
    command = "python experiments/v2_r1_revalidation.py run-p1-h1"
    rows = [
        {"pid": 10, "parent_pid": 0, "name": "powershell", "command_line": command},
        {"pid": 20, "parent_pid": 10, "name": "python", "command_line": command},
    ]
    assert runner.audit_process_rows(rows, 0, 20)["passed"] is True
    external = {"pid": 30, "parent_pid": 0, "name": "python", "command_line": command}
    assert runner.audit_process_rows([*rows, external], 0, 20)["passed"] is False
    successor = {"pid": 40, "parent_pid": 0, "name": "python", "command_line": "run-p1-f1"}
    result = runner.audit_process_rows([*rows, successor], 0, 20)
    assert result["passed"] is False
    assert result["successor_matches"] == [successor]


def test_transport_audit_requires_h1_binding_and_real_redirection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = tmp_path / "transport"
    transport.mkdir()
    monkeypatch.setattr(runner, "TRANSPORT_ROOT", Path("transport"))
    monkeypatch.setenv("YGGDRASIL_H1_ATTEMPT_ID", runner.ATTEMPT_ID)
    monkeypatch.setenv("YGGDRASIL_H1_TRANSPORT_ROOT", str(transport.resolve()))
    launch = {
        "attempt_id": runner.ATTEMPT_ID,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "cwd": tmp_path.resolve().as_posix(),
        "argv": sys.argv,
        "stdout": (transport / "stdout.log").resolve().as_posix(),
        "stderr": (transport / "stderr.log").resolve().as_posix(),
    }
    (transport / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
    with (transport / "stdout.log").open("x", encoding="utf-8") as stdout_handle:
        with (transport / "stderr.log").open("x", encoding="utf-8") as stderr_handle:
            with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
                result = runner.transport_audit(tmp_path)
    assert result["passed"] is True
    launch["attempt_id"] = "wrong"
    (transport / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
    assert runner.transport_audit(tmp_path)["passed"] is False


def test_cli_audit_requires_direct_h1_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="run-p1-h1", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner._cli_audit(tmp_path)["passed"] is True

    def old_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="run-p1-h1 run-p1-nr1", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", old_run)
    assert runner._cli_audit(tmp_path)["passed"] is False


def test_run_all_stops_after_preflight_failure_without_development(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "source_files", lambda _root: (Path("pyproject.toml"),))
    monkeypatch.setattr(runner, "formal_entry_audit", lambda _root: {"passed": True})
    calls: list[str] = []

    def fake_stage(_repo: Path, _root: Path, stage: str, *_args: object, **_kwargs: object) -> int:
        calls.append(stage)
        return 1

    monkeypatch.setattr(runner, "run_stage", fake_stage)
    assert runner.run_all(tmp_path) == 1
    assert calls == ["p1-h1-preflight"]


def test_run_all_requires_sealed_preflight_before_development(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "source_files", lambda _root: (Path("pyproject.toml"),))
    monkeypatch.setattr(runner, "formal_entry_audit", lambda _root: {"passed": True})
    monkeypatch.setattr(runner, "source_identity", lambda *_args: "IDENTITY")
    monkeypatch.setattr(runner, "preflight_valid", lambda _root, _identity: {"passed": True})
    calls: list[str] = []

    def fake_stage(_repo: Path, _root: Path, stage: str, *_args: object, **_kwargs: object) -> int:
        calls.append(stage)
        return 0

    monkeypatch.setattr(runner, "run_stage", fake_stage)
    assert runner.run_all(tmp_path) == 0
    assert calls == ["p1-h1-preflight", "p1-h1-development"]
