from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from yggdrasil_v2.v2_a.closure_c1r import runner


def _fake_hooks(events: list[str], *, fail_at: str | None = None) -> runner.StageHooks:
    def stage(name: str, **extra: object):
        def hook(_context: runner.RunContext):
            events.append(name)
            result = {"passed": name != fail_at}
            result.update(extra)
            return result

        return hook

    def gate(_context: runner.RunContext, gate_name: str):
        events.append(gate_name)
        return {"passed": gate_name != fail_at}

    return runner.StageHooks(
        fresh_init_parity=stage("fresh_init_parity"),
        answer_only_overfit32=stage("answer_only_overfit32"),
        stage_a_train=stage("stage_a_train", completed_updates=runner.STAGE_A_UPDATES),
        stage_a_gate=gate,
        freeze_deployment=stage("freeze_deployment"),
        cache_train_trajectories=stage("cache_train_trajectories"),
        frozen_probe_overfit=stage("frozen_probe_overfit"),
        stage_b_train=stage("stage_b_train", completed_updates=runner.STAGE_B_UPDATES),
        g007_trace=stage("g007_trace"),
        strip_reload=stage("strip_reload"),
    )


@pytest.fixture
def cuda_for_state_machine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_name",
        lambda _index: "NVIDIA GeForce RTX 4070 Laptop GPU",
    )
    monkeypatch.setattr(runner.torch.cuda, "get_device_capability", lambda _index: (8, 9))


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "root", tmp_path / "lease.jsonl"


def _pins(_root: Path) -> dict[str, object]:
    return {"passed": True}


def test_formal_runs_in_contract_order_and_seals(tmp_path: Path, cuda_for_state_machine):
    events: list[str] = []
    root, lease = _paths(tmp_path)
    result = runner.run_formal(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        hooks=_fake_hooks(events),
        require_preflight=False,
        pin_auditor=_pins,
    )

    assert result["status"] == "PASS_V2_A_C1R"
    assert result["passed"] is True
    assert events == [
        "fresh_init_parity",
        "answer_only_overfit32",
        "stage_a_train",
        "G004",
        "G005",
        "G006",
        "G008",
        "G009",
        "G010",
        "freeze_deployment",
        "cache_train_trajectories",
        "frozen_probe_overfit",
        "stage_b_train",
        "g007_trace",
        "strip_reload",
    ]
    assert result["authorizes"] == "two additional fresh C1R seeds only"
    assert result["c2_authorized"] is False
    assert result["v2a_passed"] is False
    assert result["lease_path"] == lease.as_posix()
    assert json.loads((root / "evidence-seal.json").read_text(encoding="utf-8"))
    assert runner._audit_evidence_seal(root)["passed"] is True


def test_stage_a_gate_fail_stops_before_later_stages(tmp_path: Path, cuda_for_state_machine):
    events: list[str] = []
    root, lease = _paths(tmp_path)
    result = runner.run_formal(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        hooks=_fake_hooks(events, fail_at="G005"),
        require_preflight=False,
        pin_auditor=_pins,
    )

    assert result["passed"] is False
    assert result["stopped_at"] == "G005"
    assert events == ["fresh_init_parity", "answer_only_overfit32", "stage_a_train", "G004", "G005"]
    assert "stage_b_train" not in events
    assert result["unrun_after"] == "G005"


def test_frozen_probe_failure_stops_before_stage_b(tmp_path: Path, cuda_for_state_machine):
    events: list[str] = []
    root, lease = _paths(tmp_path)
    result = runner.run_formal(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        hooks=_fake_hooks(events, fail_at="frozen_probe_overfit"),
        require_preflight=False,
        pin_auditor=_pins,
    )

    assert result["stopped_at"] == "frozen_probe_overfit"
    assert "stage_b_train" not in events


def test_formal_single_use_refuses_second_launch(tmp_path: Path, cuda_for_state_machine):
    events: list[str] = []
    root, lease = _paths(tmp_path)
    first = runner.run_formal(
        Path.cwd(), output_root=root, lease_path=lease, hooks=_fake_hooks(events), require_preflight=False, pin_auditor=_pins
    )
    second = runner.run_formal(
        Path.cwd(), output_root=root, lease_path=lease, hooks=_fake_hooks(events), require_preflight=False, pin_auditor=_pins
    )

    assert first["passed"] is True
    assert second["status"] == "REFUSE_V2_A_C1R_SINGLE_USE"
    assert len(events) == 15


def test_formal_requires_sealed_preflight_by_default(
    tmp_path: Path,
    cuda_for_state_machine,
    monkeypatch: pytest.MonkeyPatch,
):
    root, lease = _paths(tmp_path)
    monkeypatch.setattr(runner, "PREFLIGHT_ROOT", tmp_path / "missing-preflight")
    result = runner.run_formal(
        # Keep the prerequisite check independent of whether the repository's
        # real single-use preflight has already been sealed.
        Path.cwd(), output_root=root, lease_path=lease, hooks=_fake_hooks([])
    )

    assert result["status"] == "REFUSE_V2_A_C1R_PREFLIGHT_PREREQUISITE"
    assert not root.exists()
    assert not lease.exists()


def test_preflight_is_zero_training_and_single_use(tmp_path: Path):
    root, lease = _paths(tmp_path)
    calls: list[str] = []

    def pins(_repo_root: Path):
        calls.append("pins")
        return {"passed": True}

    def smoke(_repo_root: Path, _device: str):
        calls.append("smoke")
        return {"passed": True, "optimizer_steps": 0, "model_writes": 0}

    first = runner.run_preflight(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        pin_auditor=pins,
        model_smoke=smoke,
        device_auditor=lambda _device: {"passed": True, "selected_device_name": "NVIDIA GeForce RTX 4070 Laptop GPU"},
    )
    second = runner.run_preflight(
        Path.cwd(), output_root=root, lease_path=lease, pin_auditor=pins, model_smoke=smoke,
        device_auditor=lambda _device: {"passed": True, "selected_device_name": "NVIDIA GeForce RTX 4070 Laptop GPU"},
    )

    assert first["status"] == "PASS_V2_A_C1R_PREFLIGHT"
    assert first["training_started"] is False
    assert first["optimizer_steps"] == 0
    assert first["model_writes"] == 0
    assert calls == ["pins", "smoke"]
    assert second["status"] == "REFUSE_V2_A_C1R_PREFLIGHT_SINGLE_USE"


def test_preflight_rejects_nonzero_smoke_mutation(tmp_path: Path):
    root, lease = _paths(tmp_path)
    result = runner.run_preflight(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        pin_auditor=lambda _root: {"passed": True},
        model_smoke=lambda _root, _device: {"passed": True, "optimizer_steps": 1},
        device_auditor=lambda _device: {"passed": True, "selected_device_name": "NVIDIA GeForce RTX 4070 Laptop GPU"},
    )

    assert result["status"] == "INCOMPLETE_V2_A_C1R_PREFLIGHT"
    assert result["passed"] is False
    assert result["preflight_zero_training"] is False


def test_production_hook_builder_has_real_stage_adapters():
    hooks = runner.production_hooks(Path.cwd(), device="cpu")
    assert hooks.fresh_init_parity is not runner._not_implemented
    assert hooks.answer_only_overfit32 is not runner._not_implemented
    assert hooks.stage_a_train is not runner._not_implemented
    assert hooks.stage_a_gate is not None
    assert hooks.stage_b_train is not runner._not_implemented
    assert hooks.g007_trace is not runner._not_implemented


def test_cli_formal_uses_runner_default_production_hooks(monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("c1r_cli_test", Path("experiments/v2_a_closure_c1r.py"))
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    captured: dict[str, object] = {}

    def fake_formal(root, *, device):
        captured.update(root=root, device=device)
        return {"exit_code": 1, "passed": False, "status": "mock"}

    monkeypatch.setattr(cli, "run_formal", fake_formal)
    assert cli.main(["run-formal", "--device", "cuda"]) == 1
    assert captured["device"] == "cuda"


def test_invariance_allows_tiny_logit_drift_but_requires_same_prediction():
    baseline = {"x": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    changed = {"x": [1.0 - 1.0e-7, 1.0e-7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    report = runner._invariance(baseline, changed, tolerance=1.0e-5)
    assert report["passed"] is True
    assert report["prediction_invariance"] == 1.0


def test_failed_formal_pin_stops_before_any_stage(tmp_path: Path, cuda_for_state_machine):
    events: list[str] = []
    root, lease = _paths(tmp_path)
    result = runner.run_formal(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        hooks=_fake_hooks(events),
        require_preflight=False,
        pin_auditor=lambda _root: {"passed": False},
    )
    assert result["status"] == "INCOMPLETE_V2_A_C1R_PREREQUISITES"
    assert result["stopped_at"] == "prerequisites"
    assert events == []


def test_device_audit_rejects_non_registered_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda _index: "NVIDIA A100")
    monkeypatch.setattr(runner.torch.cuda, "get_device_capability", lambda _index: (8, 0))
    report = runner._cuda_device_audit("cuda")
    assert report["passed"] is False
    assert report["selected_device_name"] == "NVIDIA A100"


def test_coverage_schedule_audit_requires_rows_and_hash():
    plan = {
        "schedule": [{"update": 1, "epoch": 0, "example_ids": ["ERE-1", "CPS-1"]}],
        "schedule_sha256": "ABC",
    }
    report = {
        "rows": [{"update": 1, "epoch": 0, "example_ids": ["ERE-1", "CPS-1"]}],
        "sha256": "ABC",
    }
    assert runner._coverage_schedule_audit(plan, report)["passed"] is True
    report["sha256"] = "DEF"
    mismatch = runner._coverage_schedule_audit(plan, report)
    assert mismatch["rows_match"] is True
    assert mismatch["hash_match"] is False
    assert mismatch["passed"] is False


def test_arithmetic_failure_is_sealed_instead_of_escaping(
    tmp_path: Path, cuda_for_state_machine
):
    events: list[str] = []
    hooks = _fake_hooks(events)

    def explode(_context: runner.RunContext):
        raise FloatingPointError("non-finite gradient")

    hooks.stage_a_train = explode
    root, lease = _paths(tmp_path)
    result = runner.run_formal(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        hooks=hooks,
        require_preflight=False,
        pin_auditor=_pins,
    )
    assert result["status"] == "CRASH_V2_A_C1R"
    assert result["passed"] is False
    assert "FloatingPointError" in result["reason"]
    assert runner._audit_evidence_seal(root)["passed"] is True


def test_programming_failure_is_sealed_instead_of_escaping(
    tmp_path: Path, cuda_for_state_machine
):
    hooks = _fake_hooks([])

    def explode(_context: runner.RunContext):
        raise NameError("missing stage variable")

    hooks.stage_a_train = explode
    root, lease = _paths(tmp_path)
    result = runner.run_formal(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        hooks=hooks,
        require_preflight=False,
        pin_auditor=_pins,
    )
    assert result["status"] == "CRASH_V2_A_C1R"
    assert "NameError" in result["reason"]
    assert runner._audit_evidence_seal(root)["passed"] is True
