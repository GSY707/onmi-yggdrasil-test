from __future__ import annotations

from pathlib import Path

from yggdrasil_v2.v2_a.closure_c1s import artifacts, runner


def _pins(_repo_root: Path):
    return {"passed": True, "roots": {"synthetic": {"passed": True}}}


def _device(_device: str):
    return {"passed": True, "selected_device_name": "NVIDIA GeForce RTX 4070 Laptop GPU"}


def _structure(_repo_root: Path, _device_name: str):
    return {"passed": True, "checks": {"synthetic": True}}


def _cuda(_repo_root: Path, _device_name: str):
    return {"passed": True, "optimizer_steps": 0, "model_writes": 0, "backward_smoke_only": True}


def test_s0_pass_is_sealed_single_use_and_authorizes_only_s1(tmp_path: Path):
    root = tmp_path / "root"
    lease = tmp_path / "lease.jsonl"
    first = runner.run_s0_preflight(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        pin_auditor=_pins,
        device_auditor=_device,
        structure_stage=_structure,
        cuda_stage=_cuda,
    )
    second = runner.run_s0_preflight(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        pin_auditor=_pins,
        device_auditor=_device,
        structure_stage=_structure,
        cuda_stage=_cuda,
    )

    assert first["status"] == "PASS_V2_A_C1S_S0_QUALIFICATION"
    assert first["formal"] is False
    assert first["training_started"] is False
    assert first["optimizer_steps"] == 0
    assert first["model_writes"] == 0
    assert first["s1_status"] == "AUTHORIZED_NOT_RUN"
    assert first["s2_status"] == "NOT_AUTHORIZED"
    assert first["c2_authorized"] is False
    assert first["v2a_passed"] is False
    assert artifacts.audit_evidence_seal(root)["passed"] is True
    assert second["status"] == "REFUSE_V2_A_C1S_S0_SINGLE_USE"


def test_s0_structure_failure_stops_before_cuda_and_authorizes_nothing(tmp_path: Path):
    calls: list[str] = []

    def structure(_repo_root: Path, _device_name: str):
        calls.append("structure")
        return {"passed": False, "checks": {"synthetic": False}}

    def cuda(_repo_root: Path, _device_name: str):
        calls.append("cuda")
        return _cuda(_repo_root, _device_name)

    result = runner.run_s0_preflight(
        Path.cwd(),
        output_root=tmp_path / "root",
        lease_path=tmp_path / "lease.jsonl",
        pin_auditor=_pins,
        device_auditor=_device,
        structure_stage=structure,
        cuda_stage=cuda,
    )

    assert calls == ["structure"]
    assert result["status"] == "FAIL_V2_A_C1S_S0_QUALIFICATION"
    assert result["authorizes"] == "nothing"
    assert result["s1_status"] == "NOT_AUTHORIZED"


def test_s0_prerequisite_refusal_does_not_claim_root_or_lease(tmp_path: Path):
    root = tmp_path / "root"
    lease = tmp_path / "lease.jsonl"
    result = runner.run_s0_preflight(
        Path.cwd(),
        output_root=root,
        lease_path=lease,
        pin_auditor=lambda _root: {"passed": False},
        device_auditor=_device,
        structure_stage=_structure,
        cuda_stage=_cuda,
    )

    assert result["status"] == "REFUSE_V2_A_C1S_S0_PIN"
    assert not root.exists()
    assert not lease.exists()


def test_s0_rejects_nonzero_accounting(tmp_path: Path):
    result = runner.run_s0_preflight(
        Path.cwd(),
        output_root=tmp_path / "root",
        lease_path=tmp_path / "lease.jsonl",
        pin_auditor=_pins,
        device_auditor=_device,
        structure_stage=_structure,
        cuda_stage=lambda _root, _device_name: {"passed": True, "optimizer_steps": 1, "model_writes": 0},
    )

    assert result["status"] == "FAIL_V2_A_C1S_S0_QUALIFICATION"
    assert result["optimizer_steps"] == 1
    assert result["authorizes"] == "nothing"

