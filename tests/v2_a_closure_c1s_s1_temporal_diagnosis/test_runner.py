from __future__ import annotations

from pathlib import Path

from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis import contract
from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis import runner


def test_preflight_refuses_existing_fixed_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(contract, "PREFLIGHT_ROOT", Path("preflight"))
    monkeypatch.setattr(contract, "PREFLIGHT_LEASE", Path("preflight.lease"))
    (tmp_path / "preflight").mkdir()
    result = runner.run_preflight(tmp_path, device="cpu")
    assert result["status"] == "REFUSED"
    assert result["authorizes"] == "nothing"


def test_diagnosis_refuses_existing_fixed_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(contract, "OUTPUT_ROOT", Path("diagnosis"))
    monkeypatch.setattr(contract, "LEASE_PATH", Path("diagnosis.lease"))
    (tmp_path / "diagnosis").mkdir()
    result = runner.run_diagnosis(tmp_path, device="cpu")
    assert result["status"] == "REFUSED"
    assert result["authorizes"] == "nothing"


def test_stage_identity_is_temporal_only() -> None:
    assert runner.STAGE == "S1-TEMPORAL-ATTRIBUTION-REPAIR"
    assert runner.PREFLIGHT_IDENTITY == f"{contract.IDENTITY}-PREFLIGHT"
    assert "FAILURE-ATTRIBUTION-20260831-1" not in contract.IDENTITY
