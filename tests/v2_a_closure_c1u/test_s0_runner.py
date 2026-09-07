from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1u import artifacts, contract
from yggdrasil_v2.v2_a.closure_c1u.model import C1UConfig
from yggdrasil_v2.v2_a.closure_c1u.s0 import (
    audit_s0_preflight,
    audit_s0_root,
    run_s0,
    run_s0_preflight,
)


class FakeEncoder:
    def __init__(self, *, device: str | torch.device) -> None:
        self.device = str(device)
        self.calls = 0
        self.tokens = 0

    def __call__(self, text: str) -> dict[str, torch.Tensor]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        tokens, width = 2 + len(text) % 3, 12
        self.calls += 1
        self.tokens += tokens
        return {
            "hidden": torch.tensor(
                [
                    [digest[(row + column) % 32] / 255.0 for column in range(width)]
                    for row in range(tokens)
                ]
            ),
            "mask": torch.ones(tokens, dtype=torch.bool),
            "token_ids": torch.tensor([digest[index] for index in range(tokens)]),
        }

    def report(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "total_tokens": self.tokens,
            "one_card_per_forward_call": True,
            "trainable_source_parameters": 0,
        }

    def release(self) -> None:
        return None


def _asset_audit() -> dict[str, object]:
    return {
        "passed": True,
        "rows": {"fixture.bin": {"passed": True, "sha256": "A" * 64}},
    }


def _device_audit(_: str | torch.device) -> dict[str, object]:
    return {"passed": True, "checks": {"fixture": True}, "device_name": "fixture"}


def _predecessor_audit(_: Path) -> dict[str, object]:
    return {
        "passed": True,
        "checks": {"sealed_fail_fixture": True},
        "result_status": contract.C1T_S1_TERMINAL_STATUS,
    }


def _prepare_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    marker = tmp_path / "frozen-source.py"
    marker.write_text("SOURCE = 'frozen'\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "SOURCE_FILES", (Path("frozen-source.py"),))
    monkeypatch.setattr(contract, "S0_PREFLIGHT_ROOT", Path("tmp/preflight"))
    monkeypatch.setattr(contract, "S0_PREFLIGHT_LEASE", Path("tmp/preflight.lease.jsonl"))
    monkeypatch.setattr(contract, "S0_ROOT", Path("artifacts/s0"))
    monkeypatch.setattr(contract, "S0_LEASE", Path("artifacts/s0.lease.jsonl"))
    return tmp_path


def _config() -> C1UConfig:
    return C1UConfig(
        source_width=12,
        payload_width=16,
        address_width=16,
        ffn_width=32,
    )


def test_preflight_then_s0_seal_and_authorization_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _prepare_repo(tmp_path, monkeypatch)
    preflight = run_s0_preflight(
        repo,
        device="cpu",
        asset_audit_fn=_asset_audit,
        device_audit_fn=_device_audit,
        encoder_factory=FakeEncoder,
        predecessor_audit_fn=_predecessor_audit,
        config_override=_config(),
    )
    assert preflight["passed"] is True
    assert preflight["authorizes"] == contract.S0_AUTHORIZED_SCOPE
    assert preflight["optimizer_steps"] == preflight["model_writes"] == 0
    assert audit_s0_preflight(repo)["passed"] is True

    result = run_s0(
        repo,
        device="cpu",
        asset_audit_fn=_asset_audit,
        device_audit_fn=_device_audit,
        encoder_factory=FakeEncoder,
        predecessor_audit_fn=_predecessor_audit,
        config_override=_config(),
    )
    assert result["passed"] is True
    assert result["authorizes"] == contract.S0_PASS_AUTHORIZATION
    assert result["s1_status"] == "CONTRACT_DESIGN_AUTHORIZED_NOT_RUN"
    assert result["optimizer_steps"] == result["model_writes"] == 0
    assert result["training_started"] is False
    assert audit_s0_root(repo)["passed"] is True

    with pytest.raises(RuntimeError, match="pre-claim audit failed"):
        run_s0(
            repo,
            device="cpu",
            asset_audit_fn=_asset_audit,
            device_audit_fn=_device_audit,
            encoder_factory=FakeEncoder,
            predecessor_audit_fn=_predecessor_audit,
            config_override=_config(),
        )


def test_preclaim_failure_does_not_consume_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _prepare_repo(tmp_path, monkeypatch)

    def failed_assets() -> dict[str, object]:
        return {"passed": False, "rows": {}}

    with pytest.raises(RuntimeError, match="pre-claim audit failed"):
        run_s0_preflight(
            repo,
            device="cpu",
            asset_audit_fn=failed_assets,
            device_audit_fn=_device_audit,
            encoder_factory=FakeEncoder,
            predecessor_audit_fn=_predecessor_audit,
            config_override=_config(),
        )
    assert not (repo / contract.S0_PREFLIGHT_ROOT).exists()
    assert not (repo / contract.S0_PREFLIGHT_LEASE).exists()


def test_zero_training_runner_has_no_optimizer_or_checkpoint_write_path() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src/yggdrasil_v2/v2_a/closure_c1u/s0.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".step(" not in source
    assert "torch.save" not in source


def test_postclaim_fault_is_crash_sealed_and_not_retriable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _prepare_repo(tmp_path, monkeypatch)
    result = run_s0_preflight(
        repo,
        device="cpu",
        asset_audit_fn=_asset_audit,
        device_audit_fn=_device_audit,
        encoder_factory=FakeEncoder,
        predecessor_audit_fn=_predecessor_audit,
        config_override=_config(),
        fault_at="after_claim",
    )
    assert result["passed"] is False
    assert result["status"] == "CRASH_V2_A_C1U_S0_PREFLIGHT"
    assert result["authorizes"] == "nothing"
    root = repo / contract.S0_PREFLIGHT_ROOT
    replay = artifacts.audit_evidence_seal(
        root,
        expected_identity=contract.S0_PREFLIGHT_IDENTITY,
        expected_stage="s0-preflight",
    )
    assert replay["passed"] is True
    with pytest.raises(RuntimeError, match="pre-claim audit failed"):
        run_s0_preflight(
            repo,
            device="cpu",
            asset_audit_fn=_asset_audit,
            device_audit_fn=_device_audit,
            encoder_factory=FakeEncoder,
            predecessor_audit_fn=_predecessor_audit,
            config_override=_config(),
        )
