from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1u_s2 import contract
from yggdrasil_v2.v2_a.closure_c1u_s2.artifacts import SOURCE_FILES, source_hashes
from yggdrasil_v2.v2_a.closure_c1u_s2.model import S2Config
from yggdrasil_v2.v2_a.closure_c1u_s2 import runner
from yggdrasil_v2.v2_a.closure_c1u_s2.runner import (
    StepJournal,
    audit_s1_predecessor,
    audit_unconsumed_paths,
    inspect,
)


def test_s2_source_closure_is_complete_and_unique() -> None:
    root = Path(__file__).resolve().parents[2]
    assert len(SOURCE_FILES) == len(set(SOURCE_FILES))
    assert len(source_hashes(root)) == len(SOURCE_FILES)


def test_current_sealed_s1_predecessor_replays_before_s2() -> None:
    root = Path(__file__).resolve().parents[2]
    report = audit_s1_predecessor(root)
    assert report["passed"] is True, report.get("exception", report)
    assert report["result_sha256"] == contract.S1_RESULT_SHA256
    assert report["seal_sha256"] == contract.S1_EVIDENCE_SEAL_SHA256
    assert report["endpoint_sha256"] == contract.S1_ENDPOINT_SHA256


def test_inspection_is_read_only_and_reports_unconsumed_or_consumed_state() -> None:
    root = Path(__file__).resolve().parents[2]
    before = {
        path: (root / path).exists()
        for path in (
            contract.PREFLIGHT_ROOT,
            contract.PREFLIGHT_LEASE,
            contract.ROOT,
            contract.LEASE,
        )
    }
    report = inspect(root)
    after = {path: (root / path).exists() for path in before}
    assert report["passed"] is True
    assert report["inspection_mutations"] == 0
    assert report["authorizes"] == "nothing"
    assert before == after


def test_unconsumed_path_audit_is_fail_closed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(contract, "PREFLIGHT_ROOT", Path("tmp/preflight"))
    monkeypatch.setattr(contract, "PREFLIGHT_LEASE", Path("tmp/preflight.lease"))
    monkeypatch.setattr(contract, "ROOT", Path("artifacts/s2"))
    monkeypatch.setattr(contract, "LEASE", Path("artifacts/s2.lease"))
    assert audit_unconsumed_paths(tmp_path, stage="preflight")["passed"] is True
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "preflight.lease").write_text("claimed", encoding="utf-8")
    assert audit_unconsumed_paths(tmp_path, stage="preflight")["passed"] is False


def test_step_journal_requires_contiguous_durable_updates(tmp_path) -> None:
    path = tmp_path / "progress.jsonl"
    with StepJournal(path) as journal:
        journal({"update": 1, "loss": 2.0})
        journal({"update": 2, "loss": 1.0})
        with pytest.raises(RuntimeError, match="not contiguous"):
            journal({"update": 4, "loss": 0.5})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["update"] for row in rows] == [1, 2]


class _FixtureEncoder:
    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self.calls = 0

    def __call__(self, text: str) -> dict[str, torch.Tensor]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        self.calls += 1
        base = torch.tensor(list(digest), dtype=torch.float32).repeat(64) / 255.0
        return {
            "hidden": torch.stack((base, torch.roll(base, shifts=1))),
            "mask": torch.ones(2, dtype=torch.bool),
            "token_ids": torch.tensor([digest[0], digest[1]]),
        }

    def report(self) -> dict[str, object]:
        return {"calls": self.calls, "fixture": True, "device": self.device}

    def release(self) -> None:
        return None


def test_end_to_end_preflight_logic_with_disposable_fixture_identity(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(contract, "PREFLIGHT_IDENTITY", "TEST-C1U-S2-PREFLIGHT")
    monkeypatch.setattr(contract, "IDENTITY", "TEST-C1U-S2")
    monkeypatch.setattr(contract, "PREFLIGHT_ROOT", tmp_path / "preflight")
    monkeypatch.setattr(contract, "PREFLIGHT_LEASE", tmp_path / "preflight.lease.jsonl")
    monkeypatch.setattr(contract, "ROOT", tmp_path / "s2")
    monkeypatch.setattr(contract, "LEASE", tmp_path / "s2.lease.jsonl")
    monkeypatch.setattr(contract, "MAXIMUM_UPDATES_PER_ENDPOINT", 2)
    monkeypatch.setattr(contract, "TOTAL_FORMAL_OPTIMIZER_STEPS", 12)
    monkeypatch.setitem(contract.TRAINING, "maximum_updates", 2)

    def tiny_config(source_width: int, arm: str) -> S2Config:
        return S2Config(
            source_width=source_width,
            payload_width=16,
            address_width=64,
            ffn_width=32,
            workspace_slots=contract.WORKSPACE_SLOTS[arm],
        )

    monkeypatch.setattr(runner, "_config", tiny_config)
    original_spec = runner.TrainSpec
    monkeypatch.setattr(
        runner,
        "TrainSpec",
        lambda: replace(
            original_spec(),
            maximum_updates=2,
            warmup_updates=1,
            log_interval=1,
            pin_memory=False,
        ),
    )
    passed = lambda *args, **kwargs: {"passed": True}
    root = Path(__file__).resolve().parents[2]
    report = runner.run_preflight(
        root,
        device="cpu",
        asset_audit_fn=passed,
        device_audit_fn=passed,
        tests_fn=passed,
        encoder_factory=_FixtureEncoder,
    )
    assert report["passed"] is True, report.get("exception", report)
    assert report["formal_optimizer_steps"] == 0
    assert report["disposable_optimizer_steps"] == 32
    assert report["model_writes"] == 0
    assert report["seal_replay"]["passed"] is True
    replay = runner.audit_preflight(root)
    assert replay["passed"] is True

    formal = runner.run_s2(
        root,
        device="cpu",
        asset_audit_fn=passed,
        device_audit_fn=passed,
    )
    assert formal["status"] == "FAIL_V2_A_C1U_S2_QUALIFICATION"
    assert formal["formal_optimizer_steps"] == 12
    assert formal["model_writes"] == 6
    assert formal["seal_replay"]["passed"] is True
    terminal_replay = runner.audit_s2(root)
    assert terminal_replay["passed"] is True, [
        key for key, value in terminal_replay.get("checks", {}).items() if not value
    ]
