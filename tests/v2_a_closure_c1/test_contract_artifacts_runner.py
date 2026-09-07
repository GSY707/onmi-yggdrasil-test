from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from yggdrasil_v2.v2_a.closure_c1 import artifacts, contract
from yggdrasil_v2.v2_a.closure_c1 import runner
from yggdrasil_v2.v2_a.closure_c1.runner import _invariance_report, _trace_exposure_ledger
from yggdrasil_v2.v2_a.closure_c1.train import ScheduledBatch


def test_contract_freezes_architecture_and_authorization() -> None:
    manifest = contract.c1_contract_manifest()
    assert manifest["model"]["slots"] == 8
    assert manifest["model"]["recurrent_steps"] == 10
    assert manifest["training"]["maximum_updates"] == 6144
    assert manifest["authorization_on_pass"] == "two additional fresh C1 seeds only"
    assert manifest["never_authorizes"] == ["C2", "C3", "V2-B", "V2-C", "V2-A PASS"]
    assert len(contract.TRACE_GRAMMAR_TOKEN_IDS) == 42
    assert len(set(contract.TRACE_GRAMMAR_TOKEN_IDS)) == 42
    assert len(contract.C1_GATE_IDS) == 11


def test_evidence_seal_detects_mutation(tmp_path: Path) -> None:
    artifacts.write_json(tmp_path / "result.json", {"passed": True})
    checksum = artifacts.write_evidence_seal(tmp_path)
    assert len(checksum) == 64
    assert artifacts.audit_evidence_seal(tmp_path)["passed"] is True
    (tmp_path / "result.json").write_text('{"passed":false}\n', encoding="utf-8")
    audit = artifacts.audit_evidence_seal(tmp_path)
    assert audit["passed"] is False
    assert audit["mismatched"] == ["result.json"]


def test_single_use_claim_and_refusal_shape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    lease = tmp_path / "root.lease.jsonl"
    artifacts.claim_single_use(
        identity="TEST", output_root=root, lease_path=lease, formal=False
    )
    assert root.is_dir() and lease.is_file()
    payload = json.loads(lease.read_text(encoding="utf-8"))
    assert payload["identity"] == "TEST"
    refused = artifacts.refusal(
        identity="TEST", output_root=root, lease_path=lease, status="REFUSE"
    )
    assert refused["exit_code"] == 2
    assert refused["authorizes"] == "nothing"


def test_invariance_report_checks_all_raw_logits() -> None:
    baseline = {"x": [1.0] + [0.0] * 8}
    exact = _invariance_report(baseline, baseline, tolerance=0.0)
    assert exact["passed"] is True and exact["max_abs_diff"] == 0.0
    changed = {"x": [1.0, 0.1] + [0.0] * 7}
    assert _invariance_report(baseline, changed, tolerance=0.01)["passed"] is False


def test_sealed_preflight_audit_binds_current_source_identity(tmp_path: Path) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    artifacts.write_json(
        root / "result.json",
        {
            "identity": "P",
            "status": "PASS_P",
            "passed": True,
            "formal": False,
            "source_identity": "SOURCE",
            "source_stable": True,
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
        },
    )
    artifacts.write_evidence_seal(root)
    assert artifacts.audit_preflight_root(
        root, identity="P", status="PASS_P", source_identity_value="SOURCE"
    )["passed"] is True
    assert artifacts.audit_preflight_root(
        root, identity="P", status="PASS_P", source_identity_value="STALE"
    )["passed"] is False


def test_formal_runner_refuses_before_mutation_without_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "_formal_preflight_audit", lambda _root: {"passed": False})
    output = tmp_path / "formal"
    lease = tmp_path / "formal.lease"
    result = runner.run_c1_single_seed(
        tmp_path, output_root=output, lease_path=lease, formal=True, device="cuda"
    )
    assert result["status"] == "REFUSE_V2_A_C1_PREFLIGHT_PREREQUISITE"
    assert result["exit_code"] == 2
    assert not output.exists() and not lease.exists()


def test_formal_runner_refuses_cpu_before_mutation(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    lease = tmp_path / "formal.lease"
    result = runner.run_c1_single_seed(
        tmp_path, output_root=output, lease_path=lease, formal=True, device="cpu"
    )
    assert result["status"] == "REFUSE_V2_A_C1_FORMAL_DEVICE"
    assert not output.exists() and not lease.exists()


def test_formal_runner_refuses_unavailable_cuda_before_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "formal"
    lease = tmp_path / "formal.lease"
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    result = runner.run_c1_single_seed(
        tmp_path, output_root=output, lease_path=lease, formal=True, device="cuda"
    )
    assert result["status"] == "REFUSE_V2_A_C1_FORMAL_DEVICE"
    assert not output.exists() and not lease.exists()


def test_fixed_c1_preflight_refuses_unavailable_cuda_before_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    result = runner.run_c1_preflight(tmp_path)
    output = tmp_path / contract.C1_PREFLIGHT_ROOT
    lease = tmp_path / contract.C1_PREFLIGHT_LEASE
    assert result["status"] == "REFUSE_V2_A_C1_PREFLIGHT_DEVICE"
    assert not output.exists() and not lease.exists()


def test_trace_exposure_ledger_closes_every_epoch_and_token_count() -> None:
    ids = [f"e{index:04d}" for index in range(8192)]
    schedule = [
        ScheduledBatch(update=epoch * 1024 + batch + 1, epoch=epoch, example_ids=tuple(ids[batch * 8 : (batch + 1) * 8]))
        for epoch in range(6)
        for batch in range(1024)
    ]

    class Provider:
        def trace_chunk_metadata(self, example_id, *, epoch, trace_chunk_tokens):
            return {
                "example_id": example_id,
                "epoch": epoch,
                "target_tokens": 80,
                "requested_tokens": trace_chunk_tokens,
                "start": epoch,
                "stop": epoch + trace_chunk_tokens,
                "exposed_tokens": trace_chunk_tokens,
            }

    class Dataset:
        locations = {example_id: ("shard", (1, 2048), 0, 10) for example_id in ids}

    report = _trace_exposure_ledger(schedule, Provider(), Dataset())
    assert report["passed"] is True
    assert report["processed_source_tokens"] == 491_520
    assert report["processed_trace_tokens"] == 3_145_728


def test_finish_rewrites_pass_if_seal_replay_fails_once(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    real_audit = artifacts.audit_evidence_seal
    calls = 0

    def audit(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"passed": False, "reason": "forced"}
        return real_audit(path)

    monkeypatch.setattr(artifacts, "audit_evidence_seal", audit)
    result = runner._finish(
        root,
        result={
            "identity": contract.C1_IDENTITY,
            "formal": True,
            "status": "PASS_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
            "passed": True,
            "gates": {gate: True for gate in contract.C1_GATE_IDS},
            "gate_status": {gate: "passed" for gate in contract.C1_GATE_IDS},
            "training_started": True,
            "optimizer_steps": 1,
            "model_writes": 1,
            "authorizes": "two additional fresh C1 seeds only",
        },
        started_at="test",
        wall_started=time.perf_counter(),
        source_identity="SOURCE",
        source_stable=True,
    )
    sealed = json.loads((root / "result.json").read_text(encoding="utf-8"))
    assert result["passed"] is False and sealed["passed"] is False
    assert sealed["gates"]["G011"] is False
    assert artifacts.audit_evidence_seal(root)["passed"] is True
