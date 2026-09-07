from __future__ import annotations

import json
from pathlib import Path

import pytest

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis import artifacts


def test_seal_replay_and_snapshot_detect_mutations(tmp_path: Path) -> None:
    root = tmp_path / "diagnosis"
    root.mkdir()
    artifacts.write_json(root / "result.json", {"status": "INCONCLUSIVE", "passed": False})
    artifacts.write_json(root / "metrics.json", {"n": 2})
    snapshot = artifacts.snapshot_tree(root)
    seal_sha = artifacts.write_evidence_seal(root, identity="DIAG-I", stage="D000")

    replay = artifacts.audit_evidence_seal(
        root,
        expected_identity="DIAG-I",
        expected_stage="D000",
        expected_schema_prefix=artifacts.SCHEMA_PREFIX,
    )
    assert replay["passed"] is True
    assert replay["seal_sha256"] == seal_sha
    assert artifacts.audit_tree_snapshot(root, snapshot)["passed"] is False
    # The snapshot predates the seal, so compare an explicit post-seal snapshot
    # before testing content mutation.
    sealed_snapshot = artifacts.snapshot_tree(root)
    (root / "result.json").write_text('{"passed": true}\n', encoding="utf-8")
    assert artifacts.audit_tree_snapshot(root, sealed_snapshot)["passed"] is False
    assert artifacts.audit_evidence_seal(root)["passed"] is False


def test_seal_replay_rejects_unexpected_file_and_wrong_identity(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifacts.write_json(root / "result.json", {"ok": True})
    artifacts.write_evidence_seal(root, identity="I", stage="D000")
    artifacts.write_json(root / "unexpected.json", {"x": 1})
    report = artifacts.audit_evidence_seal(root, expected_identity="WRONG", expected_stage="D000")
    assert report["passed"] is False
    assert report["identity"] is False
    assert report["unexpected"] == ["unexpected.json"]


def test_claim_single_use_is_exclusive_and_records_diagnostic_scope(tmp_path: Path) -> None:
    root = tmp_path / "root"
    lease = tmp_path / "root.lease.jsonl"
    payload = artifacts.claim_single_use(
        identity="DIAG-I", stage="D000", output_root=root, lease_path=lease
    )
    assert payload["diagnostic_only"] is True
    assert json.loads(lease.read_text(encoding="utf-8"))["identity"] == "DIAG-I"
    with pytest.raises(FileExistsError):
        artifacts.claim_single_use(
            identity="DIAG-I", stage="D000", output_root=root, lease_path=lease
        )


def test_claim_rolls_back_empty_root_when_lease_creation_fails(tmp_path: Path) -> None:
    root = tmp_path / "root"
    # A directory at the exact lease path makes the exclusive lease open fail.
    lease = tmp_path / "root.lease"
    lease.mkdir()
    with pytest.raises(FileExistsError):
        artifacts.claim_single_use(
            identity="I", stage="D000", output_root=root, lease_path=lease
        )
    assert not root.exists()


def test_sealed_file_pin_checks_explicit_digest_and_tree(tmp_path: Path) -> None:
    root = tmp_path / "producer"
    root.mkdir()
    bank = root / "bank.json"
    bank.write_bytes(b"immutable")
    expected = {bank.name: artifacts.sha256_file(bank)}
    artifacts.write_evidence_seal(root, identity="I", stage="D000")
    assert artifacts.audit_sealed_file_pins(
        root,
        expected_identity="I",
        expected_stage="D000",
        expected_files=expected,
    )["passed"] is True
    bank.write_bytes(b"tampered")
    report = artifacts.audit_sealed_file_pins(
        root,
        expected_identity="I",
        expected_stage="D000",
        expected_files=expected,
    )
    assert report["passed"] is False
    assert report["files"][bank.name]["checks"]["actual_matches_expected"] is False


def test_external_pin_requires_result_seal_and_replay(tmp_path: Path) -> None:
    root = tmp_path / "old"
    root.mkdir()
    artifacts.write_json(root / "result.json", {"status": "FAIL"})
    artifacts.write_evidence_seal(root, identity="OLD", stage="S1")
    pin = {
        "root": "old",
        "result_sha256": artifacts.sha256_file(root / "result.json"),
        "seal_sha256": artifacts.sha256_file(root / "evidence-seal.json"),
    }
    assert artifacts.audit_external_pin(tmp_path, pin)["passed"] is True
    (root / "result.json").write_text('{"status":"MUTATED"}\n', encoding="utf-8")
    assert artifacts.audit_external_pin(tmp_path, pin)["passed"] is False


def test_predecessor_audit_requires_exact_structured_authorization(tmp_path: Path) -> None:
    root = tmp_path / "predecessor"
    root.mkdir()
    artifacts.write_json(
        root / "result.json",
        {
            "identity": "S1",
            "stage": "S1",
            "status": "PASS",
            "passed": True,
            "authorizes": None,
        },
    )
    artifacts.write_evidence_seal(root, identity="S1", stage="S1")
    report = artifacts.audit_stage_predecessor(
        root,
        expected_identity="S1",
        expected_stage="S1",
        expected_status="PASS",
        expected_authorizes=None,
    )
    assert report["passed"] is True
    refused = artifacts.audit_stage_predecessor(
        root,
        expected_identity="S1",
        expected_stage="S1",
        expected_status="PASS",
        expected_authorizes="S2",
    )
    assert refused["passed"] is False
