from __future__ import annotations

import json
from pathlib import Path

import pytest

from yggdrasil_v2.v2_a.closure_c1r import contract
from yggdrasil_v2.v2_a.closure_c1r import artifacts
from yggdrasil_v2.v2_a.closure_c1r.coverage import (
    build_coverage_plan,
    coverage_report,
    partition_target,
)
from yggdrasil_v2.v2_a.closure_c1r import train


def _targets() -> dict[str, dict[str, object]]:
    return {
        "ere-000": {"family": "ERE", "token_ids": list(range(130))},
        "ere-001": {"family": "ERE", "token_ids": list(range(66))},
        "cps-000": {"family": "CPS", "token_ids": list(range(65))},
        "cps-001": {"family": "CPS", "token_ids": list(range(131))},
    }


def test_contract_has_single_staged_boundary() -> None:
    manifest = contract.c1r_contract_manifest()
    assert manifest["identity"] == "V2-A-CLOSURE-C1R-STAGED-CREDIT-20260827-1"
    assert manifest["stage_a"]["objective"] == "answer_only"
    assert manifest["stage_a"]["maximum_updates"] == 6144
    assert manifest["stage_a"]["bootstrap_seed"] == contract.BOOTSTRAP_SEED
    assert manifest["stage_b"]["deployed_graph"] == "exact_freeze"
    assert manifest["stage_b"]["order_seed"] == contract.ORDER_SEED
    assert manifest["stage_b"]["batch_size"] == 32
    assert manifest["stage_b"]["family_batch_size"] == 16
    assert manifest["stage_b"]["block_tokens"] == 65
    assert manifest["stage_b"]["maximum_updates"] == 1536
    assert manifest["stage_b"]["exposure_per_token"] == 6
    assert manifest["attribution_input"]["result_sha256"].startswith("0726F48F")
    assert manifest["attribution_input"]["seal_sha256"].endswith("2941")
    assert manifest["cache_root"] == "artifacts/v2-a/closure-c1-cache-20260825-1"
    assert manifest["stage_a"]["boundary_lr"] == 1.0e-4
    assert manifest["stage_a"]["weight_decay"] == 0.01
    assert manifest["stage_b"]["trace_lr"] == 3.0e-4
    assert manifest["stage_b"]["gradient_clip"] == 1.0
    assert "C2" in manifest["never_authorizes"]
    assert "V2-A PASS" in manifest["never_authorizes"]


def test_partition_is_width_65_non_overlapping_complete() -> None:
    blocks = partition_target("ere-000", 131)
    assert [(row["start"], row["stop"], row["valid_tokens"]) for row in blocks] == [
        (0, 65, 65),
        (65, 130, 65),
        (130, 131, 1),
    ]
    assert sum(row["valid_tokens"] for row in blocks) == 131
    assert all(left["stop"] == right["start"] for left, right in zip(blocks, blocks[1:]))


def test_complete_target_schedule_has_exact_exposure_and_zero_zero() -> None:
    plan = build_coverage_plan(
        _targets(), epochs=2, batch_size=4, family_batch_size=2, order_seed=17
    )
    report = coverage_report(plan)
    assert plan["updates"] == 2
    assert plan["updates_per_epoch"] == 1
    assert report["blocks"] == 8
    assert report["raw_target_tokens"] == 392
    assert report["exact_exposure_per_token"] == 2
    assert report["min_exposure"] == report["max_exposure"] == 2
    assert report["zero_exposure_tokens"] == 0
    assert report["schedule_checks"] == {
        "schema": True,
        "rows": True,
        "blocks_aggregated_per_record": True,
        "one_record_per_epoch": True,
        "balanced_batch": True,
        "schedule_hash": True,
    }
    assert report["passed"] is True
    assert plan["blocks_aggregated_per_record"] is True


def test_coverage_schedule_matches_train_torch_generator_rows() -> None:
    targets = _targets()
    records = [
        {"example_id": example_id, "family": row["family"]}
        for example_id, row in targets.items()
    ]
    spec = train.StageBSpec(
        epochs=2, maximum_updates=2, batch_size=4, family_batch_size=2
    )
    expected = [
        {"update": row.update, "epoch": row.epoch, "example_ids": list(row.example_ids)}
        for row in train.build_balanced_schedule(records, seed=17, spec=spec)
    ]
    plan = build_coverage_plan(
        targets, epochs=2, batch_size=4, family_batch_size=2, order_seed=17
    )
    assert plan["schedule"] == expected


def test_schedule_rejects_unbalanced_record_counts() -> None:
    with pytest.raises(ValueError, match="equal ERE/CPS halves"):
        build_coverage_plan(
            {"ere-000": {"token_ids": [1]}, "cps-000": {"token_ids": [1]}},
            epochs=1,
            batch_size=2,
            family_batch_size=2,
        )


def test_coverage_report_rejects_overlapping_block_claim() -> None:
    plan = build_coverage_plan(
        _targets(), epochs=1, batch_size=4, family_batch_size=2, order_seed=17
    )
    plan["blocks"]["ere-000"][1]["start"] = 64
    with pytest.raises(ValueError, match="overlapping or incomplete"):
        coverage_report(plan)


def test_c1r_artifacts_have_new_schema_single_use_and_seal(tmp_path: Path) -> None:
    output_root = tmp_path / "formal"
    lease_path = tmp_path / "formal.preflight-lease.jsonl"
    artifacts.claim_single_use(output_root=output_root, lease_path=lease_path, formal=True)
    assert json.loads(lease_path.read_text(encoding="utf-8"))["schema_version"].startswith(
        "yggdrasil.v2-a.closure-c1r."
    )
    artifacts.write_json(output_root / "result.json", {"identity": contract.C1R_IDENTITY})
    seal_sha = artifacts.write_evidence_seal(output_root)
    replay = artifacts.audit_evidence_seal(output_root)
    assert replay["passed"] is True
    assert replay["seal_sha256"] == seal_sha
    with pytest.raises(FileExistsError):
        artifacts.claim_single_use(output_root=output_root, lease_path=lease_path, formal=True)


def test_pinned_root_audit_accepts_explicit_foreign_namespace(tmp_path: Path) -> None:
    root = tmp_path / "foreign"
    root.mkdir()
    artifacts.write_json(root / "result.json", {"identity": "OLD-NAMESPACE"})
    artifacts.write_json(
        root / "evidence-seal.json",
        {
            "schema_version": "old.namespace.evidence-seal.v1",
            "identity": "OLD-NAMESPACE",
            "files": artifacts.tree_hashes(root, exclude=("evidence-seal.json",)),
        },
    )
    audit = artifacts.audit_pinned_root(
        root,
        result_sha256=artifacts.sha256_file(root / "result.json"),
        seal_sha256=artifacts.sha256_file(root / "evidence-seal.json"),
        expected_schema_version="old.namespace.evidence-seal.v1",
        expected_identity="OLD-NAMESPACE",
    )
    assert audit["passed"] is True


def test_preflight_audit_requires_zero_writes_and_matching_source_identity(tmp_path: Path) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    repo_root = Path(__file__).resolve().parents[2]
    source_id = artifacts.source_identity(artifacts.source_hashes(repo_root))
    source_paths = {path.as_posix() for path in artifacts.SOURCE_FILES}
    assert "experiments/v2_a_closure_c1r.py" in source_paths
    assert "src/yggdrasil_v2/v2_a/closure_c1r/runner.py" in source_paths
    assert "src/yggdrasil_v2/v2_a/closure_c1r/train.py" in source_paths
    assert {path for path in source_paths if path.startswith("tests/v2_a_closure_c1r/")} == {
        "tests/v2_a_closure_c1r/test_contract_coverage.py",
        "tests/v2_a_closure_c1r/test_runner.py",
        "tests/v2_a_closure_c1r/test_train.py",
    }
    artifacts.write_json(
        root / "result.json",
        {
            "schema_version": "yggdrasil.v2-a.closure-c1r.preflight.v1",
            "identity": contract.C1R_IDENTITY,
            "status": "PASS_V2_A_C1R_PREFLIGHT",
            "passed": True,
            "formal": False,
            "source_identity": source_id,
            "source_stable": True,
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "old_root_writes": 0,
            "old_roots_stable": True,
            "device": {
                "passed": True,
                "selected_device_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
            },
        },
    )
    artifacts.write_evidence_seal(root)
    audit = artifacts.audit_preflight_root(root, source_identity_value=source_id)
    assert audit["passed"] is True
    assert all(audit["checks"].values())
