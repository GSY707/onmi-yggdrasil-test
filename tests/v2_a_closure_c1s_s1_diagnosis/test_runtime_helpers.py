from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis.runtime import (
    load_gzip_json,
    s1_ids,
)
from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis import artifacts, runner


def test_gzip_loader_and_s1_order(tmp_path) -> None:
    path = tmp_path / "bank.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"records": {}}, handle)
    assert load_gzip_json(path) == {"records": {}}
    ids = [f"id-{index:02d}" for index in range(32)]
    assert s1_ids({"s1": {"ids": ids}}) == ids


def test_s1_ids_reject_duplicate() -> None:
    with pytest.raises(ValueError):
        s1_ids({"s1": {"ids": ["same"] * 32}})


def _sealed_family_report(family: str) -> dict:
    return {
        "interventions": {"no_core": {"answer": {"overall": {"point": 0.75}}}},
        "functional_k": {
            "causal_contributors": {
                "at_least_two": {"point": 0.25},
                "mean_effective_slots": 1.5,
            }
        },
        "ownership": {
            "state_masked": {
                "dynamic_coverage_pass": True,
                "registered_dynamic_features": {family: ["feature"]},
                "families": {
                    family: {
                        "feature": {
                            "dynamic_coverage": True,
                            "balanced_accuracy": 0.90,
                        }
                    }
                },
            }
        },
    }


def test_gate_rows_are_recomputed_from_sealed_family_evidence() -> None:
    rows = runner._gate_rows(
        {
            "family_reports": {
                "CPS": _sealed_family_report("CPS"),
                "ERE": _sealed_family_report("ERE"),
            }
        }
    )
    assert len(rows) == 6
    assert {row["family"] for row in rows} == {"CPS", "ERE"}
    assert all(row["passed"] is False for row in rows)


def test_temporal_source_replay_compares_rebuilt_targets(monkeypatch) -> None:
    target = {
        "example_id": "x",
        "family": "CPS",
        "state_values": [[[1]]],
        "state_feature_mask": [[[1]]],
        "state_step_mask": [1],
        "state_feature_names": ["processed"],
        "operation_active": [1],
        "source_owner": [0],
        "target_owner": [0],
        "mechanism_supported": True,
    }
    monkeypatch.setattr(runner, "materialize_target", lambda source, tokenizer, strict: dict(target))
    report = runner._audit_temporal_source_replay([target], {"x": {"example_id": "x"}})
    assert report["passed"] is True
    assert report["records"][0]["mismatched_fields"] == []


def test_post_seal_exception_never_rewrites_result(tmp_path) -> None:
    root = tmp_path / "sealed"
    root.mkdir()
    original = {"identity": "i", "status": "COMPLETE", "authorizes": "nothing"}
    artifacts.write_json(root / "result.json", original)
    runner._seal(root, identity="i", stage="D")
    before = (root / "result.json").read_bytes()

    terminal = runner._sealed_terminal_after_exception(
        root,
        identity="i",
        stage="D",
        error=RuntimeError("after seal"),
    )

    assert terminal is not None
    assert terminal["status"] == "COMPLETE"
    assert terminal["seal_replay"]["passed"] is True
    assert (root / "result.json").read_bytes() == before


def test_own_preflight_rejects_source_identity_drift(monkeypatch, tmp_path) -> None:
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    lease = tmp_path / "preflight.lease.jsonl"
    artifacts.write_json(
        preflight / "result.json",
        {
            "identity": runner.PREFLIGHT_IDENTITY,
            "stage": runner.PREFLIGHT_STAGE,
            "status": "PASS_C1S_S1_FAILURE_ATTRIBUTION_PREFLIGHT",
            "passed": True,
            "authorizes": "THIS_DIAGNOSIS_SINGLE_USE_ONLY",
            "source_identity": "STALE",
        },
    )
    lease.write_text(
        json.dumps(
            {
                "identity": runner.PREFLIGHT_IDENTITY,
                "stage": runner.PREFLIGHT_STAGE,
                "formal": False,
                "diagnostic_only": True,
                "output_root": str(preflight.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner.contract, "PREFLIGHT_ROOT", Path("preflight"))
    monkeypatch.setattr(runner.contract, "PREFLIGHT_LEASE", Path("preflight.lease.jsonl"))
    monkeypatch.setattr(runner.artifacts, "audit_evidence_seal", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(runner, "_source_hashes", lambda root: {"x.py": "CURRENT"})

    report = runner._audit_own_preflight(tmp_path)
    assert report["checks"]["source_identity_frozen"] is False
    assert report["passed"] is False


def test_diagnosis_rechecks_source_identity_immediately_before_lease(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner.contract, "OUTPUT_ROOT", Path("diagnosis"))
    monkeypatch.setattr(runner.contract, "LEASE_PATH", Path("diagnosis.lease.jsonl"))
    monkeypatch.setattr(
        runner,
        "_audit_own_preflight",
        lambda root: {"passed": True, "sealed_source_identity": "SEALED"},
    )
    monkeypatch.setattr(runner, "_audit_inputs", lambda root, full_bank_audit: {"passed": True})
    monkeypatch.setattr(runner.artifacts, "capture_old_root_snapshot", lambda root: {})
    monkeypatch.setattr(runner, "_source_hashes", lambda root: {"changed.py": "DRIFTED"})

    def forbidden_claim(**kwargs):
        raise AssertionError("lease must not be claimed after source drift")

    monkeypatch.setattr(runner.artifacts, "claim_single_use", forbidden_claim)
    result = runner.run_diagnosis(tmp_path, device="cpu")
    assert result["status"] == "REFUSED"
    assert not (tmp_path / "diagnosis").exists()
    assert not (tmp_path / "diagnosis.lease.jsonl").exists()


def test_diagnostic_completion_requires_registered_controls_and_cells() -> None:
    def classification(label: str, counts: dict[str, int]) -> dict:
        return {
            "family_complete": True,
            "missing_families": [],
            "families": {
                family: {"classification": label, "n": counts[family]}
                for family in ("CPS", "ERE")
            },
        }
    d001 = {
        "classification": classification("OBJECTIVE_ALIGNED", {"CPS": 3, "ERE": 3}),
        "sealed_gate_rows": [
            {"family": family, "gate": gate}
            for family in ("CPS", "ERE")
            for gate in (
                "no_core_near_chance",
                "functional_multi_address",
                "dynamic_state_each_feature",
            )
        ],
        "mutation_audit": {
            "state_unchanged": True,
            "parameter_grad_fields_empty": True,
            "optimizer_steps": 0,
            "model_writes": 0,
        },
    }
    paths = {name: {} for name in runner.PATH_NAMES}
    d002 = {
        "classification": classification("MIXED", {"CPS": 16, "ERE": 16}),
        "families": {
            family: {
                "paths": paths,
                "answer_permutation_metric_null": {"control_valid": True},
            }
            for family in ("CPS", "ERE")
        },
        "path_definitions": {
            name: "registered"
            for name in (
                "final_delta",
                "h0_relevant_replace",
                "global_mean",
                "copy_all_query_owner",
                "answer_permutation_metric_null",
                "initial_slot_replace_logits",
            )
        },
    }
    d003 = {
        "classification": classification("INCONCLUSIVE", {"CPS": 16, "ERE": 16}),
        "records": [
            {
                "example_id": f"{family.lower()}-{index}",
                "family": family,
                "task": {
                    "valid": True,
                    "registered_execution": {"status": "ASSESSED"},
                    "certificate": {"status": "ASSESSED"},
                    "answer_support": {"status": "NOT_ASSESSABLE"},
                    "counterfactual_influence": {"status": "NOT_ASSESSABLE"},
                },
            }
            for family in ("CPS", "ERE")
            for index in range(16)
        ],
    }
    d004_families = {}
    for family, names in runner.HARD_FEATURES.items():
        d004_families[family] = {
            "features": {
                name: {
                    "cross_fitted_ridge_record_macro_valid_records": 1,
                    "frozen_decoder_record_macro_valid_records": 1,
                    "nulls": {
                        null: {"record_macro_valid_records": 1}
                        for null in ("time_shuffle", "temporal_mean", "target_shuffle")
                    },
                }
                for name in names
            }
        }
    d004 = {
        "classification": classification("INCONCLUSIVE", {"CPS": 2, "ERE": 4}),
        "source_visible_replay": {"passed": True},
        "families": d004_families,
    }

    report = runner._diagnostic_evidence_audit(d001, d002, d003, d004)
    assert report["passed"] is True
    d002["families"]["CPS"]["answer_permutation_metric_null"]["control_valid"] = False
    assert runner._diagnostic_evidence_audit(d001, d002, d003, d004)["passed"] is False

    d002["families"]["CPS"]["answer_permutation_metric_null"]["control_valid"] = True
    d001["sealed_gate_rows"][1]["gate"] = "no_core_near_chance"
    duplicate = runner._diagnostic_evidence_audit(d001, d002, d003, d004)
    assert duplicate["checks"]["d001_sealed_gate_cells_complete"] is False


def test_diagnostic_completion_rejects_empty_or_illegal_family_cells() -> None:
    report = {
        "family_complete": True,
        "missing_families": [],
        "families": {"CPS": {}, "ERE": {}},
    }
    # Exercise through the public evidence helper with deliberately incomplete
    # siblings: the first classification check must not accept empty cells.
    d001 = {
        "classification": report,
        "sealed_gate_rows": [],
        "mutation_audit": {},
    }
    evidence = runner._diagnostic_evidence_audit(d001, {}, {}, {})
    assert evidence["checks"]["d001_family_classification_complete"] is False
