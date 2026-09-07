from pathlib import Path

from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis import contract
from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis import runner


def test_contract_is_new_temporal_only_identity() -> None:
    manifest = contract.contract_manifest()
    assert manifest["identity"] == contract.IDENTITY
    assert manifest["scope"] == "D004R_D005R_only"
    assert manifest["diagnostic_only"] is True
    assert manifest["formal"] is False
    assert manifest["authorizes"] == "nothing"
    assert manifest["v2a_passed"] is False
    assert manifest["predecessor"]["rerun_allowed"] is False
    assert manifest["predecessor"]["write_allowed"] is False
    assert contract.OUTPUT_ROOT != contract.PREDECESSOR_DIAGNOSIS_ROOT
    assert contract.PREFLIGHT_ROOT != contract.PREDECESSOR_PREFLIGHT_ROOT
    assert set(manifest["stages"]) == {"D000R", "D004R", "D005R"}


def test_contract_freezes_record_macro_support_policy() -> None:
    policy = contract.contract_manifest()["fold_policy"]
    assert policy["basis"] == "target_only_before_any_latent_or_prediction_is_loaded"
    assert policy["record_macro_is_primary"] is True
    assert policy["observation_micro_is_diagnostic_only"] is True
    assert policy["outer_folds"] == 4
    assert policy["inner_folds"] == 3
    assert policy["minimum_outer_eligible_records"] == 3
    assert policy["minimum_inner_eligible_records"] == 3
    assert "within_record_target_shuffle_fit_null" in policy["same_ledger_for"]


def test_predecessor_partial_pins_are_explicit() -> None:
    assert contract.PREDECESSOR_RESULT_SHA256.startswith("58F45C")
    assert contract.PREDECESSOR_SEAL_SHA256.startswith("AF45A8")
    assert set(contract.PREDECESSOR_PARTIAL_FILES) == {
        "D001-objective-gate-alignment.json",
        "D002-h0-core-paths.json",
        "D003-task-model-arity.json",
        "result.json",
    }


def test_source_closure_contains_every_new_file() -> None:
    root = Path.cwd()
    missing = [path for path in runner.SOURCE_FILES if not (root / path).is_file()]
    assert missing == []
    assert len(runner.SOURCE_FILES) == len(set(runner.SOURCE_FILES))
