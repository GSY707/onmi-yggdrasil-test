from __future__ import annotations

from yggdrasil_v2.v2_a.closure_c1s_successor import contract


def test_successor_has_independent_single_use_stage_boundaries() -> None:
    manifest = contract.contract_manifest()
    stages = manifest["stages"]

    assert "srw-successor" in manifest["schema_version"]
    assert "SRW" in manifest["identity"]
    assert manifest["formal"] is False
    assert manifest["qualification"] is True
    assert len({stage["identity"] for stage in stages.values()}) == 3
    assert len({stage["root"] for stage in stages.values()}) == 3
    assert len({stage["lease"] for stage in stages.values()}) == 3
    assert all(stage["endpoint"].startswith("fixed_") for stage in stages.values())
    assert all(stage["checkpoint_selection"] is False for stage in stages.values())
    assert stages["S1"]["formal"] is False
    assert stages["S2"]["formal"] is False
    assert stages["S3"]["formal"] is True
    assert manifest["preflight_accounting"]["s1_disposable_benchmark_optimizer_steps"] == 100
    assert manifest["preflight_accounting"]["formal_stage_training_started"] is False
    assert "c1_cache_full_seal_tree_replay" in manifest["integrity_revalidation"]["all_stage_launches_and_s2_s3_preflight"]


def test_user_authorization_does_not_override_predecessor_gate() -> None:
    auth = contract.PREDECESSOR_AUTHORIZATION

    assert auth["S1"]["required_predecessor"] == contract.S0_IDENTITY
    assert auth["S1"]["required_predecessor_authorizes"] == (
        "one independent C1S S1 Overfit32 implementation and single-use run only"
    )
    assert auth["S1"]["does_not_override_predecessor_gate"] is True
    assert auth["S2"]["required_predecessor"] == contract.S1_IDENTITY
    assert auth["S3"]["required_predecessor"] == contract.S2_IDENTITY
    assert all(item["user_authorization_required"] for item in auth.values())


def test_s1_s2_s3_budget_and_stop_order_are_frozen() -> None:
    assert contract.S1_CONFIG["records_per_family"] == 16
    assert contract.S1_CONFIG["maximum_updates"] == 4_000
    assert contract.S1_CONFIG["preflight_benchmark_model_seed"] == 2026082890
    assert contract.S1_CONFIG["preflight_benchmark_order_seed"] == 2026082891
    assert "overfit_seed" not in contract.S1_CONFIG
    assert contract.S2_CONFIG["train_per_family"] == 3_072
    assert contract.S2_CONFIG["eval_per_family"] == 512
    assert contract.S2_CONFIG["maximum_updates_per_arm"] == 4_608
    assert contract.S2_CONFIG["gpu_hour_limit_per_arm"] == 4.0
    assert contract.S3_CONFIG["full_train_records"] == 8_192
    assert contract.S3_CONFIG["maximum_updates"] == 6_144
    assert contract.S3_CONFIG["gate_order"][:3] == [
        "G004_validation_behavior",
        "G004_OOD_behavior",
        "G004_causal_behavior",
    ]
    assert contract.S3_CONFIG["fail_stop_after_each_gate"] is True


def test_srw_gates_use_semantic_routes_and_measured_single_slot_control() -> None:
    s1 = contract.S1_CONFIG
    s2 = contract.S2_CONFIG
    loss = contract.OPTIMIZATION_CONFIG["loss_weights"]

    assert "query_owner_is_public_and_answer_independent" in s1["hard_gates"]
    assert "operation_active_has_per_family_change_coverage_and_balanced_accuracy" in s1["hard_gates"]
    assert "payload_and_operation_zero_or_shuffle_break_the_target" in s1["hard_gates"]
    assert "s0_single_slot_structural_control_replayed" in s1["hard_gates"]
    assert "matched_k1_is_a_valid_measured_single_state_control" in s2["hard_gates"]
    assert s2["ere_uses_entity_event_routes"] is True
    assert s2["answer_independent_query_owner"] is True
    assert s2["behavior_only_unsupported_ood_is_not_mechanism_evidence"] is True
    assert "payload_binding_margin" in loss
    assert "operation_binding_margin" in loss
    assert "target_shuffle_margin" in loss
    assert "query_swap" not in loss
    assert all("query_swap" not in gate for gate in s1["hard_gates"])
    assert all("delete" not in gate for gate in s1["hard_gates"])
    assert any("mean_replace" in gate for gate in s1["hard_gates"])


def test_public_address_and_selector_descriptions_match_the_implemented_srw_contract() -> None:
    address = contract.TARGET_CONTRACT["address_policy"]

    assert address["CPS"] == "rank_of_semantic_object_in_public_present_LABEL_KEY_rows"
    assert "NFKC_casefold" in address["ERE"]
    assert address["ERE_normalization_collision"] == "fail_closed"
    assert "first_mention" not in address["ERE"]
    assert contract.TARGET_CONTRACT["operation_active_metric"] == (
        "per_family_both_classes_temporal_change_and_balanced_accuracy"
    )
    auxiliary = contract.TARGET_CONTRACT["answer_derived_auxiliary_label_boundary"]
    assert auxiliary["standalone_mechanism_evidence"] is False
    assert auxiliary["visibility"] == "detached_loss_and_evaluator_side_only"
    assert contract.S2_SELECTION == {
        "train": 'sha256("C1S-SRW-S2-DISCOVERY-TRAIN-V1|example_id")',
        "eval": 'sha256("C1S-SRW-S2-DISCOVERY-EVAL-V1|example_id")',
    }


def test_old_c1_thresholds_are_behavior_only_and_cps_boundary_is_explicit() -> None:
    manifest = contract.contract_manifest()

    assert manifest["stages"]["S3"]["behavior_threshold_source"] == "old_C1_thresholds_only"
    assert manifest["stages"]["S2"]["candidate_level_cps_closure_only"] is True
    assert manifest["stages"]["S3"]["does_not_claim_action_level_complete_trajectory"] is True
    assert "answer_only_mechanism_claim" in manifest["formal_prohibitions"]
    assert "V2-A PASS" in manifest["never_authorizes"]


def test_stage_manifest_marks_only_s3_stage_root_formal() -> None:
    assert contract.stage_manifest("S1")["formal"] is False
    assert contract.stage_manifest("S2")["formal"] is False
    s3 = contract.stage_manifest("S3")
    assert s3["formal"] is True
    assert s3["active_stage"] == "S3"
    assert s3["stage_contract"]["formal"] is True
    assert contract.stage_manifest("S3", preflight=True)["formal"] is False
