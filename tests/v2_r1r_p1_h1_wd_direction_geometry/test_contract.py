from __future__ import annotations

from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry import screen
from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.contract import (
    AUTHORIZES,
    CAUSAL_TARGET_BANK_SHA256,
    CONTRACT_VERSION,
    PREFLIGHT_LEASE,
    SCOPE,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_TOKEN_CACHE_SHA256,
    contract_manifest,
)


def test_contract_identity_and_evidence_hashes() -> None:
    assert CONTRACT_VERSION == "h1-wd-direction-geometry-v2-full-replay"
    assert SCOPE == "NONFORMAL_H1_WD_DIRECTION_GEOMETRY_V2_ONLY"
    assert AUTHORIZES == "nothing"
    assert len(SOURCE_CHECKPOINT_SHA256) == 64
    assert CAUSAL_TARGET_BANK_SHA256 == (
        "DFC8F08776CE56EFB8022BF35B7C3DD72F8F1C53C95DBF0DD28F593DC2BFD503"
    )
    assert set(SOURCE_TOKEN_CACHE_SHA256) == {
        "source-tokens-hidden.npy",
        "source-tokens-index.json",
        "source-tokens-mask.npy",
    }


def test_manifest_seals_non_training_scope() -> None:
    manifest = contract_manifest()
    assert manifest["scope"] == SCOPE
    assert manifest["authorizes"] == "nothing"
    assert manifest["spec"]["optimizer_steps"] == 0
    assert manifest["spec"]["model_writes"] is False
    assert manifest["spec"]["old_root_mutation"] is False
    assert manifest["method"]["heldout_is_iid"] is False
    assert manifest["spec"]["replay_microbatch_size"] == 4
    assert manifest["spec"]["analysis_batch_size"] == 128
    assert manifest["spec"]["residual_null_family_tests"] == 12
    assert manifest["fail_stop"]["scientific_null_or_no_signal_is_a_valid_completed_result"] is True
    assert manifest["fail_stop"][
        "attempt_is_consumed_by_exclusive_preflight_lease_before_hard_gates"
    ] is True
    assert manifest["source"]["preflight_lease"] == PREFLIGHT_LEASE.as_posix()
    assert manifest["source"]["causal_target_bank_sha256"] == CAUSAL_TARGET_BANK_SHA256


def test_screen_surface_is_the_only_runner_surface() -> None:
    assert callable(screen.run_direction_geometry_screen)
    assert screen.run_direction_geometry_screen.__module__.endswith(".screen")
