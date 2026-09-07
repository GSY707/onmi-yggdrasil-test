from __future__ import annotations

from dataclasses import replace
import os

import pytest
import torch

from yggdrasil_v2.r1_revalidation.h1.train import load_deployment_checkpoint
from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.geometry import (
    record_global_centroid,
    record_route_centroids,
)
from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.io import (
    ReadOnlyInputs,
    load_and_validate_inputs,
)
from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry import screen
from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.capture import (
    audit_full_replay_identity,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("YGGDRASIL_DIRECTION_GEOMETRY_INTEGRATION") != "1",
    reason="explicit CUDA/artifact integration smoke only",
)


def _subset(dataset: dict, indices: torch.Tensor) -> dict:
    result = dict(dataset)
    result["record_ids"] = tuple(dataset["record_ids"][int(index)] for index in indices)
    result["route_schedule"] = dataset["route_schedule"].index_select(0, indices)
    for field in ("common", "projection", "transfer", "vjp"):
        result[field] = tuple(value.index_select(0, indices) for value in dataset[field])
    for field in ("requested_margin", "positive_common_margin_drop"):
        result[field] = dataset[field].index_select(0, indices)
    result["margins"] = {
        key: value.index_select(0, indices) for key, value in dataset["margins"].items()
    }
    return result


def test_internal_r0_r4_pipeline_on_balanced_artifact_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[2]
    inputs = load_and_validate_inputs(repo_root)
    full = inputs.target_payload["datasets"]
    train_indices = torch.tensor(list(range(8)) + list(range(2048, 2056)))
    heldout_indices = torch.tensor(list(range(4)) + list(range(512, 516)))
    datasets = {
        "train": _subset(full["train"], train_indices),
        "heldout": _subset(full["heldout"], heldout_indices),
    }
    subset_inputs = ReadOnlyInputs(
        checkpoint=inputs.checkpoint,
        cache_root=inputs.cache_root,
        target_path=inputs.target_path,
        target_payload={"datasets": datasets},
        cache=inputs.cache,
        audit=inputs.audit,
    )
    tiny = replace(
        screen.SPEC,
        replay_microbatch_size=4,
        analysis_batch_size=4,
        local_full_j_train_records=4,
        local_full_j_heldout_records=4,
        local_full_j_cg_iterations=1,
        local_full_j_fisher_probes=1,
        bootstrap_replicates=16,
        permutation_replicates=16,
        fit_null_replicates=4,
        matched_spectrum_replicates=4,
        alignment_null_replicates=4,
        route_conditional_sign_replicates=8,
        residual_spectrum_rank=4,
    )
    monkeypatch.setattr(screen, "SPEC", tiny)
    model = load_deployment_checkpoint(inputs.checkpoint, device="cuda").eval()
    identity = audit_full_replay_identity(
        model,
        inputs.cache,
        datasets,
        device="cuda",
        batch_size=4,
        tolerance=tiny.maximum_replay_absolute_error,
    )
    assert identity["passed"] is True
    assert identity["maximum_common_absolute_error"] == 0.0
    assert identity["maximum_projection_absolute_error"] == 0.0
    train = datasets["train"]
    centroids = {
        "global": record_global_centroid(train["transfer"]),
        "route": record_route_centroids(
            train["transfer"], train["route_schedule"], site_metadata=train["site_metadata"]
        ),
    }
    progress = lambda _stage, _row: None
    statistics, features, energy, replay = screen._capture_r2(
        model, subset_inputs, centroids, progress
    )
    states = replay.pop("sampled_states")
    fits, r2 = screen._fit_r2(statistics)
    r3_stats, targets = screen._accumulate_r3(
        subset_inputs,
        centroids,
        features,
        fits[tiny.primary_r2_feature],
        replay,
        energy,
        progress,
    )
    r3_fit, r3_exact = screen._fit_r3_exact(r3_stats)
    r3_full = screen._run_sampled_full_j(
        model, subset_inputs, states, targets, replay, progress
    )
    collected = screen._collect_final_geometry(
        subset_inputs,
        centroids,
        features,
        fits[tiny.primary_r2_feature],
        r3_fit,
        energy,
        progress,
    )
    r4 = screen._r4_report_v2(collected, energy, r2, r3_exact, r3_full)
    assert replay["passed"] is True
    assert "residual_classification" in r4
    assert set(r4["signals"]) >= {"r2_replicated", "r3_replicated", "r3_qualified"}
