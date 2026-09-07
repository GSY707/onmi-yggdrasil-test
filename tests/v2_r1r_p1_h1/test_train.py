from pathlib import Path

import numpy as np
import torch

from yggdrasil_v2.r1_revalidation.h1.model import H1Config, H1LatentReasoner
from yggdrasil_v2.r1_revalidation.h1.train import (
    PreparedSplit,
    build_balanced_schedule,
    build_pair,
    evaluate_model,
    schedule_audit,
    strip_and_save,
    train_fixed_budget,
    load_deployment_checkpoint,
    metamorphic_consistency,
    metamorphic_route_consistency,
)
from yggdrasil_v2.r1_revalidation.h1.contract import TrainingSpec


class _MemoryCache:
    """Minimal TokenCache-compatible fixture using the production batch contract."""

    def __init__(self, ids, *, tokens=5, width=2048):
        self.ids = tuple(ids)
        self._indices = {value: index for index, value in enumerate(self.ids)}
        self.hidden = np.zeros((len(ids), tokens, width), dtype=np.float16)
        for index in range(len(ids)):
            self.hidden[index, :, index] = 1.0
        self.mask = np.ones((len(ids), tokens), dtype=np.bool_)

    def indices(self, record_ids):
        return np.asarray([self._indices[value] for value in record_ids], dtype=np.int64)

    def batch(self, indices, *, device):
        selected = np.asarray(indices, dtype=np.int64)
        return (
            torch.from_numpy(np.array(self.hidden[selected], copy=True)).to(device),
            torch.from_numpy(np.array(self.mask[selected], copy=True)).to(device),
        )


def _prepared(steps=8):
    ids = ("n0", "n1", "r0", "r1")
    return PreparedSplit(
        split="train",
        ids=ids,
        cache_indices=np.arange(4, dtype=np.int64),
        targets=np.asarray([0, 1, 2, 3], dtype=np.int64),
        traces=np.asarray([[0, 1, 2, 3, 3, 3, 3, 3]] * 4, dtype=np.int64)[:, :steps],
        families=("numeric", "numeric", "relation", "relation"),
        family_targets=np.asarray([0, 0, 1, 1], dtype=np.int64),
    )


def _config(arm="shared"):
    return H1Config(
        arm=arm,
        source_width=2048,
        latent_width=16,
        K=8,
        T=8,
        recurrent_layers=2,
        num_heads=4,
        active_ffn_multiplier=3,
        shared_ffn_inner_width=24,
        routed_feature_width=24,
    )


def test_balanced_schedule_and_audit():
    prepared = _prepared()
    schedule = build_balanced_schedule(prepared, updates=2, batch_size=2, seed=19)
    assert schedule.shape == (2, 2)
    audit = schedule_audit(prepared, schedule, seed=19)
    assert audit["passed"]
    assert audit["family_counts"] == {"numeric": [1], "relation": [1]}
    assert audit["all_records_exposed"]


def test_short_cpu_train_evaluate_interventions_and_strip_reload(tmp_path: Path):
    prepared = _prepared()
    cache = _MemoryCache(prepared.ids)
    schedule = build_balanced_schedule(prepared, updates=2, batch_size=2, seed=7)
    spec = TrainingSpec(batch_size=2, updates=2, evaluation_interval=1)
    model = H1LatentReasoner(_config())
    result = train_fixed_budget(model, cache, prepared, schedule, spec, device="cpu")
    assert result["updates"] == 2
    assert result["examples_seen"] == 4
    assert len(result["history"]) == 2

    normal = evaluate_model(
        model,
        cache,
        prepared,
        device="cpu",
        batch_size=2,
        include_state_hash=True,
    )
    assert len(normal["predictions"]) == 4
    assert len(normal["trace_predictions"]) == 4
    assert len(normal["route_predictions"]) == 4
    assert normal["route"]["diagnostics"]["shape"] == [4, 2, 8]
    assert len(normal["route"]["diagnostics"]["cells"]) == 16
    assert set(normal["state_hashes"]) == {"logits", "final_state", "trajectory"}
    for intervention in (
        "flip_route",
        "force_route0",
        "force_route1",
        "swap_experts",
        "disable_routed_projection",
        "disable_recurrence",
        "zero_source",
        "shuffle_source",
    ):
        trial = evaluate_model(
            model, cache, prepared, device="cpu", batch_size=2, intervention=intervention
        )
        assert len(trial["predictions"]) == 4

    checkpoint_path = tmp_path / "deployment.pt"
    stripped = strip_and_save(model, checkpoint_path, cache, prepared, device="cpu")
    assert stripped["passed"]
    assert stripped["reload_predictions_equal"]
    assert model.trace_head is None
    loaded = load_deployment_checkpoint(checkpoint_path, device="cpu")
    assert loaded.trace_head is None
    reloaded = evaluate_model(
        loaded, cache, prepared, device="cpu", batch_size=2, include_trace=False
    )
    assert reloaded["predictions"] == normal["predictions"]
    assert reloaded["answer"]["prediction_sha256"] == normal["answer"]["prediction_sha256"]


def test_pair_level_metamorphic_consistency_requires_registered_relation():
    base = {
        "ids": ["b0", "b1"],
        "predictions": [0, 1],
        "targets": [0, 1],
    }
    transformed = {
        "ids": ["t0", "t1"],
        "predictions": [0, 2],
    }
    ledger = {
        "t0": {
            "base_id": "b0",
            "transform": "handle_rename",
            "family": "numeric",
            "base_target": 0,
            "target": 0,
            "expected_same_answer": True,
        },
        "t1": {
            "base_id": "b1",
            "transform": "choice_permutation",
            "family": "relation",
            "base_target": 1,
            "target": 2,
            "expected_same_answer": False,
        },
    }
    report = metamorphic_consistency(base, transformed, ledger)
    assert report["aggregate"]["count"] == 2
    assert report["aggregate"]["relation_consistency"] == 1.0
    assert report["aggregate"]["transformed_exact"] == 1.0
    assert set(report["by_transform"]) == {"choice_permutation", "handle_rename"}


def test_pair_level_metamorphic_route_consistency_uses_family_targets():
    base = {
        "ids": ["b0", "b1"],
        "route_predictions": [0, 1],
    }
    transformed = {
        "ids": ["t0", "t1"],
        "route_predictions": [0, 1],
    }
    ledger = {
        "t0": {
            "base_id": "b0",
            "transform": "handle_rename",
            "family": "numeric",
        },
        "t1": {
            "base_id": "b1",
            "transform": "surface_paraphrase",
            "family": "relation",
        },
    }
    report = metamorphic_route_consistency(base, transformed, ledger)
    assert report["aggregate"]["consistency"] == 1.0
    assert report["aggregate"]["transformed_exact"] == 1.0
    assert set(report["by_family"]) == {"numeric", "relation"}


def test_matched_pair_initialization_report():
    shared, mixed, report = build_pair(_config(), seed=31)
    assert report["passed"]
    assert report["common_mismatches"] == []
    assert len(report["shared_ffn_copies"]) == 2
    assert len(report["routed_feature_trunk_copies"]) == 2
    assert len(report["routed_projection_copies"]) == 4
    assert all(row["equal"] for row in report["shared_ffn_copies"])
    assert all(row["equal"] for row in report["routed_feature_trunk_copies"])
    assert all(row["equal"] for row in report["routed_projection_copies"])
    assert shared.arm == "shared"
    assert mixed.arm == "mixed"
