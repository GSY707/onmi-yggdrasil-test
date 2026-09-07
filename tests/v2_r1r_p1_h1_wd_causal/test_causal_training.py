from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch

from yggdrasil_v2.r1_revalidation.h1.model import H1Config, H1LatentReasoner
from yggdrasil_v2.r1_revalidation.h1.train import PreparedSplit
from yggdrasil_v2.r1_revalidation.h1_wd_causal.contract import DecisionCausalSpec
from yggdrasil_v2.r1_revalidation.h1_wd_causal.core import (
    causal_transfer_targets,
    forward_route_replay,
)
from yggdrasil_v2.r1_revalidation.h1_wd_causal.experiment import (
    build_causal_target_dataset,
    build_unlabeled_schedule,
    paired_bootstrap_comparison,
    train_delete_stage,
    train_joint_pair,
    train_write_stage,
)


class _MemoryCache:
    def __init__(self, count: int, *, tokens: int = 5, width: int = 8) -> None:
        generator = np.random.default_rng(101)
        self.hidden = generator.normal(size=(count, tokens, width)).astype(np.float32)
        self.mask = np.ones((count, tokens), dtype=np.bool_)

    def batch(self, indices, *, device):
        selected = np.asarray(indices, dtype=np.int64)
        return (
            torch.from_numpy(np.array(self.hidden[selected], copy=True)).to(device),
            torch.from_numpy(np.array(self.mask[selected], copy=True)).to(device),
        )


def _config() -> H1Config:
    return H1Config(
        arm="mixed",
        source_width=8,
        latent_width=8,
        K=2,
        T=2,
        recurrent_layers=1,
        num_heads=2,
        active_ffn_multiplier=3,
        shared_ffn_inner_width=12,
        routed_feature_width=12,
    )


def _prepared() -> PreparedSplit:
    return PreparedSplit(
        split="train",
        ids=("a", "b", "c", "d"),
        cache_indices=np.arange(4, dtype=np.int64),
        targets=np.asarray([0, 1, 2, 3], dtype=np.int64),
        traces=np.zeros((4, 2), dtype=np.int64),
        families=("numeric", "relation", "numeric", "relation"),
        family_targets=np.asarray([0, 1, 0, 1], dtype=np.int64),
    )


def _spec() -> DecisionCausalSpec:
    return replace(
        DecisionCausalSpec(),
        batch_size=2,
        target_microbatch_size=2,
        write_updates=1,
        delete_updates=1,
        joint_updates=1,
        evaluation_interval=1,
        write_learning_rate=1.0e-3,
        delete_learning_rate=1.0e-3,
        joint_learning_rate=1.0e-3,
        device="cpu",
    )


def test_causal_target_uses_margin_vjp_and_signed_control() -> None:
    torch.manual_seed(3)
    model = H1LatentReasoner(_config()).eval()
    hidden = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    answers = torch.tensor([0, 1])
    positive = causal_transfer_targets(model, hidden, mask, answers, request_fraction=0.5, site_transfer_cap=0.25)
    assert positive["uses_family_targets"] is False
    assert positive["transfers"]
    assert all(torch.isfinite(value).all() for value in positive["transfers"])
    assert all(
        bool(torch.all(
            torch.linalg.vector_norm(value, dim=(-2, -1))
            <= 0.25 * torch.linalg.vector_norm(site.common, dim=(-2, -1)) + 1e-5
        ))
        for value, site in zip(positive["transfers"], positive["full"]["captures"], strict=True)
    )


def test_route_replay_normal_path_is_equal_and_schedule_is_label_free() -> None:
    torch.manual_seed(5)
    model = H1LatentReasoner(_config()).eval()
    hidden = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    expected = model(hidden, mask, return_trajectory=True)
    actual = forward_route_replay(model, hidden, mask, capture=True)
    assert torch.equal(expected["logits"], actual["logits"])
    assert torch.equal(expected["route_assignments"], actual["route_assignments"])
    schedule, report = build_unlabeled_schedule(4, updates=3, batch_size=2, seed=7)
    assert schedule.shape == (3, 2)
    assert report["uses_family_targets"] is False


def test_short_causal_control_wd_and_joint_do_not_use_teacher_kl() -> None:
    torch.manual_seed(9)
    source = H1LatentReasoner(_config())
    cache = _MemoryCache(4)
    prepared = _prepared()
    spec = _spec()
    schedule = np.asarray([[0, 1]], dtype=np.int64)
    targets = build_causal_target_dataset(source, cache, prepared, spec)
    assert targets.stats["batch_size"] == spec.target_microbatch_size
    assert targets.stats["uses_family_targets"] is False
    assert targets.stats["target_finite"] is True
    assert all(targets.stats["finite_checks"].values())
    assert all(value.device.type == "cpu" for value in targets.transfer)
    causal = deepcopy(source)
    control = deepcopy(source)
    events = []
    write_causal = train_write_stage(causal, cache, prepared, targets, schedule, spec, sign=1.0, progress=lambda stage, row: events.append((stage, row)))
    write_control = train_write_stage(control, cache, prepared, targets, schedule, spec, sign=-1.0, progress=lambda stage, row: events.append((stage, row)))
    assert write_causal["uses_family_targets"] is False
    assert write_control["arm_sign"] == -1.0
    residual_causal, residual_control = deepcopy(causal), deepcopy(control)
    delete_causal = train_delete_stage(residual_causal, causal, cache, prepared, targets, schedule, spec, sign=1.0, progress=lambda stage, row: events.append((stage, row)))
    delete_control = train_delete_stage(residual_control, control, cache, prepared, targets, schedule, spec, sign=-1.0, progress=lambda stage, row: events.append((stage, row)))
    assert delete_causal["uses_family_targets"] is False
    assert delete_control["arm_sign"] == -1.0
    joint = train_joint_pair(residual_causal, residual_control, cache, prepared, targets, schedule, spec, progress=lambda stage, row: events.append((stage, row)))
    assert joint["teacher_kl_used"] is False
    assert joint["teacher_model_used"] is False
    assert joint["route_supervision_used"] is False
    assert joint["answer_route_mode"] == "model_free_route"
    assert joint["allocation_lock_route_mode"] == "frozen_source_route_replay"
    assert tuple(joint["objective_components"]) == (
        "ordinary_answer_cross_entropy",
        "two_path_causal_target_lock",
    )
    assert joint["uses_family_targets"] is False
    assert {stage for stage, _ in events} == {"write", "delete", "joint"}


def test_paired_bootstrap_reports_positive_gain_and_family_values() -> None:
    result = paired_bootstrap_comparison([0, 1, 1, 0], [1, 1, 1, 0], seed=4, bootstrap=200)
    assert result["gain"] == 0.25
    assert result["ci_lower"] <= result["gain"] <= result["ci_upper"]
    with_family = paired_bootstrap_comparison([0, 1, 1, 0], [1, 1, 1, 0], seed=4, bootstrap=200, families=["numeric", "relation", "numeric", "relation"])
    assert set(with_family["family_gain"]) == {"numeric", "relation"}
