from copy import deepcopy
from dataclasses import replace

import numpy as np
import torch

from yggdrasil_v2.r1_revalidation.h1.model import H1Config, H1LatentReasoner
from yggdrasil_v2.r1_revalidation.h1.train import PreparedSplit
from yggdrasil_v2.r1_revalidation.h1_wd.contract import WriteDeleteSpec
from yggdrasil_v2.r1_revalidation.h1_wd.core import (
    forward_write_delete,
    freeze_for_role,
    overlap_write_target,
)
from yggdrasil_v2.r1_revalidation.h1_wd.experiment import (
    build_unlabeled_schedule,
    train_delete_stage,
    train_write_stage,
)


class _MemoryCache:
    def __init__(self, count: int, *, tokens: int = 5, width: int = 8) -> None:
        generator = np.random.default_rng(41)
        self.hidden = generator.normal(size=(count, tokens, width)).astype(np.float16)
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


def _align_projection_with_common(model: H1LatentReasoner) -> None:
    for layer in model.layers:
        layer.routed_feature_trunk.norm.load_state_dict(
            layer.shared_ffn.ffn_norm.state_dict()
        )
        layer.routed_feature_trunk.gate.load_state_dict(
            layer.shared_ffn.ffn.gate.state_dict()
        )
        layer.routed_feature_trunk.up.load_state_dict(
            layer.shared_ffn.ffn.up.state_dict()
        )
        for projection in layer.routed_projections:
            projection.load_state_dict(layer.shared_ffn.ffn.down.state_dict())


def test_wd_forward_matches_original_normal_path() -> None:
    torch.manual_seed(7)
    model = H1LatentReasoner(_config()).eval()
    hidden = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5, dtype=torch.bool)
    with torch.inference_mode():
        expected = model(hidden, mask, return_trajectory=True)
        actual = forward_write_delete(model, hidden, mask)
    assert torch.equal(actual["logits"], expected["logits"])
    assert torch.equal(actual["final_state"], expected["final_state"])
    assert torch.equal(actual["trajectory"], expected["trajectory"])
    assert torch.equal(actual["route_assignments"], expected["route_assignments"])


def test_overlap_target_preserves_sum_after_residualization() -> None:
    projection = torch.ones(2, 2, 3)
    common = projection.clone()
    written, transfer, coefficient = overlap_write_target(
        common, projection, maximum_coefficient=0.75
    )
    residual = common - transfer
    assert torch.allclose(coefficient, torch.full((2,), 0.75))
    assert torch.allclose(residual + written, common + projection)
    assert residual.square().sum() < common.square().sum()


def test_unlabeled_schedule_does_not_require_family_targets() -> None:
    schedule, report = build_unlabeled_schedule(
        4, updates=4, batch_size=2, seed=19
    )
    assert schedule.shape == (4, 2)
    assert report["all_records_exposed"]
    assert report["uses_family_targets"] is False


def test_role_freezing_keeps_one_shared_model() -> None:
    model = H1LatentReasoner(_config())
    selected = freeze_for_role(model, "projection")
    assert selected
    assert all(parameter.requires_grad for parameter in selected)
    assert not any(
        parameter.requires_grad
        for layer in model.layers
        for parameter in layer.shared_ffn.parameters()
    )
    assert model.answer_head is not None
    assert model.layers[0].shared_ffn is not None


def test_short_write_then_delete_updates_only_registered_branch() -> None:
    torch.manual_seed(11)
    teacher = H1LatentReasoner(_config())
    teacher.strip_trace_head()
    _align_projection_with_common(teacher)
    cache = _MemoryCache(4)
    prepared = _prepared()
    schedule = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    spec = replace(
        WriteDeleteSpec(),
        batch_size=2,
        write_updates=2,
        delete_updates=2,
        joint_updates=1,
        evaluation_interval=1,
        write_learning_rate=1.0e-3,
        delete_learning_rate=1.0e-3,
        device="cpu",
    )
    events = []
    written = deepcopy(teacher)
    common_before = {
        name: value.detach().clone()
        for name, value in written.named_parameters()
        if ".shared_ffn." in name
    }
    report_w = train_write_stage(
        written,
        teacher,
        cache,
        prepared,
        schedule,
        spec,
        progress=lambda stage, row: events.append((stage, row)),
    )
    assert report_w["uses_family_targets"] is False
    assert all(
        torch.equal(common_before[name], value)
        for name, value in written.named_parameters()
        if name in common_before
    )

    residual = deepcopy(written)
    projection_before = {
        name: value.detach().clone()
        for name, value in residual.named_parameters()
        if ".routed_" in name
    }
    report_d = train_delete_stage(
        residual,
        written,
        teacher,
        cache,
        prepared,
        schedule,
        spec,
        progress=lambda stage, row: events.append((stage, row)),
    )
    assert report_d["uses_family_targets"] is False
    assert all(
        torch.equal(projection_before[name], value)
        for name, value in residual.named_parameters()
        if name in projection_before
    )
    assert {stage for stage, _ in events} == {"write", "delete"}
