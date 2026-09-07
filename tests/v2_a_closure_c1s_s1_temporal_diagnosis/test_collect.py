from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis.collect import (
    collect_temporal,
)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(slots=2)

    def boundary(self, *, source_hidden: torch.Tensor, source_mask: torch.Tensor):
        assert source_mask.dtype == torch.bool
        return SimpleNamespace(payloads=source_hidden[:, :2, :3] * self.scale)

    def forward_from_boundary(self, boundary, **kwargs):
        trajectory = torch.stack((boundary.payloads + 1, boundary.payloads + 2), dim=1)
        return {
            "trajectory": trajectory,
            "auxiliary": {"state_features": trajectory},
        }


class FakeHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_decoder = nn.Linear(3, 2)


class FakeRuntime:
    def get_batch(self, ids, *, slots: int, pin_memory: bool):
        assert slots == 2
        assert pin_memory is False
        batch = len(ids)
        return {
            "source_hidden": torch.arange(
                batch * 2 * 2048, dtype=torch.float32
            ).reshape(batch, 2, 2048),
            "source_mask": torch.ones(batch, 2, dtype=torch.bool),
            "answers": torch.zeros(batch, dtype=torch.long),
            "state_values": torch.zeros(batch, 2, 2, 2, dtype=torch.bool),
            "state_feature_mask": torch.ones(batch, 2, 2, 2, dtype=torch.bool),
            "state_step_mask": torch.ones(batch, 2, dtype=torch.bool),
            "operation_active": torch.ones(batch, 2, 2, dtype=torch.bool),
            "families": ["CPS"] * batch,
            "state_feature_names": [["processed", "running_best"] for _ in ids],
        }


def test_collect_temporal_uses_only_natural_source_boundary() -> None:
    model = FakeModel()
    heads = FakeHeads()
    result = collect_temporal(
        model,
        heads,
        FakeRuntime(),
        ["a", "b"],
        device="cpu",
        batch_size=2,
    )
    assert result["example_ids"] == ["a", "b"]
    assert result["trajectory"].shape == (2, 2, 2, 3)
    assert result["state_features"].shape == (2, 2, 2, 3)
    assert result["state_logits"].shape == (2, 2, 2, 2)
    assert result["mutation_audit"]["state_unchanged"] is True
    assert result["mutation_audit"]["parameter_grad_fields_empty"] is True
    assert all(parameter.grad is None for parameter in model.parameters())
