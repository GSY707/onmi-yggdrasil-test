from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from yggdrasil_v2.v2_a.closure_c1 import train
from yggdrasil_v2.v2_a.closure_c1.model import C1Config, C1Model


def _records() -> list[dict[str, str]]:
    return [
        {"family": family, "example_id": f"{family.lower()}-{index:04d}"}
        for family in ("ERE", "CPS")
        for index in range(4096)
    ]


def test_balanced_schedule_is_deterministic_and_epoch_exact() -> None:
    first = train.build_balanced_schedule(_records())
    second = train.build_balanced_schedule(_records())
    assert first == second
    assert len(first) == 6144
    assert all(len(row.example_ids) == 8 for row in first)
    assert all(
        sum(value.startswith("ere-") for value in row.example_ids) == 4
        for row in first
    )
    for epoch in range(6):
        ids = [
            example_id
            for row in first
            if row.epoch == epoch
            for example_id in row.example_ids
        ]
        assert len(ids) == len(set(ids)) == 8192


def test_lr_loss_and_checkpoint_selection_are_frozen() -> None:
    assert train.loss_weights(2048) == (0.5, 1.0)
    assert train.loss_weights(2049) == (1.0, 0.5)
    assert train.lr_scale(256) == pytest.approx(1.0)
    assert train.lr_scale(6144) == pytest.approx(0.0)
    selected = train.select_trace_checkpoint(
        [
            {"update": 2048, "trace_nll_by_family": {"ERE": 1.0, "CPS": 1.2}},
            {"update": 2560, "trace_nll_by_family": {"ERE": 1.1, "CPS": 1.1}},
            {"update": 3072, "trace_nll_by_family": {"ERE": 1.1, "CPS": 1.1}},
        ]
    )
    assert selected["update"] == 2560


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.boundary = nn.Linear(4, 4)
        self.core = nn.Linear(4, 4)
        self.answer_head = nn.Linear(4, 9)
        self.trace_probe = nn.Linear(4, 7)

    def forward(
        self, source_hidden: torch.Tensor, source_mask: torch.Tensor, *, return_trajectory: bool
    ) -> dict[str, torch.Tensor]:
        del source_mask
        hidden = self.core(torch.tanh(self.boundary(source_hidden.mean(dim=1))))
        return {
            "logits": self.answer_head(hidden),
            "trajectory": hidden[:, None, None, :],
        }

    def trace_logits(
        self,
        trajectory: torch.Tensor,
        step_indices: torch.Tensor,
        global_positions: torch.Tensor,
        local_positions: torch.Tensor,
    ) -> torch.Tensor:
        del step_indices, global_positions, local_positions
        hidden = trajectory[:, 0, 0].unsqueeze(1).expand(-1, 2, -1)
        return self.trace_probe(hidden)


def test_primary_loop_writes_trace_selected_checkpoint(tmp_path: Path) -> None:
    schedule = [
        train.ScheduledBatch(1, 0, ("a", "b")),
        train.ScheduledBatch(2, 0, ("a", "b")),
    ]

    def provider(ids, *, epoch, trace_chunk_tokens):
        del epoch, trace_chunk_tokens
        size = len(ids)
        return {
            "source_hidden": torch.randn(size, 3, 4),
            "source_mask": torch.ones(size, 3, dtype=torch.bool),
            "answers": torch.zeros(size, dtype=torch.long),
            "trace_targets": torch.zeros(size, 2, dtype=torch.long),
            "trace_mask": torch.ones(size, 2, dtype=torch.bool),
            "trace_step_indices": torch.zeros(size, 2, dtype=torch.long),
            "trace_global_positions": torch.zeros(size, 2, dtype=torch.long),
            "trace_local_positions": torch.zeros(size, 2, dtype=torch.long),
        }

    def evaluate(model, update):
        del model
        return {"trace_nll_by_family": {"ERE": 2.0 / update, "CPS": 2.0 / update}}

    spec = train.TrainSpec(
        batch_size=2,
        family_batch_size=1,
        epochs=1,
        maximum_updates=2,
        minimum_selection_update=1,
        evaluation_interval=1,
        warmup_updates=1,
    )
    result = train.train_primary(
        _TinyModel(), schedule, provider, evaluate, tmp_path, device="cpu", spec=spec
    )
    assert result["passed"] is True
    assert result["selected"]["update"] == 2
    assert (tmp_path / result["selected"]["checkpoint"]).is_file()


def test_overfit_strip_integrity_is_an_independent_answer_check() -> None:
    config = C1Config(
        source_width=4,
        latent_width=8,
        attention_heads=2,
        ffn_width=16,
        trace_vocab_size=5597,
        max_global_positions=16,
        max_local_positions=16,
    )
    model = C1Model(config)
    ids = [f"x-{index}" for index in range(8)]
    source = torch.randn(8, 3, 4)
    mask = torch.ones(8, 3, dtype=torch.bool)
    with torch.inference_mode():
        answers = model(source, mask, return_trajectory=False)["logits"].argmax(-1)

    def provider(requested, *, epoch, trace_chunk_tokens):
        del epoch, trace_chunk_tokens
        positions = [ids.index(value) for value in requested]
        return {
            "source_hidden": source[positions],
            "source_mask": mask[positions],
            "answers": answers[positions],
        }

    report, artifact = train.overfit_strip_integrity(
        model, ids, provider, device="cpu"
    )
    assert report["passed"] is True
    assert report["max_abs_diff"] == 0.0
    assert artifact.model.trace_probe is None
