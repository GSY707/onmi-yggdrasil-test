from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from yggdrasil_v2.v2_a.closure_c1_diagnosis.contract import POSITION_BINS
from yggdrasil_v2.v2_a.closure_c1_diagnosis.gradients import shared_gradient_report
from yggdrasil_v2.v2_a.closure_c1_diagnosis.runner import attribution_decision
from yggdrasil_v2.v2_a.closure_c1_diagnosis.trace_metrics import (
    alignment_decision,
    selection_metric_decision,
    trace_checkpoint_metrics,
)


class FakeTraceModel(nn.Module):
    def __init__(self, vocab: int = 16) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(self, source_hidden: torch.Tensor, source_mask: torch.Tensor, *, return_trajectory: bool = False):
        del source_mask
        trajectory = torch.zeros(source_hidden.shape[0], 10, 8, 2, device=source_hidden.device)
        return {"trajectory": trajectory}

    def trace_logits(
        self,
        trajectory: torch.Tensor,
        steps: torch.Tensor,
        global_positions: torch.Tensor,
        local_positions: torch.Tensor,
    ) -> torch.Tensor:
        del steps, local_positions
        result = torch.zeros(*global_positions.shape, self.vocab, device=trajectory.device)
        winners = global_positions.remainder(self.vocab).unsqueeze(-1)
        return result.scatter(-1, winners, 4.0)


class FakeDataset:
    def get_batch(self, ids, *, device="cpu"):
        hidden = torch.zeros(len(ids), 2, 2, device=device)
        mask = torch.ones(len(ids), 2, dtype=torch.bool, device=device)
        return {"source_hidden": hidden, "source_mask": mask}


def _trace_records() -> tuple[list[dict], dict]:
    records = [
        {"example_id": "ere-validation-0", "family": "ERE", "split": "validation"},
        {"example_id": "ere-validation-1", "family": "ERE", "split": "validation"},
        {"example_id": "cps-validation-0", "family": "CPS", "split": "validation"},
        {"example_id": "cps-validation-1", "family": "CPS", "split": "validation"},
    ]
    bank = {"targets": {}}
    for row in records:
        positions = [0, 1, 64, 128]
        bank["targets"][row["example_id"]] = {
            "global_positions": positions,
            "global_token_ids": [position % 16 for position in positions],
            "token_ids": [position % 16 for position in positions],
            "grammar_mask": [True, False, True, False],
            "local_positions": [0, 1, 0, 1],
            "step_indices": [1, 1, 2, 10],
        }
    return records, bank


def test_trace_metrics_reports_topk_grammar_steps_and_position_bins() -> None:
    records, bank = _trace_records()
    report = trace_checkpoint_metrics(
        FakeTraceModel(),
        FakeDataset(),
        records,
        bank["targets"],
        device="cpu",
        batch_size=2,
        per_family=2,
        token_chunk=2,
    )
    assert report["records"] == 4
    assert report["alignment_modes"] == ["registered"]
    for family in ("ERE", "CPS"):
        metrics = report["modes"]["registered"][family]
        assert metrics["all"]["top1_accuracy"] == pytest.approx(1.0)
        assert metrics["all"]["top5_accuracy"] == pytest.approx(1.0)
        assert metrics["grammar"]["tokens"] == 4
        assert metrics["content"]["tokens"] == 4
        assert metrics["steps"]["10"]["tokens"] == 2
        assert metrics["position_bins"]["000-063"]["tokens"] == 4
        assert metrics["position_bins"]["064-127"]["tokens"] == 2
        assert metrics["position_bins"]["128-191"]["tokens"] == 2


def test_trace_metrics_can_stratify_real_train_exposure_counts() -> None:
    records, bank = _trace_records()
    for row in records:
        row["split"] = "train"
        row["example_id"] = row["example_id"].replace("validation", "train")
    renamed = {}
    for old, row in list(bank["targets"].items()):
        new = old.replace("validation", "train")
        renamed[new] = row
    bank["targets"] = renamed
    exposure = {row["example_id"]: [0, 1, 2, 0] for row in records}
    report = trace_checkpoint_metrics(
        FakeTraceModel(),
        FakeDataset(),
        records,
        bank["targets"],
        device="cpu",
        batch_size=2,
        per_family=2,
        token_chunk=2,
        split="train",
        exposure_counts=exposure,
    )
    assert report["split"] == "train"
    for family in ("ERE", "CPS"):
        groups = report["modes"]["registered"][family]["exposure"]
        assert groups["0"]["all"]["tokens"] == 4
        assert groups["1"]["all"]["tokens"] == 2
        assert groups["2+"]["all"]["tokens"] == 2
        assert report["paired_exposure_effect"]["registered"][family]["paired_records"] == 2


def _mode_metrics(nll: float, top1: float) -> dict:
    return {"all": {"nll": nll, "top1_accuracy": top1}}


def _decision_report(registered: tuple[float, float], alternative: tuple[float, float]) -> dict:
    return {
        "modes": {
            "registered": {family: _mode_metrics(*registered) for family in ("ERE", "CPS")},
            "previous": {family: _mode_metrics(*alternative) for family in ("ERE", "CPS")},
            "next": {family: _mode_metrics(*alternative) for family in ("ERE", "CPS")},
            "constant_h1": {family: _mode_metrics(*alternative) for family in ("ERE", "CPS")},
            "constant_h10": {family: _mode_metrics(*alternative) for family in ("ERE", "CPS")},
        }
    }


def test_alignment_and_selection_metric_decisions_obey_registered_thresholds() -> None:
    report = _decision_report((2.0, 0.50), (1.94, 0.53))
    decision = alignment_decision(report)
    assert decision["alignment_suspect"] is True
    assert set(decision["qualified_modes_both_families"]) == {
        "previous", "next", "constant_h1", "constant_h10"
    }

    weak = alignment_decision(_decision_report((2.0, 0.50), (1.96, 0.51)))
    assert weak["alignment_suspect"] is False

    selected = _decision_report((2.0, 0.50), (2.0, 0.50))
    final = _decision_report((1.9, 0.53), (1.9, 0.53))
    assert selection_metric_decision(selected, final)["selection_metric_mismatch"] is True
    final["modes"]["registered"]["CPS"]["all"]["top1_accuracy"] = 0.51
    assert selection_metric_decision(selected, final)["selection_metric_mismatch"] is False


@dataclass
class Scheduled:
    update: int
    epoch: int
    example_ids: tuple[str, ...]


class FakeGradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.boundary = nn.Linear(2, 2)
        self.core_layers = nn.Linear(2, 2)
        self.trace_head = nn.Linear(2, 5)

    def forward(self, source_hidden: torch.Tensor, source_mask: torch.Tensor, *, return_trajectory: bool = False):
        del source_mask
        hidden = source_hidden.mean(dim=1)
        state = torch.tanh(self.core_layers(torch.tanh(self.boundary(hidden))))
        logits = state[:, :2]
        trajectory = state[:, None, None, :].expand(-1, 10, 8, -1)
        return {"logits": logits, "trajectory": trajectory}

    def trace_logits(self, trajectory, steps, global_positions, local_positions):
        del steps, global_positions, local_positions
        state = trajectory[:, 0, 0, :]
        return self.trace_head(state)[:, None, :].expand(-1, 2, -1)


def test_shared_gradient_report_is_read_only_and_has_no_optimizer_step() -> None:
    model = FakeGradientModel()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    schedule = [Scheduled(index + 1, 0, (f"id-{index}-a", f"id-{index}-b")) for index in range(2)]

    def provider(example_ids, *, epoch, trace_chunk_tokens):
        del example_ids, epoch, trace_chunk_tokens
        return {
            "source_hidden": torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.5], [1.0, -1.0]]]),
            "source_mask": torch.ones(2, 2, dtype=torch.bool),
            "answers": torch.tensor([0, 1]),
            "trace_targets": torch.tensor([[0, 1], [2, 3]]),
            "trace_mask": torch.ones(2, 2, dtype=torch.bool),
            "trace_step_indices": torch.ones(2, 2, dtype=torch.long),
            "trace_global_positions": torch.zeros(2, 2, dtype=torch.long),
            "trace_local_positions": torch.zeros(2, 2, dtype=torch.long),
        }

    report = shared_gradient_report(model, schedule, provider, device="cpu", batches=2)
    assert report["optimizer_steps"] == 0
    assert report["parameters_unchanged"] is True
    assert report["parameter_state_before"] == report["parameter_state_after"]
    assert all(value is None for value in model.parameters() if value.requires_grad for value in [value.grad])
    assert len(report["rows"]) == 2
    assert set(report["summaries"]) == {"shared", "boundary", "core"}
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_attribution_keeps_g007_mechanism_and_architecture_status_separate() -> None:
    decision = attribution_decision(
        pin={"passed": True},
        alignment={"passed": True},
        exposure={"complete_cycle": False},
        train_exposure_metrics={
            "paired_exposure_effect": {
                "registered": {
                    "ERE": {"qualified": False},
                    "CPS": {"qualified": True},
                }
            }
        },
        alignment_comparison={"alignment_suspect": False},
        selection_comparison={"selection_metric_mismatch": False},
        gradients={
            "decision": {"late_trace_dominates": False, "majority_conflict": False}
        },
        poststop_gates={"passed": False},
    )
    assert decision["g007_primary"] == "EXPOSURE_PRIMARY"
    assert decision["architecture_status"] == "POSTSTOP_ARCHITECTURE_COUNTERFACTUAL_FAIL"
    assert "POSTSTOP_ARCHITECTURE_GATES_WEAK" in decision["contributing_causes"]
