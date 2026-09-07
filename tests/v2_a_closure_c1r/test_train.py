from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from yggdrasil_v2.v2_a.closure_c1.model import C1Config, C1Model
from yggdrasil_v2.v2_a.closure_c1r import train


def _model(*, probe: bool) -> C1Model:
    return C1Model(
        C1Config(
            source_width=4,
            latent_width=8,
            slots=8,
            boundary_depth=2,
            core_depth=2,
            attention_heads=2,
            ffn_width=16,
            recurrent_steps=10,
            answer_classes=9,
            trace_vocab_size=5597,
            max_global_positions=1024,
            max_local_positions=512,
        ),
        with_trace_probe=probe,
    )


class _CountingModel(C1Model):
    def __init__(self) -> None:
        super().__init__(_model_config(), with_trace_probe=True)
        self.trace_calls = 0

    def trace_logits(self, *args, **kwargs):
        self.trace_calls += 1
        return super().trace_logits(*args, **kwargs)


def _model_config() -> C1Config:
    return C1Config(
        source_width=4,
        latent_width=8,
        slots=8,
        boundary_depth=2,
        core_depth=2,
        attention_heads=2,
        ffn_width=16,
        recurrent_steps=10,
        answer_classes=9,
        trace_vocab_size=5597,
        max_global_positions=1024,
        max_local_positions=512,
    )


def _records(count_per_family: int = 4) -> list[dict[str, str]]:
    return [
        {"family": family, "example_id": f"{family.lower()}-{index}"}
        for family in ("ERE", "CPS")
        for index in range(count_per_family)
    ]


def test_schedule_and_prefetch_are_deterministic_and_ordered() -> None:
    spec = train.StageASpec(batch_size=4, family_batch_size=2, epochs=2, maximum_updates=4, warmup_updates=1)
    first = train.build_balanced_schedule(_records(), seed=7, spec=spec)
    assert first == train.build_balanced_schedule(_records(), seed=7, spec=spec)
    assert [item for item in train.ordered_prefetch(range(7), lambda item: item * 3, max_prefetch=2)] == [0, 3, 6, 9, 12, 15, 18]
    for epoch in range(2):
        ids = [item for row in first if row.epoch == epoch for item in row.example_ids]
        assert len(ids) == len(set(ids)) == 8


def test_stage_a_is_answer_only_and_writes_final_endpoint(tmp_path: Path) -> None:
    model = _model(probe=False)
    records = _records()
    source = {row["example_id"]: torch.randn(3, 4) for row in records}
    answers = {row["example_id"]: torch.tensor(index % 9) for index, row in enumerate(records)}

    def provider(ids, *, epoch):
        del epoch
        return {
            "source_hidden": torch.stack([source[item] for item in ids]),
            "source_mask": torch.ones(len(ids), 3, dtype=torch.bool),
            "answers": torch.stack([answers[item] for item in ids]),
        }

    result = train.train_stage_a(
        model,
        records,
        provider,
        tmp_path,
        device="cpu",
        spec=train.StageASpec(batch_size=4, family_batch_size=2, epochs=2, maximum_updates=4, warmup_updates=1),
    )
    assert result["trace_loss"] is False
    assert result["checkpoint_selection"] == "fixed_final_endpoint"
    assert result["optimizer_steps"] == 4
    assert Path(result["checkpoint"]).is_file()
    assert model.has_trace_probe is False
    assert result["gradient_norms_finite"] is True
    assert result["parameter_state_finite"] is True


def test_stage_b_freezes_model_and_normalizes_all_width65_blocks(tmp_path: Path) -> None:
    model = _CountingModel()
    records = _records()
    cache = {}
    for index, record in enumerate(records):
        length = 70 + index
        targets = torch.arange(length, dtype=torch.long) % 5597
        rows = []
        for start in range(0, length, 65):
            stop = min(length, start + 65)
            width = stop - start
            rows.append(
                {
                    "targets": targets[start:stop],
                    "mask": torch.ones(width, dtype=torch.bool),
                    "step_indices": torch.ones(width, dtype=torch.long),
                    "global_positions": torch.arange(start, stop),
                    "local_positions": torch.arange(start, stop),
                }
            )
        cache[record["example_id"]] = {
            "trajectory": torch.randn(10, 8, 8),
            "target_rows": rows,
        }
    frozen_before = {name: train.parameter_hash(model, names=[name]) for name, p in model.named_parameters() if not name.startswith("trace_probe.")}
    result = train.train_stage_b(
        model,
        records,
        cache,
        tmp_path,
        device="cpu",
        spec=train.StageBSpec(batch_size=4, family_batch_size=2, epochs=2, maximum_updates=4, warmup_updates=1),
    )
    frozen_after = {name: train.parameter_hash(model, names=[name]) for name, p in model.named_parameters() if not name.startswith("trace_probe.")}
    assert result["optimizer_steps"] == 4
    assert result["trace_block_width"] == 65
    assert result["trace_forward_calls"] == model.trace_calls
    assert result["trace_forward_calls"] < result["trace_blocks"]
    assert result["one_optimizer_step_per_batch"] is True
    assert result["frozen_parameter_hashes_unchanged"] is True
    assert frozen_before == frozen_after
    assert all(name.startswith("trace_probe.") for name in result["trainable_parameters"])
    assert result["processed_target_tokens_by_epoch"][0] == result["processed_target_tokens_by_epoch"][1]
    assert Path(result["checkpoint"]).is_file()
    assert result["gradient_norms_finite"] is True
    assert result["parameter_state_finite"] is True


def test_numeric_safety_helpers_reject_non_finite_values() -> None:
    with pytest.raises(FloatingPointError, match="gradient norm"):
        train._finite_gradient_norm(torch.tensor(float("nan")))
    model = torch.nn.Linear(2, 2)
    assert train._parameters_are_finite(model) is True
    with torch.no_grad():
        model.weight[0, 0] = float("inf")
    assert train._parameters_are_finite(model) is False
