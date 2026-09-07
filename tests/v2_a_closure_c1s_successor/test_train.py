from __future__ import annotations

import torch
import pytest

from yggdrasil_v2.v2_a.closure_c1s.model import C1SConfig, C1SModel
from yggdrasil_v2.v2_a.closure_c1s_successor import train as successor_train
from yggdrasil_v2.v2_a.closure_c1s_successor.objective import SuccessorDiagnosticHeads
from yggdrasil_v2.v2_a.closure_c1s_successor.train import (
    ScheduledBatch,
    TrainSpec,
    build_balanced_schedule,
    build_overfit_schedule,
    load_endpoint,
    lr_scale,
    save_endpoint,
    schedule_report,
    train_fixed_endpoint,
)


def _records(count: int):
    return [
        {"example_id": f"ere-{index}", "family": "ERE"} for index in range(count)
    ] + [
        {"example_id": f"cps-{index}", "family": "CPS"} for index in range(count)
    ]


class TinyProvider:
    def get_batch(self, example_ids, *, slots, pin_memory=False):
        del pin_memory
        batch = len(example_ids)
        source = torch.randn(batch, 7, 16)
        active = torch.zeros(batch, 10, dtype=torch.bool)
        active[:, :slots] = True
        target_owner = torch.full((batch, 10), -1, dtype=torch.long)
        source_owner = torch.full((batch, 10), -1, dtype=torch.long)
        for step in range(slots):
            target_owner[:, step] = step
            source_owner[:, step] = (step - 1) % slots
        presence = torch.ones(batch, slots, dtype=torch.bool)
        state_values = torch.zeros(batch, 10, slots, 8)
        state_feature_mask = active.unsqueeze(-1).unsqueeze(-1).expand_as(state_values)
        return {
            "source_hidden": source,
            "source_mask": torch.ones(batch, 7, dtype=torch.bool),
            "answers": torch.zeros(batch, dtype=torch.long),
            "object_labels": torch.full((batch, slots), -1, dtype=torch.long),
            "operation_active": active,
            "source_owner": source_owner,
            "target_owner": target_owner,
            "query_owner": torch.zeros(batch, dtype=torch.long),
            "presence": presence,
            "content_mask": presence.clone(),
            "span_start": torch.zeros(batch, slots, dtype=torch.long),
            "span_end": torch.ones(batch, slots, dtype=torch.long),
            "state_values": state_values,
            "state_feature_mask": state_feature_mask,
            "state_step_mask": active,
            "alternate_owner": torch.full((batch,), 1 if slots > 1 else -1, dtype=torch.long),
            "alternate_label": torch.full((batch,), -1, dtype=torch.long),
            "irrelevant_owner": torch.full((batch,), -1, dtype=torch.long),
            "mechanism_supported": torch.ones(batch, dtype=torch.bool),
            "query_answer_independent": torch.ones(batch, dtype=torch.bool),
            "example_ids": list(example_ids),
            "families": ["ERE"] * batch,
        }


def _model(slots: int = 2):
    config = C1SConfig(
        source_width=16,
        payload_width=16,
        address_width=8,
        slots=slots,
        operations=10,
        reader_depth=1,
        attention_heads=4,
        ffn_width=32,
        answer_classes=9,
        auxiliary_width=8,
    )
    return C1SModel(config), SuccessorDiagnosticHeads(16, 8, 8)


def test_balanced_schedules_are_frozen_and_disjoint_per_epoch() -> None:
    spec = TrainSpec(batch_size=8, family_batch_size=4, epochs=2, maximum_updates=4)
    schedule = build_balanced_schedule(_records(8), seed=7, spec=spec)
    assert len(schedule) == 4
    assert all(len(row.example_ids) == 8 for row in schedule)
    assert schedule_report(schedule)["updates"] == 4
    overfit = build_overfit_schedule(_records(16), seed=9, maximum_updates=17)
    assert len(overfit) == 17
    assert len(set(value for row in overfit[:4] for value in row.example_ids)) == 32


def test_lr_has_warmup_and_fixed_zero_endpoint() -> None:
    spec = TrainSpec(maximum_updates=10, warmup_updates=2)
    assert lr_scale(1, spec) == 0.5
    assert lr_scale(2, spec) == 1.0
    assert abs(lr_scale(10, spec)) < 1.0e-12


def test_one_step_cpu_training_and_endpoint_round_trip(tmp_path) -> None:
    model, heads = _model()
    schedule = [ScheduledBatch(1, 0, ("a", "b"))]
    spec = TrainSpec(
        batch_size=2,
        family_batch_size=1,
        epochs=1,
        maximum_updates=1,
        warmup_updates=1,
        log_interval=1,
        causal_interval=1,
        pin_memory=False,
    )
    report = train_fixed_endpoint(
        model,
        heads,
        TinyProvider(),
        schedule,
        identity="TEST",
        order_seed=2,
        model_seed=1,
        spec=spec,
        device="cpu",
    )
    assert report["completed_updates"] == 1
    path = tmp_path / "endpoint.pt"
    digest = save_endpoint(
        path,
        model=model,
        diagnostic_heads=heads,
        update=1,
        identity="TEST",
        schedule_sha256="A" * 64,
        metrics=report,
    )
    loaded, loaded_heads, payload = load_endpoint(path)
    assert len(digest) == 64
    assert payload["update"] == 1
    assert loaded.parameter_report() == model.parameter_report()
    assert set(loaded_heads.state_dict()) == set(heads.state_dict())


def test_endpoint_digest_failure_leaves_a_materialized_model_write(tmp_path, monkeypatch) -> None:
    model, heads = _model()
    path = tmp_path / "endpoint.pt"
    monkeypatch.setattr(
        successor_train.hashlib,
        "sha256",
        lambda: (_ for _ in ()).throw(OSError("digest unavailable")),
    )

    with pytest.raises(OSError, match="digest unavailable"):
        save_endpoint(
            path,
            model=model,
            diagnostic_heads=heads,
            update=1,
            identity="FAULT-TEST",
            schedule_sha256="A" * 64,
            metrics={"optimizer_steps": 1},
        )

    assert path.is_file()
    assert path.stat().st_size > 0


def test_optimizer_step_callback_records_mutation_before_later_failure() -> None:
    model, heads = _model()
    schedule = [ScheduledBatch(update, 0, ("a", "b")) for update in range(1, 4)]
    spec = TrainSpec(
        batch_size=2,
        family_batch_size=1,
        epochs=1,
        maximum_updates=3,
        warmup_updates=1,
        log_interval=1,
        causal_interval=1,
        pin_memory=False,
    )
    consumed: list[int] = []

    def record(update: int) -> None:
        consumed.append(update)
        if update == 2:
            raise RuntimeError("fault after optimizer mutation")

    with pytest.raises(RuntimeError, match="fault after optimizer mutation"):
        train_fixed_endpoint(
            model,
            heads,
            TinyProvider(),
            schedule,
            identity="FAULT-TEST",
            order_seed=2,
            model_seed=1,
            spec=spec,
            device="cpu",
            on_optimizer_step=record,
        )
    assert consumed == [1, 2]
