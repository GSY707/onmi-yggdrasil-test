from __future__ import annotations

from dataclasses import replace
import hashlib

import torch

from yggdrasil_v2.v2_a.closure_c1t import contract
from yggdrasil_v2.v2_a.closure_c1t.model import C1TConfig, C1TModel
from yggdrasil_v2.v2_a.closure_c1t.runtime import (
    C1TRuntime,
    build_independent_card_cache,
)
from yggdrasil_v2.v2_a.closure_c1t.tasks import build_s1_bank
from yggdrasil_v2.v2_a.closure_c1t.train import (
    RuntimeBatchProvider,
    TrainSpec,
    build_s1_schedule,
    load_endpoint,
    save_endpoint,
    schedule_report,
    set_determinism,
    train_fixed_endpoint,
)


def _encoder(text: str) -> dict[str, torch.Tensor]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tokens, width = 2 + len(text) % 3, 12
    return {
        "hidden": torch.tensor(
            [
                [digest[(row + column) % 32] / 255.0 for column in range(width)]
                for row in range(tokens)
            ],
            dtype=torch.float32,
        ),
        "mask": torch.ones(tokens, dtype=torch.bool),
        "token_ids": torch.tensor([digest[index] for index in range(tokens)]),
    }


def _runtime() -> tuple[list[dict[str, object]], C1TRuntime]:
    records = build_s1_bank()
    cache = build_independent_card_cache(
        records, _encoder, source_identity={"model": "fixture", "revision": "fixed"}
    )
    return records, C1TRuntime(records, cache, address_width=16)


def test_s1_schedule_keeps_complete_groups_and_is_deterministic() -> None:
    records = build_s1_bank()
    left = build_s1_schedule(records, seed=19, maximum_updates=8)
    right = build_s1_schedule(records, seed=19, maximum_updates=8)
    assert left == right
    report = schedule_report(left)
    assert report["updates"] == 8
    assert report["cycles"] == 2
    assert report["complete_groups_per_batch"] is True
    assert set(report["group_counts"].values()) == {2}
    for row in left:
        assert len(row.example_ids) == 8
        assert len({value.rsplit("-", 1)[0] for value in row.example_ids}) == 2


def test_small_training_writes_only_one_fixed_endpoint(tmp_path) -> None:
    records, runtime = _runtime()
    schedule = build_s1_schedule(records, seed=23, maximum_updates=2)
    spec = replace(
        TrainSpec(),
        maximum_updates=2,
        warmup_updates=1,
        log_interval=1,
        pin_memory=False,
    )
    set_determinism(29)
    model = C1TModel(
        C1TConfig(
            source_width=12,
            payload_width=16,
            address_width=16,
            ffn_width=32,
        )
    )
    seen: list[int] = []
    training = train_fixed_endpoint(
        model,
        RuntimeBatchProvider(runtime),
        schedule,
        identity="TEST_C1T_S1",
        model_seed=29,
        order_seed=23,
        spec=spec,
        device="cpu",
        on_optimizer_step=lambda row: seen.append(int(row["update"])),
    )
    assert seen == [1, 2]
    assert training["optimizer_steps"] == 2
    assert training["model_writes"] == 0
    endpoint = tmp_path / "fixed.pt"
    digest = save_endpoint(
        endpoint,
        model=model,
        identity="TEST_C1T_S1",
        update=2,
        schedule_sha256=schedule_report(schedule)["sha256"],
    )
    loaded, payload = load_endpoint(endpoint)
    assert len(digest) == 64
    assert payload["update"] == 2
    assert payload["optimizer_state_saved"] is False
    assert payload["checkpoint_selection"] is False
    assert loaded.integrity_report()["passed"] is True
