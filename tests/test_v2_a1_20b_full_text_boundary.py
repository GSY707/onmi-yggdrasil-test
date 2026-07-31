from __future__ import annotations

import random
from types import SimpleNamespace

import torch

from yggdrasil_v2.reasoning_medium.a1_19h_data import encode_a119h_records
from yggdrasil_v2.reasoning_medium.a1_19h_h2_data import (
    A119H2DatasetSpec,
    build_a119h2_dataset,
    load_a119h2_records,
)
from yggdrasil_v2.reasoning_medium.a1_19h_model import (
    A119HConfig,
    A119HHybridReasoner,
)
from yggdrasil_v2.reasoning_medium.a1_20b_cache import (
    CACHE_SCHEMA,
    _recover_cached_split,
    select_a120b_overfit32,
)
from yggdrasil_v2.reasoning_medium.a1_20b_diagnostics import (
    ORACLE_FIELDS,
    select_a120b_variant_controls,
)
from yggdrasil_v2.reasoning_medium.a1_20b_model import (
    A120BBoundaryConfig,
    A120BLearnedFullTextBoundary,
)
from yggdrasil_v2.reasoning_medium.a1_20b_train import (
    compute_a120b_loss,
    encode_a120b_labels,
    sample_a120b_indices,
)


def _data(tmp_path):
    data_dir = tmp_path / "data"
    build_a119h2_dataset(
        data_dir,
        A119H2DatasetSpec(
            data_seed=120,
            train_size=192,
            validation_size=48,
            short_size=36,
            in_range_size=36,
            relation_in_range_size=36,
            ood_size=36,
            relation_ood_size=36,
            entity_heldout_size=36,
            entity_relation_heldout_size=36,
            causal_size=36,
        ),
    )
    return data_dir


def test_a120b_continuous_payload_core_path_matches_one_hot_exactly(tmp_path) -> None:
    records = load_a119h2_records(_data(tmp_path), "train")[:8]
    model = A119HHybridReasoner(
        A119HConfig(
            maximum_entities=5,
            workspace_slots=10,
            training_auxiliary="none",
        )
    )
    model.eval()
    batch = encode_a119h_records(
        records, 16, "cpu", mapping_seed=1207, maximum_entities=5
    )
    baseline = model(**batch.inputs, recurrent_steps=16)
    probabilities = torch.nn.functional.one_hot(
        batch.inputs["entity_values"], num_classes=model.config.value_classes
    ).float()
    payloads = model.value_initializer(probabilities) + model.shared_entity_seed
    continuous = model(
        **batch.inputs, recurrent_steps=16, entity_payloads=payloads
    )
    assert torch.equal(baseline["state_logits"], continuous["state_logits"])
    assert torch.equal(baseline["answer_logits"], continuous["answer_logits"])


def test_a120b_interrupted_cache_can_be_strictly_recovered(tmp_path) -> None:
    records = sorted(
        load_a119h2_records(_data(tmp_path), "train")[:3],
        key=lambda record: (
            int(record["entity_count"]),
            int(record["program_length"]),
            record["fingerprint"],
        ),
    )
    cache_dir = tmp_path / "cache" / "train"
    cache_dir.mkdir(parents=True)
    for shard_index, start in enumerate(range(0, len(records), 2)):
        shard_records = records[start : start + 2]
        lengths = torch.tensor(
            [5 + index for index in range(len(shard_records))],
            dtype=torch.long,
        )
        maximum = int(lengths.max())
        mask = torch.arange(maximum).unsqueeze(0) < lengths.unsqueeze(1)
        torch.save(
            {
                "schema_version": CACHE_SCHEMA,
                "last_hidden": torch.randn(
                    len(shard_records), maximum, 8, dtype=torch.float16
                ),
                "attention_mask": mask,
                "token_lengths": lengths,
                "fingerprints": [
                    record["fingerprint"] for record in shard_records
                ],
                "example_ids": [
                    record["example_id"] for record in shard_records
                ],
            },
            cache_dir / f"shard_{shard_index:05d}.pt",
        )
    report = _recover_cached_split(
        records, cache_dir, max_length=32, shard_size=2
    )
    assert report["examples"] == 3
    assert report["source_tokens"] == 16
    assert report["maximum_token_length"] == 6
    assert report["recovered_existing_shards"] is True


def test_a120b_boundary_predicts_all_masks_roles_and_controls(tmp_path) -> None:
    records = load_a119h2_records(_data(tmp_path), "train")[:4]
    core = A119HHybridReasoner(
        A119HConfig(
            maximum_entities=5,
            workspace_slots=10,
            training_auxiliary="none",
        )
    )
    model = A120BLearnedFullTextBoundary(
        A120BBoundaryConfig(
            source_width=32,
            reader_layers=1,
            ffn_width=128,
        ),
        core,
    )
    source_hidden = torch.randn(4, 23, 32)
    source_mask = torch.ones(4, 23, dtype=torch.bool)
    output = model(source_hidden, source_mask)
    assert output["entity_presence_logits"].shape == (4, 5)
    assert output["operation_presence_logits"].shape == (4, 32)
    assert output["value_mapping_logits"].shape == (4, 5, 10)
    assert output["source_pointer_logits"].shape == (4, 32, 5)
    assert output["state_logits"].shape == (4, 32, 5, 10)
    report = model.integrity_report()
    assert report["passed"]
    assert not report["oracle_entity_mask_input"]
    assert not report["oracle_operation_mask_input"]
    assert report["continuous_payload_to_core"]

    labels = encode_a120b_labels(records, "cpu")
    loss, _ = compute_a120b_loss(model, output, labels)
    loss.backward()
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("core.")
    )
    assert all(parameter.grad is None for parameter in model.core.parameters())


def test_a120b_overfit_selection_covers_counts_and_lengths(tmp_path) -> None:
    records = load_a119h2_records(_data(tmp_path), "train")
    selected = select_a120b_overfit32(records)
    assert len(selected) == 32
    assert {int(record["entity_count"]) for record in selected} == {2, 3, 4}
    assert {int(record["program_length"]) for record in selected} == set(range(1, 17))


def test_a120b_formal_sampler_balances_counts_and_all_train_lengths() -> None:
    cells = {
        (count, length): [count * 100 + length]
        for count in (2, 3, 4)
        for length in range(1, 17)
    }
    dataset = SimpleNamespace(
        indices_by_entity_count={
            count: [count * 100 + length for length in range(1, 17)]
            for count in (2, 3, 4)
        },
        indices_by_length={
            length: [count * 100 + length for count in (2, 3, 4)]
            for length in range(1, 17)
        },
        indices_by_cell=cells,
    )
    indices = sample_a120b_indices(dataset, 16, random.Random(120))
    assert {index % 100 for index in indices} == set(range(1, 17))
    counts = [index // 100 for index in indices]
    assert set(counts) == {2, 3, 4}
    assert max(counts.count(count) for count in (2, 3, 4)) - min(
        counts.count(count) for count in (2, 3, 4)
    ) <= 1


def test_a120b_diagnostic_oracle_selection_is_explicit(tmp_path) -> None:
    records = load_a119h2_records(_data(tmp_path), "train")[:2]
    labels = encode_a120b_labels(records, "cpu")
    output = {
        "predicted_entity_mask": ~labels.entity_presence.bool(),
        "predicted_operation_mask": ~labels.operation_presence.bool(),
        "predicted_family": torch.zeros_like(labels.family_targets),
        "predicted_source_pointer": torch.zeros_like(labels.source_targets),
        "predicted_target_pointer": torch.zeros_like(labels.target_targets),
        "predicted_query_pointer": torch.zeros_like(labels.query_targets),
    }
    predicted = select_a120b_variant_controls(output, labels, frozenset())
    oracle = select_a120b_variant_controls(output, labels, ORACLE_FIELDS)
    assert torch.equal(predicted["entity_mask"], output["predicted_entity_mask"])
    assert torch.equal(oracle["entity_mask"], labels.entity_presence.bool())
    assert torch.equal(oracle["operation_mask"], labels.operation_presence.bool())
    assert torch.equal(oracle["family"], labels.family_targets.clamp_min(0))
    assert torch.equal(oracle["source"], labels.source_targets.clamp_min(0))
    assert torch.equal(oracle["target"], labels.target_targets.clamp_min(0))
    assert torch.equal(oracle["query"], labels.query_targets)
    assert oracle["oracle_values"] is True


def test_a120b_state_loss_cannot_cross_hard_discrete_controls(tmp_path) -> None:
    records = load_a119h2_records(_data(tmp_path), "train")[:4]
    core = A119HHybridReasoner(
        A119HConfig(
            maximum_entities=5,
            workspace_slots=10,
            training_auxiliary="none",
        )
    )
    model = A120BLearnedFullTextBoundary(
        A120BBoundaryConfig(
            source_width=32,
            reader_layers=1,
            ffn_width=128,
        ),
        core,
    )
    labels = encode_a120b_labels(records, "cpu")
    output = model(torch.randn(4, 23, 32), torch.ones(4, 23, dtype=torch.bool))
    tracked = (
        "entity_presence_logits",
        "operation_presence_logits",
        "value_mapping_logits",
        "family_mapping_logits",
        "source_pointer_logits",
        "target_pointer_logits",
        "query_pointer_logits",
    )
    for name in tracked:
        output[name].retain_grad()
    state_ce = torch.nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, core.config.value_classes),
        labels.state_targets.reshape(-1),
        ignore_index=-100,
    )
    state_ce.backward()
    assert output["value_mapping_logits"].grad is not None
    assert output["value_mapping_logits"].grad.norm() > 0
    for name in tracked:
        if name != "value_mapping_logits":
            assert output[name].grad is None
