from __future__ import annotations

from dataclasses import asdict

import torch

from yggdrasil_v2.reasoning_medium.a1_7_data import _operation, _render_record
from yggdrasil_v2.reasoning_medium.a1_19h_assessment import classify_a119h_h1
from yggdrasil_v2.reasoning_medium.a1_19h_data import (
    A119HBatch,
    encode_a119h_records,
    index_a119h_batch,
    permute_entity_axis,
    validate_a119h_batch_addresses,
)
from yggdrasil_v2.reasoning_medium.a1_19h_model import (
    A119HConfig,
    A119HHybridReasoner,
)
from yggdrasil_v2.reasoning_medium.a1_19h_h2_assessment import classify_a119h_h2
from yggdrasil_v2.reasoning_medium.a1_19h_h2_data import (
    A119H2DatasetSpec,
    audit_a119h2_dataset,
    build_a119h2_dataset,
    load_a119h2_records,
)
from yggdrasil_v2.reasoning_medium.a1_19h_h2_train import (
    A119H2RecordSplit,
    _a119h_loss_tensors,
    _verify_cache_equivalence,
)
from yggdrasil_v2.reasoning_medium.a1_19h_train import (
    A119HTrainSpec,
    CHECKPOINT_SCHEMA,
    _export_deployment,
    compute_a119h_loss,
    load_a119h_deployment,
)


def _records() -> list[dict[str, object]]:
    operations = [
        _operation(("copy", "amber", "cobalt")),
        _operation(("swap", "cobalt", "jade")),
    ]
    return [
        _render_record(
            {"amber": "A", "cobalt": "B", "jade": "C"},
            operations,
            "amber",
            "unit",
            0,
        ),
        _render_record(
            {"amber": "D", "cobalt": "D", "jade": "F"},
            operations,
            "jade",
            "unit",
            1,
        ),
    ]


def test_a119h_opaque_handle_and_slot_invariance_is_exact() -> None:
    model = A119HHybridReasoner(A119HConfig(training_auxiliary="none"))
    model.eval()
    batch = encode_a119h_records(
        _records(), 2, "cpu", mapping_seed=77, maximum_entities=3
    )
    base = model(**batch.inputs, recurrent_steps=2)
    permutation = torch.tensor([2, 0, 1])
    permuted_batch = permute_entity_axis(batch, permutation)
    permuted = model(**permuted_batch.inputs, recurrent_steps=2)
    assert torch.equal(
        permuted["state_logits"], base["state_logits"][:, :, permutation]
    )
    assert torch.equal(permuted["answer_logits"], base["answer_logits"])
    aliased_batch = encode_a119h_records(
        _records(),
        2,
        "cpu",
        mapping_seed=77,
        maximum_entities=3,
        handle_alias_seed=99,
    )
    aliased = model(**aliased_batch.inputs, recurrent_steps=2)
    assert torch.equal(aliased["state_logits"], base["state_logits"])
    assert torch.equal(aliased["answer_logits"], base["answer_logits"])


def test_a119h_tsaux_is_set_equivariant_and_answer_is_query_coupled() -> None:
    model = A119HHybridReasoner(A119HConfig())
    model.train()
    batch = encode_a119h_records(
        _records(), 2, "cpu", mapping_seed=77, maximum_entities=3
    )
    output = model(**batch.inputs, recurrent_steps=2)
    assert output["training_auxiliary_logits"].shape == (2, 2, 3, 10)
    query_weights = (
        batch.inputs["entity_handles"]
        == batch.inputs["query_handle"].unsqueeze(-1)
    ).float()
    expected = torch.einsum(
        "be,bev->bv", query_weights, output["state_logits"][:, -1]
    )
    assert torch.equal(output["answer_logits"], expected)
    loss, metrics = compute_a119h_loss(model, output, batch, auxiliary_weight=1.0)
    loss.backward()
    assert metrics["official_answer_loss_weight"] == 0.0
    assert model.transition.operator[0].weight.grad is not None


def test_a119h_deployment_physically_strips_training_auxiliary(tmp_path) -> None:
    model = A119HHybridReasoner(A119HConfig())
    spec = A119HTrainSpec(
        model_seed=20261921,
        data_seed=20260721,
        mapping_seed=20261931,
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_config": asdict(model.config),
        "train_spec": asdict(spec),
        "data_manifest_sha256": "unit-test",
        "model": model.state_dict(),
    }
    path = tmp_path / "deployable.pt"
    report = _export_deployment(checkpoint, path)
    assert report["stripped_parameter_names"] == [
        "training_auxiliary_head.bias",
        "training_auxiliary_head.weight",
    ]
    deployed = load_a119h_deployment(path)
    assert deployed.training_auxiliary_head is None
    assert deployed.config.training_auxiliary == "none"
    assert not deployed.integrity_report()["fixed_register_slot_semantics"]


def test_a119h_h1_classification_requires_formal_and_causal_three_of_three() -> None:
    assert classify_a119h_h1(
        formal_passes=3, causal_passes=3, integrity_passed=True
    )["code"] == "opaque_handle_hybrid_core_confirmed"
    assert not classify_a119h_h1(
        formal_passes=3, causal_passes=2, integrity_passed=True
    )["h1_passed"]


def test_a119h_h2_data_holds_out_n5_and_relation(tmp_path) -> None:
    data_dir = tmp_path / "data"
    spec = A119H2DatasetSpec(
        data_seed=51,
        train_size=96,
        validation_size=48,
        short_size=36,
        in_range_size=36,
        relation_in_range_size=36,
        ood_size=36,
        relation_ood_size=36,
        entity_heldout_size=36,
        entity_relation_heldout_size=36,
        causal_size=36,
    )
    build_a119h2_dataset(data_dir, spec)
    audit = audit_a119h2_dataset(data_dir)
    assert audit["passed"], audit["gates"]
    assert {row["entity_count"] for row in load_a119h2_records(data_dir, "train")} == {
        2,
        3,
        4,
    }
    assert {row["entity_count"] for row in load_a119h2_records(data_dir, "entity_heldout")} == {5}


def test_a119h_h2_variable_mask_keeps_address_equivariance(tmp_path) -> None:
    data_dir = tmp_path / "data"
    spec = A119H2DatasetSpec(
        data_seed=52,
        train_size=48,
        validation_size=24,
        short_size=18,
        in_range_size=18,
        relation_in_range_size=18,
        ood_size=18,
        relation_ood_size=18,
        entity_heldout_size=18,
        entity_relation_heldout_size=18,
        causal_size=18,
    )
    build_a119h2_dataset(data_dir, spec)
    records = load_a119h2_records(data_dir, "train")[:9] + load_a119h2_records(
        data_dir, "entity_heldout"
    )[:3]
    model = A119HHybridReasoner(
        A119HConfig(maximum_entities=5, workspace_slots=10, training_auxiliary="none")
    )
    model.eval()
    batch = encode_a119h_records(
        records, 16, "cpu", mapping_seed=72, maximum_entities=5
    )
    base = model(**batch.inputs, recurrent_steps=16)
    permutation = torch.tensor([4, 3, 2, 1, 0])
    permuted_batch = permute_entity_axis(batch, permutation)
    permuted = model(**permuted_batch.inputs, recurrent_steps=16)
    assert torch.equal(
        permuted["state_logits"], base["state_logits"][:, :, permutation]
    )
    assert torch.equal(permuted["answer_logits"], base["answer_logits"])


def test_a119h_h2_classification_requires_full_three_seed_closure() -> None:
    assert classify_a119h_h2(
        formal_passes=3, causal_passes=3, integrity_passed=True
    )["code"] == "generalized_hybrid_core_confirmed"
    assert not classify_a119h_h2(
        formal_passes=2, causal_passes=2, integrity_passed=True
    )["a119h_complete"]


def test_a119h_h2_cached_encoding_is_exact(tmp_path) -> None:
    data_dir = tmp_path / "data"
    build_a119h2_dataset(
        data_dir,
        A119H2DatasetSpec(
            data_seed=53,
            train_size=96,
            validation_size=24,
            short_size=18,
            in_range_size=18,
            relation_in_range_size=18,
            ood_size=18,
            relation_ood_size=18,
            entity_heldout_size=18,
            entity_relation_heldout_size=18,
            causal_size=18,
        ),
    )
    dataset = A119H2RecordSplit(load_a119h2_records(data_dir, "train"))
    cache = encode_a119h_records(
        dataset.records,
        16,
        "cpu",
        mapping_seed=73,
        maximum_entities=5,
        include_semantic_slot_orders=False,
    )
    validate_a119h_batch_addresses(cache)
    report = _verify_cache_equivalence(
        dataset,
        cache,
        [0, 7, 11, 29, 45],
        mapping_seed=73,
        maximum_entities=5,
        recurrent_steps=16,
    )
    assert report["passed"]
    selected = index_a119h_batch(cache, torch.tensor([0, 7, 11, 29, 45]))
    assert selected.state_targets.shape == (5, 16, 5)


def test_a119h_h2_hot_loss_matches_frozen_objective_exactly() -> None:
    model = A119HHybridReasoner(
        A119HConfig(maximum_entities=3, workspace_slots=8)
    )
    model.train()
    batch = encode_a119h_records(
        _records(), 2, "cpu", mapping_seed=77, maximum_entities=3
    )
    output = model(**batch.inputs, recurrent_steps=2)
    reference_loss, reference_metrics = compute_a119h_loss(
        model, output, batch, auxiliary_weight=1.0
    )
    optimized_loss, _ = _a119h_loss_tensors(
        model, output, batch, auxiliary_weight=1.0
    )
    assert torch.equal(reference_loss, optimized_loss)
    assert reference_metrics["closure_weight"] == 1.0


def test_a119h_address_validation_is_outside_forward_but_remains_strict() -> None:
    batch = encode_a119h_records(
        _records(), 2, "cpu", mapping_seed=77, maximum_entities=3
    )
    validate_a119h_batch_addresses(batch)
    broken_inputs = dict(batch.inputs)
    broken_inputs["query_handle"] = torch.full_like(
        broken_inputs["query_handle"], -999
    )
    broken = A119HBatch(
        inputs=broken_inputs,
        state_targets=batch.state_targets,
        answer_targets=batch.answer_targets,
        semantic_slot_orders=batch.semantic_slot_orders,
    )
    try:
        validate_a119h_batch_addresses(broken)
    except ValueError as error:
        assert "query handle" in str(error)
    else:
        raise AssertionError("invalid query address was not rejected")
