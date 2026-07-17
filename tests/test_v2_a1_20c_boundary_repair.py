from __future__ import annotations

import torch

from yggdrasil_v2.reasoning_medium.a1_19h_h2_data import (
    A119H2DatasetSpec,
    build_a119h2_dataset,
    load_a119h2_records,
)
from yggdrasil_v2.reasoning_medium.a1_19h_model import (
    A119HConfig,
    A119HHybridReasoner,
)
from yggdrasil_v2.reasoning_medium.a1_20b_model import (
    A120BBoundaryConfig,
    A120BLearnedFullTextBoundary,
)
from yggdrasil_v2.reasoning_medium.a1_20b_train import encode_a120b_labels
from yggdrasil_v2.reasoning_medium.a1_20c_model import (
    A120CConfig,
    A120CFullTextBoundary,
)
from yggdrasil_v2.reasoning_medium.a1_20c_supervision import (
    _record_char_targets,
)


def _records(tmp_path):
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
    return load_a119h2_records(data_dir, "train")


def _core() -> A119HHybridReasoner:
    return A119HHybridReasoner(
        A119HConfig(
            maximum_entities=5,
            workspace_slots=10,
            training_auxiliary="none",
        )
    )


def test_a120c_supervision_targets_ordered_roles(tmp_path) -> None:
    record = _records(tmp_path)[0]
    names, values, operations, query = _record_char_targets(record)
    question = record["question"]
    assert [question[position : position + len(name)] for position, name in zip(names, record["entity_names"])] == list(record["entity_names"])
    assert [
        question[position : position + 1] for position in values
    ] == [record["start_state"][name] for name in record["entity_names"]]
    for positions, operation in zip(operations, record["operations"]):
        assert question[
            positions[0] : positions[0] + len(operation["family"])
        ] == operation["family"].upper()
        assert question[
            positions[1] : positions[1] + len(operation["source"])
        ] == operation["source"]
        assert question[
            positions[2] : positions[2] + len(operation["target"])
        ] == operation["target"]
    query_name = record["query_register"]
    assert question[query : query + len(query_name)] == query_name


def test_a120c_flat_hard_is_a120b_exact() -> None:
    source_hidden = torch.randn(3, 31, 32)
    source_mask = torch.ones(3, 31, dtype=torch.bool)
    torch.manual_seed(120)
    baseline = A120BLearnedFullTextBoundary(
        A120BBoundaryConfig(
            source_width=32, reader_layers=1, ffn_width=128
        ),
        _core(),
    )
    torch.manual_seed(120)
    repaired = A120CFullTextBoundary(
        A120CConfig(
            source_width=32,
            compiler="flat",
            credit_mode="hard_local",
            reader_layers=1,
            ffn_width=128,
        ),
        _core(),
    )
    baseline.eval()
    repaired.eval()
    expected = baseline(source_hidden, source_mask)
    observed = repaired(source_hidden, source_mask)
    assert torch.equal(expected["state_logits"], observed["state_logits"])
    assert torch.equal(
        expected["source_pointer_logits"],
        observed["source_pointer_logits"],
    )


def test_a120c_hierarchical_outputs_learned_anchors() -> None:
    model = A120CFullTextBoundary(
        A120CConfig(
            source_width=32,
            compiler="hierarchical",
            credit_mode="hard_local",
            reader_layers=1,
            ffn_width=128,
        ),
        _core(),
    )
    output = model(
        torch.randn(4, 29, 32), torch.ones(4, 29, dtype=torch.bool)
    )
    assert output["entity_name_anchor_logits"].shape == (4, 5, 29)
    assert output["entity_value_anchor_logits"].shape == (4, 5, 29)
    assert output["operation_anchor_logits"].shape == (4, 32, 3, 29)
    assert output["query_anchor_logits"].shape == (4, 29)
    report = model.integrity_report()
    assert report["passed"]
    assert not report["training_anchor_targets_are_forward_inputs"]


def test_a120d_factorized_separates_identity_payload_and_control() -> None:
    model = A120CFullTextBoundary(
        A120CConfig(
            source_width=32,
            compiler="factorized",
            credit_mode="hard_local",
            reader_layers=1,
            ffn_width=128,
        ),
        _core(),
    )
    output = model(
        torch.randn(4, 29, 32), torch.ones(4, 29, dtype=torch.bool)
    )
    assert output["entity_identity_latents"].shape == (4, 5, 256)
    assert output["entity_payload_latents"].shape == (4, 5, 256)
    assert output["entity_count_logits"].shape == (4, 5)
    assert output["operation_presence_logits"].shape == (4, 32)
    assert output["source_pointer_logits"].shape == (4, 32, 5)
    assert output["target_pointer_logits"].shape == (4, 32, 5)
    assert output["query_pointer_logits"].shape == (4, 5)
    assert output["hard_forward_state_max_abs_delta"] <= 1e-6
    assert output["hard_forward_answer_max_abs_delta"] <= 1e-6
    report = model.integrity_report()
    assert report["passed"]
    assert not report["training_anchor_targets_are_forward_inputs"]


def test_a120c_straight_through_is_hard_forward_and_restores_credit(
    tmp_path,
) -> None:
    records = _records(tmp_path)[:4]
    labels = encode_a120b_labels(records, "cpu")
    torch.manual_seed(120)
    model = A120CFullTextBoundary(
        A120CConfig(
            source_width=32,
            compiler="hierarchical",
            credit_mode="straight_through",
            reader_layers=1,
            ffn_width=128,
        ),
        _core(),
    )
    model.train()
    output = model(
        torch.randn(4, 37, 32), torch.ones(4, 37, dtype=torch.bool)
    )
    assert output["hard_forward_state_max_abs_delta"] <= 1e-6
    assert output["hard_forward_answer_max_abs_delta"] <= 1e-6
    tracked = (
        "entity_presence_logits",
        "operation_presence_logits",
        "family_mapping_logits",
        "source_pointer_logits",
        "target_pointer_logits",
    )
    for name in tracked:
        output[name].retain_grad()
    state_ce = torch.nn.functional.cross_entropy(
        output["state_logits"].reshape(-1, model.core.config.value_classes),
        labels.state_targets.reshape(-1),
        ignore_index=-100,
    )
    state_ce.backward()
    for name in tracked:
        assert output[name].grad is not None
        assert output[name].grad.norm() > 0
    assert all(parameter.grad is None for parameter in model.core.parameters())
