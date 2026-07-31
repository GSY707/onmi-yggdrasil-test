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
    _anchor_window_context,
    _aligned_window_pointer_logits,
    _operation_count_from_gains,
    _shared_operation_viterbi_decode,
    _shared_entity_viterbi_decode,
)
from yggdrasil_v2.reasoning_medium.a1_20c_supervision import (
    _record_char_targets,
)
from yggdrasil_v2.reasoning_medium.a1_20c_train import (
    _compatible_qwen_cache_contract,
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


def test_a120d_identity_window_reads_tokens_after_predicted_anchor() -> None:
    values = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    logits = torch.full((1, 1, 6), -100.0)
    logits[:, :, 1] = 100.0
    observed = _anchor_window_context(
        logits,
        values,
        torch.ones(1, 6, dtype=torch.bool),
        start_offset=1,
        width=3,
    )
    assert torch.allclose(observed, torch.tensor([[[3.0]]]))


def test_a120d_aligned_identity_window_preserves_token_order() -> None:
    entities = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )
    queries = entities[:, (0,)]
    query_projection = torch.nn.Linear(2, 2, bias=False)
    entity_projection = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        query_projection.weight.copy_(torch.eye(2))
        entity_projection.weight.copy_(torch.eye(2))
    logits = _aligned_window_pointer_logits(
        queries,
        entities,
        query_projection,
        entity_projection,
        scale=20.0,
    )
    assert logits.shape == (1, 1, 2)
    assert logits.argmax(dim=-1).item() == 0
    assert torch.equal(entities[:, 0].mean(dim=-2), entities[:, 1].mean(dim=-2))


def test_a120d_shared_operation_viterbi_stops_before_query() -> None:
    logits = torch.full((1, 3, 3, 14), -8.0)
    for slot in range(3):
        logits[:, slot, 0, 1] = 9.0
        logits[:, slot, 0, 5] = 9.0
        logits[:, slot, 1, 2] = 9.0
        logits[:, slot, 1, 6] = 9.0
        logits[:, slot, 2, 3] = 9.0
        logits[:, slot, 2, 7] = 9.0
    probabilities, selected, observed, gains = (
        _shared_operation_viterbi_decode(
            logits,
            torch.tensor([10]),
            triple_gain_threshold=5.0,
        )
    )
    assert torch.equal(
        observed, torch.tensor([[True, True, False]])
    )
    assert torch.equal(
        selected[0, :2].reshape(-1),
        torch.tensor([1, 2, 3, 5, 6, 7]),
    )
    assert torch.equal(
        probabilities.argmax(dim=-1)[0, :2].reshape(-1),
        torch.tensor([1, 2, 3, 5, 6, 7]),
    )
    assert gains[0, 0] > 5
    assert gains[0, 1] < 5


def test_a120d_operation_viterbi_excludes_unsupported_voter() -> None:
    logits = torch.full((1, 3, 3, 14), -8.0)
    for slot in range(2):
        logits[:, slot, 0, 1] = 9.0
        logits[:, slot, 1, 2] = 9.0
        logits[:, slot, 2, 3] = 9.0
    # The capacity-only third reader emits a stronger but invalid path.
    logits[:, 2, 0, 5] = 20.0
    logits[:, 2, 1, 6] = 20.0
    logits[:, 2, 2, 7] = 20.0
    probabilities, selected, observed, _ = (
        _shared_operation_viterbi_decode(
            logits,
            torch.tensor([10]),
            triple_gain_threshold=50.0,
            potential_voters=2,
        )
    )
    assert torch.equal(observed, torch.tensor([[True, False, False]]))
    assert torch.equal(selected[0, 0], torch.tensor([1, 2, 3]))
    assert torch.equal(
        probabilities.argmax(dim=-1)[0, 0], torch.tensor([1, 2, 3])
    )


def test_a120d_operation_count_uses_crossing_hysteresis() -> None:
    gains = torch.tensor(
        [
            [30.0, 11.5, 11.0, 8.0],
            [30.0, 21.0, 12.0, 8.0],
        ]
    )
    observed = _operation_count_from_gains(
        gains,
        supported_gain_threshold=19.0,
        recurrent_gain_threshold=10.5,
        potential_voters=2,
    )
    # A background peak before the crossing cannot unlock recurrence.  Once
    # the high-confidence crossing succeeds, attenuated true gains may extend
    # the recurrent path.
    assert torch.equal(observed, torch.tensor([2, 4]))


def test_a120d_entity_boundary_can_use_shared_operation_decode() -> None:
    names = torch.full((1, 5, 14), -8.0)
    values = torch.full((1, 5, 14), -8.0)
    for slot in range(3):
        names[:, slot, 1] = 9.0
        names[:, slot, 4] = 9.0
        values[:, slot, 2] = 9.0
        values[:, slot, 5] = 9.0
    decoded_program_start = torch.tensor([9])
    *_, observed, _ = _shared_entity_viterbi_decode(
        names, values, decoded_program_start
    )
    assert torch.equal(
        observed,
        torch.tensor([[True, True, False, False, False]]),
    )


def test_a120d_shared_entity_viterbi_stops_before_fake_pair() -> None:
    names = torch.full((1, 5, 16), -8.0)
    values = torch.full((1, 5, 16), -8.0)
    for slot in range(4):
        names[:, slot, 1] = 9.0
        names[:, slot, 4] = 9.0
        names[:, slot, 7] = 9.0
        values[:, slot, 2] = 9.0
        values[:, slot, 5] = 9.0
        values[:, slot, 8] = 9.0
    (
        _,
        _,
        decoded_names,
        decoded_values,
        observed,
        gains,
    ) = _shared_entity_viterbi_decode(
        names, values, torch.tensor([12])
    )
    assert torch.equal(
        observed,
        torch.tensor([[True, True, True, False, False]]),
    )
    assert torch.equal(decoded_names[0, :3], torch.tensor([1, 4, 7]))
    assert torch.equal(decoded_values[0, :3], torch.tensor([2, 5, 8]))
    assert gains[0, 1] > 5
    assert gains[0, 2] < 5


def test_a120d_eval_cache_can_raise_no_truncation_length_cap() -> None:
    stable = {
        "data_manifest_sha256": "data",
        "model_id": "qwen",
        "revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "dtype": "float16",
        "hidden_layer": "last",
        "silent_truncation": False,
    }
    observed = _compatible_qwen_cache_contract(
        {**stable, "max_length": 1280},
        {**stable, "max_length": 1024},
    )
    assert observed["training_max_length"] == 1024
    assert observed["evaluation_max_length"] == 1280


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
    assert output["entity_presence_logits"].shape == (4, 5)
    assert output["operation_presence_logits"].shape == (4, 32)
    assert output["source_pointer_logits"].shape == (4, 32, 5)
    assert output["target_pointer_logits"].shape == (4, 32, 5)
    assert output["query_pointer_logits"].shape == (4, 5)
    assert output["hard_forward_state_max_abs_delta"] <= 1e-6
    assert output["hard_forward_answer_max_abs_delta"] <= 1e-6
    report = model.integrity_report()
    assert report["passed"]
    assert report["structural_entity_cardinality"]
    assert report["shared_entity_viterbi_decode"]
    assert report["shared_operation_viterbi_decode"]
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
