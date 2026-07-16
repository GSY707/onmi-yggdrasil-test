import inspect
from pathlib import Path

import torch

from yggdrasil_v2.reasoning_medium.a1_8_data import A18DatasetSpec, build_a18_dataset, load_a18_records
from yggdrasil_v2.reasoning_medium.a1_10_cache import FORBIDDEN_CACHE_FIELDS, select_length_balanced_overfit32
from yggdrasil_v2.reasoning_medium.a1_10_interventions import build_a110_counterfactual_records
from yggdrasil_v2.reasoning_medium.a1_10_model import A110AnonymousRecurrentReasoner, A110ModelConfig
from yggdrasil_v2.reasoning_medium.a1_10_train import compute_a110_loss, encode_a110_targets


def _data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    build_a18_dataset(
        data_dir,
        A18DatasetSpec(
            data_seed=20261001,
            train_size=64,
            validation_size=32,
            short_size=24,
            in_range_size=24,
            relation_in_range_size=24,
            ood_size=24,
            relation_ood_size=24,
            causal_size=32,
        ),
    )
    return data_dir


def test_a110_anonymous_model_has_no_oracle_or_register_forward_inputs(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    records = load_a18_records(data_dir, "train")[:4]
    model = A110AnonymousRecurrentReasoner(
        A110ModelConfig(
            source_width=16,
            latent_width=32,
            workspace_slots=8,
            attention_heads=4,
            ffn_width=64,
            recurrent_layers=2,
        )
    )
    signature = inspect.signature(model.forward)
    assert "operation_mask" not in signature.parameters
    assert "program_length" not in signature.parameters
    assert "family_hidden" not in signature.parameters
    assert "source_role_hidden" not in signature.parameters
    assert "target_hidden" not in signature.parameters

    hidden = torch.randn(4, 20, 16)
    mask = torch.ones(4, 20, dtype=torch.bool)
    output = model(hidden, mask, recurrent_steps=16)
    labels = encode_a110_targets(records, 16, "cpu")
    loss, metrics = compute_a110_loss(output, labels)
    loss.backward()

    assert output["trajectory"].shape == (4, 16, 8, 32)
    assert output["state_logits"].shape == (4, 16, 3, 10)
    assert output["answer_logits"].shape == (4, 10)
    assert metrics["state_loss_weight"] == 1.0
    assert metrics["answer_loss_weight"] == 1.0
    integrity = model.integrity_report()
    assert integrity["passed"]
    assert integrity["workspace_slots_anonymous"]
    assert not integrity["explicit_register_slots"]
    assert not integrity["oracle_span_mask_input"]
    assert integrity["recurrent_weights_shared_across_steps"]


def test_a110_slot_permutation_invariance_and_target_extension(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    record = next(record for record in load_a18_records(data_dir, "train") if record["program_length"] == 1)
    model = A110AnonymousRecurrentReasoner(
        A110ModelConfig(source_width=16, latent_width=32, workspace_slots=8, attention_heads=4, ffn_width=64)
    ).eval()
    hidden = torch.randn(2, 12, 16)
    mask = torch.ones(2, 12, dtype=torch.bool)
    with torch.no_grad():
        normal = model(hidden, mask, recurrent_steps=4)
        permutation = torch.arange(7, -1, -1)
        permuted = model(hidden, mask, recurrent_steps=4, slot_permutation=permutation)
    assert torch.allclose(normal["answer_logits"], permuted["answer_logits"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(normal["state_logits"], permuted["state_logits"], atol=1e-6, rtol=1e-6)

    labels = encode_a110_targets([record], 4, "cpu")["state_targets"]
    assert torch.equal(labels[:, 0], labels[:, 1])
    assert torch.equal(labels[:, 1], labels[:, 3])


def test_a110_overfit_selection_and_counterfactuals_do_not_expose_spans(tmp_path: Path) -> None:
    data_dir = _data(tmp_path)
    selected = select_length_balanced_overfit32(load_a18_records(data_dir, "train"))
    assert len(selected) == 32
    assert {length: sum(record["program_length"] == length for record in selected) for length in range(1, 17)} == {
        length: 2 for length in range(1, 17)
    }

    variants = build_a110_counterfactual_records(load_a18_records(data_dir, "causal_core"), seed=17)
    assert all(variants[name] for name in ("prefix", "replacement", "deletion", "operation_shuffle", "query_swap"))
    forbidden_record_fields = {"start_value_spans", "operation_spans", "query_register_span", "operation_mask"}
    assert all(
        not (forbidden_record_fields & set(pair["record"]))
        for pairs in variants.values()
        for pair in pairs
    )
    assert "operation_token_mask" in FORBIDDEN_CACHE_FIELDS
    assert "input_ids" in FORBIDDEN_CACHE_FIELDS
