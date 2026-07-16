from pathlib import Path

import torch

from yggdrasil_v2.reasoning_medium.a1_7_core import A17CoreConfig, A17RelationAddressedCore
from yggdrasil_v2.reasoning_medium.a1_7_train import encode_a17_records
from yggdrasil_v2.reasoning_medium.a1_8_data import A18DatasetSpec, build_a18_dataset, load_a18_records
from yggdrasil_v2.reasoning_medium.a1_9_cache import collate_a19_items, select_length_balanced_overfit32
from yggdrasil_v2.reasoning_medium.a1_9_interventions import build_a19_interventions, clone_recomputed_record
from yggdrasil_v2.reasoning_medium.a1_9_model import A19BoundaryConfig, A19FrozenQwenBoundary, module_state_sha256
from yggdrasil_v2.reasoning_medium.a1_9_train import compute_a19_loss


def _item(record: dict, width: int = 16) -> dict:
    length = int(record["program_length"])
    return {
        "start_hidden": torch.randn(3, width),
        "query_hidden": torch.randn(width),
        "operation_hidden": torch.randn(length, 3, width),
        "record": record,
    }


def test_a19_frozen_core_and_boundary_shapes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    build_a18_dataset(
        data_dir,
        A18DatasetSpec(
            data_seed=20260901,
            train_size=64,
            validation_size=32,
            short_size=24,
            in_range_size=24,
            relation_in_range_size=24,
            ood_size=24,
            relation_ood_size=24,
            causal_size=16,
        ),
    )
    records = load_a18_records(data_dir, "train")[:8]
    inputs, joined = collate_a19_items([_item(record) for record in records], "cpu")
    core = A17RelationAddressedCore(A17CoreConfig(latent_width=32, ffn_width=64))
    core_hash = module_state_sha256(core)
    model = A19FrozenQwenBoundary(A19BoundaryConfig(source_width=16, latent_width=32), core)
    output = model(**inputs)
    labels = encode_a17_records(joined, "cpu")
    loss, metrics = compute_a19_loss(model, output, labels)
    loss.backward()

    assert output["trajectory"].shape[:3] == (8, max(record["program_length"] for record in records), 3)
    assert output["value_mapping_logits"].shape == (8, 3, 10)
    assert output["family_mapping_logits"].shape[-1] == 2
    assert output["query_mapping_logits"].shape == (8, 3)
    assert metrics["mapping_ce_weight"] == 1.0
    assert all(parameter.grad is None for parameter in model.core.parameters())
    assert module_state_sha256(model.core) == core_hash
    assert model.integrity_report()["passed"]
    assert model.integrity_report()["source_target_query_use_same_register_adapter"]


def test_a19_overfit_selection_and_hidden_intervention_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    build_a18_dataset(
        data_dir,
        A18DatasetSpec(
            data_seed=20260902,
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
    selected = select_length_balanced_overfit32(load_a18_records(data_dir, "train"))
    assert len(selected) == 32
    assert {length: sum(record["program_length"] == length for record in selected) for length in range(1, 17)} == {
        length: 2 for length in range(1, 17)
    }

    causal = load_a18_records(data_dir, "causal_core")
    items = [_item(record) for record in causal]
    interventions = build_a19_interventions(items, seed=19)
    assert len(interventions["prefix"]) == sum(record["program_length"] for record in causal)
    assert len(interventions["no_hidden"]) == len(causal)
    assert len(interventions["independent_role_shuffle"]) == len(causal)
    assert interventions["replacement"]
    assert interventions["query_swap"]
    assert all(torch.count_nonzero(item["start_hidden"]) == 0 for item in interventions["no_hidden"])

    record = causal[0]
    alternate_query = next(name for name in ("amber", "cobalt", "jade") if name != record["query_register"])
    recomputed = clone_recomputed_record(record, query_register=alternate_query)
    assert recomputed["state_trajectory"] == record["state_trajectory"]
    assert recomputed["query_register"] == alternate_query
