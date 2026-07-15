from __future__ import annotations

from pathlib import Path

import torch

from yggdrasil_v2.reasoning_medium.a1_5_data import (
    A15DatasetSpec,
    build_a15_dataset,
    counterfactual_report,
    load_a15_records,
)
from yggdrasil_v2.reasoning_medium.a1_5_model import P0Config, StructuredRecurrentCore
from yggdrasil_v2.reasoning_medium.a1_5_train import encode_records


def test_a15_contract_has_matched_composition_and_necessary_steps(tmp_path: Path) -> None:
    output = tmp_path / "a1_5"
    manifest = build_a15_dataset(
        output,
        A15DatasetSpec(seed=17, train_size=32, validation_size=8, test_size=8, composition_size=8, length_size=8),
    )
    assert manifest["schema_version"].endswith("a1.5.symbolic-state-machine.v1")
    assert manifest["contract"]["ordinary_and_composition_same_length_range"]
    assert manifest["splits"]["test"]["program_length_min"] == manifest["splits"]["composition_heldout"]["program_length_min"]
    assert manifest["splits"]["test"]["program_length_max"] == manifest["splits"]["composition_heldout"]["program_length_max"]
    assert manifest["splits"]["length_heldout"]["program_length_min"] >= 5
    all_records = load_a15_records(output, "test") + load_a15_records(output, "composition_heldout")
    report = counterfactual_report(all_records)
    assert report["necessary_rate"] == 1.0
    assert not report["failures"]
    for record in all_records:
        assert len(record["operation_spans"]) == record["program_length"]
        assert len(record["operation_mask"]) == record["program_length"]


def test_p0_has_shared_transition_variable_steps_and_no_gate() -> None:
    model = StructuredRecurrentCore(P0Config(latent_width=32, attention_heads=4, ffn_width=64))
    records = [
        {
            "start_state": {"amber": "A", "cobalt": "B", "jade": "C"},
            "query_register": "jade",
            "answer_index": 0,
            "answer": "A",
            "operations": [
                {"family": "copy", "source": "amber", "target": "cobalt"},
                {"family": "copy", "source": "cobalt", "target": "jade"},
            ],
            "state_trajectory": [
                {"amber": "A", "cobalt": "A", "jade": "C"},
                {"amber": "A", "cobalt": "A", "jade": "A"},
            ],
        },
        {
            "start_state": {"amber": "D", "cobalt": "E", "jade": "F"},
            "query_register": "amber",
            "answer_index": 3,
            "answer": "D",
            "operations": [{"family": "swap", "source": "amber", "target": "cobalt"}],
            "state_trajectory": [{"amber": "E", "cobalt": "D", "jade": "F"}],
        },
    ]
    batch = encode_records(records, "cpu")
    out = model(
        batch["start_values"],
        batch["query_register"],
        batch["operation_family"],
        batch["operation_source"],
        batch["operation_target"],
        batch["operation_mask"],
        return_trajectory=True,
    )
    assert out["logits"].shape == (2, 10)
    assert len(out["trajectory"]) == 2
    assert model.integrity_report()["transition_weights_shared"]
    assert model.integrity_report()["transition_delta_gate"].startswith("none")
    assert abs(model.transition.residual_scale.item() - 0.1) < 1e-6
    assert not any("step" in name for name, _ in model.named_parameters())
