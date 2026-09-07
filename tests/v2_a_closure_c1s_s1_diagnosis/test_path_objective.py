from __future__ import annotations

import torch

from yggdrasil_v2.v2_a.closure_c1s.model import C1SConfig, C1SModel
from yggdrasil_v2.v2_a.closure_c1s_successor.objective import SuccessorDiagnosticHeads
from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis.objective_alignment import (
    audit_objective_alignment,
)
from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis.path_analysis import (
    _mean_excluding_index,
    collect_paths,
    summarise_paths,
)


class _Runtime:
    def get_batch(self, example_ids, *, slots: int, pin_memory: bool = False):
        assert not pin_memory
        batch = len(example_ids)
        state_values = torch.zeros(batch, 10, slots, 8, dtype=torch.long)
        state_values[:, :, :, 0] = torch.arange(10).view(1, 10, 1) % 2
        state_values[:, :, :, 1] = torch.arange(slots).view(1, 1, slots) % 2
        return {
            "source_hidden": torch.randn(batch, 4, 2048),
            "source_mask": torch.ones(batch, 4, dtype=torch.bool),
            "answers": torch.tensor([index % 9 for index in range(batch)]),
            "presence": torch.ones(batch, slots, dtype=torch.bool),
            "content_mask": torch.ones(batch, slots, dtype=torch.bool),
            "query_owner": torch.zeros(batch, dtype=torch.long),
            "operation_active": torch.tensor([[True, False] * 5] * batch),
            "state_values": state_values,
            "state_feature_mask": torch.ones(batch, 10, slots, 8, dtype=torch.bool),
            "state_step_mask": torch.ones(batch, 10, dtype=torch.bool),
            "families": ["CPS" if str(value).startswith("c") else "ERE" for value in example_ids],
            "state_feature_names": [
                ["processed", "running_best", "final_winner", "u3", "u4", "u5", "u6", "u7"]
                if str(value).startswith("c")
                else ["touched", "changed", "operation_source", "operation_target", "u4", "u5", "u6", "u7"]
                for value in example_ids
            ],
        }


def _model():
    config = C1SConfig(
        source_width=2048,
        payload_width=16,
        address_width=8,
        slots=2,
        operations=10,
        reader_depth=1,
        attention_heads=4,
        ffn_width=32,
        answer_classes=9,
        auxiliary_width=8,
    )
    return (
        C1SModel(config),
        SuccessorDiagnosticHeads(config.source_width, config.address_width, config.auxiliary_width),
    )


def test_path_collection_keeps_family_cells_separate() -> None:
    model, heads = _model()
    collection = collect_paths(
        model, heads, _Runtime(), ["c0", "e0"], device="cpu", batch_size=1
    )
    assert collection["step_logits"].shape == (2, 11, 9)
    assert collection["initial_slot_replace_logits"].shape == (2, 2, 9)
    report = summarise_paths(collection, bootstrap_seed=7, bootstrap_replicates=20)
    assert set(report["families"]) == {"CPS", "ERE"}
    assert report["overall"]["role"] == "descriptive_summary_only"
    for family in ("CPS", "ERE"):
        assert report["families"][family]["answer_permutation_metric_null"]["control_valid"] is False
        assert "h0_query_owner_margin_drop" in report["families"][family]
    assert all("h0_relevant_margin_drop_lower" in row for row in report["decision_rows"])


def test_objective_audit_uses_autograd_grad_without_mutation() -> None:
    model, heads = _model()
    report = audit_objective_alignment(
        model,
        heads,
        _Runtime(),
        ["c0", "e0"],
        device="cpu",
        batch_size=1,
    )
    assert report["mutation_audit"]["state_unchanged"] is True
    assert report["mutation_audit"]["parameter_grad_fields_empty"] is True
    assert report["mutation_audit"]["optimizer_steps"] == 0
    assert {row["family"] for row in report["loss_rows"]} == {"CPS", "ERE"}


def test_answer_permutation_control_is_deterministic_when_family_has_label_variation() -> None:
    model, heads = _model()
    collection = collect_paths(
        model,
        heads,
        _Runtime(),
        ["c0", "c1", "e0", "e1"],
        device="cpu",
        batch_size=4,
    )
    left = summarise_paths(collection, bootstrap_seed=11, bootstrap_replicates=25)
    right = summarise_paths(collection, bootstrap_seed=11, bootstrap_replicates=25)
    for family in ("CPS", "ERE"):
        control = left["families"][family]["answer_permutation_metric_null"]
        assert control["control_valid"] is True
        assert control == right["families"][family]["answer_permutation_metric_null"]


def test_leave_one_out_replacement_excludes_selected_payload() -> None:
    payloads = torch.tensor([[[1.0], [5.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    selected = torch.tensor([0])
    replacement = _mean_excluding_index(payloads, mask, selected)
    assert replacement.tolist() == [[5.0]]
