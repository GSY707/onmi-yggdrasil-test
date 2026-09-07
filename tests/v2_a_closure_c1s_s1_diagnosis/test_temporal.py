from __future__ import annotations

import torch
import numpy as np

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis.temporal import (
    _binary_ba,
    _record_macro_ba,
    _record_macro_null_gap,
    _select_ridge,
    _cross_fitted_channel_auc,
    audit_temporal_readout,
)


def _collection():
    records = 16
    steps = 10
    slots = 2
    features = 8
    width = 8
    families = ["CPS"] * 8 + ["ERE"] * 8
    names = []
    values = torch.zeros(records, steps, slots, features, dtype=torch.bool)
    mask = torch.zeros_like(values)
    latent = torch.randn(records, steps, slots, width) * 0.01
    for record, family in enumerate(families):
        row_names = (
            ["processed", "running_best", "final_winner", "u3", "u4", "u5", "u6", "u7"]
            if family == "CPS"
            else ["touched", "changed", "operation_source", "operation_target", "u4", "u5", "u6", "u7"]
        )
        names.append(row_names)
        hard = 2 if family == "CPS" else 4
        for feature in range(hard):
            target = (
                torch.arange(steps).view(steps, 1)
                + torch.arange(slots).view(1, slots)
                + record
                + feature
            ) % 2 == 0
            values[record, :, :, feature] = target
            mask[record, :, :, feature] = True
            latent[record, :, :, feature] += target.float() * 2.0 - 1.0
    state_logits = torch.where(values, torch.tensor(4.0), torch.tensor(-4.0))
    initial = torch.randn(records, slots, width)
    trajectory = initial[:, None].repeat(1, steps, 1, 1)
    trajectory += torch.arange(1, steps + 1).view(1, steps, 1, 1) * 0.1
    return {
        "example_ids": [f"r{index:02d}" for index in range(records)],
        "families": families,
        "feature_names": names,
        "state_features": latent,
        "state_values": values,
        "state_feature_mask": mask,
        "state_step_mask": torch.ones(records, steps, dtype=torch.bool),
        "state_logits": state_logits,
        "initial_payloads": initial,
        "trajectory": trajectory,
        "operation_active_target": torch.ones(records, steps, dtype=torch.bool),
    }


def test_temporal_readout_is_cross_fitted_and_excludes_final_winner() -> None:
    report = audit_temporal_readout(
        _collection(),
        ranks=[2],
        regularizations=[0.1],
        seed=17,
        target_positive_min=8,
        target_negative_min=8,
        target_changes_min=4,
        source_visible_replay=True,
    )
    assert set(report["families"]) == {"CPS", "ERE"}
    assert "final_winner" not in report["families"]["CPS"]["features"]
    assert report["readout"]["optimizer_steps"] == 0
    assert all(row["target_adequate"] for row in report["decision_rows"])


def test_temporal_primary_metric_never_mixes_observation_and_record_macro_units() -> None:
    target = torch.tensor([0, 1] * 4 + [0, 1], dtype=torch.bool).numpy()
    prediction = torch.tensor([0, 1] * 4 + [1, 0], dtype=torch.bool).numpy()
    records = torch.tensor([0] * 8 + [1] * 2).numpy()
    assert _binary_ba(prediction, target) == 0.8
    macro, valid, total = _record_macro_ba(prediction, target, records)
    assert macro == 0.5
    assert (valid, total) == (2, 2)

    natural = {
        "record_macro_balanced_accuracy": 0.60,
        "record_macro_valid_records": 2,
    }
    nulls = {
        "time_shuffle": {
            "record_macro_balanced_accuracy": 0.55,
            "record_macro_valid_records": 2,
        },
        "temporal_mean": {
            "record_macro_balanced_accuracy": 0.50,
            "record_macro_valid_records": 2,
        },
    }
    strongest, gap = _record_macro_null_gap(natural, nulls)
    assert strongest == 0.55
    assert abs(gap - 0.05) < 1.0e-12


def test_inner_ridge_selection_reports_record_macro_metric() -> None:
    records = torch.arange(6).repeat_interleave(2).numpy().astype(object)
    target = torch.tensor([0, 1] * 6, dtype=torch.bool).numpy()
    features = torch.tensor([[0.0], [1.0]] * 6).numpy()
    _rank, _regularization, candidates = _select_ridge(
        features,
        target,
        records,
        ranks=[1],
        regularizations=[0.1],
        salt="unit-record-macro",
    )
    assert "mean_inner_record_macro_balanced_accuracy" in candidates[0]
    assert "mean_inner_balanced_accuracy" not in candidates[0]


def test_cross_fitted_channel_auc_is_not_flipped_from_outer_heldout_targets() -> None:
    features = torch.tensor(
        [
            -0.3776, 2.0428, 0.6467, 0.6631, -0.5140, -1.6481,
            0.1675, 0.1090, -1.2274, -0.6832, -0.0720, -0.9448,
            -0.0983, 0.0955, 0.0356, -0.5063,
        ]
    ).view(-1, 1).numpy()
    target = torch.tensor([0, 1] * 8, dtype=torch.bool).numpy()
    records = np.repeat(np.asarray([f"r{index}" for index in range(8)], dtype=object), 2)
    auc = _cross_fitted_channel_auc(
        features, target, records, salt="leak-test"
    )
    assert auc == 0.25
    assert auc < 0.5
