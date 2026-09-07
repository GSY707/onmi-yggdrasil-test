from __future__ import annotations

import torch

from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis import contract
from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis.folds import (
    build_fold_ledger,
)
from yggdrasil_v2.v2_a.closure_c1s_s1_temporal_diagnosis.temporal import (
    audit_temporal_readout,
)

from .test_folds import CPS_NAMES, ERE_NAMES, make_target_rows


def make_collection() -> tuple[dict, dict]:
    rows = make_target_rows()
    ledger = build_fold_ledger(rows)
    global_channel = {
        (family, feature): channel
        for channel, (family, feature) in enumerate(
            (pair for family, names in contract.HARD_FEATURES.items() for pair in ((family, name) for name in names))
        )
    }
    state_features = torch.zeros(32, 4, 2, 6)
    state_logits = torch.zeros(32, 4, 2, 8)
    state_values = torch.zeros(32, 4, 2, 8, dtype=torch.bool)
    state_masks = torch.ones(32, 4, 2, 8, dtype=torch.bool)
    names: list[list[str]] = []
    families: list[str] = []
    ids: list[str] = []
    for index, row in enumerate(rows):
        family = row["family"]
        local_names = CPS_NAMES if family == "CPS" else ERE_NAMES
        values = torch.tensor(row["state_values"], dtype=torch.bool)
        state_values[index] = values
        for feature in contract.HARD_FEATURES[family]:
            local = local_names.index(feature)
            channel = global_channel[(family, feature)]
            signal = values[:, :, local].float() * 2.0 - 1.0
            state_features[index, :, :, channel] = signal + index * 1.0e-3
            state_logits[index, :, :, local] = signal * 5.0
        names.append(local_names)
        families.append(family)
        ids.append(row["example_id"])
    initial = torch.ones(32, 2, 3)
    trajectory = torch.stack(
        [initial + (step + 1) * 0.1 for step in range(4)], dim=1
    )
    collection = {
        "example_ids": ids,
        "families": families,
        "feature_names": names,
        "initial_payloads": initial,
        "trajectory": trajectory,
        "state_features": state_features,
        "state_logits": state_logits,
        "state_values": state_values,
        "state_feature_mask": state_masks,
        "state_step_mask": torch.ones(32, 4, dtype=torch.bool),
        "operation_active_target": torch.ones(32, 4, 2, dtype=torch.bool),
    }
    return collection, ledger


def test_temporal_readout_uses_frozen_support_ledger_for_all_cells() -> None:
    collection, ledger = make_collection()
    report = audit_temporal_readout(
        collection,
        ledger,
        ranks=[1, 2],
        regularizations=[0.01],
        fit_null_seed=17,
        score_null_seed=19,
        score_null_replicates=100,
        source_visible_replay=True,
    )
    assert set(report["families"]) == {"CPS", "ERE"}
    assert len(report["decision_rows"]) == 6
    for family, features in contract.HARD_FEATURES.items():
        for feature in features:
            cell = report["families"][family]["features"][feature]
            expected = 12 if (family, feature) == ("CPS", "running_best") else 16
            assert cell["natural"]["record_macro_valid_records"] == expected
            assert cell["frozen_decoder"]["record_macro_valid_records"] == expected
            assert len(cell["natural"]["record_evidence"]) == expected
            assert len(cell["frozen_decoder"]["record_evidence"]) == expected
            assert len(cell["fixed_channel"]["record_evidence"]) == expected
            assert set(cell["fit_nulls"]) == {
                "time_shuffle",
                "temporal_mean",
                "within_record_target_shuffle",
            }
            assert all(
                len(null["record_evidence"]) == expected
                for null in cell["fit_nulls"].values()
            )
            assert cell["score_null"]["replicates"] == 100
    assert report["families"]["CPS"]["features"]["running_best"][
        "ineligible_ids"
    ] == ["cps-02", "cps-05", "cps-08", "cps-11"]


def test_same_input_and_ledger_are_deterministic() -> None:
    collection, ledger = make_collection()
    kwargs = dict(
        ranks=[1],
        regularizations=[0.01],
        fit_null_seed=23,
        score_null_seed=29,
        score_null_replicates=20,
        source_visible_replay=True,
    )
    assert audit_temporal_readout(collection, ledger, **kwargs) == audit_temporal_readout(
        collection, ledger, **kwargs
    )
