from __future__ import annotations

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1s import contract
from yggdrasil_v2.v2_a.closure_c1s.metrics import (
    effective_slot_count,
    geometry_summary,
    registered_control_report,
)


def test_registered_controls_separate_addressed_positive_from_nulls():
    report = registered_control_report()
    thresholds = contract.S0_THRESHOLDS
    positive = report["addressed_positive"]
    uniform = report["legacy_uniform_mean_null"]

    assert report["query_swap_logit_l2"] >= thresholds["query_swap_min_logit_l2"]
    assert report["duplicate_query_swap_logit_l2"] <= thresholds["duplicate_query_swap_max_logit_l2"]
    assert min(positive["relevant_effect"]) >= thresholds["positive_relevant_mean_replace_min"]
    assert max(positive["irrelevant_max_effect"]) <= thresholds["positive_irrelevant_mean_replace_max"]
    assert max(uniform["ownership_contrast"]) <= thresholds["uniform_null_ownership_contrast_max"]
    assert report["k1_null"]["single_slot_null"] is True


def test_effective_slot_count_and_geometry_report_common_mode_without_calling_it_functional():
    uniform = torch.full((2, 4), 0.25)
    hard = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    assert effective_slot_count(uniform).tolist() == pytest.approx([4.0, 4.0])
    assert effective_slot_count(hard).tolist() == pytest.approx([1.0, 1.0])

    payloads = torch.ones((2, 4, 8))
    geometry = geometry_summary(payloads)
    assert geometry["slot_centered_energy"] == pytest.approx(0.0)
    assert geometry["mean_off_diagonal_cosine"] == pytest.approx(1.0)

