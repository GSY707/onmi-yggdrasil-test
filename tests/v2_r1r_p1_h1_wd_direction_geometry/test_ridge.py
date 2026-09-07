from __future__ import annotations

import torch

from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.ridge import (
    GroupedSufficientStatistics,
    fit_grouped_ridge,
    gram_spectrum,
    sufficient_metric,
)


def test_streaming_grouped_ridge_recovers_shared_affine_maps() -> None:
    generator = torch.Generator().manual_seed(17)
    feature = torch.randn(120, 5, generator=generator)
    groups = torch.arange(120) % 2
    weights = torch.randn(2, 5, 3, generator=generator)
    biases = torch.randn(2, 3, generator=generator)
    target = torch.empty(120, 3)
    for group in (0, 1):
        selected = groups == group
        target[selected] = feature[selected] @ weights[group] + biases[group]

    fit_stats = GroupedSufficientStatistics(2, 5, 3)
    validation_stats = GroupedSufficientStatistics(2, 5, 3)
    for group in (0, 1):
        selected = torch.nonzero(groups == group, as_tuple=False).flatten()
        fit_index = selected[:48]
        validation_index = selected[48:]
        fit_stats.update(group, feature[fit_index], target[fit_index])
        validation_stats.update(
            group, feature[validation_index], target[validation_index]
        )
    fitted = fit_grouped_ridge(
        fit_stats,
        validation_stats,
        lambda_grid=(0.0, 1.0e-4, 1.0e-2),
    )
    metric = sufficient_metric(
        fit_stats.merge(validation_stats), fitted.coefficients
    )
    # Production accumulation intentionally forms per-batch Gram matrices in
    # fp32 before the small solve switches to fp64.
    assert metric["increment_nmse"] < 1.0e-7
    assert fitted.ridge_lambda == 0.0
    spectrum = gram_spectrum(fit_stats)
    assert spectrum["sum_group_ranks"] == 10


def test_grouped_ridge_prediction_rejects_bad_group_shape() -> None:
    stats = GroupedSufficientStatistics(1, 2, 1)
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    y = torch.tensor([[3.0], [5.0]])
    stats.update(0, x, y)
    fit = fit_grouped_ridge(stats, stats, lambda_grid=(1.0e-4,))
    try:
        fit.predict_rows(x, torch.zeros(1, dtype=torch.long))
    except ValueError as exc:
        assert "group shape" in str(exc)
    else:
        raise AssertionError("bad ridge group shape was accepted")
