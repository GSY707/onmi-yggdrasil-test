from __future__ import annotations

"""Streaming grouped ridge used by R2 and the exact routed-head R3 screen."""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


@dataclass
class GroupedSufficientStatistics:
    groups: int
    feature_width: int
    output_width: int
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.groups <= 0 or self.feature_width <= 0 or self.output_width <= 0:
            raise ValueError("grouped ridge dimensions must be positive")
        augmented = self.feature_width + 1
        self.xtx = torch.zeros(
            self.groups, augmented, augmented, dtype=torch.float64, device=self.device
        )
        self.xty = torch.zeros(
            self.groups, augmented, self.output_width, dtype=torch.float64, device=self.device
        )
        self.yty = torch.zeros(self.groups, dtype=torch.float64, device=self.device)
        self.rows = torch.zeros(self.groups, dtype=torch.int64, device=self.device)

    def update(self, group: int, features: Tensor, targets: Tensor) -> None:
        if not 0 <= int(group) < self.groups:
            raise ValueError("ridge group is out of range")
        if features.ndim != 2 or features.shape[-1] != self.feature_width:
            raise ValueError("ridge feature shape mismatch")
        if targets.ndim != 2 or targets.shape != (features.shape[0], self.output_width):
            raise ValueError("ridge target shape mismatch")
        if not features.shape[0]:
            return
        if features.device.type == "cpu" and not bool(
            torch.isfinite(features).all() and torch.isfinite(targets).all()
        ):
            raise ValueError("ridge update contains non-finite values")
        # The production row count is large but the sufficient matrices are
        # small.  Accumulate each batch in fp32 on the capture device, then add
        # only the small products in fp64.  This avoids both a dense Jacobian
        # and multi-gigabyte fp64 design matrices.
        x = features.float()
        y = targets.float()
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        augmented = torch.cat((x, ones), dim=1)
        local_xtx = augmented.transpose(0, 1) @ augmented
        local_xty = augmented.transpose(0, 1) @ y
        local_yty = y.square().sum()
        destination = self.xtx.device
        self.xtx[group].add_(local_xtx.to(device=destination, dtype=torch.float64))
        self.xty[group].add_(local_xty.to(device=destination, dtype=torch.float64))
        self.yty[group].add_(local_yty.to(device=destination, dtype=torch.float64))
        self.rows[group].add_(int(x.shape[0]))

    def merge(self, other: "GroupedSufficientStatistics") -> "GroupedSufficientStatistics":
        if (
            self.groups,
            self.feature_width,
            self.output_width,
        ) != (other.groups, other.feature_width, other.output_width):
            raise ValueError("cannot merge incompatible ridge statistics")
        result = GroupedSufficientStatistics(
            self.groups, self.feature_width, self.output_width, device=str(self.xtx.device)
        )
        result.xtx.copy_(self.xtx + other.xtx.to(self.xtx.device))
        result.xty.copy_(self.xty + other.xty.to(self.xty.device))
        result.yty.copy_(self.yty + other.yty.to(self.yty.device))
        result.rows.copy_(self.rows + other.rows.to(self.rows.device))
        return result

    def cpu(self) -> "GroupedSufficientStatistics":
        if self.xtx.device.type == "cpu":
            return self
        result = GroupedSufficientStatistics(
            self.groups, self.feature_width, self.output_width, device="cpu"
        )
        result.xtx.copy_(self.xtx.cpu())
        result.xty.copy_(self.xty.cpu())
        result.yty.copy_(self.yty.cpu())
        result.rows.copy_(self.rows.cpu())
        return result


@dataclass(frozen=True)
class GroupedRidgeFit:
    coefficients: Tensor
    ridge_lambda: float
    lambda_validation_nmse: Mapping[float, float]
    feature_width: int
    output_width: int
    group_keys: tuple[str, ...]
    train_rows: tuple[int, ...]

    def predict_rows(self, features: Tensor, groups: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != self.feature_width:
            raise ValueError("ridge prediction feature shape mismatch")
        if groups.ndim != 1 or groups.shape[0] != features.shape[0]:
            raise ValueError("ridge prediction group shape mismatch")
        coefficients = self.coefficients.to(device=features.device, dtype=torch.float32)
        x = torch.cat(
            (
                features.float(),
                torch.ones((features.shape[0], 1), device=features.device),
            ),
            dim=1,
        )
        output = torch.zeros(
            features.shape[0], self.output_width, dtype=torch.float32, device=features.device
        )
        for group in range(coefficients.shape[0]):
            selected = groups == group
            if bool(selected.any()):
                output[selected] = x[selected] @ coefficients[group]
        return output


def _solve(stats: GroupedSufficientStatistics, ridge_lambda: float) -> Tensor:
    stats = stats.cpu()
    if not bool(
        torch.isfinite(stats.xtx).all()
        and torch.isfinite(stats.xty).all()
        and torch.isfinite(stats.yty).all()
    ):
        raise ValueError("ridge sufficient statistic is non-finite")
    coefficients = torch.zeros_like(stats.xty)
    penalty = torch.eye(stats.feature_width + 1, dtype=torch.float64)
    penalty[-1, -1] = 0.0
    for group in range(stats.groups):
        if int(stats.rows[group]) == 0:
            continue
        feature_scale = float(
            torch.trace(stats.xtx[group, :-1, :-1])
            / max(stats.feature_width, 1)
        )
        gram = (
            stats.xtx[group]
            + float(ridge_lambda) * max(feature_scale, 1.0e-20) * penalty
        )
        rhs = stats.xty[group]
        try:
            coefficients[group] = torch.linalg.solve(gram, rhs)
        except torch.linalg.LinAlgError:
            coefficients[group] = torch.linalg.lstsq(gram, rhs).solution
    return coefficients


def sufficient_metric(
    stats: GroupedSufficientStatistics,
    coefficients: Tensor,
) -> dict[str, Any]:
    stats = stats.cpu()
    coefficients = coefficients.detach().cpu().double()
    if coefficients.shape != stats.xty.shape:
        raise ValueError("coefficient/statistic shape mismatch")
    target_energy = float(stats.yty.sum())
    prediction_energy = 0.0
    dot = 0.0
    by_group: dict[str, Any] = {}
    for group in range(stats.groups):
        weight = coefficients[group]
        group_prediction_energy = float(
            torch.sum(weight * (stats.xtx[group] @ weight))
        )
        group_dot = float(torch.sum(weight * stats.xty[group]))
        group_target = float(stats.yty[group])
        group_sse = max(0.0, group_target + group_prediction_energy - 2.0 * group_dot)
        prediction_energy += group_prediction_energy
        dot += group_dot
        by_group[str(group)] = {
            "rows": int(stats.rows[group]),
            "target_energy": group_target,
            "sse": group_sse,
            "increment_nmse": group_sse / max(group_target, 1.0e-20),
            "explained_energy_gain": 1.0 - group_sse / max(group_target, 1.0e-20),
        }
    sse = max(0.0, target_energy + prediction_energy - 2.0 * dot)
    cosine = dot / max((target_energy * prediction_energy) ** 0.5, 1.0e-20)
    return {
        "rows": int(stats.rows.sum()),
        "target_energy": target_energy,
        "prediction_energy": prediction_energy,
        "dot": dot,
        "sse": sse,
        "increment_nmse": sse / max(target_energy, 1.0e-20),
        "explained_energy_gain": 1.0 - sse / max(target_energy, 1.0e-20),
        "cosine": cosine,
        "by_group": by_group,
    }


def fit_grouped_ridge(
    fit_stats: GroupedSufficientStatistics,
    validation_stats: GroupedSufficientStatistics,
    *,
    full_stats: GroupedSufficientStatistics | None = None,
    lambda_grid: Iterable[float] = (1.0e-6, 1.0e-4, 1.0e-2, 1.0),
    group_keys: Sequence[str] | None = None,
) -> GroupedRidgeFit:
    candidates = tuple(sorted({float(value) for value in lambda_grid}))
    if not candidates or any(value < 0.0 for value in candidates):
        raise ValueError("ridge lambda grid must be non-empty and non-negative")
    if (
        fit_stats.groups,
        fit_stats.feature_width,
        fit_stats.output_width,
    ) != (
        validation_stats.groups,
        validation_stats.feature_width,
        validation_stats.output_width,
    ):
        raise ValueError("fit/validation ridge statistics mismatch")
    validation_scores: dict[float, float] = {}
    for value in candidates:
        coefficients = _solve(fit_stats, value)
        validation_scores[value] = float(
            sufficient_metric(validation_stats, coefficients)["increment_nmse"]
        )
    selected = min(candidates, key=lambda value: (validation_scores[value], value))
    all_stats = full_stats or fit_stats.merge(validation_stats)
    coefficients = _solve(all_stats, selected)
    keys = tuple(group_keys or (str(index) for index in range(fit_stats.groups)))
    if len(keys) != fit_stats.groups:
        raise ValueError("ridge group key count mismatch")
    return GroupedRidgeFit(
        coefficients=coefficients,
        ridge_lambda=selected,
        lambda_validation_nmse=validation_scores,
        feature_width=fit_stats.feature_width,
        output_width=fit_stats.output_width,
        group_keys=keys,
        train_rows=tuple(int(value) for value in all_stats.rows.cpu().tolist()),
    )


def gram_spectrum(stats: GroupedSufficientStatistics) -> dict[str, Any]:
    stats = stats.cpu()
    groups: dict[str, Any] = {}
    total_trace = 0.0
    total_square = 0.0
    total_rank = 0
    for group in range(stats.groups):
        # Exclude the affine bias from the feature Fisher spectrum.
        gram = stats.xtx[group, :-1, :-1]
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        trace = float(eigenvalues.sum())
        square = float(eigenvalues.square().sum())
        rank = int(torch.count_nonzero(eigenvalues > max(trace, 1.0) * 1.0e-10))
        effective_rank = trace * trace / max(square, 1.0e-30)
        total_trace += trace
        total_square += square
        total_rank += rank
        groups[str(group)] = {
            "rows": int(stats.rows[group]),
            "rank": rank,
            "effective_rank": effective_rank,
            "trace": trace,
            "lambda_max": float(eigenvalues.max()) if eigenvalues.numel() else 0.0,
            "lambda_min_positive": (
                float(eigenvalues[eigenvalues > 0].min())
                if bool((eigenvalues > 0).any())
                else 0.0
            ),
        }
    return {
        "groups": groups,
        "sum_group_ranks": total_rank,
        "fisher_trace": total_trace,
        "block_effective_rank": total_trace * total_trace / max(total_square, 1.0e-30),
    }


__all__ = [
    "GroupedRidgeFit",
    "GroupedSufficientStatistics",
    "fit_grouped_ridge",
    "gram_spectrum",
    "sufficient_metric",
]
