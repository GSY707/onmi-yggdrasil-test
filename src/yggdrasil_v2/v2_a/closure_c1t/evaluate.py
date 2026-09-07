from __future__ import annotations

"""Gauge-invariant C1T mechanism metrics and fail-closed S1 gates."""

import math
from typing import Any, Mapping, Sequence

import torch
from . import contract
from .model import C1TModel
from .objective import causal_outputs


def wilson_interval(successes: int, n: int, *, z: float = 1.959963984540054) -> dict[str, Any]:
    if type(successes) is not int or type(n) is not int or n < 0 or successes < 0 or successes > n:
        raise ValueError("invalid binomial counts")
    if n == 0:
        return {
            "successes": 0,
            "n": 0,
            "point": 0.0,
            "wilson_lower": 0.0,
            "wilson_upper": 1.0,
        }
    point = successes / n
    denominator = 1.0 + z * z / n
    centre = (point + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(point * (1.0 - point) / n + z * z / (4.0 * n * n)) / denominator
    return {
        "successes": successes,
        "n": n,
        "point": point,
        "wilson_lower": max(0.0, centre - radius),
        "wilson_upper": min(1.0, centre + radius),
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int = contract.BOOTSTRAP_SEED,
    replicates: int = contract.BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    if not values or replicates < 100:
        raise ValueError("bootstrap requires values and at least 100 replicates")
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("bootstrap values must be finite")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0, tensor.numel(), (int(replicates), tensor.numel()), generator=generator
    )
    means = tensor[indices].mean(dim=1)
    return {
        "n": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "lower95": float(torch.quantile(means, 0.025)),
        "upper95": float(torch.quantile(means, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def _metadata(batch: Mapping[str, Any], name: str, n: int) -> list[Any]:
    value = batch.get(name)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != n:
        raise ValueError(f"batch metadata {name} must have {n} entries")
    return list(value)


def collect_prediction_rows(model: C1TModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in contract.FORWARD_FIELDS if field not in batch]
    missing.extend(
        name
        for name in (
            "answers",
            "valid_choice_mask",
            "support_slots",
            "counterfactual_indices",
        )
        if name not in batch
    )
    if missing:
        raise KeyError(f"evaluation batch is missing fields: {sorted(set(missing))}")
    forward = {field: batch[field] for field in contract.FORWARD_FIELDS}
    answers = batch["answers"].long()
    valid = batch["valid_choice_mask"].bool()
    support_slots = batch["support_slots"].long()
    n = int(answers.numel())
    example_ids = _metadata(batch, "example_ids", n)
    families = _metadata(batch, "families", n)
    groups = _metadata(batch, "factorial_group_ids", n)
    cells = _metadata(batch, "factor_cells", n)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            boundary = model.boundary(**forward)
            outputs = causal_outputs(
                model,
                boundary,
                answers,
                valid,
                support_slots,
                batch["counterfactual_indices"].long(),
            )
            # Primary behavior is raw A-I argmax.  The external valid-choice
            # ledger may audit the target, but must never correct predictions.
            full_prediction = outputs["logits"].argmax(dim=1)
            no_core_prediction = outputs["no_core_logits"].argmax(dim=1)
            support_prediction = outputs["support_logits"].argmax(dim=2)
            counterfactual_indices = batch["counterfactual_indices"].long()
            expected_support_answers = answers[counterfactual_indices]
            permutation = torch.arange(
                model.config.slots - 1, -1, -1, device=answers.device
            )
            permuted = model.forward_from_boundary(
                boundary.permuted(permutation), return_trajectory=True
            )
            logit_delta = float((permuted["logits"] - outputs["logits"]).abs().max().cpu())
            trajectory_delta = float(
                (
                    permuted["trajectory"]
                    - outputs["trajectory"][:, :, permutation]
                )
                .abs()
                .max()
                .cpu()
            )
    finally:
        model.train(was_training)

    rows = []
    for index in range(n):
        rows.append(
            {
                "example_id": str(example_ids[index]),
                "family": str(families[index]).upper(),
                "factorial_group_id": str(groups[index]),
                "factor_cell": [int(item) for item in cells[index]],
                "answer_index": int(answers[index].cpu()),
                "prediction_index": int(full_prediction[index].cpu()),
                "no_core_prediction_index": int(no_core_prediction[index].cpu()),
                "full_correct": bool(full_prediction[index] == answers[index]),
                "no_core_correct": bool(no_core_prediction[index] == answers[index]),
                "no_core_logits": [
                    float(value)
                    for value in outputs["no_core_logits"][index].detach().cpu().tolist()
                ],
                "base_margin": float(outputs["base_margin"][index].cpu()),
                "no_core_margin": float(outputs["no_core_margin"][index].cpu()),
                "no_core_margin_drop": float(outputs["no_core_margin_drop"][index].cpu()),
                "support_margin_drops": [
                    float(value)
                    for value in outputs["support_margin_drops"][index].detach().cpu().tolist()
                ],
                "support_prediction_indices": [
                    int(value)
                    for value in support_prediction[index].detach().cpu().tolist()
                ],
                "support_expected_answer_indices": [
                    int(value)
                    for value in expected_support_answers[index].detach().cpu().tolist()
                ],
                "support_flip_correct": [
                    bool(value)
                    for value in (
                        support_prediction[index] == expected_support_answers[index]
                    ).detach().cpu().tolist()
                ],
            }
        )
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.prediction-rows.v1",
        "rows": rows,
        "invariance": {
            "permutation": permutation.detach().cpu().tolist(),
            "logit_max_abs_delta": logit_delta,
            "trajectory_max_abs_delta": trajectory_delta,
            "tolerance": float(contract.S1_GATES["permutation_logit_tolerance"]),
            "passed": logit_delta
            <= float(contract.S1_GATES["permutation_logit_tolerance"])
            and trajectory_delta <= float(contract.S1_GATES["permutation_logit_tolerance"]),
        },
    }


def _family_metrics(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("family", "")).upper() == family]
    full = wilson_interval(sum(bool(row["full_correct"]) for row in selected), len(selected))
    no_core = wilson_interval(sum(bool(row["no_core_correct"]) for row in selected), len(selected))
    no_core_drop = bootstrap_mean_interval(
        [float(row["no_core_margin_drop"]) for row in selected],
        seed=contract.BOOTSTRAP_SEED + contract.FAMILIES.index(family) * 101,
    )
    support = []
    support_flip = []
    for support_index in range(2):
        support.append(
            bootstrap_mean_interval(
                [float(row["support_margin_drops"][support_index]) for row in selected],
                seed=contract.BOOTSTRAP_SEED
                + contract.FAMILIES.index(family) * 101
                + support_index
                + 1,
            )
        )
        support_flip.append(
            wilson_interval(
                sum(bool(row["support_flip_correct"][support_index]) for row in selected),
                len(selected),
            )
        )
    contributor_floor = float(contract.S1_GATES["support_margin_drop_floor"])
    two_contributor = wilson_interval(
        sum(
            len(row["support_margin_drops"]) == 2
            and all(float(value) >= contributor_floor for value in row["support_margin_drops"])
            for row in selected
        ),
        len(selected),
    )
    return {
        "records": len(selected),
        "full_answer": full,
        "no_core_answer": no_core,
        "no_core_margin_drop": no_core_drop,
        "support_margin_drops": support,
        "support_flip_answer": support_flip,
        "two_contributor": two_contributor,
    }


def _factorial_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["factorial_group_id"]), []).append(row)
    reports = {}
    for group, group_rows in sorted(groups.items()):
        cells = {tuple(row["factor_cell"]) for row in group_rows}
        no_core = torch.tensor(
            [list(row["no_core_logits"]) for row in group_rows], dtype=torch.float64
        )
        no_core_delta = float((no_core - no_core[:1]).abs().max())
        reports[group] = {
            "records": len(group_rows),
            "cells": [list(cell) for cell in sorted(cells)],
            "complete": len(group_rows) == 4 and cells == set(contract.FACTORIAL_CELLS),
            "all_correct": len(group_rows) == 4
            and cells == set(contract.FACTORIAL_CELLS)
            and all(bool(row["full_correct"]) for row in group_rows),
            "no_core_logit_max_abs_delta": no_core_delta,
            "no_core_invariant": no_core_delta
            <= float(contract.S1_GATES["no_core_group_logit_tolerance"]),
        }
    exact = sum(bool(value["all_correct"]) for value in reports.values())
    return {
        "groups": len(reports),
        "exact_groups": exact,
        "point": exact / len(reports) if reports else 0.0,
        "rows": reports,
    }


def qualify_s1(
    prediction: Mapping[str, Any],
    *,
    require_frozen_cardinality: bool = True,
) -> dict[str, Any]:
    rows = prediction.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("prediction rows are required")
    rows = list(rows)
    factorial = _factorial_metrics(rows)
    families = {family: _family_metrics(rows, family) for family in contract.FAMILIES}
    full_exact = sum(bool(row["full_correct"]) for row in rows) / len(rows)
    checks: dict[str, bool] = {
        "record_cardinality": (not require_frozen_cardinality)
        or len(rows) == contract.S1_RECORDS,
        "family_cardinality": (not require_frozen_cardinality)
        or all(
            metrics["records"] == contract.S1_RECORDS_PER_FAMILY
            for metrics in families.values()
        ),
        "full_answer_exact": full_exact >= float(contract.S1_GATES["answer_exact"]),
        "factorial_group_exact": factorial["point"]
        >= float(contract.S1_GATES["factorial_group_exact"]),
        "factorial_no_core_invariance": bool(factorial["rows"])
        and all(value["no_core_invariant"] for value in factorial["rows"].values()),
        "permutation_invariance": prediction.get("invariance", {}).get("passed") is True,
    }
    for family, metrics in families.items():
        prefix = family.casefold()
        checks[f"{prefix}_no_core_accuracy"] = (
            metrics["no_core_answer"]["point"]
            <= float(contract.S1_GATES["no_core_accuracy_ceiling"][family])
        )
        checks[f"{prefix}_no_core_margin_drop"] = (
            metrics["no_core_margin_drop"]["lower95"]
            >= float(contract.S1_GATES["no_core_margin_drop_lower"])
        )
        checks[f"{prefix}_two_contributor_point"] = (
            metrics["two_contributor"]["point"]
            >= float(contract.S1_GATES["two_contributor_point_floor"])
        )
        checks[f"{prefix}_two_contributor_wilson"] = (
            metrics["two_contributor"]["wilson_lower"]
            >= float(contract.S1_GATES["two_contributor_wilson_lower_floor"])
        )
        for support_index, support in enumerate(metrics["support_margin_drops"]):
            checks[f"{prefix}_support_{support_index}_margin_drop"] = (
                support["lower95"]
                >= float(contract.S1_GATES["support_margin_drop_floor"])
            )
            checks[f"{prefix}_support_{support_index}_flip_answer"] = (
                metrics["support_flip_answer"][support_index]["point"]
                >= float(contract.S1_GATES["support_flip_answer_exact"])
            )
    passed = all(checks.values())
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.s1-qualification.v1",
        "status": "PASS_C1T_S1" if passed else "FAIL_C1T_S1_QUALIFICATION",
        "passed": passed,
        "authorizes": "S2_CONTRACT_DESIGN_ONLY" if passed else "nothing",
        "checks": checks,
        "full_answer_exact": full_exact,
        "factorial": factorial,
        "families": families,
        "invariance": prediction.get("invariance"),
        "unrun_successors": [] if passed else ["S2", "S3", "single-seed formal"],
    }


def evaluate_s1(model: C1TModel, batch: Mapping[str, Any]) -> dict[str, Any]:
    prediction = collect_prediction_rows(model, batch)
    qualification = qualify_s1(prediction)
    return {"prediction": prediction, "qualification": qualification}


__all__ = [
    "bootstrap_mean_interval",
    "collect_prediction_rows",
    "evaluate_s1",
    "qualify_s1",
    "wilson_interval",
]
