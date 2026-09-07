from __future__ import annotations

"""Heldout bank behavior, causal, functional-K, and paired K8/K1 metrics."""

from collections import defaultdict
import math
import random
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from yggdrasil_v2.v2_a.closure_c1u import contract as c1u_contract
from yggdrasil_v2.v2_a.closure_c1u.evaluate import (
    bootstrap_mean_interval,
    wilson_interval,
)
from yggdrasil_v2.v2_a.closure_c1u.objective import answer_margin

from . import contract
from .model import S2BoundaryOutput, S2Model
from .objective import causal_outputs


def _forward_fields(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    return {field: batch[field] for field in c1u_contract.FORWARD_FIELDS}


def _swap_public_payloads(
    payloads: Tensor, left_slots: Tensor, right_slots: Tensor
) -> Tensor:
    batch, _, width = payloads.shape
    if left_slots.shape != (batch,) or right_slots.shape != (batch,):
        raise ValueError("owner swap slots have the wrong shape")
    if bool((left_slots == right_slots).any()):
        raise ValueError("owner swap requires distinct owners")
    gather = lambda slots: payloads.gather(
        1, slots.view(batch, 1, 1).expand(batch, 1, width)
    ).squeeze(1)
    left_values = gather(left_slots)
    right_values = gather(right_slots)
    changed = payloads.clone()
    changed = changed.scatter(
        1,
        left_slots.view(batch, 1, 1).expand(batch, 1, width),
        right_values.unsqueeze(1),
    )
    return changed.scatter(
        1,
        right_slots.view(batch, 1, 1).expand(batch, 1, width),
        left_values.unsqueeze(1),
    )


def _delete_public_payload(payloads: Tensor, slots: Tensor) -> Tensor:
    batch, _, width = payloads.shape
    return payloads.clone().scatter(
        1,
        slots.view(batch, 1, 1).expand(batch, 1, width),
        torch.zeros(batch, 1, width, dtype=payloads.dtype, device=payloads.device),
    )


@torch.inference_mode()
def collect_rows(
    model: S2Model,
    batch: Mapping[str, Any],
    *,
    fold_id: str,
    arm: str,
) -> dict[str, Any]:
    answers = batch["answers"].long()
    valid = batch["valid_choice_mask"].bool()
    support_slots = batch["support_slots"].long()
    counterparts = batch["counterfactual_indices"].long()
    if not bool((counterparts >= 0).all()):
        raise ValueError("heldout evaluation requires complete factorial groups")
    boundary = model.boundary(**_forward_fields(batch))
    outputs = causal_outputs(
        model, boundary, answers, valid, support_slots, counterparts
    )
    base_logits = outputs["logits"]
    base_margin = outputs["base_margin"]
    query_owner = boundary.public_query_weights.argmax(dim=1)
    if bool((support_slots == query_owner.unsqueeze(1)).any()):
        raise ValueError("registered supports must be distinct from query owner")

    deletion_logits = []
    swap_logits = []
    for support_index in range(2):
        support = support_slots[:, support_index]
        deleted = boundary.rebuild(
            _delete_public_payload(boundary.public_initial_payloads, support)
        )
        deletion_logits.append(model.forward_from_boundary(deleted)["logits"])
        swapped_payloads = _swap_public_payloads(
            boundary.public_initial_payloads, support, query_owner
        )
        swapped = boundary.rebuild(swapped_payloads)
        swap_logits.append(model.forward_from_boundary(swapped)["logits"])
    deletion = torch.stack(deletion_logits, dim=1)
    swapped = torch.stack(swap_logits, dim=1)
    deletion_margins = torch.stack(
        [answer_margin(deletion[:, index], answers, valid) for index in range(2)],
        dim=1,
    )
    swap_l2 = torch.linalg.vector_norm(
        swapped.float() - base_logits.float().unsqueeze(1), dim=2
    )
    swap_max_abs = (
        swapped.float() - base_logits.float().unsqueeze(1)
    ).abs().amax(dim=2)

    permutation = torch.arange(
        boundary.public_initial_payloads.shape[1] - 1,
        -1,
        -1,
        device=base_logits.device,
    )
    permuted = model.forward_from_boundary(boundary.permuted(permutation))["logits"]
    permutation_delta = (permuted.float() - base_logits.float()).abs().amax(dim=1)

    predictions = base_logits.argmax(dim=1)
    no_core_predictions = outputs["no_core_logits"].argmax(dim=1)
    support_predictions = outputs["support_logits"].argmax(dim=2)
    expected = torch.stack(
        [answers[counterparts[:, index]] for index in range(2)], dim=1
    )
    rows = []
    for index, example_id in enumerate(batch["example_ids"]):
        support_flip = [
            int(support_predictions[index, support].item())
            == int(expected[index, support].item())
            for support in range(2)
        ]
        rows.append(
            {
                "example_id": str(example_id),
                "fold_id": str(fold_id),
                "arm": str(arm),
                "bank_id": str(batch["bank_ids"][index]),
                "family": str(batch["families"][index]),
                "factorial_group_id": str(batch["factorial_group_ids"][index]),
                "factor_cell": list(batch["factor_cells"][index]),
                "answer_index": int(answers[index].item()),
                "prediction_index": int(predictions[index].item()),
                "full_correct": int(predictions[index].item())
                == int(answers[index].item()),
                "no_core_prediction_index": int(no_core_predictions[index].item()),
                "no_core_correct": int(no_core_predictions[index].item())
                == int(answers[index].item()),
                "base_margin": float(base_margin[index].float().cpu()),
                "no_core_margin_drop": float(
                    outputs["no_core_margin_drop"][index].float().cpu()
                ),
                "support_slots": [
                    int(value) for value in support_slots[index].detach().cpu().tolist()
                ],
                "query_owner_slot": int(query_owner[index].item()),
                "distinct_support_owners": len(
                    set(int(value) for value in support_slots[index].tolist())
                )
                == 2,
                "support_margin_drops": [
                    float(value)
                    for value in outputs["support_margin_drops"][index]
                    .float()
                    .cpu()
                    .tolist()
                ],
                "support_prediction_indices": [
                    int(value) for value in support_predictions[index].cpu().tolist()
                ],
                "support_expected_answer_indices": [
                    int(value) for value in expected[index].cpu().tolist()
                ],
                "support_flip_correct": support_flip,
                "two_contributor": all(support_flip),
                "owner_deletion_margin_drops": [
                    float(value)
                    for value in (base_margin[index] - deletion_margins[index])
                    .float()
                    .cpu()
                    .tolist()
                ],
                "owner_swap_logit_l2": [
                    float(value) for value in swap_l2[index].float().cpu().tolist()
                ],
                "owner_swap_logit_max_abs": [
                    float(value)
                    for value in swap_max_abs[index].float().cpu().tolist()
                ],
                "permutation_logit_max_abs": float(
                    permutation_delta[index].cpu()
                ),
                "workspace_slots": int(boundary.workspace_slots),
            }
        )
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.endpoint-evaluation.v1",
        "fold_id": fold_id,
        "arm": arm,
        "rows": rows,
        "records": len(rows),
        "workspace_slots": int(boundary.workspace_slots),
        "model_integrity": model.integrity_report(),
        "permutation_max_abs": max(
            (row["permutation_logit_max_abs"] for row in rows), default=math.inf
        ),
    }


def _binary_metric(values: Sequence[bool]) -> dict[str, Any]:
    successes = sum(bool(value) for value in values)
    return wilson_interval(successes, len(values))


def _mean_metric(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    return bootstrap_mean_interval(
        [float(value) for value in values],
        seed=int(seed),
        replicates=contract.BOOTSTRAP_REPLICATES,
    )


def _slices(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result = {"overall": list(rows)}
    for family in c1u_contract.FAMILIES:
        result[family] = [row for row in rows if row["family"] == family]
    return result


def answer_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slices = {
        name: _binary_metric([bool(row["full_correct"]) for row in selected])
        for name, selected in _slices(rows).items()
    }
    banks = {}
    for bank_id in contract.BANK_IDS:
        for family in c1u_contract.FAMILIES:
            selected = [
                row
                for row in rows
                if row["bank_id"] == bank_id and row["family"] == family
            ]
            banks[f"{bank_id}/{family}"] = _binary_metric(
                [bool(row["full_correct"]) for row in selected]
            )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["factorial_group_id"])].append(row)
    factorial = _binary_metric(
        [len(group) == 4 and all(row["full_correct"] for row in group) for group in grouped.values()]
    )
    return {"slices": slices, "banks": banks, "factorial_exact": factorial}


def causal_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = {}
    for family_index, family in enumerate(c1u_contract.FAMILIES):
        selected = [row for row in rows if row["family"] == family]
        supports = []
        deletions = []
        swaps = []
        for support_index in range(2):
            offset = family_index * 20 + support_index
            supports.append(
                {
                    "margin_drop": _mean_metric(
                        [row["support_margin_drops"][support_index] for row in selected],
                        seed=contract.BOOTSTRAP_SEED + offset,
                    ),
                    "flip": _binary_metric(
                        [row["support_flip_correct"][support_index] for row in selected]
                    ),
                }
            )
            deletions.append(
                _mean_metric(
                    [row["owner_deletion_margin_drops"][support_index] for row in selected],
                    seed=contract.BOOTSTRAP_SEED + 4 + offset,
                )
            )
            swaps.append(
                _mean_metric(
                    [row["owner_swap_logit_l2"][support_index] for row in selected],
                    seed=contract.BOOTSTRAP_SEED + 8 + offset,
                )
            )
        families[family] = {
            "records": len(selected),
            "no_core_accuracy": _binary_metric(
                [bool(row["no_core_correct"]) for row in selected]
            ),
            "no_core_margin_drop": _mean_metric(
                [row["no_core_margin_drop"] for row in selected],
                seed=contract.BOOTSTRAP_SEED + 12 + family_index,
            ),
            "supports": supports,
            "two_contributor": _binary_metric(
                [bool(row["two_contributor"]) for row in selected]
            ),
            "owner_deletion_margin_drops": deletions,
            "owner_swap_logit_l2": swaps,
            "distinct_support_owners": all(
                bool(row["distinct_support_owners"]) for row in selected
            ),
        }
    return {"families": families}


def _cluster_gain(
    k8_rows: Sequence[Mapping[str, Any]],
    k1_rows: Sequence[Mapping[str, Any]],
    *,
    family: str | None,
    seed: int,
) -> dict[str, Any]:
    k1 = {str(row["example_id"]): row for row in k1_rows}
    selected = [
        row for row in k8_rows if family is None or row["family"] == family
    ]
    if not selected or any(str(row["example_id"]) not in k1 for row in selected):
        raise ValueError("paired K8/K1 rows are incomplete")
    values_by_bank: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        other = k1[str(row["example_id"])]
        if row["bank_id"] != other["bank_id"] or row["family"] != other["family"]:
            raise ValueError("paired K8/K1 row metadata drifted")
        values_by_bank[str(row["bank_id"])].append(
            float(bool(row["full_correct"])) - float(bool(other["full_correct"]))
        )
    banks = sorted(values_by_bank)
    bank_points = {bank: sum(values_by_bank[bank]) / len(values_by_bank[bank]) for bank in banks}
    point = sum(sum(values) for values in values_by_bank.values()) / sum(
        len(values) for values in values_by_bank.values()
    )
    generator = random.Random(int(seed))
    draws = []
    for _ in range(contract.BOOTSTRAP_REPLICATES):
        sampled = [banks[generator.randrange(len(banks))] for _ in banks]
        values = [value for bank in sampled for value in values_by_bank[bank]]
        draws.append(sum(values) / len(values))
    draws.sort()
    lower = draws[max(0, math.floor(0.025 * len(draws)))]
    upper = draws[min(len(draws) - 1, math.ceil(0.975 * len(draws)) - 1)]
    return {
        "point": point,
        "lower95": lower,
        "upper95": upper,
        "records": len(selected),
        "clusters": len(banks),
        "cluster_unit": "bank_id",
        "replicates": contract.BOOTSTRAP_REPLICATES,
        "seed": int(seed),
        "bank_points": bank_points,
    }


def paired_gain_report(
    k8_rows: Sequence[Mapping[str, Any]], k1_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "overall": _cluster_gain(
            k8_rows,
            k1_rows,
            family=None,
            seed=contract.BOOTSTRAP_SEED,
        ),
        "CPS": _cluster_gain(
            k8_rows,
            k1_rows,
            family="CPS",
            seed=contract.BOOTSTRAP_SEED + 1,
        ),
        "ERE": _cluster_gain(
            k8_rows,
            k1_rows,
            family="ERE",
            seed=contract.BOOTSTRAP_SEED + 2,
        ),
    }


def qualify(
    k8_rows: Sequence[Mapping[str, Any]], k1_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = contract.TOTAL_RECORDS
    if len(k8_rows) != expected or len(k1_rows) != expected:
        raise ValueError("S2 qualification requires each heldout bank exactly once per arm")
    if len({str(row["example_id"]) for row in k8_rows}) != expected:
        raise ValueError("K8 heldout rows are duplicated")
    if {str(row["example_id"]) for row in k8_rows} != {
        str(row["example_id"]) for row in k1_rows
    }:
        raise ValueError("K1/K8 heldout identities are not paired")
    k8_answer = answer_report(k8_rows)
    k1_answer = answer_report(k1_rows)
    k8_causal = causal_report(k8_rows)
    k1_causal = causal_report(k1_rows)
    gain = paired_gain_report(k8_rows, k1_rows)

    absolute_checks = {
        f"{family}_answer_point": k8_answer["slices"][family]["point"]
        >= float(contract.GATES["answer_point_floor"])
        for family in c1u_contract.FAMILIES
    }
    absolute_checks.update(
        {
            f"{family}_answer_wilson": k8_answer["slices"][family]["wilson_lower"]
            >= float(contract.GATES["answer_wilson_lower_floor"])
            for family in c1u_contract.FAMILIES
        }
    )
    absolute_checks["factorial_exact"] = (
        k8_answer["factorial_exact"]["point"]
        >= float(contract.GATES["factorial_exact_point_floor"])
    )
    absolute_checks["each_bank_family_floor"] = all(
        value["point"] >= float(contract.GATES["bank_family_answer_point_floor"])
        for value in k8_answer["banks"].values()
    )

    gain_checks = {}
    for name, value in gain.items():
        gain_checks[f"{name}_point"] = value["point"] >= float(
            contract.GATES["paired_gain_point_floor"]
        )
        gain_checks[f"{name}_bootstrap_lower"] = value["lower95"] > float(
            contract.GATES["paired_gain_bootstrap_lower_floor"]
        )

    causal_checks = {}
    functional_checks = {
        "k8_workspace_slots": all(row["workspace_slots"] == 8 for row in k8_rows),
        "k1_workspace_slots": all(row["workspace_slots"] == 1 for row in k1_rows),
        "k1_owner_swap_null": max(
            value
            for row in k1_rows
            for value in row["owner_swap_logit_max_abs"]
        )
        <= float(contract.GATES["k1_owner_swap_logit_max_abs"]),
        "k8_distinct_support_owners": all(
            bool(row["distinct_support_owners"]) for row in k8_rows
        ),
        "k8_permutation": max(
            float(row["permutation_logit_max_abs"]) for row in k8_rows
        )
        <= float(contract.GATES["permutation_logit_max_abs"]),
        "k1_permutation": max(
            float(row["permutation_logit_max_abs"]) for row in k1_rows
        )
        <= float(contract.GATES["permutation_logit_max_abs"]),
    }
    for family, value in k8_causal["families"].items():
        causal_checks[f"{family}_no_core_accuracy"] = value["no_core_accuracy"]["point"] <= float(
            contract.GATES["no_core_accuracy_ceiling"]
        )
        causal_checks[f"{family}_no_core_margin"] = value["no_core_margin_drop"]["lower95"] >= float(
            contract.GATES["no_core_margin_drop_lower"]
        )
        causal_checks[f"{family}_two_contributor_point"] = value["two_contributor"]["point"] >= float(
            contract.GATES["two_contributor_point_floor"]
        )
        causal_checks[f"{family}_two_contributor_wilson"] = value["two_contributor"]["wilson_lower"] >= float(
            contract.GATES["two_contributor_wilson_lower_floor"]
        )
        for support_index, support in enumerate(value["supports"]):
            causal_checks[f"{family}_support_{support_index}_margin"] = support["margin_drop"]["lower95"] >= float(
                contract.GATES["support_margin_drop_lower"]
            )
            causal_checks[f"{family}_support_{support_index}_flip_point"] = support["flip"]["point"] >= float(
                contract.GATES["support_flip_point_floor"]
            )
            causal_checks[f"{family}_support_{support_index}_flip_wilson"] = support["flip"]["wilson_lower"] >= float(
                contract.GATES["support_flip_wilson_lower_floor"]
            )
            causal_checks[f"{family}_support_{support_index}_deletion"] = value["owner_deletion_margin_drops"][support_index]["lower95"] >= float(
                contract.GATES["owner_deletion_margin_drop_lower"]
            )
            functional_checks[f"{family}_support_{support_index}_owner_swap"] = value["owner_swap_logit_l2"][support_index]["lower95"] >= float(
                contract.GATES["k8_owner_swap_logit_l2_floor"]
            )

    sections = {
        "absolute_behavior": absolute_checks,
        "paired_gain": gain_checks,
        "causal_behavior": causal_checks,
        "functional_k": functional_checks,
    }
    passed = all(all(checks.values()) for checks in sections.values())
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.qualification.v1",
        "status": "PASS_C1U_S2_MULTIBANK_MATCHED_K1_K8"
        if passed
        else "FAIL_C1U_S2_QUALIFICATION",
        "passed": passed,
        "authorizes": contract.PASS_AUTHORIZATION if passed else "nothing",
        "sections": sections,
        "k8_answer": k8_answer,
        "k1_answer": k1_answer,
        "paired_gain": gain,
        "k8_causal": k8_causal,
        "k1_causal_diagnostic": k1_causal,
        "unrun_successors": ["S3 training", "single-seed formal"],
    }


__all__ = [
    "answer_report",
    "causal_report",
    "collect_rows",
    "paired_gain_report",
    "qualify",
]
