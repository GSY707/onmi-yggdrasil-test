from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from .a1_13_train import _write_json
from .a1_20b_cache import A120BCachedSplit, collate_a120b_items
from .a1_20b_train import encode_a120b_labels
from .a1_20c_supervision import A120CSupervisionSplit
from .a1_20c_train import (
    compute_a120c_loss,
    load_a120c_checkpoint,
)


DIAGNOSTIC_SCHEMA = "yggdrasil.v2-a1.20c.failure-diagnostic.v1"


def _parameter_group(name: str) -> str:
    if any(
        token in name
        for token in (
            "entity_reader",
            "entity_role_query",
            "entity_name_anchor",
            "entity_value_anchor",
            "entity_output_norm",
        )
    ):
        return "entity_path"
    if any(
        token in name
        for token in (
            "operation_reader",
            "operation_role_queries",
            "operation_anchors",
            "operation_output_norm",
        )
    ):
        return "operation_path"
    if any(
        token in name
        for token in (
            "query_reader",
            "query_role_query",
            "query_anchor",
            "query_output_norm",
        )
    ):
        return "query_path"
    if "pointer_query" in name or "pointer_entity" in name:
        return "pointer_heads"
    if "presence_head" in name:
        return "presence_heads"
    if "value_head" in name or "family_head" in name:
        return "mapping_heads"
    return "shared_source"


def _gradient_statistics(
    rows: Iterable[
        tuple[
            str,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
    ],
) -> dict[str, float]:
    dot = state_squared = local_squared = 0.0
    for _, state_gradient, local_gradient in rows:
        if state_gradient is None or local_gradient is None:
            continue
        state = state_gradient.detach().double()
        local = local_gradient.detach().double()
        dot += float((state * local).sum())
        state_squared += float(state.square().sum())
        local_squared += float(local.square().sum())
    state_norm = math.sqrt(state_squared)
    local_norm = math.sqrt(local_squared)
    return {
        "state_gradient_norm": state_norm,
        "local_gradient_norm": local_norm,
        "dot_product": dot,
        "cosine": dot
        / max(state_norm * local_norm, torch.finfo(torch.float64).tiny),
    }


def diagnose_a120c_overfit_failure(
    *,
    checkpoint_path: Path,
    results_path: Path,
    history_path: Path,
    cache_dir: Path,
    supervision_dir: Path,
    data_dir: Path,
    output_path: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.20C diagnostic requested unavailable CUDA")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_step = int(checkpoint["step"])
    checkpoint_history = next(
        row for row in history if int(row["step"]) == checkpoint_step
    )
    model = load_a120c_checkpoint(checkpoint_path, device)
    model.train()
    dataset = A120BCachedSplit(cache_dir, data_dir, "train")
    supervision = A120CSupervisionSplit(
        supervision_dir, dataset, "train"
    )
    indices = list(range(len(dataset)))
    inputs, records = collate_a120b_items(
        dataset.items(indices), device
    )
    labels = encode_a120b_labels(records, device)
    anchors = supervision.select(indices, device)
    output = model(**inputs)
    total_loss, components = compute_a120c_loss(
        model, output, labels, anchors
    )
    state_loss = components["state_ce"]
    local_loss = total_loss - state_loss
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    state_gradients = torch.autograd.grad(
        state_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    local_gradients = torch.autograd.grad(
        local_loss,
        parameters,
        allow_unused=True,
    )
    gradient_rows = [
        (name, state_gradient, local_gradient)
        for (name, _), state_gradient, local_gradient in zip(
            named_parameters, state_gradients, local_gradients
        )
    ]
    grouped_rows: dict[
        str,
        list[
            tuple[
                str,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ],
    ] = {}
    for row in gradient_rows:
        grouped_rows.setdefault(_parameter_group(row[0]), []).append(row)

    best_validation = results["best_validation_at_save"]
    failed_cells = {
        cell: {
            "mapping_accuracy": row["mapping_accuracy"],
            "trajectory_full_exact": row["trajectory_full_exact"],
            "final_state_full_exact": row["final_state_full_exact"],
            "state_token_accuracy": row["state_token_accuracy"],
            "final_answer_accuracy": row["final_answer_accuracy"],
        }
        for cell, row in best_validation["by_cell"].items()
        if row["mapping_accuracy"]["minimum"] < 1.0
        or row["trajectory_full_exact"] < 1.0
        or row["final_state_full_exact"] < 1.0
    }
    result = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "stage": "V2-A1.20C",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "step": checkpoint_step,
            "fresh_initialization": results["training"][
                "fresh_initialization"
            ],
        },
        "machine_classification": (
            "anchor_localization_solved_but_execution_objectives_conflict"
        ),
        "stage_gate_passed": bool(results["overfit_passed"]),
        "stop_rule": {
            "validation_matrix_not_run": True,
            "formal_and_causal_not_run": True,
            "a1_21p_not_run": True,
            "a1_22a_not_run": True,
        },
        "best_validation": {
            "aggregate": best_validation["aggregate"],
            "eligibility_components": results[
                "eligibility_components"
            ],
            "failed_cells": failed_cells,
        },
        "anchor_validation": results["anchor_validation"],
        "hard_forward_equivalence": results[
            "hard_forward_equivalence"
        ],
        "integrity": results["integrity"],
        "loss_at_best_checkpoint": {
            "total": float(total_loss.detach()),
            "state_ce": float(state_loss.detach()),
            "local_mapping_payload_and_anchor": float(
                local_loss.detach()
            ),
            "recorded_components": {
                name: checkpoint_history[name]
                for name in (
                    "entity_presence_bce",
                    "operation_presence_bce",
                    "value_ce",
                    "family_ce",
                    "source_ce",
                    "target_ce",
                    "query_ce",
                    "state_ce",
                    "payload_closure",
                    "entity_name_anchor_ce",
                    "entity_value_anchor_ce",
                    "operation_anchor_ce",
                    "query_anchor_ce",
                    "gradient_norm",
                )
            },
        },
        "gradient_conflict": {
            "definition": (
                "state_ce versus every other local mapping, payload, "
                "and anchor objective on the same overfit32 batch"
            ),
            "global": _gradient_statistics(gradient_rows),
            "by_parameter_group": {
                name: _gradient_statistics(rows)
                for name, rows in sorted(grouped_rows.items())
            },
        },
        "root_cause_evidence": {
            "anchor_sequence_exact": results["anchor_validation"][
                "all_anchor_sequence_exact"
            ]
            == 1.0,
            "answer_exact_but_internal_state_not_exact": (
                best_validation["aggregate"]["final_answer_accuracy"]
                == 1.0
                and best_validation["aggregate"][
                    "trajectory_full_exact"
                ]
                < 1.0
            ),
            "fixed_learning_rate_late_regression": {
                "best_step": results["best_checkpoint_step"],
                "best_total_loss": checkpoint_history["loss"],
                "last_step": history[-1]["step"],
                "last_total_loss": history[-1]["loss"],
            },
            "pointer_loss_precedes_predicted_entity_mask": True,
            "straight_through_pointer_validity_set_is_boolean": True,
        },
        "next_repair_boundary": {
            "do_not_advance_stage": True,
            "required": [
                (
                    "make entity-count and pointer-validity coupling "
                    "differentiable while preserving hard forward"
                ),
                (
                    "separate compiler fitting from execution fitting or "
                    "remove destructive gradient components"
                ),
                (
                    "use learning-rate decay and late high-frequency "
                    "checkpoint validation"
                ),
            ],
            "not_sufficient": [
                "more seeds under the unchanged objective",
                "accepting answer accuracy as the mechanism Gate",
                "running A1.21P or A1.22A",
            ],
        },
    }
    _write_json(output_path, result)
    return result
