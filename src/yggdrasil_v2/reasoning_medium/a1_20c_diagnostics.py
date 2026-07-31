from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .a1_13_train import _write_json
from .a1_9_model import module_state_sha256
from .a1_20b_cache import A120BCachedSplit, collate_a120b_items
from .a1_20b_train import encode_a120b_labels, evaluate_a120b_matrix
from .a1_20c_supervision import A120CSupervisionSplit
from .a1_20c_train import (
    compute_a120c_loss,
    load_a120c_checkpoint,
)


DIAGNOSTIC_SCHEMA = "yggdrasil.v2-a1.20c.failure-diagnostic.v1"
TAIL_DIAGNOSTIC_SCHEMA = (
    "yggdrasil.v2-a1.20d.heldout-tail-diagnostic.v1"
)
HIDDEN_CAUSAL_SCHEMA = "yggdrasil.v2-a1.20d.hidden-causal-audit.v1"


class _HiddenInterventionBoundary(nn.Module):
    """Read-only wrapper that intervenes on cached Qwen hidden states."""

    def __init__(self, model: nn.Module, intervention: str) -> None:
        super().__init__()
        if intervention not in {"zero", "batch_roll", "token_reverse"}:
            raise ValueError(f"unknown hidden intervention {intervention}")
        self.model = model
        self.core = model.core
        self.intervention = intervention

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if self.intervention == "zero":
            hidden = torch.zeros_like(source_hidden)
            attention = source_attention_mask
        elif self.intervention == "batch_roll":
            if source_hidden.shape[0] == 1:
                hidden = torch.zeros_like(source_hidden)
                attention = source_attention_mask
            else:
                hidden = source_hidden.roll(1, dims=0)
                attention = source_attention_mask.roll(1, dims=0)
        else:
            lengths = source_attention_mask.sum(dim=-1)
            position = torch.arange(
                source_hidden.shape[1], device=source_hidden.device
            ).unsqueeze(0)
            reverse = (lengths.unsqueeze(-1) - 1 - position).clamp_min(0)
            reverse = torch.where(
                position < lengths.unsqueeze(-1), reverse, position
            )
            hidden = source_hidden.gather(
                1,
                reverse.unsqueeze(-1).expand_as(source_hidden),
            )
            attention = source_attention_mask
        return self.model(hidden, attention)


@torch.inference_mode()
def audit_a120d_hidden_causality(
    checkpoint_path: Path,
    cache_dir: Path,
    data_dir: Path,
    formal_path: Path,
    output_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Test whether the selected Boundary actually depends on its text hidden."""

    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.20D hidden causal audit requires passed formal")
    model = load_a120c_checkpoint(checkpoint_path, device)
    dataset = A120BCachedSplit(cache_dir, data_dir, "causal_core")
    normal = formal["splits"]["causal_core"]
    interventions = {
        name: evaluate_a120b_matrix(
            _HiddenInterventionBoundary(model, name),
            dataset,
            device,
            batch_size=batch_size,
        )
        for name in ("zero", "batch_roll", "token_reverse")
    }
    gates = {
        "normal_causal_trajectory_at_least_0_95": (
            normal["aggregate"]["trajectory_full_exact"] >= 0.95
        ),
        "normal_causal_answer_at_least_0_95": (
            normal["aggregate"]["final_answer_accuracy"] >= 0.95
        ),
        "zero_hidden_trajectory_at_most_0_20": (
            interventions["zero"]["aggregate"]["trajectory_full_exact"]
            <= 0.20
        ),
        "batch_roll_trajectory_at_most_0_20": (
            interventions["batch_roll"]["aggregate"][
                "trajectory_full_exact"
            ]
            <= 0.20
        ),
        "token_reverse_trajectory_at_most_0_20": (
            interventions["token_reverse"]["aggregate"][
                "trajectory_full_exact"
            ]
            <= 0.20
        ),
        "core_hash_unchanged": (
            module_state_sha256(model.core)
            == model.a120c_core_state_sha256
        ),
        "architecture_integrity": model.integrity_report()["passed"],
    }
    result = {
        "schema_version": HIDDEN_CAUSAL_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "formal_eval": str(formal_path),
        "scope": (
            "minimal hidden-dependence audit; paired text counterfactuals "
            "remain outside this relaxed single-seed run"
        ),
        "normal": normal,
        "interventions": interventions,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json(output_path, result)
    return result


def _wrong_categorical_rows(
    logits: torch.Tensor,
    targets: torch.Tensor,
    active: torch.Tensor,
) -> list[dict[str, Any]]:
    prediction = logits.argmax(dim=-1)
    rows: list[dict[str, Any]] = []
    for index in torch.nonzero(
        active & (prediction != targets), as_tuple=False
    ).flatten():
        row = int(index)
        expected = int(targets[row])
        observed = int(prediction[row])
        rows.append(
            {
                "index": row,
                "target": expected,
                "prediction": observed,
                "wrong_minus_target_logit": float(
                    logits[row, observed] - logits[row, expected]
                ),
            }
        )
    return rows


@torch.inference_mode()
def diagnose_a120d_heldout_tail(
    *,
    checkpoint_path: Path,
    cache_dir: Path,
    supervision_dir: Path,
    data_dir: Path,
    output_path: Path,
    split: str = "validation",
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, Any]:
    model = load_a120c_checkpoint(checkpoint_path, device)
    dataset = A120BCachedSplit(cache_dir, data_dir, split)
    supervision = A120CSupervisionSplit(
        supervision_dir, dataset, split
    )
    failures: list[dict[str, Any]] = []
    for start in range(0, len(dataset), batch_size):
        indices = list(
            range(start, min(start + batch_size, len(dataset)))
        )
        inputs, records = collate_a120b_items(
            dataset.items(indices), device
        )
        labels = encode_a120b_labels(records, device)
        anchors = supervision.select(indices, device)
        output = model(**inputs)
        entity_prediction = output["predicted_entity_mask"]
        operation_prediction = output["predicted_operation_mask"]
        family_prediction = output["predicted_family"]
        source_prediction = output["predicted_source_pointer"]
        target_prediction = output["predicted_target_pointer"]
        query_prediction = output["predicted_query_pointer"]
        entity_target = labels.entity_presence.bool()
        operation_target = labels.operation_presence.bool()
        state_active = labels.state_targets >= 0
        state_prediction = output["state_logits"].argmax(dim=-1)
        anchor_predictions = {
            "entity_name": output.get(
                "predicted_entity_name_anchor_positions",
                output["entity_name_anchor_logits"].argmax(dim=-1),
            ),
            "entity_value": output.get(
                "predicted_entity_value_anchor_positions",
                output["entity_value_anchor_logits"].argmax(dim=-1),
            ),
            "operation": output.get(
                "predicted_operation_anchor_positions",
                output["operation_anchor_logits"].argmax(dim=-1),
            ),
            "query": output["query_anchor_logits"].argmax(dim=-1),
        }
        for row, record in enumerate(records):
            active_operations = operation_target[row]
            fields: dict[str, Any] = {}
            if not torch.equal(
                entity_prediction[row], entity_target[row]
            ):
                fields["entity_presence"] = {
                    "target_count": int(entity_target[row].sum()),
                    "prediction_count": int(
                        entity_prediction[row].sum()
                    ),
                    "logits": [
                        float(value)
                        for value in output["entity_presence_logits"][
                            row
                        ]
                    ],
                    "pair_score_gains": [
                        float(value)
                        for value in output[
                            "entity_pair_score_gains"
                        ][row]
                    ],
                }
            if not torch.equal(
                operation_prediction[row], operation_target[row]
            ):
                fields["operation_presence"] = {
                    "target_count": int(operation_target[row].sum()),
                    "prediction_count": int(
                        operation_prediction[row].sum()
                    ),
                    "boundary_logits": [
                        float(value)
                        for value in output["operation_presence_logits"][
                            row,
                            max(0, int(operation_target[row].sum()) - 2) :
                            int(operation_target[row].sum()) + 2,
                        ]
                    ],
                    "triple_score_gains": [
                        float(value)
                        for value in output[
                            "operation_triple_score_gains"
                        ][row]
                    ],
                }
            for name, prediction, targets, logits_name in (
                (
                    "family",
                    family_prediction[row],
                    labels.family_targets[row],
                    "family_mapping_logits",
                ),
                (
                    "source",
                    source_prediction[row],
                    labels.source_targets[row],
                    "source_pointer_logits",
                ),
                (
                    "target",
                    target_prediction[row],
                    labels.target_targets[row],
                    "target_pointer_logits",
                ),
            ):
                if torch.any(
                    active_operations & (prediction != targets)
                ):
                    fields[name] = _wrong_categorical_rows(
                        output[logits_name][row],
                        targets,
                        active_operations,
                    )
            if query_prediction[row] != labels.query_targets[row]:
                fields["query"] = _wrong_categorical_rows(
                    output["query_pointer_logits"][row].unsqueeze(0),
                    labels.query_targets[row].unsqueeze(0),
                    torch.ones(1, dtype=torch.bool, device=device),
                )
            state_exact = bool(
                (
                    (state_prediction[row] == labels.state_targets[row])
                    | ~state_active[row]
                )
                .all()
                .item()
            )
            anchor_exact = {
                "raw_entity_name": bool(
                    (
                        (
                            output["entity_name_anchor_logits"][
                                row
                            ].argmax(dim=-1)
                            == anchors.entity_name_targets[row]
                        )
                        | (anchors.entity_name_targets[row] < 0)
                    )
                    .all()
                    .item()
                ),
                "entity_name": bool(
                    (
                        (
                            anchor_predictions["entity_name"][row]
                            == anchors.entity_name_targets[row]
                        )
                        | (anchors.entity_name_targets[row] < 0)
                    )
                    .all()
                    .item()
                ),
                "entity_value": bool(
                    (
                        (
                            anchor_predictions["entity_value"][row]
                            == anchors.entity_value_targets[row]
                        )
                        | (anchors.entity_value_targets[row] < 0)
                    )
                    .all()
                    .item()
                ),
                "operation": bool(
                    (
                        (
                            anchor_predictions["operation"][row]
                            == anchors.operation_role_targets[row]
                        )
                        | (anchors.operation_role_targets[row] < 0)
                    )
                    .all()
                    .item()
                ),
                "raw_first_operation": bool(
                    (
                        output["operation_anchor_logits"][
                            row, 0
                        ].argmax(dim=-1)
                        == anchors.operation_role_targets[row, 0]
                    )
                    .all()
                    .item()
                ),
                "query": bool(
                    (
                        anchor_predictions["query"][row]
                        == anchors.query_targets[row]
                    ).item()
                ),
            }
            if fields or not state_exact or not all(anchor_exact.values()):
                failures.append(
                    {
                        "dataset_index": indices[row],
                        "fingerprint": record["fingerprint"],
                        "example_id": record["example_id"],
                        "cell": (
                            f"N{record['entity_count']}"
                            f"-T{record['program_length']}"
                        ),
                        "fields": fields,
                        "state_exact": state_exact,
                        "anchor_exact": anchor_exact,
                    }
                )
    field_counts: dict[str, int] = {}
    for failure in failures:
        for name in failure["fields"]:
            field_counts[name] = field_counts.get(name, 0) + 1
    result = {
        "schema_version": TAIL_DIAGNOSTIC_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "split": split,
        "validation_examples": len(dataset),
        "failure_examples": len(failures),
        "field_failure_examples": field_counts,
        "failures": failures,
        "diagnostic_only": True,
    }
    _write_json(output_path, result)
    return result


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
