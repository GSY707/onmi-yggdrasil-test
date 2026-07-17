from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .a1_20b_cache import A120BCachedSplit, collate_a120b_items
from .a1_20b_train import (
    MAXIMUM_ENTITIES,
    MAXIMUM_OPERATIONS,
    A120BLabels,
    compute_a120b_loss,
    encode_a120b_labels,
    load_a120b_checkpoint,
)


DIAGNOSTIC_SCHEMA = "yggdrasil.v2-a1.20b.boundary.diagnostic.v1"
ORACLE_FIELDS = frozenset(
    {
        "entity_presence",
        "operation_presence",
        "values",
        "family",
        "source",
        "target",
        "query",
    }
)
VARIANT_ORACLE_FIELDS: dict[str, frozenset[str]] = {
    "predicted_all": frozenset(),
    "oracle_all": ORACLE_FIELDS,
    "oracle_all_except_entity_presence": ORACLE_FIELDS - {"entity_presence"},
    "oracle_all_except_operation_presence": ORACLE_FIELDS
    - {"operation_presence"},
    "oracle_all_except_values": ORACLE_FIELDS - {"values"},
    "oracle_all_except_family": ORACLE_FIELDS - {"family"},
    "oracle_all_except_source": ORACLE_FIELDS - {"source"},
    "oracle_all_except_target": ORACLE_FIELDS - {"target"},
    "oracle_all_except_query": ORACLE_FIELDS - {"query"},
    "oracle_all_except_addresses": ORACLE_FIELDS - {"source", "target", "query"},
    "predicted_plus_oracle_values": frozenset({"values"}),
    "predicted_plus_oracle_program": frozenset(
        {"operation_presence", "family", "source", "target"}
    ),
    "predicted_plus_oracle_addresses": frozenset({"source", "target", "query"}),
    "predicted_plus_oracle_entity_binding": frozenset(
        {"values", "source", "target", "query"}
    ),
    "predicted_plus_oracle_controls": ORACLE_FIELDS - {"values"},
}


def select_a120b_variant_controls(
    output: dict[str, Any],
    labels: A120BLabels,
    oracle_fields: frozenset[str],
) -> dict[str, torch.Tensor | bool]:
    """Select predicted/oracle fields without changing the learned Boundary."""

    return {
        "entity_mask": labels.entity_presence.bool()
        if "entity_presence" in oracle_fields
        else output["predicted_entity_mask"],
        "operation_mask": labels.operation_presence.bool()
        if "operation_presence" in oracle_fields
        else output["predicted_operation_mask"],
        "family": labels.family_targets.clamp_min(0)
        if "family" in oracle_fields
        else output["predicted_family"],
        "source": labels.source_targets.clamp_min(0)
        if "source" in oracle_fields
        else output["predicted_source_pointer"],
        "target": labels.target_targets.clamp_min(0)
        if "target" in oracle_fields
        else output["predicted_target_pointer"],
        "query": labels.query_targets
        if "query" in oracle_fields
        else output["predicted_query_pointer"],
        "oracle_values": "values" in oracle_fields,
    }


def _run_variant(
    model,
    output: dict[str, Any],
    labels: A120BLabels,
    oracle_fields: frozenset[str],
) -> dict[str, torch.Tensor]:
    if not oracle_fields:
        return {
            "state_logits": output["state_logits"],
            "answer_logits": output["answer_logits"],
        }
    selected = select_a120b_variant_controls(output, labels, oracle_fields)
    batch = labels.query_targets.shape[0]
    handles = (
        torch.arange(
            MAXIMUM_ENTITIES,
            device=labels.query_targets.device,
            dtype=torch.long,
        )
        .mul(104729)
        .add(10007)
        .unsqueeze(0)
        .expand(batch, -1)
    )
    source_handle = handles.gather(1, selected["source"])
    target_handle = handles.gather(1, selected["target"])
    query_handle = handles.gather(1, selected["query"].unsqueeze(-1)).squeeze(-1)
    payloads = output["continuous_entity_payloads"]
    if selected["oracle_values"]:
        payloads = model.core.closure_targets(
            labels.value_targets.unsqueeze(1)
        ).squeeze(1)
    return model.core(
        entity_values=torch.zeros_like(handles),
        entity_handles=handles,
        entity_mask=selected["entity_mask"],
        query_handle=query_handle,
        operation_family=selected["family"],
        operation_source_handle=source_handle,
        operation_target_handle=target_handle,
        operation_mask=selected["operation_mask"],
        recurrent_steps=MAXIMUM_OPERATIONS,
        entity_payloads=payloads,
    )


def _task_counts(
    state_logits: torch.Tensor,
    answer_logits: torch.Tensor,
    labels: A120BLabels,
) -> dict[str, int]:
    state_prediction = state_logits.argmax(dim=-1)
    active = labels.state_targets != -100
    equal = (state_prediction == labels.state_targets) | ~active
    return {
        "examples": int(state_prediction.shape[0]),
        "trajectory": int(equal.all(dim=(-1, -2)).sum()),
        "final_state": int(equal[:, -1].all(dim=-1).sum()),
        "state_correct": int(
            ((state_prediction == labels.state_targets) & active).sum()
        ),
        "state_total": int(active.sum()),
        "answer": int(
            (answer_logits.argmax(dim=-1) == labels.answer_targets).sum()
        ),
    }


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for name, value in source.items():
        target[name] += int(value)


def _task_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    examples = counts["examples"]
    return {
        "examples": examples,
        "trajectory_full_exact": counts["trajectory"] / examples,
        "final_state_full_exact": counts["final_state"] / examples,
        "state_token_accuracy": counts["state_correct"] / counts["state_total"],
        "final_answer_accuracy": counts["answer"] / examples,
    }


def _length_band(program_length: int) -> str:
    lower = ((program_length - 1) // 4) * 4 + 1
    return f"T{lower}-{lower + 3}"


def _joint_mapping_flags(
    output: dict[str, Any], labels: A120BLabels
) -> dict[str, torch.Tensor]:
    entity_active = labels.entity_presence.bool()
    operation_active = labels.operation_presence.bool()
    entity_presence = (
        output["predicted_entity_mask"] == entity_active
    ).all(dim=-1)
    operation_presence = (
        output["predicted_operation_mask"] == operation_active
    ).all(dim=-1)
    values = (
        (output["value_mapping_logits"].argmax(dim=-1) == labels.value_targets)
        | ~entity_active
    ).all(dim=-1)
    family = (
        (output["predicted_family"] == labels.family_targets) | ~operation_active
    ).all(dim=-1)
    source = (
        (output["predicted_source_pointer"] == labels.source_targets)
        | ~operation_active
    ).all(dim=-1)
    target = (
        (output["predicted_target_pointer"] == labels.target_targets)
        | ~operation_active
    ).all(dim=-1)
    query = output["predicted_query_pointer"] == labels.query_targets
    program = operation_presence & family & source & target
    addresses = source & target & query
    return {
        "entity_presence_exact": entity_presence,
        "operation_presence_exact": operation_presence,
        "values_all_exact": values,
        "family_all_exact": family,
        "source_all_exact": source,
        "target_all_exact": target,
        "query_exact": query,
        "program_all_exact": program,
        "addresses_all_exact": addresses,
        "boundary_all_exact": entity_presence
        & operation_presence
        & values
        & family
        & source
        & target
        & query,
    }


def _update_joint_counts(
    counters: dict[str, dict[str, int]],
    flags: dict[str, torch.Tensor],
    records: Sequence[dict[str, Any]],
) -> None:
    for row, record in enumerate(records):
        groups = (
            "aggregate",
            f"N{int(record['entity_count'])}",
            _length_band(int(record["program_length"])),
        )
        for group in groups:
            counters[group]["examples"] += 1
            for name, values in flags.items():
                counters[group][name] += int(values[row])


def _joint_metrics(counters: dict[str, int]) -> dict[str, float | int]:
    examples = counters["examples"]
    return {
        "examples": examples,
        **{
            name: value / examples
            for name, value in counters.items()
            if name != "examples"
        },
    }


GRADIENT_AUDIT_OUTPUTS = (
    "entity_presence_logits",
    "operation_presence_logits",
    "value_mapping_logits",
    "family_mapping_logits",
    "source_pointer_logits",
    "target_pointer_logits",
    "query_pointer_logits",
    "continuous_entity_payloads",
)


def _retain_a120b_output_gradients(output: dict[str, Any]) -> None:
    for name in GRADIENT_AUDIT_OUTPUTS:
        output[name].retain_grad()


def _read_a120b_output_gradients(output: dict[str, Any]) -> dict[str, float]:
    return {
        name: 0.0
        if output[name].grad is None
        else float(output[name].grad.float().norm())
        for name in GRADIENT_AUDIT_OUTPUTS
    }


def audit_a120b_gradient_credit(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    *,
    split: str = "train",
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Measure which Boundary outputs receive task-state versus local-label credit."""

    model = load_a120b_checkpoint(checkpoint, device)
    dataset = A120BCachedSplit(cache_dir, data_dir, split)
    inputs, records = collate_a120b_items(
        dataset.items(range(min(batch_size, len(dataset)))), device
    )
    labels = encode_a120b_labels(records, device)

    model.train()
    model.zero_grad(set_to_none=True)
    state_output = model(**inputs)
    _retain_a120b_output_gradients(state_output)
    state_ce = torch.nn.functional.cross_entropy(
        state_output["state_logits"].reshape(
            -1, model.core.config.value_classes
        ),
        labels.state_targets.reshape(-1),
        ignore_index=-100,
    )
    state_ce.backward()
    state_gradients = _read_a120b_output_gradients(state_output)

    model.zero_grad(set_to_none=True)
    full_output = model(**inputs)
    _retain_a120b_output_gradients(full_output)
    full_loss, components = compute_a120b_loss(model, full_output, labels)
    full_loss.backward()
    full_gradients = _read_a120b_output_gradients(full_output)
    discrete = (
        "entity_presence_logits",
        "operation_presence_logits",
        "family_mapping_logits",
        "source_pointer_logits",
        "target_pointer_logits",
        "query_pointer_logits",
    )
    return {
        "schema_version": "yggdrasil.v2-a1.20b.gradient-credit-audit.v1",
        "stage": "V2-A1.20B gradient credit audit",
        "diagnostic_only": True,
        "checkpoint": str(checkpoint),
        "split": split,
        "batch_size": len(records),
        "state_ce": float(state_ce.detach()),
        "full_loss": float(full_loss.detach()),
        "full_loss_components": {
            name: float(value.detach()) for name, value in components.items()
        },
        "output_gradient_norms": {
            "state_ce_only": state_gradients,
            "full_factorized_loss": full_gradients,
        },
        "gates": {
            "state_ce_reaches_continuous_values": state_gradients[
                "value_mapping_logits"
            ]
            > 0.0,
            "state_ce_reaches_no_discrete_control_logits": all(
                state_gradients[name] == 0.0 for name in discrete
            ),
            "full_loss_reaches_all_locally_supervised_logits": all(
                full_gradients[name] > 0.0
                for name in (*discrete, "value_mapping_logits")
            ),
            "frozen_core_received_no_gradients": not any(
                parameter.grad is not None for parameter in model.core.parameters()
            ),
        },
        "interpretation": "hard threshold/argmax controls sever execution-conditioned state credit; discrete controls are trained by factorized local labels instead",
    }


@torch.inference_mode()
def diagnose_a120b_boundary(
    checkpoint: Path,
    cache_dir: Path,
    data_dir: Path,
    *,
    split: str = "validation",
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run diagnostic-only oracle substitutions after an A1.20B failure."""

    model = load_a120b_checkpoint(checkpoint, device)
    dataset = A120BCachedSplit(cache_dir, data_dir, split)
    variant_counts: dict[str, dict[str, int]] = {
        name: defaultdict(int) for name in VARIANT_ORACLE_FIELDS
    }
    joint_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for start in range(0, len(dataset), batch_size):
        inputs, records = collate_a120b_items(
            dataset.items(range(start, min(start + batch_size, len(dataset)))),
            device,
        )
        labels = encode_a120b_labels(records, device)
        output = model(**inputs)
        _update_joint_counts(joint_counts, _joint_mapping_flags(output, labels), records)
        for name, oracle_fields in VARIANT_ORACLE_FIELDS.items():
            variant_output = _run_variant(model, output, labels, oracle_fields)
            _add_counts(
                variant_counts[name],
                _task_counts(
                    variant_output["state_logits"],
                    variant_output["answer_logits"],
                    labels,
                ),
            )
    variants = {
        name: {
            "oracle_fields": sorted(VARIANT_ORACLE_FIELDS[name]),
            **_task_metrics(counts),
        }
        for name, counts in variant_counts.items()
    }
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "stage": "V2-A1.20B learned full-text boundary failure diagnosis",
        "diagnostic_only": True,
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "split": split,
        "examples": len(dataset),
        "variants": variants,
        "joint_mapping_exactness": {
            group: _joint_metrics(counts)
            for group, counts in sorted(joint_counts.items())
        },
        "interpretation_boundary": {
            "oracle_substitutions_are_formal_evidence": False,
            "oracle_substitutions_change_the_deployment_interface": True,
            "purpose": "attribute failure inside the learned Boundary only",
        },
        "integrity": model.integrity_report(),
    }
