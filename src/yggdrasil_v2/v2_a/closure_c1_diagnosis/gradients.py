from __future__ import annotations

"""No-step gradient-credit diagnostics for the frozen selected C1 model."""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import hashlib
import json
import math
import random
from typing import Any

import torch
from torch.nn import functional as F

from . import contract


def _move(value: Any, device: torch.device) -> Any:
    return value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def _pair_metrics(
    answer: Sequence[torch.Tensor], trace: Sequence[torch.Tensor]
) -> dict[str, float | bool]:
    if len(answer) != len(trace) or not answer:
        raise ValueError("gradient vectors must be equal and non-empty")
    answer_sq = sum(float((value.detach().float() ** 2).sum().cpu()) for value in answer)
    trace_sq = sum(float((value.detach().float() ** 2).sum().cpu()) for value in trace)
    dot = sum(float((left.detach().float() * right.detach().float()).sum().cpu()) for left, right in zip(answer, trace))
    answer_norm = math.sqrt(answer_sq)
    trace_norm = math.sqrt(trace_sq)
    cosine = dot / (answer_norm * trace_norm) if answer_norm > 0.0 and trace_norm > 0.0 else 0.0
    result = {
        "answer_norm": answer_norm,
        "trace_norm": trace_norm,
        "trace_to_answer_ratio": trace_norm / answer_norm if answer_norm > 0.0 else math.inf,
        "late_weighted_trace_to_answer_ratio": 0.5 * trace_norm / answer_norm if answer_norm > 0.0 else math.inf,
        "dot": dot,
        "cosine": cosine,
        "conflict": cosine < 0.0,
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("non-finite gradient diagnostic")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    index = min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))
    return ordered[index]


def _summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | int]:
    values = [float(row[key]) for row in rows]
    rng = random.Random(contract.BOOTSTRAP_SEED)
    boot = []
    for _ in range(contract.BOOTSTRAP_REPLICATES):
        sampled = [values[rng.randrange(len(values))] for _ in values]
        boot.append(sum(sampled) / len(sampled))
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 0.50),
        "bootstrap_lower": _percentile(boot, 0.025),
        "bootstrap_upper": _percentile(boot, 0.975),
    }


def shared_gradient_report(
    model: torch.nn.Module,
    schedule: Sequence[Any],
    batch_provider: Any,
    *,
    device: str | torch.device = "cuda",
    batches: int = contract.GRADIENT_BATCHES,
) -> dict[str, Any]:
    """Compare answer/trace gradients without populating grads or taking a step."""
    if batches <= 0 or len(schedule) < batches:
        raise ValueError("gradient diagnostic schedule is too short")
    active_device = torch.device(device)
    model.to(active_device).eval()
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups = {
        "boundary": [(name, parameter) for name, parameter in named if name.startswith("boundary.")],
        "core": [(name, parameter) for name, parameter in named if name.startswith("core_layers.")],
    }
    if any(not rows for rows in groups.values()):
        raise ValueError("selected model lacks Boundary or shared-core parameters")
    shared = [parameter for group in groups.values() for _, parameter in group]
    boundaries = [0, len(groups["boundary"]), len(shared)]
    before = _state_digest(model)
    output_rows: list[dict[str, Any]] = []
    for scheduled in schedule[:batches]:
        epoch = int(getattr(scheduled, "epoch"))
        example_ids = list(getattr(scheduled, "example_ids"))
        update = int(getattr(scheduled, "update"))
        cpu_batch = batch_provider(
            example_ids,
            epoch=epoch,
            trace_chunk_tokens=64,
        )
        batch = {key: _move(value, active_device) for key, value in cpu_batch.items()}
        with _autocast(active_device):
            result = model(batch["source_hidden"], batch["source_mask"], return_trajectory=True)
            answer_loss = F.cross_entropy(result["logits"], batch["answers"].long())
            trace_logits = model.trace_logits(
                result["trajectory"],
                batch["trace_step_indices"].long(),
                batch["trace_global_positions"].long(),
                batch["trace_local_positions"].long(),
            )
            mask = batch["trace_mask"].bool()
            trace_loss = F.cross_entropy(trace_logits[mask], batch["trace_targets"].long()[mask])
        answer_grads = torch.autograd.grad(answer_loss, shared, retain_graph=True)
        trace_grads = torch.autograd.grad(trace_loss, shared)
        row: dict[str, Any] = {
            "update": update,
            "epoch": epoch,
            "answer_loss": float(answer_loss.detach().float().cpu()),
            "trace_loss": float(trace_loss.detach().float().cpu()),
        }
        row["shared"] = _pair_metrics(answer_grads, trace_grads)
        row["boundary"] = _pair_metrics(
            answer_grads[boundaries[0] : boundaries[1]],
            trace_grads[boundaries[0] : boundaries[1]],
        )
        row["core"] = _pair_metrics(
            answer_grads[boundaries[1] : boundaries[2]],
            trace_grads[boundaries[1] : boundaries[2]],
        )
        output_rows.append(row)
        del result, trace_logits, answer_grads, trace_grads, batch

    after = _state_digest(model)
    summaries: dict[str, Any] = {}
    for group in ("shared", "boundary", "core"):
        group_rows = [row[group] for row in output_rows]
        summaries[group] = {
            "cosine": _summary(group_rows, "cosine"),
            "trace_to_answer_ratio": _summary(group_rows, "trace_to_answer_ratio"),
            "late_weighted_trace_to_answer_ratio": _summary(
                group_rows, "late_weighted_trace_to_answer_ratio"
            ),
            "negative_cosine_fraction": sum(bool(row["conflict"]) for row in group_rows) / len(group_rows),
        }
    decision = {
        "late_trace_dominates": summaries["shared"]["late_weighted_trace_to_answer_ratio"]["median"]
        > float(contract.THRESHOLDS["gradient_late_trace_to_answer_ratio"]),
        "majority_conflict": summaries["shared"]["negative_cosine_fraction"]
        > float(contract.THRESHOLDS["gradient_negative_cosine_fraction"]),
    }
    report = {
        "batches": batches,
        "optimizer_steps": 0,
        "parameter_state_before": before,
        "parameter_state_after": after,
        "parameters_unchanged": before == after,
        "rows": output_rows,
        "summaries": summaries,
        "decision": decision,
    }
    json.dumps(report, allow_nan=False)
    return report


__all__ = ["shared_gradient_report"]
