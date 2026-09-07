from __future__ import annotations

"""No-step audit of whether the S1 objective can enforce its failed Gates."""

from collections import defaultdict
from contextlib import nullcontext
import hashlib
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..closure_c1s.model import C1SModel
from ..closure_c1s_successor.objective import SuccessorDiagnosticHeads
from ..closure_c1s_successor.runtime import C1SSuccessorRuntime
from .metrics import finite_json
from .runtime import chunks, model_inputs, move_batch


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _answer_margin(logits: Tensor, answers: Tensor) -> Tensor:
    values = logits.float()
    correct = values.gather(1, answers[:, None]).squeeze(1)
    wrong = values.masked_fill(
        F.one_hot(answers, num_classes=values.shape[-1]).bool(),
        float("-inf"),
    )
    return correct - torch.logsumexp(wrong, dim=-1)


def _state_digest(modules: Sequence[nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_index, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{module_index}:{name}".encode("utf-8"))
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest().upper()


def _named_groups(
    model: C1SModel,
    heads: SuccessorDiagnosticHeads,
) -> tuple[list[nn.Parameter], list[str]]:
    parameters: list[nn.Parameter] = []
    groups: list[str] = []
    for name, parameter in model.named_parameters():
        parameters.append(parameter)
        if name.startswith("boundary."):
            groups.append("boundary")
        elif name.startswith("transition."):
            groups.append("transition")
        elif name.startswith("training_auxiliary."):
            groups.append("temporal_feature_head")
        else:
            groups.append("answer_readout")
    for _name, parameter in heads.named_parameters():
        parameters.append(parameter)
        groups.append("diagnostic_decoder")
    return parameters, groups


def _gradient_snapshot(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    groups: Sequence[str],
    *,
    retain_graph: bool,
) -> tuple[dict[str, float], list[Tensor]]:
    if not loss.requires_grad:
        return ({name: 0.0 for name in sorted(set(groups))} | {"total": 0.0}, [])
    gradients = torch.autograd.grad(
        loss,
        list(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    squared: dict[str, float] = defaultdict(float)
    shared_core: list[Tensor] = []
    for gradient, group in zip(gradients, groups):
        if gradient is None:
            continue
        value = gradient.detach().float()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("non-finite no-step gradient")
        squared[group] += float(value.square().sum().cpu())
        if group == "transition":
            shared_core.append(value.cpu().reshape(-1))
    result = {group: float(value**0.5) for group, value in sorted(squared.items())}
    for group in sorted(set(groups)):
        result.setdefault(group, 0.0)
    result["total"] = float(sum(squared.values()) ** 0.5)
    return result, shared_core


def _gradient_cosine(left: Sequence[Tensor], right: Sequence[Tensor]) -> float | None:
    if len(left) != len(right):
        raise ValueError("shared-core gradient layouts differ")
    if not left:
        return None
    dot = sum(float((a * b).sum()) for a, b in zip(left, right))
    left_sq = sum(float(a.square().sum()) for a in left)
    right_sq = sum(float(b.square().sum()) for b in right)
    if left_sq <= 0.0 or right_sq <= 0.0:
        return None
    return float(dot / (left_sq * right_sq) ** 0.5)


def _feature_prevalence(batch: Mapping[str, Any]) -> dict[str, Any]:
    values = batch["state_values"].detach().cpu().bool()
    masks = batch["state_feature_mask"].detach().cpu().bool()
    names = batch["state_feature_names"]
    families = [str(value).upper() for value in batch["families"]]
    step_mask = batch["state_step_mask"].detach().cpu().bool()
    supported = batch.get(
        "mechanism_supported",
        torch.ones(values.shape[0], dtype=torch.bool),
    ).detach().cpu().bool()
    if step_mask.shape != values.shape[:2] or supported.shape != (values.shape[0],):
        raise ValueError("state step/support mask shape mismatch")
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    total = 0
    for row, family in enumerate(families):
        if len(names[row]) != values.shape[-1]:
            raise ValueError("state feature-name width mismatch")
        for feature, name in enumerate(names[row]):
            keep = masks[row, ..., feature] & step_mask[row, :, None] & supported[row]
            target = values[row, ..., feature][keep]
            positives = int(target.sum())
            negatives = int(target.numel() - positives)
            counts[(family, str(name))][0] += positives
            counts[(family, str(name))][1] += negatives
            total += positives + negatives
    return {
        family: {
            name: {
                "positives": values_[0],
                "negatives": values_[1],
                "micro_loss_weight_share": (values_[0] + values_[1]) / max(total, 1),
                "positive_rate": values_[0] / max(values_[0] + values_[1], 1),
            }
            for (cell_family, name), values_ in sorted(counts.items())
            if cell_family == family
        }
        for family in sorted({key[0] for key in counts})
    }


def audit_objective_alignment(
    model: C1SModel,
    heads: SuccessorDiagnosticHeads,
    runtime: C1SSuccessorRuntime,
    example_ids: Sequence[str],
    *,
    device: str | torch.device,
    batch_size: int = 4,
    no_core_hinge_margin: float = 0.5,
) -> dict[str, Any]:
    """Measure endpoint loss credit with ``autograd.grad`` and no mutation."""

    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA objective audit requested but unavailable")
    # The endpoint/path measurements are eval-mode measurements.  This audit
    # is different: it intentionally uses train mode because C1S routing has
    # a straight-through backward path only in train mode, and we are auditing
    # the gradient that the sealed optimizer actually saw.  No optimizer or
    # parameter .grad field is used.
    model.to(active_device).train()
    heads.to(active_device).train()
    modules: list[nn.Module] = [model, heads]
    before = _state_digest(modules)
    if any(parameter.grad is not None for module in modules for parameter in module.parameters()):
        raise RuntimeError("parameter .grad must be empty before no-step audit")
    parameters, parameter_groups = _named_groups(model, heads)
    by_family: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {
            "answer_loss": [],
            "no_core_loss": [],
            "no_core_active": [],
            "state_loss": [],
            "answer_grad": [],
            "no_core_grad": [],
            "state_grad": [],
            "answer_no_core_core_cosine": [],
            "answer_state_core_cosine": [],
            "no_core_state_core_cosine": [],
        }
    )
    prevalence_batches: list[dict[str, Any]] = []
    for ids in chunks(example_ids, batch_size):
        cpu_batch = runtime.get_batch(
            ids,
            slots=model.config.slots,
            pin_memory=active_device.type == "cuda",
        )
        prevalence_batches.append(_feature_prevalence(cpu_batch))
        batch = move_batch(cpu_batch, active_device)
        with _autocast(active_device):
            boundary = model.boundary(**model_inputs(batch))
            full = model.forward_from_boundary(
                boundary,
                return_trajectory=True,
                return_auxiliary=True,
            )
            no_core = model.forward_from_boundary(boundary, disable_recurrence=True)
            answers = batch["answers"].long()
            supported = batch.get(
                "mechanism_supported",
                torch.ones_like(answers, dtype=torch.bool),
            ).bool()
            drops = _answer_margin(full["logits"], answers) - _answer_margin(
                no_core["logits"], answers
            )
            no_core_per_row = F.relu(float(no_core_hinge_margin) - drops)
            auxiliary = full.get("auxiliary")
            if not isinstance(auxiliary, Mapping):
                raise RuntimeError("endpoint lacks temporal auxiliary")
            state_logits = heads.state_decoder(auxiliary["state_features"])
            state_error = F.binary_cross_entropy_with_logits(
                state_logits.float(),
                batch["state_values"].float(),
                reduction="none",
            )
            state_mask = batch["state_feature_mask"].to(dtype=state_error.dtype)
            state_mask = state_mask * batch["state_step_mask"].bool().unsqueeze(-1).unsqueeze(-1).to(dtype=state_error.dtype)
            state_mask = state_mask * supported.view(-1, 1, 1, 1).to(dtype=state_error.dtype)
        families = [str(value).upper() for value in batch["families"]]
        batch_families = sorted(set(families))
        for family_index, family in enumerate(batch_families):
            keep = torch.tensor(
                [value == family for value in families],
                dtype=torch.bool,
                device=active_device,
            )
            cell = by_family[family]
            family_answer_loss = F.cross_entropy(full["logits"][keep].float(), answers[keep])
            family_supported = keep & supported
            family_no_core_loss = (
                (no_core_per_row * family_supported.to(no_core_per_row.dtype)).sum()
                / family_supported.sum().clamp_min(1)
            )
            family_error = state_error[keep]
            family_mask = state_mask[keep]
            family_state_loss = (
                (family_error * family_mask).sum()
                / family_mask.sum().clamp_min(1.0)
            )
            answer_grad, answer_core = _gradient_snapshot(
                family_answer_loss, parameters, parameter_groups, retain_graph=True
            )
            no_core_grad, no_core_core = _gradient_snapshot(
                family_no_core_loss, parameters, parameter_groups, retain_graph=True
            )
            state_grad, state_core = _gradient_snapshot(
                family_state_loss,
                parameters,
                parameter_groups,
                retain_graph=family_index != len(batch_families) - 1,
            )
            cell["answer_loss"].append(float(family_answer_loss.detach().cpu()))
            cell["no_core_loss"].append(float(family_no_core_loss.detach().cpu()))
            cell["no_core_active"].extend(
                no_core_per_row[keep & supported].gt(0).detach().cpu().tolist()
            )
            cell["state_loss"].append(float(family_state_loss.detach().cpu()))
            cell["answer_grad"].append(answer_grad)
            cell["no_core_grad"].append(no_core_grad)
            cell["state_grad"].append(state_grad)
            for key, value in (
                ("answer_no_core_core_cosine", _gradient_cosine(answer_core, no_core_core)),
                ("answer_state_core_cosine", _gradient_cosine(answer_core, state_core)),
                ("no_core_state_core_cosine", _gradient_cosine(no_core_core, state_core)),
            ):
                if value is not None:
                    cell[key].append(value)
        del full, no_core, boundary

    after = _state_digest(modules)
    grads_empty = all(
        parameter.grad is None for module in modules for parameter in module.parameters()
    )
    model.eval()
    heads.eval()
    if before != after or not grads_empty:
        raise RuntimeError("no-step gradient audit mutated endpoint state")

    def mean_grad(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
        keys = sorted({key for row in rows for key in row})
        return {key: float(sum(float(row.get(key, 0.0)) for row in rows) / len(rows)) for key in keys}

    family_reports: dict[str, Any] = {}
    loss_rows: list[dict[str, Any]] = []
    for family, cell in sorted(by_family.items()):
        answer_grad = mean_grad(cell["answer_grad"])
        no_core_grad = mean_grad(cell["no_core_grad"])
        state_grad = mean_grad(cell["state_grad"])
        active_fraction = sum(bool(value) for value in cell["no_core_active"]) / max(len(cell["no_core_active"]), 1)
        shared_core_cosines = {
            name: (
                float(sum(cell[name]) / len(cell[name]))
                if cell[name] else None
            )
            for name in (
                "answer_no_core_core_cosine",
                "answer_state_core_cosine",
                "no_core_state_core_cosine",
            )
        }
        family_reports[family] = {
            "answer": {
                "loss": float(sum(cell["answer_loss"]) / len(cell["answer_loss"])),
                "gradient_norms": answer_grad,
            },
            "no_core_hinge": {
                "loss": float(sum(cell["no_core_loss"]) / len(cell["no_core_loss"])),
                "active_fraction": float(active_fraction),
                "gradient_norms": no_core_grad,
                "direct_gate_equivalent": False,
                "reason": "margin_drop_hinge_does_not_enforce_no_core_accuracy",
            },
            "state_micro_bce": {
                "loss": float(sum(cell["state_loss"]) / len(cell["state_loss"])),
                "gradient_norms": state_grad,
                "direct_gate_equivalent": False,
                "reason": "micro_bce_does_not_enforce_each_feature_balanced_accuracy",
            },
            "functional_multi_address": {
                "direct_loss_present": False,
                "direct_gate_equivalent": False,
            },
            "shared_recurrent_core_gradient_cosines": shared_core_cosines,
        }
        loss_rows.extend(
            (
                {
                    "family": family,
                    "gate": "no_core_near_chance",
                    "present": True,
                    "active_fraction": float(active_fraction),
                    "gradient_norm": float(no_core_grad.get("total", 0.0)),
                    "metric_equivalent": False,
                },
                {
                    "family": family,
                    "gate": "functional_multi_address",
                    "present": False,
                    "active_fraction": 0.0,
                    "gradient_norm": 0.0,
                    "metric_equivalent": False,
                },
                {
                    "family": family,
                    "gate": "dynamic_state_each_feature",
                    "present": True,
                    "active_fraction": 1.0,
                    "gradient_norm": float(state_grad.get("total", 0.0)),
                    "metric_equivalent": False,
                },
            )
        )

    # Merge prevalence counts collected in caller-order batches.
    merged: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"positives": 0.0, "negatives": 0.0})
    )
    for report in prevalence_batches:
        for family, features in report.items():
            for name, values in features.items():
                merged[family][name]["positives"] += float(values["positives"])
                merged[family][name]["negatives"] += float(values["negatives"])
    total_cells = sum(
        values["positives"] + values["negatives"]
        for features in merged.values()
        for values in features.values()
    )
    prevalence = {
        family: {
            name: {
                "positives": int(values["positives"]),
                "negatives": int(values["negatives"]),
                "positive_rate": values["positives"] / max(values["positives"] + values["negatives"], 1.0),
                "micro_loss_weight_share": (values["positives"] + values["negatives"]) / max(total_cells, 1.0),
            }
            for name, values in sorted(features.items())
        }
        for family, features in sorted(merged.items())
    }
    return finite_json(
        {
            "families": family_reports,
            "loss_rows": loss_rows,
            "state_target_prevalence": prevalence,
            "static_mapping": {
                "no_core": "hinge_saturates_after_margin_drop_0.5_not_accuracy_gate",
                "functional_multi_address": "no_direct_loss",
                "dynamic_state": "global_masked_micro_bce_not_family_feature_balanced_accuracy",
            },
            "mutation_audit": {
                "parameter_state_sha256_before": before,
                "parameter_state_sha256_after": after,
                "state_unchanged": before == after,
                "parameter_grad_fields_empty": grads_empty,
                "optimizer_steps": 0,
                "model_writes": 0,
                "gradient_measurement_mode": "train_straight_through_no_step",
                "behavior_measurement_mode": "eval_no_grad_in_D002_and_D004",
            },
        }
    )


__all__ = ["audit_objective_alignment"]
