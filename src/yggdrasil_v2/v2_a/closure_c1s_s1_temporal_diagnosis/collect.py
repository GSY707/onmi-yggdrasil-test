from __future__ import annotations

"""Minimal natural-rollout collector for D004R.

Unlike the predecessor path collector, this module does not execute answer
counterfactuals.  It collects only the source-only natural trajectory and the
evaluator-side temporal labels required by the new diagnosis.
"""

from contextlib import nullcontext
import hashlib
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..closure_c1s.model import C1SModel
from ..closure_c1s_successor.objective import SuccessorDiagnosticHeads
from ..closure_c1s_successor.runtime import C1SSuccessorRuntime
from ..closure_c1s_s1_diagnosis.runtime import chunks, model_inputs, move_batch


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _cpu(value: Tensor) -> Tensor:
    result = value.detach().float().cpu()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite temporal diagnostic tensor")
    return result


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


def collect_temporal(
    model: C1SModel,
    heads: SuccessorDiagnosticHeads,
    runtime: C1SSuccessorRuntime,
    example_ids: Sequence[str],
    *,
    device: str | torch.device,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Collect D004R tensors without parameter mutation or answer inputs."""

    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA temporal diagnosis requested but unavailable")
    model.to(active_device).eval()
    heads.to(active_device).eval()
    modules = (model, heads)
    before = _state_digest(modules)

    initial_payloads: list[Tensor] = []
    trajectories: list[Tensor] = []
    state_features: list[Tensor] = []
    state_logits: list[Tensor] = []
    state_values: list[Tensor] = []
    state_feature_mask: list[Tensor] = []
    state_step_mask: list[Tensor] = []
    operation_active_target: list[Tensor] = []
    families: list[str] = []
    feature_names: list[list[str]] = []
    ordered_ids: list[str] = []

    with torch.no_grad():
        for ids in chunks(example_ids, batch_size):
            cpu_batch = runtime.get_batch(
                ids,
                slots=model.config.slots,
                pin_memory=active_device.type == "cuda",
            )
            batch = move_batch(cpu_batch, active_device)
            with _autocast(active_device):
                boundary = model.boundary(**model_inputs(batch))
                full = model.forward_from_boundary(
                    boundary,
                    return_trajectory=True,
                    return_auxiliary=True,
                )
                auxiliary = full.get("auxiliary")
                if not isinstance(auxiliary, Mapping) or "state_features" not in auxiliary:
                    raise RuntimeError("endpoint has no source-only temporal state features")
                temporal = auxiliary["state_features"]
                decoded = heads.state_decoder(temporal)
            initial_payloads.append(_cpu(boundary.payloads))
            trajectories.append(_cpu(full["trajectory"]))
            state_features.append(_cpu(temporal))
            state_logits.append(_cpu(decoded))
            state_values.append(batch["state_values"].detach().cpu().bool())
            state_feature_mask.append(batch["state_feature_mask"].detach().cpu().bool())
            state_step_mask.append(batch["state_step_mask"].detach().cpu().bool())
            operation_active_target.append(batch["operation_active"].detach().cpu().bool())
            families.extend(str(value).upper() for value in batch["families"])
            feature_names.extend([list(value) for value in batch["state_feature_names"]])
            ordered_ids.extend(ids)

    expected_ids = [str(value) for value in example_ids]
    if ordered_ids != expected_ids:
        raise RuntimeError("runtime changed caller-provided record order")
    after = _state_digest(modules)
    gradients_empty = all(
        parameter.grad is None
        for module in modules
        for parameter in module.parameters()
    )
    if before != after or not gradients_empty:
        raise RuntimeError("read-only temporal collection mutated endpoint state")
    return {
        "example_ids": ordered_ids,
        "families": families,
        "feature_names": feature_names,
        "initial_payloads": torch.cat(initial_payloads, dim=0),
        "trajectory": torch.cat(trajectories, dim=0),
        "state_features": torch.cat(state_features, dim=0),
        "state_logits": torch.cat(state_logits, dim=0),
        "state_values": torch.cat(state_values, dim=0),
        "state_feature_mask": torch.cat(state_feature_mask, dim=0),
        "state_step_mask": torch.cat(state_step_mask, dim=0),
        "operation_active_target": torch.cat(operation_active_target, dim=0),
        "mutation_audit": {
            "state_before": before,
            "state_after": after,
            "state_unchanged": before == after,
            "parameter_grad_fields_empty": gradients_empty,
            "optimizer_steps": 0,
            "model_writes": 0,
        },
    }


__all__ = ["collect_temporal"]
