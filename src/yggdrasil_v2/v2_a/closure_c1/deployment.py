from __future__ import annotations

"""Physical C1 trace-probe stripping and deployment integrity helpers."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .model import C1Config, C1Model


DEPLOYMENT_SCHEMA = "yggdrasil.v2-a.closure-c1.stripped-deployment.v1"


@dataclass
class DeploymentArtifact:
    """A stripped model plus exact state and its parameter-graph report."""

    model: C1Model
    state_dict: dict[str, Tensor]
    report: dict[str, Any]

    @property
    def state(self) -> dict[str, Tensor]:
        return self.state_dict

    def __getitem__(self, key: str) -> Any:
        if key == "model":
            return self.model
        if key in {"state", "state_dict"}:
            return self.state_dict
        if key == "report":
            return self.report
        raise KeyError(key)

    def __iter__(self):
        # Permit the ergonomic ``model, state, report = strip_trace_probe(...)``
        # form while retaining named fields for artifact writers.
        yield self.model
        yield self.state_dict
        yield self.report


def deployment_state(model: C1Model) -> dict[str, Tensor]:
    """Return only deployment parameters/buffers; reject an unstripped model."""
    state = {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }
    trace = sorted(name for name in state if name.startswith("trace_probe."))
    if trace:
        raise ValueError("deployment_state requires a model without trace_probe parameters")
    return state


def _filtered_state(model: C1Model) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
        if not name.startswith("trace_probe.")
    }


def _deployment_report(model: C1Model, state: Mapping[str, Tensor]) -> dict[str, Any]:
    names = sorted(state)
    graph = model.parameter_report()
    return {
        "schema": DEPLOYMENT_SCHEMA,
        "stripped": True,
        "trace_probe_present": False,
        "trace_probe_parameters": 0,
        "state_keys": len(names),
        "state_parameter_names": names,
        "parameter_graph": graph["parameter_graph"],
        "total_parameters": graph["total_parameters"],
        "deployment_parameters": graph["deployment_parameters"],
        "source_closed_after_h0": True,
        "answer_logits_from": "H10 only",
    }


def strip_trace_probe(model: C1Model) -> DeploymentArtifact:
    """Re-instantiate C1 without the training probe and strict-reload weights."""
    if not isinstance(model, C1Model):
        raise TypeError("strip_trace_probe expects C1Model")
    filtered = _filtered_state(model)
    stripped = C1Model(model.config, with_trace_probe=False)
    stripped.load_state_dict(filtered, strict=True)
    if stripped.trace_probe is not None or any(name.startswith("trace_probe.") for name in stripped.state_dict()):
        raise RuntimeError("trace probe was not physically removed")
    report = _deployment_report(stripped, filtered)
    return DeploymentArtifact(stripped, filtered, report)


def build_deployment(model: C1Model) -> DeploymentArtifact:
    """Alias used by training/export callers."""
    return strip_trace_probe(model)


export_deployment = build_deployment
physical_strip = strip_trace_probe
strip_training_probe = strip_trace_probe


def reload_stripped(
    state_dict: Mapping[str, Tensor],
    config: C1Config,
    *,
    device: torch.device | str | None = None,
) -> DeploymentArtifact:
    """Create a fresh probe-free model and strict-load a deployment state."""
    trace = sorted(name for name in state_dict if name.startswith("trace_probe."))
    if trace:
        raise ValueError("stripped state contains trace_probe parameters")
    model = C1Model(config, with_trace_probe=False)
    model.load_state_dict(dict(state_dict), strict=True)
    if device is not None:
        model.to(device)
    state = deployment_state(model)
    return DeploymentArtifact(model, state, _deployment_report(model, state))


def parameter_graph_report(model: C1Model) -> dict[str, Any]:
    """Expose a compact parameter graph for the strip/accounting ledger."""
    report = model.parameter_report()
    report["trace_probe_present"] = model.trace_probe is not None
    report["trace_probe_parameter_names"] = sorted(
        name for name, _ in model.named_parameters() if name.startswith("trace_probe.")
    )
    return report


__all__ = [
    "DEPLOYMENT_SCHEMA",
    "DeploymentArtifact",
    "build_deployment",
    "deployment_state",
    "export_deployment",
    "parameter_graph_report",
    "physical_strip",
    "reload_stripped",
    "strip_trace_probe",
    "strip_training_probe",
]
