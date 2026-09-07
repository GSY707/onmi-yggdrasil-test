from __future__ import annotations

"""Sampled matrix-free local Jacobian/Fisher reachability.

The operator freezes each captured attention state.  It therefore measures the
local ``routed_feature_trunk + selected final head`` path, not the recurrent
full-model Jacobian.  No dense Jacobian and no model parameter mutation occur.
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor
from torch.func import functional_call, jvp, vjp


@dataclass(frozen=True)
class LocalJacobianProblem:
    states: Tensor
    routes: Tensor
    targets: Tensor

    def validate(self) -> None:
        if self.states.ndim != 4 or self.states.shape[-1] != 256:
            raise ValueError("states must have shape [records, steps, slots, 256]")
        if self.targets.shape != self.states.shape:
            raise ValueError("targets must match states")
        if self.routes.shape != (self.states.shape[0],):
            raise ValueError("routes must have shape [records]")
        if not bool(torch.all((self.routes == 0) | (self.routes == 1))):
            raise ValueError("routes must be binary")
        if not bool(torch.isfinite(self.states).all() and torch.isfinite(self.targets).all()):
            raise ValueError("local Jacobian problem contains non-finite values")


def _parameter_surface(layer: Any) -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    names: list[str] = []
    values: list[Tensor] = []
    for prefix, module in (
        ("trunk", layer.routed_feature_trunk),
        ("head0", layer.routed_projections[0]),
        ("head1", layer.routed_projections[1]),
    ):
        for name, parameter in module.named_parameters():
            names.append(f"{prefix}.{name}")
            values.append(parameter.detach())
    return tuple(names), tuple(values)


def _output_function(
    layer: Any,
    names: Sequence[str],
    states: Tensor,
    routes: Tensor,
) -> Callable[..., Tensor]:
    def output(*values: Tensor) -> Tensor:
        supplied = dict(zip(names, values, strict=True))
        trunk_params = {
            name.removeprefix("trunk."): value
            for name, value in supplied.items()
            if name.startswith("trunk.")
        }
        head0_params = {
            name.removeprefix("head0."): value
            for name, value in supplied.items()
            if name.startswith("head0.")
        }
        head1_params = {
            name.removeprefix("head1."): value
            for name, value in supplied.items()
            if name.startswith("head1.")
        }
        features = functional_call(
            layer.routed_feature_trunk, trunk_params, (states,)
        )
        output0 = functional_call(layer.routed_projections[0], head0_params, (features,))
        output1 = functional_call(layer.routed_projections[1], head1_params, (features,))
        selector = routes.reshape(-1, 1, 1, 1).to(dtype=torch.bool)
        return torch.where(selector, output1, output0)

    return output


def _flatten(values: Sequence[Tensor]) -> Tensor:
    return torch.cat(tuple(value.reshape(-1) for value in values))


def _unflatten(vector: Tensor, references: Sequence[Tensor]) -> tuple[Tensor, ...]:
    pieces: list[Tensor] = []
    cursor = 0
    for reference in references:
        size = reference.numel()
        pieces.append(vector[cursor : cursor + size].reshape_as(reference))
        cursor += size
    if cursor != vector.numel():
        raise ValueError("parameter tangent size mismatch")
    return tuple(pieces)


def _metrics(target: Tensor, prediction: Tensor) -> dict[str, float]:
    target64 = target.double().reshape(-1)
    prediction64 = prediction.double().reshape(-1)
    energy = float(torch.dot(target64, target64))
    pred_energy = float(torch.dot(prediction64, prediction64))
    dot = float(torch.dot(target64, prediction64))
    sse = max(0.0, energy + pred_energy - 2.0 * dot)
    return {
        "target_energy": energy,
        "prediction_energy": pred_energy,
        "dot": dot,
        "sse": sse,
        "increment_nmse": sse / max(energy, 1.0e-20),
        "explained_energy_gain": 1.0 - sse / max(energy, 1.0e-20),
        "cosine": dot / max((energy * pred_energy) ** 0.5, 1.0e-20),
    }


def solve_local_projection_jacobian(
    layer: Any,
    train: LocalJacobianProblem,
    heldout: LocalJacobianProblem,
    *,
    damping: float = 1.0e-4,
    cg_iterations: int = 16,
    tolerance: float = 1.0e-5,
    fisher_probes: int = 4,
    seed: int = 2026082304,
) -> dict[str, Any]:
    """Solve one shared local projection tangent with JVP/VJP conjugate gradient."""

    train.validate()
    heldout.validate()
    if damping <= 0.0 or cg_iterations <= 0 or fisher_probes <= 0:
        raise ValueError("invalid matrix-free solver configuration")
    names, base = _parameter_surface(layer)
    train_fn = _output_function(layer, names, train.states, train.routes)
    heldout_fn = _output_function(layer, names, heldout.states, heldout.routes)
    base_output, train_vjp = vjp(train_fn, *base)
    output_elements = float(base_output.numel())
    right_tuple = train_vjp(train.targets)
    right = _flatten(right_tuple) / output_elements
    if not bool(torch.isfinite(right).all()):
        raise RuntimeError("matrix-free Jacobian right-hand side is non-finite")

    matvec_calls = 0

    def matvec(vector: Tensor) -> Tensor:
        nonlocal matvec_calls
        tangent = _unflatten(vector, base)
        _, projected = jvp(train_fn, base, tangent)
        pulled = _flatten(train_vjp(projected)) / output_elements
        matvec_calls += 1
        result = pulled + float(damping) * vector
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("matrix-free Jacobian matvec is non-finite")
        return result

    solution = torch.zeros_like(right)
    residual = right - matvec(solution)
    direction = residual.clone()
    rr = torch.dot(residual, residual)
    initial_norm = float(rr.sqrt())
    history = [initial_norm]
    converged = initial_norm == 0.0
    for _ in range(cg_iterations):
        if converged:
            break
        applied = matvec(direction)
        denominator = torch.dot(direction, applied).clamp_min(1.0e-30)
        if not bool(torch.isfinite(denominator)):
            raise RuntimeError("matrix-free CG denominator is non-finite")
        alpha = rr / denominator
        if not bool(torch.isfinite(alpha)):
            raise RuntimeError("matrix-free CG step is non-finite")
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        next_rr = torch.dot(residual, residual)
        if not bool(torch.isfinite(next_rr)):
            raise RuntimeError("matrix-free CG residual is non-finite")
        residual_norm = float(next_rr.sqrt())
        history.append(residual_norm)
        if residual_norm <= tolerance * max(initial_norm, 1.0e-30):
            converged = True
            rr = next_rr
            break
        beta = next_rr / rr.clamp_min(1.0e-30)
        direction = residual + beta * direction
        rr = next_rr

    tangent = _unflatten(solution, base)
    _, train_prediction = jvp(train_fn, base, tangent)
    _, heldout_prediction = jvp(heldout_fn, base, tangent)
    if not bool(
        torch.isfinite(train_prediction).all()
        and torch.isfinite(heldout_prediction).all()
    ):
        raise RuntimeError("matrix-free Jacobian prediction is non-finite")

    generator = torch.Generator(device=train.targets.device)
    generator.manual_seed(seed)
    fisher_trace_samples: list[float] = []
    for _ in range(fisher_probes):
        signs = torch.randint(
            0,
            2,
            train.targets.shape,
            generator=generator,
            device=train.targets.device,
            dtype=torch.int8,
        ).to(dtype=train.targets.dtype)
        signs.mul_(2).sub_(1)
        pulled = _flatten(train_vjp(signs)) / output_elements**0.5
        fisher_trace_samples.append(float(torch.dot(pulled, pulled)))

    # A small, norm-controlled finite-difference audit checks that the JVP
    # operator is wired to the same frozen local path.  It does not apply the
    # fitted update to the model.
    base_norm = float(_flatten(base).norm())
    tangent_norm = float(solution.norm())
    audit_scale = (1.0e-3 * max(base_norm, 1.0)) / max(tangent_norm, 1.0e-20)
    scaled = tuple(value * audit_scale for value in tangent)
    epsilon = 1.0e-3
    shifted = tuple(
        parameter + epsilon * delta for parameter, delta in zip(base, scaled, strict=True)
    )
    finite_difference = (train_fn(*shifted) - base_output) / epsilon
    _, linearized = jvp(train_fn, base, scaled)
    linearization_error = float(
        (finite_difference - linearized).norm()
        / linearized.norm().clamp_min(1.0e-20)
    )

    return {
        "parameter_block": "routed_feature_trunk_plus_two_selected_final_heads",
        "local_path_only": True,
        "recurrent_state_jacobian_included": False,
        "dense_jacobian_materialized": False,
        "parameter_count": int(solution.numel()),
        "train_records": int(train.states.shape[0]),
        "heldout_records": int(heldout.states.shape[0]),
        "damping": float(damping),
        "cg": {
            "iterations_requested": int(cg_iterations),
            "iterations_completed": len(history) - 1,
            "converged": bool(converged),
            "relative_residual": history[-1] / max(history[0], 1.0e-30),
            "residual_history": history,
            "matvec_calls": matvec_calls,
        },
        "tangent_norm": tangent_norm,
        "base_parameter_norm": base_norm,
        "fisher_trace_hutchinson": {
            "probes": int(fisher_probes),
            "samples": fisher_trace_samples,
            "mean": sum(fisher_trace_samples) / len(fisher_trace_samples),
        },
        "finite_difference_relative_error": linearization_error,
        "train": _metrics(train.targets, train_prediction),
        "heldout": _metrics(heldout.targets, heldout_prediction),
        # Returned internally for the screen's heldout residual accounting;
        # callers must remove tensors before JSON serialization.
        "_tangent": tuple(value.detach() for value in tangent),
        "_parameter_names": tuple(names),
    }


def apply_local_tangent(
    layer: Any,
    problem: LocalJacobianProblem,
    tangent: Sequence[Tensor],
) -> Tensor:
    problem.validate()
    names, base = _parameter_surface(layer)
    if len(tangent) != len(base):
        raise ValueError("local tangent surface mismatch")
    function = _output_function(layer, names, problem.states, problem.routes)
    return jvp(function, base, tuple(tangent))[1]


__all__ = [
    "LocalJacobianProblem",
    "apply_local_tangent",
    "solve_local_projection_jacobian",
]
