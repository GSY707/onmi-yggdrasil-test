from __future__ import annotations

"""Static and runtime architecture audits for the H1 routed projection core."""

import ast
import inspect
import math
from pathlib import Path
from typing import Any

import torch

from .model import H1Config, H1LatentReasoner, build_matched_pair
_ROUTE_INTERVENTIONS = ("flip_route", "force_route0", "force_route1", "swap_experts")


def _tensor_equal(left: Any, right: Any) -> bool:
    return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)


def _matched_initialization_report(
    shared: H1LatentReasoner, mixed: H1LatentReasoner
) -> dict[str, Any]:
    shared_state = shared.state_dict()
    mixed_state = mixed.state_dict()
    common = sorted(set(shared_state) & set(mixed_state))
    mismatches = [
        name for name in common if not torch.equal(shared_state[name], mixed_state[name])
    ]
    common_copies = []
    projection_copies = []
    for layer_index, (left, right) in enumerate(zip(shared.layers, mixed.layers, strict=True)):
        common_copies.append(
            {
                "layer": layer_index,
                "feature_trunk_equal": all(
                    torch.equal(a, b)
                    for a, b in zip(
                        left.routed_feature_trunk.parameters(),
                        right.routed_feature_trunk.parameters(),
                        strict=True,
                    )
                ),
                "shared_ffn_equal": all(
                    torch.equal(a, b)
                    for a, b in zip(
                        left.shared_ffn.parameters(),
                        right.shared_ffn.parameters(),
                        strict=True,
                    )
                ),
            }
        )
        for expert_index in (0, 1):
            projection_copies.append(
                {
                    "layer": layer_index,
                    "expert": expert_index,
                    "equal": all(
                        torch.equal(a, b)
                        for a, b in zip(
                            left.routed_projections[0].parameters(),
                            right.routed_projections[expert_index].parameters(),
                            strict=True,
                        )
                    ),
                }
            )
    return {
        "common_parameter_count": len(common),
        "common_mismatches": mismatches,
        "common_copies": common_copies,
        "routed_projection_copies": projection_copies,
        "passed": (
            not mismatches
            and all(row["feature_trunk_equal"] and row["shared_ffn_equal"] for row in common_copies)
            and all(row["equal"] for row in projection_copies)
        ),
    }


def architecture_audit(
    config: H1Config,
    *,
    seed: int = 2026081700,
    source_tokens: int = 11,
) -> dict[str, Any]:
    """Exercise the architecture contract without task targets or formal data."""
    shared, mixed = build_matched_pair(config, seed=seed)
    initialization = _matched_initialization_report(shared, mixed)
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0xA11D17)
    hidden = torch.randn(
        4,
        source_tokens,
        config.source_width,
        generator=generator,
    )
    mask = torch.ones(4, source_tokens, dtype=torch.bool)
    mask[0, -2:] = False

    shared.eval()
    baseline = shared(hidden, mask, return_trajectory=True)
    noop_checks: dict[str, bool] = {}
    for intervention in _ROUTE_INTERVENTIONS:
        trial = shared(
            hidden,
            mask,
            intervention=intervention,
            return_trajectory=True,
        )
        noop_checks[intervention] = all(
            _tensor_equal(baseline[name], trial[name])
            for name in ("logits", "final_state", "trajectory", "trace_logits")
        )

    kv_calls = [0 for _ in mixed.layers]
    cross_calls = [0 for _ in mixed.layers]
    slot_calls = [0 for _ in mixed.layers]
    shared_ffn_calls = {"count": 0}
    routed_projection_calls = {0: 0, 1: 0}
    feature_trunk_calls = {"count": 0}
    hooks = []
    for layer_index, layer in enumerate(mixed.layers):
        hooks.append(
            layer.cross.key.register_forward_hook(
                lambda *_args, index=layer_index: kv_calls.__setitem__(
                    index, kv_calls[index] + 1
                )
            )
        )
        hooks.append(
            layer.cross.register_forward_hook(
                lambda *_args, index=layer_index: cross_calls.__setitem__(
                    index, cross_calls[index] + 1
                )
            )
        )
        hooks.append(
            layer.slot.register_forward_hook(
                lambda *_args, index=layer_index: slot_calls.__setitem__(
                    index, slot_calls[index] + 1
                )
            )
        )
        hooks.append(
            layer.shared_ffn.register_forward_hook(
                lambda *_args: shared_ffn_calls.__setitem__(
                    "count", shared_ffn_calls["count"] + 1
                )
            )
        )
        hooks.append(
            layer.routed_feature_trunk.register_forward_hook(
                lambda *_args: feature_trunk_calls.__setitem__(
                    "count", feature_trunk_calls["count"] + 1
                )
            )
        )
        for expert_index, projection in enumerate(layer.routed_projections):
            hooks.append(
                projection.register_forward_hook(
                    lambda *_args, index=expert_index: routed_projection_calls.__setitem__(
                        index, routed_projection_calls[index] + 1
                    )
                )
            )
    mixed.train()
    forced = mixed(
        hidden,
        mask,
        intervention="force_route0",
        return_trajectory=True,
    )
    assert forced["logits"] is not None and forced["trace_logits"] is not None
    (forced["logits"].sum() + forced["trace_logits"].sum()).backward()
    for hook in hooks:
        hook.remove()
    inactive_grad_none = all(
        parameter.grad is None
        for layer in mixed.layers
        for parameter in layer.routed_projections[1].parameters()
    )

    with torch.no_grad():
        normal = mixed(hidden, mask, return_trajectory=True)
        disabled_projection = mixed(
            hidden,
            mask,
            intervention="disable_routed_projection",
            return_trajectory=True,
        )
    disable_projection_changes_state = not _tensor_equal(
        normal["final_state"], disabled_projection["final_state"]
    )
    projection_nonzero = all(
        bool(torch.count_nonzero(parameter))
        for layer in mixed.layers
        for projection in layer.routed_projections
        for parameter in projection.parameters()
    )

    with torch.no_grad():
        source = hidden.to(dtype=next(mixed.parameters()).dtype)
        _, initial = mixed._boundary(source, mask)
        disabled = mixed(hidden, mask, intervention="disable_recurrence")
    disabled_h0 = _tensor_equal(disabled["final_state"], initial)

    import yggdrasil_v2.r1_revalidation.h1.model as model_module

    tree = ast.parse(inspect.getsource(model_module))
    imports = " ".join(
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ).casefold()
    forward_parameters = list(inspect.signature(H1LatentReasoner.forward).parameters)
    integrity = mixed.integrity_report()
    shared_flops = shared.active_flop_estimate(batch_size=4, source_tokens=source_tokens)
    mixed_flops = mixed.active_flop_estimate(batch_size=4, source_tokens=source_tokens)
    shared_parameters = shared.parameter_report()
    mixed_parameters = mixed.parameter_report()
    expected_integrity = {
        "source_width": config.source_width,
        "latent_width": config.latent_width,
        "K": config.K,
        "T": config.T,
        "recurrent_layers": config.recurrent_layers,
        "fixed_step_control": True,
        "step_control_type": "fixed_fourier",
        "step_control_shape": [config.T, config.latent_width],
        "step_control_trainable": False,
        "source_kv_projections_per_forward": config.recurrent_layers,
        "cross_attention_calls_per_forward": config.recurrent_layers * config.T,
        "source_independent_initial_state": True,
        "common_attention_and_ffn_state_write": True,
        "routed_feature_trunk_shared_across_routes": True,
        "selected_routed_projection_is_only_conditional_state_write": True,
        "route_controls_final_projection_only": True,
        "route_granularity": "record",
        "conditional_transition_scope": "factorized-routed-state-write-projection",
        "routed_projection_standard_nonzero_init": True,
        "routed_feature_width": config.routed_feature_width,
        "routed_projection_input_width": config.routed_feature_width,
        "routed_projection_output_width": config.latent_width,
        "routed_projection_count": 2,
        "answer_reads_pooled_final_state_only": True,
        "route_assignment_is_hard_top1": True,
    }
    checks = {
        "forward_signature_is_model_view_only": forward_parameters
        == ["self", "hidden", "mask", "intervention", "return_trajectory"],
        "forbidden_imports_absent": all(
            token not in imports
            for token in ("simulator", "nr1.reference", "nr1.measurement", "production.generator")
        ),
        "integrity_fields": all(integrity.get(key) == value for key, value in expected_integrity.items()),
        "matched_initialization": initialization["passed"] is True,
        "shared_route_noop": all(noop_checks.values()),
        "source_kv_once_per_layer": kv_calls == [1] * config.recurrent_layers,
        "cross_each_layer_step": cross_calls == [config.T] * config.recurrent_layers,
        "slot_each_layer_step": slot_calls == [config.T] * config.recurrent_layers,
        "common_ffn_each_layer_step": shared_ffn_calls["count"]
        == config.T * config.recurrent_layers,
        "routed_feature_trunk_each_layer_step": feature_trunk_calls["count"]
        == config.T * config.recurrent_layers,
        "one_sparse_projection_only": routed_projection_calls
        == {0: config.T * config.recurrent_layers, 1: 0},
        "inactive_projection_has_no_gradient": inactive_grad_none,
        "routed_projection_nonzero": projection_nonzero,
        "disable_routed_projection_changes_state": disable_projection_changes_state,
        "disable_recurrence_is_h0": disabled_h0,
        "route_shapes": (
            forced["route_logits"] is not None
            and list(forced["route_logits"].shape)
            == [4, config.recurrent_layers, config.T, 2]
            and forced["route_assignments"] is not None
            and list(forced["route_assignments"].shape)
            == [4, config.recurrent_layers, config.T]
        ),
        "active_flops_matched": math.isclose(shared_flops, mixed_flops, rel_tol=0.0, abs_tol=0.0),
        "active_parameters_matched": (
            shared_parameters["active_parameters_per_record"]
            == mixed_parameters["active_parameters_per_record"]
        ),
    }
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.architecture-audit.v4",
        "checks": checks,
        "passed": all(checks.values()),
        "forward_parameters": forward_parameters,
        "imports": imports,
        "integrity": integrity,
        "initialization": initialization,
        "shared_route_noop": noop_checks,
        "runtime_calls": {
            "source_kv": kv_calls,
            "cross": cross_calls,
            "slot": slot_calls,
            "shared_ffn": shared_ffn_calls["count"],
            "routed_feature_trunk": feature_trunk_calls["count"],
            "routed_projections": routed_projection_calls,
            "inactive_routed_projection_gradient_none": inactive_grad_none,
        },
        "disable_recurrence_is_h0": disabled_h0,
        "active_flops": {"shared": shared_flops, "mixed": mixed_flops},
        "parameters": {"shared": shared_parameters, "mixed": mixed_parameters},
    }


__all__ = ["architecture_audit"]
