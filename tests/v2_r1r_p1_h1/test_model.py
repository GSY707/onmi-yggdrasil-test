import ast
import inspect

import torch

from yggdrasil_v2.r1_revalidation.h1.model import (
    H1Config,
    H1LatentReasoner,
    build_matched_pair,
)


def _inputs(batch=3, tokens=11, width=24):
    torch.manual_seed(7)
    return torch.randn(batch, tokens, width), torch.ones(batch, tokens, dtype=torch.bool)


def _config(arm="shared"):
    return H1Config(
        arm=arm,
        source_width=24,
        latent_width=16,
        K=8,
        T=8,
        recurrent_layers=2,
        num_heads=4,
        active_ffn_multiplier=3,
        shared_ffn_inner_width=24,
        routed_feature_width=24,
    )


def test_shape_and_public_forward_contract():
    model = H1LatentReasoner(_config())
    hidden, mask = _inputs(width=24)
    out = model(hidden, mask, return_trajectory=True)
    assert out["logits"].shape == (3, 4)
    assert out["final_state"].shape == (3, 8, 16)
    assert out["trajectory"].shape == (3, 8, 8, 16)
    assert out["trace_logits"].shape == (3, 8, 5)
    assert out["route_logits"].shape == (3, 2, 8, 2)
    assert out["route_assignments"].shape == (3, 2, 8)

    params = list(inspect.signature(model.forward).parameters)
    assert params == ["hidden", "mask", "intervention", "return_trajectory"]


def test_answer_head_receives_only_final_pooled_state():
    model = H1LatentReasoner(_config())
    hidden, mask = _inputs(width=24)
    seen = []
    handle = model.answer_head.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    model(hidden, mask)
    handle.remove()
    assert len(seen) == 1
    assert seen[0].shape == (hidden.shape[0], 16)


def test_source_kv_is_projected_once_but_cross_query_runs_each_step():
    model = H1LatentReasoner(_config())
    hidden, mask = _inputs(width=24)
    kv_calls = [0 for _ in model.layers]
    attn_calls = [0 for _ in model.layers]
    handles = []
    for index, layer in enumerate(model.layers):
        handles.append(layer.cross.key.register_forward_hook(lambda *_args, i=index: kv_calls.__setitem__(i, kv_calls[i] + 1)))
        handles.append(layer.cross.register_forward_hook(lambda *_args, i=index: attn_calls.__setitem__(i, attn_calls[i] + 1)))
    model(hidden, mask)
    for handle in handles:
        handle.remove()
    assert kv_calls == [1, 1]
    assert attn_calls == [8, 8]


def test_fixed_fourier_step_control_is_public_and_nonsemantic():
    model = H1LatentReasoner(_config())
    assert model.step_control.shape == (8, 16)
    assert model.step_control.requires_grad is False
    assert not torch.equal(model.step_control[0], model.step_control[1])
    assert all(parameter.requires_grad for parameter in model.control_projection.parameters())
    report = model.integrity_report()
    assert report["fixed_step_control"] is True
    assert report["step_control_shape"] == [8, 16]
    assert report["step_control_trainable"] is False


def test_step_control_conditions_attention_without_becoming_payload_residual():
    model = H1LatentReasoner(_config())
    hidden, mask = _inputs(width=24)
    with torch.no_grad():
        for layer in model.layers:
            for parameter in layer.cross.parameters():
                parameter.zero_()
            for parameter in layer.slot.parameters():
                parameter.zero_()
            for parameter in layer.shared_ffn.parameters():
                parameter.zero_()
        source = hidden.to(dtype=next(model.parameters()).dtype)
        _, initial = model._boundary(source, mask)
        output = model(hidden, mask, intervention="disable_routed_projection")
    assert torch.equal(output["final_state"], initial)


def test_interventions_and_sparse_route_shape():
    model = H1LatentReasoner(_config("mixed"))
    hidden, mask = _inputs(width=24)
    normal = model(hidden, mask)
    forced = model(hidden, mask, intervention="force_route1")
    flipped = model(hidden, mask, intervention="flip_route")
    assert torch.all(forced["route_assignments"] == 1)
    assert torch.any(flipped["route_assignments"] != normal["route_assignments"])


def test_shared_route_interventions_are_strict_noops():
    model = H1LatentReasoner(_config("shared"))
    hidden, mask = _inputs(width=24)
    baseline = model(hidden, mask)
    for intervention in ("flip_route", "force_route0", "force_route1", "swap_experts"):
        trial = model(hidden, mask, intervention=intervention)
        assert torch.equal(trial["logits"], baseline["logits"])
        assert torch.equal(trial["final_state"], baseline["final_state"])


def test_mixed_swap_changes_output_when_experts_differ():
    model = H1LatentReasoner(_config("mixed"))
    with torch.no_grad():
        for layer in model.layers:
            layer.routed_projections[1].bias[0].add_(1.0)
    hidden, mask = _inputs(width=24)
    normal = model(hidden, mask)
    swapped = model(hidden, mask, intervention="swap_experts")
    assert not torch.allclose(normal["logits"], swapped["logits"])


def test_disable_recurrence_and_source_interventions_change_state():
    model = H1LatentReasoner(_config())
    hidden, mask = _inputs(width=24)
    baseline = model(hidden, mask)
    assert not torch.allclose(
        baseline["final_state"], model(hidden, mask, intervention="disable_recurrence")["final_state"]
    )
    assert not torch.allclose(
        baseline["final_state"], model(hidden, mask, intervention="zero_source")["final_state"]
    )
    assert not torch.allclose(
        baseline["final_state"], model(hidden, mask, intervention="shuffle_source")["final_state"]
    )


def test_common_ffn_and_selected_projection_are_only_writes_and_inactive_head_is_not_called():
    model = H1LatentReasoner(_config("mixed"))
    hidden, mask = _inputs(width=24)
    common_calls = {"count": 0}
    trunk_calls = {"count": 0}
    projection_calls = {0: 0, 1: 0}
    handles = []
    for layer in model.layers:
        handles.append(
            layer.shared_ffn.register_forward_hook(
                lambda *_args: common_calls.__setitem__(
                    "count", common_calls["count"] + 1
                )
            )
        )
        handles.append(
            layer.routed_feature_trunk.register_forward_hook(
                lambda *_args: trunk_calls.__setitem__(
                    "count", trunk_calls["count"] + 1
                )
            )
        )
        for expert_index, projection in enumerate(layer.routed_projections):
            handles.append(
                projection.register_forward_hook(
                    lambda *_args, i=expert_index: projection_calls.__setitem__(
                        i, projection_calls[i] + 1
                    )
                )
            )
    output = model(
        hidden,
        mask,
        intervention="force_route0",
        return_trajectory=True,
    )
    output["logits"].sum().backward()
    for handle in handles:
        handle.remove()
    assert common_calls == {"count": 16}
    assert trunk_calls == {"count": 16}
    assert projection_calls == {0: 16, 1: 0}
    assert all(
        parameter.grad is None
        for layer in model.layers
        for parameter in layer.routed_projections[1].parameters()
    )

    with torch.no_grad():
        source = hidden.to(dtype=next(model.parameters()).dtype)
        _, initial = model._boundary(source, mask)
        no_projection = model(
            hidden, mask, intervention="disable_routed_projection"
        )
        no_recurrence = model(hidden, mask, intervention="disable_recurrence")
    # FFN ablation keeps the common source/slot-attention transition.  The
    # stronger recurrence ablation is the registered H0 identity control.
    assert not torch.equal(no_projection["final_state"], initial)
    assert torch.equal(no_recurrence["final_state"], initial)


def test_routed_feature_trunk_and_projection_are_nonzero_and_projection_disable_preserves_common_ffn():
    model = H1LatentReasoner(_config("mixed"))
    hidden, mask = _inputs(width=24)
    with torch.no_grad():
        source = hidden.to(dtype=next(model.parameters()).dtype)
        _, initial = model._boundary(source, mask)
        features = model.layers[0].routed_feature_trunk(initial)
    assert features.shape == (hidden.shape[0], 8, 24)
    assert model.layers[0].routed_projections[0].weight.shape == (16, 24)
    assert all(
        torch.count_nonzero(parameter) > 0
        for layer in model.layers
        for projection in layer.routed_projections
        for parameter in projection.parameters()
    )
    normal = model(hidden, mask)
    disabled = model(hidden, mask, intervention="disable_routed_projection")
    assert not torch.equal(normal["final_state"], disabled["final_state"])
    assert disabled["final_state"].shape == normal["final_state"].shape


def test_trace_head_strip_preserves_answer_and_physical_state():
    model = H1LatentReasoner(_config())
    hidden, mask = _inputs(width=24)
    before = model(hidden, mask)["logits"]
    model.strip_trace_head()
    after = model(hidden, mask)["logits"]
    assert torch.equal(before, after)
    assert model.trace_head is None
    assert not any(name.startswith("trace_head.") for name in model.deployment_state())


def test_matched_pair_common_identity_and_active_flops():
    shared, mixed = build_matched_pair(_config(), seed=11)
    shared_state, mixed_state = shared.state_dict(), mixed.state_dict()
    for key, value in shared_state.items():
        if key in mixed_state:
            assert torch.equal(value, mixed_state[key]), key
    for left, right in zip(shared.layers, mixed.layers):
        for a, b in zip(left.shared_ffn.parameters(), right.shared_ffn.parameters()):
            assert torch.equal(a, b)
        for a, b in zip(left.routed_feature_trunk.parameters(), right.routed_feature_trunk.parameters()):
            assert torch.equal(a, b)
        for a, b in zip(left.routed_projections[0].parameters(), right.routed_projections[0].parameters()):
            assert torch.equal(a, b)
        for a, b in zip(left.routed_projections[0].parameters(), right.routed_projections[1].parameters()):
            assert torch.equal(a, b)
        assert torch.count_nonzero(left.routed_projections[0].weight) > 0
        assert torch.count_nonzero(left.routed_projections[0].bias) > 0
    hidden, mask = _inputs(width=24)
    shared_output = shared(hidden, mask, return_trajectory=True)
    mixed_output = mixed(hidden, mask, return_trajectory=True)
    for name in ("logits", "final_state", "trajectory", "trace_logits"):
        # Sparse index-select/index-copy dispatch has a harmless floating-point
        # accumulation-order difference from the dense shared call.
        assert torch.allclose(shared_output[name], mixed_output[name], atol=2e-6, rtol=0.0)
    a = shared.active_flop_estimate(batch_size=2, source_tokens=11)
    b = mixed.active_flop_estimate(batch_size=2, source_tokens=11)
    assert abs(a - b) / max(a, b) <= 0.02


def test_forward_source_has_no_forbidden_imports_or_semantic_arguments():
    import yggdrasil_v2.r1_revalidation.h1.model as module

    tree = ast.parse(inspect.getsource(module))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported = " ".join(ast.unparse(node) for node in imports)
    assert "simulator" not in imported.lower()
    signature = inspect.signature(H1LatentReasoner.forward)
    assert set(signature.parameters) == {"self", "hidden", "mask", "intervention", "return_trajectory"}
