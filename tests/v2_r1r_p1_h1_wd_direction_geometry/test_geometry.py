from __future__ import annotations

import numpy as np
import torch

from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.geometry import (
    fit_routed_linear_ridge,
    global_angular_centroid,
    global_centroid,
    predict_record_global,
    predict_record_route,
    record_global_centroid,
    record_route_centroids,
    site_route_angular_centroids,
    site_route_centroids,
    transfer_increment_report,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, int]]]:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(12, 4, 3, 5, generator=generator)
    routes = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ]
        * 4,
        dtype=torch.long,
    ).reshape(12, 4)
    metadata = [
        {"layer_index": 0, "step_index": 0},
        {"layer_index": 1, "step_index": 0},
        {"layer_index": 0, "step_index": 1},
        {"layer_index": 1, "step_index": 1},
    ]
    weight = torch.tensor(
        [[0.2, -0.1, 0.3, 0.4, -0.2], [0.5, 0.1, -0.3, 0.2, 0.7]],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.1, -0.2])
    transfer = torch.empty(12, 4, 3, 2)
    for site, item in enumerate(metadata):
        for route in (0, 1):
            selected = routes[:, site] == route
            transfer[selected, site] = features[selected, site] @ weight.T + bias
            transfer[selected, site] += 0.05 * item["layer_index"] + 0.03 * route
    return features, transfer, routes, metadata


def test_r0_global_centroid_flattens_sites_and_slots() -> None:
    value = torch.arange(2 * 3 * 4 * 2, dtype=torch.float32).reshape(2, 3, 4, 2)
    expected = value.reshape(-1, 2).mean(0)
    assert torch.equal(global_centroid(value), expected)
    assert torch.equal(global_centroid([value[:, i] for i in range(3)]), expected)


def test_record_centroids_preserve_site_slot_geometry() -> None:
    value = torch.arange(4 * 2 * 3 * 2, dtype=torch.float32).reshape(4, 2, 3, 2)
    route = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
    global_value = record_global_centroid(value)
    assert global_value.shape == (2, 3, 2)
    assert torch.equal(predict_record_global(global_value, 4)[0], global_value)
    conditioned = record_route_centroids(value, route)
    prediction = predict_record_route(conditioned, route)
    assert conditioned.shape == (2, 2, 3, 2)
    assert prediction.shape == value.shape
    assert torch.equal(prediction[0, 0], value[:2, 0].mean(0))
    assert torch.equal(prediction[0, 1], value[:2, 1].mean(0))


def test_r1_site_route_centroid_uses_route_and_slot_pooling() -> None:
    transfer = torch.zeros(2, 2, 2, 1)
    transfer[0, 0] = 2.0
    transfer[1, 0] = 4.0
    routes = torch.tensor([[0, 1], [0, 1]])
    actual = site_route_centroids(transfer, routes, route_count=2)
    assert actual.shape == (2, 2, 1)
    assert torch.equal(actual[:, 0, 0], torch.tensor([3.0, 0.0]))
    assert torch.equal(actual[:, 1, 0], torch.tensor([0.0, 0.0]))


def test_angular_centroids_remove_amplitude_before_direction_pooling() -> None:
    value = torch.tensor([[[[10.0, 0.0]], [[0.0, 2.0]]]])
    global_direction = global_angular_centroid(value)
    assert torch.allclose(global_direction, torch.tensor([2**-0.5, 2**-0.5]))
    routes = torch.tensor([[0, 1]])
    route_direction = site_route_angular_centroids(value, routes)
    assert torch.allclose(route_direction[0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(route_direction[1, 1], torch.tensor([0.0, 1.0]))


def test_analytic_routed_ridge_recovers_shared_layer_route_head() -> None:
    features, transfer, routes, metadata = _fixture()
    fit = fit_routed_linear_ridge(
        features,
        transfer,
        routes,
        site_metadata=metadata,
        lambda_grid=(0.0, 1.0e-6, 1.0e-2),
        seed=11,
    )
    prediction = fit.predict(features, routes, site_metadata=metadata)
    report = transfer_increment_report(transfer, prediction)
    assert report["transfer_increment_nmse"] < 1.0e-8
    assert report["transfer_increment_cosine"] > 0.999999
    assert report["explained_gain"] > 0.999999
    assert set(fit.lambda_validation_nmse) == {0.0, 1.0e-6, 1.0e-2}


def test_slot_sequence_and_numpy_inputs_match_tensor_path() -> None:
    features, transfer, routes, metadata = _fixture()
    feature_sites = [features[:, index].numpy() for index in range(features.shape[1])]
    transfer_sites = [transfer[:, index].numpy() for index in range(transfer.shape[1])]
    fit = fit_routed_linear_ridge(
        feature_sites,
        transfer_sites,
        routes.numpy(),
        site_metadata=metadata,
        lambda_grid=(0.0,),
    )
    prediction = fit.predict(feature_sites, routes.numpy(), site_metadata=metadata)
    assert isinstance(prediction, tuple)
    assert len(prediction) == 4
    assert np.isfinite(prediction[0].numpy()).all()


def test_report_compares_against_explicit_baseline() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    prediction = torch.tensor([[0.5, 0.0], [0.0, 1.0]])
    baseline = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    report = transfer_increment_report(target, prediction, baseline=baseline)
    assert report["transfer_increment_sse"] == 1.25
    assert report["transfer_increment_nmse"] == 0.25
    assert report["explained_gain"] == 0.75
