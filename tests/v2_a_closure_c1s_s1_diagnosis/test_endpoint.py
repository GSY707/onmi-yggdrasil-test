from __future__ import annotations

from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis import artifacts
from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis.endpoint import (
    ENDPOINT_SCHEMA,
    EndpointIntegrityError,
    load_endpoint_artifact,
)


def _write_endpoint(path: Path, *, identity: str = "S1", update: int = 4000) -> str:
    payload = {
        "schema_version": ENDPOINT_SCHEMA,
        "identity": identity,
        "update": update,
        "config": {"slots": 8, "payload_width": 4},
        "schedule_sha256": "AB" * 32,
        "model_state": {"weight": torch.arange(4, dtype=torch.float32)},
        "diagnostic_state": {"bias": torch.zeros(2, dtype=torch.float32)},
        "metrics": {"loss": 0.1},
        "optimizer_state_saved": False,
        "checkpoint_selection": False,
    }
    torch.save(payload, path)
    return artifacts.sha256_file(path)


def test_loader_validates_identity_hash_and_returns_only_cpu_states(tmp_path: Path) -> None:
    path = tmp_path / "endpoint.pt"
    digest = _write_endpoint(path)
    loaded = load_endpoint_artifact(
        path,
        expected_sha256=digest,
        expected_identity="S1",
        expected_update=4000,
        expected_schedule_sha256="AB" * 32,
        expected_model_config={"slots": 8, "payload_width": 4},
    )
    assert set(loaded) == {"model_state", "diagnostic_state", "metadata"}
    assert loaded["model_state"]["weight"].device.type == "cpu"
    assert loaded["diagnostic_state"]["bias"].device.type == "cpu"
    assert loaded["metadata"]["sha256"] == digest
    assert "optimizer_state_saved" in loaded["metadata"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_sha256": "0" * 64},
        {"expected_identity": "wrong"},
        {"expected_update": 3999},
        {"expected_schedule_sha256": "CD" * 32},
        {"expected_model_config": {"slots": 1}},
    ],
)
def test_loader_rejects_pin_mismatch_before_use(tmp_path: Path, kwargs) -> None:
    path = tmp_path / "endpoint.pt"
    digest = _write_endpoint(path)
    if "expected_sha256" not in kwargs:
        kwargs = {"expected_sha256": digest, **kwargs}
    with pytest.raises(EndpointIntegrityError):
        load_endpoint_artifact(path, **kwargs)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update({"optimizer_state_saved": True}),
        lambda p: p.update({"checkpoint_selection": True}),
        lambda p: p.update({"model_state": {"weight": torch.tensor([float("nan")])}}),
        lambda p: p.update({"schema_version": "wrong.schema"}),
    ],
)
def test_loader_rejects_unsafe_or_malformed_payload(tmp_path: Path, mutation) -> None:
    path = tmp_path / "endpoint.pt"
    payload = {
        "schema_version": ENDPOINT_SCHEMA,
        "identity": "S1",
        "update": 1,
        "config": {"slots": 1},
        "schedule_sha256": "AB" * 32,
        "model_state": {"weight": torch.ones(1)},
        "diagnostic_state": {"bias": torch.zeros(1)},
        "optimizer_state_saved": False,
        "checkpoint_selection": False,
    }
    mutation(payload)
    torch.save(payload, path)
    with pytest.raises(EndpointIntegrityError):
        load_endpoint_artifact(path)


def test_loader_hashes_before_deserializing_and_does_not_write(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "endpoint.pt"
    digest = _write_endpoint(path)
    calls: list[bool] = []
    original = torch.load

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get("weights_only") is True)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", wrapped)
    before = sorted(item.relative_to(tmp_path).as_posix() for item in tmp_path.rglob("*"))
    load_endpoint_artifact(path, expected_sha256=digest)
    after = sorted(item.relative_to(tmp_path).as_posix() for item in tmp_path.rglob("*"))
    assert calls == [True]
    assert after == before
