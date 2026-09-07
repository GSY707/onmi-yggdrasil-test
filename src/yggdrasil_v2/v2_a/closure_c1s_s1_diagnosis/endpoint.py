from __future__ import annotations

"""Read-only, identity-pinned endpoint loading for S1 diagnosis.

The loader intentionally returns state dictionaries rather than constructing a
model, heads, optimizer, or checkpoint-selection object.  The caller can then
instantiate the already-registered model in memory and prove that loading did
not write or mutate a training state.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .artifacts import sha256_file


ENDPOINT_SCHEMA = "yggdrasil.v2-a.closure-c1s-successor.endpoint.v1"


class EndpointIntegrityError(ValueError):
    """Raised when an endpoint is absent, malformed, or not contract-matching."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EndpointIntegrityError(message)


def _cpu_state(value: Any, *, field: str) -> dict[str, torch.Tensor]:
    _require(isinstance(value, Mapping), f"{field} must be a state-dict mapping")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        _require(isinstance(name, str), f"{field} key must be str")
        _require(torch.is_tensor(tensor), f"{field}[{name!r}] must be a tensor")
        _require(bool(torch.isfinite(tensor).all()), f"{field}[{name!r}] contains non-finite values")
        # clone prevents a caller from holding an alias to an mmap/storage
        # owned by torch.load; all returned values are CPU-only and detached.
        result[name] = tensor.detach().to(device="cpu").clone()
    return result


def _load_weights_only(path: Path) -> Mapping[str, Any]:
    # Do not fall back to weights_only=False: that mode can execute arbitrary
    # pickle constructors and would violate the diagnosis read-only boundary.
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, Mapping), "endpoint payload must be a mapping")
    return value


def load_endpoint_artifact(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_schema_version: str = ENDPOINT_SCHEMA,
    expected_identity: str | None = None,
    expected_update: int | None = None,
    expected_schedule_sha256: str | None = None,
    expected_model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate an endpoint, returning only CPU state and metadata.

    File hashing happens before deserialization.  All supplied expectations are
    exact (case-insensitive only for hexadecimal SHA values), and optimizer
    state/checkpoint-selection payloads are rejected even when they are merely
    present rather than used.  No output path is created or written.
    """

    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"regular endpoint file required: {path}")
    actual_sha = sha256_file(path)
    if expected_sha256 is not None:
        _require(actual_sha == str(expected_sha256).upper(), "endpoint SHA256 mismatch")

    payload = _load_weights_only(path)
    schema = payload.get("schema_version")
    _require(schema == expected_schema_version, f"endpoint schema mismatch: {schema!r}")
    if expected_identity is not None:
        _require(payload.get("identity") == expected_identity, "endpoint identity mismatch")
    if expected_update is not None:
        _require(payload.get("update") == int(expected_update), "endpoint update mismatch")
    if expected_schedule_sha256 is not None:
        _require(
            payload.get("schedule_sha256") == str(expected_schedule_sha256).upper(),
            "endpoint schedule SHA256 mismatch",
        )
    if expected_model_config is not None:
        _require(dict(payload.get("config", {})) == dict(expected_model_config), "endpoint model config mismatch")

    config = payload.get("config")
    _require(isinstance(config, Mapping), "endpoint config must be an object")
    update = payload.get("update")
    _require(isinstance(update, int) and not isinstance(update, bool) and update >= 0, "endpoint update must be a non-negative integer")
    schedule = payload.get("schedule_sha256")
    _require(isinstance(schedule, str) and len(schedule) == 64, "endpoint schedule SHA256 is malformed")
    _require(payload.get("optimizer_state_saved") is False, "optimizer state must not be saved")
    _require(payload.get("checkpoint_selection") is False, "checkpoint selection must be disabled")

    model_state = _cpu_state(payload.get("model_state"), field="model_state")
    diagnostic_state = _cpu_state(payload.get("diagnostic_state"), field="diagnostic_state")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "diagnostic_state"}
    }
    metadata["sha256"] = actual_sha
    return {
        "model_state": model_state,
        "diagnostic_state": diagnostic_state,
        "metadata": metadata,
    }


# Name chosen to mirror the train-side operation while remaining a different
# function with a deliberately different return contract.
load_endpoint = load_endpoint_artifact


__all__ = ["ENDPOINT_SCHEMA", "EndpointIntegrityError", "load_endpoint", "load_endpoint_artifact"]
