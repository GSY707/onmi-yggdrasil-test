from __future__ import annotations

"""Read-only input validation for the H1-WD direction-geometry screen."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from yggdrasil_v2.r1_revalidation.h1.cache import TokenCache

from .contract import (
    CAUSAL_TARGET_BANK,
    CAUSAL_TARGET_BANK_MANIFEST,
    CAUSAL_TARGET_BANK_MANIFEST_SHA256,
    CAUSAL_TARGET_BANK_SHA256,
    SOURCE_CACHE_MANIFEST,
    SOURCE_CACHE_MANIFEST_SHA256,
    SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_TOKEN_CACHE,
    SPEC,
)


TARGET_SCHEMA = (
    "yggdrasil.v2-r1r.p1-h1-wd-decision-causal.target-datasets.v1"
)
EXPECTED_SPLIT_COUNTS = {"train": 4096, "heldout": 1024}
EXPECTED_SITES = tuple(
    {"layer_index": layer_index, "step_index": step_index}
    for step_index in range(8)
    for layer_index in range(2)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_split(split: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    count = EXPECTED_SPLIT_COUNTS[split]
    record_ids = payload.get("record_ids")
    _require(isinstance(record_ids, (tuple, list)), f"{split} record_ids missing")
    _require(len(record_ids) == count, f"{split} record count drift")
    _require(len(set(record_ids)) == count, f"{split} duplicate record ids")
    _require(
        all(isinstance(value, str) and value for value in record_ids),
        f"{split} invalid record id",
    )
    _require(payload.get("uses_family_targets") is False, f"{split} family target leak")

    route = payload.get("route_schedule")
    _require(isinstance(route, torch.Tensor), f"{split} route schedule missing")
    _require(tuple(route.shape) == (count, 2, 8), f"{split} route shape drift")
    _require(route.dtype == torch.int64, f"{split} route dtype drift")
    _require(bool(torch.all((route == 0) | (route == 1))), f"{split} invalid route")
    flattened_route = route.reshape(count, -1)
    record_constant = bool(
        torch.all(flattened_route == flattened_route[:, :1]).item()
    )
    _require(record_constant, f"{split} route is no longer record-constant")
    route_counts = torch.bincount(flattened_route[:, 0], minlength=2)
    _require(
        route_counts.tolist() == [count // 2, count // 2],
        f"{split} binary route balance drift",
    )
    _require(
        int(route_counts.min()) >= SPEC.minimum_route_cell_records,
        f"{split} route coverage below frozen minimum",
    )

    metadata = payload.get("site_metadata")
    _require(isinstance(metadata, (tuple, list)), f"{split} site metadata missing")
    _require(tuple(metadata) == EXPECTED_SITES, f"{split} site order drift")
    for field in ("common", "projection", "transfer", "vjp"):
        tensors = payload.get(field)
        _require(
            isinstance(tensors, (tuple, list)) and len(tensors) == 16,
            f"{split} {field} site count drift",
        )
        for site_index, tensor in enumerate(tensors):
            _require(isinstance(tensor, torch.Tensor), f"{split} {field} missing tensor")
            _require(
                tuple(tensor.shape) == (count, 8, 256),
                f"{split} {field}[{site_index}] shape drift",
            )
            _require(
                tensor.dtype == torch.float32,
                f"{split} {field}[{site_index}] dtype drift",
            )
            _require(
                bool(torch.isfinite(tensor).all()),
                f"{split} {field}[{site_index}] non-finite",
            )
    for field in ("requested_margin", "positive_common_margin_drop"):
        tensor = payload.get(field)
        _require(
            isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == (count,),
            f"{split} {field} shape drift",
        )
        _require(bool(torch.isfinite(tensor).all()), f"{split} {field} non-finite")
    return {
        "records": count,
        "first_record_id": record_ids[0],
        "last_record_id": record_ids[-1],
        "record_order_sha256": sha256_json(list(record_ids)),
        "route_counts": {"0": int(route_counts[0]), "1": int(route_counts[1])},
        "route_is_record_constant": record_constant,
        "sites": 16,
    }


@dataclass(frozen=True)
class ReadOnlyInputs:
    checkpoint: Path
    cache_root: Path
    target_path: Path
    target_payload: Mapping[str, Any]
    cache: TokenCache
    audit: Mapping[str, Any]


def load_and_validate_inputs(repo_root: Path) -> ReadOnlyInputs:
    """Hash, mmap-load, and validate all immutable inputs before output creation."""

    repo_root = Path(repo_root).resolve()
    checkpoint = repo_root / SOURCE_CHECKPOINT
    cache_root = repo_root / SOURCE_TOKEN_CACHE
    target_path = repo_root / CAUSAL_TARGET_BANK
    target_manifest_path = repo_root / CAUSAL_TARGET_BANK_MANIFEST
    cache_manifest_path = repo_root / SOURCE_CACHE_MANIFEST
    _require(checkpoint.is_file(), "source checkpoint missing")
    _require(cache_root.is_dir(), "source token cache missing")
    _require(target_path.is_file(), "causal target bank missing")
    _require(target_manifest_path.is_file(), "causal target manifest missing")
    _require(cache_manifest_path.is_file(), "source cache manifest missing")

    checkpoint_hash = sha256_file(checkpoint)
    _require(checkpoint_hash == SOURCE_CHECKPOINT_SHA256, "source checkpoint hash drift")
    target_hash = sha256_file(target_path)
    _require(target_hash == CAUSAL_TARGET_BANK_SHA256, "causal target bank hash drift")
    target_manifest_hash = sha256_file(target_manifest_path)
    _require(
        target_manifest_hash == CAUSAL_TARGET_BANK_MANIFEST_SHA256,
        "causal target manifest hash drift",
    )
    cache_manifest_hash = sha256_file(cache_manifest_path)
    _require(
        cache_manifest_hash == SOURCE_CACHE_MANIFEST_SHA256,
        "source cache manifest hash drift",
    )

    # mmap keeps the 2.685 GB bank read-only and avoids a second resident copy.
    payload = torch.load(
        target_path, map_location="cpu", weights_only=False, mmap=True
    )
    _require(isinstance(payload, Mapping), "target bank payload is not a mapping")
    _require(payload.get("schema_version") == TARGET_SCHEMA, "target schema drift")
    _require(
        payload.get("source_checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256,
        "target/source checkpoint identity mismatch",
    )
    _require(payload.get("uses_family_targets") is False, "target bank family leak")
    datasets = payload.get("datasets")
    _require(isinstance(datasets, Mapping), "target datasets missing")
    _require(set(datasets) == set(EXPECTED_SPLIT_COUNTS), "target split identity drift")
    split_audit = {
        split: _validate_split(split, datasets[split])
        for split in ("train", "heldout")
    }
    train_ids = set(datasets["train"]["record_ids"])
    heldout_ids = set(datasets["heldout"]["record_ids"])
    _require(not train_ids.intersection(heldout_ids), "train/heldout record overlap")

    # TokenCache validates the manifest, component hashes, mmap shapes and the
    # absence of forbidden label fields.  indices() establishes record linkage
    # without regenerating packages or targets.
    cache = TokenCache(cache_root)
    cache.indices(datasets["train"]["record_ids"])
    cache.indices(datasets["heldout"]["record_ids"])
    audit = {
        "checkpoint_sha256": checkpoint_hash,
        "target_bank_sha256": target_hash,
        "target_schema": TARGET_SCHEMA,
        "target_mmap": True,
        "cache_schema": cache.manifest.get("schema_version"),
        "target_bank_manifest_sha256": target_manifest_hash,
        "cache_manifest_sha256": cache_manifest_hash,
        "splits": split_audit,
        "train_heldout_disjoint": True,
        "heldout_iid_assumed": False,
        "record_id_gap": {
            "known_gap_records": 1536,
            "used_for_fit": False,
            "used_for_score": False,
            "interpretation": "ordered split; family/site/time OOD not established by this bank",
        },
        "family_or_class_labels_present": False,
        "family_group_ood_measured": False,
    }
    return ReadOnlyInputs(
        checkpoint=checkpoint,
        cache_root=cache_root,
        target_path=target_path,
        target_payload=payload,
        cache=cache,
        audit=audit,
    )


__all__ = [
    "EXPECTED_SITES",
    "ReadOnlyInputs",
    "load_and_validate_inputs",
    "sha256_file",
    "sha256_json",
]
