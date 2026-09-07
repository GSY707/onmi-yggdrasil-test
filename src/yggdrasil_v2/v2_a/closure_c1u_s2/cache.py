from __future__ import annotations

"""Target-free persistent independent-card cache for all six S2 banks."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from yggdrasil_v2.v2_a.closure_c1u.artifacts import read_json, sha256_file, write_json
from yggdrasil_v2.v2_a.closure_c1u.cache import FORBIDDEN_PERSISTED_FIELDS
from yggdrasil_v2.v2_a.closure_c1u.runtime import (
    EncodedCard,
    IndependentCardCache,
    audit_independent_card_cache,
)

from . import contract


def _id_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [str(row["example_id"]) for row in records],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def persist_cache(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    cache: IndependentCardCache,
) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"S2 card cache already exists: {root}")
    audit = audit_independent_card_cache(records, cache)
    if not audit["passed"]:
        raise ValueError(f"refusing invalid S2 independent cache: {audit}")
    root.mkdir(parents=True, exist_ok=False)
    cards = []
    for (example_id, kind, index), card in sorted(cache.cards.items()):
        cards.append(
            {
                "example_id": example_id,
                "card_kind": kind,
                "card_index": index,
                "hidden": card.hidden.detach().cpu().contiguous(),
                "mask": card.mask.detach().cpu().contiguous(),
                "token_ids": card.token_ids.detach().cpu().to(torch.int64).contiguous(),
                "text_sha256": card.text_sha256,
                "token_sha256": card.token_sha256,
                "hidden_sha256": card.hidden_sha256,
            }
        )
    payload = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.card-tensors.v1",
        "identity": contract.PREFLIGHT_IDENTITY,
        "source_identity": dict(cache.source_identity),
        "source_width": cache.source_width,
        "cards": cards,
    }
    if set(payload) & FORBIDDEN_PERSISTED_FIELDS:
        raise RuntimeError("target field entered S2 cache payload")
    torch.save(payload, root / "cards.pt")
    write_json(
        root / "ledger.json",
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.card-ledger.v1",
            "identity": contract.PREFLIGHT_IDENTITY,
            "rows": list(cache.ledger),
        },
    )
    manifest = {
        **cache.manifest(),
        "schema_version": f"{contract.SCHEMA_PREFIX}.card-cache.v1",
        "stage_identity": contract.PREFLIGHT_IDENTITY,
        "record_count": len(records),
        "example_id_sha256": _id_sha256(records),
        "contains_targets": False,
        "forbidden_persisted_fields": sorted(FORBIDDEN_PERSISTED_FIELDS),
        "files": {
            "cards.pt": sha256_file(root / "cards.pt"),
            "ledger.json": sha256_file(root / "ledger.json"),
        },
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def load_cache(root: Path) -> IndependentCardCache:
    root = Path(root)
    payload = torch.load(root / "cards.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("S2 cache tensor payload must be a mapping")
    if payload.get("schema_version") != f"{contract.SCHEMA_PREFIX}.card-tensors.v1":
        raise ValueError("S2 cache tensor schema mismatch")
    if payload.get("identity") != contract.PREFLIGHT_IDENTITY:
        raise ValueError("S2 cache tensor identity mismatch")
    if set(payload) & FORBIDDEN_PERSISTED_FIELDS:
        raise ValueError("S2 cache payload contains target fields")
    cards = {}
    for row in payload.get("cards", []):
        key = (str(row["example_id"]), str(row["card_kind"]), int(row["card_index"]))
        if key in cards:
            raise ValueError(f"duplicate S2 card key: {key}")
        cards[key] = EncodedCard(
            hidden=row["hidden"].detach().cpu().contiguous(),
            mask=row["mask"].detach().cpu().contiguous(),
            token_ids=row["token_ids"].detach().cpu().to(torch.int64).contiguous(),
            text_sha256=str(row["text_sha256"]),
            token_sha256=str(row["token_sha256"]),
            hidden_sha256=str(row["hidden_sha256"]),
        )
    ledger_payload = read_json(root / "ledger.json")
    if ledger_payload.get("identity") != contract.PREFLIGHT_IDENTITY:
        raise ValueError("S2 cache ledger identity mismatch")
    ledger = ledger_payload.get("rows")
    if not isinstance(ledger, list):
        raise TypeError("S2 cache ledger rows are missing")
    return IndependentCardCache(
        source_identity=dict(payload["source_identity"]),
        source_width=int(payload["source_width"]),
        cards=cards,
        ledger=tuple(dict(row) for row in ledger),
    )


def audit_cache(
    root: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    root = Path(root)
    try:
        manifest = read_json(root / "manifest.json")
        files = {
            name: {
                "expected": manifest.get("files", {}).get(name),
                "observed": sha256_file(root / name),
            }
            for name in ("cards.pt", "ledger.json")
        }
        cache = load_cache(root)
        runtime = audit_independent_card_cache(records, cache)
        checks = {
            "schema": manifest.get("schema_version")
            == f"{contract.SCHEMA_PREFIX}.card-cache.v1",
            "identity": manifest.get("stage_identity") == contract.PREFLIGHT_IDENTITY,
            "records": manifest.get("record_count") == len(records),
            "example_ids": manifest.get("example_id_sha256") == _id_sha256(records),
            "target_free": manifest.get("contains_targets") is False,
            "file_hashes": all(
                row["expected"] == row["observed"] for row in files.values()
            ),
            "runtime": runtime["passed"],
        }
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.card-cache-audit.v1",
            "passed": all(checks.values()),
            "checks": checks,
            "files": files,
            "manifest": manifest,
            "runtime": runtime,
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


__all__ = ["audit_cache", "load_cache", "persist_cache"]
