from __future__ import annotations

"""Persistent, target-free serialization for the C1T independent-card cache."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import contract
from .artifacts import read_json, sha256_file, write_json
from .runtime import (
    EncodedCard,
    IndependentCardCache,
    audit_independent_card_cache,
)


FORBIDDEN_PERSISTED_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "ast",
        "counterfactual_indices",
        "factor_cells",
        "factorial_group_ids",
        "families",
        "family",
        "label_mapping",
        "semantic_answer",
        "support_slots",
        "task_causal_arity",
        "valid_choice_mask",
    }
)


def _id_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    ids = [str(row["example_id"]) for row in records]
    payload = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def persist_independent_cache(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    cache: IndependentCardCache,
) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"independent-card cache root already exists: {root}")
    audit = audit_independent_card_cache(records, cache)
    if not audit["passed"]:
        raise ValueError(f"refusing to persist an invalid independent cache: {audit}")
    root.mkdir(parents=True, exist_ok=False)
    card_rows = []
    for (example_id, kind, index), card in sorted(cache.cards.items()):
        card_rows.append(
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
        "schema_version": f"{contract.SCHEMA_PREFIX}.persistent-card-tensors.v1",
        "identity": contract.S0_IDENTITY,
        "source_identity": dict(cache.source_identity),
        "source_width": cache.source_width,
        "cards": card_rows,
    }
    if set(payload) & FORBIDDEN_PERSISTED_FIELDS:
        raise RuntimeError("forbidden target field entered persistent card payload")
    torch.save(payload, root / "cards.pt")
    write_json(
        root / "ledger.json",
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.persistent-card-ledger.v1",
            "identity": contract.S0_IDENTITY,
            "rows": list(cache.ledger),
        },
    )
    manifest = {
        **cache.manifest(),
        "schema_version": f"{contract.SCHEMA_PREFIX}.persistent-card-cache.v1",
        "stage_identity": contract.S0_IDENTITY,
        "record_count": len(records),
        "example_id_sha256": _id_sha256(records),
        "files": {
            "cards.pt": sha256_file(root / "cards.pt"),
            "ledger.json": sha256_file(root / "ledger.json"),
        },
        "forbidden_persisted_fields": sorted(FORBIDDEN_PERSISTED_FIELDS),
        "contains_targets": False,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def load_independent_cache(root: Path) -> IndependentCardCache:
    root = Path(root)
    payload = torch.load(root / "cards.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("persistent card tensor payload must be a mapping")
    expected_schema = f"{contract.SCHEMA_PREFIX}.persistent-card-tensors.v1"
    if payload.get("schema_version") != expected_schema:
        raise ValueError("persistent card tensor schema mismatch")
    if payload.get("identity") != contract.S0_IDENTITY:
        raise ValueError("persistent card tensor identity mismatch")
    if set(payload) & FORBIDDEN_PERSISTED_FIELDS:
        raise ValueError("persistent card payload contains a forbidden target field")
    rows = payload.get("cards")
    if not isinstance(rows, list) or not rows:
        raise ValueError("persistent card payload has no cards")
    cards = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("persistent card row must be a mapping")
        if set(row) & FORBIDDEN_PERSISTED_FIELDS:
            raise ValueError("persistent card row contains a forbidden target field")
        key = (str(row["example_id"]), str(row["card_kind"]), int(row["card_index"]))
        if key in cards:
            raise ValueError(f"duplicate persistent card key: {key}")
        cards[key] = EncodedCard(
            hidden=row["hidden"].detach().cpu().contiguous(),
            mask=row["mask"].detach().cpu().contiguous(),
            token_ids=row["token_ids"].detach().cpu().to(torch.int64).contiguous(),
            text_sha256=str(row["text_sha256"]),
            token_sha256=str(row["token_sha256"]),
            hidden_sha256=str(row["hidden_sha256"]),
        )
    ledger_payload = read_json(root / "ledger.json")
    if ledger_payload.get("identity") != contract.S0_IDENTITY:
        raise ValueError("persistent card ledger identity mismatch")
    ledger = ledger_payload.get("rows")
    if not isinstance(ledger, list):
        raise ValueError("persistent card ledger rows are missing")
    return IndependentCardCache(
        source_identity=dict(payload["source_identity"]),
        source_width=int(payload["source_width"]),
        cards=cards,
        ledger=tuple(dict(row) for row in ledger),
    )


def audit_persisted_cache(
    root: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    root = Path(root)
    try:
        manifest = read_json(root / "manifest.json")
        file_rows = {}
        for name in ("cards.pt", "ledger.json"):
            observed = sha256_file(root / name)
            expected = str(manifest.get("files", {}).get(name, ""))
            file_rows[name] = {"passed": observed == expected, "sha256": observed}
        cache = load_independent_cache(root)
        cache_audit = audit_independent_card_cache(records, cache)
        checks = {
            "manifest_schema": manifest.get("schema_version")
            == f"{contract.SCHEMA_PREFIX}.persistent-card-cache.v1",
            "stage_identity": manifest.get("stage_identity") == contract.S0_IDENTITY,
            "record_count": manifest.get("record_count") == len(records),
            "example_ids": manifest.get("example_id_sha256") == _id_sha256(records),
            "contains_targets_false": manifest.get("contains_targets") is False,
            "file_hashes": all(row["passed"] for row in file_rows.values()),
            "runtime_audit": cache_audit["passed"],
        }
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.persistent-card-cache-audit.v1",
            "passed": all(checks.values()),
            "checks": checks,
            "files": file_rows,
            "runtime_audit": cache_audit,
            "manifest": manifest,
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "FORBIDDEN_PERSISTED_FIELDS",
    "audit_persisted_cache",
    "load_independent_cache",
    "persist_independent_cache",
]
