from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1t import contract
from yggdrasil_v2.v2_a.closure_c1t.cache import (
    audit_persisted_cache,
    load_independent_cache,
    persist_independent_cache,
)
from yggdrasil_v2.v2_a.closure_c1t.runtime import (
    audit_independent_card_cache,
    build_independent_card_cache,
)
from yggdrasil_v2.v2_a.closure_c1t.source import audit_source_assets
from yggdrasil_v2.v2_a.closure_c1t.tasks import build_s1_bank


def _encoder(text: str) -> dict[str, torch.Tensor]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tokens, width = 2 + len(text) % 3, 12
    hidden = torch.tensor(
        [[digest[(row + column) % 32] / 255.0 for column in range(width)] for row in range(tokens)]
    )
    return {
        "hidden": hidden,
        "mask": torch.ones(tokens, dtype=torch.bool),
        "token_ids": torch.tensor([digest[index] for index in range(tokens)]),
    }


def test_local_source_asset_audit_hashes_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"frozen")
    expected = {
        "asset.bin": {
            "bytes": asset.stat().st_size,
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest().upper(),
        }
    }
    monkeypatch.setattr(contract, "SOURCE_ASSETS", expected)

    def resolver(**_: object) -> str:
        return str(asset)

    assert audit_source_assets(resolver=resolver)["passed"] is True
    asset.write_bytes(b"drifted")
    report = audit_source_assets(resolver=resolver)
    assert report["passed"] is False
    assert report["rows"]["asset.bin"]["checks"]["sha256"] is False


def test_persistent_independent_cache_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    records = build_s1_bank()
    cache = build_independent_card_cache(
        records,
        _encoder,
        source_identity={"model": "fixture", "one_card_per_forward_call": True},
    )
    root = tmp_path / "cache"
    manifest = persist_independent_cache(root, records, cache)
    assert manifest["contains_targets"] is False
    assert audit_persisted_cache(root, records)["passed"] is True
    loaded = load_independent_cache(root)
    assert audit_independent_card_cache(records, loaded)["passed"] is True
    assert len(loaded.cards) == contract.S0_CARDS

    with (root / "cards.pt").open("ab") as handle:
        handle.write(b"tamper")
    report = audit_persisted_cache(root, records)
    assert report["passed"] is False
    assert report["checks"]["file_hashes"] is False


def test_persistent_cache_refuses_overwrite(tmp_path: Path) -> None:
    records = build_s1_bank()
    cache = build_independent_card_cache(
        records, _encoder, source_identity={"model": "fixture"}
    )
    root = tmp_path / "cache"
    persist_independent_cache(root, records, cache)
    with pytest.raises(FileExistsError):
        persist_independent_cache(root, records, cache)
