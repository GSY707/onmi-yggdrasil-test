from __future__ import annotations

import hashlib

import torch

from yggdrasil_v2.v2_a.closure_c1u.runtime import build_independent_card_cache
from yggdrasil_v2.v2_a.closure_c1u_s2.cache import (
    audit_cache,
    load_cache,
    persist_cache,
)
from yggdrasil_v2.v2_a.closure_c1u_s2.tasks import build_bank


def _encoder(text: str) -> dict[str, torch.Tensor]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tokens, width = 2, 12
    return {
        "hidden": torch.tensor(
            [[digest[(row + col) % 32] / 255.0 for col in range(width)] for row in range(tokens)]
        ),
        "mask": torch.ones(tokens, dtype=torch.bool),
        "token_ids": torch.tensor([digest[index] for index in range(tokens)]),
    }


def test_s2_cache_roundtrip_is_target_free_and_hash_audited(tmp_path) -> None:
    records = build_bank("B0")
    cache = build_independent_card_cache(records, _encoder, source_identity={"fixture": True})
    manifest = persist_cache(tmp_path / "cache", records, cache)
    assert manifest["contains_targets"] is False
    loaded = load_cache(tmp_path / "cache")
    assert len(loaded.cards) == 192
    audit = audit_cache(tmp_path / "cache", records)
    assert audit["passed"] is True
