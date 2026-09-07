from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from yggdrasil_v2.r1_revalidation.h1 import cache


class FakeTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 0
    all_special_ids = [101, 102]

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["return_offsets_mapping"] is True
        self.calls.append(text)
        return {
            "input_ids": [101, *range(1, len(text) + 1), 102],
            # Deliberately give specials non-zero offsets: the explicit mask
            # must still keep them out of atom membership.
            "offset_mapping": [(0, 1), *[(i, i + 1) for i in range(len(text))], (0, 1)],
            "special_tokens_mask": [1, *([0] * len(text)), 1],
        }


class FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=cache.HIDDEN_WIDTH)

    def forward(self, *, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        batch, length = input_ids.shape
        values = torch.arange(
            batch * length * cache.HIDDEN_WIDTH,
            dtype=torch.float32,
            device=input_ids.device,
        )
        return SimpleNamespace(
            last_hidden_state=values.reshape(batch, length, cache.HIDDEN_WIDTH).to(torch.float16)
        )


class FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeDecoder()
        self.config = self.model.config
        self.anchor = nn.Parameter(torch.zeros(1))


def _records():
    return [
        {"id": "r0", "source_text": "alpha\nbeta", "source_atoms": ["alpha", "beta"]},
        {"id": "r1", "source_text": "gamma\ndelta", "source_atoms": ["gamma", "delta"]},
    ]


def _build(monkeypatch, tmp_path):
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(cache, "load_qwen35_text_only", lambda *args, **kwargs: FakeQwen())
    root = tmp_path / "cache"
    manifest = cache.build_token_cache(_records(), root, device="cpu", batch_size=2)
    return root, manifest, tokenizer


def test_atom_membership_excludes_special_offsets_and_maps_each_clause():
    tokenizer = FakeTokenizer()
    ids, memberships = cache._tokenize_with_atom_membership(tokenizer, "ab\ncd", ("ab", "cd"))
    assert ids[0] == 101 and ids[-1] == 102
    assert memberships == [[1, 2], [4, 5]]


def test_build_audit_and_loader_are_fp16_2048_and_encode_full_source_once(monkeypatch, tmp_path):
    root, manifest, tokenizer = _build(monkeypatch, tmp_path)
    assert tokenizer.calls == ["alpha\nbeta", "gamma\ndelta"]
    hidden = np.load(root / manifest["hidden_file"], mmap_mode="r")
    mask = np.load(root / manifest["mask_file"], mmap_mode="r")
    assert hidden.shape == (2, 13, cache.HIDDEN_WIDTH)
    assert hidden.dtype == np.float16
    assert mask.shape == (2, 13)
    assert mask.dtype == np.bool_
    result = cache.audit_token_cache(_records(), root)
    assert result["passed"] is True
    loaded = cache.TokenCache(root)
    values, loaded_mask = loaded.batch(loaded.indices(["r1"]), device="cpu")
    assert values.shape == (1, 13, cache.HIDDEN_WIDTH)
    assert values.dtype == torch.float16
    assert loaded_mask.tolist() == [[True] * 13]
    with pytest.raises(KeyError, match="unknown"):
        loaded.indices(["missing"])
    with pytest.raises(IndexError):
        loaded.batch([-1], device="cpu")


def test_audit_detects_binary_tamper(monkeypatch, tmp_path):
    root, manifest, _ = _build(monkeypatch, tmp_path)
    array = np.lib.format.open_memmap(root / manifest["hidden_file"], mode="r+")
    array[0, 0, 0] = np.nan
    array.flush()
    result = cache.audit_token_cache(_records(), root)
    assert result["passed"] is False
    assert result["checks"]["hashes"] is False
    assert result["checks"]["finite"] is False


def test_audit_rejects_forbidden_manifest_or_index_key(monkeypatch, tmp_path):
    root, manifest, _ = _build(monkeypatch, tmp_path)
    index_path = root / manifest["index_file"]
    index = cache.json.loads(index_path.read_text(encoding="utf-8"))
    index["family"] = "leak"
    index_path.write_bytes(cache._canonical(index))
    result = cache.audit_token_cache(_records(), root)
    assert result["passed"] is False
    assert "family" in result["forbidden_field_hits"]


@pytest.mark.parametrize(
    "record",
    [
        {"id": "x", "source_text": "a\nb", "source_atoms": ["a", ""]},
        {"id": "x", "source_text": "a\na", "source_atoms": ["a", "a"]},
        {"id": "x", "source_text": "a\nb", "source_atoms": ["a", 1]},
        {"id": "x", "source_text": "a\nb", "source_atoms": ["a\nb", "b"]},
        {"id": "x", "source_text": "wrong", "source_atoms": ["a", "b"]},
    ],
)
def test_invalid_empty_duplicate_or_mismatched_atoms_are_rejected(record, tmp_path):
    with pytest.raises(ValueError):
        cache.build_token_cache([record], tmp_path / "cache", device="cpu")
