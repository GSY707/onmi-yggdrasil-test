from __future__ import annotations

import json
from pathlib import Path

import torch

from yggdrasil_v2.v2_a.closure_c1.cache import CachedShardDataset, audit_cache, build_cache
from yggdrasil_v2.v2_a.closure_c1.data import CT1_GRAMMAR_IDS, load_records, public_source_view
from yggdrasil_v2.v2_a.closure_c1.trace_targets import align_trace_tokens


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **kwargs):
        words = text.split()
        ids = [((ord(word[0]) if word else 1) % 31) + 1 for word in words]
        if kwargs.get("return_offsets_mapping"):
            offsets = []
            cursor = 0
            for word in words:
                start = text.find(word, cursor)
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
            return {"input_ids": ids, "offset_mapping": offsets}
        return {"input_ids": ids}


class FakeBackbone(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache=False):
        class Output:
            pass

        output = Output()
        output.last_hidden_state = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return output

    # The production manifest records the native class name.  A fake cache is
    # allowed to retain its own identity; the strict production-only check is
    # exercised by runner qualification, not this serialization unit test.


def _records():
    return [
        {"example_id": "ere-0", "family": "ERE", "split": "train", "source_text": "alpha beta"},
        {"example_id": "cps-0", "family": "CPS", "split": "train", "source_text": "gamma delta epsilon"},
    ]


def test_public_loader_is_source_only(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({"files": {"ere-train.jsonl": "x"}}), encoding="utf-8")
    (dataset / "ere-train.jsonl").write_text(json.dumps({"example_id": "x", "family": "ERE", "source_text": "a"}) + "\n", encoding="utf-8")
    rows = load_records(tmp_path, cells=["ERE/train"])
    assert rows[0].as_dict() == {"example_id": "x", "family": "ERE", "split": "train", "source_text": "a"}
    assert public_source_view(rows[0]) == {"source_text": "a"}


def test_trace_offset_alignment_uses_midpoint_and_terminal_step():
    text = 'CT1 {"f":"ERE","t":[[0,"r",[],[],1]]}\nAnswer: A'
    rows = align_trace_tokens(text, FakeTokenizer())
    assert rows
    assert all(1 <= row.step <= 10 for row in rows)
    assert rows[-1].step == 10


def test_cache_is_fp16_full_token_and_streams_in_order(tmp_path: Path):
    output = tmp_path / "cache"
    records = _records()
    manifest = build_cache(tmp_path, tmp_path, output, records=records, tokenizer=FakeTokenizer(), backbone=FakeBackbone(), shard_size=1)
    assert manifest["identity"].startswith("V2-A-CLOSURE-C1-CACHE")
    assert audit_cache(output, records)["passed"]
    shard = torch.load(output / "shard_00000.pt", map_location="cpu", weights_only=True)
    assert shard["packed_shape"][1] == 4
    assert (output / shard["packed_hidden"]).stat().st_size % 2 == 0
    assert "input_ids" not in shard and "family" not in shard
    batches = list(CachedShardDataset(output).iter_batches(2))
    assert [item["example_id"] for item in batches[0][1]] == ["cps-0", "ere-0"]
    assert batches[0][0]["source_mask"].tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    random_batch, random_metadata = CachedShardDataset(output).get_batch(
        ["ere-0", "cps-0", "ere-0"]
    )
    assert [item["example_id"] for item in random_metadata] == [
        "ere-0",
        "cps-0",
        "ere-0",
    ]
    assert random_batch["source_hidden"].shape == (3, 3, 4)
