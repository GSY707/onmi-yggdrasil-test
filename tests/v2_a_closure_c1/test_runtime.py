from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1.data import CT1_GRAMMAR_IDS, source_byte_sha256
from yggdrasil_v2.v2_a.closure_c1.runtime import (
    C1BatchProvider,
    TraceLexicon,
    TraceTargetBank,
    audit_target_bank,
    build_trace_lexicon,
    load_offline_records,
    load_target_bank,
    materialize_target_bank,
    save_target_bank,
    validate_cache_source_identity,
)


class TinyTokenizer:
    ids = {"CT1": CT1_GRAMMAR_IDS[0], '{"f":"ERE","t":[]}': CT1_GRAMMAR_IDS[1], "Answer:": CT1_GRAMMAR_IDS[2], "A": CT1_GRAMMAR_IDS[3], "source": 100, "one": 101}

    def __call__(self, text, **kwargs):
        words = text.split()
        values = [self.ids.get(word, 1000 + index) for index, word in enumerate(words)]
        if kwargs.get("return_offsets_mapping"):
            offsets, cursor = [], 0
            for word in words:
                start = text.find(word, cursor)
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
            return {"input_ids": values, "offset_mapping": offsets}
        return {"input_ids": values}


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.source_hashes = {key: source_byte_sha256(value["source_text"]) for key, value in rows.items()}

    def get_batch(self, example_ids, *, device="cpu"):
        lengths = [self.rows[str(key)]["tokens"] for key in example_ids]
        width, maximum = 2, max(lengths)
        hidden = torch.zeros((len(lengths), maximum, width), dtype=torch.float16)
        mask = torch.zeros((len(lengths), maximum), dtype=torch.bool)
        for row, length in enumerate(lengths):
            hidden[row, :length] = row + 1
            mask[row, :length] = True
        return {"source_hidden": hidden, "source_mask": mask}, [{"example_id": str(key)} for key in example_ids]


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps({"files": {"ere-train.jsonl": "unused"}}), encoding="utf-8")
    row = {"example_id": "e0", "family": "ERE", "split": "train", "source_text": "source one", "answer_index": 2, "compact_trace": 'CT1 {"f":"ERE","t":[]}\nAnswer: A'}
    (dataset / "ere-train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


def test_full_offline_store_and_trace_bank_roundtrip(tmp_path: Path):
    store = load_offline_records(_dataset(tmp_path))
    assert list(store.records_by_cell) == ["ERE/train"]
    assert store.answer_index("e0") == 2
    lexicon = build_trace_lexicon(store.records, TinyTokenizer())
    bank = materialize_target_bank(store.records, TinyTokenizer(), lexicon)
    assert bank.targets["e0"]["token_ids"] == [lexicon.local_id(token) for token in bank.targets["e0"]["global_token_ids"]]
    assert bank.targets["e0"]["grammar_mask"] == [True, True, True, True]
    path = tmp_path / "targets.json"
    digest = save_target_bank(bank, path)
    loaded = load_target_bank(path)
    assert digest == loaded.digest
    assert audit_target_bank(path, lexicon)["passed"]


def test_target_bank_audit_materializes_vocab_mapping_once(monkeypatch):
    lexicon = TraceLexicon(tuple(sorted(set(CT1_GRAMMAR_IDS) | {100, 101})))
    global_ids = [100, 101, 100, 101]
    mapping = lexicon.id_to_local
    bank = TraceTargetBank(
        lexicon,
        {
            "e0": {
                "token_ids": [mapping[value] for value in global_ids],
                "global_token_ids": global_ids,
                "step_indices": [1] * len(global_ids),
                "global_positions": list(range(len(global_ids))),
                "local_positions": list(range(len(global_ids))),
                "grammar_mask": [False] * len(global_ids),
            }
        },
    )
    getter = TraceLexicon.id_to_local.fget
    assert getter is not None
    calls = 0

    def counted(instance):
        nonlocal calls
        calls += 1
        return getter(instance)

    monkeypatch.setattr(TraceLexicon, "id_to_local", property(counted))
    assert audit_target_bank(bank)["passed"]
    assert calls == 1


def test_batch_provider_padding_and_epoch_chunk_determinism():
    records = {"e0": {"example_id": "e0", "source_text": "source", "answer_index": 1}, "e1": {"example_id": "e1", "source_text": "source", "answer_index": 4}}
    lexicon = TraceLexicon(tuple(sorted(set(CT1_GRAMMAR_IDS) | set(range(100, 180)))))
    targets = {}
    for key in records:
        ids = [CT1_GRAMMAR_IDS[0]] * 80
        targets[key] = {"example_id": key, "global_token_ids": ids, "token_ids": [lexicon.local_id(ids[0])] * 80, "step_indices": [1] * 80, "global_positions": list(range(80)), "local_positions": list(range(80)), "grammar_mask": [True] * 80}
    from yggdrasil_v2.v2_a.closure_c1.runtime import TraceTargetBank
    provider = C1BatchProvider(FakeDataset({key: {**row, "tokens": index + 2} for index, (key, row) in enumerate(records.items())}), records, TraceTargetBank(lexicon, targets))
    first = provider(["e0", "e1"], epoch=3, trace_chunk_tokens=64)
    second = provider(["e0", "e1"], epoch=3, trace_chunk_tokens=64)
    assert set(first) == {"source_hidden", "source_mask", "answers", "trace_targets", "trace_mask", "trace_step_indices", "trace_global_positions", "trace_local_positions"}
    assert torch.equal(first["trace_targets"], second["trace_targets"])
    assert first["trace_targets"].shape == (2, 64)
    assert first["source_hidden"].shape == (2, 3, 2)
    assert first["answers"].tolist() == [1, 4]
    full = provider(["e0", "e1"], epoch=3, trace_chunk_tokens=512)
    assert full["trace_targets"].shape == (2, 80)


def test_cache_identity_mismatch_fails_closed(tmp_path: Path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": "wrong", "shards": []}), encoding="utf-8")
    records = {"e0": {"example_id": "e0", "source_text": "source"}}
    from yggdrasil_v2.v2_a.closure_c1.runtime import OfflineRecordStore
    store = OfflineRecordStore({"ERE/train": tuple(records.values())}, records, str(tmp_path))
    report = validate_cache_source_identity(root, store)
    assert not report["passed"]
    with pytest.raises(ValueError):
        C1BatchProvider(FakeDataset({"e0": {"source_text": "changed", "tokens": 2}}), records, {"e0": {}})(["e0"], epoch=0)
