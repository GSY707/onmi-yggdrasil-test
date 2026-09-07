from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from yggdrasil_v2.v2_a.closure_c1 import cache_runner


def _records() -> list[dict[str, object]]:
    return [{"example_id": "e0", "family": "ERE", "split": "train", "source_text": "one"}]


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens=False, truncation=False):
        return {"input_ids": [1, 2, 3]}


class _Backbone(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=torch.ones((*input_ids.shape, 2048), dtype=torch.float16))


def _fake_trace(rows, tokenizer, *, formal):
    bank = SimpleNamespace(digest="B" * 64)
    lexicon = SimpleNamespace(digest="L" * 64)
    return ({"passed": True, "failures": [], "source_lexicon_size": 1, "trace_lexicon_size": 43, "grammar_ids": 42, "examples": len(rows), "target_oov": 0}, lexicon, bank)


def test_preflight_is_single_use_and_does_not_create_hidden_cache(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "preflight"
    lease = tmp_path / "preflight.lease.jsonl"
    monkeypatch.setattr(cache_runner, "_pinned_c0r", lambda _root: {"passed": True, "failures": []})
    monkeypatch.setattr(cache_runner, "_trace_audit", _fake_trace)
    monkeypatch.setattr(cache_runner, "_source_hashes", lambda _root: {"source": "A" * 64})
    monkeypatch.setattr(cache_runner, "_snapshot", lambda *_args: None)
    result = cache_runner.run_cache_preflight(
        tmp_path,
        root,
        lease,
        device="cpu",
        tokenizer=_Tokenizer(),
        backbone=_Backbone(),
        records=_records(),
    )
    assert result["passed"] is True
    assert not (root / "hidden").exists()
    assert (root / "evidence-seal.json").is_file()
    assert "evidence_seal_sha256" not in (root / "result.json").read_text(encoding="utf-8")
    refusal = cache_runner.run_cache_preflight(tmp_path, root, lease, records=_records())
    assert refusal["status"] == "REFUSE_V2_A_CLOSURE_C1_CACHE_SINGLE_USE"


def test_qualification_fake_cache_writes_hidden_and_replays(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    lease = tmp_path / "cache.lease.jsonl"
    hidden = root / "hidden"
    monkeypatch.setattr(cache_runner, "_trace_audit", _fake_trace)
    monkeypatch.setattr(cache_runner, "_pinned_c0r", lambda _root: {"passed": True, "failures": []})
    monkeypatch.setattr(cache_runner, "_source_hashes", lambda _root: {"source": "A" * 64})
    monkeypatch.setattr(cache_runner, "_snapshot", lambda *_args: None)

    def fake_build(*args, **kwargs):
        hidden.mkdir(parents=True, exist_ok=True)
        (hidden / "manifest.json").write_text(
            '{"hidden_width": 2048, "examples": 1, "full_token_hidden_saved": true, '
            '"dtype": "float16", "qwen_lm_head_present": false, "qwen_trainable_parameters": 0, '
            '"qwen_component": "Qwen3_5TextModel", "model_id": "Qwen/Qwen3.5-2B", '
            '"revision": "15852e8c16360a2fea060d615a32b45270f8a8fc", '
            '"transformers_version": "5.13.1", '
            '"source_tokens": 3, "shards": []}\n',
            encoding="utf-8",
        )
        return {}

    monkeypatch.setattr(cache_runner.cache, "build_cache", fake_build)
    monkeypatch.setattr(cache_runner.cache, "audit_cache", lambda *_args, **_kwargs: {"passed": True, "failures": []})
    monkeypatch.setattr(cache_runner, "_indexed_replay", lambda *_args, **_kwargs: {"passed": True, "failures": []})
    monkeypatch.setattr(cache_runner.runtime, "save_target_bank", lambda bank, path: path.write_text("{}\n", encoding="utf-8") or "T" * 64)
    monkeypatch.setattr(cache_runner.runtime, "audit_target_bank", lambda *_args, **_kwargs: {"passed": True, "failures": []})
    result = cache_runner.run_cache_qualification(
        tmp_path,
        root,
        lease,
        device="cpu",
        tokenizer=_Tokenizer(),
        backbone=_Backbone(),
        records=_records(),
        formal=False,
    )
    assert result["passed"] is False
    assert result["diagnostic_passed"] is True
    assert result["authorizes"] == "nothing"
    assert (root / "hidden").is_dir()
    assert (root / "evidence-seal.json").is_file()
    assert "evidence_seal_sha256" not in (root / "result.json").read_text(encoding="utf-8")


def test_formal_cache_refuses_before_mutation_without_preflight(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    lease = tmp_path / "cache.lease"
    monkeypatch.setattr(cache_runner, "_source_hashes", lambda _root: {"source": "A" * 64})
    monkeypatch.setattr(cache_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        cache_runner.artifacts,
        "audit_preflight_root",
        lambda *_args, **_kwargs: {"passed": False},
    )
    result = cache_runner.run_cache_qualification(
        tmp_path, root, lease, device="cuda", formal=True
    )
    assert result["status"] == "REFUSE_V2_A_CLOSURE_C1_CACHE_PREFLIGHT_PREREQUISITE"
    assert result["exit_code"] == 2
    assert not root.exists() and not lease.exists()


def test_formal_cache_refuses_cpu_before_mutation(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    lease = tmp_path / "cache.lease"
    result = cache_runner.run_cache_qualification(
        tmp_path, root, lease, device="cpu", formal=True
    )
    assert result["status"] == "REFUSE_V2_A_CLOSURE_C1_CACHE_FORMAL_DEVICE"
    assert not root.exists() and not lease.exists()


def test_formal_cache_refuses_injected_backbone_before_mutation(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    lease = tmp_path / "cache.lease"
    monkeypatch.setattr(cache_runner.torch.cuda, "is_available", lambda: True)
    result = cache_runner.run_cache_qualification(
        tmp_path,
        root,
        lease,
        device="cuda",
        tokenizer=_Tokenizer(),
        backbone=_Backbone(),
        records=_records(),
        formal=True,
    )
    assert result["status"] == "REFUSE_V2_A_CLOSURE_C1_CACHE_FORMAL_INJECTION"
    assert not root.exists() and not lease.exists()


def test_formal_cache_refuses_unavailable_cuda_before_mutation(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    lease = tmp_path / "cache.lease"
    monkeypatch.setattr(cache_runner.torch.cuda, "is_available", lambda: False)
    result = cache_runner.run_cache_qualification(
        tmp_path, root, lease, device="cuda", formal=True
    )
    assert result["status"] == "REFUSE_V2_A_CLOSURE_C1_CACHE_FORMAL_DEVICE"
    assert not root.exists() and not lease.exists()


def test_fixed_cache_preflight_refuses_unavailable_cuda_before_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cache_runner.torch.cuda, "is_available", lambda: False)
    result = cache_runner.run_cache_preflight(tmp_path)
    root = tmp_path / cache_runner.contract.CACHE_PREFLIGHT_ROOT
    lease = tmp_path / cache_runner.contract.CACHE_PREFLIGHT_LEASE
    assert result["status"] == "REFUSE_V2_A_CLOSURE_C1_CACHE_PREFLIGHT_DEVICE"
    assert not root.exists() and not lease.exists()
