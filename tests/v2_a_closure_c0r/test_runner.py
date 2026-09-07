from __future__ import annotations

import json
from pathlib import Path

from yggdrasil_v2.v2_a.closure_c0r import runner


def test_data_runner_seals_and_refuses_second_attempt(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "data"
    lease = tmp_path / "data.lease.jsonl"

    def fake_generate(root: Path, **_: object) -> None:
        root.mkdir(parents=True)
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(runner.data_generator, "generate", fake_generate)
    monkeypatch.setattr(
        runner.data_generator,
        "load",
        lambda _root: {"ere-train": [{"example_id": "e", "family": "ERE", "split": "train"}]},
    )
    monkeypatch.setattr(
        runner.data_audit,
        "audit_successor_dataset",
        lambda *_args, **_kwargs: {
            "gates": {f"D{index:03d}": True for index in range(1, 12)},
            "passed": True,
        },
    )
    monkeypatch.setattr(
        runner.compact_trace,
        "qualify_full_bank",
        lambda *_args, **_kwargs: {"passed": True, "records": 1},
    )
    result = runner.run_data_trace_qualification(
        repo_root,
        output_root=output,
        lease_path=lease,
        formal=False,
        tokenizer=object(),
    )
    assert result["status"] == "PASS_V2_A_C0R_DATA_TRACE_QUALIFICATION"
    assert result["training_started"] is False
    seal = json.loads((output / "evidence-seal.json").read_text(encoding="utf-8"))
    assert "result.json" in seal["files"]
    assert all(not name.endswith((".pt", ".bin", ".safetensors")) for name in seal["files"])
    refusal = runner.run_data_trace_qualification(
        repo_root,
        output_root=output,
        lease_path=lease,
        formal=False,
        tokenizer=object(),
    )
    assert refusal["status"] == "REFUSE_V2_A_C0R_DATA_TRACE_SINGLE_USE"
    assert refusal["exit_code"] == 2


def test_readiness_runner_is_read_only_and_single_use(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "readiness"
    lease = tmp_path / "readiness.lease.jsonl"
    gates = {f"C00{index}": True for index in range(1, 9)}
    monkeypatch.setattr(
        runner.readiness,
        "audit_readiness",
        lambda _repo: {
            "availability_complete": True,
            "passed": True,
            "gates": gates,
            "interpretation": "qualification only",
        },
    )
    result = runner.run_readiness(repo_root, output_root=output, lease_path=lease)
    assert result["status"] == "PASS_V2_A_CLOSURE_C0R_READINESS"
    assert result["training_started"] is False
    assert result["optimizer_steps"] == 0
    assert result["model_writes"] == 0
    refusal = runner.run_readiness(repo_root, output_root=output, lease_path=lease)
    assert refusal["status"] == "REFUSE_V2_A_CLOSURE_C0R_SINGLE_USE"

