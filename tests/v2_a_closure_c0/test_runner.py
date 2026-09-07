from __future__ import annotations

import json
from pathlib import Path

from yggdrasil_v2.v2_a.closure_c0 import runner


def test_runner_seals_result_and_refuses_second_run(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "c0"
    lease = tmp_path / "c0.lease.jsonl"
    audit = {"availability_complete": True, "gates": {f"C00{i}": True for i in range(1, 9)}}
    reuse = {"schema_version": "test", "passed": True, "entries": []}
    monkeypatch.setattr(runner, "audit_closure_c0", lambda *_: (audit, reuse))

    result = runner.run_closure_c0(
        repo_root,
        output_root=output,
        lease_path=lease,
        archive_root=tmp_path / "archive",
    )
    assert result["status"] == "PASS_V2_A_CLOSURE_C0_READINESS"
    assert result["training_started"] is False
    assert (output / "evidence-seal.json").is_file()
    seal = json.loads((output / "evidence-seal.json").read_text(encoding="utf-8"))
    assert "result.json" in seal["files"]
    assert all(not name.endswith((".pt", ".bin", ".safetensors")) for name in seal["files"])

    refusal = runner.run_closure_c0(
        repo_root,
        output_root=output,
        lease_path=lease,
        archive_root=tmp_path / "archive",
    )
    assert refusal["status"] == "REFUSE_V2_A_CLOSURE_C0_SINGLE_USE"
    assert refusal["exit_code"] == 2

