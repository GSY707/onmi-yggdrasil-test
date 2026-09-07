from __future__ import annotations

import json
from pathlib import Path

import pytest

from yggdrasil_v2.v2_a.closure_c1_diagnosis.exposure import exposure_coverage_report
from yggdrasil_v2.v2_a.closure_c1_diagnosis.selection import selection_audit


def _bank() -> dict:
    def target(example_id: str, n: int, family: str) -> dict:
        return {
            "example_id": example_id,
            "family": family,
            "global_positions": list(range(n)),
            "global_token_ids": list(range(n)),
            "grammar_mask": [i % 2 == 0 for i in range(n)],
            "local_positions": list(range(n)),
            "step_indices": [1 if i < n // 2 else 10 for i in range(n)],
        }

    return {"targets": {"ere-train-0": target("ere-train-0", 4, "ERE"), "cps-train-0": target("cps-train-0", 5, "CPS")}}


def test_exposure_reconstructs_counts_and_cycle_breach() -> None:
    ledger = {
        "rows": [
            {"example_id": "ere-train-0", "epoch": 0, "start": 0, "stop": 3, "exposed_tokens": 3, "target_tokens": 4},
            {"example_id": "ere-train-0", "epoch": 1, "start": 2, "stop": 4, "exposed_tokens": 2, "target_tokens": 4},
            {"example_id": "cps-train-0", "epoch": 0, "start": 0, "stop": 2, "exposed_tokens": 2, "target_tokens": 5},
        ]
    }
    report = exposure_coverage_report(_bank(), ledger)
    assert report["complete_cycle"] is False
    assert report["EXPOSURE_CYCLE_BREACH"] is True
    assert report["by_family"]["ERE"]["micro_unique_coverage"] == 1.0
    assert report["by_family"]["CPS"]["zero_exposure_token_count"] == 3
    json.dumps(report, allow_nan=False)


def test_exposure_rejects_window_mismatch() -> None:
    with pytest.raises(ValueError, match="exposed_tokens"):
        exposure_coverage_report(
            _bank(),
            {"rows": [{"example_id": "ere-train-0", "epoch": 0, "start": 0, "stop": 2, "exposed_tokens": 1, "target_tokens": 4}]},
        )


def test_selection_replays_minmax_and_optional_hashes(tmp_path: Path) -> None:
    candidates = []
    for index, update in enumerate(range(512, 6145, 512)):
        path = tmp_path / f"update-{update}.pt"
        path.write_bytes(f"checkpoint-{update}".encode())
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        candidates.append({
            "checkpoint": path.name,
            "checkpoint_sha256": digest,
            "trace_nll_by_family": {"ERE": float(index), "CPS": float(18 - index)},
            "update": update,
        })
    primary = {
        "candidates": candidates,
        "selected": candidates[9],
        "selection_rule": "minimize max(ERE trace NLL, CPS trace NLL); tie earliest",
    }
    report = selection_audit(primary, tmp_path)
    assert report["passed"] is True
    assert report["selected_update"] == 5120
    assert report["checks"]["checkpoint_hashes"] is True
    assert selection_audit(primary)["passed"] is True
