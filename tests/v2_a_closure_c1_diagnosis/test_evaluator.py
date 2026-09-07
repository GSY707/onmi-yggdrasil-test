from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from yggdrasil_v2.v2_a.closure_c1.model import C1Config, C1Model
from yggdrasil_v2.v2_a.closure_c1 import contract
from yggdrasil_v2.v2_a.closure_c1_diagnosis.evaluator import (
    DIAGNOSTIC_STATUS,
    load_selected_checkpoint_pin,
    run_poststop_diagnosis,
)


def _tiny() -> C1Config:
    return C1Config(
        source_width=8,
        latent_width=16,
        ffn_width=32,
        attention_heads=4,
        max_global_positions=16,
        max_local_positions=8,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _formal_root(tmp_path: Path, *, checkpoint_rel: str = "checkpoints/update-05120.pt") -> Path:
    root = tmp_path / "sealed-c1"
    checkpoint = root / checkpoint_rel
    checkpoint.parent.mkdir(parents=True)
    model = C1Model(_tiny(), with_trace_probe=True)
    torch.save(
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.training-checkpoint.v1",
            "identity": contract.C1_IDENTITY,
            "update": 5120,
            "model_state": model.state_dict(),
            "metrics": {"trace_nll_by_family": {"ERE": 1.0, "CPS": 1.0}},
        },
        checkpoint,
    )
    digest = _sha256(checkpoint)
    root.joinpath("result.json").write_text(
        json.dumps(
            {
                "status": "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                "stopped_at": "G007",
                "authorizes": "nothing",
                "c2_authorized": False,
                "selected_checkpoint_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    root.joinpath("primary-training.json").write_text(
        json.dumps(
            {
                "selected": {
                    "checkpoint": checkpoint_rel,
                    "checkpoint_sha256": digest,
                    "update": 5120,
                }
            }
        ),
        encoding="utf-8",
    )
    root.joinpath("contract-manifest.json").write_text(
        json.dumps({"model": dict(_tiny().__dict__)}), encoding="utf-8"
    )
    root.joinpath("evidence-seal.json").write_text(
        json.dumps(
            {
                "files": {
                    "result.json": _sha256(root / "result.json"),
                    checkpoint_rel: digest,
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_selected_checkpoint_pin_rejects_substitution(tmp_path: Path) -> None:
    root = _formal_root(tmp_path)
    pin = load_selected_checkpoint_pin(root)
    assert pin["update"] == 5120
    assert pin["authorizes"] == "nothing"
    with pytest.raises(ValueError, match="pinned"):
        load_selected_checkpoint_pin(root, root / "checkpoints" / "not-selected.pt")


def test_selected_checkpoint_pin_rejects_non_failure_or_authorization(tmp_path: Path) -> None:
    root = _formal_root(tmp_path)
    result_path = root / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "PASS_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                "stopped_at": None,
                "authorizes": "two additional fresh C1 seeds only",
                "c2_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="failure result"):
        load_selected_checkpoint_pin(root)


def test_diagnostic_only_boundary_and_json_native_output(tmp_path: Path) -> None:
    root = _formal_root(tmp_path)
    config = _tiny()
    source = {
        key: torch.randn(4, config.source_width)
        for key in ("ere-v", "ere-v2", "cps-v", "cps-v2", "ere-o", "cps-o", "ere-b", "ere-f", "cps-b", "cps-f")
    }

    class FakeDataset:
        def get_batch(self, example_ids, *, device="cpu"):
            rows = [source[key] for key in example_ids]
            maximum = max(row.shape[0] for row in rows)
            hidden = torch.zeros(len(rows), maximum, config.source_width)
            mask = torch.zeros(len(rows), maximum, dtype=torch.bool)
            for index, row in enumerate(rows):
                hidden[index, : row.shape[0]] = row
                mask[index, : row.shape[0]] = True
            return {"source_hidden": hidden.to(device), "source_mask": mask.to(device)}, []

    records = [
        {"example_id": "ere-v", "family": "ERE", "split": "validation", "target_label": "A"},
        {"example_id": "ere-v2", "family": "ERE", "split": "validation", "target_label": "A"},
        {"example_id": "cps-v", "family": "CPS", "split": "validation", "target_label": "B"},
        {"example_id": "cps-v2", "family": "CPS", "split": "validation", "target_label": "B"},
        {"example_id": "ere-o", "family": "ERE", "split": "composition_ood", "target_label": "A"},
        {"example_id": "cps-o", "family": "CPS", "split": "composition_ood", "target_label": "B"},
        {"example_id": "ere-b", "family": "ERE", "split": "causal_pairs", "pair_id": "ere-p", "pair_role": "base", "target_label": "A"},
        {"example_id": "ere-f", "family": "ERE", "split": "causal_pairs", "pair_id": "ere-p", "pair_role": "flip", "target_label": "B"},
        {"example_id": "cps-b", "family": "CPS", "split": "causal_pairs", "pair_id": "cps-p", "pair_role": "base", "target_label": "A"},
        {"example_id": "cps-f", "family": "CPS", "split": "causal_pairs", "pair_id": "cps-p", "pair_role": "flip", "target_label": "B"},
    ]
    report = run_poststop_diagnosis(
        formal_root=root,
        dataset=FakeDataset(),
        records=records,
        device="cpu",
        batch_size=2,
    )
    assert report["status"] == DIAGNOSTIC_STATUS
    assert report["formal"] is False
    assert report["qualification"] == "not_qualified"
    assert report["authorizes"] == "nothing"
    assert report["c2_authorized"] is False
    assert report["measurements"]["integrity"]["deployment"]["stripped_probe_parameters"] == 0
    json.dumps(report, allow_nan=False)
