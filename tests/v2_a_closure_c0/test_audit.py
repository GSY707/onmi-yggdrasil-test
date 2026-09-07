from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yggdrasil_v2.v2_a.closure_c0.audit import scan_dataset, verify_evidence_seal
from yggdrasil_v2.v2_a.closure_c0.contract import DATASET_PROFILE


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_evidence_seal_checks_exact_tree_and_hashes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_json(root / "result.json", {"passed": True})
    _write_json(
        root / "evidence-seal.json",
        {"schema_version": "test", "files": {"result.json": _sha(root / "result.json")}},
    )
    seal_hash = _sha(root / "evidence-seal.json")
    assert verify_evidence_seal(root, expected_seal_sha256=seal_hash)["passed"] is True
    _write_json(root / "result.json", {"passed": False})
    audit = verify_evidence_seal(root, expected_seal_sha256=seal_hash)
    assert audit["passed"] is False
    assert "sealed_file_hash_mismatch:result.json" in audit["failures"]


def test_dataset_scan_accepts_distinct_balanced_minimal_profile(tmp_path: Path, monkeypatch) -> None:
    profile = {
        "ERE": {"train": 1, "causal_pairs": 2},
        "CPS": {"train": 1, "causal_pairs": 2},
    }
    monkeypatch.setattr(
        "yggdrasil_v2.v2_a.closure_c0.audit.DATASET_PROFILE", profile
    )
    root = tmp_path / "dataset"
    counter = 0
    for family, splits in profile.items():
        for split, count in splits.items():
            rows = []
            for index in range(count):
                counter += 1
                role = ("base", "flip")[index] if split == "causal_pairs" else None
                rows.append(
                    {
                        "example_id": f"{family}-{split}-{index}",
                        "family": family,
                        "split": split,
                        "source_text": "executable task",
                        "program_ast": (
                            {key: [] for key in ("events", "initial_state", "query", "rules")}
                            if family == "ERE"
                            else {
                                key: []
                                for key in (
                                    "actions",
                                    "budget",
                                    "candidates",
                                    "final_constraints",
                                    "goal",
                                    "initial_state",
                                )
                            }
                        ),
                        "answer_index": index % 2,
                        "valid_choice_mask": [True, True],
                        "teacher_trace": [{}],
                        "training_claims": [{"label": True}, {"label": False}],
                        "semantic_fingerprint": f"{counter:064X}",
                        "surface_fingerprint": f"{counter + 100:064X}",
                        "pair_id": f"{family}-pair" if split == "causal_pairs" else None,
                        "pair_role": role,
                        "semantic_answer": f"answer-{index}",
                    }
                )
            path = root / f"{family.lower()}-{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
    audit = scan_dataset(root)
    assert audit["passed"] is True
    assert audit["ast_families_distinct"] is True
    assert audit["semantic_fingerprint_overlap_count"] == 0


def test_dataset_scan_fails_closed_on_missing_profile(tmp_path: Path) -> None:
    audit = scan_dataset(tmp_path)
    assert audit["available"] is False
    assert len(audit["missing"]) == sum(len(splits) for splits in DATASET_PROFILE.values())

