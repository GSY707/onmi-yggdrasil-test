from __future__ import annotations

from pathlib import Path

from yggdrasil_v2.v2_a.closure_c0r import data_audit, data_generator
from yggdrasil_v2.v2_a.closure_c0r.data_audit import relation_visible_pattern_gate


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def _row(index: int, answer: str, *, split: str = "train", pair_id: str | None = None, role: str | None = None) -> dict:
    mapping = {"A": "TRUE", "B": "FALSE"}
    return {
        "example_id": f"r-{index}",
        "family": "ERE",
        "split": split,
        "semantic_answer": answer,
        "answer_index": 0 if answer == "TRUE" else 1,
        "valid_choice_mask": [True, True],
        "label_mapping": mapping,
        "source_text": "Label A means TRUE. Label B means FALSE.",
        "program_ast": {"query": {"kind": "relation"}},
        "pair_id": pair_id,
        "pair_role": role,
    }


def test_relation_visible_pattern_rejects_dominant_answer_and_reports_hash() -> None:
    rows = [_row(i, "TRUE") for i in range(4)]
    result = relation_visible_pattern_gate(rows, min_support=4, min_class_support=2, expected_causal_pairs=0)
    assert result["passed"] is False
    cell = result["value"]["cells"]["ERE/train"]
    assert cell["answer_counts"] == {"TRUE": 4}
    assert len(cell["prediction_sha256"]) == 64


def test_relation_visible_pattern_accepts_balanced_cell_and_causal_pair() -> None:
    rows = [_row(i, "TRUE" if i % 2 == 0 else "FALSE") for i in range(4)]
    rows.extend(
        [
            _row(10, "TRUE", split="causal_pairs", pair_id="p0", role="base"),
            _row(11, "FALSE", split="causal_pairs", pair_id="p0", role="flip"),
        ]
    )
    result = relation_visible_pattern_gate(rows, min_support=4, min_class_support=2, max_excess=0.5, expected_causal_pairs=1)
    assert result["passed"] is True
    assert result["value"]["causal_pair_failures"] == []


def test_d002_distinguishes_manifest_schema_from_record_schema(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    spec = data_generator.DatasetSpec("small", 2026082401, 24, 24)
    root = tmp_path / "dataset"
    data_generator.generate(
        root,
        spec=spec,
        design_doc=repo_root / "docs/v2-a-closure-c0r-data-trace-qualification.md",
        tokenizer=_Tokenizer(),
    )
    monkeypatch.setattr(data_generator, "FORMAL_SPEC", spec)
    loaded = data_generator.load(root)
    records = [row for cell in sorted(loaded) for row in loaded[cell]]
    manifest = data_audit._load_json(root / "manifest.json")
    result = data_audit._d002(repo_root, root, records, manifest, data_generator)
    assert result["passed"], result["failures"]

    for key, replacement in (
        ("schema_version", data_generator.SCHEMA_VERSION),
        ("model_id", "wrong/model"),
        ("model_revision", "wrong-revision"),
        ("add_special_tokens", 0),
    ):
        corrupted = dict(manifest)
        corrupted[key] = replacement
        rejected = data_audit._d002(
            repo_root, root, records, corrupted, data_generator
        )
        assert rejected["passed"] is False
        assert f"manifest:{key}" in rejected["failures"]
