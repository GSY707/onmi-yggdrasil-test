from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from yggdrasil_v2.r1_revalidation.production import generator as substrate
from yggdrasil_v2.v2_a.closure_c0r.data_generator import (
    DatasetSpec,
    GENERATOR_VERSION,
    MANIFEST_SCHEMA,
    SCHEMA_VERSION,
    generate,
    load,
    public_forward_view,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


def _spec() -> DatasetSpec:
    return DatasetSpec("small", 2026082401, 24, 24)


def _generate(path: Path) -> dict:
    return generate(
        path,
        spec=_spec(),
        design_doc=Path("docs/v2-a-closure-c0r-data-trace-qualification.md"),
        tokenizer=_Tokenizer(),
    )


def _rows(root: Path, family: str, split: str) -> list[dict]:
    return [json.loads(line) for line in (root / f"{family.lower()}-{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_identity_balance_and_causal_flip(tmp_path: Path) -> None:
    root = tmp_path / "c0r"
    manifest = _generate(root)
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["generator_version"] == GENERATOR_VERSION

    ordinary = _rows(root, "ERE", "train")
    relation = [row for row in ordinary if row["program_ast"]["query"]["kind"] == "relation"]
    assert len(relation) == 2
    assert {row["semantic_answer"] for row in relation} == {"true", "false"}
    assert all(row["schema_version"] == SCHEMA_VERSION for row in relation)
    assert all(row["generator_version"] == GENERATOR_VERSION for row in relation)

    causal = _rows(root, "ERE", "causal_pairs")
    for pair_id in sorted({row["pair_id"] for row in causal}):
        pair = [row for row in causal if row["pair_id"] == pair_id]
        assert {row["pair_role"] for row in pair} == {"base", "flip"}
        if pair[0]["program_ast"]["query"]["kind"] == "relation":
            assert {row["semantic_answer"] for row in pair} == {"true", "false"}
            assert pair[0]["causal_certificate"]["mutation_path"] == "/events/1/arguments/y"


def test_deterministic_bytes_and_old_identity_unchanged(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _generate(left)
    assert substrate.GENERATOR_VERSION != GENERATOR_VERSION
    _generate(right)
    files_left = sorted(left.glob("*.jsonl"))
    files_right = sorted(right.glob("*.jsonl"))
    assert [p.name for p in files_left] == [p.name for p in files_right]
    assert [hashlib.sha256(p.read_bytes()).digest() for p in files_left] == [
        hashlib.sha256(p.read_bytes()).digest() for p in files_right
    ]
    assert load(left)["ere-train"] == load(right)["ere-train"]


def test_public_forward_view_is_exactly_source_text() -> None:
    record = {"source_text": "episode", "answer_index": 3, "program_ast": {}}
    assert public_forward_view(record) == {"source_text": "episode"}
    assert tuple(public_forward_view(record)) == ("source_text",)
    with pytest.raises(ValueError):
        public_forward_view({"answer_index": 3})


def test_non_relation_ood_pair_is_not_reoriented() -> None:
    seed = 2026082401
    for split in ("composition_ood", "length_ood", "entity_ood"):
        base = substrate.build_ere_pair(seed, split, 12)
        repaired = __import__(
            "yggdrasil_v2.v2_a.closure_c0r.data_generator",
            fromlist=["_balanced_relation_builder"],
        )._balanced_relation_builder(seed, split, 12)
        assert repaired == base
