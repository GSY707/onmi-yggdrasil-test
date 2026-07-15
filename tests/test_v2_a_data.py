from __future__ import annotations

from pathlib import Path

from yggdrasil_v2.reasoning_medium.data import DatasetSpec, build_dataset, load_jsonl
from yggdrasil_v2.reasoning_medium.prompts import build_messages, has_complete_answer, parse_answer


def test_dataset_splits_are_deterministic_and_leak_free(tmp_path: Path) -> None:
    spec = DatasetSpec(
        seed=7,
        train_size=24,
        validation_size=8,
        test_size=8,
        composition_size=8,
        length_size=8,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_dataset(first, spec)
    build_dataset(second, spec)

    assert (first / "train.jsonl").read_bytes() == (second / "train.jsonl").read_bytes()
    assert manifest["leakage_checks"]["cross_split_fingerprint_overlap"] == 0
    assert manifest["unique_examples"] == 56

    train = load_jsonl(first / "train.jsonl")
    composition = load_jsonl(first / "composition_heldout.jsonl")
    length = load_jsonl(first / "length_heldout.jsonl")
    assert all(not record["composition_tags"] for record in train)
    assert all(record["composition_tags"] for record in composition)
    assert min(record["program_length"] for record in length) >= 5
    assert all(record["trace_text"] not in record["question"] for record in train)


def test_answer_parser_is_strict_about_ambiguous_symbols() -> None:
    assert parse_answer("FINAL: A") == "A"
    assert parse_answer("Final value of the amber register: B") == "B"
    assert parse_answer("The final value of amber is C.") == "C"
    assert has_complete_answer("The final value of amber is C.", "text_cot")
    assert has_complete_answer("The answer is D.", "direct")
    assert parse_answer("E") == "E"
    assert parse_answer("I considered A and selected B") is None


def test_encoder_uses_the_same_state_tracking_contract_as_visible_cot() -> None:
    example = {
        "question": "Track this register program.",
        "trace_text": "Start: amber=A, cobalt=B, jade=C.\nFINAL: A",
        "answer": "A",
    }
    encoder = build_messages(example, "encoder")
    visible = build_messages(example, "text_cot")
    assert encoder[-1]["content"] == visible[-1]["content"]

    demonstrated = build_messages(example, "encoder", [example])
    assert example["trace_text"] in demonstrated[2]["content"]
