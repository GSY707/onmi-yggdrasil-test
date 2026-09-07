from __future__ import annotations

import json
from pathlib import Path

import pytest

from yggdrasil_v2.v2_a.closure_c0r.compact_trace import (
    CompactTraceError,
    directed_fault_mutations,
    format_compact_trace,
    parse_compact_trace,
    qualify_full_bank,
    verify_compact_trace,
)


ROOT = Path("artifacts/v2-r1r/p0d-v17-full-production-20260810-1/dataset")


def _record(family: str) -> dict:
    return json.loads((ROOT / f"{family.lower()}-train.jsonl").read_text(encoding="utf-8").splitlines()[0])


def _record_at(family: str, index: int) -> dict:
    return json.loads(
        (ROOT / f"{family.lower()}-train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[index]
    )


def _record_with_steps() -> dict:
    for line in (ROOT / "cps-train.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        first = row["simulator_output"]["candidates"][0]
        if len(first["trace"]) >= 2 and first["failure_reasons"]:
            return row
    raise AssertionError("fixture bank has no CPS step trace")


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


@pytest.mark.parametrize("family", ["ERE", "CPS"])
def test_exact_format_parse_replay_and_no_ast_in_target(family: str) -> None:
    record = _record(family)
    text = format_compact_trace(record)
    assert text.startswith("CT1 ")
    assert "program_ast" not in text and "teacher_trace" not in text
    parsed = parse_compact_trace(text, expected_family=family)
    assert parsed.text == text
    assert format_compact_trace(record) == text
    result = verify_compact_trace(
        trace_text=text,
        program_ast=record["program_ast"],
        label_mapping=record["label_mapping"],
    )
    assert result.passed


@pytest.mark.parametrize(
    "bad",
    [
        "x CT1 {}\nAnswer: A",
        "CT1 {}\nAnswer: A",
        "CT1 {\"f\":\"ERE\",\"f\":\"CPS\",\"t\":[]}\nAnswer: A",
        "CT1 {\"f\": \"ERE\", \"t\": []}\nAnswer: A",
        "CT1 {\"f\":\"ERE\",\"t\":[]}\nAnswer: Z",
        "CT1 {\"f\":\"ERE\",\"t\":null}\nAnswer: A",
        "CT1 {\"f\":\"ERE\",\"t\":[[0,\"r\",[],[],true]]}\nAnswer: A",
    ],
)
def test_malformed_and_noncanonical_inputs_fail_closed(bad: str) -> None:
    with pytest.raises(CompactTraceError):
        parse_compact_trace(bad)


@pytest.mark.parametrize("family", ["ERE", "CPS"])
def test_schema_valid_directed_faults_are_all_killed(family: str) -> None:
    record = _record_with_steps() if family == "CPS" else _record(family)
    text = format_compact_trace(record)
    mutations = directed_fault_mutations(text)
    assert mutations
    required = (
        {"ere_drop_event", "ere_order", "ere_event_index", "ere_rule", "ere_drop_argument", "ere_argument", "ere_drop_op", "ere_op", "ere_delta", "ere_answer"}
        if family == "ERE" else
        {"cps_drop_candidate", "cps_order", "cps_candidate_index", "cps_drop_plan", "cps_plan", "cps_legal", "cps_valid", "cps_total_cost", "cps_budget", "cps_goal", "cps_final", "cps_drop_reason", "cps_reason", "cps_step_position", "cps_step_legal", "cps_step_failure", "cps_step_cost", "cps_drop_step", "cps_step_order", "cps_answer"}
    )
    assert required <= set(mutations)
    for name, mutated in mutations.items():
        # Mutation registry must remain syntactically/schema valid.
        parse_compact_trace(mutated, expected_family=family)
        result = verify_compact_trace(
            trace_text=mutated,
            program_ast=record["program_ast"],
            label_mapping=record["label_mapping"],
        )
        assert not result.passed, name


def test_small_full_bank_report_has_cell_and_fault_accounting() -> None:
    records = [_record("ERE"), _record("CPS")]
    report = qualify_full_bank(records, _Tokenizer())
    assert report["passed"]
    assert report["roundtrip"] == 2
    assert report["replay"] == 2
    assert report["fault_kill"] == report["faults"]
    assert set(report["cells"]) == {"ERE/train", "CPS/train"}
    assert report["malformed"]["passed"]
    assert report["token_count"]["max"] <= 512


@pytest.mark.parametrize(("family", "index"), [("ERE", 1), ("CPS", 1)])
def test_non_boolean_and_non_none_answers_bind_to_local_label(family: str, index: int) -> None:
    record = _record_at(family, index)
    text = format_compact_trace(record)
    assert verify_compact_trace(
        trace_text=text,
        program_ast=record["program_ast"],
        label_mapping=record["label_mapping"],
    ).passed
