"""Strict CT1 compact trace formatter, parser, replay verifier and qualification.

The AST is deliberately accepted only by the offline verifier.  The serialized
target contains no AST, state oracle, or claim fields.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

from ...r1_revalidation.common.simulator import evaluate_cps, simulate_ere

PREFIX = "CT1 "
LABELS = tuple("ABCDEFGHI")


class CompactTraceError(ValueError):
    def __init__(self, code: str, path: str = "$") -> None:
        self.code, self.path = code, path
        super().__init__(f"COMPACT_TRACE|{code}|{path}")


@dataclass(frozen=True)
class ParsedTrace:
    family: str
    trace: tuple[Any, ...]
    answer: str
    text: str


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    errors: tuple[str, ...] = ()
    family: str | None = None
    answer_match: bool = False
    trace_match: bool = False


def _reject_constant(value: str) -> None:
    raise CompactTraceError("nonfinite", "$.payload")


def _loads_exact(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=lambda pairs: _pairs(pairs),
            parse_constant=_reject_constant,
        )
    except CompactTraceError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CompactTraceError("invalid_json", "$.payload") from exc


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CompactTraceError("duplicate_key", "$.payload")
        out[key] = value
    return out


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _type(value: Any, typ: type, path: str) -> None:
    if type(value) is not typ:
        raise CompactTraceError("wrong_type", path)


def _int(value: Any, path: str) -> int:
    if type(value) is not int:
        raise CompactTraceError("wrong_type", path)
    return value


def _str(value: Any, path: str) -> str:
    if type(value) is not str:
        raise CompactTraceError("wrong_type", path)
    return value


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise CompactTraceError("wrong_type", path)
    return value


def _exact_keys(value: Any, keys: tuple[str, ...], path: str) -> None:
    if type(value) is not dict or tuple(value) != keys:
        raise CompactTraceError("schema_keys", path)


def _parse_payload(payload: Any) -> tuple[str, tuple[Any, ...]]:
    _exact_keys(payload, ("f", "t"), "$.payload")
    family = _str(payload["f"], "$.payload.f")
    if family not in {"ERE", "CPS"}:
        raise CompactTraceError("invalid_family", "$.payload.f")
    trace = payload["t"]
    if type(trace) is not list:
        raise CompactTraceError("wrong_type", "$.payload.t")
    for i, item in enumerate(trace):
        path = f"$.payload.t[{i}]"
        if family == "ERE":
            if type(item) is not list or len(item) != 5:
                raise CompactTraceError("schema_shape", path)
            if _int(item[0], f"{path}[0]") < 0 or (i and item[0] <= trace[i - 1][0]):
                raise CompactTraceError("invalid_index", f"{path}[0]")
            _str(item[1], f"{path}[1]")
            pairs = item[2]
            if type(pairs) is not list or any(type(p) is not list or len(p) != 2 for p in pairs):
                raise CompactTraceError("schema_shape", f"{path}[2]")
            for j, pair in enumerate(pairs):
                _str(pair[0], f"{path}[2][{j}][0]"); _str(pair[1], f"{path}[2][{j}][1]")
            if len({pair[0] for pair in pairs}) != len(pairs):
                raise CompactTraceError("duplicate_argument", f"{path}[2]")
            if pairs != sorted(pairs, key=lambda p: p[0]):
                raise CompactTraceError("noncanonical_order", f"{path}[2]")
            if type(item[3]) is not list or any(type(op) is not str for op in item[3]):
                raise CompactTraceError("wrong_type", f"{path}[3]")
            if _int(item[4], f"{path}[4]") < 0:
                raise CompactTraceError("invalid_value", f"{path}[4]")
        else:
            if type(item) is not list or len(item) != 10:
                raise CompactTraceError("schema_shape", path)
            if _int(item[0], f"{path}[0]") < 0 or (i and item[0] <= trace[i - 1][0]):
                raise CompactTraceError("invalid_index", f"{path}[0]")
            _type(item[1], list, f"{path}[1]")
            if any(type(action) is not str for action in item[1]):
                raise CompactTraceError("wrong_type", f"{path}[1]")
            _bool(item[2], f"{path}[2]"); _bool(item[3], f"{path}[3]"); _int(item[4], f"{path}[4]")
            if item[4] < 0:
                raise CompactTraceError("invalid_value", f"{path}[4]")
            for j in (5, 6, 7): _bool(item[j], f"{path}[{j}]")
            if type(item[8]) is not list or any(type(x) is not str for x in item[8]):
                raise CompactTraceError("wrong_type", f"{path}[8]")
            if type(item[9]) is not list:
                raise CompactTraceError("wrong_type", f"{path}[9]")
            for j, step in enumerate(item[9]):
                sp = f"{path}[9][{j}]"
                if type(step) is not list or len(step) != 4:
                    raise CompactTraceError("schema_shape", sp)
                if _int(step[0], f"{sp}[0]") < 0 or (j and step[0] <= item[9][j - 1][0]):
                    raise CompactTraceError("invalid_index", f"{sp}[0]")
                if step[0] >= len(item[1]):
                    raise CompactTraceError("invalid_index", f"{sp}[0]")
                _bool(step[1], f"{sp}[1]")
                if step[2] is not None: _str(step[2], f"{sp}[2]")
                _int(step[3], f"{sp}[3]")
                if step[3] < 0: raise CompactTraceError("invalid_value", f"{sp}[3]")
    return family, tuple(trace)


def parse_compact_trace(text: str, *, expected_family: str | None = None) -> ParsedTrace:
    if type(text) is not str or not text.startswith(PREFIX):
        raise CompactTraceError("prefix", "$")
    parts = text.split("\n")
    if len(parts) != 2 or not parts[1].startswith("Answer: "):
        raise CompactTraceError("layout", "$")
    answer = parts[1][8:]
    if answer not in LABELS or not answer:
        raise CompactTraceError("answer", "$.answer")
    payload_text = parts[0][len(PREFIX):]
    payload = _loads_exact(payload_text)
    if _canonical(payload) != payload_text:
        raise CompactTraceError("noncanonical", "$.payload")
    family, trace = _parse_payload(payload)
    if expected_family is not None and family != expected_family:
        raise CompactTraceError("family_mismatch", "$.payload.f")
    return ParsedTrace(family, trace, answer, text)


MALFORMED_CASES = (
    "x CT1 {}\nAnswer: A",
    "CT1 {}\nAnswer: A",
    'CT1 {"f":"ERE","f":"CPS","t":[]}\nAnswer: A',
    'CT1 {"f": "ERE", "t": []}\nAnswer: A',
    'CT1 {"f":"ERE","t":null}\nAnswer: A',
    'CT1 {"f":"ERE","t":[[0,"r",[],[],true]]}\nAnswer: A',
    'CT1 {"f":"ERE","t":[]}\nAnswer: Z',
)


def malformed_rejection_report() -> dict[str, Any]:
    rejected = 0
    for case in MALFORMED_CASES:
        try:
            parse_compact_trace(case)
        except CompactTraceError:
            rejected += 1
    return {"total": len(MALFORMED_CASES), "rejected": rejected, "passed": rejected == len(MALFORMED_CASES)}


def _answer_label(semantic: Any, mapping: Mapping[str, str]) -> str:
    if semantic is None:
        tokens = {"none"}
    elif isinstance(semantic, bool):
        tokens = {"true" if semantic else "false"}
    elif isinstance(semantic, int):
        tokens = {f"candidate:{semantic}"}
    else:
        token = str(semantic)
        tokens = {token, f"value:{token}"}
    matches = [label for label, value in mapping.items() if value in tokens]
    if len(matches) != 1 or matches[0] not in LABELS:
        raise CompactTraceError("answer_mapping", "$.answer")
    return matches[0]


def _ere_trace(output: Mapping[str, Any]) -> list[list[Any]]:
    return [[int(s["event_index"]), str(s["rule"]), [[str(k), str(v)] for k, v in sorted(s["arguments"].items())], list(s["executed_ops"]), int(s["delta_budget"])] for s in output["trace"]]


def _cps_trace(output: Mapping[str, Any]) -> list[list[Any]]:
    result = []
    for c in output["candidates"]:
        steps = [
            [int(s["position"]), bool(s["legal"]), s["failure"], int(s["cost"])]
            for s in c["trace"]
        ]
        result.append([int(c["candidate_index"]), list(c["plan"]), bool(c["legal"]), bool(c["valid"]), int(c["total_cost"]), bool(c["budget_ok"]), bool(c["goal_satisfied"]), bool(c["final_constraints_satisfied"]), list(c["failure_reasons"]), steps])
    return result


def format_compact_trace(record: Mapping[str, Any]) -> str:
    family = record.get("family")
    ast = record.get("program_ast")
    mapping = record.get("label_mapping")
    if family not in {"ERE", "CPS"} or not isinstance(ast, Mapping) or not isinstance(mapping, Mapping):
        raise CompactTraceError("record_schema", "$")
    output = simulate_ere(ast) if family == "ERE" else evaluate_cps(ast)
    payload = {"f": family, "t": _ere_trace(output) if family == "ERE" else _cps_trace(output)}
    label = _answer_label(output["answer"], mapping)
    return PREFIX + _canonical(payload) + "\nAnswer: " + label


def verify_compact_trace(*, trace_text: str, program_ast: Mapping[str, Any], label_mapping: Mapping[str, str]) -> VerificationResult:
    try:
        parsed = parse_compact_trace(trace_text)
        output = simulate_ere(program_ast) if parsed.family == "ERE" else evaluate_cps(program_ast)
        expected = _answer_label(output["answer"], label_mapping)
        expected_trace = _ere_trace(output) if parsed.family == "ERE" else _cps_trace(output)
        answer_ok = parsed.answer == expected
        trace_ok = list(parsed.trace) == expected_trace
        if not answer_ok: return VerificationResult(False, ("answer_mismatch",), parsed.family, False, trace_ok)
        if not trace_ok: return VerificationResult(False, ("trace_mismatch",), parsed.family, True, False)
        return VerificationResult(True, (), parsed.family, True, True)
    except (CompactTraceError, ValueError, KeyError, TypeError) as exc:
        return VerificationResult(False, (str(exc),))


def directed_fault_mutations(text: str) -> dict[str, str]:
    """Return schema-valid single-point mutations for verifier fault-kill tests."""
    parsed = parse_compact_trace(text)
    payload = {"f": parsed.family, "t": [list(x) for x in parsed.trace]}
    out: dict[str, str] = {}
    def emit(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        p = {"f": payload["f"], "t": json.loads(json.dumps(payload["t"]))}
        mutate(p)
        if p != payload:
            out[name] = PREFIX + _canonical(p) + "\nAnswer: " + parsed.answer
    def emit_answer(name: str) -> None:
        alternate = next(label for label in LABELS if label != parsed.answer)
        out[name] = PREFIX + _canonical(payload) + "\nAnswer: " + alternate
    if parsed.family == "ERE" and payload["t"]:
        if len(payload["t"]) > 1: emit("ere_drop_event", lambda p: p["t"].pop())
        emit("ere_event_index", lambda p: [item.__setitem__(0, item[0] + 1) for item in p["t"]])
        emit("ere_rule", lambda p: p["t"][0].__setitem__(1, p["t"][0][1] + "_fault"))
        if len(payload["t"][0][2]) > 1: emit("ere_drop_argument", lambda p: p["t"][0][2].pop())
        emit("ere_argument", lambda p: (p["t"][0][2].append(["fault", "x"]), p["t"][0][2].sort(key=lambda pair: pair[0])))
        if payload["t"][0][3]: emit("ere_drop_op", lambda p: p["t"][0][3].pop())
        emit("ere_op", lambda p: p["t"][0][3].append("SET"))
        emit("ere_delta", lambda p: p["t"][0].__setitem__(4, p["t"][0][4] + 1))
        if len(payload["t"]) > 1: emit("ere_order", lambda p: (p["t"].__setitem__(slice(0, 2), [p["t"][1], p["t"][0]]), p["t"][0].__setitem__(0, 0), p["t"][1].__setitem__(0, 1)))
        emit_answer("ere_answer")
    if parsed.family == "CPS" and payload["t"]:
        if len(payload["t"]) > 1: emit("cps_drop_candidate", lambda p: p["t"].pop())
        emit("cps_candidate_index", lambda p: [item.__setitem__(0, item[0] + 1) for item in p["t"]])
        emit("cps_plan", lambda p: p["t"][0][1].append("fault"))
        position_candidate = next(
            (
                index
                for index, candidate in enumerate(payload["t"])
                if candidate[9] and len(candidate[9]) < len(candidate[1])
            ),
            None,
        )
        if position_candidate is not None:
            emit(
                "cps_drop_plan",
                lambda p, index=position_candidate: p["t"][index][1].pop(),
            )
            emit(
                "cps_step_position",
                lambda p, index=position_candidate: [
                    step.__setitem__(0, step[0] + 1) for step in p["t"][index][9]
                ],
            )
        if payload["t"][0][9]:
            emit("cps_step_legal", lambda p: p["t"][0][9][0].__setitem__(1, not p["t"][0][9][0][1]))
            emit("cps_step_failure", lambda p: p["t"][0][9][0].__setitem__(2, "fault" if p["t"][0][9][0][2] is None else None))
            if len(payload["t"][0][9]) > 1: emit("cps_drop_step", lambda p: p["t"][0][9].pop())
            if len(payload["t"][0][9]) > 1: emit("cps_step_order", lambda p: (p["t"][0][9].__setitem__(slice(0, 2), [p["t"][0][9][1], p["t"][0][9][0]]), p["t"][0][9][0].__setitem__(0, 0), p["t"][0][9][1].__setitem__(0, 1)))
        emit("cps_valid", lambda p: p["t"][0].__setitem__(3, not p["t"][0][3]))
        emit("cps_cost", lambda p: p["t"][0].__setitem__(4, p["t"][0][4] + 1))
        emit("cps_legal", lambda p: p["t"][0].__setitem__(2, not p["t"][0][2]))
        emit("cps_total_cost", lambda p: p["t"][0].__setitem__(4, p["t"][0][4] + 1))
        emit("cps_budget", lambda p: p["t"][0].__setitem__(5, not p["t"][0][5]))
        emit("cps_goal", lambda p: p["t"][0].__setitem__(6, not p["t"][0][6]))
        emit("cps_final", lambda p: p["t"][0].__setitem__(7, not p["t"][0][7]))
        if payload["t"][0][8]: emit("cps_drop_reason", lambda p: p["t"][0][8].pop())
        emit("cps_reason", lambda p: p["t"][0][8].append("fault"))
        if payload["t"][0][9]: emit("cps_step_cost", lambda p: p["t"][0][9][0].__setitem__(3, p["t"][0][9][0][3] + 1))
        if len(payload["t"]) > 1: emit("cps_order", lambda p: (p["t"].__setitem__(slice(0, 2), [p["t"][1], p["t"][0]]), p["t"][0].__setitem__(0, 0), p["t"][1].__setitem__(0, 1)))
        emit_answer("cps_answer")
    return out


def qualify_full_bank(records: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    report: dict[str, Any] = {"records": len(records), "roundtrip": 0, "replay": 0, "fault_kill": 0, "faults": 0, "token_count": {}, "malformed": malformed_rejection_report(), "cells": {}, "passed": False}
    lengths: list[int] = []
    for record in records:
        text = format_compact_trace(record)
        parsed = parse_compact_trace(text, expected_family=record["family"])
        replay = verify_compact_trace(trace_text=text, program_ast=record["program_ast"], label_mapping=record["label_mapping"])
        roundtrip = format_compact_trace(record) == parsed.text
        faults = directed_fault_mutations(text)
        killed = sum(not verify_compact_trace(trace_text=fault, program_ast=record["program_ast"], label_mapping=record["label_mapping"]).passed for fault in faults.values())
        report["roundtrip"] += int(roundtrip); report["replay"] += int(replay.passed); report["fault_kill"] += killed; report["faults"] += len(faults)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if type(encoded) is not list or any(type(token) is not int for token in encoded):
            raise TypeError("tokenizer.encode must return list[int]")
        lengths.append(len(encoded))
        cell = f"{record.get('family')}/{record.get('split')}"
        cell_report = report["cells"].setdefault(cell, {"records": 0, "roundtrip": 0, "replay": 0, "faults": 0, "fault_kill": 0})
        cell_report.update({"records": cell_report.get("records", 0) + 1, "roundtrip": cell_report.get("roundtrip", 0) + int(roundtrip), "replay": cell_report.get("replay", 0) + int(replay.passed), "faults": cell_report.get("faults", 0) + len(faults), "fault_kill": cell_report.get("fault_kill", 0) + killed})
        report["cells"][cell] = cell_report
    lengths.sort()
    def percentile(percent: int) -> int:
        if not lengths: return 0
        index = max(0, min(len(lengths) - 1, (percent * len(lengths) + 99) // 100 - 1))
        return lengths[index]
    report["token_count"] = {"max": percentile(100), "p50": percentile(50), "p95": percentile(95), "p99": percentile(99), "cap": 512, "passed": bool(lengths) and percentile(100) <= 512}
    report["passed"] = report["roundtrip"] == report["records"] and report["replay"] == report["records"] and report["fault_kill"] == report["faults"] and report["token_count"]["passed"] and report["malformed"]["passed"]
    return report


__all__ = ["CompactTraceError", "ParsedTrace", "VerificationResult", "MALFORMED_CASES", "parse_compact_trace", "malformed_rejection_report", "format_compact_trace", "verify_compact_trace", "directed_fault_mutations", "qualify_full_bank"]
