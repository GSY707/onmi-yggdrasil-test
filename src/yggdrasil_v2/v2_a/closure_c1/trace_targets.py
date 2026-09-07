from __future__ import annotations

"""Offline CT1 target materialisation and tokenizer-to-step alignment."""

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence

from .data import CT1_GRAMMAR_IDS, PublicRecord, public_record_dict
from ..closure_c0r.compact_trace import format_compact_trace


TRACE_TARGET_SCHEMA = "yggdrasil.v2-a.closure-c1.ct1-target.v1"
TRACE_STEPS = 10


@dataclass(frozen=True)
class TraceToken:
    token_id: int
    offset: tuple[int, int]
    step: int
    local_position: int
    grammar: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "offset": list(self.offset),
            "step": self.step,
            "local_position": self.local_position,
            "grammar": self.grammar,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True)
class TraceTarget:
    example_id: str
    text: str
    tokens: tuple[TraceToken, ...]
    family: str | None = None

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(token.token_id for token in self.tokens)

    @property
    def step_ids(self) -> tuple[int, ...]:
        return tuple(token.step for token in self.tokens)

    @property
    def local_positions(self) -> tuple[int, ...]:
        return tuple(token.local_position for token in self.tokens)

    def as_dict(self) -> dict[str, Any]:
        # This is a target artifact, not a cache payload.  It deliberately
        # contains no AST, answer, family metadata beyond a reporting label.
        return {
            "schema_version": TRACE_TARGET_SCHEMA,
            "example_id": self.example_id,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "offsets": [list(token.offset) for token in self.tokens],
            "step_ids": list(self.step_ids),
            "local_positions": list(self.local_positions),
            "grammar_mask": [token.grammar for token in self.tokens],
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except TypeError:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise TypeError("C1 CT1 materialisation requires a fast tokenizer with offset_mapping")
    ids_raw, offsets_raw = encoded["input_ids"], encoded["offset_mapping"]
    if hasattr(ids_raw, "tolist"):
        ids_raw = ids_raw.tolist()
    if hasattr(offsets_raw, "tolist"):
        offsets_raw = offsets_raw.tolist()
    if ids_raw and isinstance(ids_raw[0], list):
        ids_raw = ids_raw[0]
    if offsets_raw and isinstance(offsets_raw[0], list) and offsets_raw[0] and isinstance(offsets_raw[0][0], list):
        offsets_raw = offsets_raw[0]
    ids = [int(item) for item in ids_raw]
    offsets = [(int(item[0]), int(item[1])) for item in offsets_raw]
    if len(ids) != len(offsets):
        raise ValueError("tokenizer returned mismatched input_ids/offset_mapping")
    if any(start < 0 or end < start or end > len(text) for start, end in offsets):
        raise ValueError("tokenizer offset is outside CT1 text")
    return ids, offsets


def _item_spans(text: str, family: str) -> list[tuple[int, int, int]]:
    """Locate canonical trace items without parsing target values into fields."""
    marker = "\"t\":["
    payload_start = text.find(marker)
    if payload_start < 0:
        return []
    cursor = payload_start + len(marker)
    # Parsing each item with JSONDecoder is robust to strings containing ']'.
    decoder = json.JSONDecoder()
    spans: list[tuple[int, int, int]] = []
    index = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor] in " ,":
            cursor += 1
        if cursor >= len(text) or text[cursor] == "]":
            break
        try:
            _, end = decoder.raw_decode(text[cursor:])
        except json.JSONDecodeError:
            break
        spans.append((cursor, cursor + end, min(TRACE_STEPS, index + 1)))
        cursor += end
        index += 1
    return spans


def align_trace_tokens(text: str, tokenizer: Any, *, grammar_ids: Iterable[int] = CT1_GRAMMAR_IDS) -> tuple[TraceToken, ...]:
    ids, offsets = _tokenize_with_offsets(tokenizer, text)
    spans = _item_spans(text, "")
    local: dict[int, int] = {}
    result: list[TraceToken] = []
    grammar = set(int(item) for item in grammar_ids)
    payload_start = text.find("CT1 ")
    suffix_start = text.find("\nAnswer: ")
    if payload_start < 0:
        payload_start = 0
    if suffix_start < 0:
        suffix_start = len(text)
    for token_id, (start, end) in zip(ids, offsets):
        midpoint = start if end <= start else (start + end) // 2
        step = 1
        # Fixed prefix, separators, and all text before the first item belong
        # to H1.  After the trace and Answer suffix belong to H10.
        for span_start, span_end, span_step in spans:
            if midpoint >= span_start:
                step = span_step
            if span_start <= midpoint < span_end:
                step = span_step
                break
        if midpoint >= suffix_start:
            step = TRACE_STEPS
        local[step] = local.get(step, 0) + 1
        result.append(TraceToken(int(token_id), (start, end), step, local[step] - 1, int(token_id) in grammar))
    return tuple(result)


def trace_text_for_record(record: Mapping[str, Any] | PublicRecord) -> str:
    if isinstance(record, PublicRecord):
        raise ValueError("trace text requires an offline C0R record, not a public-only record")
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    for key in ("compact_trace", "trace_text"):
        value = record.get(key)
        if type(value) is str:
            return value
    return format_compact_trace(record)


def materialize_trace_target(record: Mapping[str, Any], tokenizer: Any, lexicon: Iterable[int] | Mapping[int, Any] | None = None) -> TraceTarget:
    text = trace_text_for_record(record)
    tokens = align_trace_tokens(text, tokenizer)
    if lexicon is not None:
        allowed = set(int(item) for item in (lexicon.keys() if isinstance(lexicon, Mapping) else lexicon))
        missing = sorted({token.token_id for token in tokens} - allowed)
        if missing:
            raise ValueError(f"CT1 target OOV token ids: {missing[:12]}")
    return TraceTarget(str(record.get("example_id", "")), text, tokens, record.get("family"))


def materialize_trace_targets(records: Sequence[Mapping[str, Any] | PublicRecord], tokenizer: Any, lexicon: Iterable[int] | Mapping[int, Any] | None = None) -> list[TraceTarget]:
    return [materialize_trace_target(record, tokenizer, lexicon) for record in records]


def build_trace_lexicon(records: Sequence[Mapping[str, Any] | PublicRecord], tokenizer: Any) -> dict[str, Any]:
    """Construct the frozen source-plus-grammar lexicon and prove target OOV=0."""
    source_ids: set[int] = set()
    for record in records:
        # Lexicon construction is an offline qualification operation and may
        # receive immutable full C0R rows.  It still reads exactly one field;
        # the forbidden fields never enter the tokenizer or returned report.
        if isinstance(record, PublicRecord):
            source = record.source_text
        elif isinstance(record, Mapping) and type(record.get("source_text")) is str:
            source = record["source_text"]
        else:
            raise ValueError("record source_text must be a string")
        encoded = tokenizer(source, add_special_tokens=False, truncation=False)
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        source_ids.update(int(item) for item in ids)
    grammar_ids = set(CT1_GRAMMAR_IDS)
    lexicon = source_ids | grammar_ids
    report = {
        "schema_version": "yggdrasil.v2-a.closure-c1.trace-lexicon.v1",
        "source_token_ids": sorted(source_ids),
        "grammar_token_ids": list(CT1_GRAMMAR_IDS),
        "lexicon_ids": sorted(lexicon),
        "source_count": len(source_ids),
        "grammar_count": len(grammar_ids),
        "lexicon_count": len(lexicon),
    }
    targets: list[TraceTarget] = []
    try:
        targets = materialize_trace_targets(records, tokenizer, lexicon)
    except ValueError as exc:
        report["target_oov_error"] = str(exc)
        raise
    target_ids = {token_id for target in targets for token_id in target.token_ids}
    report["target_token_count"] = sum(len(target.tokens) for target in targets)
    report["target_oov_ids"] = sorted(target_ids - lexicon)
    report["target_coverage"] = 1.0 if not report["target_oov_ids"] else 0.0
    if report["target_oov_ids"]:
        raise ValueError(f"CT1 target OOV ids: {report['target_oov_ids'][:12]}")
    return report


# Explicit aliases used by the C1 runner and by small independent probes.
build_lexicon = build_trace_lexicon
align_tokens_to_steps = align_trace_tokens


__all__ = [
    "TRACE_STEPS", "TRACE_TARGET_SCHEMA", "TraceTarget", "TraceToken",
    "align_trace_tokens", "align_tokens_to_steps", "build_lexicon",
    "build_trace_lexicon", "materialize_trace_target", "materialize_trace_targets",
    "trace_text_for_record",
]
