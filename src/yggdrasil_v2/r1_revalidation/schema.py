from __future__ import annotations

"""P0-D v2 record schema and the explicit visibility boundaries."""

from copy import deepcopy
from typing import Any, Mapping


SCHEMA_VERSION = "yggdrasil.v2-r1r.p0.data.v2"
GENERATOR_VERSION = "r1r-p0-generator-v2"
LABELS = tuple("ABCDEFGHI")
MAX_SOURCE_TOKENS = 1024
MODEL_ALLOWED_FIELDS = frozenset({
    "source_hidden",
    "source_attention_mask",
    "reasoning_budget",
    "valid_choice_mask",
})
FORBIDDEN_MODEL_FIELDS = frozenset({
    "answer",
    "answer_index",
    "answer_label",
    "ast",
    "program_ast",
    "teacher",
    "teacher_trace",
    "claim",
    "claims",
    "training_claims",
    "causal_certificate",
    "certificate",
    "span",
    "spans",
    "role",
    "roles",
    "entity",
    "entities",
    "candidate_validity",
    "candidate_validities",
    "simulator_state",
    "state",
    "correct_rule",
    "correct_plan",
    "choice_validity",
    "semantic_fingerprint",
    "surface_fingerprint",
    "qwen_token_count",
})


def _contains_forbidden_key(value: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_MODEL_FIELDS:
                found.append(child_path)
            found.extend(_contains_forbidden_key(child, path=child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_contains_forbidden_key(child, path=f"{path}[{index}]"))
    return found


def assert_model_view(view: Mapping[str, Any]) -> None:
    if not isinstance(view, Mapping):
        raise TypeError("model_view must be a mapping")
    unknown = set(view) - MODEL_ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"model_view contains fields outside the forward contract: {sorted(unknown)}")
    forbidden = _contains_forbidden_key(view)
    if forbidden:
        raise ValueError(f"model_view contains forbidden fields: {sorted(forbidden)}")
    if "reasoning_budget" not in view or "valid_choice_mask" not in view:
        raise ValueError("model_view requires reasoning_budget and valid_choice_mask")
    mask = view["valid_choice_mask"]
    if not isinstance(mask, (list, tuple)) or len(mask) != 9 or not all(isinstance(item, bool) for item in mask):
        raise ValueError("valid_choice_mask must be a boolean mask of length 9")
    if sum(mask) < 2:
        raise ValueError("valid_choice_mask must expose at least two choices")


def make_model_view(
    record: Mapping[str, Any],
    *,
    source_hidden: Any | None = None,
    source_attention_mask: Any | None = None,
) -> dict[str, Any]:
    view: dict[str, Any] = {
        "reasoning_budget": int(record["reasoning_budget"]),
        "valid_choice_mask": list(record["valid_choice_mask"]),
    }
    if source_hidden is not None:
        view["source_hidden"] = source_hidden
    if source_attention_mask is not None:
        view["source_attention_mask"] = source_attention_mask
    assert_model_view(view)
    return view


def make_training_view(record: Mapping[str, Any]) -> dict[str, Any]:
    view = make_model_view(record)
    view.update({
        "answer_index": int(record["answer_index"]),
        "training_claims": deepcopy(record.get("training_claims", [])),
        "teacher_trace": deepcopy(record.get("teacher_trace", [])),
    })
    return view


def make_audit_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(record))


def validate_record_shape(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "generator_version", "example_id", "family", "split",
        "source_text", "reasoning_budget", "valid_choice_mask", "answer_index",
        "semantic_fingerprint", "surface_fingerprint", "program_ast", "teacher_trace",
        "training_claims", "causal_certificate", "composition_signature",
        "answer_provenance", "renderer_features", "qwen_token_count", "provenance",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"record missing required fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {record['schema_version']}")
    if record["generator_version"] != GENERATOR_VERSION:
        raise ValueError(f"unsupported generator_version: {record['generator_version']}")
    if record["family"] not in {"ere", "cps"}:
        raise ValueError("family must be ere or cps")
    mask = record["valid_choice_mask"]
    if not isinstance(mask, list) or len(mask) != 9 or not all(isinstance(item, bool) for item in mask):
        raise ValueError("valid_choice_mask must be a list[bool] of length 9")
    answer_index = record["answer_index"]
    if not isinstance(answer_index, int) or not 0 <= answer_index < 9 or not mask[answer_index]:
        raise ValueError("answer_index must be an active A-I label")
    if not isinstance(record["source_text"], str) or not record["source_text"].strip():
        raise ValueError("source_text must be non-empty text")
    if not 1 <= int(record["reasoning_budget"]) <= 24:
        raise ValueError("reasoning_budget must be in [1, 24]")
    if not isinstance(record["qwen_token_count"], int) or record["qwen_token_count"] < 0:
        raise ValueError("qwen_token_count must be a non-negative integer")
    if not isinstance(record["provenance"], Mapping):
        raise ValueError("provenance must be a mapping")
    for seed_name in ("semantic_seed", "surface_seed", "label_seed"):
        if not isinstance(record["provenance"].get(seed_name), int):
            raise ValueError(f"provenance missing integer {seed_name}")


def assert_no_model_forbidden_fields(record: Mapping[str, Any]) -> None:
    assert_model_view(record)
