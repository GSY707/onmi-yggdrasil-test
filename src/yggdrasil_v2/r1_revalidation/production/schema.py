from __future__ import annotations

"""Production record-to-model boundary shared by generator and later model stages."""

from typing import Any, Mapping

from .renderer import LABELS, RendererError


MODEL_VIEW_FIELDS = ("example_id", "source_text", "reasoning_budget", "valid_choice_mask")
FORBIDDEN_MODEL_FIELDS = {
    "answer",
    "answer_index",
    "answer_label",
    "program_ast",
    "teacher_trace",
    "training_claims",
    "causal_certificate",
    "label_mapping",
    "audit_label_mapping",
    "candidate_validity",
    "entity_handles",
    "span_mask",
    "role_mask",
    "simulator_output",
}


def _error(code: str, message: str) -> None:
    raise RendererError(code, message)


def validate_model_view(view: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(view, Mapping) or tuple(view.keys()) != MODEL_VIEW_FIELDS:
        _error("model_view_schema", f"model view keys must be {MODEL_VIEW_FIELDS!r} in order")
    if not isinstance(view["example_id"], str) or not view["example_id"]:
        _error("model_view_id", "example_id must be nonempty")
    if not isinstance(view["source_text"], str) or not view["source_text"]:
        _error("model_view_source", "source_text must be nonempty")
    budget = view["reasoning_budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 24:
        _error("model_view_budget", "reasoning_budget must be an integer in 1..24")
    mask = view["valid_choice_mask"]
    if not isinstance(mask, list) or len(mask) != len(LABELS) or any(type(item) is not bool for item in mask) or not any(mask):
        _error("model_view_mask", "valid_choice_mask must contain nine booleans and at least one true")
    if any(key in FORBIDDEN_MODEL_FIELDS for key in view):
        _error("model_view_forbidden", "forbidden field present")
    return dict(view)


def model_view(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        _error("record_schema", "record must be an object")
    missing = [field for field in MODEL_VIEW_FIELDS if field not in record]
    if missing:
        _error("record_schema", f"missing model fields: {missing!r}")
    projected = {field: record[field] for field in MODEL_VIEW_FIELDS}
    return validate_model_view(projected)
