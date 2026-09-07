from __future__ import annotations

"""Canonical JSON, hashing, normalization, and contract validation helpers."""

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


LABELS = ("L0", "L1", "L2")


class ContractError(ValueError):
    """A deterministic, machine-auditable contract failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tree_hashes(root: Path, *, exclude: Iterable[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def normalize_source(text: str) -> str:
    if not isinstance(text, str):
        raise ContractError("invalid_text", "source must be a string")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw.strip())
        if line:
            lines.append(line)
    return "\n".join(lines)


def normalize_char_text(text: str) -> str:
    return " ".join(normalize_source(text).lower().split())


def exact_keys(value: Mapping[str, Any], expected: Sequence[str], *, code: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(code, "value must be an object")
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ContractError(code, f"keys mismatch missing={missing} extra={extra}")


def require_string(value: Any, *, code: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(code, f"{field} must be a non-empty string")
    return value


def require_labels(value: Any, *, code: str = "invalid_labels") -> tuple[str, ...]:
    if not isinstance(value, list) or tuple(value) != LABELS:
        raise ContractError(code, f"labels must be {list(LABELS)}")
    return LABELS


def require_label(value: Any, labels: Sequence[str] = LABELS, *, code: str = "unknown_label") -> str:
    if not isinstance(value, str) or value not in labels:
        raise ContractError(code, f"unknown label {value!r}")
    return value


def require_mask(value: Any, labels: Sequence[str] = LABELS, *, code: str = "invalid_choice_mask") -> tuple[bool, ...]:
    if not isinstance(value, list) or len(value) != len(labels) or any(type(item) is not bool for item in value):
        raise ContractError(code, f"valid_choice_mask must contain {len(labels)} booleans")
    if not any(value):
        raise ContractError(code, "valid_choice_mask cannot be all false")
    return tuple(value)


def choose_label(scores: Mapping[str, Any], mask: Sequence[bool], labels: Sequence[str] = LABELS) -> str:
    active = [label for label, enabled in zip(labels, mask, strict=True) if enabled]
    if not active:
        raise ContractError("invalid_choice_mask", "no active labels")
    return max(active, key=lambda label: (scores[label], -labels.index(label)))


def validate_unique_ids(rows: Sequence[Mapping[str, Any]], field: str = "row_id") -> None:
    seen: set[str] = set()
    for row in rows:
        identifier = require_string(row.get(field), code="invalid_id", field=field)
        if identifier in seen:
            raise ContractError("duplicate_id", f"duplicate {field} {identifier!r}")
        seen.add(identifier)


def ensure_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError("invalid_number", f"{field} must be finite")
    return float(value)

