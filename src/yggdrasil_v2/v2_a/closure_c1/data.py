from __future__ import annotations

"""C1 public-data boundary.

The C0R records are intentionally read here only to expose ``source_text``
and opaque scheduling metadata.  ASTs, answers, labels and teacher traces are
never returned by this module.  The cache and trace modules consequently have
to obtain their model input through this small, auditable surface.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


C1_DATA_SCHEMA = "yggdrasil.v2-a.closure-c1.public-source.v1"
MAX_SOURCE_TOKENS = 1024
CT1_GRAMMAR_IDS: tuple[int, ...] = (
    58, 92, 487, 1089, 1123, 1143, 1288, 1293, 1666, 1797, 1802,
    2129, 2456, 2685, 3147, 4851, 4891, 5046, 5702, 7664, 8631, 8783,
    11534, 14522, 15050, 15666, 16352, 17709, 20691, 22357, 22642,
    25312, 31928, 32817, 34764, 40775, 45404, 46793, 55558, 80620,
    86451, 93482,
)
PUBLIC_RECORD_FIELDS = frozenset({
    "example_id", "family", "split", "pair_id", "pair_role", "source_text",
})
FORBIDDEN_PUBLIC_FIELDS = frozenset({
    "program_ast", "ast", "answer", "answer_index", "semantic_answer",
    "label_mapping", "valid_choice_mask", "teacher_trace", "trace",
    "compact_trace", "training_claims", "reasoning_budget", "causal_certificate",
    "simulator_output", "dense_state", "margin", "vjp", "fisher",
})


@dataclass(frozen=True)
class PublicRecord:
    """A source-only record with metadata usable for scheduling/reporting."""

    example_id: str
    family: str
    split: str
    source_text: str
    pair_id: str | None = None
    pair_role: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "example_id": self.example_id,
            "family": self.family,
            "split": self.split,
            "source_text": self.source_text,
        }
        if self.pair_id is not None:
            result["pair_id"] = self.pair_id
        if self.pair_role is not None:
            result["pair_role"] = self.pair_role
        return result

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def source_byte_sha256(source_text: str) -> str:
    if type(source_text) is not str:
        raise TypeError("source_text must be str")
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest().upper()


def _dataset_dir(root: Path) -> Path:
    root = Path(root)
    candidate = root / "dataset"
    if candidate.is_dir():
        return candidate
    if (root / "manifest.json").is_file() and any(root.glob("*.jsonl")):
        return root
    raise FileNotFoundError(f"C1 dataset directory not found under {root}")


def _read_jsonl(path: Path, *, family: str, split: str) -> list[PublicRecord]:
    records: list[PublicRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"record is not an object at {path}:{line_no}")
            if type(raw.get("example_id")) is not str or type(raw.get("source_text")) is not str:
                raise ValueError(f"record lacks source-only identity at {path}:{line_no}")
            if raw.get("family") != family:
                raise ValueError(f"family mismatch at {path}:{line_no}")
            pair_id = raw.get("pair_id")
            pair_role = raw.get("pair_role")
            if pair_id is not None and type(pair_id) is not str:
                raise ValueError(f"pair_id must be string at {path}:{line_no}")
            if pair_role is not None and type(pair_role) is not str:
                raise ValueError(f"pair_role must be string at {path}:{line_no}")
            records.append(PublicRecord(raw["example_id"], family, split, raw["source_text"], pair_id, pair_role))
    return records


def _available_files(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    manifest_path = dataset_dir / "manifest.json"
    names: Iterable[str]
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = manifest.get("files", {}).keys() if isinstance(manifest.get("files"), Mapping) else ()
    else:
        names = (path.name for path in sorted(dataset_dir.glob("*.jsonl")))
    result: list[tuple[str, str, Path]] = []
    for name in names:
        if not isinstance(name, str) or not name.endswith(".jsonl") or "-" not in name:
            continue
        family, split_ext = name.split("-", 1)
        split = split_ext[:-6]
        family = family.upper()
        if family not in {"ERE", "CPS"}:
            continue
        path = dataset_dir / name
        if path.is_file():
            result.append((family, split, path))
    return sorted(result, key=lambda row: (row[0], row[1]))


def load_public_source_dataset(root: Path, *, splits: Sequence[str] | None = None) -> dict[str, list[PublicRecord]]:
    """Load all C0R rows while exposing only the C1 public source surface."""
    wanted = None if splits is None else set(splits)
    dataset_dir = _dataset_dir(Path(root))
    result: dict[str, list[PublicRecord]] = {}
    for family, split, path in _available_files(dataset_dir):
        key = f"{family}/{split}"
        if wanted is not None and key not in wanted and split not in wanted:
            continue
        result[key] = _read_jsonl(path, family=family, split=split)
    if not result:
        raise ValueError("no C0R JSONL files matched requested public splits")
    return result


def load_public_records(root: Path, split: str | None = None) -> list[PublicRecord]:
    dataset = load_public_source_dataset(root, splits=None if split is None else (split,))
    rows = [row for key in sorted(dataset) if split is None or key == split or key.endswith("/" + split) for row in dataset[key]]
    if not rows:
        raise ValueError(f"no public records for split {split!r}")
    return rows


def load_records(dataset_root: Path, cells: Sequence[str] | None = None) -> list[PublicRecord]:
    """Canonical C1 runner entry point for the source-only record list."""
    dataset = load_public_source_dataset(dataset_root, splits=cells)
    return [row for key in sorted(dataset) for row in dataset[key]]


def records_by_cell(dataset_root: Path, cells: Sequence[str] | None = None) -> dict[str, list[PublicRecord]]:
    return load_public_source_dataset(dataset_root, splits=cells)


def public_source_view(record: Mapping[str, Any] | PublicRecord) -> dict[str, str]:
    """Return the sole model-facing value; metadata never reaches forward."""
    source = record.source_text if isinstance(record, PublicRecord) else record.get("source_text")
    if type(source) is not str:
        raise ValueError("record source_text must be a string")
    return {"source_text": source}


def public_record_dict(record: Mapping[str, Any] | PublicRecord) -> dict[str, Any]:
    if isinstance(record, PublicRecord):
        return record.as_dict()
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping or PublicRecord")
    forbidden = FORBIDDEN_PUBLIC_FIELDS.intersection(record)
    if forbidden:
        raise ValueError(f"public loader refuses forbidden fields: {sorted(forbidden)}")
    if type(record.get("source_text")) is not str or type(record.get("example_id")) is not str:
        raise ValueError("public record requires example_id and source_text")
    return {key: record[key] for key in PUBLIC_RECORD_FIELDS if key in record}


__all__ = [
    "C1_DATA_SCHEMA", "CT1_GRAMMAR_IDS", "FORBIDDEN_PUBLIC_FIELDS", "MAX_SOURCE_TOKENS",
    "PUBLIC_RECORD_FIELDS", "PublicRecord", "load_public_records", "load_public_source_dataset",
    "load_records", "records_by_cell", "public_record_dict", "public_source_view", "source_byte_sha256",
]
