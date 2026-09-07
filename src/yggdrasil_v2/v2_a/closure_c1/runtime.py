from __future__ import annotations

"""C1 offline ledger, CT1 target bank, and deterministic batch boundary.

This module is the only bridge between immutable C0R records and training
metadata.  Full records are retained in this offline ledger for evaluation
and answer supervision; the cache dataset is queried by opaque example IDs
and receives no AST, trace, family, or answer fields.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .cache import C1_CACHE_MANIFEST_SCHEMA, CachedShardDataset, _load_shard
from .data import CT1_GRAMMAR_IDS, source_byte_sha256
from .trace_targets import TraceTarget, materialize_trace_target, trace_text_for_record


RUNTIME_SCHEMA = "yggdrasil.v2-a.closure-c1.runtime.v1"
LEXICON_SCHEMA = "yggdrasil.v2-a.closure-c1.trace-lexicon.v1"
TARGET_BANK_SCHEMA = "yggdrasil.v2-a.closure-c1.trace-target-bank.v1"
TRACE_CHUNK_LIMIT = 64
ANSWER_LABELS = tuple("ABCDEFGHI")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical(value).encode("utf-8"))


def _dataset_dir(root: Path) -> Path:
    root = Path(root)
    if (root / "dataset").is_dir():
        return root / "dataset"
    if (root / "manifest.json").is_file():
        return root
    raise FileNotFoundError(f"C0R dataset directory not found: {root}")


@dataclass(frozen=True)
class OfflineRecordStore:
    """Full C0R JSONL ledger, indexed by cell and opaque example ID."""

    records_by_cell: Mapping[str, tuple[dict[str, Any], ...]]
    records_by_id: Mapping[str, dict[str, Any]]
    dataset_root: str
    manifest_sha256: str | None = None

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for cell in sorted(self.records_by_cell) for row in self.records_by_cell[cell])

    def __len__(self) -> int:
        return len(self.records_by_id)

    def __getitem__(self, example_id: str) -> dict[str, Any]:
        try:
            return self.records_by_id[str(example_id)]
        except KeyError as exc:
            raise KeyError(f"offline C0R record not found: {example_id}") from exc

    def cell(self, cell: str) -> tuple[dict[str, Any], ...]:
        return self.records_by_cell.get(str(cell), ())

    def answer_index(self, example_id: str) -> int:
        record = self[example_id]
        value = record.get("answer_index")
        if type(value) is int and 0 <= value < len(ANSWER_LABELS):
            return value
        label = record.get("answer_label", record.get("target_label", record.get("answer")))
        if isinstance(label, str) and label in ANSWER_LABELS:
            return ANSWER_LABELS.index(label)
        # C0R stores semantic answers plus an explicit label mapping.  This
        # resolution is offline-only and never enters cache/model forward.
        semantic = record.get("semantic_answer")
        mapping = record.get("label_mapping")
        if isinstance(mapping, Mapping):
            matches = [label for label, target in mapping.items() if target == semantic]
            if len(matches) == 1 and matches[0] in ANSWER_LABELS:
                return ANSWER_LABELS.index(matches[0])
        raise ValueError(f"cannot resolve A-I answer for {example_id}")


def _jsonl_files(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    manifest_path = dataset_dir / "manifest.json"
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
        family = family.upper()
        split = split_ext[:-6]
        if family in {"ERE", "CPS"} and (dataset_dir / name).is_file():
            result.append((family, split, dataset_dir / name))
    return sorted(result, key=lambda row: (row[0], row[1]))


def load_offline_records(dataset_root: Path, cells: Sequence[str] | None = None) -> OfflineRecordStore:
    """Read complete C0R records; no model-facing view is returned."""
    dataset_dir = _dataset_dir(Path(dataset_root))
    wanted = None if cells is None else set(str(cell) for cell in cells)
    by_cell: dict[str, tuple[dict[str, Any], ...]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for family, split, path in _jsonl_files(dataset_dir):
        cell = f"{family}/{split}"
        if wanted is not None and cell not in wanted and split not in wanted:
            continue
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict) or raw.get("family") != family:
                    raise ValueError(f"C0R family mismatch at {path}:{line_no}")
                if type(raw.get("example_id")) is not str or type(raw.get("source_text")) is not str:
                    raise ValueError(f"C0R record lacks identity/source at {path}:{line_no}")
                if raw["example_id"] in by_id:
                    raise ValueError(f"duplicate C0R example_id: {raw['example_id']}")
                raw = dict(raw)
                raw.setdefault("split", split)
                rows.append(raw)
                by_id[raw["example_id"]] = raw
        by_cell[cell] = tuple(rows)
    if not by_cell:
        raise ValueError("no C0R cells matched")
    manifest = dataset_dir / "manifest.json"
    manifest_hash = _sha256_bytes(manifest.read_bytes()) if manifest.is_file() else None
    return OfflineRecordStore(by_cell, by_id, str(dataset_dir.resolve()), manifest_hash)


load_full_records = load_offline_records


def _cache_root(dataset: Any) -> Path:
    root = getattr(dataset, "root", dataset)
    return Path(root)


def validate_cache_source_identity(dataset: Any, store: OfflineRecordStore) -> dict[str, Any]:
    """Fail closed if cache IDs or source byte hashes differ from C0R JSONL."""
    root = _cache_root(dataset)
    failures: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"schema_version": RUNTIME_SCHEMA, "passed": False, "failures": [f"manifest:{exc}"]}
    if manifest.get("schema_version") != C1_CACHE_MANIFEST_SCHEMA:
        failures.append("cache_manifest_schema")
    observed: dict[str, str] = {}
    try:
        for shard in manifest.get("shards", []):
            payload = _load_shard(root / str(shard["name"]))
            ids = [str(value) for value in payload["example_ids"]]
            hashes = [str(value) for value in payload["source_hashes"]]
            if len(ids) != len(hashes):
                failures.append(f"row_hash_length:{shard['name']}")
            for example_id, source_hash in zip(ids, hashes):
                if example_id in observed:
                    failures.append(f"duplicate_cache_id:{example_id}")
                observed[example_id] = source_hash
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        failures.append(f"cache_payload:{type(exc).__name__}:{exc}")
    expected = {example_id: source_byte_sha256(row["source_text"]) for example_id, row in store.records_by_id.items()}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing:
        failures.append(f"missing_cache_ids:{missing[:8]}")
    if extra:
        failures.append(f"extra_cache_ids:{extra[:8]}")
    mismatched = sorted(example_id for example_id in set(expected) & set(observed) if expected[example_id] != observed[example_id])
    if mismatched:
        failures.append(f"source_hash_mismatch:{mismatched[:8]}")
    return {"schema_version": RUNTIME_SCHEMA, "cache_root": str(root), "expected_examples": len(expected), "observed_examples": len(observed), "failures": failures, "passed": not failures}


validate_cache_identity = validate_cache_source_identity


def _cache_source_hashes(dataset: Any) -> dict[str, str]:
    direct = getattr(dataset, "source_hashes", None)
    if isinstance(direct, Mapping):
        return {str(key): str(value) for key, value in direct.items()}
    root = _cache_root(dataset)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for shard in manifest.get("shards", []):
        payload = _load_shard(root / str(shard["name"]))
        for example_id, source_hash in zip(payload.get("example_ids", []), payload.get("source_hashes", [])):
            result[str(example_id)] = str(source_hash)
    return result


@dataclass(frozen=True)
class TraceLexicon:
    token_ids: tuple[int, ...]
    grammar_ids: tuple[int, ...] = CT1_GRAMMAR_IDS

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.token_ids))) != self.token_ids:
            raise ValueError("trace lexicon token IDs must be sorted and unique")
        if not set(self.grammar_ids).issubset(self.token_ids):
            raise ValueError("all fixed CT1 grammar IDs must be in trace lexicon")

    @property
    def size(self) -> int:
        return len(self.token_ids)

    @property
    def id_to_local(self) -> dict[int, int]:
        return {token_id: index for index, token_id in enumerate(self.token_ids)}

    @property
    def local_to_id(self) -> tuple[int, ...]:
        return self.token_ids

    @property
    def digest(self) -> str:
        return _sha256_json({"token_ids": list(self.token_ids), "grammar_ids": list(self.grammar_ids)})

    def local_id(self, token_id: int) -> int:
        try:
            return self.id_to_local[int(token_id)]
        except KeyError as exc:
            raise ValueError(f"trace token OOV: {token_id}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": LEXICON_SCHEMA, "token_ids": list(self.token_ids), "grammar_ids": list(self.grammar_ids), "vocab_size": self.size, "sha256": self.digest}


def build_trace_lexicon(records: Sequence[Mapping[str, Any]], tokenizer: Any, *, strict_size: bool = False) -> TraceLexicon:
    ids: set[int] = set(CT1_GRAMMAR_IDS)
    for record in records:
        source = record.get("source_text")
        if type(source) is not str:
            raise ValueError("trace lexicon source_text must be a string")
        encoded = tokenizer(source, add_special_tokens=False, truncation=False)
        values = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        ids.update(int(value) for value in values)
    lexicon = TraceLexicon(tuple(sorted(ids)))
    if strict_size and lexicon.size != 5597:
        raise ValueError(f"C1 trace lexicon expected 5597 IDs, got {lexicon.size}")
    return lexicon


@dataclass(frozen=True)
class TraceTargetBank:
    lexicon: TraceLexicon
    targets: Mapping[str, Mapping[str, Any]]

    @property
    def digest(self) -> str:
        return _sha256_json(self.payload())

    def payload(self) -> dict[str, Any]:
        return {"schema_version": TARGET_BANK_SCHEMA, "lexicon": self.lexicon.as_dict(), "targets": {key: self.targets[key] for key in sorted(self.targets)}}

    def as_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["sha256"] = self.digest
        return payload

    def __len__(self) -> int:
        return len(self.targets)


def materialize_target_bank(records: Sequence[Mapping[str, Any]], tokenizer: Any, lexicon: TraceLexicon) -> TraceTargetBank:
    targets: dict[str, dict[str, Any]] = {}
    mapping = lexicon.id_to_local
    for record in records:
        example_id = str(record.get("example_id", ""))
        if not example_id or example_id in targets:
            raise ValueError(f"duplicate or empty target example_id: {example_id}")
        target = materialize_trace_target(record, tokenizer, lexicon.token_ids)
        global_ids = list(target.token_ids)
        try:
            local_ids = [mapping[token_id] for token_id in global_ids]
        except KeyError as exc:
            raise ValueError(f"trace target OOV token id: {exc.args[0]}") from exc
        target_row = {
            "example_id": example_id,
            "text": target.text,
            "global_token_ids": global_ids,
            "token_ids": local_ids,
            "step_indices": list(target.step_ids),
            "global_positions": list(range(len(global_ids))),
            "local_positions": list(target.local_positions),
            "grammar_mask": [int(token_id) in set(lexicon.grammar_ids) for token_id in global_ids],
        }
        targets[example_id] = target_row
    return TraceTargetBank(lexicon, targets)


materialize_trace_target_bank = materialize_target_bank
build_lexicon = build_trace_lexicon


def save_target_bank(bank: TraceTargetBank, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bank.as_dict()
    path.write_text(_canonical(payload) + "\n", encoding="utf-8")
    return str(payload["sha256"])


def load_target_bank(path: Path) -> TraceTargetBank:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = raw.pop("sha256", None)
    actual = _sha256_json(raw)
    if expected != actual:
        raise ValueError("trace target bank hash mismatch")
    lex_raw = raw.get("lexicon", {})
    lexicon = TraceLexicon(tuple(int(value) for value in lex_raw["token_ids"]), tuple(int(value) for value in lex_raw["grammar_ids"]))
    if lex_raw.get("sha256") != lexicon.digest:
        raise ValueError("trace lexicon hash mismatch")
    return TraceTargetBank(lexicon, raw.get("targets", {}))


def audit_target_bank(bank_or_path: TraceTargetBank | Path, lexicon: TraceLexicon | None = None) -> dict[str, Any]:
    failures: list[str] = []
    try:
        bank = load_target_bank(bank_or_path) if isinstance(bank_or_path, (str, Path)) else bank_or_path
        if lexicon is not None and bank.lexicon.digest != lexicon.digest:
            failures.append("lexicon_mismatch")
        grammar = set(bank.lexicon.grammar_ids)
        # Materialise this once for the whole audit.  ``TraceLexicon.local_id``
        # exposes the same mapping through a convenience property, but calling
        # it for every target token would rebuild the full vocabulary-sized
        # dictionary millions of times on a production bank.
        global_to_local = bank.lexicon.id_to_local
        for example_id, row in bank.targets.items():
            fields = ("token_ids", "global_token_ids", "step_indices", "global_positions", "local_positions", "grammar_mask")
            if any(field not in row for field in fields):
                failures.append(f"missing_fields:{example_id}")
                continue
            size = len(row["token_ids"])
            if any(len(row[field]) != size for field in fields):
                failures.append(f"lengths:{example_id}")
            if any(type(value) is not int or not 0 <= value < bank.lexicon.size for value in row["token_ids"]):
                failures.append(f"local_vocab:{example_id}")
            else:
                expected_local = [global_to_local[int(value)] for value in row["global_token_ids"]]
                if list(row["token_ids"]) != expected_local:
                    failures.append(f"local_mapping:{example_id}")
            if any(type(value) is not int or not 1 <= value <= 10 for value in row["step_indices"]):
                failures.append(f"step_indices:{example_id}")
            if any(type(value) is not int or value < 0 or value >= 1024 for value in row["global_positions"]):
                failures.append(f"position_bounds:{example_id}")
            if row["global_positions"] != list(range(size)):
                failures.append(f"global_positions:{example_id}")
            expected_grammar = [int(value) in grammar for value in row["global_token_ids"]]
            if list(row["grammar_mask"]) != expected_grammar:
                failures.append(f"grammar_mask:{example_id}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        failures.append(f"load:{type(exc).__name__}:{exc}")
        bank = None
    return {"schema_version": TARGET_BANK_SCHEMA, "examples": len(bank) if bank is not None else 0, "failures": failures, "passed": not failures}


save_trace_target_bank = save_target_bank
load_trace_target_bank = load_target_bank
audit_trace_target_bank = audit_target_bank


def _chunk_start(example_id: str, epoch: int, length: int, chunk: int) -> int:
    if length <= chunk:
        return 0
    digest = hashlib.sha256(f"C1-CT1-CHUNK-v1|{epoch}|{example_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (length - chunk + 1)


class C1BatchProvider:
    """Build the exact eight train.py fields from offline records + cache."""

    def __init__(self, dataset: CachedShardDataset, records: OfflineRecordStore | Mapping[str, Mapping[str, Any]], targets: TraceTargetBank | Mapping[str, Mapping[str, Any]], *, chunk_seed: str = "C1-CT1-CHUNK-v1") -> None:
        self.dataset = dataset
        self.records = records
        self.targets = targets.targets if isinstance(targets, TraceTargetBank) else targets
        self.chunk_seed = str(chunk_seed)
        self._cache_source_hashes = _cache_source_hashes(dataset)
        if isinstance(records, OfflineRecordStore):
            self.store = records
        else:
            self.store = None
        if self.store is not None:
            root = _cache_root(dataset)
            if (root / "manifest.json").is_file():
                identity = validate_cache_source_identity(dataset, self.store)
                if not identity["passed"]:
                    raise ValueError(f"C1 cache/source identity audit failed: {identity['failures'][:4]}")

    def _record(self, example_id: str) -> Mapping[str, Any]:
        if isinstance(self.records, OfflineRecordStore):
            return self.records[example_id]
        try:
            return self.records[str(example_id)]
        except KeyError as exc:
            raise KeyError(f"offline record not found: {example_id}") from exc

    def __call__(self, example_ids: Sequence[str], *, epoch: int, trace_chunk_tokens: int = TRACE_CHUNK_LIMIT) -> dict[str, torch.Tensor]:
        ids = [str(value) for value in example_ids]
        if not ids:
            raise ValueError("C1 batch cannot be empty")
        requested = int(trace_chunk_tokens)
        if not 1 <= requested <= 512:
            raise ValueError("trace_chunk_tokens must be in [1,512]")
        cache_batch = self.dataset.get_batch(ids)
        source_inputs = cache_batch[0]
        rows: list[Mapping[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        for example_id in ids:
            record = self._record(example_id)
            if source_byte_sha256(str(record["source_text"])) != self._cache_hash(example_id):
                raise ValueError(f"source identity mismatch for {example_id}")
            target = self.targets.get(example_id)
            if target is None:
                raise KeyError(f"trace target not found: {example_id}")
            chunk_spec = self.trace_chunk_metadata(
                example_id, epoch=epoch, trace_chunk_tokens=requested
            )
            start = int(chunk_spec["start"])
            stop = int(chunk_spec["stop"])
            chunks.append({key: list(target[key][start:stop]) for key in ("token_ids", "step_indices", "global_positions", "local_positions", "grammar_mask")})
            rows.append(record)
        max_trace = max(len(chunk["token_ids"]) for chunk in chunks)
        batch_size = len(ids)
        trace_targets = torch.zeros((batch_size, max_trace), dtype=torch.long)
        trace_mask = torch.zeros((batch_size, max_trace), dtype=torch.bool)
        step_indices = torch.ones((batch_size, max_trace), dtype=torch.long)
        global_positions = torch.zeros((batch_size, max_trace), dtype=torch.long)
        local_positions = torch.zeros((batch_size, max_trace), dtype=torch.long)
        grammar_mask = torch.zeros((batch_size, max_trace), dtype=torch.bool)
        for row, chunk in enumerate(chunks):
            length = len(chunk["token_ids"])
            trace_targets[row, :length] = torch.tensor(chunk["token_ids"], dtype=torch.long)
            trace_mask[row, :length] = True
            step_indices[row, :length] = torch.tensor(chunk["step_indices"], dtype=torch.long)
            global_positions[row, :length] = torch.tensor(chunk["global_positions"], dtype=torch.long)
            local_positions[row, :length] = torch.tensor(chunk["local_positions"], dtype=torch.long)
            grammar_mask[row, :length] = torch.tensor(chunk["grammar_mask"], dtype=torch.bool)
        answers = torch.tensor([self._answer_index(row) for row in rows], dtype=torch.long)
        source_mask = source_inputs.get("source_mask", source_inputs.get("source_attention_mask"))
        if not isinstance(source_mask, torch.Tensor):
            raise KeyError("cache batch lacks source_mask")
        return {
            "source_hidden": source_inputs["source_hidden"],
            "source_mask": source_mask,
            "answers": answers,
            "trace_targets": trace_targets,
            "trace_mask": trace_mask,
            "trace_step_indices": step_indices,
            "trace_global_positions": global_positions,
            "trace_local_positions": local_positions,
        }

    def trace_chunk_metadata(
        self, example_id: str, *, epoch: int, trace_chunk_tokens: int = TRACE_CHUNK_LIMIT
    ) -> dict[str, int | str]:
        """Return the exact deterministic primary-train trace exposure."""
        key = str(example_id)
        target = self.targets.get(key)
        if target is None:
            raise KeyError(f"trace target not found: {key}")
        requested = int(trace_chunk_tokens)
        if not 1 <= requested <= 512:
            raise ValueError("trace_chunk_tokens must be in [1,512]")
        length = len(target["token_ids"])
        if length < 1:
            raise ValueError(f"trace target is empty: {key}")
        start = _chunk_start(
            f"{self.chunk_seed}|{key}", int(epoch), length, requested
        )
        stop = min(length, start + requested)
        return {
            "example_id": key,
            "epoch": int(epoch),
            "target_tokens": length,
            "requested_tokens": requested,
            "start": start,
            "stop": stop,
            "exposed_tokens": stop - start,
        }

    def _answer_index(self, record: Mapping[str, Any]) -> int:
        if self.store is not None:
            return self.store.answer_index(str(record["example_id"]))
        value = record.get("answer_index")
        if type(value) is int and 0 <= value < 9:
            return value
        label = record.get("answer_label", record.get("target_label", record.get("answer")))
        if isinstance(label, str) and label in ANSWER_LABELS:
            return ANSWER_LABELS.index(label)
        raise ValueError(f"offline answer ledger missing for {record.get('example_id')}")

    def _cache_hash(self, example_id: str) -> str:
        try:
            return self._cache_source_hashes[example_id]
        except KeyError as exc:
            raise KeyError(f"cache source hash missing for {example_id}") from exc


__all__ = [
    "ANSWER_LABELS", "C1BatchProvider", "OfflineRecordStore", "TraceLexicon", "TraceTargetBank",
    "audit_target_bank", "audit_trace_target_bank", "build_lexicon", "build_trace_lexicon",
    "load_full_records", "load_offline_records", "load_target_bank", "load_trace_target_bank",
    "materialize_target_bank", "materialize_trace_target_bank", "save_target_bank",
    "save_trace_target_bank", "validate_cache_identity", "validate_cache_source_identity",
]
