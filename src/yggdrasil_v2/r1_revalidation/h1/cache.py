from __future__ import annotations

"""Frozen-Qwen contextual token cache for the P1-H1 comparison.

The cache stores only contextualized source tokens and a token mask.  Targets,
task/family identifiers, oracle spans, answers, and simulator state never enter
the persisted index or the model-facing tensors.
"""

from dataclasses import asdict, is_dataclass
import gc
import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from yggdrasil_v2.reasoning_medium.model import load_qwen35_text_only


CACHE_SCHEMA = "yggdrasil.v2-r1r.p1-h1.token-cache.v1"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MAX_SOURCE_TOKENS = 384
MAX_SOURCE_ATOMS = 40
HIDDEN_WIDTH = 2048
INFERENCE_BATCH_SIZE = 8
FORBIDDEN_CACHE_FIELDS = {
    "target",
    "targets",
    "family",
    "task",
    "task_id",
    "kind",
    "answer",
    "answer_index",
    "winner",
    "final",
    "prefix",
    "prefixes",
    "closure",
    "decision",
    "candidate_index",
    "oracle_span",
    "oracle_role",
    "simulator_state",
    "teacher_trace",
    "targets",
    "trace_targets",
    "type_target",
}


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def _record_mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    if is_dataclass(record):
        return asdict(record)
    if hasattr(record, "to_dict"):
        value = record.to_dict()
        if isinstance(value, Mapping):
            return value
    raise TypeError(f"unsupported H1 record type: {type(record)!r}")


def _record_source(record: Any) -> tuple[str, str, tuple[str, ...]]:
    row = _record_mapping(record)
    record_id = row.get("id")
    source_text = row.get("source_text")
    source_atoms = row.get("source_atoms")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("H1 cache record requires a non-empty id")
    if not isinstance(source_text, str) or not source_text or "\r" in source_text:
        raise ValueError(f"H1 cache record {record_id!r} has no source_text")
    if not isinstance(source_atoms, (list, tuple)) or not source_atoms:
        raise ValueError(f"H1 cache record {record_id!r} has no source_atoms")
    if len(source_atoms) > MAX_SOURCE_ATOMS:
        raise ValueError(f"H1 cache atom count exceeds {MAX_SOURCE_ATOMS}: {record_id}")
    if any(not isinstance(atom, str) for atom in source_atoms):
        raise ValueError(f"H1 cache atoms must be strings: {record_id}")
    atoms = tuple(source_atoms)
    if any(not atom.strip() or "\n" in atom or "\r" in atom for atom in atoms):
        raise ValueError(f"H1 cache atoms must be non-empty single lines: {record_id}")
    if len(set(atoms)) != len(atoms):
        raise ValueError(f"H1 cache atoms must not contain duplicates: {record_id}")
    if source_text != "\n".join(atoms):
        raise ValueError(f"H1 source_text must be the newline join of source_atoms: {record_id}")
    if len(atoms) > MAX_SOURCE_ATOMS:
        raise ValueError(f"H1 source atom count exceeds {MAX_SOURCE_ATOMS}: {record_id}")
    return record_id, source_text, atoms


def _line_spans(atoms: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for atom in atoms:
        stop = cursor + len(atom)
        spans.append((cursor, stop))
        cursor = stop + 1
    return spans


def _tokenize_with_atom_membership(
    tokenizer: Any,
    text: str,
    atoms: Sequence[str],
) -> tuple[list[int], list[list[int]]]:
    tokenizer_kwargs = {
        "add_special_tokens": True,
        "truncation": False,
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
    }
    # Keep lightweight test/double tokenizers compatible without a retry that
    # would violate the one-encode-per-source contract.
    try:
        signature = inspect.signature(tokenizer)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs and "return_special_tokens_mask" not in signature.parameters:
            tokenizer_kwargs.pop("return_special_tokens_mask")
    except (TypeError, ValueError):
        pass
    encoded = tokenizer(text, **tokenizer_kwargs)
    try:
        input_ids = [int(value) for value in encoded["input_ids"]]
        raw_offsets = encoded["offset_mapping"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fast tokenizer must return input_ids and offset_mapping") from exc
    try:
        offsets = [(int(pair[0]), int(pair[1])) for pair in raw_offsets]
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("fast tokenizer returned invalid offset mapping") from exc
    if not input_ids or len(input_ids) != len(offsets):
        raise ValueError("tokenizer returned invalid ids/offsets")
    if len(input_ids) > MAX_SOURCE_TOKENS:
        raise ValueError(
            f"H1 cache refuses silent truncation: {len(input_ids)} > {MAX_SOURCE_TOKENS}"
        )
    special_mask = encoded.get("special_tokens_mask")
    if special_mask is None:
        special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        special_mask = [int(token_id in special_ids) for token_id in input_ids]
    if len(special_mask) != len(input_ids):
        raise ValueError("tokenizer returned invalid special-token mask")
    valid_offsets: list[tuple[int, int]] = []
    for index, (left, right) in enumerate(offsets):
        # Special tokens are not source characters. Do not trust a backend
        # that gives a special token a non-zero fake offset.
        if bool(special_mask[index]) or (left == 0 and right == 0):
            valid_offsets.append((0, 0))
            continue
        if left < 0 or right <= left or right > len(text):
            raise ValueError("fast tokenizer returned an out-of-range offset")
        valid_offsets.append((left, right))

    memberships: list[list[int]] = []
    for atom_left, atom_right in _line_spans(atoms):
        indices = [
            index
            for index, (token_left, token_right) in enumerate(valid_offsets)
            if token_right > token_left
            and token_right > atom_left
            and token_left < atom_right
        ]
        if not indices:
            raise ValueError(
                f"no contextual tokens overlap source atom span {atom_left}:{atom_right}"
            )
        memberships.append(indices)
    return input_ids, memberships


def _last_hidden_state(
    backbone: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the text decoder across common Transformers wrapper shapes."""
    decoder = getattr(backbone, "model", None)
    if decoder is None or not callable(decoder):
        decoder = backbone
    if not callable(decoder):
        raise TypeError("Qwen backbone has no callable text decoder")
    output = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("Qwen text decoder did not return a rank-3 last_hidden_state")
    return hidden


def _gpu_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "peak_allocated": 0, "peak_reserved": 0}
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }


def build_token_cache(
    records: Sequence[Any],
    output_root: Path,
    *,
    device: str = "cuda",
    batch_size: int = INFERENCE_BATCH_SIZE,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    """Encode each full source once and persist every contextual token."""
    if not records:
        raise ValueError("H1 token-cache records cannot be empty")
    if batch_size < 1:
        raise ValueError("H1 cache batch_size must be positive")
    if output_root.exists():
        raise FileExistsError(f"H1 cache root already exists: {output_root}")

    normalized = [_record_source(record) for record in records]
    ids = [record_id for record_id, _, _ in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("H1 cache record ids must be unique")
    output_root.mkdir(parents=True, exist_ok=False)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        local_files_only=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("H1 contextual token caching requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized = [
        _tokenize_with_atom_membership(tokenizer, text, atoms)
        for _, text, atoms in normalized
    ]
    token_lengths = [len(tokens) for tokens, _ in tokenized]
    atom_counts = [len(memberships) for _, memberships in tokenized]
    maximum_atoms = max(atom_counts)
    maximum_tokens = max(token_lengths)

    load_started = time.perf_counter()
    backbone = load_qwen35_text_only(
        model_id,
        model_revision,
        dtype=torch.float16,
        device=device,
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model_load_seconds = time.perf_counter() - load_started
    hidden_width = int(backbone.config.hidden_size)
    if hidden_width != HIDDEN_WIDTH:
        raise ValueError(f"H1 cache requires Qwen hidden width {HIDDEN_WIDTH}, got {hidden_width}")

    hidden_path = output_root / "source-tokens-hidden.npy"
    mask_path = output_root / "source-tokens-mask.npy"
    hidden = np.lib.format.open_memmap(
        hidden_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), maximum_tokens, hidden_width),
    )
    mask = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(records), maximum_tokens),
    )
    hidden[:] = 0
    mask[:] = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    encode_started = time.perf_counter()
    for start in range(0, len(records), batch_size):
        batch = tokenized[start : start + batch_size]
        batch_max = max(len(tokens) for tokens, _ in batch)
        input_ids = torch.full(
            (len(batch), batch_max),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(batch), batch_max), dtype=torch.bool)
        for row_index, (tokens, _) in enumerate(batch):
            input_ids[row_index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row_index, : len(tokens)] = True
        with torch.inference_mode():
            output = _last_hidden_state(
                backbone,
                input_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
            )
            if output.shape[1] != batch_max or output.shape[2] != hidden_width:
                raise ValueError("Qwen text decoder returned an unexpected hidden shape")
        for row_index, (tokens, _) in enumerate(batch):
            absolute_index = start + row_index
            length = len(tokens)
            hidden[absolute_index, :length] = (
                output[row_index, :length].to(torch.float16).cpu().numpy()
            )
            mask[absolute_index, :length] = True
        del input_ids, attention_mask, output

    encode_seconds = time.perf_counter() - encode_started
    hidden.flush()
    mask.flush()
    index = {
        "schema_version": CACHE_SCHEMA,
        "ids": ids,
        "source_sha256": {
            record_id: _sha256_text(text)
            for record_id, text, _ in normalized
        },
        "token_lengths": token_lengths,
        "atom_counts": atom_counts,
        "hidden_shape": list(hidden.shape),
        "mask_shape": list(mask.shape),
        "hidden_dtype": str(hidden.dtype),
        "mask_dtype": str(mask.dtype),
    }
    index_path = output_root / "source-tokens-index.json"
    index_path.write_bytes(_canonical(index))
    del hidden, mask

    manifest = {
        "schema_version": CACHE_SCHEMA,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "contextualization": "full source Qwen final hidden; every contextual token retained",
        "newline_atoms_are_audit_only": True,
        "record_count": len(records),
        "hidden_width": hidden_width,
        "maximum_tokens": maximum_tokens,
        "maximum_atoms": maximum_atoms,
        "total_tokens": sum(token_lengths),
        "total_atoms": sum(atom_counts),
        "silent_truncation": False,
        "qwen_trainable_parameters": sum(
            parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
        ),
        "model_load_seconds": model_load_seconds,
        "encode_seconds": encode_seconds,
        "batch_size": batch_size,
        "hidden_file": hidden_path.name,
        "mask_file": mask_path.name,
        "index_file": index_path.name,
        "hidden_sha256": _sha256(hidden_path),
        "mask_sha256": _sha256(mask_path),
        "index_sha256": _sha256(index_path),
        "gpu_memory": _gpu_memory(),
    }
    manifest_path = output_root / "cache-manifest.json"
    manifest_path.write_bytes(_canonical(manifest))
    del backbone, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def _forbidden_hits(*values: Any) -> list[str]:
    hits: set[str] = set()
    for value in values:
        for key in _walk_keys(value):
            normalized = key.lower().replace("-", "_")
            if normalized in FORBIDDEN_CACHE_FIELDS or set(normalized.split("_")) & FORBIDDEN_CACHE_FIELDS:
                hits.add(key)
    return sorted(hits)


def _cache_path(root: Path, filename: Any) -> Path | None:
    if not isinstance(filename, str) or not filename:
        return None
    root = root.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def audit_token_cache(
    records: Sequence[Any],
    cache_root: Path,
    *,
    verify_all_finite: bool = True,
) -> dict[str, Any]:
    manifest_path = cache_root / "cache-manifest.json"
    if not manifest_path.is_file():
        return {"passed": False, "reason": "cache manifest missing"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return {"passed": False, "reason": f"invalid cache manifest: {exc}"}
    if not isinstance(manifest, Mapping):
        return {"passed": False, "reason": "cache manifest must be an object"}
    index_path = _cache_path(cache_root, manifest.get("index_file"))
    hidden_path = _cache_path(cache_root, manifest.get("hidden_file"))
    mask_path = _cache_path(cache_root, manifest.get("mask_file"))
    if not all(path is not None and path.is_file() for path in (index_path, hidden_path, mask_path)):
        return {"passed": False, "reason": "cache component missing"}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        return {"passed": False, "reason": f"invalid cache index: {exc}"}
    if not isinstance(index, Mapping):
        return {"passed": False, "reason": "cache index must be an object"}
    forbidden_hits = _forbidden_hits(manifest, index)
    try:
        normalized = [_record_source(record) for record in records]
    except (TypeError, ValueError) as exc:
        return {"passed": False, "reason": str(exc), "forbidden_field_hits": forbidden_hits}
    expected_ids = [record_id for record_id, _, _ in normalized]
    hashes = {
        "hidden": _sha256(hidden_path) == manifest.get("hidden_sha256"),
        "mask": _sha256(mask_path) == manifest.get("mask_sha256"),
        "index": _sha256(index_path) == manifest.get("index_sha256"),
    }
    sources = {
        record_id: _sha256_text(text)
        for record_id, text, _ in normalized
    }
    try:
        hidden = np.load(hidden_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        return {"passed": False, "reason": f"invalid cache arrays: {exc}", "forbidden_field_hits": forbidden_hits}
    expected_atom_counts = [len(atoms) for _, _, atoms in normalized]
    ids = index.get("ids")
    atom_counts = index.get("atom_counts")
    token_lengths = index.get("token_lengths")
    hidden_shape = index.get("hidden_shape")
    mask_shape = index.get("mask_shape")
    shapes = (
        list(hidden.shape) == list(hidden_shape or [])
        and list(mask.shape) == list(mask_shape or [])
        and tuple(mask.shape) == tuple(hidden.shape[:2])
        and hidden.ndim == 3
        and mask.ndim == 2
        and hidden.shape[0] == len(expected_ids)
        and hidden.shape[1] <= MAX_SOURCE_TOKENS
        and hidden.shape[2] == HIDDEN_WIDTH
        and hidden.dtype == np.dtype(np.float16)
        and mask.dtype == np.dtype(np.bool_)
    )
    mask_counts = mask.sum(axis=1).astype(np.int64).tolist() if mask.ndim == 2 else []
    mask_layout = False
    valid_token_lengths = (
        isinstance(token_lengths, list)
        and len(token_lengths) == len(expected_ids)
        and all(type(value) is int and 0 < value <= MAX_SOURCE_TOKENS for value in token_lengths)
    )
    if mask.ndim == 2 and valid_token_lengths:
        expected_mask = np.zeros(mask.shape, dtype=np.bool_)
        for row, count in enumerate(token_lengths):
            if count <= mask.shape[1]:
                expected_mask[row, :count] = True
        mask_layout = np.array_equal(np.asarray(mask), expected_mask)
    finite = True
    if verify_all_finite and hidden.ndim == 3:
        for start in range(0, hidden.shape[0], 256):
            block = np.asarray(hidden[start : start + 256])
            if not np.isfinite(block).all():
                finite = False
                break
    checks = {
        "schema": manifest.get("schema_version") == CACHE_SCHEMA == index.get("schema_version"),
        "model": manifest.get("model_id") == MODEL_ID,
        "revision": manifest.get("model_revision") == MODEL_REVISION,
        "ids": (
            isinstance(ids, list)
            and all(isinstance(value, str) for value in ids)
            and ids == expected_ids
            and len(set(ids)) == len(expected_ids)
        ),
        "sources": index.get("source_sha256") == sources,
        "no_forbidden_fields": not forbidden_hits,
        "hashes": all(hashes.values()),
        "shapes": shapes,
        "atom_counts": atom_counts == expected_atom_counts,
        "token_counts": valid_token_lengths and token_lengths == mask_counts,
        "mask_layout": mask_layout,
        "finite": finite,
        "frozen_qwen": manifest.get("qwen_trainable_parameters") == 0,
        "no_truncation": manifest.get("silent_truncation") is False,
        "record_count": manifest.get("record_count") == len(expected_ids),
        "width": manifest.get("hidden_width") == HIDDEN_WIDTH,
        "dtypes": index.get("hidden_dtype") == "float16" and index.get("mask_dtype") == "bool",
        "limits": (
            isinstance(manifest.get("maximum_tokens"), int)
            and manifest["maximum_tokens"] <= MAX_SOURCE_TOKENS
            and isinstance(manifest.get("maximum_atoms"), int)
            and manifest["maximum_atoms"] <= MAX_SOURCE_ATOMS
        ),
        "totals": (
            valid_token_lengths
            and manifest.get("total_tokens") == sum(token_lengths)
            and manifest.get("total_atoms") == sum(expected_atom_counts)
            and manifest.get("maximum_tokens") == max(token_lengths)
            and manifest.get("maximum_atoms") == max(expected_atom_counts)
        ),
    }
    return {
        "schema_version": CACHE_SCHEMA,
        "checks": checks,
        "hash_checks": hashes,
        "forbidden_field_hits": forbidden_hits,
        "record_count": len(expected_ids),
        "total_tokens": manifest.get("total_tokens"),
        "total_atoms": manifest.get("total_atoms"),
        "passed": all(checks.values()),
    }


class TokenCache:
    """Read-only mmap cache with deterministic batch materialization."""

    def __init__(self, root: Path) -> None:
        root = Path(root)
        manifest = json.loads((root / "cache-manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or _forbidden_hits(manifest):
            raise ValueError("H1 cache manifest is invalid or contains forbidden fields")
        index_path = _cache_path(root, manifest.get("index_file"))
        hidden_path = _cache_path(root, manifest.get("hidden_file"))
        mask_path = _cache_path(root, manifest.get("mask_file"))
        if not all(path is not None and path.is_file() for path in (index_path, hidden_path, mask_path)):
            raise ValueError("H1 cache component is missing or outside cache root")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, Mapping) or _forbidden_hits(index):
            raise ValueError("H1 cache index is invalid or contains forbidden fields")
        if manifest.get("schema_version") != CACHE_SCHEMA or index.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("unsupported H1 cache schema")
        if not isinstance(index.get("ids"), list) or any(not isinstance(value, str) for value in index["ids"]):
            raise ValueError("H1 cache index ids are invalid")
        if len(set(index["ids"])) != len(index["ids"]):
            raise ValueError("duplicate ids in H1 cache index")
        if _sha256(index_path) != manifest.get("index_sha256"):
            raise ValueError("H1 cache index hash mismatch")
        if _sha256(hidden_path) != manifest.get("hidden_sha256") or _sha256(mask_path) != manifest.get("mask_sha256"):
            raise ValueError("H1 cache array hash mismatch")
        self.root = root
        self.manifest = manifest
        self.index = index
        self.hidden = np.load(hidden_path, mmap_mode="r", allow_pickle=False)
        self.mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        if (
            self.hidden.ndim != 3
            or self.mask.ndim != 2
            or self.hidden.dtype != np.dtype(np.float16)
            or self.mask.dtype != np.dtype(np.bool_)
            or tuple(self.mask.shape) != tuple(self.hidden.shape[:2])
            or self.hidden.shape[0] != len(index["ids"])
            or self.hidden.shape[2] != HIDDEN_WIDTH
        ):
            raise ValueError("H1 cache array shape or dtype is invalid")
        self.id_to_index = {record_id: index for index, record_id in enumerate(index["ids"])}

    def indices(self, record_ids: Sequence[str]) -> np.ndarray:
        try:
            return np.asarray([self.id_to_index[value] for value in record_ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"unknown H1 cache record id: {exc.args[0]}") from exc

    def batch(
        self,
        indices: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(indices, torch.Tensor):
            selected = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("H1 cache batch indices must be one-dimensional")
        if np.any(selected < 0) or np.any(selected >= len(self.id_to_index)):
            raise IndexError("H1 cache batch index is out of range")
        hidden = torch.from_numpy(np.array(self.hidden[selected], copy=True)).to(
            device, non_blocking=True
        )
        mask = torch.from_numpy(np.array(self.mask[selected], copy=True)).to(
            device, non_blocking=True
        )
        return hidden, mask


__all__ = [
    "TokenCache",
    "CACHE_SCHEMA",
    "FORBIDDEN_CACHE_FIELDS",
    "INFERENCE_BATCH_SIZE",
    "MAX_SOURCE_ATOMS",
    "MAX_SOURCE_TOKENS",
    "MODEL_ID",
    "MODEL_REVISION",
    "audit_token_cache",
    "build_token_cache",
]
