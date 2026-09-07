from __future__ import annotations

"""Fresh C1 Qwen full-token hidden cache and indexed mmap loader.

The cache has a new identity and a deliberately narrow payload.  Token IDs
exist only for the transient encoder call; no input IDs, answer, AST, family
branch, or CT1 target is serialized in a hidden shard.
"""

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
import numpy as np

from . import contract
from .data import MAX_SOURCE_TOKENS, PublicRecord, load_records, public_record_dict, source_byte_sha256


C1_CACHE_IDENTITY = "V2-A-CLOSURE-C1-CACHE-20260825-1"
C1_CACHE_SCHEMA = "yggdrasil.v2-a.closure-c1.full-token-hidden-shard.v1"
C1_CACHE_MANIFEST_SCHEMA = "yggdrasil.v2-a.closure-c1.cache-manifest.v1"
C1_CACHE_AUDIT_SCHEMA = "yggdrasil.v2-a.closure-c1.cache-audit.v1"
QWEN_MODEL_ID = "Qwen/Qwen3.5-2B"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
HIDDEN_FIELD = "last_hidden"
FORBIDDEN_CACHE_FIELDS = frozenset({
    "input_ids", "inputs_embeds", "answer", "answer_index", "semantic_answer",
    "label_mapping", "valid_choice_mask", "program_ast", "ast", "family",
    "family_embedding", "reasoning_budget", "teacher_trace", "trace",
    "compact_trace", "claim", "training_claims", "role_mask", "span_mask",
    "source_span", "target", "target_buffer", "latent", "dense_state",
    "route_id", "projection_expert", "optimizer", "checkpoint",
})
ALLOWED_SHARD_FIELDS = frozenset({
    "schema_version", HIDDEN_FIELD, "attention_mask", "token_lengths",
    "example_ids", "source_hashes", "shard_identity", "encoder_identity",
    "tokenizer_identity", "packed_hidden", "packed_shape", "offsets",
})


def _require_transformers_runtime() -> str:
    import transformers

    version = str(transformers.__version__)
    if version != contract.TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"C1 requires transformers=={contract.TRANSFORMERS_VERSION}, got {version}"
        )
    return version


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest().upper()


def load_qwen35_text_only(
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    *,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cpu",
    local_files_only: bool = True,
) -> torch.nn.Module:
    """Load a fresh text-only Qwen model, never consulting the network/cache.

    ``local_files_only`` is intentionally explicit and defaults to true.  A
    caller asking for a non-local load must opt in, which keeps formal C1
    runs reproducible and prevents accidental model substitution.
    """
    # Qwen3.5-2B is published as a composite conditional-generation
    # checkpoint.  Loading it through an Auto causal-LM class would either
    # instantiate the vision tower or retain an LM head, both forbidden by the
    # C1 cache contract.  Materialize exactly the language-model subtree into
    # the native text backbone instead.
    _require_transformers_runtime()
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3_5TextModel

    composite_config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    text_config = composite_config.text_config
    index_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            revision=revision,
            local_files_only=local_files_only,
        )
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    prefix = "model.language_model."
    source_keys = sorted(key for key in weight_map if key.startswith(prefix))
    if not source_keys:
        raise ValueError("pinned Qwen checkpoint has no language-model subtree")
    shard_paths = {
        filename: Path(
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
        for filename in sorted({weight_map[key] for key in source_keys})
    }
    state: dict[str, torch.Tensor] = {}
    for filename, shard_path in shard_paths.items():
        keys = [key for key in source_keys if weight_map[key] == filename]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for source_key in keys:
                state[source_key.removeprefix(prefix)] = handle.get_tensor(source_key).to(dtype=dtype)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen3_5TextModel(text_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_qwen35_tokenizer(
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    *,
    local_files_only: bool = True,
) -> Any:
    """Load the pinned fast tokenizer without any network fallback."""
    _require_transformers_runtime()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if not getattr(tokenizer, "is_fast", True):
        raise TypeError("C1 requires a fast tokenizer for offset mapping")
    return tokenizer


def _token_ids(tokenizer: Any, source_text: str) -> list[int]:
    encoded = tokenizer(source_text, add_special_tokens=False, truncation=False)
    if isinstance(encoded, Mapping):
        ids = encoded.get("input_ids")
    else:
        ids = encoded
    if ids is None:
        raise TypeError("tokenizer did not return input_ids")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def _cache_view(record: Mapping[str, Any] | PublicRecord) -> dict[str, Any]:
    """Read only the immutable public fields, even when given a full C0R row."""
    if isinstance(record, PublicRecord):
        return record.as_dict()
    if not isinstance(record, Mapping) or type(record.get("source_text")) is not str or type(record.get("example_id")) is not str:
        raise ValueError("cache record requires example_id and source_text")
    return {key: record[key] for key in ("example_id", "family", "split", "pair_id", "pair_role", "source_text") if key in record}


def _forward_hidden(backbone: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # QwenForCausalLM exposes the text transformer as ``.model``.  A fake
    # backbone used by CPU tests may instead be directly callable.
    module = getattr(backbone, "model", None)
    if isinstance(module, torch.nn.Module):
        output = module(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    else:
        output = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, Mapping):
        hidden = output.get("last_hidden_state")
    if hidden is None and isinstance(output, (tuple, list)):
        hidden = output[0]
    if hidden is None and isinstance(output, torch.Tensor):
        hidden = output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("backbone did not return [batch, tokens, hidden] final hidden")
    return hidden


def _encode_batch(tokenizer: Any, records: Sequence[Mapping[str, Any] | PublicRecord], max_length: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    token_rows = [_token_ids(tokenizer, _cache_view(record)["source_text"]) for record in records]
    lengths = [len(row) for row in token_rows]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("C1 cache refuses empty source tokenization")
    if any(length > max_length for length in lengths):
        raise ValueError(f"source exceeds max_length={max_length}; truncation is forbidden")
    pad = getattr(tokenizer, "pad_token_id", 0)
    input_ids = torch.full((len(token_rows), max(lengths)), int(pad), dtype=torch.long)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, ids in enumerate(token_rows):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row, : len(ids)] = True
    return input_ids, mask, lengths


def _ordered(records: Sequence[Mapping[str, Any] | PublicRecord]) -> list[Mapping[str, Any] | PublicRecord]:
    return sorted(records, key=lambda row: (str(_cache_view(row).get("family", "")), str(_cache_view(row).get("split", "")), str(_cache_view(row)["example_id"])))


def build_cache(
    repo_root: Path | None,
    dataset_root: Path,
    output_root: Path,
    *,
    records: Sequence[Mapping[str, Any] | PublicRecord] | None = None,
    tokenizer: Any | None = None,
    backbone: Any | None = None,
    device: str | torch.device = "cpu",
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    max_length: int = MAX_SOURCE_TOKENS,
    inference_batch_size: int = 8,
    shard_size: int = 64,
) -> dict[str, Any]:
    """Build an immutable, fresh C1 hidden cache in sequential shards."""
    if max_length != MAX_SOURCE_TOKENS:
        raise ValueError(f"C1 contract fixes max_length={MAX_SOURCE_TOKENS}")
    if inference_batch_size <= 0 or shard_size <= 0:
        raise ValueError("batch and shard sizes must be positive")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite or resume C1 cache: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if records is None:
        records = load_records(Path(dataset_root))
    ordered = _ordered(records)
    if not ordered:
        raise ValueError("C1 cache requires records")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = load_qwen35_tokenizer(model_id, revision, local_files_only=True)
    if backbone is None:
        backbone = load_qwen35_text_only(model_id, revision, dtype=torch.float16, device=device, local_files_only=True)
    if hasattr(backbone, "eval"):
        backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    split_meta: dict[str, dict[str, Any]] = {}
    metadata_rows: list[dict[str, Any]] = []
    total_tokens = 0
    started = time.perf_counter()
    for shard_index, start in enumerate(range(0, len(ordered), shard_size)):
        shard_records = ordered[start : start + shard_size]
        hidden_rows: list[torch.Tensor] = []
        lengths_all: list[int] = []
        for batch_start in range(0, len(shard_records), inference_batch_size):
            batch = shard_records[batch_start : batch_start + inference_batch_size]
            input_ids, mask, lengths = _encode_batch(tokenizer, batch, max_length)
            with torch.inference_mode():
                hidden = _forward_hidden(backbone, input_ids.to(device), mask.to(device))
            for row, length in enumerate(lengths):
                hidden_rows.append(hidden[row, :length].detach().to(dtype=torch.float16, device="cpu"))
            lengths_all.extend(lengths)
            del input_ids, mask, hidden
        width = int(hidden_rows[0].shape[-1])
        ids: list[str] = []
        source_hashes: list[str] = []
        offsets: list[int] = []
        cursor = 0
        packed_path = output_root / f"shard_{shard_index:05d}.f16"
        packed = np.memmap(packed_path, mode="w+", dtype=np.float16, shape=(sum(lengths_all), width))
        for row, (hidden_row, record) in enumerate(zip(hidden_rows, shard_records)):
            length = int(hidden_row.shape[0])
            offsets.append(cursor)
            packed[cursor : cursor + length] = hidden_row.numpy()
            cursor += length
            public = _cache_view(record)
            ids.append(str(public["example_id"]))
            source_hashes.append(source_byte_sha256(public["source_text"]))
            metadata_rows.append({key: public[key] for key in ("example_id", "family", "split", "pair_id", "pair_role") if key in public})
        packed.flush()
        del packed
        payload = {
            "schema_version": C1_CACHE_SCHEMA,
            "packed_hidden": packed_path.name,
            "packed_shape": [sum(lengths_all), width],
            "offsets": offsets,
            "attention_mask": torch.ones(sum(lengths_all), dtype=torch.bool),
            "token_lengths": torch.tensor(lengths_all, dtype=torch.long),
            "example_ids": ids,
            "source_hashes": source_hashes,
            "shard_identity": f"{C1_CACHE_IDENTITY}:{shard_index:05d}",
            "encoder_identity": {"model_id": model_id, "revision": revision, "layer": "final_hidden", "dtype": "float16"},
            "tokenizer_identity": {"model_id": model_id, "revision": revision, "add_special_tokens": False, "use_fast": True},
        }
        path = output_root / f"shard_{shard_index:05d}.pt"
        torch.save(payload, path)
        cell = f"{public_record_dict(shard_records[0]).get('family', '')}/{public_record_dict(shard_records[0]).get('split', '')}"
        cell_report = split_meta.setdefault(cell, {"examples": 0, "source_tokens": 0, "shards": []})
        cell_report["examples"] += len(shard_records)
        cell_report["source_tokens"] += sum(lengths_all)
        cell_report["shards"].append(path.name)
        total_tokens += sum(lengths_all)
    shard_entries: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("shard_*.pt")):
        index_payload = torch.load(path, map_location="cpu", weights_only=True)
        packed_name = str(index_payload["packed_hidden"])
        shard_entries.append({"name": path.name, "sha256": sha256_file(path), "packed_name": packed_name, "packed_sha256": sha256_file(output_root / packed_name)})
    manifest = {
        "schema_version": C1_CACHE_MANIFEST_SCHEMA,
        "identity": C1_CACHE_IDENTITY,
        "cache_contract": "fresh-qwen3.5-text-only-fp16-full-token-final-hidden",
        "dataset_root": str(Path(dataset_root).resolve()),
        "repo_root": str(Path(repo_root).resolve()) if repo_root is not None else None,
        "model_id": model_id,
        "revision": revision,
        "tokenizer_revision": revision,
        "dtype": "float16",
        "max_source_tokens": MAX_SOURCE_TOKENS,
        "silent_truncation": False,
        "input_ids_saved": False,
        "forbidden_payload_fields": sorted(FORBIDDEN_CACHE_FIELDS),
        "forbidden_fields_saved": [],
        "full_token_hidden_saved": True,
        "qwen_trainable_parameters": sum(
            parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
        ),
        "qwen_parameters": sum(parameter.numel() for parameter in backbone.parameters()),
        "qwen_component": type(backbone).__name__,
        "qwen_lm_head_present": hasattr(backbone, "lm_head"),
        "transformers_version": _require_transformers_runtime()
        if type(backbone).__name__ == "Qwen3_5TextModel"
        else None,
        "examples": len(ordered),
        "source_tokens": total_tokens,
        "hidden_width": int(hidden_rows[0].shape[-1]),
        "shard_size": shard_size,
        "shards": shard_entries,
        "cells": split_meta,
        "metadata": "metadata.json",
        "build_seconds": time.perf_counter() - started,
    }
    (output_root / "metadata.json").write_text(json.dumps({"schema_version": C1_CACHE_MANIFEST_SCHEMA, "rows": metadata_rows}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report = audit_cache(output_root)
    (output_root / "audit.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"C1 cache audit failed: {report['failures']}")
    return manifest


def _load_shard(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=True)


def audit_cache(cache_root: Path, records: Sequence[Mapping[str, Any] | PublicRecord] | None = None) -> dict[str, Any]:
    cache_root = Path(cache_root)
    failures: list[str] = []
    try:
        manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"schema_version": C1_CACHE_AUDIT_SCHEMA, "passed": False, "failures": [f"manifest:{exc}"]}
    if manifest.get("schema_version") != C1_CACHE_MANIFEST_SCHEMA:
        failures.append("manifest_schema")
    if manifest.get("identity") != C1_CACHE_IDENTITY:
        failures.append("cache_identity")
    if manifest.get("max_source_tokens") != MAX_SOURCE_TOKENS or manifest.get("silent_truncation") is not False:
        failures.append("max_length_or_truncation")
    if manifest.get("input_ids_saved") is not False or manifest.get("forbidden_fields_saved"):
        failures.append("manifest_forbidden_fields")
    if manifest.get("qwen_lm_head_present") is not False or manifest.get("qwen_trainable_parameters") != 0:
        failures.append("text_only_encoder_identity")
    if (
        manifest.get("qwen_component") == "Qwen3_5TextModel"
        and manifest.get("transformers_version") != contract.TRANSFORMERS_VERSION
    ):
        failures.append("transformers_version")
    observed: list[str] = []
    observed_hashes: list[str] = []
    token_count = 0
    for shard in manifest.get("shards", []):
        path = cache_root / str(shard.get("name", ""))
        if not path.is_file():
            failures.append(f"missing:{path.name}")
            continue
        if shard.get("sha256") != sha256_file(path):
            failures.append(f"hash:{path.name}")
        packed_path = cache_root / str(shard.get("packed_name", ""))
        if not packed_path.is_file() or shard.get("packed_sha256") != sha256_file(packed_path):
            failures.append(f"packed_hash:{path.name}")
        try:
            payload = _load_shard(path)
            keys = set(payload)
            if payload.get("schema_version") != C1_CACHE_SCHEMA:
                failures.append(f"schema:{path.name}")
            forbidden = sorted(keys & FORBIDDEN_CACHE_FIELDS)
            if forbidden or not keys <= ALLOWED_SHARD_FIELDS:
                failures.append(f"forbidden:{path.name}:{forbidden or sorted(keys - ALLOWED_SHARD_FIELDS)}")
            lengths = payload["token_lengths"]
            attention = payload["attention_mask"]
            shape = payload.get("packed_shape")
            offsets = payload.get("offsets")
            if lengths.ndim != 1 or attention.ndim != 1 or len(payload["example_ids"]) != lengths.shape[0] or not isinstance(shape, list) or len(shape) != 2 or not isinstance(offsets, list) or len(offsets) != lengths.shape[0] or attention.shape[0] != int(shape[0]) or not bool(attention.all()):
                failures.append(f"shape:{path.name}")
            if any(int(value) <= 0 or int(value) > MAX_SOURCE_TOKENS for value in lengths):
                failures.append(f"length_cap:{path.name}")
            if len(payload["source_hashes"]) != lengths.shape[0]:
                failures.append(f"identity_rows:{path.name}")
            try:
                packed = np.memmap(packed_path, mode="r", dtype=np.float16, shape=(int(shape[0]), int(shape[1])))
                if int(shape[0]) != int(lengths.sum()) or not np.isfinite(packed).all():
                    failures.append(f"packed_shape_or_finite:{path.name}")
                del packed
            except (OSError, ValueError, TypeError):
                failures.append(f"packed_payload:{path.name}")
            observed.extend(str(value) for value in payload["example_ids"])
            observed_hashes.extend(str(value) for value in payload["source_hashes"])
            token_count += int(lengths.sum())
        except Exception as exc:  # malformed torch payload is an audit failure
            failures.append(f"payload:{path.name}:{type(exc).__name__}")
    if len(observed) != int(manifest.get("examples", -1)):
        failures.append("example_count")
    if token_count != int(manifest.get("source_tokens", -1)):
        failures.append("token_count")
    if len(set(observed)) != len(observed):
        failures.append("duplicate_example_ids")
    if records is not None:
        expected = _ordered(records)
        expected_ids = [str(_cache_view(row)["example_id"]) for row in expected]
        expected_hashes = [source_byte_sha256(_cache_view(row)["source_text"]) for row in expected]
        if observed != expected_ids or observed_hashes != expected_hashes:
            failures.append("source_identity_order")
    return {
        "schema_version": C1_CACHE_AUDIT_SCHEMA,
        "identity": C1_CACHE_IDENTITY,
        "examples": len(observed),
        "source_tokens": token_count,
        "failures": failures,
        "passed": not failures,
    }


class CachedShardDataset:
    """Immutable cache view with deterministic example-ID lookup.

    The complete index is small and lives in RAM; hidden tensors remain in
    read-only memmaps and only requested rows are copied into a batch.  This is
    what lets the balanced training schedule remain random without ever
    loading the roughly 69 GB cache into host memory.
    """

    def __init__(self, cache_root: Path, *, metadata: Mapping[str, Mapping[str, Any]] | None = None, trace_targets: Mapping[str, Any] | None = None):
        self.root = Path(cache_root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != C1_CACHE_MANIFEST_SCHEMA:
            raise ValueError("not a C1 cache")
        self.metadata = dict(metadata or {})
        if not self.metadata and (self.root / "metadata.json").is_file():
            rows = json.loads((self.root / "metadata.json").read_text(encoding="utf-8")).get("rows", [])
            self.metadata = {str(row["example_id"]): row for row in rows}
        self.trace_targets = dict(trace_targets or {})
        self.shard_paths = [self.root / str(row["name"]) for row in self.manifest["shards"]]
        self.locations: dict[str, tuple[str, tuple[int, int], int, int]] = {}
        for path in self.shard_paths:
            payload = _load_shard(path)
            shape = tuple(int(value) for value in payload["packed_shape"])
            packed_name = str(payload["packed_hidden"])
            for row, example_id in enumerate(payload["example_ids"]):
                key = str(example_id)
                if key in self.locations:
                    raise ValueError(f"duplicate example ID in C1 cache: {key}")
                self.locations[key] = (
                    packed_name,
                    shape,
                    int(payload["offsets"][row]),
                    int(payload["token_lengths"][row]),
                )
        if len(self.locations) != len(self):
            raise ValueError("C1 cache index size differs from manifest")

    def __len__(self) -> int:
        return int(self.manifest.get("examples", 0))

    def iter_batches(self, batch_size: int, *, device: str | torch.device = "cpu") -> Iterator[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        hidden_rows: list[torch.Tensor] = []
        ids: list[str] = []
        for path in self.shard_paths:
            payload = _load_shard(path)
            packed_shape = payload["packed_shape"]
            packed = np.memmap(self.root / payload["packed_hidden"], mode="r", dtype=np.float16, shape=(int(packed_shape[0]), int(packed_shape[1])))
            for row, example_id in enumerate(payload["example_ids"]):
                length = int(payload["token_lengths"][row])
                offset = int(payload["offsets"][row])
                hidden_rows.append(torch.from_numpy(packed[offset : offset + length].copy()))
                ids.append(str(example_id))
                if len(ids) < batch_size:
                    continue
                yield self._batch(hidden_rows, ids, device)
                hidden_rows, ids = [], []
            del packed
        if ids:
            yield self._batch(hidden_rows, ids, device)

    def get_batch(
        self,
        example_ids: Sequence[str],
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        """Read requested rows in exactly the caller-provided order."""
        ids = [str(value) for value in example_ids]
        if not ids:
            raise ValueError("C1 batch cannot be empty")
        missing = [value for value in ids if value not in self.locations]
        if missing:
            raise KeyError(f"example IDs absent from C1 cache: {missing[:8]}")
        maps: dict[str, np.memmap] = {}
        rows: list[torch.Tensor] = []
        try:
            for example_id in ids:
                packed_name, shape, offset, length = self.locations[example_id]
                packed = maps.get(packed_name)
                if packed is None:
                    packed = np.memmap(
                        self.root / packed_name,
                        mode="r",
                        dtype=np.float16,
                        shape=shape,
                    )
                    maps[packed_name] = packed
                rows.append(torch.from_numpy(packed[offset : offset + length].copy()))
        finally:
            maps.clear()
        return self._batch(rows, ids, device)

    def _batch(self, rows: list[torch.Tensor], ids: list[str], device: str | torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        maximum = max(int(row.shape[0]) for row in rows)
        width = int(rows[0].shape[-1])
        hidden = torch.zeros((len(rows), maximum, width), dtype=torch.float16)
        mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
        metadata: list[dict[str, Any]] = []
        for index, (row, example_id) in enumerate(zip(rows, ids)):
            length = int(row.shape[0])
            hidden[index, :length] = row
            mask[index, :length] = True
            item = dict(self.metadata.get(example_id, {"example_id": example_id}))
            if example_id in self.trace_targets:
                item["trace_target"] = self.trace_targets[example_id]
            metadata.append(item)
        return {"source_hidden": hidden.to(device), "source_mask": mask.to(device)}, metadata


def iter_batches(cache_root: Path, batch_size: int, *, device: str | torch.device = "cpu", metadata: Mapping[str, Mapping[str, Any]] | None = None, trace_targets: Mapping[str, Any] | None = None) -> Iterator[tuple[dict[str, torch.Tensor], list[dict[str, Any]]]]:
    return CachedShardDataset(cache_root, metadata=metadata, trace_targets=trace_targets).iter_batches(batch_size, device=device)


__all__ = [
    "ALLOWED_SHARD_FIELDS", "C1_CACHE_AUDIT_SCHEMA", "C1_CACHE_IDENTITY", "C1_CACHE_MANIFEST_SCHEMA", "C1_CACHE_SCHEMA",
    "CachedShardDataset", "FORBIDDEN_CACHE_FIELDS", "QWEN_MODEL_ID", "QWEN_REVISION", "audit_cache", "build_cache",
    "iter_batches", "load_qwen35_text_only", "load_qwen35_tokenizer", "sha256_file",
]
