from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .a1_7_data import REGISTER_NAMES
from .a1_8_data import TRAIN_LENGTHS, load_a18_records
from .model import load_qwen35_text_only


QWEN_MODEL_ID = "Qwen/Qwen3.5-2B"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CACHE_SCHEMA = "yggdrasil.v2-a1.9.role-pooled-qwen-cache.v1"
CACHE_MANIFEST_SCHEMA = "yggdrasil.v2-a1.9.role-pooled-qwen-cache-manifest.v1"
CACHE_AUDIT_SCHEMA = "yggdrasil.v2-a1.9.role-pooled-qwen-cache-audit.v1"
FORMAL_SPLITS = (
    "train",
    "validation",
    "short_regression",
    "supported_in_range",
    "relation_in_range",
    "supported_ood",
    "relation_ood",
    "causal_core",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_length_balanced_overfit32(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["program_length"])].append(record)
    if set(grouped) != set(TRAIN_LENGTHS):
        raise ValueError("overfit32 requires source records covering every length T1-T16")
    selected = [record for length in TRAIN_LENGTHS for record in grouped[length][:2]]
    if len(selected) != 32:
        raise ValueError("overfit32 requires two records for every length T1-T16")
    return selected


def _token_mask(offsets: Sequence[Sequence[int]], start: int, end: int, *, label: str) -> list[int]:
    indexes = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end and token_end > token_start
    ]
    if not indexes:
        raise ValueError(f"empty Qwen token mask for {label} span [{start}, {end})")
    return indexes


def _pool_span(
    hidden: torch.Tensor,
    offsets: Sequence[Sequence[int]],
    span: dict[str, int],
    *,
    prefix: str,
    label: str,
) -> torch.Tensor:
    start = int(span[f"{prefix}char_start"])
    end = int(span[f"{prefix}char_end"])
    indexes = _token_mask(offsets, start, end, label=label)
    return hidden[indexes].mean(dim=0)


def _pool_record(
    hidden: torch.Tensor,
    offsets: Sequence[Sequence[int]],
    record: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_hidden = torch.stack(
        [
            _pool_span(
                hidden,
                offsets,
                record["start_value_spans"][name],
                prefix="",
                label=f"start_value_{name}",
            )
            for name in REGISTER_NAMES
        ]
    )
    query_hidden = _pool_span(
        hidden,
        offsets,
        record["query_register_span"],
        prefix="",
        label="query_register",
    )
    operation_rows: list[torch.Tensor] = []
    for step, spans in enumerate(record["operation_spans"]):
        operation_rows.append(
            torch.stack(
                [
                    _pool_span(hidden, offsets, spans, prefix="family_", label=f"operation_{step}_family"),
                    _pool_span(hidden, offsets, spans, prefix="source_", label=f"operation_{step}_source"),
                    _pool_span(hidden, offsets, spans, prefix="target_", label=f"operation_{step}_target"),
                ]
            )
        )
    return start_hidden, query_hidden, torch.stack(operation_rows)


def _encode_batch(tokenizer: Any, records: Sequence[dict[str, Any]], max_length: int) -> tuple[torch.Tensor, torch.Tensor, list[list[list[int]]], list[int]]:
    encoded = [
        tokenizer(record["question"], add_special_tokens=True, truncation=False, return_offsets_mapping=True)
        for record in records
    ]
    lengths = [len(row["input_ids"]) for row in encoded]
    if any(length > max_length for length in lengths):
        raise ValueError(f"A1.9 refuses silent truncation: token length exceeds max_length={max_length}")
    sequence_length = max(lengths)
    input_ids = torch.full((len(records), sequence_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), sequence_length), dtype=torch.bool)
    offsets: list[list[list[int]]] = []
    for row_index, row in enumerate(encoded):
        length = lengths[row_index]
        input_ids[row_index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[row_index, :length] = True
        offsets.append([[int(start), int(end)] for start, end in row["offset_mapping"]])
    return input_ids, attention_mask, offsets, lengths


def _cache_split(
    backbone: torch.nn.Module,
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    device: str,
    max_length: int,
    inference_batch_size: int,
    shard_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    maximum_token_length = 0
    started = time.perf_counter()
    for shard_index, shard_start in enumerate(range(0, len(records), shard_size)):
        shard_records = records[shard_start : shard_start + shard_size]
        starts: list[torch.Tensor] = []
        queries: list[torch.Tensor] = []
        operations: list[torch.Tensor] = []
        operation_offsets = [0]
        token_lengths: list[int] = []
        for batch_start in range(0, len(shard_records), inference_batch_size):
            batch_records = shard_records[batch_start : batch_start + inference_batch_size]
            input_ids, attention_mask, offsets, lengths = _encode_batch(tokenizer, batch_records, max_length)
            maximum_token_length = max(maximum_token_length, max(lengths))
            with torch.inference_mode():
                hidden = backbone.model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    use_cache=False,
                ).last_hidden_state
            for row, record in enumerate(batch_records):
                start_hidden, query_hidden, operation_hidden = _pool_record(
                    hidden[row, : lengths[row]],
                    offsets[row],
                    record,
                )
                starts.append(start_hidden.detach().to(dtype=torch.float16, device="cpu"))
                queries.append(query_hidden.detach().to(dtype=torch.float16, device="cpu"))
                operations.append(operation_hidden.detach().to(dtype=torch.float16, device="cpu"))
                operation_offsets.append(operation_offsets[-1] + int(operation_hidden.shape[0]))
            token_lengths.extend(lengths)
            del hidden
        payload = {
            "schema_version": CACHE_SCHEMA,
            "start_value_hidden": torch.stack(starts),
            "query_hidden": torch.stack(queries),
            "operation_hidden": torch.cat(operations, dim=0),
            "operation_offsets": torch.tensor(operation_offsets, dtype=torch.long),
            "operation_lengths": torch.tensor([int(record["program_length"]) for record in shard_records], dtype=torch.long),
            "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
            "fingerprints": [record["fingerprint"] for record in shard_records],
            "example_ids": [record["example_id"] for record in shard_records],
        }
        path = output_dir / f"shard_{shard_index:05d}.pt"
        torch.save(payload, path)
        shard_paths.append(path.name)
    return {
        "examples": len(records),
        "role_tokens": sum(4 + 3 * int(record["program_length"]) for record in records),
        "program_lengths": sorted({int(record["program_length"]) for record in records}),
        "maximum_token_length": maximum_token_length,
        "shards": shard_paths,
        "seconds": time.perf_counter() - started,
    }


def cache_a19_role_hidden(
    data_dir: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    max_length: int = 1024,
    inference_batch_size: int = 8,
    shard_size: int = 64,
    overfit32: bool = False,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.9 Qwen cache requested CUDA but CUDA is unavailable")
    if output_dir.exists() and any(output_dir.rglob("*.pt")):
        raise RuntimeError(f"refusing to mix or resume an existing A1.9 cache: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    backbone = load_qwen35_text_only(model_id, revision, dtype=torch.float16, device=device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    qwen_parameters = sum(parameter.numel() for parameter in backbone.parameters())
    selected = select_length_balanced_overfit32(load_a18_records(data_dir, "train")) if overfit32 else None
    split_sources = {"train": "train", "validation": "train"} if overfit32 else {split: split for split in FORMAL_SPLITS}
    splits: dict[str, Any] = {}
    total_started = time.perf_counter()
    for split, source_split in split_sources.items():
        records = selected if overfit32 else load_a18_records(data_dir, source_split)
        assert records is not None
        split_result = _cache_split(
            backbone,
            tokenizer,
            records,
            output_dir / split,
            device=device,
            max_length=max_length,
            inference_batch_size=inference_batch_size,
            shard_size=shard_size,
        )
        split_result["source_data_split"] = source_split
        splits[split] = split_result
    peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "stage": "A1.9 frozen Qwen role boundary",
        "data_dir": str(data_dir.resolve()),
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "selection": "length-balanced-overfit32-fit-only" if overfit32 else "formal-all-splits",
        "model_id": model_id,
        "revision": revision,
        "dtype": "float16",
        "hidden_layer": "last_hidden_state",
        "pooling": "masked-mean-per-oracle-role-span",
        "oracle_role_span_segmentation": True,
        "full_source_saved": False,
        "full_source_cross_attention": False,
        "silent_truncation": False,
        "max_length": max_length,
        "inference_batch_size": inference_batch_size,
        "shard_size": shard_size,
        "qwen_parameters": qwen_parameters,
        "qwen_trainable_parameters": 0,
        "device": device,
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if device.startswith("cuda") else "cpu",
        "peak_allocated_bytes": peak,
        "seconds": time.perf_counter() - total_started,
        "splits": splits,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load_payload(path: Path, *, mmap: bool = True) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=False)


def cached_fingerprints(cache_dir: Path, split: str) -> set[str]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    fingerprints: set[str] = set()
    for name in manifest["splits"][split]["shards"]:
        payload = _load_payload(cache_dir / split / name)
        fingerprints.update(payload["fingerprints"])
    return fingerprints


def audit_a19_cache(cache_dir: Path, data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ValueError("not an A1.9 role-pooled Qwen cache manifest")
    overfit32 = manifest["selection"] == "length-balanced-overfit32-fit-only"
    split_reports: dict[str, Any] = {}
    all_finite = shapes_valid = fingerprints_match = schemas_valid = no_forbidden_tensor = True
    hidden_widths: set[int] = set()
    split_fingerprints: dict[str, set[str]] = {}
    forbidden = {"last_hidden", "source_hidden", "full_source_hidden", "input_ids"}
    for split, split_manifest in manifest["splits"].items():
        source_split = split_manifest["source_data_split"]
        source_records = load_a18_records(data_dir, source_split)
        expected_records = select_length_balanced_overfit32(source_records) if overfit32 else source_records
        expected = [record["fingerprint"] for record in expected_records]
        observed: list[str] = []
        examples = operations = 0
        for name in split_manifest["shards"]:
            payload = _load_payload(cache_dir / split / name)
            schemas_valid = schemas_valid and payload.get("schema_version") == CACHE_SCHEMA
            no_forbidden_tensor = no_forbidden_tensor and not (forbidden & set(payload))
            starts = payload["start_value_hidden"]
            queries = payload["query_hidden"]
            role_hidden = payload["operation_hidden"]
            offsets = payload["operation_offsets"]
            lengths = payload["operation_lengths"]
            batch = len(payload["fingerprints"])
            shapes_valid = shapes_valid and starts.ndim == 3 and starts.shape[1] == 3
            shapes_valid = shapes_valid and queries.shape == (batch, starts.shape[-1])
            shapes_valid = shapes_valid and role_hidden.ndim == 3 and role_hidden.shape[1] == 3 and role_hidden.shape[-1] == starts.shape[-1]
            shapes_valid = shapes_valid and offsets.shape == (batch + 1,) and lengths.shape == (batch,)
            shapes_valid = shapes_valid and int(offsets[-1]) == int(role_hidden.shape[0]) and torch.equal(offsets[1:] - offsets[:-1], lengths)
            all_finite = all_finite and bool(torch.isfinite(starts).all() and torch.isfinite(queries).all() and torch.isfinite(role_hidden).all())
            hidden_widths.add(int(starts.shape[-1]))
            observed.extend(payload["fingerprints"])
            examples += batch
            operations += int(role_hidden.shape[0])
        fingerprints_match = fingerprints_match and observed == expected
        split_fingerprints[split] = set(observed)
        split_reports[split] = {
            "examples": examples,
            "operations": operations,
            "expected_examples": len(expected),
            "unique_fingerprints": len(set(observed)),
            "fingerprints_match_in_order": observed == expected,
        }
    overlap: dict[str, int] = {}
    split_names = sorted(split_fingerprints)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap[f"{left}__{right}"] = len(split_fingerprints[left] & split_fingerprints[right])
    formal_overlap_zero = overfit32 or all(value == 0 for value in overlap.values())
    audit_path = data_dir.parent / "data-audit.json"
    source_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"passed": False}
    gates = {
        "manifest_schema": manifest["schema_version"] == CACHE_MANIFEST_SCHEMA,
        "data_manifest_hash_matches": manifest["data_manifest_sha256"] == file_sha256(data_dir / "manifest.json"),
        "source_a1_8_data_audit_passed": bool(source_audit.get("passed")),
        "cache_shard_schemas": schemas_valid,
        "cache_shapes_and_offsets": shapes_valid,
        "cache_hidden_finite": all_finite,
        "single_hidden_width": len(hidden_widths) == 1,
        "fingerprints_match_source_in_order": fingerprints_match,
        "formal_cross_split_overlap_zero": formal_overlap_zero,
        "full_source_tensor_absent": no_forbidden_tensor and not manifest["full_source_saved"],
        "oracle_role_spans_explicit": bool(manifest["oracle_role_span_segmentation"]),
        "qwen_frozen_and_no_truncation": manifest["qwen_trainable_parameters"] == 0 and not manifest["silent_truncation"],
    }
    return {
        "schema_version": CACHE_AUDIT_SCHEMA,
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "selection": manifest["selection"],
        "hidden_widths": sorted(hidden_widths),
        "splits": split_reports,
        "cross_split_fingerprint_overlap": overlap,
        "gates": gates,
        "passed": all(gates.values()),
    }


class A19CachedSplit:
    """Memory-mapped role-pooled hidden cache joined to immutable A1.8 records."""

    def __init__(self, cache_dir: Path, data_dir: Path, split: str) -> None:
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.split = split
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        split_manifest = manifest["splits"][split]
        source_records = load_a18_records(data_dir, split_manifest["source_data_split"])
        record_map = {record["fingerprint"]: record for record in source_records}
        self.payloads: list[dict[str, Any]] = []
        self.rows: list[tuple[int, int]] = []
        self.records: list[dict[str, Any]] = []
        for payload_index, name in enumerate(split_manifest["shards"]):
            payload = _load_payload(cache_dir / split / name)
            self.payloads.append(payload)
            for row, fingerprint in enumerate(payload["fingerprints"]):
                self.rows.append((payload_index, row))
                self.records.append(record_map[fingerprint])
        self.hidden_width = int(self.payloads[0]["start_value_hidden"].shape[-1])
        self.indices_by_length: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.indices_by_length[int(record["program_length"])].append(index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload_index, row = self.rows[index]
        payload = self.payloads[payload_index]
        operation_start = int(payload["operation_offsets"][row])
        operation_end = int(payload["operation_offsets"][row + 1])
        return {
            "start_hidden": payload["start_value_hidden"][row],
            "query_hidden": payload["query_hidden"][row],
            "operation_hidden": payload["operation_hidden"][operation_start:operation_end],
            "record": self.records[index],
        }

    def items(self, indices: Iterable[int] | None = None) -> list[dict[str, Any]]:
        selected = range(len(self)) if indices is None else indices
        return [self[index] for index in selected]


def collate_a19_items(items: Sequence[dict[str, Any]], device: str | torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if not items:
        raise ValueError("cannot collate an empty A1.9 item batch")
    starts = torch.stack([item["start_hidden"] for item in items])
    queries = torch.stack([item["query_hidden"] for item in items])
    maximum_steps = max(int(item["operation_hidden"].shape[0]) for item in items)
    hidden_width = int(starts.shape[-1])
    operations = torch.zeros((len(items), maximum_steps, 3, hidden_width), dtype=starts.dtype)
    mask = torch.zeros((len(items), maximum_steps), dtype=torch.bool)
    for row, item in enumerate(items):
        length = int(item["operation_hidden"].shape[0])
        operations[row, :length] = item["operation_hidden"]
        mask[row, :length] = True
    inputs = {
        "start_value_hidden": starts.to(device=device, dtype=torch.float32),
        "query_hidden": queries.to(device=device, dtype=torch.float32),
        "family_hidden": operations[:, :, 0].to(device=device, dtype=torch.float32),
        "source_hidden": operations[:, :, 1].to(device=device, dtype=torch.float32),
        "target_hidden": operations[:, :, 2].to(device=device, dtype=torch.float32),
        "operation_mask": mask.to(device),
    }
    return inputs, [item["record"] for item in items]


def sample_length_balanced_indices(dataset: A19CachedSplit, batch_size: int, rng: random.Random) -> list[int]:
    lengths = sorted(dataset.indices_by_length)
    if not lengths or any(not dataset.indices_by_length[length] for length in lengths):
        raise ValueError("length-balanced A1.9 sampling requires non-empty groups")
    sampled_lengths = [lengths[index % len(lengths)] for index in range(batch_size)]
    rng.shuffle(sampled_lengths)
    return [rng.choice(dataset.indices_by_length[length]) for length in sampled_lengths]
