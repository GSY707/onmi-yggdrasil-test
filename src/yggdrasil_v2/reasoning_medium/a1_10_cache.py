from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .a1_8_data import TRAIN_LENGTHS, load_a18_records
from .model import load_qwen35_text_only


QWEN_MODEL_ID = "Qwen/Qwen3.5-2B"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CACHE_SCHEMA = "yggdrasil.v2-a1.10.full-token-qwen-cache.v1"
CACHE_MANIFEST_SCHEMA = "yggdrasil.v2-a1.10.full-token-qwen-cache-manifest.v1"
CACHE_AUDIT_SCHEMA = "yggdrasil.v2-a1.10.full-token-qwen-cache-audit.v1"
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
FORBIDDEN_CACHE_FIELDS = {
    "offset_mapping",
    "operation_token_mask",
    "operation_mask",
    "role_hidden",
    "start_value_hidden",
    "query_hidden",
    "family_hidden",
    "source_role_hidden",
    "target_hidden",
    "input_ids",
}


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
        raise ValueError("A1.10 overfit32 requires source records covering T1-T16")
    selected = [record for length in TRAIN_LENGTHS for record in grouped[length][:2]]
    if len(selected) != 32:
        raise ValueError("A1.10 overfit32 requires two records per training length")
    return selected


def _ordered(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: (int(record["program_length"]), record["fingerprint"]))


def _encode_batch(tokenizer: Any, records: Sequence[dict[str, Any]], max_length: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    encoded = [tokenizer(record["question"], add_special_tokens=True, truncation=False) for record in records]
    lengths = [len(row["input_ids"]) for row in encoded]
    if any(length > max_length for length in lengths):
        raise ValueError(f"A1.10 refuses silent truncation above max_length={max_length}")
    sequence_length = max(lengths)
    input_ids = torch.full((len(records), sequence_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), sequence_length), dtype=torch.bool)
    for row_index, row in enumerate(encoded):
        length = lengths[row_index]
        input_ids[row_index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[row_index, :length] = True
    return input_ids, attention_mask, lengths


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
    records = _ordered(records)
    shard_paths: list[str] = []
    maximum_token_length = 0
    total_tokens = 0
    started = time.perf_counter()
    for shard_index, shard_start in enumerate(range(0, len(records), shard_size)):
        shard_records = records[shard_start : shard_start + shard_size]
        rows: list[torch.Tensor] = []
        token_lengths: list[int] = []
        for batch_start in range(0, len(shard_records), inference_batch_size):
            batch_records = shard_records[batch_start : batch_start + inference_batch_size]
            input_ids, attention_mask, lengths = _encode_batch(tokenizer, batch_records, max_length)
            with torch.inference_mode():
                hidden = backbone.model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    use_cache=False,
                ).last_hidden_state
            for row, length in enumerate(lengths):
                rows.append(hidden[row, :length].detach().to(dtype=torch.float16, device="cpu"))
            token_lengths.extend(lengths)
            maximum_token_length = max(maximum_token_length, max(lengths))
            total_tokens += sum(lengths)
            del hidden
        shard_max = max(token_lengths)
        hidden_width = int(rows[0].shape[-1])
        last_hidden = torch.zeros((len(rows), shard_max, hidden_width), dtype=torch.float16)
        attention_mask = torch.zeros((len(rows), shard_max), dtype=torch.bool)
        for row_index, row_hidden in enumerate(rows):
            length = int(row_hidden.shape[0])
            last_hidden[row_index, :length] = row_hidden
            attention_mask[row_index, :length] = True
        payload = {
            "schema_version": CACHE_SCHEMA,
            "last_hidden": last_hidden,
            "attention_mask": attention_mask,
            "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
            "fingerprints": [record["fingerprint"] for record in shard_records],
            "example_ids": [record["example_id"] for record in shard_records],
        }
        path = output_dir / f"shard_{shard_index:05d}.pt"
        torch.save(payload, path)
        shard_paths.append(path.name)
    return {
        "examples": len(records),
        "source_tokens": total_tokens,
        "program_lengths": sorted({int(record["program_length"]) for record in records}),
        "maximum_token_length": maximum_token_length,
        "shards": shard_paths,
        "seconds": time.perf_counter() - started,
        "source_data_order": "program_length_then_fingerprint",
    }


def cache_a110_full_hidden(
    data_dir: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    max_length: int = 1024,
    inference_batch_size: int = 16,
    shard_size: int = 64,
    overfit32: bool = False,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.10 Qwen cache requested CUDA but CUDA is unavailable")
    if output_dir.exists() and any(output_dir.rglob("*.pt")):
        raise RuntimeError(f"refusing to mix or resume an existing A1.10 cache: {output_dir}")
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
        report = _cache_split(
            backbone,
            tokenizer,
            records,
            output_dir / split,
            device=device,
            max_length=max_length,
            inference_batch_size=inference_batch_size,
            shard_size=shard_size,
        )
        report["source_data_split"] = source_split
        splits[split] = report
    peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "stage": "A1.10 full-token frozen Qwen boundary",
        "data_dir": str(data_dir.resolve()),
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "selection": "length-balanced-overfit32-fit-only" if overfit32 else "formal-all-splits",
        "model_id": model_id,
        "revision": revision,
        "tokenizer_revision": revision,
        "dtype": "float16",
        "hidden_layer": "last",
        "max_length": max_length,
        "silent_truncation": False,
        "qwen_parameters": qwen_parameters,
        "qwen_trainable_parameters": 0,
        "full_source_saved": True,
        "oracle_role_span_segmentation": False,
        "operation_token_mask_saved": False,
        "typed_role_tensor_saved": False,
        "input_ids_saved": False,
        "cache_contract": "full_last_hidden_plus_attention_mask_only",
        "inference_batch_size": inference_batch_size,
        "shard_size": shard_size,
        "splits": splits,
        "full_token_encoding_seconds_excluding_model_and_tokenizer_load": time.perf_counter() - total_started,
        "peak_allocated_bytes": peak,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
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


def audit_a110_cache(cache_dir: Path, data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ValueError("not an A1.10 full-token Qwen cache manifest")
    overfit32 = manifest["selection"] == "length-balanced-overfit32-fit-only"
    split_reports: dict[str, Any] = {}
    schemas_valid = shapes_valid = finite = fingerprints_match = forbidden_absent = True
    hidden_widths: set[int] = set()
    split_fingerprints: dict[str, set[str]] = {}
    for split, split_manifest in manifest["splits"].items():
        source_records = load_a18_records(data_dir, split_manifest["source_data_split"])
        expected_records = select_length_balanced_overfit32(source_records) if overfit32 else source_records
        expected = [record["fingerprint"] for record in _ordered(expected_records)]
        observed: list[str] = []
        examples = tokens = 0
        for name in split_manifest["shards"]:
            payload = _load_payload(cache_dir / split / name)
            schemas_valid = schemas_valid and payload.get("schema_version") == CACHE_SCHEMA
            forbidden_absent = forbidden_absent and not (FORBIDDEN_CACHE_FIELDS & set(payload))
            hidden = payload["last_hidden"]
            mask = payload["attention_mask"]
            lengths = payload["token_lengths"]
            batch = len(payload["fingerprints"])
            shapes_valid = shapes_valid and hidden.ndim == 3 and mask.shape == hidden.shape[:2] and lengths.shape == (batch,)
            shapes_valid = shapes_valid and hidden.shape[0] == batch and torch.equal(mask.sum(dim=1).to(torch.long), lengths)
            finite = finite and bool(torch.isfinite(hidden).all())
            hidden_widths.add(int(hidden.shape[-1]))
            observed.extend(payload["fingerprints"])
            examples += batch
            tokens += int(lengths.sum())
        fingerprints_match = fingerprints_match and observed == expected
        split_fingerprints[split] = set(observed)
        split_reports[split] = {
            "examples": examples,
            "source_tokens": tokens,
            "expected_examples": len(expected),
            "unique_fingerprints": len(set(observed)),
            "fingerprints_match_source_order": observed == expected,
        }
    overlap: dict[str, int] = {}
    names = sorted(split_fingerprints)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap[f"{left}__{right}"] = len(split_fingerprints[left] & split_fingerprints[right])
    formal_overlap_zero = overfit32 or all(value == 0 for value in overlap.values())
    source_audit_path = data_dir.parent / "data-audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8")) if source_audit_path.exists() else {"passed": False}
    gates = {
        "manifest_schema": manifest["schema_version"] == CACHE_MANIFEST_SCHEMA,
        "data_manifest_hash_matches": manifest["data_manifest_sha256"] == file_sha256(data_dir / "manifest.json"),
        "source_a1_8_data_audit_passed": bool(source_audit.get("passed")),
        "cache_shard_schemas": schemas_valid,
        "cache_shapes": shapes_valid,
        "cache_hidden_finite": finite,
        "single_hidden_width": len(hidden_widths) == 1,
        "fingerprints_match_source_order": fingerprints_match,
        "formal_cross_split_overlap_zero": formal_overlap_zero,
        "full_source_tensor_present": bool(manifest["full_source_saved"]),
        "oracle_span_and_role_tensors_absent": (
            forbidden_absent
            and not manifest["oracle_role_span_segmentation"]
            and not manifest["operation_token_mask_saved"]
            and not manifest["typed_role_tensor_saved"]
        ),
        "input_ids_absent": not manifest["input_ids_saved"],
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


class A110CachedSplit:
    """Memory-mapped full-token Qwen hidden joined to immutable A1.8 records."""

    def __init__(self, cache_dir: Path, data_dir: Path, split: str) -> None:
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.split = split
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != CACHE_MANIFEST_SCHEMA:
            raise ValueError("A110CachedSplit requires an A1.10 cache")
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
        self.hidden_width = int(self.payloads[0]["last_hidden"].shape[-1])
        self.indices_by_length: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.indices_by_length[int(record["program_length"])].append(index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload_index, row = self.rows[index]
        payload = self.payloads[payload_index]
        length = int(payload["token_lengths"][row])
        return {
            "last_hidden": payload["last_hidden"][row, :length],
            "attention_mask": payload["attention_mask"][row, :length],
            "record": self.records[index],
        }

    def items(self, indices: Iterable[int] | None = None) -> list[dict[str, Any]]:
        selected = range(len(self)) if indices is None else indices
        return [self[index] for index in selected]


def collate_a110_items(items: Sequence[dict[str, Any]], device: str | torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if not items:
        raise ValueError("cannot collate an empty A1.10 batch")
    maximum_tokens = max(int(item["last_hidden"].shape[0]) for item in items)
    hidden_width = int(items[0]["last_hidden"].shape[-1])
    hidden = torch.zeros((len(items), maximum_tokens, hidden_width), dtype=torch.float16)
    mask = torch.zeros((len(items), maximum_tokens), dtype=torch.bool)
    for row, item in enumerate(items):
        length = int(item["last_hidden"].shape[0])
        hidden[row, :length] = item["last_hidden"]
        mask[row, :length] = item["attention_mask"]
    return {
        "source_hidden": hidden.to(device),
        "source_attention_mask": mask.to(device),
    }, [item["record"] for item in items]


def sample_length_balanced_indices(dataset: A110CachedSplit, batch_size: int, rng: random.Random) -> list[int]:
    lengths = sorted(dataset.indices_by_length)
    if not lengths or any(not dataset.indices_by_length[length] for length in lengths):
        raise ValueError("length-balanced A1.10 sampling requires non-empty groups")
    sampled_lengths = [lengths[index % len(lengths)] for index in range(batch_size)]
    rng.shuffle(sampled_lengths)
    return [rng.choice(dataset.indices_by_length[length]) for length in sampled_lengths]
