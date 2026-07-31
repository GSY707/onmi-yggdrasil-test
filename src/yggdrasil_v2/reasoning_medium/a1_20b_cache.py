from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .a1_10_cache import QWEN_MODEL_ID, QWEN_REVISION
from .a1_19h_h2_data import FORMAL_SPLITS, load_a119h2_records
from .model import load_qwen35_text_only


CACHE_SCHEMA = "yggdrasil.v2-a1.20b.full-token-qwen-cache.v1"
MANIFEST_SCHEMA = "yggdrasil.v2-a1.20b.full-token-qwen-cache-manifest.v1"
AUDIT_SCHEMA = "yggdrasil.v2-a1.20b.full-token-qwen-cache-audit.v1"
CACHE_SPLITS = ("train", "validation", *FORMAL_SPLITS)
FORBIDDEN_FIELDS = {
    "input_ids",
    "offset_mapping",
    "entity_mask",
    "operation_mask",
    "role_hidden",
    "role_tensor",
    "span_mask",
    "source_pointer",
    "target_pointer",
    "query_pointer",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_a120b_overfit32(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cell[(int(record["entity_count"]), int(record["program_length"]))].append(
            record
        )
    counts = sorted({cell[0] for cell in by_cell})
    lengths = sorted({cell[1] for cell in by_cell})
    if counts != [2, 3, 4] or lengths != list(range(1, 17)):
        raise ValueError("A1.20B overfit32 requires N2-N4 and T1-T16")
    cursors: dict[tuple[int, int], int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for index in range(32):
        cell = (counts[index % len(counts)], lengths[index % len(lengths)])
        choices = sorted(by_cell[cell], key=lambda row: row["fingerprint"])
        selected.append(choices[cursors[cell] % len(choices)])
        cursors[cell] += 1
    return selected


def _ordered(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            int(record["entity_count"]),
            int(record["program_length"]),
            record["fingerprint"],
        ),
    )


def _encode_batch(
    tokenizer: Any, records: Sequence[dict[str, Any]], max_length: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    encoded = [
        tokenizer(record["question"], add_special_tokens=True, truncation=False)
        for record in records
    ]
    lengths = [len(row["input_ids"]) for row in encoded]
    if any(length > max_length for length in lengths):
        raise ValueError(f"A1.20B refuses truncation above max_length={max_length}")
    sequence_length = max(lengths)
    input_ids = torch.full(
        (len(records), sequence_length), tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros(
        (len(records), sequence_length), dtype=torch.bool
    )
    for row_index, row in enumerate(encoded):
        length = lengths[row_index]
        input_ids[row_index, :length] = torch.tensor(
            row["input_ids"], dtype=torch.long
        )
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
    ordered = _ordered(records)
    shards: list[str] = []
    total_tokens = maximum_token_length = 0
    started = time.perf_counter()
    for shard_index, shard_start in enumerate(range(0, len(ordered), shard_size)):
        shard_records = ordered[shard_start : shard_start + shard_size]
        rows: list[torch.Tensor] = []
        lengths: list[int] = []
        for batch_start in range(0, len(shard_records), inference_batch_size):
            batch_records = shard_records[
                batch_start : batch_start + inference_batch_size
            ]
            input_ids, attention_mask, batch_lengths = _encode_batch(
                tokenizer, batch_records, max_length
            )
            with torch.inference_mode():
                hidden = backbone.model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    use_cache=False,
                ).last_hidden_state
            for row, length in enumerate(batch_lengths):
                rows.append(
                    hidden[row, :length]
                    .detach()
                    .to(dtype=torch.float16, device="cpu")
                )
            lengths.extend(batch_lengths)
            total_tokens += sum(batch_lengths)
            maximum_token_length = max(maximum_token_length, max(batch_lengths))
            del hidden
        shard_max = max(lengths)
        hidden_width = int(rows[0].shape[-1])
        last_hidden = torch.zeros(
            (len(rows), shard_max, hidden_width), dtype=torch.float16
        )
        mask = torch.zeros((len(rows), shard_max), dtype=torch.bool)
        for row_index, row_hidden in enumerate(rows):
            length = int(row_hidden.shape[0])
            last_hidden[row_index, :length] = row_hidden
            mask[row_index, :length] = True
        payload = {
            "schema_version": CACHE_SCHEMA,
            "last_hidden": last_hidden,
            "attention_mask": mask,
            "token_lengths": torch.tensor(lengths, dtype=torch.long),
            "fingerprints": [record["fingerprint"] for record in shard_records],
            "example_ids": [record["example_id"] for record in shard_records],
        }
        path = output_dir / f"shard_{shard_index:05d}.pt"
        torch.save(payload, path)
        shards.append(path.name)
    return {
        "examples": len(ordered),
        "source_tokens": total_tokens,
        "entity_counts": sorted({int(row["entity_count"]) for row in ordered}),
        "program_lengths": sorted(
            {int(row["program_length"]) for row in ordered}
        ),
        "maximum_token_length": maximum_token_length,
        "shards": shards,
        "seconds": time.perf_counter() - started,
        "source_data_order": "entity_count_then_program_length_then_fingerprint",
    }


def _recover_cached_split(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    max_length: int,
    shard_size: int,
) -> dict[str, Any]:
    """Validate and reuse a split written before its manifest was committed."""
    ordered = _ordered(records)
    expected_paths = [
        output_dir / f"shard_{index:05d}.pt"
        for index, _ in enumerate(range(0, len(ordered), shard_size))
    ]
    actual_paths = sorted(output_dir.glob("*.pt"))
    if actual_paths != expected_paths:
        raise RuntimeError(
            "incomplete A1.20B cache recovery set: "
            f"expected {[path.name for path in expected_paths]}, "
            f"found {[path.name for path in actual_paths]}"
        )
    total_tokens = maximum_token_length = 0
    for shard_index, path in enumerate(actual_paths):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != CACHE_SCHEMA:
            raise RuntimeError(f"cache recovery schema mismatch: {path}")
        forbidden = sorted(FORBIDDEN_FIELDS.intersection(payload))
        if forbidden:
            raise RuntimeError(
                f"cache recovery found forbidden fields {forbidden}: {path}"
            )
        start = shard_index * shard_size
        expected_records = ordered[start : start + shard_size]
        expected_fingerprints = [
            record["fingerprint"] for record in expected_records
        ]
        expected_example_ids = [
            record["example_id"] for record in expected_records
        ]
        if payload.get("fingerprints") != expected_fingerprints:
            raise RuntimeError(f"cache recovery fingerprint mismatch: {path}")
        if payload.get("example_ids") != expected_example_ids:
            raise RuntimeError(f"cache recovery example-id mismatch: {path}")
        hidden = payload.get("last_hidden")
        mask = payload.get("attention_mask")
        lengths = payload.get("token_lengths")
        if (
            not isinstance(hidden, torch.Tensor)
            or not isinstance(mask, torch.Tensor)
            or not isinstance(lengths, torch.Tensor)
            or hidden.ndim != 3
            or mask.ndim != 2
            or lengths.ndim != 1
            or hidden.shape[:2] != mask.shape
            or hidden.shape[0] != lengths.shape[0]
            or hidden.shape[0] != len(expected_records)
        ):
            raise RuntimeError(f"cache recovery tensor-shape mismatch: {path}")
        length_values = [int(value) for value in lengths.tolist()]
        if (
            any(value <= 0 or value > max_length for value in length_values)
            or [int(value) for value in mask.sum(dim=1).tolist()]
            != length_values
        ):
            raise RuntimeError(f"cache recovery token-length mismatch: {path}")
        total_tokens += sum(length_values)
        maximum_token_length = max(maximum_token_length, max(length_values))
    return {
        "examples": len(ordered),
        "source_tokens": total_tokens,
        "entity_counts": sorted({int(row["entity_count"]) for row in ordered}),
        "program_lengths": sorted(
            {int(row["program_length"]) for row in ordered}
        ),
        "maximum_token_length": maximum_token_length,
        "shards": [path.name for path in actual_paths],
        "seconds": 0.0,
        "source_data_order": "entity_count_then_program_length_then_fingerprint",
        "recovered_existing_shards": True,
    }


def cache_a120b_full_hidden(
    data_dir: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    max_length: int = 1024,
    inference_batch_size: int = 12,
    shard_size: int = 64,
    overfit32: bool = False,
    splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.20B Qwen cache requested unavailable CUDA")
    if (output_dir / "manifest.json").exists():
        raise RuntimeError(f"refusing to overwrite A1.20B cache: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = (
        select_a120b_overfit32(load_a119h2_records(data_dir, "train"))
        if overfit32
        else None
    )
    if overfit32 and splits is not None:
        raise ValueError("A1.20B overfit cache does not accept custom splits")
    selected_splits = tuple(CACHE_SPLITS if splits is None else splits)
    if not overfit32:
        invalid = sorted(set(selected_splits) - set(CACHE_SPLITS))
        if invalid:
            raise ValueError(f"unknown A1.20B cache splits: {invalid}")
        if not selected_splits:
            raise ValueError("A1.20B cache requires at least one split")
    split_sources = (
        {"train": "train", "validation": "train"}
        if overfit32
        else {split: split for split in selected_splits}
    )
    unexpected_shards = [
        path
        for path in output_dir.rglob("*.pt")
        if path.parent.name not in split_sources
    ]
    if unexpected_shards:
        raise RuntimeError(
            "refusing to mix A1.20B cache with unexpected shards: "
            + ", ".join(str(path) for path in unexpected_shards)
        )
    split_records = {
        split: selected or load_a119h2_records(data_dir, source_split)
        for split, source_split in split_sources.items()
    }
    splits_to_encode = [
        split
        for split in split_sources
        if not any((output_dir / split).glob("*.pt"))
    ]
    tokenizer = backbone = None
    if splits_to_encode:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        backbone = load_qwen35_text_only(
            model_id, revision, dtype=torch.float16, device=device
        )
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    splits: dict[str, Any] = {}
    started = time.perf_counter()
    for split, source_split in split_sources.items():
        records = split_records[split]
        split_dir = output_dir / split
        if any(split_dir.glob("*.pt")):
            report = _recover_cached_split(
                records,
                split_dir,
                max_length=max_length,
                shard_size=shard_size,
            )
        else:
            assert backbone is not None and tokenizer is not None
            report = _cache_split(
                backbone,
                tokenizer,
                records,
                split_dir,
                device=device,
                max_length=max_length,
                inference_batch_size=inference_batch_size,
                shard_size=shard_size,
            )
        report["source_data_split"] = source_split
        splits[split] = report
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "V2-A1.20B learned full-text boundary",
        "data_dir": str(data_dir.resolve()),
        "data_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "selection": (
            "N2-N4-T1-T16-overfit32-fit-only"
            if overfit32
            else "formal-selected-splits"
        ),
        "model_id": model_id,
        "revision": revision,
        "tokenizer_revision": revision,
        "dtype": "float16",
        "hidden_layer": "last",
        "max_length": max_length,
        "silent_truncation": False,
        "qwen_trainable_parameters": 0,
        "full_source_saved": True,
        "oracle_span_input": False,
        "oracle_role_tensor_input": False,
        "oracle_entity_mask_input": False,
        "oracle_operation_mask_input": False,
        "input_ids_saved": False,
        "inference_batch_size": inference_batch_size,
        "shard_size": shard_size,
        "splits": splits,
        "encoding_seconds_excluding_model_and_tokenizer_load": (
            time.perf_counter() - started
        ),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if device.startswith("cuda") and backbone is not None
        else None,
        "recovered_without_reencoding": not splits_to_encode,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if backbone is not None:
        del backbone
    if device.startswith("cuda") and splits_to_encode:
        torch.cuda.empty_cache()
    return manifest


def benchmark_a120b_cache_batches(
    data_dir: Path,
    output_path: Path,
    *,
    batch_sizes: Sequence[int] = (8, 12, 16),
    device: str = "cuda",
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    max_length: int = 1024,
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.20B cache benchmark requested unavailable CUDA")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = load_qwen35_text_only(
        model_id, revision, dtype=torch.float16, device=device
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    records = _ordered(
        select_a120b_overfit32(load_a119h2_records(data_dir, "train"))
    )
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        total_tokens = padded_tokens = 0
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, len(records), int(batch_size)):
            batch = records[start : start + int(batch_size)]
            input_ids, attention_mask, lengths = _encode_batch(
                tokenizer, batch, max_length
            )
            with torch.inference_mode():
                hidden = backbone.model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    use_cache=False,
                ).last_hidden_state
            total_tokens += sum(lengths)
            padded_tokens += int(input_ids.numel())
            del hidden
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "batch_size": int(batch_size),
                "seconds": elapsed,
                "examples_per_second": len(records) / elapsed,
                "real_tokens_per_second": total_tokens / elapsed,
                "padded_tokens": padded_tokens,
                "padding_ratio": padded_tokens / max(1, total_tokens),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if device.startswith("cuda")
                else None,
            }
        )
    fastest = min(rows, key=lambda row: row["seconds"])
    result = {
        "schema_version": "yggdrasil.v2-a1.20b.qwen-cache-batch-benchmark.v1",
        "data_dir": str(data_dir),
        "device": device,
        "model_id": model_id,
        "revision": revision,
        "examples": len(records),
        "maximum_source_tokens": max(
            len(
                tokenizer(
                    record["question"],
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            for record in records
        ),
        "rows": rows,
        "selected_batch_size": fastest["batch_size"],
        "selection_rule": "minimum measured wall time within GPU memory",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=False)


def audit_a120b_cache(cache_dir: Path, data_dir: Path) -> dict[str, Any]:
    manifest = json.loads(
        (cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("not an A1.20B cache")
    overfit = manifest["selection"].endswith("overfit32-fit-only")
    hidden_widths: set[int] = set()
    schemas = shapes = finite = fingerprints = forbidden_absent = True
    split_reports: dict[str, Any] = {}
    split_fingerprints: dict[str, set[str]] = {}
    for split, split_manifest in manifest["splits"].items():
        source_records = load_a119h2_records(
            data_dir, split_manifest["source_data_split"]
        )
        expected_records = (
            select_a120b_overfit32(source_records) if overfit else source_records
        )
        expected = [row["fingerprint"] for row in _ordered(expected_records)]
        observed: list[str] = []
        examples = tokens = 0
        for name in split_manifest["shards"]:
            payload = _load_payload(cache_dir / split / name)
            schemas = schemas and payload.get("schema_version") == CACHE_SCHEMA
            forbidden_absent = forbidden_absent and not (
                FORBIDDEN_FIELDS & set(payload)
            )
            hidden = payload["last_hidden"]
            mask = payload["attention_mask"]
            lengths = payload["token_lengths"]
            shapes = shapes and hidden.ndim == 3 and mask.shape == hidden.shape[:2]
            shapes = shapes and torch.equal(
                mask.sum(dim=1).long(), lengths.long()
            )
            finite = finite and bool(torch.isfinite(hidden).all())
            hidden_widths.add(int(hidden.shape[-1]))
            observed.extend(payload["fingerprints"])
            examples += len(payload["fingerprints"])
            tokens += int(lengths.sum())
        fingerprints = fingerprints and observed == expected
        split_fingerprints[split] = set(observed)
        split_reports[split] = {
            "examples": examples,
            "source_tokens": tokens,
            "fingerprints_match": observed == expected,
        }
    overlaps: dict[str, int] = {}
    names = sorted(split_fingerprints)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(
                split_fingerprints[left] & split_fingerprints[right]
            )
    source_audit_path = data_dir.parent / "data-audit.json"
    source_audit = (
        json.loads(source_audit_path.read_text(encoding="utf-8"))
        if source_audit_path.exists()
        else {"passed": False}
    )
    gates = {
        "manifest_schema": manifest["schema_version"] == MANIFEST_SCHEMA,
        "data_manifest_hash_matches": manifest["data_manifest_sha256"]
        == file_sha256(data_dir / "manifest.json"),
        "source_data_audit_passed": bool(source_audit.get("passed")),
        "shard_schemas": schemas,
        "shapes": shapes,
        "finite": finite,
        "single_hidden_width": len(hidden_widths) == 1,
        "fingerprints_match": fingerprints,
        "formal_cross_split_overlap_zero": overfit
        or all(value == 0 for value in overlaps.values()),
        "forbidden_oracle_fields_absent": forbidden_absent
        and not manifest["oracle_span_input"]
        and not manifest["oracle_role_tensor_input"]
        and not manifest["oracle_entity_mask_input"]
        and not manifest["oracle_operation_mask_input"],
        "qwen_frozen_no_truncation": manifest["qwen_trainable_parameters"] == 0
        and not manifest["silent_truncation"],
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "cache_dir": str(cache_dir),
        "data_dir": str(data_dir),
        "selection": manifest["selection"],
        "hidden_widths": sorted(hidden_widths),
        "splits": split_reports,
        "cross_split_fingerprint_overlap": overlaps,
        "gates": gates,
        "passed": all(gates.values()),
    }


class A120BCachedSplit:
    def __init__(self, cache_dir: Path, data_dir: Path, split: str) -> None:
        manifest = json.loads(
            (cache_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("A120BCachedSplit requires an A1.20B cache")
        split_manifest = manifest["splits"][split]
        source_records = load_a119h2_records(
            data_dir, split_manifest["source_data_split"]
        )
        record_map = {row["fingerprint"]: row for row in source_records}
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
        self.indices_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.indices_by_entity_count: dict[int, list[int]] = defaultdict(list)
        self.indices_by_length: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            count = int(record["entity_count"])
            length = int(record["program_length"])
            self.indices_by_cell[(count, length)].append(index)
            self.indices_by_entity_count[count].append(index)
            self.indices_by_length[length].append(index)

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


def collate_a120b_items(
    items: Sequence[dict[str, Any]], device: str | torch.device
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    inputs, records = collate_a120b_items_cpu(items, pin_memory=False)
    return move_a120b_inputs(inputs, device), records


def collate_a120b_items_cpu(
    items: Sequence[dict[str, Any]], *, pin_memory: bool
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if not items:
        raise ValueError("cannot collate empty A1.20B items")
    maximum_tokens = max(int(item["last_hidden"].shape[0]) for item in items)
    hidden_width = int(items[0]["last_hidden"].shape[-1])
    hidden = torch.zeros(
        (len(items), maximum_tokens, hidden_width),
        dtype=torch.float16,
        pin_memory=pin_memory,
    )
    mask = torch.zeros(
        (len(items), maximum_tokens), dtype=torch.bool, pin_memory=pin_memory
    )
    for row, item in enumerate(items):
        length = int(item["last_hidden"].shape[0])
        hidden[row, :length] = item["last_hidden"]
        mask[row, :length] = item["attention_mask"]
    return {
        "source_hidden": hidden,
        "source_attention_mask": mask,
    }, [item["record"] for item in items]


def move_a120b_inputs(
    inputs: dict[str, torch.Tensor], device: str | torch.device
) -> dict[str, torch.Tensor]:
    non_blocking = torch.device(device).type == "cuda"
    return {
        name: value.to(device, non_blocking=non_blocking)
        for name, value in inputs.items()
    }
