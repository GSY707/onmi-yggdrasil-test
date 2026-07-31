from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .a1_20b_cache import A120BCachedSplit, file_sha256


SUPERVISION_SCHEMA = "yggdrasil.v2-a1.20c.compiler-supervision.v1"
MANIFEST_SCHEMA = "yggdrasil.v2-a1.20c.compiler-supervision-manifest.v1"
AUDIT_SCHEMA = "yggdrasil.v2-a1.20c.compiler-supervision-audit.v1"
MAXIMUM_ENTITIES = 5
MAXIMUM_OPERATIONS = 32
ROLE_NAMES = ("family", "source", "target")


@dataclass(frozen=True)
class A120CAnchorLabels:
    entity_name_targets: torch.Tensor
    entity_value_targets: torch.Tensor
    operation_role_targets: torch.Tensor
    query_targets: torch.Tensor


def _find_after(text: str, needle: str, cursor: int) -> tuple[int, int]:
    position = text.find(needle, cursor)
    if position < 0:
        raise ValueError(f"A1.20C supervision could not find {needle!r}")
    return position, position + len(needle)


def _record_char_targets(
    record: dict[str, Any],
) -> tuple[list[int], list[int], list[list[int]], int]:
    question = str(record["question"])
    entity_names = list(record["entity_names"])
    routing = str(record.get("task_family", "")).startswith(
        "signal_routing_relay_exchange"
    )
    entity_name_positions: list[int] = []
    entity_value_positions: list[int] = []
    entity_marker = "Stations:" if routing else "Objects:"
    cursor = question.index(entity_marker) + len(entity_marker)
    for name in entity_names:
        name_position, cursor = _find_after(question, name, cursor)
        entity_name_positions.append(name_position)
        if routing:
            _, cursor = _find_after(question, " carries ", cursor)
        else:
            _, cursor = _find_after(question, "=", cursor)
        value = str(record["start_state"][name])
        value_position, cursor = _find_after(question, value, cursor)
        entity_value_positions.append(value_position)

    operation_marker = "Dispatch plan:" if routing else "Operations:"
    cursor = question.index(operation_marker) + len(operation_marker)
    operation_positions: list[list[int]] = []
    for operation in record["operations"]:
        family = (
            "RELAY"
            if routing and operation["family"] == "copy"
            else "EXCHANGE"
            if routing and operation["family"] == "swap"
            else str(operation["family"]).upper()
        )
        family_position, cursor = _find_after(
            question, family, cursor
        )
        source_position, cursor = _find_after(
            question, str(operation["source"]), cursor
        )
        separator = (
            "=>"
            if routing and operation["family"] == "copy"
            else "<=>"
            if routing
            else "->"
        )
        _, cursor = _find_after(question, separator, cursor)
        target_position, cursor = _find_after(
            question, str(operation["target"]), cursor
        )
        operation_positions.append(
            [family_position, source_position, target_position]
        )

    query_marker = "Inspect station:" if routing else "Query:"
    query_cursor = question.rindex(query_marker) + len(query_marker)
    query_position, _ = _find_after(
        question, str(record["query_register"]), query_cursor
    )
    return (
        entity_name_positions,
        entity_value_positions,
        operation_positions,
        query_position,
    )


def _char_to_token(offsets: Sequence[Sequence[int]], position: int) -> int:
    for token, (start, end) in enumerate(offsets):
        if int(end) > int(start) and int(start) <= position < int(end):
            return token
    raise ValueError(f"A1.20C supervision char position {position} has no token")


def _encode_split(
    tokenizer: Any,
    dataset: A120BCachedSplit,
    output_path: Path,
) -> dict[str, Any]:
    examples = len(dataset)
    entity_name_targets = torch.full(
        (examples, MAXIMUM_ENTITIES), -100, dtype=torch.long
    )
    entity_value_targets = torch.full_like(entity_name_targets, -100)
    operation_role_targets = torch.full(
        (examples, MAXIMUM_OPERATIONS, len(ROLE_NAMES)),
        -100,
        dtype=torch.long,
    )
    query_targets = torch.empty((examples,), dtype=torch.long)
    token_lengths = torch.empty((examples,), dtype=torch.long)
    fingerprints: list[str] = []
    for index in range(examples):
        item = dataset[index]
        record = item["record"]
        encoded = tokenizer(
            record["question"],
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        observed_length = len(encoded["input_ids"])
        cached_length = int(item["last_hidden"].shape[0])
        if observed_length != cached_length:
            raise ValueError(
                "A1.20C tokenizer/cache length mismatch: "
                f"{observed_length} != {cached_length}"
            )
        (
            entity_names,
            entity_values,
            operations,
            query,
        ) = _record_char_targets(record)
        for entity, position in enumerate(entity_names):
            entity_name_targets[index, entity] = _char_to_token(
                offsets, position
            )
        for entity, position in enumerate(entity_values):
            entity_value_targets[index, entity] = _char_to_token(
                offsets, position
            )
        for step, role_positions in enumerate(operations):
            for role, position in enumerate(role_positions):
                operation_role_targets[index, step, role] = _char_to_token(
                    offsets, position
                )
        query_targets[index] = _char_to_token(offsets, query)
        token_lengths[index] = observed_length
        fingerprints.append(str(record["fingerprint"]))
    payload = {
        "schema_version": SUPERVISION_SCHEMA,
        "fingerprints": fingerprints,
        "token_lengths": token_lengths,
        "entity_name_targets": entity_name_targets,
        "entity_value_targets": entity_value_targets,
        "operation_role_targets": operation_role_targets,
        "query_targets": query_targets,
    }
    torch.save(payload, output_path)
    return {
        "examples": examples,
        "file": output_path.name,
        "maximum_token_length": int(token_lengths.max()),
        "active_entity_targets": int((entity_name_targets >= 0).sum()),
        "active_operation_role_targets": int(
            (operation_role_targets >= 0).sum()
        ),
    }


def prepare_a120c_supervision(
    cache_dir: Path,
    data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("*.pt")):
        raise RuntimeError(f"refusing to mix A1.20C supervision: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_dir / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cache_manifest["model_id"],
        revision=cache_manifest["tokenizer_revision"],
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A1.20C supervision requires tokenizer offsets")
    splits: dict[str, Any] = {}
    for split in cache_manifest["splits"]:
        dataset = A120BCachedSplit(cache_dir, data_dir, split)
        splits[split] = _encode_split(
            tokenizer, dataset, output_dir / f"{split}.pt"
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": "V2-A1.20C structured Boundary compiler supervision",
        "cache_dir": str(cache_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "cache_manifest_sha256": file_sha256(cache_manifest_path),
        "model_id": cache_manifest["model_id"],
        "tokenizer_revision": cache_manifest["tokenizer_revision"],
        "training_targets_only": True,
        "forward_inputs": ["source_hidden", "source_attention_mask"],
        "oracle_targets_are_forward_inputs": False,
        "splits": splits,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=False)


def audit_a120c_supervision(
    cache_dir: Path,
    data_dir: Path,
    supervision_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(
        (supervision_dir / "manifest.json").read_text(encoding="utf-8")
    )
    cache_manifest_path = cache_dir / "manifest.json"
    split_reports: dict[str, Any] = {}
    schemas = fingerprints = shapes = targets_in_range = True
    for split, report in manifest["splits"].items():
        dataset = A120BCachedSplit(cache_dir, data_dir, split)
        payload = _load(supervision_dir / report["file"])
        schemas = schemas and payload.get("schema_version") == SUPERVISION_SCHEMA
        expected_fingerprints = [
            str(record["fingerprint"]) for record in dataset.records
        ]
        fingerprints = (
            fingerprints and payload["fingerprints"] == expected_fingerprints
        )
        examples = len(dataset)
        shapes = (
            shapes
            and payload["entity_name_targets"].shape
            == (examples, MAXIMUM_ENTITIES)
            and payload["entity_value_targets"].shape
            == (examples, MAXIMUM_ENTITIES)
            and payload["operation_role_targets"].shape
            == (examples, MAXIMUM_OPERATIONS, len(ROLE_NAMES))
            and payload["query_targets"].shape == (examples,)
        )
        lengths = payload["token_lengths"]
        for name in (
            "entity_name_targets",
            "entity_value_targets",
            "operation_role_targets",
            "query_targets",
        ):
            targets = payload[name]
            active = targets >= 0
            expanded_lengths = lengths
            while expanded_lengths.ndim < targets.ndim:
                expanded_lengths = expanded_lengths.unsqueeze(-1)
            targets_in_range = targets_in_range and bool(
                (targets[active] < expanded_lengths.expand_as(targets)[active]).all()
            )
        split_reports[split] = {
            "examples": examples,
            "fingerprints_match": payload["fingerprints"]
            == expected_fingerprints,
            "maximum_token_length": int(lengths.max()),
        }
    gates = {
        "manifest_schema": manifest.get("schema_version") == MANIFEST_SCHEMA,
        "cache_manifest_hash_matches": manifest["cache_manifest_sha256"]
        == file_sha256(cache_manifest_path),
        "training_targets_only": bool(manifest["training_targets_only"]),
        "oracle_targets_are_not_forward_inputs": not manifest[
            "oracle_targets_are_forward_inputs"
        ]
        and manifest["forward_inputs"]
        == ["source_hidden", "source_attention_mask"],
        "split_schemas": schemas,
        "fingerprints_match": fingerprints,
        "shapes": shapes,
        "targets_in_token_range": targets_in_range,
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "cache_dir": str(cache_dir),
        "supervision_dir": str(supervision_dir),
        "splits": split_reports,
        "gates": gates,
        "passed": all(gates.values()),
    }


class A120CSupervisionSplit:
    def __init__(
        self,
        supervision_dir: Path,
        cached_split: A120BCachedSplit,
        split: str,
    ) -> None:
        manifest = json.loads(
            (supervision_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("not an A1.20C supervision manifest")
        self.payload = _load(
            supervision_dir / manifest["splits"][split]["file"]
        )
        observed = self.payload["fingerprints"]
        expected = [
            str(record["fingerprint"]) for record in cached_split.records
        ]
        if observed != expected:
            raise ValueError("A1.20C supervision/cache fingerprint mismatch")

    def select(
        self,
        indices: Iterable[int],
        device: str | torch.device,
    ) -> A120CAnchorLabels:
        index = torch.tensor(list(indices), dtype=torch.long)
        return A120CAnchorLabels(
            entity_name_targets=self.payload["entity_name_targets"]
            .index_select(0, index)
            .to(device),
            entity_value_targets=self.payload["entity_value_targets"]
            .index_select(0, index)
            .to(device),
            operation_role_targets=self.payload["operation_role_targets"]
            .index_select(0, index)
            .to(device),
            query_targets=self.payload["query_targets"]
            .index_select(0, index)
            .to(device),
        )
