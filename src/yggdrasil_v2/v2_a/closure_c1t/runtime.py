from __future__ import annotations

"""Independent-card cache and batch runtime for C1T.

The encoder callback receives exactly one card string per invocation.  It is
therefore impossible for this runtime to obtain a whole-record hidden state and
slice it after contextualisation.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from . import contract
from .cards import public_address_vector, text_sha256, validate_public_cards
from .tasks import LABELS


CardKey = tuple[str, str, int]
CardEncoder = Callable[[str], Mapping[str, Any]]


def _tensor_sha256(value: Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(cpu.dtype), "shape": list(cpu.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw).hexdigest().upper()


@dataclass(frozen=True)
class EncodedCard:
    hidden: Tensor
    mask: Tensor
    token_ids: Tensor
    text_sha256: str
    token_sha256: str
    hidden_sha256: str


@dataclass(frozen=True)
class IndependentCardCache:
    source_identity: Mapping[str, Any]
    source_width: int
    cards: Mapping[CardKey, EncodedCard]
    ledger: tuple[Mapping[str, Any], ...]

    def manifest(self) -> dict[str, Any]:
        source_json = json.dumps(
            dict(self.source_identity), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        ledger_json = json.dumps(
            list(self.ledger), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.independent-card-cache.v1",
            "identity": contract.IDENTITY,
            "card_count": len(self.cards),
            "source_width": self.source_width,
            "source_identity": dict(self.source_identity),
            "source_identity_sha256": hashlib.sha256(source_json.encode("utf-8")).hexdigest().upper(),
            "ledger_sha256": hashlib.sha256(ledger_json.encode("utf-8")).hexdigest().upper(),
            "whole_record_hidden_reuse": False,
            "one_card_per_encoder_call": True,
        }


def _card_specs(record: Mapping[str, Any]) -> list[tuple[CardKey, str]]:
    example_id = str(record.get("example_id", ""))
    if not example_id:
        raise ValueError("record is missing example_id")
    public = record.get("model_public")
    validate_public_cards(public)
    specs: list[tuple[CardKey, str]] = []
    for index, text in enumerate(public["object_cards"]):
        specs.append(((example_id, "object", index), text))
    for index, text in enumerate(public["operation_cards"]):
        specs.append(((example_id, "operation", index), text))
    specs.append(((example_id, "query", 0), public["query_card"]))
    return specs


def _coerce_encoded(value: Mapping[str, Any], text: str) -> EncodedCard:
    if not isinstance(value, Mapping):
        raise TypeError("card encoder must return a mapping")
    hidden = value.get("hidden")
    mask = value.get("mask")
    token_ids = value.get("token_ids")
    if not isinstance(hidden, Tensor) or hidden.ndim != 2 or hidden.shape[0] < 1:
        raise ValueError("encoded card hidden must be [tokens, source_width]")
    if not isinstance(mask, Tensor) or mask.shape != hidden.shape[:1] or mask.dtype != torch.bool:
        raise ValueError("encoded card mask must be bool [tokens]")
    if not isinstance(token_ids, Tensor) or token_ids.shape != hidden.shape[:1]:
        raise ValueError("encoded card token_ids must align with hidden tokens")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("encoded card token_ids must be integer")
    if bool(mask.sum() < 1) or not bool(torch.isfinite(hidden).all()):
        raise ValueError("encoded card must contain finite public tokens")
    hidden = hidden.detach().cpu().contiguous()
    mask = mask.detach().cpu().contiguous()
    token_ids = token_ids.detach().cpu().to(dtype=torch.int64).contiguous()
    return EncodedCard(
        hidden=hidden,
        mask=mask,
        token_ids=token_ids,
        text_sha256=text_sha256(text),
        token_sha256=_tensor_sha256(token_ids),
        hidden_sha256=_tensor_sha256(hidden),
    )


def build_independent_card_cache(
    records: Sequence[Mapping[str, Any]],
    encode_card: CardEncoder,
    *,
    source_identity: Mapping[str, Any],
) -> IndependentCardCache:
    """Encode every card in its own callback invocation and freeze a ledger."""

    if not records:
        raise ValueError("card cache requires at least one record")
    if not callable(encode_card):
        raise TypeError("encode_card must be callable")
    if not isinstance(source_identity, Mapping) or not source_identity:
        raise ValueError("source_identity must be a non-empty mapping")
    cards: dict[CardKey, EncodedCard] = {}
    ledger: list[dict[str, Any]] = []
    source_width: int | None = None
    for record in records:
        for key, text in _card_specs(record):
            if key in cards:
                raise ValueError(f"duplicate independent card key: {key}")
            # Deliberately pass no record, family, answer, or neighboring card.
            encoded = _coerce_encoded(encode_card(text), text)
            if source_width is None:
                source_width = int(encoded.hidden.shape[1])
            if encoded.hidden.shape[1] != source_width:
                raise ValueError("source width changed across independently encoded cards")
            cards[key] = encoded
            ledger.append(
                {
                    "example_id": key[0],
                    "card_kind": key[1],
                    "card_index": key[2],
                    "text_sha256": encoded.text_sha256,
                    "token_sha256": encoded.token_sha256,
                    "hidden_sha256": encoded.hidden_sha256,
                    "token_count": int(encoded.mask.sum()),
                }
            )
    if source_width is None:
        raise RuntimeError("independent card cache unexpectedly remained empty")
    cache = IndependentCardCache(
        source_identity=dict(source_identity),
        source_width=source_width,
        cards=cards,
        ledger=tuple(ledger),
    )
    audit = audit_independent_card_cache(records, cache)
    if not audit["passed"]:
        raise RuntimeError(f"fresh independent card cache failed audit: {audit}")
    return cache


def audit_independent_card_cache(
    records: Sequence[Mapping[str, Any]], cache: IndependentCardCache
) -> dict[str, Any]:
    expected: dict[CardKey, str] = {}
    for record in records:
        for key, text in _card_specs(record):
            if key in expected:
                raise ValueError(f"duplicate record/card key in audit input: {key}")
            expected[key] = text
    actual = set(cache.cards)
    expected_keys = set(expected)
    errors: list[str] = []
    if actual != expected_keys:
        errors.append(
            f"card_key_set_mismatch:missing={len(expected_keys - actual)}:extra={len(actual - expected_keys)}"
        )
    ledger_by_key: dict[CardKey, Mapping[str, Any]] = {}
    for row in cache.ledger:
        key = (str(row.get("example_id", "")), str(row.get("card_kind", "")), int(row.get("card_index", -1)))
        if key in ledger_by_key:
            errors.append(f"duplicate_ledger_key:{key}")
        ledger_by_key[key] = row
    if set(ledger_by_key) != expected_keys:
        errors.append("ledger_key_set_mismatch")
    for key in sorted(expected_keys & actual):
        card = cache.cards[key]
        row = ledger_by_key.get(key, {})
        if card.hidden.ndim != 2 or card.hidden.shape[1] != cache.source_width:
            errors.append(f"hidden_shape:{key}")
            continue
        if card.mask.shape != card.hidden.shape[:1] or card.token_ids.shape != card.hidden.shape[:1]:
            errors.append(f"token_alignment:{key}")
        if not bool(torch.isfinite(card.hidden).all()) or bool(card.mask.sum() < 1):
            errors.append(f"finite_or_empty:{key}")
        checks = {
            "text_sha256": text_sha256(expected[key]),
            "token_sha256": _tensor_sha256(card.token_ids),
            "hidden_sha256": _tensor_sha256(card.hidden),
        }
        for field, observed in checks.items():
            if getattr(card, field) != observed or row.get(field) != observed:
                errors.append(f"{field}:{key}")
    by_text: dict[str, set[tuple[str, str]]] = {}
    for row in cache.ledger:
        by_text.setdefault(str(row.get("text_sha256", "")), set()).add(
            (str(row.get("token_sha256", "")), str(row.get("hidden_sha256", "")))
        )
    nondeterministic_texts = sorted(
        text_hash for text_hash, encodings in by_text.items() if len(encodings) != 1
    )
    if nondeterministic_texts:
        errors.append(
            f"same_text_encoding_nondeterministic:{len(nondeterministic_texts)}"
        )
    kinds = {key[1] for key in expected_keys}
    return {
        "passed": not errors,
        "card_count": len(cache.cards),
        "expected_card_count": len(expected_keys),
        "source_width": cache.source_width,
        "card_kinds": sorted(kinds),
        "one_ledger_row_per_card": len(cache.ledger) == len(expected_keys),
        "whole_record_hidden_reuse": False,
        "same_text_same_hidden": not nondeterministic_texts,
        "errors": errors[:32],
        "manifest": cache.manifest(),
    }


def _pad_cards(
    cards: Sequence[Sequence[EncodedCard]],
    *,
    count: int,
    source_width: int,
) -> tuple[Tensor, Tensor]:
    if any(len(row) > count for row in cards):
        raise ValueError("record exceeds frozen card count")
    max_tokens = max(card.hidden.shape[0] for row in cards for card in row)
    hidden = torch.zeros(len(cards), count, max_tokens, source_width, dtype=cards[0][0].hidden.dtype)
    mask = torch.zeros(len(cards), count, max_tokens, dtype=torch.bool)
    for batch_index, row in enumerate(cards):
        for card_index, card in enumerate(row):
            tokens = card.hidden.shape[0]
            hidden[batch_index, card_index, :tokens] = card.hidden
            mask[batch_index, card_index, :tokens] = card.mask
    return hidden, mask


class C1TRuntime:
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        cache: IndependentCardCache,
        *,
        address_width: int = contract.ADDRESS_WIDTH,
    ) -> None:
        audit = audit_independent_card_cache(records, cache)
        if not audit["passed"]:
            raise ValueError(f"independent card cache is not qualified: {audit}")
        self.records = {str(record["example_id"]): record for record in records}
        if len(self.records) != len(records):
            raise ValueError("runtime records require unique example_id")
        self.cache = cache
        self.address_width = address_width

    def _cards(self, example_id: str, kind: str, count: int) -> list[EncodedCard]:
        result = []
        for index in range(count):
            key = (example_id, kind, index)
            if key not in self.cache.cards:
                raise KeyError(f"missing independent card: {key}")
            result.append(self.cache.cards[key])
        return result

    def get_batch(self, example_ids: Sequence[str]) -> dict[str, Any]:
        if not example_ids:
            raise ValueError("batch requires at least one example_id")
        rows = []
        for example_id in example_ids:
            if example_id not in self.records:
                raise KeyError(f"unknown C1T example_id: {example_id}")
            rows.append(self.records[example_id])
        object_rows: list[list[EncodedCard]] = []
        operation_rows: list[list[EncodedCard]] = []
        query_rows: list[list[EncodedCard]] = []
        for example_id, row in zip(example_ids, rows, strict=True):
            public = row["model_public"]
            object_rows.append(self._cards(example_id, "object", len(public["object_cards"])))
            operation_rows.append(
                self._cards(example_id, "operation", len(public["operation_cards"]))
            )
            query_rows.append(self._cards(example_id, "query", 1))
        object_hidden, object_mask = _pad_cards(
            object_rows, count=contract.MAX_SLOTS, source_width=self.cache.source_width
        )
        operation_hidden, operation_mask = _pad_cards(
            operation_rows, count=contract.MAX_OPERATIONS, source_width=self.cache.source_width
        )
        query_hidden_4d, query_mask_3d = _pad_cards(
            query_rows, count=1, source_width=self.cache.source_width
        )
        batch = len(rows)
        object_addresses = torch.zeros(
            batch, contract.MAX_SLOTS, self.address_width, dtype=torch.float32
        )
        operation_source_addresses = torch.zeros(
            batch, contract.MAX_OPERATIONS, self.address_width, dtype=torch.float32
        )
        operation_target_addresses = torch.zeros_like(operation_source_addresses)
        query_address = torch.zeros(batch, self.address_width, dtype=torch.float32)
        for batch_index, row in enumerate(rows):
            public = row["model_public"]
            for index, address in enumerate(public["object_addresses"]):
                object_addresses[batch_index, index] = public_address_vector(
                    address, width=self.address_width
                )
            for index, address in enumerate(public["operation_source_addresses"]):
                operation_source_addresses[batch_index, index] = public_address_vector(
                    address, width=self.address_width
                )
            for index, address in enumerate(public["operation_target_addresses"]):
                operation_target_addresses[batch_index, index] = public_address_vector(
                    address, width=self.address_width
                )
            query_address[batch_index] = public_address_vector(
                public["query_address"], width=self.address_width
            )
        result: dict[str, Any] = {
            "object_hidden": object_hidden,
            "object_mask": object_mask,
            "object_addresses": object_addresses,
            "object_present": object_mask.any(dim=2),
            "operation_hidden": operation_hidden,
            "operation_mask": operation_mask,
            "operation_source_addresses": operation_source_addresses,
            "operation_target_addresses": operation_target_addresses,
            "query_hidden": query_hidden_4d[:, 0],
            "query_mask": query_mask_3d[:, 0],
            "query_address": query_address,
            "answers": torch.tensor([int(row["answer_index"]) for row in rows], dtype=torch.long),
            "valid_choice_mask": torch.tensor(
                [list(row["valid_choice_mask"]) for row in rows], dtype=torch.bool
            ),
            "support_slots": torch.tensor(
                [list(row["support_slots"]) for row in rows], dtype=torch.long
            ),
            "example_ids": list(example_ids),
            "families": [str(row["family"]) for row in rows],
            "factorial_group_ids": [str(row["factorial_group_id"]) for row in rows],
            "factor_cells": [list(row["factors"]) for row in rows],
            "semantic_answers": [str(row["semantic_answer"]) for row in rows],
            "raw_answer_labels": [LABELS[int(row["answer_index"])] for row in rows],
        }
        group_cell_to_index = {
            (str(row["factorial_group_id"]), tuple(int(value) for value in row["factors"])): index
            for index, row in enumerate(rows)
        }
        counterfactual_indices = torch.full((batch, 2), -1, dtype=torch.long)
        for index, row in enumerate(rows):
            group = str(row["factorial_group_id"])
            cell = [int(value) for value in row["factors"]]
            for factor in range(2):
                alternate = list(cell)
                alternate[factor] = 1 - alternate[factor]
                counterfactual_indices[index, factor] = group_cell_to_index.get(
                    (group, tuple(alternate)), -1
                )
        result["counterfactual_indices"] = counterfactual_indices
        return result

    @staticmethod
    def forward_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
        missing = [field for field in contract.FORWARD_FIELDS if field not in batch]
        if missing:
            raise KeyError(f"batch is missing C1T forward fields: {missing}")
        result = {field: batch[field] for field in contract.FORWARD_FIELDS}
        leaked = set(result) & set(contract.FORBIDDEN_FORWARD_FIELDS)
        if leaked:
            raise RuntimeError(f"forbidden target fields crossed the model boundary: {sorted(leaked)}")
        return result


__all__ = [
    "C1TRuntime",
    "CardEncoder",
    "EncodedCard",
    "IndependentCardCache",
    "audit_independent_card_cache",
    "build_independent_card_cache",
]
