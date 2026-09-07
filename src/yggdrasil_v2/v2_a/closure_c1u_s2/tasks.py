from __future__ import annotations

"""Fresh-bank construction and fail-closed disjointness audits for C1U S2."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from yggdrasil_v2.v2_a.closure_c1u import contract as c1u_contract
from yggdrasil_v2.v2_a.closure_c1u.tasks import (
    LABELS,
    audit_s1_bank,
    build_s1_bank,
)

from . import contract


_OPAQUE_RE = re.compile(r"\bn[0-9a-f]{10}\b")


def _rotate_label(label: str, rotation: int) -> str:
    return LABELS[(LABELS.index(label) + int(rotation)) % len(LABELS)]


def _replace_choice_line(query_card: str, mapping: Mapping[str, str]) -> str:
    choices = " | ".join(
        f"{label} means {meaning}" for label, meaning in sorted(mapping.items())
    )
    rows = query_card.splitlines()
    matched = [index for index, row in enumerate(rows) if row.startswith("Choices: ")]
    if len(matched) != 1:
        raise ValueError("query card requires exactly one Choices line")
    rows[matched[0]] = f"Choices: {choices}"
    return "\n".join(rows)


def _relabel_record(
    record: Mapping[str, Any], *, bank_id: str, bank_seed: int, rotation: int
) -> dict[str, Any]:
    row = deepcopy(dict(record))
    old_mapping = dict(row["label_mapping"])
    new_mapping = {
        _rotate_label(str(label), rotation): str(meaning)
        for label, meaning in old_mapping.items()
    }
    if len(new_mapping) != len(old_mapping):
        raise RuntimeError("bank label rotation collided")
    old_answer = str(row["raw_answer_label"])
    new_answer = _rotate_label(old_answer, rotation)
    valid = [label in new_mapping for label in LABELS]
    public = deepcopy(dict(row["model_public"]))
    public["query_card"] = _replace_choice_line(str(public["query_card"]), new_mapping)
    row.update(
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.task-record.v1",
            "example_id": f"c1u-s2-{bank_id.casefold()}-{row['example_id']}",
            "factorial_group_id": (
                f"c1u-s2-{bank_id.casefold()}-{row['factorial_group_id']}"
            ),
            "bank_id": bank_id,
            "bank_seed": int(bank_seed),
            "label_rotation": int(rotation),
            "model_public": public,
            "label_mapping": new_mapping,
            "raw_answer_label": new_answer,
            "answer_index": LABELS.index(new_answer),
            "valid_choice_mask": valid,
        }
    )
    return row


def build_bank(bank_id: str) -> list[dict[str, Any]]:
    if bank_id not in contract.BANK_IDS:
        raise ValueError(f"unknown C1U S2 bank: {bank_id}")
    index = contract.BANK_IDS.index(bank_id)
    seed = contract.BANK_SEEDS[index]
    rotation = contract.LABEL_ROTATIONS[index]
    return [
        _relabel_record(
            row, bank_id=bank_id, bank_seed=seed, rotation=rotation
        )
        for row in build_s1_bank(seed=seed)
    ]


def build_multibank() -> list[dict[str, Any]]:
    rows = [row for bank_id in contract.BANK_IDS for row in build_bank(bank_id)]
    if len(rows) != contract.TOTAL_RECORDS:
        raise RuntimeError("C1U S2 multibank cardinality drifted")
    return rows


def split_records(
    records: Sequence[Mapping[str, Any]], fold_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds = [fold for fold in contract.FOLDS if fold["fold_id"] == fold_id]
    if len(folds) != 1:
        raise ValueError(f"unknown C1U S2 fold: {fold_id}")
    fold = folds[0]
    train = [dict(row) for row in records if row.get("bank_id") in fold["train"]]
    heldout = [
        dict(row) for row in records if row.get("bank_id") in fold["heldout"]
    ]
    return train, heldout


def _public_text(record: Mapping[str, Any]) -> str:
    public = record["model_public"]
    cards = (
        list(public["object_cards"])
        + list(public["operation_cards"])
        + [public["query_card"]]
    )
    return "\n".join(str(card) for card in cards)


def _opaque_tokens(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return set(_OPAQUE_RE.findall("\n".join(_public_text(row) for row in records)))


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _owner_only_majority(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        key = (
            str(row["bank_id"]),
            str(row["family"]),
            str(row["factorial_group_id"]),
        )
        groups.setdefault(key, []).append(row)
    for (bank_id, family, group), rows in sorted(groups.items()):
        counts = Counter(int(row["answer_index"]) for row in rows)
        accuracy = max(counts.values()) / len(rows)
        key = f"{bank_id}/{family}/{group}"
        cells[key] = {
            "records": len(rows),
            "majority_accuracy": accuracy,
            "answer_histogram": dict(sorted(counts.items())),
        }
        if accuracy > float(contract.GATES["owner_only_majority_accuracy_max"]):
            failures.append(key)
    return {
        "passed": not failures,
        "maximum": max(
            (row["majority_accuracy"] for row in cells.values()), default=1.0
        ),
        "failures": failures,
        "cells": cells,
    }


def audit_multibank(
    records: Sequence[Mapping[str, Any]],
    *,
    s1_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    records = [dict(row) for row in records]
    bank_audits: dict[str, Any] = {}
    token_sets: dict[str, set[str]] = {}
    permutations: dict[str, tuple[str, ...]] = {}
    for bank_id, rotation in zip(
        contract.BANK_IDS, contract.LABEL_ROTATIONS, strict=True
    ):
        bank = [row for row in records if row.get("bank_id") == bank_id]
        bank_audits[bank_id] = audit_s1_bank(bank)
        token_sets[bank_id] = _opaque_tokens(bank)
        permutations[bank_id] = tuple(_rotate_label(label, rotation) for label in LABELS)

    pairwise_overlap: dict[str, list[str]] = {}
    for left_index, left in enumerate(contract.BANK_IDS):
        for right in contract.BANK_IDS[left_index + 1 :]:
            overlap = sorted(token_sets[left] & token_sets[right])
            pairwise_overlap[f"{left}/{right}"] = overlap

    s1_tokens = _opaque_tokens(s1_records or [])
    s1_overlap = {
        bank_id: sorted(tokens & s1_tokens) for bank_id, tokens in token_sets.items()
    }
    split_audits: dict[str, Any] = {}
    heldout_counts = Counter()
    for fold in contract.FOLDS:
        train, heldout = split_records(records, str(fold["fold_id"]))
        train_ids = {str(row["example_id"]) for row in train}
        heldout_ids = {str(row["example_id"]) for row in heldout}
        heldout_counts.update(str(row["bank_id"]) for row in heldout)
        split_audits[str(fold["fold_id"])] = {
            "passed": len(train) == contract.TRAIN_RECORDS_PER_FOLD
            and len(heldout) == contract.HELDOUT_RECORDS_PER_FOLD
            and not (train_ids & heldout_ids)
            and {str(row["bank_id"]) for row in train} == set(fold["train"])
            and {str(row["bank_id"]) for row in heldout}
            == set(fold["heldout"]),
            "train_records": len(train),
            "heldout_records": len(heldout),
            "train_banks": sorted({str(row["bank_id"]) for row in train}),
            "heldout_banks": sorted({str(row["bank_id"]) for row in heldout}),
        }

    owner_only = _owner_only_majority(records)
    rotations = list(permutations.values())
    checks = {
        "record_count": len(records) == contract.TOTAL_RECORDS,
        "unique_example_ids": len({str(row["example_id"]) for row in records})
        == len(records),
        "bank_ids": {str(row.get("bank_id")) for row in records}
        == set(contract.BANK_IDS),
        "bank_seeds": {int(row.get("bank_seed", -1)) for row in records}
        == set(contract.BANK_SEEDS),
        "all_bank_audits": all(
            audit.get("passed") is True for audit in bank_audits.values()
        ),
        "pairwise_opaque_disjoint": all(not overlap for overlap in pairwise_overlap.values()),
        "s1_opaque_disjoint": s1_records is not None
        and all(not overlap for overlap in s1_overlap.values()),
        "label_permutations_unique": len(set(rotations)) == len(rotations),
        "label_permutations_nonidentity": all(tuple(LABELS) != value for value in rotations),
        "folds": all(row["passed"] for row in split_audits.values()),
        "each_bank_heldout_once": all(
            heldout_counts[bank_id] == contract.BANK_RECORDS
            for bank_id in contract.BANK_IDS
        ),
        "owner_only_shortcut": owner_only["passed"],
    }
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.multibank-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "payload_sha256": _canonical_sha256(records),
        "bank_audits": bank_audits,
        "pairwise_opaque_overlap": pairwise_overlap,
        "s1_opaque_overlap": s1_overlap,
        "label_permutations": {
            key: list(value) for key, value in permutations.items()
        },
        "owner_only": owner_only,
        "folds": split_audits,
    }


__all__ = [
    "audit_multibank",
    "build_bank",
    "build_multibank",
    "split_records",
]
