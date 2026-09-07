from __future__ import annotations

"""Public-card and record-local address primitives for C1T."""

import hashlib
import unicodedata
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from . import contract


def normalize_public_address(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("public address must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    if not normalized or any(ord(char) < 32 for char in normalized):
        raise ValueError("public address is empty or contains control characters")
    return normalized


def public_address_vector(
    value: str,
    *,
    width: int = contract.ADDRESS_WIDTH,
    salt: str = contract.ADDRESS_SALT,
) -> Tensor:
    """Return a deterministic finite unit vector from a public address string."""

    if type(width) is not int or width < 8:
        raise ValueError("address width must be an integer >= 8")
    normalized = normalize_public_address(value)
    material = bytearray()
    counter = 0
    while len(material) < width:
        payload = f"{salt}\0{normalized}\0{counter}".encode("utf-8")
        material.extend(hashlib.sha256(payload).digest())
        counter += 1
    raw = torch.tensor(list(material[:width]), dtype=torch.float32)
    raw = (raw - 127.5) / 127.5
    norm = torch.linalg.vector_norm(raw)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("public address vector is degenerate")
    return raw / norm


def address_matrix(values: Sequence[str], *, width: int = contract.ADDRESS_WIDTH) -> Tensor:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("addresses must be a non-empty sequence")
    normalized = [normalize_public_address(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError("public addresses collide after NFKC+casefold normalization")
    matrix = torch.stack([public_address_vector(value, width=width) for value in values])
    similarity = matrix @ matrix.T
    off_diagonal = similarity - torch.eye(len(values), dtype=similarity.dtype)
    if len(values) > 1 and bool((off_diagonal.abs() > 0.999).any()):
        raise ValueError("public address vectors are numerically ambiguous")
    return matrix


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("card text must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def validate_public_cards(public: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(public, Mapping):
        raise TypeError("model_public must be a mapping")
    objects = public.get("object_cards")
    addresses = public.get("object_addresses")
    operations = public.get("operation_cards")
    sources = public.get("operation_source_addresses")
    targets = public.get("operation_target_addresses")
    query = public.get("query_card")
    query_address = public.get("query_address")
    if not isinstance(objects, list) or not objects or not all(isinstance(item, str) and item for item in objects):
        raise ValueError("object_cards must be non-empty public strings")
    if not isinstance(addresses, list) or len(addresses) != len(objects):
        raise ValueError("object_addresses must align with object_cards")
    if not isinstance(operations, list) or not operations or not all(isinstance(item, str) and item for item in operations):
        raise ValueError("operation_cards must be non-empty public strings")
    if not isinstance(sources, list) or not isinstance(targets, list):
        raise ValueError("operation address lists are required")
    if len(sources) != len(operations) or len(targets) != len(operations):
        raise ValueError("operation addresses must align with operation_cards")
    if not isinstance(query, str) or not query or not isinstance(query_address, str):
        raise ValueError("query card/address are required")
    normalized = [normalize_public_address(value) for value in addresses]
    if len(set(normalized)) != len(normalized):
        raise ValueError("object addresses collide")
    known = set(normalized)
    routed = [*sources, *targets, query_address]
    if any(normalize_public_address(value) not in known for value in routed):
        raise ValueError("operation/query address is absent from object addresses")
    vectors = address_matrix(addresses)
    return {
        "passed": True,
        "object_cards": len(objects),
        "operation_cards": len(operations),
        "object_card_sha256": [text_sha256(item) for item in objects],
        "operation_card_sha256": [text_sha256(item) for item in operations],
        "query_card_sha256": text_sha256(query),
        "address_width": int(vectors.shape[1]),
        "finite": bool(torch.isfinite(vectors).all()),
        "maximum_norm_error": max(
            abs(float(torch.linalg.vector_norm(row)) - 1.0) for row in vectors
        ),
    }


__all__ = [
    "address_matrix",
    "normalize_public_address",
    "public_address_vector",
    "text_sha256",
    "validate_public_cards",
]
