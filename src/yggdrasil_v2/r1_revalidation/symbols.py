from __future__ import annotations

"""Nonce symbols and independent semantic/surface/label RNG streams."""

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Mapping

from .schema import LABELS


_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z")
_NUCLEI = ("a", "e", "i", "o", "u", "ai", "ea", "oa")
_CODAS = ("", "l", "m", "n", "r", "s", "t", "v", "x")


def derive_seed(seed: int, stream: str, index: int = 0) -> int:
    payload = f"r1r-p0-v2|{int(seed)}|{stream}|{int(index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class RNGStreams:
    semantic_seed: int
    surface_seed: int
    label_seed: int

    @property
    def semantic(self) -> random.Random:
        return random.Random(self.semantic_seed)

    @property
    def surface(self) -> random.Random:
        return random.Random(self.surface_seed)

    @property
    def label(self) -> random.Random:
        return random.Random(self.label_seed)


def make_rng_streams(base_seed: int, *, index: int = 0, attempt: int = 0) -> RNGStreams:
    root = derive_seed(base_seed, "episode", index) ^ derive_seed(base_seed, "attempt", attempt)
    return RNGStreams(
        semantic_seed=derive_seed(root, "semantic"),
        surface_seed=derive_seed(root, "surface"),
        label_seed=derive_seed(root, "label"),
    )


@dataclass
class NonceFactory:
    rng: random.Random
    used: set[str]

    @classmethod
    def from_seed(cls, seed: int) -> "NonceFactory":
        return cls(random.Random(seed), set())

    def name(self) -> str:
        for _ in range(10_000):
            syllables = self.rng.randint(2, 3)
            candidate = "".join(
                self.rng.choice(_ONSETS) + self.rng.choice(_NUCLEI) + self.rng.choice(_CODAS)
                for _ in range(syllables)
            )[:8]
            if candidate not in self.used and candidate.isalpha():
                self.used.add(candidate)
                return candidate
        raise RuntimeError("nonce generator exhausted its collision budget")

    def names(self, count: int) -> list[str]:
        return [self.name() for _ in range(count)]


def pronounceable_names(count: int, seed: int) -> list[str]:
    return NonceFactory.from_seed(seed).names(count)


def valid_choice_mask(mapping: Mapping[str, str]) -> list[bool]:
    mask = [False] * len(LABELS)
    for label in mapping.values():
        mask[LABELS.index(label)] = True
    return mask


def label_index(label: str) -> int:
    if label not in LABELS:
        raise ValueError(f"not a local A-I label: {label}")
    return LABELS.index(label)


def _balanced_label_indices(record_index: int, label_seed: int, option_count: int) -> list[int]:
    """Use only the label stream for a reproducible near-uniform schedule."""

    block = int(record_index) // len(LABELS)
    cycle = list(range(len(LABELS)))
    random.Random(derive_seed(label_seed, "label-cycle", block)).shuffle(cycle)
    start = int(record_index) % len(LABELS)
    return [cycle[(start + offset) % len(LABELS)] for offset in range(option_count)]


def balanced_label_index(record_index: int, salt: int = 0, option_count: int = 4) -> int:
    # Backwards-compatible helper used by the old unit tests.  Its only stream
    # is still a label schedule; no semantic or surface RNG is shared with it.
    return _balanced_label_indices(record_index, int(salt), option_count)[0]


def balanced_label_mapping(
    options: Iterable[str],
    anchor_option: str,
    anchor_index: int,
    *,
    record_index: int = 0,
    salt: int = 0,
) -> dict[str, str]:
    option_list = list(options)
    if anchor_option not in option_list or not 0 <= int(anchor_index) < len(LABELS):
        raise ValueError("invalid anchor option or label")
    labels = _balanced_label_indices(record_index, int(salt), len(option_list))
    if int(anchor_index) != labels[0]:
        # The old call site supplies a scheduled anchor.  For robust direct
        # use, rotate the schedule instead of silently making the answer a
        # deterministic first-label feature.
        shift = labels.index(int(anchor_index))
        labels = labels[shift:] + labels[:shift]
    option_order = list(option_list)
    random.Random(derive_seed(salt, "label-option-order", record_index)).shuffle(option_order)
    option_order.remove(anchor_option)
    anchor_position = int(record_index) % len(option_list)
    ordered: list[str | None] = [None] * len(option_list)
    ordered[anchor_position] = anchor_option
    iterator = iter(option_order)
    for pos in range(len(ordered)):
        if ordered[pos] is None:
            ordered[pos] = next(iterator)
    return {option: LABELS[label] for option, label in zip(ordered, labels)}


def label_permutation(
    options: Iterable[str],
    rng: random.Random,
    *,
    target_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    option_list = list(options)
    if not 2 <= len(option_list) <= len(LABELS):
        raise ValueError("label permutation requires 2..9 options")
    requested = dict(target_labels or {})
    if any(option not in option_list for option in requested):
        raise ValueError("target_labels contains an unknown option")
    if len(set(requested.values())) != len(requested) or any(label not in LABELS for label in requested.values()):
        raise ValueError("target labels must be distinct A-I labels")
    remaining_options = [option for option in option_list if option not in requested]
    remaining_labels = [label for label in LABELS if label not in requested.values()]
    rng.shuffle(remaining_options)
    rng.shuffle(remaining_labels)
    result = dict(requested)
    result.update(dict(zip(remaining_options, remaining_labels[: len(remaining_options)])))
    return result


def balanced_pair_mapping(
    options: Iterable[str],
    base_option: str,
    flip_option: str,
    pair_index: int,
    *,
    salt: int = 0,
) -> dict[str, str]:
    option_list = list(options)
    if base_option not in option_list or flip_option not in option_list or base_option == flip_option:
        raise ValueError("causal pair options must be distinct members of options")
    labels = _balanced_label_indices(pair_index, int(salt), len(option_list))
    base_position = int(pair_index) % max(1, len(option_list) - 1)
    result = {base_option: LABELS[labels[base_position]], flip_option: LABELS[labels[base_position + 1]]}
    remaining_options = [option for option in option_list if option not in result]
    random.Random(derive_seed(salt, "pair-label-order", pair_index)).shuffle(remaining_options)
    remaining_labels = [LABELS[label] for index, label in enumerate(labels) if index not in {base_position, base_position + 1}]
    result.update(dict(zip(remaining_options, remaining_labels)))
    return result
