from __future__ import annotations

"""C0R data identity, layered over the immutable v17 production substrate.

The substrate is deliberately used without source edits.  The only semantic
change is that ordinary ERE relation-query records alternate their base truth
value by relation ordinal; causal pairs retain the substrate's paired flip.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from yggdrasil_v2.r1_revalidation.production import generator as _substrate


GENERATOR_VERSION = "v2-a-c0r-relation-balanced-v1"
SCHEMA_VERSION = "yggdrasil.v2-a.c0r.record.v1"
MANIFEST_SCHEMA = "yggdrasil.v2-a.c0r.dataset-manifest.v1"
FORMAL_SEED = 2026082401

SPLITS = _substrate.SPLITS
SEED_DERIVATION_VERSION = _substrate.SEED_DERIVATION_VERSION


@dataclass(frozen=True)
class DatasetSpec:
    profile: str
    root_seed: int
    train_count: int
    heldout_count: int

    def count(self, split: str) -> int:
        return self.train_count if split == "train" else self.heldout_count


FORMAL_SPEC = DatasetSpec("formal", FORMAL_SEED, 4096, 1536)


_PINNED_BUILD_ERE_PAIR = _substrate.build_ere_pair


def _balanced_relation_builder(seed: int, split: str, index: int):
    pair = _PINNED_BUILD_ERE_PAIR(seed, split, index)
    query = pair[0].get("query", {})
    if (
        query.get("kind") == "relation"
        and index % 12 == 0
        and (index // 12) % 2 == 1
    ):
        base, alternate, metadata = pair
        return alternate, base, metadata
    return pair


def _write_manifest_identity(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = MANIFEST_SCHEMA
    manifest["generator_version"] = GENERATOR_VERSION
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate(
    root: Path,
    *,
    spec: DatasetSpec = FORMAL_SPEC,
    design_doc: Path | None = None,
    tokenizer: Any | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Generate a deterministic C0R package using the pinned substrate."""

    if not isinstance(spec, DatasetSpec):
        raise TypeError("spec must be a C0R DatasetSpec")
    if design_doc is None:
        design_doc = Path("docs/v2-a-closure-c0r-data-trace-qualification.md")

    old_constants = (_substrate.GENERATOR_VERSION, _substrate.SCHEMA_VERSION)
    old_builder = _substrate.build_ere_pair
    _substrate.GENERATOR_VERSION = GENERATOR_VERSION
    _substrate.SCHEMA_VERSION = SCHEMA_VERSION
    _substrate.build_ere_pair = _balanced_relation_builder
    try:
        substrate_spec = _substrate.DatasetSpec(
            spec.profile, spec.root_seed, spec.train_count, spec.heldout_count
        )
        _substrate.generate_production_dataset(
            root,
            spec=substrate_spec,
            design_doc=design_doc,
            tokenizer=tokenizer,
            progress=progress,
        )
        return _write_manifest_identity(root)
    finally:
        _substrate.build_ere_pair = old_builder
        _substrate.GENERATOR_VERSION, _substrate.SCHEMA_VERSION = old_constants


def load(root: Path) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("not a C0R dataset manifest")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("not a C0R generator identity")
    return _substrate.load_production_dataset(root)


def public_forward_view(record: dict[str, Any]) -> dict[str, str]:
    """Return the exact C0R public model input surface."""

    if not isinstance(record, dict) or not isinstance(record.get("source_text"), str):
        raise ValueError("record source_text must be a string")
    return {"source_text": record["source_text"]}


__all__ = [
    "DatasetSpec",
    "FORMAL_SPEC",
    "FORMAL_SEED",
    "GENERATOR_VERSION",
    "MANIFEST_SCHEMA",
    "SCHEMA_VERSION",
    "SPLITS",
    "generate",
    "load",
    "public_forward_view",
]
