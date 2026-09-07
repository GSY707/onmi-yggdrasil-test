from __future__ import annotations

"""Deterministic formal-data packaging and cross-history leakage audits for H1."""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import canonical
from .contract import (
    DATA_SEEDS,
    P0D_HISTORY,
    REGISTERED_NONFORMAL_DATA_SEEDS,
    REGISTERED_NONFORMAL_PACKAGE_IDENTITIES,
    SPLIT_COUNTS,
)
from .fresh_data import FreshBundle, SourceRecord, generate_bundle, validate_bundle
from .train import materialize_causal_records


@dataclass(frozen=True)
class FormalDataPackage:
    seed: int
    bundle: FreshBundle
    transformed_records: tuple[SourceRecord, ...]
    transformed_ledger: Mapping[str, Mapping[str, Any]]

    @property
    def records(self) -> tuple[SourceRecord, ...]:
        return tuple(self.bundle.records) + tuple(self.transformed_records)


def formal_counts() -> dict[str, dict[str, int]]:
    if any(count % 2 for count in SPLIT_COUNTS.values()):
        raise ValueError("H1 formal split totals must be family-balanced")
    return {
        family: {split: count // 2 for split, count in SPLIT_COUNTS.items()}
        for family in ("numeric", "relation")
    }


def _generate_package(
    seed: int,
    *,
    counts: Mapping[str, Mapping[str, int]],
    materialized_bases_per_family: int = 64,
    forbidden_semantic_fingerprints: Iterable[str] = (),
    forbidden_source_fingerprints: Iterable[str] = (),
) -> FormalDataPackage:
    bundle = generate_bundle(
        seed,
        counts,
        forbidden_semantic_fingerprints=forbidden_semantic_fingerprints,
        forbidden_source_fingerprints=forbidden_source_fingerprints,
    )
    validate_bundle(bundle)
    transformed, ledger = materialize_causal_records(
        bundle,
        per_family=materialized_bases_per_family,
    )
    if set(ledger) != {record.id for record in transformed}:
        raise ValueError("H1 transformed records and ledger disagree")
    return FormalDataPackage(seed, bundle, transformed, ledger)


def generate_formal_package(
    seed: int,
    *,
    materialized_bases_per_family: int = 64,
    forbidden_semantic_fingerprints: Iterable[str] = (),
    forbidden_source_fingerprints: Iterable[str] = (),
) -> FormalDataPackage:
    if seed not in DATA_SEEDS:
        raise ValueError(f"unregistered H1 formal data seed: {seed}")
    return _generate_package(
        seed,
        counts=formal_counts(),
        materialized_bases_per_family=materialized_bases_per_family,
        forbidden_semantic_fingerprints=forbidden_semantic_fingerprints,
        forbidden_source_fingerprints=forbidden_source_fingerprints,
    )


def _package_fingerprint_rows(
    package: FormalDataPackage,
) -> tuple[dict[str, str], dict[str, str]]:
    semantic = dict(package.bundle.semantic_fingerprints)
    semantic.update(
        {
            record_id: str(row["semantic_fingerprint"])
            for record_id, row in package.transformed_ledger.items()
        }
    )
    source = dict(package.bundle.source_fingerprints)
    source.update(
        {
            record_id: str(row["source_fingerprint"])
            for record_id, row in package.transformed_ledger.items()
        }
    )
    return semantic, source


def package_fingerprint_sets(
    package: FormalDataPackage,
) -> tuple[set[str], set[str]]:
    semantic, source = _package_fingerprint_rows(package)
    return set(semantic.values()), set(source.values())


def _generate_disjoint_packages(
    seeds: Iterable[int],
    *,
    counts: Mapping[str, Mapping[str, int]],
    materialized_bases_per_family: int,
    maximum_rejections_per_seed: int,
    forbidden_semantic_fingerprints: Iterable[str],
    forbidden_source_fingerprints: Iterable[str],
) -> tuple[FormalDataPackage, ...]:
    if maximum_rejections_per_seed < 1:
        raise ValueError("maximum_rejections_per_seed must be positive")
    accepted: list[FormalDataPackage] = []
    semantic_seen = set(forbidden_semantic_fingerprints)
    source_seen = set(forbidden_source_fingerprints)
    for seed in tuple(seeds):
        semantic_forbidden = set(semantic_seen)
        source_forbidden = set(source_seen)
        for _ in range(maximum_rejections_per_seed):
            package = _generate_package(
                seed,
                counts=counts,
                materialized_bases_per_family=materialized_bases_per_family,
                forbidden_semantic_fingerprints=semantic_forbidden,
                forbidden_source_fingerprints=source_forbidden,
            )
            semantic_rows, source_rows = _package_fingerprint_rows(package)
            semantic_collisions = {
                record_id
                for record_id, fingerprint in semantic_rows.items()
                if fingerprint in semantic_seen
            }
            source_collisions = {
                record_id
                for record_id, fingerprint in source_rows.items()
                if fingerprint in source_seen
            }
            if not semantic_collisions and not source_collisions:
                accepted.append(package)
                semantic_seen.update(semantic_rows.values())
                source_seen.update(source_rows.values())
                break

            # A derived collision rejects its base.  The fixed seed does not
            # move; only that record's existing deterministic attempt loop does.
            for record_id in semantic_collisions:
                base_id = str(
                    package.transformed_ledger.get(record_id, {}).get(
                        "base_id", record_id
                    )
                )
                semantic_forbidden.add(
                    str(package.bundle.semantic_fingerprints[base_id])
                )
            for record_id in source_collisions:
                base_id = str(
                    package.transformed_ledger.get(record_id, {}).get(
                        "base_id", record_id
                    )
                )
                source_forbidden.add(
                    str(package.bundle.source_fingerprints[base_id])
                )
        else:
            raise RuntimeError(
                f"could not generate globally disjoint H1 package for seed {seed}"
            )
    return tuple(accepted)


def generate_registered_nonformal_packages(
    *,
    materialized_bases_per_family: int = 64,
    maximum_rejections_per_seed: int = 128,
    forbidden_semantic_fingerprints: Iterable[str] = (),
    forbidden_source_fingerprints: Iterable[str] = (),
) -> dict[str, FormalDataPackage]:
    """Regenerate every consumed/current non-formal identity in evidence order.

    Prior failed directions remain part of the leakage boundary.  Returning a
    role-indexed registry makes omission of one consumed package visible to the
    formal workflow instead of silently reserving only the latest pair.
    """
    roles = tuple(REGISTERED_NONFORMAL_DATA_SEEDS)
    packages = _generate_disjoint_packages(
        tuple(REGISTERED_NONFORMAL_DATA_SEEDS[role] for role in roles),
        counts=formal_counts(),
        materialized_bases_per_family=materialized_bases_per_family,
        maximum_rejections_per_seed=maximum_rejections_per_seed,
        forbidden_semantic_fingerprints=forbidden_semantic_fingerprints,
        forbidden_source_fingerprints=forbidden_source_fingerprints,
    )
    registered = dict(zip(roles, packages, strict=True))
    mismatches = {
        role: {
            "expected": REGISTERED_NONFORMAL_PACKAGE_IDENTITIES.get(role),
            "actual": package_identity(package),
        }
        for role, package in registered.items()
        if package_identity(package)
        != REGISTERED_NONFORMAL_PACKAGE_IDENTITIES.get(role)
    }
    if mismatches:
        raise ValueError(f"registered H1 non-formal identity drift: {mismatches}")
    return registered


def generate_disjoint_formal_packages(
    *,
    materialized_bases_per_family: int = 64,
    maximum_rejections_per_seed: int = 128,
    forbidden_semantic_fingerprints: Iterable[str] = (),
    forbidden_source_fingerprints: Iterable[str] = (),
) -> tuple[FormalDataPackage, ...]:
    """Generate the registered seeds with deterministic global rejection.

    Simple relation graphs have a finite structural support, so independent
    large draws can legitimately repeat a supervised semantic example.  The
    seed remains fixed; only the already-existing per-record rejection loop is
    extended with fingerprints accepted by earlier registered seeds.  If a
    derived metamorphic example collides, its own base is rejected and that
    record alone advances to its next deterministic attempt.
    """
    return _generate_disjoint_packages(
        DATA_SEEDS,
        counts=formal_counts(),
        materialized_bases_per_family=materialized_bases_per_family,
        maximum_rejections_per_seed=maximum_rejections_per_seed,
        forbidden_semantic_fingerprints=forbidden_semantic_fingerprints,
        forbidden_source_fingerprints=forbidden_source_fingerprints,
    )


def tagged(value: Any) -> Any:
    """Convert a dataclass tree to canonical JSON while preserving concrete types."""
    if is_dataclass(value) and not isinstance(value, type):
        payload = {key: tagged(item) for key, item in asdict(value).items()}
        return {"type": type(value).__name__, "value": payload}
    if isinstance(value, Mapping):
        return {str(key): tagged(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tagged(item) for item in value]
    return value


def package_payload(package: FormalDataPackage) -> dict[str, Any]:
    bundle = package.bundle
    by_id = {record.id: record for record in bundle.records}
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.formal-data-package.v1",
        "seed": package.seed,
        "splits": {name: list(ids) for name, ids in bundle.splits.items()},
        "sources": {record_id: tagged(by_id[record_id]) for record_id in sorted(by_id)},
        "targets": {
            record_id: tagged(bundle.targets[record_id])
            for record_id in sorted(bundle.targets)
        },
        "components": {
            record_id: tagged(bundle.components[record_id])
            for record_id in sorted(bundle.components)
        },
        "fingerprints": {
            "semantic": dict(sorted(bundle.semantic_fingerprints.items())),
            "source": dict(sorted(bundle.source_fingerprints.items())),
        },
        "transformed_sources": {
            record.id: tagged(record) for record in sorted(package.transformed_records, key=lambda row: row.id)
        },
        "transformed_ledger": {
            record_id: tagged(package.transformed_ledger[record_id])
            for record_id in sorted(package.transformed_ledger)
        },
    }


def package_identity(package: FormalDataPackage) -> str:
    return hashlib.sha256(canonical(package_payload(package))).hexdigest().upper()


def multi_package_overlap_audit(
    packages: Iterable[FormalDataPackage],
) -> dict[str, Any]:
    rows = tuple(packages)
    if tuple(package.seed for package in rows) != DATA_SEEDS:
        return {
            "seeds": [package.seed for package in rows],
            "expected_seeds": list(DATA_SEEDS),
            "passed": False,
            "reason": "formal package seed order mismatch",
        }

    def collisions(kind: str) -> list[dict[str, Any]]:
        owners: dict[str, list[dict[str, Any]]] = {}
        for package in rows:
            mapping = (
                package.bundle.semantic_fingerprints
                if kind == "semantic"
                else package.bundle.source_fingerprints
            )
            split_by_id = {
                record_id: split
                for split, record_ids in package.bundle.splits.items()
                for record_id in record_ids
            }
            for record_id, fingerprint in mapping.items():
                owners.setdefault(fingerprint, []).append(
                    {
                        "seed": package.seed,
                        "split": split_by_id[record_id],
                        "record_id": record_id,
                    }
                )
            ledger_key = (
                "semantic_fingerprint" if kind == "semantic" else "source_fingerprint"
            )
            for record_id, item in package.transformed_ledger.items():
                fingerprint = str(item[ledger_key])
                owners.setdefault(fingerprint, []).append(
                    {
                        "seed": package.seed,
                        "split": "metamorphic",
                        "record_id": record_id,
                    }
                )
        return [
            {"fingerprint": fingerprint, "owners": values}
            for fingerprint, values in sorted(owners.items())
            if len({int(owner["seed"]) for owner in values}) > 1
        ]

    semantic = collisions("semantic")
    source = collisions("source")
    return {
        "seeds": [package.seed for package in rows],
        "semantic_collisions": semantic,
        "source_collisions": source,
        "passed": not semantic and not source,
    }


def _nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_strings(item)


def historical_fingerprint_registry(repo_root: Path) -> dict[str, Any]:
    """Load only explicitly registered historical record fingerprints."""
    semantic: set[str] = set()
    source: set[str] = set()
    p0d_dataset = repo_root / P0D_HISTORY["root"] / "dataset"
    p0d_files = sorted(p0d_dataset.glob("*.jsonl"))
    p0d_rows = 0
    for path in p0d_files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                semantic_value = row.get("semantic_fingerprint")
                source_value = row.get("surface_fingerprint")
                if not isinstance(semantic_value, str) or not isinstance(source_value, str):
                    raise ValueError(f"historical P0-D fingerprint missing in {path.name}")
                semantic.add(semantic_value)
                source.add(source_value)
                p0d_rows += 1

    nr1_path = (
        repo_root
        / "artifacts/v2-r1r/p1-nr1-qualification-20260817-1/case-manifest.json"
    )
    nr1 = json.loads(nr1_path.read_text(encoding="utf-8"))
    nr1_values: set[str] = set()
    for family in ("numeric", "relation"):
        nr1_values.update(_nested_strings(nr1[family]["fingerprints"]))
    semantic.update(nr1_values)
    return {
        "semantic": semantic,
        "source": source,
        "counts": {
            "p0d_rows": p0d_rows,
            "p0d_semantic": len(semantic - nr1_values),
            "p0d_source": len(source),
            "nr1": len(nr1_values),
        },
        "sources": {
            "p0d_dataset": P0D_HISTORY["root"].joinpath("dataset").as_posix(),
            "nr1_case_manifest": nr1_path.relative_to(repo_root).as_posix(),
            "p0m_note": "P0-M v5 reuses the registered P0-D v17 dataset",
        },
    }


def historical_overlap_audit(
    packages: Iterable[FormalDataPackage],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_history = set(registry["semantic"])
    source_history = set(registry["source"])
    semantic_hits: list[dict[str, Any]] = []
    source_hits: list[dict[str, Any]] = []
    for package in packages:
        for record_id, fingerprint in package.bundle.semantic_fingerprints.items():
            if fingerprint in semantic_history:
                semantic_hits.append(
                    {"seed": package.seed, "record_id": record_id, "fingerprint": fingerprint}
                )
        for record_id, fingerprint in package.bundle.source_fingerprints.items():
            if fingerprint in source_history:
                source_hits.append(
                    {"seed": package.seed, "record_id": record_id, "fingerprint": fingerprint}
                )
        for record_id, item in package.transformed_ledger.items():
            semantic_fingerprint = str(item["semantic_fingerprint"])
            source_fingerprint = str(item["source_fingerprint"])
            if semantic_fingerprint in semantic_history:
                semantic_hits.append(
                    {
                        "seed": package.seed,
                        "record_id": record_id,
                        "split": "metamorphic",
                        "fingerprint": semantic_fingerprint,
                    }
                )
            if source_fingerprint in source_history:
                source_hits.append(
                    {
                        "seed": package.seed,
                        "record_id": record_id,
                        "split": "metamorphic",
                        "fingerprint": source_fingerprint,
                    }
                )
    return {
        "registry_counts": dict(registry.get("counts", {})),
        "registry_sources": dict(registry.get("sources", {})),
        "semantic_hits": semantic_hits,
        "source_hits": source_hits,
        "passed": not semantic_hits and not source_hits,
    }


__all__ = [
    "FormalDataPackage",
    "formal_counts",
    "generate_disjoint_formal_packages",
    "generate_formal_package",
    "generate_registered_nonformal_packages",
    "historical_fingerprint_registry",
    "historical_overlap_audit",
    "multi_package_overlap_audit",
    "package_identity",
    "package_fingerprint_sets",
    "package_payload",
    "tagged",
]
