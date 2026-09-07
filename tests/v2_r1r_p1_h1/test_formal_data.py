from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from yggdrasil_v2.r1_revalidation.h1 import formal_data
from yggdrasil_v2.r1_revalidation.h1.formal_data import (
    FormalDataPackage,
    generate_disjoint_formal_packages,
    generate_registered_nonformal_packages,
    historical_overlap_audit,
    multi_package_overlap_audit,
    package_fingerprint_sets,
    tagged,
)
from yggdrasil_v2.r1_revalidation.h1.fresh_data import generate_bundle


def _package(seed: int) -> FormalDataPackage:
    counts = {
        family: {split: 1 for split in ("train", "validation", "supported", "heldout", "causal")}
        for family in ("numeric", "relation")
    }
    bundle = generate_bundle(seed, counts)
    return FormalDataPackage(seed, bundle, (), {})


def test_tagged_payload_preserves_dataclass_type() -> None:
    package = _package(1)
    value = tagged(package.bundle.records[0])
    assert value["type"] == "SourceRecord"
    assert set(value["value"]) == {"id", "source_text", "source_atoms"}


def test_multi_package_overlap_is_fail_closed(monkeypatch) -> None:
    packages = (_package(1), _package(2), _package(3))
    monkeypatch.setattr(formal_data, "DATA_SEEDS", (1, 2, 3))
    assert multi_package_overlap_audit(packages)["passed"] is True

    duplicate = replace(packages[1], bundle=packages[0].bundle)
    audit = multi_package_overlap_audit((packages[0], duplicate, packages[2]))
    assert audit["passed"] is False
    assert audit["semantic_collisions"]
    assert audit["source_collisions"]


def test_historical_overlap_reports_registered_hits(monkeypatch) -> None:
    package = _package(4)
    first_id = package.bundle.records[0].id
    registry = {
        "semantic": {package.bundle.semantic_fingerprints[first_id]},
        "source": set(),
        "counts": {"fixture": 1},
        "sources": {"fixture": "unit"},
    }
    audit = historical_overlap_audit((package,), registry)
    assert audit["passed"] is False
    assert audit["semantic_hits"][0]["record_id"] == first_id
    assert audit["source_hits"] == []


def test_registered_nonformal_and_formal_packages_reject_prior_domains(
    monkeypatch,
) -> None:
    counts = {
        family: {
            split: 1
            for split in ("train", "validation", "supported", "heldout", "causal")
        }
        for family in ("numeric", "relation")
    }
    prior = _package(20)
    prior_semantic, prior_source = package_fingerprint_sets(prior)
    monkeypatch.setattr(formal_data, "formal_counts", lambda: counts)
    role_seeds = {
        "prior_screen": 21,
        "prior_calibration": 22,
        "screen": 23,
        "calibration": 24,
    }
    monkeypatch.setattr(formal_data, "REGISTERED_NONFORMAL_DATA_SEEDS", role_seeds)
    expected_packages = formal_data._generate_disjoint_packages(
        tuple(role_seeds.values()),
        counts=counts,
        materialized_bases_per_family=1,
        maximum_rejections_per_seed=128,
        forbidden_semantic_fingerprints=prior_semantic,
        forbidden_source_fingerprints=prior_source,
    )
    monkeypatch.setattr(
        formal_data,
        "REGISTERED_NONFORMAL_PACKAGE_IDENTITIES",
        {
            role: formal_data.package_identity(package)
            for role, package in zip(role_seeds, expected_packages, strict=True)
        },
    )
    reservations = generate_registered_nonformal_packages(
        materialized_bases_per_family=1,
        forbidden_semantic_fingerprints=prior_semantic,
        forbidden_source_fingerprints=prior_source,
    )
    assert tuple(reservations) == (
        "prior_screen",
        "prior_calibration",
        "screen",
        "calibration",
    )
    reserved_semantic: set[str] = set()
    reserved_source: set[str] = set()
    for package in reservations.values():
        semantic, source = package_fingerprint_sets(package)
        assert not (prior_semantic & semantic)
        assert not (prior_source & source)
        assert not (reserved_semantic & semantic)
        assert not (reserved_source & source)
        reserved_semantic.update(semantic)
        reserved_source.update(source)

    monkeypatch.setattr(formal_data, "DATA_SEEDS", (25, 26, 27))
    formal = generate_disjoint_formal_packages(
        materialized_bases_per_family=1,
        forbidden_semantic_fingerprints=(prior_semantic | reserved_semantic),
        forbidden_source_fingerprints=(prior_source | reserved_source),
    )
    forbidden_semantic = prior_semantic | reserved_semantic
    forbidden_source = prior_source | reserved_source
    seen_semantic: set[str] = set()
    seen_source: set[str] = set()
    for package in formal:
        semantic, source = package_fingerprint_sets(package)
        assert not (semantic & forbidden_semantic)
        assert not (source & forbidden_source)
        assert not (semantic & seen_semantic)
        assert not (source & seen_source)
        seen_semantic.update(semantic)
        seen_source.update(source)
