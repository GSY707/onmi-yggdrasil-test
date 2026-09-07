from __future__ import annotations

"""Formal H1 data/cache/smoke/training actions.

The public CLI owns single-use transport and fixed-root creation.  This module
owns the expensive work performed *inside* those roots.  It has no standalone
entry point, retry path, resume mode, or tunable argument surface.
"""

from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from .artifacts import (
    canonical,
    git_identity,
    sha256,
    source_identity,
    verify_seal,
    write_json,
)
from .audit import architecture_audit
from .cache import TokenCache, audit_token_cache, build_token_cache
from .contract import (
    CONTRACT_VERSION,
    DATA_SEEDS,
    MODEL,
    MODEL_SEEDS,
    REGISTERED_NONFORMAL_DATA_SEEDS,
    ROOTS,
    SMOKE,
    SMOKE_DATA_SEED,
    SMOKE_MODEL_SEED,
    SPLIT_COUNTS,
    THRESHOLDS,
    TRAINING,
    TRANSPORT_ROOT,
    contract_manifest,
)
from .formal_data import (
    FormalDataPackage,
    generate_disjoint_formal_packages,
    generate_registered_nonformal_packages,
    historical_fingerprint_registry,
    historical_overlap_audit,
    multi_package_overlap_audit,
    package_identity,
    package_fingerprint_sets,
    package_payload,
)
from .fresh_data import (
    FreshBundle,
    SourceRecord,
    cross_split_overlap_audit,
    forbidden_source_scan,
    generate_bundle,
    source_fingerprint,
    validate_bundle,
)
from .metrics import primary_comparison
from .model import H1Config, H1LatentReasoner
from .qualification import aggregate_h1_qualification
from .train import (
    INTERVENTIONS,
    PreparedSplit,
    build_balanced_schedule,
    build_pair,
    evaluate_model,
    metamorphic_consistency,
    metamorphic_route_consistency,
    prepare_materialized_split,
    prepare_split,
    schedule_audit,
    strip_and_save,
    train_fixed_budget,
)


PASS_PREFLIGHT = "PASS_P1_H1_PREFLIGHT"
FAIL_PREFLIGHT = "FAIL_P1_H1_PREFLIGHT"
PASS_DEVELOPMENT = "PASS_P1_H1_MIXED_CORE_DEVELOPMENT"
FAIL_DEVELOPMENT = "FAIL_P1_H1_MIXED_CORE_DEVELOPMENT"
_ROUTE_INTERVENTIONS = (
    "flip_route",
    "force_route0",
    "force_route1",
    "swap_experts",
)
_CONDITIONAL_WRITE_INTERVENTION = "disable_routed_projection"


def formal_model_config(*, arm: str = "shared") -> H1Config:
    return H1Config(
        arm=arm,
        source_width=MODEL.source_width,
        latent_width=MODEL.latent_width,
        K=MODEL.latent_slots,
        T=MODEL.recurrent_steps,
        recurrent_layers=MODEL.recurrent_layers,
        num_heads=MODEL.attention_heads,
        active_ffn_multiplier=MODEL.active_ffn_multiplier,
        shared_ffn_inner_width=MODEL.shared_ffn_inner_width,
        routed_feature_width=MODEL.routed_feature_width,
        trace_classes=MODEL.trace_classes,
        answer_classes=MODEL.answer_classes,
        route_classes=MODEL.route_classes,
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_compact_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def _claims(passed: bool) -> dict[str, bool]:
    return {
        "p1_f1_design_authorized": passed,
        "p1_completed": False,
        "p2_eligible": False,
        "p2_started": False,
    }


def _tests_audit(repo_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/v2_r1r_p1_h1",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def hardware_audit(repo_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(repo_root)
    minimum_free = int(SMOKE.minimum_free_disk_gib) * 1024**3
    cuda = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda else 0
    if cuda and device_count:
        properties = torch.cuda.get_device_properties(0)
        device = {
            "index": 0,
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    else:
        device = None
    checks = {
        "cuda_available": cuda,
        "cuda_device_present": device_count >= 1,
        "minimum_vram": device is not None and int(device["total_memory"]) >= 7 * 1024**3,
        "minimum_free_disk": int(usage.free) >= minimum_free,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "device_count": device_count,
        "device": device,
        "disk": {
            "total": int(usage.total),
            "used": int(usage.used),
            "free": int(usage.free),
            "minimum_free": minimum_free,
        },
    }


def generate_registered_inputs(
    repo_root: Path,
) -> tuple[
    tuple[FormalDataPackage, ...],
    FormalDataPackage,
    Mapping[str, FormalDataPackage],
]:
    """Generate all disjoint identity domains in their registered order.

    Historical P0-D/NR1 evidence is reserved first, followed by every consumed
    H1 non-formal package, the current screen/calibration pair, the three unseen
    formal seeds, and finally smoke.  Only formal and smoke records enter the
    formal cache.
    """
    registry = historical_fingerprint_registry(repo_root)
    reservations = generate_registered_nonformal_packages(
        forbidden_semantic_fingerprints=registry["semantic"],
        forbidden_source_fingerprints=registry["source"],
    )
    semantic_forbidden = set(registry["semantic"])
    source_forbidden = set(registry["source"])
    for package in reservations.values():
        semantic, source = package_fingerprint_sets(package)
        semantic_forbidden.update(semantic)
        source_forbidden.update(source)
    packages = generate_disjoint_formal_packages(
        forbidden_semantic_fingerprints=semantic_forbidden,
        forbidden_source_fingerprints=source_forbidden,
    )
    for package in packages:
        semantic, source = package_fingerprint_sets(package)
        semantic_forbidden.update(semantic)
        source_forbidden.update(source)
    smoke_counts = {
        family: {"train": int(SMOKE.train_per_family)}
        for family in ("numeric", "relation")
    }
    smoke_bundle = generate_bundle(
        SMOKE_DATA_SEED,
        smoke_counts,
        forbidden_semantic_fingerprints=semantic_forbidden,
        forbidden_source_fingerprints=source_forbidden,
    )
    validate_bundle(smoke_bundle)
    smoke = FormalDataPackage(SMOKE_DATA_SEED, smoke_bundle, (), {})
    return packages, smoke, reservations


def combined_records(
    packages: Sequence[FormalDataPackage],
    smoke: FormalDataPackage,
) -> tuple[SourceRecord, ...]:
    return tuple(
        record
        for package in (*packages, smoke)
        for record in package.records
    )


def _record_order_identity(records: Sequence[SourceRecord]) -> str:
    payload = [
        {"id": record.id, "source_fingerprint": source_fingerprint(record)}
        for record in records
    ]
    return hashlib.sha256(canonical(payload)).hexdigest().upper()


def _source_field_hits(records: Iterable[SourceRecord]) -> list[dict[str, Any]]:
    expected = {"id", "source_text", "source_atoms"}
    hits = []
    for record in records:
        actual = set(record.to_dict())
        if actual != expected:
            hits.append(
                {
                    "record_id": record.id,
                    "unexpected": sorted(actual - expected),
                    "missing": sorted(expected - actual),
                }
            )
    return hits


def _registered_scope_overlap_audit(
    packages: Sequence[FormalDataPackage],
    smoke: FormalDataPackage,
    reservations: Mapping[str, FormalDataPackage],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    scopes: dict[str, tuple[set[str], set[str]]] = {
        "history": (set(registry["semantic"]), set(registry["source"])),
    }
    for role in REGISTERED_NONFORMAL_DATA_SEEDS:
        package = reservations.get(role)
        scopes[role] = (
            package_fingerprint_sets(package)
            if isinstance(package, FormalDataPackage)
            else (set(), set())
        )
    formal_semantic: set[str] = set()
    formal_source: set[str] = set()
    for package in packages:
        semantic, source = package_fingerprint_sets(package)
        formal_semantic.update(semantic)
        formal_source.update(source)
    scopes["formal"] = (formal_semantic, formal_source)
    scopes["smoke"] = package_fingerprint_sets(smoke)

    names = ("history", *REGISTERED_NONFORMAL_DATA_SEEDS, "formal", "smoke")

    def collisions(index: int) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                overlap = sorted(scopes[left][index] & scopes[right][index])
                if overlap:
                    hits.append(
                        {"left": left, "right": right, "fingerprints": overlap}
                    )
        return hits

    semantic = collisions(0)
    source = collisions(1)
    return {
        "scope_order": list(names),
        "scope_counts": {
            name: {"semantic": len(value[0]), "source": len(value[1])}
            for name, value in scopes.items()
        },
        "semantic_collisions": semantic,
        "source_collisions": source,
        "passed": not semantic and not source,
    }


def data_leakage_audit(
    repo_root: Path,
    packages: Sequence[FormalDataPackage],
    smoke: FormalDataPackage,
    reservations: Mapping[str, FormalDataPackage],
) -> dict[str, Any]:
    if tuple(package.seed for package in packages) != DATA_SEEDS:
        return {
            "passed": False,
            "reason": "formal seed order mismatch",
            "seeds": [package.seed for package in packages],
        }
    per_seed: dict[str, Any] = {}
    for package in packages:
        validate_bundle(package.bundle)
        counts = {
            split: len(package.bundle.splits[split])
            for split in SPLIT_COUNTS
        }
        overlap = cross_split_overlap_audit(package.bundle)
        source_scan = forbidden_source_scan(package.records)
        field_hits = _source_field_hits(package.records)
        checks = {
            "split_counts": counts == SPLIT_COUNTS,
            "cross_split_overlap": overlap["passed"] is True,
            "forbidden_source_scan": not source_scan,
            "model_view_fields": not field_hits,
        }
        per_seed[str(package.seed)] = {
            "split_counts": counts,
            "cross_split": overlap,
            "forbidden_source_scan": source_scan,
            "forbidden_field_hits": field_hits,
            "checks": checks,
            "passed": all(checks.values()),
        }

    cross_package = multi_package_overlap_audit(packages)
    registry = historical_fingerprint_registry(repo_root)
    historical = historical_overlap_audit(packages, registry)
    reservation_expected = dict(REGISTERED_NONFORMAL_DATA_SEEDS)
    reservation_rows: dict[str, Any] = {}
    for role, expected_seed in reservation_expected.items():
        package = reservations.get(role)
        if not isinstance(package, FormalDataPackage):
            reservation_rows[role] = {
                "expected_seed": expected_seed,
                "passed": False,
                "reason": "missing_registered_reservation",
            }
            continue
        validate_bundle(package.bundle)
        counts = {
            split: len(package.bundle.splits[split]) for split in SPLIT_COUNTS
        }
        overlap = cross_split_overlap_audit(package.bundle)
        source_scan = forbidden_source_scan(package.records)
        field_hits = _source_field_hits(package.records)
        reservation_checks = {
            "seed": package.seed == expected_seed,
            "split_counts": counts == SPLIT_COUNTS,
            "cross_split_overlap": overlap["passed"] is True,
            "forbidden_source_scan": not source_scan,
            "model_view_fields": not field_hits,
        }
        reservation_rows[role] = {
            "seed": package.seed,
            "expected_seed": expected_seed,
            "identity": package_identity(package),
            "split_counts": counts,
            "cross_split": overlap,
            "forbidden_source_scan": source_scan,
            "forbidden_field_hits": field_hits,
            "checks": reservation_checks,
            "passed": all(reservation_checks.values()),
        }
    registered_scopes = _registered_scope_overlap_audit(
        packages, smoke, reservations, registry
    )

    formal_semantic: set[str] = set()
    formal_source: set[str] = set()
    for package in packages:
        formal_semantic.update(package.bundle.semantic_fingerprints.values())
        formal_source.update(package.bundle.source_fingerprints.values())
        formal_semantic.update(
            str(row["semantic_fingerprint"])
            for row in package.transformed_ledger.values()
        )
        formal_source.update(
            str(row["source_fingerprint"])
            for row in package.transformed_ledger.values()
        )
    smoke_semantic = set(smoke.bundle.semantic_fingerprints.values())
    smoke_source = set(smoke.bundle.source_fingerprints.values())
    smoke_scan = forbidden_source_scan(smoke.records)
    smoke_fields = _source_field_hits(smoke.records)
    smoke_disjoint = {
        "formal_semantic_hits": sorted(smoke_semantic & formal_semantic),
        "formal_source_hits": sorted(smoke_source & formal_source),
        "historical_semantic_hits": sorted(smoke_semantic & set(registry["semantic"])),
        "historical_source_hits": sorted(smoke_source & set(registry["source"])),
        "forbidden_source_scan": smoke_scan,
        "forbidden_field_hits": smoke_fields,
    }
    smoke_disjoint["passed"] = not any(smoke_disjoint.values())

    semantic_overlap = [
        *cross_package.get("semantic_collisions", []),
        *historical.get("semantic_hits", []),
        *smoke_disjoint["formal_semantic_hits"],
        *smoke_disjoint["historical_semantic_hits"],
    ]
    source_overlap = [
        *cross_package.get("source_collisions", []),
        *historical.get("source_hits", []),
        *smoke_disjoint["formal_source_hits"],
        *smoke_disjoint["historical_source_hits"],
    ]
    forbidden_source = {
        key: value
        for key, row in per_seed.items()
        for key, value in row["forbidden_source_scan"].items()
    }
    forbidden_source.update(smoke_scan)
    for row in reservation_rows.values():
        forbidden_source.update(row.get("forbidden_source_scan", {}))
    forbidden_fields = [
        row
        for value in per_seed.values()
        for row in value["forbidden_field_hits"]
    ] + smoke_fields
    forbidden_fields.extend(
        field
        for row in reservation_rows.values()
        for field in row.get("forbidden_field_hits", [])
    )
    checks = {
        "per_seed": all(row["passed"] for row in per_seed.values()),
        "cross_package": cross_package.get("passed") is True,
        "historical": historical.get("passed") is True,
        "reservations": all(
            row.get("passed") is True for row in reservation_rows.values()
        ),
        "registered_scopes": registered_scopes["passed"] is True,
        "smoke_disjoint": smoke_disjoint["passed"] is True,
        "semantic_overlap_empty": not semantic_overlap,
        "source_overlap_empty": not source_overlap,
        "forbidden_source_empty": not forbidden_source,
        "forbidden_fields_empty": not forbidden_fields,
    }
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.data-audit.v1",
        "per_seed": per_seed,
        "cross_package": cross_package,
        "historical": historical,
        "historical_registry": {
            "counts": registry["counts"],
            "sources": registry["sources"],
        },
        "reserved_nonformal": reservation_rows,
        "registered_scope_overlap": registered_scopes,
        "smoke_disjoint": smoke_disjoint,
        "semantic_overlap": semantic_overlap,
        "source_overlap": source_overlap,
        "forbidden_source_scan": forbidden_source,
        "forbidden_field_hits": forbidden_fields,
        "checks": checks,
        "passed": all(checks.values()),
    }


def persist_data_packages(
    root: Path,
    packages: Sequence[FormalDataPackage],
    smoke: FormalDataPackage,
    reservations: Mapping[str, FormalDataPackage],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for package in (*packages, smoke):
        kind = "smoke" if package.seed == SMOKE_DATA_SEED else "formal"
        relative = Path("data") / f"{kind}-seed-{package.seed}.json"
        path = root / relative
        _write_compact_json(path, package_payload(package))
        rows[str(package.seed)] = {
            "kind": kind,
            "path": relative.as_posix(),
            "identity": package_identity(package),
            "file_sha256": sha256(path),
            "record_count": len(package.records),
        }
    records = combined_records(packages, smoke)
    reserved_nonformal: dict[str, Any] = {}
    for role, package in reservations.items():
        semantic, source = package_fingerprint_sets(package)
        reserved_nonformal[role] = {
            "seed": package.seed,
            "identity": package_identity(package),
            "record_count": len(package.records),
            "semantic_fingerprint_count": len(semantic),
            "source_fingerprint_count": len(source),
            "included_in_formal_cache": False,
        }
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.data-manifest.v2",
        "packages": rows,
        "formal_seed_order": list(DATA_SEEDS),
        "smoke_data_seed": SMOKE_DATA_SEED,
        "reserved_nonformal": reserved_nonformal,
        "record_count": len(records),
        "record_order_identity": _record_order_identity(records),
        "cache_root": "token-cache",
    }


def _preflight_result(
    *,
    identity: str,
    initial_entry: Mapping[str, Any],
    checks: Mapping[str, bool],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    passed = bool(checks) and all(checks.values())
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.preflight-result.v1",
        "contract_version": CONTRACT_VERSION,
        "status": PASS_PREFLIGHT if passed else FAIL_PREFLIGHT,
        "passed": passed,
        "source_identity": identity,
        "formal_entry_audit": dict(initial_entry),
        "checks": dict(checks),
        "details": dict(details),
        "claims": _claims(False),
    }


def preflight_action(
    repo_root: Path,
    initial_entry: Mapping[str, Any],
    root: Path,
    identity: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Run fail-fast audits, then generate and seal the only formal cache."""
    from . import runner

    files = runner.source_files(repo_root)
    allowed = (ROOTS["preflight"], TRANSPORT_ROOT)
    runtime_before = runner.formal_runtime_audit(repo_root, allowed)
    contract = runner.contract_hash_audit(repo_root)
    history = runner.history_replay(repo_root)
    tests = _tests_audit(repo_root)
    cli = runner._cli_audit(repo_root)
    hardware = hardware_audit(repo_root)
    architecture = architecture_audit(formal_model_config(), source_tokens=11)
    source_stable = identity == source_identity(repo_root, files)
    preliminary = {
        "initial_entry": initial_entry.get("passed") is True,
        "thresholds_frozen": THRESHOLDS.frozen(),
        "contract_hashes": contract["passed"] is True,
        "history_replay": history["passed"] is True,
        "tests": tests["passed"] is True,
        "cli_direct_switch": cli["passed"] is True,
        "hardware": hardware["passed"] is True,
        "architecture": architecture["passed"] is True,
        "runtime_before": runtime_before["passed"] is True,
        "git_identity_captured": environment.get("git_identity_start", {}).get("passed") is True,
        "source_identity_stable": source_stable,
    }
    details: dict[str, Any] = {
        "contract_hash_audit": contract,
        "history_replay": history,
        "tests": tests,
        "cli_audit": cli,
        "hardware_audit": hardware,
        "architecture_audit": architecture,
        "runtime_before": runtime_before,
    }
    write_json(root / "contract-manifest.json", contract_manifest())
    for name, value in details.items():
        write_json(root / f"{name.replace('_', '-')}.json", value)
    if not all(preliminary.values()):
        return _preflight_result(
            identity=identity,
            initial_entry=initial_entry,
            checks=preliminary,
            details={"stopped_before_data_or_cache": True},
        )

    packages, smoke, reservations = generate_registered_inputs(repo_root)
    data_audit = data_leakage_audit(repo_root, packages, smoke, reservations)
    data_manifest = persist_data_packages(root, packages, smoke, reservations)
    write_json(root / "data-audit.json", data_audit)
    write_json(root / "data-manifest.json", data_manifest)
    data_check = data_audit["passed"] is True
    checks = {**preliminary, "data_leakage": data_check}
    if not data_check:
        return _preflight_result(
            identity=identity,
            initial_entry=initial_entry,
            checks=checks,
            details={"stopped_before_cache": True, "data_manifest": data_manifest},
        )

    records = combined_records(packages, smoke)
    cache_root = root / data_manifest["cache_root"]
    cache_manifest = build_token_cache(
        records,
        cache_root,
        device="cuda",
        batch_size=int(SMOKE.cache_batch_size),
    )
    cache_audit = audit_token_cache(records, cache_root, verify_all_finite=True)
    write_json(root / "cache-build.json", cache_manifest)
    write_json(root / "cache-audit.json", cache_audit)
    runtime_after = runner.formal_runtime_audit(repo_root, allowed)
    write_json(root / "runtime-after.json", runtime_after)
    checks.update(
        {
            "cache": cache_audit["passed"] is True,
            "runtime_after": runtime_after["passed"] is True,
            "source_identity_after_cache": identity == source_identity(repo_root, files),
        }
    )
    return _preflight_result(
        identity=identity,
        initial_entry=initial_entry,
        checks=checks,
        details={
            "data_manifest": data_manifest,
            "cache_manifest": cache_manifest,
            "cache_audit": cache_audit,
        },
    )


def _compact_evaluation(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "split",
        "intervention",
        "ids",
        "predictions",
        "targets",
        "families",
        "answer",
        "trace",
        "route",
        "route_predictions",
        "state_hashes",
    )
    return {key: value[key] for key in keys if key in value}


def _drop(normal: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, float]:
    return {
        "answer": float(
            normal["answer"]["macro_accuracy"]
            - trial["answer"]["macro_accuracy"]
        ),
        "trace": float(
            normal["trace"]["cell_accuracy"]
            - trial["trace"]["cell_accuracy"]
        ),
    }


def _shared_route_noop(
    normal: Mapping[str, Any],
    interventions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for name in _ROUTE_INTERVENTIONS:
        trial = interventions[name]
        row = {
            "predictions": trial.get("predictions") == normal.get("predictions"),
            "trace_predictions": trial.get("trace_predictions")
            == normal.get("trace_predictions"),
            "logits": trial.get("state_hashes", {}).get("logits")
            == normal.get("state_hashes", {}).get("logits"),
            "final_state": trial.get("state_hashes", {}).get("final_state")
            == normal.get("state_hashes", {}).get("final_state"),
            "trajectory": trial.get("state_hashes", {}).get("trajectory")
            == normal.get("state_hashes", {}).get("trajectory"),
        }
        details[name] = row
        checks[name] = all(row.values())
    return {"checks": checks, "details": details, "passed": all(checks.values())}


def _arm_evaluation(
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: Mapping[str, PreparedSplit],
    transformed_ledger: Mapping[str, Mapping[str, Any]],
    *,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full: dict[str, Any] = {}
    compact: dict[str, Any] = {}
    for split in ("validation", "supported", "heldout", "causal", "metamorphic"):
        value = evaluate_model(
            model,
            cache,
            prepared[split],
            device=device,
            include_state_hash=split == "heldout",
        )
        full[split] = value
        compact[split] = _compact_evaluation(value)
    intervention_full: dict[str, Any] = {}
    intervention_compact: dict[str, Any] = {}
    for name in INTERVENTIONS:
        value = evaluate_model(
            model,
            cache,
            prepared["heldout"],
            device=device,
            intervention=name,
            include_state_hash=name in _ROUTE_INTERVENTIONS,
        )
        intervention_full[name] = value
        intervention_compact[name] = _compact_evaluation(value)
    consistency = metamorphic_consistency(
        full["causal"],
        full["metamorphic"],
        transformed_ledger,
    )
    route_consistency = metamorphic_route_consistency(
        full["causal"],
        full["metamorphic"],
        transformed_ledger,
    )
    return (
        {
            "evaluations": compact,
            "interventions": intervention_compact,
            "metamorphic_consistency": consistency,
            "metamorphic_route_consistency": route_consistency,
        },
        {**full, "interventions": intervention_full},
    )


def _run_smoke(
    root: Path,
    smoke: FormalDataPackage,
    cache: TokenCache,
    *,
    device: str,
) -> dict[str, Any]:
    prepared = prepare_split(
        smoke.bundle,
        "train",
        cache,
        steps=MODEL.recurrent_steps,
    )
    training = replace(
        TRAINING,
        batch_size=int(SMOKE.batch_size),
        updates=int(SMOKE.updates),
        evaluation_interval=int(SMOKE.evaluation_interval),
    )
    schedule_seed = SMOKE_MODEL_SEED ^ 0x534D4F4B
    schedule = build_balanced_schedule(
        prepared,
        updates=training.updates,
        batch_size=training.batch_size,
        seed=schedule_seed,
    )
    schedule_report = schedule_audit(prepared, schedule, seed=schedule_seed)
    shared, mixed, initialization = build_pair(
        formal_model_config(),
        seed=SMOKE_MODEL_SEED,
    )
    arms: dict[str, Any] = {}
    for name, model in (("shared", shared), ("mixed", mixed)):
        integrity = model.integrity_report()
        train = train_fixed_budget(
            model,
            cache,
            prepared,
            schedule,
            training,
            device=device,
        )
        evaluation = evaluate_model(model, cache, prepared, device=device)
        strip = strip_and_save(
            model,
            root / "smoke" / f"{name}-deployment.pt",
            cache,
            prepared,
            device=device,
        )
        gradients = [
            float(row["gradient_norm_before_clip"])
            for row in train["history"]
            if isinstance(row.get("gradient_norm_before_clip"), (int, float))
        ]
        checks = {
            "integrity": integrity.get("passed") is True,
            "finite_losses": bool(train["final_losses"])
            and all(math.isfinite(float(value)) for value in train["final_losses"].values()),
            "active_gradients": bool(gradients)
            and all(math.isfinite(value) and value > 0.0 for value in gradients),
            "answer_floor": evaluation["answer"]["macro_accuracy"]
            >= float(SMOKE.answer_floor),
            "trace_floor": evaluation["trace"]["cell_accuracy"]
            >= float(SMOKE.trace_floor),
            "route_floor": evaluation["route"]["accuracy"]
            >= float(SMOKE.route_floor),
            "strip_reload": strip["passed"] is True,
        }
        arms[name] = {
            "checks": checks,
            "passed": all(checks.values()),
            "integrity": integrity,
            "training": train,
            "evaluation": _compact_evaluation(evaluation),
            "strip": strip,
        }
        model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checks = {
        "schedule": schedule_report["passed"] is True,
        "initialization": initialization["passed"] is True,
        "shared": arms["shared"]["passed"] is True,
        "mixed": arms["mixed"]["passed"] is True,
    }
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.smoke.v1",
        "data_seed": SMOKE_DATA_SEED,
        "model_seed": SMOKE_MODEL_SEED,
        "training": asdict(training),
        "schedule": schedule_report,
        "initialization": initialization,
        "arms": arms,
        "checks": checks,
        "passed": all(checks.values()),
        "qualifying_evidence": False,
    }


def _runner_snapshot(
    repo_root: Path,
    identity: str,
    environment: Mapping[str, Any],
    preflight_validation: Mapping[str, Any],
) -> dict[str, Any]:
    from . import runner

    files = runner.source_files(repo_root)
    excluded = (*ROOTS.values(), TRANSPORT_ROOT)
    current_git = git_identity(repo_root, excluded)
    start_git = environment.get("git_identity_start", {})
    runtime = runner.formal_runtime_audit(
        repo_root,
        (ROOTS["preflight"], ROOTS["development"], TRANSPORT_ROOT),
    )
    source_stable = (
        identity
        == environment.get("snapshot_identity")
        == source_identity(repo_root, files)
    )
    git_stable = (
        start_git.get("passed") is True
        and current_git.get("passed") is True
        and start_git.get("head") == current_git.get("head")
        and start_git.get("branch") == current_git.get("branch")
        and start_git.get("status_sha256") == current_git.get("status_sha256")
    )
    checks = preflight_validation.get("checks", {})
    fields = {
        "preflight_seal_verified": checks.get("seal") is True,
        "source_identity_stable": source_stable,
        "git_identity_stable": git_stable,
        "later_roots_absent": runtime.get("successors", {}).get("passed") is True,
        "process_chain_unique": runtime.get("process", {}).get("passed") is True,
    }
    return {
        **fields,
        "passed": all(fields.values()),
        "current_git": current_git,
        "runtime": runtime,
    }


def _data_gate_view(data_audit: Mapping[str, Any], seed: int) -> dict[str, Any]:
    row = data_audit.get("per_seed", {}).get(str(seed), {})
    return {
        "passed": data_audit.get("passed") is True and row.get("passed") is True,
        "split_counts": row.get("split_counts"),
        "semantic_overlap": data_audit.get("semantic_overlap", []),
        "source_overlap": data_audit.get("source_overlap", []),
        "forbidden_field_hits": data_audit.get("forbidden_field_hits", []),
        "forbidden_source_scan": data_audit.get("forbidden_source_scan", {}),
    }


def _run_seed_pair(
    *,
    repo_root: Path,
    root: Path,
    package: FormalDataPackage,
    model_seed: int,
    cache: TokenCache,
    cache_audit: Mapping[str, Any],
    data_audit: Mapping[str, Any],
    architecture: Mapping[str, Any],
    smoke: Mapping[str, Any],
    identity: str,
    environment: Mapping[str, Any],
    preflight_validation: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    bundle = package.bundle
    prepared: dict[str, PreparedSplit] = {
        split: prepare_split(
            bundle,
            split,
            cache,
            steps=MODEL.recurrent_steps,
        )
        for split in ("train", "validation", "supported", "heldout", "causal")
    }
    prepared["metamorphic"] = prepare_materialized_split(
        package.transformed_records,
        package.transformed_ledger,
        cache,
        steps=MODEL.recurrent_steps,
    )
    schedule_seed = model_seed ^ 0x484131
    schedule = build_balanced_schedule(
        prepared["train"],
        updates=TRAINING.updates,
        batch_size=TRAINING.batch_size,
        seed=schedule_seed,
    )
    schedule_report = schedule_audit(
        prepared["train"],
        schedule,
        seed=schedule_seed,
    )
    shared, mixed, initialization = build_pair(
        formal_model_config(),
        seed=model_seed,
    )
    maximum_tokens = int(cache.manifest["maximum_tokens"])
    active_flops = {
        "shared": shared.active_flop_estimate(
            batch_size=TRAINING.batch_size,
            source_tokens=maximum_tokens,
        ),
        "mixed": mixed.active_flop_estimate(
            batch_size=TRAINING.batch_size,
            source_tokens=maximum_tokens,
        ),
    }
    arm_reports: dict[str, Any] = {}
    arm_full: dict[str, Any] = {}
    for name, model in (("shared", shared), ("mixed", mixed)):
        integrity = model.integrity_report()
        training = train_fixed_budget(
            model,
            cache,
            prepared["train"],
            schedule,
            TRAINING,
            device=device,
        )
        evaluation_report, full = _arm_evaluation(
            model,
            cache,
            prepared,
            package.transformed_ledger,
            device=device,
        )
        checkpoint = (
            root
            / "checkpoints"
            / f"seed-{package.seed}-{model_seed}-{name}-deployment.pt"
        )
        strip = strip_and_save(
            model,
            checkpoint,
            cache,
            prepared["supported"],
            device=device,
        )
        arm_reports[name] = {
            "integrity": integrity,
            "training": training,
            **evaluation_report,
            "strip": strip,
        }
        arm_full[name] = full
        model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shared_noop = _shared_route_noop(
        arm_full["shared"]["heldout"],
        arm_full["shared"]["interventions"],
    )
    mixed_drops = {
        name: _drop(
            arm_full["mixed"]["heldout"],
            arm_full["mixed"]["interventions"][name],
        )
        for name in (*_ROUTE_INTERVENTIONS, _CONDITIONAL_WRITE_INTERVENTION)
    }
    source_drops: dict[str, float] = {}
    for name in ("zero_source", "shuffle_source"):
        values = _drop(
            arm_full["mixed"]["heldout"],
            arm_full["mixed"]["interventions"][name],
        )
        source_drops[f"{name}.answer"] = values["answer"]
        source_drops[f"{name}.trace"] = values["trace"]
    recurrence_values = _drop(
        arm_full["mixed"]["heldout"],
        arm_full["mixed"]["interventions"]["disable_recurrence"],
    )
    recurrence_drop = {
        "disable_recurrence.answer": recurrence_values["answer"],
        "disable_recurrence.trace": recurrence_values["trace"],
    }
    comparison = primary_comparison(
        arm_full["shared"]["heldout"],
        arm_full["mixed"]["heldout"],
        bootstrap_seed=model_seed ^ 0xB00757,
        primary_gain=THRESHOLDS.primary_gain,
        family_regression_limit=THRESHOLDS.family_regression_limit,
    )
    shared_training = arm_reports["shared"]["training"]
    mixed_training = arm_reports["mixed"]["training"]
    wall_delta = float(mixed_training["wall_seconds"] - shared_training["wall_seconds"])
    runner = _runner_snapshot(
        repo_root,
        identity,
        environment,
        preflight_validation,
    )
    run = {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.seed-run.v1",
        "data_seed": package.seed,
        "model_seed": model_seed,
        "data": _data_gate_view(data_audit, package.seed),
        "cache": dict(cache_audit),
        "paired_inputs_passed": (
            schedule_report["passed"] is True
            and initialization["passed"] is True
            and cache_audit.get("passed") is True
        ),
        "architecture_passed": architecture.get("passed") is True,
        "runtime_probes_passed": (
            architecture.get("passed") is True and smoke.get("passed") is True
        ),
        "initialization": initialization,
        "shared": arm_reports["shared"],
        "mixed": arm_reports["mixed"],
        "shared_noop": shared_noop,
        "mixed_causal_drops": mixed_drops,
        "route_metamorphic": arm_reports["mixed"][
            "metamorphic_route_consistency"
        ],
        "source_causal_drops": source_drops,
        "recurrence_causal_drop": recurrence_drop,
        "metamorphic": arm_reports["mixed"]["metamorphic_consistency"],
        "budget": {
            "updates": TRAINING.updates,
            "batch_size": TRAINING.batch_size,
            "examples_seen": TRAINING.updates * TRAINING.batch_size,
        },
        "schedule": {
            **schedule_report,
            "shared_sha256": schedule_report["sha256"],
            "mixed_sha256": schedule_report["sha256"],
        },
        "active_flops": active_flops,
        "runner": runner,
        "runtime": {
            "updates_per_second": min(
                float(shared_training["updates_per_second"]),
                float(mixed_training["updates_per_second"]),
            ),
            "examples_per_second": min(
                float(shared_training["examples_per_second"]),
                float(mixed_training["examples_per_second"]),
            ),
            "dispatch_overhead": max(0.0, wall_delta),
            "mixed_minus_shared_wall_seconds": wall_delta,
        },
        "primary_comparison": comparison,
    }
    del arm_full, shared, mixed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run


def _development_failure(
    identity: str,
    reason: str,
    *,
    partial_runs: Sequence[Mapping[str, Any]] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.development-result.v1",
        "contract_version": CONTRACT_VERSION,
        "status": FAIL_DEVELOPMENT,
        "passed": False,
        "source_identity": identity,
        "stop_reason": reason,
        "completed_seed_pairs": [
            [int(run["data_seed"]), int(run["model_seed"])]
            for run in partial_runs
        ],
        "details": dict(details or {}),
        "claims": _claims(False),
    }


def development_action(
    repo_root: Path,
    root: Path,
    identity: str,
    environment: dict[str, Any],
    *,
    preflight_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run smoke and the three registered pairs, stopping on the first Gate."""
    from . import runner

    validation = dict(
        preflight_validation
        if preflight_validation is not None
        else runner.preflight_valid(repo_root, identity)
    )
    if validation.get("passed") is not True:
        return _development_failure(
            identity,
            "preflight_not_current_sealed_pass",
            details={"preflight_validation": validation},
        )

    preflight_root = repo_root / ROOTS["preflight"]
    preflight_result = _load_json(preflight_root / "result.json")
    data_manifest = _load_json(preflight_root / "data-manifest.json")
    data_audit = _load_json(preflight_root / "data-audit.json")
    architecture = _load_json(preflight_root / "architecture-audit.json")
    packages, smoke_package, reservations = generate_registered_inputs(repo_root)
    records = combined_records(packages, smoke_package)
    identity_checks = {
        "preflight_seal": verify_seal(preflight_root),
        "record_count": data_manifest.get("record_count") == len(records),
        "record_order": data_manifest.get("record_order_identity")
        == _record_order_identity(records),
        "smoke_identity": data_manifest.get("packages", {})
        .get(str(SMOKE_DATA_SEED), {})
        .get("identity")
        == package_identity(smoke_package),
        "reserved_nonformal": data_manifest.get("reserved_nonformal")
        == {
            role: {
                "seed": package.seed,
                "identity": package_identity(package),
                "record_count": len(package.records),
                "semantic_fingerprint_count": len(package_fingerprint_sets(package)[0]),
                "source_fingerprint_count": len(package_fingerprint_sets(package)[1]),
                "included_in_formal_cache": False,
            }
            for role, package in reservations.items()
        },
    }
    for package in packages:
        identity_checks[f"package_{package.seed}"] = (
            data_manifest.get("packages", {})
            .get(str(package.seed), {})
            .get("identity")
            == package_identity(package)
        )
    replay_data_audit = data_leakage_audit(
        repo_root, packages, smoke_package, reservations
    )
    identity_checks["data_audit"] = replay_data_audit["passed"] is True
    cache_root = preflight_root / str(data_manifest["cache_root"])
    cache_audit = audit_token_cache(records, cache_root, verify_all_finite=True)
    identity_checks["cache"] = cache_audit["passed"] is True
    replay = {
        "checks": identity_checks,
        "passed": all(identity_checks.values()),
        "cache_audit": cache_audit,
        "data_audit": replay_data_audit,
    }
    write_json(root / "preflight-input-replay.json", replay)
    if not replay["passed"]:
        return _development_failure(
            identity,
            "preflight_input_replay_failed",
            details={"replay": replay},
        )

    cache = TokenCache(cache_root)
    smoke = _run_smoke(root, smoke_package, cache, device="cuda")
    write_json(root / "smoke-result.json", smoke)
    if smoke["passed"] is not True:
        return _development_failure(
            identity,
            "smoke_gate_failed",
            details={"smoke": smoke},
        )

    current_runtime = runner.formal_runtime_audit(
        repo_root,
        (ROOTS["preflight"], ROOTS["development"], TRANSPORT_ROOT),
    )
    context = {
        "thresholds_frozen": THRESHOLDS.frozen(),
        "contract_hashes_frozen": runner.contract_hash_audit(repo_root)["passed"] is True,
        "prior_evidence_verified": runner.history_replay(repo_root)["passed"] is True,
        "roots_fresh": preflight_result.get("formal_entry_audit", {})
        .get("fixed_roots", {})
        .get("passed")
        is True,
        "process_chain_unique": current_runtime.get("process", {}).get("passed") is True,
        "no_successor_roots": current_runtime.get("successors", {}).get("passed") is True,
        "command_exact": runner._cli_audit(repo_root)["passed"] is True,
    }
    write_json(root / "qualification-context.json", context)
    runs: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for package, model_seed in zip(packages, MODEL_SEEDS, strict=True):
        run = _run_seed_pair(
            repo_root=repo_root,
            root=root,
            package=package,
            model_seed=model_seed,
            cache=cache,
            cache_audit=cache_audit,
            data_audit=replay_data_audit,
            architecture=architecture,
            smoke=smoke,
            identity=identity,
            environment=environment,
            preflight_validation=validation,
            device="cuda",
        )
        runs.append(run)
        relative = Path("runs") / f"seed-{package.seed}-{model_seed}.json"
        _write_compact_json(root / relative, run)
        artifact_rows.append(
            {
                "data_seed": package.seed,
                "model_seed": model_seed,
                "path": relative.as_posix(),
                "sha256": sha256(root / relative),
            }
        )

        single = aggregate_h1_qualification(
            [run],
            context=context,
            thresholds=THRESHOLDS,
            expected_pairs=((package.seed, model_seed),),
            split_counts=SPLIT_COUNTS,
            training_spec=asdict(TRAINING),
        )
        fail_stop_gates = ("H01", "H02", "H03", "H04", "H06", "H07", "H08")
        failed = [name for name in fail_stop_gates if single["gates"][name]["passed"] is not True]
        if failed:
            return _development_failure(
                identity,
                "registered_gate_failed",
                partial_runs=runs,
                details={
                    "failed_gates": failed,
                    "partial_qualification": single,
                    "run_artifacts": artifact_rows,
                },
            )
        if run["primary_comparison"]["overall"]["gain"] <= 0.0:
            return _development_failure(
                identity,
                "H05_seed_direction_failed",
                partial_runs=runs,
                details={
                    "primary_comparison": run["primary_comparison"],
                    "run_artifacts": artifact_rows,
                },
            )

    result = aggregate_h1_qualification(
        runs,
        context=context,
        thresholds=THRESHOLDS,
        expected_pairs=tuple(zip(DATA_SEEDS, MODEL_SEEDS, strict=True)),
        split_counts=SPLIT_COUNTS,
        training_spec=asdict(TRAINING),
    )
    result.update(
        {
            "contract_version": CONTRACT_VERSION,
            "source_identity": identity,
            "preflight_validation": validation,
            "run_artifacts": artifact_rows,
            "smoke_artifact": "smoke-result.json",
        }
    )
    write_json(root / "qualification-ledger.json", result)
    return result


__all__ = [
    "combined_records",
    "data_leakage_audit",
    "development_action",
    "formal_model_config",
    "generate_registered_inputs",
    "hardware_audit",
    "persist_data_packages",
    "preflight_action",
]
