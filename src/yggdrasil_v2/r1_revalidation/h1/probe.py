from __future__ import annotations

"""Isolated, non-formal H1 engineering and threshold-calibration probes.

This module deliberately has no project CLI registration.  A probe may tune
an unfrozen H1 contract, but it cannot create either fixed formal root and it
cannot authorize H1, F1, P1, or P2.
"""

from dataclasses import asdict, dataclass, replace
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from .audit import architecture_audit
from .cache import TokenCache, audit_token_cache, build_token_cache
from .contract import (
    CALIBRATION_DATA_SEED,
    CALIBRATION_MARGINS,
    CALIBRATION_MODEL_SEED,
    MODEL,
    SCREEN_DATA_SEED,
    SCREEN_MODEL_SEED,
    SPLIT_COUNTS,
    THRESHOLDS,
    TRAINING,
)
from .formal_data import FormalDataPackage, package_identity
from .fresh_data import generate_bundle, validate_bundle
from .metrics import primary_comparison
from .model import H1Config, H1LatentReasoner
from .train import (
    INTERVENTIONS,
    build_balanced_schedule,
    build_pair,
    evaluate_model,
    materialize_causal_records,
    metamorphic_consistency,
    metamorphic_route_consistency,
    prepare_materialized_split,
    prepare_split,
    schedule_audit,
    strip_and_save,
    train_fixed_budget,
)


_ROUTE_SCREEN_INTERVENTIONS = (
    "flip_route",
    "force_route0",
    "force_route1",
    "swap_experts",
)
_METAMORPHIC_TRANSFORMS = (
    "cancelling_pair",
    "choice_permutation",
    "clause_permutation",
    "handle_rename",
    "query_permutation",
    "surface_paraphrase",
    "transitive_redundancy",
)


@dataclass(frozen=True)
class ProbeSpec:
    data_seed: int
    model_seed: int
    train_per_family: int = 256
    validation_per_family: int = 32
    supported_per_family: int = 64
    heldout_per_family: int = 64
    causal_per_family: int = 32
    materialized_bases_per_family: int = 8
    updates: int = 750
    batch_size: int = 32
    cache_batch_size: int = 8

    def counts(self) -> dict[str, dict[str, int]]:
        per_split = {
            "train": self.train_per_family,
            "validation": self.validation_per_family,
            "supported": self.supported_per_family,
            "heldout": self.heldout_per_family,
            "causal": self.causal_per_family,
        }
        return {family: dict(per_split) for family in ("numeric", "relation")}


def registered_full_probe_spec(data_scope: str) -> ProbeSpec:
    """Construct the only full-budget spec allowed for a registered scope."""
    seeds = {
        "screen": (SCREEN_DATA_SEED, SCREEN_MODEL_SEED),
        "calibration": (CALIBRATION_DATA_SEED, CALIBRATION_MODEL_SEED),
    }
    if data_scope not in seeds:
        raise ValueError("registered scope must be screen or calibration")
    data_seed, model_seed = seeds[data_scope]
    per_family = {split: count // 2 for split, count in SPLIT_COUNTS.items()}
    return ProbeSpec(
        data_seed=data_seed,
        model_seed=model_seed,
        train_per_family=per_family["train"],
        validation_per_family=per_family["validation"],
        supported_per_family=per_family["supported"],
        heldout_per_family=per_family["heldout"],
        causal_per_family=per_family["causal"],
        materialized_bases_per_family=64,
        updates=TRAINING.updates,
        batch_size=TRAINING.batch_size,
        cache_batch_size=8,
    )


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _compact_eval(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": value["split"],
        "intervention": value["intervention"],
        "answer": value["answer"],
        "route": value["route"],
        "trace": value.get("trace"),
    }


def _arm_evaluations(
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: dict[str, Any],
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
        compact[split] = _compact_eval(value)
    interventions: dict[str, Any] = {}
    intervention_full: dict[str, Any] = {}
    for intervention in INTERVENTIONS:
        value = evaluate_model(
            model,
            cache,
            prepared["heldout"],
            device=device,
            intervention=intervention,
            include_state_hash=True,
        )
        intervention_full[intervention] = value
        interventions[intervention] = _compact_eval(value)
    full["interventions"] = intervention_full
    compact["interventions"] = interventions
    return full, compact


def _route_noop(normal: dict[str, Any], trials: dict[str, Any]) -> dict[str, Any]:
    checks = {
        name: (
            trials[name]["predictions"] == normal["predictions"]
            and trials[name].get("trace_predictions") == normal.get("trace_predictions")
            and trials[name].get("state_hashes") == normal.get("state_hashes")
        )
        for name in _ROUTE_SCREEN_INTERVENTIONS
    }
    return {"checks": checks, "passed": all(checks.values())}


def _drop(normal: dict[str, Any], trial: dict[str, Any]) -> dict[str, float]:
    return {
        "answer": float(normal["answer"]["macro_accuracy"] - trial["answer"]["macro_accuracy"]),
        "trace": float(normal["trace"]["cell_accuracy"] - trial["trace"]["cell_accuracy"]),
    }


def direction_screen_gate(
    report: dict[str, Any],
    *,
    route_causal_threshold: float = THRESHOLDS.route_causal_drop,
    conditional_write_causal_threshold: float = (
        THRESHOLDS.conditional_write_causal_drop
    ),
) -> dict[str, Any]:
    """Apply only the registered architecture-direction checks.

    Passing this screen does not freeze thresholds and cannot qualify H1.  It
    merely permits a new-seed calibration run; all H01--H08 formal checks stay
    outside this non-formal helper.
    """
    candidates = [
        {
            "intervention": intervention,
            "metric": metric,
            "drop": float(report["mixed_intervention_drops"][intervention][metric]),
        }
        for intervention in _ROUTE_SCREEN_INTERVENTIONS
        for metric in ("answer", "trace")
    ]
    maximum = max(candidates, key=lambda row: row["drop"])
    conditional_write = {
        "intervention": "disable_routed_projection",
        "answer": float(
            report["mixed_intervention_drops"]["disable_routed_projection"][
                "answer"
            ]
        ),
        "trace": float(
            report["mixed_intervention_drops"]["disable_routed_projection"][
                "trace"
            ]
        ),
    }
    conditional_write["maximum"] = max(
        conditional_write["answer"], conditional_write["trace"]
    )
    checks = {
        "primary_h05": report["primary_comparison"]["gate"]["passed"] is True,
        "route_causal_h06": maximum["drop"] >= float(route_causal_threshold),
        "conditional_write_h06": conditional_write["maximum"]
        >= float(conditional_write_causal_threshold),
        "shared_route_noop": report["shared_route_noop"]["passed"] is True,
        "architecture": report["architecture"]["passed"] is True,
        "cache": report["cache_audit"]["passed"] is True,
        "active_flops_matched": report["active_flops"]["matched"] is True,
        "shared_strip_reload": report["shared"]["strip"]["passed"] is True,
        "mixed_strip_reload": report["mixed"]["strip"]["passed"] is True,
    }
    return {
        "scope": "NONFORMAL_DIRECTION_SCREEN_ONLY",
        "route_causal_threshold": float(route_causal_threshold),
        "conditional_write_causal_threshold": float(
            conditional_write_causal_threshold
        ),
        "maximum_registered_route_effect": maximum,
        "conditional_write_effect": conditional_write,
        "checks": checks,
        "passed": all(checks.values()),
        "authorizes": "fresh-seed calibration only" if all(checks.values()) else "nothing",
        "h1_qualified": False,
        "p1_f1_design_authorized": False,
    }


def _number_at(value: Any, *path: str) -> float:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"missing calibration metric: {'.'.join(path)}")
        current = current[key]
    if (
        isinstance(current, bool)
        or not isinstance(current, (int, float))
        or not math.isfinite(float(current))
    ):
        raise ValueError(f"non-finite calibration metric: {'.'.join(path)}")
    result = float(current)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"out-of-range calibration metric: {'.'.join(path)}={result}")
    return result


def _evaluation_floor_values(
    report: dict[str, Any], arm: str, split: str, metric: str
) -> list[float]:
    evaluation = report[arm]["evaluations"][split]
    if metric == "answer":
        values = [_number_at(evaluation, "answer", "macro_accuracy")]
        values.extend(
            _number_at(evaluation, "answer", "by_family", family, "accuracy")
            for family in ("numeric", "relation")
        )
        return values
    values = [_number_at(evaluation, "trace", "cell_accuracy")]
    values.extend(
        _number_at(evaluation, "trace", "by_family", family, "cell_accuracy")
        for family in ("numeric", "relation")
    )
    return values


def _metamorphic_floor_values(
    value: dict[str, Any], keys: tuple[str, str]
) -> list[float]:
    by_transform = value.get("by_transform")
    by_family = value.get("by_family")
    if not isinstance(by_transform, dict) or set(by_transform) != set(
        _METAMORPHIC_TRANSFORMS
    ):
        raise ValueError("calibration metamorphic transform registry mismatch")
    if not isinstance(by_family, dict) or set(by_family) != {"numeric", "relation"}:
        raise ValueError("calibration metamorphic family registry mismatch")
    rows = [value.get("aggregate")]
    rows.extend(by_transform[name] for name in _METAMORPHIC_TRANSFORMS)
    rows.extend(by_family[name] for name in ("numeric", "relation"))
    result: list[float] = []
    for row in rows:
        for key in keys:
            result.append(_number_at(row, key))
    return result


def _margin_floor(observed: float, margin: float) -> float:
    quantum = float(CALIBRATION_MARGINS.floor_quantum)
    if quantum <= 0.0:
        raise ValueError("calibration floor quantum must be positive")
    units = math.floor(max(0.0, observed - margin) / quantum + 1.0e-12)
    return round(units * quantum, 10)


def calibration_threshold_proposal(report: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen pre-result margin rule to one registered calibration."""
    try:
        observed = {
            "supported_answer_floor": min(
                value
                for arm in ("shared", "mixed")
                for value in _evaluation_floor_values(
                    report, arm, "supported", "answer"
                )
            ),
            "supported_trace_floor": min(
                value
                for arm in ("shared", "mixed")
                for value in _evaluation_floor_values(
                    report, arm, "supported", "trace"
                )
            ),
            "heldout_shared_answer_floor": min(
                _evaluation_floor_values(report, "shared", "heldout", "answer")
            ),
            "heldout_mixed_answer_floor": min(
                _evaluation_floor_values(report, "mixed", "heldout", "answer")
            ),
            "heldout_shared_trace_floor": min(
                _evaluation_floor_values(report, "shared", "heldout", "trace")
            ),
            "heldout_mixed_trace_floor": min(
                _evaluation_floor_values(report, "mixed", "heldout", "trace")
            ),
        }
        drops = report["mixed_intervention_drops"]
        observed["source_causal_drop"] = min(
            max(
                _number_at(drops, intervention, "answer"),
                _number_at(drops, intervention, "trace"),
            )
            for intervention in ("zero_source", "shuffle_source")
        )
        observed["recurrence_causal_drop"] = max(
            _number_at(drops, "disable_recurrence", "answer"),
            _number_at(drops, "disable_recurrence", "trace"),
        )
        observed["metamorphic_consistency"] = min(
            _metamorphic_floor_values(
                report["mixed"]["metamorphic_consistency"],
                ("relation_consistency", "transformed_exact"),
            )
        )
        route_values = [
            _number_at(
                report, "mixed", "evaluations", "heldout", "route", "accuracy"
            )
        ]
        route_values.extend(
            _metamorphic_floor_values(
                report["mixed"]["metamorphic_route_consistency"],
                ("consistency", "transformed_exact"),
            )
        )
        observed["route_accuracy"] = min(route_values)
    except (KeyError, TypeError, ValueError) as error:
        return {
            "scope": "NONFORMAL_CALIBRATION_THRESHOLD_PROPOSAL",
            "passed": False,
            "error": str(error),
            "thresholds": None,
        }

    margin_by_threshold = {
        key: float(CALIBRATION_MARGINS.competence)
        for key in (
            "supported_answer_floor",
            "supported_trace_floor",
            "heldout_shared_answer_floor",
            "heldout_mixed_answer_floor",
            "heldout_shared_trace_floor",
            "heldout_mixed_trace_floor",
        )
    }
    margin_by_threshold.update(
        {
            "source_causal_drop": float(CALIBRATION_MARGINS.causal_drop),
            "recurrence_causal_drop": float(CALIBRATION_MARGINS.causal_drop),
            "metamorphic_consistency": float(CALIBRATION_MARGINS.metamorphic),
            "route_accuracy": float(CALIBRATION_MARGINS.route_accuracy),
        }
    )
    derived = {
        key: _margin_floor(value, margin_by_threshold[key])
        for key, value in observed.items()
    }
    thresholds = {
        **derived,
        "primary_gain": float(THRESHOLDS.primary_gain),
        "family_regression_limit": float(THRESHOLDS.family_regression_limit),
        "route_causal_drop": float(THRESHOLDS.route_causal_drop),
        "conditional_write_causal_drop": float(
            THRESHOLDS.conditional_write_causal_drop
        ),
    }
    checks = {
        "all_observed_finite": all(math.isfinite(value) for value in observed.values()),
        "all_derived_nontrivial": all(value > 0.0 for value in derived.values()),
        "h05_primary_unchanged": thresholds["primary_gain"]
        == float(THRESHOLDS.primary_gain),
        "h05_regression_unchanged": thresholds["family_regression_limit"]
        == float(THRESHOLDS.family_regression_limit),
        "h06_route_drop_unchanged": thresholds["route_causal_drop"]
        == float(THRESHOLDS.route_causal_drop),
        "h06_conditional_write_drop_unchanged": thresholds[
            "conditional_write_causal_drop"
        ]
        == float(THRESHOLDS.conditional_write_causal_drop),
    }
    return {
        "scope": "NONFORMAL_CALIBRATION_THRESHOLD_PROPOSAL",
        "rule": "worst registered cell minus fixed margin, rounded down to 0.01",
        "margins": asdict(CALIBRATION_MARGINS),
        "observed_minima": observed,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def calibration_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Authorize contract-freeze audit, never H1, from the reserved seed only."""
    proposal = calibration_threshold_proposal(report)
    spec = report.get("probe_spec", {})
    registration = report.get("data_registration", {})
    expected_per_family = {
        f"{split}_per_family": count // 2 for split, count in SPLIT_COUNTS.items()
    }
    checks = {
        "nonformal_status": report.get("status") == "NONFORMAL_P1_H1_PROBE_ONLY",
        "registered_calibration_data": isinstance(registration, dict)
        and registration.get("scope") == "calibration"
        and registration.get("registered") is True
        and registration.get("seed") == CALIBRATION_DATA_SEED
        and isinstance(registration.get("package_identity"), str),
        "registered_calibration_model": isinstance(spec, dict)
        and spec.get("model_seed") == CALIBRATION_MODEL_SEED,
        "full_split_counts": isinstance(spec, dict)
        and all(spec.get(key) == value for key, value in expected_per_family.items()),
        "full_materialized_set": isinstance(spec, dict)
        and spec.get("materialized_bases_per_family") == 64,
        "full_training_budget": isinstance(spec, dict)
        and spec.get("updates") == TRAINING.updates
        and spec.get("batch_size") == TRAINING.batch_size,
        "fresh_cache": report.get("cache_origin") == "built_for_this_probe",
        "direction_gate": report.get("direction_screen_gate", {}).get("passed")
        is True,
        "threshold_proposal": proposal.get("passed") is True,
        "nonqualifying_claims": all(
            report.get("claims", {}).get(key) is False
            for key in (
                "h1_qualified",
                "p1_f1_design_authorized",
                "p1_completed",
                "p2_eligible",
                "p2_started",
            )
        ),
    }
    passed = all(checks.values())
    return {
        "scope": "NONFORMAL_CALIBRATION_GATE_ONLY",
        "checks": checks,
        "threshold_proposal": proposal,
        "passed": passed,
        "authorizes": "contract-freeze audit only" if passed else "nothing",
        "h1_qualified": False,
        "p1_f1_design_authorized": False,
    }


def _train_arm(
    arm: str,
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: dict[str, Any],
    schedule: Any,
    training: Any,
    root: Path,
    *,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_report = train_fixed_budget(
        model,
        cache,
        prepared["train"],
        schedule,
        training,
        device=device,
    )
    full, compact = _arm_evaluations(model, cache, prepared, device=device)
    strip = strip_and_save(
        model,
        root / f"{arm}-deployment.pt",
        cache,
        prepared["supported"],
        device=device,
    )
    report = {
        "training": train_report,
        "evaluations": compact,
        "strip": strip,
    }
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return full, report


def run_nonformal_probe(
    output_root: Path,
    spec: ProbeSpec,
    *,
    device: str = "cuda",
    existing_cache_root: Path | None = None,
    data_package: FormalDataPackage | None = None,
    data_scope: str = "unregistered_probe",
) -> dict[str, Any]:
    """Run one isolated probe and persist an explicitly non-qualifying report."""
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"non-formal H1 probe root already exists: {output_root}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("non-formal H1 probe requested unavailable CUDA")
    architecture = architecture_audit(
        formal_model_config(),
        seed=spec.model_seed,
    )
    if not architecture["passed"]:
        raise RuntimeError(f"non-formal H1 architecture audit failed: {architecture}")
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    expected_registered_seed = {
        "screen": SCREEN_DATA_SEED,
        "calibration": CALIBRATION_DATA_SEED,
    }.get(data_scope)
    expected_registered_model_seed = {
        "screen": SCREEN_MODEL_SEED,
        "calibration": CALIBRATION_MODEL_SEED,
    }.get(data_scope)
    if (
        expected_registered_model_seed is not None
        and spec.model_seed != expected_registered_model_seed
    ):
        raise ValueError("non-formal scope and ProbeSpec model seed disagree")
    if data_package is None:
        if expected_registered_seed is not None:
            raise ValueError(f"registered {data_scope} requires a data package")
        bundle = generate_bundle(spec.data_seed, spec.counts())
        transformed_records, transformed_ledger = materialize_causal_records(
            bundle,
            per_family=spec.materialized_bases_per_family,
        )
        registration = {
            "scope": data_scope,
            "registered": False,
            "seed": spec.data_seed,
            "package_identity": None,
        }
    else:
        if expected_registered_seed is None:
            raise ValueError("registered data package requires screen or calibration scope")
        if data_package.seed != spec.data_seed or spec.data_seed != expected_registered_seed:
            raise ValueError("non-formal package, scope, and ProbeSpec seed disagree")
        validate_bundle(data_package.bundle)
        expected_split_counts = {
            split: sum(family_counts[split] for family_counts in spec.counts().values())
            for split in spec.counts()["numeric"]
        }
        actual_split_counts = {
            split: len(ids) for split, ids in data_package.bundle.splits.items()
        }
        if actual_split_counts != expected_split_counts:
            raise ValueError("registered non-formal package split counts disagree")
        bundle = data_package.bundle
        transformed_records = data_package.transformed_records
        transformed_ledger = data_package.transformed_ledger
        registration = {
            "scope": data_scope,
            "registered": True,
            "seed": data_package.seed,
            "package_identity": package_identity(data_package),
        }
    records = tuple(bundle.records) + transformed_records
    cache_root = (
        Path(existing_cache_root)
        if existing_cache_root is not None
        else output_root / "token-cache"
    )
    if existing_cache_root is None:
        cache_manifest = build_token_cache(
            records,
            cache_root,
            device=device,
            batch_size=spec.cache_batch_size,
        )
    else:
        cache_manifest = TokenCache(cache_root).manifest
    cache_audit = audit_token_cache(records, cache_root)
    if not cache_audit["passed"]:
        raise RuntimeError(f"non-formal H1 cache audit failed: {cache_audit}")
    cache = TokenCache(cache_root)
    prepared = {
        split: prepare_split(bundle, split, cache, steps=MODEL.recurrent_steps)
        for split in ("train", "validation", "supported", "heldout", "causal")
    }
    prepared["metamorphic"] = prepare_materialized_split(
        transformed_records,
        transformed_ledger,
        cache,
        steps=MODEL.recurrent_steps,
    )

    training = replace(
        TRAINING,
        batch_size=spec.batch_size,
        updates=spec.updates,
        evaluation_interval=max(1, spec.updates // 5),
    )
    schedule_seed = spec.model_seed ^ 0x484131
    schedule = build_balanced_schedule(
        prepared["train"],
        updates=training.updates,
        batch_size=training.batch_size,
        seed=schedule_seed,
    )
    schedule_report = schedule_audit(prepared["train"], schedule, seed=schedule_seed)
    if not schedule_report["passed"]:
        raise RuntimeError(f"non-formal H1 schedule audit failed: {schedule_report}")

    shared, mixed, initialization = build_pair(
        formal_model_config(),
        seed=spec.model_seed,
    )
    shared_full, shared_report = _train_arm(
        "shared", shared, cache, prepared, schedule, training, output_root, device=device
    )
    mixed_full, mixed_report = _train_arm(
        "mixed", mixed, cache, prepared, schedule, training, output_root, device=device
    )
    shared_report["metamorphic_consistency"] = metamorphic_consistency(
        shared_full["causal"], shared_full["metamorphic"], transformed_ledger
    )
    shared_report["metamorphic_route_consistency"] = metamorphic_route_consistency(
        shared_full["causal"], shared_full["metamorphic"], transformed_ledger
    )
    mixed_report["metamorphic_consistency"] = metamorphic_consistency(
        mixed_full["causal"], mixed_full["metamorphic"], transformed_ledger
    )
    mixed_report["metamorphic_route_consistency"] = metamorphic_route_consistency(
        mixed_full["causal"], mixed_full["metamorphic"], transformed_ledger
    )
    comparison = primary_comparison(
        shared_full["heldout"],
        mixed_full["heldout"],
        bootstrap_seed=spec.model_seed ^ 0xB00757,
    )
    mixed_drops = {
        name: _drop(mixed_full["heldout"], mixed_full["interventions"][name])
        for name in INTERVENTIONS
    }
    report = {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.nonformal-probe.v1",
        "status": "NONFORMAL_P1_H1_PROBE_ONLY",
        "claims": {
            "h1_qualified": False,
            "p1_f1_design_authorized": False,
            "p1_completed": False,
            "p2_eligible": False,
            "p2_started": False,
        },
        "probe_spec": asdict(spec),
        "data_registration": registration,
        "model": asdict(MODEL),
        "training": asdict(training),
        "record_count": len(records),
        "split_counts": {name: len(value.ids) for name, value in prepared.items()},
        "cache_manifest": cache_manifest,
        "cache_origin": (
            "built_for_this_probe"
            if existing_cache_root is None
            else str(cache_root.resolve())
        ),
        "cache_audit": cache_audit,
        "schedule": schedule_report,
        "architecture": architecture,
        "initialization": initialization,
        "active_flops": {
            "shared": shared.active_flop_estimate(
                batch_size=spec.batch_size,
                source_tokens=int(cache_manifest["maximum_tokens"]),
            ),
            "mixed": mixed.active_flop_estimate(
                batch_size=spec.batch_size,
                source_tokens=int(cache_manifest["maximum_tokens"]),
            ),
        },
        "shared": shared_report,
        "mixed": mixed_report,
        "primary_comparison": comparison,
        "shared_route_noop": _route_noop(
            shared_full["heldout"], shared_full["interventions"]
        ),
        "mixed_intervention_drops": mixed_drops,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["active_flops"]["matched"] = (
        report["active_flops"]["shared"] == report["active_flops"]["mixed"]
    )
    report["direction_screen_gate"] = direction_screen_gate(report)
    if data_scope == "calibration":
        report["calibration_gate"] = calibration_gate(report)
    _write_json(output_root / "probe-result.json", report)
    return report


__all__ = [
    "ProbeSpec",
    "calibration_gate",
    "calibration_threshold_proposal",
    "direction_screen_gate",
    "formal_model_config",
    "registered_full_probe_spec",
    "run_nonformal_probe",
]
