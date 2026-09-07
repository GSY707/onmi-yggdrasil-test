"""Pure fail-closed aggregation for the P1-H1 H01--H08 qualification.

This module deliberately performs no filesystem, process, model, or training
work.  A runner supplies one normalized mapping per data/model seed pair and
the aggregator returns an auditable result.  Missing fields are failures, not
implicit passes.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import DATA_SEEDS, MODEL_SEEDS, MODEL, SPLIT_COUNTS, THRESHOLDS
from .metrics import paired_bootstrap_gain


_ARMS = ("shared", "mixed")
_REQUIRED_ROUTE_INTERVENTIONS = (
    "flip_route",
    "force_route0",
    "force_route1",
    "swap_experts",
)
_REQUIRED_METAMORPHIC_TRANSFORMS = (
    "cancelling_pair",
    "choice_permutation",
    "clause_permutation",
    "handle_rename",
    "query_permutation",
    "surface_paraphrase",
    "transitive_redundancy",
)
_REQUIRED_CONTEXT_FLAGS = (
    "thresholds_frozen",
    "contract_hashes_frozen",
    "prior_evidence_verified",
    "roots_fresh",
    "process_chain_unique",
    "no_successor_roots",
    "command_exact",
)
_REQUIRED_INTEGRITY = (
    "source_independent_initial_state",
    "common_attention_and_ffn_state_write",
    "routed_feature_trunk_shared_across_routes",
    "selected_routed_projection_is_only_conditional_state_write",
    "route_controls_final_projection_only",
    "route_granularity",
    "conditional_transition_scope",
    "routed_feature_width",
    "routed_projection_input_width",
    "routed_projection_output_width",
    "answer_reads_pooled_final_state_only",
    "route_assignment_is_hard_top1",
)
_REQUIRED_RUNNER = (
    "preflight_seal_verified",
    "source_identity_stable",
    "git_identity_stable",
    "later_roots_absent",
    "process_chain_unique",
)


def _finite(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, Real):
        return bool(math.isfinite(float(value)))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _empty_collection(value: Any) -> bool:
    return isinstance(value, (Mapping, list, tuple, set)) and len(value) == 0


def _flag(mapping: Mapping[str, Any], key: str, missing: list[str], failures: list[str]) -> bool:
    if key not in mapping:
        missing.append(key)
        return False
    value = mapping[key]
    if not isinstance(value, (bool, np.bool_)):
        failures.append(f"{key}:not_bool")
        return False
    if not bool(value):
        failures.append(key)
    return bool(value)


def _number(mapping: Mapping[str, Any], key: str, missing: list[str], failures: list[str]) -> float | None:
    if key not in mapping:
        missing.append(key)
        return None
    value = mapping[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        failures.append(f"{key}:not_finite_number")
        return None
    return float(value)


def _thresholds(value: Any) -> tuple[dict[str, float], list[str]]:
    if is_dataclass(value):
        raw = asdict(value)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        return {}, ["thresholds:not_mapping"]
    missing: list[str] = []
    result: dict[str, float] = {}
    for key in (
        "supported_answer_floor",
        "supported_trace_floor",
        "heldout_shared_answer_floor",
        "heldout_mixed_answer_floor",
        "heldout_shared_trace_floor",
        "heldout_mixed_trace_floor",
        "primary_gain",
        "family_regression_limit",
        "route_causal_drop",
        "conditional_write_causal_drop",
        "source_causal_drop",
        "recurrence_causal_drop",
        "metamorphic_consistency",
        "route_accuracy",
    ):
        if key not in raw:
            missing.append(f"thresholds.{key}")
            continue
        item = raw[key]
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real) or not math.isfinite(float(item)) or float(item) < 0:
            missing.append(f"thresholds.{key}:unfrozen")
            continue
        result[key] = float(item)
    return result, missing


def _gate(name: str, checks: Mapping[str, bool], missing: Sequence[str] = (), failures: Sequence[str] = ()) -> dict[str, Any]:
    normalized = {key: bool(value) for key, value in checks.items()}
    missing_list = sorted(set(str(item) for item in missing))
    failure_list = sorted(set(str(item) for item in failures))
    passed = bool(normalized) and all(normalized.values()) and not missing_list and not failure_list
    return {
        "name": name,
        "passed": passed,
        "checks": normalized,
        "missing": missing_list,
        "failures": failure_list,
    }


def _seed_key(run: Mapping[str, Any]) -> str:
    return f"{int(run['data_seed'])}/{int(run['model_seed'])}"


def _validate_runs(runs: Sequence[Mapping[str, Any]], expected_pairs: Sequence[tuple[int, int]]) -> tuple[list[str], list[str], dict[str, Mapping[str, Any]]]:
    missing: list[str] = []
    failures: list[str] = []
    indexed: dict[str, Mapping[str, Any]] = {}
    expected = {f"{data}/{model}": (data, model) for data, model in expected_pairs}
    if len(runs) != len(expected_pairs):
        failures.append(f"seed_count:{len(runs)}!={len(expected_pairs)}")
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            failures.append(f"runs[{index}]:not_mapping")
            continue
        if "data_seed" not in run or "model_seed" not in run:
            missing.append(f"runs[{index}].seed_pair")
            continue
        try:
            key = _seed_key(run)
        except (TypeError, ValueError):
            failures.append(f"runs[{index}].seed_pair:invalid")
            continue
        if key in indexed:
            failures.append(f"duplicate_seed_pair:{key}")
        indexed[key] = run
        if key not in expected:
            failures.append(f"unexpected_seed_pair:{key}")
    for key in sorted(set(expected) - set(indexed)):
        missing.append(f"missing_seed_pair:{key}")
    return missing, failures, indexed


def _eval(run_arm: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    evaluations = run_arm.get("evaluations")
    if not isinstance(evaluations, Mapping):
        return None
    value = evaluations.get(split)
    return value if isinstance(value, Mapping) else None


def _metric(eval_value: Mapping[str, Any] | None, path: str) -> float | None:
    if eval_value is None:
        return None
    current: Any = eval_value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (bool, np.bool_)) or not isinstance(current, Real) or not math.isfinite(float(current)):
        return None
    return float(current)


def _vector(eval_value: Mapping[str, Any] | None, key: str) -> list[Any] | None:
    if eval_value is None or key not in eval_value or not isinstance(eval_value[key], list):
        return None
    return eval_value[key]


def _run_h01(context: Mapping[str, Any], thresholds_ok: bool) -> dict[str, Any]:
    missing, failures = [], []
    checks: dict[str, bool] = {"thresholds_frozen": thresholds_ok}
    if not thresholds_ok:
        failures.append("thresholds_unfrozen")
    for key in _REQUIRED_CONTEXT_FLAGS:
        if key == "thresholds_frozen":
            continue
        checks[key] = _flag(context, key, missing, failures)
    return _gate("H01", checks, missing, failures)


def _run_h02(run: Mapping[str, Any], expected_counts: Mapping[str, int]) -> dict[str, Any]:
    data = run.get("data")
    cache = run.get("cache") or run.get("cache_audit")
    missing: list[str] = []
    failures: list[str] = []
    checks: dict[str, bool] = {}
    if not isinstance(data, Mapping):
        missing.append("data")
        data = {}
    if not isinstance(cache, Mapping):
        missing.append("cache")
        cache = {}
    checks["data_passed"] = _flag(data, "passed", missing, failures)
    for key in ("forbidden_field_hits", "forbidden_source_scan", "semantic_overlap", "source_overlap", "split_counts"):
        if key not in data:
            missing.append(f"data.{key}")
    checks["data_forbidden_empty"] = _empty_collection(data.get("forbidden_field_hits")) and _empty_collection(data.get("forbidden_source_scan"))
    if not checks["data_forbidden_empty"]:
        failures.append("data_forbidden_fields")
    checks["semantic_overlap_empty"] = _empty_collection(data.get("semantic_overlap"))
    checks["source_overlap_empty"] = _empty_collection(data.get("source_overlap"))
    if not checks["semantic_overlap_empty"]:
        failures.append("semantic_overlap")
    if not checks["source_overlap_empty"]:
        failures.append("source_overlap")
    split_counts = data.get("split_counts")
    checks["split_counts_exact"] = isinstance(split_counts, Mapping) and dict(split_counts) == dict(expected_counts)
    if not checks["split_counts_exact"]:
        failures.append("split_counts")
    checks["cache_passed"] = _flag(cache, "passed", missing, failures)
    cache_checks = cache.get("checks")
    checks["cache_checks_all"] = (
        isinstance(cache_checks, Mapping)
        and bool(cache_checks)
        and all(isinstance(value, (bool, np.bool_)) and bool(value) for value in cache_checks.values())
    )
    if not checks["cache_checks_all"]:
        failures.append("cache_checks")
    if "forbidden_field_hits" not in cache:
        missing.append("cache.forbidden_field_hits")
    checks["cache_forbidden_empty"] = _empty_collection(cache.get("forbidden_field_hits"))
    if not checks["cache_forbidden_empty"]:
        failures.append("cache_forbidden_fields")
    checks["paired_inputs"] = _flag(run, "paired_inputs_passed", missing, failures)
    return _gate("H02", checks, missing, failures)


def _run_h03(run: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    checks: dict[str, bool] = {}
    checks["architecture"] = _flag(run, "architecture_passed", missing, failures)
    checks["runtime_probes"] = _flag(run, "runtime_probes_passed", missing, failures)
    checks["initialization"] = _flag(run.get("initialization", {}), "passed", missing, failures)
    for arm in _ARMS:
        value = run.get(arm)
        if not isinstance(value, Mapping):
            missing.append(arm)
            continue
        integrity = value.get("integrity")
        if not isinstance(integrity, Mapping):
            missing.append(f"{arm}.integrity")
            continue
        checks[f"{arm}.integrity_passed"] = _flag(integrity, "passed", missing, failures)
        for key in _REQUIRED_INTEGRITY:
            if key not in integrity:
                missing.append(f"{arm}.integrity.{key}")
                checks[f"{arm}.{key}"] = False
            else:
                expected = {
                    "route_granularity": "record",
                    "conditional_transition_scope": (
                        "factorized-routed-state-write-projection"
                    ),
                    "routed_feature_width": MODEL.routed_feature_width,
                    "routed_projection_input_width": MODEL.routed_feature_width,
                    "routed_projection_output_width": MODEL.latent_width,
                }.get(key, True)
                checks[f"{arm}.{key}"] = integrity[key] == expected
                if not checks[f"{arm}.{key}"]:
                    failures.append(f"{arm}.integrity.{key}")
    return _gate("H03", checks, missing, failures)


def _run_h04(run: Mapping[str, Any], thresholds: Mapping[str, float], spec: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    checks: dict[str, bool] = {}
    for arm in _ARMS:
        value = run.get(arm)
        if not isinstance(value, Mapping):
            missing.append(arm)
            continue
        training = value.get("training")
        if not isinstance(training, Mapping):
            missing.append(f"{arm}.training")
            continue
        checks[f"{arm}.updates"] = training.get("updates") == int(spec["updates"])
        checks[f"{arm}.examples_seen"] = training.get("examples_seen") == int(spec["updates"]) * int(spec["batch_size"])
        final_losses = training.get("final_losses")
        history = training.get("history")
        checks[f"{arm}.finite_training"] = (
            isinstance(final_losses, Mapping)
            and bool(final_losses)
            and _finite(final_losses)
            and isinstance(history, list)
            and bool(history)
            and _finite(history)
        )
        for key in (f"{arm}.updates", f"{arm}.examples_seen", f"{arm}.finite_training"):
            if not checks[key]:
                failures.append(key)
        for split, answer_threshold, trace_threshold in (
            ("supported", thresholds["supported_answer_floor"], thresholds["supported_trace_floor"]),
            ("heldout", thresholds[f"heldout_{arm}_answer_floor"], thresholds[f"heldout_{arm}_trace_floor"]),
        ):
            evaluation = _eval(value, split)
            answer = _metric(evaluation, "answer.macro_accuracy")
            trace = _metric(evaluation, "trace.cell_accuracy")
            if answer is None:
                missing.append(f"{arm}.evaluations.{split}.answer.macro_accuracy")
                checks[f"{arm}.{split}.answer_floor"] = False
            else:
                checks[f"{arm}.{split}.answer_floor"] = answer >= answer_threshold
            if trace is None:
                missing.append(f"{arm}.evaluations.{split}.trace.cell_accuracy")
                checks[f"{arm}.{split}.trace_floor"] = False
            else:
                checks[f"{arm}.{split}.trace_floor"] = trace >= trace_threshold
            if not checks[f"{arm}.{split}.answer_floor"]:
                failures.append(f"{arm}.{split}.answer_floor")
            if not checks[f"{arm}.{split}.trace_floor"]:
                failures.append(f"{arm}.{split}.trace_floor")
            if evaluation is not None:
                families = evaluation.get("families")
                if not isinstance(families, list) or set(families) != {"numeric", "relation"}:
                    missing.append(f"{arm}.evaluations.{split}.families")
                    failures.append(f"{arm}.{split}.family_coverage")
                for family in ("numeric", "relation"):
                    family_answer = _metric(
                        evaluation, f"answer.by_family.{family}.accuracy"
                    )
                    family_trace = _metric(
                        evaluation, f"trace.by_family.{family}.cell_accuracy"
                    )
                    answer_key = f"{arm}.{split}.{family}.answer_floor"
                    trace_key = f"{arm}.{split}.{family}.trace_floor"
                    checks[answer_key] = (
                        family_answer is not None and family_answer >= answer_threshold
                    )
                    checks[trace_key] = (
                        family_trace is not None and family_trace >= trace_threshold
                    )
                    if family_answer is None:
                        missing.append(
                            f"{arm}.evaluations.{split}.answer.by_family.{family}.accuracy"
                        )
                    if family_trace is None:
                        missing.append(
                            f"{arm}.evaluations.{split}.trace.by_family.{family}.cell_accuracy"
                        )
                    if not checks[answer_key]:
                        failures.append(answer_key)
                    if not checks[trace_key]:
                        failures.append(trace_key)
    return _gate("H04", checks, missing, failures)


def _run_h05(runs: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float], bootstrap_seed: int) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    all_shared: list[bool] = []
    all_mixed: list[bool] = []
    family_shared: dict[str, list[bool]] = {"numeric": [], "relation": []}
    family_mixed: dict[str, list[bool]] = {"numeric": [], "relation": []}
    by_seed: dict[str, Any] = {}
    for run in runs:
        key = _seed_key(run)
        shared = _eval(run.get("shared", {}), "heldout")
        mixed = _eval(run.get("mixed", {}), "heldout")
        required = (shared, mixed)
        if any(value is None for value in required):
            missing.append(f"{key}.heldout")
            continue
        assert shared is not None and mixed is not None
        if shared.get("ids") != mixed.get("ids") or shared.get("targets") != mixed.get("targets") or shared.get("families") != mixed.get("families"):
            failures.append(f"{key}.paired_heldout_identity")
            continue
        ids = shared.get("ids")
        target = shared.get("targets")
        sp = shared.get("predictions")
        mp = mixed.get("predictions")
        families = shared.get("families")
        if not all(isinstance(value, list) for value in (ids, target, sp, mp, families)):
            missing.append(f"{key}.heldout_vectors")
            continue
        lengths = {len(ids), len(target), len(sp), len(mp), len(families)}
        if len(lengths) != 1 or not ids or len(set(ids)) != len(ids):
            failures.append(f"{key}.heldout_vector_lengths")
            continue
        if any(str(family) not in ("numeric", "relation") for family in families):
            failures.append(f"{key}.heldout_unknown_family")
            continue
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or not 0 <= int(value) <= 3
            for value in (*target, *sp, *mp)
        ):
            failures.append(f"{key}.heldout_label_range")
            continue
        sc = [int(a) == int(b) for a, b in zip(sp, target, strict=True)]
        mc = [int(a) == int(b) for a, b in zip(mp, target, strict=True)]
        for family, s_value, m_value in zip(families, sc, mc, strict=True):
            family_shared[str(family)].append(s_value)
            family_mixed[str(family)].append(m_value)
        all_shared.extend(sc)
        all_mixed.extend(mc)
        seed_result = paired_bootstrap_gain(sc, mc, seed=bootstrap_seed + len(by_seed))
        by_seed[key] = seed_result
        if not seed_result["gain"] > 0:
            failures.append(f"{key}.gain_not_positive")
    if not all_shared:
        missing.append("pooled_heldout_vectors")
        return _gate("H05", {"pooled_vectors": False}, missing, failures)
    pooled = paired_bootstrap_gain(all_shared, all_mixed, seed=bootstrap_seed)
    by_family = {
        family: paired_bootstrap_gain(family_shared[family], family_mixed[family], seed=bootstrap_seed + index + 1)
        if family_shared[family] else None
        for index, family in enumerate(("numeric", "relation"))
    }
    for family, result in by_family.items():
        if result is None:
            missing.append(f"pooled.{family}")
        elif result["gain"] < -thresholds["family_regression_limit"]:
            failures.append(f"pooled.{family}.regression")
    checks = {
        "pooled_gain": pooled["gain"] >= thresholds["primary_gain"],
        "pooled_ci_lower": pooled["ci95"]["lower"] > 0,
        "three_seed_positive": len(by_seed) == 3 and all(value["gain"] > 0 for value in by_seed.values()),
        "family_regression": all(value is not None and value["gain"] >= -thresholds["family_regression_limit"] for value in by_family.values()),
    }
    for key, value in checks.items():
        if not value:
            failures.append(key)
    gate = _gate("H05", checks, missing, failures)
    gate["by_seed"] = by_seed
    gate["pooled"] = pooled
    gate["by_family"] = by_family
    return gate


def _run_h06(run: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    checks: dict[str, bool] = {}
    shared_noop = run.get("shared_noop")
    checks["shared_noop"] = isinstance(shared_noop, Mapping) and shared_noop.get("passed") is True
    if not checks["shared_noop"]:
        missing.append("shared_noop.passed") if not isinstance(shared_noop, Mapping) else failures.append("shared_noop")
    mixed = run.get("mixed")
    normal = _eval(mixed if isinstance(mixed, Mapping) else {}, "heldout")
    diagnostics = _metric_diagnostics(normal)
    checks["route_diagnostics"] = diagnostics is not None
    if diagnostics is None:
        missing.append("mixed.evaluations.heldout.route.diagnostics")
    else:
        checks["both_experts_each_cell"] = diagnostics.get("all_cells_use_both_experts") is True
        minimum_load = diagnostics.get("minimum_expert_load")
        minimum_entropy = diagnostics.get("minimum_entropy_bits")
        checks["route_load_finite"] = (
            isinstance(minimum_load, Real)
            and not isinstance(minimum_load, (bool, np.bool_))
            and math.isfinite(float(minimum_load))
            and float(minimum_load) > 0.0
            and isinstance(minimum_entropy, Real)
            and not isinstance(minimum_entropy, (bool, np.bool_))
            and math.isfinite(float(minimum_entropy))
            and float(minimum_entropy) > 0.0
        )
        checks["route_accuracy"] = _metric(normal, "route.accuracy") is not None and _metric(normal, "route.accuracy") >= thresholds["route_accuracy"]
        if not checks["both_experts_each_cell"]:
            failures.append("route_collapse")
        if not checks["route_accuracy"]:
            failures.append("route_accuracy")
        if not checks["route_load_finite"]:
            failures.append("route_load_nonfinite_or_collapsed")
    drops = run.get("mixed_causal_drops")
    if not isinstance(drops, Mapping):
        missing.append("mixed_causal_drops")
        checks["causal_drop"] = False
    else:
        values = []
        for name in _REQUIRED_ROUTE_INTERVENTIONS:
            value = drops.get(name)
            if not isinstance(value, Mapping):
                missing.append(f"mixed_causal_drops.{name}")
                continue
            answer = value.get("answer")
            trace = value.get("trace")
            if not all(isinstance(item, Real) and math.isfinite(float(item)) for item in (answer, trace)):
                failures.append(f"mixed_causal_drops.{name}:nonfinite")
                continue
            values.append(max(float(answer), float(trace)))
        checks["route_causal_drop"] = bool(values) and max(values) >= thresholds["route_causal_drop"]
        if not checks["route_causal_drop"]:
            failures.append("route_causal_drop")
        conditional = drops.get("disable_routed_projection")
        if not isinstance(conditional, Mapping):
            missing.append("mixed_causal_drops.disable_routed_projection")
            checks["conditional_write_causal_drop"] = False
        else:
            answer = conditional.get("answer")
            trace = conditional.get("trace")
            valid = all(
                isinstance(item, Real)
                and not isinstance(item, (bool, np.bool_))
                and math.isfinite(float(item))
                for item in (answer, trace)
            )
            checks["conditional_write_causal_drop"] = valid and max(
                float(answer), float(trace)
            ) >= thresholds["conditional_write_causal_drop"]
            if not valid:
                failures.append(
                    "mixed_causal_drops.disable_routed_projection:nonfinite"
                )
        if not checks["conditional_write_causal_drop"]:
            failures.append("conditional_write_causal_drop")
    route_metamorphic = run.get("route_metamorphic")
    if not isinstance(route_metamorphic, Mapping):
        missing.append("route_metamorphic")
        checks["route_metamorphic"] = False
    else:
        aggregate = route_metamorphic.get("aggregate")
        by_transform = route_metamorphic.get("by_transform")
        by_family = route_metamorphic.get("by_family")

        def route_floor(value: Any) -> bool:
            return (
                isinstance(value, Mapping)
                and all(
                    isinstance(value.get(key), Real)
                    and not isinstance(value.get(key), (bool, np.bool_))
                    and math.isfinite(float(value[key]))
                    and float(value[key]) >= thresholds["route_accuracy"]
                    for key in ("consistency", "transformed_exact")
                )
            )

        checks["route_metamorphic_aggregate"] = route_floor(aggregate)
        checks["route_metamorphic_transforms"] = (
            isinstance(by_transform, Mapping)
            and set(by_transform) == set(_REQUIRED_METAMORPHIC_TRANSFORMS)
            and all(route_floor(by_transform[name]) for name in _REQUIRED_METAMORPHIC_TRANSFORMS)
        )
        checks["route_metamorphic_families"] = (
            isinstance(by_family, Mapping)
            and set(by_family) == {"numeric", "relation"}
            and all(route_floor(by_family[name]) for name in ("numeric", "relation"))
        )
        checks["route_metamorphic"] = all(
            checks[name]
            for name in (
                "route_metamorphic_aggregate",
                "route_metamorphic_transforms",
                "route_metamorphic_families",
            )
        )
        if not checks["route_metamorphic"]:
            failures.append("route_metamorphic_consistency")
    return _gate("H06", checks, missing, failures)


def _metric_diagnostics(evaluation: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if evaluation is None or not isinstance(evaluation.get("route"), Mapping):
        return None
    value = evaluation["route"].get("diagnostics")
    return value if isinstance(value, Mapping) else None


def _run_h07(run: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    checks: dict[str, bool] = {}
    source = run.get("source_causal_drops")
    recurrence = run.get("recurrence_causal_drop")
    if not isinstance(source, Mapping):
        missing.append("source_causal_drops")
        checks["source_causal"] = False
    else:
        source_checks = []
        for intervention in ("zero_source", "shuffle_source"):
            values = [
                source.get(f"{intervention}.answer"),
                source.get(f"{intervention}.trace"),
            ]
            valid = all(
                isinstance(item, Real)
                and not isinstance(item, (bool, np.bool_))
                and math.isfinite(float(item))
                for item in values
            )
            passed = valid and max(float(item) for item in values) >= thresholds["source_causal_drop"]
            checks[f"source_{intervention}"] = passed
            source_checks.append(passed)
            if not valid:
                missing.append(f"source_causal_drops.{intervention}.answer/trace")
        checks["source_causal"] = all(source_checks)
        if not checks["source_causal"]:
            failures.append("source_causal_drop")
    if not isinstance(recurrence, Mapping):
        missing.append("recurrence_causal_drop")
        checks["recurrence_causal"] = False
    else:
        recurrence_values = [
            recurrence.get("disable_recurrence.answer"),
            recurrence.get("disable_recurrence.trace"),
        ]
        valid = all(
            isinstance(item, Real)
            and not isinstance(item, (bool, np.bool_))
            and math.isfinite(float(item))
            for item in recurrence_values
        )
        checks["recurrence_causal"] = valid and max(
            float(item) for item in recurrence_values
        ) >= thresholds["recurrence_causal_drop"]
        if not valid:
            missing.append("recurrence_causal_drop.disable_recurrence.answer/trace")
        if not checks["recurrence_causal"]:
            failures.append("recurrence_causal_drop")
    metamorphic = run.get("metamorphic")
    if not isinstance(metamorphic, Mapping):
        missing.append("metamorphic")
        checks["metamorphic"] = False
    else:
        aggregate = metamorphic.get("aggregate")
        by_transform = metamorphic.get("by_transform")
        by_family = metamorphic.get("by_family")

        def answer_floor(value: Any) -> bool:
            return (
                isinstance(value, Mapping)
                and all(
                    isinstance(value.get(key), Real)
                    and not isinstance(value.get(key), (bool, np.bool_))
                    and math.isfinite(float(value[key]))
                    and float(value[key]) >= thresholds["metamorphic_consistency"]
                    for key in ("relation_consistency", "transformed_exact")
                )
            )

        checks["metamorphic_aggregate"] = answer_floor(aggregate)
        checks["metamorphic_transforms"] = (
            isinstance(by_transform, Mapping)
            and set(by_transform) == set(_REQUIRED_METAMORPHIC_TRANSFORMS)
            and all(answer_floor(by_transform[name]) for name in _REQUIRED_METAMORPHIC_TRANSFORMS)
        )
        checks["metamorphic_families"] = (
            isinstance(by_family, Mapping)
            and set(by_family) == {"numeric", "relation"}
            and all(answer_floor(by_family[name]) for name in ("numeric", "relation"))
        )
        checks["metamorphic"] = all(
            checks[name]
            for name in (
                "metamorphic_aggregate",
                "metamorphic_transforms",
                "metamorphic_families",
            )
        )
        if not checks["metamorphic"]:
            failures.append("metamorphic_consistency")
    return _gate("H07", checks, missing, failures)


def _run_h08(run: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    failures: list[str] = []
    checks: dict[str, bool] = {}
    budget = run.get("budget")
    if not isinstance(budget, Mapping):
        missing.append("budget")
        budget = {}
    for key, expected in (
        ("updates", int(spec["updates"])),
        ("batch_size", int(spec["batch_size"])),
        ("examples_seen", int(spec["updates"]) * int(spec["batch_size"])),
    ):
        checks[key] = budget.get(key) == expected
        if not checks[key]:
            failures.append(f"budget.{key}")
    schedule = run.get("schedule")
    checks["schedule"] = isinstance(schedule, Mapping) and schedule.get("passed") is True
    checks["schedule_shared_mixed"] = isinstance(schedule, Mapping) and schedule.get("shared_sha256") == schedule.get("mixed_sha256")
    if not checks["schedule"]:
        failures.append("schedule")
    if not checks["schedule_shared_mixed"]:
        failures.append("schedule_shared_mixed")
    flops = run.get("active_flops")
    checks["active_flops"] = isinstance(flops, Mapping) and all(isinstance(flops.get(key), Real) and math.isfinite(float(flops[key])) for key in ("shared", "mixed"))
    if checks["active_flops"]:
        left, right = float(flops["shared"]), float(flops["mixed"])
        checks["active_flops_matched"] = abs(left - right) <= max(1e-9, 0.02 * max(left, right, 1.0))
    else:
        checks["active_flops_matched"] = False
        missing.append("active_flops.shared/mixed")
    if not checks["active_flops_matched"]:
        failures.append("active_flops_matched")
    for arm in _ARMS:
        value = run.get(arm)
        strip = value.get("strip") if isinstance(value, Mapping) else None
        checks[f"{arm}.strip_reload"] = isinstance(strip, Mapping) and strip.get("passed") is True and strip.get("reload_predictions_equal") is True
        if not checks[f"{arm}.strip_reload"]:
            failures.append(f"{arm}.strip_reload")
    runner = run.get("runner")
    if not isinstance(runner, Mapping):
        missing.append("runner")
        runner = {}
    for key in _REQUIRED_RUNNER:
        if key not in runner:
            missing.append(f"runner.{key}")
            checks[f"runner.{key}"] = False
        else:
            checks[f"runner.{key}"] = runner[key] is True
        if not checks[f"runner.{key}"]:
            failures.append(f"runner.{key}")
    for key in ("updates_per_second", "examples_per_second", "dispatch_overhead"):
        runtime_value = run.get("runtime", {}).get(key) if isinstance(run.get("runtime"), Mapping) else None
        checks[f"runtime.{key}"] = (
            isinstance(runtime_value, Real)
            and not isinstance(runtime_value, (bool, np.bool_))
            and math.isfinite(float(runtime_value))
            and (
                float(runtime_value) > 0.0
                if key != "dispatch_overhead"
                else float(runtime_value) >= 0.0
            )
        )
        if not checks[f"runtime.{key}"]:
            missing.append(f"runtime.{key}")
    return _gate("H08", checks, missing, failures)


def aggregate_h1_qualification(
    runs: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any] | None = None,
    thresholds: Any = THRESHOLDS,
    expected_pairs: Sequence[tuple[int, int]] | None = None,
    split_counts: Mapping[str, int] = SPLIT_COUNTS,
    training_spec: Mapping[str, Any] | None = None,
    bootstrap_seed: int = 2026081800,
) -> dict[str, Any]:
    """Aggregate H01--H08 without performing any external side effect."""
    context = dict(context or {})
    expected_pairs = tuple(expected_pairs or zip(DATA_SEEDS, MODEL_SEEDS, strict=True))
    spec = dict(training_spec or {"updates": 4000, "batch_size": 32})
    threshold_values, threshold_errors = _thresholds(thresholds)
    spec_errors = [f"training_spec.{key}" for key in ("updates", "batch_size") if key not in spec]
    run_missing, run_failures, indexed = _validate_runs(runs, expected_pairs)
    h01 = _run_h01(context, not threshold_errors)
    h01["missing"].extend(run_missing)
    h01["failures"].extend(run_failures)
    h01["passed"] = h01["passed"] and not run_missing and not run_failures
    gates: dict[str, Any] = {"H01": h01}
    per_gate_runs: dict[str, dict[str, Any]] = {name: {} for name in ("H02", "H03", "H04", "H06", "H07", "H08")}
    blocked_training_gates = list(spec_errors)
    if threshold_errors:
        blocked_training_gates.append("thresholds_unfrozen")
    for key in sorted(set(indexed) | {f"{data}/{model}" for data, model in expected_pairs}):
        run = indexed.get(key)
        if run is None:
            for name in per_gate_runs:
                per_gate_runs[name][key] = _gate(name, {}, ["missing_run"])
            continue
        per_gate_runs["H02"][key] = _run_h02(run, split_counts)
        per_gate_runs["H03"][key] = _run_h03(run)
        per_gate_runs["H04"][key] = (
            _run_h04(run, threshold_values, spec)
            if not threshold_errors and not spec_errors
            else _gate("H04", {}, blocked_training_gates)
        )
        per_gate_runs["H06"][key] = _run_h06(run, threshold_values) if not threshold_errors else _gate("H06", {}, ["thresholds_unfrozen"])
        per_gate_runs["H07"][key] = _run_h07(run, threshold_values) if not threshold_errors else _gate("H07", {}, ["thresholds_unfrozen"])
        per_gate_runs["H08"][key] = (
            _run_h08(run, spec)
            if not spec_errors
            else _gate("H08", {}, spec_errors)
        )
    for name, values in per_gate_runs.items():
        aggregate = all(value["passed"] for value in values.values()) and len(values) == len(expected_pairs)
        gates[name] = {"name": name, "passed": aggregate, "by_seed": values, "aggregate": aggregate}
        if not aggregate:
            gates[name]["failures"] = [key for key, value in values.items() if not value["passed"]]
    h05 = _run_h05([indexed[key] for key in sorted(indexed)], threshold_values, bootstrap_seed) if not threshold_errors else _gate("H05", {}, ["thresholds_unfrozen"])
    gates["H05"] = h05
    all_passed = all(gate.get("passed") is True for gate in gates.values()) and not run_missing and not run_failures and not threshold_errors
    status = "PASS_P1_H1_MIXED_CORE_DEVELOPMENT" if all_passed else "FAIL_P1_H1_MIXED_CORE_DEVELOPMENT"
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.qualification.v1",
        "status": status,
        "passed": all_passed,
        "claims": {
            "p1_f1_design_authorized": all_passed,
            "p1_completed": False,
            "p2_eligible": False,
            "p2_started": False,
        },
        "seed_pairs": [list(pair) for pair in expected_pairs],
        "threshold_errors": threshold_errors,
        "training_spec_errors": spec_errors,
        "gates": gates,
        "runner_required_fields": list(_REQUIRED_RUNNER) + ["runtime.updates_per_second", "runtime.examples_per_second", "runtime.dispatch_overhead"],
    }


qualify_h1 = aggregate_h1_qualification
aggregate_qualification = aggregate_h1_qualification
compute_qualification = aggregate_h1_qualification


__all__ = [
    "aggregate_h1_qualification",
    "aggregate_qualification",
    "compute_qualification",
    "qualify_h1",
]
