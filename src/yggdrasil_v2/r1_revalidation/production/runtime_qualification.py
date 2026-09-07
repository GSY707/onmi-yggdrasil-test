from __future__ import annotations

"""Stable runtime qualification for the P0-D v16 production path."""

from copy import deepcopy
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from ..learner.models import analyze
from .g09_qualification import V14_ROOT, verify_v14_sealed_input, _label, _sample_jsonl
from .generator import MODEL_ID, MODEL_REVISION, load_production_dataset
from .generator_audit import _audit_claims, _audit_shortcuts, _flatten, _NB_SPECS
from .p0_shortcut import (
    fit_prepared_multinomial_nb,
    predict_prepared_multinomial_nb_batch,
    predict_prepared_multinomial_nb_exact,
    prepare_nb_rows,
)
from .renderer import LABELS


BENCHMARK_SEED = 2026081601
BENCHMARK_REPETITIONS = 15
BENCHMARK_PREDICTIONS = 192
MINIMUM_MEDIAN_SPEEDUP = 5.0
MINIMUM_FAST_WINS = 14
MAXIMUM_FAST_P95_SECONDS = 0.20
MAXIMUM_FALLBACK_RATE = 0.01
MAXIMUM_OPERATIONAL_SECONDS = 240.0
MAXIMUM_PEAK_WORKING_SET_BYTES = int(4.5 * 1024**3)

V15_ROOT = Path("artifacts/v2-r1r/p0d-v15-g09-qualification-20260810-1")
V15_FORMAL_ROOT = Path("artifacts/v2-r1r/p0d-v15-full-production-20260810-1")
V15_SEAL_SHA256 = "997166EECC490F7B6AF6E80C01E28E187539BEDA351764597F0D79DE42C1EE38"
EXPECTED_V15_GATES = {
    "Q01": True,
    "Q02": True,
    "Q03": True,
    "Q04": True,
    "Q05": True,
    "Q06": True,
    "Q07": True,
    "Q08": False,
    "Q09": True,
}
EXPECTED_V14_FAILED_SOURCE_CELLS = (
    "CPS/horizon_ood/full_text_char_3_5_nb",
    "ERE/composition_ood/full_text_char_3_5_nb",
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "evidence-seal.json"
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def verify_v15_component_evidence(repo_root: Path) -> dict[str, Any]:
    root = repo_root / V15_ROOT
    required = (
        "assessment.json",
        "decision-power-report.json",
        "fault-report.json",
        "scorer-report.json",
        "performance-report.json",
        "replay-ledger.json",
        "evidence-seal.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {
            "root": V15_ROOT.as_posix(),
            "missing": missing,
            "passed": False,
            "failures": ["missing_files"],
        }

    assessment = _load_json(root / "assessment.json")
    power = _load_json(root / "decision-power-report.json")
    faults = _load_json(root / "fault-report.json")
    scorer = _load_json(root / "scorer-report.json")
    performance = _load_json(root / "performance-report.json")
    replay = _load_json(root / "replay-ledger.json")
    seal = _load_json(root / "evidence-seal.json")
    actual_files = _tree_hashes(root)

    checks = {
        "seal_file_sha256": _sha(root / "evidence-seal.json") == V15_SEAL_SHA256,
        "seal_tree_replay": actual_files == seal.get("files"),
        "assessment_status": assessment.get("status") == "FAIL_G09_QUALIFICATION",
        "assessment_passed_false": assessment.get("passed") is False,
        "assessment_gate_vector": assessment.get("gates") == EXPECTED_V15_GATES,
        "fresh_production_not_authorized": assessment.get("fresh_production_authorized") is False,
        "v15_formal_root_absent": not (repo_root / V15_FORMAL_ROOT).exists(),
        "decision_power_passed": power.get("passed") is True,
        "decision_faults_passed": faults.get("passed") is True,
        "registry_passed": scorer.get("registry", {}).get("passed") is True,
        "synthetic_passed": scorer.get("synthetic", {}).get("passed") is True,
        "ties_passed": scorer.get("ties", {}).get("passed") is True,
        "streaming_passed": scorer.get("streaming", {}).get("passed") is True,
        "performance_report_identity": performance == scorer.get("performance"),
        "old_prediction_identity": performance.get("prediction_identity") is True,
        "old_fallback_qualified": (
            isinstance(performance.get("exact_fallback_rate"), (int, float))
            and performance["exact_fallback_rate"] <= MAXIMUM_FALLBACK_RATE
        ),
        "old_single_point_speed_failure": (
            performance.get("passed") is False
            and isinstance(performance.get("speedup"), (int, float))
            and performance["speedup"] < performance.get("minimum_speedup") == 5.0
        ),
        "replay_passed": replay.get("passed") is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v16.v15-component-reference.v1",
        "root": V15_ROOT.as_posix(),
        "evidence_seal_sha256": _sha(root / "evidence-seal.json"),
        "sealed_file_count": len(actual_files),
        "assessment_sha256": _sha(root / "assessment.json"),
        "scorer_projection_sha256": scorer.get("deterministic_projection_sha256"),
        "old_performance": performance,
        "checks": checks,
        "failures": failures,
        "passed": not failures,
    }


def _nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("p95 requires at least one value")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _benchmark_fixture(repo_root: Path) -> tuple[Mapping[str, Any], list[Mapping[str, int]], list[Sequence[bool]]]:
    dataset = repo_root / V14_ROOT / "dataset"
    train = _sample_jsonl(dataset / "ere-train.jsonl", 512, BENCHMARK_SEED)
    evaluation: list[dict[str, Any]] = []
    heldout = ("validation", "composition_ood", "length_ood", "entity_ood", "language_ood", "causal_pairs")
    for split_index, split in enumerate(heldout, start=1):
        evaluation.extend(_sample_jsonl(dataset / f"ere-{split}.jsonl", 32, BENCHMARK_SEED + split_index))
    prepared = prepare_nb_rows(
        train,
        analyzer="char_3_5",
        row_id=lambda row: row["example_id"],
        text=lambda row: row["source_text"],
        label=_label,
    )
    model = fit_prepared_multinomial_nb(prepared, LABELS)
    features = [analyze(str(row["source_text"]), "char_3_5") for row in evaluation]
    masks = [row["valid_choice_mask"] for row in evaluation]
    if len(features) != BENCHMARK_PREDICTIONS:
        raise AssertionError(f"benchmark size changed: {len(features)}")
    return model, features, masks


def run_paired_benchmark(repo_root: Path) -> dict[str, Any]:
    verify_v14_sealed_input(repo_root)
    model, features, masks = _benchmark_fixture(repo_root)

    def exact() -> list[str]:
        return [
            predict_prepared_multinomial_nb_exact(model, row, mask)
            for row, mask in zip(features, masks, strict=True)
        ]

    def fast() -> tuple[list[str], int]:
        results = predict_prepared_multinomial_nb_batch(model, features, masks)
        return [str(row["prediction"]) for row in results], sum(bool(row["exact_fallback"]) for row in results)

    reference = exact()
    first_fast, fallback_count = fast()
    if reference != first_fast:
        raise AssertionError("benchmark warmup prediction mismatch")
    for _ in range(2):
        fast()
        exact()

    repetitions: list[dict[str, Any]] = []
    gc.collect()
    gc.disable()
    try:
        for index in range(BENCHMARK_REPETITIONS):
            order = ("exact", "fast") if index % 2 == 0 else ("fast", "exact")
            durations: dict[str, float] = {}
            predictions: dict[str, list[str]] = {}
            for name in order:
                started = time.perf_counter()
                if name == "exact":
                    predictions[name] = exact()
                else:
                    predictions[name] = fast()[0]
                durations[name] = time.perf_counter() - started
            identity = predictions["exact"] == predictions["fast"] == reference
            repetitions.append(
                {
                    "index": index,
                    "order": list(order),
                    "legacy_exact_seconds": durations["exact"],
                    "fast_seconds": durations["fast"],
                    "paired_speedup": durations["exact"] / max(durations["fast"], 1e-12),
                    "prediction_identity": identity,
                }
            )
    finally:
        gc.enable()

    fast_seconds = [row["fast_seconds"] for row in repetitions]
    speedups = [row["paired_speedup"] for row in repetitions]
    fast_wins = sum(row["fast_seconds"] < row["legacy_exact_seconds"] for row in repetitions)
    report = {
        "schema_version": "yggdrasil.v2-r1r.p0d-v16.paired-benchmark.v1",
        "seed": BENCHMARK_SEED,
        "prediction_count": len(reference),
        "repetition_count": len(repetitions),
        "warmup_count_per_path": 2,
        "alternating_order": True,
        "prediction_sha256": hashlib.sha256(_canonical(reference)).hexdigest().upper(),
        "prediction_identity": all(row["prediction_identity"] for row in repetitions),
        "exact_fallback_count": fallback_count,
        "exact_fallback_rate": fallback_count / len(reference),
        "median_paired_speedup": statistics.median(speedups),
        "fast_win_count": fast_wins,
        "fast_p95_seconds": _nearest_rank_p95(fast_seconds),
        "thresholds": {
            "minimum_median_speedup": MINIMUM_MEDIAN_SPEEDUP,
            "minimum_fast_wins": MINIMUM_FAST_WINS,
            "maximum_fast_p95_seconds": MAXIMUM_FAST_P95_SECONDS,
            "maximum_fallback_rate": MAXIMUM_FALLBACK_RATE,
        },
        "repetitions": repetitions,
    }
    return report


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory() -> dict[str, int]:
    if not hasattr(ctypes, "WinDLL"):
        raise RuntimeError("v16 runtime qualification is frozen to Windows GetProcessMemoryInfo")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "working_set_bytes": int(counters.WorkingSetSize),
        "private_usage_bytes": int(counters.PrivateUsage),
    }


def _compare_cells(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    fields: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    if set(observed) != set(expected):
        failures.append("cell_keys")
        return failures
    for key in sorted(observed):
        for field in fields:
            if observed[key].get(field) != expected[key].get(field):
                failures.append(f"{key}:{field}")
    return failures


def run_operational_qualification(
    repo_root: Path,
    *,
    tokenizer: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    v14_before = verify_v14_sealed_input(repo_root)
    root = repo_root / V14_ROOT
    started = time.perf_counter()
    load_started = time.perf_counter()
    records = _flatten(load_production_dataset(root / "dataset"))
    load_seconds = time.perf_counter() - load_started
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)

    events: list[str] = []
    emit = progress or (lambda _: None)

    def on_progress(event: str) -> None:
        events.append(event)
        emit(event)

    g09_started = time.perf_counter()
    source = _audit_shortcuts(records, tokenizer, on_progress)
    g09_seconds = time.perf_counter() - g09_started
    g10_started = time.perf_counter()
    claims = _audit_claims(records)
    g10_seconds = time.perf_counter() - g10_started
    total_seconds = time.perf_counter() - started
    memory = process_memory()

    expected_source = _load_json(root / "shortcut-report.json")
    expected_claims = _load_json(root / "claim-report.json")
    source_value = source["value"]
    expected_source_value = expected_source["value"]
    source_projection_failures = _compare_cells(
        source_value["cells"],
        expected_source_value["cells"],
        fields=("successes", "total", "prediction_sha256", "passed"),
    )
    source_projection_failures.extend(
        f"aggregate:{failure}"
        for failure in _compare_cells(
            source_value["family_aggregates"],
            expected_source_value["family_aggregates"],
            fields=("successes", "total", "prediction_sha256", "passed"),
        )
    )
    claim_projection_failures = _compare_cells(
        claims["value"]["cells"],
        expected_claims["value"]["cells"],
        fields=("successes", "total", "prediction_sha256", "passed"),
    )
    if claims["value"]["balance"] != expected_claims["value"]["balance"]:
        claim_projection_failures.append("balance")

    failed_source_cells = tuple(sorted(key for key, row in source_value["cells"].items() if not row["passed"]))
    equivalence = source_value["equivalence"]
    equivalence_passed = (
        set(equivalence) == {"ERE", "CPS"}
        and all(row["sparse_dense_equivalence"]["passed"] for row in equivalence.values())
    )
    peak_feature_bank_count = max(row["peak_feature_bank_count"] for row in equivalence.values())
    batch_prediction_count = sum(row["scorer_diagnostics"]["prediction_count"] for row in equivalence.values())
    expected_events = [
        f"G09:{family}:{name}:{phase}"
        for family in ("ERE", "CPS")
        for name, _, _ in _NB_SPECS
        for phase in ("prepare", "done")
    ]
    v14_after = verify_v14_sealed_input(repo_root)
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v16.operational-qualification.v1",
        "input": v14_before,
        "record_count": len(records),
        "timing": {
            "load_seconds": load_seconds,
            "g09_seconds": g09_seconds,
            "g10_seconds": g10_seconds,
            "total_seconds": total_seconds,
            "maximum_total_seconds": MAXIMUM_OPERATIONAL_SECONDS,
        },
        "memory": {**memory, "maximum_peak_working_set_bytes": MAXIMUM_PEAK_WORKING_SET_BYTES},
        "source": {
            "cell_count": len(source_value["cells"]),
            "aggregate_count": len(source_value["family_aggregates"]),
            "projection_identity": not source_projection_failures,
            "projection_failures": source_projection_failures[:50],
            "failed_source_cells": list(failed_source_cells),
            "expected_failed_source_cells": list(EXPECTED_V14_FAILED_SOURCE_CELLS),
            "all_aggregates_passed": all(row["passed"] for row in source_value["family_aggregates"].values()),
            "equivalence_passed": equivalence_passed,
            "metric": source,
        },
        "claims": {
            "cell_count": len(claims["value"]["cells"]),
            "projection_identity": not claim_projection_failures,
            "projection_failures": claim_projection_failures[:50],
            "metric_passed": claims["passed"] is True,
            "metric": claims,
        },
        "streaming": {
            "peak_feature_bank_count": peak_feature_bank_count,
            "batch_prediction_count": batch_prediction_count,
            "progress_event_count": len(events),
            "progress_events": events,
            "expected_progress_events": expected_events,
            "progress_identity": events == expected_events,
        },
        "v14_input_read_only": v14_before == v14_after,
    }


def assess_runtime_inputs(
    prior: Mapping[str, Any],
    paired: Mapping[str, Any],
    operational: Mapping[str, Any],
) -> dict[str, bool]:
    source = operational.get("source", {})
    claims = operational.get("claims", {})
    timing = operational.get("timing", {})
    memory = operational.get("memory", {})
    streaming = operational.get("streaming", {})
    return {
        "Q01": prior.get("passed") is True,
        "Q02": (
            paired.get("prediction_count") == BENCHMARK_PREDICTIONS
            and paired.get("repetition_count") == BENCHMARK_REPETITIONS
            and paired.get("prediction_identity") is True
            and isinstance(paired.get("exact_fallback_rate"), (int, float))
            and paired["exact_fallback_rate"] <= MAXIMUM_FALLBACK_RATE
        ),
        "Q03": (
            isinstance(paired.get("median_paired_speedup"), (int, float))
            and paired["median_paired_speedup"] >= MINIMUM_MEDIAN_SPEEDUP
            and isinstance(paired.get("fast_win_count"), int)
            and not isinstance(paired.get("fast_win_count"), bool)
            and paired["fast_win_count"] >= MINIMUM_FAST_WINS
            and isinstance(paired.get("fast_p95_seconds"), (int, float))
            and paired["fast_p95_seconds"] <= MAXIMUM_FAST_P95_SECONDS
        ),
        "Q04": (
            source.get("cell_count") == 98
            and source.get("aggregate_count") == 14
            and source.get("projection_identity") is True
            and tuple(source.get("failed_source_cells", ())) == EXPECTED_V14_FAILED_SOURCE_CELLS
            and source.get("all_aggregates_passed") is True
            and source.get("equivalence_passed") is True
        ),
        "Q05": (
            claims.get("cell_count") == 24
            and claims.get("projection_identity") is True
            and claims.get("metric_passed") is True
        ),
        "Q06": (
            isinstance(timing.get("total_seconds"), (int, float))
            and timing["total_seconds"] <= MAXIMUM_OPERATIONAL_SECONDS
            and isinstance(memory.get("peak_working_set_bytes"), int)
            and not isinstance(memory.get("peak_working_set_bytes"), bool)
            and memory["peak_working_set_bytes"] <= MAXIMUM_PEAK_WORKING_SET_BYTES
        ),
        "Q07": (
            streaming.get("peak_feature_bank_count") == 1
            and isinstance(streaming.get("batch_prediction_count"), int)
            and not isinstance(streaming.get("batch_prediction_count"), bool)
            and streaming["batch_prediction_count"] > 0
            and streaming.get("progress_event_count") == 12
            and streaming.get("progress_identity") is True
        ),
    }


def run_runtime_faults(
    prior: Mapping[str, Any],
    paired: Mapping[str, Any],
    operational: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = assess_runtime_inputs(prior, paired, operational)
    if not all(baseline.values()):
        raise ValueError(f"fault matrix requires a passing baseline: {baseline}")

    cases: list[tuple[str, str, Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None]]] = [
        ("prior_reference_false", "Q01", lambda p, b, o: p.__setitem__("passed", False)),
        ("prediction_identity_false", "Q02", lambda p, b, o: b.__setitem__("prediction_identity", False)),
        ("fallback_above_limit", "Q02", lambda p, b, o: b.__setitem__("exact_fallback_rate", 0.02)),
        ("median_speedup_below_limit", "Q03", lambda p, b, o: b.__setitem__("median_paired_speedup", 4.99)),
        ("fast_wins_below_limit", "Q03", lambda p, b, o: b.__setitem__("fast_win_count", 13)),
        ("fast_p95_above_limit", "Q03", lambda p, b, o: b.__setitem__("fast_p95_seconds", 0.201)),
        ("source_cell_missing", "Q04", lambda p, b, o: o["source"].__setitem__("cell_count", 97)),
        ("source_projection_changed", "Q04", lambda p, b, o: o["source"].__setitem__("projection_identity", False)),
        ("source_failed_set_changed", "Q04", lambda p, b, o: o["source"].__setitem__("failed_source_cells", [])),
        ("source_equivalence_false", "Q04", lambda p, b, o: o["source"].__setitem__("equivalence_passed", False)),
        ("claim_cell_missing", "Q05", lambda p, b, o: o["claims"].__setitem__("cell_count", 23)),
        ("claim_projection_changed", "Q05", lambda p, b, o: o["claims"].__setitem__("projection_identity", False)),
        ("claim_metric_false", "Q05", lambda p, b, o: o["claims"].__setitem__("metric_passed", False)),
        ("operational_timeout", "Q06", lambda p, b, o: o["timing"].__setitem__("total_seconds", 240.001)),
        ("peak_memory_above_limit", "Q06", lambda p, b, o: o["memory"].__setitem__("peak_working_set_bytes", MAXIMUM_PEAK_WORKING_SET_BYTES + 1)),
        ("two_feature_banks", "Q07", lambda p, b, o: o["streaming"].__setitem__("peak_feature_bank_count", 2)),
        ("missing_progress_event", "Q07", lambda p, b, o: o["streaming"].__setitem__("progress_event_count", 11)),
        ("zero_batch_predictions", "Q07", lambda p, b, o: o["streaming"].__setitem__("batch_prediction_count", 0)),
    ]
    results: dict[str, Any] = {}
    for name, expected_gate, mutate in cases:
        candidate_prior = deepcopy(prior)
        candidate_paired = deepcopy(paired)
        candidate_operational = deepcopy(operational)
        mutate(candidate_prior, candidate_paired, candidate_operational)
        gates = assess_runtime_inputs(candidate_prior, candidate_paired, candidate_operational)
        detected = gates.get(expected_gate) is False
        results[name] = {"expected_gate": expected_gate, "observed_gates": gates, "detected": detected}
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v16.runtime-faults.v1",
        "baseline_gates": baseline,
        "case_count": len(results),
        "cases": results,
        "passed": all(row["detected"] for row in results.values()),
    }


__all__ = [
    "BENCHMARK_PREDICTIONS",
    "BENCHMARK_REPETITIONS",
    "BENCHMARK_SEED",
    "EXPECTED_V14_FAILED_SOURCE_CELLS",
    "MAXIMUM_OPERATIONAL_SECONDS",
    "MAXIMUM_PEAK_WORKING_SET_BYTES",
    "assess_runtime_inputs",
    "process_memory",
    "run_operational_qualification",
    "run_paired_benchmark",
    "run_runtime_faults",
    "verify_v15_component_evidence",
]
