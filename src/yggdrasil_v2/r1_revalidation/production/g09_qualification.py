from __future__ import annotations

"""Synthetic decision-power and exact-scorer qualification for P0-D v15."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..learner.models import analyze
from .g09_decision import (
    BASELINES,
    DEFAULT_CONFIG,
    DecisionConfig,
    FAMILY_SPLITS,
    SOURCE_CELL_COUNT,
    SOURCE_FAMILY_CELL_COUNT,
    evaluate_g09_counts,
    source_cell_key,
    source_cell_specs,
    wilson_lower,
)
from .p0_shortcut import (
    fit_prepared_multinomial_nb,
    predict_prepared_multinomial_nb_batch,
    predict_prepared_multinomial_nb_diagnostic,
    predict_prepared_multinomial_nb_exact,
    prepare_nb_rows,
)
from .renderer import LABELS
from .scalable import compact_predict_multinomial_nb, fit_scalable_multinomial_nb


QUALIFICATION_SEED = 2026081501
QUALIFICATION_TRIALS = 100_000
TRAIN_COUNT = 4096
HELDOUT_COUNT = 1536
V14_ROOT = Path("artifacts/v2-r1r/p0d-v14-full-production-20260809-1")
V14_SEAL_SHA256 = "6FD9AB9CD3F7CB1793F959762BA07AF0A0931052E7A3D05E3DD2C35455D19F52"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _wilson_upper_array(successes: np.ndarray, totals: np.ndarray | int, comparisons: int) -> np.ndarray:
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1.0 - 0.05 / comparisons)
    proportion = successes / totals
    z2 = z * z
    denominator = 1.0 + z2 / totals
    center = proportion + z2 / (2.0 * totals)
    radius = z * np.sqrt(proportion * (1.0 - proportion) / totals + z2 / (4.0 * totals * totals))
    return np.minimum(1.0, (center + radius) / denominator)


def _scenario_definitions(keys: Sequence[str]) -> list[dict[str, Any]]:
    ere_local = "ERE/composition_ood/full_text_char_3_5_nb"
    cps_local = "CPS/horizon_ood/full_text_char_3_5_nb"
    for key in (ere_local, cps_local):
        if key not in keys:
            raise AssertionError(key)
    ere_diffuse = {
        source_cell_key("ERE", split, "full_text_char_3_5_nb"): 0.05
        for split in FAMILY_SPLITS["ERE"]
        if split != "train"
    }
    cps_diffuse = {
        source_cell_key("CPS", split, "full_text_char_3_5_nb"): 0.05
        for split in FAMILY_SPLITS["CPS"]
        if split != "train"
    }
    return [
        {"name": "global_null", "delta": {}, "outcome": "accept", "target_lower": 0.995},
        {"name": "ere_local_plus_0_04", "delta": {ere_local: 0.04}, "outcome": "accept", "target_lower": 0.950},
        {"name": "cps_local_plus_0_04", "delta": {cps_local: 0.04}, "outcome": "accept", "target_lower": 0.950},
        {"name": "ere_local_plus_0_10", "delta": {ere_local: 0.10}, "outcome": "reject", "target_lower": 0.995},
        {"name": "cps_local_plus_0_10", "delta": {cps_local: 0.10}, "outcome": "reject", "target_lower": 0.995},
        {"name": "ere_diffuse_plus_0_05", "delta": ere_diffuse, "outcome": "reject", "target_lower": 0.995},
        {"name": "cps_diffuse_plus_0_05", "delta": cps_diffuse, "outcome": "reject", "target_lower": 0.995},
    ]


def run_decision_power(*, trials: int = QUALIFICATION_TRIALS, seed: int = QUALIFICATION_SEED) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    specs = tuple(sorted(source_cell_specs(train_count=TRAIN_COUNT, heldout_count=HELDOUT_COUNT), key=lambda row: row.key))
    keys = [row.key for row in specs]
    totals = np.asarray([row.total for row in specs], dtype=np.int64)
    random_rates = np.asarray([row.mean_random for row in specs], dtype=np.float64)
    thresholds = random_rates + DEFAULT_CONFIG.source_excess_ceiling
    groups: list[tuple[np.ndarray, int, float]] = []
    for family in ("ERE", "CPS"):
        for baseline in BASELINES:
            indexes = np.asarray(
                [
                    index
                    for index, row in enumerate(specs)
                    if row.family == family and row.split != "train" and row.baseline == baseline
                ],
                dtype=np.int64,
            )
            total = int(totals[indexes].sum())
            random_rate = float(np.sum(random_rates[indexes] * totals[indexes]) / total)
            groups.append((indexes, total, random_rate))
    if len(groups) != SOURCE_FAMILY_CELL_COUNT:
        raise AssertionError("aggregate topology changed")

    rows: dict[str, Any] = {}
    parity_checked = 0
    parity_failures: list[str] = []
    for scenario_index, scenario in enumerate(_scenario_definitions(keys)):
        probabilities = random_rates.copy()
        for key, delta in scenario["delta"].items():
            probabilities[keys.index(key)] += delta
        rng = np.random.Generator(np.random.PCG64(seed + scenario_index * 1_000_003))
        accepted = 0
        for start in range(0, trials, 1000):
            count = min(1000, trials - start)
            observed = rng.binomial(totals, probabilities, size=(count, len(keys)))
            cell_pass = (
                (_wilson_upper_array(observed, totals, SOURCE_CELL_COUNT) <= thresholds + 1e-12)
                & (observed / totals <= thresholds + 1e-12)
            )
            overall = cell_pass.all(axis=1)
            for indexes, aggregate_total, random_rate in groups:
                successes = observed[:, indexes].sum(axis=1)
                aggregate_threshold = random_rate + DEFAULT_CONFIG.aggregate_excess_ceiling
                overall &= (
                    (_wilson_upper_array(successes, aggregate_total, SOURCE_FAMILY_CELL_COUNT) <= aggregate_threshold + 1e-12)
                    & (successes / aggregate_total <= aggregate_threshold + 1e-12)
                )
            accepted += int(overall.sum())
            if start == 0:
                for row_index in range(min(8, count)):
                    success_map = {key: int(observed[row_index, index]) for index, key in enumerate(keys)}
                    total_map = {key: int(totals[index]) for index, key in enumerate(keys)}
                    random_map = {key: float(random_rates[index] * totals[index]) for index, key in enumerate(keys)}
                    complete = {key: True for key in keys}
                    scalar = evaluate_g09_counts(
                        successes=success_map, totals=total_map, random_mass=random_map, complete=complete
                    )["passed"]
                    parity_checked += 1
                    if scalar != bool(overall[row_index]):
                        parity_failures.append(f"{scenario['name']}:{row_index}:{scalar}!={bool(overall[row_index])}")
        desired = accepted if scenario["outcome"] == "accept" else trials - accepted
        rate = desired / trials
        lower = wilson_lower(desired, trials)
        rows[scenario["name"]] = {
            "outcome": scenario["outcome"],
            "delta": scenario["delta"],
            "desired_trials": desired,
            "trials": trials,
            "rate": rate,
            "one_sided_95_wilson_lower": lower,
            "target_lower": scenario["target_lower"],
            "passed": lower >= scenario["target_lower"],
        }
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v15.decision-power.v1",
        "seed": seed,
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "trials_per_scenario": trials,
        "train_count": TRAIN_COUNT,
        "heldout_count": HELDOUT_COUNT,
        "source_cell_count": len(specs),
        "family_aggregate_count": len(groups),
        "vector_scalar_parity": {
            "checked": parity_checked,
            "failures": parity_failures,
            "passed": not parity_failures,
        },
        "scenarios": rows,
        "passed": not parity_failures and all(row["passed"] for row in rows.values()),
    }


def run_decision_faults() -> dict[str, Any]:
    specs = source_cell_specs(train_count=TRAIN_COUNT, heldout_count=HELDOUT_COUNT)
    successes = {row.key: int(math.floor(row.total * row.mean_random)) for row in specs}
    totals = {row.key: row.total for row in specs}
    random_mass = {row.key: row.total * row.mean_random for row in specs}
    complete = {row.key: True for row in specs}
    if not evaluate_g09_counts(successes=successes, totals=totals, random_mass=random_mass, complete=complete)["passed"]:
        raise AssertionError("known-good decision fixture failed")
    first = specs[0].key
    cases: list[tuple[str, Callable[[], Any], str]] = []

    def missing(mapping: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(mapping)
        result.pop(first)
        return result

    cases.extend(
        [
            ("missing_success_cell", lambda: evaluate_g09_counts(successes=missing(successes), totals=totals, random_mass=random_mass, complete=complete), "exception"),
            ("missing_total_cell", lambda: evaluate_g09_counts(successes=successes, totals=missing(totals), random_mass=random_mass, complete=complete), "exception"),
            ("missing_random_cell", lambda: evaluate_g09_counts(successes=successes, totals=totals, random_mass=missing(random_mass), complete=complete), "exception"),
            ("missing_complete_cell", lambda: evaluate_g09_counts(successes=successes, totals=totals, random_mass=random_mass, complete=missing(complete)), "exception"),
            ("extra_cell", lambda: evaluate_g09_counts(successes={**successes, "ERE/train/unknown": 0}, totals=totals, random_mass=random_mass, complete=complete), "exception"),
            ("negative_success", lambda: evaluate_g09_counts(successes={**successes, first: -1}, totals=totals, random_mass=random_mass, complete=complete), "exception"),
            ("success_above_total", lambda: evaluate_g09_counts(successes={**successes, first: totals[first] + 1}, totals=totals, random_mass=random_mass, complete=complete), "exception"),
            ("boolean_success", lambda: evaluate_g09_counts(successes={**successes, first: True}, totals=totals, random_mass=random_mass, complete=complete), "exception"),
            ("zero_total", lambda: evaluate_g09_counts(successes=successes, totals={**totals, first: 0}, random_mass=random_mass, complete=complete), "exception"),
            ("nan_random_mass", lambda: evaluate_g09_counts(successes=successes, totals=totals, random_mass={**random_mass, first: float("nan")}, complete=complete), "exception"),
            ("false_complete", lambda: evaluate_g09_counts(successes=successes, totals=totals, random_mass=random_mass, complete={**complete, first: False}), "failed_decision"),
            ("comparisons_mutation", lambda: evaluate_g09_counts(successes=successes, totals=totals, random_mass=random_mass, complete=complete, config=DecisionConfig(source_comparisons=97)), "exception"),
            ("cell_ceiling_mutation", lambda: evaluate_g09_counts(successes=successes, totals=totals, random_mass=random_mass, complete=complete, config=DecisionConfig(source_excess_ceiling=0.11)), "exception"),
            ("aggregate_ceiling_mutation", lambda: evaluate_g09_counts(successes=successes, totals=totals, random_mass=random_mass, complete=complete, config=DecisionConfig(aggregate_excess_ceiling=0.06)), "exception"),
        ]
    )
    results: dict[str, Any] = {}
    for name, operation, expected in cases:
        detected = False
        observation = "returned"
        try:
            value = operation()
            if expected == "failed_decision" and value["passed"] is False:
                detected = True
                observation = "failed_decision"
        except (TypeError, ValueError):
            observation = "exception"
            detected = expected == "exception"
        results[name] = {"expected": expected, "observed": observation, "detected": detected}
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v15.decision-faults.v1",
        "case_count": len(results),
        "cases": results,
        "passed": all(row["detected"] for row in results.values()),
    }


def verify_v14_sealed_input(repo_root: Path) -> dict[str, Any]:
    root = repo_root / V14_ROOT
    seal_path = root / "evidence-seal.json"
    if _sha(seal_path) != V14_SEAL_SHA256:
        raise ValueError("v14 evidence-seal hash changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    actual = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "evidence-seal.json"
    }
    if actual != seal["files"]:
        raise ValueError("v14 evidence seal does not replay")
    return {"root": V14_ROOT.as_posix(), "seal_sha256": V14_SEAL_SHA256, "sealed_file_count": len(actual)}


def _sample_jsonl(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) < count:
        raise ValueError(f"not enough rows in {path}")
    indexes = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[index] for index in indexes]


def _label(row: Mapping[str, Any]) -> str:
    return LABELS[int(row["answer_index"])]


def _question_objective(source: str) -> str:
    prefixes = (
        "What is ", "Does relation ", "Select the label denoting ", "Select the label stating ",
        "Goal: ", "Final constraints: ", "Budget: ", "Required goal: ", "Required final state: ",
        "Total cost may not exceed ",
    )
    selected = [line for line in source.splitlines() if line.startswith(prefixes)]
    if not selected:
        raise ValueError("no visible question/objective text")
    return "\n".join(selected)


_NB_SPECS: tuple[tuple[str, str, Callable[[Mapping[str, Any]], str]], ...] = (
    ("question_objective_word_nb", "word", lambda row: _question_objective(str(row["source_text"]))),
    ("full_text_word_nb", "word", lambda row: str(row["source_text"])),
    ("full_text_char_3_5_nb", "char_3_5", lambda row: str(row["source_text"])),
)


def _synthetic_scorer_equivalence(seed: int, count: int = 2048) -> dict[str, Any]:
    rng = random.Random(seed)
    mismatches: list[str] = []
    fallback_count = 0
    labels = ("A", "B", "C", "D", "E")
    vocabulary = tuple(f"f{index}" for index in range(24))
    for index in range(count):
        feature_counts = {
            label: Counter({feature: rng.randint(1, 7) for feature in vocabulary if rng.random() < 0.45})
            for label in labels
        }
        class_counts = {label: rng.randint(1, 80) for label in labels}
        model = {
            "labels": labels,
            "class_counts": class_counts,
            "feature_totals": {label: sum(feature_counts[label].values()) for label in labels},
            "feature_counts": feature_counts,
            "vocabulary": vocabulary,
            "vocabulary_lookup": frozenset(vocabulary),
        }
        features = Counter({feature: rng.randint(1, 5) for feature in vocabulary if rng.random() < 0.5})
        mask = [rng.random() < 0.7 for _ in labels]
        if sum(mask) < 2:
            mask[0] = mask[1] = True
        expected = predict_prepared_multinomial_nb_exact(model, features, mask)
        observed = predict_prepared_multinomial_nb_diagnostic(model, features, mask)
        fallback_count += int(observed["exact_fallback"])
        if observed["prediction"] != expected:
            mismatches.append(f"{index}:{observed['prediction']}!={expected}")
    return {
        "prediction_count": count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "exact_fallback_count": fallback_count,
        "passed": not mismatches,
    }


def _tie_scorer_equivalence(count: int = 64) -> dict[str, Any]:
    labels = ("A", "B", "C", "D")
    feature_counts = {label: Counter({"x": 4, "y": 2}) for label in labels}
    model = {
        "labels": labels,
        "class_counts": {label: 8 for label in labels},
        "feature_totals": {label: 6 for label in labels},
        "feature_counts": feature_counts,
        "vocabulary": ("x", "y"),
        "vocabulary_lookup": frozenset(("x", "y")),
    }
    failures: list[str] = []
    fallback_count = 0
    for index in range(count):
        first = index % len(labels)
        mask = [offset in {first, (first + 1) % len(labels)} for offset in range(len(labels))]
        features = Counter({"x": 1 + index % 3, "y": 1})
        expected = predict_prepared_multinomial_nb_exact(model, features, mask)
        observed = predict_prepared_multinomial_nb_diagnostic(model, features, mask)
        fallback_count += int(observed["exact_fallback"])
        if observed["prediction"] != expected or not observed["exact_fallback"]:
            failures.append(f"{index}:{observed}")
    return {
        "prediction_count": count,
        "exact_fallback_count": fallback_count,
        "failures": failures[:20],
        "passed": not failures and fallback_count == count,
    }


def run_scorer_qualification(
    repo_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    emit = progress or (lambda _: None)
    v14 = verify_v14_sealed_input(repo_root)
    dataset = repo_root / V14_ROOT / "dataset"
    registry_predictions = 0
    registry_fallbacks = 0
    registry_mismatches: list[str] = []
    benchmark: list[tuple[Mapping[str, Any], Mapping[str, int], Sequence[bool]]] = []
    for family_index, family in enumerate(("ERE", "CPS")):
        train = _sample_jsonl(dataset / f"{family.lower()}-train.jsonl", 512, QUALIFICATION_SEED + family_index)
        evaluation: list[dict[str, Any]] = []
        for split_index, split in enumerate(FAMILY_SPLITS[family]):
            if split == "train":
                continue
            evaluation.extend(
                _sample_jsonl(
                    dataset / f"{family.lower()}-{split}.jsonl",
                    32,
                    QUALIFICATION_SEED + family_index * 10_000 + split_index,
                )
            )
        for name, analyzer, text in _NB_SPECS:
            emit(f"scorer:{family}:{name}:fit")
            dense_rows = [{"row_id": row["example_id"], "text": text(row), "label": _label(row)} for row in train]
            dense = fit_scalable_multinomial_nb(dense_rows, analyzer, LABELS)
            prepared = prepare_nb_rows(
                train, analyzer=analyzer, row_id=lambda row: row["example_id"], text=text, label=_label
            )
            sparse = fit_prepared_multinomial_nb(prepared, LABELS)
            expected_rows: list[str] = []
            feature_rows: list[Mapping[str, int]] = []
            masks: list[Sequence[bool]] = []
            for row in evaluation:
                source = text(row)
                expected_rows.append(compact_predict_multinomial_nb(dense, source, row["valid_choice_mask"])["prediction"])
                features = analyze(source, analyzer)
                feature_rows.append(features)
                masks.append(row["valid_choice_mask"])
                if name == "full_text_char_3_5_nb" and len(benchmark) < 192:
                    benchmark.append((sparse, features, row["valid_choice_mask"]))
            observed_rows = predict_prepared_multinomial_nb_batch(sparse, feature_rows, masks)
            for row, expected, observed in zip(evaluation, expected_rows, observed_rows, strict=True):
                registry_predictions += 1
                registry_fallbacks += int(observed["exact_fallback"])
                if observed["prediction"] != expected:
                    registry_mismatches.append(f"{family}/{name}/{row['example_id']}:{observed['prediction']}!={expected}")
            emit(f"scorer:{family}:{name}:done")
    if registry_predictions != 1152 or len(benchmark) != 192:
        raise AssertionError(f"qualification sampling changed: {registry_predictions}/{len(benchmark)}")

    synthetic = _synthetic_scorer_equivalence(QUALIFICATION_SEED ^ 0x51A7)
    ties = _tie_scorer_equivalence()
    start = time.perf_counter()
    exact_predictions = [predict_prepared_multinomial_nb_exact(model, features, mask) for model, features, mask in benchmark]
    exact_seconds = time.perf_counter() - start
    start = time.perf_counter()
    benchmark_models = {id(model): model for model, _, _ in benchmark}
    if len(benchmark_models) != 1:
        raise AssertionError("benchmark must use one fixed model")
    benchmark_model = next(iter(benchmark_models.values()))
    fast_results = predict_prepared_multinomial_nb_batch(
        benchmark_model,
        [features for _, features, _ in benchmark],
        [mask for _, _, mask in benchmark],
    )
    fast_seconds = time.perf_counter() - start
    fast_predictions = [str(row["prediction"]) for row in fast_results]
    speedup = exact_seconds / max(fast_seconds, 1e-12)
    benchmark_fallback_rate = sum(bool(row["exact_fallback"]) for row in fast_results) / len(fast_results)
    performance = {
        "prediction_count": len(benchmark),
        "legacy_exact_seconds": exact_seconds,
        "fast_seconds": fast_seconds,
        "speedup": speedup,
        "minimum_speedup": 5.0,
        "exact_fallback_rate": benchmark_fallback_rate,
        "maximum_fallback_rate": 0.01,
        "prediction_identity": exact_predictions == fast_predictions,
    }
    performance["passed"] = (
        performance["prediction_identity"]
        and speedup >= performance["minimum_speedup"]
        and benchmark_fallback_rate <= performance["maximum_fallback_rate"]
    )
    registry = {
        "prediction_count": registry_predictions,
        "mismatch_count": len(registry_mismatches),
        "mismatches": registry_mismatches[:20],
        "exact_fallback_count": registry_fallbacks,
        "exact_fallback_rate": registry_fallbacks / registry_predictions,
        "passed": not registry_mismatches,
    }
    projection = {
        "registry": registry,
        "synthetic": synthetic,
        "ties": ties,
        "benchmark_predictions": fast_predictions,
    }
    streaming: dict[str, Any]
    if tokenizer is None:
        streaming = {"executed": False, "passed": True}
    else:
        from .generator_audit import _source_prediction_matrix

        stream_rows = _sample_jsonl(dataset / "ere-train.jsonl", 200, QUALIFICATION_SEED ^ 0x7711)
        for split_index, split in enumerate(FAMILY_SPLITS["ERE"]):
            if split != "train":
                stream_rows.extend(
                    _sample_jsonl(dataset / f"ere-{split}.jsonl", 20, QUALIFICATION_SEED ^ (0x8800 + split_index))
                )
        stream_events: list[str] = []
        stream_predictions, stream_qualification = _source_prediction_matrix(stream_rows, tokenizer, stream_events.append)
        complete = all(
            len(stream_predictions[split][baseline])
            == sum(row["split"] == split for row in stream_rows)
            for split in FAMILY_SPLITS["ERE"]
            for baseline in BASELINES
        )
        streaming = {
            "executed": True,
            "peak_feature_bank_count": stream_qualification["peak_feature_bank_count"],
            "progress_events": stream_events,
            "batch_count": stream_qualification["scorer_diagnostics"]["batch_count"],
            "prediction_complete": complete,
            "equivalence_passed": stream_qualification["sparse_dense_equivalence"]["passed"],
        }
        streaming["passed"] = (
            streaming["peak_feature_bank_count"] == 1
            and len(streaming["progress_events"]) == 6
            and streaming["batch_count"] > 0
            and streaming["prediction_complete"]
            and streaming["equivalence_passed"]
        )
        projection["streaming"] = streaming
    return {
        "schema_version": "yggdrasil.v2-r1r.p0d-v15.scorer-qualification.v1",
        "v14_input": v14,
        "registry": registry,
        "synthetic": synthetic,
        "ties": ties,
        "performance": performance,
        "streaming": streaming,
        "deterministic_projection_sha256": hashlib.sha256(_canonical(projection)).hexdigest().upper(),
        "progress_event_count": 12,
        "passed": registry["passed"] and synthetic["passed"] and ties["passed"] and performance["passed"] and streaming["passed"],
    }


__all__ = [
    "HELDOUT_COUNT",
    "QUALIFICATION_SEED",
    "QUALIFICATION_TRIALS",
    "TRAIN_COUNT",
    "V14_ROOT",
    "V14_SEAL_SHA256",
    "run_decision_faults",
    "run_decision_power",
    "run_scorer_qualification",
    "verify_v14_sealed_input",
]
