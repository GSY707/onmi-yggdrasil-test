"""Frozen C1 G004--G010 aggregation.

Missing, malformed, non-finite, or weakly structured evidence fails closed.
The aggregator never turns a hidden-state/trajectory diagnostic into an answer
metric; all behavior gates consume externally measured predictions.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from . import contract
from .metrics import finite_json
from .metrics import wilson_interval

FAMILIES = ("ERE", "CPS")
OOD_CELLS = ("composition_ood", "entity_ood", "length_ood", "language_ood", "distractor_ood", "horizon_ood")


def _metric(value: Any, *, allow_negative: bool = False) -> dict[str, Any] | None:
    if not isinstance(value, Mapping): return None
    point = _finite_number(value.get("point")); lower = _finite_number(value.get("lower")); upper = _finite_number(value.get("upper"))
    if point is None or lower is None or upper is None: return None
    floor = -1.0 if allow_negative else 0.0
    if not (floor <= lower <= point <= upper <= 1.0): return None
    return {"point": point, "lower": lower, "upper": upper}


def _pass_metric(value: Any, *, point: float | None = None, lower: float | None = None, upper: float | None = None, allow_negative: bool = False) -> bool:
    metric = _metric(value, allow_negative=allow_negative)
    return metric is not None and (point is None or metric["point"] >= point) and (lower is None or metric["lower"] >= lower) and (upper is None or metric["upper"] <= upper)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _close(left: float, right: float, *, tolerance: float = 1.0e-10) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _unbounded_interval(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    point = _finite_number(value.get("point"))
    lower = _finite_number(value.get("lower"))
    upper = _finite_number(value.get("upper"))
    if point is None or lower is None or upper is None or not lower <= point <= upper:
        return None
    return {"point": point, "lower": lower, "upper": upper}


def _wilson_metric(value: Any, *, total: int | None = None) -> dict[str, float] | None:
    metric = _metric(value)
    if metric is None or not isinstance(value, Mapping):
        return None
    observed_total = value.get("total")
    successes = _finite_number(value.get("successes"))
    if type(observed_total) is not int or observed_total <= 0 or successes is None:
        return None
    if total is not None and observed_total != total:
        return None
    if not successes.is_integer() or not 0.0 <= successes <= observed_total:
        return None
    expected = wilson_interval(int(successes), observed_total)
    if any(abs(metric[key] - float(expected[key])) > 1.0e-12 for key in ("point", "lower", "upper")):
        return None
    return metric


def _pass_wilson(value: Any, *, total: int | None = None, point: float | None = None, lower: float | None = None, upper: float | None = None) -> bool:
    metric = _wilson_metric(value, total=total)
    return metric is not None and (point is None or metric["point"] >= point) and (lower is None or metric["lower"] >= lower) and (upper is None or metric["upper"] <= upper)


def _bootstrap_interval(value: Any, *, clusters: int, unbounded: bool = False, allow_negative: bool = False) -> dict[str, float] | None:
    metric = _unbounded_interval(value) if unbounded else _metric(value, allow_negative=allow_negative)
    if metric is None or not isinstance(value, Mapping):
        return None
    if type(value.get("clusters")) is not int or value.get("clusters") != clusters:
        return None
    if type(value.get("samples")) is not int or value.get("samples") != int(contract.TRAINING_CONFIG["bootstrap_replicates"]):
        return None
    if type(value.get("seed")) is not int or value.get("seed") != contract.BOOTSTRAP_SEED:
        return None
    return metric


def _fail(gate: str, checks: Mapping[str, bool], observed: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"passed": bool(checks) and all(checks.values()), "checks": {str(k): bool(v) for k, v in checks.items()}, "observed": dict(observed or {}), "failures": [str(k) for k, v in checks.items() if not v]}


def _family_metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    families = report.get("families")
    return families if isinstance(families, Mapping) else {}


def _cell_metrics(report: Mapping[str, Any]) -> Mapping[str, Any]:
    cells = report.get("cells")
    if isinstance(cells, Mapping): return cells
    # Accept a family->cell metric shape while preserving explicit cell names.
    out: dict[str, Any] = {}
    for family, value in _family_metrics(report).items():
        if isinstance(value, Mapping):
            for cell, metric in value.get("cells", {}).items() if isinstance(value.get("cells"), Mapping) else ():
                out[str(cell) if "/" in str(cell) else f"{family}/{cell}"] = metric
    return out


def gate_g004_validation(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = {family: _pass_wilson(_family_metrics(report).get(family), total=1536, point=.75, lower=.70) for family in FAMILIES}
    return _fail("G004", checks, {family: _family_metrics(report).get(family) for family in FAMILIES})


def gate_g005_ood(report: Mapping[str, Any]) -> dict[str, Any]:
    cells = _cell_metrics(report); checks: dict[str, bool] = {}; observed: dict[str, Any] = {}
    for family in FAMILIES:
        names = ("composition_ood", "entity_ood", "length_ood", "language_ood") if family == "ERE" else ("composition_ood", "distractor_ood", "horizon_ood", "language_ood")
        for cell in names:
            key = f"{family}/{cell}"; value = cells.get(key, cells.get(cell)); checks[key] = _pass_wilson(value, total=1536, point=.65, lower=.62); observed[key] = value
    return _fail("G005", checks, observed)


def _causal_family(report: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    families = _family_metrics(report); value = families.get(family, {})
    return value if isinstance(value, Mapping) else {}


def gate_g006_causal(report: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}; observed: dict[str, Any] = {}
    for family in FAMILIES:
        value = _causal_family(report, family); observed[family] = value
        checks[f"{family}.pair_coverage"] = type(value.get("pairs")) is int and value.get("pairs") == 768
        checks[f"{family}.base"] = _pass_wilson(value.get("base"), total=768, point=.70, lower=.66)
        checks[f"{family}.flip"] = _pass_wilson(value.get("flip"), total=768, point=.70, lower=.66)
        both = value.get("both_correct", {}) if isinstance(value.get("both_correct"), Mapping) else {}
        semantic = value.get("simulator_semantic_flip", {}) if isinstance(value.get("simulator_semantic_flip"), Mapping) else {}
        both_boot = both.get("bootstrap") if isinstance(both, Mapping) else None
        sem_boot = semantic.get("bootstrap") if isinstance(semantic, Mapping) else None
        both_metric = _bootstrap_interval(both_boot, clusters=768)
        semantic_metric = _bootstrap_interval(sem_boot, clusters=768)
        both_wilson = _wilson_metric(both, total=768)
        semantic_wilson = _wilson_metric(semantic, total=768)
        checks[f"{family}.both_correct"] = (
            both_wilson is not None
            and both_wilson["point"] >= .60
            and both_metric is not None
            and both_metric["lower"] >= .56
            and _close(both_metric["point"], both_wilson["point"])
        )
        checks[f"{family}.simulator_semantic_flip"] = (
            semantic_wilson is not None
            and semantic_wilson["point"] >= .60
            and semantic_metric is not None
            and semantic_metric["lower"] >= .56
            and _close(semantic_metric["point"], semantic_wilson["point"])
        )
    return _fail("G006", checks, observed)


def gate_g007_trace(report: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}; observed: dict[str, Any] = {}
    checks["record_coverage"] = type(report.get("records")) is int and report.get("records") == 512
    for family in FAMILIES:
        value = _family_metrics(report).get(family, {}); observed[family] = value
        if not isinstance(value, Mapping): value = {}
        checks[f"{family}.record_coverage"] = type(value.get("records")) is int and value.get("records") == 256
        all_metric = _wilson_metric(value.get("all_token_accuracy"))
        content_metric = _wilson_metric(value.get("content_token_accuracy"))
        checks[f"{family}.all_token_accuracy"] = all_metric is not None and all_metric["point"] >= .80
        checks[f"{family}.content_token_accuracy"] = content_metric is not None and content_metric["point"] >= .60
        margin = value.get("nll_margin", {}) if isinstance(value.get("nll_margin"), Mapping) else {}
        bootstrap = margin.get("bootstrap") if isinstance(margin, Mapping) else None
        margin_point = _finite_number(margin.get("point")) if isinstance(margin, Mapping) else None
        record_mean = _finite_number(margin.get("record_mean_point")) if isinstance(margin, Mapping) else None
        correct_nll = _finite_number(margin.get("correct_nll_sum")) if isinstance(margin, Mapping) else None
        wrong_nll = _finite_number(margin.get("wrong_nll_sum")) if isinstance(margin, Mapping) else None
        tokens = margin.get("tokens") if isinstance(margin, Mapping) else None
        bootstrap_metric = _bootstrap_interval(bootstrap, clusters=256, unbounded=True)
        token_total = value.get("all_token_accuracy", {}).get("total") if isinstance(value.get("all_token_accuracy"), Mapping) else None
        pooled_consistent = (
            margin_point is not None
            and correct_nll is not None
            and wrong_nll is not None
            and correct_nll >= 0.0
            and wrong_nll >= 0.0
            and type(tokens) is int
            and tokens > 0
            and tokens == token_total
            and _close(margin_point, (wrong_nll - correct_nll) / tokens)
        )
        bootstrap_consistent = (
            record_mean is not None
            and bootstrap_metric is not None
            and _close(record_mean, bootstrap_metric["point"])
        )
        checks[f"{family}.nll_margin"] = (
            pooled_consistent
            and margin_point >= .15
            and bootstrap_consistent
            and bootstrap_metric["lower"] >= .05
        )
    return _fail("G007", checks, observed)


def _intervention_gate(report: Mapping[str, Any], threshold: float, upper: float) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}; observed: dict[str, Any] = {}
    for family in FAMILIES:
        value = _family_metrics(report).get(family, {}); observed[family] = value
        if not isinstance(value, Mapping): value = {}
        drop = value.get("paired_drop", {}) if isinstance(value.get("paired_drop"), Mapping) else {}
        baseline = value.get("baseline")
        intervention = value.get("intervention")
        drop_metric = _bootstrap_interval(drop, clusters=1536, allow_negative=True)
        baseline_metric = _wilson_metric(baseline, total=1536)
        intervention_metric = _wilson_metric(intervention, total=1536)
        checks[f"{family}.baseline_coverage"] = baseline_metric is not None
        checks[f"{family}.drop"] = (
            drop_metric is not None
            and drop_metric["lower"] >= threshold
            and baseline_metric is not None
            and intervention_metric is not None
            and _close(drop_metric["point"], baseline_metric["point"] - intervention_metric["point"])
        )
        checks[f"{family}.intervention_upper"] = intervention_metric is not None and intervention_metric["upper"] <= upper
    return checks, observed


def gate_g008_hidden(report: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}
    for name in ("zero_hidden", "shuffled_hidden"):
        value = report.get(name, {}) if isinstance(report, Mapping) else {}
        child_checks, child_observed = _intervention_gate(
            value if isinstance(value, Mapping) else {}, .20, .35
        )
        checks.update({f"{name}.{key}": passed for key, passed in child_checks.items()})
        observed[name] = child_observed
    return _fail("G008", checks, observed)


def gate_g009_recurrence(report: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}; observed: dict[str, Any] = {}
    # Names are frozen by the contract: H0/no-core and step-5 state shuffle.
    h0 = report.get("h0", report.get("no_core", {})); shuffle = report.get("step5_shuffle", report.get("state_shuffle", {}))
    for family in FAMILIES:
        hv = _family_metrics(h0).get(family, {}) if isinstance(h0, Mapping) else {}; sv = _family_metrics(shuffle).get(family, {}) if isinstance(shuffle, Mapping) else {}
        observed[family] = {"h0": hv, "step5_shuffle": sv}
        h0_drop = _bootstrap_interval(hv.get("paired_drop") if isinstance(hv, Mapping) else None, clusters=1536, allow_negative=True)
        shuffle_drop = _bootstrap_interval(sv.get("paired_drop") if isinstance(sv, Mapping) else None, clusters=1536, allow_negative=True)
        h0_base = _wilson_metric(hv.get("baseline") if isinstance(hv, Mapping) else None, total=1536)
        h0_intervention = _wilson_metric(hv.get("intervention") if isinstance(hv, Mapping) else None, total=1536)
        step_base = _wilson_metric(sv.get("baseline") if isinstance(sv, Mapping) else None, total=1536)
        step_intervention = _wilson_metric(sv.get("intervention") if isinstance(sv, Mapping) else None, total=1536)
        checks[f"{family}.h0_baseline_coverage"] = h0_base is not None
        checks[f"{family}.h0_intervention_coverage"] = h0_intervention is not None
        checks[f"{family}.step5_baseline_coverage"] = step_base is not None
        checks[f"{family}.step5_intervention_coverage"] = step_intervention is not None
        checks[f"{family}.h0"] = (
            h0_drop is not None
            and h0_drop["lower"] >= .15
            and h0_base is not None
            and h0_intervention is not None
            and _close(h0_drop["point"], h0_base["point"] - h0_intervention["point"])
        )
        checks[f"{family}.step5_shuffle"] = (
            shuffle_drop is not None
            and shuffle_drop["lower"] >= .05
            and step_base is not None
            and step_intervention is not None
            and _close(shuffle_drop["point"], step_base["point"] - step_intervention["point"])
        )
    return _fail("G009", checks, observed)


def gate_g010_integrity(report: Mapping[str, Any]) -> dict[str, Any]:
    slot = report.get("slot", report.get("slot_integrity", {})); strip = report.get("strip", report.get("strip_integrity", {}))
    def valid(value: Any, tol: float) -> bool:
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            return False
        invariance = _finite_number(value.get("prediction_invariance"))
        difference = _finite_number(value.get("max_abs_diff"))
        return (
            invariance == 1.0
            and difference is not None
            and 0.0 <= difference <= tol
            and type(value.get("n")) is int
            and value.get("n") == 3072
        )
    checks = {"slot_prediction_invariance": valid(slot, 1e-5), "strip_prediction_invariance": valid(strip, 1e-6)}
    # A stripped artifact may report either an explicit count or a parameter
    # list.  Any remaining probe item is a hard failure.
    probe = report.get("trace_probe_parameter_count", report.get("trace_probe_parameters", report.get("parameter_graph_trace_probe_items")))
    checks["no_trace_probe_parameters"] = type(probe) is int and probe == 0
    return _fail("G010", checks, {"slot": slot, "strip": strip, "trace_probe": probe})


def qualify_gates(reports: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate strict frozen G004--G010 without silently filling evidence."""
    if not isinstance(reports, Mapping): raise TypeError("reports must be a mapping")
    gates = {
        "G004": gate_g004_validation(reports.get("validation", {})),
        "G005": gate_g005_ood(reports.get("ood", {})),
        "G006": gate_g006_causal(reports.get("causal", {})),
        "G007": gate_g007_trace(reports.get("trace", {})),
        "G008": gate_g008_hidden(reports.get("hidden", {})),
        "G009": gate_g009_recurrence(reports.get("recurrence", {})),
        "G010": gate_g010_integrity(reports.get("integrity", reports)),
    }
    return finite_json({"gates": gates, "passed": all(value["passed"] for value in gates.values()), "required_gates": list(gates)})


def aggregate_gates(reports: Mapping[str, Any]) -> dict[str, Any]:
    return qualify_gates(reports)


__all__ = ["FAMILIES", "OOD_CELLS", "gate_g004_validation", "gate_g005_ood", "gate_g006_causal", "gate_g007_trace", "gate_g008_hidden", "gate_g009_recurrence", "gate_g010_integrity", "qualify_gates", "aggregate_gates"]
