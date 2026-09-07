from __future__ import annotations

"""Read-only C1S successor evaluation and ordered Gate decisions.

The evaluator consumes already-collated prediction tensors plus example
metadata.  Metadata is used only on the evaluator side; it is never forwarded
to a model.  Every public report is JSON-native and rejects non-finite values.
"""

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ..closure_c1.contract import THRESHOLDS as C1_BEHAVIOR_THRESHOLDS
from . import contract as successor_contract


LABEL_COUNT = 9
DEFAULT_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_BOOTSTRAP_SEED = successor_contract.S2_BOOTSTRAP_SEED
INTERVENTION_NAMES = (
    "no_core",
    "wrong_start",
    "payload_zero",
    "payload_shuffle",
    "operation_zero",
    "operation_shuffle",
    "relevant_replace",
    "irrelevant_replace",
    "target_shuffle",
)
DYNAMIC_STATE_FEATURES = {
    "ERE": ("touched", "changed", "operation_source", "operation_target"),
    "CPS": ("processed", "running_best", "final_winner"),
}


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite metric")
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Tensor):
        return _plain(value.detach().cpu().tolist())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain(tolist())
    raise TypeError(f"unsupported metric value: {type(value).__name__}")


def finite_json(value: Any) -> dict[str, Any] | list[Any] | Any:
    """Return a JSON-native value, raising rather than serialising NaN/Inf."""

    return _plain(value)


def _float(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite metric input")
    return value


def _tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach().cpu()
    else:
        result = torch.as_tensor(value)
    if not bool(torch.isfinite(result.float()).all()):
        raise ValueError("non-finite tensor input")
    return result.to(dtype=dtype)


def _rows(value: Any, count: int, *, name: str) -> list[Any]:
    if isinstance(value, Mapping):
        if "records" in value:
            value = value["records"]
        elif all(isinstance(item, Mapping) for item in value.values()):
            value = list(value.values())
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of per-example values")
    if len(value) != count:
        raise ValueError(f"{name} length {len(value)} differs from batch count {count}")
    return list(value)


def _metadata_rows(metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any], count: int) -> list[Mapping[str, Any]]:
    if isinstance(metadata, Mapping) and "records" not in metadata:
        tensor_like = [value for value in metadata.values() if isinstance(value, (Tensor, list, tuple))]
        if tensor_like and all(len(value) == count for value in tensor_like):
            return [
                {key: (value[index].item() if isinstance(value, Tensor) and value[index].ndim == 0 else value[index]) for key, value in metadata.items()}
                for index in range(count)
            ]
    rows = _rows(metadata, count, name="metadata")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("metadata rows must be mappings")
    return rows


def _metric(values: Sequence[bool | int | float]) -> dict[str, Any]:
    n = len(values)
    successes = sum(bool(value) for value in values)
    point = successes / n if n else 0.0
    if not n:
        lower = upper = 0.0
    else:
        z = 1.959963984540054
        denominator = 1.0 + z * z / n
        centre = (point + z * z / (2.0 * n)) / denominator
        radius = z * math.sqrt(point * (1.0 - point) / n + z * z / (4.0 * n * n)) / denominator
        lower, upper = max(0.0, centre - radius), min(1.0, centre + radius)
    return {"successes": successes, "n": n, "point": point, "wilson_lower": lower, "wilson_upper": upper}


def _family_metrics(rows: Sequence[Mapping[str, Any]], values: Sequence[bool]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row, value in zip(rows, values):
        groups[str(row.get("family", "unknown"))].append(bool(value))
    return {family: _metric(group) for family, group in sorted(groups.items())}


def _logits(value: Any) -> Tensor:
    logits = _tensor(value)
    if logits.ndim != 2 or logits.shape[-1] != LABEL_COUNT:
        raise ValueError("answer logits must have shape [batch, 9]")
    return logits


def _answer_index(row: Mapping[str, Any]) -> int:
    for key in ("answer_index", "target_index"):
        value = row.get(key)
        if type(value) is int and 0 <= value < LABEL_COUNT:
            return value
    for key in ("answer_label", "target_label", "answer"):
        value = row.get(key)
        if isinstance(value, str) and len(value) == 1 and "A" <= value.upper() <= "I":
            return ord(value.upper()) - ord("A")
    raise ValueError(f"metadata has no answer index/label: {row.get('example_id', '<row>')}")


def _indices(rows: Sequence[Mapping[str, Any]]) -> Tensor:
    return torch.tensor([_answer_index(row) for row in rows], dtype=torch.long)


def _field(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return default


def _owner_metric(values: Tensor, rows: Sequence[Mapping[str, Any]], keys: Sequence[str], *, mask: Tensor | None = None) -> dict[str, Any]:
    if values.ndim < 2:
        raise ValueError("ownership values must have a class/slot dimension")
    predicted = values.argmax(dim=-1)
    targets: list[int] = []
    valid: list[bool] = []
    for row in rows:
        value = _field(row, *keys, default=-1)
        if isinstance(value, Tensor):
            value = value.detach().cpu().tolist()
        targets.append(int(value))
        valid.append(int(value) >= 0)
    if predicted.ndim != 1 or len(targets) != predicted.shape[0]:
        raise ValueError("ownership target shape mismatch")
    keep = torch.tensor(valid, dtype=torch.bool)
    if mask is not None:
        keep &= mask.bool().reshape(-1).cpu()
    correct = ((predicted.cpu() == torch.tensor(targets)) & keep).tolist()
    selected = [value for value, use in zip(correct, keep.tolist()) if use]
    return _metric(selected)


def _sequence_owner_metric(values: Tensor, rows: Sequence[Mapping[str, Any]], key: str, active_key: str) -> dict[str, Any]:
    if values.ndim != 3:
        raise ValueError("sequence ownership values must have shape [batch, steps, slots]")
    predicted = values.argmax(dim=-1).cpu()
    correct: list[bool] = []
    for index, row in enumerate(rows):
        targets = _field(row, key, default=[])
        active = _field(row, active_key, default=[True] * values.shape[1])
        if len(targets) != values.shape[1] or len(active) != values.shape[1]:
            raise ValueError(f"{key} target length mismatch")
        correct.extend(bool(predicted[index, step] == int(targets[step])) for step in range(values.shape[1]) if bool(active[step]) and int(targets[step]) >= 0)
    return _metric(correct)


def _operation_metric(values: Tensor, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predicted = values.float().ge(0.5).cpu()
    correct: list[bool] = []
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"prediction": [], "target": [], "temporal_changes": 0}
    )
    for index, row in enumerate(rows):
        target = _field(row, "operation_active", "active", default=[])
        if len(target) != values.shape[1]:
            raise ValueError("operation_active target length mismatch")
        family = str(row.get("family", "unknown")).upper()
        previous: bool | None = None
        for step in range(values.shape[1]):
            expected = bool(target[step])
            observed = bool(predicted[index, step])
            correct.append(observed == expected)
            grouped[family]["prediction"].append(observed)
            grouped[family]["target"].append(expected)
            if previous is not None and previous != expected:
                grouped[family]["temporal_changes"] += 1
            previous = expected
    families: dict[str, dict[str, Any]] = {}
    for family, cell in sorted(grouped.items()):
        metric = _balanced_binary_metric(cell["prediction"], cell["target"])
        metric["temporal_changes"] = int(cell["temporal_changes"])
        metric["dynamic_coverage"] = bool(
            metric["both_classes"] and int(cell["temporal_changes"]) > 0
        )
        families[family] = metric
    return {
        **_metric(correct),
        "families": families,
        "dynamic_coverage_pass": bool(families)
        and all(cell.get("dynamic_coverage") is True for cell in families.values()),
    }


def _balanced_binary_metric(predicted: Sequence[bool], target: Sequence[bool]) -> dict[str, Any]:
    if len(predicted) != len(target):
        raise ValueError("balanced metric prediction/target length mismatch")
    positives = [index for index, value in enumerate(target) if value]
    negatives = [index for index, value in enumerate(target) if not value]
    tpr = (
        sum(bool(predicted[index]) for index in positives) / len(positives)
        if positives
        else None
    )
    tnr = (
        sum(not bool(predicted[index]) for index in negatives) / len(negatives)
        if negatives
        else None
    )
    balanced = (float(tpr) + float(tnr)) / 2.0 if tpr is not None and tnr is not None else None
    return {
        "accuracy": _metric([left == right for left, right in zip(predicted, target)]),
        "positives": len(positives),
        "negatives": len(negatives),
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "balanced_accuracy": balanced,
        "both_classes": bool(positives and negatives),
    }


def _state_metrics(state_logits: Tensor, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if state_logits.ndim != 4:
        raise ValueError("state_logits must have shape [batch, steps, slots, features]")
    correct: list[bool] = []
    predicted = state_logits.ge(0.0).cpu()
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"prediction": [], "target": [], "temporal_changes": 0}
    )
    for index, row in enumerate(rows):
        target = _tensor(_field(row, "state_values"), dtype=torch.float32).bool()
        feature_mask = _tensor(_field(row, "state_feature_mask"), dtype=torch.float32).bool()
        step_mask = _tensor(_field(row, "state_step_mask", default=[True] * state_logits.shape[1]), dtype=torch.float32).bool()
        if target.shape != predicted.shape[1:] or feature_mask.shape != target.shape or step_mask.numel() != predicted.shape[1]:
            raise ValueError("state target shape mismatch")
        names = list(_field(row, "state_feature_names", default=[]))
        if len(names) != state_logits.shape[-1]:
            raise ValueError("state_feature_names target length mismatch")
        family = str(row.get("family", "unknown")).upper()
        previous: dict[tuple[int, int], bool] = {}
        for step in range(predicted.shape[1]):
            if bool(step_mask[step]):
                correct.extend(
                    (predicted[index, step][feature_mask[step]] == target[step][feature_mask[step]]).tolist()
                )
                for slot in range(predicted.shape[2]):
                    for feature, name in enumerate(names):
                        if not bool(feature_mask[step, slot, feature]):
                            continue
                        value = bool(target[step, slot, feature])
                        cell = grouped[(family, str(name))]
                        cell["prediction"].append(bool(predicted[index, step, slot, feature]))
                        cell["target"].append(value)
                        key = (slot, feature)
                        if key in previous and previous[key] != value:
                            cell["temporal_changes"] += 1
                        previous[key] = value
    families: dict[str, dict[str, Any]] = defaultdict(dict)
    for (family, name), values in sorted(grouped.items()):
        metric = _balanced_binary_metric(values["prediction"], values["target"])
        metric["temporal_changes"] = int(values["temporal_changes"])
        metric["dynamic_coverage"] = bool(
            metric["both_classes"] and int(values["temporal_changes"]) > 0
        )
        families[family][name] = metric
    present_families = sorted({str(row.get("family", "unknown")).upper() for row in rows})
    dynamic_cells: list[dict[str, Any]] = []
    expected_dynamic_cells = 0
    for family in present_families:
        names = DYNAMIC_STATE_FEATURES.get(family, ())
        expected_dynamic_cells += len(names)
        for name in names:
            cell = families.get(family, {}).get(name)
            if isinstance(cell, Mapping):
                dynamic_cells.append(dict(cell))
    return {
        **_metric(correct),
        "families": dict(families),
        "registered_dynamic_features": {
            family: list(DYNAMIC_STATE_FEATURES.get(family, ()))
            for family in present_families
        },
        "dynamic_coverage_pass": bool(expected_dynamic_cells)
        and len(dynamic_cells) == expected_dynamic_cells
        and all(cell.get("dynamic_coverage") is True for cell in dynamic_cells),
    }


def _cluster_bootstrap(values: Sequence[float], clusters: Sequence[Any], *, seed: int, replicates: int) -> dict[str, Any]:
    if len(values) != len(clusters):
        raise ValueError("bootstrap values and clusters must have equal length")
    if not values:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "n": 0, "clusters": 0, "replicates": 0, "seed": int(seed)}
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[str(cluster)].append(_float(value))
    cluster_values = [sum(group) / len(group) for group in grouped.values()]
    point = sum(cluster_values) / len(cluster_values)
    generator = torch.Generator().manual_seed(int(seed))
    samples: list[float] = []
    for _ in range(int(replicates)):
        indices = torch.randint(len(cluster_values), (len(cluster_values),), generator=generator).tolist()
        samples.append(sum(cluster_values[index] for index in indices) / len(indices))
    samples.sort()
    lower = samples[max(0, int(0.025 * len(samples)) - 1)] if samples else point
    upper = samples[min(len(samples) - 1, int(0.975 * len(samples)))] if samples else point
    return {"point": point, "lower": lower, "upper": upper, "n": len(values), "clusters": len(cluster_values), "replicates": int(replicates), "seed": int(seed)}


def _correct_logits(logits: Tensor, answers: Tensor) -> Tensor:
    return logits.gather(1, answers.view(-1, 1)).squeeze(1)


def _answer_margin(logits: Tensor, answers: Tensor) -> Tensor:
    """Gauge-invariant correct-vs-all-wrong log-odds margin."""

    values = logits.float()
    correct = _correct_logits(values, answers)
    wrong = values.masked_fill(
        torch.nn.functional.one_hot(answers, num_classes=LABEL_COUNT).bool(),
        float("-inf"),
    )
    return correct - torch.logsumexp(wrong, dim=-1)


def answer_metrics(logits: Any, metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Any]:
    values = _logits(logits)
    rows = _metadata_rows(metadata, values.shape[0])
    answers = _indices(rows)
    correct = (values.argmax(dim=-1) == answers).tolist()
    return finite_json({"n": len(rows), "overall": _metric(correct), "families": _family_metrics(rows, correct)})


def ownership_metrics(prediction: Mapping[str, Any], metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Any]:
    query = _tensor(prediction["query_weights"])
    rows = _metadata_rows(metadata, query.shape[0])
    out: dict[str, Any] = {
        "query": _owner_metric(query, rows, ("query_owner", "owner_slot")),
    }
    if "source_weights" in prediction:
        out["source"] = _sequence_owner_metric(_tensor(prediction["source_weights"]), rows, "source_owner", "operation_active")
    if "target_weights" in prediction:
        out["target"] = _sequence_owner_metric(_tensor(prediction["target_weights"]), rows, "target_owner", "operation_active")
    if "operation_active" in prediction:
        out["operation_active"] = _operation_metric(_tensor(prediction["operation_active"]), rows)
    if "state_logits" in prediction:
        out["state_masked"] = _state_metrics(_tensor(prediction["state_logits"]), rows)
    return finite_json(out)


def _intervention_mapping(prediction: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = prediction.get("interventions")
    return nested if isinstance(nested, Mapping) else prediction


def intervention_metrics(
    baseline_logits: Any,
    intervention_logits: Mapping[str, Any],
    metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    baseline = _logits(baseline_logits)
    rows = _metadata_rows(metadata, baseline.shape[0])
    answers = _indices(rows)
    base_correct = _correct_logits(baseline, answers)
    base_margin = _answer_margin(baseline, answers)
    result: dict[str, Any] = {}
    for name in INTERVENTION_NAMES:
        raw = intervention_logits.get(name)
        if raw is None:
            raw = intervention_logits.get(f"{name}_logits")
        if raw is None:
            continue
        changed = _logits(raw)
        margin_drops = (base_margin - _answer_margin(changed, answers)).tolist()
        raw_correct_drops = (base_correct - _correct_logits(changed, answers)).tolist()
        correct = (changed.argmax(dim=-1) == answers).tolist()
        clusters = [row.get("pair_id") or row.get("example_id", index) for index, row in enumerate(rows)]
        result[name] = {
            "answer_margin_drop": _cluster_bootstrap(
                margin_drops,
                clusters,
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            ),
            "raw_correct_logit_drop_diagnostic_only": _cluster_bootstrap(
                raw_correct_drops,
                clusters,
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            ),
            "answer": {
                "overall": _metric(correct),
                "families": _family_metrics(rows, correct),
            },
            "rows": len(margin_drops),
        }
    return finite_json(result)


def _metric_scope(
    margin_drops: Sequence[float],
    raw_accuracy_drops: Sequence[float],
    baseline_correct: Sequence[bool],
    intervention_correct: Sequence[bool],
    clusters: Sequence[Any],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Build one explicitly scoped paired-intervention metric report."""

    if not (
        len(margin_drops)
        == len(raw_accuracy_drops)
        == len(baseline_correct)
        == len(intervention_correct)
        == len(clusters)
    ):
        raise ValueError("paired intervention metric vectors must have equal length")
    margin = _cluster_bootstrap(
        margin_drops, clusters, seed=bootstrap_seed, replicates=bootstrap_replicates
    )
    raw = _cluster_bootstrap(
        raw_accuracy_drops,
        clusters,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    return {
        "coverage": bool(margin_drops),
        "n": len(margin_drops),
        "baseline_accuracy": _metric(baseline_correct),
        "intervention_accuracy": _metric(intervention_correct),
        "answer_margin_drop": margin,
        "margin_drop": margin,
        "raw_accuracy_drop_diagnostic_only": raw,
        "raw_correct_accuracy_drop_diagnostic_only": raw,
    }


def paired_intervention_margin_report(
    baseline_logits: Any,
    intervention_logits: Any,
    metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Report gauge-invariant paired intervention degradation.

    The answer margin is ``correct-logsumexp(all_wrong)``.  Raw accuracy and
    raw correct-logit drops are retained as diagnostics only and are never
    used by the successor-native hidden/recurrence gates.
    """

    baseline = _logits(baseline_logits)
    changed = _logits(intervention_logits)
    if baseline.shape != changed.shape:
        raise ValueError("baseline and intervention logits must have equal shape")
    rows = _metadata_rows(metadata, baseline.shape[0])
    answers = _indices(rows)
    base_margin = _answer_margin(baseline, answers)
    changed_margin = _answer_margin(changed, answers)
    margin_drops = (base_margin - changed_margin).tolist()
    base_correct = (baseline.argmax(dim=-1) == answers).tolist()
    intervention_correct = (changed.argmax(dim=-1) == answers).tolist()
    raw_accuracy_drops = [float(a) - float(b) for a, b in zip(base_correct, intervention_correct)]
    clusters = [row.get("pair_id") or row.get("example_id", index) for index, row in enumerate(rows)]

    def scope(indices: Sequence[int]) -> dict[str, Any]:
        return _metric_scope(
            [margin_drops[i] for i in indices],
            [raw_accuracy_drops[i] for i in indices],
            [base_correct[i] for i in indices],
            [intervention_correct[i] for i in indices],
            [clusters[i] for i in indices],
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )

    families = {
        family: scope(
            [i for i, row in enumerate(rows) if str(row.get("family", "unknown")) == family]
        )
        for family in sorted({str(row.get("family", "unknown")) for row in rows})
    }
    return finite_json({"overall": scope(list(range(len(rows)))), "families": families})


def _raw_answer(row: Mapping[str, Any]) -> int:
    """Resolve the raw nine-way answer while accepting target-bank aliases."""

    candidates: list[Any] = []
    for key in ("answer_index", "target_index", "answer"):
        if key in row:
            candidates.append(row[key])
    for key in ("answer_label", "target_label"):
        if key in row:
            candidates.append(row[key])
    values: list[int] = []
    for value in candidates:
        if type(value) is int and 0 <= value < LABEL_COUNT:
            values.append(int(value))
        elif isinstance(value, str) and len(value) == 1 and "A" <= value.upper() <= "I":
            values.append(ord(value.upper()) - ord("A"))
    if not values or any(value != values[0] for value in values):
        raise ValueError(f"metadata has incomplete or inconsistent answer: {row.get('example_id', '<row>')}")
    return values[0]


def _mapping_raw_class(mapping: Any, semantic: Any, *, example_id: Any) -> int:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(f"causal pair mapping missing: {example_id}")
    matches: list[int] = []
    for raw, value in mapping.items():
        if isinstance(raw, str) and len(raw) == 1 and "A" <= raw.upper() <= "I":
            raw_index = ord(raw.upper()) - ord("A")
        elif type(raw) is int and 0 <= raw < LABEL_COUNT:
            raw_index = int(raw)
        else:
            continue
        if _semantic_key(value) == _semantic_key(semantic):
            matches.append(raw_index)
    if len(matches) != 1:
        raise ValueError(f"causal pair semantic mapping is incomplete or non-unique: {example_id}")
    return matches[0]


def _semantic_key(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return f"{type(value).__name__}:{value}"
    return repr(value)


def causal_pair_margin_report(
    logits: Any,
    metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Measure two-way semantic margin degradation across causal pairs.

    For a base/flip pair, the base semantic answer is looked up in the flip
    mapping and vice versa.  This deliberately rejects incomplete pairs,
    non-semantic flips, ambiguous mappings, and non-finite logits.
    """

    values = _logits(logits)
    rows = _metadata_rows(metadata, values.shape[0])
    groups: dict[tuple[str, str], dict[str, tuple[int, Mapping[str, Any]]]] = defaultdict(dict)
    for index, row in enumerate(rows):
        family = str(row.get("family", "unknown"))
        pair_id = row.get("pair_id")
        role = row.get("pair_role")
        if pair_id is None or role not in {"base", "flip"}:
            raise ValueError("causal pair margin evaluation requires base/flip identity on every record")
        key = (family, str(pair_id))
        if role in groups[key]:
            raise ValueError(f"duplicate causal pair role: {key}/{role}")
        # These checks happen before grouping so malformed metadata cannot be
        # silently excluded from a successful family report.
        _raw_answer(row)
        if "semantic_answer" not in row:
            raise ValueError(f"causal pair semantic_answer missing: {row.get('example_id')}")
        if not isinstance(row.get("label_mapping"), Mapping):
            raise ValueError(f"causal pair label_mapping missing: {row.get('example_id')}")
        groups[key][str(role)] = (index, row)

    family_values: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {
            "base_to_flip": [], "flip_to_base": [], "clusters": [],
            "base_correct": [], "flip_correct": [], "both_correct": [],
            "predicted_semantic_flip": [],
        }
    )
    for (family, pair_id), pair in sorted(groups.items()):
        if set(pair) != {"base", "flip"}:
            raise ValueError(f"incomplete causal pair lacks exactly base+flip: {(family, pair_id)}")
        base_index, base = pair["base"]
        flip_index, flip = pair["flip"]
        base_semantic = base["semantic_answer"]
        flip_semantic = flip["semantic_answer"]
        if _semantic_key(base_semantic) == _semantic_key(flip_semantic):
            raise ValueError(f"causal pair is not a semantic flip: {(family, pair_id)}")
        base_answer = _raw_answer(base)
        flip_answer = _raw_answer(flip)
        # Cross-map each semantic answer into the other row's raw class space.
        flip_raw_for_base = _mapping_raw_class(
            flip["label_mapping"], base_semantic, example_id=flip.get("example_id")
        )
        base_raw_for_flip = _mapping_raw_class(
            base["label_mapping"], flip_semantic, example_id=base.get("example_id")
        )
        base_margin = _answer_margin(values[base_index : base_index + 1], torch.tensor([base_answer]))[0]
        flip_margin = _answer_margin(values[flip_index : flip_index + 1], torch.tensor([flip_answer]))[0]
        flip_cross = _answer_margin(
            values[flip_index : flip_index + 1], torch.tensor([flip_raw_for_base])
        )[0]
        base_cross = _answer_margin(
            values[base_index : base_index + 1], torch.tensor([base_raw_for_flip])
        )[0]
        family_values[family]["base_to_flip"].append(_float(base_margin - flip_cross))
        family_values[family]["flip_to_base"].append(_float(flip_margin - base_cross))
        family_values[family]["clusters"].append(pair_id)
        base_pred = int(values[base_index].argmax())
        flip_pred = int(values[flip_index].argmax())
        base_pred_semantic = _mapping_value_for_raw(base["label_mapping"], base_pred, base.get("example_id"))
        flip_pred_semantic = _mapping_value_for_raw(flip["label_mapping"], flip_pred, flip.get("example_id"))
        family_values[family]["base_correct"].append(base_pred == base_answer)
        family_values[family]["flip_correct"].append(flip_pred == flip_answer)
        family_values[family]["both_correct"].append(
            base_pred == base_answer and flip_pred == flip_answer
        )
        family_values[family]["predicted_semantic_flip"].append(
            _semantic_key(base_pred_semantic) != _semantic_key(flip_pred_semantic)
        )

    def report_scope(family: str | None) -> dict[str, Any]:
        if family is None:
            selected = [value for value in family_values.values()]
            directions = {
                name: [item for value in selected for item in value[name]]
                for name in ("base_to_flip", "flip_to_base")
            }
            clusters = [item for value in selected for item in value["clusters"]]
        else:
            value = family_values.get(family, {"base_to_flip": [], "flip_to_base": [], "clusters": []})
            directions = {name: list(value[name]) for name in ("base_to_flip", "flip_to_base")}
            clusters = list(value["clusters"])
        out: dict[str, Any] = {
            "coverage": bool(clusters),
            "pairs": len(clusters),
            "base_to_flip": _cluster_bootstrap(
                directions["base_to_flip"], clusters, seed=bootstrap_seed, replicates=bootstrap_replicates
            ),
            "flip_to_base": _cluster_bootstrap(
                directions["flip_to_base"], clusters, seed=bootstrap_seed, replicates=bootstrap_replicates
            ),
        }
        behavior_values = selected if family is None else [value]
        def behavior(name: str) -> dict[str, Any]:
            raw = [item for item in behavior_values for item in item.get(name, [])]
            return {
                "metric": _metric(raw),
                "bootstrap": _cluster_bootstrap(
                    [float(item) for item in raw],
                    clusters,
                    seed=bootstrap_seed,
                    replicates=bootstrap_replicates,
                ),
            }
        out["behavior"] = {
            "base": behavior("base_correct"),
            "flip": behavior("flip_correct"),
            "both_correct": behavior("both_correct"),
            "predicted_semantic_flip": behavior("predicted_semantic_flip"),
        }
        # A concise alias makes the report easy to consume while preserving
        # both directional estimates as the gate inputs.
        out["answer_margin_drop"] = {
            "point": (out["base_to_flip"]["point"] + out["flip_to_base"]["point"]) / 2.0
            if clusters
            else 0.0,
            "lower": min(out["base_to_flip"]["lower"], out["flip_to_base"]["lower"]),
            "upper": max(out["base_to_flip"]["upper"], out["flip_to_base"]["upper"]),
            "n": len(clusters) * 2,
            "clusters": len(clusters),
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
        }
        out["margin_drop"] = out["answer_margin_drop"]
        return out

    families = {family: report_scope(family) for family in sorted(family_values)}
    return finite_json({"overall": report_scope(None), "families": families})


def _mapping_value_for_raw(mapping: Any, raw_index: int, example_id: Any) -> Any:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"causal pair label_mapping missing: {example_id}")
    labels = (chr(ord("A") + raw_index), str(raw_index), raw_index)
    matches = [mapping[key] for key in labels if key in mapping]
    if len(matches) != 1:
        raise ValueError(f"causal pair raw mapping is incomplete: {example_id}")
    return matches[0]


def gate_g006_causal_behavior(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the legacy causal behavior criteria with successor bootstrap."""

    checks: dict[str, bool] = {}
    for family in ("ERE", "CPS"):
        value = report.get("families", {}).get(family, {})
        behavior = value.get("behavior", {}) if isinstance(value, Mapping) else {}
        checks[f"{family}.coverage"] = bool(value.get("coverage", False))
        for name, point, lower in (
            ("base", 0.70, 0.66),
            ("flip", 0.70, 0.66),
            ("both_correct", 0.60, 0.56),
            ("predicted_semantic_flip", 0.60, 0.56),
        ):
            metric = behavior.get(name, {}).get("metric", {}) if isinstance(behavior, Mapping) else {}
            bootstrap = behavior.get(name, {}).get("bootstrap", {}) if isinstance(behavior, Mapping) else {}
            checks[f"{family}.{name}"] = (
                metric.get("point", 0.0) >= point
                and metric.get("wilson_lower", 0.0) >= lower
                and bootstrap.get("lower", -float("inf")) >= lower
                and bootstrap.get("seed") == report.get("overall", {}).get("answer_margin_drop", {}).get("seed")
            )
    return finite_json({"passed": bool(checks) and all(checks.values()), "checks": checks})


def _margin_gate(
    report: Mapping[str, Any],
    names: Sequence[str],
    *,
    point_threshold: float,
    lower_threshold: float,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    observed: dict[str, Any] = {}
    for name in names:
        value = report.get(name, {}) if isinstance(report, Mapping) else {}
        for family in ("ERE", "CPS"):
            child = value.get("families", {}).get(family, {}) if isinstance(value, Mapping) else {}
            drop = child.get("answer_margin_drop", {}) if isinstance(child, Mapping) else {}
            observed[f"{name}.{family}"] = child
            checks[f"{name}.{family}.coverage"] = bool(child.get("coverage", False))
            point = drop.get("point") if isinstance(drop, Mapping) else None
            lower = drop.get("lower") if isinstance(drop, Mapping) else None
            try:
                point_value = _float(point)
                lower_value = _float(lower)
            except (TypeError, ValueError):
                point_value = lower_value = float("-inf")
            checks[f"{name}.{family}.point"] = point_value >= point_threshold
            checks[f"{name}.{family}.bootstrap_lower"] = lower_value >= lower_threshold
    return finite_json({"passed": bool(checks) and all(checks.values()), "checks": checks, "observed": observed})


def gate_g008_hidden(report: Mapping[str, Any]) -> dict[str, Any]:
    """Successor-native hidden intervention Gate (margin only)."""

    return _margin_gate(
        report,
        ("zero_hidden", "shuffled_hidden"),
        point_threshold=float(successor_contract.OPTIMIZATION_CONFIG["causal_margin"]),
        lower_threshold=float(successor_contract.MECHANISM_THRESHOLDS["intervention_drop_bootstrap_lower"]),
    )


def gate_g009_recurrence(report: Mapping[str, Any]) -> dict[str, Any]:
    """Successor-native recurrence Gate (margin only)."""

    return _margin_gate(
        report,
        ("h0", "step5_shuffle"),
        point_threshold=float(successor_contract.OPTIMIZATION_CONFIG["causal_margin"]),
        lower_threshold=float(successor_contract.MECHANISM_THRESHOLDS["intervention_drop_bootstrap_lower"]),
    )


def gate_g006_causal_margin(report: Mapping[str, Any]) -> dict[str, Any]:
    """Successor-native semantic causal-pair margin sub-gate."""

    return _margin_gate(
        {"causal": report},
        ("causal",),
        point_threshold=float(successor_contract.OPTIMIZATION_CONFIG["causal_margin"]),
        lower_threshold=float(successor_contract.MECHANISM_THRESHOLDS["intervention_drop_bootstrap_lower"]),
    )


# Explicit aliases make the metric-vs-gate distinction discoverable to callers
# without changing the frozen report names used by intervention_metrics.
evaluate_paired_intervention_margin = paired_intervention_margin_report
evaluate_causal_pair_margin = causal_pair_margin_report


def functional_k_metrics(prediction: Mapping[str, Any], metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Any]:
    weights = _tensor(prediction["query_weights"])
    rows = _metadata_rows(metadata, weights.shape[0])
    probabilities = weights.clamp_min(1.0e-12)
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    effective = (-(probabilities * probabilities.log()).sum(dim=-1)).exp().tolist()
    result: dict[str, Any] = {
        "effective_selected_slots": {"point": sum(effective) / len(effective) if effective else 0.0, "per_record": effective, "n": len(effective)},
    }
    selected_owner = weights.argmax(dim=-1)
    owner_counts = torch.bincount(selected_owner, minlength=weights.shape[-1]).float()
    owner_probability = owner_counts / owner_counts.sum().clamp_min(1.0)
    nonzero = owner_probability > 0
    usage_entropy = -(owner_probability[nonzero] * owner_probability[nonzero].log()).sum()
    result["owner_usage"] = {
        "distinct_slots": int(nonzero.sum()),
        "effective_slots": float(usage_entropy.exp()) if bool(nonzero.any()) else 0.0,
        "counts": owner_counts.to(dtype=torch.long).tolist(),
        "n": int(weights.shape[0]),
    }
    effects = prediction.get("initial_slot_margin_effects")
    content_mask_value = prediction.get("content_mask")
    if effects is not None:
        effect_tensor = _tensor(effects)
        if effect_tensor.ndim != 2 or effect_tensor.shape[0] != len(rows):
            raise ValueError("functional-K effects must have shape [batch, slots]")
        if content_mask_value is None:
            raise ValueError("initial_slot_margin_effects requires content_mask")
        content_mask = _tensor(content_mask_value, dtype=torch.float32).bool()
        if content_mask.shape != effect_tensor.shape:
            raise ValueError("functional-K content_mask shape mismatch")
        base = _logits(prediction["logits"])
        no_core = _logits(_intervention_mapping(prediction).get("no_core_logits"))
        answers = _indices(rows)
        denominator = (
            _answer_margin(base, answers) - _answer_margin(no_core, answers)
        ).clamp_min(1.0e-6)
        ratios = effect_tensor / denominator.unsqueeze(1)
        ratio_threshold = float(
            successor_contract.MECHANISM_THRESHOLDS["functional_contributor_ratio"]
        )
        significant = (ratios >= ratio_threshold) & content_mask
        counts = significant.sum(dim=-1).float()
        effective_rows: list[float] = []
        for row_index in range(effect_tensor.shape[0]):
            active = effect_tensor[row_index][content_mask[row_index]].clamp_min(0.0)
            if active.numel() == 0 or float(active.sum()) <= 0.0:
                effective_rows.append(0.0)
                continue
            probabilities_row = active / active.sum()
            effective_rows.append(float((-(probabilities_row * probabilities_row.clamp_min(1.0e-12).log()).sum()).exp()))
        at_least_two = (counts >= 2.0).tolist()
        result["causal_contributors"] = {
            "effect_ratio_threshold": ratio_threshold,
            "mean_count": float(counts.mean()) if len(counts) else 0.0,
            "at_least_two": _metric(at_least_two),
            "mean_effective_slots": sum(effective_rows) / len(effective_rows) if effective_rows else 0.0,
            "per_record_count": counts.tolist(),
            "per_record_effective": effective_rows,
            "per_record_effect_ratio": ratios.tolist(),
        }
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = _field(row, "same_value_group", "semantic_value_group")
        if group is not None:
            groups[str(group)].append(index)
    same_value_pairs = 0
    different_owner_pairs = 0
    owners = [int(_field(row, "owner_slot", "query_owner", default=-1)) for row in rows]
    for group in groups.values():
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                same_value_pairs += 1
                different_owner_pairs += int(owners[group[left]] >= 0 and owners[group[left]] != owners[group[right]])
    result["same_value_different_owner"] = {"different_owner": different_owner_pairs, "pairs": same_value_pairs, "point": different_owner_pairs / same_value_pairs if same_value_pairs else 0.0}
    return finite_json(result)


def invariance_metrics(prediction: Mapping[str, Any], *, permutation_tolerance: float = 1.0e-5, strip_tolerance: float = 1.0e-6) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, key, tolerance in (("slot_permutation", "permuted_logits", permutation_tolerance), ("strip", "stripped_logits", strip_tolerance)):
        if key not in prediction:
            continue
        base = _logits(prediction["logits"])
        changed = _logits(prediction[key])
        same = base.argmax(dim=-1) == changed.argmax(dim=-1)
        maximum = (base - changed).abs().amax(dim=-1)
        result[name] = {"prediction_invariance": float(same.float().mean()), "max_abs_diff": float(maximum.max()) if len(maximum) else 0.0, "tolerance": tolerance, "passed": bool(len(maximum)) and bool((same & (maximum <= tolerance)).all())}
    return finite_json(result)


def paired_eval_gain(k8_logits: Any, k1_logits: Any, metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any], *, bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    k8 = _logits(k8_logits); k1 = _logits(k1_logits)
    if k8.shape != k1.shape:
        raise ValueError("K8 and K1 logits must have equal shape")
    rows = _metadata_rows(metadata, k8.shape[0]); answers = _indices(rows)
    first = (k8.argmax(dim=-1) == answers).tolist(); second = (k1.argmax(dim=-1) == answers).tolist()
    clusters = [row.get("pair_id") or row.get("example_id", index) for index, row in enumerate(rows)]
    family: dict[str, Any] = {}
    for family_name in sorted({str(row.get("family", "unknown")) for row in rows}):
        keep = [index for index, row in enumerate(rows) if str(row.get("family", "unknown")) == family_name]
        family[family_name] = {
            "k8": _metric([first[index] for index in keep]),
            "k1": _metric([second[index] for index in keep]),
            "gain": _cluster_bootstrap(
                [float(first[index]) - float(second[index]) for index in keep],
                [clusters[index] for index in keep],
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            ),
        }
    return finite_json({
        "k8": _metric(first),
        "k1": _metric(second),
        "gain": _cluster_bootstrap([float(a) - float(b) for a, b in zip(first, second)], clusters, seed=bootstrap_seed, replicates=bootstrap_replicates),
        "families": family,
    })


def evaluate_predictions(
    prediction: Mapping[str, Any],
    metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Evaluate one collated prediction mapping and its per-example metadata."""

    logits = _logits(prediction["logits"])
    rows = _metadata_rows(metadata, logits.shape[0])
    report: dict[str, Any] = {"answer": answer_metrics(logits, rows)}
    report["target_contract"] = {
        "mechanism_supported": bool(rows) and all(bool(row.get("mechanism_supported", True)) for row in rows),
        "query_answer_independent": bool(rows) and all(bool(row.get("query_answer_independent", False)) for row in rows),
    }
    if "query_weights" in prediction:
        report["ownership"] = ownership_metrics(prediction, rows)
        report["functional_k"] = functional_k_metrics(prediction, rows)
    report["interventions"] = intervention_metrics(
        logits,
        _intervention_mapping(prediction),
        rows,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    report["invariance"] = invariance_metrics(prediction)
    if "budget" in prediction:
        report["budget"] = finite_json(prediction["budget"])
    return finite_json(report)


def validate_k1_single_slot_control(
    prediction: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    expected_records: int,
) -> dict[str, Any]:
    """Validate a measured K1 endpoint without calling single-slot capacity a failure."""

    weights = _tensor(prediction.get("query_weights"))
    simplex_error = (
        float((weights.sum(dim=-1) - 1.0).abs().max())
        if weights.ndim == 2 and weights.shape[0]
        else float("inf")
    )
    answer_n = int(report.get("answer", {}).get("overall", {}).get("n", 0))
    no_core_n = int(report.get("interventions", {}).get("no_core", {}).get("rows", 0))
    checks = {
        "endpoint_evaluated": answer_n == expected_records and no_core_n == expected_records,
        "observed_query_weight_width_one": weights.ndim == 2 and weights.shape == (expected_records, 1),
        "simplex": math.isfinite(simplex_error) and simplex_error <= 1.0e-6,
        "target_contract": bool(report.get("target_contract", {}).get("query_answer_independent", False)),
        "single_owner_usage": report.get("functional_k", {}).get("owner_usage", {}).get("distinct_slots") == 1,
    }
    passed = all(checks.values())
    return finite_json({
        "status": "VALID_SINGLE_SLOT_CONTROL" if passed else "INVALID_SINGLE_SLOT_CONTROL",
        "passed": passed,
        "checks": checks,
        "configured_slots": 1,
        "evaluated_records": answer_n,
        "simplex_max_error": simplex_error,
        "multi_address_applicable": False,
        "multi_address_qualified": None,
        "not_applicable_controls": {
            name: {"status": "NOT_APPLICABLE", "eligible_n": 0, "reason": "single_slot"}
            for name in ("duplicate_payload", "irrelevant_delete", "wrong_start_permutation", "target_route_shuffle")
        },
    })


def dynamic_state_qualified(report: Mapping[str, Any], *, floor: float) -> bool:
    """Require every registered dynamic feature to have coverage and balanced accuracy."""

    state = report.get("ownership", {}).get("state_masked", {})
    if state.get("dynamic_coverage_pass") is not True:
        return False
    families = state.get("families", {})
    registered = state.get("registered_dynamic_features", {})
    if not isinstance(families, Mapping) or not isinstance(registered, Mapping) or not registered:
        return False
    for family, names in registered.items():
        family_rows = families.get(family, {})
        if not isinstance(family_rows, Mapping):
            return False
        for name in names:
            metric = family_rows.get(name, {})
            if metric.get("dynamic_coverage") is not True:
                return False
            balanced = metric.get("balanced_accuracy")
            if not isinstance(balanced, (int, float)) or float(balanced) < float(floor):
                return False
    return True


def operation_active_qualified(report: Mapping[str, Any], *, floor: float) -> bool:
    """Reject pooled-accuracy shortcuts for the active-operation schedule."""

    operation = report.get("ownership", {}).get("operation_active", {})
    if operation.get("dynamic_coverage_pass") is not True:
        return False
    families = operation.get("families", {})
    if not isinstance(families, Mapping) or not families:
        return False
    for metric in families.values():
        if not isinstance(metric, Mapping) or metric.get("dynamic_coverage") is not True:
            return False
        balanced = metric.get("balanced_accuracy")
        if not isinstance(balanced, (int, float)) or float(balanced) < float(floor):
            return False
    return True


def _cat_prediction_values(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, Tensor):
        return torch.cat([_tensor(value) for value in values], dim=0)
    if isinstance(first, Mapping):
        keys = set(first)
        if any(set(value) != keys for value in values if isinstance(value, Mapping)):
            raise ValueError("prediction batch mappings have different keys")
        return {key: _cat_prediction_values([value[key] for value in values]) for key in sorted(keys)}
    if isinstance(first, (list, tuple)):
        return sum((list(value) for value in values), [])
    return first


def evaluate_collated_batches(
    batches: Iterable[Mapping[str, Any]],
    metadata: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    model: Any | None = None,
) -> dict[str, Any]:
    """Evaluate an iterable of already-collated batches without training.

    With ``model=None`` each batch is treated as a prediction mapping.  When a
    model is supplied, it receives only ``source_hidden`` and ``source_mask``;
    the model's prediction mappings are concatenated before metric evaluation.
    Metadata may be one flat sequence or one sequence per batch.
    """

    batch_list = list(batches)
    if not batch_list:
        raise ValueError("batches must be non-empty")
    predictions: list[Mapping[str, Any]] = []
    batch_sizes: list[int] = []
    for batch in batch_list:
        if not isinstance(batch, Mapping):
            raise ValueError("each batch must be a mapping")
        if model is None:
            prediction = batch
        else:
            source_hidden = batch["source_hidden"]
            source_mask = batch["source_mask"]
            try:
                prediction = model(source_hidden, source_mask, return_trajectory=True, return_auxiliary=True)
            except TypeError:
                prediction = model(source_hidden, source_mask)
        if not isinstance(prediction, Mapping) or "logits" not in prediction:
            raise ValueError("each prediction batch must contain logits")
        logits = _logits(prediction["logits"])
        predictions.append(prediction)
        batch_sizes.append(int(logits.shape[0]))

    if isinstance(metadata, Mapping) and "records" in metadata:
        metadata = metadata["records"]
    metadata_list = list(metadata) if not isinstance(metadata, Mapping) else metadata
    if isinstance(metadata_list, list) and len(metadata_list) == len(batch_list) and metadata_list and all(isinstance(item, Sequence) and not isinstance(item, Mapping) for item in metadata_list):
        rows = sum((list(item) for item in metadata_list), [])
    else:
        rows = metadata_list
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("metadata must be a flat sequence or per-batch sequences")
    if len(rows) != sum(batch_sizes):
        raise ValueError("metadata count differs from collated prediction count")
    merged = {key: _cat_prediction_values([prediction[key] for prediction in predictions]) for key in predictions[0]}
    return evaluate_predictions(merged, rows)


evaluate_batches = evaluate_collated_batches


def s1_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    answer = report.get("answer", {}).get("overall", {})
    answer_families = report.get("answer", {}).get("families", {})
    ownership = report.get("ownership", {})
    functional = report.get("functional_k", {})
    contributors = functional.get("causal_contributors", {})
    interventions = report.get("interventions", {})
    duplicate = report.get("duplicate_payload_control", {})
    threshold = successor_contract.S1_CONFIG["thresholds"]
    mechanism = successor_contract.MECHANISM_THRESHOLDS
    checks = {
        "answer_exact_32": answer.get("point") == 1.0 and answer.get("n") == 32,
        "answer_exact_ere16": answer_families.get("ERE", {}).get("point") == 1.0
        and answer_families.get("ERE", {}).get("n") == 16,
        "answer_exact_cps16": answer_families.get("CPS", {}).get("point") == 1.0
        and answer_families.get("CPS", {}).get("n") == 16,
        "query_owner": ownership.get("query", {}).get("point", 0.0) >= float(threshold["owner_accuracy"]),
        "source_owner": ownership.get("source", {}).get("point", 0.0) >= float(threshold["owner_accuracy"]),
        "target_owner": ownership.get("target", {}).get("point", 0.0) >= float(threshold["owner_accuracy"]),
        "state_masked": ownership.get("state_masked", {}).get("point", 0.0) >= float(threshold["state_accuracy"]),
        "operation_active": operation_active_qualified(
            report, floor=float(threshold["state_accuracy"])
        ),
        "dynamic_state_each_feature": dynamic_state_qualified(report, floor=float(threshold["state_accuracy"])),
        "answer_independent_query_owner": report.get("target_contract", {}).get("query_answer_independent", False),
        "duplicate_payload_owner_and_content": duplicate.get("passed", False),
        "functional_multi_address": contributors.get("at_least_two", {}).get("point", 0.0) >= 0.90
        and contributors.get("mean_effective_slots", 0.0) >= float(mechanism["functional_min_effective_slots"]),
        "no_core_drop": interventions.get("no_core", {}).get("answer_margin_drop", {}).get("point", 0.0) >= float(threshold["causal_answer_margin_drop"]),
        "no_core_near_chance": interventions.get("no_core", {}).get("answer", {}).get("overall", {}).get("point", 1.0) <= float(threshold["no_core_accuracy_max"]),
        "wrong_start_drop": interventions.get("wrong_start", {}).get("answer_margin_drop", {}).get("point", 0.0) >= float(threshold["causal_answer_margin_drop"]),
        "payload_zero_drop": interventions.get("payload_zero", {}).get("answer_margin_drop", {}).get("point", 0.0) >= float(threshold["causal_answer_margin_drop"]),
        "payload_shuffle_drop": interventions.get("payload_shuffle", {}).get("answer_margin_drop", {}).get("point", 0.0) >= float(threshold["causal_answer_margin_drop"]),
        "operation_zero_drop": interventions.get("operation_zero", {}).get("answer_margin_drop", {}).get("point", 0.0) >= float(threshold["causal_answer_margin_drop"]),
        "operation_shuffle_drop": interventions.get("operation_shuffle", {}).get("answer_margin_drop", {}).get("point", 0.0) >= float(threshold["causal_answer_margin_drop"]),
        "relevant_drop": interventions.get("relevant_replace", {}).get("answer_margin_drop", {}).get("point", 0.0) >= float(threshold["causal_answer_margin_drop"]),
        "irrelevant_consistency": abs(interventions.get("irrelevant_replace", {}).get("answer_margin_drop", {}).get("point", 999.0)) <= float(threshold["irrelevant_answer_margin_drop_abs_max"]),
        "target_route_shuffle_drop": interventions.get("target_shuffle", {}).get("answer_margin_drop", {}).get("point", 0.0) >= float(threshold["causal_answer_margin_drop"]),
        "slot_permutation_invariance": report.get("invariance", {}).get("slot_permutation", {}).get("passed", False),
        "s0_single_slot_structural_control_replayed": bool(report.get("s0_single_slot_structural_control_replay", {}).get("passed", False)),
    }
    return finite_json({"stage": "S1", "checks": checks, "passed": all(checks.values()), "authorizes": "S2_DISCOVERY_ONLY" if all(checks.values()) else "nothing"})


def s2_gate(k8: Mapping[str, Any], k1: Mapping[str, Any], *, budget: Mapping[str, Any] | None = None) -> dict[str, Any]:
    answer = k8.get("answer", {}).get("overall", {})
    families = k8.get("answer", {}).get("families", {})
    paired = k8.get("paired_gain", {})
    gain = paired.get("gain", {})
    ownership = k8.get("ownership", {})
    family_reports = k8.get("family_reports", {})
    contributors = k8.get("functional_k", {}).get("causal_contributors", {})
    k1_control = k1.get("single_slot_control", {})
    family_gain = paired.get("families", {})
    config = successor_contract.S2_CONFIG
    checks = {
        "behavior_point_floor": answer.get("point", 0.0) >= float(config["behavior_point_floor"]),
        "behavior_wilson_floor": answer.get("wilson_lower", 0.0) >= float(config["behavior_wilson_lower_floor"]),
        "ere_behavior_floor": families.get("ERE", {}).get("point", 0.0) >= float(config["behavior_point_floor"])
        and families.get("ERE", {}).get("wilson_lower", 0.0) >= float(config["behavior_wilson_lower_floor"]),
        "cps_behavior_floor": families.get("CPS", {}).get("point", 0.0) >= float(config["behavior_point_floor"])
        and families.get("CPS", {}).get("wilson_lower", 0.0) >= float(config["behavior_wilson_lower_floor"]),
        "paired_k8_minus_k1_gain": gain.get("point", 0.0) >= float(config["required_relative_gain_point"])
        and gain.get("lower", -1.0) > 0.0,
        "ere_paired_gain": family_gain.get("ERE", {}).get("gain", {}).get("point", 0.0) >= float(config["required_relative_gain_point"])
        and family_gain.get("ERE", {}).get("gain", {}).get("lower", -1.0) > 0.0,
        "cps_paired_gain": family_gain.get("CPS", {}).get("gain", {}).get("point", 0.0) >= float(config["required_relative_gain_point"])
        and family_gain.get("CPS", {}).get("gain", {}).get("lower", -1.0) > 0.0,
        "ownership_floor": ownership.get("query", {}).get("point", 0.0) >= float(successor_contract.MECHANISM_THRESHOLDS["owner_accuracy"]),
        "state_floor": ownership.get("state_masked", {}).get("point", 0.0) >= float(successor_contract.MECHANISM_THRESHOLDS["state_accuracy"]),
        "ere_mechanism_floor": family_reports.get("ERE", {}).get("mechanism_pass", False),
        "cps_mechanism_floor": family_reports.get("CPS", {}).get("mechanism_pass", False),
        "k1_control_measured_and_valid": k1_control.get("status") == "VALID_SINGLE_SLOT_CONTROL"
        and k1_control.get("passed") is True
        and k1_control.get("multi_address_qualified") is None,
        "functional_k8": contributors.get("at_least_two", {}).get("point", 0.0) >= 0.80
        and contributors.get("mean_effective_slots", 0.0) >= float(successor_contract.MECHANISM_THRESHOLDS["functional_min_effective_slots"]),
    }
    if budget is not None:
        checks["budget"] = budget.get("k8_updates") == int(config["maximum_updates_per_arm"]) and budget.get("k1_updates") == int(config["maximum_updates_per_arm"]) and float(budget.get("k8_hours", 0.0)) <= float(config["gpu_hour_limit_per_arm"]) and float(budget.get("k1_hours", 0.0)) <= float(config["gpu_hour_limit_per_arm"])
    return finite_json({"stage": "S2", "checks": checks, "passed": all(checks.values()), "authorizes": "S3_SINGLE_SEED_FORMAL_ELIGIBILITY" if all(checks.values()) else "nothing", "candidate_level_cps_closure_only": True})


def s3_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate S3 in order; later gates are NOT_RUN after the first failure."""

    answer = report.get("answer", {})
    cells = answer.get("cells", {})
    if not isinstance(cells, Mapping):
        cells = {}
    behavior = answer.get("overall", {})
    checks: dict[str, bool] = {}
    def _behavior_cell(name: str) -> Mapping[str, Any]:
        value = cells.get(name)
        if isinstance(value, Mapping):
            return value
        value = answer.get(name)
        if isinstance(value, Mapping):
            return value
        return behavior if isinstance(behavior, Mapping) else {}

    validation = _behavior_cell("validation")
    ood = _behavior_cell("ood")
    causal = _behavior_cell("causal")
    checks["G004_validation_behavior"] = validation.get("point", 0.0) >= C1_BEHAVIOR_THRESHOLDS["validation"]["point"] and validation.get("wilson_lower", 0.0) >= C1_BEHAVIOR_THRESHOLDS["validation"]["wilson_lower"]
    checks["G004_OOD_behavior"] = ood.get("point", 0.0) >= C1_BEHAVIOR_THRESHOLDS["ood"]["point"] and ood.get("wilson_lower", 0.0) >= C1_BEHAVIOR_THRESHOLDS["ood"]["wilson_lower"]
    checks["G004_causal_behavior"] = causal.get("point", 0.0) >= C1_BEHAVIOR_THRESHOLDS["causal_raw"]["point"] and causal.get("wilson_lower", 0.0) >= C1_BEHAVIOR_THRESHOLDS["causal_raw"]["wilson_lower"]
    checks["G005_ownership_and_state"] = bool(report.get("mechanism", {}).get("ownership_state_pass", False))
    checks["G006_hidden_interventions"] = bool(report.get("mechanism", {}).get("hidden_pass", False))
    checks["G008_recurrence_interventions"] = bool(report.get("mechanism", {}).get("recurrence_pass", False))
    checks["G009_deployment_auxiliary_strip"] = bool(report.get("mechanism", {}).get("strip_pass", False))
    checks["G010_accounting_and_evidence_integrity"] = bool(report.get("accounting_pass", False))
    ordered: dict[str, Any] = {}
    failed: str | None = None
    for name, passed in checks.items():
        if failed is not None:
            ordered[name] = {"status": "NOT_RUN"}
        elif passed:
            ordered[name] = {"status": "PASS"}
        else:
            ordered[name] = {"status": "FAIL"}
            failed = name
    passed = failed is None
    return finite_json({"stage": "S3", "gates": ordered, "first_failed_gate": failed, "passed": passed, "formal": passed, "authorizes": "nothing"})


__all__ = [
    "answer_metrics",
    "causal_pair_margin_report",
    "dynamic_state_qualified",
    "evaluate_batches",
    "evaluate_collated_batches",
    "evaluate_causal_pair_margin",
    "evaluate_paired_intervention_margin",
    "evaluate_predictions",
    "finite_json",
    "functional_k_metrics",
    "invariance_metrics",
    "gate_g006_causal_margin",
    "gate_g006_causal_behavior",
    "gate_g008_hidden",
    "gate_g009_recurrence",
    "intervention_metrics",
    "operation_active_qualified",
    "ownership_metrics",
    "paired_eval_gain",
    "paired_intervention_margin_report",
    "s1_gate",
    "s2_gate",
    "s3_gate",
    "validate_k1_single_slot_control",
]
