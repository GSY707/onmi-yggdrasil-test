"""Behavioral, causal and training-only trace evaluation for V2-A C1.

The evaluator accepts a normal Python callable (or a small object satisfying
the :class:`Predictor` protocol).  It intentionally consumes predictions only;
record metadata is used for stratification and semantic checking, never as a
model input or as a substitute for behavior.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .metrics import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, cluster_bootstrap_mean, finite_json, paired_cluster_bootstrap_drop, wilson_interval

LABELS = tuple("ABCDEFGHI")
# Frozen CT1 syntax IDs from the C1 contract.  This is intentionally local to
# evaluation and is not accepted as an input feature by any model callable.
FIXED_SYNTAX_TOKEN_IDS = frozenset((58, 92, 487, 1089, 1123, 1143, 1288, 1293, 1666, 1797, 1802, 2129, 2456, 2685, 3147, 4851, 4891, 5046, 5702, 7664, 8631, 8783, 11534, 14522, 15050, 15666, 16352, 17709, 20691, 22357, 22642, 25312, 31928, 32817, 34764, 40775, 45404, 46793, 55558, 80620, 86451, 93482))


@runtime_checkable
class Predictor(Protocol):
    def __call__(self, record: Mapping[str, Any]) -> Any: ...


def _plain(value: Any) -> Any:
    """Convert torch/numpy-like values without making either a dependency."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    detach = getattr(value, "detach", None)
    if callable(detach):
        return _plain(detach().cpu() if hasattr(detach().cpu(), "tolist") else detach().cpu())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain(tolist())
    raise TypeError(f"unsupported prediction value {type(value).__name__}")


def _finite_float(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite prediction")
    return value


def _call(predictor: Callable[..., Any], record: Mapping[str, Any], **kwargs: Any) -> Any:
    try:
        return _plain(predictor(record, **kwargs))
    except TypeError as first:
        if kwargs:
            try:
                return _plain(predictor(record))
            except TypeError:
                raise first
        raise


def _semantic(record: Mapping[str, Any]) -> Any:
    if "semantic_answer" in record:
        return record["semantic_answer"]
    output = record.get("simulator_output")
    if isinstance(output, Mapping) and "answer" in output:
        return output["answer"]
    if "answer" in record:
        return record["answer"]
    if "answer_index" in record:
        return record["answer_index"]
    return None


def _semantic_key(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return f"{type(value).__name__}:{value}"
    return repr(value)


def target_label(record: Mapping[str, Any]) -> str:
    """Resolve a record's own label mapping, without reading choice masks."""
    mapping = record.get("label_mapping")
    if not isinstance(mapping, Mapping):
        answer = record.get("answer_label", record.get("target_label"))
        if isinstance(answer, str) and answer in LABELS:
            return answer
        raise ValueError("record has no label_mapping/target_label")
    answer = _semantic(record)
    if isinstance(answer, str) and answer in LABELS and answer in mapping:
        return answer
    candidates = {"none"} if answer is None else {str(answer), f"value:{answer}", f"candidate:{answer}"}
    if isinstance(answer, bool):
        candidates = {"true" if answer else "false"}
    matches = [str(label) for label, value in mapping.items() if str(label) in LABELS and value in candidates]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve semantic answer for {record.get('example_id', '<record>')}")
    return matches[0]


def _mapped_semantic(record: Mapping[str, Any], label: str) -> Any:
    mapping = record.get("label_mapping")
    if isinstance(mapping, Mapping) and label in mapping:
        return mapping[label]
    return label


def _logits(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        # A mapping is useful for tiny test doubles; labels are still all nine
        # classes and absent labels get -inf only for argmax, never for NLL.
        values = [_finite_float(value.get(label, -1e30)) for label in LABELS]
    else:
        values = [_finite_float(x) for x in value]
    if len(values) != 9:
        raise ValueError("answer logits must contain exactly nine raw classes")
    return values


def _extract_answer(prediction: Any) -> tuple[str, list[float] | None]:
    if isinstance(prediction, Mapping):
        for key in ("logits", "answer_logits", "prediction_logits"):
            if key in prediction:
                logits = _logits(prediction[key]); return LABELS[max(range(9), key=logits.__getitem__)], logits
        for key in ("predicted_label", "label", "prediction", "answer_label"):
            value = prediction.get(key)
            if isinstance(value, str) and value in LABELS:
                return value, None
    if isinstance(prediction, str) and prediction in LABELS:
        return prediction, None
    if isinstance(prediction, Sequence) and not isinstance(prediction, (str, bytes)):
        logits = _logits(prediction); return LABELS[max(range(9), key=logits.__getitem__)], logits
    raise ValueError("predictor must return nine logits or a label")


def _metric(correct: Sequence[bool]) -> dict[str, Any]:
    n = len(correct); successes = sum(bool(x) for x in correct)
    return wilson_interval(successes, n)


def _stratified(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        groups[str(row["family"])].append(bool(row["correct"]))
    return {key: _metric(value) for key, value in sorted(groups.items())}


def evaluate_behavior(records: Iterable[Mapping[str, Any]], predictor: Callable[..., Any]) -> dict[str, Any]:
    """Evaluate answer behavior, always returning family/cell strata."""
    rows: list[dict[str, Any]] = []
    for record in records:
        label, logits = _extract_answer(_call(predictor, record))
        truth = target_label(record)
        rows.append({"example_id": record.get("example_id"), "family": record.get("family"), "cell": f"{record.get('family')}/{record.get('split', 'unknown')}", "pair_id": record.get("pair_id"), "pair_role": record.get("pair_role"), "predicted_label": label, "predicted_semantic": _mapped_semantic(record, label), "semantic_answer": _semantic(record), "target_label": truth, "correct": label == truth, "logits": logits})
    cells: dict[str, list[bool]] = defaultdict(list)
    for row in rows: cells[row["cell"]].append(row["correct"])
    report = {"n": len(rows), "overall": _metric([r["correct"] for r in rows]), "families": _stratified(rows), "cells": {key: _metric(value) for key, value in sorted(cells.items())}, "rows": rows}
    return finite_json(report)


def evaluate_answer(records: Iterable[Mapping[str, Any]], predictor: Callable[..., Any]) -> dict[str, Any]:
    return evaluate_behavior(records, predictor)


def evaluate_causal(records: Iterable[Mapping[str, Any]], predictor: Callable[..., Any]) -> dict[str, Any]:
    """Classify causal pairs from each record's role and own label mapping."""
    records = list(records)
    behavior = evaluate_behavior(records, predictor)
    groups: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    semantic_by_key: dict[tuple[str, str, str], Any] = {}
    for record in records:
        pair_id = record.get("pair_id")
        role = record.get("pair_role")
        if pair_id is None or role not in {"base", "flip"}:
            raise ValueError("causal evaluation requires base/flip identity on every record")
        key = (str(record.get("family")), str(pair_id), str(role))
        if key in semantic_by_key:
            raise ValueError(f"duplicate causal pair role: {key}")
        semantic_by_key[key] = _semantic(record)
    for row in behavior["rows"]:
        if row.get("pair_id") is not None and row.get("pair_role") in {"base", "flip"}:
            key = (str(row["family"]), str(row["pair_id"]), str(row["pair_role"]))
            enriched = dict(row); enriched["semantic_answer"] = semantic_by_key.get(key)
            pair_key = (str(row["family"]), str(row["pair_id"]))
            role = str(row["pair_role"])
            if role in groups[pair_key]:
                raise ValueError(f"duplicate causal pair role after scoring: {pair_key}/{role}")
            groups[pair_key][role] = enriched
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (family, pair_id), pair in groups.items():
        if set(pair) != {"base", "flip"}:
            raise ValueError(
                f"incomplete causal pair lacks exactly base+flip: {(family, pair_id)}"
            )
        base, flip = pair["base"], pair["flip"]
        if base.get("cell") != flip.get("cell"):
            raise ValueError(f"causal pair crossed cells: {(family, pair_id)}")
        semantic_flip = _semantic_key(base.get("semantic_answer")) != _semantic_key(flip.get("semantic_answer"))
        predicted_flip = _semantic_key(base.get("predicted_semantic")) != _semantic_key(flip.get("predicted_semantic"))
        # The semantic values are intentionally looked up from the records in
        # a second map, not inferred from letter changes.
        family_rows[family].append({"pair_id": pair_id, "base_correct": bool(base["correct"]), "flip_correct": bool(flip["correct"]), "both_correct": bool(base["correct"] and flip["correct"]), "semantic_flip": semantic_flip, "predicted_semantic_flip": bool(predicted_flip == semantic_flip)})
    out: dict[str, Any] = {"families": {}}
    for family, pair_rows in sorted(family_rows.items()):
        base = [r["base_correct"] for r in pair_rows]; flip = [r["flip_correct"] for r in pair_rows]
        semantic_rows = [r for r in pair_rows if r["semantic_flip"]]
        both = [r["both_correct"] for r in pair_rows]
        semantic_both = [r["both_correct"] for r in semantic_rows]
        both_metric = _metric(both)
        both_metric["bootstrap"] = cluster_bootstrap_mean([float(x) for x in both], [r["pair_id"] for r in pair_rows])
        semantic_predictions = [r["predicted_semantic_flip"] for r in semantic_rows]
        semantic_metric = _metric(semantic_predictions)
        semantic_metric["bootstrap"] = cluster_bootstrap_mean(
            [float(x) for x in semantic_predictions],
            [r["pair_id"] for r in semantic_rows],
        )
        semantic_both_metric = _metric(semantic_both)
        semantic_both_metric["bootstrap"] = cluster_bootstrap_mean([float(x) for x in semantic_both], [r["pair_id"] for r in semantic_rows])
        semantic_metric["both_correct"] = semantic_both_metric
        out["families"][family] = {"pairs": len(pair_rows), "base": _metric(base), "flip": _metric(flip), "both_correct": both_metric, "simulator_semantic_flip": semantic_metric}
    out["n_pairs"] = sum(len(v) for v in family_rows.values())
    return finite_json(out)


def evaluate_paired_intervention(records: Iterable[Mapping[str, Any]], baseline_predictor: Callable[..., Any], intervention_predictor: Callable[..., Any], *, cluster_key: str = "example_id") -> dict[str, Any]:
    """Report accuracy and a record/pair-cluster bootstrap drop."""
    rows: list[dict[str, Any]] = []
    for record in records:
        truth = target_label(record)
        b, _ = _extract_answer(_call(baseline_predictor, record)); i, _ = _extract_answer(_call(intervention_predictor, record))
        cluster = record.get(cluster_key, record.get("example_id"))
        rows.append({"family": record.get("family"), "cluster": cluster, "base": b == truth, "intervention": i == truth})
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: families[str(row["family"])].append(row)
    output: dict[str, Any] = {"families": {}}
    for family, group in sorted(families.items()):
        output["families"][family] = {"baseline": _metric([r["base"] for r in group]), "intervention": _metric([r["intervention"] for r in group]), "paired_drop": paired_cluster_bootstrap_drop([float(r["base"]) for r in group], [float(r["intervention"]) for r in group], [r["cluster"] for r in group])}
    return finite_json(output)


def _matrix(value: Any) -> list[list[float]]:
    value = _plain(value)
    if not isinstance(value, list) or any(not isinstance(row, list) for row in value): raise ValueError("expected a 2-D logits matrix")
    return [[_finite_float(x) for x in row] for row in value]


def _trace_ids(record: Mapping[str, Any], output: Mapping[str, Any], target_ids: Callable[[Mapping[str, Any]], Sequence[int]] | None) -> list[int]:
    if target_ids is not None: values = target_ids(record)
    else:
        values = output.get("target_ids", output.get("token_ids", record.get("trace_token_ids", record.get("target_token_ids"))))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)): raise ValueError("trace predictor must provide target_ids")
    ids = [int(x) for x in values]
    if any(x < 0 for x in ids): raise ValueError("negative target token id")
    return ids


def _trace_output(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping): return value
    if isinstance(value, (tuple, list)) and len(value) >= 2: return {"target_logits": value[0], "local_logits": value[1], "target_ids": value[2] if len(value) > 2 else None}
    return {"target_logits": value}


def _nll(logits: Sequence[Sequence[float]], token: int) -> float:
    if token < 0 or token >= len(logits): raise ValueError("target token outside logits vocabulary")
    row = [_finite_float(x) for x in logits]; m = max(row); z = sum(math.exp(x - m) for x in row)
    return -(row[token] - m - math.log(z))


def evaluate_trace_credit(
    records: Sequence[Mapping[str, Any]],
    trace_predictor: Callable[..., Any],
    *,
    target_ids: Callable[[Mapping[str, Any]], Sequence[int]] | None = None,
    syntax_token_ids: Iterable[int] = FIXED_SYNTAX_TOKEN_IDS,
    grammar_mask: Callable[[Mapping[str, Any]], Sequence[bool]] | None = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Measure target/local trace credit and a cyclic within-family owner margin.

    ``target_ids`` may use a compact local vocabulary.  In that case callers
    must supply ``grammar_mask`` (or return ``grammar_mask`` from the trace
    predictor); comparing compact class IDs with the original Qwen token IDs
    would silently misclassify content tokens.
    """
    syntax = frozenset(int(x) for x in syntax_token_ids)
    values: list[dict[str, Any]] = []
    outputs: list[Mapping[str, Any]] = []
    for record in records: outputs.append(_trace_output(_call(trace_predictor, record)))
    for index, (record, raw) in enumerate(zip(records, outputs)):
        correct_logits = raw.get("target_logits", raw.get("logits"))
        if correct_logits is None: raise ValueError("missing target_logits")
        correct = _matrix(correct_logits)
        local = raw.get("wrong_owner_logits", raw.get("local_logits"))
        if local is None:
            family = record.get("family"); candidates = [j for j, other in enumerate(records) if other.get("family") == family]
            if not candidates: raise ValueError("no within-family cyclic owner")
            next_index = candidates[(candidates.index(index) + 1) % len(candidates)] if index in candidates else candidates[0]
            owner_raw = outputs[next_index]; local = owner_raw.get("target_logits", owner_raw.get("logits"))
        wrong = _matrix(local); ids = _trace_ids(record, raw, target_ids)
        raw_grammar = (
            grammar_mask(record)
            if grammar_mask is not None
            else raw.get("grammar_mask")
        )
        if raw_grammar is not None:
            if not isinstance(raw_grammar, Sequence) or isinstance(raw_grammar, (str, bytes)):
                raise ValueError("grammar_mask must be a boolean sequence")
            grammar = [bool(value) for value in raw_grammar]
            if len(grammar) != len(ids):
                raise ValueError("grammar_mask length differs from target_ids")
        else:
            grammar = [token in syntax for token in ids]
        if len(correct) < len(ids) or len(wrong) < len(ids): raise ValueError("trace logits shorter than target_ids")
        all_hits = content_hits = all_count = content_count = 0; correct_nll = wrong_nll = 0.0
        for pos, token in enumerate(ids):
            all_hits += int(max(range(len(correct[pos])), key=correct[pos].__getitem__) == token); all_count += 1
            if not grammar[pos]:
                content_hits += int(max(range(len(correct[pos])), key=correct[pos].__getitem__) == token); content_count += 1
            correct_nll += _nll(correct[pos], token); wrong_nll += _nll(wrong[pos], token)
        values.append({"family": record.get("family"), "example_id": record.get("example_id", index), "all_hits": all_hits, "all_count": all_count, "content_hits": content_hits, "content_count": content_count, "correct_nll": correct_nll, "wrong_nll": wrong_nll, "margin": (wrong_nll - correct_nll) / all_count if all_count else 0.0})
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in values: families[str(row["family"])].append(row)
    out: dict[str, Any] = {"records": len(values), "families": {}}
    for family, group in sorted(families.items()):
        all_n = sum(r["all_count"] for r in group); all_hit = sum(r["all_hits"] for r in group); con_n = sum(r["content_count"] for r in group); con_hit = sum(r["content_hits"] for r in group)
        margins = [r["margin"] for r in group]
        correct_nll = sum(r["correct_nll"] for r in group); wrong_nll = sum(r["wrong_nll"] for r in group)
        out["families"][family] = {"records": len(group), "all_token_accuracy": wilson_interval(all_hit, all_n), "content_token_accuracy": wilson_interval(con_hit, con_n), "nll_margin": {"point": (wrong_nll - correct_nll) / all_n if all_n else 0.0, "record_mean_point": sum(margins) / len(margins) if margins else 0.0, "correct_nll_sum": correct_nll, "wrong_nll_sum": wrong_nll, "tokens": all_n, "bootstrap": cluster_bootstrap_mean(margins, [r["example_id"] for r in group], seed=bootstrap_seed)}, "record_bootstrap": cluster_bootstrap_mean([float(r["all_hits"]) / r["all_count"] if r["all_count"] else 0.0 for r in group], [r["example_id"] for r in group], seed=bootstrap_seed)}
    return finite_json(out)


def evaluate_invariance(records: Iterable[Mapping[str, Any]], original_predictor: Callable[..., Any], changed_predictor: Callable[..., Any], *, max_abs_tolerance: float) -> dict[str, Any]:
    rows = []
    for record in records:
        _, a = _extract_answer(_call(original_predictor, record)); _, b = _extract_answer(_call(changed_predictor, record))
        if a is None or b is None: raise ValueError("invariance requires logits")
        diff = max(abs(x - y) for x, y in zip(a, b)); rows.append({"same_prediction": max(range(9), key=a.__getitem__) == max(range(9), key=b.__getitem__), "max_abs_diff": diff})
    return finite_json({"n": len(rows), "prediction_invariance": sum(r["same_prediction"] for r in rows) / len(rows) if rows else 0.0, "max_abs_diff": max((r["max_abs_diff"] for r in rows), default=0.0), "tolerance": max_abs_tolerance, "passed": bool(rows) and all(r["same_prediction"] and r["max_abs_diff"] <= max_abs_tolerance for r in rows)})


__all__ = ["Predictor", "FIXED_SYNTAX_TOKEN_IDS", "target_label", "evaluate_behavior", "evaluate_answer", "evaluate_causal", "evaluate_paired_intervention", "evaluate_trace_credit", "evaluate_invariance"]
