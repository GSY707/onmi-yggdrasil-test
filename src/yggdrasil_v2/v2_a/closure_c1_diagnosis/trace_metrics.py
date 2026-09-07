from __future__ import annotations

"""Streaming, aggregate-only trace diagnostics for frozen C1 checkpoints."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
import random
from typing import Any

import torch

from . import contract


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("example_id")
    if not isinstance(value, str) or not value:
        raise ValueError("trace diagnostic record requires example_id")
    return value


def _family(record: Mapping[str, Any]) -> str:
    value = str(record.get("family", "")).upper()
    if value not in {"ERE", "CPS"}:
        raise ValueError("trace diagnostic family must be ERE or CPS")
    return value


def _targets(value: Any) -> Mapping[str, Mapping[str, Any]]:
    target_map = getattr(value, "targets", value)
    if not isinstance(target_map, Mapping):
        raise TypeError("trace target bank must be a mapping or expose .targets")
    return target_map


def select_records(
    records: Iterable[Mapping[str, Any]], *, split: str, per_family: int
) -> list[Mapping[str, Any]]:
    if per_family <= 0:
        raise ValueError("per_family must be positive")
    if not split:
        raise ValueError("split must be non-empty")
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") == split:
            by_family[_family(record)].append(record)
    selected: list[Mapping[str, Any]] = []
    for family in ("ERE", "CPS"):
        rows = sorted(by_family[family], key=_record_id)
        if len(rows) < per_family:
            raise ValueError(
                f"trace diagnostic requires {per_family} {family} {split} records"
            )
        selected.extend(rows[:per_family])
    return selected


def select_validation_records(
    records: Iterable[Mapping[str, Any]], *, per_family: int
) -> list[Mapping[str, Any]]:
    return select_records(records, split="validation", per_family=per_family)


def _groups(
    records: Sequence[Mapping[str, Any]], batch_size: int
) -> list[list[Mapping[str, Any]]]:
    result: list[list[Mapping[str, Any]]] = []
    for family in ("ERE", "CPS"):
        rows = [row for row in records if _family(row) == family]
        result.extend(rows[start : start + batch_size] for start in range(0, len(rows), batch_size))
    return result


def _cache_batch(dataset: Any, ids: Sequence[str], device: torch.device) -> Mapping[str, torch.Tensor]:
    raw = dataset.get_batch(ids, device=device)
    batch = raw[0] if isinstance(raw, tuple) else raw
    if not isinstance(batch, Mapping):
        raise TypeError("diagnostic cache get_batch must return a mapping")
    hidden = batch.get("source_hidden")
    mask = batch.get("source_mask", batch.get("source_attention_mask"))
    if not isinstance(hidden, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise KeyError("diagnostic cache batch lacks source_hidden/source_mask")
    if hidden.shape[:2] != mask.shape or mask.dtype is not torch.bool:
        raise ValueError("diagnostic cache batch shape or mask dtype mismatch")
    return {"source_hidden": hidden.to(device), "source_mask": mask.to(device)}


def _field(entry: Mapping[str, Any], name: str) -> list[Any]:
    aliases = {
        "token_ids": ("token_ids", "target_ids", "local_token_ids"),
        "step_indices": ("step_indices", "step_ids", "steps"),
        "global_positions": ("global_positions", "global_ids"),
        "local_positions": ("local_positions", "local_ids"),
        "grammar_mask": ("grammar_mask", "grammar"),
    }
    for candidate in aliases[name]:
        value = entry.get(candidate)
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
    raise KeyError(f"trace entry lacks {name}")


def _position_bin(position: int) -> str:
    for start, stop, label in contract.POSITION_BINS:
        if start <= position < stop:
            return label
    raise ValueError(f"global trace position outside registered bins: {position}")


def _aligned_steps(steps: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "registered":
        return steps
    if mode == "previous":
        return torch.clamp(steps - 1, min=1)
    if mode == "next":
        return torch.clamp(steps + 1, max=10)
    if mode == "constant_h1":
        return torch.ones_like(steps)
    if mode == "constant_h10":
        return torch.full_like(steps, 10)
    raise ValueError(f"unknown alignment mode: {mode}")


def _empty() -> dict[str, float | int]:
    return {
        "tokens": 0,
        "top1_hits": 0,
        "top5_hits": 0,
        "top10_hits": 0,
        "top50_hits": 0,
        "nll_sum": 0.0,
        "rank_sum": 0.0,
        "reciprocal_rank_sum": 0.0,
    }


def _empty_group() -> dict[str, Any]:
    return {
        "all": _empty(),
        "grammar": _empty(),
        "content": _empty(),
        "steps": defaultdict(_empty),
        "position_bins": defaultdict(_empty),
        "records": defaultdict(_empty),
    }


def _add(
    accumulator: dict[str, float | int],
    *,
    top1: bool,
    top5: bool,
    top10: bool,
    top50: bool,
    nll: float,
    rank: int,
) -> None:
    accumulator["tokens"] = int(accumulator["tokens"]) + 1
    accumulator["top1_hits"] = int(accumulator["top1_hits"]) + int(top1)
    accumulator["top5_hits"] = int(accumulator["top5_hits"]) + int(top5)
    accumulator["top10_hits"] = int(accumulator["top10_hits"]) + int(top10)
    accumulator["top50_hits"] = int(accumulator["top50_hits"]) + int(top50)
    accumulator["nll_sum"] = float(accumulator["nll_sum"]) + float(nll)
    accumulator["rank_sum"] = float(accumulator["rank_sum"]) + int(rank)
    accumulator["reciprocal_rank_sum"] = (
        float(accumulator["reciprocal_rank_sum"]) + 1.0 / int(rank)
    )


def _finalize(value: Mapping[str, float | int]) -> dict[str, Any]:
    count = int(value["tokens"])
    if count <= 0:
        return {"tokens": 0}
    return {
        "tokens": count,
        "top1_accuracy": int(value["top1_hits"]) / count,
        "top5_accuracy": int(value["top5_hits"]) / count,
        "top10_accuracy": int(value["top10_hits"]) / count,
        "top50_accuracy": int(value["top50_hits"]) / count,
        "nll": float(value["nll_sum"]) / count,
        "mean_target_rank": float(value["rank_sum"]) / count,
        "mrr": float(value["reciprocal_rank_sum"]) / count,
    }


def _finalize_family(value: Mapping[str, Any]) -> dict[str, Any]:
    records = list(value["records"].values())
    result = {
        "all": _finalize(value["all"]),
        "grammar": _finalize(value["grammar"]),
        "content": _finalize(value["content"]),
        "steps": {key: _finalize(row) for key, row in sorted(value["steps"].items(), key=lambda item: int(item[0]))},
        "position_bins": {key: _finalize(row) for key, row in sorted(value["position_bins"].items())},
        "record_macro": {
            "records": len(records),
            "top1_accuracy": sum(int(row["top1_hits"]) / int(row["tokens"]) for row in records) / len(records),
            "nll": sum(float(row["nll_sum"]) / int(row["tokens"]) for row in records) / len(records),
        },
    }
    exposure = value.get("exposure")
    if isinstance(exposure, Mapping):
        result["exposure"] = {
            key: _finalize_family(group)
            for key, group in sorted(exposure.items())
        }
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    index = min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))
    return ordered[index]


def _paired_exposure_effect(value: Mapping[str, Any]) -> dict[str, Any]:
    exposure = value.get("exposure")
    if not isinstance(exposure, Mapping) or "0" not in exposure:
        return {"paired_records": 0, "qualified": False}
    zero_records = exposure["0"]["records"]
    exposed_buckets = [exposure[key]["records"] for key in ("1", "2+") if key in exposure]
    rows: list[tuple[str, float, float]] = []
    for example_id, zero in zero_records.items():
        zero_tokens = int(zero["tokens"])
        exposed_parts = [bucket[example_id] for bucket in exposed_buckets if example_id in bucket]
        exposed_tokens = sum(int(part["tokens"]) for part in exposed_parts)
        if zero_tokens <= 0 or exposed_tokens <= 0:
            continue
        zero_accuracy = int(zero["top1_hits"]) / zero_tokens
        exposed_accuracy = sum(int(part["top1_hits"]) for part in exposed_parts) / exposed_tokens
        zero_nll = float(zero["nll_sum"]) / zero_tokens
        exposed_nll = sum(float(part["nll_sum"]) for part in exposed_parts) / exposed_tokens
        rows.append((str(example_id), exposed_accuracy - zero_accuracy, zero_nll - exposed_nll))
    if not rows:
        return {"paired_records": 0, "qualified": False}
    accuracy = [row[1] for row in rows]
    nll = [row[2] for row in rows]
    rng = random.Random(contract.BOOTSTRAP_SEED)
    accuracy_boot: list[float] = []
    nll_boot: list[float] = []
    for _ in range(contract.BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(len(rows)) for _ in rows]
        accuracy_boot.append(sum(accuracy[index] for index in indices) / len(indices))
        nll_boot.append(sum(nll[index] for index in indices) / len(indices))
    accuracy_point = sum(accuracy) / len(accuracy)
    nll_point = sum(nll) / len(nll)
    accuracy_lower = _percentile(accuracy_boot, 0.025)
    nll_lower = _percentile(nll_boot, 0.025)
    qualified = (
        accuracy_point >= float(contract.THRESHOLDS["exposure_accuracy_gap"])
        and accuracy_lower > 0.0
    ) or (nll_point > 0.0 and nll_lower > 0.0)
    return {
        "paired_records": len(rows),
        "accuracy_gap_exposed_minus_unexposed": accuracy_point,
        "accuracy_gap_bootstrap_lower": accuracy_lower,
        "accuracy_gap_bootstrap_upper": _percentile(accuracy_boot, 0.975),
        "nll_improvement_unexposed_minus_exposed": nll_point,
        "nll_improvement_bootstrap_lower": nll_lower,
        "nll_improvement_bootstrap_upper": _percentile(nll_boot, 0.975),
        "qualified": qualified,
    }


def trace_checkpoint_metrics(
    model: Any,
    dataset: Any,
    records: Iterable[Mapping[str, Any]],
    target_bank: Any,
    *,
    device: str | torch.device = "cuda",
    batch_size: int = contract.EVALUATION_BATCH_SIZE,
    per_family: int = contract.TRACE_RECORDS_PER_FAMILY,
    token_chunk: int = contract.TRACE_TOKEN_CHUNK,
    alignment_modes: Sequence[str] = ("registered",),
    split: str = "validation",
    exposure_counts: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Decode frozen trajectories and return aggregate metrics only."""
    if batch_size <= 0 or token_chunk <= 0:
        raise ValueError("batch_size and token_chunk must be positive")
    modes = tuple(str(mode) for mode in alignment_modes)
    if not modes or any(mode not in contract.ALIGNMENT_MODES for mode in modes):
        raise ValueError("alignment_modes differ from the registered modes")
    selected = select_records(records, split=split, per_family=per_family)
    target_map = _targets(target_bank)
    active_device = torch.device(device)
    model.to(active_device).eval()
    trajectories: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for rows in _groups(selected, batch_size):
            ids = [_record_id(row) for row in rows]
            batch = _cache_batch(dataset, ids, active_device)
            output = model(batch["source_hidden"], batch["source_mask"], return_trajectory=True)
            trajectory = output.get("trajectory") if isinstance(output, Mapping) else None
            if not isinstance(trajectory, torch.Tensor) or trajectory.shape[1:3] != (10, 8):
                raise ValueError("C1 diagnostic model did not return [batch,10,8,width] trajectory")
            for index, key in enumerate(ids):
                trajectories[key] = trajectory[index].detach().float().cpu()

        aggregate: dict[str, dict[str, Any]] = {}
        for mode in modes:
            aggregate[mode] = {}
            for family in ("ERE", "CPS"):
                aggregate[mode][family] = _empty_group()
                if exposure_counts is not None:
                    aggregate[mode][family]["exposure"] = defaultdict(_empty_group)

        for rows in _groups(selected, batch_size):
            ids = [_record_id(row) for row in rows]
            entries = []
            for key in ids:
                entry = target_map.get(key)
                if not isinstance(entry, Mapping):
                    raise KeyError(f"trace target bank missing {key}")
                entries.append(entry)
            lengths = [len(_field(entry, "token_ids")) for entry in entries]
            if exposure_counts is not None:
                for key, length in zip(ids, lengths):
                    counts = exposure_counts.get(key)
                    if counts is None or len(counts) != length:
                        raise ValueError(
                            f"exposure counts missing or length-mismatched for {key}"
                        )
            trajectory = torch.stack([trajectories[key] for key in ids]).to(active_device)
            for start in range(0, max(lengths), token_chunk):
                width = min(token_chunk, max(lengths) - start)
                steps = torch.ones((len(rows), width), dtype=torch.long, device=active_device)
                global_positions = torch.zeros_like(steps)
                local_positions = torch.zeros_like(steps)
                targets = torch.zeros_like(steps)
                mask = torch.zeros_like(steps, dtype=torch.bool)
                grammar = torch.zeros_like(steps, dtype=torch.bool)
                for index, (entry, length) in enumerate(zip(entries, lengths)):
                    stop = min(length, start + width)
                    count = max(0, stop - start)
                    if not count:
                        continue
                    mask[index, :count] = True
                    steps[index, :count] = torch.tensor(_field(entry, "step_indices")[start:stop], device=active_device)
                    global_positions[index, :count] = torch.tensor(_field(entry, "global_positions")[start:stop], device=active_device)
                    local_positions[index, :count] = torch.tensor(_field(entry, "local_positions")[start:stop], device=active_device)
                    targets[index, :count] = torch.tensor(_field(entry, "token_ids")[start:stop], device=active_device)
                    grammar[index, :count] = torch.tensor(_field(entry, "grammar_mask")[start:stop], device=active_device)

                for mode in modes:
                    logits = model.trace_logits(
                        trajectory,
                        _aligned_steps(steps, mode),
                        global_positions,
                        local_positions,
                    ).float()
                    log_probs = torch.log_softmax(logits, dim=-1)
                    target_scores = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    ranks = 1 + (logits > target_scores.unsqueeze(-1)).sum(dim=-1)
                    top = logits.topk(min(50, logits.shape[-1]), dim=-1).indices
                    active_rows = mask.nonzero(as_tuple=False)
                    for batch_index, token_index in active_rows.tolist():
                        target = int(targets[batch_index, token_index])
                        candidates = top[batch_index, token_index]
                        hit1 = int(candidates[0]) == target
                        hit5 = bool((candidates[: min(5, candidates.numel())] == target).any())
                        hit10 = bool((candidates[: min(10, candidates.numel())] == target).any())
                        hit50 = bool((candidates == target).any())
                        rank = int(ranks[batch_index, token_index])
                        loss = float(nll[batch_index, token_index])
                        family = _family(rows[batch_index])
                        key = ids[batch_index]
                        kind = "grammar" if bool(grammar[batch_index, token_index]) else "content"
                        step = str(int(steps[batch_index, token_index]))
                        position = int(global_positions[batch_index, token_index])
                        values = aggregate[mode][family]
                        for accumulator in (
                            values["all"],
                            values[kind],
                            values["steps"][step],
                            values["position_bins"][_position_bin(position)],
                            values["records"][key],
                        ):
                            _add(
                                accumulator,
                                top1=hit1,
                                top5=hit5,
                                top10=hit10,
                                top50=hit50,
                                nll=loss,
                                rank=rank,
                            )
                        if exposure_counts is not None:
                            count = int(exposure_counts[key][start + token_index])
                            if count < 0:
                                raise ValueError("exposure counts must be non-negative")
                            bucket = "0" if count == 0 else ("1" if count == 1 else "2+")
                            exposure_values = values["exposure"][bucket]
                            for accumulator in (
                                exposure_values["all"],
                                exposure_values[kind],
                                exposure_values["steps"][step],
                                exposure_values["position_bins"][_position_bin(position)],
                                exposure_values["records"][key],
                            ):
                                _add(
                                    accumulator,
                                    top1=hit1,
                                    top5=hit5,
                                    top10=hit10,
                                    top50=hit50,
                                    nll=loss,
                                    rank=rank,
                                )
                    del logits, log_probs, target_scores, nll, ranks, top

    report = {
        "records": len(selected),
        "records_per_family": per_family,
        "split": split,
        "batch_size": batch_size,
        "token_chunk": token_chunk,
        "alignment_modes": list(modes),
        "modes": {
            mode: {
                family: _finalize_family(aggregate[mode][family])
                for family in ("ERE", "CPS")
            }
            for mode in modes
        },
    }
    if exposure_counts is not None:
        report["paired_exposure_effect"] = {
            mode: {
                family: _paired_exposure_effect(aggregate[mode][family])
                for family in ("ERE", "CPS")
            }
            for mode in modes
        }
    # json.dumps performs a final non-finite/type guard without retaining text.
    import json

    json.dumps(report, allow_nan=False)
    return report


def alignment_decision(report: Mapping[str, Any]) -> dict[str, Any]:
    modes = report.get("modes")
    if not isinstance(modes, Mapping) or "registered" not in modes:
        raise ValueError("alignment decision requires registered metrics")
    comparisons: dict[str, Any] = {}
    qualified_modes: list[str] = []
    for mode in contract.ALIGNMENT_MODES[1:]:
        if mode not in modes:
            continue
        rows: dict[str, Any] = {}
        both = True
        for family in ("ERE", "CPS"):
            registered = modes["registered"][family]["all"]
            alternative = modes[mode][family]["all"]
            nll_gain = float(registered["nll"]) - float(alternative["nll"])
            top1_gain = float(alternative["top1_accuracy"]) - float(registered["top1_accuracy"])
            qualified = (
                nll_gain >= float(contract.THRESHOLDS["alignment_nll_improvement"])
                and top1_gain >= float(contract.THRESHOLDS["alignment_top1_improvement"])
            )
            rows[family] = {
                "nll_improvement": nll_gain,
                "top1_improvement": top1_gain,
                "qualified": qualified,
            }
            both = both and qualified
        comparisons[mode] = rows
        if both:
            qualified_modes.append(mode)
    return {
        "comparisons": comparisons,
        "qualified_modes_both_families": qualified_modes,
        "alignment_suspect": bool(qualified_modes),
    }


def selection_metric_decision(
    selected: Mapping[str, Any], final: Mapping[str, Any]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    qualified = True
    for family in ("ERE", "CPS"):
        left = selected["modes"]["registered"][family]["all"]
        right = final["modes"]["registered"][family]["all"]
        top1_gain = float(right["top1_accuracy"]) - float(left["top1_accuracy"])
        nll_change = float(right["nll"]) - float(left["nll"])
        family_qualified = (
            top1_gain >= float(contract.THRESHOLDS["selection_top1_improvement"])
            and nll_change <= 0.0
        )
        rows[family] = {
            "top1_improvement": top1_gain,
            "nll_change": nll_change,
            "qualified": family_qualified,
        }
        qualified = qualified and family_qualified
    return {"families": rows, "selection_metric_mismatch": qualified}


__all__ = [
    "alignment_decision",
    "select_records",
    "select_validation_records",
    "selection_metric_decision",
    "trace_checkpoint_metrics",
]
