from __future__ import annotations

"""Matched fixed-budget training and evaluation for the P1-H1 two-arm study."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .cache import TokenCache
from .contract import TrainingSpec
from .fresh_data import (
    LABELS,
    UNDECIDED,
    FreshBundle,
    NumericComponent,
    NumericTarget,
    RelationComponent,
    RelationTarget,
    SourceRecord,
    materialize_metamorphic,
    semantic_fingerprint,
    source_fingerprint,
)
from .metrics import classification_metrics, route_metrics, trace_metrics
from .model import H1Config, H1LatentReasoner, build_matched_pair


LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
FAMILY_TO_INDEX = {"numeric": 0, "relation": 1}
INTERVENTIONS = (
    "flip_route",
    "force_route0",
    "force_route1",
    "swap_experts",
    "disable_routed_projection",
    "disable_recurrence",
    "zero_source",
    "shuffle_source",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest().upper()


def _family(component: NumericComponent | RelationComponent) -> str:
    return "numeric" if isinstance(component, NumericComponent) else "relation"


def _label_index(value: str | int) -> int:
    if value == UNDECIDED:
        return UNDECIDED
    if not isinstance(value, str) or value not in LABEL_TO_INDEX:
        raise ValueError(f"invalid H1 label: {value!r}")
    return LABEL_TO_INDEX[value]


def _trace_target(
    target: NumericTarget | RelationTarget,
    steps: int,
) -> tuple[int, ...]:
    if isinstance(target, NumericTarget):
        values = [_label_index(value) for value in target.prefix_labels]
        final = _label_index(target.final_label)
    else:
        by_query = {query_index: LABEL_TO_INDEX[label] for label, query_index in target.choice_to_query}
        values = [UNDECIDED if value == UNDECIDED else by_query[int(value)] for value in target.trace]
        final = LABEL_TO_INDEX[target.label]
    if not values or values[-1] != final:
        # A fixed-point trace can terminate with the correct query in the same
        # final round; numeric traces must also close on the final label.
        values.append(final)
    if len(values) > steps:
        values = values[-steps:]
    values.extend([final] * (steps - len(values)))
    return tuple(values)


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    ids: tuple[str, ...]
    cache_indices: np.ndarray
    targets: np.ndarray
    traces: np.ndarray
    families: tuple[str, ...]
    family_targets: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.ids)
        if not count:
            raise ValueError(f"empty H1 prepared split: {self.split}")
        if (
            self.cache_indices.shape != (count,)
            or self.targets.shape != (count,)
            or self.family_targets.shape != (count,)
            or self.traces.shape[0] != count
            or len(self.families) != count
        ):
            raise ValueError("H1 prepared split shape mismatch")


def prepare_split(
    bundle: FreshBundle,
    split: str,
    cache: TokenCache,
    *,
    steps: int,
) -> PreparedSplit:
    if split not in bundle.splits:
        raise ValueError(f"unknown H1 split: {split}")
    ids = tuple(bundle.splits[split])
    families = tuple(_family(bundle.components[record_id]) for record_id in ids)
    targets = []
    traces = []
    for record_id in ids:
        target = bundle.targets[record_id]
        final = target.final_label if isinstance(target, NumericTarget) else target.label
        targets.append(LABEL_TO_INDEX[final])
        traces.append(_trace_target(target, steps))
    return PreparedSplit(
        split=split,
        ids=ids,
        cache_indices=cache.indices(ids),
        targets=np.asarray(targets, dtype=np.int64),
        traces=np.asarray(traces, dtype=np.int64),
        families=families,
        family_targets=np.asarray([FAMILY_TO_INDEX[value] for value in families], dtype=np.int64),
    )


def materialize_causal_records(
    bundle: FreshBundle,
    *,
    per_family: int = 64,
) -> tuple[tuple[SourceRecord, ...], dict[str, dict[str, Any]]]:
    """Materialize a bounded, balanced set of target-side metamorphic pairs."""
    if per_family < 1:
        raise ValueError("H1 causal materialization count must be positive")
    selected: dict[str, list[str]] = {family: [] for family in FAMILY_TO_INDEX}
    for record_id in bundle.splits["causal"]:
        family = _family(bundle.components[record_id])
        if len(selected[family]) < per_family:
            selected[family].append(record_id)
    if any(len(values) != per_family for values in selected.values()):
        raise ValueError("H1 causal split lacks balanced source records")
    by_id = {record.id: record for record in bundle.records}
    records: list[SourceRecord] = []
    ledger: dict[str, dict[str, Any]] = {}
    for family in ("numeric", "relation"):
        transforms = (
            ("handle_rename", "clause_permutation", "choice_permutation", "cancelling_pair", "surface_paraphrase")
            if family == "numeric"
            else ("handle_rename", "choice_permutation", "query_permutation", "transitive_redundancy", "surface_paraphrase")
        )
        for base_id in selected[family]:
            base_record = by_id[base_id]
            component = bundle.components[base_id]
            target = bundle.targets[base_id]
            for transform in transforms:
                value = materialize_metamorphic(component, target, base_record, transform)
                record = value["record"]
                transformed_target = value["target"]
                if not isinstance(record, SourceRecord):
                    raise ValueError("materialized H1 transform omitted source record")
                final = (
                    transformed_target.final_label
                    if isinstance(transformed_target, NumericTarget)
                    else transformed_target.label
                )
                records.append(record)
                ledger[record.id] = {
                    "base_id": base_id,
                    "transform": transform,
                    "family": family,
                    "base_target": LABEL_TO_INDEX[
                        target.final_label
                        if isinstance(target, NumericTarget)
                        else target.label
                    ],
                    "target": LABEL_TO_INDEX[final],
                    "expected_same_answer": (
                        (
                            target.final_label
                            if isinstance(target, NumericTarget)
                            else target.label
                        )
                        == final
                    ),
                    "semantic_fingerprint": semantic_fingerprint(
                        value["component"],
                        transformed_target,
                    ),
                    "source_fingerprint": source_fingerprint(record),
                    "trace": list(_trace_target(transformed_target, 8)),
                }
    if len(records) != len({record.id for record in records}):
        raise ValueError("materialized H1 causal ids are not unique")
    return tuple(records), ledger


def prepare_materialized_split(
    records: Sequence[SourceRecord],
    ledger: Mapping[str, Mapping[str, Any]],
    cache: TokenCache,
    *,
    steps: int,
) -> PreparedSplit:
    ids = tuple(record.id for record in records)
    families = tuple(str(ledger[record_id]["family"]) for record_id in ids)
    traces = [tuple(int(value) for value in ledger[record_id]["trace"]) for record_id in ids]
    if any(len(value) != steps for value in traces):
        raise ValueError("materialized H1 trace length mismatch")
    return PreparedSplit(
        split="metamorphic",
        ids=ids,
        cache_indices=cache.indices(ids),
        targets=np.asarray([int(ledger[record_id]["target"]) for record_id in ids], dtype=np.int64),
        traces=np.asarray(traces, dtype=np.int64),
        families=families,
        family_targets=np.asarray([FAMILY_TO_INDEX[value] for value in families], dtype=np.int64),
    )


def build_balanced_schedule(
    prepared: PreparedSplit,
    *,
    updates: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("H1 batch size must be positive and family-balanced")
    if updates < 1:
        raise ValueError("H1 updates must be positive")
    generator = np.random.default_rng(seed)
    halves: list[np.ndarray] = []
    half = batch_size // 2
    for family in ("numeric", "relation"):
        pool = np.asarray(
            [index for index, value in enumerate(prepared.families) if value == family],
            dtype=np.int64,
        )
        if len(pool) < half:
            raise ValueError(f"H1 train split has too few {family} records")
        required = updates * half
        values: list[np.ndarray] = []
        remaining = required
        while remaining:
            permutation = generator.permutation(pool)
            take = min(remaining, len(permutation))
            values.append(permutation[:take])
            remaining -= take
        halves.append(np.concatenate(values))
    left = halves[0].reshape(updates, half)
    right = halves[1].reshape(updates, half)
    schedule = np.concatenate((left, right), axis=1)
    for row in schedule:
        generator.shuffle(row)
    return schedule


def schedule_audit(
    prepared: PreparedSplit,
    schedule: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    if schedule.ndim != 2:
        raise ValueError("H1 schedule must be [updates,batch]")
    family_counts = []
    for row in schedule:
        families = [prepared.families[int(index)] for index in row]
        family_counts.append({family: families.count(family) for family in FAMILY_TO_INDEX})
    exposure = np.bincount(schedule.reshape(-1), minlength=len(prepared.ids))
    payload = schedule.astype(np.int64).tolist()
    return {
        "seed": seed,
        "shape": list(schedule.shape),
        "sha256": _sha256_json(payload),
        "family_counts": {
            family: sorted({row[family] for row in family_counts})
            for family in FAMILY_TO_INDEX
        },
        "exposure_min": int(exposure.min()),
        "exposure_max": int(exposure.max()),
        "all_records_exposed": bool(np.all(exposure > 0)),
        "passed": (
            all(len(values) == 1 for values in (
                sorted({row["numeric"] for row in family_counts}),
                sorted({row["relation"] for row in family_counts}),
            ))
            and bool(np.all(exposure > 0))
        ),
    }


def _lr_scale(update: int, spec: TrainingSpec) -> float:
    warmup = max(1, int(spec.updates * spec.warmup_fraction))
    if update <= warmup:
        return update / warmup
    progress = (update - warmup) / max(1, spec.updates - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return spec.minimum_learning_rate_scale + (1.0 - spec.minimum_learning_rate_scale) * cosine


def _losses(
    output: Mapping[str, torch.Tensor | None],
    answers: torch.Tensor,
    traces: torch.Tensor,
    family_targets: torch.Tensor,
    spec: TrainingSpec,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = output["logits"]
    trace_logits = output["trace_logits"]
    route_logits = output["route_logits"]
    if logits is None or trace_logits is None or route_logits is None:
        raise RuntimeError("H1 training output omitted a required head")
    answer_loss = F.cross_entropy(logits, answers)
    trace_loss = F.cross_entropy(
        trace_logits.reshape(-1, trace_logits.shape[-1]),
        traces.reshape(-1),
    )
    route_target = family_targets[:, None, None].expand(route_logits.shape[:-1])
    route_loss = F.cross_entropy(
        route_logits.reshape(-1, route_logits.shape[-1]),
        route_target.reshape(-1),
    )
    total = (
        spec.answer_weight * answer_loss
        + spec.trace_weight * trace_loss
        + spec.route_weight * route_loss
    )
    return total, {
        "answer": float(answer_loss.detach().cpu()),
        "trace": float(trace_loss.detach().cpu()),
        "route": float(route_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def _gpu_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"peak_allocated": 0, "peak_reserved": 0}
    return {
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }


def train_fixed_budget(
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    schedule: np.ndarray,
    spec: TrainingSpec,
    *,
    device: str,
) -> dict[str, Any]:
    if schedule.shape != (spec.updates, spec.batch_size):
        raise ValueError("H1 fixed schedule does not match training spec")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    final_losses: dict[str, float] = {}
    for update_index, local_indices in enumerate(schedule, start=1):
        cache_indices = prepared.cache_indices[local_indices]
        hidden, mask = cache.batch(cache_indices, device=device)
        answers = torch.from_numpy(prepared.targets[local_indices].copy()).to(device)
        traces = torch.from_numpy(prepared.traces[local_indices].copy()).to(device)
        family_targets = torch.from_numpy(prepared.family_targets[local_indices].copy()).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(hidden, mask, return_trajectory=True)
        loss, final_losses = _losses(output, answers, traces, family_targets, spec)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip).detach().cpu()
        )
        scale = _lr_scale(update_index, spec)
        for group in optimizer.param_groups:
            group["lr"] = spec.learning_rate * scale
        optimizer.step()
        if update_index % spec.evaluation_interval == 0 or update_index == spec.updates:
            logits = output["logits"]
            trace_logits = output["trace_logits"]
            route_logits = output["route_logits"]
            assert logits is not None and trace_logits is not None and route_logits is not None
            route_target = family_targets[:, None, None].expand(route_logits.shape[:-1])
            history.append(
                {
                    "update": update_index,
                    "losses": final_losses,
                    "batch_answer_accuracy": float((logits.argmax(-1) == answers).float().mean().detach().cpu()),
                    "batch_trace_accuracy": float((trace_logits.argmax(-1) == traces).float().mean().detach().cpu()),
                    "batch_route_accuracy": float((route_logits.argmax(-1) == route_target).float().mean().detach().cpu()),
                    "gradient_norm_before_clip": gradient_norm,
                    "learning_rate": spec.learning_rate * scale,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        del hidden, mask, answers, traces, family_targets, output, loss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    return {
        "updates": spec.updates,
        "batch_size": spec.batch_size,
        "examples_seen": spec.updates * spec.batch_size,
        "wall_seconds": wall_seconds,
        "updates_per_second": spec.updates / wall_seconds,
        "examples_per_second": (spec.updates * spec.batch_size) / wall_seconds,
        "history": history,
        "final_losses": final_losses,
        "gpu_memory": _gpu_memory(),
        "parameter_report": model.parameter_report(),
    }


def _route_diagnostics(
    assignments: np.ndarray,
    family_targets: np.ndarray,
) -> dict[str, Any]:
    if assignments.ndim != 3 or assignments.shape[0] != family_targets.shape[0]:
        raise ValueError("H1 route diagnostics require [records,layers,steps]")
    cells: list[dict[str, Any]] = []
    min_load = 1.0
    min_entropy = 1.0
    for layer in range(assignments.shape[1]):
        for step in range(assignments.shape[2]):
            values = assignments[:, layer, step]
            count0 = int(np.sum(values == 0))
            count1 = int(np.sum(values == 1))
            count = count0 + count1
            load0 = count0 / count
            load1 = count1 / count
            entropy = 0.0
            for load in (load0, load1):
                if load > 0.0:
                    entropy -= load * math.log(load, 2)
            min_load = min(min_load, load0, load1)
            min_entropy = min(min_entropy, entropy)
            cells.append(
                {
                    "layer": layer,
                    "step": step,
                    "count0": count0,
                    "count1": count1,
                    "load0": load0,
                    "load1": load1,
                    "entropy_bits": entropy,
                    "route_accuracy": float(np.mean(values == family_targets)),
                    "selected_record_executions": {"0": count0, "1": count1},
                }
            )
    return {
        "shape": list(assignments.shape),
        "cells": cells,
        "minimum_expert_load": min_load,
        "minimum_entropy_bits": min_entropy,
        "all_cells_use_both_experts": bool(min_load > 0.0),
    }


def evaluate_model(
    model: H1LatentReasoner,
    cache: TokenCache,
    prepared: PreparedSplit,
    *,
    device: str,
    batch_size: int = 64,
    intervention: str = "none",
    include_trace: bool = True,
    include_state_hash: bool = False,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    predictions: list[int] = []
    traces: list[list[int]] = []
    routes: list[int] = []
    route_tensors: list[np.ndarray] = []
    state_hashers = {
        name: hashlib.sha256()
        for name in ("logits", "final_state", "trajectory")
    } if include_state_hash else {}
    with torch.inference_mode():
        for start in range(0, len(prepared.ids), batch_size):
            stop = min(start + batch_size, len(prepared.ids))
            hidden, mask = cache.batch(prepared.cache_indices[start:stop], device=device)
            output = model(
                hidden,
                mask,
                intervention=intervention,
                return_trajectory=include_trace and model.trace_head is not None,
            )
            logits = output["logits"]
            route_assignments = output["route_assignments"]
            assert logits is not None and route_assignments is not None
            if include_state_hash:
                final_state = output["final_state"]
                trajectory = output["trajectory"]
                assert final_state is not None
                for name, tensor in (
                    ("logits", logits),
                    ("final_state", final_state),
                    ("trajectory", trajectory),
                ):
                    if tensor is not None:
                        state_hashers[name].update(
                            tensor.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
                        )
            predictions.extend(int(value) for value in logits.argmax(-1).cpu().tolist())
            # Majority route over layers and steps is the record route.
            majority = (route_assignments.to(torch.float32).mean(dim=(1, 2)) >= 0.5).long()
            routes.extend(int(value) for value in majority.cpu().tolist())
            route_tensors.append(route_assignments.cpu().numpy().astype(np.int8, copy=False))
            if include_trace and model.trace_head is not None:
                trace_logits = output["trace_logits"]
                assert trace_logits is not None
                traces.extend(
                    [list(map(int, row)) for row in trace_logits.argmax(-1).cpu().tolist()]
                )
            del hidden, mask, output
    answer = classification_metrics(
        prepared.ids,
        predictions,
        prepared.targets.tolist(),
        prepared.families,
    )
    route_array = np.concatenate(route_tensors, axis=0)
    route = route_metrics(routes, prepared.family_targets.tolist())
    route["diagnostics"] = _route_diagnostics(route_array, prepared.family_targets)
    result: dict[str, Any] = {
        "split": prepared.split,
        "intervention": intervention,
        "ids": list(prepared.ids),
        "predictions": predictions,
        "targets": prepared.targets.tolist(),
        "families": list(prepared.families),
        "answer": answer,
        "route_predictions": routes,
        "route_assignments": route_array.tolist(),
        "route": route,
    }
    if traces:
        result["trace_predictions"] = traces
        result["trace_targets"] = prepared.traces.tolist()
        result["trace"] = trace_metrics(traces, prepared.traces.tolist(), prepared.families)
    if include_state_hash:
        result["state_hashes"] = {
            name: digest.hexdigest().upper()
            for name, digest in state_hashers.items()
            if name != "trajectory" or traces
        }
    return result


def metamorphic_consistency(
    base_evaluation: Mapping[str, Any],
    transformed_evaluation: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score registered base/transformed answer relations pair by pair."""
    base_predictions = dict(
        zip(base_evaluation["ids"], base_evaluation["predictions"], strict=True)
    )
    base_targets = dict(
        zip(base_evaluation["ids"], base_evaluation["targets"], strict=True)
    )
    transformed_predictions = dict(
        zip(
            transformed_evaluation["ids"],
            transformed_evaluation["predictions"],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    for transformed_id in transformed_evaluation["ids"]:
        item = ledger.get(transformed_id)
        if item is None:
            raise ValueError(f"missing H1 metamorphic ledger row: {transformed_id}")
        base_id = str(item["base_id"])
        if base_id not in base_predictions:
            raise ValueError(f"H1 metamorphic base missing from evaluation: {base_id}")
        base_target = int(item.get("base_target", base_targets[base_id]))
        transformed_target = int(item["target"])
        if int(base_targets[base_id]) != base_target:
            raise ValueError(f"H1 metamorphic base target mismatch: {base_id}")
        expected_same = bool(item["expected_same_answer"])
        base_prediction = int(base_predictions[base_id])
        transformed_prediction = int(transformed_predictions[transformed_id])
        relation_ok = (
            transformed_prediction == base_prediction
            if expected_same
            else (
                base_prediction == base_target
                and transformed_prediction == transformed_target
                and transformed_prediction != base_prediction
            )
        )
        rows.append(
            {
                "base_id": base_id,
                "transformed_id": transformed_id,
                "transform": str(item["transform"]),
                "family": str(item["family"]),
                "expected_same_answer": expected_same,
                "base_target": base_target,
                "transformed_target": transformed_target,
                "base_prediction": base_prediction,
                "transformed_prediction": transformed_prediction,
                "relation_ok": relation_ok,
                "base_exact": base_prediction == base_target,
                "transformed_exact": transformed_prediction == transformed_target,
            }
        )

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        if not count:
            return {
                "count": 0,
                "relation_consistency": 0.0,
                "base_exact": 0.0,
                "transformed_exact": 0.0,
            }
        return {
            "count": count,
            "relation_consistency": sum(bool(row["relation_ok"]) for row in selected) / count,
            "base_exact": sum(bool(row["base_exact"]) for row in selected) / count,
            "transformed_exact": sum(bool(row["transformed_exact"]) for row in selected) / count,
        }

    transforms = sorted({str(row["transform"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})
    return {
        "aggregate": summarize(rows),
        "by_transform": {
            name: summarize([row for row in rows if row["transform"] == name])
            for name in transforms
        },
        "by_family": {
            name: summarize([row for row in rows if row["family"] == name])
            for name in families
        },
        "relation_sha256": _sha256_json(
            [
                {
                    "base_id": row["base_id"],
                    "transformed_id": row["transformed_id"],
                    "relation_ok": row["relation_ok"],
                }
                for row in rows
            ]
        ),
        "rows": rows,
    }


def metamorphic_route_consistency(
    base_evaluation: Mapping[str, Any],
    transformed_evaluation: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify that equivalent surfaces keep the learned family route."""
    base_routes = dict(
        zip(
            base_evaluation["ids"],
            base_evaluation["route_predictions"],
            strict=True,
        )
    )
    transformed_routes = dict(
        zip(
            transformed_evaluation["ids"],
            transformed_evaluation["route_predictions"],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    for transformed_id in transformed_evaluation["ids"]:
        item = ledger.get(transformed_id)
        if item is None:
            raise ValueError(
                f"missing H1 metamorphic route ledger row: {transformed_id}"
            )
        base_id = str(item["base_id"])
        if base_id not in base_routes:
            raise ValueError(f"H1 metamorphic route base missing: {base_id}")
        family = str(item["family"])
        if family not in FAMILY_TO_INDEX:
            raise ValueError(f"unknown H1 metamorphic route family: {family}")
        base_route = int(base_routes[base_id])
        transformed_route = int(transformed_routes[transformed_id])
        target = FAMILY_TO_INDEX[family]
        rows.append(
            {
                "base_id": base_id,
                "transformed_id": transformed_id,
                "transform": str(item["transform"]),
                "family": family,
                "base_route": base_route,
                "transformed_route": transformed_route,
                "target": target,
                "consistent": transformed_route == base_route,
                "base_exact": base_route == target,
                "transformed_exact": transformed_route == target,
            }
        )

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        if not count:
            return {
                "count": 0,
                "consistency": 0.0,
                "base_exact": 0.0,
                "transformed_exact": 0.0,
            }
        return {
            "count": count,
            "consistency": sum(bool(row["consistent"]) for row in selected) / count,
            "base_exact": sum(bool(row["base_exact"]) for row in selected) / count,
            "transformed_exact": sum(
                bool(row["transformed_exact"]) for row in selected
            )
            / count,
        }

    transforms = sorted({str(row["transform"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})
    return {
        "aggregate": summarize(rows),
        "by_transform": {
            name: summarize([row for row in rows if row["transform"] == name])
            for name in transforms
        },
        "by_family": {
            name: summarize([row for row in rows if row["family"] == name])
            for name in families
        },
        "route_relation_sha256": _sha256_json(
            [
                {
                    "base_id": row["base_id"],
                    "transformed_id": row["transformed_id"],
                    "consistent": row["consistent"],
                    "transformed_exact": row["transformed_exact"],
                }
                for row in rows
            ]
        ),
        "rows": rows,
    }


def strip_and_save(
    model: H1LatentReasoner,
    checkpoint_path: Path,
    cache: TokenCache,
    equivalence_split: PreparedSplit,
    *,
    device: str,
) -> dict[str, Any]:
    before = evaluate_model(
        model,
        cache,
        equivalence_split,
        device=device,
        include_trace=False,
    )
    deployment_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.deployment_state().items()
    }
    model.strip_trace_head()
    after = evaluate_model(
        model,
        cache,
        equivalence_split,
        device=device,
        include_trace=False,
    )
    checkpoint = {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.deployment-checkpoint.v1",
        "config": asdict(model.config),
        "state_dict": deployment_state,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    reloaded = load_deployment_checkpoint(checkpoint_path, device=device)
    reload_evaluation = evaluate_model(
        reloaded,
        cache,
        equivalence_split,
        device=device,
        include_trace=False,
    )
    equivalent = (
        before["predictions"] == after["predictions"]
        and before["answer"]["prediction_sha256"] == after["answer"]["prediction_sha256"]
    )
    reload_equivalent = (
        before["predictions"] == reload_evaluation["predictions"]
        and before["answer"]["prediction_sha256"]
        == reload_evaluation["answer"]["prediction_sha256"]
    )
    return {
        "predictions_equal": equivalent,
        "reload_predictions_equal": reload_equivalent,
        "before_sha256": before["answer"]["prediction_sha256"],
        "after_sha256": after["answer"]["prediction_sha256"],
        "reload_sha256": reload_evaluation["answer"]["prediction_sha256"],
        "trace_head_present": model.trace_head is not None,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper(),
        "passed": equivalent and reload_equivalent and model.trace_head is None,
    }


def load_deployment_checkpoint(path: Path, *, device: str) -> H1LatentReasoner:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = H1LatentReasoner(H1Config(**checkpoint["config"]))
    model.strip_trace_head()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device)


def matched_initialization_report(
    shared: H1LatentReasoner,
    mixed: H1LatentReasoner,
) -> dict[str, Any]:
    shared_state = shared.state_dict()
    mixed_state = mixed.state_dict()
    common = sorted(set(shared_state) & set(mixed_state))
    mismatches = [name for name in common if not torch.equal(shared_state[name], mixed_state[name])]
    shared_ffn_copies = []
    routed_feature_trunk_copies = []
    routed_projection_copies = []
    for layer_index, (left, right) in enumerate(zip(shared.layers, mixed.layers, strict=True)):
        shared_equal = all(
            torch.equal(left_value, right_value)
            for left_value, right_value in zip(
                left.shared_ffn.state_dict().values(),
                right.shared_ffn.state_dict().values(),
                strict=True,
            )
        )
        shared_ffn_copies.append({"layer": layer_index, "equal": shared_equal})
        feature_trunk_equal = all(
            torch.equal(left_value, right_value)
            for left_value, right_value in zip(
                left.routed_feature_trunk.state_dict().values(),
                right.routed_feature_trunk.state_dict().values(),
                strict=True,
            )
        )
        routed_feature_trunk_copies.append(
            {"layer": layer_index, "equal": feature_trunk_equal}
        )
        for expert_index in (0, 1):
            equal = all(
                torch.equal(left_value, right_value)
                for left_value, right_value in zip(
                    left.routed_projections[0].state_dict().values(),
                    right.routed_projections[expert_index].state_dict().values(),
                    strict=True,
                )
            )
            routed_projection_copies.append(
                {"layer": layer_index, "expert": expert_index, "equal": equal}
            )
    return {
        "common_parameter_count": len(common),
        "common_mismatches": mismatches,
        "shared_ffn_copies": shared_ffn_copies,
        "routed_feature_trunk_copies": routed_feature_trunk_copies,
        "routed_projection_copies": routed_projection_copies,
        "passed": (
            not mismatches
            and all(row["equal"] for row in shared_ffn_copies)
            and all(row["equal"] for row in routed_feature_trunk_copies)
            and all(row["equal"] for row in routed_projection_copies)
        ),
    }


def build_pair(config: H1Config, *, seed: int) -> tuple[H1LatentReasoner, H1LatentReasoner, dict[str, Any]]:
    shared, mixed = build_matched_pair(config, seed=seed)
    report = matched_initialization_report(shared, mixed)
    return shared, mixed, report


__all__ = [
    "FAMILY_TO_INDEX",
    "INTERVENTIONS",
    "LABEL_TO_INDEX",
    "PreparedSplit",
    "build_balanced_schedule",
    "build_pair",
    "evaluate_model",
    "load_deployment_checkpoint",
    "matched_initialization_report",
    "materialize_causal_records",
    "metamorphic_consistency",
    "metamorphic_route_consistency",
    "prepare_materialized_split",
    "prepare_split",
    "schedule_audit",
    "strip_and_save",
    "train_fixed_budget",
]
