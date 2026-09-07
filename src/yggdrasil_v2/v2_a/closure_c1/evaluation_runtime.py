"""Batched, metadata-closed runtime for C1 evaluation.

The cache is queried by example ID, while labels and causal metadata remain in
the offline record list.  Every model call receives only ``source_hidden`` and
``source_mask`` plus an explicitly requested intervention.  This separation is
important: using a cached metadata row as a forward input would invalidate the
source-only C1 contract.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

import torch

from .evaluate import evaluate_behavior, evaluate_causal, evaluate_paired_intervention
from .metrics import cluster_bootstrap_mean, finite_json, wilson_interval


class ModelProtocol(Protocol):
    def __call__(self, source_hidden: torch.Tensor, source_mask: torch.Tensor, **kwargs: Any) -> Any: ...


class CacheProtocol(Protocol):
    def get_batch(self, example_ids: Sequence[str], *, device: str | torch.device = "cpu") -> tuple[Mapping[str, torch.Tensor], list[Mapping[str, Any]]]: ...


INTERVENTIONS = ("baseline", "zero_hidden", "shuffled_hidden", "within_family_shuffled_hidden", "shuffle_hidden", "h0_no_core", "no_core", "step5_state_shuffle", "step5_within_family_state_shuffle", "slot_permutation", "initial_slot_permutation")
_INTERVENTION_ALIASES = {
    "within_family_shuffled_hidden": "shuffled_hidden",
    "shuffle_hidden": "shuffled_hidden",
    "no_core": "h0_no_core",
    "step5_within_family_state_shuffle": "step5_state_shuffle",
    "initial_slot_permutation": "slot_permutation",
}


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("example_id")
    if not isinstance(value, str) or not value:
        raise ValueError("offline records require non-empty example_id")
    return value


def _cell(record: Mapping[str, Any]) -> tuple[str, str]:
    family, split = record.get("family"), record.get("split")
    if not isinstance(family, str) or not isinstance(split, str):
        raise ValueError("offline records require family and split")
    return family, split


def _tensor(value: Any, *, device: str | torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device) if device is not None else value
    return torch.as_tensor(value, device=device)


def _logits(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping):
        value = output.get("logits", output.get("answer_logits"))
    else:
        value = getattr(output, "logits", None)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim != 2 or value.shape[-1] != 9:
        raise ValueError("C1 runtime requires raw answer logits [batch,9]")
    if not bool(torch.isfinite(value.float()).all().item()):
        raise ValueError("non-finite answer logits")
    return value.detach().float().cpu()


def _trajectory(output: Any) -> torch.Tensor:
    value = output.get("trajectory") if isinstance(output, Mapping) else getattr(output, "trajectory", None)
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError("C1 trace runtime requires trajectory [batch,T,slots,width]")
    return value


def _cycle(batch_size: int, device: str | torch.device) -> torch.Tensor:
    if batch_size < 2:
        raise ValueError("within-family cyclic intervention needs at least two records per batch")
    mapping = torch.roll(torch.arange(batch_size, device=device, dtype=torch.long), shifts=-1)
    if bool((mapping == torch.arange(batch_size, device=device)).any().item()):
        raise AssertionError("cyclic mapping unexpectedly contains a fixed point")
    return mapping


def _validate_batch(batch: Mapping[str, torch.Tensor], ids: Sequence[str]) -> None:
    if set(batch) != {"source_hidden", "source_mask"}:
        raise ValueError("model batch contains fields other than source_hidden/source_mask")
    hidden, mask = batch["source_hidden"], batch["source_mask"]
    if not isinstance(hidden, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("cache batch values must be tensors")
    if hidden.shape[0] != len(ids) or mask.shape[:2] != hidden.shape[:2]:
        raise ValueError("cache batch size/mask shape mismatch")
    if mask.dtype is not torch.bool:
        raise ValueError("source_mask must be bool")


class C1EvaluationRuntime:
    """Evaluate a C1 model through a cache without exposing metadata to it."""

    def __init__(self, model: ModelProtocol, dataset: CacheProtocol, records: Iterable[Mapping[str, Any]], *, device: str | torch.device = "cpu", batch_size: int = 8):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.dataset = dataset
        self.records = list(records)
        self.device = device
        self.batch_size = batch_size
        self.by_id: dict[str, Mapping[str, Any]] = {}
        for record in self.records:
            key = _record_id(record)
            if key in self.by_id:
                raise ValueError(f"duplicate offline example_id: {key}")
            _cell(record)
            self.by_id[key] = record

    def _groups(self, records: Sequence[Mapping[str, Any]] | None = None) -> list[list[Mapping[str, Any]]]:
        chosen = list(self.records if records is None else records)
        cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        cell_order: list[tuple[str, str]] = []
        for record in chosen:
            cell = _cell(record)
            if cell not in cells:
                cell_order.append(cell)
            cells[cell].append(record)
        groups: list[list[Mapping[str, Any]]] = []
        for cell in cell_order:
            rows = cells[cell]
            groups.extend(rows[start : start + self.batch_size] for start in range(0, len(rows), self.batch_size))
        return groups

    @contextmanager
    def _inference(self):
        was_training = getattr(self.model, "training", None)
        if hasattr(self.model, "eval"):
            self.model.eval()
        try:
            with torch.inference_mode():
                yield
        finally:
            if was_training is True and hasattr(self.model, "train"):
                self.model.train()

    def _batch(self, rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, torch.Tensor], list[str]]:
        ids = [_record_id(row) for row in rows]
        batch, _cache_metadata = self.dataset.get_batch(ids, device=self.device)
        _validate_batch(batch, ids)
        # Assert the scheduler invariant from the offline records, not cache
        # metadata; cache metadata is never trusted as model input.
        if len({_cell(row) for row in rows}) != 1:
            raise AssertionError("a runtime batch crossed family/cell boundary")
        return batch, ids

    def _forward(self, batch: Mapping[str, torch.Tensor], *, intervention: str = "baseline", return_trajectory: bool = False, slot_permutation: Sequence[int] | torch.Tensor | None = None) -> Any:
        if intervention not in INTERVENTIONS:
            raise ValueError(f"unknown C1 intervention: {intervention}")
        intervention = _INTERVENTION_ALIASES.get(intervention, intervention)
        hidden = batch["source_hidden"]
        source_mask = batch["source_mask"]
        kwargs: dict[str, Any] = {"return_trajectory": return_trajectory}
        if intervention == "zero_hidden":
            hidden = torch.zeros_like(hidden)
        elif intervention in {"shuffled_hidden", "step5_state_shuffle"}:
            mapping = _cycle(hidden.shape[0], hidden.device)
            if intervention == "shuffled_hidden":
                hidden = hidden.index_select(0, mapping)
                # A cached source row is the pair (full-token hidden, padding
                # structure).  Move the complete row so padding never exposes
                # or truncates values while the offline target stays fixed.
                source_mask = source_mask.index_select(0, mapping)
            else:
                kwargs.update(state_shuffle_step=5, state_shuffle_indices=mapping)
        elif intervention == "slot_permutation":
            if slot_permutation is None:
                slots = int(getattr(getattr(self.model, "config", None), "slots", 8))
                slot_permutation = list(range(slots - 1, -1, -1))
            kwargs["initial_slot_permutation"] = _tensor(slot_permutation, device=hidden.device).long()
        if intervention == "h0_no_core":
            encode = getattr(self.model, "encode", None)
            readout = getattr(self.model, "logits_from_state", None)
            if not callable(encode) or not callable(readout):
                raise TypeError("H0/no-core requires model.encode and model.logits_from_state")
            return {"logits": readout(encode(hidden, source_mask))}
        return self.model(hidden, source_mask, **kwargs)

    def raw_answer_logits(self, *, intervention: str = "baseline", records: Sequence[Mapping[str, Any]] | None = None, slot_permutation: Sequence[int] | torch.Tensor | None = None) -> dict[str, list[float]]:
        chosen = list(self.records if records is None else records)
        output: dict[str, list[float]] = {}
        with self._inference():
            for rows in self._groups(chosen):
                batch, ids = self._batch(rows)
                logits = _logits(self._forward(batch, intervention=intervention, slot_permutation=slot_permutation))
                if logits.shape[0] != len(ids):
                    raise ValueError("model logits batch size mismatch")
                output.update({key: row.tolist() for key, row in zip(ids, logits)})
        return output

    def _predictor(self, logits: Mapping[str, Sequence[float]]):
        return lambda record: {"logits": logits[_record_id(record)]}

    def evaluate_answers(self, *, intervention: str = "baseline", records: Sequence[Mapping[str, Any]] | None = None, slot_permutation: Sequence[int] | torch.Tensor | None = None) -> dict[str, Any]:
        chosen = list(self.records if records is None else records)
        logits = self.raw_answer_logits(intervention=intervention, records=chosen, slot_permutation=slot_permutation)
        return evaluate_behavior(chosen, self._predictor(logits))

    def evaluate_causal(self, *, intervention: str = "baseline", records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        chosen = list(self.records if records is None else records)
        logits = self.raw_answer_logits(intervention=intervention, records=chosen)
        return evaluate_causal(chosen, self._predictor(logits))

    def evaluate_intervention(self, intervention: str, *, records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        if intervention == "baseline":
            raise ValueError("paired intervention requires a non-baseline intervention")
        chosen = list(self.records if records is None else records)
        baseline = self.raw_answer_logits(records=chosen)
        changed = self.raw_answer_logits(intervention=intervention, records=chosen)
        return evaluate_paired_intervention(chosen, self._predictor(baseline), self._predictor(changed), cluster_key="pair_id" if any(row.get("pair_id") is not None for row in chosen) else "example_id")

    def _trace_entry(self, example_id: str, trace_bank: Mapping[str, Any]) -> Any:
        entry = trace_bank.get(example_id)
        if entry is None:
            raise KeyError(f"trace bank missing {example_id}")
        return entry

    @staticmethod
    def _trace_field(entry: Any, *names: str) -> Any:
        for name in names:
            if isinstance(entry, Mapping) and name in entry:
                return entry[name]
            if hasattr(entry, name):
                return getattr(entry, name)
            try:
                return entry[name]
            except (KeyError, TypeError, AttributeError):
                pass
        raise KeyError(f"trace bank entry missing one of {names}")

    def trace_credit(self, trace_bank: Mapping[str, Any], *, validation_records: Sequence[Mapping[str, Any]] | None = None, per_family: int = 256, bootstrap_seed: int = 2026082503) -> dict[str, Any]:
        values, _nll = self._stream_trace_statistics(
            trace_bank,
            validation_records=validation_records,
            per_family=per_family,
        )
        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            families[str(row["family"])].append(row)
        report: dict[str, Any] = {"records": len(values), "families": {}}
        for family, group in sorted(families.items()):
            all_count = sum(int(row["all_count"]) for row in group)
            all_hits = sum(int(row["all_hits"]) for row in group)
            content_count = sum(int(row["content_count"]) for row in group)
            content_hits = sum(int(row["content_hits"]) for row in group)
            margins = [float(row["margin"]) for row in group]
            correct_nll = sum(float(row["correct_nll"]) for row in group)
            wrong_nll = sum(float(row["wrong_nll"]) for row in group)
            ids = [row["example_id"] for row in group]
            report["families"][family] = {
                "records": len(group),
                "all_token_accuracy": wilson_interval(all_hits, all_count),
                "content_token_accuracy": wilson_interval(content_hits, content_count),
                "nll_margin": {
                    # The frozen point estimate is pooled nat/token.  The
                    # interval remains record-clustered so a handful of long
                    # traces cannot manufacture confidence.
                    "point": (wrong_nll - correct_nll) / all_count
                    if all_count
                    else 0.0,
                    "record_mean_point": sum(margins) / len(margins)
                    if margins
                    else 0.0,
                    "correct_nll_sum": correct_nll,
                    "wrong_nll_sum": wrong_nll,
                    "tokens": all_count,
                    "bootstrap": cluster_bootstrap_mean(
                        margins, ids, seed=bootstrap_seed
                    ),
                },
                "record_bootstrap": cluster_bootstrap_mean(
                    [
                        float(row["all_hits"]) / int(row["all_count"])
                        if int(row["all_count"])
                        else 0.0
                        for row in group
                    ],
                    ids,
                    seed=bootstrap_seed,
                ),
            }
        return finite_json(report)

    def trace_nll_by_family(
        self,
        trace_bank: Mapping[str, Any],
        *,
        validation_records: Sequence[Mapping[str, Any]] | None = None,
        per_family: int = 256,
    ) -> dict[str, float]:
        """Streaming trace-only checkpoint-selection metric."""
        _values, nll = self._stream_trace_statistics(
            trace_bank,
            validation_records=validation_records,
            per_family=per_family,
            include_wrong_owner=False,
        )
        return nll

    def _selected_trace_records(
        self,
        validation_records: Sequence[Mapping[str, Any]] | None,
        per_family: int,
    ) -> list[Mapping[str, Any]]:
        chosen = list(validation_records if validation_records is not None else [row for row in self.records if row.get("split") == "validation"])
        by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in chosen: by_family[str(row.get("family"))].append(row)
        selected: list[Mapping[str, Any]] = []
        for family in ("ERE", "CPS"):
            if len(by_family[family]) < per_family:
                raise ValueError(f"trace credit requires {per_family} validation records for {family}")
            selected.extend(
                sorted(by_family[family], key=_record_id)[:per_family]
            )
        return selected

    def _stream_trace_statistics(
        self,
        trace_bank: Mapping[str, Any],
        *,
        validation_records: Sequence[Mapping[str, Any]] | None,
        per_family: int,
        include_wrong_owner: bool = True,
        token_chunk: int = 64,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        selected = self._selected_trace_records(validation_records, per_family)
        values: list[dict[str, Any]] = []
        with self._inference():
            trajectories: dict[str, torch.Tensor] = {}
            for rows in self._groups(selected):
                batch, ids = self._batch(rows)
                result = self._forward(batch, return_trajectory=True)
                trajectory = _trajectory(result)
                for index, key in enumerate(ids):
                    trajectories[key] = trajectory[index].detach().float().cpu().clone()
            by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in selected:
                by_family[str(row.get("family"))].append(row)
            next_id: dict[str, str] = {}
            for family_rows in by_family.values():
                if len(family_rows) < 2:
                    raise ValueError("within-family cyclic wrong-owner trace requires at least two records")
                for index, row in enumerate(family_rows):
                    next_id[_record_id(row)] = _record_id(family_rows[(index + 1) % len(family_rows)])
            for rows in self._groups(selected):
                ids = [_record_id(row) for row in rows]
                entries = [self._trace_entry(key, trace_bank) for key in ids]
                lengths = [len(self._trace_field(entry, "token_ids", "target_ids", "local_token_ids")) for entry in entries]
                if not lengths or any(length <= 0 for length in lengths):
                    raise ValueError("trace targets must be non-empty")
                current_trajectory = torch.stack([trajectories[key] for key in ids]).to(self.device)
                owner_trajectory = None
                if include_wrong_owner:
                    owner_ids = [next_id[key] for key in ids]
                    if any(_cell(self.by_id[owner]) != _cell(row) for owner, row in zip(owner_ids, rows)):
                        raise AssertionError("wrong-owner cycle crossed a family/cell boundary")
                    owner_trajectory = torch.stack([trajectories[key] for key in owner_ids]).to(self.device)
                accumulators = [
                    {
                        "family": row.get("family"),
                        "example_id": key,
                        "all_hits": 0,
                        "all_count": 0,
                        "content_hits": 0,
                        "content_count": 0,
                        "correct_nll": 0.0,
                        "wrong_nll": 0.0,
                    }
                    for row, key in zip(rows, ids)
                ]
                for start in range(0, max(lengths), token_chunk):
                    width = min(token_chunk, max(lengths) - start)
                    steps = torch.ones((len(rows), width), dtype=torch.long, device=self.device)
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
                        steps[index, :count] = _tensor(self._trace_field(entry, "step_indices", "step_ids", "steps")[start:stop], device=self.device).long()
                        global_positions[index, :count] = _tensor(self._trace_field(entry, "global_positions", "global_ids")[start:stop], device=self.device).long()
                        local_positions[index, :count] = _tensor(self._trace_field(entry, "local_positions", "local_ids")[start:stop], device=self.device).long()
                        targets[index, :count] = _tensor(self._trace_field(entry, "token_ids", "target_ids", "local_token_ids")[start:stop], device=self.device).long()
                        grammar[index, :count] = _tensor(self._trace_field(entry, "grammar_mask", "grammar")[start:stop], device=self.device).bool()
                    correct_logits = self.model.trace_logits(
                        current_trajectory, steps, global_positions, local_positions
                    ).float()
                    correct_log_probs = torch.log_softmax(correct_logits, dim=-1)
                    correct_loss = -correct_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    predictions = correct_logits.argmax(dim=-1)
                    wrong_loss = None
                    if owner_trajectory is not None:
                        wrong_logits = self.model.trace_logits(
                            owner_trajectory, steps, global_positions, local_positions
                        ).float()
                        wrong_loss = -torch.log_softmax(wrong_logits, dim=-1).gather(
                            -1, targets.unsqueeze(-1)
                        ).squeeze(-1)
                    for index, accumulator in enumerate(accumulators):
                        active = mask[index]
                        content = active & ~grammar[index]
                        accumulator["all_count"] += int(active.sum().cpu())
                        accumulator["all_hits"] += int(((predictions[index] == targets[index]) & active).sum().cpu())
                        accumulator["content_count"] += int(content.sum().cpu())
                        accumulator["content_hits"] += int(((predictions[index] == targets[index]) & content).sum().cpu())
                        accumulator["correct_nll"] += float(correct_loss[index][active].sum().cpu())
                        if wrong_loss is not None:
                            accumulator["wrong_nll"] += float(wrong_loss[index][active].sum().cpu())
                for accumulator in accumulators:
                    count = int(accumulator["all_count"])
                    accumulator["margin"] = (
                        (float(accumulator["wrong_nll"]) - float(accumulator["correct_nll"])) / count
                        if include_wrong_owner and count
                        else 0.0
                    )
                    values.append(accumulator)
                del current_trajectory, owner_trajectory
        nll: dict[str, float] = {}
        for family in ("ERE", "CPS"):
            rows = [row for row in values if row["family"] == family]
            count = sum(int(row["all_count"]) for row in rows)
            if not count:
                raise ValueError(f"no trace tokens for {family}")
            nll[family] = sum(float(row["correct_nll"]) for row in rows) / count
        return values, nll


def evaluate_runtime(model: ModelProtocol, dataset: CacheProtocol, records: Iterable[Mapping[str, Any]], *, device: str | torch.device = "cpu", batch_size: int = 8, intervention: str = "baseline") -> dict[str, Any]:
    return C1EvaluationRuntime(model, dataset, records, device=device, batch_size=batch_size).evaluate_answers(intervention=intervention)


run_batched_evaluation = evaluate_runtime
EvaluationRuntime = C1EvaluationRuntime


__all__ = ["CacheProtocol", "C1EvaluationRuntime", "EvaluationRuntime", "INTERVENTIONS", "ModelProtocol", "evaluate_runtime", "run_batched_evaluation"]
