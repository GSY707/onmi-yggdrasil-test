from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from transformers import StoppingCriteria, StoppingCriteriaList

from .a1_7_data import VALUE_LABELS
from .a1_10_cache import QWEN_MODEL_ID, QWEN_REVISION
from .a1_13_train import _write_json
from .a1_19h_h2_data import load_a119h2_records
from .a1_20b_cache import (
    A120BCachedSplit,
    CACHE_SCHEMA,
    MANIFEST_SCHEMA as CACHE_MANIFEST_SCHEMA,
    _encode_batch,
    collate_a120b_items,
    file_sha256,
)
from .a1_20b_train import encode_a120b_labels
from .a1_20b_train import evaluate_a120b_matrix
from .a1_20c_model import (
    OPERATION_POTENTIAL_VOTERS,
    OPERATION_SUPPORTED_GAIN_THRESHOLD,
    _operation_count_from_gains,
    _shared_entity_viterbi_decode,
)
from .a1_20c_supervision import A120CSupervisionSplit
from .a1_20c_train import load_a120c_checkpoint
from .model import load_qwen35_text_only


PREFLIGHT_SCHEMA = "yggdrasil.v2-a1.21p.current-family-preflight.v1"
ASSESSMENT_SCHEMA = "yggdrasil.v2-a1.21p.assessment.v1"
FINAL_RE = re.compile(r"\bFINAL\s*:\s*([A-J])\b", re.IGNORECASE)
BARE_FINAL_RE = re.compile(r"^\s*([A-J])\s*[.!]?\s*$", re.IGNORECASE)
STEP_RE = re.compile(r"\bSTEP\s*(\d+)\s*:\s*([^\n\r]+)", re.IGNORECASE)
PAIR_RE = re.compile(r"\b(object_[0-9a-f]+)\s*=\s*([A-J])\b", re.IGNORECASE)


def _operation_count_from_gain_drop(
    triple_gains: torch.Tensor,
    *,
    supported_gain_threshold: float,
    crossing_gain_threshold: float,
    drop_threshold: float,
    potential_voters: int,
) -> torch.Tensor:
    """Use absolute supported gates and a relative recurrent stop cliff."""
    if triple_gains.ndim != 2:
        raise ValueError("operation gain-drop input must be [B,O-1]")
    supported_end = max(0, potential_voters - 1)
    supported = (
        triple_gains[:, :supported_end] > supported_gain_threshold
    ).long().cumprod(dim=-1)
    counts = 1 + supported.sum(dim=-1)
    if triple_gains.shape[1] <= supported_end:
        return counts
    crossing = (
        (counts == potential_voters)
        & (
            triple_gains[:, supported_end]
            > crossing_gain_threshold
        )
    )
    recurrent = triple_gains[:, supported_end:]
    if recurrent.shape[1] == 1:
        return torch.where(
            crossing,
            counts.new_full((), potential_voters + 1),
            counts,
        )
    drops = recurrent[:, :-1] - recurrent[:, 1:]
    largest_drop, drop_index = drops.max(dim=-1)
    stopped_count = potential_voters + 1 + drop_index
    recurrent_count = torch.where(
        largest_drop > drop_threshold,
        stopped_count,
        counts.new_full((), triple_gains.shape[1] + 1),
    )
    return torch.where(crossing, recurrent_count, counts)


@torch.inference_mode()
def diagnose_a121p_entity_capacity_threshold(
    checkpoint_path: Path,
    cache_dir: Path,
    data_dir: Path,
    output_path: Path,
    *,
    splits: Sequence[str],
    capacity_thresholds: Sequence[float],
    family_filter: str | None = None,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    """Re-decode the N4-to-N5 crossing from fixed entity pair gains."""
    if not splits or not capacity_thresholds:
        raise ValueError("entity-capacity diagnostic requires inputs")
    if family_filter not in {None, "canonical", "routing"}:
        raise ValueError(f"unknown entity-capacity family {family_filter}")
    model = load_a120c_checkpoint(checkpoint_path, device)
    counters = {
        float(threshold): {
            "examples": 0,
            "correct": 0,
            "by_count": defaultdict(lambda: {"examples": 0, "correct": 0}),
        }
        for threshold in capacity_thresholds
    }
    model.eval()
    for split in splits:
        dataset = A120BCachedSplit(cache_dir, data_dir, split)
        selected = [
            index
            for index, record in enumerate(dataset.records)
            if family_filter is None
            or (
                str(record.get("task_family", "")).startswith(
                    "signal_routing"
                )
                == (family_filter == "routing")
            )
        ]
        for start in range(0, len(selected), batch_size):
            indices = selected[start : start + batch_size]
            inputs, records = collate_a120b_items(
                dataset.items(indices), device
            )
            gains = model(**inputs)["entity_pair_score_gains"]
            targets = torch.tensor(
                [int(record["entity_count"]) for record in records],
                device=gains.device,
            )
            supported = (
                gains[:, :-1] > 12.0
            ).long().cumprod(dim=-1)
            base_counts = 1 + supported.sum(dim=-1)
            for raw_threshold in capacity_thresholds:
                threshold = float(raw_threshold)
                predictions = torch.where(
                    (base_counts == gains.shape[1])
                    & (gains[:, -1] > threshold),
                    base_counts + 1,
                    base_counts,
                )
                correct = predictions == targets
                row = counters[threshold]
                row["examples"] += len(records)
                row["correct"] += int(correct.sum())
                for index, record in enumerate(records):
                    count = int(record["entity_count"])
                    count_row = row["by_count"][count]
                    count_row["examples"] += 1
                    count_row["correct"] += int(correct[index])
    rows = []
    for threshold in sorted(counters):
        counter = counters[threshold]
        rows.append(
            {
                "capacity_gain_threshold": threshold,
                "examples": counter["examples"],
                "entity_count_accuracy": (
                    counter["correct"] / counter["examples"]
                ),
                "by_entity_count": {
                    str(count): {
                        **row,
                        "accuracy": row["correct"] / row["examples"],
                    }
                    for count, row in sorted(counter["by_count"].items())
                },
            }
        )
    result = {
        "schema_version": "yggdrasil.v2-a1.21p.entity-capacity.v1",
        "stage": "A1.21P entity capacity crossing diagnosis",
        "checkpoint": str(checkpoint_path.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "splits": list(splits),
        "family_filter": family_filter,
        "supported_gain_threshold": 12.0,
        "training_performed": False,
        "rows": rows,
    }
    _write_json(output_path, result)
    return result


def interpolate_a121p_boundary_checkpoints(
    left_checkpoint: Path,
    right_checkpoint: Path,
    output_path: Path,
    *,
    right_weight: float,
    prefixes: Sequence[str],
) -> dict[str, Any]:
    """Interpolate selected Boundary tensors while preserving the frozen core."""
    if not 0.0 <= right_weight <= 1.0:
        raise ValueError("checkpoint interpolation weight must be in [0, 1]")
    if not prefixes:
        raise ValueError("checkpoint interpolation requires prefixes")
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint {output_path}")
    left = torch.load(
        left_checkpoint, map_location="cpu", weights_only=False
    )
    right = torch.load(
        right_checkpoint, map_location="cpu", weights_only=False
    )
    for field in (
        "schema_version",
        "config",
        "core_state_sha256",
        "core_checkpoint_sha256",
    ):
        if left[field] != right[field]:
            raise ValueError(f"checkpoint interpolation mismatch: {field}")
    merged = deepcopy(left)
    selected = []
    for name, left_tensor in left["boundary"].items():
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        right_tensor = right["boundary"][name]
        if left_tensor.shape != right_tensor.shape:
            raise ValueError(f"checkpoint interpolation shape mismatch: {name}")
        if torch.is_floating_point(left_tensor):
            merged["boundary"][name] = (
                left_tensor * (1.0 - right_weight)
                + right_tensor * right_weight
            )
        else:
            if not torch.equal(left_tensor, right_tensor):
                raise ValueError(
                    f"cannot interpolate non-floating tensor {name}"
                )
        selected.append(name)
    if not selected:
        raise ValueError("checkpoint interpolation selected no tensors")
    merged["a121p_interpolation"] = {
        "left_checkpoint": str(left_checkpoint.resolve()),
        "right_checkpoint": str(right_checkpoint.resolve()),
        "right_weight": right_weight,
        "prefixes": list(prefixes),
        "selected_tensor_count": len(selected),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    return {
        "schema_version": "yggdrasil.v2-a1.21p.interpolation.v1",
        "output": str(output_path.resolve()),
        **merged["a121p_interpolation"],
    }


@torch.inference_mode()
def diagnose_a121p_operation_threshold(
    checkpoint_path: Path,
    cache_dir: Path,
    data_dir: Path,
    output_path: Path,
    *,
    splits: Sequence[str],
    recurrent_thresholds: Sequence[float],
    drop_thresholds: Sequence[float] = (),
    crossing_threshold: float = 16.1,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    """Re-decode operation counts from fixed gains without retraining."""
    if not splits or not (recurrent_thresholds or drop_thresholds):
        raise ValueError("operation-threshold diagnostic requires inputs")
    model = load_a120c_checkpoint(checkpoint_path, device)
    counters = {
        ("absolute", float(threshold)): {
            "examples": 0,
            "correct": 0,
            "by_length": defaultdict(lambda: {"examples": 0, "correct": 0}),
        }
        for threshold in recurrent_thresholds
    }
    counters.update(
        {
            ("relative_drop", float(threshold)): {
                "examples": 0,
                "correct": 0,
                "by_length": defaultdict(
                    lambda: {"examples": 0, "correct": 0}
                ),
            }
            for threshold in drop_thresholds
        }
    )
    model.eval()
    for split in splits:
        dataset = A120BCachedSplit(cache_dir, data_dir, split)
        for start in range(0, len(dataset), batch_size):
            indices = list(
                range(start, min(start + batch_size, len(dataset)))
            )
            inputs, records = collate_a120b_items(
                dataset.items(indices), device
            )
            gains = model(**inputs)["operation_triple_score_gains"]
            targets = torch.tensor(
                [int(record["program_length"]) for record in records],
                device=gains.device,
            )
            for raw_threshold in recurrent_thresholds:
                threshold = float(raw_threshold)
                predictions = _operation_count_from_gains(
                    gains,
                    supported_gain_threshold=(
                        OPERATION_SUPPORTED_GAIN_THRESHOLD
                    ),
                    recurrent_gain_threshold=threshold,
                    potential_voters=OPERATION_POTENTIAL_VOTERS,
                )
                correct = predictions == targets
                row = counters[("absolute", threshold)]
                row["examples"] += len(records)
                row["correct"] += int(correct.sum())
                for index, record in enumerate(records):
                    length = int(record["program_length"])
                    length_row = row["by_length"][length]
                    length_row["examples"] += 1
                    length_row["correct"] += int(correct[index])
            for raw_threshold in drop_thresholds:
                threshold = float(raw_threshold)
                predictions = _operation_count_from_gain_drop(
                    gains,
                    supported_gain_threshold=(
                        OPERATION_SUPPORTED_GAIN_THRESHOLD
                    ),
                    crossing_gain_threshold=crossing_threshold,
                    drop_threshold=threshold,
                    potential_voters=OPERATION_POTENTIAL_VOTERS,
                )
                correct = predictions == targets
                row = counters[("relative_drop", threshold)]
                row["examples"] += len(records)
                row["correct"] += int(correct.sum())
                for index, record in enumerate(records):
                    length = int(record["program_length"])
                    length_row = row["by_length"][length]
                    length_row["examples"] += 1
                    length_row["correct"] += int(correct[index])
    rows = []
    for decoder, threshold in sorted(counters):
        counter = counters[(decoder, threshold)]
        rows.append(
            {
                "decoder": decoder,
                "threshold": threshold,
                "recurrent_gain_threshold": (
                    threshold if decoder == "absolute" else None
                ),
                "drop_threshold": (
                    threshold if decoder == "relative_drop" else None
                ),
                "crossing_gain_threshold": (
                    (
                        OPERATION_SUPPORTED_GAIN_THRESHOLD + threshold
                    )
                    / 2.0
                    if decoder == "absolute"
                    else crossing_threshold
                ),
                "examples": counter["examples"],
                "operation_count_accuracy": (
                    counter["correct"] / counter["examples"]
                ),
                "by_length": {
                    str(length): {
                        **row,
                        "accuracy": row["correct"] / row["examples"],
                    }
                    for length, row in sorted(
                        counter["by_length"].items()
                    )
                },
            }
        )
    result = {
        "schema_version": "yggdrasil.v2-a1.21p.operation-threshold.v1",
        "stage": "A1.21P recurrent operation stopping diagnosis",
        "checkpoint": str(checkpoint_path.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "splits": list(splits),
        "supported_gain_threshold": OPERATION_SUPPORTED_GAIN_THRESHOLD,
        "potential_voters": OPERATION_POTENTIAL_VOTERS,
        "relative_drop_crossing_threshold": crossing_threshold,
        "training_performed": False,
        "rows": rows,
    }
    _write_json(output_path, result)
    return result


@torch.inference_mode()
def run_a121p_cross_domain_probe(
    checkpoint_path: Path,
    cache_dir: Path,
    data_dir: Path,
    output_path: Path,
    *,
    splits: Sequence[str],
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate one Boundary checkpoint on a different frozen-Qwen corpus."""
    if not splits:
        raise ValueError("A1.21P cross-domain probe requires splits")
    manifest = json.loads(
        (cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    missing = sorted(set(splits) - set(manifest["splits"]))
    if missing:
        raise ValueError(f"A1.21P cross-domain cache missing {missing}")
    model = load_a120c_checkpoint(checkpoint_path, device)
    results = {}
    for split in splits:
        dataset = A120BCachedSplit(cache_dir, data_dir, split)
        if dataset.hidden_width != model.config.source_width:
            raise ValueError("A1.21P cross-domain hidden-width mismatch")
        results[split] = evaluate_a120b_matrix(
            model, dataset, device, batch_size=batch_size
        )
    gates = {}
    for split, matrix in results.items():
        aggregate = matrix["aggregate"]
        gates[f"{split}_trajectory_at_least_0_95"] = (
            aggregate["trajectory_full_exact"] >= 0.95
        )
        gates[f"{split}_final_at_least_0_95"] = (
            aggregate["final_state_full_exact"] >= 0.95
        )
        gates[f"{split}_state_at_least_0_995"] = (
            aggregate["state_token_accuracy"] >= 0.995
        )
        gates[f"{split}_answer_at_least_0_95"] = (
            aggregate["final_answer_accuracy"] >= 0.95
        )
        gates[f"{split}_mapping_at_least_0_995"] = (
            aggregate["mapping_accuracy"]["minimum"] >= 0.995
        )
    gates["core_hash_unchanged"] = (
        model.integrity_report()["core_state_sha256"]
        == model.a120c_core_state_sha256
    )
    gates["frozen_qwen_full_hidden_no_oracle"] = (
        manifest.get("qwen_trainable_parameters") == 0
        and manifest.get("silent_truncation") is False
        and manifest.get("oracle_span_input") is False
        and manifest.get("oracle_role_tensor_input") is False
        and manifest.get("oracle_entity_mask_input") is False
        and manifest.get("oracle_operation_mask_input") is False
    )
    result = {
        "schema_version": "yggdrasil.v2-a1.21p.cross-domain-probe.v1",
        "stage": "A1.21P shared-Boundary cross-domain regression",
        "checkpoint": str(checkpoint_path.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "splits": results,
        "gates": gates,
        "passed": all(gates.values()),
        "formal_gate": False,
        "diagnostic_only": True,
        "training_performed": False,
        "checkpoint_cache_hash_override": False,
        "cross_domain_evaluation_intentional": True,
    }
    _write_json(output_path, result)
    return result


@torch.inference_mode()
def diagnose_a121p_entity_section_window(
    checkpoint_path: Path,
    cache_dir: Path,
    data_dir: Path,
    supervision_dir: Path,
    output_path: Path,
    *,
    windows: Sequence[int],
    pair_gain_thresholds: Sequence[float] = (5.0,),
    split: str = "validation",
    family_filter: str | None = None,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, Any]:
    """Measure a generic right-bounded entity-section prior without training."""
    if not windows or any(int(window) < 0 for window in windows):
        raise ValueError("entity-section windows must be non-negative")
    if not pair_gain_thresholds:
        raise ValueError("entity pair-gain thresholds must not be empty")
    model = load_a120c_checkpoint(checkpoint_path, device)
    dataset = A120BCachedSplit(cache_dir, data_dir, split)
    supervision = A120CSupervisionSplit(
        supervision_dir, dataset, split
    )
    counters = {
        (int(window), float(threshold)): {
            "examples": 0,
            "name_correct": 0,
            "value_correct": 0,
            "active_targets": 0,
            "entity_count_correct": 0,
            "entity_sequence_exact": 0,
            "operation_first_anchor_correct": 0,
        }
        for window in windows
        for threshold in pair_gain_thresholds
    }
    model.eval()
    if family_filter not in {None, "canonical", "routing"}:
        raise ValueError(f"unknown A1.21P family filter {family_filter}")
    selected_indices = [
        index
        for index, record in enumerate(dataset.records)
        if family_filter is None
        or (
            str(record.get("task_family", "")).startswith(
                "signal_routing"
            )
            == (family_filter == "routing")
        )
    ]
    if not selected_indices:
        raise ValueError("A1.21P family filter selected no examples")
    for start in range(0, len(selected_indices), batch_size):
        indices = selected_indices[start : start + batch_size]
        inputs, records = collate_a120b_items(
            dataset.items(indices), device
        )
        anchors = supervision.select(indices, device)
        output = model(**inputs)
        first_operation = output[
            "predicted_operation_anchor_positions"
        ][:, 0, 0]
        first_target = anchors.operation_role_targets[:, 0, 0]
        sequence = output["entity_name_anchor_logits"].shape[-1]
        positions = torch.arange(sequence, device=first_operation.device)
        active = anchors.entity_name_targets >= 0
        target_counts = torch.tensor(
            [int(record["entity_count"]) for record in records],
            device=first_operation.device,
        )
        for raw_window in windows:
            window = int(raw_window)
            name_logits = output["entity_name_anchor_logits"]
            value_logits = output["entity_value_anchor_logits"]
            if window:
                lower_bound = (first_operation - window).clamp_min(0)
                before_section = (
                    positions.unsqueeze(0) < lower_bound.unsqueeze(-1)
                )
                name_logits = name_logits.masked_fill(
                    before_section.unsqueeze(1), -torch.inf
                )
                value_logits = value_logits.masked_fill(
                    before_section.unsqueeze(1), -torch.inf
                )
            for raw_threshold in pair_gain_thresholds:
                threshold = float(raw_threshold)
                (
                    _,
                    _,
                    name_positions,
                    value_positions,
                    entity_mask,
                    _,
                ) = _shared_entity_viterbi_decode(
                    name_logits,
                    value_logits,
                    first_operation,
                    pair_gain_threshold=threshold,
                )
                name_correct = (
                    name_positions == anchors.entity_name_targets
                ) & active
                value_correct = (
                    value_positions == anchors.entity_value_targets
                ) & active
                sequence_exact = (
                    (name_correct | ~active)
                    & (value_correct | ~active)
                ).all(dim=-1)
                row = counters[(window, threshold)]
                row["examples"] += len(indices)
                row["name_correct"] += int(name_correct.sum())
                row["value_correct"] += int(value_correct.sum())
                row["active_targets"] += int(active.sum())
                row["entity_count_correct"] += int(
                    (entity_mask.sum(dim=-1) == target_counts).sum()
                )
                row["entity_sequence_exact"] += int(sequence_exact.sum())
                row["operation_first_anchor_correct"] += int(
                    (first_operation == first_target).sum()
                )
    rows = []
    for window, threshold in sorted(counters):
        row = counters[(window, threshold)]
        examples = row["examples"]
        targets = row["active_targets"]
        rows.append(
            {
                "window_tokens": window,
                "pair_gain_threshold": threshold,
                "meaning": (
                    "unbounded-prefix baseline"
                    if window == 0
                    else "only entity candidates within this many tokens "
                    "before the decoded first operation"
                ),
                "examples": examples,
                "entity_name_token_accuracy": (
                    row["name_correct"] / targets
                ),
                "entity_value_token_accuracy": (
                    row["value_correct"] / targets
                ),
                "entity_count_accuracy": (
                    row["entity_count_correct"] / examples
                ),
                "entity_anchor_sequence_exact": (
                    row["entity_sequence_exact"] / examples
                ),
                "first_operation_anchor_accuracy": (
                    row["operation_first_anchor_correct"] / examples
                ),
            }
        )
    result = {
        "schema_version": "yggdrasil.v2-a1.21p.entity-window-diagnostic.v1",
        "stage": "A1.21P Boundary entity-section diagnosis",
        "checkpoint": str(checkpoint_path.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "data_dir": str(data_dir.resolve()),
        "supervision_dir": str(supervision_dir.resolve()),
        "split": split,
        "family_filter": family_filter,
        "training_performed": False,
        "oracle_forward_input": False,
        "decoded_first_operation_is_forward_boundary": True,
        "rows": rows,
    }
    _write_json(output_path, result)
    return result


def parse_a121p_visible_trace(
    completion: str,
) -> tuple[str | None, dict[int, dict[str, str]]]:
    finals = list(FINAL_RE.finditer(completion))
    if finals:
        answer = finals[-1].group(1).upper()
    else:
        bare = BARE_FINAL_RE.fullmatch(completion)
        answer = bare.group(1).upper() if bare else None
    states: dict[int, dict[str, str]] = {}
    for match in STEP_RE.finditer(completion):
        pairs = {
            name.lower(): value.upper()
            for name, value in PAIR_RE.findall(match.group(2))
        }
        if pairs:
            states[int(match.group(1))] = pairs
    return answer, states


def _trace(record: dict[str, Any]) -> str:
    lines = []
    names = list(record["entity_names"])
    for step, state in enumerate(record["state_trajectory"], start=1):
        values = ", ".join(f"{name}={state[name]}" for name in names)
        lines.append(f"STEP {step}: {values}")
    lines.append(f"FINAL: {record['answer']}")
    return "\n".join(lines)


def build_a121p_prompt(
    record: dict[str, Any],
    mode: str,
    demonstrations: Sequence[dict[str, Any]] = (),
) -> str:
    if mode not in {"direct", "text_cot"}:
        raise ValueError(f"unknown A1.21P text mode {mode}")
    if mode == "direct":
        return (
            "Execute the object operations exactly. Reply with only "
            "`FINAL: X`, where X is one letter A-J. Do not explain.\n\n"
            f"QUESTION:\n{record['question']}\nANSWER:\n"
        )
    instruction = (
        "Execute every operation in order. After each operation, write one "
        "line `STEP n: object_id=X, ...` containing every object, then finish "
        "with `FINAL: X`. Do not skip or reorder steps.\n\n"
    )
    examples = ""
    for demonstration in demonstrations:
        examples += (
            f"EXAMPLE QUESTION:\n{demonstration['question']}\n"
            f"EXAMPLE ANSWER:\n{_trace(demonstration)}\n\n"
        )
    return (
        instruction
        + examples
        + f"QUESTION:\n{record['question']}\nANSWER:\n"
    )


def select_a121p_records(
    data_dir: Path,
    splits: Sequence[str],
    examples_per_split: int,
) -> list[dict[str, Any]]:
    if examples_per_split < 1:
        raise ValueError("A1.21P examples_per_split must be positive")
    selected: list[dict[str, Any]] = []
    for split in splits:
        cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in load_a119h2_records(data_dir, split):
            cell = (
                int(record["entity_count"]),
                int(record["program_length"]),
            )
            cells[cell].append(record)
        ordered_cells = sorted(cells)
        for rows in cells.values():
            rows.sort(key=lambda row: row["fingerprint"])
        cursors = defaultdict(int)
        for index in range(examples_per_split):
            cell = ordered_cells[index % len(ordered_cells)]
            rows = cells[cell]
            selected.append(rows[cursors[cell] % len(rows)])
            cursors[cell] += 1
    return selected


def render_a121p_routing_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Reframe canonical state updates as a signal-routing task family."""
    names = list(record["entity_names"])
    stations = ", ".join(
        f"{name} carries {record['start_state'][name]}" for name in names
    )
    actions = []
    for operation in record["operations"]:
        if operation["family"] == "copy":
            actions.append(
                f"RELAY {operation['source']} => {operation['target']}"
            )
        elif operation["family"] == "swap":
            actions.append(
                f"EXCHANGE {operation['source']} <=> "
                f"{operation['target']}"
            )
        else:
            raise ValueError("unknown canonical family in routing render")
    question = (
        "Signal-routing task. Rules: RELAY overwrites the destination with "
        "the source's "
        "current signal; EXCHANGE swaps the two current signals. "
        "Stations: "
        + stations
        + ". Dispatch plan: "
        + "; ".join(actions)
        + f". Inspect station: {record['query_register']}."
    )
    fingerprint = hashlib.sha256(
        ("routing-family-v2|" + record["fingerprint"]).encode("utf-8")
    ).hexdigest()
    return {
        **record,
        "source_fingerprint": record["fingerprint"],
        "fingerprint": fingerprint,
        "example_id": "routing-" + record["example_id"],
        "task_family": "signal_routing_relay_exchange_v2",
        "question": question,
    }


def _balanced_routing_subset(
    records: Sequence[dict[str, Any]], size: int
) -> list[dict[str, Any]]:
    return [
        render_a121p_routing_record(record)
        for record in _balanced_records(records, size)
    ]


def _balanced_records(
    records: Sequence[dict[str, Any]], size: int
) -> list[dict[str, Any]]:
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cells[
            (int(record["entity_count"]), int(record["program_length"]))
        ].append(record)
    for rows in cells.values():
        rows.sort(key=lambda row: row["fingerprint"])
    ordered_cells = sorted(cells)
    cursors = defaultdict(int)
    selected: list[dict[str, Any]] = []
    for index in range(size):
        cell = ordered_cells[index % len(ordered_cells)]
        rows = cells[cell]
        row = rows[cursors[cell] % len(rows)]
        cursors[cell] += 1
        selected.append(row)
    return selected


def _load_selected_cached_items(
    cache_dir: Path,
    cache_manifest: dict[str, Any],
    split: str,
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Stream source shards and retain only requested rows."""
    wanted = {record["fingerprint"] for record in records}
    selected: dict[str, dict[str, torch.Tensor]] = {}
    for name in cache_manifest["splits"][split]["shards"]:
        payload = torch.load(
            cache_dir / split / name,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for row, fingerprint in enumerate(payload["fingerprints"]):
            if fingerprint not in wanted:
                continue
            length = int(payload["token_lengths"][row])
            selected[fingerprint] = {
                "last_hidden": payload["last_hidden"][
                    row, :length
                ].clone(),
                "attention_mask": payload["attention_mask"][
                    row, :length
                ].clone(),
            }
        if len(selected) == len(wanted):
            break
    missing = sorted(wanted - set(selected))
    if missing:
        raise RuntimeError(
            f"joint bundle source cache missing {len(missing)} rows"
        )
    return selected


def prepare_a121p_joint_training_bundle(
    canonical_cache_dir: Path,
    canonical_data_dir: Path,
    routing_cache_dir: Path,
    routing_data_dir: Path,
    output_cache_dir: Path,
    output_data_dir: Path,
    *,
    canonical_train_size: int = 1024,
    canonical_validation_size: int = 256,
    shard_size: int = 64,
) -> dict[str, Any]:
    """Merge existing hidden states into a balanced two-family cache."""
    for path in (output_cache_dir, output_data_dir):
        if path.exists() and any(path.rglob("*")):
            raise RuntimeError(f"refusing to overwrite joint bundle {path}")
        path.mkdir(parents=True, exist_ok=True)
    canonical_manifest = json.loads(
        (canonical_cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    routing_manifest = json.loads(
        (routing_cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    stable_fields = (
        "model_id",
        "revision",
        "tokenizer_revision",
        "dtype",
        "hidden_layer",
    )
    if any(
        canonical_manifest[field] != routing_manifest[field]
        for field in stable_fields
    ):
        raise ValueError("joint bundle frozen-Qwen contract mismatch")
    if (
        canonical_manifest.get("silent_truncation") is not False
        or routing_manifest.get("silent_truncation") is not False
    ):
        raise ValueError("joint bundle refuses truncated source caches")

    split_sizes = {
        "train": canonical_train_size,
        "validation": canonical_validation_size,
    }
    split_reports: dict[str, Any] = {}
    data_reports: dict[str, Any] = {}
    for split, canonical_size in split_sizes.items():
        canonical_records = _balanced_records(
            load_a119h2_records(canonical_data_dir, split),
            canonical_size,
        )
        routing_records = load_a119h2_records(routing_data_dir, split)
        if len(routing_records) != canonical_size:
            raise ValueError(
                "joint bundle requires equal canonical/routing split sizes"
            )
        canonical_items = _load_selected_cached_items(
            canonical_cache_dir,
            canonical_manifest,
            split,
            canonical_records,
        )
        routing_items = _load_selected_cached_items(
            routing_cache_dir,
            routing_manifest,
            split,
            routing_records,
        )
        sources: list[tuple[dict[str, torch.Tensor], dict[str, Any]]] = []
        for canonical, routing in zip(canonical_records, routing_records):
            sources.append(
                (
                    canonical_items[canonical["fingerprint"]],
                    canonical,
                )
            )
            sources.append(
                (
                    routing_items[routing["fingerprint"]],
                    routing,
                )
            )
        records = [record for _, record in sources]
        fingerprints = [record["fingerprint"] for record in records]
        if len(fingerprints) != len(set(fingerprints)):
            raise RuntimeError("joint bundle contains duplicate fingerprints")
        (output_data_dir / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        split_dir = output_cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_names = []
        total_tokens = maximum_token_length = 0
        for shard_index, start in enumerate(
            range(0, len(sources), shard_size)
        ):
            shard_sources = sources[start : start + shard_size]
            items = [item for item, _ in shard_sources]
            maximum = max(
                int(item["last_hidden"].shape[0]) for item in items
            )
            width = int(items[0]["last_hidden"].shape[-1])
            hidden = torch.zeros(
                len(items), maximum, width, dtype=torch.float16
            )
            mask = torch.zeros(
                len(items), maximum, dtype=torch.bool
            )
            lengths = []
            for row, item in enumerate(items):
                length = int(item["last_hidden"].shape[0])
                hidden[row, :length] = item["last_hidden"]
                mask[row, :length] = item["attention_mask"]
                lengths.append(length)
            name = f"shard_{shard_index:05d}.pt"
            torch.save(
                {
                    "schema_version": CACHE_SCHEMA,
                    "last_hidden": hidden,
                    "attention_mask": mask,
                    "token_lengths": torch.tensor(
                        lengths, dtype=torch.long
                    ),
                    "fingerprints": [
                        record["fingerprint"]
                        for _, record in shard_sources
                    ],
                    "example_ids": [
                        record["example_id"]
                        for _, record in shard_sources
                    ],
                },
                split_dir / name,
            )
            shard_names.append(name)
            total_tokens += sum(lengths)
            maximum_token_length = max(maximum_token_length, maximum)
        split_reports[split] = {
            "examples": len(records),
            "source_tokens": total_tokens,
            "entity_counts": sorted(
                {int(record["entity_count"]) for record in records}
            ),
            "program_lengths": sorted(
                {int(record["program_length"]) for record in records}
            ),
            "maximum_token_length": maximum_token_length,
            "shards": shard_names,
            "seconds": 0.0,
            "source_data_order": "canonical-routing-interleaved-balanced",
            "source_data_split": split,
        }
        data_reports[split] = {
            "examples": len(records),
            "canonical_examples": len(canonical_records),
            "routing_examples": len(routing_records),
            "source_split": split,
        }
    data_manifest = {
        "schema_version": "yggdrasil.v2-a1.21p.joint-data.v1",
        "stage": "A1.21P shared Boundary joint training",
        "canonical_data_dir": str(canonical_data_dir.resolve()),
        "routing_data_dir": str(routing_data_dir.resolve()),
        "family_balance": "1:1 within every split",
        "splits": data_reports,
    }
    (output_data_dir / "manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    cache_manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "stage": "A1.21P shared Boundary joint hidden cache",
        "data_dir": str(output_data_dir.resolve()),
        "data_manifest_sha256": file_sha256(
            output_data_dir / "manifest.json"
        ),
        **{field: canonical_manifest[field] for field in stable_fields},
        "max_length": max(
            int(canonical_manifest["max_length"]),
            int(routing_manifest["max_length"]),
        ),
        "silent_truncation": False,
        "qwen_trainable_parameters": 0,
        "full_source_saved": True,
        "oracle_span_input": False,
        "oracle_role_tensor_input": False,
        "oracle_entity_mask_input": False,
        "oracle_operation_mask_input": False,
        "input_ids_saved": False,
        "inference_batch_size": None,
        "shard_size": shard_size,
        "selection": "a1.21p-joint-canonical-routing-balanced",
        "splits": split_reports,
        "encoding_seconds_excluding_model_and_tokenizer_load": 0.0,
        "peak_allocated_bytes": None,
        "reused_existing_frozen_qwen_hidden": True,
    }
    (output_cache_dir / "manifest.json").write_text(
        json.dumps(cache_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "yggdrasil.v2-a1.21p.joint-bundle.v1",
        "data_manifest": data_manifest,
        "cache_manifest": cache_manifest,
    }


def prepare_a121p_routing_data(
    source_data_dir: Path,
    output_dir: Path,
    *,
    train_size: int = 4096,
    validation_size: int = 512,
    extra_splits: Sequence[str] = (),
    extra_size: int = 64,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.glob("*.jsonl")):
        raise RuntimeError(f"refusing to overwrite routing data {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_sizes = {
        "train": train_size,
        "validation": validation_size,
        **{split: extra_size for split in extra_splits},
    }
    split_reports: dict[str, Any] = {}
    all_fingerprints: set[str] = set()
    for split, size in split_sizes.items():
        records = _balanced_routing_subset(
            load_a119h2_records(source_data_dir, split), size
        )
        fingerprints = {row["fingerprint"] for row in records}
        if len(fingerprints) != len(records):
            raise RuntimeError("routing data contains duplicate fingerprints")
        if all_fingerprints & fingerprints:
            raise RuntimeError("routing train/validation overlap")
        all_fingerprints.update(fingerprints)
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in records
            ),
            encoding="utf-8",
        )
        split_reports[split] = {
            "examples": len(records),
            "entity_counts": sorted(
                {int(row["entity_count"]) for row in records}
            ),
            "program_lengths": sorted(
                {int(row["program_length"]) for row in records}
            ),
            "source_split": split,
        }
    manifest = {
        "schema_version": "yggdrasil.v2-a1.21p.routing-data.v1",
        "stage": "A1.21P routing-family Boundary transfer",
        "source_data_dir": str(source_data_dir.resolve()),
        "task_family": "signal_routing_relay_exchange_v2",
        "canonical_semantics_preserved": True,
        "surface_only_transform": True,
        "train_validation_overlap": 0,
        "splits": split_reports,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _demonstrations(data_dir: Path) -> list[dict[str, Any]]:
    records = load_a119h2_records(data_dir, "train")
    targets = ((3, 4), (4, 12))
    selected: list[dict[str, Any]] = []
    for count, length in targets:
        candidates = [
            row
            for row in records
            if int(row["entity_count"]) == count
            and int(row["program_length"]) == length
        ]
        selected.append(min(candidates, key=lambda row: row["fingerprint"]))
    return selected


class _FinalAnswerCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        terminal = [
            bool(
                FINAL_RE.search(
                    self.tokenizer.decode(
                        row[self.prompt_length :],
                        skip_special_tokens=True,
                    )
                )
            )
            for row in input_ids
        ]
        return torch.tensor(
            terminal, dtype=torch.bool, device=input_ids.device
        )


def _generate_to_final(
    model: nn.Module,
    tokenizer: Any,
    encoded: dict[str, torch.Tensor],
    *,
    max_wall_seconds: float,
    chunk_tokens: int = 16,
) -> tuple[torch.Tensor, str]:
    prompt_length = int(encoded["input_ids"].shape[1])
    sequence = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    context_limit = int(tokenizer.model_max_length)
    started = time.perf_counter()
    termination = "physical_context_boundary"
    while sequence.shape[1] < context_limit:
        if time.perf_counter() - started >= max_wall_seconds:
            termination = "safety_wall_timeout"
            break
        chunk = min(chunk_tokens, context_limit - int(sequence.shape[1]))
        generated = model.generate(
            input_ids=sequence,
            attention_mask=attention_mask,
            max_new_tokens=chunk,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [_FinalAnswerCriteria(tokenizer, prompt_length)]
            ),
        )
        new_tokens = generated[:, sequence.shape[1] :]
        sequence = generated
        completion = tokenizer.decode(
            sequence[0, prompt_length:], skip_special_tokens=True
        )
        if FINAL_RE.search(completion):
            termination = "semantic_final"
            break
        if not new_tokens.numel():
            termination = "eos_or_empty"
            break
        if tokenizer.eos_token_id is not None and bool(
            (new_tokens == tokenizer.eos_token_id).any()
        ):
            termination = "eos"
            break
        attention_mask = torch.ones_like(
            sequence, dtype=encoded["attention_mask"].dtype
        )
    return sequence, termination


def _aggregate_predictions(
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in predictions]
    return {
        "examples": len(predictions),
        "final_answer_accuracy": sum(
            bool(row["correct"]) for row in predictions
        )
        / max(1, len(predictions)),
        "trajectory_full_exact": sum(
            bool(row["trajectory_full_exact"]) for row in predictions
        )
        / max(1, len(predictions)),
        "input_tokens": sum(int(row["input_tokens"]) for row in predictions),
        "generated_tokens": sum(
            int(row.get("generated_tokens", 0)) for row in predictions
        ),
        "latency_seconds_total": sum(latencies),
        "latency_seconds_median": (
            statistics.median(latencies) if latencies else 0.0
        ),
        "latency_seconds_p95": _percentile(latencies, 0.95),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        int(round((len(ordered) - 1) * fraction)),
    )
    return float(ordered[index])


@torch.inference_mode()
def _run_latent(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    backbone: nn.Module,
    boundary: nn.Module,
    *,
    device: str,
    max_input_tokens: int,
) -> dict[str, Any]:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    predictions: list[dict[str, Any]] = []
    for record in records:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        input_ids, attention_mask, lengths = _encode_batch(
            tokenizer, [record], max_input_tokens
        )
        hidden = backbone.model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
        ).last_hidden_state
        output = boundary(hidden, attention_mask.to(device))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        labels = encode_a120b_labels([record], device)
        state_prediction = output["state_logits"].argmax(dim=-1)
        active = labels.state_targets >= 0
        trajectory_exact = bool(
            (
                (state_prediction == labels.state_targets) | ~active
            ).all()
        )
        answer_index = int(output["answer_logits"].argmax(dim=-1).item())
        prediction = VALUE_LABELS[answer_index]
        predictions.append(
            {
                "example_id": record["example_id"],
                "split": record["split"],
                "entity_count": int(record["entity_count"]),
                "program_length": int(record["program_length"]),
                "target": record["answer"],
                "prediction": prediction,
                "correct": prediction == record["answer"],
                "trajectory_full_exact": trajectory_exact,
                "input_tokens": int(lengths[0]),
                "generated_tokens": 0,
                "latent_recurrent_loops": int(
                    boundary.config.maximum_operations
                ),
                "active_program_transitions": int(
                    record["program_length"]
                ),
                "latency_seconds": latency,
            }
        )
        del hidden, output
    return {
        "quality_cost": _aggregate_predictions(predictions),
        "predictions": predictions,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else 0
        ),
        "cost_semantics": (
            "online tokenizer + one frozen-Qwen full-text encode + Boundary "
            "+ 32 fixed recurrent core loops; no cached-hidden substitution"
        ),
    }


@torch.inference_mode()
def _run_text_mode(
    records: Sequence[dict[str, Any]],
    demonstrations: Sequence[dict[str, Any]],
    tokenizer: Any,
    model: nn.Module,
    *,
    mode: str,
    device: str,
    max_input_tokens: int,
    max_wall_seconds: float,
) -> dict[str, Any]:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    predictions: list[dict[str, Any]] = []
    for record in records:
        user_prompt = build_a121p_prompt(
            record,
            mode,
            demonstrations if mode == "text_cot" else (),
        )
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an exact symbolic state-machine executor. "
                        "Follow the user's output format and never invent, "
                        "skip, or reorder an operation."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            prompt, return_tensors="pt", truncation=False
        )
        input_length = int(encoded["input_ids"].shape[1])
        if input_length > max_input_tokens:
            raise ValueError(
                f"A1.21P input length {input_length} exceeds "
                f"max_input_tokens={max_input_tokens}; refusing truncation"
            )
        encoded = encoded.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        generated, termination = _generate_to_final(
            model,
            tokenizer,
            encoded,
            max_wall_seconds=max_wall_seconds,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        completion_ids = generated[0, input_length:]
        completion = tokenizer.decode(
            completion_ids, skip_special_tokens=True
        )
        prediction, parsed_states = parse_a121p_visible_trace(completion)
        expected_states = {
            step: dict(state)
            for step, state in enumerate(
                record["state_trajectory"], start=1
            )
        }
        trajectory_exact = (
            mode == "text_cot"
            and bool(expected_states)
            and all(
                parsed_states.get(step) == state
                for step, state in expected_states.items()
            )
        )
        predictions.append(
            {
                "example_id": record["example_id"],
                "split": record["split"],
                "entity_count": int(record["entity_count"]),
                "program_length": int(record["program_length"]),
                "target": record["answer"],
                "prediction": prediction,
                "correct": prediction == record["answer"],
                "trajectory_full_exact": trajectory_exact,
                "termination_reason": termination,
                "input_tokens": input_length,
                "generated_tokens": int(completion_ids.numel()),
                "latency_seconds": latency,
                "completion": completion,
            }
        )
        del generated, completion_ids, encoded
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "quality_cost": _aggregate_predictions(predictions),
        "predictions": predictions,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else 0
        ),
        "cost_semantics": (
            "online autoregressive frozen-Qwen generation with KV cache; "
            "official chat template with thinking disabled; no total "
            "output-token cap, semantic FINAL or safety wall stop"
        ),
        "prompt_wrapper": (
            "official tokenizer chat template; system+user roles; "
            "add_generation_prompt=true; enable_thinking=false"
        ),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_quality = float(left["final_answer_accuracy"])
    right_quality = float(right["final_answer_accuracy"])
    left_latency = float(left["latency_seconds_median"])
    right_latency = float(right["latency_seconds_median"])
    return (
        left_quality >= right_quality
        and left_latency <= right_latency
        and (left_quality > right_quality or left_latency < right_latency)
    )


def run_a121p_current_family_preflight(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    output_path: Path,
    splits: Sequence[str],
    examples_per_split: int,
    device: str = "cuda",
    max_input_tokens: int = 2048,
    max_wall_seconds: float = 60.0,
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    task_variant: str = "canonical",
) -> dict[str, Any]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A1.21P preflight requested unavailable CUDA")
    from transformers import AutoTokenizer

    records = select_a121p_records(
        data_dir, splits, examples_per_split
    )
    if task_variant == "routing":
        records = [render_a121p_routing_record(row) for row in records]
    elif task_variant != "canonical":
        raise ValueError(f"unknown A1.21P task variant {task_variant}")
    demonstrations = _demonstrations(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = load_qwen35_text_only(
        model_id, revision, dtype=torch.float16, device=device
    ).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    boundary = load_a120c_checkpoint(checkpoint_path, device).eval()
    latent = _run_latent(
        records,
        tokenizer,
        backbone,
        boundary,
        device=device,
        max_input_tokens=max_input_tokens,
    )
    direct = _run_text_mode(
        records,
        demonstrations,
        tokenizer,
        backbone,
        mode="direct",
        device=device,
        max_input_tokens=max_input_tokens,
        max_wall_seconds=max_wall_seconds,
    )
    text_cot = _run_text_mode(
        records,
        demonstrations,
        tokenizer,
        backbone,
        mode="text_cot",
        device=device,
        max_input_tokens=max_input_tokens,
        max_wall_seconds=max_wall_seconds,
    )
    latent_metrics = latent["quality_cost"]
    direct_metrics = direct["quality_cost"]
    cot_metrics = text_cot["quality_cost"]
    dominated_by = [
        name
        for name, metrics in (
            ("direct", direct_metrics),
            ("text_cot", cot_metrics),
        )
        if _dominates(metrics, latent_metrics)
    ]
    candidate = (
        latent_metrics["final_answer_accuracy"] >= 0.90
        and not dominated_by
    )
    result = {
        "schema_version": PREFLIGHT_SCHEMA,
        "evidence_level": "smoke-preflight",
        "stage": (
            "A1.21P-P0 current-family matched preflight"
            if task_variant == "canonical"
            else "A1.21P-P1 routing-family transfer preflight"
        ),
        "model": {
            "model_id": model_id,
            "revision": revision,
            "dtype": "float16",
            "same_loaded_frozen_qwen_for_all_paths": True,
        },
        "data": {
            "data_dir": str(data_dir),
            "splits": list(splits),
            "examples_per_split": examples_per_split,
            "examples": len(records),
            "fingerprints": [row["fingerprint"] for row in records],
            "same_raw_questions_for_all_paths": True,
            "task_variant": task_variant,
        },
        "generation": {
            "do_sample": False,
            "artificial_output_token_cap": None,
            "max_wall_seconds_per_example": max_wall_seconds,
            "max_input_tokens_guard": max_input_tokens,
            "text_cot_demonstrations": [
                row["fingerprint"] for row in demonstrations
            ],
        },
        "paths": {
            "direct": direct,
            "text_cot": text_cot,
            "hybrid_latent": latent,
        },
        "pareto_preflight": {
            "latent_dominated_by": dominated_by,
            "latent_candidate": candidate,
            "rule": (
                "latent answer >=0.90 and no direct/text-CoT path has both "
                "at-least-equal answer accuracy and at-most-equal median "
                "online latency"
            ),
        },
        "gates": {
            "same_frozen_qwen": True,
            "same_raw_questions": True,
            "online_qwen_cost_included_for_latent": True,
            "no_cached_hidden_latency_substitution": True,
            "latent_answer_at_least_0_90": (
                latent_metrics["final_answer_accuracy"] >= 0.90
            ),
            "latent_not_dominated": not dominated_by,
        },
        "passed": candidate,
        "formal_a121p_closed": False,
        "missing_for_formal": [
            "K=1 matched latent baseline",
            *(
                ["second executable relation/planning task family"]
                if task_variant == "canonical"
                else ["joint canonical+routing training and evaluation"]
            ),
            "three fresh seeds",
            "training and teacher-data cost matching",
            "full heldout and intervention matrix",
            "FLOPs and activation/KV accounting",
        ],
        "boundaries": [
            "This P0 is a go/no-go preflight, not A1.21P formal.",
            "The latent Boundary/core is trained; direct/text-CoT use the "
            "frozen base with deterministic prompting, so training-budget "
            "fairness is not yet established.",
            "Visible text-CoT trajectory parsing is reported separately from "
            "final-answer quality.",
        ],
    }
    _write_json(output_path, result)
    return result


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _optional_passed_audit(path: Path | None) -> bool:
    if path is None:
        return False
    return bool(_load_required_json(path).get("passed", False))


def assess_a121p_stage(
    *,
    canonical_matrix_path: Path,
    hidden_causal_path: Path,
    routing_probe_paths: Sequence[Path],
    canonical_preflight_path: Path,
    routing_preflight_path: Path,
    routing_manifest_path: Path,
    k1_results_path: Path,
    optimization_path_results: Sequence[Path],
    output_path: Path,
    training_budget_audit_path: Path | None = None,
    cost_audit_path: Path | None = None,
    minimum_online_examples_per_family: int = 64,
) -> dict[str, Any]:
    """Separate mechanism evidence from the formal matched-Pareto contract."""
    if len(routing_probe_paths) != 3:
        raise ValueError("A1.21P assessment requires exactly three routing probes")
    if len(optimization_path_results) != 3:
        raise ValueError(
            "A1.21P assessment requires exactly three optimization-path runs"
        )
    canonical = _load_required_json(canonical_matrix_path)
    hidden_causal = _load_required_json(hidden_causal_path)
    routing_probes = [
        _load_required_json(path) for path in routing_probe_paths
    ]
    canonical_preflight = _load_required_json(canonical_preflight_path)
    routing_preflight = _load_required_json(routing_preflight_path)
    routing_manifest = _load_required_json(routing_manifest_path)
    k1_results = _load_required_json(k1_results_path)
    optimization_runs = [
        _load_required_json(path) for path in optimization_path_results
    ]

    online_examples = {
        "canonical": int(canonical_preflight["data"]["examples"]),
        "routing": int(routing_preflight["data"]["examples"]),
    }
    routing_stability = (
        len(routing_probes) == 3
        and all(bool(payload.get("passed", False)) for payload in routing_probes)
    )
    optimization_path_stability = (
        len(optimization_runs) == 3
        and all(
            bool(payload.get("formal_eligibility_passed", False))
            for payload in optimization_runs
        )
    )
    fresh_model_data_seeds = all(
        bool(payload.get("training", {}).get("fresh_initialization", False))
        for payload in optimization_runs
    )
    k1_aggregate = k1_results.get("validation", {}).get("aggregate", {})
    k1_reported = (
        int(k1_aggregate.get("examples", 0)) > 0
        and "trajectory_full_exact" in k1_aggregate
        and "final_answer_accuracy" in k1_aggregate
    )
    second_executable_family = (
        not bool(routing_manifest.get("surface_only_transform", True))
        and not bool(
            routing_manifest.get("canonical_semantics_preserved", True)
        )
    )
    online_scale_passed = all(
        count >= minimum_online_examples_per_family
        for count in online_examples.values()
    )

    mechanism_gates = {
        "canonical_full_matrix_diagnostic_passed": bool(
            canonical.get("passed", False)
        ),
        "hidden_causal_dependence_passed": bool(
            hidden_causal.get("passed", False)
        ),
        "routing_surface_transfer_three_of_three": routing_stability,
        "k1_capacity_baseline_reported": k1_reported,
        "canonical_online_pareto_smoke_passed": bool(
            canonical_preflight.get("passed", False)
        ),
        "routing_online_pareto_smoke_passed": bool(
            routing_preflight.get("passed", False)
        ),
    }
    formal_gates = {
        **mechanism_gates,
        "strict_joint_validation_eligibility_three_of_three": (
            optimization_path_stability
        ),
        "online_examples_at_least_threshold_per_family": online_scale_passed,
        "second_executable_algebra_or_planning_family": (
            second_executable_family
        ),
        "three_fresh_model_and_data_seeds": fresh_model_data_seeds,
        "matched_training_and_teacher_data_budget": (
            _optional_passed_audit(training_budget_audit_path)
        ),
        "full_flops_activation_kv_and_teacher_cost_accounting": (
            _optional_passed_audit(cost_audit_path)
        ),
    }
    relaxed_mechanism_passed = all(mechanism_gates.values())
    a121p_passed = all(formal_gates.values())
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "stage": "A1.20D mechanism repair and A1.21P matched Pareto",
        "evidence_level": "diagnostic-stage-assessment",
        "machine_classification": (
            "matched_pareto_closed"
            if a121p_passed
            else (
                "full_text_hybrid_candidate_confirmed_but_"
                "matched_pareto_not_closed"
                if relaxed_mechanism_passed
                else "full_text_hybrid_mechanism_not_confirmed"
            )
        ),
        "a120d_relaxed_mechanism_passed": relaxed_mechanism_passed,
        "a121p_passed": a121p_passed,
        "a122a_authorized": a121p_passed,
        "stop_reason": (
            None
            if a121p_passed
            else "A1.21P formal contract has failed or missing gates"
        ),
        "mechanism_gates": mechanism_gates,
        "formal_gates": formal_gates,
        "diagnostics": {
            "online_examples": online_examples,
            "minimum_online_examples_per_family": (
                minimum_online_examples_per_family
            ),
            "routing_manifest_surface_only_transform": bool(
                routing_manifest.get("surface_only_transform", True)
            ),
            "routing_manifest_canonical_semantics_preserved": bool(
                routing_manifest.get(
                    "canonical_semantics_preserved", True
                )
            ),
            "routing_probe_pass_count": sum(
                bool(payload.get("passed", False))
                for payload in routing_probes
            ),
            "optimization_path_strict_eligibility_pass_count": sum(
                bool(payload.get("formal_eligibility_passed", False))
                for payload in optimization_runs
            ),
            "optimization_path_fresh_initialization_count": sum(
                bool(
                    payload.get("training", {}).get(
                        "fresh_initialization", False
                    )
                )
                for payload in optimization_runs
            ),
            "k1_validation": {
                "examples": int(k1_aggregate.get("examples", 0)),
                "trajectory_full_exact": float(
                    k1_aggregate.get("trajectory_full_exact", 0.0)
                ),
                "final_state_full_exact": float(
                    k1_aggregate.get("final_state_full_exact", 0.0)
                ),
                "state_token_accuracy": float(
                    k1_aggregate.get("state_token_accuracy", 0.0)
                ),
                "final_answer_accuracy": float(
                    k1_aggregate.get("final_answer_accuracy", 0.0)
                ),
            },
        },
        "completed": [
            "bidirectional three-pass full-text section decoding repair",
            "canonical full split-matrix diagnostic",
            "hidden causal intervention",
            "three optimization-schedule routing surface probes",
            "K=1 recurrent capacity baseline",
            "online canonical and routing five-example Pareto smoke",
        ],
        "not_completed": [
            name for name, passed in formal_gates.items() if not passed
        ],
        "boundaries": [
            "Canonical and routing split matrices are diagnostic because "
            "their source artifacts explicitly set formal_gate=false.",
            "The three routing probes vary only the continuation schedule; "
            "they share the reader/core initialization and data.",
            "Routing is a surface rewrite that preserves COPY/SWAP algebra, "
            "so it is not a new executable relation or planning family.",
            "The five-example online preflights establish only a candidate, "
            "not a stable Pareto estimate.",
            "Direct/text-CoT are deterministic frozen-base prompts while the "
            "hybrid path is trained; training-budget fairness is unproven.",
            "A1.22A must remain stopped unless every formal gate is true.",
        ],
        "sources": {
            "canonical_matrix": {
                "path": str(canonical_matrix_path),
                "sha256": file_sha256(canonical_matrix_path),
            },
            "hidden_causal": {
                "path": str(hidden_causal_path),
                "sha256": file_sha256(hidden_causal_path),
            },
            "routing_probes": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in routing_probe_paths
            ],
            "canonical_preflight": {
                "path": str(canonical_preflight_path),
                "sha256": file_sha256(canonical_preflight_path),
            },
            "routing_preflight": {
                "path": str(routing_preflight_path),
                "sha256": file_sha256(routing_preflight_path),
            },
            "routing_manifest": {
                "path": str(routing_manifest_path),
                "sha256": file_sha256(routing_manifest_path),
            },
            "k1_results": {
                "path": str(k1_results_path),
                "sha256": file_sha256(k1_results_path),
            },
            "optimization_path_results": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in optimization_path_results
            ],
            "training_budget_audit": (
                None
                if training_budget_audit_path is None
                else {
                    "path": str(training_budget_audit_path),
                    "sha256": file_sha256(training_budget_audit_path),
                }
            ),
            "cost_audit": (
                None
                if cost_audit_path is None
                else {
                    "path": str(cost_audit_path),
                    "sha256": file_sha256(cost_audit_path),
                }
            ),
        },
    }
    _write_json(output_path, result)
    return result
