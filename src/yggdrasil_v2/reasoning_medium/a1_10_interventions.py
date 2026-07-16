from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .a1_7_data import REGISTER_NAMES, SUPPORTED_JOINTS, _operation, _render_record
from .a1_8_data import load_a18_records
from .a1_10_cache import (
    CACHE_SCHEMA,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    A110CachedSplit,
    _cache_split,
    _load_payload,
    collate_a110_items,
)
from .a1_10_train import TRAIN_RECURRENT_STEPS, encode_a110_targets, evaluate_a110_items, load_a110_checkpoint
from .model import load_qwen35_text_only


INTERVENTION_CACHE_SCHEMA = "yggdrasil.v2-a1.10.counterfactual-full-token-cache.v1"
INTERVENTION_RESULTS_SCHEMA = "yggdrasil.v2-a1.10.hidden-interventions.v1"


def _sanitize(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"start_value_spans", "operation_spans", "query_register_span", "operation_mask"}
    }


def clone_recomputed_record(
    record: dict[str, Any],
    *,
    operations: Sequence[dict[str, Any]] | None = None,
    query_register: str | None = None,
    suffix: str,
    index: int,
) -> dict[str, Any]:
    rendered = _render_record(
        dict(record["start_state"]),
        list(operations if operations is not None else record["operations"]),
        query_register or record["query_register"],
        f"a1_10_{suffix}",
        index,
    )
    return _sanitize(rendered)


def _trajectory_key(record: dict[str, Any]) -> str:
    return json.dumps(record["state_trajectory"], sort_keys=True)


def build_a110_counterfactual_records(
    records: Sequence[dict[str, Any]], *, seed: int = 20261020
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    variants: dict[str, list[dict[str, Any]]] = {
        "prefix": [],
        "replacement": [],
        "deletion": [],
        "operation_shuffle": [],
        "query_swap": [],
    }
    replacement_catalog = [_operation(joint) for joint in SUPPORTED_JOINTS]
    counters = defaultdict(int)
    for old in records:
        operations = [dict(operation) for operation in old["operations"]]
        old_key = _trajectory_key(old)
        for prefix_length in range(1, len(operations) + 1):
            new = clone_recomputed_record(
                old,
                operations=operations[:prefix_length],
                suffix="prefix",
                index=counters["prefix"],
            )
            counters["prefix"] += 1
            variants["prefix"].append({"record": new, "old_record": _sanitize(old)})
        candidates = list(range(len(operations)))
        rng.shuffle(candidates)
        replacement_pair: dict[str, Any] | None = None
        for step in candidates:
            catalog = list(replacement_catalog)
            rng.shuffle(catalog)
            for replacement in catalog:
                if all(replacement[key] == operations[step][key] for key in ("family", "source", "target")):
                    continue
                changed = list(operations)
                changed[step] = dict(replacement)
                new = clone_recomputed_record(
                    old, operations=changed, suffix="replacement", index=counters["replacement"]
                )
                if _trajectory_key(new) != old_key:
                    replacement_pair = {"record": new, "old_record": _sanitize(old)}
                    break
            if replacement_pair is not None:
                break
        if replacement_pair is not None:
            counters["replacement"] += 1
            variants["replacement"].append(replacement_pair)
        if len(operations) > 1:
            deletion_steps = list(range(len(operations)))
            rng.shuffle(deletion_steps)
            for step in deletion_steps:
                new = clone_recomputed_record(
                    old,
                    operations=operations[:step] + operations[step + 1 :],
                    suffix="deletion",
                    index=counters["deletion"],
                )
                if _trajectory_key(new) != old_key:
                    counters["deletion"] += 1
                    variants["deletion"].append({"record": new, "old_record": _sanitize(old)})
                    break
            orders = [list(reversed(operations)), operations[1:] + operations[:1]]
            for shuffled in orders:
                new = clone_recomputed_record(
                    old,
                    operations=shuffled,
                    suffix="operation_shuffle",
                    index=counters["operation_shuffle"],
                )
                if _trajectory_key(new) != old_key:
                    counters["operation_shuffle"] += 1
                    variants["operation_shuffle"].append({"record": new, "old_record": _sanitize(old)})
                    break
        alternate_queries = [name for name in REGISTER_NAMES if name != old["query_register"]]
        query = rng.choice(alternate_queries)
        new = clone_recomputed_record(
            old, query_register=query, suffix="query_swap", index=counters["query_swap"]
        )
        counters["query_swap"] += 1
        variants["query_swap"].append({"record": new, "old_record": _sanitize(old)})
    if not all(variants.values()):
        raise RuntimeError("A1.10 counterfactual construction produced an empty required variant")
    return variants


def cache_a110_interventions(
    data_dir: Path,
    formal_eval_path: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    model_id: str = QWEN_MODEL_ID,
    revision: str = QWEN_REVISION,
    max_length: int = 1024,
    inference_batch_size: int = 16,
    shard_size: int = 64,
    seed: int = 20261020,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.10 intervention cache is forbidden before the ordinary formal Gate passes")
    if output_dir.exists() and any(output_dir.rglob("*.pt")):
        raise RuntimeError(f"refusing to mix an existing A1.10 intervention cache: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    backbone = load_qwen35_text_only(model_id, revision, dtype=torch.float16, device=device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    variants = build_a110_counterfactual_records(load_a18_records(data_dir, "causal_core"), seed=seed)
    reports: dict[str, Any] = {}
    started = time.perf_counter()
    for name, pairs in variants.items():
        variant_dir = output_dir / name
        records = [pair["record"] for pair in pairs]
        report = _cache_split(
            backbone,
            tokenizer,
            records,
            variant_dir,
            device=device,
            max_length=max_length,
            inference_batch_size=inference_batch_size,
            shard_size=shard_size,
        )
        record_map = {pair["record"]["fingerprint"]: pair for pair in pairs}
        ordered_pairs = [record_map[record["fingerprint"]] for record in sorted(records, key=lambda row: (int(row["program_length"]), row["fingerprint"]))]
        (variant_dir / "records.jsonl").write_text(
            "".join(json.dumps(pair, sort_keys=True) + "\n" for pair in ordered_pairs), encoding="utf-8"
        )
        reports[name] = report
    manifest = {
        "schema_version": INTERVENTION_CACHE_SCHEMA,
        "stage": "A1.10 post-formal text counterfactual full-token cache",
        "formal_eval": str(formal_eval_path),
        "formal_passed_before_cache": True,
        "model_id": model_id,
        "revision": revision,
        "qwen_frozen": True,
        "oracle_role_span_segmentation": False,
        "cache_payload_schema": CACHE_SCHEMA,
        "variants": reports,
        "seconds_excluding_model_and_tokenizer_load": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del backbone
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def load_a110_intervention_items(cache_dir: Path, variant: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variant_dir = cache_dir / variant
    pairs = [json.loads(line) for line in (variant_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    pair_map = {pair["record"]["fingerprint"]: pair for pair in pairs}
    items: list[dict[str, Any]] = []
    old_records: list[dict[str, Any]] = []
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in manifest["variants"][variant]["shards"]:
        payload = _load_payload(variant_dir / name)
        for row, fingerprint in enumerate(payload["fingerprints"]):
            length = int(payload["token_lengths"][row])
            pair = pair_map[fingerprint]
            items.append(
                {
                    "last_hidden": payload["last_hidden"][row, :length],
                    "attention_mask": payload["attention_mask"][row, :length],
                    "record": pair["record"],
                }
            )
            old_records.append(pair["old_record"])
    return items, old_records


def _weighted(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["examples"] for row in rows)
    result: dict[str, Any] = {"examples": total}
    metrics = (
        "trajectory_full_exact",
        "final_state_full_exact",
        "state_token_accuracy",
        "final_answer_accuracy",
        "answer_state_prediction_consistency",
        "old_oracle_trajectory_full_exact",
    )
    for metric in metrics:
        values = [row for row in rows if metric in row]
        if values:
            result[metric] = sum(row[metric] * row["examples"] for row in values) / max(1, sum(row["examples"] for row in values))
    return result


def _evaluate_variant(
    model: Any,
    items: Sequence[dict[str, Any]],
    old_records: Sequence[dict[str, Any]],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[int(item["record"]["program_length"])].append(index)
    rows = []
    for length, indexes in sorted(groups.items()):
        rows.append(
            evaluate_a110_items(
                model,
                [items[index] for index in indexes],
                device,
                recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
                batch_size=batch_size,
                old_records=[old_records[index] for index in indexes],
            )
        )
    return {"aggregate": _weighted(rows), "by_length": {str(row["recurrent_steps"]): row for row in rows}}


def _same_answer_pairs(items: Sequence[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[(int(item["record"]["answer_index"]), int(item["record"]["program_length"]))].append(index)
    new_items: list[dict[str, Any]] = []
    old_records: list[dict[str, Any]] = []
    for indexes in groups.values():
        shuffled = list(indexes)
        rng.shuffle(shuffled)
        for target in indexes:
            candidates = [source for source in shuffled if _trajectory_key(items[source]["record"]) != _trajectory_key(items[target]["record"])]
            if candidates:
                source = candidates[0]
                new_items.append(items[source])
                old_records.append(items[target]["record"])
    if not new_items:
        raise RuntimeError("A1.10 same-answer intervention found no different-trajectory pairs")
    return new_items, old_records


@torch.no_grad()
def _slot_permutation_delta(model: Any, items: Sequence[dict[str, Any]], device: str, batch_size: int) -> float:
    maximum = 0.0
    permutation = torch.arange(model.config.workspace_slots - 1, -1, -1, device=device)
    for start in range(0, len(items), batch_size):
        inputs, records = collate_a110_items(items[start : start + batch_size], device)
        steps = max(TRAIN_RECURRENT_STEPS, int(records[0]["program_length"]))
        normal = model(**inputs, recurrent_steps=steps)
        permuted = model(**inputs, recurrent_steps=steps, slot_permutation=permutation)
        maximum = max(maximum, float((normal["answer_logits"] - permuted["answer_logits"]).abs().max()))
        maximum = max(maximum, float((normal["state_logits"] - permuted["state_logits"]).abs().max()))
    return maximum


def run_a110_interventions(
    checkpoint: Path,
    source_cache_dir: Path,
    intervention_cache_dir: Path,
    data_dir: Path,
    formal_eval_path: Path,
    output_path: Path,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    seed: int = 20261020,
) -> dict[str, Any]:
    formal = json.loads(formal_eval_path.read_text(encoding="utf-8"))
    if not formal.get("passed"):
        raise RuntimeError("A1.10 hidden interventions are forbidden before ordinary formal passes")
    model = load_a110_checkpoint(checkpoint, device)
    text_variants: dict[str, Any] = {}
    for variant in ("prefix", "replacement", "deletion", "operation_shuffle", "query_swap"):
        items, old_records = load_a110_intervention_items(intervention_cache_dir, variant)
        text_variants[variant] = _evaluate_variant(model, items, old_records, device, batch_size)
    causal_dataset = A110CachedSplit(source_cache_dir, data_dir, "causal_core")
    causal_items = causal_dataset.items()
    same_items, same_old = _same_answer_pairs(causal_items, seed)
    same_answer = _evaluate_variant(model, same_items, same_old, device, batch_size)
    no_source_items = [
        {**item, "last_hidden": torch.zeros_like(item["last_hidden"])} for item in causal_items
    ]
    no_source = _evaluate_variant(
        model, no_source_items, [item["record"] for item in no_source_items], device, batch_size
    )
    rng = random.Random(seed)
    permutation = list(range(len(causal_items)))
    rng.shuffle(permutation)
    shuffled_items = [
        {
            "last_hidden": causal_items[source]["last_hidden"],
            "attention_mask": causal_items[source]["attention_mask"],
            "record": causal_items[target]["record"],
        }
        for target, source in enumerate(permutation)
    ]
    independent_shuffle = _evaluate_variant(
        model, shuffled_items, [item["record"] for item in shuffled_items], device, batch_size
    )
    recurrence_rows = []
    for length, indexes in sorted(causal_dataset.indices_by_length.items()):
        recurrence_rows.append(
            evaluate_a110_items(
                model,
                causal_dataset.items(indexes),
                device,
                recurrent_steps=max(TRAIN_RECURRENT_STEPS, length),
                batch_size=batch_size,
                disable_recurrence=True,
            )
        )
    disable_recurrence = {"aggregate": _weighted(recurrence_rows)}
    same_length = min(causal_dataset.indices_by_length)
    slot_items = causal_dataset.items(causal_dataset.indices_by_length[same_length][:batch_size])
    slot_delta = _slot_permutation_delta(model, slot_items, device, batch_size)
    gates = {
        "text_counterfactuals_follow_new_trajectory_at_least_0_95": all(
            text_variants[name]["aggregate"]["trajectory_full_exact"] >= 0.95
            for name in ("prefix", "replacement", "deletion", "operation_shuffle")
        ),
        "query_swap_new_answer_at_least_0_95": text_variants["query_swap"]["aggregate"]["final_answer_accuracy"] >= 0.95,
        "same_answer_new_trajectory_at_least_0_95": same_answer["aggregate"]["trajectory_full_exact"] >= 0.95,
        "changed_counterfactual_old_trajectory_at_most_0_05": all(
            text_variants[name]["aggregate"]["old_oracle_trajectory_full_exact"] <= 0.05
            for name in ("replacement", "deletion", "operation_shuffle")
        ) and same_answer["aggregate"]["old_oracle_trajectory_full_exact"] <= 0.05,
        "no_source_trajectory_at_most_0_20": no_source["aggregate"]["trajectory_full_exact"] <= 0.20,
        "independent_source_shuffle_trajectory_at_most_0_20": independent_shuffle["aggregate"]["trajectory_full_exact"] <= 0.20,
        "disable_recurrence_trajectory_at_most_0_20": disable_recurrence["aggregate"]["trajectory_full_exact"] <= 0.20,
        "slot_permutation_logit_delta_below_1e_5": slot_delta < 1e-5,
    }
    result = {
        "schema_version": INTERVENTION_RESULTS_SCHEMA,
        "checkpoint": str(checkpoint),
        "formal_eval": str(formal_eval_path),
        "formal_passed_before_interventions": True,
        "text_counterfactuals": text_variants,
        "same_answer_different_trajectory": same_answer,
        "no_source": no_source,
        "independent_source_shuffle": independent_shuffle,
        "disable_recurrence": disable_recurrence,
        "slot_permutation_max_abs_logit_delta": slot_delta,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
