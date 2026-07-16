from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_6_core import RelationAddressedCore
from .a1_6_data import REGISTER_NAMES, VALUE_LABELS, load_a16_records
from .a1_6_qwen import FrozenQwenBoundary, masked_pool


def load_cache_items(cache_dir: Path, split: str, data_dir: Path) -> list[dict[str, Any]]:
    records = {record["fingerprint"]: record for record in load_a16_records(data_dir, split)}
    items: list[dict[str, Any]] = []
    for path in sorted((cache_dir / split).glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row, fingerprint in enumerate(payload["fingerprints"]):
            item = {key: value[row] for key, value in payload.items() if isinstance(value, torch.Tensor)}
            item["record"] = records[fingerprint]
            items.append(item)
    return items


def _pool_item(item: dict[str, Any], device: str | torch.device) -> dict[str, torch.Tensor]:
    hidden = item["last_hidden"].to(device)
    start = torch.stack([masked_pool(hidden, mask.to(device)) for mask in item["start_value_masks"]])
    query = masked_pool(hidden, item["query_register_mask"].to(device))
    family = torch.stack([masked_pool(hidden, mask.to(device)) for mask in item["operation_family_masks"]])
    source = torch.stack([masked_pool(hidden, mask.to(device)) for mask in item["operation_source_masks"]])
    target = torch.stack([masked_pool(hidden, mask.to(device)) for mask in item["operation_target_masks"]])
    return {"start": start, "query": query, "family": family, "source": source, "target": target, "mask": item["operation_mask"].to(device)}


def _batch_items(items: Sequence[dict[str, Any]], device: str | torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    pooled = [_pool_item(item, device) for item in items]
    max_steps = max(item["family"].shape[0] for item in pooled)
    width = pooled[0]["family"].shape[-1]
    result = {
        "start": torch.stack([item["start"] for item in pooled]),
        "query": torch.stack([item["query"] for item in pooled]),
        "family": torch.zeros((len(pooled), max_steps, width), device=device),
        "source": torch.zeros((len(pooled), max_steps, width), device=device),
        "target": torch.zeros((len(pooled), max_steps, width), device=device),
        "mask": torch.zeros((len(pooled), max_steps), dtype=torch.bool, device=device),
    }
    for row, item in enumerate(pooled):
        steps = item["family"].shape[0]
        result["family"][row, :steps] = item["family"]
        result["source"][row, :steps] = item["source"]
        result["target"][row, :steps] = item["target"]
        result["mask"][row, :steps] = item["mask"]
    return result, [item["record"] for item in items]


def _labels(records: Sequence[dict[str, Any]], device: str | torch.device) -> dict[str, torch.Tensor]:
    family, source, target = [], [], []
    for record in records:
        family.append([0 if op["family"] == "copy" else 1 for op in record["operations"]])
        source.append([REGISTER_NAMES.index(op["source"]) for op in record["operations"]])
        target.append([REGISTER_NAMES.index(op["target"]) for op in record["operations"]])
    max_steps = max(map(len, family))
    family_tensor = torch.zeros((len(records), max_steps), dtype=torch.long, device=device)
    source_tensor = torch.zeros_like(family_tensor)
    target_tensor = torch.zeros_like(family_tensor)
    for row in range(len(records)):
        length = len(family[row])
        family_tensor[row, :length] = torch.tensor(family[row], device=device)
        source_tensor[row, :length] = torch.tensor(source[row], device=device)
        target_tensor[row, :length] = torch.tensor(target[row], device=device)
    return {"start": torch.tensor([[VALUE_LABELS.index(record["start_state"][name]) for name in REGISTER_NAMES] for record in records], device=device), "query": torch.tensor([REGISTER_NAMES.index(record["query_register"]) for record in records], device=device), "family": family_tensor, "source": source_tensor, "target": target_tensor}


def train_c1(train_items: Sequence[dict[str, Any]], validation_items: Sequence[dict[str, Any]], core: RelationAddressedCore, output_dir: Path, *, phase: str, device: str = "cuda", seed: int = 20260715, steps: int = 3000, batch_size: int = 64, boundary_learning_rate: float = 3e-4, core_learning_rate: float = 3e-5, validation_interval: int = 100, early_stop_patience: int = 10) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_width = int(train_items[0]["last_hidden"].shape[-1])
    model = FrozenQwenBoundary(source_width, core).to(device)
    for parameter in model.core.parameters():
        parameter.requires_grad = phase == "joint"
    parameters = [{"params": [p for n, p in model.named_parameters() if not n.startswith("core.")], "lr": boundary_learning_rate}]
    if phase == "joint":
        parameters.append({"params": list(model.core.parameters()), "lr": core_learning_rate})
    optimizer = torch.optim.AdamW(parameters)
    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    best = -float("inf")
    patience = 0
    for step in range(1, steps + 1):
        model.train()
        selected = [train_items[rng.randrange(len(train_items))] for _ in range(batch_size)]
        inputs, records = _batch_items(selected, device)
        labels = _labels(records, device)
        output = model(inputs["start"], inputs["query"], inputs["family"], inputs["source"], inputs["target"], inputs["mask"])
        start_loss = nn.functional.cross_entropy(output["start_value_logits"].reshape(-1, 10), labels["start"].reshape(-1))
        query_loss = nn.functional.cross_entropy(output["query_pointer_logits"], labels["query"])
        family_loss = nn.functional.cross_entropy(output["family_logits"].reshape(-1, 2), labels["family"].reshape(-1), reduction="none")
        source_loss = nn.functional.cross_entropy(output["source_pointer_logits"].reshape(-1, 3), labels["source"].reshape(-1), reduction="none")
        target_loss = nn.functional.cross_entropy(output["target_pointer_logits"].reshape(-1, 3), labels["target"].reshape(-1), reduction="none")
        role_mask = inputs["mask"].reshape(-1)
        role_loss = (source_loss * role_mask).sum() / role_mask.sum().clamp_min(1) + (target_loss * role_mask).sum() / role_mask.sum().clamp_min(1)
        total = start_loss + query_loss + (family_loss * role_mask).sum() / role_mask.sum().clamp_min(1) + role_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if step % validation_interval == 0 or step == steps:
            with torch.no_grad():
                val_inputs, val_records = _batch_items(validation_items[:batch_size], device)
                val_labels = _labels(val_records, device)
                val_out = model(val_inputs["start"], val_inputs["query"], val_inputs["family"], val_inputs["source"], val_inputs["target"], val_inputs["mask"])
                query_acc = float((val_out["query_pointer_logits"].argmax(-1) == val_labels["query"]).float().mean())
                source_acc = float(((val_out["source_pointer_logits"].argmax(-1) == val_labels["source"]) & val_inputs["mask"]).sum() / val_inputs["mask"].sum().clamp_min(1))
                target_acc = float(((val_out["target_pointer_logits"].argmax(-1) == val_labels["target"]) & val_inputs["mask"]).sum() / val_inputs["mask"].sum().clamp_min(1))
            row = {"step": step, "loss": float(total.detach()), "validation_query_pointer_accuracy": query_acc, "validation_source_pointer_accuracy": source_acc, "validation_target_pointer_accuracy": target_acc}
            history.append(row)
            score = query_acc + source_acc + target_acc
            checkpoint = {"schema_version": "yggdrasil.v2-a1.6.c1.checkpoint.v1", "phase": phase, "source_width": source_width, "model": model.state_dict(), "core_config": core.config.__dict__}
            torch.save(checkpoint, output_dir / "latest.pt")
            if score > best:
                best = score
                patience = 0
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                patience += 1
            if patience >= early_stop_patience:
                break
    result = {"schema_version": "yggdrasil.v2-a1.6.c1.results.v1", "phase": phase, "history": history, "integrity": model.integrity_report(), "artifacts": {"best": str(output_dir / "best.pt"), "latest": str(output_dir / "latest.pt")}}
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
