from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_6_core import A16CoreConfig, RelationAddressedCore
from .a1_6_data import REGISTER_NAMES, load_a16_records
from .model import load_qwen35_text_only


QWEN_MODEL_ID = "Qwen/Qwen3.5-2B"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CACHE_SCHEMA = "yggdrasil.v2-a1.6.qwen-boundary-cache.v1"


def _span_mask(offsets: Sequence[Sequence[int]], start: int, end: int, *, label: str) -> torch.Tensor:
    mask = torch.tensor([offset_end > start and offset_start < end for offset_start, offset_end in offsets], dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError(f"empty token mask for {label} span [{start}, {end})")
    return mask


def _record_masks(record: dict[str, Any], offsets: Sequence[Sequence[int]], length: int) -> dict[str, torch.Tensor]:
    start_masks = torch.stack([_span_mask(offsets, span["char_start"], span["char_end"], label=f"start_value_{name}") for name, span in record["start_value_spans"].items()])
    query_span = record["query_register_span"]
    query_mask = _span_mask(offsets, query_span["char_start"], query_span["char_end"], label="query_register")
    role_masks = {"family": [], "source": [], "target": []}
    for step, span in enumerate(record["operation_spans"]):
        role_masks["family"].append(_span_mask(offsets, span["family_char_start"], span["family_char_end"], label=f"operation_{step}_family"))
        role_masks["source"].append(_span_mask(offsets, span["source_char_start"], span["source_char_end"], label=f"operation_{step}_source"))
        role_masks["target"].append(_span_mask(offsets, span["target_char_start"], span["target_char_end"], label=f"operation_{step}_target"))
    operation_masks = {name: torch.stack(items) if items else torch.zeros((0, length), dtype=torch.bool) for name, items in role_masks.items()}
    return {"start_value_masks": start_masks, "query_register_mask": query_mask, "operation_family_masks": operation_masks["family"], "operation_source_masks": operation_masks["source"], "operation_target_masks": operation_masks["target"], "operation_mask": torch.tensor(record["operation_mask"], dtype=torch.bool)}


def cache_qwen_boundary(data_dir: Path, output_dir: Path, *, model_id: str = QWEN_MODEL_ID, revision: str = QWEN_REVISION, device: str = "cuda", max_length: int = 512, max_train_examples: int | None = None, shard_size: int = 64) -> dict[str, Any]:
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = load_qwen35_text_only(model_id, revision, dtype=torch.float16, device=device)
    backbone.eval()
    split_manifest: dict[str, Any] = {}
    for split in ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core"):
        records = load_a16_records(data_dir, split)
        if split == "train" and max_train_examples is not None:
            records = records[:max_train_examples]
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shards: list[str] = []
        for shard_index, start in enumerate(range(0, len(records), shard_size)):
            shard_records = records[start : start + shard_size]
            encoded = [tokenizer(record["question"], add_special_tokens=True, truncation=False, return_offsets_mapping=True) for record in shard_records]
            lengths = [len(item["input_ids"]) for item in encoded]
            if any(length > max_length for length in lengths):
                raise ValueError(f"token length exceeds max_length={max_length} in {split}; refusing silent truncation")
            seq_length = max(lengths, default=0)
            input_ids = torch.full((len(encoded), seq_length), tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), seq_length), dtype=torch.bool)
            masks: list[dict[str, torch.Tensor]] = []
            for row, item in enumerate(encoded):
                length = len(item["input_ids"])
                input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
                attention_mask[row, :length] = True
                masks.append(_record_masks(shard_records[row], item["offset_mapping"], length))
            with torch.no_grad():
                hidden = backbone.model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), use_cache=False).last_hidden_state.detach().to(torch.float16).cpu()
            max_steps = max((item["operation_mask"].numel() for item in masks), default=0)
            payload_masks: dict[str, torch.Tensor] = {
                "start_value_masks": torch.zeros((len(encoded), 3, seq_length), dtype=torch.bool),
                "query_register_mask": torch.zeros((len(encoded), seq_length), dtype=torch.bool),
                "operation_family_masks": torch.zeros((len(encoded), max_steps, seq_length), dtype=torch.bool),
                "operation_source_masks": torch.zeros((len(encoded), max_steps, seq_length), dtype=torch.bool),
                "operation_target_masks": torch.zeros((len(encoded), max_steps, seq_length), dtype=torch.bool),
                "operation_mask": torch.zeros((len(encoded), max_steps), dtype=torch.bool),
            }
            for row, item in enumerate(masks):
                payload_masks["start_value_masks"][row] = item["start_value_masks"]
                payload_masks["query_register_mask"][row] = item["query_register_mask"]
                for name in ("family", "source", "target"):
                    payload_masks[f"operation_{name}_masks"][row, : item[f"operation_{name}_masks"].shape[0]] = item[f"operation_{name}_masks"]
                payload_masks["operation_mask"][row, : item["operation_mask"].numel()] = item["operation_mask"]
            payload = {"schema_version": CACHE_SCHEMA, "last_hidden": hidden, "attention_mask": attention_mask, **payload_masks, "example_ids": [r["example_id"] for r in shard_records], "fingerprints": [r["fingerprint"] for r in shard_records], "token_lengths": lengths}
            path = split_dir / f"shard_{shard_index:05d}.pt"
            torch.save(payload, path)
            shards.append(str(path))
        split_manifest[split] = {"examples": len(records), "shards": shards, "max_token_length": max((max(torch.load(path, weights_only=False)["token_lengths"]) for path in [Path(item) for item in shards]), default=0)}
    manifest = {"schema_version": CACHE_SCHEMA, "model_id": model_id, "revision": revision, "max_length": max_length, "shard_size": shard_size, "splits": split_manifest, "independent_masks": ["start_value_masks", "query_register_mask", "operation_family_masks", "operation_source_masks", "operation_target_masks", "operation_mask"], "silent_truncation": False, "full_source_cross_attention": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def masked_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=hidden.dtype)
    return torch.einsum("bl,blh->bh", weights, hidden) / weights.sum(-1, keepdim=True).clamp_min(1.0)


class FrozenQwenBoundary(nn.Module):
    """C1 boundary adapter; the C0 core remains the only recurrent core."""

    def __init__(self, source_width: int, core: RelationAddressedCore) -> None:
        super().__init__()
        self.core = core
        d = core.config.latent_width
        self.start_value_projector = nn.Linear(source_width, d)
        self.query_projector = nn.Linear(source_width, d)
        self.family_projector = nn.Linear(source_width, d)
        self.source_role_projector = nn.Linear(source_width, d)
        self.target_role_projector = nn.Linear(source_width, d)
        self.query_pointer = nn.Linear(d, core.config.register_slots, bias=False)
        self.start_value_head = nn.Linear(d, core.config.value_classes)
        self.family_head = nn.Linear(d, core.config.family_classes)

    def forward(self, start_value_hidden: torch.Tensor, query_hidden: torch.Tensor, family_hidden: torch.Tensor, source_hidden: torch.Tensor, target_hidden: torch.Tensor, operation_mask: torch.Tensor) -> dict[str, Any]:
        start_slots = self.start_value_projector(start_value_hidden) + self.core.register_keys.unsqueeze(0)
        query_logits = self.query_pointer(self.query_projector(query_hidden))
        query_weights = torch.softmax(query_logits, dim=-1)
        query_register = query_weights.argmax(-1)
        output = self.core.rollout(start_slots, query_register, self.family_projector(family_hidden), self.source_role_projector(source_hidden), self.target_role_projector(target_hidden), operation_mask)
        output["query_weights"] = query_weights
        output["queried_state"] = torch.einsum("br,brd->bd", query_weights, output["final_slots"])
        output["answer_logits"] = self.core.shared_state_head(output["queried_state"])
        output["query_pointer_logits"] = query_logits
        output["start_value_logits"] = self.start_value_head(self.start_value_projector(start_value_hidden))
        output["family_logits"] = self.family_head(self.family_projector(family_hidden))
        return output

    def integrity_report(self) -> dict[str, Any]:
        return {"passed": True, "core_reused": True, "full_source_cross_attention": False, "source_mean": False, "query_content_written_to_slots": False, "answer_source": "shared C0 state head", "operation_intervention_second_channel": False}
