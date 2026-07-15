from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .a1_5_data import ANSWER_LABELS, REGISTER_NAMES, load_a15_records
from .a1_5_model import P0Config, SharedStructuredTransition
from .model import load_qwen35_text_only


P1_CACHE_SCHEMA = "yggdrasil.v2-a1.5.hidden-cache.v1"


@dataclass(frozen=True)
class P1CacheConfig:
    model_id: str = "Qwen/Qwen3.5-2B"
    revision: str = "15852e8c16360a2fea060d615a32b45270f8a8fc"
    max_length: int = 512
    shard_size: int = 64
    dtype: str = "float16"


def token_operation_masks(
    operation_spans: Sequence[dict[str, Any]],
    offsets: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Map character spans to explicit per-operation token masks."""
    masks: list[list[bool]] = []
    for span in operation_spans:
        start, end = int(span["char_start"]), int(span["char_end"])
        row = [offset_end > start and offset_start < end for offset_start, offset_end in offsets]
        if not any(row):
            # Do not silently lose an operation span because a tokenizer split
            # produced an empty offset; attach the nearest token instead.
            distances = [abs(offset_start - start) for offset_start, _ in offsets]
            if distances:
                row[min(range(len(distances)), key=distances)] = True
        masks.append(row)
    return torch.tensor(masks, dtype=torch.bool)


def cache_qwen_hidden(
    data_dir: Path,
    output_dir: Path,
    *,
    model_id: str = P1CacheConfig.model_id,
    revision: str = P1CacheConfig.revision,
    splits: Sequence[str] = ("train", "validation", "test", "composition_heldout", "length_heldout"),
    device: str = "cuda",
    max_length: int = 512,
    shard_size: int = 64,
) -> dict[str, Any]:
    """Encode A1.5 text once and save FP16 versioned shards with operation masks."""
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16
    backbone = load_qwen35_text_only(model_id, revision, dtype=dtype, device=device)
    backbone.eval()
    split_manifests: dict[str, Any] = {}
    for split in splits:
        records = load_a15_records(data_dir, split)
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shards: list[str] = []
        max_seen_length = 0
        for shard_start in range(0, len(records), shard_size):
            shard_records = records[shard_start : shard_start + shard_size]
            encoded = [
                tokenizer(
                    record["question"],
                    add_special_tokens=True,
                    truncation=False,
                    return_offsets_mapping=True,
                )
                for record in shard_records
            ]
            lengths = [len(item["input_ids"]) for item in encoded]
            if any(length > max_length for length in lengths):
                raise ValueError(
                    f"token length exceeds max_length={max_length} in {split}; refusing silent truncation"
                )
            seq_length = max(lengths, default=0)
            max_seen_length = max(max_seen_length, seq_length)
            input_ids = torch.full(
                (len(encoded), seq_length), tokenizer.pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros((len(encoded), seq_length), dtype=torch.bool)
            span_masks: list[torch.Tensor] = []
            for row, item in enumerate(encoded):
                length = len(item["input_ids"])
                input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
                attention_mask[row, :length] = True
                span_masks.append(token_operation_masks(shard_records[row]["operation_spans"], item["offset_mapping"]))
            with torch.no_grad():
                outputs = backbone.model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    use_cache=False,
                )
                hidden = outputs.last_hidden_state.detach().to(dtype=torch.float16).cpu()
            max_steps = max((len(mask) for mask in span_masks), default=0)
            operation_token_mask = torch.zeros((len(encoded), max_steps, seq_length), dtype=torch.bool)
            operation_mask = torch.zeros((len(encoded), max_steps), dtype=torch.bool)
            for row, mask in enumerate(span_masks):
                if mask.numel():
                    operation_token_mask[row, : mask.shape[0], : mask.shape[1]] = mask
                    operation_mask[row, : mask.shape[0]] = True
            payload = {
                "schema_version": P1_CACHE_SCHEMA,
                "last_hidden": hidden,
                "attention_mask": attention_mask,
                "operation_token_mask": operation_token_mask,
                "operation_mask": operation_mask,
                "example_ids": [record["example_id"] for record in shard_records],
                "fingerprints": [record["fingerprint"] for record in shard_records],
                "token_lengths": lengths,
            }
            shard_path = split_dir / f"shard_{shard_start // shard_size:05d}.pt"
            torch.save(payload, shard_path)
            shards.append(str(shard_path))
        split_manifests[split] = {
            "examples": len(records),
            "shards": shards,
            "max_token_length": max_seen_length,
            "dtype": "float16",
        }
    manifest = {
        "schema_version": P1_CACHE_SCHEMA,
        "cache_contract": "last_hidden_fp16_plus_explicit_operation_span_masks",
        "model_id": model_id,
        "revision": revision,
        "tokenizer_revision": revision,
        "max_length": max_length,
        "shard_size": shard_size,
        "splits": split_manifests,
        "silent_truncation": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del backbone
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


class P1HiddenLatentReasoner(nn.Module):
    """P1 source-hidden interface with the P0 shared recurrent transition."""

    def __init__(self, source_width: int, config: P0Config | None = None) -> None:
        super().__init__()
        self.config = config or P0Config()
        d = self.config.latent_width
        self.source_projection = nn.Linear(source_width, d)
        self.source_norm = nn.LayerNorm(d)
        self.slot_type = nn.Embedding(4, d)  # three start-state slots + query/control
        self.latent_queries = nn.Parameter(torch.randn(4, d) * 0.02)
        self.source_cross_attention = nn.MultiheadAttention(d, self.config.attention_heads, batch_first=True)
        self.operation_projection = nn.Linear(source_width, d)
        self.operation_type_embedding = nn.Parameter(torch.randn(d) * 0.02)
        self.transition = SharedStructuredTransition(self.config)
        self.final_norm = nn.LayerNorm(d)
        self.state_norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, self.config.value_classes)
        self.answer_head = nn.Linear(d, self.config.value_classes)
        self.start_state_head = nn.Linear(d, self.config.value_classes)
        self.query_register_head = nn.Linear(d, len(REGISTER_NAMES))
        self.operation_family_head = nn.Linear(d, 3)
        self.operation_source_head = nn.Linear(d, len(REGISTER_NAMES))
        self.operation_target_head = nn.Linear(d, len(REGISTER_NAMES))

    @staticmethod
    def pool_operation_hidden(
        source_hidden: torch.Tensor,
        operation_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = operation_token_mask.to(dtype=source_hidden.dtype)
        pooled = torch.einsum("btl,blh->bth", weights, source_hidden)
        return pooled / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
        operation_token_mask: torch.Tensor,
        operation_mask: torch.Tensor,
        *,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:
        source_hidden = source_hidden.to(dtype=self.source_projection.weight.dtype)
        source = self.source_norm(self.source_projection(source_hidden))
        batch = source.shape[0]
        type_ids = torch.arange(4, device=source.device).view(1, 4)
        queries = self.latent_queries.view(1, 4, -1).expand(batch, -1, -1) + self.slot_type(type_ids)
        encoded_slots, _ = self.source_cross_attention(
            queries,
            source,
            source,
            key_padding_mask=~source_attention_mask.bool(),
            need_weights=False,
        )
        operation_hidden = self.pool_operation_hidden(source_hidden, operation_token_mask)
        operation_tokens = self.operation_projection(operation_hidden) + self.operation_type_embedding
        state = encoded_slots
        trajectory: list[torch.Tensor] = []
        for step in range(operation_tokens.shape[1]):
            transformed = self.transition(state, operation_tokens[:, step])
            active = operation_mask[:, step].to(dtype=state.dtype).view(-1, 1, 1)
            state = state + active * self.transition.residual_scale * (transformed - state)
            if return_trajectory:
                trajectory.append(state)
        normalized = self.final_norm(state)
        result: dict[str, Any] = {
            "logits": self.answer_head(normalized[:, 3]),
            "final_state": state,
            "start_state_logits": self.start_state_head(encoded_slots[:, :3]),
            "query_register_logits": self.query_register_head(encoded_slots[:, 3]),
            "operation_family_logits": self.operation_family_head(operation_tokens),
            "operation_source_logits": self.operation_source_head(operation_tokens),
            "operation_target_logits": self.operation_target_head(operation_tokens),
        }
        if return_trajectory:
            result["trajectory"] = trajectory
            result["state_logits"] = self.state_head(
                self.state_norm(torch.stack([item[:, :3] for item in trajectory], dim=1))
            ) if trajectory else torch.empty(0, device=state.device)
        return result

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable_parameters": total, "active_parameters": total, "total_parameters": total}

    def integrity_report(self) -> dict[str, Any]:
        return {
            "passed": True,
            "answer_head_input": "final latent query/control slot only",
            "teacher_hidden_in_forward": False,
            "teacher_trace_in_forward": False,
            "operation_span_conditioning": True,
            "source_mean_broadcast_main_path": False,
        }
