from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_qwen35_text_only(
    model_id: str,
    revision: str,
    *,
    dtype: torch.dtype,
    device: str,
) -> nn.Module:
    """Load only Qwen3.5 language weights from the official multimodal repo."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    composite_config = AutoConfig.from_pretrained(model_id, revision=revision)
    text_config = composite_config.text_config
    index_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            revision=revision,
        )
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    source_keys = [key for key in weight_map if key.startswith("model.language_model.")]
    shard_paths = {
        filename: Path(
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
            )
        )
        for filename in sorted({weight_map[key] for key in source_keys})
    }
    state: dict[str, torch.Tensor] = {}
    for filename, shard_path in shard_paths.items():
        keys = [key for key in source_keys if weight_map[key] == filename]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for source_key in keys:
                target_key = source_key.removeprefix("model.language_model.")
                state[f"model.{target_key}"] = handle.get_tensor(source_key).to(dtype=dtype)

    # The official composite checkpoint ties lm_head to text embeddings and
    # therefore stores only the embedding tensor.
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen3_5ForCausalLM(text_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.to(device)
    return model


@dataclass(frozen=True)
class LatentConfig:
    latent_tokens: int = 8
    recurrent_steps: int = 8
    recurrent_blocks: int = 2
    cross_attention_heads: int = 8
    answer_classes: int = 10
    source_adapter_backend: str = "none"
    source_adapter_bottleneck: int = 512
    source_mean_init: bool = False
    transition_gate_init: float = -2.0
    transition_backend: str = "copied_qwen"
    transition_bottleneck: int = 512
    answer_pooling: str = "mean"
    source_read_each_step: bool = False
    source_read_gate_init: float = -2.0


class LatentMLPTransition(nn.Module):
    """Small identity-initialized transition for a latent-only diagnostic path."""

    def __init__(self, hidden_size: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.up = nn.Linear(hidden_size, bottleneck)
        self.down = nn.Linear(bottleneck, hidden_size)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        delta = self.down(torch.nn.functional.silu(self.up(self.norm(state))))
        return state + delta


class LatentReasoner(nn.Module):
    """Continuous recurrent workspace initialized from a frozen Qwen backbone.

    The module receives only frozen-base hidden states. The answer head has no
    reference to input ids, labels, teacher traces, or the base LM head.
    """

    def __init__(self, base_model: nn.Module, config: LatentConfig) -> None:
        super().__init__()
        decoder = text_decoder(base_model)
        hidden_size = int(decoder.config.hidden_size)
        if hidden_size % config.cross_attention_heads:
            raise ValueError("hidden_size must be divisible by cross_attention_heads")
        if config.recurrent_blocks < 0 or config.recurrent_blocks > len(decoder.layers):
            raise ValueError("recurrent_blocks is outside the backbone layer range")
        if config.transition_backend not in {"copied_qwen", "mlp"}:
            raise ValueError(f"Unknown transition_backend: {config.transition_backend}")
        if config.transition_bottleneck < 1:
            raise ValueError("transition_bottleneck must be positive")
        if config.source_adapter_backend not in {"none", "mlp"}:
            raise ValueError(f"Unknown source_adapter_backend: {config.source_adapter_backend}")
        if config.source_adapter_bottleneck < 1:
            raise ValueError("source_adapter_bottleneck must be positive")
        if config.answer_pooling not in {"mean", "flatten"}:
            raise ValueError(f"Unknown answer_pooling: {config.answer_pooling}")

        self.config = config
        self.hidden_size = hidden_size
        self.latent_queries = nn.Parameter(torch.empty(config.latent_tokens, hidden_size))
        self.step_embeddings = nn.Parameter(torch.zeros(config.recurrent_steps, hidden_size))
        nn.init.normal_(self.latent_queries, mean=0.0, std=float(decoder.config.initializer_range))
        nn.init.normal_(self.step_embeddings, mean=0.0, std=float(decoder.config.initializer_range))

        self.input_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            config.cross_attention_heads,
            batch_first=True,
        )
        self.source_adapter = (
            LatentMLPTransition(hidden_size, config.source_adapter_bottleneck)
            if config.source_adapter_backend == "mlp"
            else None
        )
        self.cross_gate = nn.Parameter(torch.tensor(0.0))
        self.source_read_gate = nn.Parameter(torch.tensor(config.source_read_gate_init))
        if config.transition_backend == "mlp":
            selected_indices = []
        else:
            layer_types = getattr(decoder.config, "layer_types", None)
            if layer_types is not None:
                full_attention_indices = [
                    index for index, layer_type in enumerate(layer_types) if layer_type == "full_attention"
                ]
            else:
                full_attention_indices = []
            if config.recurrent_blocks == 0:
                selected_indices = []
            elif len(full_attention_indices) >= config.recurrent_blocks:
                selected_indices = full_attention_indices[-config.recurrent_blocks :]
            else:
                selected_indices = list(range(len(decoder.layers) - config.recurrent_blocks, len(decoder.layers)))
        self.recurrent_layer_indices = tuple(selected_indices)
        transition_count = config.recurrent_blocks if config.transition_backend == "mlp" else len(selected_indices)
        self.transition_gates = nn.Parameter(torch.full((transition_count,), config.transition_gate_init))
        if config.transition_backend == "mlp":
            self.recurrent_layers = nn.ModuleList(
                [LatentMLPTransition(hidden_size, config.transition_bottleneck) for _ in range(transition_count)]
            )
        else:
            self.recurrent_layers = nn.ModuleList(
                [copy.deepcopy(decoder.layers[index]) for index in selected_indices]
            )
        self.rotary_emb = copy.deepcopy(decoder.rotary_emb)
        self.final_norm = copy.deepcopy(decoder.norm)
        pooled_size = hidden_size if config.answer_pooling == "mean" else config.latent_tokens * hidden_size
        self.answer_head = nn.Linear(pooled_size, config.answer_classes)
        self.state_head = nn.Linear(pooled_size, 3 * config.answer_classes)
        self.query_state_head = nn.Linear(pooled_size, config.answer_classes)
        self.answer_head_input_description = (
            "final latent mean only"
            if config.answer_pooling == "mean"
            else "final latent tokens flattened"
        )

        # Frozen encoder runs in bf16, but the small trainable reasoner uses fp32
        # for stable local smoke/probe training.
        self.float()
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def _intervene(self, state: torch.Tensor, mode: str, batch_permutation: torch.Tensor | None) -> torch.Tensor:
        if mode == "none":
            return state
        if mode == "zero":
            return torch.zeros_like(state)
        if mode == "batch_shuffle":
            if batch_permutation is None:
                batch_permutation = torch.arange(state.shape[0] - 1, -1, -1, device=state.device)
            return state[batch_permutation]
        if mode == "token_shuffle":
            return state.flip(dims=(1,))
        raise ValueError(f"Unknown intervention mode: {mode}")

    def forward(
        self,
        source_hidden: torch.Tensor,
        source_attention_mask: torch.Tensor,
        *,
        intervention: str = "none",
        intervention_step: int | None = None,
        truncate_steps: int | None = None,
        batch_permutation: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if source_hidden.ndim != 3 or source_attention_mask.ndim != 2:
            raise ValueError("source_hidden/mask must be [batch, tokens, hidden] and [batch, tokens]")
        source_hidden = source_hidden.to(dtype=self.latent_queries.dtype)
        batch_size = source_hidden.shape[0]
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = ~source_attention_mask.bool()
        source_context = self.input_norm(source_hidden)
        if self.source_adapter is not None:
            source_context = self.source_adapter(source_context)
        readout, _ = self.cross_attention(
            queries,
            source_context,
            source_context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        state = queries + torch.sigmoid(self.cross_gate) * readout
        if self.config.source_mean_init:
            source_mask = source_attention_mask.to(dtype=source_hidden.dtype).unsqueeze(-1)
            source_summary = (source_hidden * source_mask).sum(dim=1) / source_mask.sum(dim=1).clamp_min(1.0)
            state = state + source_summary.unsqueeze(1)

        requested_steps = self.config.recurrent_steps if truncate_steps is None else truncate_steps
        if requested_steps < 0 or requested_steps > self.config.recurrent_steps:
            raise ValueError("truncate_steps must be within the configured recurrent step range")

        latent_length = self.config.latent_tokens
        position_ids = torch.arange(latent_length, device=state.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(state, position_ids)
        full_attention_mask = torch.zeros(
            (batch_size, 1, latent_length, latent_length),
            dtype=state.dtype,
            device=state.device,
        )
        trajectory: list[torch.Tensor] = []
        for step in range(requested_steps):
            state = state + self.step_embeddings[step].view(1, 1, -1)
            if self.config.source_read_each_step:
                step_readout, _ = self.cross_attention(
                    state,
                    source_context,
                    source_context,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                state = state + torch.sigmoid(self.source_read_gate) * step_readout
            for layer_index, layer in enumerate(self.recurrent_layers):
                if self.config.transition_backend == "mlp":
                    transformed = layer(state)
                else:
                    transformed = layer(
                        state,
                        attention_mask=full_attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                    )
                gate = torch.sigmoid(self.transition_gates[layer_index]).to(dtype=state.dtype)
                state = state + gate * (transformed - state)
            if intervention != "none" and intervention_step == step:
                state = self._intervene(state, intervention, batch_permutation)
            if return_trajectory:
                trajectory.append(state)

        if intervention != "none" and intervention_step is None:
            state = self._intervene(state, intervention, batch_permutation)
        normalized_state = self.final_norm(state)
        pooled = normalized_state.mean(dim=1) if self.config.answer_pooling == "mean" else normalized_state.reshape(batch_size, -1)
        logits = self.answer_head(pooled)
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {"logits": logits, "final_state": state}
        if return_trajectory:
            result["trajectory"] = trajectory
        return result

    def parameter_report(self) -> dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"trainable_parameters": trainable, "active_parameters": trainable}


def frozen_parameter_report(base_model: nn.Module) -> dict[str, int]:
    decoder = text_decoder(base_model)
    return {
        "frozen_backbone_parameters": sum(parameter.numel() for parameter in base_model.parameters()),
        "active_frozen_text_parameters": sum(parameter.numel() for parameter in decoder.parameters()),
        "frozen_backbone_trainable_parameters": sum(
            parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad
        ),
    }


def text_decoder(base_model: nn.Module) -> nn.Module:
    """Return the text decoder without activating a multimodal boundary path."""
    root = base_model.model
    if hasattr(root, "language_model"):
        return root.language_model
    return root


def assert_no_answer_bypass(reasoner: LatentReasoner) -> dict[str, Any]:
    module_names = [name for name, _ in reasoner.named_modules()]
    parameter_names = [name for name, _ in reasoner.named_parameters()]
    forbidden = ("lm_head", "input_ids", "labels", "teacher", "trace")
    hits = sorted({name for name in module_names + parameter_names if any(token in name for token in forbidden)})
    return {
        "passed": not hits,
        "forbidden_name_hits": hits,
        "answer_head_input": reasoner.answer_head_input_description,
        "hidden_text_token_sampling": False,
    }
