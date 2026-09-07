from __future__ import annotations

"""C1 source-closed latent model.

The model deliberately has a very small public data boundary: ``forward``
accepts only the frozen full-token hidden tensor and its padding mask.  All
semantic metadata used by the dataset (family, budget, AST, trace, and answer
mask) remains outside this module.
The optional trace probe is a training-only consumer of the latent
trajectory and is physically removed before deployment.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TRACE_VOCAB_SIZE = 5_597
ANSWER_CLASSES = 9
RECURRENT_STEPS = 10


@dataclass(frozen=True)
class C1Config:
    source_width: int = 2_048
    latent_width: int = 512
    slots: int = 8
    boundary_depth: int = 2
    core_depth: int = 2
    attention_heads: int = 8
    ffn_width: int = 2_048
    recurrent_steps: int = RECURRENT_STEPS
    answer_classes: int = ANSWER_CLASSES
    trace_vocab_size: int = TRACE_VOCAB_SIZE
    max_global_positions: int = 1_024
    max_local_positions: int = 512


# A descriptive alias makes the public API convenient for callers that prefer
# the longer name while preserving one canonical configuration type.
C1ModelConfig = C1Config


def _fixed_fourier(count: int, width: int) -> Tensor:
    positions = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    half = width // 2
    if half:
        frequencies = torch.exp(
            torch.linspace(0.0, math.log(10_000.0), half, dtype=torch.float32) * -1.0
        )
        values = positions * frequencies.unsqueeze(0)
        encoded = torch.cat((values.sin(), values.cos()), dim=1)
    else:
        encoded = positions.new_zeros(count, 0)
    return F.pad(encoded, (0, width - encoded.shape[1]))


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, state: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(state)) * self.up(state))


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("latent width must be divisible by attention heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def _heads(self, value: Tensor) -> Tensor:
        return value.view(value.shape[0], value.shape[1], self.heads, self.head_width).transpose(1, 2)

    def forward(self, state: Tensor) -> Tensor:
        query, key, value = self.qkv(state).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._heads(query), self._heads(key), self._heads(value), dropout_p=0.0
        )
        attended = attended.transpose(1, 2).contiguous().view_as(state)
        return self.out(attended)


class CrossAttention(nn.Module):
    """Dense latent-to-full-hidden attention used only at the Boundary."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("latent width must be divisible by attention heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def _heads(self, value: Tensor) -> Tensor:
        return value.view(value.shape[0], value.shape[1], self.heads, self.head_width).transpose(1, 2)

    def forward(self, latent: Tensor, source: Tensor, source_mask: Tensor) -> Tensor:
        q = self._heads(self.query(latent))
        k = self._heads(self.key(source))
        v = self._heads(self.value(source))
        attention_mask = torch.zeros(
            (source_mask.shape[0], 1, 1, source_mask.shape[1]),
            dtype=q.dtype,
            device=q.device,
        ).masked_fill(
            ~source_mask[:, None, None, :], torch.finfo(q.dtype).min
        )
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )
        result = result.transpose(1, 2).contiguous().view_as(latent)
        return self.out(result)


class PerceiverLayer(nn.Module):
    """One Boundary cross-attention/self-attention/SwiGLU layer."""

    def __init__(self, config: C1Config) -> None:
        super().__init__()
        width = config.latent_width
        self.cross_norm = nn.LayerNorm(width)
        self.self_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.cross = CrossAttention(width, config.attention_heads)
        self.self_attention = SelfAttention(width, config.attention_heads)
        self.ffn = SwiGLU(width, config.ffn_width)

    def forward(self, latent: Tensor, source: Tensor, source_mask: Tensor) -> Tensor:
        latent = latent + self.cross(self.cross_norm(latent), source, source_mask)
        latent = latent + self.self_attention(self.self_norm(latent))
        latent = latent + self.ffn(self.ffn_norm(latent))
        return latent


class DenseCoreLayer(nn.Module):
    """A source-closed dense self-attention/SwiGLU transition block."""

    def __init__(self, config: C1Config) -> None:
        super().__init__()
        width = config.latent_width
        self.attention_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.self_attention = SelfAttention(width, config.attention_heads)
        self.ffn = SwiGLU(width, config.ffn_width)

    def forward(self, state: Tensor) -> Tensor:
        state = state + self.self_attention(self.attention_norm(state))
        return state + self.ffn(self.ffn_norm(state))


class Boundary(nn.Module):
    """Learned full-token Boundary followed by a K-slot Perceiver."""

    def __init__(self, config: C1Config) -> None:
        super().__init__()
        if config.slots != 8:
            raise ValueError("C1 fixes the latent workspace at K=8")
        if config.boundary_depth != 2:
            raise ValueError("C1 fixes Boundary depth at two Perceiver layers")
        self.input_norm = nn.LayerNorm(config.source_width, elementwise_affine=False)
        self.projection = nn.Linear(config.source_width, config.latent_width)
        self.source_norm = nn.LayerNorm(config.latent_width)
        self.slot_seed = nn.Parameter(torch.randn(config.latent_width) * 0.02)
        self.register_buffer(
            "slot_fourier", _fixed_fourier(config.slots, config.latent_width), persistent=True
        )
        self.layers = nn.ModuleList(PerceiverLayer(config) for _ in range(config.boundary_depth))
        self.output_norm = nn.LayerNorm(config.latent_width)

    def forward(self, source_hidden: Tensor, source_mask: Tensor) -> Tensor:
        # Outside autocast (notably FP32 qualification) align cached FP16 with
        # the learner weights.  During CUDA BF16 training, leave the cache in
        # FP16 so autocast can dispatch the first linear without materializing
        # a full 67 MiB FP32 source copy.
        if not torch.is_autocast_enabled(source_hidden.device.type):
            source_hidden = source_hidden.to(dtype=self.projection.weight.dtype)
        source = self.projection(self.input_norm(source_hidden))
        source = self.source_norm(source)
        slots = self.slot_seed.view(1, 1, -1) + self.slot_fourier.to(source.dtype).unsqueeze(0)
        state = slots.expand(source.shape[0], -1, -1)
        for layer in self.layers:
            state = layer(state, source, source_mask)
        return self.output_norm(state)


class TraceProbe(nn.Module):
    """Non-autoregressive training-only probe over aligned latent states.

    ``step_alignment`` is one-based: 1 selects H1 and 10 selects H10.  The
    probe has no source input, token embedding, teacher prefix, or recurrent
    state of its own.  Position IDs are fixed coordinates supplied by the
    materializer, not task/family features.
    """

    def __init__(self, config: C1Config) -> None:
        super().__init__()
        width = config.latent_width
        self.global_position = nn.Embedding(config.max_global_positions, width)
        self.local_position = nn.Embedding(config.max_local_positions, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.interaction_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = SwiGLU(width, config.ffn_width)
        self.head = nn.Linear(width, config.trace_vocab_size)

    def forward(
        self,
        trajectory: Tensor,
        step_indices: Tensor,
        global_positions: Tensor,
        local_positions: Tensor,
    ) -> Tensor:
        if trajectory.ndim != 4:
            raise ValueError("trajectory must have shape [batch, T, slots, width]")
        if global_positions.shape != local_positions.shape or global_positions.shape != step_indices.shape:
            raise ValueError("trace position and step-alignment shapes must match")
        if global_positions.ndim != 2 or global_positions.shape[0] != trajectory.shape[0]:
            raise ValueError("trace positions must have shape [batch, tokens]")
        alignment = step_indices.to(torch.long)
        if bool((alignment < 1).any()) or bool((alignment > trajectory.shape[1]).any()):
            raise ValueError("step_alignment is one-based and must select H1...HT")
        if bool((global_positions < 0).any()) or bool((global_positions >= self.global_position.num_embeddings).any()):
            raise ValueError("global position outside frozen position table")
        if bool((local_positions < 0).any()) or bool((local_positions >= self.local_position.num_embeddings).any()):
            raise ValueError("local position outside frozen position table")
        batch = torch.arange(trajectory.shape[0], device=trajectory.device)[:, None]
        aligned = trajectory[batch, alignment - 1]
        query = self.query(
            self.global_position(global_positions.to(torch.long))
            + self.local_position(local_positions.to(torch.long))
        )
        keys = self.key(aligned)
        values = self.value(aligned)
        scores = torch.einsum("bnd,bnkd->bnk", query, keys) / math.sqrt(trajectory.shape[-1])
        pooled = torch.sum(torch.softmax(scores, dim=-1).unsqueeze(-1) * values, dim=-2)
        state = self.interaction_norm(query + pooled)
        state = state + self.ffn(self.ffn_norm(state))
        return self.head(state)


class C1Model(nn.Module):
    """Fresh C1 model with source-closed shared recurrence."""

    def __init__(self, config: C1Config = C1Config(), *, with_trace_probe: bool = True) -> None:
        super().__init__()
        if config.core_depth != 2:
            raise ValueError("C1 fixes the dense core at two layers")
        if config.recurrent_steps != RECURRENT_STEPS:
            raise ValueError("C1 fixes recurrence at T=10")
        if config.answer_classes != ANSWER_CLASSES:
            raise ValueError("C1 answer head has exactly nine raw A-I logits")
        if config.trace_vocab_size != TRACE_VOCAB_SIZE:
            raise ValueError("C1 trace vocabulary is fixed at 5597")
        self.config = config
        self.boundary = Boundary(config)
        # These are the only two core layer instances.  They are reused for
        # every one of the ten transitions; no per-step parameters exist.
        self.core_layers = nn.ModuleList(DenseCoreLayer(config) for _ in range(config.core_depth))
        self.final_norm = nn.LayerNorm(config.latent_width)
        self.answer_query = nn.Parameter(torch.randn(config.latent_width) * 0.02)
        self.answer_head = nn.Linear(config.latent_width, config.answer_classes)
        self.trace_probe = TraceProbe(config) if with_trace_probe else None

    @property
    def has_trace_probe(self) -> bool:
        return self.trace_probe is not None

    def encode(self, source_hidden: Tensor, source_mask: Tensor) -> Tensor:
        self._validate_source(source_hidden, source_mask)
        return self.boundary(source_hidden, source_mask)

    def rollout(
        self,
        h0: Tensor,
        *,
        state_shuffle_step: int | None = None,
        state_shuffle_mapping: Tensor | None = None,
        initial_slot_permutation: Tensor | None = None,
    ) -> Tensor:
        """Run T shared transitions from H0, optionally for interventions.

        ``state_shuffle_step`` is one-based and applies after H5 by contract
        (the generic API allows another step for controlled diagnostics).
        ``state_shuffle_mapping`` is an explicit batch index map supplied by
        the caller; no family information is accepted or inferred here.
        """
        if h0.ndim != 3 or h0.shape[1] != self.config.slots or h0.shape[2] != self.config.latent_width:
            raise ValueError("h0 must have shape [batch, 8, latent_width]")
        if initial_slot_permutation is not None:
            permutation = initial_slot_permutation.to(device=h0.device, dtype=torch.long)
            if permutation.ndim != 1 or permutation.numel() != self.config.slots:
                raise ValueError("initial slot permutation must have shape [8]")
            if not torch.equal(torch.sort(permutation).values, torch.arange(self.config.slots, device=h0.device)):
                raise ValueError("initial slot permutation must be a permutation of 0..7")
            h0 = h0[:, permutation]
        state = h0
        trajectory: list[Tensor] = []
        for step in range(1, self.config.recurrent_steps + 1):
            for layer in self.core_layers:
                state = layer(state)
            if state_shuffle_step == step and state_shuffle_mapping is not None:
                mapping = state_shuffle_mapping.to(device=state.device, dtype=torch.long)
                if mapping.ndim != 1 or mapping.shape[0] != state.shape[0]:
                    raise ValueError("state shuffle mapping must have shape [batch]")
                if not torch.equal(torch.sort(mapping).values, torch.arange(state.shape[0], device=state.device)):
                    raise ValueError("state shuffle mapping must be a batch permutation")
                state = state[mapping]
            # A diagnostic trajectory records the state owned by each output
            # row after any intervention at that numbered step.  Thus H5 is
            # post-shuffle and H6 is its actual continuation.
            trajectory.append(state)
        if state_shuffle_step is not None and not 1 <= state_shuffle_step <= self.config.recurrent_steps:
            raise ValueError("state_shuffle_step must select H1...H10")
        return torch.stack(trajectory, dim=1)

    def logits_from_state(self, state: Tensor) -> Tensor:
        if state.ndim != 3 or state.shape[1:] != (self.config.slots, self.config.latent_width):
            raise ValueError("state must have shape [batch, 8, latent_width]")
        state = self.final_norm(state)
        scores = torch.einsum("bkd,d->bk", state, self.answer_query) / math.sqrt(state.shape[-1])
        pooled = torch.sum(state * torch.softmax(scores, dim=1).unsqueeze(-1), dim=1)
        return self.answer_head(pooled)

    def forward(
        self,
        source_hidden: Tensor,
        source_mask: Tensor,
        *,
        return_trajectory: bool = False,
        initial_slot_permutation: Tensor | None = None,
        state_shuffle_step: int | None = None,
        state_shuffle_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Answer from source hidden, with optional caller-supplied interventions.

        The keyword interventions are diagnostics only.  In particular, no
        family, budget, answer mask, AST, or trace value is accepted here.  The
        source mask is only the padding structure of the public source text.
        """
        self._validate_source(source_hidden, source_mask)
        h0 = self.boundary(source_hidden, source_mask)
        trajectory = self.rollout(
            h0,
            initial_slot_permutation=initial_slot_permutation,
            state_shuffle_step=state_shuffle_step,
            state_shuffle_mapping=state_shuffle_indices,
        )
        result: dict[str, Tensor] = {
            "logits": self.logits_from_state(trajectory[:, -1]),
            "h0": h0,
            "final_state": trajectory[:, -1],
        }
        if return_trajectory:
            result["trajectory"] = trajectory
        return result

    def trace_logits(
        self,
        trajectory: Tensor,
        step_indices: Tensor,
        global_positions: Tensor,
        local_positions: Tensor,
    ) -> Tensor:
        if self.trace_probe is None:
            raise RuntimeError("trace probe was physically removed")
        return self.trace_probe(trajectory, step_indices, global_positions, local_positions)

    def _validate_source(self, source_hidden: Tensor, source_mask: Tensor) -> None:
        if source_hidden.ndim != 3 or source_hidden.shape[-1] != self.config.source_width:
            raise ValueError("source_hidden must have shape [batch, full_tokens, source_width]")
        if source_hidden.shape[1] < 1:
            raise ValueError("full-token source hidden cannot be empty")
        if source_mask.shape != source_hidden.shape[:2] or source_mask.dtype != torch.bool:
            raise ValueError("source_mask must be bool [batch, full_tokens]")
        if bool((source_mask.sum(dim=1) < 1).any()):
            raise ValueError("every source row must contain at least one public token")

    def integrity_report(self) -> dict[str, Any]:
        names = [name.lower() for name, _ in self.named_parameters()]
        forbidden = sorted(
            name
            for name in names
            if any(token in name for token in ("family", "budget", "mask", "ast", "trace_input", "teacher"))
        )
        core_ids = [id(layer) for layer in self.core_layers]
        return {
            "passed": not forbidden and len(self.core_layers) == 2 and self.config.recurrent_steps == 10,
            "config": asdict(self.config),
            "source_boundary": "full-token learned projection then K=8 Perceiver",
            "source_closed_after_h0": True,
            "core_depth": len(self.core_layers),
            "shared_core_reused_for_steps": self.config.recurrent_steps,
            "core_instance_ids": core_ids,
            "answer_logits": "raw A-I nine logits from H10 only",
            "trace_probe_training_only": self.trace_probe is not None,
            "forbidden_parameter_names": forbidden,
        }

    def parameter_report(self) -> dict[str, Any]:
        rows = [
            {"name": name, "parameters": parameter.numel(), "trainable": parameter.requires_grad}
            for name, parameter in self.named_parameters()
        ]
        trace = sum(row["parameters"] for row in rows if row["name"].startswith("trace_probe."))
        total = sum(row["parameters"] for row in rows)
        return {
            "total_parameters": total,
            "trainable_parameters": sum(row["parameters"] for row in rows if row["trainable"]),
            "trace_probe_parameters": trace,
            "deployment_parameters": total - trace,
            "parameter_graph": rows,
        }


# Common spelling used by training callers.
C1ClosureModel = C1Model
C1LatentReasoner = C1Model


__all__ = [
    "ANSWER_CLASSES",
    "RECURRENT_STEPS",
    "TRACE_VOCAB_SIZE",
    "Boundary",
    "C1Config",
    "C1ModelConfig",
    "C1Model",
    "C1ClosureModel",
    "C1LatentReasoner",
    "DenseCoreLayer",
    "TraceProbe",
]
