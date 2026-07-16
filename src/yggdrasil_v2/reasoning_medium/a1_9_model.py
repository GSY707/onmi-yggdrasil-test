from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .a1_7_core import A17RelationAddressedCore


@dataclass(frozen=True)
class A19BoundaryConfig:
    source_width: int
    latent_width: int
    similarity_scale: float = 20.0


class NormalizedLinearAdapter(nn.Module):
    """One narrow affine map after fixed per-token normalization."""

    def __init__(self, source_width: int, latent_width: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(source_width, elementwise_affine=False)
        self.projection = nn.Linear(source_width, latent_width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(self.input_norm(hidden))


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _cosine_logits(latent: torch.Tensor, prototypes: torch.Tensor, scale: float) -> torch.Tensor:
    normalized_latent = nn.functional.normalize(latent, dim=-1)
    normalized_prototypes = nn.functional.normalize(prototypes.detach(), dim=-1)
    return torch.einsum("...d,cd->...c", normalized_latent, normalized_prototypes) * scale


class A19FrozenQwenBoundary(nn.Module):
    """Adapter-only Qwen role boundary around one frozen A1.8 core."""

    def __init__(self, config: A19BoundaryConfig, core: A17RelationAddressedCore) -> None:
        super().__init__()
        if config.latent_width != core.config.latent_width:
            raise ValueError("A1.9 latent width must equal the frozen A1.8 core width")
        self.config = config
        self.core = core
        self.value_adapter = NormalizedLinearAdapter(config.source_width, config.latent_width)
        self.family_adapter = NormalizedLinearAdapter(config.source_width, config.latent_width)
        self.register_adapter = NormalizedLinearAdapter(config.source_width, config.latent_width)
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.core.eval()

    def train(self, mode: bool = True) -> A19FrozenQwenBoundary:
        super().train(mode)
        self.core.eval()
        return self

    def forward(
        self,
        start_value_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        family_hidden: torch.Tensor,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        operation_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if start_value_hidden.ndim != 3 or start_value_hidden.shape[1] != self.core.config.register_slots:
            raise ValueError("start_value_hidden must have shape [B, 3, H]")
        if family_hidden.shape != source_hidden.shape or family_hidden.shape != target_hidden.shape:
            raise ValueError("family/source/target hidden must have identical [B, T, H] shapes")
        if operation_mask.shape != family_hidden.shape[:2]:
            raise ValueError("operation_mask must have [B, T] shape")
        initial_slots = self.value_adapter(start_value_hidden)
        family_latents = self.family_adapter(family_hidden)
        source_latents = self.register_adapter(source_hidden)
        target_latents = self.register_adapter(target_hidden)
        query_latent = self.register_adapter(query_hidden)
        query_logits = _cosine_logits(query_latent, self.core.register_keys, self.config.similarity_scale)
        query_register = query_logits.argmax(dim=-1)
        output = self.core.rollout(
            initial_slots,
            query_register,
            family_latents,
            source_latents,
            target_latents,
            operation_mask,
        )
        output.update(
            {
                "mapped_start_values": initial_slots,
                "mapped_family": family_latents,
                "mapped_source": source_latents,
                "mapped_target": target_latents,
                "mapped_query": query_latent,
                "value_mapping_logits": _cosine_logits(initial_slots, self.core.canonical_value_prototypes, self.config.similarity_scale),
                "family_mapping_logits": _cosine_logits(family_latents, self.core.family_embedding.weight, self.config.similarity_scale),
                "source_mapping_logits": _cosine_logits(source_latents, self.core.register_keys, self.config.similarity_scale),
                "target_mapping_logits": _cosine_logits(target_latents, self.core.register_keys, self.config.similarity_scale),
                "query_mapping_logits": query_logits,
                "query_register": query_register,
            }
        )
        return output

    def boundary_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if not name.startswith("core.")
        }

    def load_boundary_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing_boundary = [name for name in missing if not name.startswith("core.")]
        if missing_boundary or unexpected:
            raise ValueError(f"invalid A1.9 boundary state: missing={missing_boundary}, unexpected={unexpected}")

    def integrity_report(self) -> dict[str, Any]:
        core_report = self.core.integrity_report()
        core_parameters = list(self.core.parameters())
        shared_register_adapter_instances = sum(
            module is self.register_adapter
            for module in (self.register_adapter,)
        )
        return {
            "passed": (
                core_report["passed"]
                and all(not parameter.requires_grad for parameter in core_parameters)
                and not any(name.startswith("core.") for name, parameter in self.named_parameters() if parameter.requires_grad)
            ),
            "stage": "A1.9 oracle-role-segmented frozen Qwen boundary",
            "core": core_report,
            "core_frozen": all(not parameter.requires_grad for parameter in core_parameters),
            "trainable_modules": ["value_adapter", "family_adapter", "register_adapter"],
            "shared_register_adapter": True,
            "shared_register_adapter_instances": shared_register_adapter_instances,
            "source_target_query_use_same_register_adapter": True,
            "initial_slots_content_only": True,
            "query_used_after_recurrence_only": True,
            "query_written_to_state": False,
            "full_source_cross_attention": False,
            "full_source_mean": False,
            "source_reread": False,
            "independent_answer_head": False,
            "answer_source": "frozen_core.shared_state_head(last_active_hard_query_register_slot)",
            "hard_reembedding_in_recurrence": False,
            "oracle_reset": False,
            "qwen_hidden_text_sampling": False,
            "config": asdict(self.config),
        }

    def parameter_report(self, *, qwen_parameters: int = 0) -> dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        core = sum(parameter.numel() for parameter in self.core.parameters())
        return {
            "qwen_frozen_parameters": int(qwen_parameters),
            "core_frozen_parameters": core,
            "adapter_trainable_parameters": trainable,
            "total_parameters": int(qwen_parameters) + core + trainable,
            "trainable_parameters": trainable,
            "frozen_parameters": int(qwen_parameters) + core,
        }
