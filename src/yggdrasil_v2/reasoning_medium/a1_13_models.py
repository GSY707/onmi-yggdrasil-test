from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .a1_10_model import A110RecurrentBlock
from .a1_12_models import A112ReasonerConfig, A112RootCauseReasoner


@dataclass(frozen=True)
class A113ReasonerConfig:
    structured_transition: bool
    prototype_closure_objective: bool
    query_coupled_answer: bool = False
    training_auxiliary: str = "none"
    latent_width: int = 256
    workspace_slots: int = 8
    entity_slots: int = 3
    attention_heads: int = 8
    ffn_width: int = 1024
    recurrent_layers: int = 2
    value_classes: int = 10
    family_classes: int = 2
    state_fields: int = 3
    maximum_operation_steps: int = 32


class A113RelationAddressedTransition(nn.Module):
    """One shared soft content update with exact register addressing."""

    def __init__(self, config: A113ReasonerConfig) -> None:
        super().__init__()
        d = config.latent_width
        self.entity_slots = config.entity_slots
        self.family_embedding = nn.Embedding(config.family_classes, d)
        self.operator = nn.Sequential(
            nn.Linear(3 * d, config.ffn_width),
            nn.GELU(),
            nn.Linear(config.ffn_width, 2 * d + 2),
        )

    def forward(
        self,
        slots: torch.Tensor,
        family: torch.Tensor,
        source_register: torch.Tensor,
        target_register: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        entity = slots[:, : self.entity_slots]
        source_weights = nn.functional.one_hot(
            source_register, num_classes=self.entity_slots
        ).to(dtype=slots.dtype)
        target_weights = nn.functional.one_hot(
            target_register, num_classes=self.entity_slots
        ).to(dtype=slots.dtype)
        source_read = torch.einsum("br,brd->bd", source_weights, entity)
        target_read = torch.einsum("br,brd->bd", target_weights, entity)
        update = self.operator(
            torch.cat((source_read, target_read, self.family_embedding(family)), dim=-1)
        )
        proposed_source, proposed_target, gate_logits = torch.split(
            update, [slots.shape[-1], slots.shape[-1], 2], dim=-1
        )
        source_gate = torch.sigmoid(gate_logits[:, :1])
        target_gate = torch.sigmoid(gate_logits[:, 1:])
        next_entity = entity + source_weights.unsqueeze(-1) * source_gate.unsqueeze(1) * (
            proposed_source.unsqueeze(1) - entity
        )
        next_entity = next_entity + target_weights.unsqueeze(-1) * target_gate.unsqueeze(1) * (
            proposed_target.unsqueeze(1) - next_entity
        )
        candidate = torch.cat((next_entity, slots[:, self.entity_slots :]), dim=1)
        return active.to(dtype=slots.dtype).view(-1, 1, 1) * candidate + (
            ~active
        ).to(dtype=slots.dtype).view(-1, 1, 1) * slots


class A113TransitionClosureReasoner(A112RootCauseReasoner):
    """A1.12-BOTH with only transition identity and closure objective varied."""

    def __init__(self, config: A113ReasonerConfig) -> None:
        if config.training_auxiliary not in {
            "none",
            "queried_answer",
            "full_state",
            "full_trajectory_state",
        }:
            raise ValueError("unknown A1.18 training auxiliary")
        if config.training_auxiliary != "none" and not config.query_coupled_answer:
            raise ValueError("training-only auxiliaries require the query-coupled answer path")
        self.a113_config = config
        super().__init__(
            A112ReasonerConfig(
                entity_addressable=True,
                aligned_operation_cursor=True,
                latent_width=config.latent_width,
                workspace_slots=config.workspace_slots,
                entity_slots=config.entity_slots,
                attention_heads=config.attention_heads,
                ffn_width=config.ffn_width,
                recurrent_layers=config.recurrent_layers,
                value_classes=config.value_classes,
                state_fields=config.state_fields,
                maximum_operation_steps=config.maximum_operation_steps,
            )
        )
        if config.structured_transition:
            self.recurrent_reasoner = nn.ModuleList()
            self.relation_transition: A113RelationAddressedTransition | None = (
                A113RelationAddressedTransition(config)
            )
        else:
            self.relation_transition = None
        self.training_auxiliary_head: nn.Linear | None = None
        if config.query_coupled_answer:
            if not config.structured_transition or not config.prototype_closure_objective:
                raise ValueError(
                    "query-coupled answer is reserved for the closed structured A1.15 core"
                )
            independent_answer_head = self.answer_head
            self.answer_head = None
            if config.training_auxiliary == "queried_answer":
                # Reuse the exact head that the independent A1.13 arm would have
                # received. This keeps both the shared initialization and the QAUX
                # positive-control initialization paired bit-for-bit.
                self.training_auxiliary_head = independent_answer_head
            elif config.training_auxiliary in {"full_state", "full_trajectory_state"}:
                self.training_auxiliary_head = nn.Linear(
                    config.latent_width,
                    config.entity_slots * config.value_classes,
                )

    @property
    def arm(self) -> str:
        if self.a113_config.query_coupled_answer:
            suffix = {
                "none": "",
                "queried_answer": "-QAUX",
                "full_state": "-SAUX",
                "full_trajectory_state": "-TSAUX",
            }[self.a113_config.training_auxiliary]
            return f"STRUCTURED-CLOSURE-COUPLED{suffix}"
        transition = "STRUCTURED" if self.a113_config.structured_transition else "GENERIC"
        objective = "CLOSURE" if self.a113_config.prototype_closure_objective else "CE"
        return f"{transition}-{objective}"

    def closure_targets(self, state_targets: torch.Tensor) -> torch.Tensor:
        if self.entity_initialization is None:
            raise RuntimeError("A1.13 closure requires entity initialization")
        batch, steps, registers = state_targets.shape
        if registers != self.a113_config.entity_slots:
            raise ValueError("A1.13 state targets must contain the three entity registers")
        values = nn.functional.one_hot(
            state_targets, num_classes=self.a113_config.value_classes
        ).to(dtype=self.entity_initialization.weight.dtype)
        register_identity = torch.eye(
            registers, dtype=values.dtype, device=values.device
        ).view(1, 1, registers, registers).expand(batch, steps, -1, -1)
        return self.entity_initialization(torch.cat((values, register_identity), dim=-1))

    def forward(
        self,
        start_values: torch.Tensor,
        query_register: torch.Tensor,
        operation_family: torch.Tensor,
        operation_source: torch.Tensor,
        operation_target: torch.Tensor,
        operation_mask: torch.Tensor,
        recurrent_steps: int,
        *,
        disable_recurrence: bool = False,
        start_value_permutation: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if recurrent_steps < 1:
            raise ValueError("A1.13 recurrent_steps must be positive")
        input_start_values = start_values
        if start_value_permutation is not None:
            input_start_values = start_values[:, start_value_permutation]
        typed, valid = self._typed_source(
            input_start_values,
            query_register,
            operation_family,
            operation_source,
            operation_target,
            operation_mask,
        )
        typed = typed.to(dtype=self.source_projection.weight.dtype)
        source = self.read_source_norm(self.source_projection(self.source_input_norm(typed)))
        slots = self._read(source, valid, input_start_values)
        initial_slots = slots
        trajectory: list[torch.Tensor] = []
        operation_steps = operation_mask.shape[1]
        for step in range(recurrent_steps):
            if not disable_recurrence:
                if self.relation_transition is not None:
                    index = min(step, operation_steps - 1)
                    slots = self.relation_transition(
                        slots,
                        operation_family[:, index],
                        operation_source[:, index],
                        operation_target[:, index],
                        operation_mask[:, index] if step < operation_steps else torch.zeros_like(operation_mask[:, 0]),
                    )
                else:
                    step_source, padding_mask = self._step_source(
                        source, valid, step, operation_steps
                    )
                    for block in self.recurrent_reasoner:
                        slots = block(slots, step_source, padding_mask)
            trajectory.append(slots)
        trajectory_tensor = torch.stack(trajectory, dim=1)
        normalized = self.output_norm(trajectory_tensor)
        state_logits = self.state_head(normalized[:, :, : self.a113_config.entity_slots])
        final_pooled = self.output_norm(slots).mean(dim=1)
        if self.a113_config.query_coupled_answer:
            gather = query_register.view(-1, 1, 1).expand(
                -1, 1, self.a113_config.value_classes
            )
            answer_logits = state_logits[:, -1].gather(1, gather).squeeze(1)
        else:
            if self.answer_head is None:
                raise RuntimeError("A1.13 independent answer head is missing")
            answer_logits = self.answer_head(final_pooled)
        training_auxiliary_logits: torch.Tensor | None = None
        if self.training and self.training_auxiliary_head is not None:
            auxiliary_source = (
                normalized.mean(dim=2)
                if self.a113_config.training_auxiliary == "full_trajectory_state"
                else final_pooled
            )
            training_auxiliary_logits = self.training_auxiliary_head(auxiliary_source)
            if self.a113_config.training_auxiliary in {
                "full_state",
                "full_trajectory_state",
            }:
                training_auxiliary_logits = training_auxiliary_logits.view(
                    slots.shape[0],
                    *(
                        (trajectory_tensor.shape[1],)
                        if self.a113_config.training_auxiliary
                        == "full_trajectory_state"
                        else ()
                    ),
                    self.a113_config.entity_slots,
                    self.a113_config.value_classes,
                )
        return {
            "initial_slots": initial_slots,
            "final_slots": slots,
            "trajectory": trajectory_tensor,
            "state_logits": state_logits,
            "answer_logits": answer_logits,
            "training_auxiliary_logits": training_auxiliary_logits,
        }

    def integrity_report(self) -> dict[str, Any]:
        generic_blocks = sum(isinstance(module, A110RecurrentBlock) for module in self.modules())
        structured_blocks = sum(
            isinstance(module, A113RelationAddressedTransition) for module in self.modules()
        )
        expected_generic = 0 if self.a113_config.structured_transition else self.a113_config.recurrent_layers
        expected_structured = 1 if self.a113_config.structured_transition else 0
        auxiliary_outputs = (
            self.training_auxiliary_head.out_features
            if self.training_auxiliary_head is not None
            else 0
        )
        expected_auxiliary_outputs = {
            "none": 0,
            "queried_answer": self.a113_config.value_classes,
            "full_state": (
                self.a113_config.entity_slots * self.a113_config.value_classes
            ),
            "full_trajectory_state": (
                self.a113_config.entity_slots * self.a113_config.value_classes
            ),
        }[self.a113_config.training_auxiliary]
        return {
            "passed": (
                generic_blocks == expected_generic
                and structured_blocks == expected_structured
                and auxiliary_outputs == expected_auxiliary_outputs
            ),
            "arm": self.arm,
            "exact_symbolic_oracle_roles": True,
            "qwen_or_boundary_adapter_loaded": False,
            "entity_addressable_state": True,
            "aligned_operation_cursor": True,
            "structured_transition": self.a113_config.structured_transition,
            "relation_addressing": "exact register one-hot" if self.a113_config.structured_transition else False,
            "copy_swap_branches": False,
            "discrete_value_executor": False,
            "generic_recurrent_block_instances": generic_blocks,
            "structured_transition_instances": structured_blocks,
            "transition_weights_shared_across_steps": True,
            "prototype_closure_objective": self.a113_config.prototype_closure_objective,
            "query_coupled_answer": self.a113_config.query_coupled_answer,
            "independent_answer_head": not self.a113_config.query_coupled_answer,
            "training_auxiliary": self.a113_config.training_auxiliary,
            "training_auxiliary_head_outputs": auxiliary_outputs,
            "training_auxiliary_active_in_eval": False,
            "hard_reembedding_in_forward": False,
            "teacher_state_in_forward": False,
            "state_loss": "per-step CE",
            "answer_source": (
                "shared_state_head(last_recurrent_queried_entity_slot)"
                if self.a113_config.query_coupled_answer
                else "final_workspace_mean_only"
            ),
            "config": asdict(self.a113_config),
        }
