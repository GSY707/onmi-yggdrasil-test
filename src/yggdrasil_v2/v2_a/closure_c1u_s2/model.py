from __future__ import annotations

"""Parameter-matched one-state and eight-state C1U S2 workspaces."""

from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
from typing import Any

import torch
from torch import Tensor, nn

from yggdrasil_v2.v2_a.closure_c1u import contract as c1u_contract
from yggdrasil_v2.v2_a.closure_c1u.model import (
    C1UConfig,
    PartitionedBoundary,
    PartitionedBoundaryOutput,
    QueryReadout,
    TargetOnlyTransition,
)

from . import contract


@dataclass(frozen=True)
class S2Config:
    source_width: int = c1u_contract.SOURCE_HIDDEN_WIDTH
    payload_width: int = 512
    address_width: int = c1u_contract.ADDRESS_WIDTH
    input_slots: int = c1u_contract.MAX_SLOTS
    workspace_slots: int = c1u_contract.MAX_SLOTS
    operations: int = c1u_contract.MAX_OPERATIONS
    ffn_width: int = 2_048
    answer_classes: int = c1u_contract.ANSWER_CLASSES
    route_similarity_floor: float = 0.999

    def core(self) -> C1UConfig:
        return C1UConfig(
            source_width=self.source_width,
            payload_width=self.payload_width,
            address_width=self.address_width,
            slots=self.input_slots,
            operations=self.operations,
            ffn_width=self.ffn_width,
            answer_classes=self.answer_classes,
            route_similarity_floor=self.route_similarity_floor,
        )


@dataclass
class S2BoundaryOutput:
    workspace_slots: int
    public_initial_payloads: Tensor
    public_object_addresses: Tensor
    public_object_present: Tensor
    public_source_weights: Tensor
    public_target_weights: Tensor
    public_query_weights: Tensor
    operation_states: Tensor
    operation_active: Tensor
    query_state: Tensor
    route_similarities: dict[str, Tensor]
    initial_payloads: Tensor
    object_present: Tensor
    source_weights: Tensor
    target_weights: Tensor
    query_weights: Tensor

    @classmethod
    def from_public(
        cls, public: PartitionedBoundaryOutput, *, workspace_slots: int
    ) -> "S2BoundaryOutput":
        shell = cls(
            workspace_slots=int(workspace_slots),
            public_initial_payloads=public.initial_payloads,
            public_object_addresses=public.object_addresses,
            public_object_present=public.object_present,
            public_source_weights=public.source_weights,
            public_target_weights=public.target_weights,
            public_query_weights=public.query_weights,
            operation_states=public.operation_states,
            operation_active=public.operation_active,
            query_state=public.query_state,
            route_similarities=public.route_similarities,
            initial_payloads=public.initial_payloads,
            object_present=public.object_present,
            source_weights=public.source_weights,
            target_weights=public.target_weights,
            query_weights=public.query_weights,
        )
        return shell.rebuild(public.initial_payloads)

    def rebuild(self, public_payloads: Tensor) -> "S2BoundaryOutput":
        if public_payloads.shape != self.public_initial_payloads.shape:
            raise ValueError("public payload intervention has the wrong shape")
        if self.workspace_slots == self.public_initial_payloads.shape[1]:
            return replace(
                self,
                public_initial_payloads=public_payloads,
                initial_payloads=public_payloads,
                object_present=self.public_object_present,
                source_weights=self.public_source_weights,
                target_weights=self.public_target_weights,
                query_weights=self.public_query_weights,
            )
        if self.workspace_slots != 1:
            raise ValueError("C1U S2 supports only one or eight workspace states")
        present = self.public_object_present.to(dtype=public_payloads.dtype).unsqueeze(-1)
        compressed = (public_payloads * present).sum(dim=1, keepdim=True)
        compressed = compressed / present.sum(dim=1, keepdim=True).clamp_min(1.0)
        batch = public_payloads.shape[0]
        active = self.operation_active.to(dtype=public_payloads.dtype).unsqueeze(-1)
        return replace(
            self,
            public_initial_payloads=public_payloads,
            initial_payloads=compressed,
            object_present=torch.ones(
                batch, 1, dtype=torch.bool, device=public_payloads.device
            ),
            source_weights=active,
            target_weights=active,
            query_weights=torch.ones(
                batch, 1, dtype=public_payloads.dtype, device=public_payloads.device
            ),
        )

    def permuted(self, permutation: Tensor) -> "S2BoundaryOutput":
        permutation = permutation.to(
            device=self.public_initial_payloads.device, dtype=torch.long
        )
        slots = self.public_initial_payloads.shape[1]
        if permutation.shape != (slots,) or not torch.equal(
            torch.sort(permutation).values,
            torch.arange(slots, device=permutation.device),
        ):
            raise ValueError("public object permutation is invalid")
        changed = replace(
            self,
            public_initial_payloads=self.public_initial_payloads[:, permutation],
            public_object_addresses=self.public_object_addresses[:, permutation],
            public_object_present=self.public_object_present[:, permutation],
            public_source_weights=self.public_source_weights[:, :, permutation],
            public_target_weights=self.public_target_weights[:, :, permutation],
            public_query_weights=self.public_query_weights[:, permutation],
        )
        return changed.rebuild(changed.public_initial_payloads)


class S2Boundary(nn.Module):
    def __init__(self, config: S2Config) -> None:
        super().__init__()
        self.config = config
        self.public_boundary = PartitionedBoundary(config.core())

    def forward(self, **public_fields: Tensor) -> S2BoundaryOutput:
        public = self.public_boundary(**public_fields)
        return S2BoundaryOutput.from_public(
            public, workspace_slots=self.config.workspace_slots
        )


class S2Model(nn.Module):
    def __init__(self, config: S2Config = S2Config()) -> None:
        super().__init__()
        if config.input_slots != c1u_contract.MAX_SLOTS:
            raise ValueError("S2 public input must retain eight padded object slots")
        if config.workspace_slots not in (1, config.input_slots):
            raise ValueError("S2 workspace must be learned K1 or functional K8")
        self.config = config
        core = config.core()
        self.boundary = S2Boundary(config)
        self.transition = TargetOnlyTransition(core)
        self.readout = QueryReadout(core)

    def rollout(
        self,
        boundary: S2BoundaryOutput,
        *,
        disable_recurrence: bool = False,
        initial_payloads: Tensor | None = None,
    ) -> Tensor:
        payloads = boundary.initial_payloads if initial_payloads is None else initial_payloads
        if payloads.shape != boundary.initial_payloads.shape:
            raise ValueError("workspace payload intervention has the wrong shape")
        trajectory = [payloads]
        for step in range(self.config.operations):
            if not disable_recurrence:
                payloads = self.transition(
                    payloads,
                    boundary.operation_states[:, step],
                    boundary.source_weights[:, step],
                    boundary.target_weights[:, step],
                    boundary.operation_active[:, step],
                )
            trajectory.append(payloads)
        return torch.stack(trajectory, dim=1)

    def forward_from_boundary(
        self,
        boundary: S2BoundaryOutput,
        *,
        disable_recurrence: bool = False,
        initial_payloads: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:
        trajectory = self.rollout(
            boundary,
            disable_recurrence=disable_recurrence,
            initial_payloads=initial_payloads,
        )
        selected = torch.einsum(
            "bk,bkd->bd", boundary.query_weights, trajectory[:, -1]
        )
        logits = self.readout(selected, boundary.query_state)
        result: dict[str, Any] = {
            "logits": logits,
            "initial_payloads": trajectory[:, 0],
            "final_payloads": trajectory[:, -1],
            "selected_payload": selected,
            "source_weights": boundary.source_weights,
            "target_weights": boundary.target_weights,
            "query_weights": boundary.query_weights,
            "operation_active": boundary.operation_active,
            "workspace_slots": boundary.workspace_slots,
        }
        if return_trajectory:
            result["trajectory"] = trajectory
        return result

    def forward(
        self,
        object_hidden: Tensor,
        object_mask: Tensor,
        object_addresses: Tensor,
        object_present: Tensor,
        operation_hidden: Tensor,
        operation_mask: Tensor,
        operation_source_addresses: Tensor,
        operation_target_addresses: Tensor,
        query_hidden: Tensor,
        query_mask: Tensor,
        query_address: Tensor,
        *,
        return_trajectory: bool = False,
    ) -> dict[str, Any]:
        boundary = self.boundary(
            object_hidden=object_hidden,
            object_mask=object_mask,
            object_addresses=object_addresses,
            object_present=object_present,
            operation_hidden=operation_hidden,
            operation_mask=operation_mask,
            operation_source_addresses=operation_source_addresses,
            operation_target_addresses=operation_target_addresses,
            query_hidden=query_hidden,
            query_mask=query_mask,
            query_address=query_address,
        )
        return self.forward_from_boundary(
            boundary, return_trajectory=return_trajectory
        )

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "deployment_parameters": total,
        }

    def integrity_report(self) -> dict[str, Any]:
        public_parameters = [
            name
            for name in inspect.signature(self.forward).parameters
            if name != "return_trajectory"
        ]
        forbidden = sorted(
            name
            for name, _ in self.named_parameters()
            if any(
                token in name.casefold()
                for token in (
                    "answer_target",
                    "family",
                    "factorial",
                    "support_slot",
                    "teacher",
                    "trace_target",
                )
            )
        )
        learned_gates = sorted(
            name
            for name, _ in self.transition.named_parameters()
            if "gate" in name.casefold()
        )
        exact_forward = public_parameters == list(c1u_contract.FORWARD_FIELDS)
        state_names = list(self.state_dict())
        workspace_specific_state = [
            name for name in state_names if "workspace_slots" in name.casefold()
        ]
        passed = (
            exact_forward
            and not forbidden
            and not learned_gates
            and not workspace_specific_state
        )
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.model-integrity.v1",
            "passed": passed,
            "architecture": "publicly_grounded_gate_free_matched_workspace",
            "public_forward_parameters": public_parameters,
            "exact_forward_contract": exact_forward,
            "workspace_slots": self.config.workspace_slots,
            "public_input_slots": self.config.input_slots,
            "k1_compression": "present_object_mean"
            if self.config.workspace_slots == 1
            else None,
            "target_only_transition": True,
            "learned_write_gate": False,
            "learned_write_gate_parameters": learned_gates,
            "workspace_specific_state": workspace_specific_state,
            "forbidden_parameter_names": forbidden,
            "config": asdict(self.config),
        }


def state_tensor_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest().upper()


def matched_parameter_audit(k1: S2Model, k8: S2Model) -> dict[str, Any]:
    left = {name: tuple(value.shape) for name, value in k1.state_dict().items()}
    right = {name: tuple(value.shape) for name, value in k8.state_dict().items()}
    checks = {
        "state_names_and_shapes": left == right,
        "parameter_report": k1.parameter_report() == k8.parameter_report(),
        "initial_tensor_sha256": state_tensor_sha256(k1) == state_tensor_sha256(k8),
        "k1_integrity": k1.integrity_report()["passed"],
        "k8_integrity": k8.integrity_report()["passed"],
    }
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.matched-parameter-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "k1": k1.parameter_report(),
        "k8": k8.parameter_report(),
        "k1_initial_sha256": state_tensor_sha256(k1),
        "k8_initial_sha256": state_tensor_sha256(k8),
    }


__all__ = [
    "S2Boundary",
    "S2BoundaryOutput",
    "S2Config",
    "S2Model",
    "matched_parameter_audit",
    "state_tensor_sha256",
]
