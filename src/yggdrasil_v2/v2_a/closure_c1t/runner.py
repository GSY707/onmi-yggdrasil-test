from __future__ import annotations

"""Non-consuming, stage-aware inspection for the frozen C1T chain."""

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from . import contract
from .artifacts import source_hashes, source_identity
from .model import C1TConfig, C1TModel
from .objective import compute_training_loss
from .runtime import C1TRuntime, build_independent_card_cache
from .tasks import audit_s1_bank, build_s1_bank


def _synthetic_card_encoder(text: str, *, width: int = 16) -> Mapping[str, Tensor]:
    """Deterministic inspection fixture; never a scientific source-model proxy."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tokens = 3 + len(text.encode("utf-8")) % 5
    token_ids = torch.tensor(
        [int(digest[index % len(digest)]) for index in range(tokens)], dtype=torch.long
    )
    hidden = torch.tensor(
        [
            [
                (int(digest[(token + feature) % len(digest)]) - 127.5) / 127.5
                for feature in range(width)
            ]
            for token in range(tokens)
        ],
        dtype=torch.float32,
    )
    return {
        "hidden": hidden,
        "mask": torch.ones(tokens, dtype=torch.bool),
        "token_ids": token_ids,
    }


def _fixed_path_audit(repo_root: Path) -> dict[str, Any]:
    paths = {
        "s0_preflight_root": contract.S0_PREFLIGHT_ROOT,
        "s0_preflight_lease": contract.S0_PREFLIGHT_LEASE,
        "s0_root": contract.S0_ROOT,
        "s0_lease": contract.S0_LEASE,
        "s1_preflight_root": contract.S1_PREFLIGHT_ROOT,
        "s1_preflight_lease": contract.S1_PREFLIGHT_LEASE,
        "s1_root": contract.S1_ROOT,
        "s1_lease": contract.S1_LEASE,
    }
    existence = {name: (repo_root / path).exists() for name, path in paths.items()}
    pair_checks = {
        "s0_preflight_pair": existence["s0_preflight_root"]
        == existence["s0_preflight_lease"],
        "s0_pair": existence["s0_root"] == existence["s0_lease"],
        "s1_preflight_pair": existence["s1_preflight_root"]
        == existence["s1_preflight_lease"],
        "s1_pair": existence["s1_root"] == existence["s1_lease"],
        "s1_requires_s0": (not existence["s1_preflight_root"])
        or existence["s0_root"],
        "s1_formal_requires_preflight": (not existence["s1_root"])
        or existence["s1_preflight_root"],
    }
    if existence["s1_root"] or existence["s1_lease"]:
        state = "S1_FORMAL_CONSUMED"
    elif existence["s1_preflight_root"] or existence["s1_preflight_lease"]:
        state = "S1_PREFLIGHT_CONSUMED"
    elif existence["s0_root"] and existence["s0_lease"]:
        state = "S1_CONTRACT_READY_UNCONSUMED"
    else:
        state = "S0_LAUNCH_READY_UNCONSUMED"
    return {
        "passed": all(pair_checks.values()),
        "state": state,
        "checks": pair_checks,
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "exists": existence,
        "consumed": state != "S0_LAUNCH_READY_UNCONSUMED",
    }


def _structural_smoke() -> dict[str, Any]:
    records = build_s1_bank()
    cache = build_independent_card_cache(
        records,
        _synthetic_card_encoder,
        source_identity={
            "kind": "deterministic-structural-fixture",
            "scientific_evidence": False,
            "whole_record_context": False,
        },
    )
    runtime = C1TRuntime(records, cache)
    batch = runtime.get_batch([str(row["example_id"]) for row in records])
    torch.manual_seed(contract.MODEL_SEED)
    model = C1TModel(
        C1TConfig(source_width=cache.source_width, payload_width=32, ffn_width=64)
    )
    model.train()
    boundary = model.boundary(**runtime.forward_inputs(batch))
    base = model.forward_from_boundary(boundary, return_trajectory=True)

    permutation = torch.arange(model.config.slots - 1, -1, -1)
    permuted = model.forward_from_boundary(boundary.permuted(permutation), return_trajectory=True)
    permutation_logit_delta = float(
        (base["logits"] - permuted["logits"]).abs().max().detach()
    )
    permutation_trajectory_delta = float(
        (base["trajectory"][:, :, permutation] - permuted["trajectory"])
        .abs()
        .max()
        .detach()
    )

    changed_inputs = dict(runtime.forward_inputs(batch))
    changed_inputs["object_hidden"] = changed_inputs["object_hidden"].clone()
    changed_inputs["object_hidden"][0, 0, 0, 0] += 0.25
    changed_boundary = model.boundary(**changed_inputs)
    local_delta = (
        changed_boundary.initial_payloads[0] - boundary.initial_payloads[0]
    ).abs().amax(dim=-1)
    card_isolation = bool(local_delta[0] > 0) and float(
        local_delta[1:].max().detach()
    ) == 0.0

    operation_changed = dict(runtime.forward_inputs(batch))
    operation_changed["operation_hidden"] = operation_changed["operation_hidden"].clone()
    operation_changed["operation_hidden"][0, 0, 0, 0] += 0.25
    operation_boundary = model.boundary(**operation_changed)
    no_core = model.forward_from_boundary(boundary, disable_recurrence=True)
    operation_no_core = model.forward_from_boundary(
        operation_boundary, disable_recurrence=True
    )
    no_core_operation_delta = float(
        (no_core["logits"] - operation_no_core["logits"]).abs().max().detach()
    )

    first_target = int(boundary.target_weights[0, 0].argmax())
    non_targets = [index for index in range(model.config.slots) if index != first_target]
    target_only_delta = float(
        (
            base["trajectory"][0, 1, non_targets]
            - base["trajectory"][0, 0, non_targets]
        )
        .abs()
        .max()
        .detach()
    )

    loss, components, _ = compute_training_loss(model, batch)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    checks = {
        "task_bank": audit_s1_bank(records)["passed"],
        "cache_independent_calls": cache.manifest()["one_card_per_encoder_call"] is True,
        "model_integrity": model.integrity_report()["passed"],
        "permutation_logits": permutation_logit_delta
        <= float(contract.S1_GATES["permutation_logit_tolerance"]),
        "permutation_trajectory": permutation_trajectory_delta
        <= float(contract.S1_GATES["permutation_logit_tolerance"]),
        "card_isolation": card_isolation,
        "operation_absent_from_no_core": no_core_operation_delta == 0.0,
        "target_only_write": target_only_delta == 0.0,
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": gradients_finite,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fixture_boundary": "deterministic structural fixture; not source-model evidence",
        "task_audit": audit_s1_bank(records),
        "cache_manifest": cache.manifest(),
        "model_integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(),
        "loss_components": components,
        "permutation_logit_max_abs_delta": permutation_logit_delta,
        "permutation_trajectory_max_abs_delta": permutation_trajectory_delta,
        "nonlocal_card_delta": float(local_delta[1:].max().detach()),
        "operation_to_no_core_logit_delta": no_core_operation_delta,
        "non_target_step_delta": target_only_delta,
        "optimizer_steps": 0,
        "model_writes": 0,
    }


def inspect_successor(repo_root: Path) -> dict[str, Any]:
    """Read the implementation and run disposable in-memory structural checks."""

    repo_root = Path(repo_root).resolve()
    hashes = source_hashes(repo_root)
    fixed_paths = _fixed_path_audit(repo_root)
    structural = _structural_smoke()
    checks = {
        "source_closure_complete": len(hashes) > 0,
        "fixed_path_state_consistent": fixed_paths["passed"],
        "structural_smoke": structural["passed"],
        "s1_stage_limited_authorization": contract.manifest()[
            "training_authorization_is_stage_limited"
        ] is True,
        "s1_fixed_endpoint": contract.manifest()["s1_fixed_endpoint"]
        == contract.S1_FIXED_ENDPOINT,
    }
    state = str(fixed_paths["state"])
    remaining = {
        "S0_LAUNCH_READY_UNCONSUMED": [
            "consume the registered S0 preflight before S0"
        ],
        "S1_CONTRACT_READY_UNCONSUMED": [
            "consume the registered S1 preflight before the single-use S1"
        ],
        "S1_PREFLIGHT_CONSUMED": [
            "audit the consumed S1 preflight; launch S1 only if it sealed PASS"
        ],
        "S1_FORMAL_CONSUMED": [
            "audit the consumed S1 root and obey its terminal authorization"
        ],
    }[state]
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.implementation-inspection.v1",
        "identity": contract.IDENTITY,
        "status": state if all(checks.values()) else "NOT_READY",
        "passed": all(checks.values()),
        "authorizes": "nothing",
        "evidence_level": "launch-readiness-with-synthetic-structural-smoke",
        "s0_qualified": fixed_paths["exists"]["s0_root"],
        "checks": checks,
        "contract": contract.manifest(),
        "source_files": len(hashes),
        "source_identity": source_identity(hashes),
        "source_hashes": hashes,
        "fixed_paths": fixed_paths,
        "structural": structural,
        "training_started": fixed_paths["exists"]["s1_root"],
        "preflight_consumed": fixed_paths["exists"]["s1_preflight_root"],
        "optimizer_steps": 0,
        "model_writes": 0,
        "inspection_mutations": 0,
        "remaining": remaining,
    }


__all__ = ["inspect_successor"]
