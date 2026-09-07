from __future__ import annotations

"""Single-use C1U S0 preflight and zero-training qualification."""

from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import traceback
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.nn import functional as F

from . import contract
from .artifacts import (
    audit_evidence_seal,
    claim_single_use,
    read_json,
    sha256_file,
    snapshot_sources,
    source_hashes,
    source_identity,
    tree_hashes,
    write_evidence_seal,
    write_json,
)
from .cache import audit_persisted_cache, persist_independent_cache
from .model import C1UConfig, C1UModel
from .objective import compute_training_loss, paired_counterfactual_payloads
from .runtime import (
    C1URuntime,
    IndependentCardCache,
    audit_independent_card_cache,
    build_independent_card_cache,
)
from .source import QwenCardEncoder, audit_device, audit_source_assets
from .tasks import (
    audit_bridge_fault_registry,
    audit_factorial_group,
    audit_s1_bank,
    build_s1_bank,
)


AssetAudit = Callable[[], Mapping[str, Any]]
DeviceAudit = Callable[[str | torch.device], Mapping[str, Any]]
EncoderFactory = Callable[..., Any]
PredecessorAudit = Callable[[Path], Mapping[str, Any]]


def _fixed_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "s0_preflight_root": repo_root / contract.S0_PREFLIGHT_ROOT,
        "s0_preflight_lease": repo_root / contract.S0_PREFLIGHT_LEASE,
        "s0_root": repo_root / contract.S0_ROOT,
        "s0_lease": repo_root / contract.S0_LEASE,
    }


def audit_unconsumed_paths(repo_root: Path, *, stage: str) -> dict[str, Any]:
    paths = _fixed_paths(Path(repo_root).resolve())
    if stage == "s0-preflight":
        required_absent = tuple(paths)
    elif stage == "s0":
        required_absent = ("s0_root", "s0_lease")
    else:
        raise ValueError(f"unknown path-audit stage: {stage}")
    existence = {name: path.exists() for name, path in paths.items()}
    checks = {name: not existence[name] for name in required_absent}
    unexpected_successors = sorted(
        path.as_posix()
        for parent in (repo_root / "tmp", repo_root / "artifacts/v2-a")
        if parent.is_dir()
        for path in parent.iterdir()
        if "c1u" in path.name.casefold() and "s1" in path.name.casefold()
    )
    checks["no_unfrozen_s1_paths"] = not unexpected_successors
    return {
        "passed": all(checks.values()),
        "stage": stage,
        "checks": checks,
        "exists": existence,
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "unexpected_successor_paths": unexpected_successors,
    }


def _running_c1t_s1_processes() -> dict[str, Any]:
    """Use the Windows process table to reject a concurrent old formal run."""

    command = (
        "$rows = @(Get-CimInstance Win32_Process | Where-Object { "
        "$_.ProcessId -ne $PID -and $_.CommandLine -match "
        "'v2_a_closure_c1t(\\.py)?\\s+run-s1' }); "
        "$rows | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        output = completed.stdout.strip()
        parsed = json.loads(output) if output else []
        rows = parsed if isinstance(parsed, list) else [parsed]
        return {"passed": not rows, "count": len(rows), "rows": rows}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        return {
            "passed": False,
            "count": None,
            "rows": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }


def audit_c1t_predecessor(repo_root: Path) -> dict[str, Any]:
    """Replay the immutable C1T S1 FAIL without importing its old code."""

    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.C1T_S1_ROOT
    result_path = root / "result.json"
    seal_path = root / "evidence-seal.json"
    endpoint_path = root / "fixed_4000.pt"
    process_audit = _running_c1t_s1_processes()
    try:
        result = read_json(result_path)
        seal = read_json(seal_path)
        expected_tree = dict(seal["files"])
        actual_tree = tree_hashes(root, exclude=("evidence-seal.json",))
        observed_roots = sorted(
            path.name
            for path in (repo_root / "artifacts/v2-a").glob("closure-c1t-cpw-*")
            if path.is_dir()
        )
        allowed_roots = {
            contract.C1T_S0_ROOT.name,
            contract.C1T_S1_ROOT.name,
        }
        checks = {
            "root": root.is_dir(),
            "lease": (repo_root / contract.C1T_S1_LEASE).is_file(),
            "identity": result.get("identity") == contract.C1T_S1_IDENTITY,
            "terminal_status": result.get("status") == contract.C1T_S1_TERMINAL_STATUS,
            "failed": result.get("passed") is False,
            "authorizes_nothing": result.get("authorizes") == "nothing",
            "source_identity": result.get("source_identity")
            == contract.C1T_S1_SOURCE_IDENTITY,
            "result_sha256": sha256_file(result_path) == contract.C1T_S1_RESULT_SHA256,
            "seal_sha256": sha256_file(seal_path)
            == contract.C1T_S1_EVIDENCE_SEAL_SHA256,
            "endpoint_sha256": sha256_file(endpoint_path)
            == contract.C1T_S1_ENDPOINT_SHA256,
            "result_endpoint_sha256": result.get("endpoint", {}).get("sha256")
            == contract.C1T_S1_ENDPOINT_SHA256,
            "seal_schema": seal.get("schema_version") == contract.C1T_S1_SEAL_SCHEMA,
            "seal_identity": seal.get("identity") == contract.C1T_S1_IDENTITY,
            "seal_stage": seal.get("stage") == "s1",
            "sealed_tree": expected_tree == actual_tree,
            "no_c1t_successor_root": set(observed_roots) == allowed_roots,
            "no_running_c1t_s1": process_audit["passed"],
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "root": contract.C1T_S1_ROOT.as_posix(),
            "observed_roots": observed_roots,
            "process_audit": process_audit,
            "result_status": result.get("status"),
            "result_sha256": sha256_file(result_path),
            "seal_sha256": sha256_file(seal_path),
            "endpoint_sha256": sha256_file(endpoint_path),
            "source_identity": result.get("source_identity"),
            "sealed_entries": len(expected_tree),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _preflight_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups = {
        f"c1u-{family.casefold()}-g{index:02d}"
        for family, index in contract.S0_PREFLIGHT_GROUPS
    }
    selected = [row for row in records if str(row["factorial_group_id"]) in groups]
    if len(selected) != contract.S0_PREFLIGHT_RECORDS:
        raise RuntimeError("registered S0 preflight record cardinality drifted")
    return selected


def _audit_record_subset(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row["factorial_group_id"]), []).append(row)
    audits = {name: audit_factorial_group(rows) for name, rows in sorted(grouped.items())}
    families = {str(row["family"]) for row in records}
    fault_registry = audit_bridge_fault_registry(records)
    checks = {
        "nonempty": bool(records),
        "complete_groups": bool(audits) and all(row["passed"] for row in audits.values()),
        "unique_examples": len({str(row["example_id"]) for row in records}) == len(records),
        "both_families": families == set(contract.FAMILIES),
        "bridge_fault_registry": fault_registry["passed"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "records": len(records),
        "groups": len(grouped),
        "group_audits": audits,
        "bridge_fault_registry": fault_registry,
    }


def _source_identity_payload(asset_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": contract.SOURCE_MODEL_ID,
        "revision": contract.SOURCE_MODEL_REVISION,
        "model_class": contract.SOURCE_MODEL_CLASS,
        "tokenizer_class": contract.SOURCE_TOKENIZER_CLASS,
        "hidden_width": contract.SOURCE_HIDDEN_WIDTH,
        "dtype": contract.SOURCE_DTYPE,
        "local_files_only": True,
        "one_card_per_forward_call": True,
        "whole_record_hidden_reuse": False,
        "asset_sha256": {
            name: str(row.get("sha256", ""))
            for name, row in sorted(asset_report.get("rows", {}).items())
        },
    }


def _encode_records(
    records: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device,
    asset_report: Mapping[str, Any],
    encoder_factory: EncoderFactory,
) -> tuple[IndependentCardCache, dict[str, Any]]:
    encoder = encoder_factory(device=device)
    try:
        cache = build_independent_card_cache(
            records,
            encoder,
            source_identity=_source_identity_payload(asset_report),
        )
        report = dict(encoder.report())
    finally:
        encoder.release()
    return cache, report


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _model_config(source_width: int, override: C1UConfig | None) -> C1UConfig:
    if override is None:
        return C1UConfig(source_width=source_width)
    if override.source_width != source_width:
        raise ValueError("test/registered model config source width mismatch")
    return override


def _no_core_group_invariance(
    model: C1UModel,
    runtime: C1URuntime,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    families: dict[str, str] = {}
    for row in records:
        name = str(row["factorial_group_id"])
        groups.setdefault(name, []).append(str(row["example_id"]))
        families[name] = str(row["family"])
    rows = {}
    maximum = 0.0
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        for name, ids in sorted(groups.items()):
            batch = _move_batch(runtime.get_batch(ids), device)
            boundary = model.boundary(**runtime.forward_inputs(batch))
            logits = model.forward_from_boundary(boundary, disable_recurrence=True)["logits"].float()
            delta = float((logits - logits[:1]).abs().max().cpu())
            maximum = max(maximum, delta)
            rows[name] = {"family": families[name], "records": len(ids), "max_abs_delta": delta}
    return {
        "passed": maximum
        <= float(contract.STRUCTURAL_TOLERANCES["no_core_group_logit_tolerance"]),
        "threshold": float(
            contract.STRUCTURAL_TOLERANCES["no_core_group_logit_tolerance"]
        ),
        "maximum_abs_delta": maximum,
        "groups": rows,
    }


def _gradient_report(gradients: Sequence[torch.Tensor | None]) -> dict[str, Any]:
    present = [gradient for gradient in gradients if gradient is not None]
    finite = bool(present) and all(bool(torch.isfinite(gradient).all()) for gradient in present)
    norm = float(
        sum(gradient.detach().float().square().sum() for gradient in present).sqrt().cpu()
    ) if present else 0.0
    return {
        "passed": len(present) == len(gradients) and finite and norm > 0.0,
        "expected": len(gradients),
        "present": len(present),
        "finite": finite,
        "l2_norm": norm,
    }


def _one_intervention_connectivity(
    model: C1UModel,
    boundary: Any,
    batch: Mapping[str, Any],
    *,
    support_index: int,
    device: torch.device,
    fault: str | None = None,
) -> dict[str, Any]:
    """Trace a counterpart objective through operation 2 and the readout."""

    model.zero_grad(set_to_none=True)
    initial_seed = boundary.initial_payloads.detach().clone().requires_grad_(True)
    replacements = paired_counterfactual_payloads(
        initial_seed,
        batch["support_slots"].long(),
        batch["counterfactual_indices"].long(),
    )
    payloads = replacements[:, support_index]
    operation2_source: torch.Tensor | None = None
    operation2_proposal: torch.Tensor | None = None
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for step in range(model.config.operations):
            if step != 1:
                payloads = model.transition(
                    payloads,
                    boundary.operation_states[:, step],
                    boundary.source_weights[:, step],
                    boundary.target_weights[:, step],
                    boundary.operation_active[:, step],
                )
                continue
            operation2_source = torch.einsum(
                "bk,bkd->bd", boundary.source_weights[:, step], payloads
            )
            target = torch.einsum(
                "bk,bkd->bd", boundary.target_weights[:, step], payloads
            )
            source_for_operator = (
                operation2_source.detach() if fault == "detach_source" else operation2_source
            )
            operation2_proposal = model.transition.operator(
                torch.cat(
                    (source_for_operator, target, boundary.operation_states[:, step]), dim=-1
                )
            )
            proposal_for_write = (
                operation2_proposal.detach()
                if fault == "detach_proposal"
                else operation2_proposal
            )
            target_delta = proposal_for_write - target
            candidate = payloads + boundary.target_weights[:, step].unsqueeze(
                -1
            ) * target_delta.unsqueeze(1)
            active = boundary.operation_active[:, step].to(dtype=payloads.dtype).view(
                -1, 1, 1
            )
            payloads = active * candidate + (1.0 - active) * payloads
        logits, _ = model.logits_from_payloads(payloads, boundary)
        counterpart_rows = batch["counterfactual_indices"][:, support_index].long()
        expected = batch["answers"].long()[counterpart_rows]
        objective = F.cross_entropy(logits.float(), expected)

    if operation2_source is None or operation2_proposal is None:
        raise RuntimeError("registered operation 2 was not traced")
    transition_parameters = [parameter for parameter in model.transition.parameters()]
    readout_parameters = [parameter for parameter in model.readout.parameters()]
    gradients = torch.autograd.grad(
        objective,
        [operation2_source, operation2_proposal, *transition_parameters, *readout_parameters],
        allow_unused=True,
    )
    source_report = _gradient_report(gradients[:1])
    proposal_report = _gradient_report(gradients[1:2])
    transition_end = 2 + len(transition_parameters)
    transition_report = _gradient_report(gradients[2:transition_end])
    readout_report = _gradient_report(gradients[transition_end:])
    checks = {
        "objective_finite": bool(torch.isfinite(objective)),
        "operation2_source_gradient": source_report["passed"],
        "operation2_proposal_gradient": proposal_report["passed"],
        "shared_transition_gradients": transition_report["passed"],
        "readout_gradients": readout_report["passed"],
    }
    return {
        "passed": all(checks.values()),
        "support_index": support_index,
        "fault": fault,
        "checks": checks,
        "objective": float(objective.detach().cpu()),
        "operation2_source": source_report,
        "operation2_proposal": proposal_report,
        "shared_transition": transition_report,
        "readout": readout_report,
    }


def _operation2_gradient_connectivity(
    model: C1UModel,
    boundary: Any,
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    supports = [
        _one_intervention_connectivity(
            model,
            boundary,
            batch,
            support_index=index,
            device=device,
        )
        for index in range(2)
    ]
    fault_rows = {
        "detach_source": _one_intervention_connectivity(
            model,
            boundary,
            batch,
            support_index=1,
            device=device,
            fault="detach_source",
        ),
        "detach_proposal": _one_intervention_connectivity(
            model,
            boundary,
            batch,
            support_index=1,
            device=device,
            fault="detach_proposal",
        ),
    }
    killed = {name: not row["passed"] for name, row in fault_rows.items()}
    return {
        "passed": all(row["passed"] for row in supports) and all(killed.values()),
        "supports": supports,
        "fault_registry": {
            "passed": all(killed.values()),
            "registered": len(killed),
            "killed": sum(killed.values()),
            "faults": killed,
            "measurements": fault_rows,
        },
    }


def structural_measurements(
    records: Sequence[Mapping[str, Any]],
    cache: IndependentCardCache,
    *,
    device: str | torch.device,
    config_override: C1UConfig | None = None,
) -> dict[str, Any]:
    """Run S0 structure and BF16 backward checks without an optimizer."""

    selected_device = torch.device(device)
    config = _model_config(cache.source_width, config_override)
    runtime = C1URuntime(records, cache, address_width=config.address_width)
    torch.manual_seed(contract.MODEL_SEED)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(contract.MODEL_SEED)
    model = C1UModel(config).to(selected_device)
    probe_rows = _preflight_records(records) if len(records) == contract.S0_RECORDS else list(records)
    batch = _move_batch(runtime.get_batch([str(row["example_id"]) for row in probe_rows]), selected_device)
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=selected_device.type,
        dtype=torch.bfloat16,
        enabled=selected_device.type == "cuda",
    ):
        boundary = model.boundary(**runtime.forward_inputs(batch))
        base = model.forward_from_boundary(boundary, return_trajectory=True)
        permutation = torch.arange(config.slots - 1, -1, -1, device=selected_device)
        permuted = model.forward_from_boundary(
            boundary.permuted(permutation), return_trajectory=True
        )
        permutation_logit_delta = float(
            (base["logits"] - permuted["logits"]).abs().max().float().cpu()
        )
        permutation_trajectory_delta = float(
            (
                base["trajectory"][:, :, permutation]
                - permuted["trajectory"]
            ).abs().max().float().cpu()
        )

        changed_inputs = dict(runtime.forward_inputs(batch))
        changed_inputs["object_hidden"] = changed_inputs["object_hidden"].clone()
        changed_inputs["object_hidden"][0, 0, 0, 0] += 0.25
        changed_boundary = model.boundary(**changed_inputs)
        local_delta = (
            changed_boundary.initial_payloads[0] - boundary.initial_payloads[0]
        ).abs().amax(dim=-1).float()
        card_isolation_nonlocal = float(local_delta[1:].max().cpu())
        card_isolation_local = float(local_delta[0].cpu())

        operation_changed = dict(runtime.forward_inputs(batch))
        operation_changed["operation_hidden"] = operation_changed["operation_hidden"].clone()
        operation_changed["operation_hidden"][0, 0, 0, 0] += 0.25
        operation_boundary = model.boundary(**operation_changed)
        no_core = model.forward_from_boundary(boundary, disable_recurrence=True)
        operation_no_core = model.forward_from_boundary(
            operation_boundary, disable_recurrence=True
        )
        operation_no_core_delta = float(
            (no_core["logits"] - operation_no_core["logits"]).abs().max().float().cpu()
        )

        target_only_delta = 0.0
        inactive_delta = 0.0
        for row_index in range(base["trajectory"].shape[0]):
            for step in range(config.operations):
                before = base["trajectory"][row_index, step]
                after = base["trajectory"][row_index, step + 1]
                if bool(boundary.operation_active[row_index, step]):
                    target = int(boundary.target_weights[row_index, step].argmax())
                    keep = [index for index in range(config.slots) if index != target]
                    target_only_delta = max(
                        target_only_delta,
                        float((after[keep] - before[keep]).abs().max().float().cpu()),
                    )
                else:
                    inactive_delta = max(
                        inactive_delta,
                        float((after - before).abs().max().float().cpu()),
                    )

    no_core_invariance = _no_core_group_invariance(
        model, runtime, records, selected_device
    )
    gradient_connectivity = _operation2_gradient_connectivity(
        model, boundary, batch, selected_device
    )
    model.train()
    model.zero_grad(set_to_none=True)
    parameter_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    with torch.autocast(
        device_type=selected_device.type,
        dtype=torch.bfloat16,
        enabled=selected_device.type == "cuda",
    ):
        loss, components, _ = compute_training_loss(model, batch)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    parameter_unchanged_by_backward = all(
        torch.equal(parameter_before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    checks = {
        "model_integrity": model.integrity_report()["passed"],
        "source_only_forward_fields": set(runtime.forward_inputs(batch))
        == set(contract.FORWARD_FIELDS),
        "no_forbidden_forward_fields": set(runtime.forward_inputs(batch)).isdisjoint(
            contract.FORBIDDEN_FORWARD_FIELDS
        ),
        "complete_counterfactual_groups": bool((batch["counterfactual_indices"] >= 0).all()),
        "permutation_logits": permutation_logit_delta
        <= float(contract.STRUCTURAL_TOLERANCES["permutation_logit_tolerance"]),
        "permutation_trajectory": permutation_trajectory_delta
        <= float(contract.STRUCTURAL_TOLERANCES["permutation_logit_tolerance"]),
        "card_isolation": card_isolation_local > 0.0 and card_isolation_nonlocal == 0.0,
        "operation_absent_from_no_core": operation_no_core_delta == 0.0,
        "target_only_write": target_only_delta == 0.0,
        "inactive_identity": inactive_delta == 0.0,
        "no_core_group_invariance": no_core_invariance["passed"],
        "operation2_gradient_connectivity": gradient_connectivity["passed"],
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": gradients_finite,
        "no_optimizer": parameter_unchanged_by_backward,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "config": asdict(config),
        "model_integrity": model.integrity_report(),
        "parameter_report": model.parameter_report(),
        "loss_components": components,
        "permutation_logit_max_abs_delta": permutation_logit_delta,
        "permutation_trajectory_max_abs_delta": permutation_trajectory_delta,
        "card_isolation_local_delta": card_isolation_local,
        "card_isolation_nonlocal_delta": card_isolation_nonlocal,
        "operation_to_no_core_logit_delta": operation_no_core_delta,
        "non_target_step_delta": target_only_delta,
        "inactive_step_delta": inactive_delta,
        "no_core_group_invariance": no_core_invariance,
        "operation2_gradient_connectivity": gradient_connectivity,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(selected_device))
        if selected_device.type == "cuda"
        else 0,
        "optimizer_steps": 0,
        "model_writes": 0,
    }
    del model, batch, runtime, parameter_before
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def audit_s0_preflight(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S0_PREFLIGHT_ROOT
    try:
        result = read_json(root / "result.json")
        seal = audit_evidence_seal(
            root,
            expected_identity=contract.S0_PREFLIGHT_IDENTITY,
            expected_stage="s0-preflight",
        )
        current_hashes = source_hashes(repo_root)
        recorded_hashes = dict(result.get("source_hashes", {}))
        snapshot_hashes = {
            relative: sha256_file(root / "source_snapshot" / relative)
            for relative in recorded_hashes
        }
        gates = dict(result.get("gates", {}))
        checks = {
            "lease_exists": (repo_root / contract.S0_PREFLIGHT_LEASE).is_file(),
            "identity": result.get("identity") == contract.S0_PREFLIGHT_IDENTITY,
            "status": result.get("status") == "PASS_V2_A_C1U_S0_PREFLIGHT",
            "passed": result.get("passed") is True,
            "stage": result.get("stage") == "s0-preflight",
            "authorization": result.get("authorizes") == contract.S0_AUTHORIZED_SCOPE,
            "gates": set(gates) == set(contract.S0_PREFLIGHT_GATES)
            and all(gates.values()),
            "records": result.get("records") == contract.S0_PREFLIGHT_RECORDS,
            "cards": result.get("cards") == contract.S0_PREFLIGHT_CARDS,
            "predecessor": result.get("predecessor", {}).get("passed") is True,
            "optimizer_steps": result.get("optimizer_steps") == 0,
            "model_writes": result.get("model_writes") == 0,
            "training_not_started": result.get("training_started") is False,
            "source_hashes": result.get("source_hashes") == current_hashes,
            "source_identity": result.get("source_identity") == source_identity(current_hashes),
            "source_snapshot": snapshot_hashes == recorded_hashes,
            "seal": seal.get("passed") is True,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "result": result,
            "seal": seal,
            "result_sha256": sha256_file(root / "result.json"),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def audit_s0_root(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S0_ROOT
    try:
        result = read_json(root / "result.json")
        seal = audit_evidence_seal(
            root, expected_identity=contract.S0_IDENTITY, expected_stage="s0"
        )
        current_hashes = source_hashes(repo_root)
        recorded_hashes = dict(result.get("source_hashes", {}))
        snapshot_hashes = {
            relative: sha256_file(root / "source_snapshot" / relative)
            for relative in recorded_hashes
        }
        gates = dict(result.get("gates", {}))
        checks = {
            "lease_exists": (repo_root / contract.S0_LEASE).is_file(),
            "identity": result.get("identity") == contract.S0_IDENTITY,
            "status": result.get("status") == "PASS_V2_A_C1U_S0_QUALIFICATION",
            "passed": result.get("passed") is True,
            "stage": result.get("stage") == "s0",
            "authorization": result.get("authorizes") == contract.S0_PASS_AUTHORIZATION,
            "gates": set(gates) == set(contract.S0_GATES) and all(gates.values()),
            "records": result.get("records") == contract.S0_RECORDS,
            "cards": result.get("cards") == contract.S0_CARDS,
            "predecessor": result.get("predecessor", {}).get("passed") is True,
            "cache_target_free": result.get("cache_manifest", {}).get("contains_targets")
            is False,
            "optimizer_steps": result.get("optimizer_steps") == 0,
            "model_writes": result.get("model_writes") == 0,
            "training_not_started": result.get("training_started") is False,
            "source_hashes": result.get("source_hashes") == current_hashes,
            "source_identity": result.get("source_identity") == source_identity(current_hashes),
            "source_snapshot": snapshot_hashes == recorded_hashes,
            "seal": seal.get("passed") is True,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "result": result,
            "seal": seal,
            "result_sha256": sha256_file(root / "result.json"),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _crash_result(
    root: Path, *, identity: str, stage: str, exc: Exception, hashes: Mapping[str, str] | None
) -> dict[str, Any]:
    result = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.{stage}-result.v1",
        "identity": identity,
        "stage": stage,
        "status": f"CRASH_V2_A_C1U_{stage.upper().replace('-', '_')}",
        "passed": False,
        "authorizes": "nothing",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "source_hashes": dict(hashes or {}),
        "source_identity": source_identity(hashes) if hashes else None,
        "optimizer_steps": 0,
        "model_writes": 0,
    }
    write_json(root / "result.json", result)
    write_evidence_seal(root, identity=identity, stage=stage)
    return result


def _maybe_fault(fault_at: str | None, point: str) -> None:
    if fault_at == point:
        raise RuntimeError(f"injected C1U S0 fault at {point}")


def run_s0_preflight(
    repo_root: Path,
    *,
    device: str | torch.device = "cuda",
    asset_audit_fn: AssetAudit = audit_source_assets,
    device_audit_fn: DeviceAudit = audit_device,
    encoder_factory: EncoderFactory = QwenCardEncoder,
    predecessor_audit_fn: PredecessorAudit = audit_c1t_predecessor,
    config_override: C1UConfig | None = None,
    fault_at: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S0_PREFLIGHT_ROOT
    lease = repo_root / contract.S0_PREFLIGHT_LEASE
    hashes = source_hashes(repo_root)
    paths = audit_unconsumed_paths(repo_root, stage="s0-preflight")
    predecessor = dict(predecessor_audit_fn(repo_root))
    assets = dict(asset_audit_fn())
    hardware = dict(device_audit_fn(device))
    user_authorized = contract.USER_AUTHORIZATION == "没问题，按这个顺序做"
    if (
        not paths["passed"]
        or not predecessor.get("passed")
        or not assets.get("passed")
        or not hardware.get("passed")
        or not user_authorized
    ):
        raise RuntimeError("S0 preflight pre-claim audit failed; fixed root was not consumed")
    claim_single_use(
        identity=contract.S0_PREFLIGHT_IDENTITY,
        stage="s0-preflight",
        output_root=root,
        lease_path=lease,
    )
    try:
        _maybe_fault(fault_at, "after_claim")
        snapshot_sources(repo_root, root)
        write_json(root / "contract-manifest.json", contract.manifest())
        write_json(
            root / "preclaim-audit.json",
            {
                "paths": paths,
                "predecessor": predecessor,
                "assets": assets,
                "device": hardware,
                "user_authorized": user_authorized,
            },
        )
        records = _preflight_records(build_s1_bank())
        task_audit = _audit_record_subset(records)
        write_json(root / "task-bank.json", {"records": records, "audit": task_audit})
        cache, source_report = _encode_records(
            records,
            device=device,
            asset_report=assets,
            encoder_factory=encoder_factory,
        )
        _maybe_fault(fault_at, "after_cache")
        cache_audit = audit_independent_card_cache(records, cache)
        write_json(
            root / "card-ledger.json",
            {
                "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-card-ledger.v1",
                "identity": contract.S0_PREFLIGHT_IDENTITY,
                "rows": list(cache.ledger),
            },
        )
        structural = structural_measurements(
            records, cache, device=device, config_override=config_override
        )
        write_json(root / "source-encoder-report.json", source_report)
        write_json(root / "cache-audit.json", cache_audit)
        write_json(root / "structural-measurements.json", structural)
        post_hashes = source_hashes(repo_root)
        structure_checks = structural["checks"]
        gates = {
            contract.S0_PREFLIGHT_GATES[0]: predecessor["passed"]
            and paths["passed"]
            and user_authorized,
            contract.S0_PREFLIGHT_GATES[1]: bool(hashes)
            and assets["passed"]
            and hardware["passed"],
            contract.S0_PREFLIGHT_GATES[2]: task_audit["passed"],
            contract.S0_PREFLIGHT_GATES[3]: cache_audit["passed"]
            and source_report.get("calls") == contract.S0_PREFLIGHT_CARDS,
            contract.S0_PREFLIGHT_GATES[4]: all(
                structure_checks[name]
                for name in (
                    "model_integrity",
                    "target_only_write",
                    "inactive_identity",
                    "permutation_logits",
                    "permutation_trajectory",
                    "operation_absent_from_no_core",
                    "no_core_group_invariance",
                    "operation2_gradient_connectivity",
                )
            ),
            contract.S0_PREFLIGHT_GATES[5]: structure_checks["loss_finite"]
            and structure_checks["gradients_finite"]
            and structure_checks["no_optimizer"]
            and structural["optimizer_steps"] == 0
            and structural["model_writes"] == 0
            and source_report.get("trainable_source_parameters") == 0
            and post_hashes == hashes,
        }
        passed = all(gates.values())
        _maybe_fault(fault_at, "before_result")
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s0-preflight-result.v1",
            "identity": contract.S0_PREFLIGHT_IDENTITY,
            "stage": "s0-preflight",
            "status": "PASS_V2_A_C1U_S0_PREFLIGHT" if passed else "FAIL_V2_A_C1U_S0_PREFLIGHT",
            "passed": passed,
            "authorizes": contract.S0_AUTHORIZED_SCOPE if passed else "nothing",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
            "gates": gates,
            "records": len(records),
            "cards": cache_audit["card_count"],
            "source_identity": source_identity(hashes),
            "source_hashes": hashes,
            "source_encoder_report": source_report,
            "predecessor": predecessor,
            "device": hardware,
            "optimizer_steps": 0,
            "model_writes": 0,
            "training_started": False,
            "qualification_requires_seal_replay": True,
        }
        write_json(root / "result.json", result)
        write_evidence_seal(root, identity=contract.S0_PREFLIGHT_IDENTITY, stage="s0-preflight")
        return {**result, "seal_replay": audit_evidence_seal(root, expected_identity=contract.S0_PREFLIGHT_IDENTITY, expected_stage="s0-preflight")}
    except Exception as exc:
        return _crash_result(
            root,
            identity=contract.S0_PREFLIGHT_IDENTITY,
            stage="s0-preflight",
            exc=exc,
            hashes=hashes,
        )


def run_s0(
    repo_root: Path,
    *,
    device: str | torch.device = "cuda",
    asset_audit_fn: AssetAudit = audit_source_assets,
    device_audit_fn: DeviceAudit = audit_device,
    encoder_factory: EncoderFactory = QwenCardEncoder,
    predecessor_audit_fn: PredecessorAudit = audit_c1t_predecessor,
    config_override: C1UConfig | None = None,
    fault_at: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    preflight = audit_s0_preflight(repo_root)
    paths = audit_unconsumed_paths(repo_root, stage="s0")
    hashes = source_hashes(repo_root)
    predecessor = dict(predecessor_audit_fn(repo_root))
    assets = dict(asset_audit_fn())
    hardware = dict(device_audit_fn(device))
    if (
        not preflight.get("passed")
        or not paths["passed"]
        or not predecessor.get("passed")
        or not assets.get("passed")
        or not hardware.get("passed")
    ):
        raise RuntimeError("S0 pre-claim audit failed; fixed S0 root was not consumed")
    root = repo_root / contract.S0_ROOT
    lease = repo_root / contract.S0_LEASE
    claim_single_use(
        identity=contract.S0_IDENTITY,
        stage="s0",
        output_root=root,
        lease_path=lease,
    )
    try:
        _maybe_fault(fault_at, "after_claim")
        snapshot_sources(repo_root, root)
        write_json(root / "contract-manifest.json", contract.manifest())
        write_json(
            root / "preclaim-audit.json",
            {
                "preflight": preflight,
                "paths": paths,
                "predecessor": predecessor,
                "assets": assets,
                "device": hardware,
            },
        )
        records = build_s1_bank()
        task_audit = audit_s1_bank(records)
        write_json(root / "task-bank.json", {"records": records, "audit": task_audit})
        cache, source_report = _encode_records(
            records,
            device=device,
            asset_report=assets,
            encoder_factory=encoder_factory,
        )
        _maybe_fault(fault_at, "after_cache")
        cache_audit = audit_independent_card_cache(records, cache)
        persistent_manifest = persist_independent_cache(root / "card-cache", records, cache)
        persistent_audit = audit_persisted_cache(root / "card-cache", records)
        structural = structural_measurements(
            records, cache, device=device, config_override=config_override
        )
        write_json(root / "source-encoder-report.json", source_report)
        write_json(root / "cache-audit.json", cache_audit)
        write_json(root / "persistent-cache-audit.json", persistent_audit)
        write_json(root / "structural-measurements.json", structural)
        post_hashes = source_hashes(repo_root)
        structure_checks = structural["checks"]
        task_checks = task_audit["checks"]
        gates = {
            contract.S0_GATES[0]: preflight["passed"]
            and predecessor["passed"]
            and assets["passed"]
            and hardware["passed"],
            contract.S0_GATES[1]: all(
                task_checks[name]
                for name in (
                    "record_count",
                    "group_count",
                    "family_counts",
                    "all_groups",
                    "unique_example_ids",
                    "public_only_replay",
                )
            ),
            contract.S0_GATES[2]: task_audit["bridge_fault_registry"]["passed"],
            contract.S0_GATES[3]: cache_audit["passed"]
            and persistent_audit["passed"]
            and source_report.get("calls") == contract.S0_CARDS
            and persistent_manifest.get("contains_targets") is False,
            contract.S0_GATES[4]: structure_checks["source_only_forward_fields"]
            and structure_checks["no_forbidden_forward_fields"],
            contract.S0_GATES[5]: structure_checks["model_integrity"]
            and structure_checks["target_only_write"]
            and structure_checks["inactive_identity"],
            contract.S0_GATES[6]: all(
                structure_checks[name]
                for name in (
                    "permutation_logits",
                    "permutation_trajectory",
                    "card_isolation",
                    "operation_absent_from_no_core",
                    "no_core_group_invariance",
                )
            ),
            contract.S0_GATES[7]: structure_checks["operation2_gradient_connectivity"],
            contract.S0_GATES[8]: structure_checks["complete_counterfactual_groups"]
            and structure_checks["loss_finite"]
            and structure_checks["gradients_finite"],
            contract.S0_GATES[9]: structure_checks["no_optimizer"]
            and structural["optimizer_steps"] == 0
            and structural["model_writes"] == 0
            and source_report.get("trainable_source_parameters") == 0
            and post_hashes == hashes,
        }
        passed = all(gates.values())
        _maybe_fault(fault_at, "before_result")
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s0-result.v1",
            "identity": contract.S0_IDENTITY,
            "stage": "s0",
            "status": "PASS_V2_A_C1U_S0_QUALIFICATION" if passed else "FAIL_V2_A_C1U_S0_QUALIFICATION",
            "passed": passed,
            "authorizes": contract.S0_PASS_AUTHORIZATION if passed else "nothing",
            "s1_status": "CONTRACT_DESIGN_AUTHORIZED_NOT_RUN" if passed else "NOT_AUTHORIZED",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
            "gates": gates,
            "records": len(records),
            "cards": cache_audit["card_count"],
            "source_identity": source_identity(hashes),
            "source_hashes": hashes,
            "source_encoder_report": source_report,
            "predecessor": predecessor,
            "task_audit": task_audit,
            "cache_manifest": persistent_manifest,
            "structural_summary": structural,
            "device": hardware,
            "optimizer_steps": 0,
            "model_writes": 0,
            "training_started": False,
            "qualification_requires_seal_replay": True,
        }
        write_json(root / "result.json", result)
        write_evidence_seal(root, identity=contract.S0_IDENTITY, stage="s0")
        return {**result, "seal_replay": audit_evidence_seal(root, expected_identity=contract.S0_IDENTITY, expected_stage="s0")}
    except Exception as exc:
        return _crash_result(
            root,
            identity=contract.S0_IDENTITY,
            stage="s0",
            exc=exc,
            hashes=hashes,
        )


__all__ = [
    "audit_c1t_predecessor",
    "audit_s0_preflight",
    "audit_s0_root",
    "audit_unconsumed_paths",
    "run_s0",
    "run_s0_preflight",
    "structural_measurements",
]
