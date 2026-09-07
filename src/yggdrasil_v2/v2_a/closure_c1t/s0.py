from __future__ import annotations

"""Single-use C1T S0 preflight and zero-training qualification."""

from dataclasses import asdict
from pathlib import Path
import traceback
from typing import Any, Callable, Mapping, Sequence

import torch

from . import contract
from .artifacts import (
    audit_evidence_seal,
    claim_single_use,
    read_json,
    sha256_file,
    snapshot_sources,
    source_hashes,
    source_identity,
    write_evidence_seal,
    write_json,
)
from .cache import audit_persisted_cache, persist_independent_cache
from .model import C1TConfig, C1TModel
from .objective import compute_training_loss
from .runtime import (
    C1TRuntime,
    IndependentCardCache,
    audit_independent_card_cache,
    build_independent_card_cache,
)
from .source import QwenCardEncoder, audit_device, audit_source_assets
from .tasks import audit_factorial_group, audit_s1_bank, build_s1_bank


AssetAudit = Callable[[], Mapping[str, Any]]
DeviceAudit = Callable[[str | torch.device], Mapping[str, Any]]
EncoderFactory = Callable[..., Any]


def _fixed_paths(repo_root: Path) -> dict[str, Path]:
    return {
        "s0_preflight_root": repo_root / contract.S0_PREFLIGHT_ROOT,
        "s0_preflight_lease": repo_root / contract.S0_PREFLIGHT_LEASE,
        "s0_root": repo_root / contract.S0_ROOT,
        "s0_lease": repo_root / contract.S0_LEASE,
        "s1_preflight_root": repo_root / contract.S1_PREFLIGHT_ROOT,
        "s1_preflight_lease": repo_root / contract.S1_PREFLIGHT_LEASE,
        "s1_root": repo_root / contract.S1_ROOT,
        "s1_lease": repo_root / contract.S1_LEASE,
    }


def audit_unconsumed_paths(repo_root: Path, *, stage: str) -> dict[str, Any]:
    paths = _fixed_paths(Path(repo_root).resolve())
    if stage == "s0-preflight":
        required_absent = tuple(paths)
    elif stage == "s0":
        required_absent = (
            "s0_root",
            "s0_lease",
            "s1_preflight_root",
            "s1_preflight_lease",
            "s1_root",
            "s1_lease",
        )
    else:
        raise ValueError(f"unknown path-audit stage: {stage}")
    existence = {name: path.exists() for name, path in paths.items()}
    checks = {name: not existence[name] for name in required_absent}
    return {
        "passed": all(checks.values()),
        "stage": stage,
        "checks": checks,
        "exists": existence,
        "paths": {name: path.as_posix() for name, path in paths.items()},
    }


def _preflight_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups = {
        f"c1t-{family.casefold()}-g{index:02d}"
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
    checks = {
        "nonempty": bool(records),
        "complete_groups": bool(audits) and all(row["passed"] for row in audits.values()),
        "unique_examples": len({str(row["example_id"]) for row in records}) == len(records),
        "both_families": families == set(contract.FAMILIES),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "records": len(records),
        "groups": len(grouped),
        "group_audits": audits,
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


def _model_config(source_width: int, override: C1TConfig | None) -> C1TConfig:
    if override is None:
        return C1TConfig(source_width=source_width)
    if override.source_width != source_width:
        raise ValueError("test/registered model config source width mismatch")
    return override


def _no_core_group_invariance(
    model: C1TModel,
    runtime: C1TRuntime,
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
        "passed": maximum <= float(contract.S1_GATES["no_core_group_logit_tolerance"]),
        "threshold": float(contract.S1_GATES["no_core_group_logit_tolerance"]),
        "maximum_abs_delta": maximum,
        "groups": rows,
    }


def structural_measurements(
    records: Sequence[Mapping[str, Any]],
    cache: IndependentCardCache,
    *,
    device: str | torch.device,
    config_override: C1TConfig | None = None,
) -> dict[str, Any]:
    """Run S0 structure and BF16 backward checks without an optimizer."""

    selected_device = torch.device(device)
    config = _model_config(cache.source_width, config_override)
    runtime = C1TRuntime(records, cache, address_width=config.address_width)
    torch.manual_seed(contract.MODEL_SEED)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(contract.MODEL_SEED)
    model = C1TModel(config).to(selected_device)
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

        first_target = int(boundary.target_weights[0, 0].argmax())
        non_targets = [index for index in range(config.slots) if index != first_target]
        target_only_delta = float(
            (
                base["trajectory"][0, 1, non_targets]
                - base["trajectory"][0, 0, non_targets]
            ).abs().max().float().cpu()
        )

    no_core_invariance = _no_core_group_invariance(
        model, runtime, records, selected_device
    )
    model.train()
    model.zero_grad(set_to_none=True)
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
    parameter_unchanged_by_backward = True
    checks = {
        "model_integrity": model.integrity_report()["passed"],
        "source_only_forward_fields": set(runtime.forward_inputs(batch))
        == set(contract.FORWARD_FIELDS),
        "no_forbidden_forward_fields": set(runtime.forward_inputs(batch)).isdisjoint(
            contract.FORBIDDEN_FORWARD_FIELDS
        ),
        "complete_counterfactual_groups": bool((batch["counterfactual_indices"] >= 0).all()),
        "permutation_logits": permutation_logit_delta
        <= float(contract.S1_GATES["permutation_logit_tolerance"]),
        "permutation_trajectory": permutation_trajectory_delta
        <= float(contract.S1_GATES["permutation_logit_tolerance"]),
        "card_isolation": card_isolation_local > 0.0 and card_isolation_nonlocal == 0.0,
        "operation_absent_from_no_core": operation_no_core_delta == 0.0,
        "target_only_write": target_only_delta == 0.0,
        "no_core_group_invariance": no_core_invariance["passed"],
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
        "no_core_group_invariance": no_core_invariance,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(selected_device))
        if selected_device.type == "cuda"
        else 0,
        "optimizer_steps": 0,
        "model_writes": 0,
    }
    del model, batch, runtime
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
        checks = {
            "lease_exists": (repo_root / contract.S0_PREFLIGHT_LEASE).is_file(),
            "identity": result.get("identity") == contract.S0_PREFLIGHT_IDENTITY,
            "status": result.get("status") == "PASS_V2_A_C1T_S0_PREFLIGHT",
            "passed": result.get("passed") is True,
            "authorization": result.get("authorizes") == contract.S0_AUTHORIZED_SCOPE,
            "optimizer_steps": result.get("optimizer_steps") == 0,
            "model_writes": result.get("model_writes") == 0,
            "source_hashes": result.get("source_hashes") == current_hashes,
            "source_identity": result.get("source_identity") == source_identity(current_hashes),
            "seal": seal.get("passed") is True,
        }
        return {"passed": all(checks.values()), "checks": checks, "result": result, "seal": seal}
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
        checks = {
            "lease_exists": (repo_root / contract.S0_LEASE).is_file(),
            "identity": result.get("identity") == contract.S0_IDENTITY,
            "status": result.get("status") == "PASS_V2_A_C1T_S0_QUALIFICATION",
            "passed": result.get("passed") is True,
            "authorization": result.get("authorizes") == contract.S0_PASS_AUTHORIZATION,
            "optimizer_steps": result.get("optimizer_steps") == 0,
            "model_writes": result.get("model_writes") == 0,
            "source_hashes": result.get("source_hashes") == current_hashes,
            "source_identity": result.get("source_identity") == source_identity(current_hashes),
            "seal": seal.get("passed") is True,
        }
        return {"passed": all(checks.values()), "checks": checks, "result": result, "seal": seal}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _crash_result(
    root: Path, *, identity: str, stage: str, exc: Exception, hashes: Mapping[str, str] | None
) -> dict[str, Any]:
    result = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.{stage}-result.v1",
        "identity": identity,
        "stage": stage,
        "status": f"CRASH_V2_A_C1T_{stage.upper().replace('-', '_')}",
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
        raise RuntimeError(f"injected C1T S0 fault at {point}")


def run_s0_preflight(
    repo_root: Path,
    *,
    device: str | torch.device = "cuda",
    asset_audit_fn: AssetAudit = audit_source_assets,
    device_audit_fn: DeviceAudit = audit_device,
    encoder_factory: EncoderFactory = QwenCardEncoder,
    config_override: C1TConfig | None = None,
    fault_at: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S0_PREFLIGHT_ROOT
    lease = repo_root / contract.S0_PREFLIGHT_LEASE
    hashes = source_hashes(repo_root)
    paths = audit_unconsumed_paths(repo_root, stage="s0-preflight")
    assets = dict(asset_audit_fn())
    hardware = dict(device_audit_fn(device))
    if not paths["passed"] or not assets.get("passed") or not hardware.get("passed"):
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
        write_json(root / "preclaim-audit.json", {"paths": paths, "assets": assets, "device": hardware})
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
        gates = {
            "P001_source_device_and_paths": paths["passed"] and assets["passed"] and hardware["passed"],
            "P002_task_factorial": task_audit["passed"],
            "P003_real_independent_cards": cache_audit["passed"]
            and source_report.get("calls") == contract.S0_PREFLIGHT_CARDS,
            "P004_bf16_structure_and_accounting": structural["passed"]
            and structural["optimizer_steps"] == 0
            and structural["model_writes"] == 0,
            "P005_source_stable": post_hashes == hashes,
        }
        passed = all(gates.values())
        _maybe_fault(fault_at, "before_result")
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s0-preflight-result.v1",
            "identity": contract.S0_PREFLIGHT_IDENTITY,
            "stage": "s0-preflight",
            "status": "PASS_V2_A_C1T_S0_PREFLIGHT" if passed else "FAIL_V2_A_C1T_S0_PREFLIGHT",
            "passed": passed,
            "authorizes": contract.S0_AUTHORIZED_SCOPE if passed else "nothing",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
            "gates": gates,
            "records": len(records),
            "cards": cache_audit["card_count"],
            "source_identity": source_identity(hashes),
            "source_hashes": hashes,
            "source_encoder_report": source_report,
            "device": hardware,
            "optimizer_steps": 0,
            "model_writes": 0,
            "training_started": False,
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
    config_override: C1TConfig | None = None,
    fault_at: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    preflight = audit_s0_preflight(repo_root)
    paths = audit_unconsumed_paths(repo_root, stage="s0")
    hashes = source_hashes(repo_root)
    assets = dict(asset_audit_fn())
    hardware = dict(device_audit_fn(device))
    if not preflight.get("passed") or not paths["passed"] or not assets.get("passed") or not hardware.get("passed"):
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
        write_json(root / "preclaim-audit.json", {"preflight": preflight, "paths": paths, "assets": assets, "device": hardware})
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
        gates = {
            contract.S0_GATES[0]: preflight["passed"] and assets["passed"] and hardware["passed"],
            contract.S0_GATES[1]: task_audit["passed"],
            contract.S0_GATES[2]: cache_audit["passed"]
            and persistent_audit["passed"]
            and source_report.get("calls") == contract.S0_CARDS
            and persistent_manifest.get("contains_targets") is False,
            contract.S0_GATES[3]: structure_checks["model_integrity"]
            and structure_checks["source_only_forward_fields"]
            and structure_checks["no_forbidden_forward_fields"],
            contract.S0_GATES[4]: all(
                structure_checks[name]
                for name in (
                    "permutation_logits",
                    "permutation_trajectory",
                    "card_isolation",
                    "operation_absent_from_no_core",
                    "target_only_write",
                    "complete_counterfactual_groups",
                )
            ),
            contract.S0_GATES[5]: structure_checks["no_core_group_invariance"],
            contract.S0_GATES[6]: structure_checks["loss_finite"]
            and structure_checks["gradients_finite"],
            contract.S0_GATES[7]: structural["optimizer_steps"] == 0
            and structural["model_writes"] == 0
            and post_hashes == hashes,
        }
        passed = all(gates.values())
        _maybe_fault(fault_at, "before_result")
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s0-result.v1",
            "identity": contract.S0_IDENTITY,
            "stage": "s0",
            "status": "PASS_V2_A_C1T_S0_QUALIFICATION" if passed else "FAIL_V2_A_C1T_S0_QUALIFICATION",
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
    "audit_s0_preflight",
    "audit_s0_root",
    "audit_unconsumed_paths",
    "run_s0",
    "run_s0_preflight",
    "structural_measurements",
]
