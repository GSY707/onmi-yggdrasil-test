from __future__ import annotations

"""Single-use S0 zero-update qualification runner for C1S."""

import gc
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from . import artifacts, contract
from .metrics import geometry_summary, registered_control_report
from .model import BoundaryOutput, C1SConfig, C1SModel, strip_training_auxiliary


AuditHook = Callable[[Path], Mapping[str, Any]]
StageHook = Callable[[Path, str], Mapping[str, Any]]


def _config(value: Mapping[str, Any]) -> C1SConfig:
    return C1SConfig(**dict(value))


def _refusal(root: Path, lease: Path, status: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.refusal.v1",
        "identity": contract.IDENTITY,
        "stage": "S0",
        "formal": False,
        "qualification": True,
        "status": status,
        "passed": False,
        "exit_code": 2,
        "output_root": root.as_posix(),
        "lease_path": lease.as_posix(),
        "reason": reason,
        "authorizes": "nothing",
        "c2_authorized": False,
        "v2a_passed": False,
    }


def audit_pins(repo_root: Path) -> dict[str, Any]:
    rows = {
        name: artifacts.audit_external_pin(repo_root, pin, replay=True)
        for name, pin in contract.PINNED_ROOTS.items()
    }
    return {"passed": all(row.get("passed") is True for row in rows.values()), "roots": rows}


def cuda_device_audit(device: str) -> dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        return {"passed": False, "requested_device": device, "reason": "C1S S0 requires CUDA"}
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    return {
        "passed": contract.REQUIRED_CUDA_DEVICE_NAME_FRAGMENT.casefold() in name.casefold(),
        "requested_device": device,
        "selected_device_index": index,
        "selected_device_name": name,
        "capability": list(capability),
        "device_count": torch.cuda.device_count(),
    }


def _permutation_report(model: C1SModel, source_hidden: torch.Tensor, source_mask: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        boundary = model.boundary(source_hidden, source_mask)
        baseline = model.forward_from_boundary(boundary, return_trajectory=True)
        permutation = torch.arange(model.config.slots - 1, -1, -1)
        permuted = model.forward_from_boundary(boundary.permuted(permutation), return_trajectory=True)
    logit_delta = (baseline["logits"] - permuted["logits"]).abs().max()
    expected_trajectory = baseline["trajectory"][:, :, permutation]
    trajectory_delta = (expected_trajectory - permuted["trajectory"]).abs().max()
    return {
        "permutation": permutation.tolist(),
        "logit_max_abs_delta": float(logit_delta.item()),
        "trajectory_max_abs_delta": float(trajectory_delta.item()),
    }


def _transition_report(model: C1SModel) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(contract.MODEL_SEED + 2)
    config = model.config
    payloads = torch.randn((2, config.slots, config.payload_width), generator=generator)
    operation = torch.randn((2, config.payload_width), generator=generator)
    source = torch.zeros((2, config.slots))
    target = torch.zeros((2, config.slots))
    source[:, 1] = 1.0
    target[:, min(5, config.slots - 1)] = 1.0
    model.transition.eval()
    with torch.no_grad():
        updated = model.transition(payloads, operation, source, target, torch.ones(2))
        inactive = model.transition(payloads, operation, source, target, torch.zeros(2))
    delta = (updated - payloads).abs()
    selected = torch.zeros(config.slots, dtype=torch.bool)
    selected[1] = True
    selected[min(5, config.slots - 1)] = True
    unselected_delta = delta[:, ~selected].max() if bool((~selected).any()) else delta.new_zeros(())
    selected_delta = delta[:, selected].max()
    return {
        "unselected_slot_max_delta": float(unselected_delta.item()),
        "selected_slot_max_delta": float(selected_delta.item()),
        "inactive_transition_max_delta": float((inactive - payloads).abs().max().item()),
        "source_slot": 1,
        "target_slot": min(5, config.slots - 1),
    }


def _strip_report(model: C1SModel, source_hidden: torch.Tensor, source_mask: torch.Tensor) -> dict[str, Any]:
    model.eval()
    deployment = strip_training_auxiliary(model).eval()
    with torch.no_grad():
        before = model(source_hidden, source_mask, return_auxiliary=True)
        after = deployment(source_hidden, source_mask, return_auxiliary=True)
    deployment_keys = sorted(deployment.state_dict())
    return {
        "prestrip_auxiliary_present": before.get("auxiliary") is not None,
        "poststrip_auxiliary_is_none": after.get("auxiliary") is None,
        "deployment_auxiliary_keys": [key for key in deployment_keys if key.startswith("training_auxiliary.")],
        "logit_max_abs_delta": float((before["logits"] - after["logits"]).abs().max().item()),
        "prestrip_parameters": model.parameter_report(),
        "poststrip_parameters": deployment.parameter_report(),
    }


def structural_qualification(_repo_root: Path, _device: str) -> dict[str, Any]:
    torch.manual_seed(contract.MODEL_SEED)
    config8 = _config(contract.MODEL_CONFIG)
    config1 = _config(contract.K1_CONFIG)
    model8 = C1SModel(config8).cpu()
    torch.manual_seed(contract.MODEL_SEED)
    model1 = C1SModel(config1).cpu()
    source_generator = torch.Generator(device="cpu").manual_seed(contract.MODEL_SEED + 1)
    source_hidden = torch.randn((2, 17, config8.source_width), generator=source_generator)
    source_mask = torch.ones((2, 17), dtype=torch.bool)
    source_mask[1, -3:] = False

    integrity8 = model8.integrity_report()
    integrity1 = model1.integrity_report()
    parameters8 = model8.parameter_report()
    parameters1 = model1.parameter_report()
    permutation = _permutation_report(model8, source_hidden, source_mask)
    transition = _transition_report(model8)
    controls = registered_control_report()
    strip = _strip_report(model8, source_hidden, source_mask)
    with torch.no_grad():
        output = model8(source_hidden, source_mask, return_trajectory=True)
    geometry = {
        "h0": geometry_summary(output["initial_payloads"]),
        "h10": geometry_summary(output["final_payloads"]),
    }

    thresholds = contract.S0_THRESHOLDS
    positive = controls["addressed_positive"]
    uniform = controls["legacy_uniform_mean_null"]
    checks = {
        "k8_integrity": integrity8.get("passed") is True,
        "k1_integrity": integrity1.get("passed") is True,
        "public_source_hidden_only": integrity8.get("public_source_hidden_only") is True,
        "direct_switch_core": integrity8.get("dense_all_slot_attention") is False
        and integrity8.get("global_answer_query") is False
        and integrity8.get("input_conditioned_query") is True,
        "parameter_parity": parameters8["trainable_parameters"] == parameters1["trainable_parameters"],
        "permutation_logits": permutation["logit_max_abs_delta"] <= thresholds["permutation_logit_max_abs"],
        "permutation_trajectory": permutation["trajectory_max_abs_delta"] <= thresholds["permutation_trajectory_max_abs"],
        "targeted_unselected": transition["unselected_slot_max_delta"] <= thresholds["unselected_slot_max_delta"],
        "targeted_selected": transition["selected_slot_max_delta"] > thresholds["selected_slot_min_delta"],
        "inactive_identity": transition["inactive_transition_max_delta"] <= thresholds["inactive_transition_max_delta"],
        "query_swap_positive": controls["query_swap_logit_l2"] >= thresholds["query_swap_min_logit_l2"],
        "duplicate_null": controls["duplicate_query_swap_logit_l2"] <= thresholds["duplicate_query_swap_max_logit_l2"],
        "functional_positive_relevant": min(positive["relevant_effect"]) >= thresholds["positive_relevant_mean_replace_min"],
        "functional_positive_irrelevant": max(positive["irrelevant_max_effect"]) <= thresholds["positive_irrelevant_mean_replace_max"],
        "uniform_null": max(uniform["ownership_contrast"]) <= thresholds["uniform_null_ownership_contrast_max"],
        "k1_null": controls["k1_null"]["single_slot_null"] is True,
        "auxiliary_stripped": strip["poststrip_auxiliary_is_none"] is True
        and not strip["deployment_auxiliary_keys"],
        "strip_logit_identity": strip["logit_max_abs_delta"] <= thresholds["strip_logit_max_abs"],
    }
    report = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.s0-structure.v1",
        "identity": contract.IDENTITY,
        "config_k8": integrity8["config"],
        "config_k1": integrity1["config"],
        "integrity_k8": integrity8,
        "integrity_k1": integrity1,
        "parameters_k8": parameters8,
        "parameters_k1": parameters1,
        "permutation": permutation,
        "targeted_transition": transition,
        "registered_controls": controls,
        "auxiliary_strip": strip,
        "random_initialization_geometry_diagnostic_only": geometry,
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(checks.values()),
    }
    del model8, model1, output
    gc.collect()
    return report


def cuda_bf16_smoke(_repo_root: Path, device: str) -> dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        return {"passed": False, "reason": "CUDA unavailable", "optimizer_steps": 0, "model_writes": 0}
    torch.manual_seed(contract.MODEL_SEED)
    torch.cuda.manual_seed_all(contract.MODEL_SEED)
    active_device = torch.device("cuda")
    config = _config(contract.MODEL_CONFIG)
    model = C1SModel(config).to(active_device).train()
    torch.cuda.reset_peak_memory_stats(active_device)
    source = torch.randn((2, 65, config.source_width), device=active_device, dtype=torch.float16)
    mask = torch.ones((2, 65), device=active_device, dtype=torch.bool)
    labels = torch.tensor([0, 8], device=active_device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(source, mask, return_trajectory=True, return_auxiliary=True)
        loss = torch.nn.functional.cross_entropy(output["logits"].float(), labels)
        auxiliary = output["auxiliary"]
        if auxiliary is None:
            raise RuntimeError("training auxiliary missing in registered S0 smoke")
        loss = loss + 1.0e-5 * sum(value.float().square().mean() for value in auxiliary.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    output_tensors = [value for value in output.values() if isinstance(value, torch.Tensor)]
    output_tensors.extend(auxiliary.values())
    finite_outputs = all(torch.isfinite(value).all().item() for value in output_tensors)
    finite_gradients = bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients)
    nonzero_gradients = sum(int(torch.count_nonzero(gradient).item() > 0) for gradient in gradients)
    report = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.s0-cuda-smoke.v1",
        "identity": contract.IDENTITY,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "autocast_dtype": "bfloat16",
        "batch": 2,
        "source_tokens": 65,
        "loss": float(loss.detach().item()),
        "finite_outputs": bool(finite_outputs),
        "finite_gradients": bool(finite_gradients),
        "gradient_tensors": len(gradients),
        "nonzero_gradient_tensors": nonzero_gradients,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(active_device)),
        "optimizer_steps": 0,
        "model_writes": 0,
        "backward_smoke_only": True,
        "passed": bool(finite_outputs and finite_gradients and nonzero_gradients > 0),
    }
    del model, source, mask, labels, output, loss, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_s0_preflight(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    device: str = "cuda",
    pin_auditor: AuditHook = audit_pins,
    device_auditor: Callable[[str], Mapping[str, Any]] = cuda_device_audit,
    structure_stage: StageHook = structural_qualification,
    cuda_stage: StageHook = cuda_bf16_smoke,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = Path(output_root) if output_root is not None else repo_root / contract.S0_ROOT
    lease = Path(lease_path) if lease_path is not None else repo_root / contract.S0_LEASE
    if root.exists() or lease.exists():
        return _refusal(root, lease, "REFUSE_V2_A_C1S_S0_SINGLE_USE", "fixed S0 root or lease already exists")

    try:
        source_before = artifacts.source_hashes(repo_root)
        source_id = artifacts.source_identity(source_before)
        pins_before = dict(pin_auditor(repo_root))
        device_report = dict(device_auditor(device))
    except Exception as exc:  # pre-lease refusal must not create evidence state
        return _refusal(root, lease, "REFUSE_V2_A_C1S_S0_PREREQUISITE", f"{type(exc).__name__}: {exc}")
    if pins_before.get("passed") is not True:
        return _refusal(root, lease, "REFUSE_V2_A_C1S_S0_PIN", "pinned predecessor root audit failed")
    if device_report.get("passed") is not True:
        return _refusal(root, lease, "REFUSE_V2_A_C1S_S0_DEVICE", f"registered CUDA device required: {device_report}")

    try:
        artifacts.claim_single_use(root, lease)
        artifacts.snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.contract_manifest())
        artifacts.write_json(
            root / "prerequisites.json",
            {
                "schema_version": f"{contract.SCHEMA_PREFIX}.prerequisites.v1",
                "identity": contract.IDENTITY,
                "source_identity": source_id,
                "source_hashes": source_before,
                "pins": pins_before,
                "device": device_report,
            },
        )

        structure = dict(structure_stage(repo_root, device))
        artifacts.write_json(root / "s0-structure.json", structure)
        cuda = dict(cuda_stage(repo_root, device)) if structure.get("passed") is True else {
            "passed": False,
            "status": "NOT_RUN_AFTER_S0_STRUCTURE_FAIL",
            "optimizer_steps": 0,
            "model_writes": 0,
        }
        artifacts.write_json(root / "s0-cuda-smoke.json", cuda)

        source_after = artifacts.source_hashes(repo_root)
        pins_after = dict(pin_auditor(repo_root))
        source_stable = source_before == source_after
        pins_stable = pins_before == pins_after and pins_after.get("passed") is True
        zero_accounting = (
            int(cuda.get("optimizer_steps", -1)) == 0
            and int(cuda.get("model_writes", -1)) == 0
        )
        passed = bool(
            structure.get("passed") is True
            and cuda.get("passed") is True
            and source_stable
            and pins_stable
            and zero_accounting
        )
        status = "PASS_V2_A_C1S_S0_QUALIFICATION" if passed else "FAIL_V2_A_C1S_S0_QUALIFICATION"
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s0-result.v1",
            "identity": contract.IDENTITY,
            "stage": "S0",
            "formal": False,
            "qualification": True,
            "status": status,
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "source_identity": source_id,
            "source_stable": source_stable,
            "old_roots_stable": pins_stable,
            "old_root_writes": 0 if pins_stable else None,
            "training_started": False,
            "backward_smoke_only": True,
            "optimizer_steps": int(cuda.get("optimizer_steps", 0)),
            "model_writes": int(cuda.get("model_writes", 0)),
            "device": device_report,
            "gates": {
                "S001_identity_pins": source_stable and pins_stable,
                "S002_to_S009_structure_measurement": structure.get("passed") is True,
                "S010_numerics_device": cuda.get("passed") is True and zero_accounting,
                "S011_seal": True,
            },
            "s1_status": "AUTHORIZED_NOT_RUN" if passed else "NOT_AUTHORIZED",
            "s2_status": "NOT_AUTHORIZED",
            "formal_status": "NOT_AUTHORIZED",
            "authorizes": contract.AUTHORIZATION_ON_S0_PASS if passed else "nothing",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
            "c2_authorized": False,
            "v2a_passed": False,
        }
        artifacts.write_json(root / "result.json", result)
        seal_sha = artifacts.write_evidence_seal(root)
        replay = artifacts.audit_evidence_seal(root)
        return {
            **result,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "result_sha256": artifacts.sha256_file(root / "result.json"),
            "evidence_seal_sha256": seal_sha,
            "seal_replay": replay,
        }
    except Exception as exc:
        if root.exists():
            crash = {
                "schema_version": f"{contract.SCHEMA_PREFIX}.s0-result.v1",
                "identity": contract.IDENTITY,
                "stage": "S0",
                "formal": False,
                "qualification": True,
                "status": "CRASH_V2_A_C1S_S0_QUALIFICATION",
                "passed": False,
                "exit_code": 1,
                "reason": f"{type(exc).__name__}: {exc}",
                "training_started": False,
                "optimizer_steps": 0,
                "model_writes": 0,
                "authorizes": "nothing",
                "c2_authorized": False,
                "v2a_passed": False,
            }
            artifacts.write_json(root / "result.json", crash)
            seal_sha = artifacts.write_evidence_seal(root)
            return {
                **crash,
                "output_root": root.as_posix(),
                "lease_path": lease.as_posix(),
                "result_sha256": artifacts.sha256_file(root / "result.json"),
                "evidence_seal_sha256": seal_sha,
            }
        return _refusal(root, lease, "CRASH_V2_A_C1S_S0_PREREQUISITE", f"{type(exc).__name__}: {exc}")


def load_result(root: Path) -> dict[str, Any]:
    return json.loads((Path(root) / "result.json").read_text(encoding="utf-8"))


__all__ = [
    "audit_pins",
    "cuda_bf16_smoke",
    "cuda_device_audit",
    "load_result",
    "run_s0_preflight",
    "structural_qualification",
]
