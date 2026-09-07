from __future__ import annotations

"""Single-use, read-only execution of the C1S S1 failure diagnosis."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import torch

from ..closure_c1s.model import C1SConfig, C1SModel
from ..closure_c1s_successor import artifacts as successor_artifacts
from ..closure_c1s_successor import contract as successor_contract
from ..closure_c1s_successor.objective import SuccessorDiagnosticHeads
from ..closure_c1s_successor.runner import _audit_cache_integrity
from ..closure_c1s_successor.targets import audit_target_bank, materialize_target
from . import artifacts, contract
from .arity_analysis import audit_task_and_model_arity, load_source_records
from .decision import (
    classify_a_axis,
    classify_b_axis,
    classify_c_axis,
    classify_loss_gate_alignment,
)
from .endpoint import load_endpoint_artifact
from .objective_alignment import audit_objective_alignment
from .path_analysis import PATH_NAMES, collect_paths, summarise_paths
from .runtime import bank_rows, build_runtime, load_gzip_json, load_json, s1_ids
from .temporal import HARD_FEATURES, audit_temporal_readout


STAGE = "S1-FAILURE-ATTRIBUTION"
PREFLIGHT_STAGE = f"{STAGE}-PREFLIGHT"
PREFLIGHT_IDENTITY = f"{contract.IDENTITY}-PREFLIGHT"

DIAGNOSIS_SOURCE_FILES = (
    contract.DESIGN_DOC,
    Path("experiments/v2_a_closure_c1s_s1_diagnosis.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/artifacts.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/arity_analysis.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/decision.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/endpoint.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/metrics.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/objective_alignment.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/path_analysis.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/runner.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/runtime.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/task_arity.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_diagnosis/temporal.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s/model.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/cache.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1/data.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c0r/compact_trace.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/objective.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/runtime.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/targets.py"),
    Path("src/yggdrasil_v2/r1_revalidation/common/simulator.py"),
    Path("src/yggdrasil_v2/r1_revalidation/common/__init__.py"),
    Path("src/yggdrasil_v2/r1_revalidation/production/renderer.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_artifacts.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_arity_analysis.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_decision.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_endpoint.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_metrics.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_path_objective.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_runtime_helpers.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_task_arity.py"),
    Path("tests/v2_a_closure_c1s_s1_diagnosis/test_temporal.py"),
)

# Importing the successor runtime/model also loads its package-level
# transitive closure.  Reuse that already audited list, then add the diagnosis
# package and its own tests.  This prevents a narrow hand-written list from
# silently omitting code that actually executes.
SOURCE_FILES = tuple(
    dict.fromkeys((*successor_artifacts.SOURCE_FILES, *DIAGNOSIS_SOURCE_FILES))
)


def _source_hashes(repo_root: Path) -> dict[str, str]:
    missing = [path.as_posix() for path in SOURCE_FILES if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"diagnosis source identity is incomplete: {missing}")
    return {
        path.as_posix(): artifacts.sha256_file(repo_root / path) for path in SOURCE_FILES
    }


def _source_identity(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _snapshot_sources(repo_root: Path, output_root: Path) -> None:
    for relative in SOURCE_FILES:
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)


def _device_audit(device: str | torch.device) -> dict[str, Any]:
    active = torch.device(device)
    if active.type != "cuda":
        return {
            "device": str(active),
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": False,
            "passed": False,
            "reason": "registered diagnosis requires the qualified CUDA BF16 path",
        }
    index = active.index or 0
    available = torch.cuda.is_available() and torch.cuda.device_count() > index
    if not available:
        return {"device": str(active), "cuda_available": False, "bf16_supported": False, "passed": False}
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    bf16 = bool(torch.cuda.is_bf16_supported())
    return {
        "device": str(active),
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "name": name,
        "capability": list(capability),
        "bf16_supported": bf16,
        "passed": bool("RTX 4070" in name and capability >= (8, 0) and bf16),
    }


def _run_tests(repo_root: Path) -> dict[str, Any]:
    # These two historical test directories contain same-basename modules.
    # Run them in separate interpreter processes so pytest's default import
    # mode cannot alias one directory's test_artifacts.py to the other.
    commands = [
        [sys.executable, "-m", "pytest", "-q", path]
        for path in (
            "tests/v2_a_closure_c1s_s1_diagnosis",
            "tests/v2_a_closure_c1s_successor",
        )
    ]
    runs: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        runs.append(
            {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout[-20_000:],
                "stderr": completed.stderr[-20_000:],
                "passed": completed.returncode == 0,
            }
        )
    return {
        "runs": runs,
        "passed": all(run["passed"] for run in runs),
    }


def _later_stage_absence(repo_root: Path) -> dict[str, Any]:
    paths = (
        successor_contract.S2_ROOT,
        successor_contract.S2_LEASE,
        successor_contract.S2_PREFLIGHT_ROOT,
        successor_contract.S2_PREFLIGHT_LEASE,
        successor_contract.S3_ROOT,
        successor_contract.S3_LEASE,
        successor_contract.S3_PREFLIGHT_ROOT,
        successor_contract.S3_PREFLIGHT_LEASE,
    )
    rows = {path.as_posix(): not (repo_root / path).exists() for path in paths}
    return {"paths_absent": rows, "passed": all(rows.values())}


def _audit_inputs(repo_root: Path, *, full_bank_audit: bool) -> dict[str, Any]:
    s1_root = repo_root / contract.S1_ROOT
    s1_result = artifacts.read_json(s1_root / "result.json")
    s1_seal = artifacts.audit_evidence_seal(
        s1_root,
        expected_identity=contract.S1_IDENTITY,
        expected_stage="S1",
    )
    s1_checks = {
        "result_sha256": artifacts.sha256_file(s1_root / "result.json") == contract.S1_RESULT_SHA256,
        "seal_sha256": artifacts.sha256_file(s1_root / "evidence-seal.json") == contract.S1_EVIDENCE_SEAL_SHA256,
        "endpoint_sha256": artifacts.sha256_file(s1_root / "endpoint.pt") == contract.S1_ENDPOINT_SHA256,
        "seal_replay": s1_seal.get("passed") is True,
        "identity": s1_result.get("identity") == contract.S1_IDENTITY,
        "status": s1_result.get("status") == contract.S1_STATUS,
        "failed": s1_result.get("passed") is False,
        "authorizes_nothing": s1_result.get("authorizes") == "nothing",
        "optimizer_steps": s1_result.get("optimizer_steps") == 4_000,
        "model_writes": s1_result.get("model_writes") == 1,
        "fixed_endpoint": s1_result.get("fixed_endpoint") == 4_000,
        "no_selection": s1_result.get("checkpoint_selection") is False,
        "source_identity": s1_result.get("source_identity") == contract.S1_SOURCE_IDENTITY,
    }

    preflight_root = repo_root / contract.S1_PREFLIGHT_ROOT
    preflight_result = artifacts.read_json(preflight_root / "result.json")
    preflight_seal = artifacts.audit_evidence_seal(
        preflight_root,
        expected_identity=contract.S1_PREFLIGHT_IDENTITY,
        expected_stage="S1-PREFLIGHT",
    )
    preflight_checks = {
        "result_sha256": artifacts.sha256_file(preflight_root / "result.json") == contract.S1_PREFLIGHT_RESULT_SHA256,
        "seal_sha256": artifacts.sha256_file(preflight_root / "evidence-seal.json") == contract.S1_PREFLIGHT_SEAL_SHA256,
        "seal_replay": preflight_seal.get("passed") is True,
        "identity": preflight_result.get("identity") == contract.S1_PREFLIGHT_IDENTITY,
        "status": preflight_result.get("status") == contract.S1_PREFLIGHT_STATUS,
        "passed": preflight_result.get("passed") is True,
        "source_identity": preflight_result.get("source_identity") == contract.S1_SOURCE_IDENTITY,
        "target_bank_sha256": artifacts.sha256_file(preflight_root / "target-bank.json.gz") == contract.TARGET_BANK_SHA256,
        "split_ledger_sha256": artifacts.sha256_file(preflight_root / "split-ledger.json") == contract.SPLIT_LEDGER_SHA256,
    }
    cache_integrity = artifacts.read_json(preflight_root / "cache-integrity.json")
    target_integrity = artifacts.read_json(preflight_root / "target-bank-integrity.json")
    split_audit = artifacts.read_json(preflight_root / "split-ledger-audit.json")
    preflight_checks.update(
        cache_integrity=cache_integrity.get("passed") is True,
        target_integrity=target_integrity.get("passed") is True,
        split_ledger_audit=split_audit.get("passed") is True,
    )
    # Time-of-use replay: a historical PASS report is not a substitute for
    # checking the current 68 GB cache tree and current source binding before
    # this new single-use identity is claimed.
    current_cache_integrity = _audit_cache_integrity(repo_root)
    c0r_source_integrity = artifacts.audit_external_pin(
        repo_root,
        successor_contract.PINNED_PREDECESSOR_ROOTS["c0r_data_trace"],
        replay=True,
    )
    bank = load_gzip_json(preflight_root / "target-bank.json.gz")
    ledger = load_json(preflight_root / "split-ledger.json")
    ids = s1_ids(ledger)
    rows = bank_rows(bank, ids)
    family_counts = {
        family: sum(str(row.get("family", "")).upper() == family for row in rows)
        for family in contract.FAMILIES
    }
    selection_checks = {
        "count": len(ids) == 32,
        "families": family_counts == {"CPS": 16, "ERE": 16},
        "train_only": all(row.get("split") == "train" for row in rows),
        "mechanism_supported": all(row.get("mechanism_supported") is True for row in rows),
        "query_answer_independent": all(row.get("query_answer_independent") is True for row in rows),
    }
    bank_report: dict[str, Any]
    if full_bank_audit:
        bank_report = audit_target_bank(bank, expected_count=26_624)
    else:
        bank_report = {"passed": True, "status": "PINNED_SEALED_BANK_REUSED"}
    endpoint = load_endpoint_artifact(
        s1_root / "endpoint.pt",
        expected_sha256=contract.S1_ENDPOINT_SHA256,
        expected_identity=contract.S1_IDENTITY,
        expected_update=4_000,
        expected_schedule_sha256=contract.S1_SCHEDULE_SHA256,
        expected_model_config=successor_contract.MODEL_CONFIG,
    )
    later = _later_stage_absence(repo_root)
    checks = {
        "s1": all(s1_checks.values()),
        "s1_preflight": all(preflight_checks.values()),
        "selection": all(selection_checks.values()),
        "target_bank": bank_report.get("passed") is True,
        "endpoint": endpoint["metadata"].get("sha256") == contract.S1_ENDPOINT_SHA256,
        "current_cache_tree_and_source": current_cache_integrity.get("passed") is True,
        "current_c0r_source_tree": c0r_source_integrity.get("passed") is True,
        "later_stages_absent": later.get("passed") is True,
    }
    return {
        "s1": {"checks": s1_checks, "seal": s1_seal},
        "s1_preflight": {"checks": preflight_checks, "seal": preflight_seal},
        "selection": {"checks": selection_checks, "family_counts": family_counts, "ids": ids},
        "target_bank_audit": bank_report,
        "current_cache_integrity": current_cache_integrity,
        "current_c0r_source_integrity": c0r_source_integrity,
        "endpoint": endpoint["metadata"],
        "later_stages": later,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _seal(root: Path, *, identity: str, stage: str) -> dict[str, Any]:
    seal_sha = artifacts.write_evidence_seal(
        root,
        identity=identity,
        stage=stage,
        schema_prefix=contract.SCHEMA_PREFIX,
    )
    replay = artifacts.audit_evidence_seal(
        root,
        expected_identity=identity,
        expected_stage=stage,
        expected_schema_prefix=contract.SCHEMA_PREFIX,
    )
    return {"seal_sha256": seal_sha, "seal_replay": replay}


def _refusal(reason: str, *, root: Path, lease: Path) -> dict[str, Any]:
    return {
        "status": "REFUSED",
        "passed": False,
        "reason": reason,
        "root": root.as_posix(),
        "lease": lease.as_posix(),
        "authorizes": "nothing",
    }


def _sealed_terminal_after_exception(
    root: Path,
    *,
    identity: str,
    stage: str,
    error: Exception,
) -> dict[str, Any] | None:
    """Never rewrite a result after its evidence seal has been materialized."""

    seal_path = root / "evidence-seal.json"
    result_path = root / "result.json"
    if not seal_path.is_file() or not result_path.is_file():
        return None
    existing = artifacts.read_json(result_path)
    try:
        replay = artifacts.audit_evidence_seal(
            root,
            expected_identity=identity,
            expected_stage=stage,
            expected_schema_prefix=contract.SCHEMA_PREFIX,
        )
    except Exception as replay_error:  # noqa: BLE001 - report without mutation
        replay = {
            "passed": False,
            "reason": f"{type(replay_error).__name__}: {replay_error}",
        }
    return {
        **existing,
        "root": root.as_posix(),
        "post_seal_exception_type": type(error).__name__,
        "post_seal_exception": str(error),
        "seal_sha256": artifacts.sha256_file(seal_path),
        "seal_replay": replay,
        "authorizes": "nothing",
    }


def run_preflight(repo_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.PREFLIGHT_ROOT
    lease = repo_root / contract.PREFLIGHT_LEASE
    if root.exists() or lease.exists():
        return _refusal("fixed diagnosis preflight root or lease already exists", root=root, lease=lease)
    artifacts.claim_single_use(
        identity=PREFLIGHT_IDENTITY,
        stage=PREFLIGHT_STAGE,
        output_root=root,
        lease_path=lease,
        formal=False,
    )
    try:
        started = time.perf_counter()
        source_hashes = _source_hashes(repo_root)
        source_identity = _source_identity(source_hashes)
        _snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.contract_manifest())
        artifacts.write_json(root / "source-hashes.json", {"identity": source_identity, "files": source_hashes})
        tests = _run_tests(repo_root)
        artifacts.write_json(root / "tests.json", tests)
        inputs = _audit_inputs(repo_root, full_bank_audit=True)
        artifacts.write_json(root / "D000-input-integrity.json", inputs)
        device_report = _device_audit(device)
        artifacts.write_json(root / "device.json", device_report)
        checks = {
            "source_complete": bool(source_hashes),
            "tests": tests.get("passed") is True,
            "inputs": inputs.get("passed") is True,
            "device": device_report.get("passed") is True,
            "diagnosis_root_absent": not (repo_root / contract.OUTPUT_ROOT).exists(),
            "diagnosis_lease_absent": not (repo_root / contract.LEASE_PATH).exists(),
        }
        passed = all(checks.values())
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": PREFLIGHT_IDENTITY,
            "stage": PREFLIGHT_STAGE,
            "status": "PASS_C1S_S1_FAILURE_ATTRIBUTION_PREFLIGHT" if passed else "FAIL_C1S_S1_FAILURE_ATTRIBUTION_PREFLIGHT",
            "passed": passed,
            "diagnostic_only": True,
            "checks": checks,
            "source_identity": source_identity,
            "optimizer_steps": 0,
            "model_writes": 0,
            "wall_seconds": time.perf_counter() - started,
            "authorizes": "THIS_DIAGNOSIS_SINGLE_USE_ONLY" if passed else "nothing",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
        }
        artifacts.write_json(root / "result.json", result)
        sealed = _seal(root, identity=PREFLIGHT_IDENTITY, stage=PREFLIGHT_STAGE)
        return {
            **result,
            "root": root.as_posix(),
            "result_sha256": artifacts.sha256_file(root / "result.json"),
            **sealed,
        }
    except Exception as exc:  # noqa: BLE001
        existing = _sealed_terminal_after_exception(
            root,
            identity=PREFLIGHT_IDENTITY,
            stage=PREFLIGHT_STAGE,
            error=exc,
        )
        if existing is not None:
            return existing
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": PREFLIGHT_IDENTITY,
            "stage": PREFLIGHT_STAGE,
            "status": "CRASH_C1S_S1_FAILURE_ATTRIBUTION_PREFLIGHT",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "optimizer_steps": 0,
            "model_writes": 0,
            "authorizes": "nothing",
        }
        artifacts.write_json(root / "result.json", result)
        sealed = _seal(root, identity=PREFLIGHT_IDENTITY, stage=PREFLIGHT_STAGE)
        return {**result, "root": root.as_posix(), **sealed}


def _audit_own_preflight(repo_root: Path) -> dict[str, Any]:
    root = repo_root / contract.PREFLIGHT_ROOT
    lease_path = repo_root / contract.PREFLIGHT_LEASE
    result = artifacts.read_json(root / "result.json")
    seal = artifacts.audit_evidence_seal(
        root,
        expected_identity=PREFLIGHT_IDENTITY,
        expected_stage=PREFLIGHT_STAGE,
        expected_schema_prefix=contract.SCHEMA_PREFIX,
    )
    lease_lines = [
        line
        for line in lease_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if lease_path.is_file() else []
    lease = json.loads(lease_lines[0]) if len(lease_lines) == 1 else {}
    lease_root = lease.get("output_root")
    current_source_identity = _source_identity(_source_hashes(repo_root))
    checks = {
        "identity": result.get("identity") == PREFLIGHT_IDENTITY,
        "stage": result.get("stage") == PREFLIGHT_STAGE,
        "status": result.get("status") == "PASS_C1S_S1_FAILURE_ATTRIBUTION_PREFLIGHT",
        "passed": result.get("passed") is True,
        "authorizes": result.get("authorizes") == "THIS_DIAGNOSIS_SINGLE_USE_ONLY",
        "source_identity_frozen": result.get("source_identity") == current_source_identity,
        "seal": seal.get("passed") is True,
        "lease_single_record": len(lease_lines) == 1,
        "lease_identity": lease.get("identity") == PREFLIGHT_IDENTITY,
        "lease_stage": lease.get("stage") == PREFLIGHT_STAGE,
        "lease_diagnostic_only": lease.get("formal") is False
        and lease.get("diagnostic_only") is True,
        "lease_output_root": isinstance(lease_root, str)
        and Path(lease_root).resolve() == root.resolve(),
    }
    return {
        "root": root.as_posix(),
        "lease": lease_path.as_posix(),
        "checks": checks,
        "seal": seal,
        "lease_record": lease,
        "sealed_source_identity": result.get("source_identity"),
        "current_source_identity": current_source_identity,
        "passed": all(checks.values()),
    }


def _instantiate_endpoint(repo_root: Path, device: torch.device) -> tuple[C1SModel, SuccessorDiagnosticHeads, dict[str, Any]]:
    loaded = load_endpoint_artifact(
        repo_root / contract.S1_ROOT / "endpoint.pt",
        expected_sha256=contract.S1_ENDPOINT_SHA256,
        expected_identity=contract.S1_IDENTITY,
        expected_update=4_000,
        expected_schedule_sha256=contract.S1_SCHEDULE_SHA256,
        expected_model_config=successor_contract.MODEL_CONFIG,
    )
    config = C1SConfig(**dict(loaded["metadata"]["config"]))
    model = C1SModel(config)
    heads = SuccessorDiagnosticHeads(
        config.source_width, config.address_width, config.auxiliary_width
    )
    model.load_state_dict(loaded["model_state"], strict=True)
    heads.load_state_dict(loaded["diagnostic_state"], strict=True)
    model.to(device).eval()
    heads.to(device).eval()
    return model, heads, loaded["metadata"]


def _gate_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recompute the three diagnosed family Gates from sealed S1 evidence."""

    reports = evaluation.get("family_reports")
    if not isinstance(reports, Mapping):
        raise ValueError("sealed S1 evaluation has no family_reports")
    threshold = successor_contract.S1_CONFIG["thresholds"]
    mechanism = successor_contract.MECHANISM_THRESHOLDS
    rows: list[dict[str, Any]] = []
    for family in contract.FAMILIES:
        report = reports.get(family)
        if not isinstance(report, Mapping):
            raise ValueError(f"sealed S1 evaluation lacks family {family}")
        no_core_point = (
            report.get("interventions", {})
            .get("no_core", {})
            .get("answer", {})
            .get("overall", {})
            .get("point")
        )
        contributors = report.get("functional_k", {}).get("causal_contributors", {})
        at_least_two = contributors.get("at_least_two", {}).get("point")
        effective_slots = contributors.get("mean_effective_slots")
        state = report.get("ownership", {}).get("state_masked", {})
        state_families = state.get("families", {})
        registered = state.get("registered_dynamic_features", {})
        names = registered.get(family) if isinstance(registered, Mapping) else None
        feature_rows = state_families.get(family) if isinstance(state_families, Mapping) else None
        dynamic_pass = bool(
            state.get("dynamic_coverage_pass") is True
            and isinstance(names, list)
            and bool(names)
            and isinstance(feature_rows, Mapping)
            and all(
                isinstance(feature_rows.get(name), Mapping)
                and feature_rows[name].get("dynamic_coverage") is True
                and isinstance(feature_rows[name].get("balanced_accuracy"), (int, float))
                and float(feature_rows[name]["balanced_accuracy"])
                >= float(threshold["state_accuracy"])
                for name in names
            )
        )
        rows.extend(
            (
                {
                    "family": family,
                    "gate": "no_core_near_chance",
                    "passed": isinstance(no_core_point, (int, float))
                    and float(no_core_point) <= float(threshold["no_core_accuracy_max"]),
                    "sealed_value": no_core_point,
                },
                {
                    "family": family,
                    "gate": "functional_multi_address",
                    "passed": isinstance(at_least_two, (int, float))
                    and float(at_least_two) >= 0.90
                    and isinstance(effective_slots, (int, float))
                    and float(effective_slots)
                    >= float(mechanism["functional_min_effective_slots"]),
                    "sealed_at_least_two": at_least_two,
                    "sealed_mean_effective_slots": effective_slots,
                },
                {
                    "family": family,
                    "gate": "dynamic_state_each_feature",
                    "passed": dynamic_pass,
                    "sealed_registered_features": names,
                },
            )
        )
    return rows


def _audit_temporal_source_replay(
    targets: list[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild temporal supervision from sealed source ASTs, not bank claims."""

    fields = (
        "family",
        "state_values",
        "state_feature_mask",
        "state_step_mask",
        "state_feature_names",
        "operation_active",
        "source_owner",
        "target_owner",
        "mechanism_supported",
    )
    records: list[dict[str, Any]] = []
    for target in targets:
        example_id = str(target.get("example_id", ""))
        source = sources.get(example_id)
        if not isinstance(source, Mapping):
            raise KeyError(f"source record absent for temporal replay: {example_id}")
        rebuilt = materialize_target(source, tokenizer=None, strict=True)
        mismatched = [field for field in fields if rebuilt.get(field) != target.get(field)]
        records.append(
            {
                "example_id": example_id,
                "family": str(target.get("family", "")).upper(),
                "mismatched_fields": mismatched,
                "passed": not mismatched,
            }
        )
    return {
        "records": records,
        "record_count": len(records),
        "fields": list(fields),
        "source": "sealed_C0R_record_AST_replayed_through_materialize_target",
        "passed": bool(records) and all(row["passed"] for row in records),
    }


def _diagnostic_evidence_audit(
    d001: Mapping[str, Any],
    d002: Mapping[str, Any],
    d003: Mapping[str, Any],
    d004: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on missing registered evidence without requiring a finding."""

    required = set(contract.FAMILIES)

    def classification_complete(
        report: Mapping[str, Any], allowed: set[str], expected_n: Mapping[str, int]
    ) -> bool:
        families = report.get("families", {})
        return bool(
            report.get("family_complete") is True
            and report.get("missing_families") == []
            and set(families) == required
            and all(
                isinstance(families.get(family), Mapping)
                and families[family].get("classification") in allowed
                and families[family].get("n") == int(expected_n[family])
                for family in required
            )
        )

    mutation = d001.get("mutation_audit", {})
    gate_rows = d001.get("sealed_gate_rows", [])
    gate_cells = [
        (str(row.get("family", "")).upper(), str(row.get("gate", "")))
        for row in gate_rows
        if isinstance(row, Mapping)
    ]
    expected_gate_cells = {
        (family, gate)
        for family in required
        for gate in (
            "no_core_near_chance",
            "functional_multi_address",
            "dynamic_state_each_feature",
        )
    }
    d002_families = d002.get("families", {})
    path_definitions = d002.get("path_definitions", {})
    path_names_complete = all(
        isinstance(d002_families.get(family), Mapping)
        and set(d002_families[family].get("paths", {})) == set(PATH_NAMES)
        for family in required
    )
    answer_nulls_valid = all(
        isinstance(d002_families.get(family), Mapping)
        and d002_families[family]
        .get("answer_permutation_metric_null", {})
        .get("control_valid")
        is True
        for family in required
    )
    task_records = d003.get("records", [])
    task_ids = [
        str(row.get("example_id", ""))
        for row in task_records
        if isinstance(row, Mapping)
    ]
    task_family_counts = {
        family: sum(
            isinstance(row, Mapping)
            and str(row.get("family", "")).upper() == family
            for row in task_records
        )
        for family in required
    }
    task_records_structurally_valid = (
        len(task_records) == 32
        and len(task_ids) == 32
        and len(set(task_ids)) == 32
        and "" not in task_ids
        and task_family_counts == {"CPS": 16, "ERE": 16}
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("task"), Mapping)
            and row["task"].get("valid") is True
            and row["task"].get("registered_execution", {}).get("status") == "ASSESSED"
            and row["task"].get("certificate", {}).get("status") == "ASSESSED"
            and row["task"].get("answer_support", {}).get("status")
            in {"ASSESSED", "NOT_ASSESSABLE"}
            and row["task"].get("counterfactual_influence", {}).get("status")
            in {"ASSESSED", "NOT_ASSESSABLE"}
            for row in task_records
        )
    )
    d004_families = d004.get("families", {})
    temporal_cells_complete = True
    for family, features in HARD_FEATURES.items():
        family_report = d004_families.get(family, {})
        observed = family_report.get("features", {}) if isinstance(family_report, Mapping) else {}
        if set(observed) != set(features):
            temporal_cells_complete = False
            continue
        for name in features:
            cell = observed[name]
            if (
                set(cell.get("nulls", {})) != {"time_shuffle", "temporal_mean", "target_shuffle"}
                or int(cell.get("cross_fitted_ridge_record_macro_valid_records", 0)) <= 0
                or int(cell.get("frozen_decoder_record_macro_valid_records", 0)) <= 0
                or any(
                    int(report.get("record_macro_valid_records", 0)) <= 0
                    for report in cell.get("nulls", {}).values()
                )
            ):
                temporal_cells_complete = False
    checks = {
        "d001_family_classification_complete": classification_complete(
            d001.get("classification", {}),
            {
                "OBJECTIVE_GATE_MISMATCH_CONFIRMED",
                "OBJECTIVE_NOT_EQUIVALENT",
                "OBJECTIVE_ALIGNED",
                "INCONCLUSIVE",
            },
            {"CPS": 3, "ERE": 3},
        ),
        "d001_sealed_gate_cells_complete": len(gate_cells) == 6
        and len(set(gate_cells)) == 6
        and set(gate_cells) == expected_gate_cells,
        "d001_read_only_mutation_audit": mutation.get("state_unchanged") is True
        and mutation.get("parameter_grad_fields_empty") is True
        and mutation.get("optimizer_steps") == 0
        and mutation.get("model_writes") == 0,
        "d002_family_classification_complete": classification_complete(
            d002.get("classification", {}),
            set(contract.D005_DECISIONS["axis_A"]),
            {"CPS": 16, "ERE": 16},
        ),
        "d002_registered_paths_complete": path_names_complete
        and set(path_definitions).issuperset(
            {
                "final_delta",
                "h0_relevant_replace",
                "global_mean",
                "copy_all_query_owner",
                "answer_permutation_metric_null",
                "initial_slot_replace_logits",
            }
        ),
        "d002_answer_permutation_controls_valid": answer_nulls_valid,
        "d003_family_classification_complete": classification_complete(
            d003.get("classification", {}),
            set(contract.D005_DECISIONS["axis_B"]),
            {"CPS": 16, "ERE": 16},
        ),
        "d003_task_records_structurally_valid": task_records_structurally_valid,
        "d004_family_classification_complete": classification_complete(
            d004.get("classification", {}),
            set(contract.D005_DECISIONS["axis_C"]),
            {family: len(features) for family, features in HARD_FEATURES.items()},
        ),
        "d004_source_visible_replay": d004.get("source_visible_replay", {}).get("passed")
        is True,
        "d004_registered_record_macro_cells_complete": temporal_cells_complete,
    }
    return {"checks": checks, "passed": all(checks.values())}


def run_diagnosis(repo_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.OUTPUT_ROOT
    lease = repo_root / contract.LEASE_PATH
    if root.exists() or lease.exists():
        return _refusal("fixed diagnosis root or lease already exists", root=root, lease=lease)
    own_preflight = _audit_own_preflight(repo_root)
    if own_preflight.get("passed") is not True:
        return _refusal("diagnosis preflight is not a sealed PASS", root=root, lease=lease)
    inputs = _audit_inputs(repo_root, full_bank_audit=False)
    if inputs.get("passed") is not True:
        return _refusal("pinned S1 inputs no longer pass D000", root=root, lease=lease)
    old_s1_before = artifacts.capture_old_root_snapshot(repo_root / contract.S1_ROOT)
    old_preflight_before = artifacts.capture_old_root_snapshot(repo_root / contract.S1_PREFLIGHT_ROOT)
    # This is the final operation before claiming the single-use lease.  It
    # closes the long cache-audit TOCTOU window after own-preflight replay.
    source_before = _source_hashes(repo_root)
    if _source_identity(source_before) != own_preflight.get("sealed_source_identity"):
        return _refusal(
            "source identity drifted after preflight audit",
            root=root,
            lease=lease,
        )
    artifacts.claim_single_use(
        identity=contract.IDENTITY,
        stage=STAGE,
        output_root=root,
        lease_path=lease,
        formal=False,
    )
    started = time.perf_counter()
    try:
        _snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.contract_manifest())
        artifacts.write_json(root / "preflight-audit.json", own_preflight)
        artifacts.write_json(root / "D000-input-integrity.json", inputs)
        artifacts.write_json(root / "old-s1-snapshot.json", old_s1_before)
        active_device = torch.device(device)
        model, heads, endpoint_metadata = _instantiate_endpoint(repo_root, active_device)
        bank = load_gzip_json(repo_root / contract.S1_PREFLIGHT_ROOT / "target-bank.json.gz")
        ledger = load_json(repo_root / contract.S1_PREFLIGHT_ROOT / "split-ledger.json")
        ids = s1_ids(ledger)
        targets = bank_rows(bank, ids)
        runtime = build_runtime(repo_root / contract.C1_CACHE_ROOT / "hidden", bank)

        d001 = audit_objective_alignment(
            model,
            heads,
            runtime,
            ids,
            device=active_device,
            batch_size=4,
            no_core_hinge_margin=float(
                contract.D001_OBJECTIVE_ALIGNMENT["no_core_training_hinge_margin_drop"]
            ),
        )
        training = artifacts.read_json(repo_root / contract.S1_ROOT / "training.json")
        sealed_evaluation = artifacts.read_json(
            repo_root / contract.S1_ROOT / "evaluation.json"
        )
        logs = training.get("logs", [])
        d001["sealed_training_endpoint"] = logs[-1] if isinstance(logs, list) and logs else None
        d001["sealed_gate_rows"] = _gate_rows(sealed_evaluation)
        d001["classification"] = classify_loss_gate_alignment(
            d001["sealed_gate_rows"],
            d001["loss_rows"],
            contract.DECISION_THRESHOLDS["objective"],
        )
        artifacts.write_json(root / "D001-objective-gate-alignment.json", d001)

        collection = collect_paths(
            model,
            heads,
            runtime,
            ids,
            device=active_device,
            batch_size=4,
        )
        d002 = summarise_paths(
            collection,
            bootstrap_seed=contract.BOOTSTRAP_SEED,
            bootstrap_replicates=contract.BOOTSTRAP_REPLICATES,
        )
        d002["classification"] = classify_a_axis(
            d002["decision_rows"], contract.DECISION_THRESHOLDS["axis_A"]
        )
        artifacts.write_json(root / "D002-h0-core-paths.json", d002)

        sources = load_source_records(contract.C0R_DATA_ROOT if contract.C0R_DATA_ROOT.is_absolute() else repo_root / contract.C0R_DATA_ROOT, ids)
        d003 = audit_task_and_model_arity(
            targets,
            sources,
            collection,
            contributor_ratio=float(successor_contract.MECHANISM_THRESHOLDS["functional_contributor_ratio"]),
            eligible_margin=float(contract.D003_ARITY["eligible_baseline_margin"]),
        )
        d003["classification"] = classify_b_axis(
            d003["decision_rows"], contract.DECISION_THRESHOLDS["axis_B"]
        )
        artifacts.write_json(root / "D003-task-model-arity.json", d003)

        adequacy = contract.D004_TEMPORAL["target_adequacy"]
        decoder = contract.D004_TEMPORAL["decoder"]
        temporal_source_replay = _audit_temporal_source_replay(targets, sources)
        d004 = audit_temporal_readout(
            collection,
            ranks=decoder["rank_grid"],
            regularizations=decoder["lambda_grid"],
            seed=contract.BOOTSTRAP_SEED,
            target_positive_min=int(adequacy["positive_per_family_feature"]),
            target_negative_min=int(adequacy["negative_per_family_feature"]),
            target_changes_min=int(adequacy["cross_step_changes"]),
            source_visible_replay=temporal_source_replay.get("passed") is True,
        )
        d004["source_visible_replay"] = temporal_source_replay
        d004["classification"] = classify_c_axis(
            d004["decision_rows"], contract.DECISION_THRESHOLDS["axis_C"]
        )
        artifacts.write_json(root / "D004-temporal-latent-readout.json", d004)

        d005 = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.decisions.v1",
            "identity": contract.IDENTITY,
            "family_primary": True,
            "overall_summary_only": True,
            "objective_gate_alignment": d001["classification"],
            "axis_A_h0_core": d002["classification"],
            "axis_B_task_model_arity": d003["classification"],
            "axis_C_temporal": d004["classification"],
            "composite_pass": False,
            "authorizes": "nothing",
        }
        evidence_completeness = _diagnostic_evidence_audit(d001, d002, d003, d004)
        d005["diagnostic_evidence_completeness"] = evidence_completeness
        artifacts.write_json(root / "D005-decisions.json", d005)

        # Re-read every immutable external input after measurement.  The
        # diagnosis may be long enough for a cache/source drift to occur; a
        # launch-time replay alone is not a terminal integrity proof.
        terminal_inputs = _audit_inputs(repo_root, full_bank_audit=False)
        source_after = _source_hashes(repo_root)
        old_s1_unchanged = artifacts.audit_old_root_unchanged(
            repo_root / contract.S1_ROOT, old_s1_before
        )
        old_preflight_unchanged = artifacts.audit_old_root_unchanged(
            repo_root / contract.S1_PREFLIGHT_ROOT, old_preflight_before
        )
        forbidden_model_files = [
            path.relative_to(root).as_posix()
            for suffix in ("*.pt", "*.pth", "*.ckpt")
            for path in root.rglob(suffix)
        ]
        integrity = {
            "source_stable": source_before == source_after,
            "old_s1_unchanged": old_s1_unchanged,
            "old_s1_preflight_unchanged": old_preflight_unchanged,
            "forbidden_model_files": forbidden_model_files,
            "optimizer_steps": 0,
            "model_writes": 0,
            "endpoint_metadata": endpoint_metadata,
            "diagnostic_evidence_completeness": evidence_completeness,
            "terminal_input_replay": terminal_inputs,
        }
        artifacts.write_json(root / "integrity-final.json", integrity)
        complete = bool(
            source_before == source_after
            and old_s1_unchanged.get("passed") is True
            and old_preflight_unchanged.get("passed") is True
            and not forbidden_model_files
            and d001["mutation_audit"]["state_unchanged"]
            and d001["mutation_audit"]["parameter_grad_fields_empty"]
            and evidence_completeness.get("passed") is True
            and terminal_inputs.get("passed") is True
        )
        wall = time.perf_counter() - started
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.result.v1",
            "identity": contract.IDENTITY,
            "stage": STAGE,
            "status": "COMPLETE_V2_A_C1S_S1_FAILURE_ATTRIBUTION" if complete else "INCOMPLETE_V2_A_C1S_S1_FAILURE_ATTRIBUTION",
            "passed": complete,
            "diagnostic_only": True,
            "formal": False,
            "source_identity": _source_identity(source_after),
            "source_stable": source_before == source_after,
            "optimizer_steps": 0,
            "model_writes": 0,
            "checkpoint_selection": False,
            "families": list(contract.FAMILIES),
            "decisions": d005,
            "wall_seconds": wall,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(active_device)) if active_device.type == "cuda" else 0,
            "authorizes": "nothing",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
            "v2a_passed": False,
        }
        artifacts.write_json(root / "result.json", result)
        sealed = _seal(root, identity=contract.IDENTITY, stage=STAGE)
        return {
            **result,
            "root": root.as_posix(),
            "result_sha256": artifacts.sha256_file(root / "result.json"),
            **sealed,
        }
    except Exception as exc:  # noqa: BLE001
        existing = _sealed_terminal_after_exception(
            root,
            identity=contract.IDENTITY,
            stage=STAGE,
            error=exc,
        )
        if existing is not None:
            return existing
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.result.v1",
            "identity": contract.IDENTITY,
            "stage": STAGE,
            "status": "CRASH_V2_A_C1S_S1_FAILURE_ATTRIBUTION",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "optimizer_steps": 0,
            "model_writes": 0,
            "authorizes": "nothing",
            "v2a_passed": False,
        }
        artifacts.write_json(root / "result.json", result)
        sealed = _seal(root, identity=contract.IDENTITY, stage=STAGE)
        return {**result, "root": root.as_posix(), **sealed}


__all__ = [
    "PREFLIGHT_IDENTITY",
    "PREFLIGHT_STAGE",
    "SOURCE_FILES",
    "STAGE",
    "run_diagnosis",
    "run_preflight",
]
