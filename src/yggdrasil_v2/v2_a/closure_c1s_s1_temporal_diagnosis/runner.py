from __future__ import annotations

"""Single-use runner for the D004R/D005R temporal successor diagnosis."""

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
from ..closure_c1s_successor import contract as successor_contract
from ..closure_c1s_successor.objective import SuccessorDiagnosticHeads
from ..closure_c1s_s1_diagnosis import artifacts
from ..closure_c1s_s1_diagnosis import contract as predecessor_contract
from ..closure_c1s_s1_diagnosis import runner as predecessor_runner
from ..closure_c1s_s1_diagnosis.arity_analysis import load_source_records
from ..closure_c1s_s1_diagnosis.endpoint import load_endpoint_artifact
from ..closure_c1s_s1_diagnosis.runtime import (
    bank_rows,
    build_runtime,
    load_gzip_json,
    load_json,
    s1_ids,
)
from . import contract
from .collect import collect_temporal
from .decision import classify_axis_c
from .folds import audit_fold_ledger, build_fold_ledger
from .temporal import audit_temporal_readout


STAGE = "S1-TEMPORAL-ATTRIBUTION-REPAIR"
PREFLIGHT_STAGE = f"{STAGE}-PREFLIGHT"
PREFLIGHT_IDENTITY = f"{contract.IDENTITY}-PREFLIGHT"
PREFLIGHT_PASS_STATUS = "PASS_C1S_S1_TEMPORAL_ATTRIBUTION_REPAIR_PREFLIGHT"

NEW_SOURCE_FILES = (
    contract.DESIGN_DOC,
    Path("experiments/v2_a_closure_c1s_s1_temporal_diagnosis.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_temporal_diagnosis/__init__.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_temporal_diagnosis/collect.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_temporal_diagnosis/contract.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_temporal_diagnosis/decision.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_temporal_diagnosis/folds.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_temporal_diagnosis/runner.py"),
    Path("src/yggdrasil_v2/v2_a/closure_c1s_s1_temporal_diagnosis/temporal.py"),
    Path("tests/v2_a_closure_c1s_s1_temporal_diagnosis/test_collect.py"),
    Path("tests/v2_a_closure_c1s_s1_temporal_diagnosis/test_contract.py"),
    Path("tests/v2_a_closure_c1s_s1_temporal_diagnosis/test_decision.py"),
    Path("tests/v2_a_closure_c1s_s1_temporal_diagnosis/test_folds.py"),
    Path("tests/v2_a_closure_c1s_s1_temporal_diagnosis/test_runner.py"),
    Path("tests/v2_a_closure_c1s_s1_temporal_diagnosis/test_temporal.py"),
)
SOURCE_FILES = tuple(dict.fromkeys((*predecessor_runner.SOURCE_FILES, *NEW_SOURCE_FILES)))


def _source_hashes(repo_root: Path) -> dict[str, str]:
    missing = [path.as_posix() for path in SOURCE_FILES if not (repo_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"temporal diagnosis source identity is incomplete: {missing}")
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


def _run_tests(repo_root: Path) -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "pytest", "-q", path]
        for path in (
            "tests/v2_a_closure_c1s_s1_temporal_diagnosis",
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
    return {"runs": runs, "passed": all(run["passed"] for run in runs)}


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
    root: Path, *, identity: str, stage: str, error: Exception
) -> dict[str, Any] | None:
    result_path = root / "result.json"
    seal_path = root / "evidence-seal.json"
    if not result_path.is_file() or not seal_path.is_file():
        return None
    existing = artifacts.read_json(result_path)
    replay = artifacts.audit_evidence_seal(
        root,
        expected_identity=identity,
        expected_stage=stage,
        expected_schema_prefix=contract.SCHEMA_PREFIX,
    )
    return {
        **existing,
        "root": root.as_posix(),
        "post_seal_exception_type": type(error).__name__,
        "post_seal_exception": str(error),
        "seal_sha256": artifacts.sha256_file(seal_path),
        "seal_replay": replay,
        "authorizes": "nothing",
    }


def _audit_predecessor_diagnosis(repo_root: Path) -> dict[str, Any]:
    root = repo_root / contract.PREDECESSOR_DIAGNOSIS_ROOT
    result = artifacts.read_json(root / "result.json")
    file_pins = artifacts.audit_sealed_file_pins(
        root,
        expected_identity=contract.PREDECESSOR_IDENTITY,
        expected_stage=contract.PREDECESSOR_STAGE,
        expected_files=contract.PREDECESSOR_PARTIAL_FILES,
        expected_schema_prefix=predecessor_contract.SCHEMA_PREFIX,
    )
    preflight_root = repo_root / contract.PREDECESSOR_PREFLIGHT_ROOT
    preflight_result = artifacts.read_json(preflight_root / "result.json")
    preflight_seal = artifacts.audit_evidence_seal(
        preflight_root,
        expected_identity=contract.PREDECESSOR_PREFLIGHT_IDENTITY,
        expected_stage=contract.PREDECESSOR_PREFLIGHT_STAGE,
        expected_schema_prefix=predecessor_contract.SCHEMA_PREFIX,
    )
    checks = {
        "result_sha256": artifacts.sha256_file(root / "result.json")
        == contract.PREDECESSOR_RESULT_SHA256,
        "seal_sha256": artifacts.sha256_file(root / "evidence-seal.json")
        == contract.PREDECESSOR_SEAL_SHA256,
        "sealed_partial_files": file_pins.get("passed") is True,
        "identity": result.get("identity") == contract.PREDECESSOR_IDENTITY,
        "status": result.get("status") == contract.PREDECESSOR_STATUS,
        "failed": result.get("passed") is False,
        "exact_crash": result.get("error")
        == "inner readout fold has no record with both target classes",
        "read_only": result.get("optimizer_steps") == 0
        and result.get("model_writes") == 0,
        "authorizes_nothing": result.get("authorizes") == "nothing",
        "d004_absent": not (root / "D004-temporal-latent-readout.json").exists(),
        "d005_absent": not (root / "D005-decisions.json").exists(),
        "preflight_result_sha256": artifacts.sha256_file(preflight_root / "result.json")
        == contract.PREDECESSOR_PREFLIGHT_RESULT_SHA256,
        "preflight_seal_sha256": artifacts.sha256_file(
            preflight_root / "evidence-seal.json"
        )
        == contract.PREDECESSOR_PREFLIGHT_SEAL_SHA256,
        "preflight_seal_replay": preflight_seal.get("passed") is True,
        "preflight_identity": preflight_result.get("identity")
        == contract.PREDECESSOR_PREFLIGHT_IDENTITY,
        "preflight_stage": preflight_result.get("stage")
        == contract.PREDECESSOR_PREFLIGHT_STAGE,
        "preflight_status": preflight_result.get("status")
        == "PASS_C1S_S1_FAILURE_ATTRIBUTION_PREFLIGHT",
        "preflight_was_pass": preflight_result.get("passed") is True,
        "preflight_authorization_was_consumed_scope": preflight_result.get("authorizes")
        == "THIS_DIAGNOSIS_SINGLE_USE_ONLY",
    }
    return {
        "root": root.as_posix(),
        "preflight_root": preflight_root.as_posix(),
        "checks": checks,
        "sealed_file_pins": file_pins,
        "preflight_seal": preflight_seal,
        "passed": all(checks.values()),
    }


def _audit_inputs(repo_root: Path, *, full_bank_audit: bool) -> dict[str, Any]:
    original_inputs = predecessor_runner._audit_inputs(
        repo_root, full_bank_audit=full_bank_audit
    )
    predecessor = _audit_predecessor_diagnosis(repo_root)
    checks = {
        "original_s1_inputs": original_inputs.get("passed") is True,
        "sealed_crashed_diagnosis": predecessor.get("passed") is True,
    }
    return {
        "original_s1_inputs": original_inputs,
        "predecessor_diagnosis": predecessor,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _instantiate_endpoint(
    repo_root: Path, device: torch.device
) -> tuple[C1SModel, SuccessorDiagnosticHeads, dict[str, Any]]:
    loaded = load_endpoint_artifact(
        repo_root / predecessor_contract.S1_ROOT / "endpoint.pt",
        expected_sha256=predecessor_contract.S1_ENDPOINT_SHA256,
        expected_identity=predecessor_contract.S1_IDENTITY,
        expected_update=4_000,
        expected_schedule_sha256=predecessor_contract.S1_SCHEDULE_SHA256,
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


def _temporal_source_replay(
    targets: list[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    report = predecessor_runner._audit_temporal_source_replay(targets, sources)
    return {
        **report,
        "source": "source_ast_materialize_target_regeneration_same_implementation",
        "boundary": "not an independent semantic oracle",
    }


def _predecessor_partial_context(repo_root: Path) -> dict[str, Any]:
    root = repo_root / contract.PREDECESSOR_DIAGNOSIS_ROOT
    rows: dict[str, Any] = {}
    for name in (
        "D001-objective-gate-alignment.json",
        "D002-h0-core-paths.json",
        "D003-task-model-arity.json",
    ):
        payload = artifacts.read_json(root / name)
        rows[name] = {
            "sha256": artifacts.sha256_file(root / name),
            "classification": payload.get("classification"),
        }
    return {
        "source_identity": contract.PREDECESSOR_IDENTITY,
        "source_terminal_status": contract.PREDECESSOR_STATUS,
        "role": "sealed_predecessor_partial_context_only_not_recomputed",
        "files": rows,
        "predecessor_had_no_D004_or_D005_conclusion": True,
    }


def _fold_completeness(d004: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    reports = d004.get("families", {})
    for family, features in contract.HARD_FEATURES.items():
        observed = reports.get(family, {}).get("features", {})
        checks[f"{family}.feature_set"] = set(observed) == set(features)
        for feature in features:
            cell = ledger["cells"][family][feature]
            report = observed.get(feature, {}) if isinstance(observed, Mapping) else {}
            eligible = int(cell["eligible_records"])
            checks[f"{family}.{feature}.target"] = report.get("target_adequate") in {
                True,
                False,
            }
            checks[f"{family}.{feature}.coverage"] = bool(
                cell.get("passed") is True
                and len(cell.get("outer_cells", [])) == contract.OUTER_FOLDS
                and all(
                    len(outer.get("inner_cells", [])) == contract.INNER_FOLDS
                    and outer.get("passed") is True
                    for outer in cell.get("outer_cells", [])
                )
            )
            checks[f"{family}.{feature}.natural"] = (
                report.get("natural", {}).get("record_macro_valid_records") == eligible
                and report.get("natural", {}).get("record_macro_total_records") == eligible
                and len(report.get("natural", {}).get("record_evidence", [])) == eligible
            )
            checks[f"{family}.{feature}.frozen"] = (
                report.get("frozen_decoder", {}).get("record_macro_valid_records")
                == eligible
                and report.get("frozen_decoder", {}).get("record_macro_total_records")
                == eligible
                and len(report.get("frozen_decoder", {}).get("record_evidence", []))
                == eligible
            )
            checks[f"{family}.{feature}.fixed_channel"] = (
                report.get("fixed_channel", {}).get("record_macro_valid_records")
                == eligible
                and report.get("fixed_channel", {}).get("record_macro_total_records")
                == eligible
                and len(report.get("fixed_channel", {}).get("record_evidence", []))
                == eligible
            )
            fit_nulls = report.get("fit_nulls", {})
            checks[f"{family}.{feature}.fit_nulls"] = set(fit_nulls) == {
                "time_shuffle",
                "temporal_mean",
                "within_record_target_shuffle",
            } and all(
                cell_report.get("record_macro_valid_records") == eligible
                and cell_report.get("record_macro_total_records") == eligible
                and len(cell_report.get("record_evidence", [])) == eligible
                for cell_report in fit_nulls.values()
            )
            checks[f"{family}.{feature}.score_null"] = (
                report.get("score_null", {}).get("replicates")
                == contract.SCORE_NULL_REPLICATES
            )
    return {"checks": checks, "passed": bool(checks) and all(checks.values())}


def run_preflight(repo_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.PREFLIGHT_ROOT
    lease = repo_root / contract.PREFLIGHT_LEASE
    if root.exists() or lease.exists():
        return _refusal("fixed temporal preflight root or lease already exists", root=root, lease=lease)
    artifacts.claim_single_use(
        identity=PREFLIGHT_IDENTITY,
        stage=PREFLIGHT_STAGE,
        output_root=root,
        lease_path=lease,
        formal=False,
    )
    started = time.perf_counter()
    try:
        source_hashes = _source_hashes(repo_root)
        source_identity = _source_identity(source_hashes)
        _snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.contract_manifest())
        artifacts.write_json(
            root / "source-hashes.json",
            {"identity": source_identity, "files": source_hashes},
        )
        # Freeze the target-only fold ledger before any endpoint-oriented
        # audit or test process can materialize a real latent/prediction.
        bank = load_gzip_json(
            repo_root / predecessor_contract.S1_PREFLIGHT_ROOT / "target-bank.json.gz"
        )
        split = load_json(
            repo_root / predecessor_contract.S1_PREFLIGHT_ROOT / "split-ledger.json"
        )
        ids = s1_ids(split)
        targets = bank_rows(bank, ids)
        fold_ledger = build_fold_ledger(targets)
        artifacts.write_json(root / "temporal-fold-ledger.json", fold_ledger)
        fold_audit = audit_fold_ledger(targets, fold_ledger)
        artifacts.write_json(root / "temporal-fold-audit.json", fold_audit)

        sources = load_source_records(repo_root / predecessor_contract.C0R_DATA_ROOT, ids)
        replay = _temporal_source_replay(targets, sources)
        artifacts.write_json(root / "temporal-source-replay.json", replay)

        tests = _run_tests(repo_root)
        artifacts.write_json(root / "tests.json", tests)
        inputs = _audit_inputs(repo_root, full_bank_audit=True)
        artifacts.write_json(root / "D000R-input-integrity.json", inputs)
        device_report = predecessor_runner._device_audit(device)
        artifacts.write_json(root / "device.json", device_report)
        checks = {
            "source_complete": bool(source_hashes),
            "tests": tests.get("passed") is True,
            "target_only_fold_ledger": fold_audit.get("passed") is True,
            "temporal_source_replay": replay.get("passed") is True,
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
            "status": PREFLIGHT_PASS_STATUS
            if passed
            else "FAIL_C1S_S1_TEMPORAL_ATTRIBUTION_REPAIR_PREFLIGHT",
            "passed": passed,
            "diagnostic_only": True,
            "checks": checks,
            "source_identity": source_identity,
            "fold_ledger_sha256": artifacts.sha256_file(
                root / "temporal-fold-ledger.json"
            ),
            "optimizer_steps": 0,
            "model_writes": 0,
            "wall_seconds": time.perf_counter() - started,
            "authorizes": contract.IDENTITY if passed else "nothing",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
            "v2a_passed": False,
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
            root, identity=PREFLIGHT_IDENTITY, stage=PREFLIGHT_STAGE, error=exc
        )
        if existing is not None:
            return existing
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": PREFLIGHT_IDENTITY,
            "stage": PREFLIGHT_STAGE,
            "status": "CRASH_C1S_S1_TEMPORAL_ATTRIBUTION_REPAIR_PREFLIGHT",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "optimizer_steps": 0,
            "model_writes": 0,
            "authorizes": "nothing",
            "v2a_passed": False,
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
    lease_lines = (
        [line for line in lease_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lease_path.is_file()
        else []
    )
    lease = json.loads(lease_lines[0]) if len(lease_lines) == 1 else {}
    current_source = _source_identity(_source_hashes(repo_root))
    fold_path = root / "temporal-fold-ledger.json"
    fold_sha = artifacts.sha256_file(fold_path) if fold_path.is_file() else None
    bank = load_gzip_json(
        repo_root / predecessor_contract.S1_PREFLIGHT_ROOT / "target-bank.json.gz"
    )
    split = load_json(
        repo_root / predecessor_contract.S1_PREFLIGHT_ROOT / "split-ledger.json"
    )
    targets = bank_rows(bank, s1_ids(split))
    ledger = artifacts.read_json(fold_path)
    fold_audit = audit_fold_ledger(targets, ledger)
    lease_root = lease.get("output_root")
    checks = {
        "identity": result.get("identity") == PREFLIGHT_IDENTITY,
        "stage": result.get("stage") == PREFLIGHT_STAGE,
        "status": result.get("status") == PREFLIGHT_PASS_STATUS,
        "passed": result.get("passed") is True,
        "exact_authorization": result.get("authorizes") == contract.IDENTITY,
        "source_identity_frozen": result.get("source_identity") == current_source,
        "fold_ledger_sha256": result.get("fold_ledger_sha256") == fold_sha,
        "fold_ledger_exact_regeneration": fold_audit.get("passed") is True,
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
        "fold_audit": fold_audit,
        "sealed_source_identity": result.get("source_identity"),
        "current_source_identity": current_source,
        "fold_ledger_sha256": fold_sha,
        "passed": all(checks.values()),
    }


def run_diagnosis(repo_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.OUTPUT_ROOT
    lease = repo_root / contract.LEASE_PATH
    if root.exists() or lease.exists():
        return _refusal("fixed temporal diagnosis root or lease already exists", root=root, lease=lease)
    own_preflight = _audit_own_preflight(repo_root)
    if own_preflight.get("passed") is not True:
        return _refusal("temporal diagnosis preflight is not a sealed PASS", root=root, lease=lease)
    inputs = _audit_inputs(repo_root, full_bank_audit=False)
    if inputs.get("passed") is not True:
        return _refusal("pinned temporal diagnosis inputs no longer pass D000R", root=root, lease=lease)

    watched_roots = {
        "s1": repo_root / predecessor_contract.S1_ROOT,
        "s1_preflight": repo_root / predecessor_contract.S1_PREFLIGHT_ROOT,
        "crashed_diagnosis": repo_root / contract.PREDECESSOR_DIAGNOSIS_ROOT,
        "crashed_diagnosis_preflight": repo_root / contract.PREDECESSOR_PREFLIGHT_ROOT,
        "own_preflight": repo_root / contract.PREFLIGHT_ROOT,
    }
    before_snapshots = {
        name: artifacts.capture_old_root_snapshot(path)
        for name, path in watched_roots.items()
    }
    source_before = _source_hashes(repo_root)
    if _source_identity(source_before) != own_preflight.get("sealed_source_identity"):
        return _refusal("source identity drifted after preflight", root=root, lease=lease)
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
        artifacts.write_json(root / "D000R-input-integrity.json", inputs)
        artifacts.write_json(root / "predecessor-snapshots.json", before_snapshots)

        fold_ledger = artifacts.read_json(
            repo_root / contract.PREFLIGHT_ROOT / "temporal-fold-ledger.json"
        )
        artifacts.write_json(root / "temporal-fold-ledger.json", fold_ledger)
        bank = load_gzip_json(
            repo_root / predecessor_contract.S1_PREFLIGHT_ROOT / "target-bank.json.gz"
        )
        split = load_json(
            repo_root / predecessor_contract.S1_PREFLIGHT_ROOT / "split-ledger.json"
        )
        ids = s1_ids(split)
        targets = bank_rows(bank, ids)
        fold_audit = audit_fold_ledger(targets, fold_ledger)
        if fold_audit.get("passed") is not True:
            raise RuntimeError("sealed temporal fold ledger no longer regenerates exactly")
        artifacts.write_json(root / "temporal-fold-audit.json", fold_audit)

        active_device = torch.device(device)
        model, heads, endpoint_metadata = _instantiate_endpoint(repo_root, active_device)
        runtime = build_runtime(repo_root / predecessor_contract.C1_CACHE_ROOT / "hidden", bank)
        collection = collect_temporal(
            model,
            heads,
            runtime,
            ids,
            device=active_device,
            batch_size=4,
        )
        sources = load_source_records(repo_root / predecessor_contract.C0R_DATA_ROOT, ids)
        replay = _temporal_source_replay(targets, sources)
        decoder = contract.TEMPORAL_READOUT["decoder"]
        d004 = audit_temporal_readout(
            collection,
            fold_ledger,
            ranks=decoder["rank_grid"],
            regularizations=decoder["lambda_grid"],
            fit_null_seed=contract.FOLD_SEED,
            score_null_seed=contract.SCORE_NULL_SEED,
            score_null_replicates=contract.SCORE_NULL_REPLICATES,
            source_visible_replay=replay.get("passed") is True,
        )
        d004["source_visible_replay"]["records"] = replay.get("records", [])
        d004["classification"] = classify_axis_c(d004["decision_rows"])
        d004["mutation_audit"] = collection["mutation_audit"]
        artifacts.write_json(root / "D004R-temporal-latent-readout.json", d004)

        context = _predecessor_partial_context(repo_root)
        completeness = _fold_completeness(d004, fold_ledger)
        classification = d004["classification"]
        d005 = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.D005R.v1",
            "identity": contract.IDENTITY,
            "scope": "D004R_D005R_only",
            "sealed_predecessor_partial_context": context,
            "axis_C_temporal": classification,
            "family_primary": True,
            "overall_summary_only": True,
            "diagnostic_evidence_completeness": completeness,
            "composite_pass": False,
            "authorizes": "nothing",
            "v2a_passed": False,
        }
        artifacts.write_json(root / "D005R-decisions.json", d005)

        terminal_inputs = _audit_inputs(repo_root, full_bank_audit=False)
        source_after = _source_hashes(repo_root)
        unchanged = {
            name: artifacts.audit_old_root_unchanged(path, before_snapshots[name])
            for name, path in watched_roots.items()
        }
        forbidden_model_files = [
            path.relative_to(root).as_posix()
            for suffix in ("*.pt", "*.pth", "*.ckpt")
            for path in root.rglob(suffix)
        ]
        integrity = {
            "source_stable": source_before == source_after,
            "watched_roots_unchanged": unchanged,
            "terminal_input_replay": terminal_inputs,
            "fold_ledger_exact_regeneration": fold_audit,
            "endpoint_metadata": endpoint_metadata,
            "mutation_audit": collection["mutation_audit"],
            "forbidden_model_files": forbidden_model_files,
            "optimizer_steps": 0,
            "model_writes": 0,
        }
        artifacts.write_json(root / "integrity-final.json", integrity)
        complete = bool(
            source_before == source_after
            and all(report.get("passed") is True for report in unchanged.values())
            and terminal_inputs.get("passed") is True
            and fold_audit.get("passed") is True
            and replay.get("passed") is True
            and completeness.get("passed") is True
            and classification.get("family_complete") is True
            and collection["mutation_audit"].get("state_unchanged") is True
            and collection["mutation_audit"].get("parameter_grad_fields_empty") is True
            and not forbidden_model_files
        )
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.result.v1",
            "identity": contract.IDENTITY,
            "stage": STAGE,
            "status": "COMPLETE_C1S_S1_TEMPORAL_ATTRIBUTION_REPAIR"
            if complete
            else "INCOMPLETE_C1S_S1_TEMPORAL_ATTRIBUTION_REPAIR",
            "passed": complete,
            "diagnostic_only": True,
            "formal": False,
            "source_identity": _source_identity(source_after),
            "source_stable": source_before == source_after,
            "optimizer_steps": 0,
            "model_writes": 0,
            "checkpoint_selection": False,
            "axis_C_temporal": classification,
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(active_device))
            if active_device.type == "cuda"
            else 0,
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
            root, identity=contract.IDENTITY, stage=STAGE, error=exc
        )
        if existing is not None:
            return existing
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.result.v1",
            "identity": contract.IDENTITY,
            "stage": STAGE,
            "status": "CRASH_C1S_S1_TEMPORAL_ATTRIBUTION_REPAIR",
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
    "PREFLIGHT_PASS_STATUS",
    "PREFLIGHT_STAGE",
    "SOURCE_FILES",
    "STAGE",
    "run_diagnosis",
    "run_preflight",
]
