from __future__ import annotations

"""Fail-closed preflight and single-use formal runner for C1U S2."""

from contextlib import AbstractContextManager
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import torch

from yggdrasil_v2.v2_a.closure_c1u import contract as c1u_contract
from yggdrasil_v2.v2_a.closure_c1u.runtime import build_independent_card_cache
from yggdrasil_v2.v2_a.closure_c1u.s1 import audit_s1_root
from yggdrasil_v2.v2_a.closure_c1u.source import (
    QwenCardEncoder,
    audit_device,
    audit_source_assets,
)
from yggdrasil_v2.v2_a.closure_c1u.train import set_determinism

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
from .cache import audit_cache, load_cache, persist_cache
from .evaluate import collect_rows, qualify
from .model import (
    S2Config,
    S2Model,
    matched_parameter_audit,
    state_tensor_sha256,
)
from .runtime import BoundedBatchProvider, S2Runtime
from .tasks import audit_multibank, build_multibank, split_records
from .train import (
    TrainSpec,
    build_schedule,
    load_endpoint,
    save_endpoint,
    schedule_report,
    train_fixed_endpoint,
)


AuditFn = Callable[..., Mapping[str, Any]]


def _s1_records(repo_root: Path) -> list[dict[str, Any]]:
    payload = read_json(Path(repo_root) / contract.S1_ROOT / "task-bank.json")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != c1u_contract.S1_RECORDS:
        raise ValueError("sealed S1 task bank has the wrong cardinality")
    return [dict(row) for row in records]


def audit_s1_predecessor(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    try:
        root = repo_root / contract.S1_ROOT
        result = read_json(root / "result.json")
        upstream = audit_s1_root(repo_root)
        endpoint = root / str(result.get("endpoint", {}).get("path", ""))
        checks = {
            "upstream_full_audit": upstream.get("passed") is True,
            "lease": (repo_root / contract.S1_LEASE).is_file(),
            "identity": result.get("identity") == contract.S1_IDENTITY,
            "status": result.get("status") == contract.S1_STATUS,
            "passed": result.get("passed") is True,
            "authorization": result.get("authorizes") == contract.S1_AUTHORIZATION,
            "result_sha256": sha256_file(root / "result.json")
            == contract.S1_RESULT_SHA256,
            "seal_sha256": sha256_file(root / "evidence-seal.json")
            == contract.S1_EVIDENCE_SEAL_SHA256,
            "endpoint_sha256": endpoint.is_file()
            and sha256_file(endpoint) == contract.S1_ENDPOINT_SHA256,
            "source_identity": result.get("source_identity")
            == contract.S1_SOURCE_IDENTITY,
            "formal_steps": result.get("formal_optimizer_steps")
            == c1u_contract.S1_MAXIMUM_UPDATES,
            "disposable_steps": result.get("disposable_optimizer_steps") == 0,
            "model_writes": result.get("model_writes") == 1,
            "checkpoint_selection": result.get("checkpoint_selection") is False,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "result": result,
            "upstream": upstream,
            "result_sha256": sha256_file(root / "result.json"),
            "seal_sha256": sha256_file(root / "evidence-seal.json"),
            "endpoint_sha256": sha256_file(endpoint) if endpoint.is_file() else None,
            "records": _s1_records(repo_root),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def audit_unconsumed_paths(repo_root: Path, *, stage: str) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    if stage == "preflight":
        paths = (
            contract.PREFLIGHT_ROOT,
            contract.PREFLIGHT_LEASE,
            contract.ROOT,
            contract.LEASE,
        )
    elif stage == "s2":
        paths = (contract.ROOT, contract.LEASE)
    else:
        raise ValueError(f"unknown S2 stage: {stage}")
    observed = {path.as_posix(): (repo_root / path).exists() for path in paths}
    return {"passed": not any(observed.values()), "observed": observed}


def run_registered_tests(repo_root: Path) -> dict[str, Any]:
    # The two closure directories intentionally contain repeated test module
    # basenames.  Independent pytest processes avoid import-file mismatch and
    # also keep predecessor and successor evidence attributable.
    runs = []
    for test_root in ("tests/v2_a_closure_c1u", "tests/v2_a_closure_c1u_s2"):
        command = [sys.executable, "-m", "pytest", "-q", test_root]
        completed = subprocess.run(
            command,
            cwd=Path(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        runs.append(
            {
                "test_root": test_root,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "passed": completed.returncode == 0,
            }
        )
    return {
        "passed": all(run["passed"] for run in runs),
        "runs": runs,
    }


def _config(source_width: int, arm: str) -> S2Config:
    return S2Config(
        source_width=int(source_width),
        workspace_slots=int(contract.WORKSPACE_SLOTS[arm]),
    )


def _paired_models(source_width: int, *, seed: int) -> tuple[S2Model, S2Model]:
    set_determinism(seed)
    k1 = S2Model(_config(source_width, "K1"))
    set_determinism(seed)
    k8 = S2Model(_config(source_width, "K8"))
    return k1, k8


def _schedule_bundle(records: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    result = {}
    for fold in contract.FOLDS:
        fold_id = str(fold["fold_id"])
        train, _ = split_records(records, fold_id)
        report = schedule_report(
            build_schedule(
                train,
                fold_id=fold_id,
                seed=seed,
                maximum_updates=contract.MAXIMUM_UPDATES_PER_ENDPOINT,
            )
        )
        result[fold_id] = report
    return result


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)


def _structural_controls(
    runtime: S2Runtime,
    records: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    active_device = torch.device(device)
    train, _ = split_records(records, "F0")
    scheduled = build_schedule(
        train, fold_id="F0", seed=contract.PREFLIGHT_ORDER_SEED, maximum_updates=1
    )[0]
    batch = _move_batch(runtime.get_batch(scheduled.example_ids), active_device)
    k1, k8 = _paired_models(runtime.cache.source_width, seed=contract.MODEL_SEED)
    parity = matched_parameter_audit(k1, k8)
    reports = {}
    for arm, model in (("K1", k1), ("K8", k8)):
        model.to(active_device).eval()
        with _autocast(active_device):
            reports[arm] = collect_rows(model, batch, fold_id="F0", arm=arm)
        model.to("cpu")
    k1_swap = max(
        value
        for row in reports["K1"]["rows"]
        for value in row["owner_swap_logit_max_abs"]
    )
    k8_swap = max(
        value
        for row in reports["K8"]["rows"]
        for value in row["owner_swap_logit_l2"]
    )
    checks = {
        "matched_parameters_and_initialization": parity["passed"],
        "k1_owner_swap_null": k1_swap
        <= float(contract.GATES["k1_owner_swap_logit_max_abs"]),
        "k8_address_binding_nonzero": k8_swap
        >= float(contract.GATES["k8_owner_swap_logit_l2_floor"]),
        "k1_permutation": reports["K1"]["permutation_max_abs"]
        <= float(contract.GATES["permutation_logit_max_abs"]),
        "k8_permutation": reports["K8"]["permutation_max_abs"]
        <= float(contract.GATES["permutation_logit_max_abs"]),
    }
    del k1, k8, batch
    gc.collect()
    if active_device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parity": parity,
        "k1_owner_swap_max_abs": k1_swap,
        "k8_owner_swap_max_l2": k8_swap,
        "reports": reports,
    }


def inspect(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    predecessor = audit_s1_predecessor(repo_root)
    paths = {
        "preflight_root": (repo_root / contract.PREFLIGHT_ROOT).exists(),
        "preflight_lease": (repo_root / contract.PREFLIGHT_LEASE).exists(),
        "s2_root": (repo_root / contract.ROOT).exists(),
        "s2_lease": (repo_root / contract.LEASE).exists(),
    }
    try:
        hashes = source_hashes(repo_root)
        records = build_multibank()
        task = audit_multibank(
            records,
            s1_records=predecessor.get("records")
            if predecessor.get("passed") is True
            else None,
        )
        k1, k8 = _paired_models(c1u_contract.SOURCE_HIDDEN_WIDTH, seed=contract.MODEL_SEED)
        parity = matched_parameter_audit(k1, k8)
        del k1, k8
        status = (
            "S2_FORMAL_CONSUMED"
            if paths["s2_root"] or paths["s2_lease"]
            else "S2_PREFLIGHT_CONSUMED"
            if paths["preflight_root"] or paths["preflight_lease"]
            else "S2_READY_UNCONSUMED"
        )
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.inspection.v1",
            "status": status,
            "passed": predecessor.get("passed") is True
            and task["passed"]
            and parity["passed"],
            "authorizes": "nothing",
            "inspection_mutations": 0,
            "contract": contract.manifest(),
            "predecessor": predecessor,
            "paths": paths,
            "task": task,
            "parameter_parity": parity,
            "source_identity": source_identity(hashes),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {
            "passed": False,
            "authorizes": "nothing",
            "inspection_mutations": 0,
            "paths": paths,
            "predecessor": predecessor,
            "reason": f"{type(exc).__name__}: {exc}",
        }


class StepJournal(AbstractContextManager["StepJournal"]):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle: Any = None
        self.completed = 0

    def __enter__(self) -> "StepJournal":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8", newline="\n")
        return self

    def __call__(self, row: Mapping[str, Any]) -> None:
        update = int(row["update"])
        if update != self.completed + 1:
            raise RuntimeError("S2 optimizer journal is not contiguous")
        self.handle.write(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.completed = update

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()
        return None


def _crash_result(
    root: Path,
    *,
    identity: str,
    stage: str,
    exc: BaseException,
    hashes: Mapping[str, str],
    formal_steps: int,
    disposable_steps: int,
    model_writes: int,
) -> dict[str, Any]:
    result = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.{stage}-result.v1",
        "identity": identity,
        "stage": stage,
        "status": f"CRASH_V2_A_C1U_{stage.upper().replace('-', '_')}",
        "passed": False,
        "authorizes": "nothing",
        "never_authorizes": list(contract.NEVER_AUTHORIZES),
        "exception": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "source_hashes": dict(hashes),
        "source_identity": source_identity(hashes),
        "formal_optimizer_steps": int(formal_steps),
        "disposable_optimizer_steps": int(disposable_steps),
        "model_writes": int(model_writes),
        "training_started": formal_steps > 0,
        "qualification_requires_seal_replay": True,
        "unrun_successors": ["S3", "single-seed formal"],
    }
    write_json(Path(root) / "result.json", result)
    write_evidence_seal(root, identity=identity, stage=stage)
    return {
        **result,
        "seal_replay": audit_evidence_seal(
            root, expected_identity=identity, expected_stage=stage
        ),
    }


def _benchmark(
    runtime: S2Runtime,
    records: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    train, _ = split_records(records, "F0")
    schedule = build_schedule(
        train,
        fold_id="F0",
        seed=contract.PREFLIGHT_ORDER_SEED,
        maximum_updates=contract.PREFLIGHT_STEPS_PER_ARM,
    )
    spec = replace(
        TrainSpec(),
        maximum_updates=contract.PREFLIGHT_STEPS_PER_ARM,
        warmup_updates=min(4, contract.PREFLIGHT_STEPS_PER_ARM),
        log_interval=contract.PREFLIGHT_STEPS_PER_ARM,
    )
    arms = {}
    initial_hashes = {}
    for arm in contract.ARMS:
        set_determinism(contract.PREFLIGHT_MODEL_SEED)
        model = S2Model(_config(runtime.cache.source_width, arm))
        initial_hashes[arm] = state_tensor_sha256(model)
        provider = BoundedBatchProvider(runtime)
        arms[arm] = train_fixed_endpoint(
            model,
            provider,
            schedule,
            identity=contract.PREFLIGHT_IDENTITY,
            arm=arm,
            fold_id="F0",
            model_seed=contract.PREFLIGHT_MODEL_SEED,
            order_seed=contract.PREFLIGHT_ORDER_SEED,
            spec=spec,
            device=device,
        )
        del model, provider
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    checks = {
        "both_arms": set(arms) == set(contract.ARMS),
        "finite_completed": all(
            row["passed"]
            and row["optimizer_steps"] == contract.PREFLIGHT_STEPS_PER_ARM
            for row in arms.values()
        ),
        "initialization_matched": len(set(initial_hashes.values())) == 1,
        "model_writes_zero": all(row["model_writes"] == 0 for row in arms.values()),
        "bounded_provider": all(row["provider"]["bounded"] for row in arms.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "arms": arms,
        "initial_hashes": initial_hashes,
        "disposable_optimizer_steps": sum(
            int(row["optimizer_steps"]) for row in arms.values()
        ),
        "formal_optimizer_steps": 0,
        "model_writes": 0,
    }


def run_preflight(
    repo_root: Path,
    *,
    device: str | torch.device = "cuda",
    asset_audit_fn: AuditFn = audit_source_assets,
    device_audit_fn: AuditFn = audit_device,
    tests_fn: Callable[[Path], Mapping[str, Any]] = run_registered_tests,
    encoder_factory: Callable[..., Any] = QwenCardEncoder,
    fault_at: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    predecessor = audit_s1_predecessor(repo_root)
    paths = audit_unconsumed_paths(repo_root, stage="preflight")
    tests = dict(tests_fn(repo_root))
    assets = dict(asset_audit_fn())
    hardware = dict(device_audit_fn(device))
    hashes = source_hashes(repo_root)
    records = build_multibank()
    task = audit_multibank(records, s1_records=predecessor.get("records"))
    schedules = _schedule_bundle(records, seed=contract.ORDER_SEED)
    k1, k8 = _paired_models(c1u_contract.SOURCE_HIDDEN_WIDTH, seed=contract.MODEL_SEED)
    parity = matched_parameter_audit(k1, k8)
    del k1, k8
    preclaim = {
        "predecessor": predecessor.get("passed") is True,
        "paths": paths["passed"],
        "tests": tests.get("passed") is True,
        "assets": assets.get("passed") is True,
        "device": hardware.get("passed") is True,
        "task": task["passed"],
        "schedules": set(schedules) == {str(fold["fold_id"]) for fold in contract.FOLDS}
        and all(
            row["updates"] == contract.MAXIMUM_UPDATES_PER_ENDPOINT
            and row["complete_groups_per_batch"]
            for row in schedules.values()
        ),
        "parameter_parity": parity["passed"],
        "authorization": contract.USER_AUTHORIZATION
        == contract.manifest()["user_authorization"],
    }
    if not all(preclaim.values()):
        raise RuntimeError(
            f"S2 preflight pre-claim audit failed; root was not consumed: {preclaim}"
        )
    root = repo_root / contract.PREFLIGHT_ROOT
    lease = repo_root / contract.PREFLIGHT_LEASE
    claim_single_use(
        identity=contract.PREFLIGHT_IDENTITY,
        stage="s2-preflight",
        output_root=root,
        lease_path=lease,
    )
    disposable_steps = 0
    try:
        if fault_at == "after_claim":
            raise RuntimeError("injected S2 preflight fault after claim")
        snapshot_sources(repo_root, root)
        write_json(root / "contract-manifest.json", contract.manifest())
        write_json(root / "task-banks.json", {"records": records})
        write_json(root / "task-audit.json", task)
        write_json(
            root / "preclaim-audit.json",
            {
                "checks": preclaim,
                "predecessor": predecessor,
                "paths": paths,
                "tests": tests,
                "assets": assets,
                "device": hardware,
                "parameter_parity": parity,
            },
        )
        for fold_id, report in schedules.items():
            write_json(root / "schedules" / f"{fold_id}.json", report)

        encoder = encoder_factory(device=device)
        try:
            cache = build_independent_card_cache(
                records,
                encoder,
                source_identity={
                    "model_id": c1u_contract.SOURCE_MODEL_ID,
                    "revision": c1u_contract.SOURCE_MODEL_REVISION,
                    "s2_source_identity": source_identity(hashes),
                },
            )
            encoder_report = dict(encoder.report())
        finally:
            encoder.release()
        cache_manifest = persist_cache(root / "card-cache", records, cache)
        cache_audit = audit_cache(root / "card-cache", records)
        runtime = S2Runtime(records, load_cache(root / "card-cache"))
        structural = _structural_controls(runtime, records, device=device)
        benchmark = _benchmark(runtime, records, device=device)
        disposable_steps = int(benchmark["disposable_optimizer_steps"])
        write_json(root / "source-encoder.json", encoder_report)
        write_json(root / "cache-manifest-copy.json", cache_manifest)
        write_json(root / "cache-audit.json", cache_audit)
        write_json(root / "structural-controls.json", structural)
        write_json(root / "benchmark.json", benchmark)
        post_hashes = source_hashes(repo_root)
        final_predecessor = audit_s1_predecessor(repo_root)
        gates = {
            contract.PREFLIGHT_GATES[0]: preclaim["predecessor"]
            and preclaim["paths"]
            and preclaim["authorization"],
            contract.PREFLIGHT_GATES[1]: preclaim["tests"]
            and preclaim["assets"]
            and preclaim["device"],
            contract.PREFLIGHT_GATES[2]: task["passed"],
            contract.PREFLIGHT_GATES[3]: cache_audit["passed"]
            and cache_manifest["contains_targets"] is False
            and encoder_report.get("calls") == contract.TOTAL_RECORDS * 6,
            contract.PREFLIGHT_GATES[4]: parity["passed"] and structural["passed"],
            contract.PREFLIGHT_GATES[5]: benchmark["passed"]
            and disposable_steps
            == len(contract.ARMS) * contract.PREFLIGHT_STEPS_PER_ARM
            and all(
                row["updates"] == contract.MAXIMUM_UPDATES_PER_ENDPOINT
                for row in schedules.values()
            ),
            contract.PREFLIGHT_GATES[6]: hashes == post_hashes
            and final_predecessor.get("passed") is True,
        }
        passed = all(gates.values())
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": contract.PREFLIGHT_IDENTITY,
            "stage": "s2-preflight",
            "status": "PASS_V2_A_C1U_S2_PREFLIGHT"
            if passed
            else "FAIL_V2_A_C1U_S2_PREFLIGHT",
            "passed": passed,
            "authorizes": contract.IDENTITY if passed else "nothing",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
            "gates": gates,
            "source_hashes": hashes,
            "source_identity": source_identity(hashes),
            "s1_result_sha256": predecessor.get("result_sha256"),
            "s1_seal_sha256": predecessor.get("seal_sha256"),
            "bank_payload_sha256": task["payload_sha256"],
            "cache_files": cache_manifest["files"],
            "schedule_sha256": {
                fold_id: report["sha256"] for fold_id, report in schedules.items()
            },
            "registered_initial_sha256": parity["k1_initial_sha256"],
            "formal_optimizer_steps": 0,
            "disposable_optimizer_steps": disposable_steps,
            "model_writes": 0,
            "training_started": False,
            "qualification_requires_seal_replay": True,
            "unrun_successors": ["S2 training", "S3", "single-seed formal"],
        }
        write_json(root / "result.json", result)
        write_evidence_seal(
            root, identity=contract.PREFLIGHT_IDENTITY, stage="s2-preflight"
        )
        return {
            **result,
            "seal_replay": audit_evidence_seal(
                root,
                expected_identity=contract.PREFLIGHT_IDENTITY,
                expected_stage="s2-preflight",
            ),
        }
    except Exception as exc:
        return _crash_result(
            root,
            identity=contract.PREFLIGHT_IDENTITY,
            stage="s2-preflight",
            exc=exc,
            hashes=hashes,
            formal_steps=0,
            disposable_steps=disposable_steps,
            model_writes=0,
        )


def audit_preflight(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.PREFLIGHT_ROOT
    try:
        result = read_json(root / "result.json")
        seal = audit_evidence_seal(
            root,
            expected_identity=contract.PREFLIGHT_IDENTITY,
            expected_stage="s2-preflight",
        )
        hashes = source_hashes(repo_root)
        recorded = result.get("source_hashes")
        if not isinstance(recorded, dict):
            raise TypeError("S2 preflight result lacks source hashes")
        snapshot = {
            relative: sha256_file(root / "source_snapshot" / relative)
            for relative in hashes
        }
        records_payload = read_json(root / "task-banks.json")
        records = records_payload.get("records")
        if not isinstance(records, list):
            raise TypeError("S2 preflight task banks are missing")
        regenerated = build_multibank()
        predecessor = audit_s1_predecessor(repo_root)
        task = audit_multibank(records, s1_records=predecessor.get("records"))
        regenerated_task = audit_multibank(
            regenerated, s1_records=predecessor.get("records")
        )
        cache = audit_cache(root / "card-cache", records)
        schedule_checks = {}
        for fold in contract.FOLDS:
            fold_id = str(fold["fold_id"])
            persisted = read_json(root / "schedules" / f"{fold_id}.json")
            train, _ = split_records(records, fold_id)
            rebuilt = schedule_report(
                build_schedule(
                    train,
                    fold_id=fold_id,
                    seed=contract.ORDER_SEED,
                    maximum_updates=contract.MAXIMUM_UPDATES_PER_ENDPOINT,
                )
            )
            schedule_checks[fold_id] = persisted.get("sha256") == rebuilt["sha256"] == result.get("schedule_sha256", {}).get(fold_id)
        gates = dict(result.get("gates", {}))
        checks = {
            "lease": (repo_root / contract.PREFLIGHT_LEASE).is_file(),
            "identity": result.get("identity") == contract.PREFLIGHT_IDENTITY,
            "stage": result.get("stage") == "s2-preflight",
            "terminal_pass": result.get("status") == "PASS_V2_A_C1U_S2_PREFLIGHT"
            and result.get("passed") is True
            and result.get("authorizes") == contract.IDENTITY,
            "gates": set(gates) == set(contract.PREFLIGHT_GATES)
            and all(gates.values()),
            "source": hashes == recorded
            and source_identity(hashes) == result.get("source_identity"),
            "snapshot": snapshot == recorded,
            "predecessor": predecessor.get("passed") is True,
            "bank_payload": task["passed"]
            and regenerated_task["passed"]
            and task["payload_sha256"]
            == regenerated_task["payload_sha256"]
            == result.get("bank_payload_sha256"),
            "cache": cache["passed"],
            "schedules": all(schedule_checks.values()),
            "accounting": result.get("formal_optimizer_steps") == 0
            and result.get("disposable_optimizer_steps")
            == len(contract.ARMS) * contract.PREFLIGHT_STEPS_PER_ARM
            and result.get("model_writes") == 0
            and result.get("training_started") is False,
            "seal": seal.get("passed") is True,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "result": result,
            "seal": seal,
            "predecessor": predecessor,
            "cache": cache,
            "task": task,
            "schedule_checks": schedule_checks,
            "result_sha256": sha256_file(root / "result.json"),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _evaluate_endpoint(
    model: S2Model,
    runtime: S2Runtime,
    records: Sequence[Mapping[str, Any]],
    *,
    fold_id: str,
    arm: str,
    device: str | torch.device,
) -> dict[str, Any]:
    active_device = torch.device(device)
    cpu = runtime.get_batch([str(row["example_id"]) for row in records])
    batch = _move_batch(cpu, active_device)
    model.to(active_device).eval()
    with _autocast(active_device):
        result = collect_rows(model, batch, fold_id=fold_id, arm=arm)
    result["device"] = str(active_device)
    result["dtype"] = "bfloat16_autocast" if active_device.type == "cuda" else "float32"
    return result


def run_s2(
    repo_root: Path,
    *,
    device: str | torch.device = "cuda",
    asset_audit_fn: AuditFn = audit_source_assets,
    device_audit_fn: AuditFn = audit_device,
    fault_at: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    preflight = audit_preflight(repo_root)
    paths = audit_unconsumed_paths(repo_root, stage="s2")
    predecessor = audit_s1_predecessor(repo_root)
    assets = dict(asset_audit_fn())
    hardware = dict(device_audit_fn(device))
    hashes = source_hashes(repo_root)
    preflight_root = repo_root / contract.PREFLIGHT_ROOT
    records_payload = read_json(preflight_root / "task-banks.json")
    records = records_payload.get("records")
    if not isinstance(records, list):
        raise TypeError("sealed S2 preflight task banks are missing")
    task = audit_multibank(records, s1_records=predecessor.get("records"))
    cache_audit = audit_cache(preflight_root / "card-cache", records)
    schedules = _schedule_bundle(records, seed=contract.ORDER_SEED)
    preflight_schedules = preflight.get("result", {}).get("schedule_sha256", {})
    preclaim = {
        "preflight": preflight.get("passed") is True,
        "paths": paths["passed"],
        "predecessor": predecessor.get("passed") is True,
        "assets": assets.get("passed") is True,
        "device": hardware.get("passed") is True,
        "source": source_identity(hashes)
        == preflight.get("result", {}).get("source_identity"),
        "task": task["passed"]
        and task["payload_sha256"]
        == preflight.get("result", {}).get("bank_payload_sha256"),
        "cache": cache_audit["passed"],
        "schedules": all(
            report["sha256"] == preflight_schedules.get(fold_id)
            for fold_id, report in schedules.items()
        ),
    }
    if not all(preclaim.values()):
        raise RuntimeError(
            f"S2 pre-claim audit failed; fixed formal root was not consumed: {preclaim}"
        )
    root = repo_root / contract.ROOT
    lease = repo_root / contract.LEASE
    claim_single_use(
        identity=contract.IDENTITY,
        stage="s2",
        output_root=root,
        lease_path=lease,
    )
    formal_steps = 0
    model_writes = 0
    endpoint_rows = []
    try:
        if fault_at == "after_claim":
            raise RuntimeError("injected S2 fault after claim")
        snapshot_sources(repo_root, root)
        write_json(root / "contract-manifest.json", contract.manifest())
        write_json(
            root / "preclaim-audit.json",
            {
                "checks": preclaim,
                "preflight_result_sha256": preflight.get("result_sha256"),
                "preflight_seal_sha256": preflight.get("seal", {}).get("seal_sha256"),
                "predecessor": predecessor,
                "assets": assets,
                "device": hardware,
                "task": task,
                "cache": cache_audit,
            },
        )
        for fold_id, report in schedules.items():
            write_json(root / "schedules" / f"{fold_id}.json", report)
        runtime = S2Runtime(records, load_cache(preflight_root / "card-cache"))
        k8_rows: list[dict[str, Any]] = []
        k1_rows: list[dict[str, Any]] = []
        expected_initial = str(
            preflight.get("result", {}).get("registered_initial_sha256", "")
        )
        for fold in contract.FOLDS:
            fold_id = str(fold["fold_id"])
            train_records, heldout_records = split_records(records, fold_id)
            schedule = build_schedule(
                train_records,
                fold_id=fold_id,
                seed=contract.ORDER_SEED,
                maximum_updates=contract.MAXIMUM_UPDATES_PER_ENDPOINT,
            )
            schedule_audit = schedule_report(schedule)
            for arm in contract.ARMS:
                set_determinism(contract.MODEL_SEED)
                model = S2Model(_config(runtime.cache.source_width, arm))
                initial_sha = state_tensor_sha256(model)
                provider = BoundedBatchProvider(runtime)
                progress_path = root / "progress" / f"{fold_id}-{arm}.jsonl"
                with StepJournal(progress_path) as journal:
                    training = train_fixed_endpoint(
                        model,
                        provider,
                        schedule,
                        identity=contract.IDENTITY,
                        arm=arm,
                        fold_id=fold_id,
                        model_seed=contract.MODEL_SEED,
                        order_seed=contract.ORDER_SEED,
                        spec=TrainSpec(),
                        device=device,
                        on_optimizer_step=journal,
                    )
                    completed = journal.completed
                formal_steps += completed
                write_json(
                    root / "training" / f"{fold_id}-{arm}.json", training
                )
                if fault_at == f"after_{fold_id}_{arm}_training":
                    raise RuntimeError(f"injected S2 fault after {fold_id}/{arm}")
                endpoint_path = (
                    root / "endpoints" / fold_id / arm / f"{contract.FIXED_ENDPOINT}.pt"
                )
                endpoint_sha = save_endpoint(
                    endpoint_path,
                    model=model,
                    identity=contract.IDENTITY,
                    arm=arm,
                    fold_id=fold_id,
                    update=completed,
                    schedule_sha256=schedule_audit["sha256"],
                )
                model_writes += 1
                del model, provider
                gc.collect()
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
                endpoint_model, endpoint_payload = load_endpoint(
                    endpoint_path, device=device
                )
                endpoint_checks = {
                    "identity": endpoint_payload.get("identity") == contract.IDENTITY,
                    "arm": endpoint_payload.get("arm") == arm,
                    "fold": endpoint_payload.get("fold_id") == fold_id,
                    "fixed_endpoint": endpoint_payload.get("fixed_endpoint_name")
                    == contract.FIXED_ENDPOINT,
                    "update": endpoint_payload.get("update")
                    == contract.MAXIMUM_UPDATES_PER_ENDPOINT,
                    "schedule": endpoint_payload.get("schedule_sha256")
                    == schedule_audit["sha256"],
                    "optimizer_absent": endpoint_payload.get("optimizer_state_saved")
                    is False,
                    "checkpoint_selection_absent": endpoint_payload.get(
                        "checkpoint_selection"
                    )
                    is False,
                    "file_hash": sha256_file(endpoint_path) == endpoint_sha,
                    "model_integrity": endpoint_model.integrity_report()["passed"],
                    "initial_hash": initial_sha == expected_initial,
                }
                evaluation = _evaluate_endpoint(
                    endpoint_model,
                    runtime,
                    heldout_records,
                    fold_id=fold_id,
                    arm=arm,
                    device=device,
                )
                write_json(
                    root / "evaluation" / f"{fold_id}-{arm}.json", evaluation
                )
                rows = [dict(row) for row in evaluation["rows"]]
                (k1_rows if arm == "K1" else k8_rows).extend(rows)
                endpoint_rows.append(
                    {
                        "fold_id": fold_id,
                        "arm": arm,
                        "path": endpoint_path.relative_to(root).as_posix(),
                        "sha256": endpoint_sha,
                        "bytes": endpoint_path.stat().st_size,
                        "initial_sha256": initial_sha,
                        "schedule_sha256": schedule_audit["sha256"],
                        "optimizer_steps": completed,
                        "model_writes": 1,
                        "checks": endpoint_checks,
                        "training_wall_seconds": training["wall_seconds"],
                        "peak_memory_bytes": training["peak_memory_bytes"],
                    }
                )
                del endpoint_model, evaluation
                gc.collect()
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()

        write_json(root / "endpoints.json", {"rows": endpoint_rows})
        qualification = qualify(k8_rows, k1_rows)
        write_json(root / "qualification.json", qualification)
        post_hashes = source_hashes(repo_root)
        final_predecessor = audit_s1_predecessor(repo_root)
        final_cache = audit_cache(preflight_root / "card-cache", records)
        endpoint_integrity = all(
            all(row["checks"].values())
            and sha256_file(root / row["path"]) == row["sha256"]
            for row in endpoint_rows
        )
        parity = (
            len(endpoint_rows) == contract.ENDPOINT_COUNT
            and len({row["initial_sha256"] for row in endpoint_rows}) == 1
            and all(
                schedules[row["fold_id"]]["sha256"] == row["schedule_sha256"]
                for row in endpoint_rows
            )
        )
        sections = qualification["sections"]
        gates = {
            contract.RESULT_GATES[0]: all(preclaim.values())
            and hashes == post_hashes,
            contract.RESULT_GATES[1]: formal_steps
            == contract.TOTAL_FORMAL_OPTIMIZER_STEPS
            and model_writes == contract.ENDPOINT_COUNT
            and parity
            and endpoint_integrity,
            contract.RESULT_GATES[2]: all(
                sections["absolute_behavior"].values()
            ),
            contract.RESULT_GATES[3]: all(sections["paired_gain"].values()),
            contract.RESULT_GATES[4]: all(sections["causal_behavior"].values()),
            contract.RESULT_GATES[5]: all(sections["functional_k"].values()),
            contract.RESULT_GATES[6]: hashes == post_hashes
            and final_predecessor.get("passed") is True
            and final_cache.get("passed") is True
            and endpoint_integrity,
        }
        passed = qualification["passed"] is True and all(gates.values())
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s2-result.v1",
            "identity": contract.IDENTITY,
            "stage": "s2",
            "status": "PASS_V2_A_C1U_S2_MULTIBANK_MATCHED_K1_K8"
            if passed
            else "FAIL_V2_A_C1U_S2_QUALIFICATION",
            "passed": passed,
            "authorizes": contract.PASS_AUTHORIZATION if passed else "nothing",
            "never_authorizes": list(contract.NEVER_AUTHORIZES),
            "gates": gates,
            "qualification": qualification,
            "endpoint_count": len(endpoint_rows),
            "endpoints": endpoint_rows,
            "scientific_model_seeds": [contract.MODEL_SEED],
            "multi_seed": False,
            "folds": [str(fold["fold_id"]) for fold in contract.FOLDS],
            "source_hashes": hashes,
            "source_identity": source_identity(hashes),
            "preflight_result_sha256": preflight.get("result_sha256"),
            "preflight_seal_sha256": preflight.get("seal", {}).get("seal_sha256"),
            "s1_result_sha256": predecessor.get("result_sha256"),
            "s1_seal_sha256": predecessor.get("seal_sha256"),
            "bank_payload_sha256": task["payload_sha256"],
            "formal_optimizer_steps": formal_steps,
            "disposable_optimizer_steps": 0,
            "model_writes": model_writes,
            "checkpoint_selection": False,
            "intermediate_checkpoints": 0,
            "training_started": True,
            "qualification_requires_seal_replay": True,
            "unrun_successors": ["S3 training", "single-seed formal"],
        }
        write_json(root / "result.json", result)
        write_evidence_seal(root, identity=contract.IDENTITY, stage="s2")
        return {
            **result,
            "seal_replay": audit_evidence_seal(
                root, expected_identity=contract.IDENTITY, expected_stage="s2"
            ),
        }
    except Exception as exc:
        return _crash_result(
            root,
            identity=contract.IDENTITY,
            stage="s2",
            exc=exc,
            hashes=hashes,
            formal_steps=formal_steps,
            disposable_steps=0,
            model_writes=model_writes,
        )


def audit_s2(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.ROOT
    try:
        result = read_json(root / "result.json")
        seal = audit_evidence_seal(
            root, expected_identity=contract.IDENTITY, expected_stage="s2"
        )
        hashes = source_hashes(repo_root)
        recorded = result.get("source_hashes")
        if not isinstance(recorded, dict):
            raise TypeError("S2 result lacks source hashes")
        snapshot = {
            relative: sha256_file(root / "source_snapshot" / relative)
            for relative in hashes
        }
        predecessor = audit_s1_predecessor(repo_root)
        preflight = audit_preflight(repo_root)
        endpoints = result.get("endpoints")
        if not isinstance(endpoints, list):
            raise TypeError("S2 result lacks endpoint ledger")
        endpoint_checks = all(
            (root / str(row["path"])).is_file()
            and sha256_file(root / str(row["path"])) == row.get("sha256")
            and all(dict(row.get("checks", {})).values())
            for row in endpoints
        )
        gates = dict(result.get("gates", {}))
        status = str(result.get("status", ""))
        passed_terminal = status == "PASS_V2_A_C1U_S2_MULTIBANK_MATCHED_K1_K8"
        failed_terminal = status == "FAIL_V2_A_C1U_S2_QUALIFICATION"
        crashed = status == "CRASH_V2_A_C1U_S2"
        coherent = (
            passed_terminal
            and result.get("passed") is True
            and result.get("authorizes") == contract.PASS_AUTHORIZATION
            and set(gates) == set(contract.RESULT_GATES)
            and all(gates.values())
        ) or (
            failed_terminal
            and result.get("passed") is False
            and result.get("authorizes") == "nothing"
            and set(gates) == set(contract.RESULT_GATES)
            and not all(gates.values())
        ) or (
            crashed
            and result.get("passed") is False
            and result.get("authorizes") == "nothing"
        )
        checks = {
            "lease": (repo_root / contract.LEASE).is_file(),
            "identity": result.get("identity") == contract.IDENTITY,
            "stage": result.get("stage") == "s2",
            "terminal_status": passed_terminal or failed_terminal or crashed,
            "terminal_coherence": coherent,
            "source": hashes == recorded
            and source_identity(hashes) == result.get("source_identity"),
            "snapshot": snapshot == recorded,
            "predecessor": predecessor.get("passed") is True,
            "preflight": preflight.get("passed") is True,
            "accounting": crashed
            or (
                result.get("formal_optimizer_steps")
                == contract.TOTAL_FORMAL_OPTIMIZER_STEPS
                and result.get("disposable_optimizer_steps") == 0
                and result.get("model_writes") == contract.ENDPOINT_COUNT
                and result.get("checkpoint_selection") is False
                and result.get("intermediate_checkpoints") == 0
            ),
            "scientific_model_seed": result.get("scientific_model_seeds")
            == [contract.MODEL_SEED]
            and result.get("multi_seed") is False,
            "endpoints": crashed or endpoint_checks,
            "seal": seal.get("passed") is True,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "result": result,
            "seal": seal,
            "predecessor": predecessor,
            "preflight": preflight,
            "result_sha256": sha256_file(root / "result.json"),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "StepJournal",
    "audit_preflight",
    "audit_s1_predecessor",
    "audit_s2",
    "audit_unconsumed_paths",
    "inspect",
    "run_preflight",
    "run_registered_tests",
    "run_s2",
]
