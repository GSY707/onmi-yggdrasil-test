from __future__ import annotations

"""Fail-closed single-use preflight and formal execution for C1U S1."""

from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
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
from .cache import audit_persisted_cache, load_independent_cache
from .evaluate import evaluate_s1
from .model import C1UConfig, C1UModel
from .runtime import C1URuntime, audit_independent_card_cache
from .s0 import structural_measurements
from .source import audit_device, audit_source_assets
from .tasks import audit_s1_bank, build_s1_bank
from .train import (
    RuntimeBatchProvider,
    TrainSpec,
    build_s1_schedule,
    load_endpoint,
    save_endpoint,
    schedule_report,
    set_determinism,
    train_fixed_endpoint,
)


AuditFn = Callable[..., Mapping[str, Any]]


def _s0_root(repo_root: Path) -> Path:
    return Path(repo_root).resolve() / contract.S0_ROOT


def _s0_records(repo_root: Path) -> list[dict[str, Any]]:
    payload = read_json(_s0_root(repo_root) / "task-bank.json")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != contract.S1_RECORDS:
        raise ValueError("sealed S0 task bank has the wrong cardinality")
    if not all(isinstance(row, dict) for row in records):
        raise TypeError("sealed S0 task rows must be mappings")
    return [dict(row) for row in records]


def audit_s0_predecessor(repo_root: Path) -> dict[str, Any]:
    """Replay S0 internally and pin the scientific core needed by S1."""

    repo_root = Path(repo_root).resolve()
    root = _s0_root(repo_root)
    try:
        result_path = root / "result.json"
        seal_path = root / "evidence-seal.json"
        result = read_json(result_path)
        seal = audit_evidence_seal(
            root, expected_identity=contract.S0_IDENTITY, expected_stage="s0"
        )
        records = _s0_records(repo_root)
        regenerated = build_s1_bank()
        task_audit = audit_s1_bank(records)
        cache_root = root / "card-cache"
        cache_audit = audit_persisted_cache(cache_root, records)
        expected_source_hashes = result.get("source_hashes")
        if not isinstance(expected_source_hashes, dict):
            raise TypeError("S0 result lacks source hashes")
        core_pins: dict[str, Any] = {}
        for relative in contract.S0_SCIENTIFIC_CORE_PINS:
            current_path = repo_root / relative
            snapshot_path = root / "source_snapshot" / relative
            expected = str(expected_source_hashes.get(relative, ""))
            current = sha256_file(current_path)
            snapshot = sha256_file(snapshot_path)
            core_pins[relative] = {
                "passed": bool(expected) and current == expected and snapshot == expected,
                "expected": expected,
                "current": current,
                "snapshot": snapshot,
            }
        checks = {
            "s0_lease": (repo_root / contract.S0_LEASE).is_file(),
            "identity": result.get("identity") == contract.S0_IDENTITY,
            "status": result.get("status") == "PASS_V2_A_C1U_S0_QUALIFICATION",
            "passed": result.get("passed") is True,
            "authorization": result.get("authorizes")
            == contract.S0_PASS_AUTHORIZATION,
            "zero_training": result.get("optimizer_steps") == 0
            and result.get("model_writes") == 0,
            "result_sha256": sha256_file(result_path) == contract.S0_RESULT_SHA256,
            "seal_sha256": sha256_file(seal_path)
            == contract.S0_EVIDENCE_SEAL_SHA256,
            "seal_replay": seal.get("passed") is True,
            "source_identity": result.get("source_identity")
            == contract.S0_SOURCE_IDENTITY,
            "scientific_core_pins": bool(core_pins)
            and all(row["passed"] for row in core_pins.values()),
            "task_bank": task_audit["passed"] and records == regenerated,
            "cache_audit": cache_audit.get("passed") is True,
            "cache_tensor_pin": sha256_file(cache_root / "cards.pt")
            == contract.S0_CARD_CACHE_SHA256,
            "cache_ledger_pin": sha256_file(cache_root / "ledger.json")
            == contract.S0_CARD_LEDGER_SHA256,
        }
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s1-predecessor-audit.v1",
            "passed": all(checks.values()),
            "checks": checks,
            "s0_identity": result.get("identity"),
            "s0_status": result.get("status"),
            "s0_authorizes": result.get("authorizes"),
            "s0_result_sha256": sha256_file(result_path),
            "s0_seal": seal,
            "scientific_core_pins": core_pins,
            "task_audit": task_audit,
            "cache_audit": cache_audit,
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def audit_s1_unconsumed_paths(repo_root: Path, *, stage: str) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    paths = {
        "s1_preflight_root": repo_root / contract.S1_PREFLIGHT_ROOT,
        "s1_preflight_lease": repo_root / contract.S1_PREFLIGHT_LEASE,
        "s1_root": repo_root / contract.S1_ROOT,
        "s1_lease": repo_root / contract.S1_LEASE,
    }
    if stage == "s1-preflight":
        required_absent = tuple(paths)
    elif stage == "s1":
        required_absent = ("s1_root", "s1_lease")
    else:
        raise ValueError(f"unknown S1 path audit stage: {stage}")
    existence = {name: path.exists() for name, path in paths.items()}
    checks = {name: not existence[name] for name in required_absent}
    return {
        "passed": all(checks.values()),
        "stage": stage,
        "checks": checks,
        "exists": existence,
        "paths": {name: path.as_posix() for name, path in paths.items()},
    }


def run_registered_tests(repo_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/v2_a_closure_c1u",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(repo_root),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _runtime_bundle(
    repo_root: Path,
) -> tuple[list[dict[str, Any]], C1URuntime, dict[str, Any]]:
    records = _s0_records(repo_root)
    cache_root = _s0_root(repo_root) / "card-cache"
    cache_audit = audit_persisted_cache(cache_root, records)
    if not cache_audit.get("passed"):
        raise ValueError("sealed S0 card cache failed S1 readback audit")
    cache = load_independent_cache(cache_root)
    runtime_audit = audit_independent_card_cache(records, cache)
    if not runtime_audit["passed"]:
        raise ValueError("loaded S0 card cache failed runtime audit")
    runtime = C1URuntime(records, cache)
    return records, runtime, {
        "persistent": cache_audit,
        "runtime": runtime_audit,
        "source_width": cache.source_width,
    }


def _model_config(source_width: int, override: C1UConfig | None) -> C1UConfig:
    config = C1UConfig(source_width=source_width) if override is None else override
    if config.source_width != source_width:
        raise ValueError("registered model source width differs from the sealed cache")
    return config


class StepJournal:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8", newline="\n")
        self.completed = 0

    def __enter__(self) -> "StepJournal":
        return self

    def __call__(self, row: Mapping[str, Any]) -> None:
        update = int(row["update"])
        if update != self.completed + 1:
            raise RuntimeError("optimizer-step journal lost contiguity")
        self.handle.write(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
        )
        self.handle.flush()
        if update == 1 or update % 25 == 0:
            os.fsync(self.handle.fileno())
        self.completed = update

    def __exit__(self, *_: object) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _crash_result(
    root: Path,
    *,
    identity: str,
    stage: str,
    exc: Exception,
    hashes: Mapping[str, str] | None,
    optimizer_steps: int,
    disposable_optimizer_steps: int,
    model_writes: int,
) -> dict[str, Any]:
    result = {
        "schema_version": f"{contract.SCHEMA_PREFIX}.{stage}-result.v1",
        "identity": identity,
        "stage": stage,
        "status": f"CRASH_V2_A_C1U_{stage.upper().replace('-', '_')}",
        "passed": False,
        "authorizes": "nothing",
        "never_authorizes": list(contract.S1_NEVER_AUTHORIZES),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "source_hashes": dict(hashes or {}),
        "source_identity": source_identity(hashes) if hashes else None,
        "optimizer_steps": int(optimizer_steps),
        "formal_optimizer_steps": int(optimizer_steps),
        "disposable_optimizer_steps": int(disposable_optimizer_steps),
        "model_writes": int(model_writes),
    }
    write_json(root / "result.json", result)
    write_evidence_seal(root, identity=identity, stage=stage)
    return result


def _preflight_spec(steps: int) -> TrainSpec:
    return replace(
        TrainSpec(),
        maximum_updates=int(steps),
        warmup_updates=min(8, int(steps)),
        log_interval=max(1, min(8, int(steps))),
    )


def run_s1_preflight(
    repo_root: Path,
    *,
    device: str | torch.device = "cuda",
    asset_audit_fn: AuditFn = audit_source_assets,
    device_audit_fn: AuditFn = audit_device,
    test_audit_fn: AuditFn = run_registered_tests,
    predecessor_audit_fn: AuditFn = audit_s0_predecessor,
    config_override: C1UConfig | None = None,
    benchmark_steps: int = contract.S1_PREFLIGHT_BENCHMARK_STEPS,
    fault_at: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S1_PREFLIGHT_ROOT
    lease = repo_root / contract.S1_PREFLIGHT_LEASE
    hashes = source_hashes(repo_root)
    predecessor = dict(predecessor_audit_fn(repo_root))
    paths = audit_s1_unconsumed_paths(repo_root, stage="s1-preflight")
    assets = dict(asset_audit_fn())
    hardware = dict(device_audit_fn(device))
    tests = dict(test_audit_fn(repo_root))
    records, runtime, cache_audit = _runtime_bundle(repo_root)
    task_audit = audit_s1_bank(records)
    formal_schedule = build_s1_schedule(
        records, seed=contract.S1_ORDER_SEED, maximum_updates=contract.S1_MAXIMUM_UPDATES
    )
    formal_schedule_report = schedule_report(formal_schedule)
    structural = structural_measurements(
        records,
        runtime.cache,
        device=device,
        config_override=config_override,
    )
    preclaim_hashes = source_hashes(repo_root)
    preclaim_checks = {
        "predecessor": predecessor.get("passed") is True,
        "paths": paths["passed"],
        "assets": assets.get("passed") is True,
        "device": hardware.get("passed") is True,
        "tests": tests.get("passed") is True,
        "task": task_audit["passed"],
        "cache": cache_audit["persistent"].get("passed") is True
        and cache_audit["runtime"]["passed"],
        "structural": structural["passed"],
        "source_stable": preclaim_hashes == hashes,
        "user_authorization": contract.S1_USER_AUTHORIZATION == "没问题，按这个顺序做",
    }
    if not all(preclaim_checks.values()):
        raise RuntimeError(
            f"S1 preflight pre-claim audit failed; fixed root was not consumed: {preclaim_checks}"
        )

    claim_single_use(
        identity=contract.S1_PREFLIGHT_IDENTITY,
        stage="s1-preflight",
        output_root=root,
        lease_path=lease,
    )
    disposable_steps = 0
    try:
        if fault_at == "after_claim":
            raise RuntimeError("injected C1U S1 preflight fault after claim")
        snapshot_sources(repo_root, root)
        write_json(root / "contract-manifest.json", contract.manifest())
        write_json(
            root / "preclaim-audit.json",
            {
                "checks": preclaim_checks,
                "predecessor": predecessor,
                "paths": paths,
                "assets": assets,
                "device": hardware,
                "tests": tests,
            },
        )
        write_json(root / "task-bank-audit.json", task_audit)
        write_json(root / "cache-audit.json", cache_audit)
        write_json(root / "structural-measurements.json", structural)
        write_json(root / "formal-schedule.json", formal_schedule_report)

        benchmark_schedule = build_s1_schedule(
            records,
            seed=contract.S1_PREFLIGHT_ORDER_SEED,
            maximum_updates=int(benchmark_steps),
        )
        benchmark_schedule_report = schedule_report(benchmark_schedule)
        set_determinism(contract.S1_PREFLIGHT_MODEL_SEED)
        config = _model_config(runtime.cache.source_width, config_override)
        model = C1UModel(config)
        provider = RuntimeBatchProvider(runtime)
        with StepJournal(root / "disposable-training-progress.jsonl") as journal:
            benchmark = train_fixed_endpoint(
                model,
                provider,
                benchmark_schedule,
                identity="DISPOSABLE_C1U_S1_PREFLIGHT_BENCHMARK",
                model_seed=contract.S1_PREFLIGHT_MODEL_SEED,
                order_seed=contract.S1_PREFLIGHT_ORDER_SEED,
                spec=_preflight_spec(int(benchmark_steps)),
                device=device,
                on_optimizer_step=journal,
            )
            disposable_steps = journal.completed
        del model, provider
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        benchmark["formal_stage"] = False
        benchmark["weights_saved"] = False
        benchmark["schedule"] = benchmark_schedule_report
        write_json(root / "disposable-benchmark.json", benchmark)

        estimated = (
            float(benchmark["step_p95_seconds"]) * contract.S1_MAXIMUM_UPDATES
        )
        post_hashes = source_hashes(repo_root)
        gates = {
            contract.S1_PREFLIGHT_GATES[0]: predecessor["passed"]
            and contract.S1_USER_AUTHORIZATION == "没问题，按这个顺序做",
            contract.S1_PREFLIGHT_GATES[1]: paths["passed"]
            and assets["passed"]
            and hardware["passed"]
            and tests["passed"],
            contract.S1_PREFLIGHT_GATES[2]: task_audit["passed"]
            and cache_audit["persistent"]["passed"]
            and cache_audit["runtime"]["passed"]
            and structural["passed"]
            and formal_schedule_report["updates"] == contract.S1_MAXIMUM_UPDATES
            and formal_schedule_report["complete_groups_per_batch"],
            contract.S1_PREFLIGHT_GATES[3]: benchmark["passed"]
            and disposable_steps == int(benchmark_steps)
            and benchmark["peak_memory_bytes"]
            <= contract.S1_PREFLIGHT_MAX_PEAK_CUDA_BYTES
            and benchmark["step_p95_seconds"]
            <= contract.S1_PREFLIGHT_MAX_STEP_P95_SECONDS
            and estimated <= contract.S1_PREFLIGHT_MAX_ESTIMATED_FORMAL_SECONDS,
            contract.S1_PREFLIGHT_GATES[4]: post_hashes == hashes
            and benchmark["model_writes"] == 0,
        }
        passed = all(gates.values())
        if fault_at == "before_result":
            raise RuntimeError("injected C1U S1 preflight fault before result")
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s1-preflight-result.v1",
            "identity": contract.S1_PREFLIGHT_IDENTITY,
            "stage": "s1-preflight",
            "status": "PASS_V2_A_C1U_S1_PREFLIGHT"
            if passed
            else "FAIL_V2_A_C1U_S1_PREFLIGHT",
            "passed": passed,
            "authorizes": contract.S1_AUTHORIZED_SCOPE if passed else "nothing",
            "user_authorization": contract.S1_USER_AUTHORIZATION,
            "never_authorizes": list(contract.S1_NEVER_AUTHORIZES),
            "gates": gates,
            "source_hashes": hashes,
            "source_identity": source_identity(hashes),
            "s0_predecessor": {
                "passed": predecessor["passed"],
                "result_sha256": predecessor["s0_result_sha256"],
                "seal_sha256": predecessor["s0_seal"]["seal_sha256"],
            },
            "formal_schedule_sha256": formal_schedule_report["sha256"],
            "benchmark": benchmark,
            "estimated_formal_seconds_from_p95": estimated,
            "optimizer_steps": 0,
            "formal_optimizer_steps": 0,
            "disposable_optimizer_steps": disposable_steps,
            "model_writes": 0,
            "training_started": False,
            "qualification_requires_seal_replay": True,
        }
        write_json(root / "result.json", result)
        write_evidence_seal(
            root, identity=contract.S1_PREFLIGHT_IDENTITY, stage="s1-preflight"
        )
        return {
            **result,
            "seal_replay": audit_evidence_seal(
                root,
                expected_identity=contract.S1_PREFLIGHT_IDENTITY,
                expected_stage="s1-preflight",
            ),
        }
    except Exception as exc:
        return _crash_result(
            root,
            identity=contract.S1_PREFLIGHT_IDENTITY,
            stage="s1-preflight",
            exc=exc,
            hashes=hashes,
            optimizer_steps=0,
            disposable_optimizer_steps=disposable_steps,
            model_writes=0,
        )


def audit_s1_preflight(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S1_PREFLIGHT_ROOT
    try:
        result = read_json(root / "result.json")
        seal = audit_evidence_seal(
            root,
            expected_identity=contract.S1_PREFLIGHT_IDENTITY,
            expected_stage="s1-preflight",
        )
        current_hashes = source_hashes(repo_root)
        predecessor = audit_s0_predecessor(repo_root)
        recorded_hashes = result.get("source_hashes")
        if not isinstance(recorded_hashes, dict):
            raise TypeError("S1 preflight result lacks source hashes")
        snapshot_hashes = {
            relative: sha256_file(root / "source_snapshot" / relative)
            for relative in current_hashes
        }
        gates = dict(result.get("gates", {}))
        formal_schedule_sha = schedule_report(
            build_s1_schedule(
                _s0_records(repo_root),
                seed=contract.S1_ORDER_SEED,
                maximum_updates=contract.S1_MAXIMUM_UPDATES,
            )
        )["sha256"]
        checks = {
            "lease": (repo_root / contract.S1_PREFLIGHT_LEASE).is_file(),
            "identity": result.get("identity") == contract.S1_PREFLIGHT_IDENTITY,
            "status": result.get("status") == "PASS_V2_A_C1U_S1_PREFLIGHT",
            "passed": result.get("passed") is True,
            "stage": result.get("stage") == "s1-preflight",
            "gates": set(gates) == set(contract.S1_PREFLIGHT_GATES)
            and all(gates.values()),
            "authorization": result.get("authorizes")
            == contract.S1_AUTHORIZED_SCOPE,
            "user_authorization": result.get("user_authorization")
            == contract.S1_USER_AUTHORIZATION,
            "formal_zero": result.get("formal_optimizer_steps") == 0,
            "optimizer_zero": result.get("optimizer_steps") == 0,
            "disposable_steps": result.get("disposable_optimizer_steps")
            == contract.S1_PREFLIGHT_BENCHMARK_STEPS,
            "model_writes": result.get("model_writes") == 0,
            "training_not_started": result.get("training_started") is False,
            "source_hashes": result.get("source_hashes") == current_hashes,
            "source_identity": result.get("source_identity")
            == source_identity(current_hashes),
            "source_snapshot": snapshot_hashes == recorded_hashes,
            "formal_schedule": result.get("formal_schedule_sha256")
            == formal_schedule_sha,
            "predecessor": predecessor.get("passed") is True,
            "seal_replay_required": result.get("qualification_requires_seal_replay")
            is True,
            "seal": seal.get("passed") is True,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "result": result,
            "seal": seal,
            "predecessor": predecessor,
            "result_sha256": sha256_file(root / "result.json"),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _evaluation_batch(
    model: C1UModel,
    runtime: C1URuntime,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    provider = RuntimeBatchProvider(runtime)
    ids = [str(row["example_id"]) for row in records]
    cpu = runtime.get_batch(ids)
    batch = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in cpu.items()
    }
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        result = evaluate_s1(model, batch)
    result["device"] = str(device)
    result["dtype"] = "bfloat16_autocast" if device.type == "cuda" else "float32"
    result["provider"] = provider.report()
    return result


def run_s1(
    repo_root: Path,
    *,
    device: str | torch.device = "cuda",
    asset_audit_fn: AuditFn = audit_source_assets,
    device_audit_fn: AuditFn = audit_device,
    config_override: C1UConfig | None = None,
    fault_at: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    preflight = audit_s1_preflight(repo_root)
    paths = audit_s1_unconsumed_paths(repo_root, stage="s1")
    predecessor = audit_s0_predecessor(repo_root)
    assets = dict(asset_audit_fn())
    hardware = dict(device_audit_fn(device))
    hashes = source_hashes(repo_root)
    records, runtime, cache_audit = _runtime_bundle(repo_root)
    task_audit = audit_s1_bank(records)
    schedule = build_s1_schedule(
        records,
        seed=contract.S1_ORDER_SEED,
        maximum_updates=contract.S1_MAXIMUM_UPDATES,
    )
    schedule_audit = schedule_report(schedule)
    preflight_schedule_sha = (
        preflight.get("result", {}).get("formal_schedule_sha256")
        if isinstance(preflight.get("result"), Mapping)
        else None
    )
    preclaim_checks = {
        "preflight": preflight.get("passed") is True,
        "paths": paths["passed"],
        "predecessor": predecessor.get("passed") is True,
        "assets": assets.get("passed") is True,
        "device": hardware.get("passed") is True,
        "task": task_audit["passed"],
        "cache": cache_audit["persistent"]["passed"]
        and cache_audit["runtime"]["passed"],
        "schedule": schedule_audit["updates"] == contract.S1_MAXIMUM_UPDATES
        and schedule_audit["sha256"] == preflight_schedule_sha,
    }
    if not all(preclaim_checks.values()):
        raise RuntimeError(
            f"S1 pre-claim audit failed; fixed S1 root was not consumed: {preclaim_checks}"
        )

    root = repo_root / contract.S1_ROOT
    lease = repo_root / contract.S1_LEASE
    claim_single_use(
        identity=contract.S1_IDENTITY,
        stage="s1",
        output_root=root,
        lease_path=lease,
    )
    optimizer_steps = 0
    model_writes = 0
    try:
        if fault_at == "after_claim":
            raise RuntimeError("injected C1U S1 fault after claim")
        snapshot_sources(repo_root, root)
        write_json(root / "contract-manifest.json", contract.manifest())
        write_json(
            root / "preclaim-audit.json",
            {
                "checks": preclaim_checks,
                "preflight": preflight,
                "paths": paths,
                "predecessor": predecessor,
                "assets": assets,
                "device": hardware,
            },
        )
        write_json(root / "task-bank.json", {"records": records, "audit": task_audit})
        write_json(root / "cache-audit.json", cache_audit)
        write_json(root / "schedule.json", schedule_audit)

        set_determinism(contract.MODEL_SEED)
        config = _model_config(runtime.cache.source_width, config_override)
        model = C1UModel(config)
        initial_integrity = model.integrity_report()
        parameter_report = model.parameter_report()
        provider = RuntimeBatchProvider(runtime)
        spec = TrainSpec()
        with StepJournal(root / "training-progress.jsonl") as journal:
            training = train_fixed_endpoint(
                model,
                provider,
                schedule,
                identity=contract.S1_IDENTITY,
                model_seed=contract.MODEL_SEED,
                order_seed=contract.S1_ORDER_SEED,
                spec=spec,
                device=device,
                on_optimizer_step=journal,
            )
            optimizer_steps = journal.completed
        write_json(root / "training.json", training)
        if fault_at == "after_training":
            raise RuntimeError("injected C1U S1 fault after training")

        endpoint_path = root / f"{contract.S1_FIXED_ENDPOINT}.pt"
        endpoint_sha = save_endpoint(
            endpoint_path,
            model=model,
            identity=contract.S1_IDENTITY,
            update=optimizer_steps,
            schedule_sha256=schedule_audit["sha256"],
        )
        model_writes = 1
        write_json(
            root / "endpoint.json",
            {
                "path": endpoint_path.name,
                "sha256": endpoint_sha,
                "bytes": endpoint_path.stat().st_size,
                "model_writes": model_writes,
                "optimizer_state_saved": False,
                "checkpoint_selection": False,
            },
        )
        del model
        gc.collect()
        active_device = torch.device(device)
        if active_device.type == "cuda":
            torch.cuda.empty_cache()

        endpoint_model, endpoint_payload = load_endpoint(
            endpoint_path, device=active_device
        )
        endpoint_checks = {
            "identity": endpoint_payload.get("identity") == contract.S1_IDENTITY,
            "fixed_endpoint": endpoint_payload.get("fixed_endpoint_name")
            == contract.S1_FIXED_ENDPOINT,
            "update": endpoint_payload.get("update") == contract.S1_MAXIMUM_UPDATES,
            "schedule": endpoint_payload.get("schedule_sha256")
            == schedule_audit["sha256"],
            "optimizer_absent": endpoint_payload.get("optimizer_state_saved") is False,
            "checkpoint_selection_absent": endpoint_payload.get("checkpoint_selection")
            is False,
            "file_hash": sha256_file(endpoint_path) == endpoint_sha,
            "model_integrity": endpoint_model.integrity_report()["passed"],
        }
        evaluation = _evaluation_batch(
            endpoint_model, runtime, records, device=active_device
        )
        write_json(root / "evaluation.json", evaluation)
        qualification = evaluation["qualification"]
        qchecks = dict(qualification["checks"])

        post_hashes = source_hashes(repo_root)
        final_predecessor = audit_s0_predecessor(repo_root)
        final_cache = audit_persisted_cache(_s0_root(repo_root) / "card-cache", records)
        no_core_checks = [
            value
            for name, value in qchecks.items()
            if "no_core" in name
        ]
        support_checks = [
            value
            for name, value in qchecks.items()
            if "support_" in name or "two_contributor" in name
        ]
        result_gates = {
            contract.S1_RESULT_GATES[0]: predecessor["passed"]
            and schedule_audit["sha256"] == preflight_schedule_sha
            and hashes == post_hashes,
            contract.S1_RESULT_GATES[1]: training["passed"]
            and optimizer_steps == contract.S1_MAXIMUM_UPDATES
            and training["completed_updates"] == contract.S1_MAXIMUM_UPDATES
            and model_writes == 1
            and all(endpoint_checks.values()),
            contract.S1_RESULT_GATES[2]: qchecks.get("record_cardinality") is True
            and qchecks.get("family_cardinality") is True
            and qchecks.get("full_answer_exact") is True
            and qchecks.get("factorial_group_exact") is True,
            contract.S1_RESULT_GATES[3]: bool(no_core_checks)
            and all(no_core_checks),
            contract.S1_RESULT_GATES[4]: bool(support_checks)
            and all(support_checks),
            contract.S1_RESULT_GATES[5]: qchecks.get("permutation_invariance") is True
            and initial_integrity["passed"]
            and endpoint_model.integrity_report()["passed"],
            contract.S1_RESULT_GATES[6]: hashes == post_hashes
            and final_predecessor.get("passed") is True
            and final_cache.get("passed") is True
            and sha256_file(endpoint_path) == endpoint_sha,
        }
        passed = qualification["passed"] is True and all(result_gates.values())
        if fault_at == "before_result":
            raise RuntimeError("injected C1U S1 fault before result")
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s1-result.v1",
            "identity": contract.S1_IDENTITY,
            "stage": "s1",
            "status": "PASS_V2_A_C1U_S1_OVERFIT32"
            if passed
            else "FAIL_V2_A_C1U_S1_QUALIFICATION",
            "passed": passed,
            "authorizes": contract.S1_PASS_AUTHORIZATION if passed else "nothing",
            "never_authorizes": list(contract.S1_NEVER_AUTHORIZES),
            "gates": result_gates,
            "qualification": qualification,
            "training": training,
            "endpoint": {
                "path": endpoint_path.name,
                "sha256": endpoint_sha,
                "checks": endpoint_checks,
            },
            "model_config": asdict(config),
            "parameter_report": parameter_report,
            "source_hashes": hashes,
            "source_identity": source_identity(hashes),
            "s0_result_sha256": predecessor["s0_result_sha256"],
            "s0_seal_sha256": predecessor["s0_seal"]["seal_sha256"],
            "s0_card_cache_sha256": contract.S0_CARD_CACHE_SHA256,
            "schedule_sha256": schedule_audit["sha256"],
            "optimizer_steps": optimizer_steps,
            "formal_optimizer_steps": optimizer_steps,
            "disposable_optimizer_steps": 0,
            "model_writes": model_writes,
            "checkpoint_selection": False,
            "intermediate_checkpoints": 0,
            "training_started": True,
            "unrun_successors": ["S2", "S3", "single-seed formal"]
            if not passed
            else ["S2 training", "S3", "single-seed formal"],
            "qualification_requires_seal_replay": True,
        }
        write_json(root / "result.json", result)
        write_evidence_seal(root, identity=contract.S1_IDENTITY, stage="s1")
        return {
            **result,
            "seal_replay": audit_evidence_seal(
                root, expected_identity=contract.S1_IDENTITY, expected_stage="s1"
            ),
        }
    except Exception as exc:
        return _crash_result(
            root,
            identity=contract.S1_IDENTITY,
            stage="s1",
            exc=exc,
            hashes=hashes,
            optimizer_steps=optimizer_steps,
            disposable_optimizer_steps=0,
            model_writes=model_writes,
        )


def audit_s1_root(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S1_ROOT
    try:
        result = read_json(root / "result.json")
        seal = audit_evidence_seal(
            root, expected_identity=contract.S1_IDENTITY, expected_stage="s1"
        )
        current_hashes = source_hashes(repo_root)
        endpoint = result.get("endpoint", {})
        endpoint_path = root / str(endpoint.get("path", ""))
        recorded_hashes = result.get("source_hashes")
        if not isinstance(recorded_hashes, dict):
            raise TypeError("S1 result lacks source hashes")
        status = str(result.get("status", ""))
        crashed = status == "CRASH_V2_A_C1U_S1"
        passed_terminal = status == "PASS_V2_A_C1U_S1_OVERFIT32"
        failed_terminal = status == "FAIL_V2_A_C1U_S1_QUALIFICATION"
        snapshot_paths = {
            relative: root / "source_snapshot" / relative for relative in current_hashes
        }
        snapshot_complete = all(path.is_file() for path in snapshot_paths.values())
        snapshot_hashes = (
            {relative: sha256_file(path) for relative, path in snapshot_paths.items()}
            if snapshot_complete
            else {}
        )
        gates = dict(result.get("gates", {}))
        terminal_coherent = (
            passed_terminal
            and result.get("passed") is True
            and result.get("authorizes") == contract.S1_PASS_AUTHORIZATION
            and set(gates) == set(contract.S1_RESULT_GATES)
            and all(gates.values())
        ) or (
            failed_terminal
            and result.get("passed") is False
            and result.get("authorizes") == "nothing"
            and set(gates) == set(contract.S1_RESULT_GATES)
            and not all(gates.values())
        ) or (
            crashed
            and result.get("passed") is False
            and result.get("authorizes") == "nothing"
        )
        checks = {
            "lease": (repo_root / contract.S1_LEASE).is_file(),
            "identity": result.get("identity") == contract.S1_IDENTITY,
            "terminal_status": passed_terminal or failed_terminal or crashed,
            "terminal_coherence": terminal_coherent,
            "stage": result.get("stage") == "s1",
            "source_hashes": result.get("source_hashes") == current_hashes,
            "source_identity": result.get("source_identity")
            == source_identity(current_hashes),
            "source_snapshot": (crashed and not snapshot_complete)
            or (snapshot_complete and snapshot_hashes == recorded_hashes),
            "optimizer_accounting": result.get("optimizer_steps")
            == result.get("formal_optimizer_steps"),
            "fixed_accounting": crashed
            or (
                result.get("formal_optimizer_steps") == contract.S1_MAXIMUM_UPDATES
                and result.get("disposable_optimizer_steps") == 0
                and result.get("model_writes") == 1
                and result.get("checkpoint_selection") is False
                and result.get("intermediate_checkpoints") == 0
            ),
            "endpoint": (
                result.get("status") == "CRASH_V2_A_C1U_S1"
                and not endpoint
            )
            or (
                endpoint_path.is_file()
                and sha256_file(endpoint_path) == endpoint.get("sha256")
            ),
            "predecessor": audit_s0_predecessor(repo_root).get("passed") is True,
            "seal_replay_required": crashed
            or result.get("qualification_requires_seal_replay") is True,
            "seal": seal.get("passed") is True,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "result": result,
            "seal": seal,
            "result_sha256": sha256_file(root / "result.json"),
        }
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "audit_s0_predecessor",
    "audit_s1_preflight",
    "audit_s1_root",
    "audit_s1_unconsumed_paths",
    "run_registered_tests",
    "run_s1",
    "run_s1_preflight",
]
