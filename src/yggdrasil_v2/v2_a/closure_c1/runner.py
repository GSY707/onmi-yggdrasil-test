from __future__ import annotations

"""Single-use C1 learner preflight and single-seed qualification runner."""

from datetime import UTC, datetime
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from . import artifacts, contract
from .cache import CachedShardDataset, audit_cache
from .deployment import parameter_graph_report, reload_stripped, strip_trace_probe
from .evaluate import evaluate_behavior, evaluate_causal, evaluate_paired_intervention
from .evaluation_runtime import C1EvaluationRuntime
from .model import C1Config, C1Model
from .qualification import (
    gate_g004_validation,
    gate_g005_ood,
    gate_g006_causal,
    gate_g007_trace,
    gate_g008_hidden,
    gate_g009_recurrence,
    gate_g010_integrity,
)
from .runtime import (
    C1BatchProvider,
    OfflineRecordStore,
    TraceTargetBank,
    audit_target_bank,
    load_offline_records,
    load_target_bank,
    validate_cache_source_identity,
)
from .train import (
    build_balanced_schedule,
    schedule_report,
    select_trace_checkpoint,
    set_determinism,
    train_overfit32,
    train_primary,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_rank(example_id: str, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}|{example_id}".encode("utf-8")).hexdigest()


def _fresh_model(*, seed: int, with_trace_probe: bool = True) -> C1Model:
    set_determinism(seed)
    return C1Model(C1Config(**contract.MODEL_CONFIG), with_trace_probe=with_trace_probe)


def _cache_root(repo_root: Path) -> Path:
    return repo_root / contract.CACHE_OUTPUT_ROOT


def _formal_preflight_audit(repo_root: Path) -> dict[str, Any]:
    current = artifacts.source_identity(artifacts.source_hashes(repo_root))
    audit = artifacts.audit_preflight_root(
        repo_root / contract.C1_PREFLIGHT_ROOT,
        identity=f"{contract.C1_IDENTITY}-PREFLIGHT",
        status="PASS_V2_A_C1_PREFLIGHT",
        source_identity_value=current,
    )
    production_checks = {
        "device": audit.get("result", {}).get("device") == "cuda",
        "trace64": audit.get("result", {}).get("forward_backward", {}).get("64")
        is True,
        "trace512": audit.get("result", {}).get("forward_backward", {}).get("512")
        is True,
    }
    audit["production_checks"] = production_checks
    audit["passed"] = audit.get("passed") is True and all(production_checks.values())
    return audit


def _hidden_root(repo_root: Path) -> Path:
    return _cache_root(repo_root) / "hidden"


def _target_bank_path(repo_root: Path) -> Path:
    return _cache_root(repo_root) / "trace-target-bank.json"


def _pinned_inputs(repo_root: Path) -> dict[str, Any]:
    c0r_data = artifacts.audit_pinned_root(
        repo_root / contract.C0R_DATA_ROOT,
        result_sha256=contract.C0R_DATA_RESULT_SHA256,
        seal_sha256=contract.C0R_DATA_SEAL_SHA256,
    )
    c0r_readiness = artifacts.audit_pinned_root(
        repo_root / contract.C0R_READINESS_ROOT,
        result_sha256=contract.C0R_READINESS_RESULT_SHA256,
        seal_sha256=contract.C0R_READINESS_SEAL_SHA256,
    )
    cache: dict[str, Any]
    if contract.CACHE_RESULT_SHA256 is None or contract.CACHE_SEAL_SHA256 is None:
        cache = {
            "passed": False,
            "reason": "cache result/seal pins are unset in frozen C1 contract",
        }
    else:
        cache = artifacts.audit_pinned_root(
            _cache_root(repo_root),
            result_sha256=contract.CACHE_RESULT_SHA256,
            seal_sha256=contract.CACHE_SEAL_SHA256,
        )
    return {
        "c0r_data": c0r_data,
        "c0r_readiness": c0r_readiness,
        "cache": cache,
        "passed": all(
            value.get("passed") is True
            for value in (c0r_data, c0r_readiness, cache)
        ),
    }


def _load_runtime(repo_root: Path) -> tuple[OfflineRecordStore, TraceTargetBank, CachedShardDataset, C1BatchProvider]:
    store = load_offline_records(repo_root / contract.C0R_DATA_ROOT)
    bank = load_target_bank(_target_bank_path(repo_root))
    dataset = CachedShardDataset(_hidden_root(repo_root))
    identity = validate_cache_source_identity(dataset, store)
    target_audit = audit_target_bank(bank)
    hidden_audit = audit_cache(_hidden_root(repo_root), store.records)
    if identity.get("passed") is not True:
        raise ValueError(f"cache/source identity failed: {identity.get('failures')}")
    if target_audit.get("passed") is not True or len(bank) != len(store):
        raise ValueError(f"trace target bank failed: {target_audit.get('failures')}")
    if hidden_audit.get("passed") is not True:
        raise ValueError(f"hidden cache audit failed: {hidden_audit.get('failures')}")
    provider = C1BatchProvider(dataset, store, bank)
    return store, bank, dataset, provider


def _records(store: OfflineRecordStore, *, split: str, family: str | None = None) -> list[dict[str, Any]]:
    rows = [row for row in store.records if row.get("split") == split]
    if family is not None:
        rows = [row for row in rows if row.get("family") == family]
    return rows


def _overfit32_ids(store: OfflineRecordStore) -> list[str]:
    selected: list[str] = []
    for family in ("ERE", "CPS"):
        rows = _records(store, split="train", family=family)
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            depth = int(row.get("reasoning_budget", 0))
            by_depth.setdefault(depth, []).append(row)
        for depth_rows in by_depth.values():
            depth_rows.sort(
                key=lambda row: _hash_rank(
                    str(row["example_id"]), f"C1-OVERFIT32-{family}"
                )
            )
        depths = sorted(by_depth)
        if not depths:
            raise ValueError(f"no train depths for {family}")
        family_ids: list[str] = []
        cursor = {depth: 0 for depth in depths}
        while len(family_ids) < 16:
            progressed = False
            for depth in depths:
                index = cursor[depth]
                if index < len(by_depth[depth]):
                    family_ids.append(str(by_depth[depth][index]["example_id"]))
                    cursor[depth] += 1
                    progressed = True
                    if len(family_ids) == 16:
                        break
            if not progressed:
                raise ValueError(f"insufficient overfit32 rows for {family}")
        selected.extend(family_ids)
    if len(selected) != 32 or len(set(selected)) != 32:
        raise ValueError("overfit32 selection is not 32 unique records")
    return selected


def _model_contract_report(model: C1Model) -> dict[str, Any]:
    signature = inspect.signature(model.forward)
    accepted = list(signature.parameters)
    forbidden_forward = sorted(set(accepted) & set(contract.FORBIDDEN_FORWARD_FIELDS))
    parameter_names = [name.lower() for name, _ in model.named_parameters()]
    forbidden_parameters = sorted(
        name
        for name in parameter_names
        if any(token in name for token in contract.FORBIDDEN_COMPONENT_TOKENS)
    )
    integrity = model.integrity_report()
    optimizer_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        "forward_parameters": accepted,
        "source_mask_exception": accepted[:2] == ["source_hidden", "source_mask"],
        "forbidden_forward_parameters": forbidden_forward,
        "forbidden_parameter_names": forbidden_parameters,
        "trainable_parameter_names": len(optimizer_names),
        "integrity": integrity,
        "passed": integrity.get("passed") is True
        and not forbidden_forward
        and not forbidden_parameters
        and accepted[:2] == ["source_hidden", "source_mask"],
    }


def _overfit_callback(
    dataset: CachedShardDataset,
    rows: Sequence[Mapping[str, Any]],
    bank: TraceTargetBank,
    *,
    device: str,
):
    def callback(model: C1Model, update: int) -> dict[str, Any]:
        runtime = C1EvaluationRuntime(model, dataset, rows, device=device, batch_size=8)
        behavior = runtime.evaluate_answers(records=list(rows))
        trace = runtime.trace_credit(
            bank.targets,
            validation_records=list(rows),
            per_family=16,
        )
        families = trace["families"]
        return {
            "update": update,
            "answer_accuracy": float(behavior["overall"]["point"]),
            "trace_token_accuracy": min(
                float(families[family]["all_token_accuracy"]["point"])
                for family in ("ERE", "CPS")
            ),
            "trace_content_accuracy": min(
                float(families[family]["content_token_accuracy"]["point"])
                for family in ("ERE", "CPS")
            ),
            "owner_nll_margin": min(
                float(families[family]["nll_margin"]["point"])
                for family in ("ERE", "CPS")
            ),
            "behavior": behavior,
            "trace": trace,
        }

    return callback


def _selection_callback(
    dataset: CachedShardDataset,
    validation: Sequence[Mapping[str, Any]],
    bank: TraceTargetBank,
    *,
    device: str,
):
    def callback(model: C1Model, update: int) -> dict[str, Any]:
        runtime = C1EvaluationRuntime(
            model, dataset, validation, device=device, batch_size=8
        )
        return {
            "update": update,
            "trace_nll_by_family": runtime.trace_nll_by_family(
                bank.targets,
                validation_records=validation,
                per_family=int(
                    contract.TRAINING_CONFIG["validation_trace_records_per_family"]
                ),
            ),
            "selection_uses_answer": False,
            "selection_uses_ood_or_causal": False,
        }

    return callback


def _lookup_predictor(logits: Mapping[str, Sequence[float]]):
    return lambda row: {"logits": logits[str(row["example_id"])]}


def _paired_report(
    rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Sequence[float]],
    changed: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return evaluate_paired_intervention(
        rows,
        _lookup_predictor(baseline),
        _lookup_predictor(changed),
        cluster_key="example_id",
    )


def _invariance_report(
    before: Mapping[str, Sequence[float]],
    after: Mapping[str, Sequence[float]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    if set(before) != set(after) or not before:
        raise ValueError("invariance logits have different or empty identities")
    same = 0
    maximum = 0.0
    for example_id in sorted(before):
        left = [float(value) for value in before[example_id]]
        right = [float(value) for value in after[example_id]]
        if len(left) != 9 or len(right) != 9:
            raise ValueError("invariance requires raw nine-class logits")
        same += int(max(range(9), key=left.__getitem__) == max(range(9), key=right.__getitem__))
        maximum = max(maximum, max(abs(a - b) for a, b in zip(left, right)))
    count = len(before)
    return {
        "n": count,
        "prediction_invariance": same / count,
        "max_abs_diff": maximum,
        "tolerance": tolerance,
        "passed": same == count and maximum <= tolerance,
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_all_finite(item) for item in value)
    return False


def _trace_exposure_ledger(
    schedule: Sequence[Any],
    provider: C1BatchProvider,
    dataset: CachedShardDataset,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    by_epoch: dict[int, list[str]] = defaultdict(list)
    trace_tokens = 0
    source_tokens = 0
    chunk_tokens = int(contract.TRAINING_CONFIG["trace_chunk_tokens"])
    for batch in schedule:
        for example_id in batch.example_ids:
            chunk = provider.trace_chunk_metadata(
                example_id,
                epoch=int(batch.epoch),
                trace_chunk_tokens=chunk_tokens,
            )
            source_length = int(dataset.locations[str(example_id)][3])
            entry = {
                "update": int(batch.update),
                "epoch": int(batch.epoch),
                **chunk,
                "source_tokens": source_length,
            }
            rows.append(entry)
            counts[str(example_id)] += 1
            by_epoch[int(batch.epoch)].append(str(example_id))
            trace_tokens += int(chunk["exposed_tokens"])
            source_tokens += source_length
    checks = {
        "rows": len(rows) == 49_152,
        "unique_examples": len(counts) == 8_192,
        "six_exposures_each": bool(counts)
        and set(counts.values()) == {int(contract.TRAINING_CONFIG["epochs"])},
        "six_epochs": set(by_epoch) == set(range(6)),
        "one_per_epoch": all(
            len(ids) == 8_192 and len(set(ids)) == 8_192
            for ids in by_epoch.values()
        ),
        "chunk_cap": all(
            1 <= int(row["exposed_tokens"]) <= chunk_tokens for row in rows
        ),
        "source_tokens_positive": source_tokens > 0,
        "trace_tokens_positive": trace_tokens > 0,
    }
    digest = hashlib.sha256(artifacts.canonical_json_bytes(rows)).hexdigest().upper()
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.trace-exposure-ledger.v1",
        "checks": checks,
        "passed": all(checks.values()),
        "updates": len(schedule),
        "rows_count": len(rows),
        "unique_examples": len(counts),
        "exposures_per_example": dict(sorted(Counter(counts.values()).items())),
        "processed_source_tokens": source_tokens,
        "processed_trace_tokens": trace_tokens,
        "sha256": digest,
        "rows": rows,
    }


def _primary_training_checks(
    training: Mapping[str, Any],
    *,
    root: Path,
    exposure: Mapping[str, Any],
    model: C1Model,
) -> dict[str, Any]:
    candidates = training.get("candidates")
    selected = training.get("selected")
    candidate_rows = list(candidates) if isinstance(candidates, Sequence) else []
    selected_row = dict(selected) if isinstance(selected, Mapping) else {}
    expected_updates = list(range(512, 6145, 512))
    actual_updates = [int(row.get("update", -1)) for row in candidate_rows]
    selected_path = root / str(selected_row.get("checkpoint", "missing"))
    selected_hash = (
        artifacts.sha256_file(selected_path) if selected_path.is_file() else None
    )
    recomputed: Mapping[str, Any] | None = None
    try:
        recomputed = select_trace_checkpoint(
            candidate_rows,
            minimum_update=int(contract.TRAINING_CONFIG["minimum_selection_update"]),
        )
    except (KeyError, TypeError, ValueError):
        recomputed = None
    parameters_finite = all(
        bool(torch.isfinite(parameter.detach()).all().item())
        for parameter in model.parameters()
    )
    checks = {
        "training_passed": training.get("passed") is True,
        "updates": type(training.get("completed_updates")) is int
        and training.get("completed_updates") == 6144,
        "processed_examples": type(training.get("processed_examples")) is int
        and training.get("processed_examples") == 49_152,
        "candidate_updates": actual_updates == expected_updates,
        "selection_rule": training.get("selection_rule")
        == "minimize max(ERE trace NLL, CPS trace NLL); tie earliest",
        "selection_recomputed": recomputed is not None
        and dict(recomputed) == selected_row,
        "selected_checkpoint_exists": selected_path.is_file(),
        "selected_checkpoint_hash": selected_hash
        == selected_row.get("checkpoint_sha256"),
        "finite_training_ledger": _all_finite(training),
        "finite_selected_parameters": parameters_finite,
        "exposure_ledger": exposure.get("passed") is True,
        "source_token_accounting": training.get("processed_source_tokens")
        == exposure.get("processed_source_tokens"),
        "trace_token_accounting": training.get("processed_trace_tokens")
        == exposure.get("processed_trace_tokens"),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "selected_checkpoint_path": selected_path.as_posix(),
        "selected_checkpoint_sha256": selected_hash,
        "parameters_finite": parameters_finite,
    }


def _accounting(
    *,
    model: C1Model,
    training: Mapping[str, Any],
    cache_root: Path,
    deployment_path: Path,
    schedule: Mapping[str, Any],
    exposure: Mapping[str, Any],
    primary_checks: Mapping[str, Any],
    validation_timing: Mapping[str, Any],
    deployment_expected_sha256: str,
) -> dict[str, Any]:
    hidden_manifest = json.loads(
        (cache_root / "hidden" / "manifest.json").read_text(encoding="utf-8")
    )
    cache_result = json.loads(
        (cache_root / "result.json").read_text(encoding="utf-8")
    )
    parameter_report = model.parameter_report()
    deployment_sha256 = (
        artifacts.sha256_file(deployment_path) if deployment_path.is_file() else None
    )
    selected = training.get("selected") if isinstance(training.get("selected"), Mapping) else {}
    selected_path = deployment_path.parents[1] / str(selected.get("checkpoint", "missing"))
    selected_sha256 = (
        artifacts.sha256_file(selected_path) if selected_path.is_file() else None
    )
    hidden_bytes = sum(
        path.stat().st_size
        for path in (cache_root / "hidden").glob("shard_*")
        if path.is_file()
    )
    cache_build_seconds = hidden_manifest.get("build_seconds")
    cache_wall_seconds = cache_result.get("wall_seconds")
    cache_result_sha256 = artifacts.sha256_file(cache_root / "result.json")
    cache_seal_sha256 = artifacts.sha256_file(cache_root / "evidence-seal.json")
    validation_seconds = validation_timing.get("seconds")
    validation_examples = validation_timing.get("examples")
    latency_ms = validation_timing.get("milliseconds_per_example")
    required = {
        "completed_updates": int(training.get("completed_updates", -1)) == 6144,
        "processed_examples": int(training.get("processed_examples", -1)) == 49_152,
        "schedule_updates": int(schedule.get("updates", -1)) == 6144,
        "schedule_examples": int(schedule.get("examples", -1)) == 49_152,
        "schedule_epochs": int(schedule.get("epochs", -1)) == 6,
        "exposure_ledger": exposure.get("passed") is True,
        "processed_source_tokens": type(training.get("processed_source_tokens")) is int
        and training.get("processed_source_tokens")
        == exposure.get("processed_source_tokens")
        and int(training.get("processed_source_tokens", 0)) > 0,
        "processed_trace_tokens": type(training.get("processed_trace_tokens")) is int
        and training.get("processed_trace_tokens")
        == exposure.get("processed_trace_tokens")
        and int(training.get("processed_trace_tokens", 0)) > 0,
        "cache_examples": int(hidden_manifest.get("examples", -1)) == 26_624,
        "cache_tokens_exact": type(hidden_manifest.get("source_tokens")) is int
        and hidden_manifest.get("source_tokens") == cache_result.get("source_tokens")
        and int(hidden_manifest.get("source_tokens", 0)) > 0,
        "cache_bytes_exact": hidden_bytes == cache_result.get("cache_bytes")
        and hidden_bytes > 0,
        "finite_cache_build_time": isinstance(cache_build_seconds, (int, float))
        and not isinstance(cache_build_seconds, bool)
        and math.isfinite(float(cache_build_seconds))
        and float(cache_build_seconds) > 0,
        "finite_cache_wall_time": isinstance(cache_wall_seconds, (int, float))
        and not isinstance(cache_wall_seconds, bool)
        and math.isfinite(float(cache_wall_seconds))
        and float(cache_wall_seconds) > 0,
        "deployment_exists": deployment_path.is_file(),
        "deployment_hash": isinstance(deployment_sha256, str)
        and len(deployment_sha256) == 64
        and deployment_sha256 == deployment_expected_sha256,
        "selected_checkpoint_exists": selected_path.is_file(),
        "selected_checkpoint_hash": selected_sha256
        == selected.get("checkpoint_sha256")
        == primary_checks.get("selected_checkpoint_sha256"),
        "primary_checks": primary_checks.get("passed") is True,
        "active_parameters": int(parameter_report.get("deployment_parameters", 0)) > 0,
        "trainable_parameters": int(parameter_report.get("trainable_parameters", 0))
        == int(parameter_report.get("deployment_parameters", -1)),
        "zero_trace_probe_parameters": parameter_report.get("trace_probe_parameters") == 0,
        "finite_wall": isinstance(training.get("wall_seconds"), (int, float))
        and not isinstance(training.get("wall_seconds"), bool)
        and math.isfinite(float(training.get("wall_seconds")))
        and float(training.get("wall_seconds")) > 0,
        "finite_throughput": isinstance(training.get("updates_per_second"), (int, float))
        and not isinstance(training.get("updates_per_second"), bool)
        and math.isfinite(float(training.get("updates_per_second")))
        and float(training.get("updates_per_second")) > 0,
        "finite_peak_vram": int(training.get("peak_vram_bytes", -1)) > 0,
        "finite_validation_latency": isinstance(validation_seconds, (int, float))
        and isinstance(latency_ms, (int, float))
        and type(validation_examples) is int
        and validation_examples == 3_072
        and math.isfinite(float(validation_seconds))
        and math.isfinite(float(latency_ms))
        and float(validation_seconds) > 0
        and float(latency_ms) > 0,
        "latent_transitions": int(training.get("processed_examples", 0)) * 10
        == 491_520,
        "cache_result_pin": cache_result_sha256 == contract.CACHE_RESULT_SHA256,
        "cache_seal_pin": cache_seal_sha256 == contract.CACHE_SEAL_SHA256,
    }
    return {
        "checks": required,
        "passed": all(required.values()),
        "cache": {
            "examples": hidden_manifest.get("examples"),
            "source_tokens": hidden_manifest.get("source_tokens"),
            "bytes": hidden_bytes,
            "build_seconds": cache_build_seconds,
            "wall_seconds": cache_wall_seconds,
            "result_sha256": cache_result_sha256,
            "seal_sha256": cache_seal_sha256,
        },
        "learner_parameters": parameter_report,
        "updates": training.get("completed_updates"),
        "processed_examples": training.get("processed_examples"),
        "processed_source_tokens": training.get("processed_source_tokens"),
        "processed_trace_tokens": training.get("processed_trace_tokens"),
        "latent_transitions": int(training.get("processed_examples", 0)) * 10,
        "wall_seconds": training.get("wall_seconds"),
        "peak_vram_bytes": training.get("peak_vram_bytes"),
        "validation_timing": dict(validation_timing),
        "selected_checkpoint_path": selected_path.as_posix(),
        "selected_checkpoint_sha256": selected_sha256,
        "deployment_sha256": deployment_sha256,
        "exposure_ledger_sha256": exposure.get("sha256"),
        "terminal_seal_mode": "non-recursive seal is written and replayed by _finish after result.json",
    }


def _finish(
    root: Path,
    *,
    result: dict[str, Any],
    started_at: str,
    wall_started: float,
    source_identity: str | None,
    source_stable: bool,
) -> dict[str, Any]:
    artifacts.write_json(root / "result.json", result)
    artifacts.write_json(
        root / "run-state.json",
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.run-state.v1",
            "identity": contract.C1_IDENTITY,
            "formal": bool(result.get("formal")),
            "started_at": started_at,
            "finished_at": _now(),
            "wall_seconds": time.perf_counter() - wall_started,
            "source_identity": source_identity,
            "source_stable": source_stable,
            "training_started": bool(result.get("training_started")),
            "optimizer_steps": int(result.get("optimizer_steps", 0)),
            "model_writes": int(result.get("model_writes", 0)),
        },
    )
    seal = artifacts.write_evidence_seal(root)
    replay = artifacts.audit_evidence_seal(root)
    if replay.get("passed") is not True:
        # A PASS is never allowed to survive an un-replayable seal.  Rewrite
        # the terminal result as INCOMPLETE and seal that failure evidence.
        seal_failure_status = (
            "INCOMPLETE_V2_A_C1_PREFLIGHT"
            if str(result.get("identity", "")).endswith("-PREFLIGHT")
            else "INCOMPLETE_V2_A_C1_SINGLE_SEED_ELIGIBILITY"
        )
        result = {
            **result,
            "status": seal_failure_status,
            "passed": False,
            "exit_code": 1,
            "authorizes": "nothing",
            "seal_failure": replay,
        }
        if isinstance(result.get("gates"), Mapping):
            result["gates"] = {**result["gates"], "G011": False}
        if isinstance(result.get("gate_status"), Mapping):
            result["gate_status"] = {**result["gate_status"], "G011": "failed"}
        artifacts.write_json(root / "seal-failure.json", replay)
        artifacts.write_json(root / "result.json", result)
        seal = artifacts.write_evidence_seal(root)
        replay = artifacts.audit_evidence_seal(root)
    return {
        **result,
        "output_root": root.as_posix(),
        "evidence_seal_sha256": seal,
        "seal_replay": replay,
    }


def run_c1_preflight(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Single-use real-cache forward/backward smoke; never trains a learner."""
    repo_root = Path(repo_root).resolve()
    root = output_root or repo_root / contract.C1_PREFLIGHT_ROOT
    lease = lease_path or repo_root / contract.C1_PREFLIGHT_LEASE
    if root.exists() or lease.exists():
        return artifacts.refusal(
            identity=f"{contract.C1_IDENTITY}-PREFLIGHT",
            output_root=root,
            lease_path=lease,
            status="REFUSE_V2_A_C1_PREFLIGHT_SINGLE_USE",
        )
    if root.resolve() == (repo_root / contract.C1_PREFLIGHT_ROOT).resolve() and str(device) != "cuda":
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-refusal.v1",
            "identity": f"{contract.C1_IDENTITY}-PREFLIGHT",
            "status": "REFUSE_V2_A_C1_PREFLIGHT_DEVICE",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "fixed C1 preflight requires CUDA",
            "authorizes": "nothing",
        }
    if root.resolve() == (repo_root / contract.C1_PREFLIGHT_ROOT).resolve() and not torch.cuda.is_available():
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-refusal.v1",
            "identity": f"{contract.C1_IDENTITY}-PREFLIGHT",
            "status": "REFUSE_V2_A_C1_PREFLIGHT_DEVICE",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "fixed C1 preflight requires an available CUDA runtime",
            "authorizes": "nothing",
        }
    started_at = _now()
    wall_started = time.perf_counter()
    source_identity: str | None = None
    source_stable = False
    try:
        artifacts.claim_single_use(
            identity=f"{contract.C1_IDENTITY}-PREFLIGHT",
            output_root=root,
            lease_path=lease,
            formal=False,
        )
        source_before = artifacts.source_hashes(repo_root)
        source_identity = artifacts.source_identity(source_before)
        artifacts.snapshot_sources(repo_root, root)
        prerequisites = _pinned_inputs(repo_root)
        store, bank, dataset, provider = _load_runtime(repo_root)
        train_rows = _records(store, split="train")
        longest = sorted(
            train_rows,
            key=lambda row: int(dataset.locations[str(row["example_id"])][3]),
            reverse=True,
        )[:8]
        model = _fresh_model(seed=contract.MODEL_SEED).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        reports: dict[str, Any] = {}
        for trace_tokens in (64, 512):
            batch = provider(
                [str(row["example_id"]) for row in longest],
                epoch=0,
                trace_chunk_tokens=trace_tokens,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch["source_hidden"],
                    batch["source_mask"],
                    return_trajectory=True,
                )
                trace_logits = model.trace_logits(
                    output["trajectory"],
                    batch["trace_step_indices"],
                    batch["trace_global_positions"],
                    batch["trace_local_positions"],
                )
                mask = batch["trace_mask"]
                loss = torch.nn.functional.cross_entropy(
                    output["logits"], batch["answers"]
                ) + torch.nn.functional.cross_entropy(
                    trace_logits[mask], batch["trace_targets"][mask]
                )
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            reports[str(trace_tokens)] = {
                "loss": float(loss.detach().float().cpu()),
                "finite_loss": bool(torch.isfinite(loss).item()),
                "parameters_with_gradient": sum(value is not None for value in gradients),
                "trainable_parameter_tensors": len(gradients),
                "finite_gradients": all(
                    value is not None and bool(torch.isfinite(value).all().item())
                    for value in gradients
                ),
                "source_shape": list(batch["source_hidden"].shape),
                "trace_shape": list(batch["trace_targets"].shape),
            }
        strip = strip_trace_probe(model)
        contract_report = _model_contract_report(model)
        source_after = artifacts.source_hashes(repo_root)
        source_stable = source_before == source_after
        passed = (
            prerequisites["passed"]
            and contract_report["passed"]
            and source_stable
            and strip.report.get("trace_probe_parameters") == 0
            and all(
                row["finite_loss"]
                and row["finite_gradients"]
                and row["parameters_with_gradient"]
                == row["trainable_parameter_tensors"]
                for row in reports.values()
            )
        )
        artifacts.write_json(root / "prerequisites.json", prerequisites)
        artifacts.write_json(root / "model-contract.json", contract_report)
        artifacts.write_json(root / "forward-backward.json", reports)
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": f"{contract.C1_IDENTITY}-PREFLIGHT",
            "formal": False,
            "device": device,
            "status": "PASS_V2_A_C1_PREFLIGHT" if passed else "FAIL_V2_A_C1_PREFLIGHT",
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "source_identity": source_identity,
            "source_stable": source_stable,
            "authorizes": "formal C1 single-seed launch only" if passed else "nothing",
            "forward_backward": {
                key: value["finite_loss"]
                and value["finite_gradients"]
                and value["parameters_with_gradient"]
                == value["trainable_parameter_tensors"]
                for key, value in reports.items()
            },
        }
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-crash.v1",
            "identity": f"{contract.C1_IDENTITY}-PREFLIGHT",
            "formal": False,
            "status": "CRASH_V2_A_C1_PREFLIGHT",
            "passed": False,
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "authorizes": "nothing",
        }
        artifacts.write_json(root / "crash.json", result)
    return _finish(
        root,
        result=result,
        started_at=started_at,
        wall_started=wall_started,
        source_identity=source_identity,
        source_stable=source_stable,
    )


def run_c1_single_seed(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    formal: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the frozen C1 single-seed stop tree exactly once."""
    repo_root = Path(repo_root).resolve()
    root = output_root or repo_root / contract.C1_OUTPUT_ROOT
    lease = lease_path or repo_root / contract.C1_LEASE_PATH
    if root.exists() or lease.exists():
        return artifacts.refusal(
            identity=contract.C1_IDENTITY,
            output_root=root,
            lease_path=lease,
            status="REFUSE_V2_A_C1_SINGLE_USE",
        )
    if not formal and root.resolve() == (repo_root / contract.C1_OUTPUT_ROOT).resolve():
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.refusal.v1",
            "identity": contract.C1_IDENTITY,
            "status": "REFUSE_V2_A_C1_NONFORMAL_FIXED_ROOT",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "formal C1 root cannot be consumed by a nonformal run",
            "authorizes": "nothing",
            "c2_authorized": False,
            "v2a_passed": False,
        }
    if formal and (str(device) != "cuda" or not torch.cuda.is_available()):
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.refusal.v1",
            "identity": contract.C1_IDENTITY,
            "status": "REFUSE_V2_A_C1_FORMAL_DEVICE",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "formal C1 qualification requires available CUDA BF16 execution",
            "authorizes": "nothing",
            "c2_authorized": False,
            "v2a_passed": False,
        }
    preflight: dict[str, Any] = {"passed": True, "required": False}
    if formal:
        preflight = _formal_preflight_audit(repo_root)
        if preflight.get("passed") is not True:
            return {
                "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-refusal.v1",
                "identity": contract.C1_IDENTITY,
                "status": "REFUSE_V2_A_C1_PREFLIGHT_PREREQUISITE",
                "passed": False,
                "exit_code": 2,
                "output_root": root.as_posix(),
                "lease_path": lease.as_posix(),
                "reason": "sealed C1 preflight PASS for the current source identity is required",
                "preflight_audit": preflight,
                "authorizes": "nothing",
                "c2_authorized": False,
                "v2a_passed": False,
            }
    started_at = _now()
    wall_started = time.perf_counter()
    source_identity: str | None = None
    source_stable = False
    training_started = False
    optimizer_steps = 0
    model_writes = 0
    gate_status = {gate: "not_run" for gate in contract.C1_GATE_IDS}
    stage_status = {
        stage: "not_run"
        for stage in (
            "G001",
            "G002_OVERFIT32",
            "PRIMARY_TRAIN_SELECT",
            "G007_TRACE_CREDIT",
            "G003_STRIP_RELOAD",
            "G004_VALIDATION",
            "G005_OOD",
            "G006_CAUSAL",
            "G008_HIDDEN",
            "G009_RECURRENCE",
            "G010_SLOT_INTEGRITY",
            "G011_ACCOUNTING",
        )
    }
    source_before: dict[str, str] = {}
    primary: dict[str, Any] = {}
    deployment_path = root / "deployment" / "stripped-state.pt"

    try:
        artifacts.claim_single_use(
            identity=contract.C1_IDENTITY,
            output_root=root,
            lease_path=lease,
            formal=formal,
        )
        source_before = artifacts.source_hashes(repo_root)
        source_identity = artifacts.source_identity(source_before)
        artifacts.snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.c1_contract_manifest())

        def terminal(
            status: str,
            *,
            stopped_at: str | None,
            extra: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            nonlocal source_stable
            source_after = artifacts.source_hashes(repo_root)
            source_stable = source_before == source_after
            gates = {
                gate: state == "passed" for gate, state in gate_status.items()
            }
            diagnostic_passed = (
                status == "PASS_V2_A_C1_SINGLE_SEED_ELIGIBILITY"
                and set(gates) == set(contract.C1_GATE_IDS)
                and all(gates.values())
            )
            if not formal and diagnostic_passed:
                status = "COMPLETE_V2_A_C1_NONFORMAL_DIAGNOSTIC"
            passed = formal and diagnostic_passed
            result = {
                "schema_version": f"{contract.SCHEMA_PREFIX}.single-seed-result.v1",
                "identity": contract.C1_IDENTITY,
                "formal": formal,
                "status": status,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "gates": gates,
                "gate_status": dict(gate_status),
                "stage_status": dict(stage_status),
                "stopped_at": stopped_at,
                "unrun_after": stopped_at if not passed else None,
                "unrun_gates": [
                    gate for gate, state in gate_status.items() if state == "not_run"
                ],
                "training_started": training_started,
                "optimizer_steps": optimizer_steps,
                "model_writes": model_writes,
                "source_identity": source_identity,
                "source_stable": source_stable,
                "formal_preflight": preflight,
                "authorizes": "two additional fresh C1 seeds only"
                if passed
                else "nothing",
                "c2_authorized": False,
                "v2a_passed": False,
            }
            if not formal:
                result["diagnostic_passed"] = diagnostic_passed
            if primary.get("selected") and isinstance(primary["selected"], Mapping):
                result["selected_update"] = primary["selected"].get("update")
                result["selected_checkpoint_sha256"] = primary["selected"].get(
                    "checkpoint_sha256"
                )
            if deployment_path.is_file():
                result["deployment_sha256"] = artifacts.sha256_file(deployment_path)
            if extra:
                result.update(dict(extra))
            return _finish(
                root,
                result=result,
                started_at=started_at,
                wall_started=wall_started,
                source_identity=source_identity,
                source_stable=source_stable,
            )

        prerequisites = _pinned_inputs(repo_root)
        store, bank, dataset, provider = _load_runtime(repo_root)
        probe_model = _fresh_model(seed=contract.MODEL_SEED)
        model_contract = _model_contract_report(probe_model)
        g001 = (
            preflight.get("passed") is True
            and prerequisites["passed"]
            and model_contract["passed"]
        )
        artifacts.write_json(root / "g001-prerequisites.json", prerequisites)
        artifacts.write_json(root / "g001-preflight.json", preflight)
        artifacts.write_json(root / "g001-model-contract.json", model_contract)
        stage_status["G001"] = "passed" if g001 else "failed"
        gate_status["G001"] = "passed" if g001 else "failed"
        if not g001:
            return terminal(
                "INCOMPLETE_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G001",
            )
        del probe_model

        overfit_ids = _overfit32_ids(store)
        overfit_rows = [store[example_id] for example_id in overfit_ids]
        artifacts.write_json(root / "overfit32-selection.json", {"example_ids": overfit_ids})
        overfit_model = _fresh_model(seed=contract.OVERFIT_SEED)
        training_started = True
        overfit = train_overfit32(
            overfit_model,
            overfit_ids,
            provider,
            _overfit_callback(dataset, overfit_rows, bank, device=device),
            root,
            device=device,
        )
        optimizer_steps += int(overfit.get("completed_updates", 0))
        model_writes += int(overfit.get("passed") is True) * 2
        artifacts.write_json(root / "overfit32.json", overfit)
        g002 = overfit.get("passed") is True
        stage_status["G002_OVERFIT32"] = "passed" if g002 else "failed"
        gate_status["G002"] = "passed" if g002 else "failed"
        if overfit.get("passed") is not True:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G002",
            )
        del overfit_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_rows = _records(store, split="train")
        validation = _records(store, split="validation")
        schedule = build_balanced_schedule(train_rows)
        schedule_ledger = schedule_report(schedule)
        artifacts.write_json(root / "schedule.json", schedule_ledger)
        exposure = _trace_exposure_ledger(schedule, provider, dataset)
        artifacts.write_json(root / "trace-exposure-ledger.json", exposure)
        primary_model = _fresh_model(seed=contract.MODEL_SEED)
        primary = train_primary(
            primary_model,
            schedule,
            provider,
            _selection_callback(dataset, validation, bank, device=device),
            root,
            device=device,
        )
        optimizer_steps += int(primary.get("completed_updates", 0))
        model_writes += len(primary.get("candidates", []))
        artifacts.write_json(root / "primary-training.json", primary)
        primary_checks = _primary_training_checks(
            primary,
            root=root,
            exposure=exposure,
            model=primary_model,
        )
        artifacts.write_json(root / "primary-training-checks.json", primary_checks)
        stage_status["PRIMARY_TRAIN_SELECT"] = (
            "passed" if primary_checks["passed"] else "failed"
        )
        if primary_checks["passed"] is not True:
            gate_status["G003"] = "failed"
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G003",
            )

        prestrip_runtime = C1EvaluationRuntime(
            primary_model, dataset, validation, device=device, batch_size=8
        )
        trace_report = prestrip_runtime.trace_credit(
            bank.targets,
            validation_records=validation,
            per_family=int(contract.TRAINING_CONFIG["validation_trace_records_per_family"]),
        )
        artifacts.write_json(root / "trace-credit.json", trace_report)
        g007_report = gate_g007_trace(trace_report)
        artifacts.write_json(root / "g007-trace-gate.json", g007_report)
        g007 = g007_report.get("passed") is True
        stage_status["G007_TRACE_CREDIT"] = "passed" if g007 else "failed"
        gate_status["G007"] = "passed" if g007 else "failed"
        if not g007:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G007",
            )

        prestrip_logits = prestrip_runtime.raw_answer_logits(records=validation)
        stripped = strip_trace_probe(primary_model)
        deployment_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": f"{contract.SCHEMA_PREFIX}.deployment-state.v1",
                "identity": contract.C1_IDENTITY,
                "config": contract.MODEL_CONFIG,
                "model_state": stripped.state_dict,
                "report": stripped.report,
            },
            deployment_path,
        )
        model_writes += 1
        reloaded = reload_stripped(
            stripped.state_dict,
            C1Config(**contract.MODEL_CONFIG),
            device=device,
        )
        deployment_model = reloaded.model.eval()
        deployed_validation = C1EvaluationRuntime(
            deployment_model, dataset, validation, device=device, batch_size=8
        )
        active_device = torch.device(device)
        if active_device.type == "cuda":
            torch.cuda.synchronize(active_device)
        validation_started = time.perf_counter()
        deployed_logits = deployed_validation.raw_answer_logits(records=validation)
        if active_device.type == "cuda":
            torch.cuda.synchronize(active_device)
        validation_seconds = time.perf_counter() - validation_started
        validation_timing = {
            "examples": len(validation),
            "seconds": validation_seconds,
            "examples_per_second": len(validation) / validation_seconds,
            "milliseconds_per_example": 1000.0 * validation_seconds / len(validation),
            "batch_size": 8,
        }
        strip_invariance = _invariance_report(
            prestrip_logits,
            deployed_logits,
            tolerance=float(contract.THRESHOLDS["strip"]["logit_max_abs_diff"]),
        )
        g003 = (
            primary_checks["passed"]
            and strip_invariance["passed"]
            and stripped.report.get("trace_probe_parameters") == 0
            and reloaded.report.get("trace_probe_parameters") == 0
            and not any(
                name.startswith("trace_probe.") for name in reloaded.state_dict
            )
        )
        strip_report = {
            "passed": g003,
            "strip_invariance": strip_invariance,
            "stripped": stripped.report,
            "reloaded": reloaded.report,
            "deployment_sha256": artifacts.sha256_file(deployment_path),
        }
        artifacts.write_json(root / "g003-strip-reload.json", strip_report)
        stage_status["G003_STRIP_RELOAD"] = "passed" if g003 else "failed"
        gate_status["G003"] = "passed" if g003 else "failed"
        if not g003:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G003",
            )
        del prestrip_runtime, primary_model, stripped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        validation_report = evaluate_behavior(
            validation, _lookup_predictor(deployed_logits)
        )
        artifacts.write_json(root / "validation.json", validation_report)
        artifacts.write_json(root / "validation-timing.json", validation_timing)
        g004_report = gate_g004_validation(validation_report)
        artifacts.write_json(root / "g004-validation-gate.json", g004_report)
        g004 = g004_report.get("passed") is True
        stage_status["G004_VALIDATION"] = "passed" if g004 else "failed"
        gate_status["G004"] = "passed" if g004 else "failed"
        if not g004:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G004",
            )

        ood = [
            row
            for row in store.records
            if row.get("split")
            in {
                "composition_ood",
                "entity_ood",
                "length_ood",
                "language_ood",
                "distractor_ood",
                "horizon_ood",
            }
        ]
        ood_runtime = C1EvaluationRuntime(
            deployment_model, dataset, ood, device=device, batch_size=8
        )
        ood_report = ood_runtime.evaluate_answers(records=ood)
        artifacts.write_json(root / "ood.json", ood_report)
        g005_report = gate_g005_ood(ood_report)
        artifacts.write_json(root / "g005-ood-gate.json", g005_report)
        g005 = g005_report.get("passed") is True
        stage_status["G005_OOD"] = "passed" if g005 else "failed"
        gate_status["G005"] = "passed" if g005 else "failed"
        if not g005:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G005",
            )

        causal_rows = _records(store, split="causal_pairs")
        causal_runtime = C1EvaluationRuntime(
            deployment_model, dataset, causal_rows, device=device, batch_size=8
        )
        causal_logits = causal_runtime.raw_answer_logits(records=causal_rows)
        causal_report = evaluate_causal(
            causal_rows, _lookup_predictor(causal_logits)
        )
        artifacts.write_json(root / "causal.json", causal_report)
        g006_report = gate_g006_causal(causal_report)
        artifacts.write_json(root / "g006-causal-gate.json", g006_report)
        g006 = g006_report.get("passed") is True
        stage_status["G006_CAUSAL"] = "passed" if g006 else "failed"
        gate_status["G006"] = "passed" if g006 else "failed"
        if not g006:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G006",
            )

        baseline = deployed_logits
        zero_logits = deployed_validation.raw_answer_logits(
            intervention="zero_hidden", records=validation
        )
        shuffled_logits = deployed_validation.raw_answer_logits(
            intervention="shuffled_hidden", records=validation
        )
        hidden_report = {
            "zero_hidden": _paired_report(validation, baseline, zero_logits),
            "shuffled_hidden": _paired_report(validation, baseline, shuffled_logits),
        }
        artifacts.write_json(root / "hidden-interventions.json", hidden_report)
        g008_report = gate_g008_hidden(hidden_report)
        artifacts.write_json(root / "g008-hidden-gate.json", g008_report)
        g008 = g008_report.get("passed") is True
        stage_status["G008_HIDDEN"] = "passed" if g008 else "failed"
        gate_status["G008"] = "passed" if g008 else "failed"
        if not g008:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G008",
            )

        h0_logits = deployed_validation.raw_answer_logits(
            intervention="h0_no_core", records=validation
        )
        step5_logits = deployed_validation.raw_answer_logits(
            intervention="step5_state_shuffle", records=validation
        )
        recurrence_report = {
            "h0": _paired_report(validation, baseline, h0_logits),
            "step5_shuffle": _paired_report(validation, baseline, step5_logits),
        }
        artifacts.write_json(root / "recurrence-interventions.json", recurrence_report)
        g009_report = gate_g009_recurrence(recurrence_report)
        artifacts.write_json(root / "g009-recurrence-gate.json", g009_report)
        g009 = g009_report.get("passed") is True
        stage_status["G009_RECURRENCE"] = "passed" if g009 else "failed"
        gate_status["G009"] = "passed" if g009 else "failed"
        if not g009:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G009",
            )

        slot_logits = deployed_validation.raw_answer_logits(
            intervention="slot_permutation", records=validation
        )
        slot_invariance = _invariance_report(
            baseline,
            slot_logits,
            tolerance=float(contract.THRESHOLDS["slot"]["logit_max_abs_diff"]),
        )
        integrity_report = {
            "slot": slot_invariance,
            "strip": strip_invariance,
            "trace_probe_parameter_count": 0,
            "deployment": reloaded.report,
            "parameter_graph": parameter_graph_report(deployment_model),
        }
        artifacts.write_json(root / "integrity.json", integrity_report)
        g010_report = gate_g010_integrity(integrity_report)
        artifacts.write_json(root / "g010-integrity-gate.json", g010_report)
        g010 = g010_report.get("passed") is True
        stage_status["G010_SLOT_INTEGRITY"] = "passed" if g010 else "failed"
        gate_status["G010"] = "passed" if g010 else "failed"
        if not g010:
            return terminal(
                "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G010",
            )

        accounting = _accounting(
            model=deployment_model,
            training=primary,
            cache_root=_cache_root(repo_root),
            deployment_path=deployment_path,
            schedule=schedule_ledger,
            exposure=exposure,
            primary_checks=primary_checks,
            validation_timing=validation_timing,
            deployment_expected_sha256=strip_report["deployment_sha256"],
        )
        artifacts.write_json(root / "accounting.json", accounting)
        source_after = artifacts.source_hashes(repo_root)
        source_stable = source_before == source_after
        g011 = accounting["passed"] and source_stable
        stage_status["G011_ACCOUNTING"] = "passed" if g011 else "failed"
        gate_status["G011"] = "passed" if g011 else "failed"
        if not g011:
            return terminal(
                "INCOMPLETE_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
                stopped_at="G011",
            )
        return terminal(
            "PASS_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
            stopped_at=None,
        )
    except Exception as exc:  # noqa: BLE001
        if root.exists():
            status = (
                "CRASH_V2_A_C1_SINGLE_SEED_ELIGIBILITY"
                if training_started
                else "INCOMPLETE_V2_A_C1_SINGLE_SEED_ELIGIBILITY"
            )
            result = {
                "schema_version": f"{contract.SCHEMA_PREFIX}.single-seed-crash.v1",
                "identity": contract.C1_IDENTITY,
                "formal": formal,
                "status": status,
                "passed": False,
                "exit_code": 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "training_started": training_started,
                "optimizer_steps": optimizer_steps,
                "model_writes": model_writes,
                "gates": {
                    gate: state == "passed" for gate, state in gate_status.items()
                },
                "gate_status": gate_status,
                "stage_status": stage_status,
                "source_identity": source_identity,
                "source_stable": False,
                "authorizes": "nothing",
                "c2_authorized": False,
                "v2a_passed": False,
            }
            artifacts.write_json(
                root / ("crash.json" if training_started else "incomplete.json"),
                result,
            )
        else:
            raise
    return _finish(
        root,
        result=result,
        started_at=started_at,
        wall_started=wall_started,
        source_identity=source_identity,
        source_stable=source_stable,
    )


__all__ = [
    "run_c1_preflight",
    "run_c1_single_seed",
]
