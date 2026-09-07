from __future__ import annotations

"""Single-use, read-only execution for the consumed C1 failure attribution."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

import torch

from yggdrasil_v2.v2_a.closure_c1 import artifacts as c1_artifacts
from yggdrasil_v2.v2_a.closure_c1 import contract as c1_contract
from yggdrasil_v2.v2_a.closure_c1.cache import CachedShardDataset, load_qwen35_tokenizer
from yggdrasil_v2.v2_a.closure_c1.model import C1Config, C1Model
from yggdrasil_v2.v2_a.closure_c1.qualification import (
    gate_g004_validation,
    gate_g005_ood,
    gate_g006_causal,
    gate_g008_hidden,
    gate_g009_recurrence,
    gate_g010_integrity,
)
from yggdrasil_v2.v2_a.closure_c1.runtime import (
    C1BatchProvider,
    OfflineRecordStore,
    TraceTargetBank,
    audit_target_bank,
    load_offline_records,
    load_target_bank,
)
from yggdrasil_v2.v2_a.closure_c1.trace_targets import materialize_trace_target
from yggdrasil_v2.v2_a.closure_c1.train import (
    build_balanced_schedule,
    load_checkpoint,
)

from . import artifacts, contract
from .evaluator import load_selected_checkpoint_pin, run_poststop_diagnosis
from .exposure import exposure_coverage_report
from .gradients import shared_gradient_report
from .selection import selection_audit
from .trace_metrics import (
    alignment_decision,
    selection_metric_decision,
    trace_checkpoint_metrics,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _paths(repo_root: Path) -> dict[str, Path]:
    formal = repo_root / contract.FORMAL_ROOT
    cache = repo_root / contract.CACHE_ROOT
    return {
        "formal": formal,
        "cache": cache,
        "dataset": repo_root / contract.C0R_DATA_ROOT,
        "target_bank": cache / "trace-target-bank.json",
        "hidden": cache / "hidden",
        "primary": formal / "primary-training.json",
        "ledger": formal / "trace-exposure-ledger.json",
        "selected": formal / "checkpoints" / "update-05120.pt",
        "final": formal / "checkpoints" / "update-06144.pt",
    }


def _refusal(root: Path, lease: Path, *, status: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.refusal.v1",
        "identity": contract.IDENTITY,
        "status": status,
        "formal": False,
        "passed": False,
        "exit_code": 2,
        "output_root": root.as_posix(),
        "lease_path": lease.as_posix(),
        "reason": reason,
        "authorizes": "nothing",
        "c2_authorized": False,
        "v2a_passed": False,
    }


def _current_source(repo_root: Path) -> tuple[dict[str, str], str]:
    hashes = artifacts.source_hashes(repo_root)
    return hashes, artifacts.source_identity(hashes)


def _audit_preflight(repo_root: Path) -> dict[str, Any]:
    root = repo_root / contract.PREFLIGHT_ROOT
    try:
        result = _json(root / "result.json")
    except (OSError, ValueError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}
    _hashes, current = _current_source(repo_root)
    replay = artifacts.audit_evidence_seal(root)
    checks = {
        "identity": result.get("identity") == f"{contract.IDENTITY}-PREFLIGHT",
        "status": result.get("status") == "PASS_V2_A_C1_FAILURE_ATTRIBUTION_PREFLIGHT",
        "passed": result.get("passed") is True,
        "device": result.get("device") == "cuda",
        "source_identity": result.get("source_identity") == current,
        "source_stable": result.get("source_stable") is True,
        "training_not_started": result.get("training_started") is False,
        "zero_optimizer_steps": result.get("optimizer_steps") == 0,
        "zero_model_writes": result.get("model_writes") == 0,
        "seal_replay": replay.get("passed") is True,
    }
    return {
        "root": root.as_posix(),
        "checks": checks,
        "result": result,
        "seal_replay": replay,
        "passed": all(checks.values()),
    }


def _pin_audit(repo_root: Path, *, replay_formal: bool) -> dict[str, Any]:
    paths = _paths(repo_root)
    formal = paths["formal"]
    files = {
        "formal_result": (formal / "result.json", contract.FORMAL_RESULT_SHA256),
        "formal_seal": (formal / "evidence-seal.json", contract.FORMAL_SEAL_SHA256),
        "primary_training": (paths["primary"], contract.PRIMARY_TRAINING_SHA256),
        "trace_ledger": (paths["ledger"], contract.TRACE_LEDGER_SHA256),
        "trace_credit": (formal / "trace-credit.json", contract.TRACE_CREDIT_SHA256),
        "g007": (formal / "g007-trace-gate.json", contract.G007_SHA256),
        "selected_checkpoint": (paths["selected"], contract.SELECTED_CHECKPOINT_SHA256),
        "final_checkpoint": (paths["final"], contract.FINAL_CHECKPOINT_SHA256),
        "cache_result": (paths["cache"] / "result.json", contract.CACHE_RESULT_SHA256),
        "cache_seal": (paths["cache"] / "evidence-seal.json", contract.CACHE_SEAL_SHA256),
        "trace_bank": (paths["target_bank"], contract.TRACE_BANK_SHA256),
    }
    observed: dict[str, Any] = {}
    for key, (path, expected) in files.items():
        actual = artifacts.sha256_file(path) if path.is_file() else None
        observed[key] = {
            "path": path.as_posix(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": actual == expected,
        }
    c1_hashes = c1_artifacts.source_hashes(repo_root)
    c1_identity = c1_artifacts.source_identity(c1_hashes)
    snapshot_root = formal / "source_snapshot"
    source_rows: dict[str, Any] = {}
    for relative in c1_artifacts.SOURCE_FILES:
        current_path = repo_root / relative
        snapshot_path = snapshot_root / relative
        current_hash = artifacts.sha256_file(current_path) if current_path.is_file() else None
        snapshot_hash = artifacts.sha256_file(snapshot_path) if snapshot_path.is_file() else None
        source_rows[relative.as_posix()] = {
            "current_sha256": current_hash,
            "snapshot_sha256": snapshot_hash,
            "matches": current_hash is not None and current_hash == snapshot_hash,
        }
    design_key = c1_contract.DESIGN_DOC.as_posix()
    mismatches = sorted(
        key for key, row in source_rows.items() if row["matches"] is not True
    )
    allowed_doc_delta = (
        mismatches == [design_key]
        and source_rows[design_key]["current_sha256"]
        == contract.FORMAL_POSTSTOP_DOC_SHA256
        and source_rows[design_key]["snapshot_sha256"]
        == "93C844FBA157D3C42F6A625EE192D3F587269EAA91A741DA5AAFE8A75335901D"
    )
    executable_snapshot_passed = all(
        row["matches"] is True
        for key, row in source_rows.items()
        if key != design_key
    )
    source_check = {
        "expected": contract.FORMAL_SOURCE_IDENTITY,
        "actual": c1_identity,
        "exact_identity_passed": c1_identity == contract.FORMAL_SOURCE_IDENTITY,
        "mismatched_files": mismatches,
        "allowed_poststop_doc_delta": allowed_doc_delta,
        "executable_snapshot_passed": executable_snapshot_passed,
        "files": source_rows,
        "passed": executable_snapshot_passed
        and (not mismatches or allowed_doc_delta),
    }
    formal_result = _json(formal / "result.json")
    result_check = {
        "status": formal_result.get("status") == "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY",
        "stopped_at": formal_result.get("stopped_at") == "G007",
        "authorizes_nothing": formal_result.get("authorizes") == "nothing",
        "selected_update": formal_result.get("selected_update") == contract.SELECTED_UPDATE,
        "c2_not_authorized": formal_result.get("c2_authorized") is False,
    }
    formal_replay = (
        c1_artifacts.audit_evidence_seal(formal)
        if replay_formal
        else {"passed": None, "skipped_in_preflight": True}
    )
    selection = selection_audit(paths["primary"], formal if replay_formal else None)
    pin = load_selected_checkpoint_pin(formal)
    checks = {
        "files": all(row["passed"] for row in observed.values()),
        "formal_source": source_check["passed"],
        "formal_result": all(result_check.values()),
        "formal_seal_replay": formal_replay.get("passed") is True if replay_formal else True,
        "selection": selection.get("passed") is True if replay_formal else selection.get("selection_matches") is True,
        "selected_pin": pin.get("checkpoint_sha256") == contract.SELECTED_CHECKPOINT_SHA256,
    }
    return {
        "files": observed,
        "formal_source_identity": source_check,
        "formal_result": result_check,
        "formal_seal_replay": formal_replay,
        "selection": selection,
        "selected_pin": pin,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _records_by_split(store: OfflineRecordStore, split: str) -> list[dict[str, Any]]:
    return [row for row in store.records if row.get("split") == split]


def _load_runtime(
    repo_root: Path,
) -> tuple[OfflineRecordStore, TraceTargetBank, CachedShardDataset, C1BatchProvider]:
    paths = _paths(repo_root)
    store = load_offline_records(paths["dataset"])
    bank = load_target_bank(paths["target_bank"])
    dataset = CachedShardDataset(paths["hidden"])
    provider = C1BatchProvider(dataset, store, bank)
    return store, bank, dataset, provider


def _load_model(path: Path, device: str) -> C1Model:
    model = C1Model(C1Config(**c1_contract.MODEL_CONFIG), with_trace_probe=True)
    payload = load_checkpoint(model, path)
    if int(payload["update"]) not in {contract.SELECTED_UPDATE, contract.FINAL_UPDATE}:
        raise ValueError("diagnostic checkpoint update is not selected/final")
    return model.to(device).eval()


def _alignment_replay_one(
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    tokenizer: Any,
    global_to_local: Mapping[int, int],
    grammar_ids: frozenset[int],
) -> dict[str, Any]:
    target = materialize_trace_target(record, tokenizer, global_to_local)
    global_ids = list(target.token_ids)
    expected = {
        "text": target.text,
        "global_token_ids": global_ids,
        "token_ids": [global_to_local[value] for value in global_ids],
        "step_indices": list(target.step_ids),
        "global_positions": list(range(len(global_ids))),
        "local_positions": list(target.local_positions),
        "grammar_mask": [value in grammar_ids for value in global_ids],
    }
    mismatches = [key for key, value in expected.items() if row.get(key) != value]
    family = str(record.get("family"))
    steps = Counter(str(value) for value in expected["step_indices"])
    return {
        "example_id": str(record.get("example_id")),
        "family": family,
        "tokens": len(global_ids),
        "steps": dict(steps),
        "reasoning_budget": int(record.get("reasoning_budget", 0)),
        "mismatches": mismatches,
    }


def _alignment_replay(
    records: Sequence[Mapping[str, Any]],
    bank: TraceTargetBank,
    *,
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("alignment workers must be positive")
    structural = audit_target_bank(bank)
    tokenizer = load_qwen35_tokenizer(local_files_only=True)
    global_to_local = bank.lexicon.id_to_local
    grammar_ids = frozenset(bank.lexicon.grammar_ids)

    def task(record: Mapping[str, Any]) -> dict[str, Any]:
        example_id = str(record.get("example_id"))
        row = bank.targets.get(example_id)
        if not isinstance(row, Mapping):
            return {
                "example_id": example_id,
                "family": str(record.get("family")),
                "tokens": 0,
                "steps": {},
                "reasoning_budget": int(record.get("reasoning_budget", 0)),
                "mismatches": ["missing_target"],
            }
        return _alignment_replay_one(
            record, row, tokenizer, global_to_local, grammar_ids
        )

    with ThreadPoolExecutor(max_workers=min(workers, len(records))) as executor:
        rows = list(executor.map(task, records, chunksize=16))
    del tokenizer
    mismatched = [row for row in rows if row["mismatches"]]
    families: dict[str, Any] = {}
    for family in ("ERE", "CPS"):
        family_rows = [row for row in rows if row["family"] == family]
        steps: Counter[str] = Counter()
        depths: Counter[str] = Counter()
        for row in family_rows:
            steps.update(row["steps"])
            depths[str(row["reasoning_budget"])] += 1
        token_counts = [int(row["tokens"]) for row in family_rows]
        families[family] = {
            "records": len(family_rows),
            "tokens": sum(token_counts),
            "minimum_tokens": min(token_counts) if token_counts else 0,
            "maximum_tokens": max(token_counts) if token_counts else 0,
            "step_token_counts": dict(sorted(steps.items(), key=lambda item: int(item[0]))),
            "reasoning_budget_counts": dict(sorted(depths.items(), key=lambda item: int(item[0]))),
        }
    passed = (
        structural.get("passed") is True
        and len(rows) == len(bank)
        and not mismatched
    )
    return {
        "structural": structural,
        "workers": min(workers, len(records)),
        "records": len(rows),
        "bank_records": len(bank),
        "families": families,
        "mismatch_count": len(mismatched),
        "mismatch_examples": mismatched[:32],
        "semantic_boundary": (
            "CPS targets retain top-level candidate alignment; inner action steps "
            "are not independently labelled in the frozen bank"
        ),
        "passed": passed,
    }


def _exposure_counts(
    bank: TraceTargetBank, ledger: Mapping[str, Any]
) -> dict[str, list[int]]:
    rows = ledger.get("rows")
    if not isinstance(rows, list):
        raise ValueError("trace exposure ledger lacks rows")
    result: dict[str, list[int]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("trace exposure row is not an object")
        key = str(raw.get("example_id"))
        target = bank.targets.get(key)
        if not isinstance(target, Mapping):
            raise KeyError(f"exposure target missing: {key}")
        vector = result.setdefault(key, [0] * len(target["token_ids"]))
        start, stop = int(raw["start"]), int(raw["stop"])
        if not 0 <= start < stop <= len(vector):
            raise ValueError(f"exposure window outside target: {key}")
        for index in range(start, stop):
            vector[index] += 1
    return result


def _poststop_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    measurements = report.get("measurements")
    if not isinstance(measurements, Mapping):
        raise ValueError("post-stop report lacks measurements")
    recurrence = measurements["recurrence"]
    recurrence_for_gate = {
        "h0": recurrence["h0_no_core"],
        "step5_shuffle": recurrence["step5_state_shuffle"],
    }
    integrity = measurements["integrity"]
    deployment = integrity["deployment"]
    integrity_for_gate = {
        "strip": integrity["strip"],
        "slot": integrity["slot"],
        "trace_probe_parameter_count": deployment["reloaded_probe_parameters"],
    }
    gates = {
        "G004": gate_g004_validation(measurements["validation"]),
        "G005": gate_g005_ood(measurements["ood"]),
        "G006": gate_g006_causal(measurements["causal"]),
        "G008": gate_g008_hidden(measurements["hidden"]),
        "G009": gate_g009_recurrence(recurrence_for_gate),
        "G010": gate_g010_integrity(integrity_for_gate),
    }
    return {
        "diagnostic_only": True,
        "old_threshold_counterfactual": True,
        "gates": gates,
        "passed": all(value.get("passed") is True for value in gates.values()),
    }


def attribution_decision(
    *,
    pin: Mapping[str, Any],
    alignment: Mapping[str, Any],
    exposure: Mapping[str, Any],
    train_exposure_metrics: Mapping[str, Any],
    alignment_comparison: Mapping[str, Any],
    selection_comparison: Mapping[str, Any],
    gradients: Mapping[str, Any],
    poststop_gates: Mapping[str, Any],
) -> dict[str, Any]:
    measurement_ok = pin.get("passed") is True and alignment.get("passed") is True
    cycle_breach = exposure.get("complete_cycle") is False
    paired = train_exposure_metrics.get("paired_exposure_effect", {}).get(
        "registered", {}
    )
    direct_exposure = any(
        isinstance(paired.get(family), Mapping)
        and paired[family].get("qualified") is True
        for family in ("ERE", "CPS")
    )
    alignment_suspect = alignment_comparison.get("alignment_suspect") is True
    selection_mismatch = selection_comparison.get("selection_metric_mismatch") is True
    gradient_decision = gradients.get("decision", {})
    gradient_issue = bool(
        isinstance(gradient_decision, Mapping)
        and (
            gradient_decision.get("late_trace_dominates") is True
            or gradient_decision.get("majority_conflict") is True
        )
    )
    architecture_passed = poststop_gates.get("passed") is True
    contributing: list[str] = []
    if cycle_breach:
        contributing.append("EXPOSURE_CYCLE_BREACH")
    if direct_exposure:
        contributing.append("EXPOSURE_METRIC_ASSOCIATION")
    if alignment_suspect:
        contributing.append("ALIGNMENT_SUSPECT")
    if selection_mismatch:
        contributing.append("SELECTION_METRIC_MISMATCH")
    if gradient_issue:
        contributing.append("SHARED_GRADIENT_CREDIT_ISSUE")
    if not architecture_passed:
        contributing.append("POSTSTOP_ARCHITECTURE_GATES_WEAK")

    if not measurement_ok:
        g007_primary = "MEASUREMENT_OR_SELECTION_FAILURE"
        repair = "STOP_NO_REPAIR_TRAINING"
    elif cycle_breach and direct_exposure:
        g007_primary = "EXPOSURE_PRIMARY"
        repair = "FRESH_EXPOSURE_ONLY_CAUSAL_REPAIR"
    elif alignment_suspect:
        g007_primary = "ALIGNMENT_PRIMARY"
        repair = "STOP_AND_REDESIGN_ALIGNMENT"
    elif architecture_passed:
        g007_primary = "TRACE_METRIC_OR_PROBE_PRIMARY"
        repair = "REDESIGN_TRACE_READOUT_CONTRACT_BEFORE_TRAINING"
    else:
        g007_primary = "LATENT_OR_TRAINING_PRIMARY"
        repair = "REDESIGN_C1_TRAINING_BEFORE_QUALIFICATION"
    architecture_status = (
        "POSTSTOP_ARCHITECTURE_COUNTERFACTUAL_PASS"
        if architecture_passed
        else "POSTSTOP_ARCHITECTURE_COUNTERFACTUAL_FAIL"
    )
    return {
        "measurement_ok": measurement_ok,
        "g007_primary": g007_primary,
        "architecture_status": architecture_status,
        "repair_action": repair,
        "contributing_causes": contributing,
        "cycle_breach": cycle_breach,
        "direct_exposure_association": direct_exposure,
        "alignment_suspect": alignment_suspect,
        "selection_metric_mismatch": selection_mismatch,
        "gradient_issue": gradient_issue,
        "poststop_architecture_passed": architecture_passed,
        "qualification_boundary": (
            "All G004-G010 labels are post-stop counterfactual measurements; "
            "the consumed formal result remains failed at G007."
        ),
    }


def _finish(
    root: Path,
    *,
    result: dict[str, Any],
    started_at: str,
    wall_started: float,
) -> dict[str, Any]:
    artifacts.write_json(root / "result.json", result)
    artifacts.write_json(
        root / "run-state.json",
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.run-state.v1",
            "identity": result["identity"],
            "formal": bool(result.get("formal")),
            "started_at": started_at,
            "finished_at": _now(),
            "wall_seconds": time.perf_counter() - wall_started,
            "source_identity": result.get("source_identity"),
            "source_stable": result.get("source_stable"),
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "old_root_writes": 0,
        },
    )
    seal = artifacts.write_evidence_seal(root)
    replay = artifacts.audit_evidence_seal(root)
    if replay.get("passed") is not True:
        result = {
            **result,
            "status": "INCOMPLETE_V2_A_C1_FAILURE_ATTRIBUTION",
            "passed": False,
            "exit_code": 1,
            "authorizes": "nothing",
            "seal_failure": replay,
        }
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


def run_preflight(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = output_root or repo_root / contract.PREFLIGHT_ROOT
    lease = lease_path or repo_root / contract.PREFLIGHT_LEASE
    if root.exists() or lease.exists():
        return _refusal(
            root,
            lease,
            status="REFUSE_V2_A_C1_FAILURE_ATTRIBUTION_PREFLIGHT_SINGLE_USE",
            reason="fixed preflight root or sibling lease already exists",
        )
    if str(device) != "cuda" or not torch.cuda.is_available():
        return _refusal(
            root,
            lease,
            status="REFUSE_V2_A_C1_FAILURE_ATTRIBUTION_PREFLIGHT_DEVICE",
            reason="fixed preflight requires the CUDA-visible discrete GPU",
        )
    source_hashes, source_identity = _current_source(repo_root)
    started_at, wall_started = _now(), time.perf_counter()
    artifacts.claim_single_use(
        output_root=root,
        lease_path=lease,
        formal=False,
        identity=f"{contract.IDENTITY}-PREFLIGHT",
    )
    artifacts.snapshot_sources(repo_root, root)
    artifacts.write_json(root / "contract-manifest.json", contract.manifest())
    try:
        pin = _pin_audit(repo_root, replay_formal=False)
        store, bank, dataset, provider = _load_runtime(repo_root)
        sample: list[Mapping[str, Any]] = []
        for family in ("ERE", "CPS"):
            sample.extend(
                sorted(
                    [
                        row
                        for row in store.records
                        if row.get("split") == "validation" and row.get("family") == family
                    ],
                    key=lambda row: str(row["example_id"]),
                )[:4]
            )
        alignment = _alignment_replay(sample, bank, workers=min(4, contract.ALIGNMENT_WORKERS))
        # A sampled preflight cannot equal the full bank by construction.
        alignment["passed"] = (
            alignment["structural"].get("passed") is True
            and alignment["mismatch_count"] == 0
            and alignment["records"] == len(sample)
        )
        model = _load_model(_paths(repo_root)["selected"], device)
        trace = trace_checkpoint_metrics(
            model,
            dataset,
            store.records,
            bank,
            device=device,
            per_family=4,
        )
        schedule = build_balanced_schedule(_records_by_split(store, "train"))
        gradients = shared_gradient_report(
            model, schedule, provider, device=device, batches=1
        )
        source_stable = _current_source(repo_root)[1] == source_identity
        passed = bool(
            pin.get("passed") is True
            and alignment.get("passed") is True
            and trace.get("records") == 8
            and gradients.get("parameters_unchanged") is True
            and source_stable
        )
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": f"{contract.IDENTITY}-PREFLIGHT",
            "formal": False,
            "device": str(device),
            "status": (
                "PASS_V2_A_C1_FAILURE_ATTRIBUTION_PREFLIGHT"
                if passed
                else "INCOMPLETE_V2_A_C1_FAILURE_ATTRIBUTION_PREFLIGHT"
            ),
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "source_identity": source_identity,
            "source_stable": source_stable,
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "old_root_writes": 0,
            "pin_audit": pin,
            "alignment_smoke": alignment,
            "trace_smoke": trace,
            "gradient_smoke": gradients,
            "authorizes": (
                "formal C1 failure-attribution launch only" if passed else "nothing"
            ),
            "c2_authorized": False,
            "v2a_passed": False,
        }
    except Exception as exc:
        source_stable = _current_source(repo_root)[1] == source_identity
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": f"{contract.IDENTITY}-PREFLIGHT",
            "formal": False,
            "device": str(device),
            "status": "CRASH_V2_A_C1_FAILURE_ATTRIBUTION_PREFLIGHT",
            "passed": False,
            "exit_code": 1,
            "source_identity": source_identity,
            "source_stable": source_stable,
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "old_root_writes": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "authorizes": "nothing",
            "c2_authorized": False,
            "v2a_passed": False,
        }
    return _finish(root, result=result, started_at=started_at, wall_started=wall_started)


def run_formal(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = output_root or repo_root / contract.OUTPUT_ROOT
    lease = lease_path or repo_root / contract.LEASE_PATH
    if root.exists() or lease.exists():
        return _refusal(
            root,
            lease,
            status="REFUSE_V2_A_C1_FAILURE_ATTRIBUTION_SINGLE_USE",
            reason="fixed diagnostic root or sibling lease already exists",
        )
    if str(device) != "cuda" or not torch.cuda.is_available():
        return _refusal(
            root,
            lease,
            status="REFUSE_V2_A_C1_FAILURE_ATTRIBUTION_DEVICE",
            reason="fixed diagnostic requires the CUDA-visible discrete GPU",
        )
    preflight = _audit_preflight(repo_root)
    if preflight.get("passed") is not True:
        return _refusal(
            root,
            lease,
            status="REFUSE_V2_A_C1_FAILURE_ATTRIBUTION_PREFLIGHT",
            reason="sealed current-source CUDA preflight is absent or invalid",
        ) | {"preflight": preflight}

    source_hashes, source_identity = _current_source(repo_root)
    started_at, wall_started = _now(), time.perf_counter()
    artifacts.claim_single_use(
        output_root=root,
        lease_path=lease,
        formal=True,
        identity=contract.IDENTITY,
    )
    artifacts.snapshot_sources(repo_root, root)
    artifacts.write_json(root / "contract-manifest.json", contract.manifest())
    artifacts.write_json(root / "preflight-audit.json", preflight)
    try:
        pin = _pin_audit(repo_root, replay_formal=True)
        artifacts.write_json(root / "a001-identity-selection.json", pin)
        if pin.get("passed") is not True:
            raise RuntimeError("A001 identity/selection audit failed")

        store, bank, dataset, provider = _load_runtime(repo_root)
        alignment = _alignment_replay(
            list(store.records), bank, workers=contract.ALIGNMENT_WORKERS
        )
        artifacts.write_json(root / "a002-alignment-replay.json", alignment)
        if alignment.get("passed") is not True:
            raise RuntimeError("A002 full-bank alignment replay failed")

        ledger = _json(_paths(repo_root)["ledger"])
        exposure = exposure_coverage_report(bank.payload(), ledger)
        counts = _exposure_counts(bank, ledger)
        artifacts.write_json(root / "a003-exposure-coverage.json", exposure)

        selected_model = _load_model(_paths(repo_root)["selected"], device)
        selected_metrics = trace_checkpoint_metrics(
            selected_model,
            dataset,
            store.records,
            bank,
            device=device,
        )
        artifacts.write_json(root / "a004-selected-trace-metrics.json", selected_metrics)
        train_exposure_metrics = trace_checkpoint_metrics(
            selected_model,
            dataset,
            store.records,
            bank,
            device=device,
            per_family=contract.TRAIN_EXPOSURE_RECORDS_PER_FAMILY,
            split="train",
            exposure_counts=counts,
        )
        artifacts.write_json(
            root / "a003-train-exposure-stratified-metrics.json",
            train_exposure_metrics,
        )
        alignment_metrics = trace_checkpoint_metrics(
            selected_model,
            dataset,
            store.records,
            bank,
            device=device,
            per_family=contract.ALIGNMENT_RECORDS_PER_FAMILY,
            alignment_modes=contract.ALIGNMENT_MODES,
        )
        alignment_comparison = alignment_decision(alignment_metrics)
        artifacts.write_json(root / "a002-alignment-controls.json", alignment_metrics)
        artifacts.write_json(
            root / "a002-alignment-decision.json", alignment_comparison
        )

        final_model = _load_model(_paths(repo_root)["final"], device)
        final_metrics = trace_checkpoint_metrics(
            final_model,
            dataset,
            store.records,
            bank,
            device=device,
        )
        selection_comparison = selection_metric_decision(
            selected_metrics, final_metrics
        )
        artifacts.write_json(root / "a004-final-trace-metrics.json", final_metrics)
        artifacts.write_json(
            root / "a004-selection-metric-decision.json", selection_comparison
        )
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        schedule = build_balanced_schedule(_records_by_split(store, "train"))
        gradients = shared_gradient_report(
            selected_model,
            schedule,
            provider,
            device=device,
            batches=contract.GRADIENT_BATCHES,
        )
        artifacts.write_json(root / "a005-shared-gradients.json", gradients)
        del selected_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        poststop = run_poststop_diagnosis(
            formal_root=_paths(repo_root)["formal"],
            dataset=dataset,
            records=store.records,
            device=device,
            batch_size=contract.EVALUATION_BATCH_SIZE,
        )
        poststop_gates = _poststop_gates(poststop)
        artifacts.write_json(root / "a006-poststop-architecture.json", poststop)
        artifacts.write_json(root / "a006-poststop-gates.json", poststop_gates)

        decision = attribution_decision(
            pin=pin,
            alignment=alignment,
            exposure=exposure,
            train_exposure_metrics=train_exposure_metrics,
            alignment_comparison=alignment_comparison,
            selection_comparison=selection_comparison,
            gradients=gradients,
            poststop_gates=poststop_gates,
        )
        artifacts.write_json(root / "attribution-decision.json", decision)

        # Replaying the old formal seal again proves the read-only diagnostic
        # left every sealed byte unchanged after all measurements completed.
        old_root_after = c1_artifacts.audit_evidence_seal(_paths(repo_root)["formal"])
        artifacts.write_json(root / "old-root-after-audit.json", old_root_after)
        source_stable = _current_source(repo_root)[1] == source_identity
        complete = bool(
            decision.get("measurement_ok") is True
            and old_root_after.get("passed") is True
            and source_stable
        )
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.result.v1",
            "identity": contract.IDENTITY,
            "formal": True,
            "device": str(device),
            "status": (
                "COMPLETE_V2_A_C1_FAILURE_ATTRIBUTION"
                if complete
                else "INCOMPLETE_V2_A_C1_FAILURE_ATTRIBUTION"
            ),
            "passed": complete,
            "exit_code": 0 if complete else 1,
            "source_identity": source_identity,
            "source_stable": source_stable,
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "old_root_writes": 0,
            "measurements": {key: "completed" for key in contract.MEASUREMENT_IDS},
            "attribution": decision,
            "old_root_after": old_root_after,
            "authorizes": (
                "one fresh C1 repair design only" if complete else "nothing"
            ),
            "c2_authorized": False,
            "v2a_passed": False,
        }
    except Exception as exc:
        source_stable = _current_source(repo_root)[1] == source_identity
        try:
            old_root_after = c1_artifacts.audit_evidence_seal(
                _paths(repo_root)["formal"]
            )
        except Exception as audit_exc:
            old_root_after = {
                "passed": False,
                "reason": f"{type(audit_exc).__name__}: {audit_exc}",
            }
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.result.v1",
            "identity": contract.IDENTITY,
            "formal": True,
            "device": str(device),
            "status": "CRASH_V2_A_C1_FAILURE_ATTRIBUTION",
            "passed": False,
            "exit_code": 1,
            "source_identity": source_identity,
            "source_stable": source_stable,
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "old_root_writes": 0,
            "old_root_after": old_root_after,
            "error": f"{type(exc).__name__}: {exc}",
            "authorizes": "nothing",
            "c2_authorized": False,
            "v2a_passed": False,
        }
    return _finish(root, result=result, started_at=started_at, wall_started=wall_started)


__all__ = ["attribution_decision", "run_formal", "run_preflight"]
