from __future__ import annotations

"""C1R repair runner with an explicit, fail-closed stage machine.

The implementation deliberately keeps the stage machine independent from the
old ``closure_c1`` runner.  Training/evaluation implementations are expressed
as hooks so ordering is testable without starting CUDA; the production hook
set is the default and no formal stage is represented by a fake PASS.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import torch

from yggdrasil_v2.v2_a.closure_c1 import artifacts as legacy_c1_artifacts
from yggdrasil_v2.v2_a.closure_c1 import contract as legacy_c1_contract
from yggdrasil_v2.v2_a.closure_c1 import runtime as c1_runtime
from yggdrasil_v2.v2_a.closure_c1.model import C1Config, C1Model
from yggdrasil_v2.v2_a.closure_c1 import deployment as c1_deployment
from yggdrasil_v2.v2_a.closure_c1 import evaluate as c1_evaluate
from yggdrasil_v2.v2_a.closure_c1 import evaluation_runtime as c1_evaluation_runtime
from yggdrasil_v2.v2_a.closure_c1 import qualification as c1_qualification
from yggdrasil_v2.v2_a.closure_c1_diagnosis import contract as attribution_contract
from yggdrasil_v2.v2_a.closure_c1_diagnosis import artifacts as attribution_artifacts
from yggdrasil_v2.v2_a.closure_c1r import artifacts as c1r_artifacts
from yggdrasil_v2.v2_a.closure_c1r import contract as c1r_contract
from yggdrasil_v2.v2_a.closure_c1r import coverage as c1r_coverage
from yggdrasil_v2.v2_a.closure_c1r import train as c1r_train


IDENTITY = c1r_contract.C1R_IDENTITY
SCHEMA_PREFIX = c1r_contract.SCHEMA_PREFIX
OUTPUT_ROOT = c1r_contract.OUTPUT_ROOT
LEASE_PATH = c1r_contract.LEASE_PATH
PREFLIGHT_ROOT = c1r_contract.PREFLIGHT_ROOT
PREFLIGHT_LEASE = c1r_contract.PREFLIGHT_LEASE_PATH
STAGE_A_UPDATES = int(c1r_contract.STAGE_A_CONFIG["maximum_updates"])
STAGE_B_UPDATES = int(c1r_contract.STAGE_B_CONFIG["maximum_updates"])
EXPOSURE_BLOCK = int(c1r_contract.STAGE_B_CONFIG["block_tokens"])
NEW_SOURCE_FILES = c1r_artifacts.SOURCE_FILES

ATTRIBUTION_RESULT_SHA256 = c1r_contract.ATTRIBUTION_RESULT_SHA256
ATTRIBUTION_SEAL_SHA256 = c1r_contract.ATTRIBUTION_SEAL_SHA256


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(root: Path, name: str, value: Mapping[str, Any]) -> None:
    c1r_artifacts.write_json(root / name, dict(value))


def _tree_hashes(root: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    excluded = set(exclude)
    return {
        path.relative_to(root).as_posix(): c1r_artifacts.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _old_root_metadata(repo_root: Path) -> dict[str, Any]:
    """Cheap immutable-root write detector; content is separately seal-pinned."""

    roots = {
        "formal": legacy_c1_contract.C1_OUTPUT_ROOT,
        "cache": legacy_c1_contract.CACHE_OUTPUT_ROOT,
        "attribution": c1r_contract.ATTRIBUTION_ROOT,
    }
    report: dict[str, Any] = {}
    for label, relative in roots.items():
        root = repo_root / relative
        digest = hashlib.sha256()
        entries = 0
        total_bytes = 0
        if not root.is_dir():
            report[label] = {
                "root": root.as_posix(),
                "exists": False,
                "entries": 0,
                "bytes": 0,
                "metadata_sha256": None,
            }
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            relative_name = path.relative_to(root).as_posix()
            digest.update(relative_name.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            entries += 1
            total_bytes += int(stat.st_size)
        report[label] = {
            "root": root.as_posix(),
            "exists": True,
            "entries": entries,
            "bytes": total_bytes,
            "metadata_sha256": digest.hexdigest().upper(),
        }
    return report


def _write_evidence_seal(root: Path) -> str:
    seal_name = "evidence-seal.json"
    _write(
        root,
        seal_name,
        {
            "schema_version": f"{SCHEMA_PREFIX}.evidence-seal.v1",
            "identity": IDENTITY,
            "files": _tree_hashes(root, exclude=(seal_name,)),
        },
    )
    return c1r_artifacts.sha256_file(root / seal_name)


def _audit_evidence_seal(root: Path) -> dict[str, Any]:
    path = root / "evidence-seal.json"
    if not path.is_file():
        return {"passed": False, "reason": "missing evidence-seal.json"}
    try:
        payload = _json(path)
        expected = payload["files"]
        schema_ok = payload.get("schema_version") == f"{SCHEMA_PREFIX}.evidence-seal.v1"
        identity_ok = payload.get("identity") == IDENTITY
        actual = _tree_hashes(root, exclude=("evidence-seal.json",))
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        return {
            "passed": bool(schema_ok and identity_ok and not missing and not unexpected and not mismatched),
            "schema": schema_ok,
            "identity": identity_ok,
            "entries": len(expected),
            "matched": len(expected) - len(missing) - len(mismatched),
            "missing": missing,
            "unexpected": unexpected,
            "mismatched": mismatched,
            "seal_sha256": c1r_artifacts.sha256_file(path),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _claim_single_use(*, identity: str, output_root: Path, lease_path: Path, formal: bool) -> None:
    if output_root.exists() or lease_path.exists():
        raise FileExistsError("fixed output root or sibling lease already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    payload = {
        "schema_version": f"{SCHEMA_PREFIX}.single-use-lease.v1",
        "identity": identity,
        "output_root": output_root.as_posix(),
        "formal": formal,
        "created_at": _now(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
    }
    with lease_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _source_hashes(repo_root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for relative in NEW_SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(str(path))
        rows[relative.as_posix()] = c1r_artifacts.sha256_file(path)
    return rows


def source_identity(repo_root: Path) -> str:
    return c1r_artifacts.source_identity(_source_hashes(Path(repo_root).resolve()))


def _cuda_device_audit(device: str) -> dict[str, Any]:
    """Fail closed unless the selected CUDA runtime is the registered RTX 4070."""

    if device != "cuda":
        return {
            "passed": False,
            "requested_device": device,
            "reason": "C1R requires the registered CUDA device",
        }
    if not torch.cuda.is_available():
        return {
            "passed": False,
            "requested_device": device,
            "cuda_available": False,
            "device_count": 0,
            "reason": "CUDA runtime is unavailable",
        }
    try:
        index = int(torch.cuda.current_device())
        count = int(torch.cuda.device_count())
        name = str(torch.cuda.get_device_name(index))
        capability = [int(value) for value in torch.cuda.get_device_capability(index)]
        expected = c1r_contract.REQUIRED_CUDA_DEVICE_NAME_FRAGMENT
        return {
            "passed": bool(count > index >= 0 and expected.casefold() in name.casefold()),
            "requested_device": device,
            "cuda_available": True,
            "device_count": count,
            "selected_device_index": index,
            "selected_device_name": name,
            "compute_capability": capability,
            "required_name_fragment": expected,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
    except (AssertionError, RuntimeError, ValueError) as exc:
        return {
            "passed": False,
            "requested_device": device,
            "cuda_available": True,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _coverage_schedule_audit(
    plan: Mapping[str, Any], train_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare both rows and registered digests for the Stage B schedule."""

    plan_rows = plan.get("schedule")
    report_rows = train_report.get("rows")
    if not isinstance(plan_rows, list) or not isinstance(report_rows, list):
        return {
            "passed": False,
            "rows_match": False,
            "hash_match": False,
            "reason": "coverage or training schedule rows are missing",
        }
    rows_match = len(plan_rows) == len(report_rows) and all(
        int(left["update"]) == int(right["update"])
        and int(left["epoch"]) == int(right["epoch"])
        and list(left["example_ids"]) == list(right["example_ids"])
        for left, right in zip(plan_rows, report_rows)
    )
    plan_hash = plan.get("schedule_sha256")
    train_hash = train_report.get("sha256")
    hash_match = (
        isinstance(plan_hash, str)
        and isinstance(train_hash, str)
        and plan_hash == train_hash
    )
    return {
        "passed": bool(rows_match and hash_match),
        "rows_match": rows_match,
        "hash_match": hash_match,
        "coverage_schedule_sha256": plan_hash,
        "training_schedule_sha256": train_hash,
    }


def _snapshot_sources(repo_root: Path, root: Path) -> None:
    for relative in NEW_SOURCE_FILES:
        source = repo_root / relative
        destination = root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def _refusal(root: Path, lease: Path, status: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}.refusal.v1",
        "identity": IDENTITY,
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


def _attribution_pin(repo_root: Path) -> dict[str, Any]:
    root = repo_root / attribution_contract.OUTPUT_ROOT
    result_path = root / "result.json"
    seal_path = root / "evidence-seal.json"
    result = _json(result_path) if result_path.is_file() else {}
    result_hash = attribution_artifacts.sha256_file(result_path) if result_path.is_file() else None
    seal_hash = attribution_artifacts.sha256_file(seal_path) if seal_path.is_file() else None
    replay = attribution_artifacts.audit_evidence_seal(root) if root.is_dir() else {"passed": False}
    passed = (
        result_hash == ATTRIBUTION_RESULT_SHA256
        and seal_hash == ATTRIBUTION_SEAL_SHA256
        and replay.get("passed") is True
        and result.get("identity") == attribution_contract.IDENTITY
        and result.get("status") == "COMPLETE_V2_A_C1_FAILURE_ATTRIBUTION"
        and result.get("authorizes") == "one fresh C1 repair design only"
        and result.get("c2_authorized") is False
    )
    return {
        "root": root.as_posix(),
        "result_sha256": result_hash,
        "seal_sha256": seal_hash,
        "fixed_hashes": {"result": ATTRIBUTION_RESULT_SHA256, "seal": ATTRIBUTION_SEAL_SHA256},
        "seal_replay": replay,
        "passed": passed,
    }


def audit_pins(repo_root: Path) -> dict[str, Any]:
    """Audit consumed C1, attribution, and cache roots without writing them."""
    repo_root = Path(repo_root).resolve()
    def pinned(root: Path, result_sha: str, seal_sha: str) -> dict[str, Any]:
        return legacy_c1_artifacts.audit_pinned_root(
            repo_root / root, result_sha256=result_sha, seal_sha256=seal_sha
        )

    old = {
        "formal": pinned(legacy_c1_contract.C1_OUTPUT_ROOT, attribution_contract.FORMAL_RESULT_SHA256, attribution_contract.FORMAL_SEAL_SHA256),
        "cache": pinned(legacy_c1_contract.CACHE_OUTPUT_ROOT, legacy_c1_contract.CACHE_RESULT_SHA256 or "", legacy_c1_contract.CACHE_SEAL_SHA256 or ""),
    }
    old["passed"] = all(value.get("passed") is True for value in old.values())
    attribution = _attribution_pin(repo_root)
    return {
        "old_c1_and_cache": old,
        "attribution": attribution,
        "passed": old.get("passed") is True and attribution.get("passed") is True,
    }


def _state_digest(model: torch.nn.Module) -> str:
    return c1r_train.state_hash(model)


def _state_mapping_digest(
    state: Mapping[str, torch.Tensor], *, exclude_prefix: str | None = None
) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if exclude_prefix is not None and name.startswith(exclude_prefix):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def _preflight_model_smoke(repo_root: Path, device: str) -> dict[str, Any]:
    """Run the required zero-step answer/probe backward checks when CUDA exists."""
    if device != "cuda":
        return {"passed": False, "reason": "C1R preflight requires CUDA"}
    if not torch.cuda.is_available():
        return {"passed": False, "reason": "CUDA runtime is unavailable"}
    try:
        store = c1_runtime.load_offline_records(repo_root / attribution_contract.C0R_DATA_ROOT)
        bank = c1_runtime.load_target_bank(repo_root / attribution_contract.CACHE_ROOT / "trace-target-bank.json")
        dataset = c1_runtime.CachedShardDataset(repo_root / attribution_contract.CACHE_ROOT / "hidden")
        provider = c1_runtime.C1BatchProvider(dataset, store, bank)
        ids = [str(row["example_id"]) for row in store.records if row.get("split") == "train"][:8]
        if len(ids) != 8:
            raise ValueError("preflight requires eight train records")
        batch = provider(ids, epoch=0, trace_chunk_tokens=EXPOSURE_BLOCK)
        moved = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        # Answer-only smoke: a probe-free model must be able to backpropagate
        # answer loss without constructing or touching training-only state.
        answer_model = C1Model(C1Config(**legacy_c1_contract.MODEL_CONFIG), with_trace_probe=False).to("cuda")
        answer_model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = answer_model(moved["source_hidden"], moved["source_mask"], return_trajectory=False)
            answer_loss = torch.nn.functional.cross_entropy(output["logits"], moved["answers"])
        answer_loss.backward()
        answer_grad = any(parameter.grad is not None for parameter in answer_model.parameters())
        answer_model.zero_grad(set_to_none=True)
        answer_model.to("cpu")
        del answer_model, output, answer_loss
        torch.cuda.empty_cache()

        # Probe smoke: use a distinct full model, freeze every deployment
        # parameter, and retain gradients only on trace_probe for a 65-token
        # block.  No optimizer is constructed or stepped.
        probe_model = C1Model(C1Config(**legacy_c1_contract.MODEL_CONFIG), with_trace_probe=True).to("cuda")
        probe_model.train()
        for name, parameter in probe_model.named_parameters():
            parameter.requires_grad_(name.startswith("trace_probe."))
        before = _state_digest(probe_model)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            trajectory_output = probe_model(moved["source_hidden"], moved["source_mask"], return_trajectory=True)
            packed_trajectory = trajectory_output["trajectory"].repeat_interleave(8, dim=0)
            packed_steps = moved["trace_step_indices"].repeat_interleave(8, dim=0)
            packed_global = moved["trace_global_positions"].repeat_interleave(8, dim=0)
            packed_local = moved["trace_local_positions"].repeat_interleave(8, dim=0)
            packed_targets = moved["trace_targets"].repeat_interleave(8, dim=0)
            packed_mask = moved["trace_mask"].repeat_interleave(8, dim=0).bool()
            trace_logits = probe_model.trace_logits(
                packed_trajectory, packed_steps, packed_global, packed_local
            )
            trace_loss = torch.nn.functional.cross_entropy(
                trace_logits[packed_mask], packed_targets[packed_mask]
            )
        trace_loss.backward()
        torch.cuda.synchronize()
        packed_wall_seconds = time.perf_counter() - started
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        probe_grad = all(parameter.grad is not None for parameter in probe_model.trace_probe.parameters())
        frozen_grad = all(parameter.grad is None for name, parameter in probe_model.named_parameters() if not name.startswith("trace_probe."))
        after = _state_digest(probe_model)
        return {
            "passed": bool(
                answer_grad
                and probe_grad
                and frozen_grad
                and before == after
                and packed_trajectory.shape[0] == 64
                and peak_memory_bytes > 0
                and packed_wall_seconds > 0.0
            ),
            "cuda_bfloat16": True,
            "answer_only_backward": answer_grad,
            "frozen_non_probe_65_block_backward": frozen_grad,
            "probe_65_block_backward": probe_grad,
            "frozen_probe_65_block_backward": bool(probe_grad and frozen_grad),
            "packed_blocks": int(packed_trajectory.shape[0]),
            "packed_block_width": int(packed_targets.shape[1]),
            "packed_valid_tokens": int(packed_mask.sum().detach().cpu()),
            "packed_forward_backward_wall_seconds": packed_wall_seconds,
            "peak_cuda_memory_bytes": peak_memory_bytes,
            "peak_cuda_memory_mib": peak_memory_bytes / (1024 * 1024),
            "parameter_state_before": before,
            "parameter_state_after": after,
            "optimizer_steps": 0,
            "model_writes": 0,
        }
    except (OSError, KeyError, ValueError, RuntimeError, TypeError) as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _audit_c1r_preflight(root: Path, repo_root: Path) -> dict[str, Any]:
    """Replay our preflight contract locally.

    The legacy helper has a historical early-return defect and cannot be used
    as the authority for this new root.  Keep the C1R prerequisite check local
    and explicit without changing any consumed C1 artifact.
    """
    return c1r_artifacts.audit_preflight_root(
        root,
        identity=f"{IDENTITY}-PREFLIGHT",
        status="PASS_V2_A_C1R_PREFLIGHT",
        source_identity_value=source_identity(repo_root),
    )


def run_preflight(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    device: str = "cuda",
    pin_auditor: Callable[[Path], Mapping[str, Any]] = audit_pins,
    model_smoke: Callable[[Path, str], Mapping[str, Any]] = _preflight_model_smoke,
    device_auditor: Callable[[str], Mapping[str, Any]] = _cuda_device_audit,
) -> dict[str, Any]:
    """Create a sealed, non-formal preflight root; never trains or writes a model."""
    repo_root = Path(repo_root).resolve()
    root = Path(output_root) if output_root is not None else repo_root / PREFLIGHT_ROOT
    lease = Path(lease_path) if lease_path is not None else repo_root / PREFLIGHT_LEASE
    if root.exists() or lease.exists():
        return _refusal(root, lease, "REFUSE_V2_A_C1R_PREFLIGHT_SINGLE_USE", "preflight root or lease already exists")
    device_audit = dict(device_auditor(device))
    if device_audit.get("passed") is not True:
        return _refusal(
            root,
            lease,
            "REFUSE_V2_A_C1R_PREFLIGHT_DEVICE",
            f"registered RTX 4070 CUDA device is required: {device_audit}",
        )
    started = _now()
    try:
        _claim_single_use(identity=f"{IDENTITY}-PREFLIGHT", output_root=root, lease_path=lease, formal=False)
        before = _source_hashes(repo_root)
        old_before = _old_root_metadata(repo_root)
        pin = dict(pin_auditor(repo_root))
        smoke = dict(model_smoke(repo_root, device))
        after = _source_hashes(repo_root)
        old_after = _old_root_metadata(repo_root)
        old_stable = old_before == old_after
        zero_smoke = (
            smoke.get("training_started") is not True
            and smoke.get("optimizer_steps", 0) == 0
            and smoke.get("model_writes", 0) == 0
        )
        passed = bool(
            pin.get("passed")
            and smoke.get("passed")
            and zero_smoke
            and before == after
            and old_stable
        )
        result = {
            "schema_version": f"{SCHEMA_PREFIX}.preflight.v1",
            "identity": f"{IDENTITY}-PREFLIGHT",
            "formal": False,
            "status": "PASS_V2_A_C1R_PREFLIGHT" if passed else "INCOMPLETE_V2_A_C1R_PREFLIGHT",
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "source_identity": source_identity(repo_root),
            "source_stable": before == after,
            "preflight_zero_training": zero_smoke,
            "training_started": False,
            "optimizer_steps": 0,
            "model_writes": 0,
            "old_root_writes": 0 if old_stable else 1,
            "old_roots_stable": old_stable,
            "old_roots_before": old_before,
            "old_roots_after": old_after,
            "started_at": started,
            "finished_at": _now(),
            "pins": pin,
            "device": device_audit,
            "model_smoke": smoke,
            "authorizes": "formal C1R launch only" if passed else "nothing",
            "c2_authorized": False,
            "v2a_passed": False,
        }
        c1r_artifacts.snapshot_sources(repo_root, root)
        _write(root, "preflight-audit.json", {"pins": pin, "device": device_audit, "model_smoke": smoke, "source_before": before, "source_after": after, "old_roots_before": old_before, "old_roots_after": old_after})
        _write(root, "result.json", result)
        seal = _write_evidence_seal(root)
        replay = _audit_evidence_seal(root)
        result["evidence_seal_sha256"] = seal
        result["seal_replay"] = replay
        if replay.get("passed") is not True:
            result.update({"status": "INCOMPLETE_V2_A_C1R_PREFLIGHT", "passed": False, "exit_code": 1, "authorizes": "nothing"})
            _write(root, "result.json", result)
            seal = _write_evidence_seal(root)
            result["evidence_seal_sha256"] = seal
            result["seal_replay"] = _audit_evidence_seal(root)
        return result | {"output_root": root.as_posix(), "lease_path": lease.as_posix()}
    except Exception as exc:
        if root.is_dir():
            _write(root, "result.json", {"schema_version": f"{SCHEMA_PREFIX}.preflight.v1", "identity": f"{IDENTITY}-PREFLIGHT", "formal": False, "status": "CRASH_V2_A_C1R_PREFLIGHT", "passed": False, "exit_code": 1, "reason": f"{type(exc).__name__}: {exc}", "authorizes": "nothing", "c2_authorized": False, "v2a_passed": False})
            seal = _write_evidence_seal(root)
            return {"status": "CRASH_V2_A_C1R_PREFLIGHT", "passed": False, "exit_code": 1, "output_root": root.as_posix(), "evidence_seal_sha256": seal, "reason": f"{type(exc).__name__}: {exc}"}
        return _refusal(root, lease, "CRASH_V2_A_C1R_PREFLIGHT", f"{type(exc).__name__}: {exc}")


Hook = Callable[["RunContext"], Mapping[str, Any]]


def _not_implemented(context: "RunContext") -> Mapping[str, Any]:
    del context
    return {"passed": False, "reason": "C1R formal stage hook was not provided"}


@dataclass
class StageHooks:
    fresh_init_parity: Hook = _not_implemented
    answer_only_overfit32: Hook = _not_implemented
    stage_a_train: Hook = _not_implemented
    stage_a_gate: Callable[["RunContext", str], Mapping[str, Any]] | None = None
    freeze_deployment: Hook = _not_implemented
    cache_train_trajectories: Hook = _not_implemented
    frozen_probe_overfit: Hook = _not_implemented
    stage_b_train: Hook = _not_implemented
    g007_trace: Hook = _not_implemented
    strip_reload: Hook = _not_implemented


@dataclass
class RunContext:
    repo_root: Path
    root: Path
    lease_path: Path
    device: str
    identity: str = IDENTITY
    training_started: bool = False
    optimizer_steps: int = 0
    model_writes: int = 0
    old_root_writes: int = 0
    stages: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)


def _consume(context: RunContext, name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    context.training_started = context.training_started or bool(result.get("training_started", False))
    context.optimizer_steps += int(result.get("optimizer_steps", result.get("completed_updates", 0)) or 0)
    context.model_writes += int(result.get("model_writes", 0) or 0)
    context.old_root_writes += int(result.get("old_root_writes", 0) or 0)
    context.stages[name] = "passed" if result.get("passed") is True else "failed"
    return result


def _fresh_model(*, seed: int, with_trace_probe: bool) -> C1Model:
    c1r_train.set_determinism(seed)
    return C1Model(C1Config(**c1r_contract.MODEL_CONFIG), with_trace_probe=with_trace_probe)


def _load_runtime(repo_root: Path) -> tuple[Any, Any, Any, Any]:
    store = c1_runtime.load_offline_records(repo_root / legacy_c1_contract.C0R_DATA_ROOT)
    cache_root = repo_root / legacy_c1_contract.CACHE_OUTPUT_ROOT
    bank = c1_runtime.load_target_bank(cache_root / "trace-target-bank.json")
    dataset = c1_runtime.CachedShardDataset(cache_root / "hidden")
    identity = c1_runtime.validate_cache_source_identity(dataset, store)
    target_audit = c1_runtime.audit_target_bank(bank)
    if identity.get("passed") is not True:
        raise ValueError(f"C1R cache/source identity failed: {identity.get('failures')}")
    if target_audit.get("passed") is not True or len(bank) != len(store):
        raise ValueError(f"C1R target bank audit failed: {target_audit.get('failures')}")
    return store, bank, dataset, c1_runtime.C1BatchProvider(dataset, store, bank)


def _records(store: Any, *, split: str, family: str | None = None) -> list[dict[str, Any]]:
    rows = [dict(row) for row in store.records if row.get("split") == split]
    if family is not None:
        rows = [row for row in rows if row.get("family") == family]
    return rows


def _balanced_sample(store: Any, *, split: str, per_family: int) -> list[dict[str, Any]]:
    """Reproduce the consumed C1 depth-balanced overfit selection."""

    selected: list[dict[str, Any]] = []
    for family in ("ERE", "CPS"):
        by_depth: dict[int, list[dict[str, Any]]] = {}
        for row in _records(store, split=split, family=family):
            by_depth.setdefault(int(row.get("reasoning_budget", 0)), []).append(row)
        for depth, rows in by_depth.items():
            rows.sort(
                key=lambda row: hashlib.sha256(
                    f"C1-OVERFIT32-{family}|{row['example_id']}".encode("utf-8")
                ).hexdigest()
            )
        depths = sorted(by_depth)
        cursors = {depth: 0 for depth in depths}
        family_rows: list[dict[str, Any]] = []
        while len(family_rows) < per_family:
            progressed = False
            for depth in depths:
                cursor = cursors[depth]
                if cursor < len(by_depth[depth]):
                    family_rows.append(by_depth[depth][cursor])
                    cursors[depth] += 1
                    progressed = True
                    if len(family_rows) == per_family:
                        break
            if not progressed:
                raise ValueError(f"insufficient {family} rows for balanced sample")
        selected.extend(family_rows)
    return selected


def _invariance(
    baseline: Mapping[str, list[float]],
    changed: Mapping[str, list[float]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(changed))
    if set(baseline) != set(changed):
        return {
            "passed": False,
            "n": len(keys),
            "prediction_invariance": 0.0,
            "max_abs_diff": float("inf"),
            "tolerance": tolerance,
        }
    differences: list[float] = []
    same_predictions = 0
    for key in keys:
        left = [float(value) for value in baseline[key]]
        right = [float(value) for value in changed[key]]
        if len(left) != 9 or len(right) != 9:
            raise ValueError("C1R invariance requires raw nine-class logits")
        differences.append(max(abs(a - b) for a, b in zip(left, right)))
        same_predictions += int(
            max(range(9), key=left.__getitem__)
            == max(range(9), key=right.__getitem__)
        )
    maximum = max(differences, default=0.0)
    return {
        "passed": bool(keys)
        and same_predictions == len(keys)
        and maximum <= tolerance,
        "n": len(keys),
        "prediction_invariance": same_predictions / len(keys) if keys else 0.0,
        "max_abs_diff": maximum,
        "tolerance": tolerance,
    }


def _paired_reports(runtime: Any, records: list[dict[str, Any]], intervention: str) -> dict[str, Any]:
    return runtime.evaluate_intervention(intervention, records=records)


def production_hooks(repo_root: Path, *, device: str = "cuda") -> StageHooks:
    """Build the real C1R stage adapter; no hook defaults to a fake PASS."""

    def fresh_init_parity(context: RunContext) -> Mapping[str, Any]:
        answer_only = _fresh_model(
            seed=c1r_contract.MODEL_SEED, with_trace_probe=False
        )
        joint_shape = _fresh_model(
            seed=c1r_contract.MODEL_SEED, with_trace_probe=True
        )
        answer_hash = _state_mapping_digest(answer_only.state_dict())
        joint_shared_hash = _state_mapping_digest(
            joint_shape.state_dict(), exclude_prefix="trace_probe."
        )
        return {
            "passed": answer_hash == joint_shared_hash
            and not answer_only.has_trace_probe
            and joint_shape.has_trace_probe,
            "answer_only_shared_state_sha256": answer_hash,
            "joint_shared_state_sha256": joint_shared_hash,
            "shared_initialization_exact": answer_hash == joint_shared_hash,
        }

    def answer_only_overfit32(context: RunContext) -> Mapping[str, Any]:
        model = _fresh_model(seed=c1r_contract.OVERFIT_SEED if hasattr(c1r_contract, "OVERFIT_SEED") else c1r_contract.MODEL_SEED, with_trace_probe=False)
        rows = _balanced_sample(context.data["store"], split="train", per_family=16)
        if len(rows) != 32:
            raise ValueError("C1R answer-only overfit requires 32 training records")
        result = c1r_train.train_answer_overfit32(model, [row["example_id"] for row in rows], context.data["provider"], context.root, device=device)
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def stage_a_train(context: RunContext) -> Mapping[str, Any]:
        model = _fresh_model(seed=c1r_contract.MODEL_SEED, with_trace_probe=False)
        result = c1r_train.train_stage_a(model, _records(context.data["store"], split="train"), context.data["provider"], context.root, device=device)
        context.data["model"] = model
        return result

    def stage_a_gate(context: RunContext, gate: str) -> Mapping[str, Any]:
        model = context.data["model"]
        store, dataset = context.data["store"], context.data["dataset"]
        validation = _records(store, split="validation")
        runtime = c1_evaluation_runtime.C1EvaluationRuntime(model, dataset, validation, device=device, batch_size=8)
        if gate == "G004":
            report = runtime.evaluate_answers(records=validation)
            return c1_qualification.gate_g004_validation(report) | {"report": report}
        if gate == "G005":
            ood = [row for row in store.records if row.get("split") in {"composition_ood", "entity_ood", "length_ood", "language_ood", "distractor_ood", "horizon_ood"}]
            report = c1_evaluation_runtime.C1EvaluationRuntime(model, dataset, ood, device=device, batch_size=8).evaluate_answers(records=ood)
            return c1_qualification.gate_g005_ood(report) | {"report": report}
        if gate == "G006":
            causal = _records(store, split="causal_pairs")
            causal_runtime = c1_evaluation_runtime.C1EvaluationRuntime(model, dataset, causal, device=device, batch_size=8)
            logits = causal_runtime.raw_answer_logits(records=causal)
            report = c1_evaluate.evaluate_causal(causal, lambda row: {"logits": logits[row["example_id"]]})
            return c1_qualification.gate_g006_causal(report) | {"report": report}
        if gate == "G008":
            report = {"zero_hidden": _paired_reports(runtime, validation, "zero_hidden"), "shuffled_hidden": _paired_reports(runtime, validation, "shuffled_hidden")}
            return c1_qualification.gate_g008_hidden(report) | {"report": report}
        if gate == "G009":
            report = {"h0": _paired_reports(runtime, validation, "h0_no_core"), "step5_shuffle": _paired_reports(runtime, validation, "step5_state_shuffle")}
            return c1_qualification.gate_g009_recurrence(report) | {"report": report}
        if gate == "G010":
            baseline = runtime.raw_answer_logits(records=validation)
            permuted = runtime.raw_answer_logits(intervention="slot_permutation", records=validation)
            stripped = c1_deployment.strip_trace_probe(model)
            stripped.model.to(device).eval()
            stripped_runtime = c1_evaluation_runtime.C1EvaluationRuntime(stripped.model, dataset, validation, device=device, batch_size=8)
            strip_logits = stripped_runtime.raw_answer_logits(records=validation)
            report = {
                "slot": _invariance(
                    baseline,
                    permuted,
                    tolerance=float(c1r_contract.THRESHOLDS["slot"]["logit_max_abs_diff"]),
                ),
                "strip": _invariance(
                    baseline,
                    strip_logits,
                    tolerance=float(c1r_contract.THRESHOLDS["strip"]["logit_max_abs_diff"]),
                ),
                "trace_probe_parameter_count": 0,
            }
            return c1_qualification.gate_g010_integrity(report) | {"report": report}
        raise ValueError(f"unknown C1R Stage A gate {gate}")

    def freeze_deployment(context: RunContext) -> Mapping[str, Any]:
        source_model = context.data["model"]
        state = c1_deployment.deployment_state(source_model)
        source_digest = _state_mapping_digest(state)
        full = _fresh_model(seed=c1r_contract.PROBE_SEED, with_trace_probe=True)
        loaded = full.load_state_dict(state, strict=False)
        expected_missing = {
            name for name in full.state_dict() if name.startswith("trace_probe.")
        }
        if loaded.unexpected_keys or set(loaded.missing_keys) != expected_missing:
            raise ValueError(f"deployment reload mismatch: {loaded}")
        loaded_digest = _state_mapping_digest(
            full.state_dict(), exclude_prefix="trace_probe."
        )
        full.to(device).eval()
        context.data["deployment_state"] = state
        context.data["deployment_state_sha256"] = source_digest
        context.data["model"] = full
        return {
            "passed": source_digest == loaded_digest,
            "deployment_state_sha256": source_digest,
            "reloaded_shared_state_sha256": loaded_digest,
            "trace_probe_seed": c1r_contract.PROBE_SEED,
            "trace_probe_attached": True,
            "model_writes": 0,
        }

    def cache_train_trajectories(context: RunContext) -> Mapping[str, Any]:
        model = context.data["model"].eval()
        dataset, bank = context.data["dataset"], context.data["bank"]
        cache: dict[str, Any] = {}
        rows = _records(context.data["store"], split="train")
        batch_size = 16
        chunks = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]
        started = time.perf_counter()
        prefetched = c1r_train.ordered_prefetch(
            chunks,
            lambda group: (
                group,
                dataset.get_batch([str(row["example_id"]) for row in group])[0],
            ),
            max_prefetch=2,
        )
        with torch.inference_mode():
            for group, cpu_batch in prefetched:
                source_hidden = cpu_batch["source_hidden"].to(device, non_blocking=True)
                source_mask = cpu_batch["source_mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        source_hidden, source_mask, return_trajectory=True
                    )
                trajectories = output["trajectory"].detach().to(torch.float16).cpu()
                for index, row in enumerate(group):
                    example_id = str(row["example_id"])
                    target = bank.targets[example_id]
                    target_rows = [{
                        "targets": target["token_ids"],
                        "mask": [True] * len(target["token_ids"]),
                        "step_indices": target["step_indices"],
                        "global_positions": target["global_positions"],
                        "local_positions": target["local_positions"],
                    }]
                    cache[example_id] = {
                        "trajectory": trajectories[index],
                        "target_rows": target_rows,
                    }
        context.data["trajectory_cache"] = cache
        context.data["trajectory_cache_source_deployment_state_sha256"] = context.data[
            "deployment_state_sha256"
        ]
        descriptors = [{
            "example_id": str(row["example_id"]),
            "family": row["family"],
            "token_ids": bank.targets[str(row["example_id"])]["token_ids"],
        } for row in rows]
        plan = c1r_coverage.build_coverage_plan(descriptors)
        coverage = c1r_coverage.coverage_report(plan)
        context.data["coverage_plan"] = plan
        context.data["coverage_report"] = coverage
        return {
            "passed": coverage.get("passed") is True and len(cache) == len(rows),
            "coverage": coverage,
            "records": len(cache),
            "dtype": "float16",
            "batch_size": batch_size,
            "wall_seconds": time.perf_counter() - started,
            "source_deployment_state_sha256": context.data[
                "trajectory_cache_source_deployment_state_sha256"
            ],
        }

    def frozen_probe_overfit(context: RunContext) -> Mapping[str, Any]:
        control_spec = c1r_contract.POSITIVE_CONTROL_CONFIG
        control = _fresh_model(seed=c1r_contract.PROBE_SEED, with_trace_probe=True)
        loaded = control.load_state_dict(context.data["deployment_state"], strict=False)
        if loaded.unexpected_keys or any(
            not name.startswith("trace_probe.") for name in loaded.missing_keys
        ):
            raise ValueError(f"probe control reload mismatch: {loaded}")
        rows = _balanced_sample(
            context.data["store"],
            split="train",
            per_family=int(control_spec["records_per_family"]),
        )
        spec = c1r_train.StageBSpec(
            epochs=int(control_spec["epochs"]),
            batch_size=int(control_spec["batch_size"]),
            family_batch_size=int(control_spec["family_batch_size"]),
            maximum_updates=int(control_spec["maximum_updates"]),
            trace_block_width=int(control_spec["trace_block_width"]),
            warmup_updates=int(control_spec["warmup_updates"]),
        )
        result = c1r_train.train_stage_b(
            control,
            rows,
            context.data["trajectory_cache"],
            context.root / "positive-control",
            device=device,
            spec=spec,
            model_seed=c1r_contract.PROBE_SEED,
        )
        trace = c1_evaluation_runtime.C1EvaluationRuntime(
            control,
            context.data["dataset"],
            rows,
            device=device,
            batch_size=8,
        ).trace_credit(
            context.data["bank"].targets,
            validation_records=rows,
            per_family=int(control_spec["records_per_family"]),
            bootstrap_seed=c1r_contract.BOOTSTRAP_SEED,
        )
        metrics_pass = all(
            float(trace["families"][family]["all_token_accuracy"]["point"])
            >= float(control_spec["trace_token_accuracy"])
            and float(trace["families"][family]["content_token_accuracy"]["point"])
            >= float(control_spec["trace_content_accuracy"])
            and float(trace["families"][family]["nll_margin"]["point"])
            >= float(control_spec["owner_nll_margin"])
            for family in ("ERE", "CPS")
        )
        result["training_passed"] = result.get("passed") is True
        result["trace_credit"] = trace
        result["passed"] = result.get("passed") is True and metrics_pass
        control.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fresh = _fresh_model(seed=c1r_contract.PROBE_SEED, with_trace_probe=True)
        fresh.load_state_dict(context.data["deployment_state"], strict=False)
        context.data["model"] = fresh
        result["positive_control"] = True
        return result

    def stage_b_train(context: RunContext) -> Mapping[str, Any]:
        rows = _records(context.data["store"], split="train")
        trajectory_source_matches = (
            context.data.get("trajectory_cache_source_deployment_state_sha256")
            == context.data.get("deployment_state_sha256")
        )
        result = c1r_train.train_stage_b(
            context.data["model"],
            rows,
            context.data["trajectory_cache"],
            context.root,
            device=device,
            model_seed=c1r_contract.PROBE_SEED,
        )
        schedule_audit = _coverage_schedule_audit(
            context.data["coverage_plan"], result["schedule"]
        )
        coverage = context.data["coverage_report"]
        expected_tokens = int(coverage["raw_target_tokens"])
        expected_blocks = int(coverage["blocks"])
        token_accounting = all(
            int(value) == expected_tokens
            for value in result["processed_target_tokens_by_epoch"].values()
        )
        block_accounting = all(
            int(value) == expected_blocks
            for value in result["processed_blocks_by_epoch"].values()
        )
        result.update({
            "coverage": coverage,
            "coverage_schedule_audit": schedule_audit,
            "coverage_schedule_matches": schedule_audit["rows_match"],
            "coverage_schedule_hash_matches": schedule_audit["hash_match"],
            "coverage_token_accounting": token_accounting,
            "coverage_block_accounting": block_accounting,
            "trajectory_source_deployment_state_sha256": context.data.get(
                "trajectory_cache_source_deployment_state_sha256"
            ),
            "trajectory_source_matches_frozen_deployment": trajectory_source_matches,
        })
        result["passed"] = bool(
            result.get("passed")
            and coverage.get("passed")
            and schedule_audit.get("passed") is True
            and token_accounting
            and block_accounting
            and trajectory_source_matches
        )
        return result

    def g007_trace(context: RunContext) -> Mapping[str, Any]:
        validation = _records(context.data["store"], split="validation")
        runtime = c1_evaluation_runtime.C1EvaluationRuntime(context.data["model"], context.data["dataset"], validation, device=device, batch_size=8)
        report = runtime.trace_credit(context.data["bank"].targets, validation_records=validation, per_family=256, bootstrap_seed=c1r_contract.BOOTSTRAP_SEED)
        return c1_qualification.gate_g007_trace(report) | {"report": report}

    def strip_reload(context: RunContext) -> Mapping[str, Any]:
        validation = _records(context.data["store"], split="validation")
        before = c1_evaluation_runtime.C1EvaluationRuntime(context.data["model"], context.data["dataset"], validation, device=device, batch_size=8).raw_answer_logits(records=validation)
        stripped = c1_deployment.strip_trace_probe(context.data["model"])
        reloaded = c1_deployment.reload_stripped(stripped.state_dict, C1Config(**c1r_contract.MODEL_CONFIG), device=device)
        after = c1_evaluation_runtime.C1EvaluationRuntime(reloaded.model, context.data["dataset"], validation, device=device, batch_size=8).raw_answer_logits(records=validation)
        invariance = _invariance(
            before,
            after,
            tolerance=float(c1r_contract.THRESHOLDS["strip"]["logit_max_abs_diff"]),
        )
        deployed_digest = _state_mapping_digest(reloaded.state_dict)
        path = context.root / "deployment" / "stripped-state.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": f"{SCHEMA_PREFIX}.deployment.v1",
            "identity": IDENTITY,
            "config": c1r_contract.MODEL_CONFIG,
            "model_state": reloaded.state_dict,
            "deployment_state_sha256": deployed_digest,
        }, path)
        passed = (
            invariance.get("passed") is True
            and not any(name.startswith("trace_probe.") for name in reloaded.state_dict)
            and deployed_digest == context.data["deployment_state_sha256"]
        )
        return {
            "passed": passed,
            "strip_invariance": invariance,
            "trace_probe_parameters": 0,
            "deployment_state_sha256": deployed_digest,
            "stage_a_deployment_state_sha256": context.data["deployment_state_sha256"],
            "deployment_path": path.relative_to(context.root).as_posix(),
            "deployment_sha256": c1r_artifacts.sha256_file(path),
            "model_writes": 1,
        }

    return StageHooks(fresh_init_parity=fresh_init_parity, answer_only_overfit32=answer_only_overfit32, stage_a_train=stage_a_train, stage_a_gate=stage_a_gate, freeze_deployment=freeze_deployment, cache_train_trajectories=cache_train_trajectories, frozen_probe_overfit=frozen_probe_overfit, stage_b_train=stage_b_train, g007_trace=g007_trace, strip_reload=strip_reload)


def _finish(context: RunContext, *, status: str, stopped_at: str | None, passed: bool, reason: str | None = None) -> dict[str, Any]:
    source = None
    stable = False
    try:
        source_before = context.data.get("source_before")
        source_after = _source_hashes(context.repo_root)
        source = source_identity(context.repo_root)
        stable = source_before == source_after
    except (OSError, ValueError):
        source_after = None
    old_roots_before = context.data.get("old_roots_before")
    try:
        old_roots_after = _old_root_metadata(context.repo_root)
    except OSError:
        old_roots_after = None
    old_roots_stable = (
        isinstance(old_roots_before, dict) and old_roots_before == old_roots_after
    )
    if not old_roots_stable:
        context.old_root_writes = max(1, context.old_root_writes)
    accounting_ok = stable and old_roots_stable and context.old_root_writes == 0
    if passed and not accounting_ok:
        passed = False
        status = "INCOMPLETE_V2_A_C1R_ACCOUNTING"
        stopped_at = "accounting"
        reason = reason or "source drift or old-root write detected"
    result: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.result.v1",
        "identity": IDENTITY,
        "formal": True,
        "status": status,
        "passed": bool(passed),
        "exit_code": 0 if passed else 1,
        "stopped_at": stopped_at,
        "unrun_after": stopped_at if not passed else None,
        "stage_status": dict(context.stages),
        "training_started": context.training_started,
        "optimizer_steps": context.optimizer_steps,
        "model_writes": context.model_writes,
        "old_root_writes": context.old_root_writes,
        "old_roots_stable": old_roots_stable,
        "old_roots_before": old_roots_before,
        "old_roots_after": old_roots_after,
        "device": context.data.get("device_audit"),
        "source_identity": source,
        "source_stable": stable,
        "source_after": source_after,
        "accounting_passed": accounting_ok,
        "authorizes": "two additional fresh C1R seeds only" if passed else "nothing",
        "c2_authorized": False,
        "v2a_passed": False,
    }
    if reason:
        result["reason"] = reason
    _write(context.root, "result.json", result)
    seal = _write_evidence_seal(context.root)
    replay = _audit_evidence_seal(context.root)
    result["evidence_seal_sha256"] = seal
    result["seal_replay"] = replay
    if replay.get("passed") is not True:
        result.update({"status": "INCOMPLETE_V2_A_C1R_SEAL", "passed": False, "exit_code": 1, "authorizes": "nothing"})
        _write(context.root, "result.json", result)
        result["evidence_seal_sha256"] = _write_evidence_seal(context.root)
        result["seal_replay"] = _audit_evidence_seal(context.root)
    return result | {"output_root": context.root.as_posix(), "lease_path": context.lease_path.as_posix()}


def _run_hook(context: RunContext, name: str, hook: Hook, filename: str) -> dict[str, Any]:
    value = _consume(context, name, hook(context))
    _write(context.root, filename, value)
    return value


def run_formal(
    repo_root: Path,
    *,
    output_root: Path | None = None,
    lease_path: Path | None = None,
    device: str = "cuda",
    hooks: StageHooks | None = None,
    require_preflight: bool = True,
    pin_auditor: Callable[[Path], Mapping[str, Any]] = audit_pins,
    device_auditor: Callable[[str], Mapping[str, Any]] = _cuda_device_audit,
) -> dict[str, Any]:
    """Run C1R's fail-stop stage machine exactly once.

    Tests may inject deterministic fake stages.  A real call without injected
    hooks loads the immutable runtime and uses ``production_hooks``; no stage
    after a failed stage is called.
    """
    repo_root = Path(repo_root).resolve()
    root = Path(output_root) if output_root is not None else repo_root / OUTPUT_ROOT
    lease = Path(lease_path) if lease_path is not None else repo_root / LEASE_PATH
    if root.exists() or lease.exists():
        return _refusal(root, lease, "REFUSE_V2_A_C1R_SINGLE_USE", "formal root or lease already exists")
    device_audit = dict(device_auditor(device))
    if device_audit.get("passed") is not True:
        return _refusal(
            root,
            lease,
            "REFUSE_V2_A_C1R_FORMAL_DEVICE",
            f"formal C1R requires the registered RTX 4070 CUDA device: {device_audit}",
        )
    if require_preflight:
        audit = _audit_c1r_preflight(repo_root / PREFLIGHT_ROOT, repo_root)
        if audit.get("passed") is not True:
            return _refusal(root, lease, "REFUSE_V2_A_C1R_PREFLIGHT_PREREQUISITE", "sealed C1R preflight PASS is required")
    try:
        _claim_single_use(identity=IDENTITY, output_root=root, lease_path=lease, formal=True)
        c1r_artifacts.snapshot_sources(repo_root, root)
        context = RunContext(repo_root=repo_root, root=root, lease_path=lease, device=device)
        context.data["source_before"] = _source_hashes(repo_root)
        context.data["device_audit"] = device_audit
        context.data["old_roots_before"] = _old_root_metadata(repo_root)
        if hooks is None:
            context.data["store"], context.data["bank"], context.data["dataset"], context.data["provider"] = _load_runtime(repo_root)
            context.data["records"] = context.data["store"].records
        pin = dict(pin_auditor(repo_root))
        _write(root, "contract-manifest.json", c1r_contract.c1r_contract_manifest() | {"pins": pin, "device": device_audit})
        if pin.get("passed") is not True:
            context.stages["prerequisites"] = "failed"
            _write(root, "prerequisites.json", pin)
            return _finish(
                context,
                status="INCOMPLETE_V2_A_C1R_PREREQUISITES",
                stopped_at="prerequisites",
                passed=False,
            )
        context.stages["prerequisites"] = "passed"
        _write(root, "prerequisites.json", pin)
        h = hooks or production_hooks(repo_root, device=device)
        if not _run_hook(context, "fresh_init_parity", h.fresh_init_parity, "fresh-init-parity.json").get("passed"):
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="fresh_init_parity", passed=False)
        if not _run_hook(context, "answer_only_overfit32", h.answer_only_overfit32, "answer-only-overfit32.json").get("passed"):
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="answer_only_overfit32", passed=False)
        stage_a = _run_hook(context, "stage_a_train", h.stage_a_train, "stage-a-training.json")
        if not stage_a.get("passed") or int(stage_a.get("completed_updates", STAGE_A_UPDATES)) != STAGE_A_UPDATES:
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="stage_a_train", passed=False)
        for gate in ("G004", "G005", "G006", "G008", "G009", "G010"):
            if h.stage_a_gate is None:
                gate_result = {"passed": False, "reason": "stage_a_gate hook was not provided"}
            else:
                gate_result = dict(h.stage_a_gate(context, gate))
            gate_result = _consume(context, gate, gate_result)
            _write(root, f"{gate.lower()}-stage-a.json", gate_result)
            if gate_result.get("passed") is not True:
                return _finish(context, status="FAIL_V2_A_C1R", stopped_at=gate, passed=False)
        if not _run_hook(context, "freeze_deployment", h.freeze_deployment, "deployed-freeze.json").get("passed"):
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="freeze_deployment", passed=False)
        if not _run_hook(context, "cache_train_trajectories", h.cache_train_trajectories, "train-trajectories.json").get("passed"):
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="cache_train_trajectories", passed=False)
        if not _run_hook(context, "frozen_probe_overfit", h.frozen_probe_overfit, "frozen-probe-overfit.json").get("passed"):
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="frozen_probe_overfit", passed=False)
        stage_b = _run_hook(context, "stage_b_train", h.stage_b_train, "stage-b-training.json")
        if not stage_b.get("passed") or int(stage_b.get("completed_updates", STAGE_B_UPDATES)) != STAGE_B_UPDATES:
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="stage_b_train", passed=False)
        if not _run_hook(context, "g007_trace", h.g007_trace, "g007-trace.json").get("passed"):
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="G007", passed=False)
        if not _run_hook(context, "strip_reload", h.strip_reload, "strip-reload.json").get("passed"):
            return _finish(context, status="FAIL_V2_A_C1R", stopped_at="strip_reload", passed=False)
        return _finish(context, status="PASS_V2_A_C1R", stopped_at=None, passed=True)
    except Exception as exc:
        if root.is_dir():
            context = locals().get("context")
            if not isinstance(context, RunContext):
                context = RunContext(repo_root=repo_root, root=root, lease_path=lease, device=device)
            return _finish(context, status="CRASH_V2_A_C1R", stopped_at=None, passed=False, reason=f"{type(exc).__name__}: {exc}")
        return _refusal(root, lease, "CRASH_V2_A_C1R", f"{type(exc).__name__}: {exc}")


__all__ = [
    "EXPOSURE_BLOCK", "IDENTITY", "LEASE_PATH", "NEW_SOURCE_FILES", "OUTPUT_ROOT", "PREFLIGHT_LEASE", "PREFLIGHT_ROOT", "SCHEMA_PREFIX", "STAGE_A_UPDATES", "STAGE_B_UPDATES", "RunContext", "StageHooks", "audit_pins", "run_formal", "run_preflight", "source_identity",
]
