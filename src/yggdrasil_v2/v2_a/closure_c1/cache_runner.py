from __future__ import annotations

"""Single-use C1 cache preflight and qualification.

This module owns only the infrastructure gate.  It never creates an optimizer
or learner model.  The full cache is written below ``root/hidden`` and every
terminal path is sealed without putting the seal digest into ``result.json``.
"""

from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import torch

from . import artifacts, cache, contract, data, runtime


SOURCE_FILES = artifacts.SOURCE_FILES


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha_value(value: Any) -> str:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "evidence-seal.json"
    }


def _seal(root: Path) -> str:
    path = root / "evidence-seal.json"
    _write_json(path, {"schema_version": f"{contract.SCHEMA_PREFIX}.evidence-seal.v1", "files": _tree_hashes(root)})
    return _sha_file(path)


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return artifacts.source_hashes(repo_root)


def _snapshot(repo_root: Path, root: Path) -> None:
    artifacts.snapshot_sources(repo_root, root)


def _refusal(identity: str, root: Path, lease: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.cache-refusal.v1",
        "identity": identity,
        "status": "REFUSE_V2_A_CLOSURE_C1_CACHE_SINGLE_USE",
        "passed": False,
        "exit_code": 2,
        "output_root": root.as_posix(),
        "lease_path": lease.as_posix(),
        "reason": "fixed output root or sibling lease already exists",
        "authorizes": "nothing",
    }


def _preflight_refusal(
    identity: str, root: Path, lease: Path, audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.cache-preflight-refusal.v1",
        "identity": identity,
        "status": "REFUSE_V2_A_CLOSURE_C1_CACHE_PREFLIGHT_PREREQUISITE",
        "passed": False,
        "exit_code": 2,
        "output_root": root.as_posix(),
        "lease_path": lease.as_posix(),
        "reason": "sealed cache preflight PASS for the current source identity is required",
        "preflight_audit": dict(audit),
        "authorizes": "nothing",
    }


def _claim(root: Path, lease: Path, identity: str, *, formal: bool) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=False, exist_ok=False)
    lease.parent.mkdir(parents=True, exist_ok=True)
    with lease.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"schema_version": f"{contract.SCHEMA_PREFIX}.cache-lease.v1", "identity": identity, "output_root": root.as_posix(), "formal": formal, "created_at": datetime.now(UTC).isoformat()}, sort_keys=True) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())


def _load_records(repo_root: Path, records: Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if records is not None:
        return list(records)
    store = runtime.load_offline_records(repo_root / contract.C0R_DATA_ROOT)
    return list(store.records)


def _pinned_c0r(repo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    roots: dict[str, Any] = {}
    for root_rel, result_hash, seal_hash in (
        (contract.C0R_DATA_ROOT, contract.C0R_DATA_RESULT_SHA256, contract.C0R_DATA_SEAL_SHA256),
        (contract.C0R_READINESS_ROOT, contract.C0R_READINESS_RESULT_SHA256, contract.C0R_READINESS_SEAL_SHA256),
    ):
        audit = artifacts.audit_pinned_root(
            repo_root / root_rel,
            result_sha256=result_hash,
            seal_sha256=seal_hash,
        )
        roots[root_rel.as_posix()] = audit
        if audit.get("passed") is not True:
            failures.append(f"pinned_root:{root_rel}")
    return {"passed": not failures, "failures": failures, "roots": roots}


def _bank_audit(records: Sequence[Mapping[str, Any]], *, formal: bool) -> dict[str, Any]:
    failures: list[str] = []
    counts: dict[str, int] = {}
    ids: list[str] = []
    hashes: dict[str, str] = {}
    for row in records:
        cell = f"{row.get('family')}/{row.get('split')}"
        counts[cell] = counts.get(cell, 0) + 1
        if type(row.get("example_id")) is not str or type(row.get("source_text")) is not str:
            failures.append(f"record_identity:{row.get('example_id')}")
        ids.append(str(row.get("example_id")))
        hashes[str(row.get("example_id"))] = data.source_byte_sha256(row.get("source_text", ""))
    expected = contract.DATASET_COUNTS
    if formal and counts != expected:
        failures.append("dataset_counts")
    if len(ids) != len(set(ids)):
        failures.append("duplicate_example_ids")
    return {"passed": not failures, "failures": failures, "counts": dict(sorted(counts.items())), "examples": len(records), "source_hashes_sha256": _sha_value(hashes)}


def _trace_audit(records: Sequence[Mapping[str, Any]], tokenizer: Any, *, formal: bool) -> tuple[dict[str, Any], Any, Any]:
    source_ids: set[int] = set()
    lengths: list[int] = []
    for record in records:
        token_ids = cache._token_ids(tokenizer, str(record["source_text"]))
        lengths.append(len(token_ids))
        source_ids.update(token_ids)
    lexicon = runtime.build_trace_lexicon(records, tokenizer, strict_size=formal)
    bank = runtime.materialize_target_bank(records, tokenizer, lexicon)
    audit = runtime.audit_target_bank(bank, lexicon)
    failures = list(audit.get("failures", []))
    if formal and len(source_ids) != contract.SOURCE_LEXICON_SIZE:
        failures.append(f"source_lexicon_size:{len(source_ids)}")
    if formal and lexicon.size != contract.TRACE_VOCAB_SIZE:
        failures.append(f"trace_lexicon_size:{lexicon.size}")
    if formal and len(bank) != sum(contract.DATASET_COUNTS.values()):
        failures.append(f"trace_examples:{len(bank)}")
    if len(lexicon.grammar_ids) != 42:
        failures.append(f"grammar_id_count:{len(lexicon.grammar_ids)}")
    if tuple(lexicon.grammar_ids) != tuple(contract.TRACE_GRAMMAR_TOKEN_IDS):
        failures.append("grammar_id_identity")
    if any(length < 1 or length > contract.CACHE_MAX_SOURCE_TOKENS for length in lengths):
        failures.append("source_length_bounds")
    tokenizer_fast = getattr(tokenizer, "is_fast", None)
    if formal and tokenizer_fast is not True:
        failures.append("tokenizer_not_fast")
    return ({"passed": not failures, "failures": failures, "source_lexicon_size": len(source_ids), "trace_lexicon_size": lexicon.size, "grammar_ids": len(lexicon.grammar_ids), "examples": len(bank), "target_oov": 0, "tokenizer_fast": tokenizer_fast, "add_special_tokens": False, "minimum_source_tokens": min(lengths) if lengths else 0, "maximum_source_tokens": max(lengths) if lengths else 0, "source_lengths_in_bounds": bool(lengths) and all(1 <= length <= contract.CACHE_MAX_SOURCE_TOKENS for length in lengths)}, lexicon, bank)


def _indexed_replay(hidden_root: Path, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        import numpy as np
        import torch

        dataset = cache.CachedShardDataset(hidden_root)
        ordered = cache._ordered(records)
        expected = [str(cache._cache_view(row)["example_id"]) for row in ordered]
        observed = [str(value) for shard in dataset.shard_paths for value in cache._load_shard(shard)["example_ids"]]
        if observed != expected:
            return {"passed": False, "failures": ["index_order"]}
        by_cell: dict[str, list[str]] = {}
        for row in ordered:
            public = cache._cache_view(row)
            cell = f"{public.get('family')}/{public.get('split')}"
            by_cell.setdefault(cell, []).append(str(public["example_id"]))
        sample = [value for cell in sorted(by_cell) for value in (by_cell[cell][0], by_cell[cell][-1])]
        sample = sorted(set(sample), key=lambda value: hashlib.sha256(f"K007|{value}".encode()).hexdigest())
        batch, metadata = dataset.get_batch(sample)
        if len(metadata) != len(sample) or batch["source_hidden"].shape[0] != len(sample):
            return {"passed": False, "failures": ["index_get_batch"]}
        wanted = set(sample)
        direct: dict[str, tuple[torch.Tensor, int]] = {}
        for path in dataset.shard_paths:
            payload = cache._load_shard(path)
            ids = [str(value) for value in payload["example_ids"]]
            if not wanted.intersection(ids):
                continue
            shape = tuple(int(value) for value in payload["packed_shape"])
            packed = np.memmap(
                hidden_root / str(payload["packed_hidden"]),
                mode="r",
                dtype=np.float16,
                shape=shape,
            )
            for index, example_id in enumerate(ids):
                if example_id not in wanted:
                    continue
                length = int(payload["token_lengths"][index])
                offset = int(payload["offsets"][index])
                direct[example_id] = (
                    torch.from_numpy(packed[offset : offset + length].copy()),
                    length,
                )
            del packed
        failures: list[str] = []
        for index, example_id in enumerate(sample):
            if example_id not in direct:
                failures.append(f"missing_direct:{example_id}")
                continue
            row, length = direct[example_id]
            observed_mask = batch["source_mask"][index]
            if int(observed_mask.sum()) != length or not bool(observed_mask[:length].all()) or bool(observed_mask[length:].any()):
                failures.append(f"mask:{example_id}")
            if not torch.equal(batch["source_hidden"][index, :length].cpu(), row):
                failures.append(f"hidden:{example_id}")
            if bool(batch["source_hidden"][index, length:].count_nonzero()):
                failures.append(f"padding:{example_id}")
        return {"passed": not failures, "failures": failures, "examples": len(expected), "sampled": len(sample), "cells": len(by_cell)}
    except Exception as exc:
        return {"passed": False, "failures": [f"index:{type(exc).__name__}:{exc}"]}


def _hidden_manifest_audit(hidden_root: Path, rows: Sequence[Mapping[str, Any]], cache_report: Mapping[str, Any]) -> dict[str, Any]:
    """Close the parts of K005/K006 that are not guaranteed by cache.audit_cache."""
    failures = list(cache_report.get("failures", []))
    try:
        manifest = json.loads((hidden_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"passed": False, "failures": failures + [f"manifest:{exc}"]}
    if manifest.get("hidden_width") != 2048:
        failures.append(f"hidden_width:{manifest.get('hidden_width')}")
    if manifest.get("examples") != len(rows):
        failures.append(f"examples:{manifest.get('examples')}")
    if manifest.get("full_token_hidden_saved") is not True:
        failures.append("full_token_hidden_saved")
    if manifest.get("dtype") != "float16":
        failures.append(f"dtype:{manifest.get('dtype')}")
    if manifest.get("qwen_lm_head_present") is not False or manifest.get("qwen_trainable_parameters") != 0:
        failures.append("text_only_encoder_identity")
    if manifest.get("qwen_component") != "Qwen3_5TextModel":
        failures.append(f"qwen_component:{manifest.get('qwen_component')}")
    if manifest.get("model_id") != contract.MODEL_ID or manifest.get("revision") != contract.MODEL_REVISION:
        failures.append("qwen_revision_identity")
    if manifest.get("transformers_version") != contract.TRANSFORMERS_VERSION:
        failures.append("transformers_version")
    return {"passed": not failures, "failures": failures, "hidden_width": manifest.get("hidden_width"), "examples": manifest.get("examples")}


def _run_preflight_body(repo_root: Path, root: Path, *, tokenizer: Any | None, backbone: Any | None, records: Sequence[Mapping[str, Any]] | None, device: str) -> dict[str, Any]:
    rows = _load_records(repo_root, records)
    prerequisite = _pinned_c0r(repo_root)
    bank_report = _bank_audit(rows, formal=records is None)
    tok = tokenizer or cache.load_qwen35_tokenizer(local_files_only=True)
    trace, lexicon, bank = _trace_audit(rows, tok, formal=False)
    bank_digest = bank.digest
    lexicon_digest = lexicon.digest
    public_rows: Sequence[Mapping[str, Any] | data.PublicRecord]
    if records is None:
        public_rows = data.load_records(repo_root / contract.C0R_DATA_ROOT)
        del rows, bank, lexicon
    else:
        public_rows = rows
    longest = sorted(public_rows, key=lambda row: len(cache._token_ids(tok, str(cache._cache_view(row)["source_text"]))), reverse=True)[:8]
    hidden = {"passed": True, "examples": len(longest), "finite": True, "hidden_width": None}
    production_backbone = backbone is None
    active_backbone = backbone
    if active_backbone is None:
        active_backbone = cache.load_qwen35_text_only(device=device, local_files_only=True)
    if longest:
        input_ids, mask, _ = cache._encode_batch(tok, longest, contract.CACHE_MAX_SOURCE_TOKENS)
        import torch
        with torch.inference_mode():
            value = cache._forward_hidden(active_backbone, input_ids.to(device), mask.to(device))
        hidden = {"passed": bool(torch.isfinite(value).all()) and int(value.shape[-1]) == 2048 and not hasattr(active_backbone, "lm_head") and (not production_backbone or type(active_backbone).__name__ == "Qwen3_5TextModel"), "examples": len(longest), "finite": bool(torch.isfinite(value).all()), "hidden_width": int(value.shape[-1]), "component": type(active_backbone).__name__, "lm_head_present": hasattr(active_backbone, "lm_head")}
    return {"K001": prerequisite, "K002": bank_report, "K003": {"passed": hidden["passed"] and trace["passed"], "failures": list(trace["failures"]) if not trace["passed"] else [], "longest_eight": hidden, "full_bank_length_audit": trace["passed"]}, "K004": trace, "K005": {"passed": True, "failures": [], "interpretation": "preflight does not build the large hidden cache"}, "K006": {"passed": True, "failures": []}, "K007": {"passed": True, "failures": []}, "K008": {"passed": True, "failures": []}, "trace_bank_digest": bank_digest, "lexicon_digest": lexicon_digest}


def run_cache_preflight(repo_root: Path, output_root: Path | None = None, lease_path: Path | None = None, device: str = "cuda", tokenizer: Any | None = None, backbone: Any | None = None, *, records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = Path(output_root) if output_root is not None else repo_root / contract.CACHE_PREFLIGHT_ROOT
    lease = Path(lease_path) if lease_path is not None else repo_root / contract.CACHE_PREFLIGHT_LEASE
    if root.exists() or lease.exists():
        return _refusal(contract.CACHE_IDENTITY, root, lease)
    fixed_root = (repo_root / contract.CACHE_PREFLIGHT_ROOT).resolve()
    if root.resolve() == fixed_root and (
        records is not None
        or tokenizer is not None
        or backbone is not None
        or str(device) != "cuda"
    ):
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.cache-preflight-refusal.v1",
            "identity": contract.CACHE_IDENTITY,
            "status": "REFUSE_V2_A_CLOSURE_C1_CACHE_PREFLIGHT_INJECTION",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "fixed cache preflight requires the pinned local tokenizer, real Qwen3_5TextModel, full C0R bank, and CUDA",
            "authorizes": "nothing",
        }
    if root.resolve() == fixed_root and not torch.cuda.is_available():
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.cache-preflight-refusal.v1",
            "identity": contract.CACHE_IDENTITY,
            "status": "REFUSE_V2_A_CLOSURE_C1_CACHE_PREFLIGHT_DEVICE",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "fixed cache preflight requires an available CUDA runtime",
            "authorizes": "nothing",
        }
    started = time.perf_counter()
    try:
        _claim(root, lease, contract.CACHE_IDENTITY, formal=False)
        before = _source_hashes(repo_root)
        _snapshot(repo_root, root)
        audits = _run_preflight_body(repo_root, root, tokenizer=tokenizer, backbone=backbone, records=records, device=device)
        after = _source_hashes(repo_root)
        passed = all(value.get("passed") is True for key, value in audits.items() if key.startswith("K")) and before == after
        result = {"schema_version": f"{contract.SCHEMA_PREFIX}.cache-preflight-result.v1", "identity": contract.CACHE_IDENTITY, "formal": False, "status": "PASS_V2_A_CLOSURE_C1_CACHE_PREFLIGHT" if passed else "FAIL_V2_A_CLOSURE_C1_CACHE_PREFLIGHT", "passed": passed, "exit_code": 0 if passed else 1, "device": device, "audits": audits, "source_identity": artifacts.source_identity(before), "source_stable": before == after, "training_started": False, "optimizer_steps": 0, "model_writes": 0, "wall_seconds": time.perf_counter() - started, "authorizes": "formal C1 cache qualification only" if passed else "nothing"}
        _write_json(root / "result.json", result)
        _write_json(root / "run-state.json", {"identity": contract.CACHE_IDENTITY, "source_stable": before == after, "training_started": False, "optimizer_steps": 0, "model_writes": 0})
    except Exception as exc:
        result = {"schema_version": f"{contract.SCHEMA_PREFIX}.cache-preflight-crash.v1", "identity": contract.CACHE_IDENTITY, "formal": False, "status": "CRASH_V2_A_CLOSURE_C1_CACHE_PREFLIGHT", "passed": False, "exit_code": 1, "error_type": type(exc).__name__, "error": str(exc), "training_started": False, "optimizer_steps": 0, "model_writes": 0, "authorizes": "nothing"}
        _write_json(root / "crash.json", result)
        _write_json(root / "result.json", result)
    result["evidence_seal_sha256"] = _seal(root)
    result["output_root"] = root.as_posix()
    return result


def run_cache_qualification(repo_root: Path, output_root: Path | None = None, lease_path: Path | None = None, device: str = "cuda", tokenizer: Any | None = None, backbone: Any | None = None, *, records: Sequence[Mapping[str, Any]] | None = None, formal: bool = True) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = Path(output_root) if output_root is not None else Path(repo_root) / contract.CACHE_OUTPUT_ROOT
    lease = Path(lease_path) if lease_path is not None else Path(repo_root) / contract.CACHE_LEASE_PATH
    if root.exists() or lease.exists():
        return _refusal(contract.CACHE_IDENTITY, root, lease)
    if not formal and root.resolve() == (repo_root / contract.CACHE_OUTPUT_ROOT).resolve():
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.cache-refusal.v1",
            "identity": contract.CACHE_IDENTITY,
            "status": "REFUSE_V2_A_CLOSURE_C1_CACHE_NONFORMAL_FIXED_ROOT",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "formal cache root cannot be consumed by a nonformal run",
            "authorizes": "nothing",
        }
    if formal and (str(device) != "cuda" or not torch.cuda.is_available()):
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.cache-refusal.v1",
            "identity": contract.CACHE_IDENTITY,
            "status": "REFUSE_V2_A_CLOSURE_C1_CACHE_FORMAL_DEVICE",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "formal cache qualification requires an available CUDA runtime",
            "authorizes": "nothing",
        }
    if formal and (records is not None or tokenizer is not None or backbone is not None):
        return {
            "schema_version": f"{contract.SCHEMA_PREFIX}.cache-refusal.v1",
            "identity": contract.CACHE_IDENTITY,
            "status": "REFUSE_V2_A_CLOSURE_C1_CACHE_FORMAL_INJECTION",
            "passed": False,
            "exit_code": 2,
            "output_root": root.as_posix(),
            "lease_path": lease.as_posix(),
            "reason": "formal cache qualification forbids injected records, tokenizer, or backbone",
            "authorizes": "nothing",
        }
    preflight: dict[str, Any] = {"passed": True, "required": False}
    if formal:
        current_identity = artifacts.source_identity(_source_hashes(repo_root))
        preflight = artifacts.audit_preflight_root(
            repo_root / contract.CACHE_PREFLIGHT_ROOT,
            identity=contract.CACHE_IDENTITY,
            status="PASS_V2_A_CLOSURE_C1_CACHE_PREFLIGHT",
            source_identity_value=current_identity,
        )
        longest = (
            preflight.get("result", {})
            .get("audits", {})
            .get("K003", {})
            .get("longest_eight", {})
        )
        production_checks = {
            "device": preflight.get("result", {}).get("device") == "cuda",
            "full_bank": preflight.get("result", {}).get("audits", {}).get("K002", {}).get("examples") == 26_624,
            "real_qwen_component": longest.get("component") == "Qwen3_5TextModel",
            "no_lm_head": longest.get("lm_head_present") is False,
            "eight_examples": longest.get("examples") == 8,
            "finite_width": longest.get("finite") is True and longest.get("hidden_width") == 2048,
        }
        preflight["production_checks"] = production_checks
        preflight["passed"] = preflight.get("passed") is True and all(
            production_checks.values()
        )
        if preflight.get("passed") is not True:
            return _preflight_refusal(
                contract.CACHE_IDENTITY, root, lease, preflight
            )
    started = time.perf_counter()
    gate_status = {gate: "not_run" for gate in contract.CACHE_GATE_IDS}
    audits: dict[str, Any] = {}
    before: dict[str, str] = {}

    try:
        _claim(root, lease, contract.CACHE_IDENTITY, formal=formal)
        before = _source_hashes(repo_root)
        _snapshot(repo_root, root)
        _write_json(root / "contract-manifest.json", contract.cache_contract_manifest())

        def terminal(status: str, *, stopped_at: str | None) -> dict[str, Any]:
            after = _source_hashes(repo_root)
            stable = source_before_equal(before, after)
            if not stable:
                gate_status["K008"] = "failed"
                if status == "PASS_V2_A_CLOSURE_C1_CACHE_QUALIFICATION":
                    status = "INCOMPLETE_V2_A_CLOSURE_C1_CACHE_QUALIFICATION"
                    stopped_at = "K008"
            gates = {
                gate: state == "passed" for gate, state in gate_status.items()
            }
            diagnostic_passed = (
                status == "PASS_V2_A_CLOSURE_C1_CACHE_QUALIFICATION"
                and set(gates) == set(contract.CACHE_GATE_IDS)
                and all(gates.values())
            )
            if not formal and diagnostic_passed:
                status = "COMPLETE_V2_A_CLOSURE_C1_CACHE_NONFORMAL_DIAGNOSTIC"
            passed = formal and diagnostic_passed
            hidden_manifest_path = root / "hidden" / "manifest.json"
            hidden_manifest = (
                json.loads(hidden_manifest_path.read_text(encoding="utf-8"))
                if hidden_manifest_path.is_file()
                else {}
            )
            target_path = root / "trace-target-bank.json"
            result = {
                "schema_version": f"{contract.SCHEMA_PREFIX}.cache-result.v1",
                "identity": contract.CACHE_IDENTITY,
                "formal": formal,
                "status": status,
                "passed": passed,
                "exit_code": 0 if passed else 1,
                "gates": gates,
                "gate_status": dict(gate_status),
                "stopped_at": stopped_at,
                "unrun_gates": [
                    gate for gate, state in gate_status.items() if state == "not_run"
                ],
                "audits": audits,
                "preflight_audit": preflight,
                "examples": hidden_manifest.get("examples"),
                "source_tokens": hidden_manifest.get("source_tokens"),
                "cache_shards": len(hidden_manifest.get("shards", [])),
                "cache_bytes": sum(
                    path.stat().st_size
                    for path in (root / "hidden").glob("shard_*")
                    if path.is_file()
                ),
                "cache_build_seconds": hidden_manifest.get("build_seconds"),
                "trace_bank_sha256": _sha_file(target_path)
                if target_path.is_file()
                else None,
                "source_identity": artifacts.source_identity(before),
                "training_started": False,
                "optimizer_steps": 0,
                "model_writes": 0,
                "source_stable": stable,
                "wall_seconds": time.perf_counter() - started,
                "authorizes": "C1 single-seed launcher only"
                if passed
                else "nothing",
            }
            if not formal:
                result["diagnostic_passed"] = diagnostic_passed
            _write_json(root / "audits.json", audits)
            _write_json(root / "result.json", result)
            _write_json(
                root / "run-state.json",
                {
                    "identity": contract.CACHE_IDENTITY,
                    "formal": formal,
                    "source_identity": result["source_identity"],
                    "source_stable": stable,
                    "training_started": False,
                    "optimizer_steps": 0,
                    "model_writes": 0,
                    "stopped_at": stopped_at,
                },
            )
            seal = _seal(root)
            replay = artifacts.audit_evidence_seal(root)
            if replay.get("passed") is not True:
                gate_status["K008"] = "failed"
                result = {
                    **result,
                    "status": "INCOMPLETE_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                    "passed": False,
                    "exit_code": 1,
                    "gates": {**result["gates"], "K008": False},
                    "gate_status": dict(gate_status),
                    "authorizes": "nothing",
                    "seal_failure": replay,
                }
                _write_json(root / "seal-failure.json", replay)
                _write_json(root / "result.json", result)
                seal = _seal(root)
                replay = artifacts.audit_evidence_seal(root)
            return {
                **result,
                "evidence_seal_sha256": seal,
                "seal_replay": replay,
                "output_root": root.as_posix(),
            }

        prerequisite = _pinned_c0r(repo_root) if formal else {"passed": True, "failures": []}
        audits["K001"] = prerequisite
        k001 = prerequisite.get("passed") is True
        gate_status["K001"] = "passed" if k001 else "failed"
        if not k001:
            return terminal(
                "INCOMPLETE_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                stopped_at="K001",
            )

        rows = _load_records(repo_root, records)
        bank = _bank_audit(rows, formal=formal)
        audits["K002"] = bank
        k002 = bank.get("passed") is True
        gate_status["K002"] = "passed" if k002 else "failed"
        if not k002:
            return terminal(
                "INCOMPLETE_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                stopped_at="K002",
            )

        tok = tokenizer or cache.load_qwen35_tokenizer(local_files_only=True)
        trace, lexicon, target_bank = _trace_audit(rows, tok, formal=formal)
        preflight_k003 = (
            preflight.get("result", {}).get("audits", {}).get("K003", {})
            if formal
            else {"passed": True}
        )
        k003_report = {
            "passed": trace.get("source_lengths_in_bounds", trace.get("passed")) is True
            and (trace.get("tokenizer_fast") is True if formal else True)
            and preflight_k003.get("passed") is True,
            "minimum_source_tokens": trace.get("minimum_source_tokens"),
            "maximum_source_tokens": trace.get("maximum_source_tokens"),
            "tokenizer_fast": trace.get("tokenizer_fast"),
            "add_special_tokens": trace.get("add_special_tokens"),
            "sealed_longest_eight_preflight": preflight_k003,
        }
        audits["K003"] = k003_report
        k003 = k003_report["passed"] is True
        gate_status["K003"] = "passed" if k003 else "failed"
        if not k003:
            return terminal(
                "FAIL_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                stopped_at="K003",
            )

        audits["K004"] = trace
        k004 = trace.get("passed") is True
        gate_status["K004"] = "passed" if k004 else "failed"
        if not k004:
            return terminal(
                "FAIL_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                stopped_at="K004",
            )

        target_path = root / "trace-target-bank.json"
        runtime.save_target_bank(target_bank, target_path)
        target_report = runtime.audit_target_bank(target_path, lexicon)
        public_rows: Sequence[Mapping[str, Any] | data.PublicRecord]
        if records is None:
            public_rows = data.load_records(repo_root / contract.C0R_DATA_ROOT)
            del rows, lexicon, target_bank
        else:
            public_rows = rows
        hidden_root = root / "hidden"
        cache.build_cache(repo_root, repo_root / contract.C0R_DATA_ROOT, hidden_root, records=public_rows, tokenizer=tok, backbone=backbone, device=device, max_length=contract.CACHE_MAX_SOURCE_TOKENS, inference_batch_size=contract.CACHE_INFERENCE_BATCH_SIZE, shard_size=contract.CACHE_SHARD_SIZE)
        cache_report = cache.audit_cache(hidden_root, records=public_rows)
        hidden_report = _hidden_manifest_audit(hidden_root, public_rows, cache_report)
        audits["K005"] = hidden_report
        k005 = hidden_report.get("passed") is True
        gate_status["K005"] = "passed" if k005 else "failed"
        if not k005:
            return terminal(
                "FAIL_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                stopped_at="K005",
            )

        audits["K006"] = cache_report
        k006 = cache_report.get("passed") is True
        gate_status["K006"] = "passed" if k006 else "failed"
        if not k006:
            return terminal(
                "FAIL_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                stopped_at="K006",
            )

        index_report = _indexed_replay(hidden_root, public_rows)
        audits["K007"] = index_report
        k007 = index_report.get("passed") is True
        gate_status["K007"] = "passed" if k007 else "failed"
        if not k007:
            return terminal(
                "FAIL_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                stopped_at="K007",
            )

        hidden_manifest = json.loads((hidden_root / "manifest.json").read_text(encoding="utf-8"))
        hidden_bytes = sum(
            path.stat().st_size for path in hidden_root.glob("shard_*") if path.is_file()
        )
        build_seconds = hidden_manifest.get("build_seconds")
        strict_accounting = {
            "target_bank": target_report.get("passed") is True,
            "examples": hidden_manifest.get("examples") == 26_624,
            "source_tokens_positive": type(hidden_manifest.get("source_tokens")) is int
            and int(hidden_manifest.get("source_tokens", 0)) > 0,
            "shards": len(hidden_manifest.get("shards", [])) == 416,
            "bytes": hidden_bytes > 0,
            "build_seconds": isinstance(build_seconds, (int, float))
            and not isinstance(build_seconds, bool)
            and math.isfinite(float(build_seconds))
            and float(build_seconds) > 0,
            "trace_bank_hash": len(_sha_file(target_path)) == 64,
            "source_stable": source_before_equal(before, _source_hashes(repo_root)),
            "no_training": True,
        }
        k008_report = {
            "checks": strict_accounting,
            "passed": all(strict_accounting.values()) if formal else target_report.get("passed") is True,
            "target_bank": target_report,
            "cache_bytes": hidden_bytes,
            "build_seconds": build_seconds,
            "terminal_seal_mode": "non-recursive seal is written and replayed after result.json",
        }
        audits["K008"] = k008_report
        k008 = k008_report["passed"] is True
        gate_status["K008"] = "passed" if k008 else "failed"
        if not k008:
            return terminal(
                "INCOMPLETE_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
                stopped_at="K008",
            )
        return terminal(
            "PASS_V2_A_CLOSURE_C1_CACHE_QUALIFICATION",
            stopped_at=None,
        )
    except Exception as exc:
        result = {"schema_version": f"{contract.SCHEMA_PREFIX}.cache-crash.v1", "identity": contract.CACHE_IDENTITY, "formal": formal, "status": "CRASH_V2_A_CLOSURE_C1_CACHE_QUALIFICATION", "passed": False, "exit_code": 1, "error_type": type(exc).__name__, "error": str(exc), "training_started": False, "optimizer_steps": 0, "model_writes": 0, "authorizes": "nothing"}
        result["gates"] = {gate: state == "passed" for gate, state in gate_status.items()}
        result["gate_status"] = gate_status
        result["audits"] = audits
        _write_json(root / "crash.json", result)
        _write_json(root / "result.json", result)
        _write_json(root / "run-state.json", {"identity": contract.CACHE_IDENTITY, "formal": formal, "source_stable": False, "training_started": False, "optimizer_steps": 0, "model_writes": 0})
    result["evidence_seal_sha256"] = _seal(root)
    result["seal_replay"] = artifacts.audit_evidence_seal(root)
    result["output_root"] = root.as_posix()
    return result


def source_before_equal(before: Mapping[str, str], after: Mapping[str, str]) -> bool:
    return dict(before) == dict(after)


__all__ = ["run_cache_preflight", "run_cache_qualification"]
