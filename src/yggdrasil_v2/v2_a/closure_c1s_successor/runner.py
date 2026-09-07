from __future__ import annotations

"""Fail-closed execution of C1S successor S1, S2, and S3.

Each public stage has a distinct fixed root and sibling lease.  This module
never chains stages automatically: callers must inspect a sealed predecessor
PASS and explicitly invoke the next command.
"""

from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
import gc
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ..closure_c1.cache import CachedShardDataset, audit_cache, load_qwen35_tokenizer
from ..closure_c1.evaluate import evaluate_behavior
from ..closure_c1.qualification import (
    gate_g004_validation,
    gate_g005_ood,
    gate_g010_integrity,
)
from ..closure_c1.runtime import load_offline_records, validate_cache_source_identity
from ..closure_c1s.model import C1SConfig, C1SModel, strip_training_auxiliary
from . import artifacts, contract
from .evaluate import (
    DYNAMIC_STATE_FEATURES,
    causal_pair_margin_report,
    dynamic_state_qualified,
    evaluate_predictions,
    gate_g006_causal_behavior,
    gate_g006_causal_margin,
    gate_g008_hidden,
    gate_g009_recurrence,
    operation_active_qualified,
    paired_eval_gain,
    paired_intervention_margin_report,
    s1_gate,
    s2_gate,
    validate_k1_single_slot_control,
)
from .objective import (
    CausalMargins,
    LossWeights,
    SuccessorDiagnosticHeads,
    intervention_outputs,
)
from .runtime import C1SSuccessorRuntime
from .targets import (
    audit_target_bank,
    build_split_ledger,
    materialize_target_bank,
    select_s1_ids,
    select_s2_ids,
)
from .train import (
    TrainSpec,
    build_balanced_schedule,
    build_overfit_schedule,
    save_endpoint,
    schedule_report,
    set_determinism,
    train_fixed_endpoint,
)


DATA_ROOT = Path("artifacts/v2-a/closure-c0r-data-trace-20260824-1")
CACHE_ROOT = Path("artifacts/v2-a/closure-c1-cache-20260825-1/hidden")
S0_ROOT = Path("tmp/v2-a-closure-c1s-s0-preflight-20260828-1")
TARGET_BANK_NAME = "target-bank.json.gz"
SPLIT_LEDGER_NAME = "split-ledger.json"
FULL_RECORD_COUNT = 26_624
FAMILIES = ("ERE", "CPS")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stage_config(stage: str) -> tuple[str, Path, Path, Path, Path, bool]:
    stage = stage.upper()
    if stage == "S1":
        return (
            contract.S1_IDENTITY,
            contract.S1_ROOT,
            contract.S1_LEASE,
            contract.S1_PREFLIGHT_ROOT,
            contract.S1_PREFLIGHT_LEASE,
            False,
        )
    if stage == "S2":
        return (
            contract.S2_IDENTITY,
            contract.S2_ROOT,
            contract.S2_LEASE,
            contract.S2_PREFLIGHT_ROOT,
            contract.S2_PREFLIGHT_LEASE,
            False,
        )
    if stage == "S3":
        return (
            contract.S3_IDENTITY,
            contract.S3_ROOT,
            contract.S3_LEASE,
            contract.S3_PREFLIGHT_ROOT,
            contract.S3_PREFLIGHT_LEASE,
            True,
        )
    raise ValueError(f"unknown successor stage: {stage}")


def _absolute(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _refusal(stage: str, root: Path, lease: Path, reason: str) -> dict[str, Any]:
    identity, *_rest, formal = _stage_config(stage)
    return {
        "schema_version": f"{contract.SCHEMA_PREFIX}.refusal.v1",
        "identity": identity,
        "stage": stage.upper(),
        "formal": formal,
        "status": f"REFUSE_V2_A_C1S_{stage.upper()}_SINGLE_USE",
        "passed": False,
        "exit_code": 2,
        "output_root": root.as_posix(),
        "lease_path": lease.as_posix(),
        "reason": reason,
        "authorizes": "nothing",
        "c2_authorized": False,
        "v2a_passed": False,
    }


def _device_audit(device: str = "cuda") -> dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        return {"passed": False, "requested": device, "reason": "CUDA is required"}
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    free, total = torch.cuda.mem_get_info(index)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    return {
        "passed": "RTX 4070" in name and capability[0] >= 8 and bf16_supported,
        "requested": device,
        "index": index,
        "name": name,
        "capability": list(capability),
        "device_count": torch.cuda.device_count(),
        "free_memory_bytes": int(free),
        "total_memory_bytes": int(total),
        "bf16_supported": bf16_supported,
        "distributed_boundary": {
            "ddp_used": False,
            "two_gpu_assumed": False,
            "s2_arms_serial_on_current_host": torch.cuda.device_count() == 1,
        },
    }


def _audit_static_pins(repo_root: Path, *, replay: bool) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, pin in contract.PINNED_PREDECESSOR_ROOTS.items():
        # The 68 GB cache is qualified independently by audit_cache.  Replaying
        # its outer seal here would hash the same bytes twice.
        use_replay = replay and name != "c1_cache"
        rows[name] = artifacts.audit_external_pin(repo_root, pin, replay=use_replay)
    return {"passed": all(row.get("passed") is True for row in rows.values()), "roots": rows}


def _preflight_identity(stage: str) -> str:
    return f"{_stage_config(stage)[0]}-PREFLIGHT"


def _audit_preflight(repo_root: Path, stage: str) -> dict[str, Any]:
    stage = stage.upper()
    _, _, _, relative, _, _ = _stage_config(stage)
    root = _absolute(repo_root, relative)
    try:
        result = artifacts.read_json(root / "result.json")
        seal = artifacts.audit_evidence_seal(
            root,
            expected_identity=_preflight_identity(stage),
            expected_stage=f"{stage.upper()}-PREFLIGHT",
        )
        checks = {
            "identity": result.get("identity") == _preflight_identity(stage),
            "status": result.get("status") == f"PASS_V2_A_C1S_{stage}_PREFLIGHT",
            "passed": result.get("passed") is True,
            "authorizes": result.get("authorizes") == f"{stage}_SINGLE_USE_LAUNCH_ONLY",
            "disposable_benchmark_steps": result.get("disposable_benchmark_optimizer_steps") == (100 if stage == "S1" else 0),
            "formal_optimizer_steps": result.get("formal_stage_optimizer_steps") == 0,
            "formal_training_not_started": result.get("formal_stage_training_started") is False,
            "optimization_contract": result.get("optimization_contract", {}).get("passed") is True,
            "s0_structural_instrument": (
                stage != "S1"
                or result.get("s0_single_slot_structural_control_replay", {}).get("passed") is True
            ),
            "seal": seal.get("passed") is True,
        }
        return {
            "root": root.as_posix(),
            "result_sha256": artifacts.sha256_file(root / "result.json"),
            "seal_sha256": artifacts.sha256_file(root / "evidence-seal.json"),
            "checks": checks,
            "seal_replay": seal,
            "passed": all(checks.values()),
        }
    except Exception as exc:  # noqa: BLE001 - audit is fail-closed
        return {"root": root.as_posix(), "passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _audit_predecessor(repo_root: Path, stage: str) -> dict[str, Any]:
    stage = stage.upper()
    if stage == "S1":
        result = artifacts.read_json(repo_root / S0_ROOT / "result.json")
        seal = artifacts.audit_external_pin(
            repo_root,
            contract.PINNED_PREDECESSOR_ROOTS["c1s_s0"],
            replay=True,
        )
        checks = {
            "identity": result.get("identity") == contract.S0_IDENTITY,
            "status": result.get("status") == "PASS_V2_A_C1S_S0_QUALIFICATION",
            "passed": result.get("passed") is True,
            "s1_status": result.get("s1_status") == "AUTHORIZED_NOT_RUN",
            "authorized_scope": result.get("authorizes") == contract.S0_AUTHORIZED_SCOPE,
            "s2_not_authorized": "S2" in result.get("never_authorizes", ()),
            "formal_not_authorized": "single-seed formal" in result.get("never_authorizes", ()),
            "seal": seal.get("passed") is True,
        }
        return {"root": (repo_root / S0_ROOT).as_posix(), "checks": checks, "seal": seal, "passed": all(checks.values())}
    if stage == "S2":
        return artifacts.audit_stage_predecessor(
            repo_root / contract.S1_ROOT,
            expected_identity=contract.S1_IDENTITY,
            expected_stage="S1",
            expected_status="PASS_V2_A_C1S_S1_QUALIFICATION",
            expected_authorizes="S2_DISCOVERY_ONLY",
        )
    if stage == "S3":
        return artifacts.audit_stage_predecessor(
            repo_root / contract.S2_ROOT,
            expected_identity=contract.S2_IDENTITY,
            expected_stage="S2",
            expected_status="PASS_V2_A_C1S_S2_QUALIFICATION",
            expected_authorizes="S3_SINGLE_SEED_FORMAL_ELIGIBILITY",
        )
    raise AssertionError(stage)


def _s0_single_slot_structural_control_replay(repo_root: Path) -> dict[str, Any]:
    """Replay S0's zero-update K1 instrument without treating it as learning evidence."""

    root = repo_root / S0_ROOT
    pin = artifacts.audit_external_pin(
        repo_root,
        contract.PINNED_PREDECESSOR_ROOTS["c1s_s0"],
        replay=True,
    )
    result = artifacts.read_json(root / "result.json")
    structure = artifacts.read_json(root / "s0-structure.json")
    manifest = artifacts.read_json(root / "contract-manifest.json")
    seal = artifacts.read_json(root / "evidence-seal.json")
    k1_null = structure.get("registered_controls", {}).get("k1_null", {})
    addressed_positive = structure.get("registered_controls", {}).get("addressed_positive", {})
    uniform_null = structure.get("registered_controls", {}).get("legacy_uniform_mean_null", {})
    k1_effective = k1_null.get("effective_slot_count", [])
    k1_effects = k1_null.get("effects", [])
    parameters_k1 = structure.get("parameters_k1", {})
    parameters_k8 = structure.get("parameters_k8", {})
    targeted = structure.get("targeted_transition", {})
    permutation = structure.get("permutation", {})
    thresholds = structure.get("thresholds", {})
    model_source = Path("src/yggdrasil_v2/v2_a/closure_c1s/model.py")
    sealed_model_relative = f"source_snapshot/{model_source.as_posix()}"
    current_model_sha = artifacts.sha256_file(repo_root / model_source)
    sealed_model_sha = seal.get("files", {}).get(sealed_model_relative)
    checks = {
        "sealed_pin_replay": pin.get("passed") is True,
        "qualified_zero_update_result": result.get("passed") is True
        and result.get("status") == "PASS_V2_A_C1S_S0_QUALIFICATION"
        and result.get("training_started") is False
        and result.get("optimizer_steps") == 0
        and result.get("model_writes") == 0
        and result.get("s1_status") == "AUTHORIZED_NOT_RUN",
        "sealed_model_source_unchanged": isinstance(sealed_model_sha, str)
        and current_model_sha == sealed_model_sha,
        "registered_configs_match_current": structure.get("config_k8") == dict(contract.MODEL_CONFIG)
        and structure.get("config_k1") == dict(contract.K1_CONFIG)
        and manifest.get("model") == dict(contract.MODEL_CONFIG)
        and manifest.get("matched_k1_model") == dict(contract.K1_CONFIG),
        "structure_passed": structure.get("passed") is True,
        "k1_integrity": structure.get("integrity_k1", {}).get("passed") is True,
        "parameter_parity": parameters_k1.get("trainable_parameters")
        == parameters_k8.get("trainable_parameters")
        and parameters_k1.get("deployment_parameters")
        == parameters_k8.get("deployment_parameters"),
        "registered_single_slot_null": k1_null.get("single_slot_null") is True,
        "one_effective_slot": bool(k1_effective)
        and all(abs(float(value) - 1.0) <= 1.0e-6 for value in k1_effective),
        "one_measured_effect_per_row": bool(k1_effects)
        and all(
            isinstance(row, list)
            and len(row) == 1
            and math.isfinite(float(row[0]))
            and float(row[0]) > 0.0
            for row in k1_effects
        ),
        "addressed_positive_calibrated": addressed_positive.get("single_slot_null") is False
        and min(addressed_positive.get("relevant_effect", [float("-inf")]))
        >= float(thresholds.get("positive_relevant_mean_replace_min", 1.0))
        and max(addressed_positive.get("irrelevant_max_effect", [float("inf")]))
        <= float(thresholds.get("positive_irrelevant_mean_replace_max", 1.0e-6)),
        "uniform_mean_null_calibrated": max(
            uniform_null.get("ownership_contrast", [float("inf")])
        )
        <= float(thresholds.get("uniform_null_ownership_contrast_max", 1.01)),
        "targeted_transition_calibrated": float(targeted.get("selected_slot_max_delta", 0.0))
        >= float(thresholds.get("selected_slot_min_delta", 1.0e-8))
        and float(targeted.get("unselected_slot_max_delta", float("inf")))
        <= float(thresholds.get("unselected_slot_max_delta", 1.0e-6))
        and float(targeted.get("inactive_transition_max_delta", float("inf")))
        <= float(thresholds.get("inactive_transition_max_delta", 1.0e-6)),
        "permutation_calibrated": float(permutation.get("logit_max_abs_delta", float("inf")))
        <= float(thresholds.get("permutation_logit_max_abs", 1.0e-5))
        and float(permutation.get("trajectory_max_abs_delta", float("inf")))
        <= float(thresholds.get("permutation_trajectory_max_abs", 1.0e-5)),
    }
    passed = all(checks.values())
    return {
        "status": "PASS_STRUCTURAL_INSTRUMENT_REPLAY" if passed else "FAIL_STRUCTURAL_INSTRUMENT_REPLAY",
        "passed": passed,
        "checks": checks,
        "s0_identity": result.get("identity"),
        "s0_result_sha256": artifacts.sha256_file(root / "result.json"),
        "s0_evidence_seal_sha256": artifacts.sha256_file(root / "evidence-seal.json"),
        "s0_structure_sha256": artifacts.sha256_file(root / "s0-structure.json"),
        "s0_contract_manifest_sha256": artifacts.sha256_file(root / "contract-manifest.json"),
        "current_model_source_sha256": current_model_sha,
        "sealed_model_source_sha256": sealed_model_sha,
        "instrument_role": "zero_update_structural_single_slot_null",
        "learned_k1_control": False,
        "evaluated_records": 0,
        "k1_null_implementation": "analytic_synthetic_mean_replace_not_model_forward",
        "s0_cuda_smoke_scope": "k8_only",
        "multi_address_applicable": False,
        "multi_address_qualified": None,
        "s1_k1_training_performed": False,
        "does_not_qualify_multi_address_learning": True,
        "pin_replay": pin,
    }


def _write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
        handle.write(encoded)
        handle.write("\n")


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("target bank must be a JSON object")
    return value


def _bank_root(repo_root: Path) -> Path:
    return repo_root / contract.S1_PREFLIGHT_ROOT


def _load_bank(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _bank_root(repo_root)
    bank = _read_gzip_json(root / TARGET_BANK_NAME)
    ledger = artifacts.read_json(root / SPLIT_LEDGER_NAME)
    audit = audit_target_bank(bank, expected_count=FULL_RECORD_COUNT)
    if audit.get("passed") is not True:
        raise ValueError(f"sealed successor target bank failed audit: {audit}")
    return bank, ledger


def _dynamic_supervision_coverage(
    bank_rows: Mapping[str, Any],
    selections: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Count observable dynamic targets without fitting or reading model state.

    State changes follow the evaluator's definition exactly: within one record
    and slot, compare each observed value with the preceding observed value of
    that feature.  Masked steps/cells and behavior-only records are excluded.
    Operation activity is registered on the full fixed operation sequence, so
    its inactive suffix remains observable rather than being erased by the
    state-step mask.
    """

    def cell() -> dict[str, int]:
        return {"positives": 0, "negatives": 0, "temporal_changes": 0}

    def sequence(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    reports: dict[str, Any] = {}
    for selection_name, example_ids in selections.items():
        families: dict[str, Any] = {
            family: {
                "mechanism_supported_records": 0,
                "dynamic_features": {
                    name: cell() for name in DYNAMIC_STATE_FEATURES.get(family, ())
                },
                "operation_active": cell(),
            }
            for family in FAMILIES
        }
        missing_ids: list[str] = []
        invalid_rows: list[str] = []
        ignored_unsupported = 0
        for raw_id in example_ids:
            example_id = str(raw_id)
            row = bank_rows.get(example_id)
            if not isinstance(row, Mapping):
                missing_ids.append(example_id)
                continue
            if row.get("mechanism_supported") is not True:
                ignored_unsupported += 1
                continue
            family = str(row.get("family", "")).upper()
            if family not in families:
                invalid_rows.append(example_id)
                continue
            family_report = families[family]
            family_report["mechanism_supported_records"] += 1
            names = row.get("state_feature_names")
            values = row.get("state_values")
            masks = row.get("state_feature_mask")
            step_mask = row.get("state_step_mask")
            if not all(sequence(value) for value in (names, values, masks, step_mask)):
                invalid_rows.append(example_id)
                continue
            if not (len(values) == len(masks) == len(step_mask)):
                invalid_rows.append(example_id)
                continue

            previous: dict[tuple[int, str], bool] = {}
            row_valid = True
            for step, observed_step in enumerate(step_mask):
                if not bool(observed_step):
                    continue
                step_values = values[step]
                step_masks = masks[step]
                if not sequence(step_values) or not sequence(step_masks) or len(step_values) != len(step_masks):
                    row_valid = False
                    break
                for slot, (slot_values, slot_masks) in enumerate(zip(step_values, step_masks)):
                    if not sequence(slot_values) or not sequence(slot_masks):
                        row_valid = False
                        break
                    if len(slot_values) != len(names) or len(slot_masks) != len(names):
                        row_valid = False
                        break
                    for feature, raw_name in enumerate(names):
                        name = str(raw_name)
                        feature_cell = family_report["dynamic_features"].get(name)
                        if feature_cell is None or not bool(slot_masks[feature]):
                            continue
                        target = bool(slot_values[feature])
                        feature_cell["positives" if target else "negatives"] += 1
                        key = (slot, name)
                        if key in previous and previous[key] != target:
                            feature_cell["temporal_changes"] += 1
                        previous[key] = target
                if not row_valid:
                    break
            if not row_valid:
                invalid_rows.append(example_id)
                continue

            operation_active = row.get("operation_active")
            if not sequence(operation_active):
                invalid_rows.append(example_id)
                continue
            operation_cell = family_report["operation_active"]
            previous_active: bool | None = None
            for raw_active in operation_active:
                active = bool(raw_active)
                operation_cell["positives" if active else "negatives"] += 1
                if previous_active is not None and previous_active != active:
                    operation_cell["temporal_changes"] += 1
                previous_active = active

        for family_report in families.values():
            cells = [
                *family_report["dynamic_features"].values(),
                family_report["operation_active"],
            ]
            for value in cells:
                value["passed"] = all(int(value[key]) > 0 for key in (
                    "positives", "negatives", "temporal_changes"
                ))
            family_report["passed"] = bool(family_report["mechanism_supported_records"])
            family_report["passed"] = family_report["passed"] and all(
                value["passed"] for value in cells
            )
        selection_passed = (
            not missing_ids
            and not invalid_rows
            and all(report["passed"] for report in families.values())
        )
        reports[str(selection_name)] = {
            "requested_records": len(example_ids),
            "ignored_unsupported_records": ignored_unsupported,
            "missing_ids": missing_ids,
            "invalid_rows": invalid_rows,
            "families": families,
            "passed": selection_passed,
        }
    return {
        "selections": reports,
        "passed": bool(reports) and all(report["passed"] for report in reports.values()),
    }


def _audit_split_ledger_semantics(
    records: Sequence[Mapping[str, Any]],
    bank: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute discovery selection and prove it contains only supported train rows."""

    source_by_id = {str(row.get("example_id")): row for row in records}
    bank_rows = bank.get("records", {})
    if not isinstance(bank_rows, Mapping):
        return {"passed": False, "reason": "target bank records are not a mapping"}
    expected_s1 = select_s1_ids(records)
    expected_s2 = select_s2_ids(records, s1_ids=expected_s1)
    expected = build_split_ledger(records, s1_ids=expected_s1, s2=expected_s2)
    actual_s1 = list(ledger.get("s1", {}).get("ids", []))
    actual_s2_train = list(ledger.get("s2", {}).get("train_ids", []))
    actual_s2_eval = list(ledger.get("s2", {}).get("eval_ids", []))

    def family_counts(ids: Sequence[str]) -> dict[str, int]:
        counts = {family: 0 for family in FAMILIES}
        for example_id in ids:
            row = source_by_id.get(str(example_id), {})
            family = str(row.get("family", "")).upper()
            if family in counts:
                counts[family] += 1
        return counts

    def query_slot_counts(ids: Sequence[str]) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {family: {} for family in FAMILIES}
        for example_id in ids:
            row = bank_rows.get(str(example_id), {})
            family = str(row.get("family", "")).upper()
            slot = str(row.get("query_owner"))
            if family in counts:
                counts[family][slot] = counts[family].get(slot, 0) + 1
        return counts

    selected = actual_s1 + actual_s2_train + actual_s2_eval
    unsupported_cells: dict[str, int] = {}
    for row in bank_rows.values():
        if row.get("mechanism_supported") is True:
            continue
        cell = f"{str(row.get('family', '')).upper()}/{row.get('split', '')}"
        unsupported_cells[cell] = unsupported_cells.get(cell, 0) + 1
    s1_counts = family_counts(actual_s1)
    s2_train_counts = family_counts(actual_s2_train)
    s2_eval_counts = family_counts(actual_s2_eval)
    s1_query_slots = query_slot_counts(actual_s1)
    s2_train_query_slots = query_slot_counts(actual_s2_train)
    s2_eval_query_slots = query_slot_counts(actual_s2_eval)
    formal_validation = [
        str(example_id)
        for example_id, row in bank_rows.items()
        if isinstance(row, Mapping)
        and row.get("split") == "validation"
        and row.get("mechanism_supported") is True
    ]
    supervision = _dynamic_supervision_coverage(
        bank_rows,
        {
            "s1": actual_s1,
            "s2_train": actual_s2_train,
            "s2_eval": actual_s2_eval,
            "formal_validation": formal_validation,
        },
    )
    checks = {
        "population_identity": set(source_by_id) == set(bank_rows),
        "ledger_exact_recomputation": dict(ledger) == expected,
        "selection_preimage_recomputed": ledger.get("selection_preimage_sha256")
        == expected.get("selection_preimage_sha256"),
        "s1_exact_and_balanced": len(actual_s1) == 32
        and s1_counts == {"ERE": 16, "CPS": 16},
        "s2_train_exact_and_balanced": len(actual_s2_train) == 6_144
        and s2_train_counts == {"ERE": 3_072, "CPS": 3_072},
        "s2_eval_exact_and_balanced": len(actual_s2_eval) == 1_024
        and s2_eval_counts == {"ERE": 512, "CPS": 512},
        "selection_disjoint": not (
            set(actual_s1) & (set(actual_s2_train) | set(actual_s2_eval))
            or set(actual_s2_train) & set(actual_s2_eval)
        ),
        "selected_rows_are_train": all(
            source_by_id.get(str(example_id), {}).get("split") == "train"
            for example_id in selected
        ),
        "selected_rows_mechanism_supported": all(
            bank_rows.get(str(example_id), {}).get("mechanism_supported") is True
            for example_id in selected
        ),
        "selected_queries_answer_independent": all(
            bank_rows.get(str(example_id), {}).get("query_answer_independent") is True
            for example_id in selected
        ),
        "s1_query_slots_nondegenerate": all(
            len(s1_query_slots[family]) >= 2 for family in FAMILIES
        ),
        "s2_train_query_slots_nondegenerate": all(
            len(s2_train_query_slots[family]) >= 4 for family in FAMILIES
        ),
        "s2_eval_query_slots_nondegenerate": all(
            len(s2_eval_query_slots[family]) >= 4 for family in FAMILIES
        ),
        "s1_dynamic_supervision_coverage": supervision["selections"]["s1"]["passed"],
        "s2_train_dynamic_supervision_coverage": supervision["selections"]["s2_train"]["passed"],
        "s2_eval_dynamic_supervision_coverage": supervision["selections"]["s2_eval"]["passed"],
        "formal_validation_dynamic_supervision_coverage": supervision["selections"]["formal_validation"]["passed"],
        "unsupported_rows_are_ood_only": all(
            str(row.get("split", "")).endswith("ood")
            for row in bank_rows.values()
            if row.get("mechanism_supported") is not True
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selection_preimage_sha256": ledger.get("selection_preimage_sha256"),
        "s1_family_counts": s1_counts,
        "s2_train_family_counts": s2_train_counts,
        "s2_eval_family_counts": s2_eval_counts,
        "s1_query_slot_counts": s1_query_slots,
        "s2_train_query_slot_counts": s2_train_query_slots,
        "s2_eval_query_slot_counts": s2_eval_query_slots,
        "dynamic_supervision_coverage": supervision,
        "unsupported_behavior_only_cells": dict(sorted(unsupported_cells.items())),
        "unsupported_behavior_only_count": sum(unsupported_cells.values()),
    }


def _audit_target_bank_integrity(repo_root: Path) -> dict[str, Any]:
    """Revalidate the sealed S1 bank and ledger against their recorded SHAs."""

    root = _bank_root(repo_root)
    try:
        result = artifacts.read_json(root / "result.json")
        target = result.get("target_bank")
        if not isinstance(target, Mapping):
            raise TypeError("S1 preflight result target_bank must be an object")
        expected_bank = target.get("sha256")
        expected_ledger = target.get("split_ledger_sha256")
        if not isinstance(expected_bank, str) or not isinstance(expected_ledger, str):
            raise TypeError("S1 preflight result lacks target-bank/ledger SHA pins")
        sealed_files = artifacts.audit_sealed_file_pins(
            root,
            expected_identity=_preflight_identity("S1"),
            expected_stage="S1-PREFLIGHT",
            expected_files={
                TARGET_BANK_NAME: expected_bank,
                SPLIT_LEDGER_NAME: expected_ledger,
            },
        )
        audit_file = artifacts.read_json(root / "target-bank-audit.json")
        bank, ledger = _load_bank(repo_root)
        semantic = audit_target_bank(bank, expected_count=FULL_RECORD_COUNT)
        store = load_offline_records(repo_root / DATA_ROOT)
        split_semantics = _audit_split_ledger_semantics(store.records, bank, ledger)
        checks = {
            "sealed_files": sealed_files.get("passed") is True,
            "producer_audit_bank_sha": audit_file.get("sha256") == expected_bank,
            "producer_audit_ledger_sha": audit_file.get("split_ledger_sha256") == expected_ledger,
            "semantic_bank_audit": semantic.get("passed") is True,
            "record_count": len(bank.get("records", {})) == FULL_RECORD_COUNT,
            "ledger_selection_preimage": isinstance(ledger.get("selection_preimage_sha256"), str),
            "result_selection_preimage": target.get("selection_preimage_sha256") == ledger.get("selection_preimage_sha256"),
            "split_ledger_semantics": split_semantics.get("passed") is True,
        }
        return {
            "root": root.as_posix(),
            "expected_target_bank_sha256": expected_bank,
            "actual_target_bank_sha256": artifacts.sha256_file(root / TARGET_BANK_NAME),
            "expected_split_ledger_sha256": expected_ledger,
            "actual_split_ledger_sha256": artifacts.sha256_file(root / SPLIT_LEDGER_NAME),
            "selection_preimage_sha256": ledger.get("selection_preimage_sha256"),
            "sealed_files": sealed_files,
            "semantic_audit": semantic,
            "split_ledger_semantics": split_semantics,
            "checks": checks,
            "passed": all(checks.values()),
        }
    except Exception as exc:  # noqa: BLE001 - provenance audit is fail-closed
        return {
            "root": root.as_posix(),
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _audit_fresh_s1_cache(
    cache_audit: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    dataset: CachedShardDataset,
    *,
    expected_records: int,
) -> dict[str, Any]:
    """Bind the S1 preflight PASS to the cache checks it actually ran."""

    checks = {
        "full_cache_audit": cache_audit.get("passed") is True,
        "cache_source_identity": source_audit.get("passed") is True,
        "manifest_example_count": len(dataset) == expected_records,
        "manifest_identity_present": isinstance(dataset.manifest.get("identity"), str),
        "manifest_schema_present": isinstance(dataset.manifest.get("schema_version"), str),
    }
    return {
        "status": "FULL_CACHE_AND_SOURCE_AUDITED",
        "manifest_identity": dataset.manifest.get("identity"),
        "manifest_schema_version": dataset.manifest.get("schema_version"),
        "manifest_example_count": len(dataset),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _audit_fresh_s1_target_materialization(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_bank_sha256: str,
    expected_ledger_sha256: str,
    expected_selection_preimage_sha256: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read back newly written S1 target files before any benchmark can use them."""

    bank_path = root / TARGET_BANK_NAME
    ledger_path = root / SPLIT_LEDGER_NAME
    bank = _read_gzip_json(bank_path)
    ledger = artifacts.read_json(ledger_path)
    semantic = audit_target_bank(bank, expected_count=FULL_RECORD_COUNT)
    split_semantics = _audit_split_ledger_semantics(records, bank, ledger)
    actual_bank_sha256 = artifacts.sha256_file(bank_path)
    actual_ledger_sha256 = artifacts.sha256_file(ledger_path)
    checks = {
        "target_bank_sha256": actual_bank_sha256 == expected_bank_sha256,
        "split_ledger_sha256": actual_ledger_sha256 == expected_ledger_sha256,
        "semantic_bank_audit": semantic.get("passed") is True,
        "record_count": len(bank.get("records", {})) == FULL_RECORD_COUNT,
        "selection_preimage": (
            isinstance(expected_selection_preimage_sha256, str)
            and ledger.get("selection_preimage_sha256") == expected_selection_preimage_sha256
        ),
        "split_ledger_semantics": split_semantics.get("passed") is True,
    }
    report = {
        "status": "BUILT_WRITTEN_AND_READBACK_AUDITED",
        "target_bank_sha256": actual_bank_sha256,
        "split_ledger_sha256": actual_ledger_sha256,
        "selection_preimage_sha256": ledger.get("selection_preimage_sha256"),
        "semantic_audit": semantic,
        "split_ledger_semantics": split_semantics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return report, bank, ledger


def _audit_cache_integrity(repo_root: Path) -> dict[str, Any]:
    """Replay the full immutable cache tree and bind every row to C0R source."""

    pin = contract.PINNED_PREDECESSOR_ROOTS["c1_cache"]
    cache_artifact_root = repo_root / str(pin["root"])
    try:
        seal_tree = artifacts.audit_external_pin(repo_root, pin, replay=True)
        result = artifacts.read_json(cache_artifact_root / "result.json")
        store = load_offline_records(repo_root / DATA_ROOT)
        dataset = CachedShardDataset(repo_root / CACHE_ROOT)
        source = validate_cache_source_identity(dataset, store)
        checks = {
            "sealed_cache_tree": seal_tree.get("passed") is True,
            "cache_result_passed": result.get("passed") is True,
            "cache_result_identity": result.get("identity") == dataset.manifest.get("identity"),
            "cache_result_source_stable": result.get("source_stable") is True,
            "manifest_example_count": len(dataset) == FULL_RECORD_COUNT == len(store),
            "cache_source_identity": source.get("passed") is True,
        }
        return {
            "root": cache_artifact_root.as_posix(),
            "hidden_root": (repo_root / CACHE_ROOT).as_posix(),
            "seal_tree": seal_tree,
            "source_identity": source,
            "manifest_identity": dataset.manifest.get("identity"),
            "manifest_schema_version": dataset.manifest.get("schema_version"),
            "checks": checks,
            "passed": all(checks.values()),
        }
    except Exception as exc:  # noqa: BLE001 - provenance audit is fail-closed
        return {
            "root": cache_artifact_root.as_posix(),
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _run_tests(repo_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/v2_a_closure_c1s_successor"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20_000:],
        "stderr": completed.stderr[-20_000:],
        "seconds": time.perf_counter() - started,
    }


def _fresh_model(config: Mapping[str, Any], *, seed: int) -> tuple[C1SModel, SuccessorDiagnosticHeads]:
    set_determinism(seed)
    parsed = C1SConfig(**dict(config))
    model = C1SModel(parsed)
    heads = SuccessorDiagnosticHeads(parsed.source_width, parsed.address_width, parsed.auxiliary_width)
    return model, heads


def _registered_loss_contract() -> tuple[LossWeights, CausalMargins, dict[str, Any]]:
    configured_weights = dict(contract.OPTIMIZATION_CONFIG["loss_weights"])
    weights = LossWeights(**configured_weights)
    margin_value = float(contract.OPTIMIZATION_CONFIG["causal_margin"])
    margins = CausalMargins(
        **{field.name: margin_value for field in fields(CausalMargins)}
    )
    checks = {
        "loss_weights_exact": asdict(weights) == configured_weights,
        "all_causal_margins_exact": all(
            float(value) == margin_value for value in asdict(margins).values()
        ),
    }
    return weights, margins, {
        "passed": all(checks.values()),
        "checks": checks,
        "loss_weights": asdict(weights),
        "causal_margins": asdict(margins),
    }


def _registered_train_spec(stage: str) -> TrainSpec:
    stage = stage.upper()
    if stage == "S1":
        epochs, maximum, causal_interval = 1_000, 4_000, int(
            contract.OPTIMIZATION_CONFIG["s1_causal_interval"]
        )
    elif stage == "S2":
        epochs, maximum, causal_interval = 6, 4_608, int(
            contract.OPTIMIZATION_CONFIG["s2_s3_causal_interval"]
        )
    elif stage == "S3":
        epochs, maximum, causal_interval = 6, 6_144, int(
            contract.OPTIMIZATION_CONFIG["s2_s3_causal_interval"]
        )
    else:
        raise ValueError(f"unknown training stage: {stage}")
    config = contract.OPTIMIZATION_CONFIG
    return TrainSpec(
        batch_size=int(config["batch_size"]),
        family_batch_size=int(config["family_batch_size"]),
        epochs=epochs,
        maximum_updates=maximum,
        boundary_lr=float(config["boundary_lr"]),
        transition_lr=float(config["transition_lr"]),
        head_lr=float(config["head_lr"]),
        weight_decay=float(config["weight_decay"]),
        gradient_clip=float(config["gradient_clip"]),
        warmup_updates=int(config["warmup_updates"]),
        log_interval=100,
        causal_interval=causal_interval,
        cpu_threads=int(config["cpu_intraop_threads"]),
        pin_memory=True,
    )


def _preflight_benchmark(
    store: Any,
    runtime: C1SSuccessorRuntime,
    ledger: Mapping[str, Any],
    *,
    device: str,
    on_optimizer_step: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    s1 = set(ledger["s1"]["ids"])
    s2_eval = set(ledger["s2"]["eval_ids"])
    candidates = [
        row for row in store.records
        if row.get("split") == "train" and row["example_id"] not in s1 | s2_eval
    ]
    selected: list[Mapping[str, Any]] = []
    for family in FAMILIES:
        selected.extend([row for row in candidates if row.get("family") == family][:16])
    model_seed = int(contract.S1_CONFIG["preflight_benchmark_model_seed"])
    order_seed = int(contract.S1_CONFIG["preflight_benchmark_order_seed"])
    schedule = build_overfit_schedule(selected, seed=order_seed, maximum_updates=100)
    model, heads = _fresh_model(contract.MODEL_CONFIG, seed=model_seed)
    spec = TrainSpec(
        epochs=25,
        maximum_updates=100,
        warmup_updates=10,
        log_interval=100,
        # S1 computes every registered causal intervention on every update;
        # the disposable benchmark must measure that exact hot path.
        causal_interval=1,
        cpu_threads=2,
        pin_memory=True,
    )
    loss_weights, causal_margins, loss_contract = _registered_loss_contract()
    if loss_contract.get("passed") is not True:
        raise RuntimeError("registered optimization/loss contract is inconsistent")
    report = train_fixed_endpoint(
        model,
        heads,
        runtime,
        schedule,
        identity="DISPOSABLE_C1S_SUCCESSOR_100_STEP_BENCHMARK",
        order_seed=order_seed,
        model_seed=model_seed,
        spec=spec,
        device=device,
        loss_weights=loss_weights,
        margins=causal_margins,
        on_optimizer_step=on_optimizer_step,
    )
    report["disposable"] = True
    report["weights_reused"] = False
    report["model_writes"] = 0
    report["selection_or_threshold_tuning_authorized"] = False
    del model, heads
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_preflight(repo_root: Path, stage: str, *, device: str = "cuda") -> dict[str, Any]:
    """Consume one fixed preflight identity without launching its stage."""

    repo_root = Path(repo_root).resolve()
    identity, stage_root_rel, stage_lease_rel, root_rel, lease_rel, formal = _stage_config(stage)
    stage = stage.upper()
    root = _absolute(repo_root, root_rel)
    lease = _absolute(repo_root, lease_rel)
    stage_root = _absolute(repo_root, stage_root_rel)
    stage_lease = _absolute(repo_root, stage_lease_rel)
    if root.exists() or lease.exists():
        return _refusal(stage, root, lease, "fixed preflight root or lease already exists")
    if stage_root.exists() or stage_lease.exists():
        return _refusal(stage, root, lease, "stage root or lease already exists before preflight")

    try:
        source_before = artifacts.source_hashes(repo_root)
        source_id = artifacts.source_identity(source_before)
        predecessor = _audit_predecessor(repo_root, stage)
        static_pins = _audit_static_pins(repo_root, replay=stage == "S1")
        device_report = _device_audit(device)
    except Exception as exc:  # pre-lease refusal
        return _refusal(stage, root, lease, f"prerequisite {type(exc).__name__}: {exc}")
    if predecessor.get("passed") is not True or static_pins.get("passed") is not True or device_report.get("passed") is not True:
        return _refusal(stage, root, lease, f"predecessor/static/device audit failed: predecessor={predecessor.get('passed')} static={static_pins.get('passed')} device={device_report.get('passed')}")

    preflight_identity = _preflight_identity(stage)
    artifacts.claim_single_use(
        identity=preflight_identity,
        stage=f"{stage}-PREFLIGHT",
        output_root=root,
        lease_path=lease,
        formal=False,
    )
    optimizer_steps = 0
    target_integrity: dict[str, Any] = {"status": "NOT_REQUIRED", "passed": True}
    cache_integrity: dict[str, Any] = {"status": "NOT_REQUIRED", "passed": True}
    s0_structural_replay: dict[str, Any] = {"status": "NOT_REQUIRED", "passed": True}
    try:
        artifacts.snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.stage_manifest(stage, preflight=True))
        tests = _run_tests(repo_root)
        artifacts.write_json(root / "tests.json", tests)
        if tests.get("passed") is not True:
            raise RuntimeError("successor test suite failed")
        _loss_weights, _causal_margins, loss_contract = _registered_loss_contract()
        loss_contract["train_spec"] = asdict(_registered_train_spec(stage))
        artifacts.write_json(root / "optimization-contract.json", loss_contract)
        if loss_contract.get("passed") is not True:
            raise RuntimeError("registered optimization/loss contract is inconsistent")

        bank_report: dict[str, Any] = {"status": "REUSED_FROM_S1_PREFLIGHT"}
        benchmark: dict[str, Any] = {"status": "NOT_REQUIRED", "optimizer_steps": 0, "model_writes": 0}
        if stage == "S1":
            s0_structural_replay = _s0_single_slot_structural_control_replay(repo_root)
            artifacts.write_json(
                root / "s0-single-slot-structural-control-replay.json",
                s0_structural_replay,
            )
            if s0_structural_replay.get("passed") is not True:
                raise RuntimeError("sealed S0 single-slot structural instrument replay failed")
            store = load_offline_records(repo_root / DATA_ROOT)
            if len(store) != FULL_RECORD_COUNT:
                raise ValueError(f"full C0R record count {len(store)} != {FULL_RECORD_COUNT}")
            cache_path = repo_root / CACHE_ROOT
            cache_audit = audit_cache(cache_path, records=store.records)
            artifacts.write_json(root / "cache-audit.json", cache_audit)
            if cache_audit.get("passed") is not True:
                raise RuntimeError(f"immutable cache audit failed: {cache_audit.get('failures')}")
            dataset = CachedShardDataset(cache_path)
            source_audit = validate_cache_source_identity(dataset, store)
            artifacts.write_json(root / "cache-source-identity.json", source_audit)
            if source_audit.get("passed") is not True:
                raise RuntimeError(f"cache/source identity failed: {source_audit.get('failures')}")
            cache_integrity = _audit_fresh_s1_cache(
                cache_audit,
                source_audit,
                dataset,
                expected_records=FULL_RECORD_COUNT,
            )
            artifacts.write_json(root / "cache-integrity.json", cache_integrity)
            if cache_integrity.get("passed") is not True:
                raise RuntimeError("fresh S1 cache integrity report failed")
            tokenizer = load_qwen35_tokenizer()
            metadata = {
                example_id: {
                    "token_count": int(location[3]),
                    "silent_truncation": False,
                    "identity": dataset.manifest.get("identity"),
                    "schema_version": dataset.manifest.get("schema_version"),
                }
                for example_id, location in dataset.locations.items()
            }
            materialize_started = time.perf_counter()
            bank = materialize_target_bank(
                store.records,
                tokenizer,
                cache_metadata=metadata,
                strict=True,
            )
            bank_audit = audit_target_bank(bank, expected_count=FULL_RECORD_COUNT)
            if bank_audit.get("passed") is not True:
                raise RuntimeError(f"target bank audit failed: {bank_audit}")
            s1_ids = select_s1_ids(store.records)
            s2_ids = select_s2_ids(store.records, s1_ids=s1_ids)
            ledger = build_split_ledger(store.records, s1_ids=s1_ids, s2=s2_ids)
            split_semantics = _audit_split_ledger_semantics(store.records, bank, ledger)
            artifacts.write_json(root / "split-ledger-audit.json", split_semantics)
            if split_semantics.get("passed") is not True:
                raise RuntimeError(f"split-ledger semantic audit failed: {split_semantics}")
            _write_gzip_json(root / TARGET_BANK_NAME, bank)
            artifacts.write_json(root / SPLIT_LEDGER_NAME, ledger)
            bank_report = {
                **bank_audit,
                "sha256": artifacts.sha256_file(root / TARGET_BANK_NAME),
                "compressed_bytes": (root / TARGET_BANK_NAME).stat().st_size,
                "materialize_seconds": time.perf_counter() - materialize_started,
                "split_ledger_sha256": artifacts.sha256_file(root / SPLIT_LEDGER_NAME),
                "selection_preimage_sha256": ledger.get("selection_preimage_sha256"),
                "split_ledger_semantics": split_semantics,
            }
            artifacts.write_json(root / "target-bank-audit.json", bank_report)
            target_integrity, bank, ledger = _audit_fresh_s1_target_materialization(
                root,
                store.records,
                expected_bank_sha256=str(bank_report["sha256"]),
                expected_ledger_sha256=str(bank_report["split_ledger_sha256"]),
                expected_selection_preimage_sha256=bank_report["selection_preimage_sha256"],
            )
            artifacts.write_json(root / "target-bank-integrity.json", target_integrity)
            if target_integrity.get("passed") is not True:
                raise RuntimeError("fresh S1 target-bank/ledger readback integrity failed")
            runtime = C1SSuccessorRuntime(dataset, bank)
            def record_disposable_step(update: int) -> None:
                nonlocal optimizer_steps
                optimizer_steps = int(update)

            benchmark = _preflight_benchmark(
                store,
                runtime,
                ledger,
                device=device,
                on_optimizer_step=record_disposable_step,
            )
            optimizer_steps = _reconcile_optimizer_steps(
                observed=optimizer_steps,
                base=0,
                reported=int(benchmark.get("optimizer_steps", 0)),
            )
            artifacts.write_json(root / "benchmark-100-step.json", benchmark)
            if benchmark.get("passed") is not True:
                raise RuntimeError("100-step disposable benchmark failed")
            free_memory = int(device_report.get("free_memory_bytes", 0))
            if int(benchmark.get("peak_memory_bytes", free_memory + 1)) > int(0.90 * free_memory):
                raise RuntimeError("measured batch-8 peak exceeds 90% of launch-time free VRAM")
        else:
            s1_preflight = _audit_preflight(repo_root, "S1")
            artifacts.write_json(root / "s1-preflight-pin.json", s1_preflight)
            if s1_preflight.get("passed") is not True:
                raise RuntimeError("sealed S1 preflight/target bank is unavailable")
            target_integrity = _audit_target_bank_integrity(repo_root)
            artifacts.write_json(root / "target-bank-integrity.json", target_integrity)
            if target_integrity.get("passed") is not True:
                raise RuntimeError("sealed S1 target-bank/ledger integrity failed")
            cache_integrity = _audit_cache_integrity(repo_root)
            artifacts.write_json(root / "cache-integrity.json", cache_integrity)
            if cache_integrity.get("passed") is not True:
                raise RuntimeError("immutable cache seal/tree or source identity failed")
            bank, ledger = _load_bank(repo_root)
            bank_report = {
                "status": "REUSED_FROM_S1_PREFLIGHT",
                "passed": target_integrity.get("passed") is True,
                "examples": len(bank["records"]),
                "target_bank_sha256": artifacts.sha256_file(_bank_root(repo_root) / TARGET_BANK_NAME),
                "split_ledger_sha256": artifacts.sha256_file(_bank_root(repo_root) / SPLIT_LEDGER_NAME),
                "selection_preimage_sha256": ledger.get("selection_preimage_sha256"),
            }
            artifacts.write_json(root / "target-bank-pin.json", bank_report)

        source_after = artifacts.source_hashes(repo_root)
        source_stable = source_before == source_after
        expected_disposable_steps = 100 if stage == "S1" else 0
        passed = bool(
            source_stable
            and optimizer_steps == expected_disposable_steps
            and benchmark.get("model_writes", 0) == 0
            and target_integrity.get("passed") is True
            and cache_integrity.get("passed") is True
            and s0_structural_replay.get("passed") is True
        )
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": preflight_identity,
            "stage": f"{stage}-PREFLIGHT",
            "formal": False,
            "status": f"PASS_V2_A_C1S_{stage}_PREFLIGHT" if passed else f"FAIL_V2_A_C1S_{stage}_PREFLIGHT",
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "source_identity": source_id,
            "source_stable": source_stable,
            "predecessor": predecessor,
            "static_pins": static_pins,
            "device": device_report,
            "tests": {key: tests[key] for key in ("passed", "returncode", "seconds")},
            "target_bank": bank_report,
            "target_bank_integrity": target_integrity,
            "cache_integrity": cache_integrity,
            "s0_single_slot_structural_control_replay": s0_structural_replay,
            "benchmark": {key: benchmark.get(key) for key in ("status", "completed_updates", "optimizer_steps", "model_writes", "wall_seconds", "step_median_seconds", "step_p95_seconds", "peak_memory_bytes", "data_wait_fraction")},
            "optimization_contract": loss_contract,
            "optimizer_steps": optimizer_steps,
            "optimizer_step_scope": "disposable_preflight_benchmark_only" if optimizer_steps else "none",
            "disposable_benchmark_optimizer_steps": optimizer_steps,
            "formal_stage_optimizer_steps": 0,
            "model_writes": 0,
            "stage_training_started": False,
            "formal_stage_training_started": False,
            "authorizes": f"{stage}_SINGLE_USE_LAUNCH_ONLY" if passed else "nothing",
            "c2_authorized": False,
            "v2a_passed": False,
        }
        artifacts.write_json(root / "result.json", result)
        seal_sha, replay = _seal_and_replay(
            root,
            identity=preflight_identity,
            stage=f"{stage}-PREFLIGHT",
        )
        completed = _completion_envelope(
            result,
            output_root=root,
            result_sha256=artifacts.sha256_file(root / "result.json"),
            evidence_seal_sha256=seal_sha,
            seal_replay=replay,
            incomplete_status=f"INCOMPLETE_V2_A_C1S_{stage}_PREFLIGHT_EVIDENCE_SEAL",
        )
        completed["lease_path"] = lease.as_posix()
        return completed
    except Exception as exc:  # noqa: BLE001 - consumed identity must seal failure
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.preflight-result.v1",
            "identity": preflight_identity,
            "stage": f"{stage}-PREFLIGHT",
            "formal": False,
            "status": f"FAIL_V2_A_C1S_{stage}_PREFLIGHT",
            "passed": False,
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "target_bank_integrity": target_integrity,
            "cache_integrity": cache_integrity,
            "s0_single_slot_structural_control_replay": s0_structural_replay,
            "optimizer_steps": optimizer_steps,
            "optimizer_step_scope": "disposable_preflight_benchmark_only" if optimizer_steps else "none",
            "disposable_benchmark_optimizer_steps": optimizer_steps,
            "formal_stage_optimizer_steps": 0,
            "model_writes": 0,
            "stage_training_started": False,
            "formal_stage_training_started": False,
            "authorizes": "nothing",
            "c2_authorized": False,
            "v2a_passed": False,
        }
        artifacts.write_json(root / "result.json", result)
        seal_sha, replay = _seal_and_replay(
            root,
            identity=preflight_identity,
            stage=f"{stage}-PREFLIGHT",
        )
        completed = _completion_envelope(
            result,
            output_root=root,
            result_sha256=artifacts.sha256_file(root / "result.json"),
            evidence_seal_sha256=seal_sha,
            seal_replay=replay,
            incomplete_status=f"INCOMPLETE_V2_A_C1S_{stage}_PREFLIGHT_EVIDENCE_SEAL",
        )
        completed["lease_path"] = lease.as_posix()
        return completed


def _records_for_ids(store: Any, ids: Sequence[str]) -> list[dict[str, Any]]:
    return [store[str(example_id)] for example_id in ids]


def _tensor_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _binary_metric(values: Sequence[bool]) -> dict[str, Any]:
    n = len(values)
    successes = sum(bool(value) for value in values)
    point = successes / n if n else 0.0
    if not n:
        lower = upper = 0.0
    else:
        z = 1.959963984540054
        denominator = 1.0 + z * z / n
        centre = (point + z * z / (2.0 * n)) / denominator
        radius = z * math.sqrt(point * (1.0 - point) / n + z * z / (4.0 * n * n)) / denominator
        lower, upper = max(0.0, centre - radius), min(1.0, centre + radius)
    return {"successes": successes, "n": n, "point": point, "wilson_lower": lower, "wilson_upper": upper}


def _answer_margin_tensor(logits: Tensor, answers: Tensor) -> Tensor:
    values = logits.float()
    answers = answers.long()
    correct = values.gather(1, answers.view(-1, 1)).squeeze(1)
    wrong = values.masked_fill(
        torch.nn.functional.one_hot(answers, num_classes=values.shape[-1]).bool(),
        float("-inf"),
    )
    return correct - torch.logsumexp(wrong, dim=-1)


def _initial_slot_effects(
    model: C1SModel,
    base: Mapping[str, Tensor],
    boundary: Any,
    content_mask: Tensor,
    answers: Tensor,
) -> Tensor:
    """Measure gauge-invariant content contributions before recurrence."""

    payloads = boundary.payloads
    slots = payloads.shape[1]
    base_margin = _answer_margin_tensor(base["logits"], answers)
    if slots == 1:
        changed = replace(boundary, payloads=torch.zeros_like(payloads))
        logits = model.forward_from_boundary(changed)["logits"]
        return (
            base_margin - _answer_margin_tensor(logits, answers)
        ).clamp_min(0.0).unsqueeze(1)
    weights = content_mask.to(payloads.dtype).unsqueeze(-1)
    total = (payloads * weights).sum(dim=1)
    counts = content_mask.sum(dim=1).clamp_min(1).to(payloads.dtype).unsqueeze(-1)
    rows: list[Tensor] = []
    for slot in range(slots):
        is_content = content_mask[:, slot]
        other_count = (counts - is_content.to(payloads.dtype).unsqueeze(-1)).clamp_min(1.0)
        replacement = (total - payloads[:, slot] * is_content.to(payloads.dtype).unsqueeze(-1)) / other_count
        changed = payloads.clone()
        changed[:, slot] = replacement
        changed_boundary = replace(boundary, payloads=changed)
        logits = model.forward_from_boundary(changed_boundary)["logits"]
        rows.append(
            (base_margin - _answer_margin_tensor(logits, answers)).clamp_min(0.0)
        )
    result = torch.stack(rows, dim=1)
    return result * content_mask.to(result.dtype)


def _duplicate_payload_control(
    model: C1SModel,
    base: Mapping[str, Tensor],
    boundary: Any,
    query_owner: Tensor,
    alternate_owner: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Duplicate content, then prove two addresses still select distinct owners."""

    valid = alternate_owner >= 0
    if not bool(valid.any()):
        empty = torch.zeros_like(valid)
        return empty, torch.zeros_like(valid, dtype=torch.float32), empty
    payloads = base["final_payloads"].clone()
    query_payload = payloads.gather(
        1, query_owner.view(-1, 1, 1).expand(-1, 1, payloads.shape[-1])
    ).squeeze(1)
    payloads = payloads.scatter(
        1,
        alternate_owner.clamp_min(0).view(-1, 1, 1).expand(-1, 1, payloads.shape[-1]),
        query_payload.unsqueeze(1),
    )
    alternate_key = boundary.addresses.gather(
        1, alternate_owner.clamp_min(0).view(-1, 1, 1).expand(-1, 1, boundary.addresses.shape[-1])
    ).squeeze(1)
    # The first read uses the learned public-input query.  Only the counterfactual
    # alternate read uses an exact address key as a structural instrument.
    first_logits, first_weights = model.logits_from_state(
        payloads,
        boundary.addresses,
        boundary.query_key,
    )
    second_logits, second_weights = model.logits_from_state(payloads, boundary.addresses, alternate_key)
    owner_ok = (
        (first_weights.argmax(dim=-1) == query_owner)
        & (second_weights.argmax(dim=-1) == alternate_owner)
        & valid
    )
    differences = (first_logits.float() - second_logits.float()).abs().amax(dim=-1)
    return owner_ok, differences, valid


def _metadata_from_batch(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    count = len(batch["example_ids"])
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "example_id": batch["example_ids"][index],
                "family": batch["families"][index],
                "split": batch["splits"][index],
                "pair_id": batch["pair_ids"][index],
                "pair_role": batch["pair_roles"][index],
                "answer_index": int(batch["answers"][index]),
                "query_owner": int(batch["query_owner"][index]),
                "owner_slot": int(batch["query_owner"][index]),
                "alternate_owner": int(batch["alternate_owner"][index]),
                "alternate_label": int(batch["alternate_label"][index]),
                "mechanism_supported": bool(batch["mechanism_supported"][index]),
                "query_answer_independent": bool(batch["query_answer_independent"][index]),
                "content_mask": batch["content_mask"][index].tolist(),
                "object_kinds": list(batch["object_kinds"][index]),
                "source_owner": batch["source_owner"][index].tolist(),
                "target_owner": batch["target_owner"][index].tolist(),
                "operation_active": batch["operation_active"][index].tolist(),
                "state_values": batch["state_values"][index].tolist(),
                "state_feature_mask": batch["state_feature_mask"][index].tolist(),
                "state_step_mask": batch["state_step_mask"][index].tolist(),
                "state_feature_names": list(batch["state_feature_names"][index]),
            }
        )
    return rows


def _merge_prediction_batches(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def merge(values: Sequence[Any]) -> Any:
        first = values[0]
        if isinstance(first, Tensor):
            return torch.cat([value.detach().cpu() for value in values], dim=0)
        if isinstance(first, Mapping):
            keys = set(first)
            if any(set(value) != keys for value in values):
                raise ValueError("prediction batch keys differ")
            return {key: merge([value[key] for value in values]) for key in sorted(keys)}
        if isinstance(first, list):
            return sum((list(value) for value in values), [])
        raise TypeError(f"cannot merge prediction value {type(first).__name__}")

    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("prediction mappings differ")
    return {key: merge([row[key] for row in rows]) for key in sorted(keys)}


def _slice_prediction(value: Any, indices: Tensor) -> Any:
    if isinstance(value, Tensor):
        return value.index_select(0, indices)
    if isinstance(value, Mapping):
        return {key: _slice_prediction(item, indices) for key, item in value.items()}
    if isinstance(value, list):
        return [value[int(index)] for index in indices.tolist()]
    return value


def _collect_mechanism_predictions(
    model: C1SModel,
    heads: SuccessorDiagnosticHeads,
    runtime: C1SSuccessorRuntime,
    ids: Sequence[str],
    *,
    device: str,
    batch_size: int = 8,
    include_strip: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    active_device = torch.device(device)
    model.to(active_device).eval()
    heads.to(active_device).eval()
    deployment = strip_training_auxiliary(model).to(active_device).eval() if include_strip else None
    predictions: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    duplicate_owner: list[bool] = []
    duplicate_logit_differences: list[float] = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            chunk = list(ids[start : start + batch_size])
            cpu_batch = runtime.get_batch(chunk, slots=model.config.slots, pin_memory=True)
            metadata.extend(_metadata_from_batch(cpu_batch))
            batch = _tensor_to_device(cpu_batch, active_device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                boundary = model.boundary(batch["source_hidden"], batch["source_mask"])
                base = model.forward_from_boundary(boundary, return_trajectory=True, return_auxiliary=True)
                auxiliary = base.get("auxiliary")
                if not isinstance(auxiliary, Mapping):
                    raise RuntimeError("mechanism evaluation requires training auxiliary")
                state_logits = heads.state_decoder(auxiliary["state_features"])
                changed = intervention_outputs(model, boundary, base, batch)
                reverse = torch.arange(model.config.slots - 1, -1, -1, device=active_device)
                permuted = model.forward_from_boundary(boundary.permuted(reverse))["logits"]
                stripped = deployment(batch["source_hidden"], batch["source_mask"])["logits"] if deployment is not None else base["logits"]
                effects = _initial_slot_effects(
                    model,
                    base,
                    boundary,
                    batch["content_mask"].bool(),
                    batch["answers"].long(),
                )
                owner_rows, logit_rows, duplicate_valid = _duplicate_payload_control(
                    model,
                    base,
                    boundary,
                    batch["query_owner"].long(),
                    batch["alternate_owner"].long(),
                )
            duplicate_owner.extend(owner_rows[duplicate_valid].tolist())
            duplicate_logit_differences.extend(logit_rows[duplicate_valid].tolist())
            predictions.append(
                {
                    "logits": base["logits"].float(),
                    "query_weights": base["query_weights"].float(),
                    "source_weights": boundary.source_weights.float(),
                    "target_weights": boundary.target_weights.float(),
                    "operation_active": boundary.operation_active.float(),
                    "state_logits": state_logits.float(),
                    "initial_slot_margin_effects": effects.float(),
                    "content_mask": batch["content_mask"].float(),
                    "duplicate_owner_ok": owner_rows.float(),
                    "duplicate_logit_diff": logit_rows.float(),
                    "duplicate_valid": duplicate_valid.float(),
                    "interventions": {
                        "no_core_logits": changed["no_core_logits"].float(),
                        "wrong_start_logits": changed["wrong_start_logits"].float(),
                        "payload_zero_logits": changed["payload_zero_logits"].float(),
                        "payload_shuffle_logits": changed["payload_shuffle_logits"].float(),
                        "operation_zero_logits": changed["operation_zero_logits"].float(),
                        "operation_shuffle_logits": changed["operation_shuffle_logits"].float(),
                        "relevant_replace_logits": changed["relevant_replace_logits"].float(),
                        "irrelevant_replace_logits": changed["irrelevant_replace_logits"].float(),
                        "target_shuffle_logits": changed["target_shuffle_logits"].float(),
                    },
                    "permuted_logits": permuted.float(),
                    "stripped_logits": stripped.float(),
                }
            )
    merged = _merge_prediction_batches(predictions)
    controls = {
        "duplicate_payload": {
            "eligible_n": len(duplicate_owner),
            "owner_follow": _binary_metric(duplicate_owner),
            "logit_max_abs": max(duplicate_logit_differences, default=float("inf")),
            "passed": bool(duplicate_owner)
            and all(duplicate_owner)
            and max(duplicate_logit_differences, default=float("inf")) <= 1.0e-6,
        },
    }
    del deployment
    return merged, metadata, controls


def _mechanism_pass(report: Mapping[str, Any], *, s1: bool = False) -> bool:
    threshold = 0.95 if s1 else float(contract.MECHANISM_THRESHOLDS["owner_accuracy"])
    state_threshold = 0.95 if s1 else float(contract.MECHANISM_THRESHOLDS["state_accuracy"])
    owner = report.get("ownership", {})
    interventions = report.get("interventions", {})
    contributors = report.get("functional_k", {}).get("causal_contributors", {})
    duplicate = report.get("duplicate_payload_control", {})
    target_contract = report.get("target_contract", {})
    required_drop = 0.50 if s1 else float(contract.MECHANISM_THRESHOLDS["intervention_drop_bootstrap_lower"])
    statistic = "point" if s1 else "lower"
    drop_ok = all(
        interventions.get(name, {}).get("answer_margin_drop", {}).get(statistic, -1.0)
        >= required_drop
        for name in (
            "no_core",
            "wrong_start",
            "payload_zero",
            "payload_shuffle",
            "operation_zero",
            "operation_shuffle",
            "relevant_replace",
            "target_shuffle",
        )
    )
    contributor_floor = 0.90 if s1 else 0.80
    return bool(
        target_contract.get("mechanism_supported") is True
        and target_contract.get("query_answer_independent") is True
        and owner.get("query", {}).get("point", 0.0) >= threshold
        and owner.get("source", {}).get("point", 0.0) >= threshold
        and owner.get("target", {}).get("point", 0.0) >= threshold
        and owner.get("state_masked", {}).get("point", 0.0) >= state_threshold
        and operation_active_qualified(report, floor=state_threshold)
        and dynamic_state_qualified(report, floor=state_threshold)
        and duplicate.get("passed") is True
        and contributors.get("at_least_two", {}).get("point", 0.0) >= contributor_floor
        and contributors.get("mean_effective_slots", 0.0)
        >= float(contract.MECHANISM_THRESHOLDS["functional_min_effective_slots"])
        and interventions.get("no_core", {}).get("answer", {}).get("overall", {}).get("point", 1.0)
        <= float(contract.MECHANISM_THRESHOLDS["no_core_accuracy_max"])
        and drop_ok
        and abs(interventions.get("irrelevant_replace", {}).get("answer_margin_drop", {}).get("point", 999.0))
        <= float(contract.MECHANISM_THRESHOLDS["irrelevant_drop_abs_max"])
        and report.get("invariance", {}).get("slot_permutation", {}).get("passed") is True
        and report.get("invariance", {}).get("strip", {}).get("passed") is True
    )


def _duplicate_control_report(prediction: Mapping[str, Any]) -> dict[str, Any]:
    valid = prediction["duplicate_valid"].bool()
    owner = prediction["duplicate_owner_ok"].bool()
    differences = prediction["duplicate_logit_diff"].float()
    selected_owner = owner[valid].tolist()
    maximum = float(differences[valid].max()) if bool(valid.any()) else float("inf")
    return {
        "eligible_n": int(valid.sum()),
        "owner_follow": _binary_metric(selected_owner),
        "logit_max_abs": maximum,
        "passed": bool(valid.any()) and all(selected_owner) and maximum <= 1.0e-6,
    }


def _evaluate_mechanism(
    model: C1SModel,
    heads: SuccessorDiagnosticHeads,
    runtime: C1SSuccessorRuntime,
    ids: Sequence[str],
    *,
    device: str,
    s1: bool = False,
    bootstrap_seed: int = contract.S2_BOOTSTRAP_SEED,
    bootstrap_replicates: int = 2_000,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    prediction, metadata, controls = _collect_mechanism_predictions(
        model, heads, runtime, ids, device=device, include_strip=True
    )
    report = evaluate_predictions(
        prediction,
        metadata,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    report["duplicate_payload_control"] = _duplicate_control_report(prediction)
    family_reports: dict[str, Any] = {}
    for family in FAMILIES:
        index = torch.tensor([i for i, row in enumerate(metadata) if row["family"] == family], dtype=torch.long)
        child_prediction = _slice_prediction(prediction, index)
        child_metadata = [metadata[i] for i in index.tolist()]
        child = evaluate_predictions(
            child_prediction,
            child_metadata,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )
        child["duplicate_payload_control"] = _duplicate_control_report(child_prediction)
        child["mechanism_pass"] = _mechanism_pass(child, s1=s1)
        family_reports[family] = child
    report["family_reports"] = family_reports
    report["mechanism_pass"] = _mechanism_pass(report, s1=s1) and all(
        child["mechanism_pass"] for child in family_reports.values()
    )
    report["invariance"]["slot_permutation"]["n"] = len(metadata)
    report["invariance"]["strip"]["n"] = len(metadata)
    return report, prediction, metadata


def _collect_logits(
    model: C1SModel,
    runtime: C1SSuccessorRuntime,
    ids: Sequence[str],
    *,
    device: str,
    intervention: str = "baseline",
    batch_size: int = 8,
) -> dict[str, list[float]]:
    active_device = torch.device(device)
    model.to(active_device).eval()
    output: dict[str, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            chunk = list(ids[start : start + batch_size])
            batch = runtime.get_batch(chunk, slots=model.config.slots, pin_memory=True)
            moved = _tensor_to_device(batch, active_device)
            source = moved["source_hidden"]
            mask = moved["source_mask"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if intervention == "zero_hidden":
                    logits = model(torch.zeros_like(source), mask)["logits"]
                elif intervention == "shuffled_hidden":
                    logits = model(source.roll(1, dims=0), mask.roll(1, dims=0))["logits"]
                else:
                    boundary = model.boundary(source, mask)
                    if intervention == "no_core":
                        logits = model.forward_from_boundary(boundary, disable_recurrence=True)["logits"]
                    elif intervention == "step5_shuffle":
                        payloads = boundary.payloads
                        for step in range(model.config.operations):
                            payloads = model.transition(
                                payloads,
                                boundary.operation_states[:, step],
                                boundary.source_weights[:, step],
                                boundary.target_weights[:, step],
                                boundary.operation_active[:, step],
                            )
                            if step == 4 and payloads.shape[0] > 1:
                                payloads = payloads.roll(1, dims=0)
                        logits, _ = model.logits_from_state(payloads, boundary.addresses, boundary.query_key)
                    elif intervention == "baseline":
                        logits = model.forward_from_boundary(boundary)["logits"]
                    else:
                        raise ValueError(intervention)
            for example_id, row in zip(chunk, logits.float().cpu().tolist()):
                output[example_id] = row
    return output


def _lookup(logits: Mapping[str, Sequence[float]]):
    return lambda record, **_kwargs: {"logits": list(logits[str(record["example_id"])])}


def _causal_evaluation_metadata(
    records: Sequence[Mapping[str, Any]],
    target_bank: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Combine target-bank raw answers with source-record pair semantics.

    The target bank is authoritative for the raw class consumed by the
    successor runtime.  Semantic answers and label mappings remain sourced
    from the immutable C0R records because they are deliberately not runtime
    forward inputs.
    """

    bank_records = target_bank.get("records") if isinstance(target_bank, Mapping) else None
    if not isinstance(bank_records, Mapping):
        raise ValueError("target bank records are required for causal evaluation")
    result: list[dict[str, Any]] = []
    for record in records:
        example_id = str(record.get("example_id", ""))
        target = bank_records.get(example_id)
        if not isinstance(target, Mapping):
            raise KeyError(f"causal target missing from target bank: {example_id}")
        row = dict(record)
        # Runtime targets are the source of truth for raw answer class.
        row["answer_index"] = target.get("answer_index")
        row["answer"] = target.get("answer_index")
        # Keep these explicit even if an upstream record uses a nested form;
        # the successor evaluator rejects absent or ambiguous mappings.
        if "semantic_answer" not in row and "semantic_answer" in target:
            row["semantic_answer"] = target["semantic_answer"]
        if "label_mapping" not in row and "label_mapping" in target:
            row["label_mapping"] = target["label_mapping"]
        result.append(row)
    return result


def _stage_prerequisites(repo_root: Path, stage: str, *, device: str) -> dict[str, Any]:
    stage = stage.upper()
    preflight = _audit_preflight(repo_root, stage)
    predecessor = _audit_predecessor(repo_root, stage)
    static = _audit_static_pins(repo_root, replay=False)
    device_report = _device_audit(device)
    target_integrity: dict[str, Any] = {"status": "NOT_REQUIRED", "passed": True}
    cache_integrity: dict[str, Any] = {"status": "NOT_REQUIRED", "passed": True}
    # Every stage launch revalidates the sealed S1 bank and full immutable
    # cache tree.  A PASS preflight is not a time-of-use integrity exemption.
    target_integrity = _audit_target_bank_integrity(repo_root)
    cache_integrity = _audit_cache_integrity(repo_root)
    source = artifacts.source_hashes(repo_root)
    expected_source = None
    try:
        expected_source = artifacts.read_json(_absolute(repo_root, _stage_config(stage)[3]) / "result.json").get("source_identity")
    except Exception:
        pass
    identity = artifacts.source_identity(source)
    checks = {
        "preflight": preflight.get("passed") is True,
        "predecessor": predecessor.get("passed") is True,
        "static_pins": static.get("passed") is True,
        "device": device_report.get("passed") is True,
        "source_matches_preflight": identity == expected_source,
        "sealed_target_bank_and_ledger": target_integrity.get("passed") is True,
        "cache_seal_tree_and_source_identity": cache_integrity.get("passed") is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "preflight": preflight,
        "predecessor": predecessor,
        "static_pins": static,
        "device": device_report,
        "target_bank_integrity": target_integrity,
        "cache_integrity": cache_integrity,
        "source_identity": identity,
        "source_hashes": source,
    }


def _completion_envelope(
    result: Mapping[str, Any],
    *,
    output_root: Path,
    result_sha256: str,
    evidence_seal_sha256: str,
    seal_replay: Mapping[str, Any],
    incomplete_status: str,
) -> dict[str, Any]:
    """Return a fail-closed post-seal envelope without mutating sealed files."""

    completed = {
        **dict(result),
        "output_root": output_root.as_posix(),
        "result_sha256": result_sha256,
        "evidence_seal_sha256": evidence_seal_sha256,
        "seal_replay": dict(seal_replay),
        "evidence_complete": seal_replay.get("passed") is True,
    }
    if seal_replay.get("passed") is True:
        return completed
    completed.update(
        {
            "scientific_status_before_seal_audit": result.get("status"),
            "scientific_passed_before_seal_audit": result.get("passed") is True,
            "status": incomplete_status,
            "passed": False,
            "exit_code": 1,
            "authorizes": "nothing",
        }
    )
    return completed


def _seal_and_replay(root: Path, *, identity: str, stage: str) -> tuple[str, dict[str, Any]]:
    """Turn seal write/audit errors into an explicit incomplete envelope."""

    seal_sha = ""
    try:
        seal_sha = artifacts.write_evidence_seal(root, identity=identity, stage=stage)
        replay = artifacts.audit_evidence_seal(
            root,
            expected_identity=identity,
            expected_stage=stage,
        )
    except Exception as exc:  # an unsealed consumed root must still return exit 1
        replay = {
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "seal_write_or_replay_error": True,
        }
    return seal_sha, replay


def _reconcile_optimizer_steps(*, observed: int, base: int, reported: int) -> int:
    """Verify that post-mutation accounting matches the trainer endpoint."""

    expected = int(base) + int(reported)
    if int(observed) != expected:
        raise RuntimeError(
            "optimizer step accounting mismatch: "
            f"observed={int(observed)} expected={expected} "
            f"base={int(base)} reported={int(reported)}"
        )
    return expected


def _materialized_model_writes(root: Path, names: Sequence[str]) -> int:
    """Count every registered model path that materialized on disk.

    A failed digest or a partially failed ``torch.save`` is still a model write
    for single-use accounting, even when the artifact is unusable.
    """

    return sum((Path(root) / name).is_file() for name in names)


def _finish_stage(
    root: Path,
    *,
    result: dict[str, Any],
    identity: str,
    stage: str,
) -> dict[str, Any]:
    artifacts.write_json(root / "result.json", result)
    seal_sha, replay = _seal_and_replay(root, identity=identity, stage=stage)
    return _completion_envelope(
        result,
        output_root=root,
        result_sha256=artifacts.sha256_file(root / "result.json"),
        evidence_seal_sha256=seal_sha,
        seal_replay=replay,
        incomplete_status=f"INCOMPLETE_V2_A_C1S_{stage}_EVIDENCE_SEAL",
    )


def _load_stage_runtime(repo_root: Path) -> tuple[Any, dict[str, Any], dict[str, Any], C1SSuccessorRuntime]:
    store = load_offline_records(repo_root / DATA_ROOT)
    bank, ledger = _load_bank(repo_root)
    dataset = CachedShardDataset(repo_root / CACHE_ROOT)
    runtime = C1SSuccessorRuntime(dataset, bank)
    return store, bank, ledger, runtime


def run_s1(repo_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S1_ROOT
    lease = repo_root / contract.S1_LEASE
    if root.exists() or lease.exists():
        return _refusal("S1", root, lease, "fixed S1 root or lease already exists")
    prerequisites = _stage_prerequisites(repo_root, "S1", device=device)
    if prerequisites.get("passed") is not True:
        return _refusal("S1", root, lease, f"S1 prerequisites failed: {prerequisites.get('checks')}")
    source_before = dict(prerequisites["source_hashes"])
    source_id = str(prerequisites["source_identity"])
    artifacts.claim_single_use(identity=contract.S1_IDENTITY, stage="S1", output_root=root, lease_path=lease, formal=False)
    started = time.perf_counter()
    optimizer_steps = 0
    model_writes = 0
    try:
        artifacts.snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.stage_manifest("S1"))
        artifacts.write_json(root / "prerequisites.json", prerequisites)
        store, _bank, ledger, runtime = _load_stage_runtime(repo_root)
        ids = list(ledger["s1"]["ids"])
        rows = _records_for_ids(store, ids)
        schedule = build_overfit_schedule(rows, seed=contract.S1_ORDER_SEED, maximum_updates=4_000)
        schedule_info = schedule_report(schedule)
        artifacts.write_json(root / "schedule.json", schedule_info)
        model, heads = _fresh_model(contract.MODEL_CONFIG, seed=contract.S1_MODEL_SEED)
        spec = _registered_train_spec("S1")
        loss_weights, causal_margins, loss_contract = _registered_loss_contract()
        artifacts.write_json(root / "optimization-contract.json", loss_contract)
        if loss_contract.get("passed") is not True:
            raise RuntimeError("registered optimization/loss contract is inconsistent")

        def record_s1_step(update: int) -> None:
            nonlocal optimizer_steps
            optimizer_steps = int(update)

        training = train_fixed_endpoint(
            model,
            heads,
            runtime,
            schedule,
            identity=contract.S1_IDENTITY,
            order_seed=contract.S1_ORDER_SEED,
            model_seed=contract.S1_MODEL_SEED,
            spec=spec,
            device=device,
            loss_weights=loss_weights,
            margins=causal_margins,
            on_optimizer_step=record_s1_step,
        )
        optimizer_steps = _reconcile_optimizer_steps(
            observed=optimizer_steps,
            base=0,
            reported=int(training["optimizer_steps"]),
        )
        endpoint_path = root / "endpoint.pt"
        endpoint_sha = save_endpoint(
            endpoint_path,
            model=model,
            diagnostic_heads=heads,
            update=4_000,
            identity=contract.S1_IDENTITY,
            schedule_sha256=schedule_info["sha256"],
            metrics=training,
        )
        model_writes = 1
        training["endpoint_sha256"] = endpoint_sha
        artifacts.write_json(root / "training.json", training)
        report, _prediction, _metadata = _evaluate_mechanism(
            model,
            heads,
            runtime,
            ids,
            device=device,
            s1=True,
            bootstrap_seed=contract.S1_BOOTSTRAP_SEED,
        )
        report["s0_single_slot_structural_control_replay"] = (
            _s0_single_slot_structural_control_replay(repo_root)
        )
        gate = s1_gate(report)
        gate["checks"]["mechanism_each_family"] = all(child.get("mechanism_pass") is True for child in report["family_reports"].values())
        gate["passed"] = all(gate["checks"].values())
        gate["authorizes"] = "S2_DISCOVERY_ONLY" if gate["passed"] else "nothing"
        artifacts.write_json(root / "evaluation.json", report)
        artifacts.write_json(root / "gate.json", gate)
        model_writes = _materialized_model_writes(root, ("endpoint.pt",))
        source_stable = source_before == artifacts.source_hashes(repo_root)
        passed = bool(gate["passed"] and source_stable and optimizer_steps == 4_000 and model_writes == 1)
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s1-result.v1",
            "identity": contract.S1_IDENTITY,
            "stage": "S1",
            "formal": False,
            "status": "PASS_V2_A_C1S_S1_QUALIFICATION" if passed else "FAIL_V2_A_C1S_S1_QUALIFICATION",
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "source_identity": source_id,
            "source_stable": source_stable,
            "optimizer_steps": optimizer_steps,
            "model_writes": model_writes,
            "fixed_endpoint": 4_000,
            "checkpoint_selection": False,
            "gate": gate,
            "endpoint_sha256": endpoint_sha,
            "wall_seconds": time.perf_counter() - started,
            "authorizes": "S2_DISCOVERY_ONLY" if passed else "nothing",
            "s2_status": "AUTHORIZED_NOT_RUN" if passed else "NOT_AUTHORIZED",
            "s3_status": "NOT_AUTHORIZED",
            "c2_authorized": False,
            "v2a_passed": False,
        }
        return _finish_stage(root, result=result, identity=contract.S1_IDENTITY, stage="S1")
    except Exception as exc:  # noqa: BLE001
        model_writes = _materialized_model_writes(root, ("endpoint.pt",))
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s1-result.v1",
            "identity": contract.S1_IDENTITY,
            "stage": "S1",
            "formal": False,
            "status": "CRASH_V2_A_C1S_S1_QUALIFICATION",
            "passed": False,
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "optimizer_steps": optimizer_steps,
            "model_writes": model_writes,
            "authorizes": "nothing",
            "s2_status": "NOT_AUTHORIZED",
            "s3_status": "NOT_AUTHORIZED",
            "c2_authorized": False,
            "v2a_passed": False,
        }
        return _finish_stage(root, result=result, identity=contract.S1_IDENTITY, stage="S1")


def run_s2(repo_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S2_ROOT
    lease = repo_root / contract.S2_LEASE
    if root.exists() or lease.exists():
        return _refusal("S2", root, lease, "fixed S2 root or lease already exists")
    prerequisites = _stage_prerequisites(repo_root, "S2", device=device)
    if prerequisites.get("passed") is not True:
        return _refusal("S2", root, lease, f"S2 prerequisites failed: {prerequisites.get('checks')}")
    source_before = dict(prerequisites["source_hashes"])
    source_id = str(prerequisites["source_identity"])
    artifacts.claim_single_use(identity=contract.S2_IDENTITY, stage="S2", output_root=root, lease_path=lease, formal=False)
    optimizer_steps = 0
    model_writes = 0
    arm_status = {"K8": "NOT_RUN", "K1": "NOT_RUN"}
    started = time.perf_counter()
    try:
        artifacts.snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.stage_manifest("S2"))
        artifacts.write_json(root / "prerequisites.json", prerequisites)
        store, _bank, ledger, runtime = _load_stage_runtime(repo_root)
        train_ids = list(ledger["s2"]["train_ids"])
        eval_ids = list(ledger["s2"]["eval_ids"])
        spec = _registered_train_spec("S2")
        schedule = build_balanced_schedule(
            _records_for_ids(store, train_ids),
            seed=contract.S2_ORDER_SEED,
            spec=spec,
        )
        schedule_info = schedule_report(schedule)
        artifacts.write_json(root / "schedule.json", schedule_info)
        arm_reports: dict[str, Any] = {}
        models: dict[str, C1SModel] = {}
        heads_by_arm: dict[str, SuccessorDiagnosticHeads] = {}
        loss_weights, causal_margins, loss_contract = _registered_loss_contract()
        artifacts.write_json(root / "optimization-contract.json", loss_contract)
        if loss_contract.get("passed") is not True:
            raise RuntimeError("registered optimization/loss contract is inconsistent")
        for arm, config in (("K8", contract.MODEL_CONFIG), ("K1", contract.K1_CONFIG)):
            model, heads = _fresh_model(config, seed=contract.S2_MODEL_SEED)
            arm_step_base = optimizer_steps
            arm_status[arm] = "RUNNING"

            def record_arm_step(update: int, *, base: int = arm_step_base) -> None:
                nonlocal optimizer_steps
                optimizer_steps = base + int(update)
                arm_status[arm] = "PARTIAL_TRAINING"

            training = train_fixed_endpoint(
                model,
                heads,
                runtime,
                schedule,
                identity=f"{contract.S2_IDENTITY}-{arm}",
                order_seed=contract.S2_ORDER_SEED,
                model_seed=contract.S2_MODEL_SEED,
                spec=spec,
                device=device,
                loss_weights=loss_weights,
                margins=causal_margins,
                on_optimizer_step=record_arm_step,
            )
            optimizer_steps = _reconcile_optimizer_steps(
                observed=optimizer_steps,
                base=arm_step_base,
                reported=int(training["optimizer_steps"]),
            )
            arm_status[arm] = "TRAINED_ENDPOINT_UNVERIFIED"
            endpoint = root / f"{arm.casefold()}-endpoint.pt"
            endpoint_sha = save_endpoint(
                endpoint,
                model=model,
                diagnostic_heads=heads,
                update=4_608,
                identity=f"{contract.S2_IDENTITY}-{arm}",
                schedule_sha256=schedule_info["sha256"],
                metrics=training,
            )
            model_writes += 1
            training["endpoint_sha256"] = endpoint_sha
            arm_reports[arm] = {
                "training": training,
                "schedule_sha256": schedule_info["sha256"],
            }
            arm_status[arm] = "TRAINED_FIXED_ENDPOINT"
            models[arm] = model
            heads_by_arm[arm] = heads
            artifacts.write_json(root / f"{arm.casefold()}-training.json", training)
            if arm == "K8":
                torch.cuda.empty_cache()
        k8_report, k8_prediction, k8_metadata = _evaluate_mechanism(
            models["K8"],
            heads_by_arm["K8"],
            runtime,
            eval_ids,
            device=device,
            bootstrap_seed=contract.S2_BOOTSTRAP_SEED,
        )
        k1_report, k1_prediction, _ = _evaluate_mechanism(
            models["K1"],
            heads_by_arm["K1"],
            runtime,
            eval_ids,
            device=device,
            bootstrap_seed=contract.S2_BOOTSTRAP_SEED,
        )
        k1_report["single_slot_control"] = validate_k1_single_slot_control(
            k1_prediction,
            k1_report,
            expected_records=len(eval_ids),
        )
        gain = paired_eval_gain(k8_prediction["logits"], k1_prediction["logits"], k8_metadata, bootstrap_seed=contract.S2_BOOTSTRAP_SEED)
        k8_report["paired_gain"] = gain
        budget = {
            "k8_updates": arm_reports["K8"]["training"]["optimizer_steps"],
            "k1_updates": arm_reports["K1"]["training"]["optimizer_steps"],
            "k8_hours": arm_reports["K8"]["training"]["wall_seconds"] / 3_600.0,
            "k1_hours": arm_reports["K1"]["training"]["wall_seconds"] / 3_600.0,
            "same_schedule_sha256": arm_reports["K8"]["schedule_sha256"]
            == arm_reports["K1"]["schedule_sha256"]
            == schedule_info["sha256"],
            "single_cuda_serial_arms": True,
            "ddp_used": False,
            "parameter_parity": models["K8"].parameter_report()["trainable_parameters"] == models["K1"].parameter_report()["trainable_parameters"],
        }
        gate = s2_gate(k8_report, k1_report, budget=budget)
        gate["checks"]["parameter_parity"] = budget["parameter_parity"]
        gate["checks"]["same_schedule"] = budget["same_schedule_sha256"]
        gate["passed"] = all(gate["checks"].values())
        gate["authorizes"] = "S3_SINGLE_SEED_FORMAL_ELIGIBILITY" if gate["passed"] else "nothing"
        arm_status = {"K8": "EVALUATED", "K1": "EVALUATED"}
        artifacts.write_json(root / "k8-evaluation.json", k8_report)
        artifacts.write_json(root / "k1-evaluation.json", k1_report)
        artifacts.write_json(root / "paired-gain.json", gain)
        artifacts.write_json(root / "budget.json", budget)
        artifacts.write_json(root / "gate.json", gate)
        model_writes = _materialized_model_writes(
            root,
            ("k8-endpoint.pt", "k1-endpoint.pt"),
        )
        source_stable = source_before == artifacts.source_hashes(repo_root)
        passed = bool(gate["passed"] and source_stable and optimizer_steps == 9_216 and model_writes == 2)
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s2-result.v1",
            "identity": contract.S2_IDENTITY,
            "stage": "S2",
            "formal": False,
            "status": "PASS_V2_A_C1S_S2_QUALIFICATION" if passed else "FAIL_V2_A_C1S_S2_QUALIFICATION",
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "source_identity": source_id,
            "source_stable": source_stable,
            "optimizer_steps": optimizer_steps,
            "model_writes": model_writes,
            "arm_status": arm_status,
            "gate": gate,
            "budget": budget,
            "wall_seconds": time.perf_counter() - started,
            "authorizes": "S3_SINGLE_SEED_FORMAL_ELIGIBILITY" if passed else "nothing",
            "s3_status": "AUTHORIZED_NOT_RUN" if passed else "NOT_AUTHORIZED",
            "c2_authorized": False,
            "v2a_passed": False,
        }
        return _finish_stage(root, result=result, identity=contract.S2_IDENTITY, stage="S2")
    except Exception as exc:  # noqa: BLE001
        model_writes = _materialized_model_writes(
            root,
            ("k8-endpoint.pt", "k1-endpoint.pt"),
        )
        arm_status = {
            arm: ("NOT_RUN" if status == "RUNNING" else status)
            for arm, status in arm_status.items()
        }
        result = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s2-result.v1",
            "identity": contract.S2_IDENTITY,
            "stage": "S2",
            "formal": False,
            "status": "CRASH_V2_A_C1S_S2_QUALIFICATION",
            "passed": False,
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "optimizer_steps": optimizer_steps,
            "model_writes": model_writes,
            "arm_status": arm_status,
            "authorizes": "nothing",
            "s3_status": "NOT_AUTHORIZED",
            "c2_authorized": False,
            "v2a_passed": False,
        }
        return _finish_stage(root, result=result, identity=contract.S2_IDENTITY, stage="S2")


def _formal_mechanism_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "overall": bool(report.get("mechanism_pass", False)),
        "ERE": bool(report.get("family_reports", {}).get("ERE", {}).get("mechanism_pass", False)),
        "CPS": bool(report.get("family_reports", {}).get("CPS", {}).get("mechanism_pass", False)),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _save_deployment(path: Path, model: C1SModel, *, identity: str) -> str:
    if path.exists():
        raise FileExistsError(path)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    if any(name.startswith("training_auxiliary.") for name in state):
        raise ValueError("deployment state still contains training auxiliary")
    torch.save(
        {
            "schema_version": f"{contract.SCHEMA_PREFIX}.deployment-state.v1",
            "identity": identity,
            "config": asdict(model.config),
            "model_state": state,
        },
        path,
    )
    return artifacts.sha256_file(path)


def run_s3(repo_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    root = repo_root / contract.S3_ROOT
    lease = repo_root / contract.S3_LEASE
    if root.exists() or lease.exists():
        return _refusal("S3", root, lease, "fixed S3 root or lease already exists")
    prerequisites = _stage_prerequisites(repo_root, "S3", device=device)
    if prerequisites.get("passed") is not True:
        return _refusal("S3", root, lease, f"S3 prerequisites failed: {prerequisites.get('checks')}")
    source_before = dict(prerequisites["source_hashes"])
    source_id = str(prerequisites["source_identity"])
    artifacts.claim_single_use(identity=contract.S3_IDENTITY, stage="S3", output_root=root, lease_path=lease, formal=True)
    gate_names = list(contract.S3_CONFIG["gate_order"])
    gate_status = {name: "NOT_RUN" for name in gate_names}
    optimizer_steps = 0
    model_writes = 0
    started = time.perf_counter()

    def terminal(status: str, *, stopped_at: str | None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        nonlocal model_writes
        model_writes = _materialized_model_writes(
            root,
            ("training-endpoint.pt", "deployment.pt"),
        )
        source_audit_error: str | None = None
        try:
            source_stable = source_before == artifacts.source_hashes(repo_root)
        except Exception as exc:  # source failure must not prevent crash sealing
            source_stable = False
            source_audit_error = f"{type(exc).__name__}: {exc}"
        requested_status = status
        if status == "PASS_V2_A_C1S_S3_SINGLE_SEED_FORMAL" and not source_stable:
            status = "INCOMPLETE_V2_A_C1S_S3_SOURCE_IDENTITY"
            stopped_at = stopped_at or "G010_accounting_and_evidence_integrity"
        passed = status == "PASS_V2_A_C1S_S3_SINGLE_SEED_FORMAL" and source_stable
        result: dict[str, Any] = {
            "schema_version": f"{contract.SCHEMA_PREFIX}.s3-result.v1",
            "identity": contract.S3_IDENTITY,
            "stage": "S3",
            "formal": True,
            "status": status,
            "passed": passed,
            "exit_code": 0 if passed else 1,
            "source_identity": source_id,
            "source_stable": source_stable,
            "source_audit_error": source_audit_error,
            "requested_terminal_status": requested_status,
            "optimizer_steps": optimizer_steps,
            "model_writes": model_writes,
            "fixed_endpoint": 6_144,
            "checkpoint_selection": False,
            "gate_status": dict(gate_status),
            "stopped_at": stopped_at,
            "unrun_gates": [name for name, value in gate_status.items() if value == "NOT_RUN"],
            "wall_seconds": time.perf_counter() - started,
            "authorizes": "nothing",
            "c2_authorized": False,
            "v2a_passed": False,
        }
        if extra:
            result.update(dict(extra))
        return _finish_stage(root, result=result, identity=contract.S3_IDENTITY, stage="S3")

    try:
        artifacts.snapshot_sources(repo_root, root)
        artifacts.write_json(root / "contract-manifest.json", contract.stage_manifest("S3"))
        artifacts.write_json(root / "prerequisites.json", prerequisites)
        store, _bank, ledger, runtime = _load_stage_runtime(repo_root)
        train_rows = [row for row in store.records if row.get("split") == "train"]
        spec = _registered_train_spec("S3")
        schedule = build_balanced_schedule(
            train_rows,
            seed=contract.S3_ORDER_SEED,
            spec=spec,
        )
        schedule_info = schedule_report(schedule)
        artifacts.write_json(root / "schedule.json", schedule_info)
        model, heads = _fresh_model(contract.MODEL_CONFIG, seed=contract.S3_MODEL_SEED)
        loss_weights, causal_margins, loss_contract = _registered_loss_contract()
        artifacts.write_json(root / "optimization-contract.json", loss_contract)
        if loss_contract.get("passed") is not True:
            raise RuntimeError("registered optimization/loss contract is inconsistent")

        def record_s3_step(update: int) -> None:
            nonlocal optimizer_steps
            optimizer_steps = int(update)

        training = train_fixed_endpoint(
            model,
            heads,
            runtime,
            schedule,
            identity=contract.S3_IDENTITY,
            order_seed=contract.S3_ORDER_SEED,
            model_seed=contract.S3_MODEL_SEED,
            spec=spec,
            device=device,
            loss_weights=loss_weights,
            margins=causal_margins,
            on_optimizer_step=record_s3_step,
        )
        optimizer_steps = _reconcile_optimizer_steps(
            observed=optimizer_steps,
            base=0,
            reported=int(training["optimizer_steps"]),
        )
        endpoint_sha = save_endpoint(
            root / "training-endpoint.pt",
            model=model,
            diagnostic_heads=heads,
            update=6_144,
            identity=contract.S3_IDENTITY,
            schedule_sha256=schedule_info["sha256"],
            metrics=training,
        )
        model_writes = 1
        training["endpoint_sha256"] = endpoint_sha
        artifacts.write_json(root / "training.json", training)
        deployment = strip_training_auxiliary(model).to(device).eval()

        validation_rows = [row for row in store.records if row.get("split") == "validation"]
        validation_ids = [row["example_id"] for row in validation_rows]
        validation_logits = _collect_logits(deployment, runtime, validation_ids, device=device)
        validation_report = evaluate_behavior(validation_rows, _lookup(validation_logits))
        validation_gate = gate_g004_validation(validation_report)
        artifacts.write_json(root / "validation.json", validation_report)
        artifacts.write_json(root / "g004-validation-gate.json", validation_gate)
        gate_status["G004_validation_behavior"] = "PASS" if validation_gate["passed"] else "FAIL"
        if not validation_gate["passed"]:
            return terminal("FAIL_V2_A_C1S_S3_SINGLE_SEED_FORMAL", stopped_at="G004_validation_behavior")

        ood_rows = [row for row in store.records if str(row.get("split", "")).endswith("_ood")]
        ood_ids = [row["example_id"] for row in ood_rows]
        ood_logits = _collect_logits(deployment, runtime, ood_ids, device=device)
        ood_report = evaluate_behavior(ood_rows, _lookup(ood_logits))
        ood_gate = gate_g005_ood(ood_report)
        artifacts.write_json(root / "ood.json", ood_report)
        artifacts.write_json(root / "g004-ood-gate.json", ood_gate)
        gate_status["G004_OOD_behavior"] = "PASS" if ood_gate["passed"] else "FAIL"
        if not ood_gate["passed"]:
            return terminal("FAIL_V2_A_C1S_S3_SINGLE_SEED_FORMAL", stopped_at="G004_OOD_behavior")

        causal_rows = [row for row in store.records if row.get("split") == "causal_pairs"]
        causal_ids = [row["example_id"] for row in causal_rows]
        causal_logits = _collect_logits(deployment, runtime, causal_ids, device=device)
        causal_metadata = _causal_evaluation_metadata(causal_rows, _bank)
        causal_margin_report = causal_pair_margin_report(
            [causal_logits[str(row["example_id"])] for row in causal_metadata],
            causal_metadata,
            bootstrap_seed=contract.S3_BOOTSTRAP_SEED,
        )
        causal_behavior_gate = gate_g006_causal_behavior(causal_margin_report)
        causal_margin_gate = gate_g006_causal_margin(causal_margin_report)
        causal_gate = {
            "passed": bool(causal_behavior_gate["passed"] and causal_margin_gate["passed"]),
            "behavior_subgate": causal_behavior_gate,
            "margin_subgate": causal_margin_gate,
        }
        artifacts.write_json(root / "causal.json", causal_margin_report)
        artifacts.write_json(root / "g004-causal-gate.json", causal_gate)
        gate_status["G004_causal_behavior"] = "PASS" if causal_gate["passed"] else "FAIL"
        if not causal_gate["passed"]:
            return terminal("FAIL_V2_A_C1S_S3_SINGLE_SEED_FORMAL", stopped_at="G004_causal_behavior")

        mechanism_report, _mechanism_prediction, _mechanism_metadata = _evaluate_mechanism(
            model,
            heads,
            runtime,
            validation_ids,
            device=device,
            bootstrap_seed=contract.S3_BOOTSTRAP_SEED,
        )
        mechanism_gate = _formal_mechanism_gate(mechanism_report)
        artifacts.write_json(root / "ownership-state.json", mechanism_report)
        artifacts.write_json(root / "g005-ownership-state-gate.json", mechanism_gate)
        gate_status["G005_ownership_and_state"] = "PASS" if mechanism_gate["passed"] else "FAIL"
        if not mechanism_gate["passed"]:
            return terminal("FAIL_V2_A_C1S_S3_SINGLE_SEED_FORMAL", stopped_at="G005_ownership_and_state")

        zero_logits = _collect_logits(deployment, runtime, validation_ids, device=device, intervention="zero_hidden")
        shuffle_logits = _collect_logits(deployment, runtime, validation_ids, device=device, intervention="shuffled_hidden")
        hidden_report = {
            "zero_hidden": paired_intervention_margin_report(
                [validation_logits[str(row["example_id"])] for row in _mechanism_metadata],
                [zero_logits[str(row["example_id"])] for row in _mechanism_metadata],
                _mechanism_metadata,
                bootstrap_seed=contract.S3_BOOTSTRAP_SEED,
            ),
            "shuffled_hidden": paired_intervention_margin_report(
                [validation_logits[str(row["example_id"])] for row in _mechanism_metadata],
                [shuffle_logits[str(row["example_id"])] for row in _mechanism_metadata],
                _mechanism_metadata,
                bootstrap_seed=contract.S3_BOOTSTRAP_SEED,
            ),
        }
        hidden_gate = gate_g008_hidden(hidden_report)
        artifacts.write_json(root / "hidden-interventions.json", hidden_report)
        artifacts.write_json(root / "g006-hidden-gate.json", hidden_gate)
        gate_status["G006_hidden_interventions"] = "PASS" if hidden_gate["passed"] else "FAIL"
        if not hidden_gate["passed"]:
            return terminal("FAIL_V2_A_C1S_S3_SINGLE_SEED_FORMAL", stopped_at="G006_hidden_interventions")

        no_core_logits = _collect_logits(deployment, runtime, validation_ids, device=device, intervention="no_core")
        step5_logits = _collect_logits(deployment, runtime, validation_ids, device=device, intervention="step5_shuffle")
        recurrence_report = {
            "h0": paired_intervention_margin_report(
                [validation_logits[str(row["example_id"])] for row in _mechanism_metadata],
                [no_core_logits[str(row["example_id"])] for row in _mechanism_metadata],
                _mechanism_metadata,
                bootstrap_seed=contract.S3_BOOTSTRAP_SEED,
            ),
            "step5_shuffle": paired_intervention_margin_report(
                [validation_logits[str(row["example_id"])] for row in _mechanism_metadata],
                [step5_logits[str(row["example_id"])] for row in _mechanism_metadata],
                _mechanism_metadata,
                bootstrap_seed=contract.S3_BOOTSTRAP_SEED,
            ),
        }
        recurrence_gate = gate_g009_recurrence(recurrence_report)
        artifacts.write_json(root / "recurrence-interventions.json", recurrence_report)
        artifacts.write_json(root / "g008-recurrence-gate.json", recurrence_gate)
        gate_status["G008_recurrence_interventions"] = "PASS" if recurrence_gate["passed"] else "FAIL"
        if not recurrence_gate["passed"]:
            return terminal("FAIL_V2_A_C1S_S3_SINGLE_SEED_FORMAL", stopped_at="G008_recurrence_interventions")

        integrity_report = {
            "slot": mechanism_report["invariance"]["slot_permutation"],
            "strip": mechanism_report["invariance"]["strip"],
            "trace_probe_parameter_count": 0,
        }
        integrity_gate = gate_g010_integrity(integrity_report)
        artifacts.write_json(root / "deployment-integrity.json", integrity_report)
        artifacts.write_json(root / "g009-strip-gate.json", integrity_gate)
        gate_status["G009_deployment_auxiliary_strip"] = "PASS" if integrity_gate["passed"] else "FAIL"
        if not integrity_gate["passed"]:
            return terminal("FAIL_V2_A_C1S_S3_SINGLE_SEED_FORMAL", stopped_at="G009_deployment_auxiliary_strip")
        deployment_sha = _save_deployment(root / "deployment.pt", deployment, identity=contract.S3_IDENTITY)
        model_writes = 2

        source_stable = source_before == artifacts.source_hashes(repo_root)
        accounting = {
            "passed": bool(
                optimizer_steps == 6_144
                and model_writes == 2
                and source_stable
                and training.get("completed_updates") == 6_144
                and training.get("checkpoint_selection") is False
            ),
            "optimizer_steps": optimizer_steps,
            "model_writes": model_writes,
            "source_stable": source_stable,
            "deployment_sha256": deployment_sha,
            "training_endpoint_sha256": endpoint_sha,
            "cache_reused_read_only": True,
            "ddp_used": False,
            "single_cuda": True,
        }
        artifacts.write_json(root / "accounting.json", accounting)
        gate_status["G010_accounting_and_evidence_integrity"] = "PASS" if accounting["passed"] else "FAIL"
        if not accounting["passed"]:
            return terminal("INCOMPLETE_V2_A_C1S_S3_SINGLE_SEED_FORMAL", stopped_at="G010_accounting_and_evidence_integrity")
        return terminal(
            "PASS_V2_A_C1S_S3_SINGLE_SEED_FORMAL",
            stopped_at=None,
            extra={"deployment_sha256": deployment_sha, "training_endpoint_sha256": endpoint_sha},
        )
    except Exception as exc:  # noqa: BLE001
        return terminal(
            "CRASH_V2_A_C1S_S3_SINGLE_SEED_FORMAL",
            stopped_at=next((name for name, value in gate_status.items() if value == "NOT_RUN"), None),
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )


def load_result(root: Path) -> dict[str, Any]:
    return artifacts.read_json(Path(root) / "result.json")


__all__ = [
    "CACHE_ROOT",
    "DATA_ROOT",
    "FULL_RECORD_COUNT",
    "SPLIT_LEDGER_NAME",
    "TARGET_BANK_NAME",
    "load_result",
    "run_preflight",
    "run_s1",
    "run_s2",
    "run_s3",
]
