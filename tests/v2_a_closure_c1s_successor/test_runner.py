from __future__ import annotations

import builtins
import dis
import gzip
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from yggdrasil_v2.v2_a.closure_c1s_successor import runner
from yggdrasil_v2.v2_a.closure_c1s_successor import artifacts


def _dynamic_feature_names(family: str) -> list[str]:
    if family == "ERE":
        return [
            "present", "touched", "changed", "query_owner",
            "operation_source", "operation_target", "query_semantic_match", "stable_after",
        ]
    return [
        "processed", "legal", "valid", "budget_ok", "goal_satisfied",
        "final_constraints_satisfied", "running_best", "final_winner",
    ]


def _dynamic_targets(family: str, *, static: bool = False) -> dict[str, object]:
    names = _dynamic_feature_names(family)
    if static:
        values = [
            [[1] * len(names), [0] * len(names)]
            for _ in range(3)
        ]
    else:
        values = [
            [
                [int(bool((step + slot) % 2))] * len(names)
                for slot in range(2)
            ]
            for step in range(3)
        ]
    return {
        "state_feature_names": names,
        "state_values": values,
        "state_feature_mask": [
            [[1] * len(names) for _ in range(2)] for _ in range(3)
        ],
        "state_step_mask": [1, 1, 1],
        "operation_active": [1, 0, 1],
    }


def test_stage_commands_have_distinct_identity_root_lease_and_formal_boundary() -> None:
    values = [runner._stage_config(stage) for stage in ("S1", "S2", "S3")]
    assert len({value[0] for value in values}) == 3
    assert len({value[1] for value in values}) == 3
    assert len({value[2] for value in values}) == 3
    assert [value[-1] for value in values] == [False, False, True]


def test_runner_has_no_unresolved_module_global_references() -> None:
    unresolved: dict[str, list[str]] = {}
    namespace = vars(runner)
    for name, value in namespace.items():
        if not inspect.isfunction(value) or value.__module__ != runner.__name__:
            continue
        missing = sorted(
            {
                str(instruction.argval)
                for instruction in dis.get_instructions(value)
                if instruction.opname == "LOAD_GLOBAL"
                and not hasattr(builtins, str(instruction.argval))
                and str(instruction.argval) not in namespace
            }
        )
        if missing:
            unresolved[name] = missing
    assert unresolved == {}


def test_existing_preflight_root_refuses_before_any_audit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.contract, "S1_PREFLIGHT_ROOT", Path("already"))
    monkeypatch.setattr(runner.contract, "S1_PREFLIGHT_LEASE", Path("already.lease"))
    (tmp_path / "already").mkdir()
    result = runner.run_preflight(tmp_path, "S1")
    assert result["exit_code"] == 2
    assert result["status"] == "REFUSE_V2_A_C1S_S1_SINGLE_USE"
    assert not (tmp_path / "already.lease").exists()


def test_device_audit_requires_registered_bf16_support(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA GeForce RTX 4070 Laptop GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 9))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _index: (6_000, 8_000))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    refused = runner._device_audit("cuda")
    assert refused["bf16_supported"] is False
    assert refused["passed"] is False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert runner._device_audit("cuda")["passed"] is True


def test_s1_predecessor_requires_exact_authorized_scope(tmp_path, monkeypatch) -> None:
    result = {
        "identity": runner.contract.S0_IDENTITY,
        "status": "PASS_V2_A_C1S_S0_QUALIFICATION",
        "passed": True,
        "s1_status": "AUTHORIZED_NOT_RUN",
        "authorizes": runner.contract.S0_AUTHORIZED_SCOPE,
        "never_authorizes": ["S2", "single-seed formal"],
    }
    monkeypatch.setattr(runner.artifacts, "read_json", lambda _path: dict(result))
    monkeypatch.setattr(
        runner.artifacts,
        "audit_external_pin",
        lambda *_args, **_kwargs: {"passed": True},
    )

    accepted = runner._audit_predecessor(tmp_path, "S1")
    assert accepted["passed"] is True
    assert accepted["checks"]["authorized_scope"] is True

    result["authorizes"] = "S1 and any later stage"
    refused = runner._audit_predecessor(tmp_path, "S1")
    assert refused["passed"] is False
    assert refused["checks"]["authorized_scope"] is False


def test_prediction_batch_merge_and_slice_preserve_order() -> None:
    first = {"logits": torch.tensor([[1.0, 2.0]]), "nested": {"x": torch.tensor([[3.0]])}}
    second = {"logits": torch.tensor([[4.0, 5.0]]), "nested": {"x": torch.tensor([[6.0]])}}
    merged = runner._merge_prediction_batches([first, second])
    assert merged["logits"].tolist() == [[1.0, 2.0], [4.0, 5.0]]
    sliced = runner._slice_prediction(merged, torch.tensor([1]))
    assert sliced["logits"].tolist() == [[4.0, 5.0]]


def test_binary_metric_is_finite_and_exact() -> None:
    metric = runner._binary_metric([True, False, True, True])
    assert metric["point"] == 0.75
    assert metric["n"] == 4
    assert 0.0 <= metric["wilson_lower"] <= metric["wilson_upper"] <= 1.0


def test_preflight_audit_separates_disposable_steps_from_formal_training(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.contract, "S1_PREFLIGHT_ROOT", Path("s1-preflight"))
    root = tmp_path / "s1-preflight"
    root.mkdir()
    artifacts.write_json(
        root / "result.json",
        {
            "identity": runner._preflight_identity("S1"),
            "stage": "S1-PREFLIGHT",
            "status": "PASS_V2_A_C1S_S1_PREFLIGHT",
            "passed": True,
            "authorizes": "S1_SINGLE_USE_LAUNCH_ONLY",
            "disposable_benchmark_optimizer_steps": 100,
            "formal_stage_optimizer_steps": 0,
            "formal_stage_training_started": False,
            "optimization_contract": {"passed": True},
            "s0_single_slot_structural_control_replay": {
                "status": "PASS_STRUCTURAL_INSTRUMENT_REPLAY",
                "passed": True,
                "learned_k1_control": False,
                "multi_address_qualified": None,
            },
        },
    )
    artifacts.write_evidence_seal(
        root,
        identity=runner._preflight_identity("S1"),
        stage="S1-PREFLIGHT",
    )
    report = runner._audit_preflight(tmp_path, "S1")
    assert report["passed"] is True
    assert report["checks"]["disposable_benchmark_steps"] is True
    assert report["checks"]["formal_training_not_started"] is True
    assert report["checks"]["optimization_contract"] is True
    assert report["checks"]["s0_structural_instrument"] is True


def test_s0_replay_is_structured_zero_update_instrument_not_learned_k1(tmp_path, monkeypatch) -> None:
    model_relative = Path("src/yggdrasil_v2/v2_a/closure_c1s/model.py")
    result = {
        "identity": runner.contract.S0_IDENTITY,
        "passed": True,
        "status": "PASS_V2_A_C1S_S0_QUALIFICATION",
        "training_started": False,
        "optimizer_steps": 0,
        "model_writes": 0,
        "s1_status": "AUTHORIZED_NOT_RUN",
    }
    structure = {
        "passed": True,
        "config_k8": dict(runner.contract.MODEL_CONFIG),
        "config_k1": dict(runner.contract.K1_CONFIG),
        "integrity_k1": {"passed": True},
        "parameters_k8": {"trainable_parameters": 10, "deployment_parameters": 9},
        "parameters_k1": {"trainable_parameters": 10, "deployment_parameters": 9},
        "registered_controls": {
            "k1_null": {
                "single_slot_null": True,
                "effective_slot_count": [1.0, 1.0],
                "effects": [[1.0], [2.0]],
            },
            "addressed_positive": {
                "single_slot_null": False,
                "relevant_effect": [2.0],
                "irrelevant_max_effect": [0.0],
            },
            "legacy_uniform_mean_null": {"ownership_contrast": [0.0]},
        },
        "targeted_transition": {
            "selected_slot_max_delta": 1.0,
            "unselected_slot_max_delta": 0.0,
            "inactive_transition_max_delta": 0.0,
        },
        "permutation": {
            "logit_max_abs_delta": 0.0,
            "trajectory_max_abs_delta": 0.0,
        },
        "thresholds": {},
    }
    manifest = {
        "model": dict(runner.contract.MODEL_CONFIG),
        "matched_k1_model": dict(runner.contract.K1_CONFIG),
    }
    seal = {
        "files": {f"source_snapshot/{model_relative.as_posix()}": "MODEL-SHA"}
    }

    def fake_read_json(path: Path):
        return {
            "result.json": result,
            "s0-structure.json": structure,
            "contract-manifest.json": manifest,
            "evidence-seal.json": seal,
        }[Path(path).name]

    monkeypatch.setattr(
        runner.artifacts,
        "audit_external_pin",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(runner.artifacts, "read_json", fake_read_json)
    monkeypatch.setattr(
        runner.artifacts,
        "sha256_file",
        lambda path: "MODEL-SHA" if Path(path).as_posix().endswith(model_relative.as_posix()) else "HASH",
    )

    report = runner._s0_single_slot_structural_control_replay(tmp_path)
    assert report["status"] == "PASS_STRUCTURAL_INSTRUMENT_REPLAY"
    assert report["passed"] is True
    assert report["learned_k1_control"] is False
    assert report["evaluated_records"] == 0
    assert report["multi_address_applicable"] is False
    assert report["multi_address_qualified"] is None


def test_target_bank_integrity_rehashes_gzip_and_ledger(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.contract, "S1_PREFLIGHT_ROOT", Path("s1-preflight"))
    monkeypatch.setattr(runner, "FULL_RECORD_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "audit_target_bank",
        lambda bank, expected_count=None: {"passed": len(bank.get("records", {})) == expected_count},
    )
    monkeypatch.setattr(
        runner,
        "load_offline_records",
        lambda _root: SimpleNamespace(records=[]),
    )
    monkeypatch.setattr(
        runner,
        "_audit_split_ledger_semantics",
        lambda *_args, **_kwargs: {"passed": True},
    )
    root = tmp_path / "s1-preflight"
    root.mkdir()
    bank = {"records": {"e0": {"example_id": "e0"}}}
    with gzip.open(root / runner.TARGET_BANK_NAME, "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(bank, handle, sort_keys=True)
    ledger = {"selection_preimage_sha256": "ABC"}
    artifacts.write_json(root / runner.SPLIT_LEDGER_NAME, ledger)
    bank_sha = artifacts.sha256_file(root / runner.TARGET_BANK_NAME)
    ledger_sha = artifacts.sha256_file(root / runner.SPLIT_LEDGER_NAME)
    producer_audit = {
        "sha256": bank_sha,
        "split_ledger_sha256": ledger_sha,
        "selection_preimage_sha256": "ABC",
    }
    artifacts.write_json(root / "target-bank-audit.json", producer_audit)
    artifacts.write_json(
        root / "result.json",
        {
            "target_bank": producer_audit,
        },
    )
    artifacts.write_evidence_seal(
        root,
        identity=runner._preflight_identity("S1"),
        stage="S1-PREFLIGHT",
    )

    assert runner._audit_target_bank_integrity(tmp_path)["passed"] is True
    artifacts.write_json(root / runner.SPLIT_LEDGER_NAME, {"selection_preimage_sha256": "MUTATED"})
    assert runner._audit_target_bank_integrity(tmp_path)["passed"] is False


def test_fresh_s1_target_materialization_is_read_back_before_use(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "FULL_RECORD_COUNT", 1)
    monkeypatch.setattr(
        runner,
        "audit_target_bank",
        lambda bank, expected_count=None: {
            "passed": len(bank.get("records", {})) == expected_count,
        },
    )
    monkeypatch.setattr(
        runner,
        "_audit_split_ledger_semantics",
        lambda *_args, **_kwargs: {"passed": True},
    )
    bank = {"records": {"e0": {"example_id": "e0"}}}
    ledger = {"selection_preimage_sha256": "SELECTION"}
    runner._write_gzip_json(tmp_path / runner.TARGET_BANK_NAME, bank)
    artifacts.write_json(tmp_path / runner.SPLIT_LEDGER_NAME, ledger)
    bank_sha = artifacts.sha256_file(tmp_path / runner.TARGET_BANK_NAME)
    ledger_sha = artifacts.sha256_file(tmp_path / runner.SPLIT_LEDGER_NAME)

    report, read_bank, read_ledger = runner._audit_fresh_s1_target_materialization(
        tmp_path,
        [{"example_id": "e0"}],
        expected_bank_sha256=bank_sha,
        expected_ledger_sha256=ledger_sha,
        expected_selection_preimage_sha256="SELECTION",
    )
    assert report["status"] == "BUILT_WRITTEN_AND_READBACK_AUDITED"
    assert report["passed"] is True
    assert read_bank == bank
    assert read_ledger == ledger

    mutated, _bank, _ledger = runner._audit_fresh_s1_target_materialization(
        tmp_path,
        [{"example_id": "e0"}],
        expected_bank_sha256="0" * 64,
        expected_ledger_sha256=ledger_sha,
        expected_selection_preimage_sha256="SELECTION",
    )
    assert mutated["passed"] is False
    assert mutated["checks"]["target_bank_sha256"] is False


def test_dynamic_supervision_coverage_rejects_static_cells_without_temporal_change() -> None:
    rows = {
        family: {
            "example_id": family,
            "family": family,
            "mechanism_supported": True,
            **_dynamic_targets(family, static=True),
        }
        for family in ("ERE", "CPS")
    }
    report = runner._dynamic_supervision_coverage(
        rows, {"probe": ["ERE", "CPS"]}
    )

    assert report["passed"] is False
    assert report["selections"]["probe"]["families"]["ERE"]["operation_active"]["passed"] is True
    touched = report["selections"]["probe"]["families"]["ERE"]["dynamic_features"]["touched"]
    assert touched["positives"] > 0
    assert touched["negatives"] > 0
    assert touched["temporal_changes"] == 0
    assert touched["passed"] is False


def test_dynamic_supervision_coverage_passes_and_unsupported_rows_do_not_pollute() -> None:
    rows = {
        family: {
            "example_id": family,
            "family": family,
            "mechanism_supported": True,
            **_dynamic_targets(family),
        }
        for family in ("ERE", "CPS")
    }
    rows["unsupported"] = {
        "example_id": "unsupported",
        "family": "CPS",
        "mechanism_supported": False,
        # Deliberately malformed/static: behavior-only rows must never enter
        # mechanism target coverage.
        "state_values": [],
        "state_feature_mask": [],
        "state_step_mask": [],
        "operation_active": [],
    }
    clean = runner._dynamic_supervision_coverage(
        rows, {"probe": ["ERE", "CPS"]}
    )
    mixed = runner._dynamic_supervision_coverage(
        rows, {"probe": ["ERE", "CPS", "unsupported"]}
    )

    assert clean["passed"] is True
    assert mixed["passed"] is True
    assert mixed["selections"]["probe"]["ignored_unsupported_records"] == 1
    assert mixed["selections"]["probe"]["families"] == clean["selections"]["probe"]["families"]


def test_split_ledger_semantic_audit_recomputes_balanced_supported_train_selection(monkeypatch) -> None:
    def ids(prefix: str, family: str, count: int) -> list[str]:
        return [f"{prefix}-{family}-{index:04d}" for index in range(count)]

    s1_ids = ids("s1", "ERE", 16) + ids("s1", "CPS", 16)
    s2_train = ids("s2-train", "ERE", 3_072) + ids("s2-train", "CPS", 3_072)
    s2_eval = ids("s2-eval", "ERE", 512) + ids("s2-eval", "CPS", 512)
    selected = s1_ids + s2_train + s2_eval
    records = [
        {
            "example_id": example_id,
            "family": "ERE" if "-ERE-" in example_id else "CPS",
            "split": "train",
        }
        for example_id in selected
    ]
    records.extend(
        [
            {
                "example_id": f"formal-validation-{family}-0000",
                "family": family,
                "split": "validation",
            }
            for family in ("ERE", "CPS")
        ]
    )
    records.append(
        {
            "example_id": "behavior-only-CPS-0000",
            "family": "CPS",
            "split": "cps-distractor_ood",
        }
    )
    bank_rows = {
        row["example_id"]: {
            **row,
            "mechanism_supported": row["split"] in {"train", "validation"},
            "query_answer_independent": row["split"] in {"train", "validation"},
            "query_owner": int(row["example_id"].rsplit("-", 1)[-1]) % 4
            if row["split"] in {"train", "validation"}
            else 0,
            **(
                _dynamic_targets(row["family"])
                if row["split"] in {"train", "validation"}
                else {}
            ),
        }
        for row in records
    }
    ledger = {
        "selection_preimage_sha256": "PREIMAGE",
        "s1": {"ids": s1_ids},
        "s2": {"train_ids": s2_train, "eval_ids": s2_eval},
    }
    monkeypatch.setattr(runner, "select_s1_ids", lambda _records: list(s1_ids))
    monkeypatch.setattr(
        runner,
        "select_s2_ids",
        lambda _records, *, s1_ids: {"train_ids": s2_train, "eval_ids": s2_eval},
    )
    monkeypatch.setattr(
        runner,
        "build_split_ledger",
        lambda _records, *, s1_ids, s2: dict(ledger),
    )

    report = runner._audit_split_ledger_semantics(
        records, {"records": bank_rows}, ledger
    )
    assert report["passed"] is True
    assert report["checks"]["ledger_exact_recomputation"] is True
    assert report["checks"]["selected_rows_mechanism_supported"] is True
    assert report["checks"]["selected_queries_answer_independent"] is True
    assert report["checks"]["s1_query_slots_nondegenerate"] is True
    assert report["checks"]["s2_train_query_slots_nondegenerate"] is True
    assert report["checks"]["s2_eval_query_slots_nondegenerate"] is True
    assert report["checks"]["s1_dynamic_supervision_coverage"] is True
    assert report["checks"]["s2_train_dynamic_supervision_coverage"] is True
    assert report["checks"]["s2_eval_dynamic_supervision_coverage"] is True
    assert report["checks"]["formal_validation_dynamic_supervision_coverage"] is True
    assert report["dynamic_supervision_coverage"]["passed"] is True
    assert report["checks"]["unsupported_rows_are_ood_only"] is True
    assert report["unsupported_behavior_only_count"] == 1

    mutated = {key: dict(value) for key, value in bank_rows.items()}
    mutated[s1_ids[0]]["query_answer_independent"] = False
    refused = runner._audit_split_ledger_semantics(
        records, {"records": mutated}, ledger
    )
    assert refused["passed"] is False
    assert refused["checks"]["selected_queries_answer_independent"] is False


def test_cache_integrity_requires_full_tree_and_source_identity(tmp_path, monkeypatch) -> None:
    class FakeStore:
        def __len__(self) -> int:
            return 26_624

    class FakeDataset:
        manifest = {"identity": "CACHE-I", "schema_version": "CACHE-SCHEMA"}

        def __init__(self, _root: Path) -> None:
            pass

        def __len__(self) -> int:
            return 26_624

    monkeypatch.setattr(runner.artifacts, "audit_external_pin", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(runner.artifacts, "read_json", lambda _path: {"passed": True, "identity": "CACHE-I", "source_stable": True})
    monkeypatch.setattr(runner, "load_offline_records", lambda _root: FakeStore())
    monkeypatch.setattr(runner, "CachedShardDataset", FakeDataset)
    monkeypatch.setattr(runner, "validate_cache_source_identity", lambda _dataset, _store: {"passed": True})
    assert runner._audit_cache_integrity(tmp_path)["passed"] is True

    monkeypatch.setattr(runner.artifacts, "audit_external_pin", lambda *_args, **_kwargs: {"passed": False})
    assert runner._audit_cache_integrity(tmp_path)["passed"] is False


def test_fresh_s1_cache_report_is_not_marked_not_required() -> None:
    class Dataset:
        manifest = {"identity": "CACHE-I", "schema_version": "CACHE-S"}

        def __len__(self) -> int:
            return 3

    report = runner._audit_fresh_s1_cache(
        {"passed": True},
        {"passed": True},
        Dataset(),
        expected_records=3,
    )
    assert report["status"] == "FULL_CACHE_AND_SOURCE_AUDITED"
    assert report["passed"] is True

    failed = runner._audit_fresh_s1_cache(
        {"passed": True},
        {"passed": False},
        Dataset(),
        expected_records=3,
    )
    assert failed["passed"] is False


def test_preflight_benchmark_uses_only_registered_contract_seeds(monkeypatch) -> None:
    monkeypatch.setitem(runner.contract.S1_CONFIG, "preflight_benchmark_model_seed", 111)
    monkeypatch.setitem(runner.contract.S1_CONFIG, "preflight_benchmark_order_seed", 222)
    captured: dict[str, int] = {}

    def fake_schedule(rows, *, seed, maximum_updates):
        captured["schedule_seed"] = seed
        assert len(rows) == 32
        assert maximum_updates == 100
        return ["schedule"]

    def fake_model(_config, *, seed):
        captured["fresh_model_seed"] = seed
        return object(), object()

    def fake_train(_model, _heads, _runtime, _schedule, **kwargs):
        captured["train_order_seed"] = kwargs["order_seed"]
        captured["train_model_seed"] = kwargs["model_seed"]
        if kwargs.get("on_optimizer_step") is not None:
            kwargs["on_optimizer_step"](100)
        return {"passed": True, "optimizer_steps": 100}

    monkeypatch.setattr(runner, "build_overfit_schedule", fake_schedule)
    monkeypatch.setattr(runner, "_fresh_model", fake_model)
    monkeypatch.setattr(runner, "train_fixed_endpoint", fake_train)
    monkeypatch.setattr(
        runner,
        "_registered_loss_contract",
        lambda: (object(), object(), {"passed": True}),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    records = [
        {"example_id": f"{family}-{index}", "family": family, "split": "train"}
        for family in ("ERE", "CPS")
        for index in range(16)
    ]
    store = SimpleNamespace(records=records)
    ledger = {"s1": {"ids": []}, "s2": {"eval_ids": []}}

    report = runner._preflight_benchmark(
        store,
        object(),
        ledger,
        device="cuda",
    )
    assert report["passed"] is True
    assert captured == {
        "schedule_seed": 222,
        "fresh_model_seed": 111,
        "train_order_seed": 222,
        "train_model_seed": 111,
    }


def test_all_stage_launch_prerequisites_require_bank_and_cache_revalidation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_audit_preflight", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(runner, "_audit_predecessor", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(runner, "_audit_static_pins", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(runner, "_device_audit", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(runner, "_audit_target_bank_integrity", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(runner, "_audit_cache_integrity", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(runner.artifacts, "source_hashes", lambda _root: {"runner.py": "HASH"})
    monkeypatch.setattr(runner.artifacts, "source_identity", lambda _hashes: "SOURCE-I")
    monkeypatch.setattr(runner.artifacts, "read_json", lambda _path: {"source_identity": "SOURCE-I"})

    for stage in ("S1", "S2", "S3"):
        report = runner._stage_prerequisites(tmp_path, stage, device="cuda")
        assert report["passed"] is True
        assert report["checks"]["sealed_target_bank_and_ledger"] is True
        assert report["checks"]["cache_seal_tree_and_source_identity"] is True

    monkeypatch.setattr(runner, "_audit_target_bank_integrity", lambda *_args, **_kwargs: {"passed": False})
    blocked = runner._stage_prerequisites(tmp_path, "S2", device="cuda")
    assert blocked["passed"] is False
    assert blocked["checks"]["sealed_target_bank_and_ledger"] is False


def test_post_seal_envelope_cannot_return_pass_when_replay_fails(tmp_path) -> None:
    result = {
        "status": "PASS_V2_A_C1S_S1_QUALIFICATION",
        "passed": True,
        "exit_code": 0,
        "authorizes": "S2_DISCOVERY_ONLY",
    }
    completed = runner._completion_envelope(
        result,
        output_root=tmp_path,
        result_sha256="A" * 64,
        evidence_seal_sha256="B" * 64,
        seal_replay={"passed": False, "mismatched": ["result.json"]},
        incomplete_status="INCOMPLETE_V2_A_C1S_S1_EVIDENCE_SEAL",
    )

    assert completed["scientific_status_before_seal_audit"] == result["status"]
    assert completed["status"] == "INCOMPLETE_V2_A_C1S_S1_EVIDENCE_SEAL"
    assert completed["passed"] is False
    assert completed["exit_code"] == 1
    assert completed["authorizes"] == "nothing"
    assert completed["evidence_complete"] is False


def test_seal_write_failure_becomes_explicit_incomplete_replay(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner.artifacts,
        "write_evidence_seal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("seal unavailable")),
    )

    seal_sha, replay = runner._seal_and_replay(
        tmp_path,
        identity="TEST-IDENTITY",
        stage="S1-PREFLIGHT",
    )

    assert seal_sha == ""
    assert replay["passed"] is False
    assert replay["seal_write_or_replay_error"] is True
    assert replay["reason"] == "OSError: seal unavailable"


def test_optimizer_step_reconciliation_is_cumulative_and_fail_closed() -> None:
    assert runner._reconcile_optimizer_steps(observed=4_608, base=0, reported=4_608) == 4_608
    assert runner._reconcile_optimizer_steps(observed=9_216, base=4_608, reported=4_608) == 9_216

    with pytest.raises(RuntimeError, match="optimizer step accounting mismatch"):
        runner._reconcile_optimizer_steps(observed=13_824, base=4_608, reported=4_608)


def test_materialized_model_write_accounting_counts_unverified_files(tmp_path) -> None:
    (tmp_path / "k8-endpoint.pt").write_bytes(b"partial-model")

    assert runner._materialized_model_writes(
        tmp_path,
        ("k8-endpoint.pt", "k1-endpoint.pt"),
    ) == 1

    (tmp_path / "k1-endpoint.pt").write_bytes(b"complete-model")
    assert runner._materialized_model_writes(
        tmp_path,
        ("k8-endpoint.pt", "k1-endpoint.pt"),
    ) == 2


def test_s3_source_hash_failure_is_still_sealed_as_crash(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner.contract, "S3_ROOT", Path("s3-root"))
    monkeypatch.setattr(runner.contract, "S3_LEASE", Path("s3-lease.jsonl"))
    monkeypatch.setattr(
        runner,
        "_stage_prerequisites",
        lambda *_args, **_kwargs: {
            "passed": True,
            "source_hashes": {"runner.py": "HASH"},
            "source_identity": "SOURCE-I",
        },
    )
    monkeypatch.setattr(
        runner.artifacts,
        "snapshot_sources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pre-seal fault")),
    )
    monkeypatch.setattr(
        runner.artifacts,
        "source_hashes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("source unavailable")),
    )

    result = runner.run_s3(tmp_path, device="cuda")

    assert result["status"] == "CRASH_V2_A_C1S_S3_SINGLE_SEED_FORMAL"
    assert result["passed"] is False
    assert result["source_stable"] is False
    assert result["source_audit_error"] == "OSError: source unavailable"
    assert result["evidence_complete"] is True
    assert (tmp_path / "s3-root" / "result.json").is_file()
    assert (tmp_path / "s3-root" / "evidence-seal.json").is_file()
