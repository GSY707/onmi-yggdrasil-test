from pathlib import Path

import pytest

from yggdrasil_v2.r1_revalidation.h1 import runner, workflow


def test_preflight_refuses_unfrozen_contract_before_data_or_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    monkeypatch.setattr(runner, "source_files", lambda _root: (Path("unit.py"),))
    monkeypatch.setattr(
        runner,
        "formal_runtime_audit",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        runner,
        "contract_hash_audit",
        lambda _root: {"passed": False},
    )
    monkeypatch.setattr(runner, "history_replay", lambda _root: {"passed": True})
    monkeypatch.setattr(runner, "_cli_audit", lambda _root: {"passed": True})
    monkeypatch.setattr(workflow, "_tests_audit", lambda _root: {"passed": True})
    monkeypatch.setattr(workflow, "hardware_audit", lambda _root: {"passed": True})
    monkeypatch.setattr(workflow, "architecture_audit", lambda *_args, **_kwargs: {"passed": True})
    monkeypatch.setattr(workflow, "source_identity", lambda *_args: "IDENTITY")

    def forbidden() -> None:
        raise AssertionError("data generation must not start")

    monkeypatch.setattr(workflow, "generate_registered_inputs", forbidden)
    result = workflow.preflight_action(
        tmp_path,
        {"passed": True},
        root,
        "IDENTITY",
        {"git_identity_start": {"passed": True}},
    )
    assert result["passed"] is False
    assert result["status"] == workflow.FAIL_PREFLIGHT
    assert result["details"]["stopped_before_data_or_cache"] is True


def test_development_refuses_invalid_preflight_before_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "preflight_valid",
        lambda *_args: {"passed": False, "reason": "fixture"},
    )
    result = workflow.development_action(
        tmp_path,
        tmp_path / "development",
        "IDENTITY",
        {},
    )
    assert result["passed"] is False
    assert result["status"] == workflow.FAIL_DEVELOPMENT
    assert result["stop_reason"] == "preflight_not_current_sealed_pass"


def test_shared_route_noop_requires_predictions_trace_and_state_hashes() -> None:
    normal = {
        "predictions": [1, 2],
        "trace_predictions": [[1], [2]],
        "state_hashes": {
            "logits": "L",
            "final_state": "H",
            "trajectory": "T",
        },
    }
    trials = {name: dict(normal) for name in workflow._ROUTE_INTERVENTIONS}
    assert workflow._shared_route_noop(normal, trials)["passed"] is True
    trials["flip_route"] = {
        **normal,
        "state_hashes": {**normal["state_hashes"], "trajectory": "changed"},
    }
    assert workflow._shared_route_noop(normal, trials)["passed"] is False
