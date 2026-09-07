from __future__ import annotations

from pathlib import Path

from yggdrasil_v2.r1_revalidation.nr1.artifacts import sha256
from yggdrasil_v2.r1_revalidation.nr1.contract import (
    CONTRACT_VERSION,
    DESIGN,
    EXECUTION,
    FROZEN_CONTRACT_HASHES,
    ROOTS,
    TRANSPORT_ROOT,
    contract_manifest,
)
from yggdrasil_v2.r1_revalidation.nr1.runner import source_files


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_contract_is_frozen_and_roots_are_single_use() -> None:
    assert CONTRACT_VERSION == "r1r-p1-nr1-numeric-relation-measurement"
    assert contract_manifest()["p1_completed_on_pass"] is False
    assert contract_manifest()["p2_eligible_on_pass"] is False
    assert contract_manifest()["pass_authorizes"] == "P1-H1 design only"
    assert contract_manifest()["decision_kill_min"] == 0.80
    assert contract_manifest()["metric_kill_min"] == 0.80
    assert contract_manifest()["hand_fixture_sha256"] != "TO_BE_FROZEN"
    assert ROOTS == {
        "preflight": Path("artifacts/v2-r1r/p1-nr1-preflight-20260817-1"),
        "qualification": Path("artifacts/v2-r1r/p1-nr1-qualification-20260817-1"),
    }
    assert TRANSPORT_ROOT == Path("tmp/p1-nr1-transport-20260817-1")
    for relative, expected in FROZEN_CONTRACT_HASHES.items():
        assert expected != "TO_BE_FROZEN"
        assert sha256(REPO_ROOT / relative) == expected


def test_active_source_surface_is_complete() -> None:
    assert DESIGN in source_files(REPO_ROOT)
    assert EXECUTION in source_files(REPO_ROOT)
    assert Path("experiments/v2_r1_revalidation.py") in source_files(REPO_ROOT)
    assert Path(
        "tests/v2_r1r_p1_nr1/fixtures/nr1_hand_authored_cases.json"
    ) in source_files(REPO_ROOT)
    for relative in (
        Path("README.md"),
        Path("docs/DIRECTORY_REFERENCE.md"),
        Path("docs/next-stage-test-plan.md"),
        Path("docs/v2-r1-revalidation-task-design.md"),
        Path("docs/v2-r1r-p0-result.md"),
    ):
        assert relative in source_files(REPO_ROOT)
        assert "NR1" in (REPO_ROOT / relative).read_text(encoding="utf-8")
