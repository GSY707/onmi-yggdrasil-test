from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from yggdrasil_v2.v2_a.closure_c1s_successor import artifacts


def test_source_identity_covers_actual_local_import_time_closure() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pinned = {(repo_root / relative).resolve() for relative in artifacts.SOURCE_FILES}
    script = r"""
import json
from pathlib import Path
import runpy
import sys

repo_root = Path(sys.argv[1]).resolve()
from yggdrasil_v2.v2_a.closure_c1s_successor import artifacts

for relative in artifacts.SOURCE_FILES:
    if relative.as_posix().startswith("tests/v2_a_closure_c1s_successor/"):
        runpy.run_path(str(repo_root / relative), run_name=f"_source_closure_{relative.stem}")

loaded = []
for module in tuple(sys.modules.values()):
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        continue
    path = Path(module_file).resolve()
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        continue
    if relative.suffix == ".py" and (
        relative.as_posix().startswith("src/yggdrasil_v2/")
        or relative.as_posix().startswith("tests/v2_a_closure_c1s_successor/")
    ):
        loaded.append(path.as_posix())
print(json.dumps(sorted(set(loaded))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(repo_root)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    loaded = {Path(path).resolve() for path in json.loads(completed.stdout)}

    missing = sorted(path.relative_to(repo_root).as_posix() for path in loaded - pinned)
    assert missing == []


def test_source_identity_pins_transitive_model_cache_data_and_semantic_dependencies() -> None:
    sources = set(artifacts.SOURCE_FILES)
    required = {
        Path("src/yggdrasil_v2/v2_a/closure_c1s/model.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1/cache.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1/data.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/targets.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/runtime.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/objective.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/evaluate.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1s_successor/runner.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c0r/compact_trace.py"),
        Path("src/yggdrasil_v2/r1_revalidation/production/__init__.py"),
        Path("src/yggdrasil_v2/r1_revalidation/production/renderer.py"),
        Path("src/yggdrasil_v2/r1_revalidation/learner/canonical.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1s/__init__.py"),
        Path("src/yggdrasil_v2/v2_a/closure_c1s/contract.py"),
    }
    assert required <= sources

    hashes = {path.as_posix(): "A" for path in sorted(required)}
    changed = dict(hashes)
    changed["src/yggdrasil_v2/v2_a/closure_c1s_successor/targets.py"] = "B"
    assert artifacts.source_identity(hashes) != artifacts.source_identity(changed)


def test_stage_seal_replays_and_detects_mutation(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifacts.write_json(root / "result.json", {"passed": True})
    artifacts.write_evidence_seal(root, identity="I", stage="S1")
    assert artifacts.audit_evidence_seal(root, expected_identity="I", expected_stage="S1")["passed"]
    artifacts.write_json(root / "result.json", {"passed": False})
    assert not artifacts.audit_evidence_seal(root, expected_identity="I", expected_stage="S1")["passed"]


def test_single_use_lease_is_fail_closed(tmp_path) -> None:
    root = tmp_path / "stage"
    lease = tmp_path / "stage.lease.jsonl"
    artifacts.claim_single_use(
        identity="I", stage="S1", output_root=root, lease_path=lease, formal=False
    )
    payload = json.loads(lease.read_text(encoding="utf-8"))
    assert payload["identity"] == "I"
    try:
        artifacts.claim_single_use(
            identity="I", stage="S1", output_root=root, lease_path=lease, formal=False
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("second single-use claim must fail")


def test_dynamic_stage_predecessor_requires_exact_status_and_authorization(tmp_path) -> None:
    root = tmp_path / "s1"
    root.mkdir()
    artifacts.write_json(
        root / "result.json",
        {
            "identity": "S1-I",
            "stage": "S1",
            "status": "PASS-S1",
            "passed": True,
            "authorizes": "S2-ONLY",
        },
    )
    artifacts.write_evidence_seal(root, identity="S1-I", stage="S1")
    report = artifacts.audit_stage_predecessor(
        root,
        expected_identity="S1-I",
        expected_stage="S1",
        expected_status="PASS-S1",
        expected_authorizes="S2-ONLY",
    )
    assert report["passed"] is True
    refused = artifacts.audit_stage_predecessor(
        root,
        expected_identity="S1-I",
        expected_stage="S1",
        expected_status="PASS-S1",
        expected_authorizes="S3",
    )
    assert refused["passed"] is False


def test_sealed_file_pins_require_tree_replay_and_explicit_sha(tmp_path) -> None:
    root = tmp_path / "producer"
    root.mkdir()
    bank = root / "target-bank.json.gz"
    bank.write_bytes(b"sealed-bank")
    artifacts.write_json(root / "split-ledger.json", {"selection": "fixed"})
    expected = {
        bank.name: artifacts.sha256_file(bank),
        "split-ledger.json": artifacts.sha256_file(root / "split-ledger.json"),
    }
    artifacts.write_evidence_seal(root, identity="BANK-I", stage="S1-PREFLIGHT")

    report = artifacts.audit_sealed_file_pins(
        root,
        expected_identity="BANK-I",
        expected_stage="S1-PREFLIGHT",
        expected_files=expected,
    )
    assert report["passed"] is True
    assert all(row["passed"] for row in report["files"].values())

    bank.write_bytes(b"mutated-bank")
    mutated = artifacts.audit_sealed_file_pins(
        root,
        expected_identity="BANK-I",
        expected_stage="S1-PREFLIGHT",
        expected_files=expected,
    )
    assert mutated["passed"] is False
    assert mutated["files"][bank.name]["checks"]["actual_matches_expected"] is False
