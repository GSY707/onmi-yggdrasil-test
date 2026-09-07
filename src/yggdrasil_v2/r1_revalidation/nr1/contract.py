from __future__ import annotations

from pathlib import Path
from typing import Any

from .qualification import (
    DECISION_KILL_MIN,
    FORMAL_SEED,
    HAND_FIXTURE_RELATIVE,
    HAND_FIXTURE_SHA256,
    METRIC_KILL_MIN,
    NUMERIC_COUNTS,
    RELATION_COUNTS,
)


CONTRACT_VERSION = "r1r-p1-nr1-numeric-relation-measurement"
DESIGN = Path("docs/v2-r1r-p1-nr1-numeric-relation-measurement-design.md")
EXECUTION = Path("docs/v2-r1r-p1-nr1-execution-command.md")
ROOTS = {
    "preflight": Path("artifacts/v2-r1r/p1-nr1-preflight-20260817-1"),
    "qualification": Path("artifacts/v2-r1r/p1-nr1-qualification-20260817-1"),
}
TRANSPORT_ROOT = Path("tmp/p1-nr1-transport-20260817-1")
V8L_ROOTS = {
    "preflight": {
        "root": Path("artifacts/v2-r1r/p1-v8l-causal-state-ladder-preflight-20260812-1"),
        "status": "PASS_P1_V8L_CAUSAL_STATE_LADDER_PREFLIGHT",
        "seal_sha256": "06BA4E180AED6EB2FC5BBFF3DEECC85D8FF74E958B60106E2494E4F48B685115",
    },
    "anchor_cache": {
        "root": Path("artifacts/v2-r1r/p1-v8l-causal-state-ladder-anchor-cache-20260812-1"),
        "status": "PASS_P1_V8L_CAUSAL_STATE_LADDER_ANCHOR_CACHE",
        "seal_sha256": "5F1AABBF3D601559BE3CEC1DEDC0390B4C84780962947768B089DBC3D3AF3C14",
    },
    "qualification": {
        "root": Path("artifacts/v2-r1r/p1-v8l-causal-state-ladder-qualification-20260812-1"),
        "status": "FAIL_P1_V8L_BOOTSTRAP",
        "seal_sha256": "3332CD3D24DCE85BEB3D8ECA2A7D1D669E97CAF40924CABED88AC07F2C44FE75",
    },
}

FROZEN_CONTRACT_HASHES = {
    DESIGN.as_posix(): "66B6617F8F40537B83636D87F916761353774F0B02763F6BBBE89BE67D53F84C",
    EXECUTION.as_posix(): "73864875D1F92574F7C4B6F9CC9A6E7DEB5C58B2C44BABEB2D6842293826F012",
}


def contract_manifest() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "formal_seed": FORMAL_SEED,
        "roots": {name: root.as_posix() for name, root in ROOTS.items()},
        "transport_root": TRANSPORT_ROOT.as_posix(),
        "numeric_counts": NUMERIC_COUNTS,
        "relation_counts": RELATION_COUNTS,
        "hand_fixture": HAND_FIXTURE_RELATIVE.as_posix(),
        "hand_fixture_sha256": HAND_FIXTURE_SHA256,
        "decision_kill_min": DECISION_KILL_MIN,
        "metric_kill_min": METRIC_KILL_MIN,
        "pass_authorizes": "P1-H1 design only",
        "p1_completed_on_pass": False,
        "p2_eligible_on_pass": False,
    }
