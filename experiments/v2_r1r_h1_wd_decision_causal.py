"""Isolated launcher for the decision-causal WD mechanism screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

from yggdrasil_v2.r1_revalidation.h1_wd_causal.contract import (
    OUTPUT_ROOT,
    SOURCE_CHECKPOINT,
    SOURCE_CHECKPOINT_SHA256,
)
from yggdrasil_v2.r1_revalidation.h1_wd_causal.experiment import (
    run_decision_causal_screen,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V2-R1R decision-causal WD non-formal screen"
    )
    parser.add_argument("command", choices=("run-h1-wd-decision-causal",))
    args = parser.parse_args()
    if args.command != "run-h1-wd-decision-causal":
        raise AssertionError("unreachable decision-causal command")
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = run_decision_causal_screen(repo_root)
    except Exception as exc:
        root = repo_root / OUTPUT_ROOT
        if root.is_dir():
            source_checkpoint = repo_root / SOURCE_CHECKPOINT
            payload = {
                "schema_version": (
                    "yggdrasil.v2-r1r.p1-h1-wd-decision-causal.crash.v1"
                ),
                "status": "CRASH_WD_DECISION_CAUSAL_NONFORMAL",
                "scope": "NONFORMAL_WD_DECISION_CAUSAL_SCREEN_ONLY",
                "authorizes": "nothing",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": time.time(),
                "source_checkpoint_unchanged": bool(
                    source_checkpoint.is_file()
                    and _sha256(source_checkpoint) == SOURCE_CHECKPOINT_SHA256
                ),
            }
            (root / "crash.json").write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "scope": result["scope"],
                "authorizes": result["authorizes"],
                "output_root": OUTPUT_ROOT.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
