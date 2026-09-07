"""Isolated launcher for the non-formal H1 write-then-delete screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from yggdrasil_v2.r1_revalidation.h1_wd.contract import OUTPUT_ROOT
from yggdrasil_v2.r1_revalidation.h1_wd.experiment import run_write_delete_screen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V2-R1R H1-WD isolated non-formal mechanism screen"
    )
    parser.add_argument("command", choices=("run-h1-wd",))
    args = parser.parse_args()
    if args.command != "run-h1-wd":
        raise AssertionError("unreachable H1-WD command")

    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = run_write_delete_screen(repo_root)
    except Exception as exc:
        root = repo_root / OUTPUT_ROOT
        if root.is_dir():
            payload = {
                "schema_version": "yggdrasil.v2-r1r.p1-h1-wd.crash.v1",
                "status": "CRASH_H1_WD_NONFORMAL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": time.time(),
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
