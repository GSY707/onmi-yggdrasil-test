"""Single-command entry points for the V2-A Closure C0R successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from yggdrasil_v2.v2_a.closure_c0r import contract
from yggdrasil_v2.v2_a.closure_c0r.runner import (
    run_data_trace_qualification,
    run_readiness,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V2-A Closure C0R qualification")
    parser.add_argument(
        "command",
        choices=(
            "run-data-trace-preflight",
            "run-data-trace-qualification",
            "run-v2-a-closure-c0r",
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "run-data-trace-preflight":
        result = run_data_trace_qualification(
            REPO_ROOT,
            output_root=REPO_ROOT / contract.PREFLIGHT_OUTPUT_ROOT,
            lease_path=REPO_ROOT / contract.PREFLIGHT_LEASE_PATH,
            formal=False,
        )
    elif args.command == "run-data-trace-qualification":
        result = run_data_trace_qualification(REPO_ROOT)
    else:
        result = run_readiness(REPO_ROOT)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
