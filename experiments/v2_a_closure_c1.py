"""Single-command entry points for V2-A Closure C1 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from yggdrasil_v2.v2_a.closure_c1.cache_runner import (
    run_cache_preflight,
    run_cache_qualification,
)
from yggdrasil_v2.v2_a.closure_c1.runner import (
    run_c1_preflight,
    run_c1_single_seed,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V2-A Closure C1 qualification")
    parser.add_argument(
        "command",
        choices=(
            "run-cache-preflight",
            "run-cache-qualification",
            "run-c1-preflight",
            "run-c1-single-seed",
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "run-cache-preflight":
        result = run_cache_preflight(REPO_ROOT, device=args.device)
    elif args.command == "run-cache-qualification":
        result = run_cache_qualification(REPO_ROOT, device=args.device)
    elif args.command == "run-c1-preflight":
        result = run_c1_preflight(REPO_ROOT, device=args.device)
    else:
        result = run_c1_single_seed(REPO_ROOT, device=args.device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
