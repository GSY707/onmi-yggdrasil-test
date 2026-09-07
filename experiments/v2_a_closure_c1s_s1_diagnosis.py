from __future__ import annotations

"""CLI for the single-use C1S S1 read-only failure diagnosis."""

import argparse
import json
from pathlib import Path

from yggdrasil_v2.v2_a.closure_c1s_s1_diagnosis.runner import (
    run_diagnosis,
    run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run-diagnosis"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight(args.repo_root, device=args.device)
    else:
        result = run_diagnosis(args.repo_root, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
