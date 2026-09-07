from __future__ import annotations

"""Fixed CLI for the C1S semantic-routed-workspace successor chain.

Commands are intentionally separate.  A PASS from one invocation must be
inspected before the next command is launched; this CLI never auto-chains a
preflight or experimental stage.
"""

import argparse
import json
from pathlib import Path

from yggdrasil_v2.v2_a.closure_c1s_successor.runner import (
    run_preflight,
    run_s1,
    run_s2,
    run_s3,
)


COMMANDS = (
    "preflight-s1",
    "run-s1",
    "preflight-s2",
    "run-s2",
    "preflight-s3",
    "run-s3",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    if args.command == "preflight-s1":
        result = run_preflight(repo_root, "S1", device=args.device)
    elif args.command == "run-s1":
        result = run_s1(repo_root, device=args.device)
    elif args.command == "preflight-s2":
        result = run_preflight(repo_root, "S2", device=args.device)
    elif args.command == "run-s2":
        result = run_s2(repo_root, device=args.device)
    elif args.command == "preflight-s3":
        result = run_preflight(repo_root, "S3", device=args.device)
    elif args.command == "run-s3":
        result = run_s3(repo_root, device=args.device)
    else:  # pragma: no cover - argparse freezes the set
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
