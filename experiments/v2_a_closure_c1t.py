from __future__ import annotations

"""C1T inspection and explicit single-use zero-training S0 commands."""

import argparse
import json
import os
from pathlib import Path

# Must be set before torch initialises CUDA in imported stage modules.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from yggdrasil_v2.v2_a.closure_c1t.runner import inspect_successor
from yggdrasil_v2.v2_a.closure_c1t.s0 import (
    audit_s0_preflight,
    audit_s0_root,
    run_s0,
    run_s0_preflight,
)
from yggdrasil_v2.v2_a.closure_c1t.s1 import (
    audit_s0_predecessor,
    audit_s1_preflight,
    audit_s1_root,
    run_s1,
    run_s1_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "inspect",
            "preflight-s0",
            "audit-preflight-s0",
            "run-s0",
            "audit-s0",
            "audit-s0-predecessor",
            "preflight-s1",
            "audit-preflight-s1",
            "run-s1",
            "audit-s1",
        ),
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    if args.command == "inspect":
        result = inspect_successor(repo_root)
    elif args.command == "preflight-s0":
        result = run_s0_preflight(repo_root, device=args.device)
    elif args.command == "audit-preflight-s0":
        result = audit_s0_preflight(repo_root)
    elif args.command == "run-s0":
        result = run_s0(repo_root, device=args.device)
    elif args.command == "audit-s0":
        result = audit_s0_root(repo_root)
    elif args.command == "audit-s0-predecessor":
        result = audit_s0_predecessor(repo_root)
    elif args.command == "preflight-s1":
        result = run_s1_preflight(repo_root, device=args.device)
    elif args.command == "audit-preflight-s1":
        result = audit_s1_preflight(repo_root)
    elif args.command == "run-s1":
        result = run_s1(repo_root, device=args.device)
    elif args.command == "audit-s1":
        result = audit_s1_root(repo_root)
    else:  # pragma: no cover - argparse freezes choices
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
