from __future__ import annotations

"""Fixed CLI for C1S S0 zero-update qualification."""

import argparse
import json
from pathlib import Path

from yggdrasil_v2.v2_a.closure_c1s.runner import run_s0_preflight


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-s0-preflight",))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    if args.command != "run-s0-preflight":
        raise AssertionError(args.command)
    result = run_s0_preflight(repo_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
