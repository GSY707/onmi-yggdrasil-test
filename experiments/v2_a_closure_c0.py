"""Single-command entry point for V2-A Closure C0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from yggdrasil_v2.v2_a.closure_c0 import run_closure_c0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V2-A Closure C0 readiness qualification")
    parser.add_argument("command", choices=("run-v2-a-closure-c0",))
    parser.parse_args(argv)
    result = run_closure_c0(REPO_ROOT)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
