from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_17_assessment import assess_a117


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Assess V2-A1.17 paired objective x initialization audit"
    )
    root.add_argument("--a113-assessment", type=Path, required=True)
    root.add_argument("--a115-assessment", type=Path, required=True)
    root.add_argument("--a116-assessment", type=Path, required=True)
    root.add_argument("--coupled-ce-dir", type=Path, action="append", required=True)
    root.add_argument("--coupled-noce-dir", type=Path, action="append")
    root.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    result = assess_a117(
        args.a113_assessment,
        args.a115_assessment,
        args.a116_assessment,
        args.coupled_ce_dir,
        args.output,
        args.coupled_noce_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
