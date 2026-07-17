from __future__ import annotations

import argparse
import json
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_18b_assessment import assess_a118b


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Assess V2-A1.18B per-step global trajectory-state scaffold"
    )
    root.add_argument("--qaux-overfit", type=Path, required=True)
    root.add_argument("--final-saux-overfit", type=Path, required=True)
    root.add_argument("--tsaux-overfit", type=Path, required=True)
    root.add_argument("--final-saux-dir", type=Path, action="append", required=True)
    root.add_argument("--tsaux-paired-dir", type=Path, action="append", required=True)
    root.add_argument("--tsaux-fresh-dir", type=Path, action="append", required=True)
    root.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    result = assess_a118b(
        args.qaux_overfit,
        args.final_saux_overfit,
        args.tsaux_overfit,
        args.final_saux_dir,
        args.tsaux_paired_dir,
        args.tsaux_fresh_dir,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
