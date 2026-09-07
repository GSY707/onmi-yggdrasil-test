"""Single-command entry point for the R1-R4 full-replay successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry import screen
from yggdrasil_v2.r1_revalidation.h1_wd_direction_geometry.contract import OUTPUT_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V2-R1R H1-WD direction-geometry v2 non-formal screen"
    )
    parser.add_argument("command", choices=("run-h1-wd-direction-geometry-v2",))
    args = parser.parse_args(argv)
    if args.command != "run-h1-wd-direction-geometry-v2":
        raise AssertionError("unreachable direction-geometry command")
    repo_root = Path(__file__).resolve().parents[1]
    result = screen.run_direction_geometry_screen(repo_root)
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
