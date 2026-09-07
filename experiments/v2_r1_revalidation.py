"""Single-use V2-R1R P1-H1 formal entry."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (
            candidate
            / "docs/v2-r1r-p1-h1-mixed-core-design.md"
        ).is_file():
            return candidate
    raise RuntimeError("repository root not found")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _formal_run(root: Path) -> int:
    sys.path.insert(0, str(root / "src"))
    from yggdrasil_v2.r1_revalidation.h1.contract import (
        ATTEMPT_ID,
        ROOTS,
        TRANSPORT_ROOT,
    )
    from yggdrasil_v2.r1_revalidation.h1.runner import (
        launch_readiness_audit,
        run_all,
    )

    readiness = launch_readiness_audit(root)
    if not readiness["passed"]:
        print(
            json.dumps(
                {
                    "status": "REFUSE_P1_H1_BEFORE_MUTATION",
                    "passed": False,
                    "reason": "H1 contract, history, process, CLI, or fixed paths are not ready",
                    "details": readiness,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 2

    existing_roots = [
        relative.as_posix() for relative in ROOTS.values() if (root / relative).exists()
    ]
    transport = root / TRANSPORT_ROOT
    if existing_roots or transport.exists():
        print(
            json.dumps(
                {
                    "status": "REFUSE_P1_H1_SINGLE_USE",
                    "passed": False,
                    "existing_roots": existing_roots,
                    "transport_exists": transport.exists(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 2

    transport.mkdir(parents=True, exist_ok=False)
    stdout_path = (transport / "stdout.log").resolve()
    stderr_path = (transport / "stderr.log").resolve()
    attempt_id = ATTEMPT_ID
    launch = {
        "schema_version": "yggdrasil.v2-r1r.p1-h1.launch.v1",
        "attempt_id": attempt_id,
        "started_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "cwd": root.resolve().as_posix(),
        "argv": sys.argv,
        "stdout": stdout_path.as_posix(),
        "stderr": stderr_path.as_posix(),
    }
    _write_json(transport / "launch.json", launch)
    os.environ["YGGDRASIL_H1_ATTEMPT_ID"] = attempt_id
    os.environ["YGGDRASIL_H1_TRANSPORT_ROOT"] = transport.resolve().as_posix()
    exit_code = 1
    with stdout_path.open("x", encoding="utf-8", buffering=1) as stdout_handle:
        with stderr_path.open("x", encoding="utf-8", buffering=1) as stderr_handle:
            with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(
                stderr_handle
            ):
                try:
                    exit_code = run_all(root)
                except BaseException as exc:  # noqa: BLE001
                    print(
                        json.dumps(
                            {
                                "status": "CRASH_P1_H1_LAUNCHER",
                                "passed": False,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    exit_code = 1
    _write_json(
        transport / "completion.json",
        {
            "schema_version": "yggdrasil.v2-r1r.p1-h1.completion.v1",
            "attempt_id": attempt_id,
            "finished_at": datetime.now(UTC).isoformat(),
            "exit_code": exit_code,
        },
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v2_r1_revalidation.py",
        description="V2-R1R P1-H1 single-use mixed-core development qualification",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run-p1-h1")
    args = parser.parse_args(argv)
    if args.command != "run-p1-h1":
        parser.error("unsupported command")
    return _formal_run(_repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
