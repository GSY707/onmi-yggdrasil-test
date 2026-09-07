from __future__ import annotations

"""CLI for the single-use C1U S2 multi-bank matched K1/K8 stage."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from yggdrasil_v2.v2_a.closure_c1u_s2.runner import (
    audit_preflight,
    audit_s2,
    inspect,
    run_preflight,
    run_s2,
)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "passed": report.get("passed"),
        "status": report.get("status")
        or report.get("result", {}).get("status"),
        "identity": report.get("identity")
        or report.get("result", {}).get("identity"),
        "authorizes": report.get("authorizes")
        or report.get("result", {}).get("authorizes"),
        "formal_optimizer_steps": report.get("formal_optimizer_steps")
        or report.get("result", {}).get("formal_optimizer_steps"),
        "disposable_optimizer_steps": report.get("disposable_optimizer_steps")
        or report.get("result", {}).get("disposable_optimizer_steps"),
        "model_writes": report.get("model_writes")
        or report.get("result", {}).get("model_writes"),
        "result_sha256": report.get("result_sha256"),
    }
    seal = report.get("seal_replay") or report.get("seal")
    if isinstance(seal, Mapping):
        result["seal_replay"] = {
            "passed": seal.get("passed"),
            "seal_sha256": seal.get("seal_sha256"),
            "entries": seal.get("entries"),
        }
    if report.get("reason"):
        result["reason"] = report["reason"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("inspect", "run-preflight", "audit-preflight", "run-s2", "audit-s2"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        report = inspect(args.repo_root)
    elif args.command == "run-preflight":
        report = run_preflight(args.repo_root, device=args.device)
    elif args.command == "audit-preflight":
        report = audit_preflight(args.repo_root)
    elif args.command == "run-s2":
        report = run_s2(args.repo_root, device=args.device)
    else:
        report = audit_s2(args.repo_root)
    print(json.dumps(_summary(report), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
