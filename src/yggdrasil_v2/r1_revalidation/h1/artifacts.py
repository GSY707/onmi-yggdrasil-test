from __future__ import annotations

"""Single-use artifact, source-identity, and evidence-seal utilities for H1."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


SCHEMA_PREFIX = "yggdrasil.v2-r1r.p1-h1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "evidence-seal.json"
    }


def seal(root: Path) -> str:
    path = root / "evidence-seal.json"
    write_json(
        path,
        {
            "schema_version": f"{SCHEMA_PREFIX}.evidence-seal.v1",
            "files": tree_hashes(root),
        },
    )
    return sha256(path)


def verify_seal(root: Path) -> bool:
    path = root / "evidence-seal.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    return payload.get("files") == tree_hashes(root)


def source_hashes(base: Path, source_files: tuple[Path, ...]) -> dict[str, str]:
    return {relative.as_posix(): sha256(base / relative) for relative in source_files}


def source_identity(base: Path, source_files: tuple[Path, ...]) -> str:
    return hashlib.sha256(canonical(source_hashes(base, source_files))).hexdigest().upper()


def copy_source_snapshot(
    repo_root: Path,
    artifact_root: Path,
    source_files: tuple[Path, ...],
) -> None:
    snapshot = artifact_root / "source_snapshot"
    for relative in source_files:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo_root / relative, destination)


def source_snapshot_identity(root: Path, source_files: tuple[Path, ...]) -> str:
    return source_identity(root / "source_snapshot", source_files)


def _status_path(line: str) -> str:
    payload = line[3:] if len(line) >= 4 else line
    if " -> " in payload:
        payload = payload.rsplit(" -> ", 1)[1]
    return payload.strip('"').replace("\\", "/")


def git_identity(repo_root: Path, excluded: tuple[Path, ...]) -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    head = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain=v1", "-uall")
    if head.returncode or branch.returncode or status.returncode:
        return {
            "passed": False,
            "head_returncode": head.returncode,
            "branch_returncode": branch.returncode,
            "status_returncode": status.returncode,
            "stderr": (head.stderr + branch.stderr + status.stderr).strip(),
        }
    prefixes = tuple(path.as_posix().rstrip("/") for path in excluded)
    entries = []
    for line in status.stdout.splitlines():
        path = _status_path(line)
        if any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes):
            continue
        entries.append(line)
    payload = "\n".join(entries)
    return {
        "passed": True,
        "head": head.stdout.strip(),
        "branch": branch.stdout.strip(),
        "dirty": bool(entries),
        "status_entry_count": len(entries),
        "status_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest().upper(),
        "excluded_runtime_paths": [path.as_posix() for path in excluded],
    }


StageAction = Callable[[Path, str, dict[str, Any]], dict[str, Any]]
PostActionAudit = Callable[[], dict[str, Any]]


def _controlled_refusal(stage: str, root: Path, reason: str) -> int:
    print(
        json.dumps(
            {
                "stage": stage,
                "status": "REFUSE_P1_H1_SINGLE_USE",
                "passed": False,
                "root": root.as_posix(),
                "reason": reason,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 2


def run_stage(
    repo_root: Path,
    relative_root: Path,
    stage: str,
    source_files: tuple[Path, ...],
    action: StageAction,
    excluded_git_paths: tuple[Path, ...],
    post_action_audit: PostActionAudit,
) -> int:
    root = repo_root / relative_root
    if root.exists():
        return _controlled_refusal(stage, relative_root, "fixed root already exists")

    identity = source_identity(repo_root, source_files)
    git_before = git_identity(repo_root, excluded_git_paths)
    started_at = datetime.now(UTC).isoformat()
    wall_started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {}
    exit_code = 1
    snapshot_identity: str | None = None
    source_after: str | None = None
    git_after: dict[str, Any] = {"passed": False, "status": "not-collected"}
    try:
        write_json(
            root / "attempt.json",
            {
                "schema_version": f"{SCHEMA_PREFIX}.attempt.v1",
                "stage": stage,
                "started_at": started_at,
                "source_identity": identity,
                "git_identity": git_before,
                "pid": __import__("os").getpid(),
                "parent_pid": __import__("os").getppid(),
                "argv": sys.argv,
            },
        )
        copy_source_snapshot(repo_root, root, source_files)
        snapshot_identity = source_snapshot_identity(root, source_files)
        if snapshot_identity != identity:
            raise RuntimeError("H1 source snapshot identity mismatch")
        result = action(
            root,
            identity,
            {"git_identity_start": git_before, "snapshot_identity": snapshot_identity},
        )
        runtime_after = post_action_audit()
        source_after = source_identity(repo_root, source_files)
        git_after = git_identity(repo_root, excluded_git_paths)
        runner_integrity = {
            "source_identity_start": identity,
            "source_identity_after": source_after,
            "source_snapshot_identity": snapshot_identity,
            "source_identity_stable": identity == source_after == snapshot_identity,
            "git_identity_start": git_before,
            "git_identity_after": git_after,
            "git_identity_stable": (
                git_before.get("passed") is True
                and git_after.get("passed") is True
                and git_before.get("head") == git_after.get("head")
                and git_before.get("branch") == git_after.get("branch")
                and git_before.get("status_sha256") == git_after.get("status_sha256")
            ),
            "post_action_audit": runtime_after,
        }
        runner_integrity["passed"] = bool(
            runner_integrity["source_identity_stable"]
            and runner_integrity["git_identity_stable"]
            and runtime_after.get("passed") is True
        )
        result["runner_integrity"] = runner_integrity
        if result.get("passed") is True and not runner_integrity["passed"]:
            result["pre_integrity_status"] = result.get("status")
            result["status"] = "FAIL_P1_H1_STAGE_INTEGRITY"
            result["passed"] = False
        write_json(root / "result.json", result)
        exit_code = 0 if result.get("passed") is True else 1
    except Exception as exc:  # noqa: BLE001
        try:
            source_after = source_identity(repo_root, source_files)
            git_after = git_identity(repo_root, excluded_git_paths)
        except Exception:  # noqa: BLE001
            pass
        result = {
            "schema_version": f"{SCHEMA_PREFIX}.failure.v1",
            "stage": stage,
            "status": "CRASH_P1_H1",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_identity": identity,
            "source_identity_after": source_after,
            "source_snapshot_identity": snapshot_identity,
        }
        write_json(root / "failure.json", result)
        write_json(root / "result.json", result)

    metadata = {
        "schema_version": f"{SCHEMA_PREFIX}.run-metadata.v1",
        "stage": stage,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - wall_started,
        "platform": platform.platform(),
        "python": sys.version,
        "source_identity_start": identity,
        "source_identity_after": source_after,
        "source_snapshot_identity": snapshot_identity,
        "git_identity_start": git_before,
        "git_identity_after": git_after,
        "exit_code": exit_code,
    }
    write_json(root / "run-metadata.json", metadata)
    seal_hash = seal(root)
    seal_verified = verify_seal(root)
    if not seal_verified:
        result["pre_integrity_status"] = result.get("status")
        result["status"] = "FAIL_P1_H1_EVIDENCE_SEAL"
        result["passed"] = False
        metadata["exit_code"] = 1
        metadata["seal_recovery_required"] = True
        write_json(root / "result.json", result)
        write_json(root / "run-metadata.json", metadata)
        exit_code = 1
        seal_hash = seal(root)
        seal_verified = verify_seal(root)
    print(
        json.dumps(
            {
                "stage": stage,
                "passed": exit_code == 0,
                "root": relative_root.as_posix(),
                "evidence_seal_sha256": seal_hash,
                "evidence_seal_verified": seal_verified,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return exit_code


__all__ = [
    "canonical",
    "copy_source_snapshot",
    "git_identity",
    "run_stage",
    "seal",
    "sha256",
    "source_hashes",
    "source_identity",
    "source_snapshot_identity",
    "tree_hashes",
    "verify_seal",
    "write_json",
]
