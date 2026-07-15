from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(torch_module: Any | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    if torch_module is not None:
        record["torch"] = torch_module.__version__
        record["cuda_available"] = torch_module.cuda.is_available()
        if torch_module.cuda.is_available():
            record["cuda_device"] = torch_module.cuda.get_device_name(0)
            record["cuda_version"] = torch_module.version.cuda
    return record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

