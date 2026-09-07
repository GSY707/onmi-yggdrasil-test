from __future__ import annotations

"""Read-only replay of the frozen C1 checkpoint selection rule."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import contract


SELECTION_SCHEMA = f"{contract.SCHEMA_PREFIX}.selection-audit.v1"
SELECTION_RULE = "minimize max(ERE trace NLL, CPS trace NLL); tie earliest"
_EXPECTED_UPDATES = list(range(512, 6145, 512))


def _load(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} must be a JSON object or readable JSON path") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or JSON path")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("selection checkpoint must remain inside formal_root") from exc
    return candidate


def selection_audit(primary_training: Any, formal_root: Any = None) -> dict[str, Any]:
    """Replay frozen min-max NLL selection and optionally verify all files.

    ``formal_root`` is deliberately optional: without it the function audits
    the JSON selection record only; with it, every candidate checkpoint is
    hashed in read-only mode and compared to its recorded SHA-256.
    """
    primary = _load(primary_training, "primary_training")
    candidates_raw = primary.get("candidates")
    if not isinstance(candidates_raw, list):
        raise ValueError("primary_training requires a candidates list")
    candidates: list[dict[str, Any]] = []
    seen_updates: set[int] = set()
    for index, raw in enumerate(candidates_raw):
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate {index} must be an object")
        update = raw.get("update")
        if type(update) is not int or update <= 0:
            raise ValueError(f"candidate {index} update must be a positive integer")
        if update in seen_updates:
            raise ValueError(f"duplicate candidate update: {update}")
        seen_updates.add(update)
        checkpoint = raw.get("checkpoint")
        checkpoint_sha = raw.get("checkpoint_sha256")
        if type(checkpoint) is not str or not checkpoint:
            raise ValueError(f"candidate {index} requires checkpoint path")
        if type(checkpoint_sha) is not str or len(checkpoint_sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in checkpoint_sha):
            raise ValueError(f"candidate {index} requires a hexadecimal checkpoint_sha256")
        nll = raw.get("trace_nll_by_family")
        if not isinstance(nll, Mapping):
            raise ValueError(f"candidate {index} requires trace_nll_by_family")
        if set(nll) != {"ERE", "CPS"}:
            raise ValueError(f"candidate {index} trace NLL must contain exactly ERE and CPS")
        ere, cps = _number(nll["ERE"], f"candidate {index}.ERE"), _number(nll["CPS"], f"candidate {index}.CPS")
        candidates.append(dict(raw) | {"update": update, "checkpoint_sha256": checkpoint_sha.upper(), "trace_nll_by_family": {"ERE": ere, "CPS": cps}})
    updates = [int(row["update"]) for row in candidates]
    row_checks = {
        "candidate_count": len(candidates) == 12,
        "candidate_updates": updates == _EXPECTED_UPDATES,
        "recorded_hashes": all(isinstance(row.get("checkpoint_sha256"), str) for row in candidates),
        "nll_fields": all(set(row["trace_nll_by_family"]) == {"ERE", "CPS"} for row in candidates),
        "selection_rule": primary.get("selection_rule") == SELECTION_RULE,
    }
    if not candidates:
        raise ValueError("primary_training candidates cannot be empty")
    selected = primary.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("primary_training requires selected candidate")
    selected_update = selected.get("update")
    if type(selected_update) is not int:
        raise ValueError("selected update must be an integer")
    # ``update`` is the only tie breaker after the max family NLL.  Keep the
    # candidate dict itself as the recomputed result, preserving JSON fields.
    recomputed = min(candidates, key=lambda row: (max(row["trace_nll_by_family"].values()), int(row["update"])))
    selected_canonical = dict(selected)
    if isinstance(selected_canonical.get("checkpoint_sha256"), str):
        selected_canonical["checkpoint_sha256"] = selected_canonical["checkpoint_sha256"].upper()
    if isinstance(selected_canonical.get("trace_nll_by_family"), Mapping):
        selected_canonical["trace_nll_by_family"] = {
            "ERE": _number(selected_canonical["trace_nll_by_family"].get("ERE"), "selected.ERE"),
            "CPS": _number(selected_canonical["trace_nll_by_family"].get("CPS"), "selected.CPS"),
        }
    selected_match = selected_canonical == dict(recomputed)
    row_checks["selected"] = selected_match

    root: Path | None = None
    hashes: list[dict[str, Any]] = []
    if formal_root is not None:
        root = Path(formal_root).resolve()
        if not root.is_dir():
            raise ValueError("formal_root must be an existing directory")
        for row in candidates:
            path = _inside(root, Path(str(row["checkpoint"])))
            observed: str | None = None
            exists = path.is_file()
            if exists:
                observed = _sha256(path)
            hashes.append({
                "update": int(row["update"]),
                "checkpoint": path.as_posix(),
                "exists": exists,
                "recorded_sha256": str(row["checkpoint_sha256"]),
                "observed_sha256": observed,
                "matches": bool(exists and observed == str(row["checkpoint_sha256"])),
            })
        row_checks["checkpoint_hashes"] = all(item["matches"] for item in hashes)
    else:
        row_checks["checkpoint_hashes"] = None
    # A missing formal root means file hashes are intentionally not checked;
    # ``None`` is the JSON-native not-applicable value, not a failed check.
    passed = all(value is not False for value in row_checks.values())
    report: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "candidate_count": len(candidates),
        "candidate_updates": updates,
        "recomputed_selected": recomputed,
        "recorded_selected": dict(selected),
        "selected_update": int(recomputed["update"]),
        "selection_matches": selected_match,
        "checks": row_checks,
        "checkpoint_hashes": hashes,
        "formal_root": root.as_posix() if root is not None else None,
        "passed": passed,
        "selection_failure": not passed,
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report


__all__ = ["SELECTION_SCHEMA", "SELECTION_RULE", "selection_audit"]
