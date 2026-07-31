from __future__ import annotations

"""P0-D data generation and audit entry points for V2-R1R.

This module intentionally stops at the data/simulator boundary.  It does not
import a model, cache hidden states, or start a training process.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from yggdrasil_v2.r1_revalidation.audit import (  # noqa: E402
    audit_p0,
    environment_record,
    make_assessment,
    sha256_file,
    utc_now,
)
from yggdrasil_v2.r1_revalidation.cps import (  # noqa: E402
    CPS_SPLITS,
    generate_cps_causal_pairs,
    generate_cps_split,
)
from yggdrasil_v2.r1_revalidation.ere import (  # noqa: E402
    ERE_SPLITS,
    generate_ere_causal_pairs,
    generate_ere_split,
)
from yggdrasil_v2.r1_revalidation.schema import GENERATOR_VERSION  # noqa: E402


DESIGN_DOC = REPO_ROOT / "docs" / "v2-r1-revalidation-task-design.md"


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_output_path(path: Path, *, smoke: bool, overwrite_smoke: bool) -> Path:
    output = path.expanduser().resolve()
    if not output.exists():
        return output
    if not smoke or not overwrite_smoke:
        raise FileExistsError(
            f"refusing to overwrite existing output: {output}; use --smoke --overwrite-smoke only for smoke artifacts"
        )
    if REPO_ROOT not in output.parents or "p0-smoke" not in output.name:
        raise ValueError(f"--overwrite-smoke is restricted to an explicit p0-smoke directory under {REPO_ROOT}")
    shutil.rmtree(output)
    return output


def _annotate_records(
    records: list[dict[str, Any]],
    *,
    started_at: str,
    finished_at: str,
    design_doc_sha256: str,
    command: str,
    environment: Mapping[str, Any],
) -> None:
    for record in records:
        provenance = record.setdefault("provenance", {})
        provenance.update(
            {
                "input_files_sha256": {"design_doc": design_doc_sha256},
                "started_at": started_at,
                "finished_at": finished_at,
                "command": command,
                "environment": dict(environment),
            }
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_json_dump(row) + "\n")


def _write_manifest_data(root: Path, datasets: Mapping[str, Mapping[str, list[dict[str, Any]]]]) -> dict[str, str]:
    input_files: dict[str, str] = {}
    for family, split_rows in datasets.items():
        for split, rows in split_rows.items():
            relative = Path("data") / family / f"{split}.jsonl"
            path = root / relative
            _write_jsonl(path, rows)
            input_files[relative.as_posix()] = sha256_file(path)
    return dict(sorted(input_files.items()))


def _split_seed(base_seed: int, family_index: int, split_index: int) -> int:
    return base_seed + family_index * 1_000_003 + split_index * 100_003


def _generate_dataset(
    *,
    family: str,
    train_count: int,
    split_count: int,
    pair_record_count: int,
    seed: int,
    design_doc_sha256: str,
    command: str,
    environment: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    splits = ERE_SPLITS if family == "ere" else CPS_SPLITS
    datasets: dict[str, list[dict[str, Any]]] = {}
    seen_semantic: set[str] = set()
    seen_surface: set[str] = set()
    non_pair_splits = [split for split in splits if split != "causal_pairs"]
    for split_index, split in enumerate(non_pair_splits):
        count = train_count if split == "train" else split_count
        split_seed = _split_seed(seed, 0 if family == "ere" else 1, split_index)
        if family == "ere":
            records = generate_ere_split(
                split=split,
                count=count,
                seed=split_seed,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
                seen_semantic=seen_semantic,
                seen_surface=seen_surface,
            )
        else:
            records = generate_cps_split(
                split=split,
                count=count,
                seed=split_seed,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
                seen_semantic=seen_semantic,
                seen_surface=seen_surface,
            )
        datasets[split] = records
        print(f"generated {family}/{split}: {len(records)}", flush=True)
    pair_count = pair_record_count // 2
    pair_seed = _split_seed(seed, 0 if family == "ere" else 1, len(non_pair_splits))
    if family == "ere":
        datasets["causal_pairs"] = generate_ere_causal_pairs(
            pair_count=pair_count,
            seed=pair_seed,
            design_doc_sha256=design_doc_sha256,
            command=command,
            environment=environment,
            seen_semantic=seen_semantic,
            seen_surface=seen_surface,
        )
    else:
        datasets["causal_pairs"] = generate_cps_causal_pairs(
            pair_count=pair_count,
            seed=pair_seed,
            design_doc_sha256=design_doc_sha256,
            command=command,
            environment=environment,
            seen_semantic=seen_semantic,
            seen_surface=seen_surface,
        )
    print(f"generated {family}/causal_pairs: {len(datasets['causal_pairs'])}", flush=True)
    return datasets


def generate_p0(args: argparse.Namespace) -> int:
    if not DESIGN_DOC.exists():
        raise FileNotFoundError(DESIGN_DOC)
    if args.overwrite_smoke and not args.smoke:
        raise ValueError("--overwrite-smoke is valid only together with --smoke")
    output = _safe_output_path(Path(args.output), smoke=bool(args.smoke), overwrite_smoke=bool(args.overwrite_smoke))
    design_doc_sha256 = sha256_file(DESIGN_DOC)
    command = " ".join(sys.argv)
    environment = environment_record()
    started_at = utc_now()
    if args.smoke:
        train_count, split_count, pair_count = 64, 32, 32
        mode = "smoke"
    else:
        train_count, split_count, pair_count = 4096, 512, 512
        mode = "formal"
    datasets = {
        "ere": _generate_dataset(
            family="ere",
            train_count=train_count,
            split_count=split_count,
            pair_record_count=pair_count,
            seed=int(args.seed),
            design_doc_sha256=design_doc_sha256,
            command=command,
            environment=environment,
        ),
        "cps": _generate_dataset(
            family="cps",
            train_count=train_count,
            split_count=split_count,
            pair_record_count=pair_count,
            seed=int(args.seed) + 17_000_000,
            design_doc_sha256=design_doc_sha256,
            command=command,
            environment=environment,
        ),
    }
    finished_at = utc_now()
    for split_rows in datasets.values():
        for records in split_rows.values():
            _annotate_records(
                records,
                started_at=started_at,
                finished_at=finished_at,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
            )
    input_files = _write_manifest_data(output, datasets)
    manifest = {
        "schema_version": "yggdrasil.v2-r1r.p0.manifest.v1",
        "evidence_level": mode,
        "mode": mode,
        "generator_version": GENERATOR_VERSION,
        "seed": int(args.seed),
        "design_doc": str(DESIGN_DOC.relative_to(REPO_ROOT)).replace("\\", "/"),
        "design_doc_sha256": design_doc_sha256,
        "input_files": input_files,
        "environment": environment,
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "counts": {
            family: {split: len(rows) for split, rows in split_rows.items()}
            for family, split_rows in datasets.items()
        },
        "data_contract": "P0-D only; no model/cache/train artifact",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(_json_dump(manifest) + "\n", encoding="utf-8")
    print(_json_dump({"output": str(output), "mode": mode, "counts": manifest["counts"], "design_doc_sha256": design_doc_sha256}))
    return 0


def audit_command(args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root
    if output != root:
        output.mkdir(parents=True, exist_ok=True)
    report = audit_p0(root, design_doc=DESIGN_DOC, command=" ".join(sys.argv))
    for filename in ("audit.json", "heuristics.json", "p0-assessment.json"):
        target = output / filename
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing audit artifact: {target}")
    (output / "audit.json").write_text(_json_dump(report) + "\n", encoding="utf-8")
    (output / "heuristics.json").write_text(_json_dump(report["heuristics"]) + "\n", encoding="utf-8")
    assessment = make_assessment(report)
    (output / "p0-assessment.json").write_text(_json_dump(assessment) + "\n", encoding="utf-8")
    print(_json_dump({"input": str(root), "output": str(output), "passed": report["passed"], "gate_results": report["gate_results"], "scale": report["scale"]}))
    return 0 if report["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V2-R1R P0-D generator/auditor")
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    generate = subparsers.add_parser("generate-p0")
    generate.add_argument("--seed", required=True, type=int)
    generate.add_argument("--output", required=True)
    generate.add_argument("--smoke", action="store_true", help="generate the non-formal 64/32 smoke scale")
    generate.add_argument("--overwrite-smoke", action="store_true")
    generate.set_defaults(handler=generate_p0)
    audit = subparsers.add_parser("audit-p0")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output")
    audit.set_defaults(handler=audit_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"P0-D command failed: {exc}", file=sys.stderr)
        raise
