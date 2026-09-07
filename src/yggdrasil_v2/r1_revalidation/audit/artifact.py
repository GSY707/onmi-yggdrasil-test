from __future__ import annotations

"""Fail-closed artifact, fixed-profile, schema, seal, and tokenizer checks."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Any


EXPECTED_MANIFEST_SCHEMA = "yggdrasil.v2-r1r.p0d-v9.r0b-invariant.manifest.v1"
EXPECTED_RECORD_SCHEMA = "yggdrasil.v2-r1r.p0d-v9.r0b-invariant.record.v1"
EXPECTED_SEAL_SCHEMA = "yggdrasil.v2-r1r.p0d-v9.r0b-invariant.input-seal.v1"
EXPECTED_CONTRACT = "r1r-p0d-v9-r0b-invariant"
EXPECTED_TOKENIZER = {
    "model_id": "Qwen/Qwen3.5-2B",
    "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "tokenizer_class": "Qwen2TokenizerFast",
    "add_special_tokens": False,
    "max_source_tokens": 1024,
}
EXPECTED_MODEL_VIEW_FIELDS = ["example_id", "source_text", "reasoning_budget", "valid_choice_mask"]
FORBIDDEN_MODEL_VIEW_KEYS = {
    "answer", "ast", "teacher", "claim", "span", "role", "state", "validity",
    "certificate", "derivation", "candidate_role", "witness", "provenance",
}
EXPECTED_STREAMS = {"semantic": 81101, "surface": 81103, "labels": 81107, "claims": 81109, "pairs": 81131}
EXPECTED_CLAIM_KINDS = {
    "ere": ["attribute_value", "relation_exists", "condition_truth"],
    "cps": [
        "prefix_legality", "resource_value", "cost_value", "fact_truth",
        "goal_status", "final_constraint_status",
    ],
}
DATA_FILES = ("records.jsonl", "causal-pairs.json", "language-pairs.json", "composition-controls.json")
TOP_LEVEL = {"manifest.json", *DATA_FILES, "input-seal.json", "source_snapshot"}
FIXED_COUNTS = {
    "records": 11,
    "ere_records": 7,
    "cps_records": 4,
    "claims": 18,
    "causal_pairs": 2,
    "language_pairs": 2,
}
FIXED_ROLES = {"ere_core": 5, "cps_rich": 1, "cps_none": 1, "causal_member": 4}

_REQUIRED_SNAPSHOT_BASENAMES = [
    "pyproject.toml", "uv.lock", "v2-r1r-p0d-v9-r0b-invariant-design.md",
    "v2-r1r-p0d-v9-r0b-invariant-execution-command.md", "v2_r1_revalidation.py",
    "__init__.py", "__init__.py", "simulator.py", "__init__.py", "artifact.py",
    "replay.py", "language.py", "pairs.py", "provenance.py", "structure.py", "invariant.py",
]
REQUIRED_SNAPSHOT_BASENAMES = Counter(_REQUIRED_SNAPSHOT_BASENAMES)
EXPECTED_SNAPSHOT_SUFFIXES = Counter({".py": 25, ".json": 3, ".md": 2, ".toml": 1, ".lock": 1})

MANIFEST_KEYS = {
    "schema_version", "contract_version", "qualification_mode", "counts", "data_files_sha256",
    "source_snapshot", "source_snapshot_set_sha256", "model_view_fields",
    "forbidden_model_view_keys", "tokenizer", "runtime", "named_streams", "required_claim_kinds",
}
RECORD_KEYS = {
    "schema_version", "example_id", "family", "split", "source_text", "reasoning_budget",
    "valid_choice_mask", "label_mapping", "answer_semantic", "answer_index",
    "teacher_output_sha256", "program_ast", "claims", "fold_group", "structure_certificate",
    "qwen_token_count", "model_view", "semantic_fingerprint", "surface_fingerprint",
}
CLAIM_KEYS = {"claim_id", "pair_id", "pair_role", "kind", "label", "predicate", "text"}
CAUSAL_KEYS = {
    "pair_id", "family", "base_id", "flip_id", "changed_path", "from", "to",
    "mutation_family", "expected_answer_flip",
}
LANGUAGE_KEYS = {
    "pair_id", "family", "record_id", "fold_group", "train_variant", "ood_variant",
    "train_text", "ood_text", "train_qwen_token_count", "ood_qwen_token_count",
    "semantic_fingerprint", "train_surface_fingerprint", "ood_surface_fingerprint",
}
COMPOSITION_KEYS = {
    "ere_ordinary_provenance_ids", "ere_composition_provenance_ids",
    "ere_required_component_ops", "cps_record_id", "cps_required_witnesses",
    "none_pair", "none_ratio",
}
CERTIFICATE_KEYS = {
    "ere_core": {"role", "necessary_event_indices", "ablation_answers", "provenance", "composition_signature"},
    "cps_rich": {
        "role", "pstar_index", "valid_suboptimal_indices", "invalid_indices",
        "expected_failure_reasons", "invalid_length_relations",
    },
    "cps_none": {"role", "skeleton_partner", "changed_path", "from", "to", "expected_valid_candidate_count"},
    "causal_member": {"role"},
}
_HEX64 = re.compile(r"[0-9A-F]{64}")


def _metric(value: Any, numerator: int, denominator: int, passed: bool, failures: list[str]) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": int(numerator),
        "denominator": int(denominator),
        "passed": bool(passed),
        "failures": sorted(set(str(item) for item in failures)),
    }


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("records.jsonl contains an empty row")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        value = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
        if not isinstance(value, dict):
            raise TypeError(f"record row {line_number} is not an object")
        rows.append(value)
    return rows


def _safe_relative(root: Path, relative: str) -> Path | None:
    try:
        candidate = (root / relative).resolve()
        base = root.resolve()
    except (OSError, RuntimeError):
        return None
    if candidate != base and base in candidate.parents:
        return candidate
    return None


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _schema_exact(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    causal_pairs: list[dict[str, Any]],
    language_pairs: list[dict[str, Any]],
    composition: dict[str, Any],
) -> bool:
    if set(manifest) != MANIFEST_KEYS:
        return False
    if set(manifest.get("counts", {})) != set(FIXED_COUNTS):
        return False
    if set(manifest.get("data_files_sha256", {})) != set(DATA_FILES):
        return False
    if set(manifest.get("source_snapshot", {})) != {"files"}:
        return False
    if set(manifest.get("runtime", {})) != {"python", "implementation", "platform"}:
        return False
    if set(manifest.get("tokenizer", {})) != {
        *EXPECTED_TOKENIZER, "actual_class", "is_fast", "transformers_version", "tokenizers_version"
    }:
        return False
    for row in records:
        if set(row) != RECORD_KEYS or row.get("schema_version") != EXPECTED_RECORD_SCHEMA:
            return False
        if not _nonempty(row.get("example_id")) or row.get("family") not in {"ere", "cps"}:
            return False
        if not _nonempty(row.get("split")) or not _nonempty(row.get("source_text")) or not _nonempty(row.get("fold_group")):
            return False
        if isinstance(row.get("reasoning_budget"), bool) or not isinstance(row.get("reasoning_budget"), int) or row["reasoning_budget"] < 0:
            return False
        if not isinstance(row.get("valid_choice_mask"), list) or not all(type(item) is bool for item in row["valid_choice_mask"]):
            return False
        if not isinstance(row.get("label_mapping"), dict) or type(row.get("answer_index")) is not int:
            return False
        if not _hex64(row.get("teacher_output_sha256")) or not _hex64(row.get("semantic_fingerprint")) or not _hex64(row.get("surface_fingerprint")):
            return False
        if not isinstance(row.get("program_ast"), dict) or not isinstance(row.get("claims"), list):
            return False
        if type(row.get("qwen_token_count")) is not int or row["qwen_token_count"] < 0:
            return False
        if set(row.get("model_view", {})) != set(EXPECTED_MODEL_VIEW_FIELDS):
            return False
        certificate = row.get("structure_certificate")
        if not isinstance(certificate, dict) or certificate.get("role") not in CERTIFICATE_KEYS:
            return False
        if set(certificate) != CERTIFICATE_KEYS[certificate["role"]]:
            return False
        for claim in row["claims"]:
            if not isinstance(claim, dict) or set(claim) != CLAIM_KEYS:
                return False
            if not all(_nonempty(claim.get(key)) for key in ("claim_id", "pair_id", "pair_role", "kind", "text")):
                return False
            if type(claim.get("label")) is not bool or not isinstance(claim.get("predicate"), dict):
                return False
    for pair in causal_pairs:
        if not isinstance(pair, dict) or set(pair) != CAUSAL_KEYS:
            return False
        if not all(_nonempty(pair.get(key)) for key in ("pair_id", "family", "base_id", "flip_id", "changed_path", "mutation_family")):
            return False
        if not isinstance(pair.get("expected_answer_flip"), list) or len(pair["expected_answer_flip"]) != 2:
            return False
    for pair in language_pairs:
        if not isinstance(pair, dict) or set(pair) != LANGUAGE_KEYS:
            return False
        if not all(_nonempty(pair.get(key)) for key in (
            "pair_id", "family", "record_id", "fold_group", "train_variant", "ood_variant", "train_text", "ood_text"
        )):
            return False
        if not all(type(pair.get(key)) is int and pair[key] >= 0 for key in ("train_qwen_token_count", "ood_qwen_token_count")):
            return False
        if not all(_hex64(pair.get(key)) for key in ("semantic_fingerprint", "train_surface_fingerprint", "ood_surface_fingerprint")):
            return False
    if set(composition) != COMPOSITION_KEYS:
        return False
    if not all(isinstance(composition.get(key), list) for key in COMPOSITION_KEYS - {"cps_record_id", "none_ratio"}):
        return False
    return _nonempty(composition.get("cps_record_id")) and set(composition.get("none_ratio", {})) == {"numerator", "denominator"}


_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}


def _load_pinned_tokenizer() -> tuple[Any | None, list[str]]:
    failures: list[str] = []
    key = (EXPECTED_TOKENIZER["model_id"], EXPECTED_TOKENIZER["revision"])
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key], failures
    snapshot = (
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.5-2B"
        / "snapshots" / EXPECTED_TOKENIZER["revision"]
    )
    try:
        from transformers import AutoTokenizer

        if not snapshot.is_dir():
            raise FileNotFoundError(str(snapshot))
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
        if type(tokenizer).__name__ != EXPECTED_TOKENIZER["tokenizer_class"]:
            raise TypeError(type(tokenizer).__name__)
        _TOKENIZER_CACHE[key] = tokenizer
        return tokenizer, failures
    except Exception as exc:
        failures.append(f"tokenizer:{type(exc).__name__}:{exc}")
        return None, failures


_TOKEN_COUNT_CACHE: dict[str, int] = {}


def _token_count(tokenizer: Any, text: str) -> int:
    if text not in _TOKEN_COUNT_CACHE:
        _TOKEN_COUNT_CACHE[text] = len(tokenizer.encode(text, add_special_tokens=False))
    return _TOKEN_COUNT_CACHE[text]


def _empty_bundle(root: Path, failures: list[str]) -> dict[str, Any]:
    return {
        "root": root,
        "manifest": {},
        "records": [],
        "causal_pairs": [],
        "language_pairs": [],
        "composition": {},
        "tokenizer": None,
        "g02": {},
        "errors": list(failures),
    }


def load_artifact(root: str | Path) -> dict[str, Any]:
    """Read a bundle without modifying it and calculate all G02 raw metrics."""

    bundle = Path(root).resolve()
    failures: list[str] = []
    if not bundle.is_dir():
        return _empty_bundle(bundle, ["bundle:missing-directory"])
    try:
        actual_top = {path.name for path in bundle.iterdir()}
    except Exception as exc:
        return _empty_bundle(bundle, [f"bundle:{type(exc).__name__}:{exc}"])
    top_exact = actual_top == TOP_LEVEL
    if not top_exact:
        failures.append("input-tree:top-level")

    try:
        manifest = _read_json(bundle / "manifest.json")
        records = _read_records(bundle / "records.jsonl")
        causal_pairs = _read_json(bundle / "causal-pairs.json")
        language_pairs = _read_json(bundle / "language-pairs.json")
        composition = _read_json(bundle / "composition-controls.json")
        input_seal = _read_json(bundle / "input-seal.json")
    except Exception as exc:
        failures.append(f"input-read:{type(exc).__name__}:{exc}")
        return _empty_bundle(bundle, failures)
    if not isinstance(manifest, dict) or not isinstance(input_seal, dict):
        return _empty_bundle(bundle, [*failures, "manifest-or-seal:not-object"])
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        return _empty_bundle(bundle, [*failures, "records:not-object-list"])
    if not isinstance(causal_pairs, list) or not all(isinstance(row, dict) for row in causal_pairs):
        return _empty_bundle(bundle, [*failures, "causal-pairs:not-object-list"])
    if not isinstance(language_pairs, list) or not all(isinstance(row, dict) for row in language_pairs):
        return _empty_bundle(bundle, [*failures, "language-pairs:not-object-list"])
    if not isinstance(composition, dict):
        return _empty_bundle(bundle, [*failures, "composition:not-object"])

    snapshot_map = manifest.get("source_snapshot", {}).get("files", {}) if isinstance(manifest.get("source_snapshot"), dict) else {}
    if not isinstance(snapshot_map, dict):
        snapshot_map = {}
    snapshot_root = bundle / "source_snapshot"
    actual_snapshot: dict[str, str] = {}
    if snapshot_root.is_dir():
        for path in snapshot_root.rglob("*"):
            if path.is_file():
                actual_snapshot[path.relative_to(snapshot_root).as_posix()] = _sha_file(path)
    snapshot_expected = {str(key): str(value).upper() for key, value in snapshot_map.items()}
    actual_basenames = Counter(Path(relative).name for relative in snapshot_expected)
    actual_suffixes = Counter(Path(relative).suffix for relative in snapshot_expected)
    snapshot_profile = bool(
        len(snapshot_expected) == 32
        and actual_suffixes == EXPECTED_SNAPSHOT_SUFFIXES
        and all(actual_basenames[name] >= count for name, count in REQUIRED_SNAPSHOT_BASENAMES.items())
    )
    snapshot_safe = all(_safe_relative(snapshot_root, relative) is not None for relative in snapshot_expected)
    snapshot_digest = _sha_bytes(_canonical_bytes(snapshot_expected))
    snapshot_exact = bool(
        snapshot_safe and snapshot_profile and actual_snapshot == snapshot_expected
        and manifest.get("source_snapshot_set_sha256") == snapshot_digest
    )
    if not snapshot_exact:
        failures.append("snapshot:profile-set-hash-digest")

    sealed_expected = ["manifest.json", *DATA_FILES, *(f"source_snapshot/{relative}" for relative in sorted(snapshot_expected))]
    sealed_map = input_seal.get("files")
    seal_exact = input_seal.get("schema_version") == EXPECTED_SEAL_SCHEMA and isinstance(sealed_map, dict)
    seal_exact = bool(seal_exact and set(sealed_map) == set(sealed_expected))
    if seal_exact:
        for relative in sealed_expected:
            path = _safe_relative(bundle, relative)
            if path is None or not path.is_file() or _sha_file(path) != str(sealed_map[relative]).upper():
                seal_exact = False
                break
    input_exact = top_exact and seal_exact and snapshot_exact
    if not input_exact:
        failures.append("input:tree-or-seal")

    role_counts = Counter(
        row.get("structure_certificate", {}).get("role")
        for row in records
        if isinstance(row.get("structure_certificate"), dict)
    )
    actual_counts = {
        "records": len(records),
        "ere_records": sum(row.get("family") == "ere" for row in records),
        "cps_records": sum(row.get("family") == "cps" for row in records),
        "claims": sum(len(row.get("claims", [])) if isinstance(row.get("claims"), list) else 0 for row in records),
        "causal_pairs": len(causal_pairs),
        "language_pairs": len(language_pairs),
    }
    identifiers = [row.get("example_id") for row in records]
    fixed_profile = bool(
        actual_counts == FIXED_COUNTS
        and role_counts == Counter(FIXED_ROLES)
        and len(set(identifiers)) == 11
        and all(_nonempty(identifier) for identifier in identifiers)
    )
    if not fixed_profile:
        failures.append("profile:fixed-cardinality-role")

    actual_hashes = {relative: _sha_file(bundle / relative) for relative in DATA_FILES if (bundle / relative).is_file()}
    manifest_hashes = manifest.get("data_files_sha256")
    manifest_exact = bool(
        manifest.get("schema_version") == EXPECTED_MANIFEST_SCHEMA
        and manifest.get("contract_version") == EXPECTED_CONTRACT
        and manifest.get("qualification_mode") is True
        and manifest.get("counts") == FIXED_COUNTS
        and actual_counts == FIXED_COUNTS
        and isinstance(manifest_hashes, dict)
        and {str(key): str(value).upper() for key, value in manifest_hashes.items()} == actual_hashes
    )
    if not manifest_exact:
        failures.append("manifest:identity-counts-hashes")

    schema_exact = _schema_exact(manifest, records, causal_pairs, language_pairs, composition)
    if not schema_exact:
        failures.append("schema:exact")

    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    runtime_streams_exact = bool(
        set(runtime) == {"python", "implementation", "platform"}
        and runtime.get("python") == platform.python_version()
        and runtime.get("implementation") == platform.python_implementation()
        and runtime.get("platform") == platform.platform()
        and manifest.get("named_streams") == EXPECTED_STREAMS
        and manifest.get("required_claim_kinds") == EXPECTED_CLAIM_KINDS
    )
    if not runtime_streams_exact:
        failures.append("runtime-or-stream-profile")

    tokenizer, tokenizer_failures = _load_pinned_tokenizer()
    failures.extend(tokenizer_failures)
    manifest_tokenizer = manifest.get("tokenizer") if isinstance(manifest.get("tokenizer"), dict) else {}
    try:
        import tokenizers
        import transformers

        installed_versions_exact = bool(
            manifest_tokenizer.get("transformers_version") == transformers.__version__
            and manifest_tokenizer.get("tokenizers_version") == tokenizers.__version__
        )
    except Exception:
        installed_versions_exact = False
    tokenizer_exact = bool(
        all(manifest_tokenizer.get(key) == value for key, value in EXPECTED_TOKENIZER.items())
        and manifest_tokenizer.get("actual_class") == EXPECTED_TOKENIZER["tokenizer_class"]
        and manifest_tokenizer.get("is_fast") is True
        and installed_versions_exact
        and tokenizer is not None
        and type(tokenizer).__name__ == EXPECTED_TOKENIZER["tokenizer_class"]
    )
    if not tokenizer_exact:
        failures.append("tokenizer:identity")

    model_good = forbidden_count = recount_good = recount_total = max_tokens = 0
    if tokenizer is not None:
        for row in records:
            view = row.get("model_view")
            expected_view = {
                "example_id": row.get("example_id"),
                "source_text": row.get("source_text"),
                "reasoning_budget": row.get("reasoning_budget"),
                "valid_choice_mask": row.get("valid_choice_mask"),
            }
            if isinstance(view, dict) and set(view) == set(EXPECTED_MODEL_VIEW_FIELDS) and view == expected_view:
                model_good += 1
            if isinstance(view, dict):
                forbidden_count += sum(key.casefold() in FORBIDDEN_MODEL_VIEW_KEYS for key in _walk_keys(view))
            text = row.get("source_text")
            if isinstance(text, str):
                count = _token_count(tokenizer, text)
                max_tokens = max(max_tokens, count)
                recount_total += 1
                recount_good += int(row.get("qwen_token_count") == count)
        for pair in language_pairs:
            for text_key, count_key in (("train_text", "train_qwen_token_count"), ("ood_text", "ood_qwen_token_count")):
                text = pair.get(text_key)
                if isinstance(text, str):
                    count = _token_count(tokenizer, text)
                    max_tokens = max(max_tokens, count)
                    recount_total += 1
                    recount_good += int(pair.get(count_key) == count)
    model_exact = len(records) == 11 and model_good == 11
    recount_exact = recount_total == 15 and recount_good == 15
    max_ok = tokenizer is not None and max_tokens <= EXPECTED_TOKENIZER["max_source_tokens"]

    g02 = {
        "G02_M01_input_tree_and_seal_exact": _metric(input_exact, int(input_exact), 1, input_exact, ["input tree/seal mismatch"] if not input_exact else []),
        "G02_M02_fixed_qualification_profile_exact": _metric(fixed_profile, int(fixed_profile), 1, fixed_profile, ["fixed qualification profile mismatch"] if not fixed_profile else []),
        "G02_M03_manifest_identity_counts_hashes_exact": _metric(manifest_exact, int(manifest_exact), 1, manifest_exact, ["manifest identity/count/hash mismatch"] if not manifest_exact else []),
        "G02_M04_snapshot_profile_set_hash_digest_exact": _metric(snapshot_exact, int(snapshot_exact), 1, snapshot_exact, ["snapshot profile/set/hash/digest mismatch"] if not snapshot_exact else []),
        "G02_M05_runtime_and_stream_profile_exact": _metric(runtime_streams_exact, int(runtime_streams_exact), 1, runtime_streams_exact, ["runtime/stream profile mismatch"] if not runtime_streams_exact else []),
        "G02_M06_exact_schema_rate": _metric(schema_exact, int(schema_exact), 1, schema_exact, ["exact schema mismatch"] if not schema_exact else []),
        "G02_M07_model_view_exact_rate": _metric(model_good / len(records) if records else 0.0, model_good, len(records), model_exact, ["model view mismatch"] if not model_exact else []),
        "G02_M08_recursive_forbidden_field_count": _metric(forbidden_count, forbidden_count, 1, forbidden_count == 0, ["forbidden model-view key"] if forbidden_count else []),
        "G02_M09_tokenizer_identity_exact": _metric(tokenizer_exact, int(tokenizer_exact), 1, tokenizer_exact, ["tokenizer identity mismatch"] if not tokenizer_exact else []),
        "G02_M10_token_recount_match_rate": _metric(recount_good / recount_total if recount_total else 0.0, recount_good, recount_total, recount_exact, ["token recount mismatch"] if not recount_exact else []),
        "G02_M11_max_source_tokens": _metric(max_tokens, max_tokens, 1, max_ok, ["source token limit exceeded or tokenizer unavailable"] if not max_ok else []),
    }
    return {
        "root": bundle,
        "manifest": manifest,
        "records": records,
        "causal_pairs": causal_pairs,
        "language_pairs": language_pairs,
        "composition": composition,
        "input_seal": input_seal,
        "snapshot_map": snapshot_map,
        "tokenizer": tokenizer,
        "g02": g02,
        "errors": sorted(set(failures)),
    }
