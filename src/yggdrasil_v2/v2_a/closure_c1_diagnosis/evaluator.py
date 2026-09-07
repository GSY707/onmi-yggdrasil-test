from __future__ import annotations

"""Capability-only evaluation after the consumed C1 G007 fail-stop.

The evaluator is deliberately not a formal runner.  It loads only the
selected C1 checkpoint, strips the training-only trace probe in memory, and
measures the later capabilities which the formal stop tree did not reach.
Every returned value is JSON-native and the top-level status makes it
impossible to confuse these measurements with formal qualification.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import hashlib
import json
import math
from typing import Any

import torch

from yggdrasil_v2.v2_a.closure_c1 import contract
from yggdrasil_v2.v2_a.closure_c1.deployment import reload_stripped, strip_trace_probe
from yggdrasil_v2.v2_a.closure_c1.evaluate import (
    evaluate_behavior,
    evaluate_paired_intervention,
)
from yggdrasil_v2.v2_a.closure_c1.evaluation_runtime import C1EvaluationRuntime
from yggdrasil_v2.v2_a.closure_c1.model import C1Config, C1Model


DIAGNOSTIC_SCHEMA = "yggdrasil.v2-a.closure-c1.poststop-diagnostic.v1"
DIAGNOSTIC_STATUS = "POSTSTOP_DIAGNOSTIC_ONLY"
_FORMAL_FAILURE_STATUS = "FAIL_V2_A_C1_SINGLE_SEED_ELIGIBILITY"
_EXPECTED_STOP = "G007"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _inside(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("diagnostic checkpoint must remain inside the sealed formal root") from exc
    return resolved


def _finite(value: Any) -> Any:
    """Convert tensor/NumPy-like scalars and reject non-finite output."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _finite(value.detach().cpu().item())
        return [_finite(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("diagnostic output contains a non-finite number")
        return int(value) if isinstance(value, int) else result
    if hasattr(value, "item"):
        return _finite(value.item())
    raise TypeError(f"diagnostic output is not JSON-native: {type(value).__name__}")


def _invariance(before: Mapping[str, Sequence[float]], after: Mapping[str, Sequence[float]], *, tolerance: float) -> dict[str, Any]:
    if set(before) != set(after) or not before:
        raise ValueError("invariance comparison requires equal non-empty record identities")
    same = 0
    maximum = 0.0
    for key in sorted(before):
        left = [float(item) for item in before[key]]
        right = [float(item) for item in after[key]]
        if len(left) != 9 or len(right) != 9:
            raise ValueError("C1 answer invariance requires nine raw logits")
        same += int(max(range(9), key=left.__getitem__) == max(range(9), key=right.__getitem__))
        maximum = max(maximum, max(abs(a - b) for a, b in zip(left, right)))
    return {
        "n": len(before),
        "prediction_invariance": same / len(before),
        "max_abs_diff": maximum,
        "tolerance": tolerance,
        "passed": bool(same == len(before) and maximum <= tolerance),
    }


def _split(records: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    rows = [row for row in records if row.get("split") == split]
    if not rows:
        raise ValueError(f"post-stop diagnosis requires non-empty split: {split}")
    return rows


def _predictor(logits: Mapping[str, Sequence[float]]):
    return lambda record: {"logits": logits[str(record["example_id"])]}


def _paired(
    records: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Sequence[float]],
    changed: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return evaluate_paired_intervention(
        records,
        _predictor(baseline),
        _predictor(changed),
        cluster_key="example_id",
    )


def load_selected_checkpoint_pin(formal_root: Path, checkpoint: Path | None = None) -> dict[str, Any]:
    """Verify the consumed C1 selected checkpoint without changing any file.

    The result and primary-training records are treated as an immutable pin;
    callers cannot choose another candidate checkpoint unless they explicitly
    pass the exact path recorded as selected.
    """
    root = Path(formal_root).resolve()
    result = _json_file(root / "result.json")
    if result.get("status") != _FORMAL_FAILURE_STATUS:
        raise ValueError("post-stop diagnosis requires the consumed C1 G007 failure result")
    if result.get("stopped_at") != _EXPECTED_STOP:
        raise ValueError("post-stop diagnosis requires a C1 result stopped at G007")
    if result.get("authorizes") != "nothing" or result.get("c2_authorized") is not False:
        raise ValueError("post-stop diagnosis refuses a result with any authorization")
    primary = _json_file(root / "primary-training.json")
    selected = primary.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("C1 primary-training record has no selected checkpoint")
    selected_rel = Path(str(selected.get("checkpoint", "")))
    selected_path = _inside(root, root / selected_rel)
    if checkpoint is not None and _inside(root, Path(checkpoint)) != selected_path:
        raise ValueError("diagnostic evaluator is pinned to the formal selected checkpoint")
    observed_hash = _sha256_file(selected_path)
    expected_hash = str(selected.get("checkpoint_sha256", ""))
    result_hash = str(result.get("selected_checkpoint_sha256", expected_hash))
    if not expected_hash or observed_hash != expected_hash or result_hash != observed_hash:
        raise ValueError("selected checkpoint SHA-256 does not match the sealed C1 records")
    seal_path = root / "evidence-seal.json"
    if not seal_path.is_file():
        raise ValueError("sealed C1 formal root has no evidence-seal.json")
    seal = _json_file(seal_path)
    sealed_files = seal.get("files")
    selected_key = selected_path.relative_to(root).as_posix()
    if not isinstance(sealed_files, Mapping) or sealed_files.get(selected_key) != observed_hash:
        raise ValueError("selected checkpoint is not covered by the C1 evidence seal")
    if sealed_files.get("result.json") != _sha256_file(root / "result.json"):
        raise ValueError("C1 result.json is not covered by the evidence seal")
    payload = torch.load(selected_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("selected checkpoint payload is not a mapping")
    expected_schema = f"{contract.SCHEMA_PREFIX}.training-checkpoint.v1"
    if payload.get("schema_version") != expected_schema:
        raise ValueError("selected checkpoint schema is not the C1 training schema")
    if payload.get("identity") != contract.C1_IDENTITY:
        raise ValueError("selected checkpoint identity does not match C1")
    if int(payload.get("update", -1)) != int(selected.get("update", -2)):
        raise ValueError("selected checkpoint update disagrees with primary-training.json")
    return _finite({
        "formal_root": root.as_posix(),
        "checkpoint": selected_path.as_posix(),
        "checkpoint_sha256": observed_hash,
        "update": int(payload["update"]),
        "identity": str(payload["identity"]),
        "formal_status": str(result["status"]),
        "stopped_at": str(result["stopped_at"]),
        "authorizes": "nothing",
    })


def _load_models(pin: Mapping[str, Any], formal_root: Path, device: str | torch.device) -> tuple[C1Model, Any, dict[str, Any]]:
    manifest = _json_file(Path(formal_root) / "contract-manifest.json")
    config_raw = manifest.get("model", contract.MODEL_CONFIG)
    if not isinstance(config_raw, Mapping):
        raise ValueError("C1 contract manifest has no model config")
    config = C1Config(**{str(key): value for key, value in config_raw.items()})
    model = C1Model(config, with_trace_probe=True)
    payload = torch.load(Path(str(pin["checkpoint"])), map_location="cpu", weights_only=True)
    state = payload.get("model_state") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("selected checkpoint has no model_state")
    model.load_state_dict(dict(state), strict=True)
    model.to(device)
    model.eval()
    stripped = strip_trace_probe(model)
    reloaded = reload_stripped(stripped.state_dict, config, device=device)
    deployed = reloaded.model.eval()
    report = {
        "training_probe_parameters": sum(value.numel() for key, value in state.items() if str(key).startswith("trace_probe.")),
        "stripped_probe_parameters": int(stripped.report.get("trace_probe_parameters", -1)),
        "reloaded_probe_parameters": int(reloaded.report.get("trace_probe_parameters", -1)),
        "deployment_parameter_count": int(reloaded.report.get("deployment_parameters", -1)),
        "physical_strip": stripped.report,
        "strict_reload": reloaded.report,
    }
    return model, deployed, _finite(report)


def run_poststop_diagnosis(*, formal_root: Path, dataset: Any, records: Iterable[Mapping[str, Any]], device: str | torch.device = "cpu", batch_size: int = 8) -> dict[str, Any]:
    """Run capability measurements from the sealed selected checkpoint.

    ``dataset`` must implement the existing C1 ``get_batch`` protocol.  The
    evaluator consumes offline labels only in the evaluator; they are never
    passed to the model.  This function performs no optimizer step and writes
    no artifact.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = list(records)
    if not rows:
        raise ValueError("post-stop diagnosis requires records")
    root = Path(formal_root).resolve()
    pin = load_selected_checkpoint_pin(root)
    prestrip, deployed, strip_report = _load_models(pin, root, device)
    validation = _split(rows, "validation")
    ood = [row for row in rows if row.get("split") in {"composition_ood", "entity_ood", "length_ood", "language_ood", "distractor_ood", "horizon_ood"}]
    if not ood:
        raise ValueError("post-stop diagnosis requires OOD records")
    causal = _split(rows, "causal_pairs")

    runtime = C1EvaluationRuntime(deployed, dataset, validation, device=device, batch_size=batch_size)
    validation_logits = runtime.raw_answer_logits(records=validation)
    pre_runtime = C1EvaluationRuntime(prestrip, dataset, validation, device=device, batch_size=batch_size)
    prestrip_logits = pre_runtime.raw_answer_logits(records=validation)
    strip_invariance = _invariance(prestrip_logits, validation_logits, tolerance=1.0e-6)

    validation_report = evaluate_behavior(validation, _predictor(validation_logits))
    ood_report = C1EvaluationRuntime(deployed, dataset, ood, device=device, batch_size=batch_size).evaluate_answers(records=ood)
    causal_report = C1EvaluationRuntime(deployed, dataset, causal, device=device, batch_size=batch_size).evaluate_causal(records=causal)
    zero_logits = runtime.raw_answer_logits(intervention="zero_hidden", records=validation)
    shuffled_logits = runtime.raw_answer_logits(intervention="shuffled_hidden", records=validation)
    hidden_report = {
        "zero_hidden": _paired(validation, validation_logits, zero_logits),
        "shuffled_hidden": _paired(validation, validation_logits, shuffled_logits),
    }
    h0_logits = runtime.raw_answer_logits(intervention="h0_no_core", records=validation)
    step5_logits = runtime.raw_answer_logits(intervention="step5_state_shuffle", records=validation)
    recurrence_report = {
        "h0_no_core": _paired(validation, validation_logits, h0_logits),
        "step5_state_shuffle": _paired(validation, validation_logits, step5_logits),
    }
    baseline = validation_logits
    slot_logits = runtime.raw_answer_logits(intervention="slot_permutation", records=validation)
    slot_invariance = _invariance(baseline, slot_logits, tolerance=1.0e-5)
    integrity = {
        "strip": strip_invariance,
        "slot": slot_invariance,
        "deployment": strip_report,
        "training_optimizer_steps": 0,
        "diagnostic_only": True,
    }
    return _finite({
        "schema_version": DIAGNOSTIC_SCHEMA,
        "status": DIAGNOSTIC_STATUS,
        "formal": False,
        "qualification": "not_qualified",
        "authorizes": "nothing",
        "c2_authorized": False,
        "v2a_passed": False,
        "source_formal": pin,
        "checkpoint": pin,
        "measurements": {
            "validation": validation_report,
            "ood": ood_report,
            "causal": causal_report,
            "hidden": hidden_report,
            "recurrence": recurrence_report,
            "integrity": integrity,
        },
        "formal_gate_status": {
            "G003": "not_run_formally",
            "G004": "not_run_formally",
            "G005": "not_run_formally",
            "G006": "not_run_formally",
            "G008": "not_run_formally",
            "G009": "not_run_formally",
            "G010": "not_run_formally",
            "G011": "not_run_formally",
        },
    })


__all__ = ["DIAGNOSTIC_SCHEMA", "DIAGNOSTIC_STATUS", "load_selected_checkpoint_pin", "run_poststop_diagnosis"]
