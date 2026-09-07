from __future__ import annotations

"""Frozen-surface constants for the P1-H1 development comparison.

Hashes remain explicit placeholders until implementation probes and two
independent reviews are complete.  The formal runner refuses placeholders.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "r1r-p1-h1-factorized-routed-projection-development-v7"
SCHEMA_PREFIX = "yggdrasil.v2-r1r.p1-h1"

DESIGN = Path("docs/v2-r1r-p1-h1-mixed-core-design.md")
EXECUTION = Path("docs/v2-r1r-p1-h1-execution-command.md")
ROOTS = {
    "preflight": Path("artifacts/v2-r1r/p1-h1-preflight-20260817-1"),
    "development": Path("artifacts/v2-r1r/p1-h1-development-20260817-1"),
}
TRANSPORT_ROOT = Path("tmp/p1-h1-transport-20260817-1")
ATTEMPT_ID = "p1-h1-20260817-1"

NR1_ROOTS = {
    "preflight": {
        "root": Path("artifacts/v2-r1r/p1-nr1-preflight-20260817-1"),
        "status": "PASS_P1_NR1_PREFLIGHT",
        "seal_sha256": "1825282BF46C821BFD89636B58DCCDAD53CAAE1C9F226BA87B3CE8387B095DAF",
    },
    "qualification": {
        "root": Path("artifacts/v2-r1r/p1-nr1-qualification-20260817-1"),
        "status": "PASS_P1_NR1_MEASUREMENT_QUALIFICATION",
        "seal_sha256": "DADBDDBF3B672702A4FFADE74504EAD2F941628D729B5A29B0B7D4DA18EF6FB3",
    },
}
V8L_QUALIFICATION = {
    "root": Path(
        "artifacts/v2-r1r/p1-v8l-causal-state-ladder-qualification-20260812-1"
    ),
    "status": "FAIL_P1_V8L_BOOTSTRAP",
    "seal_sha256": "3332CD3D24DCE85BEB3D8ECA2A7D1D669E97CAF40924CABED88AC07F2C44FE75",
}
P0D_HISTORY = {
    "root": Path("artifacts/v2-r1r/p0d-v17-full-production-20260810-1"),
    "status_file": "assessment.json",
    "status": "PASS_P0D_PRODUCTION",
    "seal_sha256": "453305C3F6738B92B5119426B3562DA21079E9111E5FEDBD9F47AE546AB7999D",
}
P0M_ROOTS = {
    "cache": {
        "root": Path("artifacts/v2-r1r/p0m-v5-cache-qualification-20260810-1"),
        "seal_sha256": "F5FEF16EE4D1AFC3BDBB7779E681A806B289DF9B31781B995C6712567134FF9D",
    },
    "ere": {
        "root": Path("artifacts/v2-r1r/p0m-v5-ere-k8-overfit64-20260810-1"),
        "seal_sha256": "DAB1DABE1818C0761188D03A388AC5553D207C918728C3FE756DE0E8AA8F4E35",
    },
    "cps": {
        "root": Path("artifacts/v2-r1r/p0m-v5-cps-k8-overfit64-20260810-1"),
        "seal_sha256": "5EC4B0D7222C25041D61DA7F77B85FB1C85E9110CA26104487A79AC011BD390F",
    },
    "joint": {
        "root": Path("artifacts/v2-r1r/p0m-v5-joint-k8-overfit128-20260810-1"),
        "seal_sha256": "91921CD18A245D34CB1DF3CBEE94706FA11C01E3B1735513E788AD7A2AEB341B",
    },
    "direct": {
        "root": Path("artifacts/v2-r1r/p0m-v5-direct-overfit64-20260810-1"),
        "seal_sha256": "E3573F6843A4DB2326C66D1AB01C1AE5898CE7A75DF829A734D608A0112B1AAE",
    },
    "text_cot": {
        "root": Path("artifacts/v2-r1r/p0m-v5-text-cot-overfit64-20260810-1"),
        "seal_sha256": "2F2508DE93A2781BF279C4202FF1F913A9168251F9ECD85A4368B71D43A1D618",
    },
    "throughput": {
        "root": Path("artifacts/v2-r1r/p0m-v5-throughput-20260810-1"),
        "seal_sha256": "30BD34FF6C11069E3FF854E7F7D5E9ED4D3A350817A5E7A731C741162E897684",
    },
    "assessment": {
        "root": Path("artifacts/v2-r1r/p0m-v5-assessment-20260810-1"),
        "status": "PASS_P0M",
        "seal_sha256": "17F45EAC8A3DF129B236688D3D1E2B22BD9B634D4098F903142CA42E7FEEF09E",
    },
}

DATA_SEEDS = (2026081801, 2026081802, 2026081803)
MODEL_SEEDS = (2026081811, 2026081812, 2026081813)
# Consumed non-formal identities remain permanently reserved evidence.  They
# are regenerated before the current screen/calibration so no later package
# can reuse their supervised semantic or source fingerprints.
PRIOR_NONFORMAL_SEEDS = {
    "balanced_routed_screen_history": {
        "data": 2026081791,
        "model": 2026081792,
    },
    "balanced_routed_calibration_history": {
        "data": 2026081793,
        "model": 2026081794,
    },
}
SCREEN_DATA_SEED = 2026081761
SCREEN_MODEL_SEED = 2026081762
CALIBRATION_DATA_SEED = 2026081763
CALIBRATION_MODEL_SEED = 2026081764
REGISTERED_NONFORMAL_DATA_SEEDS = {
    role: int(values["data"]) for role, values in PRIOR_NONFORMAL_SEEDS.items()
}
REGISTERED_NONFORMAL_DATA_SEEDS.update(
    {
        "screen": SCREEN_DATA_SEED,
        "calibration": CALIBRATION_DATA_SEED,
    }
)
REGISTERED_NONFORMAL_PACKAGE_IDENTITIES = {
    "balanced_routed_screen_history": (
        "CFCE5DAE42660DD0F0E03CE69ABC81AB720F108064085109DA398EA6885650F8"
    ),
    "balanced_routed_calibration_history": (
        "47D36E7D236F97D01E8DDB83F71F67F7852C2B2B6F930926A4E782B10E6BBD77"
    ),
    "screen": (
        "A52C52225921A0F834A43031A9F4E595B301D7B99AACD2419882D41198486DA4"
    ),
    "calibration": (
        "21CBD9227C0D6E39AD45C17BA206B266763DAE816E1A2CFC62FE742BDCE761E9"
    ),
}
NONFORMAL_ROOTS = {
    "screen": Path(
        "artifacts/v2-r1r/"
        "p1-h1-nonformal-factorized-routed-projection-screen-20260817-1"
    ),
    "calibration": Path(
        "artifacts/v2-r1r/"
        "p1-h1-nonformal-factorized-routed-projection-calibration-20260817-1"
    ),
}
SMOKE_DATA_SEED = 2026081891
SMOKE_MODEL_SEED = 2026081892
SPLIT_COUNTS = {
    "train": 4096,
    "validation": 512,
    "supported": 1024,
    "heldout": 1024,
    "causal": 512,
}


@dataclass(frozen=True)
class TrainingSpec:
    batch_size: int = 32
    updates: int = 4000
    learning_rate: float = 3.0e-4
    minimum_learning_rate_scale: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    answer_weight: float = 1.0
    trace_weight: float = 0.50
    route_weight: float = 0.25
    evaluation_interval: int = 500
    dtype: str = "float32"


@dataclass(frozen=True)
class ModelSpec:
    source_width: int = 2048
    latent_width: int = 256
    latent_slots: int = 8
    recurrent_steps: int = 8
    recurrent_layers: int = 2
    attention_heads: int = 8
    active_ffn_multiplier: int = 3
    shared_ffn_inner_width: int = 384
    routed_feature_width: int = 384
    trace_classes: int = 5
    answer_classes: int = 4
    route_classes: int = 2


@dataclass(frozen=True)
class SmokeSpec:
    """Registered, non-qualifying development smoke run.

    The 32 records are disjoint from calibration and formal data.  This run
    only proves that gradients, all three heads, and deployment stripping are
    operational before the first registered 4,000-update arm is started.
    """

    train_per_family: int = 16
    batch_size: int = 32
    updates: int = 800
    evaluation_interval: int = 100
    answer_floor: float = 0.95
    trace_floor: float = 0.90
    route_floor: float = 0.99
    cache_batch_size: int = 8
    minimum_free_disk_gib: int = 40


@dataclass(frozen=True)
class CalibrationMarginSpec:
    """Pre-result rule for converting one calibration seed into Gate floors.

    Each observed worst registered cell is reduced by the fixed margin and
    rounded down to one percentage point.  H05/H06 claim thresholds, including
    conditional-write necessity, are not derived and therefore cannot be
    weakened by calibration.
    """

    competence: float = 0.05
    causal_drop: float = 0.05
    metamorphic: float = 0.05
    route_accuracy: float = 0.01
    floor_quantum: float = 0.01


@dataclass(frozen=True)
class GateThresholds:
    # Final values are frozen only after a disjoint non-formal probe.  The
    # placeholders deliberately make formal execution impossible meanwhile.
    supported_answer_floor: float = -1.0
    supported_trace_floor: float = -1.0
    heldout_shared_answer_floor: float = -1.0
    heldout_mixed_answer_floor: float = -1.0
    heldout_shared_trace_floor: float = -1.0
    heldout_mixed_trace_floor: float = -1.0
    primary_gain: float = 0.05
    family_regression_limit: float = 0.02
    route_causal_drop: float = 0.10
    conditional_write_causal_drop: float = 0.05
    source_causal_drop: float = -1.0
    recurrence_causal_drop: float = -1.0
    metamorphic_consistency: float = -1.0
    route_accuracy: float = -1.0

    def frozen(self) -> bool:
        values = asdict(self)
        return all(float(value) >= 0.0 for value in values.values())


MODEL = ModelSpec()
TRAINING = TrainingSpec()
SMOKE = SmokeSpec()
CALIBRATION_MARGINS = CalibrationMarginSpec()
THRESHOLDS = GateThresholds()

FROZEN_CONTRACT_HASHES = {
    DESIGN.as_posix(): "TO_BE_FROZEN",
    EXECUTION.as_posix(): "TO_BE_FROZEN",
}


def contract_manifest() -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}.contract.v1",
        "contract_version": CONTRACT_VERSION,
        "design": DESIGN.as_posix(),
        "execution": EXECUTION.as_posix(),
        "roots": {name: path.as_posix() for name, path in ROOTS.items()},
        "transport_root": TRANSPORT_ROOT.as_posix(),
        "attempt_id": ATTEMPT_ID,
        "data_seeds": list(DATA_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "screen_data_seed": SCREEN_DATA_SEED,
        "screen_model_seed": SCREEN_MODEL_SEED,
        "calibration_data_seed": CALIBRATION_DATA_SEED,
        "calibration_model_seed": CALIBRATION_MODEL_SEED,
        "prior_nonformal_seeds": PRIOR_NONFORMAL_SEEDS,
        "registered_nonformal_data_seeds": REGISTERED_NONFORMAL_DATA_SEEDS,
        "registered_nonformal_package_identities": (
            REGISTERED_NONFORMAL_PACKAGE_IDENTITIES
        ),
        "nonformal_roots": {
            name: path.as_posix() for name, path in NONFORMAL_ROOTS.items()
        },
        "smoke_data_seed": SMOKE_DATA_SEED,
        "smoke_model_seed": SMOKE_MODEL_SEED,
        "p0d_history": {
            key: value.as_posix() if isinstance(value, Path) else value
            for key, value in P0D_HISTORY.items()
        },
        "p0m_roots": {
            name: {
                key: value.as_posix() if isinstance(value, Path) else value
                for key, value in specification.items()
            }
            for name, specification in P0M_ROOTS.items()
        },
        "split_counts": SPLIT_COUNTS,
        "model": asdict(MODEL),
        "training": asdict(TRAINING),
        "smoke": asdict(SMOKE),
        "calibration_margins": asdict(CALIBRATION_MARGINS),
        "thresholds": asdict(THRESHOLDS),
        "frozen_contract_hashes": FROZEN_CONTRACT_HASHES,
        "claims": {
            "stage": "development comparison only",
            "p1_completed": False,
            "p2_eligible": False,
            "p2_started": False,
            "pass_authorizes": "P1-F1 design only",
        },
    }


__all__ = [
    "ATTEMPT_ID",
    "CALIBRATION_DATA_SEED",
    "CALIBRATION_MARGINS",
    "CALIBRATION_MODEL_SEED",
    "CalibrationMarginSpec",
    "CONTRACT_VERSION",
    "DATA_SEEDS",
    "DESIGN",
    "EXECUTION",
    "FROZEN_CONTRACT_HASHES",
    "GateThresholds",
    "MODEL_SEEDS",
    "MODEL",
    "ModelSpec",
    "NONFORMAL_ROOTS",
    "NR1_ROOTS",
    "P0D_HISTORY",
    "P0M_ROOTS",
    "ROOTS",
    "PRIOR_NONFORMAL_SEEDS",
    "REGISTERED_NONFORMAL_DATA_SEEDS",
    "REGISTERED_NONFORMAL_PACKAGE_IDENTITIES",
    "SCHEMA_PREFIX",
    "SCREEN_DATA_SEED",
    "SCREEN_MODEL_SEED",
    "SMOKE",
    "SMOKE_DATA_SEED",
    "SMOKE_MODEL_SEED",
    "SmokeSpec",
    "SPLIT_COUNTS",
    "THRESHOLDS",
    "TRAINING",
    "TRANSPORT_ROOT",
    "TrainingSpec",
    "V8L_QUALIFICATION",
    "contract_manifest",
]
