"""Qualified production data, rendering, and measurement surface for V2-R1R."""

from .renderer import (
    LABELS,
    OOD_TEMPLATES,
    TEMPLATES,
    TRAIN_TEMPLATES,
    RendererError,
    parse_source,
    render_source,
)
from .scalable import compact_predict_multinomial_nb
from .schema import model_view, validate_model_view
from .fingerprint import semantic_fingerprint, surface_fingerprint
from .generator import (
    DatasetSpec,
    FORMAL_SPEC,
    generate_production_dataset,
    load_production_dataset,
)
from .generator_audit import audit_production_dataset

__all__ = [
    "LABELS",
    "OOD_TEMPLATES",
    "TEMPLATES",
    "TRAIN_TEMPLATES",
    "RendererError",
    "compact_predict_multinomial_nb",
    "DatasetSpec",
    "FORMAL_SPEC",
    "audit_production_dataset",
    "generate_production_dataset",
    "load_production_dataset",
    "model_view",
    "parse_source",
    "render_source",
    "semantic_fingerprint",
    "surface_fingerprint",
    "validate_model_view",
]
