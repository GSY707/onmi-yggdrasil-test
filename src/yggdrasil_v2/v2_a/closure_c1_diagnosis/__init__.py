"""Post-stop, non-qualifying diagnostics for the consumed C1 formal root.

This package intentionally lives outside :mod:`closure_c1`.  It may read the
sealed C1 result and its selected checkpoint, but it never resumes training,
changes the formal root, or emits an authorization for a later stage.
"""

from .evaluator import (
    DIAGNOSTIC_SCHEMA,
    DIAGNOSTIC_STATUS,
    load_selected_checkpoint_pin,
    run_poststop_diagnosis,
)
from .runner import run_formal, run_preflight

__all__ = [
    "DIAGNOSTIC_SCHEMA",
    "DIAGNOSTIC_STATUS",
    "load_selected_checkpoint_pin",
    "run_poststop_diagnosis",
    "run_formal",
    "run_preflight",
]
