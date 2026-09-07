from __future__ import annotations

"""C1 causal interventions.

All grouping/family knowledge stays in the caller.  The functions below take
only an explicit batch index mapping, so a within-family shuffle cannot be
silently replaced by a model-side family branch.
"""

from typing import Literal

import torch
from torch import Tensor

from .model import C1Model


def _batch_permutation(mapping: Tensor, batch: int, *, device: str | None = None) -> Tensor:
    result = mapping.to(dtype=torch.long)
    if device is not None:
        result = result.to(device)
    if result.ndim != 1 or result.numel() != batch:
        raise ValueError("intervention mapping must have shape [batch]")
    expected = torch.arange(batch, device=result.device)
    if not torch.equal(torch.sort(result).values, expected):
        raise ValueError("intervention mapping must be a batch permutation")
    return result


def zero_hidden(model: C1Model, source_hidden: Tensor, source_mask: Tensor, *, return_trajectory: bool = False) -> dict[str, Tensor]:
    """Replace the full source hidden with zeros before Boundary."""
    return model(torch.zeros_like(source_hidden), source_mask, return_trajectory=return_trajectory)


def shuffle_hidden(
    model: C1Model,
    source_hidden: Tensor,
    source_mask: Tensor,
    mapping: Tensor,
    *,
    return_trajectory: bool = False,
) -> dict[str, Tensor]:
    """Shuffle source examples with an explicit caller-supplied batch map."""
    indices = _batch_permutation(mapping, source_hidden.shape[0], device=str(source_hidden.device))
    return model(source_hidden[indices], source_mask[indices], return_trajectory=return_trajectory)


def no_core(model: C1Model, source_hidden: Tensor, source_mask: Tensor) -> dict[str, Tensor]:
    """Read out H0 directly, omitting all recurrent transitions."""
    h0 = model.encode(source_hidden, source_mask)
    logits = model.logits_from_state(h0)
    return {"logits": logits, "h0": h0, "final_state": h0}


h0_no_core = no_core


def step5_within_family_state_shuffle(
    model: C1Model,
    source_hidden: Tensor,
    source_mask: Tensor,
    mapping: Tensor,
    *,
    return_trajectory: bool = True,
) -> dict[str, Tensor]:
    """Shuffle H5 across a caller-provided within-family mapping, then continue."""
    indices = _batch_permutation(mapping, source_hidden.shape[0], device=str(source_hidden.device))
    return model(
        source_hidden,
        source_mask,
        return_trajectory=return_trajectory,
        state_shuffle_step=5,
        state_shuffle_indices=indices,
    )


def state_shuffle(
    model: C1Model,
    source_hidden: Tensor,
    source_mask: Tensor,
    mapping: Tensor,
    *,
    step: int = 5,
    return_trajectory: bool = True,
) -> dict[str, Tensor]:
    """Generic explicit latent-state shuffle; ``step=5`` is the C1 gate."""
    indices = _batch_permutation(mapping, source_hidden.shape[0], device=str(source_hidden.device))
    return model(
        source_hidden,
        source_mask,
        return_trajectory=return_trajectory,
        state_shuffle_step=step,
        state_shuffle_indices=indices,
    )


def initial_slot_permutation(
    model: C1Model,
    source_hidden: Tensor,
    source_mask: Tensor,
    permutation: Tensor,
    *,
    return_trajectory: bool = True,
) -> dict[str, Tensor]:
    """Permute H0 slots before the source-closed recurrent core."""
    return model(
        source_hidden,
        source_mask,
        return_trajectory=return_trajectory,
        initial_slot_permutation=permutation,
    )


Intervention = Literal["zero_hidden", "shuffle_hidden", "h0_no_core", "step5_state_shuffle", "initial_slot_permutation"]


def apply_intervention(
    model: C1Model,
    source_hidden: Tensor,
    source_mask: Tensor,
    kind: Intervention,
    *,
    mapping: Tensor | None = None,
    permutation: Tensor | None = None,
    return_trajectory: bool = False,
) -> dict[str, Tensor]:
    """Uniform dispatch for evaluators; mappings remain explicit arguments."""
    if kind == "zero_hidden":
        return zero_hidden(model, source_hidden, source_mask, return_trajectory=return_trajectory)
    if kind == "shuffle_hidden":
        if mapping is None:
            raise ValueError("shuffle_hidden requires caller-supplied mapping")
        return shuffle_hidden(model, source_hidden, source_mask, mapping, return_trajectory=return_trajectory)
    if kind == "h0_no_core":
        return no_core(model, source_hidden, source_mask)
    if kind == "step5_state_shuffle":
        if mapping is None:
            raise ValueError("step5_state_shuffle requires caller-supplied mapping")
        return step5_within_family_state_shuffle(model, source_hidden, source_mask, mapping, return_trajectory=return_trajectory)
    if kind == "initial_slot_permutation":
        if permutation is None:
            raise ValueError("initial_slot_permutation requires a slot permutation")
        return initial_slot_permutation(model, source_hidden, source_mask, permutation, return_trajectory=return_trajectory)
    raise ValueError(f"unknown C1 intervention: {kind}")


__all__ = [
    "apply_intervention",
    "h0_no_core",
    "initial_slot_permutation",
    "no_core",
    "shuffle_hidden",
    "state_shuffle",
    "step5_within_family_state_shuffle",
    "zero_hidden",
]
