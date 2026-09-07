from __future__ import annotations

"""Read-only path and latent collection for the failed C1S S1 endpoint."""

from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ..closure_c1s.model import C1SModel
from ..closure_c1s_successor.objective import SuccessorDiagnosticHeads
from ..closure_c1s_successor.runtime import C1SSuccessorRuntime
from .metrics import answer_margin, finite_json, paired_cluster_bootstrap, wilson_interval
from .runtime import chunks, model_inputs, move_batch


PATH_NAMES = (
    "full",
    "h0",
    "core_only",
    "h0_relevant_replace",
    "global_mean",
    "copy_all_query_owner",
    "final_delta",
    "zero_payload",
    "zero_operation_state_only",
    "zero_query",
    "reverse_operation_order",
)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _present_mean(payloads: Tensor, presence: Tensor) -> Tensor:
    if presence.shape != payloads.shape[:2]:
        raise ValueError("presence and payload shapes differ")
    weights = presence.to(dtype=payloads.dtype).unsqueeze(-1)
    return (payloads * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


def _replace_owner(payloads: Tensor, owners: Tensor, replacement: Tensor) -> Tensor:
    result = payloads.clone()
    if owners.ndim != 1 or owners.shape[0] != payloads.shape[0]:
        raise ValueError("owner vector shape mismatch")
    for index, owner in enumerate(owners.tolist()):
        if not 0 <= int(owner) < payloads.shape[1]:
            raise ValueError("query owner outside slot range")
        result[index, int(owner)] = replacement[index, 0]
    return result


def _mean_excluding_index(payloads: Tensor, mask: Tensor, index: Tensor) -> Tensor:
    """Mean masked payloads other than each row's selected physical slot."""

    if mask.shape != payloads.shape[:2] or index.shape != payloads.shape[:1]:
        raise ValueError("leave-one-out replacement shape mismatch")
    if bool(((index < 0) | (index >= payloads.shape[1])).any()):
        raise ValueError("leave-one-out index outside slot range")
    weights = mask.to(dtype=payloads.dtype).unsqueeze(-1)
    total = (payloads * weights).sum(dim=1)
    selected_mask = mask.gather(1, index[:, None]).squeeze(1)
    selected_payload = payloads.gather(
        1, index[:, None, None].expand(-1, 1, payloads.shape[-1])
    ).squeeze(1)
    other_count = (mask.sum(dim=1) - selected_mask.to(mask.sum(dim=1).dtype)).clamp_min(1)
    return (
        total - selected_payload * selected_mask.to(payloads.dtype).unsqueeze(-1)
    ) / other_count.to(payloads.dtype).unsqueeze(-1)


def _slot_logits(model: C1SModel, payloads: Tensor) -> Tensor:
    # The natural CUDA path produced these states under BF16 autocast.  This
    # helper runs after leaving that context, so restore the registered
    # readout parameter dtype explicitly rather than relying on ambient state.
    dtype = model.answer_head.weight.dtype
    return model.answer_head(model.final_norm(payloads.to(dtype=dtype)))


def _cpu(value: Tensor) -> Tensor:
    result = value.detach().float().cpu()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite diagnostic tensor")
    return result


def collect_paths(
    model: C1SModel,
    heads: SuccessorDiagnosticHeads,
    runtime: C1SSuccessorRuntime,
    example_ids: Sequence[str],
    *,
    device: str | torch.device,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Collect natural and counterfactual paths without mutating parameters."""

    active_device = torch.device(device)
    if active_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnosis requested but unavailable")
    model.to(active_device).eval()
    heads.to(active_device).eval()
    paths: dict[str, list[Tensor]] = {name: [] for name in PATH_NAMES}
    step_logits: list[Tensor] = []
    h0_slot_logits: list[Tensor] = []
    final_slot_logits: list[Tensor] = []
    initial_slot_replace_logits: list[Tensor] = []
    query_weights: list[Tensor] = []
    initial_payloads: list[Tensor] = []
    trajectories: list[Tensor] = []
    state_features: list[Tensor] = []
    state_logits: list[Tensor] = []
    answers: list[Tensor] = []
    presence: list[Tensor] = []
    content_mask: list[Tensor] = []
    query_owner: list[Tensor] = []
    operation_active_target: list[Tensor] = []
    state_values: list[Tensor] = []
    state_feature_mask: list[Tensor] = []
    state_step_mask: list[Tensor] = []
    families: list[str] = []
    feature_names: list[list[str]] = []
    ordered_ids: list[str] = []

    with torch.no_grad():
        for ids in chunks(example_ids, batch_size):
            cpu_batch = runtime.get_batch(
                ids,
                slots=model.config.slots,
                pin_memory=active_device.type == "cuda",
            )
            batch = move_batch(cpu_batch, active_device)
            with _autocast(active_device):
                boundary = model.boundary(**model_inputs(batch))
                full = model.forward_from_boundary(
                    boundary,
                    return_trajectory=True,
                    return_auxiliary=True,
                )
                trajectory = full["trajectory"]
                full_logits = full["logits"]
                h0_logits, _ = model.logits_from_state(
                    boundary.payloads, boundary.addresses, boundary.query_key
                )
                mean = _present_mean(boundary.payloads, batch["presence"].bool())
                mean_all = torch.where(
                    batch["presence"].bool().unsqueeze(-1),
                    mean.expand_as(boundary.payloads),
                    boundary.payloads,
                )
                core = model.forward_from_boundary(replace(boundary, payloads=mean_all))
                owner_indices = batch["query_owner"].long()
                owner_replacement = _mean_excluding_index(
                    boundary.payloads,
                    batch["presence"].bool(),
                    owner_indices,
                ).unsqueeze(1)
                relevant_payloads = _replace_owner(
                    boundary.payloads,
                    owner_indices,
                    owner_replacement,
                )
                h0_relevant, _ = model.logits_from_state(
                    relevant_payloads, boundary.addresses, boundary.query_key
                )
                global_mean, _ = model.logits_from_state(
                    mean_all, boundary.addresses, boundary.query_key
                )
                owner_payload = boundary.payloads.gather(
                    1,
                    batch["query_owner"].long()[:, None, None].expand(
                        -1, 1, boundary.payloads.shape[-1]
                    ),
                )
                copied_payloads = torch.where(
                    batch["presence"].bool().unsqueeze(-1),
                    owner_payload.expand_as(boundary.payloads),
                    boundary.payloads,
                )
                copied = model.forward_from_boundary(
                    replace(boundary, payloads=copied_payloads)
                )
                slot_replaced: list[Tensor] = []
                for slot in range(model.config.slots):
                    changed_payloads = boundary.payloads.clone()
                    slot_indices = torch.full(
                        (boundary.payloads.shape[0],),
                        slot,
                        dtype=torch.long,
                        device=boundary.payloads.device,
                    )
                    content_replacement = _mean_excluding_index(
                        boundary.payloads,
                        batch["content_mask"].bool(),
                        slot_indices,
                    )
                    changed_payloads[:, slot] = torch.where(
                        batch["content_mask"][:, slot].bool().unsqueeze(-1),
                        content_replacement,
                        boundary.payloads[:, slot],
                    )
                    slot_replaced.append(
                        model.forward_from_boundary(
                            replace(boundary, payloads=changed_payloads)
                        )["logits"]
                    )
                final_delta, _ = model.logits_from_state(
                    full["final_payloads"] - boundary.payloads,
                    boundary.addresses,
                    boundary.query_key,
                )
                zero_payload = model.forward_from_boundary(
                    replace(boundary, payloads=torch.zeros_like(boundary.payloads))
                )
                zero_operation = model.forward_from_boundary(
                    replace(
                        boundary,
                        operation_states=torch.zeros_like(boundary.operation_states),
                    )
                )
                zero_query, _ = model.logits_from_state(
                    full["final_payloads"],
                    boundary.addresses,
                    torch.zeros_like(boundary.query_key),
                )
                reverse = model.forward_from_boundary(
                    replace(
                        boundary,
                        operation_states=boundary.operation_states.flip(1),
                        source_weights=boundary.source_weights.flip(1),
                        target_weights=boundary.target_weights.flip(1),
                        operation_active=boundary.operation_active.flip(1),
                    )
                )
                per_step = [h0_logits]
                for step in range(model.config.operations):
                    logits, _ = model.logits_from_state(
                        trajectory[:, step], boundary.addresses, boundary.query_key
                    )
                    per_step.append(logits)
                auxiliary = full.get("auxiliary")
                if not isinstance(auxiliary, Mapping) or "state_features" not in auxiliary:
                    raise RuntimeError("endpoint has no temporal state features")
                temporal = auxiliary["state_features"]
                decoded = heads.state_decoder(temporal)

            batch_paths = {
                "full": full_logits,
                "h0": h0_logits,
                "core_only": core["logits"],
                "h0_relevant_replace": h0_relevant,
                "global_mean": global_mean,
                "copy_all_query_owner": copied["logits"],
                "final_delta": final_delta,
                "zero_payload": zero_payload["logits"],
                "zero_operation_state_only": zero_operation["logits"],
                "zero_query": zero_query,
                "reverse_operation_order": reverse["logits"],
            }
            for name, value in batch_paths.items():
                paths[name].append(_cpu(value))
            step_logits.append(_cpu(torch.stack(per_step, dim=1)))
            h0_slot_logits.append(_cpu(_slot_logits(model, boundary.payloads)))
            final_slot_logits.append(_cpu(_slot_logits(model, full["final_payloads"])))
            initial_slot_replace_logits.append(_cpu(torch.stack(slot_replaced, dim=1)))
            query_weights.append(_cpu(full["query_weights"]))
            initial_payloads.append(_cpu(boundary.payloads))
            trajectories.append(_cpu(trajectory))
            state_features.append(_cpu(temporal))
            state_logits.append(_cpu(decoded))
            answers.append(batch["answers"].detach().cpu().long())
            presence.append(batch["presence"].detach().cpu().bool())
            content_mask.append(batch["content_mask"].detach().cpu().bool())
            query_owner.append(batch["query_owner"].detach().cpu().long())
            operation_active_target.append(batch["operation_active"].detach().cpu().bool())
            state_values.append(batch["state_values"].detach().cpu().bool())
            state_feature_mask.append(batch["state_feature_mask"].detach().cpu().bool())
            state_step_mask.append(batch["state_step_mask"].detach().cpu().bool())
            families.extend(str(value).upper() for value in batch["families"])
            feature_names.extend([list(value) for value in batch["state_feature_names"]])
            ordered_ids.extend(ids)

    if ordered_ids != [str(value) for value in example_ids]:
        raise RuntimeError("runtime changed caller-provided record order")
    return {
        "example_ids": ordered_ids,
        "families": families,
        "feature_names": feature_names,
        "paths": {name: torch.cat(values, dim=0) for name, values in paths.items()},
        "step_logits": torch.cat(step_logits, dim=0),
        "h0_slot_logits": torch.cat(h0_slot_logits, dim=0),
        "final_slot_logits": torch.cat(final_slot_logits, dim=0),
        "initial_slot_replace_logits": torch.cat(initial_slot_replace_logits, dim=0),
        "query_weights": torch.cat(query_weights, dim=0),
        "initial_payloads": torch.cat(initial_payloads, dim=0),
        "trajectory": torch.cat(trajectories, dim=0),
        "state_features": torch.cat(state_features, dim=0),
        "state_logits": torch.cat(state_logits, dim=0),
        "answers": torch.cat(answers, dim=0),
        "presence": torch.cat(presence, dim=0),
        "content_mask": torch.cat(content_mask, dim=0),
        "query_owner": torch.cat(query_owner, dim=0),
        "operation_active_target": torch.cat(operation_active_target, dim=0),
        "state_values": torch.cat(state_values, dim=0),
        "state_feature_mask": torch.cat(state_feature_mask, dim=0),
        "state_step_mask": torch.cat(state_step_mask, dim=0),
    }


def _accuracy(correct: np.ndarray) -> dict[str, Any]:
    return wilson_interval(int(correct.sum()), int(correct.size))


def _answer_permutation_null(
    logits: np.ndarray,
    answers: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    """Score fixed logits against record-permuted answers (no refit)."""

    if logits.ndim != 2 or answers.ndim != 1 or logits.shape[0] != answers.size:
        raise ValueError("answer permutation inputs have incompatible shapes")
    if answers.size < 2 or np.unique(answers).size < 2:
        return {
            "control_valid": False,
            "reason": "answer_permutation_requires_two_records_and_two_answer_classes",
            "records": int(answers.size),
            "samples": int(samples),
            "seed": int(seed),
        }
    observed_margin = float(np.mean(answer_margin(logits, answers)))
    observed_accuracy = float(np.mean(logits.argmax(axis=1) == answers))
    rng = np.random.default_rng(int(seed))
    margin_draws = np.empty(int(samples), dtype=float)
    accuracy_draws = np.empty(int(samples), dtype=float)
    for draw in range(int(samples)):
        shuffled = answers[rng.permutation(answers.size)]
        margin_draws[draw] = float(np.mean(answer_margin(logits, shuffled)))
        accuracy_draws[draw] = float(np.mean(logits.argmax(axis=1) == shuffled))
    return {
        "control_valid": True,
        "kind": "fixed_logits_record_answer_permutation",
        "records": int(answers.size),
        "samples": int(samples),
        "seed": int(seed),
        "observed_mean_margin": observed_margin,
        "null_mean_margin_mean": float(margin_draws.mean()),
        "null_mean_margin_upper_95": float(np.quantile(margin_draws, 0.95)),
        "margin_right_tail_p_value": float(
            (1 + np.count_nonzero(margin_draws >= observed_margin)) / (int(samples) + 1)
        ),
        "observed_accuracy": observed_accuracy,
        "null_accuracy_mean": float(accuracy_draws.mean()),
        "null_accuracy_upper_95": float(np.quantile(accuracy_draws, 0.95)),
    }


def summarise_paths(
    collection: Mapping[str, Any],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    answers = collection["answers"].numpy().astype(np.int64)
    families = np.asarray(collection["families"], dtype=object)
    ids = list(collection["example_ids"])
    path_logits = {
        name: value.numpy().astype(np.float64)
        for name, value in collection["paths"].items()
    }
    path_margins = {name: np.asarray(answer_margin(value, answers)) for name, value in path_logits.items()}
    decisions: list[dict[str, Any]] = []
    for index, example_id in enumerate(ids):
        decisions.append(
            {
                "example_id": example_id,
                "family": str(families[index]),
                "full_correct": bool(path_logits["full"][index].argmax() == answers[index]),
                "h0_correct": bool(path_logits["h0"][index].argmax() == answers[index]),
                "core_only_correct": bool(path_logits["core_only"][index].argmax() == answers[index]),
                "final_delta_correct": bool(path_logits["final_delta"][index].argmax() == answers[index]),
                "full_margin_minus_h0_margin": float(path_margins["full"][index] - path_margins["h0"][index]),
                "full_margin_minus_core_only_margin": float(path_margins["full"][index] - path_margins["core_only"][index]),
                "full_margin_minus_final_delta_margin": float(path_margins["full"][index] - path_margins["final_delta"][index]),
            }
        )

    family_reports: dict[str, Any] = {}
    steps = collection["step_logits"].numpy().astype(np.float64)
    initial = collection["initial_payloads"].numpy().astype(np.float64)
    trajectory = collection["trajectory"].numpy().astype(np.float64)
    previous = np.concatenate((initial[:, None], trajectory[:, :-1]), axis=1)
    delta = np.linalg.norm(trajectory - previous, axis=-1)
    scale = np.median(np.linalg.norm(initial, axis=-1), axis=1).clip(min=1.0e-8)
    relative = delta / scale[:, None, None]
    active = collection["operation_active_target"].numpy().astype(bool)
    owners = collection["query_owner"].numpy().astype(np.int64)
    h0_slots = collection["h0_slot_logits"].numpy().astype(np.float64)
    final_slots = collection["final_slot_logits"].numpy().astype(np.float64)
    for family in sorted(set(str(value) for value in families.tolist())):
        keep = np.flatnonzero(families == family)
        clusters = [ids[int(index)] for index in keep]
        family_seed = int(bootstrap_seed) + sum(
            (index + 1) * ord(character)
            for index, character in enumerate(family)
        )
        report: dict[str, Any] = {"n": int(keep.size), "paths": {}, "per_step": []}
        for name in PATH_NAMES:
            logits = path_logits[name][keep]
            correct = logits.argmax(axis=1) == answers[keep]
            metric: dict[str, Any] = {
                "accuracy": _accuracy(correct),
                "mean_margin": float(path_margins[name][keep].mean()),
            }
            if name != "full":
                metric["full_minus_path_margin"] = paired_cluster_bootstrap(
                    path_margins["full"][keep],
                    path_margins[name][keep],
                    clusters,
                    seed=bootstrap_seed,
                    samples=bootstrap_replicates,
                )
            report["paths"][name] = metric
        report["h0_query_owner_margin_drop"] = paired_cluster_bootstrap(
            path_margins["h0"][keep],
            path_margins["h0_relevant_replace"][keep],
            clusters,
            seed=family_seed,
            samples=bootstrap_replicates,
        )
        report["answer_permutation_metric_null"] = _answer_permutation_null(
            path_logits["full"][keep],
            answers[keep],
            seed=family_seed,
            samples=bootstrap_replicates,
        )
        for step in range(steps.shape[1]):
            logits = steps[keep, step]
            margin = np.asarray(answer_margin(logits, answers[keep]))
            report["per_step"].append(
                {
                    "state": "h0" if step == 0 else f"h{step}",
                    "accuracy": _accuracy(logits.argmax(axis=1) == answers[keep]),
                    "mean_margin": float(margin.mean()),
                }
            )
        # The classifier consumes the lower bounds, not only the point means.
        # Attach family-level paired bootstrap results to each record row so
        # the final D005 decision remains auditable without refitting.
        for record in keep.tolist():
            decision = decisions[int(record)]
            decision["full_h0_margin_drop_lower"] = report["paths"]["h0"]["full_minus_path_margin"]["lower"]
            decision["h0_relevant_margin_drop_lower"] = report["h0_query_owner_margin_drop"]["lower"]
            decision["full_core_margin_drop_lower"] = report["paths"]["core_only"]["full_minus_path_margin"]["lower"]
            decision["full_final_delta_margin_drop_lower"] = report["paths"]["final_delta"]["full_minus_path_margin"]["lower"]
        family_relative = relative[keep]
        family_active = active[keep]
        active_values = family_relative[family_active]
        inactive_values = family_relative[~family_active]
        owner_updates = np.asarray(
            [
                family_relative[row, :, owners[int(record)]]
                for row, record in enumerate(keep.tolist())
            ]
        )
        report["trajectory_motion"] = {
            "relative_delta_mean": float(family_relative.mean()),
            "active_step_relative_delta_mean": float(active_values.mean()) if active_values.size else 0.0,
            "inactive_step_relative_delta_mean": float(inactive_values.mean()) if inactive_values.size else 0.0,
            "query_owner_relative_delta_mean": float(owner_updates.mean()),
        }
        report["per_slot_answer_decodability"] = {
            "h0_mean_correct_slots": float(
                (h0_slots[keep].argmax(axis=-1) == answers[keep, None]).sum(axis=1).mean()
            ),
            "final_mean_correct_slots": float(
                (final_slots[keep].argmax(axis=-1) == answers[keep, None]).sum(axis=1).mean()
            ),
        }
        family_reports[family] = report
    overall = {
        "n": len(ids),
        "full_accuracy": _accuracy(path_logits["full"].argmax(axis=1) == answers),
        "h0_accuracy": _accuracy(path_logits["h0"].argmax(axis=1) == answers),
        "core_only_accuracy": _accuracy(path_logits["core_only"].argmax(axis=1) == answers),
        "role": "descriptive_summary_only",
    }
    return finite_json(
        {
            "families": family_reports,
            "overall": overall,
            "decision_rows": decisions,
            "path_definitions": {
                "final_delta": "architectural_counterfactual_not_natural_rollout",
                "reverse_operation_order": "same_boundary_operations_reversed",
                "zero_operation_state_only": "operation_state_zeroed_routes_and_activity_retained; not a full route/activity zero",
                "h0_relevant_replace": "H0 query-owner payload replaced by the mean of other present slots; effect is H0 minus H0-relevant",
                "global_mean": "only present payload slots replaced by present-row mean; absent slots unchanged",
                "copy_all_query_owner": "query-owner payload copied only to present slots; absent slots unchanged",
                "answer_permutation_metric_null": "fixed full-path logits scored against record-permuted answers",
                "initial_slot_replace_logits": "each content slot replaced by the leave-one-out mean of other content slots; non-content and absent slots unchanged",
            },
        }
    )


__all__ = ["PATH_NAMES", "collect_paths", "summarise_paths"]
