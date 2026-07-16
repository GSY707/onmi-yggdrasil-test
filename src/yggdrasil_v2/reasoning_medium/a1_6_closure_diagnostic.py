from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from .a1_6_core import RelationAddressedCore
from .a1_6_data import REGISTER_NAMES, VALUE_LABELS
from .a1_6_train import _model_inputs, encode_records, load_c0_checkpoint


def _group_name(operation: dict[str, Any]) -> str:
    return f"{operation['family']}:{operation['source']}->{operation['target']}"


def _oracle_one_step_records(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    flattened: list[dict[str, Any]] = []
    groups: list[str] = []
    for record in records:
        previous = record["start_state"]
        for step, (operation, target_state) in enumerate(zip(record["operations"], record["state_trajectory"])):
            clone = dict(record)
            clone["start_state"] = dict(previous)
            clone["operations"] = [dict(operation)]
            clone["operation_mask"] = [True]
            clone["state_trajectory"] = [dict(target_state)]
            clone["program_length"] = 1
            clone["answer"] = target_state[record["query_register"]]
            clone["answer_index"] = VALUE_LABELS.index(clone["answer"])
            clone["example_id"] = f"{record['example_id']}-oracle-step-{step + 1}"
            flattened.append(clone)
            groups.append(_group_name(operation))
            previous = target_state
    return flattened, groups


@torch.no_grad()
def _diagnose_split(
    model: RelationAddressedCore,
    records: Sequence[dict[str, Any]],
    device: str | torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    free_trajectory_full = free_final_state_full = free_query_correct = 0
    hard_trajectory_full = 0
    first_error = Counter()
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch = encode_records(batch_records, device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        valid = batch["operation_mask"]
        predicted = output["state_logits"].argmax(-1)
        equal = predicted == batch["state_targets"]
        per_step_exact = equal.all(dim=-1)
        free_trajectory_full += int((per_step_exact | ~valid).all(dim=-1).sum())
        final_pred = model.shared_state_head(output["final_slots"]).argmax(-1)
        lengths = valid.sum(dim=-1)
        final_target = batch["state_targets"].gather(
            1,
            (lengths - 1).view(-1, 1, 1).expand(-1, 1, len(REGISTER_NAMES)),
        ).squeeze(1)
        free_final_state_full += int((final_pred == final_target).all(dim=-1).sum())
        free_query_correct += int((output["answer_logits"].argmax(-1) == batch["answer_targets"]).sum())
        for row in range(len(batch_records)):
            error_steps = [step + 1 for step in range(int(lengths[row])) if not bool(per_step_exact[row, step])]
            first_error[str(error_steps[0]) if error_steps else "none"] += 1

        state = model.initial_slots(batch["start_values"])
        hard_all_exact = torch.ones(len(batch_records), dtype=torch.bool, device=device)
        family_latents = model.family_embedding(batch["operation_family"])
        source_latents = model.register_embedding(batch["operation_source"])
        target_latents = model.register_embedding(batch["operation_target"])
        for step in range(valid.shape[1]):
            transition = model.transition(
                family_latents[:, step],
                source_latents[:, step],
                target_latents[:, step],
                state,
                model.register_keys,
            )
            active = valid[:, step]
            candidate = transition["next_slots"]
            decoded = model.shared_state_head(candidate).argmax(-1)
            step_exact = (decoded == batch["state_targets"][:, step]).all(dim=-1)
            hard_all_exact &= step_exact | ~active
            canonical = model.initial_slots(decoded)
            state = torch.where(active.view(-1, 1, 1), canonical, state)
        hard_trajectory_full += int(hard_all_exact.sum())

    one_step_records, groups = _oracle_one_step_records(records)
    group_correct: dict[str, int] = defaultdict(int)
    group_total: dict[str, int] = defaultdict(int)
    one_step_correct = 0
    cursor = 0
    for start in range(0, len(one_step_records), batch_size):
        rows = one_step_records[start : start + batch_size]
        batch = encode_records(rows, device)
        output = model(**_model_inputs(batch), return_trajectory=True)
        exact = (output["state_logits"].argmax(-1)[:, 0] == batch["state_targets"][:, 0]).all(dim=-1)
        one_step_correct += int(exact.sum())
        for offset, value in enumerate(exact.detach().cpu().tolist()):
            group = groups[cursor + offset]
            group_correct[group] += int(value)
            group_total[group] += 1
        cursor += len(rows)
    return {
        "examples": len(records),
        "free_recurrence": {
            "trajectory_full_exact": free_trajectory_full / max(1, len(records)),
            "final_state_full_exact": free_final_state_full / max(1, len(records)),
            "final_query_token_accuracy": free_query_correct / max(1, len(records)),
        },
        "oracle_reset_one_step": {
            "overall_accuracy": one_step_correct / max(1, len(one_step_records)),
            "steps": len(one_step_records),
            "by_family_source_target": {
                group: {"accuracy": group_correct[group] / group_total[group], "correct": group_correct[group], "total": group_total[group]}
                for group in sorted(group_total)
            },
        },
        "predicted_hard_reembed_diagnostic": {
            "trajectory_full_exact": hard_trajectory_full / max(1, len(records)),
            "architecture_path": False,
            "diagnostic_only": True,
        },
        "first_error_step_distribution": dict(sorted(first_error.items(), key=lambda item: (item[0] == "none", item[0]))),
    }


def run_a16_closure_diagnostic(
    checkpoint: Path,
    data_dir: Path,
    output: Path,
    *,
    device: str = "cuda",
    batch_size: int = 128,
) -> dict[str, Any]:
    model = load_c0_checkpoint(checkpoint, device)
    from .a1_6_data import load_a16_records

    split_names = ("test", "length_heldout", "relation_heldout", "causal_core")
    report = {
        "schema_version": "yggdrasil.v2-a1.6.closure-diagnostic.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_mutated": False,
        "hard_reembed_is_diagnostic_only": True,
        "splits": {
            split: _diagnose_split(model, load_a16_records(data_dir, split), device, batch_size=batch_size)
            for split in split_names
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report

