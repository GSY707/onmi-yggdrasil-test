from __future__ import annotations

"""CPS generation: construct P* first, then derive hard negatives."""

from copy import deepcopy
import random
from typing import Any, Mapping

from .render import OOD_CPS_TEMPLATES, TRAIN_CPS_TEMPLATES, contract_token_count, render_cps, semantic_fingerprint, surface_fingerprint
from .schema import GENERATOR_VERSION, LABELS, SCHEMA_VERSION
from .simulator import select_cps_answer, verify_cps_plan
from .symbols import NonceFactory, balanced_label_index, balanced_label_mapping, balanced_pair_mapping, label_index, label_permutation, valid_choice_mask


CPS_SPLITS = (
    "train",
    "validation",
    "composition_ood",
    "horizon_ood",
    "distractor_ood",
    "language_ood",
    "causal_pairs",
)


def _condition(kind: str, **values: Any) -> dict[str, Any]:
    return {"kind": kind, **values}


def _base_ast(seed: int, split: str, index: int) -> dict[str, Any]:
    rng = random.Random(seed)
    factory = NonceFactory(rng, set())
    facts = factory.names(7)
    resource = factory.name()
    extra_resources = factory.names(rng.randint(0, 2))
    action_names = factory.names(8)
    unlock, refill, fast, slow, bad, bonus, decoy_a, decoy_b = action_names
    unlocked, goal, final_ok, spare, unrelated, extra_fact_a, extra_fact_b = facts
    initial_facts = [final_ok, spare]
    for fact in (extra_fact_a, extra_fact_b):
        if rng.random() < 0.5:
            initial_facts.append(fact)
    initial_facts = sorted(set(initial_facts))
    initial_resource = rng.choice((0, 1))
    refill_gain = rng.choice((2, 3, 4))
    unlock_cost = rng.randint(1, 3)
    refill_cost = rng.randint(2, 5)
    fast_cost = rng.randint(3, 5)
    slow_cost = fast_cost + rng.randint(2, 5)
    bad_cost = rng.randint(2, 6)
    bonus_cost = rng.randint(1, 5)
    budget = unlock_cost + refill_cost + slow_cost + 2 + rng.randint(0, 4)
    initial = {
        "attributes": {},
        "relations": {},
        "facts": initial_facts,
        "resources": {resource: initial_resource, **{name: rng.randint(0, 6) for name in extra_resources}},
    }
    actions = [
        {
            "name": unlock,
            "preconditions": [_condition("fact_false", fact=unlocked)],
            "effects": [{"kind": "add_fact", "fact": unlocked}],
            "cost": unlock_cost,
        },
        {
            "name": refill,
            "preconditions": [_condition("fact_true", fact=unlocked)],
            "effects": [{"kind": "resource_delta", "resource": resource, "delta": refill_gain}],
            "cost": refill_cost,
        },
        {
            "name": fast,
            "preconditions": [_condition("fact_true", fact=unlocked), _condition("resource_at_least", resource=resource, amount=2)],
            "effects": [{"kind": "resource_delta", "resource": resource, "delta": -2}, {"kind": "add_fact", "fact": goal}],
            "cost": fast_cost,
        },
        {
            "name": slow,
            "preconditions": [_condition("fact_true", fact=unlocked), _condition("resource_at_least", resource=resource, amount=2)],
            "effects": [{"kind": "resource_delta", "resource": resource, "delta": -2}, {"kind": "add_fact", "fact": goal}],
            "cost": slow_cost,
        },
        {
            "name": bad,
            "preconditions": [_condition("fact_true", fact=unlocked), _condition("resource_at_least", resource=resource, amount=2)],
            "effects": [
                {"kind": "resource_delta", "resource": resource, "delta": -2},
                {"kind": "add_fact", "fact": goal},
                {"kind": "remove_fact", "fact": final_ok},
            ],
            "cost": bad_cost,
        },
        {
            "name": bonus,
            "preconditions": [_condition("fact_true", fact=unlocked)],
            "effects": [{"kind": "add_fact", "fact": unrelated}],
            "cost": bonus_cost,
        },
    ]
    candidates = [
        {"plan": [unlock, refill, fast]},
        {"plan": [unlock, refill, slow]},
        {"plan": [refill, unlock, fast]},
        {"plan": [unlock, fast]},
        {"plan": [unlock, refill, bad]},
    ]
    if split == "distractor_ood":
        for action_name, fact in ((decoy_a, unrelated), (decoy_b, spare)):
            actions.append(
                {
                    "name": action_name,
                    "preconditions": [_condition("fact_false", fact=fact)],
                    "effects": [{"kind": "add_fact", "fact": fact}],
                    "cost": 7 + len(actions),
                }
            )
        candidates.extend(
            [
                {"plan": [unlock, refill, decoy_a]},
                {"plan": [unlock, decoy_b]},
                {"plan": [unlock, refill]},
            ]
        )
    return {
        "kind": "cps",
        "initial_state": initial,
        "actions": actions,
        "goal": [_condition("fact_true", fact=goal)],
        "final_constraints": [_condition("fact_true", fact=final_ok)],
        "budget": budget,
        "candidates": candidates,
        "composition_features": ["unlock_resource", "final_constraint", "cost_budget"],
        "causal_target": {"kind": "cost", "action": fast, "original": fast_cost, "alternate": slow_cost + 2},
    }


def _horizon_ast(seed: int, index: int) -> dict[str, Any]:
    rng = random.Random(seed)
    factory = NonceFactory(rng, set())
    chain_length = 6 + index % 4
    stages = factory.names(chain_length + 1)
    action_names = factory.names(chain_length + 1)
    actions: list[dict[str, Any]] = []
    for step in range(chain_length):
        actions.append(
            {
                "name": action_names[step],
                "preconditions": [_condition("fact_true", fact=stages[step])],
                "effects": [{"kind": "remove_fact", "fact": stages[step]}, {"kind": "add_fact", "fact": stages[step + 1]}],
                "cost": 1 + rng.randint(0, 1),
            }
        )
    slow_last = action_names[chain_length]
    actions.append(
        {
            "name": slow_last,
            "preconditions": [_condition("fact_true", fact=stages[-2])],
            "effects": [{"kind": "remove_fact", "fact": stages[-2]}, {"kind": "add_fact", "fact": stages[-1]}],
            "cost": 3,
        }
    )
    p_star = [action["name"] for action in actions[:chain_length]]
    suboptimal = [*p_star[:-1], slow_last]
    return {
        "kind": "cps",
        "initial_state": {
            "attributes": {},
            "relations": {},
            "facts": sorted([stages[0], *([factory.name()] if rng.random() < 0.5 else [])]),
            "resources": {},
        },
        "actions": actions,
        "goal": [_condition("fact_true", fact=stages[-1])],
        "final_constraints": [],
        "budget": sum(action["cost"] for action in actions[:chain_length]) + 3,
        "candidates": [
            {"plan": p_star},
            {"plan": suboptimal},
            {"plan": list(reversed(p_star))},
            {"plan": p_star[:-2]},
            {"plan": [*p_star[:-1], p_star[0]]},
        ],
        "composition_features": ["long_prerequisite_chain", "delayed_effect", "valid_suboptimal"],
        "causal_target": {
            "kind": "cost",
            "action": actions[chain_length - 1]["name"],
            "original": actions[chain_length - 1]["cost"],
            "alternate": actions[chain_length - 1]["cost"] + 3,
        },
    }


def _none_ast(seed: int, split: str, index: int) -> dict[str, Any]:
    ast = _base_ast(seed, split, index)
    actions = {action["name"]: action for action in ast["actions"]}
    names = list(actions)
    unlock, refill, fast, _slow, bad, bonus = names[:6]
    ast["candidates"] = [
        {"plan": [refill, unlock, fast]},
        {"plan": [unlock, fast]},
        {"plan": [unlock, refill, bad]},
        {"plan": [unlock, refill, bonus]},
        {"plan": [unlock, refill]},
    ]
    if split == "distractor_ood":
        decoy_a, decoy_b = names[6:8]
        ast["candidates"].extend(
            [
                {"plan": [unlock, refill, decoy_a]},
                {"plan": [unlock, decoy_b]},
                {"plan": [unlock, refill, bonus, decoy_a]},
            ]
        )
    resource = next(iter(ast["initial_state"]["resources"]))
    original = int(ast["initial_state"]["resources"][resource])
    ast["causal_target"] = {"kind": "resource", "resource": resource, "original": original, "alternate": 3 if original != 3 else 4}
    return ast


def mutate_cps_ast(ast: Mapping[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(dict(ast))
    target = mutated["causal_target"]
    if target["kind"] == "cost":
        for action in mutated["actions"]:
            if action["name"] == target["action"]:
                current = int(action["cost"])
                action["cost"] = int(target["alternate"] if current == int(target["original"]) else target["original"])
                break
        else:
            raise ValueError("causal cost target action not found")
    elif target["kind"] == "resource":
        current = int(mutated["initial_state"]["resources"][target["resource"]])
        mutated["initial_state"]["resources"][target["resource"]] = int(target["alternate"] if current == int(target["original"]) else target["original"])
    else:
        raise ValueError(f"unknown CPS causal target kind: {target['kind']}")
    return mutated


def _claims(ast: Mapping[str, Any], results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for candidate_index, result in enumerate(results):
        plan = ast["candidates"][candidate_index]["plan"]
        for prefix in range(len(plan) + 1):
            positive = bool(result["valid"]) if prefix == len(plan) else prefix == 0
            claims.extend(
                [
                    {"prefix": prefix, "candidate": candidate_index, "kind": "candidate_status", "text": f"candidate_{candidate_index} is legal at prefix {prefix}", "label": positive},
                    {"prefix": prefix, "candidate": candidate_index, "kind": "candidate_status", "text": f"candidate_{candidate_index} is not legal at prefix {prefix}", "label": not positive},
                ]
            )
    return claims


def _record_from_ast(
    ast: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    index: int,
    example_id: str,
    template_id: str,
    design_doc_sha256: str,
    command: str,
    environment: Mapping[str, Any],
    pair_id: str | None = None,
    pair_role: str | None = None,
    label_targets: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    selected = select_cps_answer(ast)
    if not selected["unique_optimum"]:
        raise ValueError("CPS generator produced a non-unique optimum")
    answer_option = "NONE" if selected["answer"] is None else f"candidate_{selected['answer']}"
    options = [f"candidate_{index}" for index in range(len(ast["candidates"]))] + ["NONE"]
    mapping = label_permutation(options, random.Random(seed + index * 9973), target_labels=label_targets or {answer_option: "A"})
    source_text = render_cps(ast, mapping, template_id)
    token_count = contract_token_count(source_text)
    if token_count > 1024:
        raise ValueError(f"CPS generator rejected a record over 1024 tokens: {token_count}")
    alternate_ast = mutate_cps_ast(ast)
    alternate = select_cps_answer(alternate_ast)
    alternate_option = "NONE" if alternate["answer"] is None else f"candidate_{alternate['answer']}"
    if alternate_option == answer_option:
        raise ValueError("CPS causal mutation did not flip the semantic answer")
    record = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "example_id": example_id,
        "family": "cps",
        "split": split,
        "source_text": source_text,
        "token_count": token_count,
        "reasoning_budget": min(24, max(len(candidate["plan"]) for candidate in ast["candidates"]) + 2),
        "valid_choice_mask": valid_choice_mask(mapping),
        "answer_index": label_index(mapping[answer_option]),
        "semantic_fingerprint": semantic_fingerprint(ast),
        "surface_fingerprint": surface_fingerprint(source_text),
        "pair_id": pair_id,
        "pair_role": pair_role,
        "template_id": template_id,
        "audit_label_mapping": mapping,
        "audit_answer_semantic": answer_option,
        "program_ast": deepcopy(dict(ast)),
        "teacher_trace": [result["trace"] for result in selected["results"]],
        "training_claims": _claims(ast, selected["results"]),
        "causal_certificate": {
            "kind": ast["causal_target"]["kind"],
            "changed_path": ["actions", ast["causal_target"].get("action")] if ast["causal_target"]["kind"] == "cost" else ["initial_state", "resources", ast["causal_target"]["resource"]],
            "before_answer": answer_option,
            "after_answer": alternate_option,
            "specified_change": deepcopy(dict(ast["causal_target"])),
            "valid_suboptimal_present": any(result["valid"] and result["total_cost"] > min((item[1]["total_cost"] for item in enumerate(selected["results"]) if item[1]["valid"]), default=10**9) for result in selected["results"]),
            "pair_id": pair_id,
        },
        "provenance": {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "design_doc_sha256": design_doc_sha256,
            "command": command,
            "environment": dict(environment),
        },
    }
    return record


def _pick_ast(seed: int, split: str, index: int) -> dict[str, Any]:
    if split == "horizon_ood":
        return _horizon_ast(seed, index)
    if split == "causal_pairs":
        return _base_ast(seed, split, index)
    return _none_ast(seed, split, index) if index % 6 == 0 else _base_ast(seed, split, index)


def generate_cps_split(
    *,
    split: str,
    count: int,
    seed: int,
    design_doc_sha256: str,
    command: str,
    environment: Mapping[str, Any],
    seen_semantic: set[str] | None = None,
    seen_surface: set[str] | None = None,
) -> list[dict[str, Any]]:
    if split not in CPS_SPLITS or split == "causal_pairs":
        raise ValueError(f"generate_cps_split handles non-pair splits only: {split}")
    seen_semantic = seen_semantic if seen_semantic is not None else set()
    seen_surface = seen_surface if seen_surface is not None else set()
    records: list[dict[str, Any]] = []
    for index in range(count):
        if index % 256 == 0:
            print(f"starting cps/{split}: {index + 1}-{min(count, index + 256)}", flush=True)
        for attempt in range(10_000):
            record_seed = seed + index * 104729 + attempt * 7919
            ast = _pick_ast(record_seed, split, index)
            template_pool = sorted(OOD_CPS_TEMPLATES) if split == "language_ood" else sorted(TRAIN_CPS_TEMPLATES)
            template_id = template_pool[index % len(template_pool)]
            selected = select_cps_answer(ast)
            answer_option = "NONE" if selected["answer"] is None else f"candidate_{selected['answer']}"
            options = [f"candidate_{candidate_index}" for candidate_index in range(len(ast["candidates"]))] + ["NONE"]
            record = _record_from_ast(
                ast,
                seed=record_seed,
                split=split,
                index=index,
                example_id=f"cps-{split}-{index:05d}",
                template_id=template_id,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
                label_targets=balanced_label_mapping(
                    options,
                    answer_option,
                    balanced_label_index(index, seed, len(options)),
                    record_index=index,
                    salt=seed,
                ),
            )
            if record["semantic_fingerprint"] not in seen_semantic and record["surface_fingerprint"] not in seen_surface:
                seen_semantic.add(record["semantic_fingerprint"])
                seen_surface.add(record["surface_fingerprint"])
                records.append(record)
                if (index + 1) % 256 == 0:
                    print(f"generated cps/{split}: {index + 1}", flush=True)
                break
            if attempt and attempt % 1000 == 0:
                print(f"cps/{split} index {index}: uniqueness attempts {attempt}", flush=True)
        else:
            raise RuntimeError(f"unable to generate unique CPS {split} record {index}")
    return records


def generate_cps_causal_pairs(
    *,
    pair_count: int,
    seed: int,
    design_doc_sha256: str,
    command: str,
    environment: Mapping[str, Any],
    seen_semantic: set[str],
    seen_surface: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair_index in range(pair_count):
        pair_id = f"cps-causal-{pair_index:05d}"
        pair_seed = seed + pair_index * 104729
        for attempt in range(10_000):
            ast = _base_ast(pair_seed + attempt * 7919, "causal_pairs", pair_index)
            if pair_index % 6 == 0:
                ast = _none_ast(pair_seed + attempt * 7919, "causal_pairs", pair_index)
            mutated = mutate_cps_ast(ast)
            base = select_cps_answer(ast)
            flip = select_cps_answer(mutated)
            base_option = "NONE" if base["answer"] is None else f"candidate_{base['answer']}"
            flip_option = "NONE" if flip["answer"] is None else f"candidate_{flip['answer']}"
            if base_option == flip_option:
                continue
            options = [f"candidate_{index}" for index in range(len(ast["candidates"]))] + ["NONE"]
            mapping = balanced_pair_mapping(options, base_option, flip_option, pair_index, salt=seed)
            template_id = sorted(TRAIN_CPS_TEMPLATES)[pair_index % len(TRAIN_CPS_TEMPLATES)]
            base_record = _record_from_ast(
                ast,
                seed=pair_seed,
                split="causal_pairs",
                index=pair_index * 2,
                example_id=f"cps-causal-{pair_index:05d}-base",
                template_id=template_id,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
                pair_id=pair_id,
                pair_role="base",
                label_targets=mapping,
            )
            flip_record = _record_from_ast(
                mutated,
                seed=pair_seed,
                split="causal_pairs",
                index=pair_index * 2 + 1,
                example_id=f"cps-causal-{pair_index:05d}-flip",
                template_id=template_id,
                design_doc_sha256=design_doc_sha256,
                command=command,
                environment=environment,
                pair_id=pair_id,
                pair_role="flip",
                label_targets=mapping,
            )
            if any(item["semantic_fingerprint"] in seen_semantic or item["surface_fingerprint"] in seen_surface for item in (base_record, flip_record)):
                continue
            for item in (base_record, flip_record):
                seen_semantic.add(item["semantic_fingerprint"])
                seen_surface.add(item["surface_fingerprint"])
            records.extend((base_record, flip_record))
            break
        else:
            raise RuntimeError(f"unable to generate CPS causal pair {pair_id}")
    return records
