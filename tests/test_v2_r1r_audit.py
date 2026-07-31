from __future__ import annotations

from copy import deepcopy

from yggdrasil_v2.r1_revalidation.audit import (
    _overlap_gate,
    evaluate_claim_gate,
    evaluate_composition_gate,
    evaluate_heuristics,
    evaluate_integrity_gate,
    evaluate_language_gate,
    evaluate_surface_only_gate,
    unigram_naive_bayes_accuracy,
)
from yggdrasil_v2.r1_revalidation.cps import generate_cps_split
from yggdrasil_v2.r1_revalidation.ere import generate_ere_causal_pairs, generate_ere_split
from yggdrasil_v2.r1_revalidation.schema import assert_model_view


def test_audit_rejects_undeclared_cross_split_overlap() -> None:
    row = generate_ere_split(
        split="validation",
        count=1,
        seed=91,
        design_doc_sha256="d" * 64,
        command="pytest",
        environment={},
    )[0]
    duplicate = deepcopy(row)
    duplicate["split"] = "composition_ood"
    report = _overlap_gate({"ere": {"validation": [row], "composition_ood": [duplicate]}})
    assert report["passed"] is False
    assert report["undeclared_semantic_overlap"]
    assert report["undeclared_surface_overlap"]


def test_unigram_naive_bayes_triggers_on_an_artificial_leak() -> None:
    rows = [
        {
            "example_id": f"leak-{index:02d}",
            "source_text": "leak_a neutral text" if (index // 2) % 2 == 0 else "leak_b neutral text",
            "answer_index": 0 if (index // 2) % 2 == 0 else 1,
            "valid_choice_mask": [True] * 9,
        }
        for index in range(40)
    ]
    report = evaluate_heuristics(rows)
    assert unigram_naive_bayes_accuracy(rows) > report["chance"] + 0.10
    assert report["metrics"]["full_text_unigram_nb"] > report["max_allowed"]


def test_audit_model_view_rejects_nested_answer() -> None:
    try:
        assert_model_view({"reasoning_budget": 2, "valid_choice_mask": [True] * 9, "certificate": {"answer": 1}})
    except ValueError:
        pass
    else:
        raise AssertionError("nested answer field was accepted by the model-view audit")


def _provenance_kwargs() -> dict:
    return {"design_doc_sha256": "d" * 64, "command": "pytest", "environment": {}}


def test_gate_rejects_ere_first_set_literal_answer_shortcut() -> None:
    rows = generate_ere_split(split="train", count=12, seed=101, **_provenance_kwargs())
    forged = []
    for row in rows:
        item = deepcopy(row)
        value = item["audit_answer_semantic"]
        label = item["audit_label_mapping"][value]
        item["source_text"] += f"\nA SET literal is {value}; its local label is {label}."
        forged.append(item)
    report = evaluate_surface_only_gate({"ere": {"train": forged}})
    assert report["passed"] is False
    assert report["metrics"]["first_set_literal_label"] > report["max_allowed"]


def test_gate_rejects_cps_candidate_position_shortcut() -> None:
    rows = generate_cps_split(split="train", count=12, seed=103, **_provenance_kwargs())
    forged = []
    for row in rows:
        item = deepcopy(row)
        candidate_label = item["audit_label_mapping"]["candidate_0"]
        none_label = item["audit_label_mapping"]["NONE"]
        first_action = item["program_ast"]["actions"][0]["name"]
        item["source_text"] += (
            f"\nCandidate 1 starts with {first_action}; the first defined action is {first_action}."
            f" Candidate 1 is {candidate_label}; NONE is {none_label}."
        )
        item["answer_index"] = ord(candidate_label) - ord("A")
        item["audit_answer_semantic"] = "candidate_0"
        forged.append(item)
    report = evaluate_surface_only_gate({"cps": {"train": forged}})
    assert report["passed"] is False
    assert report["metrics"]["first_action_definition_or_none"] > report["max_allowed"]


def test_gate_rejects_balanced_but_reversed_cps_prefix_claims() -> None:
    rows = generate_cps_split(split="train", count=4, seed=107, **_provenance_kwargs())
    forged = []
    for row in rows:
        item = deepcopy(row)
        for claim in item["training_claims"]:
            claim["label"] = not bool(claim["label"])
        forged.append(item)
    report = evaluate_claim_gate({"cps": {"train": forged}})
    assert report["passed"] is False
    assert report["claim_truth_accuracy"] < 1.0
    assert report["balanced"] is True


def test_gate_rejects_train_composition_signature_reuse() -> None:
    row = generate_cps_split(split="train", count=1, seed=109, **_provenance_kwargs())[0]
    duplicate = deepcopy(row)
    duplicate["split"] = "composition_ood"
    report = evaluate_composition_gate(
        {"cps": {"train": [row], "validation": [], "composition_ood": [duplicate]}}
    )
    assert report["passed"] is False
    assert report["overlap"]


def test_gate_rejects_language_ood_that_only_changes_opening() -> None:
    row = generate_ere_split(split="validation", count=1, seed=113, **_provenance_kwargs())[0]
    ood = deepcopy(row)
    ood["split"] = "language_ood"
    body = ood["source_text"].split("\n", 1)[1]
    ood["source_text"] = "A different opening only.\n" + body
    report = evaluate_language_gate({"ere": {"validation": [row], "language_ood": [ood]}})
    assert report["passed"] is False
    assert report["body_equal"]


def test_gate_rejects_forged_token_count_and_changed_path() -> None:
    rows = generate_ere_causal_pairs(pair_count=1, seed=127, seen_semantic=set(), seen_surface=set(), **_provenance_kwargs())
    forged_token = deepcopy(rows[0])
    forged_token["qwen_token_count"] = int(forged_token["qwen_token_count"]) + 1
    token_report = evaluate_integrity_gate([forged_token])
    assert token_report["passed"] is False
    assert token_report["token_count_mismatches"]

    forged_path = deepcopy(rows)
    forged_path[0]["causal_certificate"]["changed_path"] = ["events", 0, "primitives", 0]
    path_report = evaluate_integrity_gate(forged_path)
    assert path_report["passed"] is False
    assert path_report["certificate_mismatches"]
