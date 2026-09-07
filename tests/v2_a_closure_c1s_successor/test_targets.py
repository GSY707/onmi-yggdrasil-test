from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from yggdrasil_v2.r1_revalidation.production.renderer import render_source
from yggdrasil_v2.v2_a.closure_c1s_successor.targets import (
    ADDRESS_POLICY_NAME,
    S1_SELECTION_SALT,
    TARGET_BANK_SCHEMA,
    audit_target_bank,
    build_split_ledger,
    extract_choice_slots,
    materialize_target,
    materialize_target_bank,
    select_s1_ids,
    select_s2_ids,
)


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        del kwargs
        offsets = []
        ids = []
        for index, token in enumerate(text.split()):
            start = text.find(token, 0 if not offsets else offsets[-1][1])
            offsets.append((start, start + len(token)))
            ids.append(index + 10)
        return {"input_ids": ids, "offset_mapping": offsets}


def _cps_record(
    example_id: str = "cps-0",
    *,
    none_label: str = "C",
    split: str = "train",
    candidate_count: int = 2,
) -> dict:
    if candidate_count == 2:
        ast = {
            "actions": [
                {
                    "name": "make",
                    "cost": 1,
                    "preconditions": [{"kind": "fact_true", "fact": "seed"}],
                    "effects": [{"kind": "add_fact", "fact": "goal"}],
                },
                {
                    "name": "blocked",
                    "cost": 2,
                    "preconditions": [{"kind": "fact_true", "fact": "missing"}],
                    "effects": [{"kind": "add_fact", "fact": "goal"}],
                },
            ],
            "initial_state": {"attributes": {}, "relations": {}, "facts": ["seed"], "resources": {}},
            "candidates": [{"plan": ["make"]}, {"plan": ["blocked"]}],
            "goal": [{"kind": "fact_true", "fact": "goal"}],
            "final_constraints": [],
            "budget": 3,
        }
        available = [label for label in "ABCDEFGHI" if label != none_label]
        mapping = {available[-1]: "candidate:0", available[0]: "candidate:1", none_label: "none"}
        semantic_answer = "candidate:0"
        answer_label = available[-1]
    else:
        assert candidate_count == 8
        ast = {
            "actions": [
                {
                    "name": "blocked",
                    "cost": 1,
                    "preconditions": [{"kind": "fact_true", "fact": "missing"}],
                    "effects": [{"kind": "add_fact", "fact": "goal"}],
                }
            ],
            "initial_state": {"attributes": {}, "relations": {}, "facts": [], "resources": {}},
            "candidates": [{"plan": ["blocked"]} for _ in range(8)],
            "goal": [{"kind": "fact_true", "fact": "goal"}],
            "final_constraints": [],
            "budget": 1,
        }
        candidate_labels = [label for label in "ABCDEFGHI" if label != none_label]
        mapping = {label: f"candidate:{index}" for index, label in enumerate(candidate_labels)}
        mapping[none_label] = "none"
        semantic_answer = "none"
        answer_label = none_label
    source_text = render_source("CPS", ast, mapping, "plain_v1")
    return {
        "example_id": example_id,
        "family": "CPS",
        "split": split,
        "source_text": source_text,
        "program_ast": ast,
        "label_mapping": mapping,
        "semantic_answer": semantic_answer,
        "answer_index": "ABCDEFGHI".index(answer_label),
    }


def _ere_record(
    example_id: str = "ere-0",
    *,
    split: str = "train",
    reverse_entities: bool = False,
    query_entity: str = "bob",
) -> dict:
    attributes = [("alice", {"color": "red"}), ("bob", {"color": "blue"})]
    if reverse_entities:
        attributes.reverse()
    ast = {
        "rules": {
            "copyit": {
                "params": ["x", "y"],
                "primitives": [
                    {"op": "COPY", "attribute": "color", "source": "$arg:x", "target": "$arg:y"}
                ],
            }
        },
        "initial_state": {
            "attributes": dict(attributes),
            "relations": {},
            "facts": [],
            "resources": {},
        },
        "events": [{"rule": "copyit", "arguments": {"x": "alice", "y": "bob"}}],
        "query": {"kind": "attribute", "entity": query_entity, "attribute": "color"},
    }
    mapping = {"A": "value:red", "I": "value:blue"}
    return {
        "example_id": example_id,
        "family": "ERE",
        "split": split,
        "source_text": render_source("ERE", ast, mapping, "plain_v1"),
        "program_ast": ast,
        "label_mapping": mapping,
        "semantic_answer": "value:red",
        "answer_index": 0,
    }


def _selection_row(example_id: str, family: str = "ERE", split: str = "train") -> dict:
    return {"example_id": example_id, "family": family, "split": split}


def test_extracts_raw_choice_order_but_cps_addresses_use_present_label_rank() -> None:
    record = _cps_record(none_label="C")
    choices = extract_choice_slots(record["source_text"])
    assert set(choices["slot_to_label"]) == {"A", "C", "I"}

    target = materialize_target(record, TinyTokenizer(), strict=True)
    assert target["address_policy"]["name"] == ADDRESS_POLICY_NAME
    assert target["address_policy"]["derivation"] == "present_raw_label_rank"
    assert [obj["raw_label"] for obj in target["objects"]] == ["A", None, "I"]
    assert [obj["name"] for obj in target["objects"]] == ["candidate:1", "decision", "candidate:0"]
    assert target["query_owner"] == 1


def test_cps_decision_slot_is_not_fixed_and_decision_has_no_answer_label() -> None:
    first = materialize_target(_cps_record("c0", none_label="A"), TinyTokenizer(), strict=True)
    second = materialize_target(_cps_record("c1", none_label="I"), TinyTokenizer(), strict=True)
    assert first["query_owner"] == 0
    assert second["query_owner"] == 2
    assert first["objects"][first["query_owner"]]["kind"] == "cps_decision"
    assert second["objects"][second["query_owner"]]["kind"] == "cps_decision"
    assert first["objects"][first["query_owner"]]["raw_label_index"] == -1
    assert second["objects"][second["query_owner"]]["raw_label_index"] == -1


def test_ere_uses_learnable_public_entity_order_and_true_temporal_targets() -> None:
    target = materialize_target(_ere_record(), TinyTokenizer(), strict=True)
    assert [obj["name"] for obj in target["objects"]] == ["alice", "bob"]
    assert target["address_policy"]["derivation"] == "public_entity_name_nfkc_casefold_lexical"
    assert target["address_policy"]["normalization"] == "NFKC_casefold"
    assert [obj["raw_label_index"] for obj in target["objects"]] == [-1, -1]
    assert target["query_owner"] == 1
    assert target["query_answer_independent"] is True
    assert target["operation_active"] == [1] + [0] * 9
    assert target["source_owner"][0] == 0
    assert target["target_owner"][0] == 1
    assert len(target["state_values"]) == 10
    assert all(len(step) == 8 and all(len(slot) == 8 for slot in step) for step in target["state_values"])
    assert len(target["state_feature_names"]) == 8
    assert target["irrelevant_owner"] == 2


def test_ere_address_construction_is_source_order_and_query_independent() -> None:
    baseline = materialize_target(_ere_record("e0", query_entity="bob"), TinyTokenizer(), strict=True)
    reordered = materialize_target(
        _ere_record("e1", reverse_entities=True, query_entity="bob"), TinyTokenizer(), strict=True
    )
    other_query = materialize_target(
        _ere_record("e2", reverse_entities=True, query_entity="alice"), TinyTokenizer(), strict=True
    )
    assert [obj["name"] for obj in baseline["objects"]] == [obj["name"] for obj in reordered["objects"]]
    assert [obj["name"] for obj in baseline["objects"]] == [obj["name"] for obj in other_query["objects"]]
    assert baseline["query_owner"] == reordered["query_owner"] == 1
    assert other_query["query_owner"] == 0
    assert all(row["address_policy"]["answer_independent"] for row in (baseline, reordered, other_query))


def test_ere_nfkc_casefold_collision_fails_closed() -> None:
    row = _ere_record("collision")
    ast = deepcopy(row["program_ast"])
    ast["initial_state"]["attributes"] = {
        "A": {"color": "red"},
        "a": {"color": "blue"},
    }
    ast["events"] = [{"rule": "copyit", "arguments": {"x": "A", "y": "a"}}]
    ast["query"] = {"kind": "attribute", "entity": "a", "attribute": "color"}
    row["program_ast"] = ast
    row["source_text"] = render_source("ERE", ast, row["label_mapping"], "plain_v1")
    with pytest.raises(ValueError, match=r"collide after NFKC\+casefold"):
        materialize_target(row, TinyTokenizer(), strict=True)


def test_raw_nine_choice_cps_ood_is_behavior_only_and_does_not_crash() -> None:
    source = _cps_record("cps-ood", none_label="I", split="distractor_ood", candidate_count=8)
    target = materialize_target(source, TinyTokenizer(), strict=True)
    assert len(target["objects"]) == 9
    assert target["query_owner"] == 8
    assert target["answer_index"] == 8
    assert target["mechanism_supported"] is False
    assert target["operation_active"] == [0] * 10
    assert target["state_step_mask"] == [0] * 10
    assert not any(target["content_mask"])
    assert not any(value for step in target["state_feature_mask"] for slot in step for value in slot)
    bank = materialize_target_bank([source], TinyTokenizer(), strict=True)
    report = audit_target_bank(bank, expected_count=1)
    assert report["passed"] is True
    assert report["unsupported"] == 1
    assert report["unsupported_only_ood"] is True


def test_unqualified_behavior_only_target_is_rejected_outside_ood() -> None:
    row = {
        "example_id": "bad",
        "family": "ERE",
        "split": "train",
        "source_text": "Header\nRESPONSE KEY\nA means TRUE\nB means FALSE",
        "answer_label": "A",
    }
    with pytest.raises(ValueError, match="only on an OOD split"):
        materialize_target(row, TinyTokenizer())


def test_s1_and_s2_are_srw_train_only_disjoint_and_ledger_keeps_formal_cells() -> None:
    records = [_selection_row(f"ere-{i}", "ERE") for i in range(3600)]
    records += [_selection_row(f"cps-{i}", "CPS") for i in range(3600)]
    records += [_selection_row("ere-validation", "ERE", "validation")]
    records += [_selection_row("cps-causal", "CPS", "causal_pairs")]
    s1 = select_s1_ids(records)
    assert len(s1) == 32
    s2 = select_s2_ids(records, s1_ids=s1)
    assert set(s1).isdisjoint(s2["train"])
    assert set(s1).isdisjoint(s2["eval"])
    assert set(s2["train"]).isdisjoint(s2["eval"])
    ledger = build_split_ledger(records, s1_ids=s1, s2=s2)
    assert ledger["formal"]["validation"] == 1
    assert ledger["formal"]["causal"] == 1
    assert ledger["coverage_boundary"]["validation_ood_causal_never_used_for_discovery"] is True
    assert hashlib.sha256(f"{S1_SELECTION_SALT}|ere-0".encode()).hexdigest()


def test_bank_audit_reports_owner_answer_majority_ceiling_and_rejects_proxy() -> None:
    base = materialize_target(_ere_record(), TinyTokenizer(), strict=True)
    records = {}
    for index in range(128):
        row = deepcopy(base)
        row["example_id"] = f"ere-{index:04d}"
        owner = index % 2
        answer = 0 if owner == 0 else 8
        row["query_owner"] = owner
        row["query"]["owner_slot"] = owner
        row["query"]["query_owner"] = owner
        row["ownership"]["query_owner"] = owner
        row["answer_index"] = answer
        row["answer_target"]["answer_index"] = answer
        row["answer_target"]["answer_label"] = "A" if answer == 0 else "I"
        records[row["example_id"]] = row
    report = audit_target_bank({"schema_version": TARGET_BANK_SCHEMA, "records": records}, expected_count=128)
    cell = report["owner_only_majority_accuracy"]["ERE/train"]
    assert cell["sufficient"] is True
    assert cell["accuracy"] == 1.0
    assert report["owner_answer_contingency"]["ERE/train"]["0"] == {"0": 64}
    assert report["owner_only_proxy_passed"] is False
    assert any(value.startswith("owner_answer_proxy:ERE/train") for value in report["failures"])


def test_bank_audit_applies_owner_answer_ceiling_to_validation_too() -> None:
    base = materialize_target(_ere_record(), TinyTokenizer(), strict=True)
    records = {}
    for index in range(128):
        row = deepcopy(base)
        row["example_id"] = f"ere-validation-proxy-{index:04d}"
        row["split"] = "validation"
        owner = index % 2
        answer = 0 if owner == 0 else 8
        row["query_owner"] = owner
        row["query"]["owner_slot"] = owner
        row["query"]["query_owner"] = owner
        row["ownership"]["query_owner"] = owner
        row["answer_index"] = answer
        row["answer_target"]["answer_index"] = answer
        row["answer_target"]["answer_label"] = "A" if answer == 0 else "I"
        records[row["example_id"]] = row
    report = audit_target_bank(
        {"schema_version": TARGET_BANK_SCHEMA, "records": records},
        expected_count=128,
    )

    cell = report["owner_only_majority_accuracy"]["ERE/validation"]
    assert cell["gated_split"] is True
    assert cell["accuracy"] == 1.0
    assert report["owner_only_proxy_passed"] is False


def test_bank_audit_rejects_object_count_to_query_slot_proxy() -> None:
    base = materialize_target(_ere_record(), TinyTokenizer(), strict=True)
    records = {}
    for index in range(128):
        row = deepcopy(base)
        row["example_id"] = f"ere-count-proxy-{index:04d}"
        answer = index % 9
        row["answer_index"] = answer
        row["answer_target"]["answer_index"] = answer
        row["answer_target"]["answer_label"] = "ABCDEFGHI"[answer]
        records[row["example_id"]] = row
    report = audit_target_bank({"schema_version": TARGET_BANK_SCHEMA, "records": records})
    cell = report["object_count_conditioned_query"]["ERE/train"]
    assert cell["sufficient"] is True
    assert cell["majority_accuracy"] == 1.0
    assert report["object_count_query_contingency"]["ERE/train"]["2"] == {"1": 128}
    assert report["object_count_query_cells"]["ERE/train/K2"]["majority_accuracy"] == 1.0
    assert report["object_count_query_cells"]["ERE/train/K2"]["passed"] is False
    assert report["owner_only_proxy_passed"] is True
    assert report["object_count_query_proxy_passed"] is False


def test_small_valid_bank_marks_owner_proxy_check_insufficient_without_failure() -> None:
    bank = materialize_target_bank(
        [_ere_record("e0"), _ere_record("e1")], TinyTokenizer(), strict=True
    )
    report = audit_target_bank(bank, expected_count=2)
    assert report["passed"] is True
    assert report["owner_only_majority_accuracy"]["ERE/train"]["sufficient"] is False
    assert report["construction_invariance"]["passed"] is True
    assert report["query_independence"]["passed"] is True
    broken = deepcopy(bank)
    broken["records"]["e0"]["query_owner"] = 0
    assert audit_target_bank(broken)["passed"] is False


def test_target_bank_digest_is_deterministic_and_audited() -> None:
    first = materialize_target_bank([_ere_record("e0")], TinyTokenizer(), strict=True)
    second = materialize_target_bank([_ere_record("e0")], TinyTokenizer(), strict=True)

    assert first["sha256"] == second["sha256"]
    assert len(first["sha256"]) == 64
    assert first["sha256"] == first["sha256"].upper()
    assert audit_target_bank(first, expected_count=1)["passed"] is True


@pytest.mark.parametrize("mutation", [
    lambda bank: bank["records"]["e0"].update({"answer_index": 1}),
    lambda bank: bank.update({"schema_version": "tampered-schema"}),
    lambda bank: bank.update({"record_count": 2}),
])
def test_target_bank_digest_rejects_record_and_metadata_mutation_without_reseal(mutation) -> None:
    bank = materialize_target_bank([_ere_record("e0")], TinyTokenizer(), strict=True)
    mutation(bank)

    report = audit_target_bank(bank, expected_count=1)
    assert report["passed"] is False
    assert any(value.startswith(("sha256_", "schema_version", "record_count")) for value in report["failures"])


@pytest.mark.parametrize("forged", [None, "not-a-sha256", "0" * 64])
def test_target_bank_digest_rejects_missing_malformed_or_forged_digest(forged) -> None:
    bank = materialize_target_bank([_ere_record("e0")], TinyTokenizer(), strict=True)
    if forged is None:
        del bank["sha256"]
    else:
        bank["sha256"] = forged

    report = audit_target_bank(bank, expected_count=1)
    assert report["passed"] is False
    assert any(value.startswith("sha256_") for value in report["failures"])


def test_target_bank_audit_rejects_record_count_self_report_mismatch() -> None:
    bank = materialize_target_bank([_ere_record("e0")], TinyTokenizer(), strict=True)
    bank["record_count"] = 2

    report = audit_target_bank(bank)
    assert report["passed"] is False
    assert "record_count" in report["failures"]
