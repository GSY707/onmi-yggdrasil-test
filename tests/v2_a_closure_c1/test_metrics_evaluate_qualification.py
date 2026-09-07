from __future__ import annotations

from yggdrasil_v2.v2_a.closure_c1.evaluate import evaluate_behavior, evaluate_causal, evaluate_paired_intervention, evaluate_trace_credit
from yggdrasil_v2.v2_a.closure_c1.metrics import finite_json, paired_cluster_bootstrap_drop, wilson_interval
from yggdrasil_v2.v2_a.closure_c1.qualification import gate_g004_validation, gate_g007_trace, gate_g008_hidden, gate_g010_integrity, qualify_gates


def _record(example_id: str, family: str, split: str, answer: str, *, pair_id=None, pair_role=None, semantic=None):
    mapping = {label: f"candidate:{i}" for i, label in enumerate("ABCDEFGHI")}
    mapping[answer] = semantic if semantic is not None else f"candidate:{answer}"
    return {"example_id": example_id, "family": family, "split": split, "label_mapping": mapping, "semantic_answer": semantic if semantic is not None else mapping[answer], "pair_id": pair_id, "pair_role": pair_role}


def _logits(label: str, margin: float = 5.0):
    out = [-margin] * 9; out["ABCDEFGHI".index(label)] = margin; return out


def test_wilson_and_fixed_cluster_bootstrap_are_finite_and_reproducible():
    interval = wilson_interval(8, 10)
    assert interval["point"] == 0.8 and interval["lower"] < .8 < interval["upper"]
    first = paired_cluster_bootstrap_drop([1, 1, 0, 0], [0, 1, 0, 1], ["p", "p", "q", "q"], samples=200)
    second = paired_cluster_bootstrap_drop([1, 1, 0, 0], [0, 1, 0, 1], ["p", "p", "q", "q"], samples=200)
    assert first == second and first["clusters"] == 2
    finite_json(first)


def test_behavior_is_family_cell_stratified_and_uses_all_nine_logits():
    records = [_record("e0", "ERE", "validation", "A"), _record("e1", "CPS", "validation", "B")]
    report = evaluate_behavior(records, lambda r: {"logits": _logits("A" if r["example_id"] == "e0" else "B")})
    assert set(report["families"]) == {"ERE", "CPS"}
    assert set(report["cells"]) == {"ERE/validation", "CPS/validation"}
    assert report["overall"]["point"] == 1.0


def test_causal_pair_uses_pair_role_and_semantic_flip_not_letter_change():
    records = [_record("b", "ERE", "causal_pairs", "A", pair_id="p", pair_role="base", semantic="candidate:0"), _record("f", "ERE", "causal_pairs", "B", pair_id="p", pair_role="flip", semantic="candidate:1")]
    report = evaluate_causal(records, lambda r: {"predicted_label": r["label_mapping"] and ("A" if r["pair_role"] == "base" else "B")})
    family = report["families"]["ERE"]
    assert family["base"]["point"] == 1 and family["flip"]["point"] == 1
    assert family["simulator_semantic_flip"]["point"] == 1
    assert family["simulator_semantic_flip"]["bootstrap"]["clusters"] == 1
    assert family["both_correct"]["bootstrap"]["clusters"] == 1


def test_trace_credit_reports_content_and_wrong_owner_margin():
    records = [_record("e0", "ERE", "validation", "A"), _record("e1", "ERE", "validation", "A")]
    # Correct owner is high on token 0; cyclic wrong owner is high on token 1.
    def trace(record):
        return {"target_ids": [0, 1], "target_logits": [[5, -5], [-5, 5]], "local_logits": [[-5, 5], [5, -5]]}
    report = evaluate_trace_credit(records, trace, syntax_token_ids=[1])
    family = report["families"]["ERE"]
    assert family["all_token_accuracy"]["point"] == 1
    assert family["content_token_accuracy"]["point"] == 1
    assert family["nll_margin"]["point"] > 0
    assert family["record_bootstrap"]["samples"] == 10_000


def test_trace_credit_uses_explicit_grammar_mask_for_local_vocabulary():
    records = [_record("e0", "ERE", "validation", "A")]
    report = evaluate_trace_credit(
        records,
        lambda _: {
            "target_ids": [1, 0],
            "grammar_mask": [True, False],
            "target_logits": [[-5, 5], [5, -5]],
            "local_logits": [[5, -5], [-5, 5]],
        },
        # Deliberately mark local class 0 as syntax.  The explicit mask must
        # take precedence because the fixed IDs live in the Qwen vocabulary.
        syntax_token_ids=[0],
    )
    family = report["families"]["ERE"]
    assert family["content_token_accuracy"]["point"] == 1.0


def test_gates_are_strict_and_missing_evidence_fails_closed():
    result = qualify_gates({})
    assert result["passed"] is False
    assert set(result["gates"]) == {"G004", "G005", "G006", "G007", "G008", "G009", "G010"}


def test_hidden_gate_requires_zero_and_shuffle_independently():
    baseline = wilson_interval(1300, 1536)
    intervention = wilson_interval(300, 1536)
    drop_point = baseline["point"] - intervention["point"]
    metric = {
        "families": {
            family: {
                "baseline": baseline,
                "paired_drop": {"point": drop_point, "lower": .21, "upper": .7, "clusters": 1536, "samples": 10_000, "seed": 2026082503},
                "intervention": intervention,
            }
            for family in ("ERE", "CPS")
        }
    }
    assert gate_g008_hidden({"zero_hidden": metric})["passed"] is False
    assert gate_g008_hidden(
        {"zero_hidden": metric, "shuffled_hidden": metric}
    )["passed"] is True


def test_trace_gate_accepts_nested_unbounded_nll_margin_interval():
    report = {
        "records": 512,
        "families": {
            family: {
                "records": 256,
                "all_token_accuracy": wilson_interval(900, 1000),
                "content_token_accuracy": wilson_interval(800, 1000),
                "nll_margin": {
                    "point": .2,
                    "record_mean_point": .2,
                    "correct_nll_sum": 1000.0,
                    "wrong_nll_sum": 1200.0,
                    "tokens": 1000,
                    "bootstrap": {"point": .2, "lower": .1, "upper": .3, "clusters": 256, "samples": 10_000, "seed": 2026082503},
                },
            }
            for family in ("ERE", "CPS")
        }
    }
    assert gate_g007_trace(report)["passed"] is True


def test_proportion_gate_rejects_out_of_range_or_reversed_interval():
    assert gate_g004_validation(
        {"families": {family: {"point": 2.0, "lower": 1.5, "upper": 2.5} for family in ("ERE", "CPS")}}
    )["passed"] is False
    assert gate_g004_validation(
        {"families": {family: {"point": .8, "lower": .9, "upper": 1.0} for family in ("ERE", "CPS")}}
    )["passed"] is False


def test_integrity_gate_requires_explicit_integer_zero_probe_count():
    base = {
        "slot": {"n": 3072, "prediction_invariance": 1.0, "max_abs_diff": 0.0, "passed": True},
        "strip": {"n": 3072, "prediction_invariance": 1.0, "max_abs_diff": 0.0, "passed": True},
    }
    assert gate_g010_integrity({**base, "trace_probe_parameter_count": 0})["passed"] is True
    assert gate_g010_integrity({**base, "trace_probe_parameter_count": {}})["passed"] is False


def test_trace_gate_rejects_wrong_bootstrap_contract_or_coverage():
    report = {
        "records": 512,
        "families": {
            family: {
                "records": 256,
                "all_token_accuracy": wilson_interval(900, 1000),
                "content_token_accuracy": wilson_interval(800, 1000),
                "nll_margin": {
                    "point": .2,
                    "record_mean_point": .2,
                    "correct_nll_sum": 1000.0,
                    "wrong_nll_sum": 1200.0,
                    "tokens": 1000,
                    "bootstrap": {"point": .2, "lower": .1, "upper": .3, "clusters": 256, "samples": 1, "seed": 7},
                },
            }
            for family in ("ERE", "CPS")
        },
    }
    assert gate_g007_trace(report)["passed"] is False
    report["records"] = 2
    assert gate_g007_trace(report)["passed"] is False


def test_causal_rejects_duplicate_or_incomplete_pair_roles():
    duplicate = [
        _record("b0", "ERE", "causal_pairs", "A", pair_id="p", pair_role="base"),
        _record("b1", "ERE", "causal_pairs", "A", pair_id="p", pair_role="base"),
        _record("f", "ERE", "causal_pairs", "B", pair_id="p", pair_role="flip"),
    ]
    incomplete = [_record("b", "ERE", "causal_pairs", "A", pair_id="p", pair_role="base")]
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_causal(duplicate, lambda r: {"predicted_label": "A"})
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_causal(incomplete, lambda r: {"predicted_label": "A"})
