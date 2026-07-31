from __future__ import annotations

from yggdrasil_v2.reasoning_medium.a1_21p_pareto import (
    _dominates,
    assess_a121p_stage,
    build_a121p_prompt,
    parse_a121p_visible_trace,
    render_a121p_routing_record,
)
from yggdrasil_v2.reasoning_medium.a1_20c_supervision import (
    _record_char_targets,
)
from yggdrasil_v2.reasoning_medium.a1_21p_k1 import (
    a121p_k1_config,
    compute_a121p_k1_loss,
)
from yggdrasil_v2.reasoning_medium.a1_10_model import (
    A110AnonymousRecurrentReasoner,
)
from yggdrasil_v2.reasoning_medium.a1_20b_train import A120BLabels
import torch
import json


def _record() -> dict:
    return {
        "question": (
            "Objects: object_a1=A, object_b2=B. Operations: "
            "COPY object_a1 -> object_b2. Query: object_b2."
        ),
        "entity_names": ["object_a1", "object_b2"],
        "state_trajectory": [
            {"object_a1": "A", "object_b2": "A"}
        ],
        "answer": "A",
    }


def test_a121p_trace_parser_reads_dynamic_objects() -> None:
    answer, states = parse_a121p_visible_trace(
        "STEP 1: object_a1=A, object_b2=A\nFINAL: A"
    )
    assert answer == "A"
    assert states == {1: {"object_a1": "A", "object_b2": "A"}}


def test_a121p_trace_parser_accepts_direct_bare_answer() -> None:
    answer, states = parse_a121p_visible_trace(" G\n")
    assert answer == "G"
    assert states == {}


def test_a121p_current_prompt_does_not_insert_target() -> None:
    record = _record()
    direct = build_a121p_prompt(record, "direct")
    cot = build_a121p_prompt(record, "text_cot")
    assert direct.endswith("ANSWER:\n")
    assert cot.endswith("ANSWER:\n")
    assert "FINAL: A" not in direct
    assert "FINAL: A" not in cot


def test_a121p_pareto_dominance_requires_one_strict_axis() -> None:
    equal = {
        "final_answer_accuracy": 1.0,
        "latency_seconds_median": 2.0,
    }
    faster = {
        "final_answer_accuracy": 1.0,
        "latency_seconds_median": 1.0,
    }
    weaker = {
        "final_answer_accuracy": 0.9,
        "latency_seconds_median": 2.0,
    }
    assert not _dominates(equal, equal)
    assert _dominates(faster, equal)
    assert not _dominates(weaker, equal)


def test_a121p_routing_variant_preserves_executable_semantics() -> None:
    record = {
        **_record(),
        "fingerprint": "source",
        "example_id": "source-example",
        "start_state": {"object_a1": "A", "object_b2": "B"},
        "operations": [
            {
                "family": "copy",
                "source": "object_a1",
                "target": "object_b2",
            }
        ],
        "query_register": "object_b2",
    }
    observed = render_a121p_routing_record(record)
    assert observed["operations"] == record["operations"]
    assert observed["state_trajectory"] == record["state_trajectory"]
    assert observed["answer"] == record["answer"]
    assert "RELAY object_a1 => object_b2" in observed["question"]
    assert "Operations:" not in observed["question"]
    assert observed["question"].index("Rules:") < observed["question"].index(
        "Stations:"
    )
    assert observed["question"].index(
        "Stations:"
    ) < observed["question"].index("Dispatch plan:")
    assert observed["source_fingerprint"] == "source"
    names, values, operations, query = _record_char_targets(observed)
    question = observed["question"]
    assert question[names[0] : names[0] + len("object_a1")] == "object_a1"
    assert question[values[0]] == "A"
    assert question[
        operations[0][0] : operations[0][0] + len("RELAY")
    ] == "RELAY"
    assert question[
        operations[0][1] : operations[0][1] + len("object_a1")
    ] == "object_a1"
    assert question[
        operations[0][2] : operations[0][2] + len("object_b2")
    ] == "object_b2"
    assert question[query : query + len("object_b2")] == "object_b2"


def test_a121p_k1_has_one_slot_and_five_state_fields() -> None:
    model = A110AnonymousRecurrentReasoner(a121p_k1_config(16))
    output = model(
        torch.randn(2, 7, 16),
        torch.ones(2, 7, dtype=torch.bool),
        recurrent_steps=3,
    )
    assert model.config.workspace_slots == 1
    assert output["state_logits"].shape == (2, 3, 5, 10)


def test_a121p_k1_loss_reaches_recurrent_model() -> None:
    model = A110AnonymousRecurrentReasoner(a121p_k1_config(16))
    output = model(
        torch.randn(2, 7, 16),
        torch.ones(2, 7, dtype=torch.bool),
        recurrent_steps=3,
    )
    labels = A120BLabels(
        entity_presence=torch.ones(2, 5),
        operation_presence=torch.ones(2, 32),
        value_targets=torch.zeros(2, 5, dtype=torch.long),
        family_targets=torch.zeros(2, 32, dtype=torch.long),
        source_targets=torch.zeros(2, 32, dtype=torch.long),
        target_targets=torch.zeros(2, 32, dtype=torch.long),
        query_targets=torch.zeros(2, dtype=torch.long),
        state_targets=torch.zeros(2, 3, 5, dtype=torch.long),
        answer_targets=torch.zeros(2, dtype=torch.long),
    )
    loss, _ = compute_a121p_k1_loss(output, labels)
    loss.backward()
    assert model.learned_queries.grad is not None
    assert float(model.learned_queries.grad.norm()) > 0


def test_a121p_assessment_stops_surface_only_unmatched_contract(
    tmp_path,
) -> None:
    def write(name: str, payload: dict) -> object:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    canonical = write("canonical.json", {"passed": True})
    causal = write("causal.json", {"passed": True})
    routing = [
        write(f"routing-{index}.json", {"passed": True})
        for index in range(3)
    ]
    canonical_preflight = write(
        "canonical-preflight.json",
        {"passed": True, "data": {"examples": 5}},
    )
    routing_preflight = write(
        "routing-preflight.json",
        {"passed": True, "data": {"examples": 5}},
    )
    manifest = write(
        "manifest.json",
        {
            "surface_only_transform": True,
            "canonical_semantics_preserved": True,
        },
    )
    k1 = write(
        "k1.json",
        {
            "validation": {
                "aggregate": {
                    "examples": 2048,
                    "trajectory_full_exact": 0.1,
                    "final_state_full_exact": 0.3,
                    "state_token_accuracy": 0.5,
                    "final_answer_accuracy": 0.55,
                }
            }
        },
    )
    optimization = [
        write(
            f"optimization-{index}.json",
            {
                "formal_eligibility_passed": False,
                "training": {"fresh_initialization": False},
            },
        )
        for index in range(3)
    ]
    result = assess_a121p_stage(
        canonical_matrix_path=canonical,
        hidden_causal_path=causal,
        routing_probe_paths=routing,
        canonical_preflight_path=canonical_preflight,
        routing_preflight_path=routing_preflight,
        routing_manifest_path=manifest,
        k1_results_path=k1,
        optimization_path_results=optimization,
        output_path=tmp_path / "assessment.json",
    )
    assert result["a120d_relaxed_mechanism_passed"]
    assert not result["a121p_passed"]
    assert not result["a122a_authorized"]
    assert not result["formal_gates"][
        "second_executable_algebra_or_planning_family"
    ]
    assert not result["formal_gates"][
        "matched_training_and_teacher_data_budget"
    ]
