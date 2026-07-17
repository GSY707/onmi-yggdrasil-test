from __future__ import annotations

import torch

from yggdrasil_v2.reasoning_medium.a1_10_train import encode_a110_targets
from yggdrasil_v2.reasoning_medium.a1_13_models import (
    A113ReasonerConfig,
    A113TransitionClosureReasoner,
)
from yggdrasil_v2.reasoning_medium.a1_13_train import compute_a113_loss
from yggdrasil_v2.reasoning_medium.a1_13_assessment import classify_a113
from yggdrasil_v2.reasoning_medium.a1_13f_assessment import classify_a113f
from yggdrasil_v2.reasoning_medium.a1_15_assessment import classify_a115
from yggdrasil_v2.reasoning_medium.a1_16_assessment import classify_a116
from yggdrasil_v2.reasoning_medium.a1_17_assessment import classify_a117


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "start_values": torch.tensor([[1, 2, 3], [3, 4, 5]]),
        "query_register": torch.tensor([0, 2]),
        "operation_family": torch.tensor([[0, 1], [1, 0]]),
        "operation_source": torch.tensor([[0, 1], [2, 0]]),
        "operation_target": torch.tensor([[1, 2], [1, 2]]),
        "operation_mask": torch.ones((2, 2), dtype=torch.bool),
    }


def test_a113_matrix_changes_only_transition_and_closure() -> None:
    for structured, closure in ((False, True), (True, False), (True, True)):
        model = A113TransitionClosureReasoner(
            A113ReasonerConfig(
                structured_transition=structured,
                prototype_closure_objective=closure,
            )
        )
        report = model.integrity_report()
        assert report["passed"]
        assert report["structured_transition"] is structured
        assert report["prototype_closure_objective"] is closure
        assert report["hard_reembedding_in_forward"] is False
        assert report["copy_swap_branches"] is False


def test_a113_forward_contract_is_shared_across_arms() -> None:
    for structured, closure in ((False, True), (True, False), (True, True)):
        model = A113TransitionClosureReasoner(
            A113ReasonerConfig(
                structured_transition=structured,
                prototype_closure_objective=closure,
            )
        )
        output = model(**_inputs(), recurrent_steps=4)
        assert output["trajectory"].shape == (2, 4, 8, 256)
        assert output["state_logits"].shape == (2, 4, 3, 10)
        assert output["answer_logits"].shape == (2, 10)


def test_a113_closure_is_loss_only_and_receives_gradients() -> None:
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=False,
            prototype_closure_objective=True,
        )
    )
    inputs = _inputs()
    records = [
        {
            "state_trajectory": [
                {"amber": "C", "cobalt": "C", "jade": "D"},
                {"amber": "C", "cobalt": "D", "jade": "C"},
            ],
            "query_register": "amber",
            "answer_index": 2,
        },
        {
            "state_trajectory": [
                {"amber": "F", "cobalt": "E", "jade": "E"},
                {"amber": "F", "cobalt": "E", "jade": "F"},
            ],
            "query_register": "jade",
            "answer_index": 5,
        },
    ]
    output = model(**inputs, recurrent_steps=2)
    labels = encode_a110_targets(records, 2, "cpu")
    loss, metrics = compute_a113_loss(model, output, labels)
    loss.backward()
    assert metrics["closure_weight"] == 1.0
    assert metrics["closure_loss"] > 0.0
    assert model.entity_initialization is not None
    assert model.entity_initialization.weight.grad is not None


def test_a113_wrong_start_state_changes_forward_input() -> None:
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=True,
            prototype_closure_objective=False,
        )
    )
    inputs = _inputs()
    ordinary = model(**inputs, recurrent_steps=2)["state_logits"]
    permuted = model(
        **inputs,
        recurrent_steps=2,
        start_value_permutation=torch.tensor([1, 2, 0]),
    )["state_logits"]
    assert not torch.equal(ordinary, permuted)


def test_a113_classification_is_strict_and_adaptive() -> None:
    assert classify_a113(0, 3, 0, None)["code"] == "continuous_state_closure_primary"
    assert classify_a113(0, 0, 3, None)["code"] == "generic_transition_identity_primary"
    assert classify_a113(0, 3, 3, None)["code"] == "alternative_sufficient_constraints"
    assert classify_a113(0, 0, 0, None)["code"] == "joint_arm_required"
    assert classify_a113(0, 0, 0, 3)["code"] == "transition_closure_joint_requirement"
    assert classify_a113(0, 0, 0, 0)["code"] == "transition_closure_insufficient"
    assert classify_a113(0, 1, 0, None)["code"] == "seed_unstable_inconclusive"


def test_a113f_classification_separates_budget_from_seed_instability() -> None:
    assert classify_a113f(
        generic_ce_state=0, generic_closure_state=0, structured_ce_state=1
    )["code"] == "fixed_budget_seed_instability_persists"
    assert classify_a113f(
        generic_ce_state=0, generic_closure_state=0, structured_ce_state=3
    )["code"] == "structured_transition_fixed_budget_sufficient"
    assert classify_a113f(
        generic_ce_state=1, generic_closure_state=0, structured_ce_state=0
    )["code"] == "generic_reference_rescued_by_budget"


def test_a115_answer_is_exactly_the_queried_state_readout() -> None:
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=True,
            prototype_closure_objective=True,
            query_coupled_answer=True,
        )
    )
    inputs = _inputs()
    output = model(**inputs, recurrent_steps=4)
    expected = output["state_logits"][:, -1].gather(
        1, inputs["query_register"].view(-1, 1, 1).expand(-1, 1, 10)
    ).squeeze(1)
    assert torch.equal(output["answer_logits"], expected)
    assert model.answer_head is None
    integrity = model.integrity_report()
    assert integrity["query_coupled_answer"]
    assert not integrity["independent_answer_head"]


def test_a115_keeps_closure_and_query_couples_answer() -> None:
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=True,
            prototype_closure_objective=True,
            query_coupled_answer=True,
        )
    )
    output = model(**_inputs(), recurrent_steps=3)
    assert model.arm == "STRUCTURED-CLOSURE-COUPLED"
    assert output["answer_logits"].shape == (2, 10)
    report = model.integrity_report()
    assert report["prototype_closure_objective"]
    assert report["query_coupled_answer"]


def test_a117_paired_seed_preserves_every_shared_initial_parameter() -> None:
    seed = 20261321
    torch.manual_seed(seed)
    independent = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=True,
            prototype_closure_objective=True,
            query_coupled_answer=False,
        )
    )
    torch.manual_seed(seed)
    coupled = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=True,
            prototype_closure_objective=True,
            query_coupled_answer=True,
        )
    )
    independent_state = independent.state_dict()
    coupled_state = coupled.state_dict()
    assert set(coupled_state) == {
        name for name in independent_state if not name.startswith("answer_head.")
    }
    for name, value in coupled_state.items():
        assert torch.equal(value, independent_state[name]), name


def test_a116_removes_only_the_redundant_answer_loss() -> None:
    model = A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=True,
            prototype_closure_objective=True,
            query_coupled_answer=True,
        )
    )
    records = [
        {
            "state_trajectory": [
                {"amber": "C", "cobalt": "C", "jade": "D"},
                {"amber": "C", "cobalt": "D", "jade": "C"},
            ],
            "query_register": "amber",
            "answer_index": 2,
        },
        {
            "state_trajectory": [
                {"amber": "F", "cobalt": "E", "jade": "E"},
                {"amber": "F", "cobalt": "E", "jade": "F"},
            ],
            "query_register": "jade",
            "answer_index": 5,
        },
    ]
    output = model(**_inputs(), recurrent_steps=2)
    labels = encode_a110_targets(records, 2, "cpu")
    full, full_metrics = compute_a113_loss(model, output, labels)
    without_answer, metrics = compute_a113_loss(
        model, output, labels, answer_loss_weight=0.0
    )
    assert metrics["answer_loss_weight"] == 0.0
    assert torch.allclose(full - without_answer, full.new_tensor(full_metrics["answer_ce"]))


def test_a115_classification_requires_trigger_formal_and_causal_passes() -> None:
    assert classify_a115(
        trigger_valid=True, formal_passes=3, intervention_passes=3
    )["code"] == "stable_closed_coupled_core_confirmed"
    assert classify_a115(
        trigger_valid=True, formal_passes=2, intervention_passes=2
    )["code"] == "seed_unstable_inconclusive"
    assert classify_a115(
        trigger_valid=True, formal_passes=3, intervention_passes=2
    )["code"] == "ordinary_pass_causal_failure"
    assert classify_a115(
        trigger_valid=False, formal_passes=3, intervention_passes=3
    )["code"] == "trigger_invalid"


def test_a116_classification_requires_stable_formal_and_causal_rescue() -> None:
    assert classify_a116(
        trigger_valid=True, formal_passes=3, intervention_passes=3
    )["code"] == "redundant_answer_ce_primary"
    assert classify_a116(
        trigger_valid=True, formal_passes=0, intervention_passes=0
    )["code"] == "redundant_answer_ce_removal_insufficient"
    assert classify_a116(
        trigger_valid=True, formal_passes=1, intervention_passes=1
    )["code"] == "no_answer_ce_seed_unstable"


def test_a117_classification_separates_objective_from_initialization() -> None:
    common = {"trigger_valid": True, "reference_state": 3}
    assert classify_a117(
        **common,
        coupled_ce_state=3,
        coupled_ce_full=3,
        coupled_noce_state=None,
        coupled_noce_full=None,
    )["code"] == "initialization_basin_primary"
    assert classify_a117(
        **common,
        coupled_ce_state=0,
        coupled_ce_full=0,
        coupled_noce_state=3,
        coupled_noce_full=3,
    )["code"] == "redundant_answer_ce_direct_failure"
    assert classify_a117(
        **common,
        coupled_ce_state=0,
        coupled_ce_full=0,
        coupled_noce_state=0,
        coupled_noce_full=0,
    )["code"] == "independent_answer_auxiliary_gradient_required"
    assert classify_a117(
        **common,
        coupled_ce_state=0,
        coupled_ce_full=0,
        coupled_noce_state=1,
        coupled_noce_full=1,
    )["code"] == "objective_initialization_interaction"
