from __future__ import annotations

from dataclasses import asdict

import torch

from yggdrasil_v2.reasoning_medium.a1_10_train import encode_a110_targets
from yggdrasil_v2.reasoning_medium.a1_13_models import (
    A113ReasonerConfig,
    A113TransitionClosureReasoner,
)
from yggdrasil_v2.reasoning_medium.a1_18b_assessment import classify_a118b
from yggdrasil_v2.reasoning_medium.a1_18_train import (
    A118TrainSpec,
    CHECKPOINT_SCHEMA,
    _export_deployment,
    compute_a118_loss,
    load_a118_deployment,
)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "start_values": torch.tensor([[1, 2, 3], [3, 4, 5]]),
        "query_register": torch.tensor([0, 2]),
        "operation_family": torch.tensor([[0, 1], [1, 0]]),
        "operation_source": torch.tensor([[0, 1], [2, 0]]),
        "operation_target": torch.tensor([[1, 2], [1, 2]]),
        "operation_mask": torch.ones((2, 2), dtype=torch.bool),
    }


def _records() -> list[dict[str, object]]:
    return [
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


def _model(auxiliary: str) -> A113TransitionClosureReasoner:
    return A113TransitionClosureReasoner(
        A113ReasonerConfig(
            structured_transition=True,
            prototype_closure_objective=True,
            query_coupled_answer=True,
            training_auxiliary=auxiliary,
        )
    )


def test_a118_qaux_is_bitwise_paired_with_the_a113_answer_head() -> None:
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
    qaux = _model("queried_answer")
    independent_state = independent.state_dict()
    qaux_state = qaux.state_dict()
    assert torch.equal(
        independent_state["answer_head.weight"],
        qaux_state["training_auxiliary_head.weight"],
    )
    assert torch.equal(
        independent_state["answer_head.bias"],
        qaux_state["training_auxiliary_head.bias"],
    )
    for name, value in qaux_state.items():
        if name.startswith("training_auxiliary_head."):
            continue
        assert torch.equal(value, independent_state[name]), name


def test_a118_auxiliaries_are_training_only_and_answer_stays_state_coupled() -> None:
    inputs = _inputs()
    for auxiliary, expected_shape in (
        ("queried_answer", (2, 10)),
        ("full_state", (2, 3, 10)),
        ("full_trajectory_state", (2, 2, 3, 10)),
    ):
        model = _model(auxiliary)
        model.train()
        output = model(**inputs, recurrent_steps=2)
        assert output["training_auxiliary_logits"].shape == expected_shape
        expected_answer = output["state_logits"][:, -1].gather(
            1, inputs["query_register"].view(-1, 1, 1).expand(-1, 1, 10)
        ).squeeze(1)
        assert torch.equal(output["answer_logits"], expected_answer)
        model.eval()
        inference = model(**inputs, recurrent_steps=2)
        assert inference["training_auxiliary_logits"] is None
        assert torch.equal(inference["answer_logits"], expected_answer)


def test_a118_saux_supervises_all_final_registers_without_answer_loss() -> None:
    model = _model("full_state")
    model.train()
    labels = encode_a110_targets(_records(), 2, "cpu")
    output = model(**_inputs(), recurrent_steps=2)
    loss, metrics = compute_a118_loss(model, output, labels, auxiliary_weight=1.0)
    loss.backward()
    assert metrics["training_auxiliary"] == "full_state"
    assert metrics["official_answer_loss_weight"] == 0.0
    assert model.training_auxiliary_head is not None
    assert model.training_auxiliary_head.weight.grad is not None
    assert model.relation_transition is not None
    assert model.relation_transition.operator[0].weight.grad is not None


def test_a118_tsaux_supervises_global_state_at_every_recurrent_step() -> None:
    model = _model("full_trajectory_state")
    model.train()
    labels = encode_a110_targets(_records(), 2, "cpu")
    output = model(**_inputs(), recurrent_steps=2)
    loss, metrics = compute_a118_loss(model, output, labels, auxiliary_weight=1.0)
    loss.backward()
    assert output["training_auxiliary_logits"].shape == (2, 2, 3, 10)
    assert metrics["training_auxiliary"] == "full_trajectory_state"
    assert metrics["official_answer_loss_weight"] == 0.0


def test_a118_deployment_physically_strips_the_auxiliary(tmp_path) -> None:
    model = _model("full_state")
    spec = A118TrainSpec(
        model_seed=20261321,
        data_seed=20260721,
        training_auxiliary="full_state",
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": 4000,
        "model_config": asdict(model.a113_config),
        "train_spec": asdict(spec),
        "data_manifest_sha256": "unit-test",
        "model": model.state_dict(),
    }
    output = tmp_path / "deployable.pt"
    report = _export_deployment(checkpoint, output)
    assert report["stripped_parameter_names"] == [
        "training_auxiliary_head.bias",
        "training_auxiliary_head.weight",
    ]
    deployed = load_a118_deployment(output)
    assert deployed.training_auxiliary_head is None
    assert deployed.a113_config.training_auxiliary == "none"
    assert deployed.a113_config.query_coupled_answer


def test_a118b_classification_requires_paired_and_fresh_causal_closure() -> None:
    assert classify_a118b(
        final_saux_state=2,
        tsaux_paired_full=3,
        tsaux_paired_causal=3,
        tsaux_fresh_full=3,
        tsaux_fresh_causal=3,
        integrity_passed=True,
    )["code"] == "per_step_global_state_credit_assignment_confirmed"
    assert classify_a118b(
        final_saux_state=2,
        tsaux_paired_full=3,
        tsaux_paired_causal=3,
        tsaux_fresh_full=2,
        tsaux_fresh_causal=2,
        integrity_passed=True,
    )["code"] == "trajectory_state_auxiliary_seed_unstable"
