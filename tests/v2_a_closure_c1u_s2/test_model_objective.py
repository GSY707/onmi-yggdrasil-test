from __future__ import annotations

import hashlib

import torch

from yggdrasil_v2.v2_a.closure_c1u.runtime import build_independent_card_cache
from yggdrasil_v2.v2_a.closure_c1u.train import set_determinism
from yggdrasil_v2.v2_a.closure_c1u_s2 import contract
from yggdrasil_v2.v2_a.closure_c1u_s2.evaluate import collect_rows
from yggdrasil_v2.v2_a.closure_c1u_s2.model import (
    S2Config,
    S2Model,
    matched_parameter_audit,
)
from yggdrasil_v2.v2_a.closure_c1u_s2.objective import compute_training_loss
from yggdrasil_v2.v2_a.closure_c1u_s2.runtime import S2Runtime
from yggdrasil_v2.v2_a.closure_c1u_s2.tasks import build_multibank, split_records


def _encoder(text: str) -> dict[str, torch.Tensor]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    tokens, width = 2 + len(text) % 3, 12
    return {
        "hidden": torch.tensor(
            [
                [digest[(row + column) % 32] / 255.0 for column in range(width)]
                for row in range(tokens)
            ],
            dtype=torch.float32,
        ),
        "mask": torch.ones(tokens, dtype=torch.bool),
        "token_ids": torch.tensor([digest[index] for index in range(tokens)]),
    }


def _runtime() -> tuple[S2Runtime, list[dict]]:
    records = build_multibank()
    cache = build_independent_card_cache(
        records, _encoder, source_identity={"model": "fixture", "revision": "fixed"}
    )
    return S2Runtime(records, cache, address_width=16), records


def _paired_models() -> tuple[S2Model, S2Model]:
    set_determinism(contract.MODEL_SEED)
    k1 = S2Model(
        S2Config(
            source_width=12,
            payload_width=16,
            address_width=16,
            ffn_width=32,
            workspace_slots=1,
        )
    )
    set_determinism(contract.MODEL_SEED)
    k8 = S2Model(
        S2Config(
            source_width=12,
            payload_width=16,
            address_width=16,
            ffn_width=32,
            workspace_slots=8,
        )
    )
    return k1, k8


def test_k1_k8_have_identical_parameters_and_initial_tensors() -> None:
    k1, k8 = _paired_models()
    audit = matched_parameter_audit(k1, k8)
    assert audit["passed"] is True
    assert audit["k1"] == audit["k8"]
    assert audit["k1_initial_sha256"] == audit["k8_initial_sha256"]


def test_k1_owner_swap_is_exact_null_but_k8_retains_address_binding() -> None:
    runtime, records = _runtime()
    train, _ = split_records(records, "F0")
    cps = [row for row in train if row["family"] == "CPS"][:4]
    ere = [row for row in train if row["family"] == "ERE"][:4]
    batch = runtime.get_batch(
        [row["example_id"] for row in cps + ere]
    )
    k1, k8 = _paired_models()
    k1_report = collect_rows(k1, batch, fold_id="F0", arm="K1")
    k8_report = collect_rows(k8, batch, fold_id="F0", arm="K8")
    assert max(
        value
        for row in k1_report["rows"]
        for value in row["owner_swap_logit_max_abs"]
    ) <= 1.0e-6
    assert max(
        value
        for row in k8_report["rows"]
        for value in row["owner_swap_logit_l2"]
    ) > 1.0e-4
    assert k1_report["permutation_max_abs"] <= 1.0e-6
    assert k8_report["permutation_max_abs"] <= 1.0e-5


def test_matched_objective_is_finite_for_both_real_workspace_sizes() -> None:
    runtime, records = _runtime()
    train, _ = split_records(records, "F0")
    cps = [row for row in train if row["family"] == "CPS"][:4]
    ere = [row for row in train if row["family"] == "ERE"][:4]
    batch = runtime.get_batch([row["example_id"] for row in cps + ere])
    for model in _paired_models():
        loss, parts, outputs = compute_training_loss(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert parts["total"] > 0.0
        assert outputs["support_logits"].shape == (8, 2, 9)
        assert all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
