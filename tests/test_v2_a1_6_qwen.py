import torch

from yggdrasil_v2.reasoning_medium.a1_6_core import A16CoreConfig, RelationAddressedCore
from yggdrasil_v2.reasoning_medium.a1_6_qwen import FrozenQwenBoundary, _span_mask


def test_a16_qwen_boundary_reuses_c0_core_without_full_source_path() -> None:
    core = RelationAddressedCore(A16CoreConfig(latent_width=16, ffn_width=32))
    model = FrozenQwenBoundary(8, core)
    output = model(
        torch.randn(2, 3, 8), torch.randn(2, 8), torch.randn(2, 2, 8), torch.randn(2, 2, 8), torch.randn(2, 2, 8), torch.ones(2, 2, dtype=torch.bool)
    )
    assert output["answer_logits"].shape == (2, 10)
    assert output["query_pointer_logits"].shape == (2, 3)
    assert model.core.transition is core.transition
    assert model.integrity_report()["full_source_cross_attention"] is False
    assert model.integrity_report()["source_mean"] is False
    assert not any("source_cross_attention" in name or "answer_head" in name for name, _ in model.named_modules())


def test_a16_span_mask_fails_loudly() -> None:
    try:
        _span_mask([[0, 1], [2, 3]], 5, 6, label="missing")
    except ValueError as error:
        assert "empty token mask" in str(error)
    else:
        raise AssertionError("empty spans must not be silently repaired")

