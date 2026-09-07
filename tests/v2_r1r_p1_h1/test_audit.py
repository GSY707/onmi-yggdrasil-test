from yggdrasil_v2.r1_revalidation.h1.audit import architecture_audit
from yggdrasil_v2.r1_revalidation.h1.model import H1Config


def test_compact_architecture_audit_covers_sparse_projection_contract() -> None:
    config = H1Config(
        source_width=24,
        latent_width=16,
        K=8,
        T=3,
        recurrent_layers=2,
        num_heads=4,
        active_ffn_multiplier=3,
        shared_ffn_inner_width=24,
        routed_feature_width=24,
    )
    report = architecture_audit(config, source_tokens=7)
    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["runtime_calls"]["shared_ffn"] == 6
    assert report["runtime_calls"]["routed_projections"] == {0: 6, 1: 0}
    assert report["runtime_calls"]["routed_feature_trunk"] == 6
    assert report["parameters"]["shared"]["active_parameters_per_record"] == report[
        "parameters"
    ]["mixed"]["active_parameters_per_record"]
