import math
import random
from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_7_core import A17CoreConfig, A17RelationAddressedCore
from yggdrasil_v2.reasoning_medium.a1_8_data import A18DatasetSpec, build_a18_dataset, load_a18_records
from yggdrasil_v2.reasoning_medium.a1_8_train import (
    _records_by_length,
    evaluate_a18_stability,
    sample_length_balanced_batch,
)


def test_a18_length_balanced_sampler_and_stability_diagnostic(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    build_a18_dataset(
        data_dir,
        A18DatasetSpec(
            data_seed=20260718,
            train_size=192,
            validation_size=192,
            short_size=96,
            in_range_size=96,
            relation_in_range_size=96,
            ood_size=96,
            relation_ood_size=96,
            causal_size=64,
        ),
    )
    grouped = _records_by_length(load_a18_records(data_dir, "train"))
    batch, counts = sample_length_balanced_batch(grouped, 64, random.Random(7))
    assert len(batch) == 64
    assert set(counts) == set(range(1, 17))
    assert set(counts.values()) == {4}

    model = A17RelationAddressedCore(A17CoreConfig(latent_width=32, ffn_width=64))
    report = evaluate_a18_stability(
        model,
        load_a18_records(data_dir, "supported_ood"),
        "cpu",
        batch_size=32,
        perturbation_epsilon=1e-3,
        perturbation_seed=11,
    )
    assert set(report["per_step"]) == {str(step) for step in range(1, 33)}
    assert report["per_step"]["32"]["active_examples"] == 32
    assert math.isfinite(report["per_step"]["32"]["initial_perturbation_amplification_mean"])
    assert sum(report["first_error_position"].values()) == 96
