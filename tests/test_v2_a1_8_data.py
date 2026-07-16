from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_7_data import HELDOUT_JOINT, _joint
from yggdrasil_v2.reasoning_medium.a1_8_data import (
    A18DatasetSpec,
    DATA_SPLITS,
    audit_a18_data,
    build_a18_dataset,
    load_a18_records,
)


def _spec(seed: int) -> A18DatasetSpec:
    return A18DatasetSpec(
        data_seed=seed,
        train_size=192,
        validation_size=192,
        short_size=96,
        in_range_size=96,
        relation_in_range_size=96,
        ood_size=96,
        relation_ood_size=96,
        causal_size=64,
    )


def test_a18_random_depth_data_contract_and_external_non_overlap(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_a18_dataset(first, _spec(20260718))
    first_audit = audit_a18_data(first)
    assert first_audit["passed"], first_audit["gates"]

    build_a18_dataset(second, _spec(20260719), forbidden_data_dirs=(first,))
    second_audit = audit_a18_data(second)
    assert second_audit["passed"], second_audit["gates"]
    assert second_audit["external_overlap"] == 0
    assert set(second_audit["expected_lengths"]["train"]) == set(range(1, 17))
    assert set(second_audit["expected_lengths"]["supported_ood"]) == {20, 24, 32}

    fingerprints = {
        data_dir: {
            record["fingerprint"]
            for split in DATA_SPLITS
            for record in load_a18_records(data_dir, split)
        }
        for data_dir in (first, second)
    }
    assert not (fingerprints[first] & fingerprints[second])

    for split in ("relation_in_range", "relation_ood"):
        records = load_a18_records(second, split)
        assert all(sum(_joint(operation) == HELDOUT_JOINT for operation in record["operations"]) == 1 for record in records)

