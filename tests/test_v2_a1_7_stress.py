from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_7_data import A17DatasetSpec, HELDOUT_JOINT, _joint, build_a17_dataset
from yggdrasil_v2.reasoning_medium.a1_7_stress import (
    A17StressSpec,
    audit_a17_stress_data,
    build_a17_stress_dataset,
    load_a17_stress_records,
)


def test_a17_long_recurrence_stress_contract(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    build_a17_dataset(
        base_dir,
        A17DatasetSpec(
            seed=20260715,
            train_size=128,
            validation_size=64,
            test_size=64,
            length_size=64,
            relation_size=64,
            causal_size=32,
        ),
    )
    stress_dir = tmp_path / "stress"
    manifest = build_a17_stress_dataset(
        stress_dir,
        base_dir,
        A17StressSpec(seed=20260718, supported_size=96, relation_size=96, lengths=(8, 12, 16)),
    )
    assert set(manifest["splits"]) == {"supported_length_stress", "relation_length_stress"}
    audit = audit_a17_stress_data(stress_dir, base_dir)
    assert audit["passed"], audit["gates"]
    assert audit["lengths"] == [8, 12, 16]

    relation = load_a17_stress_records(stress_dir, "relation_length_stress")
    assert all(sum(_joint(operation) == HELDOUT_JOINT for operation in record["operations"]) == 1 for record in relation)
    assert {record["program_length"] for record in relation} == {8, 12, 16}
