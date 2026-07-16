from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_7_data import (
    A17DatasetSpec,
    HELDOUT_JOINT,
    SCHEMA_VERSION,
    SUPPORTED_JOINTS,
    audit_a17_data,
    build_a17_dataset,
    load_a17_records,
)


def _joint(operation: dict) -> tuple[str, str, str]:
    return operation["family"], operation["source"], operation["target"]


def test_a17_controlled_relation_contract(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    build_a17_dataset(
        data_dir,
        A17DatasetSpec(
            seed=20260715,
            train_size=256,
            validation_size=128,
            test_size=128,
            length_size=128,
            relation_size=128,
            causal_size=128,
        ),
    )
    report = audit_a17_data(data_dir)
    assert report["passed"], report["gates"]
    assert report["heldout"]["position_distribution"] == {1: 32, 2: 32, 3: 32, 4: 32}
    assert report["causal_core"]["necessary_rate"] == 1.0
    fingerprints: set[str] = set()
    for split in ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core"):
        rows = load_a17_records(data_dir, split)
        for record in rows:
            assert record["schema_version"] == SCHEMA_VERSION
            assert record["fingerprint"] not in fingerprints
            fingerprints.add(record["fingerprint"])
            assert len(record["state_trajectory"]) == record["program_length"]
            assert len(record["operation_spans"]) == record["program_length"]
            joints = [_joint(operation) for operation in record["operations"]]
            if split == "relation_heldout":
                assert joints.count(HELDOUT_JOINT) == 1
                assert all(joint == HELDOUT_JOINT or joint in SUPPORTED_JOINTS for joint in joints)
            else:
                assert HELDOUT_JOINT not in joints
                assert all(joint in SUPPORTED_JOINTS for joint in joints)

