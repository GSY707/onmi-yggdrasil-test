from pathlib import Path

from yggdrasil_v2.reasoning_medium.a1_6_data import A16DatasetSpec, audit_a16_data, build_a16_dataset, load_a16_records


def test_a16_data_contract_and_programmatic_gates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    build_a16_dataset(data_dir, A16DatasetSpec(seed=20260715, train_size=128, validation_size=32, test_size=32, length_size=32, relation_size=32, causal_size=32))
    report = audit_a16_data(data_dir)
    assert report["passed"]
    assert report["causal_core"]["necessary_rate"] == 1.0
    assert report["relation_holdout"]["heldout_coverage"] == 1.0
    for split in ("train", "validation", "test", "length_heldout", "relation_heldout", "causal_core"):
        for record in load_a16_records(data_dir, split):
            assert len(record["operation_spans"]) == record["program_length"]
            assert len(record["state_trajectory"]) == record["program_length"]
            assert set(record["start_value_spans"]) == {"amber", "cobalt", "jade"}
            for operation, span in zip(record["operations"], record["operation_spans"]):
                assert operation["source"] != operation["target"]
                assert span["state_after"] == record["state_trajectory"][span["step"] - 1]

