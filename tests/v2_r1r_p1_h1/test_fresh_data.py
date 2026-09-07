from __future__ import annotations

from dataclasses import fields

import pytest

from src.yggdrasil_v2.r1_revalidation.h1.fresh_data import (
    LABELS,
    UNDECIDED,
    FreshBundle,
    NumericComponent,
    NumericTarget,
    RelationComponent,
    RelationTarget,
    SourceRecord,
    add_numeric_cancelling_pair,
    add_relation_transitive_redundancy,
    cross_split_overlap_audit,
    forbidden_source_scan,
    generate_bundle,
    materialize_metamorphic,
    metamorphic_transforms,
    numeric_oracle,
    paraphrase_source,
    permute_clauses,
    relation_fixed_point_trace,
    relation_oracle,
    rename_handles,
    semantic_fingerprint,
    source_fingerprint,
)


COUNTS = {kind: {split: 4 for split in ("train", "validation", "supported", "heldout", "causal")} for kind in ("numeric", "relation")}


def test_determinism_and_model_view_boundary() -> None:
    left = generate_bundle(20260817, COUNTS)
    right = generate_bundle(20260817, COUNTS)
    assert left == right
    assert all(tuple(field.name for field in fields(record)) == ("id", "source_text", "source_atoms") for record in left.records)
    assert all(record.source_text == "\n".join(record.source_atoms) and len(record.source_atoms) <= 40 for record in left.records)
    assert not forbidden_source_scan(left)
    assert cross_split_overlap_audit(left)["passed"]


def test_numeric_and_relation_targets_are_independent() -> None:
    bundle = generate_bundle(9, {"numeric": {"train": 8}, "relation": {"train": 8}})
    for record in bundle.records:
        component = bundle.components[record.id]
        target = bundle.targets[record.id]
        if isinstance(component, NumericComponent):
            assert isinstance(target, NumericTarget)
            oracle = numeric_oracle(component)
            assert target.prefix_labels[-1] == target.final_label
            assert len(oracle.prefix_winners) == len(target.prefix_labels)
            assert all(label in LABELS or label == UNDECIDED for label in target.prefix_labels)
        else:
            assert isinstance(target, RelationTarget)
            assert relation_oracle(component).query_index == target.query_index
            assert target.trace == relation_fixed_point_trace(component)
            assert all(value in (0, 1, 2, 3, UNDECIDED) for value in target.trace)


def test_split_ranges_and_non_chain_topology() -> None:
    bundle = generate_bundle(11, COUNTS)
    for split, ids in bundle.splits.items():
        for record_id in ids:
            component = bundle.components[record_id]
            if isinstance(component, NumericComponent):
                horizon = len(component.deltas[0])
                magnitude = max(abs(value) for row in component.deltas for value in row)
                if split == "heldout":
                    assert horizon >= 7 and magnitude >= 6
                else:
                    assert horizon <= 7 and magnitude <= 7
            else:
                if split == "heldout":
                    assert len(component.handles) >= 9
                indegrees = {handle: 0 for handle in component.handles}
                outdegrees = {handle: 0 for handle in component.handles}
                for source, target in component.edges:
                    outdegrees[source] += 1
                    indegrees[target] += 1
                assert max(indegrees.values()) >= 2 or max(outdegrees.values()) >= 2
                assert sum(relation_oracle(component).reachable) == 1


def test_metamorphic_transforms_preserve_labels() -> None:
    bundle = generate_bundle(19, {"numeric": {"train": 5}, "relation": {"train": 5}})
    for record in bundle.records:
        component = bundle.components[record.id]
        before = relation_oracle(component).query_index if isinstance(component, RelationComponent) else numeric_oracle(component).final_winner
        mapping = {handle: f"h{index}zzzzzz" for index, handle in enumerate(component.handles)}
        renamed = rename_handles(component, mapping)
        after = relation_oracle(renamed).query_index if isinstance(renamed, RelationComponent) else numeric_oracle(renamed).final_winner
        assert after == before if isinstance(component, RelationComponent) else after == mapping[before]
        if isinstance(component, NumericComponent):
            augmented = add_numeric_cancelling_pair(component)
            assert numeric_oracle(augmented).final_winner == before
            assert tuple(reversed(component.handles)) == permute_clauses(component, (3, 2, 1, 0)).handles
        else:
            redundant = add_relation_transitive_redundancy(component)
            assert relation_oracle(redundant).query_index == before
        assert source_fingerprint(paraphrase_source(record)) != source_fingerprint(record)
        assert paraphrase_source(record).id == record.id
        assert not forbidden_source_scan(paraphrase_source(record))



def test_unknown_handles_and_forbidden_source_fail_closed() -> None:
    component = next(iter(generate_bundle(31, {"numeric": {"train": 1}}).components.values()))
    assert isinstance(component, NumericComponent)
    with pytest.raises(ValueError):
        rename_handles(component, {component.handles[0]: "bad"})
    with pytest.raises(ValueError):
        SourceRecord("x", "This contains final", ("This", "contains", "final"))


def test_materialized_metamorphics_recompute_targets_without_leakage() -> None:
    bundle = generate_bundle(41, {"numeric": {"train": 1}, "relation": {"train": 1}})
    for record in bundle.records:
        component = bundle.components[record.id]
        target = bundle.targets[record.id]
        transforms = ("handle_rename", "choice_permutation", "cancelling_pair", "surface_paraphrase") if isinstance(component, NumericComponent) else ("handle_rename", "choice_permutation", "query_permutation", "transitive_redundancy", "surface_paraphrase")
        for transform in transforms:
            materialized = materialize_metamorphic(component, target, record, transform)
            assert tuple(field.name for field in fields(materialized["record"])) == ("id", "source_text", "source_atoms")
            assert not forbidden_source_scan(materialized["record"])


def test_supervised_semantic_fingerprint_tracks_options_but_not_opaque_renames() -> None:
    bundle = generate_bundle(
        43,
        {"numeric": {"train": 1}, "relation": {"train": 1}},
    )
    by_id = {record.id: record for record in bundle.records}
    for record_id, component in bundle.components.items():
        target = bundle.targets[record_id]
        baseline = semantic_fingerprint(component, target)
        renamed = materialize_metamorphic(
            component,
            target,
            by_id[record_id],
            "handle_rename",
        )
        assert semantic_fingerprint(renamed["component"], renamed["target"]) == baseline
        paraphrased = materialize_metamorphic(
            component,
            target,
            by_id[record_id],
            "surface_paraphrase",
        )
        assert semantic_fingerprint(paraphrased["component"], paraphrased["target"]) == baseline
        permuted = materialize_metamorphic(
            component,
            target,
            by_id[record_id],
            "choice_permutation",
        )
        assert semantic_fingerprint(permuted["component"], permuted["target"]) != baseline


def test_every_causal_relation_supports_transitive_redundancy() -> None:
    bundle = generate_bundle(2026081791, {"relation": {"causal": 128}})
    assert len(bundle.records) == 128
    for record in bundle.records:
        component = bundle.components[record.id]
        target = bundle.targets[record.id]
        assert isinstance(component, RelationComponent)
        materialized = materialize_metamorphic(
            component,
            target,
            record,
            "transitive_redundancy",
        )
        assert materialized["target"].label == target.label
