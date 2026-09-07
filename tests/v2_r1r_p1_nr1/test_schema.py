from __future__ import annotations

import pytest

from yggdrasil_v2.r1_revalidation.nr1.schema import (
    NumericCandidate,
    NumericCase,
    RelationCase,
    RelationQuery,
    RelationState,
)
from yggdrasil_v2.r1_revalidation.nr1.measurement import relation_decisions


def test_numeric_schema_fails_closed() -> None:
    with pytest.raises(ValueError):
        NumericCandidate("bad", (1, 2))
    candidate = NumericCandidate("habcdef012345678", (1, 2))
    with pytest.raises(ValueError):
        NumericCase("x", "qualification", (candidate, candidate))
    with pytest.raises(ValueError):
        NumericCandidate("habcdef012345678", (1, 0))


def test_relation_schema_rejects_alias_self_and_duplicates() -> None:
    handles = ("haaaaaa012345678", "hbbbbbb012345678", "hcccccc012345678", "hdddddd012345678")
    with pytest.raises(ValueError):
        RelationCase(
            "x", "qualification", handles,
            ((handles[0], handles[0]),),
            (RelationQuery(handles[0], handles[1]),),
        )


def test_relation_schema_rejects_cycles_and_unknown_state_handles() -> None:
    handles = (
        "hcycle001",
        "hcycle002",
        "hcycle003",
        "hcycle004",
    )
    with pytest.raises(ValueError, match="acyclic"):
        RelationCase(
            "cycle",
            "fixture",
            handles,
            ((handles[0], handles[1]), (handles[1], handles[0])),
            (RelationQuery(handles[0], handles[2]),),
        )
    with pytest.raises(ValueError, match="unknown"):
        RelationState(
            "bad-state",
            handles,
            ((handles[0], "hunknown001"),),
        )
    state = RelationState("known-state", handles, ((handles[0], handles[1]),))
    with pytest.raises(ValueError, match="unknown"):
        relation_decisions(
            state,
            (RelationQuery(handles[0], "hunknown001"),),
        )
    with pytest.raises(ValueError):
        RelationCase(
            "x", "qualification", handles,
            ((handles[0], handles[1]), (handles[0], handles[1])),
            (RelationQuery(handles[0], handles[1]),),
        )
