from __future__ import annotations

import ast
import hashlib
import inspect
import json
import random
from collections import Counter, deque
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

from .faults import (
    NumericFaultTarget,
    RelationFaultTarget,
    numeric_constant_collapse,
    numeric_drop_last,
    numeric_final_off_by_one,
    numeric_position_shortcut,
    numeric_sign_flip,
    numeric_winner_swap,
    relation_drop_bridge,
    relation_handle_alias,
    relation_query_shortcut,
    relation_reverse_pollution,
    relation_spurious,
)
from .measurement import (
    audit_numeric_state,
    audit_relation_state,
    build_numeric_state,
    build_relation_state,
    compare_scalars,
    numeric_decision,
    relation_decisions,
)
from .reference import numeric_oracle, relation_oracle
from .schema import (
    NumericCandidate,
    NumericCandidateState,
    NumericCase,
    NumericState,
    RelationCase,
    RelationQuery,
    RelationState,
)


FORMAL_SEED = 2026081703
NUMERIC_COUNTS = {"qualification": 384, "heldout": 256}
RELATION_COUNTS = {"qualification": 384, "heldout": 256}
DECISION_KILL_MIN = 0.80
METRIC_KILL_MIN = 0.80
HAND_FIXTURE_RELATIVE = Path(
    "tests/v2_r1r_p1_nr1/fixtures/nr1_hand_authored_cases.json"
)
HAND_FIXTURE_SHA256 = "C94D372B7A06448F2031ABCA03C2EEE2538B3C627E4A5CD00E522C3F18EFB57F"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _handle(namespace: str, seed: int, case_index: int, item_index: int) -> str:
    digest = hashlib.sha256(
        f"{namespace}:{seed}:{case_index}:{item_index}".encode("ascii")
    ).hexdigest()
    return f"h{digest[:15]}"


def build_numeric_cases(seed: int, split: str, count: int) -> tuple[NumericCase, ...]:
    if split == "qualification":
        candidate_range = (2, 5)
        horizons = (3, 5)
        magnitude_range = (1, 32)
        salt = 0x4E5231
    elif split == "heldout":
        candidate_range = (6, 9)
        horizons = (7, 9, 11)
        magnitude_range = (64, 1024)
        salt = 0x484F4C44
    else:
        raise ValueError(f"unknown split: {split}")
    rng = random.Random(seed ^ salt)
    cases: list[NumericCase] = []
    low, high = magnitude_range
    for case_index in range(count):
        candidate_count = rng.randint(*candidate_range)
        horizon = rng.choice(horizons)
        winner_position = case_index % candidate_count
        loser_magnitudes = rng.sample(range(low, high), candidate_count - 1)
        loser_cursor = 0
        candidates: list[NumericCandidate] = []
        for candidate_index in range(candidate_count):
            deltas: list[int] = []
            for _ in range((horizon - 1) // 2):
                magnitude = rng.randint(low, high)
                sign = -1 if rng.random() < 0.5 else 1
                deltas.extend((sign * magnitude, -sign * magnitude))
            if candidate_index == winner_position:
                final_delta = -high
            else:
                final_delta = -loser_magnitudes[loser_cursor]
                loser_cursor += 1
            deltas.append(final_delta)
            candidates.append(
                NumericCandidate(
                    handle=_handle("numeric", seed, case_index, candidate_index),
                    deltas=tuple(deltas),
                )
            )
        case = NumericCase(
            case_id=f"numeric-{split}-{case_index:04d}",
            split=split,
            candidates=tuple(candidates),
        )
        expected = numeric_oracle(case)
        if expected["winner"] != candidates[winner_position].handle:
            raise AssertionError(f"numeric critical-update construction failed: {case.case_id}")
        cases.append(case)
    return tuple(cases)


def _provisional_relation(
    case_id: str,
    split: str,
    handles: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    query: tuple[str, str],
) -> RelationCase:
    return RelationCase(
        case_id=case_id,
        split=split,
        handles=handles,
        edges=edges,
        queries=(RelationQuery(*query),),
    )


def build_relation_cases(seed: int, split: str, count: int) -> tuple[RelationCase, ...]:
    if split == "qualification":
        handle_range = (7, 8)
        salt = 0x52454C31
    elif split == "heldout":
        handle_range = (9, 16)
        salt = 0x52484F4C
    else:
        raise ValueError(f"unknown split: {split}")
    rng = random.Random(seed ^ salt)
    cases: list[RelationCase] = []
    for case_index in range(count):
        handle_count = rng.randint(*handle_range)
        topological = [
            _handle("relation", seed, case_index, item_index)
            for item_index in range(handle_count)
        ]
        rng.shuffle(topological)
        mode = case_index % 4
        edges: set[tuple[str, str]] = {
            (topological[0], topological[1]),
            (topological[0], topological[2]),
            (topological[1], topological[3]),
            (topological[2], topological[3]),
            (topological[3], topological[4]),
        }
        reserve = 2 if mode == 1 else 0
        main_stop = handle_count - reserve
        for index in range(5, main_stop):
            parents = rng.sample(range(index), 2 if mode in {0, 2} else 1)
            for parent in parents:
                edges.add((topological[parent], topological[index]))
        if reserve:
            edges.add((topological[-2], topological[-1]))
        for left in range(main_stop - 1):
            for right in range(left + 2, main_stop):
                if rng.random() < 0.10:
                    edges.add((topological[left], topological[right]))
        if case_index % 2 == 0:
            # This direct edge is intentionally redundant with both diamond paths.
            edges.add((topological[0], topological[3]))

        presented_handles = list(topological)
        rng.shuffle(presented_handles)
        edge_order = list(edges)
        rng.shuffle(edge_order)
        case_id = f"relation-{split}-{case_index:04d}"
        provisional = _provisional_relation(
            case_id,
            split,
            tuple(presented_handles),
            tuple(edge_order),
            (topological[0], topological[4]),
        )
        closure = set(relation_oracle(provisional)["closure"])
        transitive = sorted(closure - edges)
        if not transitive:
            raise AssertionError(f"relation case lacks non-direct composition: {case_id}")
        primary_true = rng.choice(transitive)
        primary_false = (primary_true[1], primary_true[0])
        all_pairs = {
            (source, target)
            for source in presented_handles
            for target in presented_handles
            if source != target
        }
        true_pool = sorted(closure - {primary_true})
        false_pool = sorted(all_pairs - closure - {primary_false})
        query_count = 4 + case_index % 5
        true_count = query_count // 2
        if query_count % 2 and case_index % 2 == 0:
            true_count += 1
        false_count = query_count - true_count
        query_pairs = [primary_true, primary_false]
        query_pairs.extend(rng.sample(true_pool, true_count - 1))
        query_pairs.extend(rng.sample(false_pool, false_count - 1))
        rng.shuffle(query_pairs)
        case = RelationCase(
            case_id=case_id,
            split=split,
            handles=tuple(presented_handles),
            edges=tuple(edge_order),
            queries=tuple(RelationQuery(*pair) for pair in query_pairs),
        )
        decisions = relation_oracle(case)["decisions"]
        if len(set(decisions)) != 2:
            raise AssertionError(f"relation query balance failed: {case_id}")
        cases.append(case)
    return tuple(cases)


def _numeric_exact(state: NumericState, expected: dict[str, Any]) -> bool:
    observed = {candidate.handle: candidate for candidate in state.candidates}
    return set(observed) == set(expected["prefixes"]) and all(
        observed[handle].prefixes == tuple(expected["prefixes"][handle])
        and observed[handle].final == expected["totals"][handle]
        for handle in expected["prefixes"]
    )


def _relation_exact(
    case: RelationCase, state: RelationState, expected: dict[str, Any]
) -> bool:
    return tuple(state.handles) == tuple(case.handles) and set(state.closure) == set(
        expected["closure"]
    )


def evaluate_numeric(cases: tuple[NumericCase, ...]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for case in cases:
        expected = numeric_oracle(case)
        state = build_numeric_state(case)
        audit = audit_numeric_state(case, state)
        counts["cases"] += 1
        counts["state_exact"] += int(_numeric_exact(state, expected))
        counts["decision_exact"] += int(numeric_decision(state) == expected["winner"])
        counts["audit_passed"] += int(audit["passed"])
        counts["steps"] += audit["step_total"]
        counts["step_exact"] += audit["step_exact"]
        counts["finals"] += audit["final_total"]
        counts["final_exact"] += audit["final_exact"]
        counts["comparisons"] += audit["comparison_total"]
        counts["comparison_antisymmetric"] += audit["comparison_antisymmetric"]
        observed = {candidate.handle: candidate.final for candidate in state.candidates}
        handles = tuple(observed)
        for left_index, left_handle in enumerate(handles):
            for right_handle in handles[left_index + 1 :]:
                counts["comparison_oracle_total"] += 1
                counts["comparison_oracle_exact"] += int(
                    compare_scalars(observed[left_handle], observed[right_handle])
                    == compare_scalars(
                        expected["totals"][left_handle], expected["totals"][right_handle]
                    )
                )
    total = counts["cases"]
    return {
        "case_count": total,
        "state_exact_rate": counts["state_exact"] / total,
        "decision_accuracy": counts["decision_exact"] / total,
        "audit_pass_rate": counts["audit_passed"] / total,
        "step_exact_rate": counts["step_exact"] / counts["steps"],
        "final_exact_rate": counts["final_exact"] / counts["finals"],
        "comparison_antisymmetry_rate": (
            counts["comparison_antisymmetric"] / counts["comparisons"]
        ),
        "comparison_oracle_rate": (
            counts["comparison_oracle_exact"] / counts["comparison_oracle_total"]
        ),
    }


def evaluate_relation(cases: tuple[RelationCase, ...]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for case in cases:
        expected = relation_oracle(case)
        state = build_relation_state(case)
        audit = audit_relation_state(case, state)
        decisions = relation_decisions(state, case.queries)
        counts["cases"] += 1
        counts["closure_exact"] += int(_relation_exact(case, state, expected))
        counts["audit_passed"] += int(audit["passed"])
        counts["queries"] += len(decisions)
        counts["query_exact"] += sum(
            observed == target
            for observed, target in zip(decisions, expected["decisions"], strict=True)
        )
        counts["composition"] += audit["composition_total"]
        counts["composition_exact"] += audit["composition_exact"]
        counts["direct"] += audit["direct_total"]
        counts["direct_exact"] += audit["direct_exact"]
        counts["antisymmetry"] += audit["closure_total"]
        counts["antisymmetry_exact"] += audit["antisymmetric_exact"]
    return {
        "case_count": counts["cases"],
        "closure_exact_rate": counts["closure_exact"] / counts["cases"],
        "query_accuracy": counts["query_exact"] / counts["queries"],
        "audit_pass_rate": counts["audit_passed"] / counts["cases"],
        "composition_exact_rate": counts["composition_exact"] / counts["composition"],
        "direct_exact_rate": counts["direct_exact"] / counts["direct"],
        "antisymmetry_rate": counts["antisymmetry_exact"] / counts["antisymmetry"],
    }


def _translated(case: NumericCase, shift: int) -> NumericCase:
    return NumericCase(
        case_id=f"{case.case_id}-translated",
        split=case.split,
        candidates=tuple(
            NumericCandidate(
                candidate.handle,
                (candidate.deltas[0] + shift, *candidate.deltas[1:]),
            )
            for candidate in case.candidates
        ),
    )


def _scaled(case: NumericCase, scale: int) -> NumericCase:
    return NumericCase(
        case_id=f"{case.case_id}-scaled",
        split=case.split,
        candidates=tuple(
            NumericCandidate(
                candidate.handle, tuple(delta * scale for delta in candidate.deltas)
            )
            for candidate in case.candidates
        ),
    )


def _renamed_numeric(case: NumericCase) -> tuple[NumericCase, dict[str, str]]:
    mapping = {
        candidate.handle: _handle("numeric-rename", FORMAL_SEED, index, 0)
        for index, candidate in enumerate(case.candidates)
    }
    return (
        NumericCase(
            case_id=f"{case.case_id}-renamed",
            split=case.split,
            candidates=tuple(
                NumericCandidate(mapping[candidate.handle], candidate.deltas)
                for candidate in case.candidates
            ),
        ),
        mapping,
    )


def evaluate_numeric_metamorphic(cases: tuple[NumericCase, ...]) -> dict[str, Any]:
    counts = Counter()
    for case in cases:
        base_winner = numeric_decision(build_numeric_state(case))
        translated = _translated(case, 4097)
        counts["translation"] += int(
            numeric_decision(build_numeric_state(translated)) == base_winner
        )
        scaled = _scaled(case, 3)
        counts["scaling"] += int(
            numeric_decision(build_numeric_state(scaled)) == base_winner
        )
        permuted = NumericCase(
            case_id=f"{case.case_id}-permuted",
            split=case.split,
            candidates=tuple(reversed(case.candidates)),
        )
        counts["candidate_permutation"] += int(
            numeric_decision(build_numeric_state(permuted)) == base_winner
        )
        renamed, mapping = _renamed_numeric(case)
        counts["handle_rename"] += int(
            numeric_decision(build_numeric_state(renamed)) == mapping[base_winner]
        )
    total = len(cases)
    return {
        name: counts[name] / total
        for name in ("translation", "scaling", "candidate_permutation", "handle_rename")
    }


def _renamed_relation(case: RelationCase) -> tuple[RelationCase, dict[str, str]]:
    mapping = {
        handle: _handle("relation-rename", FORMAL_SEED, index, 0)
        for index, handle in enumerate(case.handles)
    }
    return (
        RelationCase(
            case_id=f"{case.case_id}-renamed",
            split=case.split,
            handles=tuple(mapping[handle] for handle in case.handles),
            edges=tuple((mapping[source], mapping[target]) for source, target in case.edges),
            queries=tuple(
                RelationQuery(mapping[query.source], mapping[query.target])
                for query in case.queries
            ),
        ),
        mapping,
    )


def evaluate_relation_metamorphic(cases: tuple[RelationCase, ...]) -> dict[str, Any]:
    counts = Counter()
    for case in cases:
        base_state = build_relation_state(case)
        base_expected = relation_oracle(case)
        renamed, mapping = _renamed_relation(case)
        renamed_closure = set(build_relation_state(renamed).closure)
        mapped_closure = {
            (mapping[source], mapping[target]) for source, target in base_state.closure
        }
        counts["handle_rename"] += int(renamed_closure == mapped_closure)
        edge_permuted = RelationCase(
            case_id=f"{case.case_id}-edge-permuted",
            split=case.split,
            handles=case.handles,
            edges=tuple(reversed(case.edges)),
            queries=case.queries,
        )
        counts["edge_permutation"] += int(
            set(build_relation_state(edge_permuted).closure) == set(base_state.closure)
        )
        query_permuted = RelationCase(
            case_id=f"{case.case_id}-query-permuted",
            split=case.split,
            handles=case.handles,
            edges=case.edges,
            queries=tuple(reversed(case.queries)),
        )
        counts["query_permutation"] += int(
            relation_decisions(build_relation_state(query_permuted), query_permuted.queries)
            == tuple(reversed(base_expected["decisions"]))
        )
        redundant = sorted(set(base_expected["closure"]) - set(case.edges))[0]
        redundant_case = RelationCase(
            case_id=f"{case.case_id}-redundant",
            split=case.split,
            handles=case.handles,
            edges=(*case.edges, redundant),
            queries=case.queries,
        )
        counts["transitive_redundancy"] += int(
            set(build_relation_state(redundant_case).closure) == set(base_state.closure)
        )
    total = len(cases)
    return {
        name: counts[name] / total
        for name in (
            "handle_rename",
            "edge_permutation",
            "query_permutation",
            "transitive_redundancy",
        )
    }


def _safe_numeric_decision(state: NumericState) -> str | None:
    try:
        return numeric_decision(state)
    except ValueError:
        return None


def _numeric_target(case: NumericCase, name: str) -> NumericFaultTarget:
    expected = numeric_oracle(case)
    winner, runner = expected["ordered_handles"][:2]
    winner_source = next(candidate for candidate in case.candidates if candidate.handle == winner)
    if name in {"numeric_sign_flip", "numeric_drop_last"}:
        return NumericFaultTarget(winner, len(winner_source.deltas) - 1)
    if name == "numeric_winner_swap":
        return NumericFaultTarget(winner, secondary_handle=runner)
    return NumericFaultTarget(winner)


def _relation_target(case: RelationCase, name: str) -> RelationFaultTarget:
    expected = relation_oracle(case)
    closure = set(expected["closure"])
    true_queries = [
        (query.source, query.target)
        for query, decision in zip(case.queries, expected["decisions"], strict=True)
        if decision
    ]
    false_queries = [
        (query.source, query.target)
        for query, decision in zip(case.queries, expected["decisions"], strict=True)
        if not decision
    ]
    if name == "relation_drop_bridge":
        return RelationFaultTarget(true_queries[0])
    if name == "relation_reverse_pollution":
        reverse = next(edge for edge in false_queries if (edge[1], edge[0]) in closure)
        return RelationFaultTarget(reverse)
    if name == "relation_spurious":
        non_reverse = [edge for edge in false_queries if (edge[1], edge[0]) not in closure]
        return RelationFaultTarget((non_reverse or false_queries)[0])
    if name == "relation_handle_alias":
        edge = sorted(closure)[0]
        alias = _handle("unknown-alias", FORMAL_SEED, int(case.case_id[-4:]), 0)
        return RelationFaultTarget(edge=edge, alias_from=edge[0], alias_to=alias)
    assignments = tuple(
        ((query.source, query.target), index % 2 == 0)
        for index, query in enumerate(case.queries)
    )
    return RelationFaultTarget(edge=assignments[0][0], assignments=assignments)


NumericFault = Callable[[NumericCase, NumericState, NumericFaultTarget], NumericState]
RelationFault = Callable[[RelationCase, RelationState, RelationFaultTarget], RelationState]

_NUMERIC_FAULTS: dict[str, tuple[NumericFault, tuple[str, ...], bool]] = {
    "numeric_sign_flip": (numeric_sign_flip, ("step_exact", "decision_exact"), True),
    "numeric_drop_last": (numeric_drop_last, ("step_exact", "decision_exact"), True),
    "numeric_final_off_by_one": (numeric_final_off_by_one, ("final_exact",), False),
    "numeric_constant_collapse": (
        numeric_constant_collapse,
        ("state_exact", "decision_exact"),
        True,
    ),
    "numeric_winner_swap": (numeric_winner_swap, ("state_exact", "decision_exact"), True),
    "numeric_position_shortcut": (
        numeric_position_shortcut,
        ("state_exact", "decision_exact"),
        True,
    ),
}

_RELATION_FAULTS: dict[str, tuple[RelationFault, tuple[str, ...], bool]] = {
    "relation_drop_bridge": (relation_drop_bridge, ("closure_exact", "query_exact"), True),
    "relation_reverse_pollution": (
        relation_reverse_pollution,
        ("closure_exact", "query_exact", "antisymmetry"),
        True,
    ),
    "relation_spurious": (relation_spurious, ("closure_exact", "query_exact"), True),
    "relation_handle_alias": (relation_handle_alias, ("schema_rejection",), False),
    "relation_query_shortcut": (
        relation_query_shortcut,
        ("closure_exact", "query_exact"),
        True,
    ),
}


def evaluate_faults(
    numeric_cases: tuple[NumericCase, ...], relation_cases: tuple[RelationCase, ...]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, (fault, required_metrics, decision_required) in _NUMERIC_FAULTS.items():
        eligible = list(numeric_cases)
        if name == "numeric_position_shortcut":
            eligible = [
                case
                for case in eligible
                if numeric_oracle(case)["winner"] != case.candidates[0].handle
            ]
        selected = eligible[:64]
        metric_kills: Counter[str] = Counter()
        targets = []
        for case in selected:
            expected = numeric_oracle(case)
            target = _numeric_target(case, name)
            targets.append(asdict(target))
            mutated = fault(case, build_numeric_state(case), target)
            audit = audit_numeric_state(case, mutated)
            killed = {
                "state_exact": not _numeric_exact(mutated, expected),
                "step_exact": audit["step_exact"] != audit["step_total"],
                "final_exact": audit["final_exact"] != audit["final_total"],
                "decision_exact": _safe_numeric_decision(mutated) != expected["winner"],
                "schema_rejection": False,
            }
            for metric, value in killed.items():
                metric_kills[metric] += int(value)
        rates = {
            metric: metric_kills[metric] / len(selected) for metric in metric_kills
        }
        report[name] = {
            "case_count": len(selected),
            "target_source": "independent_accumulate_reference_manifest",
            "target_manifest_sha256": _fingerprint(targets),
            "target_samples": targets[:3],
            "state_detection_rate": rates["state_exact"],
            "decision_kill_rate": rates["decision_exact"],
            "decision_gate_required": decision_required,
            "required_metrics": required_metrics,
            "metric_kill_rates": rates,
            "metric_gate_passed": all(
                rates[metric] >= METRIC_KILL_MIN for metric in required_metrics
            ),
        }
    for name, (fault, required_metrics, decision_required) in _RELATION_FAULTS.items():
        eligible = list(relation_cases)
        if name == "relation_query_shortcut":
            eligible = [
                case
                for case in eligible
                if tuple(index % 2 == 0 for index in range(len(case.queries)))
                != relation_oracle(case)["decisions"]
            ]
        selected = eligible[:64]
        metric_kills: Counter[str] = Counter()
        targets = []
        for case in selected:
            expected = relation_oracle(case)
            target = _relation_target(case, name)
            targets.append(asdict(target))
            rejected = False
            try:
                mutated = fault(case, build_relation_state(case), target)
            except ValueError:
                rejected = True
                mutated = None
            if rejected:
                killed = {
                    "closure_exact": True,
                    "query_exact": False,
                    "antisymmetry": False,
                    "schema_rejection": True,
                }
            else:
                assert mutated is not None
                audit = audit_relation_state(case, mutated)
                killed = {
                    "closure_exact": not _relation_exact(case, mutated, expected),
                    "query_exact": (
                        relation_decisions(mutated, case.queries) != expected["decisions"]
                    ),
                    "antisymmetry": (
                        audit["antisymmetric_exact"] != audit["closure_total"]
                    ),
                    "schema_rejection": False,
                }
            for metric, value in killed.items():
                metric_kills[metric] += int(value)
        rates = {
            metric: metric_kills[metric] / len(selected) for metric in metric_kills
        }
        report[name] = {
            "case_count": len(selected),
            "target_source": "independent_bfs_reference_manifest",
            "target_manifest_sha256": _fingerprint(targets),
            "target_samples": targets[:3],
            "state_detection_rate": rates["closure_exact"],
            "decision_kill_rate": rates["query_exact"],
            "decision_gate_required": decision_required,
            "required_metrics": required_metrics,
            "metric_kill_rates": rates,
            "metric_gate_passed": all(
                rates[metric] >= METRIC_KILL_MIN for metric in required_metrics
            ),
        }
    return report


def _numeric_fixture_case(row: dict[str, Any]) -> NumericCase:
    return NumericCase(
        row["case_id"],
        row["split"],
        tuple(
            NumericCandidate(candidate["handle"], tuple(candidate["deltas"]))
            for candidate in row["candidates"]
        ),
    )


def _relation_fixture_case(row: dict[str, Any]) -> RelationCase:
    return RelationCase(
        row["case_id"],
        row["split"],
        tuple(row["handles"]),
        tuple(tuple(edge) for edge in row["edges"]),
        tuple(RelationQuery(query["source"], query["target"]) for query in row["queries"]),
    )


def _fixture_fault_probe(
    payload: dict[str, Any],
    numeric_rows: dict[str, dict[str, Any]],
    relation_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    expected_names = set(_NUMERIC_FAULTS) | set(_RELATION_FAULTS)
    for name, specification in payload["fault_targets"].items():
        target_data = specification["target"]
        rejected = False
        decision_killed = False
        if specification["domain"] == "numeric":
            source = numeric_rows[specification["case_id"]]
            case = _numeric_fixture_case(source)
            expected = {
                "prefixes": {key: tuple(value) for key, value in source["prefixes"].items()},
                "totals": source["totals"],
                "winner": source["winner"],
            }
            if name == "numeric_winner_swap":
                target = NumericFaultTarget(
                    target_data["winner_handle"],
                    secondary_handle=target_data["runner_up_handle"],
                )
            elif name == "numeric_constant_collapse":
                target = NumericFaultTarget(target_data["candidate_handles"][0])
            elif name == "numeric_position_shortcut":
                target = NumericFaultTarget(source["winner"])
            else:
                target = NumericFaultTarget(
                    target_data["candidate_handle"], target_data["delta_index"]
                )
            mutated = _NUMERIC_FAULTS[name][0](case, build_numeric_state(case), target)
            detected = not _numeric_exact(mutated, expected)
            decision_killed = _safe_numeric_decision(mutated) != source["winner"]
        else:
            source = relation_rows[specification["case_id"]]
            case = _relation_fixture_case(source)
            expected = {"closure": {tuple(edge) for edge in source["closure"]}}
            if name == "relation_drop_bridge":
                target = RelationFaultTarget(tuple(target_data["edge"]))
            elif name == "relation_reverse_pollution":
                target = RelationFaultTarget(tuple(target_data["reverse_edge"]))
            elif name == "relation_spurious":
                query = target_data["spurious_query"]
                target = RelationFaultTarget((query["source"], query["target"]))
            elif name == "relation_handle_alias":
                target = RelationFaultTarget(
                    edge=tuple(source["edges"][0]),
                    alias_from=target_data["source_handle"],
                    alias_to=target_data["alias_to_handle"],
                )
            else:
                assignments = tuple(
                    ((query["source"], query["target"]), decision)
                    for query, decision in zip(
                        target_data["query_order"],
                        target_data["shortcut_decisions"],
                        strict=True,
                    )
                )
                target = RelationFaultTarget(assignments[0][0], assignments=assignments)
            try:
                mutated = _RELATION_FAULTS[name][0](
                    case, build_relation_state(case), target
                )
            except ValueError:
                rejected = True
                mutated = None
            detected = rejected or not _relation_exact(case, mutated, expected)  # type: ignore[arg-type]
            if mutated is not None:
                decision_killed = (
                    relation_decisions(mutated, case.queries) != tuple(source["decisions"])
                )
        rows[name] = {
            "case_id": specification["case_id"],
            "target_sha256": _fingerprint(target_data),
            "detected": detected,
            "schema_rejected": rejected,
            "decision_killed": decision_killed,
        }
    return {
        "registry_complete": set(rows) == expected_names,
        "rows": rows,
        "passed": (
            set(rows) == expected_names
            and all(row["detected"] for row in rows.values())
            and rows["relation_handle_alias"]["schema_rejected"]
        ),
    }


def evaluate_hand_fixtures() -> dict[str, Any]:
    path = _repo_root() / HAND_FIXTURE_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    numeric_rows = {row["case_id"]: row for row in payload["numeric_cases"]}
    relation_rows = {row["case_id"]: row for row in payload["relation_cases"]}
    numeric_passed = 0
    for row in numeric_rows.values():
        case = _numeric_fixture_case(row)
        state = build_numeric_state(case)
        expected = {
            "prefixes": {key: tuple(value) for key, value in row["prefixes"].items()},
            "totals": row["totals"],
            "winner": row["winner"],
        }
        reference = numeric_oracle(case)
        numeric_passed += int(
            _numeric_exact(state, expected)
            and numeric_decision(state) == row["winner"]
            and reference["prefixes"] == expected["prefixes"]
            and reference["totals"] == expected["totals"]
            and reference["winner"] == row["winner"]
        )
    relation_passed = 0
    for row in relation_rows.values():
        case = _relation_fixture_case(row)
        state = build_relation_state(case)
        expected_closure = {tuple(edge) for edge in row["closure"]}
        expected_decisions = tuple(row["decisions"])
        reference = relation_oracle(case)
        relation_passed += int(
            set(state.closure) == expected_closure
            and relation_decisions(state, case.queries) == expected_decisions
            and reference["closure"] == expected_closure
            and reference["decisions"] == expected_decisions
        )
    fault_probe = _fixture_fault_probe(payload, numeric_rows, relation_rows)
    actual_hash = _file_sha256(path)
    provenance_valid = payload.get("provenance") == {
        "expected_values": "手工按逐项前缀累加与有向可达性独立核算",
        "generated_by": None,
        "measurement_or_generator_used": False,
    }
    return {
        "fixture": HAND_FIXTURE_RELATIVE.as_posix(),
        "expected_sha256": HAND_FIXTURE_SHA256,
        "actual_sha256": actual_hash,
        "hash_matches": actual_hash == HAND_FIXTURE_SHA256,
        "provenance_valid": provenance_valid,
        "numeric_case_count": len(numeric_rows),
        "numeric_exact_rate": numeric_passed / len(numeric_rows),
        "relation_case_count": len(relation_rows),
        "relation_exact_rate": relation_passed / len(relation_rows),
        "fault_registry": fault_probe,
        "passed": (
            actual_hash == HAND_FIXTURE_SHA256
            and provenance_valid
            and numeric_passed == len(numeric_rows)
            and relation_passed == len(relation_rows)
            and fault_probe["passed"]
        ),
    }


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Call):
            if isinstance(candidate.func, ast.Name):
                names.add(candidate.func.id)
            elif isinstance(candidate.func, ast.Attribute):
                names.add(candidate.func.attr)
    return names


def _schema_fail_closed_probe() -> dict[str, Any]:
    handles = ("hcycle001", "hcycle002", "hcycle003", "hcycle004")
    checks: dict[str, bool] = {}
    try:
        RelationCase(
            "cycle",
            "fixture",
            handles,
            ((handles[0], handles[1]), (handles[1], handles[0])),
            (RelationQuery(handles[0], handles[2]),),
        )
        checks["cycle_rejected"] = False
    except ValueError:
        checks["cycle_rejected"] = True
    try:
        RelationState("unknown-closure", handles, ((handles[0], "hunknown001"),))
        checks["unknown_closure_handle_rejected"] = False
    except ValueError:
        checks["unknown_closure_handle_rejected"] = True
    state = RelationState("known", handles, ((handles[0], handles[1]),))
    try:
        relation_decisions(state, (RelationQuery(handles[0], "hunknown001"),))
        checks["unknown_query_handle_rejected"] = False
    except ValueError:
        checks["unknown_query_handle_rejected"] = True
    return {"checks": checks, "passed": all(checks.values())}


def _schema_independence(fixture_probe: dict[str, Any]) -> dict[str, Any]:
    banned = {"answer", "label", "task_id", "oracle", "candidate_index"}
    public_types = (
        NumericCandidate,
        NumericCase,
        RelationQuery,
        RelationCase,
        NumericCandidateState,
        NumericState,
        RelationState,
    )
    field_names = {
        field.name for public_type in public_types for field in fields(public_type)
    }
    measurement_path = Path(inspect.getsourcefile(build_numeric_state) or "")
    reference_path = Path(inspect.getsourcefile(relation_oracle) or "")
    measurement_tree = ast.parse(measurement_path.read_text(encoding="utf-8"))
    reference_tree = ast.parse(reference_path.read_text(encoding="utf-8"))
    measurement_imports = {
        node.module or ""
        for node in ast.walk(measurement_tree)
        if isinstance(node, ast.ImportFrom)
    }
    reference_imports = {
        node.module or ""
        for node in ast.walk(reference_tree)
        if isinstance(node, ast.ImportFrom)
    }
    measured_numeric = _function_node(measurement_tree, "build_numeric_state")
    reference_numeric = _function_node(reference_tree, "numeric_oracle")
    measured_relation = _function_node(measurement_tree, "build_relation_state")
    reference_relation = _function_node(reference_tree, "relation_oracle")
    strict_schema = _schema_fail_closed_probe()
    boundaries = {
        "measurement_imports_reference": any(
            module.endswith("reference") for module in measurement_imports
        ),
        "reference_imports_measurement": any(
            module.endswith("measurement") for module in reference_imports
        ),
        "numeric_reference_calls_accumulate": "accumulate" in _called_names(reference_numeric),
        "numeric_measurement_uses_iterative_update": any(
            isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add)
            for node in ast.walk(measured_numeric)
        ),
        "relation_reference_calls_per_source_bfs": "_reachable" in _called_names(reference_relation),
        "relation_measurement_uses_fixed_point": any(
            isinstance(node, ast.While) for node in ast.walk(measured_relation)
        ),
        "relation_measurement_does_not_call_bfs": "_reachable" not in _called_names(measured_relation),
    }
    return {
        "public_fields": sorted(field_names),
        "banned_fields_absent": not bool(field_names & banned),
        "module_boundaries": boundaries,
        "strict_schema": strict_schema,
        "hand_fixture_probe": fixture_probe,
        "passed": (
            not bool(field_names & banned)
            and not boundaries["measurement_imports_reference"]
            and not boundaries["reference_imports_measurement"]
            and all(
                boundaries[name]
                for name in (
                    "numeric_reference_calls_accumulate",
                    "numeric_measurement_uses_iterative_update",
                    "relation_reference_calls_per_source_bfs",
                    "relation_measurement_uses_fixed_point",
                    "relation_measurement_does_not_call_bfs",
                )
            )
            and strict_schema["passed"]
            and fixture_probe["passed"]
        ),
    }


def _path_exists_without_edge(
    case: RelationCase, removed: tuple[str, str]
) -> bool:
    adjacency = {handle: set() for handle in case.handles}
    for edge in case.edges:
        if edge != removed:
            adjacency[edge[0]].add(edge[1])
    queue = deque([removed[0]])
    visited = {removed[0]}
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor == removed[1]:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def _component_count(case: RelationCase) -> int:
    adjacency = {handle: set() for handle in case.handles}
    for source, target in case.edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    unseen = set(case.handles)
    components = 0
    while unseen:
        components += 1
        queue = [unseen.pop()]
        while queue:
            current = queue.pop()
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            queue.extend(neighbors)
    return components


def _relation_topology(cases: tuple[RelationCase, ...]) -> dict[str, Any]:
    branch_cases = merge_cases = multi_component_cases = redundant_direct_cases = 0
    patterns = Counter()
    query_counts = Counter()
    true_queries = total_queries = 0
    for case in cases:
        outdegree = Counter(source for source, _ in case.edges)
        indegree = Counter(target for _, target in case.edges)
        branch_cases += int(max(outdegree.values()) >= 2)
        merge_cases += int(max(indegree.values()) >= 2)
        multi_component_cases += int(_component_count(case) >= 2)
        redundant_direct_cases += int(
            any(_path_exists_without_edge(case, edge) for edge in case.edges)
        )
        decisions = relation_oracle(case)["decisions"]
        patterns["".join("1" if value else "0" for value in decisions)] += 1
        query_counts[len(decisions)] += 1
        true_queries += sum(decisions)
        total_queries += len(decisions)
    return {
        "branch_case_count": branch_cases,
        "merge_case_count": merge_cases,
        "multi_component_case_count": multi_component_cases,
        "redundant_direct_case_count": redundant_direct_cases,
        "query_count_histogram": dict(sorted(query_counts.items())),
        "query_truth_pattern_count": len(patterns),
        "query_truth_pattern_histogram": dict(sorted(patterns.items())),
        "positive_query_rate": true_queries / total_queries,
    }


def _case_manifest(
    numeric: dict[str, tuple[NumericCase, ...]],
    relation: dict[str, tuple[RelationCase, ...]],
) -> dict[str, Any]:
    numeric_fingerprints = {
        split: [_fingerprint(case.to_dict()) for case in cases]
        for split, cases in numeric.items()
    }
    relation_fingerprints = {
        split: [_fingerprint(case.to_dict()) for case in cases]
        for split, cases in relation.items()
    }
    numeric_all = [case for cases in numeric.values() for case in cases]
    relation_all = [case for cases in relation.values() for case in cases]
    winner_positions = Counter()
    mixed_sign_cases = 0
    for case in numeric_all:
        winner = numeric_oracle(case)["winner"]
        winner_positions[next(
            index for index, candidate in enumerate(case.candidates) if candidate.handle == winner
        )] += 1
        mixed_sign_cases += int(
            all(
                any(delta > 0 for delta in candidate.deltas)
                and any(delta < 0 for delta in candidate.deltas)
                for candidate in case.candidates
            )
        )
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-nr1.case-manifest.v2",
        "seed": FORMAL_SEED,
        "numeric": {
            "counts": {split: len(cases) for split, cases in numeric.items()},
            "fingerprints": numeric_fingerprints,
            "candidate_histogram": dict(sorted(Counter(
                len(case.candidates) for case in numeric_all
            ).items())),
            "horizon_histogram": dict(sorted(Counter(
                len(case.candidates[0].deltas) for case in numeric_all
            ).items())),
            "winner_position_histogram": dict(sorted(winner_positions.items())),
            "mixed_sign_case_rate": mixed_sign_cases / len(numeric_all),
            "qualification_max_abs_delta": max(
                abs(delta)
                for case in numeric["qualification"]
                for candidate in case.candidates
                for delta in candidate.deltas
            ),
            "heldout_min_abs_delta": min(
                abs(delta)
                for case in numeric["heldout"]
                for candidate in case.candidates
                for delta in candidate.deltas
            ),
        },
        "relation": {
            "counts": {split: len(cases) for split, cases in relation.items()},
            "fingerprints": relation_fingerprints,
            "handle_histogram": dict(sorted(Counter(
                len(case.handles) for case in relation_all
            ).items())),
            "topology": {
                split: _relation_topology(cases) for split, cases in relation.items()
            },
        },
        "overlap": {
            "numeric": len(
                set(numeric_fingerprints["qualification"])
                & set(numeric_fingerprints["heldout"])
            ),
            "relation": len(
                set(relation_fingerprints["qualification"])
                & set(relation_fingerprints["heldout"])
            ),
        },
    }


def _all_one(metrics: dict[str, Any], excluded: set[str] | None = None) -> bool:
    excluded = excluded or set()
    return all(
        value == 1.0
        for key, value in metrics.items()
        if key not in excluded and isinstance(value, float)
    )


def run_qualification_bundle() -> dict[str, Any]:
    numeric = {
        split: build_numeric_cases(FORMAL_SEED, split, count)
        for split, count in NUMERIC_COUNTS.items()
    }
    relation = {
        split: build_relation_cases(FORMAL_SEED, split, count)
        for split, count in RELATION_COUNTS.items()
    }
    fixture_probe = evaluate_hand_fixtures()
    manifest = _case_manifest(numeric, relation)
    numeric_metrics = {
        split: evaluate_numeric(cases) for split, cases in numeric.items()
    }
    relation_metrics = {
        split: evaluate_relation(cases) for split, cases in relation.items()
    }
    metamorphic = {
        "numeric": evaluate_numeric_metamorphic(
            numeric["qualification"][:128] + numeric["heldout"][:128]
        ),
        "relation": evaluate_relation_metamorphic(
            relation["qualification"][:128] + relation["heldout"][:128]
        ),
    }
    faults = evaluate_faults(
        numeric["qualification"] + numeric["heldout"],
        relation["qualification"] + relation["heldout"],
    )
    independence = _schema_independence(fixture_probe)
    topology = manifest["relation"]["topology"]
    topology_gate = all(
        row["branch_case_count"] > 0
        and row["merge_case_count"] > 0
        and row["multi_component_case_count"] > 0
        and row["redundant_direct_case_count"] > 0
        and len(row["query_count_histogram"]) == 5
        and row["query_truth_pattern_count"] >= 16
        and 0.35 <= row["positive_query_rate"] <= 0.65
        for row in topology.values()
    )
    gates = {
        "N02_schema_reference_fixture_independence": independence["passed"],
        "N03_numeric_qualification": (
            all(_all_one(metrics, {"case_count"}) for metrics in numeric_metrics.values())
            and _all_one(metamorphic["numeric"])
        ),
        "N04_relation_topology_qualification": (
            all(_all_one(metrics, {"case_count"}) for metrics in relation_metrics.values())
            and _all_one(metamorphic["relation"])
            and topology_gate
        ),
        "N05_adversarial_metric_decision_power": all(
            row["state_detection_rate"] == 1.0
            and row["metric_gate_passed"]
            and (
                not row["decision_gate_required"]
                or row["decision_kill_rate"] >= DECISION_KILL_MIN
            )
            for row in faults.values()
        ),
        "N06_holdout_non_leakage": (
            manifest["overlap"] == {"numeric": 0, "relation": 0}
            and manifest["numeric"]["mixed_sign_case_rate"] == 1.0
            and manifest["numeric"]["qualification_max_abs_delta"] <= 32
            and manifest["numeric"]["heldout_min_abs_delta"] >= 64
            and max(len(case.candidates) for case in numeric["qualification"]) <= 5
            and min(len(case.candidates) for case in numeric["heldout"]) >= 6
            and max(len(case.candidates[0].deltas) for case in numeric["qualification"]) <= 5
            and min(len(case.candidates[0].deltas) for case in numeric["heldout"]) >= 7
            and max(len(case.handles) for case in relation["qualification"]) <= 8
            and min(len(case.handles) for case in relation["heldout"]) >= 9
        ),
    }
    return {
        "schema_version": "yggdrasil.v2-r1r.p1-nr1.qualification-bundle.v2",
        "seed": FORMAL_SEED,
        "training_performed": False,
        "audit_statistics_fit": False,
        "case_manifest": manifest,
        "independence": independence,
        "hand_fixture_probe": fixture_probe,
        "numeric_metrics": numeric_metrics,
        "relation_metrics": relation_metrics,
        "metamorphic": metamorphic,
        "fault_matrix": faults,
        "gates": gates,
        "passed": all(gates.values()),
    }
