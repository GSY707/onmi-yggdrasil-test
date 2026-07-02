from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Mapping, Sequence


Vector = tuple[float, ...]


class DimensionMismatch(ValueError):
    """Raised when latent tokens do not share one locked feature dimension."""


class HighEntropyPayloadRejected(ValueError):
    """Raised when memory storage attempts to persist raw multimodal payloads."""


def _as_vector(values: Sequence[float]) -> Vector:
    return tuple(float(value) for value in values)


def _validate_vectors(vectors: Sequence[Sequence[float]], d_model: int | None = None) -> int:
    if not vectors:
        raise ValueError("at least one vector is required")
    expected = d_model if d_model is not None else len(vectors[0])
    if expected <= 0:
        raise ValueError("d_model must be positive")
    for vector in vectors:
        if len(vector) != expected:
            raise DimensionMismatch(f"expected d_model={expected}, got {len(vector)}")
    return expected


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise DimensionMismatch(f"cannot dot vectors with dimensions {len(left)} and {len(right)}")
    return sum(a * b for a, b in zip(left, right))


def _weighted_sum(weighted_vectors: Sequence[tuple[float, Sequence[float]]], d_model: int) -> Vector:
    result = [0.0] * d_model
    for weight, vector in weighted_vectors:
        if len(vector) != d_model:
            raise DimensionMismatch(f"expected d_model={d_model}, got {len(vector)}")
        for index, value in enumerate(vector):
            result[index] += weight * value
    return tuple(result)


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    offset = max(scores)
    exps = [math.exp(score - offset) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def mean_pool_lod(tokens: Sequence[Sequence[float]], group_size: int) -> list[Vector]:
    """Compress high-LOD tokens by count while preserving the locked d_model."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    d_model = _validate_vectors(tokens)
    pooled: list[Vector] = []
    for start in range(0, len(tokens), group_size):
        group = tokens[start : start + group_size]
        scale = 1.0 / len(group)
        pooled.append(_weighted_sum([(scale, token) for token in group], d_model))
    return pooled


def lod_embedding(level: int, d_model: int, *, magnitude: float = 1.0) -> Vector:
    """Return an exactly orthogonal one-hot LOD tag for level < d_model."""

    if level < 0:
        raise ValueError("level must be non-negative")
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if level >= d_model:
        raise ValueError("exact one-hot LOD embeddings require level < d_model")
    return tuple(magnitude if index == level else 0.0 for index in range(d_model))


def add_lod_embedding(tokens: Sequence[Sequence[float]], level: int) -> list[Vector]:
    d_model = _validate_vectors(tokens)
    tag = lod_embedding(level, d_model)
    return [tuple(value + tag[index] for index, value in enumerate(token)) for token in tokens]


@dataclass(frozen=True)
class ResidualLatentBridge:
    """Zero-initialized residual bridge between text supervision and latent tokens."""

    d_model: int
    gate: float = 0.0

    def fuse(self, text_tokens: Sequence[Sequence[float]], latent_tokens: Sequence[Sequence[float]]) -> list[Vector]:
        if len(text_tokens) != len(latent_tokens):
            raise ValueError("text and latent token counts must match for residual fusion")
        _validate_vectors(text_tokens, self.d_model)
        _validate_vectors(latent_tokens, self.d_model)
        return [
            tuple(text_value + self.gate * latent_value for text_value, latent_value in zip(text, latent))
            for text, latent in zip(text_tokens, latent_tokens)
        ]


@dataclass(frozen=True)
class ExpertOutput:
    name: str
    tokens: tuple[Vector, ...]
    route_weight: float

    @classmethod
    def from_tokens(cls, name: str, tokens: Sequence[Sequence[float]], route_weight: float) -> "ExpertOutput":
        if route_weight < 0.0:
            raise ValueError("route_weight must be non-negative")
        normalized = tuple(_as_vector(token) for token in tokens)
        if normalized:
            _validate_vectors(normalized)
        return cls(name=name, tokens=normalized, route_weight=float(route_weight))


@dataclass(frozen=True)
class CrossAttentionPump:
    """Variable-input, bounded-output resampler for MoE expert products."""

    d_model: int

    def weighted_output_length(self, expert_outputs: Sequence[ExpertOutput]) -> int:
        if not expert_outputs:
            return 0
        weighted_length = sum(output.route_weight * len(output.tokens) for output in expert_outputs)
        return max(1, math.ceil(weighted_length))

    def pump(
        self,
        expert_outputs: Sequence[ExpertOutput],
        *,
        output_length: int | None = None,
    ) -> list[Vector]:
        sources: list[tuple[Vector, float]] = []
        for output in expert_outputs:
            if output.route_weight <= 0.0:
                continue
            _validate_vectors(output.tokens, self.d_model)
            sources.extend((token, output.route_weight) for token in output.tokens)
        if not sources:
            return []
        length = self.weighted_output_length(expert_outputs) if output_length is None else output_length
        if length < 0:
            raise ValueError("output_length must be non-negative")

        pumped: list[Vector] = []
        for slot in range(length):
            query = self._query(slot)
            scores = [
                _dot(query, token) / math.sqrt(self.d_model) + math.log(route_weight)
                for token, route_weight in sources
            ]
            attention = _softmax(scores)
            pumped.append(_weighted_sum(list(zip(attention, [token for token, _ in sources])), self.d_model))
        return pumped

    def _query(self, slot: int) -> Vector:
        return tuple((((slot + 1) * (index + 3)) % 11 - 5) / 5.0 for index in range(self.d_model))


@dataclass(frozen=True)
class MediaSegment:
    id: str
    start_ms: int
    end_ms: int
    summary: str
    uri: str
    keywords: tuple[str, ...]
    token_cost: int


@dataclass(frozen=True)
class SampledSlice:
    segment_id: str
    summary: str
    uri: str
    token_cost: int


@dataclass(frozen=True)
class HierarchicalMediaIndex:
    """Macro-summary first, high-detail slices only after active attention."""

    segments: tuple[MediaSegment, ...]
    group_size: int = 4

    def macro_summary(self) -> list[str]:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        summaries: list[str] = []
        for start in range(0, len(self.segments), self.group_size):
            group = self.segments[start : start + self.group_size]
            if not group:
                continue
            summaries.append(
                f"{group[0].start_ms}-{group[-1].end_ms}ms: "
                + " / ".join(segment.summary for segment in group)
            )
        return summaries

    def active_sample(self, attention_instruction: str, token_budget: int) -> list[SampledSlice]:
        if token_budget <= 0:
            return []
        query_terms = _terms(attention_instruction)
        ranked = sorted(
            self.segments,
            key=lambda segment: (-self._score(segment, query_terms), segment.token_cost, segment.start_ms),
        )
        selected: list[SampledSlice] = []
        spent = 0
        for segment in ranked:
            score = self._score(segment, query_terms)
            if score <= 0:
                continue
            if spent + segment.token_cost > token_budget:
                continue
            selected.append(
                SampledSlice(
                    segment_id=segment.id,
                    summary=segment.summary,
                    uri=segment.uri,
                    token_cost=segment.token_cost,
                )
            )
            spent += segment.token_cost
        return selected

    @staticmethod
    def _score(segment: MediaSegment, query_terms: set[str]) -> int:
        segment_terms = _terms(segment.summary).union(term.lower() for term in segment.keywords)
        return len(segment_terms.intersection(query_terms))


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_\-]+", text.lower()))


@dataclass
class MemoryTreeNode:
    id: str
    name: str
    content_summary: str
    parent_id: str | None
    detail_level: int
    uri_refs: tuple[str, ...] = ()
    child_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryTreeRelation:
    id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    description: str


@dataclass(frozen=True)
class MemoryTreeRead:
    focus_node_id: str
    nodes: tuple[MemoryTreeNode, ...]
    relations: tuple[MemoryTreeRelation, ...]
    child_names_by_node: Mapping[str, tuple[str, ...]]
    relation_target_names_by_node: Mapping[str, tuple[str, ...]]


@dataclass
class MemoryTree:
    """World Tree memory: LOD nodes plus directed relation edges, not a RAG document store."""

    max_summary_chars: int = 280
    _nodes: dict[str, MemoryTreeNode] = field(default_factory=dict)
    _relations: dict[str, MemoryTreeRelation] = field(default_factory=dict)

    def add_node(
        self,
        *,
        name: str,
        content_summary: str,
        parent_id: str | None = None,
        uri_refs: Sequence[str] = (),
        raw_payload: bytes | None = None,
    ) -> MemoryTreeNode:
        if raw_payload:
            raise HighEntropyPayloadRejected("raw multimodal payloads must stay in external storage")
        if not name.strip():
            raise ValueError("name is required")
        if not content_summary.strip():
            raise ValueError("content_summary is required")
        if len(content_summary) > self.max_summary_chars:
            raise ValueError("content_summary is too long for a memory tree node")
        if parent_id is not None and parent_id not in self._nodes:
            raise ValueError(f"unknown parent memory node: {parent_id}")

        node_id = f"mem-{len(self._nodes) + 1}"
        detail_level = 0 if parent_id is None else self._nodes[parent_id].detail_level + 1
        node = MemoryTreeNode(
            id=node_id,
            name=name,
            content_summary=content_summary,
            parent_id=parent_id,
            detail_level=detail_level,
            uri_refs=tuple(uri_refs),
        )
        self._nodes[node_id] = node
        if parent_id is not None:
            self._nodes[parent_id].child_ids.append(node_id)
        return node

    def add_relation(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        description: str,
    ) -> MemoryTreeRelation:
        if source_node_id not in self._nodes:
            raise ValueError(f"unknown source memory node: {source_node_id}")
        if target_node_id not in self._nodes:
            raise ValueError(f"unknown target memory node: {target_node_id}")
        relation_id = f"rel-{len(self._relations) + 1}"
        relation = MemoryTreeRelation(
            id=relation_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=relation_type,
            description=description,
        )
        self._relations[relation_id] = relation
        return relation

    def get(self, node_id: str) -> MemoryTreeNode:
        return self._nodes[node_id]

    def read(self, node_id: str, *, depth: int = 1, breadth: int = 1) -> MemoryTreeRead:
        if node_id not in self._nodes:
            raise ValueError(f"unknown memory node: {node_id}")
        if depth < 0 or breadth < 0:
            raise ValueError("depth and breadth must be non-negative")

        node_ids = self._descendants_to_depth(node_id, depth)
        relation_ids: set[str] = set()
        frontier = set(node_ids)
        for _ in range(breadth):
            next_frontier: set[str] = set()
            for relation in self._relations.values():
                if relation.source_node_id in frontier:
                    relation_ids.add(relation.id)
                    next_frontier.add(relation.target_node_id)
            node_ids.extend(node for node in sorted(next_frontier) if node not in node_ids)
            frontier = next_frontier

        nodes = tuple(self._nodes[item] for item in node_ids)
        relations = tuple(self._relations[item] for item in sorted(relation_ids))
        child_names_by_node = {
            item: tuple(self._nodes[child_id].name for child_id in self._nodes[item].child_ids)
            for item in node_ids
        }
        relation_target_names_by_node = {
            item: tuple(
                self._nodes[relation.target_node_id].name
                for relation in self._relations.values()
                if relation.source_node_id == item
            )
            for item in node_ids
        }
        return MemoryTreeRead(
            focus_node_id=node_id,
            nodes=nodes,
            relations=relations,
            child_names_by_node=child_names_by_node,
            relation_target_names_by_node=relation_target_names_by_node,
        )

    def _descendants_to_depth(self, node_id: str, depth: int) -> list[str]:
        result = [node_id]
        if depth == 0:
            return result
        for child_id in self._nodes[node_id].child_ids:
            result.extend(self._descendants_to_depth(child_id, depth - 1))
        return result


@dataclass
class CacheNode:
    id: str
    parent_id: str | None
    token_count: int
    child_ids: list[str] = field(default_factory=list)


@dataclass
class KVCacheTree:
    """Logical tree index for coarse branch-level KV cache reclamation."""

    _nodes: dict[str, CacheNode] = field(default_factory=dict)

    def add_node(self, node_id: str, *, parent_id: str | None, token_count: int) -> None:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        if node_id in self._nodes:
            raise ValueError(f"duplicate cache node: {node_id}")
        if parent_id is not None and parent_id not in self._nodes:
            raise ValueError(f"unknown parent cache node: {parent_id}")
        self._nodes[node_id] = CacheNode(id=node_id, parent_id=parent_id, token_count=token_count)
        if parent_id is not None:
            self._nodes[parent_id].child_ids.append(node_id)

    def active_token_count(self) -> int:
        return sum(node.token_count for node in self._nodes.values())

    def path_token_count(self, node_id: str) -> int:
        total = 0
        current = self._nodes[node_id]
        while True:
            total += current.token_count
            if current.parent_id is None:
                return total
            current = self._nodes[current.parent_id]

    def drop_branch(self, node_id: str) -> int:
        if node_id not in self._nodes:
            raise ValueError(f"unknown cache node: {node_id}")
        if self._nodes[node_id].parent_id is None:
            raise ValueError("dropping the root branch is intentionally disallowed")
        to_drop = self._descendants_inclusive(node_id)
        freed = sum(self._nodes[item].token_count for item in to_drop)
        parent_id = self._nodes[node_id].parent_id
        if parent_id is not None:
            self._nodes[parent_id].child_ids = [
                child_id for child_id in self._nodes[parent_id].child_ids if child_id != node_id
            ]
        for item in to_drop:
            del self._nodes[item]
        return freed

    def _descendants_inclusive(self, node_id: str) -> list[str]:
        result = [node_id]
        for child_id in list(self._nodes[node_id].child_ids):
            result.extend(self._descendants_inclusive(child_id))
        return result


@dataclass(frozen=True)
class Constitution:
    rule_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReasoningTrace:
    id: str
    summary: str
    checks: Mapping[str, bool]


@dataclass(frozen=True)
class LoRAProposal:
    accepted_trace_ids: tuple[str, ...]
    rejected_trace_ids: tuple[str, ...]
    score: float
    base_model_mutated: bool = False


@dataclass(frozen=True)
class OfflineAlignmentLab:
    constitution: Constitution

    def reflect(self, traces: Sequence[ReasoningTrace], *, live_mode: bool = False) -> LoRAProposal:
        if live_mode:
            raise ValueError("alignment reflection must run offline, not during live interaction")
        accepted: list[str] = []
        rejected: list[str] = []
        for trace in traces:
            if all(trace.checks.get(rule_id, False) for rule_id in self.constitution.rule_ids):
                accepted.append(trace.id)
            else:
                rejected.append(trace.id)
        score = len(accepted) / len(traces) if traces else 0.0
        return LoRAProposal(
            accepted_trace_ids=tuple(accepted),
            rejected_trace_ids=tuple(rejected),
            score=score,
        )


@dataclass(frozen=True)
class ResourceState:
    time_budget_ms: int
    compute_budget_units: int
    priority: float = 1.0


@dataclass(frozen=True)
class GenerationPlan:
    max_tree_depth: int
    denoising_steps: int
    detail_label: str


def progressive_generation_plan(resource_state: ResourceState) -> GenerationPlan:
    if resource_state.time_budget_ms < 0 or resource_state.compute_budget_units < 0:
        raise ValueError("resource budgets must be non-negative")
    normalized_time = min(resource_state.time_budget_ms / 10_000.0, 1.0)
    normalized_compute = min(resource_state.compute_budget_units / 1_000.0, 1.0)
    pressure = max(0.0, min(1.0, ((normalized_time + normalized_compute) / 2.0) * resource_state.priority))
    if pressure < 0.25:
        return GenerationPlan(max_tree_depth=1, denoising_steps=4, detail_label="sketch")
    if pressure < 0.6:
        return GenerationPlan(max_tree_depth=2, denoising_steps=12, detail_label="draft")
    return GenerationPlan(max_tree_depth=4, denoising_steps=32, detail_label="full")
