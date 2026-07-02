import math

import pytest

from latent_space_agent_feasibility import (
    Constitution,
    CrossAttentionPump,
    DimensionMismatch,
    ExpertOutput,
    HierarchicalMediaIndex,
    HighEntropyPayloadRejected,
    KVCacheTree,
    MemoryTree,
    MediaSegment,
    OfflineAlignmentLab,
    ReasoningTrace,
    ResidualLatentBridge,
    ResourceState,
    add_lod_embedding,
    lod_embedding,
    mean_pool_lod,
    progressive_generation_plan,
)


def test_zero_initialized_residual_bridge_preserves_text_path():
    bridge = ResidualLatentBridge(d_model=3, gate=0.0)

    fused = bridge.fuse(
        text_tokens=[(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)],
        latent_tokens=[(10.0, 10.0, 10.0), (20.0, 20.0, 20.0)],
    )

    assert fused == [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)]


def test_residual_bridge_rejects_dimension_drift():
    bridge = ResidualLatentBridge(d_model=3)

    with pytest.raises(DimensionMismatch):
        bridge.fuse(text_tokens=[(1.0, 2.0, 3.0)], latent_tokens=[(1.0, 2.0)])


def test_lod_pooling_changes_token_count_not_feature_dimension():
    pooled = mean_pool_lod(
        tokens=[
            (1.0, 0.0, 0.0, 0.0),
            (3.0, 0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0, 0.0),
            (0.0, 4.0, 0.0, 0.0),
        ],
        group_size=2,
    )

    assert len(pooled) == 2
    assert all(len(token) == 4 for token in pooled)
    assert pooled[0] == (2.0, 0.0, 0.0, 0.0)


def test_lod_embeddings_are_orthogonal_within_d_model_capacity():
    level_0 = lod_embedding(0, 4)
    level_1 = lod_embedding(1, 4)

    assert sum(left * right for left, right in zip(level_0, level_1)) == 0.0
    assert add_lod_embedding([(1.0, 1.0, 1.0, 1.0)], 1) == [(1.0, 2.0, 1.0, 1.0)]

    with pytest.raises(ValueError):
        lod_embedding(4, 4)


def test_attention_pump_resamples_variable_expert_lengths():
    outputs = [
        ExpertOutput.from_tokens("logic", [(1.0, 0.0), (0.8, 0.2)], route_weight=0.25),
        ExpertOutput.from_tokens(
            "vision",
            [(0.0, 1.0), (0.2, 0.8), (0.1, 0.9), (0.3, 0.7)],
            route_weight=0.75,
        ),
    ]
    pump = CrossAttentionPump(d_model=2)

    pumped = pump.pump(outputs)

    assert len(pumped) == math.ceil(0.25 * 2 + 0.75 * 4)
    assert all(len(token) == 2 for token in pumped)


def test_attention_pump_ignores_zero_weight_expert_outputs():
    pump = CrossAttentionPump(d_model=2)

    pumped = pump.pump(
        [
            ExpertOutput.from_tokens("inactive", [(100.0, 100.0)], route_weight=0.0),
            ExpertOutput.from_tokens("active", [(1.0, 0.0)], route_weight=1.0),
        ]
    )

    assert pumped == [(1.0, 0.0)]


def test_active_sampling_reads_macro_first_then_targeted_detail():
    index = HierarchicalMediaIndex(
        segments=(
            MediaSegment("s1", 0, 1000, "opening scene", "file:///video#s1", ("intro",), 12),
            MediaSegment("s2", 1000, 2000, "robot arm picks red cube", "file:///video#s2", ("robot", "cube"), 18),
            MediaSegment("s3", 2000, 3000, "wide room pan", "file:///video#s3", ("room",), 15),
        ),
        group_size=2,
    )

    macro = index.macro_summary()
    detail = index.active_sample("inspect robot cube motion", token_budget=20)

    assert len(macro) == 2
    assert [item.segment_id for item in detail] == ["s2"]
    assert sum(item.token_cost for item in detail) <= 20


def test_memory_tree_uses_lod_nodes_relations_and_uri_refs_not_raw_payloads():
    tree = MemoryTree()
    root = tree.add_node(name="我要干什么", content_summary="当前任务工作状态根节点")
    video = tree.add_node(
        name="机器人视频观察",
        content_summary="机器人在 1000-2000ms 抓取红色方块",
        parent_id=root.id,
        uri_refs=("file:///video#s2",),
    )
    decision = tree.add_node(name="抓取策略", content_summary="优先检查手臂路径与方块接触点", parent_id=root.id)
    relation = tree.add_relation(
        source_node_id=decision.id,
        target_node_id=video.id,
        relation_type="evidence",
        description="抓取策略需要读取视频证据节点",
    )

    read = tree.read(root.id, depth=1, breadth=1)

    assert video.detail_level == 1
    assert tree.get(video.id).uri_refs == ("file:///video#s2",)
    assert relation in read.relations
    assert read.child_names_by_node[root.id] == ("机器人视频观察", "抓取策略")
    assert read.relation_target_names_by_node[decision.id] == ("机器人视频观察",)

    with pytest.raises(HighEntropyPayloadRejected):
        tree.add_node(
            name="原始帧批次",
            content_summary="原始视频帧不得进入记忆树",
            parent_id=video.id,
            raw_payload=b"\x00\x01\x02",
        )


def test_kv_cache_branch_drop_reclaims_dead_branch_without_touching_trunk():
    cache = KVCacheTree()
    cache.add_node("root", parent_id=None, token_count=100)
    cache.add_node("branch-a", parent_id="root", token_count=40)
    cache.add_node("branch-a-leaf", parent_id="branch-a", token_count=25)
    cache.add_node("branch-b", parent_id="root", token_count=30)

    freed = cache.drop_branch("branch-a")

    assert freed == 65
    assert cache.active_token_count() == 130
    assert cache.path_token_count("branch-b") == 130


def test_alignment_lab_only_proposes_offline_lora_updates():
    lab = OfflineAlignmentLab(Constitution(rule_ids=("truthful", "non_destructive")))

    proposal = lab.reflect(
        [
            ReasoningTrace("t1", "good trace", {"truthful": True, "non_destructive": True}),
            ReasoningTrace("t2", "bad trace", {"truthful": False, "non_destructive": True}),
        ]
    )

    assert proposal.accepted_trace_ids == ("t1",)
    assert proposal.rejected_trace_ids == ("t2",)
    assert proposal.base_model_mutated is False

    with pytest.raises(ValueError):
        lab.reflect([], live_mode=True)


def test_progressive_generation_degrades_with_low_resources():
    low = progressive_generation_plan(ResourceState(time_budget_ms=500, compute_budget_units=50))
    high = progressive_generation_plan(ResourceState(time_budget_ms=10_000, compute_budget_units=1_000))

    assert low.detail_label == "sketch"
    assert high.detail_label == "full"
    assert high.max_tree_depth > low.max_tree_depth
    assert high.denoising_steps > low.denoising_steps
