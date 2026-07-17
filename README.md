# Project-Yggdrasil V2 研究工作区

本仓库当前只保留 Project-Yggdrasil V2 的架构真源、路线决策、V2-A 分层实验和最近测试计划。A1.8 以 T1–16 均衡训练使结构化 core 通过 T24，A1.9 证明 oracle-role-segmented Qwen hidden 可稳定驱动该 frozen core。A1.10–A1.18B 完成了 relation transition、closure 与训练期逐步全局状态信用分配的故障定位；A1.19H 进一步把固定三寄存器正控制切换为 equality-only opaque handles + shared continuous payload，并在训练 `N=2,3,4`、heldout `N=5` 上形成 H1/H2 formal/causal `3/3`。A1.20B 的无 oracle learned full-text Boundary 在 heldout validation 失败后，A1.20C 又加入分层 entity/program compiler 与 straight-through execution credit。A1.20C 已把所有训练 token anchor 学到 `1.0`，但目标臂 overfit32 的 trajectory/final-state 仍只有 `0.875/0.90625`；实际梯度审计显示 state 与 local compiler objective 强负冲突。A1.19H core 正证据保留，但 full-text Boundary 与 V2-A 均未通过。

当前主线已经从旧 Stage A—AV-J-C 直接切换为：

1. V2-A：先在成熟文本基座上比较显式文本 CoT、单向量 latent recurrence 和多向量 latent recurrence。
2. V2-B：只有 V2-A 通过后，才验证文本/视觉/动作融合、Boundary-MoE 与 FFN-MoE。

旧实验不是当前实现路线。原有阶段报告、代理脚本、测试、源码、artifact、旧 README 和思考稿已完整归档到 [`archive/legacy-proxy-route-2026-07-11/`](archive/legacy-proxy-route-2026-07-11/)，保留作历史证据，不建立兼容入口。

## 当前入口

- [V2 架构白皮书](docs/Project-Yggdrasil%20%E5%A4%9A%E6%A8%A1%E6%80%81%E6%BD%9C%E5%8F%98%E9%87%8F%E6%8E%A8%E7%90%86%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6%20V2.md)：唯一目标架构规范。
- [V2 总路线图](docs/Project-Yggdrasil%20V2%20%E4%BB%8E%E6%9E%B6%E6%9E%84%E9%AA%8C%E8%AF%81%E5%88%B0%E5%95%86%E7%94%A8%E8%B7%AF%E7%BA%BF%E5%9B%BE.md)：从 R0 到架构完整版和商用版的唯一高层路线。
- [下一阶段测试计划](docs/next-stage-test-plan.md)：当前执行真源，定义 V2-A/V2-B 的顺序、证据等级、Gate 和旧路线收口。
- [V2-A 推理介质实验记录](docs/v2-a-reasoning-medium-experiment.md)：当前实现、命令、smoke 结果和失败边界。
- [V2-A1.5 潜空间基础实验记录](docs/v2-a1.5-latent-foundation.md)：独立数据合同、P0/P1/P2 结果、反事实干预和停止门禁。
- [V2-A1.6 连续核心失败记录](docs/v2-a1.6-core.md)：relation Gate 失败、只读 closure 诊断和无效旧指标修正。
- [V2-A1.7 连续核心实验记录](docs/v2-a1.7-core.md)：受控 relation holdout、三 seed formal/causal、2×2 消融、长程压力和分层成功概率。
- [V2-A1.8 长程递归实验记录](docs/v2-a1.8-long-horizon.md)：T1–16 随机深度训练、三组独立数据/model seed、T20/T24 Gate、T32 诊断、稳定性与成本。
- [V2-A1.9 冻结 Qwen hidden 边界记录](docs/v2-a1.9-qwen-boundary.md)：三组真实 Qwen role cache、adapter-only formal、hidden 反事实、成本与严格证据边界。
- [V2-A1.10 完整文本匿名工作区记录](docs/v2-a1.10-anonymous-workspace.md)：full-token cache、匿名 K-slot、通用 recurrent reasoner、方法修正、三 seed 正式失败与成本边界。
- [V2-A1.11 正交故障定位记录](docs/v2-a1.11-fault-localization.md)：Boundary strict-overfit 诊断、exact-symbolic Reasoner 三 seed formal、2×2 归因和证据边界。
- [V2-A1.12 Reasoner 根因拆分](docs/v2-a1.12-reasoner-root-cause.md)：binding × cursor 三臂合同、formal `0/3` 和排除结论。
- [V2-A1.13 Transition × Closure](docs/v2-a1.13-transition-closure-root-cause.md)：relation transition、soft closure、state/full 分离和因果结果。
- [V2-A1.13F Fixed-budget 审计](docs/v2-a1.13f-fixed-budget-method-audit.md)：固定 4000-step 排除 early-stop 主因。
- [V2-A1.15 Query-coupled core](docs/v2-a1.15-closed-coupled-core.md)：query-coupled + answer CE 的 fresh-seed 失败。
- [V2-A1.16 冗余答案损失](docs/v2-a1.16-redundant-answer-loss.md)：删除重复 answer CE 后仍为 state/full `0/3`。
- [V2-A1.17 配对目标审计](docs/v2-a1.17-paired-objective-initialization-audit.md)：同 seed、共享初始化位相等的 objective × initialization 根因定位。
- [V2-A1.18/A1.18B 训练脚手架](docs/v2-a1.18-training-scaffold.md)：FINAL-SAUX `2/3`、逐步全局 TSAUX 六 seed formal/causal、部署剥离和机制结论。
- [V2-A1.19H 可泛化混合 Core](docs/v2-a1.19h-hybrid-core.md)：opaque handle、可变实体数量、heldout N5、三 seed formal/causal 与参数级等价训练优化。
- [V2-A1.20B 完整文本 Boundary](docs/v2-a1.20b-full-text-boundary.md)：full-token Qwen cache、无 oracle Boundary、训练吞吐审计、run-1 eligibility 失败、oracle/梯度归因与停线结论。
- [V2-A1.20C 分层编译与信用修复](docs/v2-a1.20c-boundary-repair.md)：token anchor、entity-table-first compiler、hard-forward straight-through bridge、overfit32 失败、梯度冲突与停线结论。
- [V2-A 实验脚本](experiments/v2_a_reasoning_medium.py)：数据生成、文本基线和 latent reasoner 入口。
- [V2-A1.5 实验脚本](experiments/v2_a1_5_latent_foundation.py)：A1.5 数据、P0、P1 cache/train/eval、P2 train/intervention 和 matched text baseline 入口。
- [V2-A1.7 实验脚本](experiments/v2_a1_7_core.py)：A1.6 closure diagnostic，以及 A1.7 data/audit/train/evaluate/intervene/stress/assessment 入口。
- [V2-A1.8 实验脚本](experiments/v2_a1_8_core.py)：A1.8 data/audit/random-depth train/evaluate/intervene/cost/assessment 入口。
- [V2-A1.9 实验脚本](experiments/v2_a1_9_boundary.py)：A1.9 cache/audit/train/evaluate/intervene/cost/assessment 入口。
- [V2-A1.10 实验脚本](experiments/v2_a1_10_anonymous_workspace.py)：full-token cache/audit、匿名 reasoner train/evaluate、formal 后干预、cost/assessment 入口。
- [V2-A1.11 实验脚本](experiments/v2_a1_11_localization.py)：Boundary/Reasoner 两臂 train/evaluate、Gate 后干预和 2×2 assessment 入口。
- [V2-A1.12 实验脚本](experiments/v2_a1_12_root_cause.py)：binding/cursor train/evaluate/intervene/assess 入口。
- [V2-A1.13 实验脚本](experiments/v2_a1_13_transition_closure.py)：transition/closure train/evaluate/intervene/assess 与 fixed-budget audit 入口。
- [V2-A1.15/A1.16/A1.17 实验入口](experiments/v2_a1_17_paired_objective_initialization.py)：query-coupled 两种 objective 的训练入口分别见相邻 A1.15/A1.16 脚本，本入口负责 paired 总判定。
- [V2-A1.18 训练入口](experiments/v2_a1_18_training_scaffold.py)：QAUX、FINAL-SAUX、TSAUX 的 train/evaluate/intervene 与辅助剥离部署入口。
- [V2-A1.18B 总判定](experiments/v2_a1_18b_trajectory_state_scaffold.py)：FINAL-SAUX、TSAUX paired/fresh 和 causal 的机器聚合入口。
- [V2-A1.19H 实验入口](experiments/v2_a1_19h_hybrid_core.py)：H1/H2 data/audit/train/benchmark/evaluate/intervene/assess 统一入口。
- [V2-A1.20B 实验入口](experiments/v2_a1_20b_full_text_boundary.py)：cache/audit/train/evaluate、失败 oracle diagnostic、gradient-credit audit 与 assessment 入口。
- [V2-A1.20C 实验入口](experiments/v2_a1_20c_boundary_repair.py)：compiler supervision、flat/hierarchical × hard/ST 训练、严格 Gate 与实际 checkpoint 梯度冲突诊断入口。
- [架构审阅记录](docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md)：V2 决策形成依据，不与白皮书并行定义规范。
- [目录索引](docs/DIRECTORY_REFERENCE.md)：当前保留项和归档边界。

## 当前状态与未完成项

当前可运行 V2-A0、文本基线、A1.5 分层入口、A1.8 structured core、A1.9 frozen-Qwen oracle-role boundary，以及 A1.10–A1.20C 的故障定位、训练机制、generalized hybrid core 与 full-text Boundary 修复实验。A1.19H-H1/H2 formal 与 causal 均为 `3/3`；opaque handles 只用于相等寻址，continuous payload 与 shared transition 完成更新，训练辅助头在部署前物理删除。H2 在训练 `N=2,3,4` 后对 heldout `N=5` 与 `N=5+relation` 三 seed 全为 `1.0`。A1.20C 目标臂的 anchors、hard-forward equivalence、answer 与 core integrity 通过，但 overfit32 mapping/trajectory/final-state 未严格全对；state-vs-local 梯度 cosine 为 `-0.7366`，presence heads 为 `-0.9944`。因此 2×2 heldout matrix、formal/causal、A1.21P、A1.22A 均未运行。当前路线停在 V2-A，继续前必须重新立项解决 entity-count/pointer-validity 的可微信用和冲突梯度，不能直接进入 Pareto、audit 或 V2-B。

## 文件治理原则

归档代表降级为历史证据，不代表删除。后续实现 V2 时应直接创建新的 V2-A 代码和测试，删除或重写旧契约，不在旧代理代码上添加兼容 wrapper。
