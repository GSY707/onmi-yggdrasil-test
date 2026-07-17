# Project-Yggdrasil V2 研究工作区

本仓库当前只保留 Project-Yggdrasil V2 的架构真源、路线决策、V2-A 分层实验和最近测试计划。A1.8 以 T1–16 均衡训练使结构化 core 通过 T24，A1.9 证明 oracle-role-segmented Qwen hidden 可稳定驱动该 frozen core。A1.10–A1.17 把 exact-symbolic Reasoner 的失败定位到 relation-addressed transition、soft closure 与训练目标悖论；A1.18/A1.18B 已用训练期 TSAUX 解决机制层：每个递归步从全局 workspace 预测完整 state，正式答案始终 query-coupled，部署前物理删除辅助头。三个 paired seed 与三个 fresh seed 的 formal/causal 均为 `3/3`。该结论只验证 exact-symbolic 三寄存器 core，开放任务中的可扩展 state target 来源仍未解决。

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
- [架构审阅记录](docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md)：V2 决策形成依据，不与白皮书并行定义规范。
- [目录索引](docs/DIRECTORY_REFERENCE.md)：当前保留项和归档边界。

## 当前状态与未完成项

当前可运行 V2-A0、文本基线、A1.5 分层入口、A1.8 structured core、A1.9 frozen-Qwen oracle-role boundary，以及 A1.10–A1.18B 故障定位与机制解决链。FINAL-SAUX 在 paired seed 上 formal/causal `2/3`；把全局完整状态监督扩展到每个递归步后，TSAUX paired 与 fresh 分别 formal/causal `3/3`。机器结论为 `per_step_global_state_credit_assignment_confirmed`、`mechanism_solved=true`、`diagnostic_core_architecture_validated=true`。六个通过模型的正式答案只来自 queried state，部署artifact不含辅助头。下一阶段应验证没有逐步 oracle state 时的 target-source 退火；learned full-text boundary、匿名 workspace、matched text-CoT Pareto、A3/A4 和全部 V2-B 仍未完成。

## 文件治理原则

归档代表降级为历史证据，不代表删除。后续实现 V2 时应直接创建新的 V2-A 代码和测试，删除或重写旧契约，不在旧代理代码上添加兼容 wrapper。
