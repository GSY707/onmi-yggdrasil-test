# Project-Yggdrasil V2 研究工作区

本仓库当前只保留 Project-Yggdrasil V2 的架构真源、路线决策、V2-A 分层实验、最近测试计划和远期 V2-C 系统实验合同。A1.19H 已为 mixed latent core 形成 formal/causal `3/3`，A1.20D 又修复 full-text compiler 的双向 section 一致性并形成强诊断正证据；但 A1.21P 因任务同构、baseline 不公平和 formal 证据缺失而停止。当前执行路线仍是 V2-R1R：冻结 A1.20D 为正控制，用同一 Boundary/core 在情境规则执行和约束计划选择两个不同任务族上重新验证跨任务质量—成本优势。R1R 已冻结实现级模型、数据生成算法、训练监督、公平基线和分阶段 Gate；P0-D generator v1 虽曾由旧审计器自判 10/10，通过主设计层复核后因结构化捷径、错误 claim 与 split/audit 缺口被否决。当前只允许按 v2 修订重做 P0-D；V2-A 仍未通过，V2-C 只有设计地位。

当前主线已经从旧 Stage A—AV-J-C 直接切换为：

1. V2-A：先在成熟文本基座上比较显式文本 CoT、单向量 latent recurrence 和多向量 latent recurrence。
2. V2-B：只有 V2-A 通过后，才验证固定输入/输出下的文本/视觉/动作表示、Boundary-MoE、FFN-MoE 与 Attention Pump。
3. V2-C：只有 V2-B 通过后，才把主动 Boundary、真实工具、工作树/记忆、人类目标连续性和隔离专家演化聚合为多层、多线程、树图混合全双工智能体实验。

旧实验不是当前实现路线。原有阶段报告、代理脚本、测试、源码、artifact、旧 README 和思考稿已完整归档到 [`archive/legacy-proxy-route-2026-07-11/`](archive/legacy-proxy-route-2026-07-11/)，保留作历史证据，不建立兼容入口。

## 当前入口

- [V2 架构白皮书](docs/Project-Yggdrasil%20%E5%A4%9A%E6%A8%A1%E6%80%81%E6%BD%9C%E5%8F%98%E9%87%8F%E6%8E%A8%E7%90%86%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6%20V2.md)：唯一目标架构规范。
- [V2 总路线图](docs/Project-Yggdrasil%20V2%20%E4%BB%8E%E6%9E%B6%E6%9E%84%E9%AA%8C%E8%AF%81%E5%88%B0%E5%95%86%E7%94%A8%E8%B7%AF%E7%BA%BF%E5%9B%BE.md)：从 R0 到架构完整版和商用版的唯一高层路线。
- [下一阶段测试计划](docs/next-stage-test-plan.md)：当前执行真源，定义 V2-A/V2-B/V2-C 的顺序、证据等级、Gate 和旧路线收口。
- [V2-C 全双工智能体实验合同](docs/v2-c-hierarchical-full-duplex-agent-experiment.md)：聚合多层、多线程、树图混合 Boundary-MoE、工具、目标连续性和专家演化；当前仅设计完成。
- [`thinking` 草稿综合分析](docs/thinking-draft-synthesis-2026-08-01.md)：保留原稿不动，区分五篇近期草稿与项目初期 V1 参照，并说明它们如何统一进入 V2-C。
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
- [V2-A1.20D full-text 机制修复](docs/v2-a1.20d-full-text-mechanism-repair.md)：双向 section 因果闭环、N5 修复、完整矩阵与 hidden causal 诊断。
- [V2-A1.21P matched Pareto](docs/v2-a1.21p-pareto.md)：K=1 负基线、在线 smoke、正式合同缺口与 A1.22A 停机判定。
- [V2-R1R 冻结设计与执行合同](docs/v2-r1-revalidation-task-design.md)：ERE/CPS 构造算法、统一样本 view、R1R-Latent v1、claim verifier 训练监督、公平 SFT、因果审计、Pareto Gate 与执行 CLI。
- [V2-R1R P0-D 结果](docs/v2-r1r-p0-result.md)：generator v1 与旧审计器的历史结果；机器 10/10 已被主设计层否决，P0-M 未授权。
- [V2-R1R P0-D v1 主设计层复核](docs/v2-r1r-p0-main-review.md)：记录 ERE/CPS 满分捷径、claim/split/token/certificate 缺口及 v2 修订依据。
- [V2-A 实验脚本](experiments/v2_a_reasoning_medium.py)：数据生成、文本基线和 latent reasoner 入口。
- [V2-R1R P0-D CLI](experiments/v2_r1_revalidation.py)：`generate-p0` / `audit-p0` 数据生成与机器审计入口；正式输出默认拒绝覆盖。
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
- [V2-A1.21P 实验入口](experiments/v2_a1_21p_pareto.py)：跨域 probe、K=1、在线 matched preflight 与机器 assessment 入口。
- [架构审阅记录](docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md)：V2 决策形成依据，不与白皮书并行定义规范。
- [目录索引](docs/DIRECTORY_REFERENCE.md)：当前保留项和归档边界。

## 当前状态与未完成项

当前可运行的旧实验仍保留为历史和正控制。新的主线 V2-R1R 已完成主设计层冻结；generator v1 的独立 package、formal 数据和旧机器审计均已生成，但旧 10/10 是无效自判，artifact 现仅为 rejected diagnostic。冻结 Qwen hidden Boundary、共享 K=1/K=8 core、direct/text-CoT 基线、P0-M/P1、在线成本和三 seed formal 均尚未实现或运行；当前停止在 P0-D v2 修复，不允许进入 P0-M。A1.22A 未运行，V2-A/R1 未关闭，V2-B 不启动。V2-C 的文档合同已经完成，但运行时、工具接入、任务数据、代码、训练和 formal artifact 均不存在，且必须继续等待 V2-A、V2-B。

## 文件治理原则

归档代表降级为历史证据，不代表删除。后续实现 V2 时应直接创建新的 V2-A 代码和测试，删除或重写旧契约，不在旧代理代码上添加兼容 wrapper。
