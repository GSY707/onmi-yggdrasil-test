# Project-Yggdrasil V2 研究工作区

本仓库当前只保留 Project-Yggdrasil V2 的架构真源、路线决策、V2-A 分层实验和最近测试计划。A1.8 以 T1–16 均衡训练使结构化 core 通过 T24，A1.9 证明 oracle-role-segmented Qwen hidden 可稳定驱动该 frozen core。A1.10 的 full-token + anonymous `K=8` + generic recurrence 联合配置 formal `0/3`；A1.11 随后完成正交定位：learned full-text Boundary 存在连续地址接口缺陷，exact-symbolic typed roles 输入匿名通用 reasoner 仍为 formal `0/3`。因此失败不是两个健康组件的纯组合效应，主要独立风险已定位到匿名 state binding／generic transition／当前 readout objective 这一整臂。

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
- [V2-A 实验脚本](experiments/v2_a_reasoning_medium.py)：数据生成、文本基线和 latent reasoner 入口。
- [V2-A1.5 实验脚本](experiments/v2_a1_5_latent_foundation.py)：A1.5 数据、P0、P1 cache/train/eval、P2 train/intervention 和 matched text baseline 入口。
- [V2-A1.7 实验脚本](experiments/v2_a1_7_core.py)：A1.6 closure diagnostic，以及 A1.7 data/audit/train/evaluate/intervene/stress/assessment 入口。
- [V2-A1.8 实验脚本](experiments/v2_a1_8_core.py)：A1.8 data/audit/random-depth train/evaluate/intervene/cost/assessment 入口。
- [V2-A1.9 实验脚本](experiments/v2_a1_9_boundary.py)：A1.9 cache/audit/train/evaluate/intervene/cost/assessment 入口。
- [V2-A1.10 实验脚本](experiments/v2_a1_10_anonymous_workspace.py)：full-token cache/audit、匿名 reasoner train/evaluate、formal 后干预、cost/assessment 入口。
- [V2-A1.11 实验脚本](experiments/v2_a1_11_localization.py)：Boundary/Reasoner 两臂 train/evaluate、Gate 后干预和 2×2 assessment 入口。
- [架构审阅记录](docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md)：V2 决策形成依据，不与白皮书并行定义规范。
- [目录索引](docs/DIRECTORY_REFERENCE.md)：当前保留项和归档边界。

## 当前状态与未完成项

当前可运行 V2-A0、文本基线、A1.5 分层入口、A1.8 structured core、A1.9 frozen-Qwen oracle-role boundary、A1.10 联合目标配置和 A1.11 正交定位。A1.11 Boundary 的 trajectory/answer/mapping overfit 为 `1.0`，但 source/target pointer 最低为 `0.9643`；只读 hard re-embedding 全通过，formal 按 Gate 停止。A1.11 exact-symbolic Reasoner overfit32 全通过，但三组 formal 为 `0/3`，训练子集 trajectory 也接近 `0`；所有 ordinary formal 失败的干预均未运行。当前对“V2-A 形成 matched text-CoT 质量—成本 Pareto”的工程判断调整为 `22%–35%`、中心约 `28%`；这不是统计置信区间。下一阶段 A1.12 只恢复最小 entity-addressable state workspace，继续隔离 anonymous binding 与 generic transition。A3/A4 和全部 V2-B 仍未启动。

## 文件治理原则

归档代表降级为历史证据，不代表删除。后续实现 V2 时应直接创建新的 V2-A 代码和测试，删除或重写旧契约，不在旧代理代码上添加兼容 wrapper。
