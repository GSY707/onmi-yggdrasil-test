# Project-Yggdrasil V2 研究工作区

本仓库当前只保留 Project-Yggdrasil V2 的架构真源、路线决策、V2-A/A1.5 实现和最近测试计划。Qwen3.5-2B A0 text-CoT 与旧结构 A1 mechanism smoke 已完成；旧 A2 latent probe 没有形成稳定 Pareto。A1.5 已完成独立数据合同、P0 结构化正控制、P1 frozen-hidden surrogate formal 和 P2 learned-slot formal：P0/P2 ordinary test 均可完成逐步状态更新，P1 ordinary validation 也可拟合；composition-heldout 仍未通过，因此 A1.5 不作为 V2 的架构通过结论。

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
- [V2-A 实验脚本](experiments/v2_a_reasoning_medium.py)：数据生成、文本基线和 latent reasoner 入口。
- [V2-A1.5 实验脚本](experiments/v2_a1_5_latent_foundation.py)：A1.5 数据、P0、P1 cache/train/eval、P2 train/intervention 和 matched text baseline 入口。
- [架构审阅记录](docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md)：V2 决策形成依据，不与白皮书并行定义规范。
- [目录索引](docs/DIRECTORY_REFERENCE.md)：当前保留项和归档边界。

## 当前状态与未完成项

当前可运行 V2-A0 数据生成、Qwen3.5-2B/0.8B 文本基线、旧 V2-A1 smoke 和独立 A1.5 数据/P0/P1/P2 入口；baseline 生成没有人为总输出 token cap，只以语义终态/EOS/物理上下文边界停止，并可选显式 wall-time safety timeout。A1.5 P0 32-example overfit final/state full exact `1.0/1.0`，4096-example best ordinary test final/state `1.0/1.0`，composition-heldout `0.2734/0`，length-heldout final/state full exact `1.0/0.2266`；P1 4096-cache ordinary validation final/state `1.0/1.0`；P2 ordinary test `1.0/1.0`、composition `0.2773/0`、length `1.0/0.6484`。0.8B no-cap zero-shot test/composition/length 各128条 formal parse/final/state 分别为 `0.1797/0.0625/0.0234`、`0.4063/0.3828/0.3750`、`0.0625/0/0`；2-shot test/composition/length final/state 为 `0.1328/0.0156`、`0.1953/0`、`0.0938/0`，格式解析提升但状态仍失败。单样本 smoke parse `1.0`、final/state `0/0`；不能用隐藏 token cap 掩盖未终态样本。A1.5 未通过，A3 audit、A4 formal 和全部 V2-B 均未启动。旧 A2/归档代理结果仍不能升级为 V2 或 A1.5 证据。

## 文件治理原则

归档代表降级为历史证据，不代表删除。后续实现 V2 时应直接创建新的 V2-A 代码和测试，删除或重写旧契约，不在旧代理代码上添加兼容 wrapper。
