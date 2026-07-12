# Directory Reference

本仓库是 Project-Yggdrasil V2 的独立研究工作区。当前主线已经从旧 Stage A—AV-J-C 直接切换到 V2-A，再在 V2-A 通过后进入 V2-B。关联项目 `C:\skzy\QuickFileTransport\世界树计划` 的设计哲学是上位概念真源，本仓库不修改其源码。

## 当前真源

| 路径 | 地位与用途 |
| --- | --- |
| `docs/Project-Yggdrasil 多模态潜变量推理架构白皮书 V2.md` | 当前唯一目标架构规范；定义连续 latent recurrence、Boundary-MoE、FFN-MoE、审计和边界。 |
| `docs/Project-Yggdrasil V2 从架构验证到商用路线图.md` | 当前唯一高层路线；按 Gate 推进 R0、R1/R2/R3、架构完整版和商用路线。 |
| `docs/next-stage-test-plan.md` | 当前最近执行真源；冻结 V2-A/V2-B 的实验顺序、证据口径、Gate、消融、成本指标和旧路线收口。 |
| `docs/project-yggdrasil-latent-reasoning-architecture-review-2026-07-11.md` | V2 决策形成记录；只保存推导和审阅依据，不与白皮书并行定义规范。 |
| `docs/DIRECTORY_REFERENCE.md` | 本索引；新代码、测试、文档和归档必须同步这里。 |
| `README.md` | 面向仓库使用者的当前状态说明；不再展示旧阶段结果或旧运行命令。 |

## 当前实现状态

V2-A 尚未实现。当前根目录没有 V2-A/V2-B 的训练脚本、源码、测试或 formal artifact，也不应把归档内容当作当前运行入口。最近工作顺序以 `next-stage-test-plan.md` 第 7 节为准：V2-A0 基座与任务 → schema/规格 → A1 smoke → A2 K/T sweep → A3 audit → A4 formal；A4 通过后才设计 V2-B0。

## 保留的用户资料

| 路径 | 用途 |
| --- | --- |
| `AGENT.md` | 项目协作提示和实验习惯。 |
| `note.txt` | 用户笔记；按要求保留，不删除。 |
| `下一件事.txt` | 用户待办笔记；保留。 |
| `HF token.txt` | 用户本地文件；保留，不在文档中展开其内容。 |
| `.gitignore` | 本地环境、缓存和未来实验产物的忽略规则。 |

## 旧路线归档

`archive/legacy-proxy-route-2026-07-11/` 是一次完整的非删除归档，保留切换前的原始相对目录：

| 子目录或文件 | 内容 | 当前地位 |
| --- | --- | --- |
| `docs/` | 旧阶段报告、V1 白皮书、旧路线和旧命令清单 | 历史思想与代理证据 |
| `experiments/` | Stage A—AV-J-C 及相关代理脚本 | 不再运行 |
| `src/` | 旧代理公共实现 | 不再作为 V2 基座 |
| `tests/` | 绑定旧契约的测试 | 不再作为当前验收 |
| `artifacts/` | 旧 smoke/probe/formal 输出 | 只用于追溯和失败分析 |
| `thinking/` | 旧路线思考稿 | 历史材料 |
| `pyproject.toml` | 旧实验依赖和 pytest 配置 | 随旧代码归档 |
| `README-legacy.md` | 归档前 README | 历史入口 |

归档中的报告可能仍引用原始根目录路径；这些引用用于还原当时的实验语境，不构成当前可运行入口。后续 V2 实现必须新建代码和测试，不在旧代理代码上增加兼容 wrapper。

## 维护规则

新增 V2 文件应先归入明确的 V2-A/V2-B 语义包，再更新本索引。实验结果只能在 evidence level、配置、seed、heldout、消融、成本和失败边界齐全后进入当前证据；旧 Stage 结果不得静默升级为 V2 架构证据。任何再次废弃的当前文件都移动到带日期的 `archive/`，不保留并列旧入口。
