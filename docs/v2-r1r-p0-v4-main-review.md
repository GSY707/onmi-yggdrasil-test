# V2-R1R P0-D v4 R0 主设计层验收

日期：2026-08-01  
验收对象：`artifacts/v2-r1r/p0d-v4-r0-20260801-1/`、当前 v4 source snapshot、预测试与结果同步  
判决：接受 `FAIL_R0` 与停机纪律；拒绝 v4 R0 实现和 G02–G08 正向结果作为有效审计证据

## 1. 核心判断

正式命令在 `F401/matrix_row_missing` 停止是有效的执行失败记录，但这不是唯一问题。主设计层复核确认，v4 实现没有兑现合同最关键的认识论边界：reference expected values、语义回放、表面启发式和统计捷径审计并未形成相互独立的测量链。因此 `audit-core.json` 中的 G02–G08 全真只说明当前实现内部自洽，不能证明审计器能够区分正确世界与错误世界。

本轮不构成 ERE/CPS 任务失败、latent core 失败或训练失败。它只说明 P0-D 测量工具仍不合格。R1 generator、P0-M、模型和训练继续禁止。

## 2. 直接故障与预测试缺口

`tests/v2_r1r_v4/fault_harness.py` 的 `run_case()` 在进入 F401 专用 synthetic-matrix 分支前，无条件调用 `mutate(case_root, group, case)`；`mutate()` 只实现 F402–F420，因而第一条 F401 必然抛出 `ValueError("unknown fault F401/matrix_row_missing")`。

预测试没有覆盖这条路径。v4 只提供三个 pytest：检查 reference spec 版本与数组长度、CLI help 中的命令名、F401–F420 group/case 数量。它们没有运行任一 canonical expected-world、公开 `run_case()`、真实 fault 副本、metamorphic、import boundary、assessment 或 snapshot replay。因此“pytest 3 passed”不是 sealed-run 资格证据。

## 3. reference 独立性失败

冻结合同要求 `reference-spec.json` 保存完整手写 AST、initial world、逐步 trace、answer、claim truth 与 pair diff；builder 只能做一一重命名和显式 block 展开。

实际 `reference-spec.json` 只保存 archetype id、recipe、operation spine、占位 answer/provenance 和若干循环数组。`reference_builder.py` 自己构造 ERE/CPS AST，并实现 `_ere_trace`、`_cps_eval`、`_cps_results`、`_cps_trace`、`_cps_answer` 与 claims 计算。这等价于在 reference 侧建立第二个 generator/simulator。它还通过 `% 8`、`% 9`、`% 3` 等取模同时分配答案、label、cue、template、candidate/action position，违反了有限正交 basis 的明确限制。

所以 expected values 不是独立手写事实，audit 与 reference 可以共同犯错而仍然全真。

## 4. G07/G08 没有实现合同算法

G07 的 source-only parser 实际读取 source 中显式写出的 `Structured features:` 行，再与 record 内 `surface_assignment` 对照；它没有运行合同规定的 ERE/CPS source-only heuristics，也没有计算 heuristic accuracy 相对 chance `+0.10`。若逐记录 mismatch 只可能是 `0/1`，再判断 `max <= 1`，该指标天然通过，不能代表 split 配额偏差。

G08 没有拟合 multinomial Naive Bayes、条件多数或 character n-gram，没有构造 grouped five-fold、train-fit-heldout 模型、vocabulary、confusion matrix 或真实 accuracy。实现只核对 manifest 中声明的算法名/布尔字段，并用正则寻找显式 `answer label` 或 `truth=` 字样。因此 G08 是自我声明检查，不是统计捷径审计。

此外，`audit_root()` 的顶层 `passed_target_gates` 只认真检查以 `_valid` 结尾的部分指标，不能替代 registry 对全部阈值的逐项 conjunction。

## 5. 证据保留与下一轮直接切换

保留 `p0d-v4-r0-20260801-1` 只用于以下失败诊断：

- sealed build 与正向 audit 曾实际运行；
- fault matrix 在第一条 F401 的编排错误处停止；
- 后续 faults、metamorphic、import boundary、assessment、evidence seal 与 replay 均没有运行；
- 执行层没有越过失败继续 R1。

不得修复、覆盖或重跑该 artifact，也不得把 G02–G08 写成已通过的正式 Gate。

下一轮直接切换为 P0-D v5，并把测量系统拆开验证：R0A 只验证手写语义 oracle 与独立 simulator；R0B 才验证 G02–G06；R0C 单独验证 G07/G08 的真实算法；R0D 最后做一次完整 reference/fault integration。主设计层必须直接交付 oracle fixture、expected outputs、validator 和资格测试，执行层不再从长篇文字自行推导这些事实。

## 6. 完成与未完成

已完成：v4 partial artifact、当前实现与预测试的独立复核；直接故障、测试覆盖、reference 独立性和 G07/G08 根因已经定位。

未完成：任何有效 R0 Gate、production generator、候选训练数据、P0-M、模型训练或架构验证。P0-D 仍失败。
