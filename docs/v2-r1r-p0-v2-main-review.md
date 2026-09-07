# V2-R1R P0-D v2 主设计层独立验收

日期：2026-08-01  
验收对象：`artifacts/v2-r1r/p0-v2-failed-audit-20260801-1/`、generator/audit v2 当前工作树与第 16 节冻结合同  
证据地位：主设计层只读复核；不是模型或训练证据

## 核心判决

本轮的**执行纪律与失败证据通过验收**：正式数据规模完整，机器 assessment 可复现地为 `false`，执行 agent 在 Gate 失败后停止，没有创建 P0-M、model、cache、Boundary、core 或 baseline，也没有启动 GPU 训练。

但本轮的**P0-D v2 合同实现不通过验收**。机器报告的 position、claim-kind coverage、longest-plan 三类红灯都是真问题；主设计层进一步发现了审计器未接入 conjunction 的强 surface shortcut、split 构造偏差、CPS 任务深度不足和 provenance 缺口。因此该 artifact 只能作为可复现的 failed diagnostic，不能解释成“修复三项红灯后即可进入训练”。P0-M 继续禁止。

## 独立复算结果

主设计层从保存的 JSONL 重新运行完整 `audit_p0`，再次得到 13 项中 10 项为真、总判定 `passed=false`；三项机器失败与保存的 assessment 一致：

- `answer_choice_candidate_action_definition_position=false`；
- `surface_only_and_statistical_heuristics=false`；
- `claim_truth_kind_coverage_balance=false`。

独立完整性检查确认：14,336 条记录全部可读，14 个 JSONL 的 SHA-256 与 manifest 一致，设计文档 hash 一致；固定 Qwen tokenizer 逐条重算后最大长度为 `1011`，无记录超过 `1024`，也无 token count 不一致。目标 pytest 共 16 项，独立运行全部通过。由此可确认，当前问题不是 artifact 损坏、旧 v1 数据串用或执行 agent 伪造机器结果。

## 机器红灯之外的合同失败

### 1. Horizon 存在 0.832 的 source-only 条件捷径

CPS `horizon_ood` 的 action 数量由 `index mod 5` 决定，正确 candidate 的表面位置也由同一个量决定。只从 `source_text` 数 action definition 的数量 `N`，再选择一基编号为 `N-7` 的 Candidate 并查其局部字母标签，就能在 `426/512` 条记录上答对，准确率为 `0.83203125`；不需要重放 action、检查 precondition、goal、budget 或 cost。

这正是第 16.5 节要求的 candidate-role × position 条件多数/互信息 Gate 应捕获的泄漏。当前 audit 只看全局 candidate position 均衡，所以全局计数近似完美仍没有阻止条件泄漏。正式 position/heuristic Gate 因其他指标已为 false，但现有失败报告没有揭示这个更强的根因。

### 2. ERE length OOD 与答案来源完全绑定

`ere/length_ood` 的 512 条记录全部使用 `answer_provenance=initial-copy`，最大来源占比为 `1.0`，超过第 16.4 节规定的 `0.35`。当前 structure/position audit 没有检查 answer provenance，因此错误地把相关 conjunction 留为 true。

### 3. CPS 没有实现冻结合同中的基础计划深度

第 5.2 节要求 train/validation 的目标计划 `P*` 长度为 `2–5`，再由局部变换构造 hard negatives。当前 CPS train 的 3,413 条非 NONE 记录中，正确计划长度只有 `1` 或 `2`：长度 1 为 1,365 条，长度 2 为 2,048 条；validation 同样只有长度 1/2，没有长度 3–5。horizon 则直接跳到 6–10。当前 structure Gate 只检查 candidate 数量和 horizon 最大长度，没有验证基础计划构造合同。

这会把 CPS 退化为少数一步/两步模板上的 legality/cost 选择，而不是原设计要验证的通用约束计划执行。256 个 CPS causal pair 也全部只修改 `cost` 字段；虽然满足“单字段答案翻转”的窄 Gate，但没有覆盖 resource、precondition 或 effect 的因果变化，不能单独支撑规划机制的代表性。

### 4. 多项 formal audit 仍是占位或同值换名

当前 conjunction 不能视为第 16 节的完整实现，主要缺口如下：

- CPS 的 `raw_cost_sum`、`precondition_only`、`goal_only`、`budget_only`、`final_constraint_only` 五项 heuristic 返回 `None`，随后按永远答错的 `0` 计入通过，而不是实际从 source 解析并运行；
- candidate-role × position 的互信息或条件多数准确率没有实现；
- `claim_only_shortcut_passed` 被无条件写成 `true`。本次另行计算的 claim-only unigram/length 指标本身未超标，但 formal conjunction 没有证明它；
- `split_local_grouped_cv` 与 `train_fit_heldout` 都只是复制同一个 split 内二分 Naive Bayes 数值，没有实现“train 拟合、validation/OOD 评估”；
- language Gate 没有按合同对同一 AST 同时渲染 train/OOD 模板并逐对证明正文变化；现有跨 split 分布比较不能替代这个测试；
- ERE 若干命名为不同结构的 surface heuristic 实际复用同一组正则结果，未覆盖 first/last event 静态输出等完整清单。

因此，现有 10 个 true 只能表示当前审计器的窄检查通过，不能等价为第 16.9 节对应十项合同已经被证明。

### 5. 生成 provenance 只达到部分可复现

数据文件、设计 hash、seed、命令、Python 环境和 tokenizer revision 均已保存并可复算；但 manifest 没有记录生成时的 Git commit、dirty status、dirty diff hash或 generator/source 文件 SHA-256。当前实现本身位于 dirty working tree，故未来仅凭 artifact 不能重建生成它的精确代码快照。这与第 13 节要求不符，应在下一次 formal 生成前修正。

## 路线判定

下一步仍属于 P0-D，不是 P0-M。正确顺序是先把审计器补成对冻结合同的可执行反证器，并为上述已知泄漏增加故障注入；然后重做 CPS 计划生成、horizon 调度、ERE length provenance 和 claim 分布，以新的 generator version 全量重生。只有新 artifact 在完整 conjunction 下通过并再次经主设计层做未知捷径复核，才讨论 P0-M。

本轮已完成的是：v2 实现、smoke、正式生成、正式失败审计、独立完整性复算和停止边界。未完成的是：有效 P0-D、可信的完整 shortcut audit、P0-M，以及之后全部模型、训练、因果、Pareto 和多 seed 阶段。
