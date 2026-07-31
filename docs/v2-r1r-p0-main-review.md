# V2-R1R P0-D v1 主设计层复核

日期：2026-08-01  
地位：失败证据与修订依据；规范真源仍为 `docs/v2-r1-revalidation-task-design.md` 第 16 节

## 核心判决

`artifacts/v2-r1r/p0-v1/p0-assessment.json` 的 `passed=true` 不被主设计层接受。v1 证明了 generator、simulator 和 audit 在同一套内部假设下能够自洽，却没有证明表面任务必须经过预期计算才能回答。两个任务都存在满分或近满分的结构化捷径，CPS 的训练 claim 还包含系统性错误。因此路线状态是 P0-D 失败，而不是“等待进入 P0-M”。

这次失败不归因于 latent 架构，也不归因于训练。它发生在模型出现之前，说明 P0 的任务与 verifier 还没有形成有效测量工具。

## 独立复核证据

主设计层对 formal 14,336 条 JSONL 做了只读复核。ERE 的 surface-only heuristic 读取第一条 `SET` 定义的 literal，再从末尾局部标签说明查表；它在 train、validation、composition/length/entity/language OOD 和 causal pairs 上均为 `1.0`。这说明所谓三至二十步 dependency 并未阻止模型直接读取答案来源。

CPS 的合法最优计划在常规生成器里固定放在 Candidate 1，NONE 记录则把一个顺序错误计划放在 Candidate 1。只需比较 Candidate 1 的首 action 与第一条 action 定义：相同就返回 Candidate 1 的标签，否则返回 NONE 标签，便可在 train、validation、composition/distractor/horizon/language OOD 全部达到 `1.0`。因此字母标签均衡与 unigram NB 通过没有触及真正的关系型位置泄漏。

split 也没有实现冻结语义：CPS train 与 composition OOD 使用同一 `_base_ast` 拓扑；ERE language OOD 改用 relation/swap-if 等不同模式，混入了 composition shift。OOD renderer 的正文语法基本不变，只改 opening 与引导短语。

CPS claim 把 prefix 0 之外的中间状态大多标成“不合法”，而不是从逐步执行状态求真。按 teacher trace 复算后，正 claim 真值准确率在不同 split 只有约 `0.38–0.57`。由于每条文字 claim 都机械配了一个相反 label，旧 audit 仍会报告正负平衡；这揭示了“数量正确”与“监督正确”的区别。

实际 tokenizer 复核使用冻结的 Qwen revision。所有 14,336 条 source 都未超过 1,024 token，最大为 745；但每一条的实际 token 数都大于 v1 regex 计数，所以“regex 是保守上界”这一 Gate 依据为假。certificate 复核还发现保存路径没有和 AST 的真实唯一 diff path 对齐。

## 对路线的影响

旧 artifact 保留为 rejected diagnostic，用来测试 v2 audit 能否抓住已知故障；它不能作为训练数据、cache 输入或 P0-M 许可。无需删除 P0 任务方向，也不需要修改 R1R-Latent 模型：当前失败集中在测量层，修复成本远低于启动一次无意义训练。

v2 不追求把数据做成无法被任何方法求解，而是要求简单 surface/position/单线索策略不能超过随机基线，且 heldout split 真正改变指定结构。详细生成、claim、tokenizer、故障注入和 13 项 conjunction 已冻结在设计文档第 16 节。执行层只能实现和运行该修订；v2 任一 Gate 失败仍停在 P0-D。
