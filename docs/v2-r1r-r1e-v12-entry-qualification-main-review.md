# V2-R1R R1E v12 production entry qualification 主设计层验收

日期：2026-08-09  
formal artifact：`artifacts/v2-r1r/r1e-v12-entry-qualification-20260809-1/`  
机器状态：`PASS_R1E_ENTRY`  
主设计层判定：**有限 `production entry accepted`；授权另立 R1 generator smoke，不授权 P0-M 或训练**

## 1. 核心判定

v12 直接新增 `production/` 层，没有修改 v7/v9/v10/v11 accepted runtime 或 artifact。三个受控自然语言 grammar 在六个 hand-authored semantic/length envelopes 上完成 AST + choice exact roundtrip `18/18`，fresh simulator output identity `18/18`；七种 ERE primitive、两种 predicate、嵌套 `FOREACH_LINKED→IF`、两种 query、五种 CPS condition、六种 effect、empty state/plan 和两步 rule 均被覆盖。model view 只保留四个公开字段，十项禁止结构和十二项语义字段改动全部被拒绝或检测。

长度 census 证明 scaling 不是预防性补丁：最小 ERE source 已为 463 characters，最大 envelope 为 3322 normalized characters、984 个 pinned Qwen token。v12 因此真实执行 scaling branch。新 learner 保留 v10 feature/count/Laplace/mask 与 rational ordering 语义，用 canonical prime-exponent score expression 替代巨大 Fraction decimal report；512–8192 character 的 word/char profiles 全部确定性完成，v11 24 个 adapter 预测与 v10 一致。

因此接受两个 generator entry condition 已关闭：production renderer 在规定生成域内语义充分且 source-only；shortcut learner 不再受 400-character 报告崩溃限制。

## 2. Formal 与独立复算

切换后的 pytest 为 `51 passed`；compileall、CLI help、`git diff --check`、四个 prior seal 和 18 个 shared runtime 文件身份均通过。唯一 formal 输出 G01–G08 `8/8`、15 个 metric 全真、29/29 adversary controls、20/20 metamorphic、双次 report bytes identity 和 accepted input read-only。

fixed root 精确 12 项。root evidence seal 的 SHA-256 为 `86B49779D4E5A7EF496EE6FC0C0C722071983E5B3C6FFB9F8D90D92FBB9E78CA`；独立复算的 57 个非 root-seal 文件与 map 完全相等。与 v11 不同，sealer 只排除当前 root seal；nested evidence seals 被正常直接哈希。

主设计层在 formal 后做了两组 registry 外探针，formal bytes 未修改：

- 对 60 个随机 fit fixture、word/char 两种 analyzer、每个四条随机 eval，共 480 个预测比较 compact learner 与 v10 exact predictor，全部相等；
- 构造未登记的 `IF → FOREACH_LINKED → IF` 深层 ERE primitive，在三个 grammar 下 source-only exact roundtrip `3/3`。

这些探针没有发现 renderer 只记住冻结树形或 scalable comparator 在普通近邻分布上改变决策。

## 3. 仍然存在的边界

第一，`production` 在这里指本实验的受控自然语言生成面，不是任意用户自然语言 parser。它对 episode-local nonce 定义、完整状态、顺序和 choice 语义是充分的，但仍可能存在 template/statistical shortcut；这正是下一阶段 generator smoke 的审计对象，而不是 v12 可以提前证明的结论。

第二，最大 envelope 已到 984/1024 token，只剩 40 token 余量。R1 generator 必须使用 pinned tokenizer 对每条最终 source 计数并拒绝/重生超长记录；不得按字符估算、静默截断或把高难结构删掉来过 Gate。generator 的正式生成域应把常规上限控制得更低，并把接近上限的样本集中在 length/horizon OOD。

第三，compact learner 是数据捷径检测器，不是训练模型。log comparison 的保守误差带在接近时回退整数精确比较；v12 验证了确定性、长输入和大量 overlap 等价，但没有把它提升为通用数值库或模型质量证据。

## 4. 完成与未完成

| 项目 | 状态 |
| --- | --- |
| production ERE/CPS renderer/parser | 已完成；规定生成域 exact roundtrip accepted |
| source-only/model-view 边界 | 已完成并通过 |
| pinned tokenizer length census | 已完成；463–3322 chars，最大 984 tokens |
| scalable word/char learner | 已完成；v10 overlap 与 v11 adapter 等价 |
| v12 formal/seal/registry 外探针 | 已完成并通过 |
| R1 generator smoke | 已授权另立合同，尚未执行 |
| 完整 P0-D、P0-M、模型/cache/GPU/训练 | 未执行、未授权 |

下一步只能设计并执行 R1 generator smoke：冻结小规模 split、构造算法、因果 pair、teacher/claim、真实 tokenizer、shortcut baseline、Gate 和失败即停止。generator smoke 通过后仍不能自动进入 P0-M；完整 P0-D 与 P0-M 必须另立授权。
