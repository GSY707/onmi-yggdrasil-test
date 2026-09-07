# V2-R1R P1 v5 semantic-transfer 失败复盘

日期：2026-08-11

## 1. 正式判决

P1 v5 的唯一 preflight sealed PASS；`qualification7168` 按合同跑满 14,336 updates、每个 optimization episode 平均暴露 32 次后 sealed `FAIL_P1_V5_QUALIFICATION7168`。ERE/CPS 训练答案均为 `1.0`，但隔离 audit answer 为 `0.474609375/0.1845703125`，audit claim 为 `0.5/0.5`，zero-state 为 `0.5/0.50341796875`，state-dependency drop 为 `0/-0.00341796875`。F8192 与 assessment roots 不存在。

正式失败 root 为 `artifacts/v2-r1r/p1-v5-transfer-qualification7168-20260811-1/`，seal 文件 SHA-256 为 `599E1A3FBE9ED447ECEAA3B6E095DBF60955CDF27CD1E307CB7028FA637C591A`。该 root 与 v5 preflight 均不可修改、覆盖或重跑。

## 2. 失败不再归因于什么

v5 已删除 v4 的跨 episode owner-contrast，因此 v5 证明该错误负例不是唯一问题。14 个评测点中，claim minibatch、见过 episode 和隔离 audit 始终约为 `0.5`；最终 paired loss 仍为随机点。只读梯度审计显示 claim loss 对 Boundary/core/probe 的梯度 L2 分别为 `0.1923/0.0618/0.1393`，claim probe 参数相对初始化移动 `16.64%`。因此不是断图、零梯度、学习率未生效或简单 GPU 故障。

prefix 审计也没有发现 state-index collision：ERE claims 位于 `0/中段`、reasoning budget 为 5–6；CPS claims 位于 `0/1/计划末端`、budget 为 5。失败不能简化为一个 off-by-one。

## 3. 当前根因

owner-free claim-only 在 8+8 episodes 上经过 600 updates 可以达到 seen claim `0.9635`，证明现有 query/state/probe 有局部可学习性；但直到约 100 次 episode 暴露才开始明显下降。v5 每 episode 只有 32 次暴露，每次又只抽 2/约6 个 claim pairs，单一语义目标实际只见约 10.7 次。把每次暴露改成全部 claims 后，64+64 episodes、64 次暴露仍只有 train `0.5104/0.5`、audit `0.4844/0.5`。

这组结果指向规模相关的对称信用问题：正负 claim 在 nonce、标签、值与表面上刻意平衡；在模型尚未形成 predicate—state binding 前，不同 episode 的梯度相互抵消。小集合可以靠高重复记忆任意绑定，大集合则由独立 answer loss先学会 episode→label 映射，claim 分支停在随机点。

现有训练 claims 对动态的覆盖也不足。按“同一 canonical predicate 在不同 prefix 上是否翻转”复算，ERE 只有 `593/8192` episodes、`705` queries 提供 truth flip；CPS 为 `6826/8192` episodes、`13,652` queries。静态同-prefix pair 虽然真值合法，却没有直接规定同一 query 随 recurrent state 改变，因而不是足够强的规模化 symmetry breaker。

## 4. 后继约束

后继不得继续增加 v5 updates、claim 权重或数据规模。新的训练信号必须同时满足：同 episode、真值在两个状态上都有定义、同一 query embedding、只有 state prefix 改变、标签严格翻转。这样可用相邻状态 temporal contrast 取代错误的跨 episode owner contrast。

该机制只能作为训练期 query head，formal 前仍需物理剥离；不得把 AST、oracle span、显式寄存器、答案或 simulator state 送入部署模型。下一阶段先做等计算 static-pair 与 causal-temporal-witness 比较，不直接重启完整 P1。
