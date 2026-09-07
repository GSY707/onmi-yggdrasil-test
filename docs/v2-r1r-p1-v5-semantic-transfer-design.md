# V2-R1R P1 v5 语义迁移资格合同

日期：2026-08-11

证据目标：single-seed P1 K=8 学习路径资格与原 K01–K09 复验；不是多 seed 稳定性、K=1/文本基线或 P2 Pareto 证据。

## 1. 判决问题

P1 v3 覆盖完整数据但只有约 2.34 次 episode 暴露，未进入学习区；v4 在 512/族、75 次暴露下完成训练记忆，却在未见 episode 上回到随机。v5 测试缺失象限：在高语义多样性、足够暴露、且不奖励 episode 身份的目标下，匿名 K-slot recurrent reasoner 能否学习可迁移的状态算法并通过原 P1 K=8 Gate。

本轮不改 K=8 部署结构：2048→512 tokenwise Boundary、8 个匿名 slot、2 层共享 recurrent block、最多 24 步、latent-only answer readout。禁止 task embedding、任务专用参数、teacher/AST 输入、oracle span/role、显式寄存器和输入到答案旁路。训练 probe 仍在 checkpoint 部署前物理移除。

## 2. 目标修复

活动训练目标只保留两类真值有定义的监督：同 episode 的最终答案交叉熵，以及同 episode、同 prefix、只改变一个 predicate value 的 positive/negative paired claim CE + ranking。v4 的任意跨 episode owner-contrast 与 owner-shuffle 成功 Gate被直接删除。

这不是一次局部调参，而是长期训练不变量：跨样本负例只有在目标真值对该组合有定义时才能进入损失；任何辅助机制的成功必须在未参与梯度的 episode 上验证。zero-state 只作为诊断与 Gate，用来确认 claim 的正确率来自状态，而不是 claim 文本自身。

## 3. 固定数据划分与阶段

继续只读复用 P1 v2 的数据、model-eval subset 与 hidden cache。cache formal 状态仍为 FAIL，训练授权只来自 sealed P1 v3 recovery。v4 三个已创建 roots 仅作为失败来源，不得修改、重封或继续训练。

训练 train split 先按新 `SELECTION_SEED` 做标签、pattern、reasoning-budget 平衡的确定性排列。每族前 7,168 条是 qualification optimization，后 1,024 条是永久隔离 audit；两者不相交且并集严格等于 8,192 条。full 阶段把 audit 并回，但只能读取 sealed PASS qualification checkpoint。

| 阶段 | 梯度数据/族 | audit/族 | batch | 最大 updates | 最早判定 | 最大新增暴露 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q7168 | 7,168 | 1,024 未见 | 32 | 14,336 | 7,168 | 32× |
| F8192 | 8,192 | 原 audit 已并入 | 32 | 12,288 | 4,096 | 24× |

两阶段分别重置 AdamW 和 cosine-with-floor 日程。Q7168 前 2,048 updates 使用 answer/claim 权重 `0.25/1.0`，之后为 `1.0/0.5`；F8192 为 `1.0/0.25`。Boundary/core/readout 的基础学习率保持 `1e-4/2e-4/3e-4`，weight decay `0.01`，gradient clip `1.0`。每阶段固定 schedule，达到全部 Gate 可提前停止，否则跑满后失败封存。

## 4. 迁移 Gate

Q7168 每 1,024 updates 评测：optimization answer 每族固定 512 条；audit answer 每族全部 1,024 条；audit claim 每族固定 256 条、两个确定性 claim rotation，共 2,048 claims/族；P1 validation 每族 1,024 条只报告、不用于选择。

Q7168 全部条件为：update ≥7,168；optimization answer 每族 ≥0.75；audit answer 每族 ≥0.50；audit claim accuracy 每族 ≥0.65；audit claim accuracy 减 zero-state accuracy 每族 ≥0.10；所有指标有限。该 Gate 直接拒绝 v4 型身份记忆。

F8192 每 1,024 updates 评测 optimization answer 与 P1 validation。preliminary 条件为：update ≥4,096；train answer 每族 ≥0.80；validation 每族 ≥0.85；所有指标有限。未通过不得运行 OOD/causal formal evaluation。

preliminary 通过后，物理剥离 probe 并运行原 P1 K01–K09：validation 每族 `0.85`；八个 OOD cell 各 `0.75`；causal pair flip 每族 `0.80`；zero/shuffled middle 平均 drop `0.40`；T1 hard-cell drop 每族 `0.15`；probe stripping delta `≤0.01`；结构、finite 与吞吐 Gate 保持原定义。

## 5. 停止与证据

顺序固定为 `preflight → Q7168 → F8192 → assessment`。任一阶段 non-PASS 后立即停止，后序 roots 不得创建。每个 root 只能创建一次；失败、异常、checkpoint、selection/audit manifest、planned/actual ledger、完整学习曲线、吞吐、源码快照与 seal 都必须保留。

assessment 只有在 Q7168 transfer Gate、F8192 preliminary、原 K01–K09、continuation hash、probe stripping、source/cache 不变量全部通过时才可 `PASS_P1_V5_TRANSFER`。成功只授权另立 fresh-seed 完整 P1 合同并补 K=1、direct 与 text-CoT matched controls；不得直接宣称 P1/P2 完成。

